"""
Stage 6.3 — Score the Chinese panel with the six US full-sample models.

Six scores: xgb, rf, enet x distress, rating. COMPLETE ROWS ONLY — a row with
any missing feature gets no score. RAW probability, no calibrator: stage4_1 and
stage4_2 calibrate out of fold against the US base rate (D-47) and that mapping
does not transfer.

WHY THE US BASE RATE IS NOT THE BENCHMARK. The models are fitted with class
weighting (scale_pos_weight 80.9 on distress), so their raw probabilities sit
far above the base rate ON US DATA TOO: distress xgb raw mean 0.1427 against a
1.22% base rate, 11.7x, with raw Brier 0.0624 against 0.0111 calibrated. A
Chinese mean above 1.22% would therefore mostly reflect the weighting, not
Chinese fragility. The benchmark is the US CROSS-FITTED RAW score instead.

Run:  python3 code/stage6_3_score_china.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
sys.path.insert(0, str(Path(__file__).resolve().parent))

PANEL_IN = OUT_DIR / "stage6_2_china_ranked.parquet"
PANEL_OUT = OUT_DIR / "stage6_3_china_scores.parquet"
KEY = ["gvkey", "datadate", "fyear", "ind_cn"]
MODELS = ["xgb", "rf", "enet"]

# The six models. TAG is "base" for BOTH outcomes: for distress that is the
# 27-feature set (NOT s31, which is the superseded 31), and for rating it is the
# only set there is. See the 2026-09-19 naming caveat — "base" denoted 31 before
# D-62 and denotes 27 now, so the metadata disambiguates it and the name does not.
def model_stem(outcome, model):
    o = "" if outcome == "distress" else f"_{outcome}"
    return OUT_DIR / f"stage5_0_model{o}_L3_K12_base_{model}"


def model_file(outcome, model):
    stem = model_stem(outcome, model)
    return stem.with_suffix(".json" if model == "xgb" else ".joblib")


def meta_file(outcome, model):
    return Path(str(model_stem(outcome, model)) + "_meta.json")


# The US benchmark: CROSS-FITTED raw scores. stage4_1 names its score files with
# the lag and K; stage4_2 does not.
def us_scores_file(outcome, model):
    if outcome == "distress":
        return OUT_DIR / f"stage4_1_scores_L3_K12_{model}.parquet"
    return OUT_DIR / f"stage4_2_scores_{model}.parquet"


def load(outcome, model):
    """Load exactly as stage5_0 saved it (stage5_0:385-388)."""
    p = model_file(outcome, model)
    if model == "xgb":
        est = XGBClassifier()
        est.load_model(str(p))
        return est
    return joblib.load(p)


def check_versions(meta):
    """The metadata warns that joblib WILL NOT load under a different
    scikit-learn version. Check before loading, not in a stack trace."""
    import sklearn
    import xgboost
    want = meta["library_versions"]
    have = {"numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "joblib": joblib.__version__}
    bad = {k: (want[k], v) for k, v in have.items() if want.get(k) != v}
    if bad:
        raise SystemExit(
            "STOP: library versions differ from the fit environment, and the "
            "joblib models may not load or may load WRONG:\n  "
            + "\n  ".join(f"{k}: fitted with {a}, installed {b}"
                          for k, (a, b) in bad.items()))
    print(f"  library versions match the fit environment: "
          + ", ".join(f"{k} {v}" for k, v in have.items()))


def main():
    if PANEL_OUT.exists():
        raise SystemExit(f"STOP: {PANEL_OUT.name} exists; this script never "
                         "overwrites. Move or rename it first.")
    if not PANEL_IN.exists():
        raise SystemExit(f"STOP: {PANEL_IN.name} not found — run stage6_2 first.")

    print("=" * 92)
    print("STAGE 6.3 — SCORE THE CHINESE PANEL   6 scores, complete rows only, "
          "RAW probability")
    print("=" * 92)

    df = pd.read_parquet(PANEL_IN).reset_index(drop=True)

    # ---- the six models' metadata, and the feature order they agree on -----
    metas = {}
    for outcome in ("distress", "rating"):
        for m in MODELS:
            mp = meta_file(outcome, m)
            if not mp.exists():
                raise SystemExit(f"STOP: {mp.name} not found. Fit it first "
                                 f"(stage5_0, --outcome {outcome}).")
            metas[(outcome, m)] = json.load(open(mp))
    check_versions(next(iter(metas.values())))

    # All six must agree on the features AND their ORDER, or the same Chinese
    # matrix cannot be fed to all six. Checked as SEQUENCES, not as sets.
    orders = {k: v["features_in_order"] for k, v in metas.items()}
    feats = orders[("distress", "xgb")]
    odd = [k for k, v in orders.items() if v != feats]
    if odd:
        raise SystemExit(f"STOP: these models disagree on the feature order: "
                         f"{odd}. A permuted vector scores silently wrong.")
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} features absent from "
                         f"{PANEL_IN.name}: {missing}")
    print(f"  all six models agree on {len(feats)} features, in the same order")

    # ---- complete rows only -------------------------------------------------
    complete = df[feats].notna().all(axis=1)
    n_c = int(complete.sum())
    print(f"  {PANEL_IN.name}: {len(df):,} rows, {df.gvkey.nunique():,} issuers")
    print(f"  scoring {n_c:,} COMPLETE rows ({n_c / len(df) * 100:.2f}%); "
          f"{len(df) - n_c:,} rows keep NA on all six scores")
    print(f"      NOT scored, and deferred by decision: a row with a missing "
          f"feature needs an XGBoost\n      trained WITH missing values, so its "
          f"handling of them is learned rather than a library default.")

    # Select IN THE MODEL'S OWN ORDER, never the panel's.
    X = df.loc[complete, feats].to_numpy(dtype="float64")

    out = df[KEY].copy()
    for outcome in ("distress", "rating"):
        for m in MODELS:
            est = load(outcome, m)
            p = est.predict_proba(X)[:, 1]
            col = f"score_{outcome}_{m}"
            out[col] = np.nan
            out.loc[complete, col] = p
            if not ((p >= 0).all() and (p <= 1).all()):
                raise SystemExit(f"STOP: {col} outside [0, 1]")
            print(f"      {col:<24} scored {len(p):,}")

    score_cols = [f"score_{o}_{m}" for o in ("distress", "rating") for m in MODELS]

    # ---- guards -------------------------------------------------------------
    if int(out[score_cols].notna().all(axis=1).sum()) != n_c:
        raise SystemExit("STOP: scored-row count does not equal the "
                         "complete-vector count.")
    if out.loc[~complete, score_cols].notna().values.any():
        raise SystemExit("STOP: an incomplete row carries a score.")
    print(f"  guards passed: {n_c:,} rows scored on all six, "
          f"{len(df) - n_c:,} rows NA on all six")

    # ---- the comparison that matters ---------------------------------------
    print("\n" + "=" * 92)
    print("CHINA vs the US CROSS-FITTED RAW SCORES")
    print("=" * 92)
    print("  NOT against the US base rate. The models carry class weighting "
          "(scale_pos_weight 80.9 on")
    print("  distress), so their raw probabilities sit far above the base rate "
          "on US data too — US")
    print("  distress xgb raw mean is 0.1427 against a 1.22% base rate. The "
          "base rate would therefore")
    print("  manufacture a finding. CAVEAT: the US column is the FOLD models' "
          "out-of-fold output, while")
    print("  China is scored by the FULL-SAMPLE models — same specification and "
          "rows, different fitted")
    print("  object. The full-sample models cannot produce a US benchmark; they "
          "saw every US row.")
    print(f"\n  {'outcome / model':<22}{'CN mean':>10}{'US mean':>10}"
          f"{'CN med':>10}{'US med':>10}{'CN p90':>10}{'US p90':>10}")
    for outcome in ("distress", "rating"):
        for m in MODELS:
            cn = out[f"score_{outcome}_{m}"].dropna()
            up = us_scores_file(outcome, m)
            if not up.exists():
                print(f"  {outcome + '/' + m:<22}{cn.mean():>10.4f}{'n/a':>10}"
                      f"{cn.median():>10.4f}{'n/a':>10}"
                      f"{cn.quantile(.9):>10.4f}{'n/a':>10}")
                continue
            us = pd.read_parquet(up, columns=["score_raw"])["score_raw"]
            print(f"  {outcome + '/' + m:<22}{cn.mean():>10.4f}{us.mean():>10.4f}"
                  f"{cn.median():>10.4f}{us.median():>10.4f}"
                  f"{cn.quantile(.9):>10.4f}{us.quantile(.9):>10.4f}")

    print(f"\n  CHINESE SCORE DISTRIBUTION")
    print(f"  {'column':<24}{'n':>9}{'median':>10}{'p90':>10}{'p99':>10}{'max':>10}")
    for c in score_cols:
        s = out[c].dropna()
        print(f"  {c:<24}{len(s):>9,}{s.median():>10.4f}{s.quantile(.9):>10.4f}"
              f"{s.quantile(.99):>10.4f}{s.max():>10.4f}")

    # ---- write ---------------------------------------------------------------
    cfg = {"input": PANEL_IN.name, "n_features": len(feats),
           "features_in_order": feats, "models": MODELS,
           "outcomes": ["distress", "rating"],
           "score_column": "score_<outcome>_<model> = RAW predict_proba[:, 1] "
                           "from the full-sample US model. NOT calibrated: "
                           "D-47's out-of-fold calibration is fitted to the US "
                           "base rate and does not transfer.",
           "scored_rows": n_c, "unscored_rows": int(len(df) - n_c),
           "unscored_reason": (
               "A row with any missing feature is NOT scored. Scoring it "
               "properly needs an XGBoost trained on data that INCLUDES missing "
               "values, so its handling of them is learned at each split rather "
               "than being the library's fallback direction; the US models were "
               "trained on complete rows only. DEFERRED to a later retraining, "
               "not part of this segment."),
           "benchmark_note": (
               "Compare these scores against the US CROSS-FITTED RAW scores "
               "(score_raw in stage4_1_scores_L3_K12_*/stage4_2_scores_*), NOT "
               "against the US base rate: class weighting lifts raw "
               "probabilities far above the base rate on US data too."),
           "percentile": False}
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}),
         b"stage6_3_config": json.dumps(cfg, ensure_ascii=False).encode()})
    pq.write_table(table, PANEL_OUT)
    print(f"\n  wrote {PANEL_OUT.name}: {len(out):,} rows x {out.shape[1]} cols "
          f"({PANEL_OUT.stat().st_size / 1e6:.1f} MB)")
    print("\nNO PERCENTILE YET. That is stage6_4.")


if __name__ == "__main__":
    main()
