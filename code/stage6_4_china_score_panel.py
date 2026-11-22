"""
Stage 6.4 — The Chinese SCORE PANEL: one row per issuer-year, six scores and
their within-China percentiles.

PERCENTILES ARE POOLED ACROSS THE PANEL, on a 0-100 scale, exactly as the US
stage5_1 does it (`score.rank(pct=True) * 100`, no groupby).

NOTE, because it looks like an inconsistency and is not: FEATURES are ranked
WITHIN FISCAL YEAR (stage6_2, D-44); SCORE PERCENTILES are POOLED (here). Both
mirror the US.

stage5_1's assembly is NOT reused as a function: it asserts that no score or
percentile is NA, which holds for the US panel and is false here by design —
13.5% of Chinese rows are deliberately unscored.

Run:  python3 code/stage6_4_china_score_panel.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
sys.path.insert(0, str(Path(__file__).resolve().parent))

PANEL_IN = OUT_DIR / "stage6_3_china_scores.parquet"
KEY = ["gvkey", "datadate", "fyear", "ind_cn"]
MODELS = ["xgb", "rf", "enet"]
OUTCOMES = ["distress", "rating"]

PCT_NOTE = ("pct_<outcome>_<model> = the score's percentile WITHIN THIS PANEL, "
            "on a 0-100 SCALE (not 0-1): pandas rank(pct=True) * 100, pooled "
            "across the panel and NOT within fiscal year, matching the US "
            "stage5_1. ANY CROSS-MARKET COMPARISON MUST USE THE SAME 0-100 "
            "UNIT — a 0-1/0-100 mismatch gives a plausible-looking wrong "
            "answer rather than an error.")


def panel_path(ext):
    return OUT_DIR / f"stage6_4_china_score_panel.{ext}"


def agreement(a, b, label):
    """Spearman and top-decile overlap on the rows both scores cover.

    D-51: score rank-correlation and top-decile overlap come FIRST, before any
    aggregate. A low overlap is a finding about the two objects compared, not a
    defect.
    """
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 10:
        return None
    x, y = both.iloc[:, 0], both.iloc[:, 1]
    n = max(1, int(round(len(both) * 0.10)))
    top_x, top_y = set(x.nlargest(n).index), set(y.nlargest(n).index)
    return label, x.corr(y, method="spearman"), len(top_x & top_y) / n * 100, len(both)


def main():
    for ext in ("csv", "parquet"):
        if panel_path(ext).exists():
            raise SystemExit(f"STOP: {panel_path(ext).name} exists; this script "
                             "never overwrites. Move or rename it first.")
    if not PANEL_IN.exists():
        raise SystemExit(f"STOP: {PANEL_IN.name} not found — run stage6_3 first.")

    print("=" * 92)
    print("STAGE 6.4 — CHINESE SCORE PANEL   pooled 0-100 percentiles")
    print("=" * 92)
    for line in PCT_NOTE.split(". "):
        print(f"  !! {line.strip()}")

    df = pd.read_parquet(PANEL_IN).reset_index(drop=True)
    score_cols = [f"score_{o}_{m}" for o in OUTCOMES for m in MODELS]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"STOP: {PANEL_IN.name} lacks {missing}")

    # Pooled percentile, one per score column, over the rows THAT score covers.
    # rank(pct=True) ignores NaN, so this falls out of the operation rather than
    # needing a mask. Computed per column so the panel does not silently depend
    # on all six covering the same rows, even though today they do.
    for c in score_cols:
        df[f"pct_{c[6:]}"] = df[c].rank(pct=True) * 100
    pct_cols = [f"pct_{c[6:]}" for c in score_cols]

    # ---- guards -------------------------------------------------------------
    if df.duplicated(subset=["gvkey", "fyear"]).any():
        raise SystemExit("STOP: (gvkey, fyear) is not unique; the panel "
                         "misdescribes itself.")
    for s, p in zip(score_cols, pct_cols):
        if not df[s].isna().equals(df[p].isna()):
            raise SystemExit(f"STOP: {p} is missing on different rows from {s}.")
    lo, hi = df[pct_cols].min().min(), df[pct_cols].max().max()
    if lo <= 0 or hi > 100:
        raise SystemExit(f"STOP: percentiles outside (0, 100]: {lo} to {hi}")
    n_sc = int(df[score_cols[0]].notna().sum())
    print(f"\n  {len(df):,} rows, {df.gvkey.nunique():,} issuers, fyear "
          f"{int(df.fyear.min())}-{int(df.fyear.max())}")
    print(f"  {n_sc:,} scored ({n_sc / len(df) * 100:.2f}%), "
          f"{len(df) - n_sc:,} unscored and left NA on score and percentile "
          f"alike")
    print(f"  guards passed: (gvkey, fyear) unique; every percentile non-NA "
          f"exactly where its score is; range [{lo:.2f}, {hi:.2f}] on 0-100")

    # ---- D-51: who is weak, and do the models and outcomes agree? ----------
    print("\n" + "=" * 92)
    print("WHO IS WEAK IN CHINA — rank correlation and top-decile overlap "
          "(D-51 reads these first)")
    print("=" * 92)
    print(f"\n  DISTRESS vs RATING, per model — do the two outcomes flag the "
          f"same Chinese firms?")
    print(f"      {'model':<10}{'Spearman':>11}{'top-decile overlap':>21}{'n':>10}")
    for m in MODELS:
        r = agreement(df[f"score_distress_{m}"], df[f"score_rating_{m}"], m)
        if r:
            _, rho, ov, n = r
            print(f"      {m:<10}{rho:>11.4f}{ov:>20.1f}%{n:>10,}")
    print(f"      US equivalent, for comparison: Spearman 0.64-0.78, "
          f"overlap 61-73%.")

    for o in OUTCOMES:
        print(f"\n  ACROSS MODELS within {o} — do the three models agree?")
        print(f"      {'pair':<14}{'Spearman':>11}{'top-decile overlap':>21}")
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                r = agreement(df[f"score_{o}_{a}"], df[f"score_{o}_{b}"],
                              f"{a}/{b}")
                if r:
                    lab, rho, ov, _ = r
                    print(f"      {lab:<14}{rho:>11.4f}{ov:>20.1f}%")

    # ---- write ---------------------------------------------------------------
    cols = KEY + [c for s, p in zip(score_cols, pct_cols) for c in (s, p)]
    out = df[cols].sort_values(["gvkey", "fyear"]).reset_index(drop=True)

    cfg = {"input": PANEL_IN.name, "key": ["gvkey", "fyear"],
           "models": MODELS, "outcomes": OUTCOMES,
           "percentile_column": PCT_NOTE,
           "score_column": "score_<outcome>_<model> = RAW predict_proba from "
                           "the full-sample US model, uncalibrated (see "
                           "stage6_3 metadata).",
           "n_rows": len(out), "n_scored": n_sc,
           "n_unscored": int(len(out) - n_sc),
           "unscored_note": ("Rows with any missing feature carry NA on every "
                             "score AND every percentile. Excluded by decision; "
                             "see stage6_3's unscored_reason."),
           "csv_format": "plain CSV, header on line 1; pd.read_csv(path) works"}
    out.to_csv(panel_path("csv"), index=False)
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}),
         b"stage6_4_config": json.dumps(cfg, ensure_ascii=False).encode()})
    pq.write_table(table, panel_path("parquet"))
    json.dump(cfg, open(OUT_DIR / "stage6_4_china_score_panel_meta.json", "w"),
              indent=2, ensure_ascii=False)

    print(f"\n  wrote {panel_path('parquet').name}, {panel_path('csv').name}, "
          f"stage6_4_china_score_panel_meta.json")
    print(f"  {len(out):,} rows x {out.shape[1]} cols")
    print("\nCHINESE FIRMS ARE SCORED. Nothing here is a regression — that is "
          "the wider study's work.")


if __name__ == "__main__":
    main()
