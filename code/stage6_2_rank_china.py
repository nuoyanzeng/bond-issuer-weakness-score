"""
Stage 6.2 — Chinese rank transform. RANKING ONLY.

Calls stage3_10's `rank_within_fyear` unchanged, so the two markets cannot
diverge. No scoring, no percentile, no window cut, no sample restriction: ranks
are taken within ALL Chinese bond issuers in the panel, mirroring the US, which
ranks within all FISD issuers. Any later sample restriction happens after
scoring, not here.

NOTE, because it looks like an inconsistency and is not: FEATURES are ranked
WITHIN FISCAL YEAR (D-44, here); SCORE PERCENTILES are POOLED across the panel
(stage5_1, and stage6_4 later). Both mirror the US exactly.

Run:  python3 code/stage6_2_rank_china.py
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
import stage3_10_prepare_model_input as s3
import stage6_1_impute_china as s61

PANEL_IN = OUT_DIR / "stage6_1_china_imputed.parquet"
PANEL_OUT = OUT_DIR / "stage6_2_china_ranked.parquet"
KEY = ["gvkey", "datadate", "fyear"]


def main():
    if PANEL_OUT.exists():
        raise SystemExit(f"STOP: {PANEL_OUT.name} exists; this script never "
                         "overwrites. Move or rename it first.")
    if not PANEL_IN.exists():
        raise SystemExit(f"STOP: {PANEL_IN} not found.")

    # reset_index so the groupby inside rank_within_fyear cannot misalign on a
    # duplicate index — that, not the arithmetic, is the real failure mode.
    df = pd.read_parquet(PANEL_IN).reset_index(drop=True)
    feats = s61.read_feature_names()
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} features absent from "
                         f"{PANEL_IN.name}: {missing}")
    todo = [f for f in feats if f not in s3.RANK_EXEMPT]
    exempt = [f for f in feats if f in s3.RANK_EXEMPT]

    print("=" * 84)
    print("STAGE 6.2 — RANK TRANSFORM, within fyear, all Chinese bond issuers")
    print("=" * 84)
    print(f"  {PANEL_IN.name}: {len(df):,} rows, {df.gvkey.nunique():,} issuers, "
          f"fyear {int(df.fyear.min())}-{int(df.fyear.max())}")

    ranked = s3.rank_within_fyear(df, feats, "china")   # prints its own report

    # ---- the three property checks ---------------------------------------
    # A self-comparison against rank_within_fyear would compare the function to
    # itself and always pass. These test the PROPERTIES that define a rank
    # transform instead, and share no code with the formula.
    print("\n  CHECKS")
    if not ranked[feats].isna().equals(df[feats].isna()):
        raise SystemExit("STOP: the NA pattern changed. A rank transform must "
                         "leave missing values missing and rank nothing else.")
    print("      NA mask unchanged, cell for cell")

    # Ranking the OUTPUT must reproduce the ranks of the INPUT, within year.
    gi = df.groupby("fyear")[todo].rank()
    go = ranked.assign(fyear=df.fyear).groupby("fyear")[todo].rank()
    if not (((gi.isna() == go.isna()) & ((gi == go) | gi.isna())).all().all()):
        raise SystemExit("STOP: within-year ordering was not preserved; the "
                         "transform is not monotone, which means misalignment.")
    print(f"      within-year order preserved, all {len(todo)} ranked features")

    lo, hi = ranked[todo].min().min(), ranked[todo].max().max()
    if lo < -0.5 or hi > 0.5:
        raise SystemExit(f"STOP: ranks outside [-0.5, 0.5]: min {lo}, max {hi}")
    cnt = df.groupby("fyear")[todo].count()
    mx = ranked.assign(fyear=df.fyear).groupby("fyear")[todo].max()
    # THE MAX IS NOT ALWAYS +0.5, and assuming it was is what this guard got
    # wrong first time. `method="average"` gives the top k TIED rows a shared
    # rank of n-(k-1)/2, so after pct=True and the -0.5 shift the maximum is
    # exactly 0.5-(k-1)/(2n) — below +0.5 whenever the highest value is tied.
    # Measured on this panel: 12 year-features, all in 1994-1999, all top-ties,
    # all matching the formula to 1e-12. So require the EXACT value rather than
    # merely "a tie exists", which would wave through a real fault.
    top = df[todo] == df.groupby("fyear")[todo].transform("max")
    k_top = top.groupby(df.fyear).sum()          # rows at the max, per year-feature
    expect = 0.5 - (k_top - 1) / (2 * cnt.where(cnt >= 2))
    bad_max = int(((cnt >= 2) & ((mx - expect).abs() > 1e-12)).values.sum())
    n_row = df.groupby("fyear")[todo].transform("count")
    bad_one = int(((n_row == 1) & df[todo].notna()
                   & (ranked[todo] != 0.0)).values.sum())
    if bad_max or bad_one:
        raise SystemExit(f"STOP: edge rules broken — {bad_max} year-features "
                         f"whose max is not 0.5-(k-1)/(2n) for k rows tied at "
                         f"the top, {bad_one} cells with n=1 not set to 0.0")
    n_tied = int(((cnt >= 2) & (k_top > 1)).values.sum())
    print(f"      range [{lo:.4f}, {hi:.4f}]; max is 0.5-(k-1)/(2n) with k "
          f"tied at the top ({n_tied} year-features tied, rest at +0.5); "
          f"n=1 maps to 0.0")

    # Exempt columns must come through untouched, still three-valued.
    if not ranked[exempt].equals(df[exempt]):
        raise SystemExit(f"STOP: exempt columns changed: {exempt}")
    vals = set(pd.unique(ranked[exempt].values.ravel()))
    if not vals <= {-1.0, 0.0, 1.0}:
        raise SystemExit(f"STOP: {exempt} hold values outside "
                         f"{{-1, 0, 1}}: {sorted(vals)[:8]}")
    print(f"      exempt untouched and still only {{-1, 0, 1}}: "
          f"{', '.join(exempt)}")

    # ---- short report -----------------------------------------------------
    print(f"\n  PER FEATURE     {'ranked':>10}{'left NA':>10}")
    for f in feats:
        tag = "  (exempt, passed through)" if f in exempt else ""
        print(f"      {f:<22}{int(ranked[f].notna().sum()):>10,}"
              f"{int(ranked[f].isna().sum()):>10,}{tag}")

    print(f"\n  PER YEAR — the cross-section the ranks are taken within")
    print(f"      {'fyear':>6}{'rows':>8}{'min':>8}{'median':>9}   "
          f"(non-missing values across the {len(todo)} ranked features)")
    for y, r in cnt.iterrows():
        print(f"      {int(y):>6}{int((df.fyear == y).sum()):>8,}"
              f"{int(r.min()):>8,}{int(r.median()):>9,}")

    # ---- write -------------------------------------------------------------
    out = pd.concat([df[KEY + ["ind_cn"]], ranked[feats]], axis=1)
    cfg = {"input": PANEL_IN.name, "n_features": len(feats),
           "features_in_order": feats,
           "rank_transform": "within-fyear cross-sectional rank, ties averaged, "
                             "scaled to [-0.5, 0.5] (D-44); stage3_10's "
                             "rank_within_fyear called unchanged",
           "rank_exempt": exempt, "ranked_within": "all Chinese bond issuers, "
           "all fyears, no window cut and no sample restriction",
           "scored": False, "percentile": False}
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}),
         b"stage6_2_config": json.dumps(cfg, ensure_ascii=False).encode()})
    pq.write_table(table, PANEL_OUT)
    print(f"\n  wrote {PANEL_OUT.name}: {len(out):,} rows x {out.shape[1]} cols "
          f"({PANEL_OUT.stat().st_size / 1e6:.1f} MB)")
    print("\nNOTHING SCORED, NO PERCENTILE. Those are stage6_3 and stage6_4.")


if __name__ == "__main__":
    main()
