"""
Stage 3.5 — Add the availability guard column `date_avail` to the feature panel.

date_avail = datadate + AVAIL_LAG_MONTHS months.

This step ONLY adds the column. It does not build the modelling panel, does not
apply the guard to any event, and does not touch the bridge or stage2_4.

Run:  python3 code/stage3_5_add_dateavail.py
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# THE ONE SETTING. The main lag, per D-41.
#   3  main setting: the 10-K filing deadline
#   6  robustness
#   0  no-guard comparison
AVAIL_LAG_MONTHS = 3

# Which versions to build. One file per lag, named _L{lag}. A lag whose file
# already exists is skipped rather than rebuilt, so an existing verified panel
# is never recomputed.
LAGS_TO_BUILD = [AVAIL_LAG_MONTHS, 0]
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

PANEL_IN = OUT_DIR / "stage2_4_features_reduced.parquet"

EXPECTED_ROWS = 601_250


def out_path(lag):
    """One output file per lag, so the versions cannot be confused."""
    return OUT_DIR / f"stage3_5_panel_with_dateavail_L{lag}.parquet"


def build(panel, lag, rows_in, cols_in, nulls_datadate):
    """Write one version of the panel for a given lag, with its guards."""
    path = out_path(lag)
    if path.exists():
        print(f"\n  L={lag}: {path.name} already exists — SKIPPED, not rebuilt.")
        return

    out = panel.copy()
    # Calendar month arithmetic, not a day approximation: adding 3 months to a
    # month-end date lands on the corresponding month end, so 30 November + 3
    # gives 28 February rather than 28 February plus two stray days. At lag 0
    # this is the identity, so date_avail equals datadate exactly.
    out["date_avail"] = out.datadate + pd.DateOffset(months=lag)

    # ---- the guards, per version -----------------------------------------
    if len(out) != rows_in or len(out) != EXPECTED_ROWS:
        raise SystemExit(
            f"STOP: row count wrong for L={lag}: {len(out):,} against "
            f"{rows_in:,} in and {EXPECTED_ROWS:,} expected."
        )
    if out.shape[1] != cols_in + 1:
        raise SystemExit(
            f"STOP: expected exactly one new column for L={lag}, went from "
            f"{cols_in} to {out.shape[1]}."
        )
    nulls_avail = int(out.date_avail.isna().sum())
    if nulls_avail != nulls_datadate:
        raise SystemExit(
            f"STOP: date_avail has {nulls_avail:,} nulls against datadate's "
            f"{nulls_datadate:,} for L={lag}; the offset lost rows."
        )
    # At lag 0 the guard is switched off, so the two columns must be identical.
    if lag == 0 and not out.date_avail.equals(out.datadate):
        raise SystemExit("STOP: at L=0, date_avail does not equal datadate.")

    out.to_parquet(path, index=False)
    print(f"\n  L={lag}: wrote {path.name}")
    print(f"      {len(out):,} rows x {out.shape[1]} columns  (+1: date_avail)")
    print(f"      date_avail nulls {nulls_avail:,}  range "
          f"{out.date_avail.min().date()} to {out.date_avail.max().date()}")


def main():
    panel = pd.read_parquet(PANEL_IN)
    rows_in, cols_in = panel.shape
    print("=" * 74)
    print("STEP 0 — the input")
    print("=" * 74)
    print(f"  {PANEL_IN.name}: {rows_in:,} rows x {cols_in} columns")
    print(f"  datadate dtype : {panel.datadate.dtype}")
    nulls_datadate = int(panel.datadate.isna().sum())
    print(f"  datadate nulls : {nulls_datadate:,}")

    # A missing statement date cannot be given an availability date, and
    # guessing one would defeat the guard. Stop rather than impute.
    if nulls_datadate:
        raise SystemExit(
            f"STOP: datadate is null on {nulls_datadate:,} rows. No availability "
            "date can be derived for them and none will be imputed."
        )
    if "date_avail" in panel.columns:
        raise SystemExit("STOP: the panel already carries a date_avail column.")

    print("\n" + "=" * 74)
    print(f"STEP 1 — building versions for L = {LAGS_TO_BUILD}")
    print("=" * 74)
    for lag in LAGS_TO_BUILD:
        build(panel, lag, rows_in, cols_in, nulls_datadate)

    # ---- cross-check the versions against each other ----------------------
    print("\n" + "=" * 74)
    print("CROSS-CHECK")
    print("=" * 74)
    versions = {lag: pd.read_parquet(out_path(lag))
                for lag in LAGS_TO_BUILD if out_path(lag).exists()}
    for lag, v in versions.items():
        same = v.date_avail.equals(v.datadate + pd.DateOffset(months=lag))
        print(f"  L={lag}: shape {v.shape}   "
              f"date_avail == datadate + {lag} months: {same}")
        if not same:
            raise SystemExit(f"STOP: L={lag} column does not match its own rule.")

    lags = sorted(versions)
    if len(lags) == 2:
        a, b = versions[lags[0]], versions[lags[1]]
        if a.shape != b.shape:
            raise SystemExit(f"STOP: shapes differ, {a.shape} vs {b.shape}.")
        others = [c for c in a.columns if c != "date_avail"]
        # Only the availability column may differ between the versions.
        identical = a[others].equals(b[others])
        print(f"  the two versions share the same shape: {a.shape == b.shape}")
        print(f"  every column except date_avail is identical: {identical}")
        if not identical:
            raise SystemExit("STOP: the versions differ outside date_avail.")
        print(f"  date_avail differs between them on "
              f"{int((a.date_avail != b.date_avail).sum()):,} of {len(a):,} rows")

        print("\n  5 sample rows, both versions side by side:")
        cmp = a[["gvkey", "fyear", "datadate", "date_avail"]].head(5).copy()
        cmp.columns = ["gvkey", "fyear", "datadate", f"date_avail_L{lags[0]}"]
        cmp[f"date_avail_L{lags[1]}"] = b.date_avail.head(5).values
        print(cmp.to_string(index=False))

    print("\nColumn added only. No modelling panel, no guard applied to any event, "
          "no change to stage2_4 or the bridge.")


if __name__ == "__main__":
    main()
