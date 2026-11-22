"""
Stage 2.4 — Apply the redundancy removal decided by the Stage 2.3 screen.

THE RULE IS D-58, and it replaces D-38's. Stage 2.3 no longer produces GROUPS:
it collapses PAIRWISE, so there are no connected components and no group
representatives. Eleven pairs cleared |rho| >= 0.70 on the 1993-2025 bond-issuer
object, one was skipped because a member had already gone, and the remaining ten
each dropped one member. 55 features become 45.

WHERE THE DROP LIST COMES FROM. It is READ from
`output/stage2_3_redundancy_drops.csv`, not hard-coded here. D-55 records six
files carrying duplicated hard-coded feature lists with nothing linking them,
every one of which went stale and failed during the D-50 rebuild; this would
have been a seventh. The ten names are still written out in EXPECTED_DROPS
below, but as a GUARD against the file rather than as the source of truth - so
the applied set stays visible in a diff and a changed file cannot silently
change what is dropped.

ONE LABEL IN THE DROPS FILE IS MISLEADING, AND IT IS NOT A BUG TO FIX HERE.
The file records `ChNCOA` as dropped for `dNoa`, and then records `dNoa` as
dropped for `AssetGrowth`. So ChNCOA's stated survivor DOES NOT SURVIVE - both
are gone. This is the hub case D-58 anticipated when it rejected the
consensus-rank tie-break, and it occurred on the very first run. It does NOT
change which features are kept: had the ChNCOA/dNoa pair not fired first, the
AssetGrowth/ChNCOA pair (rho 0.7781, order 6) would have dropped ChNCOA anyway,
since AssetGrowth scores 0.2180 against ChNCOA's 0.1779. Only the attribution
is wrong, not the outcome. Do not read `kept_instead` as a promise of survival.

This script removes COLUMNS and nothing else. No value is recomputed, no row is
dropped, nothing is standardised or imputed. The four non-continuous features
and the three key columns are carried through untouched.

Run:  python3 code/stage2_4_apply_redundancy.py
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

PANEL_IN = OUT_DIR / "stage2_features_clean_winsored.parquet"
PANEL_OUT = OUT_DIR / "stage2_4_features_reduced.parquet"

# The screen's decision, read rather than restated. See the docstring.
DROPS_IN = OUT_DIR / "stage2_3_redundancy_drops.csv"

KEY_COLS = ["gvkey", "datadate", "fyear"]

# The 0/1 indicators. NONE of them is dropped here and none was ever screened -
# Stage 2.3 holds all four out of its correlation matrix.
# ALL FOUR, not two: OENEG and INTWO are D-50's Ohlson states, and ConvDebt and
# NumEarnIncrease are equally non-continuous (D-25 exempts all four from
# winsorising for the same reason). Counting the latter two as continuous is
# what made the earlier version of this file expect 41 continuous out of a set
# that contained 43.
NON_CONTINUOUS = ["OENEG", "INTWO", "ConvDebt", "NumEarnIncrease"]

# THE TEN DROPS, as a guard on the file - not as the source of truth.
# Produced by the run of 2026-08-30, which reproduced all six of D-58's
# published figures exactly. If the file no longer matches this list, the screen
# has been re-run to a different result and that must be looked at, not applied.
EXPECTED_DROPS = [
    "AssetLiquidityBook",   # for AssetGrowth        rho  0.8168
    "CapTurnover",          # for OPLeverage         rho  0.7995
    "ChEQ",                 # for DelEqu             rho  0.7115
    "ChNCOA",               # for dNoa               rho  0.8276  <- see docstring
    "DelNetFin",            # for DelFINL            rho -0.7956
    "GrLTNOA",              # for ChNNCOA            rho  0.7031
    "NetDebtFinance",       # for DelFINL            rho  0.7704
    "NetEquityFinance",     # for XFIN               rho  0.7482
    "currat",               # for quick              rho  0.9822
    "dNoa",                 # for AssetGrowth        rho  0.7697
]

# What the result must look like. Checked, not assumed.
# 55 kept by the Stage 1.6 screen, minus the 10 dropped here = 45, of which 41
# are continuous and 4 are the indicators above.
EXPECTED_DROP_COUNT = 10
EXPECTED_FEATURES = 45
EXPECTED_CONTINUOUS = 41


def load_drops():
    """Read the screen's decision and check it against EXPECTED_DROPS."""
    if not DROPS_IN.exists():
        raise SystemExit(
            f"STOP: {DROPS_IN.name} is not there. Run stage2_3_redundancy.py first; "
            "this script does not carry its own copy of the drop list."
        )

    drops = pd.read_csv(DROPS_IN)
    names = sorted(drops.dropped)

    print(f"Drop list: {DROPS_IN.name}")
    print(f"  {len(names)} features to drop")

    if len(names) != EXPECTED_DROP_COUNT:
        raise SystemExit(
            f"STOP: the drop list has {len(names)} entries, expected "
            f"{EXPECTED_DROP_COUNT}. The screen has produced a different result; "
            "look at it before applying it."
        )
    if names != sorted(EXPECTED_DROPS):
        raise SystemExit(
            "STOP: the drop list does not match the ten names this script was "
            "written against.\n"
            f"  in the file, not expected: {sorted(set(names) - set(EXPECTED_DROPS))}\n"
            f"  expected, not in the file: {sorted(set(EXPECTED_DROPS) - set(names))}"
        )
    print(f"  matches EXPECTED_DROPS exactly")

    # Print the decision as the screen recorded it, with the caveat attached.
    print("\n  each drop, as Stage 2.3 recorded it:")
    for row in drops.sort_values("dropped").itertuples():
        print(f"      drop {row.dropped:<20} kept_instead {row.kept_instead:<14} "
              f"rho {row.rho:>8.4f}")
    unkept = [r.dropped for r in drops.itertuples()
              if r.kept_instead in set(drops.dropped)]
    if unkept:
        print(f"\n  NOTE: {', '.join(unkept)} - the feature named in kept_instead "
              "was ITSELF dropped later.")
        print("        `kept_instead` records which pair caused the drop, not that "
              "that feature survived.")
        print("        This is the hub case in D-58; it does not change which "
              "features are kept. See the docstring.")

    return list(drops.dropped)


def main():
    # Refuse to overwrite. This script only ever creates a new file.
    if PANEL_OUT.exists():
        raise SystemExit(
            f"STOP: {PANEL_OUT.name} already exists. This script never overwrites; "
            "move or rename the existing file first."
        )

    to_drop = load_drops()

    # Read the schema before loading, so the checks below run on the real
    # columns rather than on an assumption about them.
    all_cols = pq.ParquetFile(PANEL_IN).schema_arrow.names
    features_in = [c for c in all_cols if c not in KEY_COLS]

    print(f"\nInput: {PANEL_IN.name}")
    print(f"  {len(features_in)} feature columns")

    # Every feature we intend to drop must actually be there.
    absent = [f for f in to_drop if f not in features_in]
    if absent:
        raise SystemExit(f"STOP: features to drop are not in the panel: {absent}")

    # None of the four indicators may be dropped - the screen never saw them.
    dropped_indicators = [f for f in to_drop if f in NON_CONTINUOUS]
    if dropped_indicators:
        raise SystemExit(
            f"STOP: the drop list contains non-continuous features, which Stage "
            f"2.3 never screened: {dropped_indicators}"
        )

    # The column removal itself.
    keep_cols = [c for c in all_cols if c not in to_drop]
    panel = pd.read_parquet(PANEL_IN, columns=keep_cols)

    features_out = [c for c in panel.columns if c not in KEY_COLS]
    continuous_out = [c for c in features_out if c not in NON_CONTINUOUS]
    indicators_out = [c for c in features_out if c in NON_CONTINUOUS]

    # The three count assertions.
    if len(features_out) != EXPECTED_FEATURES:
        raise SystemExit(
            f"STOP: expected {EXPECTED_FEATURES} feature columns, got {len(features_out)}."
        )
    if len(continuous_out) != EXPECTED_CONTINUOUS:
        raise SystemExit(
            f"STOP: expected {EXPECTED_CONTINUOUS} continuous features, got "
            f"{len(continuous_out)}."
        )
    if len(indicators_out) != len(NON_CONTINUOUS):
        raise SystemExit(
            f"STOP: expected {len(NON_CONTINUOUS)} non-continuous features, got "
            f"{len(indicators_out)}: {indicators_out}"
        )

    # No row may be lost — this is a column operation.
    rows_in = pq.ParquetFile(PANEL_IN).metadata.num_rows
    if len(panel) != rows_in:
        raise SystemExit(f"STOP: row count changed, {rows_in:,} in, {len(panel):,} out.")

    panel.to_parquet(PANEL_OUT, index=False)

    # Report what was written.
    print(f"\nOutput: {PANEL_OUT.name}")
    print(f"  {len(panel.columns)} columns = {len(features_out)} features + "
          f"{len(KEY_COLS)} keys")
    print(f"  {len(panel):,} rows (input had {rows_in:,} — unchanged)")
    print(f"  {panel['gvkey'].nunique():,} firms")

    print(f"\n  {len(continuous_out)} continuous features kept:")
    for i, name in enumerate(continuous_out, start=1):
        print(f"      {i:>2}. {name}")

    print(f"\n  {len(indicators_out)} non-continuous features kept "
          f"(never screened):")
    for name in indicators_out:
        print(f"          {name}")

    print("\nColumn removal only — no value recomputed, no row dropped, "
          "nothing standardised or imputed.")


if __name__ == "__main__":
    main()
