"""
Stage 3.6 — The FEATURES panel, keyed issuer_id x gvkey x fyear.

Features only. No labels, no availability guard applied, no year filtering, no
row dropping. Nothing is read from stage3_0 or stage3_2.

THE KEY IS issuer_id x gvkey x fyear, NOT issuer_id x fyear. Some issuers match
more than one gvkey, and all of those pairs are kept: choosing which gvkey an
issuer "is" would be a firm-identity decision, and it is deferred to the
modelling-table stage along with the multiple-issuer_id-per-firm duplication.

Run:  python3 code/stage3_6_features_panel.py
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

BRIDGE_IN = OUT_DIR / "stage3_4_issuer_gvkey_bridge.parquet"

# Availability lag of the input panel, in months (D-41). 3 is the main setting;
# 0 is the no-guard comparison. ONLY the two filenames below depend on it — no
# rule anywhere in this script does.
LAG_MONTHS = 3

FEATURES_IN = OUT_DIR / f"stage3_5_panel_with_dateavail_L{LAG_MONTHS}.parquet"
# The L = 3 panel was built before this switch existed and carries no suffix.
# Renaming it would break stage3_7 and stage3_10, so it keeps its name.
PANEL_OUT = OUT_DIR / ("stage3_6_features_panel.parquet" if LAG_MONTHS == 3
                       else f"stage3_6_features_panel_L{LAG_MONTHS}.parquet")

# Bridge rows with this match type are unresolved one-to-many candidates and
# are not part of the sample.
AMBIGUOUS = "crsp_ambiguous"

KEY_COLS = ["gvkey", "fyear", "datadate", "date_avail"]
# 45 after V3: 55 kept by the Stage 1.6 screen, minus the 10 dropped by D-58's
# redundancy screen. 41 continuous + OENEG + INTWO + ConvDebt + NumEarnIncrease.
# (Was 43 under D-38's collapse of 54.) This is one of D-55's six duplicated
# feature counts; it is still hard-coded here because fixing that is its own
# round, so it must be changed by hand whenever the screen changes.
EXPECTED_FEATURES = 45


def main():
    # This script only ever creates a new file.
    if PANEL_OUT.exists():
        raise SystemExit(
            f"STOP: {PANEL_OUT.name} already exists. This script never overwrites; "
            "move or rename the existing file first."
        )

    # ---- step 1: the sample ------------------------------------------------
    print("=" * 78)
    print("STEP 1 — the sample")
    print("=" * 78)
    br = pd.read_parquet(BRIDGE_IN)
    print(f"  bridge rows: {len(br):,}")
    acc = br[br.match_type != AMBIGUOUS]
    print(f"  dropped '{AMBIGUOUS}': {len(br) - len(acc):,} rows")
    pairs = acc[["issuer_id", "gvkey"]].drop_duplicates()
    print(f"  accepted issuers: {pairs.issuer_id.nunique():,}   "
          f"(issuer, gvkey) pairs: {len(pairs):,}")

    # Record the one-to-many extent in the run output, not only in conversation.
    per = pairs.groupby("issuer_id")["gvkey"].nunique()
    multi = per[per > 1]
    print(f"  issuers matching MORE THAN ONE gvkey: {len(multi):,} (max {int(per.max())})")
    print(f"  extra pairs they contribute: {int(multi.sum() - len(multi)):,}")

    # ---- step 2: the join --------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 2 — join to the feature panel")
    print("=" * 78)
    feat = pd.read_parquet(FEATURES_IN)
    print(f"  {FEATURES_IN.name}: {len(feat):,} rows, "
          f"{feat.gvkey.nunique():,} gvkeys")

    features = [c for c in feat.columns if c not in ("gvkey", "datadate", "fyear",
                                                     "date_avail")]
    if len(features) != EXPECTED_FEATURES:
        raise SystemExit(
            f"STOP: expected {EXPECTED_FEATURES} features, found {len(features)}."
        )

    # Every row of each matched gvkey, the issuer's full history. Inner join, so
    # a gvkey absent from the panel simply contributes nothing.
    panel = pairs.merge(feat, on="gvkey", how="inner")

    with_rows = set(panel.issuer_id)
    zero_rows = set(pairs.issuer_id) - with_rows

    # ---- guards ------------------------------------------------------------
    out_cols = ["issuer_id"] + KEY_COLS + features
    if sorted(panel.columns) != sorted(out_cols):
        raise SystemExit(
            f"STOP: unexpected column set. Missing "
            f"{sorted(set(out_cols) - set(panel.columns))}, extra "
            f"{sorted(set(panel.columns) - set(out_cols))}."
        )
    if len(panel.columns) != EXPECTED_FEATURES + 5:
        raise SystemExit(
            f"STOP: expected {EXPECTED_FEATURES + 5} columns, got {len(panel.columns)}."
        )
    # The declared key must actually be unique, or the file misdescribes itself.
    # It is datadate, not fyear: three gvkeys changed fiscal year-end and carry
    # two statements inside one fyear, so (gvkey, fyear) is not unique in the
    # feature panel itself. Nothing is dropped for this; the key is named for
    # what is actually unique.
    if panel.duplicated(subset=["issuer_id", "gvkey", "datadate"]).any():
        raise SystemExit(
            "STOP: issuer_id x gvkey x datadate is not unique; the declared key "
            "does not hold."
        )

    panel = panel[out_cols].sort_values(["issuer_id", "gvkey", "fyear"]) \
        .reset_index(drop=True)
    panel.to_parquet(PANEL_OUT, index=False)

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("REPORT")
    print("=" * 78)
    print(f"  issuers in the sample          : {pairs.issuer_id.nunique():,}")
    print(f"  issuers with >=1 feature row   : {len(with_rows):,}")
    print(f"  issuers with ZERO rows         : {len(zero_rows):,}"
          f"{'  ' + str(sorted(zero_rows)) if 0 < len(zero_rows) <= 10 else ''}")
    print(f"  total rows                     : {len(panel):,}")
    print(f"  distinct gvkeys                : {panel.gvkey.nunique():,}")
    print(f"  fyear range                    : {int(panel.fyear.min())} to "
          f"{int(panel.fyear.max())}")

    # The consequence of keeping every pair, stated in the run output.
    def dups(cols):
        n = int(panel.duplicated(subset=cols).sum())
        k = panel[panel.duplicated(subset=cols, keep=False)].issuer_id.nunique()
        return n, k

    d_key, k_key = dups(["issuer_id", "gvkey", "datadate"])
    d_fy, k_fy = dups(["issuer_id", "gvkey", "fyear"])
    d_if, k_if = dups(["issuer_id", "fyear"])
    print("\n  ON THE KEY — this panel is issuer_id x gvkey x DATADATE.")
    print("  It is NOT issuer_id x fyear, and not issuer_id x gvkey x fyear.")
    print(f"      issuers matching more than one gvkey     : {len(multi):,}")
    print(f"      extra (issuer, gvkey) pairs              : {int(multi.sum() - len(multi)):,}")
    print(f"      dup (issuer_id, gvkey, datadate) rows    : {d_key:,}   "
          f"issuers {k_key:,}   <- the declared key, must be 0")
    print(f"      dup (issuer_id, gvkey, fyear) rows       : {d_fy:,}   "
          f"issuers {k_fy:,}   <- fiscal year-end changes")
    print(f"      dup (issuer_id, fyear) rows              : {d_if:,}   "
          f"issuers {k_if:,}   <- multi-gvkey issuers")
    print("      Nothing is dropped. Which gvkey an issuer 'is' is a firm-identity")
    print("      question, settled at the modelling-table stage, not here.")

    print(f"\nWrote {PANEL_OUT.name}: {len(panel):,} rows x {len(panel.columns)} columns")
    print("\n  head:")
    print(panel[["issuer_id", "gvkey", "fyear", "datadate", "date_avail"]]
          .head(5).to_string(index=False))
    print("\nFeatures only. No labels, no guard applied, no year filtering, "
          "nothing read from stage3_0 or stage3_2.")


if __name__ == "__main__":
    main()
