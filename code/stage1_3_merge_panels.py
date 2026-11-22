# ABOUTME: Stage 1.3 - merges the two feature panels and builds the feature inventory
# ABOUTME: Inventory is SignalDoc.csv joined to measured statistics; no judgements, only numbers
"""
stage1_3_merge_panels.py

WHAT THIS DOES:
    1. Merges the batch 1 and batch 2 feature panels into one firm-year panel.
    2. Builds a cross-sectional inventory of every characteristic, based on
       `SignalDoc.csv` and joined to statistics measured on our own panel.

    NO WINSORISATION IS APPLIED. D-25 calls for a 1%/99% trim and it is still
    the project's only outlier control, but whether it is taken on the pooled
    distribution or within fiscal year has not been decided, so it is not done
    here. The panel this writes is un-winsorised and must not be treated
    otherwise.

THE INVENTORY IS DELIBERATELY FREE OF JUDGEMENT.
    Every column is either a fact copied from SignalDoc or a number measured on
    the panel. There is no "problem" flag, no rating and no recommendation:
    missingness, zero-inflation, sign and range are reported so that the reader
    draws the conclusion. The columns that expose the known issues are:

      pct_missing     coverage
      first_year      characteristics that only begin late, e.g. after 1997
      pct_zero        exact zeros, which reveal a zero-filled input
      pct_negative    sign, which matters where a ratio should be positive
      p1 / p99 / min / max / sd    scale and tail behaviour
      n_abs_gt_1e6    values a later winsorisation will have to handle

Inputs:
    output/stage1_1_feature_panel.parquet    601,250 x 84   (77 CZ + 4 ours)
    output/stage1_2_feature_panel.parquet    601,250 x 25   (22 CZ)
    data_raw/metadata/SignalDoc.csv
    output/stage0_1_accounting_features.csv  the 101 selected characteristics

Outputs:
    output/stage1_3_feature_panel.parquet    601,250 x 106
    output/stage1_3_feature_inventory.csv    one row per characteristic

How to run:
    python3 code/stage1_3_merge_panels.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = PROJECT_ROOT / "output"
# Public Chen-Zimmermann feature catalog: download SignalDoc.csv from
# https://github.com/OpenSourceAP/CrossSection/blob/8db892442c2c3a3779b0f1eac4370d3655be15a1/SignalDoc.csv and
# save it under data_raw/metadata/ with the name below.
SIGNALDOC = PROJECT_ROOT / "data_raw" / "metadata" / "SignalDoc.csv"
# Project-authored accounting-feature screen, not a vendor download. Supply the
# original selection CSV in output/; no script in this repository creates it.
SELECTED = OUTDIR / "stage0_1_accounting_features.csv"

KEY = ["gvkey", "datadate"]

# Ours, not CZ's - they are not in SignalDoc and must never be counted among
# the 101. Kept in the panel; flagged in the inventory.
# MUST MATCH stage1_1_construct_features.py's OURS_NOT_CZ. D-50 replaced D-17's
# four `_negden` columns with OENEG and INTWO (Ohlson 1980) and ocf_at (JKP,
# Bouchaud et al. 2019); this list was stale and the 99-CZ guard below caught
# it, which is what that guard is for.
OURS_NOT_CZ = ["OENEG", "INTWO", "TLTA", "ocf_at"]

# Characteristics deliberately not built - see D-33.
NOT_BUILT = ["AccrualQuality", "AccrualQualityJune"]


def merge_panels():
    """Join the two panels on the firm-year key, refusing to proceed if they
    do not describe exactly the same rows."""
    a = pd.read_parquet(OUTDIR / "stage1_1_feature_panel.parquet")
    b = pd.read_parquet(OUTDIR / "stage1_2_feature_panel.parquet")
    print(f"  batch 1: {a.shape[0]:,} x {a.shape[1]}")
    print(f"  batch 2: {b.shape[0]:,} x {b.shape[1]}")

    # Both are built from the same master, so the keys must match exactly. If
    # they do not, one panel has been rebuilt against a different input and
    # merging would align the wrong firm-years.
    assert len(a) == len(b), "panels differ in length"
    assert a[KEY].equals(b[KEY]), "panels have different keys or ordering"

    b = b.drop(columns=[c for c in ["fyear"] if c in b.columns])
    m = a.merge(b, on=KEY, how="inner", validate="one_to_one")
    assert len(m) == len(a), "merge changed the row count"
    print(f"  merged : {m.shape[0]:,} x {m.shape[1]}   "
          f"{m.gvkey.nunique():,} firms")
    return m


def describe(panel, name):
    """Measured statistics for one characteristic. Numbers only."""
    s = panel[name].astype("float64")
    ok = s.notna()
    n = int(ok.sum())
    row = {
        "n_obs": n,
        "pct_missing": round(100 * (1 - ok.mean()), 2),
        "n_firms": int(panel.loc[ok, "gvkey"].nunique()) if n else 0,
        "first_year": int(panel.loc[ok, "fyear"].min()) if n else None,
        "last_year": int(panel.loc[ok, "fyear"].max()) if n else None,
    }
    if n:
        row.update({
            "pct_zero": round(100 * (s == 0).sum() / n, 2),
            "pct_negative": round(100 * (s < 0).sum() / n, 2),
            "mean": s.mean(), "sd": s.std(),
            "min": s.min(), "p1": s.quantile(0.01), "p25": s.quantile(0.25),
            "p50": s.median(), "p75": s.quantile(0.75),
            "p99": s.quantile(0.99), "max": s.max(),
            "n_abs_gt_1e6": int((s.abs() > 1e6).sum()),
        })
    return row


def build_inventory(panel):
    """
    One row per BUILT characteristic, based on SignalDoc and joined to our
    numbers.

    SignalDoc supplies the base list and the descriptive fields. Characteristics
    that were selected but not built are skipped rather than shown as empty
    rows, so every row here carries measured statistics. The two that are
    skipped are named in the run log and in D-33; the count of built
    characteristics is asserted in main().
    """
    doc = pd.read_csv(SIGNALDOC)
    doc.columns = [c.strip() for c in doc.columns]
    doc = doc.rename(columns={"Cat.Signal": "cat_signal",
                              "Cat.Economic": "cat_economic",
                              "Cat.Form": "cat_form"})
    keep = pd.read_csv(SELECTED).query("decision == 'keep'")["Acronym"].tolist()

    rows, skipped = [], []
    for acro in keep:
        if acro not in panel.columns:
            skipped.append(acro)
            continue
        d = doc.loc[doc["Acronym"] == acro]
        d = d.iloc[0] if len(d) else {}
        row = {
            "feature": acro,
            "in_signaldoc": True,
            "built": True,
            "cat_economic": d.get("cat_economic"),
            "cat_signal": d.get("cat_signal"),
            "cat_form": d.get("cat_form"),
            "authors": d.get("Authors"),
            "year": d.get("Year"),
            "journal": d.get("Journal"),
            "sample_start": d.get("SampleStartYear"),
            "sample_end": d.get("SampleEndYear"),
        }
        row.update(describe(panel, acro))
        rows.append(row)
    if skipped:
        print(f"  not built, so not listed: {skipped}  (D-33)")

    # Our own diagnostic variables, which are not in SignalDoc.
    for name in OURS_NOT_CZ:
        if name not in panel.columns:
            continue
        row = {"feature": name, "in_signaldoc": False, "built": True,
               "cat_economic": "(ours, not CZ)", "cat_signal": None,
               # OENEG and INTWO are 0/1 states; ocf_at is a continuous ratio.
               "cat_form": "indicator" if name in ("OENEG", "INTWO") else "ratio",
               "authors": "Ohlson" if name in ("OENEG", "INTWO", "TLTA")
                          else "Bouchaud, Kruger, Landier and Thesmar",
               "year": 1980 if name in ("OENEG", "INTWO", "TLTA") else 2019,
               "journal": None, "sample_start": None, "sample_end": None}
        row.update(describe(panel, name))
        rows.append(row)

    inv = pd.DataFrame(rows)
    order = ["feature", "cat_economic", "cat_signal", "cat_form", "built",
             "in_signaldoc", "n_obs", "pct_missing", "n_firms", "first_year",
             "last_year", "pct_zero", "pct_negative", "mean", "sd", "min",
             "p1", "p25", "p50", "p75", "p99", "max", "n_abs_gt_1e6",
             "authors", "year", "journal", "sample_start", "sample_end"]
    inv = inv[[c for c in order if c in inv.columns]]
    return inv.sort_values(["cat_economic", "pct_missing"],
                           na_position="last").reset_index(drop=True)


def main():
    print("stage1_3_merge_panels.py")
    print("  NOTE: no winsorisation applied - see D-25\n")

    panel = merge_panels()

    feats = [c for c in panel.columns if c not in KEY + ["fyear"]]
    cz = [c for c in feats if c not in OURS_NOT_CZ]
    print(f"\n  characteristics: {len(cz)} CZ + {len(OURS_NOT_CZ)} ours "
          f"= {len(feats)}")
    missing = [c for c in NOT_BUILT if c not in panel.columns]
    print(f"  deliberately not built: {missing}  (D-33)")
    assert len(cz) == 99, f"expected 99 CZ characteristics, found {len(cz)}"

    print()
    inv = build_inventory(panel)
    print(f"  inventory: {len(inv)} rows, all with measured statistics")

    panel.to_parquet(OUTDIR / "stage1_3_feature_panel.parquet", index=False)
    inv.to_csv(OUTDIR / "stage1_3_feature_inventory.csv", index=False)
    print(f"\n  -> stage1_3_feature_panel.parquet     "
          f"{panel.shape[0]:,} x {panel.shape[1]}")
    print(f"  -> stage1_3_feature_inventory.csv     {len(inv)} rows x "
          f"{inv.shape[1]} cols")


if __name__ == "__main__":
    main()
