"""
Stage 8.5 — Export the rows behind the default test, for figure previews (rf only).

Writes  output/stage8_5_default_test_rows.csv
        columns: market, firm_id, fyear, score_rf, y_default

Written only after every check has passed. It never overwrites: the script stops
at the start if the file already exists. `data_raw/` is not read at all.

THE ROWS. These are the rows of the 2026-11-04 tests of default (`tests123.py`),
fiscal 2013-2024:
  US     output/stage8_2_us_distress_firmyear.parquet, all rows in those years,
         the "full distress panel"; 25,221 firm-years, 3,064 firms, 235 events.
  China  output/stage8_2_china_labels.parquet, rows with `scored` and `in_default`
         true in those years; 23,545 firm-years, 4,262 firms, 152 events.

THE AUC TEST AND TEST 3 USE THE SAME ROWS in each market, so there is one set of
rows and no column saying which test a row belongs to. Test 3 also ran on China's
`y_credit` rows, a different sample with a different label. Those are not exported.

WHICH COLUMN IS WHICH. `score_rf` here is always the RAW rf distress score. In the
US that is `raw_rf`, NOT the input column also named `score_rf`, which is
calibrated (D-47) and gives AUC 0.9236 and different cut points. In China it is
`score_distress_rf`, which is raw (D-67). `y_default` here is `label` in the US
and `y_default` in China. Both put one positive per issuer on its last statement
knowable before the event, but the events differ: the US counts bankruptcy
filings, and China has no bankruptcy data.

FIRM_ID. A plain integer from 1 up, numbered separately in each market, so a firm
is identified by market and firm_id together. The numbers are shuffled with a
fixed seed: numbered in sorted order, they could be mapped back from a sorted list
of gvkeys or Orgids. The real identifiers are not in the file.

THE BANDS are test 3's: cut at the 30th and 70th percentiles of the raw score on
the US rows, and those two cuts applied unchanged to China. A row is in a band if
low <= score < high.

EVERY CHECK IS COMPUTED FROM THE ROWS TO BE EXPORTED, not from the input files.
All checks are printed, and any failure stops the run with nothing written. Test 3
printed the China rates at 2 decimals, so they are compared at 2 decimals; the US
rates are compared at the 3 decimals you gave.

HIGHER SCORE = WEAKER.

Runtime: a few seconds.

Run:  python3 code/stage8_5_export_default_rows.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
OUT_CSV = OUT_DIR / "stage8_5_default_test_rows.csv"

YEAR_MIN, YEAR_MAX = 2013, 2024
SEED = 42     # fixes the shuffled firm numbers, so a re-run gives the same file

COLUMNS = ["market", "firm_id", "fyear", "score_rf", "y_default"]
MARKETS = ["US", "China"]
BAND_NAMES = ["strong", "middle", "weak"]

# The earlier results the export must reproduce: market -> firm-years, firms, events.
EXPECT_SAMPLE = {"US": (25221, 3064, 235), "China": (23545, 4262, 152)}
EXPECT_AUC = {"US": 0.924, "China": 0.713}
EXPECT_CUTS = (0.016284, 0.068767)     # test 3, distress_rf, 6 decimals

# Test 3, distress_rf: (market, band) -> firm-years, firms, events, rate in %.
EXPECT_BANDS = {("US", "strong"): (7566, 1708, 2, 0.026),
                ("US", "middle"): (10088, 2450, 15, 0.149),
                ("US", "weak"): (7567, 1998, 218, 2.881),
                ("China", "strong"): (1432, 905, 0, 0.00),
                ("China", "middle"): (11922, 3493, 45, 0.38),
                ("China", "weak"): (10191, 3209, 107, 1.05)}
RATE_DECIMALS = {"US": 3, "China": 2}


def plain_firm_ids(real_ids):
    """Replace each real identifier by an integer from 1 up, in shuffled order."""
    # Sorted first, so that the same seed gives the same numbers on every run
    # whatever the order of the input rows.
    unique_ids = np.sort(real_ids.unique())
    numbers = np.random.default_rng(SEED).permutation(len(unique_ids)) + 1
    return real_ids.map(dict(zip(unique_ids, numbers))).astype(int)


def check_shape(export, problems):
    """The file holds what was asked for, and nothing else."""
    if list(export.columns) != COLUMNS:
        problems.append(f"columns are {list(export.columns)}, not {COLUMNS}")
    if export.duplicated(["market", "firm_id", "fyear"]).any():
        problems.append("a firm-year appears more than once")
    if export[COLUMNS].isna().any().any():
        problems.append("a value is missing")
    if not export.y_default.isin([0, 1]).all():
        problems.append("y_default holds a value other than 0 and 1")
    if not export.score_rf.between(0, 1).all():
        problems.append("score_rf holds a value outside 0 to 1")
    if not export.fyear.between(YEAR_MIN, YEAR_MAX).all():
        problems.append(f"fyear falls outside {YEAR_MIN}-{YEAR_MAX}")


def check_samples_and_aucs(export, problems):
    """Sample size and exact AUC in each market, compared at the 3rd decimal."""
    print(f"  {'market':<8}{'firm-years':>11}{'firms':>8}{'events':>8}"
          f"{'AUC':>9}{'expected':>10}")
    for market in MARKETS:
        rows = export[export.market == market]
        sample = (len(rows), rows.firm_id.nunique(), int(rows.y_default.sum()))
        auc = roc_auc_score(rows.y_default, rows.score_rf)
        print(f"  {market:<8}{sample[0]:>11,}{sample[1]:>8,}{sample[2]:>8,}"
              f"{auc:>9.4f}{EXPECT_AUC[market]:>10.3f}")
        if sample != EXPECT_SAMPLE[market]:
            problems.append(f"{market}: firm-years, firms, events are {sample}, "
                            f"not {EXPECT_SAMPLE[market]}")
        if round(auc, 3) != EXPECT_AUC[market]:
            problems.append(f"{market}: AUC is {auc:.4f}, not "
                            f"{EXPECT_AUC[market]:.3f} at 3 decimals")


def check_bands(export, problems):
    """Test 3's band table, with both cuts taken from the US rows."""
    us_scores = export.loc[export.market == "US", "score_rf"]
    cut30, cut70 = np.percentile(us_scores, [30, 70])
    print(f"\n  US cuts: 30th pct = {cut30:.6f}   70th pct = {cut70:.6f}")
    if (round(cut30, 6), round(cut70, 6)) != EXPECT_CUTS:
        problems.append(f"cuts are {cut30:.6f} and {cut70:.6f}, not {EXPECT_CUTS}")

    limits = [(-np.inf, cut30), (cut30, cut70), (cut70, np.inf)]
    print(f"\n  {'market':<8}{'band':<8}{'firm-years':>11}{'firms':>8}{'events':>8}"
          f"{'rate %':>9}{'expected rate %':>17}")
    for market in MARKETS:
        rows = export[export.market == market]
        for band, (low, high) in zip(BAND_NAMES, limits):
            in_band = rows[(rows.score_rf >= low) & (rows.score_rf < high)]
            n_rows, n_firms = len(in_band), in_band.firm_id.nunique()
            n_events = int(in_band.y_default.sum())
            rate = n_events / n_rows * 100 if n_rows else float("nan")
            expected = EXPECT_BANDS[(market, band)]
            print(f"  {market:<8}{band:<8}{n_rows:>11,}{n_firms:>8,}{n_events:>8,}"
                  f"{rate:>9.3f}{expected[3]:>17}")
            found = (n_rows, n_firms, n_events,
                     round(rate, RATE_DECIMALS[market]))
            if found != expected:
                problems.append(f"{market} {band}: {found}, not {expected}")


def main():
    # Never overwrite. Checked first, so a refused run costs nothing.
    if OUT_CSV.exists():
        raise SystemExit(f"STOP: {OUT_CSV.name} already exists. This script never "
                         "overwrites. NOTHING WRITTEN.")

    # ---- the rows behind the tests -----------------------------------------
    # Only the columns needed. The calibrated US `score_rf` is deliberately not
    # read, so it cannot be picked up by mistake.
    us = pd.read_parquet(OUT_DIR / "stage8_2_us_distress_firmyear.parquet",
                         columns=["gvkey", "fyear", "raw_rf", "label"])
    cn = pd.read_parquet(OUT_DIR / "stage8_2_china_labels.parquet",
                         columns=["Orgid", "fyear", "scored", "in_default",
                                  "score_distress_rf", "y_default"])
    us = us[us.fyear.between(YEAR_MIN, YEAR_MAX)]
    cn = cn[cn.scored & cn.in_default & cn.fyear.between(YEAR_MIN, YEAR_MAX)]

    # ---- the export: same five columns in both markets, real identifiers gone
    us_rows = pd.DataFrame({"market": "US",
                            "firm_id": plain_firm_ids(us.gvkey),
                            "fyear": us.fyear.astype(int),
                            "score_rf": us.raw_rf,
                            "y_default": us.label.astype(int)})
    cn_rows = pd.DataFrame({"market": "China",
                            "firm_id": plain_firm_ids(cn.Orgid),
                            "fyear": cn.fyear.astype(int),
                            "score_rf": cn.score_distress_rf,
                            "y_default": cn.y_default.astype(int)})

    # Rows are ordered by the new number, so the file order follows no real
    # identifier either.
    export = pd.concat([us_rows.sort_values(["firm_id", "fyear"]),
                        cn_rows.sort_values(["firm_id", "fyear"])],
                       ignore_index=True)

    # ---- every check, before anything is written ---------------------------
    problems = []
    check_shape(export, problems)
    check_samples_and_aucs(export, problems)
    check_bands(export, problems)

    if problems:
        print("\nSTOP: a check failed. NOTHING WRITTEN.")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("\n  PASSED — samples, AUCs, cuts and bands all match the earlier output")

    # ---- save --------------------------------------------------------------
    export.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV.name}: {len(export):,} rows, "
          f"{OUT_CSV.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
