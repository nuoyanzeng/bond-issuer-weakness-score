"""
Stage 8.3 — Export binned rf scores for the next-year deterioration figures.

WRITES THREE FILES, all from the same rows, only after every check has passed.
`data_raw/` is read only.

  output/stage8_3_rf_score_bins.csv        100 bins of width 0.01, 1,200 rows
  output/stage8_3_rf_score_bins_1000.csv   1,000 bins of width 0.001, 12,000 rows
  output/stage8_3_rf_auc.csv               market, family, outcome, n, auc; 12 rows

THE 100-BIN FILE IS NEVER REWRITTEN. If it already exists, the counts are computed
again and compared with it character by character: a match leaves the file
untouched, and any difference stops the run with nothing written. The two other
files are never overwritten: the script stops at the start if either exists.

WHAT IT PRODUCES. rf only: 2 markets x 2 score families x 3 next-year outcomes =
12 cells. In each cell the RAW score goes into equal-width bins on [0, 1], and
the firm-years with outcome = 1 and outcome = 0 are counted in each bin, in
columns market, family, outcome, bin_lo, n_pos, n_neg. `auc` in the 12-row file
is the exact AUC on the RAW score, so for the US it is not the calibrated figure
in the earlier table (see below).

THE SAMPLE IS THE ONE BEHIND THE EARLIER AUC TABLE (auc_three_outcomes.py, run
2026-11-03), fiscal 2003-2024, each family on its own panel. It is not rebuilt:
the outcomes are imported from `check_china_score_validity.py`, the same
construction that table and stage8_0 import. Rows with the outcome unobserved
are left out of that cell, so n differs by outcome.

The panels are read from the stage8_2 files because they are the only gvkey x
fyear files holding the raw US scores. They hold the same rows as that table's
panels: measured on 2026-11-04, 24,557 / 46,031 / 27,600 rows all matched, and
the rf scores were identical, with a maximum difference of 0.

WHICH COLUMN IS WHICH. In China `score_distress_rf` and `score_rating_rf` ARE
raw (D-67), and there is no calibrated version. In the US `raw_rf` is raw and
`score_rf` is out-of-fold calibrated (D-47).

THE TABLE CHECK, AND WHY THE US SIDE USES `score_rf`. The earlier table's US
figures were computed on the CALIBRATED `score_rf`, not on raw scores. Comparing
raw AUCs with them would compare two different scores, and one cell could never
pass: US distress, interest cover, is 0.8656 on raw against 0.8649 in the table,
so 0.866 against 0.865 at 3 decimals. So each cell is checked on the column the
table actually used, which confirms the rows and the outcomes are the same. The
raw AUC is printed beside it. The bins always hold the RAW score.

HIGHER SCORE = WEAKER, so an AUC above 0.5 is the expected direction.

Runtime: about a minute, nearly all of it reading the CSMAR files for the outcomes.

Run:  python3 code/stage8_3_rf_score_bins.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
OUT_CSV = OUT_DIR / "stage8_3_rf_score_bins.csv"
OUT_CSV_1000 = OUT_DIR / "stage8_3_rf_score_bins_1000.csv"
OUT_AUC = OUT_DIR / "stage8_3_rf_auc.csv"

# The outcome construction is imported, never re-derived, so it cannot drift from
# the earlier table. Importing the module runs only its constants and function
# definitions; its main() sits behind a __name__ guard.
sys.path.insert(0, str(PROJECT_DIR / "code"))
from check_china_score_validity import (auc, china_items, csmar, lookup,
                                        outcomes, us_items)

YEAR_MIN, YEAR_MAX = 2003, 2024
OUTCOMES = ["intcov_t1", "ebitdaneg_t1", "loss_t1"]   # next-year outcomes

# 101 edges make 100 bins: 0.00, 0.01, ... 1.00. np.histogram closes only the
# last bin on the right, so a score of exactly 1.0 goes into the last bin.
EDGES = np.arange(101) / 100
# The same for the finer file: 1,001 edges make 1,000 bins of width 0.001.
EDGES_1000 = np.arange(1001) / 1000

# Row counts of the three samples. A mismatch means the sample is not the same.
EXPECT_ROWS = {"China": 24557, "US distress": 46031, "US rating": 27600}

# The rf columns of the earlier AUC table, at the 4 decimals it printed.
EXPECT_AUC = {("China", "distress", "intcov_t1"): 0.8203,
              ("China", "distress", "ebitdaneg_t1"): 0.8463,
              ("China", "distress", "loss_t1"): 0.8163,
              ("China", "rating", "intcov_t1"): 0.8347,
              ("China", "rating", "ebitdaneg_t1"): 0.8623,
              ("China", "rating", "loss_t1"): 0.8406,
              ("US", "distress", "intcov_t1"): 0.8649,
              ("US", "distress", "ebitdaneg_t1"): 0.8396,
              ("US", "distress", "loss_t1"): 0.8386,
              ("US", "rating", "intcov_t1"): 0.8491,
              ("US", "rating", "ebitdaneg_t1"): 0.7851,
              ("US", "rating", "loss_t1"): 0.8368}

MAX_BIN_DIFF = 0.005     # largest allowed |binned AUC - exact AUC|, 100 bins
MAX_BIN_DIFF_1000 = 0.002    # the same, 1,000 bins


def binned_auc(n_pos, n_neg):
    """AUC from the bin counts alone.

    A positive outranks every negative in a lower bin, and counts a half against
    each negative in its own bin, because their order inside a bin is unknown.
    """
    negatives_below = np.concatenate([[0], np.cumsum(n_neg)[:-1]])
    wins = (n_pos * (negatives_below + 0.5 * n_neg)).sum()
    return wins / (n_pos.sum() * n_neg.sum())


def main():
    # Never overwrite. Checked first, so a refused run costs nothing. The 100-bin
    # file is not refused here: it is compared with the new counts further down.
    for path in [OUT_CSV_1000, OUT_AUC]:
        if path.exists():
            raise SystemExit(f"STOP: {path.name} already exists. This script "
                             "never overwrites. NOTHING WRITTEN.")

    # ---- the three panels, fiscal 2003-2024 --------------------------------
    cn = pd.read_parquet(OUT_DIR / "stage8_2_china_labels.parquet",
                         columns=["Orgid", "fyear", "scored",
                                  "score_distress_rf", "score_rating_rf"])
    usd = pd.read_parquet(OUT_DIR / "stage8_2_us_distress_firmyear.parquet",
                          columns=["gvkey", "fyear", "raw_rf", "score_rf"])
    usr = pd.read_parquet(OUT_DIR / "stage8_2_us_rating_firmyear.parquet",
                          columns=["gvkey", "fyear", "raw_rf", "score_rf"])

    # China keeps scored rows only. The index is reset so that outcomes and scores
    # share one index, which `auc` insists on.
    cn = cn[cn.scored & cn.fyear.between(YEAR_MIN, YEAR_MAX)].reset_index(drop=True)
    usd = usd[usd.fyear.between(YEAR_MIN, YEAR_MAX)].reset_index(drop=True)
    usr = usr[usr.fyear.between(YEAR_MIN, YEAR_MAX)].reset_index(drop=True)

    for name, panel in [("China", cn), ("US distress", usd), ("US rating", usr)]:
        print(f"  {name:<12} {len(panel):>7,} rows   expected "
              f"{EXPECT_ROWS[name]:,}")
        if len(panel) != EXPECT_ROWS[name]:
            raise SystemExit(f"STOP: {name} is not the earlier sample. "
                             "NOTHING WRITTEN.")

    # ---- the outcomes, from the imported construction ----------------------
    print("\n  building the raw item tables (CSMAR and Compustat) — the slow part")
    bond_master = csmar("债券基本情况表.csv",
                        ["Liscd", "Sctcd", "Orgid"]).drop_duplicates()
    cn_items = china_items(bond_master)
    us_item_table = us_items()

    def outcomes_for(panel, firm_col, items):
        """Observed outcomes on the panel's own index, built as the table did."""
        return outcomes({o: lookup(panel, firm_col, items, o)
                         for o in (0, 1, 2, 3)})

    y_cn = outcomes_for(cn, "Orgid", cn_items)
    y_usd = outcomes_for(usd, "gvkey", us_item_table)
    y_usr = outcomes_for(usr, "gvkey", us_item_table)

    # One line per market x family: the panel, its outcomes, the raw score that
    # gets binned, the column the earlier table used, and the firm identifier.
    groups = [("US", "distress", usd, y_usd, "raw_rf", "score_rf", "gvkey"),
              ("US", "rating", usr, y_usr, "raw_rf", "score_rf", "gvkey"),
              ("China", "distress", cn, y_cn, "score_distress_rf",
               "score_distress_rf", "Orgid"),
              ("China", "rating", cn, y_cn, "score_rating_rf",
               "score_rating_rf", "Orgid")]

    # ---- bin each cell and compute its checks ------------------------------
    print("\n  raw = exact AUC on the raw score; binned = the same from the 100 "
          "bins; bin1000 =\n  the same from the 1,000 bins; table = the earlier "
          "AUC table; on table col = the\n  exact AUC on the column that table "
          "used, which is what is compared with it at\n  3 decimals.\n")
    print(f"  {'market':<7}{'family':<10}{'outcome':<14}{'n':>7}{'pos':>7}"
          f"{'firms':>7}{'raw':>9}{'binned':>9}{'diff':>9}{'bin1000':>9}"
          f"{'diff':>9}{'table':>8}{'on table col':>14}  table col")

    tables, tables_1000, auc_rows, problems = [], [], [], []
    for market, family, panel, ys, raw_col, table_col, firm_col in groups:
        for outcome in OUTCOMES:
            label = f"{market} {family} {outcome}"

            # Exact AUC. `auc` returns None if either class is below its floor.
            exact, n, n_pos_total = auc(ys[outcome], panel[raw_col])
            on_table_col, _, _ = auc(ys[outcome], panel[table_col])
            if exact is None or on_table_col is None:
                raise SystemExit(f"STOP: {label} has too few of one class for "
                                 f"an AUC (n={n:,}, pos={n_pos_total:,}). "
                                 "NOTHING WRITTEN.")

            # The same rows `auc` used: outcome observed and score present.
            keep = ys[outcome].notna() & panel[raw_col].notna()
            y = ys.loc[keep, outcome].astype(int).to_numpy()
            score = panel.loc[keep, raw_col].to_numpy()
            n_firms = panel.loc[keep, firm_col].nunique()

            # Count positives and negatives per bin. np.histogram silently drops
            # anything outside [0, 1], so the totals are checked just below.
            n_pos, _ = np.histogram(score[y == 1], bins=EDGES)
            n_neg, _ = np.histogram(score[y == 0], bins=EDGES)
            n_pos_1000, _ = np.histogram(score[y == 1], bins=EDGES_1000)
            n_neg_1000, _ = np.histogram(score[y == 0], bins=EDGES_1000)
            if (n_pos.sum() != n_pos_total or n_neg.sum() != n - n_pos_total
                    or n_pos_1000.sum() != n_pos_total
                    or n_neg_1000.sum() != n - n_pos_total):
                raise SystemExit(f"STOP: {label}: a score lies outside [0, 1], "
                                 "so the bins do not hold every row. "
                                 "NOTHING WRITTEN.")

            binned = binned_auc(n_pos, n_neg)
            diff = binned - exact
            binned_1000 = binned_auc(n_pos_1000, n_neg_1000)
            diff_1000 = binned_1000 - exact
            expected = EXPECT_AUC[(market, family, outcome)]

            print(f"  {market:<7}{family:<10}{outcome:<14}{n:>7,}"
                  f"{n_pos_total:>7,}{n_firms:>7,}{exact:>9.4f}{binned:>9.4f}"
                  f"{diff:>+9.4f}{binned_1000:>9.4f}{diff_1000:>+9.5f}"
                  f"{expected:>8.4f}{on_table_col:>14.4f}  {table_col}")

            # The two stop rules, collected so every failure prints before stopping.
            if round(on_table_col, 3) != round(expected, 3):
                problems.append(f"{label}: AUC on {table_col} is "
                                f"{on_table_col:.4f}, the table has "
                                f"{expected:.4f}")
            if abs(diff) > MAX_BIN_DIFF:
                problems.append(f"{label}: binned AUC differs from exact by "
                                f"{diff:+.4f}, limit {MAX_BIN_DIFF}")
            if abs(diff_1000) > MAX_BIN_DIFF_1000:
                problems.append(f"{label}: 1,000-bin AUC differs from exact by "
                                f"{diff_1000:+.5f}, limit {MAX_BIN_DIFF_1000}")

            tables.append(pd.DataFrame({"market": market, "family": family,
                                        "outcome": outcome, "bin_lo": EDGES[:-1],
                                        "n_pos": n_pos, "n_neg": n_neg}))
            tables_1000.append(pd.DataFrame({"market": market, "family": family,
                                             "outcome": outcome,
                                             "bin_lo": EDGES_1000[:-1],
                                             "n_pos": n_pos_1000,
                                             "n_neg": n_neg_1000}))
            auc_rows.append({"market": market, "family": family,
                             "outcome": outcome, "n": n, "auc": exact})

    # ---- the three tables, built but not yet written -----------------------
    out = pd.concat(tables, ignore_index=True)
    out_1000 = pd.concat(tables_1000, ignore_index=True)
    out_auc = pd.DataFrame(auc_rows)
    for name, table, rows in [(OUT_CSV.name, out, 1200),
                              (OUT_CSV_1000.name, out_1000, 12000),
                              (OUT_AUC.name, out_auc, 12)]:
        if len(table) != rows:
            problems.append(f"{name}: {len(table):,} rows, not {rows:,}")

    # An existing 100-bin file must equal the new counts exactly. It is compared
    # as text, the form it would be written in, and is never rewritten.
    text_100 = out.to_csv(index=False)
    keep_existing = OUT_CSV.exists()
    if keep_existing and OUT_CSV.read_text(encoding="utf-8") != text_100:
        problems.append(f"{OUT_CSV.name} already exists and differs from the "
                        "counts computed now")

    # ---- stop, or write ----------------------------------------------------
    if problems:
        print("\nSTOP: a check failed.")
        for p in problems:
            print(f"  {p}")
        raise SystemExit("NOTHING WRITTEN.")
    print("\n  PASSED — all 12 cells match the earlier table to 3 decimals, and "
          f"every binned AUC is within {MAX_BIN_DIFF} (100 bins) and "
          f"{MAX_BIN_DIFF_1000} (1,000 bins) of its exact AUC\n")

    if keep_existing:
        print(f"  {OUT_CSV.name}: matches the counts computed now, character by "
              "character; left untouched")
    else:
        OUT_CSV.write_text(text_100, encoding="utf-8")
        print(f"  wrote {OUT_CSV.name}: {len(out):,} rows, "
              f"{OUT_CSV.stat().st_size:,} bytes")
    out_1000.to_csv(OUT_CSV_1000, index=False)
    print(f"  wrote {OUT_CSV_1000.name}: {len(out_1000):,} rows, "
          f"{OUT_CSV_1000.stat().st_size:,} bytes")
    out_auc.to_csv(OUT_AUC, index=False)
    print(f"  wrote {OUT_AUC.name}: {len(out_auc):,} rows, "
          f"{OUT_AUC.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
