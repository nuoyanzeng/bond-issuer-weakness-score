"""
Stage 8.4 — Draw the 12 next-year deterioration figures (rf).

Reads   output/stage8_3_rf_score_bins.csv        100 bins, for the density figures
        output/stage8_3_rf_score_bins_1000.csv   1,000 bins, for the ROC figures
        output/stage8_3_rf_auc.csv               the exact AUCs printed on them
Writes  output/figures/val_density_{family}_{outcome}.png and .pdf
        output/figures/val_roc_{family}_{outcome}.png and .pdf

2 families x 3 outcomes x 2 kinds of figure = 12 figures, each showing the US and
China together. A re-run redraws them under the same names.

THIS SCRIPT COMPUTES NOTHING AND CHOOSES NO STYLE. Every number comes from the
three csvs, and all drawing lives in `code/validation_figures.py`, which is not
modified here.

THE CSVS NAME OUTCOMES `intcov_t1`, `ebitdaneg_t1`, `loss_t1`; the drawing module
and the figure names use `intcov`, `ebitdaneg`, `loss`. OUTCOMES below maps one to
the other.

EVERY CHECK RUNS BEFORE ANY DRAWING, so a failure draws nothing.

The AUCs are the exact AUCs on the RAW score. For the US they are therefore not
the calibrated figures in the earlier AUC table.

HIGHER SCORE = WEAKER.

Run:  python3 code/stage8_4_validation_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import validation_figures as vf

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
FIG_DIR = OUT_DIR / "figures"

BINS_100 = OUT_DIR / "stage8_3_rf_score_bins.csv"
BINS_1000 = OUT_DIR / "stage8_3_rf_score_bins_1000.csv"
AUCS = OUT_DIR / "stage8_3_rf_auc.csv"

MARKETS = ["US", "China"]
FAMILIES = ["distress", "rating"]
# Name in the csvs -> name in the drawing module and the figure files.
OUTCOMES = {"intcov_t1": "intcov", "ebitdaneg_t1": "ebitdaneg",
            "loss_t1": "loss"}

# The 12 cells that must be present in each of the three files.
CELLS = [(m, f, o) for m in MARKETS for f in FAMILIES for o in OUTCOMES]
KEY = ["market", "family", "outcome"]


def check_cells_present(table, name, problems):
    """Each file must hold exactly the 12 cells, no more and no fewer."""
    found = set(table[KEY].drop_duplicates().itertuples(index=False, name=None))
    if found != set(CELLS):
        problems.append(f"{name}: missing cells {sorted(set(CELLS) - found)}, "
                        f"unexpected cells {sorted(found - set(CELLS))}")


def check_bins(table, name, n_bins, problems):
    """Each cell must hold every bin exactly once, from 0 upwards."""
    expected = np.arange(n_bins) / n_bins
    for cell, g in table.groupby(KEY):
        if len(g) != n_bins:
            problems.append(f"{name} {cell}: {len(g):,} bins, not {n_bins:,}")
        elif not np.allclose(np.sort(g.bin_lo.to_numpy()), expected):
            problems.append(f"{name} {cell}: bin_lo is not the expected "
                            f"{n_bins:,} equal-width bins")


def one_market(bins, aucs, market, family, outcome):
    """The dict the drawing functions take: counts from low to high, and the AUC."""
    g = bins[(bins.market == market) & (bins.family == family)
             & (bins.outcome == outcome)].sort_values("bin_lo")
    row = aucs[(aucs.market == market) & (aucs.family == family)
               & (aucs.outcome == outcome)]
    return {"n_pos": g.n_pos.to_numpy(), "n_neg": g.n_neg.to_numpy(),
            "auc": float(row.auc.iloc[0])}


def main():
    bins_100 = pd.read_csv(BINS_100)
    bins_1000 = pd.read_csv(BINS_1000)
    aucs = pd.read_csv(AUCS)

    # ---- every check, before any drawing -----------------------------------
    problems = []

    # All 12 cells in all three files, and the AUC file holds each exactly once.
    check_cells_present(bins_100, BINS_100.name, problems)
    check_cells_present(bins_1000, BINS_1000.name, problems)
    check_cells_present(aucs, AUCS.name, problems)
    if len(aucs) != len(CELLS):
        problems.append(f"{AUCS.name}: {len(aucs)} rows, not {len(CELLS)}")

    # Each cell has its 100 and its 1,000 bins.
    check_bins(bins_100, BINS_100.name, 100, problems)
    check_bins(bins_1000, BINS_1000.name, 1000, problems)

    # The positive and negative totals agree between the two bin files, and with
    # the n in the AUC file. Only cells present in all three can be compared.
    totals_100 = bins_100.groupby(KEY)[["n_pos", "n_neg"]].sum()
    totals_1000 = bins_1000.groupby(KEY)[["n_pos", "n_neg"]].sum()
    n_by_cell = aucs.set_index(KEY).n

    print(f"  {'market':<7}{'family':<10}{'outcome':<14}{'n':>8}{'positives':>11}"
          f"{'negatives':>11}{'AUC':>8}")
    for cell in CELLS:
        if (cell not in totals_100.index or cell not in totals_1000.index
                or cell not in n_by_cell.index):
            continue             # already reported as a missing cell above
        pos, neg = totals_100.loc[cell, "n_pos"], totals_100.loc[cell, "n_neg"]
        pos_1000 = totals_1000.loc[cell, "n_pos"]
        neg_1000 = totals_1000.loc[cell, "n_neg"]
        if pos != pos_1000 or neg != neg_1000:
            problems.append(f"{cell}: 100 bins hold {pos:,} / {neg:,} positive / "
                            f"negative, 1,000 bins hold {pos_1000:,} / "
                            f"{neg_1000:,}")
        if pos + neg != n_by_cell.loc[cell]:
            problems.append(f"{cell}: the bins hold {pos + neg:,} rows, the AUC "
                            f"file says n = {n_by_cell.loc[cell]:,}")
        market, family, outcome = cell
        auc_value = aucs[(aucs.market == market) & (aucs.family == family)
                         & (aucs.outcome == outcome)].auc.iloc[0]
        print(f"  {market:<7}{family:<10}{outcome:<14}{pos + neg:>8,}{pos:>11,}"
              f"{neg:>11,}{auc_value:>8.3f}")

    if problems:
        print("\nSTOP: a check failed. NO FIGURE DRAWN.")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print("\n  PASSED — 12 cells in all three files; 100 and 1,000 bins in each; "
          "totals agree\n")

    # ---- the 12 figures ----------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        for csv_outcome, outcome in OUTCOMES.items():
            # Density: the 100-bin counts.
            stem = FIG_DIR / f"val_density_{family}_{outcome}"
            vf.density(one_market(bins_100, aucs, "US", family, csv_outcome),
                       one_market(bins_100, aucs, "China", family, csv_outcome),
                       family, outcome, str(stem))
            print(f"  drew {stem.name}.png and .pdf")

            # ROC: the 1,000-bin counts.
            stem = FIG_DIR / f"val_roc_{family}_{outcome}"
            vf.roc(one_market(bins_1000, aucs, "US", family, csv_outcome),
                   one_market(bins_1000, aucs, "China", family, csv_outcome),
                   family, outcome, str(stem))
            print(f"  drew {stem.name}.png and .pdf")


if __name__ == "__main__":
    main()
