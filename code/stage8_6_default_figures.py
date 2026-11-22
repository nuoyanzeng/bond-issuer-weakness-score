"""
Stage 8.6 — Draw the default-by-band figure and the default ROC (rf).

Reads   output/stage8_5_default_test_rows.csv
Writes  output/figures/default_by_band_rf.png and .pdf
        output/figures/roc_default_rf.png and .pdf

A re-run redraws them under the same names. All drawing lives in
`code/validation_figures.py`, which is not modified here. No other file is written.

THE BANDS. Both markets are cut at the 30th, 50th, 70th, 80th, 90th and 95th
percentiles of score_rf on the US rows, which gives 7 bands. A row is in a band if
low <= score < high, the same rule stage8_5 used for test 3's bands.

THE 95% INTERVAL. Firms are resampled with replacement 1,000 times, seed 43 for the
US and 42 for China, and the interval is the 2.5th and 97.5th percentiles of the
band's rate. Band membership stays fixed, and a resampled firm brings all of its
rows in every band. firm_id numbers differ from the earlier real identifiers, so
these draws are not the same draws as in the earlier tests, even with the same seed.

DRAWN = at least 20 distinct firms in the band. It does not consider the number of
events, so a band with very few events can still be drawn.

EVERY CHECK RUNS BEFORE ANY DRAWING, so a failure draws nothing.

HIGHER SCORE = WEAKER.

Runtime: a few seconds.

Run:  python3 code/stage8_6_default_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc as area_under_curve
from sklearn.metrics import roc_auc_score, roc_curve

import validation_figures as vf

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
FIG_DIR = OUT_DIR / "figures"
ROWS_CSV = OUT_DIR / "stage8_5_default_test_rows.csv"

COLUMNS = ["market", "firm_id", "fyear", "score_rf", "y_default"]
MARKETS = ["US", "China"]

PERCENTILES = [30, 50, 70, 80, 90, 95]
BAND_LABELS = ["0-30", "30-50", "50-70", "70-80", "80-90", "90-95", "95-100"]

N_DRAWS = 1000
SEEDS = {"US": 43, "China": 42}
MIN_FIRMS = 20     # a band with fewer firms is not drawn

# The earlier results these rows must reproduce.
EXPECT_AUC = {"US": 0.924, "China": 0.713}
EXPECT_TOTALS = {"US": (25221, 235), "China": (23545, 152)}     # firm-years, events
EXPECT_RATES = {"US": [0.03, 0.04, 0.26, 0.32, 0.79, 3.73, 11.33],
                "China": [0.00, 0.37, 0.38, 0.64, 0.54, 1.81, 6.21]}


def report(passed, text, problems):
    """Print one check, and remember it if it failed."""
    print(f"  {'PASSED' if passed else 'FAILED'}   {text}")
    if not passed:
        problems.append(text)


def bootstrap_interval(rows, seed):
    """The 2.5th and 97.5th percentiles of each band's rate, resampling firms."""
    # One line per firm, one column per band: that firm's rows and events. Summing
    # the lines of the picked firms gives the same totals as gathering their rows.
    firm_codes = pd.factorize(rows.firm_id, sort=True)[0]
    bands = rows.band.to_numpy()
    n_firms = firm_codes.max() + 1
    n_bands = len(BAND_LABELS)
    firm_rows = np.zeros((n_firms, n_bands))
    firm_events = np.zeros((n_firms, n_bands))
    np.add.at(firm_rows, (firm_codes, bands), 1)
    np.add.at(firm_events, (firm_codes, bands), rows.y_default.to_numpy())

    # Each draw picks n_firms firms with replacement. A draw with no row in a band
    # leaves that band's rate missing, and the percentiles ignore it.
    rng = np.random.default_rng(seed)
    rates = np.full((N_DRAWS, n_bands), np.nan)
    for draw in range(N_DRAWS):
        picked = rng.integers(0, n_firms, n_firms)
        n_rows = firm_rows[picked].sum(axis=0)
        n_events = firm_events[picked].sum(axis=0)
        has_rows = n_rows > 0
        rates[draw, has_rows] = n_events[has_rows] / n_rows[has_rows] * 100

    # Reported, because an interval from fewer draws should be read with care.
    n_empty = np.isnan(rates).sum()
    if n_empty:
        print(f"  NOTE: {n_empty:,} band-draws held no row and were ignored")
    return np.nanpercentile(rates, [2.5, 97.5], axis=0)


def band_table(rows, market):
    """Firm-years, firms, events, rate and interval for each band of one market."""
    lo, hi = bootstrap_interval(rows, SEEDS[market])
    table = []
    for i, label in enumerate(BAND_LABELS):
        in_band = rows[rows.band == i]
        n_rows = len(in_band)
        n_firms = in_band.firm_id.nunique()
        n_events = int(in_band.y_default.sum())
        table.append({"market": market, "band": label, "firm_years": n_rows,
                      "firms": n_firms, "events": n_events,
                      "rate": n_events / n_rows * 100 if n_rows else np.nan,
                      "lo": lo[i], "hi": hi[i], "drawn": n_firms >= MIN_FIRMS})
    return pd.DataFrame(table)


def main():
    rows = pd.read_csv(ROWS_CSV, float_precision="round_trip")
    if list(rows.columns) != COLUMNS:
        raise SystemExit(f"STOP: {ROWS_CSV.name} has columns {list(rows.columns)}, "
                         f"not {COLUMNS}. NOTHING DRAWN.")

    # ---- the bands: both cuts come from the US rows only --------------------
    us_scores = rows.loc[rows.market == "US", "score_rf"]
    cuts = np.percentile(us_scores, PERCENTILES)
    print("  US cuts: " + "   ".join(f"p{p} = {c:.6f}"
                                     for p, c in zip(PERCENTILES, cuts)))
    # np.digitize gives 0 below the first cut, 1 from the first cut up to just
    # under the second, and so on: low <= score < high, as in stage8_5.
    rows["band"] = np.digitize(rows.score_rf, cuts)

    # ---- the band table, then the ROC, for each market ---------------------
    tables = {}
    curves = {}
    for market in MARKETS:
        market_rows = rows[rows.market == market]
        tables[market] = band_table(market_rows, market)
        # drop_intermediate=False keeps one point per distinct score, plus the
        # starting point at (0, 0).
        fpr, tpr, _ = roc_curve(market_rows.y_default, market_rows.score_rf,
                                drop_intermediate=False)
        curves[market] = {"fpr": fpr, "tpr": tpr,
                          "auc": roc_auc_score(market_rows.y_default,
                                               market_rows.score_rf)}

    # ---- the full band table -----------------------------------------------
    print(f"\n  {'market':<8}{'band':<8}{'firm-years':>11}{'firms':>8}{'events':>8}"
          f"{'rate %':>9}{'lo %':>9}{'hi %':>9}{'drawn':>7}")
    for market in MARKETS:
        for r in tables[market].itertuples():
            print(f"  {r.market:<8}{r.band:<8}{r.firm_years:>11,}{r.firms:>8,}"
                  f"{r.events:>8,}{r.rate:>9.3f}{r.lo:>9.3f}{r.hi:>9.3f}"
                  f"{str(r.drawn):>7}")
    print("  Firms do not add up across bands: a firm can be in several.")

    # ---- every check, before any drawing -----------------------------------
    print("\n  CHECKS")
    problems = []
    for market in MARKETS:
        table, curve = tables[market], curves[market]

        # The AUC, at 3 decimals.
        report(round(curve["auc"], 3) == EXPECT_AUC[market],
               f"{market} AUC {curve['auc']:.4f}, expected "
               f"{EXPECT_AUC[market]:.3f}", problems)

        # The 7 bands hold every row and every event of the market.
        totals = (int(table.firm_years.sum()), int(table.events.sum()))
        report(totals == EXPECT_TOTALS[market],
               f"{market} firm-years and events in the bands {totals}, expected "
               f"{EXPECT_TOTALS[market]}", problems)

        # The rates, at 2 decimals.
        found = [round(float(rate), 2) for rate in table.rate]
        report(found == EXPECT_RATES[market],
               f"{market} rates {found}, expected {EXPECT_RATES[market]}", problems)

        # The curve must run from (0, 0) to (1, 1), have one point per distinct
        # score, and enclose the exact AUC.
        n_distinct = rows.loc[rows.market == market, "score_rf"].nunique()
        area = area_under_curve(curve["fpr"], curve["tpr"])
        report(curve["fpr"][0] == 0 and curve["tpr"][0] == 0
               and curve["fpr"][-1] == 1 and curve["tpr"][-1] == 1
               and len(curve["fpr"]) == n_distinct + 1
               and abs(area - curve["auc"]) < 1e-9,
               f"{market} ROC {len(curve['fpr']):,} points for {n_distinct:,} "
               f"distinct scores, area {area:.4f}", problems)

    if problems:
        print("\nSTOP: a check failed. NOTHING DRAWN.")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("\n  PASSED — all checks\n")

    # ---- the two figures ---------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def band_inputs(market):
        """The dict default_by_band takes for one market."""
        table = tables[market]
        return {"rate": table.rate.to_numpy(), "lo": table.lo.to_numpy(),
                "hi": table.hi.to_numpy(), "drawn": table.drawn.to_numpy()}

    stem = FIG_DIR / "default_by_band_rf"
    vf.default_by_band(BAND_LABELS, band_inputs("US"), band_inputs("China"),
                       str(stem))
    print(f"  drew {stem.name}.png and .pdf")

    stem = FIG_DIR / "roc_default_rf"
    vf.roc_default(curves["US"], curves["China"], str(stem))
    print(f"  drew {stem.name}.png and .pdf")


if __name__ == "__main__":
    main()
