"""
Report exhibits — every table and figure of the methodology report, each with an
Excel file of its data, for replication.

READ-ONLY ON THE PROJECT. It fits no model (no estimator's fit method is called
anywhere in it), changes no existing file, and writes only into the NEW folder
output/report_exhibits/. If that folder already exists the script stops at once
and writes nothing, so it can never overwrite. `data_raw/` is read, never written.
Python's bytecode cache is switched off, so not even a __pycache__ file appears.

WHAT IT WRITES, for each exhibit
    a table   <name>.xlsx
    a figure  <name>.png, <name>.pdf, and <name>.xlsx holding the numbers plotted
Every .xlsx has two sheets: "data" (the numbers) and "notes" (source files, sample,
period, unit of observation, and a definition of every column).

EVERY EXHIBIT IS COMPUTED, AND EVERY CHECK RUN, BEFORE ANYTHING IS WRITTEN. Each
check prints PASSED or FAILED. An exhibit with a failed check is not written; the
others are. An error in the code stops the run with nothing written.

HOW A NUMBER IS COMPARED WITH AN EXPECTED VALUE. The value is rounded to one more
decimal than the expected value has, and must then lie within half a unit of the
expected value's last decimal. This accepts numbers that were rounded twice, e.g.
0.76149 -> 0.7615 -> .762. Counts must match exactly.

THE SAMPLES ARE THE ONES THAT PRODUCED EACH NUMBER BEFORE. Nothing is rebuilt in a
new way. Where an earlier script can be imported, its own functions are called:
    stage5_2_results_summary.py       SIC divisions (logic of passes 1, 3-6 copied)
    check_china_score_validity.py     outcomes, the CSMAR readers, the rating notches
    stage7_2_inflation_descriptive.py rating-inflation panels, bootstraps, gap logic
    stage8_6_default_figures.py       default-by-band table and its bootstrap
The scratch scripts of 2026-11-04 (tests123.py, score_desc.py, within_group2.py)
are not project files, so their logic is copied here and their outputs are
reproduced as checks.

Figures are drawn only by code/validation_figures.py and code/inflation_figures.py,
which this script does not modify.

HIGHER SCORE = WEAKER throughout.

Runtime: about 3 to 5 minutes, untimed. Most of it is reading the CSMAR statement
files for the deterioration outcomes (as stage8_3 does) and the stage7_2 bootstraps.
Memory: the CSMAR files are read a few columns at a time, as stage8_3 and
check_china_score_validity.py already do on this machine.

Run:  python3 code/report_exhibits.py
"""

import sys

# Set before any project module is imported, so no __pycache__ is written.
sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc as area_under_curve
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "code"))

import check_china_score_validity as cv          # noqa: E402
import inflation_figures as ifg                  # noqa: E402
import stage5_2_results_summary as s52           # noqa: E402
import stage7_2_inflation_descriptive as s72     # noqa: E402
import stage8_6_default_figures as s86           # noqa: E402
import validation_figures as vf                  # noqa: E402

EXHIBIT_DIR = PROJECT_DIR / "output" / "report_exhibits"

MODELS = ["xgb", "rf", "enet"]
KEY = ["issuer_id", "gvkey", "fyear"]
INDICATORS = ["OENEG", "INTWO"]
MIN_FIRMS = 20

# ---------------------------------------------------------------------------
# INPUT FILES, relative to the project folder, so the notes can name them
# ---------------------------------------------------------------------------
PANEL_L3 = "output/stage4_0_distress_L3_K12.parquet"
META_XGB = "output/stage5_0_model_L3_K12_base_xgb_meta.json"
COEF_D = "output/stage5_0_model_L3_K12_base_enet_coefficients.csv"
COEF_R = "output/stage5_0_model_rating_L3_K12_base_enet_coefficients.csv"
EVENTS = "output/stage3_0_distress_events.parquet"
ANNUAL = "output/compustat_annual_clean.parquet"
NAMES = "data_raw/us/comp_names.parquet"
V4 = "output/v4_results_summary.md"
CN_LABELS = "output/stage8_2_china_labels.parquet"
US_DIST_FY = "output/stage8_2_us_distress_firmyear.parquet"
US_RATE_FY = "output/stage8_2_us_rating_firmyear.parquet"
BINS_100 = "output/stage8_3_rf_score_bins.csv"
BINS_1000 = "output/stage8_3_rf_score_bins_1000.csv"
AUC_RF = "output/stage8_3_rf_auc.csv"
ROWS_85 = "output/stage8_5_default_test_rows.csv"
CN_PANEL = "output/stage6_4_china_score_panel.parquet"
SP_DIST = "output/stage5_1_score_panel_L3_K12.parquet"
SP_RATE = "output/stage5_1_score_panel_rating_L3_K12.parquet"
US_RATING_PANEL = "output/stage7_0_us_rating_panel.parquet"
CN_RATING_PANEL = "output/stage7_1_china_rating_panel.parquet"
FISD_DEFAULT = "data_raw/us/fisd_issue_default.parquet"
FISD_MERGED = "data_raw/us/fisd_mergedissue.parquet"
CSMAR_BOND = "data_raw/china/债券基本情况表.csv"
CSMAR_RATING = "data_raw/china/债券及主体评级情况表.csv"
CSMAR_STATEMENTS = ["data_raw/china/" + f for f in cv.CN_CODES]
WITHIN_GROUP_LOG = ("scratchpad within_group2.log of 2026-11-04 (session "
                    "75e5b9d2), not a project file")


def d_results(model, lag="L3"):
    return f"output/stage4_1_results_{lag}_K12_{model}.json"


def d_scores(model, lag="L3"):
    return f"output/stage4_1_scores_{lag}_K12_{model}.parquet"


def r_results(model):
    return f"output/stage4_2_results_{model}.json"


def r_scores(model):
    return f"output/stage4_2_scores_{model}.parquet"


def P(relative):
    """The full path of a project file."""
    return PROJECT_DIR / relative


def read_json(relative):
    return json.loads(P(relative).read_text())


# ---------------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------------
def close(value, expected, decimals):
    """Round to one more decimal than `expected`; then within half a unit."""
    if value is None or pd.isna(value):
        return False
    return (abs(round(float(value), decimals + 1) - expected)
            <= 0.5 * 10 ** -decimals + 1e-9)


def chk(checks, passed, text):
    """Print one check and remember its result."""
    passed = bool(passed)
    checks.append((passed, text))
    print(f"    {'PASSED' if passed else 'FAILED'}   {text}")
    return passed


def header(name):
    print("\n" + "-" * 92)
    print(f"  {name}")
    print("-" * 92)


def exhibit(name, kind, data, columns, notes, checks, draw=None):
    """One exhibit: its data, column definitions, notes, checks, and drawing."""
    return {"name": name, "kind": kind, "data": data, "columns": columns,
            "notes": notes, "checks": checks, "draw": draw}


# ---------------------------------------------------------------------------
# v4_results_summary.md, read as tables, for checks only
# ---------------------------------------------------------------------------
def v4_blocks():
    """Every table of the summary, keyed by the heading or bold line above it.

    Rows are lists of cell texts, backticks removed. The header row is kept; rows
    are looked up by their first cell.
    """
    blocks, current = {}, None
    for line in P(V4).read_text().splitlines():
        if line.startswith("## ") or line.startswith("**"):
            current = line
            blocks.setdefault(current, [])
        elif line.startswith("|") and not line.startswith("|---") and current:
            cells = [c.strip().replace("`", "") for c in line.strip("|").split("|")]
            blocks[current].append(cells)
    return blocks


def v4_block(blocks, prefix):
    """The rows under the first heading that starts with `prefix`."""
    for key, rows in blocks.items():
        if key.startswith(prefix):
            return {r[0]: r for r in rows}
    return {}


def same_as_cell(value, cell):
    """Does `value` equal the number printed in a summary cell, at its decimals?"""
    text = cell.replace(",", "").replace("%", "").replace("x", "").strip()
    if text in ("—", ""):
        return pd.isna(value)
    decimals = len(text.split(".")[1]) if "." in text else 0
    target = float(text)
    if pd.isna(value):
        return False
    return abs(float(value) - target) <= 0.5 * 10 ** -decimals + 1e-9


# ---------------------------------------------------------------------------
# LOADING — the US fitted sample, exactly as stage5_2 builds it
# ---------------------------------------------------------------------------
def load_us():
    feats = read_json(META_XGB)["features_in_order"]
    panel = pd.read_parquet(P(PANEL_L3))
    dsc = {m: pd.read_parquet(P(d_scores(m))) for m in MODELS}
    dres = {m: read_json(d_results(m)) for m in MODELS}
    # The fitted rows are the scored rows: stage4_1 dropped rows with a remaining
    # NA before fitting. This is stage5_2's own join.
    fit = panel.merge(dsc["xgb"][KEY + ["score_calibrated"]], on=KEY, how="inner")

    id_cols = KEY + ["datadate", "date_avail", "y"]
    if len(feats) != 27:
        raise SystemExit(f"STOP: {META_XGB} lists {len(feats)} features, not 27.")
    if [c for c in panel.columns if c not in id_cols] != feats:
        raise SystemExit(f"STOP: {PANEL_L3}'s feature columns are not the "
                         "models' list, in order.")
    if len(fit) != dres["xgb"]["n_rows"] or int(fit.y.sum()) != dres["xgb"]["n_events"]:
        raise SystemExit(f"STOP: {len(fit):,} fitted rows joined, "
                         f"{dres['xgb']['n_rows']:,} recorded.")
    return {"feats": feats, "panel": panel, "fit": fit, "dsc": dsc, "dres": dres,
            "l0res": {m: read_json(d_results(m, "L0")) for m in MODELS},
            "l0sc": {m: pd.read_parquet(P(d_scores(m, "L0"))) for m in MODELS},
            "rres": {m: read_json(r_results(m)) for m in MODELS},
            "rsc": {m: pd.read_parquet(P(r_scores(m))) for m in MODELS},
            "coef_d": pd.read_csv(P(COEF_D), float_precision="round_trip"),
            "coef_r": pd.read_csv(P(COEF_R), float_precision="round_trip")}


US_FIT_SAMPLE = ("US fitted distress sample, L3, K12, 27 features: 95,370 rows, "
                 "5,545 gvkeys, 7,486 issuers, 1,164 default events (the rows "
                 "stage4_1 fitted after dropping 3,038 rows with a remaining NA)")
US_FIT_UNIT = ("issuer x gvkey x fiscal-year statement row (issuer_id, gvkey, "
               "fyear); 29,501 of the 95,370 rows repeat a gvkey x datadate "
               "statement under another issuer")


def rank_table(results, feats):
    """Section 1's ranks: each model ranks |importance|, rank 1 = largest."""
    ranks = {}
    for m in MODELS:
        values = pd.Series({f: abs(float(results[m]["importances"][f]))
                            for f in feats})
        ranks[m] = values.rank(ascending=False, method="min").astype(int)
    table = pd.DataFrame(ranks)
    table["mean_rank"] = table[MODELS].mean(axis=1)
    table["spread"] = table[MODELS].max(axis=1) - table[MODELS].min(axis=1)
    return table.sort_values("mean_rank", kind="mergesort")


def coefficients(coef):
    """Signed elastic-net coefficients by feature, intercept removed."""
    return coef[coef.feature != "_intercept_"].set_index("feature").coefficient


# ===========================================================================
# 1. tab_us_performance
# ===========================================================================
def ex_us_performance(us):
    header("tab_us_performance")
    checks, rows = [], []
    for m in MODELS:
        r3, r0 = us["dres"][m], us["l0res"][m]
        rows.append({
            "model": m,
            "auc_pr_L3": r3["metrics"]["AUC-PR"],
            "auc_roc_L3": r3["metrics"]["AUC-ROC"],
            "top10_capture_L3_pct": r3["metrics"]["rec@10%"] * 100,
            "auc_pr_L0": r0["metrics"]["AUC-PR"],
            "auc_roc_L0": r0["metrics"]["AUC-ROC"],
            "rows_L3": r3["n_rows"], "events_L3": r3["n_events"],
            "gvkeys_L3": us["dsc"][m].gvkey.nunique(), "groups_L3": r3["n_groups"],
            "rows_L0": r0["n_rows"], "events_L0": r0["n_events"],
            "gvkeys_L0": us["l0sc"][m].gvkey.nunique(), "groups_L0": r0["n_groups"]})
    data = pd.DataFrame(rows)

    expect = {"xgb": (0.1450, 0.8656, 65.8, 0.1702, 0.8753),
              "rf": (0.1385, 0.8764, 68.0, 0.1675, 0.8790),
              "enet": (0.1024, 0.8540, 63.2, 0.1314, 0.8659)}
    for r in data.itertuples():
        e = expect[r.model]
        got = (r.auc_pr_L3, r.auc_roc_L3, r.top10_capture_L3_pct, r.auc_pr_L0,
               r.auc_roc_L0)
        ok = all(close(g, x, d) for g, x, d in zip(got, e, (4, 4, 1, 4, 4)))
        chk(checks, ok, f"{r.model}: {got[0]:.4f} {got[1]:.4f} {got[2]:.1f}% "
            f"{got[3]:.4f} {got[4]:.4f}; expected {e[0]:.4f} {e[1]:.4f} "
            f"{e[2]:.1f}% {e[3]:.4f} {e[4]:.4f}")
    chk(checks, all(us["dres"][m]["status"] == "OK" and
                    us["l0res"][m]["status"] == "OK" for m in MODELS),
        "all six results files have status OK")
    chk(checks, (data.rows_L3 == 95370).all() and (data.events_L3 == 1164).all(),
        "L3: 95,370 rows and 1,164 events for all three models")

    columns = {
        "model": "xgb = XGBoost, rf = random forest, enet = elastic-net logistic",
        "auc_pr_L3": "area under the precision-recall curve, cross-fitted "
                     "out-of-fold scores, L3 (date_avail = fiscal year-end + 3 months)",
        "auc_roc_L3": "area under the ROC curve, same scores, L3",
        "top10_capture_L3_pct": "share of the default events whose score is in "
                                "the top 10% of scores, %, L3 (rec@10% in the "
                                "results file)",
        "auc_pr_L0": "area under the precision-recall curve, L0 (date_avail = "
                     "fiscal year-end), robustness setting",
        "auc_roc_L0": "area under the ROC curve, L0",
        "rows_L3": "rows fitted and scored, L3", "events_L3": "default events, L3",
        "gvkeys_L3": "distinct gvkeys (firms), L3, counted from the scores file",
        "groups_L3": "cross-fitting groups (connected components of issuer_id "
                     "and gvkey), L3",
        "rows_L0": "rows fitted and scored, L0", "events_L0": "default events, L0",
        "gvkeys_L0": "distinct gvkeys, L0", "groups_L0": "cross-fitting groups, L0"}
    notes = {
        "source files": [d_results(m, lag) for lag in ("L3", "L0") for m in MODELS]
        + [d_scores(m, lag) for lag in ("L3", "L0") for m in MODELS],
        "sample": US_FIT_SAMPLE + "; L0: 94,285 rows, 1,163 events",
        "period": "fiscal years 1992-2025 as present in the fitted panel",
        "unit of observation": US_FIT_UNIT,
        "method": "Metrics are read from the saved stage4_1 results files: 5 "
                  "grouped folds, seed 42, every score out-of-fold. No model is "
                  "refitted. The ranking metrics are the same on the raw and the "
                  "calibrated score, because calibration is monotone within a fold."}
    return exhibit("tab_us_performance", "table", data, columns, notes, checks)


# ===========================================================================
# 2. tab_leading_features
# ===========================================================================
def ex_leading_features(us):
    header("tab_leading_features")
    checks = []
    ranks = rank_table(us["dres"], us["feats"])
    coef = coefficients(us["coef_d"])
    top = ranks[(ranks.xgb <= 10) & (ranks.rf <= 10)].copy()
    top["sum"] = top.xgb + top.rf
    top = top.sort_values(["sum", "xgb"], kind="mergesort")
    data = pd.DataFrame({"feature": top.index, "xgb_gain_rank": top.xgb.values,
                         "rf_impurity_rank": top.rf.values,
                         "enet_coefficient": coef.reindex(top.index).values})

    expect = {"PM": (1, 1, -0.4377), "roaq": (2, 2, -0.4524),
              "TLTA": (3, 3, 0.6153), "ocf_at": (4, 8, -0.2237),
              "roavol": (8, 6, 0.5831), "Cash": (7, 9, -0.1067),
              "cashdebt": (10, 4, 0.0)}
    chk(checks, set(data.feature) == set(expect),
        f"the features in both top tens are {sorted(data.feature)}; expected "
        f"{sorted(expect)}")
    for r in data.itertuples():
        if r.feature not in expect:
            continue
        e = expect[r.feature]
        chk(checks, r.xgb_gain_rank == e[0] and r.rf_impurity_rank == e[1]
            and close(r.enet_coefficient, e[2], 4),
            f"{r.feature}: {r.xgb_gain_rank} {r.rf_impurity_rank} "
            f"{r.enet_coefficient:+.4f}; expected {e[0]} {e[1]} {e[2]:+.4f}")

    columns = {
        "feature": "feature name; every feature enters as its within-fiscal-year "
                   "rank scaled to [-0.5, +0.5]",
        "xgb_gain_rank": "rank of the feature's XGBoost importance (normalised "
                         "gain), 1 = largest, of 27",
        "rf_impurity_rank": "rank of the feature's random-forest importance (mean "
                            "impurity decrease), 1 = largest, of 27",
        "enet_coefficient": "signed elastic-net coefficient, per one standard "
                            "deviation of the feature (the features are "
                            "standardised before fitting); positive = a higher "
                            "value raises the default score; 0 = removed by the "
                            "penalty"}
    notes = {
        "source files": [d_results("xgb"), d_results("rf"), d_results("enet"),
                         COEF_D],
        "sample": US_FIT_SAMPLE, "period": "fiscal years 1992-2025",
        "unit of observation": "feature; the importances come from fits on "
                               + US_FIT_UNIT,
        "method": "Features ranked in the top ten of BOTH xgb and rf, as in "
                  "Section 1 of v4_results_summary.md (ranks with method 'min'). "
                  "Importances and coefficients come from one full-sample fit "
                  "each, not from the cross-fitted folds. Rows are ordered by "
                  "xgb rank + rf rank, then xgb rank.",
        "caveats": "Gain, impurity decrease and coefficient size are different "
                   "quantities, so only ranks and signs are compared (D-56)."}
    return exhibit("tab_leading_features", "table", data, columns, notes, checks)


# ===========================================================================
# 3. tab_feature_deciles   (pass 3 of stage5_2)
# ===========================================================================
def decile_table(fit, f):
    """Realised default rate by decile of one feature, stage5_2 pass 3."""
    d = fit[[f, "y"]].dropna(subset=[f]).copy()
    d["dec"] = pd.qcut(d[f], 10, labels=False, duplicates="drop") + 1
    g = d.groupby("dec").agg(n=("y", "size"), events=("y", "sum")).reset_index()
    g["rate_pct"] = g.events / g.n * 100
    return g


def ex_feature_deciles(us):
    header("tab_feature_deciles")
    checks, parts = [], []
    for f in ["roaq", "PM", "cashdebt", "ocf_at", "TLTA"]:
        g = decile_table(us["fit"], f)
        g.insert(0, "feature", f)
        parts.append(g.rename(columns={"dec": "decile"}))
    data = pd.concat(parts, ignore_index=True)

    expect = {"roaq": (6.08, 2.47, 0.45, 0.44), "PM": (4.98, 3.73, 0.26, 0.37),
              "cashdebt": (5.21, 2.24, 0.64, 0.27),
              "ocf_at": (4.28, 1.46, 0.73, 0.62), "TLTA": (0.35, 0.51, 0.47, 5.23)}
    for f, e in expect.items():
        g = data[data.feature == f].set_index("decile").rate_pct
        got = [g.get(k) for k in (1, 2, 5, 10)]
        chk(checks, len(g) == 10 and all(close(a, b, 2) for a, b in zip(got, e)),
            f"{f} D1 D2 D5 D10: " + " ".join(f"{v:.2f}" for v in got)
            + f"; expected " + " ".join(f"{v:.2f}" for v in e))
    chk(checks, (data.groupby("feature").n.sum() == 95370).all()
        and (data.groupby("feature").events.sum() == 1164).all(),
        "each feature's ten deciles hold all 95,370 rows and 1,164 events")

    columns = {"feature": "feature name",
               "decile": "decile of the feature on the fitted rows, 1 = lowest "
                         "value, 10 = highest (pandas qcut)",
               "n": "rows in the decile", "events": "default events in the decile",
               "rate_pct": "events / n, %"}
    notes = {"source files": [PANEL_L3, d_scores("xgb")], "sample": US_FIT_SAMPLE,
             "period": "fiscal years 1992-2025", "unit of observation": US_FIT_UNIT,
             "method": "Pass 3 of stage5_2_results_summary.py: deciles are cut on "
                       "the feature's within-year rank as the models received "
                       "it, pooled over all years. Univariate: no other feature "
                       "is held fixed."}
    return exhibit("tab_feature_deciles", "table", data, columns, notes, checks)


# ===========================================================================
# 4. fig_early_warning and tab_early_warning_all   (pass 4 of stage5_2)
# ===========================================================================
def early_warning_table(dsc):
    ev = pd.read_parquet(P(EVENTS))
    ev["ev_year"] = pd.to_datetime(ev.first_distress_date).dt.year
    rows = []
    for m in MODELS:
        s = dsc[m].merge(ev[["issuer_id", "ev_year"]], on="issuer_id", how="left")
        never = s[s.ev_year.isna()]
        for h in range(1, 6):
            g = s[s.ev_year.notna() & (s.fyear == s.ev_year - h)]
            # Never-distressed rows in the same fiscal years, so the baseline
            # moves with the sample.
            base = never[never.fyear.isin(g.fyear.unique())]
            p90 = base.score_calibrated.quantile(0.90)
            rows.append({"model": m, "horizon": f"t-{h}", "firm_years": len(g),
                         "issuers": g.issuer_id.nunique(),
                         "mean_score": g.score_calibrated.mean(),
                         "never_mean_same_years": base.score_calibrated.mean(),
                         "ratio": g.score_calibrated.mean()
                         / base.score_calibrated.mean(),
                         "never_p90_same_years": p90,
                         "share_above_never_p90_pct":
                             (g.score_calibrated > p90).mean() * 100})
        rows.append({"model": m, "horizon": "never distressed",
                     "firm_years": len(never), "issuers": never.issuer_id.nunique(),
                     "mean_score": never.score_calibrated.mean(),
                     "never_mean_same_years": np.nan, "ratio": np.nan,
                     "never_p90_same_years": np.nan,
                     "share_above_never_p90_pct": np.nan})
    return pd.DataFrame(rows)


def ex_early_warning(us, blocks):
    header("fig_early_warning and tab_early_warning_all")
    checks = []
    data = early_warning_table(us["dsc"])

    rf = data[(data.model == "rf") & (data.horizon != "never distressed")]
    rf = rf.set_index("horizon").share_above_never_p90_pct
    shares = [rf[f"t-{h}"] for h in (5, 4, 3, 2, 1)]
    expect = [35.7, 41.0, 49.1, 64.6, 78.8]
    chk(checks, all(close(a, b, 1) for a, b in zip(shares, expect)),
        "rf share above the never-distressed p90, t-5 to t-1: "
        + " ".join(f"{v:.1f}" for v in shares) + "; expected "
        + " ".join(f"{v:.1f}" for v in expect))

    # Every row of all three models against Section 4 of the summary.
    for m in MODELS:
        ref = v4_block(blocks, f"**{m}**")
        bad = []
        for r in data[data.model == m].itertuples():
            cells = ref.get(r.horizon)
            if cells is None:
                bad.append(r.horizon)
                continue
            got = [r.firm_years, r.issuers, r.mean_score]
            if r.horizon != "never distressed":
                got += [r.never_mean_same_years, r.ratio, r.share_above_never_p90_pct]
            if not all(same_as_cell(g, c) for g, c in zip(got, cells[1:])):
                bad.append(r.horizon)
        chk(checks, ref and not bad, f"{m}: all six rows equal Section 4 of "
            f"{V4}" + (f" — differs at {bad}" if bad else ""))

    columns = {
        "model": "xgb, rf or enet",
        "horizon": "t-h = fiscal year h years before the fiscal year of the "
                   "issuer's first distress date; 'never distressed' = issuers "
                   "with no distress event",
        "firm_years": "rows at that horizon", "issuers": "distinct issuer_id",
        "mean_score": "mean CALIBRATED out-of-fold default score",
        "never_mean_same_years": "mean score of never-distressed rows in the "
                                 "same fiscal years as that horizon's rows",
        "ratio": "mean_score / never_mean_same_years",
        "never_p90_same_years": "90th percentile of the never-distressed scores "
                                "in those fiscal years",
        "share_above_never_p90_pct": "share of the horizon's rows scoring above "
                                     "that 90th percentile, %; 10% by "
                                     "construction for never-distressed rows"}
    notes = {"source files": [d_scores(m) for m in MODELS] + [EVENTS],
             "sample": US_FIT_SAMPLE, "period": "fiscal years 1992-2025",
             "unit of observation": US_FIT_UNIT,
             "method": "Pass 4 of stage5_2_results_summary.py, unchanged. Scores "
                       "are the calibrated out-of-fold scores of the summary."}
    fig_notes = dict(notes, figure="validation_figures.early_warning(shares), "
                     "shares = the rf rows t-5 to t-1 of "
                     "share_above_never_p90_pct; dashed line at 10")

    def draw(stem):
        vf.early_warning(shares, stem)

    return [exhibit("fig_early_warning", "figure", data, columns, fig_notes,
                    checks, draw),
            exhibit("tab_early_warning_all", "table", data, columns, notes, checks)]


# ===========================================================================
# THE stage8_2 PANELS — the samples of the 2026-11-04 tests
# ===========================================================================
def load_stage8_2():
    cn = pd.read_parquet(P(CN_LABELS))
    ud = pd.read_parquet(P(US_DIST_FY))
    ur = pd.read_parquet(P(US_RATE_FY))
    # Fiscal 2003-2024, China scored rows only: score_desc.py, stage8_3 and
    # within_group2.py. Index reset so outcomes and scores share one index.
    s = {"cn": cn[cn.scored & cn.fyear.between(2003, 2024)].reset_index(drop=True),
         "ud": ud[ud.fyear.between(2003, 2024)].reset_index(drop=True),
         "ur": ur[ur.fyear.between(2003, 2024)].reset_index(drop=True)}
    for k, n in [("cn", 24557), ("ud", 46031), ("ur", 27600)]:
        if len(s[k]) != n:
            raise SystemExit(f"STOP: the {k} 2003-2024 panel has {len(s[k]):,} "
                             f"rows, not {n:,}. It is not the earlier sample.")
    # Fiscal 2013-2024, the default test of tests123.py.
    s["ud13"] = ud[ud.fyear.between(2013, 2024)].reset_index(drop=True)
    s["cn13"] = cn[cn.scored & cn.in_default.astype(bool)
                   & cn.fyear.between(2013, 2024)].reset_index(drop=True)
    return s


def score_sets(s):
    """(market, score, model, panel, firm column, score column) for raw scores."""
    out = []
    for m in MODELS:
        out.append(("China", "default", m, s["cn"], "Orgid", f"score_distress_{m}"))
    for m in MODELS:
        out.append(("China", "rating", m, s["cn"], "Orgid", f"score_rating_{m}"))
    for m in MODELS:
        out.append(("US", "default", m, s["ud"], "gvkey", f"raw_{m}"))
    for m in MODELS:
        out.append(("US", "rating", m, s["ur"], "gvkey", f"raw_{m}"))
    return out


PANEL_SAMPLE = ("fiscal 2003-2024: China, scored issuer-years of "
                "stage8_2_china_labels, 24,557 rows; US default score, "
                "stage8_2_us_distress_firmyear, 46,031 rows; US rating score, "
                "stage8_2_us_rating_firmyear, 27,600 rows")
PANEL_UNIT = ("issuer-year: China Orgid x fyear; US gvkey x fyear (stage8_2 "
              "collapsed the issuer rows to one per gvkey x fyear)")
RAW_NOTE = ("RAW scores on both sides: the US cross-fitted out-of-fold raw "
            "score (raw_*), the Chinese full-sample raw score (score_*, D-67). "
            "Neither is a probability.")


# ===========================================================================
# 5. tab_score_summary
# ===========================================================================
def ex_score_summary(s):
    header("tab_score_summary")
    checks, rows = [], []
    for market, score, m, panel, firm, col in score_sets(s):
        v = panel[col].dropna()
        rows.append({"market": market, "score": score, "model": m, "n": len(v),
                     "firms": panel.loc[v.index, firm].nunique(),
                     "mean": v.mean(), "median": v.median(), "sd": v.std(),
                     "skewness": v.skew(), "p10": v.quantile(0.10),
                     "p90": v.quantile(0.90)})
    data = pd.DataFrame(rows)

    med = data[data.score == "default"].set_index(["market", "model"])["median"]
    for market, e in [("China", (0.087, 0.061, 0.235)),
                      ("US", (0.051, 0.035, 0.239))]:
        got = [med[(market, m)] for m in MODELS]
        chk(checks, all(close(a, b, 3) for a, b in zip(got, e)),
            f"median default score, {market}, xgb rf enet: "
            + " ".join(f"{v:.3f}" for v in got) + "; expected "
            + " ".join(f"{v:.3f}" for v in e))
    chk(checks, list(data.n) == [24557] * 6 + [46031] * 3 + [27600] * 3,
        "n = 24,557 (China, both scores), 46,031 (US default), 27,600 (US rating)")

    columns = {"market": "US or China", "score": "default weakness score "
               "(fitted on US defaults) or rating weakness score (fitted on the "
               "US speculative-grade label)", "model": "xgb, rf or enet",
               "n": "issuer-years with the score", "firms": "distinct firms",
               "mean": "mean raw score", "median": "median raw score",
               "sd": "standard deviation (ddof = 1)",
               "skewness": "bias-corrected Fisher-Pearson skewness (pandas skew)",
               "p10": "10th percentile (linear interpolation)",
               "p90": "90th percentile (linear interpolation)"}
    notes = {"source files": [CN_LABELS, US_DIST_FY, US_RATE_FY],
             "sample": PANEL_SAMPLE, "period": "fiscal years 2003-2024",
             "unit of observation": PANEL_UNIT,
             "method": "The samples of the 2026-11-04 score-distribution summary "
                       "(score_desc.py). " + RAW_NOTE}
    return exhibit("tab_score_summary", "table", data, columns, notes, checks)


# ===========================================================================
# 6. fig_score_distribution
# ===========================================================================
EDGES_100 = np.arange(101) / 100


def ex_score_distribution(s):
    header("fig_score_distribution")
    checks, rows = [], []
    shares = {}
    xmax = {"default": 0.5, "rating": 1.0}       # validation_figures defaults
    sigma = {"default": 1.5, "rating": 2.5}
    mid = EDGES_100[:-1] + 0.005
    for market, score, m, panel, firm, col in score_sets(s):
        v = panel[col].dropna().to_numpy()
        # np.histogram closes only the last bin on the right, so 1.0 is kept.
        counts, _ = np.histogram(v, bins=EDGES_100)
        chk(checks, counts.sum() == len(v),
            f"{market} {score} {m}: all {len(v):,} scores lie in [0, 1]")
        share = counts / counts.sum()
        shares[(market, score, m)] = share
        # The curve the figure draws, from the module's own smoothing.
        plotted = (m == "rf") & (mid <= xmax[score])
        dens = vf._smooth(share, sigma[score]) if m == "rf" else np.full(100, np.nan)
        rows.append(pd.DataFrame({
            "market": market, "score": score, "model": m,
            "bin_lo": EDGES_100[:-1], "bin_hi": EDGES_100[1:], "count": counts,
            "share": share, "plotted": plotted,
            "density_plotted": np.where(plotted, dens, np.nan)}))
    data = pd.concat(rows, ignore_index=True)
    chk(checks, np.allclose(data.groupby(["market", "score", "model"]).share.sum(), 1),
        "each score's 100 shares sum to 1")

    us_in = {"distress": shares[("US", "default", "rf")],
             "rating": shares[("US", "rating", "rf")]}
    cn_in = {"distress": shares[("China", "default", "rf")],
             "rating": shares[("China", "rating", "rf")]}

    def draw(stem):
        vf.score_distribution(us_in, cn_in, stem)

    columns = {"market": "US or China", "score": "default or rating weakness score",
               "model": "xgb, rf or enet",
               "bin_lo": "lower edge of the 0.01-wide bin of the raw score",
               "bin_hi": "upper edge; the last bin [0.99, 1.00] is closed",
               "count": "issuer-years in the bin", "share": "count / all "
               "issuer-years of that market and score (sums to 1)",
               "plotted": "True where the figure draws this bin: rf only, up to "
                          "x = 0.5 for the default score and 1.0 for the rating "
                          "score",
               "density_plotted": "the curve drawn: the shares smoothed by "
                                  "validation_figures._smooth (Gaussian, width "
                                  "1.5 bins default, 2.5 bins rating, reflected "
                                  "at 0 and 1), as a density on [0, 1]"}
    notes = {"source files": [CN_LABELS, US_DIST_FY, US_RATE_FY],
             "sample": PANEL_SAMPLE, "period": "fiscal years 2003-2024",
             "unit of observation": PANEL_UNIT, "method": RAW_NOTE,
             "figure": "validation_figures.score_distribution(us, cn), rf shares, "
                       "default panel then rating panel"}
    return exhibit("fig_score_distribution", "figure", data, columns, notes,
                   checks, draw)


# ===========================================================================
# OUTCOMES — check_china_score_validity's construction, as stage8_3 imports it
# ===========================================================================
OUTCOMES = {"intcov_t1": ("intcov", "interest cover below 1x at t+1"),
            "ebitdaneg_t1": ("ebitdaneg", "EBITDA negative at t+1"),
            "loss_t1": ("loss", "net loss at t+1")}


def build_outcomes(s):
    print("\n  Reading the CSMAR and Compustat items for the outcomes (the slow "
          "part, about a minute) ...")
    bond_master = cv.csmar("债券基本情况表.csv",
                           ["Liscd", "Sctcd", "Orgid"]).drop_duplicates()
    cn_items = cv.china_items(bond_master)
    us_items = cv.us_items()

    def outcomes_for(panel, firm_col, items):
        return cv.outcomes({o: cv.lookup(panel, firm_col, items, o)
                            for o in (0, 1, 2, 3)})

    ys = {"cn": outcomes_for(s["cn"], "Orgid", cn_items),
          "ud": outcomes_for(s["ud"], "gvkey", us_items),
          "ur": outcomes_for(s["ur"], "gvkey", us_items)}
    print("  ... done")
    return bond_master, cn_items, ys


OUTCOME_NOTE = ("Outcomes from check_china_score_validity.py (outcomes, lookup, "
                "china_items, us_items), as stage8_3 imports them: raw OBSERVED "
                "items for the firm at fiscal year t+1, CSMAR consolidated 12-31 "
                "statements for China and compustat_annual_clean for the US, never "
                "imputed. interest cover = EBIT / interest expense (CSMAR "
                "B001211100, Compustat xint), NA where interest expense <= 0; "
                "EBITDA = EBIT + depreciation and amortisation; loss = net income "
                "< 0. A row whose t+1 outcome is unobserved is left out of that "
                "cell, so n differs by outcome.")


def det_cells(s, ys):
    """(market, score, panel, outcomes, firm, raw column, table column)."""
    out = []
    for m in MODELS:
        out.append(("China", "default", m, s["cn"], ys["cn"], "Orgid",
                    f"score_distress_{m}", f"score_distress_{m}"))
    for m in MODELS:
        out.append(("China", "rating", m, s["cn"], ys["cn"], "Orgid",
                    f"score_rating_{m}", f"score_rating_{m}"))
    for m in MODELS:
        out.append(("US", "default", m, s["ud"], ys["ud"], "gvkey", f"raw_{m}",
                    f"score_{m}"))
    for m in MODELS:
        out.append(("US", "rating", m, s["ur"], ys["ur"], "gvkey", f"raw_{m}",
                    f"score_{m}"))
    return out


# ===========================================================================
# 7. tab_deterioration_auc
# ===========================================================================
def ex_deterioration_auc(s, ys):
    header("tab_deterioration_auc")
    checks, rows = [], []
    for market, score, m, panel, y, firm, raw_col, table_col in det_cells(s, ys):
        for o in OUTCOMES:
            a_table, n, npos = cv.auc(y[o], panel[table_col])
            a_raw, n_raw, _ = cv.auc(y[o], panel[raw_col])
            keep = y[o].notna() & panel[raw_col].notna()
            rows.append({"market": market, "score": score, "model": m,
                         "outcome": OUTCOMES[o][0], "n": n, "events": npos,
                         "firms": panel.loc[keep, firm].nunique(),
                         "auc_roc": a_table,
                         "auc_roc_us_raw": a_raw if market == "US" else np.nan,
                         "auc_column": table_col})
    data = pd.DataFrame(rows)

    expect = {("China", "default", "xgb"): (.762, .795, .741),
              ("China", "default", "rf"): (.820, .846, .816),
              ("China", "default", "enet"): (.799, .839, .794),
              ("China", "rating", "xgb"): (.792, .829, .784),
              ("China", "rating", "rf"): (.835, .862, .841),
              ("China", "rating", "enet"): (.800, .840, .790),
              ("US", "default", "xgb"): (.777, .694, .772),
              ("US", "default", "rf"): (.865, .840, .839),
              ("US", "default", "enet"): (.845, .830, .831),
              ("US", "rating", "xgb"): (.831, .764, .817),
              ("US", "rating", "rf"): (.849, .785, .837),
              ("US", "rating", "enet"): (.830, .795, .826)}
    idx = data.set_index(["market", "score", "model", "outcome"]).auc_roc
    for (market, score, m), e in expect.items():
        got = [idx[(market, score, m, OUTCOMES[o][0])] for o in OUTCOMES]
        chk(checks, all(close(a, b, 3) for a, b in zip(got, e)),
            f"{market} {score} {m} (coverage, EBITDA, loss): "
            + " ".join(f"{v:.4f}" for v in got) + "; expected "
            + " ".join(f"{v:.3f}" for v in e))

    # The rf raw cells must equal stage8_3's saved AUCs exactly: same rows, same
    # outcomes. That is what ties this table to the figures of exhibit 8.
    saved = pd.read_csv(P(AUC_RF), float_precision="round_trip")
    bad = []
    for r in saved.itertuples():
        score = "default" if r.family == "distress" else "rating"
        live = data[(data.market == r.market) & (data.score == score)
                    & (data.model == "rf")
                    & (data.outcome == OUTCOMES[r.outcome][0])].iloc[0]
        raw = live.auc_roc_us_raw if r.market == "US" else live.auc_roc
        if live.n != r.n or abs(raw - r.auc) > 1e-12:
            bad.append(f"{r.market} {r.family} {r.outcome}")
    chk(checks, not bad, f"the 12 rf raw AUCs and n equal {AUC_RF}"
        + (f" — differ at {bad}" if bad else ""))

    columns = {
        "market": "US or China",
        "score": "default or rating weakness score",
        "model": "xgb, rf or enet",
        "outcome": "intcov = interest cover below 1x at t+1; ebitdaneg = EBITDA "
                   "negative at t+1; loss = net loss at t+1",
        "n": "issuer-years with the t+1 outcome observed and the score present",
        "events": "issuer-years with the outcome = 1",
        "firms": "distinct firms among the n issuer-years",
        "auc_roc": "AUC-ROC on the column the earlier AUC table used "
                   "(auc_column): US = CALIBRATED out-of-fold score; China = raw "
                   "score, which is the only Chinese score",
        "auc_roc_us_raw": "US only: AUC-ROC on the RAW out-of-fold score, the "
                          "score the deterioration figures (exhibit 8) print; "
                          "blank for China",
        "auc_column": "the score column auc_roc is computed on"}
    notes = {
        "source files": [CN_LABELS, US_DIST_FY, US_RATE_FY, CSMAR_BOND]
        + CSMAR_STATEMENTS + [ANNUAL, AUC_RF],
        "sample": PANEL_SAMPLE + " (the stage8_3 samples)",
        "period": "fiscal years 2003-2024 (outcome at t+1)",
        "unit of observation": PANEL_UNIT, "method": OUTCOME_NOTE,
        "caveats": "The US auc_roc is on the CALIBRATED score because the "
                   "expected values of this table come from the earlier AUC "
                   "table, which used it; D-67 makes the raw score the "
                   "comparison benchmark, and that is auc_roc_us_raw. The two "
                   "differ by at most about 0.001. An AUC needs 100 of each "
                   "class (check_china_score_validity.MIN_CLASS)."}
    return exhibit("tab_deterioration_auc", "table", data, columns, notes, checks)


# ===========================================================================
# 8. fig_deterioration_roc_* and fig_deterioration_density_*  (stage8_4)
# ===========================================================================
def ex_deterioration_figures(det):
    header("fig_deterioration_roc_* and fig_deterioration_density_* (12 figures)")
    checks = []
    bins100 = pd.read_csv(P(BINS_100))
    bins1000 = pd.read_csv(P(BINS_1000))
    aucs = pd.read_csv(P(AUC_RF), float_precision="round_trip")
    cells = [(mk, fam, o) for mk in ("US", "China") for fam in ("distress", "rating")
             for o in OUTCOMES]
    key = ["market", "family", "outcome"]

    # stage8_4's checks: 12 cells in each file, complete bins, totals agree.
    for name, table, nb in [(BINS_100, bins100, 100), (BINS_1000, bins1000, 1000)]:
        found = set(table[key].drop_duplicates().itertuples(index=False, name=None))
        sizes = table.groupby(key).size()
        complete = all(np.allclose(np.sort(g.bin_lo.to_numpy()), np.arange(nb) / nb)
                       for _, g in table.groupby(key))
        chk(checks, found == set(cells) and (sizes == nb).all() and complete,
            f"{name}: the 12 cells, each with its {nb:,} equal-width bins")
    t100 = bins100.groupby(key)[["n_pos", "n_neg"]].sum()
    t1000 = bins1000.groupby(key)[["n_pos", "n_neg"]].sum()
    n_by_cell = aucs.set_index(key).n
    chk(checks, len(aucs) == 12 and t100.equals(t1000)
        and ((t100.n_pos + t100.n_neg) == n_by_cell.reindex(t100.index)).all(),
        "the two bin files agree on every cell's totals, and with n in " + AUC_RF)

    # And with this run's own rf cells (exhibit 7), from the live outcomes.
    bad = []
    for mk, fam, o in cells:
        score = "default" if fam == "distress" else "rating"
        live = det[(det.market == mk) & (det.score == score) & (det.model == "rf")
                   & (det.outcome == OUTCOMES[o][0])].iloc[0]
        if (t100.loc[(mk, fam, o), "n_pos"] != live.events
                or t100.loc[(mk, fam, o), "n_neg"] != live.n - live.events):
            bad.append((mk, fam, o))
    chk(checks, not bad, "the bins hold exactly the positives and negatives of "
        "this run's rf cells" + (f" — differ at {bad}" if bad else ""))

    def one(bins, mk, fam, o):
        g = bins[(bins.market == mk) & (bins.family == fam)
                 & (bins.outcome == o)].sort_values("bin_lo")
        a = aucs[(aucs.market == mk) & (aucs.family == fam) & (aucs.outcome == o)]
        return g, {"n_pos": g.n_pos.to_numpy(), "n_neg": g.n_neg.to_numpy(),
                   "auc": float(a.auc.iloc[0])}

    out = []
    common = {"source files": [BINS_100, BINS_1000, AUC_RF],
              "sample": PANEL_SAMPLE + " (the stage8_3 samples); rf only; rows "
                        "with the outcome observed",
              "period": "fiscal years 2003-2024 (outcome at t+1)",
              "unit of observation": PANEL_UNIT,
              "method": "Read from stage8_3's bin files and drawn as stage8_4 "
                        "draws them. " + RAW_NOTE + " " + OUTCOME_NOTE}
    for fam in ("distress", "rating"):
        score = "default" if fam == "distress" else "rating"
        for o, (short, label) in OUTCOMES.items():
            # ---- ROC, from the 1,000-bin counts ----------------------------
            parts, inputs = [], {}
            for mk in ("US", "China"):
                g, d = one(bins1000, mk, fam, o)
                inputs[mk] = d
                pos = d["n_pos"][::-1].astype(float)     # high score first
                neg = d["n_neg"][::-1].astype(float)
                parts.append(pd.DataFrame({
                    "market": mk, "point": np.arange(len(pos) + 1),
                    "threshold_bin_lo": np.r_[np.nan, g.bin_lo.to_numpy()[::-1]],
                    "n_pos_in_bin": np.r_[0, pos], "n_neg_in_bin": np.r_[0, neg],
                    "tpr": np.r_[0, np.cumsum(pos) / pos.sum()],
                    "fpr": np.r_[0, np.cumsum(neg) / neg.sum()],
                    "auc_printed": d["auc"]}))
            roc_data = pd.concat(parts, ignore_index=True)

            def draw_roc(stem, a=inputs, f=fam, oc=short):
                vf.roc(a["US"], a["China"], f, oc, stem)

            roc_cols = {
                "market": "US or China",
                "point": "0 = the start (0, 0); point k includes the k highest "
                         "score bins",
                "threshold_bin_lo": "lower edge of the last bin included: point k "
                                    "flags scores >= this value; blank at point 0",
                "n_pos_in_bin": "issuer-years with the outcome in that bin",
                "n_neg_in_bin": "issuer-years without it in that bin",
                "tpr": "true positive rate, the curve's y", "fpr": "false "
                "positive rate, the curve's x",
                "auc_printed": "exact AUC on the raw score, printed in the legend "
                               f"(from {AUC_RF})"}
            out.append(exhibit(
                f"fig_deterioration_roc_{score}_{short}", "figure", roc_data,
                roc_cols, dict(common, figure=f"validation_figures.roc, {score} "
                               f"score (rf), {label}, 1,000 bins of width 0.001"),
                checks, draw_roc))

            # ---- density, from the 100-bin counts --------------------------
            parts, inputs = [], {}
            for mk in ("US", "China"):
                g, d = one(bins100, mk, fam, o)
                inputs[mk] = d
                pos, neg = d["n_pos"].astype(float), d["n_neg"].astype(float)
                parts.append(pd.DataFrame({
                    "market": mk, "bin_lo": g.bin_lo.to_numpy(),
                    "bin_mid": g.bin_lo.to_numpy() + 0.005,
                    "n_pos": d["n_pos"], "n_neg": d["n_neg"],
                    "share_pos": pos / pos.sum(), "share_neg": neg / neg.sum(),
                    "density_deteriorated_plotted":
                        vf._smooth(pos / pos.sum(), vf.SIGMA[fam]),
                    "density_not_deteriorated_plotted":
                        vf._smooth(neg / neg.sum(), vf.SIGMA[fam])}))
            den_data = pd.concat(parts, ignore_index=True)

            def draw_den(stem, a=inputs, f=fam, oc=short):
                vf.density(a["US"], a["China"], f, oc, stem)

            den_cols = {
                "market": "US or China",
                "bin_lo": "lower edge of the 0.01-wide bin of the raw rf score",
                "bin_mid": "bin centre, the x of the curves",
                "n_pos": "issuer-years with the outcome in the bin",
                "n_neg": "issuer-years without it",
                "share_pos": "n_pos / all n_pos of the market",
                "share_neg": "n_neg / all n_neg of the market",
                "density_deteriorated_plotted": "the coloured curve: share_pos "
                    "smoothed by validation_figures._smooth (width "
                    f"{vf.SIGMA[fam]} bins), as a density on [0, 1]",
                "density_not_deteriorated_plotted": "the grey curve: share_neg, "
                    "smoothed the same way"}
            out.append(exhibit(
                f"fig_deterioration_density_{score}_{short}", "figure", den_data,
                den_cols, dict(common, figure=f"validation_figures.density, "
                               f"{score} score (rf), {label}, 100 bins"),
                checks, draw_den))
    return out


# ===========================================================================
# 9. tab_group_auc   (within_group2.py, point estimates)
# ===========================================================================
GROUPS = [(0, "strongest 30%"), (1, "middle 40%"), (2, "weakest 30%")]
MIN_POS = 10


def three_groups(score):
    """0 / 1 / 2 = strongest 30% / middle 40% / weakest 30%, ranked in-sample."""
    p = score.rank(pct=True).to_numpy()
    return np.where(p <= 0.30, 0, np.where(p > 0.70, 2, 1))


def ex_group_auc(s, ys):
    header("tab_group_auc")
    checks, rows = [], []
    for market, score, m, panel, y, firm, raw_col, _ in det_cells(s, ys):
        sv = panel[raw_col].to_numpy()
        if np.isnan(sv).any():
            raise SystemExit(f"STOP: {market} {raw_col} has a missing score.")
        g = three_groups(panel[raw_col])
        firms = panel[firm].to_numpy()
        for o in OUTCOMES:
            yv = y[o].astype(float).to_numpy()
            obs = ~np.isnan(yv)
            overall = yv[obs].mean() * 100
            for k, label in GROUPS:
                cell = obs & (g == k)
                n, npos = int(cell.sum()), int(yv[cell].sum())
                nf = len(np.unique(firms[cell]))
                big = npos >= MIN_POS and nf >= MIN_FIRMS and npos < n
                rows.append({"market": market, "score": score, "model": m,
                             "outcome": OUTCOMES[o][0], "group": label, "n": n,
                             "events": npos, "firms": nf,
                             "rate_pct": npos / n * 100 if n else np.nan,
                             "market_rate_pct": overall,
                             "rate_relative_to_market":
                                 (npos / n * 100) / overall if n else np.nan,
                             "auc_roc": roc_auc_score(yv[cell], sv[cell])
                             if big else np.nan,
                             "auc_status": "computed" if big else "too small"})
    data = pd.DataFrame(rows)

    # Reproduce the earlier analysis's log: market totals and three rf rows.
    lab = {"intcov": "coverage", "ebitdaneg": "EBITDA", "loss": "loss"}
    totals = {("China", "default"): [(11475, 617), (11266, 303), (19384, 1143)],
              ("US", "default"): [(39287, 8209), (40944, 3641), (43416, 10172)],
              ("US", "rating"): [(24640, 3411), (25190, 833), (26339, 4572)]}
    for (market, score), e in totals.items():
        d = data[(data.market == market) & (data.score == score)
                 & (data.model == "rf")]
        got = [(int(d[d.outcome == o].n.sum()), int(d[d.outcome == o].events.sum()))
               for o in lab]
        chk(checks, got == e, f"{market} {score} panel, observed and positives "
            f"by outcome: {got}; expected {e} [{WITHIN_GROUP_LOG}]")
    weak = {("China", "default"): [(3190, 458, .7935), (3128, 240, .8234),
                                   (5632, 849, .7718)],
            ("US", "default"): [(11939, 6173, .7154), (12021, 2873, .6024),
                                (12692, 6988, .7048)],
            ("US", "rating"): [(7576, 2610, .7072), (7603, 623, .5628),
                               (7734, 3271, .6927)]}
    for (market, score), e in weak.items():
        d = data[(data.market == market) & (data.score == score)
                 & (data.model == "rf") & (data.group == "weakest 30%")]
        got = [tuple(d[d.outcome == o][["n", "events", "auc_roc"]].iloc[0])
               for o in lab]
        ok = all(int(a[0]) == b[0] and int(a[1]) == b[1] and close(a[2], b[2], 4)
                 for a, b in zip(got, e))
        chk(checks, ok, f"{market} {score} rf, weakest 30%: n, events, AUC by "
            f"outcome match the log [{WITHIN_GROUP_LOG}]")

    columns = {
        "market": "US or China", "score": "default or rating weakness score",
        "model": "xgb, rf or enet",
        "outcome": "intcov, ebitdaneg or loss, all at t+1",
        "group": "the score ranked within the market panel (pandas rank, "
                 "pct): strongest 30% = rank <= 0.30, weakest 30% = rank > 0.70, "
                 "middle 40% = the rest; membership is fixed on the whole panel, "
                 "before rows with the outcome unobserved are dropped",
        "n": "issuer-years in the group with the outcome observed",
        "events": "issuer-years in the group with the outcome = 1",
        "firms": "distinct firms in the cell",
        "rate_pct": "events / n, % (the deterioration rate)",
        "market_rate_pct": "the rate over all observed issuer-years of that "
                           "market panel and outcome, %",
        "rate_relative_to_market": "rate_pct / market_rate_pct",
        "auc_roc": "AUC-ROC of the raw score INSIDE the group",
        "auc_status": "'too small' where the cell has fewer than 10 events, "
                      "fewer than 20 firms, or no non-events; auc_roc is then "
                      "blank"}
    notes = {"source files": [CN_LABELS, US_DIST_FY, US_RATE_FY, CSMAR_BOND]
             + CSMAR_STATEMENTS + [ANNUAL],
             "sample": PANEL_SAMPLE, "period": "fiscal years 2003-2024 (outcome "
             "at t+1)", "unit of observation": PANEL_UNIT,
             "method": "The point estimates of the 2026-11-04 within-group "
                       "analysis (within_group2.py); its bootstrap intervals are "
                       "not reproduced here. " + RAW_NOTE + " " + OUTCOME_NOTE}
    return exhibit("tab_group_auc", "table", data, columns, notes, checks)


# ===========================================================================
# 10. tab_own_outcome_auc
# ===========================================================================
def china_rating_panel(bond_master, cn_items):
    """check_china_score_validity.main lines 414-452: the §9 sample."""
    cn = pd.read_parquet(P(CN_PANEL))
    cn = cn[cn["score_distress_xgb"].notna()].copy()
    rt = cv.csmar("债券及主体评级情况表.csv",
                  ["Liscd", "Sctcd", "RatingDate", "Ctcr"])
    have = ~rt.Ctcr.isin(["", "nan"]) & rt.Ctcr.notna()
    unknown = set(rt.loc[have, "Ctcr"].unique()) - set(cv.NOTCH) - cv.NOT_LONG_TERM
    if unknown:
        raise SystemExit(f"STOP: Ctcr values not on the notch scale: "
                         f"{sorted(unknown)}")
    rt = rt.merge(bond_master, on=["Liscd", "Sctcd"], how="left")
    rt["notch"] = rt.Ctcr.map(cv.NOTCH)
    rt = rt[rt.notch.notna() & rt.Orgid.notna()]
    rt["year"] = pd.to_datetime(rt.RatingDate, errors="coerce").dt.year
    # The LOWEST rating in the year = the highest notch.
    rp = rt.groupby(["Orgid", "year"])["notch"].max().reset_index()
    if rp.duplicated(["Orgid", "year"]).any():
        raise SystemExit("STOP: rating panel not unique on (Orgid, year).")
    m = cn.merge(rp, left_on=["gvkey", "fyear"], right_on=["Orgid", "year"],
                 how="inner")
    m = m[m.fyear >= 2010].copy()
    m["log_at"] = np.log(pd.MultiIndex.from_arrays(
        [m.gvkey, m.fyear]).map(cn_items["at"]).astype("float64"))
    return m


def ex_own_outcome_auc(s, us, bond_master, cn_items):
    header("tab_own_outcome_auc")
    checks, rows = [], []

    def add(part, market, sample, score, model, group, measure, value, n, firms,
            events):
        rows.append({"part": part, "market": market, "sample": sample,
                     "score": score, "model": model, "group": group,
                     "measure": measure, "value": value, "rows": n,
                     "firms": firms, "events": events})

    # ---- (a) and (b): default score against default, fiscal 2013-2024 -------
    sets = [("US", s["ud13"], "label", "gvkey", "raw_{}",
             "stage8_2_us_distress_firmyear, fiscal 2013-2024"),
            ("China", s["cn13"], "y_default", "Orgid", "score_distress_{}",
             "stage8_2_china_labels, scored and in_default, fiscal 2013-2024")]
    expect_a = {"US": (.907, .924, .905), "China": (.677, .713, .777)}
    share_range = {"US": (89, 95), "China": (55, 74)}
    auc_a = {}
    for market, d, ycol, firm, fmt, sample in sets:
        y = d[ycol].astype(int).to_numpy()
        chk(checks, (len(d), d[firm].nunique(), int(y.sum()))
            == {"US": (25221, 3064, 235), "China": (23545, 4262, 152)}[market],
            f"{market} sample: {len(d):,} firm-years, {d[firm].nunique():,} "
            f"firms, {int(y.sum()):,} defaults")
        for m in MODELS:
            sc = d[fmt.format(m)]
            a = roc_auc_score(y, sc)
            auc_a[(market, m)] = a
            add("a", market, sample, "default", m, "all", "AUC-ROC", a, len(d),
                d[firm].nunique(), int(y.sum()))
            g = three_groups(sc)
            for k, label in GROUPS:
                cell = g == k
                ev = int(y[cell].sum())
                add("b", market, sample, "default", m, label, "default rate (%)",
                    ev / cell.sum() * 100, int(cell.sum()),
                    d.loc[cell, firm].nunique(), ev)
                add("b", market, sample, "default", m, label,
                    "share of the sample's defaults (%)", ev / y.sum() * 100,
                    int(cell.sum()), d.loc[cell, firm].nunique(), ev)
        got = [auc_a[(market, m)] for m in MODELS]
        chk(checks, all(close(a, b, 3) for a, b in zip(got, expect_a[market])),
            f"{market} default AUC xgb rf enet: " + " ".join(f"{v:.4f}" for v in got)
            + "; expected " + " ".join(f"{v:.3f}" for v in expect_a[market]))
        lo, hi = share_range[market]
        shares = [r["value"] for r in rows if r["part"] == "b"
                  and r["market"] == market and r["group"] == "weakest 30%"
                  and r["measure"].startswith("share")]
        chk(checks, all(lo <= round(v) <= hi for v in shares),
            f"{market} weakest-30% share of defaults, xgb rf enet: "
            + " ".join(f"{v:.1f}%" for v in shares) + f"; expected in "
            f"[{lo}%, {hi}%] (each rounded to a whole per cent)")

    # stage8_5's exported rows are these rows: the rf AUC must be identical.
    exported = pd.read_csv(P(ROWS_85), float_precision="round_trip")
    for market in ("US", "China"):
        e = exported[exported.market == market]
        a = roc_auc_score(e.y_default, e.score_rf)
        chk(checks, abs(a - auc_a[(market, "rf")]) < 1e-12
            and len(e) == {"US": 25221, "China": 23545}[market],
            f"{market}: rf AUC from {ROWS_85} equals the stage8_2 rows' "
            f"({a:.6f})")

    # ---- (c) the rating score against US speculative grade, out-of-fold -----
    expect_c = (.831, .827, .788)
    got = []
    for m in MODELS:
        r = us["rres"][m]
        firms = us["rsc"][m].gvkey.nunique()
        for measure in ("AUC-ROC", "AUC-PR"):
            add("c", "US", "stage4_2 fitted rating sample, 52,553 rows", "rating",
                m, "all", measure, r["metrics"][measure], r["n_rows"], firms,
                r["n_events"])
        got.append(r["metrics"]["AUC-ROC"])
    chk(checks, all(close(a, b, 3) for a, b in zip(got, expect_c)),
        "US speculative-grade AUC xgb rf enet: " + " ".join(f"{v:.4f}" for v in got)
        + "; expected .831 .827 .788")

    # ---- (d) China, the six scores and log total assets against AA+ ---------
    m_panel = china_rating_panel(bond_master, cn_items)
    cut = m_panel.notch <= 3                     # AA+ or above (AAA+ = 0)
    samples = [("2010-2024 pooled", m_panel.fyear >= 2010)]
    samples += [(lab, m_panel.fyear.between(lo, hi)) for lab, lo, hi in cv.PERIODS]
    samples += [(str(int(y)), m_panel.fyear == y)
                for y in sorted(m_panel.fyear.unique())]
    pooled = {}
    for lab, sel in samples:
        n, npos = int(sel.sum()), int(cut[sel].sum())
        nf = m_panel.loc[sel, "gvkey"].nunique()
        add("d", "China", lab, "", "", "all", "AA+ or above share (%)",
            npos / n * 100, n, nf, npos)
        for col, score, model in ([("log_at", "log total assets", "")]
                                  + [(f"score_{x}", x.split("_")[0],
                                      x.split("_")[1]) for x in cv.SCORES]):
            a, _, _ = cv.auc(cut[sel], m_panel.loc[sel, col])
            add("d", "China", lab, score, model, "all", "AUC-ROC, AA+ or above",
                np.nan if a is None else a, n, nf, npos)
            if lab == "2010-2024 pooled":
                pooled[col] = a
    six = [pooled[f"score_{x}"] for x in cv.SCORES]
    chk(checks, all(0.50 <= round(v, 2) <= 0.73 for v in six),
        "pooled 2010-2024, the six scores: " + " ".join(f"{v:.4f}" for v in six)
        + "; expected in [0.50, 0.73] (each rounded to 2 decimals)")
    chk(checks, abs(pooled["log_at"] - 0.87) <= 0.01,
        f"pooled 2010-2024, log total assets: {pooled['log_at']:.4f}; expected "
        f"about 0.87 (within 0.01)")
    n_all, n_aa = len(m_panel), int(cut.sum())
    chk(checks, n_all == 21726 and m_panel.gvkey.nunique() == 4003
        and close(n_aa / n_all * 100, 46.0, 1),
        f"§9 sample: {n_all:,} issuer-years, {m_panel.gvkey.nunique():,} "
        f"issuers, AA+ or above share {n_aa / n_all * 100:.1f}%; expected "
        "21,726, 4,003, 46.0%")
    yearly = [r["value"] for r in rows if r["part"] == "d"
              and r["sample"].isdigit() and r["score"] in ("distress", "rating")
              and pd.notna(r["value"])]
    print(f"    NOTE   by year the six scores range {min(yearly):.4f} to "
          f"{max(yearly):.4f}; the pooled figures are the ones checked")

    # ---- (e) Chinese rated issuer-years below BBB- ---------------------------
    below = m_panel.notch > 11                   # BBB- = 11 on this scale
    add("e", "China", "2010-2024 pooled", "", "", "all",
        "issuer-years rated below BBB-", int(below.sum()), n_all,
        m_panel.gvkey.nunique(), int(below.sum()))
    add("e", "China", "2010-2024 pooled", "", "", "all", "issuers rated below "
        "BBB- in at least one year", m_panel.loc[below, "gvkey"].nunique(),
        n_all, m_panel.gvkey.nunique(), int(below.sum()))
    chk(checks, int(below.sum()) == 65 and n_all == 21726,
        f"{int(below.sum())} of {n_all:,} below BBB-; expected 65 of 21,726")

    data = pd.DataFrame(rows)
    columns = {
        "part": "a = default score against default; b = share of defaults by "
                "score group; c = rating score against US speculative grade; d = "
                "China, scores and log total assets against the AA+-or-above "
                "split; e = Chinese issuer-years rated below BBB-",
        "market": "US or China",
        "sample": "the sample of the row (for d: pooled, a period, or one year)",
        "score": "default (distress) or rating weakness score; for d also 'log "
                 "total assets'; blank where the row is a count or a share",
        "model": "xgb, rf or enet; blank where no model is involved",
        "group": "for b: strongest 30% / middle 40% / weakest 30% of the score "
                 "ranked within the sample (rank <= 0.30, middle, > 0.70); "
                 "otherwise 'all'",
        "measure": "what value is",
        "value": "the number: an AUC (0-1), a rate or share in %, or a count",
        "rows": "rows (issuer-years) in the sample or group",
        "firms": "distinct firms (gvkey or Orgid) in the sample or group",
        "events": "a, b: defaults; c: speculative-grade rows; d: issuer-years "
                  "rated AA+ or above; e: issuer-years rated below BBB-"}
    notes = {
        "source files": [US_DIST_FY, CN_LABELS, ROWS_85]
        + [r_results(m) for m in MODELS] + [r_scores(m) for m in MODELS]
        + [CN_PANEL, CSMAR_RATING, CSMAR_BOND, CSMAR_STATEMENTS[0]],
        "sample": "a, b: tests123.py of 2026-11-04 — US 25,221 firm-years, 3,064 "
                  "firms, 235 defaults; China 23,545 issuer-years, 4,262 issuers, "
                  "152 defaults (scored and in_default). c: the stage4_2 fitted "
                  "rating sample, 52,553 rows, 23,121 speculative grade. d, e: "
                  "check_china_score_validity.py (the code behind PROJECT_REPORT "
                  "§9) — scored Chinese issuer-years with a rating dated in the "
                  "same year, fiscal 2010 onward, 21,726 issuer-years, 4,003 "
                  "issuers",
        "period": "a, b: fiscal 2013-2024; c: fiscal 1993-2025; d, e: fiscal "
                  "2010-2024",
        "unit of observation": "a, b: US gvkey x fyear, China Orgid x fyear; c: "
                               "issuer x gvkey x fyear; d, e: Orgid x fyear",
        "method": "a: AUC of the RAW default score against y_default (China) and "
                  "the stage8_2 label (US); both put one positive per issuer on "
                  "its last statement knowable before the event; the US event "
                  "includes bankruptcy filings, China has no bankruptcy data. b: "
                  "the tests123.py group split. c: out-of-fold AUCs from the "
                  "stage4_2 results files, no refit. d: AA+ or above = notch <= 3 "
                  "on check_china_score_validity's scale (AAA+ = 0); where an "
                  "issuer has several ratings in a year the LOWEST is taken; "
                  "log total assets from CSMAR A001000000. The pooled row is new: "
                  "the earlier code printed only by period and by year. An AUC "
                  "needs 100 of each class, so 2010-2012 are blank. e: below "
                  "BBB- = notch >= 12.",
        "caveats": "The by-year AUCs of the six scores range below 0.50 in some "
                   "years (minimum printed when the script runs); the range "
                   "checked is the pooled one."}
    return exhibit("tab_own_outcome_auc", "table", data, columns, notes, checks)


# ===========================================================================
# 11 and 13. fig_default_roc, tab_default_by_band, fig_default_by_band  (8_6)
# ===========================================================================
def ex_default_figures():
    header("fig_default_roc, tab_default_by_band, fig_default_by_band")
    checks_roc, checks_band = [], []
    rows = pd.read_csv(P(ROWS_85), float_precision="round_trip")
    cuts = np.percentile(rows.loc[rows.market == "US", "score_rf"], s86.PERCENTILES)
    rows["band"] = np.digitize(rows.score_rf, cuts)    # low <= score < high
    tables, curves, roc_parts = {}, {}, []
    for mk in ("US", "China"):
        r = rows[rows.market == mk]
        tables[mk] = s86.band_table(r, mk)
        fpr, tpr, thr = roc_curve(r.y_default, r.score_rf, drop_intermediate=False)
        a = roc_auc_score(r.y_default, r.score_rf)
        curves[mk] = {"fpr": fpr, "tpr": tpr, "auc": a}
        roc_parts.append(pd.DataFrame({
            "market": mk, "point": np.arange(len(fpr)),
            "threshold": np.where(np.isinf(thr), np.nan, thr),
            "fpr": fpr, "tpr": tpr, "auc": a}))
        n_distinct = r.score_rf.nunique()
        want = {"US": 0.924, "China": 0.713}[mk]
        chk(checks_roc, round(a, 3) == want
            and fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1
            and len(fpr) == n_distinct + 1
            and abs(area_under_curve(fpr, tpr) - a) < 1e-9,
            f"{mk}: AUC {a:.4f} (expected {want:.3f}), {len(fpr):,} points "
            f"for {n_distinct:,} distinct scores, from (0,0) to (1,1)")
    band = pd.concat([tables["US"], tables["China"]], ignore_index=True)
    lows = np.r_[-np.inf, cuts]
    highs = np.r_[cuts, np.inf]
    band["cut_low"] = np.tile(lows, 2)
    band["cut_high"] = np.tile(highs, 2)
    band = band.rename(columns={"rate": "rate_pct", "lo": "ci_lo_pct",
                                "hi": "ci_hi_pct"})
    expect = {"US": [0.03, 0.04, 0.26, 0.32, 0.79, 3.73, 11.33],
              "China": [0.00, 0.37, 0.38, 0.64, 0.54, 1.81, 6.21]}
    for mk, e in expect.items():
        got = list(band[band.market == mk].rate_pct)
        chk(checks_band, all(close(a, b, 2) for a, b in zip(got, e)),
            f"{mk} rates: " + " ".join(f"{v:.2f}" for v in got) + "; expected "
            + " ".join(f"{v:.2f}" for v in e))
        t = band[band.market == mk]
        chk(checks_band, (int(t.firm_years.sum()), int(t.events.sum()))
            == {"US": (25221, 235), "China": (23545, 152)}[mk],
            f"{mk}: the seven bands hold every firm-year and every default")

    def draw_roc(stem):
        vf.roc_default(curves["US"], curves["China"], stem)

    def band_dict(mk):
        t = band[band.market == mk]
        return {"rate": t.rate_pct.to_numpy(), "lo": t.ci_lo_pct.to_numpy(),
                "hi": t.ci_hi_pct.to_numpy(), "drawn": t.drawn.to_numpy()}

    def draw_band(stem):
        vf.default_by_band(s86.BAND_LABELS, band_dict("US"), band_dict("China"),
                           stem)

    common = {"source files": [ROWS_85, "(its source) " + US_DIST_FY,
                               "(its source) " + CN_LABELS],
              "sample": "the default test of tests123.py: US 25,221 firm-years, "
                        "3,064 firms, 235 defaults; China 23,545 issuer-years, "
                        "4,262 issuers, 152 defaults",
              "period": "fiscal years 2013-2024",
              "unit of observation": "US gvkey x fyear; China Orgid x fyear; "
                                     "firm_id in the source file is a shuffled "
                                     "number, not a real identifier",
              "method": "RAW rf default score, stage8_6_default_figures.py, whose "
                        "band_table() is called. Seven bands cut at the 30th, "
                        "50th, 70th, 80th, 90th and 95th percentiles of the US "
                        "rows; the US cuts are applied unchanged to China; a row "
                        "is in a band if low <= score < high. 95% interval: firms "
                        "resampled with replacement 1,000 times, seed 43 US and "
                        "42 China, band membership fixed. The ROC uses every "
                        "distinct score as a threshold (drop_intermediate=False)."}
    roc_cols = {"market": "US or China", "point": "0 = the start (0, 0)",
                "threshold": "flag scores >= this value; blank at point 0 "
                             "(above every score)",
                "fpr": "false positive rate, the curve's x",
                "tpr": "true positive rate, the curve's y",
                "auc": "exact AUC, printed in the legend"}
    band_cols = {"market": "US or China",
                 "band": "percentile band of the RAW rf default score, cut on "
                         "the US rows (both markets use the US cuts)",
                 "firm_years": "rows in the band", "firms": "distinct firms",
                 "events": "defaults in the band",
                 "rate_pct": "events / firm_years, %",
                 "ci_lo_pct": "2.5th percentile of the band's rate over 1,000 "
                              "firm resamples, %",
                 "ci_hi_pct": "97.5th percentile, %",
                 "drawn": "True if the band has at least 20 firms, so the "
                          "figure draws it",
                 "cut_low": "lower cut of the band (score >= this); blank for "
                            "the lowest band, which has no lower bound",
                 "cut_high": "upper cut (score < this); blank for the highest "
                             "band, which has no upper bound"}
    return [exhibit("fig_default_roc", "figure", pd.concat(roc_parts,
                    ignore_index=True), roc_cols,
                    dict(common, figure="validation_figures.roc_default"),
                    checks_roc, draw_roc),
            exhibit("tab_default_by_band", "table", band, band_cols, common,
                    checks_band),
            exhibit("fig_default_by_band", "figure", band, band_cols,
                    dict(common, figure="validation_figures.default_by_band, "
                         "bands with drawn = False are not drawn"),
                    checks_band, draw_band)]


# ===========================================================================
# 12. The rating-inflation exhibits   (stage7_2 --scores raw)
# ===========================================================================
SCALE_NAME = {"a": "notch", "b": "S&P category", "c": "investment grade, pp"}
FIG2_LABELS = ["Strong fundamentals (score < 0.3)", "Middle (0.3 to 0.7)",
               "Weak fundamentals (0.7 and above)"]


def ex_rating_inflation():
    header("rating inflation: fig_rating_distribution, fig_rating_by_fundamentals, "
           "fig_rating_by_score_band, tab_rating_inflation")
    # stage7_2's own loading, under --scores raw: the US score is swapped for
    # stage4_2's raw out-of-fold score. The Chinese score is raw either way.
    s72.SCORES_VERSION = "raw"
    us, cn = s72.load()
    s72.add_cols(us, cn)
    seed_us, seed_cn = s72.SEED_US, s72.SEED_CN
    rows, band_rf = [], []
    checks = []

    def row(**kw):
        base = {k: np.nan for k in ["us_rows", "us_firms", "us_value", "us_ci_lo",
                                    "us_ci_hi", "china_rows", "china_firms",
                                    "china_value", "china_ci_lo", "china_ci_hi",
                                    "china_minus_us", "diff_ci_lo", "diff_ci_hi"]}
        base.update({"part": "", "model": "", "scale": "", "score_band": "",
                     "difference_reported": True})
        base.update(kw)
        rows.append(base)

    # ---- Table 7: the gap overall and by score band, three models, 3 scales ---
    for m in MODELS:
        a = us.assign(b=s72.band_of(us[f"s_{m}"]))
        b = cn.assign(b=s72.band_of(cn[f"s_{m}"]))
        for tag, val, _name, _nd, is_share, _split in s72.SCALES:
            k = 100.0 if is_share else 1.0
            for i in range(10):
                pa, pb = a[a.b == i], b[b.b == i]
                nfa, nfb = pa.firm.nunique(), pb.firm.nunique()
                am, ad = s72.firm_boot(pa, val, seed_us)
                bm, bd = s72.firm_boot(pb, val, seed_cn)
                alo, ahi = s72.pct_ci(ad)
                blo, bhi = s72.pct_ci(bd)
                ok = nfa >= MIN_FIRMS and nfb >= MIN_FIRMS
                dl, dh = s72.pct_ci(bd - ad) if ok else (np.nan, np.nan)
                lab = f"{i / 10:.1f}-{(i + 1) / 10:.1f}"
                row(part="gap by score band", model=m, scale=SCALE_NAME[tag],
                    score_band=lab, us_rows=len(pa), us_firms=nfa,
                    us_value=am * k, us_ci_lo=alo * k, us_ci_hi=ahi * k,
                    china_rows=len(pb), china_firms=nfb, china_value=bm * k,
                    china_ci_lo=blo * k, china_ci_hi=bhi * k,
                    china_minus_us=(bm - am) * k if ok else np.nan,
                    diff_ci_lo=dl * k, diff_ci_hi=dh * k, difference_reported=ok)
                if m == "rf" and tag == "a":
                    band_rf.append({"score_band": lab, "us_rows": len(pa),
                                    "us_firms": nfa, "us_mean_notch": am,
                                    "us_ci_lo": alo, "us_ci_hi": ahi,
                                    "us_drawn": nfa >= MIN_FIRMS,
                                    "china_rows": len(pb), "china_firms": nfb,
                                    "china_mean_notch": bm, "china_ci_lo": blo,
                                    "china_ci_hi": bhi,
                                    "china_drawn": nfb >= MIN_FIRMS})
            bk, draws, valid = s72.gap_machinery(a, b, val)
            d = draws(bk)
            lo, hi = s72.pct_ci(d[~np.isnan(d)])
            row(part="overall gap", model=m, scale=SCALE_NAME[tag],
                score_band="all bands with 20+ US firms", us_rows=len(a),
                us_firms=a.firm.nunique(), us_value=bk.exp_us.mean() * k,
                china_rows=len(bk), china_firms=bk.firm.nunique(),
                china_value=bk[val].mean() * k, china_minus_us=bk.gap.mean() * k,
                diff_ci_lo=lo * k, diff_ci_hi=hi * k)

    # ---- share rated AA- or above (notch <= 4 on stage7_2's scale) -----------
    us_sh, cn_sh = (us.notch <= 4).mean() * 100, (cn.notch <= 4).mean() * 100
    row(part="share rated AA- or above, %", us_rows=len(us),
        us_firms=us.firm.nunique(), us_value=us_sh, china_rows=len(cn),
        china_firms=cn.firm.nunique(), china_value=cn_sh,
        china_minus_us=cn_sh - us_sh)

    # ---- same rating, AA- and above: Table 6's unadjusted comparison --------
    a_t, b_t = (us[us.notch <= 4].reset_index(drop=True),
                cn[cn.notch <= 4].reset_index(drop=True))
    nfa, nfb = a_t.firm.nunique(), b_t.firm.nunique()
    for m in MODELS:
        am, ad = s72.boot_w(a_t, f"s_{m}", seed_us)
        bm, bd = s72.boot_w(b_t, f"s_{m}", seed_cn)
        ok = nfa >= MIN_FIRMS and nfb >= MIN_FIRMS
        dl, dh = s72.pct_ci(bd - ad)
        row(part="same rating, AA- and above: mean rating score", model=m,
            us_rows=len(a_t), us_firms=nfa, us_value=am,
            us_ci_lo=s72.pct_ci(ad)[0], us_ci_hi=s72.pct_ci(ad)[1],
            china_rows=len(b_t), china_firms=nfb, china_value=bm,
            china_ci_lo=s72.pct_ci(bd)[0], china_ci_hi=s72.pct_ci(bd)[1],
            china_minus_us=bm - am if ok else np.nan,
            diff_ci_lo=dl if ok else np.nan, diff_ci_hi=dh if ok else np.nan,
            difference_reported=ok)
    tab = pd.DataFrame(rows)
    order = ["part", "model", "scale", "score_band", "us_rows", "us_firms",
             "us_value", "us_ci_lo", "us_ci_hi", "china_rows", "china_firms",
             "china_value", "china_ci_lo", "china_ci_hi", "china_minus_us",
             "diff_ci_lo", "diff_ci_hi", "difference_reported"]
    tab = tab[order]

    # ---- checks ---------------------------------------------------------------
    chk(checks, (len(us), us.firm.nunique(), len(cn), cn.firm.nunique())
        == (18276, 2066, 23085, 4084),
        f"samples: US {len(us):,} firm-years / {us.firm.nunique():,} firms, China "
        f"{len(cn):,} / {cn.firm.nunique():,}; expected 18,276 / 2,066, 23,085 / "
        "4,084")
    ov = tab[tab.part == "overall gap"].set_index(["model", "scale"])
    x = ov.loc[("xgb", "notch")]
    chk(checks, close(x.china_minus_us, -8.74, 2) and close(x.diff_ci_lo, -8.86, 2)
        and close(x.diff_ci_hi, -8.60, 2),
        f"xgb overall gap {x.china_minus_us:+.3f} [{x.diff_ci_lo:+.3f}, "
        f"{x.diff_ci_hi:+.3f}] notches; expected -8.74 [-8.86, -8.60]")
    for (mm, sc), e, dd in [(("xgb", "S&P category"), -2.90, 2),
                            (("xgb", "investment grade, pp"), 49.8, 1),
                            (("rf", "notch"), -8.67, 2),
                            (("enet", "notch"), -8.52, 2)]:
        v = ov.loc[(mm, sc)].china_minus_us
        chk(checks, close(v, e, dd), f"{mm} overall gap, {sc}: {v:+.4f}; "
            f"expected {e:+}")
    bb = tab[(tab.part == "gap by score band") & (tab.model == "rf")
             & (tab.scale == "notch")].set_index("score_band").china_minus_us
    chk(checks, close(bb["0.1-0.2"], -5.7, 1) and close(bb["0.9-1.0"], -13.5, 1),
        f"rf gap in band 0.1-0.2 {bb['0.1-0.2']:+.3f}, in 0.9-1.0 "
        f"{bb['0.9-1.0']:+.3f}; expected -5.7 and -13.5")
    chk(checks, close(cn_sh, 99.1, 1) and close(us_sh, 1.7, 1),
        f"rated AA- or above: China {cn_sh:.2f}%, US {us_sh:.2f}%; expected "
        "99.1% and 1.7%")
    sr = tab[(tab.part.str.startswith("same rating")) & (tab.model == "xgb")]
    chk(checks, close(sr.china_minus_us.iloc[0], 0.257, 3),
        f"xgb same-rating difference at AA- and above "
        f"{sr.china_minus_us.iloc[0]:+.4f}; expected +0.257")

    # ---- Figure 1: the 22 notch shares -----------------------------------------
    fig_checks = []
    chk(fig_checks, list(s72.NOTCH_LETTER[1:]) == list(ifg.LETTERS),
        "stage7_2's notch letters equal inflation_figures.LETTERS")
    us_pct = [(us.notch == n).mean() * 100 for n in range(1, 23)]
    cn_pct = [(cn.notch == n).mean() * 100 for n in range(1, 23)]
    f1 = pd.DataFrame({"notch": range(1, 23), "letter": ifg.LETTERS,
                       "us_rows": [int((us.notch == n).sum()) for n in range(1, 23)],
                       "us_pct": us_pct,
                       "china_rows": [int((cn.notch == n).sum())
                                      for n in range(1, 23)],
                       "china_pct": cn_pct})
    chk(fig_checks, f1.us_rows.sum() == len(us) and f1.china_rows.sum() == len(cn),
        "figure 1: the 22 notches hold every rated row of each market")

    # ---- Figure 2 (rf): rating distribution by level of fundamentals ----------
    tests = [lambda v: v < 0.3, lambda v: (v >= 0.3) & (v < 0.7),
             lambda v: v >= 0.7]
    groups, f2rows = [], []
    for label, test in zip(FIG2_LABELS, tests):
        ga, gb = us[test(us.s_rf)], cn[test(cn.s_rf)]
        su = [(ga.band == bd).mean() * 100 if len(ga) else 0.0 for bd in s72.BANDS]
        sc = [(gb.band == bd).mean() * 100 if len(gb) else 0.0 for bd in s72.BANDS]
        groups.append((label, su, sc))
        for bd, u, c in zip(s72.BANDS, su, sc):
            f2rows.append({"group": label, "rating_category": bd,
                           "us_rows_in_group": len(ga), "us_pct": u,
                           "china_rows_in_group": len(gb), "china_pct": c})
    f2 = pd.DataFrame(f2rows)
    n_us = f2.drop_duplicates("group").us_rows_in_group.sum()
    n_cn = f2.drop_duplicates("group").china_rows_in_group.sum()
    chk(fig_checks, n_us == 18276 and n_cn == 23085,
        f"figure 2: the rf groups partition the samples (US {n_us:,}, China "
        f"{n_cn:,})")

    # ---- Figure 3 (rf): mean rating by score band, with the 20-firm flags -----
    f3 = pd.DataFrame(band_rf)
    chk(fig_checks, len(f3) == 10, "figure 3: ten score bands")

    sample = ("rated firm-years, fiscal 2010-2024: US 18,276 firm-years, 2,066 "
              "gvkeys (stage7_0); China 23,085 issuer-years, 4,084 Orgids "
              "(stage7_1)")
    common = {"source files": [US_RATING_PANEL, CN_RATING_PANEL]
              + [r_scores(m) for m in MODELS],
              "sample": sample, "period": "fiscal years 2010-2024",
              "unit of observation": "US gvkey x fyear; China Orgid x fyear",
              "method": "stage7_2_inflation_descriptive.py --scores raw: its own "
                        "load(), add_cols(), firm_boot(), boot_w(), "
                        "gap_machinery() and band_of() are called. RAW rating "
                        "scores on both sides. Ratings on one notch scale, AAA = 1 "
                        "to C = 21, D = 22; US = worst rating in force at 31 "
                        "December across agencies and bonds; China = issuer rating "
                        "Ctcr, lowest in the year or carried forward. Bootstrap: "
                        "1,000 draws resampling firms, seed 42 China, 43 US."}
    tab_cols = {
        "part": "overall gap (Table 7) / gap by score band (Table 7) / share "
                "rated AA- or above / same rating, AA- and above (Table 6, "
                "unadjusted)",
        "model": "the rating-score model: xgb, rf or enet; blank for the shares",
        "scale": "notch (AAA = 1 ... D = 22); S&P category (modifiers removed, "
                 "AAA = 1 ... D = 10); investment grade, pp (share BBB- or "
                 "better, in percentage points). A higher notch or category is "
                 "a WORSE rating",
        "score_band": "tenth of the raw rating score, the last closed on the "
                      "right; 'all bands with 20+ US firms' for the overall gap",
        "us_rows": "US rows in the cell", "us_firms": "US firms in the cell",
        "us_value": "gap by band: US mean on the scale; overall gap: mean of the "
                    "US band means assigned to the Chinese rows (the expected US "
                    "rating at the same score); share: % of US rows; same "
                    "rating: US mean rating score",
        "us_ci_lo": "2.5th percentile of the US firm bootstrap",
        "us_ci_hi": "97.5th percentile of the US firm bootstrap",
        "china_rows": "Chinese rows in the cell",
        "china_firms": "Chinese firms in the cell",
        "china_value": "the Chinese counterpart of us_value (for the overall "
                       "gap, the Chinese mean actual rating)",
        "china_ci_lo": "2.5th percentile of the Chinese firm bootstrap",
        "china_ci_hi": "97.5th percentile of the Chinese firm bootstrap",
        "china_minus_us": "China minus US. Negative on notch and category, or "
                          "positive on investment grade, means China rates the "
                          "same fundamentals better; for same rating, positive "
                          "means Chinese issuers of that rating are weaker",
        "diff_ci_lo": "2.5th percentile of the difference, the two markets "
                      "resampled independently",
        "diff_ci_hi": "97.5th percentile of the difference",
        "difference_reported": "False where either market has fewer than 20 "
                               "firms in the cell; the difference is then blank"}
    out = [exhibit("tab_rating_inflation", "table", tab, tab_cols, common, checks)]

    def draw1(stem):
        ifg.fig1(us_pct, cn_pct, stem)

    def draw2(stem):
        ifg.fig2(groups, stem)

    def draw3(stem):
        ifg.fig3(f3.us_mean_notch.to_numpy(), f3.us_ci_lo.to_numpy(),
                 f3.us_ci_hi.to_numpy(), f3.china_mean_notch.to_numpy(),
                 f3.china_ci_lo.to_numpy(), f3.china_ci_hi.to_numpy(), stem,
                 model="rf", us_ok=f3.us_drawn.to_numpy(),
                 cn_ok=f3.china_drawn.to_numpy())

    out.append(exhibit(
        "fig_rating_distribution", "figure", f1,
        {"notch": "rating notch, AAA = 1 ... D = 22", "letter": "the rating",
         "us_rows": "US firm-years at that notch", "us_pct": "% of US rated "
         "firm-years", "china_rows": "Chinese issuer-years at that notch",
         "china_pct": "% of Chinese rated issuer-years"},
        dict(common, figure="inflation_figures.fig1(us_pct, cn_pct); stage7_2 "
             "Figure 1, uses no score"), fig_checks, draw1))
    out.append(exhibit(
        "fig_rating_by_fundamentals", "figure", f2,
        {"group": "panel: the raw rf rating score < 0.3, 0.3 to < 0.7, >= 0.7",
         "rating_category": "AAA, AA, A, BBB, BB, B, CCC-C (CCC+ to C), D",
         "us_rows_in_group": "US firm-years in the group",
         "us_pct": "% of the group's US firm-years in the category",
         "china_rows_in_group": "Chinese issuer-years in the group",
         "china_pct": "% of the group's Chinese issuer-years in the category"},
        dict(common, figure="inflation_figures.fig2(groups), rf, panel labels "
             "as in the group column"), fig_checks, draw2))
    out.append(exhibit(
        "fig_rating_by_score_band", "figure", f3,
        {"score_band": "tenth of the raw rf rating score",
         "us_rows": "US firm-years", "us_firms": "US firms",
         "us_mean_notch": "US mean notch", "us_ci_lo": "2.5th percentile of the "
         "US firm bootstrap", "us_ci_hi": "97.5th percentile",
         "us_drawn": "True if at least 20 US firms, so the band is drawn",
         "china_rows": "Chinese issuer-years", "china_firms": "Chinese firms",
         "china_mean_notch": "Chinese mean notch", "china_ci_lo": "2.5th "
         "percentile of the Chinese firm bootstrap", "china_ci_hi": "97.5th "
         "percentile", "china_drawn": "True if at least 20 Chinese firms"},
        dict(common, figure="inflation_figures.fig3(..., model='rf', us_ok, "
             "cn_ok); Table 7(a), rf"), fig_checks, draw3))
    return out


# ===========================================================================
# 14. Appendix: tab_all_features, tab_stability, tab_rating_score
# ===========================================================================
def partial_correlations(fit, feats):
    """OLS residual correlation, and a matrix-inverse second route."""
    y = fit.y.to_numpy(dtype=float)
    X = fit[feats].to_numpy(dtype=float)
    inverse = np.linalg.inv(np.corrcoef(np.column_stack([X, y]), rowvar=False))
    out = {}
    for f in ["OPLeverage", "DelEqu"]:
        j = feats.index(f)
        x = X[:, j]
        controls = np.column_stack([np.ones(len(y)), np.delete(X, j, axis=1)])
        rx = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
        ry = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
        out[f] = {"simple": np.corrcoef(x, y)[0, 1],
                  "partial": np.corrcoef(rx, ry)[0, 1],
                  "r2_feature": 1 - rx.var() / x.var(),
                  "r2_outcome": 1 - ry.var() / y.var(),
                  "other_route": -inverse[j, -1]
                  / np.sqrt(inverse[j, j] * inverse[-1, -1])}
    return out


def ex_all_features(us, blocks):
    header("tab_all_features")
    checks = []
    feats, fit = us["feats"], us["fit"]
    ranks = rank_table(us["dres"], feats)
    coef = coefficients(us["coef_d"])
    pc = partial_correlations(fit, feats)
    rows = []
    for f in ranks.index:
        r = {"feature": f, "xgb_rank": ranks.loc[f, "xgb"],
             "rf_rank": ranks.loc[f, "rf"], "enet_rank": ranks.loc[f, "enet"],
             "mean_rank": ranks.loc[f, "mean_rank"],
             "rank_spread": ranks.loc[f, "spread"], "enet_coefficient": coef[f]}
        if f not in INDICATORS:
            g = decile_table(fit, f).set_index("dec")
            for dec in (1, 10):
                r[f"decile{dec}_n"] = g.loc[dec, "n"]
                r[f"decile{dec}_events"] = g.loc[dec, "events"]
                r[f"decile{dec}_rate_pct"] = g.loc[dec, "rate_pct"]
        else:
            for v, lab in [(1.0, "1"), (0.0, "0"), (-1.0, "undetermined")]:
                st = fit[fit[f] == v]
                r[f"state_{lab}_n"] = len(st)
                r[f"state_{lab}_events"] = int(st.y.sum())
                r[f"state_{lab}_rate_pct"] = st.y.mean() * 100
        if f in pc:
            r["simple_corr_with_default"] = pc[f]["simple"]
            r["partial_corr_with_default"] = pc[f]["partial"]
            r["r2_feature_on_other_26"] = pc[f]["r2_feature"]
            r["r2_default_on_other_26"] = pc[f]["r2_outcome"]
        rows.append(r)
    order = ["feature", "xgb_rank", "rf_rank", "enet_rank", "mean_rank",
             "rank_spread", "enet_coefficient", "decile1_n", "decile1_events",
             "decile1_rate_pct", "decile10_n", "decile10_events",
             "decile10_rate_pct"] + [f"state_{lab}_{k}" for lab in
                                     ("1", "0", "undetermined")
                                     for k in ("n", "events", "rate_pct")] + [
        "simple_corr_with_default", "partial_corr_with_default",
        "r2_feature_on_other_26", "r2_default_on_other_26"]
    data = pd.DataFrame(rows).reindex(columns=order)

    # Section 1 ranks, Section 3 deciles and states, the coefficients.
    s1 = v4_block(blocks, "## 1. Consensus")
    bad = [f for f in feats if f not in s1 or not all(
        same_as_cell(v, c) for v, c in zip(ranks.loc[f, MODELS + ["mean_rank",
                                                                  "spread"]],
                                           s1[f][1:6]))]
    chk(checks, len(s1) == 28 and not bad,
        "all 27 rank rows equal Section 1 of " + V4 + (f" — differ: {bad}"
                                                       if bad else ""))
    bad = []
    for f in [x for x in feats if x not in INDICATORS]:
        ref = v4_block(blocks, f"**`{f}`** — ")
        g = decile_table(fit, f)
        for r in g.itertuples():
            c = ref.get(str(int(r.dec)))
            if c is None or not (same_as_cell(r.n, c[1]) and same_as_cell(
                    r.events, c[2]) and same_as_cell(r.rate_pct, c[3])):
                bad.append(f"{f} D{int(r.dec)}")
    chk(checks, not bad, "all 250 decile rows of the 25 continuous features "
        "equal Section 3" + (f" — differ: {bad[:10]}" if bad else ""))
    bad = []
    for f in INDICATORS:
        ref = v4_block(blocks, f"**`{f}`** — as a state")
        for v, lab in [(1.0, "1"), (0.0, "0"), (-1.0, "-1 (undetermined)")]:
            st, c = fit[fit[f] == v], ref.get(lab)
            if c is None or not (same_as_cell(len(st), c[1]) and same_as_cell(
                    st.y.sum(), c[2]) and same_as_cell(st.y.mean() * 100, c[3])):
                bad.append(f"{f} {lab}")
    chk(checks, not bad, "OENEG and INTWO state rows equal Section 3"
        + (f" — differ: {bad}" if bad else ""))
    en = v4_block(blocks, "**Elastic net.**")
    nz = coef[coef != 0]
    chk(checks, len(nz) == 22 and all(f in en and same_as_cell(v, en[f][1])
                                      for f, v in nz.items())
        and sorted(coef[coef == 0].index) == sorted(
            ["ChNWC", "DelCOL", "OENEG", "cashdebt", "roic"]),
        "22 non-zero coefficients equal the summary's; zero: ChNWC, DelCOL, "
        "OENEG, cashdebt, roic")
    for f, e in [("OPLeverage", (0.04025, -0.00392)), ("DelEqu", (-0.08519, 0.00365))]:
        chk(checks, close(pc[f]["simple"], e[0], 5) and close(pc[f]["partial"],
                                                               e[1], 5)
            and abs(pc[f]["partial"] - pc[f]["other_route"]) < 1e-8,
            f"{f}: simple {pc[f]['simple']:+.5f}, partial {pc[f]['partial']:+.5f} "
            f"(inverse-matrix route {pc[f]['other_route']:+.5f}); expected "
            f"{e[0]:+.5f}, {e[1]:+.5f}")

    columns = {
        "feature": "feature name",
        "xgb_rank": "rank of XGBoost gain, 1 = largest, of 27 (method 'min')",
        "rf_rank": "rank of random-forest impurity decrease",
        "enet_rank": "rank of |elastic-net coefficient|; zero coefficients tie "
                     "at the lowest rank",
        "mean_rank": "mean of the three ranks",
        "rank_spread": "largest minus smallest of the three ranks",
        "enet_coefficient": "signed coefficient per one standard deviation of "
                            "the feature; 0 = removed by the penalty",
        "decile1_n": "rows in the feature's lowest decile",
        "decile1_events": "defaults in the lowest decile",
        "decile1_rate_pct": "default rate in the lowest decile, %",
        "decile10_n": "rows in the highest decile",
        "decile10_events": "defaults in the highest decile",
        "decile10_rate_pct": "default rate in the highest decile, %",
        "state_1_n": "OENEG, INTWO only: rows with the indicator = 1",
        "state_1_events": "defaults among them",
        "state_1_rate_pct": "their default rate, %",
        "state_0_n": "rows with the indicator = 0",
        "state_0_events": "defaults among them",
        "state_0_rate_pct": "their default rate, %",
        "state_undetermined_n": "rows coded -1 (undetermined)",
        "state_undetermined_events": "defaults among them",
        "state_undetermined_rate_pct": "their default rate, %",
        "simple_corr_with_default": "OPLeverage, DelEqu only: Pearson "
                                    "correlation of the feature with the 0/1 "
                                    "default outcome",
        "partial_corr_with_default": "correlation of the residuals of two OLS "
                                     "regressions, the feature and the outcome "
                                     "each on a constant and the other 26 "
                                     "features (unweighted)",
        "r2_feature_on_other_26": "R2 of the feature on the other 26",
        "r2_default_on_other_26": "R2 of the outcome on the other 26"}
    notes = {"source files": [PANEL_L3] + [d_results(m) for m in MODELS]
             + [d_scores("xgb"), COEF_D],
             "sample": US_FIT_SAMPLE, "period": "fiscal years 1992-2025",
             "unit of observation": US_FIT_UNIT,
             "method": "Sections 1 and 3 of v4_results_summary.md (stage5_2 "
                       "passes 1 and 3) and the partial correlations of "
                       "check_methodology_numbers.py. Features as the models "
                       "received them (within-year ranks; OENEG and INTWO keep "
                       "-1 = undetermined, which the OLS treats as a number). "
                       "Importances from one full-sample fit."}
    return exhibit("tab_all_features", "table", data, columns, notes, checks)


def ex_stability(us, blocks):
    header("tab_stability")
    checks, rows = [], []
    ann = pd.read_parquet(P(ANNUAL), columns=["gvkey", "datadate", "at", "sich"])
    names = pd.read_parquet(P(NAMES), columns=["gvkey", "sic"])
    names["sic"] = names["sic"].replace("", pd.NA)
    ann = ann.merge(names, on="gvkey", how="left")
    ann["sic_use"] = ann["sich"].where(ann["sich"].notna(), ann["sic"])
    ann["division"] = ann["sic_use"].map(s52.sic_division)
    ctx = ann[["gvkey", "datadate", "division"]]
    for m in MODELS:
        d = us["fit"][KEY + ["datadate", "y"]].merge(
            us["dsc"][m][KEY + ["score_calibrated"]], on=KEY, how="inner")
        d = d.merge(ctx, on=["gvkey", "datadate"], how="left")
        if len(d) != 95370:
            raise SystemExit(f"STOP: the SIC join changed the row count to "
                             f"{len(d):,}.")
        d["block"] = (d.fyear // 3) * 3
        for kind, col in [("three-year block", "block"),
                          ("SIC division", "division")]:
            keys = d[col] if col == "block" else d[col].fillna("unknown")
            for k, g in d.groupby(keys):
                two = g.y.nunique() == 2
                rows.append({
                    "model": m, "split": kind,
                    "cell": f"{int(k)}-{int(k) + 2}" if col == "block" else k,
                    "rows": len(g), "firms": g.gvkey.nunique(),
                    "events": int(g.y.sum()),
                    "auc_roc": roc_auc_score(g.y, g.score_calibrated) if two
                    else np.nan,
                    "auc_pr": average_precision_score(g.y, g.score_calibrated)
                    if two else np.nan})
    data = pd.DataFrame(rows)
    for m in MODELS:
        for kind in ("three-year block", "SIC division"):
            ref = v4_block(blocks, f"**{m} — by {kind}**")
            sub = data[(data.model == m) & (data.split == kind)]
            bad = [r.cell for r in sub.itertuples()
                   if r.cell not in ref or not all(
                       same_as_cell(v, c) for v, c in zip(
                           [r.rows, r.firms, r.events, r.auc_roc, r.auc_pr],
                           ref[r.cell][1:6]))]
            chk(checks, ref and len(sub) == len(ref) - 1 and not bad,
                f"{m} by {kind}: {len(sub)} rows equal Section 5 of {V4}"
                + (f" — differ: {bad}" if bad else ""))
    columns = {"model": "xgb, rf or enet",
               "split": "three-year block of fiscal years, or SIC division",
               "cell": "the block (e.g. 1992-1994) or the division (SIC from "
                       "sich, falling back to comp_names.sic, D-10)",
               "rows": "rows in the cell", "firms": "distinct gvkeys",
               "events": "default events",
               "auc_roc": "AUC-ROC of the calibrated out-of-fold default score "
                          "within the cell; blank with no events",
               "auc_pr": "AUC-PR within the cell; blank with no events"}
    notes = {"source files": [PANEL_L3] + [d_scores(m) for m in MODELS]
             + [ANNUAL, NAMES], "sample": US_FIT_SAMPLE,
             "period": "fiscal years 1992-2025 (the 2025-2027 block holds only "
                       "fiscal 2025)",
             "unit of observation": US_FIT_UNIT,
             "method": "Pass 5 of stage5_2_results_summary.py, its own "
                       "sic_division(). AUC-ROC within a cell is computed on "
                       "the pooled out-of-fold scores of that cell.",
             "caveats": "Cells with few events (e.g. 1 in public admin, 3 in "
                        "2025-2027) give unstable AUCs; read them with the "
                        "event count beside them."}
    return exhibit("tab_stability", "table", data, columns, notes, checks)


def ex_rating_score(us, blocks):
    header("tab_rating_score")
    checks, rows = [], []
    for m in MODELS:
        r = us["rres"][m]
        rows.append({"part": "performance", "model": m,
                     "auc_pr": r["metrics"]["AUC-PR"],
                     "auc_roc": r["metrics"]["AUC-ROC"],
                     "top10_capture_pct": r["metrics"]["rec@10%"] * 100,
                     "rows": r["n_rows"], "events": r["n_events"],
                     "groups": r["n_groups"]})
    ranks = rank_table(us["rres"], us["feats"])
    coef = coefficients(us["coef_r"])
    for f in ranks.index:
        rows.append({"part": "features", "feature": f,
                     "xgb_rank": ranks.loc[f, "xgb"], "rf_rank": ranks.loc[f, "rf"],
                     "enet_rank": ranks.loc[f, "enet"],
                     "mean_rank": ranks.loc[f, "mean_rank"],
                     "rank_spread": ranks.loc[f, "spread"],
                     "enet_coefficient": coef[f],
                     "top10_both_xgb_rf": bool(ranks.loc[f, "xgb"] <= 10
                                               and ranks.loc[f, "rf"] <= 10)})
    data = pd.DataFrame(rows).reindex(columns=[
        "part", "model", "auc_pr", "auc_roc", "top10_capture_pct", "rows",
        "events", "groups", "feature", "xgb_rank", "rf_rank", "enet_rank",
        "mean_rank", "rank_spread", "enet_coefficient", "top10_both_xgb_rf"])

    head = v4_block(blocks, "**Headline, rating.**")
    perf = data[data.part == "performance"]
    chk(checks, all(m in head and same_as_cell(r.auc_pr, head[m][1])
                    and same_as_cell(r.auc_roc, head[m][2])
                    for m, r in zip(MODELS, perf.itertuples())),
        "AUC-PR and AUC-ROC equal the summary's rating headline "
        "(.8001/.8314, .7967/.8268, .7459/.7876)")
    chk(checks, (perf.rows == 52553).all() and (perf.events == 23121).all(),
        "52,553 rows and 23,121 speculative-grade rows for all three models")
    expect = {"PM": (1, 1, -0.4012), "INTWO": (2, 7, 0.0640),
              "roaq": (3, 3, -0.2984), "roavol": (4, 2, 0.6570),
              "TLTA": (7, 4, 0.3578), "DelEqu": (8, 6, 0.3098),
              "cashdebt": (10, 5, -0.1070)}
    both = ranks[(ranks.xgb <= 10) & (ranks.rf <= 10)]
    chk(checks, set(both.index) == set(expect) and all(
        both.loc[f, "xgb"] == e[0] and both.loc[f, "rf"] == e[1]
        and close(coef[f], e[2], 4) for f, e in expect.items()),
        "top ten of both xgb and rf: PM, INTWO, roaq, roavol, TLTA, DelEqu, "
        "cashdebt, with the expected ranks and coefficients")
    chk(checks, sorted(coef[coef == 0].index) == ["AssetGrowth", "Cash"],
        f"zero coefficients: {sorted(coef[coef == 0].index)}; expected "
        "AssetGrowth, Cash")
    saved = pd.Series(us["rres"]["enet"]["importances"])
    chk(checks, (saved.reindex(coef.index) - coef).abs().max() < 1e-9,
        f"the coefficients in {COEF_R} equal those in {r_results('enet')}")

    columns = {
        "part": "performance (one row per model) or features (one row per "
                "feature)",
        "model": "xgb, rf or enet (performance rows)",
        "auc_pr": "out-of-fold AUC-PR against the US speculative-grade label",
        "auc_roc": "out-of-fold AUC-ROC",
        "top10_capture_pct": "share of speculative-grade rows in the top 10% "
                             "of scores, %",
        "rows": "rows fitted", "events": "speculative-grade rows (label = 1)",
        "groups": "cross-fitting groups",
        "feature": "feature name (feature rows)",
        "xgb_rank": "rank of XGBoost gain, 1 = largest, of 27",
        "rf_rank": "rank of random-forest impurity decrease",
        "enet_rank": "rank of |elastic-net coefficient|",
        "mean_rank": "mean of the three ranks",
        "rank_spread": "largest minus smallest rank",
        "enet_coefficient": "signed coefficient per one standard deviation; "
                            "positive = a higher value raises the rating score",
        "top10_both_xgb_rf": "True if in the top ten of both xgb and rf"}
    notes = {"source files": [r_results(m) for m in MODELS]
             + [r_scores(m) for m in MODELS] + [COEF_R],
             "sample": "stage4_2 fitted rating sample: 52,553 rows, 23,121 "
                       "speculative grade (43.996%), 3,933 groups",
             "period": "fiscal years 1993-2025",
             "unit of observation": "issuer x gvkey x fiscal-year statement row",
             "method": "Read from the saved stage4_2 results (out-of-fold "
                       "metrics; importances from one full-sample fit) and the "
                       "stage5_0 rating coefficient file. Same 27 features and "
                       "model settings as the default score."}
    return exhibit("tab_rating_score", "table", data, columns, notes, checks)


# ===========================================================================
# NUMBERS QUOTED IN THE TEXT — printed with their source, no file
# ===========================================================================
def text_numbers(us):
    print("\n" + "=" * 92)
    print("NUMBERS QUOTED IN THE TEXT (printed only, with their source)")
    print("=" * 92)

    def say(passed, text, source):
        print(f"    {'PASSED' if passed else 'FAILED'}   {text}\n"
              f"             source: {source}")

    # Agreement among the three default models (stage5_2 pass 6).
    dsc = us["dsc"]
    base = dsc["xgb"][KEY + ["y"]].copy()
    for m in MODELS:
        base = base.merge(dsc[m][KEY + ["score_calibrated"]].rename(
            columns={"score_calibrated": m}), on=KEY, how="inner")
    n_top = int(round(len(base) * 0.10))
    tops = {m: set(base[m].nlargest(n_top).index) for m in MODELS}
    pairs = [("xgb", "rf"), ("xgb", "enet"), ("rf", "enet")]
    rho = [base[a].corr(base[b], method="spearman") for a, b in pairs]
    ovl = [len(tops[a] & tops[b]) / n_top * 100 for a, b in pairs]
    say(all(0.69 <= round(v, 2) <= 0.72 for v in rho)
        and all(72 <= round(v) <= 81 for v in ovl),
        "three default models, Spearman " + ", ".join(f"{v:.4f}" for v in rho)
        + "; top-decile overlap " + ", ".join(f"{v:.1f}%" for v in ovl)
        + f" (xgb/rf, xgb/enet, rf/enet; {len(base):,} rows); quoted 0.69-0.72, "
        "72%-81%", ", ".join(d_scores(m) for m in MODELS)
        + " (calibrated out-of-fold; stage5_2 pass 6)")

    # L3 against L0.
    rho = []
    for m in MODELS:
        j = dsc[m][KEY + ["score_calibrated"]].merge(
            us["l0sc"][m][KEY + ["score_calibrated"]], on=KEY,
            suffixes=("_l3", "_l0"), how="inner")
        rho.append(j.score_calibrated_l3.corr(j.score_calibrated_l0,
                                              method="spearman"))
    say(all(0.85 <= round(v, 2) <= 0.95 for v in rho),
        f"L3 vs L0 Spearman xgb rf enet " + ", ".join(f"{v:.4f}" for v in rho)
        + f" ({len(j):,} rows in both); quoted 0.85-0.95",
        "stage4_1_scores_L3_K12_* and stage4_1_scores_L0_K12_* (stage5_2 pass 6)")

    # The default against the rating score, US issuer-years in both panels.
    rate = pd.read_parquet(P(SP_RATE))
    dist = pd.read_parquet(P(SP_DIST))
    both = rate.merge(dist, on=KEY, suffixes=("_rate", "_dist"))
    n = int(round(len(both) * 0.10))
    rho, ovl = [], []
    for m in MODELS:
        a, b = both[f"score_{m}_rate"], both[f"score_{m}_dist"]
        rho.append(a.corr(b, method="spearman"))
        ovl.append(len(set(a.nlargest(n).index) & set(b.nlargest(n).index))
                   / n * 100)
    say(all(0.64 <= round(v, 2) <= 0.78 for v in rho)
        and all(61 <= round(v) <= 73 for v in ovl),
        "default vs rating score, xgb rf enet: Spearman "
        + ", ".join(f"{v:.4f}" for v in rho) + "; top-decile overlap "
        + ", ".join(f"{v:.1f}%" for v in ovl) + f" ({len(both):,} issuer-years "
        "in both panels); quoted 0.64-0.78, 61%-73%",
        f"{SP_RATE} and {SP_DIST} (stage5_1 compare_with_distress, calibrated "
        "scores)")

    # Chinese coverage.
    cn = pd.read_parquet(P(CN_PANEL))
    scored = cn.score_distress_xgb.notna()
    never = int((~scored.groupby(cn.gvkey).any()).sum())
    say(len(cn) == 28399 and int(scored.sum()) == 24557 and never == 474,
        f"Chinese issuer-years scored {int(scored.sum()):,} of {len(cn):,}; "
        f"issuers never scored {never:,} of {cn.gvkey.nunique():,}; quoted "
        "24,557 of 28,399 and 474", CN_PANEL)

    # Mean raw xgb default score.
    raw_mean = dsc["xgb"].score_raw.mean()
    say(close(raw_mean, 0.1427, 4),
        f"mean US raw xgb default score {raw_mean:.4f} over {len(dsc['xgb']):,} "
        "out-of-fold rows; quoted 0.1427", d_scores("xgb") + " (score_raw)")

    # FISD records coded bankruptcy, 2013-2024 US sample (check_us_default_types
    # Part 1).
    defaults = pd.read_parquet(P(FISD_DEFAULT),
                               columns=["issue_id", "default_date", "default_type"])
    link = pd.read_parquet(P(FISD_MERGED), columns=["issue_id", "issuer_id"])
    defaults["issue_id"] = defaults.issue_id.astype("Int64")
    link["issue_id"] = link.issue_id.astype("Int64")
    link["issuer_id"] = link.issuer_id.astype("Int64")
    if link.issue_id.duplicated().any():
        raise SystemExit("STOP: fisd_mergedissue is not unique on issue_id.")
    defaults = defaults.merge(link, on="issue_id", how="left")
    defaults["date"] = pd.to_datetime(defaults.default_date, errors="coerce")
    events = pd.read_parquet(P(EVENTS), columns=["issuer_id"])
    events["issuer_id"] = events.issuer_id.astype("Int64")
    defaults = defaults[defaults.issuer_id.isin(events.issuer_id)
                        & defaults.date.notna()].reset_index(drop=True)
    defaults["record"] = np.arange(len(defaults))
    panel = pd.read_parquet(P(SP_DIST), columns=["issuer_id", "gvkey", "fyear"])
    panel["issuer_id"] = panel.issuer_id.astype("Int64")
    pairs = panel.loc[panel.fyear.between(2013, 2024),
                      ["issuer_id", "gvkey"]].drop_duplicates()
    linked = defaults.merge(pairs, on="issuer_id")
    n_all = linked.record.nunique()
    n_b = linked.loc[linked.default_type == "B", "record"].nunique()
    say(n_b == 520 and n_all == 537,
        f"FISD default records coded bankruptcy (B), firms of the 2013-2024 US "
        f"sample: {n_b} of {n_all} ({n_b / n_all * 100:.1f}%); quoted 520 of 537",
        f"{FISD_DEFAULT}, {FISD_MERGED}, {EVENTS}, {SP_DIST} "
        "(check_us_default_types.py Part 1)")


# ===========================================================================
# WRITING
# ===========================================================================
REQUIRED_NOTES = ["source files", "sample", "period", "unit of observation"]
NOTE_ORDER = REQUIRED_NOTES + ["method", "caveats", "figure"]


def validate(ex):
    """Stop on a programming slip: an undefined column or a missing note."""
    missing = [c for c in ex["data"].columns if c not in ex["columns"]]
    extra = [c for c in ex["columns"] if c not in ex["data"].columns]
    absent = [k for k in REQUIRED_NOTES if k not in ex["notes"]]
    if missing or extra or absent:
        raise SystemExit(f"STOP: {ex['name']}: columns without a definition "
                         f"{missing}, definitions without a column {extra}, notes "
                         f"missing {absent}. NOTHING WRITTEN.")


def write(ex):
    stem = EXHIBIT_DIR / ex["name"]
    notes = []
    for k in NOTE_ORDER:
        if k in ex["notes"]:
            v = ex["notes"][k]
            notes.append({"item": k, "text": "; ".join(v) if isinstance(v, list)
                          else v})
    notes.append({"item": "checks", "text": " | ".join(
        f"{'PASSED' if p else 'FAILED'}: {t}" for p, t in ex["checks"])})
    for c, text in ex["columns"].items():
        notes.append({"item": f"column: {c}", "text": text})
    data = ex["data"].replace([np.inf, -np.inf], np.nan)
    with pd.ExcelWriter(f"{stem}.xlsx", engine="openpyxl") as xw:
        data.to_excel(xw, sheet_name="data", index=False)
        pd.DataFrame(notes).to_excel(xw, sheet_name="notes", index=False)
    files = [f"{ex['name']}.xlsx"]
    if ex["kind"] == "figure":
        ex["draw"](str(stem))
        files = [f"{ex['name']}.png", f"{ex['name']}.pdf"] + files
    return files


def main():
    if EXHIBIT_DIR.exists():
        raise SystemExit(f"STOP: {EXHIBIT_DIR} already exists. This script never "
                         "overwrites; move or rename the folder first. NOTHING "
                         "WRITTEN.")
    print("=" * 92)
    print("REPORT EXHIBITS — computing every exhibit and every check before "
          "writing anything")
    print("=" * 92)

    blocks = v4_blocks()
    us = load_us()
    s = load_stage8_2()
    bond_master, cn_items, ys = build_outcomes(s)

    main_text, appendix = [], []
    main_text.append(ex_us_performance(us))
    main_text.append(ex_leading_features(us))
    main_text.append(ex_feature_deciles(us))
    fig_ew, tab_ew = ex_early_warning(us, blocks)
    main_text.append(fig_ew)
    main_text.append(ex_score_summary(s))
    main_text.append(ex_score_distribution(s))
    det = ex_deterioration_auc(s, ys)
    main_text.append(det)
    main_text += ex_deterioration_figures(det["data"])
    main_text.append(ex_group_auc(s, ys))
    main_text.append(ex_own_outcome_auc(s, us, bond_master, cn_items))
    fig_roc, tab_band, fig_band = ex_default_figures()
    main_text.append(fig_roc)
    inflation = ex_rating_inflation()
    main_text += inflation[1:] + inflation[:1]     # figures 1-3, then the table
    main_text += [tab_band, fig_band]
    appendix.append(ex_all_features(us, blocks))
    appendix.append(tab_ew)
    appendix.append(ex_stability(us, blocks))
    appendix.append(ex_rating_score(us, blocks))
    text_numbers(us)

    exhibits = main_text + appendix
    for ex in exhibits:
        validate(ex)

    # ---- write the exhibits whose checks all passed -------------------------
    EXHIBIT_DIR.mkdir(parents=True)
    print("\n" + "=" * 92)
    print(f"WRITING into {EXHIBIT_DIR.relative_to(PROJECT_DIR)}/")
    print("=" * 92)
    summary = []
    for ex in exhibits:
        n_ok = sum(p for p, _ in ex["checks"])
        n_all = len(ex["checks"])
        files = write(ex) if n_ok == n_all else []
        summary.append((ex["name"], files, n_ok, n_all))

    print("\n" + "=" * 92)
    print("SUMMARY — one line per exhibit")
    print("=" * 92)
    for name, files, n_ok, n_all in summary:
        written = ", ".join(files) if files else "NOT WRITTEN (a check failed)"
        print(f"  {name:<40} checks {n_ok}/{n_all} passed   {written}")
    n_files = sum(len(f) for _, f, _, _ in summary)
    size = sum(p.stat().st_size for p in EXHIBIT_DIR.iterdir())
    print(f"\n  {n_files} files, {size / 1e6:.1f} MB, in "
          f"{EXHIBIT_DIR.relative_to(PROJECT_DIR)}/")


if __name__ == "__main__":
    main()
