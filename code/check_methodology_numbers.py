"""
READ-ONLY CHECK. Prints three sets of numbers for the methodology document.

It writes no file and changes no file. It fits no model. `data_raw/` is not read.
The current specification: 27 features, L3, K12, as in output/v4_results_summary.md.

PART 1  Model settings for xgb, rf and enet, on the default and the rating outcome.
PART 2  Partial correlations of OPLeverage and DelEqu with the default outcome.
PART 3  The rating score's importance ranks and elastic-net coefficients.

WHERE THE HYPERPARAMETERS COME FROM. The results files store only a short summary
string, which leaves out reg_lambda, tree_method and every default. So the unfitted
estimators are built with each stage's own `build_models`, using the saved row and
event counts, and asked for all their parameters. A check confirms that the same
call gives back the exact summary string saved at fitting time, and that the
installed library versions are the ones recorded then, since an unset parameter
takes its value from the library.

XGBOOST'S UNSET PARAMETERS. `get_params` shows None for anything the code left
unset. The value applied comes from the saved booster's configuration, and is read
ONLY for those names. That configuration does not hold the training parameters:
for a parameter the code did set, it shows the library default (eta 0.3, not the
0.05 actually used), so it must never be read for those.

THE PARTIAL-CORRELATION SAMPLE. output/stage4_0_distress_L3_K12.parquet is the
panel after imputation and the rank transformation, with 98,408 rows and 1,193
events. stage4_1 then drops rows still holding a missing value, leaving 95,370 rows
and 1,164 events, and that same drop is made here. The columns are used exactly as
the models received them, so OENEG and INTWO keep their code of -1 for undetermined
and OLS treats that code as a number. Each OLS has a constant. It is unweighted,
unlike the class-weighted models. Rows are issuer rows, so a statement shared by
several issuers counts once per issuer.

RANKS follow stage5_2's Section 1: rank 1 is the largest absolute value in that
model's own measure, and ties share the lowest rank. They are computed from the
saved importances, which come from one full-sample fit, not from cross-fitting.

Any failed check stops the run.

Runtime: under a minute.

Run:  python3 code/check_methodology_numbers.py
"""

import json
import sys
from pathlib import Path

# Importing the two sibling scripts would otherwise write code/__pycache__, and
# this script writes no files.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import sklearn
import xgboost

import stage4_1_distress_models as stage4_1
import stage4_2_rating_models as stage4_2

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS = ["xgb", "rf", "enet"]
BAR = "=" * 92

# Every file used for each outcome, as paths relative to the project folder.
OUTCOMES = {
    "default": {
        "module": stage4_1,
        "code": "code/stage4_1_distress_models.py",
        "results": "output/stage4_1_results_L3_K12_{m}.json",
        "meta": "output/stage5_0_model_L3_K12_base_{m}_meta.json",
        "coef": "output/stage5_0_model_L3_K12_base_enet_coefficients.csv",
        "booster": "output/stage5_0_model_L3_K12_base_xgb.json",
        "rows": 95370, "events": 1164},
    "rating": {
        "module": stage4_2,
        "code": "code/stage4_2_rating_models.py",
        "results": "output/stage4_2_results_{m}.json",
        "meta": "output/stage5_0_model_rating_L3_K12_base_{m}_meta.json",
        "coef": "output/stage5_0_model_rating_L3_K12_base_enet_coefficients.csv",
        "booster": "output/stage5_0_model_rating_L3_K12_base_xgb.json",
        "rows": 52553, "events": 23121},
}

FINAL_FIT_CODE = "code/stage5_0_fit_final_models.py"
PANEL = "output/stage4_0_distress_L3_K12.parquet"
SCORES = "output/stage4_1_scores_L3_K12_xgb.parquet"
SUMMARY = "output/v4_results_summary.md"

PANEL_ROWS, PANEL_EVENTS = 98408, 1193
FITTED_FIRMS, FITTED_ISSUERS = 5545, 7486
FEATURES_TO_TEST = ["OPLeverage", "DelEqu"]
N_FEATURES = 27
TOP_N = 10
TOLERANCE = 1e-8

# The lines of the fitting code that show each setting: text to find, lines to show.
CODE_COMMON = [("RANDOM_STATE = 42", 1), ("N_SPLITS = 5", 1), ("N_JOBS = 4", 1),
               ("build_models(n_pos, n_neg).items()", 1)]
CODE_BY_MODEL = {
    "xgb": [("TUNE_XGB = False", 1), ("spw = n_neg / n_pos", 1),
            ("return XGBClassifier(", 5)],
    "rf": [("RandomForestClassifier(", 4)],
    "enet": [("ENET_ALPHAS = np", 1), ("ENET_L1_RATIO = ", 1),
             ("ENET_MAX_ITER = ", 1), ("ENET_TOL = ", 1),
             ("ENET_N_ITER_NO_CHANGE = ", 1), ("grid = GridSearchCV(", 8),
             ("Pipeline([", 2)],
}


def read_json(path):
    return json.loads((PROJECT_DIR / path).read_text())


def stop(text):
    raise SystemExit(f"\nSTOP: {text}")


def check(passed, text):
    """Print one check, and stop the run if it failed."""
    print(f"  {'PASSED' if passed else 'FAILED'}   {text}")
    if not passed:
        stop(f"a check failed: {text}")


def print_code(path, marker, n_lines=1):
    """Print the lines of a code file that start at the one line holding `marker`."""
    lines = (PROJECT_DIR / path).read_text().splitlines()
    hits = [i for i, line in enumerate(lines) if marker in line]
    if len(hits) != 1:
        stop(f"{path}: expected one line holding '{marker}', found {len(hits)}")
    for i in range(hits[0], hits[0] + n_lines):
        print(f"      {path}:{i + 1}:  {lines[i].strip()}")


def rank_table(results, features):
    """Section 1's table: rank of each measure, mean rank and spread."""
    ranks = {}
    for m in MODELS:
        values = pd.Series({f: abs(float(results[m]["importances"][f]))
                            for f in features})
        ranks[m] = values.rank(ascending=False, method="min").astype(int)
    table = pd.DataFrame(ranks)
    table["mean_rank"] = table[MODELS].mean(axis=1)
    table["spread"] = table[MODELS].max(axis=1) - table[MODELS].min(axis=1)
    return table.sort_values("mean_rank")


def section1_rows():
    """The rank rows of Section 1 of the summary, as text, keyed by feature."""
    lines = (PROJECT_DIR / SUMMARY).read_text().splitlines()
    start = lines.index("## 1. Consensus importance, and the elastic-net coefficients")
    rows = {}
    for line in lines[start:]:
        if line.startswith("**Elastic net"):
            break
        if line.startswith("| `"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            rows[cells[0].strip("`")] = cells[1:]
    return rows


# ===========================================================================
# THE CHECKS THAT TIE EVERYTHING TO THE FITTED SPECIFICATION
# ===========================================================================
def load_and_check():
    print(BAR)
    print("CHECKS — are these the files, and the code, of the fitted specification?")
    print(BAR)
    results, metas, estimators = {}, {}, {}
    for outcome, o in OUTCOMES.items():
        results[outcome] = {m: read_json(o["results"].format(m=m)) for m in MODELS}
        metas[outcome] = {m: read_json(o["meta"].format(m=m)) for m in MODELS}
        built = o["module"].build_models(o["events"], o["rows"] - o["events"])
        estimators[outcome] = built
        for m in MODELS:
            r, meta = results[outcome][m], metas[outcome][m]
            spec = built[m][2]
            where = f"{outcome} {m}"
            check(r["status"] == "OK" and r["n_rows"] == o["rows"]
                  and r["n_events"] == o["events"],
                  f"{where}: status OK, {o['rows']:,} rows, {o['events']:,} events "
                  f"[{o['results'].format(m=m)}]")
            check(spec == r["settings"] and spec == meta["specification"],
                  f"{where}: the code today gives the summary string saved at "
                  f"fitting [{o['code']}]")
            check(len(r["importances"]) == N_FEATURES
                  and meta["n_features"] == N_FEATURES,
                  f"{where}: {N_FEATURES} features [{o['meta'].format(m=m)}]")
            check(meta["reproduces_stage4_1_max_abs_diff"] == 0.0,
                  f"{where}: final fit reproduces the stage 4 importances, maximum "
                  f"difference 0 [{o['meta'].format(m=m)}]")
            versions = meta["library_versions"]
            check(versions["scikit-learn"] == sklearn.__version__
                  and versions["xgboost"] == xgboost.__version__,
                  f"{where}: scikit-learn {sklearn.__version__} and xgboost "
                  f"{xgboost.__version__} installed, as at fitting "
                  f"[{o['meta'].format(m=m)}]")
    return results, metas, estimators


# ===========================================================================
# PART 1 — MODEL SETTINGS
# ===========================================================================
def print_all_params(outcome, model, estimator, booster_config):
    """Every parameter of one estimator, with where the value comes from."""
    o = OUTCOMES[outcome]
    print(f"\n    All parameters:")
    for name, value in estimator.get_params(deep=True).items():
        # Skip entries that hold a whole estimator: their parameters are listed.
        if name == "steps" or hasattr(value, "get_params"):
            continue
        source = f"{o['code']}, build_models"
        if model == "enet" and name == "clf__estimator__alpha":
            value = f"{value} (replaced by the alpha chosen from the grid)"
        if model == "xgb" and value is None:
            value, source = unset_xgb_value(name, booster_config, o["booster"])
        print(f"      {name:<38}{str(value):<36}{source}")


def unset_xgb_value(name, booster_config, booster_file):
    """The value xgboost applied to a parameter the code left unset."""
    gradient_booster = booster_config["gradient_booster"]
    sections = [gradient_booster["tree_train_param"],
                gradient_booster["gbtree_model_param"],
                booster_config["learner_train_param"],
                booster_config["learner_model_param"]]
    for section in sections:
        if name in section:
            return f"{section[name]} (unset, library default)", booster_file
    return "None (unset)", f"xgboost {xgboost.__version__}, not in the saved booster"


def part1(results, metas, estimators):
    print("\n" + BAR)
    print("PART 1 — MODEL SETTINGS")
    print(BAR)
    for outcome, o in OUTCOMES.items():
        booster = xgboost.Booster(model_file=str(PROJECT_DIR / o["booster"]))
        booster_config = json.loads(booster.save_config())["learner"]
        print(f"\n  OUTCOME: {outcome}")
        print("    Common to the three models:")
        for marker, n_lines in CODE_COMMON:
            print_code(o["code"], marker, n_lines)

        for m in MODELS:
            r, meta = results[outcome][m], metas[outcome][m]
            results_file = o["results"].format(m=m)
            meta_file = o["meta"].format(m=m)
            print(f"\n  ---- {outcome} / {m} ({r['name']}) ----")
            print(f"    Fitted on {r['n_rows']:,} rows, {r['n_groups']:,} groups, "
                  f"{r['n_events']:,} positives, input {r['input']} [{results_file}]")
            print(f"    Saved summary: {r['settings']} [{results_file}]")
            print(f"    Grouped {r['n_splits']}-fold cross-fitting by "
                  f"{r['group_name']}, seed {r['random_state']} [{results_file}]")
            print("\n    The fitting code:")
            for marker, n_lines in CODE_BY_MODEL[m]:
                print_code(o["code"], marker, n_lines)

            print_all_params(outcome, m, estimators[outcome][m][1], booster_config)

            # The class weighting, in the form each model takes it.
            print("\n    Class weighting:")
            if m == "xgb":
                weight = (r["n_rows"] - r["n_events"]) / r["n_events"]
                print(f"      scale_pos_weight = negatives / positives = {weight:.6f}"
                      f", from the full sample's counts [{results_file}]")
                print("      One value for all folds: build_models runs once, "
                      "before the folds.")
            else:
                estimator = estimators[outcome][m][1]
                params = estimator.get_params(deep=True)
                setting = (params["class_weight"] if m == "rf"
                           else params["clf__estimator__class_weight"])
                print(f"      class_weight = {setting} [{o['code']}]")

            if m == "enet":
                print_enet_details(outcome, r, meta, results_file, meta_file)


def print_enet_details(outcome, r, meta, results_file, meta_file):
    """The alpha chosen in each fold, and how the features were rescaled."""
    o = OUTCOMES[outcome]
    print("\n    Alpha chosen, by fold:")
    for fold in r["folds"]:
        print(f"      fold {fold['fold']}: alpha {fold['alpha']:g}, "
              f"{fold['n_iter']} iterations, hit cap {fold['hit_cap']} "
              f"[{results_file}]")
    extra = meta["extra"]
    print(f"      full-sample fit: alpha {extra['alpha']:g}, {extra['n_iter']} "
          f"iterations, {extra['n_nonzero']} non-zero, intercept "
          f"{extra['intercept']:+.4f} [{meta_file}]")

    print("\n    Standardization: YES. StandardScaler, z = (x - mean) / sd, applied "
          "to all 27 features,")
    print("    including OENEG and INTWO, inside the pipeline, so it is computed on "
          "the training rows")
    print("    of each fit. The coefficients are therefore per one standard "
          "deviation of the feature.")
    print_code(o["code"], '("scale", StandardScaler())')
    print_code(o["code"], "fit(X[tr], y[tr])")      # each cross-fitting fold
    print_code(o["code"], "fit(X, y)")              # the full-sample importances
    print_code(FINAL_FIT_CODE, "fit(X, y)")         # the final model

    print(f"\n    The scaler of the full-sample fit [{o['coef']}].")
    print("    The scalers of the cross-fitting folds are not saved.")
    coef = pd.read_csv(PROJECT_DIR / o["coef"], float_precision="round_trip")
    print(f"      {'feature':<18}{'mean':>12}{'sd':>12}")
    for row in coef[coef.feature != "_intercept_"].itertuples():
        print(f"      {row.feature:<18}{row.scaler_mean:>12.6f}"
              f"{row.scaler_scale:>12.6f}")


# ===========================================================================
# PART 2 — PARTIAL CORRELATIONS
# ===========================================================================
def part2(metas):
    print("\n" + BAR)
    print("PART 2 — PARTIAL CORRELATIONS, default outcome")
    print(BAR)
    panel = pd.read_parquet(PROJECT_DIR / PANEL)
    features = metas["default"]["xgb"]["features_in_order"]
    id_columns = stage4_1.ID_COLS

    # The feature columns, in the order the models received them.
    check([c for c in panel.columns if c not in id_columns] == features,
          f"the panel's feature columns are the models' list, in order [{PANEL}]")
    check(len(panel) == PANEL_ROWS and int(panel.y.sum()) == PANEL_EVENTS,
          f"panel before the drop: {PANEL_ROWS:,} rows, {PANEL_EVENTS:,} events "
          f"[{PANEL}]")

    # The same drop stage4_1 makes before fitting.
    fitted = panel[~panel[features].isna().any(axis=1)]
    o = OUTCOMES["default"]
    check(len(fitted) == o["rows"] and int(fitted.y.sum()) == o["events"]
          and fitted.gvkey.nunique() == FITTED_FIRMS
          and fitted.issuer_id.nunique() == FITTED_ISSUERS,
          f"after the drop: {len(fitted):,} rows, {fitted.gvkey.nunique():,} gvkeys, "
          f"{fitted.issuer_id.nunique():,} issuers, {int(fitted.y.sum()):,} events "
          f"[{PANEL}]")

    # Row for row, these are the rows that were scored, with the same outcome.
    scores = pd.read_parquet(PROJECT_DIR / SCORES)
    same_rows = all((scores[c].to_numpy() == fitted[c].to_numpy()).all()
                    for c in ["issuer_id", "gvkey", "fyear", "y"])
    check(same_rows, f"row for row, the same rows and outcome as [{SCORES}]")

    # The saved scaler describes exactly these rows.
    coef = pd.read_csv(PROJECT_DIR / o["coef"], float_precision="round_trip")
    coef = coef[coef.feature != "_intercept_"].set_index("feature")
    gap = max((fitted[features].mean() - coef.scaler_mean).abs().max(),
              (fitted[features].std(ddof=0) - coef.scaler_scale).abs().max())
    check(gap < TOLERANCE, f"the saved scaler's means and sds are these rows' own, "
          f"largest difference {gap:.1e} [{o['coef']}]")

    y = fitted.y.to_numpy(dtype=float)
    X = fitted[features].to_numpy(dtype=float)

    # A second route to the same number: the inverse of the correlation matrix of
    # the 27 features and the outcome. It shares no step with the residuals below.
    inverse = np.linalg.inv(np.corrcoef(np.column_stack([X, y]), rowvar=False))
    y_position = len(features)

    print(f"\n  Sample: {len(fitted):,} rows, {fitted.gvkey.nunique():,} gvkeys, "
          f"{fitted.issuer_id.nunique():,} issuers, {int(y.sum()):,} events "
          f"[{PANEL}]")
    print("  Each OLS: constant plus the other 26 features.")
    print(f"\n  {'feature':<14}{'simple corr':>13}{'partial corr':>14}"
          f"{'R2, feature on 26':>19}{'R2, outcome on 26':>19}")
    for feature in FEATURES_TO_TEST:
        position = features.index(feature)
        x = X[:, position]
        controls = np.column_stack([np.ones(len(y)), np.delete(X, position, axis=1)])

        # OLS of the feature, and of the outcome, on the same 26 features.
        x_coefs, _, x_rank, _ = np.linalg.lstsq(controls, x, rcond=None)
        y_coefs, _, y_rank, _ = np.linalg.lstsq(controls, y, rcond=None)
        x_residual = x - controls @ x_coefs
        y_residual = y - controls @ y_coefs

        partial = np.corrcoef(x_residual, y_residual)[0, 1]
        simple = np.corrcoef(x, y)[0, 1]
        x_r2 = 1 - x_residual.var() / x.var()
        y_r2 = 1 - y_residual.var() / y.var()
        print(f"  {feature:<14}{simple:>+13.5f}{partial:>+14.5f}"
              f"{x_r2:>19.4f}{y_r2:>19.4f}   [{PANEL}]")

        # The checks for this feature.
        other_route = (-inverse[position, y_position]
                       / np.sqrt(inverse[position, position]
                                 * inverse[y_position, y_position]))
        check(abs(partial - other_route) < TOLERANCE,
              f"{feature}: the matrix-inverse route gives {other_route:+.5f}")
        check(x_rank == 27 and y_rank == 27,
              f"{feature}: the controls have full rank, 27 columns")


# ===========================================================================
# PART 3 — THE RATING SCORE
# ===========================================================================
def part3(results, metas):
    print("\n" + BAR)
    print("PART 3 — RATING WEAKNESS SCORE, from the saved rating results")
    print(BAR)
    o = OUTCOMES["rating"]
    features = metas["rating"]["xgb"]["features_in_order"]
    rating = results["rating"]
    files = ", ".join(o["results"].format(m=m) for m in MODELS)

    # The same function must give Section 1 of the summary on the default outcome,
    # so the rating table below is built the way that table was.
    default_table = rank_table(results["default"],
                               metas["default"]["xgb"]["features_in_order"])
    recomputed = {f: [str(int(row.xgb)), str(int(row.rf)), str(int(row.enet)),
                      f"{row.mean_rank:.1f}", str(int(row.spread))]
                  for f, row in default_table.iterrows()}
    check(recomputed == section1_rows(),
          f"the ranking code gives all 27 rows of Section 1 of [{SUMMARY}]")

    # The rating coefficients in the results file and in the saved coefficient file.
    enet = rating["enet"]
    coefs = pd.Series({f: float(enet["importances"][f]) for f in features})
    saved = pd.read_csv(PROJECT_DIR / o["coef"], float_precision="round_trip")
    intercept = saved.loc[saved.feature == "_intercept_", "coefficient"].iloc[0]
    saved = saved[saved.feature != "_intercept_"].set_index("feature").coefficient
    gap = (coefs - saved.reindex(features)).abs().max()
    check(gap < TOLERANCE, f"rating coefficients agree between the two files, largest "
          f"difference {gap:.1e} [{o['results'].format(m='enet')}, {o['coef']}]")

    zeros = sorted(f for f in features if coefs[f] == 0)
    check(N_FEATURES - len(zeros) == enet["extra"]["n_nonzero"],
          f"{N_FEATURES - len(zeros)} non-zero coefficients, as recorded "
          f"[{o['results'].format(m='enet')}]")

    # ---- the table, in Section 1's layout, plus the signed coefficient ------
    table = rank_table(rating, features)
    print(f"\n  Sample: {enet['n_rows']:,} rows, {enet['n_events']:,} positives "
          f"[{o['results'].format(m='enet')}]")
    print(f"  Ranks and coefficients: one full-sample fit each, not cross-fitted.")
    print(f"  Source: {files}\n")
    print("| feature | xgb | rf | enet | mean rank | spread | enet coefficient |")
    print("|---|---|---|---|---|---|---|")
    for f, row in table.iterrows():
        print(f"| `{f}` | {int(row.xgb)} | {int(row.rf)} | {int(row.enet)} | "
              f"{row.mean_rank:.1f} | {int(row.spread)} | {coefs[f]:+.4f} |")

    # ---- the coefficients, largest first, as in Section 1 -------------------
    print(f"\n**Elastic net.** alpha {enet['extra']['alpha']:.4g} "
          f"[{o['results'].format(m='enet')}]; intercept {intercept:+.4f} "
          f"[{o['coef']}]; **{N_FEATURES - len(zeros)} of {N_FEATURES} coefficients "
          "non-zero**.\n")
    print("| feature | coefficient |")
    print("|---|---|")
    non_zero = coefs[coefs != 0]
    for f in non_zero.abs().sort_values(ascending=False).index:
        print(f"| `{f}` | {coefs[f]:+.4f} |")

    # ---- the top ten of both tree models ------------------------------------
    both = table[(table.xgb <= TOP_N) & (table.rf <= TOP_N)].sort_values("xgb")
    print(f"\n**In the top {TOP_N} of both xgb and rf: {len(both)} features.** "
          f"[{o['results'].format(m='xgb')}, {o['results'].format(m='rf')}, "
          f"{o['results'].format(m='enet')}]\n")
    print("| feature | xgb | rf | enet coefficient |")
    print("|---|---|---|---|")
    for f, row in both.iterrows():
        print(f"| `{f}` | {int(row.xgb)} | {int(row.rf)} | {coefs[f]:+.4f} |")

    # ---- the zeros ----------------------------------------------------------
    print(f"\n**Coefficients equal to zero: {len(zeros)}.** "
          f"{', '.join('`' + z + '`' for z in zeros) or 'none'} "
          f"[{o['results'].format(m='enet')}]")


def main():
    results, metas, estimators = load_and_check()
    part1(results, metas, estimators)
    part2(metas)
    part3(results, metas)
    print("\n" + BAR)
    print("END. Nothing was fitted and nothing was written.")
    print(BAR)


if __name__ == "__main__":
    main()
