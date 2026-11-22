"""
Stage 4.2 — the RATING outcome: model classes compared on one prepared sample.

THIS IS VALIDATION, NOT A SECOND MODEL OF DEFAULT. D-39 bars ratings as a
training label: a strong result here says the features track agency opinion, it
does NOT say the weakness score predicts default. The default model is stage4_1.

Reads output/stage4_0_rating_{SAMPLE_LAG}_K{K}.parquet (cut, imputed, ranked,
windowed) and
fits each model class on IDENTICAL rows, so a difference in the table is
attributable to the model and not to the data (D-45).

The base rate here is ~44%, not 1.2%, so AUC-ROC is the headline and AUC-PR is
reported alongside — the reverse of stage4_1.

Scores are CROSS-FITTED: every row's prediction comes from a model that never
saw that firm (D-27, BACKGROUND section 2). "The same firm" is GROUP_MODE, and
the default is the strict one — grouping by issuer_id alone was MEASURED to be
leaky on this panel. Predictions are then RECALIBRATED out of fold, because
class weighting puts them on a reweighted scale, not the base-rate scale (D-47).

ONE MODEL AT A TIME. Each model is fitted, evaluated, printed and SAVED before
the next begins, so a later failure can never destroy completed work. A model
that fails saves a FAILED record with its diagnostics and the loop continues.
The comparison table is assembled by READING those files.

Writes, per model:
  output/stage4_2_results_{model}.json      metrics, settings, grouping key
  output/stage4_2_scores_{model}.parquet    cross-fitted predictions

Run:  python3 code/stage4_2_rating_models.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"


# ===========================================================================
# CONFIG
# ===========================================================================

SAMPLE_LAG = "L3"                   # must match the prepared file
K = 12

# A label for a FEATURE-SET EXPERIMENT. Empty means the main run and the
# filenames are unchanged, so nothing existing is ever replaced. Set it to a
# short name and every output of this stage carries that suffix, letting an
# alternative feature set coexist with the main one instead of overwriting it.
SET_TAG = ""
_T = f"_{SET_TAG}" if SET_TAG else ""

# The INPUT panel's tag, when it differs from the OUTPUT tag (a tuning run reads
# a feature set's panel but must not overwrite that set's default results).
INPUT_SET_TAG = None            # None -> same as SET_TAG
_TI = _T if INPUT_SET_TAG is None else (f"_{INPUT_SET_TAG}" if INPUT_SET_TAG else "")

# --- XGBoost hyperparameter search -----------------------------------------
# Same structure as stage4_1: a GridSearchCV INSIDE each outer training fold,
# mirroring what the elastic net already does for its alpha. RF and enet are
# untouched. NO EARLY STOPPING — sklearn's carves out a random internal split
# that would let a firm's rows straddle it.
#
# SCORING IS roc_auc HERE, NOT average_precision. The rating outcome runs at a
# ~44% base rate against distress's 1.2%, so the rare-event metric is not the
# right selection criterion for it.
#
# KNOWN AND ACCEPTED LIMITATION: the inner CV is NOT grouped. Outer folds are
# component-grouped so every REPORTED metric is clean, but hyperparameters are
# selected on an inner split that permits a firm's rows to straddle.
TUNE_XGB = False
XGB_SCORING = "roc_auc"
XGB_GRID = {
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [3, 5, 10],
    "learning_rate": [0.03, 0.05, 0.1],
}
# n_estimators, subsample and colsample_bytree are NOT searched.

INPUT_FILE = OUT_DIR / f"stage4_0_rating_{SAMPLE_LAG}_K{K}{_TI}.parquet"
# Read-only, and only for the Tax_negden separation measure below.
DISTRESS_FILE = OUT_DIR / f"stage4_0_distress_{SAMPLE_LAG}_K{K}{_TI}.parquet"

OUTCOME = "spec_grade"
ID_COLS = ["issuer_id", "gvkey", "fyear", "datadate", "date_avail",
           "spec_grade"]

# What defines "the same firm" for cross-fitting. MEASURED on THIS sample, not
# assumed: grouping by issuer_id alone is LEAKY here too — 634 gvkeys map to
# more than one issuer_id, 43.1% of rows sit on such a gvkey, and 30.4% of rows
# are literal duplicate statements (same gvkey, same datadate) under different
# issuer_ids. (The distress sample's figures are 911 / 48.9% / 43.3%; see D-48.)
#   "component"  connected components of the issuer_id <-> gvkey graph. The
#                strictest, and the default.
#   "gvkey"      the accounting entity; still splits the 36 issuers that map to
#                several gvkeys.
#   "issuer_id"  KNOWN LEAKY. Kept only for comparison.
GROUP_MODE = "component"

# Fit order is deliberate and permanent: trees are quick, the penalised logit is
# slow, and there is no reason to wait on it to see the tree results.
FIT_MODELS = ["xgb", "rf", "enet"]
# A model whose results are already on disk is not refitted. Set True to force.
REFIT = False

N_SPLITS = 5
RANDOM_STATE = 42
N_JOBS = 4                          # bounded on purpose; this is a laptop
N_DECILES = 10

# The loss-state flag whose univariate separation is reported. INTWO since D-50.
SEPARATION_FLAG = "INTWO"

# Out-of-fold calibration (D-47). Both are computed and both reported.
#   "sigmoid"   Platt. Balanced weighting shifts the intercept in LOG-ODDS by
#               log(w), which a sigmoid correction removes exactly; also the
#               stable choice with only ~1,200 events.
#   "isotonic"  flexible and monotone, but can overfit the sparse upper tail.
CALIBRATION_HEADLINE = "sigmoid"
CALIBRATION_FOLDS = 5

# --- the penalised logit: a SPECIFICATION change, not tuning ---------------
# `saga` (LogisticRegressionCV) failed to converge at EVERY penalty strength,
# including C = 1e-4, the strongest in its grid, on a design matrix that is
# well conditioned: condition number 12.4, full rank 31 of 31, max VIF 16.6.
# The optimiser is the problem, not the data or the grid. SGDClassifier keeps
# the model class D-27 specifies — elastic-net penalised logistic regression —
# and changes only how it is fitted.
#
# early_stopping stays FALSE deliberately: scikit-learn carves out a RANDOM
# internal validation split, which would put a firm's own rows on both sides
# and reintroduce exactly the leak the grouped CV exists to prevent.
#
# "Convergence" now means a stalled training loss rather than a closed duality
# gap. The same hard guard applies: n_iter_ >= max_iter stops the model.
ENET_ALPHAS = np.logspace(-6, -2, 5)
ENET_L1_RATIO = 0.5
ENET_MAX_ITER = 5000
ENET_TOL = 1e-4
ENET_N_ITER_NO_CHANGE = 10
STOP_ON_NONCONVERGENCE = True

# The five non-continuous features, reported SEPARATELY from the other 26:
# they are ours, not CZ's, and they never faced the Stage 2 redundancy screen
# (D-17, D-38). BookLeverage_negden and RoE_negden correlate at 0.966.
# D-50 retired the four `_negden` flags and NumEarnIncrease left the model at
# ALWAYS_EXCLUDE; OENEG and INTWO (Ohlson 1980) are the remaining indicators.
# Only names actually present are reported, so a changed feature set cannot
# crash the run again.
NON_CONTINUOUS = ["OENEG", "INTWO"]

# ===========================================================================
# END CONFIG
# ===========================================================================


def results_path(key):
    return OUT_DIR / f"stage4_2_results{_T}_{key}.json"


def scores_path(key):
    return OUT_DIR / f"stage4_2_scores{_T}_{key}.parquet"


def _xgb_base(spw, n_jobs):
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
        n_jobs=n_jobs, random_state=RANDOM_STATE)


def _xgb_estimator(spw):
    if not TUNE_XGB:
        return _xgb_base(spw, N_JOBS)
    return GridSearchCV(_xgb_base(spw, 1), XGB_GRID, cv=3,
                        scoring=XGB_SCORING, n_jobs=N_JOBS)


def _xgb_spec(spw):
    base = f"400 trees, subsample 0.8, colsample 0.8, scale_pos_weight={spw:.1f}"
    if not TUNE_XGB:
        return f"depth 4, lr 0.05, min_child_weight 5, {base}"
    return (f"TUNED by inner GridSearchCV (cv=3, scoring={XGB_SCORING}, "
            f"UNGROUPED inner split) over {XGB_GRID}; {base}")


def build_models(n_pos, n_neg):
    """The model classes, with every default stated in the output."""
    spw = n_neg / n_pos
    grid = GridSearchCV(
        SGDClassifier(loss="log_loss", penalty="elasticnet",
                      l1_ratio=ENET_L1_RATIO, class_weight="balanced",
                      learning_rate="optimal", max_iter=ENET_MAX_ITER,
                      tol=ENET_TOL, n_iter_no_change=ENET_N_ITER_NO_CHANGE,
                      early_stopping=False, random_state=RANDOM_STATE),
        {"alpha": ENET_ALPHAS}, cv=3, scoring="average_precision",
        n_jobs=N_JOBS)
    return {k: v for k, v in {
        "xgb": ("XGBoost", _xgb_estimator(spw), _xgb_spec(spw)),

        "rf": ("random forest", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=N_JOBS,
            random_state=RANDOM_STATE),
            "500 trees, min_samples_leaf 20, max_features sqrt, "
            "class_weight balanced_subsample"),

        "enet": ("elastic-net logistic, SGD (D-27)", Pipeline([
            ("scale", StandardScaler()), ("clf", grid)]),
            f"scaler + SGD log_loss elasticnet, l1_ratio={ENET_L1_RATIO}, "
            f"alpha in logspace(-6,-2,5) by inner cv=3, learning_rate=optimal, "
            f"max_iter={ENET_MAX_ITER}, tol={ENET_TOL}, early_stopping=False, "
            f"class_weight=balanced"),
    }.items() if k in FIT_MODELS}


def build_groups(d):
    """The cross-fitting group. See GROUP_MODE for why the default is strict."""
    if GROUP_MODE == "issuer_id":
        return d.issuer_id.to_numpy(), "issuer_id (KNOWN LEAKY)"
    if GROUP_MODE == "gvkey":
        return d.gvkey.to_numpy(), "gvkey"
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ic = d.issuer_id.astype("category").cat.codes.to_numpy()
    gc = d.gvkey.astype("category").cat.codes.to_numpy()
    n_i = ic.max() + 1
    size = n_i + gc.max() + 1
    m = coo_matrix((np.ones(len(d)), (ic, gc + n_i)), shape=(size, size))
    _, lab = connected_components(m + m.T, directed=False)
    return lab[ic], "connected components of issuer_id <-> gvkey"


def cross_fit(est, X, y, groups, cv, key):
    """Out-of-fold predictions, one fold at a time.

    Written out rather than using cross_val_predict so each fold's fitted
    estimator can be inspected — specifically, so a non-converged penalised
    logit is caught instead of silently reaching a results table.
    """
    p = np.full(len(y), np.nan)
    folds = []
    for k, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        est.fit(X[tr], y[tr])
        p[te] = est.predict_proba(X[te])[:, 1]
        if key == "xgb" and TUNE_XGB:
            folds.append({"fold": k, **est.best_params_,
                          "inner_score": float(est.best_score_)})
            print(f"      fold {k}: {est.best_params_}  "
                  f"inner {XGB_SCORING}={est.best_score_:.4f}", flush=True)
        if key == "enet":
            best = est.named_steps["clf"].best_estimator_
            it = int(np.max(best.n_iter_))
            capped = it >= ENET_MAX_ITER
            folds.append({"fold": k, "alpha": float(best.alpha),
                          "n_iter": it, "hit_cap": bool(capped)})
            print(f"      fold {k}: alpha={best.alpha:.3g}  n_iter={it:,}"
                  f"{'   <-- HIT THE CAP' if capped else ''}", flush=True)
    if np.isnan(p).any():
        raise RuntimeError("some rows never appeared in a test fold")
    capped = [f for f in folds if f.get("hit_cap")]
    if capped and STOP_ON_NONCONVERGENCE:
        raise RuntimeError(
            f"did not converge: hit max_iter={ENET_MAX_ITER:,} in "
            f"{len(capped)} of {len(folds)} folds; per-fold {folds}. "
            "A non-converged fit must not reach a results table.")
    return p, folds


def calibrate_oof(p_raw, y, groups, method):
    """Recalibrate the cross-fitted scores, OUT OF FOLD.

    The calibrator is fitted on other folds and applied to the held-out rows, so
    it never sees the rows it calibrates, and the folds are grouped the same way
    so a firm cannot leak into its own calibration.
    """
    out = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=CALIBRATION_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
    eps = 1e-9
    clip = lambda v: np.clip(v, eps, 1 - eps)
    for tr, te in cv.split(p_raw.reshape(-1, 1), y, groups):
        if method == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            m.fit(p_raw[tr], y[tr])
            out[te] = m.predict(p_raw[te])
        else:                                     # Platt, on the log-odds
            lo = lambda v: np.log(clip(v) / (1 - clip(v)))
            m = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            m.fit(lo(p_raw[tr]).reshape(-1, 1), y[tr])
            out[te] = m.predict_proba(lo(p_raw[te]).reshape(-1, 1))[:, 1]
    return clip(out)


def recall_at_top(y, p, frac):
    """Share of all events caught in the top `frac` of scores."""
    k = max(1, int(round(len(p) * frac)))
    return float(y[np.argsort(-p)[:k]].sum() / y.sum())


def evaluate(y, p_raw, p_cal):
    """Every metric in the table. Accuracy is never among them."""
    return {
        "AUC-PR": float(average_precision_score(y, p_raw)),
        "AUC-ROC": float(roc_auc_score(y, p_raw)),
        "Brier": float(brier_score_loss(y, p_raw)),
        "logloss": float(log_loss(y, p_raw, labels=[0, 1])),
        "Brier_cal": float(brier_score_loss(y, p_cal)),
        "logloss_cal": float(log_loss(y, p_cal, labels=[0, 1])),
        "rec@1%": recall_at_top(y, p_raw, 0.01),
        "rec@5%": recall_at_top(y, p_raw, 0.05),
        "rec@10%": recall_at_top(y, p_raw, 0.10),
    }


def decile_table(y, p, label):
    """Predicted against realised, by decile of score."""
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d.p.rank(method="first"), N_DECILES,
                          labels=range(1, N_DECILES + 1))
    g = d.groupby("decile", observed=True).agg(
        n=("y", "size"), events=("y", "sum"),
        pred=("p", "mean"), actual=("y", "mean"))
    print(f"\n    DECILE CALIBRATION — {label}")
    print(f"      {'decile':<8}{'n':>9}{'events':>9}{'predicted':>12}"
          f"{'actual':>10}{'ratio':>9}")
    for i, r in g.iterrows():
        ratio = f"{r.pred / r.actual:>9.2f}" if r.actual > 0 else f"{'n/a':>9}"
        print(f"      {int(i):<8}{int(r.n):>9,}{int(r.events):>9,}"
              f"{r.pred:>12.5f}{r.actual:>10.5f}{ratio}")


def importances(key, est, X, y, feats):
    """Descriptive only: ONE full-sample fit, never cross-fitted.

    Every metric elsewhere comes from cross-fitted predictions. This does not,
    and it is labelled as such wherever it is printed.
    """
    est.fit(X, y)
    if key == "enet":
        clf = est.named_steps["clf"].best_estimator_
        v = pd.Series(clf.coef_[0], index=feats)
        extra = {"alpha": float(clf.alpha), "n_iter": int(np.max(clf.n_iter_)),
                 "n_nonzero": int((v != 0).sum())}
        return v, extra, "coefficient"
    if key == "xgb" and TUNE_XGB:
        best = est.best_estimator_
        return (pd.Series(best.feature_importances_, index=feats),
                {"best_params": est.best_params_}, "importance")
    v = pd.Series(est.feature_importances_, index=feats)
    return v, {}, "importance"


def report_indicators(v, kind, key):
    """The five hand-made non-continuous features, called out separately.

    They are ours, not CZ's, and they never faced the Stage 2 redundancy screen
    that the 51 CZ characteristics did (D-17, D-38). Their rank here is the
    evidence for whether they earn their place.
    """
    rank = v.abs().rank(ascending=False).astype(int)
    present = [f for f in NON_CONTINUOUS if f in v.index]
    print(f"\n    THE {len(present)} NON-CONTINUOUS FEATURES ({key}) — ours, "
          "not CZ's (D-50)")
    print(f"      {'feature':<24}{kind:>14}{'rank of ' + str(len(v)):>13}")
    for f in present:
        print(f"      {f:<24}{v[f]:>+14.5f}{rank[f]:>13}")
    others = [f for f in v.index if f not in present]
    print(f"      — for contrast, the top 5 of the other {len(others)}:")
    for f, val in v[others].abs().nlargest(5).items():
        print(f"      {f:<24}{v[f]:>+14.5f}{rank[f]:>13}")
    return {f: {"value": float(v[f]), "rank": int(rank[f])} for f in present}


def flag_separation():
    """How much of the story is one plain loss flag?

    SEPARATION_FLAG is INTWO since D-50 — Ohlson's two consecutive loss years,
    which replaced Tax_negden (ib <= 0, a single year). The share of positives
    against negatives carrying it is worth stating outright rather than leaving
    implicit behind 29 features. -1 is the "cannot be determined" code (D-44).
    """
    print("\n" + "=" * 92)
    print(f"{SEPARATION_FLAG} — the loss-state flag. How far does it separate "
          "the classes on its own?")
    print("=" * 92)
    for path, outcome, lab in [(DISTRESS_FILE, "y", "DISTRESS (y=1)"),
                               (INPUT_FILE, "spec_grade", "RATING (spec_grade=1)")]:
        if not path.exists():
            print(f"  {lab}: {path.name} not on disk, skipped")
            continue
        d = pd.read_parquet(path, columns=[outcome, SEPARATION_FLAG])
        pos, neg = d[d[outcome] == 1], d[d[outcome] == 0]
        print(f"\n  {lab}   positives {len(pos):,}   negatives {len(neg):,}")
        print(f"      {SEPARATION_FLAG:<14}{'positives':>14}{'negatives':>14}"
              f"{'difference':>13}")
        for v, name in [(1, "1  loss"), (0, "0  profit"),
                        (-1, "-1 unknown")]:
            a = (pos[SEPARATION_FLAG] == v).mean() * 100
            b = (neg[SEPARATION_FLAG] == v).mean() * 100
            print(f"      {name:<14}{a:>13.2f}%{b:>13.2f}%{a-b:>+12.2f}pp")
        a = (pos[SEPARATION_FLAG] == 1).mean()
        b = (neg[SEPARATION_FLAG] == 1).mean()
        odds = (a / (1 - a)) / (b / (1 - b)) if 0 < b < 1 and a < 1 else float("nan")
        print(f"      univariate odds ratio for a loss year: {odds:.2f}")


def fit_one(key, name, est, spec, X, y, groups, group_name, cv, d, feats):
    """Fit, evaluate, print and SAVE one model. Failure is recorded, not fatal."""
    print("\n" + "=" * 92)
    print(f"MODEL {key} — {name}")
    print("=" * 92)
    print(f"  grouping: {group_name}   defaults: {spec}")
    meta = {"model": key, "name": name, "tag": SET_TAG, "settings": spec,
            "group_mode": GROUP_MODE, "group_name": group_name,
            "n_rows": int(len(d)), "n_groups": int(len(np.unique(groups))),
            "n_events": int(y.sum()), "base_rate": float(y.mean()),
            "n_splits": N_SPLITS, "random_state": RANDOM_STATE,
            "input": INPUT_FILE.name, "when": datetime.now().isoformat(timespec="seconds")}
    try:
        p_raw, folds = cross_fit(est, X, y, groups, cv, key)
    except RuntimeError as e:
        meta.update({"status": "FAILED", "error": str(e)})
        results_path(key).write_text(json.dumps(meta, indent=2))
        print(f"  FAILED: {e}")
        print(f"  recorded in {results_path(key).name}; other models continue")
        return
    cal = {m: calibrate_oof(p_raw, y, groups, m) for m in ("sigmoid", "isotonic")}
    m = evaluate(y, p_raw, cal[CALIBRATION_HEADLINE])
    m["Brier_isotonic"] = float(brier_score_loss(y, cal["isotonic"]))

    # AUC-ROC leads here: at a ~44% base rate it is adequate, and AUC-PR is
    # not comparable with stage4_1's, whose base rate is 1.2% (D-45).
    print(f"\n    AUC-ROC {m['AUC-ROC']:.4f}   AUC-PR {m['AUC-PR']:.4f}   "
          f"base rate {y.mean()*100:.3f}%")
    print(f"    Brier  raw {m['Brier']:.5f} -> {CALIBRATION_HEADLINE} "
          f"{m['Brier_cal']:.5f} (isotonic {m['Brier_isotonic']:.5f})")
    print(f"    logloss raw {m['logloss']:.4f} -> {m['logloss_cal']:.4f}")
    print(f"    recall @1% {m['rec@1%']:.3f}  @5% {m['rec@5%']:.3f}  "
          f"@10% {m['rec@10%']:.3f}")
    decile_table(y, p_raw, f"{key} RAW (reweighted scale)")
    decile_table(y, cal[CALIBRATION_HEADLINE],
                 f"{key} CALIBRATED out-of-fold, {CALIBRATION_HEADLINE}")

    print(f"\n    full-sample fit for importances (DESCRIPTIVE, not "
          f"cross-fitted) ...", flush=True)
    v, extra, kind = importances(key, est, X, y, feats)
    ind = report_indicators(v, kind, key)

    meta.update({"status": "OK", "metrics": m, "folds": folds,
                 "indicators": ind, "extra": extra,
                 "importance_kind": kind,
                 "importances": {f: float(x) for f, x in v.items()}})
    results_path(key).write_text(json.dumps(meta, indent=2))
    out = d[["issuer_id", "gvkey", "fyear"]].copy()
    out["y"] = y
    out["score_raw"] = p_raw
    out["score_calibrated"] = cal[CALIBRATION_HEADLINE]
    out.to_parquet(scores_path(key), index=False)
    print(f"\n  saved {results_path(key).name} and {scores_path(key).name}")


def comparison_table():
    """Assembled by READING the per-model files, so it prints whatever exists."""
    print("\n" + "=" * 92)
    print("COMPARISON — assembled from the saved per-model results")
    print("=" * 92)
    found = []
    for key in FIT_MODELS:
        if results_path(key).exists():
            found.append(json.loads(results_path(key).read_text()))
    if not found:
        print("  nothing on disk yet.")
        return
    gm = {r["group_mode"] for r in found}
    print(f"  grouping: {', '.join(sorted(gm))}"
          f"{'   <-- MIXED, NOT COMPARABLE' if len(gm) > 1 else ''}")
    print(f"  {'model':<7}{'AUC-ROC':>9}{'AUC-PR':>9}{'Brier':>9}{'Brier*':>9}"
          f"{'logloss':>9}{'logloss*':>10}{'rec@1%':>8}{'rec@5%':>8}"
          f"{'rec@10%':>9}{'n':>10}{'groups':>8}{'events':>8}")
    for r in found:
        if r["status"] != "OK":
            print(f"  {r['model']:<7}{'FAILED':>9}   {r['error'][:70]}")
            continue
        m = r["metrics"]
        print(f"  {r['model']:<7}{m['AUC-ROC']:>9.4f}{m['AUC-PR']:>9.4f}"
              f"{m['Brier']:>9.5f}{m['Brier_cal']:>9.5f}{m['logloss']:>9.4f}"
              f"{m['logloss_cal']:>10.4f}{m['rec@1%']:>8.3f}{m['rec@5%']:>8.3f}"
              f"{m['rec@10%']:>9.3f}{r['n_rows']:>10,}{r['n_groups']:>8,}"
              f"{r['n_events']:>8,}")
    print(f"  * = after out-of-fold {CALIBRATION_HEADLINE} calibration (D-47). "
          "AUC-PR and AUC-ROC are rank statistics and do not change.")


def main():
    print("=" * 92)
    print(f"STAGE 4.2 — RATING MODELS   input {INPUT_FILE.name}   "
          f"GROUP_MODE={GROUP_MODE}")
    print("  VALIDATION, NOT A SECOND MODEL OF DEFAULT (D-39). A strong result "
          "here means the")
    print("  features track agency opinion; it says nothing about predicting "
          "default.")
    print("=" * 92)
    df = pd.read_parquet(INPUT_FILE)
    feats = [c for c in df.columns if c not in ID_COLS]
    na = df[feats].isna().any(axis=1)
    pos = df[OUTCOME] == 1
    print(f"  rows {len(df):,}   issuers {df.issuer_id.nunique():,}   "
          f"features {len(feats)}   fyear {int(df.fyear.min())}-"
          f"{int(df.fyear.max())}")
    print(f"  dropped for remaining NA: {int(na.sum()):,} rows "
          f"({int((na & pos).sum()):,} of {int(pos.sum()):,} events)")

    d = df[~na].copy()
    groups, group_name = build_groups(d)
    y = d[OUTCOME].to_numpy().astype(int)
    X = d[feats].to_numpy(dtype="float64")
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"  FITTED ON: rows {len(d):,}   issuers {d.issuer_id.nunique():,}   "
          f"events {n_pos:,} ({n_pos/len(d)*100:.3f}%)")
    print(f"  grouped by {group_name}: {len(np.unique(groups)):,} groups, "
          f"largest {np.bincount(pd.factorize(groups)[0]).max():,} rows")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=RANDOM_STATE)
    for key, (name, est, spec) in build_models(n_pos, n_neg).items():
        if results_path(key).exists() and not REFIT:
            print(f"\n  {key}: results already on disk "
                  f"({results_path(key).name}), skipped. REFIT=True to refit.")
            continue
        fit_one(key, name, est, spec, X, y, groups, group_name, cv, d, feats)

    comparison_table()
    flag_separation()
    print("\nVALIDATION ONLY. D-39 bars ratings as a training label: this says "
          "the features track agency opinion, not that they predict default.")


if __name__ == "__main__":
    main()
