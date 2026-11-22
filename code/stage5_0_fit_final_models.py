"""
Stage 5.0 — Fit each model ONCE on the full US sample and SAVE the fitted object.

WHY THIS EXISTS. Nothing else on disk holds a usable model. stage4_1 evaluates
by cross-fitting: it fits five fold-models, takes their out-of-fold predictions
and DISCARDS the models. That is correct for evaluation and useless for
transfer, because there is no object left to carry to China. This script closes
that gap and does nothing else.

===========================================================================
!!  THESE MODELS MUST NEVER BE USED TO SCORE A US FIRM.                  !!
!!                                                                       !!
!!  Every one is fitted on ALL 95,290 rows. Each has seen every firm's   !!
!!  own realised outcome, so a US score from it would carry the outcome  !!
!!  directly into the second-stage regression and bias the coefficient   !!
!!  the wider study exists to estimate (BACKGROUND section 2).           !!
!!                                                                       !!
!!  US scores come from the CROSS-FITTED predictions of stage4_1        !!
!!  (distress) or stage4_2 (rating), assembled by stage5_1_score_panel. !!
!!  These objects exist ONLY to be applied to CHINESE data, where no     !!
!!  row was in the training sample and the leak cannot arise.            !!
===========================================================================

WHICH OUTCOME. `OUTCOME_SET` selects it. "distress" is the default and behaves
exactly as this script did before the switch existed — same input, same
filenames, same metadata. "rating" fits the same three classes on the US rating
panel using stage4_2's specification, and every output it writes carries
"rating" in its filename, so the two can never be confused on disk.

WHERE THE SPECIFICATIONS COME FROM. They are IMPORTED from stage4_1 or stage4_2,
not retyped: `build_models` is called with the same (n_pos, n_neg), so the
hyperparameters, the class weighting and random_state=42 are identical BY
CONSTRUCTION — including `scale_pos_weight`, which each outcome derives from its
own class balance. D-55 records six files that went stale because they carried
their own copies of a shared list; this script refuses to become the seventh.
The guard at the end proves the specification has not drifted: the cross-fit
script's saved `importances` block is itself a full-sample descriptive fit at
the same seed, so this fit must reproduce it to tolerance or stop.

WHAT TRAVELS TO CHINA IS THE COEFFICIENT VECTOR, NOT THE PICKLE. BACKGROUND
section 7 transfers coefficients, so for the elastic net this script also writes
a plain CSV holding every coefficient together with the scaler's mean and scale
per feature. That is the complete, portable, human-checkable description of the
fitted linear model, and unlike a .joblib it cannot be broken by a scikit-learn
upgrade.

Run:  python3 code/stage5_0_fit_final_models.py                 (distress)
      python3 code/stage5_0_fit_final_models.py --outcome rating
"""

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

# Every script here is run as `python3 code/stage5_0_fit_final_models.py` from
# the PROJECT ROOT, and from there code/ is not on sys.path — so importing a
# sibling module fails with ModuleNotFoundError unless we put code/ on the path
# first. This line is what makes the two imports below work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The specifications themselves, imported rather than copied.
# CHECKED, not assumed: both modules were parsed for module-level statements
# before this was written. NEITHER has any side effect at import time — the only
# module-level calls in either are `Path(...)` and `list(...)`, and all real work
# sits under `if __name__ == "__main__"`. So importing both is safe, and the AST
# parsing an earlier draft used for stage3_10 was unnecessary complexity. It was
# also WRONG: it would have recorded IMPUTE_EXEMPT as the source text
# "list(RANK_EXEMPT)" rather than its value.
# stage3_10 is imported as a MODULE, not with `from`, because it defines K and
# SAMPLE_LAG too and a bare import would shadow stage4_1's.
import stage3_10_prepare_model_input as s310

# WHICH OUTCOME. "distress" is the default and reproduces this script exactly as
# it was before the switch existed — same input, same filenames, same metadata.
# "rating" fits the same three model classes on the US rating panel instead,
# using stage4_2's specification. stage4_2 exports the same seven names as
# stage4_1 and, like it, does all real work under `if __name__ == "__main__"`,
# so either import is safe and neither runs anything.
# SELECTED ON THE COMMAND LINE, not by editing this file:
#     python3 code/stage5_0_fit_final_models.py --outcome rating
# With no argument it is "distress", so the default is never edited and cannot
# be left on "rating" by accident after a run.
OUTCOME_SET = "distress"          # "distress" | "rating"
if "--outcome" in sys.argv:
    _i = sys.argv.index("--outcome")
    if _i + 1 >= len(sys.argv):
        raise SystemExit("STOP: --outcome needs a value: distress or rating.")
    OUTCOME_SET = sys.argv[_i + 1]

if OUTCOME_SET == "distress":
    from stage4_1_distress_models import (ID_COLS, INPUT_FILE, K, OUTCOME,
                                          RANDOM_STATE, SAMPLE_LAG, build_models)
    SPEC_MODULE = "stage4_1"
elif OUTCOME_SET == "rating":
    from stage4_2_rating_models import (ID_COLS, INPUT_FILE, K, OUTCOME,
                                        RANDOM_STATE, SAMPLE_LAG, build_models)
    SPEC_MODULE = "stage4_2"
else:
    raise SystemExit(f"STOP: OUTCOME_SET must be 'distress' or 'rating', "
                     f"not {OUTCOME_SET!r}.")

# The filename infix. EMPTY for distress, so every existing distress filename is
# unchanged character for character; "_rating" otherwise, so the two outcomes'
# models can never be confused for one another on disk.
_O = "" if OUTCOME_SET == "distress" else f"_{OUTCOME_SET}"

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

# WHERE OUTPUTS GO. Reads always come from OUT_DIR; only writes follow this.
# The two are separated so the parity check can re-run this script into a
# temporary directory and compare, without breaking its reads of the stage4_*
# results it checks itself against. Unset — which is always, outside that check
# — this IS OUT_DIR and nothing behaves differently.
WRITE_DIR = Path(os.environ["BOND_WRITE_DIR"]) if os.environ.get("BOND_WRITE_DIR") else OUT_DIR

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# A free-text tag, not a TUNED boolean. A boolean only ever separates two runs;
# a tag lets `base`, `tuned`, `no_oplev` and `oeneg_dropped` coexist, which is
# what the parked decisions will actually produce. It goes in every filename and
# in every metadata file, so two fits can be compared instead of one
# overwriting the other.
TAG = "base"

MODELS = ["xgb", "rf", "enet"]

# stage3_10's configuration, recorded so the input panel can be REBUILT from the
# metadata alone. A model file without this is unusable in six months and
# unusable in China: the same 31 numbers must be constructed the same way, or
# the coefficients are being applied to quantities they were not fitted on.
# Read from the module itself, so it cannot drift from what actually ran.
STAGE3_10_FIELDS = [
    "SAMPLE_LAG", "K", "WINDOW_START", "WINDOW_END", "IMPUTE_METHOD",
    "LOOKBACK_OBS", "MIN_CELL_FIRMS", "STRICT_AVAIL_CELLS", "RANK_TRANSFORM",
    "RANK_EXEMPT", "IMPUTE_EXEMPT", "INDICATOR_NA_CODE", "ALWAYS_EXCLUDE",
    "ALWAYS_KEEP",
]

# The prohibition, carried in the data as well as in the prose.
SCOPE = "CHINA_TRANSFER_ONLY"
# SPEC_MODULE renders as "stage4_1" for distress, so this string is unchanged
# there; for rating it correctly names stage4_2 as the source of US scores.
WARNING = ("Fitted on ALL rows. MUST NOT be used to score any US firm — it has "
           "seen every US observation and its own outcome. US scores come from "
           f"{SPEC_MODULE}'s cross-fitted predictions (stage5_1_score_panel.py). "
           "This object exists only to be applied to Chinese data.")

# stage4_1's importances are themselves a full-sample descriptive fit at the
# same seed, so this script's fit must reproduce them.
# 1e-6, NOT 1e-9. Both tree models run at n_jobs=4 and threaded floating-point
# summation order is not guaranteed identical between runs, so an honest rerun
# can differ at ~1e-7 with nothing having drifted. A guard that fires on thread
# scheduling is a guard people learn to ignore. Real drift — a changed
# hyperparameter, a changed seed, a changed row set — moves importances by
# orders of magnitude more than this, so the max |diff| is printed every run and
# a genuine problem will look nothing like numerical noise.
REPRODUCTION_TOL = 1e-6


def model_path(model):
    """xgb saves natively; the scikit-learn objects go through joblib."""
    ext = "json" if model == "xgb" else "joblib"
    return WRITE_DIR / f"stage5_0_model{_O}_{SAMPLE_LAG}_K{K}_{TAG}_{model}.{ext}"


def meta_path(model):
    return WRITE_DIR / f"stage5_0_model{_O}_{SAMPLE_LAG}_K{K}_{TAG}_{model}_meta.json"


def coef_path():
    return WRITE_DIR / f"stage5_0_model{_O}_{SAMPLE_LAG}_K{K}_{TAG}_enet_coefficients.csv"


def results_path(model):
    """The cross-fitted results this fit must reproduce.

    stage4_1 names its results with the lag and K; stage4_2 does not. That
    asymmetry is on disk and is not worth a rename, so it is absorbed here.
    """
    if OUTCOME_SET == "distress":
        return OUT_DIR / f"stage4_1_results_{SAMPLE_LAG}_K{K}_{model}.json"
    return OUT_DIR / f"stage4_2_results_{model}.json"


def read_stage3_10_config():
    """Read the named constants off the imported stage3_10 module.

    Taking the VALUES rather than the source text matters: IMPUTE_EXEMPT is
    `list(RANK_EXEMPT)`, so a text-based reader would record the expression
    instead of the three feature names it evaluates to.

    Anything absent is reported and recorded as null rather than silently
    omitted — a missing field is exactly the gap that makes a saved model
    unusable later.
    """
    found = {}
    for field in STAGE3_10_FIELDS:
        value = getattr(s310, field, None)
        # numpy scalars and sets are not JSON-serialisable; normalise them.
        if isinstance(value, set):
            value = sorted(value)
        elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
            value = value.item()
        found[field] = value

    missing = [f for f in STAGE3_10_FIELDS if found[f] is None]
    if missing:
        print(f"  NOTE: absent from stage3_10, recorded as null: {missing}")

    # The prepared file this script reads must be the one stage3_10 is currently
    # configured to write, or the recorded config describes a different panel.
    if (s310.SAMPLE_LAG, s310.K) != (SAMPLE_LAG, K):
        raise SystemExit(
            f"STOP: stage3_10 is currently set to {s310.SAMPLE_LAG} K={s310.K} "
            f"but this script is fitting on {SAMPLE_LAG} K={K}. The config that "
            "would be recorded in the metadata is NOT the config that built the "
            "input. Restore stage3_10's switches before running.")
    return found


def versions():
    """Everything needed to know whether a saved object can still be loaded."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
    }


def save_enet_coefficients(est, feats):
    """The portable form of the linear model: coefficients plus scaler parameters.

    A .joblib is a fragile container; this CSV is the model. To apply it by hand:
        z = (x - scaler_mean) / scaler_scale ;  logit = intercept + sum(coef * z)
    """
    scaler = est.named_steps["scale"]
    clf = est.named_steps["clf"].best_estimator_
    out = pd.DataFrame({
        "feature": feats,
        "coefficient": clf.coef_[0],
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    })
    # The intercept is not per-feature, so it goes on its own row, named.
    out = pd.concat([
        pd.DataFrame([{"feature": "_intercept_", "coefficient": float(clf.intercept_[0]),
                       "scaler_mean": np.nan, "scaler_scale": np.nan}]),
        out,
    ], ignore_index=True)
    out.to_csv(coef_path(), index=False)
    return out


def check_reproduces_stage4_1(model, fitted_importances):
    """The fit must reproduce the cross-fit script's importances, or stop.

    That script's `importances` block is ONE full-sample fit at the same seed on
    the same rows, so agreement is not a coincidence — it is the definition of
    the specification not having drifted. Which script that is depends on
    OUTCOME_SET: stage4_1 for distress, stage4_2 for rating.
    """
    path = results_path(model)
    if not path.exists():
        raise SystemExit(
            f"STOP: {path.name} is not there, so the fit cannot be checked "
            f"against {SPEC_MODULE}. Run {SPEC_MODULE} first; this script will "
            "not save an unverified model."
        )
    saved = json.load(open(path))["importances"]
    if sorted(saved) != sorted(fitted_importances):
        raise SystemExit(
            f"STOP: {model} feature sets differ from {SPEC_MODULE}'s.\n"
            f"  here not there: {sorted(set(fitted_importances) - set(saved))}\n"
            f"  there not here: {sorted(set(saved) - set(fitted_importances))}"
        )
    diffs = {f: abs(fitted_importances[f] - saved[f]) for f in saved}
    worst = max(diffs, key=diffs.get)
    print(f"      reproduces {SPEC_MODULE}: max |diff| {diffs[worst]:.3g} on {worst}")
    if diffs[worst] > REPRODUCTION_TOL:
        raise SystemExit(
            f"STOP: {model} does not reproduce {SPEC_MODULE}'s descriptive fit. "
            f"Largest disagreement {diffs[worst]:.6g} on {worst}, tolerance "
            f"{REPRODUCTION_TOL:g}.\n"
            "  The two are meant to be the SAME fit — same rows, same "
            "specification, same seed — so a difference means the specification "
            "has drifted. Resolve before saving a model that claims to be the "
            "one that was evaluated."
        )
    return diffs[worst]


def main():
    print("=" * 84)
    print(f"STAGE 5.0 — FULL-SAMPLE MODELS FOR TRANSFER   "
          f"outcome={OUTCOME_SET} spec={SPEC_MODULE} {SAMPLE_LAG} K={K} tag={TAG}")
    print("=" * 84)
    for line in WARNING.split(". "):
        print(f"  !! {line.strip()}")
    print("=" * 84)

    # Refuse to overwrite; a different TAG gives a different filename.
    existing = [p for m in MODELS for p in (model_path(m), meta_path(m))
                if p.exists()] + ([coef_path()] if coef_path().exists() else [])
    if existing:
        raise SystemExit(
            "STOP: these already exist and this script never overwrites — "
            f"change TAG or move them first:\n  "
            + "\n  ".join(p.name for p in existing))

    # ---- the data, on exactly the cross-fit script's rule -------------------
    df = pd.read_parquet(INPUT_FILE)
    feats = [c for c in df.columns if c not in ID_COLS]

    # The rating models are applied to the SAME Chinese feature matrix as the
    # distress ones, so the two must agree on the features AND on their ORDER.
    # Checked as a SEQUENCE, not as a set: a permuted vector scores silently
    # wrong rather than raising.
    if OUTCOME_SET == "rating":
        ref = OUT_DIR / f"stage5_0_model_{SAMPLE_LAG}_K{K}_{TAG}_xgb_meta.json"
        if not ref.exists():
            raise SystemExit(
                f"STOP: {ref.name} is not there, so the rating features cannot "
                "be checked against the distress models they will be applied "
                "beside. Fit the distress models first.")
        ref_feats = json.load(open(ref))["features_in_order"]
        if feats != ref_feats:
            raise SystemExit(
                "STOP: the rating panel's features do not match the distress "
                f"models' order.\n  distress: {ref_feats}\n  rating  : {feats}")
        print(f"  feature order matches the distress models exactly "
              f"({len(feats)} features, checked as a sequence)")
    na = df[feats].isna().any(axis=1)
    d = df[~na].copy()
    y = d[OUTCOME].to_numpy().astype(int)
    X = d[feats].to_numpy(dtype="float64")
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())

    print(f"\n  input {INPUT_FILE.name}")
    print(f"  rows {len(df):,} -> {len(d):,} after dropping {int(na.sum()):,} "
          f"with a remaining NA ({SPEC_MODULE}'s rule)")
    print(f"  events {n_pos:,} ({n_pos / len(d) * 100:.3f}%)   features {len(feats)}")
    print(f"  FEATURE ORDER IS PART OF THE MODEL and is recorded in every "
          f"metadata file.")

    cfg = read_stage3_10_config()
    vers = versions()

    # ---- fit, check, save --------------------------------------------------
    for model, (name, est, spec) in build_models(n_pos, n_neg).items():
        if model not in MODELS:
            continue
        print(f"\n  {model} — {name}")
        est.fit(X, y)

        if model == "enet":
            clf = est.named_steps["clf"].best_estimator_
            imp = dict(zip(feats, (float(v) for v in clf.coef_[0])))
            extra = {"alpha": float(clf.alpha), "n_iter": int(np.max(clf.n_iter_)),
                     "n_nonzero": int((clf.coef_[0] != 0).sum()),
                     "intercept": float(clf.intercept_[0])}
            print(f"      alpha {extra['alpha']:.4g}   n_nonzero "
                  f"{extra['n_nonzero']} of {len(feats)}   n_iter {extra['n_iter']}")
        else:
            imp = dict(zip(feats, (float(v) for v in est.feature_importances_)))
            extra = {}

        worst = check_reproduces_stage4_1(model, imp)

        # Save the object.
        if model == "xgb":
            est.save_model(str(model_path(model)))
        else:
            joblib.dump(est, model_path(model))

        meta = {
            "scope": SCOPE,
            "WARNING": WARNING,
            "model": model,
            "name": name,
            "tag": TAG,
            "specification": spec,
            "random_state": RANDOM_STATE,
            "features_in_order": feats,
            "n_features": len(feats),
            "n_rows_fitted": len(d),
            "n_events_fitted": n_pos,
            "base_rate": n_pos / len(d),
            "input_file": INPUT_FILE.name,
            "sample_lag": SAMPLE_LAG,
            "K": K,
            "stage3_10_config": cfg,
            "rank_transform": (
                "Within-FYEAR cross-sectional ranks (D-44), computed inside the "
                f"{cfg.get('WINDOW_START')}-{cfg.get('WINDOW_END')} window. "
                "Features listed in stage3_10 RANK_EXEMPT are NOT ranked; their "
                f"missing values are coded {cfg.get('INDICATOR_NA_CODE')}. "
                "CHINESE DATA MUST BE RANKED THE SAME WAY, within its own "
                "market and year (BACKGROUND section 7): the coefficients act "
                "on a firm's relative position, not on a raw ratio."),
            "importances": imp,
            "extra": extra,
            "reproduces_stage4_1_max_abs_diff": worst,
            "library_versions": vers,
            "serialisation": ("xgboost native JSON — portable across versions"
                              if model == "xgb" else
                              "joblib/pickle — WILL NOT load under a different "
                              "scikit-learn version; rebuild from source if so"),
            "when": datetime.now().isoformat(timespec="seconds"),
        }
        json.dump(meta, open(meta_path(model), "w"), indent=2)
        print(f"      wrote {model_path(model).name} "
              f"({model_path(model).stat().st_size / 1e6:.2f} MB) "
              f"and {meta_path(model).name}")

        if model == "enet":
            co = save_enet_coefficients(est, feats)
            nz = int((co.coefficient != 0).sum()) - 1     # less the intercept row
            print(f"      wrote {coef_path().name} — the PORTABLE form: "
                  f"{nz} non-zero coefficients + scaler parameters")

    print("\n" + "=" * 84)
    print("  DONE. Three fitted models, three metadata files, one coefficient CSV.")
    print("  REMINDER: none of these may score a US firm. They are for CHINA.")
    print("=" * 84)


if __name__ == "__main__":
    main()
