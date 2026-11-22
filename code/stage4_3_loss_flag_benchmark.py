"""
Stage 4.3 — how much of the model is one loss flag?

A univariate logistic regression on the loss flag ALONE, with the same grouping,
the same cross-fitting and the same calibration as the full models, so its
AUC-ROC and AUC-PR are directly comparable with theirs.

Why it exists: 74.77% of distress events are loss years against 21.25% of
everything else, a univariate odds ratio of 10.98. If a single flag reaches most
of the full model's discrimination, then the INCREMENTAL contribution of the
other features is the honest headline, not the full-model AUC.

FLAG_COL is parameterised so this runs before the rebuild (Tax_negden, ib <= 0)
and after it (loss_year), and it prints which definition it used.

Writes NO files.

Run:  python3 code/stage4_3_loss_flag_benchmark.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

# ===========================================================================
# CONFIG — mirrors stage4_1 exactly, so the numbers are comparable
# ===========================================================================
SAMPLE_LAG = "L3"
K = 12
INPUT_FILE = OUT_DIR / f"stage4_0_distress_{SAMPLE_LAG}_K{K}.parquet"
OUTCOME = "y"
FLAG_COL = "INTWO"          # the loss-state flag to benchmark
ID_COLS = ["issuer_id", "gvkey", "fyear", "datadate", "date_avail", "y"]
GROUP_MODE = "component"
N_SPLITS = 5
RANDOM_STATE = 42
# The full models, for comparison. From output/stage4_1_results_*.json.
# V3 (L3, K=12), from the refit of 2026-08-30. Was V2's
# xgb (0.8656, 0.1427) / rf (0.8745, 0.1370) / enet (0.8445, 0.0987).
FULL_MODEL_REFERENCE = {"xgb": (0.8731, 0.1558), "rf": (0.8742, 0.1414),
                        "enet": (0.8582, 0.1060)}
# ===========================================================================


def build_groups(d):
    """Identical to stage4_1's, so the folds are the same kind of folds."""
    if GROUP_MODE != "component":
        return d[GROUP_MODE].to_numpy()
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ic = d.issuer_id.astype("category").cat.codes.to_numpy()
    gc = d.gvkey.astype("category").cat.codes.to_numpy()
    n_i = ic.max() + 1
    size = n_i + gc.max() + 1
    m = coo_matrix((np.ones(len(d)), (ic, gc + n_i)), shape=(size, size))
    _, lab = connected_components(m + m.T, directed=False)
    return lab[ic]


def main():
    df = pd.read_parquet(INPUT_FILE)
    feats = [c for c in df.columns if c not in ID_COLS]
    # The SAME rows the full models used: drop anything still missing anywhere,
    # not just in the flag, or the benchmark would sit on a different sample.
    d = df[~df[feats].isna().any(axis=1)].copy()
    groups = build_groups(d)
    y = d[OUTCOME].to_numpy().astype(int)
    x = d[[FLAG_COL]].to_numpy(dtype="float64")

    print("=" * 84)
    print(f"STAGE 4.3 — UNIVARIATE BENCHMARK on {FLAG_COL} alone")
    print("=" * 84)
    defn = {"Tax_negden": "ib <= 0, a single year",
            "INTWO": "ni < 0 in BOTH of the last two years (Ohlson 1980)",
            "OENEG": "lt > at (Ohlson 1980)"}.get(FLAG_COL, "see stage1_1")
    print(f"  definition: {FLAG_COL} = 1 where {defn}"
          f"   (-1 = cannot be determined, D-44)")
    print(f"  rows {len(d):,}   groups {len(np.unique(groups)):,}   "
          f"events {int(y.sum()):,} ({y.mean()*100:.3f}%)")
    print(f"  values: " + ", ".join(
        f"{v:g} -> {int((x[:, 0] == v).sum()):,}"
        for v in np.unique(x[:, 0])))

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=RANDOM_STATE)
    p = np.full(len(y), np.nan)
    for tr, te in cv.split(x, y, groups):
        m = LogisticRegression(class_weight="balanced", max_iter=1000)
        m.fit(x[tr], y[tr])
        p[te] = m.predict_proba(x[te])[:, 1]

    auc, ap = roc_auc_score(y, p), average_precision_score(y, p)
    print(f"\n  UNIVARIATE:  AUC-ROC {auc:.4f}   AUC-PR {ap:.4f}")
    print(f"\n  {'model':<10}{'AUC-ROC':>9}{'AUC-PR':>9}"
          f"{'ROC gain over flag':>20}{'PR gain':>10}")
    print(f"  {FLAG_COL + ' only':<10}{auc:>9.4f}{ap:>9.4f}{'—':>20}{'—':>10}")
    for k, (a, q) in FULL_MODEL_REFERENCE.items():
        print(f"  {k:<10}{a:>9.4f}{q:>9.4f}{a - auc:>+20.4f}{q - ap:>+10.4f}")

    # The honest reading: how much of the full model's edge over a coin flip is
    # already in the flag. Chance is 0.5 for ROC and the base rate for PR.
    base = y.mean()
    print(f"\n  SHARE OF THE FULL MODEL'S SKILL ALREADY IN THE FLAG")
    print(f"      (skill = the excess over chance: 0.5 for ROC, "
          f"{base:.5f} for PR)")
    for k, (a, q) in FULL_MODEL_REFERENCE.items():
        sr = (auc - 0.5) / (a - 0.5) * 100
        sp = (ap - base) / (q - base) * 100
        print(f"      vs {k:<6} ROC {sr:5.1f}%   PR {sp:5.1f}%")
    print("\nNo file written.")


if __name__ == "__main__":
    main()
