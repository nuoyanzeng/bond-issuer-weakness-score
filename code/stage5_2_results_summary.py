"""
Stage 5.2 — The US results summary, regenerated from saved artefacts.

WHY THIS EXISTS. `output/v3_results_summary.md` describes the SUPERSEDED
31-feature set, and there is no equivalent for the current 27. This script
produces the counterpart so that both versions are described by the SAME code:
set SET_TAG = "s31" and it regenerates the 31-set account from that set's own
saved files. Any difference between the two documents is then a difference in
the data, not in how it was measured.

IT NEVER REFITS. It reads saved cross-fitted scores, saved results files and the
saved coefficient CSV. No estimator is imported and `.fit()` is never called.
That is not an optimisation: a refit would silently change what the numbers
describe, because the scores in the panel are cross-fitted over folds this
script cannot reconstruct.

Run:  python3 code/stage5_2_results_summary.py
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Fixed for reproducibility as a matter of course. NOTHING IN THIS SCRIPT
# CONSUMES IT — there is no sampling, no split and no model here.
np.random.seed(42)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
RAW_US = PROJECT_DIR / "data_raw" / "us"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SET_TAG = ""            # "" = the current 27-set; "s31" regenerates the 31-set
LAG = "L3"              # L0 is read in pass 6 for the comparison
K = 12
OUT_NAME = "v4_results_summary.md"

MODELS = ["xgb", "rf", "enet"]
KEY = ["issuer_id", "gvkey", "fyear"]
ID_COLS = KEY + ["datadate", "date_avail", "y"]
SPREAD_FLAG = 10        # a rank spread at or above this gets its correlations printed
TOP_CORR = 4            # how many correlates to print for such a feature

# The two indicators are STATES, not ratios, and are reported as states.
INDICATORS = ["OENEG", "INTWO"]

# THEME LABELS. This is the ONE hard-coded list in this file, and it is here
# under protest: no input this script reads carries a theme, and §3.8 of the
# project report sorts its table by theme then consensus rank. These are LABELS,
# not data — no number depends on them. D-55 records eight files that went stale
# carrying their own copies of a list, so every feature is asserted to have a
# theme and the script halts if one is missing. Check against §3.8 before use.
THEMES = {
    "roaq": "profitability", "PM": "profitability", "roavol": "profitability",
    "ocf_at": "profitability", "roic": "profitability", "ChPM": "profitability",
    "GrGMToGrSales": "profitability",
    "TLTA": "leverage/financing", "cashdebt": "leverage/financing",
    "DelEqu": "leverage/financing", "DelFINL": "leverage/financing",
    "WW": "leverage/financing",
    "Cash": "liquidity", "DelSTI": "liquidity",
    "DelLTI": "growth", "AssetGrowth": "growth", "grcapx1y": "growth",
    "sgr": "growth",
    "salecash": "efficiency", "OPLeverage": "efficiency",
    "salerec": "efficiency", "AssetTurnover": "efficiency",
    "PctAcc": "accruals", "Accruals": "accruals", "ChNCOL": "accruals",
    "DelCOL": "accruals", "DelCOA": "accruals", "ChNWC": "accruals",
    "TotalAccruals": "accruals",
    "INTWO": "state", "OENEG": "state",
}
THEME_ORDER = ["profitability", "leverage/financing", "liquidity", "growth",
               "efficiency", "accruals", "state"]

# SIC divisions, by two-digit code. D-10: sich first, comp_names.sic as fallback.
SIC_DIV = [(1, 9, "A agriculture"), (10, 14, "B mining"),
           (15, 17, "C construction"), (20, 39, "D manufacturing"),
           (40, 49, "E transport/utilities"), (50, 51, "F wholesale"),
           (52, 59, "G retail"), (60, 67, "H finance"),
           (70, 89, "I services"), (91, 99, "J public admin")]

_T = f"_{SET_TAG}" if SET_TAG else ""
# stage5_0 puts the feature-set name in the slot where an untagged run writes
# "base"; see the 2026-09-19 naming note. "base" now denotes the 27-set.
_S5 = SET_TAG if SET_TAG else "base"

OUT = []                # every line of the document, written once at the end


def w(line=""):
    OUT.append(line)


# ---------------------------------------------------------------------------
# PATHS — every one derived from SET_TAG / LAG / K
# ---------------------------------------------------------------------------
def panel_path(lag=LAG):
    return OUT_DIR / f"stage4_0_distress_{lag}_K{K}{_T}.parquet"


def d_results(model, lag=LAG):
    return OUT_DIR / f"stage4_1_results_{lag}_K{K}{_T}_{model}.json"


def d_scores(model, lag=LAG):
    return OUT_DIR / f"stage4_1_scores_{lag}_K{K}{_T}_{model}.parquet"


def r_results(model):
    return OUT_DIR / f"stage4_2_results{_T}_{model}.json"


def r_scores(model):
    return OUT_DIR / f"stage4_2_scores{_T}_{model}.parquet"


def coef_path():
    return OUT_DIR / f"stage5_0_model_{LAG}_K{K}_{_S5}_enet_coefficients.csv"


def meta_path(model):
    return OUT_DIR / f"stage5_0_model_{LAG}_K{K}_{_S5}_{model}_meta.json"


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------
def load():
    """Read everything. Assert the feature list before any figure is computed."""
    out_file = OUT_DIR / OUT_NAME
    # v3_results_summary.md is the historical record of the 31-set. No
    # configuration of this script may write over it.
    if OUT_NAME == "v3_results_summary.md":
        raise SystemExit(
            "STOP: v3_results_summary.md is the historical 31-set record and is "
            "never written by this script. Change OUT_NAME.")
    if out_file.exists():
        raise SystemExit(f"STOP: {OUT_NAME} exists; this script never "
                         "overwrites. Move or rename it first.")

    feats = json.loads(meta_path("xgb").read_text())["features_in_order"]
    if SET_TAG == "" and len(feats) != 27:
        raise SystemExit(f"STOP: SET_TAG is the current set but the metadata "
                         f"lists {len(feats)} features, not 27.")

    panel = pd.read_parquet(panel_path())
    got = [c for c in panel.columns if c not in ID_COLS]
    # AS A SEQUENCE, not as a set: a permuted vector would make every rank in
    # this document describe the wrong feature.
    if got != feats:
        raise SystemExit(f"STOP: the panel's feature columns are not the "
                         f"metadata's list in order.\n  panel: {got}\n"
                         f"  meta : {feats}")
    missing_theme = [f for f in feats if f not in THEMES]
    if missing_theme:
        raise SystemExit(f"STOP: no theme label for {missing_theme}; add them to "
                         "THEMES after checking §3.8 of the project report.")

    dres = {m: json.loads(d_results(m).read_text()) for m in MODELS}
    rres = {m: json.loads(r_results(m).read_text()) for m in MODELS}
    dsc = {m: pd.read_parquet(d_scores(m)) for m in MODELS}
    rsc = {m: pd.read_parquet(r_scores(m)) for m in MODELS}
    coef = pd.read_csv(coef_path())
    enet_meta = json.loads(meta_path("enet").read_text())

    # The FITTED rows are the scored rows, not the whole panel: stage4_1 drops
    # rows carrying a residual NA before fitting.
    fit = panel.merge(dsc["xgb"][KEY + ["score_calibrated"]], on=KEY, how="inner")
    if len(fit) != dres["xgb"]["n_rows"]:
        raise SystemExit(f"STOP: joined {len(fit):,} fitted rows against "
                         f"{dres['xgb']['n_rows']:,} recorded.")
    return feats, panel, fit, dres, rres, dsc, rsc, coef, enet_meta


# ---------------------------------------------------------------------------
# PASS 1 — consensus importance and coefficients
# ---------------------------------------------------------------------------
def pass1(feats, fit, dres, coef, enet_meta):
    w("\n## 1. Consensus importance, and the elastic-net coefficients\n")
    w("Ranks only. The three measures — XGBoost gain, random-forest impurity "
      "decrease, and elastic-net coefficient magnitude — are different "
      "quantities and are never placed in one column. Rank 1 is the largest "
      "in that model's own measure. The elastic net stores SIGNED "
      "coefficients, so it is ranked on magnitude.\n")
    ranks = {}
    for m in MODELS:
        imp = dres[m]["importances"]
        s = pd.Series({f: abs(float(imp[f])) for f in feats})
        ranks[m] = s.rank(ascending=False, method="min").astype(int)
    r = pd.DataFrame(ranks)
    r["mean_rank"] = r[MODELS].mean(axis=1)
    r["spread"] = r[MODELS].max(axis=1) - r[MODELS].min(axis=1)
    r = r.sort_values("mean_rank")

    w(f"| feature | {' | '.join(MODELS)} | mean rank | spread |")
    w("|---|" + "---|" * (len(MODELS) + 2))
    for f, row in r.iterrows():
        w(f"| `{f}` | " + " | ".join(str(int(row[m])) for m in MODELS)
          + f" | {row.mean_rank:.1f} | {int(row.spread)} |")

    # Coefficients, with signs, from the saved CSV.
    c = coef[coef.feature != "_intercept_"].copy()
    icept = coef.loc[coef.feature == "_intercept_", "coefficient"].iloc[0]
    nz = c[c.coefficient != 0]
    alpha = enet_meta.get("extra", {}).get("alpha")
    if alpha is None:
        raise SystemExit(f"STOP: no alpha in {meta_path('enet').name}.")
    w(f"\n**Elastic net.** alpha {alpha:.4g}; intercept {icept:+.4f}; "
      f"**{len(nz)} of {len(c)} coefficients non-zero**.\n")
    w("| feature | coefficient |")
    w("|---|---|")
    for _, row in nz.reindex(nz.coefficient.abs().sort_values(
            ascending=False).index).iterrows():
        w(f"| `{row.feature}` | {row.coefficient:+.4f} |")
    zero = sorted(c.loc[c.coefficient == 0, "feature"])
    w(f"\nDriven to exactly zero: {', '.join('`' + z + '`' for z in zero) or 'none'}.")

    # Where the models disagree on rank, print the correlations rather than a
    # label. The panel is already within-year ranks, which is what the models
    # saw; Spearman is used so the figure does not depend on that.
    wide = r[r.spread >= SPREAD_FLAG]
    w(f"\n**Features with a rank spread of {SPREAD_FLAG} or more.** The "
      f"strongest correlations among the other features are printed so the "
      f"spread can be read; no classification is made here.\n")
    if not len(wide):
        w(f"None.")
    else:
        corr = fit[feats].corr(method="spearman")
        w("| feature | spread | strongest correlates (Spearman) |")
        w("|---|---|---|")
        for f in wide.index:
            o = corr[f].drop(f).abs().sort_values(ascending=False).head(TOP_CORR)
            txt = ", ".join(f"`{k}` {corr.loc[k, f]:+.3f}" for k in o.index)
            w(f"| `{f}` | {int(wide.loc[f, 'spread'])} | {txt} |")
    return r, nz


# ---------------------------------------------------------------------------
# PASS 3 — decile profiles (computed before pass 2, which consumes it)
# ---------------------------------------------------------------------------
def pass3(feats, fit):
    """Returns (direction, lines). The lines are NOT appended here: pass 2
    consumes `direction`, so pass 3 must be computed first while appearing
    after it in the document."""
    L = ["\n## 3. Univariate decile profiles\n",
         "Realised default rate by decile of each feature, on the fitted rows. "
         "The two indicators are reported as STATES, not deciles.\n"]
    direction = {}
    for f in [x for x in feats if x not in INDICATORS]:
        d = fit[[f, "y"]].dropna(subset=[f]).copy()
        d["dec"] = pd.qcut(d[f], 10, labels=False, duplicates="drop") + 1
        g = d.groupby("dec").agg(n=("y", "size"), events=("y", "sum"))
        g["rate"] = g.events / g.n * 100
        direction[f] = (g["rate"].iloc[0], g["rate"].iloc[-1])
        L.append(f"\n**`{f}`** — {len(d):,} rows, {int(d.y.sum()):,} events")
        L.append("| decile | n | events | rate % |")
        L.append("|---|---|---|---|")
        for dec, row in g.iterrows():
            L.append(f"| {int(dec)} | {int(row.n):,} | {int(row.events):,} | "
                     f"{row['rate']:.2f} |")

    for f in [x for x in feats if x in INDICATORS]:
        L.append(f"\n**`{f}`** — as a state")
        L.append("| state | n | events | rate % |")
        L.append("|---|---|---|---|")
        for v, lab in [(1.0, "1"), (0.0, "0"), (-1.0, "-1 (undetermined)")]:
            s = fit[fit[f] == v]
            if not len(s):
                continue
            L.append(f"| {lab} | {len(s):,} | {int(s.y.sum()):,} | "
                     f"{s.y.mean() * 100:.2f} |")
    return direction, L


# ---------------------------------------------------------------------------
# PASS 2 — direction of effect
# ---------------------------------------------------------------------------
def pass2(nz, direction):
    w("\n## 2. Direction of effect\n")
    w("The coefficient's sign beside the direction of the feature's own "
      "univariate decile profile from pass 3. **The prior and verdict columns "
      "are deliberately empty**: assigning an economic prior is judgement and "
      "does not belong in a script.\n")
    w("| feature | coefficient | sign | decile 1 rate % | decile 10 rate % | "
      "profile | prior | verdict |")
    w("|---|---|---|---|---|---|---|---|")
    for _, row in nz.reindex(nz.coefficient.abs().sort_values(
            ascending=False).index).iterrows():
        f = row.feature
        if f not in direction:
            w(f"| `{f}` | {row.coefficient:+.4f} | "
              f"{'+' if row.coefficient > 0 else '-'} | — | — | "
              f"state, not a decile |  |  |")
            continue
        lo, hi = direction[f]
        prof = "rises" if hi > lo else ("falls" if hi < lo else "flat")
        w(f"| `{f}` | {row.coefficient:+.4f} | "
          f"{'+' if row.coefficient > 0 else '-'} | {lo:.2f} | {hi:.2f} | "
          f"{prof} with the feature |  |  |")


# ---------------------------------------------------------------------------
# PASS 4 — early-warning profile
# ---------------------------------------------------------------------------
def pass4(dsc):
    w("\n## 4. Early-warning profile\n")
    w("Firm-years at each horizon before the issuer's first distress year, "
      "against never-distressed rows **in the same fiscal years**, so the "
      "baseline moves with the sample. The never-distressed 90th percentile is "
      "by construction exceeded by 10% of never-distressed rows.\n")
    ev = pd.read_parquet(OUT_DIR / "stage3_0_distress_events.parquet")
    ev["ev_year"] = pd.to_datetime(ev.first_distress_date).dt.year
    for m in MODELS:
        s = dsc[m].merge(ev[["issuer_id", "ev_year"]], on="issuer_id", how="left")
        never = s[s.ev_year.isna()]
        w(f"\n**{m}**")
        w("| horizon | firm-years | firms | mean score | never-distressed mean "
          "| ratio | share above never-distressed p90 |")
        w("|---|---|---|---|---|---|---|")
        for h in range(1, 6):
            g = s[s.ev_year.notna() & (s.fyear == s.ev_year - h)]
            if not len(g):
                w(f"| t-{h} | 0 | 0 | — | — | — | — |")
                continue
            base = never[never.fyear.isin(g.fyear.unique())]
            if not len(base):
                w(f"| t-{h} | {len(g):,} | {g.issuer_id.nunique():,} | "
                  f"{g.score_calibrated.mean():.4f} | — | — | — |")
                continue
            p90 = base.score_calibrated.quantile(0.90)
            w(f"| t-{h} | {len(g):,} | {g.issuer_id.nunique():,} | "
              f"{g.score_calibrated.mean():.4f} | "
              f"{base.score_calibrated.mean():.4f} | "
              f"{g.score_calibrated.mean() / base.score_calibrated.mean():.2f}x | "
              f"{(g.score_calibrated > p90).mean() * 100:.1f}% |")
        w(f"| never distressed | {len(never):,} | "
          f"{never.issuer_id.nunique():,} | "
          f"{never.score_calibrated.mean():.4f} | — | 1.00x | 10.0% |")


# ---------------------------------------------------------------------------
# PASS 5 — period, sector, misses
# ---------------------------------------------------------------------------
def sic_division(code):
    if pd.isna(code):
        return "unknown"
    try:
        two = int(str(int(float(code))).zfill(4)[:2])
    except (ValueError, TypeError):
        return "unknown"
    for lo, hi, lab in SIC_DIV:
        if lo <= two <= hi:
            return lab
    return "unknown"


def auc_row(y, s):
    """AUC-ROC and AUC-PR, or None where one class is absent."""
    if y.nunique() < 2:
        return None, None
    return roc_auc_score(y, s), average_precision_score(y, s)


def pass5(feats, fit, dsc, consensus):
    w("\n## 5. Period, sector, and the composition of misses\n")
    w("Rows and event counts are printed beside every figure, so a figure "
      "resting on single-digit events is visibly doing so.\n")

    ann = pd.read_parquet(OUT_DIR / "compustat_annual_clean.parquet",
                          columns=["gvkey", "datadate", "at", "sich"])
    names = pd.read_parquet(RAW_US / "comp_names.parquet",
                            columns=["gvkey", "sic"])
    names["sic"] = names["sic"].replace("", pd.NA)
    ann = ann.merge(names, on="gvkey", how="left")
    ann["sic_use"] = ann["sich"].where(ann["sich"].notna(), ann["sic"])
    ann["division"] = ann["sic_use"].map(sic_division)
    ctx = ann[["gvkey", "datadate", "at", "division"]]

    for m in MODELS:
        d = fit[KEY + ["datadate", "y"]].merge(
            dsc[m][KEY + ["score_calibrated"]], on=KEY, how="inner")
        d = d.merge(ctx, on=["gvkey", "datadate"], how="left")
        w(f"\n**{m} — by three-year block**")
        w("| block | rows | firms | events | AUC-ROC | AUC-PR |")
        w("|---|---|---|---|---|---|")
        d["block"] = (d.fyear // 3) * 3
        for b, g in d.groupby("block"):
            roc, pr = auc_row(g.y, g.score_calibrated)
            w(f"| {int(b)}-{int(b) + 2} | {len(g):,} | {g.gvkey.nunique():,} | "
              f"{int(g.y.sum()):,} | "
              f"{'—' if roc is None else f'{roc:.4f}'} | "
              f"{'—' if pr is None else f'{pr:.4f}'} |")

        w(f"\n**{m} — by SIC division**")
        w("| division | rows | firms | events | AUC-ROC | AUC-PR |")
        w("|---|---|---|---|---|---|")
        for div, g in d.groupby(d.division.fillna("unknown")):
            roc, pr = auc_row(g.y, g.score_calibrated)
            w(f"| {div} | {len(g):,} | {g.gvkey.nunique():,} | "
              f"{int(g.y.sum()):,} | "
              f"{'—' if roc is None else f'{roc:.4f}'} | "
              f"{'—' if pr is None else f'{pr:.4f}'} |")

    # The four groups, per model, on the calibrated score.
    top5 = list(consensus.index[:5])
    w(f"\n**Composition of misses, hits and false alarms.** The feature column "
      f"is the mean of the five highest-consensus features: "
      f"{', '.join('`' + f + '`' for f in top5)}.\n")
    w("| model | group | n | firms | median total assets | share financial | "
      + " | ".join(f"mean `{f}`" for f in top5) + " |")
    w("|---|---|---|---|---|---|" + "---|" * len(top5))
    for m in MODELS:
        d = fit[KEY + ["datadate", "y"] + feats].merge(
            dsc[m][KEY + ["score_calibrated"]], on=KEY, how="inner")
        d = d.merge(ctx, on=["gvkey", "datadate"], how="left")
        med = d.score_calibrated.median()
        cut = d.score_calibrated.quantile(0.90)
        groups = [
            ("events, bottom half (misses)", (d.y == 1) & (d.score_calibrated <= med)),
            ("events, top decile (hits)", (d.y == 1) & (d.score_calibrated > cut)),
            ("non-events, top decile (false alarms)",
             (d.y == 0) & (d.score_calibrated > cut)),
            ("non-events, outside top decile", (d.y == 0) & (d.score_calibrated <= cut)),
        ]
        for lab, sel in groups:
            g = d[sel]
            if not len(g):
                continue
            fin = (g.division == "H finance").mean() * 100
            means = " | ".join(f"{g[f].mean():+.3f}" for f in top5)
            w(f"| {m} | {lab} | {len(g):,} | {g.gvkey.nunique():,} | "
              f"{g['at'].median():,.0f} | {fin:.1f}% | {means} |")


# ---------------------------------------------------------------------------
# PASS 6 — model agreement, and L3 against L0
# ---------------------------------------------------------------------------
def pass6(dsc, dres):
    w("\n## 6. Model agreement, and L3 against L0\n")
    base = dsc[MODELS[0]][KEY + ["y"]].copy()
    for m in MODELS:
        base = base.merge(dsc[m][KEY + ["score_calibrated"]].rename(
            columns={"score_calibrated": m}), on=KEY, how="inner")
    w(f"{len(base):,} rows common to all three models, "
      f"{int(base.y.sum()):,} events.\n")

    w("| pair | Spearman | top-decile overlap |")
    w("|---|---|---|")
    n_top = max(1, int(round(len(base) * 0.10)))
    tops = {m: set(base[m].nlargest(n_top).index) for m in MODELS}
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            w(f"| {a} / {b} | {base[a].corr(base[b], method='spearman'):.4f} | "
              f"{len(tops[a] & tops[b]) / n_top * 100:.1f}% |")

    inter = tops[MODELS[0]] & tops[MODELS[1]] & tops[MODELS[2]]
    ev_i = int(base.loc[list(inter), "y"].sum())
    w(f"\n**Three-way intersection of the top deciles.** {len(inter):,} rows, "
      f"{ev_i:,} events, precision {ev_i / max(1, len(inter)) * 100:.2f}%.\n")
    w(f"**Size-matched comparison.** Each single model's top {len(inter):,} "
      f"rows — the intersection's size — so a smaller flagged set cannot be "
      f"mistaken for a better one.\n")
    w("| flagged set | n | events | precision % |")
    w("|---|---|---|---|")
    w(f"| three-way intersection | {len(inter):,} | {ev_i:,} | "
      f"{ev_i / max(1, len(inter)) * 100:.2f} |")
    for m in MODELS:
        idx = base[m].nlargest(len(inter)).index
        e = int(base.loc[idx, "y"].sum())
        w(f"| {m}, top {len(inter):,} | {len(idx):,} | {e:,} | "
          f"{e / max(1, len(idx)) * 100:.2f} |")

    # L3 against L0, from the saved results files and the saved scores.
    w(f"\n**{LAG} against L0.**\n")
    w("| model | " + " | ".join(
        f"{LAG} {k}" for k in ("AUC-PR", "AUC-ROC")) + " | "
      + " | ".join(f"L0 {k}" for k in ("AUC-PR", "AUC-ROC")) + " |")
    w("|---|---|---|---|---|")
    l0res, l0sc = {}, {}
    for m in MODELS:
        p, q = d_results(m, "L0"), d_scores(m, "L0")
        if not (p.exists() and q.exists()):
            w(f"| {m} | {dres[m]['metrics']['AUC-PR']:.4f} | "
              f"{dres[m]['metrics']['AUC-ROC']:.4f} | not on disk | not on disk |")
            continue
        l0res[m] = json.loads(p.read_text())
        l0sc[m] = pd.read_parquet(q)
        w(f"| {m} | {dres[m]['metrics']['AUC-PR']:.4f} | "
          f"{dres[m]['metrics']['AUC-ROC']:.4f} | "
          f"{l0res[m]['metrics']['AUC-PR']:.4f} | "
          f"{l0res[m]['metrics']['AUC-ROC']:.4f} |")

    if l0sc:
        w(f"\n**Agreement between the two lags**, on rows present in both.\n")
        w("| model | rows in both | Spearman | top-decile overlap |")
        w("|---|---|---|---|")
        for m in MODELS:
            if m not in l0sc:
                continue
            j = dsc[m][KEY + ["score_calibrated"]].merge(
                l0sc[m][KEY + ["score_calibrated"]], on=KEY,
                suffixes=("_l3", "_l0"), how="inner")
            n = max(1, int(round(len(j) * 0.10)))
            t3 = set(j.score_calibrated_l3.nlargest(n).index)
            t0 = set(j.score_calibrated_l0.nlargest(n).index)
            w(f"| {m} | {len(j):,} | "
              f"{j.score_calibrated_l3.corr(j.score_calibrated_l0, method='spearman'):.4f} | "
              f"{len(t3 & t0) / n * 100:.1f}% |")


# ---------------------------------------------------------------------------
# THE BLOCK THE PROJECT REPORT CONSUMES
# ---------------------------------------------------------------------------
def report_block(feats, consensus):
    n = len(feats)
    w("\n## 7. For docs/PROJECT_REPORT.md §3.8\n")
    w(f"**The four features that are ours.** Ranks are within each model's own "
      f"measure, of {n}.\n")
    w(f"| feature | mean rank | {' | '.join(MODELS)} | source |")
    w("|---|---|" + "---|" * len(MODELS) + "---|")
    ours = {"TLTA": "Ohlson (1980)", "INTWO": "Ohlson (1980)",
            "ocf_at": "JKP, citing Bouchaud, Krüger, Landier & Thesmar (2019)",
            "OENEG": "Ohlson (1980)"}
    for f, src in ours.items():
        if f not in consensus.index:
            w(f"| `{f}` | not in this feature set | " + " | ".join(
                "—" for _ in MODELS) + f" | {src} |")
            continue
        row = consensus.loc[f]
        w(f"| `{f}` | {row.mean_rank:.1f} of {n} | "
          + " | ".join(f"{int(row[m])} of {n}" for m in MODELS)
          + f" | {src} |")

    w(f"\n**Theme, then consensus rank within theme** — the ordering §3.8 sorts "
      f"its table by. Theme labels are from this script's THEMES map, not from "
      f"any input file.\n")
    w("| # | feature | theme | mean rank |")
    w("|---|---|---|---|")
    i = 0
    for th in THEME_ORDER:
        block = [f for f in consensus.index if THEMES[f] == th]
        for f in block:
            i += 1
            w(f"| {i} | `{f}` | {th} | {consensus.loc[f, 'mean_rank']:.1f} |")
    if i != len(feats):
        raise SystemExit(f"STOP: the theme ordering placed {i} features, not "
                         f"{len(feats)}; THEME_ORDER is missing a theme.")


# ---------------------------------------------------------------------------
def header(feats, fit, dres, rres):
    r = dres["xgb"]
    w(f"# US results summary — the "
      f"{'current ' + str(len(feats)) + '-feature set' if not SET_TAG else SET_TAG + ' feature set'}")
    w("")
    w(f"**Date** {date.today().isoformat()} · **feature set** "
      f"{len(feats)} features, tag `{SET_TAG or 'untagged (current)'}` · "
      f"**lag** {LAG} · **K** {K}")
    w("")
    w(f"**Fitted sample, distress.** {r['n_rows']:,} rows, "
      f"{fit.gvkey.nunique():,} gvkeys, {fit.issuer_id.nunique():,} issuers, "
      f"{r['n_groups']:,} cross-fitting groups, {r['n_events']:,} events, "
      f"base rate {r['base_rate'] * 100:.3f}%.")
    w(f"**Fitted sample, rating.** {rres['xgb']['n_rows']:,} rows, "
      f"{rres['xgb']['n_events']:,} positives, "
      f"base rate {rres['xgb']['base_rate'] * 100:.3f}%.")
    w("")
    w("**EVERY SCORE IN THIS DOCUMENT IS CROSS-FITTED, AND NO MODEL WAS "
      "REFITTED TO PRODUCE IT.** Scores are read from the saved out-of-fold "
      "predictions; importances and metrics from the saved results files; "
      "coefficients from the saved coefficient CSV. This script imports no "
      "estimator and calls no `.fit()`.")
    w("")
    w("**Inputs, all read-only.**")
    w("")
    for p in ([panel_path()] + [d_results(m) for m in MODELS]
              + [d_scores(m) for m in MODELS] + [r_results(m) for m in MODELS]
              + [r_scores(m) for m in MODELS]
              + [coef_path(), meta_path("xgb"), meta_path("enet"),
                 OUT_DIR / "stage3_0_distress_events.parquet",
                 OUT_DIR / "compustat_annual_clean.parquet",
                 RAW_US / "comp_names.parquet"]):
        w(f"- `{p.relative_to(PROJECT_DIR)}`")
    w("")
    w("**Headline, distress.**")
    w("")
    w("| model | AUC-PR | AUC-ROC | Brier (calibrated) | rec@1% | rec@5% | rec@10% |")
    w("|---|---|---|---|---|---|---|")
    for m in MODELS:
        x = dres[m]["metrics"]
        w(f"| {m} | {x['AUC-PR']:.4f} | {x['AUC-ROC']:.4f} | "
          f"{x['Brier_cal']:.5f} | {x['rec@1%']:.3f} | {x['rec@5%']:.3f} | "
          f"{x['rec@10%']:.3f} |")
    w("")
    w("**Headline, rating.**")
    w("")
    w("| model | AUC-PR | AUC-ROC |")
    w("|---|---|---|")
    for m in MODELS:
        x = rres[m]["metrics"]
        w(f"| {m} | {x['AUC-PR']:.4f} | {x['AUC-ROC']:.4f} |")


def main():
    feats, panel, fit, dres, rres, dsc, rsc, coef, enet_meta = load()
    header(feats, fit, dres, rres)
    consensus, nz = pass1(feats, fit, dres, coef, enet_meta)
    # Pass 3 is COMPUTED before pass 2, which needs its decile directions, but
    # its lines are appended after so the document reads 1, 2, 3.
    direction, decile_lines = pass3(feats, fit)
    pass2(nz, direction)
    OUT.extend(decile_lines)
    pass4(dsc)
    pass5(feats, fit, dsc, consensus)
    pass6(dsc, dres)
    report_block(feats, consensus)

    out = OUT_DIR / OUT_NAME
    out.write_text("\n".join(OUT) + "\n")
    print(f"wrote {out.name}: {len(OUT):,} lines")
    print("No model was refitted. Nothing in output/ was overwritten.")


if __name__ == "__main__":
    main()
