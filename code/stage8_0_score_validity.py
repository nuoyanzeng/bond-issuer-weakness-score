"""
Stage 8.0 — Validity of the six scores. CHECK ONLY: reads, computes, PRINTS.

WRITES NOTHING. No parquet, no csv, no md. Nothing is modified, no scoring
model is refitted or altered, and `data_raw/` is read only. The only estimation
here is V4's small evaluation logits, which are fitted on outcomes and scores
that already exist and are never saved.

HIGHER SCORE = WEAKER.

THE ONE THING TO KEEP IN MIND ABOUT LEVELS. The US scores in the stage5_1
panels are out-of-fold sigmoid-CALIBRATED (D-47); the Chinese scores are RAW
`predict_proba` with no US calibrator (D-67). So the two sides' score LEVELS are
not comparable. Everything below is rank-based — AUC, Spearman, top-decile
overlap, percentile ventiles — except V4's `logit(score)` term, and V4 is run
WITHIN each market separately. No line in this script compares levels across
markets.

WHAT IS REUSED, AND HOW. V0 does not re-derive any outcome. It IMPORTS the
outcome construction from `code/check_china_score_validity.py` — `csmar`, `auc`,
`lookup`, `outcomes`, `china_items`, `us_items` and the constants `MODELS`,
`SCORES`, `MIN_CLASS` — so the definitions cannot drift from the run that
produced the figures the loud check asserts. Importing that module executes only
its constants and function definitions; its `main()` is behind a `__name__`
guard and does not run.

V0 LOUD CHECK. Six AUCs are asserted to 4 decimals against the earlier run, and
the script STOPS if any differs. V1 asserts the xgb/rf distress agreement
against `output/v4_results_summary.md` section 6 and also stops.

TWO ROW-BASIS DIFFERENCES, STATED BECAUSE THEY SILENTLY CHANGE NUMBERS.
  - V0, V3 and V4 de-duplicate the US panels on `gvkey x fyear`, because
    `lookup` maps `gvkey x fyear` to an item and a duplicated key would
    double-count one statement. This is what `check_china_score_validity.py`
    does.
  - V1 uses the US distress panel UN-de-duplicated, all 95,370 rows, because
    that is the row set `v4_results_summary.md` section 6 measured and the loud
    check has to be comparable to it.

Runtime: a few minutes. V4 fits 144 logits — 3 specifications x 4 outcomes x
(6 Chinese scores + 3 US distress + 3 US rating). The largest sample is the
de-duplicated US distress panel, and specification (c) carries 20 parameters
against 2 for (a) and (b). statsmodels is listed in requirements.txt.

Run:  python3 code/stage8_0_score_validity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The outcome construction, imported rather than copied, so it cannot drift.
from check_china_score_validity import (MIN_CLASS, MODELS, SCORES, auc, csmar,
                                        china_items, lookup, outcomes,
                                        us_items)

CN_PANEL = OUT_DIR / "stage6_4_china_score_panel.parquet"
CN_RANKED = OUT_DIR / "stage6_2_china_ranked.parquet"
US_DISTRESS = OUT_DIR / "stage5_1_score_panel_L3_K12.parquet"
US_RATING = OUT_DIR / "stage5_1_score_panel_rating_L3_K12.parquet"

# V0's outcomes, in the order they print. All are next-year except the last two.
V0_SPEC = [("interest cover < 1x  (MAIN: EBIT / 利息费用 · xint)", "intcov_t1"),
           ("EBITDA turns negative", "ebitdaneg_t1"),
           ("LOSS (ni < 0)", "loss_t1"),
           ("two consecutive loss years", "twoloss_t1"),
           ("LOSS at t+2", "loss_t2"),
           ("LOSS at t+3", "loss_t3")]

# V4 runs on these four only.
V4_OUTCOMES = ["loss_t1", "intcov_t1", "ebitdaneg_t1", "twoloss_t1"]

# V0 loud check: (panel label, outcome key, score, expected AUC).
V0_EXPECT = [("CHINA", "loss_t1", "distress_xgb", 0.7410),
             ("CHINA", "loss_t1", "rating_rf", 0.8406),
             ("CHINA", "intcov_t1", "distress_xgb", 0.7615),
             ("CHINA", "ebitdaneg_t1", "distress_xgb", 0.7946),
             ("CHINA", "twoloss_t1", "distress_xgb", 0.7523),
             ("US distress", "loss_t1", "xgb", 0.7742)]
# V1 loud check, from v4_results_summary.md section 6.
V1_EXPECT_RHO, V1_EXPECT_OVL = 0.7212, 80.7

CLIP_LO, CLIP_HI = 0.0001, 0.9999
N_VENT = 20
GOT = {}            # (label, outcome, score) -> AUC, filled by V0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def top_overlap(a, b):
    """Share of A's top decile also in B's top decile.

    The definition is copied from `stage5_2_results_summary.py:453-458`:
    n_top = round(0.10 * n), `nlargest` on each series, intersect the indexes.
    `nlargest` breaks ties by index order, so this is reproducible only on the
    same row ORDER as that script read — which is why V1 does not sort.
    """
    n_top = max(1, int(round(len(a) * 0.10)))
    ta, tb = set(a.nlargest(n_top).index), set(b.nlargest(n_top).index)
    return len(ta & tb) / n_top * 100, n_top


def fit_logit(y, X, groups):
    """Logit with cluster-robust covariance by firm.

    Returns (result, status). The covariance type changes only the standard
    errors; the log-likelihood, AIC and McFadden R2 are the ordinary MLE
    quantities, so the LR test below uses unadjusted likelihoods.
    """
    try:
        res = sm.Logit(y, X).fit(disp=0, maxiter=200, method="newton",
                                 cov_type="cluster",
                                 cov_kwds={"groups": groups})
    except Exception as exc:                        # separation, singularity
        try:
            res = sm.Logit(y, X).fit(disp=0, maxiter=500, method="bfgs",
                                     cov_type="cluster",
                                     cov_kwds={"groups": groups})
        except Exception as exc2:
            return None, f"FAILED  newton: {type(exc).__name__}; " \
                         f"bfgs: {type(exc2).__name__}: {exc2}"
        conv = res.mle_retvals.get("converged", False)
        return res, ("bfgs, converged" if conv else "bfgs, DID NOT CONVERGE")
    conv = res.mle_retvals.get("converged", False)
    return res, ("converged" if conv else "DID NOT CONVERGE")


def ventile_design(pct):
    """Intercept plus 19 ventile indicators, ventile 1 as the base.

    This is the "20 ventile dummies" specification written with a base
    category: an intercept and 19 indicators span exactly the same column space
    as 20 dummies with no intercept, and leaving the intercept in keeps the null
    model — and therefore McFadden's R2 — well defined.
    """
    v = np.clip(np.ceil(pct * N_VENT), 1, N_VENT).astype(int)
    X = pd.DataFrame({"const": 1.0}, index=pct.index)
    for k in range(2, N_VENT + 1):
        X[f"v{k:02d}"] = (v == k).astype(float)
    return X, pd.Series(v, index=pct.index)


# ---------------------------------------------------------------------------
# V0
# ---------------------------------------------------------------------------
def v0(cn, cn_items, usd, usr, us_it):
    print("\n" + "=" * 104)
    print("V0. PREDICTIVE VALIDITY — reproduces CHECK B of "
          "check_china_score_validity.py")
    print("=" * 104)
    print("  Outcomes are RAW and OBSERVED (CSMAR for China, "
          "compustat_annual_clean for the US), never imputed.")
    print("  The construction is IMPORTED from that script, not re-derived. "
          f"An AUC needs {MIN_CLASS} of each class.")

    panels = [("CHINA", cn, cn_items, SCORES),
              ("US distress", usd, us_it, MODELS),
              ("US rating", usr, us_it, MODELS)]
    ys = {}
    for label, panel, items, cols in panels:
        ys[label] = outcomes({o: lookup(panel, "gvkey", items, o)
                              for o in (0, 1, 2, 3)})

    for desc, key in V0_SPEC:
        print(f"\n  {desc}")
        print(f"      {'panel':<14}{'observed':>10}{'of':>9}{'base %':>9}"
              f"{'positives':>11}   AUC by score")
        for label, panel, items, cols in panels:
            y = ys[label][key] if key in ys[label].columns else None
            if y is None or y.notna().sum() == 0:
                print(f"      {label:<14}{'not computable on this side':>40}")
                continue
            obs = y.notna()
            base = y[obs].mean() * 100
            cells = []
            for s in cols:
                a, n, npos = auc(y, panel[f"score_{s}"])
                GOT[(label, key, s)] = a
                cells.append(f"{s}={a:.4f}" if a is not None
                             else f"{s}=n/a(n={n:,},pos={npos:,})")
            print(f"      {label:<14}{int(obs.sum()):>10,}{len(panel):>9,}"
                  f"{base:>9.1f}{int(y[obs].sum()):>11,}   "
                  + "  ".join(cells))

    # ---- the loud check ---------------------------------------------------
    print("\n  LOUD CHECK — the six asserted AUCs")
    bad = []
    for label, key, s, want in V0_EXPECT:
        got = GOT.get((label, key, s))
        ok = got is not None and round(got, 4) == round(want, 4)
        print(f"      {label:<14}{key:<14}{s:<14} expected {want:.4f}   got "
              + (f"{got:.4f}" if got is not None else "n/a")
              + ("   MATCH" if ok else "   DIFFERS"))
        if not ok:
            bad.append(f"{label} {key} {s}: expected {want:.4f}, got "
                       + (f"{got:.4f}" if got is not None else "n/a"))
    if bad:
        print("\n  STOP: V0 does not reproduce the earlier run.")
        for b in bad:
            print(f"      {b}")
        raise SystemExit("Nothing further computed.")
    print("      PASSED — all six reproduce to 4 decimals")
    return ys


# ---------------------------------------------------------------------------
# V1
# ---------------------------------------------------------------------------
def v1(cn, usd_full, usr):
    print("\n" + "=" * 104)
    print("V1. AGREEMENT AMONG THE THREE MODELS, within each score family")
    print("=" * 104)
    print("  Top-decile overlap is the share of model A's top 10% that is also "
          "in model B's top 10%,")
    print("  defined exactly as stage5_2_results_summary.py:453-458 "
          "(n_top = round(0.10 n), nlargest).")
    print("  The US distress rows here are NOT de-duplicated, so the basis "
          "matches v4 section 6.")

    sets = [("CHINA distress", cn, [f"distress_{m}" for m in MODELS]),
            ("CHINA rating", cn, [f"rating_{m}" for m in MODELS]),
            ("US distress", usd_full, list(MODELS)),
            ("US rating", usr, list(MODELS))]
    got_v1 = {}
    for label, panel, cols in sets:
        print(f"\n  {label}   {len(panel):,} rows, "
              f"{panel.gvkey.nunique():,} firms")
        print(f"      {'pair':<22}{'Spearman':>10}{'top-decile overlap':>22}"
              f"{'n_top':>9}")
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                sa, sb = panel[f"score_{a}"], panel[f"score_{b}"]
                rho = sa.corr(sb, method="spearman")
                ovl, n_top = top_overlap(sa, sb)
                got_v1[(label, a, b)] = (rho, ovl)
                print(f"      {a.split('_')[-1]} / {b.split('_')[-1]:<14}"
                      f"{rho:>10.4f}{ovl:>21.1f}%{n_top:>9,}")

    print("\n  LOUD CHECK — US distress xgb/rf against v4_results_summary.md "
          "section 6")
    rho, ovl = got_v1[("US distress", "xgb", "rf")]
    ok_r = round(rho, 4) == round(V1_EXPECT_RHO, 4)
    ok_o = round(ovl, 1) == round(V1_EXPECT_OVL, 1)
    print(f"      Spearman          expected {V1_EXPECT_RHO:.4f}   got "
          f"{rho:.4f}   {'MATCH' if ok_r else 'DIFFERS'}")
    print(f"      top-decile overlap expected {V1_EXPECT_OVL:.1f}%   got "
          f"{ovl:.1f}%   {'MATCH' if ok_o else 'DIFFERS'}")
    if not (ok_r and ok_o):
        print("      NOTE: `nlargest` breaks ties by index order, so the "
              "overlap can move by a row or two\n            if this script "
              "reads the panel in a different order from stage5_2. The "
              "Spearman\n            cannot move for that reason.")
        raise SystemExit("STOP: V1 does not reproduce v4 section 6. "
                         "Nothing further computed.")
    print("      PASSED")


# ---------------------------------------------------------------------------
# V2
# ---------------------------------------------------------------------------
def v2(cn, ys_cn):
    print("\n" + "=" * 104)
    print("V2. ROWS WITH INTWO UNDETERMINED")
    print("=" * 104)
    print("  INTWO is read from stage6_2_china_ranked.parquet, which holds it "
          "on its original 0/1/-1")
    print("  scale: it is exempt from the rank transform (D-44). -1 means "
          "'cannot be determined'.")

    r = pd.read_parquet(CN_RANKED, columns=["gvkey", "fyear", "INTWO"])
    n_before = len(cn)
    j = cn.merge(r, on=["gvkey", "fyear"], how="left")
    if len(j) != n_before:
        raise SystemExit(f"STOP: the INTWO join changed the row count, "
                         f"{n_before:,} -> {len(j):,}.")
    j.index = cn.index                      # keep the outcome index alignment
    n_na = int(j.INTWO.isna().sum())
    print(f"  joined on (gvkey, fyear): {len(j):,} rows, "
          f"{n_na:,} with no INTWO value")

    loss, icov = ys_cn["loss_t1"], ys_cn["intcov_t1"]
    print(f"\n  {'INTWO':<8}{'rows':>8}{'firms':>8}{'share':>8}"
          + "".join(f"{s.replace('distress','d').replace('rating','r'):>10}"
                    for s in SCORES)
          + f"{'loss t+1':>11}{'intcov t+1':>12}")
    print(f"  {'':<8}{'':>8}{'':>8}{'':>8}"
          + "".join(f"{'mean pct':>10}" for _ in SCORES)
          + f"{'rate %':>11}{'rate %':>12}")
    for state in (1.0, 0.0, -1.0, None):
        sel = j.INTWO.isna() if state is None else j.INTWO.eq(state)
        if not sel.any():
            continue
        lab = "NA" if state is None else f"{int(state):+d}"
        cells = "".join(f"{j.loc[sel, f'pct_{s}'].mean():>10.2f}"
                        for s in SCORES)
        lr = loss[sel].dropna()
        ir = icov[sel].dropna()
        print(f"  {lab:<8}{int(sel.sum()):>8,}{j.loc[sel,'gvkey'].nunique():>8,}"
              f"{sel.mean()*100:>7.2f}%" + cells
              + f"{lr.mean()*100:>10.2f}%{ir.mean()*100:>11.2f}%")
        print(f"  {'':<8}{'':>8}{'':>8}{'':>8}"
              + "".join(f"{'':>10}" for _ in SCORES)
              + f"{'n=' + format(len(lr), ','):>11}"
              + f"{'n=' + format(len(ir), ','):>12}")

    print(f"\n  INTWO = -1 by fiscal year")
    print(f"      {'fyear':>6}{'rows':>8}{'n(-1)':>8}{'share':>9}")
    for yv, g in j.groupby("fyear"):
        n1 = int(g.INTWO.eq(-1.0).sum())
        print(f"      {int(yv):>6}{len(g):>8,}{n1:>8,}"
              f"{n1/len(g)*100:>8.2f}%")


# ---------------------------------------------------------------------------
# V3
# ---------------------------------------------------------------------------
def v3(cn, usd, usr):
    print("\n" + "=" * 104)
    print("V3. AGREEMENT BETWEEN THE TWO SCORE FAMILIES, distress against "
          "rating")
    print("=" * 104)
    print(f"\n  CHINA   {len(cn):,} scored rows, {cn.gvkey.nunique():,} issuers")
    print(f"      {'model':<8}{'Spearman':>10}")
    for m in MODELS:
        rho = cn[f"score_distress_{m}"].corr(cn[f"score_rating_{m}"],
                                             method="spearman")
        print(f"      {m:<8}{rho:>10.4f}")

    # The two US panels are different samples; only their common rows can be
    # compared, and the key is gvkey x fyear after de-duplication.
    a = usd[["gvkey", "fyear"] + [f"score_{m}" for m in MODELS]].rename(
        columns={f"score_{m}": f"d_{m}" for m in MODELS})
    b = usr[["gvkey", "fyear"] + [f"score_{m}" for m in MODELS]].rename(
        columns={f"score_{m}": f"r_{m}" for m in MODELS})
    j = a.merge(b, on=["gvkey", "fyear"], how="inner")
    print(f"\n  US      distress panel {len(usd):,} rows, rating panel "
          f"{len(usr):,} rows")
    print(f"          rows present in BOTH on gvkey x fyear: {len(j):,}  "
          f"({j.gvkey.nunique():,} gvkeys)")
    print(f"      {'model':<8}{'Spearman':>10}")
    for m in MODELS:
        print(f"      {m:<8}{j[f'd_{m}'].corr(j[f'r_{m}'], method='spearman'):>10.4f}")


# ---------------------------------------------------------------------------
# V4
# ---------------------------------------------------------------------------
def v4_one(tag, y_raw, score, pct, firm):
    """The three regressions for one (market, score, outcome).

    `pct` arrives on the 0-1 scale, so (a)'s slope is the change in log-odds
    from the bottom of the market's score distribution to the top.
    """
    ok = y_raw.notna() & score.notna() & pct.notna()
    y = y_raw[ok].astype(int)
    s = score[ok]
    p = pct[ok]
    g = firm[ok]
    n, npos = len(y), int(y.sum())
    if npos < MIN_CLASS or (n - npos) < MIN_CLASS:
        print(f"    {tag:<34} n={n:,} pos={npos:,}  SKIPPED, below the "
              f"{MIN_CLASS} floor on one class")
        return

    # (b)'s regressor. Clipping bounds logit(score) where the score is 0 or 1.
    n_clip = int(((s < CLIP_LO) | (s > CLIP_HI)).sum())
    sc = np.clip(s, CLIP_LO, CLIP_HI)
    lg = pd.Series(np.log(sc / (1 - sc)), index=s.index, name="logit_score")

    const = pd.Series(1.0, index=s.index, name="const")
    Xa = pd.concat([const, p.rename("percentile")], axis=1)
    Xb = pd.concat([const, lg], axis=1)
    # (c) uses the SAME percentile column as (a), cut into ventiles.
    Xc, vent = ventile_design(p)

    pos_by_v = y.groupby(vent).sum()
    rate_by_v = y.groupby(vent).mean()
    empty = sorted(int(k) for k in pos_by_v.index[pos_by_v == 0])
    full = sorted(int(k) for k in rate_by_v.index[rate_by_v == 1])

    print(f"    {tag:<34} n={n:,}  pos={npos:,} ({npos/n*100:.2f}%)  "
          f"clipped={n_clip:,}  firms={g.nunique():,}")
    if empty or full:
        print(f"        ventiles with NO positives: {empty or 'none'};  "
              f"with ALL positives: {full or 'none'}  "
              f"(a zero cell drives its coefficient to -inf)")

    res = {}
    for name, X, slope in (("(a) percentile", Xa, "percentile"),
                           ("(b) logit(score)", Xb, "logit_score"),
                           ("(c) 20 ventile dummies", Xc, None)):
        r, status = fit_logit(y, X, g)
        res[name] = r
        if r is None:
            print(f"        {name:<26} {status}")
            continue
        line = (f"        {name:<26} n={n:,}  k={X.shape[1]:>3}  "
                f"llf={r.llf:>12.3f}  pseudoR2={r.prsquared:>7.4f}  "
                f"AIC={r.aic:>12.3f}   {status}")
        if slope is not None and slope in r.params.index:
            line += (f"\n            slope {r.params[slope]:+.4f}  "
                     f"cluster-robust SE {r.bse[slope]:.4f}  "
                     f"z={r.params[slope] / r.bse[slope]:+.2f}")
        print(line)

    ra, rb, rc = (res.get("(a) percentile"), res.get("(b) logit(score)"),
                  res.get("(c) 20 ventile dummies"))
    if ra is not None and rb is not None:
        print(f"        llf(b) - llf(a) = {rb.llf - ra.llf:+.3f}")
    if rc is not None and ra is not None and rb is not None:
        print(f"        llf(c) - llf(a) = {rc.llf - ra.llf:+.3f}     "
              f"llf(c) - llf(b) = {rc.llf - rb.llf:+.3f}")


def v4(cn, usd, usr, ys):
    print("\n" + "=" * 104)
    print("V4. DOES THE MAGNITUDE CARRY INFORMATION BEYOND RANK?")
    print("=" * 104)
    print("  WITHIN each market separately — nothing here compares levels "
          "across markets.")
    print("  (a) the score's percentile, ONE linear term.")
    print(f"  (b) logit(score), ONE linear term, score clipped to "
          f"[{CLIP_LO}, {CLIP_HI}].")
    print("  (c) 20 ventile dummies of the SAME percentile as (a): intercept "
          "+ 19 indicators, ventile 1")
    print("      the base. Same column space as 20 dummies with no intercept, "
          "but with a well-defined")
    print("      null model for McFadden's R2.")
    print("  Slopes for (a) and (b) carry standard errors clustered by FIRM. "
          "The three log-likelihood")
    print("  differences are printed as differences; no test statistic is "
          "formed from them.")

    # ---- the percentile each market's (a) and (c) are built on -------------
    pcts = {}
    print("\n  THE PERCENTILE COLUMN")
    for s in SCORES:
        col = cn[f"pct_{s}"]
        lo, hi = col.min(), col.max()
        scale = 100.0 if hi > 1.5 else 1.0
        pcts[("CHINA", s)] = col / scale
        print(f"      CHINA       pct_{s:<14} saved column, range "
              f"[{lo:.4f}, {hi:.4f}] -> read as 0-{int(scale)}, "
              f"divided by {int(scale)} for use here")
    # The US panels DO carry pct_ columns, but they were computed on the
    # UN-de-duplicated panels (stage5_1), whereas V4 runs on the de-duplicated
    # ones. Recomputed here on the de-duplicated panel's FULL rows — before any
    # outcome is joined — so the percentile is the market-wide position of the
    # rows V4 actually uses. The saved columns are left untouched.
    for label, panel in (("US distress", usd), ("US rating", usr)):
        for m in MODELS:
            pcts[(label, m)] = panel[f"score_{m}"].rank(pct=True)
        print(f"      {label:<12}recomputed on this panel's {len(panel):,} "
              f"full rows (rank pct). The saved pct_ columns are on the "
              f"un-de-duplicated basis and are NOT used.")

    jobs = [("CHINA", cn, SCORES, ys["CHINA"]),
            ("US distress", usd, MODELS, ys["US distress"]),
            ("US rating", usr, MODELS, ys["US rating"])]
    for label, panel, cols, y_all in jobs:
        print(f"\n  {'=' * 100}\n  {label}   {len(panel):,} rows\n  {'=' * 100}")
        for key in V4_OUTCOMES:
            if key not in y_all.columns or y_all[key].notna().sum() == 0:
                print(f"\n  outcome {key}: not computable on this side")
                continue
            print(f"\n  outcome {key}")
            for s in cols:
                v4_one(s, y_all[key], panel[f"score_{s}"],
                       pcts[(label, s)], panel["gvkey"])


# ---------------------------------------------------------------------------
def main():
    print("=" * 104)
    print("STAGE 8.0 — SCORE VALIDITY CHECKS.  READ-ONLY: nothing is written, "
          "nothing is modified,")
    print("            no scoring model is refitted.  HIGHER SCORE = WEAKER.")
    print("=" * 104)
    print("  The US scores are out-of-fold sigmoid-CALIBRATED (D-47); the "
          "Chinese scores are RAW with")
    print("  no US calibrator (D-67). Everything below is RANK-BASED except "
          "V4's logit(score) term,")
    print("  and V4 runs within each market only. No line compares score "
          "LEVELS across markets.")

    cn = pd.read_parquet(CN_PANEL)
    n_cn_all = len(cn)
    cn = cn[cn[f"score_{SCORES[0]}"].notna()].reset_index(drop=True)
    print(f"\n  {CN_PANEL.name}: {n_cn_all:,} rows, "
          f"{len(cn):,} scored ({n_cn_all - len(cn):,} unscored, dropped), "
          f"{cn.gvkey.nunique():,} Orgids")
    if cn[[f"score_{s}" for s in SCORES]].isna().values.any():
        raise SystemExit("STOP: a scored Chinese row has a missing score on "
                         "one of the six.")

    # V1 needs the US distress panel as v4 section 6 read it; everything that
    # joins an OUTCOME needs it de-duplicated on gvkey x fyear.
    usd_full = pd.read_parquet(US_DISTRESS).reset_index(drop=True)
    usd = usd_full.drop_duplicates(["gvkey", "fyear"]).reset_index(drop=True)
    usr_full = pd.read_parquet(US_RATING).reset_index(drop=True)
    usr = usr_full.drop_duplicates(["gvkey", "fyear"]).reset_index(drop=True)
    print(f"  {US_DISTRESS.name}: {len(usd_full):,} rows, "
          f"{len(usd):,} after de-duplicating on gvkey x fyear")
    print(f"  {US_RATING.name}: {len(usr_full):,} rows, "
          f"{len(usr):,} after de-duplicating on gvkey x fyear")

    # The raw item tables. The bridge is needed to put CSMAR rows on Orgid.
    print("\n  building the raw item tables (CSMAR and Compustat) — this is "
          "the slow part")
    bm = csmar("债券基本情况表.csv", ["Liscd", "Sctcd", "Orgid"]).drop_duplicates()
    cn_items = china_items(bm)
    us_it = us_items()
    print(f"      Chinese items {cn_items.shape[0]:,} (Orgid, fyear) cells; "
          f"US items {us_it.shape[0]:,} (gvkey, fyear) cells")

    ys = v0(cn, cn_items, usd, usr, us_it)
    v1(cn, usd_full, usr)
    v2(cn, ys["CHINA"])
    v3(cn, usd, usr)
    v4(cn, usd, usr, ys)

    print("\n" + "=" * 104)
    print("NO VERDICT IS STATED. Every line above is a measured quantity. "
          "Nothing was written.")
    print("=" * 104)


if __name__ == "__main__":
    main()
