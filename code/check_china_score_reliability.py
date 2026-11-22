"""
Reliability validation of the transferred Chinese scores. READ-ONLY, prints only.

The question: are the six transferred scores usable in China, and if so which.

CHECK 1  do they predict Chinese ratings, and by more than firm size alone?
CHECK 2  is a firm's ordering stable from year to year, against the US benchmark?
CHECK 3  why do the three distress models disagree in China?

THIS SCRIPT STATES NO VERDICT. Every line it prints is a measured quantity; the
reading is done afterwards, by people.

Run:  python3 code/check_china_score_reliability.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
CH = PROJECT_DIR / "data_raw" / "china"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage3_10_prepare_model_input as s3
import stage6_1_impute_china as s61
from stage6_0_construct_china_features import _unwrap

OUTCOMES = ["distress", "rating"]
MODELS = ["xgb", "rf", "enet"]
SCORES = [f"{o}_{m}" for o in OUTCOMES for m in MODELS]
YEAR_FROM = 2010
MIN_CLASS = 100          # an AUC on fewer than this many of either class is n/a

# D-64's notch scale. A-1 is a SHORT-TERM code, not on this scale, and is
# excluded rather than forced onto it.
NOTCH = {"AAA+": 0, "AAA": 1, "AAA-": 2, "AA+": 3, "AA": 4, "AA-": 5,
         "A+": 6, "A": 7, "A-": 8, "BBB+": 9, "BBB": 10, "BBB-": 11,
         "BB+": 12, "BB": 13, "BB-": 14, "B+": 15, "B": 16, "B-": 17,
         "CCC": 18, "CC": 19, "C": 20}
NOT_LONG_TERM = {"A-1"}
SPEC_NOTCH = 12          # BB+ and below = below BBB-, the US speculative line


def csmar(fname, want):
    """Read named columns from one CSMAR file, following stage6_0's conventions."""
    h = pd.read_csv(CH / fname, nrows=0, encoding="utf-8-sig")
    cols = [c.replace('="', "").replace('"', "").strip() for c in h.columns]
    idx = sorted(cols.index(w) for w in want)
    d = pd.read_csv(CH / fname, usecols=idx, skiprows=[1, 2], header=0,
                    dtype=str, encoding="utf-8-sig", low_memory=False)
    d.columns = [cols[i] for i in idx]
    for c in d.columns:
        d[c] = _unwrap(d[c])
    return d


def auc(y, s):
    """AUC, or None where either class is too thin to mean anything."""
    ok = pd.notna(y) & pd.notna(s)
    y, s = y[ok], s[ok]
    if int(y.sum()) < MIN_CLASS or int((~y.astype(bool)).sum()) < MIN_CLASS:
        return None
    return roc_auc_score(y.astype(int), s)


def stability(df, key, pct_cols, label):
    """Spearman of a firm's percentile against its own the year before."""
    d = df.drop_duplicates([key, "fyear"])
    prev = d.copy()
    prev["fyear"] = prev["fyear"] + 1
    m = d.merge(prev, on=[key, "fyear"], suffixes=("", "_prev"))
    print(f"\n  {label}   {len(m):,} consecutive firm-year pairs, "
          f"{m[key].nunique():,} firms")
    print(f"      {'score':<16}{'Spearman':>10}{'med |dpct|':>12}"
          f"{'p90 |dpct|':>12}{'moved >25':>11}")
    out = {}
    for c in pct_cols:
        a, b = m[c], m[f"{c}_prev"]
        ok = a.notna() & b.notna()
        if ok.sum() < 100:
            print(f"      {c:<16}{'n/a':>10}")
            continue
        d_abs = (a[ok] - b[ok]).abs()
        rho = a[ok].corr(b[ok], method="spearman")
        out[c] = rho
        print(f"      {c:<16}{rho:>10.4f}{d_abs.median():>12.1f}"
              f"{d_abs.quantile(.9):>12.1f}{(d_abs > 25).mean() * 100:>10.1f}%")
    return out


def main():
    print("=" * 100)
    print("RELIABILITY OF THE TRANSFERRED CHINESE SCORES — read-only, "
          "nothing written")
    print("=" * 100)

    panel = pd.read_parquet(OUT_DIR / "stage6_4_china_score_panel.parquet")
    scored = panel[f"score_{SCORES[0]}"].notna()
    print(f"  stage6_4 panel: {len(panel):,} rows, {panel.gvkey.nunique():,} "
          f"issuers; {int(scored.sum()):,} scored")

    # ---------------------------------------------------------------- CHECK 1
    print("\n" + "=" * 100)
    print("CHECK 1 — do the scores predict Chinese ratings?")
    print("=" * 100)

    bm = csmar("债券基本情况表.csv", ["Liscd", "Sctcd", "Orgid"]).drop_duplicates()
    rt = csmar("债券及主体评级情况表.csv",
               ["Liscd", "Sctcd", "RatingDate", "Ctcr"])
    have = ~rt.Ctcr.isin(["", "nan"]) & rt.Ctcr.notna()
    unknown = set(rt.loc[have, "Ctcr"].unique()) - set(NOTCH) - NOT_LONG_TERM
    if unknown:
        raise SystemExit(f"STOP: Ctcr values not on the notch scale and not "
                         f"declared as excluded: {sorted(unknown)}")
    rt = rt.merge(bm, on=["Liscd", "Sctcd"], how="left")
    rt["notch"] = rt.Ctcr.map(NOTCH)
    rt = rt[rt.notch.notna() & rt.Orgid.notna()]
    rt["year"] = pd.to_datetime(rt.RatingDate, errors="coerce").dt.year

    # LOWEST rating in the year = HIGHEST notch. The harder test: the hypothesis
    # is that Chinese ratings are too high, so we take the least generous one.
    rp = rt.groupby(["Orgid", "year"])["notch"].max().reset_index()
    if rp.duplicated(["Orgid", "year"]).any():
        raise SystemExit("STOP: the rating panel is not unique on (Orgid, year).")

    # Firm-year total assets, as stage6_0 reads them: consolidated, 12-31.
    bs = csmar("企业债公司债资产负债表.csv",
               ["Liscd", "Sctcd", "Accper", "Typrep", "A001000000"])
    bs = bs[bs.Typrep.eq("A") & bs.Accper.str.endswith("12-31")]
    bs = bs.merge(bm, on=["Liscd", "Sctcd"], how="left")
    bs["at"] = pd.to_numeric(bs["A001000000"], errors="coerce")
    bs = bs[(bs["at"] > 0) & bs.Orgid.notna()]
    bs["year"] = bs.Accper.str.slice(0, 4).astype(int)
    size = bs.groupby(["Orgid", "year"])["at"].median()

    m = panel.merge(rp, left_on=["gvkey", "fyear"], right_on=["Orgid", "year"],
                    how="inner")
    m = m[(m.fyear >= YEAR_FROM) & m[f"score_{SCORES[0]}"].notna()].copy()
    m["log_at"] = np.log(pd.MultiIndex.from_arrays(
        [m.gvkey, m.fyear]).map(size).astype("float64"))
    cov = m.log_at.notna().mean()
    if cov < 0.95:
        raise SystemExit(f"STOP: total assets cover only {cov:.1%} of the "
                         "matched sample; the size reference would be a "
                         "different sample from the scores.")
    print(f"  matched, {YEAR_FROM}+, scored: {len(m):,} issuer-years, "
          f"{m.gvkey.nunique():,} issuers; size on {cov:.2%}")

    spec = m.notch >= SPEC_NOTCH
    print(f"\n  WHY THE US SPECULATIVE LINE IS NOT USED — descriptive, not an AUC")
    print(f"      below BBB- (the US line applied literally): "
          f"{int(spec.sum()):,} rows ({spec.mean() * 100:.3f}%), "
          f"{m.loc[spec, 'gvkey'].nunique():,} issuers")
    print(f"      That is far below the {MIN_CLASS}-positive floor, and in China "
          f"it separates ~99% of ordinary firms\n      from ~1% already in "
          f"trouble — not the question the US line asks.")

    for cut_lab, cut, note in [
        ("AA+ or above", m.notch <= 3,
         "PRIMARY — the Chinese counterpart of the US speculative-grade split "
         "BY SHARE OF THE MARKET,\n      not by letter: 46% here against 43.8% "
         "speculative in the US rating sample (D-42)."),
        ("is AAA", m.notch <= 1, "second cut")]:
        print(f"\n  CUT: {cut_lab}   {int(cut.sum()):,} positives "
              f"({cut.mean() * 100:.1f}% / {(1 - cut.mean()) * 100:.1f}%)")
        print(f"      {note}")
        size_auc = auc(cut, m.log_at)
        sa = f"{size_auc:.4f}" if size_auc else "n/a"
        print(f"\n      {'score':<22}{'AUC':>9}{'size-only AUC':>15}{'margin':>10}")
        for s in SCORES:
            a = auc(cut, m[f"score_{s}"])
            if a is None:
                print(f"      {s:<22}{'n/a':>9}")
                continue
            marg = f"{a - size_auc:+.4f}" if size_auc else "n/a"
            print(f"      {s:<22}{a:>9.4f}{sa:>15}{marg:>10}")
        print(f"      The size line is a REFERENCE, not a control: a score that "
              f"beats it by little is\n      mostly reporting how large the firm "
              f"is.")

    # ---------------------------------------------------------------- CHECK 2
    print("\n" + "=" * 100)
    print("CHECK 2 — is the ordering stable year to year?")
    print("=" * 100)
    print("  Fundamentals move gradually, so a firm swinging across the "
          "distribution is noise.")
    cn = stability(panel, "gvkey", [f"pct_{s}" for s in SCORES], "CHINA")
    us = pd.read_parquet(OUT_DIR / "stage5_1_score_panel_L3_K12.parquet")
    us_rho = stability(us, "gvkey", [f"pct_{m_}" for m_ in MODELS],
                       "US distress (benchmark; de-duplicated on gvkey x fyear)")
    print(f"\n      SIDE BY SIDE, distress only")
    print(f"      {'model':<10}{'China':>10}{'US':>10}{'diff':>10}")
    for m_ in MODELS:
        c, u = cn.get(f"pct_distress_{m_}"), us_rho.get(f"pct_{m_}")
        if c is not None and u is not None:
            print(f"      {m_:<10}{c:>10.4f}{u:>10.4f}{c - u:>+10.4f}")

    # ---------------------------------------------------------------- CHECK 3
    print("\n" + "=" * 100)
    print("CHECK 3 — why do the three distress models disagree in China?")
    print("=" * 100)

    print(f"\n  PAIRWISE SPEARMAN — is the Chinese spread transfer, or just what "
          f"these models do?")
    print(f"      {'pair':<12}{'China':>10}{'US':>10}{'diff':>10}")
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            c = panel[f"score_distress_{a}"].corr(
                panel[f"score_distress_{b}"], method="spearman")
            u = us[f"score_{a}"].corr(us[f"score_{b}"], method="spearman")
            print(f"      {a + '/' + b:<12}{c:>10.4f}{u:>10.4f}{c - u:>+10.4f}")

    # Disagreement per row: spread of the three distress percentiles, 0-100.
    pc = [f"pct_distress_{m_}" for m_ in MODELS]
    d = panel.loc[scored, ["gvkey", "fyear"] + pc].copy()
    d["spread"] = d[pc].max(axis=1) - d[pc].min(axis=1)
    d["mean_pct"] = d[pc].mean(axis=1)
    print(f"\n  SPREAD of the three distress percentiles (max - min, 0-100): "
          f"median {d.spread.median():.1f}, mean {d.spread.mean():.1f}, "
          f"p90 {d.spread.quantile(.9):.1f}")

    # (a) by imputation — the EXACT Level-1/Level-2 split, recomputed with the
    #     same functions stage6_1 used.
    raw = pd.read_parquet(OUT_DIR / "stage6_0_china_features.parquet")
    imp = pd.read_parquet(OUT_DIR / "stage6_1_china_imputed.parquet")
    feats = s61.read_feature_names()
    base = imp[["gvkey", "datadate", "fyear", "ind_cn"]].merge(
        raw[["gvkey", "fyear"] + feats], on=["gvkey", "fyear"], how="left")
    base["date_avail"] = base["datadate"]
    s3.IMPUTE_INDUSTRY = s61.IND_COL
    X = base[feats].astype("float64")
    miss = X.isna()
    ok = pd.DataFrame(False, index=X.index, columns=feats)
    ok[[f for f in feats if f not in s3.IMPUTE_EXEMPT]] = True
    l1 = s3.level1_firm_past(base, feats)
    f1 = miss & l1.notna() & ok
    after1 = X.where(~f1, l1)
    cell, _, _, _ = s3.level2_industry_cell(base, feats)
    f2 = (after1.isna() & ok) & cell.notna()
    fills = pd.DataFrame({"gvkey": base.gvkey, "fyear": base.fyear,
                          "n_l1": f1.sum(axis=1), "n_l2": f2.sum(axis=1)})
    d = d.merge(fills, on=["gvkey", "fyear"], how="left")
    print(f"\n  (a) BY IMPUTATION — does the disagreement come from filled cells?")
    print(f"      {'Level-2 fills in row':<24}{'rows':>9}{'mean spread':>13}")
    for lab, sel in [("0", d.n_l2 == 0), ("1-2", d.n_l2.between(1, 2)),
                     ("3-5", d.n_l2.between(3, 5)), ("6-9", d.n_l2.between(6, 9)),
                     ("10+", d.n_l2 >= 10)]:
        if sel.sum():
            print(f"      {lab:<24}{int(sel.sum()):>9,}"
                  f"{d.loc[sel, 'spread'].mean():>13.1f}")
    print(f"      (Level-1 fills for reference: median {d.n_l1.median():.0f}, "
          f"mean {d.n_l1.mean():.2f} per row)")

    # (b) by position — do they agree where it matters?
    print(f"\n  (b) BY POSITION — do the three agree about the riskiest firms?")
    dec = pd.qcut(d.mean_pct, 10, labels=False, duplicates="drop") + 1
    print(f"      {'decile of mean pct':<24}{'rows':>9}{'mean spread':>13}")
    for q in sorted(pd.unique(dec.dropna())):
        sel = dec == q
        tag = "  <- riskiest" if q == dec.max() else ""
        print(f"      {int(q):<24}{int(sel.sum()):>9,}"
              f"{d.loc[sel, 'spread'].mean():>13.1f}{tag}")
    n_top = max(1, int(round(len(d) * 0.10)))
    print(f"\n      pairwise TOP-DECILE overlap ({n_top:,} rows each)")
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            ta = set(d[f"pct_distress_{a}"].nlargest(n_top).index)
            tb = set(d[f"pct_distress_{b}"].nlargest(n_top).index)
            print(f"      {a + '/' + b:<12}{len(ta & tb) / n_top * 100:>8.1f}%")

    # (c) by INTWO — 22% of Chinese rows are coded -1, against rare in the US.
    print(f"\n  (c) BY INTWO — rows coded -1 mean 'could not be determined'")
    rk = pd.read_parquet(OUT_DIR / "stage6_2_china_ranked.parquet",
                         columns=["gvkey", "fyear", "INTWO"])
    d = d.merge(rk, on=["gvkey", "fyear"], how="left")
    print(f"      {'INTWO':<12}{'rows':>9}{'mean spread':>13}"
          + "".join(f"{'med ' + m_:>12}" for m_ in MODELS))
    for v, lab in [(-1.0, "-1 (unknown)"), (0.0, "0"), (1.0, "1")]:
        sel = d.INTWO == v
        if not sel.sum():
            continue
        meds = "".join(f"{d.loc[sel, f'pct_distress_{m_}'].median():>12.1f}"
                       for m_ in MODELS)
        print(f"      {lab:<12}{int(sel.sum()):>9,}"
              f"{d.loc[sel, 'spread'].mean():>13.1f}{meds}")

    print("\n" + "=" * 100)
    print("NO VERDICT IS STATED HERE. Every line above is a measured quantity.")
    print("=" * 100)


if __name__ == "__main__":
    main()
