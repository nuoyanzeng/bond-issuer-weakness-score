"""
Do the transferred scores measure anything real in China? READ-ONLY, prints only.

CHECK B (first, because it matters most) — can the score predict the firm's OWN
future financials? This involves no ratings at all. If it can, the score is
measuring something real and any failure against ratings is a fact about
ratings. If it cannot, the score is not working in China.

CHECK A — does the relationship with ratings change over time?

Outcomes are read RAW and OBSERVED — CSMAR for China, compustat_annual_clean for
the US — never from an imputed panel, so a filled value can never become an
outcome. The US run is the benchmark: a Chinese AUC only means something beside
what these models manage on the data they were fitted on.

THIS SCRIPT STATES NO VERDICT. Every line is a measured quantity.

Run:  python3 code/check_china_score_validity.py
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
from stage6_0_construct_china_features import _unwrap

MODELS = ["xgb", "rf", "enet"]
SCORES = [f"{o}_{m}" for o in ("distress", "rating") for m in MODELS]
MIN_CLASS = 100          # an AUC on fewer than this many of either class is n/a
LEV_JUMP = 0.10          # "leverage rises by more than 10 percentage points"
DE_JUMP = 1.0            # Debt/EBITDA deteriorating by more than one turn

# THE TWO GATES. Both are floors on things already known to be true, so a
# failure means this script is broken rather than that the models are weak.
# stage4_1 measured 0.8656 / 0.8764 / 0.8540 for xgb / rf / enet against the US
# default label; 0.80 leaves room for nothing but a real fault.
US_DEFAULT_FLOOR = 0.80
# Correct alignment returns ~0.85 on same-year loss; a misaligned merge returns
# ~0.50. The floor sits between them, deliberately loose.
SAME_YEAR_LOSS_FLOOR = 0.70

NOTCH = {"AAA+": 0, "AAA": 1, "AAA-": 2, "AA+": 3, "AA": 4, "AA-": 5,
         "A+": 6, "A": 7, "A-": 8, "BBB+": 9, "BBB": 10, "BBB-": 11,
         "BB+": 12, "BB": 13, "BB-": 14, "B+": 15, "B": 16, "B-": 17,
         "CCC": 18, "CC": 19, "C": 20}
NOT_LONG_TERM = {"A-1"}          # short-term code, not on the notch scale
PERIODS = [("2010-2014", 2010, 2014), ("2015-2017", 2015, 2017),
           ("2018-2020", 2018, 2020), ("2021-2024", 2021, 2024)]


def csmar(fname, want):
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
    """AUC, or (None, n, n_pos) where either class is below the floor.

    The index equality check is not ceremony: combining two Series on different
    indexes aligns them BY LABEL and quietly returns the overlap, which is how
    a broken AUC once passed for a finding.
    """
    if not y.index.equals(s.index):
        raise SystemExit(
            f"STOP: outcome and score are on different indexes "
            f"({len(y):,} vs {len(s):,}); pandas would align them by label and "
            "the result would be an accidental overlap, not a sample.")
    ok = pd.notna(y) & pd.notna(s)
    y, s = y[ok].astype(int), s[ok]
    n, npos = len(y), int(y.sum())
    if npos < MIN_CLASS or (n - npos) < MIN_CLASS:
        return None, n, npos
    return roc_auc_score(y, s), n, npos


def lookup(panel, key, items, offset):
    """Raw items for (firm, fyear + offset), ON THE PANEL'S OWN INDEX.

    THE `set_index` IS THE WHOLE POINT, and its absence was a real bug.
    `MultiIndex.map` returns a bare Index, so building a DataFrame from it
    gives a fresh RangeIndex 0..n-1. The panel's index is NOT that — it is
    gappy, because the panel was filtered — so every later `y & s` aligned two
    unrelated index namespaces BY LABEL and silently kept their accidental
    overlap. Measured when it happened: the US t+1-loss AUC read 0.5115 on
    41,916 rows instead of 0.7742 on 59,956, and the surviving rows were 2.5
    years older with twice the default rate. It looked exactly like a finding.
    """
    idx = pd.MultiIndex.from_arrays([panel[key], panel.fyear + offset])
    return pd.DataFrame({k: idx.map(items[k]) for k in items.columns},
                        index=panel.index)


def _cover(num, den, thresh, den_positive=True):
    """A ratio test that REFUSES a non-positive denominator.

    A zero or negative denominator does not raise — it silently flips or
    explodes the ratio, which is the D-17 pathology and the single most
    dangerous thing in this file. Such rows become NA, not False.
    """
    ok = num.notna() & den.notna() & ((den > 0) if den_positive else den.ne(0))
    return (num / den.where(ok) < thresh).where(ok)


def outcomes(f):
    """Every objective outcome, from raw observed items.

    `f` maps an offset (0, 1, 2, 3) to a frame of items on the panel's index.
    EVERY t+1 outcome has a SAME-YEAR twin: the t version is the alignment
    control, and it is what caught the index bug that made the whole of Check B
    read 0.51. A t+1 number without its t twin is not to be trusted.
    """
    idx = f[0].index
    for k, v in f.items():
        if not v.index.equals(idx):
            raise SystemExit(f"STOP: the t+{k} frame is on a different index.")
    out = {}

    def lev(o):
        return f[o]["lt"] / f[o]["at"]

    def de(o):
        """Debt / EBITDA, NA unless EBITDA is positive."""
        return (f[o]["debt"] / f[o]["ebitda"].where(f[o]["ebitda"] > 0))

    for o, tag in [(0, "t0"), (1, "t1")]:
        d = f[o]
        # 1 MAIN — interest cover below 1x, on TRUE interest expense.
        out[f"intcov_{tag}"] = _cover(d["ebit"], d["interest"], 1.0)
        # 1 ROBUSTNESS — China only; finexp is NaN on the US side, so this
        # column simply comes out empty there rather than being faked.
        out[f"intcovfx_{tag}"] = _cover(d["ebit_fx"], d["finexp"], 1.0)
        # 2a EBITDA negative.
        out[f"ebitdaneg_{tag}"] = (d["ebitda"] < 0).where(d["ebitda"].notna())
        # 3 loss.
        out[f"loss_{tag}"] = (d["ni"] < 0).where(d["ni"].notna())
        # 4 current ratio below 1.
        out[f"currat_{tag}"] = _cover(d["ca"], d["cl"], 1.0)
        # the original two, kept.
        out[f"ocfneg_{tag}"] = (d["oancf"] < 0).where(d["oancf"].notna())

    # 2b Debt/EBITDA deteriorating by more than DE_JUMP turns, both years
    #    needing a positive EBITDA for the ratio to exist at all.
    out["debtebitda_t1"] = ((de(1) - de(0)) > DE_JUMP).where(
        de(0).notna() & de(1).notna())
    out["debtebitda_t0"] = ((de(0) - de(-1)) > DE_JUMP).where(
        de(-1).notna() & de(0).notna()) if -1 in f else pd.Series(
        np.nan, index=idx)

    # leverage jump, kept from the first version.
    out["levup_t1"] = ((lev(1) - lev(0)) > LEV_JUMP).where(
        lev(0).notna() & lev(1).notna())
    out["levup_t0"] = pd.Series(np.nan, index=idx)

    # 3 longer horizons, and 5 two consecutive loss years.
    for o in (2, 3):
        if o in f:
            out[f"loss_t{o}"] = (f[o]["ni"] < 0).where(f[o]["ni"].notna())
    if 2 in f:
        a = (f[1]["ni"] < 0).where(f[1]["ni"].notna())
        b = (f[2]["ni"] < 0).where(f[2]["ni"].notna())
        out["twoloss_t1"] = (a.astype("boolean") & b.astype("boolean"))
        c = (f[0]["ni"] < 0).where(f[0]["ni"].notna())
        out["twoloss_t0"] = (c.astype("boolean") & a.astype("boolean"))
    return pd.DataFrame(out, index=idx)


# The CSMAR codes, verified against the codebook label rows.
CN_CODES = {
    "企业债公司债资产负债表.csv": {
        "A001000000": "at", "A002000000": "lt",
        "A001100000": "ca", "A002100000": "cl",
        "A002101000": "d_st", "A002201000": "d_lt",
        "A002203000": "d_bond", "A002125000": "d_due"},
    "企业债公司债利润分配表.csv": {
        "B002000101": "ni", "B001000000": "pretax",
        "B001211100": "interest",     # 其中：利息费用 — the TRUE interest line
        "B001211000": "finexp"},      # 财务费用 — the bundle, robustness only
    "企业债公司债现金流量表(间接法).csv": {
        "D000103": "dep", "D000104": "amort", "D000105": "amort_lt"},
    "企业债公司债现金流量表(直接法).csv": {"C001000000": "oancf"},
}


def china_items(bm):
    """The common item schema, from CSMAR, consolidated (Typrep 'A') at 12-31."""
    parts = {}
    for fn, mapping in CN_CODES.items():
        codes = list(mapping)
        d = csmar(fn, ["Liscd", "Sctcd", "Accper", "Typrep"] + codes)
        d = d[d.Typrep.eq("A") & d.Accper.str.endswith("12-31")]
        d = d.merge(bm, on=["Liscd", "Sctcd"], how="left")
        d["fyear"] = d.Accper.str.slice(0, 4).astype(int)
        for c in codes:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        g = d[d.Orgid.notna()].groupby(["Orgid", "fyear"])[codes].median()
        for c, name in mapping.items():
            parts[name] = g[c]
    x = pd.DataFrame(parts)
    # EBIT = pre-tax profit + interest expense, the standard construction.
    x["ebit"] = x["pretax"] + x["interest"]
    # The robustness EBIT pairs with 财务费用 so numerator and denominator are
    # consistent; negative 财务费用 is excluded downstream by _cover's den > 0.
    x["ebit_fx"] = x["pretax"] + x["finexp"]
    x["ebitda"] = x["ebit"] + x[["dep", "amort", "amort_lt"]].sum(
        axis=1, min_count=1)
    # INTEREST-BEARING DEBT. CSMAR leaves an absent line BLANK rather than
    # zero, so a firm with no bonds payable has NaN there, not 0. Summing with
    # min_count=1 therefore reads a missing sub-line as zero, which understates
    # debt for firms that have it and did not report it. Stated, not hidden:
    # this is the one place here that treats a blank as a number.
    x["debt"] = x[["d_st", "d_lt", "d_bond", "d_due"]].sum(axis=1, min_count=1)
    return x


def us_items():
    """The same schema from Compustat.

    act and lct are ZERO-FILLED under CZ's D-11 list on ~23% of rows, and a
    zero current-liabilities makes the current ratio infinite rather than
    raising. Those rows are set back to NA here, which is why this reads the
    _was_missing flags rather than the values alone.
    """
    cols = ["gvkey", "fyear", "ni", "oancf", "lt", "at", "xint", "ebit",
            "ebitda", "act", "lct", "dltt", "dlc",
            "act_was_missing", "lct_was_missing"]
    d = pd.read_parquet(OUT_DIR / "compustat_annual_clean.parquet", columns=cols)
    d = d.dropna(subset=["fyear"])
    d["fyear"] = d["fyear"].astype(int)
    d["ca"] = d["act"].where(~d["act_was_missing"].astype(bool))
    d["cl"] = d["lct"].where(~d["lct_was_missing"].astype(bool))
    d["interest"] = d["xint"]
    d["debt"] = d[["dltt", "dlc"]].sum(axis=1, min_count=1)
    # No 财务费用 analogue exists: xint IS interest expense. The robustness
    # columns are left NA so the US row prints n/a rather than a fabricated
    # counterpart.
    d["finexp"] = np.nan
    d["ebit_fx"] = np.nan
    keep = ["ni", "oancf", "lt", "at", "ebit", "ebitda", "interest",
            "ca", "cl", "debt", "finexp", "ebit_fx"]
    return d.groupby(["gvkey", "fyear"])[keep].median()


def gate(us, us_items):
    """TWO HARD ASSERTIONS. Nothing downstream is trustworthy without them.

    1. The US scores must reproduce ~0.87 against the panel's OWN default
       label. That number holds BY CONSTRUCTION — it is what stage4_1 measured
       — so anything materially below it means the data handling is broken,
       not that the models are weak.
    2. Same-year loss must clear a loose floor. Correct alignment returns ~0.85;
       a misalignment returns ~0.50. The floor sits between, at 0.70.

    Both are assertions rather than printed lines because a printed line can be
    glanced past, and the failure they catch LOOKS LIKE A RESULT.
    """
    print("\n" + "=" * 100)
    print("GATE — the US benchmark must reproduce what is already known")
    print("=" * 100)
    for m in MODELS:
        a = roc_auc_score(us.y.astype(int), us[f"score_{m}"])
        print(f"      score_{m:<6} vs the panel's own default label y : "
              f"{a:>7.4f}   (floor {US_DEFAULT_FLOOR})")
        if a < US_DEFAULT_FLOOR:
            raise SystemExit(
                f"STOP: score_{m} reaches only {a:.4f} against the US default "
                f"label, where stage4_1 measured ~0.87. The models are not "
                "wrong; the data handling in this script is. Every number "
                "below would be noise. Do not read past this line.")

    y = outcomes({o: lookup(us, "gvkey", us_items, o) for o in (0, 1)})
    for m in MODELS:
        a, n, npos = auc(y["loss_t0"], us[f"score_{m}"])
        shown = f"{a:.4f}" if a is not None else "n/a"
        print(f"      score_{m:<6} vs SAME-YEAR loss (ni<0 at t)       : "
              f"{shown:>7}   n={n:,}  (floor {SAME_YEAR_LOSS_FLOOR})")
        if a is None or a < SAME_YEAR_LOSS_FLOOR:
            raise SystemExit(
                f"STOP: score_{m} reaches {shown} against same-year loss. "
                f"Correct alignment returns ~0.85 and a misaligned merge "
                "returns ~0.50. The outcome is not lined up with the score.")
    print("      both gates passed — the US side reproduces what is known, so "
          "the handling is sound")


def check_b(cn, cn_items, us, us_items):
    print("\n" + "=" * 100)
    print("CHECK B — can the score predict the firm's OWN future financials? "
          "(no ratings involved)")
    print("=" * 100)
    print("  Outcomes are RAW and OBSERVED, never imputed. n is counted from "
          "the outcome's own\n  non-missing values, so an alignment fault shows "
          "as a row-count discrepancy rather than\n  as a plausible AUC.")

    print("  Every t+1 outcome is printed BESIDE its same-year twin. The t "
          "column is the alignment\n  control — it is what caught the index bug "
          "that made this whole section read 0.51.")

    # (label, t+1 key, t key, description)
    SPEC = [
        ("1  MAIN  interest cover < 1x  (EBIT / 利息费用 · xint)",
         "intcov_t1", "intcov_t0"),
        ("1  ROBUST interest cover < 1x  (EBIT / 财务费用, neg. dropped)",
         "intcovfx_t1", "intcovfx_t0"),
        ("2a EBITDA turns negative", "ebitdaneg_t1", "ebitdaneg_t0"),
        ("2b Debt/EBITDA worsens > %.1f turns" % DE_JUMP,
         "debtebitda_t1", "debtebitda_t0"),
        ("3  LOSS (ni < 0)", "loss_t1", "loss_t0"),
        ("3  LOSS at t+2", "loss_t2", None),
        ("3  LOSS at t+3", "loss_t3", None),
        ("4  current ratio < 1", "currat_t1", "currat_t0"),
        ("5  two consecutive loss years", "twoloss_t1", "twoloss_t0"),
        ("-  operating cash flow negative", "ocfneg_t1", "ocfneg_t0"),
        ("-  leverage rises > %.0f%% of assets" % (LEV_JUMP * 100),
         "levup_t1", None),
    ]

    for label, panel, cols in [("CHINA", cn, SCORES),
                               ("US distress (benchmark)", us, MODELS)]:
        items = cn_items if label == "CHINA" else us_items
        y = outcomes({o: lookup(panel, "gvkey", items, o)
                      for o in (0, 1, 2, 3)})
        print(f"\n\n  {'=' * 92}\n  {label}   ({len(panel):,} scored rows)"
              f"\n  {'=' * 92}")
        for desc, k1, k0 in SPEC:
            if k1 not in y.columns or y[k1].notna().sum() == 0:
                print(f"\n      {desc}\n          not computable on this side "
                      f"(no observed values)")
                continue
            obs = y[k1].notna()
            base = y.loc[obs, k1].mean() * 100
            print(f"\n      {desc}")
            print(f"          t+1: observed {int(obs.sum()):,}/{len(panel):,} "
                  f"({obs.mean() * 100:.1f}%)   BASE RATE {base:.1f}% "
                  f"({int(y.loc[obs, k1].sum()):,} positives)")
            if k0 and k0 in y.columns and y[k0].notna().sum():
                o0 = y[k0].notna()
                print(f"          t  : observed {int(o0.sum()):,} "
                      f"({o0.mean() * 100:.1f}%)   base rate "
                      f"{y.loc[o0, k0].mean() * 100:.1f}%   "
                      f"<- alignment control")
            hdr = f"{'score':<16}{'AUC t+1':>10}{'base%':>8}"
            hdr += f"{'AUC t':>10}" if k0 else ""
            print(f"          {hdr}")
            for s in cols:
                a1, n1, p1 = auc(y[k1], panel[f"score_{s}"])
                c1 = f"{a1:>10.4f}" if a1 is not None else f"{'n/a':>10}"
                line = f"{s:<16}{c1}{base:>7.1f}%"
                if k0 and k0 in y.columns:
                    a0, _, _ = auc(y[k0], panel[f"score_{s}"])
                    line += f"{a0:>10.4f}" if a0 is not None else f"{'n/a':>10}"
                if a1 is None:
                    line += f"   (n={n1:,}, pos={p1:,} — below the "
                    line += f"{MIN_CLASS} floor)"
                print(f"          {line}")


def check_a(m):
    print("\n" + "=" * 100)
    print("CHECK A — does the relationship with ratings change over time?")
    print("=" * 100)
    print("  CUT: AA+ or above — the Chinese counterpart of the US "
          "speculative-grade split BY SHARE OF\n  THE MARKET, not by letter "
          "(~46% here against 43.8% speculative in the US sample, D-42).")
    cut = m.notch <= 3

    def row(sel, lab):
        sa, n, npos = auc(cut[sel], m.loc[sel, "log_at"])
        cells = []
        for s in SCORES:
            a, _, _ = auc(cut[sel], m.loc[sel, f"score_{s}"])
            cells.append(f"{a:>8.4f}" if a is not None else f"{'n/a':>8}")
        size = f"{sa:>8.4f}" if sa is not None else f"{'n/a':>8}"
        flag = "" if npos >= MIN_CLASS else f"   <- only {npos} positives"
        print(f"  {lab:<12}{int(sel.sum()):>7,}{npos:>7,}{size}"
              + "".join(cells) + flag)

    head = (f"  {'period':<12}{'n':>7}{'pos':>7}{'size':>8}"
            + "".join(f"{s.replace('distress','d').replace('rating','r'):>9}"
                      for s in SCORES))
    print(f"\n  BY PERIOD\n{head}")
    for lab, lo, hi in PERIODS:
        sel = m.fyear.between(lo, hi)
        if sel.sum():
            row(sel, lab)

    print(f"\n  BY YEAR\n{head}")
    for y in sorted(m.fyear.unique()):
        row(m.fyear == y, str(int(y)))
    print(f"\n  The 'size' column is the AUC of log(total assets) ALONE — a "
          f"reference, not a control.\n  If size predicts ratings less well "
          f"after 2020, that is the same finding from the other side.")


def main():
    print("=" * 100)
    print("VALIDITY OF THE TRANSFERRED CHINESE SCORES — read-only, nothing "
          "written")
    print("=" * 100)

    cn = pd.read_parquet(OUT_DIR / "stage6_4_china_score_panel.parquet")
    cn = cn[cn[f"score_{SCORES[0]}"].notna()].copy()
    us = pd.read_parquet(OUT_DIR / "stage5_1_score_panel_L3_K12.parquet")
    us = us.drop_duplicates(["gvkey", "fyear"]).copy()
    print(f"  China scored rows {len(cn):,} ({cn.gvkey.nunique():,} issuers)   "
          f"US rows {len(us):,} ({us.gvkey.nunique():,} gvkeys, de-duplicated "
          f"on gvkey x fyear)")

    bm = csmar("债券基本情况表.csv", ["Liscd", "Sctcd", "Orgid"]).drop_duplicates()
    cn_items = china_items(bm)
    us_it = us_items()
    gate(us, us_it)
    check_b(cn, cn_items, us, us_it)

    # ---- the rating panel, for CHECK A -------------------------------------
    rt = csmar("债券及主体评级情况表.csv",
               ["Liscd", "Sctcd", "RatingDate", "Ctcr"])
    have = ~rt.Ctcr.isin(["", "nan"]) & rt.Ctcr.notna()
    unknown = set(rt.loc[have, "Ctcr"].unique()) - set(NOTCH) - NOT_LONG_TERM
    if unknown:
        raise SystemExit(f"STOP: Ctcr values not on the notch scale and not "
                         f"declared excluded: {sorted(unknown)}")
    rt = rt.merge(bm, on=["Liscd", "Sctcd"], how="left")
    rt["notch"] = rt.Ctcr.map(NOTCH)
    rt = rt[rt.notch.notna() & rt.Orgid.notna()]
    rt["year"] = pd.to_datetime(rt.RatingDate, errors="coerce").dt.year
    # LOWEST rating in the year = HIGHEST notch: the harder test.
    rp = rt.groupby(["Orgid", "year"])["notch"].max().reset_index()
    if rp.duplicated(["Orgid", "year"]).any():
        raise SystemExit("STOP: rating panel not unique on (Orgid, year).")

    m = cn.merge(rp, left_on=["gvkey", "fyear"], right_on=["Orgid", "year"],
                 how="inner")
    m = m[m.fyear >= 2010].copy()
    m["log_at"] = np.log(pd.MultiIndex.from_arrays(
        [m.gvkey, m.fyear]).map(cn_items["at"]).astype("float64"))
    print(f"\n  rating-matched, 2010+, scored: {len(m):,} issuer-years, "
          f"{m.gvkey.nunique():,} issuers; size on "
          f"{m.log_at.notna().mean() * 100:.2f}%")
    check_a(m)

    print("\n" + "=" * 100)
    print("NO VERDICT IS STATED HERE. Every line above is a measured quantity.")
    print("=" * 100)


if __name__ == "__main__":
    main()
