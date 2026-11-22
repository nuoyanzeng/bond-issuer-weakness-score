"""
Stage 7.2 — Descriptive statistics for the rating-inflation comparison (D-72).

Reads   output/stage7_0_us_rating_panel.parquet     (US, gvkey x fyear)
        output/stage7_1_china_rating_panel.parquet  (China, Orgid x fyear)
        output/compustat_annual_clean.parquet       (SIC, robustness (iii) only)
        data_raw/us/comp_names.parquet              (SIC fallback, same)
Writes  output/stage7_2_inflation_descriptive.md    (and prints the same)

Rated rows only, fiscal years 2010-2024 on both sides (D-72 items 6 and 7).

NO SIZE, NO REGRESSION. Size enters at the regression step (D-72 item 14).
Every line is a measured quantity. Nothing here is interpreted.

Inference: bootstrap resampling FIRMS, not firm-years, 1,000 draws, seed 42
(D-72 item 11). A cross-market difference is printed only where each market has
at least 20 distinct firms in the cell (item 10); the counts print regardless.

Run:  python3 code/stage7_2_inflation_descriptive.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
RAW_US = PROJECT_DIR / "data_raw" / "us"

US_IN = OUT_DIR / "stage7_0_us_rating_panel.parquet"
CN_IN = OUT_DIR / "stage7_1_china_rating_panel.parquet"
ANN_IN = OUT_DIR / "compustat_annual_clean.parquet"
NAMES_IN = RAW_US / "comp_names.parquet"

YEAR_MIN, YEAR_MAX = 2010, 2024
MODELS = ["xgb", "rf", "enet"]

# WHICH US SCORE. `--scores old` is the behaviour this script has always had:
# the US column is stage7_0's `score_<model>`, which is stage4_2's
# `score_calibrated` carried through stage5_1 — out-of-fold sigmoid-calibrated
# (D-47). `--scores raw` replaces it with stage4_2's `score_raw`, the
# uncalibrated out-of-fold prediction, so that both markets' columns are raw.
# The Chinese column is raw in both runs (D-67) and is never touched.
#
# The two runs write to DIFFERENT files, so a raw run cannot overwrite the old
# one, and the five-estimate loud check applies to `old` only.
SCORES_VERSION = "old"
if "--scores" in sys.argv:
    _i = sys.argv.index("--scores")
    if _i + 1 >= len(sys.argv):
        raise SystemExit("STOP: --scores needs a value, 'old' or 'raw'.")
    SCORES_VERSION = sys.argv[_i + 1]
if SCORES_VERSION not in ("old", "raw"):
    raise SystemExit(f"STOP: --scores must be 'old' or 'raw', not "
                     f"{SCORES_VERSION!r}.")

MD_OUT = OUT_DIR / ("stage7_2_inflation_descriptive.md" if SCORES_VERSION == "old"
                    else "stage7_2_inflation_descriptive_raw.md")
RAW_SCORES = {m: OUT_DIR / f"stage4_2_scores_{m}.parquet" for m in MODELS}
US_SCORE_LABEL = ("calibrated (D-47)" if SCORES_VERSION == "old"
                  else "raw out-of-fold, stage4_2")
SPEC_FROM = 11                 # BB+ and below
MIN_FIRMS = 20                 # D-72 item 10
N_BOOT, SEED = 1000, 42        # D-72 item 11
# The two markets share no firms, so their bootstraps must be independent draws.
# One seeded stream used for both would make them move together and understate
# the width of the difference CI, so each market gets its own seed.
SEED_CN, SEED_US = SEED, SEED + 1

BANDS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC-C", "D"]
PERIODS = [("2010-14", 2010, 2014), ("2015-17", 2015, 2017),
           ("2018-20", 2018, 2020), ("2021-24", 2021, 2024)]

# The Table 2 cells, defined on the CHINESE letter (D-72 item 12). The last one
# pools everything from A+ down, because China has almost nothing there.
#
# The third element says whether a CROSS-MARKET DIFFERENCE may be printed for
# that cell. It is False for the pooled cell: "A+ and below" holds a different
# letter mix in each market — in China mostly A with a little C, in the US every
# rating from A+ down and mostly speculative grade — so a difference of means
# across it would compare two different things and read as inflation whatever
# the fundamentals did. Each market's own figures are still printed.
CELLS = [("AAA", lambda n: n == 1, True), ("AA+", lambda n: n == 2, True),
         ("AA", lambda n: n == 3, True), ("AA-", lambda n: n == 4, True),
         ("A+ and below", lambda n: n >= 5, False)]
POOLED_NOTE = "pooled cell, letter mix differs"

# Fixed score bands for Table 4, set on the score alone.
SCORE_EDGES = [round(0.1 * i, 1) for i in range(11)]

LINES = []


def w(s=""):
    """Print and collect, so the file and the console never diverge."""
    print(s)
    LINES.append(s)


# ---------------------------------------------------------------------------
# bootstrap over firms
# ---------------------------------------------------------------------------
def firm_agg(df, col, firm):
    """Per-firm sum and count of `col`, so a firm bootstrap is exact and fast:
    the mean over a resampled set of firms is sum(sums) / sum(counts)."""
    g = df.groupby(firm)[col]
    return g.sum().to_numpy(dtype=float), g.size().to_numpy(dtype=float)


def boot_draws(sums, counts, seed):
    """N_BOOT bootstrap means, resampling firms with replacement.

    The generator is rebuilt from `seed` on every call, so a cell's interval does
    not depend on how many cells were computed before it."""
    rng = np.random.default_rng(seed)
    k = len(sums)
    if k == 0:
        return np.array([])
    idx = rng.integers(0, k, size=(N_BOOT, k))
    return sums[idx].sum(1) / counts[idx].sum(1)


def mean_ci(df, col, firm, seed):
    """Mean, and the bootstrap 2.5/97.5 percentiles of the firm-resampled mean."""
    if len(df) == 0:
        return np.nan, np.nan, np.nan, np.array([])
    s, c = firm_agg(df, col, firm)
    d = boot_draws(s, c, seed)
    return df[col].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), d


def diff_ci(dc, du):
    """China-minus-US difference in means, with a bootstrap CI.

    The two draw vectors come from DIFFERENT seeded streams (SEED_CN, SEED_US),
    so pairing them element by element pairs two independent resamples. The two
    markets share no firms, so there is nothing to pair on and independence is
    the right assumption."""
    if len(dc) == 0 or len(du) == 0:
        return np.nan, np.nan
    d = dc - du
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def fmt(x, nd=4):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# the raw-score swap, used only by --scores raw
# ---------------------------------------------------------------------------
def swap_in_raw_scores(us):
    """Replace each `s_<model>` with stage4_2's out-of-fold RAW score.

    The stage4_2 files are ISSUER level — one row per (issuer_id, gvkey, fyear) —
    while this panel is gvkey x fyear, so the join is many-to-one. The raw score
    is a property of the statement, so it must be constant inside a group; that
    is asserted with an exact-zero spread rather than assumed, which is the same
    guard stage7_0 applies to the calibrated score before collapsing it.

    The calibrated column is kept as `old_<model>` so the rank-based comparison
    at the end can report both.
    """
    print("\n" + "=" * 78)
    print("--scores raw — replacing the US calibrated scores with stage4_2's raw")
    print("=" * 78)
    n0 = len(us)
    for m in MODELS:
        p = RAW_SCORES[m]
        if not p.exists():
            raise SystemExit(f"STOP: {p} not found.")
        s = pd.read_parquet(p, columns=["gvkey", "fyear", "score_raw"])
        g = s.groupby(["gvkey", "fyear"])["score_raw"]
        spread = float((g.max() - g.min()).max())
        if spread != 0:
            raise SystemExit(f"STOP: {p.name}'s score_raw varies inside a "
                             f"gvkey x fyear group, max spread {spread:.3e}. "
                             "Collapsing it would pick one of several values.")
        coll = g.first().rename(f"_raw_{m}").reset_index()
        us[f"old_{m}"] = us[f"s_{m}"]
        us = us.merge(coll, on=["gvkey", "fyear"], how="left")
        if len(us) != n0:
            raise SystemExit(f"STOP: the {m} raw-score join changed the row "
                             f"count, {n0:,} -> {len(us):,}.")
        n_miss = int(us[f"_raw_{m}"].isna().sum())
        if n_miss:
            raise SystemExit(f"STOP: {n_miss:,} of {n0:,} panel rows have no "
                             f"raw {m} score on gvkey x fyear.")
        us[f"s_{m}"] = us[f"_raw_{m}"]
        us = us.drop(columns=[f"_raw_{m}"])
        print(f"  {p.name:<34} {len(s):,} issuer-level rows -> "
              f"{len(coll):,} gvkey x fyear; within-group spread {spread:.3e}; "
              f"matched all {n0:,} panel rows")
    return us


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load():
    us = pd.read_parquet(US_IN)
    cn = pd.read_parquet(CN_IN)
    print("=" * 78)
    print("LOAD")
    print("=" * 78)
    print(f"  US panel  : {len(us):,} rows")
    print(f"  CN panel  : {len(cn):,} rows")

    us_rated = us.notch.notna()
    cn_rated = cn.notch.notna()
    print(f"  US unrated rows dropped : {int((~us_rated).sum()):,}")
    print(f"  CN unrated rows dropped : {int((~cn_rated).sum()):,}")
    us, cn = us[us_rated].copy(), cn[cn_rated].copy()

    in_us = us.fyear.between(YEAR_MIN, YEAR_MAX)
    in_cn = cn.fyear.between(YEAR_MIN, YEAR_MAX)
    print(f"  US outside {YEAR_MIN}-{YEAR_MAX} dropped : {int((~in_us).sum()):,}")
    print(f"  CN outside {YEAR_MIN}-{YEAR_MAX} dropped : {int((~in_cn).sum()):,}")
    us, cn = us[in_us].copy(), cn[in_cn].copy()

    # One name per model on each side, so every table below is symmetric.
    us = us.rename(columns={f"score_{m}": f"s_{m}" for m in MODELS})
    cn = cn.rename(columns={f"score_rating_{m}": f"s_{m}" for m in MODELS})
    if SCORES_VERSION == "raw":
        us = swap_in_raw_scores(us)
    us["firm"], cn["firm"] = us.gvkey, cn.Orgid
    us["notch"], cn["notch"] = us.notch.astype(int), cn.notch.astype(int)
    for d, lab in ((us, "US"), (cn, "China")):
        print(f"  {lab:<6} kept: {len(d):,} rows, {d.firm.nunique():,} firms, "
              f"fyear {int(d.fyear.min())}-{int(d.fyear.max())}")
        if d[[f"s_{m}" for m in MODELS]].isna().any().any():
            raise SystemExit(f"STOP: {lab} has a missing score on a rated row.")
    return us, cn


def spec_share(d):
    return (d.notch >= SPEC_FROM).mean()


# ---------------------------------------------------------------------------
# TABLE 1
# ---------------------------------------------------------------------------
def table1(us, cn):
    w("\n## Table 1 — rating distribution, 2010-2024, rated rows\n")
    w("Firm-years, distinct firms, and share of that market's rated rows.\n")
    w("| letter | notch | US rows | US firms | US % | CN rows | CN firms | CN % |")
    w("|---|---|---|---|---|---|---|---|")
    letters = sorted(set(us.notch) | set(cn.notch))
    t1 = []
    for n in letters:
        a, b = us[us.notch == n], cn[cn.notch == n]
        lab = (a.letter.iloc[0] if len(a) else b.letter.iloc[0])
        w(f"| `{lab}` | {n} | {len(a):,} | {a.firm.nunique():,} | "
          f"{len(a)/len(us)*100:.2f} | {len(b):,} | {b.firm.nunique():,} | "
          f"{len(b)/len(cn)*100:.2f} |")
        # Kept so Figure 1 plots these shares rather than recomputing them.
        t1.append({"notch": n, "letter": lab, "US %": len(a) / len(us) * 100,
                   "China %": len(b) / len(cn) * 100})
    FIGDATA["t1"] = t1

    w("\n| band | US rows | US firms | US % | CN rows | CN firms | CN % |")
    w("|---|---|---|---|---|---|---|")
    for bd in BANDS:
        a, b = us[us.band == bd], cn[cn.band == bd]
        w(f"| {bd} | {len(a):,} | {a.firm.nunique():,} | {len(a)/len(us)*100:.2f} "
          f"| {len(b):,} | {b.firm.nunique():,} | {len(b)/len(cn)*100:.2f} |")

    w(f"\n**Speculative grade (notch >= {SPEC_FROM}).** "
      f"US {spec_share(us)*100:.2f}% of {len(us):,} rows; "
      f"China {spec_share(cn)*100:.2f}% of {len(cn):,} rows.\n")

    w("**Band shares by period, %.**\n")
    w("| period | market | rows | " + " | ".join(BANDS) + " |")
    w("|---|---|---|" + "---|" * len(BANDS))
    for lab, lo, hi in PERIODS:
        for mk, d in (("US", us), ("China", cn)):
            p = d[d.fyear.between(lo, hi)]
            if len(p) == 0:
                w(f"| {lab} | {mk} | 0 | " + " | ".join(["—"] * len(BANDS)) + " |")
                continue
            sh = [f"{(p.band == bd).mean()*100:.1f}" for bd in BANDS]
            w(f"| {lab} | {mk} | {len(p):,} | " + " | ".join(sh) + " |")


# ---------------------------------------------------------------------------
# TABLE 2
# ---------------------------------------------------------------------------
def cell_rows(d, test):
    return d[[test(n) for n in d.notch]]


def table2(us, cn):
    w("\n## Table 2 — HEADLINE: mean rating score at the same rating letter\n")
    # Same split as the ORIENTATION paragraph: the share reading holds only
    # where the US column is calibrated.
    reading = ("A group's mean score is the estimated share of that group whose "
               "fundamentals warrant speculative grade."
               if SCORES_VERSION == "old" else
               "Scores are uncalibrated: means are compared across markets, "
               "not read as probabilities.")
    w("The cells are defined on the rating LETTER and each market's own rows at "
      "that letter are summarised. " + reading + " CI is the "
      f"bootstrap 95% interval over {N_BOOT:,} draws resampling FIRMS. A "
      f"difference is printed only where each market has at least {MIN_FIRMS} "
      "distinct firms in the cell. **No difference is printed for the pooled "
      "`A+ and below` cell**: it holds a different letter mix in each market — in "
      "China mostly A with a little C, in the US every rating from A+ down and "
      "mostly speculative grade — so a difference across it would compare two "
      "different things. Each market's own figures are still given.\n")
    for m in MODELS:
        col = f"s_{m}"
        w(f"\n**{m}**\n")
        w("| letter | US rows | US firms | US mean | US 95% CI | US p25 | US med "
          "| US p75 | CN rows | CN firms | CN mean | CN 95% CI | CN p25 | CN med "
          "| CN p75 | CN - US | diff 95% CI |")
        w("|---|" + "---|" * 16)
        for lab, test, diffable in CELLS:
            a, b = cell_rows(us, test), cell_rows(cn, test)
            am, alo, ahi, ad = mean_ci(a, col, "firm", SEED_US)
            bm, blo, bhi, bd = mean_ci(b, col, "firm", SEED_CN)
            nfa, nfb = a.firm.nunique(), b.firm.nunique()
            if not diffable:
                dtxt = ctxt = POOLED_NOTE
            elif nfa >= MIN_FIRMS and nfb >= MIN_FIRMS:
                dlo, dhi = diff_ci(bd, ad)
                dtxt, ctxt = fmt(bm - am), f"[{fmt(dlo)}, {fmt(dhi)}]"
            else:
                dtxt = ctxt = f"below min cell ({nfa}, {nfb})"
            w(f"| `{lab}` | {len(a):,} | {nfa:,} | {fmt(am)} | "
              f"[{fmt(alo)}, {fmt(ahi)}] | {fmt(a[col].quantile(.25)) if len(a) else '—'} "
              f"| {fmt(a[col].median()) if len(a) else '—'} | "
              f"{fmt(a[col].quantile(.75)) if len(a) else '—'} | {len(b):,} | "
              f"{nfb:,} | {fmt(bm)} | [{fmt(blo)}, {fmt(bhi)}] | "
              f"{fmt(b[col].quantile(.25)) if len(b) else '—'} | "
              f"{fmt(b[col].median()) if len(b) else '—'} | "
              f"{fmt(b[col].quantile(.75)) if len(b) else '—'} | {dtxt} | {ctxt} |")

    w("\n**All Chinese rated rows together, beside the realised Chinese "
      "speculative-grade share.**\n")
    w("| model | rows | firms | mean score | 95% CI | CN speculative share |")
    w("|---|---|---|---|---|---|")
    for m in MODELS:
        mm, lo, hi, _ = mean_ci(cn, f"s_{m}", "firm", SEED_CN)
        w(f"| {m} | {len(cn):,} | {cn.firm.nunique():,} | {fmt(mm)} | "
          f"[{fmt(lo)}, {fmt(hi)}] | {spec_share(cn)*100:.2f}% |")


def table2b(cn):
    w("\n## Table 2b — Table 2's China column by period, xgb only\n")
    w("| period | letter | rows | firms | mean score | 95% CI |")
    w("|---|---|---|---|---|---|")
    for lab, lo, hi in PERIODS:
        p = cn[cn.fyear.between(lo, hi)]
        for clab, test, _diffable in CELLS:
            b = cell_rows(p, test)
            if len(b) == 0:
                w(f"| {lab} | `{clab}` | 0 | 0 | — | — |")
                continue
            mm, blo, bhi, _ = mean_ci(b, "s_xgb", "firm", SEED_CN)
            w(f"| {lab} | `{clab}` | {len(b):,} | {b.firm.nunique():,} | "
              f"{fmt(mm)} | [{fmt(blo)}, {fmt(bhi)}] |")


# ---------------------------------------------------------------------------
# TABLE 3
# ---------------------------------------------------------------------------
def table3(us, cn):
    w("\n## Table 3 — share-matched cutoff, set on the US sample\n")
    share = spec_share(us)
    w(f"The cutoff `c` is the US score at which the share of US rows scoring at "
      f"or above `c` equals the US speculative-grade share, "
      f"{share*100:.2f}%. It is then applied unchanged to both markets. The "
      f"pooled `A+ and below` cell carries no difference, for the reason given "
      f"under Table 2.\n")
    cuts = {}
    w("| model | cutoff c | US share >= c (check) |")
    w("|---|---|---|")
    for m in MODELS:
        c = us[f"s_{m}"].quantile(1 - share)
        cuts[m] = c
        w(f"| {m} | {fmt(c)} | {(us[f's_{m}'] >= c).mean()*100:.2f}% |")

    for m in MODELS:
        c = cuts[m]
        col = f"s_{m}"
        w(f"\n**{m}, cutoff {fmt(c)} — share of rows with score >= c**\n")
        w("| letter | US rows | US share | US 95% CI | CN rows | CN share "
          "| CN 95% CI | CN - US |")
        w("|---|---|---|---|---|---|---|---|")
        for lab, test, diffable in CELLS:
            a, b = cell_rows(us, test).copy(), cell_rows(cn, test).copy()
            a["hit"], b["hit"] = (a[col] >= c).astype(float), (b[col] >= c).astype(float)
            am, alo, ahi, ad = mean_ci(a, "hit", "firm", SEED_US)
            bm, blo, bhi, bd = mean_ci(b, "hit", "firm", SEED_CN)
            nfa, nfb = a.firm.nunique(), b.firm.nunique()
            if not diffable:
                dtxt = POOLED_NOTE
            elif nfa >= MIN_FIRMS and nfb >= MIN_FIRMS:
                dtxt = fmt((bm - am) * 100, 2) + "pp"
            else:
                dtxt = f"below min cell ({nfa}, {nfb})"
            w(f"| `{lab}` | {len(a):,} | {fmt(am*100, 2)}% | "
              f"[{fmt(alo*100, 2)}, {fmt(ahi*100, 2)}] | {len(b):,} | "
              f"{fmt(bm*100, 2)}% | [{fmt(blo*100, 2)}, {fmt(bhi*100, 2)}] "
              f"| {dtxt} |")


# ---------------------------------------------------------------------------
# TABLE 4
# ---------------------------------------------------------------------------
def table4(us, cn):
    w("\n## Table 4 — reverse cut: actual ratings within fixed score bands, xgb\n")
    w("The bands are fixed on the score alone. Within each, the row gives that "
      "market's distribution over actual rating bands, in %.\n")
    w("| score band | market | rows | firms | " + " | ".join(BANDS) + " |")
    w("|---|---|---|---|" + "---|" * len(BANDS))
    for i in range(len(SCORE_EDGES) - 1):
        lo, hi = SCORE_EDGES[i], SCORE_EDGES[i + 1]
        for mk, d in (("US", us), ("China", cn)):
            # The last band is closed on the right so a score of exactly 1 counts.
            sel = ((d.s_xgb >= lo) & (d.s_xgb < hi) if i < len(SCORE_EDGES) - 2
                   else (d.s_xgb >= lo) & (d.s_xgb <= hi))
            p = d[sel]
            if len(p) == 0:
                w(f"| {lo:.1f}-{hi:.1f} | {mk} | 0 | 0 | "
                  + " | ".join(["—"] * len(BANDS)) + " |")
                continue
            sh = [f"{(p.band == bd).mean()*100:.1f}" for bd in BANDS]
            w(f"| {lo:.1f}-{hi:.1f} | {mk} | {len(p):,} | {p.firm.nunique():,} | "
              + " | ".join(sh) + " |")


# ---------------------------------------------------------------------------
# TABLE 5
# ---------------------------------------------------------------------------
def table5(us, cn):
    w("\n## Table 5 — ordering: Spearman correlation of score with notch\n")
    w("A positive value means a worse rating goes with a higher score, the two "
      "agreeing. Computed on firm-years; no firm clustering is applied to a "
      "correlation.\n")
    w("| market | model | rows | all rated rows | rows AAA..AA- | AAA..AA- only |")
    w("|---|---|---|---|---|---|")
    for mk, d in (("US", us), ("China", cn)):
        top = d[d.notch <= 4]
        for m in MODELS:
            r_all = spearmanr(d[f"s_{m}"], d.notch).statistic
            if top.notch.nunique() < 2 or len(top) < 3:
                r_top = np.nan
            else:
                r_top = spearmanr(top[f"s_{m}"], top.notch).statistic
            w(f"| {mk} | {m} | {len(d):,} | {fmt(r_all)} | {len(top):,} | "
              f"{fmt(r_top)} |")


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------
def table2_xgb_only(us, cn, title, note):
    w(f"\n### {title}\n")
    w(note + "\n")
    w("| letter | US rows | US firms | US mean | CN rows | CN firms | CN mean "
      "| CN - US | diff 95% CI |")
    w("|---|---|---|---|---|---|---|---|---|")
    for lab, test, diffable in CELLS:
        a, b = cell_rows(us, test), cell_rows(cn, test)
        am, _, _, ad = mean_ci(a, "s_xgb", "firm", SEED_US)
        bm, _, _, bd = mean_ci(b, "s_xgb", "firm", SEED_CN)
        nfa, nfb = a.firm.nunique(), b.firm.nunique()
        if not diffable:
            dtxt = ctxt = POOLED_NOTE
        elif nfa >= MIN_FIRMS and nfb >= MIN_FIRMS:
            dlo, dhi = diff_ci(bd, ad)
            dtxt, ctxt = fmt(bm - am), f"[{fmt(dlo)}, {fmt(dhi)}]"
        else:
            dtxt = ctxt = f"below min cell ({nfa}, {nfb})"
        w(f"| `{lab}` | {len(a):,} | {nfa:,} | {fmt(am)} | {len(b):,} | {nfb:,} "
          f"| {fmt(bm)} | {dtxt} | {ctxt} |")


def us_sic(us):
    """SIC per gvkey x datadate: `sich` first, comp_names.sic as fallback (D-10)."""
    ann = pd.read_parquet(ANN_IN, columns=["gvkey", "datadate", "sich"])
    names = pd.read_parquet(NAMES_IN, columns=["gvkey", "sic"])
    names["sic"] = names["sic"].replace("", pd.NA)
    d = us.merge(ann, on=["gvkey", "datadate"], how="left").merge(
        names, on="gvkey", how="left")
    if len(d) != len(us):
        raise SystemExit(f"STOP: the SIC merge changed the row count, "
                         f"{len(us):,} -> {len(d):,}.")
    code = pd.to_numeric(d.sich, errors="coerce")
    code = code.where(code.notna(), pd.to_numeric(d.sic, errors="coerce"))
    print(f"  US rows with no SIC from either source: {int(code.isna().sum()):,}")
    return code


def robustness(us, cn):
    w("\n## Robustness — Table 2, xgb only\n")

    # (i) same-year Chinese ratings only
    if "rating_source" in cn.columns:
        sy = cn[cn.rating_source == "same_year"]
        table2_xgb_only(us, sy, "(i) China: same-year ratings only",
                        f"Carried-forward rows removed: {len(cn) - len(sy):,} of "
                        f"{len(cn):,}. US side unchanged.")
    else:
        w("\n### (i) SKIPPED — the China panel has no `rating_source` column.\n")

    # (ii) LGFVs — no flag exists anywhere on disk (D-72)
    w("\n### (ii) LGFVs excluded — SKIPPED, NO FLAG EXISTS\n")
    w("Searched every candidate column in `output/` and `data_raw/china/`: "
      "`IssuerNature` is a financial classifier (一般企业 / 商业银行 / 证券公司 …), "
      "`Varsort` a coupon type, `BondNature` a placement type, `IssuerIndustry` "
      "the 20 门类 with no 城投 category, and `Bndtype` an undecoded numeric code "
      "with no codebook on disk. Inferring LGFV status from 建筑业 or 综合 would "
      "manufacture a variable with no referent, so this table is "
      "not produced. D-72 item 8 records it.\n")

    # (iii) financial firms excluded on both sides
    code = us_sic(us)
    fin_us = code.between(6000, 6999).to_numpy()
    fin_v4 = code.between(6000, 6799).to_numpy()      # v4's own "H finance" range
    fin_cn = (cn.ind_cn == "金融业").to_numpy()
    print(f"  US financial rows, SIC 6000-6999 : {int(fin_us.sum()):,}")
    print(f"  US financial rows, SIC 6000-6799 : {int(fin_v4.sum()):,} "
          "(v4's own division bound, printed for comparison)")
    print(f"  CN financial rows, ind_cn == 金融业: {int(fin_cn.sum()):,}")
    table2_xgb_only(
        us[~fin_us], cn[~fin_cn], "(iii) financial firms excluded, both sides",
        f"US: SIC 6000-6999 from `sich` with `comp_names.sic` as fallback (D-10), "
        f"**{int(fin_us.sum()):,} of {len(us):,} rows removed** "
        f"({int(fin_v4.sum()):,} under v4's narrower 6000-6799 bound). "
        f"China: `ind_cn == 金融业`, **{int(fin_cn.sum()):,} of {len(cn):,} rows "
        f"removed**. The two exclusions are not symmetric in effect: the Chinese "
        f"panel holds almost no large financial institutions (D-66).")


# ===========================================================================
# TABLES 6, 7 and 8 — folded in from the one-off checks of 2026-09-30 (D-73).
#
# The four estimators below are the ones those runs used, kept verbatim so the
# new tables reproduce them exactly. `loud_check()` at the end asserts five of
# the point estimates against the figures those runs printed, and refuses to
# write the .md if any differs. Nothing above this line is touched.
# ===========================================================================

# Nested, cumulative from the top of the scale.
TIERS = [("AAA", 1), ("AA+ and above", 2), ("AA and above", 3),
         ("AA- and above", 4), ("A- and above", 7), ("BBB- and above", 10)]

# S&P Global Ratings Definitions: "+" and "-" show relative standing WITHIN the
# rating categories, so removing them leaves the category itself.
NOTCH_TO_CAT = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4, 10: 4,
                11: 5, 12: 5, 13: 5, 14: 6, 15: 6, 16: 6, 17: 7, 18: 7, 19: 7,
                20: 8, 21: 9, 22: 10}
CAT_LETTER = [None, "AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
NOTCH_LETTER = [None, "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB",
                "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC",
                "CCC-", "CC", "C", "D"]

# Table 7's three scales: (label, column, name, decimals of the overall gap,
# whether it is a share, whether the by-letter split is printed).
SCALES = [("a", "notch", "notch", 3, False, True),
          ("b", "cat", "S&P rating category", 4, False, True),
          ("c", "ig", "investment grade", 4, True, False)]

DEC_GROUPS = [("strongest 10%", [1]), ("strongest 20%", [1, 2]),
              ("strongest 30%", [1, 2, 3]), ("weakest 10%", [10]),
              ("weakest 20%", [9, 10]), ("weakest 30%", [8, 9, 10])]

# Point estimates the one-off runs printed, and the intervals beside them.
# (value, (lo, hi), decimals, what it is)
EXPECT = {
    "t6":  (+0.3202, (+0.1571, +0.4364), 4,
            "Table 6, xgb, AA+ and above, China-minus-US"),
    "t7a": (-9.216,  (-9.346, -9.074),   3, "Table 7(a), xgb, overall gap"),
    "t7b": (-3.0652, (-3.1116, -3.0141), 4, "Table 7(b), xgb, overall gap"),
    "t7c": (+0.5573, (+0.5375, +0.5773), 4,
            "Table 7(c), xgb, overall IG-share gap"),
    "t8":  (-13.320, (-13.609, -13.036), 3,
            "Table 8, xgb, weakest 10%, China-minus-US mean notch"),
}
CHECKS = {}
# Series the tables compute as they run, kept so the figures plot exactly those
# numbers instead of a second derivation of them. Only `--scores raw` reads it.
FIGDATA = {}

# --scores raw only: the three figures. All drawing and every style choice
# lives in `code/inflation_figures.py`; this script computes numbers and hands
# them over. The stems carry no extension — that module adds .png and .pdf.
FIG_DIR = OUT_DIR / "figures"
FIG_NAMES = {1: "fig1_rating_distribution", 2: "fig2_rating_by_fundamentals",
             3: "fig3_rating_by_score_band"}
# The rf versions of Figures 2 and 3, drawn after the xgb ones and under their
# own names, so the xgb files are never replaced. Figure 1 uses no score, so it
# has no rf version. The two totals are the rated 2010-2024 samples that the
# rf groups of Figure 2 must add up to.
FIG_NAMES_RF = {2: "fig2_rating_by_fundamentals_rf",
                3: "fig3_rating_by_score_band_rf"}
FIG_RF_TOTAL_US, FIG_RF_TOTAL_CN = 18276, 23085
# Figure 2's three groups, cut on the xgb rating score and again, for the rf
# version, on the rf rating score. The second element is
# the label the figure prints and is passed through verbatim; the first is a
# short key used only by the printed table here.
FIG_GROUPS = [("strong", "Strong fundamentals (score < 0.3)", lambda s: s < 0.3),
              ("middle", "Middle (0.3 to 0.7)",
               lambda s: (s >= 0.3) & (s < 0.7)),
              ("weak", "Weak fundamentals (0.7 and above)", lambda s: s >= 0.7)]


def band8(n):
    """Notch -> one of BANDS. Must agree with the panels' own `band` column."""
    if n == 1:
        return "AAA"
    if n <= 4:
        return "AA"
    if n <= 7:
        return "A"
    if n <= 10:
        return "BBB"
    if n <= 13:
        return "BB"
    if n <= 16:
        return "B"
    if n <= 21:
        return "CCC-C"
    return "D"


def band_of(s):
    """Score -> 0..9. The last band is closed on the right."""
    return np.clip(np.floor(np.asarray(s, float) * 10), 0, 9).astype(int)


def boot_w(df, col, seed, wcol=None):
    """Table 6's estimator. wcol=None means equal weights."""
    ws = np.ones(len(df)) if wcol is None else df[wcol].to_numpy(float)
    v = df[col].to_numpy(float) * ws
    tmp = pd.DataFrame({"firm": df.firm.to_numpy(), "v": v, "w": ws})
    g = tmp.groupby("firm")[["v", "w"]].sum()
    sw, ww = g.v.to_numpy(float), g.w.to_numpy(float)
    k = len(sw)
    idx = np.random.default_rng(seed).integers(0, k, (N_BOOT, k))
    return v.sum() / ws.sum(), sw[idx].sum(1) / ww[idx].sum(1)


def firm_boot(df, col, seed):
    """Tables 7 and 8's estimator."""
    t = pd.DataFrame({"f": df.firm.to_numpy(), "v": df[col].to_numpy(float),
                      "n": 1.0})
    g = t.groupby("f")[["v", "n"]].sum()
    sv, sn = g.v.to_numpy(), g.n.to_numpy()
    k = len(sv)
    idx = np.random.default_rng(seed).integers(0, k, (N_BOOT, k))
    return df[col].mean(), sv[idx].sum(1) / sn[idx].sum(1)


def pct_ci(d):
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def mult(firms, seed):
    """Sorted firm list, and a (N_BOOT x n_firms) matrix of resample counts."""
    u = np.unique(firms)
    k = len(u)
    idx = np.random.default_rng(seed).integers(0, k, (N_BOOT, k))
    M = np.empty((N_BOOT, k))
    for d in range(N_BOOT):
        M[d] = np.bincount(idx[d], minlength=k)
    return u, M


def by_firm_band(df, firms, val, nb=10):
    """Per-firm-by-band sum of `val` and row count, on a FIXED firm list, so one
    resample matrix serves any subgroup of the same market."""
    pos = pd.Series(np.arange(len(firms)), index=firms)
    r = pos.reindex(df.firm.to_numpy()).to_numpy()
    S = np.zeros((len(firms), nb))
    C = np.zeros((len(firms), nb))
    np.add.at(S, (r, df.b.to_numpy()), df[val].to_numpy(float))
    np.add.at(C, (r, df.b.to_numpy()), 1.0)
    return S, C


def add_cols(us, cn):
    """The three derived columns Tables 6-8 need. `band8` is asserted against
    the panels' own `band` column, so a drift in either mapping stops the run."""
    for d, lab in ((us, "US"), (cn, "China")):
        d["cat"] = d.notch.map(NOTCH_TO_CAT).astype(float)
        d["ig"] = (d.notch <= SPEC_FROM - 1).astype(float)
        if d.cat.isna().any():
            raise SystemExit(f"STOP: {lab} has a notch outside NOTCH_TO_CAT.")
        mine = pd.Series([band8(n) for n in d.notch], index=d.index)
        if not mine.equals(d.band.astype(object)):
            raise SystemExit(f"STOP: {lab}'s band8 disagrees with the panel's "
                             "own `band` column. NOTHING WRITTEN.")
        # One row per firm per year, so ranking rows ranks firms (Table 8).
        if d.duplicated(["firm", "fyear"]).any():
            raise SystemExit(f"STOP: {lab} is not unique on firm x fyear.")
    print("  add_cols: cat, ig and band8 built; band8 agrees with `band`; "
          "both panels unique on firm x fyear")


# ---------------------------------------------------------------------------
# TABLE 6
# ---------------------------------------------------------------------------
def table6(us, cn):
    w("\n## Table 6 — same rating tier, compare scores\n")
    w("Tiers are **nested and cumulative from the top of the scale**, so each "
      "one contains the one above it. Within a tier both markets' own rows are "
      "summarised and compared. CI is the bootstrap 95% interval over "
      f"{N_BOOT:,} draws resampling FIRMS, seeds {SEED_CN} (China) / {SEED_US} "
      f"(US). A difference needs at least {MIN_FIRMS} distinct firms in each "
      "market.\n")
    w("**Mix-adjusted** reweights the US rows so the US letter shares inside "
      "the tier equal China's. **The weights are fixed from the observed mix "
      "and are NOT redrawn inside the bootstrap**, so the adjusted intervals "
      "reflect sampling in the scores and in the firms but not uncertainty in "
      "the mix itself. Where the weights concentrate on a few letters the "
      "adjusted mean is computed from those rows alone, and its interval "
      "widens accordingly.\n")
    for tlab, cut in TIERS:
        a = us[us.notch <= cut].reset_index(drop=True).copy()
        b = cn[cn.notch <= cut].reset_index(drop=True).copy()
        nfa, nfb = a.firm.nunique(), b.firm.nunique()
        w(f"\n**{tlab}** (notch <= {cut}) — US {len(a):,} rows / {nfa:,} firms; "
          f"China {len(b):,} rows / {nfb:,} firms\n")
        w("| letter | US rows | US % | CN rows | CN % |")
        w("|---|---|---|---|---|")
        for n in range(1, cut + 1):
            ra, rb = int((a.notch == n).sum()), int((b.notch == n).sum())
            w(f"| `{NOTCH_LETTER[n]}` | {ra:,} | {ra/len(a)*100:.2f} | {rb:,} "
              f"| {rb/len(b)*100:.2f} |")

        missing = [NOTCH_LETTER[n] for n in range(1, cut + 1)
                   if (b.notch == n).any() and not (a.notch == n).any()]
        wmap = None
        if missing:
            w(f"\n**Mix adjustment skipped**: letters present in China's mix "
              f"with no US rows in this tier: {missing}.")
        else:
            wmap = {n: ((b.notch == n).mean() / (a.notch == n).mean())
                    for n in range(1, cut + 1) if (a.notch == n).any()}
            a["w"] = a.notch.map(wmap).fillna(0.0)
            w("\nmix weights: " + ", ".join(f"`{NOTCH_LETTER[n]}` x{v:.3f}"
                                            for n, v in sorted(wmap.items())))

        ok = nfa >= MIN_FIRMS and nfb >= MIN_FIRMS
        w("\n| model | US mean | US 95% CI | CN mean | CN 95% CI | CN - US | "
          "diff 95% CI | adj US mean | CN - adjUS | adj diff 95% CI |")
        w("|---|" + "---|" * 9)
        for m in MODELS:
            am, ad = boot_w(a, f"s_{m}", SEED_US)
            bm, bd = boot_w(b, f"s_{m}", SEED_CN)
            alo, ahi = pct_ci(ad)
            blo, bhi = pct_ci(bd)
            if ok:
                dlo, dhi = pct_ci(bd - ad)
                dtxt, dci = f"{bm-am:+.4f}", f"[{dlo:+.4f}, {dhi:+.4f}]"
                if m == "xgb" and tlab == "AA+ and above":
                    CHECKS["t6"] = (bm - am, dlo, dhi)
            else:
                dtxt = dci = f"below min cell ({nfa}, {nfb})"
            if wmap is None:
                wm = wtxt = wci = "—"
            else:
                wmn, wdr = boot_w(a, f"s_{m}", SEED_US, wcol="w")
                wm = f"{wmn:.4f}"
                if ok:
                    wlo, whi = pct_ci(bd - wdr)
                    wtxt, wci = f"{bm-wmn:+.4f}", f"[{wlo:+.4f}, {whi:+.4f}]"
                else:
                    wtxt = wci = f"below min cell ({nfa}, {nfb})"
            w(f"| {m} | {am:.4f} | [{alo:.4f}, {ahi:.4f}] | {bm:.4f} | "
              f"[{blo:.4f}, {bhi:.4f}] | {dtxt} | {dci} | {wm} | {wtxt} "
              f"| {wci} |")


# ---------------------------------------------------------------------------
# TABLE 7
# ---------------------------------------------------------------------------
def gap_machinery(a, b, val):
    """The China rows kept, a draws() closure, and the valid band list.

    Expected US value = the US mean of `val` in the row's own score band. The
    US band means are RECOMPUTED in every bootstrap draw, so the interval
    includes their uncertainty; the band edges are fixed."""
    valid = [i for i in range(10) if a[a.b == i].firm.nunique() >= MIN_FIRMS]
    bk = b[b.b.isin(valid)].copy()
    uf, Mus = mult(a.firm.to_numpy(), SEED_US)
    cf, Mcn = mult(bk.firm.to_numpy(), SEED_CN)
    US_S, US_C = by_firm_band(a, uf, val)
    with np.errstate(invalid="ignore", divide="ignore"):
        E = (Mus @ US_S) / (Mus @ US_C)
    e_pt = np.divide(US_S.sum(0), US_C.sum(0), out=np.full(10, np.nan),
                     where=US_C.sum(0) > 0)
    bk["exp_us"] = e_pt[bk.b.to_numpy()]
    bk["gap"] = bk[val] - bk.exp_us

    def draws(g):
        CS, CC = by_firm_band(g, cf, val)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((Mcn @ CS.sum(1)) - ((Mcn @ CC) * E).sum(1)) \
                / (Mcn @ CC.sum(1))
    return bk, draws, valid


def table7(us, cn):
    w("\n## Table 7 — same score band, compare ratings\n")
    w("Score bands are fixed tenths, the last closed on the right. Three "
      "scales: **(a) notch** (AAA = 1 … C = 21, D = 22); **(b) S&P rating "
      "category**, modifiers removed — S&P Global Ratings Definitions state "
      'that "+" and "-" show relative standing WITHIN the rating categories, '
      "so removing them leaves the category (AAA = 1, AA = 2, A = 3, BBB = 4, "
      "BB = 5, B = 6, CCC = 7, CC = 8, C = 9, D = 10); **(c) investment "
      f"grade**, notch <= {SPEC_FROM - 1}, i.e. BBB- and above.\n")
    w("A higher notch or category is a WORSE rating, so a **negative** "
      "China-minus-US figure on scales (a) and (b), or a **positive** one on "
      "scale (c), means China rates the same fundamentals better.\n")
    w("**A cell in which every row takes the same value gives a zero-width "
      "interval.** That is a property of the bootstrap on a cell with no "
      "variation, not a claim of precision.\n")
    for m in MODELS:
        a = us.assign(b=band_of(us[f"s_{m}"]))
        b = cn.assign(b=band_of(cn[f"s_{m}"]))
        for tag, val, name, nd, is_share, split in SCALES:
            def fv(x):
                return f"{x*100:.2f}%" if is_share else f"{x:.3f}"

            def fc(lo, hi):
                return (f"[{lo*100:.2f}, {hi*100:.2f}]" if is_share
                        else f"[{lo:.3f}, {hi:.3f}]")
            w(f"\n**{m} — 7({tag}) {name}, by score band**\n")
            w("| band | US rows | US firms | US | US 95% CI | CN rows "
              "| CN firms | CN | CN 95% CI | CN - US | diff 95% CI |")
            w("|---|" + "---|" * 10)
            for i in range(10):
                pa, pb = a[a.b == i], b[b.b == i]
                nfa, nfb = pa.firm.nunique(), pb.firm.nunique()
                am, ad = firm_boot(pa, val, SEED_US)
                bm, bd = firm_boot(pb, val, SEED_CN)
                alo, ahi = pct_ci(ad)
                blo, bhi = pct_ci(bd)
                # Kept so Figure 3 reuses these means and intervals; the
                # bootstrap is never run a second time.
                if m == "xgb" and tag == "a":
                    FIGDATA.setdefault("t7a", []).append(
                        {"band": f"{i/10:.1f}-{(i+1)/10:.1f}",
                         "US n": len(pa), "US mean": am, "US lo": alo,
                         "US hi": ahi, "China n": len(pb), "China mean": bm,
                         "China lo": blo, "China hi": bhi})
                # The same numbers for the rf version of Figure 3, plus the firm
                # counts that decide which bands it may draw. A separate list,
                # so the xgb list and its printed table do not change.
                if m == "rf" and tag == "a":
                    FIGDATA.setdefault("t7a_rf", []).append(
                        {"band": f"{i/10:.1f}-{(i+1)/10:.1f}",
                         "US n": len(pa), "US firms": nfa, "US mean": am,
                         "US lo": alo, "US hi": ahi, "China n": len(pb),
                         "China firms": nfb, "China mean": bm,
                         "China lo": blo, "China hi": bhi})
                if nfa >= MIN_FIRMS and nfb >= MIN_FIRMS:
                    dl, dh = pct_ci(bd - ad)
                    dtxt = (f"{(bm-am)*100:+.2f}pp" if is_share
                            else f"{bm-am:+.3f}")
                    dci = fc(dl, dh)
                else:
                    dtxt = dci = f"below min cell ({nfa}, {nfb})"
                w(f"| {i/10:.1f}-{(i+1)/10:.1f} | {len(pa):,} | {nfa:,} | "
                  f"{fv(am)} | {fc(alo, ahi)} | {len(pb):,} | {nfb:,} | "
                  f"{fv(bm)} | {fc(blo, bhi)} | {dtxt} | {dci} |")

            bk, draws, valid = gap_machinery(a, b, val)
            drop = b[~b.b.isin(valid)]
            w(f"\nBands excluded, US cell below {MIN_FIRMS} firms: "
              f"{[f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10) if i not in valid] or 'none'}. "
              f"Chinese rows excluded on that rule: {len(drop):,} of {len(b):,} "
              f"({len(drop)/len(b)*100:.2f}%), {drop.firm.nunique():,} firms.\n")
            d = draws(bk)
            nan_n = int(np.isnan(d).sum())
            lo, hi = pct_ci(d[~np.isnan(d)])
            unit = "share points" if is_share else name
            w(f"**Overall mean gap ({unit}): {bk.gap.mean():+.{nd}f}**, 95% CI "
              f"[{lo:+.{nd}f}, {hi:+.{nd}f}] — {len(bk):,} rows, "
              f"{bk.firm.nunique():,} firms."
              + (f" NaN draws: {nan_n}." if nan_n else ""))
            if m == "xgb":
                CHECKS[f"t7{tag}"] = (bk.gap.mean(), lo, hi)
            if not split:
                continue
            LET = NOTCH_LETTER if val == "notch" else CAT_LETTER
            w("\n| Chinese letter | rows | firms | mean actual | mean expected "
              "US | expected as a letter | mean gap | 95% CI |")
            w("|---|---|---|---|---|---|---|---|")
            for lab, test, _diffable in CELLS:
                g = bk[[test(n) for n in bk.notch]]
                if len(g) == 0:
                    w(f"| `{lab}` | 0 | 0 | — | — | — | — | — |")
                    continue
                dd = draws(g)
                dd = dd[~np.isnan(dd)]
                l, h = pct_ci(dd)
                w(f"| `{lab}` | {len(g):,} | {g.firm.nunique():,} | "
                  f"{g[val].mean():.3f} | {g.exp_us.mean():.3f} | "
                  f"{LET[int(round(g.exp_us.mean()))]} | {g.gap.mean():+.3f} "
                  f"| [{l:+.3f}, {h:+.3f}] |")


# ---------------------------------------------------------------------------
# TABLE 8
# ---------------------------------------------------------------------------
def table8(us, cn):
    w("\n## Table 8 — rank groups: deciles of score within market and year\n")
    w("Firms are ranked by score **within each market AND each fiscal year**, "
      "then cut into tenths. **Decile 1 = the lowest scores = the strongest "
      "fundamentals; decile 10 = the weakest.** Each market's weakest tenth is "
      "therefore compared with the other market's weakest tenth, so the "
      f"{MIN_FIRMS}-firm minimum cannot bind: every decile holds a tenth of "
      "its own market. The category columns are the S&P categories of Table "
      "7(b), with CCC, CC and C pooled.\n")
    for m in MODELS:
        a, b = us.copy(), cn.copy()
        for d in (a, b):
            pct = d.groupby("fyear")[f"s_{m}"].rank(pct=True)
            d["dec"] = np.clip(np.ceil(pct * 10), 1, 10).astype(int)
        for title, groups in (("by decile", [(f"D{k}", [k]) for k in
                                             range(1, 11)]),
                              ("cumulative groups", DEC_GROUPS)):
            w(f"\n**{m} — {title}**\n")
            w("| group | market | rows | firms | " + " | ".join(BANDS)
              + " | median | IG % | mean notch | 95% CI |")
            w("|---|---|---|---|" + "---|" * len(BANDS) + "---|---|---|---|")
            for lab, ks in groups:
                pa, pb = a[a.dec.isin(ks)], b[b.dec.isin(ks)]
                for mk, p in (("US", pa), ("China", pb)):
                    sh = " | ".join(f"{(p.band == bd).mean()*100:.1f}"
                                    for bd in BANDS)
                    mn, dr = firm_boot(p, "notch",
                                       SEED_US if mk == "US" else SEED_CN)
                    lo, hi = pct_ci(dr)
                    w(f"| {lab} | {mk} | {len(p):,} | {p.firm.nunique():,} | "
                      f"{sh} | {NOTCH_LETTER[int(round(p.notch.median()))]} | "
                      f"{p.ig.mean()*100:.2f} | {mn:.3f} | "
                      f"[{lo:.3f}, {hi:.3f}] |")
                nfa, nfb = pa.firm.nunique(), pb.firm.nunique()
                if nfa < MIN_FIRMS or nfb < MIN_FIRMS:
                    w(f"| {lab} | **CN - US** | | | "
                      + " | ".join([""] * len(BANDS))
                      + f" | | | below min cell ({nfa}, {nfb}) | |")
                    continue
                an, ad = firm_boot(pa, "notch", SEED_US)
                bn, bd = firm_boot(pb, "notch", SEED_CN)
                ai, aid = firm_boot(pa, "ig", SEED_US)
                bi, bid = firm_boot(pb, "ig", SEED_CN)
                nl, nh = pct_ci(bd - ad)
                il, ih = pct_ci((bid - aid) * 100)
                w(f"| {lab} | **CN - US** | | | "
                  + " | ".join([""] * len(BANDS))
                  + f" | | IG {(bi-ai)*100:+.2f}pp [{il:+.2f}, {ih:+.2f}] "
                  f"| {bn-an:+.3f} | [{nl:+.3f}, {nh:+.3f}] |")
                if m == "xgb" and lab == "weakest 10%":
                    CHECKS["t8"] = (bn - an, nl, nh)


# ---------------------------------------------------------------------------
# --scores raw only: the two rank-based tables under both US score versions
# ---------------------------------------------------------------------------
def rank_based_comparison(us, cn):
    """Tables 5 and 8 under the raw and the calibrated US score, side by side.

    POINT ESTIMATES ONLY, no bootstrap. Both tables are rank-based, so if the
    calibration is a near-monotone transform of the raw score these should
    barely move; the difference column is printed so that can be read off.

    The CHINESE scores are identical in the two runs, so the Chinese columns of
    both tables cannot move. Only the US side is shown, plus the China-minus-US
    difference, which moves because the US deciles are re-cut on the US score.
    """
    w("\n## Rank-based tables under both US score versions — point estimates only\n")
    w("Printed because this run used the RAW US scores. The Chinese scores are "
      "the same in both runs, so the Chinese columns of Tables 5 and 8 cannot "
      "move and are not repeated; the US side and the China-minus-US difference "
      "can. **No bootstrap is computed here.**\n")

    w("\n**Table 5 — Spearman of score with notch, US side**\n")
    w("| model | rows | all rated: raw | all rated: old | diff | rows AAA..AA- "
      "| AAA..AA-: raw | AAA..AA-: old | diff |")
    w("|---|---|---|---|---|---|---|---|---|")
    top = us[us.notch <= 4]
    for m in MODELS:
        ra = spearmanr(us[f"s_{m}"], us.notch).statistic
        oa = spearmanr(us[f"old_{m}"], us.notch).statistic
        if top.notch.nunique() < 2 or len(top) < 3:
            rt = ot = np.nan
        else:
            rt = spearmanr(top[f"s_{m}"], top.notch).statistic
            ot = spearmanr(top[f"old_{m}"], top.notch).statistic
        w(f"| {m} | {len(us):,} | {fmt(ra)} | {fmt(oa)} | {fmt(ra - oa)} | "
          f"{len(top):,} | {fmt(rt)} | {fmt(ot)} | {fmt(rt - ot)} |")

    w("\n**Table 8 — by score decile within market and fiscal year.** The "
      "deciles are re-cut on whichever US score the column names, so the US row "
      "membership itself can differ between raw and old.\n")
    groups = [(f"D{k}", [k]) for k in range(1, 11)] + DEC_GROUPS
    for m in MODELS:
        w(f"\n*{m}*\n")
        w("| group | US n raw | US n old | US mean notch raw | old | diff | "
          "US IG % raw | old | diff | CN-US notch raw | old | diff |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        a = us.copy()
        for tag, col in (("raw", f"s_{m}"), ("old", f"old_{m}")):
            p = a.groupby("fyear")[col].rank(pct=True)
            a[f"dec_{tag}"] = np.clip(np.ceil(p * 10), 1, 10).astype(int)
        b = cn.copy()
        p = b.groupby("fyear")[f"s_{m}"].rank(pct=True)
        b["dec"] = np.clip(np.ceil(p * 10), 1, 10).astype(int)
        for lab, ks in groups:
            pb = b[b.dec.isin(ks)]
            cells = []
            for tag in ("raw", "old"):
                pa = a[a[f"dec_{tag}"].isin(ks)]
                cells.append((len(pa), pa.notch.mean(), pa.ig.mean() * 100,
                              pb.notch.mean() - pa.notch.mean()))
            (nr, mr, ir, dr), (no, mo, io, do) = cells
            w(f"| {lab} | {nr:,} | {no:,} | {mr:.3f} | {mo:.3f} | {mr - mo:+.3f} "
              f"| {ir:.2f} | {io:.2f} | {ir - io:+.2f} | {dr:+.3f} | {do:+.3f} "
              f"| {dr - do:+.3f} |")


# ---------------------------------------------------------------------------
# --scores raw only: three figures
# ---------------------------------------------------------------------------
def figures(us, cn):
    """Build the three series and hand them to `inflation_figures`.

    THIS FUNCTION DRAWS NOTHING. Every style choice — colours, fonts, titles,
    tick labels, the speculative shading, the peak and gap annotations — lives
    in `code/inflation_figures.py`, which is not modified by this script.

    Figures 1 and 3 plot `FIGDATA`, filled by `table1` and `table7` as they ran,
    so no share is recomputed and the firm bootstrap is not run twice. Figure
    2's grouping is not a table, so it is cut here from the same in-memory `us`
    and `cn`.

    `inflation_figures` imports matplotlib at module scope, so it is imported
    HERE rather than at the top of this file: `--scores old` neither imports it
    nor depends on it.
    """
    import inflation_figures as ifg

    print("\n" + "=" * 78)
    print("FIGURES — numbers from the tables, drawing from inflation_figures.py")
    print("=" * 78)
    w("\n## Figures\n")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: the 22 Table 1 notch shares ----------------------------
    # Indexed by notch so a notch absent from BOTH markets becomes 0.0 rather
    # than shortening the list and silently shifting every later letter.
    us_pct, cn_pct = [0.0] * 22, [0.0] * 22
    for r in FIGDATA["t1"]:
        n = int(r["notch"])
        if NOTCH_LETTER[n] != ifg.LETTERS[n - 1]:
            raise SystemExit(f"STOP: notch {n} is {NOTCH_LETTER[n]!r} here and "
                             f"{ifg.LETTERS[n - 1]!r} in inflation_figures.")
        us_pct[n - 1], cn_pct[n - 1] = r["US %"], r["China %"]
    t1 = pd.DataFrame({"letter": ifg.LETTERS, "US %": us_pct,
                       "China %": cn_pct})
    print("\n  FIGURE 1 — rating distribution, the Table 1 shares")
    print(t1.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ifg.fig1(us_pct, cn_pct, str(FIG_DIR / FIG_NAMES[1]))
    print(f"      drew {FIG_NAMES[1]}.png and .pdf")
    w(f"- **Figure 1**, overall rating distribution: "
      f"`output/figures/{FIG_NAMES[1]}.png` and `.pdf`")

    # ---- Figure 2: by level of fundamentals -------------------------------
    groups, rows = [], []
    for key, label, test in FIG_GROUPS:
        a, b = us[test(us.s_xgb)], cn[test(cn.s_xgb)]
        su = [(a.band == bd).mean() * 100 if len(a) else 0.0 for bd in BANDS]
        sc = [(b.band == bd).mean() * 100 if len(b) else 0.0 for bd in BANDS]
        groups.append((label, len(a), len(b), su, sc))
        for bd, u, c in zip(BANDS, su, sc):
            rows.append({"group": key, "band": bd, "US %": u, "China %": c})
    t2 = pd.DataFrame(rows)
    print("\n  FIGURE 2 — rating distribution by level of fundamentals")
    print("      group counts: " + ";  ".join(
        f"{k} US {g[1]:,} / China {g[2]:,}"
        for (k, _l, _t), g in zip(FIG_GROUPS, groups)))
    print(t2.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    su_n = sum(g[1] for g in groups)
    sc_n = sum(g[2] for g in groups)
    ok = su_n == len(us) and sc_n == len(cn)
    print(f"      {'PASSED' if ok else 'FAILED'}   group counts sum to each "
          f"market's total: US {su_n:,} of {len(us):,}, "
          f"China {sc_n:,} of {len(cn):,}")
    if not ok:
        raise SystemExit("STOP: the three score groups do not partition the "
                         "sample. NO FIGURE DRAWN.")
    # fig2 takes (label, US shares, China shares) only; the counts stay here for
    # the printed lines and the check above.
    ifg.fig2([(g[0], g[3], g[4]) for g in groups], str(FIG_DIR / FIG_NAMES[2]))
    print(f"      drew {FIG_NAMES[2]}.png and .pdf")
    w(f"- **Figure 2**, rating distribution by level of fundamentals: "
      f"`output/figures/{FIG_NAMES[2]}.png` and `.pdf`")

    # ---- Figure 3: the Table 7(a) xgb means and intervals ------------------
    t3 = pd.DataFrame(FIGDATA["t7a"])
    if len(t3) != 10:
        raise SystemExit(f"STOP: Table 7(a) gave {len(t3)} bands, not 10.")
    print("\n  FIGURE 3 — mean actual rating by score band, Table 7(a) xgb")
    print(t3.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ifg.fig3(t3["US mean"].to_numpy(), t3["US lo"].to_numpy(),
             t3["US hi"].to_numpy(), t3["China mean"].to_numpy(),
             t3["China lo"].to_numpy(), t3["China hi"].to_numpy(),
             str(FIG_DIR / FIG_NAMES[3]), model="xgb")
    print(f"      drew {FIG_NAMES[3]}.png and .pdf")
    w(f"- **Figure 3**, mean actual rating by score band: "
      f"`output/figures/{FIG_NAMES[3]}.png` and `.pdf`")

    # ---- Figure 2, rf version: the same three groups, cut on the rf score ---
    groups_rf, rows_rf = [], []
    for key, label, test in FIG_GROUPS:
        a, b = us[test(us.s_rf)], cn[test(cn.s_rf)]
        su = [(a.band == bd).mean() * 100 if len(a) else 0.0 for bd in BANDS]
        sc = [(b.band == bd).mean() * 100 if len(b) else 0.0 for bd in BANDS]
        groups_rf.append((label, len(a), len(b), su, sc))
        for bd, u, c in zip(BANDS, su, sc):
            rows_rf.append({"group": key, "band": bd, "US %": u, "China %": c})
    print("\n  FIGURE 2, rf — rating distribution by level of fundamentals")
    print("      group counts: " + ";  ".join(
        f"{k} US {g[1]:,} / China {g[2]:,}"
        for (k, _l, _t), g in zip(FIG_GROUPS, groups_rf)))
    print(pd.DataFrame(rows_rf).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    # The groups must cover each market exactly once, and each market must still
    # be the known sample. Checked before drawing, so a failure draws nothing.
    su_n = sum(g[1] for g in groups_rf)
    sc_n = sum(g[2] for g in groups_rf)
    ok = (su_n == len(us) == FIG_RF_TOTAL_US
          and sc_n == len(cn) == FIG_RF_TOTAL_CN)
    print(f"      {'PASSED' if ok else 'FAILED'}   rf group counts: US {su_n:,} "
          f"(sample {len(us):,}, expected {FIG_RF_TOTAL_US:,}), China {sc_n:,} "
          f"(sample {len(cn):,}, expected {FIG_RF_TOTAL_CN:,})")
    if not ok:
        raise SystemExit("STOP: the rf score groups do not add up to the "
                         "expected samples. NO RF FIGURE DRAWN.")
    ifg.fig2([(g[0], g[3], g[4]) for g in groups_rf],
             str(FIG_DIR / FIG_NAMES_RF[2]))
    print(f"      drew {FIG_NAMES_RF[2]}.png and .pdf")
    w(f"- **Figure 2, rf**, rating distribution by level of fundamentals: "
      f"`output/figures/{FIG_NAMES_RF[2]}.png` and `.pdf`")

    # ---- Figure 3, rf version: the Table 7(a) rf means and intervals --------
    t3r = pd.DataFrame(FIGDATA["t7a_rf"])
    if len(t3r) != 10:
        raise SystemExit(f"STOP: Table 7(a) rf gave {len(t3r)} bands, not 10.")
    # Table 7's 20-firm rule, applied to each market on its own: a band below
    # MIN_FIRMS in a market is not drawn for that market.
    t3r["US drawn"] = t3r["US firms"] >= MIN_FIRMS
    t3r["China drawn"] = t3r["China firms"] >= MIN_FIRMS
    print("\n  FIGURE 3, rf — mean actual rating by score band, Table 7(a) rf")
    print(t3r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ifg.fig3(t3r["US mean"].to_numpy(), t3r["US lo"].to_numpy(),
             t3r["US hi"].to_numpy(), t3r["China mean"].to_numpy(),
             t3r["China lo"].to_numpy(), t3r["China hi"].to_numpy(),
             str(FIG_DIR / FIG_NAMES_RF[3]), model="rf",
             us_ok=t3r["US drawn"].to_numpy(),
             cn_ok=t3r["China drawn"].to_numpy())
    print(f"      drew {FIG_NAMES_RF[3]}.png and .pdf")
    w(f"- **Figure 3, rf**, mean actual rating by score band: "
      f"`output/figures/{FIG_NAMES_RF[3]}.png` and `.pdf`")



# ---------------------------------------------------------------------------
def loud_check():
    """Assert the five point estimates against the one-off runs of 2026-09-30.

    The point estimates STOP the run if they differ. The intervals are reported
    either way: a bootstrap interval can legitimately shift if the firm set a
    cell resamples over changes, so it is reported rather than enforced."""
    print("\n" + "=" * 78)
    print("LOUD CHECK — the new tables against the one-off runs of 2026-09-30")
    print("=" * 78)
    bad = []
    for key, (want, (wlo, whi), nd, what) in EXPECT.items():
        if key not in CHECKS:
            bad.append(f"{what}: NOT COMPUTED, key {key} missing")
            print(f"  {what}\n      NOT COMPUTED — key {key} never set")
            continue
        got, glo, ghi = CHECKS[key]
        same = round(got, nd) == round(want, nd)
        print(f"  {what}")
        print(f"      point    expected {want:+.{nd}f}   got {got:+.{nd}f}   "
              f"{'MATCH' if same else 'DIFFERS'}")
        ci_same = (round(glo, nd) == round(wlo, nd)
                   and round(ghi, nd) == round(whi, nd))
        print(f"      interval expected [{wlo:+.{nd}f}, {whi:+.{nd}f}]   "
              f"got [{glo:+.{nd}f}, {ghi:+.{nd}f}]   "
              f"{'MATCH' if ci_same else 'DIFFERS'}")
        if not same:
            bad.append(f"{what}: expected {want:+.{nd}f}, got {got:+.{nd}f}")
    if bad:
        print("\nSTOP: the new tables do not reproduce the one-off runs.")
        for x in bad:
            print(f"  {x}")
        raise SystemExit("NOTHING WRITTEN.")
    print("\n  PASSED — all five point estimates reproduce.")


# ---------------------------------------------------------------------------
def main():
    us, cn = load()

    w("# Rating inflation — descriptive statistics")
    w("")
    w(f"**Date of run** see `docs/CHANGELOG.md` · **window** fiscal years "
      f"{YEAR_MIN}-{YEAR_MAX} · **design: D-72 (Tables 1-5), D-73 "
      f"(Tables 6-8)**")
    w("")
    w(f"**US scores: {US_SCORE_LABEL}.** The Chinese scores are raw in either "
      f"case (D-67). Run with `--scores {SCORES_VERSION}`.")
    w("")
    w("**THE FRAMING.** Ratings should reflect fundamental strength. The rating "
      "model is an **exogenous model of the rating a firm's fundamentals "
      "warrant**, estimated on the US sample, where ratings are taken to reflect "
      "fundamentals. The identifying assumption is that **US ratings carry no "
      "systematic inflation**; it is an assumption, not a finding, and is not "
      "testable inside this design. **Inflation is the actual Chinese rating "
      "measured against the fundamentals-warranted rating.**")
    w("")
    # The middle sentence differs by score version. Under `old` the US column is
    # a CALIBRATED probability, so the share reading holds; under `raw` neither
    # side is calibrated and the means are comparable but are not probabilities.
    if SCORES_VERSION == "old":
        reading = ("A group's mean score is the estimated share of that group "
                   "whose fundamentals warrant speculative grade.")
    else:
        reading = ("The scores are uncalibrated model outputs on both sides "
                   "(US out-of-fold, China full-sample). They are compared "
                   "across the two markets on the same footing and are not "
                   "read as probabilities.")
    w("**ORIENTATION. HIGHER RATING SCORE = FUNDAMENTALS WARRANT A WORSE "
      "RATING.** " + reading + " At the same actual rating letter, "
      "a higher Chinese mean means Chinese issuers of that letter are weaker than "
      "US issuers of that letter, which is inflation; a lower Chinese mean is the "
      "opposite.")
    w("")
    # The conservative-reading paragraph argues from the size of the CALIBRATION
    # gap, so it applies only where the US column is calibrated.
    if SCORES_VERSION == "old":
        w("**THE READING IS CONSERVATIVE.** The calibration check on the US "
          "rating panel found the mean score below the realised "
          "speculative-grade share in every decile of every model, by 3.42 to "
          "3.59 points overall. So this reading understates inflation rather "
          "than overstating it (D-72 item 3).")
        w("")
    w(f"**Samples.** US {len(us):,} firm-years, {us.firm.nunique():,} gvkeys. "
      f"China {len(cn):,} firm-years, {cn.firm.nunique():,} Orgids. Rated rows "
      f"only. `xgb` is the headline model; `rf` and `enet` are reported alongside.")
    w("")
    w("**No size variable appears below.** Size enters at the regression step "
      "(D-72 item 14). **Any time pattern carries D-64**: the Chinese rated pool "
      "changes composition through exit, which year effects do not absorb.")

    table1(us, cn)
    table2(us, cn)
    table2b(cn)
    table3(us, cn)
    table4(us, cn)
    table5(us, cn)
    robustness(us, cn)

    # Tables 6-8, folded in from the one-off checks (D-73). The loud check runs
    # BEFORE the write, so a failure leaves no .md behind.
    print("\n" + "=" * 78)
    print("TABLES 6-8 — derived columns")
    print("=" * 78)
    add_cols(us, cn)
    table6(us, cn)
    table7(us, cn)
    table8(us, cn)
    # The five asserted point estimates were measured on the CALIBRATED US
    # scores, so they can only be reproduced by the `old` path.
    if SCORES_VERSION == "old":
        loud_check()
    else:
        print("\n" + "=" * 78)
        print("LOUD CHECK SKIPPED — it asserts five point estimates measured on "
              "the CALIBRATED\n  US scores, which this run replaced. It applies "
              "to --scores old only.")
        print("=" * 78)
        rank_based_comparison(us, cn)
        figures(us, cn)

    w("")
    w("---")
    w("")
    w("NO VERDICT IS STATED IN THIS FILE. Every line above is a measured "
      "quantity.")

    MD_OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nWrote {MD_OUT.name}: {len(LINES):,} lines")


if __name__ == "__main__":
    main()
