# ABOUTME: Stage 1 batch 2 - the 22 characteristics that need more than the annual panel
# ABOUTME: 9 annual (rolling windows, industry adjustment), 13 quarterly
"""
stage1_construct_features_2.py

BATCH 2: the 22 characteristics that batch 1 could not build, because each
needs an input `stage1_1_construct_features.py` never opens.

    13 quarterly   read output/compustat_quarterly_clean.parquet
     3 annual      need a multi-year rolling window per firm
     6 annual      need an industry code or a cross-sectional regression

`AccrualQuality` and `AccrualQualityJune` were dropped by decision - see D-33.
The project therefore builds 99 of CZ's 101 characteristics, not 101.

Conventions, helpers and guards are IMPORTED from batch 1 so there is one
source of truth: `safe_div`, `lag`, `year_lag`, `one_row_per_year`,
`back_to_panel`, `load_master`, and the permno / minimum-firm assertions.
Read that file's header first; it explains the four standing deviations from CZ
(key, frequency, SignalMasterTable, industry code) that apply here too.

HOW A QUARTERLY OBSERVATION IS PLACED ON AN ANNUAL ROW is settled and is CZ's
rule, not ours - see D-31 and `quarterly_to_annual()`. It reads as an open
question in PROJECT_REPORT.md section 5 because the answer lives in
`DataDownloads/CompustatQuarterly.py` rather than in any predictor script.

ONE THING IS STILL OPEN: `WW_Q`'s firm sales-growth lag, where D-15's wording
and D-15's own reasoning point in different directions. Flagged in that
function's docstring; one quarter is implemented.

Inputs:
    output/compustat_annual_clean.parquet
    output/compustat_quarterly_clean.parquet
    data_raw/us/comp_names.parquet

Outputs:
    output/stage1_2_feature_panel.parquet     gvkey x datadate, 22 columns
    output/stage1_2_feature_coverage.csv

How to run:
    python3 code/stage1_construct_features_2.py
"""

import numpy as np
import pandas as pd

from stage1_1_construct_features import (
    KEY, MIN_FIRMS, OUTDIR,
    safe_div, year_lag, one_row_per_year, back_to_panel, load_master,
)

QUARTERLY = OUTDIR / "compustat_quarterly_clean.parquet"

# How many months a quarterly observation stays in force. THIS IS CZ's, not a
# choice of ours: CZ expands its quarterly panel over `month_offsets = [0,1,2]`
# (CompustatQuarterly.py line 175), so a quarter is carried for three months and
# then goes missing rather than growing stale. See quarterly_to_annual().
QUARTERLY_VALID_MONTHS = 3


# ===========================================================================
# QUARTERLY PANEL - loading, lagging, and the undecided alignment
# ===========================================================================

def load_quarterly():
    """The quarterly panel, with a single integer quarter index for lagging.

    CZ lags quarterly data in months on a monthly panel: `l3.` is one quarter
    back, `l12.` is four. Our panel is keyed on the fiscal quarter, so
    `qindex = fyearq*4 + fqtr - 1` turns both into plain integer arithmetic and
    a gap in a firm's quarterly history correctly yields a missing lag.
    """
    q = pd.read_parquet(QUARTERLY)
    q = q[q["fyearq"].notna() & q["fqtr"].notna()].copy()
    q["qindex"] = (q["fyearq"].astype("int64") * 4
                   + q["fqtr"].astype("int64") - 1)
    q = q.sort_values(["gvkey", "qindex"]).reset_index(drop=True)
    print(f"  quarterly panel: {len(q):,} rows, {q.gvkey.nunique():,} firms")
    assert "permno" not in q.columns, "permno in quarterly panel"
    return q


def qlag(q, cols, quarters=1):
    """Lag `cols` by `quarters` fiscal quarters, matched on the quarter index."""
    if isinstance(cols, str):
        cols = [cols]
    src = q[["gvkey", "qindex"] + cols].copy()
    src["qindex"] = src["qindex"] + quarters
    src = src.rename(columns={c: f"q{quarters}_{c}" for c in cols})
    return q.merge(src, on=["gvkey", "qindex"], how="left")


def dense_quarters(d, cols):
    """
    Each firm on a complete quarter grid, the quarterly twin of dense_years().

    Needed wherever a characteristic takes many lags or a rolling window. A
    `groupby.shift(k)` or a `rolling(n)` counts ROWS, so a firm that skipped two
    quarters would silently have its window reach further back than intended.
    On a dense grid the skipped quarter is a missing row, so a positional shift
    is a true quarter lag and `min_periods` rejects a window with a hole.
    """
    d = d.drop_duplicates(["gvkey", "qindex"], keep="last")
    span = d.groupby("gvkey")["qindex"].agg(["min", "max"])
    width = (span["max"] - span["min"] + 1).astype(int)
    grid = pd.DataFrame({
        "gvkey": np.repeat(span.index.to_numpy(), width.to_numpy()),
        "qindex": np.concatenate([np.arange(a, b + 1)
                                  for a, b in zip(span["min"], span["max"])]),
    })
    return grid.merge(d[["gvkey", "qindex"] + cols], on=["gvkey", "qindex"],
                      how="left")


def back_to_quarters(q, g, col):
    """Carry a column computed on the dense quarter grid back onto `q`."""
    return q[["gvkey", "qindex"]].merge(
        g[["gvkey", "qindex", col]], on=["gvkey", "qindex"], how="left")[col]


def quarterly_to_annual(master, q, cols):
    """
    Place a quarterly characteristic on the annual rows, the way CZ does it.

    THIS IS NOT A CHOICE WE INVENTED. CZ answers it, and the answer had to be
    read out of `DataDownloads/CompustatQuarterly.py` rather than out of any
    predictor script, which is why it looked open for so long:

      - the quarterly panel gets `time_avail_m = datadate + 3 months`,
        overridden by the report date `rdq` when that is later (lines 92-102) -
        the same rule our Stage 1.0 already uses;
      - it is then expanded over `month_offsets = [0, 1, 2]` (line 175), so a
        quarter is in force for THREE months only, not twelve like the annual
        panel, and afterwards the cell is missing;
      - predictors then join the two monthly panels on `gvkey time_avail_m`
        with `keep(match)`, so a row survives only where both have a value.

    Translated to an annual panel: take the most recent quarterly observation
    whose availability date is at or before the annual row's, **and discard it
    if it is more than three months stale**. Without that cap a firm that
    stopped filing quarterly reports in 2015 would still carry its 2015 figures
    on a 2020 row, which CZ explicitly does not allow. `merge_asof`'s
    `tolerance` argument is exactly this.

    The one genuine choice left is the ANCHOR: CZ has twelve monthly values per
    annual observation and we need one. We anchor on the annual row's own
    `date_avail`, the first month at which it becomes usable, which keeps the
    whole panel on a single availability convention.
    """
    left = (master[["gvkey", "date_avail"]]
            .sort_values("date_avail").reset_index())
    right = (q[["gvkey", "date_avail"] + cols]
             .dropna(subset=["date_avail"])
             .sort_values("date_avail"))
    out = pd.merge_asof(
        left, right, on="date_avail", by="gvkey", direction="backward",
        tolerance=pd.Timedelta(days=31 * QUARTERLY_VALID_MONTHS))
    return out.set_index("index").sort_index()[cols]


# ===========================================================================
# ROLLING WINDOWS - shared machinery for the five annual rolling features
# ===========================================================================

def dense_years(d):
    """
    Every firm laid out on a complete fiscal-year grid, min to max.

    A rolling window counted in ROWS would silently reach further back than it
    should for a firm with a gap in its fiscal-year sequence. On a dense grid
    the gap becomes a missing value, and `min_periods` then rejects the window.
    """
    span = d.groupby("gvkey")["fyear"].agg(["min", "max"])
    width = (span["max"] - span["min"] + 1).astype(int)
    grid = pd.DataFrame({
        "gvkey": np.repeat(span.index.to_numpy(), width.to_numpy()),
        "fyear": np.concatenate([np.arange(a, b + 1)
                                 for a, b in zip(span["min"], span["max"])]),
    })
    return grid.merge(d, on=["gvkey", "fyear"], how="left")


def rolling_stat(g, col, window, minobs, stat):
    """Rolling `stat` over `window` fiscal years, needing `minobs` values.

    `g` must come from dense_years(). A window is only valid once the row has
    `window - 1` earlier rows of its OWN firm, which the position check below
    enforces - the grid is sorted and each firm's rows are contiguous.
    """
    s = g[col].astype("float64")
    out = s.rolling(window, min_periods=minobs).agg(stat)
    within_firm = g.groupby("gvkey").cumcount() >= window - 1
    return out.where(within_firm)


def ols_resid(df, y, xs):
    """Residuals from an OLS of `y` on `xs` with an intercept, or NaN.

    Used for the by-group cross-sectional regressions. Returns NaN for the
    whole group if the group is rank-deficient, rather than raising.
    """
    d = df[[y] + xs].astype("float64").dropna()
    if len(d) <= len(xs) + 1:
        return pd.Series(np.nan, index=df.index)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in xs])
    try:
        beta, *_ = np.linalg.lstsq(X, d[y].to_numpy(), rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=df.index)
    r = pd.Series(np.nan, index=df.index)
    r.loc[d.index] = d[y].to_numpy() - X @ beta
    return r


# ===========================================================================
# ANNUAL - rolling windows
# ===========================================================================

def f_Investment(m):
    """CZ: Investment.do.
        gen Investment = capx/revt
        asrol Investment, window(time_avail_m 36) min(24) stat(mean) -> tempMean
        replace Investment = Investment/tempMean
        replace Investment = . if revt < 10
    Deviations: (1) key, (2) 36/24 months -> 3 years, minimum 2.
    The revt < 10 cut is in millions and is the source paper's, so it stays."""
    d = one_row_per_year(m, ["capx", "revt"])
    d["inv_"] = safe_div(d["capx"], d["revt"])
    g = dense_years(d)
    g["mean3"] = rolling_stat(g, "inv_", window=3, minobs=2, stat="mean")
    g["Investment"] = safe_div(g["inv_"], g["mean3"])
    g.loc[(g["revt"] < 10).fillna(False), "Investment"] = np.nan
    return back_to_panel(m, g, "Investment")


def f_EarningsSmoothness(m):
    """CZ: ZZ2 EarningsSmoothness.do.
        tempEarnings = ib/l.at
        tempCF = (ib - ((act-l.act) - (lct-l.lct) - (che-l.che) + (dlc-l.dlc) - dp))/l.at
        10-year rolling sd of each, min 10
        EarningsSmoothness = sd(tempEarnings)/sd(tempCF)
    Deviations: (1) key only - CZ already runs this annually on gvkey/fyear."""
    d = one_row_per_year(m, ["ib", "at", "act", "lct", "che", "dlc", "dp"])
    d = year_lag(d, ["at", "act", "lct", "che", "dlc"], 1)
    accr = ((d["act"] - d["l1_act"]) - (d["lct"] - d["l1_lct"])
            - (d["che"] - d["l1_che"]) + (d["dlc"] - d["l1_dlc"]) - d["dp"])
    d["earn_"] = safe_div(d["ib"], d["l1_at"])
    d["cf_"] = safe_div(d["ib"] - accr, d["l1_at"])
    g = dense_years(d)
    sd_e = rolling_stat(g, "earn_", 10, 10, "std")
    sd_c = rolling_stat(g, "cf_", 10, 10, "std")
    g["EarningsSmoothness"] = safe_div(sd_e, sd_c)
    return back_to_panel(m, g, "EarningsSmoothness")


def f_RDAbility(m):
    """
    CZ: RDAbility.do. R&D productivity, Cohen-Diether-Malloy.
        tempY = log(sale/l.sale) ; tempX = log(1 + xrd/sale), both sides
            set missing where the raw item is negative
        for n in 1..5: regress tempY on l`n'.tempX, window(fyear 8) min(6),
            keep the slope as gammaAbility`n`; blank it where fewer than half
            the window's lagged X values are positive
        RDAbility = rowmean(gammaAbility1..5)
        keep only the top tercile of xrd/sale within the period; blank xrd <= 0

    Deviations: (1) key, (2) the tercile is taken within the FISCAL YEAR rather
    than the month, and the 8-year window is fiscal years.

    Five rolling regressions of one regressor each, so the closed form from
    rolling sums is exact and is what makes this tractable - the same device as
    EarningsPersistence in batch 1.
    """
    d = one_row_per_year(m, ["xrd", "sale"])
    xrd = d["xrd"].astype("float64").where(lambda s: s >= 0)
    sale = d["sale"].astype("float64").where(lambda s: s >= 0)
    d["y_"] = np.log(safe_div(sale, year_lag(d.assign(s_=sale), "s_", 1)["l1_s_"])
                     .where(lambda s: s > 0).astype("float64"))
    d["x_"] = np.log((1 + safe_div(xrd, sale)).where(lambda s: s > 0)
                     .astype("float64"))
    g = dense_years(d)

    slopes = []
    for n in range(1, 6):
        g[f"xl{n}"] = g.groupby("gvkey")["x_"].shift(n)
        ok = g["y_"].notna() & g[f"xl{n}"].notna()
        x = g[f"xl{n}"].where(ok)
        y = g["y_"].where(ok)
        N, MIN = 8, 6

        def rs(s):
            return s.rolling(N, min_periods=MIN).sum()

        cnt = ok.astype("float64").rolling(N, min_periods=MIN).sum()
        sx, sy = rs(x), rs(y)
        cxx = rs(x * x) - sx ** 2 / cnt
        cxy = rs(x * y) - sx * sy / cnt
        beta = cxy / cxx.where(cxx.abs() > 1e-10)
        within = g.groupby("gvkey").cumcount() >= N - 1
        # "half of R&D observations" - at least half the window's lagged X > 0
        share = ((g[f"xl{n}"] > 0).astype("float64")
                 .rolling(N, min_periods=MIN).mean())
        slopes.append(beta.where(within & (share >= 0.5)))
    g["RDAbility"] = pd.concat(slopes, axis=1).mean(axis=1, skipna=True)

    rd = safe_div(g["xrd"].astype("float64"), g["sale"].astype("float64"))
    rd = rd.where(g["xrd"].astype("float64") > 0)
    # Top tercile of R&D intensity within the fiscal year, and positive R&D
    # only - both are the source paper's, see the docstring. CZ's `qcut` with
    # `duplicates="drop"` is kept rather than a rank: it can in principle
    # collapse to fewer than three bins and then blank a whole year, but
    # measured on this panel it never does - 0 of 77 fiscal years - and the two
    # implementations disagree on 1 row out of 160,894.
    tercile = rd.groupby(g["fyear"]).transform(
        lambda s: pd.qcut(s, 3, labels=False, duplicates="drop")
        if s.notna().sum() > 3 else pd.Series(np.nan, index=s.index))
    g.loc[~(tercile == 2).fillna(False), "RDAbility"] = np.nan
    return back_to_panel(m, g, "RDAbility")


# ===========================================================================
# ANNUAL - industry adjustment and cross-sectional regressions
# ===========================================================================

def f_tang(m):
    """CZ: tang.do. Tangibility, Hahn-Lee 2009.
        drop if sic < 2000 | sic > 3999      // manufacturing only
        tang = (che + .715*rect + .547*invt + .535*ppegt)/at
    Deviations: (1) key, (4) our industry code.
    The manufacturing restriction is the source paper's - Hahn and Lee define
    the measure for manufacturers only - so it stays, per D-30. CZ's script
    also builds a financial-constraint tercile from `at` and never uses it;
    that is dead code and is not carried over."""
    t = safe_div(m["che"] + 0.715 * m["rect"] + 0.547 * m["invt"]
                 + 0.535 * m["ppegt"], m["at"])
    sic = pd.to_numeric(m["ind_sic"], errors="coerce")
    manufacturing = ((sic >= 2000) & (sic <= 3999)).fillna(False)
    return t.mask(~manufacturing.values)


def f_realestate(m):
    """CZ: realestate.do. Real estate holdings relative to industry.
        re_new = (fatb+fatl)/ppegt ; re_old = (ppenb+ppenls)/ppent
        re = re_new, falling back to re_old
        keep industry-years with at least 5 firms
        realestate = re - mean(re) by (2-digit industry, period)
    Deviations: (1) key, (2) the industry mean is taken within the FISCAL YEAR
    rather than the month, (3) SignalMasterTable merge deleted, (4) our
    industry code instead of `sicCRSP`."""
    re_new = safe_div(m["fatb"] + m["fatl"], m["ppegt"])
    re_old = safe_div(m["ppenb"] + m["ppenls"], m["ppent"])
    re = re_new.fillna(re_old).astype("float64")
    re = re.where(m["at"].notna() & (m["ppent"].notna() | m["ppegt"].notna()))
    grp = [m["sic2D"], m["fyear"]]
    big_enough = re.groupby(grp).transform("count") >= 5
    re = re.where(big_enough.fillna(False))
    return re - re.groupby(grp).transform("mean")


def f_ChInvIA(m):
    """CZ: ChInvIA.do. Change in capital investment, industry adjusted.
        replace capx = ppent - l12.ppent if capx == .
        pchcapx = (capx - .5*(l12.capx + l24.capx))/(.5*(l12.capx + l24.capx))
        replace pchcapx = (capx - l12.capx)/l12.capx if missing
        ChInvIA = pchcapx - mean(pchcapx) by (2-digit industry, period)
    Deviations: (1) key, (2) lags in years and the mean within the fiscal year,
    (3) SignalMasterTable merge deleted, (4) our industry code."""
    d = one_row_per_year(m, ["capx", "ppent"])
    d = year_lag(d, "ppent", 1)
    sub = d["capx"].isna()
    d["capx"] = d["capx"].where(~sub, d["ppent"] - d["l1_ppent"])
    d = year_lag(year_lag(d, "capx", 1), "capx", 2)
    base = 0.5 * (d["l1_capx"] + d["l2_capx"])
    pch = safe_div(d["capx"] - base, base)
    pch = pch.fillna(safe_div(d["capx"] - d["l1_capx"], d["l1_capx"]))
    d["pch_"] = pch
    d = d.merge(m[["gvkey", "fyear", "sic2D"]].drop_duplicates(["gvkey", "fyear"]),
                on=["gvkey", "fyear"], how="left")
    grp = [d["sic2D"], d["fyear"]]
    d["ChInvIA"] = d["pch_"] - d["pch_"].groupby(grp).transform("mean")
    return back_to_panel(m, d, "ChInvIA")


def f_WW(m):
    """CZ: WW.do. Whited-Wu financial-constraint index.
        WW = -.091*(ib+dp)/(4*at) - .062*(dvpsx_c>0) + .021*dltt/at
             - .044*log(at) + .102*(IndSales/l12.IndSales - 1)/4
             - .035*(sale/l.sale - 1)/4
        IndSales = total sale by (3-digit industry, period)
    Deviations: (1) key, (2) the firm sales-growth term uses a ONE-YEAR lag per
    D-15, and the industry total is within the fiscal year, (3)
    SignalMasterTable merge deleted, (4) our industry code instead of
    `sicCRSP`."""
    d = one_row_per_year(m, ["ib", "dp", "at", "dvpsx_c", "dltt", "sale"])
    d = d.merge(m[["gvkey", "fyear", "sic3D"]].drop_duplicates(["gvkey", "fyear"]),
                on=["gvkey", "fyear"], how="left")
    d["indsale_"] = d.groupby([d["sic3D"], d["fyear"]])["sale"].transform("sum")
    d = year_lag(d, ["indsale_", "sale"], 1)
    pays_dividend = ((d["dvpsx_c"] > 0) & d["dvpsx_c"].notna()).astype("float64")
    d["WW"] = (-0.091 * safe_div(d["ib"] + d["dp"], 4 * d["at"])
               - 0.062 * pays_dividend
               + 0.021 * safe_div(d["dltt"], d["at"])
               - 0.044 * np.log(d["at"].where(d["at"] > 0).astype("float64"))
               + 0.102 * (safe_div(d["indsale_"], d["l1_indsale_"]) - 1) / 4
               - 0.035 * (safe_div(d["sale"], d["l1_sale"]) - 1) / 4)
    return back_to_panel(m, d, "WW")


def _abnormal_accruals(m):
    """
    Shared by AbnormalAccruals and AbnormalAccrualsPercent. CZ: ZZ2 script.
        tempCFO = oancf, else fopt - d(act) + d(che) + d(lct) - d(dlc)
        tempAccruals = (ib - tempCFO)/l.at
        tempInvTA = 1/l.at ; tempDelRev = d(sale)/l.at ; tempPPE = ppegt/l.at
        winsor2 temp*, cuts(0.1 99.9) trim by(fyear)
        regress tempAccruals on tempInvTA tempDelRev tempPPE, by (fyear, sic2)
        drop if fewer than 6 observations in the cell
        AbnormalAccruals = residual

    Deviations: (1) key, (2) lags in years, (3) SignalMasterTable merge deleted,
    (4) our industry code. CZ's `drop if exchcd == 3 & fyear < 1982` is NOT
    applied, per D-14.

    Note the winsorisation here is CZ's own, inside the characteristic, at
    0.1/99.9 by fiscal year and TRIMMING rather than capping. It is kept
    because the regression is fitted on it; it is not D-25.
    """
    d = one_row_per_year(m, ["oancf", "fopt", "act", "che", "lct", "dlc",
                             "ib", "sale", "ppegt", "at", "ni"])
    d = year_lag(d, ["act", "che", "lct", "dlc", "at", "sale"], 1)
    cfo = d["oancf"].astype("float64")
    alt = (d["fopt"] - (d["act"] - d["l1_act"]) + (d["che"] - d["l1_che"])
           + (d["lct"] - d["l1_lct"]) - (d["dlc"] - d["l1_dlc"]))
    cfo = cfo.fillna(alt.astype("float64"))
    d["acc_"] = safe_div(d["ib"] - cfo, d["l1_at"])
    d["inv_"] = safe_div(pd.Series(1.0, index=d.index), d["l1_at"])
    d["drev_"] = safe_div(d["sale"] - d["l1_sale"], d["l1_at"])
    d["ppe_"] = safe_div(d["ppegt"], d["l1_at"])

    # CZ trims, i.e. sets the tails missing, rather than capping them.
    # Cast to plain float64 first: these are built from nullable Float64
    # columns, and a comparison on those yields pd.NA, which `where` refuses.
    for c in ["acc_", "inv_", "drev_", "ppe_"]:
        d[c] = d[c].astype("float64")
        lo = d.groupby("fyear")[c].transform(lambda s: s.quantile(0.001))
        hi = d.groupby("fyear")[c].transform(lambda s: s.quantile(0.999))
        d[c] = d[c].where(((d[c] >= lo) & (d[c] <= hi)).fillna(False))

    d = d.merge(m[["gvkey", "fyear", "sic2D"]].drop_duplicates(["gvkey", "fyear"]),
                on=["gvkey", "fyear"], how="left")
    resid = (d.groupby(["fyear", "sic2D"], group_keys=False)
             .apply(lambda g: ols_resid(g, "acc_", ["inv_", "drev_", "ppe_"])
                    if g["acc_"].notna().sum() >= 6
                    else pd.Series(np.nan, index=g.index)))
    d["AbnormalAccruals"] = resid.reindex(d.index)
    d["AbnormalAccrualsPercent"] = safe_div(
        d["AbnormalAccruals"] * d["l1_at"], d["ni"].abs())
    return d


_ABNORMAL_CACHE = {}


def _abnormal_cached(m):
    """~5,000 small regressions, so compute once and serve both characteristics."""
    if "d" not in _ABNORMAL_CACHE:
        _ABNORMAL_CACHE["d"] = _abnormal_accruals(m)
    return _ABNORMAL_CACHE["d"]


def f_AbnormalAccruals(m):
    """Residual from the by-industry-year accruals regression. See
    _abnormal_accruals()."""
    return back_to_panel(m, _abnormal_cached(m), "AbnormalAccruals")


def f_AbnormalAccrualsPercent(m):
    """AbnormalAccruals x lagged assets / |ni|. See _abnormal_accruals()."""
    return back_to_panel(m, _abnormal_cached(m), "AbnormalAccrualsPercent")


# ===========================================================================
# QUARTERLY - computed on the quarterly panel, then aligned by
# quarterly_to_annual(). CZ's `l3.` is one quarter, `l12.` is four.
# ===========================================================================

def q_Cash(q):
    """CZ: Cash.py.   Cash = cheq/atq, only where atq > 0."""
    return safe_div(q["cheq"], q["atq"].where(q["atq"] > 0))


def q_roaq(q):
    """CZ: roaq.py.   roaq = ibq/l3.atq  (one quarter back)."""
    d = qlag(q, "atq", 1)
    return safe_div(d["ibq"], d["q1_atq"])


def q_CapTurnover_q(q):
    """CZ: CapTurnover_q.py.   saleq/l3.atq  (one quarter back)."""
    d = qlag(q, "atq", 1)
    return safe_div(d["saleq"], d["q1_atq"])


def q_PM_q(q):
    """CZ: PM_q.py.   PM_q = niq/revtq.
    CZ's Stata loads `mve_c` from SignalMasterTable and never uses it; the
    Python does not load it at all, which is why this survived the Stage 0.1
    market-variable filter. The merge is deleted."""
    return safe_div(q["niq"], q["revtq"])


def q_AssetGrowth_q(q):
    """CZ: AssetGrowth_q.py.   (atq - l12.atq)/l12.atq  (four quarters back)."""
    d = qlag(q, "atq", 4)
    return safe_div(d["atq"] - d["q4_atq"], d["q4_atq"])


def q_AssetLiquidityBookQuart(q):
    """CZ: AssetLiquidityBookQuart.do.
        gdwlq and intanq set to 0 where missing
        (cheq + .75*(actq-cheq) + .5*(atq-actq-gdwlq-intanq))/l.atq
    Deviation: CZ's `l.` is ONE MONTH on its monthly panel, a no-op in two
    months of three. Per D-15 this uses the PRIOR QUARTER's assets, which is
    the annual decision's quarterly counterpart."""
    d = qlag(q, "atq", 1)
    num = (d["cheq"] + 0.75 * (d["actq"] - d["cheq"])
           + 0.5 * (d["atq"] - d["actq"] - d["gdwlq"].fillna(0)
                    - d["intanq"].fillna(0)))
    return safe_div(num, d["q1_atq"])


def q_OperProfLag_q(q):
    """CZ: OperProfLag_q.do.
        temp<v> = <v> with 0 for missing, for cogsq xsgaq xintq
        num = revtq - tempcogsq - tempxsgaq - tempxintq, missing if all three
              of cogsq/xsgaq/xintq are missing
        tempSE = seqq, else ceqq + pstkq, else atq - ltq
        OperProfLag_q = num/(tempSE + txditcq - pstkq), and num/(tempSE - pstkq)
              where txditcq is missing
    Deviations: (1) key, (3) SignalMasterTable merge deleted.

    `pstkq` IS ZERO-FILLED, matching CZ's Python, which builds a `pstkq_filled`
    and uses it in both the equity fallback and the denominator. Our first
    version used the raw item and so returned missing wherever preferred stock
    was unreported; that is a difference from CZ, not a decision of ours.

    CZ's second denominator, `tempSE - pstkq` for missing `txditcq`, is NOT
    reproduced because it cannot differ: `txditcq` is zero-filled first, and
    `tempSE + 0 - pstkq` is the same number. The branch is arithmetically dead
    in CZ's own Stata too, which fills `txditcq` on the line above it."""
    num = (q["revtq"] - q["cogsq"].fillna(0) - q["xsgaq"].fillna(0)
           - q["xintq"].fillna(0))
    all_missing = (q["cogsq"].isna() & q["xsgaq"].isna()
                   & q["xintq"].isna()).to_numpy(dtype=bool)
    num = num.mask(all_missing)
    pstkq = q["pstkq"].fillna(0)
    se = q["seqq"].fillna(q["ceqq"] + pstkq).fillna(q["atq"] - q["ltq"])
    return safe_div(num, se + q["txditcq"].fillna(0) - pstkq)


def q_ChTax(q, annual):
    """
    CZ: ChTax.do.   ChTax = (txtq - l12.txtq)/l12.at

    The ONLY characteristic that mixes the two panels: a quarterly numerator
    over an ANNUAL lagged asset figure. CZ merges `m_aCompustat` and
    `m_QCompustat` on gvkey and month; we merge the prior fiscal year's annual
    `at` onto the quarter by fiscal year, which is the same quantity.
    """
    d = qlag(q, "txtq", 4)
    prior = annual[["gvkey", "fyear", "at"]].copy()
    prior["fyear"] = prior["fyear"] + 1
    prior = prior.rename(columns={"at": "l1_at_annual", "fyear": "fyearq"})
    prior = prior.drop_duplicates(["gvkey", "fyearq"], keep="last")
    d = d.merge(prior, on=["gvkey", "fyearq"], how="left")
    return safe_div(d["txtq"] - d["q4_txtq"], d["l1_at_annual"])


def _surprise(q, value_col):
    """
    Shared by EarningsSurprise and RevenueSurprise. CZ: their two .do files.
        GrTemp   = value - l12.value                   (four quarters back)
        Drift    = rowmean of GrTemp lagged 1..8 quarters
        Surprise = GrTemp - Drift
        SD       = rowsd of Surprise lagged 1..8 quarters
        result   = Surprise/SD

    Standardised unexpected earnings: a year-over-year quarterly change, less
    the average of the last eight such changes, scaled by their dispersion.
    CZ's `l3.` through `l24.` are quarterly lags 1 to 8, not months.

    Computed on the dense quarter grid so the sixteen lags are true quarter
    lags, and so this costs sixteen grouped shifts rather than sixteen merges
    of the whole two-million-row panel.
    """
    g = dense_quarters(q[["gvkey", "qindex", value_col]], [value_col])
    v = g[value_col].astype("float64")
    g["gr_"] = v - g.groupby("gvkey")[value_col].shift(4).astype("float64")
    drift = pd.concat([g.groupby("gvkey")["gr_"].shift(k) for k in range(1, 9)],
                      axis=1)
    g["surprise_"] = g["gr_"] - drift.mean(axis=1, skipna=True)
    spread = pd.concat(
        [g.groupby("gvkey")["surprise_"].shift(k) for k in range(1, 9)], axis=1)
    g["out_"] = safe_div(g["surprise_"], spread.std(axis=1, skipna=True))
    return back_to_quarters(q, g, "out_")


def q_EarningsSurprise(q):
    """CZ: EarningsSurprise.do. SUE on `epspxq`. See _surprise()."""
    return _surprise(q, "epspxq")


def q_RevenueSurprise(q):
    """CZ: RevenueSurprise.do. SUE on revenue per share, `revtq/cshprq`.
    CZ's Stata loads `mve_c` and never uses it; the merge is deleted."""
    d = q.assign(revps=safe_div(q["revtq"], q["cshprq"]))
    return _surprise(d, "revps")


def q_NumEarnIncrease(q):
    """
    CZ: NumEarnIncrease.do. Consecutive quarters of year-over-year earnings
    growth.
        chearn = ibq - l12.ibq
        nincr = k  when chearn > 0 for the current quarter and the k-1 quarters
                before it, and not for the k-th
    CZ writes eight nested conditions; the loop below is the same ladder, and
    it stops at 8 exactly as CZ's does - a run of nine or more consecutive
    increases satisfies no rung, because every rung requires the NEXT quarter
    to be non-positive, so the value stays 0. That is CZ's behaviour, not an
    oversight of ours.

    STATA MISSING SEMANTICS, and CZ's Python preserves them: `chearn > 0` is
    written there as `(chearn > 0) | chearn.isna()`, because missing is larger
    than any number in Stata. **A missing quarter therefore counts as a
    continuing increase.** The terminating test `l24.chearn <= 0` is false on
    missing, in Stata and in pandas alike, so it needs no special handling.
    """
    g = dense_quarters(q[["gvkey", "qindex", "ibq"]], ["ibq"])
    g["chearn_"] = (g["ibq"].astype("float64")
                    - g.groupby("gvkey")["ibq"].shift(4).astype("float64"))
    lags = [g["chearn_"]] + [g.groupby("gvkey")["chearn_"].shift(k)
                             for k in range(1, 9)]
    # Stata: missing > 0 is TRUE; missing <= 0 is FALSE.
    increased = [(s > 0) | s.isna() for s in lags]
    ended = [(s <= 0).fillna(False) for s in lags]

    nincr = pd.Series(0.0, index=g.index)
    run = increased[0]
    for k in range(1, 9):
        nincr = nincr.where(~(run & ended[k]), float(k))
        run = run & increased[k]
    g["nincr_"] = nincr
    return back_to_quarters(q, g, "nincr_")


def q_roavol(q):
    """CZ: roavol.do.
        roaq = ibq/l3.atq
        roavol = rolling sd of roaq, window 48 months, min 24
    Deviations: (1) key, (2) 48/24 months -> 16 quarters, minimum 8.

    On the dense quarter grid, so a firm that skipped quarters cannot have its
    window reach back further than sixteen of them."""
    d = qlag(q, "atq", 1)
    d["roaq_"] = safe_div(d["ibq"], d["q1_atq"]).astype("float64")
    g = dense_quarters(d[["gvkey", "qindex", "roaq_"]], ["roaq_"])
    rolled = g["roaq_"].rolling(16, min_periods=8).std()
    within_firm = g.groupby("gvkey").cumcount() >= 15
    g["roavol_"] = rolled.where(within_firm)
    return back_to_quarters(q, g, "roavol_")


def q_WW_Q(q, annual):
    """
    Quarterly Whited-Wu. CZ: WW_Q.py.
        tempIndSales = total saleq by (3-digit industry, period)
        WW_Q = -.091*(ibq+dpq)/atq - .062*(dvpsxq>0) + .021*dlttq/atq
               -.044*log(atq) + .102*(tempIndSales/l3.tempIndSales - 1)
               -.035*(saleq/l1.saleq - 1)

    Note this is NOT the annual formula on quarterly items: there is no `/4` on
    the profitability term or on either growth term, because the quarterly panel
    is already at a quarterly rate.

    Deviations: (1) key, (3) SignalMasterTable merge deleted, (4) our industry
    code instead of `sicCRSP`, and (2) the firm sales-growth lag - see below.

    THE FIRM SALES-GROWTH LAG, AND A TENSION WITH D-15. CZ's `l3.` on the
    industry term is three months, i.e. ONE QUARTER, and is used as such here.
    CZ's `l1.` on the firm term is ONE MONTH, which on its own monthly panel
    returns the same quarterly figure in two months of three - the degenerate
    lag D-15 was written about. **D-15's wording says `WW_Q` takes a one-YEAR
    lag there, but D-15's reasoning does not transfer**: it turns on CZ's `/4`
    "to approximate quarterly rates", and this formula has no `/4` at all. The
    internally consistent reading is ONE QUARTER, matching the industry term in
    the same expression. That is what is implemented; D-15 needs amending or
    this needs changing, and the choice is flagged rather than taken silently.

    The industry code lives on the annual master, not the quarterly panel, so
    it is merged on gvkey and fiscal year.
    """
    ind = (annual[["gvkey", "fyear", "sic3D"]]
           .drop_duplicates(["gvkey", "fyear"], keep="last")
           .rename(columns={"fyear": "fyearq"}))
    d = q.merge(ind, on=["gvkey", "fyearq"], how="left")
    d["indsale_"] = d.groupby([d["sic3D"], d["qindex"]])["saleq"].transform("sum")
    d = qlag(d, ["indsale_", "saleq"], 1)
    pays_dividend = ((d["dvpsxq"] > 0) & d["dvpsxq"].notna()).astype("float64")
    atq = d["atq"].astype("float64")
    return (-0.091 * safe_div(d["ibq"] + d["dpq"], atq)
            - 0.062 * pays_dividend
            + 0.021 * safe_div(d["dlttq"], atq)
            - 0.044 * np.log(atq.where(atq > 0))
            + 0.102 * (safe_div(d["indsale_"], d["q1_indsale_"]) - 1)
            - 0.035 * (safe_div(d["saleq"], d["q1_saleq"]) - 1))


# ===========================================================================
# REGISTRY
# ===========================================================================

ANNUAL_FEATURES = {
    "Investment":              (f_Investment,              ["capx", "revt", "fyear"]),
    "EarningsSmoothness":      (f_EarningsSmoothness,      ["ib", "at", "act", "lct", "che", "dlc", "dp", "fyear"]),
    "RDAbility":               (f_RDAbility,               ["xrd", "sale", "fyear"]),
    "tang":                    (f_tang,                    ["che", "rect", "invt", "ppegt", "at", "ind_sic"]),
    "realestate":              (f_realestate,              ["fatb", "fatl", "ppegt", "ppenb", "ppenls", "ppent", "at", "sic2D", "fyear"]),
    "ChInvIA":                 (f_ChInvIA,                 ["capx", "ppent", "sic2D", "fyear"]),
    "WW":                      (f_WW,                      ["ib", "dp", "at", "dvpsx_c", "dltt", "sale", "sic3D", "fyear"]),
    "AbnormalAccruals":        (f_AbnormalAccruals,        ["oancf", "fopt", "act", "che", "lct", "dlc", "ib", "sale", "ppegt", "at", "ni", "sic2D", "fyear"]),
    "AbnormalAccrualsPercent": (f_AbnormalAccrualsPercent, ["oancf", "fopt", "act", "che", "lct", "dlc", "ib", "sale", "ppegt", "at", "ni", "sic2D", "fyear"]),
}

# name -> (function, quarterly columns it reads, needs the annual panel too)
QUARTERLY_FEATURES = {
    "Cash":                    (q_Cash,                    ["cheq", "atq"], False),
    "roaq":                    (q_roaq,                    ["ibq", "atq"], False),
    "CapTurnover_q":           (q_CapTurnover_q,           ["saleq", "atq"], False),
    "PM_q":                    (q_PM_q,                    ["niq", "revtq"], False),
    "AssetGrowth_q":           (q_AssetGrowth_q,           ["atq"], False),
    "AssetLiquidityBookQuart": (q_AssetLiquidityBookQuart, ["cheq", "actq", "atq", "gdwlq", "intanq"], False),
    "OperProfLag_q":           (q_OperProfLag_q,           ["revtq", "cogsq", "xsgaq", "xintq", "seqq", "ceqq", "pstkq", "atq", "ltq", "txditcq"], False),
    "ChTax":                   (q_ChTax,                   ["txtq"], True),
    "EarningsSurprise":        (q_EarningsSurprise,        ["epspxq"], False),
    "RevenueSurprise":         (q_RevenueSurprise,         ["revtq", "cshprq"], False),
    "NumEarnIncrease":         (q_NumEarnIncrease,         ["ibq"], False),
    "roavol":                  (q_roavol,                  ["ibq", "atq"], False),
    "WW_Q":                    (q_WW_Q,                    ["ibq", "dpq", "atq", "dlttq", "saleq", "dvpsxq"], True),
}


# ===========================================================================
# RUN
# ===========================================================================

def check_requirements(m, q):
    """Verify every declared column exists BEFORE computing anything."""
    print("\nChecking inputs...")
    gaps = {}
    for name, (_, cols) in ANNUAL_FEATURES.items():
        miss = [c for c in cols if c not in m.columns]
        if miss:
            gaps[name] = miss
    for name, (_, cols, _) in QUARTERLY_FEATURES.items():
        miss = [c for c in cols if c not in q.columns]
        if miss:
            gaps[name] = miss
    if gaps:
        print("\nERROR: missing columns:")
        for k, v in gaps.items():
            print(f"  {k:<26} {v}")
        raise SystemExit("Stopping - fix the extract or the registry first.")
    n = len(ANNUAL_FEATURES) + len(QUARTERLY_FEATURES)
    print(f"  all {n} features have their inputs")


def main():
    print("stage1_construct_features_2.py  (batch 2)")
    print(f"  quarterly value stays in force for "
          f"{QUARTERLY_VALID_MONTHS} months (CZ's rule)")

    m = load_master()
    q = load_quarterly()
    check_requirements(m, q)

    out = m[KEY + ["fyear"]].copy()
    failed = {}

    print(f"\nComputing {len(ANNUAL_FEATURES)} annual features...")
    for name, (fn, _) in ANNUAL_FEATURES.items():
        try:
            out[name] = pd.array(fn(m), dtype="Float64")
        except NotImplementedError as err:
            failed[name] = f"BLOCKED: {err}"
            print(f"  BLOCKED {name}: {err}")
        except Exception as err:
            failed[name] = f"{type(err).__name__}: {err}"
            print(f"  FAILED  {name}: {type(err).__name__}: {err}")

    print(f"\nComputing {len(QUARTERLY_FEATURES)} quarterly features...")
    qvals = {}
    for name, (fn, _, needs_annual) in QUARTERLY_FEATURES.items():
        try:
            qvals[name] = pd.Series(fn(q, m) if needs_annual else fn(q)
                                    ).astype("float64").to_numpy()
        except NotImplementedError as err:
            failed[name] = f"BLOCKED: {err}"
            print(f"  BLOCKED {name}: {err}")
        except Exception as err:
            failed[name] = f"{type(err).__name__}: {err}"
            print(f"  FAILED  {name}: {type(err).__name__}: {err}")

    if qvals:
        print(f"\nAligning {len(qvals)} quarterly features to annual rows...")
        qq = q.assign(**qvals)
        aligned = quarterly_to_annual(m, qq, list(qvals))
        for name in qvals:
            out[name] = pd.array(aligned[name], dtype="Float64")

    n_firms = out.gvkey.nunique()
    print(f"\n  panel covers {n_firms:,} firms")
    assert n_firms >= MIN_FIRMS, (
        f"only {n_firms:,} firms - a listing screen may have crept in")

    names = [c for c in out.columns if c not in KEY + ["fyear"]]
    rows = []
    for c in names:
        s = out[c]
        ok = s.notna()
        rows.append(dict(feature=c, n_obs=int(ok.sum()),
                         pct_missing=round(100 * (1 - ok.mean()), 2),
                         n_firms=int(out.loc[ok, "gvkey"].nunique()),
                         first_year=int(out.loc[ok, "fyear"].min()) if ok.any() else None,
                         last_year=int(out.loc[ok, "fyear"].max()) if ok.any() else None,
                         p01=s.quantile(0.01), p50=s.quantile(0.50),
                         p99=s.quantile(0.99)))
    cov = pd.DataFrame(rows).sort_values("pct_missing")
    print("\nCoverage:")
    print(cov[["feature", "n_obs", "pct_missing", "n_firms",
               "first_year", "last_year"]].to_string(index=False))

    out.to_parquet(OUTDIR / "stage1_2_feature_panel.parquet", index=False)
    cov.to_csv(OUTDIR / "stage1_2_feature_coverage.csv", index=False)
    print(f"\n  -> stage1_2_feature_panel.parquet   {len(out):,} x {out.shape[1]}")
    print(f"  -> stage1_2_feature_coverage.csv    {len(cov)} features")
    if failed:
        print(f"\n  {len(failed)} not built:")
        for k, v in failed.items():
            print(f"    {k:<26} {v}")


if __name__ == "__main__":
    main()
