# ABOUTME: Stage 1.1 - builds the CZ characteristics on our firm-year panel of bond issuers
# ABOUTME: One function per characteristic, formulas from CZ, keys and frequency changed for us
"""
stage1_1_construct_features.py

WHAT THIS DOES:
    Computes the accounting characteristics selected in Stage 0.1 on the clean
    panel produced by Stage 1.0, and writes a firm-year feature panel plus a
    coverage report that Stage 2 uses to narrow the set.

    THIS FILE IS BATCH 1: 77 OF THE 101 CHARACTERISTICS - every one that can be
    built from the annual panel and nothing else. It is complete.

    The 101 are split into exactly two batches, and the split is by INPUT, not
    by difficulty:
        batch 1  77 features  the annual panel only    <- THIS FILE, complete
        batch 2  24 features  everything else          <- code/stage1_2_*.py

    Batch 2 is a separate script because its 24 features each need an input this
    one never opens: the quarterly panel (12 features), a multi-year rolling
    window per firm (6), or an industry code plus a year-by-industry
    cross-sectional regression (6). Keeping them apart means a failure in the
    quarterly layer cannot be confused with a failure in the annual layer.

    An earlier version of this comment described four batches. That scheme came
    from a scanner that read CZ's script text for `.shift()`, `qcut`, `.rank(`
    and `.rolling(`, and it misjudged 15 features - see docs/CHANGELOG.md. Two
    batches, defined by which file must be read, is a criterion that cannot be
    got wrong by reading a script badly.

HOW THIS RELATES TO CZ's CODE:
    The formula inside each function is CZ's, unchanged. What changes, and why,
    is always one of these four - and each function's docstring names which ones
    it needed:

      (1) KEY. CZ groups and lags by `permno`, a CRSP security identifier that
          does not exist for an unlisted firm. We use `gvkey`. Affects all.

      (2) FREQUENCY. CZ expands annual accounting data to monthly because its
          outcome is a monthly portfolio return ("All characteristics are
          computed at a monthly frequency... the monthly value is the most
          recently observed value", CZ p.9). Our outcome is a firm-year default
          indicator, so we stay annual and CZ's `l12.` becomes a one-year lag.

      (3) SIGNALMASTERTABLE. CZ merges it for identifiers, but it is built from
          monthly CRSP and screened to `shrcd in (10,11,12)` and
          `exchcd in (1,2,3)` - common stocks on NYSE/AMEX/NASDAQ. Merging it
          would silently delete every unlisted issuer. Measured cost if kept:
          179 of 1,046 issuers with a default event. The merge is deleted; this
          panel is the master.

      (4) INDUSTRY CODE. CZ uses CRSP's `sicCRSP`, unavailable for unlisted
          firms. We use `sich` where present and `comp.names.sic` otherwise.

    CZ's code is a reference, not a specification, and it is not always right.
    Where a function deviates for a reason other than the four above, the
    docstring says so explicitly.

WHAT WE ADDED, AND THE ONE THING CZ NEVER DOES:
    CZ and JKP only ever SORT firms on a characteristic - JKP into terciles,
    CZ into deciles - so the level of the number never enters an estimate and a
    ratio that explodes near a zero denominator costs them nothing. We are the
    first to use the levels, as inputs to a penalised regression. Four changes
    follow from that, and only from that.

      A. FOUR COMPANION INDICATORS, `<name>_negden`. BookLeverage, RoE, roic
         and Tax divide by a signed quantity - book equity or book income -
         that turns negative as a firm approaches default, so each has a pole
         at zero and orders the most distressed firms at the wrong end. The
         formulas are untouched; the indicator carries the side of the pole,
         which is the one thing the division destroys. It has to be a separate
         column because winsorising, standardising and ranking are all monotone
         and cannot repair an inverted ordering. PctAcc and PctTotAcc divide by
         an ABSOLUTE value, so their ordering is fine and only the magnitude is
         extreme - they get nothing.

      B. UNDISCLOSED ITEMS RESTORED TO MISSING, for eight characteristics.
         Stage 1.0's blanket zero-fill turns "not reported" into "has none".
         Applied after the formula, behind `RESTORE_UNDISCLOSED`. NOT undoable
         by a later fillna(0) - see the note on that switch.

      C. CZ's OBSERVATION-DELETING RULES ARE NOT APPLIED, in NetDebtFinance,
         NetEquityFinance, ChEQ and AssetTurnover. They delete firms that
         borrowed more than their own assets, or whose equity just crossed
         zero. Outlier control belongs in one documented winsorisation, not in
         five inconsistent per-characteristic rules. Safe because all four act
         on the finished value, so CZ's version is one mask away.

      D. TWO FABRICATED VALUES REMOVED: `pchcurrat`'s zero for "no change" and
         `Tax`'s constant 1 for loss years. Both act on the finished
         characteristic, so both are recoverable.

    The four `_negden` columns are OURS, listed in `OURS_NOT_CZ`, and must
    never be counted among the 101 characteristics.

LAGS:
    Lags are matched on FISCAL YEAR, not by row position. 221 firm-years in the
    panel sit next to a gap in the firm's fiscal-year sequence, and 26 rows
    share a gvkey and fyear; a positional `.shift(1)` would silently pull a
    non-adjacent year for those and produce a wrong growth rate. `lag()` merges
    on `fyear - k` instead, which is also what CZ does (its `stata_multi_lag`
    is calendar-based, not positional).

Inputs:
    output/compustat_annual_clean.parquet   601,250 x 117
    data_raw/us/comp_names.parquet           47,194 x 11  (for the sic fallback)

Outputs:
    output/stage1_1_feature_panel.parquet   gvkey x datadate, 81 columns:
                                            77 CZ characteristics + 4 of ours
    output/stage1_1_feature_coverage.csv    per feature: missing rate, year span,
                                            quantiles

How to run:
    python3 code/stage1_1_construct_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data_raw" / "us"
OUTDIR = PROJECT_ROOT / "output"

ANNUAL = OUTDIR / "compustat_annual_clean.parquet"
NAMES = RAW / "comp_names.parquet"

# The panel key. Never permno - see note (1) above.
KEY = ["gvkey", "datadate"]

# If the finished panel covers fewer firms than this, something has silently
# re-introduced a listing screen. Stop rather than report a halved sample.
MIN_FIRMS = 40_000


# ===========================================================================
# MASTER PANEL - our replacement for CZ's SignalMasterTable
# ===========================================================================

def load_master():
    """
    The panel every characteristic is computed on: one row per firm-year, no
    listing screen of any kind.

    Adds the industry code CZ takes from CRSP. `sich` (Compustat's historical,
    year-by-year SIC) is used where present and `comp.names.sic` (its static
    header SIC) fills the rest, which takes coverage from 64% of post-1982
    firm-years to 100%. `sic_source` records which was used for each row. The
    two agree on 91.2% of firms at the 2-digit level; CRSP's code agrees with
    `sich` only 59.2% of the time, so these features will differ materially
    from CZ's published values - unavoidably, since CRSP has no record of an
    unlisted issuer at all.
    """
    m = pd.read_parquet(ANNUAL)
    print(f"  annual panel: {len(m):,} rows x {m.shape[1]} cols, "
          f"{m.gvkey.nunique():,} firms")

    # Static header SIC, one row per gvkey, as the fallback.
    names = pd.read_parquet(NAMES, columns=["gvkey", "sic"])
    names["sic"] = names["sic"].replace("", pd.NA)
    m = m.merge(names, on="gvkey", how="left")

    # sich is numeric in funda; render it as a 4-character code to match sic.
    sich4 = m["sich"].astype("Int64").astype(str).str.zfill(4)
    sich4 = sich4.where(m["sich"].notna())
    m["ind_sic"] = sich4.fillna(m["sic"])
    m["sic_source"] = np.where(sich4.notna(), "sich",
                               np.where(m["sic"].notna(), "sic", None))
    m["sic2D"] = m["ind_sic"].str[:2]
    m["sic3D"] = m["ind_sic"].str[:3]
    print(f"  industry code: {100 * m.ind_sic.notna().mean():.1f}% covered "
          f"({dict(m.sic_source.value_counts(dropna=False))})")

    # HARD GUARD. If permno ever appears here, some CZ code has been copied
    # without stripping its CRSP dependency, and the sample is about to shrink
    # to listed firms without anyone noticing.
    assert "permno" not in m.columns, "permno in master - a listing screen has crept in"

    m = m.sort_values(KEY).reset_index(drop=True)
    return m


def lag(m, cols, years=1):
    """
    Return `m` with lagged copies of `cols` attached as `l{years}_{col}`.

    Matched on fiscal year, NOT by row position - see the LAGS note at the top.
    Where a firm has two rows for one fiscal year (26 rows in the panel), the
    one with the later datadate is used as the lag source.
    """
    if isinstance(cols, str):
        cols = [cols]
    src = m.sort_values(KEY)[["gvkey", "fyear"] + cols].copy()
    src = src.drop_duplicates(["gvkey", "fyear"], keep="last")
    src["fyear"] = src["fyear"] + years          # shift forward so it aligns to t
    src = src.rename(columns={c: f"l{years}_{c}" for c in cols})
    return m.merge(src, on=["gvkey", "fyear"], how="left")


def one_row_per_year(m, cols):
    """
    Collapse the panel to exactly one row per gvkey x fiscal year, ordered.

    Some characteristics cannot be written as a column-wise formula: they need
    the firm's years as an ordered sequence (a recursion, a rolling window, a
    firm-age counter). Those need a frame with no repeated fiscal years, which
    is what Stata's `xtset gvkey fyear` requires and would refuse to run
    without. 26 rows in our panel share a gvkey and a fyear - firms that changed
    their fiscal year end - so the later `datadate` is kept, exactly as lag()
    does, and both original rows later receive the same value.
    """
    d = m.sort_values(KEY)[["gvkey", "fyear"] + cols]
    d = d.drop_duplicates(["gvkey", "fyear"], keep="last")
    return d.sort_values(["gvkey", "fyear"]).reset_index(drop=True)


def year_lag(d, cols, years=1):
    """
    The same fiscal-year lag as lag(), for a frame already collapsed by
    one_row_per_year() and therefore having no `datadate` column.
    """
    if isinstance(cols, str):
        cols = [cols]
    src = d[["gvkey", "fyear"] + cols].copy()
    src["fyear"] = src["fyear"] + years
    src = src.rename(columns={c: f"l{years}_{c}" for c in cols})
    return d.merge(src, on=["gvkey", "fyear"], how="left")


def back_to_panel(m, d, col):
    """
    Carry a column computed on a collapsed frame back onto the full panel.

    `d` must be unique on gvkey x fyear (one_row_per_year guarantees it), so
    this cannot change the number of rows. The two rows of a firm-year that
    appears twice both receive the same value.
    """
    return m.merge(d[["gvkey", "fyear", col]], on=["gvkey", "fyear"],
                   how="left")[col].values


def safe_div(num, den, tol=1e-6):
    """
    Divide, treating a denominator smaller than `tol` as zero.

    CZ writes this out with np.where in every script and tests `den == 0`.
    Testing against exact zero is not enough here, and the reason is worth
    stating because it produced values of 10^18 in the first run.

    Most denominators are sums and differences of reported items, and decimal
    figures that cancel exactly do not cancel exactly in binary. gvkey 001084 in
    1997 reports ceq 0.562, lt 0.051, che 0.613; `ceq + lt - che` is zero in
    decimal but 1.11e-16 in floating point, so `roic` came out as 1.4e15 rather
    than missing. 798 firm-years across the panel have a denominator in
    (0, 1e-6), and the residues measured there run from 1e-19 to 1e-14.

    The threshold is not arbitrary. Compustat annual data is reported in
    millions to three decimals, so the smallest figure it can express is 0.001;
    1e-6 is a thousand times smaller than that and still eight orders of
    magnitude above the largest residue observed. Nothing real falls in the gap.

    This is NOT winsorising. It replaces arithmetic artefacts with missing,
    where the denominator is economically zero. Trimming genuine extreme values
    is a separate decision and does not belong in feature construction.
    """
    den = den.where(den.abs() > tol)
    return num / den


# ===========================================================================
# CHARACTERISTICS
#
# One function per characteristic, taking the master panel and returning a
# Series aligned to it. Each docstring gives: CZ's source file, the formula in
# CZ's own Stata notation where the script preserves it, and every deviation.
#
# NOTE: `at` (total assets) collides with the pandas `.at` accessor, so it is
# always written m["at"], never m.at.
# ===========================================================================

def f_GP(m):
    """Gross profitability, Novy-Marx 2013. CZ: GP.py.
        keep if (sic < 6000 | sic >= 7000)
        gen GP = (revt - cogs)/at
    Deviations: (1) key, (4) the industry code is our `ind_sic`.

    The financial-firm exclusion is the SOURCE PAPER's, not a listing screen, so
    it stays. Evidence: SignalDoc's `Filter` column is empty for GP and the
    restriction appears in the Detailed Definition, which ends "Drop if
    financial" - CZ files its own portfolio screens (`abs(prc)>5`, `exchcd`) in
    the `Filter` column instead. Economically, gross profit over assets measures
    operating efficiency; a bank has no cost of goods sold, so the ratio has no
    referent there. Added 2026-08-18, having been missed in the first build.

    Applied as a mask rather than a row drop, so the firm keeps its other
    characteristics."""
    gp = safe_div(m["revt"] - m["cogs"], m["at"])
    sic = pd.to_numeric(m["ind_sic"], errors="coerce")
    financial = ((sic >= 6000) & (sic < 7000)).fillna(False)
    return gp.mask(financial.values)


def f_RoE(m):
    """Return on equity. CZ: RoE.py.   RoE = ni/ceq
    Deviations: (1) key, (2) NA where ceq <= 0 (D-50): with negative equity a
    loss produces a POSITIVE return on equity, which is backwards. The boundary
    is <= 0, not < 0, following JKP ("only take the ratio if the denominator is
    above zero") and Fama-French's exclusion of negative-book-equity firms."""
    return safe_div(m["ni"], m["ceq"]).mask(m["ceq"] <= 0)


def f_PM(m):
    """Profit margin. CZ: ZZ1_PM_ChPM.py.   gen PM = ni/revt
    Deviations: (1) key only."""
    return safe_div(m["ni"], m["revt"])


def f_ChPM(m):
    """Change in profit margin. CZ: ZZ1_PM_ChPM.py.   gen ChPM = PM - l12.PM
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = m.assign(PM=f_PM(m))
    d = lag(d, "PM")
    return (d["PM"] - d["l1_PM"]).values


def f_quick(m):
    """Quick ratio. CZ: quick.py.   gen quick = (act - invt)/lct
    Deviations: (1) key only."""
    return safe_div(m["act"] - m["invt"], m["lct"])


def f_currat(m):
    """Current ratio. CZ: ZZ1_currat_pchcurrat.py.
        gen act2 = act
        replace act2 = che + rect + invt if act2 == .
        replace lct = ap if lct == .
        gen currat = act2/lct
    Deviations: (1) key only. The two fallbacks are CZ's and are kept."""
    act2 = m["act"].fillna(m["che"] + m["rect"] + m["invt"])
    lct = m["lct"].fillna(m["ap"])
    return safe_div(act2, lct)


def f_salecash(m):
    """Sales to cash. CZ: salecash.py.   gen salecash = sale/che
    Deviations: (1) key only."""
    return safe_div(m["sale"], m["che"])


def f_saleinv(m):
    """Sales to inventory. CZ: saleinv.py.   gen saleinv = sale/invt
    Deviations: (1) key only."""
    return safe_div(m["sale"], m["invt"])


def f_salerec(m):
    """Sales to receivables. CZ: salerec.py.   gen salerec = sale/rect
    Deviations: (1) key only."""
    return safe_div(m["sale"], m["rect"])


def f_roic(m):
    """Return on invested capital. CZ: roic.py.
        gen roic = (ebit - nopi)/(ceq + lt - che)
    Deviations: (1) key only.
    Note: ebit and nopi were missing from the first download - the Stage 0.1
    variable extraction only parsed pandas `columns=[...]` and this script is
    polars. Re-downloaded 2026-08-16."""
    den = m["ceq"] + m["lt"] - m["che"]
    # D-50: NA where the invested-capital denominator is not POSITIVE (<= 0).
    return safe_div(m["ebit"] - m["nopi"], den).mask(den <= 0)


def f_depr(m):
    """Depreciation to PP&E. CZ: depr.py.   depr = dp/ppent, missing if ppent is 0
    Deviations: (1) key only."""
    return safe_div(m["dp"], m["ppent"])


def f_secured(m):
    """Secured debt share. CZ: secured.py.
        gen secured = dm/(dltt+dlc)
        replace secured = 0 if dltt == . | dltt == 0 | dlc == .
    Deviations: (1) key only.
    WATCH THIS ONE. `dm` is 64.6% missing in the raw data and is on CZ's
    zero-fill list, so Stage 1.0 has already set it to 0 for those rows. The
    result is that `secured` is exactly 0 for roughly two-thirds of firm-years,
    and a 0 cannot be told apart from a non-disclosure. `dm_was_missing` in the
    panel preserves the distinction; the Stage 2 screen should look at this
    feature closely."""
    s = safe_div(m["dm"], m["dltt"] + m["dlc"])
    zero = m["dltt"].isna() | (m["dltt"] == 0) | m["dlc"].isna()
    return s.mask(zero, 0.0)


def f_securedind(m):
    """Secured debt indicator. CZ: securedind.py.
        gen securedind = 0
        replace securedind = 1 if dm != . & dm != 0
    Deviations: (1) key only. Same `dm` caveat as f_secured."""
    return ((m["dm"].notna()) & (m["dm"] != 0)).astype("Float64")


def f_XFIN(m):
    """External financing. CZ: XFIN.py.
        XFIN = (sstk - dv - prstkc + dltis - dltr + dlcch)/at, with dlcch -> 0
    Deviations: (1) key only."""
    dlcch = m["dlcch"].fillna(0)
    num = m["sstk"] - m["dv"] - m["prstkc"] + m["dltis"] - m["dltr"] + dlcch
    return safe_div(num, m["at"])


def f_Tax(m):
    """
    Taxable income to income. CZ: Tax.do.
        gen year = yofd(dofm(time_avail_m))
        gen tr = .48
        replace tr = .46 if year >= 1979 & year <= 1986
        replace tr = .4  if year == 1987
        replace tr = .34 if year >= 1988 & year <= 1992
        replace tr = .35 if year >= 1993
        gen     Tax = ((txfo + txfed)/tr)/ib
        replace Tax = ((txt - txdi)/tr)/ib          if txfo == . | txfed == .
        replace Tax = 1 if (txfo + txfed > 0 | txt > txdi) & ib <= 0

    Deviations: (1) key only. The statutory-rate schedule is CZ's, verbatim.

    What it measures: a firm keeps two sets of books, one for shareholders and
    one for the tax authority. Taxable income is not observable, so it is backed
    out of the tax expense at the statutory rate. A ratio near 1 means the two
    agree; well below 1 means book income exceeds what the firm declared for
    tax, which is a red flag for earnings quality.

    Deviations: (1) key, (3) CZ's `replace Tax = 1` line is NOT applied.

    Two corrections on 2026-08-17, both confirmed against Tax.py: the rate year
    is CZ's `time_avail_m` (our `date_avail`), not `datadate`, and CZ's last
    line applies to EVERY loss year rather than only `ib == 0`.

    CZ's last line then dropped on 2026-08-18. Setting the ratio to 1 asserts
    "no book-tax difference" for a firm that lost money on the books while
    paying tax on positive taxable income - which is the largest book-tax gap
    there is, so the assertion points the wrong way. It costs CZ nothing, since
    a constant parks those firms in the middle deciles of a sort. It costs us a
    great deal, because the fabricated value lands on exactly the rows where
    default risk is highest, and a model would read "firms about to default
    look perfectly normal on this measure".

    The denominator is book income, so this ratio has the same pole at zero as
    BookLeverage has at zero book equity. It is handled the same way: the real
    number is kept, however negative, and `Tax_negden` marks the loss years.
    CZ's published version is recovered by re-applying CZ's own condition."""
    year = m["date_avail"].dt.year
    tr = pd.Series(0.48, index=m.index)
    tr[(year >= 1979) & (year <= 1986)] = 0.46
    tr[year == 1987] = 0.40
    tr[(year >= 1988) & (year <= 1992)] = 0.34
    tr[year >= 1993] = 0.35

    tax = safe_div(safe_div(m["txfo"] + m["txfed"], tr), m["ib"])
    fallback = safe_div(safe_div(m["txt"] - m["txdi"], tr), m["ib"])
    both_present = (m["txfo"].notna() & m["txfed"].notna()).to_numpy(dtype=bool)
    # D-50: NA where pretax income is not positive. The denominator is book
    # income, so a loss makes the ratio change sign for a reason that has
    # nothing to do with book-tax differences. The loss state is carried by
    # INTWO instead.
    return tax.where(both_present, fallback).mask(m["ib"] <= 0)


def f_BookLeverage(m):
    """Book leverage. CZ: BookLeverage.py.
        tempPS = pstk, else pstkrv, else pstkl
        tempSE = seq,  else ceq + tempPS, else at - lt
        BookLeverage = at/(tempSE + txditc - tempPS)
    Deviations: (1) key only. Both fallback chains are decades of convention -
    kept exactly. There is no log here: Fama-French take Ln(A/BE) and require
    positive book equity, CZ takes the plain ratio and does not.

    ECONOMIC NOTE - READ BEFORE USING. Book equity is the denominator and it
    goes NEGATIVE for distressed firms. A firm with at = 100 and BE = -5 gets
    -20, which sorts BELOW a firm with no leverage at all, so the variable is
    not monotone in leverage exactly where leverage matters most to us. See the
    Stage 1.1 economic review."""
    temp_ps = m["pstk"].fillna(m["pstkrv"]).fillna(m["pstkl"])
    temp_se = m["seq"].fillna(m["ceq"] + temp_ps).fillna(m["at"] - m["lt"])
    den = temp_se + m["txditc"] - temp_ps
    # D-50: DENOMINATOR NOT POSITIVE -> NA. A firm with negative book equity
    # gets a large NEGATIVE ratio, which sorts BELOW an unlevered firm, so the
    # variable is not monotone in leverage exactly where leverage matters most.
    # The state itself is carried by OENEG; the ratio is undefined here and
    # says so. Boundary <= 0 per JKP and Fama-French.
    return safe_div(m["at"], den).mask(den <= 0)


def f_sgr(m):
    """Sales growth. CZ: sgr.py.   gen sgr = (sale/l12.sale) - 1
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, "sale")
    return (safe_div(d["sale"], d["l1_sale"]) - 1).values


def f_AssetGrowth(m):
    """Asset growth, Cooper-Gulen-Schill 2008. CZ: AssetGrowth.py.
        AssetGrowth = (at - l12.at)/l12.at
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, "at")
    return safe_div(d["at"] - d["l1_at"], d["l1_at"]).values


def f_cashdebt(m):
    """Cash flow to debt. CZ: cashdebt.py.
        gen cashdebt = (ib + dp)/((lt + l12.lt)/2)
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, "lt")
    return safe_div(d["ib"] + d["dp"], (d["lt"] + d["l1_lt"]) / 2).values


def f_ChNWC(m):
    """Change in net working capital. CZ: ChNWC.py.
        NWC = ((act - che) - (lct - dlc))/at, with at set missing where at <= 0
        ChNWC = NWC - l12.NWC
    Deviations: (1) key, (2) l12. -> one-year lag."""
    at_pos = m["at"].where(m["at"] > 0)
    nwc = safe_div((m["act"] - m["che"]) - (m["lct"] - m["dlc"]), at_pos)
    d = lag(m.assign(NWC=nwc), "NWC")
    return (d["NWC"] - d["l1_NWC"]).values


def f_AssetTurnover(m):
    """Asset turnover. CZ: AssetTurnover.py.
        gen temp = rect + invt + aco + ppent + intan - ap - lco - lo
        gen AssetTurnover = sale/((temp + l12.temp)/2)
        replace AssetTurnover = . if AssetTurnover < 0     <- NOT applied
    Deviations: (1) key, (2) l12. -> one-year lag, (3) CZ's negative-value drop
    is not applied. A negative value means negative net operating capital,
    which is a business model - retailers and restaurants funded by supplier
    credit - not an error. Recoverable by masking where the value is negative.
    `ChAssetTurnover` reads this function, so it recovers those rows too."""
    temp = (m["rect"] + m["invt"] + m["aco"] + m["ppent"] + m["intan"]
            - m["ap"] - m["lco"] - m["lo"])
    d = lag(m.assign(temp=temp), "temp")
    return safe_div(d["sale"], (d["temp"] + d["l1_temp"]) / 2).values


# ---------------------------------------------------------------------------
# BATCH 1, PART 2 - 40 further characteristics.
#
# Same conventions as above. Where CZ's script preserves the original Stata
# line, it is quoted verbatim in the docstring so the translation can be
# checked against it without opening the upstream repository.
# ---------------------------------------------------------------------------

def f_GPlag(m):
    """CZ: GPlag.py.   gen GPlag = (sale-cogs)/l12.at
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, "at")
    return safe_div(d["sale"] - d["cogs"], d["l1_at"]).values


def f_OPLeverage(m):
    """Operating leverage. CZ: OPLeverage.py.
        tempxsga = xsga, 0 if missing;  OPLeverage = (tempxsga + cogs)/at
    Deviations: (1) key only."""
    return safe_div(m["xsga"].fillna(0) + m["cogs"], m["at"])


def f_ShareRepurchase(m):
    """CZ: ShareRepurchase.py.   1 if prstkc > 0, else 0; missing where prstkc is
    missing. Deviations: (1) key only."""
    out = (m["prstkc"] > 0).astype("Float64")
    return out.mask(m["prstkc"].isna())


def f_ConvDebt(m):
    """Convertible debt indicator. CZ: ConvDebt.py.
        ConvDebt = 1 if dc is present and non-zero, or cshrc is present and
        non-zero; 0 otherwise.
    Deviations: (1) key only. `dc` is the convertible-debt item Stage 1.0
    rebuilds from dcpstk/pstk/dcvt, following CZ's download step."""
    has = ((m["dc"].notna()) & (m["dc"] != 0)) | \
          ((m["cshrc"].notna()) & (m["cshrc"] != 0))
    return has.astype("Float64")


def f_PctTotAcc(m):
    """Percent total accruals. CZ: PctTotAcc.py.
        PctTotAcc = (ni - (prstkcc - sstk + dvt + oancf + fincf + ivncf))/|ni|
    Deviations: (1) key only."""
    num = m["ni"] - (m["prstkcc"] - m["sstk"] + m["dvt"] + m["oancf"]
                     + m["fincf"] + m["ivncf"])
    return safe_div(num, m["ni"].abs())


def f_AssetLiquidityBook(m):
    """Asset liquidity over book assets, Ortiz-Molina & Phillips 2014.
    CZ: AssetLiquidityBook.py.
        gen AssetLiquidityBook = (che + .75*(act - che)
                                  + .5*(at - act - gdwl - intan))/l.at
    Deviations: (1) key, (2) CZ's `l.` is one MONTH on its monthly panel, which
    returns the same annual figure in 11 months of 12. An annual panel has no
    one-month lag, so this uses the prior YEAR's `at`, matching the documented
    'lagged book assets'. Values therefore differ from CZ's published ones."""
    d = lag(m, "at")
    num = (d["che"] + 0.75 * (d["act"] - d["che"])
           + 0.5 * (d["at"] - d["act"] - d["gdwl"] - d["intan"]))
    return safe_div(num, d["l1_at"]).values


def f_CapTurnover(m):
    """CZ: CapTurnover.py.   gen CapTurnover = l12.sale/l24.at
    Deviations: (1) key, (2) l12./l24. -> one- and two-year lags."""
    d = lag(lag(m, "sale", 1), "at", 2)
    return safe_div(d["l1_sale"], d["l2_at"]).values


def f_ChEQ(m):
    """Growth in book equity. CZ: ChEQ.py.
        ChEQ = ceq/l12.ceq, kept only where both years are strictly positive
    Deviations: (1) key, (2) l12. -> one-year lag.

    THE POSITIVITY RESTRICTION IS THE DOMAIN OF DEFINITION, NOT OUTLIER
    CONTROL, and is applied. A ratio reads as growth only when the denominator
    is positive and both terms share a sign. Outside that, equity of -10
    followed by -20 divides to 2.0, which looks like a doubling and describes a
    doubling of the deficit. That is not an extreme value; it is a number with
    no referent, in the same way that gross profitability has none for a bank.

    Restored 2026-08-18, reversing part of D-19. Nothing is lost by restoring
    it: the same economic event is already measured by `DelEqu`, which divides
    the change in equity by AVERAGE TOTAL ASSETS rather than by equity and is
    therefore well defined and correctly signed at any equity level, and
    `RoE_negden` records whether this year's equity is non-positive. A companion
    indicator was considered for the sign flip and rejected as redundant with
    those two."""
    d = lag(m, "ceq")
    out = safe_div(d["ceq"], d["l1_ceq"])
    positive_both = ((d["ceq"] > 0) & (d["l1_ceq"] > 0)).fillna(False)
    return out.mask(~positive_both.to_numpy(dtype=bool)).values


def f_ChInv(m):
    """Change in inventory over average assets. CZ: ChInv.py.
        ChInv = (invt - l12.invt)/((at + l12.at)/2)
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(lag(m, "invt"), "at")
    return safe_div(d["invt"] - d["l1_invt"], (d["at"] + d["l1_at"]) / 2).values


def f_ChNCOA(m):
    """Change in net non-current operating assets. CZ: ChNCOA.py.
        gen temp = at - act - ivao ; replace temp = at - act if mi(ivao)
        gen ChNCOA = (temp - l12.temp)/l12.at
    Deviations: (1) key, (2) l12. -> one-year lag."""
    temp = (m["at"] - m["act"] - m["ivao"]).fillna(m["at"] - m["act"])
    d = lag(lag(m.assign(temp=temp), "temp"), "at")
    return safe_div(d["temp"] - d["l1_temp"], d["l1_at"]).values


def f_ChNCOL(m):
    """Change in net non-current operating liabilities. CZ: ChNCOL.py.
        gen temp = lt - dlc - dltt ; replace temp = lt - dlc if mi(dltt)
        gen ChNCOL = (temp - l12.temp)/l12.at
    Deviations: (1) key, (2) l12. -> one-year lag."""
    temp = (m["lt"] - m["dlc"] - m["dltt"]).fillna(m["lt"] - m["dlc"])
    d = lag(lag(m.assign(temp=temp), "temp"), "at")
    return safe_div(d["temp"] - d["l1_temp"], d["l1_at"]).values


def f_ChNNCOA(m):
    """Change in net non-current operating assets, net of liabilities.
    CZ: ChNNCOA.py.
        temp = ((at - act - ivao) - (lt - dlc - dltt))/at
        ChNNCOA = temp - l12.temp
    Deviations: (1) key, (2) l12. -> one-year lag."""
    temp = safe_div((m["at"] - m["act"] - m["ivao"])
                    - (m["lt"] - m["dlc"] - m["dltt"]), m["at"])
    d = lag(m.assign(temp=temp), "temp")
    return (d["temp"] - d["l1_temp"]).values


def f_ChAssetTurnover(m):
    """Change in asset turnover. CZ: ChAssetTurnover.py.
        ChAssetTurnover = AssetTurnover - l12.AssetTurnover
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m.assign(AT_=f_AssetTurnover(m)), "AT_")
    return (d["AT_"] - d["l1_AT_"]).values


def f_DelCOA(m):
    """Change in current operating assets, Richardson et al. 2005. CZ: DelCOA.py.
        DelCOA = ((act - che) - (l12.act - l12.che))/(0.5*(at + l12.at))
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, ["act", "che", "at"])
    num = (d["act"] - d["che"]) - (d["l1_act"] - d["l1_che"])
    return safe_div(num, 0.5 * (d["at"] + d["l1_at"])).values


def f_DelCOL(m):
    """Change in current operating liabilities. CZ: DelCOL.py.
        DelCOL = ((lct - dlc) - (l12.lct - l12.dlc))/(0.5*(at + l12.at))
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, ["lct", "dlc", "at"])
    num = (d["lct"] - d["dlc"]) - (d["l1_lct"] - d["l1_dlc"])
    return safe_div(num, 0.5 * (d["at"] + d["l1_at"])).values


def f_DelEqu(m):
    """Change in common equity. CZ: DelEqu.py.
        DelEqu = (ceq - l12.ceq)/(0.5*(at + l12.at))
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, ["ceq", "at"])
    return safe_div(d["ceq"] - d["l1_ceq"], 0.5 * (d["at"] + d["l1_at"])).values


def f_DelLTI(m):
    """Change in long-term investment. CZ: DelLTI.py.
        DelLTI = (ivao - l12.ivao)/(0.5*(at + l12.at))
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, ["ivao", "at"])
    return safe_div(d["ivao"] - d["l1_ivao"], 0.5 * (d["at"] + d["l1_at"])).values


def f_DelSTI(m):
    """Change in short-term investment. CZ: DelSTI.py.
        gen tempAvAT = .5*(at + l12.at)
        gen DelSTI = (ivst - l12.ivst)/tempAvAT
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, ["ivst", "at"])
    return safe_div(d["ivst"] - d["l1_ivst"], 0.5 * (d["at"] + d["l1_at"])).values


def f_DelFINL(m):
    """Change in financial liabilities. CZ: DelFINL.py.
        tempPSTK = pstk, 0 if missing
        DelFINL = ((dltt + dlc + tempPSTK) - lagged same)/(0.5*(at + l12.at))
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = m.assign(tempPSTK=m["pstk"].fillna(0))
    d = lag(d, ["dltt", "dlc", "tempPSTK", "at"])
    num = ((d["dltt"] + d["dlc"] + d["tempPSTK"])
           - (d["l1_dltt"] + d["l1_dlc"] + d["l1_tempPSTK"]))
    return safe_div(num, 0.5 * (d["at"] + d["l1_at"])).values


def f_DelDRC(m):
    """Change in deferred revenue. CZ: DelDRC.py.
        DelDRC = (drc - l12.drc)/(0.5*(at + l12.at))
        then set missing where ceq <= 0, or drc and DelDRC are both 0, or
        sale < 5, or the firm is financial (6000 <= sic < 7000)
    Deviations: (1) key, (2) l12. -> one-year lag, (4) `sic` is our combined
    industry code rather than CZ's, so the financial-firm exclusion is drawn on
    a slightly different classification - see load_master()."""
    d = lag(m, ["drc", "at"])
    out = safe_div(d["drc"] - d["l1_drc"], 0.5 * (d["at"] + d["l1_at"]))
    sic_num = pd.to_numeric(m["ind_sic"], errors="coerce")
    drop = ((m["ceq"] <= 0)
            | ((m["drc"] == 0) & (out == 0))
            | (m["sale"] < 5)
            | ((sic_num >= 6000) & (sic_num < 7000)))
    return out.mask(drop.values).values


def f_NOA(m):
    """Net operating assets, Hirshleifer et al. 2004. CZ: NOA.py.
        OA  = at - che
        OL  = at - dltt - mib - dc - ceq
        NOA = (OA - OL)/l12.at
    Deviations: (1) key, (2) l12. -> one-year lag."""
    oa = m["at"] - m["che"]
    ol = m["at"] - m["dltt"] - m["mib"] - m["dc"] - m["ceq"]
    d = lag(m.assign(NOA_num=oa - ol), "at")
    return safe_div(d["NOA_num"], d["l1_at"]).values


def f_NetDebtFinance(m):
    """CZ: NetDebtFinance.py.
        NetDebtFinance = (dltis - dltr + dlcch)/(0.5*(at + l12.at)), dlcch -> 0
        replace NetDebtFinance = . if abs(...) > 1      <- NOT applied

    Deviations: (1) key, (2) l12. -> one-year lag, (3) CZ's cap dropped.

    The cap deletes firms whose net borrowing in a year exceeded their own
    average total assets - a leveraged buyout, or a firm borrowing to stay
    alive. For CZ that is an outlier distorting a portfolio mean. For us it is
    the observation the model most needs, since our outcome is default.
    Recoverable in one line: `NetDebtFinance.mask(abs() > 1)`. Outlier control
    now rests on the single documented 1%/99% winsorisation, not on five
    different per-characteristic rules."""
    d = m.assign(dlcch0=m["dlcch"].fillna(0))
    d = lag(d, ["at"])
    return safe_div(d["dltis"] - d["dltr"] + d["dlcch0"],
                    0.5 * (d["at"] + d["l1_at"])).values


def f_NetEquityFinance(m):
    """CZ: NetEquityFinance.py.
        NetEquityFinance = (sstk - prstkc - dv)/(0.5*(at + l12.at))
        replace NetEquityFinance = . if abs(...) > 1     <- NOT applied
    Deviations: (1) key, (2) l12. -> one-year lag, (3) CZ's cap dropped, for
    the same reason as NetDebtFinance. Recoverable by masking on abs() > 1."""
    d = lag(m, ["at"])
    return safe_div(d["sstk"] - d["prstkc"] - d["dv"],
                    0.5 * (d["at"] + d["l1_at"])).values


def f_CompositeDebtIssuance(m):
    """CZ: CompositeDebtIssuance.py.
        tempBD = dltt + dlc
        CompositeDebtIssuance = log(tempBD/l60.tempBD)
    Deviations: (1) key, (2) l60. -> five-year lag."""
    d = lag(m.assign(tempBD=m["dltt"] + m["dlc"]), "tempBD", 5)
    ratio = safe_div(d["tempBD"], d["l5_tempBD"])
    return np.log(ratio.where(ratio > 0).astype("float64")).values


def f_InvestPPEInv(m):
    """Investment in PP&E and inventory. CZ: InvestPPEInv.py.
        InvestPPEInv = ((ppegt - l12.ppegt) + (invt - l12.invt))/l12.at
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, ["ppegt", "invt", "at"])
    return safe_div((d["ppegt"] - d["l1_ppegt"]) + (d["invt"] - d["l1_invt"]),
                    d["l1_at"]).values


def f_LaborforceEfficiency(m):
    """CZ: LaborforceEfficiency.py.
        gen temp = sale/emp
        gen LaborforceEfficiency = (temp - l12.temp)/l12.temp
    Deviations: (1) key, (2) l12. -> one-year lag.
    Note: `emp` was one of the 24 variables missing from the first download."""
    temp = safe_div(m["sale"], m["emp"])
    d = lag(m.assign(temp=temp), "temp")
    return safe_div(d["temp"] - d["l1_temp"], d["l1_temp"]).values


def f_OrderBacklog(m):
    """CZ: OrderBacklog.py.
        OrderBacklog = ob/(0.5*(at + l12.at)), missing where ob == 0
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, "at")
    out = safe_div(d["ob"], 0.5 * (d["at"] + d["l1_at"]))
    return out.mask(d["ob"] == 0).values


def f_OrderBacklogChg(m):
    """CZ: OrderBacklogChg.py.
        OrderBacklogChg = OrderBacklog - l12.OrderBacklog
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m.assign(OB_=f_OrderBacklog(m)), "OB_")
    return (d["OB_"] - d["l1_OB_"]).values


def f_RetNOA(m):
    """Return on net operating assets. CZ: RetNOA.py.
        gen tempOA = at - che - ivao ; replace tempOA = at - che if mi(ivao)
        temp<v> = <v> with 0 for missing, for dlc, dltt, mib, pstk
        gen tempOL = at - tempdlc - tempdltt - tempmib - temppstk - ceq
        gen RetNOA = l12.oiadp/(l24.tempOA - l24.tempOL)
    Deviations: (1) key, (2) l12./l24. -> one- and two-year lags.
    NOTE: `SignalDoc.csv`'s Detailed Definition for RetNOA is wrong - it repeats
    PM's text ('Net income over revenue'). The code is correct and is followed
    here."""
    temp_oa = (m["at"] - m["che"] - m["ivao"]).fillna(m["at"] - m["che"])
    temp_ol = (m["at"] - m["dlc"].fillna(0) - m["dltt"].fillna(0)
               - m["mib"].fillna(0) - m["pstk"].fillna(0) - m["ceq"])
    d = m.assign(tempOA=temp_oa, tempOL=temp_ol)
    d = lag(lag(d, "oiadp", 1), ["tempOA", "tempOL"], 2)
    return safe_div(d["l1_oiadp"], d["l2_tempOA"] - d["l2_tempOL"]).values


def f_SurpriseRD(m):
    """R&D surprise indicator. CZ: SurpriseRD.py.
        1 when xrd/revt > 0 and xrd/at > 0 and xrd/l12.xrd > 1.05 and
          (xrd/at)/(l12.xrd/l12.at) > 1.05; else 0 where xrd and l12.xrd exist.
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, ["xrd", "at"])
    both = d["xrd"].notna() & d["l1_xrd"].notna()
    hit = (safe_div(d["xrd"], d["revt"]) > 0) & (safe_div(d["xrd"], d["at"]) > 0) \
        & (safe_div(d["xrd"], d["l1_xrd"]) > 1.05) \
        & (safe_div(safe_div(d["xrd"], d["at"]),
                    safe_div(d["l1_xrd"], d["l1_at"])) > 1.05)
    out = pd.Series(pd.NA, index=d.index, dtype="Float64")
    out[both] = 0.0
    out[both & hit.fillna(False)] = 1.0
    return out.values


def f_rd_sale(m):
    """CZ: rd_sale.py.
        gen rd_sale = l12.xrd/l12.sale ; replace rd_sale = . if rd_sale == 0
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m, ["xrd", "sale"])
    out = safe_div(d["l1_xrd"], d["l1_sale"])
    return out.mask(out == 0).values


def f_Accruals(m):
    """Working-capital accruals, Sloan 1996. CZ: Accruals.py.
        tempTXP = txp, 0 if missing
        Accruals = ((act - l12.act) - (che - l12.che)
                    - ((lct - l12.lct) - (dlc - l12.dlc) - (tempTXP - l12.tempTXP))
                    - dp) / ((at + l12.at)/2)
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = m.assign(tempTXP=m["txp"].fillna(0))
    d = lag(d, ["act", "che", "lct", "dlc", "tempTXP", "at"])
    num = ((d["act"] - d["l1_act"]) - (d["che"] - d["l1_che"])
           - ((d["lct"] - d["l1_lct"]) - (d["dlc"] - d["l1_dlc"])
              - (d["tempTXP"] - d["l1_tempTXP"]))
           - d["dp"])
    return safe_div(num, (d["at"] + d["l1_at"]) / 2).values


def f_pchcurrat(m):
    """Change in current ratio. CZ: ZZ1_currat_pchcurrat.py.
        gen pchcurrat = (currat - l12.currat)/l12.currat
        replace pchcurrat = 0 if pchcurrat == .    <- NOT applied, see below

    Deviations: (1) key, (2) l12. -> one-year lag, (3) CZ's final line dropped.

    CZ's last line sets every value it could not compute to 0, which asserts
    that liquidity did not change in cases where it is unknown - roughly a
    third of firm-years. That is fabrication rather than a missing-value rule,
    and it conflicts with the standing rule that missing stays missing. Dropped
    on 2026-08-18.

    Nothing is lost by dropping it. CZ's fill acts on the FINISHED
    characteristic, not on an input item, so CZ's published version is exactly
    `pchcurrat.fillna(0)` and can be recovered at any time. That is what makes
    this case different from the zero-filled items, where the fill happens
    before the arithmetic and cannot be undone afterwards."""
    d = lag(m.assign(currat_=f_currat(m)), "currat_")
    return safe_div(d["currat_"] - d["l1_currat_"], d["l1_currat_"]).values


def f_pchquick(m):
    """Change in quick ratio. CZ: pchquick.py.
        gen pchquick = ((act-invt)/lct - (l12.act-l12.invt)/l12.lct)
                       / ((l12.act-l12.invt)/l12.lct)
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m.assign(quick_=f_quick(m)), "quick_")
    return safe_div(d["quick_"] - d["l1_quick_"], d["l1_quick_"]).values


def f_pchsaleinv(m):
    """Change in sales-to-inventory. CZ: pchsaleinv.py.
        gen pchsaleinv = ((sale/invt)-(l12.sale/l12.invt))/(l12.sale/l12.invt)
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m.assign(si_=f_saleinv(m)), "si_")
    return safe_div(d["si_"] - d["l1_si_"], d["l1_si_"]).values


def f_pchdepr(m):
    """Change in depreciation rate. CZ: pchdepr.py.
        gen pchdepr = ((dp/ppent)-(l12.dp/l12.ppent))/(l12.dp/l12.ppent)
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m.assign(dep_=f_depr(m)), "dep_")
    return safe_div(d["dep_"] - d["l1_dep_"], d["l1_dep_"]).values


def f_pchgm_pchsale(m):
    """Gross-margin growth minus sales growth. CZ: pchgm_pchsale.py.
        gen pchgm_pchsale = (((sale-cogs)-(l12.sale-l12.cogs))/(l12.sale-l12.cogs))
                            - ((sale-l12.sale)/l12.sale)
    Deviations: (1) key, (2) l12. -> one-year lag."""
    d = lag(m.assign(gm_=m["sale"] - m["cogs"]), ["gm_", "sale"])
    return (safe_div(d["gm_"] - d["l1_gm_"], d["l1_gm_"])
            - safe_div(d["sale"] - d["l1_sale"], d["l1_sale"])).values


def f_GrGMToGrSales(m):
    """Gross-margin growth relative to sales growth, two-year base.
    CZ: GrGMToGrSales.py.
        gen tempGM = sale-cogs
        GrGMToGrSales = (tempGM - .5*(l12.tempGM+l24.tempGM))/(.5*(l12.tempGM+l24.tempGM))
                        - (sale  - .5*(l12.sale  +l24.sale  ))/(.5*(l12.sale  +l24.sale  ))
        replace with the one-year version where the above is missing
    Deviations: (1) key, (2) l12./l24. -> one- and two-year lags."""
    d = m.assign(gm_=m["sale"] - m["cogs"])
    d = lag(lag(d, ["gm_", "sale"], 1), ["gm_", "sale"], 2)
    base_gm = 0.5 * (d["l1_gm_"] + d["l2_gm_"])
    base_sa = 0.5 * (d["l1_sale"] + d["l2_sale"])
    out = (safe_div(d["gm_"] - base_gm, base_gm)
           - safe_div(d["sale"] - base_sa, base_sa))
    alt = (safe_div(d["gm_"] - d["l1_gm_"], d["l1_gm_"])
           - safe_div(d["sale"] - d["l1_sale"], d["l1_sale"]))
    return out.fillna(alt).values


def f_GrSaleToGrReceivables(m):
    """Sales growth relative to receivables growth. CZ: GrSaleToGrReceivables.py.
    Same two-year-base structure as GrGMToGrSales, with rect in place of gross
    margin, and the same one-year fallback.
    Deviations: (1) key, (2) l12./l24. -> one- and two-year lags."""
    d = lag(lag(m, ["rect", "sale"], 1), ["rect", "sale"], 2)
    base_sa = 0.5 * (d["l1_sale"] + d["l2_sale"])
    base_re = 0.5 * (d["l1_rect"] + d["l2_rect"])
    out = (safe_div(d["sale"] - base_sa, base_sa)
           - safe_div(d["rect"] - base_re, base_re))
    alt = (safe_div(d["sale"] - d["l1_sale"], d["l1_sale"])
           - safe_div(d["rect"] - d["l1_rect"], d["l1_rect"]))
    return out.fillna(alt).values


def f_GrSaleToGrInv(m):
    """Sales growth relative to inventory growth. CZ: GrSaleToGrInv.py.
    Same structure, with invt.
    Deviations: (1) key, (2) l12./l24. -> one- and two-year lags."""
    d = lag(lag(m, ["invt", "sale"], 1), ["invt", "sale"], 2)
    base_sa = 0.5 * (d["l1_sale"] + d["l2_sale"])
    base_in = 0.5 * (d["l1_invt"] + d["l2_invt"])
    out = (safe_div(d["sale"] - base_sa, base_sa)
           - safe_div(d["invt"] - base_in, base_in))
    alt = (safe_div(d["sale"] - d["l1_sale"], d["l1_sale"])
           - safe_div(d["invt"] - d["l1_invt"], d["l1_invt"]))
    return out.fillna(alt).values


def f_GrSaleToGrOverhead(m):
    """Sales growth relative to SG&A growth. CZ: GrSaleToGrOverhead.py.
    Same structure, with xsga.
    Deviations: (1) key, (2) l12./l24. -> one- and two-year lags."""
    d = lag(lag(m, ["xsga", "sale"], 1), ["xsga", "sale"], 2)
    base_sa = 0.5 * (d["l1_sale"] + d["l2_sale"])
    base_xs = 0.5 * (d["l1_xsga"] + d["l2_xsga"])
    out = (safe_div(d["sale"] - base_sa, base_sa)
           - safe_div(d["xsga"] - base_xs, base_xs))
    alt = (safe_div(d["sale"] - d["l1_sale"], d["l1_sale"])
           - safe_div(d["xsga"] - d["l1_xsga"], d["l1_xsga"]))
    return out.fillna(alt).values


# ---------------------------------------------------------------------------
# BATCH 1, PART 3 - the last 15 characteristics of batch 1.
#
# These were put in batch 1 by a scanner that looked for `.shift()`, `qcut`,
# `.rank(` and `.rolling(` in CZ's script text. All 15 need something that
# scanner could not see: a recursion, a rolling regression, a cross-sectional
# rank done through a merge, a year-dependent branch, or a firm-age counter.
# They are written out one at a time rather than bulk-translated.
#
# For these 15 the PYTHON in CZ's repo was checked against CZ's own LEGACY
# STATA (`Signals/LegacyStataCode/`), and the Stata is followed where the two
# disagree, because the Python files are translations of it. Every place they
# disagree is named in the docstring below - there are several, and two of them
# change the values materially.
# ---------------------------------------------------------------------------

def brand_capital(m):
    """
    CZ's perpetual-inventory brand capital, shared by BrandCapital and
    BrandInvest. CZ: BrandCapital.do / BrandInvest.do (identical block).

        gen byte OK = !missing(xad)
        bys gvkey OK (fyear): gen BrandCapital = xad/(.5 + .1) if OK==1 & _n==1
        bys gvkey OK (fyear): gen tempYear     = fyear         if OK==1 & _n==1
        egen FirstNMyear = min(tempYear), by(gvkey)
        gen tempxad = xad ; replace tempxad = 0 if mi(xad)
        replace BrandCapital = 0 if mi(BrandCapital)
        xtset gvkey fyear
        by gvkey: replace BrandCapital = (1-.5)*l.BrandCapital + tempxad if _n > 1
        replace BrandCapital = . if mi(FirstNMyear) | fyear < FirstNMyear
        replace BrandCapital = . if mi(xad)
        replace BrandCapital = BrandCapital/at

    Returns a frame of gvkey, fyear, BrandCapital.

    Deviations: (1) key only - CZ already runs this on gvkey and fyear.

    TWO THINGS TO KNOW ABOUT WHAT THIS FORMULA ACTUALLY DOES. Neither is a
    choice of ours; both are what CZ's Stata computes, and both are reproduced
    here deliberately rather than tidied up.

      (a) THE SEED IS USUALLY THROWN AWAY. The line that sets brand capital to
          `xad/0.6` in the firm's first year with advertising data runs BEFORE
          the recursion, and the recursion then overwrites every row except the
          firm's very first. So the seed only survives when the first year with
          advertising data is also the firm's first year in the panel; in every
          other case that year's brand capital comes out as plain `xad`
          instead of `xad/0.6`. CZ's two Python translations both differ from
          the Stata here, and from each other.

      (b) A GAP IN THE FISCAL-YEAR SEQUENCE ENDS THE SERIES. `l.BrandCapital`
          under `xtset gvkey fyear` is missing when the firm has no row for the
          previous fiscal year, so brand capital becomes missing that year - and
          because the next year reads this year's value, it stays missing for
          the rest of the firm's life. The recursion below reproduces that.

    Note `xad` is only 24.8% populated in our panel, so both features are sparse
    by construction; that is a matter for the Stage 2 screen.
    """
    d = one_row_per_year(m, ["xad", "at"])
    xad = d["xad"].astype("float64")
    tempxad = xad.fillna(0.0).to_numpy()

    # FirstNMyear: the firm's first fiscal year with a non-missing xad.
    first = (d.loc[xad.notna()].groupby("gvkey")["fyear"].min()
             .rename("FirstNMyear"))
    d = d.merge(first, on="gvkey", how="left")

    # Seed that year at xad/0.6; every other row starts at 0, as CZ does.
    seed = xad.notna() & (d["fyear"] == d["FirstNMyear"])
    bc = np.where(seed.fillna(False), xad / (0.5 + 0.1), 0.0).astype("float64")

    # The recursion. Sequential by necessity - each year reads the year before.
    firm = pd.factorize(d["gvkey"])[0]
    fy = d["fyear"].to_numpy()
    for i in range(1, len(d)):
        if firm[i] != firm[i - 1]:
            continue                      # firm's first row: Stata's `_n > 1`
        if fy[i] - fy[i - 1] != 1:
            bc[i] = np.nan                # no row last year -> l. is missing
        else:
            bc[i] = 0.5 * bc[i - 1] + tempxad[i]

    d["BrandCapital"] = bc
    blank = (d["FirstNMyear"].isna() | (d["fyear"] < d["FirstNMyear"])
             | xad.isna())
    d.loc[blank.fillna(True), "BrandCapital"] = np.nan
    d["BrandCapital"] = safe_div(d["BrandCapital"], d["at"].astype("float64"))
    return d[["gvkey", "fyear", "BrandCapital"]]


def f_BrandCapital(m):
    """Brand capital, Belo-Lin-Vitorino 2014. CZ: BrandCapital.do (Placebo).
    See brand_capital() for the formula and its two quirks.
    Deviations: (1) key only."""
    return back_to_panel(m, brand_capital(m), "BrandCapital")


def f_BrandInvest(m):
    """
    Brand investment rate, Belo-Lin-Vitorino 2014 Table 1A r^S.
    CZ: BrandInvest.do.
        gen BrandInvest = xad0/l.BrandCapital
        destring sic, replace
        drop if sic >= 4900 & sic <= 4999
        drop if sic >= 6000 & sic <= 6999
        keep if month(datadate) == 12

    Deviations: (1) key, (2) l. -> one-year lag, (4) industry code is our
    `ind_sic` rather than CZ's, (2) again for the December filter - see below.

    THE DECEMBER FILTER IS DROPPED. `keep if month(datadate) == 12` exists
    because the next lines expand each annual row into 12 monthly rows, and
    restricting to December fiscal year ends is how CZ stops one firm's rows
    from overlapping in the monthly panel. We have no monthly panel. Keeping it
    would delete every firm-year of every firm whose fiscal year does not end in
    December - roughly a third of the panel - for a reason that does not exist
    here. Recorded as a deviation; CZ's own BrandCapital.do does not have it.

    The industry filter IS kept: it comes from the paper (OP page 4), not from
    the monthly layer. CZ drops those rows; we cannot drop rows from a shared
    panel, so the feature is set missing there instead, which is equivalent.

    Note the ratio's denominator is brand capital ALREADY divided by total
    assets, so BrandInvest is not the dimensionless rate its name suggests.
    That is what SignalDoc documents ('Brand capital is scaled by total assets')
    as well as what the code does, so it is kept."""
    d = m.merge(brand_capital(m), on=["gvkey", "fyear"], how="left")
    d = lag(d, "BrandCapital")
    out = safe_div(d["xad0"].astype("float64"),
                   d["l1_BrandCapital"].astype("float64"))
    sic = pd.to_numeric(m["ind_sic"], errors="coerce")
    excluded = (((sic >= 4900) & (sic <= 4999))
                | ((sic >= 6000) & (sic <= 6999))).fillna(False)
    return out.mask(excluded.values).values


def f_EarningsConsistency(m):
    """
    Earnings consistency, Alwathainani 2009 Table 11A. CZ: EarningsConsistency.do.
        gen temp = (epspx - l12.epspx)/(.5*(abs(l12.epspx) + abs(l24.epspx)))
        foreach n of numlist 12(12)48 { gen temp`n' = l`n'.temp }
        egen EarningsConsistency = rowmean(temp*)
        replace EarningsConsistency = . if abs(epspx/l12.epspx) > 6 | ///
            (temp > 0 & l12.temp < 0 & !mi(temp)) | (temp < 0 & l12.temp > 0 & !mi(temp))

    Deviations: (1) key, (2) l12...l48 -> one- to four-year lags.

    TWO PIECES OF STATA BEHAVIOUR THAT DO NOT SURVIVE A LITERAL TRANSLATION,
    and that CZ's own Python only half handles:

      - `abs(epspx/l12.epspx) > 6` is TRUE in Stata whenever the ratio cannot be
        computed, because Stata sorts missing above every number. So a missing
        current or prior EPS, or a prior EPS of zero, is an exclusion. In pandas
        the same comparison is False. Written out below as `ratio.isna()`.
      - `l12.temp > 0` in the third clause is likewise TRUE when last year's
        growth is missing. Written out below as `| l1_eg.isna()`.

    Both make the feature stricter, not looser; missing them would let through
    rows CZ excludes."""
    d = lag(lag(m, "epspx", 1), "epspx", 2)
    eg = safe_div(d["epspx"] - d["l1_epspx"],
                  0.5 * (d["l1_epspx"].abs() + d["l2_epspx"].abs()))
    d = d.assign(eg=eg.astype("float64"))
    for k in (1, 2, 3, 4):
        d = lag(d, "eg", k)

    # egen rowmean averages over whichever of the five are present.
    out = d[["eg", "l1_eg", "l2_eg", "l3_eg", "l4_eg"]].mean(axis=1, skipna=True)

    ratio = safe_div(d["epspx"], d["l1_epspx"])
    excluded = (ratio.isna() | (ratio.abs() > 6)
                | ((d["eg"] > 0) & (d["l1_eg"] < 0))
                | ((d["eg"] < 0) & ((d["l1_eg"] > 0) | d["l1_eg"].isna())))
    return out.mask(excluded.fillna(True).to_numpy(dtype=bool)).values


def earnings_regression(m):
    """
    The rolling AR(1) earnings regression behind EarningsPersistence and
    EarningsPredictability. CZ: ZZ1_EarningsPersistence_EarningsPredictability.do.
        xtset gvkey fyear
        gen temp = epspx/ajex
        gen tempLag = l.temp
        asreg temp tempLag, window(fyear 10) min(10) by(gvkey) fitted rmse
        rename _b_tempLag EarningsPersistence
        gen EarningsPredictability = _rmse^2

    Returns a frame of gvkey, fyear, EarningsPersistence, EarningsPredictability.

    Deviations: (1) key only. CZ already runs this annually on gvkey and fyear,
    so the ten-year window carries over unchanged.

    The window is ten FISCAL YEARS wide and needs ten usable observations, so
    every one of those ten years must be present with both the current and the
    prior year's adjusted EPS. The panel is therefore laid out on a complete
    year grid per firm first, so that a firm missing 1994 does not silently get
    a window reaching back eleven years.

    The regression is not fitted by least-squares software; with one regressor
    the closed form from rolling sums is exact and is what makes this run in
    seconds rather than hours over 47,187 firms. Its one weakness is that sums
    of squares lose precision when the values are enormous, which for an EPS
    figure they are not.

    WHERE THIS DIFFERS FROM CZ's PYTHON: CZ's Python divides the squared
    residuals by N. Stata's `_rmse`, which the .do file asks for, divides by
    N - k with k = 2 for a slope and an intercept. The Stata is followed, so
    EarningsPredictability = RSS/8 rather than RSS/10 - CZ's Python is 20% low.

    WHERE THIS DIFFERS FROM SignalDoc: SignalDoc says EarningsPredictability is
    "the R2 of this regression". The code computes the squared root-mean-square
    error, which is the residual variance - not R2, and it moves the opposite
    way (large means LESS predictable). The two cannot both be right. The code
    is followed here, because an error-variance measure is what the underlying
    earnings-quality literature calls predictability, whereas R2 is not
    something `_rmse` could ever produce by accident. FLAGGED, NOT SILENTLY
    FIXED: I could not verify the original paper's definition from anything
    inside this project folder - the paper is not in `papers/` - so treat the
    justification above as my reading, not as a checked fact. The choice only
    affects EarningsPredictability; EarningsPersistence is a slope coefficient
    in both readings."""
    N = 10
    d = one_row_per_year(m, ["epspx", "ajex"])
    d["y"] = safe_div(d["epspx"].astype("float64"), d["ajex"].astype("float64"))
    d = year_lag(d, "y", 1).rename(columns={"l1_y": "x"})

    # Complete fiscal-year grid per firm, so "10 years back" means 10 years.
    span = d.groupby("gvkey")["fyear"].agg(["min", "max"])
    width = (span["max"] - span["min"] + 1).astype(int)
    grid = pd.DataFrame({
        "gvkey": np.repeat(span.index.to_numpy(), width.to_numpy()),
        "fyear": np.concatenate([np.arange(a, b + 1)
                                 for a, b in zip(span["min"], span["max"])]),
    })
    g = grid.merge(d[["gvkey", "fyear", "x", "y"]], on=["gvkey", "fyear"],
                   how="left")

    # Both sides must be present, so all five rolling sums cover the same rows.
    usable = g["x"].notna() & g["y"].notna()
    x = g["x"].where(usable)
    y = g["y"].where(usable)

    # min_periods counts non-missing values, so a window with even one hole
    # gives a missing sum - which is exactly asreg's min(10) on a 10-year window.
    def rsum(s):
        return s.rolling(N, min_periods=N).sum()

    sx, sy = rsum(x), rsum(y)
    sxx, sxy, syy = rsum(x * x), rsum(x * y), rsum(y * y)

    # The grid is sorted and each firm's rows are contiguous, so a window
    # belongs to one firm exactly when the row has 9 earlier rows of its own.
    within_firm = grid.groupby("gvkey").cumcount() >= N - 1

    cxx = sxx - sx ** 2 / N
    cxy = sxy - sx * sy / N
    cyy = syy - sy ** 2 / N

    # A window whose regressor never moves has no slope; left missing and counted.
    beta = cxy / cxx.where(cxx.abs() > 1e-10)
    rss = (cyy - beta * cxy).clip(lower=0)     # clip: float error can go < 0

    g["EarningsPersistence"] = beta.where(within_firm)
    g["EarningsPredictability"] = (rss / (N - 2)).where(within_firm)
    return g[["gvkey", "fyear", "EarningsPersistence", "EarningsPredictability"]]


def f_EarningsPersistence(m):
    """Slope of adjusted EPS on its own prior year, ten-year rolling window.
    See earnings_regression(). Deviations: (1) key only."""
    return back_to_panel(m, earnings_regression(m), "EarningsPersistence")


def f_EarningsPredictability(m):
    """Residual variance of that same regression. See earnings_regression(),
    including the conflict with SignalDoc, which calls this an R2.
    Deviations: (1) key only."""
    return back_to_panel(m, earnings_regression(m), "EarningsPredictability")


def f_GrLTNOA(m):
    """
    Growth in long-term net operating assets, Fairfield-Whisenant-Yohn 2003.
    CZ: GrLTNOA.do.
        gen GrLTNOA = (rect + invt + ppent + aco + intan + ao - ap - lco - lo)/at
            - (l12.rect + l12.invt + l12.ppent + l12.aco + l12.intan + l12.ao
               - l12.ap - l12.lco - l12.lo)/l12.at
            - (rect - l12.rect + invt - l12.invt + aco - l12.aco
               - (ap - l12.ap + lco - l12.lco) - dp)/((at + l12.at)/2)
    Deviations: (1) key, (2) l12. -> one-year lag."""
    ltnoa = (m["rect"] + m["invt"] + m["ppent"] + m["aco"] + m["intan"]
             + m["ao"] - m["ap"] - m["lco"] - m["lo"])
    d = lag(m.assign(ltnoa_=ltnoa),
            ["ltnoa_", "at", "rect", "invt", "aco", "ap", "lco"])
    accrual = ((d["rect"] - d["l1_rect"]) + (d["invt"] - d["l1_invt"])
               + (d["aco"] - d["l1_aco"])
               - ((d["ap"] - d["l1_ap"]) + (d["lco"] - d["l1_lco"]))
               - d["dp"])
    return (safe_div(d["ltnoa_"], d["at"])
            - safe_div(d["l1_ltnoa_"], d["l1_at"])
            - safe_div(accrual, (d["at"] + d["l1_at"]) / 2)).values


def f_MeanRankRevGrowth(m):
    """
    Weighted average revenue-growth rank, Lakonishok-Shleifer-Vishny 1994.
    CZ: MeanRankRevGrowth.do.
        gen temp = log(revt) - log(l12.revt)
        gsort time_avail_m -temp
        by time_avail_m: gen tempRank = _n if temp != .
        gen MeanRankRevGrowth = (5*l12.tempRank + 4*l24.tempRank + 3*l36.tempRank
                                 + 2*l48.tempRank + l60.tempRank)/15

    Deviations: (1) key, (2) the cross-section CZ ranks within is a calendar
    month; ours is a fiscal year, and l12...l60 become one- to five-year lags.

    Stata's log of a zero or negative number is missing, so firms with
    non-positive revenue in either year are unranked; written out explicitly.
    Rank 1 is the fastest grower (`gsort -temp`, with Stata placing missing last).

    SignalDoc adds "Exclude NASDAQ stocks" to the definition. The Stata code
    contains no such filter, and we have no exchange code and would not use one
    - see decision D-13 on listing status in docs/DECISIONS.md. Not applied;
    recorded as a known difference from CZ's published values.

    READ THIS BEFORE USING THE FEATURE. The rank is a raw position, 1 to N,
    where N is the number of firms with computable growth THAT YEAR. N runs from
    a few hundred in the early 1950s to about ten thousand in the 2000s, so a
    rank of 500 means "near the top" in one year and "top 5%" in another, and
    the feature is not comparable across years. That costs CZ nothing, because
    CZ only ever sorts on it WITHIN one month to form portfolios, and any
    monotone transform gives the same portfolios. It costs us a great deal,
    because our model pools firm-years across seventy years and reads the level.
    Dividing by N within the year would fix it and is a one-line change. That is
    a change to the feature definition, so it is NOT made here - it needs your
    decision. Left as CZ computes it and flagged for Stage 2."""
    d = one_row_per_year(m, ["revt"])
    d = year_lag(d, "revt", 1)
    revt = d["revt"].astype("float64")
    prior = d["l1_revt"].astype("float64")

    positive = ((revt > 0) & (prior > 0)).fillna(False).to_numpy()
    growth = pd.Series(np.nan, index=d.index)
    growth[positive] = np.log(revt[positive]) - np.log(prior[positive])

    # Cross-sectional rank within the fiscal year; 1 = fastest growth.
    d["tempRank"] = growth.groupby(d["fyear"]).rank(ascending=False,
                                                    method="first")
    for k in (1, 2, 3, 4, 5):
        d = year_lag(d, "tempRank", k)
    d["MeanRankRevGrowth"] = (5 * d["l1_tempRank"] + 4 * d["l2_tempRank"]
                              + 3 * d["l3_tempRank"] + 2 * d["l4_tempRank"]
                              + d["l5_tempRank"]) / 15
    return back_to_panel(m, d, "MeanRankRevGrowth")


def f_OperProfRDLagAT(m):
    """
    Operating profits including R&D, over lagged assets. CZ: OperProfRDLagAT.do.
        gen tempXRD = xrd ; replace tempXRD = 0 if mi(tempXRD)
        gen OperProfRDLagAT = (revt - cogs - xsga + tempXRD)/l12.at
    Deviations: (1) key, (2) l12. -> one-year lag.

    Note CZ uses raw `xsga`, not the zero-filled `xsga0`, so a firm that does
    not report SG&A gets no value - about a third of the panel. That is CZ's
    formula and it is kept. CZ's Python additionally forward-fills every input
    within the firm before computing; that is patching holes in its own monthly
    expansion and has no place on an annual panel, so it is not carried over.

    FINANCIAL FIRMS EXCLUDED, added 2026-08-18. SignalDoc's exclusion clause for
    this characteristic is MIXED - "Exclude if share code is greater 11, market
    value of equity, BM or total assets are missing, or if SIC code between 6000
    and 6999." The share-code and market-value halves are listing screens and go;
    the SIC half is a population restriction and stays. The first build stripped
    the whole clause because it began with market variables, which is the lesson:
    an exclusion clause can mix the two kinds and cannot be handled by keyword."""
    d = lag(m, "at")
    out = safe_div(d["revt"] - d["cogs"] - d["xsga"] + d["xrd"].fillna(0),
                   d["l1_at"])
    sic = pd.to_numeric(m["ind_sic"], errors="coerce")
    financial = ((sic >= 6000) & (sic < 7000)).fillna(False)
    return out.mask(financial.values).values


def f_PctAcc(m):
    """
    Percent operating accruals, Hafzalla-Lundholm-Van Winkle 2011. CZ: PctAcc.do.
        gen PctAcc = (ib - oancf)/abs(ib)
        replace PctAcc = (ib - oancf)/.01 if ib == 0
        replace PctAcc = ((act-l12.act) - (che-l12.che)
                          - ((lct-l12.lct) - (dlc-l12.dlc) - (txp-l12.txp) - dp))/abs(ib)
                         if oancf == .
        replace PctAcc = (same numerator)/.01 if oancf == . & ib == 0
    Deviations: (1) key, (2) l12. -> one-year lag. The four lines are applied in
    CZ's order, because each overwrites the one before."""
    d = lag(m, ["act", "che", "lct", "dlc", "txp"])
    balance_sheet = ((d["act"] - d["l1_act"]) - (d["che"] - d["l1_che"])
                     - ((d["lct"] - d["l1_lct"]) - (d["dlc"] - d["l1_dlc"])
                        - (d["txp"] - d["l1_txp"]) - d["dp"]))
    zero_ib = (d["ib"] == 0).fillna(False).to_numpy(dtype=bool)
    no_cash_flow = d["oancf"].isna().to_numpy(dtype=bool)

    out = safe_div(d["ib"] - d["oancf"], d["ib"].abs())
    out = out.mask(zero_ib, (d["ib"] - d["oancf"]) / 0.01)
    out = out.mask(no_cash_flow, safe_div(balance_sheet, d["ib"].abs()))
    out = out.mask(no_cash_flow & zero_ib, balance_sheet / 0.01)
    return out.values


def f_TotalAccruals(m):
    """
    Total accruals, Richardson-Sloan-Soliman-Tuna 2005 Table 8A. CZ: TotalAccruals.do.
        temp<v> = <v> with 0 for missing, for ivao ivst dltt dlc pstk sstk prstkc dv
        gen tempWc = (act - che) - (lct - tempdlc)
        gen tempNc = (at - act - tempivao) - (lt - tempdlc - tempdltt)
        gen tempFi = (tempivst + tempivao) - (tempdltt + tempdlc + temppstk)
        gen year = yofd(dofm(time_avail_m))
        gen     TotalAccruals = (tempWc - l12.tempWc) + (tempNc - l12.tempNc)
                                + (tempFi - l12.tempFi)              if year <= 1989
        replace TotalAccruals = ni - (oancf + ivncf + fincf)
                                + (sstk - prstkc - dv)               if year >  1989
        replace TotalAccruals = TotalAccruals/l12.at

    Deviations: (1) key, (2) l12. -> one-year lag.

    THE YEAR IN THE BRANCH IS THE AVAILABILITY YEAR, NOT THE FISCAL YEAR.
    CZ's `year` is built from `time_avail_m`, which is the fiscal year end plus
    six months - our `date_avail`, which is used here for exactly that reason.
    For a December filer this puts fiscal 1988 on the balance-sheet method and
    fiscal 1989 onward on the cash-flow method.

    Two things follow, and both are CZ's, not ours. First, SignalDoc says the
    cash-flow method starts in 1988 and CZ's code starts it a year later; CZ's
    comment in the .do file says so outright - "HXZ use 1988 here but that leads
    to break in data availability". Second, because the cut is drawn on a
    calendar year of availability rather than on the fiscal year, a June filer
    and a December filer with the SAME fiscal year 1989 land on opposite sides
    of it. Switching the branch to `fyear <= 1988` would remove both oddities
    and match the documented definition. It would also make our values differ
    from CZ's for a reason unrelated to our population, so it is not done here
    without your say-so. FLAGGED for your decision."""
    filled = {v: m[v].fillna(0) for v in
              ["ivao", "ivst", "dltt", "dlc", "pstk"]}
    working_capital = (m["act"] - m["che"]) - (m["lct"] - filled["dlc"])
    non_current = ((m["at"] - m["act"] - filled["ivao"])
                   - (m["lt"] - filled["dlc"] - filled["dltt"]))
    financial = ((filled["ivst"] + filled["ivao"])
                 - (filled["dltt"] + filled["dlc"] + filled["pstk"]))

    d = m.assign(wc_=working_capital, nc_=non_current, fi_=financial)
    d = lag(d, ["wc_", "nc_", "fi_", "at"])
    balance_sheet = ((d["wc_"] - d["l1_wc_"]) + (d["nc_"] - d["l1_nc_"])
                     + (d["fi_"] - d["l1_fi_"]))
    # CZ's cash-flow branch uses the raw items, not the zero-filled temp ones.
    cash_flow = (d["ni"] - (d["oancf"] + d["ivncf"] + d["fincf"])
                 + (d["sstk"] - d["prstkc"] - d["dv"]))

    early = (d["date_avail"].dt.year <= 1989).fillna(False).to_numpy(dtype=bool)
    return safe_div(balance_sheet.where(early, cash_flow), d["l1_at"]).values


def f_dNoa(m):
    """
    Change in net operating assets, Hirshleifer-Hou-Teoh-Zhang 2004 Table 7B.
    CZ: dNoa.do.
        gen tempOA = at - che
        temp<v> = <v> with 0 for missing, for dltt dlc mib pstk
        gen tempOL = at - tempdltt - tempmib - tempdlc - temppstk - ceq
        gen tempNOA = tempOA - tempOL
        gen dNoa = (tempNOA - l12.tempNOA)/l12.at
    Deviations: (1) key, (2) l12. -> one-year lag.
    Note this is not the same net-operating-assets definition as NOA.py, which
    subtracts convertible debt `dc` instead of `dlc` and `pstk`. Both are kept
    as CZ writes them."""
    operating_assets = m["at"] - m["che"]
    operating_liabs = (m["at"] - m["dltt"].fillna(0) - m["mib"].fillna(0)
                       - m["dlc"].fillna(0) - m["pstk"].fillna(0) - m["ceq"])
    d = lag(m.assign(noa_=operating_assets - operating_liabs), ["noa_", "at"])
    return safe_div(d["noa_"] - d["l1_noa_"], d["l1_at"]).values


def f_DelNetFin(m):
    """
    Change in net financial assets, Richardson et al. 2005 Table 8B. CZ: DelNetFin.do.
        gen tempAvAT = .5*(at + l12.at)
        gen tempPSTK = pstk ; replace tempPSTK = 0 if mi(pstk)
        gen temp = (ivst + ivao) - (dltt + dlc + tempPSTK)
        gen DelNetFin = (temp - l12.temp)/tempAvAT
    Deviations: (1) key, (2) l12. -> one-year lag."""
    net_financial = ((m["ivst"] + m["ivao"])
                     - (m["dltt"] + m["dlc"] + m["pstk"].fillna(0)))
    d = lag(m.assign(nf_=net_financial), ["nf_", "at"])
    return safe_div(d["nf_"] - d["l1_nf_"],
                    0.5 * (d["at"] + d["l1_at"])).values


def capex_adjusted(m):
    """
    CZ's capital expenditure with its missing-value substitution, plus the one-,
    two- and three-year lags the three grcapx features share.
    CZ: ZZ1_grcapx_grcapx1y_grcapx3y.do.
        merge 1:1 permno time_avail_m using SignalMasterTable, keep(using match)
            nogenerate keepusing(exchcd)
        bys permno (time_avail_m): gen FirmAge = _n
        gen tempcrsptime = time_avail_m - mofd(mdy(7,1,1926)) + 1
        replace FirmAge = . if tempcrsptime == FirmAge
        *replace FirmAge = . if exchcd != 1   // <- commented out in CZ's own file
        replace capx = ppent - l12.ppent if capx == . & FirmAge >= 24

    Deviations: (1) key, (2) frequency, (3) SignalMasterTable - and (3) is the
    one that matters here, so it is spelled out.

    THE MERGE IS DELETED AND FIRM AGE IS REDEFINED. `exchcd` is the only column
    the merge brings in, and the single line that used it is commented out in
    CZ's own script, so nothing in the formula needs it. But the merge is not
    inert: `keep(using match)` makes SignalMasterTable the spine of the panel,
    and FirmAge is then just a row counter down that spine - so CZ's "firm age"
    is the number of MONTHS the security has been in CRSP, which for an unlisted
    issuer is zero months forever. Ours counts the number of YEARS the firm has
    appeared in Compustat, and CZ's `>= 24` months becomes `>= 2` years.

    CZ's `tempcrsptime == FirmAge` line blanks the age of firms present in every
    month since CRSP opened in July 1926, whose true age it cannot know. The
    same test is applied to our panel's own first year, 1950. The only thing
    firm age gates is whether a missing capx may be replaced by the change in
    net PP&E, so the effect either way is small - but it is reported rather than
    dropped.

    The substitution happens BEFORE the lags are taken, as in CZ, so a
    substituted value can appear inside l1, l2 and l3 as well."""
    d = one_row_per_year(m, ["capx", "ppent"])

    age = (d.groupby("gvkey").cumcount() + 1).astype("float64")
    years_since_panel_start = d["fyear"] - int(d["fyear"].min()) + 1
    age = age.where(age != years_since_panel_start)     # age unknowable
    d["age"] = age

    d = year_lag(d, "ppent", 1)
    substitute = (d["capx"].isna() & (d["age"] >= 2)).fillna(False)
    d["capx"] = d["capx"].where(~substitute, d["ppent"] - d["l1_ppent"])

    for k in (1, 2, 3):
        d = year_lag(d, "capx", k)
    return d


def f_grcapx(m):
    """Two-year capex growth, Anderson-Garcia-Feijoo 2006 Table 3B.
        gen grcapx = (capx - l24.capx)/l24.capx
    See capex_adjusted(). Deviations: (1) key, (2) l24. -> two-year lag,
    (3) SignalMasterTable merge deleted and firm age redefined."""
    d = capex_adjusted(m)
    d["grcapx"] = safe_div(d["capx"] - d["l2_capx"], d["l2_capx"])
    return back_to_panel(m, d, "grcapx")


def f_grcapx1y(m):
    """One-year capex growth, measured a year ago.
        gen grcapx1y = (l12.capx - l24.capx)/l24.capx
    See capex_adjusted(). Deviations: as f_grcapx, with l12./l24."""
    d = capex_adjusted(m)
    d["grcapx1y"] = safe_div(d["l1_capx"] - d["l2_capx"], d["l2_capx"])
    return back_to_panel(m, d, "grcapx1y")


def f_grcapx3y(m):
    """Capex against its own three-year average, Table 3D.
        gen grcapx3y = capx/(l12.capx + l24.capx + l36.capx)*3
    See capex_adjusted(). Deviations: as f_grcapx, with l12./l24./l36."""
    d = capex_adjusted(m)
    d["grcapx3y"] = 3 * safe_div(
        d["capx"], d["l1_capx"] + d["l2_capx"] + d["l3_capx"])
    return back_to_panel(m, d, "grcapx3y")


# ---------------------------------------------------------------------------
# COMPANION INDICATORS - OURS, NOT CZ's.
#
# These three are NOT CZ characteristics and must never be counted among the
# 101. They exist because BookLeverage, RoE and roic all divide by a book-equity
# figure that goes negative once a firm's accumulated losses exceed its paid-in
# capital. Each ratio therefore has a pole at zero equity: it returns a large
# NEGATIVE number for the most heavily indebted firms, ordering them below firms
# with no leverage at all.
#
# No formula is changed. The indicator simply records which side of the pole the
# observation sits on, which is the one piece of information the ratio destroys.
# It has to be a separate column, because winsorising, standardising and ranking
# are all monotone transformations and cannot repair an ordering that is already
# inverted. Decided 2026-08-17; see docs/DECISIONS.md and
# docs/PROJECT_REPORT.md section 5.
# ---------------------------------------------------------------------------

def f_OENEG(m):
    """OENEG: 1 where total liabilities exceed total assets. Ours, not CZ's.

    Ohlson (1980), the O-score's negative-book-equity state. It is NOT a
    standalone signal in CZ's SignalDoc or JKP's list — it appears only as the
    term -1.72*I(lt > at) inside `OScore`, which this project does not build
    (it needs a GNP deflator, barred by the no-outside-data rule, and carries a
    price screen). Added by hand for that reason, and listed in OURS_NOT_CZ.

    D-50 replaced the four `_negden` indicators of D-17 with this and INTWO:
    two states with published provenance instead of four denominators' signs.
    """
    return ((m["lt"] > m["at"]).astype("Float64")
            .mask(m["lt"].isna() | m["at"].isna()))


def f_INTWO(m):
    """INTWO: 1 where NET INCOME is negative in BOTH of the last two years.

    Ohlson (1980), verbatim: "one if net income was negative for the last two
    years". Ours, not CZ's, and listed in OURS_NOT_CZ.

    NOTE — OUR DEFINITION DIFFERS FROM CZ'S O-SCORE TERM, deliberately. CZ
    implements `(ib + ib_{t-12} + ib_{t-24} < 0)`: the SUM of three years being
    negative, which is a different variable. We follow Ohlson because that is
    what the source paper defines.

    IT ALSO USES `ni`, NOT `ib`, AND THAT DEPARTS FROM THIS PROJECT'S OWN
    CONVENTION: every other loss-related quantity here keys off `ib` (see
    f_Tax), and the two differ by extraordinary items. Ohlson says "net
    income", we cite Ohlson, so we implement Ohlson — deviating to `ib` for
    internal consistency would mean documenting a modified version of a
    published variable, which defeats the reason it was chosen. Recorded here
    so the difference is read, not discovered.

    A firm without a prior year is NA, not 0: two years of history are required
    and their absence is not evidence of profitability.
    """
    d = lag(m, "ni")
    both = (d["ni"] < 0) & (d["l1_ni"] < 0)
    return (both.astype("Float64")
            .mask(d["ni"].isna() | d["l1_ni"].isna()).values)


def f_TLTA(m):
    """TLTA: total liabilities over total assets. Ours, not CZ's; OURS_NOT_CZ.

    Ohlson (1980), the O-score's leverage term (`6.03*(lt/at)`). Not a
    standalone signal in CZ's SignalDoc or JKP's list — checked 2026-08-29: CZ
    carries it only inside `OScore`, which we do not build, and JKP has no
    `debt_at`.

    Added by D-57 as the ASSET-scaled counterpart to `BookLeverage`, which is
    equity-scaled. `BookLeverage` is KEPT and still masked (D-50); both are
    built and the choice between them is a selection decision downstream (D-02),
    so it can be revisited without re-running construction.

    Measured before adding: 1.41% missing overall and 2.43% on distress events,
    against `BookLeverage`'s 37.48% on events after masking; and its realised
    default rate rises 0.31% to 5.22% across deciles rather than bending into
    the U-shape that the equity denominator produced.

    NOT masked. Its denominator is total assets, which is positive for any firm
    with a balance sheet, so the D-50 problem does not arise. `TLTA > 1` is
    definitionally `lt > at`, i.e. exactly OENEG's test — agreement measured at
    100.00% — but the ratio carries severity where the dummy carries only the
    threshold, and the rank transform compresses that severity, so both are
    kept and their joint behaviour is measured after fitting.
    """
    return safe_div(m["lt"], m["at"])


def f_ocf_at(m):
    """Operating cash flow to total assets. Ours, not CZ's; in OURS_NOT_CZ.

    JKP `ocf_at`, cited to Bouchaud, Kruger, Landier and Thesmar (2019), and a
    standalone signal in their list. It has no CZ equivalent: our nearest
    feature, `cashdebt`, is (ib + dp) over average total LIABILITIES, a
    different numerator over a different denominator.

    Chosen over the other JKP gaps because it is first-order for credit and
    transfers cleanly: CSMAR gives both items at ~100% coverage —
    经营活动产生的现金流量净额 (C001000000) over 资产总计 (A001000000).
    """
    return safe_div(m["oancf"], m["at"])


# ---------------------------------------------------------------------------
# REGISTRY. Each entry declares the panel columns the function reads, so
# check_requirements() can verify them BEFORE anything runs. This exists
# because the Stage 0.1 variable extraction silently missed 24 variables and
# the gap was only found by hand - a KeyError halfway through a run is a far
# worse way to discover the same thing.
# ---------------------------------------------------------------------------
FEATURES = {
    "GP":            (f_GP,            ["revt", "cogs", "at", "ind_sic"]),
    "RoE":           (f_RoE,           ["ni", "ceq"]),
    "PM":            (f_PM,            ["ni", "revt"]),
    "ChPM":          (f_ChPM,          ["ni", "revt", "fyear"]),
    "quick":         (f_quick,         ["act", "invt", "lct"]),
    "currat":        (f_currat,        ["act", "che", "rect", "invt", "lct", "ap"]),
    "salecash":      (f_salecash,      ["sale", "che"]),
    "saleinv":       (f_saleinv,       ["sale", "invt"]),
    "salerec":       (f_salerec,       ["sale", "rect"]),
    "roic":          (f_roic,          ["ebit", "nopi", "ceq", "lt", "che"]),
    "depr":          (f_depr,          ["dp", "ppent"]),
    "secured":       (f_secured,       ["dm", "dltt", "dlc"]),
    "securedind":    (f_securedind,    ["dm"]),
    "XFIN":          (f_XFIN,          ["sstk", "dv", "prstkc", "dltis", "dltr",
                                        "dlcch", "at"]),
    "Tax":           (f_Tax,           ["txfo", "txfed", "txt", "txdi", "ib",
                                        "date_avail"]),
    "BookLeverage":  (f_BookLeverage,  ["pstk", "pstkrv", "pstkl", "seq", "ceq",
                                        "at", "lt", "txditc"]),
    "sgr":           (f_sgr,           ["sale", "fyear"]),
    "AssetGrowth":   (f_AssetGrowth,   ["at", "fyear"]),
    "cashdebt":      (f_cashdebt,      ["ib", "dp", "lt", "fyear"]),
    "ChNWC":         (f_ChNWC,         ["act", "che", "lct", "dlc", "at", "fyear"]),
    "AssetTurnover": (f_AssetTurnover, ["rect", "invt", "aco", "ppent", "intan",
                                        "ap", "lco", "lo", "sale", "fyear"]),
    # --- batch 1, part 2 ---
    "GPlag":         (f_GPlag,         ["sale", "cogs", "at", "fyear"]),
    "OPLeverage":    (f_OPLeverage,    ["xsga", "cogs", "at"]),
    "ShareRepurchase": (f_ShareRepurchase, ["prstkc"]),
    "ConvDebt":      (f_ConvDebt,      ["dc", "cshrc"]),
    "PctTotAcc":     (f_PctTotAcc,     ["ni", "prstkcc", "sstk", "dvt", "oancf",
                                        "fincf", "ivncf"]),
    "AssetLiquidityBook": (f_AssetLiquidityBook,
                                       ["che", "act", "at", "gdwl", "intan", "fyear"]),
    "CapTurnover":   (f_CapTurnover,   ["sale", "at", "fyear"]),
    "ChEQ":          (f_ChEQ,          ["ceq", "fyear"]),
    "ChInv":         (f_ChInv,         ["invt", "at", "fyear"]),
    "ChNCOA":        (f_ChNCOA,        ["at", "act", "ivao", "fyear"]),
    "ChNCOL":        (f_ChNCOL,        ["lt", "dlc", "dltt", "at", "fyear"]),
    "ChNNCOA":       (f_ChNNCOA,       ["at", "act", "ivao", "lt", "dlc", "dltt",
                                        "fyear"]),
    "ChAssetTurnover": (f_ChAssetTurnover, ["rect", "invt", "aco", "ppent", "intan",
                                            "ap", "lco", "lo", "sale", "fyear"]),
    "DelCOA":        (f_DelCOA,        ["act", "che", "at", "fyear"]),
    "DelCOL":        (f_DelCOL,        ["lct", "dlc", "at", "fyear"]),
    "DelEqu":        (f_DelEqu,        ["ceq", "at", "fyear"]),
    "DelLTI":        (f_DelLTI,        ["ivao", "at", "fyear"]),
    "DelSTI":        (f_DelSTI,        ["ivst", "at", "fyear"]),
    "DelFINL":       (f_DelFINL,       ["dltt", "dlc", "pstk", "at", "fyear"]),
    "DelDRC":        (f_DelDRC,        ["drc", "at", "ceq", "sale", "ind_sic",
                                        "fyear"]),
    "NOA":           (f_NOA,           ["at", "che", "dltt", "mib", "dc", "ceq",
                                        "fyear"]),
    "NetDebtFinance": (f_NetDebtFinance, ["dltis", "dltr", "dlcch", "at", "fyear"]),
    "NetEquityFinance": (f_NetEquityFinance, ["sstk", "prstkc", "dv", "at", "fyear"]),
    "CompositeDebtIssuance": (f_CompositeDebtIssuance, ["dltt", "dlc", "fyear"]),
    "InvestPPEInv":  (f_InvestPPEInv,  ["ppegt", "invt", "at", "fyear"]),
    "LaborforceEfficiency": (f_LaborforceEfficiency, ["sale", "emp", "fyear"]),
    "OrderBacklog":  (f_OrderBacklog,  ["ob", "at", "fyear"]),
    "OrderBacklogChg": (f_OrderBacklogChg, ["ob", "at", "fyear"]),
    "RetNOA":        (f_RetNOA,        ["at", "che", "ivao", "dlc", "dltt", "mib",
                                        "pstk", "ceq", "oiadp", "fyear"]),
    "SurpriseRD":    (f_SurpriseRD,    ["xrd", "revt", "at", "fyear"]),
    "rd_sale":       (f_rd_sale,       ["xrd", "sale", "fyear"]),
    "Accruals":      (f_Accruals,      ["act", "che", "lct", "dlc", "txp", "dp",
                                        "at", "fyear"]),
    "pchcurrat":     (f_pchcurrat,     ["act", "che", "rect", "invt", "lct", "ap",
                                        "fyear"]),
    "pchquick":      (f_pchquick,      ["act", "invt", "lct", "fyear"]),
    "pchsaleinv":    (f_pchsaleinv,    ["sale", "invt", "fyear"]),
    "pchdepr":       (f_pchdepr,       ["dp", "ppent", "fyear"]),
    "pchgm_pchsale": (f_pchgm_pchsale, ["sale", "cogs", "fyear"]),
    "GrGMToGrSales": (f_GrGMToGrSales, ["sale", "cogs", "fyear"]),
    "GrSaleToGrReceivables": (f_GrSaleToGrReceivables, ["sale", "rect", "fyear"]),
    "GrSaleToGrInv": (f_GrSaleToGrInv, ["sale", "invt", "fyear"]),
    "GrSaleToGrOverhead": (f_GrSaleToGrOverhead, ["sale", "xsga", "fyear"]),
    # --- batch 1, part 3 ---
    "BrandCapital":  (f_BrandCapital,  ["xad", "at", "fyear"]),
    "BrandInvest":   (f_BrandInvest,   ["xad", "xad0", "at", "ind_sic",
                                        "fyear"]),
    "EarningsConsistency": (f_EarningsConsistency, ["epspx", "fyear"]),
    "EarningsPersistence": (f_EarningsPersistence, ["epspx", "ajex", "fyear"]),
    "EarningsPredictability": (f_EarningsPredictability,
                                       ["epspx", "ajex", "fyear"]),
    "GrLTNOA":       (f_GrLTNOA,       ["rect", "invt", "ppent", "aco", "intan",
                                        "ao", "ap", "lco", "lo", "at", "dp",
                                        "fyear"]),
    "MeanRankRevGrowth": (f_MeanRankRevGrowth, ["revt", "fyear"]),
    "OperProfRDLagAT": (f_OperProfRDLagAT, ["revt", "cogs", "xsga", "xrd", "at",
                                            "ind_sic", "fyear"]),
    "PctAcc":        (f_PctAcc,        ["ib", "oancf", "dp", "act", "che",
                                        "lct", "txp", "dlc", "fyear"]),
    "TotalAccruals": (f_TotalAccruals, ["ivao", "ivst", "dltt", "dlc", "pstk",
                                        "sstk", "prstkc", "dv", "act", "che",
                                        "lct", "at", "lt", "ni", "oancf",
                                        "ivncf", "fincf", "date_avail",
                                        "fyear"]),
    "dNoa":          (f_dNoa,          ["at", "che", "dltt", "dlc", "mib",
                                        "pstk", "ceq", "fyear"]),
    "DelNetFin":     (f_DelNetFin,     ["at", "pstk", "dltt", "dlc", "ivst",
                                        "ivao", "fyear"]),
    "grcapx":        (f_grcapx,        ["capx", "ppent", "fyear"]),
    "grcapx1y":      (f_grcapx1y,      ["capx", "ppent", "fyear"]),
    "grcapx3y":      (f_grcapx3y,      ["capx", "ppent", "fyear"]),
    # --- ours, not CZ's; NOT part of the 101. D-50 replaced D-17's four
    #     `_negden` companions with two Ohlson (1980) states, and added one
    #     JKP signal that CZ has no equivalent for. ---
    "OENEG":         (f_OENEG,         ["lt", "at"]),
    "TLTA":          (f_TLTA,          ["lt", "at"]),
    "INTWO":         (f_INTWO,         ["ni", "fyear"]),
    "ocf_at":        (f_ocf_at,        ["oancf", "at"]),
}

# Everything we added ourselves. Not CZ characteristics; anything that counts
# the 101 must exclude these.
# Sources, so nothing here is ever mislabelled as a CZ characteristic:
#   OENEG, INTWO, TLTA   Ohlson (1980) — components of the O-score, which we do
#                 not build; none is a standalone CZ or JKP signal.
#   ocf_at        JKP `ocf_at`, Bouchaud, Kruger, Landier and Thesmar (2019) —
#                 a standalone JKP signal with no CZ equivalent.
OURS_NOT_CZ = ["OENEG", "INTWO", "TLTA", "ocf_at"]


# ---------------------------------------------------------------------------
# UNDISCLOSED ITEMS - restoring a filled zero to missing. Decided 2026-08-17.
#
# Stage 1.0 follows CZ in replacing missing with zero for 26 items. For a
# handful of characteristics that turns "the firm did not report this" into
# "the firm has none of this", which are different statements. The clearest
# case is securedind, whose formula is `1 if dm != . & dm != 0` - CZ tests
# explicitly for missing, and the blanket fill has already destroyed the thing
# being tested.
#
# The rule for what appears below, and it is deliberately narrow:
#   mask ONLY where the zero came from Stage 1.0's blanket fill AND CZ's own
#   formula does not itself instruct filling that item with zero.
# Where CZ writes `replace tempX = 0 if mi(X)` the zero is part of the formula,
# not a preparation artefact, so those are left alone - that covers dNoa,
# TotalAccruals, DelNetFin, DelFINL and RetNOA. ChNCOA is also left alone
# because CZ's own `replace temp = at - act if mi(ivao)` gives the identical
# answer. OrderBacklog needs nothing because CZ already drops ob == 0.
#
# This is applied AFTER the formula runs, so no formula is modified. Setting the
# switch to False reproduces the previous behaviour exactly.
# ---------------------------------------------------------------------------

RESTORE_UNDISCLOSED = True

# feature -> (items whose disclosure matters, which years to check)
UNDISCLOSED_MASK = {
    "secured":    (["dm"],          [0]),
    "securedind": (["dm"],          [0]),
    "ConvDebt":   (["dc"],          [0]),
    "NOA":        (["mib", "dc"],   [0]),
    "DelLTI":     (["ivao"],        [0, 1]),
    "DelSTI":     (["ivst"],        [0, 1]),
    "ChNNCOA":    (["ivao"],        [0, 1]),
    "DelDRC":     (["drc"],         [0, 1]),
}


def undisclosed(m, items, years):
    """True where any of `items` was missing before Stage 1.0's zero-fill, in
    the current year or the listed number of years back."""
    flags = [f"{i}_was_missing" for i in items]
    out = np.zeros(len(m), dtype=bool)
    for k in years:
        if k == 0:
            hit = m[flags].any(axis=1)
        else:
            d = lag(m, flags, k)
            hit = d[[f"l{k}_{f}" for f in flags]].any(axis=1)
        out |= hit.fillna(False).to_numpy(dtype=bool)
    return out


# ===========================================================================
# RUN
# ===========================================================================

def check_requirements(m):
    """Verify every declared column exists BEFORE computing anything."""
    print("\nChecking every feature's inputs are present...")
    missing = {}
    for name, (_, cols) in FEATURES.items():
        gap = [c for c in cols if c not in m.columns]
        if gap:
            missing[name] = gap
    if missing:
        print("\nERROR: these features reference columns the panel does not have:")
        for name, gap in missing.items():
            print(f"  {name:<18} {gap}")
        raise SystemExit("Stopping - fix the download or the registry first.")
    print(f"  all {len(FEATURES)} features have their inputs")


def run_features(m):
    """Compute each feature. A failure is recorded and the run continues, so one
    broken formula does not hide the state of the other hundred."""
    print(f"\nComputing {len(FEATURES)} features...")
    out = m[KEY + ["fyear"]].copy()
    failed = {}
    for name, (fn, _) in FEATURES.items():
        try:
            out[name] = pd.array(fn(m), dtype="Float64")
        except Exception as err:
            failed[name] = f"{type(err).__name__}: {err}"
            print(f"  FAILED {name}: {type(err).__name__}: {err}")
    print(f"  succeeded {len(FEATURES) - len(failed)} / {len(FEATURES)}")

    # Restore a filled zero to missing where the item was never disclosed.
    if RESTORE_UNDISCLOSED:
        print("\nRestoring undisclosed items to missing...")
        for name, (items, years) in UNDISCLOSED_MASK.items():
            if name in failed:
                continue
            hit = undisclosed(m, items, years)
            before = int(out[name].notna().sum())
            out.loc[hit, name] = pd.NA
            after = int(out[name].notna().sum())
            print(f"  {name:<12} -{before - after:>7,} obs "
                  f"({', '.join(items)} undisclosed)")

    return out, failed


def coverage_report(panel):
    """Per-feature diagnostics. This is what the Stage 2 screen reads to decide
    which characteristics survive, so it reports missingness and shape rather
    than just a count."""
    names = [c for c in panel.columns if c not in KEY + ["fyear"]]
    rows = []
    for c in names:
        s = panel[c]
        ok = s.notna()
        rows.append(dict(
            feature=c,
            n_obs=int(ok.sum()),
            pct_missing=round(100 * (1 - ok.mean()), 2),
            n_firms=int(panel.loc[ok, "gvkey"].nunique()),
            first_year=int(panel.loc[ok, "fyear"].min()) if ok.any() else None,
            last_year=int(panel.loc[ok, "fyear"].max()) if ok.any() else None,
            p01=s.quantile(0.01), p50=s.quantile(0.50), p99=s.quantile(0.99),
            n_inf=int(np.isinf(s.astype("float64")).sum()),
        ))
    return pd.DataFrame(rows).sort_values("pct_missing")


def main():
    print("stage1_1_construct_features.py")
    print(f"  reading from {ANNUAL.name}")
    print(f"  writing to   {OUTDIR}")

    m = load_master()
    check_requirements(m)
    panel, failed = run_features(m)

    # Same guard as the master: if the finished panel has lost a large share of
    # firms, a listing screen has crept in somewhere.
    n_firms = panel.gvkey.nunique()
    print(f"\n  panel covers {n_firms:,} firms")
    assert n_firms >= MIN_FIRMS, (
        f"only {n_firms:,} firms, expected at least {MIN_FIRMS:,} - "
        "a listing screen may have been re-introduced")

    cov = coverage_report(panel)
    print("\nCoverage (sorted by missing rate):")
    print(cov[["feature", "n_obs", "pct_missing", "n_firms",
               "first_year", "last_year"]].to_string(index=False))

    panel.to_parquet(OUTDIR / "stage1_1_feature_panel.parquet", index=False)
    cov.to_csv(OUTDIR / "stage1_1_feature_coverage.csv", index=False)
    print(f"\n  -> stage1_1_feature_panel.parquet    {len(panel):,} rows x "
          f"{panel.shape[1]} cols")
    print(f"  -> stage1_1_feature_coverage.csv     {len(cov)} features")
    if failed:
        print(f"\n  {len(failed)} feature(s) failed: {list(failed)}")


if __name__ == "__main__":
    main()
