# ABOUTME: Stage 1.0 - turns the raw Compustat download into one clean row per firm-period
# ABOUTME: De-duplicates, builds CZ's derived items, applies CZ's zero-fill, adds date_avail
"""
stage1_0_prep_compustat.py

WHAT THIS DOES:
    Two things, in order.

    PART A - CHECKS. Prints a report on the raw data. Writes nothing. This is
    what the decisions in PART B were made from, and it is kept in the script so
    the numbers can be re-verified rather than taken on trust. Set RUN_CHECKS to
    False to skip it once you have seen the output.

    PART B - PREP. Produces one clean row per firm-period:
        output/compustat_annual_clean.parquet
        output/compustat_quarterly_clean.parquet

WHY THIS IS NOT A COPY OF CZ:
    CZ's DataDownloads/CompustatAnnual.py builds its input layer for LISTED
    EQUITIES. Our sample is BOND ISSUERS. Two of its lines are NOT carried over:

        dropna(subset=['at','prcc_c','ni'])   -> deletes every firm with no share price
        merge(ccm_link, how='inner')          -> keeps only firms with a CRSP permno

    Both would delete exactly the unlisted issuers Stage 0 preserved. Everything
    else - the formulas, the derived items, the zero-fill, the six-month
    availability lag - is taken from CZ unchanged.

THE FOUR DECISIONS IN PART B, and the numbers behind them:

    1. DE-DUPLICATE by keeping datafmt == 'STD' and indfmt == 'INDL'.
       Raw funda is 941,735 rows for 606,366 firm-years (1.55 each, 69.9% of
       rows in a duplicate). Why these two:
         - SUMM_STD reports 0.000 non-missing for lt, revt, ceq and oancf
           against 0.829 / 0.837 / 0.815 / 0.506 under STD. It is a summary
           format carrying almost nothing.
         - FS is dominated by INDL. On the 97,436 rows where a firm-year has
           both, INDL matches FS on every balance-sheet item (at 0.925 vs 0.926)
           and additionally carries ni (0.919 vs 0.000) and oancf (0.582 vs
           0.000), which FS does not report at all. 48,722 of 48,807 FS
           firm-years (99.8%) also exist as INDL; dropping FS loses 1 firm.
       curcd is deliberately NOT filtered: the 107,075 CAD rows are real
       Canadian firm-years, not duplicates, and FISD has Canadian issuers.
       That is a Stage 3 sample decision, not a cleaning step.
       consol is deliberately NOT filtered either. CZ keeps consol == 'C', but
       here that would delete 1,683 firm-years across 382 firms that exist ONLY
       under P/R/D and carry good data (ni non-missing 98.6%), in order to
       resolve a SINGLE collision. consol is used as a local tie-breaker
       instead - see CONSOL_PREFERENCE and prep_annual().

    2. INDUSTRY CODE: use sich where present, leave missing otherwise.
       sich covers 58.6% of firm-years and 0% before 1980. It is NOT CZ's
       sicCRSP, so the 4 industry-adjusted features are an approximation of CZ,
       not a replication. No filling within firm across years - a firm's
       industry genuinely changes and filling would manufacture data.

    3. DERIVED ITEMS: build CZ's dc, xint0, xsga0, xad0 here, because we
       deliberately kept the raw layer unmodified in Stage 0.
       NOTE on CZ's `dr` (deferred revenue): it needs drc AND drlt, and we did
       not download drlt. This is fine - none of our 101 features require dr.
       The script asserts this rather than assuming it.

    4. ZERO-FILL: adopt CZ's 30-item fillna(0) so the feature formulas stay
       identical to CZ's and reproducible. We hold 22 of the 30; the other 8 are
       either CZ-derived (dc) or unused by our 101 features.
       Scale, measured on the de-duplicated panel: 30.9% of affected cells, and
       for single items far more (prstkcc 97.3%, ob 81.7%, drc 69.3%).
       Because "not disclosed" may itself be informative for DEFAULT prediction
       in a way it is not for return prediction, a <item>_was_missing flag is
       written alongside each filled item. This changes no feature value; it
       only preserves the information so Stage 3 can test it. Set
       SAVE_MISSING_FLAGS = False to drop them.

Inputs:
    data_raw/us/compustat_funda.parquet      941,735 x 78
    data_raw/us/compustat_fundq.parquet    2,133,913 x 20

Outputs:
    output/compustat_annual_clean.parquet
    output/compustat_quarterly_clean.parquet

How to run:
    python3 code/stage1_0_prep_compustat.py

Memory note:
    The annual file is ~1.1 GB if loaded whole. The row filter is pushed down
    into the parquet reader, so only the ~602k surviving rows are ever
    materialised (roughly 700 MB). Checks read one small column set at a time.
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

# Print PART A. Turn off once the numbers have been reviewed.
RUN_CHECKS = True

# Write <item>_was_missing alongside each zero-filled item. See decision 4.
SAVE_MISSING_FLAGS = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data_raw" / "us"
OUTDIR = PROJECT_ROOT / "output"

FUNDA = RAW / "compustat_funda.parquet"
FUNDQ = RAW / "compustat_fundq.parquet"

# The five Compustat reporting-format flags. Not accounting data, never
# features - they exist only so duplicate firm-periods can be told apart.
FLAGS_A = ["consol", "popsrc", "datafmt", "curcd", "indfmt"]
FLAGS_Q = ["consol", "popsrc", "datafmt", "curcdq", "indfmt"]

# The 30 items CZ fills with zero, copied verbatim from
# the upstream OpenSourceAP/CrossSection CompustatAnnual.py implementation.
CZ_ZERO_FILL = [
    "nopi", "dvt", "ob", "dm", "dc", "aco", "ap", "intan", "ao",
    "lco", "lo", "rect", "invt", "drc", "spi", "gdwl", "che",
    "dp", "act", "lct", "tstkp", "dvpa", "scstkc", "sstk",
    "mib", "ivao", "prstkc", "prstkcc", "txditc", "ivst",
]

# CZ's SEPARATE quarterly zero-fill list, copied verbatim from
# DataDownloads/CompustatQuarterly.py. It is NOT the annual list with 'q'
# appended - an earlier version of this script invented a list, which would
# have zero-filled atq, ibq, revtq and txtq. CZ does not touch those, and
# filling them would have manufactured firms with zero total assets and zero
# revenue. Of these 18 we hold exactly one, cheq; apply_zero_fill() intersects
# with the columns actually present and reports what it applied.
CZ_ZERO_FILL_Q = [
    "acoq", "actq", "apq", "cheq", "dpq", "drcq", "invtq", "intanq",
    "ivaoq", "gdwlq", "lcoq", "lctq", "loq", "mibq", "prstkcy",
    "rectq", "sstky", "txditcq",
]

# The quarterly panel key, following CZ: the FISCAL quarter, not the calendar
# date. A firm changing its fiscal year end produces one calendar date carrying
# two different fiscal quarters, and those are genuinely different periods -
# 843 of 1,697 such pairs hold conflicting values, concentrated in flow items
# (cshprq 565, ibq 446, revtq 382) rather than stocks (atq 52).
QUARTER_KEY = ["gvkey", "fyearq", "fqtr"]

# Tie-breaker used ONLY where a firm-year survives the main filter twice.
# Consolidated first, matching CZ's preference. This is not a global filter -
# see the note in prep_annual() for why a global consol filter is rejected.
CONSOL_PREFERENCE = ["C", "P", "R", "D"]

# The availability lag - the earliest date a statement could plausibly have been
# known. This is the project's no-look-ahead safeguard.
#
# CZ uses SIX months for annual and THREE for quarterly, and we follow both.
# The asymmetry is real, not an oversight: a 10-K is due 60-90 days after year
# end, a 10-Q within 40-45 days. For quarterly the lag is only a floor, because
# the actual earnings release date rdq overrides it whenever rdq is later.
AVAIL_LAG_MONTHS_A = 6
AVAIL_LAG_MONTHS_Q = 3

# CZ additionally DROPS quarterly rows whose rdq is more than 6 months after
# quarter end. We deliberately do NOT - decision 2026-08-16. Those rows are
# flagged instead. Dropping them is not needed to prevent look-ahead: the rdq
# override above has already pushed their availability date out to the actual
# filing date, which is the honest answer.
LATE_FILING_MONTHS = 6


def banner(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


# ===========================================================================
# PART A - CHECKS. Read-only.
# ===========================================================================

def check_1_duplicates():
    """
    How bad is the duplication, and do the format flags separate it cleanly?

    NOTE on a trap: groupby() drops rows whose grouping key is missing, so
    counting groups understates the row count wherever a flag is null (curcd has
    476 nulls). duplicated() is used instead because it does not.
    """
    banner("CHECK 1 - duplicate firm-years, and whether the flags separate them")

    df = pd.read_parquet(FUNDA, columns=["gvkey", "datadate"] + FLAGS_A)
    n = len(df)
    n_fy = df.groupby(["gvkey", "datadate"], observed=True).ngroups
    print(f"  rows                          : {n:,}")
    print(f"  distinct gvkey x datadate     : {n_fy:,}")
    print(f"  average rows per firm-year    : {n / n_fy:.2f}")

    unseparated = df.duplicated(["gvkey", "datadate"] + FLAGS_A, keep=False).sum()
    print(f"  rows NOT separated by the five flags: {unseparated:,}")
    if unseparated:
        print("  -> WARNING: de-duplication needs a further rule. Investigate.")
    else:
        print("  -> the flags uniquely identify every row. De-dup is well defined.")

    print("\n  flag combinations present (top 8 of "
          f"{df.groupby(FLAGS_A, observed=True).ngroups}):")
    combo = df.groupby(FLAGS_A, observed=True).size().sort_values(ascending=False)
    for keys, cnt in combo.head(8).items():
        print(f"    {' | '.join(str(k) for k in keys):<40} {cnt:>9,}")

    dup_mask = df.duplicated(["gvkey", "datadate"], keep=False)
    print(f"\n  rows involved in a duplicated firm-year: {dup_mask.sum():,} "
          f"({100 * dup_mask.mean():.1f}%)")
    for c in FLAGS_A:
        varies = (df.loc[dup_mask]
                    .groupby(["gvkey", "datadate"], observed=True)[c]
                    .nunique() > 1).sum()
        print(f"    {c:<8} varies within {varies:,} duplicated firm-years")


def check_2_which_row_has_the_data():
    """
    When a firm-year appears twice, which copy carries the numbers? This is what
    decision 1 rests on. The comparison is made WITHIN datafmt == 'STD' so that
    the summary format does not contaminate the indfmt comparison.
    """
    banner("CHECK 2 - which format flag carries the accounting data")

    core = ["at", "lt", "revt", "ni", "ceq", "oancf"]
    df = pd.read_parquet(FUNDA, columns=["gvkey", "datadate"] + FLAGS_A + core)

    print("\n  non-missing rate of core items, by datafmt (all rows):")
    tab = df.groupby("datafmt", observed=True)[core].apply(lambda g: g.notna().mean())
    tab.insert(0, "rows", df.groupby("datafmt", observed=True).size())
    print(tab.round(3).to_string())

    # Apples to apples: STD only, and only firm-years that have BOTH formats.
    std = df[df["datafmt"] == "STD"]
    both = std[std.duplicated(["gvkey", "datadate"], keep=False)]
    both = both[both["indfmt"].isin(["INDL", "FS"])]
    print(f"\n  non-missing rate by indfmt, within STD, restricted to the "
          f"{len(both):,} rows\n  whose firm-year appears under BOTH formats:")
    print(both.groupby("indfmt", observed=True)[core]
              .apply(lambda g: g.notna().mean()).round(3).to_string())

    # Is FS ever the only presentation of a firm-year?
    fs = set(map(tuple, std.loc[std.indfmt == "FS", ["gvkey", "datadate"]].values))
    indl = set(map(tuple, std.loc[std.indfmt == "INDL", ["gvkey", "datadate"]].values))
    print(f"\n  FS firm-years {len(fs):,} | also present as INDL {len(fs & indl):,} "
          f"({100 * len(fs & indl) / len(fs):.1f}%) | FS-only {len(fs - indl):,}")


def check_3_industry_codes():
    """
    Four of the 101 features are industry-adjusted. CZ uses CRSP's sicCRSP,
    which we do not have for unlisted firms, so Compustat's sich is substituted.
    That is only viable where sich is populated - decision 2 rests on this.
    """
    banner("CHECK 3 - industry code coverage (decides 4 industry-adjusted features)")

    df = pd.read_parquet(FUNDA, columns=["gvkey", "fyear", "sich", "naicsh"])
    for c in ["sich", "naicsh"]:
        print(f"  {c:<8} non-missing: {df[c].notna().sum():>9,} / {len(df):,} "
              f"= {100 * df[c].notna().mean():.1f}%")

    print("\n  coverage by decade (share non-missing):")
    df["decade"] = (df["fyear"] // 10 * 10).astype("Int64")
    tab = df.groupby("decade", observed=True)[["sich", "naicsh"]].apply(
        lambda g: g.notna().mean())
    tab.insert(0, "rows", df.groupby("decade", observed=True).size())
    print(tab.round(3).to_string())

    per_firm = df.groupby("gvkey", observed=True)["sich"].apply(lambda s: s.notna().any())
    print(f"\n  firms with sich in at least one year: {per_firm.sum():,} / "
          f"{len(per_firm):,} = {100 * per_firm.mean():.1f}%")
    print("  (we do NOT fill across years - see decision 2)")


def check_4_zero_fill():
    """
    How much data would CZ's zero-fill manufacture? Decision 4 rests on this.
    Measured both on all rows and on the rows that survive de-duplication - the
    difference is large, because the SUMM_STD rows we drop are nearly empty.
    """
    banner("CHECK 4 - CZ's 30 zero-filled items: what we have, and what it would fill")

    # Column names from parquet metadata only - opening the file with pandas
    # just to see its columns would pull all 1.1 GB into memory.
    have = set(pq.ParquetFile(FUNDA).schema_arrow.names)
    present = [c for c in CZ_ZERO_FILL if c in have]
    absent = [c for c in CZ_ZERO_FILL if c not in have]
    print(f"  of CZ's {len(CZ_ZERO_FILL)} zero-filled items we downloaded {len(present)}")
    print(f"  not downloaded ({len(absent)}): {absent}")
    print("  ('dc' is CZ-derived, built below from dcpstk/pstk/dcvt; the rest are")
    print("   not required by any of our 101 features)")

    df = pd.read_parquet(FUNDA, columns=present + ["datafmt", "indfmt"])
    kept = df[(df["datafmt"] == "STD") & (df["indfmt"] == "INDL")]
    print(f"\n  share of cells that would go from missing to 0.0:")
    print(f"    all rows                : {100 * df[present].isna().mean().mean():.1f}%"
          f"  ({len(df):,} rows)")
    print(f"    after de-duplication    : {100 * kept[present].isna().mean().mean():.1f}%"
          f"  ({len(kept):,} rows)")
    print("\n  worst items after de-duplication:")
    for c, m in kept[present].isna().mean().sort_values(ascending=False).head(8).items():
        print(f"    {c:<10} {100 * m:>5.1f}%  {'#' * int(round(m * 40))}")


def check_5_dates():
    """
    date_avail = datadate + 6 months is the no-look-ahead rule. datadate arrived
    as text, so verify it parses before depending on it.
    """
    banner("CHECK 5 - can date_avail be built from datadate")

    for path, name in [(FUNDA, "funda"), (FUNDQ, "fundq")]:
        s = pd.read_parquet(path, columns=["datadate"])["datadate"]
        parsed = pd.to_datetime(s, errors="coerce")
        bad = parsed.isna() & s.notna()
        print(f"  {name}: {len(s):,} rows | unparseable {bad.sum():,} | "
              f"null {s.isna().sum():,}")
        print(f"         range {parsed.min().date()} to {parsed.max().date()}")


def check_6_identifiers():
    """gvkey is the panel key; cusip is the bridge to FISD. Both must be clean."""
    banner("CHECK 6 - identifier integrity")

    df = pd.read_parquet(FUNDA, columns=["gvkey", "cusip", "fyear"])
    print(f"  gvkey  null {df.gvkey.isna().sum():,} | distinct {df.gvkey.nunique():,} "
          f"| lengths {df.gvkey.str.len().value_counts().to_dict()}")
    print(f"  cusip  null {df.cusip.isna().sum():,} | distinct {df.cusip.nunique():,} "
          f"| lengths {df.cusip.str.len().value_counts().to_dict()}")
    print(f"  fyear  null {df.fyear.isna().sum():,} | range {df.fyear.min()} "
          f"to {df.fyear.max()}")

    multi = df.assign(c6=df.cusip.str[:6]).groupby("gvkey", observed=True)["c6"].nunique()
    print(f"\n  firms with more than one 6-digit cusip ever: {(multi > 1).sum():,} "
          f"of {len(multi):,}")
    print("  (zero confirms Compustat stores only the CURRENT cusip - which is")
    print("   why FISD's historical cusips fail to match, and why route B exists)")


# ===========================================================================
# PART B - PREP. Writes to output/.
# ===========================================================================

def build_derived_items(df):
    """
    Rebuild the variables CZ creates during its download and then treats as
    primitives in the feature scripts. Logic copied from CZ's CompustatAnnual.py
    unchanged - this is exactly the kind of thing we borrow rather than reinvent.
    """
    # Convertible debt. CZ's three-branch rule, in its original order.
    df["dc"] = pd.NA
    m1 = (df["dcpstk"] > df["pstk"]) & df["pstk"].notna() & \
         df["dcpstk"].notna() & df["dcvt"].isna()
    df.loc[m1, "dc"] = df.loc[m1, "dcpstk"] - df.loc[m1, "pstk"]

    m2 = df["pstk"].isna() & df["dcpstk"].notna() & df["dcvt"].isna()
    df.loc[m2, "dc"] = df.loc[m2, "dcpstk"]

    m3 = df["dc"].isna()
    df.loc[m3, "dc"] = df.loc[m3, "dcvt"]
    df["dc"] = df["dc"].astype("Float64")

    # Zero-filled expense variants. CZ keeps these SEPARATE from the originals,
    # so a feature can use either. We do the same rather than overwriting.
    df["xint0"] = df["xint"].fillna(0)
    df["xsga0"] = df["xsga"].fillna(0)
    df["xad0"] = df["xad"].fillna(0)

    # CZ also builds `dr` (deferred revenue) from drc + drlt. We did not
    # download drlt, and none of our 101 features need dr, so it is not built.
    # Left as a comment rather than silently omitted.

    print(f"    derived: dc ({df['dc'].notna().sum():,} non-missing), "
          f"xint0, xsga0, xad0")
    return df


def apply_zero_fill(df, items):
    """
    CZ's fillna(0). Records what was filled first, so the information is not
    destroyed - see decision 4 in the module docstring.
    """
    items = [c for c in items if c in df.columns]
    if SAVE_MISSING_FLAGS:
        flags = {f"{c}_was_missing": df[c].isna() for c in items}
        df = pd.concat([df, pd.DataFrame(flags, index=df.index)], axis=1)
    filled = int(df[items].isna().sum().sum())
    for c in items:
        df[c] = df[c].fillna(0)
    print(f"    zero-filled {filled:,} cells across {len(items)} items"
          f"{' (+ _was_missing flags)' if SAVE_MISSING_FLAGS else ''}")
    return df


def prep_annual():
    """One clean row per gvkey x datadate, with date_avail attached."""
    banner("PREP - annual (comp.funda)")

    # Push the row filter into the parquet reader so the ~340k rows we are
    # discarding are never materialised in memory.
    df = pd.read_parquet(
        FUNDA, filters=[("datafmt", "==", "STD"), ("indfmt", "==", "INDL")])
    print(f"  after keeping datafmt=='STD' and indfmt=='INDL': {len(df):,} rows")

    # Drop the shell records: null curcd and null fyear are the same 476 rows,
    # and they carry 0.0% non-missing on every core item.
    shells = df["curcd"].isna() | df["fyear"].isna()
    df = df[~shells].copy()
    print(f"  after dropping {int(shells.sum()):,} shell rows: {len(df):,} rows")

    # One firm-year survives the filter twice, differing only in `consol`.
    #
    # CZ resolves this globally with consol == 'C'. We deliberately do NOT,
    # because measured on this data that filter would delete 1,683 firm-years
    # across 382 firms that exist ONLY under P/R/D and carry good data (ni
    # non-missing 98.6%, revt 95.3%) - a large real loss to fix one collision.
    # Break the tie locally instead, preferring the consolidated presentation.
    dup = df.duplicated(["gvkey", "datadate"], keep=False)
    if dup.any():
        print(f"\n  {int(dup.sum()):,} rows share a gvkey x datadate; breaking the tie "
              f"on consol, preferring {CONSOL_PREFERENCE[0]!r}:")
        print(df.loc[dup, ["gvkey", "datadate"] + FLAGS_A]
                .sort_values(["gvkey", "datadate"]).to_string(index=False))
        rank = {v: i for i, v in enumerate(CONSOL_PREFERENCE)}
        df["_rank"] = df["consol"].map(rank).fillna(len(rank)).astype(int)
        before = len(df)
        df = (df.sort_values(["gvkey", "datadate", "_rank"])
                .drop_duplicates(["gvkey", "datadate"], keep="first")
                .drop(columns="_rank"))
        print(f"  dropped {before - len(df):,} tied row(s)")

    # The panel key must now be unique. If it is not, stop rather than pick a
    # row arbitrarily - an arbitrary choice here would silently corrupt every
    # lag computed downstream.
    still = df.duplicated(["gvkey", "datadate"], keep=False)
    if still.any():
        print(f"\n  ERROR: {int(still.sum()):,} rows still share a gvkey x datadate.")
        print(df.loc[still, ["gvkey", "datadate"] + FLAGS_A].head(10).to_string(index=False))
        raise SystemExit("Stopping - the de-duplication rule is incomplete.")
    print(f"  gvkey x datadate is unique: {len(df):,} firm-years")

    # Dates. datadate is text in the raw layer; parse it once here.
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["date_avail"] = df["datadate"] + pd.DateOffset(months=AVAIL_LAG_MONTHS_A)
    print(f"  date_avail = datadate + {AVAIL_LAG_MONTHS_A} months  "
          f"({df.date_avail.min().date()} to {df.date_avail.max().date()})")

    df = build_derived_items(df)
    df = apply_zero_fill(df, CZ_ZERO_FILL)

    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    out = OUTDIR / "compustat_annual_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"  -> {out.name}  {len(df):,} rows x {df.shape[1]} cols  "
          f"({out.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"     {df.gvkey.nunique():,} firms, fiscal years "
          f"{int(df.fyear.min())}-{int(df.fyear.max())}")
    return df


def prep_quarterly():
    """
    One clean row per gvkey x datadate at quarterly frequency.

    Three things differ from the annual panel, all following CZ:
      - the key is the FISCAL quarter (gvkey x fyearq x fqtr), not the calendar
        date, because one calendar date can carry two different fiscal quarters
        when a firm changes its year end;
      - the availability lag is 3 months rather than 6, overridden by the actual
        earnings release date rdq whenever that is later;
      - the zero-fill list is CZ's separate quarterly one, of which we hold only
        cheq.

    One thing does NOT follow CZ: it drops rows filed more than 6 months after
    quarter end, and we flag them instead. Decision of 2026-08-16.
    """
    banner("PREP - quarterly (comp.fundq)")

    df = pd.read_parquet(FUNDQ, filters=[("datafmt", "==", "STD")])
    print(f"  after keeping datafmt=='STD': {len(df):,} rows")
    # indfmt is INDL for every row of fundq, so no filter is needed there.
    # curcdq and consol are left unfiltered, consistent with the annual panel.

    # A row with no fiscal-quarter label cannot be placed in a quarterly panel.
    # Report these rather than dropping them quietly.
    unkeyed = df["fyearq"].isna() | df["fqtr"].isna()
    if unkeyed.any():
        vals = [c for c in ["atq", "cheq", "ibq", "revtq", "txtq"] if c in df.columns]
        print(f"  {int(unkeyed.sum()):,} rows have no fyearq/fqtr and cannot be keyed; "
              f"excluded.")
        print(f"    NOTE: these are NOT empty - their core items are "
              f"{100 * df.loc[unkeyed, vals].notna().mean().mean():.1f}% non-missing. "
              f"This is a real,")
        print(f"    if tiny, loss ({100 * unkeyed.mean():.3f}% of rows): a quarterly "
              f"observation with no")
        print(f"    fiscal-quarter label cannot be placed in a quarterly panel.")
        df = df[~unkeyed].copy()

    # De-duplicate on the FISCAL quarter, keeping the record with the latest
    # datadate - CZ's rule exactly (sort then .last()). Within one fiscal
    # quarter, repeated records are revisions and the latest datadate is the
    # current version.
    df["datadate"] = pd.to_datetime(df["datadate"])
    dup = df.duplicated(QUARTER_KEY, keep=False)
    if dup.any():
        # How many of those are still tied after datadate, i.e. resolved
        # arbitrarily rather than by recency? Report it; do not hide it.
        tied = (df[dup].groupby(QUARTER_KEY, observed=True)["datadate"]
                       .nunique() == 1).sum()
        print(f"  {int(dup.sum()):,} rows share {'x'.join(QUARTER_KEY)}; keeping the "
              f"latest datadate in each")
        print(f"    of those groups, {int(tied):,} are tied on datadate too and are "
              f"resolved arbitrarily")
        before = len(df)
        df = (df.sort_values(QUARTER_KEY + ["datadate"])
                .drop_duplicates(QUARTER_KEY, keep="last"))
        print(f"    dropped {before - len(df):,} rows")

    still = df.duplicated(QUARTER_KEY, keep=False)
    if still.any():
        print(f"\n  ERROR: {int(still.sum()):,} rows still share {QUARTER_KEY}.")
        raise SystemExit("Stopping - the quarterly de-duplication rule is incomplete.")
    print(f"  {'x'.join(QUARTER_KEY)} is unique: {len(df):,} firm-quarters")

    # Availability date: three months after quarter end, or the actual earnings
    # release date rdq when that is later. Taking the later of the two can only
    # push availability further out, so it cannot introduce look-ahead.
    df["rdq"] = pd.to_datetime(df["rdq"], errors="coerce")
    lagged = df["datadate"] + pd.DateOffset(months=AVAIL_LAG_MONTHS_Q)
    use_rdq = df["rdq"].notna() & (df["rdq"] > lagged)
    df["date_avail"] = lagged.where(~use_rdq, df["rdq"])
    print(f"  date_avail: {int(use_rdq.sum()):,} rows use the actual report date rdq, "
          f"{len(df) - int(use_rdq.sum()):,} use datadate + {AVAIL_LAG_MONTHS_Q}m")

    # CZ drops rows filed more than 6 months after quarter end. We flag instead
    # - see LATE_FILING_MONTHS.
    months_late = ((df["rdq"].dt.year - df["datadate"].dt.year) * 12
                   + (df["rdq"].dt.month - df["datadate"].dt.month))
    df["late_filing"] = df["rdq"].notna() & (months_late > LATE_FILING_MONTHS)
    print(f"  late_filing flagged (rdq > {LATE_FILING_MONTHS}m after quarter end): "
          f"{int(df['late_filing'].sum()):,} rows - CZ would drop these, we keep them")

    df = apply_zero_fill(df, CZ_ZERO_FILL_Q)

    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    out = OUTDIR / "compustat_quarterly_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"  -> {out.name}  {len(df):,} rows x {df.shape[1]} cols  "
          f"({out.stat().st_size / 1024 / 1024:.1f} MB)")
    return df


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("stage1_0_prep_compustat.py")
    print(f"  reading from {RAW}")
    print(f"  writing to   {OUTDIR}")
    print(f"  RUN_CHECKS={RUN_CHECKS}  SAVE_MISSING_FLAGS={SAVE_MISSING_FLAGS}")

    if RUN_CHECKS:
        check_1_duplicates()
        check_2_which_row_has_the_data()
        check_3_industry_codes()
        check_4_zero_fill()
        check_5_dates()
        check_6_identifiers()

    prep_annual()
    prep_quarterly()

    banner("DONE")
    print("Next: stage1_1_build_features.py - the 101 characteristics, with")
    print("CZ's monthly lags translated to annual and permno re-keyed to gvkey.")


if __name__ == "__main__":
    main()
