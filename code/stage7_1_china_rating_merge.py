"""
Stage 7.1 — Split the Chinese score panel in two, and attach the actual Chinese
ISSUER rating to the rating-score half.

Outputs
  output/stage7_1_china_distress_panel.parquet   the three distress scores
  output/stage7_1_china_rating_panel.parquet     the three rating scores + rating

THE RATING IS `Ctcr` (主体评级), the ISSUER rating. NOT `CtcrAdj`, which strips
the +/- modifier — measured: it equals `Ctcr` on only 73.63% of rows, collapsing
AA+ and AA- into AA.

THE COMMON SCALE is the US one: AAA = 1 … C = 21, so the two markets' notches are
the same numbers. `AAA+` and `AAA-` have no US counterpart and both fold into
AAA = 1; both counts are printed. `A-1` is a short-term code and is excluded, as
in `check_china_score_validity.py`. Any other symbol stops the script.

YEAR OFFSET: ZERO, the same as the US side. A rating whose `RatingDate` falls in
calendar year t is lined up with feature `fyear` t. The US construction takes the
rating in force at 31 December of year t and joins it to `fyear` t; both are
concurrent, neither looks forward nor back a year.

THE ANNUAL RULE, per Orgid and year t:
  a. ratings dated in year t exist -> the LOWEST of them, across all agencies
     (lowest rating = highest notch)
  b. none -> carry forward the most recent EARLIER year that had one. Never a
     later year.
  c. stop carrying forward once the issuer has no bond with `Matdt` on or after
     31 December of year t. A blank `Matdt` does not count as outstanding.
  d. otherwise the rating is missing.
The maturity stop applies to the CARRIED values only; a rating dated in year t is
taken at face value whatever the bond master says.

NOTE, recorded not repaired: the Chinese table records NO withdrawal or
termination — a regex for 终止|撤销|撤消|退出|暂停|取消|失效|withdraw|WD over
every symbol and code column matches zero rows. Maturity is therefore the only
available stop, and it is a proxy for "no bond outstanding", not a record of a
withdrawn rating.

Reads   output/stage6_4_china_score_panel.parquet
        data_raw/china/债券及主体评级情况表.csv   (4 columns)
        data_raw/china/债券基本情况表.csv         (4 columns)
Writes  the two panels above.

Runtime: a minute or two, mostly scanning the two large CSVs. Only the needed
columns are read, so peak memory is small.

Run:  python3 code/stage7_1_china_rating_merge.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CH_DIR = PROJECT_DIR / "data_raw" / "china"
OUT_DIR = PROJECT_DIR / "output"

PANEL_IN = OUT_DIR / "stage6_4_china_score_panel.parquet"
# Export both named tables from CSMAR (https://data.csmar.com/) through an
# authorized account. Keep the three-row CSV header and save them in
# data_raw/china/ under these exact names.
RATINGS_CSV = "债券及主体评级情况表.csv"  # issuer and bond rating history
BONDS_CSV = "债券基本情况表.csv"        # bond-to-issuer mapping
DISTRESS_OUT = OUT_DIR / "stage7_1_china_distress_panel.parquet"
RATING_OUT = OUT_DIR / "stage7_1_china_rating_panel.parquet"

# The common scale, identical to stage7_0's. AAA+ and AAA- fold into AAA.
NOTCH = {"AAA+": 1, "AAA": 1, "AAA-": 1,
         "AA+": 2, "AA": 3, "AA-": 4,
         "A+": 5, "A": 6, "A-": 7,
         "BBB+": 8, "BBB": 9, "BBB-": 10,
         "BB+": 11, "BB": 12, "BB-": 13,
         "B+": 14, "B": 15, "B-": 16,
         "CCC+": 17, "CCC": 18, "CCC-": 19, "CC": 20, "C": 21}
NOT_LONG_TERM = {"A-1"}        # short-term code, not on the notch scale

LETTERS = [None, "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB",
           "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-",
           "CC", "C", "D"]

# What the existing same-year construction produced, for the loud check.
EXPECT_ROWS, EXPECT_ISSUERS, EXPECT_FROM = 21726, 4003, 2010


def band_of(notch):
    if notch == 1:
        return "AAA"
    if notch <= 4:
        return "AA"
    if notch <= 7:
        return "A"
    if notch <= 10:
        return "BBB"
    if notch <= 13:
        return "BB"
    if notch <= 16:
        return "B"
    if notch <= 21:
        return "CCC-C"
    return "D"


def unwrap(s):
    """CSMAR writes every value as ="...". Strip the wrapper."""
    return s.astype(str).str.replace(r'^="?|"$', "", regex=True).str.strip()


def csmar(fname, want):
    """Read named columns. Header is line 1; lines 2-3 are label rows."""
    h = pd.read_csv(CH_DIR / fname, nrows=0, encoding="utf-8-sig")
    cols = [c.replace('="', "").replace('"', "").strip() for c in h.columns]
    idx = sorted(cols.index(w) for w in want)
    d = pd.read_csv(CH_DIR / fname, usecols=idx, skiprows=[1, 2], header=0,
                    dtype=str, encoding="utf-8-sig", low_memory=False)
    d.columns = [cols[i] for i in idx]
    for c in d.columns:
        d[c] = unwrap(d[c])
    return d


def blank(s):
    return s.isna() | s.isin(["", "nan", "None"])


def main():
    for p in (DISTRESS_OUT, RATING_OUT):
        if p.exists():
            raise SystemExit(f"STOP: {p.name} already exists. This script never "
                             "overwrites; move or rename it first.")

    # ---- 1. the score panel, split in two ----------------------------------
    print("=" * 78)
    print("STEP 1 — the score panel: drop unscored rows, split, rename gvkey")
    print("=" * 78)
    pan = pd.read_parquet(PANEL_IN)
    dis_cols = [c for c in pan.columns if c.startswith("score_distress_")]
    rat_cols = [c for c in pan.columns if c.startswith("score_rating_")]
    scored = pan[dis_cols + rat_cols].notna().all(axis=1)
    partial = pan[dis_cols + rat_cols].notna().any(axis=1) & ~scored
    print(f"  rows in panel      : {len(pan):,}")
    print(f"  dropped, unscored  : {int((~scored & ~partial).sum()):,}")
    print(f"  partially scored   : {int(partial.sum()):,}  (also dropped)")

    # The column named gvkey holds the CSMAR Orgid. Rename it so nothing
    # downstream mistakes it for a Compustat key. The index is reset so the
    # as-of merge below can be joined back by position without surprises.
    pan = pan[scored].rename(columns={"gvkey": "Orgid"}).reset_index(drop=True)
    pan["fyear"] = pan.fyear.astype(int)
    keep = ["Orgid", "datadate", "fyear", "ind_cn"]
    print(f"  rows scored        : {len(pan):,}   distinct Orgid "
          f"{pan.Orgid.nunique():,}   fyear {pan.fyear.min()}-{pan.fyear.max()}")

    # pct_ columns are within-panel percentiles, uniform 0-100 by construction
    # and never comparable across markets. They are not carried forward.
    dis = pan[keep + dis_cols].copy()
    dis.to_parquet(DISTRESS_OUT, index=False)
    print(f"  wrote {DISTRESS_OUT.name}: {len(dis):,} rows x {len(dis.columns)} cols")
    rat = pan[keep + rat_cols].copy()

    # ---- 2. the raw rating table -------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 2 — the issuer ratings (Ctcr), decoded to the common scale")
    print("=" * 78)
    rt = csmar(RATINGS_CSV, ["Liscd", "Sctcd", "RatingDate", "Ctcr"])
    print(f"  rating rows           : {len(rt):,}")
    have = ~blank(rt.Ctcr)
    print(f"  with an issuer rating : {int(have.sum()):,} ({have.mean()*100:.2f}%)")
    unknown = set(rt.loc[have, "Ctcr"].unique()) - set(NOTCH) - NOT_LONG_TERM
    if unknown:
        raise SystemExit(f"STOP: Ctcr values not on the notch scale and not "
                         f"declared excluded: {sorted(unknown)}. NOTHING WRITTEN.")
    print(f"  excluded short-term 'A-1'      : {int((rt.Ctcr == 'A-1').sum()):,}")
    print(f"  AAA+ folded into AAA (notch 1) : {int((rt.Ctcr == 'AAA+').sum()):,}")
    print(f"  AAA- folded into AAA (notch 1) : {int((rt.Ctcr == 'AAA-').sum()):,}")
    print(f"  AAA itself                     : {int((rt.Ctcr == 'AAA').sum()):,}")
    rt["notch"] = rt.Ctcr.map(NOTCH)
    rt["year"] = pd.to_datetime(rt.RatingDate, errors="coerce").dt.year
    print(f"  RatingDate parse failures      : {int(rt.year.isna().sum()):,}")

    # ---- 3. LOUD CHECK, on the construction as it stood --------------------
    print("\n" + "=" * 78)
    print("STEP 3 — LOUD CHECK: reproduce the existing same-year construction")
    print("=" * 78)
    # Exactly as check_china_score_validity.py:429-445 — the bridge de-duplicated
    # on all three columns (NOT on the bond key, so the ambiguous codes are still
    # in), same-year only, no carry forward, fyear >= 2010.
    bm_raw = csmar(BONDS_CSV, ["Liscd", "Sctcd", "Orgid"]).drop_duplicates()
    chk = rt.merge(bm_raw, on=["Liscd", "Sctcd"], how="left")
    chk = chk[chk.notch.notna() & chk.Orgid.notna() & chk.year.notna()]
    rp = chk.groupby(["Orgid", "year"])["notch"].max().reset_index()
    rp["year"] = rp.year.astype(int)
    m = rat.merge(rp, left_on=["Orgid", "fyear"], right_on=["Orgid", "year"],
                  how="inner")
    m = m[m.fyear >= EXPECT_FROM]
    print(f"  same-year, fyear >= {EXPECT_FROM}: {len(m):,} issuer-years, "
          f"{m.Orgid.nunique():,} issuers")
    print(f"  expected                : {EXPECT_ROWS:,} issuer-years, "
          f"{EXPECT_ISSUERS:,} issuers")
    if len(m) != EXPECT_ROWS or m.Orgid.nunique() != EXPECT_ISSUERS:
        raise SystemExit("STOP: the existing construction was not reproduced. "
                         "NOTHING FURTHER WRITTEN.")
    print("  PASSED")

    # ---- 4. the bridge, with the ambiguous bond codes dropped --------------
    print("\n" + "=" * 78)
    print("STEP 4 — bridge (Liscd, Sctcd) -> Orgid, dropping ambiguous bond codes")
    print("=" * 78)
    bm = csmar(BONDS_CSV, ["Liscd", "Sctcd", "Orgid", "Matdt"])
    pairs = bm[["Liscd", "Sctcd", "Orgid"]].drop_duplicates()
    n_orgid = pairs.groupby(["Liscd", "Sctcd"]).Orgid.nunique()
    ambiguous = set(n_orgid[n_orgid > 1].index)
    print(f"  (Liscd, Sctcd) pairs carrying more than one Orgid: {len(ambiguous):,}")
    is_amb = pd.Series(list(zip(rt.Liscd, rt.Sctcd)), index=rt.index).isin(ambiguous)
    print(f"  rating rows on those bond codes, DROPPED: {int(is_amb.sum()):,}  "
          f"(of which with an issuer rating: {int((is_amb & have).sum()):,})")

    bridge = pairs[~pd.Series(list(zip(pairs.Liscd, pairs.Sctcd)),
                              index=pairs.index).isin(ambiguous)]
    rtb = rt[~is_amb].merge(bridge, on=["Liscd", "Sctcd"], how="left")
    print(f"  rating rows bridged to an Orgid: {int(rtb.Orgid.notna().sum()):,} "
          f"of {len(rtb):,}")

    ok = rtb.notch.notna() & rtb.Orgid.notna() & rtb.year.notna()
    # (a) the LOWEST rating dated in the year = the HIGHEST notch.
    ann = rtb[ok].groupby(["Orgid", "year"])["notch"].max().reset_index()
    ann["year"] = ann.year.astype(int)
    print(f"  (Orgid, year) cells with a rating: {len(ann):,}, "
          f"{ann.Orgid.nunique():,} issuers")

    # ---- 5. the annual rule -------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 5 — the annual rule: same year, else carry forward, maturity stop")
    print("=" * 78)
    # (a)+(b) in one as-of merge: the most recent year AT OR BEFORE fyear. Which
    # of the two it was is then read off src_year.
    left = rat[["Orgid", "fyear"]].reset_index().sort_values("fyear")
    right = ann.rename(columns={"year": "src_year", "notch": "notch_src"}) \
               .sort_values("src_year")
    cf = pd.merge_asof(left, right, left_on="fyear", right_on="src_year",
                       by="Orgid", direction="backward")
    cf = cf.set_index("index").sort_index()
    rat["notch"] = cf.notch_src
    rat["src_year"] = cf.src_year

    same_year = rat.notch.notna() & (rat.src_year == rat.fyear)
    carried = rat.notch.notna() & (rat.src_year < rat.fyear)

    # (c) the maturity stop, applied to CARRIED values only.
    blank_mat = blank(bm.Matdt)
    print(f"  bonds with a blank Matdt (NOT counted as outstanding): "
          f"{int(blank_mat.sum()):,} of {len(bm):,}")
    mat = bm[~blank_mat].copy()
    mat["Matdt"] = pd.to_datetime(mat.Matdt, errors="coerce")
    print(f"  Matdt parse failures: {int(mat.Matdt.isna().sum()):,}  (dropped)")
    last_mat = mat.dropna(subset=["Matdt"]).groupby("Orgid").Matdt.max()
    rat["last_matdt"] = rat.Orgid.map(last_mat)
    year_end = pd.to_datetime(dict(year=rat.fyear, month=12, day=31))
    outstanding = rat.last_matdt.notna() & (rat.last_matdt >= year_end)
    stopped = carried & ~outstanding
    rat.loc[stopped, ["notch", "src_year"]] = np.nan

    # (d) what is left: an issuer with ratings only after fyear, or none at all.
    ever = set(ann.Orgid.unique())
    no_rating = rat.notch.isna()
    only_later = no_rating & rat.Orgid.isin(ever) & ~stopped
    never = no_rating & ~rat.Orgid.isin(ever)

    print(f"\n  scored rows                    : {len(rat):,}")
    print(f"    same-year rating             : {int(same_year.sum()):,} "
          f"({same_year.mean()*100:.2f}%)")
    print(f"    carried forward, kept        : {int((carried & outstanding).sum()):,} "
          f"({(carried & outstanding).mean()*100:.2f}%)")
    print(f"    carried, stopped by maturity : {int(stopped.sum()):,} "
          f"({stopped.mean()*100:.2f}%)")
    print(f"    only LATER ratings exist     : {int(only_later.sum()):,} "
          f"({only_later.mean()*100:.2f}%)")
    print(f"    never rated at any date      : {int(never.sum()):,} "
          f"({never.mean()*100:.2f}%)")
    print(f"    rows with a rating           : {int(rat.notch.notna().sum()):,} "
          f"({rat.notch.notna().mean()*100:.2f}%)")
    gap = (rat.fyear - rat.src_year)[carried & outstanding]
    if len(gap):
        print(f"  carry distance, years: min {int(gap.min())}, median "
              f"{gap.median():.0f}, max {int(gap.max())}")

    # ---- 6. the within-year rule diagnostic --------------------------------
    print("\n" + "=" * 78)
    print("STEP 6 — DIAGNOSTIC: 'lowest in the year' against 'last of the year'")
    print("=" * 78)
    # The US side takes the rating IN FORCE at 31 December, i.e. the LAST action
    # of the year; this side takes the LOWEST. Where several ratings share the
    # latest date the worst of them is taken, so the comparison is deterministic.
    d = rtb[ok].copy()
    d["dt"] = pd.to_datetime(d.RatingDate, errors="coerce")
    d = d.sort_values(["Orgid", "year", "dt", "notch"])
    last = d.groupby(["Orgid", "year"]).tail(1)[["Orgid", "year", "notch"]] \
            .rename(columns={"notch": "notch_last"})
    last["year"] = last.year.astype(int)
    both = ann.merge(last, on=["Orgid", "year"], how="inner")
    sy = rat.loc[same_year, ["Orgid", "fyear"]].merge(
        both, left_on=["Orgid", "fyear"], right_on=["Orgid", "year"], how="inner")
    diff = sy.notch != sy.notch_last
    print(f"  same-year scored rows compared : {len(sy):,}")
    print(f"  lowest differs from last       : {int(diff.sum()):,} "
          f"({diff.mean()*100:.2f}%)")
    if diff.any():
        print(f"  where they differ, mean notch gap (lowest - last): "
              f"{(sy.notch - sy.notch_last)[diff].mean():+.3f}")

    # ---- 7. letters, and write ---------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 7 — letters and bands")
    print("=" * 78)
    rat["rating_source"] = np.where(same_year, "same_year",
                                    np.where(carried & outstanding, "carried", None))
    rat["notch"] = rat.notch.round().astype("Int64")
    rat["src_year"] = rat.src_year.round().astype("Int64")
    rat["letter"] = [LETTERS[int(n)] if pd.notna(n) else None for n in rat.notch]
    rat["band"] = [band_of(int(n)) if pd.notna(n) else None for n in rat.notch]

    r = rat[rat.notch.notna()]
    print(f"  {'notch':>5} {'letter':<7} {'band':<7} {'n':>8} {'share of rated':>15}")
    for n, k in r.groupby(r.notch).size().items():
        print(f"  {int(n):>5} {LETTERS[int(n)]:<7} {band_of(int(n)):<7} {k:>8,} "
              f"{k/len(r)*100:>14.2f}%")
    print("\n  band distribution, rated rows")
    for b, k in r.groupby(r.band).size().sort_values(ascending=False).items():
        print(f"    {b:<7} {k:>8,} {k/len(r)*100:>7.2f}%")

    out = rat[["Orgid", "datadate", "fyear", "ind_cn", "notch", "letter", "band",
               "rating_source", "src_year"] + rat_cols]
    out.to_parquet(RATING_OUT, index=False)
    print(f"\nWrote {RATING_OUT.name}: {len(out):,} rows x {len(out.columns)} "
          f"columns, {out.Orgid.nunique():,} distinct Orgid")
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
