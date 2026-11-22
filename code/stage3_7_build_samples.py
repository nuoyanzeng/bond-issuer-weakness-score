"""
Stage 3.7 — Two labelled samples built from the features panel.

  SAMPLE 1  distress   output/stage3_7_distress_sample_L3.parquet
  SAMPLE 2  rating     output/stage3_7_rating_sample.parquet

SAMPLE 1 aligns on `date_avail`, never on `fyear`: it predicts a dated event, and
three gvkeys changed fiscal year-end and carry two statements inside one fyear.

SAMPLE 2 aligns on `fyear` deliberately — a statement is matched to the rating of
its OWN year (D-42, amended). It validates rather than predicts, so there is no
event date to stand off from. The consequence of the fiscal-year-end change is
that those three gvkeys' paired rows share one year's rating; that is correct for
a state panel, but the pair is not two independent observations.

No modelling here. Only the labels, and the availability guard on sample 1.

Run:  python3 code/stage3_7_build_samples.py
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

# Availability lag of the input panel, in months (D-41). 3 is the main setting;
# 0 is the no-guard comparison. ONLY filenames depend on it — no rule below does.
LAG_MONTHS = 3

PANEL_IN = OUT_DIR / ("stage3_6_features_panel.parquet" if LAG_MONTHS == 3
                      else f"stage3_6_features_panel_L{LAG_MONTHS}.parquet")
EVENTS_IN = OUT_DIR / "stage3_0_distress_events.parquet"
RATINGS_IN = OUT_DIR / "stage3_1_rating_panel.parquet"

DISTRESS_OUT = OUT_DIR / f"stage3_7_distress_sample_L{LAG_MONTHS}.parquet"
# The rating sample has no lag suffix: its alignment is concurrent and it applies
# no guard, so L reaches it only through the first-offering cut.
RATING_OUT = OUT_DIR / "stage3_7_rating_sample.parquet"

KEYS = ["issuer_id", "gvkey", "fyear", "datadate", "date_avail"]
# Only these two features are shown in the printouts, to keep rows readable.
SHOW = ["BookLeverage", "quick"]   # was currat, dropped for quick by D-58

# Which samples this run builds. Whatever is False is not read, not built and
# not opened for writing.
BUILD_DISTRESS = False
BUILD_RATING = False

# The rating panel starts here (D-39). A feature row with an earlier fiscal year
# could never match a rating: that is a coverage limit, not a matching failure,
# and the two are counted separately below.
RATING_PANEL_START = 1993


def show(df, extra):
    """Print a readable slice: the keys, the named columns, two features."""
    cols = KEYS + extra + [c for c in SHOW if c in df.columns]
    return df[cols].to_string(index=False)


def build_distress(panel, events, master):
    print("=" * 80)
    print("SAMPLE 1 — DISTRESS")
    print("=" * 80)

    # STEP 1 — the research population, applied FIRST. A firm is not a bond
    # issuer before it has issued, so statements available earlier are outside
    # the population. A null first_offering_date fails the test, so those
    # issuers are dropped: we cannot establish they were issuers at any date.
    before = panel.merge(master, on="issuer_id", how="left")
    n_before = len(before)
    d_all = set(events.issuer_id)
    no_fo = before[before.first_offering_date.isna()]
    print(f"  rows before step 1             : {n_before:,}   "
          f"issuers {before.issuer_id.nunique():,}")
    print(f"  issuers with NO first_offering_date: {no_fo.issuer_id.nunique():,} "
          f"({len(no_fo):,} rows), of which distress "
          f"{len(set(no_fo.issuer_id) & d_all):,} — dropped, null fails the test")

    in_pop = before.first_offering_date.notna() & (
        before.date_avail >= before.first_offering_date)
    pop = before[in_pop].copy()
    lost_pop = set(before.issuer_id) - set(pop.issuer_id)
    print(f"\n  STEP 1 — keep date_avail >= first_offering_date")
    print(f"      rows after step 1          : {len(pop):,}   "
          f"(removed {n_before - len(pop):,}, "
          f"{(n_before-len(pop))/n_before*100:.1f}%)")
    print(f"      issuers left with ZERO rows: {len(lost_pop):,}")
    print(f"          of which DISTRESS      : {len(lost_pop & d_all):,} "
          "— they leave the sample entirely")
    print(f"      issuers remaining          : {pop.issuer_id.nunique():,}")

    m = pop.merge(events, on="issuer_id", how="left")
    is_d = m.first_distress_date.notna()
    print(f"\n  STEP 2 — merge the outcome, within the surviving rows only")
    print(f"      rows entering step 2       : {len(m):,}")
    print(f"      distressed issuers         : {m[is_d].issuer_id.nunique():,}")
    print(f"      never-distressed issuers   : {m[~is_d].issuer_id.nunique():,}")

    # STEP 2 — the guard. A statement available on or after the distress date
    # cannot be used to predict it. Dropped, never zero-labelled.
    drop_mask = is_d & (m.date_avail >= m.first_distress_date)
    dropped = m[drop_mask].copy()
    kept = m[~drop_mask].copy()
    print(f"\n  guard: dropped date_avail >= first_distress_date: {len(dropped):,}")
    print(f"  rows kept                      : {len(kept):,}")

    # STEP 3 — absorbing state. Nothing after the event may survive; assert it.
    after = kept.first_distress_date.notna() & (
        kept.date_avail >= kept.first_distress_date)
    if after.any():
        raise SystemExit(f"STOP: {int(after.sum()):,} rows survive dated at or "
                         "after the distress event.")
    print("  absorbing-state assertion: passed, 0 rows after the event")

    # STEP 4 — the label. y=1 on the single latest surviving statement of each
    # distressed issuer: the last thing knowable before it defaulted.
    kept["y"] = 0
    d_rows = kept[kept.first_distress_date.notna()]
    last_idx = d_rows.groupby("issuer_id")["date_avail"].idxmax()
    kept.loc[last_idx, "y"] = 1

    all_d = set(m[is_d].issuer_id)
    surv_d = set(d_rows.issuer_id)
    lost = all_d - surv_d
    print(f"\n  y=1 rows                       : {int(kept.y.sum()):,}")
    print(f"  distressed issuers with a y=1   : {len(surv_d):,}")
    print(f"  distressed issuers LOST to guard: {len(lost):,}")
    print(f"  issuers in the sample          : {kept.issuer_id.nunique():,}")

    # one y=1 per distressed issuer, and none for anyone else
    if int(kept.y.sum()) != len(surv_d):
        raise SystemExit("STOP: y=1 count does not equal surviving distressed issuers.")
    if kept[kept.y == 1].first_distress_date.isna().any():
        raise SystemExit("STOP: a never-distressed issuer carries y=1.")

    ones = kept[kept.y == 1].copy()
    ones["gap_years"] = (ones.first_distress_date - ones.date_avail).dt.days / 365.25

    def gap_table(q, label):
        print(f"\n  {label}")
        print(f"      min {q.min():.2f} | p25 {q.quantile(.25):.2f} | median "
              f"{q.median():.2f} | p75 {q.quantile(.75):.2f} | max {q.max():.2f}")
        for lo, hi, lab in [(0, 1, "<1y"), (1, 2, "1-2y"), (2, 3, "2-3y"),
                            (3, 5, "3-5y"), (5, 1e9, "5y+")]:
            n = int(((q >= lo) & (q < hi)).sum())
            print(f"      {lab:<6}{n:>5}  ({n/len(q)*100:4.1f}%)")

    # NO K FILTER on this sample (D-42 as amended). The y=1 label stands
    # however large the gap; the full distribution is kept for robustness work.
    gap_table(ones.gap_years, f"FULL gap distribution, no K applied, "
                              f"y=1 count {len(ones):,}:")

    # Two denominators, both printed, because they are easy to confuse: only a
    # subset of FISD's distress issuers ever reached the features panel, and the
    # losses below are counted against that subset, not against the whole.
    d_panel = set(before.issuer_id) & d_all
    n_lost = len(lost_pop & d_all) + len(lost)
    print(f"\n  DISTRESS ISSUERS LOST, BY STAGE:")
    print(f"      step 1, no post-issuance financials : {len(lost_pop & d_all):,}")
    print(f"      the guard, no row before the event  : {len(lost):,}")
    print(f"      total lost                          : {n_lost:,} of "
          f"{len(d_panel):,} (distress issuers present in the features panel)")
    print(f"      for reference, distress issuers in stage3_0 overall: "
          f"{len(d_all):,} — the other {len(d_all) - len(d_panel):,} never "
          "reached the features panel and are not counted above")
    print(f"\n  FINAL: rows {len(kept):,}   issuers {kept.issuer_id.nunique():,}   "
          f"y=1 {int(kept.y.sum()):,}")

    print("\n  THE 5 LARGEST GAPS, kept deliberately:")
    top = ones.nlargest(5, "gap_years")
    print(top[["issuer_id", "date_avail", "first_distress_date", "gap_years"]]
          .to_string(index=False))

    return kept, dropped, ones, lost


def print_distress_samples(kept, dropped, ones):
    print("\n" + "=" * 80)
    print("(a) THREE DISTRESSED ISSUERS IN FULL — kept rows, then guard-dropped rows")
    print("=" * 80)
    # One whose event fell early in the calendar year, so the guard actually bit.
    early = (dropped[dropped.first_distress_date.dt.month <= 3]
             .issuer_id.value_counts().index.tolist())
    ordinary = [i for i in ones.issuer_id.tolist() if i not in early]
    picks = ([early[0]] if early else []) + ordinary[:2]
    for iid in picks:
        k = kept[kept.issuer_id == iid].sort_values("date_avail")
        d = dropped[dropped.issuer_id == iid].sort_values("date_avail")
        dd = (k.first_distress_date.iloc[0] if len(k)
              else d.first_distress_date.iloc[0])
        print(f"\n  --- issuer {int(iid)}  first_distress_date {dd.date()} "
              f"({len(k)} kept, {len(d)} dropped) ---")
        if len(k):
            print("  KEPT:")
            print(show(k, ["first_offering_date", "first_distress_date", "y"]))
        if len(d):
            print("  DROPPED BY GUARD (date_avail >= distress date):")
            print(show(d.assign(y="dropped"), ["first_offering_date", "first_distress_date", "y"]))

    print("\n" + "=" * 80)
    print("(b) TWO NEVER-DISTRESSED ISSUERS, 5 rows each — all y=0")
    print("=" * 80)
    never = kept[kept.first_distress_date.isna()].issuer_id.unique()[:2]
    for iid in never:
        g = kept[kept.issuer_id == iid].sort_values("date_avail").head(5)
        print(f"\n  --- issuer {int(iid)} (never distressed) ---")
        print(show(g, ["first_offering_date", "first_distress_date", "y"]))

    print("\n" + "=" * 80)
    print("(c) THE EXTREMES OF THE GAP")
    print("=" * 80)
    for lab, row in [("LARGEST gap", ones.loc[ones.gap_years.idxmax()]),
                     ("SMALLEST gap", ones.loc[ones.gap_years.idxmin()])]:
        iid = row.issuer_id
        g = kept[kept.issuer_id == iid].sort_values("date_avail").tail(4)
        print(f"\n  --- {lab}: issuer {int(iid)}, gap {row.gap_years:.2f} years, "
              f"distress {row.first_distress_date.date()} (last 4 kept rows) ---")
        print(show(g, ["first_offering_date", "first_distress_date", "y"]))


def build_rating(panel, ratings, master):
    """Sample 2 — CONCURRENT alignment (D-42 as amended 2026-08-26).

    The rating panel is a STATE panel and this sample is a validation tool, not
    a prediction table: the question is what the agencies thought of the firm at
    the SAME time as the statement describes it. So a feature row of fiscal year
    t is matched to the rating of year t. No forward search, no lag, and hence
    no staleness filter — K existed only to cut away the forward search's absurd
    pairings and has no work left to do.

    No availability guard and no absorbing state either: a rating is a
    contemporaneous opinion, not a future event, and a grade may move 1 -> 0 as
    freely as 0 -> 1.
    """
    print("\n" + "=" * 80)
    print("SAMPLE 2 — RATING (validation state panel, CONCURRENT alignment)")
    print("=" * 80)

    # STEP 1 — the same research population as sample 1. A firm is not a bond
    # issuer before it has issued; a null first_offering_date fails the test.
    before = panel.merge(master, on="issuer_id", how="left")
    n_before = len(before)
    print(f"  rows before the population cut : {n_before:,}   "
          f"issuers {before.issuer_id.nunique():,}")
    print(f"  rows with NO first_offering_date (dropped, null fails the test): "
          f"{int(before.first_offering_date.isna().sum()):,}")
    in_pop = before.first_offering_date.notna() & (
        before.date_avail >= before.first_offering_date)
    pop = before[in_pop].copy()
    print(f"\n  STEP 1 — keep date_avail >= first_offering_date")
    print(f"      rows after the cut         : {len(pop):,}   "
          f"(removed {n_before - len(pop):,}, "
          f"{(n_before-len(pop))/n_before*100:.1f}%)")
    print(f"      issuers remaining          : {pop.issuer_id.nunique():,}")

    # STEP 2 — the same-year join. spec_grade NA is dropped BEFORE the join, so
    # a row facing only an NA rating counts as unmatched rather than as a match
    # with a missing label. (issuer_id, year) is unique in the rating panel, so
    # this is many-to-one and cannot multiply rows; asserted.
    rv = ratings.dropna(subset=["spec_grade"])[["issuer_id", "year", "spec_grade"]]
    print(f"\n  STEP 2 — join (issuer_id, fyear) = (issuer_id, year)")
    print(f"      rating panel: {len(ratings):,} rows, non-NA spec_grade "
          f"{len(rv):,}, issuers {ratings.issuer_id.nunique():,}")
    if rv.duplicated(["issuer_id", "year"]).any():
        raise SystemExit("STOP: (issuer_id, year) is not unique in the rating panel.")

    j = pop.merge(rv, left_on=["issuer_id", "fyear"],
                  right_on=["issuer_id", "year"], how="left")
    if len(j) != len(pop):
        raise SystemExit(f"STOP: the join changed the row count, {len(pop):,} "
                         f"-> {len(j):,}.")

    hit = j.spec_grade.notna()
    un = j[~hit]
    # Two causes of a non-match, kept apart: a fiscal year before the panel
    # begins could never have matched, which is coverage; a fiscal year inside
    # the panel that still finds nothing is an issuer simply unrated that year.
    n_pre = int((un.fyear < RATING_PANEL_START).sum())
    print(f"      matched to a SAME-YEAR rating : {int(hit.sum()):,} of "
          f"{len(j):,} ({hit.sum()/len(j)*100:.1f}%)")
    print(f"      unmatched, dropped            : {len(un):,}")
    print(f"          fyear < {RATING_PANEL_START} (panel start, D-39): {n_pre:,}"
          "   <- coverage limit, could never match")
    print(f"          fyear >= {RATING_PANEL_START}, issuer had no valid rating "
          f"that year: {len(un)-n_pre:,}")

    j = j[hit].copy()
    # Every kept row must sit on its OWN year. This is the whole point of the
    # rebuild, so it is asserted rather than trusted.
    if not (j.fyear == j.year).all():
        raise SystemExit("STOP: a kept row is matched to a year other than its fyear.")
    if j.spec_grade.isna().any():
        raise SystemExit("STOP: a kept row carries a missing spec_grade.")
    j = j.drop(columns=["year"])  # identical to fyear; keeping it invites confusion

    n1 = int((j.spec_grade == 1).sum()); n0 = int((j.spec_grade == 0).sum())
    print(f"\n  FINAL: rows {len(j):,}   issuers {j.issuer_id.nunique():,}")
    print(f"  spec_grade 0/1 : {n0:,} ({n0/len(j)*100:.1f}%) / "
          f"{n1:,} ({n1/len(j)*100:.1f}%)")
    print(f"  fyear range    : {int(j.fyear.min())} to {int(j.fyear.max())}")
    return j


def print_rating_overlap(j):
    """How many issuers carry BOTH labels — the rows validation can run on."""
    print("\n  OVERLAP WITH SAMPLE 1 (the distress sample on disk):")
    if not DISTRESS_OUT.exists():
        print("      distress sample not on disk; overlap not computed.")
        return
    ds = pd.read_parquet(DISTRESS_OUT, columns=["issuer_id", "y"])
    ds["issuer_id"] = ds.issuer_id.astype("Int64")
    a, b = set(ds.issuer_id), set(j.issuer_id)
    ones = set(ds[ds.y == 1].issuer_id)
    print(f"      distress-sample issuers        : {len(a):,}")
    print(f"      rating-sample issuers          : {len(b):,}")
    print(f"      in BOTH                        : {len(a & b):,} "
          "<- validation runs on these")
    print(f"      rating-sample issuers NOT in the distress sample: {len(b - a):,}")
    print(f"      distress issuers with y=1 that are also rated: "
          f"{len(ones & b):,} of {len(ones):,}")


def print_rating_samples(j):
    print("\n" + "=" * 80)
    print("(d) TWO ISSUERS IN FULL — check each statement sits on its OWN year")
    print("=" * 80)
    # Prefer an issuer whose grade goes 0 -> 1 over time.
    # astype(bool): the aggregate comes back as Int64 on a nullable column, and
    # a 0/1 series would index by label rather than as a mask.
    chg = (j.sort_values("fyear").groupby("issuer_id")["spec_grade"]
             .agg(lambda s: (s.iloc[0] == 0) and (s.iloc[-1] == 1))).astype(bool)
    switchers = chg[chg].index.tolist()
    counts = j.issuer_id.value_counts()
    pick = [i for i in switchers if counts.get(i, 0) >= 6][:1]
    pick += [i for i in counts.index if i not in pick][:1]
    for iid in pick:
        g = j[j.issuer_id == iid].sort_values("fyear").head(12)
        lab = " (spec_grade goes 0 -> 1)" if iid in switchers else ""
        print(f"\n  --- issuer {int(iid)}{lab} ---")
        print(g[["issuer_id", "gvkey", "fyear", "datadate", "date_avail",
                 "first_offering_date", "spec_grade"]].to_string(index=False))


def main():
    # This script never overwrites a sample it is asked to build — either one.
    # To rebuild, move or rename the existing file first, deliberately.
    if BUILD_DISTRESS and DISTRESS_OUT.exists():
        raise SystemExit(f"STOP: {DISTRESS_OUT.name} already exists; this script "
                         "never overwrites a sample. Move or rename it first.")
    if BUILD_RATING and RATING_OUT.exists():
        raise SystemExit(f"STOP: {RATING_OUT.name} already exists; this script "
                         "never overwrites a sample. Move or rename it first.")

    panel = pd.read_parquet(PANEL_IN)
    panel["issuer_id"] = panel.issuer_id.astype("Int64")
    ratings = pd.read_parquet(RATINGS_IN)
    ratings["issuer_id"] = ratings.issuer_id.astype("Int64")

    master = pd.read_parquet(OUT_DIR / "stage3_2_issuer_master.parquet",
                            columns=["issuer_id", "first_offering_date"])
    master["issuer_id"] = master.issuer_id.astype("Int64")

    if BUILD_DISTRESS:
        events = pd.read_parquet(EVENTS_IN,
                                 columns=["issuer_id", "first_distress_date"])
        events["issuer_id"] = events.issuer_id.astype("Int64")
        kept, dropped, ones, lost = build_distress(panel, events, master)
        kept.sort_values(["issuer_id", "gvkey", "date_avail"]).reset_index(drop=True) \
            .to_parquet(DISTRESS_OUT, index=False)
        print(f"\nWrote {DISTRESS_OUT.name}: {len(kept):,} rows x "
              f"{len(kept.columns)} columns")
        print_distress_samples(kept, dropped, ones)
    else:
        print("BUILD_DISTRESS is False: sample 1 is left untouched on disk.")

    if BUILD_RATING:
        j = build_rating(panel, ratings, master)
        j.sort_values(["issuer_id", "gvkey", "fyear"]).reset_index(drop=True) \
            .to_parquet(RATING_OUT, index=False)
        print(f"\nWrote {RATING_OUT.name}: {len(j):,} rows x {len(j.columns)} columns")
        print_rating_overlap(j)
        print_rating_samples(j)
    else:
        print("BUILD_RATING is False: sample 2 is left untouched on disk.")
    print("\nNo modelling. Labels and the availability guard only.")


if __name__ == "__main__":
    main()
