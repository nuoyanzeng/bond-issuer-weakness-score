"""
Stage 7.0 — Attach the ACTUAL US rating notch to the US rating-score panel.

This is `code/stage3_1_rating_panel.py` copied, with ONE change: the full notch
is kept instead of being collapsed to a 0/1 speculative flag. Everything else is
deliberately identical, because the `y` in the score panel came from that
construction and the two must agree row for row:

  - the rating in force at 31 DECEMBER of the year (most recent action at or
    before it), per (issue, agency)
  - the same year offset: rating year t lines up with fyear t, concurrently
  - NR and SUSP END a series and are not carried past it
  - default codes (D, SD, DD, DDD) are live speculative ratings, not terminators
  - across agencies and bonds of one issuer: the WORST rating that year

THE COMMON SCALE. `fisd_code.description` holds a per-agency integer ordinal,
stored as text, and the four agencies agree notch by notch over AAA = 1 … C = 21
(verified before this script was written). Above 21 they diverge and also encode
things that are not ratings, so:

    1 … 21          the letter scale, AAA = 1 … C = 21, used as is
    D, SD, DD, DDD  all mapped to 22, one default notch
    NR, SUSP        terminators, not ratings (26 and 27 in the file)

ONE JUDGEMENT CALL, STATED. A symbol that is NOT in its agency's official
`fisd_code` list is SKIPPED, not stopped on — that is what `stage3_1` does and
what D-39 records (203 such rows: Fitch `(EXP)` suffixes, Moody's `P-1`, `0`,
`NAV`, four combined S&P strings). Stopping on them would halt on rows the
existing construction deliberately treats as missing, and the panel would then
not reproduce `y`. A symbol that IS in the official list but cannot be placed on
the common scale STOPS the script and is printed. Every skipped symbol is
printed with its count.

Reads   data_raw/us/fisd_ratings.parquet
        data_raw/us/fisd_code.parquet
        data_raw/us/fisd_mergedissue.parquet          (issue -> issuer only)
        output/stage5_1_score_panel_rating_L3_K12.parquet
Writes  output/stage7_0_us_rating_panel.parquet

Runtime: a few minutes, mostly the 4.7m-row as-of merge. Nothing is held in
memory beyond a few hundred MB.

Run:  python3 code/stage7_0_us_rating_merge.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data_raw" / "us"
OUT_DIR = PROJECT_DIR / "output"

RATINGS_IN = RAW_DIR / "fisd_ratings.parquet"
CODE_IN = RAW_DIR / "fisd_code.parquet"
ISSUE_IN = RAW_DIR / "fisd_mergedissue.parquet"
SCORES_IN = OUT_DIR / "stage5_1_score_panel_rating_L3_K12.parquet"
PANEL_OUT = OUT_DIR / "stage7_0_us_rating_panel.parquet"

# The four agencies. SNP is dropped: it is not in fisd_code. (stage3_1)
AGENCIES = ["FR", "MR", "SPR", "DPR"]
CODE_TYPE = {"SPR": "SP_RATING", "MR": "MOODY_RATING",
             "FR": "FITCH_RATING", "DPR": "DP_RATING"}

TERMINATORS = {"NR", "SUSP"}          # end a series, exactly as in stage3_1
DEFAULT_CODES = {"D", "SD", "DD", "DDD"}
DEFAULT_NOTCH = 22                    # one notch below C = 21
SPEC_FROM = 11                        # BB+ and below is speculative grade

YEAR_MIN, YEAR_MAX = 1993, 2026       # the panel's year range (D-39)

# Notch -> letter, on the common scale. Index 0 is unused.
LETTERS = [None, "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB",
           "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-",
           "CC", "C", "D"]


def band_of(notch):
    """Coarse rating band from the notch."""
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


def build_notch_map(codes):
    """symbol -> notch on the common scale, per agency. Stops on the undecodable."""
    out, bad = {}, []
    for agency, ctype in CODE_TYPE.items():
        sub = codes[codes.type == ctype]
        m = {}
        for sym, desc in zip(sub.code, sub.description):
            ordinal = pd.to_numeric(desc, errors="coerce")
            if sym in TERMINATORS:
                continue                              # handled separately
            if sym in DEFAULT_CODES:
                m[sym] = DEFAULT_NOTCH
            elif pd.notna(ordinal) and 1 <= int(ordinal) <= 21:
                m[sym] = int(ordinal)
            else:
                # In the official list, but not placeable on the common scale.
                bad.append((agency, sym, desc))
        out[agency] = m
    if bad:
        print("\nSTOP: symbols in an agency's official fisd_code list that cannot "
              "be placed on the common scale:")
        for a, s, d in bad:
            print(f"    {a:<5} {str(s):<12} description={d!r}")
        raise SystemExit("Nothing written.")
    return out


def main():
    if PANEL_OUT.exists():
        raise SystemExit(f"STOP: {PANEL_OUT.name} already exists. This script "
                         "never overwrites; move or rename it first.")

    # ---- 1. load and decode ------------------------------------------------
    print("=" * 78)
    print("STEP 1 — load, decode to the common notch scale")
    print("=" * 78)
    r = pd.read_parquet(RATINGS_IN,
                        columns=["issue_id", "rating_type", "rating_date", "rating"])
    codes = pd.read_parquet(CODE_IN)
    notch_map = build_notch_map(codes)
    official = {a: set(codes[codes.type == t].code) for a, t in CODE_TYPE.items()}

    print(f"  rating rows in                 : {len(r):,}")
    print(f"  dropped rating_type == 'SNP'   : {int((r.rating_type == 'SNP').sum()):,}")
    r = r[r.rating_type.isin(AGENCIES)].copy()
    r["issue_id"] = r["issue_id"].astype("Int64")     # Int64 before any merge (D-04)
    print(f"  kept                           : {len(r):,} rows, agencies "
          f"{sorted(r.rating_type.unique())}")
    print("  common scale, per agency: " +
          ", ".join(f"{a} {len(m)} symbols" for a, m in notch_map.items()))

    # Dates arrive as text and are parsed with coerce; failures are reported.
    already_missing = int(r.rating_date.isna().sum())
    r["dt"] = pd.to_datetime(r.rating_date, errors="coerce")
    print(f"  date parse failures            : "
          f"{int(r.dt.isna().sum()) - already_missing:,}  "
          f"(already missing {already_missing:,}) — dropped")
    r = r.dropna(subset=["dt"])

    # Map each symbol to a notch, by agency. Out-of-list symbols become NA.
    r["notch"] = [notch_map[a].get(s) for a, s in zip(r.rating_type, r.rating)]
    r["is_term"] = r.rating.isin(TERMINATORS)
    skipped = r[r.notch.isna() & ~r.is_term]
    print(f"\n  valid notch                    : {int(r.notch.notna().sum()):,}")
    print(f"  NR / SUSP (terminators)        : {int(r.is_term.sum()):,}")
    print(f"  SKIPPED, outside the agency's official list (D-39): {len(skipped):,}")
    if len(skipped):
        for (a, s), n in skipped.groupby(["rating_type", "rating"]).size() \
                .sort_values(ascending=False).items():
            print(f"      {a:<5} {str(s):<14} {n:>6,}"
                  + ("   <- in official list?? " if s in official[a] else ""))

    after = r[r.dt.dt.year > YEAR_MAX]
    print(f"\n  records dated after {YEAR_MAX} (data errors, DROPPED): {len(after):,}")
    r = r[r.dt.dt.year <= YEAR_MAX].copy()

    # ---- 2. carry forward within (issue, agency) ---------------------------
    print("\n" + "=" * 78)
    print("STEP 2 — rating in force at 31 December, per (issue, agency)")
    print("=" * 78)
    events = r[r.notch.notna() | r.is_term].copy()
    # -1 marks "cleared by NR/SUSP", so it is distinguishable from a real notch.
    events["st"] = np.where(events.notch.notna(), events.notch, -1).astype(int)
    events = events.sort_values(["issue_id", "rating_type", "dt"])

    # A series runs from its first valid rating to its last. Nothing is invented
    # beyond the last thing FISD says about it.
    valid = events[events.st >= 0]
    span = valid.groupby(["issue_id", "rating_type"])["dt"].agg(["min", "max"])
    span["y0"] = span["min"].dt.year.clip(lower=YEAR_MIN)
    span["y1"] = span["max"].dt.year.clip(upper=YEAR_MAX)
    span = span[span.y0 <= span.y1]
    print(f"  (issue, agency) series with a valid rating: {len(span):,}")

    n_years = (span.y1 - span.y0 + 1).to_numpy()
    idx = span.index.repeat(n_years)
    years = np.concatenate([np.arange(a, b + 1) for a, b in zip(span.y0, span.y1)])
    grid = pd.DataFrame({"issue_id": idx.get_level_values(0),
                         "rating_type": idx.get_level_values(1),
                         "year": years})
    grid["asof"] = pd.to_datetime(dict(year=grid.year, month=12, day=31))
    grid = grid.sort_values("asof")
    ev = events[["issue_id", "rating_type", "dt", "st"]].sort_values("dt")
    grid = pd.merge_asof(grid, ev, left_on="asof", right_on="dt",
                         by=["issue_id", "rating_type"], direction="backward")
    print(f"  (issue, agency, year) cells               : {len(grid):,}")
    grid = grid[grid.st >= 0]          # -1 = withdrawn, no rating in force
    print(f"  cells with a rating in force              : {len(grid):,}")

    # ---- 3. collapse to issuer-year, worst notch ---------------------------
    print("\n" + "=" * 78)
    print("STEP 3 — collapse to issuer-year: the WORST notch that year")
    print("=" * 78)
    link = pd.read_parquet(ISSUE_IN, columns=["issue_id", "issuer_id"])
    link["issue_id"] = link["issue_id"].astype("Int64")
    link["issuer_id"] = link["issuer_id"].astype("Int64")
    if link.issue_id.duplicated().any():
        raise SystemExit("STOP: fisd_mergedissue is not unique on issue_id.")
    grid = grid.merge(link, on="issue_id", how="left")
    if grid.issuer_id.isna().any():
        raise SystemExit(f"STOP: {int(grid.issuer_id.isna().sum()):,} rated "
                         "issue-years have no issuer_id.")
    sentinel = int((grid.issuer_id <= 0).sum())
    print(f"  excluded sentinel issuer_id <= 0: {sentinel:,} issue-years")
    grid = grid[grid.issuer_id > 0].copy()

    notch_panel = grid.groupby(["issuer_id", "year"]).agg(
        notch=("st", "max"),                      # worst across bonds + agencies
        n_bonds_rated=("issue_id", "nunique"),
        n_agencies=("rating_type", "nunique"),
    ).reset_index()
    print(f"  issuer-years with a rating in force: {len(notch_panel):,}  "
          f"issuers {notch_panel.issuer_id.nunique():,}")

    # ---- 4. attach to the score panel --------------------------------------
    print("\n" + "=" * 78)
    print("STEP 4 — attach to the rating-score panel (issuer_id, fyear) = (.., year)")
    print("=" * 78)
    sc = pd.read_parquet(SCORES_IN)
    sc["issuer_id"] = sc.issuer_id.astype("Int64")
    print(f"  score panel: {len(sc):,} rows, {sc.issuer_id.nunique():,} issuers, "
          f"{sc.gvkey.nunique():,} gvkeys")
    m = sc.merge(notch_panel, left_on=["issuer_id", "fyear"],
                 right_on=["issuer_id", "year"], how="left")
    if len(m) != len(sc):
        raise SystemExit(f"STOP: the join changed the row count, {len(sc):,} -> {len(m):,}.")
    missing = m.notch.isna()
    print(f"  rows with no notch: {int(missing.sum()):,}")

    # ---- 5. LOUD CHECK: the notch must reproduce y row for row -------------
    print("\n" + "=" * 78)
    print("STEP 5 — LOUD CHECK: (notch >= 11) must equal y on EVERY row")
    print("=" * 78)
    # Compared as floats, so an Int64/float64 dtype difference cannot make two
    # equal values look unequal.
    spec_from_notch = (m.notch >= SPEC_FROM).astype(float)
    ok = (~missing) & (spec_from_notch == m.y.astype(float))
    rate = ok.mean() * 100
    print(f"  agreement: {int(ok.sum()):,} of {len(m):,} rows ({rate:.4f}%)")
    if not ok.all():
        bad = m[~ok].copy()
        bad["spec_from_notch"] = (bad.notch >= SPEC_FROM).astype("Int64")
        print(f"\n  DISAGREEING ROWS: {len(bad):,}. First 10:")
        cols = ["issuer_id", "gvkey", "fyear", "y", "notch", "spec_from_notch",
                "n_bonds_rated", "n_agencies"]
        with pd.option_context("display.width", 200):
            print(bad[cols].head(10).to_string(index=False))
        raise SystemExit("STOP: the notch does not reproduce y. NOTHING WRITTEN.")
    print("  PASSED — the notch reproduces y exactly.")

    # ---- 6. collapse to gvkey x fyear --------------------------------------
    print("\n" + "=" * 78)
    print("STEP 6 — collapse to gvkey x fyear: worst notch, y = MAX")
    print("=" * 78)
    score_cols = [c for c in m.columns if c.startswith("score_")]
    print(f"  before: {len(m):,} rows, {m.gvkey.nunique():,} gvkeys, "
          f"{m.issuer_id.nunique():,} issuer_ids")

    # The scores are a property of the statement, so they must not vary inside a
    # gvkey x fyear group. Asserted rather than assumed.
    g = m.groupby(["gvkey", "fyear"])[score_cols]
    spread = float((g.max() - g.min()).max().max())
    print(f"  max within-group score range: {spread:.3e}")
    if spread != 0:
        raise SystemExit("STOP: scores vary inside a gvkey x fyear group. "
                         "NOTHING WRITTEN.")

    agg = {"y": ("y", "max"), "notch": ("notch", "max"),
           "datadate": ("datadate", "first"), "date_avail": ("date_avail", "first"),
           "n_issuer_ids": ("issuer_id", "nunique"),
           "n_bonds_rated": ("n_bonds_rated", "max"),
           "n_agencies": ("n_agencies", "max")}
    agg.update({c: (c, "first") for c in score_cols})
    out = m.groupby(["gvkey", "fyear"]).agg(**agg).reset_index()
    print(f"  after : {len(out):,} rows, {out.gvkey.nunique():,} gvkeys")

    out["notch"] = out.notch.astype(int)
    out["letter"] = [LETTERS[n] for n in out.notch]
    out["band"] = [band_of(n) for n in out.notch]
    out["y"] = out.y.astype("Int64")
    # pct_ columns are within-panel percentiles and are never comparable across
    # markets, so they are not carried forward.
    out = out[["gvkey", "fyear", "datadate", "date_avail", "y", "notch", "letter",
               "band", "n_issuer_ids", "n_bonds_rated", "n_agencies"] + score_cols]

    print(f"\n  spec-grade share after collapse: {out.y.mean()*100:.3f}% "
          f"({int(out.y.sum()):,} of {len(out):,})")
    print("\n  letter distribution")
    print(f"    {'notch':>5} {'letter':<7} {'band':<7} {'n':>8} {'share':>8}")
    for n, k in out.groupby("notch").size().items():
        print(f"    {n:>5} {LETTERS[n]:<7} {band_of(n):<7} {k:>8,} "
              f"{k/len(out)*100:>7.2f}%")
    print("\n  band distribution")
    for b, k in out.groupby("band").size().sort_values(ascending=False).items():
        print(f"    {b:<7} {k:>8,} {k/len(out)*100:>7.2f}%")

    out.to_parquet(PANEL_OUT, index=False)
    print(f"\nWrote {PANEL_OUT.name}: {len(out):,} rows x {len(out.columns)} columns")
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
