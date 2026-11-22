"""
Stage 3.1 — Clean FISD ratings into an issuer-year speculative-grade panel.

This panel is a VALIDATION TOOL. It is not the training label and it is not a
feature. Ratings are barred as Y by D-08 and BACKGROUND section 3.

What it does, in order:
  1. drop rating_type == 'SNP' (undocumented, 301 rows)
  2. agency = rating_type: FR Fitch, MR Moody's, SPR S&P, DPR Duff and Phelps
  3. map each symbol to 0 investment grade / 1 speculative / NA, BY AGENCY,
     against the official symbol lists in fisd_code
  4. parse rating_date from text, reporting failures
  5. forward fill within each (issue, agency): a rating stays in force until
     the next one changes it; NR and SUSP end it and are not carried past
  6. collapse to issuer-year: 1 if ANY bond under ANY agency is speculative

The panel covers 1993-2026 only. Pre-1993 records are still READ, so that a
rating set before 1993 and still in force in 1993 is counted; only the emitted
years are restricted. Records dated after 2026 are data errors and are dropped,
after being printed.

Run:  python3 code/stage3_1_rating_panel.py
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
PANEL_OUT = OUT_DIR / "stage3_1_rating_panel.parquet"

# The four agencies kept. SNP is dropped: it is not in fisd_code.
AGENCIES = {"FR": "Fitch", "MR": "Moody's", "SPR": "S&P", "DPR": "Duff and Phelps"}

# Which fisd_code list holds each agency's official symbols.
CODE_TYPE = {"SPR": "SP_RATING", "MR": "MOODY_RATING",
             "FR": "FITCH_RATING", "DPR": "DP_RATING"}

# Investment grade, S&P / Fitch / Duff and Phelps scale.
IG_SFD = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"}
# Speculative, same scale.
SPEC_SFD = {"BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-",
            "CC", "C", "D", "DD", "DDD", "SD"}
# Investment grade, Moody's scale.
IG_M = {"Aaa", "Aa1", "Aa2", "Aa3", "Aa", "A1", "A2", "A3", "A",
        "Baa1", "Baa2", "Baa3", "Baa"}
# Speculative, Moody's scale.
SPEC_M = {"Ba1", "Ba2", "Ba3", "Ba", "B1", "B2", "B3", "B",
          "Caa1", "Caa2", "Caa3", "Caa", "Ca", "C"}

# NR and SUSP end a rating. They are not merely missing: they stop the carry.
TERMINATORS = {"NR", "SUSP"}

# The panel's year range. A sample decision, recorded in DECISIONS.md.
YEAR_MIN, YEAR_MAX = 1993, 2026


def classify(agency, symbol, official):
    """0 investment grade, 1 speculative, None otherwise. Per agency."""
    # Anything outside the agency's own official list is not guessed at.
    if symbol not in official[agency]:
        return None
    ig, spec = (IG_M, SPEC_M) if agency == "MR" else (IG_SFD, SPEC_SFD)
    if symbol in ig:
        return 0
    if symbol in spec:
        return 1
    return None


def main():
    # This script only ever creates a new file.
    if PANEL_OUT.exists():
        raise SystemExit(
            f"STOP: {PANEL_OUT.name} already exists. This script never overwrites; "
            "move or rename the existing file first."
        )

    # ---- load -------------------------------------------------------------
    r = pd.read_parquet(RATINGS_IN,
                        columns=["issue_id", "rating_type", "rating_date", "rating"])
    codes = pd.read_parquet(CODE_IN)
    official = {a: set(codes[codes.type == t].code) for a, t in CODE_TYPE.items()}

    print("=" * 78)
    print("STEP 1-2 — agencies")
    print("=" * 78)
    print(f"  rating rows in: {len(r):,}")
    dropped_snp = int((r.rating_type == "SNP").sum())
    r = r[r.rating_type.isin(AGENCIES)].copy()
    print(f"  dropped rating_type == 'SNP': {dropped_snp:,}")
    print(f"  kept {len(r):,} rows across {r.rating_type.nunique()} agencies: "
          f"{sorted(r.rating_type.unique())}")

    # Ids arrive as float and must be Int64 before any merge (D-04).
    r["issue_id"] = r["issue_id"].astype("Int64")

    # ---- step 4: dates ----------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 4 — dates")
    print("=" * 78)
    already_missing = int(r.rating_date.isna().sum())
    r["dt"] = pd.to_datetime(r.rating_date, errors="coerce")
    failures = int(r.dt.isna().sum()) - already_missing
    print(f"  parse failures: {failures}   already missing: {already_missing}")
    if failures:
        print("  NOTE: unparseable dates are dropped from the panel; rows listed above.")
    r = r.dropna(subset=["dt"])

    # ---- step 3: classify -------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 3 — symbol -> 0 / 1 / NA, by agency")
    print("=" * 78)
    r["state"] = [classify(a, s, official) for a, s in zip(r.rating_type, r.rating)]
    r["is_term"] = r.rating.isin(TERMINATORS)
    n_valid = int(r.state.notna().sum())
    print(f"  valid (0/1)      : {n_valid:,}  ({n_valid/len(r)*100:.1f}%)")
    print(f"  NR / SUSP        : {int(r.is_term.sum()):,}  (terminators)")
    other_na = r[r.state.isna() & ~r.is_term]
    print(f"  other NA         : {len(other_na):,}  (outside the official list; "
          "skipped, not terminators)")

    # SANITY CHECK 6 — what fell through to NA for not being in any list.
    print("\n  SANITY 6 — symbols NA because not in the agency's official list, top 15:")
    if len(other_na):
        for (a, s), n in other_na.groupby(["rating_type", "rating"]).size() \
                .sort_values(ascending=False).head(15).items():
            print(f"      {a:<5} {str(s):<14} {n:>6,}")
        print(f"      total such rows: {len(other_na):,}")
    else:
        print("      none")

    # ---- out-of-range records, printed before being dropped ---------------
    print("\n" + "=" * 78)
    print("OUT-OF-RANGE RECORDS — printed before dropping, as agreed")
    print("=" * 78)
    yr = r.dt.dt.year
    after = r[yr > YEAR_MAX]
    print(f"  records dated after {YEAR_MAX} (data errors, DROPPED): {len(after):,}")
    if len(after):
        print(after[["issue_id", "rating_type", "rating_date", "rating", "state"]]
              .sort_values("rating_date").to_string(index=False))
    before = r[yr < YEAR_MIN]
    print(f"\n  records dated before {YEAR_MIN}: {len(before):,} "
          f"(KEPT as input, so a rating in force in {YEAR_MIN} is counted; "
          f"they simply emit no year of their own)")
    if len(before):
        vb = before[before.state.notna()]
        print(f"      of which valid (0/1): {len(vb):,}")
        print("      valid rows per year, earliest first:")
        for y, n in vb.dt.dt.year.value_counts().sort_index().items():
            print(f"        {int(y)}  {n:>5,}")

    # Only the post-2026 errors are removed. Pre-1993 rows stay as input.
    r = r[r.dt.dt.year <= YEAR_MAX].copy()

    # ---- step 5: forward fill within (issue, agency) ----------------------
    print("\n" + "=" * 78)
    print("STEP 5 — forward fill within (issue, agency)")
    print("=" * 78)

    # Only rows that change the state matter: a valid rating sets it, a
    # terminator clears it. Everything else is skipped entirely.
    events = r[r.state.notna() | r.is_term].copy()
    # -1 marks "cleared by NR/SUSP", so it can be told apart from a real 0.
    events["st"] = np.where(events.state.notna(), events.state, -1).astype(int)
    events = events.sort_values(["issue_id", "rating_type", "dt"])

    # SANITY CHECK 4 — how often does an issue get 2+ records in one year,
    # measured before the fill.
    ev_year = events.assign(y=events.dt.dt.year)
    per = ev_year.groupby(["issue_id", "rating_type", "y"]).size()
    print(f"  SANITY 4 — (issue, agency, year) cells with 2+ rating records: "
          f"{int((per >= 2).sum()):,} of {len(per):,} "
          f"({(per >= 2).mean()*100:.1f}%)")

    # A series runs from its first VALID rating to its last valid rating.
    # Nothing is invented beyond the last thing FISD says about it.
    valid = events[events.st >= 0]
    span = valid.groupby(["issue_id", "rating_type"])["dt"].agg(["min", "max"])
    span["y0"] = span["min"].dt.year.clip(lower=YEAR_MIN)
    span["y1"] = span["max"].dt.year.clip(upper=YEAR_MAX)
    span = span[span.y0 <= span.y1]
    print(f"  (issue, agency) series with a valid rating: {len(span):,}")

    # Build one row per (issue, agency, year) across each series' own span.
    n_years = (span.y1 - span.y0 + 1).to_numpy()
    idx = span.index.repeat(n_years)
    years = np.concatenate([np.arange(a, b + 1)
                            for a, b in zip(span.y0, span.y1)])
    grid = pd.DataFrame({"issue_id": idx.get_level_values(0),
                         "rating_type": idx.get_level_values(1),
                         "year": years})
    print(f"  (issue, agency, year) cells: {len(grid):,}")

    # The rating in force is the most recent event at or before 31 December.
    grid["asof"] = pd.to_datetime(dict(year=grid.year, month=12, day=31))
    grid = grid.sort_values("asof")
    ev = events[["issue_id", "rating_type", "dt", "st"]].sort_values("dt")
    grid = pd.merge_asof(grid, ev, left_on="asof", right_on="dt",
                         by=["issue_id", "rating_type"], direction="backward")

    # -1 means the series was withdrawn: no rating in force, so no row.
    grid = grid[grid.st >= 0]
    print(f"  cells with a rating in force: {len(grid):,}")

    # ---- step 6: collapse to issuer-year ----------------------------------
    print("\n" + "=" * 78)
    print("STEP 6 — collapse to issuer-year")
    print("=" * 78)
    link = pd.read_parquet(ISSUE_IN, columns=["issue_id", "issuer_id"])
    link["issue_id"] = link["issue_id"].astype("Int64")
    link["issuer_id"] = link["issuer_id"].astype("Int64")
    if link.issue_id.duplicated().any():
        raise SystemExit("STOP: fisd_mergedissue is not unique on issue_id.")
    grid = grid.merge(link, on="issue_id", how="left")
    unmapped = int(grid.issuer_id.isna().sum())
    if unmapped:
        raise SystemExit(f"STOP: {unmapped:,} rated issue-years have no issuer_id.")

    # Exclude sentinel identifiers. FISD carries issuer_id = -1 on one issue as
    # a placeholder for an unknown issuer; it is absent from fisd_mergedissuer
    # and is not a company. The unmapped check above does not catch it, because
    # the id is present, merely not real.
    sentinel = grid[grid.issuer_id <= 0]
    if len(sentinel):
        print(f"  excluded sentinel issuer_id <= 0: {len(sentinel):,} issue-years, "
              f"ids {sorted(sentinel.issuer_id.unique().tolist())}")
        grid = grid[grid.issuer_id > 0].copy()
    else:
        print("  no sentinel issuer_id <= 0 present")

    panel = grid.groupby(["issuer_id", "year"]).agg(
        spec_grade=("st", "max"),                 # any speculative -> 1
        n_bonds_rated=("issue_id", "nunique"),
        n_agencies=("rating_type", "nunique"),
    ).reset_index()
    panel["spec_grade"] = panel.spec_grade.astype("Int64")
    print(f"  issuer-year rows with at least one valid rating: {len(panel):,}")

    # Years inside an issuer's own span but with no rating in force are NA.
    sp = panel.groupby("issuer_id")["year"].agg(["min", "max"])
    n_years = (sp["max"] - sp["min"] + 1).to_numpy()
    idx = sp.index.repeat(n_years)
    years = np.concatenate([np.arange(a, b + 1) for a, b in zip(sp["min"], sp["max"])])
    full = pd.DataFrame({"issuer_id": idx, "year": years})
    panel = full.merge(panel, on=["issuer_id", "year"], how="left")
    panel["n_bonds_rated"] = panel.n_bonds_rated.fillna(0).astype("Int64")
    panel["n_agencies"] = panel.n_agencies.fillna(0).astype("Int64")
    print(f"  issuer-year rows after filling gaps inside each issuer's span: "
          f"{len(panel):,}")

    # ---- sanity checks ----------------------------------------------------
    print("\n" + "=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)

    n = len(panel)
    n1 = int((panel.spec_grade == 1).sum())
    n0 = int((panel.spec_grade == 0).sum())
    nna = int(panel.spec_grade.isna().sum())
    print(f"\n  1. spec_grade share: 0 {n0:,} ({n0/n*100:.1f}%) | "
          f"1 {n1:,} ({n1/n*100:.1f}%) | NA {nna:,} ({nna/n*100:.1f}%)")

    print("\n  2. speculative share BY YEAR")
    print(f"      {'year':<6}{'n':>8}{'spec':>8}{'spec %':>9}")
    by = panel.dropna(subset=["spec_grade"]).groupby("year")["spec_grade"]
    for y, g in by:
        print(f"      {int(y):<6}{len(g):>8,}{int(g.sum()):>8,}{g.mean()*100:>8.1f}%")

    dis = grid.groupby(["issuer_id", "year"])["st"].agg(["min", "max"])
    n_dis = int(((dis["min"] == 0) & (dis["max"] == 1)).sum())
    print(f"\n  3. issuer-years where raters disagree (some IG, some spec): "
          f"{n_dis:,} of {len(dis):,} rated issuer-years ({n_dis/len(dis)*100:.1f}%)")

    print(f"\n  5. panel dimensions")
    print(f"      distinct issuers : {panel.issuer_id.nunique():,}")
    print(f"      year range       : {int(panel.year.min())} to {int(panel.year.max())}")
    print(f"      issuer-year rows : {len(panel):,}")
    print(f"      non-NA / NA      : {n - nna:,} / {nna:,}")

    # ---- write ------------------------------------------------------------
    out = panel[["issuer_id", "year", "spec_grade", "n_bonds_rated", "n_agencies"]] \
        .sort_values(["issuer_id", "year"]).reset_index(drop=True)
    out.to_parquet(PANEL_OUT, index=False)
    print(f"\nWrote {PANEL_OUT.name}: {len(out):,} rows x {len(out.columns)} columns")
    print("\n  head:")
    print(out.head(10).to_string(index=False))
    print("\nValidation tool only. Not the training label, not a feature. "
          "No gvkey merge.")


if __name__ == "__main__":
    main()
