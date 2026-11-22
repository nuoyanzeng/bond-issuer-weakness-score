"""
Stage 3.2 — Cross-sectional bond-issuer master table. One row per issuer.

Everything here stays inside FISD. No Compustat, no gvkey, no panel expansion,
no contact with the feature set.

Sources:
  fisd_mergedissuer                      the issuer attributes
  fisd_mergedissue                       first offering date, via issue_id
  output/stage3_0_distress_events.parquet the distress outcome, issuer level

Exclusion: industry_group == 4 (government, sovereign, agency) only, to match
stage3_0. No country filter; country_domicile is kept as a column.

Run:  python3 code/stage3_2_issuer_master.py
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data_raw" / "us"
OUT_DIR = PROJECT_DIR / "output"

ISSUER_IN = RAW_DIR / "fisd_mergedissuer.parquet"
ISSUE_IN = RAW_DIR / "fisd_mergedissue.parquet"
EVENTS_IN = OUT_DIR / "stage3_0_distress_events.parquet"
MASTER_OUT = OUT_DIR / "stage3_2_issuer_master.parquet"

# Attributes carried from the issuer table.
MASTER_COLS = ["issuer_id", "cusip_name", "legal_name", "industry_group",
               "industry_code", "sic_code", "naics_code", "country_domicile",
               "parent_id", "in_bankruptcy"]

EXCLUDE_INDUSTRY_GROUP = "4"


def main():
    # This script only ever creates a new file.
    if MASTER_OUT.exists():
        raise SystemExit(
            f"STOP: {MASTER_OUT.name} already exists. This script never overwrites; "
            "move or rename the existing file first."
        )

    # ---- step 1: the issuer attributes ------------------------------------
    m = pd.read_parquet(ISSUER_IN, columns=MASTER_COLS)
    m["issuer_id"] = m["issuer_id"].astype("Int64")
    m["parent_id"] = m["parent_id"].astype("Int64")
    start_n = len(m)

    print("=" * 78)
    print("STEP 1 — issuer attributes")
    print("=" * 78)
    print(f"  rows: {start_n:,}   distinct issuer_id: {m.issuer_id.nunique():,}")

    # The master must already be one row per issuer, or every count below is wrong.
    if m.issuer_id.duplicated().any():
        raise SystemExit("STOP: fisd_mergedissuer is not unique on issuer_id.")

    # Sentinel ids are not companies. fisd_mergedissue carries issuer_id = -1 on
    # one bond; it should not exist here, and that is asserted rather than assumed.
    if (m.issuer_id <= 0).any():
        raise SystemExit(
            f"STOP: {int((m.issuer_id <= 0).sum())} sentinel issuer_id <= 0 in the "
            "issuer table; the master would carry rows that are not companies."
        )

    # ---- step 2: first offering date --------------------------------------
    print("\n" + "=" * 78)
    print("STEP 2 — first offering date")
    print("=" * 78)
    iss = pd.read_parquet(ISSUE_IN, columns=["issue_id", "issuer_id", "offering_date"])
    iss["issuer_id"] = iss["issuer_id"].astype("Int64")

    # Dates arrive as text (D-04) and are parsed here rather than trusted.
    already_missing = int(iss.offering_date.isna().sum())
    iss["odt"] = pd.to_datetime(iss.offering_date, errors="coerce")
    failures = int(iss.odt.isna().sum()) - already_missing
    print(f"  bond rows: {len(iss):,}")
    print(f"  offering_date missing: {already_missing:,}   parse failures: {failures}")

    first = (iss.dropna(subset=["odt"])
             .groupby("issuer_id", as_index=False)["odt"].min()
             .rename(columns={"odt": "first_offering_date"}))
    print(f"  issuers with a first offering date: {len(first):,}")

    m = m.merge(first, on="issuer_id", how="left")
    no_date = int(m.first_offering_date.isna().sum())
    print(f"  in the master with NO offering date: {no_date:,} (kept, date left null)")

    # ---- step 3: the distress outcome -------------------------------------
    print("\n" + "=" * 78)
    print("STEP 3 — distress outcome")
    print("=" * 78)
    ev = pd.read_parquet(EVENTS_IN)
    ev["issuer_id"] = ev["issuer_id"].astype("Int64")
    print(f"  stage3_0 rows: {len(ev):,}   distinct issuers: {ev.issuer_id.nunique():,}")
    if ev.issuer_id.duplicated().any():
        raise SystemExit("STOP: stage3_0 is not unique on issuer_id.")
    events_total = len(ev)

    m = m.merge(ev, on="issuer_id", how="left")
    m["has_distress"] = m.first_distress_date.notna().map({True: "Y", False: "N"})
    print(f"  matched into the master: {int((m.has_distress == 'Y').sum()):,}")

    # ---- step 4: the group-4 exclusion ------------------------------------
    print("\n" + "=" * 78)
    print(f"STEP 4 — exclude industry_group == {EXCLUDE_INDUSTRY_GROUP}")
    print("=" * 78)
    # NA-safe. industry_group is a nullable string: NA != "4" evaluates to NA,
    # and boolean indexing then drops the row. Missing must be KEPT here, since
    # missing is not group 4 and one such issuer appears in stage3_0.
    n_missing = int(m.industry_group.isna().sum())
    print(f"  industry_group missing: {n_missing} (KEPT — missing is not 4)")
    drop_mask = m.industry_group.eq(EXCLUDE_INDUSTRY_GROUP).fillna(False)
    print(f"  dropped: {int(drop_mask.sum()):,}")
    m = m[~drop_mask].copy()
    print(f"  final rows: {len(m):,}")

    # ---- step 5: the alignment check --------------------------------------
    print("\n" + "=" * 78)
    print("STEP 5 — ALIGNMENT CHECK")
    print("=" * 78)
    n_yes = int((m.has_distress == "Y").sum())
    # stage3_0 already applied the group-4 exclusion, so every one of its
    # issuers must survive it here too.
    survivors = int(ev.issuer_id.isin(m.issuer_id).sum())
    print(f"  stage3_0 issuers                         : {events_total:,}")
    print(f"  of those surviving group-4 exclusion here: {survivors:,}")
    print(f"  has_distress == Y in the master          : {n_yes:,}")
    if n_yes != survivors:
        raise SystemExit(
            f"STOP: has_distress == Y is {n_yes:,} but {survivors:,} stage3_0 "
            "issuers survive the exclusion. The two disagree; not proceeding."
        )
    print("  ALIGNED.")

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("REPORT")
    print("=" * 78)
    print(f"\n  counts: start {start_n:,} | dropped group-4 "
          f"{start_n - len(m):,} | final {len(m):,}")

    n_no = int((m.has_distress == "N").sum())
    print(f"\n  has_distress: Y {n_yes:,} ({n_yes/len(m)*100:.1f}%) | "
          f"N {n_no:,} ({n_no/len(m)*100:.1f}%)")

    have = int(m.first_offering_date.notna().sum())
    print(f"\n  first_offering_date: {have:,} of {len(m):,} = {have/len(m)*100:.1f}%")
    print("    by decade:")
    dec = (m.first_offering_date.dt.year // 10 * 10).value_counts().sort_index()
    for d, n in dec.items():
        print(f"      {int(d)}s  {n:>6,}  {'#' * min(int(n/100), 60)}")

    print("\n  first_distress_date by decade:")
    dd = (m.first_distress_date.dt.year // 10 * 10).value_counts().sort_index()
    for d, n in dd.items():
        print(f"      {int(d)}s  {n:>6,}  {'#' * min(int(n/10), 60)}")

    cd = m.country_domicile
    non_usa = cd.notna() & (cd.str.strip().str.upper() != "USA")
    print(f"\n  country_domicile (reported only, NOT filtered):")
    print(f"      USA     {int((cd.notna() & ~non_usa).sum()):,}")
    print(f"      non-USA {int(non_usa.sum()):,}")
    print(f"      missing {int(cd.isna().sum()):,}")

    sub = m.parent_id.notna() & (m.parent_id != m.issuer_id)
    print(f"\n  subsidiaries (parent_id present and != issuer_id): {int(sub.sum()):,} "
          f"({sub.sum()/len(m)*100:.1f}%)")
    print(f"      parent_id missing: {int(m.parent_id.isna().sum()):,}")

    # ---- write -------------------------------------------------------------
    out = m[MASTER_COLS + ["first_offering_date", "has_distress",
                           "first_distress_date", "has_default", "has_bankruptcy"]] \
        .sort_values("issuer_id").reset_index(drop=True)
    out.to_parquet(MASTER_OUT, index=False)
    print(f"\nWrote {MASTER_OUT.name}: {len(out):,} rows x {len(out.columns)} columns")
    print("\n  head:")
    print(out.head(5).to_string(index=False))
    print("\nOne row per issuer. No Compustat, no gvkey, no panel expansion.")


if __name__ == "__main__":
    main()
