"""
Stage 3.0 — Build the firm-level distress-event table from FISD.

One row per issuer_id: the date of its FIRST distress event, where an event is
either a bond default (fisd_issue_default, issue level) or a bankruptcy filing
(fisd_bankruptcy, already issuer level).

This script does NOT link to Compustat, does not build any modelling table, and
defines no label beyond the first-event date.

Run:  python3 code/stage3_0_distress_events.py
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data_raw" / "us"
OUT_DIR = PROJECT_DIR / "output"

DEFAULT_IN = RAW_DIR / "fisd_issue_default.parquet"
BANKRUPT_IN = RAW_DIR / "fisd_bankruptcy.parquet"
ISSUE_IN = RAW_DIR / "fisd_mergedissue.parquet"
ISSUER_IN = RAW_DIR / "fisd_mergedissuer.parquet"
EVENTS_OUT = OUT_DIR / "stage3_0_distress_events.parquet"

# Issuers in this industry group are governments, agencies and sovereigns.
# They are excluded on instruction.
EXCLUDE_INDUSTRY_GROUP = "4"


def parse_dates(series, label):
    """Parse a text date column, reporting how many non-null values fail."""
    # FISD date columns arrive as text (D-04), so they are parsed here rather
    # than trusted. errors='coerce' turns anything unparseable into NaT.
    already_missing = int(series.isna().sum())
    parsed = pd.to_datetime(series, errors="coerce")
    failed = int(parsed.isna().sum()) - already_missing
    print(f"    {label}: {failed} unparseable, {already_missing} already missing, "
          f"{int(parsed.notna().sum()):,} usable")
    return parsed, failed


def main():
    # This script only ever creates a new file.
    if EVENTS_OUT.exists():
        raise SystemExit(
            f"STOP: {EVENTS_OUT.name} already exists. This script never overwrites; "
            "move or rename the existing file first."
        )

    print("=" * 90)
    print("STEP 1 — DEFAULT SIDE (issue level -> issuer level)")
    print("=" * 90)

    defaults = pd.read_parquet(DEFAULT_IN, columns=["issue_id", "default_date"])
    link = pd.read_parquet(ISSUE_IN, columns=["issue_id", "issuer_id"])

    # Id fields arrive as float and must be Int64 before any merge (D-04).
    defaults["issue_id"] = defaults["issue_id"].astype("Int64")
    link["issue_id"] = link["issue_id"].astype("Int64")
    link["issuer_id"] = link["issuer_id"].astype("Int64")

    print(f"  {len(defaults):,} default records, {len(link):,} issue-to-issuer rows")

    # The map must be one row per issue, or the merge would duplicate defaults.
    if link["issue_id"].duplicated().any():
        raise SystemExit(
            f"STOP: fisd_mergedissue is not unique on issue_id "
            f"({int(link['issue_id'].duplicated().sum()):,} duplicates); the map would "
            "multiply default records."
        )

    print("  date parsing:")
    defaults["default_dt"], failed_default = parse_dates(
        defaults["default_date"], "default_date")

    # Attach the issuer to each defaulted bond.
    defaults = defaults.merge(link, on="issue_id", how="left")
    unmapped = int(defaults["issuer_id"].isna().sum())
    if unmapped:
        raise SystemExit(
            f"STOP: {unmapped:,} defaulted issues have no issuer_id in "
            "fisd_mergedissue; they cannot be collapsed to a firm."
        )
    print(f"  all {len(defaults):,} defaulted issues mapped to an issuer")

    # Collapse to the earliest default per issuer.
    default_first = (
        defaults.dropna(subset=["default_dt"])
        .groupby("issuer_id", as_index=False)["default_dt"].min()
        .rename(columns={"default_dt": "first_default_date"})
    )
    print(f"  -> {len(default_first):,} issuers with a dated default")

    print("\n" + "=" * 90)
    print("STEP 2 — BANKRUPTCY SIDE (already issuer level)")
    print("=" * 90)

    bankrupt = pd.read_parquet(BANKRUPT_IN, columns=["issuer_id", "filing_date"])
    bankrupt["issuer_id"] = bankrupt["issuer_id"].astype("Int64")
    print(f"  {len(bankrupt):,} bankruptcy records")

    print("  date parsing:")
    bankrupt["filing_dt"], failed_filing = parse_dates(
        bankrupt["filing_date"], "filing_date")

    bankrupt_first = (
        bankrupt.dropna(subset=["filing_dt"])
        .groupby("issuer_id", as_index=False)["filing_dt"].min()
        .rename(columns={"filing_dt": "first_filing_date"})
    )
    print(f"  -> {len(bankrupt_first):,} issuers with a dated filing")

    print("\n" + "=" * 90)
    print("STEP 3 — UNION")
    print("=" * 90)

    events = default_first.merge(bankrupt_first, on="issuer_id", how="outer")

    # The earlier of the two dates, ignoring whichever side is absent.
    events["first_distress_date"] = events[
        ["first_default_date", "first_filing_date"]].min(axis=1)

    # Keep the provenance of each row visible.
    events["has_default"] = events["first_default_date"].notna().map({True: "Y", False: "N"})
    events["has_bankruptcy"] = events["first_filing_date"].notna().map({True: "Y", False: "N"})

    n_default_only = int(((events.has_default == "Y") & (events.has_bankruptcy == "N")).sum())
    n_bankrupt_only = int(((events.has_default == "N") & (events.has_bankruptcy == "Y")).sum())
    n_both = int(((events.has_default == "Y") & (events.has_bankruptcy == "Y")).sum())

    print(f"  union issuers before exclusion: {len(events):,}")
    print(f"      default only    : {n_default_only:,}")
    print(f"      bankruptcy only : {n_bankrupt_only:,}")
    print(f"      both            : {n_both:,}")

    # Every row must have a date and at least one source.
    if events["first_distress_date"].isna().any():
        raise SystemExit("STOP: some issuers have no first_distress_date.")
    if ((events.has_default == "N") & (events.has_bankruptcy == "N")).any():
        raise SystemExit("STOP: some issuers are flagged with neither source.")

    print("\n" + "=" * 90)
    print(f"STEP 4 — EXCLUDE industry_group == {EXCLUDE_INDUSTRY_GROUP} (Government)")
    print("=" * 90)

    issuers = pd.read_parquet(ISSUER_IN,
                              columns=["issuer_id", "cusip_name", "industry_group"])
    issuers["issuer_id"] = issuers["issuer_id"].astype("Int64")

    events = events.merge(issuers, on="issuer_id", how="left")

    # Missing industry_group is not equal to "4", so those rows are kept.
    # Reported rather than decided here.
    n_missing_group = int(events["industry_group"].isna().sum())
    if n_missing_group:
        print(f"  NOTE: {n_missing_group} issuer(s) have no industry_group and are KEPT, "
              "since missing is not equal to 4")

    to_drop = events[events["industry_group"] == EXCLUDE_INDUSTRY_GROUP]
    print(f"  dropping {len(to_drop)} issuer(s):")
    for _, r in to_drop.sort_values("cusip_name").iterrows():
        print(f"      {int(r.issuer_id):>7}  {r.cusip_name}")

    # NA-safe. `industry_group` is a nullable string column, where a missing
    # value compares to anything as NA rather than True or False, and boolean
    # indexing then treats that NA as false and drops the row. Writing the
    # filter as "not equal to 4, with missing counted as not-equal" keeps
    # unclassified issuers, which a plain `!= "4"` silently removed.
    events = events[~events["industry_group"].eq(EXCLUDE_INDUSTRY_GROUP).fillna(False)].copy()
    print(f"  union issuers after exclusion: {len(events):,}")

    print("\n" + "=" * 90)
    print("REPORT")
    print("=" * 90)

    n_default_only = int(((events.has_default == "Y") & (events.has_bankruptcy == "N")).sum())
    n_bankrupt_only = int(((events.has_default == "N") & (events.has_bankruptcy == "Y")).sum())
    n_both = int(((events.has_default == "Y") & (events.has_bankruptcy == "Y")).sum())
    print(f"  issuers            : {len(events):,}")
    print(f"      default only   : {n_default_only:,}")
    print(f"      bankruptcy only: {n_bankrupt_only:,}")
    print(f"      both           : {n_both:,}")
    print(f"  date parse failures: default_date {failed_default}, "
          f"filing_date {failed_filing}")

    years = events["first_distress_date"].dt.year
    print(f"  first_distress_date range: {int(years.min())} to {int(years.max())}")
    print("\n  count by year of first distress:")
    for y, n in years.value_counts().sort_index().items():
        print(f"      {int(y)}  {n:>4}  {'#' * min(int(n), 70)}")

    # Write the four columns specified, and nothing else.
    out = events[["issuer_id", "first_distress_date", "has_default", "has_bankruptcy"]] \
        .sort_values("issuer_id").reset_index(drop=True)
    out.to_parquet(EVENTS_OUT, index=False)

    print(f"\nWrote {EVENTS_OUT.name}: {len(out):,} rows x {len(out.columns)} columns")
    print("\n  head:")
    print(out.head(10).to_string(index=False))
    print("\nNo Compustat link, no modelling table, no label beyond the first-event date.")


if __name__ == "__main__":
    main()
