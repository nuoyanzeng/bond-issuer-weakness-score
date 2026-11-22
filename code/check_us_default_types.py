"""
Check — US default types, for the default-rate comparison (test 3).

READ-ONLY. PRINTS ONLY. It writes no file and changes no existing file, and
`data_raw/` is read only. Run time is a few seconds.

THE SAMPLE IS TEST 3'S. output/stage8_2_us_distress_firmyear.parquet, fiscal
2013-2024, the raw rf distress score, bands cut at that sample's 30th and 70th
percentiles: strong (below the 30th), middle, weak (at or above the 70th). A "firm"
is a gvkey. The band counts are asserted against test 3 before anything is printed.

THE SPLIT IS THE ONE BEHIND "87.6% BANKRUPTCY-CODED": `default_type == 'B'` in
fisd_issue_default, 5,037 of 5,753 rows, a share of BOND-LEVEL rows over all years.

TWO THINGS THAT SPLIT DOES NOT SETTLE, BOTH PRINTED RATHER THAN DECIDED HERE.

  1. 284 records have NO default_type. They are inside the 12.4% that is not 'B',
     but a blank code is not evidence of a non-bankruptcy event: 77.1% of them
     belong to an issuer with a filing in fisd_bankruptcy. So two non-bankruptcy
     versions are printed:
         "not B"     every record not coded B, the literal complement of the 87.6%
         "I, P, C"   only records positively coded Interest, Principal or Covenant

  2. The main label also takes bankruptcy filings from fisd_bankruptcy, which has
     no default_type at all. They never count as non-bankruptcy events. Part 3
     prints two bankruptcy versions: B rows plus filings, and B rows alone.

THE LABEL RULE, copied from stage3_7 and stage8_2, and asserted to reproduce the
main label exactly before it is used on anything else:
    - one first-event date per ISSUER (stage3_0)
    - rows with date_avail on or after that date are not usable (stage3_7:104)
    - y = 1 on the issuer's single latest usable row (stage3_7:122)
    - that is done on the full stage3_7 sample, so a positive whose row did not
      reach the score panel is lost here exactly as it is in the main label
    - gvkey x fyear takes the maximum over its issuer rows (stage8_2:240)

WHAT THAT RULE DOES HERE, WHICH AFFECTS HOW PART 2 READS.
  - The firm-year counts are the same in all three columns of part 2. Scored rows
    exist only before a firm's FIRST event of any kind, so no row can be on or
    after its first non-bankruptcy event. The script stops if one ever is.
  - So every non-bankruptcy positive sits on a row that is already a main-label
    positive, including for a firm that filed for bankruptcy before its first
    non-bankruptcy record: its rows between those two dates were never scored.
  - A firm with only bankruptcy records has no non-bankruptcy event. Its rows stay
    in the denominator with y = 0, as for a firm that never defaulted.

Run:  python3 code/check_us_default_types.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data_raw" / "us"
OUT_DIR = PROJECT_DIR / "output"

YEAR_MIN, YEAR_MAX = 2013, 2024
BANDS = ["strong", "middle", "weak"]
BAR = "=" * 100

BANKRUPTCY_CODE = "B"
NON_BANKRUPTCY_CODES = ["I", "P", "C"]
UNCODED = "(blank)"

# What the check must reproduce. Each is a number already on the record.
EXPECT_B_ROWS, EXPECT_ALL_ROWS = 5037, 5753      # the 87.6%
EXPECT_SAMPLE_ROWS = 25221
EXPECT_CUTS = (0.016284, 0.068767)
# Test 3, distress_rf, US default: band -> (firm-years, firms, positives).
EXPECT_BANDS = {"strong": (7566, 1708, 2), "middle": (10088, 2450, 15),
                "weak": (7567, 1998, 218)}


def stop_if(condition, message):
    """Stop the run with a plain message when a check fails."""
    if condition:
        raise SystemExit(f"STOP: {message}")


def apply_label_rule(sample, first_event):
    """The stage3_7 label: y = 1 on each issuer's latest row before its event.

    `first_event` holds one date per issuer_id. An issuer not in it has no event,
    so all of its rows are 0.
    """
    event_date = sample.issuer_id.map(first_event)
    has_event = event_date.notna()
    n_after = int((has_event & (sample.date_avail >= event_date)).sum())
    stop_if(n_after, f"{n_after:,} rows sit on or after the event. Every row "
                     "should precede the firm's first event of any kind.")
    y = pd.Series(0, index=sample.index)
    last_rows = sample[has_event].groupby("issuer_id")["date_avail"].idxmax()
    y.loc[last_rows] = 1
    return y


def first_date_by_firm(records, pairs):
    """Each gvkey's earliest date across all of its issuers' records."""
    return records.merge(pairs, on="issuer_id").groupby("gvkey")["date"].min()


def main():
    print(BAR)
    print("US DEFAULT TYPES. READ-ONLY: nothing is written.")
    print(BAR)

    # ---- FISD records, each with its issuer and a parsed date ---------------
    defaults = pd.read_parquet(RAW_DIR / "fisd_issue_default.parquet",
                               columns=["issue_id", "default_date", "default_type"])
    link = pd.read_parquet(RAW_DIR / "fisd_mergedissue.parquet",
                           columns=["issue_id", "issuer_id"])
    filings = pd.read_parquet(RAW_DIR / "fisd_bankruptcy.parquet",
                              columns=["issuer_id", "filing_date", "filing_type"])
    codes = pd.read_parquet(RAW_DIR / "fisd_code.parquet")

    # The split behind the 87.6%, on the raw file before any restriction.
    n_b = int((defaults.default_type == BANKRUPTCY_CODE).sum())
    print(f"\n  Raw fisd_issue_default: {n_b:,} of {len(defaults):,} rows coded B "
          f"= {n_b / len(defaults) * 100:.1f}%")
    stop_if(n_b != EXPECT_B_ROWS or len(defaults) != EXPECT_ALL_ROWS,
            "the 87.6% does not reproduce.")

    # Id fields arrive as float and must be Int64 before a merge (D-04).
    defaults["issue_id"] = defaults.issue_id.astype("Int64")
    link["issue_id"] = link.issue_id.astype("Int64")
    link["issuer_id"] = link.issuer_id.astype("Int64")
    filings["issuer_id"] = filings.issuer_id.astype("Int64")
    stop_if(link.issue_id.duplicated().any(),
            "fisd_mergedissue is not unique on issue_id.")
    defaults = defaults.merge(link, on="issue_id", how="left")
    stop_if(defaults.issuer_id.isna().any(), "a default record has no issuer.")

    defaults["date"] = pd.to_datetime(defaults.default_date, errors="coerce")
    filings["date"] = pd.to_datetime(filings.filing_date, errors="coerce")
    defaults["code"] = defaults.default_type.fillna(UNCODED)
    unknown = set(defaults.code) - {BANKRUPTCY_CODE, UNCODED, *NON_BANKRUPTCY_CODES}
    stop_if(unknown, f"default_type codes not classified here: {sorted(unknown)}")

    # stage3_0 keeps dated records of non-government issuers only, so the same
    # records are used here. The rebuilt dates must then equal stage3_0's.
    events = pd.read_parquet(OUT_DIR / "stage3_0_distress_events.parquet",
                             columns=["issuer_id", "first_distress_date"])
    events["issuer_id"] = events.issuer_id.astype("Int64")
    in_scope = defaults.issuer_id.isin(events.issuer_id) & defaults.date.notna()
    defaults = defaults[in_scope].reset_index(drop=True)
    filings = filings[filings.issuer_id.isin(events.issuer_id)
                      & filings.date.notna()].reset_index(drop=True)

    # One row number per record, so an issuer linked to two gvkeys counts once.
    defaults["record"] = np.arange(len(defaults))
    filings["record"] = np.arange(len(filings))

    all_records = pd.concat([defaults[["issuer_id", "date"]],
                             filings[["issuer_id", "date"]]])
    first_any = all_records.groupby("issuer_id")["date"].min()
    recorded = events.set_index("issuer_id").first_distress_date
    stop_if(set(first_any.index) != set(recorded.index)
            or (first_any.reindex(recorded.index) != recorded).any(),
            "the first-event dates do not reproduce stage3_0.")

    # The two versions of a non-bankruptcy record, and of a bankruptcy record.
    not_b = defaults[defaults.code != BANKRUPTCY_CODE]
    coded = defaults[defaults.code.isin(NON_BANKRUPTCY_CODES)]
    b_rows = defaults[defaults.code == BANKRUPTCY_CODE]
    b_and_filings = pd.concat([b_rows[["issuer_id", "date"]],
                               filings[["issuer_id", "date"]]])

    # ---- the three labels, by the same rule, on the full stage3_7 sample ----
    sample = pd.read_parquet(OUT_DIR / "stage3_7_distress_sample_L3.parquet",
                             columns=["issuer_id", "gvkey", "datadate",
                                      "date_avail", "y"])
    sample["issuer_id"] = sample.issuer_id.astype("Int64")
    sample["y_main"] = apply_label_rule(sample, first_any)
    n_diff = int((sample.y_main != sample.y).sum())
    stop_if(n_diff, f"the rule differs from stage3_7's y on {n_diff:,} rows, so "
                    "it cannot be trusted on a new event date.")
    sample["y_not_b"] = apply_label_rule(
        sample, not_b.groupby("issuer_id")["date"].min())
    sample["y_coded"] = apply_label_rule(
        sample, coded.groupby("issuer_id")["date"].min())
    labels = ["y_main", "y_not_b", "y_coded"]

    # Down to the score panel's issuer rows, which is what the main label does.
    panel = pd.read_parquet(OUT_DIR / "stage5_1_score_panel_L3_K12.parquet",
                            columns=["issuer_id", "gvkey", "fyear", "datadate", "y"])
    panel["issuer_id"] = panel.issuer_id.astype("Int64")
    panel = panel.merge(sample[["issuer_id", "gvkey", "datadate"] + labels],
                        on=["issuer_id", "gvkey", "datadate"], how="left",
                        validate="one_to_one")
    stop_if(panel.y_main.isna().any(), "a panel row is not in the stage3_7 sample.")
    stop_if((panel.y_main != panel.y).any(), "the panel's y does not reproduce.")

    # Down to gvkey x fyear by the maximum, as stage8_2 does.
    cells = panel.groupby(["gvkey", "fyear"], as_index=False)[labels].max()
    firmyear = pd.read_parquet(OUT_DIR / "stage8_2_us_distress_firmyear.parquet",
                               columns=["gvkey", "fyear", "raw_rf", "label"])
    firmyear = firmyear.merge(cells, on=["gvkey", "fyear"], how="left",
                              validate="one_to_one")
    stop_if((firmyear.y_main != firmyear.label).any(),
            "the collapsed label does not reproduce stage8_2's.")

    # ---- test 3's sample and bands ------------------------------------------
    us = firmyear[firmyear.fyear.between(YEAR_MIN, YEAR_MAX)].copy()
    cut30, cut70 = np.percentile(us.raw_rf, [30, 70])
    us["band"] = np.where(us.raw_rf < cut30, "strong",
                          np.where(us.raw_rf < cut70, "middle", "weak"))

    print(f"  Sample: fiscal {YEAR_MIN}-{YEAR_MAX}, {len(us):,} firm-years, "
          f"{us.gvkey.nunique():,} firms")
    print(f"  Cuts on the raw rf distress score: 30th = {cut30:.6f}, "
          f"70th = {cut70:.6f}")
    stop_if(len(us) != EXPECT_SAMPLE_ROWS, "this is not test 3's sample.")
    stop_if((round(cut30, 6), round(cut70, 6)) != EXPECT_CUTS,
            "these are not test 3's cut points.")
    for band, (rows, firms, positives) in EXPECT_BANDS.items():
        g = us[us.band == band]
        stop_if((len(g), g.gvkey.nunique(), int(g.y_main.sum()))
                != (rows, firms, positives),
                f"the {band} band does not reproduce test 3.")
    print("  PASSED: the 87.6%, stage3_0's dates, the main label at all three "
          "levels, and test 3's bands")

    # The issuer x gvkey links of this sample. Records attach to firms through it.
    in_window = panel.fyear.between(YEAR_MIN, YEAR_MAX)
    pairs = panel.loc[in_window, ["issuer_id", "gvkey"]].drop_duplicates()

    # ---- PART 1 — events by raw default type --------------------------------
    print("\n" + BAR)
    print("1. DEFAULT EVENTS BY RAW FISD TYPE, for the firms in this sample")
    print(BAR)
    print("  An event here is one record, and fisd_issue_default holds one per "
          "BOND, so an issuer with\n  many bonds has many events. All of a firm's "
          "records count, whatever their date and whether\n  or not they are its "
          "first. A firm with several types appears on several lines.\n")

    default_names = codes[codes.type == "DEFAULT_TYPE"].set_index("code").description
    filing_names = codes[codes.type == "FILING_TYPE"].set_index("code").description

    print(f"  {'code':<9}{'description':<32}{'class':<40}{'events':>8}{'firms':>7}"
          f"{'issuers':>9}")
    d_linked = defaults.merge(pairs, on="issuer_id")
    for code in [BANKRUPTCY_CODE] + NON_BANKRUPTCY_CODES + [UNCODED]:
        g = d_linked[d_linked.code == code]
        if code == BANKRUPTCY_CODE:
            kind = "bankruptcy"
        elif code == UNCODED:
            kind = "uncoded: non-bankruptcy under 'not B' only"
        else:
            kind = "non-bankruptcy"
        name = default_names.get(code, "no default_type on the record")
        print(f"  {code:<9}{name:<32}{kind:<42}{g.record.nunique():>6,}"
              f"{g.gvkey.nunique():>7,}{g.issuer_id.nunique():>9,}")
    n_events = d_linked.record.nunique()
    n_b_sample = d_linked[d_linked.code == BANKRUPTCY_CODE].record.nunique()
    print(f"  {'all':<9}{'fisd_issue_default':<32}{'':<42}{n_events:>6,}"
          f"{d_linked.gvkey.nunique():>7,}{d_linked.issuer_id.nunique():>9,}")
    print(f"\n  Coded B: {n_b_sample:,} of {n_events:,} records = "
          f"{n_b_sample / n_events * 100:.1f}% (87.6% in the whole file)")

    print("\n  fisd_bankruptcy: a separate table with no default_type. The main "
          "label uses it.\n")
    print(f"  {'code':<9}{'description':<32}{'class':<42}{'events':>6}{'firms':>7}"
          f"{'issuers':>9}")
    f_linked = filings.merge(pairs, on="issuer_id")
    filing_code = f_linked.filing_type.fillna(UNCODED)
    for code in sorted(filing_code.unique()):
        g = f_linked[filing_code == code]
        name = filing_names.get(code, "no filing_type on the record")
        print(f"  {code:<9}{name:<32}{'bankruptcy':<42}{g.record.nunique():>6,}"
              f"{g.gvkey.nunique():>7,}{g.issuer_id.nunique():>9,}")
    print(f"  {'all':<9}{'fisd_bankruptcy':<32}{'':<42}"
          f"{f_linked.record.nunique():>6,}{f_linked.gvkey.nunique():>7,}"
          f"{f_linked.issuer_id.nunique():>9,}")

    # ---- PART 2 — the rates, by band, side by side ---------------------------
    print("\n" + BAR)
    print("2. DEFAULT RATE BY BAND: the main label against non-bankruptcy events only")
    print(BAR)
    print("  Events = firm-years with y = 1, under the same rule in every column. "
          "Rate = events / firm-years, %.\n")
    print(f"  {'':<9}{'':>11}{'':>7}{'main label':>17}{'not B':>17}"
          f"{'I, P, C':>17}")
    print(f"  {'band':<9}{'firm-years':>11}{'firms':>7}"
          + f"{'events':>10}{'rate':>7}" * 3)

    def rate_row(name, g):
        """One line of the table: the counts, then events and rate per label."""
        line = f"  {name:<9}{len(g):>11,}{g.gvkey.nunique():>7,}"
        for label in labels:
            n = int(g[label].sum())
            line += f"{n:>10,}{n / len(g) * 100:>7.3f}"
        print(line)

    for band in BANDS:
        rate_row(band, us[us.band == band])
    rate_row("all", us)

    # ---- PART 3 — firms holding both kinds of record -------------------------
    print("\n" + BAR)
    print("3. FIRMS WITH BOTH A BANKRUPTCY AND A NON-BANKRUPTCY RECORD")
    print(BAR)
    print("  Compares each firm's FIRST record of each kind, across all of its "
          "issuers. Days = the absolute\n  gap between the two dates; the median "
          "in the 'all' column includes same-date firms at zero.\n")
    print(f"  {'non-bankruptcy':<16}{'bankruptcy':<18}{'firms':>7}{'non-bk first':>14}"
          f"{'bk first':>10}{'same date':>11}{'median days':>13}"
          f"{'non-bk first':>14}{'bk first':>10}")

    for nb_name, nb_records in [("not B", not_b), ("I, P, C", coded)]:
        for bk_name, bk_records in [("B + filings", b_and_filings),
                                    ("B rows only", b_rows)]:
            both = pd.DataFrame({
                "nb": first_date_by_firm(nb_records[["issuer_id", "date"]], pairs),
                "bk": first_date_by_firm(bk_records[["issuer_id", "date"]], pairs),
            }).dropna()
            # Positive means the non-bankruptcy record came first.
            days = (both.bk - both.nb).dt.days
            nb_first, bk_first = days[days > 0], days[days < 0]

            def median_text(values):
                """Median absolute days, or a dash for an empty group."""
                return f"{values.abs().median():,.0f}" if len(values) else "-"

            print(f"  {nb_name:<16}{bk_name:<18}{len(both):>7,}{len(nb_first):>14,}"
                  f"{len(bk_first):>10,}{int((days == 0).sum()):>11,}"
                  f"{median_text(days):>13}{median_text(nb_first):>14}"
                  f"{median_text(bk_first):>10}")

    print("\n" + BAR)
    print("END. Every line above is a measured quantity. Nothing was written.")
    print(BAR)


if __name__ == "__main__":
    main()
