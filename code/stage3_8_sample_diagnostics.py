"""
Stage 3.8 — Missingness diagnostics for the two labelled samples.

DIAGNOSTIC ONLY. It prints to the terminal and writes NO files. Nothing is
imputed, nothing is dropped, no method is proposed. The "drop-columns curve" in
sections 5 and 6 is a MEASUREMENT of what dropping would cost, not a proposal to
drop anything.

  A  output/stage3_7_distress_sample_L3.parquet   positives: y = 1
  B  output/stage3_7_rating_sample.parquet        positives: spec_grade = 1

Run:  python3 code/stage3_8_sample_diagnostics.py
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

DISTRESS_IN = OUT_DIR / "stage3_7_distress_sample_L3.parquet"
RATING_IN = OUT_DIR / "stage3_7_rating_sample.parquet"

# Everything that is not a feature. Whatever remains is a feature, and the count
# is asserted to be 44 so a changed upstream file cannot slip past unnoticed.
NON_FEATURES = ["issuer_id", "gvkey", "fyear", "datadate", "date_avail",
                "first_offering_date", "first_distress_date", "y", "spec_grade"]
N_FEATURES = 44

# The k values at which the drop-columns curve is evaluated.
K_VALUES = [0, 1, 2, 3, 5, 8, 10, 15, 20]

# Bins for the per-row missing count.
ROW_BINS = [(0, 1, "0"), (1, 5, "1-4"), (5, 10, "5-9"), (10, 20, "10-19"),
            (20, 30, "20-29"), (30, 1e9, "30+")]

DECADES = [1970, 1980, 1990, 2000, 2010, 2020]


def rule(char, text=""):
    """A separator line, with an optional title."""
    print("\n" + char * 84)
    if text:
        print(text)
        print(char * 84)


def feature_list(df):
    """The feature columns of a sample, in file order."""
    feats = [c for c in df.columns if c not in NON_FEATURES]
    if len(feats) != N_FEATURES:
        raise SystemExit(f"STOP: expected {N_FEATURES} features, found "
                         f"{len(feats)}: {feats}")
    return feats


def missing_table(df, feats, pos):
    """% missing overall, on positives, on negatives, and the difference."""
    t = pd.DataFrame({
        "feature": feats,
        "miss_all": [df[f].isna().mean() * 100 for f in feats],
        "miss_pos": [df.loc[pos, f].isna().mean() * 100 for f in feats],
        "miss_neg": [df.loc[~pos, f].isna().mean() * 100 for f in feats],
    })
    # Positive minus negative. A large positive value means the feature is
    # systematically LESS available for the positive group.
    t["diff"] = t.miss_pos - t.miss_neg
    return t


def print_missing_table(t, pos_label, neg_label, head=None):
    """Print the missing-rate table; head=None prints every row."""
    rows = t if head is None else t.head(head)
    print(f"  {'feature':<24}{'all %':>9}{pos_label + ' %':>14}"
          f"{neg_label + ' %':>14}{'diff':>10}")
    print("  " + "-" * 69)
    for _, r in rows.iterrows():
        print(f"  {r.feature:<24}{r.miss_all:>9.2f}{r.miss_pos:>14.2f}"
              f"{r.miss_neg:>14.2f}{r['diff']:>+10.2f}")


def section_1_2(df, feats, pos, pos_label, neg_label):
    """Per-feature missing rates, then the same table by informative gap."""
    t = missing_table(df, feats, pos)

    rule("-", f"1. PER-FEATURE MISSING RATES — all {len(feats)} features, "
              "worst first")
    by_all = t.sort_values("miss_all", ascending=False).reset_index(drop=True)
    print_missing_table(by_all, pos_label, neg_label)

    rule("-", "2. INFORMATIVE MISSINGNESS — top 15 by (positive - negative), "
              "i.e. least available for the positive group")
    by_diff = t.sort_values("diff", ascending=False).reset_index(drop=True)
    print_missing_table(by_diff, pos_label, neg_label, head=15)
    print("\n  For contrast, the 5 MOST available on positives relative to "
          "negatives:")
    print_missing_table(by_diff.tail(5).reset_index(drop=True),
                        pos_label, neg_label)
    return t


def section_3(df, feats, pos, pos_label, neg_label):
    """How many features are missing on a single row."""
    rule("-", "3. MISSING-COUNT PER ROW")
    n_miss = df[feats].isna().sum(axis=1)
    print(f"  {'missing features':<20}{pos_label:>20}{neg_label:>22}")
    print("  " + "-" * 60)
    for lo, hi, lab in ROW_BINS:
        in_bin = (n_miss >= lo) & (n_miss < hi)
        p, n = int((in_bin & pos).sum()), int((in_bin & ~pos).sum())
        print(f"  {lab:<20}{p:>10,} ({p/int(pos.sum())*100:5.1f}%)"
              f"{n:>12,} ({n/int((~pos).sum())*100:5.1f}%)")
    print(f"\n  median missing per row: {pos_label} "
          f"{n_miss[pos].median():.0f}   {neg_label} {n_miss[~pos].median():.0f}")
    print(f"  mean   missing per row: {pos_label} "
          f"{n_miss[pos].mean():.1f}   {neg_label} {n_miss[~pos].mean():.1f}")


def section_4(df, feats, pos, pos_label, neg_label):
    """What a complete-case rule would leave."""
    rule("-", "4. COMPLETE-CASE TEST — keep only rows with NO missing feature")
    complete = df[feats].notna().all(axis=1)
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    c_p, c_n = int((complete & pos).sum()), int((complete & ~pos).sum())
    print(f"  rows          : {int(complete.sum()):>8,} of {len(df):>8,}  "
          f"({complete.mean()*100:5.2f}% kept)")
    print(f"  {pos_label:<14}: {c_p:>8,} of {n_p:>8,}  "
          f"({c_p/n_p*100 if n_p else 0:5.2f}% kept)")
    print(f"  {neg_label:<14}: {c_n:>8,} of {n_n:>8,}  "
          f"({c_n/n_n*100 if n_n else 0:5.2f}% kept)")
    print(f"  positive rate : {n_p/len(df)*100:.3f}% now  ->  "
          f"{c_p/int(complete.sum())*100 if complete.sum() else 0:.3f}% "
          "after complete-case")
    print(f"  issuers       : {df.issuer_id.nunique():,} now  ->  "
          f"{df.loc[complete, 'issuer_id'].nunique():,} after complete-case")


def drop_curve(df, feats, pos, order, title, pos_label):
    """Complete rows left after dropping the k worst features in `order`."""
    rule("-", title)
    print(f"  {'k':>3}  {'features dropped at this k':<40}"
          f"{'complete rows':>15}{'complete ' + pos_label:>20}")
    print("  " + "-" * 80)
    n_p = int(pos.sum())
    prev = 0
    for k in K_VALUES:
        newly = order[prev:k]
        prev = k
        keep = [f for f in feats if f not in order[:k]]
        complete = df[keep].notna().all(axis=1) if keep else pd.Series(
            True, index=df.index)
        c_all, c_p = int(complete.sum()), int((complete & pos).sum())
        label = ", ".join(newly) if newly else "(none — all 44 kept)"
        # Long lists wrap onto continuation lines rather than being truncated.
        first, rest = label[:40], label[40:]
        print(f"  {k:>3}  {first:<40}{c_all:>10,} "
              f"({c_all/len(df)*100:4.1f}%){c_p:>12,} "
              f"({c_p/n_p*100 if n_p else 0:4.1f}%)")
        while rest:
            print(f"       {rest[:40]:<40}")
            rest = rest[40:]


def section_7(df, feats, t, pos_label):
    """Missingness by decade, for the 5 worst and the 5 best features."""
    rule("-", "7. COVERAGE OVER TIME — % missing by decade of fyear")
    by_all = t.sort_values("miss_all", ascending=False).feature.tolist()
    picks = [("WORST 5", by_all[:5]), ("BEST 5", by_all[-5:])]
    dec = (df.fyear // 10 * 10).astype("Int64")
    counts = [int((dec == d).sum()) for d in DECADES]
    print(f"  rows per decade: " + "  ".join(
        f"{d}s {c:,}" for d, c in zip(DECADES, counts)))
    # Rows outside the printed decades are counted rather than silently omitted.
    outside = int((~dec.isin(DECADES)).sum())
    if outside:
        print(f"  rows OUTSIDE these decades (fyear < {DECADES[0]}), not shown "
              f"in the table below: {outside:,}")
    for lab, fs in picks:
        print(f"\n  {lab}")
        print(f"  {'feature':<24}" + "".join(f"{str(d)+'s':>9}" for d in DECADES))
        print("  " + "-" * (24 + 9 * len(DECADES)))
        for f in fs:
            cells = []
            for d in DECADES:
                m = dec == d
                cells.append("      -  " if not m.any()
                             else f"{df.loc[m, f].isna().mean()*100:>8.1f} ")
            print(f"  {f:<24}" + "".join(cells))


def section_8(df, feats, t):
    """Is a feature missing for a firm always, or only sometimes?"""
    rule("-", "8. WHOLE-FIRM MISSINGNESS — the 10 worst features")
    worst = t.sort_values("miss_all", ascending=False).feature.head(10).tolist()
    n_iss = df.issuer_id.nunique()
    print(f"  issuers in the sample: {n_iss:,}")
    print(f"  {'feature':<24}{'never reports it':>20}{'sometimes':>14}"
          f"{'always has it':>16}")
    print("  " + "-" * 74)
    for f in worst:
        # Per issuer: is the feature missing on every row, some rows, or none?
        g = df.groupby("issuer_id")[f].apply(lambda s: s.isna().mean())
        never, always = int((g == 1).sum()), int((g == 0).sum())
        some = n_iss - never - always
        print(f"  {f:<24}{never:>9,} ({never/n_iss*100:5.1f}%)"
              f"{some:>8,} ({some/n_iss*100:4.1f}%)"
              f"{always:>9,} ({always/n_iss*100:4.1f}%)")
    print("\n  'never reports it' = the feature is missing on EVERY row of that "
          "issuer.")


def diagnose(df, sample_name, pos_col, pos_label, neg_label):
    """The full eight-section diagnostic for one sample."""
    rule("=", f"SAMPLE {sample_name}")
    feats = feature_list(df)
    pos = df[pos_col] == 1
    print(f"  rows {len(df):,}   issuers {df.issuer_id.nunique():,}   "
          f"features {len(feats)}")
    print(f"  {pos_label}: {int(pos.sum()):,}   {neg_label}: "
          f"{int((~pos).sum()):,}   positive rate "
          f"{pos.sum()/len(df)*100:.3f}%")
    print(f"  fyear {int(df.fyear.min())} to {int(df.fyear.max())}")

    t = section_1_2(df, feats, pos, pos_label, neg_label)
    section_3(df, feats, pos, pos_label, neg_label)
    section_4(df, feats, pos, pos_label, neg_label)

    by_all = t.sort_values("miss_all", ascending=False).feature.tolist()
    drop_curve(df, feats, pos, by_all,
               "5. DROP-COLUMNS CURVE — features ranked by OVERALL missing rate",
               pos_label)
    by_pos = t.sort_values("miss_pos", ascending=False).feature.tolist()
    drop_curve(df, feats, pos, by_pos,
               f"6. DROP-COLUMNS CURVE — ranked by missing rate on "
               f"{pos_label} ROWS ONLY", pos_label)

    section_7(df, feats, t, pos_label)
    section_8(df, feats, t)
    return t


def compare(ta, tb):
    """Are the same features worst in both samples?"""
    rule("=", "CROSS-SAMPLE COMPARISON")
    wa = ta.sort_values("miss_all", ascending=False).feature.head(10).tolist()
    wb = tb.sort_values("miss_all", ascending=False).feature.head(10).tolist()
    both = [f for f in wa if f in wb]
    print(f"  A's 10 worst: {', '.join(wa)}")
    print(f"  B's 10 worst: {', '.join(wb)}")
    print(f"\n  in BOTH top-10 ({len(both)} of 10): "
          f"{', '.join(both) if both else '(none)'}")
    print(f"  A only: {', '.join(f for f in wa if f not in wb)}")
    print(f"  B only: {', '.join(f for f in wb if f not in wa)}")

    # Overall level, and the per-feature change from A to B.
    m = ta[["feature", "miss_all"]].merge(
        tb[["feature", "miss_all"]], on="feature", suffixes=("_A", "_B"))
    m["change"] = m.miss_all_B - m.miss_all_A
    print(f"\n  mean missing rate across the 44: A {ta.miss_all.mean():.2f}%   "
          f"B {tb.miss_all.mean():.2f}%")
    print(f"  median:                          A {ta.miss_all.median():.2f}%   "
          f"B {tb.miss_all.median():.2f}%")
    print(f"  features MORE missing in B than A: "
          f"{int((m.change > 0).sum())} of 44")
    print("\n  the 5 features that improve most from A to B "
          "(B minus A, negative = better in B):")
    for _, r in m.nsmallest(5, "change").iterrows():
        print(f"      {r.feature:<24}A {r.miss_all_A:6.2f}%  ->  "
              f"B {r.miss_all_B:6.2f}%   {r.change:+7.2f}")
    print("\n  the 5 that worsen most:")
    for _, r in m.nlargest(5, "change").iterrows():
        print(f"      {r.feature:<24}A {r.miss_all_A:6.2f}%  ->  "
              f"B {r.miss_all_B:6.2f}%   {r.change:+7.2f}")


def main():
    a = pd.read_parquet(DISTRESS_IN)
    b = pd.read_parquet(RATING_IN)
    ta = diagnose(a, "A — DISTRESS (stage3_7_distress_sample_L3.parquet)",
                  "y", "y=1", "y=0")
    tb = diagnose(b, "B — RATING (stage3_7_rating_sample.parquet)",
                  "spec_grade", "spec_grade=1", "spec_grade=0")
    compare(ta, tb)
    print("\nDiagnostic only. No files written, nothing imputed, nothing dropped.")


if __name__ == "__main__":
    main()
