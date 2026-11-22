# ABOUTME: Stage 2.1 - selects the 55 kept features from the Stage 1.3 panel and reports their state
# ABOUTME: Previews both winsorisation modes and writes the selected trimmed panel.
"""
stage2_1_clean_features.py

WHAT THIS DOES, IN ORDER
    1. Selects the 55 features the Stage 1.6 screen kept (verdict != 'DROP')
       from `output/stage1_3_feature_panel.parquet`, plus the three key
       columns, and writes `output/stage2_input_features.parquet`.
       This is a COLUMN SELECTION. No value is recomputed, renamed or altered.
    2. Prints the shape: rows, firms, feature columns.
    3. Prints one row per feature: % missing and % zero.
    4. Previews both winsorisation groupings, then writes untrimmed and trimmed
       panels with the current WINSORISE = True and WINSOR_MODE = "pooled" settings.

WINSORISATION SETTING
    The current configuration applies the 1%/99% trim on the pooled
    distribution. The preview also computes the within-year alternative so
    that the two choices can be compared. The non-continuous features are
    exempt from trimming.

    Note on the D-number: the winsorisation is **D-25**, not D-19. D-19 is the
    decision that CZ's own outlier rules are NOT applied, and its justification
    is precisely that D-25 takes over the job ("outlier control now rests
    entirely on D-25"). They point in opposite directions, so citing the wrong
    one would send a reader to the wrong entry.

WHAT IT DELIBERATELY DOES NOT DO
    No feature is dropped, added, renamed or recomputed. Missing values are
    NOT imputed - they stay missing, and winsorisation acts only on non-missing
    values. Nothing is
    standardised or scaled. The RESCUE feature (EarningsConsistency) and the
    five FLAG features are treated exactly like the other 49. No clustering.

THE SAFETY GUARD
    THREE of the 55 are pure 0/1 indicators - ConvDebt, OENEG, INTWO - and
    one, NumEarnIncrease, is a
    count on 0-8. Trimming a 0/1 column at the 1st and 99th percentiles is
    normally a no-op, but if the indicator fires in fewer than 1% of rows the
    99th percentile is 0 and EVERY 1 is silently clipped away - the column is
    destroyed without an error. The preview checks for exactly that and
    refuses to write the trimmed file if it would happen.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "output"

PANEL_IN = OUT_DIR / "stage1_3_feature_panel.parquet"
# Project-authored 55-feature review, not a vendor download. Supply the
# original CSV in output/; no script in this repository creates it.
SELECTION = OUT_DIR / "stage1_6_feature_selection.csv"
PANEL_SELECTED = OUT_DIR / "stage2_input_features.parquet"
# Two versions are written from this one run, so they can never drift apart:
# the same 55 columns, the same rows, differing only by the trim.
PANEL_CLEAN = OUT_DIR / "stage2_features_clean.parquet"                    # untrimmed
PANEL_CLEAN_WINSORED = OUT_DIR / "stage2_features_clean_winsored.parquet"  # trimmed

KEY_COLS = ["gvkey", "datadate", "fyear"]

# --- the winsorisation switch ----------------------------------------------
# False: run the trim as a DRY RUN, print what it would do, write nothing.
# True : apply it and write PANEL_CLEAN.
# The switch stays: the panel can be produced trimmed or untrimmed from the
# same script, so the two versions are never separate pieces of code.
WINSORISE = True

# EVERY NON-CONTINUOUS FEATURE IS EXEMPT FROM THE TRIM.
# The exempt set is not hard-coded; it is whatever `classify_columns` finds to
# be a 0/1 indicator or a small-integer count, so a feature cannot drift out of
# the exemption by being renamed or by a later rebuild changing its range.
# The reason is that a percentile trim on such a column does not remove an
# outlier - there are none - it removes a value the variable is defined to
# take: the 1 of an indicator, or the top rung of NumEarnIncrease's 0-8 ladder.
EXEMPT_NON_CONTINUOUS = True

# "pooled"  - one pair of percentiles for the whole panel
# "by_year" - percentiles recomputed within each fiscal year
# The preview reports both; the selected output currently uses pooled limits.
WINSOR_MODE = "pooled"

LOWER_Q, UPPER_Q = 0.01, 0.99


# ---------------------------------------------------------------------------
# Step 1 - select the kept features
# ---------------------------------------------------------------------------


def select_features():
    """Take the 55 kept features plus the keys. A pure column selection."""
    sel = pd.read_csv(SELECTION)
    kept = sel.loc[sel["verdict"] != "DROP", "feature"].tolist()
    dropped = sel.loc[sel["verdict"] == "DROP", "feature"].tolist()

    available = set(pq.ParquetFile(PANEL_IN).schema_arrow.names)
    missing = [f for f in kept if f not in available]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} kept features are not in the panel: {missing}")
    for k in KEY_COLS:
        if k not in available:
            raise SystemExit(f"STOP: key column {k} is not in the panel")

    # Read only what is needed rather than the whole 106-column panel.
    panel = pd.read_parquet(PANEL_IN, columns=KEY_COLS + kept)

    # Guards: the right number of features, and nothing dropped leaked back in.
    if len(kept) != 55:
        raise SystemExit(f"STOP: expected 55 kept features, the selection gives {len(kept)}")
    leaked = sorted(set(panel.columns) & set(dropped))
    if leaked:
        raise SystemExit(f"STOP: {len(leaked)} DROP features leaked into the panel: {leaked}")

    panel.to_parquet(PANEL_SELECTED, index=False)
    return panel, kept, dropped


# ---------------------------------------------------------------------------
# Steps 2 and 3 - shape, then missing and zero per feature
# ---------------------------------------------------------------------------


def report_shape(panel, features):
    print("=" * 78)
    print("SHAPE")
    print("=" * 78)
    print(f"  rows (firm-years)   {len(panel):>10,}")
    print(f"  firms (gvkey)       {panel['gvkey'].nunique():>10,}")
    print(f"  feature columns     {len(features):>10}")
    print(f"  key columns         {len(KEY_COLS):>10}  {KEY_COLS}")
    print(f"  fiscal years        {int(panel['fyear'].min())}-{int(panel['fyear'].max())}")


def report_missing_and_zero(panel, features):
    """One row per feature: how much is absent, and how much is exactly zero.

    The two must be read together. A column that is 90% present but whose
    present values are nearly all zero carries far less information than the
    missing rate alone suggests.
    """
    rows = []
    n = len(panel)
    for f in features:
        s = panel[f]
        n_obs = int(s.notna().sum())
        rows.append({
            "feature": f,
            "n_obs": n_obs,
            "pct_missing": 100 * (n - n_obs) / n,
            "pct_zero_of_obs": 100 * float((s.dropna() == 0).mean()) if n_obs else np.nan,
        })
    tab = pd.DataFrame(rows).sort_values("pct_missing")

    print()
    print("=" * 78)
    print("PER FEATURE - MISSINGNESS AND ZERO SHARE (55 features)")
    print("=" * 78)
    print(f"  {'feature':<26}{'n_obs':>10}{'% missing':>12}{'% zero of obs':>16}")
    print("  " + "-" * 62)
    for _, r in tab.iterrows():
        print(f"  {r['feature']:<26}{r['n_obs']:>10,.0f}"
              f"{r['pct_missing']:>11.2f}%{r['pct_zero_of_obs']:>15.2f}%")
    return tab


# ---------------------------------------------------------------------------
# Step 4 - preview both winsorisation modes and apply the selected mode
# ---------------------------------------------------------------------------


def classify_columns(panel, features):
    """Split the features into continuous, 0/1 indicator, and small-integer.

    Only the first kind is what a percentile trim is meant for. The other two
    are reported separately because the trim does something to them that it
    was never designed to do.
    """
    binary, discrete, continuous = [], [], []
    for f in features:
        v = panel[f].dropna()
        if v.empty:
            continuous.append(f)
            continue
        uniq = pd.unique(v)
        if set(np.unique(uniq)).issubset({0.0, 1.0}):
            binary.append(f)
        elif len(uniq) <= 20 and np.allclose(uniq, np.round(uniq)):
            discrete.append(f)
        else:
            continuous.append(f)
    return binary, discrete, continuous


def bounds_pooled(panel, features):
    """One (p1, p99) pair per feature, over the whole panel."""
    lo = panel[features].quantile(LOWER_Q)
    hi = panel[features].quantile(UPPER_Q)
    return lo, hi


def bounds_by_year(panel, features):
    """A (p1, p99) pair per feature PER FISCAL YEAR, broadcast back to rows.

    Recomputing within year removes the drift that makes a 1970 balance sheet
    look like an outlier beside a 2020 one; it also trims roughly 1% from each
    end of every year, which is a different and larger intervention.
    """
    g = panel.groupby("fyear")[features]
    lo = g.transform(lambda s: s.quantile(LOWER_Q))
    hi = g.transform(lambda s: s.quantile(UPPER_Q))
    return lo, hi


def winsorise(panel, trim_cols, mode):
    """Clip the given features to their 1st/99th percentiles.

    Only `trim_cols` is touched; every other column is copied through
    unchanged, which is how the non-continuous features stay exempt. Missing
    stays missing: `clip` leaves NaN alone, so the trim acts only on values
    that exist and nothing is imputed.
    """
    out = panel.copy()
    if not trim_cols:
        return out
    if mode == "pooled":
        lo, hi = bounds_pooled(panel, trim_cols)
        out[trim_cols] = panel[trim_cols].clip(lower=lo, upper=hi, axis=1)
    elif mode == "by_year":
        lo, hi = bounds_by_year(panel, trim_cols)
        out[trim_cols] = panel[trim_cols].clip(lower=lo, upper=hi)
    else:
        raise SystemExit(f"STOP: unknown WINSOR_MODE {mode!r}")
    return out


def count_touched(before, after, features):
    """How many existing values the trim actually moved, per feature."""
    rows = []
    for f in features:
        b, a = before[f], after[f]
        moved = int(((b != a) & b.notna()).sum())
        n_obs = int(b.notna().sum())
        rows.append({
            "feature": f,
            "n_obs": n_obs,
            "n_touched": moved,
            "pct_touched": 100 * moved / n_obs if n_obs else np.nan,
        })
    return pd.DataFrame(rows)


def dry_run(panel, features):
    """Show what the trim would do under BOTH groupings, changing nothing."""
    binary, discrete, continuous = classify_columns(panel, features)
    trim_cols = continuous if EXEMPT_NON_CONTINUOUS else features
    exempt = [f for f in features if f not in trim_cols]

    print()
    print("=" * 78)
    print("WINSORISATION PREVIEW - NOTHING IS MODIFIED AT THIS POINT")
    print("=" * 78)
    print(f"  continuous features        {len(continuous)}")
    print(f"  0/1 indicator features     {len(binary)}  {binary}")
    print(f"  small-integer features     {len(discrete)}  {discrete}")
    print(f"  -> trimmed  {len(trim_cols)} features")
    print(f"  -> EXEMPT   {len(exempt)} features: {exempt}")

    results = {}
    for mode in ("pooled", "by_year"):
        after = winsorise(panel, trim_cols, mode)
        results[mode] = count_touched(panel[features], after[features], features)

    tot_obs = int(panel[features].notna().sum().sum())
    print()
    print(f"  total non-missing values in the 55 features: {tot_obs:,}")
    for mode in ("pooled", "by_year"):
        t = int(results[mode]["n_touched"].sum())
        print(f"    {mode:<8} would move {t:>10,} values  ({100*t/tot_obs:5.2f}% of them)")

    # The dangerous case: an indicator whose rarer value is under 1% of rows.
    print()
    print("  EFFECT ON THE 6 NON-CONTINUOUS FEATURES")
    print(f"  {'feature':<26}{'kind':<10}{'pooled':>10}{'by_year':>10}   verdict")
    print("  " + "-" * 70)
    fatal = []
    after_pooled = winsorise(panel, trim_cols, "pooled")
    for f in binary + discrete:
        kind = "0/1" if f in binary else "int"
        p = int(results["pooled"].set_index("feature").loc[f, "n_touched"])
        y = int(results["by_year"].set_index("feature").loc[f, "n_touched"])
        before_u = panel[f].dropna().nunique()
        after_u = after_pooled[f].dropna().nunique()
        if f in binary and before_u > 1 and after_u == 1:
            verdict = "*** WOULD BE FLATTENED TO A CONSTANT ***"
            fatal.append(f)
        elif p or y:
            verdict = "values clipped - trim is doing something here"
        else:
            verdict = "no-op"
        print(f"  {f:<26}{kind:<10}{p:>10,}{y:>10,}   {verdict}")

    if fatal:
        print()
        print(f"  WARNING: {len(fatal)} indicator column(s) would lose one of their two")
        print(f"  values entirely: {fatal}. With WINSORISE = True the script refuses")
        print("  to write the file rather than destroy them silently.")

    # The features the trim bites hardest, so the cost is visible.
    print()
    print("  10 FEATURES MOST AFFECTED (pooled)")
    top = results["pooled"].sort_values("pct_touched", ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"    {r['feature']:<26}{r['n_touched']:>9,.0f} values "
              f"({r['pct_touched']:5.2f}% of its observations)")
    return results, fatal, trim_cols, exempt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Selecting the 55 kept features ...")
    panel, features, dropped = select_features()
    print(f"  wrote {PANEL_SELECTED.name}")
    print(f"  confirmed: {len(features)} feature columns, "
          f"0 of the {len(dropped)} DROP features present")

    report_shape(panel, features)
    report_missing_and_zero(panel, features)
    results, fatal, trim_cols, exempt = dry_run(panel, features)

    print()
    print("=" * 78)
    if not WINSORISE:
        print("WINSORISE = False - the trim was NOT applied.")
        print(f"{PANEL_CLEAN.name} was NOT written.")
        print("The panel on disk is un-winsorised; D-25 remains unimplemented.")
        print("To apply it: set WINSORISE = True and choose WINSOR_MODE.")
    else:
        if fatal:
            raise SystemExit(
                f"STOP: winsorising would flatten {fatal} to a constant. "
                f"Refusing to write. Exempt them or change the decision first."
            )
        clean = winsorise(panel, trim_cols, WINSOR_MODE)

        # Prove the exemption held and that nothing else was disturbed: the
        # exempt columns must come through identical value for value.
        for f in exempt:
            if not panel[f].equals(clean[f]):
                raise SystemExit(f"STOP: exempt feature {f} was modified")
        if list(clean.columns) != list(panel.columns) or len(clean) != len(panel):
            raise SystemExit("STOP: the clean panel has different columns or rows")

        # Both versions, from the same selection and the same rows.
        panel.to_parquet(PANEL_CLEAN, index=False)
        clean.to_parquet(PANEL_CLEAN_WINSORED, index=False)

        touched = int(results[WINSOR_MODE]["n_touched"].sum())
        tot = int(panel[features].notna().sum().sum())
        print(f"WINSORISE = True, mode = {WINSOR_MODE}")
        print(f"  trimmed {len(trim_cols)} features, {len(exempt)} exempt and verified "
              f"unchanged")
        print(f"  moved {touched:,} of {tot:,} non-missing values ({100*touched/tot:.2f}%)")
        print(f"  rows {len(clean):,}  firms {clean['gvkey'].nunique():,}  "
              f"features {len(features)}")
        print(f"  wrote {PANEL_CLEAN.name:<42} (untrimmed)")
        print(f"  wrote {PANEL_CLEAN_WINSORED.name:<42} (trimmed, {WINSOR_MODE})")
    print("=" * 78)
    print("Stopping here. No clustering.")


if __name__ == "__main__":
    main()
