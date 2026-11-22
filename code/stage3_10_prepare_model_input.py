"""
Stage 3.10 — model input preparation, shared by BOTH outcomes.

Feature cut (K), hierarchical imputation, cross-sectional rank transform within
fyear. No model is fitted here: stage4_1 fits the distress models and stage4_2
the rating models, both on what this script writes.

Writes two files:
  output/stage4_0_distress_{SAMPLE_LAG}_K{K}.parquet    cut, imputed, ranked
  output/stage4_0_rating_{SAMPLE_LAG}_K{K}.parquet      same treatment
The name carries the lag and K, so runs at different settings coexist.

Both outcomes end up on the IDENTICAL features with the IDENTICAL transforms,
and that is asserted rather than assumed. Every model is fitted on the same
prepared data: no un-imputed variant is written, because a model that can see
the missingness pattern when the others cannot would confound the model with
the data (D-45).

Everything you would want to change lives in the CONFIG block below, in the same
style as AVAIL_LAG_MONTHS in stage3_5. The drop list is REGENERATED from the
ranking every run, never frozen.

Renamed from stage3_10_model.py on 2026-08-27; earlier CHANGELOG entries use the
old name.

Run:  python3 code/stage3_10_prepare_model_input.py
"""

from pathlib import Path

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"


# ===========================================================================
# CONFIG — everything adjustable is here
# ===========================================================================

# Which distress sample. L3 is the main setting (D-41, availability lag L = 3).
# L0 is the no-guard comparison. NOTE: only the L3 sample exists on disk today;
# selecting "L0" fails with a clear message until that sample is built.
SAMPLE_LAG = "L3"                       # "L3" (main) or "L0" (no guard)
SAMPLE_FILE = OUT_DIR / f"stage3_7_distress_sample_{SAMPLE_LAG}.parquet"
RATING_FILE = OUT_DIR / "stage3_7_rating_sample.parquet"
# Read-only, for the industry code used by imputation Level 2.
ANNUAL_IN = OUT_DIR / "compustat_annual_clean.parquet"
NAMES_IN = PROJECT_DIR / "data_raw" / "us" / "comp_names.parquet"

# How many features the ranking drops. Main setting K = 12 -> 31 features kept.
# Measured alternatives, complete-case POSITIVES in the distress sample
# (of 1,438; the unfiltered 44-feature panel gives 119):
#     K=5: 792 | K=8: 862 | K=10: 968 | K=12: 1,076 | K=14: 1,126
K = 12

# Output names carry the two parameters that get varied — the availability lag
# and K — so runs at different settings COEXIST instead of superseding one
# another, and the never-overwrite guard stops firing on every rebuild.
# WINDOW_START/WINDOW_END are NOT in the name: they are a settled default
# (D-46), not a parameter, and putting them in a filename would imply otherwise.

# A label for a FEATURE-SET EXPERIMENT. Empty means the main run and the
# filenames are unchanged, so nothing existing is ever replaced. Set it to a
# short name and every output of this stage carries that suffix, letting an
# alternative feature set coexist with the main one instead of overwriting it.
SET_TAG = ""
_T = f"_{SET_TAG}" if SET_TAG else ""

DISTRESS_OUT = OUT_DIR / f"stage4_0_distress_{SAMPLE_LAG}_K{K}{_T}.parquet"
RATING_OUT = OUT_DIR / f"stage4_0_rating_{SAMPLE_LAG}_K{K}{_T}.parquet"

# The ranking that generates the drop list: US missingness on POSITIVE rows of
# the distress sample, worst first; ties broken by overall US missingness, then
# by name so the order is deterministic. Changing K regenerates the list from
# this ranking mechanically — there is no hard-coded list of dropped features.
def rank_key(name, miss_pos, miss_all):
    """Sort key: worst positive-row missingness first."""
    return (-miss_pos, -miss_all, name)

# Dropped regardless of K, each for a reason that is not about missingness.
ALWAYS_EXCLUDE = {
    "WW": "D-62: not computable from CSMAR (WW, TotalAccruals, AssetTurnover) or too costly in Chinese coverage (DelSTI); the feature set is the 27",
    "TotalAccruals": "D-62: not computable from CSMAR (WW, TotalAccruals, AssetTurnover) or too costly in Chinese coverage (DelSTI); the feature set is the 27",
    "AssetTurnover": "D-62: not computable from CSMAR (WW, TotalAccruals, AssetTurnover) or too costly in Chinese coverage (DelSTI); the feature set is the 27",
    "DelSTI": "D-62: not computable from CSMAR (WW, TotalAccruals, AssetTurnover) or too costly in Chinese coverage (DelSTI); the feature set is the 27",
    "ConvDebt":
        "not China-computable: dc and cshrc are both blank in the CSMAR "
        "mapping, and no approximation exists (convertibles sit "
        "undifferentiated inside 应付债券)",
    "EarningsConsistency":
        "D-34 RESCUE unresolved — still built on epspx, not ni/at — and the "
        "worst feature on BOTH criteria (US positives 68.78%, China 24.83%)",
    "NumEarnIncrease":
        "earns nothing in six of six model-outcome fits (D-50): ranks 21-30 "
        "everywhere and one model drove its coefficient to exactly zero. It is "
        "one of CZ's 101, so it stays BUILT (D-02) and is excluded HERE, where "
        "selection decisions belong.",
}

# Pin a feature so the ranking can never drop it. Empty for now.
ALWAYS_KEEP = []

# --- missing values ------------------------------------------------------
# "hierarchical"     MAIN SETTING. Level 1 then Level 2, below; anything still
#                    missing stays NA. No Level 3 (it recovered 1 extra y=1).
# "cs_median"        same-fyear cross-sectional median only — the
#                    Gu-Kelly-Xiu convention. Robustness.
# "median_indicator" cs_median plus a per-feature 0/1 missing dummy. Robustness.
# "complete_case"    fill nothing, so the 89,843-row / 1,076-positive
#                    complete-case figures stay reproducible.
IMPUTE_METHOD = "hierarchical"

# LEVEL 1 — the firm's own past. Median of its most recent LOOKBACK_OBS
# available prior observations of that feature, counted in OBSERVATIONS, not
# calendar years, and taken strictly from earlier fiscal years: `fyear <`, so
# the three gvkeys carrying two statements inside one fyear cannot fill from a
# row sharing their fyear.
LOOKBACK_OBS = 2

# LEVEL 2 — the industry cell, for what Level 1 could not fill. Median of the
# same fyear x industry cell. sic2D is D-10's industry code: Compustat `sich`
# where present, `comp.names.sic` as fallback. 100% covered on both samples.
IMPUTE_INDUSTRY = "sic2D"
# Restrict the cell to rows already AVAILABLE at the target row's date_avail.
# Without this a cell-mate with a later fiscal year-end inside the same fyear
# lends a statement published after the row being filled.
STRICT_AVAIL_CELLS = True
# A cell median needs at least this many firms behind it. Below it, no fill.
MIN_CELL_FIRMS = 5

# --- sample window -------------------------------------------------------
# Fiscal years kept, inclusive. Applied to BOTH samples AFTER the feature cut
# (which is settled on the full sample) and BEFORE imputation, so Level-2 cell
# medians and the ranks are computed inside the window.
#
# 1993 is for COMPARABILITY with the rating panel, which starts in 1993 (D-39).
# It is NOT for rank validity: ranking is already sound from 1971, at zero cost.
# It costs 245 of 1,438 distress events, 17%. See D-46.
WINDOW_START = 1993
# 2026 is a partial fiscal year: 49 rows and 31 firms, against 3,109 rows in
# 2025. A 31-firm cross-section cannot support a within-year rank. See D-46.
WINDOW_END = 2025

# --- rank transform ------------------------------------------------------
# Cross-sectional rank WITHIN fyear, ties averaged, mapped to [-0.5, 0.5]:
# the JKP / Gu-Kelly-Xiu convention. Applied after imputation.
#
# Two consequences, both deliberate and both recorded in D-44:
#   * it makes D-25's 1st/99th winsorisation INERT — a within-year rank is
#     bounded by construction, so the project's only outlier control no longer
#     binds on the modelling file (it still governs the Stage 2 files);
#   * it is a cross-sectional operation, so a row's value depends on its
#     contemporaneous peers — the same mild, documented leak class as the
#     Level-2 cell medians, and the same GKX precedent.
RANK_TRANSFORM = True
# Non-continuous features, exempt: ranking a 0/1 indicator collapses it to two
# constants. Same set D-25 exempts from winsorisation.
# D-50 retired the four `_negden` flags; OENEG and INTWO replace them.
RANK_EXEMPT = ["OENEG", "INTWO", "NumEarnIncrease"]

# The SAME three are exempt from imputation, and for the same structural reason.
# A _negden indicator states a FACT — the denominator was negative — not a
# quantity to be estimated. A cell median of 0.5 has no meaning under D-17, and
# rounding it would fabricate the fact rather than impute a value. They stay NA.
IMPUTE_EXEMPT = list(RANK_EXEMPT)

# ...and their NA is then ENCODED as a third category rather than left a gap.
# The NA means "the denominator itself was missing, so the fact cannot be
# determined" — recording that is not fabricating it, and it stops rows being
# dropped at fitting time purely because an indicator is blank. Set to None to
# leave them NA. See D-44.
INDICATOR_NA_CODE = -1

# Independent reasons carried alongside a feature the ranking drops. These do
# not cause the drop; they are recorded so the printout says why the drop is
# also justified on other grounds.
INDEPENDENT_REASONS = {
    "EarningsSmoothness": "also one of the seven +6-anchored features (D-41)",
    "Investment":         "also one of the seven +6-anchored features (D-41)",
    "roavol":             "also one of the seven +6-anchored features (D-41)",
    "currat":             "also structural: 79.5% of issuers that never report "
                          "it are SIC 6000-6999, financial firms",
    "GP":                 "also structural: 82.7% of issuers that never report "
                          "it are SIC 6000-6999, financial firms",
}

# ===========================================================================
# END CONFIG
# ===========================================================================


# Columns that are keys or labels, never features.
NON_FEATURES = ["issuer_id", "gvkey", "fyear", "datadate", "date_avail",
                "first_offering_date", "first_distress_date", "y", "spec_grade"]
# Carried through to the model-input files alongside the features.
ID_COLS = ["issuer_id", "gvkey", "fyear", "datadate", "date_avail"]


def feature_list(df):
    """Feature columns of a sample, in file order."""
    return [c for c in df.columns if c not in NON_FEATURES]


def build_drop_list(distress, feats):
    """Regenerate the drop list from the ranking. Never a frozen list.

    Returns (dropped, kept, ranking) where ranking is the full ordered table.
    """
    pos = distress.y == 1
    ranking = []
    for f in feats:
        ranking.append((f,
                        distress.loc[pos, f].isna().mean() * 100,
                        distress[f].isna().mean() * 100))
    # ALWAYS_KEEP is pinned: removed from the pool so the ranking cannot take it.
    pool = [r for r in ranking if r[0] not in ALWAYS_KEEP]
    pool.sort(key=lambda r: rank_key(r[0], r[1], r[2]))

    by_rank = [r[0] for r in pool[:K]]
    dropped = list(dict.fromkeys(by_rank + list(ALWAYS_EXCLUDE)))
    kept = [f for f in feats if f not in dropped]
    return dropped, kept, dict((r[0], (r[1], r[2])) for r in ranking), by_rank


def print_cut(dropped, kept, stats, by_rank):
    """Print what went and what stayed, each with its reason."""
    print("=" * 84)
    print(f"FEATURE CUT — K = {K}, sample {SAMPLE_LAG}")
    print("=" * 84)
    print(f"  features in    : {len(dropped) + len(kept)}")
    print(f"  dropped        : {len(dropped)}  "
          f"({len(by_rank)} by the ranking at K={K}, "
          f"{len(set(ALWAYS_EXCLUDE) - set(by_rank))} from ALWAYS_EXCLUDE)")
    print(f"  KEPT           : {len(kept)}")
    if ALWAYS_KEEP:
        print(f"  pinned by ALWAYS_KEEP: {', '.join(ALWAYS_KEEP)}")

    print(f"\n  DROPPED, in ranking order:")
    print(f"      {'feature':<24}{'US pos %':>10}{'US all %':>10}  reason")
    for f in dropped:
        mp, ma = stats[f]
        if f in ALWAYS_EXCLUDE and f not in by_rank:
            why = f"ALWAYS_EXCLUDE — {ALWAYS_EXCLUDE[f]}"
        else:
            why = f"ranked {by_rank.index(f) + 1} of {K} on positive-row missingness"
            if f in ALWAYS_EXCLUDE:
                why += f"; ALSO ALWAYS_EXCLUDE — {ALWAYS_EXCLUDE[f]}"
            if f in INDEPENDENT_REASONS:
                why += f"; {INDEPENDENT_REASONS[f]}"
        print(f"      {f:<24}{mp:>10.2f}{ma:>10.2f}  {why}")

    print(f"\n  KEPT ({len(kept)}):")
    for f in sorted(kept, key=lambda x: -stats[x][0]):
        mp, ma = stats[f]
        print(f"      {f:<24}{mp:>10.2f}{ma:>10.2f}")


def payoff(distress, rating, kept):
    """Complete-case counts on both samples. Measurement, not a proposal."""
    print("\n" + "=" * 84)
    print("COMPLETE-CASE PAYOFF at this cut — measurement only, NOT applied")
    print("=" * 84)
    pos = distress.y == 1
    c = distress[kept].notna().all(axis=1)
    print(f"  DISTRESS ({SAMPLE_LAG})")
    print(f"      rows      : {int(c.sum()):>8,} of {len(distress):>8,} "
          f"({c.mean()*100:5.2f}%)")
    print(f"      y=1       : {int((c & pos).sum()):>8,} of {int(pos.sum()):>8,} "
          f"({(c & pos).sum()/pos.sum()*100:5.2f}%)   "
          f"[pre-window complete-case baseline: 1,076 of 1,438]")

    posr = rating.spec_grade == 1
    cr = rating[kept].notna().all(axis=1)
    print(f"  RATING")
    print(f"      rows      : {int(cr.sum()):>8,} of {len(rating):>8,} "
          f"({cr.mean()*100:5.2f}%)")
    print(f"      spec=1    : {int((cr & posr).sum()):>8,} of {int(posr.sum()):>8,} "
          f"({(cr & posr).sum()/posr.sum()*100:5.2f}%)   "
          f"[pre-window complete-case baseline: 19,592 of 23,907]")


# ===========================================================================
# IMPUTATION
# ===========================================================================

def add_industry(df):
    """Attach sic2D, rebuilt exactly as D-10 specifies: Compustat's historical
    `sich` where present, `comp.names.sic` as the fallback. Read-only inputs."""
    ann = pd.read_parquet(ANNUAL_IN, columns=["gvkey", "datadate", "sich"])
    names = pd.read_parquet(NAMES_IN, columns=["gvkey", "sic"])
    names["sic"] = names["sic"].replace("", pd.NA)
    ann = ann.merge(names, on="gvkey", how="left")
    sich4 = ann["sich"].astype("Int64").astype(str).str.zfill(4)
    sich4 = sich4.where(ann["sich"].notna())
    ann["ind_sic"] = sich4.fillna(ann["sic"])
    ann["sic2D"] = ann["ind_sic"].str[:2]
    out = df.merge(ann[["gvkey", "datadate", "sic2D"]],
                   on=["gvkey", "datadate"], how="left")
    n = int(out.sic2D.isna().sum())
    print(f"  industry {IMPUTE_INDUSTRY}: missing on {n:,} rows "
          f"({n/len(out)*100:.2f}%) — those rows can never take a Level-2 fill")
    return out


def level1_firm_past(df, feats):
    """Median of the firm's most recent LOOKBACK_OBS prior observations.

    Prior means STRICTLY an earlier fiscal year, so a gvkey with two statements
    inside one fyear cannot fill from its own twin. Counted in observations,
    not calendar years: two statements ten years apart still count as two.
    """
    out = pd.DataFrame(np.nan, index=df.index, columns=feats, dtype="float64")
    order = df.sort_values(["gvkey", "fyear", "datadate"]).index
    d = df.loc[order]
    vals = d[feats].astype("float64").to_numpy()
    fy = d["fyear"].to_numpy()
    res = np.full(vals.shape, np.nan)

    start = 0
    gv = d["gvkey"].to_numpy()
    for end in np.flatnonzero(np.r_[gv[1:] != gv[:-1], True]) + 1:
        v, f = vals[start:end], fy[start:end]
        # ring buffer of the last LOOKBACK_OBS non-missing values per feature
        ring = np.full((LOOKBACK_OBS, v.shape[1]), np.nan)
        count = np.zeros(v.shape[1], dtype=int)
        j = 0                       # rows already released into the ring
        for i in range(len(v)):
            while j < i and f[j] < f[i]:
                m = ~np.isnan(v[j])
                if m.any():
                    ring[1:, m] = ring[:-1, m]
                    ring[0, m] = v[j][m]
                    count[m] += 1
                j += 1
            have = count > 0
            if have.any():
                # nanmedian over the ring; columns with no history stay NaN
                res[start + i, have] = np.nanmedian(ring[:, have], axis=0)
        start = end

    out.loc[order, feats] = res
    return out


def level2_industry_cell(df, feats):
    """Median of the same fyear x sic2D cell.

    Two restrictions, each reported separately so its cost is visible:
      STRICT_AVAIL_CELLS  only rows already available at the target's
                          date_avail may contribute — otherwise a cell-mate
                          with a later fiscal year-end lends a statement
                          published after the row being filled;
      MIN_CELL_FIRMS      the median must rest on at least this many firms.

    Cell statistics are computed on DISTINCT STATEMENTS, deduplicated on
    (gvkey, datadate): a firm matched to several issuer_ids appears several
    times in the sample and must not vote several times. (The three gvkeys with
    two statements in one fyear do count twice in the firm tally.)
    """
    plain = pd.DataFrame(np.nan, index=df.index, columns=feats, dtype="float64")
    restricted = plain.copy()
    firms = plain.copy()

    src = df.drop_duplicates(["gvkey", "datadate"])
    key = [IMPUTE_INDUSTRY, "fyear"]

    for f in feats:
        # (a) plain cell median, the unrestricted comparison
        p = src.groupby(key, dropna=True)[f].median()
        plain[f] = df.set_index(key).index.map(p)

        # (b) availability-restricted running median and firm count
        g = (src[key + ["date_avail", f]].dropna(subset=[IMPUTE_INDUSTRY])
             .sort_values(key + ["date_avail"]))
        grp = g.groupby(key, dropna=True)[f]
        run_med = grp.expanding().median().reset_index(level=[0, 1], drop=True)
        run_cnt = grp.expanding().count().reset_index(level=[0, 1], drop=True)
        g = g.assign(_med=run_med, _cnt=run_cnt)
        # every row sharing a date_avail sees the whole of that date's block
        blk = g.groupby(key + ["date_avail"])
        g["_med"] = blk["_med"].transform("last")
        g["_cnt"] = blk["_cnt"].transform("last")
        lut = g.drop_duplicates(key + ["date_avail"], keep="last") \
               .set_index(key + ["date_avail"])[["_med", "_cnt"]]
        idx = df.set_index(key + ["date_avail"]).index
        restricted[f] = idx.map(lut["_med"])
        firms[f] = idx.map(lut["_cnt"])

    use = restricted if STRICT_AVAIL_CELLS else plain
    thin = firms < MIN_CELL_FIRMS
    final = use.where(~thin)
    return final, plain, use, thin


def impute(df, feats, label):
    """Fill by IMPUTE_METHOD. Returns (filled frame, report dict)."""
    X = df[feats].astype("float64")
    miss = X.isna()
    rep = {"missing": int(miss.values.sum()), "l1": 0, "l2": 0,
           "blocked_avail": 0, "blocked_thin": 0, "indicators": None}
    print(f"\n  IMPUTE_METHOD = {IMPUTE_METHOD}   missing cells "
          f"{rep['missing']:,} of {miss.size:,} ({miss.values.mean()*100:.2f}%)")

    if IMPUTE_METHOD == "complete_case":
        rep["l1_rows_pos"] = 0
        return X, rep

    if IMPUTE_METHOD in ("cs_median", "median_indicator"):
        med = pd.DataFrame({f: df.groupby("fyear")[f].transform("median")
                            .astype("float64") for f in feats})
        filled = X.where(~miss, med)
        if IMPUTE_METHOD == "median_indicator":
            rep["indicators"] = {f: miss[f].astype("int8") for f in feats}
        rep["l2"] = int((miss & med.notna()).values.sum())
        rep["l1_rows_pos"] = 0
        return filled, rep

    # --- hierarchical, the main setting ---------------------------------
    # Exempt features are never filled: their missingness is a fact, not a gap.
    fillable = [f for f in feats if f not in IMPUTE_EXEMPT]
    ex = [f for f in feats if f in IMPUTE_EXEMPT]
    if ex:
        print(f"      IMPUTE_EXEMPT, never filled: {', '.join(ex)}")
    l1 = level1_firm_past(df, feats)
    ok = pd.DataFrame(False, index=X.index, columns=feats)
    ok[fillable] = True                       # exempt columns can never be filled
    f1 = miss & l1.notna() & ok
    after1 = X.where(~f1, l1)

    cell, plain, use, thin = level2_industry_cell(df, feats)
    still = after1.isna() & ok
    f2 = still & cell.notna()
    filled = after1.where(~f2, cell)

    # what each restriction cost, counted only where it actually mattered
    rep["blocked_avail"] = int((still & plain.notna() & use.isna()).values.sum())
    rep["blocked_thin"] = int((still & use.notna() & thin).values.sum())
    rep["l1"] = int(f1.values.sum())
    rep["l2"] = int(f2.values.sum())
    rep["f1"] = f1
    return filled, rep


def report_imputation(df, feats, filled, rep, poscol, label):
    """Everything the run must state about the fill."""
    pos = df[poscol] == 1
    n_pos = int(pos.sum())
    before = df[feats].notna().all(axis=1)
    after = filled.notna().all(axis=1)
    left = int(filled.isna().values.sum())

    print(f"\n  CELLS")
    print(f"      Level 1, firm's own past (LOOKBACK_OBS={LOOKBACK_OBS}) : "
          f"{rep['l1']:>9,}")
    print(f"      Level 2, {IMPUTE_INDUSTRY} x fyear cell                : "
          f"{rep['l2']:>9,}")
    print(f"      left NA (no Level 3)                        : {left:>9,}")
    print(f"      Level-2 fills BLOCKED by STRICT_AVAIL_CELLS : "
          f"{rep['blocked_avail']:>9,}")
    print(f"      Level-2 fills BLOCKED by MIN_CELL_FIRMS={MIN_CELL_FIRMS}    : "
          f"{rep['blocked_thin']:>9,}")

    print(f"\n  ROWS      complete before : {int(before.sum()):>9,} "
          f"({before.mean()*100:5.2f}%)")
    print(f"            complete after  : {int(after.sum()):>9,} "
          f"({after.mean()*100:5.2f}%)")
    print(f"  {poscol}=1     complete before : {int((before & pos).sum()):>9,} "
          f"of {n_pos:,}")
    print(f"            complete after  : {int((after & pos).sum()):>9,} "
          f"of {n_pos:,} ({(after & pos).sum()/n_pos*100:5.2f}%)")

    if left:
        print(f"\n  REMAINING NAs, top 10 features")
        na = filled.isna().sum().sort_values(ascending=False).head(10)
        for f, n in na.items():
            if n:
                print(f"      {f:<24}{int(n):>9,}")

    if "f1" in rep:
        # Where the conservative bias sits: a firm's healthier past written onto
        # its last statements before distress.
        any1 = rep["f1"].any(axis=1)
        print(f"\n  {poscol}=1 rows with at least one LEVEL-1 fill: "
              f"{int((any1 & pos).sum()):,} of {n_pos:,} "
              f"({(any1 & pos).sum()/n_pos*100:.2f}%)  <- the conservative bias")




def run_config():
    """Every setting that shapes the output, stamped into the file."""
    return {"sample_lag": SAMPLE_LAG, "K": K, "window_start": WINDOW_START,
            "window_end": WINDOW_END, "impute_method": IMPUTE_METHOD,
            "lookback_obs": LOOKBACK_OBS, "impute_industry": IMPUTE_INDUSTRY,
            "min_cell_firms": MIN_CELL_FIRMS,
            "strict_avail_cells": STRICT_AVAIL_CELLS,
            "rank_transform": RANK_TRANSFORM, "rank_exempt": RANK_EXEMPT,
            "impute_exempt": IMPUTE_EXEMPT,
            "indicator_na_code": INDICATOR_NA_CODE,
            "always_exclude": sorted(ALWAYS_EXCLUDE), "always_keep": ALWAYS_KEEP,
            "input": str(SAMPLE_FILE.name)}


def rank_within_fyear(df, feats, label):
    """Cross-sectional rank within fyear, ties averaged, mapped to [-0.5, 0.5].

    A single non-missing value in a year ranks at 0.0, the middle, rather than
    at an extreme: with n = 1 the rank carries no cross-sectional information.
    Exempt features pass through untouched.
    """
    todo = [f for f in feats if f not in RANK_EXEMPT]
    exempt = [f for f in feats if f in RANK_EXEMPT]
    out = df[feats].copy()
    for f in todo:
        r = df.groupby("fyear")[f].rank(method="average", pct=True)
        n = df.groupby("fyear")[f].transform("count")
        out[f] = (r - 0.5).where(n > 1, 0.0).where(df[f].notna())
    print(f"  rank transform: {len(todo)} features ranked, {len(exempt)} exempt "
          f"({', '.join(exempt)})")
    # The TRUE minimum across every ranked feature and every year, not the
    # first feature's. A rank across a handful of firms is not a cross-section.
    counts = df.groupby("fyear")[todo].count()
    flat = counts.stack()
    lo_year, lo_feat = flat.idxmin()
    print(f"      fyears {int(df.fyear.min())}-{int(df.fyear.max())}, "
          f"median cross-section {int(counts.median().median()):,} rows")
    print(f"      SMALLEST cross-section across all {len(todo)} ranked features: "
          f"{int(flat.min()):,} rows — {lo_feat} in fyear {int(lo_year)}")
    thin_years = int((counts.min(axis=1) < 30).sum())
    print(f"      fyears where SOME ranked feature has fewer than 30 values: "
          f"{thin_years} of {len(counts)}")
    return out


def apply_window(df, label, poscol):
    """Keep WINDOW_START <= fyear <= WINDOW_END. Reports what it removes."""
    n0, i0 = len(df), df.issuer_id.nunique()
    p0 = int((df[poscol] == 1).sum())
    out = df[(df.fyear >= WINDOW_START) & (df.fyear <= WINDOW_END)].copy()
    p1 = int((out[poscol] == 1).sum())
    print(f"  WINDOW {WINDOW_START}-{WINDOW_END} on {label}: "
          f"rows {n0:,} -> {len(out):,} ({n0-len(out):,} removed)   "
          f"issuers {i0:,} -> {out.issuer_id.nunique():,}   "
          f"{poscol}=1 {p0:,} -> {p1:,} ({p0-p1:,} removed)")
    return out


def main():
    if not SAMPLE_FILE.exists():
        raise SystemExit(
            f"STOP: {SAMPLE_FILE.name} does not exist. Only the L3 distress "
            "sample has been built; SAMPLE_LAG='L0' needs that sample built "
            "first. Nothing was read.")

    distress = pd.read_parquet(SAMPLE_FILE)
    rating = pd.read_parquet(RATING_FILE)

    feats = feature_list(distress)
    # Both samples must start from the same features, or the cut is not
    # comparable across them.
    if feats != feature_list(rating):
        raise SystemExit("STOP: the two samples do not carry the same feature "
                         "columns before filtering.")

    dropped, kept, stats, by_rank = build_drop_list(distress, feats)
    print_cut(dropped, kept, stats, by_rank)

    # The rating validation is only comparable if it runs on the SAME feature
    # set as the model it validates. Asserted after filtering, not assumed.
    d_cols = [c for c in distress.columns if c in kept]
    r_cols = [c for c in rating.columns if c in kept]
    if d_cols != r_cols:
        raise SystemExit(f"STOP: filtered feature columns differ between the "
                         f"samples.\n  distress: {d_cols}\n  rating:   {r_cols}")
    print(f"\n  assertion passed: both samples carry the identical "
          f"{len(d_cols)} feature columns after filtering")

    # ---- the sample window, BEFORE imputation -----------------------------
    # The cut above is settled on the FULL sample; the window narrows the rows
    # the model sees, so cell medians and ranks are computed inside it.
    print("\n" + "=" * 84)
    print(f"SAMPLE WINDOW — fyear {WINDOW_START} to {WINDOW_END} (D-46)")
    print("=" * 84)
    distress = apply_window(distress, "distress", "y")
    rating = apply_window(rating, "rating", "spec_grade")

    payoff(distress, rating, kept)

    # ---- imputation, identical rule and identical features on both --------
    out = {}
    for name, df, poscol in [("DISTRESS " + SAMPLE_LAG, distress, "y"),
                             ("RATING", rating, "spec_grade")]:
        print("\n" + "=" * 84)
        print(f"IMPUTATION — {name}")
        print("=" * 84)
        df = add_industry(df)
        filled, rep = impute(df, kept, name)
        report_imputation(df, kept, filled, rep, poscol, name)
        out[name] = (filled, df, poscol)

    a, b = out["DISTRESS " + SAMPLE_LAG][0], out["RATING"][0]
    if list(a.columns) != list(b.columns):
        raise SystemExit("STOP: feature columns differ between the samples "
                         "after imputation.")
    print(f"\n  assertion passed: both samples carry the identical "
          f"{len(a.columns)} feature columns after imputation")

    # ---- rank transform, identical treatment on both ----------------------
    written = {}
    for name, (filled, frame, poscol) in out.items():
        print("\n" + "=" * 84)
        print(f"RANK TRANSFORM — {name}")
        print("=" * 84)
        feats_out = (rank_within_fyear(frame.assign(**filled), kept, name)
                     if RANK_TRANSFORM else filled)
        # Encode the exempt indicators' NA as its own category.
        if INDICATOR_NA_CODE is not None:
            # Only those exempt features that actually survived the cut: since
            # D-50, NumEarnIncrease is in IMPUTE_EXEMPT but also in
            # ALWAYS_EXCLUDE, so it is not among the kept columns.
            enc_cols = [f for f in IMPUTE_EXEMPT if f in feats_out.columns]
            n_enc = int(feats_out[enc_cols].isna().values.sum())
            feats_out[enc_cols] = feats_out[enc_cols].fillna(
                INDICATOR_NA_CODE)
            print(f"  indicator NA encoded as {INDICATOR_NA_CODE}: "
                  f"{n_enc:,} cells across {len(enc_cols)} kept features "
                  f"({', '.join(enc_cols)}) — rows are no longer dropped "
                  "for these")
        keys = [c for c in ID_COLS + [poscol] if c in frame.columns]
        written[name] = pd.concat([frame[keys].reset_index(drop=True),
                                   feats_out.reset_index(drop=True)], axis=1)

    da = written["DISTRESS " + SAMPLE_LAG]
    rb = written["RATING"]
    if [c for c in da.columns if c in kept] != [c for c in rb.columns if c in kept]:
        raise SystemExit("STOP: feature columns differ between the samples "
                         "after the rank transform.")
    print(f"\n  assertion passed: both samples carry the identical "
          f"{len(kept)} feature columns after the rank transform, with the "
          f"identical transforms (RANK_TRANSFORM={RANK_TRANSFORM}, "
          f"IMPUTE_METHOD={IMPUTE_METHOD})")

    print("\n" + "=" * 84)
    print("WRITING")
    print("=" * 84)
    for path, frame, what in [
            (DISTRESS_OUT, da, "distress, imputed + ranked"),
            (RATING_OUT, rb, "rating, imputed + ranked")]:
        if path.exists():
            raise SystemExit(f"STOP: {path.name} already exists; this script "
                             "never overwrites. Move or rename it first.")
        # Write the full config into the file's own metadata, so a file always
        # carries the settings that produced it.
        table = pa.Table.from_pandas(frame, preserve_index=False)
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}),
             b"stage3_10_config": json.dumps(run_config()).encode()})
        pq.write_table(table, path)
        mb = path.stat().st_size / 1e6
        print(f"  {path.name:<44} {len(frame):>8,} rows x {frame.shape[1]:>3} "
              f"cols  {mb:6.1f} MB   [{what}]")
    # Rows still carrying an NA after imputation cannot be fitted by the
    # elastic net or the random forest. Reported here so stage4_1/stage4_2 do
    # not silently drop them: every model is fitted on the same rows.
    for lab, frame, poscol in [("distress", da, "y"), ("rating", rb, "spec_grade")]:
        na_rows = frame[kept].isna().any(axis=1)
        pos = frame[poscol] == 1
        print(f"  {lab}: {int(na_rows.sum()):,} rows still carry an NA "
              f"({int((na_rows & pos).sum()):,} of them {poscol}=1) — every "
              "model must drop the same rows")
    print("\nNo model fitted here. stage4_1 and stage4_2 fit on these files.")


if __name__ == "__main__":
    main()
