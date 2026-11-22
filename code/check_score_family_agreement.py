"""
Read-only check: how closely do the DEFAULT and the RATING weakness scores agree,
for Chinese issuers, with the US as a benchmark on the same definitions?

PRINTS ONLY. It writes no file, changes no existing file, fits no model and does
not read data_raw/.

WHAT IS COMPARED. For each model (xgb, rf, enet) separately: the default
(distress) score against the rating score of the SAME model, on the issuer-years
that carry both. Two measures, both on the pooled panel (all fiscal years
together):
  - the Spearman rank correlation of the two scores;
  - the top-decile overlap: each score is ranked high to low (higher = weaker),
    the top floor(0.1 x N) rows are taken, ties at the cutoff are broken by firm
    identifier then fyear so the result is reproducible, and the overlap is the
    share of one top set that is also in the other. Both sets have N_top rows, so
    the share is the same read from either side.
Only RAW (uncalibrated) scores are used. The percentile columns are never read.

CHINA. output/stage6_4_china_score_panel.parquet, one row per Orgid x fyear
(`gvkey` in that file holds the Orgid). Its six raw score columns are first
checked, value by value, against output/china_distress_score.csv and
output/china_rating_score.csv. Chinese scores are raw by construction (D-67).

US. The cross-fitted raw scores, `score_raw`:
  default  output/stage4_1_scores_L3_K12_{model}.parquet
  rating   output/stage4_2_scores_{model}.parquet
These files are keyed issuer_id x gvkey x fyear: one firm-year appears once per
bond issuer linked to it, with the same score. They are collapsed to one row per
gvkey x fyear (the score is asserted identical inside each cell first), then the
two are matched at gvkey x fyear, so the US unit is the firm-year.

WHY THE US NUMBERS DIFFER FROM THE EARLIER ONES. The earlier US figures (D-68:
51,165 issuer-years, Spearman 0.6410 / 0.7639 / 0.7824, overlap 60.9% / 73.2% /
67.3%) come from stage5_1's `compare_with_distress`, which differs in three ways:
  1. it matched at issuer_id x gvkey x fyear, so a firm-year linked to several
     issuers counts several times (51,165 rows, against about 40,900 firm-years);
  2. it used the CALIBRATED scores;
  3. it took round(0.1 x N), not floor, and broke ties by row order (`nlargest`).
So the script first REPRODUCES those figures with stage5_1's own key, scores and
rule, from the stage5_1 panels on disk, and prints the raw scores on that same key
beside them. That shows how much of any gap is the key and how much is raw versus
calibrated. If the reproduction fails, the script stops before the main table.

HIGHER SCORE = WEAKER on both scores.

Runtime: a few seconds. Memory: well under 100 MB.

Run:  python3 code/check_score_family_agreement.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

MODELS = ["xgb", "rf", "enet"]

# China: the score panel and the two exported files it must agree with.
CN_PANEL = OUT_DIR / "stage6_4_china_score_panel.parquet"
CN_CSV = {"distress": OUT_DIR / "china_distress_score.csv",
          "rating": OUT_DIR / "china_rating_score.csv"}

# US: the cross-fitted score files, raw column `score_raw`.
US_DEFAULT = {m: OUT_DIR / f"stage4_1_scores_L3_K12_{m}.parquet" for m in MODELS}
US_RATING = {m: OUT_DIR / f"stage4_2_scores_{m}.parquet" for m in MODELS}

# US: the stage5_1 panels behind the earlier calibrated figures (D-68).
S51_DEFAULT = OUT_DIR / "stage5_1_score_panel_L3_K12.parquet"
S51_RATING = OUT_DIR / "stage5_1_score_panel_rating_L3_K12.parquet"
S51_KEY = ["issuer_id", "gvkey", "fyear"]

# The earlier calibrated US figures, as recorded in D-68.
EARLIER_N = 51165
EARLIER_RHO = {"xgb": 0.6410, "rf": 0.7639, "enet": 0.7824}
EARLIER_OVERLAP = {"xgb": 60.9, "rf": 73.2, "enet": 67.3}


def stop(text):
    """Halt with a message. Nothing is ever written, so nothing needs undoing."""
    raise SystemExit(f"\nSTOP: {text}")


def report(passed, text):
    """Print one check as PASSED or FAILED, and return whether it passed."""
    print(f"  {'PASSED' if passed else 'FAILED'}   {text}")
    return passed


def top_decile(df, col, id_cols):
    """The rows in the top floor(0.1 x N) of `col`, high to low.

    Ties are broken by `id_cols` in ascending order, so the same input always
    gives the same set. Also returns how many rows share the score at the cutoff,
    and how many of those made it into the top set: if the first number is 1,
    there was no tie at the cutoff.
    """
    n_top = int(np.floor(0.1 * len(df)))
    # A stable sort on score (descending) then the identifiers (ascending).
    order = df.sort_values([col] + id_cols,
                          ascending=[False] + [True] * len(id_cols),
                          kind="mergesort")
    top = order.head(n_top)
    cutoff = top[col].iloc[-1]
    tied_total = int((df[col] == cutoff).sum())
    tied_inside = int((top[col] == cutoff).sum())
    return set(top.index), n_top, cutoff, tied_total, tied_inside


def agreement(df, col_default, col_rating, id_cols, label):
    """Spearman correlation and top-decile overlap of two score columns.

    Prints the tie counts at each cutoff, and returns (rho, overlap %)."""
    rho = df[col_default].corr(df[col_rating], method="spearman")
    top_d, n_top, cut_d, tied_d, in_d = top_decile(df, col_default, id_cols)
    top_r, _, cut_r, tied_r, in_r = top_decile(df, col_rating, id_cols)
    overlap = len(top_d & top_r) / n_top * 100
    print(f"    {label}: top set {n_top:,} of {len(df):,} rows")
    print(f"      default cutoff {cut_d:.10f}: {tied_d:,} row(s) hold this score, "
          f"{in_d:,} inside the top set, {tied_d - in_d:,} left out by the "
          f"tie-break")
    print(f"      rating  cutoff {cut_r:.10f}: {tied_r:,} row(s) hold this score, "
          f"{in_r:,} inside the top set, {tied_r - in_r:,} left out by the "
          f"tie-break")
    return rho, overlap


# ---------------------------------------------------------------------------
# Step 1 — China: read the panel, and check it against the two exported files
# ---------------------------------------------------------------------------
def load_china():
    print("=" * 84)
    print("STEP 1 — CHINA: the score panel, checked against the exported files")
    print("=" * 84)
    score_cols = [f"score_{fam}_{m}" for fam in ("distress", "rating")
                  for m in MODELS]
    panel = pd.read_parquet(CN_PANEL, columns=["gvkey", "fyear"] + score_cols)
    print(f"  file     {CN_PANEL.relative_to(PROJECT_DIR)}  ({len(panel):,} rows)")
    print(f"  columns  gvkey (the Orgid), fyear, " + ", ".join(score_cols))
    print("  (the pct_* percentile columns are not read)")

    # The panel's `gvkey` is the Orgid as text. It is renamed here, and a numeric
    # copy is kept only to break ties in a reproducible order.
    panel = panel.rename(columns={"gvkey": "Orgid"})
    if panel[["Orgid", "fyear"]].isna().any().any():
        stop("the China panel has a missing Orgid or fyear.")
    panel["fyear"] = panel["fyear"].astype("int64")
    panel["orgid_num"] = pd.to_numeric(panel["Orgid"], errors="coerce")
    if panel["orgid_num"].isna().any():
        stop("an Orgid in the China panel is not a whole number, so the "
             "tie-break order cannot be numeric.")

    # Each exported file must hold the same rows and the same raw scores. The CSV
    # is read with the Orgid as text, as the panel stores it, and with
    # round_trip precision, without which pandas drops digits on reading.
    all_ok = True
    for fam, path in CN_CSV.items():
        cols = [f"score_{fam}_{m}" for m in MODELS]
        csv = pd.read_csv(path, usecols=["Orgid", "fyear"] + cols,
                          dtype={"Orgid": str}, float_precision="round_trip")
        print(f"\n  file     {path.relative_to(PROJECT_DIR)}  ({len(csv):,} rows)")
        print(f"  columns  Orgid, fyear, " + ", ".join(cols))
        both = panel[["Orgid", "fyear"] + cols].merge(
            csv, on=["Orgid", "fyear"], how="outer", suffixes=("_panel", "_csv"),
            indicator=True)
        same_rows = (len(csv) == len(panel)
                     and (both["_merge"] == "both").all())
        all_ok &= report(same_rows,
                         f"same Orgid x fyear rows: panel {len(panel):,}, "
                         f"csv {len(csv):,}, unmatched "
                         f"{int((both['_merge'] != 'both').sum()):,}")
        for c in cols:
            a, b = both[f"{c}_panel"], both[f"{c}_csv"]
            same_missing = (a.isna() == b.isna()).all()
            gap = (a - b).abs().max()       # NaN pairs are skipped
            all_ok &= report(same_missing and gap == 0,
                             f"{c}: missing in the same rows "
                             f"({int(a.isna().sum()):,}), largest difference "
                             f"{gap:.3g}")
    if not all_ok:
        stop("the China panel and the exported files disagree. Nothing compared.")
    return panel


# ---------------------------------------------------------------------------
# Step 2 — China: the matched sample of each model
# ---------------------------------------------------------------------------
def china_rows(panel):
    print("\n" + "=" * 84)
    print("STEPS 2-4 — CHINA: matched sample, Spearman, top-decile overlap")
    print("=" * 84)
    print("  unit: issuer-year (Orgid x fyear); ties broken by Orgid "
          "(numeric) then fyear")
    rows = []
    for m in MODELS:
        d_col, r_col = f"score_distress_{m}", f"score_rating_{m}"
        # Keep issuer-years carrying both scores of this model.
        s = panel.dropna(subset=[d_col, r_col])
        if s.duplicated(["Orgid", "fyear"]).any():
            stop(f"China {m}: Orgid x fyear is not unique.")
        print(f"\n  {m}: {len(s):,} issuer-years with both scores, "
              f"{s.Orgid.nunique():,} issuers, fyear {s.fyear.min()}-"
              f"{s.fyear.max()}; Orgid x fyear unique")
        rho, overlap = agreement(s, d_col, r_col, ["orgid_num", "fyear"], m)
        rows.append({"Market": "China", "Model": m, "n": len(s),
                     "firms": s.Orgid.nunique(), "rho": rho, "overlap": overlap})
    return rows


# ---------------------------------------------------------------------------
# Step 5a — US: reproduce the earlier calibrated figures, on stage5_1's own terms
# ---------------------------------------------------------------------------
def reproduce_earlier_us():
    print("\n" + "=" * 84)
    print("STEP 5a — US: the earlier calibrated figures (D-68), reproduced on "
          "stage5_1's own terms")
    print("=" * 84)
    print(f"  files    {S51_RATING.relative_to(PROJECT_DIR)}, "
          f"{S51_DEFAULT.relative_to(PROJECT_DIR)}")
    print(f"  columns  {', '.join(S51_KEY)}, score_xgb, score_rf, score_enet "
          f"(calibrated)")
    print("  key issuer_id x gvkey x fyear; top set round(0.1 x N) by nlargest, "
          "ties by row order")
    cols = S51_KEY + [f"score_{m}" for m in MODELS]
    rating = pd.read_parquet(S51_RATING, columns=cols)
    default = pd.read_parquet(S51_DEFAULT, columns=cols)
    # The same merge, in the same order, as stage5_1's compare_with_distress.
    both = rating.merge(default, on=S51_KEY, suffixes=("_rate", "_dist"))
    ok = report(len(both) == EARLIER_N,
                f"issuer-years in both panels {len(both):,}, earlier "
                f"{EARLIER_N:,}")
    for m in MODELS:
        a, b = both[f"score_{m}_rate"], both[f"score_{m}_dist"]
        rho = a.corr(b, method="spearman")
        n = max(1, int(round(len(both) * 0.10)))
        overlap = len(set(a.nlargest(n).index) & set(b.nlargest(n).index)) / n * 100
        ok &= report(round(rho, 4) == EARLIER_RHO[m]
                     and round(overlap, 1) == EARLIER_OVERLAP[m],
                     f"{m}: Spearman {rho:.4f} (earlier {EARLIER_RHO[m]:.4f}), "
                     f"overlap {overlap:.1f}% (earlier {EARLIER_OVERLAP[m]:.1f}%)")
    if not ok:
        stop("the earlier calibrated US figures do not reproduce, so the US "
             "sample cannot be taken as the same. Nothing further compared.")
    return both


# ---------------------------------------------------------------------------
# Step 5b — US: raw scores, read from the cross-fitted score files
# ---------------------------------------------------------------------------
def load_us_raw(m):
    """Default and rating raw scores of one model, keyed issuer_id x gvkey x fyear."""
    out = {}
    for fam, path in (("default", US_DEFAULT[m]), ("rating", US_RATING[m])):
        s = pd.read_parquet(path, columns=S51_KEY + ["score_raw"])
        print(f"  file     {path.relative_to(PROJECT_DIR)}  ({len(s):,} rows), "
              f"columns {', '.join(S51_KEY)}, score_raw")
        if s.score_raw.isna().any():
            stop(f"{path.name} has a missing score_raw.")
        if s.duplicated(S51_KEY).any():
            stop(f"{path.name} is not unique on {S51_KEY}.")
        out[fam] = s
    return out["default"], out["rating"]


def to_firm_year(s, name):
    """Collapse to one row per gvkey x fyear, after checking the score is the
    same for every issuer linked to that firm-year."""
    spread = s.groupby(["gvkey", "fyear"]).score_raw.agg(lambda v: v.max() - v.min())
    if spread.max() != 0:
        stop(f"{name}: score_raw differs inside a gvkey x fyear cell "
             f"(largest spread {spread.max():.3g}), so it cannot be collapsed.")
    return (s.sort_values(S51_KEY)
             .drop_duplicates(["gvkey", "fyear"])[["gvkey", "fyear", "score_raw"]])


def us_rows():
    print("\n" + "=" * 84)
    print("STEPS 2-4 ON THE US — raw cross-fitted scores")
    print("=" * 84)
    rows, same_key = [], []
    for m in MODELS:
        print(f"\n  {m}")
        default, rating = load_us_raw(m)

        # (i) For reference: raw scores on stage5_1's key and rule, so the effect
        # of raw versus calibrated can be read apart from the effect of the key.
        k = rating.merge(default, on=S51_KEY, suffixes=("_rate", "_dist"))
        a, b = k.score_raw_rate, k.score_raw_dist
        n = max(1, int(round(len(k) * 0.10)))
        same_key.append({"model": m, "n": len(k),
                         "rho": a.corr(b, method="spearman"),
                         "overlap": len(set(a.nlargest(n).index)
                                        & set(b.nlargest(n).index)) / n * 100})

        # (ii) The main definition: one row per gvkey x fyear, matched there.
        d_fy = to_firm_year(default, US_DEFAULT[m].name)
        r_fy = to_firm_year(rating, US_RATING[m].name)
        s = d_fy.merge(r_fy, on=["gvkey", "fyear"], suffixes=("_dist", "_rate"))
        if s.duplicated(["gvkey", "fyear"]).any():
            stop(f"US {m}: gvkey x fyear is not unique after matching.")
        print(f"  {len(s):,} firm-years with both scores, {s.gvkey.nunique():,} "
              f"gvkeys, fyear {s.fyear.min()}-{s.fyear.max()}; gvkey x fyear "
              f"unique (default {len(d_fy):,} firm-years, rating "
              f"{len(r_fy):,})")
        rho, overlap = agreement(s, "score_raw_dist", "score_raw_rate",
                                 ["gvkey", "fyear"], m)
        rows.append({"Market": "US", "Model": m, "n": len(s),
                     "firms": s.gvkey.nunique(), "rho": rho, "overlap": overlap})
    return rows, same_key


# ---------------------------------------------------------------------------
def main():
    panel = load_china()
    cn = china_rows(panel)
    reproduce_earlier_us()
    us, same_key = us_rows()

    # ---- the earlier US figures beside raw ones, on stage5_1's key ----------
    print("\n" + "=" * 84)
    print("US ON STAGE5_1'S KEY (issuer_id x gvkey x fyear) AND RULE — earlier "
          "calibrated beside raw")
    print("=" * 84)
    print(f"  {'model':<7}{'issuer-years':>14}{'Spearman cal':>14}"
          f"{'Spearman raw':>14}{'overlap cal %':>15}{'overlap raw %':>15}")
    for r in same_key:
        m = r["model"]
        print(f"  {m:<7}{r['n']:>14,}{EARLIER_RHO[m]:>14.4f}{r['rho']:>14.4f}"
              f"{EARLIER_OVERLAP[m]:>15.1f}{r['overlap']:>15.1f}")
    print("  cal = the earlier figures (D-68); raw = score_raw from the stage4_1 "
          "and stage4_2 files.")

    # ---- the one table -------------------------------------------------------
    print("\n" + "=" * 84)
    print("TABLE — default versus rating weakness score, pooled over all fiscal "
          "years, raw scores")
    print("=" * 84)
    head = (f"  {'Market':<7}| {'Model':<6}| {'Matched issuer-years':>21} | "
            f"{'Unique issuers':>15} | {'Spearman correlation':>21} | "
            f"{'Top-decile overlap (%)':>23}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in cn + us:
        print(f"  {r['Market']:<7}| {r['Model']:<6}| {r['n']:>21,} | "
              f"{r['firms']:>15,} | {r['rho']:>21.4f} | {r['overlap']:>23.1f}")
    print("  China: issuer-years are Orgid x fyear, issuers are Orgids.")
    print("  US: matched rows are firm-years (gvkey x fyear), issuers are gvkeys; "
          "the score files'\n  issuer_id x gvkey x fyear rows are collapsed to "
          "firm-years first.")
    print("  Top decile = the top floor(0.1 x N) rows of each score, ties broken "
          "by identifier then fyear.")
    print("\nNothing was written.")


if __name__ == "__main__":
    main()
