"""
Stage 5.1 — The SCORE PANEL: one row per issuer-year, three models' scores.

THIS IS THE SUPERSET, AND THAT IS WHY IT IS BUILT FIRST. Any "one score per
firm" convention is a ROW SELECTION from this panel. Building the panel first
means the choice of a baseline date costs nothing to postpone — and if the wider
study is organised as a firm-year regression, this panel IS the deliverable and
no timepoint ever has to be chosen.

===========================================================================
!!  EVERY SCORE HERE IS CROSS-FITTED, AND THAT IS WHAT MAKES THE PANEL   !!
!!  USABLE. It is not visible from the data, so it is stated here, in    !!
!!  the metadata file, and in the CSV header.                            !!
!!                                                                       !!
!!  Each score comes from the `cross_val_predict` of stage4_1 (distress) !!
!!  or stage4_2 (rating), over 5 folds                                    !!
!!  grouped by CONNECTED COMPONENTS of issuer_id <-> gvkey (D-48), so    !!
!!  NO FIRM'S SCORE COMES FROM A MODEL TRAINED ON THAT FIRM. That is a   !!
!!  requirement, not best practice: a score that had seen its own firm's !!
!!  realised default would carry the outcome into the second-stage       !!
!!  regression and bias the coefficient the wider study exists to        !!
!!  estimate (BACKGROUND section 2, D-27).                               !!
!!                                                                       !!
!!  The scores are the CALIBRATED ones (out-of-fold sigmoid, D-47),      !!
!!  because a control variable enters as a number and must be honestly   !!
!!  scaled, not merely well-ordered.                                     !!
!!                                                                       !!
!!  DO NOT regenerate these with the stage5_0 models. Those are fitted   !!
!!  on ALL rows and would reintroduce exactly this leak.                 !!
===========================================================================

WHICH OUTCOME. `OUTCOME_SET` selects it. "distress" is the default and behaves
exactly as this script did before the switch existed — same inputs, same
filenames, same metadata. "rating" assembles the US rating panel from stage4_2's
cross-fitted scores instead, and its outputs carry "rating" in their filename.

This script MERGES saved files. It fits nothing, predicts nothing and changes
no value.

Run:  python3 code/stage5_1_score_panel.py                 (distress)
      python3 code/stage5_1_score_panel.py --outcome rating
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

# WHERE OUTPUTS GO. Reads always come from OUT_DIR; only writes follow this, so
# the parity check can re-run this script into a temporary directory without
# breaking its reads. Unset — which is always, outside that check — this IS
# OUT_DIR and nothing behaves differently.
WRITE_DIR = Path(os.environ["BOND_WRITE_DIR"]) if os.environ.get("BOND_WRITE_DIR") else OUT_DIR

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
K = 12

# The FEATURE-SET tag, matching stage3_10 / stage4_1 / stage4_2 / stage5_0.
# Empty means the current chosen specification and the filenames are unchanged.
# It governs BOTH the files this script reads and the files it writes, so a
# tagged panel can never be assembled from untagged scores.
SET_TAG = ""
_T = f"_{SET_TAG}" if SET_TAG else ""

# WHICH OUTCOME. "distress" reproduces this script exactly as it was before the
# switch existed. "rating" assembles the US rating panel from stage4_2's
# cross-fitted scores instead.
# SELECTED ON THE COMMAND LINE, not by editing this file:
#     python3 code/stage5_1_score_panel.py --outcome rating
# With no argument it is "distress", so the default is never edited and cannot
# be left on "rating" by accident after a run.
OUTCOME_SET = "distress"          # "distress" | "rating"
if "--outcome" in sys.argv:
    _i = sys.argv.index("--outcome")
    if _i + 1 >= len(sys.argv):
        raise SystemExit("STOP: --outcome needs a value: distress or rating.")
    OUTCOME_SET = sys.argv[_i + 1]
if OUTCOME_SET not in ("distress", "rating"):
    raise SystemExit(f"STOP: OUTCOME_SET must be 'distress' or 'rating', "
                     f"not {OUTCOME_SET!r}.")
# EMPTY for distress, so every existing filename is unchanged character for
# character; "_rating" otherwise.
_O = "" if OUTCOME_SET == "distress" else f"_{OUTCOME_SET}"

# The cross-fit script for this outcome, and the column its prepared panel
# carries the outcome in. The SCORE files both write use "y" either way, so only
# the prepared panel differs.
SPEC_MODULE = "stage4_1" if OUTCOME_SET == "distress" else "stage4_2"
OUTCOME_COL = "y" if OUTCOME_SET == "distress" else "spec_grade"

# Both availability settings are reported specifications for distress (D-53), so
# both panels are built in one run rather than by flipping a switch twice.
# RATING HAS ONLY L3: D-42 matches a rating CONCURRENTLY, same year, with no
# availability guard, so there is no L0 variant of it on disk. The L3 in its
# filename is stamped from stage3_10's SAMPLE_LAG, which selects only the
# distress input — it labels the run's configuration, not a property of the
# sample.
LAGS = ["L3", "L0"] if OUTCOME_SET == "distress" else ["L3"]
MODELS = ["xgb", "rf", "enet"]

KEY = ["issuer_id", "gvkey", "fyear"]
# Carried through from the prepared panel so the row can be traced back to the
# statement it describes. datadate is the statement date; date_avail is when it
# could first have been known (D-41).
CONTEXT = ["datadate", "date_avail"]

CROSS_FITTED_NOTE = (
    "EVERY score in this file is CROSS-FITTED: 5-fold cross_val_predict grouped "
    "by connected components of issuer_id<->gvkey (D-48), so no firm's score "
    "comes from a model trained on that firm. Scores are out-of-fold "
    "sigmoid-CALIBRATED (D-47), on the 0-1 probability scale. Percentiles are "
    "within this panel, on a 0-100 scale."
)


def panel_path(lag, ext):
    return WRITE_DIR / f"stage5_1_score_panel{_O}_{lag}_K{K}{_T}.{ext}"


def meta_path(lag):
    return WRITE_DIR / f"stage5_1_score_panel{_O}_{lag}_K{K}{_T}_meta.json"


def scores_path(lag, model):
    """The cross-fitted scores for one model.

    stage4_1 names its score files with the lag and K; stage4_2 does not. That
    asymmetry is on disk and is not worth a rename, so it is absorbed here.
    """
    if OUTCOME_SET == "distress":
        return OUT_DIR / f"stage4_1_scores_{lag}_K{K}{_T}_{model}.parquet"
    return OUT_DIR / f"stage4_2_scores{_T}_{model}.parquet"


def results_path(lag, model):
    """The cross-fit results, read only to check the panel's AUC against them."""
    if OUTCOME_SET == "distress":
        return OUT_DIR / f"stage4_1_results_{lag}_K{K}{_T}_{model}.json"
    return OUT_DIR / f"stage4_2_results{_T}_{model}.json"


def prepared_path(lag):
    """The prepared panel supplying row context. One definition, used by both
    the read in build() and the path recorded in the metadata, so the recorded
    path and the opened path cannot diverge (D-55)."""
    return OUT_DIR / f"stage4_0_{OUTCOME_SET}_{lag}_K{K}{_T}.parquet"


def build(lag):
    """Merge the three models' scores onto one issuer-year frame."""
    print("=" * 84)
    print(f"PANEL {OUTCOME_SET} {lag} K={K}")
    print("=" * 84)

    for ext in ("csv", "parquet"):
        if panel_path(lag, ext).exists():
            raise SystemExit(
                f"STOP: {panel_path(lag, ext).name} already exists. This script "
                "never overwrites; move or delete it first.")

    # The prepared panel supplies the row context; the scores supply the scores.
    # The outcome column is `y` on the distress panel and `spec_grade` on the
    # rating one; it is renamed to `y` here so the rest of this function — the
    # merge checks, the report — needs no branch. The SCORE files already use
    # `y` for both outcomes.
    prepared = prepared_path(lag)
    ctx = pd.read_parquet(prepared, columns=KEY + CONTEXT + [OUTCOME_COL])
    ctx = ctx.rename(columns={OUTCOME_COL: "y"})
    print(f"  {prepared.name}: {len(ctx):,} rows")

    out = None
    for m in MODELS:
        p = scores_path(lag, m)
        if not p.exists():
            raise SystemExit(f"STOP: {p.name} is not there. "
                             f"Run {SPEC_MODULE} first.")
        s = pd.read_parquet(p)[KEY + ["y", "score_calibrated"]]
        s = s.rename(columns={"score_calibrated": f"score_{m}"})
        print(f"  {p.name}: {len(s):,} rows")
        if out is None:
            out = s
        else:
            # Capture the length BEFORE the merge. Comparing only against len(s)
            # afterwards cannot detect the first file having FEWER rows than a
            # later one — the inner join would silently equal len(s) and pass.
            # The invariant is that all three files cover the SAME keys, so the
            # merge must equal BOTH sides.
            n_before = len(out)
            # y is carried once, from the first file, and checked against the
            # rest rather than merged repeatedly.
            merged = out.merge(s, on=KEY, suffixes=("", "_chk"))
            if len(merged) != n_before or len(merged) != len(s):
                raise SystemExit(
                    f"STOP: {m} does not cover the same rows as the earlier "
                    f"score files — merge gave {len(merged):,} against "
                    f"{n_before:,} carried so far and {len(s):,} in {p.name}.")
            if not merged.y.equals(merged.y_chk):
                raise SystemExit(
                    f"STOP: the y column in {p.name} disagrees with the first "
                    "score file on the same keys.")
            out = merged.drop(columns="y_chk")

    # Within-panel percentiles, one per model, on a 0-100 SCALE (not 0-1).
    # The unit is stated here, in the metadata and in the docstring because a
    # 0-1 / 0-100 mismatch against another market's percentiles would produce a
    # plausible-looking wrong answer instead of an error.
    for m in MODELS:
        out[f"pct_{m}"] = out[f"score_{m}"].rank(pct=True) * 100

    # Attach the statement context. Inner join: a scored row must exist in the
    # prepared panel, and a failure here would mean the two disagree about the
    # sample, which is worth stopping for.
    before = len(out)
    out = out.merge(ctx.drop(columns="y"), on=KEY, how="inner")
    if len(out) != before:
        raise SystemExit(
            f"STOP: {before - len(out):,} scored rows have no matching row in "
            f"{prepared.name}; the two files disagree about the sample.")

    # Column order: identifiers, statement context, outcome, then the scores.
    cols = KEY + CONTEXT + ["y"] + \
        [c for m in MODELS for c in (f"score_{m}", f"pct_{m}")]
    out = out[cols].sort_values(KEY).reset_index(drop=True)

    # The declared key must actually be unique, or the panel misdescribes itself.
    if out.duplicated(subset=KEY).any():
        raise SystemExit(
            f"STOP: {KEY} is not unique in the panel; "
            f"{int(out.duplicated(subset=KEY).sum()):,} duplicate rows.")
    if out[[c for c in cols if c.startswith(("score_", "pct_"))]].isna().any().any():
        raise SystemExit("STOP: a score or percentile is missing.")

    return out


def report(out, lag):
    """Everything asked for, plus the rows-per-issuer shape."""
    print(f"\n  {'-' * 78}")
    print(f"  PANEL {lag}")
    print(f"  {'-' * 78}")
    print(f"      rows                 : {len(out):,}")
    print(f"      issuers              : {out.issuer_id.nunique():,}")
    print(f"      gvkeys               : {out.gvkey.nunique():,}")
    print(f"      events (y=1)         : {int(out.y.sum()):,} "
          f"({out.y.mean() * 100:.3f}%)")
    print(f"      issuers with an event: {out[out.y == 1].issuer_id.nunique():,}")
    print(f"      fyear range          : {int(out.fyear.min())}-{int(out.fyear.max())}")
    print(f"      datadate range       : {out.datadate.min().date()} to "
          f"{out.datadate.max().date()}")

    # A firm with 25 rows and a firm with 2 are different objects downstream.
    per = out.groupby("issuer_id").size()
    print(f"\n      ISSUER-YEARS PER ISSUER — this is not a balanced panel:")
    print(f"          min {per.min()}   median {per.median():.0f}   "
          f"mean {per.mean():.1f}   max {per.max()}")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
        print(f"          p{int(q * 100):<3}{per.quantile(q):>6.0f}")
    for n in [1, 2, 5]:
        k = int((per <= n).sum())
        print(f"          issuers with <= {n:>2} rows: {k:,} "
              f"({k / len(per) * 100:.1f}% of issuers, "
              f"{int(per[per <= n].sum()):,} rows)")
    print(f"          issuers with >= 20 rows: {int((per >= 20).sum()):,} "
          f"({int(per[per >= 20].sum()):,} rows)")
    print(f"      A downstream panel regression must decide how to weight these; "
          f"this script does not.")

    print(f"\n      SCORE DISTRIBUTION (calibrated, cross-fitted)")
    print(f"          {'model':<8}{'median':>10}{'p90':>10}{'p99':>10}{'max':>10}"
          f"{'median y=1':>13}{'median y=0':>13}")
    for m in MODELS:
        c = out[f"score_{m}"]
        print(f"          {m:<8}{c.median():>10.5f}{c.quantile(.90):>10.5f}"
              f"{c.quantile(.99):>10.5f}{c.max():>10.5f}"
              f"{out[out.y == 1][f'score_{m}'].median():>13.5f}"
              f"{out[out.y == 0][f'score_{m}'].median():>13.5f}")

    # AUC-ROC RECOMPUTED FROM THE ASSEMBLED PANEL, against the value the
    # cross-fit script recorded. The panel is a merge of files, so a disagreement
    # here would mean the merge lost or misaligned rows — which the row-count
    # checks in build() would not necessarily catch if two files happened to
    # share a row count. This READS ONLY and writes nothing.
    #
    # The panel carries the CALIBRATED score and the recorded AUC-ROC is on the
    # raw one. A SINGLE sigmoid would be strictly monotone and leave AUC-ROC
    # exactly unchanged — but D-47 calibrates OUT OF FOLD, so each fold gets its
    # OWN sigmoid, and pooling several different monotone maps does not preserve
    # the global ordering. Expect agreement to about 1e-3, consistently signed,
    # NOT to floating point. Measured on the rating panel: -2e-4 (xgb), -2e-4
    # (rf), -7e-4 (enet). A merge fault would move this by orders of magnitude
    # more and would not be sign-consistent.
    print(f"\n      AUC-ROC, recomputed here vs {SPEC_MODULE}'s record")
    print(f"          {'model':<8}{'panel':>10}{'recorded':>11}{'diff':>12}")
    for m in MODELS:
        auc = roc_auc_score(out.y, out[f"score_{m}"])
        rp = results_path(lag, m)
        # "AUC-ROC" is the key in BOTH stage4_1 and stage4_2 results — checked,
        # not assumed, against the files on disk.
        rec = None
        if rp.exists():
            rec = json.load(open(rp)).get("metrics", {}).get("AUC-ROC")
        if rec is not None:
            diff = auc - rec
            print(f"          {m:<8}{auc:>10.4f}{rec:>11.4f}{diff:>+12.6f}")
        else:
            print(f"          {m:<8}{auc:>10.4f}{'n/a':>11}{'':>12}   "
                  f"(no ROC field found in {rp.name})")


def compare_with_distress(out, lag):
    """Do the distress and rating panels agree on WHO is weak?

    D-51: score rank-correlation and top-decile overlap come FIRST, before any
    aggregate metric. Printed only on the rating run and only if the distress
    panel is already on disk. READS ONLY.
    """
    other = OUT_DIR / f"stage5_1_score_panel_{lag}_K{K}{_T}.parquet"
    if OUTCOME_SET != "rating" or not other.exists():
        return
    d = pd.read_parquet(other)
    both = out.merge(d, on=KEY, suffixes=("_rate", "_dist"))
    print(f"\n      DISTRESS vs RATING — the same issuer-years, both outcomes")
    print(f"          issuer-years in both panels: {len(both):,} "
          f"(rating {len(out):,}, distress {len(d):,})")
    if not len(both):
        return
    print(f"          {'model':<8}{'Spearman':>11}{'top-decile overlap':>21}")
    for m in MODELS:
        a, b = both[f"score_{m}_rate"], both[f"score_{m}_dist"]
        rho = a.corr(b, method="spearman")
        n = max(1, int(round(len(both) * 0.10)))
        top_a = set(a.nlargest(n).index)
        top_b = set(b.nlargest(n).index)
        print(f"          {m:<8}{rho:>11.4f}"
              f"{len(top_a & top_b) / n * 100:>20.1f}%")
    print(f"          A low overlap means the two outcomes flag DIFFERENT firms, "
          f"which is a finding about\n          the outcomes, not a defect — the "
          f"rating panel is a validation tool (D-39), not the deliverable.")


def main():
    print("=" * 84)
    print(f"STAGE 5.1 — SCORE PANEL   outcome={OUTCOME_SET} spec={SPEC_MODULE}")
    print("=" * 84)
    for line in CROSS_FITTED_NOTE.split(". "):
        print(f"  !! {line.strip()}")
    print()

    built = {}
    for lag in LAGS:
        out = build(lag)
        built[lag] = out

        # PLAIN CSV — no comment lines, no preamble, header on line 1, so a bare
        # pd.read_csv() works. An earlier draft wrote the cross-fitting note as
        # two leading '#' lines; those are invisible in a spreadsheet but make
        # pd.read_csv take the first comment as the header unless the reader
        # knows to pass comment='#'. A file that needs special handling to read
        # is a trap. The note lives in the .meta.json and the docstring instead.
        out.to_csv(panel_path(lag, "csv"), index=False)
        out.to_parquet(panel_path(lag, "parquet"), index=False)

        json.dump({
            "cross_fitted": True,
            "NOTE": CROSS_FITTED_NOTE,
            "tag": SET_TAG,
            "lag": lag, "K": K, "models": MODELS,
            "score_column": "score_<model> = out-of-fold sigmoid-calibrated "
                            "probability, on the 0-1 probability scale",
            "percentile_column": (
                "pct_<model> = the score's percentile WITHIN THIS PANEL, on a "
                "0-100 SCALE (not 0-1): pandas rank(pct=True) * 100, so the "
                "largest score is 100.0 and the smallest is just above 0. "
                "ANY OTHER MARKET'S PERCENTILES MUST USE THE SAME 0-100 UNIT "
                "before the two are compared or pooled — a 0-1/0-100 mismatch "
                "produces a plausible-looking wrong answer rather than an error."),
            "csv_format": ("plain CSV, header on line 1, no comment or preamble "
                           "lines: pd.read_csv(path) works with no arguments"),
            "key": KEY,
            "n_rows": len(out), "n_issuers": int(out.issuer_id.nunique()),
            "n_gvkeys": int(out.gvkey.nunique()),
            "n_events": int(out.y.sum()),
            "fyear_min": int(out.fyear.min()), "fyear_max": int(out.fyear.max()),
            # Derived from the SAME functions the reads use, so the recorded
            # path and the opened path cannot diverge (D-55). For distress these
            # render exactly as the hand-written strings they replace. The
            # outcome is not stored under its own key — adding one would change
            # the distress metadata — but it is legible from both of these.
            "source_scores": [scores_path(lag, m).name for m in MODELS],
            "source_panel": prepared_path(lag).name,
            "not_a_balanced_panel": (
                "Issuer-years per issuer vary; see the run output. A downstream "
                "panel regression must decide how to weight them."),
            "when": datetime.now().isoformat(timespec="seconds"),
        }, open(meta_path(lag), "w"), indent=2)

        print(f"\n  wrote {panel_path(lag, 'csv').name}, "
              f"{panel_path(lag, 'parquet').name}, {meta_path(lag).name}")
        report(out, lag)
        compare_with_distress(out, lag)

    # The two lags are different samples by construction (D-53), so the row
    # counts are expected to differ; the comparison is reported, not asserted.
    if len(built) == 2:
        a, b = built["L3"], built["L0"]
        common = a[KEY].merge(b[KEY], on=KEY).shape[0]
        print(f"\n  {'-' * 78}")
        print(f"  L3 vs L0: {len(a):,} and {len(b):,} rows, {common:,} issuer-years "
              f"in both.")
        print(f"  They are different samples by construction (D-53) — L0 removes "
              f"the availability")
        print(f"  guard, so the first-offering cut bites differently. L3 is the "
              f"primary specification.")

    print("\n  Merged from saved files only — nothing fitted, nothing predicted, "
          "no value changed.")


if __name__ == "__main__":
    main()
