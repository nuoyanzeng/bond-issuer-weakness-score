"""
Stage 8.1 — Export the Chinese scores to two CSV files. A COPY, not a
computation.

Every value written is copied from `output/stage6_4_china_score_panel.parquet`.
Nothing is derived, rescaled or recomputed; `pct` is the saved 0-100 percentile
copied as it stands. The only column that does not exist in the panel is
`scored`, which is the panel's own missingness written down as 0/1.

NO EXISTING FILE IS MODIFIED. If either output already exists the script stops
before writing anything, so a re-run cannot silently replace an export someone
has already used.

  output/china_distress_score.csv   score_distress_* and pct_distress_*
  output/china_rating_score.csv     score_rating_*   and pct_rating_*

Both: UTF-8, comma-separated, one header row, all 28,399 panel rows, these ten
columns in this order —

    Orgid, fyear, datadate, scored,
    score_<family>_xgb, score_<family>_rf, score_<family>_enet,
    pct_<family>_xgb,   pct_<family>_rf,   pct_<family>_enet

`Orgid` is the panel's `gvkey` column renamed; every other name is the panel's
own. Unscored rows carry empty score and pct cells.

FLOAT PRECISION, AND THE TRAP IN VERIFYING IT. pandas writes a float with
Python's shortest representation that round-trips exactly, so the written text
loses nothing. But `pd.read_csv` DOES lose digits: its default converter keeps
about 15 significant figures, so reading back
`0.0029576411470770836` returns `0.002957641147077` and a value-by-value
comparison fails on almost every row. Measured on this panel before the check
was written. The read-back below therefore passes
`float_precision="round_trip"`. Anyone else reading these files and needing the
exact values must pass it too.

CHECKS. Four before writing — the script writes nothing if any fails — and a
full value-by-value read-back afterwards.

Run:  python3 code/stage8_1_export_china_scores.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
PANEL_IN = OUT_DIR / "stage6_4_china_score_panel.parquet"

FAMILIES = {"distress": OUT_DIR / "china_distress_score.csv",
            "rating": OUT_DIR / "china_rating_score.csv"}
MODELS = ["xgb", "rf", "enet"]

EXPECT_ROWS = 28399
EXPECT_SCORED = 24557

# The four columns that are not floats, and how each is written as text.
KEY_COLS = ["Orgid", "fyear", "datadate", "scored"]


def family_cols(fam):
    """The six value columns of one family, score first then pct."""
    return ([f"score_{fam}_{m}" for m in MODELS]
            + [f"pct_{fam}_{m}" for m in MODELS])


def report(ok, label, detail=""):
    """Print one check line and return its result."""
    print(f"      {'PASSED' if ok else 'FAILED'}   {label}"
          + (f"   {detail}" if detail else ""))
    return ok


def main():
    print("=" * 92)
    print("STAGE 8.1 — EXPORT THE CHINESE SCORES. Values are COPIED, nothing "
          "is computed.")
    print("=" * 92)

    # ---- refuse to touch an existing export -------------------------------
    existing = [p for p in FAMILIES.values() if p.exists()]
    if existing:
        for p in existing:
            print(f"      {p.name} already exists")
        raise SystemExit("STOP: this script modifies no existing file. Move or "
                         "rename the file(s) above first. NOTHING WRITTEN.")

    d = pd.read_parquet(PANEL_IN)
    print(f"\n  {PANEL_IN.name}: {len(d):,} rows x {d.shape[1]} columns")

    # ---- the checks that run before anything is written -------------------
    print("\n  CHECKS BEFORE WRITING")
    all_scores = [f"score_{f}_{m}" for f in FAMILIES for m in MODELS]
    scored_mask = d[all_scores].notna().all(axis=1)
    n_scored = int(scored_mask.sum())

    ok = []
    ok.append(report(len(d) == EXPECT_ROWS, "row count",
                     f"{len(d):,} (expected {EXPECT_ROWS:,})"))
    ok.append(report(n_scored == EXPECT_SCORED, "scored row count",
                     f"{n_scored:,} (expected {EXPECT_SCORED:,})"))

    dup = int(d.duplicated(["gvkey", "fyear"]).sum())
    ok.append(report(dup == 0, "Orgid x fyear is unique",
                     f"{dup:,} duplicate rows"))

    # Per family: every one of the six columns filled on a scored row and empty
    # on an unscored one. Checked per family rather than in aggregate, so a
    # family scored on a different row set than the other would show up here.
    for fam in FAMILIES:
        cols = family_cols(fam)
        fam_mask = d[cols].notna().all(axis=1)
        n_part = int((d[cols].notna().any(axis=1) & ~fam_mask).sum())
        agree = bool(fam_mask.equals(scored_mask))
        n_bad_empty = int((~scored_mask & d[cols].notna().any(axis=1)).sum())
        ok.append(report(
            agree and n_part == 0 and n_bad_empty == 0,
            f"{fam:<8} 6 columns filled on scored rows, empty on unscored",
            f"filled on {int(fam_mask.sum()):,}; partially filled {n_part:,}; "
            f"values on unscored rows {n_bad_empty:,}; "
            f"agrees with the panel-wide scored mask: {agree}"))

    n_nat = int(d.datadate.isna().sum())
    ok.append(report(n_nat == 0, "datadate has no missing value",
                     f"{n_nat:,} missing"))

    if not all(ok):
        raise SystemExit("\nSTOP: a pre-write check failed. NOTHING WRITTEN.")

    # ---- build and write ---------------------------------------------------
    # datadate is written as an explicit YYYY-MM-DD string so the file carries
    # no dependence on a default datetime format.
    base = pd.DataFrame({
        "Orgid": d.gvkey.astype(str),
        "fyear": d.fyear.astype(int),
        "datadate": d.datadate.dt.strftime("%Y-%m-%d"),
        "scored": scored_mask.astype(int)})

    print("\n  WRITING")
    frames = {}
    for fam, path in FAMILIES.items():
        cols = family_cols(fam)
        out = pd.concat([base, d[cols]], axis=1)
        if list(out.columns) != KEY_COLS + cols:
            raise SystemExit(f"STOP: {path.name} column order is "
                             f"{list(out.columns)}. NOTHING WRITTEN for it.")
        # No float_format: pandas writes the shortest text that round-trips.
        out.to_csv(path, index=False, encoding="utf-8")
        frames[fam] = out
        print(f"      {path.name}: {len(out):,} rows x {out.shape[1]} columns, "
              f"{int(out.scored.sum()):,} scored, "
              f"{path.stat().st_size / 1e6:.2f} MB")
        print(f"          columns: {', '.join(out.columns)}")

    # ---- read back and compare, value by value -----------------------------
    print("\n  READ-BACK, value by value   (float_precision='round_trip')")
    all_ok = True
    for fam, path in FAMILIES.items():
        src = frames[fam]
        back = pd.read_csv(path, dtype={c: str for c in KEY_COLS},
                           float_precision="round_trip", encoding="utf-8")
        checks = []
        checks.append(report(list(back.columns) == list(src.columns),
                             f"{fam:<8} column names and order"))
        checks.append(report(len(back) == len(src), f"{fam:<8} row count",
                             f"{len(back):,}"))
        # The four text columns, compared as the strings the file holds.
        for c in KEY_COLS:
            want = src[c].astype(str)
            got = back[c].astype(str)
            n_diff = int((want.to_numpy() != got.to_numpy()).sum())
            checks.append(report(n_diff == 0, f"{fam:<8} {c}",
                                 f"{n_diff:,} differing values"))
        # The six float columns, missing counting as equal to missing.
        for c in family_cols(fam):
            a, b = src[c].to_numpy(float), back[c].to_numpy(float)
            same = np.array_equal(a, b, equal_nan=True)
            detail = ""
            if not same:
                bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
                i = int(np.flatnonzero(bad)[0])
                detail = (f"{int(bad.sum()):,} differ; first at row {i}: "
                          f"source {a[i]!r} -> file {b[i]!r}")
            checks.append(report(same, f"{fam:<8} {c}", detail))
        all_ok = all_ok and all(checks)

    if not all_ok:
        raise SystemExit("\nSTOP: a written file does not match the source. "
                         "The files above are on disk and are WRONG; delete "
                         "them before anything reads them.")
    print("\n  Both files reproduce the source exactly, value by value.")
    print("=" * 92)


if __name__ == "__main__":
    main()
