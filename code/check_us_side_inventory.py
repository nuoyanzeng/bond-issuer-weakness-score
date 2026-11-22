"""
Inventory of everything the US side produces. READ-ONLY — writes nothing.

Groups the US modelling chain by what each file IS FOR, because the filenames
alone do not say which scores are safe to use on a US firm and which exist only
for China. Ends by naming the US deliverable and what travels to China.

Run:  python3 code/check_us_side_inventory.py
"""

import json
import re
from pathlib import Path

import pyarrow.parquet as pq

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# A feature-set tag means an arm of the D-62 comparison, not the main line.
# NOT `\b`: an underscore is a word character, so `_s31\b` fails to match
# `_s31_xgb.json` and matches only tags that end the filename. That bug
# misfiled every tagged results/scores file as main line. `(?![0-9])` instead
# anchors on "no more digits", which is what the tag boundary actually is.
TAGGED = re.compile(r"_s(25|27|28|31)(?![0-9])")
# K != 12 means a D-52 sensitivity arm. L0 does NOT qualify: D-53 makes L3 and
# L0 both reported specifications, so both are main line.
OFF_K = re.compile(r"_K(10|14)(?![0-9])")


def shape(p):
    """Rows x columns where the format has them; a short descriptor otherwise."""
    try:
        if p.suffix == ".parquet":
            f = pq.ParquetFile(p)                 # reads the footer, not the data
            return f"{f.metadata.num_rows:,} x {len(f.schema_arrow.names)}"
        if p.suffix == ".csv":
            with open(p) as fh:
                head = fh.readline()
                n = sum(1 for _ in fh)
            return f"{n:,} x {len(head.split(','))}"
        if p.suffix == ".json":
            # A SAVED MODEL is also .json — xgboost serialises natively — so the
            # extension alone cannot tell a model from its metadata. Decide by
            # FILENAME: anything under stage5_0_model that does not end
            # "_meta.json" is the model itself, and gets a size like the
            # .joblib models rather than being described as metadata.
            if p.name.startswith("stage5_0_model") and not p.name.endswith("_meta.json"):
                return f"{p.stat().st_size / 1e6:.1f} MB"
            d = json.load(open(p))
            if "importances" in d:
                auc = d.get("metrics", {}).get("AUC-ROC")
                return f"{len(d['importances'])} feats" + (f", AUC {auc:.4f}" if auc else "")
            if "features_in_order" in d:
                return f"{d.get('n_features', '?')} feats, {d.get('n_rows_fitted', 0):,} rows"
            if "n_rows" in d:
                return f"{d['n_rows']:,} rows"
            return "metadata"
        return f"{p.stat().st_size / 1e6:.1f} MB"
    except Exception as e:                        # never let one odd file stop the report
        return f"(unreadable: {type(e).__name__})"


def outcome_of(n):
    """Which outcome a file belongs to.

    Taken from the STAGE NUMBER wherever that is what decides it — stage4_1 is
    the distress cross-fit, stage4_2 is the rating one — and from the filename
    only for the scripts that write both outcomes themselves. An earlier version
    inferred it from the string "_rating" alone and so labelled every stage4_2
    file "distress", which is the error this function exists to prevent.
    """
    if n.startswith("stage4_1_"):
        return "distress"
    if n.startswith("stage4_2_"):
        return "rating"
    if n.startswith("stage4_0_"):
        return "rating" if n.startswith("stage4_0_rating") else "distress"
    # stage5_0 / stage5_1 write both outcomes; the infix distinguishes them.
    return "rating" if "_rating_" in n else "distress"


def describe(p):
    """One sentence, derived from the filename rather than hand-listed."""
    n = p.name
    out = outcome_of(n)
    m = next((k for k in ("xgb", "rf", "enet") if f"_{k}" in n), None)
    extra = ("  [K-sensitivity run on a SUPERSEDED feature set — evidence for "
             "D-52, not a current model]" if OFF_K.search(n) else "")
    if n.startswith("stage4_0_rating_L0"):
        sib = p.parent / n.replace("_L0_", "_L3_")
        same = ""
        if sib.exists():
            try:
                a, b = (pq.ParquetFile(x).metadata.num_rows for x in (p, sib))
                same = (f"; rows {'==' if a == b else '!='} the L3 file's "
                        f"({a:,} vs {b:,})")
            except Exception:
                pass
        return ("prepared rating panel, L0 — PRESENT BUT UNUSED: no model is "
                "fitted on it. D-42 matches a rating CONCURRENTLY with no "
                f"availability lag, so rating uses L3 only{same}" + extra)
    if n.startswith("stage4_0_"):
        return (f"prepared {out} panel — imputed, rank-transformed; the matrix "
                f"models are fitted on" + extra)
    if n.startswith(("stage4_1_scores", "stage4_2_scores")):
        return (f"CROSS-FITTED out-of-fold {m} scores, {out} — the honest US "
                f"score" + extra)
    if n.startswith(("stage4_1_results", "stage4_2_results")):
        return f"cross-fitted {m} metrics and importances, {out}" + extra
    if n.startswith("stage5_0_model"):
        if "coefficients" in n:
            return f"enet coefficients + scaler, {out} — the portable form that travels"
        if n.endswith("_meta.json"):
            return f"{m} metadata, {out} — features IN ORDER, config, versions"
        return f"FULL-SAMPLE {m} model, {out} — CHINA ONLY, never score a US firm"
    if n.startswith("stage5_1_score_panel"):
        return (f"issuer-year panel, {out} — cross-fitted scores + pooled "
                f"0-100 percentiles" + extra)
    return extra.strip()


def classify(n):
    # K10/K14 are SUPERSEDED, not a robustness arm of the current model. They
    # carry more features than the main line (33 and 29 against 27), so they
    # were built on an earlier feature set: they are evidence for D-52's K
    # sensitivity, not an alternative setting of the model now in force.
    if TAGGED.search(n) or OFF_K.search(n):
        return "SUPERSEDED"
    for pref, grp in (("stage4_0_", "INPUTS"),
                      ("stage4_1_", "CROSSFIT"), ("stage4_2_", "CROSSFIT"),
                      ("stage5_0_model", "FULLSAMPLE"),
                      ("stage5_1_score_panel", "PANELS")):
        if n.startswith(pref):
            return grp
    return None


GROUPS = [
    ("INPUTS", "MODEL INPUTS — the prepared panels"),
    ("CROSSFIT", "CROSS-FITTED RESULTS AND SCORES — the honest US scores"),
    ("FULLSAMPLE", "FULL-SAMPLE MODELS — CHINA ONLY, must never score a US firm"),
    ("PANELS", "SCORE PANELS — the US deliverable"),
    ("SUPERSEDED", "SUPERSEDED / NOT FOR USE — s25/s27/s28/s31 feature-set arms, "
                   "and K10/K14 runs on a superseded feature set"),
]


def main():
    files = sorted(p for p in OUT_DIR.iterdir() if p.is_file())
    bucket = {g: [] for g, _ in GROUPS}
    for p in files:
        g = classify(p.name)
        if g:
            bucket[g].append(p)

    print("=" * 108)
    print("US SIDE INVENTORY — read-only, nothing written")
    print("=" * 108)

    for key, title in GROUPS:
        rows = bucket[key]
        print(f"\n{title}   [{len(rows)} files]")
        print("-" * 108)
        for p in rows:
            print(f"  {p.name:<52}{shape(p):>22}   {describe(p)}")

    zips = [p for p in files if p.suffix == ".zip"]
    print(f"\nARCHIVES   [{len(zips)} files]")
    print("-" * 108)
    for p in zips:
        print(f"  {p.name:<52}{p.stat().st_size / 1e6:>18.1f} MB   "
              "frozen snapshot; not part of the live tree")

    n_listed = sum(len(v) for v in bucket.values()) + len(zips)
    print(f"\n{'=' * 108}")
    print(f"{n_listed} of {len(files)} files in output/ belong to the US chain; "
          f"the rest are earlier stages.")
    print("=" * 108)
    print("  THE US DELIVERABLE : stage5_1_score_panel_L3_K12.{parquet,csv} "
          "(+ the L0 pair, D-53).")
    print("                       Every score in it is CROSS-FITTED, so no firm "
          "was scored by a model")
    print("                       that had seen it. This is the only file whose "
          "scores may be used for a US firm.")
    print()
    print("  WHAT GOES TO CHINA : stage5_0_model_* — the full-sample models and "
          "the enet coefficient CSV.")
    print("                       They saw every US row and their US scores "
          "would be leaked, which is why")
    print("                       they exist ONLY to be applied to Chinese data.")


if __name__ == "__main__":
    main()
