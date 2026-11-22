"""
Stage 2.3 — Identify redundant features and decide which one of each redundant
pair is dropped. Reports only: nothing is removed, no panel is written.

THE RULE IS D-58, and it replaces D-38's rule entirely:

  1. OBJECT. The correlation is computed on the 1993-2025 BOND-ISSUER rows of
     the winsorised Stage 2 panel, not on the full Compustat cross-section.
     D-58: OPLeverage/CapTurnover correlate 0.799 among bond issuers and below
     0.60 on the full panel, so no full-panel cell catches the pair at any
     threshold. The caveat that travels with this: the issuer object is a
     non-random subset - larger, more levered firms - so part of the
     correlation rise is population, not redundancy.
  2. MEASURE. v1 ONLY: Pearson correlation on winsorised levels, no demeaning.
     v2 (Spearman, within-firm demeaned) is NOT consulted. D-54/D-58: the
     screen guards a CROSS-SECTIONAL regression against splitting a coefficient
     between collinear columns, that risk is driven by POOLED correlation, and
     within-firm demeaning removes exactly the between-firm variation that
     generates it.
  3. THRESHOLD. |rho| >= 0.70. At 0.65 the pair roaq/ocf_at clears at 0.679 and
     the tie-break would drop roaq, a CZ characteristic, in favour of one of
     ours; at 0.70 the profitability block is intact.
  4. COLLAPSE. PAIRWISE, not connected components. Clearing pairs are processed
     in |rho| DESCENDING order, ties broken by name. For each pair: if either
     member has already been dropped, SKIP the pair; otherwise drop the member
     with the LOWER average |correlation| to the others and keep the other.
     There is no chaining and no group representative.

The 0/1 indicators never enter the correlation matrix: OENEG and INTWO (D-50),
and ConvDebt and NumEarnIncrease, which are also non-continuous. 51 continuous
features are screened. The indicators are untouched by this script and are
carried through by Stage 2.4.

WHAT THIS SCRIPT DOES NOT DO. It does not adjudicate which member of a pair is
economically right. D-58: in every issuer cell the tie-break drops CapTurnover
and keeps OPLeverage - the feature whose coefficient sign contradicts its own
univariate profile survives, and the one whose sign matches it is dropped. The
screen removes the collinearity; that is all it claims to do.

Run:  python3 code/stage2_3_redundancy.py
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"

# The winsorised Stage 2 panel, in levels. Same file Stage 2.2 clusters on and
# same file Stage 2.4 cuts columns from.
PANEL_IN = OUT_DIR / "stage2_features_clean_winsored.parquet"

# THE ISSUER ROW SET COMES FROM THE BRIDGE, NOT FROM stage3_6.
# stage3_6_features_panel.parquet is the obvious place to take it from, but the
# copy on disk is V2-era: it carries V2's 43 feature columns and a guard that
# asserts them, so it is stale and must not be read. The bridge is upstream of
# every feature decision - it maps FISD issuers to gvkeys and knows nothing
# about which characteristics exist - so it cannot be stale in that sense.
# Taking the row set from it is EXACTLY EQUIVALENT to joining stage3_6 on
# gvkey + datadate, and the equivalence is by construction rather than by
# coincidence:
#   * stage3_6 is an UNFILTERED INNER JOIN of the accepted bridge pairs onto
#     every row of each matched gvkey. It applies no screen of its own beyond
#     dropping the ambiguous match_type, which is done here too.
#   * stage3_5 only ADDS A COLUMN (date_avail) and asserts its own row count is
#     unchanged, so it can neither add nor remove a row.
# Therefore {rows of gvkeys in the accepted bridge} and {rows of stage3_6} are
# the same set, and this script reaches it without reading a file whose feature
# columns have gone stale. The bridge is also STRUCTURALLY IMMUNE to that kind
# of staleness: it maps FISD issuers to gvkeys and knows nothing about which
# characteristics exist, so no feature-set change can invalidate it.
BRIDGE_IN = OUT_DIR / "stage3_4_issuer_gvkey_bridge.parquet"

# Bridge rows with this match type are unresolved one-to-many candidates and
# are not part of the sample. Same rule as stage3_6.
AMBIGUOUS = "crsp_ambiguous"

CORR_OUT = OUT_DIR / "stage2_3_redundancy_corr_issuers.parquet"
PAIRS_OUT = OUT_DIR / "stage2_3_redundancy_pairs.csv"
DROPS_OUT = OUT_DIR / "stage2_3_redundancy_drops.csv"

KEY_COLS = ["gvkey", "datadate", "fyear"]

# The sample window, D-46. Applied on fyear, inclusive at both ends.
WINDOW_START = 1993
WINDOW_END = 2025

# D-58's threshold.
THRESHOLD = 0.70

# The 0/1 indicators, excluded from the correlation matrix. OENEG and INTWO are
# D-50's Ohlson states; ConvDebt and NumEarnIncrease are the other two
# non-continuous features (they are also the two exempted from winsorising by
# D-25 alongside the indicators).
NON_CONTINUOUS = ["OENEG", "INTWO", "ConvDebt", "NumEarnIncrease"]

# HOW THE TIE-BREAK'S "average |correlation| to the others" IS READ.
# D-58 inherits the phrase from D-38, where "the others" meant the other
# members of the connected-component group. Pairwise collapse has no groups, so
# the phrase needs a scope. This script reads it as the average |correlation| to
# ALL OTHER FEATURES IN THE MATRIX, computed once from the fixed correlation
# matrix before any dropping starts. That keeps the rule model-free, order-
# independent and free of the components that "pairwise" was chosen to abandon.
# The alternative reading - average within the connected component the pair
# belongs to - would reintroduce components through the back door.
# The expected results below test this reading against D-58's published table.
TIEBREAK_SCOPE = "all other features in the matrix"

# WHAT D-58's DIAGNOSTIC REPORTED for exactly this cell (issuers, 0.70). These
# are reproduction checks on a diagnostic that was run in a previous session and
# whose script was not kept. A mismatch means this script's object or tie-break
# is not the one that produced the decision, which must be resolved before the
# screen is applied - so each is a hard stop, not a warning.
EXPECTED_ISSUER_ROWS = 97_678
EXPECTED_FEATURES = 51
EXPECTED_PAIRS = 11
EXPECTED_DROPPED = 10
EXPECTED_LEFT = 41

# The pair the object was chosen to catch, and which way the tie-break goes on
# it. D-58: rho 0.799 among issuers, CapTurnover dropped, OPLeverage kept.
#
# THE ARITHMETIC IDENTITY, which verifies the OBJECT rather than asserting it.
# D-54 decomposed this pair's v1 correlation across the four bases:
#       full panel                        0.581
#       + window 1993-2025               -0.126
#       + bond issuers                   +0.344   <- the dominant term
#       + within-fyear ranks             +0.070   <- NOT applied here
# so 0.581 - 0.126 + 0.344 = 0.799 exactly, and the ranked figure would be
# 0.869. Measuring 0.799 therefore confirms two things at once: the row set is
# the issuer object, AND the screen is running on WINSORISED LEVELS rather than
# on the rank-transformed values the model sees. An independent identity is
# worth more than an assertion that the implementation matches the diagnostic.
EXPECTED_MARKER_PAIR = ("CapTurnover", "OPLeverage")
EXPECTED_MARKER_RHO = 0.799
EXPECTED_MARKER_DROPPED = "CapTurnover"


def issuer_gvkeys():
    """The gvkeys of bond issuers, from the bridge, with the ambiguous rows out."""
    br = pd.read_parquet(BRIDGE_IN)
    print(f"  {BRIDGE_IN.name}: {len(br):,} rows")

    accepted = br[br.match_type != AMBIGUOUS]
    print(f"      dropped '{AMBIGUOUS}': {len(br) - len(accepted):,} rows")

    pairs = accepted[["issuer_id", "gvkey"]].drop_duplicates()
    gvkeys = set(pairs.gvkey)
    print(f"      accepted issuers: {pairs.issuer_id.nunique():,}   "
          f"(issuer, gvkey) pairs: {len(pairs):,}   distinct gvkeys: {len(gvkeys):,}")
    return gvkeys


def load_object():
    """The 1993-2025 bond-issuer rows of the winsorised panel, continuous features only."""
    print("=" * 78)
    print("STEP 1 — the object the screen is computed on (D-58)")
    print("=" * 78)

    gvkeys = issuer_gvkeys()

    panel = pd.read_parquet(PANEL_IN)
    print(f"\n  {PANEL_IN.name}: {len(panel):,} rows, {panel.gvkey.nunique():,} firms")

    # The two cuts, reported one at a time so each one's cost is visible.
    issuer_rows = panel[panel.gvkey.isin(gvkeys)]
    print(f"      bond issuers only          : {len(issuer_rows):,} rows, "
          f"{issuer_rows.gvkey.nunique():,} firms")

    obj = issuer_rows[(issuer_rows.fyear >= WINDOW_START) &
                      (issuer_rows.fyear <= WINDOW_END)]
    print(f"      fyear {WINDOW_START}-{WINDOW_END}            : {len(obj):,} rows, "
          f"{obj.gvkey.nunique():,} firms")

    # A gvkey present in the bridge but absent from the panel contributes
    # nothing and is not an error; report the count rather than let it pass.
    absent = len(gvkeys - set(panel.gvkey))
    print(f"      issuer gvkeys absent from the panel: {absent:,} "
          f"(they contribute no rows)")

    if len(obj) != EXPECTED_ISSUER_ROWS:
        raise SystemExit(
            f"STOP: the issuer object has {len(obj):,} rows against D-58's "
            f"{EXPECTED_ISSUER_ROWS:,}. This script's row set is not the one the "
            "decision was measured on. Resolve before applying the screen."
        )
    print(f"\n  CHECK: {len(obj):,} rows matches D-58's {EXPECTED_ISSUER_ROWS:,}.")

    # The features that are screened.
    all_features = [c for c in panel.columns if c not in KEY_COLS]
    absent_indicators = [c for c in NON_CONTINUOUS if c not in all_features]
    if absent_indicators:
        raise SystemExit(
            f"STOP: non-continuous features named in NON_CONTINUOUS are not in "
            f"the panel: {absent_indicators}"
        )
    features = [c for c in all_features if c not in NON_CONTINUOUS]

    print(f"\n  {len(all_features)} features in the panel")
    print(f"      {len(NON_CONTINUOUS)} non-continuous, EXCLUDED from the matrix: "
          f"{', '.join(NON_CONTINUOUS)}")
    print(f"      {len(features)} continuous features are screened")

    if len(features) != EXPECTED_FEATURES:
        raise SystemExit(
            f"STOP: {len(features)} continuous features against D-58's "
            f"{EXPECTED_FEATURES}."
        )

    return obj[features], features


def correlation(data, features):
    """Pearson, undemeaned, pairwise-complete, with the overlap counts reported."""
    print("\n" + "=" * 78)
    print("STEP 2 — v1 correlation: Pearson, undemeaned, on winsorised levels")
    print("=" * 78)

    corr = data.corr(method="pearson")

    # A NaN means a pair shares no observations at all. Stop rather than patch.
    if corr.isna().any().any():
        nan_pairs = [(r, c) for r in corr.index for c in corr.columns
                     if r < c and pd.isna(corr.loc[r, c])]
        raise SystemExit(
            "STOP: the correlation matrix contains NaN, so these pairs share no "
            f"observations:\n  {nan_pairs}"
        )

    # Every correlation here rests on a different number of rows, because the
    # object is only ~98k rows and some features are sparse on it. Report the
    # thinnest pairs so a correlation resting on very little is not mistaken for
    # one resting on the whole object.
    overlap = data.notna().astype(float).T.dot(data.notna().astype(float))
    off_diagonal = [(overlap.loc[a, b], a, b)
                    for i, a in enumerate(features) for b in features[i + 1:]]
    off_diagonal.sort()
    print(f"  pairwise-complete overlap over {len(off_diagonal):,} pairs:")
    print(f"      minimum {int(off_diagonal[0][0]):,} rows   "
          f"maximum {int(off_diagonal[-1][0]):,} rows")
    print("      five thinnest pairs:")
    for n, a, b in off_diagonal[:5]:
        print(f"          {a} / {b}: {int(n):,} rows")

    corr.to_parquet(CORR_OUT)
    print(f"\n  wrote {CORR_OUT.name}  ({corr.shape[0]}x{corr.shape[1]})")
    return corr


def average_abs_corr_to_others(corr, features):
    """Each feature's mean |correlation| to every other feature in the matrix."""
    # The tie-break score, computed once, before anything is dropped. See
    # TIEBREAK_SCOPE above for how "the others" is being read.
    scores = {}
    for f in features:
        others = [o for o in features if o != f]
        scores[f] = sum(abs(corr.loc[f, o]) for o in others) / len(others)
    return scores


def clearing_pairs(corr, features):
    """All unordered pairs at or above the threshold, in |rho| descending order."""
    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            rho = corr.loc[a, b]
            if abs(rho) >= THRESHOLD:
                # Names are stored sorted so the tie-break on name is stable.
                lo, hi = sorted([a, b])
                pairs.append({"a": lo, "b": hi, "rho": rho, "abs_rho": abs(rho)})

    # |rho| descending; ties broken by name so the order is reproducible.
    pairs.sort(key=lambda p: (-p["abs_rho"], p["a"], p["b"]))
    return pairs


def collapse(pairs, scores):
    """D-58's pairwise collapse. Returns the pair table with each decision on it."""
    dropped = set()
    rows = []

    for order, p in enumerate(pairs, start=1):
        a, b = p["a"], p["b"]
        row = dict(p, order=order,
                   avg_abs_corr_a=round(scores[a], 4),
                   avg_abs_corr_b=round(scores[b], 4))

        # If either member is already gone the pair is resolved; skip it. This
        # is what stops A-B and B-C from chaining A, B and C into one group.
        if a in dropped or b in dropped:
            row["action"] = "skipped"
            row["dropped"] = ""
            row["kept"] = ""
            row["reason"] = (f"already dropped: "
                             f"{', '.join(m for m in (a, b) if m in dropped)}")
            rows.append(row)
            continue

        # Drop the member with the LOWER average |correlation| to the others.
        if scores[a] < scores[b]:
            loser, winner, reason = a, b, "lower average |corr| to the others"
        elif scores[b] < scores[a]:
            loser, winner, reason = b, a, "lower average |corr| to the others"
        else:
            # Exactly equal scores are not expected on real data; break by name
            # so the outcome is still deterministic if it ever happens.
            loser, winner, reason = b, a, "TIE on the score, broken by name"

        dropped.add(loser)
        row["action"] = "collapsed"
        row["dropped"] = loser
        row["kept"] = winner
        row["reason"] = reason
        rows.append(row)

    return pd.DataFrame(rows), dropped


def main():
    data, features = load_object()
    corr = correlation(data, features)

    print("\n" + "=" * 78)
    print(f"STEP 3 — pairwise collapse at |rho| >= {THRESHOLD} (D-58)")
    print("=" * 78)

    scores = average_abs_corr_to_others(corr, features)
    pairs = clearing_pairs(corr, features)
    print(f"  pairs clearing |rho| >= {THRESHOLD}: {len(pairs)}")
    print(f"  tie-break score: average |correlation| to {TIEBREAK_SCOPE}, "
          "computed before any dropping")

    table, dropped = collapse(pairs, scores)

    print("\n  every clearing pair, in the order processed:")
    print(table[["order", "a", "b", "rho", "avg_abs_corr_a", "avg_abs_corr_b",
                 "action", "dropped", "kept"]]
          .round(4).to_string(index=False))

    kept = [f for f in features if f not in dropped]

    # ---- the reproduction checks against D-58 ------------------------------
    print("\n" + "=" * 78)
    print("STEP 4 — checks against D-58's published diagnostic")
    print("=" * 78)

    checks = [
        ("clearing pairs", len(pairs), EXPECTED_PAIRS),
        ("features dropped", len(dropped), EXPECTED_DROPPED),
        ("continuous features left", len(kept), EXPECTED_LEFT),
    ]
    for label, got, want in checks:
        mark = "OK" if got == want else "MISMATCH"
        print(f"  {label:<28} {got:>3}   D-58 says {want:>3}   {mark}")

    # The marker pair, which is the whole reason the object was changed.
    ma, mb = EXPECTED_MARKER_PAIR
    marker = table[(table.a == ma) & (table.b == mb)]
    if marker.empty:
        print(f"\n  {ma}/{mb}: DID NOT CLEAR the threshold on this object.")
    else:
        m = marker.iloc[0]
        print(f"\n  {ma}/{mb}: rho {m.rho:.4f}   D-58 says {EXPECTED_MARKER_RHO}")
        print(f"      dropped {m.dropped or '(skipped)'}   "
              f"D-58 says {EXPECTED_MARKER_DROPPED}")

    failures = [f"{label}: got {got}, D-58 says {want}"
                for label, got, want in checks if got != want]
    if marker.empty:
        failures.append(f"{ma}/{mb} did not clear the threshold")
    elif marker.iloc[0]["dropped"] != EXPECTED_MARKER_DROPPED:
        failures.append(
            f"{ma}/{mb} dropped {marker.iloc[0]['dropped'] or '(skipped)'}, "
            f"D-58 says {EXPECTED_MARKER_DROPPED}")

    # ---- what the screen produces ------------------------------------------
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  continuous features screened : {len(features)}")
    print(f"  dropped                      : {len(dropped)}")
    for name in sorted(dropped):
        cause = table[table.dropped == name].iloc[0]
        print(f"      {name:<20} for {cause.kept} (rho {cause.rho:.4f})")
    print(f"  continuous features kept     : {len(kept)}")
    print(f"  plus the {len(NON_CONTINUOUS)} non-continuous, never screened: "
          f"{', '.join(NON_CONTINUOUS)}")
    print(f"  FEATURE SET FOR STAGE 2.4    : {len(kept) + len(NON_CONTINUOUS)}")

    drops = pd.DataFrame([
        {"dropped": name,
         "kept_instead": table[table.dropped == name].iloc[0]["kept"],
         "rho": round(table[table.dropped == name].iloc[0]["rho"], 4)}
        for name in sorted(dropped)
    ])

    table.to_csv(PAIRS_OUT, index=False)
    drops.to_csv(DROPS_OUT, index=False)
    print(f"\n  wrote {PAIRS_OUT.name} and {DROPS_OUT.name}")
    print("  Nothing was removed from any panel; no panel was written.")

    # Stop LAST, so the full report above is visible even when a check fails.
    if failures:
        raise SystemExit(
            "\nSTOP: this run does not reproduce D-58's diagnostic:\n  "
            + "\n  ".join(failures)
            + "\nThe object or the tie-break scope differs from the one the "
              "decision was measured on. Do NOT apply this screen until it is "
              "resolved."
        )


if __name__ == "__main__":
    main()
