"""
Stage 2.2 — Cluster the features into themes (D-26).

Reads the winsorised Stage 2 panel and clusters the FEATURES (not the firms) by
how similarly they move, so that a later step can cut the tree into themes.

Two versions are produced from this one script. They run exactly the same
downstream steps and differ only by two switches:

    v1  "JKP baseline"     no demeaning,          Pearson correlation
    v2  "panel-adjusted"   within-firm demeaning, Spearman correlation

Within-firm demeaning subtracts each firm's own average of a feature from that
firm's rows, so what is left is how the firm moves relative to itself over time
rather than how firms differ from one another.

The 0/1 indicators OENEG and INTWO (D-50) are flags, not continuous ratios. They
are EXCLUDED from the clustering entirely — they never enter the correlation
matrix and cannot influence the tree in either version. Afterwards each
indicator is given the cluster of its own parent feature, and the assignment
file shows those separately, marked as inherited rather than clustered.

THEY ARE THE ONLY TWO HELD OUT, so ConvDebt and NumEarnIncrease — also 0/1 —
are clustered here. Stage 2.3's redundancy screen holds out all four and runs
on 51 features (D-58), so the two scripts' feature counts describe different
sets and must not be compared.

Nothing here picks a cluster count, computes cluster means, or chooses a
representative feature. Those are later steps.

Run:  python3 code/stage2_2_cluster.py
"""

from pathlib import Path

import matplotlib

# Draw to a file rather than to a window — this script is run from the terminal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

# Fixed for reproducibility. Ward linkage and fcluster are deterministic, so
# nothing in this script actually varies run to run; the seed is set anyway.
np.random.seed(42)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
PANEL = OUT_DIR / "stage2_features_clean_winsored.parquet"

# The identifier columns in the panel. Everything else is a feature.
KEY_COLS = ["gvkey", "datadate", "fyear"]

# The 0/1 indicators, each mapped to the continuous feature it belongs to. They
# are excluded from the clustering and afterwards inherit their parent's
# cluster. D-50 replaced D-17's four `_negden` flags with two Ohlson (1980)
# states; ConvDebt was always an indicator and was simply never listed here.
# OENEG sits with BookLeverage (both are about the equity/liability balance) and
# INTWO with RoE (both are about persistent losses).
NEGDEN_PARENT = {
    "OENEG": "BookLeverage",
    "INTWO": "RoE",
}

# The tree is cut at each of these cluster counts.
K_VALUES = [10, 12, 13, 15]

# The two versions, as (label, demean?, correlation method).
VERSIONS = [
    ("v1", False, "pearson"),
    ("v2", True, "spearman"),
]


def load_panel():
    """Read only what we need, and split the features into clustered and inherited."""
    # Look at the file's schema first so we can read the feature columns only,
    # rather than loading the whole panel and dropping columns afterwards.
    import pyarrow.parquet as pq

    all_cols = pq.ParquetFile(PANEL).schema_arrow.names
    all_features = [c for c in all_cols if c not in KEY_COLS]

    print(f"Panel: {PANEL.name}")
    print(f"  {len(all_features)} feature columns found")

    # The indicators and their parents must all be present, or the
    # inheritance step below would have nothing to work from.
    missing = [c for c in NEGDEN_PARENT if c not in all_features]
    if missing:
        raise SystemExit(f"STOP: indicator columns absent from the panel: {missing}")

    missing_parents = [p for p in NEGDEN_PARENT.values() if p not in all_features]
    if missing_parents:
        raise SystemExit(
            f"STOP: parent features absent from the panel, so their indicators "
            f"cannot inherit a cluster: {missing_parents}"
        )

    # The clustering runs on everything except the two indicators above.
    # NOTE THE SCOPE, because it differs from Stage 2.3's. Only OENEG and INTWO
    # are held out here, so ConvDebt and NumEarnIncrease — which are 0/1, not
    # continuous ratios — ARE clustered and DO enter this correlation matrix.
    # Stage 2.3's redundancy screen holds out all four and runs on 51 features
    # (D-58). Do not read the two feature counts as describing the same set.
    cluster_features = [c for c in all_features if c not in NEGDEN_PARENT]
    print(f"  {len(cluster_features)} features will be clustered "
          f"(includes ConvDebt and NumEarnIncrease, which are 0/1)")
    print(f"  {len(NEGDEN_PARENT)} 0/1 indicators are EXCLUDED from clustering "
          f"and will inherit their parent's cluster:")
    for indicator, parent in NEGDEN_PARENT.items():
        print(f"      {indicator} -> {parent}")

    # gvkey is needed for the within-firm demeaning in v2. The indicator columns
    # are not read at all, so they cannot reach the correlation matrix.
    panel = pd.read_parquet(PANEL, columns=["gvkey"] + cluster_features)
    print(f"  {len(panel):,} firm-years, {panel['gvkey'].nunique():,} firms")

    # Report the single-year firms, which v2's demeaning turns into rows of
    # exactly zero. Reported only — nothing is dropped or altered.
    rows_per_firm = panel["gvkey"].value_counts()
    single_year_firms = int((rows_per_firm == 1).sum())
    single_year_rows = int(rows_per_firm[rows_per_firm == 1].sum())
    print(f"\n  Single-year firms (affects v2 demeaning only):")
    print(f"      {single_year_firms:,} firms have one year of data")
    print(f"      {single_year_rows:,} rows belong to them "
          f"({single_year_rows / len(panel) * 100:.2f}% of the panel)")
    print(f"      Reported only — none are dropped or altered.")

    return panel, cluster_features


def demean_within_firm(panel, features):
    """Subtract each firm's own mean of each feature from that firm's rows."""
    # transform('mean') gives every row its firm's mean, so the subtraction
    # lines up row by row without an explicit merge.
    firm_means = panel.groupby("gvkey")[features].transform("mean")
    return panel[features] - firm_means


def correlation_matrix(data, method):
    """Pairwise-complete correlation over the features, with a NaN check."""
    # pandas .corr() uses each pair's overlapping non-missing rows, which is the
    # pairwise-complete rule asked for.
    corr = data.corr(method=method)

    # A NaN here means a feature pair shares no observations at all, so no
    # distance can be formed for it. Stop rather than patch the hole.
    if corr.isna().any().any():
        nan_pairs = [
            (r, c)
            for r in corr.index
            for c in corr.columns
            if r < c and pd.isna(corr.loc[r, c])
        ]
        raise SystemExit(
            "STOP: the correlation matrix contains NaN, so these feature pairs "
            f"share no observations and no distance can be formed:\n  {nan_pairs}"
        )

    return corr


def ward_linkage(corr):
    """Distance = 1 - correlation, then Ward on the condensed distance matrix."""
    dist = 1.0 - corr.values

    # Force exact symmetry and an exact zero diagonal. Floating-point arithmetic
    # can leave a value like 1e-17 where a 0 belongs, which squareform rejects
    # as an invalid distance matrix. This is a numerical tidy-up, not a change
    # of method.
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    try:
        condensed = squareform(dist, checks=True)
        link = linkage(condensed, method="ward")
    except Exception as err:
        # Report whether the correlation matrix is positive-definite so the
        # cause is visible, then stop. Do not silently switch to another method.
        eigenvalues = np.linalg.eigvalsh(corr.values)
        raise SystemExit(
            f"STOP: Ward clustering failed: {err}\n"
            f"  minimum eigenvalue of the correlation matrix: {eigenvalues.min():.6g}\n"
            f"  (a negative value means it is not positive-definite)\n"
            "  No fallback method has been applied."
        )

    # How faithfully the tree's distances reproduce the original ones. Closer to
    # 1 is better; well below ~0.7 means the tree is a poor summary.
    coph_corr, _ = cophenet(link, condensed)

    return link, coph_corr


def save_dendrogram(link, features, version, method, demeaned, path):
    """Draw the tree with one labelled leaf per clustered feature."""
    # The labels need vertical room, so the tree is drawn on its side.
    fig, ax = plt.subplots(figsize=(10, 14))
    dendrogram(link, labels=features, orientation="right", ax=ax, leaf_font_size=9)

    demean_text = "within-firm demeaned" if demeaned else "no demeaning"
    ax.set_title(
        f"Stage 2.2 feature clustering — {version}\n"
        f"{method.capitalize()} correlation, {demean_text}, Ward linkage\n"
        f"{len(features)} clustered features; "
        f"{len(NEGDEN_PARENT)} indicators excluded and assigned their parent's cluster"
    )
    ax.set_xlabel("distance (1 - correlation)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_assignments(features, cluster_labels_by_k):
    """The clustered features, then the inherited indicators appended below."""
    # Section 1 — the features that were actually clustered.
    clustered = pd.DataFrame({"feature": features})
    clustered["status"] = "clustered"
    clustered["parent_feature"] = ""
    for k, labels in cluster_labels_by_k.items():
        clustered[f"cluster_k{k}"] = labels

    # Section 2 — the indicators, each taking its parent's cluster at
    # every k. They were never in the correlation matrix.
    inherited_rows = []
    for indicator, parent in NEGDEN_PARENT.items():
        row = {
            "feature": indicator,
            "status": "inherited from parent, not clustered",
            "parent_feature": parent,
        }
        parent_position = features.index(parent)
        for k, labels in cluster_labels_by_k.items():
            row[f"cluster_k{k}"] = labels[parent_position]
        inherited_rows.append(row)
    inherited = pd.DataFrame(inherited_rows)

    # Appended below the clustered block, so the two sections stay visibly apart.
    return pd.concat([clustered, inherited], ignore_index=True)


def run_version(panel, features, version, demeaned, method):
    """Run the full clustering for one version and write its three outputs."""
    print(f"\n{'=' * 70}")
    print(f"{version}: {method} correlation, "
          f"{'within-firm demeaned' if demeaned else 'no demeaning'}")
    print("=" * 70)

    # Step 0 — the two switches.
    data = demean_within_firm(panel, features) if demeaned else panel[features]

    # Steps 1-3 — correlation, distance, Ward. Everything except OENEG and INTWO.
    corr = correlation_matrix(data, method)
    link, coph_corr = ward_linkage(corr)

    # Step 4 — cophenetic correlation.
    print(f"  clustered on {len(features)} features "
          f"({len(NEGDEN_PARENT)} indicators are not in this matrix)")
    print(f"  cophenetic correlation: {coph_corr:.4f}")

    # Step 5 — cut the tree at each k.
    cluster_labels_by_k = {}
    for k in K_VALUES:
        labels = fcluster(link, t=k, criterion="maxclust")
        cluster_labels_by_k[k] = labels

        # Report how the clustered features divide at this k.
        sizes = pd.Series(labels).value_counts().sort_index()
        print(f"  k={k:>2}: sizes {sizes.tolist()}  "
              f"(min {sizes.min()}, max {sizes.max()})")

    assignments = build_assignments(features, cluster_labels_by_k)

    # Write the three outputs for this version.
    assign_path = OUT_DIR / f"stage2_2_cluster_assignments_{version}.csv"
    corr_path = OUT_DIR / f"stage2_2_corr_{version}.parquet"
    dendro_path = OUT_DIR / f"stage2_2_dendrogram_{version}.png"

    assignments.to_csv(assign_path, index=False)
    corr.to_parquet(corr_path)
    save_dendrogram(link, features, version, method, demeaned, dendro_path)

    print(f"  wrote {assign_path.name}, {corr_path.name}, {dendro_path.name}")

    return coph_corr


def main():
    panel, cluster_features = load_panel()

    for version, demeaned, method in VERSIONS:
        run_version(panel, cluster_features, version, demeaned, method)

    print("\nDone. Nothing was chosen from these trees — no cluster count is "
          "selected, no representative feature is picked.")


if __name__ == "__main__":
    main()
