"""Drawing functions for the three rating-inflation figures.

Pure drawing: every number comes in as an argument; nothing is read from disk
and nothing is computed except shares that are passed in. Figures carry no
title and no numeric annotation: titles, sample sizes, and magnitudes belong
in the document's captions and tables.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

US_C, CN_C = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
LETTERS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
           "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC",
           "C", "D"]
CATS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC-C", "D"]
W = 0.4


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURF, "axes.facecolor": SURF,
    })


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1(us_pct, cn_pct, stem):
    """us_pct, cn_pct: 22 shares (%) in notch order AAA ... D."""
    _style()
    x = np.arange(len(LETTERS))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - W / 2 - 0.01, us_pct, W, color=US_C, label="US")
    ax.bar(x + W / 2 + 0.01, cn_pct, W, color=CN_C, label="China")
    ax.axvline(9.5, color=INK2, lw=1, ls="--")
    ax.text(9.35, 49, "investment grade", ha="right", va="top", color=INK2, fontsize=9)
    ax.text(9.65, 49, "speculative grade", ha="left", va="top", color=INK2, fontsize=9)
    ax.set_xticks(x, LETTERS, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Share of rated firm-years (%)")
    ax.set_ylim(0, 50)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.88))
    _save(fig, stem)


def fig2(groups, stem):
    """groups: three tuples (label, us_share8, cn_share8), the shares (%) in
    band8 order AAA, AA, A, BBB, BB, B, CCC-C, D. The label is the panel
    heading, e.g. "Strong fundamentals (score < 0.3)"."""
    _style()
    plt.rcParams["font.size"] = 9
    x = np.arange(len(CATS))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, (lab, su, sc) in zip(axes, groups):
        ax.bar(x - W / 2 - 0.01, su, W, color=US_C, label="US")
        ax.bar(x + W / 2 + 0.01, sc, W, color=CN_C, label="China")
        ax.axvline(3.5, color=INK2, lw=0.9, ls="--")
        ax.set_title(lab, fontsize=9, color=INK)
        ax.set_xticks(x, CATS, rotation=45, ha="right", rotation_mode="anchor")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of firm-years in the group (%)")
    axes[0].set_ylim(0, 100)
    axes[-1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, stem)


def fig3(us_m, us_lo, us_hi, cn_m, cn_lo, cn_hi, stem, model="rf",
         us_ok=None, cn_ok=None):
    """Ten score bands 0.0-0.1 ... 0.9-1.0: mean notch and its 95% CI.

    us_ok / cn_ok: optional ten booleans; a band marked False (below the
    20-firm minimum) is not drawn. Default: every band drawn."""
    us_ok = np.ones(10, bool) if us_ok is None else np.asarray(us_ok, bool)
    cn_ok = np.ones(10, bool) if cn_ok is None else np.asarray(cn_ok, bool)
    mask = lambda v, ok: np.where(ok, np.asarray(v, float), np.nan)
    us_m, us_lo, us_hi = (mask(v, us_ok) for v in (us_m, us_lo, us_hi))
    cn_m, cn_lo, cn_hi = (mask(v, cn_ok) for v in (cn_m, cn_lo, cn_hi))
    _style()
    xc = np.arange(10) * 0.1 + 0.05
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(10.5, 23, color="#f3f2ee", zorder=0)
    for m, lo, hi, c, lab in ((us_m, us_lo, us_hi, US_C, "US"),
                              (cn_m, cn_lo, cn_hi, CN_C, "China")):
        ax.fill_between(xc, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(xc, m, color=c, lw=2, marker="o", ms=5, label=lab)
    ax.set_ylim(23, 0)
    ticks = [1, 3, 6, 9, 12, 15, 18, 21]
    ax.set_yticks(ticks, [LETTERS[t - 1] for t in ticks])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel(f"Rating weakness score band ({model})")
    ax.set_ylabel("Mean rating (notch scale)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    _save(fig, stem)
