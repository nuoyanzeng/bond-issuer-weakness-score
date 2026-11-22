"""Drawing functions for the score, validation, and default figures (rf).

Pure drawing: every number comes in as an argument; nothing is read from disk.
The only computing done here is turning bin counts into shapes for drawing
(shares, Gaussian smoothing for the density curves, cumulative rates for the
ROC curves). The AUCs in the ROC legends are passed in, not computed here.
Figures carry no title: titles, sample sizes, and periods belong in the
document's captions. Same style as inflation_figures.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from inflation_figures import US_C, CN_C, INK, INK2, GRID, _style, _save

GREY = "#9a9893"
FAMILY = {"distress": "Default weakness score",
          "rating": "Rating weakness score"}
OUTCOME = {"intcov": "next-year interest cover < 1x",
           "ebitdaneg": "next-year EBITDA negative",
           "loss": "next-year loss"}
SIGMA = {"distress": 1.5, "rating": 2.5}   # smoothing width, in 0.01 bins
YMAX = {"distress": 21, "rating": 4.2}     # shared y-limit for the two panels


def _smooth(share, sigma):
    """Gaussian smoothing of 0.01-bin shares, reflected at 0 and 1; returns a
    density (area 1 over [0, 1])."""
    nb = len(share)
    v = np.r_[share[::-1], share, share[::-1]]
    k = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    return np.convolve(v, w, mode="same")[nb:2 * nb] * nb


def density(us, cn, family, outcome, stem):
    """us, cn: dicts with n_pos and n_neg (100 counts each, bins of width 0.01
    from 0 to 1, low to high) and auc (exact AUC to print).
    family: 'distress' or 'rating'; outcome: 'intcov', 'ebitdaneg' or 'loss'."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (mkt, d, col) in zip(axes, (("US", us, US_C), ("China", cn, CN_C))):
        pos = np.asarray(d["n_pos"], float)
        neg = np.asarray(d["n_neg"], float)
        if len(pos) != 100 or len(neg) != 100:
            raise ValueError(f"{mkt}: density needs 100 bins")
        mid = np.arange(100) / 100 + 0.005
        for arr, c, lab in ((neg, GREY, "no deterioration"),
                            (pos, col, "deteriorated")):
            y = _smooth(arr / arr.sum(), SIGMA[family])
            ax.fill_between(mid, y, color=c, alpha=0.18, lw=0)
            ax.plot(mid, y, color=c, lw=2, label=lab)
        ax.set_xlim(0, 1)
        ax.set_title(mkt, loc="left", fontsize=10, color=INK)
        ax.set_xlabel(f"{FAMILY[family]} (rf)")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9,
                  loc="upper right" if family == "distress" else "upper left")
    axes[0].set_ylabel("Density")
    axes[0].set_ylim(0, YMAX[family])
    fig.tight_layout()
    _save(fig, stem)


def roc(us, cn, family, outcome, stem):
    """us, cn: dicts with n_pos and n_neg (equal-width bin counts from 0 to 1,
    low to high, any number of bins) and auc (exact AUC to print)."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    for mkt, d, col in (("US", us, US_C), ("China", cn, CN_C)):
        pos = np.asarray(d["n_pos"], float)[::-1]   # high score first
        neg = np.asarray(d["n_neg"], float)[::-1]
        tpr = np.r_[0, np.cumsum(pos) / pos.sum()]
        fpr = np.r_[0, np.cumsum(neg) / neg.sum()]
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{mkt}  AUC {d['auc']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, stem)


# ---- Default figures -------------------------------------------------------


def default_by_band(labels, us, cn, stem):
    """Default rate by band, bands cut on the US distribution of the rf
    distress score. labels: band labels, low to high score. us, cn: dicts
    with rate, lo, hi (in %, one per band) and drawn (bool, one per band;
    a band that is False is not drawn for that market)."""
    _style()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for d, col, lab in ((us, US_C, "US"), (cn, CN_C, "China")):
        ok = np.asarray(d["drawn"], bool)
        rate, lo, hi = (np.asarray(d[k], float) for k in ("rate", "lo", "hi"))
        ax.fill_between(x[ok], lo[ok], hi[ok], color=col, alpha=0.15, lw=0)
        ax.plot(x[ok], rate[ok], color=col, lw=2, marker="o", ms=5, label=lab)
    ax.set_ylim(0, None)
    ax.set_xticks(x, labels)
    ax.set_xlabel("US percentile band of the default weakness score (rf)")
    ax.set_ylabel("Firm-years followed by default (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, stem)


def roc_default(us, cn, stem):
    """ROC for next-year default. us, cn: dicts with fpr and tpr (the curve
    points, starting at 0 and ending at 1) and auc (exact AUC to print)."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    for mkt, d, col in (("US", us, US_C), ("China", cn, CN_C)):
        ax.plot(d["fpr"], d["tpr"], color=col, lw=2,
                label=f"{mkt}  AUC {d['auc']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, stem)


# ---- Score distributions ------------------------------------------------------
def score_distribution(us, cn, stem, xmax=(0.5, 1.0), sigma=(1.5, 2.5)):
    """Two panels, default weakness score and rating weakness score (rf).
    us, cn: dicts with keys "distress" and "rating", each 100 shares of
    issuer-years in bins of width 0.01 from 0 to 1 (low to high), summing to 1.
    xmax, sigma: x-axis limit and smoothing width (in 0.01 bins) per panel."""
    _style()
    mid = np.arange(100) / 100 + 0.005
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, fam, xm, sg in zip(axes, ("distress", "rating"), xmax, sigma):
        for d, col, lab in ((us, US_C, "US"), (cn, CN_C, "China")):
            share = np.asarray(d[fam], float)
            if len(share) != 100:
                raise ValueError("score_distribution needs 100 bins")
            y = _smooth(share / share.sum(), sg)
            m = mid <= xm
            ax.fill_between(mid[m], y[m], color=col, alpha=0.18, lw=0)
            ax.plot(mid[m], y[m], color=col, lw=2, label=lab)
        ax.set_xlim(0, xm)
        ax.set_ylim(0, None)
        ax.set_title(f"{FAMILY[fam]} (rf)", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Raw score")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, stem)


# ---- Score before default ------------------------------------------------------
def early_warning(shares, stem, benchmark=10.0):
    """shares: five values (%), the share of pre-default issuer-years above the
    never-defaulted 90th percentile at t-5, t-4, t-3, t-2, t-1, in that order.
    benchmark: the never-defaulted share above that percentile (10 by
    construction), drawn as a dashed line."""
    _style()
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(benchmark, color=INK2, lw=1, ls="--")
    ax.plot(x, np.asarray(shares, float), color=US_C, lw=2.2, marker="o", ms=6)
    ax.set_xticks(x, ["t\u22125", "t\u22124", "t\u22123", "t\u22122", "t\u22121"])
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Years before the issuer's first default")
    ax.set_ylabel("Share above the never-defaulted 90th percentile (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _save(fig, stem)
