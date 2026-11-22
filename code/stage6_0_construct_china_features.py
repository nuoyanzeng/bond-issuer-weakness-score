"""
Stage 6.0 — Construct the Chinese feature panel from CSMAR.

WHAT MAKES THIS CORRECT IS WHAT IT DOES *NOT* DO: it does not reimplement a
single formula. It builds a COMPUSTAT-SHAPED frame out of CSMAR — columns
literally named `at`, `lt`, `ni`, `ceq`, `oancf`, keyed on `gvkey` and `fyear` —
and then calls stage1_1's and stage1_construct_features_2's OWN functions on it,
unchanged.

That is not a stylistic preference. Coefficient transfer assumes the feature is
the same quantity in both markets, and re-deriving 37 formulas by hand would
make that true only by careful copying. Calling the same functions makes it true
BY CONSTRUCTION, and it means D-50's non-positive-denominator masking, the
book-equity fallback chain and the 18-August source-paper restrictions cannot
silently diverge between the two markets — they live inside those functions.

SCOPE: the 37 features of the 45 that survive the D-58 screen and are computable
from CSMAR. **NOT the 29 that also survive the US K = 12 screen.** The K screen
ranks on US positive-row missingness; China's missingness structure differs, so
building only the 29 would impose the US pattern on Chinese data and foreclose a
Chinese K screen before it can be run. Same reasoning as D-02.

THE 8 BLOCKED FEATURES ARE NOT BUILT AND NOT APPROXIMATED AROUND. Their absence
is reported. See `output/stage3_9_compustat_csmar_mapping.md`.

WHAT THIS SCRIPT DOES NOT DO: no imputation, no rank transform, no scoring, and
no availability alignment. Each has unresolved decisions of its own — the
Chinese imputation has no industry classification to condition on, and PRC
filing deadlines (4 months annual, 2 months interim) differ from the SEC ones
that D-41's L = 3 and D-31's three-month carry encode — and mixing them in here
would hide which stage lost what.

Run:  python3 code/stage6_0_construct_china_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
CHINA_DIR = PROJECT_DIR / "data_raw" / "china"

# code/ is not on sys.path when this is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_1_construct_features as s1
import stage1_construct_features_2 as s2

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# The Chinese sample window — the counterpart to D-46's 1993 for the US, and a
# DECISION rather than attrition discovered later. 87.1% of issuer-years are
# 2015 or later; t-1 availability is 64.8% before 2015 against 81.2% after; and
# pre-2012 cross-sections of 300-600 issuers are too thin for a within-year rank
# to mean anything. 2016 is reported alongside so the choice stays visible.
WINDOW_START = 2015
WINDOW_ALT = 2016
WINDOW_END = 2024

# roavol's rolling window. THE ONE PLACE A PARAMETER CHANGES.
#
# CZ's US construction is a rolling sd of quarterly ROA over 16 QUARTERS with
# min_periods 8 — FOUR calendar years, minimum two. The Chinese sub-annual panel
# is SEMI-ANNUAL, not quarterly: H1 and FY are 98% of sub-annual issuer-periods,
# Q1 and Q3 together are 1.8%, and only 7.2% of issuers ever file one. So "16
# periods" and "4 years" are different things here and we must choose which to
# match.
#
# WE MATCH THE CALENDAR WINDOW: 8 half-years, min_periods 4. Matching the period
# count instead would measure dispersion over EIGHT years where the US measures
# four, absorbing low-frequency variation the US measure excludes — a different
# economic quantity. Matching the window keeps the horizon identical and leaves
# only sampling frequency differing, where the level effect (half-year earnings
# are roughly twice quarterly ones) is monotone and therefore invariant under
# D-44's within-year ranking, and the seasonality effect makes the Chinese
# measure slightly CLEANER rather than different.
#
# The calendar-matched version also has the higher coverage — 64.8% of annual
# rows against 37.0%. THAT IS A HAPPY ACCIDENT, NOT THE REASON. The choice is
# made on fidelity; it should not later be read as coverage-driven.
#
# RESIDUAL CAVEAT, recorded rather than resolved: the Chinese roavol is identical
# to the US quantity in HORIZON and ECONOMICS, not in COMPOSITION — half-year
# sampling averages out quarterly seasonality the US measure contains. Loose
# enough to state, not loose enough to block.
ROAVOL_WINDOW = 8
ROAVOL_MIN_PERIODS = 4

# Reconciliation tolerances.
SCALE_TOL = 1e-4       # how close a ratio must sit to a power of ten
ROUNDING_TOL = 1e-9    # below this, two values are the same number

PANEL_IN_45 = OUT_DIR / "stage2_4_features_reduced.parquet"   # defines the 45
PANEL_OUT = OUT_DIR / "stage6_0_china_features.parquet"
REPORT_OUT = OUT_DIR / "stage6_0_china_coverage.csv"

# Export these five tables from CSMAR (https://data.csmar.com/) through an
# authorized account. Preserve the column codes and three-row CSV header, then
# place the files with these exact names in data_raw/china/.
FILES = {"bs": "企业债公司债资产负债表.csv",          # balance sheets
         "is": "企业债公司债利润分配表.csv",          # income statements
         "cfd": "企业债公司债现金流量表(直接法).csv",  # direct cash flows
         "cfi": "企业债公司债现金流量表(间接法).csv",  # indirect cash flows
         "bm": "债券基本情况表.csv"}              # bond-to-issuer master

KEY = ["gvkey", "datadate", "fyear"]
NON_FEATURE = {"gvkey", "datadate", "fyear"}

# The 8 features with no CSMAR route. NOT built, NOT approximated around.
BLOCKED = {
    "WW": "needs dvpsx_c (no usable dividend indicator: the only cash-flow line "
          "mixes dividends with interest and is positive for 99.8% of a "
          "bond-issuer sample) and sic3D (no industry classification in CSMAR, "
          "and the aggregate would be over bond issuers rather than the economy)",
    "TotalAccruals": "needs prstkc; no PRC buyback line. CZ's pre-1990 "
                     "balance-sheet branch would avoid it but would make the "
                     "two markets' feature a different quantity",
    "XFIN": "needs prstkc, as TotalAccruals",
    "GP": "needs an industry code for Novy-Marx's financial-firm exclusion. "
          "CHECKED against all five CSMAR files: no industry classification "
          "exists in any of them",
    "BookLeverage": "needs pstkrv and pstkl; no PRC counterpart",
    "ConvDebt": "needs dc and cshrc; convertible detail not disclosed",
    "NOA": "needs dc, as ConvDebt",
    "Tax": "needs txfo and txfed; PRC does not split current tax by jurisdiction",
}

# NOT BLOCKED BY DATA — DEFERRED PENDING A DECISION. Distinct from BLOCKED
# above, which is an absence of Chinese line items. Every input here exists; what
# does not exist is a faithful construction on a SEMI-ANNUAL grid.
DEFERRED = {
    "NumEarnIncrease":
        "CZ counts consecutive QUARTERS of year-over-year earnings growth, and "
        "the function hard-codes both constants: `shift(4)` for the "
        "year-over-year change, and an eight-rung ladder capped at 8. On a "
        "half-year grid `shift(4)` becomes a TWO-year change rather than a "
        "year-over-year one, and the 8-rung ladder spans FOUR years rather than "
        "two. Unlike roavol there is no single window parameter to re-point: "
        "halving the ladder to four rungs would change the variable's RANGE "
        "from 0-8 to 0-4, which changes what the number is and not merely its "
        "horizon. Building it would mean rewriting CZ's function, which is what "
        "this script's architecture exists to prevent. FLAGGED, NOT DECIDED.",
}

# ---------------------------------------------------------------------------
# THE ITEM MAP — Compustat item -> CSMAR, transcribed from stage3_9 PART 2.
# Each entry is (file, [(code, sign), ...]). A multi-code entry is a sum, or a
# difference where the sign is -1; every one of those combinations is a decision
# recorded in the mapping document, not an invention here.
# ---------------------------------------------------------------------------
ITEM_MAP = {
    # --- balance sheet, DIRECT ---
    "at":     ("bs", [("A001000000", 1)]),
    "act":    ("bs", [("A001100000", 1)]),
    "lct":    ("bs", [("A002100000", 1)]),
    "lt":     ("bs", [("A002000000", 1)]),
    "invt":   ("bs", [("A001123000", 1)]),
    # --- balance sheet, PROXY ---
    "che":    ("bs", [("A001101000", 1)]),        # 货币资金 only: narrower
    "ivst":   ("bs", [("A001107000", 1)]),
    "rect":   ("bs", [("A001111000", 1)]),        # excludes 应收票据, 其他应收款
    "ap":     ("bs", [("A002108000", 1)]),        # excludes 应付票据
    "aco":    ("bs", [("A001125000", 1)]),
    "lco":    ("bs", [("A002126000", 1)]),
    "lo":     ("bs", [("A002209000", 1)]),
    "ppent":  ("bs", [("A001212000", 1), ("A001213000", 1)]),   # + 在建工程
    "intan":  ("bs", [("A001218000", 1)]),        # excludes 商誉
    "ivao":   ("bs", [("A001205000", 1)]),
    "dlc":    ("bs", [("A002101000", 1), ("A002125000", 1)]),
    "dltt":   ("bs", [("A002201000", 1), ("A002203000", 1), ("A002204000", 1)]),
    "ceq":    ("bs", [("A003100000", 1)]),
    "seq":    ("bs", [("A003100000", 1)]),        # same line as ceq; GAAP §2
    "mib":    ("bs", [("A003200000", 1)]),
    "txp":    ("bs", [("A002113000", 1)]),        # all taxes payable, incl. VAT
    "txditc": ("bs", [("A002208000", 1)]),
    "pstk":   ("bs", [("A003112100", 1)]),
    # --- income statement ---
    "revt":   ("is", [("B001100000", 1)]),
    "txt":    ("is", [("B002100000", 1)]),
    "sale":   ("is", [("B001101000", 1)]),
    "cogs":   ("is", [("B001201000", 1)]),
    "xsga":   ("is", [("B001209000", 1), ("B001210000", 1)]),
    "ni":     ("is", [("B002000101", 1)]),
    "ib":     ("is", [("B002000101", 1)]),        # collapses onto ni; GAAP §1
    "ebit":   ("is", [("B001300000", 1), ("B001211000", 1)]),   # add back 财务费用
    "nopi":   ("is", [("B001400000", 1), ("B001500000", -1)]),
    "epspx":  ("is", [("B003000000", 1)]),
    # --- cash flow, direct ---
    "oancf":  ("cfd", [("C001000000", 1)]),
    "ivncf":  ("cfd", [("C002000000", 1)]),
    "fincf":  ("cfd", [("C003000000", 1)]),
    "capx":   ("cfd", [("C002006000", 1)]),
    "dv":     ("cfd", [("C003006000", 1)]),       # mixes dividends and interest
    "sstk":   ("cfd", [("C003008000", 1)]),
    # --- cash flow, indirect ---
    "dp":     ("cfi", [("D000103", 1), ("D000104", 1), ("D000105", 1)]),
    "txdi":   ("cfi", [("D000111", 1), ("D000112", 1)]),
}

# Items flagged `not 1:1` or below 70% coverage in stage3_9 — the "serious"
# proxies, reported per feature so a reader can see what each rests on.
SERIOUS_PROXY = {"ap", "che", "dv", "epspx", "ib", "intan", "ivst", "lco", "lo",
                 "oancf", "pstk", "rect", "seq", "txdi", "txditc", "txp"}

# The three quarterly features, handled by INPUT SUBSTITUTION rather than by
# rewriting a formula. See the module docstring.
#   roaq / roavol — read from the SEMI-ANNUAL frame: CZ divides one period's
#       income by the previous period's assets, which on this grid is one
#       half-year over the previous half-year. Near-uniform, since only 7.2% of
#       issuers could ever contribute a true quarter, and a uniform monotone
#       rescaling leaves D-44's within-year ranks unchanged.
#   Cash — cheq/atq is a stock over a stock, so the ANNUAL analogue is exact
#       rather than a substitution. Read from the annual frame.
FROM_SEMIANNUAL = {"roaq", "roavol"}
FROM_ANNUAL_AS_Q = {"Cash"}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _unwrap(s):
    """CSMAR ships every cell as `="value"`. Strip that to the value."""
    return (s.astype(str).str.replace(r'^="?|"?$', "", regex=True)
            .str.strip('"').str.strip())


def read_csmar(key, want):
    """Read named columns from one CSMAR file.

    Rows 2 and 3 of every file are the Chinese column names and the units, not
    data, so they are skipped. Everything is read as text and converted after
    unwrapping, because a column read natively would be typed from its first
    chunk and the `="..."` wrapper makes every column look like text anyway.
    """
    path = CHINA_DIR / FILES[key]
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    cols = [c.replace('="', "").replace('"', "").strip() for c in header.columns]
    missing = [w for w in want if w not in cols]
    if missing:
        raise SystemExit(f"STOP: {FILES[key]} has no column {missing}")
    idx = sorted(cols.index(w) for w in want)
    d = pd.read_csv(path, usecols=idx, skiprows=[1, 2], header=0, dtype=str,
                    encoding="utf-8-sig", low_memory=False)
    d.columns = [cols[i] for i in idx]
    for c in d.columns:
        d[c] = _unwrap(d[c])
    return d


def codes_for(fkey):
    """Every CSMAR code this script needs out of one file."""
    return sorted({c for _, (f, parts) in ITEM_MAP.items() if f == fkey
                   for c, _ in parts})


def load_statements():
    """All four statements, consolidated rows only, joined to the issuer."""
    print("=" * 84)
    print("STEP 1 — read CSMAR, consolidated statements only")
    print("=" * 84)

    bridge = read_csmar("bm", ["Liscd", "Sctcd", "Orgid"]).drop_duplicates()
    print(f"  bond master: {len(bridge):,} (bond, issuer) pairs, "
          f"{bridge.Orgid.nunique():,} issuers")

    frames = {}
    for fkey in ["bs", "is", "cfd", "cfi"]:
        codes = codes_for(fkey)
        d = read_csmar(fkey, ["Liscd", "Sctcd", "Accper", "Typrep"] + codes)
        n_all = len(d)
        # Typrep 'A' is 合并报表, the consolidated statement, and is the analogue
        # of Compustat's indfmt == 'INDL': the SAME firm-periods under a second
        # presentation, not additional coverage. Measured on the balance sheet,
        # 197,668 of B's 197,723 bond-periods also appear under A, so taking A
        # costs 55 bond-periods — NOT the 48.9% of rows that B represents. That
        # figure has been quoted as a cost and it is not one.
        d = d[d.Typrep.eq("A")].drop(columns="Typrep")
        for c in codes:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.merge(bridge, on=["Liscd", "Sctcd"], how="left")
        unmatched = int(d.Orgid.isna().sum())
        d = d[d.Orgid.notna()].drop(columns=["Liscd", "Sctcd"])
        print(f"  {fkey:<4} {n_all:>9,} rows -> {len(d):>9,} consolidated & matched "
              f"({unmatched:,} rows with no issuer)")
        frames[fkey] = d
    return frames


# ---------------------------------------------------------------------------
# Reconciliation — at ROW level, not item level
# ---------------------------------------------------------------------------
def _power_of_ten(ratio):
    """True where a ratio sits within tolerance of 10^k for an integer k >= 1."""
    if not np.isfinite(ratio) or ratio <= 0:
        return False
    k = np.log10(ratio)
    return abs(k - round(k)) < SCALE_TOL and round(k) >= 1


def reconcile(d, value_cols, label, ref_log=None):
    """Collapse to one row per (Orgid, Accper), detecting units errors.

    THE PROBLEM. 326 multi-bond issuer-periods disagree on total assets, 77% of
    them by 10% or more, and the worst share a mantissa and differ by a power of
    ten — 8.707515e+08 against 8.707515e+12. A "most non-missing values" rule
    cannot see this: both rows are fully populated, so it picks arbitrarily
    between a correct figure and one wrong by four orders of magnitude. Because
    nearly every feature is asset-scaled, one bad `at` corrupts that firm-year's
    ENTIRE vector, and a ranked feature then places the firm at an extreme on
    everything at once.

    THE RULE, three branches, each counted:
      1. identical      — the rows agree; take the one with most values present.
      2. scaling        — the rows are related by a power of ten, so it is a
                          units error and the right magnitude is identifiable.
                          Pick the candidate closest to the firm's own
                          neighbouring periods.
      3. irreconcilable — NOT a clean power of ten. DO NOT GUESS: drop the
                          issuer-period and report which issuer it belonged to.

    Detection is at ROW level on purpose: a scaling error is a property of the
    source row, so dropping the row carries the correction to every other
    monetary item that row holds, which item-level detection would miss.
    """
    print(f"\n  reconciling {label}: {len(d):,} rows")
    sizes = d.groupby(["Orgid", "Accper"], sort=False).size()
    keys_single = sizes[sizes == 1].index
    keys_multi = sizes[sizes > 1].index
    idx = d.set_index(["Orgid", "Accper"]).sort_index()

    singles = (idx.loc[keys_single].reset_index() if len(keys_single)
               else idx.head(0).reset_index())
    print(f"      single-row issuer-periods : {len(singles):,}")
    print(f"      multi-row issuer-periods  : {len(keys_multi):,}")

    counts = {"identical": 0, "scaling": 0, "irreconcilable": 0}
    dropped = {}
    kept = []

    # The overwhelming majority of multi-row groups are the SAME statement filed
    # against several bonds, and resolving those one at a time in Python would
    # dominate the runtime for no benefit. Settle them vectorised, and loop only
    # over the groups that genuinely disagree.
    multi = idx.loc[keys_multi] if len(keys_multi) else idx.head(0)
    if len(multi):
        grp = multi.groupby(level=[0, 1], sort=False)[value_cols]
        span = (grp.max() - grp.min()).abs()
        scale = grp.max().abs()
        rel = (span / scale.replace(0, np.nan)).fillna(0)
        agrees = (rel < ROUNDING_TOL).all(axis=1)
        keys_same = agrees[agrees].index
        keys_diff = agrees[~agrees].index
        counts["identical"] = len(keys_same)
        if len(keys_same):
            same = multi.loc[keys_same].copy()
            same["_n"] = same[value_cols].notna().sum(axis=1)
            same = (same.sort_values("_n")
                        .groupby(level=[0, 1], sort=False).tail(1)
                        .drop(columns="_n"))
            kept.append(same)
    else:
        keys_diff = []

    for key in keys_diff:
        rows = idx.loc[[key]]
        vals = rows[value_cols]
        # Branch 2 — related by a power of ten? Anchor on `at` where present,
        # since it is the item every scaled ratio depends on.
        anchor_col = "at" if "at" in rows.columns else value_cols[0]
        cand = rows[anchor_col].dropna()
        cand = cand[cand > 0]
        uniq = np.unique(cand.to_numpy())
        is_scaling = len(uniq) > 1 and all(
            _power_of_ten(v / uniq.min()) or abs(v / uniq.min() - 1) < ROUNDING_TOL
            for v in uniq)
        if is_scaling and ref_log is not None and key[0] in ref_log.index:
            target = ref_log.loc[key[0]]
            best = min(uniq, key=lambda v: abs(np.log10(v) - target))
            pick = rows[rows[anchor_col].eq(best)]
            counts["scaling"] += 1
            kept.append(pick.iloc[[int(pick[value_cols].notna().sum(axis=1)
                                       .to_numpy().argmax())]])
            continue
        # Branch 3 — cannot be resolved without guessing.
        counts["irreconcilable"] += 1
        dropped[key[0]] = dropped.get(key[0], 0) + 1

    out = pd.concat([singles] + [k.reset_index() for k in kept], ignore_index=True) \
        if kept else singles
    n_groups = len(keys_single) + len(keys_multi)
    disagreed = counts["scaling"] + counts["irreconcilable"]
    print(f"      identical                  : {counts['identical']:,}")
    print(f"      RECONCILED as a units error: {counts['scaling']:,}")
    print(f"      DROPPED as irreconcilable  : {counts['irreconcilable']:,}")
    # THE RULE MOSTLY REFUSES RATHER THAN REPAIRS, and "reconciliation" invites
    # the opposite reading. Say so in the log: someone reading this in three
    # months should not have to infer that most disagreements went unresolved.
    if disagreed:
        print(f"      => of {disagreed:,} genuinely disagreeing issuer-periods, "
              f"RECONCILED {counts['scaling']:,} and DROPPED "
              f"{counts['irreconcilable']:,} "
              f"({counts['irreconcilable']/max(n_groups,1)*100:.2f}% of all "
              f"issuer-periods). The units-error branch resolves the minority; "
              f"most disagreements are ordinary conflicting numbers, which are "
              f"REFUSED rather than repaired.")
    if dropped:
        s = pd.Series(dropped).sort_values(ascending=False)
        conc = s.head(10).sum() / s.sum() * 100
        print(f"      -> {len(s):,} issuers affected; periods dropped per issuer: "
              f"max {int(s.max())}, median {s.median():.0f}")
        print(f"      -> worst 10 issuers (Orgid: periods dropped)")
        for o, n in s.head(10).items():
            print(f"           {o}: {int(n)}")
        if conc > 50:
            print(f"      -> the worst 10 hold {conc:.1f}% of dropped periods: "
                  f"CONCENTRATED, a filer-level problem. A subgroup could be "
                  f"excluded and the bias characterised.")
        else:
            print(f"      -> the worst 10 hold only {conc:.1f}% of dropped periods: "
                  f"DISPERSED. This is SCATTERED DATA ERROR, not a few bad "
                  f"reporters — so there is no subgroup to exclude and no "
                  f"systematic bias to correct for.")
    return out


def reference_magnitude(frames):
    """log10(median total assets) per issuer, from UNDISPUTED periods only.

    Built from issuer-periods that have exactly one balance-sheet row, so the
    reference cannot be contaminated by the rows it is used to adjudicate.
    """
    bs = frames["bs"]
    sizes = bs.groupby(["Orgid", "Accper"]).size()
    clean_keys = sizes[sizes == 1].index
    clean = bs.set_index(["Orgid", "Accper"]).loc[clean_keys].reset_index()
    clean = clean[clean["A001000000"] > 0]
    return np.log10(clean.groupby("Orgid")["A001000000"].median())


# ---------------------------------------------------------------------------
# Building the Compustat-shaped frame
# ---------------------------------------------------------------------------
def build_items(frames, period_ends, label, ref_log):
    """Assemble one Compustat-shaped frame for the given period ends."""
    per_file = {}
    for fkey, d in frames.items():
        sel = d[d.Accper.str.slice(5, 10).isin(period_ends)]
        per_file[fkey] = reconcile(sel, codes_for(fkey), f"{label} / {fkey}",
                                   ref_log if fkey == "bs" else None)

    base = per_file["bs"]
    for fkey in ["is", "cfd", "cfi"]:
        base = base.merge(per_file[fkey], on=["Orgid", "Accper"], how="left")

    out = base[["Orgid", "Accper"]].copy()
    for item, (_, parts) in ITEM_MAP.items():
        col = None
        for code, sign in parts:
            if code not in base.columns:
                continue
            v = base[code] * sign
            col = v if col is None else col.add(v, fill_value=0)
        out[item] = col
    return out


def to_frame(items, semiannual=False):
    """Rename CSMAR keys to the names stage1_1's helpers expect."""
    d = items.rename(columns={"Orgid": "gvkey"}).copy()
    d["datadate"] = pd.to_datetime(d["Accper"], errors="coerce")
    d["fyear"] = d["datadate"].dt.year
    if semiannual:
        # A DENSE half-year index, the analogue of CZ's quarter index: H1 = 0,
        # FY = 1 within each year, so consecutive periods differ by exactly one
        # and a rolling window cannot silently reach across a gap.
        is_fy = items["Accper"].str.slice(5, 10).eq("12-31").astype(int)
        d["qindex"] = d["fyear"] * 2 + is_fy.values
    return d.drop(columns="Accper").sort_values(["gvkey", "fyear"]).reset_index(drop=True)


def china_roavol(q):
    """roavol on the semi-annual grid — THE ONLY FUNCTION NOT REUSED.

    CZ's `q_roavol` hard-codes `rolling(16, min_periods=8)` and a 15-period
    warm-up, which on a half-year grid would be an eight-year window. Everything
    else here is line-for-line CZ's: same lag, same ratio, same rolling sd, same
    warm-up rule expressed against the configured window. Only the two window
    constants differ, for the reason set out at ROAVOL_WINDOW.
    """
    d = s2.qlag(q, "atq", 1)
    d["roaq_"] = s1.safe_div(d["ibq"], d["q1_atq"]).astype("float64")
    g = s2.dense_quarters(d[["gvkey", "qindex", "roaq_"]], ["roaq_"])
    rolled = g["roaq_"].rolling(ROAVOL_WINDOW, min_periods=ROAVOL_MIN_PERIODS).std()
    within_firm = g.groupby("gvkey").cumcount() >= (ROAVOL_MIN_PERIODS - 1)
    g["roavol_"] = rolled.where(within_firm)
    return s2.back_to_quarters(q, g, "roavol_")


def registry_of(name):
    """Where a feature's formula lives, and what it needs."""
    if name in s1.FEATURES:
        fn, items = s1.FEATURES[name]
        return fn, items, "annual"
    if name in s2.ANNUAL_FEATURES:
        fn, items = s2.ANNUAL_FEATURES[name]
        return fn, items, "annual"
    fn, items, _ = s2.QUARTERLY_FEATURES[name]
    return fn, items, "quarterly"


def main():
    if PANEL_OUT.exists():
        raise SystemExit(f"STOP: {PANEL_OUT.name} exists; this script never overwrites.")

    # ---- the target set, read from the panel rather than retyped ----------
    import pyarrow.parquet as pq
    the_45 = [c for c in pq.ParquetFile(PANEL_IN_45).schema_arrow.names
              if c not in NON_FEATURE]
    if len(the_45) != 45:
        raise SystemExit(f"STOP: expected 45 features in {PANEL_IN_45.name}, "
                         f"found {len(the_45)}.")
    missing_blocked = [b for b in BLOCKED if b not in the_45]
    if missing_blocked:
        raise SystemExit(f"STOP: BLOCKED names not in the 45: {missing_blocked}")
    targets = [f for f in the_45 if f not in BLOCKED and f not in DEFERRED]
    if len(targets) + len(DEFERRED) + len(BLOCKED) != len(the_45):
        raise SystemExit("STOP: targets + deferred + blocked does not equal 45.")
    print(f"\n  {len(the_45)} features = {len(targets)} built + "
          f"{len(DEFERRED)} deferred + {len(BLOCKED)} blocked")

    frames = load_statements()
    ref_log = reference_magnitude(frames)

    print("\n" + "=" * 84)
    print("STEP 2 — reconcile and assemble, ANNUAL (12-31)")
    print("=" * 84)
    ann = build_items(frames, {"12-31"}, "annual", ref_log)

    print("\n" + "=" * 84)
    print("STEP 3 — reconcile and assemble, SEMI-ANNUAL (06-30 + 12-31)")
    print("=" * 84)
    sub = build_items(frames, {"06-30", "12-31"}, "semi-annual", ref_log)

    m = to_frame(ann)
    q = to_frame(sub, semiannual=True)
    q["ibq"], q["atq"] = q["ib"], q["at"]
    m_q = to_frame(ann, semiannual=True)
    m_q["cheq"], m_q["atq"] = m_q["che"], m_q["at"]

    # Every item each target needs must be one we actually supply.
    # The sub-annual names are DERIVED onto the q / m_q frames below, so they
    # are supplied even though they are not keys of ITEM_MAP.
    supplied = set(ITEM_MAP) | {"fyear", "gvkey", "datadate",
                                "ibq", "atq", "cheq"}
    for f in targets:
        _, items, _ = registry_of(f)
        gap = [i for i in items if i not in supplied]
        if gap:
            raise SystemExit(f"STOP: {f} needs unmapped items {gap}. The mapping "
                             "and the blocked list disagree; resolve before building.")

    print("\n" + "=" * 84)
    print(f"STEP 4 — construct {len(targets)} features; "
          f"{len(BLOCKED)} BLOCKED and not built")
    print("=" * 84)
    for f, why in BLOCKED.items():
        print(f"  BLOCKED   {f:<16}{why}\n")
    for f, why in DEFERRED.items():
        print(f"  DEFERRED  {f:<16}{why}\n")

    print(f"\n  building:")
    panel = m[KEY].copy()
    for f in targets:
        fn, items, kind = registry_of(f)
        if kind == "annual":
            panel[f] = fn(m)
        elif f == "roavol":
            v = china_roavol(q)
            panel[f] = _to_annual(q, v, m)
        elif f in FROM_SEMIANNUAL:
            v = fn(q)
            panel[f] = _to_annual(q, v, m)
        else:                                   # FROM_ANNUAL_AS_Q
            panel[f] = np.asarray(fn(m_q))
        n = int(panel[f].notna().sum())
        print(f"      {f:<24}{n:>9,} non-missing ({n/len(panel)*100:>6.2f}%)")

    report(panel, targets, m)
    panel.to_parquet(PANEL_OUT, index=False)
    print(f"\nWrote {PANEL_OUT.name}: {len(panel):,} rows x {panel.shape[1]} columns")
    print("No imputation, no rank transform, no scoring, no availability alignment.")


def _to_annual(q, values, m):
    """Read a semi-annual series off at the FY period and place it on the annual row.

    NOT an availability question and so NOT D-31's three-month carry: the annual
    row IS the FY period, so the value is taken at that same period rather than
    carried forward from an earlier one. Nothing is imputed and nothing is
    staled. When availability alignment is added later it will have to settle
    the PRC filing deadlines, which differ from the SEC ones D-31 and D-41 encode.

    NEVER ALIGN TWO FRAMES BY POSITION WHEN A KEY EXISTS. An earlier draft of
    this function assigned the semi-annual results onto the annual rows
    positionally. It would not have raised: `roaq`, `roavol` and `Cash` would
    have landed on the WRONG ROWS while showing normal ranges and normal
    coverage, so the panel would have been silently wrong throughout and nothing
    downstream would have flagged it. The merge below is on (gvkey, fyear).
    """
    src = q[["gvkey", "qindex", "fyear"]].copy()
    src["v"] = np.asarray(values)
    src = src[src["qindex"] % 2 == 1]                  # FY periods only
    src = src.drop_duplicates(["gvkey", "fyear"], keep="last")
    return m[["gvkey", "fyear"]].merge(src[["gvkey", "fyear", "v"]],
                                       on=["gvkey", "fyear"], how="left")["v"].values


def report(panel, feats, m):
    """Coverage per feature and across the panel, on both candidate windows."""
    # Which features use a lag: the registry lists `fyear` for exactly those.
    lagged = set()
    deps = {}
    for f in feats:
        _, items, _ = registry_of(f)
        if "fyear" in items:
            lagged.add(f)
        deps[f] = sorted(i for i in items if i in SERIOUS_PROXY)

    has_prev = set(zip(m.gvkey, m.fyear + 1))          # (gvkey, y) has a y-1 row
    prev_ok = pd.Series([(g, y) in has_prev for g, y in zip(panel.gvkey, panel.fyear)],
                        index=panel.index)

    rows = []
    for start in (WINDOW_START, WINDOW_ALT):
        w = panel[(panel.fyear >= start) & (panel.fyear <= WINDOW_END)]
        pw = prev_ok.loc[w.index]
        print("\n" + "=" * 84)
        print(f"COVERAGE — window {start}-{WINDOW_END}"
              + ("   <- WINDOW_START" if start == WINDOW_START else "   (alternative)"))
        print("=" * 84)
        print(f"  rows {len(w):,}   issuers {w.gvkey.nunique():,}   "
              f"years {int(w.fyear.min())}-{int(w.fyear.max())}")
        print(f"  issuer-years with a usable t-1: {int(pw.sum()):,} "
              f"({pw.mean()*100:.1f}%)")
        complete = int(w[feats].notna().all(axis=1).sum())
        print(f"  issuer-years with a COMPLETE {len(feats)}-feature vector: "
              f"{complete:,} ({complete/len(w)*100:.2f}%)")

        print(f"\n  {'feature':<22}{'cov %':>8}{'lag?':>6}{'lost: no t-1':>14}"
              f"{'lost: item':>12}   serious proxies")
        for f in sorted(feats, key=lambda x: w[x].notna().mean()):
            miss = w[f].isna()
            # THE DISTINCTION THAT MATTERS: a change feature can be missing
            # because the firm has no previous year (panel length) or because an
            # input is absent (data gap). They have different remedies.
            no_prev = int((miss & ~pw).sum()) if f in lagged else 0
            item_gap = int(miss.sum()) - no_prev
            cov = w[f].notna().mean() * 100
            flag = " <50%" if cov < 50 else ""
            print(f"  {f:<22}{cov:>7.2f}%{'yes' if f in lagged else '-':>6}"
                  f"{no_prev:>14,}{item_gap:>12,}   {', '.join(deps[f]) or '-'}{flag}")
            if start == WINDOW_START:
                rows.append({"feature": f, "coverage_pct": round(cov, 2),
                             "uses_lag": f in lagged, "lost_no_prev_year": no_prev,
                             "lost_missing_item": item_gap,
                             "serious_proxy_items": ", ".join(deps[f])})
        low = [f for f in feats if w[f].notna().mean() < 0.50]
        print(f"\n  BELOW 50% COVERAGE: {len(low)}  {', '.join(sorted(low)) or '(none)'}")

        if start == WINDOW_START:
            print(f"\n  BY YEAR — non-missing %, every feature")
            by = w.groupby("fyear")[feats].apply(lambda g: g.notna().mean() * 100)
            print(by.round(1).to_string())
            print(f"\n  roavol BY YEAR — a rolling window is mechanically "
                  f"unavailable at the start of any panel, so its coverage is "
                  f"NOT evenly spread and this interacts with WINDOW_START:")
            for y, v in w.groupby("fyear")["roavol"].apply(
                    lambda s: s.notna().mean() * 100).items():
                print(f"      {int(y)}  {v:>6.2f}%")

    pd.DataFrame(rows).to_csv(REPORT_OUT, index=False)
    print(f"\n  wrote {REPORT_OUT.name}")


if __name__ == "__main__":
    main()
