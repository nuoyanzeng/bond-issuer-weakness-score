"""
Stage 6.1 — Chinese imputation. IMPUTATION ONLY.

WHAT THIS DOES: applies D-43's two-level hierarchical imputation to the Chinese
feature panel, and reports what each level achieved.

WHAT IT DOES NOT DO: no rank transform, no scoring, no percentile, no window
cut. Those are separate scripts and separate decisions; mixing them in here
would hide which step lost what. Nothing from `rank_within_fyear` onwards is
imported, so the omission is structural rather than a matter of discipline.

WHY IT CALLS THE US FUNCTIONS RATHER THAN REIMPLEMENTING THEM. Coefficient
transfer assumes a feature means the same thing in both markets, and that
extends to how its gaps were filled: a Chinese Level-1 median computed by
different code from the US one is a different operation wearing the same name.
`stage3_10`'s `level1_firm_past` and `level2_industry_cell` are called here
unchanged, exactly as `stage6_0` calls `stage1_1`'s formula helpers. Importing
`stage3_10` runs nothing — it is guarded by `if __name__ == "__main__"`.

THREE STRUCTURAL FACTS ABOUT THE CHINESE PANEL, EACH OF WHICH CHANGES A READING:

1. EVERY ROW IS DATED 12-31 of its fiscal year — all 28,399, no exceptions. So
   every donor inside a `fyear x industry` cell shares one date, and
   `STRICT_AVAIL_CELLS` CANNOT BLOCK ANYTHING. The US restriction exists solely
   to stop "a cell-mate with a later fiscal year-end inside the same fyear"
   lending a statement published after the row being filled (stage3_10:130-132),
   and China has no within-year fiscal-year-end variation for it to catch.
   `blocked_avail` will therefore be EXACTLY 0, and that is an artefact of the
   panel's shape — NOT evidence that Chinese coverage is better than US. The
   restriction is left switched on so both markets run the same code. Because
   all donors share a date, the VALUE assigned to `date_avail` cannot change any
   result, so nothing here pre-empts the open question of PRC filing deadlines
   (4 months annual, 2 interim) which differ from the SEC ones D-41 encodes.

2. LEVEL 1 CANNOT REACH CHINA'S DOMINANT GAP. `(gvkey, fyear)` is unique, and
   the 27-set's missingness sits in the change/lag features — grcapx1y 39.2%,
   DelLTI 32.4%, roavol 30.2%, Accruals 27.2%, ChNWC 24.8% — which are missing
   precisely in a firm's FIRST OBSERVED YEAR, where by definition there is no
   prior observation to look back to. That is the mechanism behind the already
   measured "Level 1 alone closes only 17.8% of the gap". The report counts the
   first-year cells so the ceiling is visible rather than mysterious.

3. INDUSTRY IS TIME-INVARIANT PER ISSUER, WHICH DEVIATES FROM THE US SIDE. The
   project table's `AcceptanceDate` is 19% blank and starts in 2017, so it
   cannot date an industry label for a panel that begins in 1994. Each issuer
   therefore gets ONE industry, applied to all its years — a 2020 project's
   label grouping a 1998 row. This is not outcome leakage (no label is
   involved), but it IS anachronistic grouping, and the US `sic2D` is
   time-varying, merged on (gvkey, datadate). Recorded, not hidden.

SCOPE: the 27 features the model actually uses, read from the stage5_0 metadata
rather than retyped (D-55 records feature lists duplicated across six files with
eight recurrences; this avoids a ninth). A CONSEQUENCE, CHOSEN KNOWINGLY: this
imposes the US K = 12 screen on the Chinese panel and forecloses a Chinese
screen later — the thing `stage6_0` deliberately avoided for the 37 it built.

COST: seconds. 28,399 rows x 27 features, three Level-2 passes (the main one,
the raw-taxonomy comparison, and the cross-check). Peak memory is a few hundred
MB. Set COMPARE_GRANULARITY = False to drop one pass.

Run:  python3 code/stage6_1_impute_china.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
CHINA_DIR = PROJECT_DIR / "data_raw" / "china"

# code/ is not on sys.path when this is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage3_10_prepare_model_input as s3          # the imputation functions
from stage6_0_construct_china_features import _unwrap   # CSMAR `="v"` stripper

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PANEL_IN = OUT_DIR / "stage6_0_china_features.parquet"
# The model whose feature set defines the scope. Read for its ordered name list.
MODEL_META = OUT_DIR / "stage5_0_model_L3_K12_base_xgb_meta.json"
# CSMAR bond project table, exported from https://data.csmar.com/ with an
# authorized account and saved in data_raw/china/. Its IssuerIndustry field is
# the only Chinese industry classification used by this stage (D-63).
PROJ_FILE = CHINA_DIR / "债券项目信息表.csv"

PANEL_OUT = OUT_DIR / "stage6_1_china_imputed.parquet"
FEAT_REPORT_OUT = OUT_DIR / "stage6_1_china_imputation_by_feature.csv"
YEAR_REPORT_OUT = OUT_DIR / "stage6_1_china_imputation_by_year.csv"

# Column the industry lands in. NOT called `sic2D`: it is a CSRC taxonomy, and
# naming it after a US SIC code would be a lie in the column header. Level 2 is
# pointed at this name by assigning s3.IMPUTE_INDUSTRY below.
IND_COL = "ind_cn"
IND_RAW = "IssuerIndustry"
# The raw-taxonomy counterpart, built only to price the collapse. Never written.
IND_COL_RAW = "ind_cn_raw"

# CSMAR's catch-all. It is not an industry — it is the absence of one — so where
# an issuer carries it ALONGSIDE a real industry, the real one wins. It stays
# only for issuers that have nothing else.
CATCH_ALL = "综合"

# Also run Level 2 under the uncollapsed 35 values, to price the collapse. Adds
# one Level-2 pass and changes nothing that is written.
COMPARE_GRANULARITY = True

# --- the taxonomy ----------------------------------------------------------
# CSMAR's `IssuerIndustry` MIXES TWO HIERARCHY LEVELS: 门类 (top-level, e.g.
# 建筑业) sit beside 大类 nested inside them (土木工程建筑业, 房屋建筑业). Grouping a
# firm labelled 建筑业 apart from one labelled 土木工程建筑业 would split one
# industry into two cells for no reason, so the 大类 are mapped to their parents.
#
# SOURCE OF THE MAPPING: it follows the 门类 level of the CSRC industry
# classification guideline (上市公司行业分类指引, 2012), so this reproduces a
# documented hierarchy rather than a reading of the label strings.
#
# PROVENANCE CAVEAT, stated because the rule is to flag a guess: no copy of that
# guideline is held in this project, so the attribution above rests on the
# supervisor's statement of it and has NOT been checked against the source text.
#
# Row counts as at the 2026-09-20 survey (D-63); 15 values, 128 rows, 0.30% of
# the table — so the collapse is about coherence, not coverage.
COLLAPSE_MAP = {
    "土木工程建筑业": "建筑业",                          # 79
    "房屋建筑业": "建筑业",                              # 4
    "建筑装饰、装修和其他建筑业": "建筑业",               # 1
    "租赁业": "租赁和商务服务业",                        # 8
    "商务服务业": "租赁和商务服务业",                    # 4
    "资本市场服务": "金融业",                            # 6
    "其他金融业": "金融业",                              # 3
    "货币金融服务": "金融业",                            # 2
    "计算机、通信和其他电子设备制造业": "制造业",         # 6
    "汽车制造业": "制造业",                              # 1
    "电力、热力生产和供应业": "电力、热力、燃气及水生产和供应业",   # 6
    "批发业": "批发和零售业",                            # 4
    "农、林、牧、渔专业及辅助性活动": "农、林、牧、渔业",  # 2
    "公共设施管理业": "水利、环境和公共设施管理业",        # 1
    "铁路运输业": "交通运输、仓储和邮政业",               # 1
}

# The 20 values that survive the collapse. Listed so an UNKNOWN value — a new
# CSMAR label, or a renamed one — stops the script instead of silently forming
# its own cell or vanishing.
PARENTS = {
    "建筑业", "综合", "金融业", "租赁和商务服务业", "采矿业",
    "电力、热力、燃气及水生产和供应业", "制造业", "交通运输、仓储和邮政业",
    "房地产业", "批发和零售业", "农、林、牧、渔业",
    "水利、环境和公共设施管理业", "信息传输、软件和信息技术服务业",
    "科学研究和技术服务业", "文化、体育和娱乐业", "住宿和餐饮业",
    "居民服务、修理和其他服务业", "教育", "卫生和社会工作", "国际组织",
}

KEY = ["gvkey", "datadate", "fyear"]


# ---------------------------------------------------------------------------
# READING
# ---------------------------------------------------------------------------
def read_project_table():
    """Read the CSMAR project table, following stage6_0.read_csmar's conventions.

    Rows 2 and 3 of every CSMAR file are the Chinese column names and the units,
    not data, so they are skipped. Everything is read as text and unwrapped,
    because every cell ships as `="value"` and a natively-typed read would be
    decided by the first chunk.

    This duplicates ~10 lines of `stage6_0.read_csmar`, which could not be
    reused directly because it resolves its path through stage6_0's own `FILES`
    dict. Duplicating the reader was preferred to editing a completed, verified
    script; it is the same class of defect D-55 records for feature lists, and
    is noted rather than hidden.
    """
    want = ["InstitutionID", IND_RAW]
    header = pd.read_csv(PROJ_FILE, nrows=0, encoding="utf-8-sig")
    cols = [c.replace('="', "").replace('"', "").strip() for c in header.columns]
    missing = [w for w in want if w not in cols]
    if missing:
        raise SystemExit(f"STOP: {PROJ_FILE.name} has no column {missing}")
    idx = sorted(cols.index(w) for w in want)
    d = pd.read_csv(PROJ_FILE, usecols=idx, skiprows=[1, 2], header=0, dtype=str,
                    encoding="utf-8-sig", low_memory=False)
    d.columns = [cols[i] for i in idx]
    for c in d.columns:
        d[c] = _unwrap(d[c])
    return d


def read_feature_names():
    """The 27, IN ORDER, from the model that will consume them.

    Read rather than retyped: D-55 records feature lists hard-coded across six
    files, every one of which went stale during the D-50 rebuild. Order matters
    because a permuted vector scores silently wrong rather than raising.
    """
    meta = json.loads(MODEL_META.read_text())
    feats = list(meta["features_in_order"])
    if len(feats) != meta["n_features"]:
        raise SystemExit(f"STOP: {MODEL_META.name} says n_features="
                         f"{meta['n_features']} but lists {len(feats)} names.")
    return feats


# ---------------------------------------------------------------------------
# INDUSTRY
# ---------------------------------------------------------------------------
def collapse(series):
    """Map 大类 to their 门类 parents, stopping on any value we do not know."""
    unknown = sorted(set(series.dropna().unique()) - set(COLLAPSE_MAP) - PARENTS)
    if unknown:
        raise SystemExit(
            f"STOP: {len(unknown)} IssuerIndustry value(s) are neither a known "
            f"top-level category nor in COLLAPSE_MAP: {unknown}. CSMAR has "
            "added or renamed a label; decide where it belongs before running.")
    return series.replace(COLLAPSE_MAP)


def resolve_one_per_issuer(proj, col):
    """One industry per issuer: MODAL, with 综合 deprioritised.

    An issuer's project rows often disagree — 35.9% of the issuers we match
    carry more than one label — from two causes: genuine hierarchy mixing
    (which `collapse` removes) and apparent data-entry error (采矿业 attached to
    factoring companies, 交通运输 to securities brokers), which nothing can
    remove. Modal is the rule that lets the weight of an issuer's own filings
    decide.

    综合 is dropped from the contest unless it is all the issuer has: it is a
    catch-all, so an issuer with two 综合 rows and one 房地产业 row is better
    described by 房地产业 than by the absence of a description.

    Ties are broken by table-wide frequency, then by name. FULLY DETERMINISTIC —
    no random state is involved anywhere in this function.
    """
    freq = proj[col].value_counts()
    counts = (proj.groupby(["InstitutionID", col], sort=False)
              .size().rename("n").reset_index())
    # drop the catch-all unless it is the issuer's only value
    n_vals = counts.groupby("InstitutionID")[col].transform("size")
    counts = counts[~((counts[col] == CATCH_ALL) & (n_vals > 1))]
    # winner first: most rows, then commonest overall, then alphabetical
    counts["_freq"] = counts[col].map(freq)
    counts = counts.sort_values(["InstitutionID", "n", "_freq", col],
                                ascending=[True, False, False, True])
    return (counts.drop_duplicates("InstitutionID", keep="first")
            .set_index("InstitutionID")[col])


def build_industry(panel):
    """The whole industry pipeline, reporting every step's cost."""
    print("\n" + "=" * 84)
    print("STEP 1 — industry, from the CSMAR project table (D-63)")
    print("=" * 84)

    proj = read_project_table()
    print(f"  {PROJ_FILE.name}: {len(proj):,} rows")

    # An issuer id we cannot key on, or a label that is blank, is unusable.
    no_id = proj["InstitutionID"].isin(["", "nan"]) | proj["InstitutionID"].isna()
    no_ind = proj[IND_RAW].isin(["", "nan"]) | proj[IND_RAW].isna()
    print(f"      dropped, blank InstitutionID : {int(no_id.sum()):>7,} "
          f"({no_id.mean()*100:.2f}%)")
    print(f"      dropped, blank {IND_RAW}: {int(no_ind.sum()):>7,}")
    proj = proj[~no_id & ~no_ind].copy()
    print(f"      usable                       : {len(proj):>7,} rows, "
          f"{proj['InstitutionID'].nunique():,} issuers")

    # --- the collapse -----------------------------------------------------
    n_before = proj[IND_RAW].nunique()
    proj[IND_COL] = collapse(proj[IND_RAW])
    n_after = proj[IND_COL].nunique()
    moved = int((proj[IND_COL] != proj[IND_RAW]).sum())
    print(f"\n  COLLAPSE 门类: {n_before} values -> {n_after}; "
          f"{moved:,} rows moved ({moved/len(proj)*100:.2f}%)")
    for src, dst in sorted(COLLAPSE_MAP.items(), key=lambda kv: kv[1]):
        n = int((proj[IND_RAW] == src).sum())
        if n:
            print(f"      {src:<34} -> {dst:<28}{n:>6,}")

    # --- what the collapse does to the disagreement -----------------------
    print(f"\n  DISTINCT INDUSTRIES PER ISSUER — the collapse's real effect")
    for lab, col in [("raw 35 ", IND_RAW), ("collapsed", IND_COL)]:
        per = proj.groupby("InstitutionID")[col].nunique()
        multi = int((per > 1).sum())
        dist = per.value_counts().sort_index()
        shown = ", ".join(f"{k}:{v:,}" for k, v in dist.items())
        print(f"      {lab}: {multi:,} of {len(per):,} issuers disagree "
              f"({multi/len(per)*100:.1f}%)   [{shown}]")

    # --- one industry per issuer ------------------------------------------
    ind = resolve_one_per_issuer(proj, IND_COL)
    # How often the catch-all rule changed the answer, so its effect is a
    # measured number rather than an assumption.
    plain = (proj.groupby(["InstitutionID", IND_COL], sort=False).size()
             .rename("n").reset_index()
             .sort_values(["InstitutionID", "n"], ascending=[True, False])
             .drop_duplicates("InstitutionID", keep="first")
             .set_index("InstitutionID")[IND_COL])
    changed = int((ind != plain.reindex(ind.index)).sum())
    print(f"\n  ONE PER ISSUER: modal, 综合 deprioritised")
    print(f"      issuers resolved                     : {len(ind):>7,}")
    print(f"      where 综合 rule beat plain modal      : {changed:>7,}")
    print(f"      still resolved TO 综合 (nothing else) : "
          f"{int((ind == CATCH_ALL).sum()):>7,}")

    out = {IND_COL: ind}
    if COMPARE_GRANULARITY:
        out[IND_COL_RAW] = resolve_one_per_issuer(proj, IND_RAW)

    # --- join to the panel -------------------------------------------------
    # gvkey IS the CSMAR Orgid (stage6_0:469) and InstitutionID is the same id;
    # both are all-digit strings with no leading zeros, so a string join is
    # lossless and identical to an int-cast join.
    res = panel[["gvkey"]].copy()
    for name, s in out.items():
        res[name] = panel["gvkey"].map(s)
    cov_i = panel.loc[res[IND_COL].notna(), "gvkey"].nunique()
    cov_r = int(res[IND_COL].notna().sum())
    print(f"\n  JOIN gvkey == InstitutionID")
    print(f"      issuers with an industry : {cov_i:>7,} of "
          f"{panel['gvkey'].nunique():,} ({cov_i/panel['gvkey'].nunique()*100:.2f}%)")
    print(f"      panel ROWS covered       : {cov_r:>7,} of {len(panel):,} "
          f"({cov_r/len(panel)*100:.2f}%)")
    print(f"      rows with NO industry    : {len(panel)-cov_r:>7,} "
          f"({(len(panel)-cov_r)/len(panel)*100:.2f}%) — these can NEVER take a "
          "Level-2 fill, and there is no Level 3")

    print(f"\n  THE {n_after} INDUSTRIES, by issuers on the panel")
    vc = res.loc[res[IND_COL].notna()].drop_duplicates("gvkey")[IND_COL].value_counts()
    for k, v in vc.items():
        print(f"      {k:<34}{v:>7,}")

    return res.drop(columns="gvkey")


# ---------------------------------------------------------------------------
# THE FILL
# ---------------------------------------------------------------------------
def check_constants():
    """Assert the US settings are what we believe, before anything is filled.

    stage3_10's constants are module globals shared with the US pipeline. A
    change made there for a US reason would otherwise change China silently.
    """
    expect = {"LOOKBACK_OBS": 2, "MIN_CELL_FIRMS": 5, "STRICT_AVAIL_CELLS": True,
              "IMPUTE_METHOD": "hierarchical", "INDICATOR_NA_CODE": -1}
    bad = {k: getattr(s3, k) for k, v in expect.items() if getattr(s3, k) != v}
    if bad:
        raise SystemExit(f"STOP: stage3_10 constants have moved: {bad}; "
                         f"expected {expect}. D-43 describes the settings this "
                         "script assumes.")
    if not {"OENEG", "INTWO"} <= set(s3.IMPUTE_EXEMPT):
        raise SystemExit(f"STOP: OENEG/INTWO are not in stage3_10's "
                         f"IMPUTE_EXEMPT ({s3.IMPUTE_EXEMPT}); the indicators "
                         "would be imputed, which D-44 forbids.")
    print(f"  stage3_10 constants verified: " +
          ", ".join(f"{k}={v}" for k, v in expect.items()))
    print(f"  IMPUTE_EXEMPT (never filled): {s3.IMPUTE_EXEMPT}")


def fill(df, feats):
    """Level 1 then Level 2, composed exactly as stage3_10.impute composes them.

    The two levels are called separately rather than through `impute` because
    `impute` returns only TOTALS, and the report needs the component frames to
    break the result down per feature and to separate the three reasons a
    Level-2 fill did not happen. Step 7 of the plan then checks this
    composition against `impute` cell-for-cell, so the reuse is proved rather
    than asserted.
    """
    X = df[feats].astype("float64")
    miss = X.isna()

    # Exempt features are never filled: their missingness is a fact about the
    # firm, not a gap to be estimated (D-44).
    ok = pd.DataFrame(False, index=X.index, columns=feats)
    ok[[f for f in feats if f not in s3.IMPUTE_EXEMPT]] = True

    l1 = s3.level1_firm_past(df, feats)
    f1 = miss & l1.notna() & ok
    after1 = X.where(~f1, l1)

    cell, plain, use, thin = s3.level2_industry_cell(df, feats)
    still = after1.isna() & ok
    f2 = still & cell.notna()
    filled = after1.where(~f2, cell)

    return filled, dict(miss=miss, ok=ok, f1=f1, f2=f2, still=still,
                        plain=plain, use=use, thin=thin)


def classify_blocked(df, parts):
    """Why each unfilled cell stayed unfilled. Three reasons, mutually exclusive.

    The US code reports two (STRICT_AVAIL_CELLS and MIN_CELL_FIRMS) because its
    industry code is 100% covered and a third cannot arise. China's is 77.5%
    covered, so "the row has no industry at all" is a separate and much larger
    category, and collapsing it into the thin-cell count would misattribute it.

    A residual is computed and asserted to zero, so the three are shown to be
    exhaustive rather than assumed to be.
    """
    unfilled = parts["still"] & ~parts["f2"]
    # A row-level fact broadcast to every feature column, so it can be combined
    # with the cell-level masks.
    has_ind = pd.DataFrame(
        np.repeat(df[IND_COL].notna().to_numpy()[:, None],
                  parts["still"].shape[1], axis=1),
        index=parts["still"].index, columns=parts["still"].columns)

    blocked = {}
    blocked["no_industry"] = unfilled & ~has_ind
    blocked["thin"] = unfilled & has_ind & parts["thin"].fillna(False)
    blocked["avail"] = (unfilled & has_ind & ~parts["thin"].fillna(False)
                        & parts["use"].isna() & parts["plain"].notna())
    blocked["residual"] = (unfilled & ~blocked["no_industry"]
                           & ~blocked["thin"] & ~blocked["avail"])
    return unfilled, blocked


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------
def report_cells(df, feats, parts, unfilled, blocked, first_year):
    """Per feature and in total: what was filled, what was not, and why."""
    print("\n" + "=" * 84)
    print("STEP 3 — cells")
    print("=" * 84)

    rows = []
    for f in feats:
        rows.append({
            "feature": f,
            "exempt": f in s3.IMPUTE_EXEMPT,
            "missing_before": int(parts["miss"][f].sum()),
            "level1": int(parts["f1"][f].sum()),
            "level2": int(parts["f2"][f].sum()),
            "left_na": int(unfilled[f].sum()) + int((parts["miss"][f] & ~parts["ok"][f]).sum()),
            "blocked_no_industry": int(blocked["no_industry"][f].sum()),
            "blocked_thin_cell": int(blocked["thin"][f].sum()),
            "blocked_avail": int(blocked["avail"][f].sum()),
            "residual": int(blocked["residual"][f].sum()),
            "missing_in_firms_first_year": int((parts["miss"][f] & first_year).sum()),
        })
    rep = pd.DataFrame(rows)

    print(f"  {'feature':<24}{'miss':>8}{'L1':>8}{'L2':>8}{'left':>8}"
          f"{'noInd':>8}{'thin':>8}{'1stYr':>8}")
    for r in rep.itertuples():
        tag = "  (exempt)" if r.exempt else ""
        print(f"  {r.feature:<24}{r.missing_before:>8,}{r.level1:>8,}"
              f"{r.level2:>8,}{r.left_na:>8,}{r.blocked_no_industry:>8,}"
              f"{r.blocked_thin_cell:>8,}{r.missing_in_firms_first_year:>8,}{tag}")

    t = rep[["missing_before", "level1", "level2", "left_na",
             "blocked_no_industry", "blocked_thin_cell", "blocked_avail",
             "residual", "missing_in_firms_first_year"]].sum()
    n_cells = len(df) * len(feats)
    print(f"\n  TOTAL missing cells {t.missing_before:,} of {n_cells:,} "
          f"({t.missing_before/n_cells*100:.2f}%)")
    print(f"      Level 1, firm's own past (LOOKBACK_OBS={s3.LOOKBACK_OBS}) : "
          f"{t.level1:>9,}  ({t.level1/t.missing_before*100:5.2f}% of missing)")
    print(f"      Level 2, {IND_COL} x fyear cell                    : "
          f"{t.level2:>9,}  ({t.level2/t.missing_before*100:5.2f}%)")
    print(f"      left NA (NO Level 3, by decision)               : "
          f"{t.left_na:>9,}  ({t.left_na/t.missing_before*100:5.2f}%)")
    print(f"\n  WHY THE UNFILLED CELLS WERE NOT FILLED — three exclusive reasons")
    print(f"      row has NO INDUSTRY (Level 2 unreachable)       : "
          f"{t.blocked_no_industry:>9,}")
    print(f"      cell under MIN_CELL_FIRMS={s3.MIN_CELL_FIRMS} (thin)             : "
          f"{t.blocked_thin_cell:>9,}")
    print(f"      blocked by STRICT_AVAIL_CELLS                   : "
          f"{t.blocked_avail:>9,}  <- 0 BY CONSTRUCTION, see the docstring; "
          "NOT a coverage finding")
    print(f"      residual (must be 0)                            : "
          f"{t.residual:>9,}")
    print(f"\n  LEVEL 1's STRUCTURAL CEILING: {t.missing_in_firms_first_year:,} "
          f"missing cells ({t.missing_in_firms_first_year/t.missing_before*100:.1f}%) "
          f"sit in a firm's FIRST observed year,\n      where there is no prior "
          "observation for Level 1 to use. Level 2 is the only route for those.")

    # The three reasons must account for every unfilled cell, and the
    # availability restriction must be inert. Asserted, not hoped for.
    if t.residual != 0:
        raise SystemExit(f"STOP: {t.residual:,} unfilled cells fall in none of "
                         "the three categories; the classification is not "
                         "exhaustive and the report would understate a cause.")
    if t.blocked_avail != 0:
        raise SystemExit(f"STOP: blocked_avail is {t.blocked_avail:,}, not 0. "
                         "Every panel row is dated 12-31, so the availability "
                         "restriction cannot bind — this means an assumption in "
                         "the docstring is wrong. Investigate before trusting "
                         "anything else in this run.")
    return rep


def report_rows(df, feats, filled, first_year):
    """Complete-vector share before and after, overall, by year, and by era."""
    print("\n" + "=" * 84)
    print("STEP 4 — rows")
    print("=" * 84)
    before = df[feats].notna().all(axis=1)
    after = filled.notna().all(axis=1)
    print(f"  complete {len(feats)}-feature vector")
    print(f"      before : {int(before.sum()):>8,} of {len(df):,} "
          f"({before.mean()*100:5.2f}%)   issuers "
          f"{df.loc[before, 'gvkey'].nunique():>6,}")
    print(f"      after  : {int(after.sum()):>8,} of {len(df):,} "
          f"({after.mean()*100:5.2f}%)   issuers "
          f"{df.loc[after, 'gvkey'].nunique():>6,}")

    y = pd.DataFrame({
        "fyear": df["fyear"], "before": before, "after": after,
        "ind": df[IND_COL].notna(), "gvkey": df["gvkey"]})
    g = y.groupby("fyear")
    tab = pd.DataFrame({
        "rows": g.size(),
        "issuers": g["gvkey"].nunique(),
        "industry_pct": g["ind"].mean() * 100,
        "complete_before": g["before"].sum(),
        "complete_after": g["after"].sum()})
    tab["before_pct"] = tab.complete_before / tab.rows * 100
    tab["after_pct"] = tab.complete_after / tab.rows * 100

    print(f"\n  BY YEAR")
    print(f"      {'fyear':>6}{'rows':>8}{'ind %':>8}{'compl before':>14}"
          f"{'compl after':>13}")
    for fy, r in tab.iterrows():
        print(f"      {int(fy):>6}{int(r.rows):>8,}{r.industry_pct:>7.1f}%"
              f"{int(r.complete_before):>9,} ({r.before_pct:4.1f}%)"
              f"{int(r.complete_after):>8,} ({r.after_pct:4.1f}%)")

    # PRE-2015 SEPARATELY. Those years hold 3-1,156 issuers, so almost every
    # industry x fyear cell is below MIN_CELL_FIRMS and Level 2 cannot work.
    print(f"\n  BY ERA — the thin-cell problem is a YEAR problem, not an "
          f"industry one")
    for lab, m in [("1994-2014", df["fyear"] <= 2014),
                   ("2015-2024", df["fyear"] >= 2015)]:
        if not m.any():
            continue
        print(f"      {lab}: {int(m.sum()):>7,} rows  "
              f"{df.loc[m, 'gvkey'].nunique():>5,} issuers   "
              f"industry {df.loc[m, IND_COL].notna().mean()*100:5.1f}%   "
              f"complete {before[m].mean()*100:5.2f}% -> {after[m].mean()*100:5.2f}%")
    return tab.reset_index()


def report_granularity(df, feats, parts, ind_raw_col):
    """Price the collapse: the same Level 2 run under the uncollapsed 35.

    Read-only. Nothing written depends on it; it exists so the choice between
    20 and 35 categories is a measured one and stays on the record.
    """
    print("\n" + "=" * 84)
    print("STEP 5 — what the collapse cost or bought (read-only comparison)")
    print("=" * 84)
    s3.IMPUTE_INDUSTRY = ind_raw_col
    cell_r, _, _, thin_r = s3.level2_industry_cell(df, feats)
    s3.IMPUTE_INDUSTRY = IND_COL                    # restore immediately

    f2_r = parts["still"] & cell_r.notna()
    for lab, col, f2, thin in [("collapsed (20)", IND_COL, parts["f2"], parts["thin"]),
                               ("raw       (35)", ind_raw_col, f2_r, thin_r)]:
        cells = df.dropna(subset=[col]).groupby([col, "fyear"]).size()
        small = int((cells < s3.MIN_CELL_FIRMS).sum())
        print(f"  {lab}: {len(cells):,} industry x fyear cells, "
              f"{small:,} below {s3.MIN_CELL_FIRMS} firms "
              f"({small/len(cells)*100:.1f}%)   |   "
              f"Level-2 fills {int(f2.values.sum()):,}   "
              f"thin-blocked {int((parts['still'] & thin.fillna(False)).values.sum()):,}")
    d = int(parts["f2"].values.sum()) - int(f2_r.values.sum())
    print(f"\n  The collapse changes Level-2 fills by {d:+,} cells. The 15 "
          f"sub-level values hold 0.30% of the project table,\n  so the "
          "collapse is a coherence decision, not a coverage one; the thin "
          "cells are driven by the thin YEARS.")


# ---------------------------------------------------------------------------
def run_config(feats):
    """Every setting that shaped this file, stamped into it."""
    return {"input": PANEL_IN.name, "feature_source": MODEL_META.name,
            "n_features": len(feats), "features_in_order": feats,
            "impute_method": s3.IMPUTE_METHOD, "lookback_obs": s3.LOOKBACK_OBS,
            "impute_industry": IND_COL, "industry_source": PROJ_FILE.name,
            "industry_rule": "modal per issuer, 综合 deprioritised, "
                             "time-invariant, collapsed to 门类",
            "industry_collapsed_to": len(PARENTS),
            "min_cell_firms": s3.MIN_CELL_FIRMS,
            "strict_avail_cells": s3.STRICT_AVAIL_CELLS,
            "strict_avail_note": "inert: every row is dated 12-31, so no cell "
                                 "mate can post-date another",
            "level3": False, "window": "none — all fyears 1994-2024",
            "impute_exempt": list(s3.IMPUTE_EXEMPT),
            "indicator_na_code": s3.INDICATOR_NA_CODE,
            "rank_transform": False, "scored": False}


def main():
    for p in (PANEL_OUT, FEAT_REPORT_OUT, YEAR_REPORT_OUT):
        if p.exists():
            raise SystemExit(f"STOP: {p.name} exists; this script never "
                             "overwrites. Move or rename it first.")
    for p in (PANEL_IN, MODEL_META, PROJ_FILE):
        if not p.exists():
            raise SystemExit(f"STOP: required input {p} not found.")

    print("=" * 84)
    print("STEP 0 — inputs and constants")
    print("=" * 84)
    check_constants()

    panel = pd.read_parquet(PANEL_IN)
    feats = read_feature_names()
    missing = [f for f in feats if f not in panel.columns]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} model features absent from "
                         f"{PANEL_IN.name}: {missing}")
    print(f"  {PANEL_IN.name}: {len(panel):,} rows, "
          f"{panel['gvkey'].nunique():,} issuers, fyear "
          f"{int(panel.fyear.min())}-{int(panel.fyear.max())}, "
          f"{panel.shape[1]} columns")
    print(f"  {MODEL_META.name}: {len(feats)} features, taken IN ORDER")
    print(f"  panel carries {len([c for c in panel.columns if c not in KEY])} "
          f"features; {len(feats)} are used here and "
          f"{len([c for c in panel.columns if c not in KEY and c not in feats])}"
          " are left aside (the model does not use them)")

    df = panel[KEY + feats].copy()

    # Industry, joined on gvkey == Orgid == InstitutionID.
    df = pd.concat([df, build_industry(df)], axis=1)

    # level2_industry_cell needs a date_avail column. Every row is 12-31 of its
    # fyear, so all donors in a cell share one date and the restriction is inert
    # whatever value is used — see the docstring. This does NOT decide the open
    # PRC filing-deadline question; no choice made here could affect the result.
    df["date_avail"] = df["datadate"]
    # Point Level 2 at the Chinese industry column. One assignment, made here so
    # it is visible rather than buried.
    s3.IMPUTE_INDUSTRY = IND_COL

    print("\n" + "=" * 84)
    print("STEP 2 — imputation (Level 1 then Level 2; NO Level 3)")
    print("=" * 84)
    print(f"  IMPUTE_METHOD = {s3.IMPUTE_METHOD}   industry = {IND_COL}   "
          f"LOOKBACK_OBS = {s3.LOOKBACK_OBS}   MIN_CELL_FIRMS = "
          f"{s3.MIN_CELL_FIRMS}")
    filled, parts = fill(df, feats)

    # CROSS-CHECK: stage3_10.impute composes the same two levels. If our
    # composition differs from the US one by a single cell, the features are not
    # the same quantity in both markets and the transfer premise fails.
    check, _ = s3.impute(df, feats, "china (cross-check)")
    same = ((filled.isna() == check.isna()) &
            (filled.fillna(0) == check.fillna(0))).all().all()
    if not same:
        n = int(((filled.isna() != check.isna()) |
                 (filled.fillna(0) != check.fillna(0))).values.sum())
        raise SystemExit(f"STOP: our composition disagrees with "
                         f"stage3_10.impute on {n:,} cells. The reuse is not "
                         "faithful; do not trust this output.")
    print(f"\n  CROSS-CHECK: our composition and stage3_10.impute agree "
          f"cell-for-cell on all {filled.size:,} cells.")

    first_year = df["fyear"] == df.groupby("gvkey")["fyear"].transform("min")
    unfilled, blocked = classify_blocked(df, parts)
    feat_rep = report_cells(df, feats, parts, unfilled, blocked, first_year)
    year_rep = report_rows(df, feats, filled, first_year)
    if COMPARE_GRANULARITY:
        report_granularity(df, feats, parts, IND_COL_RAW)

    # --- integrity checks, before anything is written ---------------------
    print("\n" + "=" * 84)
    print("STEP 6 — integrity")
    print("=" * 84)
    exempt = [f for f in feats if f in s3.IMPUTE_EXEMPT]
    n_ex = int(parts["f1"][exempt].values.sum() + parts["f2"][exempt].values.sum())
    if n_ex:
        raise SystemExit(f"STOP: {n_ex:,} fills landed on exempt columns "
                         f"{exempt}; D-44 forbids imputing an indicator.")
    bad = int(parts["f2"].to_numpy()[df[IND_COL].isna().to_numpy()].sum())
    if bad:
        raise SystemExit(f"STOP: {bad:,} Level-2 fills landed on rows with no "
                         "industry, which is impossible; the join or the cell "
                         "lookup is wrong.")
    # Cast both sides to plain float64 first: OENEG/INTWO arrive as pandas
    # nullable Float64, and comparing that against float64 yields pd.NA rather
    # than a bool, which would make this check silently vacuous.
    orig = df[feats].astype("float64")
    changed = int((orig.notna() & (filled != orig)).to_numpy().sum())
    if changed:
        raise SystemExit(f"STOP: {changed:,} values that were present before "
                         "imputation changed. Imputation must only ADD.")
    print(f"  no fill on the {len(exempt)} exempt columns ({', '.join(exempt)})")
    print(f"  no Level-2 fill on a row without an industry")
    print(f"  every pre-existing value unchanged — imputation only added")

    # --- indicator NA -> -1, AFTER the report -----------------------------
    # Matches stage3_10:622-634. Done last so "left NA" above counts the
    # indicators as NA exactly as the US report does.
    n_enc = int(filled[exempt].isna().values.sum())
    filled[exempt] = filled[exempt].fillna(s3.INDICATOR_NA_CODE)
    print(f"  indicator NA encoded as {s3.INDICATOR_NA_CODE}: {n_enc:,} cells "
          f"across {len(exempt)} features ({', '.join(exempt)})")
    for f in exempt:
        vals = set(pd.unique(filled[f].dropna()))
        if not vals <= {-1.0, 0.0, 1.0}:
            raise SystemExit(f"STOP: {f} holds values outside "
                             f"{{-1, 0, 1}} after encoding: {sorted(vals)[:8]}")

    out = pd.concat([df[KEY + [IND_COL]].reset_index(drop=True),
                     filled.reset_index(drop=True)], axis=1)
    if len(out) != len(panel) or out.duplicated(["gvkey", "fyear"]).any():
        raise SystemExit("STOP: the output lost rows or gained a duplicate "
                         "(gvkey, fyear); the key must survive imputation.")

    print("\n" + "=" * 84)
    print("WRITING")
    print("=" * 84)
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}),
         b"stage6_1_config": json.dumps(run_config(feats),
                                        ensure_ascii=False).encode()})
    pq.write_table(table, PANEL_OUT)
    print(f"  {PANEL_OUT.name:<44} {len(out):>8,} rows x {out.shape[1]:>3} cols "
          f"{PANEL_OUT.stat().st_size/1e6:6.1f} MB")
    feat_rep.to_csv(FEAT_REPORT_OUT, index=False)
    year_rep.to_csv(YEAR_REPORT_OUT, index=False)
    print(f"  {FEAT_REPORT_OUT.name}")
    print(f"  {YEAR_REPORT_OUT.name}")

    na_rows = int(out[feats].isna().any(axis=1).sum())
    print(f"\n  {na_rows:,} of {len(out):,} rows still carry an NA "
          f"({na_rows/len(out)*100:.1f}%) — no Level 3, by decision.")
    print("\nNOTHING RANKED, NOTHING SCORED. Those are stage6_2 and later.")


if __name__ == "__main__":
    main()
