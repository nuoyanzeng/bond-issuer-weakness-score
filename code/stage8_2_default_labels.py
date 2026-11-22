"""
Stage 8.2 — Attach default and downgrade labels to the score panels.

THREE NEW FILES, NOTHING EXISTING MODIFIED. `data_raw/` is read only. The
script refuses to start if any of its outputs already exists.

    output/stage8_2_us_distress_firmyear.parquet
    output/stage8_2_us_rating_firmyear.parquet
    output/stage8_2_china_labels.parquet

WHAT IT DOES, IN FOUR PARTS

A. US — collapse the issuer-level panels to one row per gvkey x fyear WITHOUT
   losing positives. The models were cross-fitted on issuer-level rows, so
   several rows of one firm-year carry the same score; `drop_duplicates` keeps
   the FIRST of them, which need not be the labelled one, and that silently
   destroyed 207 of 1,164 distress positives in earlier work. Here the score is
   ASSERTED constant inside the cell and taken once, and the label is the MAX
   over the cell: a firm-year is 1 if ANY of its issuer rows is 1.

B. China — three manually exported bond event tables, mapped to Orgid
   by the method measured in the read-only pass: (Liscd, Sctcd) first, then
   exact issuer name. The method is carried as a column, so a later reader can
   tell how each issuer was resolved.

C. China — two default labels copying the US rule EXACTLY as stage3_7 writes
   it: the availability guard, the absorbing state, and ONE positive per issuer
   on its last statement knowable before the event. There is no fixed horizon,
   because there is none on the US side either.

D. China — 12-month forward downgrade labels. Supplementary, and kept in
   separate columns for that reason.

THE US RULE THIS COPIES, with its source lines so the copy can be checked:
    stage3_7:50-52   keep date_avail >= the first offering date
    stage3_7:72-74   DROP rows with date_avail >= the event date
    stage3_7:79-84   assert nothing survives at or after the event
    stage3_7:88-91   y = 1 on the single latest surviving row per issuer
    stage3_5:21,54   date_avail = datadate + 3 calendar months

WHAT IS NOT DECIDED HERE. Two default labels are written, not one: `y_default`
from defaults and extensions alone, `y_credit` from those plus rating
downgrades. Four downgrade widths are written in part D. Both sets exist so the
choice can be made on the numbers rather than asserted now.

FIVE THINGS THE READER SHOULD KNOW ABOUT THE INPUTS

  1. The notch scale is stage7_1's, in which AAA+, AAA and AAA- ALL map to 1.
     A move inside the AAA band is therefore not a notch downgrade here. Said
     plainly because it is a real consequence of reusing that scale.
  2. The entity downgrade file holds a large minority of moves that are not
     notch downgrades. They are split into `outlook` (the rating string
     identical) and `within-band` (the string changed, the notch did not,
     which this scale permits only inside AAA), and NEITHER is merged into the
     notch count. `within-band` enters no label at all.
  3. A rating off the notch scale makes a row `unclassified`, never
     `outlook` — treating it as unchanged would inflate the outlook count.
  4. `Ipodt` (发行起始日期) is the offering-date column, the analogue of the
     `offering_date` D-08 names for the US. It is non-null on all 502,669
     bond-master rows, but its maximum is 2100-01-01: a date from
     IPO_SENTINEL_YEAR on is a placeholder and is blanked BEFORE each issuer's
     earliest offering date is taken, with the count printed.
  5. A guard-excluded row gets y = <NA>, not 0. On the US side stage3_7 DROPS
     those rows rather than calling them zero, and a 0 here would invent a
     non-event. The companion `in_default` / `in_credit` flags say which rows
     are in the labelled sample.

Run:  python3 code/stage8_2_default_labels.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output"
CH_DIR = PROJECT_DIR / "data_raw" / "china"

# The CSMAR reader is IMPORTED, not copied, so it cannot drift from the one
# stage8_0 and check_china_score_validity use.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_china_score_validity import csmar

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
US_DISTRESS = OUT_DIR / "stage5_1_score_panel_L3_K12.parquet"
US_RATING = OUT_DIR / "stage5_1_score_panel_rating_L3_K12.parquet"
CN_PANEL = OUT_DIR / "stage6_4_china_score_panel.parquet"

# Manually export these three tables from Wind Financial Terminal through an
# authorized account: Bond > Credit Bond Research > Bond Defaults and Extensions
# for F_DEFAULT, and Bond > Credit Bond Research > Rating Warning for the two
# downgrade tables. Use the date ranges in the filenames and save each CSV in
# data_raw/china/ under the exact name below. The scripts do not create them.
# Check the expected columns before using exports from a different Wind release.
# All three files have an English header and a Chinese label row to skip.
F_DEFAULT = "债券违约及展期大全 2010-2025.csv"  # defaults and extensions
F_ENT_DG = "主体评级调低的企业2005-2025.csv"    # issuer rating downgrades
F_BND_DG = "债项评级调低债券2000-2025.csv"      # bond rating downgrades

# Export these two tables from CSMAR (https://data.csmar.com/) through an
# authorized account. Preserve the three-row header and save them in
# data_raw/china/ under these exact names.
F_BONDMASTER_NAME = "债券基本情况表.csv"  # bond IDs and issuer links
F_ISSUER_NAME = "发行人主体资料.csv"      # issuer identifiers and names

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
US_D_OUT = OUT_DIR / "stage8_2_us_distress_firmyear.parquet"
US_R_OUT = OUT_DIR / "stage8_2_us_rating_firmyear.parquet"
CN_OUT = OUT_DIR / "stage8_2_china_labels.parquet"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODELS = ["xgb", "rf", "enet"]
SCORES = [f"{o}_{m}" for o in ("distress", "rating") for m in MODELS]

AVAIL_LAG_MONTHS = 3          # identical to stage3_5's AVAIL_LAG_MONTHS (D-41)
DG_WINDOW_MONTHS = 12         # part D's forward window

# An `Ipodt` from this year on is a placeholder, not an offering date. The
# bond master's maximum is 2100-01-01.
IPO_SENTINEL_YEAR = 2099

# Symbol suffix to CSMAR market code, verified on the bond master:
# 125099.SH -> Sctcd 1, 118123.SZ -> 2, 041458025.IB -> 3.
SUF2SCT = {"SH": "1", "SZ": "2", "IB": "3"}

# Securitisations: excluded on the default side, and on the bond-downgrade side
# under that file's own taxonomy.
ABS_DEFAULT = {"证监会主管ABS", "交易商协会ABN"}
ABS_DOWNGRADE = {"资产支持证券"}

# Entity-downgrade rows kept: those rating the issuer. Dropped are 对担保人 and
# 对差额支付承诺人, which rate somebody else.
KEEP_OBJECT = {"发行人委托", "公司", "对发行人"}

# stage7_1's notch scale. AAA+ / AAA / AAA- collapse to 1; see the docstring.
NOTCH = {"AAA+": 1, "AAA": 1, "AAA-": 1,
         "AA+": 2, "AA": 3, "AA-": 4,
         "A+": 5, "A": 6, "A-": 7,
         "BBB+": 8, "BBB": 9, "BBB-": 10,
         "BB+": 11, "BB": 12, "BB-": 13,
         "B+": 14, "B": 15, "B-": 16,
         "CCC+": 17, "CCC": 18, "CCC-": 19, "CC": 20, "C": 21}

# ---- the loud checks' recorded constants, from the 2026-11-04 read-only pass
EXPECT_US_DISTRESS_CELLS = 1054        # gvkey x fyear cells holding a positive
EXPECT_US_DISTRESS_OLD_WINDOW = 468    # what first-row de-duplication gave
EXPECT_CN_ISSUERS = 322                # matched issuer names
EXPECT_CN_ORGIDS = 320                 # distinct Orgids behind them
EXPECT_ENT_DG_DROPPED = 41             # guarantor / promisor rows

BAR = "=" * 100
CHECKS = []                            # (ok, label), the gate before writing


def hdr(t):
    print("\n" + BAR)
    print(t)
    print(BAR)


def check(ok, label, detail=""):
    """Print one check line, remember its result, and return it."""
    ok = bool(ok)
    CHECKS.append((ok, label))
    print(f"      {'PASSED' if ok else 'FAILED'}   {label}"
          + (f"   {detail}" if detail else ""))
    return ok


def read_event_csv(name):
    """An event file: English header on row 0, Chinese labels on row 1.

    UTF-8 with a BOM and no units row — unlike the CSMAR panel extracts, which
    carry three header rows and are read by the imported `csmar`.
    """
    return pd.read_csv(CH_DIR / name, encoding="utf-8-sig", dtype=str,
                       header=0, skiprows=[1], low_memory=False)


# ===========================================================================
# PART A — US, collapse to firm-year without losing positives
# ===========================================================================
def us_firmyear(panel_path, raw_prefix, out_path, expect_cells, expect_old):
    """Collapse one issuer-level US panel to gvkey x fyear.

    `raw_prefix` names the stage4 cross-fitting output, which carries the RAW
    probability beside the calibrated one the panel holds.
    """
    hdr(f"PART A — {panel_path.name}")
    panel = pd.read_parquet(panel_path)

    # ---- 1. the cross-fitting basis ---------------------------------------
    # Every stage4 score file must hold exactly the panel's rows. That is what
    # "cross-fitted on the un-de-duplicated issuer-level rows" means, and it is
    # checked rather than assumed.
    print("\n  1. CROSS-FITTING BASIS")
    print(f"      {panel_path.name}: {len(panel):,} rows, "
          f"{panel.gvkey.nunique():,} gvkeys, "
          f"{panel.issuer_id.nunique():,} issuer_ids")
    key = ["issuer_id", "gvkey", "fyear"]
    for m in MODELS:
        p = OUT_DIR / f"{raw_prefix}_{m}.parquet"
        s = pd.read_parquet(p)
        check(len(s) == len(panel), f"{p.name} row count",
              f"{len(s):,} against the panel's {len(panel):,}")
        j = panel.merge(s, on=key, how="inner", suffixes=("", "_s"))
        check(len(j) == len(panel), f"{p.name} joins one-to-one",
              f"{len(j):,} joined rows")
        # The calibrated column must BE the panel's score. This is the test
        # that identifies the right file among the several stage4 variants.
        d = float(np.abs(j[f"score_{m}"] - j["score_calibrated"]).max())
        check(d == 0.0,
              f"{p.name} score_calibrated equals the panel's score_{m}",
              f"max |diff| {d:.3e}")
        check(bool((j["y"] == j["y_s"]).all()), f"{p.name} y agrees")
        panel = panel.merge(
            s[key + ["score_raw"]].rename(columns={"score_raw": f"raw_{m}"}),
            on=key, how="left")
    check(panel[[f"raw_{m}" for m in MODELS]].notna().all().all(),
          "every row has all three raw scores")

    # ---- 2. the collapse --------------------------------------------------
    print("\n  2. COLLAPSE TO gvkey x fyear")
    g = panel.groupby(["gvkey", "fyear"], sort=False)
    score_cols = [f"score_{m}" for m in MODELS] + [f"raw_{m}" for m in MODELS]
    date_cols = ["datadate", "date_avail"]

    # A score may be taken once only if it is identical across the cell; for
    # the dates the test is that the cell holds a single value.
    for c in score_cols:
        spread = float((g[c].max() - g[c].min()).max())
        check(spread == 0.0, f"{c} is constant within the cell",
              f"max spread {spread:.3e}")
    for c in date_cols:
        n_multi = int((g[c].nunique() > 1).sum())
        check(n_multi == 0, f"{c} is single-valued within the cell",
              f"{n_multi} cells with more than one value")

    taken = {c: (c, "first") for c in score_cols + date_cols}
    out = g.agg(label=("y", "max"), n_issuer_rows=("issuer_id", "size"),
                **taken).reset_index()

    # ---- 3. the loud checks ----------------------------------------------
    print("\n  3. POSITIVES KEPT, AND WHAT THE OLD RULE GAVE")
    pos_rows = int(panel.y.sum())
    cells_with_pos = int(g.y.max().sum())
    old = panel.drop_duplicates(["gvkey", "fyear"])
    print(f"      positive issuer ROWS                     {pos_rows:>8,}")
    print(f"      gvkey x fyear CELLS holding a positive   {cells_with_pos:>8,}")
    print(f"      first-row de-duplication kept            "
          f"{int(old.y.sum()):>8,}   (loses "
          f"{pos_rows - int(old.y.sum()):,})")
    print(f"      this MAX collapse keeps                  "
          f"{int(out.label.sum()):>8,}")
    check(int(out.label.sum()) == cells_with_pos,
          "the MAX collapse loses no cell that held a positive")

    w, wo = out.fyear.between(2003, 2024), old.fyear.between(2003, 2024)
    print(f"\n      fiscal 2003-2024: this collapse "
          f"{int(out.loc[w, 'label'].sum()):,}, "
          f"against the old rule's {int(old.loc[wo, 'y'].sum()):,}")
    if expect_cells is None:
        print("      no recorded constant exists for this panel, so only the "
              "no-loss identity above is asserted")
    else:
        check(cells_with_pos == expect_cells,
              f"cells holding a positive equals the recorded {expect_cells:,}",
              f"got {cells_with_pos:,}")
        check(int(old.loc[wo, "y"].sum()) == expect_old,
              f"the old rule reproduces the recorded {expect_old:,} in "
              f"2003-2024", f"got {int(old.loc[wo, 'y'].sum()):,}")

    # `n_issuer_rows` is one column beyond the specified set: it records how
    # many issuer rows the cell collapsed, which is the provenance of the MAX.
    cols = (["gvkey", "fyear", "datadate", "date_avail", "label"]
            + score_cols + ["n_issuer_rows"])
    out = out[cols].sort_values(["gvkey", "fyear"]).reset_index(drop=True)
    print(f"\n      -> {out_path.name}: {len(out):,} rows x "
          f"{len(out.columns)} columns, {int(out.label.sum()):,} positives")
    return out


# ===========================================================================
# PART B — China, the event tables
# ===========================================================================
def build_maps():
    """The bond map, the two name maps, and the first offering date per Orgid."""
    bm = csmar(F_BONDMASTER_NAME, ["Liscd", "Sctcd", "Orgid", "Conme", "Ipodt"])
    pairs = bm[["Liscd", "Sctcd", "Orgid"]].drop_duplicates()

    # A bond code alone is NOT unique — 112061 exists under both Shanghai and
    # Shenzhen with different issuers — so the market is part of the key. The
    # code-only map is used only for the minor suffixes, which carry no market,
    # and only where the code resolves to exactly one Orgid.
    per_code = pairs.groupby("Liscd").Orgid.nunique()
    code_only = (pairs[pairs.Liscd.map(per_code).eq(1)]
                 .drop_duplicates("Liscd")[["Liscd", "Orgid"]]
                 .rename(columns={"Liscd": "code", "Orgid": "Orgid_code"}))

    # Exact-name maps. A name is usable only where it resolves to one Orgid.
    iss = csmar(F_ISSUER_NAME, ["InstitutionID", "InstitutionName"])
    nm_inst = (iss.dropna(subset=["InstitutionName"])
               .drop_duplicates("InstitutionName")
               .set_index("InstitutionName").InstitutionID)
    cm = bm[["Conme", "Orgid"]].dropna().drop_duplicates()
    nm_conme = (cm[cm.Conme.map(cm.groupby("Conme").Orgid.nunique()).eq(1)]
                .drop_duplicates("Conme").set_index("Conme").Orgid)

    # `Ipodt` carries a 2100-01-01 sentinel for bonds with no real issue date.
    # It is blanked BEFORE the per-Orgid minimum, because a sentinel left in
    # place cannot change a minimum but a 2099 placeholder could become one for
    # an issuer whose only other bonds are sentinels — and either way the value
    # is not an offering date. Everything from IPO_SENTINEL_YEAR on is treated
    # as missing, and the count is printed rather than absorbed.
    bm["ipo"] = pd.to_datetime(bm.Ipodt, errors="coerce")
    sentinel = bm.ipo.dt.year >= IPO_SENTINEL_YEAR
    print(f"      Ipodt in {IPO_SENTINEL_YEAR} or later, treated as MISSING: "
          f"{int(sentinel.sum()):,} bond rows "
          f"({sentinel.mean()*100:.3f}%), on {bm.loc[sentinel, 'Orgid'].nunique():,} "
          f"Orgids")
    bm.loc[sentinel, "ipo"] = pd.NaT
    first_off = bm.groupby("Orgid").ipo.min().rename("first_offering_date")
    print(f"      Orgids left with NO offering date at all: "
          f"{int(first_off.isna().sum()):,} of {len(first_off):,}")
    return bm, pairs, code_only, nm_inst, nm_conme, first_off


def map_bonds(df, pairs, code_only):
    """Attach `Org` to a frame carrying a `Symbol` column."""
    df = df.copy()
    df["suffix"] = (df.Symbol.str.extract(r"\.([A-Za-z]+)$")[0]
                    .fillna("NO SUFFIX"))
    df["code"] = df.Symbol.str.split(".").str[0]
    df["sct"] = df.suffix.map(SUF2SCT)
    df = df.merge(pairs, left_on=["code", "sct"],
                  right_on=["Liscd", "Sctcd"], how="left")
    df = df.merge(code_only, on="code", how="left")
    # The code-only fallback applies ONLY where the suffix carried no market.
    fallback = df.Orgid.isna() & df.sct.isna()
    df["Org"] = df.Orgid.where(~fallback, df.Orgid_code)
    return df


def name_lookup(name, nm_inst, nm_conme):
    """Exact name match: 发行人主体资料 first, then 债券基本情况表."""
    if name in nm_inst.index:
        return nm_inst[name], "name_institution"
    if name in nm_conme.index:
        return nm_conme[name], "name_conme"
    return None, None


def classify_move(prev, latest):
    """How one issuer-downgrade row moved, in four mutually exclusive classes.

      notch        the rating fell on the notch scale
      within-band  the rating STRING changed but the notch did not, which this
                   scale makes possible only inside the AAA band (AAA+ -> AAA
                   -> AAA- all map to 1). NOT a notch downgrade, and not an
                   outlook-only move either, so it gets its own class rather
                   than being filed under one of them.
      outlook      the rating string is identical; whatever was cut, it was
                   not the rating
      unclassified either side is off the notch scale
      upgrade      the rating rose, inside a file of downgrades: reported,
                   never hidden
    """
    p, l = NOTCH.get(prev), NOTCH.get(latest)
    if p is None or l is None:
        return "unclassified"
    if l > p:
        return "notch"
    if l < p:
        return "upgrade"
    return "outlook" if prev == latest else "within-band"


def build_china_events(pairs, code_only, nm_inst, nm_conme):
    """The three event tables on Orgid, plus the per-Orgid first dates."""
    hdr("PART B — CHINA EVENT TABLES")

    # ---- B.1 defaults and extensions --------------------------------------
    print("\n  B.1  DEFAULTS AND EXTENSIONS")
    d = read_event_csv(F_DEFAULT)
    print(f"      {F_DEFAULT}: {len(d):,} records")
    nb = d[~d.BondType.isin(ABS_DEFAULT)].copy()
    print(f"      excluding {sorted(ABS_DEFAULT)}: {len(d) - len(nb):,} rows "
          f"dropped, {len(nb):,} kept, {nb.IssuerName.nunique():,} issuer names")
    nb["dt"] = pd.to_datetime(nb.DefaultExtensionDate, errors="coerce")
    check(int(nb.dt.isna().sum()) == 0, "every kept default row has a date",
          f"{int(nb.dt.isna().sum())} unparseable")

    nb = map_bonds(nb, pairs, code_only)
    # One Orgid per issuer NAME: the modal Orgid across that issuer's matched
    # bonds. Disagreement is reported, not hidden.
    got = (nb[nb.Org.notna()].groupby(["IssuerName", "Org"]).size()
           .reset_index(name="n").sort_values("n", ascending=False)
           .drop_duplicates("IssuerName"))
    n_multi = int((nb[nb.Org.notna()].groupby("IssuerName").Org.nunique() > 1
                   ).sum())
    iss2org = {n: (o, "bond_code") for n, o in zip(got.IssuerName, got.Org)}
    for n in sorted(set(nb.IssuerName.dropna()) - set(iss2org)):
        o, how = name_lookup(n, nm_inst, nm_conme)
        if o is not None:
            iss2org[n] = (o, how)
    print(f"      issuers whose bonds point to more than one Orgid: "
          f"{n_multi} (the modal Orgid is used)")
    for k, v in pd.Series([v[1] for v in iss2org.values()]
                          ).value_counts().items():
        print(f"          {k:<20}{v:>5}")

    check(len(iss2org) == EXPECT_CN_ISSUERS,
          f"matched issuer names equals the recorded {EXPECT_CN_ISSUERS}",
          f"got {len(iss2org)}")
    n_org = len({v[0] for v in iss2org.values()})
    check(n_org == EXPECT_CN_ORGIDS,
          f"distinct matched Orgids equals the recorded {EXPECT_CN_ORGIDS}",
          f"got {n_org}")

    nb["Org"] = nb.IssuerName.map(lambda n: iss2org.get(n, (None, None))[0])
    nb["method"] = nb.IssuerName.map(lambda n: iss2org.get(n, (None, None))[1])
    dfl = nb.dropna(subset=["Org"])[["Org", "dt", "EventType", "method"]].copy()
    print(f"      event rows on an Orgid: {len(dfl):,}; 展期 among them: "
          f"{int(dfl.EventType.eq('展期').sum()):,} — every type counts")

    # ---- B.2 issuer downgrades --------------------------------------------
    print("\n  B.2  ISSUER DOWNGRADES")
    e = read_event_csv(F_ENT_DG)
    print(f"      {F_ENT_DG}: {len(e):,} records")
    keep = e.RatingObjectNature.isin(KEEP_OBJECT)
    print(f"      keeping RatingObjectNature in {sorted(KEEP_OBJECT)}: "
          f"{int(keep.sum()):,} kept, {int((~keep).sum()):,} dropped")
    check(int((~keep).sum()) == EXPECT_ENT_DG_DROPPED,
          f"guarantor / promisor rows dropped equals the recorded "
          f"{EXPECT_ENT_DG_DROPPED}", f"got {int((~keep).sum())}")
    e = e[keep].copy()
    e["dt"] = pd.to_datetime(e.RatingAdjustmentDate, errors="coerce")
    check(int(e.dt.isna().sum()) == 0,
          "every kept issuer-downgrade row has a date",
          f"{int(e.dt.isna().sum())} unparseable")
    e["move"] = [classify_move(p, l) for p, l
                 in zip(e.PreviousRating, e.LatestIssuerRating)]
    print("      move classification:")
    for k, v in e.move.value_counts().items():
        print(f"          {k:<16}{v:>6,}")
    n_wb = int(e.move.eq("within-band").sum())
    print(f"      within-band moves: {n_wb:,} — the rating string changed but "
          f"the notch did not.")
    print("      They are NOT notch downgrades, so they set no "
          "first_credit_date and enter")
    print("      NEITHER y_dg_issuer NOR y_dg_issuer_wide. Stated because it "
          "is a consequence.")
    off_scale = sorted((set(e.PreviousRating.dropna())
                        | set(e.LatestIssuerRating.dropna())) - set(NOTCH))
    if off_scale:
        print(f"      rating codes off the notch scale, reported not dropped "
              f"({len(off_scale)}): {off_scale[:12]}")

    ent2org = {}
    for n in sorted(set(e.EnterpriseName.dropna())):
        o, how = name_lookup(n, nm_inst, nm_conme)
        if o is not None:
            ent2org[n] = o
    e["Org"] = e.EnterpriseName.map(ent2org)
    print(f"      names {e.EnterpriseName.nunique():,}, matched "
          f"{len(ent2org):,}; event rows on an Orgid "
          f"{int(e.Org.notna().sum()):,}")
    edg = e.dropna(subset=["Org"])[["Org", "dt", "move"]].copy()

    # ---- B.3 bond downgrades ----------------------------------------------
    print("\n  B.3  BOND DOWNGRADES")
    b = read_event_csv(F_BND_DG)
    print(f"      {F_BND_DG}: {len(b):,} records")
    nab = b[~b.BondType.isin(ABS_DOWNGRADE)].copy()
    print(f"      excluding {sorted(ABS_DOWNGRADE)}: {len(b) - len(nab):,} "
          f"rows dropped, {len(nab):,} kept")
    nab["dt"] = pd.to_datetime(nab.LatestRatingDate, errors="coerce")
    check(int(nab.dt.isna().sum()) == 0,
          "every kept bond-downgrade row has a date",
          f"{int(nab.dt.isna().sum())} unparseable")
    nab["guaranteed"] = nab.GuarantorName.notna() & nab.GuarantorName.ne("")
    print(f"      rows with a GuarantorName: "
          f"{int(nab.guaranteed.sum()):,} ({nab.guaranteed.mean()*100:.1f}%)")
    nab = map_bonds(nab, pairs, code_only)
    print(f"      bonds {nab.Symbol.nunique():,}, matched "
          f"{nab[nab.Org.notna()].Symbol.nunique():,}; event rows on an Orgid "
          f"{int(nab.Org.notna().sum()):,}")
    bdg = nab.dropna(subset=["Org"])[["Org", "dt", "guaranteed"]].copy()

    # ---- B.4 the per-Orgid first dates ------------------------------------
    print("\n  B.4  FIRST EVENT DATES PER Orgid")
    fd = (dfl.sort_values("dt").drop_duplicates("Org")
          .set_index("Org")[["dt", "EventType"]]
          .rename(columns={"dt": "first_default_date",
                           "EventType": "default_event_type"}))
    # first_credit_date: the earliest of the three kinds, each tagged so the
    # type that SET the date is recoverable from the output.
    allev = pd.concat(
        [dfl.rename(columns={"EventType": "tag"})[["Org", "dt", "tag"]],
         edg[edg.move.eq("notch")].assign(tag="issuer_notch_downgrade")
         [["Org", "dt", "tag"]],
         bdg.assign(tag="bond_downgrade")[["Org", "dt", "tag"]]],
        ignore_index=True)
    fc = (allev.sort_values("dt").drop_duplicates("Org")
          .set_index("Org")[["dt", "tag"]]
          .rename(columns={"dt": "first_credit_date",
                           "tag": "credit_event_type"}))
    dg_tags = {"issuer_notch_downgrade", "bond_downgrade"}
    print(f"      Orgids with a default or extension       {len(fd):>6,}")
    print(f"      Orgids with any credit event             {len(fc):>6,}")
    print(f"      of those, whose FIRST event is a downgrade "
          f"{int(fc.credit_event_type.isin(dg_tags).sum()):>6,}")
    print(f"      first_credit_date earlier than first_default_date for "
          f"{int((fc.first_credit_date.reindex(fd.index) < fd.first_default_date).sum()):,} "
          f"of the {len(fd):,} default issuers")

    org_method = (dfl.drop_duplicates("Org").set_index("Org").method
                  .rename("match_method"))
    return fd, fc, edg, bdg, org_method


# ===========================================================================
# PART C — China, the two default labels
# ===========================================================================
def label_one(panel, anchor, first_off, name):
    """One US-style label. Returns (in_sample, y).

    `y` is <NA> outside the sample, because stage3_7 DROPS guard-excluded rows
    rather than calling them zero.
    """
    print(f"\n  {name}")
    off = panel.Org.map(first_off)
    in_pop = off.notna() & (panel.date_avail >= off)
    print(f"      rows before the guard                      {len(panel):>8,}")
    print(f"      rows whose Orgid has no offering date      "
          f"{int(off.isna().sum()):>8,}")
    print(f"      rows with date_avail >= first offering     "
          f"{int(in_pop.sum()):>8,}")

    a = panel.Org.map(anchor)
    drop = a.notna() & (panel.date_avail >= a)
    keep = in_pop & ~drop
    print(f"      rows dropped by date_avail >= the anchor   "
          f"{int((in_pop & drop).sum()):>8,}")
    print(f"      rows kept                                  "
          f"{int(keep.sum()):>8,}")

    # The absorbing state, asserted exactly as stage3_7:79-84 does.
    check(int((keep & a.notna() & (panel.date_avail >= a)).sum()) == 0,
          f"{name}: no kept row on or after its anchor")

    y = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    y[keep] = 0
    with_event = keep & a.notna()
    n_orgs = panel.loc[with_event, "Org"].nunique()
    if int(with_event.sum()):
        y[panel.loc[with_event].groupby("Org").date_avail.idxmax()] = 1
    n_ones = int((y == 1).sum())
    check(n_ones == n_orgs,
          f"{name}: exactly one positive per Orgid with a kept pre-event row",
          f"{n_ones} positives for {n_orgs} Orgids")
    return keep, y


def report_label(panel, y, anchor, name):
    """Positives by anchor year, how many are scored, and what is forgone."""
    ones = panel[y.eq(1)].copy()
    ones["anchor_year"] = ones.Org.map(anchor).dt.year
    print(f"\n      {name}: {len(ones):,} positives, "
          f"{int(ones.scored.sum()):,} on a SCORED row")
    print(f"      {'anchor year':<14}{'positives':>11}{'scored':>9}{'%':>7}")
    for yy, g in ones.groupby("anchor_year"):
        print(f"      {int(yy):<14}{len(g):>11,}{int(g.scored.sum()):>9,}"
              f"{g.scored.mean()*100:>6.0f}%")
    print(f"      {'TOTAL':<14}{len(ones):>11,}{int(ones.scored.sum()):>9,}"
          f"{ones.scored.mean()*100:>6.0f}%")

    # What the alternative rule would gain. COUNTED ONLY — not built, not
    # written, and no decision taken here.
    cand = panel[y.notna() & panel.scored & panel.Org.map(anchor).notna()]
    n_alt = cand.Org.nunique()
    print(f"      if the 1 went on the last SCORED pre-event row instead, "
          f"{n_alt:,} Orgids would carry a scored positive — "
          f"{n_alt - int(ones.scored.sum()):,} MORE than now "
          "(count only, not built)")


# ===========================================================================
# PART D — China, 12-month downgrade labels
# ===========================================================================
def downgrade_labels(panel, edg, bdg, fd):
    """Four forward labels on the (date_avail, date_avail + 12 months] window."""
    hdr("PART D — 12-MONTH DOWNGRADE LABELS (SUPPLEMENTARY)")
    print(f"  Window: (date_avail, date_avail + {DG_WINDOW_MONTHS} months]. A "
          "row whose date_avail is on or")
    print("  after its first_default_date gets a MISSING label, not 0: the firm "
          "is already in")
    print("  default, so a later downgrade is not a prediction.")

    spec = [("y_dg_issuer", edg[edg.move.eq("notch")]),
            ("y_dg_issuer_wide", edg[edg.move.isin(["notch", "outlook"])]),
            ("y_dg_bond", bdg),
            ("y_dg_bond_nogtee", bdg[~bdg.guaranteed])]

    base = panel[["Org", "date_avail"]].copy()
    base["win_end"] = panel.date_avail + pd.DateOffset(months=DG_WINDOW_MONTHS)
    base["row"] = base.index
    already = panel.Org.map(fd.first_default_date)
    in_default = already.notna() & (panel.date_avail >= already)

    out = {}
    for col, ev in spec:
        e = ev[["Org", "dt"]].dropna()
        j = base.merge(e, on="Org", how="inner")
        hit = j.loc[(j.dt > j.date_avail) & (j.dt <= j.win_end), "row"].unique()
        s = pd.Series(0, index=panel.index, dtype="Int64")
        s[hit] = 1
        s[in_default] = pd.NA
        out[col] = s
        print(f"      {col:<20}{len(e):>8,} events on an Orgid")

    w = panel.scored & panel.fyear.between(2013, 2024)
    print(f"\n  SCORED ROWS, FISCAL 2013-2024: {int(w.sum()):,} rows")
    print(f"      {'label':<20}{'labelled':>10}{'positives':>11}{'rate %':>9}")
    for col, s in out.items():
        sw = s[w]
        n, k = int(sw.notna().sum()), int((sw == 1).sum())
        print(f"      {col:<20}{n:>10,}{k:>11,}"
              f"{(k / n * 100 if n else float('nan')):>8.2f}")
    return out


# ===========================================================================
# main
# ===========================================================================
def main():
    print(BAR)
    print("STAGE 8.2 — DEFAULT AND DOWNGRADE LABELS.  Three NEW files; nothing "
          "existing is modified.")
    print(BAR)

    exist = [p for p in (US_D_OUT, US_R_OUT, CN_OUT) if p.exists()]
    if exist:
        for p in exist:
            print(f"      {p.name} already exists")
        raise SystemExit("STOP: this script never overwrites. Move or rename "
                         "the file(s) above first. NOTHING WRITTEN.")

    # ---- Part A -----------------------------------------------------------
    us_d = us_firmyear(US_DISTRESS, "stage4_1_scores_L3_K12", US_D_OUT,
                       EXPECT_US_DISTRESS_CELLS, EXPECT_US_DISTRESS_OLD_WINDOW)
    us_r = us_firmyear(US_RATING, "stage4_2_scores", US_R_OUT, None, None)

    # ---- Part B -----------------------------------------------------------
    hdr("READING THE BOND MASTER AND THE ISSUER MASTER (the slow part)")
    bm, pairs, code_only, nm_inst, nm_conme, first_off = build_maps()
    print(f"      {F_BONDMASTER_NAME}: {len(bm):,} rows, "
          f"{bm.Orgid.nunique():,} Orgids")
    n_future = int((first_off > pd.Timestamp.today()).sum())
    print(f"      Orgids whose first offering date is in the FUTURE: "
          f"{n_future:,} — kept and reported; every row of such an Orgid fails "
          "the guard")

    fd, fc, edg, bdg, org_method = build_china_events(
        pairs, code_only, nm_inst, nm_conme)

    # ---- Part C -----------------------------------------------------------
    hdr("PART C — CHINA, TWO DEFAULT LABELS ON THE US RULE")
    panel = pd.read_parquet(CN_PANEL)
    panel["Org"] = panel.gvkey.astype(str)
    panel["scored"] = panel[f"score_{SCORES[0]}"].notna()
    # date_avail exactly as stage3_5:54 builds it: calendar months, so a
    # month-end lands on a month-end.
    panel["date_avail"] = panel.datadate + pd.DateOffset(
        months=AVAIL_LAG_MONTHS)
    print(f"  {CN_PANEL.name}: {len(panel):,} rows, {panel.Org.nunique():,} "
          f"Orgids, {int(panel.scored.sum()):,} scored")
    check(int(panel.datadate.isna().sum()) == 0,
          "every panel row has a datadate")

    in_def, y_def = label_one(
        panel, fd.first_default_date, first_off,
        "y_default  (anchor: first default or extension)")
    report_label(panel, y_def, fd.first_default_date, "y_default")
    in_cr, y_cr = label_one(
        panel, fc.first_credit_date, first_off,
        "y_credit   (anchor: first default, extension or downgrade)")
    report_label(panel, y_cr, fc.first_credit_date, "y_credit")

    # ---- Part D -----------------------------------------------------------
    dg = downgrade_labels(panel, edg, bdg, fd)

    # ---- assemble ---------------------------------------------------------
    hdr("ASSEMBLING output/stage8_2_china_labels.parquet")
    cn = (panel[["Org", "fyear", "datadate", "date_avail", "scored"]]
          .rename(columns={"Org": "Orgid"}).copy())
    for s in SCORES:
        cn[f"score_{s}"] = panel[f"score_{s}"]
    cn["y_default"], cn["in_default"] = y_def, in_def
    cn["y_credit"], cn["in_credit"] = y_cr, in_cr
    for col, s in dg.items():
        cn[col] = s
    cn["first_default_date"] = panel.Org.map(fd.first_default_date)
    cn["default_event_type"] = panel.Org.map(fd.default_event_type)
    cn["first_credit_date"] = panel.Org.map(fc.first_credit_date)
    cn["credit_event_type"] = panel.Org.map(fc.credit_event_type)
    cn["match_method"] = panel.Org.map(org_method)
    cn = cn.sort_values(["Orgid", "fyear"]).reset_index(drop=True)
    check(len(cn) == len(panel), "the label table has one row per panel row",
          f"{len(cn):,} against {len(panel):,}")
    print(f"      {len(cn):,} rows x {len(cn.columns)} columns")
    print(f"      columns: {', '.join(cn.columns)}")

    # ---- the gate ---------------------------------------------------------
    hdr("ALL CHECKS")
    failed = [lab for ok, lab in CHECKS if not ok]
    print(f"      {len(CHECKS)} checks, {len(CHECKS) - len(failed)} passed, "
          f"{len(failed)} failed")
    for lab in failed:
        print(f"      FAILED   {lab}")
    if failed:
        raise SystemExit("\nSTOP: a check failed. NOTHING WRITTEN.")

    us_d.to_parquet(US_D_OUT, index=False)
    us_r.to_parquet(US_R_OUT, index=False)
    cn.to_parquet(CN_OUT, index=False)
    print(f"\n  Wrote {US_D_OUT.name}: {len(us_d):,} rows")
    print(f"  Wrote {US_R_OUT.name}: {len(us_r):,} rows")
    print(f"  Wrote {CN_OUT.name}: {len(cn):,} rows")
    print("\n  No definition is settled here. Two default labels and four "
          "downgrade labels are\n  written so the choice can be made on the "
          "numbers.")
    print(BAR)


if __name__ == "__main__":
    main()
