"""
Stage 3.3 — Match the FISD issuer master to Compustat gvkey. MEASUREMENT ONLY.

Builds a bridge table and reports match rates. It does NOT build the modelling
table, does not expand to a panel, does not attach features, and does not merge
the outcome onto anything.

Two passes, reported separately:
  pass 1 DIRECT — the issuer's own cusip6 to Compustat cusip6
  pass 2 PARENT — for pass-1 failures only, the parent's cusip6

This is D-06's route A followed by route C. Route B, the CRSP historical-CUSIP
route, is NOT run here; its gain remains unmeasured.

Run:  python3 code/stage3_3_issuer_gvkey_bridge.py
"""

from pathlib import Path

import pandas as pd

# A CUSIP issuer prefix is exactly six alphanumeric characters. Anything else
# is dirty and is dropped rather than truncated into a false match.
CUSIP6_RE = r"[A-Z0-9]{6}"


def clean_cusip6(series, label):
    """Upper-case, strip, take the first six, and keep only valid prefixes.

    Returns (cleaned series aligned to the input, boolean mask of what is
    usable). Whitespace and case are the two things that silently break a
    string join, so both are removed before anything is compared.
    """
    s = series.astype("string").str.strip().str.upper()
    six = s.str[:6]
    ok = six.str.fullmatch(CUSIP6_RE).fillna(False)
    n_in = int(series.notna().sum())
    print(f"    {label}: {n_in:,} non-null -> {int(ok.sum()):,} valid cusip6, "
          f"{n_in - int(ok.sum()):,} dropped as invalid")
    bad = six[series.notna() & ~ok]
    if len(bad):
        vc = bad.value_counts(dropna=False).head(5)
        print(f"      examples of dropped values: "
              f"{[(repr(k), int(v)) for k, v in vc.items()]}")
    return six, ok

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data_raw" / "us"
OUT_DIR = PROJECT_DIR / "output"

MASTER_IN = OUT_DIR / "stage3_2_issuer_master.parquet"
CUSIP_IN = RAW_DIR / "fisd_issuer_cusip.parquet"
ISSUE_IN = RAW_DIR / "fisd_mergedissue.parquet"
COMP_IN = RAW_DIR / "comp_names.parquet"
BRIDGE_OUT = OUT_DIR / "stage3_3_issuer_gvkey_bridge.parquet"


def main():
    # This script only ever creates a new file.
    if BRIDGE_OUT.exists():
        raise SystemExit(
            f"STOP: {BRIDGE_OUT.name} already exists. This script never overwrites; "
            "move or rename the existing file first."
        )

    # ---- the Compustat side ------------------------------------------------
    print("=" * 78)
    print("COMPUSTAT SIDE")
    print("=" * 78)
    comp = pd.read_parquet(COMP_IN, columns=["gvkey", "cusip"])
    print(f"  comp_names.parquet: {len(comp):,} rows, {comp.gvkey.nunique():,} gvkey")
    print(f"  cusip non-null: {comp.cusip.notna().mean()*100:.1f}%")
    print(f"  cusip lengths: {comp.cusip.dropna().str.len().value_counts().to_dict()}")

    # A 6-character CUSIP prefix identifies the issuing entity; the rest
    # identifies the security. D-06 matches on the first six.
    print("  cleaning:")
    comp["cusip6"], ok = clean_cusip6(comp.cusip, "Compustat cusip")
    comp = comp[ok][["gvkey", "cusip6"]].drop_duplicates()
    print(f"  usable (gvkey, cusip6) pairs: {len(comp):,}")
    print(f"  distinct cusip6: {comp.cusip6.nunique():,}  "
          f"-> cusip6 is NOT unique to gvkey; "
          f"{len(comp) - comp.cusip6.nunique():,} pairs share a cusip6")

    # ---- the FISD side -----------------------------------------------------
    print("\n" + "=" * 78)
    print("FISD SIDE")
    print("=" * 78)
    master = pd.read_parquet(MASTER_IN)
    master["issuer_id"] = master.issuer_id.astype("Int64")
    master["parent_id"] = master.parent_id.astype("Int64")
    print(f"  issuer master: {len(master):,} issuers")

    ic = pd.read_parquet(CUSIP_IN, columns=["issuer_id", "issuer_cusip"])
    ic["issuer_id"] = ic.issuer_id.astype("Int64")
    print(f"  fisd_issuer_cusip: {len(ic):,} rows, "
          f"{ic.issuer_id.nunique():,} issuers, "
          f"{ic.issuer_cusip.nunique():,} cusips")
    print(f"  issuer_cusip lengths: "
          f"{ic.issuer_cusip.dropna().str.len().value_counts().to_dict()}")
    print("  cleaning:")
    ic["cusip6"], ok = clean_cusip6(ic.issuer_cusip, "FISD issuer_cusip")
    ic = ic[ok][["issuer_id", "cusip6"]].drop_duplicates()
    per = ic.groupby("issuer_id").size()
    print(f"  usable (issuer_id, cusip6) pairs: {len(ic):,}   "
          f"issuers with >1 cusip: {int((per > 1).sum()):,}")

    # "Primary" is not a field in FISD. It is defined here as the cusip the
    # issuer's own bonds actually carry, most often.
    bonds = pd.read_parquet(ISSUE_IN, columns=["issuer_id", "issuer_cusip"])
    bonds["issuer_id"] = bonds.issuer_id.astype("Int64")
    print("  cleaning:")
    bonds["cusip6"], ok = clean_cusip6(bonds.issuer_cusip, "bond issuer_cusip")
    bonds = bonds[ok]
    primary = (bonds.groupby(["issuer_id", "cusip6"]).size()
               .reset_index(name="n")
               .sort_values(["issuer_id", "n"], ascending=[True, False])
               .drop_duplicates("issuer_id")[["issuer_id", "cusip6"]]
               .rename(columns={"cusip6": "primary_cusip6"}))
    print(f"  primary cusip6 derived for {len(primary):,} issuers "
          "(modal cusip on the issuer's own bonds)")

    # ---- pass 1: direct ----------------------------------------------------
    print("\n" + "=" * 78)
    print("PASS 1 — DIRECT")
    print("=" * 78)
    own = master[["issuer_id"]].merge(ic, on="issuer_id", how="inner")
    p1 = own.merge(comp, on="cusip6", how="inner")
    p1 = p1.rename(columns={"cusip6": "cusip_used"})
    p1["match_type"] = "direct"
    matched1 = set(p1.issuer_id)
    print(f"  issuers matched directly: {len(matched1):,} of {len(master):,} "
          f"({len(matched1)/len(master)*100:.1f}%)")
    print(f"  (issuer_id, gvkey) pairs: {len(p1[['issuer_id','gvkey']].drop_duplicates()):,}")

    # ---- pass 2: parent ----------------------------------------------------
    print("\n" + "=" * 78)
    print("PASS 2 — PARENT FALLBACK (pass-1 failures only)")
    print("=" * 78)
    resid = master[~master.issuer_id.isin(matched1)][["issuer_id", "parent_id"]]
    print(f"  unmatched after pass 1: {len(resid):,}")
    # A parent_id equal to the issuer's own id is not a parent.
    resid = resid[resid.parent_id.notna() & (resid.parent_id != resid.issuer_id)]
    print(f"  of those, with a distinct parent_id: {len(resid):,}")

    p2 = (resid.merge(ic.rename(columns={"issuer_id": "parent_id"}),
                      on="parent_id", how="inner")
          .merge(comp, on="cusip6", how="inner"))
    p2 = p2.rename(columns={"cusip6": "cusip_used"})[["issuer_id", "gvkey", "cusip_used"]]
    p2["match_type"] = "parent"
    matched2 = set(p2.issuer_id)
    print(f"  EXTRA issuers recovered via parent: {len(matched2):,} "
          f"({len(matched2)/len(master)*100:.1f}% of the master)")

    # ---- the bridge --------------------------------------------------------
    bridge = pd.concat(
        [p1[["issuer_id", "gvkey", "match_type", "cusip_used"]],
         p2[["issuer_id", "gvkey", "match_type", "cusip_used"]]],
        ignore_index=True).drop_duplicates()
    matched_all = matched1 | matched2

    print("\n" + "=" * 78)
    print("REPORT")
    print("=" * 78)

    # This caveat sits next to the numbers deliberately. Read the match rates
    # below as a floor, not as an estimate.
    print("""
  *** EVERY MATCH RATE BELOW IS A LOWER BOUND ***

  1. comp_names.parquet carries only the CURRENT cusip for each gvkey. It has
     no cusip history. A firm that changed cusip -- through a rename, a
     reorganisation, a merger -- is matched only if its LAST cusip happens to
     be the one FISD recorded.
  2. Cusip changes cluster around exactly the events this project studies:
     distress, restructuring and reorganisation. The DISTRESS-issuer match
     rate is therefore biased down more than the overall rate.
  3. D-06 route B -- CRSP historical cusips, crsp_stocknames.ncusip -> permno
     -> ccm_link -> gvkey -- is the route that exists to recover those firms.
     IT IS NOT RUN HERE. Its gain remains unmeasured.

  An unmatched issuer below therefore means "not matched by these two passes",
  NOT "absent from Compustat".
""")
    n = len(master)
    print(f"\n  of {n:,} issuers:")
    print(f"      pass 1 direct   : {len(matched1):>6,}  {len(matched1)/n*100:5.1f}%")
    print(f"      pass 2 parent   : {len(matched2):>6,}  {len(matched2)/n*100:5.1f}%")
    print(f"      TOTAL matched   : {len(matched_all):>6,}  {len(matched_all)/n*100:5.1f}%")
    print(f"      unmatched       : {n-len(matched_all):>6,}  "
          f"{(n-len(matched_all))/n*100:5.1f}%")

    # The count that decides whether a model can be estimated at all.
    dist = master[master.has_distress == "Y"]
    d1 = dist.issuer_id.isin(matched1).sum()
    d2 = dist.issuer_id.isin(matched2).sum()
    dall = dist.issuer_id.isin(matched_all).sum()
    print(f"\n  of {len(dist):,} DISTRESS issuers (the surviving-positive count):")
    print(f"      matched direct  : {int(d1):>6,}  {d1/len(dist)*100:5.1f}%")
    print(f"      matched parent  : {int(d2):>6,}  {d2/len(dist)*100:5.1f}%")
    print(f"      TOTAL matched   : {int(dall):>6,}  {dall/len(dist)*100:5.1f}%")
    print(f"      LOST            : {len(dist)-int(dall):>6,}  "
          f"{(len(dist)-dall)/len(dist)*100:5.1f}%")

    mm = master[master.issuer_id.isin(matched_all)]
    ny = int((mm.has_distress == "Y").sum()); nn = int((mm.has_distress == "N").sum())
    print(f"\n  the estimable sample, {len(mm):,} matched issuers:")
    print(f"      has_distress Y : {ny:,} ({ny/len(mm)*100:.1f}%)")
    print(f"      has_distress N : {nn:,} ({nn/len(mm)*100:.1f}%)")

    per_iss = bridge.groupby("issuer_id")["gvkey"].nunique()
    multi = int((per_iss > 1).sum())
    print(f"\n  ONE-TO-MANY: issuer_ids mapping to >1 gvkey: {multi:,} "
          f"({multi/len(per_iss)*100:.1f}% of matched)")
    if multi:
        print(f"      max gvkeys for one issuer: {int(per_iss.max())}")
        print(f"      total extra rows they contribute: "
              f"{int(per_iss[per_iss>1].sum() - multi):,}")

    # Matched only through a cusip that is not the issuer's primary one.
    # Written as a flag-then-max rather than groupby().apply(): the vectorised
    # form works on every pandas version and needs no include_groups argument.
    dm = bridge[bridge.match_type == "direct"].merge(primary, on="issuer_id", how="left")
    dm["used_primary"] = (dm.cusip_used == dm.primary_cusip6)
    by_iss = dm.groupby("issuer_id")["used_primary"].max()
    nonprimary = int((~by_iss.astype(bool)).sum())
    print(f"\n  matched ONLY via a non-primary cusip: {nonprimary:,} "
          f"of {len(by_iss):,} direct matches")

    print(f"\n  UNMATCHED DISTRESS ISSUERS — 10 examples:")
    um = dist[~dist.issuer_id.isin(matched_all)]
    print(f"      total unmatched distress issuers: {len(um):,}")
    cols = ["issuer_id", "cusip_name", "industry_group", "country_domicile",
            "first_distress_date"]
    print(um[cols].head(10).to_string(index=False))
    print("\n      unmatched distress issuers by industry_group:")
    for v, c in um.industry_group.value_counts(dropna=False).items():
        print(f"        group {str(v):<6} {c:>5,}")
    print("      unmatched distress issuers by domicile:")
    cd = um.country_domicile
    nonusa = cd.notna() & (cd.str.strip().str.upper() != "USA")
    print(f"        USA {int((cd.notna() & ~nonusa).sum()):,} | "
          f"non-USA {int(nonusa.sum()):,} | missing {int(cd.isna().sum()):,}")

    # ---- write -------------------------------------------------------------
    out = bridge.sort_values(["issuer_id", "gvkey"]).reset_index(drop=True)
    out.to_parquet(BRIDGE_OUT, index=False)
    print(f"\nWrote {BRIDGE_OUT.name}: {len(out):,} rows x {len(out.columns)} columns")
    print(f"  covering {out.issuer_id.nunique():,} issuers and "
          f"{out.gvkey.nunique():,} gvkeys")
    print("\n  head:")
    print(out.head(10).to_string(index=False))
    print("\nMeasurement only. No features attached, no panel, no outcome merged, "
          "nothing filtered out of any other file.")


if __name__ == "__main__":
    main()
