# ABOUTME: Stage 1.5 - measures how well the CSMAR Chinese data covers the items our features need
# ABOUTME: Pure measurement. It builds no features and decides nothing about which columns to keep.
"""
stage1_5_china_coverage.py

WHAT THIS DOES
    Reads the five CSMAR CSVs in `data_raw/china/` and reports, for every
    column, how much of it is actually populated. Nothing else. No feature is
    constructed, no column is judged, no filtering decision is taken here.

    This script reads the five CSVs without modifying them and writes its
    coverage reports exclusively into `output/`.

THE PANEL DEFINITION, WHICH IS WHERE THIS COULD SILENTLY GO WRONG
    The four statement files are keyed on Liscd (bond code) x Accper (period
    end). Many bonds belong to one issuer, and CSMAR repeats the issuer's
    financials on every one of its bonds. Counting raw rows would weight a firm
    by how many bonds it has sold, which has nothing to do with coverage. So,
    before anything is counted:

      1. Rows are restricted to `Typrep == 'A'` (合并报表, consolidated), per
         the CSMAR codebook. The parent-company rows are dropped and counted.
      2. Each bond is mapped to its issuer through `Orgid` (发行机构编码) in
         `债券基本情况表.csv`, joined on (Liscd, Sctcd) - the BOND CODE AND THE
         MARKET, not the bond code alone. A bond that maps to no issuer is
         DROPPED and counted - never kept under its own bond code, which would
         inflate the issuer count with one-bond pseudo-firms.

         The market must be part of the key. Chinese bond codes are reused
         across exchanges by unrelated issuers: `018001` is 中国中信集团有限公司
         in market 3 and 国家开发银行 in market 1, and `010203` is 国家开发银行
         in market 3 and 中华人民共和国财政部 in market 1. Measured on this
         data, 12,531 of 485,194 bond codes point at more than one issuer when
         Liscd is used alone, and those codes carry 22.77% of the consolidated
         balance-sheet rows - so a Liscd-only join would attach the wrong
         issuer to almost a quarter of the panel and the de-duplication would
         then merge unrelated firms. On (Liscd, Sctcd) only 13 pairs remain
         ambiguous; they are listed in the summary.
      3. Within each (issuer, period) the rows collapse to ONE: the row with
         the most non-missing values, ties broken by the smallest Liscd then
         the smallest Sctcd. Deterministic, and unlike "keep the first row" it
         cannot understate coverage.

         Whether the rule matters at all is MEASURED rather than assumed. Each
         row's data values are hashed, and the summary reports what share of
         multi-row (issuer, period) groups are identical across their rows. If
         that share is ~100%, CSMAR is simply repeating the issuer's accounts
         and the choice of rule is immaterial; if it is not, the rule is doing
         real work and the reader needs to know.

      4. Annual periods (Accper ending 12-31) and quarterly periods
         (03-31 / 06-30 / 09-30) are reported SEPARATELY and never pooled. Any
         other period end is reported and excluded from both, rather than
         forced into one.

WHAT COUNTS AS MISSING
    A blank cell is missing. A `0.0` is NOT missing - it is a number and it
    asserts something. CSMAR frequently writes 0 for a line item a firm simply
    does not have, which inflates a naive coverage rate. That is why "% zero
    among the non-missing" is a column of its own: read the two together, never
    the coverage rate alone.

THE FILE FORMAT
    Every cell is Excel-armoured as `="value"` inside ordinary CSV quoting, and
    each file has THREE header rows - column codes, Chinese names, units - with
    data from the fourth line. The armour is stripped after parsing. Every cell
    of the first chunk of each file is checked against the pattern, and after
    that any surviving armour would fail numeric conversion, which is counted
    per column and reported. A malformed cell cannot pass unnoticed by either
    route.

MEMORY
    ~2 GB of CSV, largest file 726 MB, but only ~400k rows per file: the files
    are wide with long numbers, not deep. Nothing is loaded whole.

    Each statement file is streamed TWICE. Pass 1 decides which rows survive
    the panel definition and retains only a small key table. Pass 2 re-reads
    the data columns and converts each chunk to float64 AS IT ARRIVES, so the
    panel that is held in memory is numeric, not strings - roughly 150k rows x
    152 columns ~= 180 MB for the balance sheet. Holding it as strings would
    have been about eight times that, which is the reason for the conversion.

    Chunk size adapts to the file's width so that a chunk is about two million
    cells whatever the file, and a guard refuses to allocate a panel larger
    than MEMORY_GUARD.

OUTPUTS
    output/stage1_5_china_coverage.csv   one row per (file, frequency, column)
    output/stage1_5_china_coverage.txt   the plain-text summary
"""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
CHINA_DIR = PROJECT / "data_raw" / "china"
OUT_DIR = PROJECT / "output"
# Local workbook assembled from CSMAR variable documentation for this audit.
# It is not a standard public download; obtain the corresponding metadata from
# an authorized CSMAR account and supply this project-specific workbook under
# data_raw/metadata/.
# Only this optional coverage audit reads it.
CODEBOOK = PROJECT / "data_raw" / "metadata" / "CSMAR_Dataset_Codebook.xlsx"

# Export these named tables as CSV from CSMAR (https://data.csmar.com/) through
# institutional access. Keep the original column codes and the three-row header,
# and save each export under data_raw/china/ with the exact filename below.
# The four statement files: keyed Liscd x Sctcd x Accper x Typrep.
STATEMENT_FILES = [
    "企业债公司债资产负债表.csv",           # issuer balance sheets
    "企业债公司债利润分配表.csv",           # issuer income statements
    "企业债公司债现金流量表(直接法).csv",   # direct-method cash flows
    "企业债公司债现金流量表(间接法).csv",   # indirect-method cash flows
]

# The bond master: one row per bond, carries the issuer identity. It has no
# Accper and no Typrep, so the panel definition above cannot apply to it; it is
# reported separately, at bond level.
MASTER_FILE = "债券基本情况表.csv"  # CSMAR bond master; maps bonds to issuers

# Each statement gets its own output file. EVERY statement is measured against
# its OWN issuer-period count and the five are never pooled - not in a
# denominator, not in a ranking, not in a table.
FILE_SLUG = {
    "企业债公司债资产负债表.csv": "balance_sheet",
    "企业债公司债利润分配表.csv": "income_statement",
    "企业债公司债现金流量表(直接法).csv": "cashflow_direct",
    "企业债公司债现金流量表(间接法).csv": "cashflow_indirect",
    "债券基本情况表.csv": "bond_master",
}

KEY_COLS = ["Liscd", "Sctcd", "Accper", "Typrep"]
ISSUER_CODE = "Orgid"      # 发行机构编码 - our issuer identifier
ISSUER_NAME = "Conme"      # 发行机构全称 - carried for human checking only

CONSOLIDATED = "A"         # Typrep 'A' = 合并报表
RECENT_FROM = 2015         # the "2015 and later" cut
QUARTER_ENDS = {"03-31", "06-30", "09-30"}
ANNUAL_END = "12-31"

CELLS_PER_CHUNK = 2_000_000    # chunk size adapts to file width
MEMORY_GUARD = 1_500_000_000   # refuse to allocate a panel bigger than this

# Every anomaly, warning and surprise goes here and is printed in the summary,
# so nothing is discoverable only by reading the console log closely.
NOTES = []


def note(msg):
    """Record something the reader must see, and echo it as it happens."""
    NOTES.append(msg)
    print("  [NOTE] " + msg)


def normalise(text):
    """Fold full-width brackets and spaces so names can be compared.

    The codebook writes 现金流量表（间接法） with full-width brackets while the
    file on disk uses half-width ones. Matching without this silently finds no
    codebook section for two of the four statement files.
    """
    return (
        str(text)
        .replace("（", "(")
        .replace("）", ")")
        .replace("：", ":")
        .replace(" ", "")
        .strip()
    )


# ---------------------------------------------------------------------------
# Reading the armoured CSMAR format
# ---------------------------------------------------------------------------


def _strip_list(cells):
    """Strip the `="..."` armour from a list of raw header cells."""
    return [c[2:-1] if len(c) >= 3 and c.startswith('="') and c.endswith('"') else c
            for c in cells]


def read_header(path):
    """Return the three header rows (codes, Chinese names, units).

    Read with the plain csv module: pandas cannot express "three header rows of
    different kinds", and guessing would be worse than reading them explicitly.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        rows = [next(reader) for _ in range(3)]
    codes, names, units = (_strip_list(r) for r in rows)
    if len(set(codes)) != len(codes):
        raise SystemExit(f"STOP: {path.name} has duplicate column codes in its header")
    return codes, names, units


def strip_armour(df, path_label, first_chunk):
    """Strip `="..."` from every cell of a string DataFrame, in place."""
    for col in df.columns:
        s = df[col]
        if first_chunk:
            nonempty = s != ""
            ok = s.str.startswith('="') & s.str.endswith('"')
            bad = int((nonempty & ~ok).sum())
            if bad:
                note(
                    f"{path_label}: column {col} has {bad} cells in the first chunk that "
                    f'do not match the `="..."` armour; they are sliced anyway and will '
                    f"show up as numeric conversion failures"
                )
        df[col] = s.str[2:-1]
    return df


def iter_chunks(path, codes, usecols=None):
    """Stream a CSMAR CSV in chunks, armour stripped, everything as strings.

    `keep_default_na=False` so an empty cell arrives as "" rather than NaN: we
    decide what missing means, not the parser.
    """
    width = len(usecols) if usecols else len(codes)
    chunksize = max(5_000, CELLS_PER_CHUNK // max(width, 1))
    use_idx = [codes.index(c) for c in usecols] if usecols else None

    reader = pd.read_csv(
        path,
        encoding="utf-8-sig",
        skiprows=3,              # the three header rows
        header=None,
        names=codes,
        usecols=use_idx,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        engine="c",
    )
    for i, chunk in enumerate(reader):
        if usecols:
            chunk = chunk[usecols].copy()   # copy: we are about to write to it
        yield strip_armour(chunk, path.name, first_chunk=(i == 0))


def to_numeric_counting_failures(s):
    """String column -> float column, plus the count of what would not parse.

    An empty string is missing. Anything non-empty that fails to parse is a
    genuine surprise in a column of 元; it is counted and reported, never
    dropped quietly.
    """
    nonempty = s != ""
    num = pd.to_numeric(s.where(nonempty, other=np.nan), errors="coerce")
    return num, int((nonempty & num.isna()).sum())


def hash_rows(df, cols):
    """One uint64 fingerprint per row, over `cols`, for detecting equal rows.

    Used only to answer "are an issuer's duplicate rows actually identical?".
    The per-column hashes are folded together with a multiply-and-xor, the
    usual way to combine hashes; uint64 arithmetic wraps around, as intended.
    """
    h = np.zeros(len(df), dtype=np.uint64)
    for c in cols:
        hc = pd.util.hash_pandas_object(df[c], index=False).to_numpy(dtype=np.uint64)
        h = (h * np.uint64(1000003)) ^ hc
    return h


# ---------------------------------------------------------------------------
# The codebook
# ---------------------------------------------------------------------------


def load_codebook():
    """Return {section title: {code: (chinese name, description)}}.

    The workbook is a sequence of sections, each opening with a row whose first
    cell starts with 文件 ("file"), then a header row 英文变量名 / 对应中文名 /
    变量说明, then one row per column. It documents more files than we hold on
    disk; the extras are reported rather than assumed irrelevant.
    """
    sections = {}
    for sheet in pd.ExcelFile(CODEBOOK).sheet_names:
        raw = pd.read_excel(CODEBOOK, sheet_name=sheet, header=None)
        current = None
        for _, row in raw.iterrows():
            first = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            if first.startswith("文件"):
                current = first
                sections[current] = {}
                continue
            if current is None or first in ("", "nan", "英文变量名"):
                continue
            name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
            desc = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
            sections[current][first] = (name, desc)
    return sections


def match_section(sections, filename):
    """Find the codebook section describing `filename`, by normalised name."""
    stem = normalise(Path(filename).stem)
    hits = [t for t in sections if stem in normalise(t)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        note(f"{filename}: no codebook section matches this file by name")
        return None
    note(f"{filename}: {len(hits)} codebook sections match ({hits}); using the first")
    return hits[0]


# ---------------------------------------------------------------------------
# Step 1 - the issuer map
# ---------------------------------------------------------------------------


def build_issuer_map():
    """Map (Liscd, Sctcd) -> (Orgid, Conme) from the bond master.

    Keyed on the bond code AND the market, because Chinese bond codes are
    reused across exchanges by unrelated issuers - see the module docstring for
    the measured size of the problem. The ambiguity under each of the two
    candidate keys is measured and reported, so the choice is evidenced rather
    than asserted.
    """
    path = CHINA_DIR / MASTER_FILE
    codes, _, _ = read_header(path)
    for needed in ("Liscd", "Sctcd", ISSUER_CODE, ISSUER_NAME):
        if needed not in codes:
            raise SystemExit(f"STOP: {MASTER_FILE} has no column {needed}")

    parts = [c for c in iter_chunks(path, codes,
                                    usecols=["Liscd", "Sctcd", ISSUER_CODE, ISSUER_NAME])]
    master = pd.concat(parts, ignore_index=True)
    del parts

    n_rows = len(master)
    n_liscd = int(master["Liscd"].nunique())
    if n_rows != n_liscd:
        note(f"{MASTER_FILE}: {n_rows} rows but {n_liscd} distinct Liscd - "
             f"the master is not one row per bond code")

    known = master[master[ISSUER_CODE] != ""]

    # Measure the ambiguity under BOTH candidate keys, so that the choice of
    # key is evidenced. Liscd alone is not safe: bond codes are reused across
    # exchanges by unrelated issuers.
    per_liscd = known.groupby("Liscd")[ISSUER_CODE].nunique()
    ambiguous_liscd = per_liscd[per_liscd > 1].index.tolist()
    per_pair = known.groupby(["Liscd", "Sctcd"])[ISSUER_CODE].nunique()
    ambiguous_pairs = per_pair[per_pair > 1].index.tolist()

    if ambiguous_liscd:
        note(f"{MASTER_FILE}: {len(ambiguous_liscd)} Liscd point at more than one Orgid, "
             f"which is why the join key is (Liscd, Sctcd) and not Liscd alone")
    if ambiguous_pairs:
        note(f"{MASTER_FILE}: {len(ambiguous_pairs)} (Liscd, Sctcd) pairs STILL point at "
             f"more than one Orgid; the smallest Orgid is used for those and they are "
             f"listed in full in the summary")

    # One issuer per (bond code, market); sorting makes the choice deterministic.
    lookup = (known.sort_values([ISSUER_CODE])
                   .drop_duplicates(["Liscd", "Sctcd"], keep="first")
                   .set_index(["Liscd", "Sctcd"])[[ISSUER_CODE, ISSUER_NAME]])

    stats = {
        "master_rows": n_rows,
        "master_distinct_liscd": n_liscd,
        "master_distinct_pairs": int(len(per_pair)),
        "master_missing_orgid": int((master[ISSUER_CODE] == "").sum()),
        "master_distinct_orgid": int(known[ISSUER_CODE].nunique()),
        "ambiguous_liscd": len(ambiguous_liscd),
        "ambiguous_liscd_examples": ambiguous_liscd[:10],
        "ambiguous_pairs": len(ambiguous_pairs),
        "ambiguous_pairs_all": ambiguous_pairs,
    }
    del master
    return lookup, stats


# ---------------------------------------------------------------------------
# Step 2 - which rows survive the panel definition
# ---------------------------------------------------------------------------


def select_panel_rows(path, codes, lookup):
    """Pass 1 over a statement file: decide which rows form the issuer panel.

    Retains only a small key table - never any values - so this pass costs
    almost nothing in memory however wide the file is.
    """
    data_cols = [c for c in codes if c not in KEY_COLS]
    keep_parts = []
    typrep_counts = {}
    offset = 0
    total_rows = 0

    for chunk in iter_chunks(path, codes):
        n = len(chunk)
        total_rows += n
        # Count every Typrep value BEFORE filtering, so the drop is explainable.
        for val, cnt in chunk["Typrep"].value_counts().items():
            typrep_counts[val] = typrep_counts.get(val, 0) + int(cnt)

        mask = (chunk["Typrep"] == CONSOLIDATED).to_numpy()
        if mask.any():
            sub = chunk[mask]
            keep_parts.append(pd.DataFrame({
                "row_pos": np.arange(offset, offset + n)[mask],
                "Liscd": sub["Liscd"].to_numpy(),
                "Sctcd": sub["Sctcd"].to_numpy(),
                "Accper": sub["Accper"].to_numpy(),
                # How much of the row is filled in - the collapse tie-break.
                "n_nonmiss": (sub[data_cols] != "").sum(axis=1).to_numpy(),
                # Fingerprint of the row's values, to test whether an issuer's
                # duplicate rows are truly identical.
                "row_hash": hash_rows(sub, data_cols),
            }))
        offset += n

    keys = pd.concat(keep_parts, ignore_index=True)
    del keep_parts
    n_consolidated = len(keys)

    # Attach the issuer on (bond code, market) - see build_issuer_map for why
    # the market is part of the key. A bond with no issuer cannot enter an
    # issuer panel, so those rows are dropped and counted.
    keys = keys.join(lookup, on=["Liscd", "Sctcd"])
    unmapped = int(keys[ISSUER_CODE].isna().sum())
    unmapped_liscd = int(keys.loc[keys[ISSUER_CODE].isna(), "Liscd"].nunique())
    keys = keys[keys[ISSUER_CODE].notna()]

    # Period end -> year and frequency.
    accper = keys["Accper"].astype(str)
    well_formed = accper.str.match(r"^\d{4}-\d{2}-\d{2}$")
    n_bad_accper = int((~well_formed).sum())
    if n_bad_accper:
        note(f"{path.name}: {n_bad_accper} rows have an Accper that is not YYYY-MM-DD "
             f"(examples: {accper[~well_formed].unique()[:5].tolist()}); excluded")
    keys = keys[well_formed]

    accper = keys["Accper"].astype(str)
    keys["year"] = accper.str.slice(0, 4).astype(int)
    md = accper.str.slice(5, 10)
    keys["freq"] = np.where(md == ANNUAL_END, "annual",
                            np.where(md.isin(QUARTER_ENDS), "quarterly", "other"))
    other = keys["freq"] == "other"
    n_other = int(other.sum())
    if n_other:
        note(f"{path.name}: {n_other} rows have a period end that is neither 12-31 nor a "
             f"quarter end (examples: {md[other].unique()[:5].tolist()}); excluded from "
             f"both the annual and the quarterly panel")
    keys = keys[~other]

    # Are an issuer's duplicate rows for one period actually the same numbers?
    # Measured, not assumed: it decides whether the collapse rule matters.
    grp = keys.groupby([ISSUER_CODE, "Accper"], sort=False)
    sizes = grp.size()
    distinct_hashes = grp["row_hash"].nunique()
    multi = sizes > 1

    # Collapse to one row per issuer-period.
    keys = (keys.sort_values([ISSUER_CODE, "Accper", "n_nonmiss", "Liscd", "Sctcd"],
                             ascending=[True, True, False, True, True])
                .drop_duplicates([ISSUER_CODE, "Accper"], keep="first"))

    counts = {
        "total_rows": total_rows,
        "typrep_counts": typrep_counts,
        "n_consolidated": n_consolidated,
        "n_dropped_not_A": total_rows - n_consolidated,
        "n_unmapped_rows": unmapped,
        "n_unmapped_liscd": unmapped_liscd,
        "n_bad_accper": n_bad_accper,
        "n_other_period_end": n_other,
        "n_groups": int(len(sizes)),
        "n_dup_groups": int(multi.sum()),
        "n_dup_groups_identical": int((multi & (distinct_hashes == 1)).sum()),
        "n_panel_rows": len(keys),
    }
    return keys, counts


def fetch_panel_values(path, codes, row_positions):
    """Pass 2: re-read the data columns, keeping only the chosen rows.

    Each chunk is converted to float AS IT ARRIVES. Holding the panel as
    strings would cost roughly eight times the memory for no benefit.
    """
    data_cols = [c for c in codes if c not in KEY_COLS]
    est = len(row_positions) * len(data_cols) * 8
    if est > MEMORY_GUARD:
        raise SystemExit(
            f"STOP: {path.name} would need about {est/1e9:.1f} GB "
            f"({len(row_positions)} rows x {len(data_cols)} columns). Refusing to allocate."
        )

    wanted = np.zeros(int(row_positions.max()) + 1, dtype=bool)
    wanted[row_positions] = True

    parts = []
    failures = {c: 0 for c in data_cols}
    offset = 0
    for chunk in iter_chunks(path, codes, usecols=data_cols):
        n = len(chunk)
        mask = np.zeros(n, dtype=bool)
        end = min(offset + n, len(wanted))
        if end > offset:
            mask[: end - offset] = wanted[offset:end]
        if mask.any():
            sub = chunk[mask]
            out = {}
            for c in data_cols:
                num, bad = to_numeric_counting_failures(sub[c])
                out[c] = num.to_numpy(dtype="float64")
                failures[c] += bad
            parts.append(pd.DataFrame(out))
        offset += n

    panel = pd.concat(parts, ignore_index=True)
    for c, bad in failures.items():
        if bad:
            note(f"{path.name}: column {c} has {bad} non-empty values that are not numeric")
    return panel, failures


# ---------------------------------------------------------------------------
# Step 3 - the statistics
# ---------------------------------------------------------------------------


def column_stats(values, years, col, file_label, freq_label, n_rows, n_rows_recent, failures):
    """Every measurement asked for, for one column of one panel."""
    obs = values.notna()
    n_obs = int(obs.sum())
    n_obs_recent = int((obs & (years >= RECENT_FROM)).sum())

    row = {
        "file": file_label,
        "freq": freq_label,
        "col_code": col,
        "n_panel_rows": n_rows,
        "n_obs": n_obs,
        "pct_non_missing": round(100 * n_obs / n_rows, 3) if n_rows else np.nan,
        "n_panel_rows_2015plus": n_rows_recent,
        "n_obs_2015plus": n_obs_recent,
        "pct_non_missing_2015plus": (round(100 * n_obs_recent / n_rows_recent, 3)
                                     if n_rows_recent else np.nan),
        "n_numeric_failures": failures,
        "dtype_guess": "numeric",
    }
    if n_obs:
        v = values[obs]
        yrs = years[obs]
        row["pct_zero_of_non_missing"] = round(100 * float((v == 0).mean()), 3)
        row["p1"] = float(v.quantile(0.01))
        row["median"] = float(v.median())
        row["p99"] = float(v.quantile(0.99))
        row["first_year"] = int(yrs.min())
        row["last_year"] = int(yrs.max())
    else:
        for k in ("pct_zero_of_non_missing", "p1", "median", "p99", "first_year", "last_year"):
            row[k] = np.nan
    return row


def measure_statement_file(filename, lookup, sections):
    """Measure one statement file, annual and quarterly reported separately."""
    path = CHINA_DIR / filename
    print(f"\n=== {filename} ===")
    codes, names, units = read_header(path)
    name_by_code = dict(zip(codes, names))
    unit_by_code = dict(zip(codes, units))
    data_cols = [c for c in codes if c not in KEY_COLS]
    print(f"  {len(codes)} columns ({len(data_cols)} data columns)")

    print("  pass 1: selecting the issuer panel ...")
    keys, counts = select_panel_rows(path, codes, lookup)
    print(f"  {counts['total_rows']} rows -> {counts['n_consolidated']} consolidated "
          f"-> {counts['n_panel_rows']} issuer-periods")

    print("  pass 2: reading the selected rows ...")
    panel, failures = fetch_panel_values(path, codes, keys["row_pos"].to_numpy())

    # Re-attach year, frequency and issuer BY ROW POSITION, so a re-sort in
    # either pass cannot silently misalign the two.
    order = keys.sort_values("row_pos")
    if len(panel) != len(order):
        raise SystemExit(f"STOP: {filename} pass 2 returned {len(panel)} rows "
                         f"but pass 1 chose {len(order)}")
    panel["_year"] = order["year"].to_numpy()
    panel["_freq"] = order["freq"].to_numpy()
    panel["_issuer"] = order[ISSUER_CODE].to_numpy()

    rows = []
    per_freq = {}
    for freq in ("annual", "quarterly"):
        sub = panel[panel["_freq"] == freq]
        n_rows = len(sub)
        years = sub["_year"]
        n_recent = int((years >= RECENT_FROM).sum())
        per_freq[freq] = {
            "n_rows": n_rows,
            "n_issuers": int(sub["_issuer"].nunique()) if n_rows else 0,
            # The issuer set is kept so the summary can report how far the five
            # statements agree on WHICH firms they cover, not merely how many.
            "issuers": set(sub["_issuer"].unique()) if n_rows else set(),
            "year_min": int(years.min()) if n_rows else None,
            "year_max": int(years.max()) if n_rows else None,
        }
        if n_rows == 0:
            note(f"{filename}: no {freq} rows at all")
            continue
        for col in data_cols:
            r = column_stats(sub[col], years, col, filename, freq,
                             n_rows, n_recent, failures[col])
            r["col_name_data"] = name_by_code.get(col, "")
            r["unit"] = unit_by_code.get(col, "")
            rows.append(r)

    del panel
    # The codebook documents the key columns too, so the set returned for the
    # codebook comparison must be ALL columns, not just the data columns -
    # otherwise Liscd/Sctcd/Accper/Typrep are reported as "documented but
    # absent", which they plainly are not.
    return rows, counts, per_freq, match_section(sections, filename), set(codes)


def measure_master_file(sections):
    """Measure the bond master at bond level, in two passes.

    It has no Accper and no Typrep, so the issuer-period panel definition does
    not apply: coverage is reported per bond row. The columns that need a
    period - the 2015+ rate, the first and last year - are left BLANK rather
    than derived from a listing or issue date, which would be a judgement this
    step is not permitted to make.

    Pass 1 counts non-missing values and numeric failures per column without
    retaining anything. Pass 2 re-reads only the columns that proved fully
    numeric and accumulates their values for the percentiles - most of this
    file is text and dates, and holding those in memory would buy nothing.
    """
    filename = MASTER_FILE
    path = CHINA_DIR / filename
    print(f"\n=== {filename} (bond level) ===")
    codes, names, units = read_header(path)
    name_by_code = dict(zip(codes, names))
    unit_by_code = dict(zip(codes, units))

    print("  pass 1: counting non-missing and numeric failures ...")
    n_obs = {c: 0 for c in codes}
    failures = {c: 0 for c in codes}
    n_rows = 0
    liscd_seen = set()
    for chunk in iter_chunks(path, codes):
        n_rows += len(chunk)
        liscd_seen.update(chunk["Liscd"].unique().tolist())
        for c in codes:
            s = chunk[c]
            nonempty = s != ""
            n_obs[c] += int(nonempty.sum())
            _, bad = to_numeric_counting_failures(s)
            failures[c] += bad

    # A column counts as numeric only if EVERY non-empty value parsed.
    numeric_cols = [c for c in codes if n_obs[c] > 0 and failures[c] == 0]
    print(f"  {n_rows} rows; {len(numeric_cols)} of {len(codes)} columns are fully numeric")

    print("  pass 2: reading the numeric columns for percentiles ...")
    values = {}
    if numeric_cols:
        parts = []
        for chunk in iter_chunks(path, codes, usecols=numeric_cols):
            parts.append(pd.DataFrame(
                {c: to_numeric_counting_failures(chunk[c])[0].to_numpy(dtype="float64")
                 for c in numeric_cols}
            ))
        values = pd.concat(parts, ignore_index=True)
        del parts

    rows = []
    for col in codes:
        numeric = col in numeric_cols
        r = {
            "file": filename,
            "freq": "bond-level",
            "col_code": col,
            "col_name_data": name_by_code.get(col, ""),
            "unit": unit_by_code.get(col, ""),
            "n_panel_rows": n_rows,
            "n_obs": n_obs[col],
            "pct_non_missing": round(100 * n_obs[col] / n_rows, 3) if n_rows else np.nan,
            "n_panel_rows_2015plus": np.nan,
            "n_obs_2015plus": np.nan,
            "pct_non_missing_2015plus": np.nan,
            "first_year": np.nan,
            "last_year": np.nan,
            "n_numeric_failures": 0 if numeric else failures[col],
            "dtype_guess": "numeric" if numeric else "text",
        }
        if numeric:
            v = values[col].dropna()
            r["pct_zero_of_non_missing"] = round(100 * float((v == 0).mean()), 3)
            r["p1"] = float(v.quantile(0.01))
            r["median"] = float(v.median())
            r["p99"] = float(v.quantile(0.99))
        else:
            for k in ("pct_zero_of_non_missing", "p1", "median", "p99"):
                r[k] = np.nan
        rows.append(r)

    stats = {"n_rows": n_rows, "n_distinct_liscd": len(liscd_seen)}
    return rows, stats, match_section(sections, filename), set(codes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    started = datetime.now()
    OUT_DIR.mkdir(exist_ok=True)

    for f in STATEMENT_FILES + [MASTER_FILE]:
        if not (CHINA_DIR / f).exists():
            raise SystemExit(f"STOP: missing input {CHINA_DIR / f}")

    print("Reading the codebook ...")
    sections = load_codebook()
    print(f"  {len(sections)} sections")

    print("\nBuilding the issuer map from the bond master ...")
    lookup, master_stats = build_issuer_map()
    print(f"  {master_stats['master_rows']} bond rows, "
          f"{master_stats['master_distinct_orgid']} distinct issuers (Orgid), "
          f"{master_stats['master_missing_orgid']} rows with no Orgid")

    all_rows, file_counts, file_freqs, file_sections, file_cols = [], {}, {}, {}, {}

    for filename in STATEMENT_FILES:
        rows, counts, per_freq, section, cols = measure_statement_file(filename, lookup, sections)
        all_rows.extend(rows)
        file_counts[filename] = counts
        file_freqs[filename] = per_freq
        file_sections[filename] = section
        file_cols[filename] = cols

    rows, master_level, section, cols = measure_master_file(sections)
    all_rows.extend(rows)
    file_sections[MASTER_FILE] = section
    file_cols[MASTER_FILE] = cols

    # ---- assemble the table ------------------------------------------------
    table = pd.DataFrame(all_rows)
    cb_name, cb_desc, in_cb = [], [], []
    for f, code in zip(table["file"], table["col_code"]):
        sec = file_sections.get(f)
        entry = sections.get(sec, {}).get(code) if sec else None
        cb_name.append(entry[0] if entry else "")
        cb_desc.append(entry[1] if entry else "")
        in_cb.append(bool(entry))
    table["col_name_codebook"] = cb_name
    table["codebook_description"] = cb_desc
    table["in_codebook"] = in_cb

    table = table[[
        "file", "freq", "col_code", "col_name_data", "col_name_codebook", "unit",
        "in_codebook", "dtype_guess",
        "n_panel_rows", "n_obs", "pct_non_missing",
        "n_panel_rows_2015plus", "n_obs_2015plus", "pct_non_missing_2015plus",
        "pct_zero_of_non_missing", "median", "p1", "p99",
        "first_year", "last_year", "n_numeric_failures", "codebook_description",
    ]]

    # ONE FILE PER STATEMENT. Every column is measured against its own
    # statement's issuer-period count, so the five are never comparable by
    # denominator and are not written into a shared table.
    print()
    for filename, slug in FILE_SLUG.items():
        sub = table[table["file"] == filename]
        path = OUT_DIR / f"stage1_5_china_coverage_{slug}.csv"
        sub.to_csv(path, index=False, encoding="utf-8-sig")
        denoms = sorted(sub["n_panel_rows"].unique())
        print(f"Wrote {path.name} ({len(sub)} rows, own denominators {denoms})")

    # ---- the summary -------------------------------------------------------
    L = []
    L.append("Stage 1.5 - coverage of the CSMAR Chinese data")
    L.append(f"Generated {started:%Y-%m-%d %H:%M}. Measurement only: no feature is built")
    L.append("here and nothing is decided about which columns to keep.")
    L.append("")
    L.append("HOW THE PANEL WAS DEFINED")
    L.append("  Issuer = Orgid (发行机构编码) from 债券基本情况表.csv, joined to each")
    L.append("  statement file on (Liscd, Sctcd) - bond code AND market, never the bond")
    L.append("  code alone; see THE ISSUER MAP below for the measured reason.")
    L.append("  Rows kept only where Typrep == 'A' (合并报表).")
    L.append("  One row per (issuer, Accper): the row with the most non-missing values,")
    L.append("  ties broken by the smallest Liscd then Sctcd. Annual (12-31) and")
    L.append("  quarterly (03-31/06-30/09-30) periods are reported separately.")
    L.append("  A blank is missing; a 0.0 is a value. Read pct_zero_of_non_missing")
    L.append("  together with pct_non_missing, never the coverage rate on its own -")
    L.append("  CSMAR writes 0 for line items a firm does not have.")
    L.append("")
    L.append("THE ISSUER MAP")
    L.append("  The join key is (Liscd, Sctcd) - bond code AND market. Chinese bond")
    L.append("  codes are reused across exchanges by unrelated issuers, so a Liscd-only")
    L.append("  join attaches the wrong issuer to a large minority of rows and the")
    L.append("  de-duplication then merges different firms. The two counts below are")
    L.append("  the evidence for that choice.")
    L.append(f"  bond master rows                 {master_stats['master_rows']}")
    L.append(f"  distinct Liscd                   {master_stats['master_distinct_liscd']}")
    L.append(f"  distinct (Liscd, Sctcd)          {master_stats['master_distinct_pairs']}")
    L.append(f"  distinct issuers (Orgid)         {master_stats['master_distinct_orgid']}")
    L.append(f"  rows with no Orgid               {master_stats['master_missing_orgid']}")
    L.append(f"  Liscd alone -> >1 Orgid          {master_stats['ambiguous_liscd']}  "
             f"(REJECTED as a key for this reason)")
    if master_stats["ambiguous_liscd_examples"]:
        L.append(f"    examples: {master_stats['ambiguous_liscd_examples']}")
    L.append(f"  (Liscd, Sctcd) -> >1 Orgid       {master_stats['ambiguous_pairs']}  "
             f"(the residual ambiguity we live with)")
    for pair in master_stats["ambiguous_pairs_all"]:
        L.append(f"    {pair}")
    L.append("")

    for filename in STATEMENT_FILES:
        c, pf = file_counts[filename], file_freqs[filename]
        L.append(f"FILE: {filename}")
        L.append(f"  raw data rows                    {c['total_rows']}")
        L.append(f"  Typrep values                    {c['typrep_counts']}")
        L.append(f"  dropped, not Typrep=='A'         {c['n_dropped_not_A']}")
        L.append(f"  dropped, Liscd has no issuer     {c['n_unmapped_rows']} rows "
                 f"({c['n_unmapped_liscd']} distinct bond codes)")
        L.append(f"  dropped, malformed Accper        {c['n_bad_accper']}")
        L.append(f"  dropped, period end neither 12-31 nor a quarter end   "
                 f"{c['n_other_period_end']}")
        L.append(f"  (issuer, period) groups          {c['n_groups']}")
        L.append(f"    with more than one row         {c['n_dup_groups']}")
        L.append(f"    of those, all rows identical   {c['n_dup_groups_identical']}")
        L.append(f"  issuer-periods after collapse    {c['n_panel_rows']}")
        for freq in ("annual", "quarterly"):
            d = pf[freq]
            L.append(f"  {freq:<10} rows {d['n_rows']:>8}  issuers {d['n_issuers']:>7}  "
                     f"years {d['year_min']}-{d['year_max']}")
        L.append(f"  -> every percentage for this statement is taken on ITS OWN counts "
                 f"above, never on another statement's")
        L.append("")

    # How far do the statements agree on WHICH issuers they cover? Reported
    # because equal issuer COUNTS across statements do not imply the same
    # firms, and a later merge would silently lose the difference.
    L.append("DO THE STATEMENTS COVER THE SAME ISSUERS?")
    for freq in ("annual", "quarterly"):
        sets = {f: file_freqs[f][freq]["issuers"] for f in STATEMENT_FILES}
        union = set().union(*sets.values())
        inter = set.intersection(*sets.values()) if sets else set()
        L.append(f"  {freq}: union {len(union)} issuers, present in all four "
                 f"{len(inter)}")
        for f in STATEMENT_FILES:
            only_missing = len(union - sets[f])
            L.append(f"    {Path(f).stem:<24} {len(sets[f]):>6}  "
                     f"({only_missing} of the union absent here)")
    L.append("")

    L.append(f"FILE: {MASTER_FILE} (bond level - it has no Accper and no Typrep)")
    L.append(f"  rows                             {master_level['n_rows']}")
    L.append(f"  distinct Liscd                   {master_level['n_distinct_liscd']}")
    L.append("  The 2015+ rate and the first/last year are blank for this file: it")
    L.append("  carries no accounting period, and deriving one from a listing or issue")
    L.append("  date would be a judgement this step is not permitted to make.")
    L.append("")

    # Worst and best coverage, ranked WITHIN each statement and each frequency.
    # Ranking across statements would compare percentages taken on different
    # denominators, which is the pooling this report exists to avoid.
    for filename in STATEMENT_FILES:
        for freq in ("annual", "quarterly"):
            sub = table[(table["file"] == filename) & (table["freq"] == freq)]
            if sub.empty:
                continue
            denom = int(sub["n_panel_rows"].iloc[0])
            for label, asc in (("WORST", True), ("BEST", False)):
                n = min(20, len(sub))
                top = sub.sort_values(["pct_non_missing", "col_code"],
                                      ascending=[asc, True]).head(n)
                L.append(f"{label} {n} COVERAGE - {filename} [{freq}], "
                         f"out of {denom} issuer-periods of this statement")
                for _, r in top.iterrows():
                    L.append(f"  {r['pct_non_missing']:>7.2f}% filled, "
                             f"{r['pct_zero_of_non_missing']:>6.2f}% of those zero  "
                             f"{r['col_code']:<12} {str(r['col_name_data'])[:26]}")
                L.append("")

    L.append("IN THE CODEBOOK BUT NOT IN THE DATA")
    gaps = False
    for filename, section in file_sections.items():
        if not section:
            L.append(f"  {filename}: no codebook section matched; cannot compare")
            gaps = True
            continue
        absent = sorted(set(sections[section]) - file_cols[filename])
        if absent:
            gaps = True
            L.append(f"  {filename}: {len(absent)} documented columns absent from the header")
            for code in absent:
                L.append(f"    {code}  {sections[section][code][0]}")
    if not gaps:
        L.append("  none - every documented column is present in its file's header")
    L.append("")

    L.append("CODEBOOK SECTIONS WITH NO FILE ON DISK")
    unmatched = [t for t in sections if t not in set(file_sections.values())]
    if unmatched:
        L.append("  (documented datasets we do not hold; listed so the gap is visible)")
        for t in unmatched:
            L.append(f"    {t}  ({len(sections[t])} columns)")
    else:
        L.append("  none")
    L.append("")

    L.append("PRESENT IN THE DATA BUT ENTIRELY EMPTY")
    empty = table[table["n_obs"] == 0]
    if empty.empty:
        L.append("  none - every column has at least one value")
    else:
        for _, r in empty.iterrows():
            L.append(f"  {Path(r['file']).stem} [{r['freq']}] {r['col_code']}  "
                     f"{r['col_name_data']}")
    L.append("")

    L.append("PRESENT IN THE DATA BUT NOT IN THE CODEBOOK")
    undoc = table.loc[~table["in_codebook"], ["file", "col_code", "col_name_data"]]
    undoc = undoc.drop_duplicates()
    if undoc.empty:
        L.append("  none")
    else:
        for _, r in undoc.iterrows():
            L.append(f"  {Path(r['file']).stem} {r['col_code']}  {r['col_name_data']}")
    L.append("")

    L.append("EVERYTHING UNUSUAL SEEN DURING THE RUN")
    for m in (NOTES or ["nothing"]):
        L.append(f"  - {m}")
    L.append("")
    L.append(f"Run took {(datetime.now() - started).total_seconds():.0f} seconds.")

    L.insert(5, "  Each of the five statements is measured SEPARATELY and written to its")
    L.insert(6, "  own CSV. A column's denominator is its own statement's issuer-period")
    L.insert(7, "  count for that frequency; the worst/best lists are ranked within a")
    L.insert(8, "  statement. No percentage in this report mixes two statements.")
    L.insert(9, "")

    txt_path = OUT_DIR / "stage1_5_china_coverage.txt"
    txt_path.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {txt_path}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
