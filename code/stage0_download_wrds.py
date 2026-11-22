# ABOUTME: Stage 0 - downloads every raw table we need from WRDS into data_raw/us/
# ABOUTME: Compustat, company and CRSP identifiers, and 8 FISD bond tables via WRDS
"""
stage0_download_wrds.py

What this does, in plain language:
    Connects to WRDS and downloads 13 files into data_raw/us/ as parquet files
    with the current DOWNLOAD_CRSP = True setting.
    It downloads and saves ONLY - no cleaning, no merging, no feature building.
    Those happen in later stages, so this raw layer stays a faithful copy of
    what WRDS gave us.

    Sources:
      1. Compustat annual  (comp.funda)   - the accounting items our features need
      2. Compustat quarterly (comp.fundq) - 13 features need quarterly items
      3. Compustat company master (comp.names) - identifiers and classifications
      4. Two CRSP identifier tables for the fallback issuer-linking route
      5. Eight FISD bond tables, each downloaded in full

What is and is not restricted, so the raw layer's coverage is never in doubt:
      - No date filter anywhere. Full sample period, every year WRDS holds.
      - No row filter anywhere, including on Compustat. See the note above
        FUNDA_KEYS: this means comp.funda contains roughly 1.55 rows per
        firm-year and Stage 1 MUST de-duplicate it.
      - Columns ARE restricted, but only on comp.funda and comp.fundq: 86 of
        funda's 949 and 36 of fundq's 648. Share prices are excluded on purpose
        (hard rule 1). comp.names and the 8 FISD tables are SELECT *.
      - No values are altered. Column TYPES are pinned to what Postgres
        declares - see column_types() - which changes no data.

Logging in:
    Set WRDS_USERNAME below to your WRDS username. The password is never stored
    in this project: the first run asks for it once and saves ~/.pgpass in your
    home directory, and every run after that is silent.

How to run:
    Test first, then the real thing:
        MAX_ROWS = 1000   ->  python3 code/stage0_download_wrds.py
        MAX_ROWS = None   ->  python3 code/stage0_download_wrds.py

Inputs:
    - WRDS access; the FISD table list is defined below

Outputs (data_raw/us/) - 13 files with DOWNLOAD_CRSP = True:
    compustat_funda.parquet, compustat_fundq.parquet, comp_names.parquet,
    fisd_mergedissue.parquet, fisd_mergedissuer.parquet, fisd_issue_default.parquet,
    fisd_bankruptcy.parquet, fisd_issuer_cusip.parquet, fisd_code.parquet,
    fisd_ratings.parquet, fisd_issue_enhancement.parquet,
    ccm_link.parquet, crsp_stocknames.parquet
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import wrds

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

# Your WRDS username. This is NOT a secret - the password never appears in this
# project, it lives in ~/.pgpass in your home folder.
#
# It has to be set here. wrds.Connection() with no username builds a connection
# string with an empty user, so Postgres falls back to your local login name.
# That may not match the username stored in .pgpass, the lookup
# fails, and it asks for your password on every single run.
WRDS_USERNAME = "YOUR_WRDS_USERNAME"

# Row cap per table. Set to 1000 for a quick test run, None for the real
# download. Start with 1000 and check the output before setting this to None.
MAX_ROWS = None

# How many rows to pull from the server at a time. Two of the FISD tables would
# not comfortably fit in memory in one piece, so every table is streamed in
# chunks and written to parquet as it arrives. Memory stays at roughly one chunk.
#
# 50,000 rather than 100,000 because fisd_mergedissue has 224 columns, 180 of
# them text. At 100,000 rows one chunk was estimated at around 1 GB of RAM,
# which is uncomfortably close to this laptop's budget. Halving it costs only a
# few extra writes.
CHUNK_SIZE = 50_000

# Full sample period - no date filter anywhere in this script, by decision.
# Compustat goes back to the 1960s, FISD to 1984.

# Whether to download the two CRSP identifier tables (ccm_link, crsp_stocknames).
#
# OFF by default. They are only needed for linking route B - FISD issuer_cusip
# -> CRSP historical cusip -> permno -> gvkey - which is the fallback for when
# the direct cusip match fails. We do route A first, measure the match rate, and
# only come back for these if route A is not good enough.
#
# Neither table contributes any market data. They are pure identifier tables,
# and nothing from them may enter the feature set (hard rule 1).
#
# Switched ON 2026-08-15. Route A alone matches only 42.0% of the 2,446 FISD
# issuers that actually defaulted, well under the ~70% originally hoped for, so
# route B has to be measured rather than assumed. Compustat's `cusip` is the
# CURRENT header cusip, so any issuer that changed cusip - which distressed
# issuers often do - fails the direct match. crsp.stocknames keeps the
# historical cusips with their validity dates, which is exactly that gap.
DOWNLOAD_CRSP = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = PROJECT_ROOT / "data_raw" / "us"

# ---------------------------------------------------------------------------
# COMPUSTAT ANNUAL - the 65 accounting items our 101 features need.
#
# Derived from output/stage0_1_required_variables.csv, plus xint and ebitda,
# which are not needed by any CZ feature but are required for the book leverage
# ratios (debt/EBITDA, interest coverage) we may build ourselves in Stage 1.
#
# NOTE: prcc_c and prcc_f are deliberately absent - hard rule 1, no share prices.
# ---------------------------------------------------------------------------
FUNDA_ITEMS = """
    aco act ao ap at capx ceq che cogs cshrc dcpstk dcvt dlc dlcch dltis dltr
    dltt dp drc dv dvt ebitda epspx fatb fatl fincf fopt ib intan invt ivao
    ivncf ivst lco lct lo lt mib ni oancf ob ppegt ppenb ppenls ppent prstkc
    prstkcc pstk pstkl pstkrv rect revt sale seq sstk txdi txditc txfed txfo
    txp txt xad xint xrd xsga
    ajex dm dvpsx_c ebit emp gdwl nopi oiadp
""".split()

# ---------------------------------------------------------------------------
# THE LAST LINE ABOVE - 8 items added 2026-08-16, and why they were missing.
#
# The original list came from output/stage0_1_required_variables.csv, which was
# built by parsing `columns=[...]` out of CZ's feature scripts. That method was
# incomplete: CZ's repo mixes two dataframe libraries. Of its 94 scripts, 45 use
# pandas `columns=[...]`, 42 use polars `.select([...])`, and 19 read the whole
# table and select later. Only the first group was visible, so **54 of our 101
# features had their variable requirements missed entirely**.
#
# Re-derived with three cross-checked methods - selection statements, script
# bodies with locally-created names excluded, and SignalDoc.csv's prose
# definitions - which converge on the same answer. Validated by scanning against
# CZ's own download SQL: zero referenced variables fall outside it.
#
#   ajex     share adjustment factor  -> EarningsPersistence, EarningsPredictability
#                                        (epspx/ajex; SignalDoc states this, my
#                                        first scan just failed to parse it)
#   dm       mortgage / secured debt  -> secured, securedind
#   dvpsx_c  dividends per share      -> WW (used as a dividend-payer indicator)
#   ebit     earnings before int. tax -> roic
#   emp      employees                -> LaborforceEfficiency
#   gdwl     goodwill                 -> AssetLiquidityBook
#   nopi     non-operating income     -> roic
#   oiadp    operating income after D -> RetNOA
#
# Note: dm, gdwl and nopi are also on CZ's 30-item zero-fill list, so Stage 1.0
# will start zero-filling them automatically - apply_zero_fill() intersects the
# list with the columns actually present.
# ---------------------------------------------------------------------------

# Identifiers and classification fields we need alongside the accounting items.
#
# The last five (consol, popsrc, datafmt, curcd, indfmt) are Compustat's format
# flags. They are NOT accounting data and never become features - they are here
# because this script no longer filters rows, and without them the duplicate
# rows described below could not be told apart.
FUNDA_KEYS = ["gvkey", "datadate", "fyear", "cusip", "conm", "tic", "sich", "naicsh",
              "consol", "popsrc", "datafmt", "curcd", "indfmt"]

# Compustat quarterly items, needed by 13 of the 101 features.
# Same five format flags, except the currency column is curcdq here, not curcd.
# The last 16 were added 2026-08-16 for the same reason as the annual block
# above. Six quarterly features could not be built without them:
#   OperProfLag_q           ceqq cogsq ltq pstkq seqq txditcq xintq xsgaq
#   AssetLiquidityBookQuart actq gdwlq intanq
#   WW_Q                    dlttq dpq dvpsxq saleq
#   CapTurnover_q           saleq
#   PM_q                    niq
# Note: actq, dpq, gdwlq, intanq and txditcq are on CZ's quarterly zero-fill
# list, so Stage 1.0 will begin zero-filling them automatically.
FUNDQ_ITEMS = """
    atq cheq cshprq epspxq ibq rdq revtq txtq
    actq ceqq cogsq dlttq dpq dvpsxq gdwlq intanq
    ltq niq pstkq saleq seqq txditcq xintq xsgaq
""".split()
FUNDQ_KEYS = ["gvkey", "datadate", "fyearq", "fqtr", "datacqtr", "datafqtr", "cusip",
              "consol", "popsrc", "datafmt", "curcdq", "indfmt"]

# NO ROW FILTER ON COMPUSTAT, by decision (2026-08-15).
#
# CZ filters to consol='C', popsrc='D', datafmt='STD', curcd='USD', indfmt='INDL'.
# We deliberately do not, because this raw layer should be a faithful copy and
# because one of those conditions is questionable for us: indfmt='INDL' keeps
# the industrial reporting format and drops the financial-services format, a
# habit inherited from equity asset-pricing work. Financial firms are a large
# share of bond ISSUERS, which is our population, so that choice belongs to
# Stage 1 where it can be measured, not baked in here where it is invisible.
#
# The cost, measured on WRDS before making this change:
#   comp.funda   941,735 rows unfiltered vs 528,515 filtered
#                but only 606,366 distinct gvkey x datadate
#                -> 1.55 rows per firm-year. THE PANEL HAS DUPLICATES.
#   comp.fundq 2,133,913 rows unfiltered vs 1,897,146 filtered
#                -> 1.00 rows per firm-quarter, essentially no duplication.
#
# So Stage 1 MUST de-duplicate comp.funda on gvkey x datadate using the five
# format flags above before building any feature. This is not optional.

# ---------------------------------------------------------------------------
# FISD - 8 tables, downloaded in full. Schema name and table list come from
# the WRDS data dictionary and is defined directly below.
# No column selection and no row filters, by decision.
# ---------------------------------------------------------------------------
FISD_SCHEMA = "fisd_fisd"
FISD_TABLES = [
    "fisd_mergedissue",       # ~773k rows, ~297 MiB - the big one
    "fisd_mergedissuer",      # ~17k rows
    "fisd_issue_default",     # ~5.8k rows - default_date, the event we predict
    "fisd_bankruptcy",        # ~2.5k rows - filing_date
    "fisd_issuer_cusip",      # ~23k rows - issuer_id to 6-digit CUSIP
    "fisd_code",              # ~1.1k rows - code lookup
    "fisd_ratings",           # ~4.7M rows, ~251 MiB - diagnostics ONLY, never a label
    "fisd_issue_enhancement",  # ~187k rows
]


# ---------------------------------------------------------------------------
# COLUMN TYPES
#
# The problem this solves: pandas decides a column's type by looking at the rows
# it can see in the current chunk. A column that happens to be entirely empty in
# chunk 1 but full of numbers in chunk 30 therefore gets two different types.
# The parquet writer locks its schema on chunk 1 and refuses the later chunk, so
# the download dies partway through a large table.
#
# The fix is to stop letting pandas guess: ask Postgres what each column is
# declared as, and force every chunk into that one type. Schema drift then
# becomes impossible rather than merely unlikely.
#
# Across our 10 tables Postgres uses exactly six types (checked on WRDS):
#   numeric 1515, character varying 226, double precision 68,
#   date 49, integer 14, smallint 4.
# ---------------------------------------------------------------------------
PG_TO_PANDAS = {
    "numeric": "Float64",
    "double precision": "Float64",
    "real": "Float64",
    "integer": "Int64",
    "smallint": "Int64",
    "bigint": "Int64",
    "character varying": "string",
    "character": "string",
    "text": "string",
    # Dates stay TEXT on purpose. FISD records perpetual and very long-dated
    # bonds with sentinel maturities, and pandas' datetime64 only spans
    # 1677-2262, so converting here would either crash or quietly drop rows.
    # Stage 1 converts with pd.to_datetime(errors='coerce') and reports failures.
    "date": "string",
    "timestamp without time zone": "string",
}


def column_types(db, sources):
    """
    Return {column_name: postgres_type} for every column in the given tables.

    `sources` is a list of (schema, table) pairs - a list, not one table,
    because the CRSP query joins two tables and needs both.
    """
    types = {}
    for schema, table in sources:
        info = db.raw_sql(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_schema = '{schema}' AND table_name = '{table}'"
        )
        types.update(dict(zip(info["column_name"], info["data_type"])))
    return types


def preflight(db):
    """
    Check that every table we intend to download actually exists BEFORE we start.
    If a name is wrong we want to know immediately, not after a 20-minute
    download of everything else.

    On failure this prints similarly-named tables in the same schema, so the
    right name is easy to spot. It does NOT pick a substitute on its own.
    """
    print("\nChecking every table exists before downloading...")

    required = [("comp", "funda"), ("comp", "fundq"), ("comp", "names")]
    if DOWNLOAD_CRSP:
        required += [("crsp", "ccmxpf_lnkhist"), ("crsp", "stocknames")]
    required += [(FISD_SCHEMA, t) for t in FISD_TABLES]

    # Ask WRDS what each schema actually contains.
    contents = {}
    for schema in sorted({s for s, _ in required}):
        try:
            contents[schema] = set(db.list_tables(library=schema))
        except Exception as err:
            sys.exit(f"ERROR: could not list schema '{schema}': {err}")

    missing = [(s, t) for s, t in required if t not in contents.get(s, set())]
    if missing:
        print("\nERROR: these tables were not found:")
        for schema, table in missing:
            print(f"  {schema}.{table}")
            stem = table.split("_")[0][:5]
            near = sorted(x for x in contents.get(schema, set()) if stem in x)[:12]
            print(f"      similar names in '{schema}': {near or 'none'}")
        sys.exit("\nStopping - nothing was downloaded.")

    print(f"  all {len(required)} tables found")


def download(db, sql, outfile, label, sources):
    """
    Run one query and stream the result straight into a parquet file.

    We write chunk by chunk instead of building one big DataFrame, so memory
    stays at roughly one chunk regardless of how large the table is.

    `sources` is the list of (schema, table) pairs the query reads, used to look
    up the declared column types - see column_types() for why that matters.
    """
    if outfile.exists():
        # data_raw/ is read-only apart from new files, so never overwrite.
        print(f"  SKIP {label}: {outfile.name} already exists")
        return

    print(f"  downloading {label} ...", flush=True)

    pg_types = column_types(db, sources)

    # Write to a temporary name first, then rename. If the download dies halfway
    # we are left with a .partial file, not a truncated file that looks complete.
    temp_path = outfile.with_suffix(".parquet.partial")
    writer = None
    n_rows = 0

    try:
        chunks = db.raw_sql(sql, chunksize=CHUNK_SIZE, return_iter=True)
        for chunk in chunks:
            # Force every column to the type Postgres declares for it, so every
            # chunk has an identical schema and the parquet writer cannot
            # reject a later one. See column_types() above.
            for col in chunk.columns:
                pg_type = pg_types.get(col)
                if pg_type not in PG_TO_PANDAS:
                    sys.exit(
                        f"ERROR: {label} column '{col}' has Postgres type "
                        f"'{pg_type}', which this script does not know how to "
                        f"handle. Add it to PG_TO_PANDAS. Stopping."
                    )
                want = PG_TO_PANDAS[pg_type]
                # Skip columns already of the right type - astype copies, and
                # these tables are wide.
                if str(chunk[col].dtype) != want:
                    chunk[col] = chunk[col].astype(want)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema)
            writer.write_table(table)

            n_rows += len(chunk)
            print(f"    {n_rows:,} rows", end="\r", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if n_rows == 0:
        # An empty result is never expected here - report it rather than
        # leaving a valid-looking empty file behind.
        temp_path.unlink(missing_ok=True)
        sys.exit(f"ERROR: {label} returned 0 rows. Stopping.")

    temp_path.rename(outfile)
    size_mb = outfile.stat().st_size / 1024 / 1024
    print(f"    {label}: {n_rows:,} rows -> {outfile.name} ({size_mb:.1f} MB)")


def limit_clause():
    """Row cap for test runs. Empty string on a full download."""
    return f" LIMIT {MAX_ROWS}" if MAX_ROWS else ""


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if MAX_ROWS:
        print(f"TEST MODE - capping every table at {MAX_ROWS:,} rows")
        print("Set MAX_ROWS = None for the real download.")
    else:
        print("FULL DOWNLOAD - no row cap.")
        print("Expect roughly 600 MB and 10-30 minutes depending on connection.")

    # Passing the username explicitly lets Postgres find the matching line in
    # ~/.pgpass, so no password prompt. Only the very first run (before .pgpass
    # exists) asks for a password.
    print("\nConnecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    preflight(db)

    # ---- 1. Compustat annual ------------------------------------------------
    # Selected columns, but every row - see the FUNDA_KEYS block for why.
    print("\nCompustat")
    cols = ", ".join(FUNDA_KEYS + FUNDA_ITEMS)
    download(
        db,
        f"SELECT {cols} FROM comp.funda{limit_clause()}",
        OUTDIR / "compustat_funda.parquet",
        "comp.funda",
        [("comp", "funda")],
    )

    # ---- 2. Compustat quarterly --------------------------------------------
    cols = ", ".join(FUNDQ_KEYS + FUNDQ_ITEMS)
    download(
        db,
        f"SELECT {cols} FROM comp.fundq{limit_clause()}",
        OUTDIR / "compustat_fundq.parquet",
        "comp.fundq",
        [("comp", "fundq")],
    )

    # ---- 3. Compustat company master (comp.names) ---------------------------
    #
    # Why this table is needed: six of the 101 features are industry-adjusted.
    # CZ takes their industry code from CRSP (`sicCRSP`), which we cannot use -
    # CRSP is a database of LISTED SECURITIES, so an unlisted issuer has no
    # record in it at all, and using it would push those six features back onto
    # the listed-only sample we spent Stage 0 avoiding.
    #
    # The alternative is Compustat's own SIC. We already download `sich` (the
    # historical, year-by-year code in funda) but it covers only 64% of
    # firm-years after 1982 and falls to ~46% by 2025. comp.names carries `sic`,
    # a static header code that is 100% populated, and the two agree on 91.2% of
    # firms at the two-digit level - they are both Compustat, unlike CRSP's,
    # which agrees with `sich` only ~59% of the time.
    #
    # So Stage 1 uses `sich` where present and falls back to this `sic`,
    # recording which source each row used.
    #
    # NOT joined to the CCM link table here. CZ joins the two, but that join is
    # an inner join to CRSP and would hand us the listed-only subset again -
    # which is precisely the problem this table is meant to solve.
    #
    # SELECT * rather than a column list: the table is small, purely
    # identifiers and classifications, and taking it whole avoids having to
    # re-download when a later stage wants one more field.
    print("\nCompustat company master")
    download(
        db,
        f"SELECT * FROM comp.names{limit_clause()}",
        OUTDIR / "comp_names.parquet",
        "comp.names",
        [("comp", "names")],
    )

    # ---- 4. CRSP identifier tables (controlled by DOWNLOAD_CRSP) -----------
    # Only needed for linking route B. See the DOWNLOAD_CRSP note at the top.
    if DOWNLOAD_CRSP:
        ccm_sql = f"""
            SELECT a.gvkey, a.conm, a.tic, a.cusip, a.cik, a.sic, a.naics,
                   b.linkprim, b.linktype, b.liid, b.lpermno, b.lpermco,
                   b.linkdt, b.linkenddt
            FROM comp.names AS a
            INNER JOIN crsp.ccmxpf_lnkhist AS b ON a.gvkey = b.gvkey
            WHERE b.linktype IN ('LC', 'LU') AND b.linkprim IN ('P', 'C')
            ORDER BY a.gvkey{limit_clause()}
        """
        # Two sources because this query joins them; both sets of column types
        # are needed to type the result.
        download(db, ccm_sql, OUTDIR / "ccm_link.parquet", "CRSP-Compustat link",
                 [("comp", "names"), ("crsp", "ccmxpf_lnkhist")])

        # crsp.stocknames keeps every HISTORICAL cusip (ncusip) with the date
        # range it was valid for, plus the permno. That gives us:
        #     FISD issuer_cusip -> ncusip -> permno -> ccm_link -> gvkey
        # which recovers issuers whose cusip changed and so fail the direct match.
        stocknames_sql = f"""
            SELECT permno, permco, namedt, nameenddt, cusip, ncusip,
                   ticker, comnam, siccd, shrcd, exchcd
            FROM crsp.stocknames
            ORDER BY permno, namedt{limit_clause()}
        """
        download(db, stocknames_sql, OUTDIR / "crsp_stocknames.parquet",
                 "CRSP historical cusips", [("crsp", "stocknames")])
    else:
        print("\nCRSP: skipped (DOWNLOAD_CRSP = False)")
        print("  Linking route A only for now - direct FISD cusip to Compustat cusip.")
        print("  Set DOWNLOAD_CRSP = True if route A's match rate turns out too low.")

    # ---- 5. FISD, all 8 tables in full -------------------------------------
    print("\nFISD")
    for table in FISD_TABLES:
        download(
            db,
            f"SELECT * FROM {FISD_SCHEMA}.{table}{limit_clause()}",
            OUTDIR / f"{table}.parquet",
            f"{FISD_SCHEMA}.{table}",
            [(FISD_SCHEMA, table)],
        )

    db.close()

    print("\nDone. Files in data_raw/us/:")
    for f in sorted(OUTDIR.glob("*.parquet")):
        print(f"  {f.name:<34} {f.stat().st_size / 1024 / 1024:>8.1f} MB")

    if MAX_ROWS:
        print(
            "\nThis was a TEST run - every table is capped and INCOMPLETE.\n"
            "Delete data_raw/us/*.parquet, set MAX_ROWS = None, and run again."
        )


if __name__ == "__main__":
    main()
