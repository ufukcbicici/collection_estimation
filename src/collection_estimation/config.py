"""Paths, and everything that describes the SHAPE of the source workbooks.

This is the one file to edit when pointing the pipeline at a different set of files. No
other module hardcodes a path or a column name.

--------------------------------------------------------------------------------------
RUNNING ON REAL EY FILES
--------------------------------------------------------------------------------------

Set an environment variable and change nothing in the code:

    COLLECTIONS_CORPUS_DIR = D:\\ey\\tahsilat        (the folder of daily .xlsx files)
    COLLECTIONS_DATA_DIR   = D:\\ey\\work            (optional; where the cache is written)

Then run `preflight.py` against one real workbook before running the pipeline. It reads a
file and reports what it found against what the constants below declare, which turns a
`WorkbookFormatError` into a list of exactly what to change here.

See `HANDOVER.md`.
"""
from __future__ import annotations

import os

# Repository root: this file is at <root>/src/collection_estimation/config.py
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Where intermediates are written. Override if the repo lives on a drive you would rather
#: not fill, or if the real data must stay on a specific volume.
DATA = os.environ.get("COLLECTIONS_DATA_DIR", os.path.join(ROOT, "data"))
RAW = os.path.join(DATA, "raw")
INTERIM = os.path.join(DATA, "interim")

#: The default corpus: the synthetic workbooks, one per business day.
SYNTHETIC_CORPUS_DIR = os.path.join(RAW, "SYNTHETIC_ey_tahsilat_20250602_20260828")

#: The folder of daily workbooks the pipeline actually reads.
#:
#: **This is the drop-in point.** Point `COLLECTIONS_CORPUS_DIR` at the real EY files and
#: nothing else needs to change, provided they match the layout declared below.
CORPUS_DIR = os.environ.get("COLLECTIONS_CORPUS_DIR", SYNTHETIC_CORPUS_DIR)

#: True when running on the synthetic corpus. Used only to decide whether to print the
#: "this is fabricated data" banner and whether an oracle can be consulted.
IS_SYNTHETIC = os.path.abspath(CORPUS_DIR) == os.path.abspath(SYNTHETIC_CORPUS_DIR)

#: Ground truth describing the synthetic corpus. Does not exist for real data.
#: NEVER use its fields as model features — they exist only because the data is synthetic.
ORACLE = os.path.join(RAW, "collections_corpus.json")

#: Tidy collections table written by `ingest.py`, read by everything downstream. Derived
#: from the corpus folder name so that switching corpora cannot silently reuse the wrong
#: cache — the single nastiest failure mode when swapping data sets.
COLLECTIONS = os.path.join(
    INTERIM, f"collections__{os.path.basename(os.path.normpath(CORPUS_DIR))}.csv")

# ======================================================================================
# Workbook layout — edit these to match the real files
# ======================================================================================

#: Header is on row 2; row 1 is a merged title such as "MERCURY COLLECTIONS".
#: If the real files put the header on row 1, set these to 1 and 2.
HEADER_ROW = 2
FIRST_DATA_ROW = 3

#: How the filename encodes its date. `workbook_paths` parses the first 8 characters.
#: `02072026.xlsx` is 2 July 2026. Change if the real files use `YYYYMMDD` or a prefix.
FILENAME_DATE_FORMAT = "%d%m%Y"
FILENAME_DATE_LENGTH = 8

#: Header text -> the field name used everywhere downstream. Keys are compared with
#: whitespace collapsed and case folded, so "Company  Code " still matches.
#:
#: **Columns are bound by HEADER TEXT, never by position.** This is the single most
#: important decision in the ingest path. The corpus spans fourteen months of a
#: hand-touched daily extract, and a column inserted partway through would shift
#: everything to its right by one letter. A parser using `row[7]` for the amount would then
#: silently read the currency instead and keep running.
#:
#: If the real files use Turkish headers, ADD the Turkish spelling as another key mapping
#: to the same field. Do not remove the English one — a corpus that changes header language
#: partway through then still reads.
FIELD_BY_HEADER = {
    "company code": "company_code",
    "customer": "customer",
    "customer name": "customer_name",
    "document date": "document_date",
    "bank": "bank",
    "sap document no": "sap_document_no",
    "government inv": "government_invoice",
    "amount in original currency": "amount_original",
    "document currency": "document_currency",
    "amount in local currency": "amount_local",
    "local currency": "local_currency",
}

#: Fields the reader tolerates being absent. `bank` is optional because EY's imported
#: schema has a bank field but its position in the Excel layout was never observed — the
#: generator places it in hidden column E as an assumption, and a real file may not carry
#: it at all. Everything else is required and its absence raises.
OPTIONAL_FIELDS = frozenset({"bank"})

#: Tabs that are not the data sheet. The synthetic workbooks carry a marker tab; real EY
#: files have a single sheet and will not match.
MARKER_TABS = frozenset({"_synthetic"})

# ======================================================================================
# Business vocabulary — edit these to match the real files
# ======================================================================================

#: EY legal entities. A row is a DETAIL row if and only if its Company Code is one of
#: these; subtotal rows carry the label in the name column and leave the code empty, so
#: they exclude themselves.
#:
#: **If the real files carry entity codes not listed here, every row is skipped and the
#: ingest returns nothing.** `preflight.py` reports the codes it actually saw.
ENTITY_LABEL = {"TR02": "Güney", "TR04": "Eykf", "TR05": "Av.Ort",
                "TR06": "Bey", "TR91": "Kuzey"}

#: `collection_group` as it appears in EY's imported schema. Diagnostics only.
ENTITY_GROUP = {"TR02": "G", "TR04": "E", "TR05": "Ao", "TR06": "B", "TR91": "K"}

#: The invoice-number column carries one of these Turkish status notes instead of a number
#: on UNAPPLIED CASH rows: money received but not yet matched to a receivable. Such rows
#: are part of the day's total and must not be dropped. Used only to count them.
STATUS_NOTES = ("Ödeme Detayı Araştırılıyor", "Eğitim Ödemesi")

CURRENCIES = ("TRY", "EUR", "USD", "GBP")
