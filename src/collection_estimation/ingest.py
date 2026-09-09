"""Reading the daily EY TAHSILAT workbooks into rows.

Sits below the four algorithm boxes of the design document: Algorithm 1 takes "collection
rows" as its input and does not say where they come from. This is where they come from.

See `daily_collections_corpus.md` in the cockpit repo for what the files look like.
"""
from __future__ import annotations

import os
from datetime import date, datetime

import pandas as pd
from openpyxl import load_workbook

from .config import CORPUS_DIR, ENTITY_LABEL, FIRST_DATA_ROW, HEADER_ROW


class WorkbookFormatError(ValueError):
    """A workbook does not have the shape we require.

    Raised rather than skipped. With a corpus of uniform files a malformed one means
    something is genuinely wrong, and a parser that quietly drops a file is the exact
    failure mode the corpus notes warn about — it fails silently and the loss shows up
    later as an unexplained dip in a weekly total.
    """


#: Tabs that are not the data sheet. Our synthetic workbooks carry a marker tab first;
#: real EY files have a single sheet and will not match this.
MARKER_TABS = frozenset({"_synthetic"})

#: Header text -> the field name we use. Keys are normalised (see `_normalise`).
#:
#: Columns are bound by HEADER TEXT, never by position. This is the single most important
#: decision in this module. The corpus spans fourteen months of a hand-touched daily
#: extract, and a column inserted partway through would shift everything to its right by
#: one letter. A parser using `row[7]` for the amount would then silently read the currency
#: instead and keep running.
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

#: `bank` is deliberately NOT required. EY's imported schema has a bank field, but its
#: position in the Excel layout was never observed — our generator places it in the hidden
#: column E as an assumption. A real file may not carry it at all.
OPTIONAL_FIELDS = frozenset({"bank"})
REQUIRED_FIELDS = frozenset(FIELD_BY_HEADER.values()) - OPTIONAL_FIELDS

#: Fields that must hold a number on every detail row. Zero is legitimate — EY's TR91 TRY
#: bucket has a minimum of 0 — and so is a negative, which is a refund or a reversal.
NUMERIC_FIELDS = ("amount_original", "amount_local")


def _normalise(value: object) -> str:
    """Fold a header cell to a comparison key.

    Collapses whitespace and case, so a header that gains a trailing space or is retyped
    in different case still matches. That is cheap insurance against exactly the kind of
    one-off anomaly a hand-maintained daily file accumulates.
    """
    if value is None:
        return ""
    return " ".join(str(value).split()).casefold()


def _pick_data_sheet(workbook, path: str) -> str:
    """Choose the sheet holding the collections grid.

    Prefers a sheet named `DDMMYYYY`, which is how the real files name theirs. Falls back
    to the first sheet that is not a known marker tab, so this still works on a real EY
    file whose single sheet is named something else.
    """
    for name in workbook.sheetnames:
        stripped = name.strip()
        if len(stripped) == 8 and stripped.isdigit():
            return name
    for name in workbook.sheetnames:
        if _normalise(name) not in MARKER_TABS:
            return name
    raise WorkbookFormatError(
        f"{os.path.basename(path)}: no data sheet — only {workbook.sheetnames}")


def _header_map(worksheet, path: str) -> dict[str, int]:
    """Map field name -> zero-based column index, from the header row.

    Raises if a required header is missing, which is how a shifted or reworded layout
    announces itself here instead of corrupting every amount downstream.
    """
    header_row = next(
        worksheet.iter_rows(min_row=HEADER_ROW, max_row=HEADER_ROW, values_only=True),
        None,
    )
    if header_row is None:
        raise WorkbookFormatError(
            f"{os.path.basename(path)}: no header row at row {HEADER_ROW}")

    mapping: dict[str, int] = {}
    for index, cell in enumerate(header_row):
        field = FIELD_BY_HEADER.get(_normalise(cell))
        if field is not None and field not in mapping:
            mapping[field] = index

    missing = REQUIRED_FIELDS - set(mapping)
    if missing:
        seen = [_normalise(c) for c in header_row if _normalise(c)]
        raise WorkbookFormatError(
            f"{os.path.basename(path)}: missing required column(s) "
            f"{sorted(missing)}; header row reads {seen}")
    return mapping


def _coerce_date(value: object, path: str, excel_row: int) -> date:
    """A real date, with any time component dropped.

    openpyxl hands back `datetime` for a date-formatted cell. The time is always midnight
    here, and carrying it forward makes every later grouping and join depend on a
    component that means nothing.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise WorkbookFormatError(
        f"{os.path.basename(path)} row {excel_row}: Document Date is "
        f"{value!r} ({type(value).__name__}), not a date")


def read_workbook(path: str) -> list[dict]:
    """Parse one daily workbook into its detail rows.

    Returns one dict per detail row, in sheet order. Subtotal rows, the grand total and
    blank spacer rows are excluded; nothing else is.

    **Nothing is filtered on business grounds.** Unapplied-cash rows (a Turkish status note
    in place of an invoice number, and no SAP document number) and negative rows (refunds
    and reversals) come back like any other row. They are part of the day's money and
    dropping them here would understate every total that follows.

    **Nothing is derived.** No week number, no collection group, no unapplied flag. Those
    belong to the panel step; computing them here would mean two places to change.

    Raises `WorkbookFormatError` if the sheet cannot be found, a required column is
    missing, or a detail row holds a value of the wrong type.
    """
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[_pick_data_sheet(workbook, path)]
        columns = _header_map(worksheet, path)
        source_file = os.path.basename(path)

        rows: list[dict] = []
        for offset, values in enumerate(
                worksheet.iter_rows(min_row=FIRST_DATA_ROW, values_only=True)):
            excel_row = FIRST_DATA_ROW + offset

            def cell(field: str):
                index = columns.get(field)
                if index is None or index >= len(values):
                    return None
                return values[index]

            # A detail row is one carrying a company code. Subtotal rows put their label
            # in the customer-name column and leave the code empty, so they exclude
            # themselves; so do blank spacers and any stray note under the grand total.
            code = cell("company_code")
            if not isinstance(code, str) or code.strip() not in ENTITY_LABEL:
                continue

            customer = cell("customer")
            if not isinstance(customer, str):
                # Customer numbers must stay text. Read as a number they lose any leading
                # zero, and every join against them silently misses those customers.
                raise WorkbookFormatError(
                    f"{source_file} row {excel_row}: Customer is "
                    f"{customer!r} ({type(customer).__name__}), not text — the cell is "
                    f"stored as a number and its leading zeros are already lost")

            record = {
                "company_code": code.strip(),
                "customer": customer.strip(),
                "customer_name": (cell("customer_name") or "").strip(),
                "document_date": _coerce_date(
                    cell("document_date"), path, excel_row),
                "bank": cell("bank"),
                "sap_document_no": cell("sap_document_no"),
                "government_invoice": cell("government_invoice"),
                "document_currency": cell("document_currency"),
                "local_currency": cell("local_currency"),
                "source_file": source_file,
                "source_row": excel_row,
            }

            for field in NUMERIC_FIELDS:
                raw = cell(field)
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise WorkbookFormatError(
                        f"{source_file} row {excel_row}: {field} is {raw!r} "
                        f"({type(raw).__name__}), not a number")
                record[field] = float(raw)

            rows.append(record)
        return rows
    finally:
        workbook.close()


# --------------------------------------------------------------------------------------
# The whole corpus
# --------------------------------------------------------------------------------------

#: Column dtypes for the merged table. `customer` is the one that matters: as an inferred
#: integer it loses any leading zero and every join against those customers then misses
#: them, silently. See `read_collections_csv`.
#: Second precision, pinned. A fresh read infers `datetime64[s]` from `date` objects while
#: the CSV cache comes back as `datetime64[us]`, so without this the table's dtype depends
#: on whether the cache was hit — and code downstream would behave differently on a first
#: run than on every run after it. These are dates; sub-second precision means nothing.
DATE_DTYPE = "datetime64[s]"

DTYPES = {
    "company_code": "string",
    "customer": "string",
    "customer_name": "string",
    "bank": "string",
    "sap_document_no": "string",
    "government_invoice": "string",
    "document_currency": "string",
    "local_currency": "string",
    "source_file": "string",
    "source_row": "int64",
    "amount_original": "float64",
    "amount_local": "float64",
}


def workbook_paths(directory: str = CORPUS_DIR) -> list[str]:
    """Every workbook in a corpus directory, in CHRONOLOGICAL order.

    Two details, both of which have bitten this project already:

    * The filenames are `DDMMYYYY`, so a plain lexical sort puts `01072026` before
      `02062025` — a year out of order. The date is parsed out and sorted on.
    * Excel writes a `~$…` lock file whenever a workbook is open. It is not a valid
      archive and `load_workbook` raises a PermissionError on it.
    """
    dated: list[tuple[date, str]] = []
    for name in os.listdir(directory):
        if not name.upper().endswith(".XLSX") or name.startswith("~$"):
            continue
        try:
            when = datetime.strptime(name[:8], "%d%m%Y").date()
        except ValueError as exc:
            raise WorkbookFormatError(
                f"{name}: filename does not start with a DDMMYYYY date") from exc
        dated.append((when, os.path.join(directory, name)))
    return [path for _, path in sorted(dated)]


def read_corpus(directory: str = CORPUS_DIR, *, progress: bool = False) -> pd.DataFrame:
    """Read every workbook in a corpus directory into one table.

    One row per collection, in chronological order, with `source_file` and `source_row`
    kept so any row can be traced back to the cell it came from.

    **Cross-checks the date**, which `read_workbook` deliberately does not: a file named
    `02072026` whose rows carry a different document date is an anomaly, and consistent
    with this module's strict stance it raises rather than being quietly accepted.
    """
    paths = workbook_paths(directory)
    if not paths:
        raise WorkbookFormatError(f"no workbooks found in {directory}")

    frames: list[dict] = []
    for index, path in enumerate(paths, start=1):
        name = os.path.basename(path)
        if progress and index % 50 == 0:
            print(f"  ... {index}/{len(paths)} workbooks")

        rows = read_workbook(path)
        expected = datetime.strptime(name[:8], "%d%m%Y").date()
        wrong = {r["document_date"] for r in rows} - {expected}
        if wrong:
            raise WorkbookFormatError(
                f"{name}: filename says {expected} but rows carry "
                f"{sorted(wrong)} in Document Date")
        frames.extend(rows)

    table = pd.DataFrame(frames)
    table["document_date"] = pd.to_datetime(
        table["document_date"]).astype(DATE_DTYPE)
    for column, dtype in DTYPES.items():
        if column in table.columns:
            table[column] = table[column].astype(dtype)
    return table.sort_values(["document_date", "company_code", "customer"],
                             kind="stable").reset_index(drop=True)


def write_collections_csv(table: pd.DataFrame, path: str) -> None:
    """Cache the merged table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8")


def read_collections_csv(path: str) -> pd.DataFrame:
    """Load the cached table, preserving the types the CSV cannot carry itself.

    **The explicit dtype is load-bearing.** `customer` holds digits, so pandas infers
    `int64` for it by default and any leading zero is gone — after which every join
    against those customers misses them and nothing complains. CSV has no dtype of its
    own, so this is the only place the distinction can be restored.
    """
    present = pd.read_csv(path, nrows=0, encoding="utf-8").columns
    table = pd.read_csv(
        path,
        dtype={col: dt for col, dt in DTYPES.items() if col in present},
        parse_dates=["document_date"] if "document_date" in present else None,
        encoding="utf-8",
    )
    if "document_date" in present:
        table["document_date"] = table["document_date"].astype(DATE_DTYPE)
    return table
