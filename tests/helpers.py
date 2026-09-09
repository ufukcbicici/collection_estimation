"""Shared test helpers: building tiny workbooks in the real layout.

Kept out of the test modules themselves because `tests/` is a package, so a bare sibling
import (`from test_ingest import ...`) does not resolve.
"""
import os
import sys
from datetime import datetime

from openpyxl import Workbook

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from collection_estimation import config  # noqa: E402

#: The header row of a real workbook, in order.
HEADERS = ["Company Code", "Customer", "Customer Name", "Document Date", "Bank",
           "SAP Document No", "Government Inv", "Amount in Original Currency",
           "Document currency", "Amount in local currency", "Local Currency"]


def corpus_files() -> list[str]:
    """Workbook names in the real corpus, lock files excluded."""
    return sorted(f for f in os.listdir(config.CORPUS_DIR)
                  if f.upper().endswith(".XLSX") and not f.startswith("~$"))


def build(tmp_path, headers=HEADERS, rows=(), title="MERCURY COLLECTIONS",
          sheet="02072026", name="book.xlsx", marker_tab=True) -> str:
    """A minimal workbook in the real layout: merged title, header on row 2, data below."""
    wb = Workbook()
    wb.remove(wb.active)
    if marker_tab:
        wb.create_sheet("_SYNTHETIC", 0).cell(row=1, column=1, value="synthetic")
    ws = wb.create_sheet(sheet)
    if title is not None:
        ws.cell(row=1, column=1, value=title)
    for i, h in enumerate(headers, start=1):
        ws.cell(row=config.HEADER_ROW, column=i, value=h)
    for r, values in enumerate(rows, start=config.FIRST_DATA_ROW):
        for i, v in enumerate(values, start=1):
            ws.cell(row=r, column=i, value=v)
    path = os.path.join(tmp_path, name)
    wb.save(path)
    return path


def detail(customer="11520039", code="TR02", amount=1000.0, cur="TRY",
           local=None, when=datetime(2026, 7, 2), bank="hsbc",
           sap="TR20021232", inv="GNF2026000002400") -> list:
    """One detail row, in header order."""
    return [code, customer, "A CUSTOMER A.Ş.", when, bank, sap, inv,
            amount, cur, amount if local is None else local, "TRY"]
