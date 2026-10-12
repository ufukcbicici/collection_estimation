"""Builders for small daily workbooks, shared by the hurdle reader tests.

`make_workbook` writes a workbook in the layout of the daily files: an optional marker sheet
`_SYNTHETIC`, one data sheet, a title on row 1, the header on row 2, payment lines from row 3,
and a grand-total row last. Customers are fake.
"""
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook

HEADER = ('Company Code', 'Customer', 'Customer Name', 'Document Date', 'Bank',
          'SAP Document No', 'Government Inv', 'Amount in Original Currency',
          'Document currency', 'Amount in local currency', 'Local Currency')

# Column positions in HEADER, for building subtotal and grand-total rows.
NAME_COLUMN = HEADER.index('Customer Name')
AMOUNT_TRY_COLUMN = HEADER.index('Amount in local currency')
LOCAL_CURRENCY_COLUMN = HEADER.index('Local Currency')


def line(customer='0012345', amount_try=100.0, *, company='TR02', name='ACME TEST A.S.',
         when=datetime(2025, 6, 2), bank='TEST BANK', sap='1400000001', invoice='GIB2025000001',
         amount_original=None, currency='TRY'):
    """One payment line as a tuple in HEADER order."""
    if amount_original is None:
        amount_original = amount_try
    return (company, customer, name, when, bank, sap, invoice, amount_original, currency,
            amount_try, 'TRY')


def total_row(label, amount, width=len(HEADER)):
    """A subtotal or grand-total row: no company code or customer, a label and a TRY amount."""
    row = [None] * width
    row[NAME_COLUMN] = label
    row[AMOUNT_TRY_COLUMN] = amount
    row[LOCAL_CURRENCY_COLUMN] = 'TRY'
    return tuple(row)


GRAND_TOTAL_DEFAULT = object()  # "use the sum of the lines"


def make_workbook(folder, lines, grand_total=GRAND_TOTAL_DEFAULT, *, marker=True,
                  extra_sheets=(), last_rows=None, header=HEADER, name='02062025'):
    """Write a daily workbook to `folder` and return its path.

    `lines` are rows written from row 3 (payment lines, subtotal rows, or None for a blank
    spacer row). A grand-total row follows, `grand_total` defaulting to the sum of the payment
    lines' TRY amounts; `last_rows` replaces that grand-total row with rows of your choice.
    """
    workbook = Workbook()
    first = workbook.active
    if marker:
        first.title = '_SYNTHETIC'
        sheet = workbook.create_sheet(name)
    else:
        first.title = name
        sheet = first
    for extra in extra_sheets:
        workbook.create_sheet(extra)

    sheet.append(['MERCURY COLLECTIONS'])
    sheet.append(list(header))
    for row in lines:
        if row is None:
            sheet.append([])  # openpyxl skips empty appends' cells; the row number still moves
        else:
            sheet.append(list(row))

    if last_rows is None:
        if grand_total is GRAND_TOTAL_DEFAULT:
            grand_total = sum(row[AMOUNT_TRY_COLUMN] for row in lines
                              if row is not None and row[0] is not None)
        last_rows = [total_row('GRAND TOTAL', grand_total)]
    for row in last_rows:
        sheet.append(list(row))

    path = Path(folder) / f'{name} EY TAHSILAT.XLSX'
    workbook.save(path)
    return path
