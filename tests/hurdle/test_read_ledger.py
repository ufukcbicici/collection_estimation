"""Tests for `hurdle/read_ledger.py`: reading the AR ledger (FBL5N layout) and its checks.

Every workbook is built in `tmp_path` by `make_ledger_workbook` (see `workbooks.py`).
Customers are fake. The integration tests at the end read the synthetic ledger in
`data/raw/` once per session (~30 s) and are skipped when it is absent.
"""
import dataclasses
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.read_ledger import (
    DOCUMENT_ROLES,
    LEDGER_AMOUNT_COLUMNS,
    LEDGER_COLUMNS,
    LEDGER_DATE_COLUMNS,
    LEDGER_FIELDS,
    LEDGER_OPTIONAL_FIELDS,
    LEDGER_TEXT_COLUMNS,
    REQUIRED_TEXT,
    check_cleared_after_posting,
    check_document_numbers_unique,
    check_invoices_have_due_dates,
    check_nothing_after,
    date_or_none,
    document_roles,
    ledger_line,
    number,
    read_ledger,
    text_or_none,
)
from collection_estimation.hurdle.settings import LedgerLayout, document_types_setting
from tests.hurdle.workbooks import (
    LEDGER_HEADER,
    SYNTHETIC_LEDGER_HEADERS,
    SYNTHETIC_LEDGER_LAYOUT,
    credit_memo,
    invoice,
    ledger_row,
    make_ledger_workbook,
    payment,
    refund,
    unapplied_payment,
)

AS_OF = date(2026, 8, 28)
WHERE = 'test.xlsx row 2'


def five_lines():
    """An invoice, its payment, an unapplied payment, a credit memo and its refund."""
    return [invoice(), payment(), unapplied_payment(), credit_memo(), refund()]


LAYOUT = SYNTHETIC_LEDGER_LAYOUT  # header on row 1, lines from row 2, RV / DZ / DG


def read(tmp_path, lines, layout=LAYOUT, **kwargs):
    return read_ledger(make_ledger_workbook(tmp_path, lines, **kwargs), AS_OF, layout)


def with_types(**codes_per_role):
    """The synthetic layout with other document-type codes for some roles."""
    return dataclasses.replace(LAYOUT, document_types=dict(LAYOUT.document_types,
                                                           **codes_per_role))


def without(header, *removed):
    return tuple(text for text in header if text not in removed)


# --- constants -----------------------------------------------------------------------------

def test_constants_are_consistent():
    assert DOCUMENT_ROLES == ['invoice', 'payment', 'credit_memo']
    assert LEDGER_COLUMNS == LEDGER_FIELDS + ['document_role', 'source_row']
    assert len(LEDGER_FIELDS) == 17
    assert len(set(LEDGER_FIELDS)) == 17
    assert LEDGER_OPTIONAL_FIELDS <= set(LEDGER_FIELDS)
    assert set(REQUIRED_TEXT) <= set(LEDGER_TEXT_COLUMNS)
    assert 'document_role' in LEDGER_TEXT_COLUMNS
    # Every column is exactly one of date / amount / text / source_row.
    groups = LEDGER_DATE_COLUMNS + LEDGER_AMOUNT_COLUMNS + LEDGER_TEXT_COLUMNS + ['source_row']
    assert sorted(groups) == sorted(LEDGER_COLUMNS)
    assert len(groups) == len(set(groups))


def test_builder_header_matches_the_shared_layout():
    assert LEDGER_HEADER == tuple(SYNTHETIC_LEDGER_HEADERS)
    assert list(SYNTHETIC_LEDGER_HEADERS.values()) == LEDGER_FIELDS
    assert LAYOUT == LedgerLayout(header_row=1, first_data_row=2,
                                  headers=SYNTHETIC_LEDGER_HEADERS,
                                  document_types={'invoice': ('RV',), 'payment': ('DZ',),
                                                  'credit_memo': ('DG',)})


# --- happy path ----------------------------------------------------------------------------

def test_columns_are_ledger_columns_in_order(tmp_path):
    ledger = read(tmp_path, five_lines())
    assert list(ledger.columns) == LEDGER_COLUMNS


def test_one_row_per_line_in_sheet_order(tmp_path):
    ledger = read(tmp_path, five_lines())
    assert len(ledger) == 5
    assert ledger['document_no'].tolist() == [
        '9000000001', '1400000001', '1400000002', '9100000001', '1400000003']
    assert ledger['document_type'].tolist() == ['RV', 'DZ', 'DZ', 'DG', 'DZ']


def test_source_row_is_the_excel_row(tmp_path):
    ledger = read(tmp_path, five_lines())
    assert ledger['source_row'].tolist() == [2, 3, 4, 5, 6]
    assert ledger['source_row'].dtype == 'int64'


def test_date_columns_are_datetime64(tmp_path):
    ledger = read(tmp_path, five_lines())
    for column in LEDGER_DATE_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(ledger[column]), column


def test_dates_have_the_right_days(tmp_path):
    ledger = read(tmp_path, five_lines())
    first = ledger.iloc[0]
    assert first['posting_date'] == pd.Timestamp(2026, 3, 2)
    assert first['net_due_date'] == pd.Timestamp(2026, 4, 1)
    assert first['baseline_date'] == pd.Timestamp(2026, 3, 2)
    assert ledger.iloc[1]['posting_date'] == pd.Timestamp(2026, 3, 20)


def test_time_of_day_is_dropped(tmp_path):
    line = invoice(posted=datetime(2026, 3, 2, 17, 45, 12), due=datetime(2026, 4, 1, 9, 30))
    line['Clearing Date'] = datetime(2026, 4, 2, 23, 59, 59)
    ledger = read(tmp_path, [line])
    assert ledger.loc[0, 'posting_date'] == pd.Timestamp(2026, 3, 2)
    assert ledger.loc[0, 'net_due_date'] == pd.Timestamp(2026, 4, 1)
    assert ledger.loc[0, 'clearing_date'] == pd.Timestamp(2026, 4, 2)


def test_empty_dates_are_nat(tmp_path):
    ledger = read(tmp_path, five_lines())
    # The payment has no net due date and nothing is cleared.
    assert pd.isna(ledger.loc[1, 'net_due_date'])
    assert ledger['clearing_date'].isna().all()


def test_amounts_are_float64_and_ints_are_converted(tmp_path):
    lines = [invoice(amount=1000), payment(amount=250.5)]
    lines[0]['Amount in Local Currency'] = 1000
    ledger = read(tmp_path, lines)
    for column in LEDGER_AMOUNT_COLUMNS:
        assert ledger[column].dtype == 'float64', column
    assert ledger['amount_document'].tolist() == [1000.0, -250.5]
    assert ledger['amount_local'].tolist() == [1000.0, -250.5]


def test_sign_convention_of_the_line_makers_survives_reading(tmp_path):
    ledger = read(tmp_path, five_lines())
    assert ledger['amount_document'].tolist() == [1000.0, -1000.0, -500.0, -200.0, 200.0]


def test_text_columns_are_object_with_none_for_empty(tmp_path):
    ledger = read(tmp_path, five_lines())
    for column in LEDGER_TEXT_COLUMNS:
        assert ledger[column].dtype == object, column
    # The invoice has no invoice reference; the unapplied payment has none either.
    assert ledger.loc[0, 'invoice_reference'] is None
    assert ledger.loc[2, 'invoice_reference'] is None
    assert ledger.loc[1, 'reference'] is None
    assert ledger['clearing_document'].tolist() == [None] * 5


def test_references_link_payment_to_invoice_and_refund_to_memo(tmp_path):
    ledger = read(tmp_path, five_lines())
    assert ledger.loc[1, 'invoice_reference'] == ledger.loc[0, 'document_no']
    assert ledger.loc[4, 'invoice_reference'] == ledger.loc[3, 'document_no']
    assert ledger.loc[3, 'reference'] == ledger.loc[0, 'reference'] == 'GIB2026000001'


def test_customer_keeps_its_leading_zeros(tmp_path):
    ledger = read(tmp_path, [invoice(customer='0012345')])
    assert ledger.loc[0, 'customer'] == '0012345'


def test_document_type_is_upper_cased(tmp_path):
    line = invoice()
    line['Document Type'] = 'rv'
    second = payment()
    second['Document Type'] = 'Dz'
    ledger = read(tmp_path, [line, second])
    assert ledger['document_type'].tolist() == ['RV', 'DZ']


def test_document_type_is_stripped_and_upper_cased(tmp_path):
    line = credit_memo()
    line['Document Type'] = ' dg '
    ledger = read(tmp_path, [line])
    assert ledger.loc[0, 'document_type'] == 'DG'


def test_text_is_stripped(tmp_path):
    line = invoice(customer='  0012345 ', company=' TR02 ', currency=' USD ',
                   name='  ACME TEST A.S.  ')
    line['Document Number'] = ' 9000000001 '
    ledger = read(tmp_path, [line])
    row = ledger.iloc[0]
    assert row['customer'] == '0012345'
    assert row['company_code'] == 'TR02'
    assert row['document_currency'] == 'USD'
    assert row['customer_name'] == 'ACME TEST A.S.'
    assert row['document_no'] == '9000000001'


def test_cleared_line_keeps_clearing_date_and_document(tmp_path):
    line = invoice(clearing_date=datetime(2026, 3, 20), clearing_document='1400000001')
    ledger = read(tmp_path, [line])
    assert ledger.loc[0, 'clearing_date'] == pd.Timestamp(2026, 3, 20)
    assert ledger.loc[0, 'clearing_document'] == '1400000001'


def test_read_ledger_accepts_a_string_path(tmp_path):
    path = make_ledger_workbook(tmp_path, [invoice()])
    assert len(read_ledger(str(path), AS_OF, LAYOUT)) == 1


# --- empty rows ----------------------------------------------------------------------------

def test_empty_rows_anywhere_are_skipped(tmp_path):
    spaces = ['   '] * len(LEDGER_HEADER)
    lines = [None, invoice(), None, spaces, payment(), [None, ' ', '', None], unapplied_payment(),
             spaces]
    ledger = read(tmp_path, lines)
    assert ledger['document_no'].tolist() == ['9000000001', '1400000001', '1400000002']
    # Excel rows: 2 blank, 3 invoice, 4 blank, 5 spaces, 6 payment, 7 blank-ish, 8 unapplied.
    assert ledger['source_row'].tolist() == [3, 6, 8]


def test_header_only_gives_an_empty_table_with_the_right_dtypes(tmp_path):
    ledger = read(tmp_path, [])
    assert len(ledger) == 0
    assert list(ledger.columns) == LEDGER_COLUMNS
    for column in LEDGER_DATE_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(ledger[column]), column
    for column in LEDGER_AMOUNT_COLUMNS:
        assert ledger[column].dtype == 'float64', column
    for column in LEDGER_TEXT_COLUMNS:
        assert ledger[column].dtype == object, column
    assert ledger['source_row'].dtype == 'int64'


def test_header_and_only_blank_rows_gives_an_empty_table(tmp_path):
    ledger = read(tmp_path, [None, ['  '] * len(LEDGER_HEADER)])
    assert len(ledger) == 0
    assert list(ledger.columns) == LEDGER_COLUMNS


def test_completely_empty_sheet_raises_no_header_row(tmp_path):
    from openpyxl import Workbook
    workbook = Workbook()
    workbook.active.title = '_SYNTHETIC'
    workbook.create_sheet('FBL5N')
    path = tmp_path / 'empty.xlsx'
    workbook.save(path)
    with pytest.raises(ValueError, match=r'empty\.xlsx: no header row \(row 1\)'):
        read_ledger(path, AS_OF, LAYOUT)


# --- columns found by header text ----------------------------------------------------------

def test_shuffled_columns_are_found_by_header(tmp_path):
    shuffled = tuple(reversed(LEDGER_HEADER))
    straight = read(tmp_path, five_lines())
    other = tmp_path / 'other'
    other.mkdir()
    reordered = read(other, five_lines(), header=shuffled)
    pd.testing.assert_frame_equal(straight, reordered)


def test_header_case_and_spaces_do_not_matter(tmp_path):
    header = tuple('  ' + text.upper().replace(' ', '   ') + ' ' for text in LEDGER_HEADER)
    lines = [list(invoice().values())]
    ledger = read(tmp_path, lines, header=header)
    assert ledger.loc[0, 'document_no'] == '9000000001'


def test_extra_unknown_column_is_ignored(tmp_path):
    header = LEDGER_HEADER[:5] + ('Profit Center',) + LEDGER_HEADER[5:]
    line = invoice()
    line['Profit Center'] = 'PC-0001'
    ledger = read(tmp_path, [line], header=header)
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert ledger.loc[0, 'document_type'] == 'RV'
    assert ledger.loc[0, 'reference'] == 'GIB2026000001'


def test_optional_columns_absent_come_back_none(tmp_path):
    # The four optional columns all absent. 'Clearing Document' stays in the header (it is
    # required); its cells are empty because nothing in five_lines() is cleared.
    header = without(LEDGER_HEADER, 'Customer Name', 'Document Date', 'Baseline Date',
                     'Terms of Payment')
    ledger = read(tmp_path, five_lines(), header=header)
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert ledger['customer_name'].tolist() == [None] * 5
    assert ledger['payment_terms'].tolist() == [None] * 5
    assert ledger['clearing_document'].tolist() == [None] * 5
    assert ledger['document_date'].isna().all()
    assert ledger['baseline_date'].isna().all()
    assert pd.api.types.is_datetime64_any_dtype(ledger['document_date'])


OPTIONAL_HEADER_TEXTS = [('Customer Name', 'customer_name'), ('Document Date', 'document_date'),
                         ('Baseline Date', 'baseline_date'),
                         ('Terms of Payment', 'payment_terms')]


def test_the_optional_ledger_fields_are_exactly_the_four():
    assert LEDGER_OPTIONAL_FIELDS == {'customer_name', 'document_date', 'baseline_date',
                                      'payment_terms'}
    assert 'clearing_document' not in LEDGER_OPTIONAL_FIELDS
    assert {field for _, field in OPTIONAL_HEADER_TEXTS} == LEDGER_OPTIONAL_FIELDS


@pytest.mark.parametrize('header_text, field', OPTIONAL_HEADER_TEXTS)
def test_each_optional_column_alone_may_be_absent(tmp_path, header_text, field):
    ledger = read(tmp_path, five_lines(), header=without(LEDGER_HEADER, header_text))
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert ledger[field].isna().all()
    # The other columns still read as with the full header.
    other = tmp_path / 'full'
    other.mkdir()
    full = read(other, five_lines())
    pd.testing.assert_frame_equal(ledger.drop(columns=field), full.drop(columns=field))


@pytest.mark.parametrize('header_text, field', [
    ('Net Due Date', 'net_due_date'),
    ('Amount in Doc. Curr.', 'amount_document'),
    ('Posting Date', 'posting_date'),
    ('Customer', 'customer'),
    ('Clearing Date', 'clearing_date'),
    ('Invoice Reference', 'invoice_reference'),
    ('Clearing Document', 'clearing_document'),
])
def test_required_column_absent_raises_naming_it(tmp_path, header_text, field):
    header = without(LEDGER_HEADER, header_text)
    with pytest.raises(ValueError, match=f"missing columns.*'{field}'"):
        read(tmp_path, [invoice()], header=header)


# --- the Clearing Document column is required ----------------------------------------------
# The receivables step links lines that cleared together by their clearing document, so a
# ledger without that column fails at reading rather than loading with it all empty.

def test_a_ledger_without_the_clearing_document_column_raises_the_exact_message(tmp_path):
    header = without(LEDGER_HEADER, 'Clearing Document')
    with pytest.raises(ValueError) as error:
        read(tmp_path, five_lines(), header=header)
    assert str(error.value) == (f"missing columns ['clearing_document']; "
                                f"the header row reads {header}")


def test_a_ledger_without_the_clearing_document_column_raises_even_with_no_lines(tmp_path):
    header = without(LEDGER_HEADER, 'Clearing Document')
    with pytest.raises(ValueError, match=r"^missing columns \['clearing_document'\]; "):
        read(tmp_path, [], header=header)


def test_a_ledger_without_clearing_document_and_optional_columns_names_only_it(tmp_path):
    # The optional columns absent beside it are not named: only the required one is missing.
    header = without(LEDGER_HEADER, 'Clearing Document', 'Customer Name', 'Document Date',
                     'Baseline Date', 'Terms of Payment')
    with pytest.raises(ValueError) as error:
        read(tmp_path, five_lines(), header=header)
    assert str(error.value) == (f"missing columns ['clearing_document']; "
                                f"the header row reads {header}")


def test_clearing_document_missing_with_another_required_column_lists_both_sorted(tmp_path):
    header = without(LEDGER_HEADER, 'Clearing Document', 'Clearing Date')
    with pytest.raises(ValueError) as error:
        read(tmp_path, five_lines(), header=header)
    assert str(error.value) == (f"missing columns ['clearing_date', 'clearing_document']; "
                                f"the header row reads {header}")


def test_a_turkish_ledger_without_denklestirme_belgesi_raises(tmp_path):
    layout = dataclasses.replace(LAYOUT, headers=TURKISH_HEADERS)
    header = without(tuple(TURKISH_HEADERS), 'Denkleştirme Belgesi')
    lines = [as_turkish(row) for row in five_lines()]
    with pytest.raises(ValueError) as error:
        read(tmp_path, lines, layout=layout, header=header)
    assert str(error.value) == (f"missing columns ['clearing_document']; "
                                f"the header row reads {header}")


def test_the_clearing_document_header_is_matched_ignoring_case_and_spaces(tmp_path):
    header = tuple('  CLEARING   document ' if text == 'Clearing Document' else text
                   for text in LEDGER_HEADER)
    line = invoice(clearing_date=datetime(2026, 3, 20), clearing_document='1400000001')
    ledger = read(tmp_path, [list(line.values())], header=header)
    assert ledger.loc[0, 'clearing_document'] == '1400000001'


@pytest.mark.parametrize('empty', [None, '', '   '])
def test_an_empty_clearing_document_on_an_open_line_reads_as_none(tmp_path, empty):
    line = invoice()  # open: no clearing date
    line['Clearing Document'] = empty
    ledger = read(tmp_path, [line])
    assert ledger.loc[0, 'clearing_document'] is None
    assert pd.isna(ledger.loc[0, 'clearing_date'])
    assert ledger['clearing_document'].dtype == object


def test_open_and_cleared_lines_side_by_side_keep_their_clearing_documents(tmp_path):
    cleared = invoice('9000000001', clearing_date=datetime(2026, 3, 20),
                      clearing_document='1400000001')
    paid = payment(clearing_date=datetime(2026, 3, 20), clearing_document='1400000001')
    still_open = invoice('9000000002', e_number='GIB2026000002')
    ledger = read(tmp_path, [cleared, paid, still_open])
    assert ledger['clearing_document'].tolist() == ['1400000001', '1400000001', None]
    assert ledger.loc[2, 'clearing_document'] is None


def test_a_clearing_document_given_as_a_number_raises(tmp_path):
    # An ID read as a number may have lost its leading zeros; text only, as for other IDs.
    line = invoice(clearing_date=datetime(2026, 3, 20))
    line['Clearing Document'] = 1400000001
    with pytest.raises(ValueError,
                       match=r'row 2: clearing_document is 1400000001, not text'):
        read(tmp_path, [line])


def test_a_hand_built_layout_with_no_clearing_document_mapping_reads_it_as_none(tmp_path):
    # PINNED, not endorsed: read_ledger only requires the fields the layout maps. A layout
    # built by hand (not through the settings file, whose headers_setting demands a header
    # for clearing_document) that leaves it out reads a file without the column, all None.
    # The guard for this case is in settings.py (see test_settings.py).
    headers = {text: field for text, field in SYNTHETIC_LEDGER_HEADERS.items()
               if field != 'clearing_document'}
    layout = dataclasses.replace(LAYOUT, headers=headers)
    ledger = read(tmp_path, five_lines(), layout=layout,
                  header=without(LEDGER_HEADER, 'Clearing Document'))
    assert ledger['clearing_document'].tolist() == [None] * 5


def test_duplicate_column_raises(tmp_path):
    header = LEDGER_HEADER + ('Customer',)
    with pytest.raises(ValueError, match="field 'customer' is in both column B and column R"):
        read(tmp_path, [invoice()], header=header)


# --- type errors ---------------------------------------------------------------------------

def test_customer_as_a_number_raises(tmp_path):
    line = invoice()
    line['Customer'] = 12345
    with pytest.raises(ValueError, match=r'SYNTHETIC_ledger\.xlsx row 2: customer is 12345, not text'):
        read(tmp_path, [line])


def test_document_number_as_a_number_raises(tmp_path):
    line = payment()
    line['Document Number'] = 1400000001
    with pytest.raises(ValueError,
                       match=r'SYNTHETIC_ledger\.xlsx row 3: document_no is 1400000001, not text'):
        read(tmp_path, [invoice(), line])


def test_optional_text_as_a_number_also_raises(tmp_path):
    line = invoice()
    line['Terms of Payment'] = 30
    with pytest.raises(ValueError, match='row 2: payment_terms is 30, not text'):
        read(tmp_path, [line])


def test_date_cell_holding_text_raises(tmp_path):
    line = invoice()
    line['Net Due Date'] = '01.04.2026'
    with pytest.raises(ValueError,
                       match=r"SYNTHETIC_ledger\.xlsx row 2: net_due_date is '01\.04\.2026', not a date"):
        read(tmp_path, [line])


def test_amount_as_text_raises(tmp_path):
    line = invoice()
    line['Amount in Doc. Curr.'] = '1.000,00'
    with pytest.raises(ValueError,
                       match=r"SYNTHETIC_ledger\.xlsx row 2: amount_document is '1\.000,00', not a number"):
        read(tmp_path, [line])


def test_amount_as_true_raises(tmp_path):
    line = invoice()
    line['Amount in Local Currency'] = True
    with pytest.raises(ValueError, match='row 2: amount_local is True, not a number'):
        read(tmp_path, [line])


def test_empty_amount_raises(tmp_path):
    line = invoice()
    line['Amount in Doc. Curr.'] = None
    with pytest.raises(ValueError, match='row 2: amount_document is None, not a number'):
        read(tmp_path, [line])


def test_amount_of_only_spaces_raises(tmp_path):
    line = payment()
    line['Amount in Local Currency'] = '   '
    with pytest.raises(ValueError, match="row 3: amount_local is '   ', not a number"):
        read(tmp_path, [invoice(), line])


# --- required fields empty -----------------------------------------------------------------

@pytest.mark.parametrize('header_text, field', [
    ('Company Code', 'company_code'),
    ('Customer', 'customer'),
    ('Document Number', 'document_no'),
    ('Document Type', 'document_type'),
    ('Document Currency', 'document_currency'),
])
@pytest.mark.parametrize('empty', [None, '   '])
def test_required_text_empty_raises(tmp_path, header_text, field, empty):
    line = invoice()
    line[header_text] = empty
    with pytest.raises(ValueError, match=rf'SYNTHETIC_ledger\.xlsx row 2: {field} is empty'):
        read(tmp_path, [line])


def test_empty_posting_date_raises(tmp_path):
    line = payment()
    line['Posting Date'] = None
    with pytest.raises(ValueError, match=r'SYNTHETIC_ledger\.xlsx row 3: posting_date is empty'):
        read(tmp_path, [invoice(), line])


def test_empty_optional_and_unrequired_fields_are_fine(tmp_path):
    # The Clearing Document column must be present, but its cell may be empty (an open line).
    line = unapplied_payment()
    for text in ('Customer Name', 'Reference', 'Invoice Reference', 'Document Date',
                 'Baseline Date', 'Terms of Payment', 'Net Due Date', 'Clearing Date',
                 'Clearing Document'):
        line[text] = None
    ledger = read(tmp_path, [line])
    assert len(ledger) == 1


# --- ledger_line directly ------------------------------------------------------------------

def test_ledger_line_builds_the_dict():
    columns = {field: position for position, field in enumerate(LEDGER_FIELDS)}
    row = tuple(invoice().values())
    line = ledger_line(row, columns, WHERE, 7)
    assert list(line) == LEDGER_COLUMNS
    assert line['document_role'] is None  # set for the whole table by read_ledger
    assert line['source_row'] == 7
    assert line['posting_date'] == date(2026, 3, 2)
    assert type(line['posting_date']) is date
    assert line['amount_document'] == 1000.0


def test_ledger_line_short_row_gives_none_for_missing_cells():
    columns = {field: position for position, field in enumerate(LEDGER_FIELDS)}
    # A row cut short after the local amount: clearing date and document missing.
    row = tuple(invoice().values())[:15]
    line = ledger_line(row, columns, WHERE, 2)
    assert line['clearing_date'] is None
    assert line['clearing_document'] is None


# --- document_roles ------------------------------------------------------------------

def frame(**columns):
    """A small ledger-like table for the checks."""
    table = pd.DataFrame(columns)
    if 'source_row' not in table:
        table['source_row'] = range(2, 2 + len(table))
    return table


ROLE_OF_TYPE = {'RV': 'invoice', 'DZ': 'payment', 'DG': 'credit_memo'}


def with_roles(**columns):
    """`frame`, plus the document_role of each document_type (RV / DZ / DG)."""
    table = frame(**columns)
    table['document_role'] = table['document_type'].map(ROLE_OF_TYPE).astype(object)
    return table


TYPES = {'invoice': ('RV',), 'payment': ('DZ',), 'credit_memo': ('DG',)}


def test_document_roles_gives_the_role_of_each_line():
    roles = document_roles(pd.Series(['RV', 'DZ', 'DG', 'RV', 'DZ']), TYPES, 'x.xlsx')
    assert roles.tolist() == ['invoice', 'payment', 'credit_memo', 'invoice', 'payment']


def test_document_roles_is_object_dtype():
    roles = document_roles(pd.Series(['RV', 'DZ', 'DG']), TYPES, 'x.xlsx')
    assert roles.dtype == object
    assert all(type(role) is str for role in roles)


def test_document_roles_keeps_the_index():
    types = pd.Series(['DG', 'RV'], index=[10, 3])
    roles = document_roles(types, TYPES, 'x.xlsx')
    assert roles.index.tolist() == [10, 3]
    assert roles.to_dict() == {10: 'credit_memo', 3: 'invoice'}


def test_document_roles_several_codes_for_one_role():
    types = dict(TYPES, payment=('DZ', 'DA'))
    roles = document_roles(pd.Series(['DA', 'DZ', 'RV']), types, 'x.xlsx')
    assert roles.tolist() == ['payment', 'payment', 'invoice']


def test_document_roles_of_an_empty_series_is_empty_and_object():
    roles = document_roles(pd.Series([], dtype=object), TYPES, 'x.xlsx')
    assert len(roles) == 0
    assert roles.dtype == object


def test_document_roles_does_not_modify_its_inputs():
    types = pd.Series(['RV', 'DZ'])
    codes = {'invoice': ('RV',), 'payment': ('DZ',), 'credit_memo': ('DG',)}
    document_roles(types, codes, 'x.xlsx')
    assert types.tolist() == ['RV', 'DZ']
    assert codes == TYPES


def test_unknown_document_types_raise_with_counts_and_known_list():
    types = pd.Series(['RV', 'AB', 'DZ', 'AB', 'DA'])
    with pytest.raises(ValueError) as error:
        document_roles(types, TYPES, 'x.xlsx')
    message = str(error.value)
    assert message.startswith('x.xlsx: unknown document types')
    assert "'AB': 2" in message
    assert "'DA': 1" in message
    unknown_part = message.split('the settings list')[0]
    assert "'RV'" not in unknown_part and "'DZ'" not in unknown_part
    assert message.endswith("the settings list ['DG', 'DZ', 'RV']")


def test_unknown_document_types_list_the_known_codes_sorted_across_roles():
    types = {'invoice': ('ZZ', 'RV'), 'payment': ('DZ', 'AA'), 'credit_memo': ('DG',)}
    with pytest.raises(ValueError, match=r"the settings list \['AA', 'DG', 'DZ', 'RV', 'ZZ'\]$"):
        document_roles(pd.Series(['XX']), types, 'x.xlsx')


def test_a_code_dropped_from_the_settings_becomes_unknown():
    types = dict(TYPES, credit_memo=('DX',))
    with pytest.raises(ValueError, match=r"unknown document types .*\{'DG': 1\}"):
        document_roles(pd.Series(['RV', 'DG']), types, 'x.xlsx')


def test_unknown_document_type_in_a_workbook_raises(tmp_path):
    lines = [invoice(), ledger_row(document_type='ab', document_no='1'),
             ledger_row(document_type='AB', document_no='2'),
             ledger_row(document_type='DA', document_no='3')]
    with pytest.raises(ValueError,
                       match=r"SYNTHETIC_ledger\.xlsx: unknown document types .*'AB': 2.*'DA': 1"
                             r".*the settings list \['DG', 'DZ', 'RV'\]"):
        read(tmp_path, lines)


def test_a_type_not_in_the_layout_raises_from_read_ledger_with_counts(tmp_path):
    # DA is a real SAP code, but this layout does not list it.
    lines = [invoice(), payment(), payment(document_no='2'), credit_memo()]
    for row in lines[1:3]:
        row['Document Type'] = 'DA'
    with pytest.raises(ValueError, match=r"unknown document types .*\{'DA': 2\}"):
        read(tmp_path, lines)


# --- document_roles: codes in a hand-built layout are normalised --------------------

def test_document_roles_matches_lower_case_and_padded_layout_codes():
    # A layout built by hand (not through the settings file) with ('rv',) and (' Dz ',).
    types = {'invoice': ('rv',), 'payment': (' Dz ',), 'credit_memo': ('DG',)}
    roles = document_roles(pd.Series(['RV', 'DZ', 'DG']), types, 'x.xlsx')
    assert roles.tolist() == ['invoice', 'payment', 'credit_memo']


def test_document_roles_message_lists_the_layout_codes_normalised():
    types = {'invoice': ('rv',), 'payment': (' Dz ',), 'credit_memo': ('dg',)}
    with pytest.raises(ValueError, match=r"the settings list \['DG', 'DZ', 'RV'\]$"):
        document_roles(pd.Series(['XX']), types, 'x.xlsx')


def test_document_roles_does_not_modify_a_hand_built_layout():
    types = {'invoice': ('rv',), 'payment': (' Dz ',), 'credit_memo': ('DG',)}
    document_roles(pd.Series(['RV', 'DZ']), types, 'x.xlsx')
    assert types == {'invoice': ('rv',), 'payment': (' Dz ',), 'credit_memo': ('DG',)}


def test_a_hand_built_layout_with_lower_case_codes_reads_a_workbook(tmp_path):
    layout = with_types(invoice=('rv',), payment=(' Dz ',))
    ledger = read(tmp_path, [invoice(), payment(), credit_memo()], layout)
    assert ledger['document_role'].tolist() == ['invoice', 'payment', 'credit_memo']


def test_a_hand_built_layout_with_lower_case_codes_reads_like_the_upper_case_one(tmp_path):
    lines = [invoice(), payment(), credit_memo()]
    lower = read(tmp_path, lines, with_types(invoice=('rv',), payment=(' Dz ',),
                                             credit_memo=('dg',)))
    upper = read(tmp_path, lines)
    pd.testing.assert_frame_equal(lower, upper)


# --- document_roles: a missing type is named ----------------------------------------

def test_a_missing_document_type_is_named_in_the_unknown_types_message():
    with pytest.raises(ValueError) as error:
        document_roles(pd.Series(['RV', None, 'DZ'], dtype=object), TYPES, 'x.xlsx')
    message = str(error.value)
    assert message.startswith('x.xlsx: unknown document types')
    assert 'None: 1' in message


def test_missing_document_types_are_counted_together():
    with pytest.raises(ValueError, match=r"\{None: 2\}"):
        document_roles(pd.Series([None, 'RV', None], dtype=object), TYPES, 'x.xlsx')


def test_a_missing_type_is_named_beside_an_unknown_code():
    with pytest.raises(ValueError) as error:
        document_roles(pd.Series(['AB', None, 'AB'], dtype=object), TYPES, 'x.xlsx')
    message = str(error.value)
    assert "'AB': 2" in message
    assert 'None: 1' in message


def test_a_message_with_only_a_missing_type_does_not_show_empty_counts():
    # With value_counts() (dropna=True) the missing type was dropped and the counts read {}.
    with pytest.raises(ValueError) as error:
        document_roles(pd.Series([None], dtype=object), TYPES, 'x.xlsx')
    assert '{}' not in str(error.value)


# --- the document_role column --------------------------------------------------------

def test_document_role_column_is_present_in_ledger_columns_order(tmp_path):
    ledger = read(tmp_path, five_lines())
    assert 'document_role' in ledger.columns
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert list(ledger.columns).index('document_role') == len(LEDGER_FIELDS)
    assert ledger.columns[-1] == 'source_row'


def test_document_role_is_right_per_line(tmp_path):
    ledger = read(tmp_path, five_lines())
    # invoice, payment, unapplied payment, credit memo, refund (a DZ: a payment)
    assert ledger['document_role'].tolist() == [
        'invoice', 'payment', 'payment', 'credit_memo', 'payment']


def test_document_role_is_object_dtype(tmp_path):
    ledger = read(tmp_path, five_lines())
    assert ledger['document_role'].dtype == object
    assert all(type(role) is str for role in ledger['document_role'])


def test_document_role_of_an_empty_ledger_is_an_empty_object_column(tmp_path):
    ledger = read(tmp_path, [])
    assert ledger['document_role'].dtype == object
    assert len(ledger['document_role']) == 0


def test_document_role_follows_lower_case_types_in_the_ledger(tmp_path):
    lines = [invoice(), payment(), credit_memo()]
    for row, text in zip(lines, ['rv', ' dz ', 'Dg']):
        row['Document Type'] = text
    ledger = read(tmp_path, lines)
    assert ledger['document_type'].tolist() == ['RV', 'DZ', 'DG']
    assert ledger['document_role'].tolist() == ['invoice', 'payment', 'credit_memo']


def test_several_codes_for_one_role_both_map_to_payment(tmp_path):
    second = payment(document_no='1400000009')
    second['Document Type'] = 'DA'
    ledger = read(tmp_path, [invoice(), payment(), second],
                  layout=with_types(payment=('DZ', 'DA')))
    assert ledger['document_type'].tolist() == ['RV', 'DZ', 'DA']
    assert ledger['document_role'].tolist() == ['invoice', 'payment', 'payment']


def test_other_codes_in_the_layout_give_the_roles(tmp_path):
    # SAP codes are configured per company: e.g. DR for invoices.
    lines = [invoice(), payment(), credit_memo()]
    lines[0]['Document Type'] = 'DR'
    ledger = read(tmp_path, lines, layout=with_types(invoice=('DR',)))
    assert ledger['document_role'].tolist() == ['invoice', 'payment', 'credit_memo']


def test_rv_is_unknown_when_the_layout_lists_another_invoice_code(tmp_path):
    with pytest.raises(ValueError, match=r"unknown document types .*\{'RV': 1\}"):
        read(tmp_path, [invoice(), payment()], layout=with_types(invoice=('DR',)))


def test_lower_case_codes_in_the_settings_match_upper_case_ledger_types(tmp_path):
    codes = document_types_setting(
        {'d': {'invoice': ['rv'], 'payment': [' dz ', 'Da'], 'credit_memo': ['dg']}}, 'd', 'w')
    assert codes == {'invoice': ('RV',), 'payment': ('DZ', 'DA'), 'credit_memo': ('DG',)}
    second = payment(document_no='1400000009')
    second['Document Type'] = 'DA'
    ledger = read(tmp_path, [invoice(), payment(), second, credit_memo()],
                  layout=dataclasses.replace(LAYOUT, document_types=codes))
    assert ledger['document_role'].tolist() == ['invoice', 'payment', 'payment', 'credit_memo']


def test_the_due_date_check_follows_the_role_not_the_code(tmp_path):
    # With DR as the invoice code, a DR without a due date is caught ...
    line = invoice()
    line['Document Type'] = 'DR'
    line['Net Due Date'] = None
    with pytest.raises(ValueError, match=r'1 invoices have no net due date \(first rows: \[2\]\)'):
        read(tmp_path, [line], layout=with_types(invoice=('DR',), credit_memo=('DG', 'RV')))


def test_an_rv_that_the_layout_calls_a_credit_memo_needs_no_due_date(tmp_path):
    # ... and an RV the layout calls a credit memo is not an invoice, so it needs none.
    line = invoice()
    line['Net Due Date'] = None
    ledger = read(tmp_path, [line], layout=with_types(invoice=('DR',), credit_memo=('DG', 'RV')))
    assert ledger['document_role'].tolist() == ['credit_memo']


# --- layouts other than the synthetic one --------------------------------------------

TURKISH_HEADERS = {
    'Şirket Kodu': 'company_code',
    'Müşteri': 'customer',
    'Müşteri Adı': 'customer_name',
    'Belge Numarası': 'document_no',
    'Belge Türü': 'document_type',
    'Referans': 'reference',
    'Fatura Referansı': 'invoice_reference',
    'Belge Tarihi': 'document_date',
    'Kayıt Tarihi': 'posting_date',
    'Temel Tarih': 'baseline_date',
    'Ödeme Koşulu': 'payment_terms',
    'Net Vade Tarihi': 'net_due_date',
    'Belge Para Birimi Tutarı': 'amount_document',
    'Belge Para Birimi': 'document_currency',
    'Yerel Para Birimi Tutarı': 'amount_local',
    'Denkleştirme Tarihi': 'clearing_date',
    'Denkleştirme Belgesi': 'clearing_document',
}


def as_turkish(row):
    """A line dict keyed by LEDGER_HEADER -> the same values keyed by TURKISH_HEADERS."""
    english_to_turkish = dict(zip(LEDGER_HEADER, TURKISH_HEADERS))
    return {english_to_turkish[text]: value for text, value in row.items()}


def only_type_in_turkish(text):
    return 'Belge Türü' if text == 'Document Type' else text


def test_turkish_header_texts_read_a_workbook_built_that_way(tmp_path):
    layout = dataclasses.replace(LAYOUT, headers=TURKISH_HEADERS)
    lines = [as_turkish(row) for row in five_lines()]
    ledger = read(tmp_path, lines, layout=layout, header=tuple(TURKISH_HEADERS))
    other = tmp_path / 'english'
    other.mkdir()
    pd.testing.assert_frame_equal(ledger, read(other, five_lines()))
    assert ledger['document_type'].tolist() == ['RV', 'DZ', 'DZ', 'DG', 'DZ']


def test_belge_turu_alone_renamed_maps_to_document_type(tmp_path):
    headers = {only_type_in_turkish(text): field
               for text, field in SYNTHETIC_LEDGER_HEADERS.items()}
    header = tuple(only_type_in_turkish(text) for text in LEDGER_HEADER)
    lines = [{only_type_in_turkish(text): value for text, value in row.items()}
             for row in five_lines()]
    ledger = read(tmp_path, lines, layout=dataclasses.replace(LAYOUT, headers=headers),
                  header=header)
    assert ledger['document_type'].tolist() == ['RV', 'DZ', 'DZ', 'DG', 'DZ']
    assert ledger['document_role'].tolist() == [
        'invoice', 'payment', 'payment', 'credit_memo', 'payment']


def test_english_headers_against_a_turkish_layout_raise_missing_columns(tmp_path):
    layout = dataclasses.replace(LAYOUT, headers=TURKISH_HEADERS)
    with pytest.raises(ValueError, match='missing columns'):
        read(tmp_path, five_lines(), layout=layout)


def test_a_layout_with_the_header_on_row_3_reads_lines_from_row_4(tmp_path):
    layout = dataclasses.replace(LAYOUT, header_row=3, first_data_row=4)
    title_rows = (('CUSTOMER LINE ITEMS',), ('Extracted 28.08.2026',))
    ledger = read(tmp_path, five_lines(), layout=layout, title_rows=title_rows)
    assert ledger['source_row'].tolist() == [4, 5, 6, 7, 8]
    other = tmp_path / 'plain'
    other.mkdir()
    plain = read(other, five_lines())
    pd.testing.assert_frame_equal(ledger.drop(columns='source_row'),
                                  plain.drop(columns='source_row'))


def test_a_gap_between_header_and_first_data_row_skips_the_rows_between(tmp_path):
    # Header on row 1, data from row 3: the row in between (e.g. a sub-header) is not read.
    layout = dataclasses.replace(LAYOUT, header_row=1, first_data_row=3)
    sub_header = ['(text)'] * len(LEDGER_HEADER)
    ledger = read(tmp_path, [sub_header, invoice(), payment()], layout=layout)
    assert ledger['source_row'].tolist() == [3, 4]
    assert ledger['document_role'].tolist() == ['invoice', 'payment']


def test_the_synthetic_rows_against_a_title_row_workbook_raise(tmp_path):
    # The header is on row 2 but the layout says row 1: the title is taken as the header.
    with pytest.raises(ValueError, match='missing columns'):
        read(tmp_path, five_lines(), title_rows=(('CUSTOMER LINE ITEMS',),))


def test_a_header_row_past_the_end_of_the_sheet_raises(tmp_path):
    layout = dataclasses.replace(LAYOUT, header_row=50, first_data_row=51)
    with pytest.raises(ValueError, match=r'no header row \(row 50\)'):
        read(tmp_path, five_lines(), layout=layout)


def test_read_ledger_does_not_modify_the_layout(tmp_path):
    layout = LedgerLayout(header_row=1, first_data_row=2, headers=dict(SYNTHETIC_LEDGER_HEADERS),
                          document_types={'invoice': ('RV',), 'payment': ('DZ',),
                                          'credit_memo': ('DG',)})
    read(tmp_path, five_lines(), layout=layout)
    assert layout == LAYOUT


# --- check_invoices_have_due_dates ---------------------------------------------------------

def test_invoice_without_due_date_raises_with_count_and_rows():
    table = with_roles(document_type=['RV', 'RV', 'DZ', 'RV'],
                       net_due_date=pd.to_datetime([None, '2026-04-01', None, None]))
    with pytest.raises(ValueError,
                       match=r'x\.xlsx: 2 invoices have no net due date \(first rows: \[2, 5\]\)'):
        check_invoices_have_due_dates(table, 'x.xlsx')


def test_invoice_without_due_date_lists_at_most_five_rows():
    table = with_roles(document_type=['RV'] * 7, net_due_date=pd.to_datetime([None] * 7))
    with pytest.raises(ValueError, match=r'7 invoices .*\[2, 3, 4, 5, 6\]\)'):
        check_invoices_have_due_dates(table, 'x.xlsx')


def test_payment_and_credit_memo_without_due_date_are_fine():
    table = with_roles(document_type=['RV', 'DZ', 'DG'],
                       net_due_date=pd.to_datetime(['2026-04-01', None, None]))
    check_invoices_have_due_dates(table, 'x.xlsx')


def test_the_due_date_check_reads_the_role_column_not_the_type():
    # A line whose role is 'invoice' is checked whatever its code; an RV whose role is not
    # 'invoice' is not.
    table = frame(document_type=['DR', 'RV'], document_role=['invoice', 'credit_memo'],
                  net_due_date=pd.to_datetime([None, None]))
    with pytest.raises(ValueError, match=r'1 invoices have no net due date \(first rows: \[2\]\)'):
        check_invoices_have_due_dates(table, 'x.xlsx')


def test_the_due_date_check_needs_the_role_column():
    table = frame(document_type=['RV'], net_due_date=pd.to_datetime([None]))
    with pytest.raises(KeyError, match='document_role'):
        check_invoices_have_due_dates(table, 'x.xlsx')


def test_invoice_without_due_date_in_a_workbook_raises(tmp_path):
    line = invoice()
    line['Net Due Date'] = None
    with pytest.raises(ValueError, match=r'1 invoices have no net due date \(first rows: \[3\]\)'):
        read(tmp_path, [payment(), line, credit_memo()])


def test_lower_case_rv_without_due_date_is_still_caught(tmp_path):
    line = invoice()
    line['Document Type'] = 'rv'
    line['Net Due Date'] = None
    with pytest.raises(ValueError, match='invoices have no net due date'):
        read(tmp_path, [line])


# --- check_document_numbers_unique ---------------------------------------------------------

SHARED_COLUMNS = ['company_code', 'document_no', 'document_role', 'posting_date',
                  'net_due_date', 'amount_document', 'source_row']
SHARED_INTRO = (' document numbers are used by more than one invoice or credit memo of the '
                'same company, so payments cannot be matched to one document; the fiscal year '
                'and line item would tell them apart. First lines:\n')


def numbered(*lines):
    """A ledger-like table from (company, number, role) triples; source rows from 2.

    Each line gets its own posting date, due date and amount so the listed table shows which
    line is which.
    """
    count = len(lines)
    return pd.DataFrame({
        'company_code': [company for company, _, _ in lines],
        'document_no': [number for _, number, _ in lines],
        'document_role': pd.Series([role for _, _, role in lines], dtype=object),
        'posting_date': pd.to_datetime([f'2025-03-{day:02d}' for day in range(1, count + 1)]),
        'net_due_date': pd.to_datetime([f'2025-04-{day:02d}' for day in range(1, count + 1)]),
        'amount_document': [100.0 * (position + 1) for position in range(count)],
        'source_row': list(range(2, 2 + count)),
    })


def expected_message(file_name, groups, table, rows):
    """The message the check should give, listing `rows` (source rows, in listed order)."""
    listed = table.set_index('source_row', drop=False).loc[rows, SHARED_COLUMNS]
    return f'{file_name}: {groups}{SHARED_INTRO}{listed.to_string(index=False)}'


def message_of(table, file_name='x.xlsx'):
    with pytest.raises(ValueError) as error:
        check_document_numbers_unique(table, file_name)
    return str(error.value)


def listed_rows(message):
    """The source rows in the table of the message (its last column), in listed order."""
    table_lines = message.split('First lines:\n', 1)[1].splitlines()
    return [int(text.split()[-1]) for text in table_lines[1:]]


def test_two_invoices_of_one_company_with_one_number_raise():
    table = numbered(('TR02', '9000000001', 'invoice'), ('TR02', '9000000001', 'invoice'))
    assert message_of(table) == expected_message('x.xlsx', 1, table, [2, 3])


def test_an_invoice_and_a_credit_memo_with_one_number_raise():
    table = numbered(('TR02', '9000000001', 'invoice'), ('TR02', '9000000001', 'credit_memo'))
    message = message_of(table)
    assert message == expected_message('x.xlsx', 1, table, [2, 3])
    assert 'credit_memo' in message


def test_two_credit_memos_with_one_number_raise():
    table = numbered(('TR02', '9100000001', 'credit_memo'), ('TR02', '9100000001', 'credit_memo'))
    assert message_of(table).startswith('x.xlsx: 1 document numbers')


def test_the_same_number_in_two_companies_is_fine():
    table = numbered(('TR01', '9000000001', 'invoice'), ('TR02', '9000000001', 'invoice'),
                     ('TR03', '9000000001', 'credit_memo'))
    check_document_numbers_unique(table, 'x.xlsx')


def test_payments_sharing_a_number_with_each_other_are_fine():
    table = numbered(('TR02', '1400000001', 'payment'), ('TR02', '1400000001', 'payment'),
                     ('TR02', '1400000001', 'payment'))
    check_document_numbers_unique(table, 'x.xlsx')


def test_a_payment_sharing_a_number_with_an_invoice_is_fine():
    table = numbered(('TR02', '9000000001', 'invoice'), ('TR02', '9000000001', 'payment'),
                     ('TR02', '9100000001', 'credit_memo'), ('TR02', '9100000001', 'payment'))
    check_document_numbers_unique(table, 'x.xlsx')


def test_a_payment_with_the_shared_number_is_not_listed():
    table = numbered(('TR02', '9000000001', 'invoice'), ('TR02', '9000000001', 'payment'),
                     ('TR02', '9000000001', 'invoice'))
    message = message_of(table)
    assert message == expected_message('x.xlsx', 1, table, [2, 4])
    assert 'payment' not in message.split('First lines:\n', 1)[1]


def test_n_counts_the_shared_pairs():
    table = numbered(('TR02', '1', 'invoice'), ('TR02', '2', 'invoice'),
                     ('TR02', '1', 'invoice'), ('TR02', '2', 'credit_memo'),
                     ('TR02', '3', 'invoice'))
    message = message_of(table)
    assert message.startswith('x.xlsx: 2 document numbers')
    assert message == expected_message('x.xlsx', 2, table, [2, 4, 3, 5])


def test_one_number_shared_in_two_companies_counts_twice():
    # (TR01, 1) and (TR02, 1) are two pairs, even though the number is the same.
    table = numbered(('TR01', '1', 'invoice'), ('TR02', '1', 'invoice'),
                     ('TR01', '1', 'invoice'), ('TR02', '1', 'invoice'))
    assert message_of(table) == expected_message('x.xlsx', 2, table, [2, 4, 3, 5])


def test_a_triple_counts_as_one_pair_and_lists_all_three_lines():
    table = numbered(('TR02', '7', 'invoice'), ('TR02', '7', 'credit_memo'),
                     ('TR02', '7', 'invoice'))
    message = message_of(table)
    assert message.startswith('x.xlsx: 1 document numbers')
    assert listed_rows(message) == [2, 3, 4]
    assert message == expected_message('x.xlsx', 1, table, [2, 3, 4])


def test_more_than_ten_shared_lines_lists_only_ten():
    lines = [('TR02', str(number), 'invoice') for number in range(1, 7) for _ in range(2)]
    table = numbered(*lines)  # 6 pairs, 12 lines
    message = message_of(table)
    assert message.startswith('x.xlsx: 6 document numbers')
    rows = listed_rows(message)
    assert rows == list(range(2, 12))
    assert len(message.split('First lines:\n', 1)[1].splitlines()) == 1 + 10  # header + 10
    assert message == expected_message('x.xlsx', 6, table, rows)


def test_exactly_ten_shared_lines_lists_all_ten():
    lines = [('TR02', str(number), 'invoice') for number in range(1, 6) for _ in range(2)]
    assert listed_rows(message_of(numbered(*lines))) == list(range(2, 12))


def test_only_the_shared_lines_are_listed():
    table = numbered(('TR02', '1', 'invoice'), ('TR02', '2', 'invoice'),
                     ('TR01', '1', 'invoice'), ('TR02', '1', 'credit_memo'),
                     ('TR02', '3', 'credit_memo'), ('TR02', '2', 'payment'))
    message = message_of(table)
    assert listed_rows(message) == [2, 5]
    assert message == expected_message('x.xlsx', 1, table, [2, 5])


def test_listed_lines_are_sorted_by_company_number_and_source_row():
    # Sheet order is scrambled; the list is company, then number, then source row.
    table = numbered(('TR02', 'B', 'invoice'), ('TR01', 'Z', 'invoice'),
                     ('TR02', 'A', 'invoice'), ('TR02', 'B', 'credit_memo'),
                     ('TR01', 'Z', 'invoice'), ('TR02', 'A', 'invoice'))
    message = message_of(table)
    # TR01 Z (rows 3, 6), TR02 A (rows 4, 7), TR02 B (rows 2, 5)
    assert listed_rows(message) == [3, 6, 4, 7, 2, 5]
    assert message == expected_message('x.xlsx', 3, table, [3, 6, 4, 7, 2, 5])


def test_sorting_follows_source_row_not_the_table_order():
    table = numbered(('TR02', '1', 'invoice'), ('TR02', '1', 'invoice'))
    reversed_table = table.iloc[::-1]
    assert listed_rows(message_of(reversed_table)) == [2, 3]


def test_no_error_when_all_numbers_are_distinct():
    table = numbered(('TR02', '1', 'invoice'), ('TR02', '2', 'invoice'),
                     ('TR02', '3', 'credit_memo'), ('TR01', '4', 'invoice'))
    check_document_numbers_unique(table, 'x.xlsx')


def test_an_empty_ledger_passes():
    check_document_numbers_unique(numbered(), 'x.xlsx')


def test_the_check_does_not_modify_its_input():
    table = numbered(('TR02', 'B', 'invoice'), ('TR02', 'A', 'payment'),
                     ('TR02', 'B', 'credit_memo'))
    before = table.copy(deep=True)
    message_of(table)
    pd.testing.assert_frame_equal(table, before)
    passing = numbered(('TR02', 'B', 'invoice'), ('TR02', 'A', 'payment'))
    before = passing.copy(deep=True)
    check_document_numbers_unique(passing, 'x.xlsx')
    pd.testing.assert_frame_equal(passing, before)


def test_the_file_name_starts_the_message():
    table = numbered(('TR02', '1', 'invoice'), ('TR02', '1', 'invoice'))
    assert message_of(table, 'ledger 2026.xlsx').startswith('ledger 2026.xlsx: 1 document numbers')


# --- check_document_numbers_unique through read_ledger -------------------------------------

def test_a_number_reused_a_year_later_in_a_workbook_raises(tmp_path):
    lines = [invoice('9000000001', posted=datetime(2025, 3, 2), due=datetime(2025, 4, 1)),
             payment(posted=datetime(2025, 3, 20)),
             invoice('9000000001', amount=750.0, posted=datetime(2026, 3, 2),
                     due=datetime(2026, 4, 1))]
    with pytest.raises(ValueError) as error:
        read(tmp_path, lines)
    message = str(error.value)
    expected = pd.DataFrame({
        'company_code': ['TR02', 'TR02'],
        'document_no': ['9000000001', '9000000001'],
        'document_role': ['invoice', 'invoice'],
        'posting_date': pd.to_datetime(['2025-03-02', '2026-03-02']),
        'net_due_date': pd.to_datetime(['2025-04-01', '2026-04-01']),
        'amount_document': [1000.0, 750.0],
        'source_row': [2, 4],
    })
    assert message == (f'SYNTHETIC_ledger.xlsx: 1{SHARED_INTRO}'
                       f'{expected.to_string(index=False)}')


def test_an_invoice_and_a_credit_memo_with_one_number_in_a_workbook_raise(tmp_path):
    lines = [invoice('9000000001'), credit_memo('9000000001')]
    with pytest.raises(ValueError, match=r'SYNTHETIC_ledger\.xlsx: 1 document numbers are used'):
        read(tmp_path, lines)


def test_the_same_number_in_two_companies_in_a_workbook_is_fine(tmp_path):
    lines = [invoice('9000000001', company='TR01'), invoice('9000000001', company='TR02')]
    assert len(read(tmp_path, lines)) == 2


def test_payments_sharing_numbers_in_a_workbook_are_fine(tmp_path):
    lines = [invoice('9000000001'), payment('9000000001'), payment('1400000001'),
             payment('1400000001'), unapplied_payment('1400000001')]
    assert read(tmp_path, lines)['document_role'].tolist() == [
        'invoice', 'payment', 'payment', 'payment', 'payment']


def test_the_check_follows_the_role_in_a_workbook(tmp_path):
    # Two DR lines with one number: DR is an invoice code in this layout, so they are caught.
    lines = [invoice('9000000001'), invoice('9000000001')]
    for row in lines:
        row['Document Type'] = 'DR'
    with pytest.raises(ValueError, match='1 document numbers'):
        read(tmp_path, lines, layout=with_types(invoice=('DR',)))


def test_the_five_lines_pass_the_check(tmp_path):
    assert len(read(tmp_path, five_lines())) == 5


def test_the_due_date_check_runs_before_the_number_check(tmp_path):
    first = invoice('9000000001')
    first['Net Due Date'] = None
    lines = [first, invoice('9000000001')]
    with pytest.raises(ValueError, match='invoices have no net due date'):
        read(tmp_path, lines)


def test_the_number_check_runs_before_the_extract_date_check(tmp_path):
    lines = [invoice('9000000001'), invoice('9000000001', posted=datetime(2026, 9, 1),
                                            due=datetime(2026, 10, 1))]
    with pytest.raises(ValueError, match='1 document numbers'):
        read(tmp_path, lines)


# --- check_cleared_after_posting -----------------------------------------------------------

def posted_and_cleared(posting, clearing, roles=None):
    """A ledger-like table of posting and clearing dates (and roles); source rows from 2."""
    table = frame(posting_date=pd.to_datetime(posting), clearing_date=pd.to_datetime(clearing))
    if roles is not None:
        table['document_role'] = pd.Series(roles, dtype=object)
    return table


def test_one_line_cleared_before_posting_raises():
    table = posted_and_cleared(['2026-03-02', '2026-03-10', '2026-03-20'],
                               ['2026-03-20', '2026-03-09', None])
    with pytest.raises(ValueError,
                       match=r'^x\.xlsx: 1 lines have a clearing date before their posting date '
                             r'\(first rows: \[3\]\)$'):
        check_cleared_after_posting(table, 'x.xlsx')


def test_several_lines_cleared_before_posting_give_the_count_and_rows():
    table = posted_and_cleared(['2026-03-10', '2026-03-02', '2026-03-10', '2026-03-10'],
                               ['2026-03-01', '2026-03-05', '2026-03-09', '2026-02-28'])
    with pytest.raises(ValueError,
                       match=r'^x\.xlsx: 3 lines have a clearing date before their posting date '
                             r'\(first rows: \[2, 4, 5\]\)$'):
        check_cleared_after_posting(table, 'x.xlsx')


def test_cleared_before_posting_lists_at_most_five_rows():
    table = posted_and_cleared(['2026-03-10'] * 8, ['2026-03-01'] * 8)
    with pytest.raises(ValueError, match=r': 8 lines .*\(first rows: \[2, 3, 4, 5, 6\]\)$'):
        check_cleared_after_posting(table, 'x.xlsx')


def test_exactly_five_bad_lines_lists_all_five():
    table = posted_and_cleared(['2026-03-10'] * 6, ['2026-03-01'] * 5 + ['2026-03-10'])
    with pytest.raises(ValueError, match=r': 5 lines .*\(first rows: \[2, 3, 4, 5, 6\]\)$'):
        check_cleared_after_posting(table, 'x.xlsx')


def test_cleared_before_posting_rows_are_in_table_order():
    # The first rows are taken in table order, not sorted by source row.
    table = posted_and_cleared(['2026-03-10'] * 3, ['2026-03-01'] * 3)
    table['source_row'] = [9, 4, 7]
    with pytest.raises(ValueError, match=r'\(first rows: \[9, 4, 7\]\)$'):
        check_cleared_after_posting(table, 'x.xlsx')


def test_cleared_one_day_before_posting_raises():
    table = posted_and_cleared(['2026-03-10'], ['2026-03-09'])
    with pytest.raises(ValueError, match='1 lines have a clearing date before'):
        check_cleared_after_posting(table, 'x.xlsx')


def test_cleared_on_the_posting_date_passes():
    table = posted_and_cleared(['2026-03-10', '2026-03-02'], ['2026-03-10', '2026-03-02'])
    check_cleared_after_posting(table, 'x.xlsx')


def test_cleared_after_posting_passes():
    table = posted_and_cleared(['2026-03-02', '2026-03-02'], ['2026-03-03', '2027-01-01'])
    check_cleared_after_posting(table, 'x.xlsx')


def test_open_lines_pass_the_clearing_check():
    table = posted_and_cleared(['2026-03-02', '2026-08-28'], [None, None])
    check_cleared_after_posting(table, 'x.xlsx')


def test_open_lines_are_not_counted_beside_a_bad_line():
    table = posted_and_cleared(['2026-03-02', '2026-03-10', '2026-03-02'],
                               [None, '2026-03-01', None])
    with pytest.raises(ValueError, match=r': 1 lines .*\(first rows: \[3\]\)$'):
        check_cleared_after_posting(table, 'x.xlsx')


@pytest.mark.parametrize('role', DOCUMENT_ROLES)
def test_every_role_is_checked_for_clearing_before_posting(role):
    others = [other for other in DOCUMENT_ROLES if other != role]
    table = posted_and_cleared(['2026-03-02', '2026-03-10', '2026-03-02'],
                               ['2026-03-05', '2026-03-01', '2026-03-05'],
                               roles=[others[0], role, others[1]])
    with pytest.raises(ValueError, match=r': 1 lines .*\(first rows: \[3\]\)$'):
        check_cleared_after_posting(table, 'x.xlsx')


def test_an_empty_ledger_passes_the_clearing_check():
    check_cleared_after_posting(posted_and_cleared([], []), 'x.xlsx')


def test_the_clearing_check_does_not_modify_its_input():
    failing = posted_and_cleared(['2026-03-10', '2026-03-02'], ['2026-03-01', None],
                                 roles=['invoice', 'payment'])
    before = failing.copy(deep=True)
    with pytest.raises(ValueError):
        check_cleared_after_posting(failing, 'x.xlsx')
    pd.testing.assert_frame_equal(failing, before)
    passing = posted_and_cleared(['2026-03-02', '2026-03-02'], ['2026-03-02', None],
                                 roles=['invoice', 'payment'])
    before = passing.copy(deep=True)
    check_cleared_after_posting(passing, 'x.xlsx')
    pd.testing.assert_frame_equal(passing, before)


def test_the_file_name_starts_the_clearing_message():
    table = posted_and_cleared(['2026-03-10'], ['2026-03-01'])
    with pytest.raises(ValueError, match=r'^ledger 2026\.xlsx: 1 lines have a clearing date'):
        check_cleared_after_posting(table, 'ledger 2026.xlsx')


# --- check_cleared_after_posting through read_ledger ---------------------------------------

def test_an_invoice_cleared_before_posting_in_a_workbook_raises(tmp_path):
    lines = [invoice(clearing_date=datetime(2026, 3, 1)), payment()]
    with pytest.raises(ValueError,
                       match=r'^SYNTHETIC_ledger\.xlsx: 1 lines have a clearing date before their '
                             r'posting date \(first rows: \[2\]\)$'):
        read(tmp_path, lines)


def test_several_lines_cleared_before_posting_in_a_workbook_raise(tmp_path):
    bad = [invoice(str(9000000001 + position), clearing_date=datetime(2026, 3, 1))
           for position in range(7)]
    # Row 2 blank (skipped); rows 3, 4 bad; row 5 a payment cleared after posting; 6..10 bad.
    lines = [None, bad[0], bad[1], payment(clearing_date=datetime(2026, 3, 20))] + bad[2:]
    with pytest.raises(ValueError,
                       match=r': 7 lines have a clearing date before their posting date '
                             r'\(first rows: \[3, 4, 6, 7, 8\]\)$'):
        read(tmp_path, lines)


def test_lines_cleared_on_their_posting_date_in_a_workbook_pass(tmp_path):
    day = datetime(2026, 3, 2)
    lines = [invoice(clearing_date=day), payment(posted=day, clearing_date=day),
             credit_memo(posted=day, clearing_date=day)]
    ledger = read(tmp_path, lines)
    assert len(ledger) == 3
    assert (ledger['clearing_date'] == ledger['posting_date']).all()


def test_open_lines_in_a_workbook_pass_the_clearing_check(tmp_path):
    ledger = read(tmp_path, five_lines())
    assert ledger['clearing_date'].isna().all()


@pytest.mark.parametrize('maker, role', [(invoice, 'invoice'), (payment, 'payment'),
                                         (unapplied_payment, 'payment'),
                                         (credit_memo, 'credit_memo'), (refund, 'payment')])
def test_every_role_cleared_before_posting_in_a_workbook_raises(tmp_path, maker, role):
    # Every maker posts in March 2026; the bad line goes last (row 6).
    bad = maker(clearing_date=datetime(2026, 2, 1))
    others = [line for line in five_lines() if line['Document Number'] != bad['Document Number']]
    with pytest.raises(ValueError, match=r': 1 lines .*\(first rows: \[6\]\)$'):
        read(tmp_path, others + [bad])
    # The same line cleared after posting reads, with the role named.
    good = maker(clearing_date=datetime(2026, 4, 30))
    other = tmp_path / 'good'
    other.mkdir()
    ledger = read(other, others + [good])
    assert ledger['document_role'].iloc[-1] == role


def test_an_empty_ledger_workbook_passes_the_clearing_check(tmp_path):
    assert len(read(tmp_path, [])) == 0


def test_the_number_check_runs_before_the_clearing_check(tmp_path):
    lines = [invoice('9000000001'), invoice('9000000001', clearing_date=datetime(2026, 3, 1))]
    with pytest.raises(ValueError, match='1 document numbers are used'):
        read(tmp_path, lines)


def test_the_due_date_check_runs_before_the_clearing_check(tmp_path):
    line = invoice(clearing_date=datetime(2026, 3, 1))
    line['Net Due Date'] = None
    with pytest.raises(ValueError, match='invoices have no net due date'):
        read(tmp_path, [line])


def test_the_clearing_check_runs_before_the_extract_date_check(tmp_path):
    lines = [invoice(clearing_date=datetime(2026, 3, 1)),
             payment(posted=datetime(2026, 9, 1))]  # posted after AS_OF
    with pytest.raises(ValueError, match='clearing date before their posting date'):
        read(tmp_path, lines)


def test_one_line_failing_both_date_checks_reports_the_clearing_check(tmp_path):
    # Posted and cleared after AS_OF, and cleared before it was posted.
    lines = [invoice(posted=datetime(2026, 9, 10), due=datetime(2026, 10, 10),
                     clearing_date=datetime(2026, 9, 5))]
    with pytest.raises(ValueError, match=r'1 lines have a clearing date before .*\[2\]'):
        read(tmp_path, lines)


# --- check_nothing_after -------------------------------------------------------------------

def dated(posting, clearing):
    return frame(posting_date=pd.to_datetime(posting), clearing_date=pd.to_datetime(clearing))


def test_posting_date_after_as_of_raises():
    table = dated(['2026-08-27', '2026-08-29', '2026-08-28', '2026-09-01'], [None] * 4)
    with pytest.raises(ValueError,
                       match=r'x\.xlsx: 2 lines have a posting_date after the extract date '
                             r'2026-08-28 \(first rows: \[3, 5\]\)'):
        check_nothing_after(table, AS_OF, 'x.xlsx')


def test_clearing_date_after_as_of_raises():
    table = dated(['2026-08-01', '2026-08-02'], ['2026-08-29', None])
    with pytest.raises(ValueError,
                       match=r'1 lines have a clearing_date after the extract date '
                             r'2026-08-28 \(first rows: \[2\]\)'):
        check_nothing_after(table, AS_OF, 'x.xlsx')


def test_dates_equal_to_as_of_pass():
    check_nothing_after(dated(['2026-08-28'], ['2026-08-28']), AS_OF, 'x.xlsx')


def test_open_lines_without_clearing_date_pass():
    check_nothing_after(dated(['2026-08-01', '2026-08-02'], [None, None]), AS_OF, 'x.xlsx')


def test_first_rows_are_at_most_five():
    table = dated(['2026-09-01'] * 8, [None] * 8)
    with pytest.raises(ValueError, match=r'8 lines .*\[2, 3, 4, 5, 6\]\)'):
        check_nothing_after(table, AS_OF, 'x.xlsx')


def test_posting_date_after_as_of_in_a_workbook_raises(tmp_path):
    lines = [invoice(), payment(posted=datetime(2026, 8, 29))]
    with pytest.raises(ValueError, match=r'1 lines have a posting_date after the extract date'):
        read(tmp_path, lines)


def test_clearing_date_after_as_of_in_a_workbook_raises(tmp_path):
    lines = [invoice(clearing_date=datetime(2026, 9, 3))]
    with pytest.raises(ValueError, match=r'1 lines have a clearing_date .*first rows: \[2\]'):
        read(tmp_path, lines)


def test_line_dated_on_as_of_in_a_workbook_passes(tmp_path):
    lines = [invoice(posted=datetime(2026, 8, 28), clearing_date=datetime(2026, 8, 28))]
    assert len(read(tmp_path, lines)) == 1


def test_net_due_date_after_as_of_is_fine(tmp_path):
    # An open invoice is normally due after the extract date.
    lines = [invoice(posted=datetime(2026, 8, 20), due=datetime(2026, 10, 19))]
    assert len(read(tmp_path, lines)) == 1


# --- text_or_none --------------------------------------------------------------------------

@pytest.mark.parametrize('value', [None, '', '  ', '\t \n'])
def test_text_or_none_empty_gives_none(value):
    assert text_or_none(value, WHERE, 'customer') is None


def test_text_or_none_strips():
    assert text_or_none(' x ', WHERE, 'customer') == 'x'


@pytest.mark.parametrize('value', [5, 5.0, True, datetime(2026, 1, 1)])
def test_text_or_none_non_text_raises(value):
    with pytest.raises(ValueError, match=rf'{WHERE}: customer is .*, not text'):
        text_or_none(value, WHERE, 'customer')


# --- date_or_none --------------------------------------------------------------------------

def test_date_or_none_none_gives_none():
    assert date_or_none(None, WHERE, 'posting_date') is None


def test_date_or_none_spaces_give_none():
    assert date_or_none('   ', WHERE, 'posting_date') is None


def test_date_or_none_datetime_gives_its_date():
    result = date_or_none(datetime(2026, 3, 2, 13, 5), WHERE, 'posting_date')
    assert result == date(2026, 3, 2)
    assert type(result) is date


def test_date_or_none_plain_date_raises():
    # Pinned behaviour: openpyxl delivers Excel dates as datetime, and the helper accepts
    # only datetime, so a plain `date` object is rejected.
    with pytest.raises(ValueError, match='posting_date is datetime.date\\(2026, 3, 2\\), not a date'):
        date_or_none(date(2026, 3, 2), WHERE, 'posting_date')


@pytest.mark.parametrize('value', ['2026-03-02', 46083, 46083.0])
def test_date_or_none_text_or_number_raises(value):
    with pytest.raises(ValueError, match='not a date'):
        date_or_none(value, WHERE, 'posting_date')


# --- number --------------------------------------------------------------------------------

def test_number_int_becomes_float():
    result = number(5, WHERE, 'amount_document')
    assert result == 5.0
    assert type(result) is float


def test_number_float_stays():
    assert number(-12.34, WHERE, 'amount_document') == -12.34


def test_number_zero_is_fine():
    assert number(0, WHERE, 'amount_document') == 0.0


@pytest.mark.parametrize('value', [True, False, '5', None, '', date(2026, 1, 1)])
def test_number_non_number_raises(value):
    with pytest.raises(ValueError, match=rf'{WHERE}: amount_document is .*, not a number'):
        number(value, WHERE, 'amount_document')


def test_number_nan_raises():
    with pytest.raises(ValueError, match=rf'{WHERE}: amount_document is NaN, not a number'):
        number(float('nan'), WHERE, 'amount_document')


# --- sheet choice --------------------------------------------------------------------------

def test_marker_sheet_is_ignored(tmp_path):
    with_marker = read(tmp_path, five_lines(), marker=True)
    other = tmp_path / 'other'
    other.mkdir()
    no_marker = read(other, five_lines(), marker=False)
    pd.testing.assert_frame_equal(with_marker, no_marker)


def test_two_data_sheets_raise(tmp_path):
    with pytest.raises(ValueError, match=r"expected exactly one data sheet, found \['FBL5N', 'Notes'\]"):
        read(tmp_path, five_lines(), extra_sheets=('Notes',))


def test_extra_marker_sheets_are_fine(tmp_path):
    assert len(read(tmp_path, five_lines(), extra_sheets=('_README',))) == 5


# --- integration with the synthetic ledger (SLOW: one ~30 s read per session) ---------------

LEDGER_FILE = (Path(__file__).resolve().parents[2] / 'data' / 'raw'
               / 'SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx')


@pytest.fixture(scope='module')
def synthetic_ledger():
    if not LEDGER_FILE.exists():
        pytest.skip(f'{LEDGER_FILE.name} is not in data/raw')
    return read_ledger(LEDGER_FILE, date(2026, 8, 28), LAYOUT)


def test_slow_synthetic_ledger_line_counts(synthetic_ledger):
    assert len(synthetic_ledger) == 75_100
    assert synthetic_ledger['document_type'].value_counts().to_dict() == {
        'RV': 37_808, 'DZ': 37_164, 'DG': 128}


def test_slow_synthetic_ledger_document_role_counts(synthetic_ledger):
    assert synthetic_ledger['document_role'].value_counts().to_dict() == {
        'invoice': 37_808, 'payment': 37_164, 'credit_memo': 128}
    assert synthetic_ledger['document_role'].dtype == object
    pairs = synthetic_ledger.groupby(['document_type', 'document_role']).size().to_dict()
    assert pairs == {('RV', 'invoice'): 37_808, ('DZ', 'payment'): 37_164,
                     ('DG', 'credit_memo'): 128}


def test_slow_synthetic_ledger_open_lines(synthetic_ledger):
    assert synthetic_ledger['clearing_date'].isna().sum() == 4_286


def test_slow_synthetic_ledger_posting_date_range(synthetic_ledger):
    assert synthetic_ledger['posting_date'].min() == pd.Timestamp(2024, 10, 25)
    assert synthetic_ledger['posting_date'].max() == pd.Timestamp(2026, 8, 28)


def test_slow_synthetic_ledger_every_invoice_has_a_due_date(synthetic_ledger):
    invoices = synthetic_ledger[synthetic_ledger['document_role'] == 'invoice']
    assert invoices['net_due_date'].notna().all()


def test_slow_synthetic_ledger_shape_and_types(synthetic_ledger):
    assert list(synthetic_ledger.columns) == LEDGER_COLUMNS
    assert synthetic_ledger['source_row'].is_monotonic_increasing
    assert synthetic_ledger['source_row'].iloc[0] == LAYOUT.first_data_row == 2
    for column in LEDGER_DATE_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(synthetic_ledger[column]), column


def test_slow_synthetic_ledger_has_no_shared_document_numbers(synthetic_ledger):
    # read_ledger already ran the check; run it again on the table and confirm directly.
    check_document_numbers_unique(synthetic_ledger, LEDGER_FILE.name)
    can_be_paid = synthetic_ledger[
        synthetic_ledger['document_role'].isin(['invoice', 'credit_memo'])]
    assert not can_be_paid.duplicated(['company_code', 'document_no']).any()


def test_slow_synthetic_ledger_earlier_as_of_raises(synthetic_ledger):
    # Same check read_ledger runs, applied to the table already read (avoids a second read).
    check_nothing_after(synthetic_ledger, date(2026, 8, 28), LEDGER_FILE.name)
    with pytest.raises(ValueError, match='after the extract date 2026-08-01'):
        check_nothing_after(synthetic_ledger, date(2026, 8, 1), LEDGER_FILE.name)


def test_slow_synthetic_ledger_has_no_line_cleared_before_posting(synthetic_ledger):
    # read_ledger already ran the check; run it again on the table and confirm directly.
    check_cleared_after_posting(synthetic_ledger, LEDGER_FILE.name)
    assert synthetic_ledger['clearing_date'].notna().sum() > 0
    assert not (synthetic_ledger['clearing_date'] < synthetic_ledger['posting_date']).any()
