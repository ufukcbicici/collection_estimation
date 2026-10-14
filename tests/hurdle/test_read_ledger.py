"""Tests for `hurdle/read_ledger.py`: reading the AR ledger (FBL5N layout) and its checks.

Every workbook is built in `tmp_path` by `make_ledger_workbook` (see `workbooks.py`).
Customers are fake. The integration tests at the end read the synthetic ledger in
`data/raw/` once per session (~30 s) and are skipped when it is absent.
"""
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.read_ledger import (
    DOCUMENT_TYPES,
    LEDGER_AMOUNT_COLUMNS,
    LEDGER_COLUMNS,
    LEDGER_DATE_COLUMNS,
    LEDGER_FIRST_DATA_ROW,
    LEDGER_HEADER_ROW,
    LEDGER_HEADER_TO_FIELD,
    LEDGER_OPTIONAL_FIELDS,
    LEDGER_TEXT_COLUMNS,
    REQUIRED_TEXT,
    check_document_types,
    check_invoices_have_due_dates,
    check_nothing_after,
    date_or_none,
    ledger_line,
    number,
    read_ledger,
    text_or_none,
)
from tests.hurdle.workbooks import (
    LEDGER_HEADER,
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


def read(tmp_path, lines, **kwargs):
    return read_ledger(make_ledger_workbook(tmp_path, lines, **kwargs), AS_OF)


def without(header, *removed):
    return tuple(text for text in header if text not in removed)


# --- constants -----------------------------------------------------------------------------

def test_constants_are_consistent():
    assert LEDGER_HEADER_ROW == 1
    assert LEDGER_FIRST_DATA_ROW == 2
    assert DOCUMENT_TYPES == {'RV', 'DZ', 'DG'}
    assert LEDGER_COLUMNS == list(LEDGER_HEADER_TO_FIELD.values()) + ['source_row']
    assert len(LEDGER_HEADER_TO_FIELD) == 17
    assert LEDGER_OPTIONAL_FIELDS <= set(LEDGER_HEADER_TO_FIELD.values())
    assert set(REQUIRED_TEXT) <= set(LEDGER_TEXT_COLUMNS)
    # Every column is exactly one of date / amount / text / source_row.
    groups = LEDGER_DATE_COLUMNS + LEDGER_AMOUNT_COLUMNS + LEDGER_TEXT_COLUMNS + ['source_row']
    assert sorted(groups) == sorted(LEDGER_COLUMNS)
    assert len(groups) == len(set(groups))


def test_builder_header_matches_the_module_dictionary():
    simplified = [' '.join(text.split()).casefold() for text in LEDGER_HEADER]
    assert simplified == list(LEDGER_HEADER_TO_FIELD)


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
    assert len(read_ledger(str(path), AS_OF)) == 1


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
    with pytest.raises(ValueError, match='no header row'):
        read_ledger(path, AS_OF)


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
    header = without(LEDGER_HEADER, 'Customer Name', 'Document Date', 'Baseline Date',
                     'Terms of Payment', 'Clearing Document')
    ledger = read(tmp_path, five_lines(), header=header)
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert ledger['customer_name'].tolist() == [None] * 5
    assert ledger['payment_terms'].tolist() == [None] * 5
    assert ledger['clearing_document'].tolist() == [None] * 5
    assert ledger['document_date'].isna().all()
    assert ledger['baseline_date'].isna().all()
    assert pd.api.types.is_datetime64_any_dtype(ledger['document_date'])


@pytest.mark.parametrize('header_text, field', [
    ('Net Due Date', 'net_due_date'),
    ('Amount in Doc. Curr.', 'amount_document'),
    ('Posting Date', 'posting_date'),
    ('Customer', 'customer'),
    ('Clearing Date', 'clearing_date'),
    ('Invoice Reference', 'invoice_reference'),
])
def test_required_column_absent_raises_naming_it(tmp_path, header_text, field):
    header = without(LEDGER_HEADER, header_text)
    with pytest.raises(ValueError, match=f"missing columns.*'{field}'"):
        read(tmp_path, [invoice()], header=header)


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
    line = unapplied_payment()
    for text in ('Customer Name', 'Reference', 'Invoice Reference', 'Document Date',
                 'Baseline Date', 'Terms of Payment', 'Net Due Date', 'Clearing Date',
                 'Clearing Document'):
        line[text] = None
    ledger = read(tmp_path, [line])
    assert len(ledger) == 1


# --- ledger_line directly ------------------------------------------------------------------

def test_ledger_line_builds_the_dict():
    columns = {field: position for position, field in enumerate(LEDGER_HEADER_TO_FIELD.values())}
    row = tuple(invoice().values())
    line = ledger_line(row, columns, WHERE, 7)
    assert list(line) == LEDGER_COLUMNS
    assert line['source_row'] == 7
    assert line['posting_date'] == date(2026, 3, 2)
    assert type(line['posting_date']) is date
    assert line['amount_document'] == 1000.0


def test_ledger_line_short_row_gives_none_for_missing_cells():
    columns = {field: position for position, field in enumerate(LEDGER_HEADER_TO_FIELD.values())}
    # A row cut short after the local amount: clearing date and document missing.
    row = tuple(invoice().values())[:15]
    line = ledger_line(row, columns, WHERE, 2)
    assert line['clearing_date'] is None
    assert line['clearing_document'] is None


# --- check_document_types ------------------------------------------------------------------

def frame(**columns):
    """A small ledger-like table for the checks."""
    table = pd.DataFrame(columns)
    if 'source_row' not in table:
        table['source_row'] = range(2, 2 + len(table))
    return table


def test_known_document_types_pass():
    check_document_types(frame(document_type=['RV', 'DZ', 'DG', 'RV']), 'x.xlsx')


def test_empty_table_passes_document_type_check():
    check_document_types(frame(document_type=pd.Series([], dtype=object)), 'x.xlsx')


def test_unknown_document_types_raise_with_counts_and_known_list():
    table = frame(document_type=['RV', 'AB', 'DZ', 'AB', 'DA'])
    with pytest.raises(ValueError) as error:
        check_document_types(table, 'x.xlsx')
    message = str(error.value)
    assert message.startswith('x.xlsx: unknown document types')
    assert "'AB': 2" in message
    assert "'DA': 1" in message
    assert "'RV'" not in message.split('known types')[0]
    assert "known types are ['DG', 'DZ', 'RV']" in message


def test_unknown_document_type_in_a_workbook_raises(tmp_path):
    lines = [invoice(), ledger_row(document_type='ab', document_no='1'),
             ledger_row(document_type='AB', document_no='2'),
             ledger_row(document_type='DA', document_no='3')]
    with pytest.raises(ValueError, match=r"unknown document types.*'AB': 2.*'DA': 1"):
        read(tmp_path, lines)


# --- check_invoices_have_due_dates ---------------------------------------------------------

def test_invoice_without_due_date_raises_with_count_and_rows():
    table = frame(document_type=['RV', 'RV', 'DZ', 'RV'],
                  net_due_date=pd.to_datetime([None, '2026-04-01', None, None]))
    with pytest.raises(ValueError,
                       match=r'x\.xlsx: 2 invoices have no net due date \(first rows: \[2, 5\]\)'):
        check_invoices_have_due_dates(table, 'x.xlsx')


def test_invoice_without_due_date_lists_at_most_five_rows():
    table = frame(document_type=['RV'] * 7, net_due_date=pd.to_datetime([None] * 7))
    with pytest.raises(ValueError, match=r'7 invoices .*\[2, 3, 4, 5, 6\]\)'):
        check_invoices_have_due_dates(table, 'x.xlsx')


def test_payment_and_credit_memo_without_due_date_are_fine():
    table = frame(document_type=['RV', 'DZ', 'DG'],
                  net_due_date=pd.to_datetime(['2026-04-01', None, None]))
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
    return read_ledger(LEDGER_FILE, date(2026, 8, 28))


def test_slow_synthetic_ledger_line_counts(synthetic_ledger):
    assert len(synthetic_ledger) == 75_100
    assert synthetic_ledger['document_type'].value_counts().to_dict() == {
        'RV': 37_808, 'DZ': 37_164, 'DG': 128}


def test_slow_synthetic_ledger_open_lines(synthetic_ledger):
    assert synthetic_ledger['clearing_date'].isna().sum() == 4_286


def test_slow_synthetic_ledger_posting_date_range(synthetic_ledger):
    assert synthetic_ledger['posting_date'].min() == pd.Timestamp(2024, 10, 25)
    assert synthetic_ledger['posting_date'].max() == pd.Timestamp(2026, 8, 28)


def test_slow_synthetic_ledger_every_invoice_has_a_due_date(synthetic_ledger):
    invoices = synthetic_ledger[synthetic_ledger['document_type'] == 'RV']
    assert invoices['net_due_date'].notna().all()


def test_slow_synthetic_ledger_shape_and_types(synthetic_ledger):
    assert list(synthetic_ledger.columns) == LEDGER_COLUMNS
    assert synthetic_ledger['source_row'].is_monotonic_increasing
    assert synthetic_ledger['source_row'].iloc[0] == LEDGER_FIRST_DATA_ROW
    for column in LEDGER_DATE_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(synthetic_ledger[column]), column


def test_slow_synthetic_ledger_earlier_as_of_raises(synthetic_ledger):
    # Same check read_ledger runs, applied to the table already read (avoids a second read).
    check_nothing_after(synthetic_ledger, date(2026, 8, 28), LEDGER_FILE.name)
    with pytest.raises(ValueError, match='after the extract date 2026-08-01'):
        check_nothing_after(synthetic_ledger, date(2026, 8, 1), LEDGER_FILE.name)
