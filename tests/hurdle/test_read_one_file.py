"""Tests for `read_one_file` in `hurdle/read_collections.py` and its helpers.

Every workbook is built in `tmp_path` by `make_workbook` (see `workbooks.py`), in the layout
of the daily files. Customers are fake.
"""
import dataclasses
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from collection_estimation.hurdle.read_collections import (
    LINE_COLUMNS,
    OPTIONAL_FIELDS,
    TEXT_COLUMNS,
    TIE_OUT_TOLERANCE,
    cell,
    check_grand_total,
    find_columns,
    is_blank,
    is_nan,
    last_non_empty_row,
    list_daily_files,
    payment_line,
    pick_data_sheet,
    read_one_file,
)
from collection_estimation.hurdle.settings import CollectionsLayout
from tests.hurdle.workbooks import (
    HEADER,
    NAME_COLUMN,
    SYNTHETIC_HEADERS,
    SYNTHETIC_LAYOUT,
    line,
    make_workbook,
    total_row,
)

DATE_FORMAT = SYNTHETIC_LAYOUT.file_name_date_format  # "%d%m%Y"

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"
SYNTHETIC_FIRST_FILE = SYNTHETIC_CORPUS / "02062025 EY TAHSILAT.XLSX"


# --- constants -------------------------------------------------------------------------

def test_shared_test_layout_has_header_on_row_2_and_data_from_row_3():
    # The rows make_workbook writes; settings/synthetic.toml says the same.
    assert (SYNTHETIC_LAYOUT.header_row, SYNTHETIC_LAYOUT.first_data_row) == (2, 3)


def test_tie_out_tolerance_is_one_kurus():
    assert TIE_OUT_TOLERANCE == 0.01


def test_line_columns_are_the_fields_then_source():
    assert LINE_COLUMNS == [
        'company_code', 'customer', 'customer_name', 'document_date', 'bank',
        'sap_document_no', 'invoice_no', 'amount_original', 'currency_original',
        'amount_try', 'currency_local', 'source_file', 'source_row']


def test_text_columns_are_all_but_the_date_amounts_and_row_number():
    assert TEXT_COLUMNS == [
        'company_code', 'customer', 'customer_name', 'bank', 'sap_document_no', 'invoice_no',
        'currency_original', 'currency_local', 'source_file']


# --- happy path ------------------------------------------------------------------------

def test_returns_line_columns_in_order(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line()]), SYNTHETIC_LAYOUT)
    assert list(table.columns) == LINE_COLUMNS


def test_one_row_per_payment_line_in_sheet_order(tmp_path):
    lines = [line('0000001', 10.0), line('0000002', 20.0), line('0000003', 30.0)]
    table = read_one_file(make_workbook(tmp_path, lines), SYNTHETIC_LAYOUT)
    assert list(table['customer']) == ['0000001', '0000002', '0000003']


def test_subtotal_blank_and_grand_total_rows_are_left_out(tmp_path):
    lines = [line('0000001', 10.0), line('0000002', 20.0), total_row('TR02 Total', 30.0), None,
             line('0000003', 5.0), total_row('TR03 Total', 5.0), None]
    table = read_one_file(make_workbook(tmp_path, lines), SYNTHETIC_LAYOUT)
    assert list(table['customer']) == ['0000001', '0000002', '0000003']


def test_source_file_is_the_file_name(tmp_path):
    path = make_workbook(tmp_path, [line()])
    table = read_one_file(path, SYNTHETIC_LAYOUT)
    assert list(table['source_file']) == ['02062025 EY TAHSILAT.XLSX']


def test_source_row_is_the_excel_row_number(tmp_path):
    # Rows 3 and 4 are lines, 5 a subtotal, 6 blank, 7 a line.
    lines = [line('0000001', 1.0), line('0000002', 2.0), total_row('Sub', 3.0), None,
             line('0000003', 4.0)]
    table = read_one_file(make_workbook(tmp_path, lines), SYNTHETIC_LAYOUT)
    assert list(table['source_row']) == [3, 4, 7]


def test_accepts_a_string_path(tmp_path):
    table = read_one_file(str(make_workbook(tmp_path, [line()])), SYNTHETIC_LAYOUT)
    assert len(table) == 1


def test_document_date_is_a_datetime64_column(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line(when=datetime(2025, 6, 2))]),
                          SYNTHETIC_LAYOUT)
    assert pd.api.types.is_datetime64_any_dtype(table['document_date'])


def test_document_date_has_the_right_date(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line(when=datetime(2025, 7, 31))]),
                          SYNTHETIC_LAYOUT)
    assert table['document_date'].iloc[0] == pd.Timestamp(2025, 7, 31)


def test_document_date_drops_the_time_of_day(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line(when=datetime(2025, 7, 31, 14, 30))]),
                          SYNTHETIC_LAYOUT)
    assert table['document_date'].iloc[0] == pd.Timestamp(2025, 7, 31)


def test_amounts_are_floats(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line(amount_try=100, amount_original=100)]),
                          SYNTHETIC_LAYOUT)
    assert table['amount_try'].dtype == float
    assert table['amount_original'].dtype == float
    assert isinstance(table['amount_try'].iloc[0], float)


def test_other_fields_are_carried_as_written(tmp_path):
    row = line('0012345', 3500.0, company='TR02', name='ACME TEST A.S.', bank='TEST BANK',
               sap='1400000001', invoice='GIB2025000001', amount_original=100.0, currency='USD')
    record = read_one_file(make_workbook(tmp_path, [row]), SYNTHETIC_LAYOUT).iloc[0]
    assert record['company_code'] == 'TR02'
    assert record['customer_name'] == 'ACME TEST A.S.'
    assert record['bank'] == 'TEST BANK'
    assert record['sap_document_no'] == '1400000001'
    assert record['invoice_no'] == 'GIB2025000001'
    assert record['amount_original'] == 100.0
    assert record['currency_original'] == 'USD'
    assert record['amount_try'] == 3500.0
    assert record['currency_local'] == 'TRY'


# --- text fields -----------------------------------------------------------------------

def test_customer_with_leading_zeros_stays_text(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line('0012345')]), SYNTHETIC_LAYOUT)
    assert table['customer'].iloc[0] == '0012345'


def test_spaces_around_customer_are_stripped(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line('  0012345 ')]), SYNTHETIC_LAYOUT)
    assert table['customer'].iloc[0] == '0012345'


def test_spaces_around_company_code_are_stripped(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line(company=' TR02  ')]), SYNTHETIC_LAYOUT)
    assert table['company_code'].iloc[0] == 'TR02'


# --- nothing is dropped ----------------------------------------------------------------

def test_refund_with_negative_amount_is_kept(tmp_path):
    lines = [line('0000001', 500.0), line('0000002', -120.5, amount_original=-120.5)]
    table = read_one_file(make_workbook(tmp_path, lines), SYNTHETIC_LAYOUT)
    assert list(table['amount_try']) == [500.0, -120.5]


def test_unapplied_cash_line_is_kept(tmp_path):
    unapplied = line('0000009', 75.0, sap=None, invoice='Ödeme Detayı Araştırılıyor')
    table = read_one_file(make_workbook(tmp_path, [line('0000001', 10.0), unapplied]),
                          SYNTHETIC_LAYOUT)
    record = table.iloc[1]
    assert record['invoice_no'] == 'Ödeme Detayı Araştırılıyor'
    assert record['sap_document_no'] is None


# --- missing text is None --------------------------------------------------------------

def test_empty_sap_number_next_to_filled_ones_is_none(tmp_path):
    lines = [line('0000001', 10.0, sap='1400000001'), line('0000002', 20.0, sap=None),
             line('0000003', 30.0, sap='1400000003')]
    table = read_one_file(make_workbook(tmp_path, lines), SYNTHETIC_LAYOUT)
    assert table['sap_document_no'].iloc[0] == '1400000001'
    assert table['sap_document_no'].iloc[1] is None
    assert table['sap_document_no'].iloc[2] == '1400000003'


def test_text_column_missing_on_every_line_is_none(tmp_path):
    lines = [line('0000001', 10.0, invoice=None), line('0000002', 20.0, invoice=None)]
    table = read_one_file(make_workbook(tmp_path, lines), SYNTHETIC_LAYOUT)
    assert all(value is None for value in table['invoice_no'])


def test_no_missing_text_value_is_nan(tmp_path):
    lines = [line('0000001', 10.0, sap=None, bank=None), line('0000002', 20.0, invoice=None)]
    table = read_one_file(make_workbook(tmp_path, lines), SYNTHETIC_LAYOUT)
    for column in TEXT_COLUMNS:
        for value in table[column]:
            assert not isinstance(value, float), (column, value)


def test_text_columns_are_object_dtype(tmp_path):
    lines = [line('0000001', 10.0), line('0000002', 20.0, sap=None)]
    table = read_one_file(make_workbook(tmp_path, lines), SYNTHETIC_LAYOUT)
    for column in TEXT_COLUMNS:
        assert table[column].dtype == object, column


def test_text_columns_of_an_empty_table_are_object_dtype(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [], grand_total=0.0), SYNTHETIC_LAYOUT)
    for column in TEXT_COLUMNS:
        assert table[column].dtype == object, column


def test_line_without_company_code_is_skipped_and_caught_by_the_tie_out(tmp_path):
    # A real payment line with its company code missing looks like a subtotal and is skipped;
    # the grand total still counts it, so the tie-out must fail rather than lose it silently.
    lines = [line('0000001', 10.0), line('0000002', 20.0, company=None)]
    path = make_workbook(tmp_path, lines, grand_total=30.0)
    with pytest.raises(ValueError, match='10.00 TRY'):
        read_one_file(path, SYNTHETIC_LAYOUT)


# --- choosing the data sheet -----------------------------------------------------------

def test_marker_sheet_is_ignored(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line()], marker=True), SYNTHETIC_LAYOUT)
    assert len(table) == 1


def test_single_sheet_with_any_name_and_no_marker_works(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line()], marker=False, name='Sheet1'),
                          SYNTHETIC_LAYOUT)
    assert len(table) == 1


def test_two_data_sheets_raise_naming_both(tmp_path):
    path = make_workbook(tmp_path, [line()], extra_sheets=['Pivot'])
    with pytest.raises(ValueError) as error:
        read_one_file(path, SYNTHETIC_LAYOUT)
    assert "'02062025'" in str(error.value)
    assert "'Pivot'" in str(error.value)
    assert path.name in str(error.value)


def test_extra_marker_sheets_are_all_ignored(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [line()], extra_sheets=['_NOTES']),
                          SYNTHETIC_LAYOUT)
    assert len(table) == 1


def test_pick_data_sheet_ignores_marker_sheets():
    assert pick_data_sheet(['_SYNTHETIC', '02062025', '_NOTES'], 'f.xlsx') == '02062025'


def test_pick_data_sheet_only_marker_sheets_raises():
    with pytest.raises(ValueError, match=r"f\.xlsx.*found \[\]"):
        pick_data_sheet(['_SYNTHETIC'], 'f.xlsx')


def test_pick_data_sheet_no_sheets_raises():
    with pytest.raises(ValueError, match='exactly one data sheet'):
        pick_data_sheet([], 'f.xlsx')


def test_pick_data_sheet_two_data_sheets_raise_naming_them():
    with pytest.raises(ValueError, match=r"\['A', 'B'\]"):
        pick_data_sheet(['_SYNTHETIC', 'A', 'B'], 'f.xlsx')


# --- columns found by header text ------------------------------------------------------

def test_extra_unknown_column_is_read_around(tmp_path):
    # 'Notes' inserted after 'Document Date': every value to its right moves one column.
    header = HEADER[:4] + ('Notes',) + HEADER[4:]
    row = line('0012345', 250.0, bank='TEST BANK')
    row = row[:4] + ('a note',) + row[4:]
    grand = [None] * len(header)
    grand[header.index('Amount in local currency')] = 250.0
    path = make_workbook(tmp_path, [row], header=header, last_rows=[tuple(grand)])
    record = read_one_file(path, SYNTHETIC_LAYOUT).iloc[0]
    assert record['bank'] == 'TEST BANK'
    assert record['amount_try'] == 250.0
    assert record['currency_local'] == 'TRY'


def test_without_bank_column_bank_is_none(tmp_path):
    bank = HEADER.index('Bank')
    header = HEADER[:bank] + HEADER[bank + 1:]
    row = line('0012345', 40.0)
    row = row[:bank] + row[bank + 1:]
    grand = [None] * len(header)
    grand[header.index('Amount in local currency')] = 40.0
    path = make_workbook(tmp_path, [row], header=header, last_rows=[tuple(grand)])
    table = read_one_file(path, SYNTHETIC_LAYOUT)
    assert table['bank'].iloc[0] is None
    assert table['amount_try'].iloc[0] == 40.0


def test_missing_required_column_raises(tmp_path):
    header = tuple(h for h in HEADER if h != 'Customer Name')
    path = make_workbook(tmp_path, [], header=header, last_rows=[])
    with pytest.raises(ValueError, match='customer_name'):
        read_one_file(path, SYNTHETIC_LAYOUT)


# --- type errors -----------------------------------------------------------------------

def test_customer_stored_as_a_number_raises_naming_file_and_row(tmp_path):
    path = make_workbook(tmp_path, [line('0000001'), line(12345)])
    with pytest.raises(ValueError, match=r'02062025 EY TAHSILAT\.XLSX row 4: customer is 12345'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_document_date_as_text_raises_naming_file_and_row(tmp_path):
    path = make_workbook(tmp_path, [line(when='02.06.2025')])
    with pytest.raises(ValueError, match=r"XLSX row 3: document date is '02\.06\.2025'"):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_missing_document_date_raises(tmp_path):
    path = make_workbook(tmp_path, [line(when=None)])
    with pytest.raises(ValueError, match='row 3: document date is None'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_amount_try_as_text_raises_naming_file_and_row(tmp_path):
    path = make_workbook(tmp_path, [line(amount_try='100,00', amount_original=100.0)],
                         grand_total=100.0)
    with pytest.raises(ValueError, match=r"XLSX row 3: amount_try is '100,00'"):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_amount_original_as_text_raises(tmp_path):
    path = make_workbook(tmp_path, [line(amount_try=100.0, amount_original='100')])
    with pytest.raises(ValueError, match="row 3: amount_original is '100'"):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_amount_as_bool_raises(tmp_path):
    path = make_workbook(tmp_path, [line(amount_try=True, amount_original=1.0)], grand_total=1.0)
    with pytest.raises(ValueError, match='row 3: amount_try is True'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_missing_amount_raises(tmp_path):
    path = make_workbook(tmp_path, [line(amount_try=None, amount_original=1.0)], grand_total=1.0)
    with pytest.raises(ValueError, match='row 3: amount_try is None'):
        read_one_file(path, SYNTHETIC_LAYOUT)


# --- tie-out against the grand total ---------------------------------------------------

def test_lines_not_adding_up_raise_with_both_totals(tmp_path):
    path = make_workbook(tmp_path, [line('0000001', 1000.0), line('0000002', 234.5)],
                         grand_total=1234.52)
    with pytest.raises(ValueError) as error:
        read_one_file(path, SYNTHETIC_LAYOUT)
    assert '1,234.50' in str(error.value)
    assert '1,234.52' in str(error.value)
    assert path.name in str(error.value)


def test_lines_off_by_half_a_kurus_pass(tmp_path):
    path = make_workbook(tmp_path, [line('0000001', 1000.0)], grand_total=1000.005)
    assert len(read_one_file(path, SYNTHETIC_LAYOUT)) == 1


def test_grand_total_ignores_subtotals(tmp_path):
    # Subtotals are not payment lines; counting them would double the total.
    lines = [line('0000001', 10.0), line('0000002', 20.0), total_row('Sub', 30.0)]
    path = make_workbook(tmp_path, lines, grand_total=30.0)
    assert read_one_file(path, SYNTHETIC_LAYOUT)['amount_try'].sum() == 30.0


def test_uncalculated_grand_total_skips_tie_out_with_a_note(tmp_path, capsys):
    path = make_workbook(tmp_path, [line('0000001', 10.0)], grand_total=None)
    table = read_one_file(path, SYNTHETIC_LAYOUT)
    assert len(table) == 1
    assert 'has no number in its grand total; tie-out skipped' in capsys.readouterr().out


def test_skip_note_names_the_file(tmp_path, capsys):
    path = make_workbook(tmp_path, [line('0000001', 10.0)], grand_total=None)
    read_one_file(path, SYNTHETIC_LAYOUT)
    assert path.name in capsys.readouterr().out


def test_last_row_a_payment_line_skips_tie_out_with_a_note(tmp_path, capsys):
    # No grand-total row: the last row is a payment line with a company code.
    path = make_workbook(tmp_path, [line('0000001', 10.0)],
                         last_rows=[line('0000002', 20.0)])
    table = read_one_file(path, SYNTHETIC_LAYOUT)
    assert list(table['customer']) == ['0000001', '0000002']
    assert 'has no grand-total row; tie-out skipped' in capsys.readouterr().out


def test_last_payment_line_then_spaces_row_skips_with_no_grand_total_row_note(tmp_path, capsys):
    # The spaces row is skipped, so the last non-empty row is the payment line.
    spaces = ('   ',) * len(HEADER)
    path = make_workbook(tmp_path, [line('0000001', 10.0)],
                         last_rows=[line('0000002', 20.0), spaces])
    assert len(read_one_file(path, SYNTHETIC_LAYOUT)) == 2
    assert 'has no grand-total row; tie-out skipped' in capsys.readouterr().out


def test_grand_total_as_text_raises(tmp_path):
    path = make_workbook(tmp_path, [line('0000001', 10.0)], grand_total='10.00')
    with pytest.raises(ValueError, match=r"the grand total is '10\.00', not a number"):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_grand_total_as_text_raise_names_the_file(tmp_path):
    path = make_workbook(tmp_path, [line('0000001', 10.0)], grand_total='10.00')
    with pytest.raises(ValueError) as error:
        read_one_file(path, SYNTHETIC_LAYOUT)
    assert path.name in str(error.value)


def test_grand_total_as_bool_raises(tmp_path):
    path = make_workbook(tmp_path, [line('0000001', 1.0)], grand_total=True)
    with pytest.raises(ValueError, match='the grand total is True, not a number'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_integer_grand_total_ties_out(tmp_path, capsys):
    path = make_workbook(tmp_path, [line('0000001', 10.0)], grand_total=10)
    assert len(read_one_file(path, SYNTHETIC_LAYOUT)) == 1
    assert capsys.readouterr().out == ''


def test_spaces_row_after_grand_total_is_skipped_and_matching_total_passes(tmp_path, capsys):
    # openpyxl writes this row: its cells hold text, only spaces.
    spaces = ('   ',) * len(HEADER)
    path = make_workbook(tmp_path, [line('0000001', 10.0)],
                         last_rows=[total_row('GRAND TOTAL', 10.0), spaces])
    assert len(read_one_file(path, SYNTHETIC_LAYOUT)) == 1
    assert capsys.readouterr().out == ''


def test_spaces_row_after_grand_total_is_skipped_and_mismatch_still_raises(tmp_path):
    spaces = ('   ',) * len(HEADER)
    path = make_workbook(tmp_path, [line('0000001', 10.0)],
                         last_rows=[total_row('GRAND TOTAL', 99.0), spaces])
    with pytest.raises(ValueError, match='99.00 TRY'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_several_blank_rows_after_grand_total_are_skipped(tmp_path):
    spaces = ('   ',) * len(HEADER)
    tab = ('\t',) + (None,) * (len(HEADER) - 1)
    path = make_workbook(tmp_path, [line('0000001', 10.0)],
                         last_rows=[total_row('GRAND TOTAL', 99.0), spaces, tab, spaces])
    with pytest.raises(ValueError, match='99.00 TRY'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_stray_note_after_grand_total_is_read_as_total_without_number(tmp_path, capsys):
    # CURRENT BEHAVIOUR, pinned: a note below the grand total becomes the last non-empty row,
    # so it is taken as the total row; its TRY cell is empty, so the tie-out is skipped with a
    # note even though the real grand total above it (99.00) does not match the lines.
    note = [None] * len(HEADER)
    note[NAME_COLUMN] = 'Hazırlayan: test'
    path = make_workbook(tmp_path, [line('0000001', 10.0)],
                         last_rows=[total_row('GRAND TOTAL', 99.0), tuple(note)])
    assert len(read_one_file(path, SYNTHETIC_LAYOUT)) == 1
    assert 'has no number in its grand total; tie-out skipped' in capsys.readouterr().out


def test_matching_grand_total_prints_nothing(tmp_path, capsys):
    read_one_file(make_workbook(tmp_path, [line('0000001', 10.0)]), SYNTHETIC_LAYOUT)
    assert capsys.readouterr().out == ''


# --- empty and malformed files ---------------------------------------------------------

def test_header_without_lines_and_zero_grand_total_gives_empty_table(tmp_path):
    table = read_one_file(make_workbook(tmp_path, [], grand_total=0.0), SYNTHETIC_LAYOUT)
    assert table.empty
    assert list(table.columns) == LINE_COLUMNS


def test_header_without_lines_but_nonzero_grand_total_raises(tmp_path):
    path = make_workbook(tmp_path, [], grand_total=50.0)
    with pytest.raises(ValueError, match='50.00 TRY'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_file_with_only_a_title_row_raises_no_header_row(tmp_path):
    workbook = Workbook()
    workbook.active.title = '02062025'
    workbook.active.append(['MERCURY COLLECTIONS'])
    path = tmp_path / '02062025 EY TAHSILAT.XLSX'
    workbook.save(path)
    with pytest.raises(ValueError, match='no header row'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_empty_sheet_raises_no_header_row(tmp_path):
    workbook = Workbook()
    workbook.active.title = '02062025'
    path = tmp_path / '02062025 EY TAHSILAT.XLSX'
    workbook.save(path)
    with pytest.raises(ValueError, match='no header row'):
        read_one_file(path, SYNTHETIC_LAYOUT)


# --- cell and is_blank -----------------------------------------------------------------

COLUMNS = {'company_code': 0, 'customer': 1, 'amount_try': 3}


def test_cell_returns_the_value_at_the_field_position():
    assert cell(('TR02', '0012345', 'x', 5.0), COLUMNS, 'amount_try') == 5.0


def test_cell_of_a_short_row_is_none():
    assert cell(('TR02', '0012345'), COLUMNS, 'amount_try') is None


def test_cell_of_an_empty_row_is_none():
    assert cell((), COLUMNS, 'company_code') is None


def test_cell_of_a_field_the_file_lacks_is_none():
    assert cell(('TR02', '0012345', 'x', 5.0), COLUMNS, 'bank') is None


def test_cell_at_position_zero_is_found():
    assert cell(('TR02',), COLUMNS, 'company_code') == 'TR02'


@pytest.mark.parametrize('value', [None, '', '   ', '\t\n'])
def test_is_blank_true(value):
    assert is_blank(value)


@pytest.mark.parametrize('value', ['TR02', 0, 0.0, ' x '])
def test_is_blank_false(value):
    assert not is_blank(value)


# --- last_non_empty_row ----------------------------------------------------------------

def test_last_non_empty_row_of_no_rows_is_empty():
    assert last_non_empty_row([]) == ()


def test_last_non_empty_row_of_all_blank_rows_is_empty():
    assert last_non_empty_row([(None, None), ('', '  '), (), ('\t', None)]) == ()


def test_last_non_empty_row_is_the_last_row_when_it_has_content():
    rows = [('a', 1), (None, 2)]
    assert last_non_empty_row(rows) == (None, 2)


def test_last_non_empty_row_skips_trailing_blank_rows():
    rows = [('a', 1), (None, 'TOTAL', 5.0), (None, None), ('', ''), ('  ', None), ()]
    assert last_non_empty_row(rows) == (None, 'TOTAL', 5.0)


def test_last_non_empty_row_counts_a_zero_as_content():
    rows = [('a', 1), (None, 0), (None, None)]
    assert last_non_empty_row(rows) == (None, 0)


def test_last_non_empty_row_returns_the_row_object_itself():
    target = (None, 'GRAND TOTAL', 5.0)
    assert last_non_empty_row([('a',), target, (None,)]) is target


# --- is_nan ----------------------------------------------------------------------------

def test_is_nan_true_for_float_nan():
    assert is_nan(float('nan'))


@pytest.mark.parametrize('value', [None, 0, 0.0, 1, 1.5, 'nan', 'NaN', '', True,
                                   float('inf')])
def test_is_nan_false(value):
    assert not is_nan(value)


# --- check_grand_total on hand-made rows -----------------------------------------------

GRAND_COLUMNS = find_columns(HEADER, SYNTHETIC_HEADERS, OPTIONAL_FIELDS)


def grand_rows(*last_rows):
    """Title, header, one payment line of 10.0 TRY, then `last_rows`."""
    return [('MERCURY COLLECTIONS',), HEADER, line('0000001', 10.0), *last_rows]


def lines_of(*amounts):
    return pd.DataFrame({'amount_try': [float(a) for a in amounts]})


def test_check_nan_grand_total_skips_with_no_number_note(capsys):
    rows = grand_rows(total_row('GRAND TOTAL', float('nan')))
    check_grand_total(lines_of(10.0), rows, GRAND_COLUMNS, 'f.xlsx')
    assert ('Note: f.xlsx has no number in its grand total; tie-out skipped'
            in capsys.readouterr().out)


def test_check_nan_grand_total_does_not_pass_silently_with_mismatching_lines(capsys):
    # Every comparison with NaN is False, so without the guard 999 vs NaN would "pass".
    rows = grand_rows(total_row('GRAND TOTAL', float('nan')))
    check_grand_total(lines_of(999.0), rows, GRAND_COLUMNS, 'f.xlsx')
    assert 'has no number in its grand total' in capsys.readouterr().out


def test_check_matching_total_prints_nothing(capsys):
    rows = grand_rows(total_row('GRAND TOTAL', 10.0))
    check_grand_total(lines_of(10.0), rows, GRAND_COLUMNS, 'f.xlsx')
    assert capsys.readouterr().out == ''


def test_check_mismatching_total_raises():
    rows = grand_rows(total_row('GRAND TOTAL', 11.0))
    with pytest.raises(ValueError, match='10.00 TRY.*11.00 TRY'):
        check_grand_total(lines_of(10.0), rows, GRAND_COLUMNS, 'f.xlsx')


def test_check_none_grand_total_skips_with_no_number_note(capsys):
    rows = grand_rows(total_row('GRAND TOTAL', None))
    check_grand_total(lines_of(10.0), rows, GRAND_COLUMNS, 'f.xlsx')
    assert 'f.xlsx has no number in its grand total' in capsys.readouterr().out


@pytest.mark.parametrize('value', ['10.00', 'nan', True, False])
def test_check_non_number_grand_total_raises(value):
    rows = grand_rows(total_row('GRAND TOTAL', value))
    with pytest.raises(ValueError, match=r'f\.xlsx: the grand total is .*, not a number'):
        check_grand_total(lines_of(10.0), rows, GRAND_COLUMNS, 'f.xlsx')


def test_check_last_row_with_company_code_skips_with_no_grand_total_row_note(capsys):
    rows = grand_rows(line('0000002', 5.0))
    check_grand_total(lines_of(10.0, 5.0), rows, GRAND_COLUMNS, 'f.xlsx')
    assert 'Note: f.xlsx has no grand-total row; tie-out skipped' in capsys.readouterr().out


def test_check_skips_trailing_blank_rows_to_find_the_total():
    rows = grand_rows(total_row('GRAND TOTAL', 11.0), (None,) * len(HEADER),
                      ('  ',) * len(HEADER))
    with pytest.raises(ValueError, match='11.00 TRY'):
        check_grand_total(lines_of(10.0), rows, GRAND_COLUMNS, 'f.xlsx')


def test_check_short_total_row_without_try_cell_skips_with_no_number_note(capsys):
    # openpyxl may hand back a row shorter than the header; the missing TRY cell is None.
    rows = grand_rows((None, None, 'GRAND TOTAL'))
    check_grand_total(lines_of(10.0), rows, GRAND_COLUMNS, 'f.xlsx')
    assert 'has no number in its grand total' in capsys.readouterr().out


# --- payment_line on hand-made rows: NaN amounts ---------------------------------------
# openpyxl cannot write a NaN into a cell, so these call payment_line directly.

LINE_FIELD_COLUMNS = find_columns(HEADER, SYNTHETIC_HEADERS, OPTIONAL_FIELDS)


def test_payment_line_nan_amount_try_raises_naming_file_row_and_field():
    row = line(amount_try=float('nan'), amount_original=100.0)
    with pytest.raises(ValueError, match=r'^f\.xlsx row 7: amount_try is NaN, not a number$'):
        payment_line(row, LINE_FIELD_COLUMNS, 'f.xlsx', 7)


def test_payment_line_nan_amount_original_raises_naming_file_row_and_field():
    row = line(amount_try=100.0, amount_original=float('nan'))
    with pytest.raises(ValueError,
                       match=r'^f\.xlsx row 9: amount_original is NaN, not a number$'):
        payment_line(row, LINE_FIELD_COLUMNS, 'f.xlsx', 9)


def test_payment_line_nan_in_both_amounts_names_amount_original_first():
    # The loop checks amount_original before amount_try.
    row = line(amount_try=float('nan'), amount_original=float('nan'))
    with pytest.raises(ValueError, match='amount_original is NaN'):
        payment_line(row, LINE_FIELD_COLUMNS, 'f.xlsx', 3)


def test_payment_line_float_amounts_pass_as_floats():
    result = payment_line(line(amount_try=123.45, amount_original=10.5),
                          LINE_FIELD_COLUMNS, 'f.xlsx', 3)
    assert result['amount_try'] == 123.45
    assert result['amount_original'] == 10.5
    assert type(result['amount_try']) is float
    assert type(result['amount_original']) is float


def test_payment_line_int_amounts_pass_converted_to_float():
    result = payment_line(line(amount_try=100, amount_original=7),
                          LINE_FIELD_COLUMNS, 'f.xlsx', 3)
    assert result['amount_try'] == 100.0 and type(result['amount_try']) is float
    assert result['amount_original'] == 7.0 and type(result['amount_original']) is float


def test_payment_line_carries_file_and_row_number():
    result = payment_line(line(), LINE_FIELD_COLUMNS, 'f.xlsx', 5)
    assert (result['source_file'], result['source_row']) == ('f.xlsx', 5)


@pytest.mark.parametrize('field', ['amount_try', 'amount_original'])
@pytest.mark.parametrize('value', [float('inf'), float('-inf')])
def test_payment_line_pin_infinite_amount_is_not_rejected_by_the_nan_guard(field, value):
    """PIN of current behaviour: the NaN guard lets +/-inf through unchanged.

    If infinities should be rejected too, change this test deliberately.
    """
    amounts = {'amount_try': 100.0, 'amount_original': 100.0, field: value}
    result = payment_line(line(**amounts), LINE_FIELD_COLUMNS, 'f.xlsx', 3)
    assert result[field] == value


# --- the layout comes from the settings -------------------------------------------------
# `read_one_file` takes the header row, the first data row and the header texts from the
# `CollectionsLayout` it is given; nothing is fixed in the module any more.

def layout_with(**changes):
    """SYNTHETIC_LAYOUT with some fields replaced."""
    return dataclasses.replace(SYNTHETIC_LAYOUT, **changes)


def test_layout_header_on_row_1_reads_a_workbook_without_a_title_row(tmp_path):
    path = make_workbook(tmp_path, [line('0000001', 10.0), line('0000002', 20.0)],
                         title_rows=())
    table = read_one_file(path, layout_with(header_row=1, first_data_row=2))
    assert list(table['customer']) == ['0000001', '0000002']
    assert list(table['amount_try']) == [10.0, 20.0]
    assert list(table['source_row']) == [2, 3]


def test_layout_header_on_row_1_on_a_workbook_with_a_title_row_raises_missing_columns(tmp_path):
    # Row 1 is the title: none of the headers are there.
    path = make_workbook(tmp_path, [line()])
    with pytest.raises(ValueError, match='missing columns'):
        read_one_file(path, layout_with(header_row=1, first_data_row=2))


def test_synthetic_layout_on_a_workbook_without_a_title_row_raises(tmp_path):
    # The header is on row 1, but the layout looks on row 2 (a payment line).
    path = make_workbook(tmp_path, [line(), line('0000002')], title_rows=())
    with pytest.raises(ValueError, match='missing columns'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_layout_header_lower_down_reads_a_workbook_with_two_title_rows(tmp_path):
    path = make_workbook(tmp_path, [line('0000001', 10.0)],
                         title_rows=(('MERCURY COLLECTIONS',), ('June 2025',)))
    table = read_one_file(path, layout_with(header_row=3, first_data_row=4))
    assert list(table['customer']) == ['0000001']
    assert list(table['source_row']) == [4]


def note_row(text):
    """A row with text in the company-code column and nothing else: not a payment line."""
    return (text,) + (None,) * (len(HEADER) - 1)


def test_layout_first_data_row_skips_the_rows_between_header_and_data(tmp_path):
    # Row 3 is a note under the header; the lines start on row 4.
    path = make_workbook(tmp_path, [note_row('(amounts in TRY)'), line('0000001', 10.0)],
                         grand_total=10.0)
    table = read_one_file(path, layout_with(first_data_row=4))
    assert list(table['customer']) == ['0000001']
    assert list(table['source_row']) == [4]


def test_layout_first_data_row_too_early_reads_the_note_and_raises(tmp_path):
    # The same file with first_data_row=3: the note is read as a payment line.
    path = make_workbook(tmp_path, [note_row('(amounts in TRY)'), line('0000001', 10.0)],
                         grand_total=10.0)
    with pytest.raises(ValueError, match='row 3'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_layout_header_row_past_the_end_of_the_sheet_raises_naming_the_row(tmp_path):
    path = make_workbook(tmp_path, [line()])
    with pytest.raises(ValueError, match=r'no header row \(row 50\)'):
        read_one_file(path, layout_with(header_row=50, first_data_row=51))


TURKISH_HEADERS = {
    'ŞİRKET KODU': 'company_code',
    'MÜŞTERİ': 'customer',
    'MÜŞTERİ ADI': 'customer_name',
    'BELGE TARİHİ': 'document_date',
    'BANKA': 'bank',
    'SAP BELGE NO': 'sap_document_no',
    'E-FATURA NO': 'invoice_no',
    'BELGE PARA BİRİMİ TUTARI': 'amount_original',
    'BELGE PARA BİRİMİ': 'currency_original',
    'YEREL PARA BİRİMİ TUTARI': 'amount_try',
    'YEREL PARA BİRİMİ': 'currency_local',
}
# In HEADER's order, so `line` and `total_row` put each value under the right header.
TURKISH_HEADER = tuple(TURKISH_HEADERS)
TURKISH_LAYOUT = CollectionsLayout(file_name_date_format='%d%m%Y', header_row=2,
                                   first_data_row=3, headers=TURKISH_HEADERS)


def test_turkish_test_header_is_in_the_order_of_the_english_one():
    assert [TURKISH_HEADERS[text] for text in TURKISH_HEADER] == \
        [SYNTHETIC_HEADERS[text] for text in HEADER]


def test_turkish_layout_reads_a_workbook_with_turkish_headers(tmp_path):
    path = make_workbook(tmp_path, [line('0012345', 250.0, bank='TEST BANK')],
                         header=TURKISH_HEADER)
    record = read_one_file(path, TURKISH_LAYOUT).iloc[0]
    assert record['company_code'] == 'TR02'
    assert record['customer'] == '0012345'
    assert record['customer_name'] == 'ACME TEST A.S.'
    assert record['bank'] == 'TEST BANK'
    assert record['amount_try'] == 250.0
    assert record['currency_local'] == 'TRY'


def test_turkish_layout_matches_lower_case_turkish_headers(tmp_path):
    # 'MÜŞTERİ' in the settings, 'müşteri' in the file: the dotted capital I must fold.
    lower = tuple(text.lower() for text in TURKISH_HEADER)
    path = make_workbook(tmp_path, [line('0012345', 250.0)], header=lower)
    record = read_one_file(path, TURKISH_LAYOUT).iloc[0]
    assert record['customer'] == '0012345'
    assert record['customer_name'] == 'ACME TEST A.S.'
    assert record['document_date'] == pd.Timestamp(2025, 6, 2)


def test_turkish_layout_on_an_english_workbook_raises_missing_columns(tmp_path):
    path = make_workbook(tmp_path, [line()])
    with pytest.raises(ValueError, match='missing columns'):
        read_one_file(path, TURKISH_LAYOUT)


def test_english_layout_on_a_turkish_workbook_raises_missing_columns(tmp_path):
    path = make_workbook(tmp_path, [line()], header=TURKISH_HEADER)
    with pytest.raises(ValueError, match='missing columns'):
        read_one_file(path, SYNTHETIC_LAYOUT)


def test_layout_with_english_and_turkish_aliases_reads_either_workbook(tmp_path):
    both = layout_with(headers={**SYNTHETIC_HEADERS, **TURKISH_HEADERS})
    english_path = make_workbook(tmp_path, [line('0000001', 5.0)], name='02062025')
    turkish_path = make_workbook(tmp_path, [line('0000001', 5.0)], name='03062025',
                                 header=TURKISH_HEADER)
    english = read_one_file(english_path, both)
    turkish = read_one_file(turkish_path, both)
    assert list(english.columns) == list(turkish.columns) == LINE_COLUMNS
    assert english.drop(columns='source_file').equals(turkish.drop(columns='source_file'))


def test_layout_with_aliases_raises_when_a_file_carries_both_spellings(tmp_path):
    both = layout_with(headers={**SYNTHETIC_HEADERS, 'MÜŞTERİ': 'customer'})
    header = HEADER + ('MÜŞTERİ',)
    row = line() + ('0012345',)
    grand = list(total_row('GRAND TOTAL', 100.0)) + [None]
    path = make_workbook(tmp_path, [row], header=header, last_rows=[tuple(grand)])
    with pytest.raises(ValueError, match="field 'customer' is in both column B and column L"):
        read_one_file(path, both)


def test_layout_without_a_bank_header_ignores_the_bank_column(tmp_path):
    headers = {text: field for text, field in SYNTHETIC_HEADERS.items() if field != 'bank'}
    path = make_workbook(tmp_path, [line(bank='TEST BANK')])
    table = read_one_file(path, layout_with(headers=headers))
    assert table['bank'].iloc[0] is None


def test_layout_file_name_format_does_not_matter_to_read_one_file(tmp_path):
    # read_one_file reads the file it is given; only list_daily_files uses the name format.
    path = make_workbook(tmp_path, [line()])
    pd.testing.assert_frame_equal(read_one_file(path, layout_with(file_name_date_format='%Y%m%d')),
                                  read_one_file(path, SYNTHETIC_LAYOUT))


def test_read_one_file_does_not_modify_the_layout(tmp_path):
    headers = dict(TURKISH_HEADERS)
    layout = layout_with(headers=headers)
    read_one_file(make_workbook(tmp_path, [line()], header=TURKISH_HEADER), layout)
    assert layout.headers == TURKISH_HEADERS
    assert list(layout.headers) == list(TURKISH_HEADERS)


def test_read_one_file_requires_the_layout(tmp_path):
    with pytest.raises(TypeError):
        read_one_file(make_workbook(tmp_path, [line()]))


# --- the synthetic corpus --------------------------------------------------------------

@pytest.mark.skipif(not SYNTHETIC_FIRST_FILE.is_file(), reason='synthetic corpus not installed')
def test_synthetic_first_file_has_33_lines_totalling_18_899_496_71():
    table = read_one_file(SYNTHETIC_FIRST_FILE, SYNTHETIC_LAYOUT)
    assert len(table) == 33
    assert table['amount_try'].sum() == pytest.approx(18_899_496.71, abs=0.01)
    assert list(table.columns) == LINE_COLUMNS


@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason='synthetic corpus not installed')
def test_synthetic_corpus_reads_27311_lines_from_all_files():
    """SLOW (reads all 306 workbooks)."""
    files = list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT)
    assert len(files) == 306
    total_lines = sum(len(read_one_file(path, SYNTHETIC_LAYOUT)) for path in files.values())
    assert total_lines == 27_311
