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


def test_slow_synthetic_ledger_earlier_as_of_raises(synthetic_ledger):
    # Same check read_ledger runs, applied to the table already read (avoids a second read).
    check_nothing_after(synthetic_ledger, date(2026, 8, 28), LEDGER_FILE.name)
    with pytest.raises(ValueError, match='after the extract date 2026-08-01'):
        check_nothing_after(synthetic_ledger, date(2026, 8, 1), LEDGER_FILE.name)
