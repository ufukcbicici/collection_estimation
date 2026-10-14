"""Tests for `read_all_files` and `check_dates_match` in `hurdle/read_collections.py`.

Each day is a small workbook built in `tmp_path` by `make_workbook` (see `workbooks.py`),
named like the real files ('02062025 EY TAHSILAT.XLSX'), and found by `list_daily_files`.
Customers are fake.
"""
import dataclasses
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.read_collections import (
    LINE_COLUMNS,
    TEXT_COLUMNS,
    check_dates_match,
    list_daily_files,
    read_all_files,
    read_one_file,
)
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, line, make_workbook

DATE_FORMAT = SYNTHETIC_LAYOUT.file_name_date_format  # "%d%m%Y"

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"


def make_day(folder, day, *lines, **options):
    """Write the workbook for `day`; `lines` are (customer, amount) pairs dated `day`."""
    when = datetime(day.year, day.month, day.day)
    rows = [line(customer, amount, when=when) for customer, amount in lines]
    return make_workbook(folder, rows, name=day.strftime('%d%m%Y'), **options)


def three_days(folder):
    """2 June (two lines), 3 June (one line), 1 July (two lines), 2025."""
    make_day(folder, date(2025, 6, 2), ('0000001', 10.0), ('0000002', 20.0))
    make_day(folder, date(2025, 6, 3), ('0000003', 30.0))
    make_day(folder, date(2025, 7, 1), ('0000004', 40.0), ('0000005', 50.0))
    return list_daily_files(folder, DATE_FORMAT)


def wrong_dates_part(message):
    """The part of a date-mismatch message after 'lines carry'."""
    return message.split('lines carry', 1)[1]


def names_date(text, day):
    """True if `text` names `day`, as '2025-06-03' or as 'datetime.date(2025, 6, 3)'."""
    return str(day) in text or repr(day) in text


# --- the layout is passed to every file -------------------------------------------------

def test_layout_with_header_on_row_1_and_yyyymmdd_names_reads_every_file(tmp_path):
    # Another layout end to end: names '20250602 ...', no title row, lines from row 2.
    layout = dataclasses.replace(SYNTHETIC_LAYOUT, file_name_date_format='%Y%m%d',
                                 header_row=1, first_data_row=2)
    for day, customer in ((date(2025, 6, 2), '0000001'), (date(2025, 6, 3), '0000002')):
        make_workbook(tmp_path, [line(customer, 10.0, when=datetime(day.year, day.month,
                                                                    day.day))],
                      name=day.strftime('%Y%m%d'), title_rows=())
    daily_files = list_daily_files(tmp_path, layout.file_name_date_format)
    assert list(daily_files) == [date(2025, 6, 2), date(2025, 6, 3)]
    table = read_all_files(daily_files, layout)
    assert list(table['customer']) == ['0000001', '0000002']
    assert list(table['source_row']) == [2, 2]
    assert list(table['source_file']) == ['20250602 EY TAHSILAT.XLSX',
                                          '20250603 EY TAHSILAT.XLSX']


def test_a_layout_that_fits_no_file_raises_on_the_first_file(tmp_path):
    layout = dataclasses.replace(SYNTHETIC_LAYOUT, header_row=1, first_data_row=2)
    with pytest.raises(ValueError, match='missing columns'):
        read_all_files(three_days(tmp_path), layout)


def test_read_all_files_requires_the_layout(tmp_path):
    with pytest.raises(TypeError):
        read_all_files(three_days(tmp_path))


# --- stacking the days -----------------------------------------------------------------

def test_stacks_every_line_of_every_day(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    assert len(table) == 5


def test_oldest_day_first_then_sheet_order(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    assert list(table['customer']) == ['0000001', '0000002', '0000003', '0000004', '0000005']


def test_index_runs_from_zero_without_repeats(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    assert list(table.index) == [0, 1, 2, 3, 4]


def test_columns_are_exactly_line_columns(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    assert list(table.columns) == LINE_COLUMNS


def test_source_file_tells_the_days_apart(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    assert list(table['source_file']) == [
        '02062025 EY TAHSILAT.XLSX', '02062025 EY TAHSILAT.XLSX',
        '03062025 EY TAHSILAT.XLSX',
        '01072025 EY TAHSILAT.XLSX', '01072025 EY TAHSILAT.XLSX']


def test_source_row_restarts_in_each_file(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    assert list(table['source_row']) == [3, 4, 3, 3, 4]


def test_order_follows_the_dates_not_the_file_names(tmp_path):
    # By name '01072025' sorts before '02062025'; by date 1 July comes after 2 June.
    make_day(tmp_path, date(2025, 7, 1), ('0000002', 20.0))
    make_day(tmp_path, date(2025, 6, 2), ('0000001', 10.0))
    table = read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)
    assert list(table['customer']) == ['0000001', '0000002']


def test_order_follows_the_dict_order(tmp_path):
    june = make_day(tmp_path, date(2025, 6, 2), ('0000001', 10.0))
    july = make_day(tmp_path, date(2025, 7, 1), ('0000002', 20.0))
    table = read_all_files({date(2025, 7, 1): july, date(2025, 6, 2): june}, SYNTHETIC_LAYOUT)
    assert list(table['customer']) == ['0000002', '0000001']


def test_one_file_gives_the_same_lines_as_read_one_file(tmp_path):
    path = make_day(tmp_path, date(2025, 6, 2), ('0000001', 10.0), ('0000002', 20.0))
    pd.testing.assert_frame_equal(read_all_files({date(2025, 6, 2): path}, SYNTHETIC_LAYOUT),
                                  read_one_file(path, SYNTHETIC_LAYOUT))


def test_amounts_add_up_across_days(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    assert table['amount_try'].sum() == 150.0


# --- column types after joining --------------------------------------------------------

def test_missing_text_stays_none_after_joining(tmp_path):
    june_2 = datetime(2025, 6, 2)
    june_3 = datetime(2025, 6, 3)
    make_workbook(tmp_path, [line('0000001', 10.0, when=june_2, sap=None)], name='02062025')
    make_workbook(tmp_path, [line('0000002', 20.0, when=june_3, invoice=None, bank=None)],
                  name='03062025')
    table = read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)
    assert table['sap_document_no'].iloc[0] is None
    assert table['invoice_no'].iloc[1] is None
    assert table['bank'].iloc[1] is None


def test_no_text_value_is_a_nan_float_after_joining(tmp_path):
    # A column empty on one day and filled on another is where pandas likes to slip in NaN.
    june_2 = datetime(2025, 6, 2)
    june_3 = datetime(2025, 6, 3)
    make_workbook(tmp_path, [line('0000001', 10.0, when=june_2, sap=None, bank=None)],
                  name='02062025')
    make_workbook(tmp_path, [line('0000002', 20.0, when=june_3)], name='03062025')
    table = read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)
    for column in TEXT_COLUMNS:
        for value in table[column]:
            assert not isinstance(value, float), (column, value)


def test_text_columns_stay_object_dtype(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    for column in TEXT_COLUMNS:
        assert table[column].dtype == object, column


def test_amounts_are_float64(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    assert table['amount_try'].dtype == 'float64'
    assert table['amount_original'].dtype == 'float64'


def test_document_date_is_datetime64(tmp_path):
    table = read_all_files(three_days(tmp_path), SYNTHETIC_LAYOUT)
    assert pd.api.types.is_datetime64_any_dtype(table['document_date'])


# --- a day with no payment lines -------------------------------------------------------

def days_with_an_empty_one_between(folder):
    make_day(folder, date(2025, 6, 2), ('0000001', 10.0))
    make_day(folder, date(2025, 6, 3), grand_total=0.0)  # header and a zero grand total
    make_day(folder, date(2025, 6, 4), ('0000002', 20.0))
    return list_daily_files(folder, DATE_FORMAT)


def test_empty_day_contributes_no_lines(tmp_path):
    table = read_all_files(days_with_an_empty_one_between(tmp_path), SYNTHETIC_LAYOUT)
    assert list(table['customer']) == ['0000001', '0000002']
    assert '03062025 EY TAHSILAT.XLSX' not in set(table['source_file'])


def test_empty_day_keeps_amounts_float64(tmp_path):
    table = read_all_files(days_with_an_empty_one_between(tmp_path), SYNTHETIC_LAYOUT)
    assert table['amount_try'].dtype == 'float64'
    assert table['amount_original'].dtype == 'float64'


def test_empty_day_keeps_document_date_datetime64(tmp_path):
    table = read_all_files(days_with_an_empty_one_between(tmp_path), SYNTHETIC_LAYOUT)
    assert pd.api.types.is_datetime64_any_dtype(table['document_date'])


def test_empty_day_keeps_source_row_integer(tmp_path):
    table = read_all_files(days_with_an_empty_one_between(tmp_path), SYNTHETIC_LAYOUT)
    assert pd.api.types.is_integer_dtype(table['source_row'])


def test_empty_day_keeps_text_columns_object_dtype(tmp_path):
    table = read_all_files(days_with_an_empty_one_between(tmp_path), SYNTHETIC_LAYOUT)
    for column in TEXT_COLUMNS:
        assert table[column].dtype == object, column


def test_empty_day_first_keeps_amounts_float64(tmp_path):
    make_day(tmp_path, date(2025, 6, 2), grand_total=0.0)
    make_day(tmp_path, date(2025, 6, 3), ('0000001', 10.0))
    table = read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)
    assert table['amount_try'].dtype == 'float64'


def test_only_empty_days_give_an_empty_table_with_line_columns(tmp_path):
    make_day(tmp_path, date(2025, 6, 2), grand_total=0.0)
    make_day(tmp_path, date(2025, 6, 3), grand_total=0.0)
    table = read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)
    assert table.empty
    assert list(table.columns) == LINE_COLUMNS


# --- errors ----------------------------------------------------------------------------

def test_empty_dict_raises():
    with pytest.raises(ValueError, match='no daily files'):
        read_all_files({}, SYNTHETIC_LAYOUT)


def test_lines_dated_another_day_raise_naming_file_and_both_dates(tmp_path):
    make_workbook(tmp_path, [line('0000001', 10.0, when=datetime(2025, 6, 3))],
                  name='02062025')
    with pytest.raises(ValueError) as error:
        read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)
    message = str(error.value)
    assert '02062025 EY TAHSILAT.XLSX' in message
    assert '2025-06-02' in message
    assert names_date(wrong_dates_part(message), date(2025, 6, 3))


def test_file_mixing_its_date_and_another_lists_only_the_wrong_one(tmp_path):
    lines = [line('0000001', 10.0, when=datetime(2025, 6, 2)),
             line('0000002', 20.0, when=datetime(2025, 6, 5))]
    make_workbook(tmp_path, lines, name='02062025')
    with pytest.raises(ValueError) as error:
        read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)
    wrong = wrong_dates_part(str(error.value))
    assert names_date(wrong, date(2025, 6, 5))
    assert not names_date(wrong, date(2025, 6, 2))


def test_two_wrong_dates_are_both_listed(tmp_path):
    lines = [line('0000001', 10.0, when=datetime(2025, 6, 9)),
             line('0000002', 20.0, when=datetime(2025, 6, 5))]
    make_workbook(tmp_path, lines, name='02062025')
    with pytest.raises(ValueError) as error:
        read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)
    wrong = wrong_dates_part(str(error.value))
    assert names_date(wrong, date(2025, 6, 5))
    assert names_date(wrong, date(2025, 6, 9))


def test_date_mismatch_in_a_later_file_names_that_file(tmp_path):
    make_day(tmp_path, date(2025, 6, 2), ('0000001', 10.0))
    make_workbook(tmp_path, [line('0000002', 20.0, when=datetime(2025, 6, 2))],
                  name='03062025')
    with pytest.raises(ValueError, match=r'03062025 EY TAHSILAT\.XLSX'):
        read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)


def test_tie_out_error_in_one_file_propagates(tmp_path):
    make_day(tmp_path, date(2025, 6, 2), ('0000001', 10.0))
    make_day(tmp_path, date(2025, 6, 3), ('0000002', 20.0), grand_total=99.0)
    with pytest.raises(ValueError, match=r'03062025 EY TAHSILAT\.XLSX.*99\.00 TRY'):
        read_all_files(list_daily_files(tmp_path, DATE_FORMAT), SYNTHETIC_LAYOUT)


# --- check_dates_match on its own ------------------------------------------------------

def dated_lines(*days):
    return pd.DataFrame({'document_date': pd.to_datetime(list(days))})


def test_check_dates_match_passes_silently_when_all_dates_match(capsys):
    lines = dated_lines(date(2025, 6, 2), date(2025, 6, 2))
    assert check_dates_match(lines, date(2025, 6, 2), 'f.xlsx') is None
    assert capsys.readouterr().out == ''


def test_check_dates_match_passes_an_empty_table():
    check_dates_match(dated_lines(), date(2025, 6, 2), 'f.xlsx')


def test_check_dates_match_passes_the_empty_table_read_one_file_returns(tmp_path):
    lines = read_one_file(make_day(tmp_path, date(2025, 6, 2), grand_total=0.0), SYNTHETIC_LAYOUT)
    check_dates_match(lines, date(2025, 6, 2), 'f.xlsx')


def test_check_dates_match_raises_naming_file_and_dates():
    lines = dated_lines(date(2025, 6, 2), date(2025, 6, 4))
    with pytest.raises(ValueError, match=re.escape('f.xlsx: the file name says 2025-06-02')):
        check_dates_match(lines, date(2025, 6, 2), 'f.xlsx')


def test_check_dates_match_lists_wrong_dates_oldest_first():
    lines = dated_lines(date(2025, 6, 9), date(2025, 6, 4), date(2025, 6, 9))
    with pytest.raises(ValueError) as error:
        check_dates_match(lines, date(2025, 6, 2), 'f.xlsx')
    wrong = wrong_dates_part(str(error.value))
    assert position_of(wrong, date(2025, 6, 4)) < position_of(wrong, date(2025, 6, 9))


def position_of(text, day):
    """Where `day` first appears in `text`, in either form `names_date` accepts."""
    positions = [text.find(form) for form in (str(day), repr(day)) if form in text]
    return min(positions)


# --- progress --------------------------------------------------------------------------

def make_days(folder, count):
    """`count` one-line workbooks on consecutive days from 2 June 2025."""
    for offset in range(count):
        make_day(folder, date(2025, 6, 2) + timedelta(days=offset), ('0000001', 1.0))
    return list_daily_files(folder, DATE_FORMAT)


def test_progress_line_after_every_50_files(tmp_path, capsys):
    table = read_all_files(make_days(tmp_path, 51), SYNTHETIC_LAYOUT)
    assert len(table) == 51
    assert capsys.readouterr().out == '  ... read 50 of 51 files\n'


def test_no_progress_line_under_50_files(tmp_path, capsys):
    read_all_files(make_days(tmp_path, 3), SYNTHETIC_LAYOUT)
    assert capsys.readouterr().out == ''


# --- the synthetic corpus --------------------------------------------------------------

@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason='synthetic corpus not installed')
def test_synthetic_corpus_all_files_give_27311_lines_and_the_known_total():
    """SLOW (reads all 306 workbooks)."""
    table = read_all_files(list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT), SYNTHETIC_LAYOUT)
    assert len(table) == 27_311
    assert table['amount_try'].sum() == pytest.approx(8_647_602_968.89, abs=0.01)
    assert table['document_date'].nunique() == 306
    assert list(table.index) == list(range(27_311))
