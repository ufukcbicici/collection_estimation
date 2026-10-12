"""Tests for `hurdle/read_collections.py`: finding the daily workbooks and their dates, and
finding the columns in a workbook's header row.

`list_daily_files` never opens the files, so the test files are empty.
"""
from datetime import date
from pathlib import Path

import pytest

from collection_estimation.hurdle import read_collections
from collection_estimation.hurdle.read_collections import (
    date_from_file_name,
    find_columns,
    list_daily_files,
    simplify_header,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"


def make_files(folder, *names):
    for name in names:
        (folder / name).write_bytes(b"")


# --- date_from_file_name ---------------------------------------------------------------

def test_date_from_file_name_reads_ddmmyyyy():
    assert date_from_file_name("02062025 EY TAHSILAT.XLSX") == date(2025, 6, 2)


def test_date_from_file_name_reads_day_before_month():
    assert date_from_file_name("01072026 EY TAHSILAT.XLSX") == date(2026, 7, 1)


def test_date_from_file_name_rejects_name_without_date():
    with pytest.raises(ValueError, match="notes.xlsx"):
        date_from_file_name("notes.xlsx")


def test_date_from_file_name_rejects_impossible_day():
    with pytest.raises(ValueError, match="32012026 x.xlsx"):
        date_from_file_name("32012026 x.xlsx")


def test_date_from_file_name_rejects_impossible_month():
    with pytest.raises(ValueError):
        date_from_file_name("01132026 x.xlsx")


def test_date_from_file_name_rejects_yyyymmdd_order():
    with pytest.raises(ValueError):
        date_from_file_name("20250602 EY TAHSILAT.XLSX")


# --- list_daily_files ------------------------------------------------------------------

def test_empty_folder_gives_empty_dict(tmp_path):
    assert list_daily_files(tmp_path) == {}


def test_returns_date_to_path(tmp_path):
    make_files(tmp_path, "02062025 EY TAHSILAT.XLSX")
    assert list_daily_files(tmp_path) == {date(2025, 6, 2): tmp_path / "02062025 EY TAHSILAT.XLSX"}


def test_accepts_a_string_folder(tmp_path):
    make_files(tmp_path, "02062025 EY TAHSILAT.XLSX")
    assert list(list_daily_files(str(tmp_path))) == [date(2025, 6, 2)]


def test_sorted_by_date_not_by_name(tmp_path):
    # By name, "01072026" would come before "02062025"; by date it is a year later.
    make_files(tmp_path, "01072026 EY TAHSILAT.XLSX", "02062025 EY TAHSILAT.XLSX",
               "15122025 EY TAHSILAT.XLSX")
    assert list(list_daily_files(tmp_path)) == [
        date(2025, 6, 2), date(2025, 12, 15), date(2026, 7, 1)]


def test_suffix_check_ignores_case(tmp_path):
    make_files(tmp_path, "02062025 EY TAHSILAT.XLSX", "03062025 EY TAHSILAT.xlsx")
    assert list(list_daily_files(tmp_path)) == [date(2025, 6, 2), date(2025, 6, 3)]


def test_skips_excel_lock_files(tmp_path):
    make_files(tmp_path, "02062025 EY TAHSILAT.XLSX", "~$02062025 EY TAHSILAT.XLSX")
    assert list_daily_files(tmp_path) == {date(2025, 6, 2): tmp_path / "02062025 EY TAHSILAT.XLSX"}


def test_skips_non_xlsx_files(tmp_path):
    make_files(tmp_path, "02062025 EY TAHSILAT.XLSX", "README.md",
               "03062025 EY TAHSILAT.csv", "04062025 EY TAHSILAT.xls")
    assert list(list_daily_files(tmp_path)) == [date(2025, 6, 2)]


def test_two_files_for_one_date_raise_naming_both(tmp_path):
    make_files(tmp_path, "02062025 EY TAHSILAT.XLSX", "02062025 EY TAHSILAT v2.xlsx")
    with pytest.raises(ValueError) as error:
        list_daily_files(tmp_path)
    assert "02062025 EY TAHSILAT.XLSX" in str(error.value)
    assert "02062025 EY TAHSILAT v2.xlsx" in str(error.value)


def test_xlsx_without_a_date_raises_with_its_name(tmp_path):
    make_files(tmp_path, "02062025 EY TAHSILAT.XLSX", "notes.xlsx")
    with pytest.raises(ValueError, match="notes.xlsx"):
        list_daily_files(tmp_path)


def test_xlsx_with_impossible_date_raises_with_its_name(tmp_path):
    make_files(tmp_path, "32012026 x.xlsx")
    with pytest.raises(ValueError, match="32012026 x.xlsx"):
        list_daily_files(tmp_path)


def test_skips_a_folder_named_like_a_workbook(tmp_path):
    # A sub-folder named "02062025.xlsx" is not a workbook. It must be skipped, and so must
    # not clash with the real file for the same date.
    (tmp_path / "02062025.xlsx").mkdir()
    make_files(tmp_path, "02062025 EY TAHSILAT.XLSX")
    assert list_daily_files(tmp_path) == {date(2025, 6, 2): tmp_path / "02062025 EY TAHSILAT.XLSX"}


def test_folder_named_like_a_workbook_alone_gives_empty_dict(tmp_path):
    (tmp_path / "02062025.xlsx").mkdir()
    assert list_daily_files(tmp_path) == {}


# --- simplify_header -------------------------------------------------------------------

def test_simplify_header_turns_empty_cell_into_empty_text():
    assert simplify_header(None) == ""


def test_simplify_header_folds_case_and_spaces():
    assert simplify_header("  Company  CODE ") == "company code"


def test_simplify_header_folds_tabs_and_line_breaks():
    assert simplify_header("Company\tCode\n") == "company code"


def test_simplify_header_turns_a_number_into_text():
    assert simplify_header(5) == "5"


def test_simplify_header_folds_turkish_capitals():
    # 'İ' must become a plain 'i', not 'i' followed by a combining dot (U+0307).
    assert simplify_header('MÜŞTERİ') == 'müşteri'


def test_simplify_header_turkish_dotted_capital_i_becomes_plain_i():
    assert simplify_header('İ') == 'i'


# --- find_columns ----------------------------------------------------------------------

# The header row of the daily files, as on row 2 of the synthetic workbooks.
REAL_HEADER = ('Company Code', 'Customer', 'Customer Name', 'Document Date', 'Bank',
               'SAP Document No', 'Government Inv', 'Amount in Original Currency',
               'Document currency', 'Amount in local currency', 'Local Currency')

REAL_COLUMNS = {
    "company_code": 0,
    "customer": 1,
    "customer_name": 2,
    "document_date": 3,
    "bank": 4,
    "sap_document_no": 5,
    "invoice_no": 6,
    "amount_original": 7,
    "currency_original": 8,
    "amount_try": 9,
    "currency_local": 10,
}


def without(header_row, text):
    """The header row with the column `text` removed."""
    return tuple(header for header in header_row if header != text)


def test_find_columns_maps_the_real_header_row():
    assert find_columns(REAL_HEADER) == REAL_COLUMNS


def test_find_columns_finds_columns_by_text_not_position():
    shuffled = ('Local Currency', 'Customer Name', 'Amount in local currency', 'Company Code',
                'Government Inv', 'Bank', 'Document currency', 'Customer', 'SAP Document No',
                'Amount in Original Currency', 'Document Date')
    assert find_columns(shuffled) == {
        "currency_local": 0,
        "customer_name": 1,
        "amount_try": 2,
        "company_code": 3,
        "invoice_no": 4,
        "bank": 5,
        "currency_original": 6,
        "customer": 7,
        "sap_document_no": 8,
        "amount_original": 9,
        "document_date": 10,
    }


def test_find_columns_inserted_unknown_column_shifts_later_columns():
    # 'Notes' is inserted after 'Document Date': every column to its right moves one place.
    header = REAL_HEADER[:4] + ('Notes',) + REAL_HEADER[4:]
    assert find_columns(header) == {
        "company_code": 0,
        "customer": 1,
        "customer_name": 2,
        "document_date": 3,
        "bank": 5,
        "sap_document_no": 6,
        "invoice_no": 7,
        "amount_original": 8,
        "currency_original": 9,
        "amount_try": 10,
        "currency_local": 11,
    }


def test_find_columns_ignores_an_unknown_column():
    header = REAL_HEADER[:4] + ('Notes',) + REAL_HEADER[4:]
    assert "notes" not in find_columns(header)
    assert len(find_columns(header)) == 11


def test_find_columns_ignores_trailing_empty_cells():
    assert find_columns(REAL_HEADER + (None, None)) == REAL_COLUMNS


def test_find_columns_ignores_empty_cells_in_the_middle():
    header = REAL_HEADER[:3] + (None,) + REAL_HEADER[3:]
    columns = find_columns(header)
    assert columns["customer_name"] == 2
    assert columns["document_date"] == 4


def test_find_columns_folds_case_and_spaces():
    header = ('  COMPANY   code ',) + REAL_HEADER[1:]
    assert find_columns(header) == REAL_COLUMNS


def test_find_columns_bank_is_optional():
    columns = find_columns(without(REAL_HEADER, 'Bank'))
    assert "bank" not in columns
    assert columns["sap_document_no"] == 4  # moved one place left


def test_find_columns_missing_required_column_raises_naming_it():
    with pytest.raises(ValueError, match="customer_name"):
        find_columns(without(REAL_HEADER, 'Customer Name'))


def test_find_columns_missing_columns_are_all_named():
    header = without(without(REAL_HEADER, 'Government Inv'), 'Amount in local currency')
    with pytest.raises(ValueError) as error:
        find_columns(header)
    assert "invoice_no" in str(error.value)
    assert "amount_try" in str(error.value)


def test_find_columns_missing_optional_column_is_not_named_as_missing():
    # Without 'Bank' and 'Customer', only 'customer' is missing.
    header = without(without(REAL_HEADER, 'Bank'), 'Customer')
    with pytest.raises(ValueError) as error:
        find_columns(header)
    assert "'customer'" in str(error.value)
    assert "'bank'" not in str(error.value)


def test_find_columns_empty_header_row_raises():
    with pytest.raises(ValueError, match="missing columns"):
        find_columns(())


def test_find_columns_header_twice_raises_naming_field_and_both_columns():
    # 'Customer' is in column B (position 1) and again in column L (position 11).
    with pytest.raises(ValueError) as error:
        find_columns(REAL_HEADER + ('Customer',))
    assert str(error.value) == "field 'customer' is in both column B and column L"


def test_find_columns_header_twice_uses_excel_column_letters():
    # 'Customer' at position 1 and again at position 4 -> columns B and E.
    header = REAL_HEADER[:4] + ('Customer',) + REAL_HEADER[4:]
    with pytest.raises(ValueError) as error:
        find_columns(header)
    assert str(error.value) == "field 'customer' is in both column B and column E"


def test_find_columns_header_twice_after_folding_raises():
    # ' customer ' is the same header as 'Customer' once case and spaces are folded.
    with pytest.raises(ValueError) as error:
        find_columns(REAL_HEADER + (' customer ',))
    assert str(error.value) == "field 'customer' is in both column B and column L"


def test_find_columns_optional_header_twice_raises():
    # 'Bank' is in column E (position 4); 'BANK' is added in column L.
    with pytest.raises(ValueError) as error:
        find_columns(REAL_HEADER + ('BANK',))
    assert str(error.value) == "field 'bank' is in both column E and column L"


# Turkish headers: casefold() turns 'İ' into 'i' plus a combining dot, which must be dropped.

def with_customer_header(text):
    """The real header row with 'Customer' replaced by `text`."""
    return tuple(text if header == 'Customer' else header for header in REAL_HEADER)


def test_find_columns_turkish_key_in_capitals_matches_lower_case_header(monkeypatch):
    monkeypatch.setitem(read_collections.HEADER_TO_FIELD, 'MÜŞTERİ', 'customer')
    columns = find_columns(with_customer_header('müşteri'))
    assert columns["customer"] == 1


def test_find_columns_turkish_key_in_lower_case_matches_capital_header(monkeypatch):
    monkeypatch.setitem(read_collections.HEADER_TO_FIELD, 'müşteri', 'customer')
    columns = find_columns(with_customer_header('MÜŞTERİ'))
    assert columns["customer"] == 1


def test_find_columns_unknown_header_twice_is_fine():
    # Columns we don't use may repeat; only our own fields must be unique.
    assert find_columns(REAL_HEADER + ('Notes', 'Notes')) == REAL_COLUMNS


# --- the real synthetic corpus ---------------------------------------------------------

@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason="synthetic corpus not installed")
def test_synthetic_corpus_has_306_days_from_2025_06_02_to_2026_08_28():
    files = list_daily_files(SYNTHETIC_CORPUS)
    dates = list(files)
    assert len(files) == 306
    assert dates[0] == date(2025, 6, 2)
    assert dates[-1] == date(2026, 8, 28)
    assert dates == sorted(dates)


SYNTHETIC_FIRST_FILE = SYNTHETIC_CORPUS / "02062025 EY TAHSILAT.XLSX"


@pytest.mark.skipif(not SYNTHETIC_FIRST_FILE.is_file(), reason="synthetic corpus not installed")
def test_synthetic_workbook_header_on_row_2_has_all_11_fields():
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.load_workbook(SYNTHETIC_FIRST_FILE, read_only=True)
    try:
        sheet = workbook["02062025"]
        header_row = next(sheet.iter_rows(min_row=2, max_row=2, values_only=True))
    finally:
        workbook.close()
    assert set(find_columns(header_row)) == set(REAL_COLUMNS)
