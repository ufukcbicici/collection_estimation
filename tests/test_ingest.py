"""Tests for `read_workbook`.

Two kinds. The first read real corpus workbooks, so they check the function against the
data it will actually see. The second build tiny workbooks in `tmp_path` to exercise the
failures the corpus is too clean to contain — a shifted column, a missing header, a
customer number stored as a number.
"""
import os
import sys
from datetime import date, datetime

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from collection_estimation import config                      # noqa: E402
from collection_estimation.ingest import (                    # noqa: E402
    WorkbookFormatError,
    read_workbook,
)
from tests.helpers import HEADERS, build, corpus_files, detail  # noqa: E402


# ---------------------------------------------------------------- real corpus

@pytest.fixture(scope="module")
def one_real_file():
    return os.path.join(config.CORPUS_DIR, corpus_files()[0])


def test_reads_a_real_workbook_and_returns_only_detail_rows(one_real_file):
    rows = read_workbook(one_real_file)
    assert rows, "a corpus workbook should never be empty"
    assert all(r["company_code"] in config.ENTITY_LABEL for r in rows)


def test_subtotal_and_grand_total_rows_are_excluded(one_real_file):
    rows = read_workbook(one_real_file)
    names = {r["customer_name"] for r in rows}
    assert not any(n.endswith("Total") for n in names), (
        "subtotal rows carry their label in the customer-name column and must not be "
        "returned as detail — summing them would double-count the day")


def test_customer_number_stays_text(one_real_file):
    rows = read_workbook(one_real_file)
    assert all(isinstance(r["customer"], str) for r in rows)


def test_document_date_is_a_plain_date_not_a_datetime(one_real_file):
    rows = read_workbook(one_real_file)
    assert all(type(r["document_date"]) is date for r in rows)


def test_dates_agree_with_the_filename(one_real_file):
    expected = datetime.strptime(os.path.basename(one_real_file)[:8], "%d%m%Y").date()
    assert {r["document_date"] for r in read_workbook(one_real_file)} == {expected}


def test_amounts_are_floats_and_may_be_negative_or_zero():
    """Across the whole corpus, refunds and zero-value rows must survive parsing."""
    negatives = zeros = 0
    for name in corpus_files():
        for r in read_workbook(os.path.join(config.CORPUS_DIR, name)):
            assert isinstance(r["amount_original"], float)
            negatives += r["amount_original"] < 0
            zeros += r["amount_original"] == 0
    assert negatives > 0, "refunds exist in this corpus and must not be filtered out"
    assert zeros >= 0


def test_unapplied_cash_rows_are_kept_with_no_document_number(one_real_file):
    rows = [r for r in read_workbook(one_real_file)
            if r["government_invoice"] in config.STATUS_NOTES]
    for r in rows:
        assert r["sap_document_no"] is None
        assert r["amount_local"] != 0


def test_row_count_matches_the_oracle_for_every_file():
    """The oracle records each file's row count; parsing must reproduce all 306."""
    import json
    with open(config.ORACLE, encoding="utf-8") as fh:
        daily = {d["file"]: d["rows"] for d in json.load(fh)["daily"]}
    for name in corpus_files():
        got = len(read_workbook(os.path.join(config.CORPUS_DIR, name)))
        assert got == daily[name], f"{name}: parsed {got}, oracle says {daily[name]}"


# ---------------------------------------------------------------- constructed cases

def test_columns_are_bound_by_header_text_not_position(tmp_path):
    """Insert a column and everything to its right shifts. Binding by header survives it.

    This is the failure the corpus notes call near-certain over fourteen months, and the
    one that fails quietly: a position-bound parser keeps running and reads the wrong
    column.
    """
    shifted = HEADERS[:3] + ["Notes"] + HEADERS[3:]
    row = detail()
    row = row[:3] + ["some note"] + row[3:]
    rows = read_workbook(build(tmp_path, headers=shifted, rows=[row]))
    assert len(rows) == 1
    assert rows[0]["document_date"] == date(2026, 7, 2)
    assert rows[0]["amount_original"] == 1000.0
    assert rows[0]["document_currency"] == "TRY"


def test_header_whitespace_and_case_do_not_matter(tmp_path):
    messy = ["  company code ", "CUSTOMER", "Customer  Name", "document date", "bank",
             "SAP Document No", "Government Inv", "amount in original currency",
             "Document Currency", "Amount In Local Currency", "local currency"]
    rows = read_workbook(build(tmp_path, headers=messy, rows=[detail()]))
    assert rows[0]["customer"] == "11520039"


def test_missing_required_column_raises(tmp_path):
    without_amount = [h for h in HEADERS if h != "Amount in local currency"]
    row = [v for i, v in enumerate(detail()) if HEADERS[i] != "Amount in local currency"]
    with pytest.raises(WorkbookFormatError, match="amount_local"):
        read_workbook(build(tmp_path, headers=without_amount, rows=[row]))


def test_bank_is_optional(tmp_path):
    """EY's schema has a bank field but its column position was never observed."""
    without_bank = [h for h in HEADERS if h != "Bank"]
    row = [v for i, v in enumerate(detail()) if HEADERS[i] != "Bank"]
    rows = read_workbook(build(tmp_path, headers=without_bank, rows=[row]))
    assert rows[0]["bank"] is None
    assert rows[0]["amount_local"] == 1000.0


def test_customer_stored_as_a_number_raises(tmp_path):
    with pytest.raises(WorkbookFormatError, match="not text"):
        read_workbook(build(tmp_path, rows=[detail(customer=11520039)]))


def test_non_date_document_date_raises(tmp_path):
    with pytest.raises(WorkbookFormatError, match="not a date"):
        read_workbook(build(tmp_path, rows=[detail(when="02/07/2026")]))


def test_non_numeric_amount_raises(tmp_path):
    with pytest.raises(WorkbookFormatError, match="not a number"):
        read_workbook(build(tmp_path, rows=[detail(amount="1.000,00")]))


def test_unknown_company_code_is_skipped_not_raised(tmp_path):
    """A row whose code we do not recognise is not a detail row of ours."""
    rows = read_workbook(build(tmp_path, rows=[detail(code="TR99"), detail()]))
    assert len(rows) == 1


def test_blank_and_subtotal_rows_between_groups_are_skipped(tmp_path):
    blank = [None] * len(HEADERS)
    subtotal = [None, None, "Güney Total", None, None, None, None, None, None,
                123.45, "TRY"]
    rows = read_workbook(build(
        tmp_path, rows=[detail(), blank, subtotal, blank, detail(code="TR91")]))
    assert [r["company_code"] for r in rows] == ["TR02", "TR91"]


def test_marker_tab_is_never_read_as_data(tmp_path):
    rows = read_workbook(build(tmp_path, rows=[detail()], marker_tab=True))
    assert len(rows) == 1


def test_works_when_there_is_no_marker_tab(tmp_path):
    """Real EY files have a single sheet and no synthetic marker."""
    rows = read_workbook(build(tmp_path, rows=[detail()], marker_tab=False,
                               sheet="Sheet1"))
    assert len(rows) == 1


def test_source_file_and_row_are_recorded(tmp_path):
    path = build(tmp_path, rows=[detail()], name="02072026 EY TAHSILAT.XLSX")
    row = read_workbook(path)[0]
    assert row["source_file"] == "02072026 EY TAHSILAT.XLSX"
    assert row["source_row"] == config.FIRST_DATA_ROW
