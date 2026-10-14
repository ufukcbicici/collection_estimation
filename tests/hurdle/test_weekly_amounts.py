"""Tests for `weekly_amounts` and `check_weekly_totals` in `hurdle/series.py`.

Inputs are small tables of payment lines built by hand with `lines_table`, with the columns
`read_one_file -> add_keys -> add_week_index` produce that these functions use. Customers are
fake. The synthetic-corpus tests read the daily files directly (never `data/interim/`).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.keys import add_keys
from collection_estimation.hurdle.read_collections import list_daily_files, read_one_file
from collection_estimation.hurdle.series import (
    add_week_index,
    check_weekly_totals,
    weekly_amounts,
)
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT

DATE_FORMAT = SYNTHETIC_LAYOUT.file_name_date_format  # "%d%m%Y"

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"
COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]
WEEK_ZERO = pd.Timestamp("2025-06-02").date()

WEEKLY_COLUMNS = ["key", "company_code", "customer_id", "week", "gross_receipts", "refunds",
                  "unapplied_lines", "lines"]


def lines_table(*lines):
    """Payment lines from (company, customer, week, amount_try, sap_document_no) tuples.

    The key is built the way `add_keys` builds it. An extra `source_row` column stands in for
    the other columns of the real table.
    """
    rows = [{"company_code": company, "customer_id": customer, "key": f"{company}-{customer}",
             "week": week, "amount_try": float(amount), "sap_document_no": sap}
            for company, customer, week, amount, sap in lines]
    table = pd.DataFrame(rows, columns=["company_code", "customer_id", "key", "week",
                                        "amount_try", "sap_document_no"])
    table["week"] = table["week"].astype("int64")
    table["amount_try"] = table["amount_try"].astype("float64")
    table["source_row"] = np.arange(len(table), dtype="int64") + 5
    return table


def paid(company, customer, week, amount, sap="1400000001"):
    """One payment line with an SAP document number (applied cash)."""
    return (company, customer, week, amount, sap)


def unapplied(company, customer, week, amount):
    """One payment line with no SAP document number."""
    return (company, customer, week, amount, None)


def only_row(weekly, key, week):
    """The one row of `weekly` for this key and week, as a dict."""
    rows = weekly[(weekly["key"] == key) & (weekly["week"] == week)]
    assert len(rows) == 1
    return rows.iloc[0].to_dict()


# --- shape -----------------------------------------------------------------------------

def test_columns_are_exactly_these():
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, 100)))
    assert list(weekly.columns) == WEEKLY_COLUMNS


def test_one_row_per_key_and_week():
    weekly = weekly_amounts(lines_table(
        paid("TR02", "1", 0, 100), paid("TR02", "1", 0, 50), paid("TR02", "1", 1, 10),
        paid("TR02", "2", 0, 7)))
    assert list(zip(weekly["key"], weekly["week"])) == [("TR02-1", 0), ("TR02-1", 1),
                                                        ("TR02-2", 0)]


def test_sorted_by_key_then_week():
    weekly = weekly_amounts(lines_table(
        paid("TR04", "9", 3, 1), paid("TR02", "5", 2, 1), paid("TR04", "9", 1, 1),
        paid("TR02", "5", 0, 1), paid("TR02", "1", 7, 1)))
    assert list(zip(weekly["key"], weekly["week"])) == [
        ("TR02-1", 7), ("TR02-5", 0), ("TR02-5", 2), ("TR04-9", 1), ("TR04-9", 3)]


def test_index_is_a_plain_range():
    weekly = weekly_amounts(lines_table(
        paid("TR04", "9", 3, 1), paid("TR02", "5", 2, 1), paid("TR02", "5", 0, 1)))
    assert weekly.index.tolist() == [0, 1, 2]


@pytest.mark.parametrize("column", ["week", "unapplied_lines", "lines"])
def test_count_columns_are_int64(column):
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, 100), unapplied("TR02", "1", 0, 5)))
    assert weekly[column].dtype == "int64"


@pytest.mark.parametrize("column", ["gross_receipts", "refunds"])
def test_amount_columns_are_float64(column):
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, 100), paid("TR02", "1", 0, -5)))
    assert weekly[column].dtype == "float64"


def test_company_code_and_customer_id_are_carried_through():
    weekly = weekly_amounts(lines_table(paid("TR05", "11226452", 2, 100)))
    assert only_row(weekly, "TR05-11226452", 2) == {
        "key": "TR05-11226452", "company_code": "TR05", "customer_id": "11226452", "week": 2,
        "gross_receipts": 100.0, "refunds": 0.0, "unapplied_lines": 0, "lines": 1}


# --- amounts ---------------------------------------------------------------------------

def test_gross_receipts_sums_positive_amounts():
    weekly = weekly_amounts(lines_table(
        paid("TR02", "1", 0, 100.25), paid("TR02", "1", 0, 50.50), paid("TR02", "1", 0, 0.25)))
    assert only_row(weekly, "TR02-1", 0)["gross_receipts"] == pytest.approx(151.0)


def test_refunds_sum_negative_amounts_and_stay_negative():
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, -30), paid("TR02", "1", 0, -12.5)))
    assert only_row(weekly, "TR02-1", 0)["refunds"] == pytest.approx(-42.5)


def test_mixed_lines_in_one_key_week_are_split_not_netted():
    weekly = weekly_amounts(lines_table(
        paid("TR02", "1", 0, 100), paid("TR02", "1", 0, -30), paid("TR02", "1", 0, 20)))
    row = only_row(weekly, "TR02-1", 0)
    assert row["gross_receipts"] == pytest.approx(120.0)
    assert row["refunds"] == pytest.approx(-30.0)
    assert row["lines"] == 3


def test_refund_larger_than_receipts_still_split():
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, 10), paid("TR02", "1", 0, -500)))
    row = only_row(weekly, "TR02-1", 0)
    assert (row["gross_receipts"], row["refunds"]) == (10.0, -500.0)


def test_only_refunds_give_zero_gross_receipts():
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, -80)))
    row = only_row(weekly, "TR02-1", 0)
    assert row["gross_receipts"] == 0.0
    assert row["refunds"] == -80.0
    assert row["lines"] == 1


def test_only_a_zero_line_gives_a_row_of_zeros_with_one_line():
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, 0)))
    row = only_row(weekly, "TR02-1", 0)
    assert (row["gross_receipts"], row["refunds"], row["lines"]) == (0.0, 0.0, 1)


def test_refunds_are_never_positive_and_gross_never_negative():
    weekly = weekly_amounts(lines_table(
        paid("TR02", "1", 0, 5), paid("TR02", "1", 0, -5), paid("TR02", "2", 1, -1),
        paid("TR04", "3", 2, 0), paid("TR04", "3", 3, 9)))
    assert (weekly["gross_receipts"] >= 0).all()
    assert (weekly["refunds"] <= 0).all()


def test_amounts_of_a_week_do_not_leak_into_another_key_or_week():
    weekly = weekly_amounts(lines_table(
        paid("TR02", "1", 0, 100), paid("TR02", "1", 1, 7), paid("TR02", "2", 0, 3)))
    assert only_row(weekly, "TR02-1", 0)["gross_receipts"] == 100.0
    assert only_row(weekly, "TR02-1", 1)["gross_receipts"] == 7.0
    assert only_row(weekly, "TR02-2", 0)["gross_receipts"] == 3.0


# --- lines and unapplied lines ---------------------------------------------------------

def test_lines_counts_every_line_including_refunds_and_zeros():
    weekly = weekly_amounts(lines_table(
        paid("TR02", "1", 0, 100), paid("TR02", "1", 0, -5), paid("TR02", "1", 0, 0),
        unapplied("TR02", "1", 0, 1)))
    assert only_row(weekly, "TR02-1", 0)["lines"] == 4


def test_line_with_none_sap_number_is_unapplied():
    weekly = weekly_amounts(lines_table(unapplied("TR02", "1", 0, 100)))
    assert only_row(weekly, "TR02-1", 0)["unapplied_lines"] == 1


def test_line_with_nan_sap_number_is_unapplied():
    table = lines_table(paid("TR02", "1", 0, 100), paid("TR02", "1", 0, 20))
    table["sap_document_no"] = table["sap_document_no"].astype(object)
    table.loc[0, "sap_document_no"] = float("nan")
    assert math.isnan(table.loc[0, "sap_document_no"])
    weekly = weekly_amounts(table)
    assert only_row(weekly, "TR02-1", 0)["unapplied_lines"] == 1


def test_all_nan_sap_column_counts_every_line_unapplied():
    table = lines_table(paid("TR02", "1", 0, 100), paid("TR02", "1", 0, 20))
    table["sap_document_no"] = np.nan
    assert only_row(weekly_amounts(table), "TR02-1", 0)["unapplied_lines"] == 2


def test_line_with_sap_number_is_not_unapplied():
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, 100, sap="1400000099")))
    assert only_row(weekly, "TR02-1", 0)["unapplied_lines"] == 0


def test_unapplied_and_applied_lines_mix_in_one_week():
    weekly = weekly_amounts(lines_table(
        paid("TR02", "1", 0, 100), unapplied("TR02", "1", 0, 40), unapplied("TR02", "1", 0, -3),
        paid("TR02", "1", 0, 10)))
    row = only_row(weekly, "TR02-1", 0)
    assert row["unapplied_lines"] == 2
    assert row["lines"] == 4


def test_unapplied_amounts_still_count_in_gross_receipts():
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, 100), unapplied("TR02", "1", 0, 40)))
    assert only_row(weekly, "TR02-1", 0)["gross_receipts"] == pytest.approx(140.0)


def test_unapplied_refund_still_counts_in_refunds():
    weekly = weekly_amounts(lines_table(unapplied("TR02", "1", 0, -40)))
    row = only_row(weekly, "TR02-1", 0)
    assert (row["refunds"], row["unapplied_lines"]) == (-40.0, 1)


def test_unapplied_lines_never_exceed_lines():
    weekly = weekly_amounts(lines_table(
        unapplied("TR02", "1", 0, 1), unapplied("TR02", "1", 0, 2), paid("TR02", "1", 1, 3)))
    assert (weekly["unapplied_lines"] <= weekly["lines"]).all()


# --- which rows exist ------------------------------------------------------------------

def test_weeks_without_a_line_get_no_row():
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 0, 1), paid("TR02", "1", 5, 1)))
    assert weekly["week"].tolist() == [0, 5]


def test_same_customer_in_two_companies_gives_two_keys():
    weekly = weekly_amounts(lines_table(paid("TR02", "77", 0, 10), paid("TR05", "77", 0, 20)))
    assert weekly["key"].tolist() == ["TR02-77", "TR05-77"]
    assert weekly["company_code"].tolist() == ["TR02", "TR05"]
    assert weekly["customer_id"].tolist() == ["77", "77"]
    assert weekly["gross_receipts"].tolist() == [10.0, 20.0]


def test_two_weeks_of_one_key_give_two_rows():
    weekly = weekly_amounts(lines_table(paid("TR02", "1", 3, 10), paid("TR02", "1", 4, 20)))
    assert len(weekly) == 2
    assert weekly["gross_receipts"].tolist() == [10.0, 20.0]


def test_negative_weeks_are_kept_as_their_own_rows():
    weekly = weekly_amounts(lines_table(
        paid("TR02", "1", -2, 5), paid("TR02", "1", -1, 6), paid("TR02", "1", 0, 7)))
    assert weekly["week"].tolist() == [-2, -1, 0]
    assert weekly["gross_receipts"].tolist() == [5.0, 6.0, 7.0]


def test_row_order_of_input_does_not_matter():
    forward = lines_table(paid("TR02", "1", 0, 1), paid("TR04", "2", 1, -2),
                          unapplied("TR02", "1", 2, 3), paid("TR02", "1", 0, 4))
    backward = forward.iloc[::-1]
    pd.testing.assert_frame_equal(weekly_amounts(backward), weekly_amounts(forward))


def test_non_default_input_index_gives_plain_range_index():
    table = lines_table(paid("TR02", "1", 0, 1), paid("TR04", "2", 1, 2))
    table.index = [40, 17]
    assert weekly_amounts(table).index.tolist() == [0, 1]


# --- the input -------------------------------------------------------------------------

def test_does_not_modify_the_input():
    table = lines_table(paid("TR02", "1", 0, 100), paid("TR02", "1", 0, -5),
                        unapplied("TR02", "2", 1, 3))
    before = table.copy()
    weekly_amounts(table)
    pd.testing.assert_frame_equal(table, before)


def test_leaves_no_helper_columns_on_the_input():
    table = lines_table(paid("TR02", "1", 0, 100))
    columns_before = list(table.columns)
    weekly_amounts(table)
    assert list(table.columns) == columns_before
    for helper in ("positive_amount", "negative_amount", "is_unapplied"):
        assert helper not in table.columns


def test_extra_input_columns_do_not_reach_the_output():
    table = lines_table(paid("TR02", "1", 0, 100))
    table["customer"] = "0000001"
    table["document_date"] = pd.Timestamp("2025-06-02")
    assert list(weekly_amounts(table).columns) == WEEKLY_COLUMNS


# --- empty input -----------------------------------------------------------------------

def test_empty_input_returns_empty_table_with_the_columns():
    weekly = weekly_amounts(lines_table())
    assert len(weekly) == 0
    assert list(weekly.columns) == WEEKLY_COLUMNS


def test_empty_input_keeps_the_numeric_dtypes():
    weekly = weekly_amounts(lines_table())
    assert {column: str(weekly[column].dtype) for column in WEEKLY_COLUMNS[3:]} == {
        "week": "int64", "gross_receipts": "float64", "refunds": "float64",
        "unapplied_lines": "int64", "lines": "int64"}


# --- totals tie out --------------------------------------------------------------------

def test_totals_tie_out_on_a_mixed_table():
    table = lines_table(
        paid("TR02", "1", 0, 100.10), paid("TR02", "1", 0, -20.05), unapplied("TR04", "2", 0, 3),
        paid("TR04", "2", -1, 0), paid("TR05", "3", 9, 1e6), unapplied("TR05", "3", 9, -0.01))
    weekly = weekly_amounts(table)
    assert (weekly["gross_receipts"].sum() + weekly["refunds"].sum()
            == pytest.approx(table["amount_try"].sum(), abs=1e-9))
    assert weekly["lines"].sum() == len(table)
    assert weekly["unapplied_lines"].sum() == table["sap_document_no"].isna().sum()


def test_line_with_a_missing_key_is_not_silently_dropped():
    # groupby drops missing keys by default; the total check must catch the lost line.
    table = lines_table(paid("TR02", "1", 0, 100), paid("TR02", "2", 0, 50))
    table.loc[1, "key"] = None
    with pytest.raises(ValueError):
        weekly_amounts(table)


def test_zero_amount_line_with_a_missing_key_is_caught_by_the_line_count():
    table = lines_table(paid("TR02", "1", 0, 100), paid("TR02", "2", 0, 0))
    table.loc[1, "key"] = None
    with pytest.raises(ValueError, match="lines"):
        weekly_amounts(table)


# --- check_weekly_totals ---------------------------------------------------------------

def weekly_pair():
    """A table of lines and its correct weekly table."""
    table = lines_table(paid("TR02", "1", 0, 100), paid("TR02", "1", 0, -30),
                        paid("TR04", "2", 1, 50))
    return table, weekly_amounts(table)


def test_check_passes_for_a_correct_pair():
    table, weekly = weekly_pair()
    check_weekly_totals(table, weekly)


def test_check_raises_naming_both_totals_when_try_differs():
    table, weekly = weekly_pair()
    weekly.loc[0, "gross_receipts"] += 1000.0
    with pytest.raises(ValueError) as error:
        check_weekly_totals(table, weekly)
    message = str(error.value)
    assert "1,120.00 TRY" in message  # the weekly total
    assert "to 120.00 TRY" in message  # the lines' total


def test_check_raises_when_try_differs_by_just_over_a_kurus():
    table, weekly = weekly_pair()
    weekly.loc[0, "refunds"] -= 0.02
    with pytest.raises(ValueError):
        check_weekly_totals(table, weekly)


def test_check_passes_for_half_a_kurus_difference():
    table, weekly = weekly_pair()
    weekly.loc[0, "gross_receipts"] += 0.005
    check_weekly_totals(table, weekly)


def test_check_raises_when_line_counts_differ():
    table, weekly = weekly_pair()
    weekly.loc[0, "lines"] += 1
    with pytest.raises(ValueError, match="lines"):
        check_weekly_totals(table, weekly)


def test_check_raises_when_a_weekly_row_is_lost():
    table, weekly = weekly_pair()
    with pytest.raises(ValueError):
        check_weekly_totals(table, weekly.iloc[:-1])


def test_check_raises_when_a_zero_line_is_added_to_the_input():
    table, weekly = weekly_pair()
    extra = pd.concat([table, lines_table(paid("TR05", "9", 0, 0))], ignore_index=True)
    with pytest.raises(ValueError, match="lines"):
        check_weekly_totals(extra, weekly)


def test_check_does_not_modify_its_inputs():
    table, weekly = weekly_pair()
    table_before, weekly_before = table.copy(), weekly.copy()
    check_weekly_totals(table, weekly)
    pd.testing.assert_frame_equal(table, table_before)
    pd.testing.assert_frame_equal(weekly, weekly_before)


# --- the synthetic corpus --------------------------------------------------------------

def corpus_lines(daily_files):
    """The payment lines of these daily files, keyed and with week numbers."""
    table = pd.concat([read_one_file(path, SYNTHETIC_LAYOUT) for path in daily_files.values()],
                      ignore_index=True)
    return add_week_index(add_keys(table, COMPANIES), WEEK_ZERO)


@pytest.fixture(scope="module")
def first_days():
    if not SYNTHETIC_CORPUS.is_dir():
        pytest.skip("synthetic corpus not installed")
    files = list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT)
    first_three = dict(list(files.items())[:3])
    lines = corpus_lines(first_three)
    return lines, weekly_amounts(lines)


def test_synthetic_first_days_amounts_tie_out(first_days):
    lines, weekly = first_days
    assert len(lines) > 0
    assert (weekly["gross_receipts"].sum() + weekly["refunds"].sum()
            == pytest.approx(lines["amount_try"].sum(), abs=0.01))


def test_synthetic_first_days_line_count_ties_out(first_days):
    lines, weekly = first_days
    assert weekly["lines"].sum() == len(lines)


def test_synthetic_first_days_unapplied_lines_tie_out(first_days):
    lines, weekly = first_days
    assert weekly["unapplied_lines"].sum() == lines["sap_document_no"].isna().sum()


def test_synthetic_first_days_one_row_per_key_week(first_days):
    lines, weekly = first_days
    assert not weekly.duplicated(["key", "week"]).any()
    assert len(weekly) == lines.groupby(["key", "week"]).ngroups
    assert list(weekly.columns) == WEEKLY_COLUMNS


@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason="synthetic corpus not installed")
def test_synthetic_all_files():
    # Slow: reads all 306 daily files.
    lines = corpus_lines(list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT))
    weekly = weekly_amounts(lines)
    assert len(weekly) == 20_479
    assert (weekly["gross_receipts"] > 0).sum() == 20_365
    assert weekly["unapplied_lines"].sum() == 815
    assert weekly["gross_receipts"].sum() == pytest.approx(8_677_737_670.99, abs=0.01)
    assert weekly["refunds"].sum() == pytest.approx(-30_134_702.10, abs=0.01)
