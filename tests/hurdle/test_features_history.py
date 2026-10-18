"""Tests for `hurdle/features_history.py`: `ledger_receipts_before`, `first_history_week`,
`EARLY_RECEIPT_COLUMNS`, `weekly_matrix`, `check_matrix_totals`, `MATRIX_TOLERANCE`, and
the history features (`history_features`, `check_matrix_for_features`, `key_origins`,
`trailing_sum`, `values_at`).

Hand-built tables for the unit tests; slow integration blocks against the synthetic AR
ledger and corpus in data/raw (skipped when they are absent). Nothing here reads or writes data/interim.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import math
import re

import numpy as np

from collection_estimation.hurdle.features_history import (
    EARLY_RECEIPT_COLUMNS,
    MATRIX_TOLERANCE,
    check_matrix_for_features,
    check_matrix_totals,
    first_history_week,
    history_features,
    key_origins,
    ledger_receipts_before,
    trailing_sum,
    values_at,
    weekly_matrix,
)

WEEK_ZERO = date(2025, 6, 2)        # a Monday
FIRST_FILE_DAY = date(2025, 6, 2)


def ledger_table(rows, extra=False):
    """rows: (key, document_role, posting_date text, amount_local)."""
    table = pd.DataFrame({
        "key": pd.Series([row[0] for row in rows], dtype=object),
        "document_role": pd.Series([row[1] for row in rows], dtype=object),
        "posting_date": pd.to_datetime(pd.Series([row[2] for row in rows], dtype=object)),
        "amount_local": pd.Series([row[3] for row in rows], dtype="float64"),
    })
    if extra:
        table["company_code"] = "TR02"
        table["document_no"] = [f"D{i}" for i in range(len(rows))]
    return table


def receipts(rows, first_file_day=FIRST_FILE_DAY, week_zero=WEEK_ZERO):
    return ledger_receipts_before(ledger_table(rows), first_file_day, week_zero)


def as_dict(result):
    return {(row.key, row.week): row.gross_receipts for row in result.itertuples()}


# --- EARLY_RECEIPT_COLUMNS ------------------------------------------------------------

def test_early_receipt_columns_are_key_week_gross_receipts():
    assert EARLY_RECEIPT_COLUMNS == ["key", "week", "gross_receipts"]


# --- ledger_receipts_before: what is counted -------------------------------------------

def test_a_payment_before_the_first_file_day_gives_its_flipped_amount():
    result = receipts([("TR02-1", "payment", "2025-05-28", -300.0)])
    assert as_dict(result) == {("TR02-1", -1): 300.0}


def test_a_payment_on_the_first_file_day_is_excluded():
    result = receipts([("TR02-1", "payment", "2025-06-02", -300.0)])
    assert result.empty
    assert list(result.columns) == EARLY_RECEIPT_COLUMNS


def test_a_payment_after_the_first_file_day_is_excluded():
    result = receipts([("TR02-1", "payment", "2025-06-10", -300.0),
                       ("TR02-1", "payment", "2025-05-30", -10.0)])
    assert as_dict(result) == {("TR02-1", -1): 10.0}


def test_the_day_before_the_first_file_day_is_included():
    result = receipts([("TR02-1", "payment", "2025-06-01", -42.5)])
    assert as_dict(result) == {("TR02-1", -1): 42.5}


def test_first_file_day_not_equal_to_week_zero_is_the_cut():
    # The cut is first_file_day, not week_zero: with a later first file, a payment after
    # week zero but before the first file counts (week 0 here).
    rows = [("TR02-1", "payment", "2025-06-03", -5.0),
            ("TR02-1", "payment", "2025-06-04", -7.0)]
    result = receipts(rows, first_file_day=date(2025, 6, 4))
    assert as_dict(result) == {("TR02-1", 0): 5.0}


@pytest.mark.parametrize("role", ["invoice", "credit_memo"])
def test_invoices_and_credit_memos_are_ignored(role):
    result = receipts([("TR02-1", role, "2025-05-20", -1000.0),
                       ("TR02-1", role, "2025-05-21", 1000.0)])
    assert result.empty


def test_only_payments_count_among_mixed_roles():
    result = receipts([("TR02-1", "invoice", "2025-05-20", 1000.0),
                       ("TR02-1", "credit_memo", "2025-05-21", -200.0),
                       ("TR02-1", "payment", "2025-05-22", -800.0)])
    assert as_dict(result) == {("TR02-1", -2): 800.0}


def test_role_matching_is_exact():
    # Pinned: 'Payment' or ' payment' is not 'payment' (the reader normalises roles).
    result = receipts([("TR02-1", "Payment", "2025-05-22", -800.0),
                       ("TR02-1", " payment", "2025-05-22", -800.0)])
    assert result.empty


# --- ledger_receipts_before: amounts ----------------------------------------------------

def test_a_refund_paid_out_counts_as_nothing_but_still_makes_a_row():
    # Pinned: the refund is clipped to 0 before grouping, so its (key, week) row stays,
    # with 0.0.
    result = receipts([("TR02-1", "payment", "2025-05-28", 50.0)])
    assert as_dict(result) == {("TR02-1", -1): 0.0}
    assert len(result) == 1


def test_a_refund_does_not_reduce_a_payment_in_the_same_week():
    result = receipts([("TR02-1", "payment", "2025-05-27", -300.0),
                       ("TR02-1", "payment", "2025-05-28", 50.0)])
    assert as_dict(result) == {("TR02-1", -1): 300.0}


def test_a_zero_payment_gives_zero():
    result = receipts([("TR02-1", "payment", "2025-05-28", 0.0)])
    assert as_dict(result) == {("TR02-1", -1): 0.0}


def test_several_payments_in_one_week_are_summed():
    result = receipts([("TR02-1", "payment", "2025-05-26", -100.0),   # Monday
                       ("TR02-1", "payment", "2025-05-28", -20.25),
                       ("TR02-1", "payment", "2025-06-01", -0.75)])   # Sunday, same week
    assert as_dict(result) == {("TR02-1", -1): 121.0}


def test_payments_in_different_weeks_stay_apart():
    result = receipts([("TR02-1", "payment", "2025-05-25", -1.0),    # Sunday, week -2
                       ("TR02-1", "payment", "2025-05-26", -2.0)])   # Monday, week -1
    assert as_dict(result) == {("TR02-1", -2): 1.0, ("TR02-1", -1): 2.0}


def test_two_keys_in_one_week_stay_apart():
    result = receipts([("TR02-1", "payment", "2025-05-27", -10.0),
                       ("TR04-1", "payment", "2025-05-27", -20.0),
                       ("TR02-1", "payment", "2025-05-28", -1.0)])
    assert as_dict(result) == {("TR02-1", -1): 11.0, ("TR04-1", -1): 20.0}


def test_a_nan_amount_counts_as_zero():
    # Pinned: a payment line with no amount gives 0.0 (clip keeps NaN, sum skips it).
    result = receipts([("TR02-1", "payment", "2025-05-28", float("nan"))])
    assert as_dict(result) == {("TR02-1", -1): 0.0}


# --- ledger_receipts_before: weeks ------------------------------------------------------

@pytest.mark.parametrize("day, week", [
    ("2025-06-01", -1), ("2025-05-26", -1), ("2025-05-25", -2),
    ("2024-10-28", -31), ("2024-10-27", -32), ("2024-01-01", -74),
])
def test_weeks_are_numbered_from_week_zero_and_negative_before_it(day, week):
    result = receipts([("TR02-1", "payment", day, -1.0)])
    assert list(result["week"]) == [week]


def test_a_week_zero_that_is_not_a_monday_raises():
    with pytest.raises(ValueError, match="Monday"):
        receipts([("TR02-1", "payment", "2025-05-28", -1.0)], week_zero=date(2025, 6, 3))


# --- ledger_receipts_before: shape -------------------------------------------------------

def test_columns_are_exactly_early_receipt_columns_even_with_extra_input_columns():
    table = ledger_table([("TR02-1", "payment", "2025-05-28", -1.0)], extra=True)
    result = ledger_receipts_before(table, FIRST_FILE_DAY, WEEK_ZERO)
    assert list(result.columns) == EARLY_RECEIPT_COLUMNS


def test_dtypes():
    result = receipts([("TR02-1", "payment", "2025-05-28", -1.0),
                       ("TR04-1", "payment", "2025-04-28", -2.0)])
    assert result["key"].dtype == object
    assert result["week"].dtype == "int64"
    assert result["gross_receipts"].dtype == "float64"


def test_rows_are_sorted_by_key_then_week_with_a_clean_index():
    result = receipts([("TR04-1", "payment", "2025-05-28", -1.0),
                       ("TR02-1", "payment", "2025-05-28", -2.0),
                       ("TR02-1", "payment", "2025-04-28", -3.0)])
    assert list(zip(result["key"], result["week"])) == [
        ("TR02-1", -5), ("TR02-1", -1), ("TR04-1", -1)]
    assert list(result.index) == [0, 1, 2]


def test_an_empty_ledger_gives_an_empty_table_with_the_columns_and_dtypes():
    result = receipts([])
    assert result.empty
    assert list(result.columns) == EARLY_RECEIPT_COLUMNS
    assert result["week"].dtype == "int64"
    assert result["gross_receipts"].dtype == "float64"


def test_no_early_payment_gives_an_empty_table_with_the_columns():
    result = receipts([("TR02-1", "invoice", "2025-05-20", 10.0),
                       ("TR02-1", "payment", "2025-07-01", -10.0)])
    assert result.empty
    assert list(result.columns) == EARLY_RECEIPT_COLUMNS
    assert result["week"].dtype == "int64"
    assert result["gross_receipts"].dtype == "float64"


def test_the_input_ledger_is_not_modified():
    table = ledger_table([("TR02-1", "payment", "2025-05-28", -300.0),
                          ("TR02-1", "payment", "2025-05-29", 50.0),
                          ("TR02-1", "invoice", "2025-05-20", 300.0)], extra=True)
    before = table.copy(deep=True)
    ledger_receipts_before(table, FIRST_FILE_DAY, WEEK_ZERO)
    pd.testing.assert_frame_equal(table, before)


def test_the_same_input_gives_the_same_output():
    table = ledger_table([("TR02-1", "payment", "2025-05-28", -300.0),
                          ("TR04-1", "payment", "2025-03-03", -1.5)])
    pd.testing.assert_frame_equal(ledger_receipts_before(table, FIRST_FILE_DAY, WEEK_ZERO),
                                  ledger_receipts_before(table, FIRST_FILE_DAY, WEEK_ZERO))


# --- first_history_week ----------------------------------------------------------------

def first_week(first_day, role="payment", later=("2025-05-28",)):
    rows = [("TR02-1", "payment", day, -1.0) for day in later]
    rows.append(("TR02-1", role, first_day, -1.0))
    return first_history_week(ledger_table(rows), WEEK_ZERO)


def test_first_posting_on_a_monday_gives_that_week():
    assert first_week("2024-10-28") == -31


def test_first_posting_on_a_friday_gives_the_following_monday():
    # Friday 2024-10-25 -> Monday 2024-10-28 -> week -31.
    assert first_week("2024-10-25") == -31


def test_first_posting_on_a_sunday_gives_the_next_day():
    # Sunday 2024-10-27 -> Monday 2024-10-28 -> week -31.
    assert first_week("2024-10-27") == -31


def test_first_posting_on_a_tuesday_skips_to_the_next_week():
    assert first_week("2024-10-22") == -31   # Tuesday -> Monday 2024-10-28


def test_first_posting_on_week_zero_gives_zero():
    assert first_week("2025-06-02", later=("2025-06-10",)) == 0


@pytest.mark.parametrize("role", ["invoice", "credit_memo"])
def test_the_first_posting_need_not_be_a_payment(role):
    assert first_week("2024-10-25", role=role) == -31


def test_row_order_does_not_matter():
    rows = [("TR02-1", "payment", "2025-05-28", -1.0),
            ("TR04-1", "invoice", "2024-10-25", 5.0),
            ("TR02-1", "payment", "2025-01-01", -1.0)]
    table = ledger_table(rows)
    assert first_history_week(table, WEEK_ZERO) == -31
    assert first_history_week(table.iloc[::-1], WEEK_ZERO) == -31


def test_first_history_week_returns_a_python_int():
    assert type(first_week("2024-10-25")) is int


def test_first_history_week_does_not_modify_the_ledger():
    table = ledger_table([("TR02-1", "payment", "2024-10-25", -1.0)], extra=True)
    before = table.copy(deep=True)
    first_history_week(table, WEEK_ZERO)
    pd.testing.assert_frame_equal(table, before)


def test_first_history_week_with_a_week_zero_not_a_monday_raises():
    with pytest.raises(ValueError, match="Monday"):
        first_history_week(ledger_table([("TR02-1", "payment", "2024-10-25", -1.0)]),
                           date(2025, 6, 3))


def test_first_history_week_on_an_empty_ledger_raises():
    with pytest.raises(ValueError,
                       match="the ledger has no lines, so it gives no payment history"):
        first_history_week(ledger_table([]), WEEK_ZERO)


def test_first_history_week_on_an_empty_ledger_with_extra_columns_raises_the_same():
    with pytest.raises(ValueError, match="the ledger has no lines"):
        first_history_week(ledger_table([], extra=True), WEEK_ZERO)


def test_first_history_week_on_a_ledger_of_only_invoices_does_not_raise():
    # Not empty: any line counts, so the empty-ledger message is not given.
    table = ledger_table([("TR02-1", "invoice", "2025-06-02", 100.0)])
    assert first_history_week(table, WEEK_ZERO) == 0


# --- integration with the synthetic ledger (SLOW) -----------------------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
AS_OF = date(2026, 8, 28)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]


@pytest.fixture(scope="module")
def synthetic_ledger():
    if not LEDGER_FILE.exists():
        pytest.skip("the synthetic ledger is not in data/raw")
    from collection_estimation.hurdle.keys import add_keys
    from collection_estimation.hurdle.read_ledger import read_ledger
    from tests.hurdle.workbooks import SYNTHETIC_LEDGER_LAYOUT

    return add_keys(read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT), KNOWN_COMPANIES)


@pytest.fixture(scope="module")
def synthetic_early(synthetic_ledger):
    return ledger_receipts_before(synthetic_ledger, date(2025, 6, 2), WEEK_ZERO)


def test_slow_synthetic_first_history_week_is_minus_31(synthetic_ledger):
    assert first_history_week(synthetic_ledger, WEEK_ZERO) == -31


def test_slow_synthetic_early_receipts_total(synthetic_early):
    assert synthetic_early["gross_receipts"].sum() == pytest.approx(2_240_423_149.13, abs=0.01)


def test_slow_synthetic_early_receipts_weeks_are_between_minus_32_and_minus_1(synthetic_early):
    assert synthetic_early["week"].min() >= -32
    assert synthetic_early["week"].max() <= -1


def test_slow_synthetic_from_the_first_history_week(synthetic_early):
    kept = synthetic_early[synthetic_early["week"] >= -31]
    assert len(kept) == 6854
    assert kept["key"].nunique() == 1768
    assert kept["gross_receipts"].sum() == pytest.approx(2_240_423_149.13, abs=0.01)


def test_slow_synthetic_no_payment_in_week_minus_32(synthetic_early):
    week_32 = synthetic_early[synthetic_early["week"] == -32]
    assert week_32["gross_receipts"].sum() == 0.0


def test_slow_synthetic_early_receipts_shape(synthetic_early):
    assert list(synthetic_early.columns) == EARLY_RECEIPT_COLUMNS
    assert not synthetic_early.duplicated(["key", "week"]).any()
    assert (synthetic_early["gross_receipts"] >= 0).all()


# === weekly_matrix and check_matrix_totals ===============================================

def weekly_table(rows):
    """rows: (key, week, gross_receipts), as `weekly_amounts` would give them."""
    return pd.DataFrame({
        "key": pd.Series([row[0] for row in rows], dtype=object),
        "company_code": pd.Series([row[0].split("-")[0] for row in rows], dtype=object),
        "customer_id": pd.Series([row[0].split("-")[1] for row in rows], dtype=object),
        "week": pd.Series([row[1] for row in rows], dtype="int64"),
        "gross_receipts": pd.Series([row[2] for row in rows], dtype="float64"),
        "refunds": pd.Series([0.0] * len(rows), dtype="float64"),
        "unapplied_lines": pd.Series([0] * len(rows), dtype="int64"),
        "lines": pd.Series([1] * len(rows), dtype="int64"),
    })


def early_table(rows):
    """rows: (key, week, gross_receipts), as `ledger_receipts_before` would give them."""
    return pd.DataFrame({
        "key": pd.Series([row[0] for row in rows], dtype=object),
        "week": pd.Series([row[1] for row in rows], dtype="int64"),
        "gross_receipts": pd.Series([row[2] for row in rows], dtype="float64"),
    })


def calendar(last_week, missing=()):
    """A week calendar for weeks 0 ... last_week; the weeks in `missing` are missing."""
    weeks = list(range(last_week + 1))
    is_missing = [week in missing for week in weeks]
    return pd.DataFrame({
        "week": pd.Series(weeks, dtype="int64"),
        "monday": [WEEK_ZERO + pd.Timedelta(weeks=week) for week in weeks],
        "business_days": pd.Series([5] * len(weeks), dtype="int64"),
        "days_with_file": pd.Series([4 if m else 5 for m in is_missing], dtype="int64"),
        "filled_days": pd.Series([0] * len(weeks), dtype="int64"),
        "missing_days": pd.Series([1 if m else 0 for m in is_missing], dtype="int64"),
        "is_missing": pd.Series(is_missing, dtype=bool),
    })


def flat_levels(first, last, level=0.5):
    return pd.Series(level, index=range(first, last + 1), dtype="float64", name="price_level")


def matrix(weekly_rows=(), early_rows=(), keys=("TR02-1",), first_week=-2, last_week=3,
           missing=(), levels=None):
    if levels is None:
        levels = flat_levels(first_week, last_week)
    return weekly_matrix(weekly_table(list(weekly_rows)), early_table(list(early_rows)),
                         pd.Series(list(keys), dtype=object), calendar(last_week, missing),
                         levels, first_week, last_week)


def test_matrix_tolerance_is_one_kurus():
    assert MATRIX_TOLERANCE == 0.01


# --- weekly_matrix: shape and labels ----------------------------------------------------

def test_one_row_per_key_sorted_and_deduplicated():
    result = matrix(keys=["TR04-9", "TR02-1", "TR04-9", "TR02-5", "TR02-1"])
    assert result.index.tolist() == ["TR02-1", "TR02-5", "TR04-9"]


def test_index_is_named_key_whatever_the_series_name():
    result = weekly_matrix(weekly_table([]), early_table([]),
                           pd.Series(["TR02-1"], name="something_else", dtype=object),
                           calendar(3), flat_levels(-2, 3), -2, 3)
    assert result.index.name == "key"


def test_one_column_per_week_from_first_to_last():
    result = matrix(first_week=-2, last_week=3)
    assert result.columns.tolist() == [-2, -1, 0, 1, 2, 3]
    assert result.columns.name == "week"


def test_column_labels_are_ints():
    result = matrix(first_week=-2, last_week=3)
    assert all(isinstance(week, (int, np.integer)) for week in result.columns)
    assert result.columns.dtype == "int64"


def test_first_week_zero_gives_no_negative_columns():
    result = matrix(weekly_rows=[("TR02-1", 0, 10.0)], first_week=0, last_week=3)
    assert result.columns.tolist() == [0, 1, 2, 3]


def test_a_single_week():
    result = matrix(weekly_rows=[("TR02-1", 0, 10.0)], first_week=0, last_week=0)
    assert result.shape == (1, 1)
    assert result.loc["TR02-1", 0] == pytest.approx(20.0)


def test_every_column_is_float64():
    result = matrix(weekly_rows=[("TR02-1", 0, 10.0)], early_rows=[("TR02-1", -1, 4.0)],
                    missing=(2,))
    assert (result.dtypes == "float64").all()


def test_float64_even_without_any_receipt_or_missing_week():
    result = matrix()
    assert (result.dtypes == "float64").all()


def test_empty_keys_give_an_empty_matrix_with_the_week_columns():
    result = matrix(weekly_rows=[("TR02-1", 0, 10.0)], keys=[])
    assert result.shape == (0, 6)
    assert result.columns.tolist() == [-2, -1, 0, 1, 2, 3]
    assert result.index.name == "key"


# --- weekly_matrix: cell values -------------------------------------------------------

def test_a_cell_is_the_gross_receipts_divided_by_the_level():
    result = matrix(weekly_rows=[("TR02-1", 1, 300.0)], levels=flat_levels(-2, 3, 0.5))
    assert result.loc["TR02-1", 1] == pytest.approx(600.0)


def test_no_receipt_in_a_week_with_data_is_zero_not_nan():
    result = matrix(weekly_rows=[("TR02-1", 1, 300.0)])
    row = result.loc["TR02-1"]
    for week in (-2, -1, 0, 2, 3):
        assert row[week] == 0.0


def test_a_key_in_keys_with_no_receipt_at_all_gets_a_row_of_zeros():
    result = matrix(weekly_rows=[("TR02-1", 1, 300.0)], keys=["TR02-1", "TR02-2"])
    assert result.loc["TR02-2"].tolist() == [0.0] * 6


def test_a_key_with_no_receipt_is_nan_in_missing_weeks():
    result = matrix(weekly_rows=[("TR02-1", 1, 300.0)], keys=["TR02-1", "TR02-2"],
                    missing=(2,))
    row = result.loc["TR02-2"]
    assert math.isnan(row[2])
    assert [row[week] for week in (-2, -1, 0, 1, 3)] == [0.0] * 5


def test_a_missing_week_is_nan_for_every_key():
    result = matrix(weekly_rows=[("TR02-1", 0, 1.0)], keys=["TR02-1", "TR02-2", "TR04-3"],
                    missing=(1, 3))
    assert result[1].isna().all()
    assert result[3].isna().all()
    assert result[[-2, -1, 0, 2]].notna().all().all()


def test_a_missing_week_is_nan_even_with_partial_receipts_in_it():
    # The weekly table has receipts in week 2, but the week is missing: unknown, not 250.
    result = matrix(weekly_rows=[("TR02-1", 2, 250.0), ("TR02-2", 2, 5.0)],
                    keys=["TR02-1", "TR02-2"], missing=(2,))
    assert result[2].isna().all()


def test_a_missing_week_with_receipts_does_not_fail_the_tie_out():
    # Blanked only after the tie-out, so its receipts are still counted there.
    matrix(weekly_rows=[("TR02-1", 2, 250.0)], missing=(2,))


def test_a_calendar_missing_week_after_last_week_adds_nothing():
    weekly = weekly_table([("TR02-1", 0, 10.0)])
    result = weekly_matrix(weekly, early_table([]), pd.Series(["TR02-1"], dtype=object),
                           calendar(6, missing=(5,)), flat_levels(0, 6), 0, 3)
    assert result.columns.tolist() == [0, 1, 2, 3]
    assert result.notna().all().all()


def test_negative_weeks_never_nan_when_first_week_is_negative():
    result = matrix(early_rows=[("TR02-1", -1, 8.0)], first_week=-3, last_week=2,
                    missing=(0,))
    assert result[[-3, -2, -1]].notna().all().all()
    assert result.loc["TR02-1", -1] == pytest.approx(16.0)


def test_negative_weeks_come_from_early_receipts():
    result = matrix(early_rows=[("TR02-1", -2, 30.0), ("TR02-1", -1, 40.0)])
    assert result.loc["TR02-1", -2] == pytest.approx(60.0)
    assert result.loc["TR02-1", -1] == pytest.approx(80.0)


def test_weeks_from_zero_come_from_the_weekly_table():
    result = matrix(weekly_rows=[("TR02-1", 0, 1.0), ("TR02-1", 3, 2.0)])
    assert result.loc["TR02-1", 0] == pytest.approx(2.0)
    assert result.loc["TR02-1", 3] == pytest.approx(4.0)


def test_each_week_is_divided_by_its_own_level():
    levels = pd.Series([0.8, 0.9, 1.0, 1.25], index=[0, 1, 2, 3], dtype="float64")
    result = matrix(weekly_rows=[("TR02-1", week, 100.0) for week in range(4)],
                    first_week=0, last_week=3, levels=levels)
    assert result.loc["TR02-1"].tolist() == pytest.approx([125.0, 100.0 / 0.9, 100.0, 80.0])


def test_deflation_applies_to_negative_weeks_too():
    levels = pd.Series([0.4, 0.5, 1.0], index=[-2, -1, 0], dtype="float64")
    result = matrix(early_rows=[("TR02-1", -2, 10.0), ("TR02-1", -1, 10.0)],
                    weekly_rows=[("TR02-1", 0, 10.0)], first_week=-2, last_week=0,
                    levels=levels)
    assert result.loc["TR02-1"].tolist() == pytest.approx([25.0, 20.0, 10.0])


def test_levels_for_extra_weeks_are_ignored():
    levels = flat_levels(-40, 80, 0.5)
    result = matrix(weekly_rows=[("TR02-1", 0, 1.0)], levels=levels)
    assert result.columns.tolist() == [-2, -1, 0, 1, 2, 3]
    assert result.loc["TR02-1", 0] == pytest.approx(2.0)


def test_level_one_leaves_amounts_unchanged():
    result = matrix(weekly_rows=[("TR02-1", 0, 123.45)], levels=flat_levels(-2, 3, 1.0))
    assert result.loc["TR02-1", 0] == 123.45


def test_only_gross_receipts_are_used_not_refunds():
    weekly = weekly_table([("TR02-1", 0, 100.0)])
    weekly["refunds"] = -70.0
    result = weekly_matrix(weekly, early_table([]), pd.Series(["TR02-1"], dtype=object),
                           calendar(3), flat_levels(0, 3, 1.0), 0, 3)
    assert result.loc["TR02-1", 0] == 100.0


def test_zero_gross_receipts_give_zero():
    result = matrix(weekly_rows=[("TR02-1", 1, 0.0)])
    assert result.loc["TR02-1", 1] == 0.0


def test_two_keys_in_one_week_stay_apart():
    result = matrix(weekly_rows=[("TR02-1", 1, 10.0), ("TR04-1", 1, 30.0)],
                    keys=["TR02-1", "TR04-1"], levels=flat_levels(-2, 3, 1.0))
    assert result.loc["TR02-1", 1] == 10.0
    assert result.loc["TR04-1", 1] == 30.0


# --- weekly_matrix: duplicates are summed (pinned) --------------------------------------

def test_duplicate_key_week_rows_in_the_weekly_table_are_summed():
    result = matrix(weekly_rows=[("TR02-1", 1, 10.0), ("TR02-1", 1, 5.0)],
                    levels=flat_levels(-2, 3, 1.0))
    assert result.loc["TR02-1", 1] == pytest.approx(15.0)


def test_duplicate_key_week_rows_in_early_receipts_are_summed():
    result = matrix(early_rows=[("TR02-1", -1, 10.0), ("TR02-1", -1, 7.0)],
                    levels=flat_levels(-2, 3, 1.0))
    assert result.loc["TR02-1", -1] == pytest.approx(17.0)


def test_duplicates_within_each_source_are_still_summed_alongside_the_other():
    result = matrix(weekly_rows=[("TR02-1", 0, 10.0), ("TR02-1", 0, 2.0)],
                    early_rows=[("TR02-1", -1, 3.0), ("TR02-1", -1, 4.0)],
                    levels=flat_levels(-2, 3, 1.0))
    assert result.loc["TR02-1", 0] == pytest.approx(12.0)
    assert result.loc["TR02-1", -1] == pytest.approx(7.0)


# --- weekly_matrix: early receipts must be before week 0 ---------------------------------

def test_an_early_receipt_in_a_weekly_week_raises_with_the_exact_message():
    with pytest.raises(ValueError) as error:
        matrix(weekly_rows=[("TR02-1", 0, 10.0)], early_rows=[("TR02-1", 2, 3.0)])
    assert str(error.value) == ("early receipts must be before week 0, but some are in weeks "
                                "[2]: they would overlap the weekly amounts")


def test_an_early_receipt_in_week_zero_itself_raises():
    with pytest.raises(ValueError, match=r"early receipts must be before week 0, but some are "
                                         r"in weeks \[0\]"):
        matrix(early_rows=[("TR02-1", 0, 3.0)])


def test_an_early_receipt_in_week_minus_one_is_fine():
    result = matrix(early_rows=[("TR02-1", -1, 3.0)], levels=flat_levels(-2, 3, 1.0))
    assert result.loc["TR02-1", -1] == pytest.approx(3.0)


def test_the_overlap_is_rejected_even_without_a_weekly_row_in_that_week():
    with pytest.raises(ValueError, match="early receipts must be before week 0"):
        matrix(early_rows=[("TR02-1", 1, 3.0)])


def test_late_early_receipt_weeks_are_listed_sorted_once_each_at_most_ten():
    rows = [("TR02-1", week, 1.0) for week in range(14, -1, -1)]   # weeks 14 ... 0
    rows += [("TR02-1", 3, 1.0), ("TR02-1", -1, 1.0)]
    with pytest.raises(ValueError) as error:
        matrix(early_rows=rows, last_week=20, levels=flat_levels(-2, 20))
    assert str(error.value) == ("early receipts must be before week 0, but some are in weeks "
                                "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]: they would overlap the "
                                "weekly amounts")


def test_the_early_receipt_check_applies_to_keys_not_in_keys():
    # Pinned: it runs before the key filter, so a dropped key's late receipt still raises.
    with pytest.raises(ValueError, match=r"early receipts must be before week 0, but some are "
                                         r"in weeks \[1\]"):
        matrix(weekly_rows=[("TR02-1", 0, 10.0)], early_rows=[("TR02-99999", 1, 5.0)])


def test_the_early_receipt_check_runs_before_the_outside_weeks_check():
    # Week 40 is also outside first_week ... last_week, but the overlap error comes first.
    with pytest.raises(ValueError, match=r"early receipts must be before week 0, but some are "
                                         r"in weeks \[40\]"):
        matrix(early_rows=[("TR02-1", 40, 1.0)])


def test_the_early_receipt_check_runs_before_the_price_level_check():
    with pytest.raises(ValueError, match="early receipts must be before week 0"):
        matrix(early_rows=[("TR02-1", 0, 1.0)], levels=pd.Series([], dtype="float64"))


# --- weekly_matrix: keys not in `keys` ----------------------------------------------------

def test_a_key_not_in_keys_is_dropped():
    result = matrix(weekly_rows=[("TR02-1", 0, 10.0), ("TR02-99999", 0, 500.0)])
    assert result.index.tolist() == ["TR02-1"]


def test_a_dropped_key_does_not_enter_the_tie_out():
    # Its 500 TRY is not in the matrix; the check must not count it on the source side.
    matrix(weekly_rows=[("TR02-1", 0, 10.0), ("TR02-99999", 0, 500.0)],
           early_rows=[("TR02-99999", -1, 50.0)])


def test_a_dropped_key_outside_the_weeks_does_not_raise():
    result = matrix(weekly_rows=[("TR02-1", 0, 10.0), ("TR02-99999", 40, 500.0)],
                    early_rows=[("TR02-99999", -30, 1.0)])
    assert result.index.tolist() == ["TR02-1"]


# --- weekly_matrix: errors -------------------------------------------------------------

def test_an_early_receipt_before_first_week_raises():
    with pytest.raises(ValueError, match=r"receipts in weeks \[-5\] fall outside weeks -2 to 3"):
        matrix(early_rows=[("TR02-1", -5, 1.0)])


def test_a_weekly_receipt_after_last_week_raises():
    with pytest.raises(ValueError, match=r"receipts in weeks \[4\] fall outside weeks -2 to 3"):
        matrix(weekly_rows=[("TR02-1", 4, 1.0)])


def test_a_negative_week_with_first_week_zero_raises():
    with pytest.raises(ValueError, match=r"weeks \[-1\] fall outside weeks 0 to 3"):
        matrix(early_rows=[("TR02-1", -1, 1.0)], first_week=0, last_week=3)


def test_a_zero_receipt_outside_the_weeks_still_raises():
    with pytest.raises(ValueError, match="fall outside"):
        matrix(weekly_rows=[("TR02-1", 9, 0.0)])


def test_outside_weeks_listed_sorted_and_at_most_ten():
    rows = [("TR02-1", week, 1.0) for week in range(20, 4, -1)]   # weeks 20 ... 5
    with pytest.raises(ValueError) as error:
        matrix(weekly_rows=rows)
    assert str(error.value) == ("receipts in weeks [5, 6, 7, 8, 9, 10, 11, 12, 13, 14] "
                                "fall outside weeks -2 to 3")


def test_outside_weeks_listed_once_each():
    with pytest.raises(ValueError, match=r"receipts in weeks \[-9, 7\] fall outside"):
        matrix(weekly_rows=[("TR02-1", 7, 1.0), ("TR02-2", 7, 1.0)],
               early_rows=[("TR02-1", -9, 1.0)], keys=["TR02-1", "TR02-2"])


def test_a_week_with_no_price_level_raises():
    levels = flat_levels(-2, 3).drop(2)
    with pytest.raises(ValueError, match=r"no price level for weeks \[2\]"):
        matrix(levels=levels)


def test_a_nan_price_level_raises():
    levels = flat_levels(-2, 3)
    levels[-1] = float("nan")
    with pytest.raises(ValueError, match=r"no price level for weeks \[-1\]"):
        matrix(levels=levels)


def test_no_price_level_weeks_listed_in_order_at_most_ten():
    with pytest.raises(ValueError) as error:
        matrix(first_week=-5, last_week=9, levels=flat_levels(10, 20))
    assert str(error.value) == "no price level for weeks [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]"


def test_empty_levels_raise():
    with pytest.raises(ValueError, match="no price level"):
        matrix(levels=pd.Series([], dtype="float64"))


# --- weekly_matrix: inputs and determinism ------------------------------------------------

def test_the_inputs_are_not_modified():
    weekly = weekly_table([("TR02-1", 0, 10.0), ("TR02-99999", 1, 5.0)])
    early = early_table([("TR02-1", -1, 4.0)])
    keys = pd.Series(["TR04-1", "TR02-1"], dtype=object)
    weeks = calendar(3, missing=(2,))
    levels = flat_levels(-2, 3)
    copies = [weekly.copy(), early.copy(), keys.copy(), weeks.copy(), levels.copy()]
    weekly_matrix(weekly, early, keys, weeks, levels, -2, 3)
    pd.testing.assert_frame_equal(weekly, copies[0])
    pd.testing.assert_frame_equal(early, copies[1])
    pd.testing.assert_series_equal(keys, copies[2])
    pd.testing.assert_frame_equal(weeks, copies[3])
    pd.testing.assert_series_equal(levels, copies[4])


def test_the_same_inputs_give_the_same_matrix():
    args = dict(weekly_rows=[("TR02-1", 0, 10.0)], early_rows=[("TR02-1", -1, 4.0)],
                keys=["TR02-1", "TR02-2"], missing=(1,))
    pd.testing.assert_frame_equal(matrix(**args), matrix(**args))


def test_row_order_of_the_inputs_does_not_matter():
    rows = [("TR02-1", 0, 10.0), ("TR02-2", 1, 3.0), ("TR02-1", 2, 7.0)]
    forward = matrix(weekly_rows=rows, keys=["TR02-1", "TR02-2"])
    backward = matrix(weekly_rows=rows[::-1], keys=["TR02-2", "TR02-1"])
    pd.testing.assert_frame_equal(forward, backward)


def test_nominal_totals_tie_to_the_sources():
    weekly_rows = [("TR02-1", 0, 10.0), ("TR02-2", 0, 2.5), ("TR02-1", 3, 7.25)]
    early_rows = [("TR02-2", -2, 1.0), ("TR02-2", -1, 9.0)]
    levels = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=range(-2, 4), dtype="float64")
    result = matrix(weekly_rows, early_rows, keys=["TR02-1", "TR02-2"], levels=levels)
    nominal = result.mul(levels.reindex(result.columns), axis=1)
    assert nominal.sum().sum() == pytest.approx(10.0 + 2.5 + 7.25 + 1.0 + 9.0)


# --- check_matrix_totals ---------------------------------------------------------------

def nominal_frame(values, weeks=(0, 1, 2), keys=("TR02-1", "TR02-2")):
    """values: one list per key, one value per week."""
    return pd.DataFrame(values, index=list(keys), columns=list(weeks), dtype="float64")


SOURCE = early_table([("TR02-1", 0, 10.0), ("TR02-2", 0, 5.0), ("TR02-1", 2, 1.0)])


def test_check_passes_when_every_week_ties():
    assert check_matrix_totals(nominal_frame([[10.0, 0.0, 1.0], [5.0, 0.0, 0.0]]),
                               SOURCE) is None


def test_check_passes_within_the_tolerance():
    check_matrix_totals(nominal_frame([[10.004, 0.0, 1.0], [5.0, 0.0, 0.0]]), SOURCE)


def test_check_raises_just_outside_the_tolerance():
    with pytest.raises(ValueError, match=r"the weekly matrix differs from its sources in "
                                         r"weeks \[0\]"):
        check_matrix_totals(nominal_frame([[10.02, 0.0, 1.0], [5.0, 0.0, 0.0]]), SOURCE)


def test_check_raises_when_the_matrix_holds_too_little():
    with pytest.raises(ValueError, match=r"weeks \[2\]"):
        check_matrix_totals(nominal_frame([[10.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), SOURCE)


def test_check_raises_for_cash_in_a_week_with_no_source():
    with pytest.raises(ValueError, match=r"weeks \[1\]"):
        check_matrix_totals(nominal_frame([[10.0, 3.0, 1.0], [5.0, 0.0, 0.0]]), SOURCE)


def test_check_compares_week_totals_not_cells():
    # Cash moved between keys inside one week still ties: the check is per week.
    check_matrix_totals(nominal_frame([[0.0, 0.0, 1.0], [15.0, 0.0, 0.0]]), SOURCE)


def test_check_names_at_most_five_weeks_in_column_order():
    nominal = pd.DataFrame([[1.0] * 8], index=["TR02-1"], columns=list(range(8)))
    with pytest.raises(ValueError) as error:
        check_matrix_totals(nominal, early_table([]))
    assert str(error.value) == ("the weekly matrix differs from its sources in weeks "
                                "[0, 1, 2, 3, 4]")


def test_check_counts_a_nan_cell_as_nothing():
    # A week blanked BEFORE the check would lose its cash: the check catches it.
    with pytest.raises(ValueError, match=r"weeks \[0\]"):
        check_matrix_totals(nominal_frame([[float("nan"), 0.0, 1.0], [5.0, 0.0, 0.0]]),
                            SOURCE)


def test_check_with_an_empty_source_and_an_all_zero_matrix_passes():
    check_matrix_totals(nominal_frame([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), early_table([]))


def test_check_ignores_source_weeks_that_are_not_matrix_columns():
    # Pinned: the source is reindexed to the matrix's weeks, so a receipt in another week is
    # not seen here (weekly_matrix rejects such receipts before calling the check).
    source = early_table([("TR02-1", 0, 10.0), ("TR02-2", 0, 5.0), ("TR02-1", 2, 1.0),
                          ("TR02-1", 9, 99.0)])
    check_matrix_totals(nominal_frame([[10.0, 0.0, 1.0], [5.0, 0.0, 0.0]]), source)


def test_check_rejects_a_third_argument():
    # Its old, unused `missing_weeks` argument is gone.
    with pytest.raises(TypeError):
        check_matrix_totals(nominal_frame([[10.0, 0.0, 1.0], [5.0, 0.0, 0.0]]), SOURCE, [])


def test_check_does_not_modify_its_inputs():
    nominal = nominal_frame([[10.0, 0.0, 1.0], [5.0, 0.0, 0.0]])
    source = SOURCE.copy()
    before = nominal.copy(), source.copy()
    check_matrix_totals(nominal, source)
    pd.testing.assert_frame_equal(nominal, before[0])
    pd.testing.assert_frame_equal(source, before[1])


# --- integration with the synthetic corpus and ledger (SLOW) -----------------------------

CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
HOLIDAYS_FILE = Path(__file__).resolve().parents[2] / "settings" / "holidays_tr.toml"
WEEKLY_GROSS_0_TO_64 = 8_885_177_992.23
EARLY_GROSS_MINUS_31_TO_MINUS_1 = 2_240_423_149.13
UNFILLED_MISSING_WEEKS = [14, 28, 35, 40, 46, 52]


@pytest.fixture(scope="module")
def synthetic_pipeline():
    """Steps 1-11 of run_hurdle_pipeline.main() up to the weekly matrix, read directly from
    data/raw (never the data/interim cache), with filling from the ledger on AND off."""
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    from collection_estimation.hurdle.fill_from_ledger import fill_missing_days
    from collection_estimation.hurdle.holidays import load_holidays
    from collection_estimation.hurdle.keys import add_keys
    from collection_estimation.hurdle.known import known_weeks
    from collection_estimation.hurdle.prices import published_levels
    from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
    from collection_estimation.hurdle.read_ledger import read_ledger
    from collection_estimation.hurdle.series import (add_week_index, week_calendar,
                                                     weekly_amounts)
    from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

    daily_files = list_daily_files(CORPUS_FOLDER, SYNTHETIC_LAYOUT.file_name_date_format)
    collections = add_keys(read_all_files(daily_files, SYNTHETIC_LAYOUT), KNOWN_COMPANIES)
    ledger = add_keys(read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT),
                      KNOWN_COMPANIES)
    holidays = load_holidays(HOLIDAYS_FILE)
    first_file_day = min(daily_files)

    filled, filled_days = fill_missing_days(collections, ledger, list(daily_files))
    unfilled = collections.assign(filled_from_ledger=False)
    known = known_weeks(filled, ledger, WEEK_ZERO)
    levels = published_levels(WEEK_ZERO, -32, 64)
    history_start = first_history_week(ledger, WEEK_ZERO)
    early = ledger_receipts_before(ledger, first_file_day, WEEK_ZERO)
    early = early[early["week"] >= history_start]

    def build(lines, days_filled):
        weeks = week_calendar(list(daily_files), WEEK_ZERO, holidays, days_filled)
        weekly = weekly_amounts(add_week_index(lines, WEEK_ZERO))
        return weekly, weeks, int(weeks["week"].max())

    weekly_on, weeks_on, last_on = build(filled, filled_days)
    weekly_off, weeks_off, last_off = build(unfilled, [])
    return {
        "first_file_day": first_file_day,
        "known": known, "levels": levels, "history_start": history_start, "early": early,
        "weekly_on": weekly_on, "weeks_on": weeks_on, "last_on": last_on,
        "weekly_off": weekly_off, "weeks_off": weeks_off, "last_off": last_off,
        "on": weekly_matrix(weekly_on, early, known["key"], weeks_on, levels,
                            history_start, last_on),
        "off": weekly_matrix(weekly_off, early, known["key"], weeks_off, levels,
                             history_start, last_off),
    }


def nominal_of(result, levels):
    return result.mul(levels.reindex(result.columns), axis=1)


def test_slow_synthetic_week_zero_is_the_monday_of_the_first_file(synthetic_pipeline):
    first_file_day = synthetic_pipeline["first_file_day"]
    assert first_file_day - pd.Timedelta(days=first_file_day.weekday()) == WEEK_ZERO


def test_slow_synthetic_matrix_is_4590_by_96(synthetic_pipeline):
    result = synthetic_pipeline["on"]
    assert synthetic_pipeline["history_start"] == -31
    assert synthetic_pipeline["last_on"] == 64
    assert result.shape == (4590, 96)
    assert result.columns.tolist() == list(range(-31, 65))
    assert result.index.tolist() == sorted(synthetic_pipeline["known"]["key"])


def test_slow_synthetic_matrix_filled_has_no_nan(synthetic_pipeline):
    assert not synthetic_pipeline["on"].isna().any().any()


def test_slow_synthetic_matrix_is_float64_and_non_negative(synthetic_pipeline):
    result = synthetic_pipeline["on"]
    assert (result.dtypes == "float64").all()
    assert (result >= 0).all().all()


def test_slow_synthetic_nominal_from_week_zero_equals_weekly_gross(synthetic_pipeline):
    nominal = nominal_of(synthetic_pipeline["on"], synthetic_pipeline["levels"])
    assert synthetic_pipeline["weekly_on"]["gross_receipts"].sum() == pytest.approx(
        WEEKLY_GROSS_0_TO_64, abs=0.01)
    assert nominal[list(range(0, 65))].sum().sum() == pytest.approx(WEEKLY_GROSS_0_TO_64,
                                                                   abs=0.01)


def test_slow_synthetic_nominal_before_week_zero_equals_early_gross(synthetic_pipeline):
    nominal = nominal_of(synthetic_pipeline["on"], synthetic_pipeline["levels"])
    assert nominal[list(range(-31, 0))].sum().sum() == pytest.approx(
        EARLY_GROSS_MINUS_31_TO_MINUS_1, abs=0.01)


def test_slow_synthetic_nominal_ties_week_by_week(synthetic_pipeline):
    nominal = nominal_of(synthetic_pipeline["on"], synthetic_pipeline["levels"])
    sources = pd.concat([synthetic_pipeline["weekly_on"][["week", "gross_receipts"]],
                         synthetic_pipeline["early"][["week", "gross_receipts"]]])
    per_week = sources.groupby("week")["gross_receipts"].sum().reindex(nominal.columns,
                                                                       fill_value=0.0)
    assert ((nominal.sum() - per_week).abs() <= MATRIX_TOLERANCE).all()


def test_slow_synthetic_unfilled_missing_weeks_are_the_six(synthetic_pipeline):
    weeks = synthetic_pipeline["weeks_off"]
    assert sorted(weeks.loc[weeks["is_missing"], "week"].tolist()) == UNFILLED_MISSING_WEEKS


def test_slow_synthetic_unfilled_matrix_blanks_exactly_the_six_weeks(synthetic_pipeline):
    result = synthetic_pipeline["off"]
    assert result.shape == (4590, 96)
    all_nan = [week for week in result.columns if result[week].isna().all()]
    assert all_nan == UNFILLED_MISSING_WEEKS
    assert not result.drop(columns=UNFILLED_MISSING_WEEKS).isna().any().any()


def test_slow_synthetic_first_week_zero_without_history_has_65_columns(synthetic_pipeline):
    empty = synthetic_pipeline["early"].iloc[0:0]
    result = weekly_matrix(synthetic_pipeline["weekly_on"], empty,
                           synthetic_pipeline["known"]["key"], synthetic_pipeline["weeks_on"],
                           synthetic_pipeline["levels"], 0, synthetic_pipeline["last_on"])
    assert result.shape == (4590, 65)
    assert result.columns.tolist() == list(range(0, 65))
    pd.testing.assert_frame_equal(result, synthetic_pipeline["on"][list(range(0, 65))])


# =========================================================================================
# history_features (U6.2) and its helpers
# =========================================================================================

def feature_matrix(rows, first_week=0, last_week=10):
    """rows: {key: {week: value}}; every other cell is 0.0. Shaped like `weekly_matrix`."""
    columns = list(range(first_week, last_week + 1))
    keys = sorted(rows)
    matrix = pd.DataFrame(0.0, index=pd.Index(keys, name="key", dtype=object),
                          columns=pd.Index(columns, name="week"))
    for key, cells in rows.items():
        for week, value in cells.items():
            matrix.loc[key, week] = value
    return matrix


def known_table(pairs):
    """pairs: {key: known_week}."""
    return pd.DataFrame({"key": pd.Series(list(pairs), dtype=object),
                         "known_week": pd.Series(list(pairs.values()), dtype="int64")})


def feature_row(result, key, week):
    selected = result[(result["key"] == key) & (result["origin_week"] == week)]
    assert len(selected) == 1, f"expected one row for {key} at {week}, got {len(selected)}"
    return selected.iloc[0]


def history_columns(windows):
    longest = max(windows)
    columns = ["key", "origin_week", "customer_age_weeks", "weeks_since_last_payment",
               "last_payment_amount", f"median_payment_{longest}"]
    for window in sorted(windows):
        columns += [f"observed_weeks_{window}", f"paying_weeks_{window}",
                    f"mean_receipts_{window}"]
    return columns


NAN = float("nan")


def history_features_example(windows=(4, 13)):
    """The worked example: A pays 20 in week 2 and 50 in week 8, week 9 has no data; B is
    known in week 9 (no data) and pays 30 in week 10."""
    matrix = feature_matrix({"A": {2: 20.0, 8: 50.0, 9: NAN}, "B": {9: NAN, 10: 30.0}})
    known = known_table({"A": -5, "B": 9})
    return matrix, known, history_features(matrix, known, 10, windows)


@pytest.fixture
def worked_example():
    return history_features_example((4, 13))


# --- the worked example, value by value -------------------------------------------------

def test_worked_example_key_a_at_week_10(worked_example):
    row = feature_row(worked_example[2], "A", 10)
    assert row["observed_weeks_4"] == 3
    assert row["paying_weeks_4"] == 1
    assert row["mean_receipts_4"] == pytest.approx(50.0 / 3)
    assert row["observed_weeks_13"] == 10
    assert row["paying_weeks_13"] == 2
    assert row["mean_receipts_13"] == pytest.approx(7.0)
    assert row["weeks_since_last_payment"] == 2
    assert row["last_payment_amount"] == 50.0
    assert row["median_payment_13"] == 35.0
    assert row["customer_age_weeks"] == 15


def test_worked_example_key_b_at_week_10(worked_example):
    row = feature_row(worked_example[2], "B", 10)
    assert row["observed_weeks_4"] == 3
    assert row["paying_weeks_4"] == 1
    assert row["mean_receipts_4"] == pytest.approx(10.0)
    assert row["observed_weeks_13"] == 10
    assert row["paying_weeks_13"] == 1
    assert row["mean_receipts_13"] == pytest.approx(3.0)
    assert row["weeks_since_last_payment"] == 0
    assert row["last_payment_amount"] == 30.0
    assert row["median_payment_13"] == 30.0
    assert row["customer_age_weeks"] == 1


def test_worked_example_key_b_has_rows_only_for_weeks_9_and_10(worked_example):
    result = worked_example[2]
    assert result.loc[result["key"] == "B", "origin_week"].tolist() == [9, 10]


def test_worked_example_key_b_at_week_9_has_no_payment_features(worked_example):
    row = feature_row(worked_example[2], "B", 9)
    assert math.isnan(row["weeks_since_last_payment"])
    assert math.isnan(row["last_payment_amount"])
    assert math.isnan(row["median_payment_13"])
    assert row["paying_weeks_4"] == 0
    assert row["paying_weeks_13"] == 0
    assert row["customer_age_weeks"] == 0
    # Weeks 6, 7, 8 have data (zeros), week 9 has none.
    assert row["observed_weeks_4"] == 3
    assert row["mean_receipts_4"] == 0.0


def test_worked_example_key_a_at_week_1_has_no_payment_features(worked_example):
    row = feature_row(worked_example[2], "A", 1)
    assert math.isnan(row["weeks_since_last_payment"])
    assert math.isnan(row["last_payment_amount"])
    assert math.isnan(row["median_payment_13"])
    assert row["observed_weeks_4"] == 2      # weeks 0 and 1; -2 and -1 are not in the matrix
    assert row["customer_age_weeks"] == 6


def test_worked_example_key_a_at_week_2_paid_this_week(worked_example):
    row = feature_row(worked_example[2], "A", 2)
    assert row["weeks_since_last_payment"] == 0
    assert row["last_payment_amount"] == 20.0
    assert row["median_payment_13"] == 20.0
    assert row["mean_receipts_4"] == pytest.approx(20.0 / 3)


def test_worked_example_key_a_at_week_9_the_missing_week(worked_example):
    row = feature_row(worked_example[2], "A", 9)
    assert row["weeks_since_last_payment"] == 1
    assert row["last_payment_amount"] == 50.0
    assert row["observed_weeks_4"] == 3        # 6, 7, 8 (9 has no data)
    assert row["mean_receipts_4"] == pytest.approx(50.0 / 3)


def test_worked_example_row_count(worked_example):
    # A: weeks 0..10 (11 rows), B: weeks 9 and 10.
    assert len(worked_example[2]) == 13


# --- missing week is not a zero ---------------------------------------------------------

def test_a_missing_week_lowers_observed_weeks_and_does_not_pull_the_mean_down():
    with_gap = feature_matrix({"A": {0: 10.0, 1: NAN, 2: 10.0, 3: 10.0}}, last_week=3)
    with_zero = feature_matrix({"A": {0: 10.0, 1: 0.0, 2: 10.0, 3: 10.0}}, last_week=3)
    known = known_table({"A": 0})
    gap = feature_row(history_features(with_gap, known, 3, (4,)), "A", 3)
    zero = feature_row(history_features(with_zero, known, 3, (4,)), "A", 3)
    assert gap["observed_weeks_4"] == 3
    assert gap["mean_receipts_4"] == pytest.approx(10.0)
    assert zero["observed_weeks_4"] == 4
    assert zero["mean_receipts_4"] == pytest.approx(7.5)
    assert gap["paying_weeks_4"] == zero["paying_weeks_4"] == 3


def test_a_missing_week_is_not_a_paying_week():
    matrix = feature_matrix({"A": {0: 5.0, 1: NAN}}, last_week=1)
    row = feature_row(history_features(matrix, known_table({"A": 0}), 1, (2,)), "A", 1)
    assert row["paying_weeks_2"] == 1
    assert row["weeks_since_last_payment"] == 1
    assert row["last_payment_amount"] == 5.0


def test_a_window_of_only_missing_weeks_has_observed_zero_and_mean_nan():
    matrix = feature_matrix({"A": {0: 5.0, 1: NAN, 2: NAN}}, last_week=2)
    row = feature_row(history_features(matrix, known_table({"A": 0}), 2, (2,)), "A", 2)
    assert row["observed_weeks_2"] == 0
    assert row["paying_weeks_2"] == 0
    assert math.isnan(row["mean_receipts_2"])
    # The payment two weeks back is still the last payment.
    assert row["weeks_since_last_payment"] == 2
    assert row["last_payment_amount"] == 5.0
    assert math.isnan(row["median_payment_2"])


def test_a_window_of_only_zero_weeks_has_mean_zero_not_nan():
    matrix = feature_matrix({"A": {}}, last_week=3)
    row = feature_row(history_features(matrix, known_table({"A": 0}), 3, (4,)), "A", 3)
    assert row["observed_weeks_4"] == 4
    assert row["paying_weeks_4"] == 0
    assert row["mean_receipts_4"] == 0.0


def test_a_zero_receipt_is_not_a_paying_week():
    matrix = feature_matrix({"A": {0: 7.0, 1: 0.0}}, last_week=1)
    row = feature_row(history_features(matrix, known_table({"A": 0}), 1, (2,)), "A", 1)
    assert row["paying_weeks_2"] == 1
    assert row["weeks_since_last_payment"] == 1
    assert row["last_payment_amount"] == 7.0
    assert row["median_payment_2"] == 7.0


def test_a_negative_receipt_is_not_a_paying_week_but_enters_the_mean():
    matrix = feature_matrix({"A": {0: 10.0, 1: -5.0}}, last_week=1)
    row = feature_row(history_features(matrix, known_table({"A": 0}), 1, (2,)), "A", 1)
    assert row["paying_weeks_2"] == 1
    assert row["observed_weeks_2"] == 2
    assert row["mean_receipts_2"] == pytest.approx(2.5)
    assert row["weeks_since_last_payment"] == 1
    assert row["last_payment_amount"] == 10.0
    assert row["median_payment_2"] == 10.0


def test_a_tiny_positive_receipt_is_a_paying_week():
    matrix = feature_matrix({"A": {1: 1e-12}}, last_week=1)
    row = feature_row(history_features(matrix, known_table({"A": 0}), 1, (2,)), "A", 1)
    assert row["paying_weeks_2"] == 1
    assert row["weeks_since_last_payment"] == 0
    assert row["last_payment_amount"] == 1e-12


# --- windows reaching before the matrix's first week -------------------------------------

def test_weeks_before_the_first_column_count_as_no_data():
    matrix = feature_matrix({"A": {1: 8.0}}, first_week=0, last_week=3)
    result = history_features(matrix, known_table({"A": 0}), 3, (4,))
    assert feature_row(result, "A", 0)["observed_weeks_4"] == 1
    assert feature_row(result, "A", 1)["observed_weeks_4"] == 2
    assert feature_row(result, "A", 1)["mean_receipts_4"] == pytest.approx(4.0)
    assert feature_row(result, "A", 3)["observed_weeks_4"] == 4
    assert feature_row(result, "A", 3)["mean_receipts_4"] == pytest.approx(2.0)


def test_a_matrix_from_week_minus_31_fills_the_window_from_the_start():
    matrix = feature_matrix({"A": {1: 8.0}}, first_week=-31, last_week=3)
    result = history_features(matrix, known_table({"A": -40}), 3, (4, 52))
    row0 = feature_row(result, "A", 0)
    assert row0["observed_weeks_4"] == 4
    assert row0["observed_weeks_52"] == 32          # weeks -31 .. 0
    row1 = feature_row(result, "A", 1)
    assert row1["observed_weeks_4"] == 4
    assert row1["mean_receipts_4"] == pytest.approx(2.0)
    assert row1["observed_weeks_52"] == 33


def test_early_weeks_change_features_only_through_the_window():
    # Weeks 0..10 with and without 31 earlier zero weeks: equal once the window no longer
    # reaches before week 0.
    late = feature_matrix({"A": {3: 4.0, 7: 9.0}}, first_week=0)
    early = feature_matrix({"A": {3: 4.0, 7: 9.0}}, first_week=-31)
    known = known_table({"A": 0})
    from_zero = history_features(late, known, 10, (4,))
    from_minus_31 = history_features(early, known, 10, (4,))
    for week in range(3, 11):
        assert (feature_row(from_zero, "A", week)["observed_weeks_4"]
                == feature_row(from_minus_31, "A", week)["observed_weeks_4"])
        assert (feature_row(from_zero, "A", week)["mean_receipts_4"]
                == pytest.approx(feature_row(from_minus_31, "A", week)["mean_receipts_4"]))
    assert feature_row(from_zero, "A", 2)["observed_weeks_4"] == 3
    assert feature_row(from_minus_31, "A", 2)["observed_weeks_4"] == 4


def test_a_payment_before_week_zero_is_the_last_payment():
    matrix = feature_matrix({"A": {-31: 12.0}}, first_week=-31, last_week=5)
    result = history_features(matrix, known_table({"A": -31}), 5, (4, 52))
    row = feature_row(result, "A", 5)
    assert row["weeks_since_last_payment"] == 36
    assert row["last_payment_amount"] == 12.0
    assert row["median_payment_52"] == 12.0      # week -31 is within the 52 weeks ending 5
    assert row["paying_weeks_52"] == 1
    assert row["paying_weeks_4"] == 0
    assert row["customer_age_weeks"] == 36


def test_the_median_covers_only_the_longest_window_but_the_last_payment_does_not():
    matrix = feature_matrix({"A": {0: 100.0, 5: 10.0}}, last_week=8)
    result = history_features(matrix, known_table({"A": 0}), 8, (2,))
    assert feature_row(result, "A", 1)["median_payment_2"] == 100.0
    assert math.isnan(feature_row(result, "A", 2)["median_payment_2"])
    assert feature_row(result, "A", 5)["median_payment_2"] == 10.0
    assert feature_row(result, "A", 6)["median_payment_2"] == 10.0
    late = feature_row(result, "A", 7)
    assert math.isnan(late["median_payment_2"])
    assert late["weeks_since_last_payment"] == 2
    assert late["last_payment_amount"] == 10.0


def test_the_median_skips_zero_negative_and_missing_weeks():
    matrix = feature_matrix({"A": {0: 3.0, 1: 0.0, 2: -50.0, 3: NAN, 4: 9.0, 5: 6.0}},
                            last_week=5)
    row = feature_row(history_features(matrix, known_table({"A": 0}), 5, (6,)), "A", 5)
    assert row["median_payment_6"] == 6.0       # median of 3, 9, 6
    assert row["paying_weeks_6"] == 3


def test_a_window_of_one_week_is_that_week_alone():
    matrix = feature_matrix({"A": {2: 5.0, 3: NAN}}, last_week=4)
    result = history_features(matrix, known_table({"A": 0}), 4, (1,))
    assert feature_row(result, "A", 2)["mean_receipts_1"] == 5.0
    assert feature_row(result, "A", 2)["paying_weeks_1"] == 1
    assert feature_row(result, "A", 3)["observed_weeks_1"] == 0
    assert math.isnan(feature_row(result, "A", 3)["mean_receipts_1"])
    assert feature_row(result, "A", 4)["mean_receipts_1"] == 0.0
    assert math.isnan(feature_row(result, "A", 4)["median_payment_1"])


def test_a_window_longer_than_the_matrix_uses_every_column_so_far():
    matrix = feature_matrix({"A": {0: 2.0, 3: 4.0}}, last_week=3)
    row = feature_row(history_features(matrix, known_table({"A": 0}), 3, (100,)), "A", 3)
    assert row["observed_weeks_100"] == 4
    assert row["mean_receipts_100"] == pytest.approx(1.5)
    assert row["median_payment_100"] == 3.0


# --- the row set, order, columns and dtypes ----------------------------------------------

def test_rows_are_exactly_the_known_keys_at_their_known_origin_weeks():
    matrix = feature_matrix({"A": {}, "B": {}, "C": {}, "D": {}}, first_week=-3, last_week=6)
    known_at = {"A": -3, "B": 0, "C": 4, "D": 6}
    result = history_features(matrix, known_table(known_at), 6, (4,))
    expected = sorted([(key, t) for key, week in known_at.items() for t in range(0, 7)
                       if week <= t], key=lambda pair: (pair[1], pair[0]))
    assert list(zip(result["key"], result["origin_week"])) == expected


def test_a_key_known_at_last_week_gets_one_row():
    matrix = feature_matrix({"A": {}})
    result = history_features(matrix, known_table({"A": 10}), 10, (4,))
    assert result["origin_week"].tolist() == [10]


def test_a_key_known_after_last_week_gets_no_rows():
    matrix = feature_matrix({"A": {}, "B": {}}, last_week=12)
    result = history_features(matrix, known_table({"A": 0, "B": 11}), 10, (4,))
    assert set(result["key"]) == {"A"}
    assert result["origin_week"].max() == 10


def test_no_origin_week_beyond_last_week_even_when_the_matrix_goes_further():
    matrix = feature_matrix({"A": {}}, last_week=20)
    result = history_features(matrix, known_table({"A": 0}), 10, (4,))
    assert result["origin_week"].tolist() == list(range(11))


def test_a_key_in_the_matrix_but_not_known_gets_no_rows():
    matrix = feature_matrix({"A": {3: 1.0}, "Z": {3: 99.0}})
    result = history_features(matrix, known_table({"A": 0}), 10, (4,))
    assert set(result["key"]) == {"A"}


def test_a_negative_known_week_gives_rows_from_week_zero_and_its_age():
    matrix = feature_matrix({"A": {}}, first_week=-10)
    result = history_features(matrix, known_table({"A": -7}), 10, (4,))
    assert result["origin_week"].tolist() == list(range(11))
    assert result["customer_age_weeks"].tolist() == [t + 7 for t in range(11)]


def test_rows_are_sorted_by_origin_week_then_key():
    matrix = feature_matrix({"b": {}, "A": {}, "a": {}, "B": {}}, last_week=3)
    known = known_table({"b": 0, "a": 1, "B": 0, "A": 2})
    result = history_features(matrix, known, 3, (2,))
    pairs = list(zip(result["origin_week"], result["key"]))
    assert pairs == sorted(pairs)
    assert [key for week, key in pairs if week == 3] == ["A", "B", "a", "b"]


def test_the_index_is_a_fresh_range():
    _, _, result = history_features_example()
    assert result.index.equals(pd.RangeIndex(len(result)))


def test_columns_are_in_the_documented_order():
    _, _, result = history_features_example((4, 13))
    assert list(result.columns) == history_columns((4, 13))


def test_unsorted_windows_give_sorted_columns_and_the_median_uses_the_longest():
    _, _, result = history_features_example((52, 4, 13))
    assert list(result.columns) == [
        "key", "origin_week", "customer_age_weeks", "weeks_since_last_payment",
        "last_payment_amount", "median_payment_52",
        "observed_weeks_4", "paying_weeks_4", "mean_receipts_4",
        "observed_weeks_13", "paying_weeks_13", "mean_receipts_13",
        "observed_weeks_52", "paying_weeks_52", "mean_receipts_52"]


def test_unsorted_windows_give_the_same_values_as_sorted_ones():
    _, _, unsorted = history_features_example((52, 4, 13))
    _, _, ordered = history_features_example((4, 13, 52))
    pd.testing.assert_frame_equal(unsorted, ordered)


def test_windows_may_be_a_list():
    _, _, as_list = history_features_example([13, 4])
    _, _, as_tuple = history_features_example((4, 13))
    pd.testing.assert_frame_equal(as_list, as_tuple)


def test_a_single_window_gives_its_median_column():
    _, _, result = history_features_example((4,))
    assert list(result.columns) == history_columns((4,))


def test_dtypes():
    _, _, result = history_features_example((4, 13))
    expected = {"key": "object", "origin_week": "int64", "customer_age_weeks": "int64",
                "weeks_since_last_payment": "float64", "last_payment_amount": "float64",
                "median_payment_13": "float64"}
    for window in (4, 13):
        expected[f"observed_weeks_{window}"] = "int64"
        expected[f"paying_weeks_{window}"] = "int64"
        expected[f"mean_receipts_{window}"] = "float64"
    assert {column: str(dtype) for column, dtype in result.dtypes.items()} == expected


def test_no_known_keys_gives_an_empty_table_with_the_columns_and_dtypes():
    matrix = feature_matrix({"A": {}})
    result = history_features(matrix, known_table({}), 10, (4, 13))
    assert result.empty
    assert list(result.columns) == history_columns((4, 13))
    assert result["key"].dtype == object
    assert result["origin_week"].dtype == "int64"
    assert result["observed_weeks_4"].dtype == "int64"
    assert result["mean_receipts_4"].dtype == "float64"


def test_extra_known_columns_are_ignored():
    matrix, known, result = history_features_example()
    wider = known.assign(company_code="TR02", customer_id=["1", "2"],
                         first_seen=pd.Timestamp("2025-06-02"))
    pd.testing.assert_frame_equal(history_features(matrix, wider, 10, (4, 13)), result)


def test_matrix_and_known_row_order_do_not_matter():
    matrix, known, result = history_features_example()
    shuffled = history_features(matrix.iloc[::-1], known.iloc[::-1].reset_index(drop=True),
                                10, (4, 13))
    pd.testing.assert_frame_equal(shuffled, result)


def test_a_matrix_without_index_or_column_names_gives_the_same_result():
    matrix, known, result = history_features_example()
    plain = matrix.copy()
    plain.index.name = None
    plain.columns.name = None
    pd.testing.assert_frame_equal(history_features(plain, known, 10, (4, 13)), result)


def test_the_same_inputs_give_the_same_features():
    matrix, known, result = history_features_example()
    pd.testing.assert_frame_equal(history_features(matrix, known, 10, (4, 13)), result)


def test_history_features_does_not_modify_its_inputs():
    matrix = feature_matrix({"A": {2: 20.0, 8: 50.0, 9: NAN}, "B": {9: NAN, 10: 30.0}},
                            first_week=-3)
    known = known_table({"A": -5, "B": 9})
    before = matrix.copy(), known.copy()
    history_features(matrix, known, 10, (4, 13))
    pd.testing.assert_frame_equal(matrix, before[0])
    pd.testing.assert_frame_equal(known, before[1])
    assert list(known.columns) == ["key", "known_week"]


def test_the_windows_list_is_not_modified():
    windows = [13, 4]
    matrix, known, _ = history_features_example()
    history_features(matrix, known, 10, windows)
    assert windows == [13, 4]


# --- random matrices: a brute-force reference --------------------------------------------

def random_matrix(seed, n_keys=12, first_week=-6, last_week=25, missing_weeks=(3, 11)):
    """Zeros, NaN cells, negative cells, whole missing weeks; two keys are not known and the
    known weeks run from before the first column to after the last."""
    rng = np.random.default_rng(seed)
    columns = list(range(first_week, last_week + 1))
    keys = [f"K{i:02d}" for i in range(n_keys)]
    draw = rng.random((n_keys, len(columns)))
    values = np.round(rng.lognormal(3.0, 1.0, (n_keys, len(columns))), 2)
    values[draw < 0.5] = 0.0
    values[(draw >= 0.5) & (draw < 0.6)] = np.nan
    values[(draw >= 0.6) & (draw < 0.65)] *= -1.0
    matrix = pd.DataFrame(values, index=pd.Index(keys, name="key", dtype=object),
                          columns=pd.Index(columns, name="week"))
    for week in missing_weeks:
        matrix[week] = np.nan               # a week the calendar marks missing
    known_at = rng.integers(first_week - 4, last_week + 4, n_keys)
    known = known_table(dict(zip(keys[:-2], known_at[:-2].tolist())))
    return matrix, known


def reference_history(matrix, known, last_week, windows):
    """The features by plain loops over the definitions."""
    columns = list(matrix.columns)
    longest = max(windows)
    known_at = dict(zip(known["key"], known["known_week"]))
    records = []
    for t in range(last_week + 1):
        for key in sorted(k for k, week in known_at.items() if week <= t):
            row = matrix.loc[key]

            def cell(week):
                return row[week] if week in columns else NAN

            paying = [week for week in columns if week <= t and cell(week) > 0]
            record = {"key": key, "origin_week": t, "customer_age_weeks": t - known_at[key]}
            if paying:
                record["weeks_since_last_payment"] = float(t - paying[-1])
                record["last_payment_amount"] = float(cell(paying[-1]))
            else:
                record["weeks_since_last_payment"] = NAN
                record["last_payment_amount"] = NAN
            recent = [cell(week) for week in paying if week > t - longest]
            record[f"median_payment_{longest}"] = (float(np.median(recent)) if recent
                                                   else NAN)
            for window in sorted(windows):
                weeks = range(t - window + 1, t + 1)
                values = [cell(week) for week in weeks if not math.isnan(cell(week))]
                record[f"observed_weeks_{window}"] = len(values)
                record[f"paying_weeks_{window}"] = sum(1 for value in values if value > 0)
                record[f"mean_receipts_{window}"] = (sum(values) / len(values) if values
                                                     else NAN)
            records.append(record)
    table = pd.DataFrame(records, columns=history_columns(windows))
    table["key"] = table["key"].astype(object)
    for column in table.columns:
        if column.startswith(("origin_week", "customer_age", "observed_", "paying_")):
            table[column] = table[column].astype("int64")
        elif column != "key":
            table[column] = table[column].astype("float64")
    return table


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("windows", [(4, 13, 52), (1,), (3, 8), (2, 5, 40)])
def test_matches_a_brute_force_reference_on_random_matrices(seed, windows):
    matrix, known = random_matrix(seed)
    result = history_features(matrix, known, 25, windows)
    expected = reference_history(matrix, known, 25, windows)
    pd.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-9, atol=1e-9)


def test_matches_the_reference_when_the_matrix_starts_at_week_zero():
    matrix, known = random_matrix(7, first_week=0, last_week=20, missing_weeks=(0, 9))
    result = history_features(matrix, known, 20, (4, 13))
    expected = reference_history(matrix, known, 20, (4, 13))
    pd.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-9, atol=1e-9)


def test_matches_the_reference_with_last_week_short_of_the_matrix_end():
    matrix, known = random_matrix(8)
    result = history_features(matrix, known, 15, (4, 13))
    expected = reference_history(matrix, known, 15, (4, 13))
    pd.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-9, atol=1e-9)


# --- the leak test: nothing after t changes the features at t -----------------------------

def scramble_after(matrix, week, seed):
    """Every cell after `week` replaced: NaN, huge, negative, zero or a new amount."""
    rng = np.random.default_rng(seed)
    scrambled = matrix.copy()
    later = [column for column in matrix.columns if column > week]
    choices = np.array([NAN, 1e15, -1e6, 0.0, 123.45, 1e-9])
    scrambled[later] = rng.choice(choices, size=(len(matrix), len(later)))
    return scrambled


@pytest.mark.parametrize("cut", [0, 1, 5, 12, 20, 24])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scrambling_weeks_after_t_leaves_features_up_to_t_unchanged(cut, seed):
    matrix, known = random_matrix(seed)
    windows = (4, 13, 52)
    original = history_features(matrix, known, 25, windows)
    changed = history_features(scramble_after(matrix, cut, seed + 100), known, 25, windows)
    keep = original["origin_week"] <= cut
    pd.testing.assert_frame_equal(changed[keep], original[keep], check_exact=True)
    # The scramble did take effect: the later origin weeks moved.
    assert not changed[~keep].equals(original[~keep])


@pytest.mark.parametrize("cut", [0, 3, 11, 19])
def test_blanking_weeks_after_t_leaves_features_up_to_t_unchanged(cut):
    matrix, known = random_matrix(5)
    blanked = matrix.copy()
    blanked[[column for column in matrix.columns if column > cut]] = NAN
    original = history_features(matrix, known, 25, (4, 13))
    changed = history_features(blanked, known, 25, (4, 13))
    keep = original["origin_week"] <= cut
    pd.testing.assert_frame_equal(changed[keep], original[keep], check_exact=True)


@pytest.mark.parametrize("cut", [0, 2, 10, 17, 25])
def test_a_matrix_cut_at_t_gives_the_same_features_up_to_t(cut):
    matrix, known = random_matrix(9)
    full = history_features(matrix, known, 25, (4, 13, 52))
    truncated = matrix[[column for column in matrix.columns if column <= cut]]
    cut_result = history_features(truncated, known, cut, (4, 13, 52))
    expected = full[full["origin_week"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(cut_result, expected, check_exact=True)


def test_a_huge_payment_after_t_does_not_reach_t():
    matrix = feature_matrix({"A": {3: 10.0}})
    later = matrix.copy()
    later.loc["A", 4] = 1e12
    known = known_table({"A": 0})
    before = history_features(matrix, known, 10, (4,))
    after = history_features(later, known, 10, (4,))
    keep = before["origin_week"] <= 3
    pd.testing.assert_frame_equal(after[keep], before[keep], check_exact=True)
    assert feature_row(after, "A", 4)["last_payment_amount"] == 1e12


# --- check_matrix_for_features ------------------------------------------------------------

def test_check_passes_on_a_good_matrix():
    matrix = feature_matrix({"A": {}}, first_week=-3)
    check_matrix_for_features(matrix, known_table({"A": 0}), 10, (4,))


def test_a_known_key_without_a_matrix_row_raises():
    matrix = feature_matrix({"A": {}})
    with pytest.raises(ValueError, match=re.escape(
            "1 known keys have no row in the weekly matrix (first: ['C'])")):
        history_features(matrix, known_table({"A": 0, "C": 2}), 10, (4,))


def test_missing_keys_are_counted_and_the_first_five_listed_sorted():
    matrix = feature_matrix({"A": {}})
    known = known_table({key: 0 for key in ["A", "G", "F", "E", "D", "C", "B", "H"]})
    with pytest.raises(ValueError, match=re.escape(
            "7 known keys have no row in the weekly matrix "
            "(first: ['B', 'C', 'D', 'E', 'F'])")):
        check_matrix_for_features(matrix, known, 10, (4,))


def test_a_known_key_after_last_week_still_needs_a_matrix_row():
    matrix = feature_matrix({"A": {}})
    with pytest.raises(ValueError, match="1 known keys have no row"):
        check_matrix_for_features(matrix, known_table({"A": 0, "B": 99}), 10, (4,))


def test_a_gap_in_the_columns_raises():
    matrix = feature_matrix({"A": {}}).drop(columns=5)
    with pytest.raises(ValueError, match="columns must be consecutive week numbers"):
        history_features(matrix, known_table({"A": 0}), 10, (4,))


def test_descending_columns_raise():
    matrix = feature_matrix({"A": {}})
    matrix = matrix[list(reversed(matrix.columns))]
    with pytest.raises(ValueError, match="columns must be consecutive week numbers"):
        history_features(matrix, known_table({"A": 0}), 10, (4,))


def test_shuffled_columns_raise():
    matrix = feature_matrix({"A": {}})
    matrix = matrix[[0, 2, 1] + list(range(3, 11))]
    with pytest.raises(ValueError, match="columns must be consecutive week numbers"):
        check_matrix_for_features(matrix, known_table({"A": 0}), 10, (4,))


def test_a_repeated_column_raises():
    matrix = feature_matrix({"A": {}})
    matrix = pd.concat([matrix, matrix[[10]]], axis=1)
    with pytest.raises(ValueError, match="columns must be consecutive week numbers"):
        check_matrix_for_features(matrix, known_table({"A": 0}), 10, (4,))


def test_no_columns_raise():
    matrix = feature_matrix({"A": {}})[[]]
    with pytest.raises(ValueError, match="columns must be consecutive week numbers"):
        history_features(matrix, known_table({"A": 0}), 10, (4,))


@pytest.mark.parametrize("labels", [
    pytest.param([str(week) for week in range(11)], id="text"),
    pytest.param([float(week) for week in range(11)], id="float"),
    pytest.param([week + 0.5 for week in range(11)], id="fractional-float"),
    pytest.param(pd.Index(list(range(10)) + ["10"], dtype=object), id="object-mixed"),
    pytest.param(pd.Index(list(range(11)), dtype=object), id="object-all-int"),
])
def test_non_integer_columns_raise_value_error(labels):
    matrix = feature_matrix({"A": {}})
    matrix.columns = labels
    with pytest.raises(ValueError, match=re.escape(
            "the weekly matrix's columns must be consecutive week numbers")):
        check_matrix_for_features(matrix, known_table({"A": 0}), 10, (4,))


def test_non_integer_columns_raise_through_history_features_too():
    matrix = feature_matrix({"A": {}})
    matrix.columns = [float(week) for week in range(11)]
    with pytest.raises(ValueError, match="consecutive week numbers"):
        history_features(matrix, known_table({"A": 0}), 10, (4,))


@pytest.mark.parametrize("dtype", ["int64", "int32", "Int64"])
def test_integer_columns_of_any_integer_dtype_pass(dtype):
    matrix = feature_matrix({"A": {3: 5.0}})
    matrix.columns = pd.Index(list(range(11)), dtype=dtype, name="week")
    check_matrix_for_features(matrix, known_table({"A": 0}), 10, (4,))
    result = history_features(matrix, known_table({"A": 0}), 10, (4,))
    assert feature_row(result, "A", 3)["last_payment_amount"] == 5.0


# --- check_matrix_for_features: a key repeated in `known` ---------------------------------

def known_rows(pairs):
    """pairs: [(key, known_week), ...]; unlike `known_table`, keys may repeat."""
    return pd.DataFrame({"key": pd.Series([key for key, _ in pairs], dtype=object),
                         "known_week": pd.Series([week for _, week in pairs], dtype="int64")})


def test_a_repeated_known_key_raises_with_the_exact_message():
    matrix = feature_matrix({"A": {}, "B": {}})
    known = known_rows([("A", 0), ("B", 1), ("A", 0)])
    with pytest.raises(ValueError, match=re.escape(
            "1 keys appear more than once in the known table (first: ['A'])")):
        check_matrix_for_features(matrix, known, 10, (4,))


def test_a_repeated_known_key_raises_even_with_different_known_weeks():
    matrix = feature_matrix({"A": {}})
    with pytest.raises(ValueError, match=re.escape(
            "1 keys appear more than once in the known table (first: ['A'])")):
        check_matrix_for_features(matrix, known_rows([("A", 0), ("A", 3)]), 10, (4,))


def test_a_key_three_times_counts_as_one_repeated_key():
    matrix = feature_matrix({"A": {}, "B": {}})
    known = known_rows([("A", 0), ("A", 0), ("B", 0), ("A", 0)])
    with pytest.raises(ValueError, match=re.escape(
            "1 keys appear more than once in the known table (first: ['A'])")):
        check_matrix_for_features(matrix, known, 10, (4,))


def test_repeated_keys_are_counted_distinct_and_the_first_five_listed_sorted():
    keys = ["G", "F", "E", "D", "C", "B", "A", "H"]
    matrix = feature_matrix({key: {} for key in keys})
    # Seven distinct keys repeated (H is not), G three times; listed out of order.
    pairs = [(key, 0) for key in keys] + [(key, 0) for key in keys[:7]] + [("G", 0)]
    with pytest.raises(ValueError, match=re.escape(
            "7 keys appear more than once in the known table "
            "(first: ['A', 'B', 'C', 'D', 'E'])")):
        check_matrix_for_features(matrix, known_rows(pairs), 10, (4,))


def test_a_repeated_known_key_raises_through_history_features():
    matrix = feature_matrix({"A": {2: 1.0}})
    with pytest.raises(ValueError, match=re.escape(
            "1 keys appear more than once in the known table (first: ['A'])")):
        history_features(matrix, known_rows([("A", 0), ("A", 0)]), 10, (4,))


def test_the_repeated_key_check_comes_before_the_missing_key_check():
    matrix = feature_matrix({"A": {}})
    # B repeats and also has no matrix row; C is only missing.
    known = known_rows([("A", 0), ("B", 0), ("B", 0), ("C", 0)])
    with pytest.raises(ValueError, match=re.escape(
            "1 keys appear more than once in the known table (first: ['B'])")):
        check_matrix_for_features(matrix, known, 10, (4,))


def test_the_repeated_key_check_comes_before_the_column_and_window_checks():
    matrix = feature_matrix({"A": {}})
    matrix.columns = [str(week) for week in range(11)]
    with pytest.raises(ValueError, match="keys appear more than once"):
        check_matrix_for_features(matrix, known_rows([("A", 0), ("A", 0)]), 10, ())


def test_distinct_known_keys_do_not_trip_the_repeated_key_check():
    matrix = feature_matrix({"A": {}, "a": {}, "B": {}})
    check_matrix_for_features(matrix, known_rows([("A", 0), ("a", 0), ("B", 5)]), 10, (4,))


def test_columns_starting_after_week_zero_raise():
    matrix = feature_matrix({"A": {}}, first_week=1)
    with pytest.raises(ValueError, match=re.escape(
            "the weekly matrix has weeks 1 to 10; it must cover weeks 0 to 10")):
        history_features(matrix, known_table({"A": 1}), 10, (4,))


def test_columns_ending_before_last_week_raise():
    matrix = feature_matrix({"A": {}}, last_week=9)
    with pytest.raises(ValueError, match=re.escape(
            "the weekly matrix has weeks 0 to 9; it must cover weeks 0 to 10")):
        history_features(matrix, known_table({"A": 0}), 10, (4,))


def test_columns_exactly_zero_to_last_week_pass():
    check_matrix_for_features(feature_matrix({"A": {}}), known_table({"A": 0}), 10, (4,))


def test_a_matrix_entirely_before_week_zero_raises():
    matrix = feature_matrix({"A": {}}, first_week=-5, last_week=-1)
    with pytest.raises(ValueError, match="it must cover weeks 0 to 0"):
        check_matrix_for_features(matrix, known_table({"A": 0}), 0, (4,))


@pytest.mark.parametrize("windows", [(), [], (4, 0), (0,), (-1,), (13, -4, 52)])
def test_bad_windows_raise(windows):
    with pytest.raises(ValueError, match=re.escape(
            f"windows are {windows}; give at least one, each 1 week or more")):
        history_features(feature_matrix({"A": {}}), known_table({"A": 0}), 10, windows)


def test_a_window_of_one_passes():
    check_matrix_for_features(feature_matrix({"A": {}}), known_table({"A": 0}), 10, (1,))


def test_the_check_does_not_modify_its_inputs():
    matrix, known = feature_matrix({"A": {1: 3.0}}), known_table({"A": 0})
    before = matrix.copy(), known.copy()
    check_matrix_for_features(matrix, known, 10, (4,))
    pd.testing.assert_frame_equal(matrix, before[0])
    pd.testing.assert_frame_equal(known, before[1])


# --- key_origins ---------------------------------------------------------------------------

def test_key_origins_lists_each_key_from_its_known_week():
    result = key_origins(known_table({"B": 1, "A": 2, "C": -4}), 2)
    assert list(result.columns) == ["key", "origin_week"]
    assert list(zip(result["key"], result["origin_week"])) == [
        ("C", 0), ("B", 1), ("C", 1), ("A", 2), ("B", 2), ("C", 2)]


def test_key_origins_drops_keys_known_after_last_week():
    result = key_origins(known_table({"A": 0, "B": 3}), 2)
    assert set(result["key"]) == {"A"}


def test_key_origins_dtypes_and_index():
    result = key_origins(known_table({"A": 0, "B": 1}), 3)
    assert result["key"].dtype == object
    assert result["origin_week"].dtype == "int64"
    assert result.index.equals(pd.RangeIndex(len(result)))


def test_key_origins_with_last_week_zero():
    result = key_origins(known_table({"A": -2, "B": 0, "C": 1}), 0)
    assert list(zip(result["key"], result["origin_week"])) == [("A", 0), ("B", 0)]


def test_key_origins_of_no_keys_is_empty_with_the_columns():
    result = key_origins(known_table({}), 5)
    assert result.empty
    assert list(result.columns) == ["key", "origin_week"]


# --- trailing_sum ---------------------------------------------------------------------------

def test_trailing_sum_sums_the_window_ending_at_each_column():
    frame = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=["A"], columns=[0, 1, 2, 3])
    result = trailing_sum(frame, 2)
    assert result.loc["A"].tolist() == [1.0, 3.0, 5.0, 7.0]
    assert list(result.columns) == [0, 1, 2, 3]
    assert list(result.index) == ["A"]


def test_trailing_sum_uses_fewer_columns_at_the_start():
    frame = pd.DataFrame([[1.0, 2.0, 3.0]], index=["A"], columns=[5, 6, 7])
    assert trailing_sum(frame, 10).loc["A"].tolist() == [1.0, 3.0, 6.0]


def test_trailing_sum_window_one_is_the_frame():
    frame = pd.DataFrame([[1.0, 0.0], [2.5, 4.0]], index=["A", "B"], columns=[0, 1])
    pd.testing.assert_frame_equal(trailing_sum(frame, 1), frame)


def test_trailing_sum_keeps_rows_apart():
    frame = pd.DataFrame([[1.0, 1.0], [10.0, 10.0]], index=["A", "B"], columns=[0, 1])
    result = trailing_sum(frame, 2)
    assert result.loc["A"].tolist() == [1.0, 2.0]
    assert result.loc["B"].tolist() == [10.0, 20.0]


# --- values_at ------------------------------------------------------------------------------

def test_values_at_picks_each_rows_cell_in_row_order():
    wide = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], index=["A", "B"],
                        columns=[-1, 0, 1])
    rows = pd.DataFrame({"key": ["B", "A", "B"], "origin_week": [1, 0, -1]},
                        index=[7, 3, 5])
    result = values_at(wide, rows)
    assert result.tolist() == [6.0, 2.0, 4.0]
    assert result.index.tolist() == [7, 3, 5]


def test_values_at_keeps_nan():
    wide = pd.DataFrame([[NAN, 2.0]], index=["A"], columns=[0, 1])
    result = values_at(wide, pd.DataFrame({"key": ["A", "A"], "origin_week": [0, 1]}))
    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == 2.0


SENTINEL = 999.0


def sentinel_wide():
    """Keys A, B x weeks 0, 1, 2; the last cell (B, 2) holds a sentinel, the value a
    get_indexer -1 on both axes would read."""
    return pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, SENTINEL]], index=["A", "B"],
                        columns=[0, 1, 2])


def lookup(keys, weeks, index=None):
    return pd.DataFrame({"key": list(keys), "origin_week": list(weeks)}, index=index)


def test_values_at_a_missing_key_raises_with_the_exact_message():
    with pytest.raises(ValueError, match=r"^" + re.escape("keys ['Z'] are not in the table") + r"$"):
        values_at(sentinel_wide(), lookup(["Z"], [2]))


def test_values_at_a_missing_week_raises_with_the_exact_message():
    with pytest.raises(ValueError, match=r"^" + re.escape("weeks [7] are not in the table") + r"$"):
        values_at(sentinel_wide(), lookup(["A"], [7]))


def test_values_at_a_missing_negative_week_raises():
    with pytest.raises(ValueError, match=r"^" + re.escape("weeks [-1] are not in the table") + r"$"):
        values_at(sentinel_wide(), lookup(["A"], [-1]))


def test_values_at_a_missing_key_does_not_return_the_last_rows_value():
    # Old behaviour: -1 row position read row B, so ("Z", 2) gave the sentinel.
    with pytest.raises(ValueError):
        values_at(sentinel_wide(), lookup(["Z"], [2]))


def test_values_at_a_missing_week_does_not_return_the_last_columns_value():
    # Old behaviour: -1 column position read week 2, so ("B", 7) gave the sentinel.
    with pytest.raises(ValueError):
        values_at(sentinel_wide(), lookup(["B"], [7]))


def test_values_at_a_missing_key_among_valid_rows_still_raises():
    with pytest.raises(ValueError, match=re.escape("keys ['Z']")):
        values_at(sentinel_wide(), lookup(["A", "Z", "B"], [0, 2, 1]))


def test_values_at_a_missing_week_among_valid_rows_still_raises():
    with pytest.raises(ValueError, match=re.escape("weeks [3]")):
        values_at(sentinel_wide(), lookup(["A", "B", "A"], [0, 3, 1]))


def test_values_at_missing_keys_are_listed_sorted_distinct_at_most_five():
    keys = ["K7", "K3", "A", "K1", "K3", "K9", "K5", "K2", "K7", "K8"]
    with pytest.raises(ValueError) as caught:
        values_at(sentinel_wide(), lookup(keys, [0] * len(keys)))
    assert str(caught.value) == "keys ['K1', 'K2', 'K3', 'K5', 'K7'] are not in the table"


def test_values_at_missing_weeks_are_listed_sorted_distinct_at_most_five():
    weeks = [40, 5, 0, -3, 5, 12, 40, 8, 30, 1]
    with pytest.raises(ValueError) as caught:
        values_at(sentinel_wide(), lookup(["A"] * len(weeks), weeks))
    assert str(caught.value) == "weeks [-3, 5, 8, 12, 30] are not in the table"


def test_values_at_a_repeated_missing_key_is_listed_once():
    with pytest.raises(ValueError) as caught:
        values_at(sentinel_wide(), lookup(["Z", "Z", "Z"], [0, 1, 2]))
    assert str(caught.value) == "keys ['Z'] are not in the table"


def test_values_at_exactly_five_missing_weeks_are_all_listed():
    weeks = [9, 7, 5, 4, 3]
    with pytest.raises(ValueError) as caught:
        values_at(sentinel_wide(), lookup(["A"] * 5, weeks))
    assert str(caught.value) == "weeks [3, 4, 5, 7, 9] are not in the table"


def test_values_at_the_key_check_runs_before_the_week_check():
    with pytest.raises(ValueError) as caught:
        values_at(sentinel_wide(), lookup(["Z", "A"], [0, 7]))
    assert str(caught.value) == "keys ['Z'] are not in the table"


def test_values_at_a_row_missing_both_reports_the_key():
    with pytest.raises(ValueError) as caught:
        values_at(sentinel_wide(), lookup(["Z"], [7]))
    assert str(caught.value) == "keys ['Z'] are not in the table"


def test_values_at_empty_rows_give_an_empty_series():
    result = values_at(sentinel_wide(), lookup([], []))
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_values_at_empty_rows_keep_their_empty_index():
    rows = lookup([], [], index=pd.Index([], dtype="int64"))
    result = values_at(sentinel_wide(), rows)
    assert result.index.equals(rows.index)


def test_values_at_valid_lookups_still_reach_the_last_cell():
    result = values_at(sentinel_wide(), lookup(["B", "A", "B"], [2, 0, 0], index=[4, 8, 2]))
    assert result.tolist() == [SENTINEL, 1.0, 4.0]
    assert result.index.tolist() == [4, 8, 2]


def test_values_at_every_cell_of_the_table():
    wide = sentinel_wide()
    keys = [k for k in wide.index for _ in wide.columns]
    weeks = [w for _ in wide.index for w in wide.columns]
    result = values_at(wide, lookup(keys, weeks))
    assert result.tolist() == wide.to_numpy().ravel().tolist()


def test_values_at_does_not_modify_its_inputs():
    wide = sentinel_wide()
    rows = lookup(["B", "A"], [2, 1])
    wide_before, rows_before = wide.copy(), rows.copy()
    values_at(wide, rows)
    pd.testing.assert_frame_equal(wide, wide_before)
    pd.testing.assert_frame_equal(rows, rows_before)


def test_values_at_a_failed_lookup_does_not_modify_its_inputs():
    wide = sentinel_wide()
    rows = lookup(["Z", "A"], [0, 7])
    wide_before, rows_before = wide.copy(), rows.copy()
    with pytest.raises(ValueError):
        values_at(wide, rows)
    pd.testing.assert_frame_equal(wide, wide_before)
    pd.testing.assert_frame_equal(rows, rows_before)


# --- history features on the synthetic corpus and ledger (SLOW) ---------------------------

SYNTHETIC_WINDOWS = (4, 13, 52)


@pytest.fixture(scope="module")
def synthetic_history(synthetic_pipeline):
    matrix = synthetic_pipeline["off"]       # has the six missing weeks as NaN
    known = synthetic_pipeline["known"]
    last_week = synthetic_pipeline["last_off"]
    return matrix, known, last_week, history_features(matrix, known, last_week,
                                                      SYNTHETIC_WINDOWS)


def test_slow_synthetic_history_row_count(synthetic_history):
    _, known, last_week, result = synthetic_history
    expected = sum(max(0, last_week - max(int(week), 0) + 1)
                   for week in known["known_week"])
    assert len(result) == expected
    assert list(result.columns) == history_columns(SYNTHETIC_WINDOWS)


def test_slow_synthetic_history_counts_are_consistent(synthetic_history):
    result = synthetic_history[3]
    counts = [column for column in result.columns
              if column.startswith(("observed_", "paying_", "customer_age"))]
    assert not result[counts].isna().any().any()
    assert (result["customer_age_weeks"] >= 0).all()
    for window in SYNTHETIC_WINDOWS:
        assert (result[f"paying_weeks_{window}"] <= result[f"observed_weeks_{window}"]).all()
        assert (result[f"observed_weeks_{window}"] <= window).all()


def test_slow_synthetic_history_missing_week_14_lowers_observed_weeks(synthetic_history):
    result = synthetic_history[3]
    # Week 14 is missing; weeks 11 to 13 have data, so 3 of the 4 weeks are observed.
    assert (result.loc[result["origin_week"] == 14, "observed_weeks_4"] == 3).all()


@pytest.mark.parametrize("cut", [0, 16, 32, 48])
def test_slow_synthetic_history_matrix_cut_at_t_gives_the_same_features(synthetic_history,
                                                                       cut):
    matrix, known, _, result = synthetic_history
    truncated = matrix[[column for column in matrix.columns if column <= cut]]
    cut_result = history_features(truncated, known, cut, SYNTHETIC_WINDOWS)
    expected = result[result["origin_week"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(cut_result, expected, check_exact=True)


def test_slow_synthetic_history_matches_the_reference_on_sampled_keys(synthetic_history):
    matrix, known, last_week, result = synthetic_history
    sample = sorted(known.sample(25, random_state=3)["key"])
    sub_known = known[known["key"].isin(sample)][["key", "known_week"]]
    expected = reference_history(matrix.loc[sample], sub_known, last_week, SYNTHETIC_WINDOWS)
    got = result[result["key"].isin(sample)].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected, check_exact=False, rtol=1e-9, atol=1e-6)
