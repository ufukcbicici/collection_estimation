"""Tests for `hurdle/prices.py`: week numbers to TÜFE price levels, the coverage guard, and
deflation.

Week zero is Monday 2025-06-02 throughout, as in the synthetic settings. The synthetic range
is weeks -32 (Monday 2024-10-21) to 64 (Monday 2026-08-24).
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import inflation
from collection_estimation.hurdle.prices import (
    check_price_coverage, deflate, price_weeks, published_levels)

WEEK_ZERO = date(2025, 6, 2)
FIRST, LAST = -32, 64


# --------------------------------------------------------------------------- price_weeks

def test_price_weeks_has_one_row_per_week_inclusive():
    weeks = price_weeks(WEEK_ZERO, -3, 4)
    assert list(weeks["week"]) == list(range(-3, 5))
    assert len(weeks) == 8


def test_price_weeks_columns():
    assert list(price_weeks(WEEK_ZERO, 0, 2).columns) == ["week", "week_start", "week_end"]


def test_price_weeks_week_is_int64():
    assert price_weeks(WEEK_ZERO, -2, 2)["week"].dtype == np.int64


def test_price_weeks_week_zero_starts_on_week_zero():
    row = price_weeks(WEEK_ZERO, 0, 0).iloc[0]
    assert row["week_start"] == pd.Timestamp("2025-06-02")
    assert row["week_end"] == pd.Timestamp("2025-06-08")


def test_price_weeks_start_is_week_zero_plus_seven_days_per_week():
    weeks = price_weeks(WEEK_ZERO, -40, 70)
    expected = [pd.Timestamp(WEEK_ZERO) + pd.Timedelta(days=7 * w) for w in weeks["week"]]
    assert list(weeks["week_start"]) == expected


def test_price_weeks_starts_are_timestamps_mondays_and_ends_sundays():
    weeks = price_weeks(WEEK_ZERO, -40, 70)
    assert pd.api.types.is_datetime64_any_dtype(weeks["week_start"])
    assert pd.api.types.is_datetime64_any_dtype(weeks["week_end"])
    assert (weeks["week_start"].dt.dayofweek == 0).all()
    assert (weeks["week_end"].dt.dayofweek == 6).all()
    assert ((weeks["week_end"] - weeks["week_start"]) == pd.Timedelta(days=6)).all()


def test_price_weeks_negative_week():
    row = price_weeks(WEEK_ZERO, -32, -32).iloc[0]
    assert row["week_start"] == pd.Timestamp("2024-10-21")
    assert row["week_end"] == pd.Timestamp("2024-10-27")


def test_price_weeks_last_week_of_synthetic_range():
    row = price_weeks(WEEK_ZERO, 64, 64).iloc[0]
    assert row["week_start"] == pd.Timestamp("2026-08-24")
    assert row["week_end"] == pd.Timestamp("2026-08-30")


def test_price_weeks_first_equals_last_gives_one_row():
    weeks = price_weeks(WEEK_ZERO, 5, 5)
    assert len(weeks) == 1
    assert weeks["week"].iloc[0] == 5


def test_price_weeks_last_before_first_raises():
    with pytest.raises(ValueError, match="before first_week"):
        price_weeks(WEEK_ZERO, 3, 2)


def test_price_weeks_has_default_index():
    assert list(price_weeks(WEEK_ZERO, -2, 2).index) == list(range(5))


# ------------------------------------------------------------------- check_price_coverage

def test_coverage_passes_for_the_synthetic_range():
    check_price_coverage(price_weeks(WEEK_ZERO, FIRST, LAST))  # no raise


def test_coverage_error_is_a_value_error():
    assert issubclass(inflation.InflationCoverageError, ValueError)


def test_coverage_raises_when_the_range_reaches_a_missing_month_at_the_end():
    # Week 65 starts 2026-08-31 and ends 2026-09-06: September 2026 is not in the table.
    with pytest.raises(inflation.InflationCoverageError, match="2026-09"):
        check_price_coverage(price_weeks(WEEK_ZERO, 0, 65))


def test_coverage_raises_when_the_last_week_is_wholly_in_a_missing_month():
    with pytest.raises(inflation.InflationCoverageError, match="2026-09"):
        check_price_coverage(price_weeks(WEEK_ZERO, 0, 66))


def test_coverage_raises_when_the_first_weeks_published_month_is_missing():
    # Week -39 starts Monday 2024-09-02: its own month 2024-09 IS in the table, but the
    # figure published by then is 2024-08, which is not. inflation.published_price_level
    # would silently fall back to the earliest month here.
    weeks = price_weeks(WEEK_ZERO, -39, 0)
    assert weeks["week_start"].iloc[0] == pd.Timestamp("2024-09-02")
    assert (2024, 9) in inflation.TUFE and (2024, 8) not in inflation.TUFE
    inflation.check_coverage(date(2024, 9, 2), date(2025, 6, 8))  # the copied check passes
    with pytest.raises(inflation.InflationCoverageError, match="2024-08"):
        check_price_coverage(weeks)


def test_coverage_first_published_month_message_names_the_week():
    with pytest.raises(inflation.InflationCoverageError, match="2024-09-02"):
        check_price_coverage(price_weeks(WEEK_ZERO, -39, 0))


def test_coverage_first_week_in_october_2024_passes_with_one_month_lag():
    # Week -32 (2024-10-21) needs 2024-09, which is the table's first month.
    check_price_coverage(price_weeks(WEEK_ZERO, -32, 0), lag_months=1)


def test_coverage_lag_two_months_looks_two_months_back():
    # Week -32 with a two-month lag needs 2024-08: missing.
    with pytest.raises(inflation.InflationCoverageError, match="2024-08"):
        check_price_coverage(price_weeks(WEEK_ZERO, -32, 0), lag_months=2)


def test_coverage_lag_two_months_passes_when_covered():
    # Week -27 starts 2024-11-25; two months back is 2024-09.
    weeks = price_weeks(WEEK_ZERO, -27, 0)
    assert weeks["week_start"].iloc[0] == pd.Timestamp("2024-11-25")
    check_price_coverage(weeks, lag_months=2)


def test_coverage_january_first_week_looks_back_into_december_of_previous_year():
    # Week -21 starts 2025-01-06; one month back is 2024-12, which is in the table.
    weeks = price_weeks(WEEK_ZERO, -21, 0)
    assert weeks["week_start"].iloc[0] == pd.Timestamp("2025-01-06")
    check_price_coverage(weeks, lag_months=1)


def test_coverage_january_rollover_detects_a_missing_previous_year_month():
    weeks = price_weeks(WEEK_ZERO, -21, 0)
    check_price_coverage(weeks, lag_months=4)  # 2024-09: present
    with pytest.raises(inflation.InflationCoverageError, match="2024-08"):
        check_price_coverage(weeks, lag_months=5)  # 2024-08: missing


def test_coverage_twelve_month_lag_from_january_rolls_back_one_year():
    weeks = price_weeks(WEEK_ZERO, -21, 0)  # 2025-01-06
    with pytest.raises(inflation.InflationCoverageError, match="2024-01"):
        check_price_coverage(weeks, lag_months=12)


def test_coverage_thirteen_month_lag_rolls_back_more_than_a_year():
    # Week 31 starts 2026-01-05; 13 months back is 2024-12 (present), 17 back is 2024-08.
    weeks = price_weeks(WEEK_ZERO, 31, 40)
    assert weeks["week_start"].iloc[0] == pd.Timestamp("2026-01-05")
    check_price_coverage(weeks, lag_months=13)
    with pytest.raises(inflation.InflationCoverageError, match="2024-08"):
        check_price_coverage(weeks, lag_months=17)


def test_coverage_lag_zero_checks_only_the_months_in_range():
    check_price_coverage(price_weeks(WEEK_ZERO, -39, 0), lag_months=0)


def test_coverage_default_lag_is_the_publication_lag():
    assert inflation.PUBLICATION_LAG_MONTHS == 1
    with pytest.raises(inflation.InflationCoverageError, match="2024-08"):
        check_price_coverage(price_weeks(WEEK_ZERO, -39, 0))


def test_coverage_does_not_modify_the_weeks():
    weeks = price_weeks(WEEK_ZERO, FIRST, LAST)
    before = weeks.copy()
    check_price_coverage(weeks)
    pd.testing.assert_frame_equal(weeks, before)


# ---------------------------------------------------------------------- published_levels

@pytest.fixture(scope="module")
def synthetic_levels():
    return published_levels(WEEK_ZERO, FIRST, LAST)


def test_published_levels_is_a_named_series_indexed_by_week(synthetic_levels):
    assert isinstance(synthetic_levels, pd.Series)
    assert synthetic_levels.name == "price_level"
    assert list(synthetic_levels.index) == list(range(FIRST, LAST + 1))
    assert pd.api.types.is_integer_dtype(synthetic_levels.index)


def test_published_levels_are_floats(synthetic_levels):
    assert synthetic_levels.dtype == np.float64


def test_published_levels_synthetic_range_has_97_values(synthetic_levels):
    assert len(synthetic_levels) == 97


def test_published_levels_equal_the_copied_published_price_level(synthetic_levels):
    expected = inflation.published_price_level(price_weeks(WEEK_ZERO, FIRST, LAST))
    np.testing.assert_array_equal(synthetic_levels.to_numpy(), expected)


@pytest.mark.parametrize("first,last", [(-32, -32), (-21, 10), (0, 64), (30, 40)])
def test_published_levels_equal_the_copied_function_on_sub_ranges(first, last):
    expected = inflation.published_price_level(price_weeks(WEEK_ZERO, first, last))
    got = published_levels(WEEK_ZERO, first, last)
    np.testing.assert_array_equal(got.to_numpy(), expected)
    assert list(got.index) == list(range(first, last + 1))


def test_published_levels_are_constant_within_a_calendar_month(synthetic_levels):
    weeks = price_weeks(WEEK_ZERO, FIRST, LAST)
    month = weeks["week_start"].dt.to_period("M").to_numpy()
    per_month = pd.Series(synthetic_levels.to_numpy()).groupby(month).nunique()
    assert (per_month == 1).all()


def test_published_levels_step_at_month_boundaries(synthetic_levels):
    # Week -30 starts 2024-11-04 (published: Oct), week -31 starts 2024-10-28 (Sep).
    assert synthetic_levels.loc[-31] == synthetic_levels.loc[-32]
    assert synthetic_levels.loc[-30] > synthetic_levels.loc[-31]


def test_published_levels_follow_tufe_of_the_lagged_month(synthetic_levels):
    weeks = price_weeks(WEEK_ZERO, FIRST, LAST)
    raw_last = inflation._raw_weekly_level(weeks)[-1]
    for week, start in zip(weeks["week"], weeks["week_start"]):
        year, month = start.year, start.month - 1
        if month == 0:
            year, month = year - 1, 12
        assert synthetic_levels.loc[week] == pytest.approx(
            inflation.TUFE[(year, month)] / raw_last, rel=1e-12)


def test_published_levels_last_week_is_relative_to_last_weeks_actual_level(synthetic_levels):
    weeks = price_weeks(WEEK_ZERO, FIRST, LAST)
    raw_last = inflation._raw_weekly_level(weeks)[-1]
    assert synthetic_levels.loc[LAST] == pytest.approx(
        inflation.TUFE[(2026, 7)] / raw_last, rel=1e-12)
    # Same basis as the actual (hindsight) level, which is exactly 1.0 at the last week.
    assert inflation.weekly_price_level(weeks)[-1] == pytest.approx(1.0)


def test_published_levels_are_non_decreasing(synthetic_levels):
    assert synthetic_levels.is_monotonic_increasing


def test_published_levels_last_value_below_one(synthetic_levels):
    # Published lags actual, and prices rose: the last week's published level is below 1.
    assert synthetic_levels.iloc[-1] < 1.0


def test_published_levels_never_exceed_the_actual_level(synthetic_levels):
    actual = inflation.weekly_price_level(price_weeks(WEEK_ZERO, FIRST, LAST))
    assert (synthetic_levels.to_numpy() <= actual + 1e-12).all()


def test_published_levels_pinned_values(synthetic_levels):
    assert synthetic_levels.loc[-32] == pytest.approx(0.589, abs=5e-4)
    assert synthetic_levels.loc[0] == pytest.approx(0.720, abs=5e-4)
    assert synthetic_levels.loc[64] == pytest.approx(0.982, abs=5e-4)


def test_published_levels_raise_for_a_range_past_the_table():
    with pytest.raises(inflation.InflationCoverageError, match="2026-09"):
        published_levels(WEEK_ZERO, 0, 65)


def test_published_levels_raise_instead_of_silent_fallback_at_the_start():
    with pytest.raises(inflation.InflationCoverageError, match="2024-08"):
        published_levels(WEEK_ZERO, -39, 0)


def test_published_levels_raise_for_last_before_first():
    with pytest.raises(ValueError):
        published_levels(WEEK_ZERO, 2, 1)


def test_published_levels_are_deterministic():
    a = published_levels(WEEK_ZERO, FIRST, LAST)
    b = published_levels(WEEK_ZERO, FIRST, LAST)
    pd.testing.assert_series_equal(a, b)


# -------------------------------------------------------------------------------- deflate

@pytest.fixture
def levels():
    return pd.Series([0.5, 0.8, 1.0], index=[1, 2, 3], name="price_level")


def test_deflate_divides_each_amount_by_its_own_weeks_level(levels):
    amounts = pd.Series([100.0, 80.0, 30.0, 50.0])
    weeks = pd.Series([1, 2, 3, 1])
    result = deflate(amounts, weeks, levels)
    np.testing.assert_allclose(result.to_numpy(), [200.0, 100.0, 30.0, 100.0])


def test_deflate_keeps_the_amounts_index(levels):
    index = pd.Index(["a", "b", "c"], name="row")
    amounts = pd.Series([10.0, 20.0, 30.0], index=index)
    weeks = pd.Series([3, 1, 2], index=index)
    result = deflate(amounts, weeks, levels)
    assert list(result.index) == ["a", "b", "c"]
    assert result.index.name == "row"
    np.testing.assert_allclose(result.to_numpy(), [10.0, 40.0, 37.5])


def test_deflate_with_a_non_monotonic_integer_index(levels):
    index = pd.Index([50, 7, 12])
    result = deflate(pd.Series([1.0, 2.0, 3.0], index=index),
                     pd.Series([1, 2, 3], index=index), levels)
    assert list(result.index) == [50, 7, 12]
    np.testing.assert_allclose(result.to_numpy(), [2.0, 2.5, 3.0])


def test_deflate_at_level_one_leaves_amounts_unchanged(levels):
    amounts = pd.Series([123.45, -6.0])
    result = deflate(amounts, pd.Series([3, 3]), levels)
    np.testing.assert_allclose(result.to_numpy(), amounts.to_numpy())


def test_deflate_handles_negative_amounts(levels):
    result = deflate(pd.Series([-40.0]), pd.Series([2]), levels)
    assert result.iloc[0] == pytest.approx(-50.0)


def test_deflate_empty_input(levels):
    result = deflate(pd.Series([], dtype=float), pd.Series([], dtype="int64"), levels)
    assert len(result) == 0


def test_deflate_week_without_a_level_raises_naming_it(levels):
    with pytest.raises(ValueError, match=r"\[9\]"):
        deflate(pd.Series([1.0, 2.0]), pd.Series([1, 9]), levels)


def test_deflate_missing_weeks_are_sorted_unique_and_at_most_ten(levels):
    missing = [30, 12, 25, 11, 14, 13, 20, 19, 18, 17, 16, 15, 12]
    weeks = pd.Series([1] + missing)
    amounts = pd.Series([1.0] * len(weeks))
    with pytest.raises(ValueError) as error:
        deflate(amounts, weeks, levels)
    message = str(error.value)
    assert "[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]" in message
    assert "25" not in message and "30" not in message


def test_deflate_level_that_is_nan_raises(levels):
    bad = levels.copy()
    bad.loc[2] = np.nan
    with pytest.raises(ValueError, match=r"\[2\]"):
        deflate(pd.Series([1.0]), pd.Series([2]), bad)


def test_deflate_does_not_modify_its_inputs(levels):
    amounts = pd.Series([100.0, 80.0], index=["x", "y"])
    weeks = pd.Series([1, 2], index=["x", "y"])
    a0, w0, l0 = amounts.copy(), weeks.copy(), levels.copy()
    deflate(amounts, weeks, levels)
    pd.testing.assert_series_equal(amounts, a0)
    pd.testing.assert_series_equal(weeks, w0)
    pd.testing.assert_series_equal(levels, l0)


def test_deflate_with_published_levels_reads_as_last_weeks_try(synthetic_levels):
    weeks = pd.Series([FIRST, 0, LAST])
    amounts = pd.Series([1000.0, 1000.0, 1000.0])
    result = deflate(amounts, weeks, synthetic_levels)
    np.testing.assert_allclose(
        result.to_numpy(), 1000.0 / synthetic_levels.loc[[FIRST, 0, LAST]].to_numpy())
    # An older TRY is worth more of the last week's TRY.
    assert result.iloc[0] > result.iloc[1] > result.iloc[2] > 1000.0


def test_deflate_week_outside_published_range_raises(synthetic_levels):
    with pytest.raises(ValueError, match=r"\[65\]"):
        deflate(pd.Series([1.0]), pd.Series([65]), synthetic_levels)


# ------------------------------------------------------------ deflate: the index guard

INDEX_MESSAGE = "amounts and weeks must have the same index (e.g. two columns of one table)"


def _raises_index_message(amounts, weeks, levels):
    with pytest.raises(ValueError) as error:
        deflate(amounts, weeks, levels)
    assert str(error.value) == INDEX_MESSAGE


def test_deflate_different_index_labels_raise_with_the_exact_message(levels):
    # The old silent case: pandas divides by label, so [5, 6] / [0, 1] was all NaN.
    amounts = pd.Series([100.0, 80.0], index=[5, 6])
    weeks = pd.Series([1, 2])
    assert (amounts / weeks.map(levels)).isna().all()  # what the guard prevents
    _raises_index_message(amounts, weeks, levels)


def test_deflate_same_labels_in_a_different_order_raise(levels):
    # Index.equals is order-sensitive; pinned so a switch to a set comparison is noticed.
    _raises_index_message(pd.Series([100.0, 80.0], index=[0, 1]),
                          pd.Series([1, 2], index=[1, 0]), levels)


def test_deflate_different_lengths_raise(levels):
    _raises_index_message(pd.Series([100.0, 80.0, 30.0]), pd.Series([1, 2]), levels)


def test_deflate_weeks_longer_than_amounts_raise(levels):
    _raises_index_message(pd.Series([100.0]), pd.Series([1, 2]), levels)


def test_deflate_two_columns_of_one_table_with_arbitrary_int_labels(levels):
    table = pd.DataFrame({"amount": [100.0, 80.0, 30.0], "week": [1, 2, 3]},
                         index=[10, 3, 7])
    result = deflate(table["amount"], table["week"], levels)
    assert list(result.index) == [10, 3, 7]
    np.testing.assert_allclose(result.to_numpy(), [200.0, 100.0, 30.0])


def test_deflate_two_columns_of_one_table_with_a_string_index(levels):
    table = pd.DataFrame({"amount": [40.0, 30.0], "week": [2, 3]},
                         index=pd.Index(["inv-b", "inv-a"], name="invoice"))
    result = deflate(table["amount"], table["week"], levels)
    assert list(result.index) == ["inv-b", "inv-a"]
    assert result.index.name == "invoice"
    np.testing.assert_allclose(result.to_numpy(), [50.0, 30.0])


def test_deflate_two_columns_of_a_filtered_table_keep_working(levels):
    # A filtered table keeps gaps in its index; both columns share them.
    table = pd.DataFrame({"amount": [100.0, 80.0, 30.0, 50.0], "week": [1, 2, 3, 1]})
    part = table[table["amount"] > 40.0]
    result = deflate(part["amount"], part["week"], levels)
    assert list(result.index) == [0, 1, 3]
    np.testing.assert_allclose(result.to_numpy(), [200.0, 100.0, 100.0])


def test_deflate_empty_columns_of_one_table_work(levels):
    table = pd.DataFrame({"amount": pd.Series([], dtype=float),
                          "week": pd.Series([], dtype="int64")})
    result = deflate(table["amount"], table["week"], levels)
    assert len(result) == 0


def test_deflate_empty_with_equal_explicit_empty_indexes_works(levels):
    index = pd.Index([], dtype="int64")
    result = deflate(pd.Series([], dtype=float, index=index),
                     pd.Series([], dtype="int64", index=index), levels)
    assert len(result) == 0


def test_deflate_index_check_comes_before_the_missing_level_check(levels):
    # Mismatched index AND week 9 with no level: the index message wins.
    _raises_index_message(pd.Series([1.0, 2.0], index=[5, 6]),
                          pd.Series([1, 9]), levels)


def test_deflate_matching_index_with_a_missing_week_still_reports_the_week(levels):
    with pytest.raises(ValueError) as error:
        deflate(pd.Series([1.0, 2.0], index=[5, 6]),
                pd.Series([1, 9], index=[5, 6]), levels)
    assert "[9]" in str(error.value)
    assert "same index" not in str(error.value)
