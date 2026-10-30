"""Tests for `hurdle/prices.py`: week numbers to TÜFE price levels, the coverage guard, and
deflation.

Week zero is Monday 2025-06-02 throughout, as in the synthetic settings. The synthetic range
is weeks -32 (Monday 2024-10-21) to 64 (Monday 2026-08-24). Published levels are normalised
to 1.0 at week 0 (not at the last week, as the copied module does), so they must not depend
on where the range stops.
"""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import inflation
from collection_estimation.hurdle.prices import (
    _month_number, check_price_coverage, deflate, price_weeks, projected_levels,
    published_levels, published_month)

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
#
# published_levels is the copied module's published level (which that module normalises to
# the ACTUAL level of the range's last week) divided by its value at week 0, so it reads 1.0
# at week 0. Week 0 (Mon 2025-06-02) publishes May 2025's TÜFE.

GUARD_MESSAGE = ("weeks {first} to {last} do not include week 0, the week the price levels "
                 "are normalised to")
WEEK_ZERO_PUBLISHED_MONTH = (2025, 5)


def copied_levels(first, last):
    """The copied module's published level as a Series by week: the OLD normalisation."""
    weeks = price_weeks(WEEK_ZERO, first, last)
    return pd.Series(inflation.published_price_level(weeks), index=weeks["week"].to_numpy())


def lagged_month(start):
    year, month = start.year, start.month - 1
    if month == 0:
        year, month = year - 1, 12
    return year, month


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


def test_published_levels_are_exactly_one_at_week_zero(synthetic_levels):
    assert synthetic_levels.loc[0] == 1.0


@pytest.mark.parametrize("first,last", [(-32, 0), (0, 0), (0, 64), (-21, 10), (-1, 1),
                                        (-5, 30)])
def test_published_levels_are_exactly_one_at_week_zero_on_any_range(first, last):
    assert published_levels(WEEK_ZERO, first, last).loc[0] == 1.0


def test_published_levels_equal_the_copied_level_divided_by_its_week_zero_value(
        synthetic_levels):
    old = copied_levels(FIRST, LAST)
    np.testing.assert_allclose(synthetic_levels.to_numpy(), (old / old.loc[0]).to_numpy(),
                               rtol=1e-15, atol=0)


@pytest.mark.parametrize("first,last", [(-32, 0), (-21, 10), (0, 64), (0, 0), (-5, 30)])
def test_published_levels_equal_the_rescaled_copied_function_on_sub_ranges(first, last):
    old = copied_levels(first, last)
    got = published_levels(WEEK_ZERO, first, last)
    np.testing.assert_allclose(got.to_numpy(), (old / old.loc[0]).to_numpy(),
                               rtol=1e-15, atol=0)
    assert list(got.index) == list(range(first, last + 1))


def test_published_levels_keep_the_copied_ratios_between_weeks(synthetic_levels):
    # Rescaling by one constant leaves every ratio between two weeks unchanged.
    old = copied_levels(FIRST, LAST)
    for a, b in [(-32, 0), (-32, 64), (0, 64), (-1, 5), (20, 40)]:
        assert synthetic_levels.loc[a] / synthetic_levels.loc[b] == pytest.approx(
            old.loc[a] / old.loc[b], rel=1e-14)


def test_published_levels_differ_from_the_copied_levels_by_one_constant(synthetic_levels):
    old = copied_levels(FIRST, LAST)
    factor = (old / synthetic_levels).to_numpy()
    np.testing.assert_allclose(factor, old.loc[0], rtol=1e-14)
    # And the constant is not 1: the copied module's level at week 0 is about 0.72.
    assert old.loc[0] == pytest.approx(0.720, abs=5e-4)


def test_published_levels_are_constant_within_a_calendar_month(synthetic_levels):
    weeks = price_weeks(WEEK_ZERO, FIRST, LAST)
    month = weeks["week_start"].dt.to_period("M").to_numpy()
    per_month = pd.Series(synthetic_levels.to_numpy()).groupby(month).nunique()
    assert (per_month == 1).all()


def test_published_levels_step_at_month_boundaries(synthetic_levels):
    # Week -30 starts 2024-11-04 (published: Oct), week -31 starts 2024-10-28 (Sep).
    assert synthetic_levels.loc[-31] == synthetic_levels.loc[-32]
    assert synthetic_levels.loc[-30] > synthetic_levels.loc[-31]


def test_published_levels_follow_tufe_of_the_lagged_month_relative_to_week_zero(
        synthetic_levels):
    weeks = price_weeks(WEEK_ZERO, FIRST, LAST)
    base = inflation.TUFE[WEEK_ZERO_PUBLISHED_MONTH]
    for week, start in zip(weeks["week"], weeks["week_start"]):
        assert synthetic_levels.loc[week] == pytest.approx(
            inflation.TUFE[lagged_month(start)] / base, rel=1e-12)


def test_published_levels_at_the_last_week_are_tufe_july_2026_over_may_2025(synthetic_levels):
    assert synthetic_levels.loc[LAST] == pytest.approx(
        inflation.TUFE[(2026, 7)] / inflation.TUFE[(2025, 5)], rel=1e-12)


def test_published_levels_are_non_decreasing(synthetic_levels):
    assert synthetic_levels.is_monotonic_increasing


def test_published_levels_before_week_zero_are_below_one(synthetic_levels):
    # Prices rose: every week before 0 published an earlier, lower TÜFE than week 0 did.
    assert (synthetic_levels.loc[:-1] < 1.0).all()


def test_published_levels_in_week_zeros_publication_month_are_one(synthetic_levels):
    # Weeks 1-4 start in June 2025 (week 4 on Mon 30 June), so they publish May, like week 0.
    assert (synthetic_levels.loc[0:4] == 1.0).all()


def test_published_levels_after_the_first_monthly_step_are_above_one(synthetic_levels):
    # Week 5 starts Mon 7 July 2025 and publishes June: the first step above week 0.
    assert (synthetic_levels.loc[5:] > 1.0).all()


def test_published_levels_last_value_above_one(synthetic_levels):
    # Under the old normalisation (1.0 = the last week's ACTUAL level) this was ~0.98.
    assert synthetic_levels.iloc[-1] > 1.0


def test_published_levels_never_exceed_the_actual_level_on_a_common_basis(synthetic_levels):
    # Published lags actual. Compare both on the copied module's basis (1.0 = the actual
    # level of the last week): published_levels times the copied week-0 value.
    weeks = price_weeks(WEEK_ZERO, FIRST, LAST)
    actual = inflation.weekly_price_level(weeks)
    old_week_zero = copied_levels(FIRST, LAST).loc[0]
    assert (synthetic_levels.to_numpy() * old_week_zero <= actual + 1e-12).all()


def test_published_levels_pinned_values(synthetic_levels):
    assert synthetic_levels.loc[-32] == pytest.approx(0.8176, abs=5e-4)
    assert synthetic_levels.loc[-1] == pytest.approx(0.9849, abs=5e-4)
    assert synthetic_levels.loc[0] == 1.0
    assert synthetic_levels.loc[5] == pytest.approx(1.0137, abs=5e-4)
    assert synthetic_levels.loc[64] == pytest.approx(1.3631, abs=5e-4)


# --- no future number enters: the levels do not depend on where the range stops ----------

@pytest.mark.parametrize("longer_last", [11, 30, 64])
def test_published_levels_do_not_depend_on_last_week(longer_last):
    short = published_levels(WEEK_ZERO, -5, 10)
    long = published_levels(WEEK_ZERO, -5, longer_last).loc[-5:10]
    np.testing.assert_allclose(short.to_numpy(), long.to_numpy(), rtol=1e-14, atol=0)
    assert list(long.index) == list(short.index)


@pytest.mark.parametrize("first", [-32, -20, -5, 0])
@pytest.mark.parametrize("last", [0, 5, 10, 30, 64])
def test_published_levels_agree_on_common_weeks_whatever_the_range(synthetic_levels, first,
                                                                    last):
    got = published_levels(WEEK_ZERO, first, last)
    np.testing.assert_allclose(got.to_numpy(), synthetic_levels.loc[first:last].to_numpy(),
                               rtol=1e-14, atol=0)


def test_published_levels_up_to_week_zero_use_no_later_tufe():
    # A range that stops at week 0 uses only months published by week 0 (up to May 2025),
    # and still agrees with the full range.
    levels = published_levels(WEEK_ZERO, -32, 0)
    full = published_levels(WEEK_ZERO, -32, 64).loc[-32:0]
    np.testing.assert_allclose(levels.to_numpy(), full.to_numpy(), rtol=1e-14, atol=0)


def test_published_levels_ignore_a_changed_future_tufe_month(monkeypatch):
    # Change July 2026 (published only in week 63 and 64): weeks -5..10 must not move.
    before = published_levels(WEEK_ZERO, -5, 64).loc[-5:10]
    monkeypatch.setitem(inflation.TUFE, (2026, 7), inflation.TUFE[(2026, 7)] * 1.5)
    after = published_levels(WEEK_ZERO, -5, 64)
    np.testing.assert_allclose(after.loc[-5:10].to_numpy(), before.to_numpy(),
                               rtol=1e-14, atol=0)
    assert after.loc[64] != pytest.approx(before.iloc[-1])  # the change did reach week 64


# --- the week-0 guard --------------------------------------------------------------------

@pytest.mark.parametrize("first,last", [(1, 10), (30, 40), (64, 64), (-32, -1), (-10, -5),
                                        (-32, -32)])
def test_published_levels_without_week_zero_raise_the_exact_message(first, last):
    with pytest.raises(ValueError) as error:
        published_levels(WEEK_ZERO, first, last)
    assert str(error.value) == GUARD_MESSAGE.format(first=first, last=last)
    assert not isinstance(error.value, inflation.InflationCoverageError)


@pytest.mark.parametrize("first,last", [(-39, -1), (-45, -40), (65, 70), (100, 120)])
def test_published_levels_guard_comes_before_the_coverage_check(first, last):
    # These ranges also fail the coverage check; the week-0 guard must win.
    with pytest.raises(ValueError) as error:
        published_levels(WEEK_ZERO, first, last)
    assert str(error.value) == GUARD_MESSAGE.format(first=first, last=last)
    assert not isinstance(error.value, inflation.InflationCoverageError)


def test_published_levels_guard_ranges_really_fail_coverage_on_their_own():
    # Pins the premise of the test above.
    for first, last in [(-39, -1), (65, 70)]:
        with pytest.raises(inflation.InflationCoverageError):
            check_price_coverage(price_weeks(WEEK_ZERO, first, last))


@pytest.mark.parametrize("first,last", [(0, 0), (0, 10), (-10, 0), (-32, 0), (0, 64)])
def test_published_levels_range_starting_or_ending_at_week_zero_works(first, last):
    levels = published_levels(WEEK_ZERO, first, last)
    assert list(levels.index) == list(range(first, last + 1))
    assert levels.loc[0] == 1.0


def test_published_levels_single_week_zero():
    levels = published_levels(WEEK_ZERO, 0, 0)
    assert levels.tolist() == [1.0]
    assert levels.name == "price_level"


def test_published_levels_raise_for_a_range_past_the_table():
    with pytest.raises(inflation.InflationCoverageError, match="2026-09"):
        published_levels(WEEK_ZERO, 0, 65)


def test_published_levels_raise_instead_of_silent_fallback_at_the_start():
    with pytest.raises(inflation.InflationCoverageError, match="2024-08"):
        published_levels(WEEK_ZERO, -39, 0)


def test_published_levels_raise_for_last_before_first():
    # With last < first no range can hold week 0, so the guard answers first.
    with pytest.raises(ValueError) as error:
        published_levels(WEEK_ZERO, 2, 1)
    assert str(error.value) == GUARD_MESSAGE.format(first=2, last=1)


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


def test_deflate_with_published_levels_reads_as_week_zeros_try(synthetic_levels):
    weeks = pd.Series([FIRST, 0, LAST])
    amounts = pd.Series([1000.0, 1000.0, 1000.0])
    result = deflate(amounts, weeks, synthetic_levels)
    np.testing.assert_allclose(
        result.to_numpy(), 1000.0 / synthetic_levels.loc[[FIRST, 0, LAST]].to_numpy())
    # Week 0's TRY is the unit: an older TRY is worth more of it, a later TRY less.
    assert result.iloc[1] == 1000.0
    assert result.iloc[0] > 1000.0 > result.iloc[2]


def test_deflate_with_published_levels_changes_by_one_constant_versus_the_old_basis(
        synthetic_levels):
    # Deflating by the old (1.0 = the last week's actual level) and the new (1.0 = week 0)
    # levels gives amounts that differ by one constant factor, the old level at week 0.
    old = copied_levels(FIRST, LAST).rename("price_level")
    weeks = pd.Series(list(range(FIRST, LAST + 1, 3)) + [0, 0, LAST])
    amounts = pd.Series(np.linspace(-500.0, 9000.0, len(weeks)))
    new_result = deflate(amounts, weeks, synthetic_levels)
    old_result = deflate(amounts, weeks, old)
    np.testing.assert_allclose(new_result.to_numpy(), old_result.to_numpy() * old.loc[0],
                               rtol=1e-13)


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


# ======================================================================== U13.1 additions
#
# published_month, _month_number and projected_levels: the price level a forecast made at
# the end of week `origin` re-inflates each target week with, as known at the origin.

# ---------------------------------------------------------------------- published_month

@pytest.mark.parametrize("day", [date(2026, 7, 1), date(2026, 7, 6), date(2026, 7, 27),
                                 date(2026, 7, 31)])
def test_published_month_a_week_in_july_has_june(day):
    assert published_month(day) == (2026, 6)


@pytest.mark.parametrize("day", [date(2026, 1, 1), date(2026, 1, 5), date(2026, 1, 26)])
def test_published_month_january_has_december_of_the_previous_year(day):
    assert published_month(day) == (2025, 12)


def test_published_month_february_has_january_of_the_same_year():
    assert published_month(date(2026, 2, 2)) == (2026, 1)


def test_published_month_december_has_november():
    assert published_month(date(2025, 12, 29)) == (2025, 11)


def test_published_month_default_lag_is_the_publication_lag():
    day = date(2025, 6, 2)
    assert published_month(day) == published_month(day, inflation.PUBLICATION_LAG_MONTHS)


def test_published_month_lag_zero_is_the_weeks_own_month():
    assert published_month(date(2026, 1, 5), 0) == (2026, 1)


@pytest.mark.parametrize("lag,expected", [(2, (2025, 11)), (12, (2025, 1)), (13, (2024, 12)),
                                          (24, (2024, 1)), (25, (2023, 12))])
def test_published_month_long_lags_roll_back_whole_years(lag, expected):
    assert published_month(date(2026, 1, 5), lag) == expected


def test_published_month_returns_a_tuple_of_ints():
    result = published_month(date(2026, 3, 9))
    assert isinstance(result, tuple) and len(result) == 2
    assert all(type(part) is int for part in result)


def test_published_month_month_is_always_between_one_and_twelve():
    day = date(2023, 1, 2)
    for week in range(0, 200):
        for lag in range(0, 15):
            year, month = published_month(day + timedelta(days=7 * week), lag)
            assert 1 <= month <= 12


def test_published_month_agrees_with_the_copied_modules_published_key():
    # Same rule, different code: inflation._published_key steps back month by month.
    day = date(2023, 1, 2)
    for week in range(0, 200, 3):
        start = day + timedelta(days=7 * week)
        for lag in range(0, 26):
            assert published_month(start, lag) == inflation._published_key(
                pd.Timestamp(start), lag)


def test_published_month_accepts_a_timestamp():
    # projected_levels passes dates; a Timestamp has the same year/month attributes.
    assert published_month(pd.Timestamp("2026-07-06")) == (2026, 6)


# ------------------------------------------------------------------------ _month_number

def test_month_number_consecutive_months_differ_by_one():
    assert _month_number((2024, 2)) - _month_number((2024, 1)) == 1


def test_month_number_across_a_year_boundary():
    assert _month_number((2024, 1)) - _month_number((2023, 12)) == 1


def test_month_number_a_year_is_twelve():
    assert _month_number((2026, 7)) - _month_number((2025, 7)) == 12


def test_month_number_is_strictly_increasing_in_calendar_order():
    months = [(y, m) for y in range(2020, 2028) for m in range(1, 13)]
    numbers = [_month_number(month) for month in months]
    assert numbers == list(range(numbers[0], numbers[0] + len(months)))


def test_month_number_matches_published_month_lag():
    # published_month(d, lag) lies exactly `lag` months before the month of d.
    start = date(2026, 3, 2)
    for lag in range(0, 30):
        assert (_month_number((start.year, start.month))
                - _month_number(published_month(start, lag))) == lag


# ----------------------------------------------------------------------- projected_levels
#
# Most arithmetic tests use a hand-made TÜFE table (monkeypatched) and week zero Monday
# 2024-01-01, which publishes December 2023.

HAND_ZERO = date(2024, 1, 1)


def week_in(year, month, week_zero=HAND_ZERO):
    """The first week number whose Monday falls in the given month."""
    return math.ceil((date(year, month, 1) - week_zero).days / 7)


def months_from(first, count):
    """`count` consecutive (year, month) keys starting at `first`."""
    year, month = first
    keys = []
    for _ in range(count):
        keys.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return keys


def constant_growth_table(rate=0.02, first=(2023, 1), count=24, start=100.0):
    """A table growing at exactly `rate` a month: month i (from `first`) is start*(1+rate)**i."""
    return {key: start * (1 + rate) ** i for i, key in enumerate(months_from(first, count))}


def ratio(month_a, month_b, rate=0.02):
    return (1 + rate) ** (_month_number(month_a) - _month_number(month_b))


@pytest.fixture
def constant_table(monkeypatch):
    table = constant_growth_table()
    monkeypatch.setattr(inflation, "TUFE", table)
    return table


def test_week_in_helper_premises():
    assert HAND_ZERO.weekday() == 0
    assert published_month(HAND_ZERO) == (2023, 12)
    for year, month in [(2024, 1), (2024, 4), (2024, 7), (2025, 6), (2023, 12)]:
        start = HAND_ZERO + timedelta(days=7 * week_in(year, month))
        assert (start.year, start.month) == (year, month)
        assert (start - timedelta(days=7)).month != month


# --- shape --------------------------------------------------------------------------------

def test_projected_levels_is_a_named_float_series_indexed_by_the_targets():
    result = projected_levels(WEEK_ZERO, 40, [38, 40, 45, 50])
    assert isinstance(result, pd.Series)
    assert result.name == "price_level"
    assert list(result.index) == [38, 40, 45, 50]
    assert result.dtype == np.float64


def test_projected_levels_keep_target_order_and_duplicates():
    targets = [10, 3, 10, 70, 3, 45]
    result = projected_levels(WEEK_ZERO, 40, targets)
    assert list(result.index) == targets
    assert result.iloc[0] == result.iloc[2]
    assert result.iloc[1] == result.iloc[4]


def test_projected_levels_values_do_not_depend_on_the_other_targets():
    alone = {t: projected_levels(WEEK_ZERO, 40, [t]).iloc[0] for t in [3, 40, 45, 70]}
    together = projected_levels(WEEK_ZERO, 40, [70, 45, 40, 3])
    for t, value in alone.items():
        assert together.loc[t] == value


def test_projected_levels_accept_a_range_of_targets():
    result = projected_levels(WEEK_ZERO, 40, range(38, 43))
    assert list(result.index) == [38, 39, 40, 41, 42]


def test_projected_levels_do_not_modify_the_targets_or_the_table():
    targets = [5, 40, 70]
    before_targets = list(targets)
    before_table = dict(inflation.TUFE)
    projected_levels(WEEK_ZERO, 40, targets)
    assert targets == before_targets
    assert inflation.TUFE == before_table


def test_projected_levels_are_deterministic():
    a = projected_levels(WEEK_ZERO, 40, list(range(-32, 90)))
    b = projected_levels(WEEK_ZERO, 40, list(range(-32, 90)))
    pd.testing.assert_series_equal(a, b)


# --- week 0 and the normalisation ----------------------------------------------------------

@pytest.mark.parametrize("origin", [0, 1, 5, 13, 40, 61, 64, 69])
def test_projected_levels_are_exactly_one_at_week_zero_for_a_non_negative_origin(origin):
    assert projected_levels(WEEK_ZERO, origin, [0]).iloc[0] == 1.0


def test_projected_levels_week_zero_is_not_one_when_the_origin_publishes_an_earlier_month():
    # Origin -10 (2025-03-24) knows only February 2025: week 0's own figure (May 2025) is
    # projected, but the result is still divided by the ACTUAL May figure. Pinned behaviour.
    assert published_month(WEEK_ZERO + timedelta(days=-70)) == (2025, 2)
    value = projected_levels(WEEK_ZERO, -10, [0]).iloc[0]
    assert value != 1.0
    # The table starts September 2024, so only 6 months (a 5-month span) are known.
    history = sorted(k for k in inflation.TUFE if k <= (2025, 2))[-7:]
    assert history[0] == (2024, 9) and len(history) == 6
    growth = (math.log(inflation.TUFE[history[-1]]) - math.log(inflation.TUFE[history[0]])) / 5
    expected = inflation.TUFE[(2025, 2)] * math.exp(3 * growth) / inflation.TUFE[(2025, 5)]
    assert value == pytest.approx(expected, rel=1e-14)


# --- published by the origin: the table figure, equal to published_levels -----------------
#
# Equal to within one or two ulps, not bit for bit: published_levels divides
# (TÜFE / c) by (TÜFE_base / c), projected_levels divides TÜFE by TÜFE_base directly.

@pytest.fixture(scope="module")
def synthetic_published():
    return published_levels(WEEK_ZERO, FIRST, LAST)


@pytest.mark.parametrize("origin", [0, 4, 5, 13, 30, 40, 61, 64])
def test_projected_levels_up_to_the_origin_equal_published_levels(synthetic_published,
                                                                   origin):
    targets = list(range(FIRST, origin + 1))
    got = projected_levels(WEEK_ZERO, origin, targets)
    np.testing.assert_allclose(got.to_numpy(), synthetic_published.loc[FIRST:origin].to_numpy(),
                               rtol=1e-15, atol=0)


def test_projected_levels_up_to_the_origin_equal_published_levels_for_every_origin(
        synthetic_published):
    for origin in range(FIRST, LAST + 1):
        targets = list(range(FIRST, origin + 1))
        got = projected_levels(WEEK_ZERO, origin, targets)
        np.testing.assert_allclose(got.to_numpy(),
                                   synthetic_published.loc[FIRST:origin].to_numpy(),
                                   rtol=1e-15, atol=0)


@pytest.mark.parametrize("origin", [0, 13, 40, 61])
def test_projected_levels_up_to_the_origin_are_tufe_ratios_exactly(origin):
    # TÜFE[needed] / TÜFE[week 0's month], one division: exact.
    targets = list(range(FIRST, origin + 1))
    got = projected_levels(WEEK_ZERO, origin, targets)
    base = inflation.TUFE[WEEK_ZERO_PUBLISHED_MONTH]
    for week in targets:
        needed = published_month(WEEK_ZERO + timedelta(days=7 * week))
        assert got.loc[week] == inflation.TUFE[needed] / base


def test_projected_levels_after_the_origin_in_the_same_published_month_are_exact(
        synthetic_published):
    # Origin 40 starts 2026-03-09 (publishes February); weeks 41-43 start in March too.
    assert published_month(WEEK_ZERO + timedelta(days=7 * 43)) == (2026, 2)
    got = projected_levels(WEEK_ZERO, 40, [41, 42, 43])
    np.testing.assert_allclose(got.to_numpy(), synthetic_published.loc[41:43].to_numpy(),
                               rtol=1e-15, atol=0)


def test_projected_levels_origins_in_one_published_month_give_the_same_projection():
    # Weeks 40-43 all publish February 2026: whatever the origin among them, the same.
    targets = list(range(30, 90))
    first = projected_levels(WEEK_ZERO, 40, targets)
    for origin in [41, 42, 43]:
        pd.testing.assert_series_equal(projected_levels(WEEK_ZERO, origin, targets), first)


def test_projected_levels_a_later_published_month_changes_the_projection():
    targets = [80]
    assert (projected_levels(WEEK_ZERO, 43, targets).iloc[0]
            != projected_levels(WEEK_ZERO, 44, targets).iloc[0])


# --- not yet published: projected from the last figure ------------------------------------

def test_projected_levels_real_table_projection_arithmetic():
    # Origin 40 (2026-03-09) knows February 2026. Target 52 starts 2026-06-01: needs May,
    # three months past February. Rate: the last 7 months up to February (Aug..Feb), 6 steps.
    origin, target = 40, 52
    assert published_month(WEEK_ZERO + timedelta(days=7 * target)) == (2026, 5)
    tufe = inflation.TUFE
    growth = (math.log(tufe[(2026, 2)]) - math.log(tufe[(2025, 8)])) / 6
    expected = tufe[(2026, 2)] * math.exp(3 * growth) / tufe[(2025, 5)]
    assert projected_levels(WEEK_ZERO, origin, [target]).iloc[0] == pytest.approx(
        expected, rel=1e-14)


def test_projected_levels_do_not_return_a_month_published_after_the_origin():
    # Week 66 (2026-09-07) needs August 2026, which IS in the table, but origin 40 knows only
    # February: the result must be a projection, not the table's August figure.
    got = projected_levels(WEEK_ZERO, 40, [66]).iloc[0]
    actual = inflation.TUFE[(2026, 8)] / inflation.TUFE[(2025, 5)]
    assert got != pytest.approx(actual, rel=1e-6)


def test_projected_levels_work_past_the_end_of_the_table():
    # Origin 64 (2026-08-24) knows July 2026. Week 65 (2026-08-31) still publishes July: exact.
    # Weeks 66 (Sep 7: needs Aug), 70 (Oct 5: needs Sep) and 120 lie past what is known; the
    # last two past the table's end (August 2026).
    tufe = inflation.TUFE
    got = projected_levels(WEEK_ZERO, 64, [64, 65, 66, 70, 120])
    base = tufe[(2025, 5)]
    assert got.loc[64] == tufe[(2026, 7)] / base
    assert got.loc[65] == tufe[(2026, 7)] / base
    growth = (math.log(tufe[(2026, 7)]) - math.log(tufe[(2026, 1)])) / 6
    for week in [66, 70, 120]:
        needed = published_month(WEEK_ZERO + timedelta(days=7 * week))
        months = _month_number(needed) - _month_number((2026, 7))
        assert got.loc[week] == pytest.approx(
            tufe[(2026, 7)] * math.exp(growth * months) / base, rel=1e-14)
    assert np.isfinite(got.to_numpy()).all()


def test_projected_levels_last_covered_origin_and_far_targets():
    # Origin 69 (2026-09-28) knows August 2026, the table's last month.
    got = projected_levels(WEEK_ZERO, 69, [69, 80, 200])
    assert got.loc[69] == inflation.TUFE[(2026, 8)] / inflation.TUFE[(2025, 5)]
    assert got.loc[69] < got.loc[80] < got.loc[200]


@pytest.mark.parametrize("origin", [0, 13, 40, 61, 64])
def test_projected_levels_are_non_decreasing_in_the_target_on_the_real_table(origin):
    # The real table only rises, so both the figures and every projection rate are positive.
    targets = list(range(FIRST, 130))
    assert projected_levels(WEEK_ZERO, origin, targets).is_monotonic_increasing


@pytest.mark.parametrize("origin", [0, 13, 40])
def test_projected_levels_are_a_monthly_step(origin):
    # Targets publishing the same month get the same value, projected or not.
    targets = list(range(FIRST, 130))
    got = projected_levels(WEEK_ZERO, origin, targets)
    months = [str(published_month(WEEK_ZERO + timedelta(days=7 * w))) for w in targets]
    per_month = pd.Series(got.to_numpy()).groupby(months).nunique()
    assert (per_month == 1).all()


# --- constant growth: the projection equals the table extended at that rate ---------------

def test_constant_growth_projection_is_exactly_the_rate_compounded(constant_table):
    # Origin in April 2024 knows March 2024; target in July 2024 needs June: 3 months ahead.
    origin, target = week_in(2024, 4), week_in(2024, 7)
    got = projected_levels(HAND_ZERO, origin, [target]).iloc[0]
    assert got == pytest.approx(ratio((2024, 6), (2023, 12)), rel=1e-12)
    assert got == pytest.approx(constant_table[(2024, 6)] / constant_table[(2023, 12)],
                                rel=1e-12)


@pytest.mark.parametrize("months_ahead", [1, 2, 5, 12, 30])
def test_constant_growth_projection_k_months_ahead_is_1_02_to_the_k(constant_table,
                                                                    months_ahead):
    origin = week_in(2024, 4)                        # knows 2024-03
    needed = months_from((2024, 3), months_ahead + 1)[-1]
    year, month = months_from(needed, 2)[-1]         # a week in the month after `needed`
    target = week_in(year, month)
    got = projected_levels(HAND_ZERO, origin, [origin, target])
    assert got.iloc[1] / got.iloc[0] == pytest.approx(1.02 ** months_ahead, rel=1e-12)


def test_constant_growth_projection_past_the_tables_end_extends_the_table(constant_table):
    # The table ends 2024-12. A target in June 2025 needs May 2025, 14 months past March.
    origin = week_in(2024, 4)
    got = projected_levels(HAND_ZERO, origin, [week_in(2025, 6)]).iloc[0]
    extended = constant_growth_table(count=60)
    assert (2025, 5) not in constant_table
    assert got == pytest.approx(extended[(2025, 5)] / extended[(2023, 12)], rel=1e-12)


def test_constant_growth_projection_equals_the_full_table_for_every_target(constant_table):
    # Wherever the table has the month, the projection (computed without it) agrees with it.
    origin = week_in(2024, 2)                        # knows 2024-01
    targets = list(range(-40, 50))
    got = projected_levels(HAND_ZERO, origin, targets)
    for week in targets:
        needed = published_month(HAND_ZERO + timedelta(days=7 * week))
        assert got.loc[week] == pytest.approx(
            constant_table[needed] / constant_table[(2023, 12)], rel=1e-12)


def test_constant_growth_projection_does_not_depend_on_the_window(constant_table):
    origin, target = week_in(2024, 6), week_in(2024, 12)
    values = [projected_levels(HAND_ZERO, origin, [target], window=w).iloc[0]
              for w in [1, 2, 6, 12, 100]]
    assert values == pytest.approx([values[0]] * 5, rel=1e-12)


def test_projection_with_zero_growth_is_flat(monkeypatch):
    monkeypatch.setattr(inflation, "TUFE", {k: 500.0 for k in months_from((2023, 1), 12)})
    got = projected_levels(HAND_ZERO, 0, [0, 10, 50, 300])
    assert got.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_projection_with_falling_prices_falls(monkeypatch):
    monkeypatch.setattr(inflation, "TUFE", constant_growth_table(rate=-0.01))
    got = projected_levels(HAND_ZERO, week_in(2024, 1), [0, week_in(2024, 3), week_in(2024, 9)])
    assert got.iloc[0] == 1.0
    assert got.iloc[1] == pytest.approx(0.99 ** 2, rel=1e-12)
    assert got.iloc[2] == pytest.approx(0.99 ** 8, rel=1e-12)


# --- the window ---------------------------------------------------------------------------

@pytest.fixture
def two_speed_table(monkeypatch):
    """2023: 1% a month from Jan to Sep (8 steps), then 3% for Oct, Nov, Dec (3 steps)."""
    rates = [0.01] * 8 + [0.03] * 3
    level, table = 100.0, {(2023, 1): 100.0}
    for key, rate in zip(months_from((2023, 2), 11), rates):
        level *= 1 + rate
        table[key] = level
    monkeypatch.setattr(inflation, "TUFE", table)
    return table


# Origin: week 0 (2024-01-01), which knows December 2023, also week 0's month. A target in
# March 2024 needs February: two months ahead, so the level is exp(2 * growth).
TWO_AHEAD = week_in(2024, 3)


@pytest.mark.parametrize("window,growth", [
    (1, math.log(1.03)),
    (3, math.log(1.03)),
    (4, (math.log(1.01) + 3 * math.log(1.03)) / 4),
    (6, (3 * math.log(1.01) + 3 * math.log(1.03)) / 6),
    (11, (8 * math.log(1.01) + 3 * math.log(1.03)) / 11),
    (100, (8 * math.log(1.01) + 3 * math.log(1.03)) / 11),   # the whole table
])
def test_projection_window_limits_the_history_used(two_speed_table, window, growth):
    got = projected_levels(HAND_ZERO, 0, [TWO_AHEAD], window=window).iloc[0]
    assert got == pytest.approx(math.exp(2 * growth), rel=1e-12)


def test_projection_default_window_is_the_projection_window(two_speed_table):
    assert inflation.PROJECTION_WINDOW == 6
    assert (projected_levels(HAND_ZERO, 0, [TWO_AHEAD]).iloc[0]
            == projected_levels(HAND_ZERO, 0, [TWO_AHEAD], window=6).iloc[0])


def test_projection_window_does_not_change_published_targets(two_speed_table):
    targets = [-4, -1, 0, 3]                         # publish Nov, Nov, Dec, Dec 2023
    for window in [1, 3, 100]:
        got = projected_levels(HAND_ZERO, 0, targets, window=window)
        for week in targets:
            needed = published_month(HAND_ZERO + timedelta(days=7 * week))
            assert got.loc[week] == two_speed_table[needed] / two_speed_table[(2023, 12)]


WINDOW_MESSAGE = "window is {window}; the rate of inflation needs at least 1 month"
LAG_MESSAGE = "lag_months is {lag}; a negative lag reads TÜFE months not yet published"


@pytest.mark.parametrize("window", [0, -1, -2, -7, -100])
def test_projection_window_below_one_raises_the_exact_message(two_speed_table, window):
    # -1 used to take the whole history silently ([-0:]), -2 dropped the oldest month.
    with pytest.raises(ValueError) as error:
        projected_levels(HAND_ZERO, 0, [TWO_AHEAD], window=window)
    assert type(error.value) is ValueError
    assert str(error.value) == WINDOW_MESSAGE.format(window=window)


@pytest.mark.parametrize("window", [0, -1, -2])
def test_projection_window_below_one_raises_even_with_nothing_to_project(two_speed_table,
                                                                          window):
    with pytest.raises(ValueError) as error:
        projected_levels(HAND_ZERO, 0, [-4, 0, 3], window=window)    # all published
    assert str(error.value) == WINDOW_MESSAGE.format(window=window)


def test_projection_window_one_is_the_smallest_valid_window(two_speed_table):
    got = projected_levels(HAND_ZERO, 0, [TWO_AHEAD], window=1).iloc[0]
    assert got == pytest.approx(1.03 ** 2, rel=1e-12)


@pytest.mark.parametrize("lag", [-1, -2, -12])
def test_projection_negative_lag_raises_the_exact_message(constant_table, lag):
    with pytest.raises(ValueError) as error:
        projected_levels(HAND_ZERO, 0, [0, week_in(2024, 5)], lag_months=lag)
    assert type(error.value) is ValueError
    assert str(error.value) == LAG_MESSAGE.format(lag=lag)


def test_projection_negative_lag_raises_even_with_nothing_to_project(constant_table):
    with pytest.raises(ValueError) as error:
        projected_levels(HAND_ZERO, 0, [0], lag_months=-1)
    assert str(error.value) == LAG_MESSAGE.format(lag=-1)


def test_projection_negative_lag_raises_on_the_real_table():
    # Previously lag -1 at origin 40 quietly built its rate from March 2026, unpublished.
    with pytest.raises(ValueError) as error:
        projected_levels(WEEK_ZERO, 40, [60], lag_months=-1)
    assert str(error.value) == LAG_MESSAGE.format(lag=-1)


def test_projection_lag_zero_is_still_valid(constant_table):
    assert projected_levels(HAND_ZERO, 0, [0], lag_months=0).iloc[0] == 1.0


# --- the order of the argument checks: empty targets, window, lag, then coverage ---------

def test_check_order_empty_targets_beat_a_bad_window_and_lag():
    with pytest.raises(ValueError, match="no target weeks"):
        projected_levels(WEEK_ZERO, 40, [], window=0, lag_months=-1)


def test_check_order_empty_targets_beat_a_bad_window():
    with pytest.raises(ValueError, match="no target weeks"):
        projected_levels(WEEK_ZERO, 40, [], window=-1)


def test_check_order_empty_targets_beat_a_negative_lag():
    with pytest.raises(ValueError, match="no target weeks"):
        projected_levels(WEEK_ZERO, 40, [], lag_months=-1)


def test_check_order_bad_window_beats_a_negative_lag():
    with pytest.raises(ValueError) as error:
        projected_levels(WEEK_ZERO, 40, [60], window=0, lag_months=-1)
    assert str(error.value) == WINDOW_MESSAGE.format(window=0)


def test_check_order_bad_window_beats_an_uncovered_origin():
    with pytest.raises(ValueError) as error:
        projected_levels(WEEK_ZERO, 70, [70], window=0)
    assert type(error.value) is ValueError
    assert str(error.value) == WINDOW_MESSAGE.format(window=0)


def test_check_order_negative_lag_beats_an_uncovered_origin():
    with pytest.raises(ValueError) as error:
        projected_levels(WEEK_ZERO, 70, [70], lag_months=-1)
    assert type(error.value) is ValueError
    assert str(error.value) == LAG_MESSAGE.format(lag=-1)


def test_check_order_bad_window_and_lag_beat_an_uncovered_week_zero():
    # Week zero 2024-09-02 publishes August 2024, not in the table.
    for kwargs, message in [({"window": 0}, WINDOW_MESSAGE.format(window=0)),
                            ({"lag_months": -1}, LAG_MESSAGE.format(lag=-1))]:
        with pytest.raises(ValueError) as error:
            projected_levels(date(2024, 9, 2), 10, [10], **kwargs)
        assert type(error.value) is ValueError
        assert str(error.value) == message


def test_check_order_bad_window_beats_a_missing_target_month():
    with pytest.raises(ValueError) as error:
        projected_levels(WEEK_ZERO, 10, [-39], window=0)
    assert str(error.value) == WINDOW_MESSAGE.format(window=0)


# --- the Monday guard: week_zero must be a Monday ----------------------------------------

def monday_message(day):
    # The weekday name comes from %A, as in the source (locale-dependent; see below).
    return f"week_zero {day} is a {day:%A}, not a Monday; weeks run Monday to Sunday"


NOT_MONDAYS = [WEEK_ZERO + timedelta(days=offset) for offset in range(1, 7)]   # Tue..Sun


def test_monday_guard_premise_week_zeros_used_in_this_file_are_mondays():
    for day in [WEEK_ZERO, HAND_ZERO, date(2024, 9, 2)]:
        assert day.weekday() == 0


@pytest.mark.parametrize("day", NOT_MONDAYS, ids=lambda d: f"{d:%a}")
def test_monday_guard_rejects_every_other_weekday_with_the_exact_message(day):
    with pytest.raises(ValueError) as error:
        projected_levels(day, 40, [40, 60])
    assert type(error.value) is ValueError
    assert str(error.value) == monday_message(day)


@pytest.mark.parametrize("day,name", zip(NOT_MONDAYS, ["Tuesday", "Wednesday", "Thursday",
                                                       "Friday", "Saturday", "Sunday"]))
def test_monday_guard_message_names_the_weekday_in_english(day, name):
    # Holds under an English/C locale; %A follows the process locale.
    with pytest.raises(ValueError) as error:
        projected_levels(day, 40, [40])
    assert f"week_zero {day.isoformat()} is a {name}, not a Monday" in str(error.value)


@pytest.mark.parametrize("day", NOT_MONDAYS, ids=lambda d: f"{d:%a}")
def test_monday_guard_raises_even_when_nothing_needs_projecting(day):
    with pytest.raises(ValueError, match="not a Monday"):
        projected_levels(day, 0, [0])


def test_monday_guard_accepts_a_monday():
    assert projected_levels(WEEK_ZERO, 40, [0]).iloc[0] == 1.0


@pytest.mark.parametrize("monday", [date(2024, 10, 7), date(2025, 1, 6), date(2026, 8, 31)])
def test_monday_guard_accepts_other_mondays(monday):
    assert monday.weekday() == 0
    assert projected_levels(monday, 0, [0]).tolist() == [1.0]


def test_monday_guard_accepts_a_timestamp_monday_with_the_same_result():
    targets = [-10, 0, 40, 41, 66, 90]
    by_date = projected_levels(WEEK_ZERO, 40, targets)
    by_timestamp = projected_levels(pd.Timestamp(WEEK_ZERO), 40, targets)
    pd.testing.assert_series_equal(by_timestamp, by_date)


def test_monday_guard_rejects_a_timestamp_tuesday():
    day = pd.Timestamp("2025-06-03")
    with pytest.raises(ValueError, match="not a Monday") as error:
        projected_levels(day, 40, [40])
    assert type(error.value) is ValueError
    assert "Tuesday" in str(error.value)


def test_check_order_empty_targets_beat_a_non_monday():
    with pytest.raises(ValueError, match="no target weeks"):
        projected_levels(NOT_MONDAYS[0], 40, [])


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"lag_months": -1},
                                    {"window": -1, "lag_months": -1}])
def test_check_order_non_monday_beats_a_bad_window_or_lag(kwargs):
    day = NOT_MONDAYS[2]
    with pytest.raises(ValueError) as error:
        projected_levels(day, 40, [60], **kwargs)
    assert str(error.value) == monday_message(day)


def test_check_order_non_monday_beats_an_uncovered_origin():
    day = NOT_MONDAYS[0]
    with pytest.raises(ValueError) as error:
        projected_levels(day, 70, [70])
    assert type(error.value) is ValueError
    assert str(error.value) == monday_message(day)


def test_check_order_non_monday_beats_an_uncovered_week_zero():
    # 2024-09-03 (Tuesday) would publish August 2024, not in the table.
    day = date(2024, 9, 3)
    with pytest.raises(ValueError) as error:
        projected_levels(day, 10, [10])
    assert type(error.value) is ValueError
    assert str(error.value) == monday_message(day)


# --- gaps in the table --------------------------------------------------------------------

def test_projection_rate_divides_by_calendar_months_not_table_rows(monkeypatch):
    # November 2023 is missing: October to December is two months, not one step.
    monkeypatch.setattr(inflation, "TUFE", {(2023, 10): 100.0, (2023, 12): 100.0 * 1.02 ** 2})
    got = projected_levels(HAND_ZERO, 0, [0, week_in(2024, 2)])
    assert got.iloc[0] == 1.0
    assert got.iloc[1] == pytest.approx(1.02, rel=1e-12)


def test_projection_target_on_a_missing_published_month_raises(monkeypatch):
    # Week -1 starts 2023-12-25 and needs November 2023: published, but not in the table.
    monkeypatch.setattr(inflation, "TUFE", {(2023, 10): 100.0, (2023, 12): 104.04})
    with pytest.raises(inflation.InflationCoverageError, match="target week -1") as error:
        projected_levels(HAND_ZERO, 0, [0, -1])
    assert "2023-11" in str(error.value)


def test_projection_window_counts_table_rows_across_a_gap(monkeypatch):
    # Rows: Jun, Jul, Sep, Dec 2023 (gaps). window=2 takes the last 3 rows: Jul, Sep, Dec,
    # a 5-month span. Growth 1%/month throughout, so the rate is log(1.01) whatever the gap.
    table = {(2023, 6): 100.0, (2023, 7): 101.0, (2023, 9): 101.0 * 1.01 ** 2,
             (2023, 12): 101.0 * 1.01 ** 5}
    monkeypatch.setattr(inflation, "TUFE", table)
    got = projected_levels(HAND_ZERO, 0, [week_in(2024, 4)], window=2).iloc[0]   # needs Mar
    assert got == pytest.approx(1.01 ** 3, rel=1e-12)
    # And the first row (June) really is outside the window: a different June shows nothing.
    monkeypatch.setitem(table, (2023, 6), 50.0)
    assert projected_levels(HAND_ZERO, 0, [week_in(2024, 4)], window=2).iloc[0] == got


# --- no leak: nothing published after the origin is read ----------------------------------

def doubled_after(table, known):
    return {k: (v * 2 if k > known else v) for k, v in table.items()}


@pytest.mark.parametrize("origin", [0, 5, 13, 40, 61, 64])
def test_projected_levels_ignore_every_month_published_after_the_origin(monkeypatch, origin):
    targets = list(range(FIRST, 130))
    before = projected_levels(WEEK_ZERO, origin, targets)
    known = published_month(WEEK_ZERO + timedelta(days=7 * origin))
    monkeypatch.setattr(inflation, "TUFE", doubled_after(dict(inflation.TUFE), known))
    after = projected_levels(WEEK_ZERO, origin, targets)
    pd.testing.assert_series_equal(after, before)


def test_projected_levels_ignore_dropping_every_month_after_the_origin(monkeypatch):
    origin, targets = 40, list(range(FIRST, 130))
    before = projected_levels(WEEK_ZERO, origin, targets)
    known = published_month(WEEK_ZERO + timedelta(days=7 * origin))
    monkeypatch.setattr(inflation, "TUFE",
                        {k: v for k, v in inflation.TUFE.items() if k <= known})
    pd.testing.assert_series_equal(projected_levels(WEEK_ZERO, origin, targets), before)


def test_projected_levels_leak_check_has_teeth(monkeypatch):
    # Doubling the month the origin itself publishes DOES move the projection.
    origin, targets = 40, [60]
    before = projected_levels(WEEK_ZERO, origin, targets).iloc[0]
    monkeypatch.setitem(inflation.TUFE, (2026, 2), inflation.TUFE[(2026, 2)] * 2)
    assert projected_levels(WEEK_ZERO, origin, targets).iloc[0] != before


def test_projected_levels_negative_origin_ratios_ignore_later_months(monkeypatch):
    # Origin -10 knows February 2025, before week 0's May 2025. The normalising figure is
    # May's (read from the table), so the LEVELS move with it, but every ratio between two
    # targets depends only on what the origin knows.
    origin, targets = -10, list(range(FIRST, 100))
    before = projected_levels(WEEK_ZERO, origin, targets)
    monkeypatch.setattr(inflation, "TUFE", doubled_after(dict(inflation.TUFE), (2025, 2)))
    after = projected_levels(WEEK_ZERO, origin, targets)
    np.testing.assert_allclose(after.to_numpy() * 2, before.to_numpy(), rtol=1e-14)


def test_projection_with_a_hand_table_ignores_months_after_known_with_lag_two(monkeypatch,
                                                                             constant_table):
    # Lag 2: origin week 0 (January 2024) knows November 2023, not December.
    targets = list(range(-20, 60))
    before = projected_levels(HAND_ZERO, 0, targets, lag_months=2)
    monkeypatch.setattr(inflation, "TUFE", doubled_after(constant_table, (2023, 11)))
    after = projected_levels(HAND_ZERO, 0, targets, lag_months=2)
    pd.testing.assert_series_equal(after, before)
    # Under the default lag, December is known, so the same change moves the result.
    monkeypatch.setattr(inflation, "TUFE", constant_table)
    lag_one = projected_levels(HAND_ZERO, 0, targets)
    monkeypatch.setattr(inflation, "TUFE", doubled_after(constant_table, (2023, 11)))
    assert not projected_levels(HAND_ZERO, 0, targets).equals(lag_one)


# --- lag_months and year boundaries -------------------------------------------------------

def test_projection_lag_two_normalises_to_the_month_two_before_week_zero(constant_table):
    # Week 0 (January 2024) under lag 2 publishes November 2023, so week 0 is 1.0 and a week
    # in March 2024 (needs January, 2 months past November) is 1.02 ** 2.
    got = projected_levels(HAND_ZERO, 0, [0, week_in(2024, 3)], lag_months=2)
    assert got.iloc[0] == 1.0
    assert got.iloc[1] == pytest.approx(1.02 ** 2, rel=1e-12)


def test_projection_lag_zero_uses_the_weeks_own_month(constant_table):
    # Lag 0: week 0 publishes January 2024 itself; a week in May 2024 needs May: 4 months.
    got = projected_levels(HAND_ZERO, 0, [0, week_in(2024, 5)], lag_months=0)
    assert got.iloc[0] == 1.0
    assert got.iloc[1] == pytest.approx(1.02 ** 4, rel=1e-12)


def test_projection_lag_zero_projection_ignores_the_next_month(monkeypatch, constant_table):
    targets = [0, week_in(2024, 5)]
    before = projected_levels(HAND_ZERO, 0, targets, lag_months=0)
    monkeypatch.setattr(inflation, "TUFE", doubled_after(constant_table, (2024, 1)))
    pd.testing.assert_series_equal(projected_levels(HAND_ZERO, 0, targets, lag_months=0),
                                   before)


def test_projection_across_december_to_january(constant_table):
    # Origin in December 2024 knows November 2024; a target in February 2025 needs January
    # 2025, two months on across the year boundary.
    origin = week_in(2024, 12)
    got = projected_levels(HAND_ZERO, origin, [origin, week_in(2025, 2)])
    assert got.iloc[0] == pytest.approx(ratio((2024, 11), (2023, 12)), rel=1e-12)
    assert got.iloc[1] / got.iloc[0] == pytest.approx(1.02 ** 2, rel=1e-12)


def test_projection_origin_in_january_knows_december(constant_table):
    # Origin in January 2024 (week 0): a target in the same January needs December 2023,
    # which is known: exact, 1.0. A target in February needs January: projected one month.
    got = projected_levels(HAND_ZERO, 0, [week_in(2024, 1) + 3, week_in(2024, 2)])
    assert got.iloc[0] == 1.0
    assert got.iloc[1] == pytest.approx(1.02, rel=1e-12)


def test_projection_real_table_year_boundary_origin(synthetic_published):
    # Origin 31 starts 2026-01-05, knows December 2025. Week 35 (2026-02-02) needs January.
    assert published_month(WEEK_ZERO + timedelta(days=7 * 31)) == (2025, 12)
    got = projected_levels(WEEK_ZERO, 31, [30, 31, 35])
    tufe = inflation.TUFE
    assert got.loc[30] == pytest.approx(synthetic_published.loc[30], rel=1e-15)
    assert got.loc[31] == pytest.approx(synthetic_published.loc[31], rel=1e-15)
    growth = (math.log(tufe[(2025, 12)]) - math.log(tufe[(2025, 6)])) / 6
    assert got.loc[35] == pytest.approx(
        tufe[(2025, 12)] * math.exp(growth) / tufe[(2025, 5)], rel=1e-14)


# --- errors --------------------------------------------------------------------------------

def test_projected_levels_empty_targets_raise_a_plain_value_error():
    with pytest.raises(ValueError, match="no target weeks") as error:
        projected_levels(WEEK_ZERO, 40, [])
    assert type(error.value) is ValueError


def test_projected_levels_empty_targets_win_over_an_uncovered_origin():
    with pytest.raises(ValueError, match="no target weeks") as error:
        projected_levels(WEEK_ZERO, 500, [])
    assert not isinstance(error.value, inflation.InflationCoverageError)


@pytest.mark.parametrize("origin", [70, 75, 200])
def test_projected_levels_origin_past_the_table_raises_coverage_error(origin):
    # Origin 70 starts 2026-10-05: its published month, September 2026, is not in the table.
    with pytest.raises(inflation.InflationCoverageError, match=f"week {origin} "):
        projected_levels(WEEK_ZERO, origin, [origin])


def test_projected_levels_origin_error_names_the_month_and_the_start():
    with pytest.raises(inflation.InflationCoverageError) as error:
        projected_levels(WEEK_ZERO, 70, [0])
    assert "2026-09" in str(error.value)
    assert "2026-10-05" in str(error.value)


def test_projected_levels_origin_before_the_table_raises_coverage_error():
    # Origin -39 starts 2024-09-02: it publishes August 2024, before the table starts.
    with pytest.raises(inflation.InflationCoverageError, match="2024-08"):
        projected_levels(WEEK_ZERO, -39, [0])


def test_projected_levels_origin_check_does_not_depend_on_the_targets():
    # Even with only covered targets that need no projection, the origin is checked.
    with pytest.raises(inflation.InflationCoverageError):
        projected_levels(WEEK_ZERO, 70, [0, 1, 2])


def test_projected_levels_week_zero_not_covered_raises_naming_week_zero():
    # Week zero 2024-09-02 publishes August 2024, not in the table; origin 10 is covered.
    week_zero = date(2024, 9, 2)
    assert published_month(week_zero + timedelta(days=70)) in inflation.TUFE
    with pytest.raises(inflation.InflationCoverageError, match="week 0 ") as error:
        projected_levels(week_zero, 10, [10])
    assert "2024-08" in str(error.value)


def test_projected_levels_origin_error_comes_before_the_week_zero_error():
    # Week zero 2024-09-02 (needs 2024-08) and origin -5 (2024-07-29, needs 2024-06).
    with pytest.raises(inflation.InflationCoverageError, match="2024-06"):
        projected_levels(date(2024, 9, 2), -5, [0])


def test_projected_levels_target_before_the_table_raises_coverage_error():
    with pytest.raises(inflation.InflationCoverageError, match="target week -39") as error:
        projected_levels(WEEK_ZERO, 10, [0, -39])
    assert "2024-08" in str(error.value)


def test_projected_levels_coverage_errors_are_value_errors():
    with pytest.raises(ValueError):
        projected_levels(WEEK_ZERO, 70, [0])


def test_projection_with_one_month_up_to_known_raises_plain_value_error(monkeypatch):
    # Only December 2023 is known at week 0; June 2024 exists but is in the future.
    monkeypatch.setattr(inflation, "TUFE", {(2023, 12): 100.0, (2024, 6): 200.0})
    with pytest.raises(ValueError, match="fewer than two months") as error:
        projected_levels(HAND_ZERO, 0, [week_in(2024, 7)])   # needs June: must project
    assert type(error.value) is ValueError
    assert "2023-12" in str(error.value)


def test_projection_with_one_month_is_fine_if_no_target_needs_projecting(monkeypatch):
    monkeypatch.setattr(inflation, "TUFE", {(2023, 12): 100.0})
    got = projected_levels(HAND_ZERO, 0, [0, 1, 2, 3, 4])   # all January 2024
    assert got.tolist() == [1.0] * 5


def test_projection_errors_follow_target_order(monkeypatch):
    # Targets are handled in order: whichever failing target comes first decides the error.
    monkeypatch.setattr(inflation, "TUFE", {(2023, 12): 100.0})
    with pytest.raises(inflation.InflationCoverageError):
        projected_levels(HAND_ZERO, 0, [-1, week_in(2024, 7)])
    with pytest.raises(ValueError, match="fewer than two"):
        projected_levels(HAND_ZERO, 0, [week_in(2024, 7), -1])
