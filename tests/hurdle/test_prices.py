"""Tests for `hurdle/prices.py`: week numbers to TÜFE price levels, the coverage guard, and
deflation.

Week zero is Monday 2025-06-02 throughout, as in the synthetic settings. The synthetic range
is weeks -32 (Monday 2024-10-21) to 64 (Monday 2026-08-24). Published levels are normalised
to 1.0 at week 0 (not at the last week, as the copied module does), so they must not depend
on where the range stops.
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
