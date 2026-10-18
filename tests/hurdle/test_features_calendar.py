"""Tests for `hurdle/features_calendar.py`: `calendar_features`, `last_business_days`,
`last_day_of_month`, `CALENDAR_COLUMNS`, `QUARTER_END_MONTHS` and `MID_MONTH_DAY`.

Hand-built `HolidayTable`s for the unit cases, a day-by-day reference for a brute-force
comparison over several years, and the real settings/holidays_tr.toml (read only) for the
verified facts. Nothing here reads or writes data/interim.
"""
import copy
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.features_calendar import (
    CALENDAR_COLUMNS,
    MID_MONTH_DAY,
    QUARTER_END_MONTHS,
    calendar_features,
    last_business_days,
    last_day_of_month,
)
from collection_estimation.hurdle.holidays import HolidayTable, load_holidays

from .workbooks import SYNTHETIC_HOLIDAYS

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLIDAYS_TR = REPO_ROOT / "settings" / "holidays_tr.toml"
WEEK_ZERO = date(2025, 6, 2)  # a Monday


# --- helpers ------------------------------------------------------------------------------

def first_sunday_of_february(year):
    day = date(year, 2, 1)
    return day + timedelta(days=(6 - day.weekday()) % 7)


def harmless_bayrams(years):
    """Both Bayrams for each year, each on a single Sunday in February: they satisfy the
    coverage check without changing any weekday count, month-end or mid-month flag."""
    bayrams = []
    for year in years:
        sunday = first_sunday_of_february(year)
        bayrams.append((sunday, sunday, "Ramazan Bayramı"))
        bayrams.append((sunday + timedelta(days=7), sunday + timedelta(days=7),
                        "Kurban Bayramı"))
    return tuple(bayrams)


def table(fixed=None, extra=(), years=range(2023, 2031)):
    """A HolidayTable with `fixed` holidays, `extra` one-day holidays (listed as Ramazan
    Bayramı entries: only their dates matter here), and harmless Bayrams for `years`."""
    extra_bayrams = tuple((day, day, "Ramazan Bayramı") for day in extra)
    return HolidayTable(fixed=dict(fixed or {}), bayrams=harmless_bayrams(years) + extra_bayrams)


def week_of(monday, week_zero=WEEK_ZERO):
    assert monday.weekday() == 0, f"{monday} is not a Monday"
    delta = (monday - week_zero).days
    assert delta % 7 == 0
    return delta // 7


def one_week(monday, holidays, week_zero=WEEK_ZERO):
    """The row of the week starting on `monday`, from a one-week range."""
    week = week_of(monday, week_zero)
    result = calendar_features(week_zero, week, week, holidays)
    assert len(result) == 1
    return plain(result.iloc[0])


def row(result, monday, week_zero=WEEK_ZERO):
    rows = result[result["week"] == week_of(monday, week_zero)]
    assert len(rows) == 1
    return plain(rows.iloc[0])


def plain(series):
    """A row as a dict of plain Python values (so that `is True` checks a real bool)."""
    return {key: value.item() if hasattr(value, "item") else value
            for key, value in series.items()}


def holiday_dates(holidays, start, end):
    """Every holiday date from `start` to `end`, built independently of the module."""
    days = set()
    for year in range(start.year, end.year + 1):
        for (month, day) in holidays.fixed:
            try:
                days.add(date(year, month, day))
            except ValueError:
                pass
    for first, last, _ in holidays.bayrams:
        day = first
        while day <= last:
            days.add(day)
            day += timedelta(days=1)
    return days


def reference(week_zero, first_week, last_week, holidays):
    """A slow, day-by-day reference for calendar_features."""
    monday0 = week_zero + timedelta(days=7 * first_week)
    sunday_n = week_zero + timedelta(days=7 * last_week + 6)
    holidays_set = holiday_dates(holidays, monday0 - timedelta(days=60),
                                 sunday_n + timedelta(days=60))

    def business(day):
        return day.weekday() < 5 and day not in holidays_set

    def is_last_business_day(day):
        if not business(day):
            return False
        later = day + timedelta(days=1)
        while later.month == day.month:
            if business(later):
                return False
            later += timedelta(days=1)
        return True

    rows = []
    for week in range(first_week, last_week + 1):
        monday = week_zero + timedelta(days=7 * week)
        days = [monday + timedelta(days=i) for i in range(7)]
        before = [monday - timedelta(days=7) + timedelta(days=i) for i in range(5)]
        month_ends = [day for day in days if is_last_business_day(day)]
        rows.append({
            "week": week,
            "business_days": sum(business(day) for day in days),
            "has_month_end": bool(month_ends),
            "has_quarter_end": any(day.month in (3, 6, 9, 12) for day in month_ends),
            "has_mid_month": any(day.day == 15 for day in days),
            "holidays_week_before": sum(day in holidays_set for day in before),
        })
    return pd.DataFrame(rows, columns=CALENDAR_COLUMNS).astype({
        "week": "int64", "business_days": "int64", "holidays_week_before": "int64",
        "has_month_end": bool, "has_quarter_end": bool, "has_mid_month": bool})


# --- constants ----------------------------------------------------------------------------

def test_calendar_columns_are_in_the_documented_order():
    assert CALENDAR_COLUMNS == ["week", "business_days", "has_month_end", "has_quarter_end",
                                "has_mid_month", "holidays_week_before"]


def test_quarter_end_months_and_mid_month_day():
    assert tuple(QUARTER_END_MONTHS) == (3, 6, 9, 12)
    assert MID_MONTH_DAY == 15


# --- last_day_of_month --------------------------------------------------------------------

@pytest.mark.parametrize("day, expected", [
    (date(2025, 1, 1), date(2025, 1, 31)),
    (date(2025, 2, 10), date(2025, 2, 28)),
    (date(2024, 2, 10), date(2024, 2, 29)),   # leap year
    (date(2100, 2, 1), date(2100, 2, 28)),    # century, not a leap year
    (date(2000, 2, 1), date(2000, 2, 29)),    # 400-year leap year
    (date(2025, 4, 30), date(2025, 4, 30)),   # already the last day
    (date(2025, 11, 1), date(2025, 11, 30)),
    (date(2025, 12, 1), date(2025, 12, 31)),  # year rollover in the computation
    (date(2025, 12, 31), date(2025, 12, 31)),
])
def test_last_day_of_month(day, expected):
    assert last_day_of_month(day) == expected


def test_last_day_of_month_every_month_of_a_year():
    for month in range(1, 13):
        last = last_day_of_month(date(2026, month, 1))
        assert last.month == month
        assert (last + timedelta(days=1)).day == 1


# --- last_business_days -------------------------------------------------------------------

def weekday_only(day):
    return day.weekday() < 5


def test_last_business_days_weekdays_only():
    result = last_business_days(date(2025, 8, 1), date(2025, 10, 31), weekday_only)
    # Aug 31 2025 is a Sunday, Sep 30 a Tuesday, Oct 31 a Friday.
    assert result == {date(2025, 8, 29), date(2025, 9, 30), date(2025, 10, 31)}


def test_last_business_days_only_within_the_given_range():
    # The range stops on 15 Sep: the latest business day seen for September is 15 Sep.
    result = last_business_days(date(2025, 9, 1), date(2025, 9, 15), weekday_only)
    assert result == {date(2025, 9, 15)}


def test_last_business_days_month_without_any_business_day_is_absent():
    result = last_business_days(date(2025, 8, 1), date(2025, 9, 30),
                                lambda day: day.month == 9 and day.weekday() < 5)
    assert result == {date(2025, 9, 30)}


def test_last_business_days_empty_range():
    assert last_business_days(date(2025, 9, 2), date(2025, 9, 1), weekday_only) == set()


# --- errors -------------------------------------------------------------------------------

def test_last_week_before_first_week_raises():
    with pytest.raises(ValueError, match=r"^last_week 3 is before first_week 4$"):
        calendar_features(WEEK_ZERO, 4, 3, SYNTHETIC_HOLIDAYS)


def test_last_week_before_first_week_negative_weeks():
    with pytest.raises(ValueError, match=r"last_week -5 is before first_week -4"):
        calendar_features(WEEK_ZERO, -4, -5, SYNTHETIC_HOLIDAYS)


def test_range_reaching_a_year_without_bayrams_raises():
    # SYNTHETIC_HOLIDAYS lists 2025 and 2026. Monday 28 Dec 2026's week ends on 3 Jan 2027.
    week = week_of(date(2026, 12, 28))
    with pytest.raises(ValueError, match=r"no dates listed for \['Ramazan Bayramı 2027', "
                                         r"'Kurban Bayramı 2027'\]"):
        calendar_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS)


def test_last_week_ending_in_december_does_not_need_the_next_year():
    week = week_of(date(2026, 12, 21))  # Sunday 27 Dec; the range ends on 31 Dec 2026
    result = calendar_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS)
    assert len(result) == 1


def test_week_before_the_range_reaching_a_year_without_bayrams_raises():
    # The first week starts on Monday 5 Jan 2025; the week before it starts on 29 Dec 2024.
    week = week_of(date(2025, 1, 6))
    with pytest.raises(ValueError, match=r"Ramazan Bayramı 2024"):
        calendar_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS)


def test_end_of_last_weeks_month_reaching_a_year_without_bayrams_raises():
    # The last week (Mon 28 Dec 2026 .. Sun 3 Jan 2027) ends in January: the range runs to
    # 31 Jan 2027, a year without Bayrams.
    holidays = table(years=[2026])
    week = week_of(date(2026, 12, 28))
    with pytest.raises(ValueError, match=r"2027"):
        calendar_features(WEEK_ZERO, week, week, holidays)


# --- week_zero must be a Monday ------------------------------------------------------------

@pytest.mark.parametrize("day, weekday_name", [
    (date(2025, 6, 3), "Tuesday"),
    (date(2025, 6, 4), "Wednesday"),
    (date(2025, 6, 5), "Thursday"),
    (date(2025, 6, 6), "Friday"),
    (date(2025, 6, 7), "Saturday"),
    (date(2025, 6, 8), "Sunday"),
])
def test_week_zero_not_a_monday_raises(day, weekday_name):
    expected = f"week_zero {day.isoformat()} is a {weekday_name}, not a Monday"
    with pytest.raises(ValueError) as caught:
        calendar_features(day, 0, 3, SYNTHETIC_HOLIDAYS)
    assert str(caught.value) == expected


def test_week_zero_tuesday_exact_example_message():
    with pytest.raises(ValueError, match=r"^week_zero 2025-06-03 is a Tuesday, not a Monday$"):
        calendar_features(date(2025, 6, 3), 0, 0, SYNTHETIC_HOLIDAYS)


@pytest.mark.parametrize("week_zero", [
    pd.Timestamp("2025-06-08"),            # Sunday
    datetime(2025, 6, 4, 13, 45, 10),      # Wednesday with a time of day
    pd.Timestamp("2025-06-08 23:59:59"),   # Sunday, one second before Monday 9 Jun
])
def test_week_zero_not_a_monday_as_timestamp_or_datetime_raises_with_the_date_only(week_zero):
    day = pd.Timestamp(week_zero).date()
    expected = f"week_zero {day.isoformat()} is a {day.strftime('%A')}, not a Monday"
    with pytest.raises(ValueError) as caught:
        calendar_features(week_zero, 0, 3, SYNTHETIC_HOLIDAYS)
    assert str(caught.value) == expected
    assert ":" not in str(caught.value)  # no time of day in the message


@pytest.mark.parametrize("week_zero", [
    date(2025, 6, 2),
    pd.Timestamp("2025-06-02"),
    pd.Timestamp("2025-06-02 17:30:00"),
    datetime(2025, 6, 2),
    datetime(2025, 6, 2, 23, 59, 59, 999999),
    datetime(2025, 6, 2, 0, 0, 1),
])
def test_week_zero_monday_in_any_form_passes_and_matches_the_date_table(week_zero):
    expected = calendar_features(WEEK_ZERO, -2, 12, SYNTHETIC_HOLIDAYS)
    result = calendar_features(week_zero, -2, 12, SYNTHETIC_HOLIDAYS)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("monday", [date(2023, 1, 9), date(2024, 12, 30), date(2026, 3, 30)])
def test_other_mondays_pass(monday):
    assert monday.weekday() == 0
    result = calendar_features(monday, 0, 2, table())
    assert result["week"].tolist() == [0, 1, 2]


def test_non_monday_error_takes_precedence_over_last_week_before_first_week():
    with pytest.raises(ValueError, match=r"^week_zero 2025-06-05 is a Thursday, not a Monday$"):
        calendar_features(date(2025, 6, 5), 4, 3, SYNTHETIC_HOLIDAYS)


def test_non_monday_error_takes_precedence_over_missing_bayram_years():
    # The weeks would reach 2027, which SYNTHETIC_HOLIDAYS does not cover.
    with pytest.raises(ValueError, match=r"^week_zero 2025-06-03 is a Tuesday, not a Monday$"):
        calendar_features(date(2025, 6, 3), 0, 200, SYNTHETIC_HOLIDAYS)


def test_monday_week_zero_still_gives_last_week_before_first_week_error():
    with pytest.raises(ValueError, match=r"^last_week 3 is before first_week 4$"):
        calendar_features(pd.Timestamp("2025-06-02 09:00"), 4, 3, SYNTHETIC_HOLIDAYS)


def test_random_week_zeros_of_the_brute_force_test_are_mondays():
    # Guard: the brute-force test draws week_zero at random; every draw must be a Monday.
    for seed in range(12):
        rng = random.Random(1000 + seed)
        week_zero = date(2025, 6, 2) + timedelta(days=7 * rng.randint(-60, 60))
        assert week_zero.weekday() == 0, (seed, week_zero)


def test_a_year_with_only_one_bayram_raises():
    holidays = HolidayTable(fixed={}, bayrams=harmless_bayrams([2025])[:1])
    with pytest.raises(ValueError, match=r"no dates listed for \['Kurban Bayramı 2025'\]"):
        calendar_features(WEEK_ZERO, 0, 2, holidays)


# --- shape, dtypes, inputs ----------------------------------------------------------------

def test_columns_dtypes_and_row_count():
    result = calendar_features(WEEK_ZERO, -3, 10, SYNTHETIC_HOLIDAYS)
    assert list(result.columns) == CALENDAR_COLUMNS
    assert len(result) == 14
    assert result["week"].tolist() == list(range(-3, 11))
    for column in ("week", "business_days", "holidays_week_before"):
        assert result[column].dtype == "int64", column
    for column in ("has_month_end", "has_quarter_end", "has_mid_month"):
        assert result[column].dtype == bool, column
    assert isinstance(result.index, pd.RangeIndex)
    assert result.index.tolist() == list(range(14))


def test_value_ranges():
    result = calendar_features(WEEK_ZERO, 0, 69, SYNTHETIC_HOLIDAYS)
    assert result["business_days"].between(0, 5).all()
    assert result["holidays_week_before"].between(0, 5).all()
    # A quarter-end week is always a month-end week.
    assert not (result["has_quarter_end"] & ~result["has_month_end"]).any()


def test_single_week_range_has_one_row():
    result = calendar_features(WEEK_ZERO, 7, 7, SYNTHETIC_HOLIDAYS)
    assert len(result) == 1
    assert result["week"].tolist() == [7]
    assert list(result.columns) == CALENDAR_COLUMNS


def test_inputs_are_not_mutated():
    holidays = HolidayTable(fixed=dict(SYNTHETIC_HOLIDAYS.fixed),
                            bayrams=SYNTHETIC_HOLIDAYS.bayrams)
    fixed_before = copy.deepcopy(holidays.fixed)
    bayrams_before = copy.deepcopy(holidays.bayrams)
    week_zero = date(2025, 6, 2)
    calendar_features(week_zero, 0, 30, holidays)
    assert holidays.fixed == fixed_before
    assert holidays.bayrams == bayrams_before
    assert week_zero == date(2025, 6, 2)


def test_week_zero_as_timestamp_or_datetime_gives_the_same_table():
    expected = calendar_features(WEEK_ZERO, -5, 40, SYNTHETIC_HOLIDAYS)
    for week_zero in (pd.Timestamp("2025-06-02"), datetime(2025, 6, 2)):
        pd.testing.assert_frame_equal(
            calendar_features(week_zero, -5, 40, SYNTHETIC_HOLIDAYS), expected)


def test_deterministic():
    first = calendar_features(WEEK_ZERO, 0, 50, SYNTHETIC_HOLIDAYS)
    second = calendar_features(WEEK_ZERO, 0, 50, SYNTHETIC_HOLIDAYS)
    pd.testing.assert_frame_equal(first, second)


def test_a_sub_range_matches_the_same_weeks_of_a_longer_range():
    # The month-end of a range's last week must not depend on where the range stops.
    full = calendar_features(WEEK_ZERO, -10, 70, SYNTHETIC_HOLIDAYS)
    for last in range(-10, 71, 3):
        part = calendar_features(WEEK_ZERO, -10, last, SYNTHETIC_HOLIDAYS)
        pd.testing.assert_frame_equal(part, full.iloc[:len(part)].reset_index(drop=True))
    for week in range(-10, 71, 7):
        single = calendar_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS)
        pd.testing.assert_frame_equal(
            single, full[full["week"] == week].reset_index(drop=True))


# --- business_days ------------------------------------------------------------------------

def test_plain_week_has_five_business_days():
    assert one_week(date(2025, 9, 8), table())["business_days"] == 5


def test_holiday_on_a_weekday_removes_a_business_day():
    # Republic Day, Wed 29 Oct 2025.
    result = one_week(date(2025, 10, 27), table({(10, 29): "Republic Day"}))
    assert result["business_days"] == 4


def test_holiday_on_saturday_and_sunday_does_not_change_business_days():
    # Victory Day, Sat 30 Aug 2025; and Sun 31 Aug 2025.
    holidays = table({(8, 30): "Victory Day", (8, 31): "A Sunday holiday"})
    assert one_week(date(2025, 8, 25), holidays)["business_days"] == 5


def test_whole_week_of_holidays_has_zero_business_days():
    days = [date(2025, 9, 8) + timedelta(days=i) for i in range(5)]
    assert one_week(date(2025, 9, 8), table(extra=days))["business_days"] == 0


def test_multi_day_bayram_across_a_weekend():
    # Kurban Bayramı Fri 6 .. Mon 9 Jun 2025 (SYNTHETIC_HOLIDAYS).
    result = calendar_features(WEEK_ZERO, 0, 1, SYNTHETIC_HOLIDAYS)
    assert result["business_days"].tolist() == [4, 4]


# --- holidays_week_before -----------------------------------------------------------------

def test_holidays_week_before_counts_the_previous_weeks_weekday_holidays():
    # Republic Day Wed 29 Oct 2025, in the week of Mon 27 Oct.
    holidays = table({(10, 29): "Republic Day"})
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 10, 27)),
                               week_of(date(2025, 11, 3)), holidays)
    assert row(result, date(2025, 10, 27))["holidays_week_before"] == 0
    assert row(result, date(2025, 11, 3))["holidays_week_before"] == 1


def test_first_weeks_holidays_week_before_uses_the_week_before_the_range():
    # Commemoration of Atatürk, Mon 19 May 2025; the range starts the week after.
    result = calendar_features(WEEK_ZERO, -1, 0, SYNTHETIC_HOLIDAYS)
    assert row(result, date(2025, 5, 26))["holidays_week_before"] == 1
    assert row(result, date(2025, 6, 2))["holidays_week_before"] == 0


def test_first_weeks_holidays_week_before_with_a_bayram_before_the_range():
    # Kurban Bayramı starts Fri 6 Jun 2025: the range starting on Mon 9 Jun counts it.
    result = calendar_features(WEEK_ZERO, 1, 1, SYNTHETIC_HOLIDAYS)
    assert result["holidays_week_before"].tolist() == [1]
    assert result["business_days"].tolist() == [4]


def test_first_weeks_holidays_week_before_across_new_year():
    # New Year's Day, Thu 1 Jan 2026, in the week of Mon 29 Dec 2025.
    week = week_of(date(2026, 1, 5))
    result = calendar_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS)
    assert result["holidays_week_before"].tolist() == [1]


def test_saturday_holiday_does_not_count_in_holidays_week_before():
    # Victory Day Sat 30 Aug 2025; and a Sunday holiday on 31 Aug.
    holidays = table({(8, 30): "Victory Day", (8, 31): "A Sunday holiday"})
    assert one_week(date(2025, 9, 1), holidays)["holidays_week_before"] == 0


def test_holidays_week_before_five_when_the_whole_previous_week_is_holidays():
    days = [date(2025, 9, 8) + timedelta(days=i) for i in range(7)]  # Mon .. Sun
    assert one_week(date(2025, 9, 15), table(extra=days))["holidays_week_before"] == 5


def test_holidays_in_the_week_itself_do_not_count_in_holidays_week_before():
    days = [date(2025, 9, 8) + timedelta(days=i) for i in range(5)]
    assert one_week(date(2025, 9, 8), table(extra=days))["holidays_week_before"] == 0


def test_holidays_two_weeks_before_do_not_count():
    assert one_week(date(2025, 9, 15),
                    table(extra=[date(2025, 9, 5)]))["holidays_week_before"] == 0


# --- has_month_end ------------------------------------------------------------------------

def test_month_ending_on_a_sunday_flags_the_week_of_the_last_friday():
    # 31 Aug 2025 is a Sunday; the last business day is Fri 29 Aug.
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 8, 18)),
                               week_of(date(2025, 9, 1)), table())
    assert row(result, date(2025, 8, 18))["has_month_end"] is False
    assert row(result, date(2025, 8, 25))["has_month_end"] is True
    assert row(result, date(2025, 9, 1))["has_month_end"] is False


def test_month_ending_on_a_saturday():
    # 28 Feb 2026 is a Saturday; the last business day is Fri 27 Feb, week of Mon 23 Feb.
    result = calendar_features(WEEK_ZERO, week_of(date(2026, 2, 16)),
                               week_of(date(2026, 3, 2)), table())
    assert row(result, date(2026, 2, 16))["has_month_end"] is False
    assert row(result, date(2026, 2, 23))["has_month_end"] is True
    assert row(result, date(2026, 3, 2))["has_month_end"] is False


def test_month_ending_on_a_monday_flags_that_monday_week_not_the_week_before():
    # 30 Jun 2025 is a Monday.
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 6, 23)),
                               week_of(date(2025, 6, 30)), table())
    assert row(result, date(2025, 6, 23))["has_month_end"] is False
    assert row(result, date(2025, 6, 30))["has_month_end"] is True
    assert row(result, date(2025, 6, 30))["has_quarter_end"] is True


def test_month_whose_last_weekdays_are_holidays_moves_the_month_end_a_week_earlier():
    # 31 Mar 2026 is a Tuesday; Mon 30 and Tue 31 are holidays, so the last business day of
    # March is Fri 27 Mar, in the week of Mon 23 Mar.
    holidays = table(extra=[date(2026, 3, 30), date(2026, 3, 31)])
    result = calendar_features(WEEK_ZERO, week_of(date(2026, 3, 23)),
                               week_of(date(2026, 3, 30)), holidays)
    early = row(result, date(2026, 3, 23))
    late = row(result, date(2026, 3, 30))
    assert early["has_month_end"] is True
    assert early["has_quarter_end"] is True
    assert late["has_month_end"] is False
    assert late["has_quarter_end"] is False
    assert late["business_days"] == 3


def test_month_ending_on_a_sunday_after_a_week_of_holidays():
    # Aug 2025: every weekday of Mon 25 .. Fri 29 Aug is a holiday, 31 Aug is a Sunday: the
    # last business day is Fri 22 Aug.
    days = [date(2025, 8, 25) + timedelta(days=i) for i in range(5)]
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 8, 18)),
                               week_of(date(2025, 9, 1)), table(extra=days))
    assert row(result, date(2025, 8, 18))["has_month_end"] is True
    assert row(result, date(2025, 8, 25))["has_month_end"] is False
    assert row(result, date(2025, 8, 25))["business_days"] == 0
    assert row(result, date(2025, 9, 1))["holidays_week_before"] == 5


def test_month_end_in_the_last_week_when_the_month_continues_after_the_range():
    # The range ends on Sun 28 Sep 2025; Tue 30 Sep is the last business day of September,
    # after the range: the last week must NOT be flagged.
    last = week_of(date(2025, 9, 22))
    result = calendar_features(WEEK_ZERO, last - 2, last, table())
    assert result["has_month_end"].tolist() == [False, False, False]
    assert result["has_quarter_end"].tolist() == [False, False, False]


def test_month_end_in_the_last_week_when_the_rest_of_the_month_is_holidays():
    # The range ends on Sun 28 Sep 2025; Mon 29 and Tue 30 Sep (after the range) are
    # holidays, so Fri 26 Sep, inside the last week, is the last business day of September.
    holidays = table(extra=[date(2025, 9, 29), date(2025, 9, 30)])
    last = week_of(date(2025, 9, 22))
    result = calendar_features(WEEK_ZERO, last, last, holidays)
    assert result["has_month_end"].tolist() == [True]
    assert result["has_quarter_end"].tolist() == [True]


def test_week_spanning_two_months():
    # Mon 29 Sep .. Sun 5 Oct 2025: holds Tue 30 Sep (last business day of September).
    result = one_week(date(2025, 9, 29), table())
    assert result["has_month_end"] is True
    assert result["has_quarter_end"] is True
    assert result["has_mid_month"] is False
    assert result["business_days"] == 5


def test_week_spanning_two_months_without_the_month_end():
    # Mon 27 Oct .. Sun 2 Nov 2025: Fri 31 Oct is the last business day of October.
    assert one_week(date(2025, 10, 27), table())["has_month_end"] is True
    # Mon 3 Nov .. Sun 9 Nov holds no month-end.
    assert one_week(date(2025, 11, 3), table())["has_month_end"] is False


def test_month_with_no_business_day_flags_no_week():
    # Every weekday of February 2027 is a holiday: February has no last business day.
    days = []
    day = date(2027, 2, 1)
    while day.month == 2:
        days.append(day)
        day += timedelta(days=1)
    holidays = table(extra=days)
    result = calendar_features(WEEK_ZERO, week_of(date(2027, 1, 25)),
                               week_of(date(2027, 3, 1)), holidays)
    flagged = result.loc[result["has_month_end"], "week"].tolist()
    # Only January's (Fri 29 Jan, week of Mon 25 Jan) and March's are possible; March 2027's
    # last business day (Wed 31) is not in the range.
    assert flagged == [week_of(date(2027, 1, 25))]


# --- has_quarter_end ----------------------------------------------------------------------

def test_quarter_end_only_for_march_june_september_december():
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 6, 2)),
                               week_of(date(2026, 5, 25)), table())
    month_end_weeks = result[result["has_month_end"]]
    quarter_weeks = result[result["has_quarter_end"]]
    assert len(month_end_weeks) == 12
    assert len(quarter_weeks) == 4
    mondays = [WEEK_ZERO + timedelta(days=7 * int(week)) for week in quarter_weeks["week"]]
    assert mondays == [date(2025, 6, 30), date(2025, 9, 29), date(2025, 12, 29),
                       date(2026, 3, 30)]


def test_december_year_end_is_a_quarter_end():
    # Wed 31 Dec 2025, in the week of Mon 29 Dec .. Sun 4 Jan 2026.
    result = one_week(date(2025, 12, 29), table({(1, 1): "New Year's Day"}))
    assert result["has_month_end"] is True
    assert result["has_quarter_end"] is True
    assert result["business_days"] == 4  # Thu 1 Jan is a holiday


def test_december_year_end_moved_by_holidays_into_the_week_before():
    # Mon 29 .. Wed 31 Dec 2025 made holidays: the last business day is Fri 26 Dec.
    holidays = table(extra=[date(2025, 12, 29), date(2025, 12, 30), date(2025, 12, 31)])
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 12, 22)),
                               week_of(date(2025, 12, 29)), holidays)
    assert row(result, date(2025, 12, 22))["has_quarter_end"] is True
    assert row(result, date(2025, 12, 29))["has_quarter_end"] is False
    assert row(result, date(2025, 12, 29))["has_month_end"] is False


def test_a_week_holding_a_non_quarter_month_end_is_not_a_quarter_end():
    # Fri 31 Oct 2025.
    result = one_week(date(2025, 10, 27), table())
    assert result["has_month_end"] is True
    assert result["has_quarter_end"] is False


# --- has_mid_month ------------------------------------------------------------------------

def test_mid_month_on_a_weekday():
    # Mon 15 Sep 2025.
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 9, 8)),
                               week_of(date(2025, 9, 15)), table())
    assert result["has_mid_month"].tolist() == [False, True]


def test_holiday_on_the_15th_still_gives_mid_month():
    # Democracy and National Unity Day, Tue 15 Jul 2025.
    result = one_week(date(2025, 7, 14), table({(7, 15): "Democracy Day"}))
    assert result["has_mid_month"] is True
    assert result["business_days"] == 4


def test_the_15th_on_a_saturday_still_gives_mid_month():
    # Sat 15 Nov 2025, in the week of Mon 10 Nov.
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 11, 10)),
                               week_of(date(2025, 11, 17)), table())
    assert result["has_mid_month"].tolist() == [True, False]


def test_the_15th_on_a_sunday_still_gives_mid_month():
    # Sun 15 Feb 2026, in the week of Mon 9 Feb; the next week does not hold it.
    result = calendar_features(WEEK_ZERO, week_of(date(2026, 2, 9)),
                               week_of(date(2026, 2, 16)), table())
    assert result["has_mid_month"].tolist() == [True, False]


def test_every_month_has_exactly_one_mid_month_week():
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 6, 2)),
                               week_of(date(2026, 5, 25)), table())
    assert result["has_mid_month"].sum() == 12


# --- negative weeks -----------------------------------------------------------------------

def test_negative_weeks():
    # week -1 = Mon 26 May 2025, week -2 = Mon 19 May 2025 (a holiday).
    result = calendar_features(WEEK_ZERO, -2, -1, SYNTHETIC_HOLIDAYS)
    assert result["week"].tolist() == [-2, -1]
    minus_two, minus_one = result.iloc[0], result.iloc[1]
    assert minus_two["business_days"] == 4
    assert minus_one["business_days"] == 5
    assert minus_one["holidays_week_before"] == 1
    assert bool(minus_one["has_month_end"]) is True   # Fri 30 May 2025
    assert bool(minus_two["has_month_end"]) is False
    assert bool(minus_two["has_mid_month"]) is False  # Mon 19 .. Sun 25 May


def test_range_crossing_week_zero():
    result = calendar_features(WEEK_ZERO, -3, 3, SYNTHETIC_HOLIDAYS)
    assert result["week"].tolist() == [-3, -2, -1, 0, 1, 2, 3]
    pd.testing.assert_frame_equal(result, reference(WEEK_ZERO, -3, 3, SYNTHETIC_HOLIDAYS))


# --- brute force against the day-by-day reference -----------------------------------------

def random_holidays(seed, years):
    rng = random.Random(seed)
    fixed = dict(SYNTHETIC_HOLIDAYS.fixed)
    for _ in range(rng.randint(0, 6)):
        month = rng.randint(1, 12)
        day = rng.randint(1, 31 if month != 2 else 29)  # 29 Feb: only in leap years
        try:
            date(2024, month, day)
        except ValueError:
            continue
        fixed[(month, day)] = "Random fixed holiday"
    bayrams = []
    for year in years:
        for name in ("Ramazan Bayramı", "Kurban Bayramı"):
            first = date(year, 1, 1) + timedelta(days=rng.randint(0, 360))
            last = first + timedelta(days=rng.randint(0, 4))
            bayrams.append((first, last, name))
    # Occasionally a block covering the end of a month, to move month-ends.
    for _ in range(rng.randint(0, 4)):
        year = rng.choice(list(years))
        month = rng.randint(1, 12)
        end = last_day_of_month_ref(year, month)
        first = end - timedelta(days=rng.randint(0, 6))
        bayrams.append((first, end, "Ramazan Bayramı"))
    return HolidayTable(fixed=fixed, bayrams=tuple(bayrams))


def last_day_of_month_ref(year, month):
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


@pytest.mark.parametrize("seed", range(12))
def test_matches_the_day_by_day_reference_over_several_years(seed):
    years = range(2022, 2031)
    holidays = random_holidays(seed, years)
    rng = random.Random(1000 + seed)
    week_zero = date(2025, 6, 2) + timedelta(days=7 * rng.randint(-60, 60))
    # From early 2023 to late 2029, within the covered years.
    first = week_of(date(2023, 1, 9), week_zero) + rng.randint(0, 20)
    last = week_of(date(2029, 11, 26), week_zero) - rng.randint(0, 20)
    result = calendar_features(week_zero, first, last, holidays)
    pd.testing.assert_frame_equal(result, reference(week_zero, first, last, holidays))


@pytest.mark.parametrize("seed", range(40))
def test_short_ranges_match_the_reference(seed):
    # Short ranges stop at arbitrary weeks: the last week's month-end must be judged
    # against the whole month, not the range.
    years = range(2022, 2031)
    holidays = random_holidays(500 + seed, years)
    rng = random.Random(2000 + seed)
    first = week_of(date(2023, 1, 9)) + rng.randint(0, 300)
    last = first + rng.randint(0, 6)
    result = calendar_features(WEEK_ZERO, first, last, holidays)
    pd.testing.assert_frame_equal(result, reference(WEEK_ZERO, first, last, holidays))


def test_real_holidays_match_the_reference():
    holidays = load_holidays(HOLIDAYS_TR)
    result = calendar_features(WEEK_ZERO, 0, 69, holidays)
    pd.testing.assert_frame_equal(result, reference(WEEK_ZERO, 0, 69, holidays))


# --- tmp_path TOML file ---------------------------------------------------------------------

def test_holidays_from_a_tmp_path_toml_file(tmp_path):
    path = tmp_path / "holidays_test.toml"
    path.write_text(
        '[fixed]\n"10-29" = "Republic Day"\n\n'
        '[[bayram]]\nname = "Ramazan Bayramı"\nfirst_day = 2025-03-30\nlast_day = 2025-04-01\n\n'
        '[[bayram]]\nname = "Kurban Bayramı"\nfirst_day = 2025-09-29\nlast_day = 2025-09-30\n',
        encoding="utf-8")
    holidays = load_holidays(path)
    # The "Kurban Bayramı" here covers Mon 29 and Tue 30 Sep 2025, so September's last
    # business day is Fri 26 Sep.
    result = calendar_features(WEEK_ZERO, week_of(date(2025, 9, 22)),
                               week_of(date(2025, 10, 6)), holidays)
    assert row(result, date(2025, 9, 22))["has_quarter_end"] is True
    assert row(result, date(2025, 9, 29))["has_month_end"] is False
    assert row(result, date(2025, 9, 29))["business_days"] == 3
    assert row(result, date(2025, 10, 6))["holidays_week_before"] == 2


# --- the real settings file: verified facts -----------------------------------------------

@pytest.fixture(scope="module")
def real_calendar():
    return calendar_features(WEEK_ZERO, 0, 69, load_holidays(HOLIDAYS_TR))


def test_real_week_51_kurban_bayrami(real_calendar):
    week = row(real_calendar, date(2026, 5, 25))
    assert week["week"] == 51
    assert week["business_days"] == 2
    assert week["has_month_end"] is True  # Tue 26 May 2026
    assert week["has_quarter_end"] is False


def test_real_week_52_after_kurban_bayrami(real_calendar):
    week = row(real_calendar, date(2026, 6, 1))
    assert week["week"] == 52
    assert week["holidays_week_before"] == 3  # Wed 27 .. Fri 29 May; Sat 30 not counted
    assert week["has_month_end"] is False


def test_real_counts_over_weeks_0_to_69(real_calendar):
    assert len(real_calendar) == 70
    assert real_calendar["has_month_end"].sum() == 16
    assert real_calendar["has_quarter_end"].sum() == 6
    assert real_calendar["has_mid_month"].sum() == 16


def test_real_quarter_end_mondays(real_calendar):
    weeks = real_calendar.loc[real_calendar["has_quarter_end"], "week"]
    mondays = [WEEK_ZERO + timedelta(days=7 * int(week)) for week in weeks]
    assert mondays == [date(2025, 6, 30), date(2025, 9, 29), date(2025, 12, 29),
                       date(2026, 3, 30), date(2026, 6, 29), date(2026, 9, 28)]


def test_real_short_weeks(real_calendar):
    short = dict(zip(real_calendar.loc[real_calendar["business_days"] < 5, "week"],
                     real_calendar.loc[real_calendar["business_days"] < 5, "business_days"]))
    assert short == {0: 4, 1: 4, 6: 4, 21: 4, 30: 4, 41: 4, 46: 4, 47: 4, 50: 4, 51: 2,
                     58: 4}
