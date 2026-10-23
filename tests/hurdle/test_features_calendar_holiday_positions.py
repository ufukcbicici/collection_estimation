"""Tests for `hurdle/features_calendar.py` U7b: `holiday_position_features`,
`HOLIDAY_ENCODINGS` and `WEEKDAY_NAMES`.

Hand-built `HolidayTable`s for the unit cases (harmless Bayrams on February Sundays satisfy the
coverage check without touching any weekday), a slow day-by-day reference written the other
way round (from each holiday back to the days it reaches) for a brute-force comparison, and
the real settings/holidays_tr.toml (read only) for the verified facts. Nothing here reads or
writes data/interim.
"""
import copy
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle import settings as settings_module
from collection_estimation.hurdle.features_calendar import (
    HOLIDAY_ENCODINGS,
    WEEKDAY_NAMES,
    calendar_features,
    holiday_position_features,
)
from collection_estimation.hurdle.holidays import HolidayTable, load_holidays

from .workbooks import SYNTHETIC_HOLIDAYS

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLIDAYS_TR = REPO_ROOT / "settings" / "holidays_tr.toml"
WEEK_ZERO = date(2025, 6, 2)  # a Monday
DRAIN = ["drain_days", "catch_up_days"]
DAYS = [f"{label}_{name}" for label in ("prev", "cur", "next")
        for name in ("mon", "tue", "wed", "thu", "fri")]


# --- helpers ------------------------------------------------------------------------------

def first_sunday_of_february(year):
    day = date(year, 2, 1)
    return day + timedelta(days=(6 - day.weekday()) % 7)


def harmless_bayrams(years):
    bayrams = []
    for year in years:
        sunday = first_sunday_of_february(year)
        bayrams.append((sunday, sunday, "Ramazan Bayramı"))
        bayrams.append((sunday + timedelta(days=7), sunday + timedelta(days=7),
                        "Kurban Bayramı"))
    return tuple(bayrams)


def table(fixed=None, extra=(), years=range(2023, 2031)):
    """A HolidayTable with `fixed` holidays, `extra` one-day holidays and harmless Bayrams."""
    extra_bayrams = tuple((day, day, "Ramazan Bayramı") for day in extra)
    return HolidayTable(fixed=dict(fixed or {}), bayrams=harmless_bayrams(years) + extra_bayrams)


def week_of(monday, week_zero=WEEK_ZERO):
    assert monday.weekday() == 0, f"{monday} is not a Monday"
    delta = (monday - week_zero).days
    assert delta % 7 == 0
    return delta // 7


def positions(holidays, monday, encodings=("drain", "days"), drain=3, catch_up=5,
              week_zero=WEEK_ZERO):
    """The row of the week starting on `monday`, from a one-week range, as plain ints."""
    week = week_of(monday, week_zero)
    result = holiday_position_features(week_zero, week, week, holidays, encodings, drain,
                                       catch_up)
    assert len(result) == 1
    return {key: int(value) for key, value in result.iloc[0].items()}


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
    return {day for day in days if start <= day <= end}


def reference(week_zero, first_week, last_week, holidays, encodings, drain, catch_up):
    """A slow reference, formulated from the holidays outwards: the days a weekday holiday
    drains (the `drain` days before it) and the days it feeds (the `catch_up` days after)."""
    monday0 = week_zero + timedelta(days=7 * first_week)
    sunday_n = week_zero + timedelta(days=7 * last_week + 6)
    weekday_holidays = {day for day in holiday_dates(holidays, monday0 - timedelta(days=400),
                                                     sunday_n + timedelta(days=400))
                        if day.weekday() < 5}
    drained, fed = set(), set()
    for holiday in weekday_holidays:
        for k in range(1, drain + 1):
            drained.add(holiday - timedelta(days=k))
        for k in range(1, catch_up + 1):
            fed.add(holiday + timedelta(days=k))

    def business(day):
        return day.weekday() < 5 and day not in weekday_holidays

    rows = []
    for week in range(first_week, last_week + 1):
        monday = week_zero + timedelta(days=7 * week)
        week_days = [monday + timedelta(days=i) for i in range(7)]
        row = {"week": week}
        if "drain" in encodings:
            row["drain_days"] = len([d for d in week_days if business(d) and d in drained])
            row["catch_up_days"] = len([d for d in week_days if business(d) and d in fed])
        if "days" in encodings:
            for label, shift in (("prev", -7), ("cur", 0), ("next", 7)):
                for i, name in enumerate(("mon", "tue", "wed", "thu", "fri")):
                    day = monday + timedelta(days=shift + i)
                    row[f"{label}_{name}"] = int(day not in weekday_holidays)
        rows.append(row)
    columns = ["week"] + (DRAIN if "drain" in encodings else []) + \
        (DAYS if "days" in encodings else [])
    return pd.DataFrame(rows, columns=columns).astype("int64")


# --- constants ----------------------------------------------------------------------------

def test_holiday_encodings():
    assert HOLIDAY_ENCODINGS == ("drain", "days")


def test_weekday_names():
    assert WEEKDAY_NAMES == ("mon", "tue", "wed", "thu", "fri")


def test_settings_keeps_the_same_holiday_encodings():
    # settings.py keeps its own copy (a circular import prevents importing it): they must agree.
    assert settings_module.HOLIDAY_ENCODINGS == HOLIDAY_ENCODINGS


# --- errors -------------------------------------------------------------------------------

@pytest.mark.parametrize("day, weekday_name", [
    (date(2025, 6, 3), "Tuesday"), (date(2025, 6, 6), "Friday"), (date(2025, 6, 8), "Sunday"),
])
def test_week_zero_not_a_monday_raises_the_exact_message(day, weekday_name):
    with pytest.raises(ValueError) as caught:
        holiday_position_features(day, 0, 3, SYNTHETIC_HOLIDAYS, ("drain",), 3, 5)
    assert str(caught.value) == f"week_zero {day.isoformat()} is a {weekday_name}, not a Monday"


def test_week_zero_as_a_datetime_not_a_monday_shows_the_date_only():
    with pytest.raises(ValueError) as caught:
        holiday_position_features(datetime(2025, 6, 4, 13, 45), 0, 3, SYNTHETIC_HOLIDAYS,
                                  ("drain",), 3, 5)
    assert str(caught.value) == "week_zero 2025-06-04 is a Wednesday, not a Monday"


def test_last_week_before_first_week_raises():
    with pytest.raises(ValueError, match=r"^last_week 3 is before first_week 4$"):
        holiday_position_features(WEEK_ZERO, 4, 3, SYNTHETIC_HOLIDAYS, ("drain",), 3, 5)


def test_non_monday_error_comes_before_the_range_error():
    with pytest.raises(ValueError, match=r"^week_zero 2025-06-05 is a Thursday, not a Monday$"):
        holiday_position_features(date(2025, 6, 5), 4, 3, SYNTHETIC_HOLIDAYS, ("x",), 0, 0)


@pytest.mark.parametrize("encodings, unknown", [
    (("weekly",), ["weekly"]),
    (("drain", "Days"), ["Days"]),
    (("DRAIN",), ["DRAIN"]),
    (("drain", "days", "drain_days"), ["drain_days"]),
    (("", "days"), [""]),
])
def test_an_unknown_encoding_raises_the_exact_message(encodings, unknown):
    with pytest.raises(ValueError) as caught:
        holiday_position_features(WEEK_ZERO, 0, 3, SYNTHETIC_HOLIDAYS, encodings, 3, 5)
    assert str(caught.value) == (f"unknown holiday encodings {unknown}; "
                                 f"known: ['drain', 'days']")


@pytest.mark.parametrize("encodings", [("drain", "drain"), ("days", "drain", "days")])
def test_a_repeated_encoding_raises_the_exact_message(encodings):
    with pytest.raises(ValueError) as caught:
        holiday_position_features(WEEK_ZERO, 0, 3, SYNTHETIC_HOLIDAYS, encodings, 3, 5)
    assert str(caught.value) == f"holiday encodings {list(encodings)} name one more than once"


def test_an_unknown_encoding_is_reported_before_a_repeat():
    with pytest.raises(ValueError, match=r"^unknown holiday encodings \['x', 'x'\]"):
        holiday_position_features(WEEK_ZERO, 0, 3, SYNTHETIC_HOLIDAYS, ("x", "x"), 3, 5)


@pytest.mark.parametrize("bad", [0, -1, -7, True, False, 1.0, 3.0, 2.5, "3", None])
def test_a_bad_drain_window_raises_the_exact_message(bad):
    with pytest.raises(ValueError) as caught:
        holiday_position_features(WEEK_ZERO, 0, 3, SYNTHETIC_HOLIDAYS, ("drain",), bad, 5)
    assert str(caught.value) == (f"drain_window is {bad!r}, not a whole number of days, "
                                 f"1 or more")


@pytest.mark.parametrize("bad", [0, -1, True, False, 5.0, "5", None])
def test_a_bad_catch_up_window_raises_the_exact_message(bad):
    with pytest.raises(ValueError) as caught:
        holiday_position_features(WEEK_ZERO, 0, 3, SYNTHETIC_HOLIDAYS, ("drain",), 3, bad)
    assert str(caught.value) == (f"catch_up_window is {bad!r}, not a whole number of days, "
                                 f"1 or more")


def test_the_drain_window_is_checked_before_the_catch_up_window():
    with pytest.raises(ValueError, match=r"^drain_window is 0"):
        holiday_position_features(WEEK_ZERO, 0, 3, SYNTHETIC_HOLIDAYS, ("drain",), 0, 0)


@pytest.mark.parametrize("encodings", [(), ("days",)])
def test_windows_are_checked_whatever_the_encodings(encodings):
    # A bad window is refused even when no "drain" column would use it.
    with pytest.raises(ValueError, match=r"^drain_window is 0"):
        holiday_position_features(WEEK_ZERO, 0, 3, SYNTHETIC_HOLIDAYS, encodings, 0, 5)


def test_an_encoding_error_comes_before_a_window_error():
    with pytest.raises(ValueError, match=r"^unknown holiday encodings"):
        holiday_position_features(WEEK_ZERO, 0, 3, SYNTHETIC_HOLIDAYS, ("x",), 0, 0)


def test_window_of_one_is_accepted():
    result = holiday_position_features(WEEK_ZERO, 0, 3, SYNTHETIC_HOLIDAYS, ("drain",), 1, 1)
    assert len(result) == 4


# --- the holidays the function asks for -----------------------------------------------------
# From the first Monday - max(7, catch_up_window) days to the last Monday +
# max(11, 4 + drain_window) days; SYNTHETIC_HOLIDAYS lists the Bayrams of 2025 and 2026.

def test_the_next_weeks_friday_in_a_year_without_bayrams_raises():
    # Mon 21 Dec 2026: the next week's Friday is 1 Jan 2027.
    week = week_of(date(2026, 12, 21))
    with pytest.raises(ValueError, match=r"no dates listed for \['Ramazan Bayramı 2027', "
                                         r"'Kurban Bayramı 2027'\]"):
        holiday_position_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS, ("drain",), 3, 5)


def test_the_range_reaches_the_next_weeks_friday_even_without_the_days_encoding():
    # The documented range does not depend on the encodings: [] still asks for it.
    week = week_of(date(2026, 12, 21))
    with pytest.raises(ValueError, match=r"2027"):
        holiday_position_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS, (), 3, 5)


def test_a_week_whose_next_friday_is_31_december_needs_no_next_year():
    week = week_of(date(2026, 12, 21)) - 1  # Mon 14 Dec 2026; the next week's Friday 25 Dec
    result = holiday_position_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS,
                                       ("drain", "days"), 13, 5)  # 14 Dec + 4 + 13 = 31 Dec
    assert result["week"].tolist() == [week]


def test_a_drain_window_reaching_the_next_year_raises():
    week = week_of(date(2026, 12, 14))  # + 4 + 14 = 1 Jan 2027
    with pytest.raises(ValueError, match=r"2027"):
        holiday_position_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS, ("drain",), 14, 5)


def test_a_catch_up_window_reaching_the_year_before_raises():
    week = week_of(date(2025, 1, 13))  # - 13 = 31 Dec 2024
    with pytest.raises(ValueError, match=r"Ramazan Bayramı 2024"):
        holiday_position_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS, ("drain",), 3, 13)


def test_a_catch_up_window_ending_on_1_january_needs_no_year_before():
    week = week_of(date(2025, 1, 13))  # - 12 = 1 Jan 2025
    result = holiday_position_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS,
                                       ("drain", "days"), 3, 12)
    assert len(result) == 1


def test_the_week_before_in_the_year_before_raises():
    # Mon 6 Jan 2025: the week before starts on 30 Dec 2024.
    week = week_of(date(2025, 1, 6))
    with pytest.raises(ValueError, match=r"2024"):
        holiday_position_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS, ("drain",), 1, 1)


# --- shape, dtypes, inputs ----------------------------------------------------------------

@pytest.mark.parametrize("encodings, columns", [
    ((), ["week"]),
    (("drain",), ["week"] + DRAIN),
    (("days",), ["week"] + DAYS),
    (("drain", "days"), ["week"] + DRAIN + DAYS),
    (("days", "drain"), ["week"] + DRAIN + DAYS),  # the code's order, not the given one
])
def test_columns_follow_the_codes_order(encodings, columns):
    result = holiday_position_features(WEEK_ZERO, -2, 5, SYNTHETIC_HOLIDAYS, encodings, 3, 5)
    assert list(result.columns) == columns


def test_the_days_columns_are_the_15_documented_names():
    assert DAYS == ["prev_mon", "prev_tue", "prev_wed", "prev_thu", "prev_fri",
                    "cur_mon", "cur_tue", "cur_wed", "cur_thu", "cur_fri",
                    "next_mon", "next_tue", "next_wed", "next_thu", "next_fri"]


@pytest.mark.parametrize("encodings", [(), ("drain",), ("days",), ("drain", "days")])
def test_every_column_is_int64_with_one_row_per_week(encodings):
    result = holiday_position_features(WEEK_ZERO, -3, 10, SYNTHETIC_HOLIDAYS, encodings, 3, 5)
    assert len(result) == 14
    assert result["week"].tolist() == list(range(-3, 11))
    for column in result.columns:
        assert result[column].dtype == "int64", column
    assert isinstance(result.index, pd.RangeIndex)
    assert result.index.tolist() == list(range(14))


def test_single_week_range_has_one_row():
    result = holiday_position_features(WEEK_ZERO, 7, 7, SYNTHETIC_HOLIDAYS, ("drain",), 3, 5)
    assert result["week"].tolist() == [7]


def test_encodings_given_as_a_list_give_the_same_table():
    a = holiday_position_features(WEEK_ZERO, 0, 20, SYNTHETIC_HOLIDAYS, ("drain", "days"), 3, 5)
    b = holiday_position_features(WEEK_ZERO, 0, 20, SYNTHETIC_HOLIDAYS, ["drain", "days"], 3, 5)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("week_zero", [pd.Timestamp("2025-06-02"), datetime(2025, 6, 2),
                                       pd.Timestamp("2025-06-02 17:30:00")])
def test_week_zero_as_timestamp_or_datetime_gives_the_same_table(week_zero):
    expected = holiday_position_features(WEEK_ZERO, -5, 40, SYNTHETIC_HOLIDAYS,
                                         ("drain", "days"), 3, 5)
    result = holiday_position_features(week_zero, -5, 40, SYNTHETIC_HOLIDAYS,
                                       ("drain", "days"), 3, 5)
    pd.testing.assert_frame_equal(result, expected)


def test_inputs_are_not_mutated():
    holidays = HolidayTable(fixed=dict(SYNTHETIC_HOLIDAYS.fixed),
                            bayrams=SYNTHETIC_HOLIDAYS.bayrams)
    fixed_before = copy.deepcopy(holidays.fixed)
    encodings = ["days", "drain"]
    holiday_position_features(WEEK_ZERO, 0, 30, holidays, encodings, 3, 5)
    assert holidays.fixed == fixed_before
    assert holidays.bayrams == SYNTHETIC_HOLIDAYS.bayrams
    assert encodings == ["days", "drain"]


def test_deterministic():
    a = holiday_position_features(WEEK_ZERO, 0, 50, SYNTHETIC_HOLIDAYS, ("drain", "days"), 3, 5)
    b = holiday_position_features(WEEK_ZERO, 0, 50, SYNTHETIC_HOLIDAYS, ("drain", "days"), 3, 5)
    pd.testing.assert_frame_equal(a, b)


def test_one_encoding_alone_gives_the_same_columns_as_within_both():
    both = holiday_position_features(WEEK_ZERO, 0, 60, SYNTHETIC_HOLIDAYS, ("drain", "days"), 3, 5)
    drain = holiday_position_features(WEEK_ZERO, 0, 60, SYNTHETIC_HOLIDAYS, ("drain",), 3, 5)
    days = holiday_position_features(WEEK_ZERO, 0, 60, SYNTHETIC_HOLIDAYS, ("days",), 3, 5)
    pd.testing.assert_frame_equal(drain, both[["week"] + DRAIN])
    pd.testing.assert_frame_equal(days, both[["week"] + DAYS])


def test_a_weeks_row_does_not_depend_on_the_range():
    full = holiday_position_features(WEEK_ZERO, -10, 70, SYNTHETIC_HOLIDAYS, ("drain", "days"),
                                     4, 6)
    for week in range(-10, 71):
        single = holiday_position_features(WEEK_ZERO, week, week, SYNTHETIC_HOLIDAYS,
                                           ("drain", "days"), 4, 6)
        pd.testing.assert_frame_equal(single, full[full["week"] == week].reset_index(drop=True))
    for first, last in ((-10, 0), (5, 33), (40, 70), (69, 70)):
        part = holiday_position_features(WEEK_ZERO, first, last, SYNTHETIC_HOLIDAYS,
                                         ("drain", "days"), 4, 6)
        expected = full[(full["week"] >= first) & (full["week"] <= last)]
        pd.testing.assert_frame_equal(part, expected.reset_index(drop=True))


# --- "drain": hand-built cases ----------------------------------------------------------------

def test_a_plain_week_is_normal():
    row = positions(table(), date(2025, 9, 8))
    assert row["drain_days"] == 0 and row["catch_up_days"] == 0
    assert all(row[column] == 1 for column in DAYS)


def test_a_friday_holiday_drains_the_three_days_before_it():
    # Fri 12 Sep 2025, drain window 3: Tue, Wed and Thu; Mon is 4 days before.
    holidays = table(extra=[date(2025, 9, 12)])
    row = positions(holidays, date(2025, 9, 8), drain=3, catch_up=5)
    assert row["drain_days"] == 3
    assert row["catch_up_days"] == 0


def test_a_friday_holiday_feeds_the_next_week_up_to_the_catch_up_window():
    # Fri 12 Sep 2025, catch-up window 5: Mon 15 (3 days), Tue 16 (4), Wed 17 (5); not Thu.
    holidays = table(extra=[date(2025, 9, 12)])
    row = positions(holidays, date(2025, 9, 15), drain=3, catch_up=5)
    assert row["catch_up_days"] == 3
    assert row["drain_days"] == 0


def test_a_monday_holiday_drains_the_friday_before_across_the_weekend():
    holidays = table(extra=[date(2025, 9, 15)])
    before = positions(holidays, date(2025, 9, 8), drain=3, catch_up=5)
    assert before["drain_days"] == 1  # Fri 12 Sep, 3 days before; Thu is 4
    same = positions(holidays, date(2025, 9, 15), drain=3, catch_up=5)
    assert same["catch_up_days"] == 4  # Tue .. Fri, 1 to 4 days after
    assert same["drain_days"] == 0


def test_a_midweek_holiday_is_never_counted_itself():
    # Wed 10 Sep 2025, windows 3 / 5: drains Mon and Tue, feeds Thu and Fri; Wed is neither.
    holidays = table(extra=[date(2025, 9, 10)])
    row = positions(holidays, date(2025, 9, 8), drain=3, catch_up=5)
    assert row["drain_days"] == 2
    assert row["catch_up_days"] == 2


def test_a_holiday_day_reached_by_another_holiday_is_not_counted():
    # Tue 9 and Wed 10 Sep: Tue is 1 day before Wed but is a holiday, so not a business day.
    holidays = table(extra=[date(2025, 9, 9), date(2025, 9, 10)])
    row = positions(holidays, date(2025, 9, 8), drain=3, catch_up=5)
    assert row["drain_days"] == 1   # Mon only
    assert row["catch_up_days"] == 2  # Thu, Fri


def test_each_business_day_counts_once_with_several_holidays_in_reach():
    # Wed 10 and Fri 12 Sep 2025, drain window 5: Mon, Tue and Thu each reach both holidays.
    holidays = table(extra=[date(2025, 9, 10), date(2025, 9, 12)])
    row = positions(holidays, date(2025, 9, 8), drain=5, catch_up=5)
    assert row["drain_days"] == 3
    assert row["catch_up_days"] == 1  # Thu (Wed + 1); Fri is a holiday


def test_counts_never_exceed_the_business_days():
    # Holidays everywhere around a week with one business day.
    days = [date(2025, 9, 1) + timedelta(days=i) for i in range(21)
            if date(2025, 9, 1) + timedelta(days=i) != date(2025, 9, 10)]
    row = positions(table(extra=days), date(2025, 9, 8), drain=10, catch_up=10)
    assert row["drain_days"] == 1 and row["catch_up_days"] == 1
    assert sum(row[f"cur_{name}"] for name in WEEKDAY_NAMES) == 1


def test_a_week_of_holidays_has_no_drain_and_no_catch_up():
    days = [date(2025, 9, 8) + timedelta(days=i) for i in range(5)]
    row = positions(table(extra=days), date(2025, 9, 8), drain=7, catch_up=7)
    assert row["drain_days"] == 0 and row["catch_up_days"] == 0
    assert all(row[f"cur_{name}"] == 0 for name in WEEKDAY_NAMES)


@pytest.mark.parametrize("weekend_day", [date(2025, 9, 13), date(2025, 9, 14)])
def test_a_weekend_holiday_changes_nothing(weekend_day):
    holidays = table(extra=[weekend_day])
    for monday in (date(2025, 9, 1), date(2025, 9, 8), date(2025, 9, 15), date(2025, 9, 22)):
        row = positions(holidays, monday, drain=7, catch_up=7)
        assert row["drain_days"] == 0 and row["catch_up_days"] == 0, monday
        assert all(row[column] == 1 for column in DAYS), monday


def test_window_one_reaches_only_the_neighbouring_day():
    holidays = table(extra=[date(2025, 9, 10)])  # Wed
    row = positions(holidays, date(2025, 9, 8), drain=1, catch_up=1)
    assert row["drain_days"] == 1   # Tue
    assert row["catch_up_days"] == 1  # Thu


def test_a_long_drain_window_reaches_beyond_the_next_week():
    # Fri 19 Sep 2025 is 11 days after Mon 8 Sep; window 10: Tue 9 .. Fri 12 drain, not Mon.
    holidays = table(extra=[date(2025, 9, 19)])
    row = positions(holidays, date(2025, 9, 8), drain=10, catch_up=5)
    assert row["drain_days"] == 4


def test_a_long_catch_up_window_reaches_before_the_previous_week():
    # Fri 29 Aug 2025; Mon 8 Sep is 10 days later, Tue 9 is 11.
    holidays = table(extra=[date(2025, 8, 29)])
    row = positions(holidays, date(2025, 9, 8), drain=3, catch_up=10)
    assert row["catch_up_days"] == 1


@pytest.mark.parametrize("drain, expected", [(1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (9, 4)])
def test_the_drain_count_grows_with_the_window(drain, expected):
    # Fri 12 Sep 2025: Thu, Wed, Tue, Mon are 1, 2, 3, 4 days before it.
    row = positions(table(extra=[date(2025, 9, 12)]), date(2025, 9, 8), drain=drain)
    assert row["drain_days"] == expected


@pytest.mark.parametrize("catch_up, expected", [(1, 0), (2, 0), (3, 1), (4, 2), (5, 3),
                                                (6, 4), (7, 5), (20, 5)])
def test_the_catch_up_count_grows_with_the_window(catch_up, expected):
    # Fri 12 Sep 2025: Mon 15 .. Fri 19 are 3 .. 7 days after it.
    row = positions(table(extra=[date(2025, 9, 12)]), date(2025, 9, 15), catch_up=catch_up)
    assert row["catch_up_days"] == expected


def test_drain_and_catch_up_use_their_own_windows():
    holidays = table(extra=[date(2025, 9, 12)])
    a = positions(holidays, date(2025, 9, 8), drain=1, catch_up=20)
    b = positions(holidays, date(2025, 9, 8), drain=4, catch_up=1)
    assert a["drain_days"] == 1 and b["drain_days"] == 4


def test_a_bayram_spanning_a_weekend_counts_only_its_weekdays():
    # Fri 6 .. Mon 9 Jun 2025 (SYNTHETIC_HOLIDAYS): week 0 (Mon 2 Jun) drains Tue .. Thu; week 1
    # (Mon 9 Jun, itself a holiday) catches up Tue .. Fri (1 to 4 days after Mon 9).
    result = holiday_position_features(WEEK_ZERO, 0, 1, SYNTHETIC_HOLIDAYS, ("drain",), 3, 5)
    assert result["drain_days"].tolist() == [3, 0]
    assert result["catch_up_days"].tolist() == [0, 4]


# --- "days": hand-built cases -----------------------------------------------------------------

def test_days_mark_holidays_in_the_previous_current_and_next_week():
    holidays = table(extra=[date(2025, 9, 3), date(2025, 9, 11), date(2025, 9, 19)])
    row = positions(holidays, date(2025, 9, 8), encodings=("days",))
    zeros = {column for column in DAYS if row[column] == 0}
    assert zeros == {"prev_wed", "cur_thu", "next_fri"}
    assert set(row) == {"week"} | set(DAYS)


def test_days_ignore_holidays_two_weeks_away():
    holidays = table(extra=[date(2025, 8, 29), date(2025, 9, 22)])
    row = positions(holidays, date(2025, 9, 8), encodings=("days",))
    assert all(row[column] == 1 for column in DAYS)


@pytest.mark.parametrize("encodings", [("days",), ("drain", "days")])
def test_days_prev_and_next_are_the_neighbouring_weeks_cur(encodings):
    result = holiday_position_features(WEEK_ZERO, -10, 70, SYNTHETIC_HOLIDAYS, encodings, 3, 5)
    by_week = result.set_index("week")
    for week in range(-9, 70):
        for name in WEEKDAY_NAMES:
            assert by_week.loc[week, f"prev_{name}"] == by_week.loc[week - 1, f"cur_{name}"]
            assert by_week.loc[week, f"next_{name}"] == by_week.loc[week + 1, f"cur_{name}"]


@pytest.mark.parametrize("seed", range(6))
def test_days_cur_sum_equals_calendar_features_business_days(seed):
    holidays = random_holidays(seed, range(2022, 2031))
    first = week_of(date(2023, 1, 9)) + random.Random(seed).randint(0, 30)
    last = first + 250
    days = holiday_position_features(WEEK_ZERO, first, last, holidays, ("days",), 3, 5)
    calendar = calendar_features(WEEK_ZERO, first, last, holidays)
    cur = days[[f"cur_{name}" for name in WEEKDAY_NAMES]].sum(axis=1)
    assert cur.tolist() == calendar["business_days"].tolist()


def test_days_cur_sum_equals_business_days_with_the_real_holidays():
    holidays = load_holidays(HOLIDAYS_TR)
    days = holiday_position_features(WEEK_ZERO, 0, 69, holidays, ("days",), 3, 5)
    calendar = calendar_features(WEEK_ZERO, 0, 69, holidays)
    cur = days[[f"cur_{name}" for name in WEEKDAY_NAMES]].sum(axis=1)
    assert cur.tolist() == calendar["business_days"].tolist()


def test_days_prev_sum_is_five_minus_holidays_week_before():
    holidays = load_holidays(HOLIDAYS_TR)
    days = holiday_position_features(WEEK_ZERO, 0, 69, holidays, ("days",), 3, 5)
    calendar = calendar_features(WEEK_ZERO, 0, 69, holidays)
    prev = days[[f"prev_{name}" for name in WEEKDAY_NAMES]].sum(axis=1)
    assert prev.tolist() == (5 - calendar["holidays_week_before"]).tolist()


def test_drain_and_catch_up_are_within_the_business_days():
    holidays = load_holidays(HOLIDAYS_TR)
    result = holiday_position_features(WEEK_ZERO, 0, 69, holidays, ("drain",), 7, 7)
    calendar = calendar_features(WEEK_ZERO, 0, 69, holidays)
    assert (result["drain_days"] >= 0).all() and (result["catch_up_days"] >= 0).all()
    assert (result["drain_days"] <= calendar["business_days"]).all()
    assert (result["catch_up_days"] <= calendar["business_days"]).all()


# --- brute force against the reference --------------------------------------------------------

def random_holidays(seed, years):
    rng = random.Random(seed)
    fixed = dict(SYNTHETIC_HOLIDAYS.fixed)
    for _ in range(rng.randint(0, 8)):
        month = rng.randint(1, 12)
        day = rng.randint(1, 31 if month != 2 else 29)
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
    # A few clusters of holidays close together, to exercise overlapping windows.
    for _ in range(rng.randint(0, 10)):
        year = rng.choice(list(years))
        first = date(year, 1, 1) + timedelta(days=rng.randint(0, 358))
        for offset in rng.sample(range(7), rng.randint(1, 3)):
            day = first + timedelta(days=offset)
            bayrams.append((day, day, "Ramazan Bayramı"))
    return HolidayTable(fixed=fixed, bayrams=tuple(bayrams))


@pytest.mark.parametrize("seed", range(40))
def test_matches_the_reference_over_random_windows(seed):
    years = range(2022, 2031)
    holidays = random_holidays(seed, years)
    rng = random.Random(3000 + seed)
    week_zero = date(2025, 6, 2) + timedelta(days=7 * rng.randint(-60, 60))
    first = week_of(date(2023, 3, 6), week_zero) + rng.randint(0, 200)
    last = first + rng.randint(0, 120)
    drain, catch_up = rng.randint(1, 15), rng.randint(1, 15)
    encodings = rng.choice([("drain",), ("days",), ("drain", "days"), ("days", "drain"), ()])
    result = holiday_position_features(week_zero, first, last, holidays, encodings, drain,
                                       catch_up)
    expected = reference(week_zero, first, last, holidays, encodings, drain, catch_up)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("drain, catch_up", [(1, 1), (3, 5), (5, 3), (7, 7), (14, 14)])
def test_real_holidays_match_the_reference(drain, catch_up):
    holidays = load_holidays(HOLIDAYS_TR)
    result = holiday_position_features(WEEK_ZERO, 0, 69, holidays, ("drain", "days"), drain,
                                       catch_up)
    pd.testing.assert_frame_equal(
        result, reference(WEEK_ZERO, 0, 69, holidays, ("drain", "days"), drain, catch_up))


def test_holidays_from_a_tmp_path_toml_file(tmp_path):
    path = tmp_path / "holidays_test.toml"
    path.write_text(
        '[fixed]\n"10-29" = "Republic Day"\n\n'
        '[[bayram]]\nname = "Ramazan Bayramı"\nfirst_day = 2025-03-30\nlast_day = 2025-04-01\n\n'
        '[[bayram]]\nname = "Kurban Bayramı"\nfirst_day = 2025-09-29\nlast_day = 2025-09-30\n',
        encoding="utf-8")
    holidays = load_holidays(path)
    # Kurban Mon 29 and Tue 30 Sep 2025: the week before drains Fri 26 (3 days before Mon 29);
    # the week itself catches up Wed .. Fri; Republic Day Wed 29 Oct is far away.
    result = holiday_position_features(WEEK_ZERO, week_of(date(2025, 9, 22)),
                                       week_of(date(2025, 9, 29)), holidays, ("drain", "days"),
                                       3, 5)
    assert result["drain_days"].tolist() == [1, 0]
    assert result["catch_up_days"].tolist() == [0, 3]
    assert result["next_mon"].tolist() == [0, 1]
    assert result["cur_tue"].tolist() == [1, 0]


# --- the real settings file: verified facts ---------------------------------------------------

@pytest.fixture(scope="module")
def real_positions():
    return holiday_position_features(WEEK_ZERO, 0, 69, load_holidays(HOLIDAYS_TR),
                                     ("drain", "days"), 3, 5).set_index("week")


@pytest.mark.parametrize("week, monday, drain, catch_up", [
    (41, date(2026, 3, 16), 3, 0),  # Fri 20 Mar Ramazan Bayramı: Tue, Wed, Thu
    (42, date(2026, 3, 23), 0, 3),  # Mon .. Wed; the Bayram's Sat / Sun do not count
    (50, date(2026, 5, 18), 1, 3),  # Tue 19 May: Mon drains; Wed .. Fri catch up
    (51, date(2026, 5, 25), 2, 0),  # Kurban Wed 27 .. Fri 29 May: Mon, Tue drain
    (12, date(2025, 8, 25), 0, 0),  # Victory Day Sat 30 Aug 2025: nothing
    (13, date(2025, 9, 1), 0, 0),
])
def test_real_known_weeks(real_positions, week, monday, drain, catch_up):
    assert WEEK_ZERO + timedelta(days=7 * week) == monday
    assert real_positions.loc[week, "drain_days"] == drain
    assert real_positions.loc[week, "catch_up_days"] == catch_up


def test_real_week_42_would_be_4_if_the_bayram_weekend_counted(real_positions):
    # Thu 26 Mar is 6 days after Fri 20 but 4 after Sun 22: only a weekday holiday counts.
    assert real_positions.loc[42, "catch_up_days"] == 3


def test_real_week_51_days(real_positions):
    row = real_positions.loc[51]
    assert [int(row[f"cur_{name}"]) for name in WEEKDAY_NAMES] == [1, 1, 0, 0, 0]
    assert [int(row[f"prev_{name}"]) for name in WEEKDAY_NAMES] == [1, 0, 1, 1, 1]
    assert [int(row[f"next_{name}"]) for name in WEEKDAY_NAMES] == [1, 1, 1, 1, 1]


def test_real_weeks_12_and_13_days_are_all_business_days(real_positions):
    for week in (12, 13):
        assert all(real_positions.loc[week, column] == 1 for column in DAYS), week
