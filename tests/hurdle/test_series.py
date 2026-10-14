"""Tests for `hurdle/series.py`: week numbers counted from a fixed Monday, "week zero".

Weeks run Monday to Sunday. The synthetic corpus starts on Monday 2025-06-02, so that date
is week zero throughout.
"""
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.holidays import HolidayTable
from collection_estimation.hurdle.read_collections import list_daily_files, read_one_file
from collection_estimation.hurdle.series import add_week_index, week_calendar, week_numbers
from tests.hurdle.workbooks import SYNTHETIC_HOLIDAYS as HOLIDAYS
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT

DATE_FORMAT = SYNTHETIC_LAYOUT.file_name_date_format  # "%d%m%Y"

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"

WEEK_ZERO = date(2025, 6, 2)  # a Monday


def week_of(day, week_zero=WEEK_ZERO):
    """The week number of a single date."""
    return int(week_numbers(pd.Series([day]), week_zero).iloc[0])


def payment_table():
    """A small table of payment lines with a non-default index."""
    return pd.DataFrame(
        {"company_code": ["C1", "C2", "C1"],
         "customer": ["A", "B", "C"],
         "document_date": pd.to_datetime(["2025-06-10", "2025-06-02", "2025-06-01"]),
         "amount_try": [100.0, 200.0, 300.0]},
        index=[7, 3, 5])


# --- weeks run Monday to Sunday --------------------------------------------------------

def test_week_zero_monday_is_week_0():
    assert week_of(date(2025, 6, 2)) == 0


def test_sunday_of_week_zero_is_week_0():
    assert week_of(date(2025, 6, 8)) == 0


def test_next_monday_is_week_1():
    assert week_of(date(2025, 6, 9)) == 1


def test_sunday_before_week_zero_is_week_minus_1():
    assert week_of(date(2025, 6, 1)) == -1


def test_dates_well_before_week_zero_are_negative():
    assert week_of(date(2025, 5, 26)) == -1
    assert week_of(date(2025, 5, 25)) == -2


def test_last_synthetic_day_is_week_64():
    assert week_of(date(2026, 8, 28)) == 64


def test_every_day_of_a_week_has_the_same_number():
    days = pd.Series([date(2025, 6, 9) + timedelta(days=n) for n in range(7)])
    assert week_numbers(days, WEEK_ZERO).tolist() == [1] * 7


# --- input types -----------------------------------------------------------------------

def test_accepts_date_objects():
    days = pd.Series([date(2025, 6, 2), date(2025, 6, 9)])
    assert week_numbers(days, WEEK_ZERO).tolist() == [0, 1]


def test_accepts_datetime64_values():
    days = pd.Series(pd.to_datetime(["2025-06-02", "2025-06-09"]))
    assert days.dtype.kind == "M"
    assert week_numbers(days, WEEK_ZERO).tolist() == [0, 1]


def test_time_of_day_is_ignored():
    days = pd.Series([pd.Timestamp("2025-06-08 23:59:59"), pd.Timestamp("2025-06-09 00:00:01")])
    assert week_numbers(days, WEEK_ZERO).tolist() == [0, 1]


def test_datetime_objects_with_time_are_accepted():
    days = pd.Series([datetime(2025, 6, 1, 18, 30), datetime(2025, 6, 2, 9, 0)])
    assert week_numbers(days, WEEK_ZERO).tolist() == [-1, 0]


def test_result_dtype_is_int64():
    days = pd.Series([date(2025, 6, 2), date(2025, 6, 1)])
    assert week_numbers(days, WEEK_ZERO).dtype == "int64"


def test_input_index_is_kept():
    days = pd.Series([date(2025, 6, 2), date(2025, 6, 9)], index=[10, 4])
    assert week_numbers(days, WEEK_ZERO).index.tolist() == [10, 4]


def test_empty_series_gives_empty_int64_series():
    result = week_numbers(pd.Series([], dtype="datetime64[ns]"), WEEK_ZERO)
    assert len(result) == 0
    assert result.dtype == "int64"


def test_empty_object_series_gives_empty_int64_series():
    result = week_numbers(pd.Series([], dtype=object), WEEK_ZERO)
    assert len(result) == 0
    assert result.dtype == "int64"


def test_input_series_is_not_modified():
    days = pd.Series([date(2025, 6, 2), date(2025, 6, 9)])
    before = days.copy()
    week_numbers(days, WEEK_ZERO)
    pd.testing.assert_series_equal(days, before)


# --- week zero -------------------------------------------------------------------------

def test_week_zero_not_a_monday_raises_naming_date_and_weekday():
    with pytest.raises(ValueError) as error:
        week_numbers(pd.Series([date(2025, 6, 3)]), date(2025, 6, 3))
    assert "2025-06-03" in str(error.value)
    assert "Tuesday" in str(error.value)


def test_week_zero_on_a_sunday_raises():
    with pytest.raises(ValueError, match="Sunday"):
        week_numbers(pd.Series([date(2025, 6, 2)]), date(2025, 6, 1))


def test_week_zero_check_applies_to_an_empty_series():
    with pytest.raises(ValueError, match="Monday"):
        week_numbers(pd.Series([], dtype=object), date(2025, 6, 3))


def test_week_zero_across_a_year_boundary():
    week_zero = date(2025, 12, 29)  # a Monday
    days = pd.Series([date(2025, 12, 29), date(2026, 1, 4), date(2026, 1, 5)])
    assert week_numbers(days, week_zero).tolist() == [0, 0, 1]


def test_consecutive_mondays_differ_by_one_over_two_years():
    # Spans 2026 (a 53-ISO-week year) and the 2028 leap day: no resets at year ends.
    mondays = pd.Series([date(2025, 12, 29) + timedelta(weeks=n) for n in range(110)])
    weeks = week_numbers(mondays, date(2025, 12, 29))
    assert weeks.tolist() == list(range(110))


def test_every_day_over_two_years_matches_days_divided_by_seven():
    days = [WEEK_ZERO + timedelta(days=n) for n in range(-30, 800)]
    weeks = week_numbers(pd.Series(days), WEEK_ZERO)
    assert weeks.tolist() == [(day - WEEK_ZERO).days // 7 for day in days]


# --- add_week_index --------------------------------------------------------------------

def test_add_week_index_adds_week_from_document_date():
    result = add_week_index(payment_table(), WEEK_ZERO)
    assert result["week"].tolist() == [1, 0, -1]


def test_add_week_index_keeps_every_other_column():
    table = payment_table()
    result = add_week_index(table, WEEK_ZERO)
    assert list(result.columns) == list(table.columns) + ["week"]
    pd.testing.assert_frame_equal(result.drop(columns="week"), table)


def test_add_week_index_keeps_row_order_and_index():
    result = add_week_index(payment_table(), WEEK_ZERO)
    assert result.index.tolist() == [7, 3, 5]
    assert result["customer"].tolist() == ["A", "B", "C"]


def test_add_week_index_week_dtype_is_int64():
    assert add_week_index(payment_table(), WEEK_ZERO)["week"].dtype == "int64"


def test_add_week_index_does_not_modify_the_input():
    table = payment_table()
    before = table.copy()
    add_week_index(table, WEEK_ZERO)
    assert "week" not in table.columns
    pd.testing.assert_frame_equal(table, before)


def test_add_week_index_accepts_date_objects():
    table = pd.DataFrame({"document_date": [date(2025, 6, 2), date(2025, 6, 9)]})
    assert add_week_index(table, WEEK_ZERO)["week"].tolist() == [0, 1]


def test_add_week_index_on_empty_table_gives_empty_week_column():
    table = pd.DataFrame({"customer": pd.Series([], dtype=object),
                          "document_date": pd.Series([], dtype="datetime64[ns]")})
    result = add_week_index(table, WEEK_ZERO)
    assert "week" in result.columns
    assert len(result) == 0
    assert result["week"].dtype == "int64"


def test_add_week_index_raises_if_week_zero_not_a_monday():
    with pytest.raises(ValueError, match="Monday"):
        add_week_index(payment_table(), date(2025, 6, 4))


# --- the synthetic corpus --------------------------------------------------------------

@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason="synthetic corpus not installed")
def test_first_synthetic_file_is_week_0():
    files = list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT)
    first = read_one_file(files[min(files)], SYNTHETIC_LAYOUT)
    assert len(first) > 0
    assert set(add_week_index(first, WEEK_ZERO)["week"]) == {0}


@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason="synthetic corpus not installed")
def test_last_synthetic_file_is_week_64():
    files = list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT)
    last = read_one_file(files[max(files)], SYNTHETIC_LAYOUT)
    assert len(last) > 0
    assert set(add_week_index(last, WEEK_ZERO)["week"]) == {64}


# --- week_calendar ---------------------------------------------------------------------
# A holiday-free stretch: Monday 2026-02-02 to Sunday 2026-03-15 has no public holiday.

CALENDAR_ZERO = date(2026, 2, 2)  # a Monday
CALENDAR_COLUMNS = ["week", "monday", "business_days", "days_with_file", "missing_days",
                    "is_missing"]


def weekdays(monday):
    """Monday to Friday of the week starting on `monday`."""
    return [monday + timedelta(days=n) for n in range(5)]


def row(calendar, week):
    """The calendar row of one week, as a dict."""
    rows = calendar[calendar["week"] == week]
    assert len(rows) == 1
    return rows.iloc[0].to_dict()


def three_weeks_with_a_gap():
    """Files Mon-Fri in weeks 0 and 2, none in week 1."""
    return weekdays(CALENDAR_ZERO) + weekdays(CALENDAR_ZERO + timedelta(weeks=2))


def test_calendar_columns_are_exactly_these():
    calendar = week_calendar(weekdays(CALENDAR_ZERO), CALENDAR_ZERO, HOLIDAYS)
    assert list(calendar.columns) == CALENDAR_COLUMNS


def test_calendar_full_week_is_complete():
    calendar = week_calendar(weekdays(CALENDAR_ZERO), CALENDAR_ZERO, HOLIDAYS)
    assert len(calendar) == 1
    assert row(calendar, 0) == {"week": 0, "monday": CALENDAR_ZERO, "business_days": 5,
                                "days_with_file": 5, "missing_days": 0, "is_missing": False}


def test_calendar_monday_is_a_date_object():
    calendar = week_calendar(three_weeks_with_a_gap(), CALENDAR_ZERO, HOLIDAYS)
    for monday in calendar["monday"]:
        assert type(monday) is date
        assert monday.weekday() == 0


def test_calendar_mondays_are_seven_days_apart():
    calendar = week_calendar(three_weeks_with_a_gap(), CALENDAR_ZERO, HOLIDAYS)
    assert calendar["monday"].tolist() == [CALENDAR_ZERO + timedelta(weeks=n) for n in range(3)]


def test_calendar_weeks_are_consecutive_one_row_each():
    calendar = week_calendar(three_weeks_with_a_gap(), CALENDAR_ZERO, HOLIDAYS)
    assert calendar["week"].tolist() == [0, 1, 2]


def test_calendar_week_without_any_file_is_kept_and_missing():
    calendar = week_calendar(three_weeks_with_a_gap(), CALENDAR_ZERO, HOLIDAYS)
    gap = row(calendar, 1)
    assert gap["business_days"] == 5
    assert gap["days_with_file"] == 0
    assert gap["missing_days"] == 5
    assert gap["is_missing"]


def test_calendar_weeks_around_the_gap_are_complete():
    calendar = week_calendar(three_weeks_with_a_gap(), CALENDAR_ZERO, HOLIDAYS)
    assert not row(calendar, 0)["is_missing"]
    assert not row(calendar, 2)["is_missing"]


def test_calendar_one_missing_weekday_makes_the_week_missing():
    days = weekdays(CALENDAR_ZERO)
    days.remove(date(2026, 2, 4))  # Wednesday
    days.append(date(2026, 2, 9))  # next Monday, so week 0 is not the last week
    first = row(week_calendar(days, CALENDAR_ZERO, HOLIDAYS), 0)
    assert first["days_with_file"] == 4
    assert first["missing_days"] == 1
    assert first["is_missing"]


@pytest.mark.parametrize("missing", weekdays(date(2026, 2, 9)))
def test_calendar_any_single_missing_weekday_makes_the_week_missing(missing):
    days = [d for d in weekdays(date(2026, 2, 9)) if d != missing] + [date(2026, 2, 16)]
    week = row(week_calendar(days, date(2026, 2, 9), HOLIDAYS), 0)
    assert (week["missing_days"], week["is_missing"]) == (1, True)


def test_calendar_missing_plus_with_file_equals_business_days():
    days = [date(2026, 2, 2), date(2026, 2, 5), date(2026, 2, 10), date(2026, 2, 27)]
    calendar = week_calendar(days, CALENDAR_ZERO, HOLIDAYS)
    assert (calendar["days_with_file"] + calendar["missing_days"]
            == calendar["business_days"]).all()
    assert (calendar["is_missing"] == (calendar["missing_days"] > 0)).all()


# --- week_calendar: holidays -----------------------------------------------------------

def test_calendar_holiday_weekday_without_file_is_not_missing():
    # 23 April 2026 is a Thursday and a public holiday.
    monday = date(2026, 4, 20)
    days = [d for d in weekdays(monday) if d != date(2026, 4, 23)]
    week = row(week_calendar(days, monday, HOLIDAYS), 0)
    assert week["business_days"] == 4
    assert week["days_with_file"] == 4
    assert week["missing_days"] == 0
    assert not week["is_missing"]


def test_calendar_kurban_bayrami_week_has_two_business_days():
    # 27, 28, 29 May 2026 (Wed-Fri) are listed as Kurban Bayrami.
    monday = date(2026, 5, 25)
    week = row(week_calendar([date(2026, 5, 25), date(2026, 5, 26)], monday, HOLIDAYS), 0)
    assert week["business_days"] == 2
    assert week["days_with_file"] == 2
    assert not week["is_missing"]


def test_calendar_bayram_file_does_not_cover_a_missing_business_day():
    monday = date(2026, 5, 25)
    week = row(week_calendar([date(2026, 5, 25), date(2026, 5, 29)], monday, HOLIDAYS), 0)
    # 29 May is a holiday: it does not count and does not cover the missing Tuesday.
    assert week["business_days"] == 2
    assert week["days_with_file"] == 1
    assert week["missing_days"] == 1
    assert week["is_missing"]


def test_calendar_fixed_holiday_on_a_friday():
    # 1 May 2026 is a Friday and a public holiday.
    monday = date(2026, 4, 27)
    week = row(week_calendar(weekdays(monday)[:4], monday, HOLIDAYS), 0)
    assert week["business_days"] == 4
    assert not week["is_missing"]


# --- week_calendar: the holiday table it is given ---------------------------------------

def test_calendar_uses_the_fixed_holidays_of_the_table_it_is_given():
    # A table with one more fixed holiday, Wednesday 4 February: that day is no longer a
    # business day, so the week without a file on it is complete.
    table = HolidayTable(fixed={**HOLIDAYS.fixed, (2, 4): "Test Day"}, bayrams=HOLIDAYS.bayrams)
    days = [d for d in weekdays(CALENDAR_ZERO) if d != date(2026, 2, 4)]
    assert row(week_calendar(days, CALENDAR_ZERO, HOLIDAYS), 0)["is_missing"]
    week = row(week_calendar(days, CALENDAR_ZERO, table), 0)
    assert week["business_days"] == 4
    assert not week["is_missing"]


def test_calendar_uses_the_bayrams_of_the_table_it_is_given():
    # Move Kurban 2026 to Mon 9 - Thu 12 February: four business days fewer that week.
    bayrams = tuple(entry for entry in HOLIDAYS.bayrams
                    if not (entry[0].year == 2026 and entry[2] == "Kurban Bayramı"))
    table = HolidayTable(fixed=HOLIDAYS.fixed,
                         bayrams=bayrams + ((date(2026, 2, 9), date(2026, 2, 12),
                                             "Kurban Bayramı"),))
    monday = date(2026, 2, 9)
    week = row(week_calendar([date(2026, 2, 13)], monday, table), 0)
    assert week["business_days"] == 1
    assert not week["is_missing"]


def test_calendar_with_a_table_missing_a_bayram_raises_naming_it():
    bayrams = tuple(entry for entry in HOLIDAYS.bayrams
                    if not (entry[0].year == 2026 and entry[2] == "Kurban Bayramı"))
    table = HolidayTable(fixed=HOLIDAYS.fixed, bayrams=bayrams)
    with pytest.raises(ValueError, match="Kurban Bayramı 2026"):
        week_calendar(weekdays(CALENDAR_ZERO), CALENDAR_ZERO, table)


def test_calendar_needs_the_holiday_table():
    with pytest.raises(TypeError):
        week_calendar(weekdays(CALENDAR_ZERO), CALENDAR_ZERO)


def test_calendar_does_not_modify_the_holiday_table():
    table = HolidayTable(fixed=dict(HOLIDAYS.fixed), bayrams=HOLIDAYS.bayrams)
    week_calendar(weekdays(date(2026, 4, 20)), date(2026, 4, 20), table)
    assert table == HOLIDAYS


# --- week_calendar: files on weekends and holidays -------------------------------------

def test_calendar_saturday_file_is_not_counted():
    days = weekdays(CALENDAR_ZERO) + [date(2026, 2, 7)]  # Saturday
    week = row(week_calendar(days, CALENDAR_ZERO, HOLIDAYS), 0)
    assert week["days_with_file"] == 5
    assert week["business_days"] == 5


def test_calendar_saturday_file_does_not_hide_a_missing_weekday():
    days = [d for d in weekdays(CALENDAR_ZERO) if d != date(2026, 2, 3)] + [date(2026, 2, 7)]
    week = row(week_calendar(days, CALENDAR_ZERO, HOLIDAYS), 0)
    assert week["days_with_file"] == 4
    assert week["missing_days"] == 1
    assert week["is_missing"]


def test_calendar_sunday_file_does_not_hide_a_missing_weekday():
    days = [d for d in weekdays(CALENDAR_ZERO) if d != date(2026, 2, 6)] + [date(2026, 2, 8)]
    week = row(week_calendar(days, CALENDAR_ZERO, HOLIDAYS), 0)
    assert (week["days_with_file"], week["missing_days"], week["is_missing"]) == (4, 1, True)


def test_calendar_holiday_file_is_not_counted():
    monday = date(2026, 4, 20)
    week = row(week_calendar(weekdays(monday), monday, HOLIDAYS), 0)  # includes Thursday 23 April
    assert week["business_days"] == 4
    assert week["days_with_file"] == 4
    assert week["missing_days"] == 0


def test_calendar_holiday_file_does_not_hide_a_missing_weekday():
    monday = date(2026, 4, 20)
    days = [d for d in weekdays(monday) if d != date(2026, 4, 21)]  # keeps 23 April
    week = row(week_calendar(days, monday, HOLIDAYS), 0)
    assert week["days_with_file"] == 3
    assert week["missing_days"] == 1
    assert week["is_missing"]


# --- week_calendar: the last week ------------------------------------------------------

def test_calendar_files_ending_on_wednesday_leave_thursday_and_friday_missing():
    days = weekdays(CALENDAR_ZERO) + weekdays(date(2026, 2, 9))[:3]
    calendar = week_calendar(days, CALENDAR_ZERO, HOLIDAYS)
    assert len(calendar) == 2
    last = row(calendar, 1)
    assert last["business_days"] == 5
    assert last["days_with_file"] == 3
    assert last["missing_days"] == 2
    assert last["is_missing"]


def test_calendar_files_ending_on_friday_leave_last_week_complete():
    days = weekdays(CALENDAR_ZERO) + weekdays(date(2026, 2, 9))
    calendar = week_calendar(days, CALENDAR_ZERO, HOLIDAYS)
    assert len(calendar) == 2
    assert not row(calendar, 1)["is_missing"]


def test_calendar_ends_at_the_week_of_the_last_file():
    days = [CALENDAR_ZERO, date(2026, 2, 25)]  # a Wednesday in week 3
    assert week_calendar(days, CALENDAR_ZERO, HOLIDAYS)["week"].tolist() == [0, 1, 2, 3]


def test_calendar_last_file_on_a_sunday_ends_in_that_week():
    days = [CALENDAR_ZERO, date(2026, 2, 15)]  # Sunday of week 1
    assert week_calendar(days, CALENDAR_ZERO, HOLIDAYS)["week"].tolist() == [0, 1]


def test_calendar_single_day_gives_one_row():
    calendar = week_calendar([CALENDAR_ZERO], CALENDAR_ZERO, HOLIDAYS)
    assert len(calendar) == 1
    assert row(calendar, 0) == {"week": 0, "monday": CALENDAR_ZERO, "business_days": 5,
                                "days_with_file": 1, "missing_days": 4, "is_missing": True}


def test_calendar_single_late_day_still_starts_at_week_zero():
    calendar = week_calendar([date(2026, 2, 18)], CALENDAR_ZERO, HOLIDAYS)
    assert calendar["week"].tolist() == [0, 1, 2]
    assert calendar["days_with_file"].tolist() == [0, 0, 1]


# --- week_calendar: errors -------------------------------------------------------------

def test_calendar_empty_list_raises():
    with pytest.raises(ValueError, match="no days with a file"):
        week_calendar([], CALENDAR_ZERO, HOLIDAYS)


def test_calendar_file_before_week_zero_raises_naming_the_date():
    with pytest.raises(ValueError, match="2026-01-30"):
        week_calendar([date(2026, 1, 30), CALENDAR_ZERO], CALENDAR_ZERO, HOLIDAYS)


def test_calendar_week_zero_not_a_monday_raises():
    with pytest.raises(ValueError, match="Monday"):
        week_calendar([date(2026, 2, 4)], date(2026, 2, 3), HOLIDAYS)


def test_calendar_range_reaching_a_year_without_bayrams_raises():
    with pytest.raises(ValueError, match="2027"):
        week_calendar([CALENDAR_ZERO, date(2027, 1, 4)], CALENDAR_ZERO, HOLIDAYS)


def test_calendar_last_week_ending_in_a_year_without_bayrams_raises():
    # The last file is Thursday 31 December 2026; its week ends on Sunday 3 January 2027.
    with pytest.raises(ValueError, match="2027"):
        week_calendar([date(2026, 12, 31)], date(2026, 12, 28), HOLIDAYS)


# --- week_calendar: input order --------------------------------------------------------

def test_calendar_unsorted_input_gives_the_same_result():
    days = three_weeks_with_a_gap()
    shuffled = days[::-1][3:] + days[::-1][:3]
    pd.testing.assert_frame_equal(week_calendar(shuffled, CALENDAR_ZERO, HOLIDAYS),
                                  week_calendar(days, CALENDAR_ZERO, HOLIDAYS))


def test_calendar_duplicate_days_give_the_same_result():
    days = three_weeks_with_a_gap()
    pd.testing.assert_frame_equal(week_calendar(days + days[:4], CALENDAR_ZERO, HOLIDAYS),
                                  week_calendar(days, CALENDAR_ZERO, HOLIDAYS))


def test_calendar_does_not_modify_the_input_list():
    days = [date(2026, 2, 6), CALENDAR_ZERO]
    week_calendar(days, CALENDAR_ZERO, HOLIDAYS)
    assert days == [date(2026, 2, 6), CALENDAR_ZERO]


# --- week_calendar: dtypes -------------------------------------------------------------

@pytest.mark.parametrize("column", ["week", "business_days", "days_with_file", "missing_days"])
def test_calendar_count_columns_are_int64(column):
    calendar = week_calendar(three_weeks_with_a_gap(), CALENDAR_ZERO, HOLIDAYS)
    assert calendar[column].dtype == "int64"


def test_calendar_is_missing_is_bool():
    calendar = week_calendar(three_weeks_with_a_gap(), CALENDAR_ZERO, HOLIDAYS)
    assert calendar["is_missing"].dtype == bool


def test_calendar_index_is_a_plain_range():
    calendar = week_calendar(three_weeks_with_a_gap(), CALENDAR_ZERO, HOLIDAYS)
    assert calendar.index.tolist() == [0, 1, 2]


# --- week_calendar: the synthetic corpus -----------------------------------------------

@pytest.fixture(scope="module")
def synthetic_calendar():
    if not SYNTHETIC_CORPUS.is_dir():
        pytest.skip("synthetic corpus not installed")
    days = list(list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT))
    return week_calendar(days, WEEK_ZERO, HOLIDAYS)


def test_synthetic_calendar_has_65_weeks(synthetic_calendar):
    assert synthetic_calendar["week"].tolist() == list(range(65))


def test_synthetic_calendar_missing_weeks(synthetic_calendar):
    missing = synthetic_calendar[synthetic_calendar["is_missing"]]
    assert missing["week"].tolist() == [14, 28, 35, 40, 46, 52]
    assert missing["missing_days"].tolist() == [1] * 6


def test_synthetic_calendar_totals(synthetic_calendar):
    assert synthetic_calendar["business_days"].sum() == 312
    assert synthetic_calendar["days_with_file"].sum() == 306


# --- week_calendar: datetimes and Timestamps -------------------------------------------
# Every input day is made a plain date first, so a time of day or a Timestamp cannot make a
# day with a file look like a missing day.

def at_half_past_two(day):
    """The same day as a datetime at 14:30."""
    return datetime(day.year, day.month, day.day, 14, 30)


def test_calendar_datetimes_with_time_give_the_same_frame_as_dates():
    days = three_weeks_with_a_gap()
    pd.testing.assert_frame_equal(week_calendar([at_half_past_two(d) for d in days],
                                                CALENDAR_ZERO, HOLIDAYS),
                                  week_calendar(days, CALENDAR_ZERO, HOLIDAYS))


def test_calendar_timestamps_give_the_same_frame_as_dates():
    days = three_weeks_with_a_gap()
    stamps = [pd.Timestamp(d) for d in days]
    pd.testing.assert_frame_equal(week_calendar(stamps, CALENDAR_ZERO, HOLIDAYS),
                                  week_calendar(days, CALENDAR_ZERO, HOLIDAYS))


def test_calendar_timestamps_with_time_give_the_same_frame_as_dates():
    days = three_weeks_with_a_gap()
    stamps = [pd.Timestamp(at_half_past_two(d)) for d in days]
    pd.testing.assert_frame_equal(week_calendar(stamps, CALENDAR_ZERO, HOLIDAYS),
                                  week_calendar(days, CALENDAR_ZERO, HOLIDAYS))


def test_calendar_mixed_dates_datetimes_and_timestamps_give_the_same_frame():
    days = three_weeks_with_a_gap()
    converters = [lambda d: d, at_half_past_two, pd.Timestamp]
    mixed = [converters[n % 3](d) for n, d in enumerate(days)]
    assert {type(d) for d in mixed} == {date, datetime, pd.Timestamp}
    pd.testing.assert_frame_equal(week_calendar(mixed, CALENDAR_ZERO, HOLIDAYS),
                                  week_calendar(days, CALENDAR_ZERO, HOLIDAYS))


def test_calendar_wednesday_as_a_timestamp_is_counted_not_missing():
    days = weekdays(CALENDAR_ZERO)
    days[2] = pd.Timestamp(2026, 2, 4)  # Wednesday
    days.append(date(2026, 2, 9))  # next Monday, so week 0 is not the last week
    week = row(week_calendar(days, CALENDAR_ZERO, HOLIDAYS), 0)
    assert week["days_with_file"] == 5
    assert week["missing_days"] == 0
    assert not week["is_missing"]


def test_calendar_wednesday_as_a_datetime_with_time_is_counted_not_missing():
    days = weekdays(CALENDAR_ZERO)
    days[2] = datetime(2026, 2, 4, 14, 30)
    week = row(week_calendar(days, CALENDAR_ZERO, HOLIDAYS), 0)
    assert (week["days_with_file"], week["missing_days"], week["is_missing"]) == (5, 0, False)


def test_calendar_saturday_timestamp_still_does_not_hide_a_missing_weekday():
    days = [d for d in weekdays(CALENDAR_ZERO) if d != date(2026, 2, 3)]
    days.append(pd.Timestamp("2026-02-07 09:00"))  # Saturday
    week = row(week_calendar(days, CALENDAR_ZERO, HOLIDAYS), 0)
    assert (week["days_with_file"], week["missing_days"], week["is_missing"]) == (4, 1, True)


def test_calendar_last_day_as_a_late_timestamp_ends_in_its_own_week():
    # 23:59 on Sunday of week 1 is still week 1; no extra week is added.
    days = [CALENDAR_ZERO, pd.Timestamp("2026-02-15 23:59:59")]
    assert week_calendar(days, CALENDAR_ZERO, HOLIDAYS)["week"].tolist() == [0, 1]


def test_calendar_timestamp_before_week_zero_raises_naming_the_date():
    with pytest.raises(ValueError, match="2026-01-30"):
        week_calendar([pd.Timestamp("2026-01-30 10:00"), CALENDAR_ZERO], CALENDAR_ZERO, HOLIDAYS)


def test_calendar_week_zero_as_midnight_datetime_gives_the_same_frame():
    days = three_weeks_with_a_gap()
    pd.testing.assert_frame_equal(week_calendar(days, datetime(2026, 2, 2), HOLIDAYS),
                                  week_calendar(days, CALENDAR_ZERO, HOLIDAYS))


def test_calendar_week_zero_as_timestamp_gives_the_same_frame():
    days = three_weeks_with_a_gap()
    pd.testing.assert_frame_equal(week_calendar(days, pd.Timestamp(CALENDAR_ZERO), HOLIDAYS),
                                  week_calendar(days, CALENDAR_ZERO, HOLIDAYS))


def test_calendar_week_zero_as_timestamp_with_time_gives_the_same_frame():
    days = three_weeks_with_a_gap()
    pd.testing.assert_frame_equal(week_calendar(days, pd.Timestamp("2026-02-02 14:30"), HOLIDAYS),
                                  week_calendar(days, CALENDAR_ZERO, HOLIDAYS))


@pytest.mark.parametrize("week_zero", [datetime(2026, 2, 2), pd.Timestamp(2026, 2, 2)],
                         ids=["datetime", "Timestamp"])
def test_calendar_monday_is_a_date_object_when_week_zero_is_not(week_zero):
    stamps = [pd.Timestamp(d) for d in three_weeks_with_a_gap()]
    calendar = week_calendar(stamps, week_zero, HOLIDAYS)
    for monday in calendar["monday"]:
        assert type(monday) is date
        assert monday.weekday() == 0
    assert calendar["monday"].iloc[0] == CALENDAR_ZERO


def test_calendar_week_zero_timestamp_not_a_monday_raises():
    with pytest.raises(ValueError, match="Monday"):
        week_calendar([date(2026, 2, 4)], pd.Timestamp(2026, 2, 3), HOLIDAYS)


def test_calendar_week_zero_datetime_not_a_monday_raises():
    with pytest.raises(ValueError, match="Monday"):
        week_calendar([date(2026, 2, 4)], datetime(2026, 2, 3, 14, 30), HOLIDAYS)


def test_calendar_does_not_modify_an_input_list_of_datetimes():
    days = [datetime(2026, 2, 6, 14, 30), pd.Timestamp("2026-02-02 09:15"), CALENDAR_ZERO]
    before = list(days)
    week_calendar(days, CALENDAR_ZERO, HOLIDAYS)
    assert days == before
    assert [type(d) for d in days] == [datetime, pd.Timestamp, date]
    assert days[0].hour == 14 and days[1].hour == 9
