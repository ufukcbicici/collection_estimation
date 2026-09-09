"""Tests for the calendar block (section 9.2).

The decision this pins hardest: month end means the last BUSINESS day of the month, not
the last calendar day. Getting that wrong shifts the flag by a week roughly two months in
seven, onto exactly the weeks the effect is not in.
"""
import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from collection_estimation import config                            # noqa: E402
from collection_estimation.calendar_features import (               # noqa: E402
    HolidayCoverageError,
    build_calendar_features,
    business_days,
    check_holiday_coverage,
    covered_years,
    cross_check_business_days,
    holiday_dates,
)
from collection_estimation.ingest import read_corpus                # noqa: E402
from collection_estimation.panel import build_week_index            # noqa: E402


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return read_corpus(config.CORPUS_DIR)


@pytest.fixture(scope="module")
def weeks(corpus) -> pd.DataFrame:
    return build_week_index(corpus)


@pytest.fixture(scope="module")
def calendar(weeks) -> pd.DataFrame:
    return build_calendar_features(weeks)


def spine(start: str, n: int) -> pd.DataFrame:
    """A week spine starting on a given Monday."""
    starts = pd.date_range(start, periods=n, freq="7D")
    frame = pd.DataFrame({"week": range(n), "week_start": starts})
    frame["week_end"] = frame["week_start"] + pd.Timedelta(days=6)
    assert (frame["week_start"].dt.dayofweek == 0).all(), "spine must start on a Monday"
    return frame


# ---------------------------------------------------------------- holidays

def test_fixed_holidays_are_present():
    got = holiday_dates(date(2026, 1, 1), date(2026, 12, 31))
    for when in (date(2026, 1, 1), date(2026, 4, 23), date(2026, 5, 1),
                 date(2026, 5, 19), date(2026, 7, 15), date(2026, 8, 30),
                 date(2026, 10, 29)):
        assert when in got


def test_moving_holidays_are_present_and_span_several_days():
    got = holiday_dates(date(2026, 1, 1), date(2026, 12, 31))
    kurban = [d for d, name in got.items() if name == "Kurban Bayramı"]
    assert len(kurban) == 4, "Kurban Bayramı runs four days in 2026"


def test_business_days_exclude_weekends_and_holidays():
    got = business_days(date(2026, 4, 20), date(2026, 4, 26))   # Mon .. Sun
    assert date(2026, 4, 23) not in got, "23 April is a national holiday"
    assert date(2026, 4, 25) not in got and date(2026, 4, 26) not in got
    assert len(got) == 4


# ---------------------------------------------------------------- month ends

def test_month_end_follows_the_last_business_day_not_the_calendar_day():
    """May 2026: the 31st is a Sunday AND the 27th-30th are Kurban Bayramı.

    So the last business day of May is Tuesday the 26th, five days before the calendar
    month ends. A calendar-day rule would flag the week of 1 June — the week after the
    batch actually cleared. This is the single decision most worth pinning here.
    """
    assert date(2026, 5, 31).weekday() == 6, "31 May 2026 is a Sunday"
    holidays = holiday_dates(date(2026, 5, 1), date(2026, 5, 31))
    assert date(2026, 5, 29) in holidays, "29 May 2026 is Kurban Bayramı, not a workday"

    last_may = business_days(date(2026, 5, 1), date(2026, 5, 31))[-1]
    assert last_may == date(2026, 5, 26)

    table = build_calendar_features(spine("2026-05-25", 3))
    flags = dict(zip(table["week"], table["has_month_end"]))
    assert flags[0] == 1, "the week containing Tuesday 26 May carries the flag"
    assert flags[1] == 0, "the week of 1 June must NOT carry May's month end"


def test_month_end_flag_sits_in_the_week_of_the_last_business_day():
    """August 2026 ends on a Monday, so the flag belongs to the FOLLOWING week.

    The mirror of the case above: here the last business day of the month is the 31st,
    which starts a new week. The rule is the same — follow the last business day — but it
    lands the other side of a week boundary.
    """
    assert date(2026, 8, 31).weekday() == 0, "31 August 2026 is a Monday"
    august = build_calendar_features(spine("2026-08-24", 2))
    assert august["has_month_end"].iloc[0] == 0
    assert august["has_month_end"].iloc[1] == 1


def test_quarter_ends_are_a_subset_of_month_ends(calendar):
    assert (calendar["has_quarter_end"] <= calendar["has_month_end"]).all()


def test_quarter_ends_fall_in_the_right_months(calendar, weeks):
    marked = calendar[calendar["has_quarter_end"] == 1]["week"]
    months = weeks.set_index("week").loc[marked, "week_start"].dt.month
    # The last business day of a quarter can sit in the following calendar month only if
    # the quarter's final days are all non-working, which does not occur here.
    assert set(months) <= {3, 6, 9, 12}


def test_there_is_roughly_one_month_end_per_month(calendar, weeks):
    span_months = len(weeks) / (52 / 12)
    assert abs(calendar["has_month_end"].sum() - span_months) < 2


# ---------------------------------------------------------------- shape

def test_one_row_per_week(calendar, weeks):
    assert len(calendar) == len(weeks)
    assert list(calendar["week"]) == list(weeks["week"])


def test_business_days_are_between_zero_and_five(calendar):
    assert calendar["n_business_days"].between(0, 5).all()


def test_a_bayram_week_loses_business_days(calendar, weeks):
    """Kurban Bayramı 2026 runs 27-30 May, so that week is short."""
    target = weeks[weeks["week_start"] == pd.Timestamp("2026-05-25")]["week"]
    assert len(target) == 1
    row = calendar[calendar["week"] == target.iloc[0]].iloc[0]
    assert row["n_business_days"] < 5
    assert row["n_holidays"] > 0


def test_weeks_to_month_end_is_zero_on_the_month_end_week(calendar):
    on_end = calendar[calendar["has_month_end"] == 1]
    assert (on_end["weeks_to_month_end"] == 0).all()


def test_weeks_to_month_end_counts_down(calendar):
    """Falls by one each week, then resets to the gap before the NEXT month end.

    The reset is why this is not simply `later == earlier - 1`: after a month-end week
    the value is 0, and the following week starts counting toward a month end three or
    four weeks away.
    """
    values = calendar["weeks_to_month_end"].dropna().tolist()
    for earlier, later in zip(values, values[1:]):
        assert later == earlier - 1 or earlier == 0, (
            f"{earlier} -> {later}: the distance must fall by one, or reset after a "
            f"month-end week")


def test_no_customer_information_leaks_in(calendar):
    assert "customer" not in calendar.columns
    assert "y" not in calendar.columns and "z" not in calendar.columns


def test_dtypes_are_compact(calendar):
    assert calendar["n_business_days"].dtype == "int8"
    assert calendar["has_month_end"].dtype == "int8"


# ---------------------------------------------------------------- the cross-check

def test_cross_check_reports_the_corpus_disagreements(weeks, corpus):
    """The declared table against the days the corpus actually has files for.

    This is the number that says how much the hardcoded holiday table is worth. The 6
    deliberately-absent files should show up here as `expected_working_no_data` — they
    are missing observations, not holidays, and conflating them is what EY's instruction
    8 warns against.
    """
    report = cross_check_business_days(weeks, corpus)

    holiday_with_data = report[report["kind"] == "holiday_with_data"]
    assert holiday_with_data.empty, (
        f"our table calls these holidays but the corpus has collections on them:\n"
        f"{holiday_with_data}")

    no_data = report[report["kind"] == "expected_working_no_data"]
    assert len(no_data) <= 10, (
        f"more working days without data than the corpus's 6 deliberately missing "
        f"files — the holiday table may be wrong:\n{no_data}")


def test_cross_check_flags_a_wrong_holiday(weeks, corpus):
    """A day we call a holiday that has collections must be reported, not ignored."""
    fake = corpus.head(1).copy()
    fake["document_date"] = pd.Timestamp("2026-04-23")     # a national holiday
    report = cross_check_business_days(weeks, pd.concat([corpus, fake]))
    flagged = report[report["kind"] == "holiday_with_data"]["date"].tolist()
    assert date(2026, 4, 23) in flagged

# ---------------------------------------------------------------- coverage guard

def test_declared_coverage_matches_the_table():
    assert covered_years() == {2025, 2026}


def test_uncovered_year_raises_rather_than_silently_dropping_bayram():
    """The whole point of the guard.

    Without it, `holiday_dates` for 2027 returns no Bayram at all: a four-day religious
    holiday is counted as five ordinary business days, every model reads a collapsed week
    as a normal one, and nothing errors. A wrong answer that looks right is worse than a
    refusal.
    """
    with pytest.raises(HolidayCoverageError, match="2027"):
        holiday_dates(date(2027, 1, 1), date(2027, 12, 31))

    with pytest.raises(HolidayCoverageError):
        business_days(date(2026, 6, 1), date(2027, 6, 1))


def test_the_error_says_what_to_do():
    with pytest.raises(HolidayCoverageError) as caught:
        holiday_dates(date(2028, 1, 1), date(2028, 2, 1))
    message = str(caught.value)
    assert "MOVING_HOLIDAYS" in message
    assert "2025" in message and "2026" in message, "it must say what IS covered"


def test_strict_false_is_an_explicit_escape_hatch():
    """Available, but the caller has to ask for it and the counts are then wrong."""
    got = holiday_dates(date(2027, 1, 1), date(2027, 12, 31), strict=False)
    assert date(2027, 1, 1) in got, "fixed national holidays still resolve"
    assert not any(name.endswith("Bayramı") for name in got.values()), (
        "and the religious holidays are silently absent, which is the danger")


def test_the_corpus_range_is_covered(weeks):
    """No guard should fire on the data we actually have."""
    first = weeks["week_start"].iloc[0].date()
    last = weeks["week_end"].iloc[-1].date()
    check_holiday_coverage(first, last)
    assert build_calendar_features(weeks) is not None
