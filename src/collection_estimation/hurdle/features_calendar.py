"""The calendar of each target week: business days, payment-batch dates and holidays before.

These features need no data, only the calendar and the holidays file, so they also cover
future weeks that have no daily files yet. Holidays are announced in advance (the Bayram
dates years ahead), so the calendar of a target week is known on the forecast date.
"""
from datetime import date, timedelta

import pandas as pd

from .holidays import HolidayTable, public_holidays

# The columns of the table `calendar_features` returns, in order.
CALENDAR_COLUMNS = ["week", "business_days", "has_month_end", "has_quarter_end",
                    "has_mid_month", "holidays_week_before"]

QUARTER_END_MONTHS = (3, 6, 9, 12)
MID_MONTH_DAY = 15


def calendar_features(week_zero: date, first_week: int, last_week: int,
                      holidays: HolidayTable) -> pd.DataFrame:
    """One row per week from `first_week` to `last_week` (Monday to Sunday, numbered from
    `week_zero`), with:
      business_days         weekdays that are not public holidays (0 to 5)
      has_month_end         the week holds the last business day of a month
      has_quarter_end       the same, for March, June, September and December
      has_mid_month         the week holds the 15th of a month
      holidays_week_before  weekdays that were public holidays in the week before (0 to 5)

    Raises ValueError if `week_zero` is not a Monday, if `last_week` is before `first_week`,
    or (from `public_holidays`) if the weeks reach a year whose Bayrams are not in the
    holidays file.
    """
    week_zero = pd.Timestamp(week_zero).date()
    # Every week runs Monday to Sunday from week_zero: another weekday would shift all weeks.
    if week_zero.weekday() != 0:
        raise ValueError(f"week_zero {week_zero} is a {week_zero.strftime('%A')}, not a Monday")
    if last_week < first_week:
        raise ValueError(f"last_week {last_week} is before first_week {first_week}")
    first_monday = week_zero + timedelta(days=7 * first_week)
    last_sunday = week_zero + timedelta(days=7 * last_week + 6)
    # From the week before the first week (for holidays_week_before) to the end of the last
    # week's month (to know whether a later business day of that month exists).
    start = first_monday - timedelta(days=7)
    end = last_day_of_month(last_sunday)
    holiday_days = public_holidays(start, end, holidays)

    def is_business_day(day: date) -> bool:
        return day.weekday() < 5 and day not in holiday_days

    month_ends = last_business_days(start, end, is_business_day)

    rows = []
    for week in range(first_week, last_week + 1):
        monday = week_zero + timedelta(days=7 * week)
        days = [monday + timedelta(days=offset) for offset in range(7)]
        weekdays_before = [monday - timedelta(days=7 - offset) for offset in range(5)]
        week_month_ends = [day for day in days if day in month_ends]
        rows.append({
            "week": week,
            "business_days": sum(is_business_day(day) for day in days),
            "has_month_end": bool(week_month_ends),
            "has_quarter_end": any(day.month in QUARTER_END_MONTHS for day in week_month_ends),
            "has_mid_month": any(day.day == MID_MONTH_DAY for day in days),
            "holidays_week_before": sum(day in holiday_days for day in weekdays_before),
        })
    table = pd.DataFrame(rows, columns=CALENDAR_COLUMNS)
    for column in ("week", "business_days", "holidays_week_before"):
        table[column] = table[column].astype("int64")
    for column in ("has_month_end", "has_quarter_end", "has_mid_month"):
        table[column] = table[column].astype(bool)
    return table


def last_business_days(start: date, end: date, is_business_day) -> set[date]:
    """The last business day of each month, among the days from `start` to `end`."""
    last_in_month = {}  # (year, month) -> the latest business day seen so far
    day = start
    while day <= end:
        if is_business_day(day):
            last_in_month[(day.year, day.month)] = day
        day += timedelta(days=1)
    return set(last_in_month.values())


def last_day_of_month(day: date) -> date:
    """The last calendar day of `day`'s month."""
    first_of_next = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return first_of_next - timedelta(days=1)
