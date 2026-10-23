"""The calendar of each target week: business days, payment-batch dates and holidays before,
and where the holidays fall around the week (`holiday_position_features`).

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


# The holiday encodings `holiday_position_features` can add, and the weekday names of "days".
HOLIDAY_ENCODINGS = ("drain", "days")
WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri")


def holiday_position_features(week_zero: date, first_week: int, last_week: int,
                              holidays: HolidayTable, encodings: tuple[str, ...],
                              drain_window: int, catch_up_window: int) -> pd.DataFrame:
    """Where the public holidays fall around each week from `first_week` to `last_week`: one
    row per week, a `week` column and the columns of each of `encodings`:

      "drain"   drain_days      business days of the week with a holiday 1 to `drain_window`
                                calendar days later: payments due then tend to move past the
                                holiday, often into the next week
                catch_up_days   business days of the week with a holiday 1 to
                                `catch_up_window` calendar days earlier: they receive the
                                moved payments
      "days"    prev_mon ... prev_fri, cur_mon ... cur_fri, next_mon ... next_fri
                                each weekday of the week before, the week itself and the week
                                after: 1 = business day, 0 = public holiday

    Only holidays on weekdays count: nobody pays on a weekend anyway, so a Saturday holiday
    moves nothing. A day without a daily file is not a holiday (it is missing data); only the
    holidays file is used, so the features are known in advance for any week.

    Raises ValueError if `week_zero` is not a Monday, `last_week` is before `first_week`, an
    encoding is unknown or repeated, a window is below 1, and (from `public_holidays`) if the
    days reach a year whose Bayrams are not in the holidays file.
    """
    week_zero = pd.Timestamp(week_zero).date()
    if week_zero.weekday() != 0:
        raise ValueError(f"week_zero {week_zero} is a {week_zero.strftime('%A')}, not a Monday")
    if last_week < first_week:
        raise ValueError(f"last_week {last_week} is before first_week {first_week}")
    unknown = [encoding for encoding in encodings if encoding not in HOLIDAY_ENCODINGS]
    if unknown:
        raise ValueError(f"unknown holiday encodings {unknown}; known: {list(HOLIDAY_ENCODINGS)}")
    if len(set(encodings)) != len(encodings):
        raise ValueError(f"holiday encodings {list(encodings)} name one more than once")
    for name, window in (("drain_window", drain_window), ("catch_up_window", catch_up_window)):
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"{name} is {window!r}, not a whole number of days, 1 or more")

    first_monday = week_zero + timedelta(days=7 * first_week)
    last_monday = week_zero + timedelta(days=7 * last_week)
    # Far enough on both sides for the week before / after and for both windows.
    start = first_monday - timedelta(days=max(7, catch_up_window))
    end = last_monday + timedelta(days=max(11, 4 + drain_window))
    holiday_weekdays = {day for day in public_holidays(start, end, holidays)
                        if day.weekday() < 5}

    rows = []
    for week in range(first_week, last_week + 1):
        monday = week_zero + timedelta(days=7 * week)
        row = {"week": week}
        if "drain" in encodings:
            business = [monday + timedelta(days=offset) for offset in range(5)
                        if monday + timedelta(days=offset) not in holiday_weekdays]
            row["drain_days"] = sum(
                any(day + timedelta(days=k) in holiday_weekdays
                    for k in range(1, drain_window + 1)) for day in business)
            row["catch_up_days"] = sum(
                any(day - timedelta(days=k) in holiday_weekdays
                    for k in range(1, catch_up_window + 1)) for day in business)
        if "days" in encodings:
            for label, weeks_away in (("prev", -1), ("cur", 0), ("next", 1)):
                for offset, name in enumerate(WEEKDAY_NAMES):
                    day = monday + timedelta(days=7 * weeks_away + offset)
                    row[f"{label}_{name}"] = 0 if day in holiday_weekdays else 1
        rows.append(row)
    return pd.DataFrame(rows).astype("int64")


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
