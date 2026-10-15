"""Weekly numbers per key: week index, the week calendar, and weekly amounts.

Weeks run Monday to Sunday and are numbered from a fixed Monday, "week zero". Plain week
numbers make "week t + h" simple arithmetic for every later step.
"""
from datetime import date, timedelta

import pandas as pd

from .holidays import HolidayTable, public_holidays


def week_numbers(dates: pd.Series, week_zero: date) -> pd.Series:
    """The week number of each date: 0 for the week starting on `week_zero`, 1 for the next.

    Dates before `week_zero` get negative numbers (e.g. old invoices in the AR ledger).
    Raises ValueError if `week_zero` is not a Monday.
    """
    if week_zero.weekday() != 0:
        raise ValueError(f"week zero must be a Monday; {week_zero} is a {week_zero:%A}")
    dates = pd.to_datetime(dates)
    mondays = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    days_since_week_zero = (mondays - pd.Timestamp(week_zero)).dt.days
    return (days_since_week_zero // 7).astype("int64")


def add_week_index(collections: pd.DataFrame, week_zero: date) -> pd.DataFrame:
    """The payment lines with a new column `week`, the week number of `document_date`."""
    with_weeks = collections.copy()
    with_weeks["week"] = week_numbers(with_weeks["document_date"], week_zero)
    return with_weeks


def week_calendar(days_with_file: list[date], week_zero: date, holidays: HolidayTable,
                  filled_days: list[date] = ()) -> pd.DataFrame:
    """One row per week, from week zero to the week of the last file.

    Columns: week, monday, business_days (weekdays that are not public holidays),
    days_with_file (business days that have a daily file), filled_days (business days
    without a file whose payments were filled from the ledger, see `fill_missing_days`),
    missing_days (business days with neither) and is_missing.

    A week is missing if ANY of its business days has neither a file nor filled payments:
    one missing day can turn a customer's payment into "did not pay", so the whole week is
    unknown. A file on a weekend or a holiday does not count, so it cannot hide a missing
    business day.
    """
    if not days_with_file:
        raise ValueError("no days with a file")
    # Make every input a plain date: a datetime or Timestamp would not match the plain dates
    # of the day loop below, and its day would quietly count as missing.
    days_with_file = [pd.Timestamp(day).date() for day in days_with_file]
    filled = {pd.Timestamp(day).date() for day in filled_days}
    week_zero = pd.Timestamp(week_zero).date()
    if min(days_with_file) < week_zero:
        raise ValueError(f"a file is dated {min(days_with_file)}, before week zero {week_zero}")

    last_day = max(days_with_file)
    last_sunday = last_day + timedelta(days=6 - last_day.weekday())
    holiday_days = public_holidays(week_zero, last_sunday, holidays)
    has_file = set(days_with_file)

    # One row per calendar day; a few hundred days, so a plain loop is clear and fast enough.
    rows = []
    day = week_zero
    while day <= last_sunday:
        is_business_day = day.weekday() < 5 and day not in holiday_days
        with_file = is_business_day and day in has_file
        filled_from_ledger = is_business_day and not with_file and day in filled
        rows.append({
            "day": day,
            "is_business_day": is_business_day,
            "is_business_day_with_file": with_file,
            "is_filled_day": filled_from_ledger,
            "is_missing_day": is_business_day and not with_file and not filled_from_ledger,
        })
        day += timedelta(days=1)
    days = pd.DataFrame(rows)
    days["week"] = week_numbers(days["day"], week_zero)

    calendar = days.groupby("week").agg(
        monday=("day", "first"),
        business_days=("is_business_day", "sum"),
        days_with_file=("is_business_day_with_file", "sum"),
        filled_days=("is_filled_day", "sum"),
        missing_days=("is_missing_day", "sum"),
    ).reset_index()
    calendar["is_missing"] = calendar["missing_days"] > 0
    return calendar


def weekly_amounts(collections: pd.DataFrame) -> pd.DataFrame:
    """One row per key and week with at least one payment line.

    Columns: key, company_code, customer_id, week, gross_receipts (sum of positive TRY
    amounts; the forecast target), refunds (sum of negative TRY amounts, kept negative),
    unapplied_lines (lines with no SAP document number: cash not yet matched to an invoice)
    and lines. A key-week with only refunds or zero amounts gets gross_receipts = 0.
    Weeks without any line get no row; whether such a week is zero or unknown is decided
    later with the week calendar.
    """
    lines = collections.copy()
    lines["positive_amount"] = lines["amount_try"].clip(lower=0)
    lines["negative_amount"] = lines["amount_try"].clip(upper=0)
    lines["is_unapplied"] = lines["sap_document_no"].isna()

    weekly = lines.groupby(["key", "company_code", "customer_id", "week"], as_index=False).agg(
        gross_receipts=("positive_amount", "sum"),
        refunds=("negative_amount", "sum"),
        unapplied_lines=("is_unapplied", "sum"),
        lines=("amount_try", "size"),
    )
    check_weekly_totals(collections, weekly)
    return weekly


def check_weekly_totals(collections: pd.DataFrame, weekly: pd.DataFrame) -> None:
    """Raise ValueError unless the weekly table holds every TRY and every line of the input."""
    lines_total = collections["amount_try"].sum()
    weekly_total = weekly["gross_receipts"].sum() + weekly["refunds"].sum()
    if abs(lines_total - weekly_total) > 0.01:
        raise ValueError(f"the weekly amounts add up to {weekly_total:,.2f} TRY, "
                         f"but the payment lines to {lines_total:,.2f} TRY")
    if weekly["lines"].sum() != len(collections):
        raise ValueError(f"the weekly table counts {weekly['lines'].sum()} lines, "
                         f"but there are {len(collections)}")
