"""Section 9.2 — the calendar block.

One row per week describing that week's calendar. Joined to the panel on ``target_week``,
which is what makes these features legitimate at every horizon: they describe the week
being predicted, and a calendar is knowable years in advance.

This is the block that carries the signal. A regression on the recent level plus four of
these facts reaches 16.5% MAPE against the 8-week average's 24.2% — see section 11.2.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

# --------------------------------------------------------------------------------------
# Turkish public holidays
# --------------------------------------------------------------------------------------

#: Fixed-date national holidays. These are certain.
#:
#: 1 Jan   New Year            23 Apr  National Sovereignty and Children's Day
#: 1 May   Labour and Solidarity   19 May  Commemoration of Atatürk, Youth and Sports
#: 15 Jul  Democracy and National Unity    30 Aug  Victory Day
#: 29 Oct  Republic Day
FIXED_HOLIDAYS = ((1, 1), (4, 23), (5, 1), (5, 19), (7, 15), (8, 30), (10, 29))

#: The religious holidays, which move about eleven days earlier each Gregorian year and
#: therefore cannot be computed from a rule here. **These dates are APPROXIMATE** and are
#: declared rather than derived, so that anyone reading a result knows which part of the
#: calendar is solid and which is not.
#:
#: (first day, last day, name)
MOVING_HOLIDAYS = (
    (date(2025, 3, 30), date(2025, 4, 1), "Ramazan Bayramı"),
    (date(2025, 6, 6), date(2025, 6, 9), "Kurban Bayramı"),
    (date(2026, 3, 20), date(2026, 3, 22), "Ramazan Bayramı"),
    (date(2026, 5, 27), date(2026, 5, 30), "Kurban Bayramı"),
)


class HolidayCoverageError(ValueError):
    """The requested range reaches beyond the declared moving-holiday table.

    Raised rather than tolerated, because the failure it prevents is silent. Without the
    guard, `holiday_dates` for an uncovered year simply returns no Bayram: a four-day
    religious holiday shows up as five ordinary business days, and every model reads a
    collapsed week as a normal one. Nothing errors, nothing looks wrong, and the
    business-day count is quietly false for two weeks a year.
    """


#: Years for which `MOVING_HOLIDAYS` declares the religious holidays. Derived from the
#: table itself so the two cannot drift apart.
def covered_years() -> set[int]:
    return {y for first, last, _ in MOVING_HOLIDAYS
            for y in range(first.year, last.year + 1)}


def check_holiday_coverage(start: date, end: date) -> None:
    """Fail loudly if any year in range has no declared religious holidays."""
    covered = covered_years()
    missing = sorted(set(range(start.year, end.year + 1)) - covered)
    if missing:
        raise HolidayCoverageError(
            f"no Ramazan or Kurban Bayramı declared for {missing}. "
            f"MOVING_HOLIDAYS covers {sorted(covered)}. Extend it before using data "
            f"from {start} to {end}: without those dates a four-day Bayram week would "
            f"be counted as five ordinary business days and nothing would complain.")


def holiday_dates(start: date, end: date, *, strict: bool = True) -> dict[date, str]:
    """Every public holiday in range, mapped to its name.

    Raises `HolidayCoverageError` if the range covers a year the moving-holiday table does
    not declare. Pass ``strict=False`` only when the missing dates genuinely do not matter
    — and say so at the call site, because the resulting business-day counts are wrong.
    """
    if strict:
        check_holiday_coverage(start, end)
    out: dict[date, str] = {}
    for year in range(start.year, end.year + 1):
        for month, day in FIXED_HOLIDAYS:
            when = date(year, month, day)
            if start <= when <= end:
                out[when] = "national holiday"
    for first, last, name in MOVING_HOLIDAYS:
        when = first
        while when <= last:
            if start <= when <= end:
                out[when] = name
            when += timedelta(days=1)
    return out


def business_days(start: date, end: date, *, strict: bool = True) -> list[date]:
    """Weekdays that are not public holidays."""
    holidays = holiday_dates(start, end, strict=strict)
    out, when = [], start
    while when <= end:
        if when.weekday() < 5 and when not in holidays:
            out.append(when)
        when += timedelta(days=1)
    return out


# --------------------------------------------------------------------------------------
# The weekly calendar block
# --------------------------------------------------------------------------------------

def build_calendar_features(weeks: pd.DataFrame) -> pd.DataFrame:
    """One row per week: the calendar facts a forecaster knows in advance.

    ``week``                the spine's integer index
    ``n_business_days``     0-5; fewer around public holidays
    ``n_holidays``          weekday public holidays falling in the week
    ``has_month_end``       the week contains the last BUSINESS day of a month
    ``has_quarter_end``     ... of March, June, September or December
    ``weeks_to_month_end``  0 if this week has one, else how many weeks until the next
    ``month``               1-12, taken from the week's Monday
    ``week_of_year``        ISO week number

    **Month end means the last BUSINESS day of the month, not the last calendar day.**
    The flag has to follow the money, and the money moves on the last day anyone is at
    work. May 2026 is the sharp case in this corpus: the 31st is a Sunday and the 27th to
    the 30th are Kurban Bayramı, so the last business day of May is **Tuesday the 26th** —
    five days before the calendar month ends. A calendar-day rule would put the flag on
    the week of 1 June, the week after the batch actually cleared.

    This is not rare. A month ends on a weekend roughly two months in seven, before
    holidays are considered at all.

    No customer information appears here: the block is identical for every customer in a
    given week. Origin-week features are section 9.1.
    """
    first = weeks["week_start"].iloc[0].date()
    last = weeks["week_end"].iloc[-1].date()

    holidays = holiday_dates(first, last)
    workdays = business_days(first, last)

    # The last business day of each month, which is what "month end" means here.
    last_business: dict[tuple[int, int], date] = {}
    for day in workdays:
        key = (day.year, day.month)
        if key not in last_business or day > last_business[key]:
            last_business[key] = day
    month_ends = set(last_business.values())
    quarter_ends = {d for (_, m), d in last_business.items() if m in (3, 6, 9, 12)}

    rows = []
    for _, week in weeks.iterrows():
        start = week["week_start"].date()
        end = week["week_end"].date()
        days = [start + timedelta(days=i) for i in range(7)]

        rows.append({
            "week": int(week["week"]),
            "n_business_days": sum(1 for d in days
                                   if d.weekday() < 5 and d not in holidays),
            "n_holidays": sum(1 for d in days
                              if d.weekday() < 5 and d in holidays),
            "has_month_end": int(any(d in month_ends for d in days)),
            "has_quarter_end": int(any(d in quarter_ends for d in days)),
            "month": start.month,
            "week_of_year": start.isocalendar().week,
        })

    table = pd.DataFrame(rows)

    # Weeks until the next month end, counting this one as 0. Computed backwards so the
    # tail of the corpus — which may have no further month end — is filled with the
    # distance to the end rather than silently left as zero.
    distance, seen = [], None
    for has_end in reversed(table["has_month_end"].tolist()):
        if has_end:
            seen = 0
        elif seen is not None:
            seen += 1
        distance.append(seen)
    table["weeks_to_month_end"] = pd.Series(
        list(reversed(distance)), dtype="Int16")

    return table.astype({
        "week": "int16", "n_business_days": "int8", "n_holidays": "int8",
        "has_month_end": "int8", "has_quarter_end": "int8",
        "month": "int8", "week_of_year": "int8",
    })


def cross_check_business_days(weeks: pd.DataFrame,
                              collections: pd.DataFrame) -> pd.DataFrame:
    """Compare the declared holiday table against the days the corpus actually has files.

    **This exists because the hardcoded table is the weak part of this module.** The fixed
    national holidays are certain, but Ramazan and Kurban Bayramı move about eleven days a
    year and are declared approximately. Worse, this corpus was *generated* using a
    holiday table of its own, so a matching table here lines up with the generative
    process and would make results look better than they will be on real data.

    Reporting the disagreement turns that from an assumption into a number. Two kinds:

    ``expected_working_no_data``  a business day by our table with no collections at all.
                                  Either our holiday table is missing a date, or the day's
                                  file is one of the corpus's deliberately absent ones —
                                  and those are different things. EY's own instruction 8
                                  insists a missing source day is not a zero.
    ``holiday_with_data``         a day our table calls a holiday that nonetheless has
                                  collections. Our table is wrong for that date.
    """
    first = weeks["week_start"].iloc[0].date()
    last = weeks["week_end"].iloc[-1].date()

    observed = {d.date() for d in pd.to_datetime(
        collections["document_date"]).dt.normalize().unique()}
    holidays = holiday_dates(first, last)
    workdays = set(business_days(first, last))

    rows = []
    for day in sorted(workdays - observed):
        rows.append({"date": day, "kind": "expected_working_no_data",
                     "note": "our table says a working day, the corpus has no rows"})
    for day, name in sorted(holidays.items()):
        if day in observed:
            rows.append({"date": day, "kind": "holiday_with_data",
                         "note": f"our table says {name}, the corpus has rows"})
    return pd.DataFrame(rows, columns=["date", "kind", "note"])
