"""Turkish public holidays: a day without a daily file on a holiday is not a missing day.

The fixed-date holidays are certain. The two Bayrams (religious holidays) move about eleven
days earlier every year and cannot be computed by a simple rule, so their dates are listed
by hand and are APPROXIMATE. Half-day holidays (the eve, "arife") are not listed.
"""
from datetime import date, timedelta

# (month, day): name. The same every year.
FIXED_HOLIDAYS = {
    (1, 1): "New Year's Day",
    (4, 23): "National Sovereignty and Children's Day",
    (5, 1): "Labour and Solidarity Day",
    (5, 19): "Commemoration of Atatürk, Youth and Sports Day",
    (7, 15): "Democracy and National Unity Day",
    (8, 30): "Victory Day",
    (10, 29): "Republic Day",
}

# Every year has both; `check_bayrams_cover` requires both to be listed for each year.
BAYRAM_NAMES = ["Ramazan Bayramı", "Kurban Bayramı"]

# (first day, last day, name). Approximate; extend before using data from a later year.
BAYRAMS = [
    (date(2025, 3, 30), date(2025, 4, 1), "Ramazan Bayramı"),
    (date(2025, 6, 6), date(2025, 6, 9), "Kurban Bayramı"),
    (date(2026, 3, 20), date(2026, 3, 22), "Ramazan Bayramı"),
    (date(2026, 5, 27), date(2026, 5, 30), "Kurban Bayramı"),
]


def public_holidays(start: date, end: date) -> dict[date, str]:
    """{date: holiday name} for every public holiday from `start` to `end`, both included.

    Raises ValueError if `start` is after `end`, or if the range reaches a year whose two
    Bayrams are not both listed: without that check a four-day Bayram would quietly look
    like four missing business days.
    """
    check_bayrams_cover(start, end)

    holidays = {}
    for year in range(start.year, end.year + 1):
        for (month, day), name in FIXED_HOLIDAYS.items():
            holidays[date(year, month, day)] = name

    for first_day, last_day, name in BAYRAMS:
        day = first_day
        while day <= last_day:
            holidays[day] = name
            day += timedelta(days=1)

    in_range = {day: name for day, name in holidays.items() if start <= day <= end}
    return dict(sorted(in_range.items()))


def check_bayrams_cover(start: date, end: date) -> None:
    """Raise ValueError if `start` is after `end`, or if a year in the range does not have
    both of its Bayrams listed in BAYRAMS."""
    if start > end:
        raise ValueError(f"the start {start} is after the end {end}")

    listed = {(first_day.year, name) for first_day, _, name in BAYRAMS}
    not_listed = []
    for year in range(start.year, end.year + 1):
        for name in BAYRAM_NAMES:
            if (year, name) not in listed:
                not_listed.append(f"{name} {year}")
    if not_listed:
        raise ValueError(
            f"no dates listed for {not_listed}; add them to BAYRAMS in holidays.py")
