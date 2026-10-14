"""Public holidays: a day without a daily file on a holiday is not a missing day.

The holidays themselves are data, kept in a TOML file (settings/holidays_tr.toml for
Turkey) and named by `holidays_file` in the settings. This module reads and checks that
file, and lists the holidays in a date range.

The fixed-date holidays are certain. The two Bayrams (religious holidays) move about eleven
days earlier every year and cannot be computed by a simple rule, so their dates are listed
by hand and are APPROXIMATE. Half-day holidays (the eve, "arife") are not listed.
"""
import re
import tomllib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .settings import check_keys, date_setting

# Every year has both; `check_bayrams_cover` requires both to be listed for each year.
BAYRAM_NAMES = ["Ramazan Bayramı", "Kurban Bayramı"]


@dataclass(frozen=True)
class HolidayTable:
    """The holidays read from a holidays file."""
    fixed: dict[tuple[int, int], str]             # (month, day) -> name, the same every year
    bayrams: tuple[tuple[date, date, str], ...]   # (first day, last day, name)


def load_holidays(holidays_file: Path) -> HolidayTable:
    """Read and check a holidays file.

    It has a [fixed] table of "MM-DD" = name, and one [[bayram]] entry (name, first_day,
    last_day) per Bayram per year. Raises ValueError naming the file and the entry if
    anything is missing, unknown, of the wrong type, or contradictory.
    """
    holidays_file = Path(holidays_file)
    with open(holidays_file, "rb") as file:  # tomllib reads bytes
        raw = tomllib.load(file)
    name = holidays_file.name
    check_keys(raw, {"fixed", "bayram"}, name)
    return HolidayTable(
        fixed=read_fixed_holidays(raw["fixed"], f"{name} [fixed]"),
        bayrams=read_bayrams(raw["bayram"], f"{name} [[bayram]]"),
    )


def read_fixed_holidays(table, where: str) -> dict[tuple[int, int], str]:
    """The [fixed] table: "MM-DD" = name, e.g. "10-29" = "Republic Day"."""
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{where} is {table!r}, not a table of \"MM-DD\" = name")
    fixed = {}
    for month_day, holiday_name in table.items():
        try:
            # Exactly two digits, a hyphen and two digits: "1-1" or " 01-01" would otherwise
            # be accepted, and "1-1" and "01-01" would then silently be the same holiday.
            if re.fullmatch(r"[0-9]{2}-[0-9]{2}", month_day) is None:
                raise ValueError
            month, day = int(month_day[:2]), int(month_day[3:])
            date(2024, month, day)  # a leap year, so that "02-29" would be accepted
        except ValueError:
            raise ValueError(f"{where}: '{month_day}' is not a month and day such as "
                             f"\"10-29\"") from None
        if not isinstance(holiday_name, str) or holiday_name.strip() == "":
            raise ValueError(f"{where}: '{month_day}' has the name {holiday_name!r}, not a text")
        fixed[(month, day)] = holiday_name
    return fixed


def read_bayrams(entries, where: str) -> tuple[tuple[date, date, str], ...]:
    """The [[bayram]] entries: name (one of BAYRAM_NAMES), first_day and last_day."""
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{where} is {entries!r}, not a list of [[bayram]] entries")
    bayrams = []
    listed = set()  # (year, name) already seen
    for number, entry in enumerate(entries, start=1):
        entry_where = f"{where} entry {number}"
        check_keys(entry, {"name", "first_day", "last_day"}, entry_where)
        bayram_name = entry["name"]
        if bayram_name not in BAYRAM_NAMES:
            raise ValueError(f"{entry_where}: name is {bayram_name!r}, not one of {BAYRAM_NAMES}")
        first_day = date_setting(entry, "first_day", entry_where)
        last_day = date_setting(entry, "last_day", entry_where)
        if first_day > last_day:
            raise ValueError(f"{entry_where}: first_day {first_day} is after last_day {last_day}")
        if (first_day.year, bayram_name) in listed:
            raise ValueError(f"{entry_where}: {bayram_name} {first_day.year} is listed twice")
        listed.add((first_day.year, bayram_name))
        bayrams.append((first_day, last_day, bayram_name))
    return tuple(bayrams)


def public_holidays(start: date, end: date, holidays: HolidayTable) -> dict[date, str]:
    """{date: holiday name} for every public holiday from `start` to `end`, both included.

    Raises ValueError if `start` is after `end`, or if the range reaches a year whose two
    Bayrams are not both listed: without that check a four-day Bayram would quietly look
    like four missing business days.
    """
    check_bayrams_cover(start, end, holidays)

    in_any_year = {}
    for year in range(start.year, end.year + 1):
        for (month, day), name in holidays.fixed.items():
            try:
                in_any_year[date(year, month, day)] = name
            except ValueError:
                continue  # 29 February in a year that is not a leap year

    for first_day, last_day, name in holidays.bayrams:
        day = first_day
        while day <= last_day:
            in_any_year[day] = name
            day += timedelta(days=1)

    in_range = {day: name for day, name in in_any_year.items() if start <= day <= end}
    return dict(sorted(in_range.items()))


def check_bayrams_cover(start: date, end: date, holidays: HolidayTable) -> None:
    """Raise ValueError if `start` is after `end`, or if a year in the range does not have
    both of its Bayrams listed in the holidays file."""
    if start > end:
        raise ValueError(f"the start {start} is after the end {end}")

    listed = {(first_day.year, name) for first_day, _, name in holidays.bayrams}
    not_listed = []
    for year in range(start.year, end.year + 1):
        for name in BAYRAM_NAMES:
            if (year, name) not in listed:
                not_listed.append(f"{name} {year}")
    if not_listed:
        raise ValueError(f"no dates listed for {not_listed}; add them to the holidays file")
