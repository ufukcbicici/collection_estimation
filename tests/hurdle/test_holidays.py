"""Tests for `hurdle/holidays.py`: reading and checking a holidays file (`load_holidays`,
`read_fixed_holidays`, `read_bayrams`), the public holidays between two dates, and the guard
that refuses a year whose Bayram dates are not listed.

Every broken holidays file is written into `tmp_path` by `write_holidays`, which starts from a
small valid file and changes one thing.
"""
import dataclasses
import json
import tomllib
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from collection_estimation.hurdle.holidays import (
    BAYRAM_NAMES,
    HolidayTable,
    check_bayrams_cover,
    load_holidays,
    public_holidays,
    read_bayrams,
    read_fixed_holidays,
)
from collection_estimation.hurdle.read_collections import list_daily_files
from tests.hurdle.workbooks import SYNTHETIC_HOLIDAYS as HOLIDAYS
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT

DATE_FORMAT = SYNTHETIC_LAYOUT.file_name_date_format  # "%d%m%Y"

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLIDAYS_FILE = REPO_ROOT / "settings" / "holidays_tr.toml"
ORACLE = REPO_ROOT / "data" / "raw" / "collections_corpus.json"
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"

SYNTHETIC_START = date(2025, 6, 2)
SYNTHETIC_END = date(2026, 8, 28)

FIXED_DAYS = [(1, 1), (4, 23), (5, 1), (5, 19), (7, 15), (8, 30), (10, 29)]
FIXED_NAMES = {
    (1, 1): "New Year's Day",
    (4, 23): "National Sovereignty and Children's Day",
    (5, 1): "Labour and Solidarity Day",
    (5, 19): "Commemoration of Atatürk, Youth and Sports Day",
    (7, 15): "Democracy and National Unity Day",
    (8, 30): "Victory Day",
    (10, 29): "Republic Day",
}

RAMAZAN_2027 = (date(2027, 3, 9), date(2027, 3, 11), "Ramazan Bayramı")
KURBAN_2027 = (date(2027, 5, 16), date(2027, 5, 19), "Kurban Bayramı")


def days_between(first, last):
    """Every date from `first` to `last`, both included."""
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


def with_bayrams(*extra, drop=()):
    """HOLIDAYS with the Bayrams in `extra` added and those (year, name) in `drop` removed."""
    kept = tuple(entry for entry in HOLIDAYS.bayrams if (entry[0].year, entry[2]) not in drop)
    return HolidayTable(fixed=HOLIDAYS.fixed, bayrams=kept + tuple(extra))


# --- the repo's holidays file ----------------------------------------------------------

def test_the_file_loads_into_a_holiday_table():
    assert isinstance(load_holidays(HOLIDAYS_FILE), HolidayTable)


def test_the_file_equals_the_shared_test_table():
    assert load_holidays(HOLIDAYS_FILE) == HOLIDAYS


def test_the_file_lists_the_seven_national_days():
    assert sorted(load_holidays(HOLIDAYS_FILE).fixed) == FIXED_DAYS


def test_the_file_names_the_seven_national_days():
    assert load_holidays(HOLIDAYS_FILE).fixed == FIXED_NAMES


def test_the_file_lists_four_bayrams():
    assert len(load_holidays(HOLIDAYS_FILE).bayrams) == 4


def test_the_file_lists_both_bayrams_for_2025_and_2026():
    listed = sorted((first.year, name) for first, _, name in load_holidays(HOLIDAYS_FILE).bayrams)
    assert listed == [(2025, "Kurban Bayramı"), (2025, "Ramazan Bayramı"),
                      (2026, "Kurban Bayramı"), (2026, "Ramazan Bayramı")]


def test_the_file_bayram_dates():
    assert load_holidays(HOLIDAYS_FILE).bayrams == (
        (date(2025, 3, 30), date(2025, 4, 1), "Ramazan Bayramı"),
        (date(2025, 6, 6), date(2025, 6, 9), "Kurban Bayramı"),
        (date(2026, 3, 20), date(2026, 3, 22), "Ramazan Bayramı"),
        (date(2026, 5, 27), date(2026, 5, 30), "Kurban Bayramı"),
    )


def test_every_bayram_in_the_file_ends_on_or_after_it_starts():
    for first_day, last_day, _ in load_holidays(HOLIDAYS_FILE).bayrams:
        assert first_day <= last_day


def test_the_file_covers_2025_and_2026():
    assert check_bayrams_cover(date(2025, 1, 1), date(2026, 12, 31),
                               load_holidays(HOLIDAYS_FILE)) is None


# --- fixed holidays --------------------------------------------------------------------

@pytest.mark.parametrize("year", [2025, 2026])
@pytest.mark.parametrize("month, day", FIXED_DAYS)
def test_fixed_holiday_appears_in_every_year(year, month, day):
    holidays = public_holidays(date(2025, 1, 1), date(2026, 12, 31), HOLIDAYS)
    assert date(year, month, day) in holidays


def test_fixed_holidays_carry_their_names():
    holidays = public_holidays(date(2026, 1, 1), date(2026, 12, 31), HOLIDAYS)
    assert holidays[date(2026, 1, 1)] == "New Year's Day"
    assert holidays[date(2026, 4, 23)] == "National Sovereignty and Children's Day"
    assert holidays[date(2026, 5, 1)] == "Labour and Solidarity Day"
    assert holidays[date(2026, 5, 19)] == "Commemoration of Atatürk, Youth and Sports Day"
    assert holidays[date(2026, 7, 15)] == "Democracy and National Unity Day"
    assert holidays[date(2026, 8, 30)] == "Victory Day"
    assert holidays[date(2026, 10, 29)] == "Republic Day"


# --- Bayrams ---------------------------------------------------------------------------

def test_kurban_2026_includes_all_four_days():
    holidays = public_holidays(date(2026, 5, 20), date(2026, 6, 5), HOLIDAYS)
    assert list(holidays) == [date(2026, 5, 27), date(2026, 5, 28),
                              date(2026, 5, 29), date(2026, 5, 30)]


def test_kurban_2026_is_named_kurban_bayrami():
    holidays = public_holidays(date(2026, 5, 27), date(2026, 5, 30), HOLIDAYS)
    assert set(holidays.values()) == {"Kurban Bayramı"}


def test_ramazan_2026_includes_20_to_22_march():
    holidays = public_holidays(date(2026, 3, 15), date(2026, 3, 25), HOLIDAYS)
    assert holidays == {date(2026, 3, 20): "Ramazan Bayramı",
                        date(2026, 3, 21): "Ramazan Bayramı",
                        date(2026, 3, 22): "Ramazan Bayramı"}


def test_day_before_and_after_a_bayram_are_not_holidays():
    holidays = public_holidays(date(2026, 3, 1), date(2026, 3, 31), HOLIDAYS)
    assert date(2026, 3, 19) not in holidays
    assert date(2026, 3, 23) not in holidays


def test_every_day_of_every_listed_bayram_is_included():
    holidays = public_holidays(date(2025, 1, 1), date(2026, 12, 31), HOLIDAYS)
    for first_day, last_day, name in HOLIDAYS.bayrams:
        for day in days_between(first_day, last_day):
            assert holidays[day] == name


# --- range edges -----------------------------------------------------------------------

def test_range_starting_2_jan_excludes_1_jan():
    holidays = public_holidays(date(2026, 1, 2), date(2026, 1, 31), HOLIDAYS)
    assert date(2026, 1, 1) not in holidays


def test_range_ending_on_a_holiday_includes_it():
    holidays = public_holidays(date(2026, 10, 1), date(2026, 10, 29), HOLIDAYS)
    assert date(2026, 10, 29) in holidays


def test_range_starting_on_a_holiday_includes_it():
    holidays = public_holidays(date(2026, 1, 1), date(2026, 1, 31), HOLIDAYS)
    assert date(2026, 1, 1) in holidays


def test_range_cutting_a_bayram_keeps_only_the_days_inside():
    holidays = public_holidays(date(2026, 5, 28), date(2026, 5, 29), HOLIDAYS)
    assert list(holidays) == [date(2026, 5, 28), date(2026, 5, 29)]


def test_range_with_no_holidays_gives_empty_dict():
    assert public_holidays(date(2026, 2, 2), date(2026, 2, 5), HOLIDAYS) == {}


def test_single_day_on_a_holiday_gives_that_day():
    assert public_holidays(date(2026, 4, 23), date(2026, 4, 23), HOLIDAYS) == {
        date(2026, 4, 23): "National Sovereignty and Children's Day"}


def test_single_day_not_a_holiday_gives_empty_dict():
    assert public_holidays(date(2026, 4, 24), date(2026, 4, 24), HOLIDAYS) == {}


def test_start_after_end_raises_naming_both_dates():
    with pytest.raises(ValueError, match="is after the end") as error:
        public_holidays(date(2026, 5, 30), date(2026, 5, 1), HOLIDAYS)
    assert "2026-05-30" in str(error.value)
    assert "2026-05-01" in str(error.value)


def test_start_one_day_after_end_raises():
    with pytest.raises(ValueError, match="is after the end"):
        public_holidays(date(2026, 5, 2), date(2026, 5, 1), HOLIDAYS)


def test_start_equal_to_end_does_not_raise():
    assert public_holidays(date(2026, 5, 1), date(2026, 5, 1), HOLIDAYS) == {
        date(2026, 5, 1): "Labour and Solidarity Day"}


# --- shape of the result ---------------------------------------------------------------

def test_result_is_sorted_by_date():
    holidays = public_holidays(date(2025, 1, 1), date(2026, 12, 31), HOLIDAYS)
    assert list(holidays) == sorted(holidays)


def test_bayram_falling_between_fixed_holidays_is_sorted_in_place():
    # Fixed holidays are added first, Bayrams later; the result must still be in date order.
    holidays = public_holidays(date(2026, 5, 1), date(2026, 5, 31), HOLIDAYS)
    assert list(holidays) == [date(2026, 5, 1), date(2026, 5, 19), date(2026, 5, 27),
                              date(2026, 5, 28), date(2026, 5, 29), date(2026, 5, 30)]


def test_keys_are_dates():
    holidays = public_holidays(date(2025, 1, 1), date(2026, 12, 31), HOLIDAYS)
    assert all(type(day) is date for day in holidays)


def test_weekends_are_included():
    # It lists holidays, not business days: Victory Day 2025 falls on a Saturday.
    assert date(2025, 8, 30).weekday() == 5
    assert date(2025, 8, 30) in public_holidays(date(2025, 8, 1), date(2025, 8, 31), HOLIDAYS)


def test_calling_twice_gives_the_same_result():
    first = public_holidays(date(2025, 1, 1), date(2026, 12, 31), HOLIDAYS)
    second = public_holidays(date(2025, 1, 1), date(2026, 12, 31), HOLIDAYS)
    assert first == second


# --- the coverage guard ----------------------------------------------------------------

def test_range_reaching_2027_raises_naming_the_year_and_the_holidays_file():
    with pytest.raises(ValueError) as error:
        public_holidays(date(2026, 12, 1), date(2027, 1, 31), HOLIDAYS)
    assert "2027" in str(error.value)
    assert "add them to the holidays file" in str(error.value)
    assert "BAYRAMS" not in str(error.value)  # the old constant is gone


def test_range_reaching_2024_raises_naming_the_year():
    with pytest.raises(ValueError) as error:
        public_holidays(date(2024, 12, 1), date(2025, 1, 31), HOLIDAYS)
    assert "2024" in str(error.value)
    assert "2025" not in str(error.value)


def test_both_ends_uncovered_names_both_years():
    with pytest.raises(ValueError) as error:
        public_holidays(date(2024, 12, 1), date(2027, 1, 31), HOLIDAYS)
    assert "2024" in str(error.value)
    assert "2027" in str(error.value)


def test_uncovered_range_raises_even_if_it_holds_no_holiday():
    # 2 to 5 Feb 2027 has no fixed holiday, but its Bayrams are unknown.
    with pytest.raises(ValueError, match="2027"):
        public_holidays(date(2027, 2, 2), date(2027, 2, 5), HOLIDAYS)


def test_check_bayrams_cover_passes_silently_on_a_covered_range():
    assert check_bayrams_cover(date(2025, 1, 1), date(2026, 12, 31), HOLIDAYS) is None


def test_check_bayrams_cover_raises_on_an_uncovered_year():
    with pytest.raises(ValueError, match="add them to the holidays file"):
        check_bayrams_cover(date(2027, 1, 1), date(2027, 1, 1), HOLIDAYS)


def test_check_bayrams_cover_start_after_end_raises_naming_both_dates():
    with pytest.raises(ValueError, match="is after the end") as error:
        check_bayrams_cover(date(2026, 5, 30), date(2026, 5, 1), HOLIDAYS)
    assert "2026-05-30" in str(error.value)
    assert "2026-05-01" in str(error.value)


def test_check_bayrams_cover_start_equal_to_end_passes():
    assert check_bayrams_cover(date(2026, 5, 1), date(2026, 5, 1), HOLIDAYS) is None


def test_reversed_range_is_reported_before_missing_bayrams():
    # Both ends uncovered and reversed: the reversal is the error reported.
    with pytest.raises(ValueError, match="is after the end"):
        check_bayrams_cover(date(2027, 1, 1), date(2024, 1, 1), HOLIDAYS)


# --- both Bayrams per year -------------------------------------------------------------

def test_bayram_names_are_ramazan_and_kurban():
    assert BAYRAM_NAMES == ["Ramazan Bayramı", "Kurban Bayramı"]


def test_every_bayram_in_the_shared_table_has_a_known_name():
    for _, _, name in HOLIDAYS.bayrams:
        assert name in BAYRAM_NAMES


def test_every_covered_year_lists_both_bayrams():
    listed = {(first_day.year, name) for first_day, _, name in HOLIDAYS.bayrams}
    for year in {first_day.year for first_day, _, _ in HOLIDAYS.bayrams}:
        for name in BAYRAM_NAMES:
            assert (year, name) in listed


def test_year_with_only_ramazan_raises_naming_kurban():
    table = with_bayrams(RAMAZAN_2027)  # Kurban 2027 missing
    with pytest.raises(ValueError) as error:
        check_bayrams_cover(date(2027, 1, 1), date(2027, 12, 31), table)
    assert "Kurban Bayramı 2027" in str(error.value)
    assert "Ramazan Bayramı 2027" not in str(error.value)


def test_year_with_only_ramazan_raises_via_public_holidays():
    table = with_bayrams(RAMAZAN_2027)
    with pytest.raises(ValueError, match="Kurban Bayramı 2027"):
        public_holidays(date(2027, 2, 1), date(2027, 2, 5), table)


def test_year_with_both_bayrams_listed_passes():
    table = with_bayrams(RAMAZAN_2027, KURBAN_2027)
    assert check_bayrams_cover(date(2027, 1, 1), date(2027, 12, 31), table) is None
    holidays = public_holidays(date(2027, 5, 1), date(2027, 5, 31), table)
    assert holidays[date(2027, 5, 16)] == "Kurban Bayramı"


def test_uncovered_year_names_both_its_bayrams():
    with pytest.raises(ValueError) as error:
        check_bayrams_cover(date(2027, 6, 1), date(2027, 6, 30), HOLIDAYS)
    assert "Ramazan Bayramı 2027" in str(error.value)
    assert "Kurban Bayramı 2027" in str(error.value)


def test_missing_bayram_in_one_year_does_not_flag_covered_years():
    # Drop Kurban 2026 only: 2025 stays clean, 2026 names only Kurban.
    table = with_bayrams(drop={(2026, "Kurban Bayramı")})
    with pytest.raises(ValueError) as error:
        check_bayrams_cover(date(2025, 1, 1), date(2026, 12, 31), table)
    message = str(error.value)
    assert "Kurban Bayramı 2026" in message
    assert "Ramazan Bayramı 2026" not in message
    assert "2025" not in message


def test_a_table_without_bayrams_names_every_year_of_the_range():
    table = HolidayTable(fixed=HOLIDAYS.fixed, bayrams=())
    with pytest.raises(ValueError) as error:
        check_bayrams_cover(date(2025, 1, 1), date(2026, 1, 1), table)
    for name in BAYRAM_NAMES:
        for year in (2025, 2026):
            assert f"{name} {year}" in str(error.value)


# --- public_holidays uses the table it is given ------------------------------------------

def test_public_holidays_uses_the_fixed_holidays_of_the_table():
    table = HolidayTable(fixed={(2, 4): "Test Day"}, bayrams=HOLIDAYS.bayrams)
    assert public_holidays(date(2026, 1, 1), date(2026, 2, 28), table) == {
        date(2026, 2, 4): "Test Day"}


def test_public_holidays_with_no_fixed_holidays_gives_only_bayrams():
    table = HolidayTable(fixed={}, bayrams=HOLIDAYS.bayrams)
    holidays = public_holidays(date(2026, 1, 1), date(2026, 12, 31), table)
    assert set(holidays.values()) == {"Ramazan Bayramı", "Kurban Bayramı"}
    assert len(holidays) == 3 + 4


def test_public_holidays_uses_the_bayrams_of_the_table():
    table = with_bayrams(RAMAZAN_2027, KURBAN_2027)
    holidays = public_holidays(date(2027, 3, 1), date(2027, 3, 31), table)
    assert holidays == {day: "Ramazan Bayramı" for day in days_between(*RAMAZAN_2027[:2])}


def test_a_one_day_bayram_gives_one_day():
    table = with_bayrams((date(2027, 3, 9), date(2027, 3, 9), "Ramazan Bayramı"), KURBAN_2027)
    holidays = public_holidays(date(2027, 3, 1), date(2027, 3, 31), table)
    assert holidays == {date(2027, 3, 9): "Ramazan Bayramı"}


def test_a_bayram_overrides_a_fixed_holiday_on_the_same_day():
    # Bayrams are added after the fixed holidays, so a day in both carries the Bayram's name.
    table = with_bayrams((date(2027, 5, 1), date(2027, 5, 3), "Kurban Bayramı"), RAMAZAN_2027)
    holidays = public_holidays(date(2027, 5, 1), date(2027, 5, 1), table)
    assert holidays == {date(2027, 5, 1): "Kurban Bayramı"}


LEAP_TABLE = HolidayTable(
    fixed={(2, 29): "Leap Day"},
    bayrams=(RAMAZAN_2027, KURBAN_2027,
             (date(2028, 2, 26), date(2028, 2, 28), "Ramazan Bayramı"),
             (date(2028, 5, 4), date(2028, 5, 7), "Kurban Bayramı")),
)


def test_a_fixed_29_february_appears_in_a_leap_year():
    holidays = public_holidays(date(2028, 2, 1), date(2028, 3, 31), LEAP_TABLE)
    assert holidays[date(2028, 2, 29)] == "Leap Day"


def test_a_fixed_29_february_is_skipped_in_a_non_leap_year():
    holidays = public_holidays(date(2027, 1, 1), date(2027, 12, 31), LEAP_TABLE)
    assert "Leap Day" not in holidays.values()
    assert date(2027, 3, 1) not in holidays


def test_a_fixed_29_february_over_two_years_appears_once():
    holidays = public_holidays(date(2027, 1, 1), date(2028, 12, 31), LEAP_TABLE)
    assert [day for day, name in holidays.items() if name == "Leap Day"] == [date(2028, 2, 29)]


def test_public_holidays_does_not_modify_the_table():
    table = HolidayTable(fixed=dict(HOLIDAYS.fixed), bayrams=HOLIDAYS.bayrams)
    public_holidays(date(2025, 1, 1), date(2026, 12, 31), table)
    assert table == HOLIDAYS
    assert list(table.fixed) == list(HOLIDAYS.fixed)


def test_public_holidays_needs_the_table():
    with pytest.raises(TypeError):
        public_holidays(date(2026, 1, 1), date(2026, 1, 31))


# --- HolidayTable ----------------------------------------------------------------------

@pytest.mark.parametrize("attribute", ["fixed", "bayrams"])
def test_holiday_table_is_frozen(attribute):
    table = load_holidays(HOLIDAYS_FILE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(table, attribute, None)


def test_holiday_table_fields_are_fixed_and_bayrams():
    assert [field.name for field in dataclasses.fields(HolidayTable)] == ["fixed", "bayrams"]


# --- the synthetic corpus --------------------------------------------------------------

@pytest.mark.skipif(not ORACLE.is_file(), reason="synthetic oracle not installed")
def test_synthetic_range_matches_the_oracle_holidays():
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    oracle_dates = {date.fromisoformat(entry["date"]) for entry in oracle["calendar"]["holidays"]}
    assert len(oracle_dates) == 19
    assert set(public_holidays(SYNTHETIC_START, SYNTHETIC_END, HOLIDAYS)) == oracle_dates


@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason="synthetic corpus not installed")
def test_no_synthetic_daily_file_falls_on_a_holiday():
    files = list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT)
    holidays = public_holidays(SYNTHETIC_START, SYNTHETIC_END, HOLIDAYS)
    assert set(files) & set(holidays) == set()


@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason="synthetic corpus not installed")
def test_synthetic_missing_business_days_are_the_six_known_ones():
    files = list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT)
    holidays = public_holidays(SYNTHETIC_START, SYNTHETIC_END, HOLIDAYS)
    missing = [day for day in days_between(SYNTHETIC_START, SYNTHETIC_END)
               if day.weekday() < 5 and day not in holidays and day not in files]
    assert missing == [date(2025, 9, 8), date(2025, 12, 17), date(2026, 2, 4),
                       date(2026, 3, 13), date(2026, 4, 20), date(2026, 6, 1)]
