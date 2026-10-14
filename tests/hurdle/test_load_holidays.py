"""Tests for reading a holidays file: `load_holidays` and its helpers `read_fixed_holidays`
and `read_bayrams` in `hurdle/holidays.py`.

Every holidays file is written into `tmp_path` by `write_holidays`, which starts from a small
valid file and changes, removes or adds one thing. Every error must name the file, the
section ([fixed] or [[bayram]] entry N) and the problem.
"""
import dataclasses
import tomllib
from datetime import date, datetime
from pathlib import Path

import pytest

from collection_estimation.hurdle.holidays import (
    BAYRAM_NAMES,
    HolidayTable,
    load_holidays,
    read_bayrams,
    read_fixed_holidays,
)

FILE_NAME = "test_holidays.toml"
REMOVE = object()

# Each value is written as TOML text, exactly as it would appear in the file.
VALID_FIXED = {'"01-01"': '"New Year\'s Day"', '"10-29"': '"Republic Day"'}
VALID_BAYRAMS = [
    {"name": '"Ramazan Bayramı"', "first_day": "2026-03-20", "last_day": "2026-03-22"},
    {"name": '"Kurban Bayramı"', "first_day": "2026-05-27", "last_day": "2026-05-30"},
]


def write_holidays(folder, fixed_changes=None, bayram_changes=None, *, fixed=True,
                   bayrams=True, extra_bayrams=(), top="", extra=""):
    """Write a valid holidays file into `folder` and return its path.

    fixed_changes: {key: toml_text or REMOVE} for [fixed].
    bayram_changes: {(entry index from 0, key): toml_text or REMOVE} for the [[bayram]] entries.
    fixed / bayrams: False leaves that section out.  extra_bayrams: more entries (dicts).
    top: text before the first section.  extra: text after the last section.
    """
    fixed_table = dict(VALID_FIXED)
    for key, value in (fixed_changes or {}).items():
        if value is REMOVE:
            del fixed_table[key]
        else:
            fixed_table[key] = value
    entries = [dict(entry) for entry in VALID_BAYRAMS] + [dict(e) for e in extra_bayrams]
    for (index, key), value in (bayram_changes or {}).items():
        if value is REMOVE:
            del entries[index][key]
        else:
            entries[index][key] = value

    lines = [top] if top else []
    if fixed:
        lines.append("[fixed]")
        lines += [f"{key} = {value}" for key, value in fixed_table.items()]
        lines.append("")
    if bayrams:
        for entry in entries:
            lines.append("[[bayram]]")
            lines += [f"{key} = {value}" for key, value in entry.items()]
            lines.append("")
    lines.append(extra)
    path = Path(folder) / FILE_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load(tmp_path, *args, **kwargs):
    return load_holidays(write_holidays(tmp_path, *args, **kwargs))


def raises_naming(tmp_path, *names, **kwargs):
    """Load a broken file; check it raises ValueError naming the file and each of `names`."""
    with pytest.raises(ValueError) as error:
        load(tmp_path, **kwargs)
    message = str(error.value)
    assert FILE_NAME in message, message
    for name in names:
        assert name in message, f"{name!r} not in {message!r}"
    return message


# --- happy path --------------------------------------------------------------------------

def test_returns_a_holiday_table(tmp_path):
    assert isinstance(load(tmp_path), HolidayTable)


def test_fixed_is_month_day_to_name(tmp_path):
    assert load(tmp_path).fixed == {(1, 1): "New Year's Day", (10, 29): "Republic Day"}


def test_fixed_keys_are_int_pairs(tmp_path):
    for key in load(tmp_path).fixed:
        assert type(key) is tuple and [type(part) for part in key] == [int, int]


def test_fixed_keeps_the_file_order(tmp_path):
    # Remove 01-01, then add it again: it is written after 10-29.
    table = load(tmp_path, {'"01-01"': REMOVE, "'01-01'": '"New Year\'s Day"'})
    assert list(table.fixed) == [(10, 29), (1, 1)]


def test_bayrams_are_first_last_name_in_file_order(tmp_path):
    assert load(tmp_path).bayrams == (
        (date(2026, 3, 20), date(2026, 3, 22), "Ramazan Bayramı"),
        (date(2026, 5, 27), date(2026, 5, 30), "Kurban Bayramı"),
    )


def test_bayrams_is_a_tuple_of_tuples(tmp_path):
    bayrams = load(tmp_path).bayrams
    assert type(bayrams) is tuple
    assert all(type(entry) is tuple for entry in bayrams)


def test_bayram_days_are_plain_dates(tmp_path):
    for first_day, last_day, _ in load(tmp_path).bayrams:
        assert type(first_day) is date and type(last_day) is date


def test_a_one_day_bayram_is_accepted(tmp_path):
    table = load(tmp_path, bayram_changes={(0, "last_day"): "2026-03-20"})
    assert table.bayrams[0] == (date(2026, 3, 20), date(2026, 3, 20), "Ramazan Bayramı")


def test_a_single_fixed_holiday_is_enough(tmp_path):
    assert load(tmp_path, {'"01-01"': REMOVE}).fixed == {(10, 29): "Republic Day"}


def test_a_single_bayram_is_enough_for_the_file(tmp_path):
    # Coverage of both Bayrams per year is checked by check_bayrams_cover, not on reading.
    with_one = write_holidays(tmp_path)
    text = with_one.read_text(encoding="utf-8")
    kurban = text.index('[[bayram]]\nname = "Kurban')
    with_one.write_text(text[:kurban], encoding="utf-8")
    assert [name for _, _, name in load_holidays(with_one).bayrams] == ["Ramazan Bayramı"]


def test_bayrams_of_several_years_load(tmp_path):
    table = load(tmp_path, extra_bayrams=[
        {"name": '"Ramazan Bayramı"', "first_day": "2027-03-09", "last_day": "2027-03-11"}])
    assert len(table.bayrams) == 3


def test_29_february_is_accepted(tmp_path):
    assert load(tmp_path, {'"02-29"': '"Leap Day"'}).fixed[(2, 29)] == "Leap Day"


def test_a_name_with_turkish_letters_is_kept_exactly(tmp_path):
    name = "Atatürk'ü Anma, Gençlik ve Spor Bayramı"
    table = load(tmp_path, {'"05-19"': f'"{name}"'})
    assert table.fixed[(5, 19)] == name


def test_the_path_may_be_a_string(tmp_path):
    path = write_holidays(tmp_path)
    assert load_holidays(str(path)) == load_holidays(path)


def test_loading_twice_gives_equal_tables(tmp_path):
    path = write_holidays(tmp_path)
    assert load_holidays(path) == load_holidays(path)


@pytest.mark.parametrize("attribute", ["fixed", "bayrams"])
def test_the_result_is_frozen(tmp_path, attribute):
    table = load(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(table, attribute, None)


# --- the file itself ---------------------------------------------------------------------

def test_a_toml_syntax_error_raises_the_toml_error(tmp_path):
    with pytest.raises(tomllib.TOMLDecodeError):
        load(tmp_path, extra="this is = = not toml")


def test_an_unclosed_string_raises_the_toml_error(tmp_path):
    with pytest.raises(tomllib.TOMLDecodeError):
        load(tmp_path, {'"10-29"': '"Republic Day'})


def test_the_same_key_twice_in_fixed_raises_the_toml_error(tmp_path):
    path = write_holidays(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "[fixed]\n", '[fixed]\n"10-29" = "Other"\n', 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_holidays(path)


def test_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_holidays(tmp_path / "missing.toml")


def test_an_empty_file_names_both_sections(tmp_path):
    path = tmp_path / FILE_NAME
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_holidays(path)
    message = str(error.value)
    assert FILE_NAME in message
    assert "'fixed'" in message and "'bayram'" in message
    assert "missing" in message


# --- top-level keys ----------------------------------------------------------------------

def test_a_missing_fixed_section_raises(tmp_path):
    message = raises_naming(tmp_path, "'fixed'", fixed=False)
    assert "missing" in message


def test_a_missing_bayram_section_raises(tmp_path):
    message = raises_naming(tmp_path, "'bayram'", bayrams=False)
    assert "missing" in message


def test_an_unknown_top_level_section_raises(tmp_path):
    message = raises_naming(tmp_path, "'extra'", extra="[extra]\nx = 1\n")
    assert "unknown" in message


def test_an_unknown_top_level_value_raises(tmp_path):
    message = raises_naming(tmp_path, "'version'", top="version = 1")
    assert "unknown" in message


def test_a_misspelt_section_names_the_missing_and_the_unknown_one(tmp_path):
    message = raises_naming(tmp_path, "'bayram'", "'bayrams'", bayrams=False,
                            extra='[[bayrams]]\nname = "Kurban Bayramı"\n'
                                  'first_day = 2026-05-27\nlast_day = 2026-05-30\n')
    assert "missing" in message and "unknown" in message


def test_the_top_level_message_lists_the_expected_sections(tmp_path):
    message = raises_naming(tmp_path, fixed=False)
    assert "['bayram', 'fixed']" in message.split("expected")[-1]


# --- [fixed] -----------------------------------------------------------------------------

def test_an_empty_fixed_table_raises(tmp_path):
    raises_naming(tmp_path, "[fixed]", "{}",
                  fixed_changes={'"01-01"': REMOVE, '"10-29"': REMOVE})


@pytest.mark.parametrize("value", ["5", '"01-01"', '["01-01"]', "true"],
                         ids=["number", "text", "list", "bool"])
def test_fixed_given_as_a_plain_value_raises(tmp_path, value):
    raises_naming(tmp_path, "[fixed]", "not a table", fixed=False, top=f"fixed = {value}")


def test_fixed_given_as_a_list_of_tables_raises(tmp_path):
    raises_naming(tmp_path, "[fixed]", "not a table", fixed=False,
                  extra='[[fixed]]\n"01-01" = "New Year\'s Day"\n')


@pytest.mark.parametrize("key", ['"02-30"', '"13-01"', '"00-01"', '"01-00"', '"04-31"',
                                 '"01-32"', '"1029"', '"x"', '"10-x"', '"01-01-01"',
                                 '"10/29"', '""'],
                         ids=["30-february", "month-13", "month-0", "day-0", "31-april",
                              "day-32", "no-dash", "letter", "letter-day", "three-parts",
                              "slash", "empty"])
def test_a_key_that_is_not_a_real_month_and_day_raises(tmp_path, key):
    written = key.strip('"')
    message = raises_naming(tmp_path, "[fixed]", f"'{written}'", "not a month and day",
                            fixed_changes={key: '"Some Day"'})
    assert "[[bayram]]" not in message


@pytest.mark.parametrize("value", ['""', '"   "', "5", "true", '["Day"]', "2026-10-29",
                                   "{ a = 1 }"],
                         ids=["empty", "blank", "number", "bool", "list", "date", "table"])
def test_a_name_that_is_not_non_empty_text_raises(tmp_path, value):
    raises_naming(tmp_path, "[fixed]", "'10-29'", "not a text",
                  fixed_changes={'"10-29"': value})


def test_a_bad_key_is_reported_before_a_bad_name(tmp_path):
    raises_naming(tmp_path, "'02-30'", "not a month and day",
                  fixed_changes={'"02-30"': "5"})


def test_a_problem_in_fixed_is_reported_before_one_in_bayram(tmp_path):
    message = raises_naming(tmp_path, "[fixed]", fixed_changes={'"13-01"': '"Day"'},
                            bayram_changes={(0, "name"): '"Seker Bayramı"'})
    assert "[[bayram]]" not in message


# --- [[bayram]] --------------------------------------------------------------------------

def test_an_empty_bayram_list_raises(tmp_path):
    raises_naming(tmp_path, "[[bayram]]", "not a list", bayrams=False, top="bayram = []")


@pytest.mark.parametrize("value", ["5", '"Kurban Bayramı"', "true"],
                         ids=["number", "text", "bool"])
def test_bayram_given_as_a_plain_value_raises(tmp_path, value):
    raises_naming(tmp_path, "[[bayram]]", "not a list", bayrams=False, top=f"bayram = {value}")


def test_bayram_given_as_a_single_table_raises(tmp_path):
    raises_naming(tmp_path, "[[bayram]]", "not a list", bayrams=False,
                  extra='[bayram]\nname = "Kurban Bayramı"\n'
                        'first_day = 2026-05-27\nlast_day = 2026-05-30\n')


def test_a_bayram_entry_that_is_not_a_table_raises_naming_its_number(tmp_path):
    raises_naming(tmp_path, "[[bayram]] entry 1", "not a section", bayrams=False,
                  top="bayram = [5]")


@pytest.mark.parametrize("key", ["name", "first_day", "last_day"])
def test_a_missing_bayram_key_raises_naming_entry_and_key(tmp_path, key):
    message = raises_naming(tmp_path, "[[bayram]] entry 2", f"'{key}'",
                            bayram_changes={(1, key): REMOVE})
    assert "missing" in message


def test_an_unknown_bayram_key_raises_naming_entry_and_key(tmp_path):
    message = raises_naming(tmp_path, "[[bayram]] entry 1", "'note'",
                            bayram_changes={(0, "note"): '"approximate"'})
    assert "unknown" in message


def test_a_misspelt_bayram_key_names_both(tmp_path):
    message = raises_naming(tmp_path, "[[bayram]] entry 2", "'last_day'", "'lastday'",
                            bayram_changes={(1, "last_day"): REMOVE,
                                            (1, "lastday"): "2026-05-30"})
    assert "missing" in message and "unknown" in message


def test_the_entry_number_counts_from_one_in_file_order(tmp_path):
    raises_naming(tmp_path, "[[bayram]] entry 3", extra_bayrams=[
        {"name": '"Ramazan Bayramı"', "first_day": "2027-03-09"}])


@pytest.mark.parametrize("value", ['"Seker Bayramı"', '"ramazan bayramı"', '"Ramazan Bayrami"',
                                   '"Ramazan Bayramı "', '""', "5", '["Kurban Bayramı"]'],
                         ids=["other-name", "lower-case", "dotless-i-missing",
                              "trailing-space", "empty", "number", "list"])
def test_a_name_not_in_bayram_names_raises(tmp_path, value):
    raises_naming(tmp_path, "[[bayram]] entry 1", "name is", "not one of",
                  bayram_changes={(0, "name"): value})


def test_the_bad_name_message_lists_the_bayram_names(tmp_path):
    message = raises_naming(tmp_path, bayram_changes={(0, "name"): '"Seker Bayramı"'})
    for name in BAYRAM_NAMES:
        assert name in message


@pytest.mark.parametrize("key", ["first_day", "last_day"])
@pytest.mark.parametrize("value", ['"2026-03-20"', "2026-03-20T10:00:00", "2026-03-20T00:00:00",
                                   "2026-03-20 10:00:00Z", "20260320", "10:00:00"],
                         ids=["string", "datetime", "midnight-datetime", "datetime-with-zone",
                              "number", "time"])
def test_a_day_that_is_not_a_toml_date_raises(tmp_path, key, value):
    raises_naming(tmp_path, "[[bayram]] entry 1", key, "not a date",
                  bayram_changes={(0, key): value})


def test_first_day_after_last_day_raises_naming_both(tmp_path):
    raises_naming(tmp_path, "[[bayram]] entry 2", "2026-05-31", "2026-05-30", "is after",
                  bayram_changes={(1, "first_day"): "2026-05-31"})


def test_first_day_a_year_after_last_day_raises(tmp_path):
    raises_naming(tmp_path, "[[bayram]] entry 1", "is after",
                  bayram_changes={(0, "first_day"): "2027-03-20"})


def test_the_same_bayram_and_year_twice_raises_listed_twice(tmp_path):
    raises_naming(tmp_path, "[[bayram]] entry 3", "Kurban Bayramı 2026", "listed twice",
                  extra_bayrams=[{"name": '"Kurban Bayramı"', "first_day": "2026-06-01",
                                  "last_day": "2026-06-03"}])


def test_the_same_entry_copied_twice_raises_listed_twice(tmp_path):
    raises_naming(tmp_path, "[[bayram]] entry 3", "listed twice",
                  extra_bayrams=[dict(VALID_BAYRAMS[0])])


def test_the_same_bayram_in_two_years_is_fine(tmp_path):
    table = load(tmp_path, extra_bayrams=[
        {"name": '"Kurban Bayramı"', "first_day": "2027-05-16", "last_day": "2027-05-19"}])
    assert [name for _, _, name in table.bayrams].count("Kurban Bayramı") == 2


def test_the_name_is_checked_before_the_dates(tmp_path):
    message = raises_naming(tmp_path, "[[bayram]] entry 1", "not one of",
                            bayram_changes={(0, "name"): '"Seker Bayramı"',
                                            (0, "first_day"): '"not a date"'})
    assert "not a date" not in message


# --- read_fixed_holidays directly --------------------------------------------------------

def test_read_fixed_holidays_returns_month_day_to_name():
    assert read_fixed_holidays({"01-01": "A", "10-29": "B"}, "here") == {(1, 1): "A",
                                                                          (10, 29): "B"}


def test_read_fixed_holidays_accepts_29_february():
    assert read_fixed_holidays({"02-29": "Leap Day"}, "here") == {(2, 29): "Leap Day"}


@pytest.mark.parametrize("key", ["02-30", "13-01", "1029", "x", "", "00-00"])
def test_read_fixed_holidays_rejects_a_bad_key_naming_where_and_key(key):
    with pytest.raises(ValueError) as error:
        read_fixed_holidays({key: "Day"}, "my_file [fixed]")
    message = str(error.value)
    assert message.startswith("my_file [fixed]")
    assert f"'{key}'" in message


@pytest.mark.parametrize("table", [{}, [], None, "01-01", 5, [("01-01", "Day")]])
def test_read_fixed_holidays_rejects_an_empty_or_non_table(table):
    with pytest.raises(ValueError, match=r"^here is .*not a table"):
        read_fixed_holidays(table, "here")


@pytest.mark.parametrize("name", ["", "  ", None, 5, ["Day"]])
def test_read_fixed_holidays_rejects_a_bad_name(name):
    with pytest.raises(ValueError, match=r"^here: '01-01' has the name .* not a text"):
        read_fixed_holidays({"01-01": name}, "here")


# A key must be exactly two ASCII digits, a hyphen and two ASCII digits. Without that, int()
# would accept padding, signs, underscores and non-ASCII digits, and "1-1" would silently be
# the same holiday as "01-01".
NOT_TWO_DIGITS_HYPHEN_TWO_DIGITS = [
    "1-1", "01-1", "1-01", " 01-01", "01-01 ", "10 - 29", "+1-01", "1_0-29",
    "٠١-٠١",   # Arabic-Indic digits
    "０１-０１",   # full-width digits
    "001-01",
]
NOT_TWO_DIGITS_IDS = [
    "one-digit-both", "one-digit-day", "one-digit-month", "leading-space", "trailing-space",
    "spaces-around-hyphen", "plus-sign", "underscore", "arabic-indic-digits",
    "full-width-digits", "three-digit-month",
]


@pytest.mark.parametrize("key", NOT_TWO_DIGITS_HYPHEN_TWO_DIGITS, ids=NOT_TWO_DIGITS_IDS)
def test_read_fixed_holidays_rejects_a_key_not_two_digits_hyphen_two_digits(key):
    with pytest.raises(ValueError) as error:
        read_fixed_holidays({key: "Day"}, "my_file [fixed]")
    assert str(error.value) == (f"my_file [fixed]: '{key}' is not a month and day such as "
                                f"\"10-29\"")


@pytest.mark.parametrize("key", NOT_TWO_DIGITS_HYPHEN_TWO_DIGITS, ids=NOT_TWO_DIGITS_IDS)
def test_a_key_not_two_digits_hyphen_two_digits_in_the_file_raises(tmp_path, key):
    message = raises_naming(tmp_path, "[fixed]", f"'{key}'", "not a month and day",
                            fixed_changes={f'"{key}"': '"Some Day"'})
    assert "[[bayram]]" not in message


@pytest.mark.parametrize("key, month_day", [("01-01", (1, 1)), ("12-31", (12, 31)),
                                            ("02-29", (2, 29))])
def test_read_fixed_holidays_still_accepts_two_digits_hyphen_two_digits(key, month_day):
    assert read_fixed_holidays({key: "Day"}, "here") == {month_day: "Day"}


def test_a_short_key_beside_its_padded_twin_raises_instead_of_a_silent_duplicate(tmp_path):
    # "01-01" and "1-1" are different TOML keys; read loosely, both would be (1, 1) and the
    # second name would quietly replace the first.
    message = raises_naming(tmp_path, "[fixed]", "'1-1'", "not a month and day",
                            fixed_changes={'"1-1"': '"Another New Year"'})
    assert "'01-01'" not in message


def test_read_fixed_holidays_does_not_modify_its_input():
    table = {"01-01": "A", "10-29": "B"}
    read_fixed_holidays(table, "here")
    assert table == {"01-01": "A", "10-29": "B"}


def test_read_fixed_holidays_result_does_not_follow_the_input():
    table = {"01-01": "A"}
    fixed = read_fixed_holidays(table, "here")
    table["01-01"] = "changed"
    assert fixed == {(1, 1): "A"}


# --- read_bayrams directly ---------------------------------------------------------------

def bayram(name="Kurban Bayramı", first_day=date(2026, 5, 27), last_day=date(2026, 5, 30)):
    return {"name": name, "first_day": first_day, "last_day": last_day}


def test_read_bayrams_returns_a_tuple_of_first_last_name():
    assert read_bayrams([bayram()], "here") == (
        (date(2026, 5, 27), date(2026, 5, 30), "Kurban Bayramı"),)


@pytest.mark.parametrize("entries", [[], None, {}, bayram(), "Kurban Bayramı", 5])
def test_read_bayrams_rejects_an_empty_or_non_list(entries):
    with pytest.raises(ValueError, match=r"^here is .*not a list"):
        read_bayrams(entries, "here")


def test_read_bayrams_names_the_entry_number():
    with pytest.raises(ValueError, match=r"^here entry 2: "):
        read_bayrams([bayram(), bayram(name="Seker Bayramı")], "here")


def test_read_bayrams_rejects_a_datetime():
    with pytest.raises(ValueError, match=r"^here entry 1: first_day .* not a date"):
        read_bayrams([bayram(first_day=datetime(2026, 5, 27))], "here")


def test_read_bayrams_rejects_first_day_after_last_day():
    with pytest.raises(ValueError, match="here entry 1: first_day 2026-05-31 is after "
                                         "last_day 2026-05-30"):
        read_bayrams([bayram(first_day=date(2026, 5, 31))], "here")


def test_read_bayrams_rejects_the_same_name_and_year_twice():
    with pytest.raises(ValueError, match="here entry 2: Kurban Bayramı 2026 is listed twice"):
        read_bayrams([bayram(), bayram(first_day=date(2026, 1, 1),
                                       last_day=date(2026, 1, 2))], "here")


def test_read_bayrams_year_is_that_of_the_first_day():
    # A Bayram from 30 December 2025 to 2 January 2026 belongs to 2025, so a 2026 one of the
    # same name is not a repeat.
    result = read_bayrams([bayram(first_day=date(2025, 12, 30), last_day=date(2026, 1, 2)),
                           bayram()], "here")
    assert len(result) == 2


def test_read_bayrams_does_not_modify_its_input():
    entries = [bayram()]
    read_bayrams(entries, "here")
    assert entries == [bayram()]
