"""Tests for `hurdle/settings.py`: `load_settings` and its helpers `check_keys`,
`path_setting`, `date_setting` and `text_list_setting`.

Every settings file is written into `tmp_path` by `write_settings`, which starts from a valid
file and changes, removes or adds one thing.
"""
import dataclasses
import tomllib
from datetime import date, datetime
from pathlib import Path

import pytest

from collection_estimation.hurdle.settings import (
    CompanySettings,
    DataSettings,
    DebugSettings,
    Settings,
    check_keys,
    date_setting,
    load_settings,
    path_setting,
    text_list_setting,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_SETTINGS = REPO_ROOT / "settings" / "synthetic.toml"

FILE_NAME = "test_run.toml"
REMOVE = object()

# Each value is written as TOML text, exactly as it would appear in the file.
VALID = {
    "data": {
        "corpus_folder": '"data/raw/corpus"',
        "ledger_file": '"data/raw/ledger.xlsx"',
        "ledger_as_of": "2026-08-28",
        "cache_folder": '"data/interim"',
    },
    "companies": {"known": '["TR02", "TR04"]'},
    "debug": {"one_day": "2025-06-02"},
}


def write_settings(folder, changes=None, drop_section=None, top="", extra=""):
    """Write a valid settings file into `folder` and return its path.

    changes: {(section, key): toml_text or REMOVE}; a key not in VALID is added.
    drop_section: a section left out entirely.  top: text before the first section.
    extra: text after the last section.
    """
    sections = {name: dict(keys) for name, keys in VALID.items()}
    for (section, key), value in (changes or {}).items():
        if value is REMOVE:
            del sections[section][key]
        else:
            sections[section][key] = value
    lines = [top] if top else []
    for name, keys in sections.items():
        if name == drop_section:
            continue
        lines.append(f"[{name}]")
        lines += [f"{key} = {value}" for key, value in keys.items()]
        lines.append("")
    lines.append(extra)
    path = Path(folder) / FILE_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    return root


def load(tmp_path, repo, **kwargs):
    return load_settings(write_settings(tmp_path, **kwargs), repo)


def raises_naming(tmp_path, repo, *names, **kwargs):
    """Load a broken file; check it raises ValueError naming the file and every one of `names`."""
    with pytest.raises(ValueError) as error:
        load(tmp_path, repo, **kwargs)
    message = str(error.value)
    assert FILE_NAME in message
    for name in names:
        assert name in message, f"{name!r} not in {message!r}"
    return message


# --- happy path ----------------------------------------------------------------------

def test_returns_a_settings_object(tmp_path, repo):
    settings = load(tmp_path, repo)
    assert isinstance(settings, Settings)
    assert isinstance(settings.data, DataSettings)
    assert isinstance(settings.companies, CompanySettings)
    assert isinstance(settings.debug, DebugSettings)


def test_paths_are_read_from_the_repo_root(tmp_path, repo):
    data = load(tmp_path, repo).data
    assert data.corpus_folder == (repo / "data" / "raw" / "corpus").resolve()
    assert data.ledger_file == (repo / "data" / "raw" / "ledger.xlsx").resolve()
    assert data.cache_folder == (repo / "data" / "interim").resolve()


def test_paths_are_absolute_path_objects(tmp_path, repo):
    data = load(tmp_path, repo).data
    for path in (data.corpus_folder, data.ledger_file, data.cache_folder):
        assert isinstance(path, Path)
        assert path.is_absolute()


def test_ledger_as_of_is_a_date(tmp_path, repo):
    value = load(tmp_path, repo).data.ledger_as_of
    assert type(value) is date
    assert value == date(2026, 8, 28)


def test_one_day_is_a_date(tmp_path, repo):
    value = load(tmp_path, repo).debug.one_day
    assert type(value) is date
    assert value == date(2025, 6, 2)


def test_known_is_a_tuple_of_texts_in_file_order(tmp_path, repo):
    known = load(tmp_path, repo).companies.known
    assert known == ("TR02", "TR04")
    assert type(known) is tuple


def test_a_single_known_company_is_fine(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("companies", "known"): '["TR02"]'})
    assert settings.companies.known == ("TR02",)


def test_settings_file_is_the_resolved_path(tmp_path, repo):
    path = write_settings(tmp_path)
    assert load_settings(path, repo).settings_file == path.resolve()


def test_settings_file_given_relative_is_made_absolute(tmp_path, repo, monkeypatch):
    path = write_settings(tmp_path)
    monkeypatch.chdir(tmp_path)
    settings = load_settings(Path(FILE_NAME), repo)
    assert settings.settings_file.is_absolute()
    assert settings.settings_file == path.resolve()


def test_settings_file_and_repo_root_may_be_strings(tmp_path, repo):
    path = write_settings(tmp_path)
    assert load_settings(str(path), str(repo)) == load_settings(path, repo)


def test_loading_twice_gives_equal_settings(tmp_path, repo):
    path = write_settings(tmp_path)
    assert load_settings(path, repo) == load_settings(path, repo)


@pytest.mark.parametrize("attribute", ["settings_file", "data", "companies", "debug"])
def test_settings_is_frozen(tmp_path, repo, attribute):
    settings = load(tmp_path, repo)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(settings, attribute, None)


@pytest.mark.parametrize("part, attribute", [
    ("data", "corpus_folder"), ("data", "ledger_file"), ("data", "ledger_as_of"),
    ("data", "cache_folder"), ("companies", "known"), ("debug", "one_day"),
])
def test_every_section_is_frozen(tmp_path, repo, part, attribute):
    section = getattr(load(tmp_path, repo), part)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(section, attribute, None)


# --- paths ---------------------------------------------------------------------------

def test_relative_paths_do_not_depend_on_the_working_directory(tmp_path, repo, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    data = load(tmp_path, repo).data
    assert data.corpus_folder == (repo / "data" / "raw" / "corpus").resolve()
    assert elsewhere.resolve() not in data.corpus_folder.parents


def test_an_absolute_path_is_kept(tmp_path, repo):
    absolute = (tmp_path / "somewhere" / "ledger.xlsx").resolve()
    settings = load(tmp_path, repo,
                    changes={("data", "ledger_file"): f"'{absolute}'"})  # literal string
    assert settings.data.ledger_file == absolute


def test_dot_dot_segments_are_resolved(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("data", "cache_folder"): '"../shared/cache"'})
    assert settings.data.cache_folder == (tmp_path / "shared" / "cache").resolve()
    assert ".." not in settings.data.cache_folder.parts


def test_paths_need_not_exist(tmp_path, repo):
    data = load(tmp_path, repo).data
    assert not data.corpus_folder.exists()
    assert not data.ledger_file.exists()


# --- strictness: sections --------------------------------------------------------------

@pytest.mark.parametrize("section", ["data", "companies", "debug"])
def test_a_missing_section_raises(tmp_path, repo, section):
    message = raises_naming(tmp_path, repo, section, drop_section=section)
    assert "missing" in message


def test_an_unknown_section_raises(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "extra", extra="[extra]\nx = 1\n")
    assert "unknown" in message


def test_an_unknown_top_level_value_raises(tmp_path, repo):
    raises_naming(tmp_path, repo, "version", top="version = 2")


def test_a_section_given_as_a_plain_value_raises(tmp_path, repo):
    raises_naming(tmp_path, repo, "[debug]", "5", drop_section="debug", top="debug = 5")


def test_a_section_given_as_a_list_of_tables_raises(tmp_path, repo):
    raises_naming(tmp_path, repo, "[debug]", drop_section="debug",
                  extra="[[debug]]\none_day = 2025-06-02\n")


# --- strictness: settings --------------------------------------------------------------

@pytest.mark.parametrize("section, key", [
    ("data", "corpus_folder"), ("data", "ledger_file"), ("data", "ledger_as_of"),
    ("data", "cache_folder"), ("companies", "known"), ("debug", "one_day"),
])
def test_a_missing_setting_raises(tmp_path, repo, section, key):
    message = raises_naming(tmp_path, repo, f"[{section}]", key,
                            changes={(section, key): REMOVE})
    assert "missing" in message


@pytest.mark.parametrize("section", ["data", "companies", "debug"])
def test_an_unknown_setting_raises(tmp_path, repo, section):
    message = raises_naming(tmp_path, repo, f"[{section}]", "surprise",
                            changes={(section, "surprise"): "1"})
    assert "unknown" in message


def test_a_sub_table_counts_as_an_unknown_setting(tmp_path, repo):
    raises_naming(tmp_path, repo, "[data]", "nested", extra="[data.nested]\nx = 1\n")


def test_a_typo_names_the_missing_and_the_unknown_setting_in_one_message(tmp_path, repo):
    message = raises_naming(
        tmp_path, repo, "[data]", "'ledger_as_of'", "'ledger_as_off'",
        changes={("data", "ledger_as_of"): REMOVE, ("data", "ledger_as_off"): "2026-08-28"})
    assert "missing" in message and "unknown" in message


def test_a_typo_message_lists_the_expected_settings(tmp_path, repo):
    message = raises_naming(
        tmp_path, repo,
        changes={("data", "ledger_as_of"): REMOVE, ("data", "ledger_as_off"): "2026-08-28"})
    for key in ["cache_folder", "corpus_folder", "ledger_as_of", "ledger_file"]:
        assert key in message.split("expected")[-1]


def test_a_problem_in_a_later_section_names_that_section(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[debug]",
                            changes={("debug", "one_dya"): "2025-06-02",
                                     ("debug", "one_day"): REMOVE})
    assert "[data]" not in message


# --- types: paths ----------------------------------------------------------------------

@pytest.mark.parametrize("key", ["corpus_folder", "ledger_file", "cache_folder"])
@pytest.mark.parametrize("bad", ["5", '""', '"   "', "true", '["data"]', "2026-08-28"],
                         ids=["number", "empty", "blank", "bool", "list", "date"])
def test_a_path_that_is_not_text_or_is_blank_raises(tmp_path, repo, key, bad):
    raises_naming(tmp_path, repo, "[data]", key, changes={("data", key): bad})


# --- types: dates ----------------------------------------------------------------------

@pytest.mark.parametrize("section, key", [("data", "ledger_as_of"), ("debug", "one_day")])
@pytest.mark.parametrize("bad", [
    '"2026-08-28"', "2026-08-28T10:00:00", "2026-08-28T00:00:00", "2026-08-28 10:00:00Z",
    "20260828", "10:00:00",
], ids=["string", "datetime", "midnight-datetime", "datetime-with-zone", "number", "time"])
def test_a_date_that_is_not_a_toml_date_raises(tmp_path, repo, section, key, bad):
    raises_naming(tmp_path, repo, f"[{section}]", key, changes={(section, key): bad})


# --- types: known ----------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    '"TR02"', "[]", '["TR02", 4]', '["TR02", ""]', '["TR02", "  "]', '[["TR02"]]',
    "{ a = 1 }",
], ids=["text", "empty-list", "number-item", "empty-item", "blank-item", "nested-list",
        "inline-table"])
def test_a_bad_known_raises(tmp_path, repo, bad):
    raises_naming(tmp_path, repo, "[companies]", "known",
                  changes={("companies", "known"): bad})


# --- the file itself -----------------------------------------------------------------

def test_a_toml_syntax_error_raises_the_toml_error(tmp_path, repo):
    with pytest.raises(tomllib.TOMLDecodeError):
        load(tmp_path, repo, extra="this is = = not toml")


def test_an_unclosed_string_raises_the_toml_error(tmp_path, repo):
    with pytest.raises(tomllib.TOMLDecodeError):
        load(tmp_path, repo, changes={("data", "corpus_folder"): '"data/raw'})


def test_a_missing_file_raises_file_not_found(tmp_path, repo):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.toml", repo)


def test_an_empty_file_names_every_section(tmp_path, repo):
    path = tmp_path / FILE_NAME
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_settings(path, repo)
    for section in ["data", "companies", "debug"]:
        assert section in str(error.value)


# --- check_keys ----------------------------------------------------------------------

def test_check_keys_passes_on_exactly_the_expected_keys():
    assert check_keys({"a": 1, "b": 2}, {"a", "b"}, "where") is None


def test_check_keys_passes_on_an_empty_table_when_nothing_is_expected():
    check_keys({}, set(), "where")


@pytest.mark.parametrize("not_a_table", [5, "a", [{"a": 1}], None])
def test_check_keys_rejects_a_non_table(not_a_table):
    with pytest.raises(ValueError, match="where.*not a section"):
        check_keys(not_a_table, {"a"}, "where")


def test_check_keys_message_starts_with_where():
    with pytest.raises(ValueError) as error:
        check_keys({"a": 1}, {"b"}, "f.toml [s]")
    assert str(error.value).startswith("f.toml [s]")


def test_check_keys_lists_missing_and_unknown_sorted():
    with pytest.raises(ValueError) as error:
        check_keys({"z": 1, "y": 1}, {"b", "a"}, "w")
    message = str(error.value)
    assert message == "w: missing ['a', 'b']; unknown ['y', 'z']; expected exactly ['a', 'b']"


def test_check_keys_message_with_only_missing_keys():
    with pytest.raises(ValueError) as error:
        check_keys({"a": 1}, {"a", "b"}, "w")
    assert str(error.value) == "w: missing ['b']; expected exactly ['a', 'b']"


def test_check_keys_message_with_only_unknown_keys():
    with pytest.raises(ValueError) as error:
        check_keys({"a": 1, "c": 2}, {"a"}, "w")
    assert str(error.value) == "w: unknown ['c']; expected exactly ['a']"


def test_check_keys_message_does_not_say_settings():
    # The wording is neutral: check_keys also checks sections, which are not settings.
    with pytest.raises(ValueError) as error:
        check_keys({"z": 1}, {"a"}, "w")
    assert "settings" not in str(error.value)


def test_check_keys_does_not_modify_the_table():
    table = {"a": 1}
    with pytest.raises(ValueError):
        check_keys(table, {"b"}, "w")
    assert table == {"a": 1}


# --- path_setting --------------------------------------------------------------------

def test_path_setting_joins_a_relative_path_to_the_root(tmp_path):
    assert path_setting({"p": "x/y"}, "p", tmp_path, "w") == (tmp_path / "x" / "y").resolve()


def test_path_setting_keeps_an_absolute_path(tmp_path):
    absolute = (tmp_path / "abs").resolve()
    assert path_setting({"p": str(absolute)}, "p", tmp_path / "root", "w") == absolute


def test_path_setting_accepts_the_root_as_a_string(tmp_path):
    assert path_setting({"p": "x"}, "p", str(tmp_path), "w") == (tmp_path / "x").resolve()


@pytest.mark.parametrize("bad", [5, None, "", " \t", Path("x")],
                         ids=["number", "none", "empty", "blank", "path-object"])
def test_path_setting_rejects_a_non_text_or_blank_value(tmp_path, bad):
    with pytest.raises(ValueError, match="w: p is"):
        path_setting({"p": bad}, "p", tmp_path, "w")


# --- date_setting --------------------------------------------------------------------

def test_date_setting_returns_a_date_unchanged():
    day = date(2026, 8, 28)
    assert date_setting({"d": day}, "d", "w") is day


@pytest.mark.parametrize("bad", [datetime(2026, 8, 28), datetime(2026, 8, 28, 10),
                                 "2026-08-28", 20260828, None],
                         ids=["midnight-datetime", "datetime", "text", "number", "none"])
def test_date_setting_rejects_anything_but_a_date(bad):
    with pytest.raises(ValueError, match="w: d is"):
        date_setting({"d": bad}, "d", "w")


# --- text_list_setting ---------------------------------------------------------------

def test_text_list_setting_returns_a_tuple():
    assert text_list_setting({"k": ["A", "B"]}, "k", "w") == ("A", "B")


def test_text_list_setting_does_not_modify_the_list():
    value = ["A", "B"]
    result = text_list_setting({"k": value}, "k", "w")
    assert value == ["A", "B"]
    assert result is not value


def test_text_list_setting_result_does_not_follow_later_changes_to_the_list():
    value = ["A"]
    result = text_list_setting({"k": value}, "k", "w")
    value.append("B")
    assert result == ("A",)


@pytest.mark.parametrize("bad", ["AB", ("A",), [], None],
                         ids=["text", "tuple", "empty", "none"])
def test_text_list_setting_rejects_a_non_list_or_empty_list(bad):
    with pytest.raises(ValueError, match="w: k is"):
        text_list_setting({"k": bad}, "k", "w")


@pytest.mark.parametrize("bad_item", [1, None, "", "  "], ids=["number", "none", "empty", "blank"])
def test_text_list_setting_rejects_a_bad_item(bad_item):
    with pytest.raises(ValueError, match="w: k holds"):
        text_list_setting({"k": ["A", bad_item]}, "k", "w")


def test_text_list_setting_without_duplicates_passes():
    assert text_list_setting({"k": ["TR02", "TR04", "TR05"]}, "k", "w") == (
        "TR02", "TR04", "TR05")


def test_text_list_setting_rejects_a_duplicate():
    with pytest.raises(ValueError) as error:
        text_list_setting({"known": ["TR02", "TR02"]}, "known", "w")
    assert str(error.value) == "w: known lists ['TR02'] more than once"


def test_text_list_setting_names_two_duplicates_sorted_once_each():
    with pytest.raises(ValueError) as error:
        text_list_setting({"k": ["TR06", "TR02", "TR06", "TR04", "TR02"]}, "k", "w")
    message = str(error.value)
    assert message == "w: k lists ['TR02', 'TR06'] more than once"
    assert message.count("TR02") == 1 and message.count("TR06") == 1
    assert "TR04" not in message


def test_text_list_setting_names_a_triple_once():
    with pytest.raises(ValueError) as error:
        text_list_setting({"k": ["TR02", "TR02", "TR02"]}, "k", "w")
    assert str(error.value) == "w: k lists ['TR02'] more than once"


@pytest.mark.parametrize("variants", [["TR02", "tr02"], ["TR02", "TR02 "], ["TR02", " TR02"]],
                         ids=["case", "trailing-space", "leading-space"])
def test_text_list_setting_compares_items_exactly(variants):
    # Pinned: items are compared as written; no case-folding or stripping.
    assert text_list_setting({"k": variants}, "k", "w") == tuple(variants)


def test_text_list_setting_does_not_modify_the_list_when_it_raises():
    value = ["A", "A"]
    with pytest.raises(ValueError):
        text_list_setting({"k": value}, "k", "w")
    assert value == ["A", "A"]


def test_a_duplicate_known_company_in_the_file_raises(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[companies]", "known", "'TR02'",
                            changes={("companies", "known"): '["TR02", "TR04", "TR02"]'})
    assert message.endswith("known lists ['TR02'] more than once")


def test_two_duplicate_known_companies_in_the_file_are_both_named(tmp_path, repo):
    message = raises_naming(
        tmp_path, repo, "[companies]", "['TR02', 'TR04']",
        changes={("companies", "known"): '["TR04", "TR02", "TR04", "TR02", "TR02"]'})
    assert "more than once" in message


def test_case_variants_of_a_known_company_in_the_file_load(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("companies", "known"): '["TR02", "tr02"]'})
    assert settings.companies.known == ("TR02", "tr02")


# --- the repo's own settings file -----------------------------------------------------

@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_settings_load_and_point_at_existing_inputs():
    settings = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT)
    assert settings.data.corpus_folder.is_dir()
    assert settings.data.ledger_file.is_file()


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_settings_hold_the_expected_values():
    settings = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT)
    assert settings.data.ledger_as_of == date(2026, 8, 28)
    assert settings.companies.known == ("TR02", "TR04", "TR05", "TR06", "TR91")
    assert settings.debug.one_day == date(2025, 6, 2)
    assert settings.data.cache_folder.parts[-2:] == ("data", "interim")
    assert settings.data.cache_folder == (REPO_ROOT / "data" / "interim").resolve()
    assert settings.settings_file == SYNTHETIC_SETTINGS.resolve()
