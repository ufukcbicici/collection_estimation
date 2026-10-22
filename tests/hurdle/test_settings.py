"""Tests for `hurdle/settings.py`: `load_settings` and its helpers `check_keys`,
`path_setting`, `date_setting`, `text_list_setting` (with `may_be_empty`), `rows_setting`, `date_format_setting`,
`headers_setting`, `document_types_setting`, `read_collections_layout` and
`read_ledger_layout`.

Every settings file is written into `tmp_path` by `write_settings`, which starts from a valid
file and changes, removes or adds one thing. The layout sub-tables (headers, document_types)
are written as TOML inline tables so that one change can replace a whole sub-table.
"""
import dataclasses
import tomllib
from datetime import date, datetime
from pathlib import Path

import pytest

from collection_estimation.hurdle import read_collections
from collection_estimation.hurdle.read_collections import (
    COLLECTION_FIELDS,
    OPTIONAL_FIELDS,
)
from collection_estimation.hurdle.read_ledger import (
    DOCUMENT_ROLES,
    LEDGER_FIELDS,
    LEDGER_OPTIONAL_FIELDS,
)
from collection_estimation.hurdle.settings import (
    FORMAT_SAMPLE_DATES,
    CollectionsLayout,
    CompanySettings,
    DataSettings,
    CLASS_WEIGHTS,
    DebugSettings,
    DiagnosticSettings,
    HistorySettings,
    LedgerLayout,
    FEATURE_BLOCKS,
    ReceivablesSettings,
    Settings,
    Stage1Settings,
    Stage2Settings,
    STAGE2_LOSSES,
    check_keys,
    class_weight_setting,
    grid_setting,
    feature_blocks_setting,
    leaf_count_setting,
    rate_setting,
    seed_setting,
    stage2_loss_setting,
    positive_number_setting,
    count_setting,
    date_format_setting,
    date_setting,
    day_edges_setting,
    document_types_setting,
    headers_setting,
    load_settings,
    path_setting,
    read_collections_layout,
    read_ledger_layout,
    rows_setting,
    text_list_setting,
    week_list_setting,
    week_setting,
    bool_setting,
    window_list_setting,
)
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_SETTINGS = REPO_ROOT / "settings" / "synthetic.toml"

FILE_NAME = "test_run.toml"
REMOVE = object()


def inline(mapping):
    """A dict of TOML texts -> a TOML inline table, e.g. {"a" = "b", c = ["D"]}."""
    return "{ " + ", ".join(f"{key} = {value}" for key, value in mapping.items()) + " }"


def quoted(text):
    return f'"{text}"'


# The daily collection files' headers: every field, including the optional bank.
COLLECTION_HEADERS = {
    "Company Code": "company_code",
    "Customer": "customer",
    "Customer Name": "customer_name",
    "Document Date": "document_date",
    "Bank": "bank",
    "SAP Document No": "sap_document_no",
    "Government Inv": "invoice_no",
    "Amount in Original Currency": "amount_original",
    "Document currency": "currency_original",
    "Amount in local currency": "amount_try",
    "Local Currency": "currency_local",
}
# The ledger's headers: only the required fields (the optional ones are left out on purpose,
# which also shows an optional field may be absent).
LEDGER_HEADERS = {
    "Company Code": "company_code",
    "Customer": "customer",
    "Document Number": "document_no",
    "Document Type": "document_type",
    "Reference": "reference",
    "Invoice Reference": "invoice_reference",
    "Posting Date": "posting_date",
    "Net Due Date": "net_due_date",
    "Amount in Doc. Curr.": "amount_document",
    "Document Currency": "document_currency",
    "Amount in Local Currency": "amount_local",
    "Clearing Date": "clearing_date",
    # Required since the receivables step links lines that cleared together by it.
    "Clearing Document": "clearing_document",
}
DOCUMENT_TYPES ={"invoice": '["RV"]', "payment": '["DZ"]', "credit_memo": '["DG"]'}


def headers_toml(headers):
    return inline({quoted(header): quoted(field) for header, field in headers.items()})


# Each value is written as TOML text, exactly as it would appear in the file.
VALID = {
    "data": {
        "corpus_folder": '"data/raw/corpus"',
        "ledger_file": '"data/raw/ledger.xlsx"',
        "ledger_as_of": "2026-08-28",
        "holidays_file": '"settings/holidays.toml"',
        "fill_missing_days_from_ledger": "true",
        "cache_folder": '"data/interim"',
    },
    "companies": {"known": '["TR02", "TR04"]', "unidentified_customers": '["99999"]'},
    "collections_layout": {
        "file_name_date_format": '"%d%m%Y"',
        "header_row": "2",
        "first_data_row": "3",
        "headers": headers_toml(COLLECTION_HEADERS),
    },
    "ledger_layout": {
        "header_row": "1",
        "first_data_row": "2",
        "headers": headers_toml(LEDGER_HEADERS),
        "document_types": inline(DOCUMENT_TYPES),
    },
    "receivables": {"min_cleared_for_delay": "3", "horizons": "5",
                    "leak_check_weeks": "[0, 30]", "overdue_age_days": "[30, 90]"},
    "history": {"windows": "[4, 13, 52]", "use_ledger_before_first_file": "true"},
    "stage1": {"max_iter": "300", "learning_rate": "0.05", "max_leaf_nodes": "31",
               "min_samples_leaf": "20", "random_seed": "1",
               "feature_blocks": '["history", "calendar", "receivables"]',
               "company_as_feature": "true", "baseline_c": "1.0"},
    # U11.1: stage 2's regressor. Values differ from [stage1] on purpose (loss "gamma", 200
    # trees, ...), so a test can tell which section a value was read from.
    "stage2": {"loss": '"gamma"', "max_iter": "200", "learning_rate": "0.1",
               "max_leaf_nodes": "15", "min_samples_leaf": "10", "random_seed": "7",
               "feature_blocks": '["calendar", "history"]'},
    "diagnostic": {"run": "true", "origin": "40", "hgb_max_iter": "[100, 300, 600]",
                   "hgb_learning_rate": "[0.05, 0.1]", "hgb_max_leaf_nodes": "[15, 31, 63]",
                   "lr_c": "[0.01, 0.1, 1.0, 10.0]", "class_weights": '["none", "balanced"]',
                   "selection_metric": '"log loss"', "calibration_groups": "4",
                   "reference_window": "13", "reference_prior_weeks": "1.0",
                   "importance_repeats": "3", "output_folder": '"results/diagnostic"'},
    "debug": {"one_day": "2025-06-02", "sample_week": "30", "demo_origin": "40"},
}
ALL_SECTIONS = list(VALID)


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
    assert isinstance(settings.collections_layout, CollectionsLayout)
    assert isinstance(settings.ledger_layout, LedgerLayout)
    assert isinstance(settings.receivables, ReceivablesSettings)
    assert isinstance(settings.history, HistorySettings)


def test_paths_are_read_from_the_repo_root(tmp_path, repo):
    data = load(tmp_path, repo).data
    assert data.corpus_folder == (repo / "data" / "raw" / "corpus").resolve()
    assert data.ledger_file == (repo / "data" / "raw" / "ledger.xlsx").resolve()
    assert data.holidays_file == (repo / "settings" / "holidays.toml").resolve()
    assert data.cache_folder == (repo / "data" / "interim").resolve()


def test_paths_are_absolute_path_objects(tmp_path, repo):
    data = load(tmp_path, repo).data
    for path in (data.corpus_folder, data.ledger_file, data.holidays_file, data.cache_folder):
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


@pytest.mark.parametrize("attribute", ["settings_file", "data", "companies",
                                       "collections_layout", "ledger_layout", "receivables",
                                       "history", "stage1", "stage2", "diagnostic",
                                       "debug"])
def test_settings_is_frozen(tmp_path, repo, attribute):
    settings = load(tmp_path, repo)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(settings, attribute, None)


@pytest.mark.parametrize("part, attribute", [
    ("data", "corpus_folder"), ("data", "ledger_file"), ("data", "ledger_as_of"),
    ("data", "holidays_file"), ("data", "fill_missing_days_from_ledger"),
    ("data", "cache_folder"), ("companies", "known"), ("companies", "unidentified_customers"),
    ("debug", "one_day"), ("debug", "sample_week"), ("collections_layout", "file_name_date_format"), ("collections_layout", "header_row"),
    ("collections_layout", "first_data_row"), ("collections_layout", "headers"),
    ("ledger_layout", "header_row"), ("ledger_layout", "first_data_row"),
    ("ledger_layout", "headers"), ("ledger_layout", "document_types"),
    ("receivables", "min_cleared_for_delay"), ("receivables", "horizons"),
    ("receivables", "leak_check_weeks"), ("receivables", "overdue_age_days"),
    ("history", "windows"), ("history", "use_ledger_before_first_file"),
    ("stage1", "max_iter"), ("stage1", "learning_rate"), ("stage1", "max_leaf_nodes"),
    ("stage1", "min_samples_leaf"), ("stage1", "random_seed"), ("stage1", "feature_blocks"),
    ("stage1", "company_as_feature"), ("stage1", "baseline_c"),
    ("stage2", "loss"), ("stage2", "max_iter"), ("stage2", "learning_rate"),
    ("stage2", "max_leaf_nodes"), ("stage2", "min_samples_leaf"), ("stage2", "random_seed"),
    ("stage2", "feature_blocks"),
    ("diagnostic", "run"), ("diagnostic", "origin"), ("diagnostic", "hgb_max_iter"),
    ("diagnostic", "hgb_learning_rate"), ("diagnostic", "hgb_max_leaf_nodes"),
    ("diagnostic", "lr_c"), ("diagnostic", "class_weights"),
    ("diagnostic", "selection_metric"), ("diagnostic", "calibration_groups"),
    ("diagnostic", "reference_window"), ("diagnostic", "reference_prior_weeks"),
    ("diagnostic", "importance_repeats"), ("diagnostic", "output_folder"),
    ("debug", "demo_origin"),
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
    assert not data.holidays_file.exists()


# --- holidays_file ---------------------------------------------------------------------

def test_holidays_file_is_required(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[data]", "holidays_file",
                            changes={("data", "holidays_file"): REMOVE})
    assert "missing" in message


def test_holidays_file_is_resolved_from_the_repo_root(tmp_path, repo):
    settings = load(tmp_path, repo,
                    changes={("data", "holidays_file"): '"settings/holidays_tr.toml"'})
    assert settings.data.holidays_file == (repo / "settings" / "holidays_tr.toml").resolve()


def test_holidays_file_does_not_depend_on_the_working_directory(tmp_path, repo, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert load(tmp_path, repo).data.holidays_file ==         (repo / "settings" / "holidays.toml").resolve()


def test_an_absolute_holidays_file_is_kept(tmp_path, repo):
    absolute = (tmp_path / "elsewhere" / "holidays.toml").resolve()
    settings = load(tmp_path, repo, changes={("data", "holidays_file"): f"'{absolute}'"})
    assert settings.data.holidays_file == absolute


def test_a_typo_in_holidays_file_names_both(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "'holidays_file'", "'holiday_file'",
                            changes={("data", "holidays_file"): REMOVE,
                                     ("data", "holiday_file"): '"settings/h.toml"'})
    assert "missing" in message and "unknown" in message


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_holidays_file_loads_with_load_holidays():
    from collection_estimation.hurdle.holidays import HolidayTable, load_holidays
    settings = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT)
    table = load_holidays(settings.data.holidays_file)
    assert isinstance(table, HolidayTable)
    assert len(table.fixed) == 7
    assert len(table.bayrams) == 4


# --- strictness: sections --------------------------------------------------------------

@pytest.mark.parametrize("section", ALL_SECTIONS)
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
    ("data", "holidays_file"), ("data", "fill_missing_days_from_ledger"),
    ("data", "cache_folder"), ("companies", "known"), ("companies", "unidentified_customers"),
    ("debug", "one_day"), ("debug", "sample_week"), ("receivables", "min_cleared_for_delay"),
    ("receivables", "horizons"), ("receivables", "leak_check_weeks"),
    ("receivables", "overdue_age_days"),
    ("history", "windows"), ("history", "use_ledger_before_first_file"),
    ("stage1", "max_iter"), ("stage1", "learning_rate"), ("stage1", "max_leaf_nodes"),
    ("stage1", "min_samples_leaf"), ("stage1", "random_seed"), ("stage1", "feature_blocks"),
    ("stage1", "company_as_feature"), ("stage1", "baseline_c"),
    ("stage2", "loss"), ("stage2", "max_iter"), ("stage2", "learning_rate"),
    ("stage2", "max_leaf_nodes"), ("stage2", "min_samples_leaf"), ("stage2", "random_seed"),
    ("stage2", "feature_blocks"),
    ("diagnostic", "run"), ("diagnostic", "origin"), ("diagnostic", "hgb_max_iter"),
    ("diagnostic", "hgb_learning_rate"), ("diagnostic", "hgb_max_leaf_nodes"),
    ("diagnostic", "lr_c"), ("diagnostic", "class_weights"),
    ("diagnostic", "selection_metric"), ("diagnostic", "calibration_groups"),
    ("diagnostic", "reference_window"), ("diagnostic", "reference_prior_weeks"),
    ("diagnostic", "importance_repeats"), ("diagnostic", "output_folder"),
    ("debug", "demo_origin"),
])
def test_a_missing_setting_raises(tmp_path, repo, section, key):
    message = raises_naming(tmp_path, repo, f"[{section}]", key,
                            changes={(section, key): REMOVE})
    assert "missing" in message


@pytest.mark.parametrize("section", ALL_SECTIONS)
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
    for key in ["cache_folder", "corpus_folder", "fill_missing_days_from_ledger",
                "holidays_file", "ledger_as_of", "ledger_file"]:
        assert key in message.split("expected")[-1]


def test_a_problem_in_a_later_section_names_that_section(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[debug]",
                            changes={("debug", "one_dya"): "2025-06-02",
                                     ("debug", "one_day"): REMOVE})
    assert "[data]" not in message


# --- types: paths ----------------------------------------------------------------------

@pytest.mark.parametrize("key", ["corpus_folder", "ledger_file", "holidays_file",
                                 "cache_folder"])
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


# --- types: sample_week ----------------------------------------------------------------

def test_sample_week_is_an_int(tmp_path, repo):
    value = load(tmp_path, repo).debug.sample_week
    assert type(value) is int
    assert value == 30


@pytest.mark.parametrize("good, expected", [("0", 0), ("1", 1), ("63", 63), ("1_000", 1000)])
def test_sample_week_accepts_a_whole_number_of_0_or_more(tmp_path, repo, good, expected):
    settings = load(tmp_path, repo, changes={("debug", "sample_week"): good})
    assert settings.debug.sample_week == expected
    assert type(settings.debug.sample_week) is int


@pytest.mark.parametrize("bad", [
    "-1", "-30", "1.5", "30.0", '"30"', "true", "false", "[30]", "2025-06-02", "{ week = 30 }",
], ids=["minus-one", "negative", "fraction", "whole-float", "string", "true", "false", "list",
        "date", "table"])
def test_sample_week_that_is_not_a_week_number_raises(tmp_path, repo, bad):
    message = raises_naming(tmp_path, repo, "[debug]", "sample_week",
                            changes={("debug", "sample_week"): bad})
    assert "week number" in message


def test_a_sample_week_typo_names_the_missing_and_the_unknown_setting(tmp_path, repo):
    message = raises_naming(
        tmp_path, repo, "[debug]", "'sample_week'", "'sample_weak'",
        changes={("debug", "sample_week"): REMOVE, ("debug", "sample_weak"): "30"})
    assert "missing" in message and "unknown" in message


def test_week_setting_accepts_zero_and_positive_ints():
    assert week_setting({"w": 0}, "w", "here") == 0
    assert week_setting({"w": 7}, "w", "here") == 7


@pytest.mark.parametrize("bad", [-1, 1.5, 30.0, "30", True, False, None])
def test_week_setting_rejects_non_week_values_naming_where_and_key(bad):
    with pytest.raises(ValueError, match="here: w is"):
        week_setting({"w": bad}, "w", "here")


# --- types: min_cleared_for_delay (count_setting) -----------------------------------------

def test_min_cleared_for_delay_is_read_as_an_int(tmp_path, repo):
    value = load(tmp_path, repo).receivables.min_cleared_for_delay
    assert value == 3
    assert type(value) is int


@pytest.mark.parametrize("good, expected", [("1", 1), ("2", 2), ("10", 10), ("1_000", 1000)])
def test_min_cleared_for_delay_accepts_a_whole_number_of_1_or_more(tmp_path, repo, good,
                                                                  expected):
    settings = load(tmp_path, repo, changes={("receivables", "min_cleared_for_delay"): good})
    assert settings.receivables.min_cleared_for_delay == expected
    assert type(settings.receivables.min_cleared_for_delay) is int


@pytest.mark.parametrize("bad", [
    "0", "-1", "1.5", "3.0", '"3"', "true", "false", "[3]", "2025-06-02", "{ n = 3 }",
], ids=["zero", "minus-one", "fraction", "whole-float", "string", "true", "false", "list",
        "date", "table"])
def test_min_cleared_for_delay_that_is_not_a_count_raises(tmp_path, repo, bad):
    message = raises_naming(tmp_path, repo, "[receivables]", "min_cleared_for_delay",
                            changes={("receivables", "min_cleared_for_delay"): bad})
    assert "1 or more" in message


def test_a_missing_receivables_section_names_it(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "receivables", drop_section="receivables")
    assert "missing" in message


def test_an_empty_receivables_section_names_the_missing_setting(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[receivables]", "min_cleared_for_delay",
                            changes={("receivables", "min_cleared_for_delay"): REMOVE})
    assert "missing" in message


def test_a_min_cleared_for_delay_typo_names_the_missing_and_the_unknown_setting(tmp_path, repo):
    message = raises_naming(
        tmp_path, repo, "[receivables]", "'min_cleared_for_delay'", "'min_clear_for_delay'",
        changes={("receivables", "min_cleared_for_delay"): REMOVE,
                 ("receivables", "min_clear_for_delay"): "3"})
    assert "missing" in message and "unknown" in message


# --- types: horizons (count_setting) ------------------------------------------------------

def test_horizons_is_read_as_an_int(tmp_path, repo):
    value = load(tmp_path, repo).receivables.horizons
    assert value == 5
    assert type(value) is int


@pytest.mark.parametrize("good, expected", [("1", 1), ("3", 3), ("13", 13)])
def test_horizons_accepts_a_whole_number_of_1_or_more(tmp_path, repo, good, expected):
    settings = load(tmp_path, repo, changes={("receivables", "horizons"): good})
    assert settings.receivables.horizons == expected
    assert type(settings.receivables.horizons) is int


@pytest.mark.parametrize("bad", [
    "0", "-1", "1.5", "5.0", '"5"', "true", "false", "[5]",
], ids=["zero", "minus-one", "fraction", "whole-float", "string", "true", "false", "list"])
def test_horizons_that_is_not_a_count_raises(tmp_path, repo, bad):
    message = raises_naming(tmp_path, repo, "[receivables]", "horizons",
                            changes={("receivables", "horizons"): bad})
    assert "1 or more" in message


def test_horizons_does_not_borrow_min_cleared_for_delay(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("receivables", "horizons"): "2",
                                             ("receivables", "min_cleared_for_delay"): "7"})
    assert settings.receivables.horizons == 2
    assert settings.receivables.min_cleared_for_delay == 7


def test_a_missing_horizons_names_it(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[receivables]", "horizons",
                            changes={("receivables", "horizons"): REMOVE})
    assert "missing" in message


def test_a_horizons_typo_names_the_missing_and_the_unknown_setting(tmp_path, repo):
    message = raises_naming(
        tmp_path, repo, "[receivables]", "'horizons'", "'horizon'",
        changes={("receivables", "horizons"): REMOVE, ("receivables", "horizon"): "5"})
    assert "missing" in message and "unknown" in message


# --- types: leak_check_weeks (week_list_setting) ------------------------------------------

def test_leak_check_weeks_is_read_as_a_tuple_of_ints_in_file_order(tmp_path, repo):
    value = load(tmp_path, repo).receivables.leak_check_weeks
    assert value == (0, 30)
    assert type(value) is tuple
    assert all(type(week) is int for week in value)


@pytest.mark.parametrize("good, expected", [
    ("[]", ()), ("[0]", (0,)), ("[0, 30]", (0, 30)), ("[64, 0, 30]", (64, 0, 30)),
], ids=["empty", "zero", "two", "unsorted-kept-in-order"])
def test_leak_check_weeks_accepts_distinct_week_numbers(tmp_path, repo, good, expected):
    settings = load(tmp_path, repo, changes={("receivables", "leak_check_weeks"): good})
    assert settings.receivables.leak_check_weeks == expected


@pytest.mark.parametrize("bad, shown", [
    ("[-1]", "-1"), ("[0, 1.5]", "1.5"), ('["3"]', "'3'"), ("[true]", "True"),
    ("[0, false]", "False"), ("[[3]]", "[3]"),
], ids=["minus-one", "fraction", "string", "true", "false", "nested-list"])
def test_leak_check_weeks_with_a_non_week_item_raises_naming_it(tmp_path, repo, bad, shown):
    message = raises_naming(tmp_path, repo, "[receivables]", "leak_check_weeks", shown,
                            changes={("receivables", "leak_check_weeks"): bad})
    assert "0 or more" in message


@pytest.mark.parametrize("bad", ["3", '"0, 30"', "true", "{ a = 1 }"],
                         ids=["int", "string", "bool", "table"])
def test_leak_check_weeks_that_is_not_a_list_raises(tmp_path, repo, bad):
    message = raises_naming(tmp_path, repo, "[receivables]", "leak_check_weeks",
                            changes={("receivables", "leak_check_weeks"): bad})
    assert "not a list" in message


def test_leak_check_weeks_with_duplicates_raises_naming_them(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[receivables]", "leak_check_weeks", "[0, 30]",
                            changes={("receivables", "leak_check_weeks"): "[30, 0, 30, 5, 0]"})
    assert "more than once" in message


def test_a_missing_leak_check_weeks_names_it(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[receivables]", "leak_check_weeks",
                            changes={("receivables", "leak_check_weeks"): REMOVE})
    assert "missing" in message


def test_week_list_setting_direct_calls():
    assert week_list_setting({"w": []}, "w", "here") == ()
    assert week_list_setting({"w": [3, 0]}, "w", "here") == (3, 0)
    with pytest.raises(ValueError, match="here: w holds True"):
        week_list_setting({"w": [True]}, "w", "here")
    with pytest.raises(ValueError, match="here: w is"):
        week_list_setting({"w": (0, 1)}, "w", "here")
    with pytest.raises(ValueError, match=r"here: w lists \[2\] more than once"):
        week_list_setting({"w": [2, 2]}, "w", "here")


# --- types: overdue_age_days (day_edges_setting) ----------------------------------------

def test_overdue_age_days_is_read_as_a_tuple_of_ints(tmp_path, repo):
    value = load(tmp_path, repo).receivables.overdue_age_days
    assert value == (30, 90)
    assert type(value) is tuple
    assert all(type(days) is int for days in value)


@pytest.mark.parametrize("good, expected", [
    ("[1]", (1,)), ("[30]", (30,)), ("[30, 90]", (30, 90)), ("[30, 60, 90]", (30, 60, 90)),
    ("[1, 2]", (1, 2)), ("[7, 14, 30, 60, 90, 180, 365]", (7, 14, 30, 60, 90, 180, 365)),
], ids=["one-day", "one-edge", "two", "three", "adjacent", "seven"])
def test_overdue_age_days_accepts_increasing_day_counts(tmp_path, repo, good, expected):
    settings = load(tmp_path, repo, changes={("receivables", "overdue_age_days"): good})
    assert settings.receivables.overdue_age_days == expected


@pytest.mark.parametrize("bad, shown", [
    ("[0]", "0"), ("[-30, 90]", "-30"), ("[30, 1.5]", "1.5"), ("[30.0]", "30.0"),
    ('["30"]', "'30'"), ("[true]", "True"), ("[30, false]", "False"), ("[[30]]", "[30]"),
], ids=["zero", "negative", "fraction", "whole-float", "string", "true", "false", "nested"])
def test_overdue_age_days_with_a_bad_item_raises_naming_it(tmp_path, repo, bad, shown):
    message = raises_naming(tmp_path, repo, "[receivables]", "overdue_age_days", shown,
                            changes={("receivables", "overdue_age_days"): bad})
    assert "not a number of days (1 or more)" in message


@pytest.mark.parametrize("bad", ["30", '"30, 90"', "true", "{ a = 1 }", "[]"],
                         ids=["int", "string", "bool", "table", "empty"])
def test_overdue_age_days_that_is_not_a_non_empty_list_raises(tmp_path, repo, bad):
    message = raises_naming(tmp_path, repo, "[receivables]", "overdue_age_days",
                            changes={("receivables", "overdue_age_days"): bad})
    assert "not a non-empty list of days" in message


@pytest.mark.parametrize("bad", ["[90, 30]", "[30, 30]", "[30, 90, 60]", "[30, 90, 90]"],
                         ids=["reversed", "repeated", "unsorted", "repeated-last"])
def test_overdue_age_days_not_strictly_increasing_raises(tmp_path, repo, bad):
    message = raises_naming(tmp_path, repo, "[receivables]", "overdue_age_days",
                            changes={("receivables", "overdue_age_days"): bad})
    assert "give the days from smallest to largest, each once" in message


def test_a_missing_overdue_age_days_names_it(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[receivables]", "overdue_age_days",
                            changes={("receivables", "overdue_age_days"): REMOVE})
    assert "missing" in message


def test_an_overdue_age_days_typo_names_the_missing_and_the_unknown_setting(tmp_path, repo):
    message = raises_naming(
        tmp_path, repo, "[receivables]", "'overdue_age_days'", "'overdue_age_day'",
        changes={("receivables", "overdue_age_days"): REMOVE,
                 ("receivables", "overdue_age_day"): "[30, 90]"})
    assert "missing" in message and "unknown" in message


def test_overdue_age_days_does_not_borrow_leak_check_weeks(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("receivables", "overdue_age_days"): "[15]",
                                             ("receivables", "leak_check_weeks"): "[3, 1]"})
    assert settings.receivables.overdue_age_days == (15,)
    assert settings.receivables.leak_check_weeks == (3, 1)


def test_day_edges_setting_direct_calls_and_exact_messages():
    assert day_edges_setting({"d": [30, 90]}, "d", "here") == (30, 90)
    assert type(day_edges_setting({"d": [5]}, "d", "here")) is tuple
    with pytest.raises(ValueError) as error:
        day_edges_setting({"d": []}, "d", "here")
    assert str(error.value) == "here: d is [], not a non-empty list of days"
    with pytest.raises(ValueError) as error:
        day_edges_setting({"d": (30, 90)}, "d", "here")
    assert str(error.value) == "here: d is (30, 90), not a non-empty list of days"
    with pytest.raises(ValueError) as error:
        day_edges_setting({"d": [30, True]}, "d", "here")
    assert str(error.value) == "here: d holds True, not a number of days (1 or more)"
    with pytest.raises(ValueError) as error:
        day_edges_setting({"d": [0, 30]}, "d", "here")
    assert str(error.value) == "here: d holds 0, not a number of days (1 or more)"
    with pytest.raises(ValueError) as error:
        day_edges_setting({"d": [90, 30]}, "d", "here")
    assert str(error.value) == ("here: d is [90, 30]; give the days from smallest to largest, "
                                "each once")


def test_day_edges_setting_does_not_sort_or_modify_its_input():
    table = {"d": [30, 90]}
    day_edges_setting(table, "d", "here")
    assert table == {"d": [30, 90]}


def test_count_setting_accepts_one_and_more():
    assert count_setting({"n": 1}, "n", "here") == 1
    assert count_setting({"n": 3}, "n", "here") == 3
    assert type(count_setting({"n": 3}, "n", "here")) is int


@pytest.mark.parametrize("bad", [0, -1, 1.5, 3.0, "3", True, False, None, [3]])
def test_count_setting_rejects_non_counts_naming_where_and_key(bad):
    with pytest.raises(ValueError, match="here: n is"):
        count_setting({"n": bad}, "n", "here")


def test_count_setting_message_shows_the_value():
    with pytest.raises(ValueError) as error:
        count_setting({"n": 0}, "n", "here")
    assert str(error.value) == "here: n is 0, not a whole number of 1 or more"


# --- types: known ----------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    '"TR02"', "[]", '["TR02", 4]', '["TR02", ""]', '["TR02", "  "]', '[["TR02"]]',
    "{ a = 1 }",
], ids=["text", "empty-list", "number-item", "empty-item", "blank-item", "nested-list",
        "inline-table"])
def test_a_bad_known_raises(tmp_path, repo, bad):
    raises_naming(tmp_path, repo, "[companies]", "known",
                  changes={("companies", "known"): bad})


def test_an_empty_known_raises_with_the_is_empty_message(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[companies]", "known",
                            changes={("companies", "known"): "[]"})
    assert message.endswith("known is empty; it needs at least one item")


def test_a_known_text_raises_with_the_not_a_list_message(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[companies]", "known",
                            changes={("companies", "known"): '"TR02"'})
    assert message.endswith("known is 'TR02', not a list")


# --- types: unidentified_customers ------------------------------------------------------

def test_unidentified_customers_is_a_tuple_of_texts_in_file_order(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("companies", "unidentified_customers"):
                                             '["99999", "0000012345", "SUSPENSE"]'})
    value = settings.companies.unidentified_customers
    assert value == ("99999", "0000012345", "SUSPENSE")
    assert type(value) is tuple


def test_unidentified_customers_single_item(tmp_path, repo):
    settings = load(tmp_path, repo)  # VALID has ["99999"]
    assert settings.companies.unidentified_customers == ("99999",)


def test_unidentified_customers_may_be_empty(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("companies", "unidentified_customers"): "[]"})
    assert settings.companies.unidentified_customers == ()
    assert type(settings.companies.unidentified_customers) is tuple
    assert settings.companies.known == ("TR02", "TR04")


def test_unidentified_customers_is_kept_as_written_not_normalised(tmp_path, repo):
    # Pinned: settings keep the text as written; is_unidentified normalises when matching.
    settings = load(tmp_path, repo, changes={("companies", "unidentified_customers"):
                                             '["0099999", " ab12 "]'})
    assert settings.companies.unidentified_customers == ("0099999", " ab12 ")


def test_a_missing_unidentified_customers_names_the_setting(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[companies]", "unidentified_customers",
                            changes={("companies", "unidentified_customers"): REMOVE})
    assert "missing ['unidentified_customers']" in message


@pytest.mark.parametrize("bad, expected", [
    ('"99999"', "not a list"),
    ("99999", "not a list"),
    ("{ a = 1 }", "not a list"),
    ('["99999", "99999"]', "more than once"),
    ('["99999", ""]', "not a non-empty text"),
    ('["99999", "   "]', "not a non-empty text"),
    ('["99999", 99999]', "not a non-empty text"),
    ('[["99999"]]', "not a non-empty text"),
    ("[true]", "not a non-empty text"),
], ids=["text", "number", "inline-table", "duplicate", "empty-item", "blank-item",
        "number-item", "nested-list", "bool-item"])
def test_a_bad_unidentified_customers_raises(tmp_path, repo, bad, expected):
    raises_naming(tmp_path, repo, "[companies]", "unidentified_customers", expected,
                  changes={("companies", "unidentified_customers"): bad})


def test_unidentified_customers_duplicates_are_compared_as_written(tmp_path, repo):
    # Pinned: "99999" and "0099999" are different texts, so the setting loads although
    # both normalise to the same ID (harmless: is_unidentified builds a set).
    settings = load(tmp_path, repo, changes={("companies", "unidentified_customers"):
                                             '["99999", "0099999"]'})
    assert settings.companies.unidentified_customers == ("99999", "0099999")


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
    for section in ALL_SECTIONS:
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


@pytest.mark.parametrize("bad", ["AB", ("A",), None, {"a": 1}],
                         ids=["text", "tuple", "none", "dict"])
def test_text_list_setting_rejects_a_non_list(bad):
    with pytest.raises(ValueError) as error:
        text_list_setting({"k": bad}, "k", "w")
    assert str(error.value) == f"w: k is {bad!r}, not a list"


@pytest.mark.parametrize("bad", ["AB", ("A",), None, ()],
                         ids=["text", "tuple", "none", "empty-tuple"])
def test_text_list_setting_rejects_a_non_list_even_when_it_may_be_empty(bad):
    with pytest.raises(ValueError) as error:
        text_list_setting({"k": bad}, "k", "w", may_be_empty=True)
    assert str(error.value) == f"w: k is {bad!r}, not a list"


def test_text_list_setting_rejects_an_empty_list_by_default():
    with pytest.raises(ValueError) as error:
        text_list_setting({"k": []}, "k", "w")
    assert str(error.value) == "w: k is empty; it needs at least one item"


def test_text_list_setting_rejects_an_empty_list_when_may_be_empty_is_false():
    with pytest.raises(ValueError, match="k is empty; it needs at least one item"):
        text_list_setting({"k": []}, "k", "w", may_be_empty=False)


def test_text_list_setting_accepts_an_empty_list_when_it_may_be_empty():
    result = text_list_setting({"k": []}, "k", "w", may_be_empty=True)
    assert result == ()
    assert type(result) is tuple


def test_text_list_setting_may_be_empty_still_reads_a_non_empty_list():
    assert text_list_setting({"k": ["A", "B"]}, "k", "w", may_be_empty=True) == ("A", "B")


@pytest.mark.parametrize("bad_item", [1, None, "", "  ", ["A"]],
                         ids=["number", "none", "empty", "blank", "list"])
def test_text_list_setting_may_be_empty_still_rejects_a_bad_item(bad_item):
    with pytest.raises(ValueError, match="w: k holds"):
        text_list_setting({"k": [bad_item]}, "k", "w", may_be_empty=True)


def test_text_list_setting_may_be_empty_still_rejects_a_duplicate():
    with pytest.raises(ValueError) as error:
        text_list_setting({"k": ["99999", "99999"]}, "k", "w", may_be_empty=True)
    assert str(error.value) == "w: k lists ['99999'] more than once"


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
    assert settings.data.holidays_file.is_file()


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_settings_hold_the_expected_values():
    settings = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT)
    assert settings.data.ledger_as_of == date(2026, 8, 28)
    assert settings.companies.known == ("TR02", "TR04", "TR05", "TR06", "TR91")
    assert settings.companies.unidentified_customers == ()
    assert settings.debug.one_day == date(2025, 6, 2)
    assert settings.debug.sample_week == 30
    assert type(settings.debug.sample_week) is int
    assert settings.receivables.min_cleared_for_delay == 3
    assert type(settings.receivables.min_cleared_for_delay) is int
    assert settings.receivables.horizons == 5
    assert type(settings.receivables.horizons) is int
    assert settings.receivables.leak_check_weeks == (0, 16, 32, 48)
    assert type(settings.receivables.leak_check_weeks) is tuple
    assert settings.receivables.overdue_age_days == (30, 90)
    assert type(settings.receivables.overdue_age_days) is tuple
    assert settings.data.cache_folder.parts[-2:] == ("data", "interim")
    assert settings.data.cache_folder == (REPO_ROOT / "data" / "interim").resolve()
    assert settings.data.holidays_file == (REPO_ROOT / "settings" / "holidays_tr.toml").resolve()
    assert settings.settings_file == SYNTHETIC_SETTINGS.resolve()
    assert settings.data.fill_missing_days_from_ledger is True
    assert settings.history.windows == (4, 13, 52)
    assert type(settings.history.windows) is tuple
    assert settings.history.use_ledger_before_first_file is True
    assert settings.debug.demo_origin == 40
    assert type(settings.debug.demo_origin) is int
    assert settings.stage1 == Stage1Settings(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
        random_seed=1, feature_blocks=("history", "calendar", "receivables"),
        company_as_feature=True, baseline_c=1.0)



# The 11 headers of the daily files, as [collections_layout.headers] in synthetic.toml gives
# them (the module's built-in HEADER_TO_FIELD is gone since L.2: this is the reference now).
EXPECTED_COLLECTIONS_HEADERS = {
    "Company Code": "company_code",
    "Customer": "customer",
    "Customer Name": "customer_name",
    "Document Date": "document_date",
    "Bank": "bank",
    "SAP Document No": "sap_document_no",
    "Government Inv": "invoice_no",
    "Amount in Original Currency": "amount_original",
    "Document currency": "currency_original",
    "Amount in local currency": "amount_try",
    "Local Currency": "currency_local",
}


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_collections_headers_are_the_expected_11():
    headers = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT).collections_layout.headers
    assert headers == EXPECTED_COLLECTIONS_HEADERS
    assert list(headers) == list(EXPECTED_COLLECTIONS_HEADERS)  # same order too


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_collections_headers_map_every_field_once_in_field_order():
    headers = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT).collections_layout.headers
    assert list(headers.values()) == COLLECTION_FIELDS


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_collections_layout_equals_the_shared_test_layout():
    # The tests' SYNTHETIC_LAYOUT (tests/hurdle/workbooks.py) must stay what the settings say.
    layout = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT).collections_layout
    assert layout == SYNTHETIC_LAYOUT


# The 17 headers of the FBL5N ledger, as [ledger_layout.headers] in synthetic.toml gives them
# (the module's built-in LEDGER_HEADER_TO_FIELD is gone since L.3: this is the reference now).
EXPECTED_LEDGER_HEADERS = {
    "Company Code": "company_code",
    "Customer": "customer",
    "Customer Name": "customer_name",
    "Document Number": "document_no",
    "Document Type": "document_type",
    "Reference": "reference",
    "Invoice Reference": "invoice_reference",
    "Document Date": "document_date",
    "Posting Date": "posting_date",
    "Baseline Date": "baseline_date",
    "Terms of Payment": "payment_terms",
    "Net Due Date": "net_due_date",
    "Amount in Doc. Curr.": "amount_document",
    "Document Currency": "document_currency",
    "Amount in Local Currency": "amount_local",
    "Clearing Date": "clearing_date",
    "Clearing Document": "clearing_document",
}


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_ledger_headers_are_the_expected_17():
    headers = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT).ledger_layout.headers
    assert headers == EXPECTED_LEDGER_HEADERS
    assert list(headers) == list(EXPECTED_LEDGER_HEADERS)  # same order too
    assert list(headers.values()) == LEDGER_FIELDS


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_ledger_layout_equals_the_shared_test_layout():
    # The tests' SYNTHETIC_LEDGER_LAYOUT (tests/hurdle/workbooks.py) must stay what the
    # settings say.
    layout = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT).ledger_layout
    assert layout == SYNTHETIC_LEDGER_LAYOUT


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_layouts_hold_the_expected_values():
    settings = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT)
    collections = settings.collections_layout
    assert collections.file_name_date_format == "%d%m%Y"
    assert (collections.header_row, collections.first_data_row) == (2, 3)
    ledger = settings.ledger_layout
    assert (ledger.header_row, ledger.first_data_row) == (1, 2)
    assert ledger.document_types == {
        "invoice": ("RV",), "payment": ("DZ",), "credit_memo": ("DG",)}
    codes = {code for role in ledger.document_types.values() for code in role}
    assert codes == {"RV", "DZ", "DG"}


def test_the_field_lists_follow_the_built_in_mappings():
    assert COLLECTION_FIELDS == list(EXPECTED_COLLECTIONS_HEADERS.values())
    assert LEDGER_FIELDS == list(EXPECTED_LEDGER_HEADERS.values())


def test_the_optional_fields_are_fields():
    assert OPTIONAL_FIELDS <= set(COLLECTION_FIELDS)
    assert LEDGER_OPTIONAL_FIELDS <= set(LEDGER_FIELDS)


# --- layouts: happy path ---------------------------------------------------------------

def test_the_collections_layout_is_read(tmp_path, repo):
    layout = load(tmp_path, repo).collections_layout
    assert layout == CollectionsLayout(file_name_date_format="%d%m%Y", header_row=2,
                                       first_data_row=3, headers=COLLECTION_HEADERS)
    assert type(layout.header_row) is int and type(layout.first_data_row) is int
    assert type(layout.headers) is dict
    assert list(layout.headers) == list(COLLECTION_HEADERS)  # file order


def test_the_ledger_layout_is_read(tmp_path, repo):
    layout = load(tmp_path, repo).ledger_layout
    assert layout.header_row == 1 and layout.first_data_row == 2
    assert layout.headers == LEDGER_HEADERS
    assert list(layout.headers) == list(LEDGER_HEADERS)
    assert layout.document_types == {
        "invoice": ("RV",), "payment": ("DZ",), "credit_memo": ("DG",)}
    for codes in layout.document_types.values():
        assert type(codes) is tuple


def test_headers_written_as_a_sub_table_section_are_read(tmp_path, repo):
    lines = ["[collections_layout.headers]"]
    lines += [f'"{header}" = "{field}"' for header, field in COLLECTION_HEADERS.items()]
    settings = load(tmp_path, repo, changes={("collections_layout", "headers"): REMOVE},
                    extra="\n".join(lines) + "\n")
    assert settings.collections_layout.headers == COLLECTION_HEADERS


def test_several_codes_per_role_load(tmp_path, repo):
    types = dict(DOCUMENT_TYPES, payment='["DZ", "DA"]')
    settings = load(tmp_path, repo,
                    changes={("ledger_layout", "document_types"): inline(types)})
    assert settings.ledger_layout.document_types["payment"] == ("DZ", "DA")


def test_lower_case_codes_in_the_file_load_upper_cased(tmp_path, repo):
    types = {"invoice": '["rv"]', "payment": '[" dz ", "Da"]', "credit_memo": '["dG"]'}
    settings = load(tmp_path, repo,
                    changes={("ledger_layout", "document_types"): inline(types)})
    assert settings.ledger_layout.document_types == {
        "invoice": ("RV",), "payment": ("DZ", "DA"), "credit_memo": ("DG",)}


def test_case_variants_across_roles_in_the_file_raise(tmp_path, repo):
    types = dict(DOCUMENT_TYPES, payment='["DZ", "rv"]')
    raises_naming(tmp_path, repo, "[ledger_layout].document_types", "'RV'", "invoice",
                  "payment", changes={("ledger_layout", "document_types"): inline(types)})


def test_read_collections_layout_directly():
    table = {"file_name_date_format": "%Y%m%d", "header_row": 1, "first_data_row": 5,
             "headers": dict(COLLECTION_HEADERS)}
    assert read_collections_layout(table, "w") == CollectionsLayout("%Y%m%d", 1, 5,
                                                                    COLLECTION_HEADERS)


def test_read_ledger_layout_directly():
    table = {"header_row": 3, "first_data_row": 4, "headers": dict(LEDGER_HEADERS),
             "document_types": {"invoice": ["RV"], "payment": ["DZ"], "credit_memo": ["DG"]}}
    layout = read_ledger_layout(table, "w")
    assert layout == LedgerLayout(3, 4, LEDGER_HEADERS,
                                  {"invoice": ("RV",), "payment": ("DZ",),
                                   "credit_memo": ("DG",)})


def test_read_ledger_layout_checks_its_headers_against_the_ledger_fields():
    # The collections headers are not a valid ledger mapping (e.g. 'invoice_no').
    table = {"header_row": 1, "first_data_row": 2, "headers": dict(COLLECTION_HEADERS),
             "document_types": {"invoice": ["RV"], "payment": ["DZ"], "credit_memo": ["DG"]}}
    with pytest.raises(ValueError, match="w.headers"):
        read_ledger_layout(table, "w")


# --- layouts through load_settings: strictness ------------------------------------------

@pytest.mark.parametrize("section, key", [
    ("collections_layout", "file_name_date_format"), ("collections_layout", "header_row"),
    ("collections_layout", "first_data_row"), ("collections_layout", "headers"),
    ("ledger_layout", "header_row"), ("ledger_layout", "first_data_row"),
    ("ledger_layout", "headers"), ("ledger_layout", "document_types"),
])
def test_a_missing_layout_setting_raises(tmp_path, repo, section, key):
    message = raises_naming(tmp_path, repo, f"[{section}]", key,
                            changes={(section, key): REMOVE})
    assert "missing" in message


def test_a_missing_ledger_layout_section_raises(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "ledger_layout", drop_section="ledger_layout")
    assert "missing" in message


def test_a_missing_headers_sub_table_raises(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[ledger_layout]", "headers",
                            changes={("ledger_layout", "headers"): REMOVE})
    assert "missing" in message


def test_an_unknown_key_in_collections_layout_raises(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[collections_layout]", "sheet_name",
                            changes={("collections_layout", "sheet_name"): '"Sheet1"'})
    assert "unknown" in message


def test_a_typo_in_a_layout_key_names_both(tmp_path, repo):
    message = raises_naming(
        tmp_path, repo, "[collections_layout]", "'header_row'", "'headr_row'",
        changes={("collections_layout", "header_row"): REMOVE,
                 ("collections_layout", "headr_row"): "2"})
    assert "missing" in message and "unknown" in message


def test_a_bad_row_in_the_file_names_file_and_section(tmp_path, repo):
    raises_naming(tmp_path, repo, "[ledger_layout]", "header_row",
                  changes={("ledger_layout", "header_row"): "0"})


def test_a_bad_date_format_in_the_file_names_file_and_section(tmp_path, repo):
    raises_naming(tmp_path, repo, "[collections_layout]", "file_name_date_format", "'%m%Y'",
                  changes={("collections_layout", "file_name_date_format"): '"%m%Y"'})


def test_a_bad_header_in_the_file_names_file_and_section(tmp_path, repo):
    headers = dict(COLLECTION_HEADERS, Bank="bank_name")
    raises_naming(tmp_path, repo, "[collections_layout].headers", "'Bank'", "'bank_name'",
                  changes={("collections_layout", "headers"): headers_toml(headers)})


def test_a_header_mapped_to_a_ledger_field_in_collections_raises(tmp_path, repo):
    headers = dict(COLLECTION_HEADERS, Reference="reference")
    raises_naming(tmp_path, repo, "[collections_layout].headers", "'Reference'",
                  changes={("collections_layout", "headers"): headers_toml(headers)})


def test_bad_document_types_in_the_file_name_file_and_section(tmp_path, repo):
    types = dict(DOCUMENT_TYPES, credit_memo='["DZ"]')
    raises_naming(tmp_path, repo, "[ledger_layout].document_types", "'DZ'", "payment",
                  "credit_memo",
                  changes={("ledger_layout", "document_types"): inline(types)})


def test_an_unknown_role_in_the_file_names_file_and_section(tmp_path, repo):
    types = dict(DOCUMENT_TYPES, reversal='["AB"]')
    message = raises_naming(tmp_path, repo, "[ledger_layout].document_types", "reversal",
                            changes={("ledger_layout", "document_types"): inline(types)})
    assert "unknown" in message


def test_document_types_as_a_plain_value_raises(tmp_path, repo):
    raises_naming(tmp_path, repo, "[ledger_layout].document_types", "not a section",
                  changes={("ledger_layout", "document_types"): '["RV", "DZ", "DG"]'})


def test_headers_as_a_plain_value_raises(tmp_path, repo):
    raises_naming(tmp_path, repo, "[collections_layout]", "headers",
                  changes={("collections_layout", "headers"): '"Company Code"'})


# --- rows_setting ----------------------------------------------------------------------

def rows(header_row, first_data_row):
    return {"header_row": header_row, "first_data_row": first_data_row}


@pytest.mark.parametrize("header_row, first_data_row", [(1, 2), (2, 3), (1, 10), (5, 6)])
def test_rows_setting_returns_the_two_rows(header_row, first_data_row):
    assert rows_setting(rows(header_row, first_data_row), "w") == (header_row, first_data_row)


@pytest.mark.parametrize("bad", [0, -1, 1.5, 2.0, "2", True, False, None, [2]],
                         ids=["zero", "negative", "fraction", "float-whole", "text", "true",
                              "false", "none", "list"])
def test_rows_setting_rejects_a_bad_header_row(bad):
    with pytest.raises(ValueError) as error:
        rows_setting(rows(bad, 10), "w")
    message = str(error.value)
    assert message.startswith("w: header_row is")
    assert repr(bad) in message


@pytest.mark.parametrize("bad", [0, -1, 1.5, 2.0, "2", True, None],
                         ids=["zero", "negative", "fraction", "float-whole", "text", "true",
                              "none"])
def test_rows_setting_rejects_a_bad_first_data_row(bad):
    with pytest.raises(ValueError) as error:
        rows_setting(rows(1, bad), "w")
    assert str(error.value).startswith("w: first_data_row is")


def test_rows_setting_rejects_equal_rows():
    with pytest.raises(ValueError) as error:
        rows_setting(rows(2, 2), "w")
    message = str(error.value)
    assert "first_data_row 2" in message and "header_row 2" in message


def test_rows_setting_rejects_data_before_the_header():
    with pytest.raises(ValueError, match="must come after"):
        rows_setting(rows(3, 2), "w")


@pytest.mark.parametrize("section", ["collections_layout", "ledger_layout"])
@pytest.mark.parametrize("bad", ["0", "-1", "1.5", '"2"', "true"],
                         ids=["zero", "negative", "fraction", "text", "bool"])
def test_a_bad_header_row_in_the_file_raises(tmp_path, repo, section, bad):
    raises_naming(tmp_path, repo, f"[{section}]", "header_row",
                  changes={(section, "header_row"): bad})


@pytest.mark.parametrize("section", ["collections_layout", "ledger_layout"])
def test_equal_rows_in_the_file_raise(tmp_path, repo, section):
    raises_naming(tmp_path, repo, f"[{section}]", "first_data_row",
                  changes={(section, "header_row"): "4", (section, "first_data_row"): "4"})


# --- date_format_setting ---------------------------------------------------------------

@pytest.mark.parametrize("good", ["%d%m%Y", "%Y%m%d", "%Y-%m-%d", "%d.%m.%Y"])
def test_date_format_setting_accepts_a_full_date_format(good):
    assert date_format_setting({"f": good}, "f", "w") == good


@pytest.mark.parametrize("bad", ["%m%Y", "%Y", "%d%m", "%Y%m", "%d%Y", "abc"],
                         ids=["no-day", "year-only", "no-year", "no-day-2", "no-month",
                              "not-a-format"])
def test_date_format_setting_rejects_a_format_that_loses_part_of_a_date(bad):
    with pytest.raises(ValueError) as error:
        date_format_setting({"f": bad}, "f", "w")
    message = str(error.value)
    assert message.startswith("w: f")
    assert repr(bad) in message


@pytest.mark.parametrize("bad", ["", "   ", 5, None, ["%d%m%Y"]],
                         ids=["empty", "blank", "number", "none", "list"])
def test_date_format_setting_rejects_an_empty_or_non_text_value(bad):
    with pytest.raises(ValueError, match="w: f is .*not a date format"):
        date_format_setting({"f": bad}, "f", "w")


def test_date_format_setting_rejects_a_dangling_percent():
    with pytest.raises(ValueError, match="w: f"):
        date_format_setting({"f": "%d%m%Y%"}, "f", "w")


def test_format_sample_dates_vary_day_month_weekday_and_name_lengths():
    assert FORMAT_SAMPLE_DATES == [date(2025, 5, 2), date(2025, 6, 2), date(2026, 9, 30),
                                   date(2026, 12, 31)]
    assert len({sample.weekday() for sample in FORMAT_SAMPLE_DATES}) == 4
    assert {sample.month for sample in FORMAT_SAMPLE_DATES} >= {5, 9}  # May / September


def written_lengths(date_format):
    return sorted({len(sample.strftime(date_format)) for sample in FORMAT_SAMPLE_DATES})


@pytest.mark.parametrize("bad", ["%d%B%Y", "%A%d%m%Y", "%d%m%Y%B", "%Y%m%d %A", "%B %d %Y"],
                         ids=["month-name", "weekday-first", "month-name-last",
                              "weekday-last", "month-name-first"])
def test_date_format_setting_rejects_a_format_of_varying_length(bad):
    assert len(written_lengths(bad)) > 1  # guards the sample: it must vary in length
    with pytest.raises(ValueError) as error:
        date_format_setting({"f": bad}, "f", "w")
    message = str(error.value)
    assert message == (f"w: f {bad!r} writes dates of different lengths "
                       f"{written_lengths(bad)}; use numbers only, e.g. '%d%m%Y'")


def test_date_format_setting_month_name_lengths_are_named():
    # In the C / English locale: '02May2025' 9, '02June2025' 10, '31December2026' 14,
    # '30September2026' 15.
    if written_lengths("%d%B%Y") != [9, 10, 14, 15]:
        pytest.skip("month names are not the English ones in this locale")
    with pytest.raises(ValueError, match=r"different lengths \[9, 10, 14, 15\]"):
        date_format_setting({"f": "%d%B%Y"}, "f", "w")


def test_date_format_setting_abbreviated_month_depends_on_the_locale():
    """'%d%b%Y' passes when every abbreviated month has the same length (e.g. 'May', 'Jun',
    'Sep', 'Dec' in English) and raises when the locale's abbreviations differ."""
    if len(written_lengths("%d%b%Y")) == 1:
        assert date_format_setting({"f": "%d%b%Y"}, "f", "w") == "%d%b%Y"
    else:
        with pytest.raises(ValueError, match="different lengths"):
            date_format_setting({"f": "%d%b%Y"}, "f", "w")


@pytest.mark.parametrize("good", ["%d%m%Y", "%Y%m%d", "%d%m%y", "%y%m%d", "%Y-%m-%d",
                                  "%d.%m.%Y", "%Y_%m_%d", "%Y%j"])
def test_date_format_setting_accepts_numeric_formats(good):
    assert len(written_lengths(good)) == 1
    assert date_format_setting({"f": good}, "f", "w") == good


def test_date_format_setting_checks_read_back_before_length():
    # '%m%Y' loses the day: the read-back message, not the length one.
    with pytest.raises(ValueError, match="cannot write and read back a full date"):
        date_format_setting({"f": "%m%Y"}, "f", "w")


def test_a_varying_length_date_format_in_the_file_names_file_and_section(tmp_path, repo):
    raises_naming(tmp_path, repo, "[collections_layout]", "file_name_date_format",
                  "'%d%B%Y'", "different lengths",
                  changes={("collections_layout", "file_name_date_format"): '"%d%B%Y"'})


# --- headers_setting -------------------------------------------------------------------

FIELDS = ["company_code", "customer", "bank"]
OPTIONAL = {"bank"}


def headers(value):
    return headers_setting({"h": value}, "h", FIELDS, OPTIONAL, "w")


def test_headers_setting_returns_the_mapping_in_file_order():
    value = {"Customer": "customer", "Company Code": "company_code", "Bank": "bank"}
    result = headers(value)
    assert result == value
    assert list(result) == ["Customer", "Company Code", "Bank"]


def test_headers_setting_returns_a_copy():
    value = {"Company Code": "company_code", "Customer": "customer"}
    result = headers(value)
    assert result is not value
    result["Extra"] = "bank"
    assert value == {"Company Code": "company_code", "Customer": "customer"}


def test_headers_setting_result_does_not_follow_later_changes_to_the_input():
    value = {"Company Code": "company_code", "Customer": "customer"}
    result = headers(value)
    value["Bank"] = "bank"
    assert "Bank" not in result


def test_headers_setting_allows_an_optional_field_to_be_absent():
    assert headers({"Company Code": "company_code", "Customer": "customer"}) == {
        "Company Code": "company_code", "Customer": "customer"}


def test_headers_setting_allows_two_headers_for_one_field():
    value = {"Company Code": "company_code", "Customer": "customer",
             "MÜŞTERİ": "customer"}
    assert headers(value) == value


def test_headers_setting_rejects_an_unknown_field_naming_header_and_value():
    with pytest.raises(ValueError) as error:
        headers({"Company Code": "company_code", "Customer": "client"})
    message = str(error.value)
    assert message.startswith("w.h:")
    assert "'Customer'" in message and "'client'" in message


@pytest.mark.parametrize("bad", [5, None, ["customer"], {"a": "b"}, True],
                         ids=["number", "none", "list", "table", "bool"])
def test_headers_setting_rejects_a_non_text_field(bad):
    with pytest.raises(ValueError, match="'Customer' maps to"):
        headers({"Company Code": "company_code", "Customer": bad})


def test_headers_setting_field_names_are_compared_exactly():
    with pytest.raises(ValueError, match="'Customer'"):
        headers({"Company Code": "company_code", "Customer": "Customer"})


def test_headers_setting_names_a_missing_required_field_as_a_list():
    with pytest.raises(ValueError) as error:
        headers({"Company Code": "company_code", "Bank": "bank"})
    assert str(error.value) == "w.h: no header for the required fields ['customer']"


def test_headers_setting_names_every_missing_required_field_in_field_order():
    with pytest.raises(ValueError) as error:
        headers({"Bank": "bank"})
    assert "['company_code', 'customer']" in str(error.value)


@pytest.mark.parametrize("first, second", [
    ("Customer", " customer "), ("Customer", "CUSTOMER"), ("Customer", "Customer\t"),
    ("MÜŞTERİ", "müşteri"),
], ids=["spaces-and-case", "upper", "tab", "turkish-dotted-i"])
def test_headers_setting_rejects_two_headers_equal_after_simplifying(first, second):
    value = {"Company Code": "company_code", first: "customer", second: "customer"}
    with pytest.raises(ValueError) as error:
        headers(value)
    message = str(error.value)
    assert f"'{first}'" in message and f"'{second}'" in message
    assert "same header" in message


def test_headers_setting_rejects_inner_space_variants():
    with pytest.raises(ValueError, match="'Company Code' and 'company  code'"):
        headers({"Company Code": "company_code", "company  code": "company_code",
                 "Customer": "customer"})


def test_headers_setting_rejects_simplified_duplicates_even_for_different_fields():
    with pytest.raises(ValueError, match="same header"):
        headers({"Company Code": "company_code", "Customer": "customer", "CUSTOMER": "bank"})


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "blank"])
def test_headers_setting_rejects_an_empty_header_text(blank):
    with pytest.raises(ValueError, match="empty"):
        headers({"Company Code": "company_code", "Customer": "customer", blank: "bank"})


@pytest.mark.parametrize("bad", [{}, "Customer", ["Customer"], None, 5],
                         ids=["empty-table", "text", "list", "none", "number"])
def test_headers_setting_rejects_an_empty_or_non_table_value(bad):
    with pytest.raises(ValueError, match="w: h is .*not a table"):
        headers(bad)


def test_headers_setting_does_not_modify_the_input_when_it_raises():
    value = {"Customer": "customer"}
    with pytest.raises(ValueError):
        headers(value)
    assert value == {"Customer": "customer"}


def test_headers_with_turkish_and_english_spellings_load(tmp_path, repo):
    with_turkish = dict(COLLECTION_HEADERS)
    with_turkish["MÜŞTERİ"] = "customer"
    with_turkish["ŞİRKET KODU"] = "company_code"
    settings = load(tmp_path, repo,
                    changes={("collections_layout", "headers"): headers_toml(with_turkish)})
    assert settings.collections_layout.headers == with_turkish


def test_turkish_case_variants_in_the_file_raise(tmp_path, repo):
    with_turkish = dict(COLLECTION_HEADERS)
    with_turkish["MÜŞTERİ"] = "customer"
    with_turkish["müşteri"] = "customer"
    raises_naming(tmp_path, repo, "[collections_layout].headers", "same header",
                  changes={("collections_layout", "headers"): headers_toml(with_turkish)})


def test_collections_headers_without_the_optional_bank_load(tmp_path, repo):
    without_bank = {h: f for h, f in COLLECTION_HEADERS.items() if f != "bank"}
    settings = load(tmp_path, repo,
                    changes={("collections_layout", "headers"): headers_toml(without_bank)})
    assert "bank" not in settings.collections_layout.headers.values()


@pytest.mark.parametrize("field", sorted(set(COLLECTION_FIELDS) - OPTIONAL_FIELDS))
def test_every_required_collection_field_must_have_a_header(tmp_path, repo, field):
    without = {h: f for h, f in COLLECTION_HEADERS.items() if f != field}
    raises_naming(tmp_path, repo, "[collections_layout].headers", f"'{field}'",
                  changes={("collections_layout", "headers"): headers_toml(without)})


@pytest.mark.parametrize("field", sorted(set(LEDGER_FIELDS) - LEDGER_OPTIONAL_FIELDS))
def test_every_required_ledger_field_must_have_a_header(tmp_path, repo, field):
    without = {h: f for h, f in LEDGER_HEADERS.items() if f != field}
    raises_naming(tmp_path, repo, "[ledger_layout].headers", f"'{field}'",
                  changes={("ledger_layout", "headers"): headers_toml(without)})


def test_the_ledger_headers_in_the_test_file_cover_exactly_the_required_fields():
    # Guards this file's own sample: LEDGER_HEADERS is meant to be the minimal valid mapping.
    assert set(LEDGER_HEADERS.values()) == set(LEDGER_FIELDS) - LEDGER_OPTIONAL_FIELDS


def test_the_ledger_optional_fields_are_the_four_and_not_clearing_document():
    assert LEDGER_OPTIONAL_FIELDS == {"customer_name", "document_date", "baseline_date",
                                      "payment_terms"}
    assert "clearing_document" in set(LEDGER_FIELDS) - LEDGER_OPTIONAL_FIELDS


def test_ledger_headers_without_clearing_document_raise_the_exact_message(tmp_path, repo):
    without = {h: f for h, f in LEDGER_HEADERS.items() if f != "clearing_document"}
    message = raises_naming(tmp_path, repo, "[ledger_layout].headers",
                            changes={("ledger_layout", "headers"): headers_toml(without)})
    assert message.endswith(
        "[ledger_layout].headers: no header for the required fields ['clearing_document']")


def test_headers_setting_with_the_ledger_fields_requires_clearing_document():
    without = {h: f for h, f in LEDGER_HEADERS.items() if f != "clearing_document"}
    with pytest.raises(ValueError) as error:
        headers_setting({"h": without}, "h", LEDGER_FIELDS, LEDGER_OPTIONAL_FIELDS, "w")
    assert str(error.value) == "w.h: no header for the required fields ['clearing_document']"


def test_ledger_headers_without_clearing_date_and_document_name_both_in_field_order(tmp_path,
                                                                                    repo):
    without = {h: f for h, f in LEDGER_HEADERS.items()
               if f not in ("clearing_date", "clearing_document")}
    message = raises_naming(tmp_path, repo, "[ledger_layout].headers",
                            changes={("ledger_layout", "headers"): headers_toml(without)})
    assert message.endswith(
        "no header for the required fields ['clearing_date', 'clearing_document']")


@pytest.mark.parametrize("field", sorted(LEDGER_OPTIONAL_FIELDS))
def test_each_optional_ledger_field_may_be_mapped(tmp_path, repo, field):
    # The minimal mapping plus one optional field still loads.
    header = next(h for h, f in EXPECTED_LEDGER_HEADERS.items() if f == field)
    with_one = dict(LEDGER_HEADERS, **{header: field})
    settings = load(tmp_path, repo,
                    changes={("ledger_layout", "headers"): headers_toml(with_one)})
    assert settings.ledger_layout.headers == with_one


def test_ledger_headers_with_all_four_optional_fields_absent_load(tmp_path, repo):
    settings = load(tmp_path, repo)
    mapped = set(settings.ledger_layout.headers.values())
    assert mapped.isdisjoint(LEDGER_OPTIONAL_FIELDS)
    assert "clearing_document" in mapped


def test_a_clearing_document_header_mapped_twice_loads(tmp_path, repo):
    # An English and a Turkish header for the same field are allowed, as for other fields.
    both = dict(LEDGER_HEADERS, **{"Denkleştirme Belgesi": "clearing_document"})
    settings = load(tmp_path, repo, changes={("ledger_layout", "headers"): headers_toml(both)})
    assert list(settings.ledger_layout.headers.values()).count("clearing_document") == 2


# --- document_types_setting ------------------------------------------------------------

def document_types(value):
    return document_types_setting({"d": value}, "d", "w")


GOOD_TYPES = {"invoice": ["RV"], "payment": ["DZ"], "credit_memo": ["DG"]}


def test_document_types_setting_returns_tuples_per_role():
    result = document_types(GOOD_TYPES)
    assert result == {"invoice": ("RV",), "payment": ("DZ",), "credit_memo": ("DG",)}
    assert set(result) == set(DOCUMENT_ROLES)


def test_document_types_setting_allows_several_codes_per_role():
    result = document_types(dict(GOOD_TYPES, payment=["DZ", "DA"]))
    assert result["payment"] == ("DZ", "DA")


def test_document_types_setting_does_not_modify_the_input():
    value = {role: list(codes) for role, codes in GOOD_TYPES.items()}
    document_types(value)
    assert value == GOOD_TYPES


@pytest.mark.parametrize("role", DOCUMENT_ROLES)
def test_document_types_setting_rejects_a_missing_role(role):
    value = {r: c for r, c in GOOD_TYPES.items() if r != role}
    with pytest.raises(ValueError) as error:
        document_types(value)
    message = str(error.value)
    assert message.startswith("w.d")
    assert "missing" in message and role in message


def test_document_types_setting_rejects_an_unknown_role():
    with pytest.raises(ValueError) as error:
        document_types(dict(GOOD_TYPES, reversal=["AB"]))
    assert "unknown ['reversal']" in str(error.value)


def test_document_types_setting_names_both_sides_of_a_typo():
    value = {"invoice": ["RV"], "payment": ["DZ"], "credit_memos": ["DG"]}
    with pytest.raises(ValueError) as error:
        document_types(value)
    message = str(error.value)
    assert "missing ['credit_memo']" in message and "unknown ['credit_memos']" in message


@pytest.mark.parametrize("bad", [["RV", "DZ", "DG"], "RV", None, 5],
                         ids=["list", "text", "none", "number"])
def test_document_types_setting_rejects_a_non_table(bad):
    with pytest.raises(ValueError, match="not a section"):
        document_types(bad)


@pytest.mark.parametrize("bad", ["RV", [], ["RV", ""], ["RV", "  "], ["RV", 1], [["RV"]]],
                         ids=["text", "empty-list", "empty-code", "blank-code", "number",
                              "nested"])
def test_document_types_setting_rejects_a_bad_code_list(bad):
    with pytest.raises(ValueError) as error:
        document_types(dict(GOOD_TYPES, invoice=bad))
    assert "invoice" in str(error.value)


def test_document_types_setting_rejects_a_duplicate_within_a_role():
    with pytest.raises(ValueError) as error:
        document_types(dict(GOOD_TYPES, payment=["DZ", "DA", "DZ"]))
    assert str(error.value) == "w.d: payment lists ['DZ'] more than once"


def test_document_types_setting_rejects_a_code_in_two_roles():
    with pytest.raises(ValueError) as error:
        document_types(dict(GOOD_TYPES, credit_memo=["DG", "RV"]))
    message = str(error.value)
    assert "'RV'" in message and "invoice" in message and "credit_memo" in message


def test_document_types_setting_strips_and_upper_cases_codes():
    result = document_types({"invoice": [" rv "], "payment": ["dz", "Da "],
                             "credit_memo": ["DG"]})
    assert result == {"invoice": ("RV",), "payment": ("DZ", "DA"), "credit_memo": ("DG",)}
    for codes in result.values():
        assert type(codes) is tuple


def test_document_types_setting_keeps_the_order_of_codes_in_a_role():
    result = document_types(dict(GOOD_TYPES, payment=["zz", "DZ", "aa"]))
    assert result["payment"] == ("ZZ", "DZ", "AA")


def test_document_types_setting_does_not_modify_lower_case_input():
    value = {"invoice": [" rv "], "payment": ["dz"], "credit_memo": ["dg"]}
    document_types(value)
    assert value == {"invoice": [" rv "], "payment": ["dz"], "credit_memo": ["dg"]}


@pytest.mark.parametrize("written", [["DZ", "dz"], ["DZ", " DZ "], ["dz", "Dz"], ["DA", "dz", "DZ"]],
                         ids=["case", "spaces", "two-cases", "among-others"])
def test_document_types_setting_rejects_a_code_twice_after_normalising(written):
    with pytest.raises(ValueError) as error:
        document_types(dict(GOOD_TYPES, payment=written))
    assert str(error.value) == (f"w.d: payment lists the same code twice when case and "
                                f"spaces are ignored: {written}")


def test_document_types_setting_exact_duplicate_uses_the_plain_message():
    # An exact duplicate is caught first, by text_list_setting.
    with pytest.raises(ValueError, match=r"payment lists \['DZ'\] more than once"):
        document_types(dict(GOOD_TYPES, payment=["DZ", "DZ"]))


@pytest.mark.parametrize("invoice_codes, payment_codes", [
    (["RV"], ["rv"]), (["rv"], ["RV"]), ([" rv"], ["Rv "]), (["RV", "AB"], ["DZ", "rv"]),
], ids=["upper-lower", "lower-upper", "spaces", "among-others"])
def test_document_types_setting_rejects_a_code_in_two_roles_after_upper_casing(invoice_codes,
                                                                             payment_codes):
    with pytest.raises(ValueError) as error:
        document_types({"invoice": invoice_codes, "payment": payment_codes,
                        "credit_memo": ["DG"]})
    assert str(error.value) == "w.d: code 'RV' is listed for both invoice and payment"


def test_document_types_setting_names_the_roles_in_role_order():
    # The first role (in DOCUMENT_ROLES order) holding the code is named first.
    with pytest.raises(ValueError) as error:
        document_types({"invoice": ["RV"], "payment": ["DZ"], "credit_memo": ["DZ"]})
    assert str(error.value) == "w.d: code 'DZ' is listed for both payment and credit_memo"


# --- types: fill_missing_days_from_ledger (bool_setting) ------------------------------

@pytest.mark.parametrize("text, expected", [("true", True), ("false", False)])
def test_fill_missing_days_from_ledger_accepts_true_and_false(tmp_path, repo, text, expected):
    value = load(tmp_path, repo,
                 changes={("data", "fill_missing_days_from_ledger"): text}
                 ).data.fill_missing_days_from_ledger
    assert value is expected


@pytest.mark.parametrize("bad", ["1", "0", '"true"', '"false"', '"yes"', "1.0", "[true]",
                                 "2026-08-28"],
                         ids=["one", "zero", "string-true", "string-false", "string-yes",
                              "float", "list", "date"])
def test_fill_missing_days_from_ledger_that_is_not_a_bool_raises(tmp_path, repo, bad):
    message = raises_naming(tmp_path, repo, "[data]", "fill_missing_days_from_ledger",
                            changes={("data", "fill_missing_days_from_ledger"): bad})
    assert "true or false" in message


def test_a_missing_fill_missing_days_from_ledger_names_it(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[data]", "fill_missing_days_from_ledger",
                            changes={("data", "fill_missing_days_from_ledger"): REMOVE})
    assert "missing" in message


def test_fill_missing_days_from_ledger_is_a_field_of_data_settings():
    names = [field.name for field in dataclasses.fields(DataSettings)]
    assert "fill_missing_days_from_ledger" in names


@pytest.mark.parametrize("value", [True, False])
def test_bool_setting_returns_a_bool_unchanged(value):
    assert bool_setting({"flag": value}, "flag", "f.toml [s]") is value


@pytest.mark.parametrize("bad", [1, 0, "true", "false", None, 1.0, [True]],
                         ids=["one", "zero", "string-true", "string-false", "none", "float",
                              "list"])
def test_bool_setting_rejects_anything_but_a_bool_naming_where_and_key(bad):
    with pytest.raises(ValueError) as error:
        bool_setting({"flag": bad}, "flag", "f.toml [s]")
    message = str(error.value)
    assert message.startswith("f.toml [s]")
    assert "flag" in message
    assert repr(bad) in message


def test_bool_setting_with_the_key_absent_raises_key_error():
    # load_settings checks the keys first, so a direct call without the key is a KeyError.
    with pytest.raises(KeyError):
        bool_setting({}, "flag", "f.toml [s]")


# --- [history]: windows (window_list_setting) ----------------------------------------------

def test_the_history_section_is_read(tmp_path, repo):
    history = load(tmp_path, repo).history
    assert history == HistorySettings(windows=(4, 13, 52), use_ledger_before_first_file=True)
    assert type(history.windows) is tuple
    assert all(type(window) is int for window in history.windows)


def test_a_missing_history_section_names_it(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "history", drop_section="history")
    assert "missing" in message


@pytest.mark.parametrize("good, expected", [
    ("[4, 13, 52]", (4, 13, 52)), ("[13, 4, 52]", (4, 13, 52)), ("[52, 13, 4]", (4, 13, 52)),
    ("[1]", (1,)), ("[8]", (8,)), ("[104, 1]", (1, 104)),
], ids=["sorted", "unsorted", "reversed", "one", "single", "two-unsorted"])
def test_windows_are_returned_sorted(tmp_path, repo, good, expected):
    # reference_window must be one of the windows (U10.4d), so it follows the list here.
    settings = load(tmp_path, repo, changes={("history", "windows"): good,
                                             ("diagnostic", "reference_window"): str(expected[0])})
    assert settings.history.windows == expected
    assert type(settings.history.windows) is tuple


def test_an_empty_windows_list_raises(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[history]", "windows",
                            changes={("history", "windows"): "[]"})
    assert "non-empty list" in message


@pytest.mark.parametrize("bad", ["4", '"4, 13"', "true", "{ a = 4 }"],
                         ids=["int", "string", "bool", "table"])
def test_windows_that_is_not_a_list_raises(tmp_path, repo, bad):
    message = raises_naming(tmp_path, repo, "[history]", "windows",
                            changes={("history", "windows"): bad})
    assert "non-empty list" in message


@pytest.mark.parametrize("bad, shown", [
    ("[4, 0]", "0"), ("[-1, 13]", "-1"), ("[4, 1.5]", "1.5"), ("[4.0]", "4.0"),
    ('["4"]', "'4'"), ("[true]", "True"), ("[4, false]", "False"), ("[[4]]", "[4]"),
], ids=["zero", "minus-one", "fraction", "whole-float", "string", "true", "false", "nested"])
def test_windows_with_a_bad_item_raises_naming_it(tmp_path, repo, bad, shown):
    message = raises_naming(tmp_path, repo, "[history]", "windows", f"holds {shown}",
                            changes={("history", "windows"): bad})
    assert "1 or more" in message


def test_windows_with_duplicates_raises_naming_them(tmp_path, repo):
    message = raises_naming(tmp_path, repo, "[history]", "windows", "lists [4, 13] more than once",
                            changes={("history", "windows"): "[13, 4, 52, 4, 13]"})
    assert "52" not in message


def test_a_windows_typo_names_the_missing_and_the_unknown_setting(tmp_path, repo):
    message = raises_naming(
        tmp_path, repo, "[history]", "'windows'", "'window'",
        changes={("history", "windows"): REMOVE, ("history", "window"): "[4]"})
    assert "missing" in message and "unknown" in message


def test_window_list_setting_direct_calls():
    assert window_list_setting({"w": [13, 4, 52]}, "w", "here") == (4, 13, 52)
    assert type(window_list_setting({"w": [3]}, "w", "here")) is tuple
    with pytest.raises(ValueError, match="here: w is"):
        window_list_setting({"w": []}, "w", "here")
    with pytest.raises(ValueError, match="here: w is"):
        window_list_setting({"w": (4, 13)}, "w", "here")
    with pytest.raises(ValueError, match="here: w holds 0"):
        window_list_setting({"w": [4, 0]}, "w", "here")
    with pytest.raises(ValueError, match="here: w holds True"):
        window_list_setting({"w": [True]}, "w", "here")
    with pytest.raises(ValueError, match=r"here: w lists \[2\] more than once"):
        window_list_setting({"w": [2, 2]}, "w", "here")


def test_window_list_setting_does_not_modify_the_list():
    value = [13, 4, 52]
    window_list_setting({"w": value}, "w", "here")
    assert value == [13, 4, 52]


# --- [history]: use_ledger_before_first_file (bool_setting) -------------------------------

@pytest.mark.parametrize("text, expected", [("true", True), ("false", False)])
def test_use_ledger_before_first_file_accepts_true_and_false(tmp_path, repo, text, expected):
    value = load(tmp_path, repo, changes={("history", "use_ledger_before_first_file"): text}
                 ).history.use_ledger_before_first_file
    assert value is expected


@pytest.mark.parametrize("bad", ["1", "0", '"true"', '"false"', "1.0", "[true]"],
                         ids=["one", "zero", "string-true", "string-false", "float", "list"])
def test_use_ledger_before_first_file_that_is_not_a_bool_raises(tmp_path, repo, bad):
    message = raises_naming(tmp_path, repo, "[history]", "use_ledger_before_first_file",
                            changes={("history", "use_ledger_before_first_file"): bad})
    assert "true or false" in message


def test_history_settings_do_not_borrow_from_data(tmp_path, repo):
    settings = load(tmp_path, repo,
                    changes={("history", "use_ledger_before_first_file"): "false",
                             ("data", "fill_missing_days_from_ledger"): "true"})
    assert settings.history.use_ledger_before_first_file is False
    assert settings.data.fill_missing_days_from_ledger is True


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_history_settings():
    history = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT).history
    assert history == HistorySettings(windows=(4, 13, 52), use_ledger_before_first_file=True)


# --- [stage1] ---------------------------------------------------------------------------
# Every value is checked through the file (the message names the file and the section,
# exactly) and, for the new helpers, by direct calls.

STAGE1 = f"{FILE_NAME} [stage1]"
DEBUG = f"{FILE_NAME} [debug]"


def exact_message(tmp_path, repo, changes):
    with pytest.raises(ValueError) as error:
        load(tmp_path, repo, changes=changes)
    return str(error.value)


def test_the_stage1_section_is_read(tmp_path, repo):
    stage1 = load(tmp_path, repo).stage1
    assert isinstance(stage1, Stage1Settings)
    assert stage1 == Stage1Settings(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
        random_seed=1, feature_blocks=("history", "calendar", "receivables"),
        company_as_feature=True, baseline_c=1.0)


def test_the_stage1_values_have_the_right_python_types(tmp_path, repo):
    stage1 = load(tmp_path, repo).stage1
    for name in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "random_seed"):
        assert type(getattr(stage1, name)) is int, name
    assert type(stage1.learning_rate) is float
    assert type(stage1.feature_blocks) is tuple
    assert stage1.company_as_feature is True
    assert type(stage1.baseline_c) is float


def test_stage1_settings_fields_are_in_the_documented_order():
    assert [field.name for field in dataclasses.fields(Stage1Settings)] == [
        "max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "random_seed",
        "feature_blocks", "company_as_feature", "baseline_c"]


def test_settings_has_a_stage1_field_between_history_and_debug():
    names = [field.name for field in dataclasses.fields(Settings)]
    assert "stage1" in names
    assert names.index("history") < names.index("stage1") < names.index("debug")


def test_a_missing_stage1_section_raises_the_exact_message(tmp_path, repo):
    with pytest.raises(ValueError) as error:
        load(tmp_path, repo, drop_section="stage1")
    assert str(error.value) == (
        f"{FILE_NAME}: missing ['stage1']; expected exactly ['collections_layout', "
        f"'companies', 'data', 'debug', 'diagnostic', 'history', 'ledger_layout', "
        f"'receivables', 'stage1', 'stage2']")


def test_an_empty_stage1_section_names_every_stage1_setting(tmp_path, repo):
    changes = {("stage1", key): REMOVE for key in VALID["stage1"]}
    message = raises_naming(tmp_path, repo, "[stage1]", changes=changes)
    for key in VALID["stage1"]:
        assert key in message
    assert "missing" in message


def test_a_stage1_typo_names_the_missing_and_the_unknown_setting(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("stage1", "max_iter"): REMOVE,
                                             ("stage1", "max_iters"): "300"})
    assert message == (
        f"{STAGE1}: missing ['max_iter']; unknown ['max_iters']; expected exactly "
        f"['baseline_c', 'company_as_feature', 'feature_blocks', "
        f"'learning_rate', "
        f"'max_iter', 'max_leaf_nodes', 'min_samples_leaf', 'random_seed']")


# --- [stage1] max_iter and min_samples_leaf (count_setting) ------------------------------

@pytest.mark.parametrize("key", ["max_iter", "min_samples_leaf"])
@pytest.mark.parametrize("good, expected", [("1", 1), ("2", 2), ("1_000", 1000)])
def test_stage1_counts_accept_a_whole_number_of_1_or_more(tmp_path, repo, key, good, expected):
    value = getattr(load(tmp_path, repo, changes={("stage1", key): good}).stage1, key)
    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize("key", ["max_iter", "min_samples_leaf"])
@pytest.mark.parametrize("bad, shown", [
    ("0", "0"), ("-1", "-1"), ("1.5", "1.5"), ("300.0", "300.0"), ('"300"', "'300'"),
    ("true", "True"), ("false", "False"), ("[300]", "[300]"),
], ids=["zero", "negative", "fraction", "whole-float", "string", "true", "false", "list"])
def test_stage1_counts_reject_non_counts_with_the_exact_message(tmp_path, repo, key, bad,
                                                                 shown):
    message = exact_message(tmp_path, repo, {("stage1", key): bad})
    assert message == f"{STAGE1}: {key} is {shown}, not a whole number of 1 or more"


def test_max_iter_and_min_samples_leaf_are_read_separately(tmp_path, repo):
    stage1 = load(tmp_path, repo, changes={("stage1", "max_iter"): "7",
                                           ("stage1", "min_samples_leaf"): "3"}).stage1
    assert (stage1.max_iter, stage1.min_samples_leaf) == (7, 3)


# --- [stage1] learning_rate (rate_setting) --------------------------------------------------

@pytest.mark.parametrize("good, expected", [
    ("0.05", 0.05), ("1", 1.0), ("1.0", 1.0), ("0.5", 0.5), ("1e-9", 1e-9), ("0.999", 0.999),
], ids=["usual", "one-int", "one-float", "half", "tiny", "just-below-one"])
def test_learning_rate_accepts_numbers_above_0_and_at_most_1(tmp_path, repo, good, expected):
    value = load(tmp_path, repo, changes={("stage1", "learning_rate"): good}).stage1.learning_rate
    assert value == expected
    assert type(value) is float


@pytest.mark.parametrize("bad, shown", [
    ("0", "0"), ("0.0", "0.0"), ("-0.05", "-0.05"), ("1.0000001", "1.0000001"), ("2", "2"),
    ("inf", "inf"), ("nan", "nan"), ("-inf", "-inf"), ('"0.05"', "'0.05'"), ("true", "True"),
    ("false", "False"), ("[0.05]", "[0.05]"),
], ids=["zero", "zero-float", "negative", "just-above-one", "two", "inf", "nan", "minus-inf",
        "string", "true", "false", "list"])
def test_learning_rate_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage1", "learning_rate"): bad})
    assert message == f"{STAGE1}: learning_rate is {shown}, not a number above 0 and at most 1"


def test_rate_setting_direct_calls():
    assert rate_setting({"r": 0.05}, "r", "here") == 0.05
    assert rate_setting({"r": 1}, "r", "here") == 1.0
    assert type(rate_setting({"r": 1}, "r", "here")) is float
    assert rate_setting({"r": 5e-324}, "r", "here") == 5e-324


@pytest.mark.parametrize("bad", [0, 0.0, -1, 1.01, True, False, None, "0.1", [0.1],
                                 float("nan"), float("inf")])
def test_rate_setting_rejects_naming_where_and_key(bad):
    with pytest.raises(ValueError) as error:
        rate_setting({"r": bad}, "r", "here")
    assert str(error.value) == f"here: r is {bad!r}, not a number above 0 and at most 1"


def test_rate_setting_true_is_rejected_although_it_equals_1():
    # True == 1 in Python and 0 < True <= 1 holds: only the explicit bool check stops it.
    with pytest.raises(ValueError, match="here: r is True"):
        rate_setting({"r": True}, "r", "here")


# --- [stage1] max_leaf_nodes (leaf_count_setting) -------------------------------------------

@pytest.mark.parametrize("good, expected", [("2", 2), ("3", 3), ("31", 31), ("1_024", 1024)])
def test_max_leaf_nodes_accepts_a_whole_number_of_2_or_more(tmp_path, repo, good, expected):
    value = load(tmp_path, repo,
                 changes={("stage1", "max_leaf_nodes"): good}).stage1.max_leaf_nodes
    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize("bad, shown", [
    ("1", "1"), ("0", "0"), ("-31", "-31"), ("31.0", "31.0"), ("2.5", "2.5"), ('"31"', "'31'"),
    ("true", "True"), ("false", "False"), ("[31]", "[31]"),
], ids=["one", "zero", "negative", "whole-float", "fraction", "string", "true", "false",
        "list"])
def test_max_leaf_nodes_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage1", "max_leaf_nodes"): bad})
    assert message == f"{STAGE1}: max_leaf_nodes is {shown}, not a whole number of 2 or more"


def test_leaf_count_setting_direct_calls():
    assert leaf_count_setting({"n": 2}, "n", "here") == 2
    assert type(leaf_count_setting({"n": 31}, "n", "here")) is int
    for bad in (1, 0, -2, 2.0, True, None, "2"):
        with pytest.raises(ValueError) as error:
            leaf_count_setting({"n": bad}, "n", "here")
        assert str(error.value) == f"here: n is {bad!r}, not a whole number of 2 or more"


# --- [stage1] random_seed (seed_setting) ----------------------------------------------------

@pytest.mark.parametrize("good, expected", [("0", 0), ("1", 1), ("123_456", 123456),
                                            ("4294967295", 4294967295)])
def test_random_seed_accepts_a_whole_number_of_0_or_more(tmp_path, repo, good, expected):
    value = load(tmp_path, repo, changes={("stage1", "random_seed"): good}).stage1.random_seed
    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize("bad, shown", [
    ("-1", "-1"), ("-42", "-42"), ("1.0", "1.0"), ("1.5", "1.5"), ('"1"', "'1'"),
    ("true", "True"), ("false", "False"), ("[1]", "[1]"),
    ("2025-06-02", "datetime.date(2025, 6, 2)"), ("{ seed = 1 }", "{'seed': 1}"),
], ids=["negative", "very-negative", "whole-float", "fraction", "string", "true", "false",
        "list", "date", "table"])
def test_random_seed_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage1", "random_seed"): bad})
    assert message == (f"{STAGE1}: random_seed is {shown}, not a random seed (a whole number, "
                       f"0 or more)")


def test_random_seed_message_no_longer_says_week_number(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("stage1", "random_seed"): "-1"})
    assert "week number" not in message


def test_seed_setting_accepts_0_and_positive_ints():
    assert seed_setting({"s": 0}, "s", "here") == 0
    assert seed_setting({"s": 1}, "s", "here") == 1
    assert seed_setting({"s": 2**40}, "s", "here") == 2**40
    assert type(seed_setting({"s": 7}, "s", "here")) is int


@pytest.mark.parametrize("bad", [-1, -2**40, True, False, 1.0, 0.0, 1.5, float("nan"), None,
                                 "1", [1], (1,), {"s": 1}],
                         ids=["minus-one", "very-negative", "true", "false", "one-float",
                              "zero-float", "fraction", "nan", "none", "string", "list",
                              "tuple", "dict"])
def test_seed_setting_rejects_with_the_exact_message(bad):
    with pytest.raises(ValueError) as error:
        seed_setting({"s": bad}, "s", "here")
    assert str(error.value) == (f"here: s is {bad!r}, not a random seed (a whole number, "
                                f"0 or more)")


def test_seed_setting_false_is_rejected_although_it_equals_0():
    # False == 0 and isinstance(False, int): only the explicit bool check stops it.
    with pytest.raises(ValueError, match="^here: s is False, not a random seed"):
        seed_setting({"s": False}, "s", "here")


def test_seed_setting_does_not_change_week_setting_wording():
    # demo_origin and sample_week keep week_setting's own message.
    with pytest.raises(ValueError) as error:
        week_setting({"w": -1}, "w", "here")
    assert "week number" in str(error.value)
    assert "random seed" not in str(error.value)


# --- [stage1] feature_blocks (feature_blocks_setting) ---------------------------------------

def test_feature_blocks_constant():
    assert FEATURE_BLOCKS == ("history", "calendar", "receivables")
    assert type(FEATURE_BLOCKS) is tuple


@pytest.mark.parametrize("good, expected", [
    ('["history", "calendar", "receivables"]', ("history", "calendar", "receivables")),
    ('["receivables", "calendar", "history"]', ("history", "calendar", "receivables")),
    ('["calendar", "history", "receivables"]', ("history", "calendar", "receivables")),
    ('["receivables", "history"]', ("history", "receivables")),
    ('["calendar"]', ("calendar",)),
    ('["receivables"]', ("receivables",)),
    ('["history"]', ("history",)),
], ids=["in-order", "reversed", "shuffled", "two-out-of-order", "calendar-only",
        "receivables-only", "history-only"])
def test_feature_blocks_are_returned_in_feature_blocks_order(tmp_path, repo, good, expected):
    value = load(tmp_path, repo,
                 changes={("stage1", "feature_blocks"): good}).stage1.feature_blocks
    assert value == expected
    assert type(value) is tuple


@pytest.mark.parametrize("bad, shown", [
    ("[]", "[]"), ('"history"', "'history'"), ("1", "1"), ("true", "True"),
    ("{ history = true }", "{'history': True}"),
], ids=["empty", "text", "number", "bool", "table"])
def test_feature_blocks_not_a_non_empty_list_has_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage1", "feature_blocks"): bad})
    assert message == (f"{STAGE1}: feature_blocks is {shown}, not a non-empty list of feature "
                       f"blocks")


@pytest.mark.parametrize("bad, shown", [
    ('["history", "weather"]', "['weather']"),
    ('["History"]', "['History']"),
    ('[" history"]', "[' history']"),
    ('["history", 1]', "[1]"),
    ('["history", true]', "[True]"),
    ('[["history"]]', "[['history']]"),
    ('["weather", "calendar", "sales"]', "['weather', 'sales']"),
    ('["weather", "weather"]', "['weather', 'weather']"),
], ids=["one-unknown", "case", "space", "number", "bool", "nested", "two-unknown-in-order",
        "unknown-repeated"])
def test_feature_blocks_with_an_unknown_block_has_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage1", "feature_blocks"): bad})
    assert message == (f"{STAGE1}: feature_blocks holds {shown}; the feature blocks are "
                       f"['history', 'calendar', 'receivables']")


@pytest.mark.parametrize("bad, shown", [
    ('["history", "history"]', "['history', 'history']"),
    ('["calendar", "history", "calendar"]', "['calendar', 'history', 'calendar']"),
], ids=["twice", "twice-apart"])
def test_feature_blocks_listing_a_block_twice_has_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage1", "feature_blocks"): bad})
    assert message == (f"{STAGE1}: feature_blocks lists a feature block more than once: "
                       f"{shown}")


def test_feature_blocks_setting_direct_calls():
    assert feature_blocks_setting({"b": ["receivables", "history"]}, "b", "here") == (
        "history", "receivables")
    with pytest.raises(ValueError) as error:
        feature_blocks_setting({"b": ("history",)}, "b", "here")
    assert str(error.value) == "here: b is ('history',), not a non-empty list of feature blocks"
    with pytest.raises(ValueError) as error:
        feature_blocks_setting({"b": None}, "b", "here")
    assert str(error.value) == "here: b is None, not a non-empty list of feature blocks"


def test_feature_blocks_setting_does_not_modify_the_list():
    value = ["receivables", "history"]
    feature_blocks_setting({"b": value}, "b", "here")
    assert value == ["receivables", "history"]


# --- [stage1] company_as_feature (bool_setting) ----------------------------------------------

@pytest.mark.parametrize("key", ["company_as_feature"])
@pytest.mark.parametrize("text, expected", [("true", True), ("false", False)])
def test_stage1_switches_accept_true_and_false(tmp_path, repo, key, text, expected):
    value = getattr(load(tmp_path, repo, changes={("stage1", key): text}).stage1, key)
    assert value is expected


@pytest.mark.parametrize("key", ["company_as_feature"])
@pytest.mark.parametrize("bad, shown", [("1", "1"), ("0", "0"), ('"true"', "'true'"),
                                        ("1.0", "1.0"), ("[true]", "[True]")],
                         ids=["one", "zero", "string", "float", "list"])
def test_stage1_switches_reject_non_bools_with_the_exact_message(tmp_path, repo, key, bad,
                                                                 shown):
    message = exact_message(tmp_path, repo, {("stage1", key): bad})
    assert message == f"{STAGE1}: {key} is {shown}, not true or false"


def test_company_as_feature_and_diagnostic_run_are_read_separately(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("stage1", "company_as_feature"): "false",
                                             ("diagnostic", "run"): "true"})
    assert settings.stage1.company_as_feature is False
    assert settings.diagnostic.run is True
    settings = load(tmp_path, repo, changes={("stage1", "company_as_feature"): "true",
                                             ("diagnostic", "run"): "false"})
    assert settings.stage1.company_as_feature is True
    assert settings.diagnostic.run is False


def test_diagnose_class_weights_is_gone_from_stage1(tmp_path, repo):
    # Removed in U10.4a: an old file that still has it stops with an unknown setting.
    message = exact_message(tmp_path, repo, {("stage1", "diagnose_class_weights"): "true"})
    assert message == (
        f"{STAGE1}: unknown ['diagnose_class_weights']; expected exactly "
        f"['baseline_c', 'company_as_feature', 'feature_blocks', 'learning_rate', 'max_iter', "
        f"'max_leaf_nodes', 'min_samples_leaf', 'random_seed']")


def test_stage1_settings_has_no_diagnose_class_weights_field():
    assert "diagnose_class_weights" not in [f.name for f in dataclasses.fields(Stage1Settings)]


# --- [stage1] baseline_c (positive_number_setting) -------------------------------------------

@pytest.mark.parametrize("good, expected", [
    ("1.0", 1.0), ("1", 1.0), ("0.01", 0.01), ("1e-9", 1e-9), ("5e-324", 5e-324),
    ("1000", 1000.0), ("1e300", 1e300), ("1_000", 1000.0),
], ids=["one-float", "one-int", "small", "tiny", "smallest-subnormal", "large-int", "huge",
        "underscored-int"])
def test_baseline_c_accepts_numbers_above_0(tmp_path, repo, good, expected):
    value = load(tmp_path, repo, changes={("stage1", "baseline_c"): good}).stage1.baseline_c
    assert value == expected
    assert type(value) is float


@pytest.mark.parametrize("bad, shown", [
    ("0", "0"), ("0.0", "0.0"), ("-0.0", "-0.0"), ("-1", "-1"), ("-1e-9", "-1e-09"),
    ("inf", "inf"), ("+inf", "inf"), ("-inf", "-inf"), ("nan", "nan"), ('"1.0"', "'1.0'"),
    ("true", "True"), ("false", "False"), ("[1.0]", "[1.0]"),
    ("2025-06-02", "datetime.date(2025, 6, 2)"), ("{ c = 1 }", "{'c': 1}"),
], ids=["zero", "zero-float", "minus-zero", "negative", "tiny-negative", "inf", "plus-inf",
        "minus-inf", "nan", "string", "true", "false", "list", "date", "table"])
def test_baseline_c_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage1", "baseline_c"): bad})
    assert message == f"{STAGE1}: baseline_c is {shown}, not a number above 0"


def test_baseline_c_has_no_upper_bound_unlike_learning_rate(tmp_path, repo):
    # C is an inverse regularisation strength: above 1 is ordinary, not an error.
    stage1 = load(tmp_path, repo, changes={("stage1", "baseline_c"): "50"}).stage1
    assert stage1.baseline_c == 50.0


def test_baseline_c_is_read_separately_from_learning_rate(tmp_path, repo):
    stage1 = load(tmp_path, repo, changes={("stage1", "baseline_c"): "3.5",
                                           ("stage1", "learning_rate"): "0.2"}).stage1
    assert (stage1.baseline_c, stage1.learning_rate) == (3.5, 0.2)


def test_a_missing_baseline_c_has_the_exact_message(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("stage1", "baseline_c"): REMOVE})
    assert message == (
        f"{STAGE1}: missing ['baseline_c']; expected exactly "
        f"['baseline_c', 'company_as_feature', 'feature_blocks', "
        f"'learning_rate', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf', 'random_seed']")


def test_a_baseline_c_typo_names_the_missing_and_the_unknown_setting(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("stage1", "baseline_c"): REMOVE,
                                             ("stage1", "baseline_C"): "1.0"})
    assert message.startswith(f"{STAGE1}: missing ['baseline_c']; unknown ['baseline_C']; ")


def test_positive_number_setting_direct_calls():
    assert positive_number_setting({"c": 1.0}, "c", "here") == 1.0
    assert positive_number_setting({"c": 0.25}, "c", "here") == 0.25
    assert positive_number_setting({"c": 5e-324}, "c", "here") == 5e-324
    assert positive_number_setting({"c": 1e308}, "c", "here") == 1e308
    assert positive_number_setting({"c": 2**80}, "c", "here") == float(2**80)


@pytest.mark.parametrize("value", [1, 7, 2**40])
def test_positive_number_setting_returns_an_int_as_a_float(value):
    result = positive_number_setting({"c": value}, "c", "here")
    assert type(result) is float
    assert result == float(value)


def test_positive_number_setting_returns_a_float_as_a_float():
    assert type(positive_number_setting({"c": 0.5}, "c", "here")) is float


@pytest.mark.parametrize("bad", [0, 0.0, -0.0, -1, -1.5, -5e-324, float("nan"), float("inf"),
                                 float("-inf"), True, False, None, "1.0", "1", [1.0], (1.0,),
                                 {"c": 1}])
def test_positive_number_setting_rejects_naming_where_and_key(bad):
    with pytest.raises(ValueError) as error:
        positive_number_setting({"c": bad}, "c", "here")
    assert str(error.value) == f"here: c is {bad!r}, not a number above 0"


def test_positive_number_setting_true_is_rejected_although_it_equals_1():
    # True == 1 and True > 0 in Python: only the explicit bool check stops it.
    with pytest.raises(ValueError, match="here: c is True"):
        positive_number_setting({"c": True}, "c", "here")


def test_positive_number_setting_with_the_key_absent_raises_key_error():
    with pytest.raises(KeyError):
        positive_number_setting({}, "c", "here")


def test_positive_number_setting_does_not_modify_the_table():
    table = {"c": 2, "other": "x"}
    positive_number_setting(table, "c", "here")
    assert table == {"c": 2, "other": "x"}


# --- [debug] demo_origin (week_setting)-----------------------------------------------------

def test_demo_origin_is_read_as_an_int(tmp_path, repo):
    value = load(tmp_path, repo).debug.demo_origin
    assert value == 40
    assert type(value) is int


@pytest.mark.parametrize("good, expected", [("0", 0), ("1", 1), ("59", 59), ("1_000", 1000)])
def test_demo_origin_accepts_a_whole_number_of_0_or_more(tmp_path, repo, good, expected):
    value = load(tmp_path, repo, changes={("debug", "demo_origin"): good}).debug.demo_origin
    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize("bad, shown", [
    ("-1", "-1"), ("40.0", "40.0"), ("40.5", "40.5"), ('"40"', "'40'"), ("true", "True"),
    ("[40]", "[40]"), ("2025-06-02", "datetime.date(2025, 6, 2)"),
], ids=["negative", "whole-float", "fraction", "string", "true", "list", "date"])
def test_demo_origin_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("debug", "demo_origin"): bad})
    assert message == f"{DEBUG}: demo_origin is {shown}, not a week number (0 or more)"


def test_demo_origin_does_not_borrow_sample_week(tmp_path, repo):
    debug = load(tmp_path, repo, changes={("debug", "demo_origin"): "12",
                                          ("debug", "sample_week"): "30"}).debug
    assert (debug.demo_origin, debug.sample_week) == (12, 30)


def test_a_demo_origin_typo_has_the_exact_message(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("debug", "demo_origin"): REMOVE,
                                             ("debug", "demo_orign"): "40"})
    assert message == (f"{DEBUG}: missing ['demo_origin']; unknown ['demo_orign']; expected "
                       f"exactly ['demo_origin', 'one_day', 'sample_week']")


def test_debug_settings_fields():
    assert [field.name for field in dataclasses.fields(DebugSettings)] == [
        "one_day", "sample_week", "demo_origin"]


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_stage1_and_demo_origin_settings():
    settings = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT)
    assert settings.stage1 == Stage1Settings(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
        random_seed=1, feature_blocks=("history", "calendar", "receivables"),
        company_as_feature=True, baseline_c=1.0)
    assert settings.debug.demo_origin == 40
    assert type(settings.debug.demo_origin) is int


# --- [diagnostic] (U10.4a) ---------------------------------------------------------------------
# The stage 1 diagnostic's settings. Every value is checked through the file (the message names
# the file and the section, exactly); grid_setting and class_weight_setting also by direct calls.

DIAG = f"{FILE_NAME} [diagnostic]"
DIAGNOSTIC_KEYS = ['calibration_groups', 'class_weights', 'hgb_learning_rate', 'hgb_max_iter',
                   'hgb_max_leaf_nodes', 'importance_repeats', 'lr_c', 'origin',
                   'output_folder', 'reference_prior_weeks', 'reference_window', 'run',
                   'selection_metric']


def test_the_diagnostic_section_is_read(tmp_path, repo):
    diagnostic = load(tmp_path, repo).diagnostic
    assert isinstance(diagnostic, DiagnosticSettings)
    assert diagnostic == DiagnosticSettings(
        run=True, origin=40, hgb_max_iter=(100, 300, 600), hgb_learning_rate=(0.05, 0.1),
        hgb_max_leaf_nodes=(15, 31, 63), lr_c=(0.01, 0.1, 1.0, 10.0),
        class_weights=("none", "balanced"), selection_metric="log loss", calibration_groups=4,
        reference_window=13, reference_prior_weeks=1.0, importance_repeats=3,
        output_folder=(repo / "results" / "diagnostic").resolve())


def test_the_diagnostic_values_have_the_right_python_types(tmp_path, repo):
    diagnostic = load(tmp_path, repo).diagnostic
    assert diagnostic.run is True
    assert type(diagnostic.origin) is int
    assert type(diagnostic.importance_repeats) is int
    assert type(diagnostic.selection_metric) is str
    for name, item_type in (("hgb_max_iter", int), ("hgb_learning_rate", float),
                            ("hgb_max_leaf_nodes", int), ("lr_c", float),
                            ("class_weights", str)):
        grid = getattr(diagnostic, name)
        assert type(grid) is tuple, name
        assert all(type(item) is item_type for item in grid), (name, grid)
    assert isinstance(diagnostic.output_folder, Path)
    assert diagnostic.output_folder.is_absolute()


def test_diagnostic_settings_fields_are_in_the_documented_order():
    assert [field.name for field in dataclasses.fields(DiagnosticSettings)] == [
        "run", "origin", "hgb_max_iter", "hgb_learning_rate", "hgb_max_leaf_nodes", "lr_c",
        "class_weights", "selection_metric", "calibration_groups", "reference_window",
        "reference_prior_weeks", "importance_repeats", "output_folder"]


def test_settings_has_a_diagnostic_field_between_stage1_and_debug():
    names = [field.name for field in dataclasses.fields(Settings)]
    assert (names.index("history") < names.index("stage1") < names.index("diagnostic")
            < names.index("debug"))


def test_a_missing_diagnostic_section_raises_the_exact_message(tmp_path, repo):
    with pytest.raises(ValueError) as error:
        load(tmp_path, repo, drop_section="diagnostic")
    assert str(error.value) == (
        f"{FILE_NAME}: missing ['diagnostic']; expected exactly ['collections_layout', "
        f"'companies', 'data', 'debug', 'diagnostic', 'history', 'ledger_layout', "
        f"'receivables', 'stage1', 'stage2']")


def test_a_diagnostic_typo_has_the_exact_message(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("diagnostic", "lr_c"): REMOVE,
                                             ("diagnostic", "lr_C"): "[1.0]"})
    assert message == (f"{DIAG}: missing ['lr_c']; unknown ['lr_C']; expected exactly "
                       f"{DIAGNOSTIC_KEYS}")


def test_an_empty_diagnostic_section_names_every_setting(tmp_path, repo):
    changes = {("diagnostic", key): REMOVE for key in VALID["diagnostic"]}
    message = exact_message(tmp_path, repo, changes)
    assert message == (f"{DIAG}: missing {DIAGNOSTIC_KEYS}; expected exactly "
                       f"{DIAGNOSTIC_KEYS}")


def test_diagnose_class_weights_in_the_diagnostic_section_is_unknown(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("diagnostic", "diagnose_class_weights"): "true"})
    assert message.startswith(f"{DIAG}: unknown ['diagnose_class_weights']; ")


# run (bool_setting)

@pytest.mark.parametrize("text, expected", [("true", True), ("false", False)])
def test_diagnostic_run_accepts_true_and_false(tmp_path, repo, text, expected):
    assert load(tmp_path, repo, changes={("diagnostic", "run"): text}).diagnostic.run is expected


@pytest.mark.parametrize("bad, shown", [("1", "1"), ("0", "0"), ('"true"', "'true'"),
                                        ("[true]", "[True]")])
def test_diagnostic_run_rejects_non_bools_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("diagnostic", "run"): bad})
    assert message == f"{DIAG}: run is {shown}, not true or false"


def test_the_other_diagnostic_settings_are_still_checked_when_run_is_false(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("diagnostic", "run"): "false",
                                             ("diagnostic", "lr_c"): "[]"})
    assert message == f"{DIAG}: lr_c is [], not a non-empty list of values to try"


# origin (week_setting)

@pytest.mark.parametrize("good, expected", [("0", 0), ("1", 1), ("49", 49)])
def test_diagnostic_origin_accepts_week_numbers(tmp_path, repo, good, expected):
    value = load(tmp_path, repo, changes={("diagnostic", "origin"): good}).diagnostic.origin
    assert value == expected and type(value) is int


@pytest.mark.parametrize("bad, shown", [("-1", "-1"), ("40.0", "40.0"), ('"40"', "'40'"),
                                        ("true", "True"), ("[40]", "[40]")])
def test_diagnostic_origin_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("diagnostic", "origin"): bad})
    assert message == f"{DIAG}: origin is {shown}, not a week number (0 or more)"


def test_diagnostic_origin_does_not_borrow_demo_origin(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("diagnostic", "origin"): "33",
                                             ("debug", "demo_origin"): "12"})
    assert (settings.diagnostic.origin, settings.debug.demo_origin) == (33, 12)


# importance_repeats (count_setting)

@pytest.mark.parametrize("bad, shown", [("0", "0"), ("-2", "-2"), ("3.0", "3.0"),
                                        ('"3"', "'3'"), ("false", "False"), ("[3]", "[3]")])
def test_importance_repeats_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("diagnostic", "importance_repeats"): bad})
    assert message == f"{DIAG}: importance_repeats is {shown}, not a whole number of 1 or more"


def test_importance_repeats_accepts_1(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("diagnostic", "importance_repeats"): "1"})
    value = settings.diagnostic.importance_repeats
    assert value == 1 and type(value) is int


# output_folder (path_setting)

def test_output_folder_is_taken_from_the_repo_root(tmp_path, repo, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    folder = load(tmp_path, repo).diagnostic.output_folder
    assert folder == (repo / "results" / "diagnostic").resolve()


def test_an_absolute_output_folder_is_kept(tmp_path, repo):
    absolute = (tmp_path / "out" / "diag").resolve()
    settings = load(tmp_path, repo, changes={("diagnostic", "output_folder"): f"'{absolute}'"})
    assert settings.diagnostic.output_folder == absolute


def test_output_folder_need_not_exist_and_is_not_created(tmp_path, repo):
    folder = load(tmp_path, repo).diagnostic.output_folder
    assert not folder.exists()


@pytest.mark.parametrize("bad, shown", [("5", "5"), ('""', "''"), ('"  "', "'  '"),
                                        ("true", "True"), ('["a"]', "['a']")])
def test_output_folder_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("diagnostic", "output_folder"): bad})
    assert message == f"{DIAG}: output_folder is {shown}, not a path"


# the grids, through the file

GRIDS = {   # key -> (good TOML, expected tuple)
    "hgb_max_iter": ("[600, 100]", (600, 100)),
    "hgb_learning_rate": ("[0.1, 1, 0.05]", (0.1, 1.0, 0.05)),
    "hgb_max_leaf_nodes": ("[2, 63, 15]", (2, 63, 15)),
    "lr_c": ("[10, 0.01, 1e3]", (10.0, 0.01, 1000.0)),
    "class_weights": ('["balanced", "none"]', ("balanced", "none")),
}


@pytest.mark.parametrize("key", list(GRIDS))
def test_a_grid_keeps_the_given_order_and_converts_each_item(tmp_path, repo, key):
    text, expected = GRIDS[key]
    value = getattr(load(tmp_path, repo, changes={("diagnostic", key): text}).diagnostic, key)
    assert value == expected
    assert [type(item) for item in value] == [type(item) for item in expected]
    assert type(value) is tuple


@pytest.mark.parametrize("key, single", [("hgb_max_iter", "300"), ("hgb_learning_rate", "0.1"),
                                         ("hgb_max_leaf_nodes", "31"), ("lr_c", "1.0"),
                                         ("class_weights", '"none"')])
def test_a_grid_of_one_item_is_fine(tmp_path, repo, key, single):
    settings = load(tmp_path, repo, changes={("diagnostic", key): f"[{single}]"})
    assert len(getattr(settings.diagnostic, key)) == 1


@pytest.mark.parametrize("key", list(GRIDS))
@pytest.mark.parametrize("bad, shown", [("[]", "[]"), ("5", "5"), ('"none"', "'none'"),
                                        ("true", "True"), ("{ a = 1 }", "{'a': 1}")],
                         ids=["empty", "number", "text", "bool", "table"])
def test_a_grid_that_is_not_a_non_empty_list_has_the_exact_message(tmp_path, repo, key, bad,
                                                                   shown):
    message = exact_message(tmp_path, repo, {("diagnostic", key): bad})
    assert message == f"{DIAG}: {key} is {shown}, not a non-empty list of values to try"


@pytest.mark.parametrize("key, bad, item, reason", [
    ("hgb_max_iter", "[100, 0]", "0", "not a whole number of 1 or more"),
    ("hgb_max_iter", "[100, 3.0]", "3.0", "not a whole number of 1 or more"),
    ("hgb_max_iter", "[true]", "True", "not a whole number of 1 or more"),
    ("hgb_max_iter", "[[100]]", "[100]", "not a whole number of 1 or more"),
    ("hgb_learning_rate", "[0.05, 1.5]", "1.5", "not a number above 0 and at most 1"),
    ("hgb_learning_rate", "[0]", "0", "not a number above 0 and at most 1"),
    ("hgb_learning_rate", '["0.1"]', "'0.1'", "not a number above 0 and at most 1"),
    ("hgb_max_leaf_nodes", "[15, 1]", "1", "not a whole number of 2 or more"),
    ("hgb_max_leaf_nodes", "[31.0]", "31.0", "not a whole number of 2 or more"),
    ("lr_c", "[1.0, 0]", "0", "not a number above 0"),
    ("lr_c", "[-1.0]", "-1.0", "not a number above 0"),
    ("lr_c", "[inf]", "inf", "not a number above 0"),
    ("lr_c", "[nan]", "nan", "not a number above 0"),
    ("class_weights", '["none", "Balanced"]', "'Balanced'", "not one of ['none', 'balanced']"),
    ("class_weights", '["None"]', "'None'", "not one of ['none', 'balanced']"),
    ("class_weights", '[""]', "''", "not one of ['none', 'balanced']"),
    ("class_weights", "[1]", "1", "not one of ['none', 'balanced']"),
    ("class_weights", "[true]", "True", "not one of ['none', 'balanced']"),
])
def test_a_bad_grid_item_has_the_exact_message(tmp_path, repo, key, bad, item, reason):
    message = exact_message(tmp_path, repo, {("diagnostic", key): bad})
    assert message == f"{DIAG} {key} item: {key} is {item}, {reason}"


@pytest.mark.parametrize("key, bad, shown", [
    ("hgb_max_iter", "[100, 300, 100]", "[100, 300, 100]"),
    ("hgb_learning_rate", "[0.1, 0.1]", "[0.1, 0.1]"),
    ("hgb_learning_rate", "[1, 1.0]", "[1, 1.0]"),
    ("hgb_max_leaf_nodes", "[31, 31]", "[31, 31]"),
    ("lr_c", "[1.0, 10, 1]", "[1.0, 10, 1]"),
    ("class_weights", '["none", "balanced", "none"]', "['none', 'balanced', 'none']"),
])
def test_a_grid_with_a_repeat_has_the_exact_message(tmp_path, repo, key, bad, shown):
    message = exact_message(tmp_path, repo, {("diagnostic", key): bad})
    assert message == f"{DIAG}: {key} lists a value more than once: {shown}"


def test_a_bad_item_is_reported_before_a_repeat(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("diagnostic", "hgb_max_iter"): "[0, 0]"})
    assert message == (f"{DIAG} hgb_max_iter item: hgb_max_iter is 0, not a whole number of "
                       f"1 or more")


def test_the_first_bad_item_is_the_one_reported(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("diagnostic", "lr_c"): "[1.0, -2, 0]"})
    assert message == f"{DIAG} lr_c item: lr_c is -2, not a number above 0"


def test_the_grids_are_read_separately(tmp_path, repo):
    diagnostic = load(tmp_path, repo, changes={("diagnostic", "hgb_max_iter"): "[7]",
                                               ("diagnostic", "hgb_max_leaf_nodes"): "[9]",
                                               ("diagnostic", "lr_c"): "[0.5]",
                                               ("diagnostic", "hgb_learning_rate"): "[0.25]"}
                      ).diagnostic
    assert (diagnostic.hgb_max_iter, diagnostic.hgb_max_leaf_nodes, diagnostic.lr_c,
            diagnostic.hgb_learning_rate) == ((7,), (9,), (0.5,), (0.25,))


# grid_setting and class_weight_setting, by direct calls

def test_class_weights_constant():
    assert CLASS_WEIGHTS == ("none", "balanced")


@pytest.mark.parametrize("good", ["none", "balanced"])
def test_class_weight_setting_accepts_none_and_balanced(good):
    value = class_weight_setting({"w": good}, "w", "here")
    assert value == good and type(value) is str


@pytest.mark.parametrize("bad", ["None", "BALANCED", " none", "", None, True, 1, 0.5,
                                 ["none"], ("none",), {"none": 1}])
def test_class_weight_setting_rejects_with_the_exact_message(bad):
    with pytest.raises(ValueError) as error:
        class_weight_setting({"w": bad}, "w", "here")
    assert str(error.value) == f"here: w is {bad!r}, not one of ['none', 'balanced']"


def test_grid_setting_returns_a_tuple_in_the_given_order():
    assert grid_setting({"g": [3, 1, 2]}, "g", "here", count_setting) == (3, 1, 2)


def test_grid_setting_converts_with_the_check():
    value = grid_setting({"g": [1, 2]}, "g", "here", positive_number_setting)
    assert value == (1.0, 2.0)
    assert [type(item) for item in value] == [float, float]


def test_grid_setting_uses_the_given_check():
    # 1 is a count but not a leaf count.
    assert grid_setting({"g": [1]}, "g", "here", count_setting) == (1,)
    with pytest.raises(ValueError) as error:
        grid_setting({"g": [1]}, "g", "here", leaf_count_setting)
    assert str(error.value) == "here g item: g is 1, not a whole number of 2 or more"


@pytest.mark.parametrize("bad", [[], 5, "abc", (1, 2), {"a": 1}, None, True])
def test_grid_setting_rejects_a_non_list_or_empty_list_with_the_exact_message(bad):
    with pytest.raises(ValueError) as error:
        grid_setting({"g": bad}, "g", "here", count_setting)
    assert str(error.value) == f"here: g is {bad!r}, not a non-empty list of values to try"


def test_grid_setting_item_message_names_where_key_and_item():
    with pytest.raises(ValueError) as error:
        grid_setting({"g": [5, 0]}, "g", "here", count_setting)
    assert str(error.value) == "here g item: g is 0, not a whole number of 1 or more"


def test_grid_setting_repeat_message_shows_the_list_as_written():
    with pytest.raises(ValueError) as error:
        grid_setting({"g": [1, 2.0, 2]}, "g", "here", positive_number_setting)
    assert str(error.value) == "here: g lists a value more than once: [1, 2.0, 2]"


def test_grid_setting_with_class_weights():
    assert grid_setting({"g": ["balanced"]}, "g", "here", class_weight_setting) == ("balanced",)
    with pytest.raises(ValueError) as error:
        grid_setting({"g": ["none", "none"]}, "g", "here", class_weight_setting)
    assert str(error.value) == "here: g lists a value more than once: ['none', 'none']"


def test_grid_setting_does_not_change_the_table():
    table = {"g": [3, 1], "other": "x"}
    grid_setting(table, "g", "here", count_setting)
    assert table == {"g": [3, 1], "other": "x"}


def test_grid_setting_with_the_key_absent_raises_key_error():
    with pytest.raises(KeyError):
        grid_setting({}, "g", "here", count_setting)


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_diagnostic_settings():
    settings = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT)
    assert settings.diagnostic == DiagnosticSettings(
        run=True, origin=40, hgb_max_iter=(100, 300, 600, 900),
        hgb_learning_rate=(0.01, 0.05, 0.1, 0.15, 0.2, 0.5), hgb_max_leaf_nodes=(15, 31, 63),
        lr_c=(0.01, 0.1, 1.0, 10.0), class_weights=("none",),
        selection_metric="log loss", calibration_groups=10, reference_window=13,
        reference_prior_weeks=1.0, importance_repeats=3,
        output_folder=(REPO_ROOT / "data" / "interim" / "hurdle_diagnostic").resolve())


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_file_has_no_diagnose_class_weights():
    raw = tomllib.loads(SYNTHETIC_SETTINGS.read_text(encoding="utf-8"))
    assert "diagnose_class_weights" not in raw["stage1"]
    assert "diagnostic" in raw


# --- [diagnostic] selection_metric (U10.4b') ---------------------------------------------------
# The validation score that chooses each model's setting in the stage 1 diagnostic: exactly one
# of "log loss", "brier", "avg precision" (case and spacing matter).

from collection_estimation.hurdle.settings import (  # noqa: E402
    SELECTION_METRICS,
    selection_metric_setting,
)

METRIC_CHOICES = "['log loss', 'brier', 'avg precision']"


def test_selection_metrics_constant():
    assert SELECTION_METRICS == ("log loss", "brier", "avg precision")
    assert type(SELECTION_METRICS) is tuple


@pytest.mark.parametrize("good", ["log loss", "brier", "avg precision"])
def test_selection_metric_setting_accepts_each_metric(good):
    value = selection_metric_setting({"m": good}, "m", "here")
    assert value == good and type(value) is str


@pytest.mark.parametrize("bad", ["Log loss", "LOG LOSS", "log_loss", "logloss", "log-loss",
                                 " log loss", "log loss ", "Brier", "brier score",
                                 "average precision", "avg_precision", "AUC", "F1", "", None,
                                 True, 1, 0.5, ["log loss"], ("log loss",), {"log loss": 1}])
def test_selection_metric_setting_rejects_with_the_exact_message(bad):
    with pytest.raises(ValueError) as error:
        selection_metric_setting({"m": bad}, "m", "here")
    assert str(error.value) == f"here: m is {bad!r}, not one of {METRIC_CHOICES}"


def test_selection_metric_setting_with_the_key_absent_raises_key_error():
    with pytest.raises(KeyError):
        selection_metric_setting({}, "m", "here")


def test_selection_metric_setting_does_not_change_the_table():
    table = {"m": "brier", "other": 1}
    selection_metric_setting(table, "m", "here")
    assert table == {"m": "brier", "other": 1}


@pytest.mark.parametrize("good", ["log loss", "brier", "avg precision"])
def test_the_selection_metric_is_read_from_the_file(tmp_path, repo, good):
    settings = load(tmp_path, repo, changes={("diagnostic", "selection_metric"): f'"{good}"'})
    assert settings.diagnostic.selection_metric == good
    assert type(settings.diagnostic.selection_metric) is str


def test_the_selection_metric_changes_nothing_else(tmp_path, repo):
    base = load(tmp_path, repo).diagnostic
    other = load(tmp_path, repo,
                 changes={("diagnostic", "selection_metric"): '"avg precision"'}).diagnostic
    assert other == dataclasses.replace(base, selection_metric="avg precision")


@pytest.mark.parametrize("bad, shown", [('"Log loss"', "'Log loss'"), ('"AUC"', "'AUC'"),
                                        ('"average precision"', "'average precision'"),
                                        ('""', "''"), ("1", "1"), ("true", "True"),
                                        ('["log loss"]', "['log loss']"),
                                        ('{ a = "brier" }', "{'a': 'brier'}")])
def test_a_bad_selection_metric_in_the_file_has_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("diagnostic", "selection_metric"): bad})
    assert message == f"{DIAG}: selection_metric is {shown}, not one of {METRIC_CHOICES}"


def test_a_missing_selection_metric_has_the_exact_message(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("diagnostic", "selection_metric"): REMOVE})
    assert message == (f"{DIAG}: missing ['selection_metric']; expected exactly "
                       f"{DIAGNOSTIC_KEYS}")


def test_a_selection_metric_typo_is_both_missing_and_unknown(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("diagnostic", "selection_metric"): REMOVE,
                                             ("diagnostic", "selection_metrics"): '"brier"'})
    assert message == (f"{DIAG}: missing ['selection_metric']; unknown "
                       f"['selection_metrics']; expected exactly {DIAGNOSTIC_KEYS}")


def test_selection_metric_in_the_stage1_section_is_unknown(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("stage1", "selection_metric"): '"brier"'})
    assert message.startswith(f"{FILE_NAME} [stage1]: unknown ['selection_metric']; ")


def test_the_selection_metric_is_checked_when_run_is_false(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("diagnostic", "run"): "false",
                                             ("diagnostic", "selection_metric"): '"AUC"'})
    assert message == f"{DIAG}: selection_metric is 'AUC', not one of {METRIC_CHOICES}"


def test_diagnostic_settings_selection_metric_sits_between_class_weights_and_calibration():
    # U10.4d put the three calibration settings between selection_metric and the repeats.
    names = [field.name for field in dataclasses.fields(DiagnosticSettings)]
    assert (names.index("class_weights") + 1 == names.index("selection_metric")
            == names.index("calibration_groups") - 1 == names.index("importance_repeats") - 4)


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_file_chooses_by_log_loss():
    raw = tomllib.loads(SYNTHETIC_SETTINGS.read_text(encoding="utf-8"))
    assert raw["diagnostic"]["selection_metric"] == "log loss"
    assert load_settings(SYNTHETIC_SETTINGS, REPO_ROOT).diagnostic.selection_metric == "log loss"


# --- [diagnostic] calibration settings (U10.4d) ------------------------------------------------
# calibration_groups (group_count_setting: a whole number, 2 or more), reference_window
# (reference_window_setting: a count that is one of the [history] windows) and
# reference_prior_weeks (positive_number_setting). All three are required and strict.

from collection_estimation.hurdle.settings import (  # noqa: E402
    group_count_setting,
    reference_window_setting,
)


# group_count_setting, by direct calls

@pytest.mark.parametrize("good", [2, 3, 10, 1000, 2**40])
def test_group_count_setting_accepts_whole_numbers_of_2_or_more(good):
    value = group_count_setting({"g": good}, "g", "here")
    assert value == good and type(value) is int


@pytest.mark.parametrize("bad", [1, 0, -1, -10, True, False, 2.0, 10.0, 2.5, "10", None,
                                 [10], (10,), {"g": 10}, float("nan"), float("inf")])
def test_group_count_setting_rejects_with_the_exact_message(bad):
    with pytest.raises(ValueError) as error:
        group_count_setting({"g": bad}, "g", "here")
    assert str(error.value) == f"here: g is {bad!r}, not a whole number of 2 or more"


def test_group_count_setting_with_the_key_absent_raises_key_error():
    with pytest.raises(KeyError):
        group_count_setting({}, "g", "here")


def test_group_count_setting_does_not_change_the_table():
    table = {"g": 5, "other": "x"}
    group_count_setting(table, "g", "here")
    assert table == {"g": 5, "other": "x"}


# reference_window_setting, by direct calls

@pytest.mark.parametrize("good", [4, 13, 52])
def test_reference_window_setting_accepts_each_history_window(good):
    value = reference_window_setting({"w": good}, "w", "here", (4, 13, 52))
    assert value == good and type(value) is int


@pytest.mark.parametrize("bad", [1, 5, 12, 14, 51, 53, 100])
def test_reference_window_setting_rejects_a_count_not_among_the_windows(bad):
    with pytest.raises(ValueError) as error:
        reference_window_setting({"w": bad}, "w", "here", (4, 13, 52))
    assert str(error.value) == f"here: w is {bad}, not one of the [history] windows [4, 13, 52]"


@pytest.mark.parametrize("bad", [0, -4, True, 13.0, "13", None, [13]])
def test_reference_window_setting_rejects_a_non_count_as_count_setting_does(bad):
    # Not a count at all: count_setting's message, before the windows are looked at.
    with pytest.raises(ValueError) as error:
        reference_window_setting({"w": bad}, "w", "here", (1, 4, 13))
    assert str(error.value) == f"here: w is {bad!r}, not a whole number of 1 or more"


def test_reference_window_setting_true_is_not_window_1():
    # True == 1 in Python; it must still be refused even when 1 is a window.
    with pytest.raises(ValueError):
        reference_window_setting({"w": True}, "w", "here", (1, 4))


def test_reference_window_setting_with_the_key_absent_raises_key_error():
    with pytest.raises(KeyError):
        reference_window_setting({}, "w", "here", (4, 13))


# Through the file

def test_the_calibration_settings_are_read(tmp_path, repo):
    diagnostic = load(tmp_path, repo).diagnostic
    assert diagnostic.calibration_groups == 4 and type(diagnostic.calibration_groups) is int
    assert diagnostic.reference_window == 13 and type(diagnostic.reference_window) is int
    assert (diagnostic.reference_prior_weeks == 1.0
            and type(diagnostic.reference_prior_weeks) is float)


def test_the_calibration_settings_change_nothing_else(tmp_path, repo):
    base = load(tmp_path, repo).diagnostic
    other = load(tmp_path, repo, changes={("diagnostic", "calibration_groups"): "7",
                                          ("diagnostic", "reference_window"): "52",
                                          ("diagnostic", "reference_prior_weeks"): "2.5"})
    assert other.diagnostic == dataclasses.replace(base, calibration_groups=7,
                                                   reference_window=52,
                                                   reference_prior_weeks=2.5)


@pytest.mark.parametrize("good, expected", [("2", 2), ("3", 3), ("20", 20), ("1_000", 1000)])
def test_calibration_groups_accepts_whole_numbers_of_2_or_more(tmp_path, repo, good, expected):
    value = load(tmp_path, repo,
                 changes={("diagnostic", "calibration_groups"): good}).diagnostic
    assert value.calibration_groups == expected and type(value.calibration_groups) is int


@pytest.mark.parametrize("bad, shown", [("1", "1"), ("0", "0"), ("-3", "-3"), ("10.0", "10.0"),
                                        ('"10"', "'10'"), ("true", "True"), ("[10]", "[10]"),
                                        ("nan", "nan"), ("inf", "inf")])
def test_calibration_groups_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("diagnostic", "calibration_groups"): bad})
    assert message == f"{DIAG}: calibration_groups is {shown}, not a whole number of 2 or more"


@pytest.mark.parametrize("good", ["4", "13", "52"])
def test_reference_window_accepts_each_history_window(tmp_path, repo, good):
    value = load(tmp_path, repo,
                 changes={("diagnostic", "reference_window"): good}).diagnostic.reference_window
    assert value == int(good) and type(value) is int


def test_reference_window_follows_the_history_windows(tmp_path, repo):
    settings = load(tmp_path, repo, changes={("history", "windows"): "[26, 8]",
                                             ("diagnostic", "reference_window"): "8"})
    assert settings.history.windows == (8, 26)
    assert settings.diagnostic.reference_window == 8


@pytest.mark.parametrize("bad", ["5", "12", "26", "104"])
def test_reference_window_not_a_history_window_has_the_exact_message(tmp_path, repo, bad):
    message = exact_message(tmp_path, repo, {("diagnostic", "reference_window"): bad})
    assert message == (f"{DIAG}: reference_window is {bad}, not one of the [history] windows "
                       f"[4, 13, 52]")


def test_reference_window_message_lists_the_windows_sorted(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("history", "windows"): "[52, 4]",
                                             ("diagnostic", "reference_window"): "13"})
    assert message == (f"{DIAG}: reference_window is 13, not one of the [history] windows "
                       f"[4, 52]")


@pytest.mark.parametrize("bad, shown", [("0", "0"), ("-13", "-13"), ("13.0", "13.0"),
                                        ('"13"', "'13'"), ("true", "True"), ("[13]", "[13]")])
def test_reference_window_that_is_not_a_count_has_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("diagnostic", "reference_window"): bad})
    assert message == f"{DIAG}: reference_window is {shown}, not a whole number of 1 or more"


def test_a_bad_history_window_is_reported_before_the_reference_window(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("history", "windows"): "[]",
                                             ("diagnostic", "reference_window"): "99"})
    assert message == (f"{FILE_NAME} [history]: windows is [], not a non-empty list of "
                       f"window lengths")


@pytest.mark.parametrize("good, expected", [("1", 1.0), ("1.0", 1.0), ("0.5", 0.5),
                                            ("3", 3.0), ("1e-9", 1e-9), ("100", 100.0)])
def test_reference_prior_weeks_accepts_numbers_above_0(tmp_path, repo, good, expected):
    value = load(tmp_path, repo, changes={("diagnostic", "reference_prior_weeks"): good})
    assert value.diagnostic.reference_prior_weeks == expected
    assert type(value.diagnostic.reference_prior_weeks) is float


@pytest.mark.parametrize("bad, shown", [("0", "0"), ("0.0", "0.0"), ("-1", "-1"),
                                        ("nan", "nan"), ("inf", "inf"), ('"1"', "'1'"),
                                        ("true", "True"), ("[1]", "[1]")])
def test_reference_prior_weeks_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("diagnostic", "reference_prior_weeks"): bad})
    assert message == f"{DIAG}: reference_prior_weeks is {shown}, not a number above 0"


@pytest.mark.parametrize("key", ["calibration_groups", "reference_window",
                                 "reference_prior_weeks"])
def test_a_missing_calibration_setting_has_the_exact_message(tmp_path, repo, key):
    message = exact_message(tmp_path, repo, {("diagnostic", key): REMOVE})
    assert message == f"{DIAG}: missing [{key!r}]; expected exactly {DIAGNOSTIC_KEYS}"


@pytest.mark.parametrize("key, value", [("calibration_groups", "4"), ("reference_window", "13"),
                                        ("reference_prior_weeks", "1.0")])
def test_a_calibration_setting_in_the_stage1_section_is_unknown(tmp_path, repo, key, value):
    message = exact_message(tmp_path, repo, {("stage1", key): value})
    assert message.startswith(f"{FILE_NAME} [stage1]: unknown [{key!r}]; ")


@pytest.mark.parametrize("key, bad", [("calibration_groups", "1"),
                                      ("reference_window", "5"),
                                      ("reference_prior_weeks", "0")])
def test_the_calibration_settings_are_checked_when_run_is_false(tmp_path, repo, key, bad):
    message = exact_message(tmp_path, repo, {("diagnostic", "run"): "false",
                                             ("diagnostic", key): bad})
    assert message.startswith(f"{DIAG}: {key} is ")


def test_the_calibration_settings_are_checked_in_field_order(tmp_path, repo):
    # All three broken: calibration_groups is reported, then (with it fixed) reference_window.
    changes = {("diagnostic", "calibration_groups"): "1",
               ("diagnostic", "reference_window"): "5",
               ("diagnostic", "reference_prior_weeks"): "0"}
    assert exact_message(tmp_path, repo, changes).startswith(f"{DIAG}: calibration_groups is ")
    changes[("diagnostic", "calibration_groups")] = "4"
    assert exact_message(tmp_path, repo, changes).startswith(f"{DIAG}: reference_window is ")
    changes[("diagnostic", "reference_window")] = "13"
    assert exact_message(tmp_path, repo, changes).startswith(
        f"{DIAG}: reference_prior_weeks is ")


def test_diagnostic_settings_calibration_fields_sit_between_selection_metric_and_repeats():
    names = [field.name for field in dataclasses.fields(DiagnosticSettings)]
    start = names.index("selection_metric")
    assert names[start:start + 5] == ["selection_metric", "calibration_groups",
                                      "reference_window", "reference_prior_weeks",
                                      "importance_repeats"]


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_file_calibration_settings():
    raw = tomllib.loads(SYNTHETIC_SETTINGS.read_text(encoding="utf-8"))
    assert raw["diagnostic"]["calibration_groups"] == 10
    assert raw["diagnostic"]["reference_window"] == 13
    assert raw["diagnostic"]["reference_prior_weeks"] == 1
    settings = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT)
    assert settings.diagnostic.calibration_groups == 10
    assert settings.diagnostic.reference_window == 13
    assert settings.diagnostic.reference_window in settings.history.windows
    assert settings.diagnostic.reference_prior_weeks == 1.0
    assert type(settings.diagnostic.reference_prior_weeks) is float


# --- [stage2] (U11.1) ------------------------------------------------------------------------
# Stage 2, the "how much, if it pays?" regressor. Its own loss (stage2_loss_setting), and the
# other six keys through the validators stage 1 already uses; every message names the file and
# [stage2], exactly. VALID's [stage2] differs from its [stage1] on purpose, so a test can tell
# which section a value was read from.

STAGE2 = f"{FILE_NAME} [stage2]"
STAGE2_KEYS = ['feature_blocks', 'learning_rate', 'loss', 'max_iter', 'max_leaf_nodes',
               'min_samples_leaf', 'random_seed']
LOSS_CHOICES = "['poisson', 'gamma', 'squared_error']"


def exact_message_from(tmp_path, repo, **kwargs):
    with pytest.raises(ValueError) as error:
        load(tmp_path, repo, **kwargs)
    return str(error.value)


def test_the_stage2_section_is_read(tmp_path, repo):
    stage2 = load(tmp_path, repo).stage2
    assert isinstance(stage2, Stage2Settings)
    assert stage2 == Stage2Settings(
        loss="gamma", max_iter=200, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=10,
        random_seed=7, feature_blocks=("history", "calendar"))


def test_the_stage2_values_have_the_right_python_types(tmp_path, repo):
    stage2 = load(tmp_path, repo).stage2
    assert type(stage2.loss) is str
    for name in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "random_seed"):
        assert type(getattr(stage2, name)) is int, name
    assert type(stage2.learning_rate) is float
    assert type(stage2.feature_blocks) is tuple


def test_stage2_is_read_from_its_own_section_not_from_stage1(tmp_path, repo):
    settings = load(tmp_path, repo)
    assert settings.stage1.max_iter == 300 and settings.stage2.max_iter == 200
    assert settings.stage1.learning_rate == 0.05 and settings.stage2.learning_rate == 0.1
    assert settings.stage1.max_leaf_nodes == 31 and settings.stage2.max_leaf_nodes == 15
    assert settings.stage1.min_samples_leaf == 20 and settings.stage2.min_samples_leaf == 10
    assert settings.stage1.random_seed == 1 and settings.stage2.random_seed == 7
    assert settings.stage1.feature_blocks == ("history", "calendar", "receivables")
    assert settings.stage2.feature_blocks == ("history", "calendar")


def test_changing_stage2_leaves_stage1_alone(tmp_path, repo):
    base = load(tmp_path, repo)
    other = load(tmp_path, repo, changes={("stage2", "max_iter"): "5",
                                          ("stage2", "loss"): '"poisson"'})
    assert other.stage1 == base.stage1
    assert other.stage2 == dataclasses.replace(base.stage2, max_iter=5, loss="poisson")


def test_changing_stage1_leaves_stage2_alone(tmp_path, repo):
    base = load(tmp_path, repo)
    other = load(tmp_path, repo, changes={("stage1", "max_iter"): "5"})
    assert other.stage2 == base.stage2


def test_stage2_settings_fields_are_in_the_documented_order():
    assert [field.name for field in dataclasses.fields(Stage2Settings)] == [
        "loss", "max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
        "random_seed", "feature_blocks"]


def test_stage2_settings_has_no_company_as_feature_or_baseline():
    # The company code goes in exactly when it does for stage 1; there is no stage 2 baseline.
    names = [field.name for field in dataclasses.fields(Stage2Settings)]
    assert "company_as_feature" not in names
    assert "baseline_c" not in names


def test_settings_has_a_stage2_field_between_stage1_and_diagnostic():
    names = [field.name for field in dataclasses.fields(Settings)]
    assert names.index("stage1") < names.index("stage2") < names.index("diagnostic")


def test_stage2_settings_is_frozen_and_hashable():
    stage2 = Stage2Settings(loss="poisson", max_iter=1, learning_rate=0.1, max_leaf_nodes=2,
                            min_samples_leaf=1, random_seed=0, feature_blocks=("history",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        stage2.loss = "gamma"
    assert hash(stage2) == hash(dataclasses.replace(stage2))


def test_a_missing_stage2_section_raises_the_exact_message(tmp_path, repo):
    message = exact_message_from(tmp_path, repo, drop_section="stage2")
    assert message == (
        f"{FILE_NAME}: missing ['stage2']; expected exactly ['collections_layout', "
        f"'companies', 'data', 'debug', 'diagnostic', 'history', 'ledger_layout', "
        f"'receivables', 'stage1', 'stage2']")


def test_a_stage2_section_given_as_a_plain_value_raises(tmp_path, repo):
    message = exact_message_from(tmp_path, repo, drop_section="stage2", top="stage2 = 5")
    assert message == f"{STAGE2} is 5, not a section"


def test_an_empty_stage2_section_has_the_exact_message(tmp_path, repo):
    changes = {("stage2", key): REMOVE for key in VALID["stage2"]}
    message = exact_message(tmp_path, repo, changes)
    assert message == f"{STAGE2}: missing {STAGE2_KEYS}; expected exactly {STAGE2_KEYS}"


@pytest.mark.parametrize("key", STAGE2_KEYS)
def test_a_missing_stage2_setting_has_the_exact_message(tmp_path, repo, key):
    message = exact_message(tmp_path, repo, {("stage2", key): REMOVE})
    assert message == f"{STAGE2}: missing [{key!r}]; expected exactly {STAGE2_KEYS}"


@pytest.mark.parametrize("key, value", [
    ("company_as_feature", "true"), ("baseline_c", "1.0"), ("early_stopping", "false"),
    ("selection_metric", '"log loss"'), ("Loss", '"poisson"'),
], ids=["company_as_feature", "baseline_c", "early_stopping", "selection_metric", "Loss"])
def test_an_unknown_stage2_setting_has_the_exact_message(tmp_path, repo, key, value):
    message = exact_message(tmp_path, repo, {("stage2", key): value})
    assert message == f"{STAGE2}: unknown [{key!r}]; expected exactly {STAGE2_KEYS}"


def test_a_stage2_typo_names_the_missing_and_the_unknown_setting(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("stage2", "loss"): REMOVE,
                                             ("stage2", "losses"): '"poisson"'})
    assert message == (f"{STAGE2}: missing ['loss']; unknown ['losses']; expected exactly "
                       f"{STAGE2_KEYS}")


def test_loss_in_the_stage1_section_is_unknown(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("stage1", "loss"): '"poisson"'})
    assert message.startswith(f"{FILE_NAME} [stage1]: unknown ['loss']; ")


# --- [stage2] loss (stage2_loss_setting) ------------------------------------------------------

def test_stage2_losses_constant():
    assert STAGE2_LOSSES == ("poisson", "gamma", "squared_error")
    assert type(STAGE2_LOSSES) is tuple


@pytest.mark.parametrize("good", ["poisson", "gamma", "squared_error"])
def test_each_stage2_loss_is_read_from_the_file(tmp_path, repo, good):
    value = load(tmp_path, repo, changes={("stage2", "loss"): f'"{good}"'}).stage2.loss
    assert value == good
    assert type(value) is str


@pytest.mark.parametrize("bad, shown", [
    ('"Poisson"', "'Poisson'"), ('"POISSON"', "'POISSON'"), ('" poisson"', "' poisson'"),
    ('"poisson "', "'poisson '"), ('"squared error"', "'squared error'"),
    ('"squared-error"', "'squared-error'"), ('"absolute_error"', "'absolute_error'"),
    ('"quantile"', "'quantile'"), ('"tweedie"', "'tweedie'"), ('"log_loss"', "'log_loss'"),
    ('""', "''"), ("1", "1"), ("0.5", "0.5"), ("true", "True"), ("false", "False"),
    ('["poisson"]', "['poisson']"), ('{ loss = "poisson" }', "{'loss': 'poisson'}"),
    ("2025-06-02", "datetime.date(2025, 6, 2)"),
], ids=["capital", "upper", "lead-space", "trail-space", "space", "dash", "absolute-error",
        "quantile", "tweedie", "log-loss", "empty", "int", "float", "true", "false", "list",
        "table", "date"])
def test_a_bad_stage2_loss_in_the_file_has_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage2", "loss"): bad})
    assert message == f"{STAGE2}: loss is {shown}, not one of {LOSS_CHOICES}"


@pytest.mark.parametrize("good", ["poisson", "gamma", "squared_error"])
def test_stage2_loss_setting_accepts_each_loss(good):
    value = stage2_loss_setting({"l": good}, "l", "here")
    assert value == good and type(value) is str


@pytest.mark.parametrize("bad", ["Poisson", "poisson ", "absolute_error", "quantile", "",
                                 None, True, False, 0, 1, 1.5, float("nan"), ["poisson"],
                                 ("poisson",), {"poisson": 1}],
                         ids=["capital", "space", "absolute", "quantile", "empty", "none",
                              "true", "false", "zero", "one", "float", "nan", "list", "tuple",
                              "dict"])
def test_stage2_loss_setting_rejects_with_the_exact_message(bad):
    with pytest.raises(ValueError) as error:
        stage2_loss_setting({"l": bad}, "l", "here")
    assert str(error.value) == f"here: l is {bad!r}, not one of {LOSS_CHOICES}"


def test_stage2_loss_setting_with_the_key_absent_raises_key_error():
    with pytest.raises(KeyError):
        stage2_loss_setting({}, "l", "here")


def test_stage2_loss_setting_does_not_change_the_table():
    table = {"l": "gamma", "other": 1}
    stage2_loss_setting(table, "l", "here")
    assert table == {"l": "gamma", "other": 1}


def test_a_bad_stage2_loss_is_reported_before_the_other_stage2_settings(tmp_path, repo):
    # loss is read first: with loss and max_iter both broken, loss is named.
    message = exact_message(tmp_path, repo, {("stage2", "loss"): '"tweedie"',
                                             ("stage2", "max_iter"): "0"})
    assert message.startswith(f"{STAGE2}: loss is 'tweedie'")


# --- [stage2] the reused validators -----------------------------------------------------------

@pytest.mark.parametrize("key", ["max_iter", "min_samples_leaf"])
@pytest.mark.parametrize("good, expected", [("1", 1), ("2", 2), ("1_000", 1000)])
def test_stage2_counts_accept_a_whole_number_of_1_or_more(tmp_path, repo, key, good, expected):
    value = getattr(load(tmp_path, repo, changes={("stage2", key): good}).stage2, key)
    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize("key", ["max_iter", "min_samples_leaf"])
@pytest.mark.parametrize("bad, shown", [
    ("0", "0"), ("-1", "-1"), ("1.5", "1.5"), ("300.0", "300.0"), ('"300"', "'300'"),
    ("true", "True"), ("[300]", "[300]"),
], ids=["zero", "negative", "fraction", "whole-float", "string", "true", "list"])
def test_stage2_counts_reject_with_the_exact_message(tmp_path, repo, key, bad, shown):
    message = exact_message(tmp_path, repo, {("stage2", key): bad})
    assert message == f"{STAGE2}: {key} is {shown}, not a whole number of 1 or more"


@pytest.mark.parametrize("good, expected", [("0.05", 0.05), ("1", 1.0), ("1e-9", 1e-9)])
def test_stage2_learning_rate_accepts_above_0_and_at_most_1(tmp_path, repo, good, expected):
    value = load(tmp_path, repo,
                 changes={("stage2", "learning_rate"): good}).stage2.learning_rate
    assert value == expected
    assert type(value) is float


@pytest.mark.parametrize("bad, shown", [
    ("0", "0"), ("-0.05", "-0.05"), ("1.0000001", "1.0000001"), ("nan", "nan"),
    ("inf", "inf"), ('"0.05"', "'0.05'"), ("true", "True"),
], ids=["zero", "negative", "above-one", "nan", "inf", "string", "true"])
def test_stage2_learning_rate_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage2", "learning_rate"): bad})
    assert message == f"{STAGE2}: learning_rate is {shown}, not a number above 0 and at most 1"


@pytest.mark.parametrize("good, expected", [("2", 2), ("63", 63)])
def test_stage2_max_leaf_nodes_accepts_2_or_more(tmp_path, repo, good, expected):
    value = load(tmp_path, repo,
                 changes={("stage2", "max_leaf_nodes"): good}).stage2.max_leaf_nodes
    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize("bad, shown", [("1", "1"), ("0", "0"), ("31.0", "31.0"),
                                        ('"31"', "'31'"), ("true", "True")],
                         ids=["one", "zero", "whole-float", "string", "true"])
def test_stage2_max_leaf_nodes_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage2", "max_leaf_nodes"): bad})
    assert message == f"{STAGE2}: max_leaf_nodes is {shown}, not a whole number of 2 or more"


@pytest.mark.parametrize("good, expected", [("0", 0), ("1", 1), ("4294967295", 4294967295)])
def test_stage2_random_seed_accepts_0_or_more(tmp_path, repo, good, expected):
    value = load(tmp_path, repo, changes={("stage2", "random_seed"): good}).stage2.random_seed
    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize("bad, shown", [("-1", "-1"), ("1.0", "1.0"), ('"1"', "'1'"),
                                        ("false", "False")],
                         ids=["negative", "whole-float", "string", "false"])
def test_stage2_random_seed_rejects_with_the_exact_message(tmp_path, repo, bad, shown):
    message = exact_message(tmp_path, repo, {("stage2", "random_seed"): bad})
    assert message == (f"{STAGE2}: random_seed is {shown}, not a random seed (a whole number, "
                       f"0 or more)")


@pytest.mark.parametrize("good, expected", [
    ('["receivables", "calendar", "history"]', ("history", "calendar", "receivables")),
    ('["receivables"]', ("receivables",)),
], ids=["reordered", "one"])
def test_stage2_feature_blocks_are_returned_in_feature_blocks_order(tmp_path, repo, good,
                                                                   expected):
    value = load(tmp_path, repo,
                 changes={("stage2", "feature_blocks"): good}).stage2.feature_blocks
    assert value == expected
    assert type(value) is tuple


def test_stage2_feature_blocks_empty_has_the_exact_message(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("stage2", "feature_blocks"): "[]"})
    assert message == (f"{STAGE2}: feature_blocks is [], not a non-empty list of feature "
                       f"blocks")


def test_stage2_feature_blocks_unknown_has_the_exact_message(tmp_path, repo):
    message = exact_message(tmp_path, repo, {("stage2", "feature_blocks"): '["weather"]'})
    assert message == (f"{STAGE2}: feature_blocks holds ['weather']; the feature blocks are "
                       f"['history', 'calendar', 'receivables']")


def test_stage2_feature_blocks_twice_has_the_exact_message(tmp_path, repo):
    message = exact_message(tmp_path, repo,
                            {("stage2", "feature_blocks"): '["history", "history"]'})
    assert message == (f"{STAGE2}: feature_blocks lists a feature block more than once: "
                       f"['history', 'history']")


def test_the_stage2_settings_are_checked_in_field_order(tmp_path, repo):
    # All six broken: each is reported in turn, in field order, as the earlier ones are fixed.
    changes = {("stage2", "max_iter"): "0", ("stage2", "learning_rate"): "2",
               ("stage2", "max_leaf_nodes"): "1", ("stage2", "min_samples_leaf"): "0",
               ("stage2", "random_seed"): "-1", ("stage2", "feature_blocks"): "[]"}
    for key, good in (("max_iter", "200"), ("learning_rate", "0.1"),
                      ("max_leaf_nodes", "15"), ("min_samples_leaf", "10"),
                      ("random_seed", "7"), ("feature_blocks", '["history"]')):
        assert exact_message(tmp_path, repo, changes).startswith(f"{STAGE2}: {key} is ")
        changes[("stage2", key)] = good
    assert load(tmp_path, repo, changes=changes).stage2.feature_blocks == ("history",)


def test_a_bad_stage2_value_is_reported_after_stage1(tmp_path, repo):
    # [stage1] is read before [stage2]: both broken, [stage1] is named.
    message = exact_message(tmp_path, repo, {("stage1", "max_iter"): "0",
                                             ("stage2", "max_iter"): "0"})
    assert message.startswith(f"{STAGE1}: max_iter is 0")


# --- [stage2] in the repo's own settings file ---------------------------------------------------

@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_stage2_settings():
    settings = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT)
    assert settings.stage2 == Stage2Settings(
        loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, random_seed=1,
        feature_blocks=("history", "calendar", "receivables"))


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml not added yet")
def test_the_synthetic_file_stage2_section_has_exactly_the_stage2_keys():
    raw = tomllib.loads(SYNTHETIC_SETTINGS.read_text(encoding="utf-8"))
    assert sorted(raw["stage2"]) == STAGE2_KEYS
    assert raw["stage2"]["loss"] in STAGE2_LOSSES
