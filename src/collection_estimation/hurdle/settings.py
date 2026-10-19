"""Settings for a run, read from a TOML file (e.g. settings/synthetic.toml).

Every setting is required, and any key the code does not know raises: a typo would
otherwise be ignored without notice and a different value used.
"""
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .read_collections import COLLECTION_FIELDS, OPTIONAL_FIELDS, simplify_header
from .read_ledger import DOCUMENT_ROLES, LEDGER_FIELDS, LEDGER_OPTIONAL_FIELDS


@dataclass(frozen=True)
class DataSettings:
    """Where the inputs are, and where caches go."""
    corpus_folder: Path   # the folder of daily collection files
    ledger_file: Path     # the AR ledger
    ledger_as_of: date    # the date the ledger was extracted
    holidays_file: Path   # the public holidays (e.g. settings/holidays_tr.toml)
    fill_missing_days_from_ledger: bool   # fill days without a daily file from the ledger
    cache_folder: Path


@dataclass(frozen=True)
class CompanySettings:
    """The company codes we expect, and the placeholder customers of unidentified cash."""
    known: tuple[str, ...]                    # any other company code stops the run
    unidentified_customers: tuple[str, ...]   # customer IDs that hold unidentified cash


@dataclass(frozen=True)
class ReceivablesSettings:
    """How the receivables features are built from the AR ledger."""
    min_cleared_for_delay: int   # fewer cleared invoices than this: usual delay unknown (NaN)
    horizons: int                # how many weeks ahead the "due in week t+h" columns cover
    leak_check_weeks: tuple[int, ...]   # weeks rebuilt from a cut-off ledger and compared
    overdue_age_days: tuple[int, ...]   # edges of the aging buckets, in days past due


@dataclass(frozen=True)
class HistorySettings:
    """How the customers' payment-history features are built."""
    windows: tuple[int, ...]             # look-back windows, in weeks, e.g. (4, 13, 52)
    use_ledger_before_first_file: bool   # ledger payments before the daily files, as history


@dataclass(frozen=True)
class Stage1Settings:
    """Stage 1, the "will it pay?" classifier."""
    max_iter: int                     # number of trees
    learning_rate: float              # how much each tree adds (above 0, at most 1)
    max_leaf_nodes: int               # leaves per tree (2 or more)
    min_samples_leaf: int             # rows per leaf, at least
    random_seed: int                  # makes a fit repeatable (0 or more)
    feature_blocks: tuple[str, ...]   # which feature blocks go in: history, calendar, receivables
    company_as_feature: bool          # the company code (an EY entity) as a categorical feature
    diagnose_class_weights: bool      # also fit a class-weighted model, for diagnosis only
    baseline_c: float                 # the logistic baseline's C: smaller = stronger regularisation


@dataclass(frozen=True)
class DebugSettings:
    """Settings that only change what is printed."""
    one_day: date         # the day whose payment lines step 2 prints
    sample_week: int      # the week step 7 shows in detail (0 = the week of the first file)
    demo_origin: int      # the forecast week step 12 fits and checks stage 1 at


@dataclass(frozen=True)
class CollectionsLayout:
    """How the daily collection files are laid out."""
    file_name_date_format: str   # e.g. "%d%m%Y": "02062025 EY TAHSILAT.XLSX" is 2 June 2025
    header_row: int
    first_data_row: int
    headers: dict[str, str]      # header text in the file -> our field name


@dataclass(frozen=True)
class LedgerLayout:
    """How the AR ledger is laid out."""
    header_row: int
    first_data_row: int
    headers: dict[str, str]                      # header text in the file -> our field name
    document_types: dict[str, tuple[str, ...]]   # role -> the SAP codes used for it


@dataclass(frozen=True)
class Settings:
    """All settings of a run, and the file they came from."""
    settings_file: Path
    data: DataSettings
    companies: CompanySettings
    collections_layout: CollectionsLayout
    ledger_layout: LedgerLayout
    receivables: ReceivablesSettings
    history: HistorySettings
    stage1: Stage1Settings
    debug: DebugSettings


def load_settings(settings_file: Path, repo_root: Path) -> Settings:
    """Read and check a settings file. Relative paths in it are taken from `repo_root`.

    Raises ValueError naming the file and the setting if anything is missing, unknown, or
    of the wrong type.
    """
    settings_file = Path(settings_file)
    with open(settings_file, "rb") as file:  # tomllib reads bytes
        raw = tomllib.load(file)
    name = settings_file.name

    check_keys(raw, {"data", "companies", "collections_layout", "ledger_layout", "receivables",
                     "history", "stage1", "debug"}, name)

    data = raw["data"]
    check_keys(data, {"corpus_folder", "ledger_file", "ledger_as_of", "holidays_file",
                      "fill_missing_days_from_ledger", "cache_folder"},
               f"{name} [data]")
    companies = raw["companies"]
    check_keys(companies, {"known", "unidentified_customers"}, f"{name} [companies]")
    receivables = raw["receivables"]
    check_keys(receivables, {"min_cleared_for_delay", "horizons", "leak_check_weeks",
                             "overdue_age_days"}, f"{name} [receivables]")
    history = raw["history"]
    check_keys(history, {"windows", "use_ledger_before_first_file"}, f"{name} [history]")
    stage1 = raw["stage1"]
    check_keys(stage1, {"max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                        "random_seed", "feature_blocks", "company_as_feature",
                        "diagnose_class_weights", "baseline_c"}, f"{name} [stage1]")
    debug = raw["debug"]
    check_keys(debug, {"one_day", "sample_week", "demo_origin"}, f"{name} [debug]")
    collections_layout = raw["collections_layout"]
    check_keys(collections_layout,
               {"file_name_date_format", "header_row", "first_data_row", "headers"},
               f"{name} [collections_layout]")
    ledger_layout = raw["ledger_layout"]
    check_keys(ledger_layout, {"header_row", "first_data_row", "headers", "document_types"},
               f"{name} [ledger_layout]")

    return Settings(
        settings_file=settings_file.resolve(),
        data=DataSettings(
            corpus_folder=path_setting(data, "corpus_folder", repo_root, f"{name} [data]"),
            ledger_file=path_setting(data, "ledger_file", repo_root, f"{name} [data]"),
            ledger_as_of=date_setting(data, "ledger_as_of", f"{name} [data]"),
            holidays_file=path_setting(data, "holidays_file", repo_root, f"{name} [data]"),
            fill_missing_days_from_ledger=bool_setting(data, "fill_missing_days_from_ledger",
                                                       f"{name} [data]"),
            cache_folder=path_setting(data, "cache_folder", repo_root, f"{name} [data]"),
        ),
        companies=CompanySettings(
            known=text_list_setting(companies, "known", f"{name} [companies]"),
            unidentified_customers=text_list_setting(companies, "unidentified_customers",
                                                     f"{name} [companies]", may_be_empty=True),
        ),
        collections_layout=read_collections_layout(collections_layout,
                                                   f"{name} [collections_layout]"),
        ledger_layout=read_ledger_layout(ledger_layout, f"{name} [ledger_layout]"),
        receivables=ReceivablesSettings(
            min_cleared_for_delay=count_setting(receivables, "min_cleared_for_delay",
                                                f"{name} [receivables]"),
            horizons=count_setting(receivables, "horizons", f"{name} [receivables]"),
            leak_check_weeks=week_list_setting(receivables, "leak_check_weeks",
                                               f"{name} [receivables]"),
            overdue_age_days=day_edges_setting(receivables, "overdue_age_days",
                                               f"{name} [receivables]"),
        ),
        history=HistorySettings(
            windows=window_list_setting(history, "windows", f"{name} [history]"),
            use_ledger_before_first_file=bool_setting(history, "use_ledger_before_first_file",
                                                      f"{name} [history]"),
        ),
        stage1=Stage1Settings(
            max_iter=count_setting(stage1, "max_iter", f"{name} [stage1]"),
            learning_rate=rate_setting(stage1, "learning_rate", f"{name} [stage1]"),
            max_leaf_nodes=leaf_count_setting(stage1, "max_leaf_nodes", f"{name} [stage1]"),
            min_samples_leaf=count_setting(stage1, "min_samples_leaf", f"{name} [stage1]"),
            random_seed=seed_setting(stage1, "random_seed", f"{name} [stage1]"),
            feature_blocks=feature_blocks_setting(stage1, "feature_blocks", f"{name} [stage1]"),
            company_as_feature=bool_setting(stage1, "company_as_feature", f"{name} [stage1]"),
            diagnose_class_weights=bool_setting(stage1, "diagnose_class_weights",
                                                f"{name} [stage1]"),
            baseline_c=positive_number_setting(stage1, "baseline_c", f"{name} [stage1]"),
        ),
        debug=DebugSettings(
            one_day=date_setting(debug, "one_day", f"{name} [debug]"),
            sample_week=week_setting(debug, "sample_week", f"{name} [debug]"),
            demo_origin=week_setting(debug, "demo_origin", f"{name} [debug]"),
        ),
    )


def read_collections_layout(table: dict, where: str) -> CollectionsLayout:
    """The [collections_layout] section, checked."""
    header_row, first_data_row = rows_setting(table, where)
    return CollectionsLayout(
        file_name_date_format=date_format_setting(table, "file_name_date_format", where),
        header_row=header_row,
        first_data_row=first_data_row,
        headers=headers_setting(table, "headers", COLLECTION_FIELDS, OPTIONAL_FIELDS, where),
    )


def read_ledger_layout(table: dict, where: str) -> LedgerLayout:
    """The [ledger_layout] section, checked."""
    header_row, first_data_row = rows_setting(table, where)
    return LedgerLayout(
        header_row=header_row,
        first_data_row=first_data_row,
        headers=headers_setting(table, "headers", LEDGER_FIELDS, LEDGER_OPTIONAL_FIELDS, where),
        document_types=document_types_setting(table, "document_types", where),
    )


def check_keys(table, expected: set[str], where: str) -> None:
    """Raise ValueError if `table` is not a TOML table with exactly the `expected` keys."""
    if not isinstance(table, dict):
        raise ValueError(f"{where} is {table!r}, not a section")
    missing = expected - set(table)
    unknown = set(table) - expected
    # Both in one message: a typo shows up as one missing and one unknown setting.
    problems = []
    if missing:
        problems.append(f"missing {sorted(missing)}")
    if unknown:
        problems.append(f"unknown {sorted(unknown)}")
    if problems:
        raise ValueError(f"{where}: {'; '.join(problems)}; expected exactly {sorted(expected)}")


def path_setting(table: dict, key: str, repo_root: Path, where: str) -> Path:
    """A path written as text; a relative path is taken from `repo_root`."""
    value = table[key]
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{where}: {key} is {value!r}, not a path")
    path = Path(value)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def date_setting(table: dict, key: str, where: str) -> date:
    """A date written in TOML's date form, e.g. 2026-08-28 (no quotes, no time of day)."""
    value = table[key]
    # A TOML date-time arrives as datetime, which is a subclass of date; it is rejected.
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ValueError(f"{where}: {key} is {value!r}, not a date such as 2026-08-28")
    return value


def text_list_setting(table: dict, key: str, where: str,
                      may_be_empty: bool = False) -> tuple[str, ...]:
    """A list of different, non-empty texts, e.g. ["TR02", "TR04"]; it must have at least
    one item unless `may_be_empty`."""
    value = table[key]
    if not isinstance(value, list):
        raise ValueError(f"{where}: {key} is {value!r}, not a list")
    if not value and not may_be_empty:
        raise ValueError(f"{where}: {key} is empty; it needs at least one item")
    for item in value:
        if not isinstance(item, str) or item.strip() == "":
            raise ValueError(f"{where}: {key} holds {item!r}, not a non-empty text")
    duplicates = sorted({item for item in value if value.count(item) > 1})
    if duplicates:
        # Most likely a copy-paste slip that hides a missing entry.
        raise ValueError(f"{where}: {key} lists {duplicates} more than once")
    return tuple(value)


def rows_setting(table: dict, where: str) -> tuple[int, int]:
    """header_row (a whole number, at least 1) and first_data_row (after the header)."""
    rows = {}
    for key in ("header_row", "first_data_row"):
        value = table[key]
        # bool is a subclass of int in Python, so it is excluded explicitly.
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{where}: {key} is {value!r}, not a row number (1 or more)")
        rows[key] = value
    if rows["first_data_row"] <= rows["header_row"]:
        raise ValueError(f"{where}: first_data_row {rows['first_data_row']} must come after "
                         f"header_row {rows['header_row']}")
    return rows["header_row"], rows["first_data_row"]


# Dates that test a file-name date format: different days, months and weekdays, including
# month and weekday names of different lengths (May / September, Friday / Wednesday).
FORMAT_SAMPLE_DATES = [date(2025, 5, 2), date(2025, 6, 2), date(2026, 9, 30), date(2026, 12, 31)]


def date_format_setting(table: dict, key: str, where: str) -> str:
    """A date format such as "%d%m%Y", checked by writing sample dates with it and reading
    them back.

    A format that loses part of a date (e.g. "%m%Y" has no day) raises. So does one whose
    written length changes from date to date (e.g. "%d%B%Y": "02May2025" vs "30September2026"),
    because the reader takes the date from a fixed number of characters at the start of a
    file name.
    """
    value = table[key]
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{where}: {key} is {value!r}, not a date format such as '%d%m%Y'")
    for sample in FORMAT_SAMPLE_DATES:
        try:
            read_back = datetime.strptime(sample.strftime(value), value).date()
        except ValueError:
            read_back = None
        if read_back != sample:
            raise ValueError(f"{where}: {key} {value!r} cannot write and read back a full "
                             f"date (tried {sample})")
    lengths = {len(sample.strftime(value)) for sample in FORMAT_SAMPLE_DATES}
    if len(lengths) > 1:
        raise ValueError(f"{where}: {key} {value!r} writes dates of different lengths "
                         f"{sorted(lengths)}; use numbers only, e.g. '%d%m%Y'")
    return value


def headers_setting(table: dict, key: str, fields: list[str], optional_fields: set[str],
                    where: str) -> dict[str, str]:
    """A table of header text = field name.

    Every field name must be one of `fields`, every field not in `optional_fields` must be
    mapped at least once, and no two header texts may be the same after simplifying (case,
    spaces, Turkish dotted I). Two header texts may map to the same field, e.g. an English
    and a Turkish one.
    """
    value = table[key]
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{where}: {key} is {value!r}, not a table of header = field")

    seen = {}  # simplified header text -> the header text as written
    for header, field in value.items():
        if not isinstance(field, str) or field not in fields:
            raise ValueError(f"{where}.{key}: '{header}' maps to {field!r}, not one of {fields}")
        simple = simplify_header(header)
        if simple == "":
            raise ValueError(f"{where}.{key}: a header text is empty")
        if simple in seen:
            raise ValueError(f"{where}.{key}: '{seen[simple]}' and '{header}' are the same "
                             f"header once case and spaces are ignored")
        seen[simple] = header

    required = [field for field in fields if field not in optional_fields]
    not_mapped = [field for field in required if field not in value.values()]
    if not_mapped:
        raise ValueError(f"{where}.{key}: no header for the required fields {not_mapped}")
    return dict(value)


def document_types_setting(table: dict, key: str, where: str) -> dict[str, tuple[str, ...]]:
    """A table of role = list of SAP codes, with exactly the roles in DOCUMENT_ROLES; no code
    may belong to two roles.

    Codes are stripped and upper-cased, as the ledger reader does with each line's document
    type, so "rv" in the settings matches "RV" in the ledger.
    """
    value = table[key]
    check_keys(value, set(DOCUMENT_ROLES), f"{where}.{key}")
    roles = {}
    role_of_code = {}
    for role in DOCUMENT_ROLES:
        written = text_list_setting(value, role, f"{where}.{key}")
        codes = tuple(code.strip().upper() for code in written)
        if len(set(codes)) < len(codes):
            raise ValueError(f"{where}.{key}: {role} lists the same code twice when case and "
                             f"spaces are ignored: {list(written)}")
        for code in codes:
            if code in role_of_code:
                raise ValueError(f"{where}.{key}: code '{code}' is listed for both "
                                 f"{role_of_code[code]} and {role}")
            role_of_code[code] = role
        roles[role] = codes
    return roles


def week_setting(table: dict, key: str, where: str) -> int:
    """A week number: a whole number, 0 or more (0 is the week of the first daily file)."""
    value = table[key]
    # bool is a subclass of int in Python, so it is excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: {key} is {value!r}, not a week number (0 or more)")
    return value


def count_setting(table: dict, key: str, where: str) -> int:
    """A count: a whole number, 1 or more."""
    value = table[key]
    # bool is a subclass of int in Python, so it is excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: {key} is {value!r}, not a whole number of 1 or more")
    return value


def week_list_setting(table: dict, key: str, where: str) -> tuple[int, ...]:
    """A list of different week numbers (each a whole number, 0 or more); may be empty."""
    value = table[key]
    if not isinstance(value, list):
        raise ValueError(f"{where}: {key} is {value!r}, not a list of week numbers")
    for item in value:
        # bool is a subclass of int in Python, so it is excluded explicitly.
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{where}: {key} holds {item!r}, not a week number (0 or more)")
    duplicates = sorted({item for item in value if value.count(item) > 1})
    if duplicates:
        raise ValueError(f"{where}: {key} lists {duplicates} more than once")
    return tuple(value)


def bool_setting(table: dict, key: str, where: str) -> bool:
    """A yes/no setting written as true or false (no quotes)."""
    value = table[key]
    if not isinstance(value, bool):
        raise ValueError(f"{where}: {key} is {value!r}, not true or false")
    return value


def window_list_setting(table: dict, key: str, where: str) -> tuple[int, ...]:
    """A non-empty list of different window lengths in weeks (whole numbers, 1 or more),
    returned sorted from shortest to longest."""
    value = table[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: {key} is {value!r}, not a non-empty list of window lengths")
    for item in value:
        # bool is a subclass of int in Python, so it is excluded explicitly.
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"{where}: {key} holds {item!r}, not a window length (1 or more)")
    duplicates = sorted({item for item in value if value.count(item) > 1})
    if duplicates:
        raise ValueError(f"{where}: {key} lists {duplicates} more than once")
    return tuple(sorted(value))


def day_edges_setting(table: dict, key: str, where: str) -> tuple[int, ...]:
    """A non-empty list of whole numbers of days (1 or more), given from smallest to largest
    with no repeats, e.g. [30, 90]: the edges of buckets 0-30, 31-90 and over 90 days."""
    value = table[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: {key} is {value!r}, not a non-empty list of days")
    for item in value:
        # bool is a subclass of int in Python, so it is excluded explicitly.
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"{where}: {key} holds {item!r}, not a number of days (1 or more)")
    if value != sorted(set(value)):
        raise ValueError(f"{where}: {key} is {value}; give the days from smallest to largest, "
                         f"each once")
    return tuple(value)


# The feature blocks a model can use, in the order their columns are taken.
FEATURE_BLOCKS = ("history", "calendar", "receivables")


def rate_setting(table: dict, key: str, where: str) -> float:
    """A rate above 0 and at most 1, e.g. a learning rate of 0.05."""
    value = table[key]
    # bool is a subclass of int in Python, so it is excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ValueError(f"{where}: {key} is {value!r}, not a number above 0 and at most 1")
    return float(value)


def leaf_count_setting(table: dict, key: str, where: str) -> int:
    """A number of leaves per tree: a whole number, 2 or more (one leaf would not split)."""
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{where}: {key} is {value!r}, not a whole number of 2 or more")
    return value


def feature_blocks_setting(table: dict, key: str, where: str) -> tuple[str, ...]:
    """A non-empty list of feature blocks, each once, from FEATURE_BLOCKS; returned in the
    order of FEATURE_BLOCKS."""
    value = table[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: {key} is {value!r}, not a non-empty list of feature blocks")
    unknown = [item for item in value if item not in FEATURE_BLOCKS]
    if unknown:
        raise ValueError(f"{where}: {key} holds {unknown}; the feature blocks are "
                         f"{list(FEATURE_BLOCKS)}")
    if len(set(value)) != len(value):
        raise ValueError(f"{where}: {key} lists a feature block more than once: {value}")
    return tuple(block for block in FEATURE_BLOCKS if block in value)


def seed_setting(table: dict, key: str, where: str) -> int:
    """A random seed: a whole number, 0 or more."""
    value = table[key]
    # bool is a subclass of int in Python, so it is excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: {key} is {value!r}, not a random seed (a whole number, "
                         f"0 or more)")
    return value


def positive_number_setting(table: dict, key: str, where: str) -> float:
    """A number above 0 (whole or not, finite), returned as a float."""
    value = table[key]
    # bool is a subclass of int in Python, so it is excluded explicitly. value == value is False
    # only for NaN, and inf is not a usable setting either.
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or value != value
            or value <= 0 or value == float("inf")):
        raise ValueError(f"{where}: {key} is {value!r}, not a number above 0")
    return float(value)
