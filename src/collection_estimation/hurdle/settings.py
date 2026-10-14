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
    cache_folder: Path


@dataclass(frozen=True)
class CompanySettings:
    """The company codes we expect; a line with any other code stops the run."""
    known: tuple[str, ...]


@dataclass(frozen=True)
class DebugSettings:
    """Settings that only change what is printed."""
    one_day: date         # the day whose payment lines step 2 prints


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

    check_keys(raw, {"data", "companies", "collections_layout", "ledger_layout", "debug"},
               name)

    data = raw["data"]
    check_keys(data, {"corpus_folder", "ledger_file", "ledger_as_of", "holidays_file",
                      "cache_folder"},
               f"{name} [data]")
    companies = raw["companies"]
    check_keys(companies, {"known"}, f"{name} [companies]")
    debug = raw["debug"]
    check_keys(debug, {"one_day"}, f"{name} [debug]")
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
            cache_folder=path_setting(data, "cache_folder", repo_root, f"{name} [data]"),
        ),
        companies=CompanySettings(
            known=text_list_setting(companies, "known", f"{name} [companies]"),
        ),
        collections_layout=read_collections_layout(collections_layout,
                                                   f"{name} [collections_layout]"),
        ledger_layout=read_ledger_layout(ledger_layout, f"{name} [ledger_layout]"),
        debug=DebugSettings(
            one_day=date_setting(debug, "one_day", f"{name} [debug]"),
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


def text_list_setting(table: dict, key: str, where: str) -> tuple[str, ...]:
    """A non-empty list of different, non-empty texts, e.g. ["TR02", "TR04"]."""
    value = table[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: {key} is {value!r}, not a non-empty list")
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
