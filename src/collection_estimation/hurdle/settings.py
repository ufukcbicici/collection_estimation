"""Settings for a run, read from a TOML file (e.g. settings/synthetic.toml).

Every setting is required, and any key the code does not know raises: a typo would
otherwise be ignored without notice and a different value used.
"""
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class DataSettings:
    """Where the inputs are, and where caches go."""
    corpus_folder: Path   # the folder of daily collection files
    ledger_file: Path     # the AR ledger
    ledger_as_of: date    # the date the ledger was extracted
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
class Settings:
    """All settings of a run, and the file they came from."""
    settings_file: Path
    data: DataSettings
    companies: CompanySettings
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

    check_keys(raw, {"data", "companies", "debug"}, name)

    data = raw["data"]
    check_keys(data, {"corpus_folder", "ledger_file", "ledger_as_of", "cache_folder"},
               f"{name} [data]")
    companies = raw["companies"]
    check_keys(companies, {"known"}, f"{name} [companies]")
    debug = raw["debug"]
    check_keys(debug, {"one_day"}, f"{name} [debug]")

    return Settings(
        settings_file=settings_file.resolve(),
        data=DataSettings(
            corpus_folder=path_setting(data, "corpus_folder", repo_root, f"{name} [data]"),
            ledger_file=path_setting(data, "ledger_file", repo_root, f"{name} [data]"),
            ledger_as_of=date_setting(data, "ledger_as_of", f"{name} [data]"),
            cache_folder=path_setting(data, "cache_folder", repo_root, f"{name} [data]"),
        ),
        companies=CompanySettings(
            known=text_list_setting(companies, "known", f"{name} [companies]"),
        ),
        debug=DebugSettings(
            one_day=date_setting(debug, "one_day", f"{name} [debug]"),
        ),
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
