"""Run the hurdle model (option B), step by step.

Open this file in PyCharm and press Run (or Debug). There are no command-line arguments:
the settings come from a TOML file (below), and each step is one line in `main()`, so you can
put a breakpoint after any step and look at what it produced.

Steps (filled in as the units are built; see option_b_implementation_plan.md):

    1  daily files     which days have a collections file          (U0)   <- built
    2  one day         the payment lines of one chosen file        (U0)   <- built
    3  collections     the payment lines of every file             (U0)   <- built
    4  keys            company + customer as one key               (U1)   <- built
    5  weekly series   receipts per key and week, missing weeks    (U2)   <- built
    6  AR ledger       the ledger lines (U9a); open invoices as known at each week (U9b)
    7  panel           rows customer x week x horizon, with labels (U3, U4, U5)
    8  features        history, calendar, inflation, receivables   (U6-U9c)
    9  models          stage 1, stage 2, top-ups                   (U10-U12)
   10  forecast, evaluation, calibration                           (U13-U15)

The settings (data paths, company codes, ...) are in settings/synthetic.toml. To run on
other data (e.g. the real EY files), write another settings file and point the environment
variable HURDLE_SETTINGS to it; nothing in the code needs to change.
"""
import os
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# The package lives under src/. Adding it here lets this script run from PyCharm or a
# terminal without installing the project.
sys.path.insert(0, str(REPO_ROOT / "src"))

# Customer names are Turkish; a Windows console would otherwise crash printing "İ".
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collection_estimation.hurdle.read_collections import (  # noqa: E402
    list_daily_files,
    load_collections,
    read_one_file,
)
from collection_estimation.hurdle.keys import add_keys  # noqa: E402
from collection_estimation.hurdle.read_ledger import (  # noqa: E402
    check_ledger_against_collections,
    load_ledger,
)
from collection_estimation.hurdle.settings import load_settings  # noqa: E402
from collection_estimation.hurdle.series import (  # noqa: E402
    add_week_index,
    week_calendar,
    weekly_amounts,
)

# ----------------------------------------------------------------------------------------
# Settings: read from a TOML file. The environment variable HURDLE_SETTINGS can point to
# another one (e.g. settings/real.toml for the real EY files).
# ----------------------------------------------------------------------------------------
DEFAULT_SETTINGS_FILE = REPO_ROOT / "settings" / "synthetic.toml"
SETTINGS_FILE = Path(os.environ.get("HURDLE_SETTINGS", DEFAULT_SETTINGS_FILE))


def main():
    settings = load_settings(SETTINGS_FILE, REPO_ROOT)
    print(f"Settings: {settings.settings_file}")
    print(f"Corpus folder: {settings.data.corpus_folder}")

    # Step 1: which days have a file. A day without a file is missing, not zero.
    daily_files = list_daily_files(settings.data.corpus_folder)
    show_daily_files(daily_files)

    # Step 2: the payment lines of one day, checked against the file's grand total.
    day = settings.debug.one_day
    if day not in daily_files:
        raise ValueError(f"{settings.settings_file.name} [debug]: one_day {day} has no daily file")
    one_day = read_one_file(daily_files[day])
    show_one_day(day, one_day)

    # Step 3: the payment lines of every day, each file checked as in step 2, and every
    # line's date checked against its file's date. Read from the cache if the files have
    # not changed since the last run.
    collections = load_collections(daily_files, settings.data.cache_folder)
    show_collections(collections)

    # Step 4: one key per company and customer, e.g. 'TR02-11226452'.
    collections = add_keys(collections, list(settings.companies.known))
    show_keys(collections)

    # Step 5: weekly series. Weeks run Monday to Sunday; week 0 is the week of the first
    # daily file.
    first_day = min(daily_files)
    week_zero = first_day - timedelta(days=first_day.weekday())
    collections = add_week_index(collections, week_zero)
    print(f"Step 5: week 0 starts {week_zero}; lines run from week "
          f"{collections['week'].min()} to week {collections['week'].max()}")

    # The week calendar: which weeks are complete. A week with any business day
    # without a file is missing: its amounts are unknown, not zero.
    weeks = week_calendar(list(daily_files), week_zero)
    show_weeks(weeks)

    # The weekly amounts per key: gross receipts (the target), refunds, unapplied lines.
    # Checked inside: the weekly table holds every TRY and every line of step 3.
    weekly = weekly_amounts(collections)
    show_weekly(weekly)

    # Step 6: the AR ledger lines, with the same keys as the payment lines. Read from the
    # cache if nothing has changed since the last run.
    ledger = load_ledger(settings.data.ledger_file, settings.data.ledger_as_of,
                         settings.data.cache_folder)
    ledger = add_keys(ledger, list(settings.companies.known))
    show_ledger(ledger)

    # The ledger's payments must equal the daily files' totals on every day with a file.
    tie = check_ledger_against_collections(ledger, collections, list(daily_files))
    show_tie(tie)

    # Step 6 continues, and later steps follow, as each unit is built.


def show_daily_files(daily_files):
    """A short printout of step 1."""
    dates = list(daily_files)
    print(f"Step 1: {len(dates)} daily files, from {dates[0]} to {dates[-1]}")
    print(f"        first file: {daily_files[dates[0]].name}")


def show_one_day(day, lines):
    """A short printout of step 2."""
    print(f"Step 2: {day}: {len(lines)} payment lines, "
          f"{lines['amount_try'].sum():,.2f} TRY in total (ties to the file's grand total)")
    print(lines[["company_code", "customer", "amount_try", "currency_original"]].head())


def show_collections(collections):
    """A short printout of step 3."""
    print(f"Step 3: {len(collections):,} payment lines, "
          f"{collections['amount_try'].sum():,.2f} TRY in total, "
          f"{collections['document_date'].nunique()} days")


def show_keys(collections):
    """A short printout of step 4."""
    keys_per_company = collections.groupby("company_code")["key"].nunique()
    print(f"Step 4: {collections['key'].nunique():,} keys; per company: "
          f"{keys_per_company.to_dict()}")


def show_weeks(weeks):
    """A short printout of the week calendar."""
    missing = weeks[weeks["is_missing"]]
    print(f"        {len(weeks)} weeks, {len(missing)} missing:")
    for row in missing.itertuples():
        print(f"          week {row.week} (Monday {row.monday}): "
              f"{row.missing_days} of {row.business_days} business days without a file")


def show_weekly(weekly):
    """A short printout of the weekly amounts."""
    paid = weekly["gross_receipts"] > 0
    print(f"        weekly amounts: {len(weekly):,} key-weeks with a line "
          f"({paid.sum():,} with a payment); "
          f"gross receipts {weekly['gross_receipts'].sum():,.2f} TRY, "
          f"refunds {weekly['refunds'].sum():,.2f} TRY, "
          f"{weekly['unapplied_lines'].sum():,} unapplied lines")


def show_ledger(ledger):
    """A short printout of the ledger."""
    lines_per_type = ledger["document_type"].value_counts().to_dict()
    still_open = ledger["clearing_date"].isna().sum()
    print(f"Step 6: {len(ledger):,} ledger lines {lines_per_type}, {still_open:,} still open; "
          f"{ledger['key'].nunique():,} keys; posted from "
          f"{ledger['posting_date'].min():%Y-%m-%d} to {ledger['posting_date'].max():%Y-%m-%d}")


def show_tie(tie):
    """A short printout of the ledger-to-daily-files check."""
    print(f"        ledger payments tie to the daily files on all {tie['days_compared']} days "
          f"with a file; also in the ledger: {tie['days_before_first_file']} days before the "
          f"first file ({tie['try_before_first_file']:,.2f} TRY) and "
          f"{tie['missing_file_days_in_ledger']} missing-file days "
          f"({tie['try_on_missing_file_days']:,.2f} TRY) and {tie['days_after_last_file']} days "
          f"after the last file ({tie['try_after_last_file']:,.2f} TRY)")


if __name__ == "__main__":
    main()
