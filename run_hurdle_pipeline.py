"""Run the hurdle model (option B), step by step.

Open this file in PyCharm and press Run (or Debug). There are no command-line arguments:
the settings are constants at the top, and each step is one line in `main()`, so you can
put a breakpoint after any step and look at what it produced.

Steps (filled in as the units are built; see option_b_implementation_plan.md):

    1  daily files     which days have a collections file          (U0)   <- built
    2  one day         the payment lines of one chosen file        (U0)   <- built
    3  collections     the payment lines of every file             (U0)   <- built
    4  keys            company + customer as one key               (U1)   <- built
    5  weekly series   receipts per key and week, missing weeks    (U2)   <- built
    6  AR ledger       open invoices as known at each week         (U9a, U9b)
    7  panel           rows customer x week x horizon, with labels (U3, U4, U5)
    8  features        history, calendar, inflation, receivables   (U6-U9c)
    9  models          stage 1, stage 2, top-ups                   (U10-U12)
   10  forecast, evaluation, calibration                           (U13-U15)

To run on other files (e.g. the real EY files), set the environment variable
COLLECTIONS_CORPUS_DIR to their folder; nothing in the code needs to change.
"""
import os
import sys
from datetime import date, timedelta  # date: for ONE_DAY
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
from collection_estimation.hurdle.series import (  # noqa: E402
    add_week_index,
    week_calendar,
    weekly_amounts,
)

# ----------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"
CORPUS_FOLDER = Path(os.environ.get("COLLECTIONS_CORPUS_DIR", SYNTHETIC_CORPUS))

# EY's company codes; a line with any other code stops the run (step 4).
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]

# Where step 3 caches the table of all payment lines (rebuilt when the files change).
CACHE_FOLDER = REPO_ROOT / "data" / "interim"

# Step 2 reads this one day; None means the first day with a file.
ONE_DAY = None  # e.g. date(2026, 3, 2)


def main():
    print(f"Corpus folder: {CORPUS_FOLDER}")

    # Step 1: which days have a file. A day without a file is missing, not zero.
    daily_files = list_daily_files(CORPUS_FOLDER)
    show_daily_files(daily_files)

    # Step 2: the payment lines of one day, checked against the file's grand total.
    day = ONE_DAY or min(daily_files)
    one_day = read_one_file(daily_files[day])
    show_one_day(day, one_day)

    # Step 3: the payment lines of every day, each file checked as in step 2, and every
    # line's date checked against its file's date. Read from the cache if the files have
    # not changed since the last run.
    collections = load_collections(daily_files, CACHE_FOLDER)
    show_collections(collections)

    # Step 4: one key per company and customer, e.g. 'TR02-11226452'.
    collections = add_keys(collections, KNOWN_COMPANIES)
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

    # Step 6 onwards: added as each unit is built.


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


if __name__ == "__main__":
    main()
