"""Run the hurdle model (option B), step by step.

Open this file in PyCharm and press Run (or Debug). There are no command-line arguments:
the settings are constants at the top, and each step is one line in `main()`, so you can
put a breakpoint after any step and look at what it produced.

Steps (filled in as the units are built; see option_b_implementation_plan.md):

    1  daily files     which days have a collections file          (U0)   <- built
    2  one day         the payment lines of one chosen file        (U0)   <- built
    3  collections     the payment lines of every file             (U0)   <- built
    4  keys            company + customer as one key               (U1)
    5  weekly series   receipts per key and week, missing weeks    (U2)
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
from datetime import date  # noqa: F401  (for ONE_DAY)
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

# ----------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"
CORPUS_FOLDER = Path(os.environ.get("COLLECTIONS_CORPUS_DIR", SYNTHETIC_CORPUS))

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

    # Step 4 onwards: added as each unit is built.


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


if __name__ == "__main__":
    main()
