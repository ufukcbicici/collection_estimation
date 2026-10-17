"""Run the hurdle model (option B), step by step.

Open this file in PyCharm and press Run (or Debug). There are no command-line arguments:
the settings come from a TOML file (below), and each step is one line in `main()`, so you can
put a breakpoint after any step and look at what it produced.

Steps (filled in as the units are built; see option_b_implementation_plan.md):

    1  daily files     which days have a collections file          (U0)   <- built
    2  one day         the payment lines of one chosen file        (U0)   <- built
    3  collections     the payment lines of every file             (U0)   <- built
    4  keys            company + customer as one key               (U1)   <- built
    5  AR ledger       the ledger lines and their tie to the daily files (U9a)   <- built
    6  missing days    days without a file, filled from the ledger (or left missing) <- built
    7  weekly series   receipts per key and week, missing weeks    (U2)   <- built
    8  receivables     each key's receivables as known at each week (U9b)    <- built
    9  panel           rows customer x week x horizon, with labels (U3, U4, U5)  <- in progress
   10  features        history, calendar, inflation, receivables   (U6-U9c)
   11  models          stage 1, stage 2, top-ups                   (U10-U12)
   12  forecast, evaluation, calibration                           (U13-U15)

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

import pandas as pd  # noqa: E402

from collection_estimation.hurdle.read_collections import (  # noqa: E402
    list_daily_files,
    load_collections,
    read_one_file,
)
from collection_estimation.hurdle.fill_from_ledger import fill_missing_days  # noqa: E402
from collection_estimation.hurdle.holidays import load_holidays  # noqa: E402
from collection_estimation.hurdle.keys import add_keys  # noqa: E402
from collection_estimation.hurdle.known import known_weeks  # noqa: E402
from collection_estimation.hurdle.panel import add_labels, panel_rows  # noqa: E402
from collection_estimation.hurdle.side_tables import (  # noqa: E402
    check_side_tables,
    is_unidentified,
    new_customer_cash,
    refunds_by_week,
    unidentified_cash_by_week,
)
from collection_estimation.hurdle.receivables_asof import (  # noqa: E402
    exchange_rates,
    receivables_by_week,
)
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
    layout = settings.collections_layout
    daily_files = list_daily_files(settings.data.corpus_folder,
                                   layout.file_name_date_format)
    show_daily_files(daily_files)

    # Step 2: the payment lines of one day, checked against the file's grand total.
    day = settings.debug.one_day
    if day not in daily_files:
        raise ValueError(f"{settings.settings_file.name} [debug]: one_day {day} has no daily file")
    one_day = read_one_file(daily_files[day], layout)
    show_one_day(day, one_day)

    # Step 3: the payment lines of every day, each file checked as in step 2, and every
    # line's date checked against its file's date. Read from the cache if the files have
    # not changed since the last run.
    collections = load_collections(daily_files, layout, settings.data.cache_folder)
    show_collections(collections)

    # Step 4: one key per company and customer, e.g. 'TR02-11226452'.
    collections = add_keys(collections, list(settings.companies.known))
    show_keys(collections)

    # Step 5: the AR ledger lines, with the same keys as the payment lines. Read from the
    # cache if nothing has changed since the last run. Its payments must equal the daily
    # files' totals on every day with a file.
    ledger = load_ledger(settings.data.ledger_file, settings.data.ledger_as_of,
                         settings.ledger_layout, settings.data.cache_folder)
    ledger = add_keys(ledger, list(settings.companies.known))
    show_ledger(ledger)
    tie = check_ledger_against_collections(ledger, collections, list(daily_files))
    show_tie(tie)

    # Step 6: days without a daily file. If [data] fill_missing_days_from_ledger is true,
    # their payments are taken from the ledger (flagged filled_from_ledger); otherwise
    # their weeks stay missing.
    if settings.data.fill_missing_days_from_ledger:
        collections, filled_days = fill_missing_days(collections, ledger, list(daily_files))
    else:
        collections = collections.assign(filled_from_ledger=False)
        filled_days = []
    show_filled(collections, filled_days, settings.data.fill_missing_days_from_ledger)

    # Step 7: weekly series. Weeks run Monday to Sunday; week 0 is the week of the first
    # daily file.
    first_day = min(daily_files)
    week_zero = first_day - timedelta(days=first_day.weekday())
    collections = add_week_index(collections, week_zero)
    print(f"Step 7: week 0 starts {week_zero}; lines run from week "
          f"{collections['week'].min()} to week {collections['week'].max()}")

    # The week calendar: which weeks are complete. A week with any business day with
    # neither a file nor filled payments is missing: its amounts are unknown, not zero.
    holidays = load_holidays(settings.data.holidays_file)
    weeks = week_calendar(list(daily_files), week_zero, holidays, filled_days)
    show_weeks(weeks)

    # The weekly amounts per key: gross receipts (the target), refunds, unapplied lines.
    # Checked inside: the weekly table holds every TRY and every line of steps 3 and 6.
    weekly = weekly_amounts(collections)
    show_weekly(weekly)

    # Step 8: each key's receivables as known at the end of every week, in TRY at the rates
    # known then (one row per key and week). Weeks named in [receivables] leak_check_weeks
    # are rebuilt from a ledger cut off at that week and compared.
    rates = exchange_rates(collections, ledger)
    show_rates(rates)
    receivables = receivables_by_week(
        ledger, rates, week_zero, last_week=int(weeks["week"].max()),
        as_of=settings.data.ledger_as_of, horizons=settings.receivables.horizons,
        min_cleared=settings.receivables.min_cleared_for_delay,
        leak_check_weeks=settings.receivables.leak_check_weeks)
    show_receivables_by_week(receivables, settings.receivables.leak_check_weeks,
                             settings.debug.sample_week)

    # Step 9: the panel. Built unit by unit; for now, when each key becomes known (the
    # week of its first interaction, in the ledger or the daily files). From that week on
    # it gets panel rows.
    known = known_weeks(collections, ledger, week_zero)
    show_known(known, collections, int(weeks["week"].max()))

    # Placeholder customers of unidentified cash ([companies] unidentified_customers) are not
    # customers: they get no panel rows, and their cash goes to table G below.
    unidentified_customers = settings.companies.unidentified_customers
    known = known[~is_unidentified(known, unidentified_customers)]
    unidentified_weekly = weekly[is_unidentified(weekly, unidentified_customers)]
    identified_weekly = weekly[~is_unidentified(weekly, unidentified_customers)]
    # The companies we expect (from the settings), not the ones left in `known`: a company
    # whose only keys are placeholders must still get its rows in tables R and G.
    companies = list(settings.companies.known)

    # The panel's rows (no labels yet): for every forecast week t, every key known by then,
    # once for each week ahead h = 1 ... horizons.
    panel = panel_rows(known, int(weeks["week"].max()), settings.receivables.horizons)
    show_panel_rows(panel)

    # The labels: what each key paid in the target week (gross receipts, TRY), or unknown
    # when the target week is missing or after the last week.
    panel = add_labels(panel, identified_weekly, weeks)
    show_labels(panel, identified_weekly)

    # The side tables: the cash the panel does not hold. A: gross receipts in week t+h of keys
    # not yet known at forecast week t.
    new_cash = new_customer_cash(identified_weekly, known, weeks, settings.receivables.horizons,
                                 companies)
    show_new_customer_cash(new_cash, weekly)

    # R: each company's refunds per week (negative), all keys together.
    refunds = refunds_by_week(weekly, weeks, companies)
    show_refunds(refunds)

    # G: each company's cash per week with no identifiable customer.
    unidentified = unidentified_cash_by_week(unidentified_weekly, weeks, companies)
    print(f"        unidentified cash (G): {len(unidentified):,} rows, "
          f"{unidentified['unidentified_cash'].sum():,.2f} TRY from "
          f"{len(unidentified_customers)} placeholder customers")

    # The tie-out: panel + A + G must equal every company's gross receipts in every labelled
    # target week, and R its refunds, so no cash is lost or counted twice.
    checked = check_side_tables(panel, new_cash, unidentified, refunds, weekly, companies)
    print(f"        tie-out passed: {checked:,} company-week-horizon checks "
          f"(panel + A + G = gross receipts; R = refunds)")

    # Step 9 continues, and later steps follow, as each unit is built.


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
    lines_per_type = ledger["document_role"].value_counts().to_dict()
    still_open = ledger["clearing_date"].isna().sum()
    print(f"Step 5: {len(ledger):,} ledger lines {lines_per_type}, {still_open:,} still open; "
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


def show_rates(rates):
    """A short printout of the exchange rates."""
    print(f"Step 8: exchange rates for {rates['currency'].nunique()} currencies:")
    for currency, rows in rates.groupby("currency"):
        print(f"          {currency}: {len(rows)} days from {rows['day'].min()} to "
              f"{rows['day'].max()}, {rows['rate'].min():.2f} to {rows['rate'].max():.2f} TRY")


def show_receivables_by_week(receivables, leak_check_weeks, sample_week):
    """A short printout of the receivables of every week."""
    print(f"        receivables: {len(receivables):,} rows, {receivables['key'].nunique():,} "
          f"keys, weeks {receivables['week'].min()} to {receivables['week'].max()}; "
          f"leak check passed for weeks {list(leak_check_weeks)}")
    columns = ["open_balance", "overdue", "due_in_week_1", "unapplied_cash"]
    totals = receivables.groupby("week")[columns].sum() / 1e6
    every_eighth_week = totals.loc[totals.index % 8 == 0]
    print("        totals per week, million TRY (every 8th week):")
    print(every_eighth_week.round(1).to_string())
    one_week = receivables[receivables["week"] == sample_week]
    print(f"        week {sample_week}: {len(one_week):,} keys, "
          f"{(one_week['open_invoices'] > 0).sum():,} with an open invoice, "
          f"{one_week['usual_delay_days'].notna().sum():,} with a usual delay")


def show_known(known, collections, last_week):
    """A short printout of when keys become known."""
    before = (known["known_week"] < 0).sum()
    during = known["known_week"].between(0, last_week)
    print(f"Step 9: {len(known):,} keys; {before:,} known before week 0, {during.sum():,} "
          f"first seen in weeks 0-{last_week} (about {during.sum() / (last_week + 1):.0f} a week)")
    print(f"        first seen in: {known['first_seen_in'].value_counts().to_dict()}")
    first_payment = collections.groupby("key")["document_date"].min()
    gap = (first_payment - known.set_index("key")["first_seen"].reindex(first_payment.index))
    print(f"        from first interaction to first payment in the daily files: median "
          f"{gap.dt.days.median():.0f} days")


def show_filled(collections, filled_days, switched_on):
    """A short printout of the filling of days without a daily file."""
    if not switched_on:
        print("Step 6: filling from the ledger is switched off; weeks with a missing day stay "
              "missing")
        return
    filled = collections[collections["filled_from_ledger"]]
    print(f"Step 6: {len(filled_days)} days without a daily file filled from the ledger: "
          f"{len(filled):,} payment lines, {filled['amount_try'].sum():,.2f} TRY "
          f"({', '.join(str(day) for day in filled_days)})")


def show_panel_rows(panel):
    """A short printout of the panel's rows."""
    per_origin = panel.groupby("origin_week")["key"].nunique()
    print(f"        panel rows: {len(panel):,} (keys x forecast weeks {panel['origin_week'].min()}"
          f"-{panel['origin_week'].max()} x horizons 1-{panel['h'].max()}); known keys per "
          f"forecast week from {per_origin.iloc[0]:,} to {per_origin.iloc[-1]:,}")


def show_labels(panel, weekly):
    """A short printout of the panel's labels."""
    print(f"        label status: {panel['label_status'].value_counts().to_dict()}")
    labelled = panel[panel["label_status"] == "labelled"]
    paid = labelled[labelled["paid"] == True]  # noqa: E712 (paid is a nullable boolean)
    print(f"        labelled rows with a payment: {len(paid):,} of {len(labelled):,} "
          f"({len(paid) / len(labelled):.1%}); median amount when paid "
          f"{paid['amount'].median():,.0f} TRY")
    # Cash in the target week from keys not yet known at the origin: the new-customer allowance.
    one_horizon = labelled[labelled["h"] == 1]
    in_panel = one_horizon.groupby("target_week")["amount"].sum()
    receipts = weekly.groupby("week")["gross_receipts"].sum().reindex(in_panel.index)
    print(f"        at h=1, the panel holds {in_panel.sum() / receipts.sum():.2%} of the target "
          f"weeks' receipts; the rest is from keys not yet known (the allowance, U5)")


def show_new_customer_cash(new_cash, weekly):
    """A short printout of table A: its share of the target weeks' cash, by horizon."""
    week_receipts = weekly.groupby("week")["gross_receipts"].sum()
    per_h = new_cash.groupby("h")["new_customer_cash"].sum()
    # Each (origin, h) points at one target week: its cash, all companies together.
    one_per_target = new_cash.drop_duplicates(["origin_week", "h"]).copy()
    one_per_target["week_cash"] = one_per_target["target_week"].map(week_receipts)
    cash_per_h = one_per_target.groupby("h")["week_cash"].sum()
    shares = {h: f"{per_h[h] / cash_per_h[h]:.2%}" for h in per_h.index}
    print(f"        new-customer cash (A): {len(new_cash):,} rows; share of the target weeks' "
          f"cash by h: {shares}")


def show_refunds(refunds):
    """A short printout of table R."""
    weeks_with_refunds = refunds.loc[refunds["refunds"] < 0, "week"].nunique()
    per_company = refunds.groupby("company_code")["refunds"].sum().round().to_dict()
    print(f"        refunds (R): {len(refunds):,} rows, {refunds['refunds'].sum():,.2f} TRY, "
          f"in {weeks_with_refunds} of {refunds['week'].nunique()} weeks; per company: "
          f"{per_company}")


if __name__ == "__main__":
    main()
