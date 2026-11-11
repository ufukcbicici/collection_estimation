"""The model's inputs, prepared once: steps 1-11 of the pipeline in one call.

`prepare_inputs` reads the daily files and the AR ledger, builds the weeks, the receivables,
the panel and its side tables, the price levels and the features, and returns what the
forecast, the evaluation and the grid need (`ModelInputs`). `run_hurdle_pipeline.py`, the
grid's own entry point and later the production entry call it, so the feature steps exist
in one place only.

Every check that stops a run stays in it and fails where it did: the ledger's tie to the
daily files (step 5), the weekly table's tie to the payment lines (step 7), the receivables'
leak check (step 8) and the side tables' tie-out, panel + A + G = gross receipts (step 9).

It prints nothing of its own. With `with_details=True` it also returns the tables made on
the way (`PreparationDetails`), so the pipeline can print each step as before; without, they
are dropped as soon as the inputs are made (memory).
"""
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .features_calendar import calendar_features, holiday_position_features
from .features_history import (
    first_history_week,
    history_features,
    ledger_receipts_before,
    weekly_matrix,
)
from .features_receivables import receivables_features
from .fill_from_ledger import fill_missing_days
from .holidays import HolidayTable, load_holidays
from .keys import add_keys
from .known import known_weeks
from .panel import add_deflated_label, add_labels, panel_rows
from .prices import published_levels
from .read_collections import list_daily_files, load_collections
from .read_ledger import check_ledger_against_collections, load_ledger
from .receivables_asof import exchange_rates, receivables_by_week
from .series import add_week_index, week_calendar, week_numbers, weekly_amounts
from .side_tables import (
    check_side_tables,
    is_unidentified,
    new_customer_cash,
    refunds_by_week,
    unidentified_cash_by_week,
)
from .training_table import block_feature_columns, join_features


@dataclass(frozen=True)
class ModelInputs:
    """What the forecast (U13), the evaluation (U14) and the grid (U16) are made from."""
    companies: list[str]        # [companies] known, in the settings' order
    week_zero: date             # the Monday of week 0, the week of the first daily file
    last_week: int              # the last week with data
    weeks: pd.DataFrame         # the week calendar (`week_calendar`): which weeks are missing
    weekly: pd.DataFrame        # each key's weekly amounts (`weekly_amounts`), all keys
    ledger: pd.DataFrame        # the AR ledger lines, with keys
    rates: pd.DataFrame         # the exchange rates (`exchange_rates`)
    levels: pd.Series           # the published TÜFE level per week, 1.0 = week 0
    calendar: pd.DataFrame      # the calendar features with the holiday positions, per week
    table: pd.DataFrame         # the training table (`join_features`)
    block_columns: dict[str, list[str]]   # each block's feature columns
    new_cash: pd.DataFrame      # table A: cash of keys not yet known at the origin
    refunds: pd.DataFrame       # table R: each company's refunds per week
    unidentified: pd.DataFrame  # table G: each company's cash with no identifiable customer


@dataclass(frozen=True)
class PreparationDetails:
    """The tables made on the way, for the pipeline's printouts only. Where a table changes
    from one step to the next, the one kept is named after the step it is printed at."""
    daily_files: dict[date, Path]       # step 1: day -> its file
    collections_keyed: pd.DataFrame     # steps 3-4: the payment lines with keys, not filled
    tie: dict                           # step 5: the ledger's tie to the daily files
    filled_days: list[date]             # step 6: days filled from the ledger
    collections: pd.DataFrame           # steps 6-9: filled, with the week index
    holidays: HolidayTable              # step 7: the public holidays
    receivables: pd.DataFrame           # step 8: each key's receivables at every week
    known: pd.DataFrame                 # step 9: when each key becomes known, placeholders too
    identified_weekly: pd.DataFrame     # step 9: `weekly` without the placeholder customers
    panel: pd.DataFrame                 # steps 9-11: the panel, with its deflated label
    checked: int                        # step 9: company-week-horizon checks of the tie-out
    first_price_week: int               # step 10: the first week with a price level
    history_start: int                  # step 11: the first week of payment history
    early_receipts: pd.DataFrame        # step 11: ledger payments before the first daily file
    matrix: pd.DataFrame                # step 11: each key's deflated receipts per week
    history: pd.DataFrame               # step 11: the history features
    positions: pd.DataFrame             # step 11: the holiday positions alone
    owed: pd.DataFrame                  # step 11: the receivables features


def prepare_inputs(settings, *, with_details: bool = False
                   ) -> tuple[ModelInputs, PreparationDetails | None]:
    """Steps 1-11 of the pipeline (see the module docstring) from `settings` (a `Settings`).

    Returns (inputs, details); details is None unless `with_details`. Raises whatever the
    steps' own checks raise (ValueError), at the step that fails.
    """
    # Step 1: which days have a file. A day without a file is missing, not zero.
    layout = settings.collections_layout
    daily_files = list_daily_files(settings.data.corpus_folder, layout.file_name_date_format)

    # Steps 3 and 4: the payment lines of every day (each file checked against its grand
    # total, read from the cache if the files have not changed), with one key per company
    # and customer, e.g. 'TR02-11226452'.
    companies = list(settings.companies.known)
    collections = load_collections(daily_files, layout, settings.data.cache_folder)
    collections = add_keys(collections, companies)
    collections_keyed = collections

    # Step 5: the AR ledger lines, with the same keys. Its payments must equal the daily
    # files' totals on every day with a file.
    ledger = load_ledger(settings.data.ledger_file, settings.data.ledger_as_of,
                         settings.ledger_layout, settings.data.cache_folder)
    ledger = add_keys(ledger, companies)
    tie = check_ledger_against_collections(ledger, collections, list(daily_files))

    # Step 6: days without a daily file, filled from the ledger if [data]
    # fill_missing_days_from_ledger; otherwise their weeks stay missing.
    if settings.data.fill_missing_days_from_ledger:
        collections, filled_days = fill_missing_days(collections, ledger, list(daily_files))
    else:
        collections = collections.assign(filled_from_ledger=False)
        filled_days = []

    # Step 7: weeks run Monday to Sunday; week 0 is the week of the first daily file. A week
    # with a business day with neither a file nor filled payments is missing.
    first_day = min(daily_files)
    week_zero = first_day - timedelta(days=first_day.weekday())
    collections = add_week_index(collections, week_zero)
    holidays = load_holidays(settings.data.holidays_file)
    weeks = week_calendar(list(daily_files), week_zero, holidays, filled_days)
    last_week = int(weeks["week"].max())
    weekly = weekly_amounts(collections)

    # Step 8: each key's receivables as known at the end of every week, in TRY at the rates
    # known then; the [receivables] leak_check_weeks are rebuilt from a cut ledger and compared.
    rates = exchange_rates(collections, ledger)
    receivables = receivables_by_week(
        ledger, rates, week_zero, last_week=last_week, as_of=settings.data.ledger_as_of,
        horizons=settings.receivables.horizons,
        min_cleared=settings.receivables.min_cleared_for_delay,
        leak_check_weeks=settings.receivables.leak_check_weeks,
        age_edges=settings.receivables.overdue_age_days)

    # Step 9: the panel (every key known by forecast week t, once per week ahead h) and its
    # labels; placeholder customers of unidentified cash get no rows, their cash goes to G.
    horizons = settings.receivables.horizons
    unidentified_customers = settings.companies.unidentified_customers
    known_all = known_weeks(collections, ledger, week_zero)
    known = known_all[~is_unidentified(known_all, unidentified_customers)]
    unidentified_weekly = weekly[is_unidentified(weekly, unidentified_customers)]
    identified_weekly = weekly[~is_unidentified(weekly, unidentified_customers)]
    panel = panel_rows(known, last_week, horizons)
    panel = add_labels(panel, identified_weekly, weeks)
    # The side tables, the cash the panel does not hold: A (keys not yet known at t),
    # R (refunds), G (no identifiable customer); then the tie-out.
    new_cash = new_customer_cash(identified_weekly, known, weeks, horizons, companies)
    refunds = refunds_by_week(weekly, weeks, companies)
    unidentified = unidentified_cash_by_week(unidentified_weekly, weeks, companies)
    checked = check_side_tables(panel, new_cash, unidentified, refunds, weekly, companies)

    # Step 10: the published TÜFE levels by week, 1.0 = week 0, from the first ledger line
    # (history goes back that far) or week 0 if the ledger starts later.
    first_ledger_week = int(week_numbers(pd.Series([ledger["posting_date"].min()]),
                                         week_zero).iloc[0])
    first_price_week = min(first_ledger_week, 0)
    levels = published_levels(week_zero, first_price_week, last_week)

    # Step 11: the features. The ledger's payments before the first daily file are payment
    # history if [history] use_ledger_before_first_file (from week 0 at the latest).
    early_receipts = ledger_receipts_before(ledger, min(daily_files), week_zero)
    if settings.history.use_ledger_before_first_file:
        history_start = min(first_history_week(ledger, week_zero), 0)
        early_receipts = early_receipts[early_receipts["week"] >= history_start]
    else:
        history_start = 0
        early_receipts = early_receipts.iloc[0:0]
    matrix = weekly_matrix(identified_weekly, early_receipts, known["key"], weeks, levels,
                           history_start, last_week)
    history = history_features(matrix, known, last_week, settings.history.windows)
    calendar = calendar_features(week_zero, 0, last_week + horizons, holidays)
    positions = holiday_position_features(
        week_zero, 0, last_week + horizons, holidays, settings.calendar.holiday_encodings,
        settings.calendar.drain_window, settings.calendar.catch_up_window)
    calendar = calendar.merge(positions, on="week", how="left", validate="one_to_one")
    owed = receivables_features(panel, receivables, levels, horizons,
                                settings.receivables.overdue_age_days)
    # The label in TRY of week 0, like the features; then one row per panel row with every
    # feature block.
    panel = add_deflated_label(panel, levels)
    table = join_features(panel, history, calendar, owed)

    inputs = ModelInputs(
        companies=companies, week_zero=week_zero, last_week=last_week, weeks=weeks,
        weekly=weekly, ledger=ledger, rates=rates, levels=levels, calendar=calendar,
        table=table, block_columns=block_feature_columns(history, calendar, owed),
        new_cash=new_cash, refunds=refunds, unidentified=unidentified)
    if not with_details:
        return inputs, None
    details = PreparationDetails(
        daily_files=daily_files, collections_keyed=collections_keyed, tie=tie,
        filled_days=filled_days, collections=collections, holidays=holidays,
        receivables=receivables, known=known_all, identified_weekly=identified_weekly,
        panel=panel, checked=checked, first_price_week=first_price_week,
        history_start=history_start, early_receipts=early_receipts, matrix=matrix,
        history=history, positions=positions, owed=owed)
    return inputs, details
