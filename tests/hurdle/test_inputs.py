"""Behaviour tests of `hurdle/inputs.py`: `prepare_inputs`, `ModelInputs`, `PreparationDetails`.

End to end on a tiny hand-built corpus written to a temporary folder (no monkeypatching of
the step functions): eight weeks of daily workbooks from Monday 2025-06-16 (week 0), one
day without a file inside the range (2025-07-02, Wednesday of week 2) whose payments are in
the ledger, the 15 July holiday (no file needed), and an FBL5N ledger that ties to the daily
files to the kurus and starts six weeks before them (2025-05-05, week -6).

Customers: TR02-1 and TR04-3 are real customers; TR02-99999 is the placeholder customer of
unidentified cash ([companies] unidentified_customers). Nothing reads the real corpus or
writes data/interim: the caches go to the temporary folder.
"""
import dataclasses
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from collection_estimation.hurdle import inputs as inputs_module
from collection_estimation.hurdle.features_calendar import (
    calendar_features,
    holiday_position_features,
)
from collection_estimation.hurdle.features_history import (
    first_history_week,
    history_features,
    ledger_receipts_before,
    weekly_matrix,
)
from collection_estimation.hurdle.features_receivables import receivables_features
from collection_estimation.hurdle.fill_from_ledger import fill_missing_days
from collection_estimation.hurdle.holidays import load_holidays
from collection_estimation.hurdle.inputs import ModelInputs, PreparationDetails, prepare_inputs
from collection_estimation.hurdle.keys import add_keys
from collection_estimation.hurdle.known import known_weeks
from collection_estimation.hurdle.panel import add_deflated_label, add_labels, panel_rows
from collection_estimation.hurdle.prices import published_levels
from collection_estimation.hurdle.read_collections import list_daily_files, load_collections
from collection_estimation.hurdle.read_ledger import (
    check_ledger_against_collections,
    load_ledger,
)
from collection_estimation.hurdle.receivables_asof import exchange_rates, receivables_by_week
from collection_estimation.hurdle.series import (
    add_week_index,
    week_calendar,
    week_numbers,
    weekly_amounts,
)
from collection_estimation.hurdle.settings import (
    CalendarSettings,
    CompanySettings,
    DataSettings,
    HistorySettings,
    ReceivablesSettings,
)
from collection_estimation.hurdle.side_tables import (
    check_side_tables,
    is_unidentified,
    new_customer_cash,
    refunds_by_week,
    unidentified_cash_by_week,
)
from collection_estimation.hurdle.training_table import block_feature_columns, join_features

from .workbooks import (
    SYNTHETIC_LAYOUT,
    SYNTHETIC_LEDGER_LAYOUT,
    invoice,
    line,
    make_ledger_workbook,
    make_workbook,
    payment,
    unapplied_payment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

FIRST_DAY = date(2025, 6, 16)          # a Monday: week 0
WEEKS = 8                              # weeks 0-7
HOLIDAY = date(2025, 7, 15)            # Democracy and National Unity Day (a Tuesday, week 4)
MISSING_DAY = date(2025, 7, 2)         # no daily file; its payments are in the ledger
LEDGER_AS_OF = date(2025, 8, 15)
EARLY_PAYMENT = 1000.0                 # TR02-1's ledger payment of 2025-05-20 (week -4)
PLACEHOLDER = "99999"
COMPANIES = ("TR02", "TR04")
MODEL_FIELDS = ["companies", "week_zero", "last_week", "weeks", "weekly", "ledger", "rates",
                "levels", "calendar", "table", "block_columns", "new_cash", "refunds",
                "unidentified"]
DETAIL_FIELDS = ["daily_files", "collections_keyed", "tie", "filled_days", "collections",
                 "holidays", "receivables", "known", "identified_weekly", "panel", "checked",
                 "first_price_week", "history_start", "early_receipts", "matrix", "history",
                 "positions", "owed"]


def business_days():
    days = [FIRST_DAY + timedelta(days=d) for d in range(7 * WEEKS)]
    return [day for day in days if day.weekday() < 5 and day != HOLIDAY]


def day_payments(index):
    """The payments of the index-th business day: (company, customer, TRY)."""
    payments = []
    if index % 3 == 0:
        payments.append(("TR02", "0000001", 200.0))
    if index % 4 == 1:
        payments.append(("TR04", "0000003", 150.0))
    if index % 5 == 2:
        payments.append(("TR02", PLACEHOLDER, 75.0))
    return payments


def build_corpus(root, *, late_ledger=False, broken_tie=False, stray_company=False):
    """Write the daily workbooks, the ledger and the holidays file under `root`.

    late_ledger: the ledger starts in week 1 (no lines before; week 0 files are empty).
    broken_tie: one ledger payment on a day with a file is 1 TRY off.
    stray_company: one daily line has a company code the settings do not know.
    Returns (corpus folder, ledger file, holidays file, {day: [payments]}).
    """
    root = Path(root)
    corpus = root / "corpus"
    corpus.mkdir()
    if late_ledger:
        ledger = [
            invoice("9000000001", 5000.0, customer="0000001", posted=datetime(2025, 6, 24),
                    due=datetime(2025, 7, 24)),
            invoice("9000000002", 3000.0, company="TR04", customer="0000003",
                    posted=datetime(2025, 6, 24), due=datetime(2025, 7, 24)),
        ]
    else:
        ledger = [
            invoice("9000000001", 5000.0, customer="0000001", posted=datetime(2025, 5, 5),
                    due=datetime(2025, 6, 4)),
            payment("1400000001", EARLY_PAYMENT, invoice_no="9000000001", customer="0000001",
                    posted=datetime(2025, 5, 20)),
            invoice("9000000002", 3000.0, company="TR04", customer="0000003",
                    posted=datetime(2025, 6, 2), due=datetime(2025, 7, 2)),
        ]
    document = 1400000100
    by_day = {}
    broken = False
    for index, day in enumerate(business_days()):
        payments = day_payments(index)
        if late_ledger and day < date(2025, 6, 25):
            payments = []    # nothing before the ledger's first payment day
        by_day[day] = payments
        when = datetime(day.year, day.month, day.day)
        for company, customer, amount in payments:
            document += 1
            ledger_amount = amount
            if broken_tie and not broken and day != MISSING_DAY:
                ledger_amount, broken = amount + 1.0, True
            if customer == PLACEHOLDER:
                ledger.append(unapplied_payment(str(document), ledger_amount, company=company,
                                                customer=customer, posted=when))
            else:
                invoice_no = "9000000001" if company == "TR02" else "9000000002"
                ledger.append(payment(str(document), ledger_amount, invoice_no=invoice_no,
                                      company=company, customer=customer, posted=when))
        if day == MISSING_DAY:
            continue
        rows = [line(customer, amount, company=company, when=when, sap=str(1400000000 + k),
                     invoice=f"GIB2025{k:06d}")
                for k, (company, customer, amount) in enumerate(payments)]
        if stray_company and index == 0:
            rows.append(line("0000005", 10.0, company="TR77", when=when, sap="1400009999",
                             invoice="GIB2025999999"))
        make_workbook(corpus, rows, name=day.strftime("%d%m%Y"))
    ledger_file = make_ledger_workbook(root, ledger)
    holidays_file = root / "holidays.toml"
    shutil.copy(REPO_ROOT / "settings" / "holidays_tr.toml", holidays_file)
    return corpus, ledger_file, holidays_file, by_day


def make_settings(corpus, ledger_file, holidays_file, cache, *, fill=True, history=True,
                  unidentified=(PLACEHOLDER,)):
    """Only the sections `prepare_inputs` reads, as the real settings dataclasses."""
    return SimpleNamespace(
        data=DataSettings(corpus_folder=corpus, ledger_file=ledger_file,
                          ledger_as_of=LEDGER_AS_OF, holidays_file=holidays_file,
                          fill_missing_days_from_ledger=fill, cache_folder=cache),
        companies=CompanySettings(known=COMPANIES, unidentified_customers=unidentified),
        collections_layout=SYNTHETIC_LAYOUT,
        ledger_layout=SYNTHETIC_LEDGER_LAYOUT,
        receivables=ReceivablesSettings(min_cleared_for_delay=1, horizons=2,
                                        leak_check_weeks=(0, 4), overdue_age_days=(30, 90)),
        history=HistorySettings(windows=(2, 4), use_ledger_before_first_file=history),
        calendar=CalendarSettings(holiday_encodings=("drain",), drain_window=3,
                                  catch_up_window=5),
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    folder, ledger_file, holidays_file, by_day = build_corpus(root)
    cache = root / "cache"
    cache.mkdir()
    return SimpleNamespace(folder=folder, ledger_file=ledger_file, holidays_file=holidays_file,
                           cache=cache, by_day=by_day)


@pytest.fixture(scope="module")
def prepared(corpus):
    """prepare_inputs per (fill, history, with_details), computed once each."""
    results = {}

    def get(*, fill=True, history=True, with_details=True):
        key = (fill, history, with_details)
        if key not in results:
            settings = make_settings(corpus.folder, corpus.ledger_file, corpus.holidays_file,
                                     corpus.cache, fill=fill, history=history)
            results[key] = prepare_inputs(settings, with_details=with_details)
        return results[key]
    return get


def settings_for(corpus, **changes):
    return make_settings(corpus.folder, corpus.ledger_file, corpus.holidays_file, corpus.cache,
                         **changes)


def assert_same(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b)
    else:
        assert a == b


def direct_chain(settings):
    """Steps 1-11 called one by one, straight from the step functions (the reference)."""
    layout = settings.collections_layout
    companies = list(settings.companies.known)
    daily_files = list_daily_files(settings.data.corpus_folder, layout.file_name_date_format)
    collections = add_keys(load_collections(daily_files, layout, settings.data.cache_folder),
                           companies)
    ledger = add_keys(load_ledger(settings.data.ledger_file, settings.data.ledger_as_of,
                                  settings.ledger_layout, settings.data.cache_folder), companies)
    check_ledger_against_collections(ledger, collections, list(daily_files))
    if settings.data.fill_missing_days_from_ledger:
        collections, filled = fill_missing_days(collections, ledger, list(daily_files))
    else:
        collections, filled = collections.assign(filled_from_ledger=False), []
    first = min(daily_files)
    week_zero = first - timedelta(days=first.weekday())
    collections = add_week_index(collections, week_zero)
    holidays = load_holidays(settings.data.holidays_file)
    weeks = week_calendar(list(daily_files), week_zero, holidays, filled)
    last_week = int(weeks["week"].max())
    weekly = weekly_amounts(collections)
    rates = exchange_rates(collections, ledger)
    r = settings.receivables
    receivables = receivables_by_week(ledger, rates, week_zero, last_week=last_week,
                                      as_of=settings.data.ledger_as_of, horizons=r.horizons,
                                      min_cleared=r.min_cleared_for_delay,
                                      leak_check_weeks=r.leak_check_weeks,
                                      age_edges=r.overdue_age_days)
    placeholders = settings.companies.unidentified_customers
    known_all = known_weeks(collections, ledger, week_zero)
    known = known_all[~is_unidentified(known_all, placeholders)]
    identified = weekly[~is_unidentified(weekly, placeholders)]
    panel = add_labels(panel_rows(known, last_week, r.horizons), identified, weeks)
    new_cash = new_customer_cash(identified, known, weeks, r.horizons, companies)
    refunds = refunds_by_week(weekly, weeks, companies)
    unidentified = unidentified_cash_by_week(weekly[is_unidentified(weekly, placeholders)],
                                             weeks, companies)
    check_side_tables(panel, new_cash, unidentified, refunds, weekly, companies)
    first_ledger = int(week_numbers(pd.Series([ledger["posting_date"].min()]),
                                    week_zero).iloc[0])
    levels = published_levels(week_zero, min(first_ledger, 0), last_week)
    early = ledger_receipts_before(ledger, first, week_zero)
    if settings.history.use_ledger_before_first_file:
        start = min(first_history_week(ledger, week_zero), 0)
        early = early[early["week"] >= start]
    else:
        start, early = 0, early.iloc[0:0]
    matrix = weekly_matrix(identified, early, known["key"], weeks, levels, start, last_week)
    history = history_features(matrix, known, last_week, settings.history.windows)
    calendar = calendar_features(week_zero, 0, last_week + r.horizons, holidays)
    c = settings.calendar
    positions = holiday_position_features(week_zero, 0, last_week + r.horizons, holidays,
                                          c.holiday_encodings, c.drain_window,
                                          c.catch_up_window)
    calendar = calendar.merge(positions, on="week", how="left", validate="one_to_one")
    owed = receivables_features(panel, receivables, levels, r.horizons, r.overdue_age_days)
    table = join_features(add_deflated_label(panel, levels), history, calendar, owed)
    return dict(companies=companies, week_zero=week_zero, last_week=last_week, weeks=weeks,
                weekly=weekly, ledger=ledger, rates=rates, levels=levels, calendar=calendar,
                table=table, block_columns=block_feature_columns(history, calendar, owed),
                new_cash=new_cash, refunds=refunds, unidentified=unidentified)


@pytest.fixture(scope="module")
def direct(corpus):
    """direct_chain per (fill, history), computed once each."""
    results = {}

    def get(*, fill, history):
        if (fill, history) not in results:
            results[(fill, history)] = direct_chain(settings_for(corpus, fill=fill,
                                                                 history=history))
        return results[(fill, history)]
    return get


# --- the return value -------------------------------------------------------------------

def test_default_returns_inputs_and_none(prepared):
    inputs, details = prepared(with_details=False)
    assert isinstance(inputs, ModelInputs)
    assert details is None


def test_with_details_returns_preparation_details(prepared):
    inputs, details = prepared(with_details=True)
    assert isinstance(inputs, ModelInputs)
    assert isinstance(details, PreparationDetails)


def test_returns_a_two_tuple(corpus):
    result = prepare_inputs(settings_for(corpus))
    assert isinstance(result, tuple) and len(result) == 2


@pytest.mark.parametrize("field", MODEL_FIELDS)
def test_inputs_are_the_same_with_or_without_details(prepared, field):
    without, _ = prepared(with_details=False)
    with_, _ = prepared(with_details=True)
    assert_same(getattr(without, field), getattr(with_, field))


@pytest.mark.parametrize("field", MODEL_FIELDS)
def test_a_second_call_gives_the_same_inputs(corpus, prepared, field):
    first, _ = prepared(with_details=True)
    again, _ = prepare_inputs(settings_for(corpus), with_details=True)
    assert_same(getattr(first, field), getattr(again, field))


def test_model_inputs_fields_in_order():
    assert [f.name for f in dataclasses.fields(ModelInputs)] == MODEL_FIELDS


def test_preparation_details_fields_in_order():
    assert [f.name for f in dataclasses.fields(PreparationDetails)] == DETAIL_FIELDS


def test_model_inputs_is_frozen(prepared):
    inputs, _ = prepared()
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.last_week = 99
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.table = pd.DataFrame()


def test_preparation_details_is_frozen(prepared):
    _, details = prepared()
    with pytest.raises(dataclasses.FrozenInstanceError):
        details.checked = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        details.filled_days = []


def test_prints_no_step_lines_of_its_own(corpus, capsys):
    prepare_inputs(settings_for(corpus), with_details=True)
    printed = capsys.readouterr().out
    assert "Step " not in printed
    assert "tie-out passed" not in printed


# --- the fields against the step functions called directly -------------------------------

@pytest.mark.parametrize("fill,history", [(True, True), (False, True), (True, False),
                                          (False, False)])
@pytest.mark.parametrize("field", MODEL_FIELDS)
def test_each_field_equals_the_direct_calls(direct, prepared, fill, history, field):
    inputs, _ = prepared(fill=fill, history=history)
    assert_same(getattr(inputs, field), direct(fill=fill, history=history)[field])


# --- facts about the tiny corpus ---------------------------------------------------------

def test_companies_is_a_list_in_settings_order(prepared):
    inputs, _ = prepared()
    assert inputs.companies == list(COMPANIES)
    assert isinstance(inputs.companies, list)


def test_week_zero_is_the_monday_of_the_first_file(prepared):
    inputs, _ = prepared()
    assert inputs.week_zero == FIRST_DAY


def test_last_week_is_the_last_calendar_week(prepared):
    inputs, _ = prepared()
    assert inputs.last_week == WEEKS - 1
    assert inputs.last_week == int(inputs.weeks["week"].max())
    assert isinstance(inputs.last_week, int)


def test_levels_are_one_at_week_zero(prepared):
    inputs, _ = prepared()
    assert inputs.levels.loc[0] == pytest.approx(1.0)


def test_calendar_covers_week_zero_to_last_week_plus_horizons(prepared):
    inputs, _ = prepared()
    assert list(inputs.calendar["week"]) == list(range(0, WEEKS - 1 + 2 + 1))


def test_calendar_holds_the_holiday_position_columns(prepared):
    inputs, details = prepared()
    for column in details.positions.columns:
        assert column in inputs.calendar.columns
    assert "drain_days" in inputs.calendar.columns


def test_block_columns_has_the_three_blocks(prepared):
    inputs, _ = prepared()
    assert set(inputs.block_columns) == {"history", "calendar", "receivables"}
    for columns in inputs.block_columns.values():
        assert set(columns) <= set(inputs.table.columns)


def test_block_columns_calendar_includes_the_holiday_positions(prepared):
    inputs, _ = prepared()
    assert "drain_days" in inputs.block_columns["calendar"]


def test_table_has_the_deflated_label(prepared):
    inputs, details = prepared()
    assert "amount_deflated" in inputs.table.columns
    assert "amount_deflated" in details.panel.columns


def test_table_has_one_row_per_panel_row(prepared):
    inputs, details = prepared()
    assert len(inputs.table) == len(details.panel)


# --- details ------------------------------------------------------------------------------

def test_details_daily_files_are_the_listed_files(corpus, prepared):
    _, details = prepared()
    assert details.daily_files == list_daily_files(corpus.folder, "%d%m%Y")
    assert MISSING_DAY not in details.daily_files
    assert len(details.daily_files) == len(business_days()) - 1


def test_details_tie_compares_every_day_with_a_file(prepared):
    _, details = prepared()
    assert details.tie["days_compared"] == len(details.daily_files)
    assert details.tie["missing_file_days_in_ledger"] == 1


def test_details_collections_keyed_is_before_fill_and_week_index(prepared):
    _, details = prepared()
    assert "key" in details.collections_keyed.columns
    assert "week" not in details.collections_keyed.columns
    assert "filled_from_ledger" not in details.collections_keyed.columns
    assert "week" in details.collections.columns


def test_details_known_keeps_the_placeholders(prepared):
    _, details = prepared()
    assert "TR02-99999" in set(details.known["key"])


def test_details_identified_weekly_drops_the_placeholders(prepared):
    inputs, details = prepared()
    assert "TR02-99999" not in set(details.identified_weekly["key"])
    assert "TR02-99999" in set(inputs.weekly["key"])


def test_details_checked_counts_tie_out_checks(prepared):
    _, details = prepared()
    assert isinstance(details.checked, int) and details.checked > 0


def test_details_first_price_week_is_the_first_ledger_week(prepared):
    inputs, details = prepared()
    assert details.first_price_week == -6    # 2025-05-05 is the Monday of week -6
    assert inputs.levels.index.min() == -6


# --- [data] fill_missing_days_from_ledger --------------------------------------------------

def test_fill_on_fills_the_missing_day(prepared):
    _, details = prepared(fill=True)
    assert details.filled_days == [MISSING_DAY]
    filled = details.collections[details.collections["filled_from_ledger"]]
    assert set(filled["document_date"].dt.date) == {MISSING_DAY}


def test_fill_on_week_with_the_missing_day_is_complete(prepared):
    inputs, _ = prepared(fill=True)
    assert not inputs.weeks.set_index("week").loc[2, "is_missing"]


def test_fill_on_weekly_holds_the_filled_cash(corpus, prepared):
    inputs, _ = prepared(fill=True)
    expected = sum(amount for day, payments in corpus.by_day.items()
                   if FIRST_DAY + timedelta(weeks=2) <= day < FIRST_DAY + timedelta(weeks=3)
                   for _, _, amount in payments)
    week2 = inputs.weekly[inputs.weekly["week"] == 2]
    assert week2["gross_receipts"].sum() == pytest.approx(expected)


def test_fill_off_leaves_no_filled_days(prepared):
    _, details = prepared(fill=False)
    assert details.filled_days == []
    assert "filled_from_ledger" in details.collections.columns
    assert not details.collections["filled_from_ledger"].any()


def test_fill_off_week_with_the_missing_day_is_missing(prepared):
    inputs, _ = prepared(fill=False)
    calendar = inputs.weeks.set_index("week")
    assert calendar.loc[2, "is_missing"]
    assert not calendar.drop(index=2)["is_missing"].any()


def test_fill_off_labels_of_the_missing_week_are_unknown(prepared):
    inputs, _ = prepared(fill=False)
    target = inputs.table["origin_week"] + inputs.table["h"]
    rows = inputs.table[target == 2]
    assert len(rows) > 0
    assert (rows["label_status"] != "labelled").all()


def test_fill_changes_the_table(prepared):
    on, _ = prepared(fill=True)
    off, _ = prepared(fill=False)
    assert not on.table["label_status"].equals(off.table["label_status"])


# --- [history] use_ledger_before_first_file ------------------------------------------------

def test_history_on_starts_at_the_ledgers_first_full_week(prepared):
    _, details = prepared(history=True)
    assert details.history_start == -6


def test_history_on_early_receipts_hold_the_ledger_payment(prepared):
    _, details = prepared(history=True)
    assert len(details.early_receipts) > 0
    assert (details.early_receipts["week"] >= details.history_start).all()
    assert details.early_receipts["gross_receipts"].sum() == pytest.approx(EARLY_PAYMENT)
    assert set(details.early_receipts["week"]) == {-4}    # Tuesday 2025-05-20


def test_history_on_matrix_starts_at_history_start(prepared):
    _, details = prepared(history=True)
    assert min(details.matrix.columns) == -6
    assert max(details.matrix.columns) == WEEKS - 1


def test_history_off_starts_at_week_zero(prepared):
    _, details = prepared(history=False)
    assert details.history_start == 0
    assert details.early_receipts.empty
    assert min(details.matrix.columns) == 0


def test_history_off_keeps_the_early_receipts_columns(prepared):
    _, on = prepared(history=True)
    _, off = prepared(history=False)
    assert list(off.early_receipts.columns) == list(on.early_receipts.columns)


def test_history_off_does_not_move_the_price_range(prepared):
    on, _ = prepared(history=True)
    off, _ = prepared(history=False)
    pd.testing.assert_series_equal(on.levels, off.levels)


def test_history_switch_changes_the_history_features(prepared):
    on, _ = prepared(history=True)
    off, _ = prepared(history=False)
    columns = on.block_columns["history"]
    assert not on.table[columns].equals(off.table[columns])


# --- placeholder customers ---------------------------------------------------------------

def test_placeholders_get_no_table_rows(prepared):
    inputs, details = prepared()
    assert "TR02-99999" not in set(inputs.table["key"])
    assert "TR02-99999" not in set(details.panel["key"])
    assert set(inputs.table["key"]) == {"TR02-1", "TR04-3"}


def test_placeholder_cash_is_in_unidentified(corpus, prepared):
    inputs, _ = prepared()
    placeholder_cash = sum(amount for payments in corpus.by_day.values()
                           for _, customer, amount in payments if customer == PLACEHOLDER)
    by_company = inputs.unidentified.groupby("company_code")["unidentified_cash"].sum()
    assert by_company["TR02"] == pytest.approx(placeholder_cash)
    assert by_company["TR04"] == pytest.approx(0.0)


def test_unidentified_has_a_row_per_company_and_week(prepared):
    inputs, _ = prepared()
    assert set(inputs.unidentified["company_code"]) == set(COMPANIES)
    assert len(inputs.unidentified) == len(COMPANIES) * WEEKS


def test_without_placeholders_the_placeholder_is_a_customer(corpus):
    settings = settings_for(corpus, unidentified=())
    inputs, _ = prepare_inputs(settings)
    assert "TR02-99999" in set(inputs.table["key"])
    assert inputs.unidentified["unidentified_cash"].sum() == pytest.approx(0.0)


# --- checks that stop a run ----------------------------------------------------------------

def test_ledger_not_tying_to_the_daily_files_raises(tmp_path):
    folder, ledger_file, holidays_file, _ = build_corpus(tmp_path, broken_tie=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="differ from the daily files"):
        prepare_inputs(make_settings(folder, ledger_file, holidays_file, cache))


def test_ledger_not_tying_raises_with_details_too(tmp_path):
    folder, ledger_file, holidays_file, _ = build_corpus(tmp_path, broken_tie=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="differ from the daily files"):
        prepare_inputs(make_settings(folder, ledger_file, holidays_file, cache),
                       with_details=True)


def test_unknown_company_code_raises(tmp_path):
    folder, ledger_file, holidays_file, _ = build_corpus(tmp_path, stray_company=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError):
        prepare_inputs(make_settings(folder, ledger_file, holidays_file, cache))


# --- a ledger starting after week 0 --------------------------------------------------------

@pytest.fixture(scope="module")
def late(tmp_path_factory):
    root = tmp_path_factory.mktemp("late")
    folder, ledger_file, holidays_file, _ = build_corpus(root, late_ledger=True)
    cache = root / "cache"
    cache.mkdir()
    return prepare_inputs(make_settings(folder, ledger_file, holidays_file, cache),
                          with_details=True)


def test_late_ledger_price_range_starts_at_week_zero(late):
    inputs, details = late
    assert details.first_price_week == 0
    assert inputs.levels.index.min() == 0


def test_late_ledger_history_starts_at_week_zero(late):
    _, details = late
    assert details.history_start == 0
    assert details.early_receipts.empty
    assert min(details.matrix.columns) == 0


# --- the module's imports ------------------------------------------------------------------

def test_module_exposes_prepare_inputs_and_the_dataclasses():
    assert inputs_module.prepare_inputs is prepare_inputs
    assert dataclasses.is_dataclass(inputs_module.ModelInputs)
    assert dataclasses.is_dataclass(inputs_module.PreparationDetails)
