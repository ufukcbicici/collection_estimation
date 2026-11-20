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
from collection_estimation.hurdle.features_rhythm import (
    EXPECTED_DUE_COLUMNS,
    HABIT_COLUMNS,
    expected_dues,
    payment_habits,
)
from collection_estimation.hurdle.features_series import (
    SERIES_COLUMNS,
    SERIES_KEYS,
    series_features,
)
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
    RhythmSettings,
    SeriesSettings,
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
                 "positions", "owed", "dues", "habits", "series"]
RHYTHM = dict(delays_kept=5, company_delays_kept=20, payment_days_kept=10, min_payment_days=3)
MEAN_WEEKS = 3                         # [series] mean_weeks
BASE_BLOCKS = ("history", "calendar", "receivables")
JOIN_KEYS = ["key", "origin_week", "h"]


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
                  unidentified=(PLACEHOLDER,), rhythm=None, mean_weeks=MEAN_WEEKS):
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
        rhythm=RhythmSettings(**{**RHYTHM, **(rhythm or {})}),
        series=SeriesSettings(mean_weeks=mean_weeks),
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
    rhythm = settings.rhythm
    dues = expected_dues(panel[["key", "company_code", "origin_week", "h"]], ledger, rates,
                         levels, week_zero, holidays, r.horizons, r.min_cleared_for_delay,
                         rhythm.delays_kept, rhythm.company_delays_kept, r.leak_check_weeks)
    habits = payment_habits(panel[JOIN_KEYS], collections, ledger, week_zero, holidays, first,
                            settings.history.use_ledger_before_first_file, r.horizons,
                            rhythm.payment_days_kept, rhythm.min_payment_days,
                            r.leak_check_weeks)
    series = series_features(weekly, weeks, levels, calendar, companies,
                             sorted(set(panel["origin_week"])), r.horizons,
                             settings.series.mean_weeks)
    table = join_features(add_deflated_label(panel, levels), history, calendar, owed, dues,
                          habits, series)
    return dict(companies=companies, week_zero=week_zero, last_week=last_week, weeks=weeks,
                weekly=weekly, ledger=ledger, rates=rates, levels=levels, calendar=calendar,
                table=table,
                block_columns=block_feature_columns(history, calendar, owed, dues, habits,
                                                    series),
                new_cash=new_cash, refunds=refunds, unidentified=unidentified,
                dues=dues, habits=habits, series=series)


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


def test_block_columns_has_the_six_blocks_in_feature_blocks_order(prepared):
    from collection_estimation.hurdle.settings import FEATURE_BLOCKS
    inputs, _ = prepared()
    assert tuple(inputs.block_columns) == FEATURE_BLOCKS
    for columns in inputs.block_columns.values():
        assert set(columns) <= set(inputs.table.columns)


# --- U18.4: the rhythm blocks (step 11b) ---------------------------------------------------

def test_block_columns_dues_and_habits_are_the_documented_columns(prepared):
    inputs, _ = prepared()
    assert inputs.block_columns["dues"] == EXPECTED_DUE_COLUMNS
    assert inputs.block_columns["habits"] == HABIT_COLUMNS


def test_details_dues_and_habits_have_one_row_per_panel_row_in_its_order(prepared):
    _, details = prepared()
    for block in (details.dues, details.habits):
        pd.testing.assert_frame_equal(block[JOIN_KEYS],
                                      details.panel[JOIN_KEYS].reset_index(drop=True))


@pytest.mark.parametrize("name", ["dues", "habits"])
def test_table_rhythm_columns_equal_the_details_blocks(prepared, name):
    inputs, details = prepared()
    block = getattr(details, name)
    columns = inputs.block_columns[name]
    pd.testing.assert_frame_equal(inputs.table[JOIN_KEYS + columns].reset_index(drop=True),
                                  block[JOIN_KEYS + columns])


@pytest.mark.parametrize("fill,history", [(True, True), (False, True), (True, False),
                                          (False, False)])
@pytest.mark.parametrize("name", ["dues", "habits"])
def test_details_rhythm_blocks_equal_the_direct_calls(direct, prepared, fill, history, name):
    _, details = prepared(fill=fill, history=history)
    pd.testing.assert_frame_equal(getattr(details, name),
                                  direct(fill=fill, history=history)[name])


def test_base_blocks_feature_columns_take_no_rhythm_column(prepared):
    from collection_estimation.hurdle.stage1 import feature_columns
    inputs, _ = prepared()
    base = feature_columns(inputs.block_columns, BASE_BLOCKS)
    rhythm = set(inputs.block_columns["dues"]) | set(inputs.block_columns["habits"])
    assert not set(base) & rhythm
    expected = ["h"] + [c for block in BASE_BLOCKS for c in inputs.block_columns[block]]
    assert base == expected


def test_habits_hold_nan_below_min_payment_days_and_the_table_keeps_them(prepared):
    inputs, details = prepared()
    few = details.habits["payment_days_used"] < RHYTHM["min_payment_days"]
    assert few.any(), "no row below min_payment_days in the tiny corpus"
    others = [c for c in HABIT_COLUMNS if c != "payment_days_used"]
    assert details.habits.loc[few, others].isna().all().all()
    assert len(inputs.table) == len(details.panel)


def test_a_min_payment_days_above_every_customer_makes_every_habit_nan(corpus):
    inputs, details = prepare_inputs(settings_for(corpus, rhythm={"min_payment_days": 1000}),
                                     with_details=True)
    others = [c for c in HABIT_COLUMNS if c != "payment_days_used"]
    assert details.habits[others].isna().all().all()
    assert inputs.table[others].isna().all().all()
    assert details.habits["payment_days_used"].notna().all()


def test_payment_days_kept_caps_payment_days_used(corpus):
    _, details = prepare_inputs(settings_for(corpus, rhythm={"payment_days_kept": 2,
                                                             "min_payment_days": 2}),
                                with_details=True)
    assert details.habits["payment_days_used"].max() == 2


def test_a_min_payment_days_below_two_is_refused_by_payment_habits(corpus):
    with pytest.raises(ValueError, match="min_payment_days is 1; it must be 2 or more"):
        prepare_inputs(settings_for(corpus, rhythm={"min_payment_days": 1}))


def spy_on(monkeypatch, name):
    """Record each call of inputs.<name> (args, kwargs) and still run the real function."""
    calls = []
    real = getattr(inputs_module, name)

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        result = real(*args, **kwargs)
        calls[-1] = (args, kwargs, result)
        return result
    monkeypatch.setattr(inputs_module, name, spy)
    return calls


@pytest.fixture
def spied(corpus, monkeypatch):
    """prepare_inputs on settings with unusual [rhythm] values, with expected_dues,
    payment_habits, join_features and block_feature_columns spied on."""
    rhythm = {"delays_kept": 7, "company_delays_kept": 13, "payment_days_kept": 9,
              "min_payment_days": 4}
    calls = {name: spy_on(monkeypatch, name) for name in
             ("expected_dues", "payment_habits", "join_features", "block_feature_columns")}
    settings = settings_for(corpus, rhythm=rhythm, history=False)
    inputs, details = prepare_inputs(settings, with_details=True)
    return SimpleNamespace(calls=calls, settings=settings, inputs=inputs, details=details)


def test_expected_dues_is_called_once_with_the_settings(spied):
    assert len(spied.calls["expected_dues"]) == 1
    args, kwargs, result = spied.calls["expected_dues"][0]
    s = spied.settings
    assert kwargs == {"horizons": s.receivables.horizons,
                      "min_cleared": s.receivables.min_cleared_for_delay,
                      "delays_kept": 7, "company_delays_kept": 13,
                      "leak_check_weeks": s.receivables.leak_check_weeks}
    panel_rows, ledger, rates, levels, week_zero, holidays = args
    assert list(panel_rows.columns) == ["key", "company_code", "origin_week", "h"]
    assert len(panel_rows) == len(spied.details.panel)
    pd.testing.assert_frame_equal(ledger, spied.inputs.ledger)
    pd.testing.assert_frame_equal(rates, spied.inputs.rates)
    pd.testing.assert_series_equal(levels, spied.inputs.levels)
    assert week_zero == spied.inputs.week_zero
    assert holidays == spied.details.holidays
    assert result is spied.details.dues


def test_payment_habits_is_called_once_with_the_settings(spied):
    assert len(spied.calls["payment_habits"]) == 1
    args, kwargs, result = spied.calls["payment_habits"][0]
    s = spied.settings
    assert kwargs == {"use_ledger_before_first_file": False,
                      "horizons": s.receivables.horizons, "payment_days_kept": 9,
                      "min_payment_days": 4,
                      "leak_check_weeks": s.receivables.leak_check_weeks}
    panel_rows, collections, ledger, week_zero, holidays, first_file_day = args
    assert list(panel_rows.columns) == JOIN_KEYS
    assert len(panel_rows) == len(spied.details.panel)
    pd.testing.assert_frame_equal(collections, spied.details.collections)
    pd.testing.assert_frame_equal(ledger, spied.inputs.ledger)
    assert week_zero == spied.inputs.week_zero
    assert holidays == spied.details.holidays
    assert first_file_day == min(spied.details.daily_files)
    assert result is spied.details.habits


def test_the_history_switch_reaches_payment_habits(corpus, monkeypatch):
    calls = spy_on(monkeypatch, "payment_habits")
    prepare_inputs(settings_for(corpus, history=True))
    assert calls[0][1]["use_ledger_before_first_file"] is True


def test_join_features_and_block_feature_columns_get_the_rhythm_blocks(spied):
    (join_args, join_kwargs, table), = spied.calls["join_features"]
    (block_args, block_kwargs, columns), = spied.calls["block_feature_columns"]
    assert join_kwargs == {} and block_kwargs == {}
    # U19.2: the series block is the last argument of both.
    assert len(join_args) == 7 and len(block_args) == 6
    assert join_args[4] is spied.details.dues and join_args[5] is spied.details.habits
    assert block_args[3] is spied.details.dues and block_args[4] is spied.details.habits
    assert join_args[6] is spied.details.series and block_args[5] is spied.details.series
    assert all(a is b for a, b in zip(join_args[1:4], block_args[:3]))
    assert table is spied.inputs.table
    assert columns is spied.inputs.block_columns


def test_the_rhythm_settings_reach_the_features(corpus, prepared):
    _, default = prepared()
    _, changed = prepare_inputs(settings_for(corpus, rhythm={"min_payment_days": 2,
                                                             "payment_days_kept": 2}),
                                with_details=True)
    assert not default.habits.equals(changed.habits)
    pd.testing.assert_frame_equal(default.dues, changed.dues)    # dues do not read these


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


# --- the pipeline's printouts on the details' rhythm blocks (U18.4) ------------------------

def test_show_expected_dues_runs_on_details_dues(prepared, capsys):
    from collection_estimation.hurdle.reports.preparation import show_expected_dues
    inputs, details = prepared()
    show_expected_dues(details.dues, inputs.table, (2, 3), 2)
    printed = capsys.readouterr().out
    assert printed.strip(), "nothing printed"
    assert f"{len(details.dues):,}" in printed


def test_show_payment_habits_runs_on_details_habits(prepared, capsys):
    from collection_estimation.hurdle.reports.preparation import show_payment_habits
    inputs, details = prepared()
    show_payment_habits(details.habits, inputs.table, (2, 3))
    printed = capsys.readouterr().out
    assert printed.strip(), "nothing printed"
    assert f"{len(details.habits):,}" in printed


# --- U19.2: the series block (step 11c) ---------------------------------------------------
# The tiny corpus has 8 weeks, fewer than the deflated_trend reference needs to fit
# (`benchmark.MIN_TRAINING_WEEKS`), so every series value here is NaN: these tests check the
# wiring (which arguments, which rows, the leak check stopping a run), not the numbers, which
# test_features_series.py checks.

def test_block_columns_series_are_the_documented_columns(prepared):
    inputs, _ = prepared()
    assert inputs.block_columns["series"] == SERIES_COLUMNS


def test_details_series_has_one_row_per_company_origin_and_h(prepared):
    inputs, details = prepared()
    origins = sorted(set(details.panel["origin_week"]))
    horizons = int(details.panel["h"].max())
    assert len(details.series) == len(COMPANIES) * len(origins) * horizons
    assert sorted(set(details.series["company_code"])) == sorted(COMPANIES)
    assert sorted(set(details.series["origin_week"])) == origins
    assert list(details.series.columns) == SERIES_KEYS + SERIES_COLUMNS


def test_details_series_is_all_nan_on_the_tiny_corpus(prepared):
    # Too few weeks for any trend: NaN, kept as NaN in the table, not a refused row.
    inputs, details = prepared()
    assert details.series[SERIES_COLUMNS].isna().all().all()
    assert inputs.table[SERIES_COLUMNS].isna().all().all()
    assert len(inputs.table) == len(details.panel)


def test_table_series_columns_follow_the_company_origin_and_h(prepared):
    inputs, details = prepared()
    expected = (inputs.table[SERIES_KEYS]
                .merge(details.series, on=SERIES_KEYS, how="left", validate="many_to_one"))
    pd.testing.assert_frame_equal(inputs.table[SERIES_KEYS + SERIES_COLUMNS]
                                  .reset_index(drop=True), expected)


@pytest.mark.parametrize("fill,history", [(True, True), (False, False)])
def test_details_series_equals_the_direct_calls(direct, prepared, fill, history):
    _, details = prepared(fill=fill, history=history)
    pd.testing.assert_frame_equal(details.series, direct(fill=fill, history=history)["series"])


@pytest.fixture
def series_spied(corpus, monkeypatch):
    """prepare_inputs with [series] mean_weeks 6, series_features and check_series_no_leak
    spied on."""
    calls = {name: spy_on(monkeypatch, name)
             for name in ("series_features", "check_series_no_leak")}
    settings = settings_for(corpus, mean_weeks=6)
    inputs, details = prepare_inputs(settings, with_details=True)
    return SimpleNamespace(calls=calls, settings=settings, inputs=inputs, details=details)


def test_series_features_is_called_once_at_every_panel_origin(series_spied):
    (args, kwargs, result), = series_spied.calls["series_features"]
    weekly, weeks, levels, calendar, companies, origins, horizons, mean_weeks = args
    assert kwargs == {}
    assert origins == sorted(set(series_spied.details.panel["origin_week"]))
    assert horizons == series_spied.settings.receivables.horizons
    assert mean_weeks == 6
    assert companies == list(COMPANIES)
    pd.testing.assert_frame_equal(weekly, series_spied.inputs.weekly)
    pd.testing.assert_frame_equal(weeks, series_spied.inputs.weeks)
    pd.testing.assert_series_equal(levels, series_spied.inputs.levels)
    pd.testing.assert_frame_equal(calendar, series_spied.inputs.calendar)
    assert result is series_spied.details.series


def test_check_series_no_leak_gets_the_series_and_the_leak_check_weeks(series_spied):
    (args, kwargs, result), = series_spied.calls["check_series_no_leak"]
    series, weekly, weeks, levels, calendar, companies, horizons, mean_weeks, check_weeks = args
    assert kwargs == {}
    assert result is None
    assert series is series_spied.details.series
    assert check_weeks == series_spied.settings.receivables.leak_check_weeks == (0, 4)
    assert mean_weeks == 6
    assert horizons == series_spied.settings.receivables.horizons
    (made_args, _, _), = series_spied.calls["series_features"]
    assert all(a is b for a, b in zip(args[1:6], made_args[:5]))


def test_the_series_leak_check_stops_the_run_when_the_stored_rows_differ(corpus, monkeypatch):
    # series_features as seen by prepare_inputs gives rows the data cannot give (a ratio of 1
    # where too few weeks give NaN); the check rebuilds them with the real function and stops.
    real = inputs_module.series_features

    def tampered(*args, **kwargs):
        return real(*args, **kwargs).assign(all_trend_ratio=1.0)
    monkeypatch.setattr(inputs_module, "series_features", tampered)
    with pytest.raises(ValueError, match=r"^leak check: the series features of week 0 differ"):
        prepare_inputs(settings_for(corpus))
