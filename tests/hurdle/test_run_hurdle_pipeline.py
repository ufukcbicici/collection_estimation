"""Smoke tests for `run_hurdle_pipeline.py` (the step-by-step script).

Running the real `main()` reads the whole synthetic corpus and writes to data/interim, so it
is not run here. The import checks, and the step-6 branch tests (main() with every reader
stubbed and stopped right after step 6), run in a separate Python process: the script
reconfigures sys.stdout and puts src/ on sys.path, which should not leak into the rest of the
test session.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "run_hurdle_pipeline.py"

CHECK = """
import inspect
import run_hurdle_pipeline as script
assert callable(script.main)
assert script.DEFAULT_SETTINGS_FILE.name == "synthetic.toml"
# The names the script calls must have the signatures it uses after L.2.
inspect.signature(script.list_daily_files).bind("folder", "%d%m%Y")
inspect.signature(script.read_one_file).bind("path", "layout")
inspect.signature(script.load_collections).bind({}, "layout", "cache")
# After L.3: the ledger layout comes from the settings file.
inspect.signature(script.load_ledger).bind("path", "as_of", "layout", "cache")
try:
    inspect.signature(script.load_ledger).bind("path", "as_of", "cache")
except TypeError:
    pass
else:
    raise AssertionError("load_ledger still accepts the old three-argument call")
inspect.signature(script.check_ledger_against_collections).bind("ledger", "collections", [])
# After L.4: the holidays come from the holidays file named in the settings.
inspect.signature(script.load_holidays).bind("path")
inspect.signature(script.week_calendar).bind([], "week_zero", "holidays")
try:
    inspect.signature(script.week_calendar).bind([], "week_zero")
except TypeError:
    pass
else:
    raise AssertionError("week_calendar still accepts the old two-argument call")
# After U9b.5: step 7 builds every week's receivables in one call.
inspect.signature(script.exchange_rates).bind("collections", "ledger")
inspect.signature(script.receivables_by_week).bind(
    "ledger", "rates", "week_zero", last_week=64, as_of="as_of", horizons=5, min_cleared=3,
    leak_check_weeks=(0, 30, 64))
# Step 8: when each key becomes known, from the daily files and the ledger.
inspect.signature(script.known_weeks).bind("collections", "ledger", "week_zero")
# Step 9: the panel's rows, from the known keys, the last week and the horizons.
inspect.signature(script.panel_rows).bind("known", 64, 5)
assert callable(script.show_panel_rows)
# Step 9, U4.2: the labels, from the panel, the weekly amounts and the week calendar.
inspect.signature(script.add_labels).bind("panel", "weekly", "weeks")
assert callable(script.show_labels)
# Step 9, side table A: new-customer cash, from the weekly amounts, the known keys, the
# calendar, the horizons and the companies (required: the grid is built from them).
inspect.signature(script.new_customer_cash).bind("weekly", "known", "weeks", 5, ["TR02"])
try:
    inspect.signature(script.new_customer_cash).bind("weekly", "known", "weeks", 5)
except TypeError:
    pass
else:
    raise AssertionError("new_customer_cash still accepts the old four-argument call")
assert callable(script.show_new_customer_cash)
# Side table R: refunds per company and week, from the weekly amounts, the calendar and
# the companies.
inspect.signature(script.refunds_by_week).bind("weekly", "weeks", ["TR02"])
assert callable(script.show_refunds)
# Side table G: placeholder customers are split off by customer_id, and their gross
# receipts summed per company and week.
inspect.signature(script.is_unidentified).bind("table", ("99999",))
inspect.signature(script.unidentified_cash_by_week).bind("unidentified", "weeks", ["TR02"])
# The tie-out: the panel and the side tables A, G and R against all the weekly amounts.
inspect.signature(script.check_side_tables).bind("panel", "new_cash", "unidentified",
                                                 "refunds", "weekly", ["TR02"])
# Step 6: days without a file filled from the ledger; the calendar takes the filled days.
inspect.signature(script.fill_missing_days).bind("collections", "ledger", [])
try:
    inspect.signature(script.fill_missing_days).bind("collections", "ledger")
except TypeError:
    pass
else:
    raise AssertionError("fill_missing_days accepts a call without the days with a file")
inspect.signature(script.week_calendar).bind([], "week_zero", "holidays", [])
inspect.signature(script.week_calendar).bind([], "week_zero", "holidays", filled_days=[])
assert callable(script.show_filled)
# Step 10: TÜFE price levels by week, from week zero and the first and last week numbers.
inspect.signature(script.price_weeks).bind("week_zero", -32, 64)
inspect.signature(script.published_levels).bind("week_zero", -32, 64)
assert callable(script.show_prices)
# Step 11: the ledger's payments before the first daily file, as payment history.
inspect.signature(script.first_history_week).bind("ledger", "week_zero")
inspect.signature(script.ledger_receipts_before).bind("ledger", "first_file_day", "week_zero")
assert callable(script.show_early_receipts)
# Step 11, U6.1b: the weekly matrix, from the weekly amounts, the early receipts, the keys,
# the calendar, the price levels and the first and last weeks.
inspect.signature(script.weekly_matrix).bind("weekly", "early", "keys", "weeks", "levels",
                                             -31, 64)
try:
    inspect.signature(script.weekly_matrix).bind("weekly", "early", "keys", "weeks", "levels")
except TypeError:
    pass
else:
    raise AssertionError("weekly_matrix accepts a call without the first and last weeks")
assert callable(script.show_matrix)
print("ok")
"""


# Step 6 of main(), with every reader before it replaced by a stub, so that no file is read
# and nothing is written (no corpus, no ledger, no data/interim cache). main() is stopped at
# week_calendar, the first call after step 6 that receives the filled days.
STEP_6 = """
import sys
from datetime import date
from types import SimpleNamespace
import pandas as pd
import run_hurdle_pipeline as script

SWITCH = {switch}
DAYS = [date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 5)]
COLLECTIONS = pd.DataFrame({{"key": ["TR02-1", "TR02-2"], "amount_try": [100.0, 30.0],
                            "document_date": pd.to_datetime(["2026-03-02", "2026-03-03"])}})
LEDGER = pd.DataFrame({{"key": ["TR02-1"]}})
FILLED = COLLECTIONS.assign(filled_from_ledger=[False, False])
seen = {{}}

class Stop(Exception):
    pass

settings = SimpleNamespace(
    settings_file=script.REPO_ROOT / "settings" / "stub.toml",
    data=SimpleNamespace(corpus_folder="stub-corpus", cache_folder="stub-cache",
                         ledger_file="stub-ledger", ledger_as_of=date(2026, 3, 31),
                         holidays_file="stub-holidays", fill_missing_days_from_ledger=SWITCH),
    collections_layout=SimpleNamespace(file_name_date_format="%d%m%Y"),
    ledger_layout="stub-layout",
    debug=SimpleNamespace(one_day=DAYS[0]),
    companies=SimpleNamespace(known={{"TR02": "stub"}}),
)

def fake_fill(collections, ledger, days):
    seen["fill_args"] = (collections, ledger, days)
    return FILLED, [date(2026, 3, 4)]

def fake_show_filled(collections, filled_days, switched_on):
    seen["show_filled"] = (collections, filled_days, switched_on)
    real_show_filled(collections, filled_days, switched_on)

def fake_add_week_index(collections, week_zero):
    seen["week_index_input"] = collections
    return collections.assign(week=0)

def fake_week_calendar(days, week_zero, holidays, filled_days=None):
    seen["calendar_filled_days"] = filled_days
    raise Stop

real_show_filled = script.show_filled
script.load_settings = lambda path, root: settings
script.list_daily_files = lambda folder, fmt: {{day: f"file-{{day}}" for day in DAYS}}
script.read_one_file = lambda path, layout: COLLECTIONS.iloc[:1]
script.load_collections = lambda files, layout, cache: COLLECTIONS
script.add_keys = lambda table, companies: table
script.load_ledger = lambda path, as_of, layout, cache: LEDGER
script.check_ledger_against_collections = lambda ledger, collections, days: {{}}
script.fill_missing_days = fake_fill
script.show_filled = fake_show_filled
script.add_week_index = fake_add_week_index
script.load_holidays = lambda path: []
script.week_calendar = fake_week_calendar
for name in ("show_daily_files", "show_one_day", "show_collections", "show_keys",
             "show_ledger", "show_tie"):
    setattr(script, name, lambda *args, **kwargs: None)

try:
    script.main()
except Stop:
    pass
else:
    raise AssertionError("main() did not reach week_calendar")

collections, filled_days, switched_on = seen["show_filled"]
assert switched_on is SWITCH
assert seen["week_index_input"] is collections
assert seen["calendar_filled_days"] == filled_days
if SWITCH:
    assert seen["fill_args"][0] is COLLECTIONS
    assert seen["fill_args"][1] is LEDGER
    assert seen["fill_args"][2] == DAYS
    assert collections is FILLED
    assert filled_days == [date(2026, 3, 4)]
else:
    assert "fill_args" not in seen, "fill_missing_days ran although switched off"
    assert filled_days == []
    assert list(collections.columns) == list(COLLECTIONS.columns) + ["filled_from_ledger"]
    assert collections["filled_from_ledger"].dtype == bool
    assert not collections["filled_from_ledger"].any()
    pd.testing.assert_frame_equal(collections.drop(columns="filled_from_ledger"), COLLECTIONS)
    # assign gives a new table: the loaded collections are not modified.
    assert "filled_from_ledger" not in COLLECTIONS.columns
print("ok")
"""


def run_step_6(switch):
    return subprocess.run([sys.executable, "-c", STEP_6.format(switch=switch)], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)


def test_step_6_switched_off_flags_every_line_false_and_fills_no_day():
    result = run_step_6(False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    assert "filling from the ledger is switched off" in result.stdout


def test_step_6_switched_on_takes_the_lines_and_days_from_fill_missing_days():
    result = run_step_6(True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    assert "1 days without a daily file filled from the ledger" in result.stdout


def test_the_pipeline_script_exists():
    assert SCRIPT.is_file()


def test_the_pipeline_script_imports_without_running():
    result = subprocess.run([sys.executable, "-c", CHECK], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


# Step 11, U6.2: the history features, from the weekly matrix, the known keys, the last week
# and the windows from the settings; and the printout of them. Run on a small hand-built
# matrix, in a separate process like the checks above.
HISTORY = """
import inspect
import pandas as pd
import run_hurdle_pipeline as script

inspect.signature(script.history_features).bind("matrix", "known", 64, (4, 13, 52))
try:
    inspect.signature(script.history_features).bind("matrix", "known", 64)
except TypeError:
    pass
else:
    raise AssertionError("history_features accepts a call without the windows")
assert callable(script.show_history)
source = inspect.getsource(script.main)
assert "history_features(matrix, known, last_week, settings.history.windows)" in source
assert "show_history(history, matrix)" in source
assert source.index("show_matrix(matrix)") < source.index("history_features(")

matrix = pd.DataFrame(0.0, index=pd.Index(["A", "B"], name="key", dtype=object),
                      columns=pd.Index(list(range(0, 14)), name="week"))
matrix.loc["A", 2] = 20.0
matrix.loc["A", 8] = 50.0
matrix.loc["B", 13] = 30.0
known = pd.DataFrame({"key": pd.Series(["A", "B"], dtype=object), "known_week": [-5, 9]})
history = script.history_features(matrix, known, 13, (4, 13))
script.show_history(history, matrix)
print("ok")
"""


def test_history_features_are_wired_in_and_printed():
    result = subprocess.run([sys.executable, "-c", HISTORY], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    # A: origin weeks 0..13 (14 rows), B: 9..13 (5 rows); 12 columns less key and origin.
    assert "history features: 19 rows (key x origin week), 10 features" in result.stdout
    assert "share NaN" in result.stdout
    assert "the key with the most receipts at origin weeks 0, 13 and 52" in result.stdout
    assert "median_payment_13" in result.stdout


# Step 11, U7: the calendar features of every target week, from week zero, week 0 to the
# last week + the horizons, and the holidays; and the printout of them. Run on the real
# settings/holidays_tr.toml (read only), in a separate process like the checks above.
CALENDAR = """
import inspect
from datetime import date
import run_hurdle_pipeline as script

inspect.signature(script.calendar_features).bind("week_zero", 0, 69, "holidays")
try:
    inspect.signature(script.calendar_features).bind("week_zero", 0, 69)
except TypeError:
    pass
else:
    raise AssertionError("calendar_features accepts a call without the holidays")
inspect.signature(script.show_calendar).bind("calendar", "week_zero", "holidays")
source = " ".join(inspect.getsource(script.main).split())
call = ("calendar = calendar_features(week_zero, 0, last_week + settings.receivables.horizons,"
        " holidays)")
assert call in source, "calendar_features is not called as expected"
assert "show_calendar(calendar, week_zero, holidays)" in source
assert source.index("show_history(history, matrix)") < source.index("calendar_features(")
assert source.index("calendar_features(") < source.index("show_calendar(")

holidays = script.load_holidays(script.REPO_ROOT / "settings" / "holidays_tr.toml")
week_zero = date(2025, 6, 2)
calendar = script.calendar_features(week_zero, 0, 69, holidays)
script.show_calendar(calendar, week_zero, holidays)
print("ok")
"""


def test_calendar_features_are_wired_in_and_printed():
    # The script writes UTF-8 (it reconfigures sys.stdout); the holiday names are Turkish.
    result = subprocess.run([sys.executable, "-c", CALENDAR], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    assert ("calendar features: weeks 0 to 69; weeks with a month-end: 16, a quarter-end: 6, "
            "the 15th: 16") in result.stdout
    assert "weeks with fewer than 5 business days:" in result.stdout
    assert "week 51 (Monday 2026-05-25): 2 business days (Kurban Bayramı)" in result.stdout
    assert ("quarter-end weeks start on: 2025-06-30, 2025-09-29, 2025-12-29, 2026-03-30, "
            "2026-06-29, 2026-09-28") in result.stdout
    assert "52: 3" in result.stdout
    assert "np.int64" not in result.stdout
