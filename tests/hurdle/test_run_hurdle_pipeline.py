"""Smoke tests for `run_hurdle_pipeline.py` (the step-by-step script).

Running the real `main()` reads the whole synthetic corpus and writes to data/interim, so it
is not run here. The import checks, and the step-6 branch tests (main() with every reader
stubbed and stopped right after step 6), run in a separate Python process: the script
reconfigures sys.stdout and puts src/ on sys.path, which should not leak into the rest of the
test session.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

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
    leak_check_weeks=(0, 30, 64), age_edges=(30, 90))
# After U9c.1: the aging-bucket edges are required.
try:
    inspect.signature(script.receivables_by_week).bind(
        "ledger", "rates", "week_zero", last_week=64, as_of="as_of", horizons=5,
        min_cleared=3, leak_check_weeks=(0, 30, 64))
except TypeError:
    pass
else:
    raise AssertionError("receivables_by_week accepts a call without the age edges")
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


# Step 8, U9c.1: the receivables are built with the aging edges from the settings, and the
# printout shows the aging table. Step 11, U9c.2: the receivables features, after the
# calendar features, and their printout. Run on small hand-built tables, in a separate
# process like the checks above.
RECEIVABLES = """
import inspect
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle.receivables_asof import receivables_columns

source = " ".join(inspect.getsource(script.main).split())
assert ("leak_check_weeks=settings.receivables.leak_check_weeks, "
        "age_edges=settings.receivables.overdue_age_days)") in source, "step 8 call"
assert ("show_receivables_by_week(receivables, settings.receivables.leak_check_weeks, "
        "settings.debug.sample_week, settings.receivables.overdue_age_days)") in source
call = ("owed = receivables_features(panel, receivables, levels, "
        "settings.receivables.horizons, settings.receivables.overdue_age_days)")
assert call in source, "receivables_features is not called as expected"
show = "show_owed(owed, int(before_first_ledger_week(panel, receivables).sum()))"
assert show in source, "show_owed is not called with the not-yet-in-ledger count"
assert source.index("show_calendar(") < source.index("receivables_features(")
assert source.index("receivables_features(") < source.index("show_owed(owed")
assert callable(script.before_first_ledger_week)
inspect.signature(script.show_owed).bind("owed", 0)
assert source.index("receivables_by_week(") < source.index("receivables_features(")
inspect.signature(script.receivables_features).bind("panel", "receivables", "levels", 5, (30, 90))
try:
    inspect.signature(script.receivables_features).bind("panel", "receivables", "levels", 5)
except TypeError:
    pass
else:
    raise AssertionError("receivables_features accepts a call without the age edges")
inspect.signature(script.show_receivables_by_week).bind("r", (0,), 30, (30, 90))
assert callable(script.show_owed)

columns = receivables_columns(5, (30, 90))
rows = []
for key in ["A", "B"]:
    for week in range(0, 17):
        row = dict.fromkeys(columns, 0.0)
        row.update(key=key, week=week, open_invoices=1, cleared_invoices=0,
                   usual_delay_days=float("nan"))
        row["due_in_week_1"] = 1e6
        row["overdue"] = 3e6
        row["overdue_up_to_30"], row["overdue_31_to_90"], row["overdue_over_90"] = 1e6, 0.5e6, 1.5e6
        rows.append(row)
receivables = pd.DataFrame(rows, columns=columns)
for column in ("week", "open_invoices", "cleared_invoices"):
    receivables[column] = receivables[column].astype("int64")
script.show_receivables_by_week(receivables, (0, 16), 8, (30, 90))

# Key C is known but not yet in the ledger: its ledger starts in week 10, after its panel
# weeks 3 and 4, so those two rows are filled as nothing owed. (A key with no receivables
# row at all would make receivables_features raise.) Added after the aging printout so the
# aging totals above stay those of A and B.
c_rows = []
for week in range(10, 17):
    row = dict.fromkeys(columns, 0.0)
    row.update(key="C", week=week, open_invoices=0, cleared_invoices=0,
               usual_delay_days=float("nan"))
    c_rows.append(row)
c_table = pd.DataFrame(c_rows, columns=columns)
for column in ("week", "open_invoices", "cleared_invoices"):
    c_table[column] = c_table[column].astype("int64")
receivables = pd.concat([receivables, c_table], ignore_index=True)
panel = pd.DataFrame({"key": ["A", "A", "B", "C", "C"], "origin_week": [3, 3, 16, 3, 4],
                      "h": [1, 2, 5, 1, 3]})
levels = pd.Series(1.0, index=range(0, 17))
owed = script.receivables_features(panel, receivables, levels, 5, (30, 90))
filled = int(script.before_first_ledger_week(panel, receivables).sum())
assert filled == 2, filled
script.show_owed(owed, filled)
# A key with no receivables row at all is refused, not filled.
absent = pd.concat([panel, pd.DataFrame({"key": ["D"], "origin_week": [3], "h": [1]})],
                   ignore_index=True)
try:
    script.receivables_features(absent, receivables, levels, 5, (30, 90))
except ValueError as error:
    assert str(error) == "1 panel keys have no receivables row at all (first: ['D'])", error
else:
    raise AssertionError("a key with no receivables row was filled")
print("ok")
"""


def test_receivables_aging_and_features_are_wired_in_and_printed():
    result = subprocess.run([sys.executable, "-c", RECEIVABLES], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    assert "overdue by days past due, million TRY (every 8th week):" in result.stdout
    for column in ("overdue_up_to_30", "overdue_31_to_90", "overdue_over_90"):
        assert column in result.stdout
    # Two keys of 3.0 million overdue each: 6.0 million, split 2.0 / 1.0 / 3.0 per week.
    aging = result.stdout.split("overdue by days past due")[1].splitlines()
    assert aging[1].split() == ["overdue", "overdue_up_to_30", "overdue_31_to_90",
                                "overdue_over_90"]
    assert [line.split() for line in aging[3:6]] == [
        [week, "6.0", "2.0", "1.0", "3.0"] for week in ("0", "8", "16")]
    assert "receivables features: 5 rows (key x origin week x h), 12 features" in result.stdout
    lines = [line.strip() for line in result.stdout.splitlines()]
    assert "rows of keys not yet in the ledger (filled as nothing owed): 2" in lines
    assert "share NaN" in result.stdout
    assert "due before / in / after the target week, by h" in result.stdout
    # Since the price levels are normalised to week 0, amounts read as TRY of week 0.
    assert ("due before / in / after the target week, by h (billion TRY of week 0, summed "
            "over rows):") in lines
    assert "TRY of the last week" not in result.stdout


# Step 10: the price-level printout. published_levels() is the copied module's published
# level rescaled to 1.0 at week 0; the printout checks that and shows week 0 and the last
# week. Run on the real TÜFE table for the synthetic weeks -32 to 64, in a separate process.
# PRICES is run twice: once with the real published_levels(), once with the copied
# module's level (the OLD normalisation) passed as `levels`, which must print a mismatch.
PRICES = """
from datetime import date
import pandas as pd
import run_hurdle_pipeline as script

week_zero = date(2025, 6, 2)
weeks = script.price_weeks(week_zero, -32, 64)
actual = script.inflation.weekly_price_level(weeks)
published = script.inflation.published_price_level(weeks)
projected = script.inflation.projected_price_level(weeks)
if {old_basis}:
    levels = pd.Series(published, index=weeks["week"].to_numpy(), name="price_level")
else:
    levels = script.published_levels(week_zero, -32, 64)
script.show_prices(weeks, actual, published, projected, levels)

matrix = pd.DataFrame(0.0, index=pd.Index(["A", "B"], name="key", dtype=object),
                      columns=pd.Index(list(range(-2, 4)), name="week"))
matrix.loc["A", 1] = 20.0
matrix.loc["A", 3] = 5.0
script.show_matrix(matrix)
print("ok")
"""


def run_prices(old_basis):
    return subprocess.run([sys.executable, "-c", PRICES.format(old_basis=old_basis)],
                          cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
                          timeout=120)


def test_price_printout_says_the_copied_module_is_rescaled_to_week_zero():
    result = run_prices(False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    first_line = result.stdout.splitlines()[0]
    assert first_line.startswith("Step 10: price levels for weeks -32 to 64 "
                                 "(2024-10-21 to 2026-08-30)")
    assert first_line.endswith("the copied module gives 1.0 = the last week, "
                               "published_levels() rescales to 1.0 = week 0")


def test_price_printout_reports_a_match_and_week_zero_and_last_week_levels():
    result = run_prices(False)
    assert result.returncode == 0, result.stderr
    # Week 0 is 1.000; the last week publishes July 2026, 1.363 times May 2025.
    assert "published_levels() matches: True; week 0 = 1.000, last week = 1.363" in result.stdout


def test_price_printout_reports_a_mismatch_for_the_old_normalisation():
    # The copied module's level (1.0 = the last week's actual level) is not what
    # published_levels() should give any more: the check must say so.
    result = run_prices(True)
    assert result.returncode == 0, result.stderr
    assert "published_levels() matches: False" in result.stdout
    assert "week 0 = 0.720, last week = 0.982" in result.stdout


def test_matrix_printout_says_try_of_week_zero():
    result = run_prices(False)
    assert result.returncode == 0, result.stderr
    assert ("the key with the most receipts, its first weeks with a payment (TRY of week 0):"
            in result.stdout)
    assert "TRY of the last week" not in result.stdout


# Steps 10 and 11 with a ledger that starts AFTER the first daily file (week 0): the price
# range and the history start are floored at week 0, so published_levels() (normalised to
# week 0) and weekly_matrix() (which must hold the daily files' weeks from 0) do not raise.
# Checked on main()'s syntax tree, read from the file without importing or running it; the
# behaviour itself is tested in test_ledger_starting_late.py.
import ast  # noqa: E402


def main_tree():
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    mains = [node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == "main"]
    assert len(mains) == 1
    return mains[0]


def main_source():
    return " ".join(ast.get_source_segment(SCRIPT.read_text(encoding="utf-8"),
                                           main_tree()).split())


def assignments_to(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]


def calls_to(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == name]


def is_min_with_zero(value, first_arg_check):
    """value is `min(<first>, 0)` with exactly two positional arguments."""
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "min" and len(value.args) == 2 and not value.keywords
            and first_arg_check(value.args[0])
            and isinstance(value.args[1], ast.Constant) and value.args[1].value == 0)


def test_step_10_first_ledger_week_is_the_week_of_the_first_posting():
    source = main_source()
    assert ('first_ledger_week = int(week_numbers(pd.Series([ledger["posting_date"].min()]), '
            'week_zero).iloc[0])') in source
    assert len(assignments_to(main_tree(), "first_ledger_week")) == 1


def test_step_10_first_price_week_is_floored_at_week_zero():
    found = assignments_to(main_tree(), "first_price_week")
    assert len(found) == 1, "first_price_week must be assigned exactly once"
    assert is_min_with_zero(found[0].value,
                            lambda arg: isinstance(arg, ast.Name)
                            and arg.id == "first_ledger_week"), ast.dump(found[0].value)
    assert "first_price_week = min(first_ledger_week, 0)" in main_source()


def test_step_10_price_weeks_and_published_levels_take_the_floored_week():
    tree = main_tree()
    for name in ("price_weeks", "published_levels"):
        calls = calls_to(tree, name)
        assert len(calls) == 1, name
        args = [ast.unparse(arg) for arg in calls[0].args]
        assert args == ["week_zero", "first_price_week", "last_week"], (name, args)
    source = main_source()
    assert "weeks_for_prices = price_weeks(week_zero, first_price_week, last_week)" in source
    assert "levels = published_levels(week_zero, first_price_week, last_week)" in source


def test_step_10_order_ledger_week_then_floor_then_prices():
    source = main_source()
    assert (source.index("first_ledger_week = ")
            < source.index("first_price_week = min(first_ledger_week, 0)")
            < source.index("price_weeks(week_zero, first_price_week")
            < source.index("published_levels(week_zero, first_price_week")
            < source.index("show_prices("))
    # The floor comes after last_week is known and after the ledger is loaded.
    assert source.index("ledger = load_ledger(") < source.index("first_ledger_week = ")
    assert source.index("last_week = int(weeks[\"week\"].max())") < source.index(
        "first_price_week = ")


def history_switch():
    """The `if settings.history.use_ledger_before_first_file:` statement in main()."""
    found = [node for node in ast.walk(main_tree()) if isinstance(node, ast.If)
             and ast.unparse(node.test) == "settings.history.use_ledger_before_first_file"]
    assert len(found) == 1
    return found[0]


def test_step_11_switched_on_history_start_is_floored_at_week_zero():
    switch = history_switch()
    found = [node for statement in switch.body for node in assignments_to(statement,
                                                                          "history_start")]
    assert len(found) == 1
    assert is_min_with_zero(found[0].value,
                            lambda arg: ast.unparse(arg) == "first_history_week(ledger, "
                                                            "week_zero)"), ast.dump(found[0].value)
    assert "history_start = min(first_history_week(ledger, week_zero), 0)" in main_source()


def test_step_11_switched_off_history_start_is_still_zero():
    switch = history_switch()
    found = [node for statement in switch.orelse for node in assignments_to(statement,
                                                                            "history_start")]
    assert len(found) == 1
    assert isinstance(found[0].value, ast.Constant) and found[0].value.value == 0


def test_step_11_history_start_is_assigned_only_in_the_switch():
    assert len(assignments_to(main_tree(), "history_start")) == 2


def test_step_11_early_receipts_are_cut_at_the_history_start_and_the_matrix_starts_there():
    source = main_source()
    assert 'early_receipts = early_receipts[early_receipts["week"] >= history_start]' in source
    calls = calls_to(main_tree(), "weekly_matrix")
    assert len(calls) == 1
    args = [ast.unparse(arg) for arg in calls[0].args]
    assert args == ["identified_weekly", "early_receipts", "known['key']", "weeks", "levels",
                    "history_start", "last_week"], args
    assert (source.index("history_start = min(first_history_week(ledger, week_zero), 0)")
            < source.index('early_receipts = early_receipts[early_receipts["week"] >= '
                           'history_start]')
            < source.index("matrix = weekly_matrix("))
    # The price levels the matrix divides by are built (step 10) before the matrix (step 11).
    assert source.index("levels = published_levels(") < source.index("matrix = weekly_matrix(")


# Step 11: the deflated label, added to the panel after the receivables features and printed.
# The wiring is checked on main()'s source; the printout is run on a hand-built panel in a
# separate process: target weeks 0, 16, 17, 32 labelled, one missing-week and one future row,
# and one h=2 row (target 17) that the h=1 table must leave out.
DEFLATED = """
import inspect
import pandas as pd
import run_hurdle_pipeline as script

source = " ".join(inspect.getsource(script.main).split())
assert "panel = add_deflated_label(panel, levels)" in source, "add_deflated_label not called"
assert "show_deflated_label(panel, levels)" in source, "show_deflated_label not called"
assert (source.index("show_owed(owed")
        < source.index("panel = add_deflated_label(panel, levels)")
        < source.index("show_deflated_label(panel, levels)"))
assert source.index("levels = published_levels(") < source.index("add_deflated_label(")
assert source.index("panel = add_labels(") < source.index("add_deflated_label(")
inspect.signature(script.add_deflated_label).bind("panel", "levels")
inspect.signature(script.show_deflated_label).bind("panel", "levels")

nan = float("nan")
panel = pd.DataFrame({
    "key": ["A", "B", "A", "B", "A", "A", "B", "A"],
    "origin_week": [-1, -1, 15, 15, 16, 31, 31, 64],
    "h": [1, 1, 1, 2, 1, 1, 1, 1],
    "label_status": ["labelled"] * 6 + ["missing_week", "future"],
    "amount": [1e6, 1e6, 5e6, 3e6, 2e6, 8e6, nan, nan],
})
panel["target_week"] = panel["origin_week"] + panel["h"]
# No level for the future week 65: it must not be looked up.
levels = pd.Series({0: 1.0, 16: 1.25, 17: 2.0, 32: 1.6})
panel = script.add_deflated_label(panel, levels)
assert panel["amount_deflated"].tolist()[:6] == [1e6, 1e6, 4e6, 1.5e6, 1e6, 5e6]
script.show_deflated_label(panel, levels)
print("ok")
"""


def run_deflated():
    return subprocess.run([sys.executable, "-c", DEFLATED], cwd=REPO_ROOT,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)


def test_deflated_label_is_wired_in_after_the_receivables_features():
    result = run_deflated()
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_deflated_label_printout_counts():
    result = run_deflated()
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines()]
    assert "deflated label: 6 labelled rows, 2 NaN (rows not labelled: 2)" in lines


def test_deflated_label_printout_h1_table_every_16th_week():
    result = run_deflated()
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines()]
    title = ("at h=1, label per target week, million TRY (nominal / of week 0), every 16th:")
    assert title in lines
    table = lines[lines.index(title) + 1:]
    assert table[0].split() == ["amount", "amount_deflated", "ratio", "level"]
    assert table[1].split() == ["target_week"]
    # Week 0: 1 + 1 million, level 1.0. Week 16: 5 million at 1.25. Week 32: 8 million at
    # 1.6. Week 17 (not a multiple of 16) is left out; the h=2 row never counts.
    rows = [line.split() for line in table[2:5]]
    assert [[float(value) for value in row] for row in rows] == [
        [0, 2.0, 2.0, 1.0, 1.0],
        [16, 5.0, 4.0, 1.25, 1.25],
        [32, 8.0, 5.0, 1.6, 1.6],
    ]
    assert table[5] == "ok"


# Step 11: the training table, joined after the deflated label and printed. The wiring is
# checked on main()'s source; the printout is run on a small hand-built panel and blocks in a
# separate process: 4 panel rows (2 labelled, one of them paid; one missing week; one
# future), one NaN in a history feature, none in the calendar or receivables features.
TABLE = """
import inspect
import pandas as pd
import run_hurdle_pipeline as script

source = " ".join(inspect.getsource(script.main).split())
assert "table = join_features(panel, history, calendar, owed)" in source, "join_features"
assert "show_table(table, panel)" in source, "show_table not called"
assert (source.index("show_deflated_label(panel, levels)")
        < source.index("table = join_features(panel, history, calendar, owed)")
        < source.index("show_table(table, panel)"))
# Every block is built before the join, and the panel has its deflated label by then.
for built in ("history = history_features(", "calendar = calendar_features(",
              "owed = receivables_features(", "panel = add_deflated_label("):
    assert source.index(built) < source.index("table = join_features("), built
inspect.signature(script.join_features).bind("panel", "history", "calendar", "owed")
inspect.signature(script.show_table).bind("table", "panel")

nan = float("nan")
panel = pd.DataFrame({
    "key": pd.Series(["TR02-A", "TR02-A", "TR02-B", "TR02-B"], dtype=object),
    "origin_week": pd.Series([0, 0, 1, 1], dtype="int64"),
    "h": pd.Series([1, 2, 1, 2], dtype="int64"),
    "label_status": pd.Series(["labelled", "missing_week", "labelled", "future"],
                              dtype=object),
    "amount": [250.0, nan, 0.0, nan],
})
panel["target_week"] = panel["origin_week"] + panel["h"]
panel["amount_deflated"] = panel["amount"] / 2
panel["paid"] = pd.array([True, None, False, None], dtype="boolean")
history = pd.DataFrame({"key": pd.Series(["TR02-A", "TR02-B"], dtype=object),
                        "origin_week": pd.Series([0, 1], dtype="int64"),
                        "weeks_since_last_payment": [3.0, nan]})
calendar = pd.DataFrame({"week": pd.Series([1, 2, 3], dtype="int64"),
                         "business_days": pd.Series([5, 4, 5], dtype="int64")})
owed = pd.DataFrame({"key": pd.Series(["TR02-A", "TR02-A", "TR02-B", "TR02-B"], dtype=object),
                     "origin_week": pd.Series([0, 0, 1, 1], dtype="int64"),
                     "h": pd.Series([1, 2, 1, 2], dtype="int64"),
                     "open_balance": [10.0, 10.0, 20.0, 20.0]})
table = script.join_features(panel, history, calendar, owed)
assert len(table) == 4 and table.shape[1] == 11, table.shape
script.show_table(table, panel)
print("ok")
"""


def run_table():
    return subprocess.run([sys.executable, "-c", TABLE], cwd=REPO_ROOT,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)


def test_training_table_is_wired_in_after_the_deflated_label():
    result = run_table()
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_training_table_printout_rows_columns_and_size():
    result = run_table()
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines()]
    assert "training table: 4 rows (panel: 4), 11 columns, about 0 MB" in lines


def test_training_table_printout_nan_shares_of_columns_with_any():
    result = run_table()
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines()]
    title = "share NaN, columns that have any:"
    assert title in lines
    shares = {}
    for line in lines[lines.index(title) + 1:]:
        if line == "one labelled, paid row:":
            break
        name, value = line.split()
        shares[name] = float(value)
    # amount, amount_deflated and paid: 2 of 4 rows; weeks_since_last_payment: B's 2 rows.
    assert shares == {"amount": 0.5, "amount_deflated": 0.5, "paid": 0.5,
                      "weeks_since_last_payment": 0.5}


def test_training_table_printout_shows_one_labelled_paid_row():
    result = run_table()
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines()]
    title = "one labelled, paid row:"
    assert title in lines
    row = dict(line.split(maxsplit=1) for line in lines[lines.index(title) + 1:-1])
    assert row["key"] == "TR02-A"
    assert row["label_status"] == "labelled"
    assert row["paid"] == "True"
    assert float(row["amount"]) == 250.0
    assert float(row["open_balance"]) == 10.0
    assert int(row["business_days"]) == 5


# show_table on a table with no labelled, paid row: it says so and returns (it used to raise
# IndexError on .iloc[0]). Run in a separate process on hand-built tables; the variant picks
# which label_status / paid values the four rows carry.
NO_PAID = """
import pandas as pd
import run_hurdle_pipeline as script

variants = {{
    "no_labelled": (["missing_week", "future", "missing_week", "future"],
                    [None, None, None, None]),
    "all_unpaid": (["labelled", "labelled", "missing_week", "future"],
                   [False, False, None, None]),
    "paid_but_not_labelled": (["future", "missing_week", "labelled", "future"],
                              [True, True, False, None]),
    "labelled_with_na_paid": (["labelled", "labelled", "future", "future"],
                              [None, False, True, None]),
    "empty": ([], []),
}}
status, paid = variants["{variant}"]
n = len(status)
table = pd.DataFrame({{
    "key": pd.Series(["TR02-A"] * n, dtype=object),
    "origin_week": pd.Series(range(n), dtype="int64"),
    "h": pd.Series([1] * n, dtype="int64"),
    "label_status": pd.Series(status, dtype=object),
    "amount": pd.Series([0.0 if p is False else float("nan") for p in paid], dtype="float64"),
    "paid": pd.array(paid, dtype="boolean"),
    "business_days": pd.Series([5] * n, dtype="int64"),
}})
script.show_table(table, table)
print("ok")
"""


def run_no_paid(variant):
    return subprocess.run([sys.executable, "-c", NO_PAID.format(variant=variant)],
                          cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
                          timeout=120)


NO_PAID_LINE = "        no labelled, paid row to show"


@pytest.mark.parametrize("variant", ["no_labelled", "all_unpaid", "paid_but_not_labelled",
                                     "labelled_with_na_paid", "empty"])
def test_training_table_printout_without_a_labelled_paid_row_says_so_and_returns(variant):
    result = run_no_paid(variant)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    # The exact line (8-space indent), last before "ok", and no row is shown.
    assert lines[-2:] == [NO_PAID_LINE, "ok"], lines
    assert "one labelled, paid row:" not in [line.strip() for line in lines]
    assert lines[0].strip().startswith("training table:")


def test_training_table_printout_with_a_paid_row_does_not_say_none_to_show():
    result = run_table()
    assert result.returncode == 0, result.stderr
    assert NO_PAID_LINE.strip() not in [line.strip() for line in result.stdout.splitlines()]


# Step 12 (U10.1): stage 1's rows and model input at [debug] demo_origin. The wiring is
# checked on main()'s source; the guard on demo_origin (in step 7, right after
# show_weeks(weeks)) is run from main()'s own syntax tree; the printout is run on a small hand-built training table in a separate process.
# The table: keys TR02-A (always paid) and TR04-B (never paid), origins 0..3, h = 1, 2, last
# week 3 (targets 4 and 5 are future). At origin 2 the training rows are (0,1), (0,2), (1,1)
# per key = 6 rows, half of them paid; the forecast rows are 2 keys x 2 h = 4 rows. Features:
# h, mean_receipts_4 (NaN for B), has_month_end (bool), open_invoices (int).
STAGE1 = """
import inspect
import pandas as pd
import run_hurdle_pipeline as script

source = " ".join(inspect.getsource(script.main).split())
steps = [
    "show_table(table, panel)",
    "origin = settings.debug.demo_origin",
    ("columns = feature_columns(block_feature_columns(history, calendar, owed), "
     "settings.stage1.feature_blocks)"),
    "train_rows = rows_known_at(table, origin)",
    "forecast_rows = rows_at_origin(table, origin)",
    ("train_x = stage1_features(train_rows, columns, companies, "
     "settings.stage1.company_as_feature)"),
    ("forecast_x = stage1_features(forecast_rows, columns, companies, "
     "settings.stage1.company_as_feature)"),
    "show_stage1_rows(origin, train_rows, forecast_rows, train_x)",
]
for step in steps:
    assert step in source, step
positions = [source.index(step) for step in steps]
assert positions == sorted(positions), positions
assert source.index("table = join_features(") < source.index("train_rows = rows_known_at(")
assert source.index("companies = list(settings.companies.known)") < source.index(
    "train_x = stage1_features(")
inspect.signature(script.show_stage1_rows).bind("origin", "train", "forecast", "train_x")
inspect.signature(script.feature_columns).bind({}, ())
inspect.signature(script.block_feature_columns).bind("history", "calendar", "owed")
inspect.signature(script.rows_known_at).bind("table", 40)
inspect.signature(script.rows_at_origin).bind("table", 40)
inspect.signature(script.stage1_features).bind("rows", [], [], True)

nan = float("nan")
rows = [(key, t, h) for key in ("TR02-A", "TR04-B") for t in range(0, 4) for h in (1, 2)]
table = pd.DataFrame({
    "key": pd.Series([r[0] for r in rows], dtype=object),
    "company_code": pd.Series([r[0][:4] for r in rows], dtype=object),
    "customer_id": pd.Series([r[0][5:] for r in rows], dtype=object),
    "origin_week": pd.Series([r[1] for r in rows], dtype="int64"),
    "h": pd.Series([r[2] for r in rows], dtype="int64"),
})
table["target_week"] = table["origin_week"] + table["h"]
future = table["target_week"] > 3
table["label_status"] = pd.Series(["future" if f else "labelled" for f in future], dtype=object)
is_a = table["key"] == "TR02-A"
table["amount"] = [nan if f else (100.0 if a else 0.0) for f, a in zip(future, is_a)]
table["amount_deflated"] = table["amount"]
table["paid"] = pd.array([None if f else bool(a) for f, a in zip(future, is_a)],
                         dtype="boolean")
table["mean_receipts_4"] = [1.0 if a else nan for a in is_a]
table["has_month_end"] = table["target_week"] % 2 == 1
table["open_invoices"] = pd.Series(table["origin_week"] * 2, dtype="int64")

history = table[["key", "origin_week", "mean_receipts_4"]].drop_duplicates(["key", "origin_week"])
calendar = (table[["target_week", "has_month_end"]].drop_duplicates("target_week")
            .rename(columns={"target_week": "week"}))
owed = table[["key", "origin_week", "h", "open_invoices"]]
columns = script.feature_columns(script.block_feature_columns(history, calendar, owed),
                                 ("history", "calendar", "receivables"))
assert columns == ["h", "mean_receipts_4", "has_month_end", "open_invoices"], columns
origin = ORIGIN
train_rows = script.rows_known_at(table, origin)
forecast_rows = script.rows_at_origin(table, origin)
companies = ["TR02", "TR04", "TR05"]
train_x = script.stage1_features(train_rows, columns, companies, COMPANY_AS_FEATURE)
script.show_stage1_rows(origin, train_rows, forecast_rows, train_x)
print("ok")
"""


def run_stage1(company=True, origin=2):
    # str.replace, not str.format: the script holds dict literals.
    script = STAGE1.replace("COMPANY_AS_FEATURE", str(company)).replace("ORIGIN", str(origin))
    return subprocess.run([sys.executable, "-c", script],
                          cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
                          timeout=120)


def test_step_12_is_wired_in_after_the_training_table():
    result = run_stage1()
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_step_12_printout_rows_and_paid_share():
    result = run_stage1()
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == ("Step 12: stage 1 at forecast week 2: 6 training rows (labelled, "
                        "target week <= 2; 50.0% paid), 4 forecast rows (origin 2, "
                        "2 customers x h)")


def test_step_12_printout_model_input_with_the_company():
    result = run_stage1()
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[1] == ("        model input: 5 columns {'float64': 4, 'category': 1}; "
                        "columns with NaN: 1")
    assert lines[2] == "        company_code categories: ['TR02', 'TR04', 'TR05']"
    assert lines[3:] == ["ok"]


def test_step_12_printout_model_input_without_the_company():
    result = run_stage1(company=False)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[1] == "        model input: 4 columns {'float64': 4}; columns with NaN: 1"
    assert "company_code categories" not in result.stdout
    assert lines[2:] == ["ok"]


GUARD_TEST = ("settings.debug.demo_origin + settings.receivables.horizons > "
              "int(weeks['week'].max())")


def is_demo_origin_guard(node):
    return isinstance(node, ast.If) and ast.unparse(node.test) == GUARD_TEST


def demo_origin_guard():
    """The `if settings.debug.demo_origin + ... > int(weeks['week'].max()):` statement, which
    since the move sits in step 7 of main()."""
    found = [node for node in ast.walk(main_tree()) if is_demo_origin_guard(node)]
    assert len(found) == 1
    return found[0]


def run_guard(origin, horizons, last_week, first_week=0, week_dtype="int64"):
    """Run main()'s guard statement on its own, with the two names it reads: settings and a
    week calendar whose weeks run first_week..last_week."""
    from types import SimpleNamespace
    import pandas as pd
    module = ast.Module(body=[demo_origin_guard()], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(SCRIPT), "exec")
    settings = SimpleNamespace(debug=SimpleNamespace(demo_origin=origin),
                               receivables=SimpleNamespace(horizons=horizons))
    weeks = pd.DataFrame({"week": pd.Series(range(first_week, last_week + 1), dtype=week_dtype)})
    exec(code, {"settings": settings, "weeks": weeks})


def test_step_12_origin_is_read_from_debug_demo_origin():
    found = assignments_to(main_tree(), "origin")
    assert len(found) == 1
    assert ast.unparse(found[0].value) == "settings.debug.demo_origin"


def test_demo_origin_guard_sits_right_after_show_weeks():
    body = main_tree().body
    shows = [i for i, node in enumerate(body)
             if isinstance(node, ast.Expr) and ast.unparse(node) == "show_weeks(weeks)"]
    assert len(shows) == 1
    assert is_demo_origin_guard(body[shows[0] + 1]), ast.unparse(body[shows[0] + 1])


def test_demo_origin_guard_is_a_top_level_statement_of_main():
    # Not nested in a branch: it runs on every call of main().
    assert sum(is_demo_origin_guard(node) for node in main_tree().body) == 1


def test_demo_origin_guard_comes_before_step_8_receivables():
    source = main_source()
    guard = source.index("if settings.debug.demo_origin + settings.receivables.horizons")
    assert source.index("show_weeks(weeks)") < guard
    assert guard < source.index("receivables = receivables_by_week(")
    assert guard < source.index("exchange_rates(")


def test_demo_origin_guard_comes_before_any_stage_1_work():
    source = main_source()
    guard = source.index("if settings.debug.demo_origin + settings.receivables.horizons")
    for later in ("table = join_features(", "origin = settings.debug.demo_origin",
                  "feature_columns(", "rows_known_at(", "rows_at_origin(", "stage1_features(",
                  "show_stage1_rows("):
        assert guard < source.index(later), later


def test_step_12_has_no_guard_of_its_own_any_more():
    # The old step-12 guard compared `origin` with a `last_week` name; it must be gone.
    tree = main_tree()
    old = [node for node in ast.walk(tree) if isinstance(node, ast.If)
           and "origin" in ast.unparse(node.test) and not is_demo_origin_guard(node)]
    assert old == [], [ast.unparse(node.test) for node in old]
    assert "so the check would have no answers" not in main_source()


def test_demo_origin_guard_body_is_a_single_raise_of_value_error():
    guard = demo_origin_guard()
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Raise)
    assert ast.unparse(guard.body[0].exc.func) == "ValueError"
    assert guard.orelse == []


def test_demo_origin_guard_raises_with_the_exact_message():
    with pytest.raises(ValueError) as error:
        run_guard(origin=60, horizons=5, last_week=64)
    assert str(error.value) == ("[debug] demo_origin 60: its target weeks reach week 65, "
                                "after the last week 64, so step 12 would have no answers")


def test_demo_origin_guard_message_shows_plain_ints_for_numpy_weeks():
    # weeks['week'].max() is a numpy integer; int() keeps "64", not "np.int64(64)".
    with pytest.raises(ValueError) as error:
        run_guard(origin=100, horizons=3, last_week=64, week_dtype="int32")
    assert str(error.value) == ("[debug] demo_origin 100: its target weeks reach week 103, "
                                "after the last week 64, so step 12 would have no answers")


def test_demo_origin_guard_allows_targets_up_to_the_last_week():
    run_guard(origin=59, horizons=5, last_week=64)     # 59 + 5 = 64: the last week has data
    run_guard(origin=40, horizons=5, last_week=64)
    run_guard(origin=0, horizons=1, last_week=1)
    run_guard(origin=0, horizons=0, last_week=0)


def test_demo_origin_guard_uses_the_largest_week_not_the_last_row():
    # Weeks out of order: max() is 64 although the last row is 10.
    from types import SimpleNamespace
    import pandas as pd
    module = ast.Module(body=[demo_origin_guard()], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(SCRIPT), "exec")
    settings = SimpleNamespace(debug=SimpleNamespace(demo_origin=59),
                               receivables=SimpleNamespace(horizons=5))
    weeks = pd.DataFrame({"week": [64, 0, 10]})
    exec(code, {"settings": settings, "weeks": weeks})


@pytest.mark.parametrize("origin, horizons, last_week", [(60, 5, 64), (64, 5, 64),
                                                          (100, 5, 64), (0, 2, 1)])
def test_demo_origin_guard_raises_once_a_target_is_after_the_last_week(origin, horizons,
                                                                       last_week):
    with pytest.raises(ValueError) as error:
        run_guard(origin, horizons, last_week)
    assert str(error.value) == (f"[debug] demo_origin {origin}: its target weeks reach week "
                                f"{origin + horizons}, after the last week {last_week}, so "
                                f"step 12 would have no answers")


def test_step_12_printout_with_no_training_rows_does_not_crash():
    # Origin 0: the earliest target is week 1, so nothing is known yet. The printout must
    # still run (main()'s guard only bounds demo_origin from above).
    result = run_stage1(origin=0)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("Step 12: stage 1 at forecast week 0: 0 training rows ")
    assert "4 forecast rows (origin 0, 2 customers x h)" in lines[0]
    assert lines[-1] == "ok"


# Step 12 (U10.2, U10.3): fit stage 1 (HGB) and the logistic-regression baseline at the demo
# origin, predict its forecast rows, score them. The wiring is checked on main()'s source; the
# printout runs in a separate process on a hand-made frame with hand-made p's, so every number
# can be worked out by hand: six forecast rows, of which four are labelled (h=1: paid, not
# paid; h=2: paid, not paid); one "missing_week" row and one "future" row, both with
# paid = NA, must be left out. Training share paid: 1 of 4 = 0.25.
# Model A ("stage 1 (HGB)"), p = 0.8, 0.4, [0.99], 0.6, 0.2, [0.01]:
#   log loss = -(ln .8 + ln .6)/2 = 0.3670, Brier = (.04+.16+.16+.04)/4 = 0.1,
#   AUC = 1 (both payers above both non-payers); per h: h=1 mean p 0.6, h=2 0.4
# Model B ("baseline (LR)"), p = 0.3, 0.9, [0.01], 0.6, 0.4, [0.99]:
#   log loss = -(ln .3 + ln .1 + ln .6 + ln .6)/4 = 1.1321,
#   Brier = (.49+.81+.16+.16)/4 = 0.405, AUC = 1 of 4 pairs = 0.25; per h: h=1 0.6, h=2 0.5
# Base rate: log loss = -(ln .25 + ln .75)/2 = 0.8370, Brier = (.5625+.0625)/2 = 0.3125,
#   AUC = 0.5.  Share paid per h: 0.5, 0.5.
SCORES_WIRING = """
import inspect
import time
import sklearn.metrics
import run_hurdle_pipeline as script

assert script.time is time
assert script.log_loss is sklearn.metrics.log_loss
assert script.brier_score_loss is sklearn.metrics.brier_score_loss
assert script.roc_auc_score is sklearn.metrics.roc_auc_score
from collection_estimation.hurdle import stage1
assert script.fit_stage1 is stage1.fit_stage1
assert script.fit_stage1_baseline is stage1.fit_stage1_baseline
assert script.predict_paid is stage1.predict_paid
inspect.signature(script.show_stage1_scores).bind("forecast_rows", {}, "train_rows", {})

source = " ".join(inspect.getsource(script.main).split())
# Step 12 only: cut main() after its last call. Step 14 (U11.1) times its own fit with the same
# `started = time.perf_counter()` line, which must not count against step 12's.
end = "show_stage1_scores(forecast_rows, predictions, train_rows, fit_seconds)"
assert source.count(end) == 1, end
source = source[:source.index(end) + len(end)]
steps = [
    "show_stage1_rows(origin, train_rows, forecast_rows, train_x)",
    "predictions, fit_seconds = {}, {}",
    ('for name, fit in (("stage 1 (HGB)", fit_stage1), ("baseline (LR)", '
     'fit_stage1_baseline)):'),
    "started = time.perf_counter()",
    'model = fit(train_x, train_rows["paid"], settings.stage1)',
    "fit_seconds[name] = time.perf_counter() - started",
    "predictions[name] = predict_paid(model, forecast_x)",
    "show_stage1_scores(forecast_rows, predictions, train_rows, fit_seconds)",
]
for step in steps:
    assert step in source, step
    assert source.count(step) == 1, step
positions = [source.index(step) for step in steps]
assert positions == sorted(positions), positions
# The old single-model wiring is gone.
assert "stage1 = fit_stage1(" not in source
assert "p = predict_paid(stage1" not in source
# Each fit is timed alone: nothing else between the two clock reads.
between = source[source.index("started = time.perf_counter()"):
                 source.index("fit_seconds[name] = time.perf_counter() - started")]
assert between.count("(") == 2, between   # perf_counter() and fit(...)
print("ok")
"""

SCORES_WIRING_LOOP = """
import ast
import inspect
import textwrap
import run_hurdle_pipeline as script

tree = ast.parse(textwrap.dedent(inspect.getsource(script.main)))
loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
         and ast.unparse(node.target) == "(name, fit)"]
assert len(loops) == 1, len(loops)
loop = loops[0]
pairs = [(item.elts[0].value, ast.unparse(item.elts[1])) for item in loop.iter.elts]
assert pairs == [("stage 1 (HGB)", "fit_stage1"), ("baseline (LR)", "fit_stage1_baseline")]
body = [ast.unparse(statement) for statement in loop.body]
assert body == ["started = time.perf_counter()",
                "model = fit(train_x, train_rows['paid'], settings.stage1)",
                "fit_seconds[name] = time.perf_counter() - started",
                "predictions[name] = predict_paid(model, forecast_x)"], body
assert loop.orelse == []
print("ok")
"""

MODEL_A = "[0.8, 0.4, 0.99, 0.6, 0.2, 0.01]"
MODEL_B = "[0.3, 0.9, 0.01, 0.6, 0.4, 0.99]"

SCORES = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script

forecast_rows = pd.DataFrame({
    "key": ["TR02-A", "TR04-B", "TR02-C", "TR02-A", "TR04-B", "TR02-C"],
    "h": np.array([1, 1, 1, 2, 2, 2], dtype="int64"),
    "label_status": ["labelled", "labelled", "missing_week", "labelled", "labelled", "future"],
    "paid": pd.array([True, False, None, True, False, None], dtype="boolean"),
}, index=pd.Index([10, 11, 12, 13, 14, 15]))
predictions = {name: pd.Series(values, index=forecast_rows.index, name="p")
               for name, values in MODELS}
fit_seconds = FIT_SECONDS
train_rows = pd.DataFrame({"paid": pd.array([True, False, False, False], dtype="boolean")},
                          index=pd.Index([1, 2, 3, 4]))
copies = (forecast_rows.copy(deep=True), {n: p.copy() for n, p in predictions.items()},
          train_rows.copy(deep=True), dict(fit_seconds))
returned = script.show_stage1_scores(forecast_rows, predictions, train_rows, fit_seconds)
assert returned is None
pd.testing.assert_frame_equal(forecast_rows, copies[0])
assert list(predictions) == list(copies[1])
for name in predictions:
    pd.testing.assert_series_equal(predictions[name], copies[1][name])
pd.testing.assert_frame_equal(train_rows, copies[2])
assert fit_seconds == copies[3]
print("ok")
"""

ONE_MODEL = [("stage 1 (HGB)", MODEL_A)]
TWO_MODELS = [("stage 1 (HGB)", MODEL_A), ("baseline (LR)", MODEL_B)]


def run_scores(models=None, fit_seconds=None):
    """Run SCORES with `models` [(name, p list text)] (default: model A alone) and
    `fit_seconds` {name: seconds} (default: 1.23 s per model)."""
    models = ONE_MODEL if models is None else models
    if fit_seconds is None:
        fit_seconds = {name: 1.23 for name, _ in models}
    models_text = "[" + ", ".join(f"({name!r}, {values})" for name, values in models) + "]"
    script = SCORES.replace("MODELS", models_text).replace("FIT_SECONDS", repr(fit_seconds))
    return subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True,
                          text=True, encoding="utf-8", timeout=120)


def score_table(lines, n_rows):
    """The header words of the score table and its `n_rows` rows as {name: [log loss, Brier,
    AUC]}, in printed order."""
    header = lines[1].split()
    table = {}
    for line in lines[2:2 + n_rows]:
        parts = line.rsplit(maxsplit=3)
        table[parts[0].strip()] = [float(n) for n in parts[1:]]
    return header, table


def test_step_12_fit_predict_and_scores_are_wired_in_after_the_rows():
    result = subprocess.run([sys.executable, "-c", SCORES_WIRING], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_step_12_loops_over_hgb_then_the_baseline_and_collects_both_dicts():
    result = subprocess.run([sys.executable, "-c", SCORES_WIRING_LOOP], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_step_12_scores_runs_and_does_not_change_its_inputs():
    result = run_scores(TWO_MODELS)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "ok"


def test_step_12_scores_first_line_counts_only_labelled_rows():
    result = run_scores()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == ("        fitted in stage 1 (HGB) 1.2 s; scored on 4 "
                                             "forecast rows (50.0% paid)")


def test_step_12_scores_first_line_names_every_model_time_in_order():
    result = run_scores(TWO_MODELS, {"stage 1 (HGB)": 12.34, "baseline (LR)": 0.26})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == (
        "        fitted in stage 1 (HGB) 12.3 s, baseline (LR) 0.3 s; scored on 4 forecast "
        "rows (50.0% paid)")


def test_step_12_scores_fit_time_has_one_decimal():
    result = run_scores(fit_seconds={"stage 1 (HGB)": 0.04})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0].startswith("        fitted in stage 1 (HGB) 0.0 s; ")


def test_step_12_scores_table_by_hand():
    result = run_scores()
    assert result.returncode == 0, result.stderr
    header, table = score_table(result.stdout.splitlines(), 2)
    assert header == ["log", "loss", "Brier", "AUC"]
    assert list(table) == ["stage 1 (HGB)", "base rate"]
    assert table["stage 1 (HGB)"] == [0.367, 0.1, 1.0]
    assert table["base rate"] == [0.837, 0.3125, 0.5]


def test_step_12_scores_table_with_two_models_by_hand():
    result = run_scores(TWO_MODELS)
    assert result.returncode == 0, result.stderr
    header, table = score_table(result.stdout.splitlines(), 3)
    assert header == ["log", "loss", "Brier", "AUC"]
    assert list(table) == ["stage 1 (HGB)", "baseline (LR)", "base rate"]
    assert table["stage 1 (HGB)"] == [0.367, 0.1, 1.0]
    assert table["baseline (LR)"] == [1.1321, 0.405, 0.25]
    assert table["base rate"] == [0.837, 0.3125, 0.5]


def test_step_12_scores_rows_follow_the_order_of_the_predictions_dict():
    result = run_scores(list(reversed(TWO_MODELS)))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("        fitted in baseline (LR) 1.2 s, stage 1 (HGB) 1.2 s; ")
    _, table = score_table(lines, 3)
    assert list(table) == ["baseline (LR)", "stage 1 (HGB)", "base rate"]
    assert lines[6].split() == ["mean", "p,", "baseline", "(LR)", "mean", "p,", "stage", "1",
                                "(HGB)", "share", "paid"]


def test_step_12_scores_a_model_is_not_scored_on_another_models_p():
    # Two models, each scored on its own p: swapping them swaps their rows.
    one = run_scores(TWO_MODELS)
    two = run_scores([("stage 1 (HGB)", MODEL_B), ("baseline (LR)", MODEL_A)])
    assert one.returncode == 0 and two.returncode == 0, (one.stderr, two.stderr)
    _, first = score_table(one.stdout.splitlines(), 3)
    _, second = score_table(two.stdout.splitlines(), 3)
    assert first["stage 1 (HGB)"] == second["baseline (LR)"]
    assert first["baseline (LR)"] == second["stage 1 (HGB)"]


def test_step_12_scores_per_h_table_by_hand():
    # If the missing_week (p 0.99) or future (p 0.01) row were scored, mean p would move.
    result = run_scores()
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[4] == "        per h, mean p vs share that actually paid:"
    assert lines[5].split() == ["mean", "p,", "stage", "1", "(HGB)", "share", "paid"]
    assert lines[6].split() == ["h"]
    assert lines[7].split() == ["1", "0.6", "0.5"]
    assert lines[8].split() == ["2", "0.4", "0.5"]
    assert lines[9:] == ["ok"]


def test_step_12_scores_per_h_table_with_two_models_by_hand():
    # Model B's unscored rows are 0.01 and 0.99, the other way round from A's: if either
    # were included, its h's mean p would move.
    result = run_scores(TWO_MODELS)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[5] == "        per h, mean p vs share that actually paid:"
    assert lines[6].split() == ["mean", "p,", "stage", "1", "(HGB)", "mean", "p,", "baseline",
                                "(LR)", "share", "paid"]
    assert lines[7].split() == ["h"]
    assert lines[8].split() == ["1", "0.6", "0.6", "0.5"]
    assert lines[9].split() == ["2", "0.4", "0.5", "0.5"]
    assert lines[10:] == ["ok"]


def test_step_12_scores_per_h_table_has_no_base_rate_column():
    result = run_scores(TWO_MODELS)
    assert result.returncode == 0, result.stderr
    per_h = result.stdout.split("per h, mean p vs share that actually paid:")[1]
    assert "base rate" not in per_h


# The real chain on the tiny STAGE1 table: fit_stage1 and fit_stage1_baseline -> predict_paid
# -> show_stage1_scores. At origin 2 the forecast targets are weeks 3 (labelled) and 4
# (future, last week 3): two labelled rows, TR02-A paid and TR04-B not, so the score table
# is defined.
STAGE1_FIT = STAGE1.replace('print("ok")', """
from collection_estimation.hurdle.settings import Stage1Settings
settings = Stage1Settings(max_iter=5, learning_rate=0.1, max_leaf_nodes=3, min_samples_leaf=1,
                          random_seed=1, feature_blocks=("history", "calendar", "receivables"),
                          company_as_feature=COMPANY_AS_FEATURE, baseline_c=1.0)
forecast_x = script.stage1_features(forecast_rows, columns, companies, COMPANY_AS_FEATURE)
predictions = {}
for name, fit in (("stage 1 (HGB)", script.fit_stage1),
                  ("baseline (LR)", script.fit_stage1_baseline)):
    model = fit(train_x, train_rows["paid"], settings)
    predictions[name] = script.predict_paid(model, forecast_x)
    assert list(predictions[name].index) == list(forecast_rows.index)
    assert predictions[name].between(0, 1).all()
script.show_stage1_scores(forecast_rows, predictions, train_rows,
                          {"stage 1 (HGB)": 0.5, "baseline (LR)": 0.25})
print("ok")
""")


@pytest.mark.parametrize("company", [True, False])
def test_step_12_real_fit_predict_and_scores_on_the_tiny_table(company):
    script = STAGE1_FIT.replace("COMPANY_AS_FEATURE", str(company)).replace("ORIGIN", "2")
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    scores = [line for line in lines if line.startswith("        fitted in")]
    assert scores == ["        fitted in stage 1 (HGB) 0.5 s, baseline (LR) 0.2 s; scored on 2 "
                      "forecast rows (50.0% paid)"]
    start = lines.index("        per h, mean p vs share that actually paid:")
    assert [line.split()[0] for line in lines[start - 3:start]] == ["stage", "baseline", "base"]
    # Only h=1 is labelled at origin 2 (h=2 targets week 4, after the last week 3).
    assert [line.split()[0] for line in lines[start + 3:-1]] == ["1"]
    assert lines[-1] == "ok"


ONE_CLASS = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script

forecast_rows = pd.DataFrame({"key": list("abcd"), "h": np.array([1, 1, 2, 2]),
                              "label_status": ["labelled"] * 3 + ["missing_week"],
                              "paid": pd.array([True, True, True, None], dtype="boolean")},
                             index=[5, 6, 7, 8])
predictions = {"stage 1 (HGB)": pd.Series([0.7, 0.3, 0.6, 0.2], index=forecast_rows.index),
               "baseline (LR)": pd.Series([0.5, 0.5, 0.5, 0.5], index=forecast_rows.index)}
train_rows = pd.DataFrame({"paid": pd.array([True, False], dtype="boolean")})
script.show_stage1_scores(forecast_rows, predictions, train_rows,
                          {"stage 1 (HGB)": 0.1, "baseline (LR)": 0.1})
print("ok")
"""


def test_step_12_scores_with_one_class_among_labelled_rows_shows_nan_auc_and_runs():
    # sklearn 1.9's roc_auc_score warns and returns NaN for a single class; the printout
    # must still finish, for every model.
    result = subprocess.run([sys.executable, "-c", ONE_CLASS], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == ("        fitted in stage 1 (HGB) 0.1 s, baseline (LR) 0.1 s; scored on 3 "
                        "forecast rows (100.0% paid)")
    assert lines[2].split()[-1] == "NaN" and lines[2].startswith("stage 1 (HGB)")
    assert lines[3].split()[-1] == "NaN" and lines[3].startswith("baseline (LR)")
    assert lines[4].split()[-1] == "0.5" and lines[4].startswith("base rate")
    assert lines[-1] == "ok"


# No labelled forecast row: show_stage1_scores prints one line and returns. Every target week
# of the origin can be a missing week, after the last week (future), a mix of both, or there
# may be no forecast row at all.
NO_LABELLED = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script

statuses = STATUSES
forecast_rows = pd.DataFrame({
    "key": [f"TR02-{i}" for i in range(len(statuses))],
    "h": np.array([1 + i % 2 for i in range(len(statuses))], dtype="int64"),
    "label_status": pd.array(statuses, dtype=object),
    "paid": pd.array([None] * len(statuses), dtype="boolean"),
}, index=pd.Index(range(20, 20 + len(statuses))))
fit_seconds = FIT_SECONDS
predictions = {name: pd.Series(np.linspace(0.1, 0.9, len(statuses)), index=forecast_rows.index,
                               name="p") for name in fit_seconds}
train_rows = pd.DataFrame({"paid": pd.array([True, False, False], dtype="boolean")})
copies = (forecast_rows.copy(deep=True), {n: p.copy() for n, p in predictions.items()},
          train_rows.copy(deep=True), dict(fit_seconds))
assert list(forecast_rows["label_status"]) == statuses  # the frame holds what we meant
returned = script.show_stage1_scores(forecast_rows, predictions, train_rows, fit_seconds)
assert returned is None
pd.testing.assert_frame_equal(forecast_rows, copies[0])
for name in predictions:
    pd.testing.assert_series_equal(predictions[name], copies[1][name])
pd.testing.assert_frame_equal(train_rows, copies[2])
assert fit_seconds == copies[3]
print("ok")
"""

NO_LABELLED_CASES = {
    "all_missing_week": ["missing_week"] * 4,
    "all_future": ["future"] * 3,
    "missing_week_and_future": ["missing_week", "future", "missing_week", "future", "future"],
    "empty": [],
}

TWO_TIMES = {"stage 1 (HGB)": 2.345, "baseline (LR)": 0.06}


def run_no_labelled(case, fit_seconds=None):
    fit_seconds = TWO_TIMES if fit_seconds is None else fit_seconds
    script = (NO_LABELLED.replace("STATUSES", repr(NO_LABELLED_CASES[case]))
              .replace("FIT_SECONDS", repr(fit_seconds)))
    return subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True,
                          text=True, encoding="utf-8", timeout=120)


@pytest.mark.parametrize("case", list(NO_LABELLED_CASES))
def test_step_12_scores_without_labelled_rows_does_not_raise(case):
    result = run_no_labelled(case)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("case", list(NO_LABELLED_CASES))
def test_step_12_scores_without_labelled_rows_prints_only_the_one_line(case):
    # The exact line, then nothing from the function: the next line is the caller's "ok".
    result = run_no_labelled(case)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "        fitted in stage 1 (HGB) 2.3 s, baseline (LR) 0.1 s; no labelled forecast "
        "rows to score", "ok"]


@pytest.mark.parametrize("case", list(NO_LABELLED_CASES))
def test_step_12_scores_without_labelled_rows_prints_no_table(case):
    result = run_no_labelled(case)
    assert result.returncode == 0, result.stderr
    for word in ("scored on", "log loss", "Brier", "AUC", "base rate", "per h", "mean p"):
        assert word not in result.stdout, word


def test_step_12_scores_without_labelled_rows_fit_time_has_one_decimal():
    result = run_no_labelled("all_missing_week", fit_seconds={"stage 1 (HGB)": 0.04})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == (
        "        fitted in stage 1 (HGB) 0.0 s; no labelled forecast rows to score")


def test_step_12_scores_with_labelled_rows_does_not_print_the_no_rows_line():
    # The normal case (SCORES: four labelled rows) keeps its full printout.
    result = run_scores(TWO_MODELS)
    assert result.returncode == 0, result.stderr
    assert "no labelled forecast rows to score" not in result.stdout
    lines = result.stdout.splitlines()
    assert lines[0] == ("        fitted in stage 1 (HGB) 1.2 s, baseline (LR) 1.2 s; scored on 4 "
                        "forecast rows (50.0% paid)")
    assert lines[1].split() == ["log", "loss", "Brier", "AUC"]
    assert lines[2].startswith("stage 1 (HGB)") and lines[3].startswith("baseline (LR)")
    assert lines[4].startswith("base rate")
    assert lines[5] == "        per h, mean p vs share that actually paid:"
    assert len(lines) == 11 and lines[-1] == "ok"


def test_step_12_scores_with_one_labelled_row_among_unlabelled_still_scores():
    # The guard is on "no labelled row", not on "any unlabelled row".
    script = NO_LABELLED.replace(
        "STATUSES", repr(["missing_week", "labelled", "future"])).replace(
        "FIT_SECONDS", repr({"stage 1 (HGB)": 1.0}))
    script = script.replace('"paid": pd.array([None] * len(statuses), dtype="boolean")',
                            '"paid": pd.array([None, True, None], dtype="boolean")')
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True,
                            text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    assert "no labelled forecast rows to score" not in result.stdout
    assert result.stdout.splitlines()[0] == (
        "        fitted in stage 1 (HGB) 1.0 s; scored on 1 forecast rows (100.0% paid)")


# Step 7's second guard (U10.4a): when [diagnostic] run is on, its test weeks (up to origin +
# 2 horizons) must have answers. Run from main()'s own syntax tree, like the demo_origin guard.
# U11.5: step 15's joint diagnostic uses the same splits, so its run switch arms the guard too.
DIAGNOSTIC_END = "settings.diagnostic.origin + 2 * settings.receivables.horizons"
DIAGNOSTIC_GUARD_TEST = ("(settings.diagnostic.run or settings.joint_diagnostic.run) and "
                         "diagnostic_end > int(weeks['week'].max())")


def is_diagnostic_guard(node):
    return isinstance(node, ast.If) and ast.unparse(node.test) == DIAGNOSTIC_GUARD_TEST


def is_diagnostic_end(node):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "diagnostic_end"
            and ast.unparse(node.value) == DIAGNOSTIC_END)


def diagnostic_guard_statements():
    """main()'s top-level `diagnostic_end = ...` and the `if` that follows it."""
    body = main_tree().body
    ends = [i for i, node in enumerate(body) if is_diagnostic_end(node)]
    assert len(ends) == 1
    assert is_diagnostic_guard(body[ends[0] + 1]), ast.unparse(body[ends[0] + 1])
    return body[ends[0]], body[ends[0] + 1]


def run_diagnostic_guard(run, origin, horizons, last_week, demo_origin=0, week_dtype="int64",
                         weeks=None, with_demo_guard=False, joint_run=False):
    """Run the diagnostic guard (and, if asked, the demo_origin guard before it) with
    settings and a week calendar whose weeks run 0..last_week."""
    from types import SimpleNamespace
    import pandas as pd
    statements = list(diagnostic_guard_statements())
    if with_demo_guard:
        statements = [demo_origin_guard()] + statements
    module = ast.Module(body=statements, type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(SCRIPT), "exec")
    settings = SimpleNamespace(debug=SimpleNamespace(demo_origin=demo_origin),
                               diagnostic=SimpleNamespace(run=run, origin=origin),
                               joint_diagnostic=SimpleNamespace(run=joint_run),
                               receivables=SimpleNamespace(horizons=horizons))
    if weeks is None:
        weeks = pd.DataFrame({"week": pd.Series(range(0, last_week + 1), dtype=week_dtype)})
    exec(code, {"settings": settings, "weeks": weeks})


def diagnostic_guard_message(origin, end, last_week):
    return (f"[diagnostic] origin {origin}: its test weeks reach week {end}, after the last "
            f"week {last_week}")


def test_diagnostic_guard_raises_with_the_exact_message():
    with pytest.raises(ValueError) as error:
        run_diagnostic_guard(True, origin=50, horizons=5, last_week=59)
    assert str(error.value) == diagnostic_guard_message(50, 60, 59)


@pytest.mark.parametrize("origin, horizons, last_week", [(50, 5, 59), (55, 5, 64), (0, 1, 1),
                                                          (100, 5, 64), (60, 3, 64)])
def test_diagnostic_guard_raises_once_a_test_week_is_after_the_last_week(origin, horizons,
                                                                         last_week):
    with pytest.raises(ValueError) as error:
        run_diagnostic_guard(True, origin, horizons, last_week)
    assert str(error.value) == diagnostic_guard_message(origin, origin + 2 * horizons,
                                                        last_week)


@pytest.mark.parametrize("origin, horizons, last_week", [(49, 5, 59), (40, 5, 64), (54, 5, 64),
                                                          (0, 1, 2), (0, 0, 0)])
def test_diagnostic_guard_allows_test_weeks_up_to_the_last_week(origin, horizons, last_week):
    run_diagnostic_guard(True, origin, horizons, last_week)


@pytest.mark.parametrize("origin", [50, 60, 1000])
def test_diagnostic_guard_does_not_raise_when_run_is_false(origin):
    run_diagnostic_guard(False, origin, horizons=5, last_week=59)


# U11.5: the joint diagnostic's run switch arms the same guard, alone or with [diagnostic]'s.

@pytest.mark.parametrize("run, joint_run", [(False, True), (True, True), (True, False)])
def test_diagnostic_guard_raises_when_either_diagnostic_runs(run, joint_run):
    with pytest.raises(ValueError) as error:
        run_diagnostic_guard(run, origin=50, horizons=5, last_week=59, joint_run=joint_run)
    assert str(error.value) == diagnostic_guard_message(50, 60, 59)


@pytest.mark.parametrize("origin", [50, 60, 1000])
def test_diagnostic_guard_does_not_raise_when_neither_diagnostic_runs(origin):
    run_diagnostic_guard(False, origin, horizons=5, last_week=59, joint_run=False)


def test_diagnostic_guard_allows_the_joint_diagnostic_up_to_the_last_week():
    run_diagnostic_guard(False, origin=49, horizons=5, last_week=59, joint_run=True)


def test_diagnostic_guard_with_only_the_joint_run_uses_the_diagnostic_origin():
    # Step 15 has no origin of its own: [diagnostic] origin is checked even with run false.
    with pytest.raises(ValueError) as error:
        run_diagnostic_guard(False, origin=55, horizons=5, last_week=64, joint_run=True)
    assert str(error.value) == diagnostic_guard_message(55, 65, 64)


def test_diagnostic_guard_counts_two_horizons_not_one():
    # 55 + 5 = 60 <= 64 (step 12's rule would pass), but 55 + 10 = 65 > 64.
    with pytest.raises(ValueError) as error:
        run_diagnostic_guard(True, origin=55, horizons=5, last_week=64)
    assert str(error.value) == diagnostic_guard_message(55, 65, 64)


def test_diagnostic_guard_message_shows_plain_ints_for_numpy_weeks():
    with pytest.raises(ValueError) as error:
        run_diagnostic_guard(True, origin=100, horizons=3, last_week=64, week_dtype="int32")
    assert str(error.value) == diagnostic_guard_message(100, 106, 64)


def test_diagnostic_guard_uses_the_largest_week_not_the_last_row():
    import pandas as pd
    run_diagnostic_guard(True, origin=54, horizons=5, last_week=None,
                         weeks=pd.DataFrame({"week": [64, 0, 10]}))


def test_diagnostic_guard_does_not_use_demo_origin():
    run_diagnostic_guard(True, origin=40, horizons=5, last_week=64, demo_origin=63)
    with pytest.raises(ValueError):
        run_diagnostic_guard(True, origin=60, horizons=5, last_week=64, demo_origin=0)


def test_the_demo_origin_guard_runs_first_when_both_fail():
    with pytest.raises(ValueError) as error:
        run_diagnostic_guard(True, origin=60, horizons=5, last_week=64, demo_origin=62,
                             with_demo_guard=True)
    assert str(error.value).startswith("[debug] demo_origin 62: ")


def test_with_both_guards_a_good_demo_origin_and_a_bad_diagnostic_raise_the_diagnostic():
    with pytest.raises(ValueError) as error:
        run_diagnostic_guard(True, origin=60, horizons=5, last_week=64, demo_origin=40,
                             with_demo_guard=True)
    assert str(error.value) == diagnostic_guard_message(60, 70, 64)


def test_diagnostic_guard_comes_right_after_the_demo_origin_guard():
    body = main_tree().body
    guards = [i for i, node in enumerate(body) if is_demo_origin_guard(node)]
    assert len(guards) == 1
    assert is_diagnostic_end(body[guards[0] + 1]), ast.unparse(body[guards[0] + 1])
    assert is_diagnostic_guard(body[guards[0] + 2]), ast.unparse(body[guards[0] + 2])


def test_diagnostic_guard_is_a_single_top_level_statement_of_main():
    assert sum(is_diagnostic_guard(node) for node in main_tree().body) == 1
    assert sum(is_diagnostic_guard(node) for node in ast.walk(main_tree())) == 1


def test_diagnostic_guard_body_is_a_single_raise_of_value_error():
    _, guard = diagnostic_guard_statements()
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Raise)
    assert ast.unparse(guard.body[0].exc.func) == "ValueError"
    assert guard.orelse == []


def test_diagnostic_guard_comes_before_the_long_steps():
    source = main_source()
    guard = source.index("and diagnostic_end > int(weeks")
    assert source.index("show_weeks(weeks)") < guard
    for later in ("weekly = weekly_amounts(", "exchange_rates(", "receivables_by_week(",
                  "table = join_features(", "diagnostic_splits("):
        assert guard < source.index(later), later


# Step 13 (U10.4b/c): the stage 1 diagnostic. For HGB, then LR: grid_search on validation,
# choose_and_test on the test rows, the chosen setting refitted on train for the permutation
# importance on validation, then show_diagnostic_grid; after both, show_diagnostic_test and
# write_tables. (The U10.4a demo on step 12's predictions, with its "too few outcomes" skip,
# is gone; its tests went with it.)
#
# Wiring is checked on main()'s syntax tree. The block itself is run in a separate process
# (exec of main()'s own `if settings.diagnostic.run:` statement) on a tiny hand-made table:
# T = 4, H = 2, 30 rows per target week 0..9, every tenth row (i % 10 == 9) a missing_week
# row -> 27 labelled rows per week: train weeks 0..4 = 135 rows, validation 5..6 = 54, refit
# 189, test 7..8 = 54; week 9 is after the test weeks. Grids: HGB 2 x 1 x 1 x 2 = 4 settings
# of at most 10 trees, LR 2 x 2 = 4 settings. CSV output goes to pytest's tmp_path only.

STEP_13_SPLIT_TABLE = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle.diagnostic import diagnostic_splits

# T = 2, H = 2. Target weeks 0, 1, 2 (paid T, F, T) -> train 3 rows, 66.7% paid; 3 (T) and
# 4 (F) -> validation 2 rows, 50.0% (a missing_week row at week 3 is left out); refit 5 rows,
# 60.0%; 5, 6 (T, T) -> test 2 rows, 100.0%; week 7 is after the test weeks.
weeks = [0, 1, 2, 3, 3, 4, 5, 6, 7]
statuses = ["labelled"] * 4 + ["missing_week"] + ["labelled"] * 4
table = pd.DataFrame({
    "target_week": np.array(weeks, dtype="int64"),
    "label_status": np.array(statuses, dtype=object),
    "paid": pd.array([True, False, True, True, None, False, True, True, False],
                     dtype="boolean"),
})
script.show_diagnostic_splits(diagnostic_splits(table, 2, 2), 2, 2)
print("ok")
"""

STEP_13_SPLIT_LINES = [
    "Step 13: stage 1 diagnostic at origin 2 (diagnostic only, never the forecast)",
    "        train              3 rows,  66.7% paid (target <= 2)",
    "        validation         2 rows,  50.0% paid (2 < target <= 4)",
    "        refit              5 rows,  60.0% paid (target <= 4)",
    "        test               2 rows, 100.0% paid (4 < target <= 6)",
]


def run_code(code):
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                          text=True, encoding="utf-8", timeout=300)


def test_step_13_split_printout_by_hand():
    result = run_code(STEP_13_SPLIT_TABLE)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == STEP_13_SPLIT_LINES + ["ok"]


STEP_13 = """
import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle.settings import Stage1Settings

tree = ast.parse(textwrap.dedent(inspect.getsource(script.main)))
blocks = [node for node in tree.body[0].body
          if isinstance(node, ast.If) and ast.unparse(node.test) == "settings.diagnostic.run"]
assert len(blocks) == 1, len(blocks)
code = compile(ast.fix_missing_locations(ast.Module(body=blocks, type_ignores=[])),
               "step13", "exec")

rng = np.random.default_rng(5)
n = 300
weeks = np.repeat(np.arange(10), 30).astype("int64")
signal = rng.normal(size=n)
codes = rng.choice(["TR02", "TR04"], size=n)
paid = rng.random(n) < 1 / (1 + np.exp(-(2.0 * signal - 0.3)))
labelled = np.arange(n) % 10 != 9
table = pd.DataFrame({
    "key": [f"{c}-{i}" for i, c in enumerate(codes)],
    "company_code": codes,
    "origin_week": weeks - 1,
    "target_week": weeks,
    "label_status": np.where(labelled, "labelled", "missing_week").astype(object),
    "paid": pd.array([bool(p) if keep else None for p, keep in zip(paid, labelled)],
                     dtype="boolean"),
    "h": rng.integers(1, 3, size=n).astype("int64"),
    "signal": signal,
    "noise": rng.normal(size=n),
}, index=pd.RangeIndex(1000, 1000 + n))
# U10.4d: the window-13 history the "own rate" reference reads (drawn after everything else,
# so the rows above are unchanged); not a feature.
observed = rng.integers(0, 14, size=n)
table["observed_weeks_13"] = observed.astype("int64")
table["paying_weeks_13"] = rng.integers(0, observed + 1).astype("int64")
columns = ["h", "signal", "noise"]
companies = ["TR04", "TR02"]
settings = SimpleNamespace(
    diagnostic=SimpleNamespace(run=__RUN__, origin=4, hgb_max_iter=(3, 10),
                               hgb_learning_rate=(0.3,), hgb_max_leaf_nodes=(3,),
                               lr_c=(0.1, 10.0), class_weights=__WEIGHTS__,
                               selection_metric=__METRIC__, calibration_groups=4,
                               reference_window=13, reference_prior_weeks=1.0,
                               importance_repeats=2,
                               output_folder=Path(__OUTPUT__)),
    receivables=SimpleNamespace(horizons=2),
    stage1=Stage1Settings(max_iter=5, learning_rate=0.2, max_leaf_nodes=4, min_samples_leaf=5,
                          random_seed=3, feature_blocks=("history",),
                          company_as_feature=__COMPANY__, baseline_c=1.0),
)
copies = (table.copy(deep=True), settings.stage1)
namespace = dict(vars(script))
namespace.update(settings=settings, table=table, columns=columns, companies=companies)
__STUBS__
exec(code, namespace)
pd.testing.assert_frame_equal(table, copies[0])
assert settings.stage1 is copies[1]
assert settings.stage1.max_iter == 5 and settings.stage1.baseline_c == 1.0
__AFTER__
print("ok")
"""

from types import SimpleNamespace  # noqa: E402

TABLE_NAMES = ["hgb_grid", "hgb_importance", "lr_grid", "lr_importance", "calibration",
               "reliability", "test"]


def run_step_13(output, run=True, company=False, stubs="", after="", metric="avg precision",
                weights=("none", "balanced")):
    code = (STEP_13.replace("__RUN__", str(run)).replace("__OUTPUT__", repr(str(output)))
            .replace("__METRIC__", repr(metric)).replace("__WEIGHTS__", repr(tuple(weights)))
            .replace("__COMPANY__", str(company)).replace("__STUBS__", stubs)
            .replace("__AFTER__", after))
    return run_code(code)


@pytest.fixture(scope="module", params=[False, True], ids=["no company", "company"])
def real_run(request, tmp_path_factory):
    """The real block on the tiny table (with and without the company as a feature)."""
    output = tmp_path_factory.mktemp("step13") / "nested" / "hurdle_diagnostic"
    result = run_step_13(output, company=request.param)
    return SimpleNamespace(result=result, output=output, company=request.param,
                           lines=result.stdout.splitlines())


def test_step_13_real_run_succeeds(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    assert real_run.lines[-1] == "ok"


def test_step_13_real_run_prints_the_splits_first(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    lines = real_run.lines
    assert lines[0] == ("Step 13: stage 1 diagnostic at origin 4 (diagnostic only, never the "
                        "forecast)")
    for line, name, rows, weeks in zip(
            lines[1:5], ("train", "validation", "refit", "test"), (135, 54, 189, 54),
            ("(target <= 4)", "(4 < target <= 6)", "(target <= 6)", "(6 < target <= 8)")):
        assert line.startswith(f"        {name:<10} {rows:>9,} rows, "), line
        assert line.endswith(f"paid {weeks}"), line


def test_step_13_real_run_progress_lines_hgb_then_lr(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    progress = [line for line in real_run.lines if line.startswith("  ... ")]
    assert [line.split(":")[0] for line in progress] == (
        [f"  ... HGB {i} of 4" for i in range(1, 5)] + [f"  ... LR {i} of 4" for i in range(1, 5)])
    assert progress[0].startswith("  ... HGB 1 of 4: {'max_iter': 3, 'learning_rate': 0.3, "
                                  "'max_leaf_nodes': 3, 'class_weight': 'none'}, ")
    assert progress[3].startswith("  ... HGB 4 of 4: {'max_iter': 10, 'learning_rate': 0.3, "
                                  "'max_leaf_nodes': 3, 'class_weight': 'balanced'}, ")
    assert progress[4].startswith("  ... LR 1 of 4: {'C': 0.1, 'class_weight': 'none'}, ")
    assert progress[7].startswith("  ... LR 4 of 4: {'C': 10.0, 'class_weight': 'balanced'}, ")


def test_step_13_real_run_printout_order(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    lines = real_run.lines

    def first(prefix):
        matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        assert len(matches) == 1, (prefix, matches)
        return matches[0]

    order = [first("  ... HGB 4 of 4"),
             first("        HGB: the 5 best of 4 settings by validation avg precision"),
             first("        HGB: the 15 most important features"),
             first("        HGB: Platt correction fitted on validation: "),
             first("  ... LR 1 of 4"),
             first("        LR: the 5 best of 4 settings by validation avg precision"),
             first("        LR: the 15 most important features"),
             first("        LR: Platt correction fitted on validation: "),
             first("        CALIBRATION on test, all horizons"),
             first("        log-loss skill per horizon, on test:"),
             first("        TEST (target weeks after the validation weeks)"),
             first("        tables written to ")]
    assert order == sorted(order)
    assert order[-1] == len(lines) - 2   # then "ok"


def test_step_13_real_run_reports_the_files_written(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    expected = (f"        tables written to {real_run.output}: "
                f"{[name + '.csv' for name in TABLE_NAMES]}")
    assert real_run.lines[-2] == expected


def test_step_13_real_run_writes_the_seven_tables_into_a_new_folder(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    assert real_run.output.is_dir()
    assert sorted(path.name for path in real_run.output.iterdir()) == sorted(
        name + ".csv" for name in TABLE_NAMES)


def test_step_13_real_run_grid_tables(real_run):
    import pandas as pd
    assert real_run.result.returncode == 0, real_run.result.stderr
    hgb = pd.read_csv(real_run.output / "hgb_grid.csv")
    lr = pd.read_csv(real_run.output / "lr_grid.csv")
    scores = ["F1", "precision", "recall", "threshold", "AUC", "avg precision", "log loss",
              "brier"]
    assert list(hgb.columns) == (["candidate", "max_iter", "learning_rate", "max_leaf_nodes",
                                  "class_weight"] + scores + ["fit seconds"])
    assert list(lr.columns) == ["candidate", "C", "class_weight"] + scores + ["fit seconds"]
    for grid in (hgb, lr):
        assert sorted(grid["candidate"]) == [0, 1, 2, 3]
        assert grid["avg precision"].is_monotonic_decreasing


def test_step_13_real_run_importance_tables(real_run):
    import pandas as pd
    assert real_run.result.returncode == 0, real_run.result.stderr
    features = ["h", "signal", "noise"] + (["company_code"] if real_run.company else [])
    for name in ("hgb_importance", "lr_importance"):
        table = pd.read_csv(real_run.output / f"{name}.csv")
        assert list(table.columns) == ["feature", "log loss rise", "std"]
        assert sorted(table["feature"]) == sorted(features)
        assert table["log loss rise"].is_monotonic_decreasing
        # paid follows `signal` only: it is the most important feature for both models.
        assert table.loc[0, "feature"] == "signal", (name, table)


def test_step_13_real_run_test_table_uses_each_models_validation_threshold(real_run):
    import pandas as pd
    assert real_run.result.returncode == 0, real_run.result.stderr
    test = pd.read_csv(real_run.output / "test.csv")
    assert test["model"].tolist() == ["HGB", "LR"]
    assert list(test.columns[:1]) == ["model"]
    for model, name in (("HGB", "hgb_grid"), ("LR", "lr_grid")):
        top = pd.read_csv(real_run.output / f"{name}.csv").iloc[0]
        row = test[test["model"] == model].iloc[0]
        assert row["threshold"] == pytest.approx(top["threshold"], rel=1e-12)
        assert row["class_weight"] == top["class_weight"]
    hgb = pd.read_csv(real_run.output / "hgb_grid.csv").iloc[0]
    assert test.loc[0, "max_iter"] == hgb["max_iter"]
    lr = pd.read_csv(real_run.output / "lr_grid.csv").iloc[0]
    assert test.loc[1, "C"] == pytest.approx(lr["C"])


def test_step_13_with_run_false_calls_nothing_and_writes_nothing(tmp_path):
    output = tmp_path / "never"
    refuse = ("def refuse(*args, **kwargs):\n"
              "    raise AssertionError('called although [diagnostic] run is false')\n"
              "for name in ('diagnostic_splits', 'show_diagnostic_splits', 'stage1_features', "
              "'hgb_candidates', 'lr_candidates', 'grid_search', 'choose_and_test', "
              "'permutation_importances', 'show_diagnostic_grid', 'show_diagnostic_test', "
              "'write_tables', 'fit_stage1', 'fit_stage1_baseline', 'best_unweighted', "
              "'fit_platt', 'apply_platt', 'predict_paid', 'reference_probabilities', "
              "'calibration_comparison', 'show_calibration'):\n"
              "    namespace[name] = refuse\n")
    result = run_step_13(output, run=False, stubs=refuse)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["ok"]
    assert not output.exists()


def test_step_13_the_run_false_stubs_bite_when_run_is_true(tmp_path):
    refuse = ("def refuse(*args, **kwargs):\n"
              "    raise AssertionError('called although [diagnostic] run is false')\n"
              "namespace['grid_search'] = refuse\n")
    result = run_step_13(tmp_path / "out", run=True, stubs=refuse)
    assert result.returncode != 0
    assert "called although [diagnostic] run is false" in result.stderr


# The spy run: every diagnostic function and both fits are replaced by recorders, so the
# block's data flow is checked call by call. The stubbed grid_search puts candidate 1 on top
# for HGB (max_iter 3, balanced) and candidate 3 for LR (C 10, balanced) by default; TOPS can
# be changed per test (spy_stubs). U10.4d: best_unweighted is the real one behind a recorder
# (SKIP_FIRST makes its first call return None), predict_paid, fit_platt, apply_platt,
# reference_probabilities, calibration_comparison and show_calibration are recorders.
# HGB candidates: 0 (3, none), 1 (3, balanced), 2 (10, none), 3 (10, balanced); LR candidates:
# 0 (C .1, none), 1 (.1, balanced), 2 (10, none), 3 (10, balanced).
SPY_STUBS = """
from collection_estimation.hurdle import diagnostic
from collection_estimation.hurdle.stage1 import stage1_features
calls = []
real = {"fit_stage1": script.fit_stage1, "fit_stage1_baseline": script.fit_stage1_baseline}
TOPS = __TOPS__
SKIP_FIRST = __SKIP_FIRST__
PLATT = [(1.5, -0.25), (0.75, 0.125)]   # (a, b) of the first and second fit_platt call

def make_fit(label):
    def fit(train_x, paid, settings_, class_weight="NOT GIVEN"):
        calls.append(("fit", label, train_x, paid, settings_, class_weight))
        return ("model", label, len(calls))
    fit.label = label
    return fit

def spy_grid_search(name, fit, candidates, train_x, train_paid, validation_x, validation_paid,
                    *, metric):
    calls.append(("grid_search", name, fit.label, candidates, train_x, train_paid,
                  validation_x, validation_paid, metric))
    return pd.DataFrame({"candidate": TOPS[name], "threshold": [0.11, 0.22, 0.33, 0.44],
                         "avg precision": [0.9, 0.8, 0.7, 0.6]})

def spy_choose_and_test(name, fit, candidates, grid, refit_x, refit_paid, test_x, test_paid):
    calls.append(("choose_and_test", name, fit.label, candidates, grid, refit_x, refit_paid,
                  test_x, test_paid))
    return {"model": name, "F1": 0.5 if name == "HGB" else 0.25}

def spy_importances(model, x, paid, repeats, seed):
    calls.append(("permutation_importances", model, x, paid, repeats, seed))
    return pd.DataFrame({"feature": list(x.columns), "log loss rise": 0.0, "std": 0.0})

def spy_show_grid(name, grid, importance, metric):
    calls.append(("show_diagnostic_grid", name, grid, importance, metric))

def spy_show_test(test):
    calls.append(("show_diagnostic_test", test))

def spy_write_tables(folder, tables):
    calls.append(("write_tables", folder, tables))
    return [Path(folder) / f"{name}.csv" for name in tables]

def spy_best_unweighted(grid, candidates):
    result = diagnostic.best_unweighted(grid, candidates)
    if SKIP_FIRST and not any(call[0] == "best_unweighted" for call in calls):
        result = None
    calls.append(("best_unweighted", grid, candidates, result))
    return result

def spy_predict_paid(model, x):
    p = pd.Series(np.linspace(0.1, 0.9, len(x)), index=x.index)
    calls.append(("predict_paid", model, x, p))
    return p

def spy_fit_platt(actual, p):
    calls.append(("fit_platt", actual, p))
    return PLATT[sum(1 for call in calls if call[0] == "fit_platt") - 1]

def spy_apply_platt(p, a, b):
    corrected = p * 0 + 0.5
    calls.append(("apply_platt", p, a, b, corrected))
    return corrected

def spy_references(train_rows, rows, window, prior_weeks):
    references = pd.DataFrame({"base rate": 0.4, "base rate per h": 0.45, "own rate": 0.5},
                              index=rows.index)
    calls.append(("reference_probabilities", train_rows, rows, window, prior_weeks, references))
    return references

def spy_comparison(actual, h, predictions, reference_p, groups):
    summary = pd.DataFrame({"model": list(predictions), "h": "all",
                            "log loss skill": np.arange(len(predictions), dtype=float)})
    reliability = pd.DataFrame({"model": list(predictions), "h": "all", "group": 1})
    calls.append(("calibration_comparison", actual, h, dict(predictions), reference_p, groups,
                  summary, reliability))
    return summary, reliability

def spy_show_calibration(calibration):
    calls.append(("show_calibration", calibration))

namespace.update(fit_stage1=make_fit("HGB fit"), fit_stage1_baseline=make_fit("LR fit"),
                 grid_search=spy_grid_search, choose_and_test=spy_choose_and_test,
                 permutation_importances=spy_importances, show_diagnostic_grid=spy_show_grid,
                 show_diagnostic_test=spy_show_test, write_tables=spy_write_tables,
                 best_unweighted=spy_best_unweighted, predict_paid=spy_predict_paid,
                 fit_platt=spy_fit_platt, apply_platt=spy_apply_platt,
                 reference_probabilities=spy_references,
                 calibration_comparison=spy_comparison,
                 show_calibration=spy_show_calibration)
"""

DEFAULT_TOPS = {"HGB": [1, 0, 2, 3], "LR": [3, 2, 1, 0]}


def spy_stubs(tops=None, skip_first=False):
    return (SPY_STUBS.replace("__TOPS__", repr(tops or DEFAULT_TOPS))
            .replace("__SKIP_FIRST__", str(skip_first)))


SPY_CHECKS = """
splits = diagnostic.diagnostic_splits(table, 4, 2)
def expected(part):
    rows = getattr(splits, part)
    return (stage1_features(rows, columns, companies, settings.stage1.company_as_feature),
            rows["paid"].astype(bool))
def same(pair, part):
    x, paid = pair
    want_x, want_paid = expected(part)
    pd.testing.assert_frame_equal(x, want_x)
    pd.testing.assert_series_equal(paid, want_paid)
    assert paid.dtype == bool
def position(call):
    return next(i for i, other in enumerate(calls) if other is call)

queue = list(calls)
def take(kind):
    assert queue, f"no call left where {kind} was expected"
    call = queue.pop(0)
    assert call[0] == kind, (kind, call[0], [c[0] for c in calls])
    return call

hgb_candidates = diagnostic.hgb_candidates(settings.stage1, settings.diagnostic)
lr_candidates = diagnostic.lr_candidates(settings.stage1, settings.diagnostic)
grids, importances, tests_shown, stored, calibrated, refitted = {}, {}, {}, {}, [], []
stored = {"validation": {}, "test": {}}
for name, label, candidates in (("HGB", "HGB fit", hgb_candidates),
                                ("LR", "LR fit", lr_candidates)):
    top = TOPS[name][0]
    grid_call = take("grid_search")
    # grid_search: this model's fit and candidates, the train and validation inputs.
    assert grid_call[1] == name and grid_call[2] == label
    assert [c.params for c in grid_call[3]] == [c.params for c in candidates]
    assert [c.settings for c in grid_call[3]] == [c.settings for c in candidates]
    same(grid_call[4:6], "train")
    same(grid_call[6:8], "validation")
    # ... ranked by the settings' selection metric, passed by keyword.
    assert grid_call[8] == settings.diagnostic.selection_metric, grid_call[8]
    # choose_and_test: the same candidates and the grid just returned, refit and test inputs.
    test_call = take("choose_and_test")
    assert test_call[1] == name and test_call[2] == label
    assert test_call[3] is grid_call[3]
    assert test_call[4]["candidate"].tolist() == TOPS[name]
    same(test_call[5:7], "refit")
    same(test_call[7:9], "test")
    # The importance model: the grid's top candidate, refitted on the TRAIN rows.
    fit_call = take("fit")
    assert fit_call[1] == label
    same(fit_call[2:4], "train")
    assert fit_call[4] == candidates[top].settings
    assert fit_call[5] == diagnostic.class_weight_argument(candidates[top].class_weight)
    model = ("model", label, position(fit_call) + 1)
    # Its permutation importance on the VALIDATION rows, with the settings' repeats and seed.
    importance_call = take("permutation_importances")
    assert importance_call[1] == model
    same(importance_call[2:4], "validation")
    assert importance_call[4] == 2 and importance_call[5] == 3
    # The printout gets this model's grid and importance.
    show_call = take("show_diagnostic_grid")
    assert show_call[1] == name
    assert show_call[2] is test_call[4]
    assert show_call[3]["feature"].tolist() == list(expected("validation")[0].columns)
    assert show_call[4] == settings.diagnostic.selection_metric, show_call[4]
    grids[name], importances[name] = show_call[2], show_call[3]
    # U10.4d: the best unweighted setting of this grid.
    best_call = take("best_unweighted")
    assert best_call[1] is test_call[4] and best_call[2] is grid_call[3]
    unweighted = best_call[3]
    if unweighted is None:
        continue
    assert unweighted.class_weight == "none"
    chosen = grid_call[3][top]
    if unweighted is not chosen:
        # Refitted on the TRAIN rows with the unweighted settings and no class_weight.
        refit = take("fit")
        assert refit[1] == label
        same(refit[2:4], "train")
        assert refit[4] is unweighted.settings
        assert refit[5] == "NOT GIVEN", refit[5]
        model = ("model", label, position(refit) + 1)
        refitted.append(name)
    # Platt: fitted on this model's validation p against the validation outcomes.
    platt_predict = take("predict_paid")
    assert platt_predict[1] == model, (platt_predict[1], model)
    pd.testing.assert_frame_equal(platt_predict[2], expected("validation")[0])
    platt_call = take("fit_platt")
    pd.testing.assert_series_equal(platt_call[1], expected("validation")[1])
    assert platt_call[2] is platt_predict[3]
    a, b = PLATT[len(calibrated)]
    for part in ("validation", "test"):
        predict_call = take("predict_paid")
        assert predict_call[1] == model
        pd.testing.assert_frame_equal(predict_call[2], expected(part)[0])
        apply_call = take("apply_platt")
        assert apply_call[1] is predict_call[3]
        assert (apply_call[2], apply_call[3]) == (a, b)
        stored[part][name] = predict_call[3]
        stored[part][name + " + Platt"] = apply_call[4]
    calibrated.append(name)

REFERENCES = ["base rate", "base rate per h", "own rate"]
comparisons = {}
for part in ("validation", "test"):
    reference_call = take("reference_probabilities")
    pd.testing.assert_frame_equal(reference_call[1], splits.train)
    pd.testing.assert_frame_equal(reference_call[2], getattr(splits, part))
    assert reference_call[3] == settings.diagnostic.reference_window
    assert reference_call[4] == settings.diagnostic.reference_prior_weeks
    references = reference_call[5]
    comparison_call = take("calibration_comparison")
    pd.testing.assert_series_equal(comparison_call[1], expected(part)[1])
    pd.testing.assert_series_equal(comparison_call[2], getattr(splits, part)["h"])
    predictions = comparison_call[3]
    assert list(predictions) == list(stored[part]) + REFERENCES, list(predictions)
    for label, p in stored[part].items():
        assert predictions[label] is p, label
    for column in REFERENCES:
        pd.testing.assert_series_equal(predictions[column], references[column])
    pd.testing.assert_series_equal(comparison_call[4], references["base rate"])
    assert comparison_call[5] == settings.diagnostic.calibration_groups
    comparisons[part] = comparison_call

show_calibration_call = take("show_calibration")
show_test_call = take("show_diagnostic_test")
write_call = take("write_tables")
assert queue == [], [call[0] for call in queue]

test_table = show_test_call[1]
pd.testing.assert_frame_equal(test_table, pd.DataFrame([{"model": "HGB", "F1": 0.5},
                                                        {"model": "LR", "F1": 0.25}]))
folder, tables = write_call[1], write_call[2]
assert folder == settings.diagnostic.output_folder
assert list(tables) == ["hgb_grid", "hgb_importance", "lr_grid", "lr_importance",
                        "calibration", "reliability", "test"], list(tables)
assert tables["hgb_grid"] is grids["HGB"] and tables["lr_grid"] is grids["LR"]
assert tables["hgb_importance"] is importances["HGB"]
assert tables["lr_importance"] is importances["LR"]
assert tables["test"] is test_table
# The calibration tables: validation then test, each with "part" put first.
calibration, reliability = tables["calibration"], tables["reliability"]
assert show_calibration_call[1] is calibration
labels = list(stored["validation"]) + REFERENCES
assert calibration["part"].tolist() == ["validation"] * len(labels) + ["test"] * len(labels)
assert calibration["model"].tolist() == labels * 2
assert list(calibration.columns) == ["part", "model", "h", "log loss skill"]
assert list(reliability.columns) == ["part", "model", "h", "group"]
assert reliability["part"].tolist() == ["validation"] * len(labels) + ["test"] * len(labels)
assert list(calibration.index) == list(range(2 * len(labels)))
pd.testing.assert_frame_equal(
    calibration, pd.concat([comparisons["validation"][6], comparisons["test"][6]],
                           ignore_index=True))
pd.testing.assert_frame_equal(
    reliability, pd.concat([comparisons["validation"][7], comparisons["test"][7]],
                           ignore_index=True))
"""


@pytest.mark.parametrize("company", [False, True], ids=["no company", "company"])
def test_step_13_data_flow_call_by_call(company, tmp_path):
    after = SPY_CHECKS + ("assert calibrated == ['HGB', 'LR'], calibrated\n"
                          "assert refitted == ['HGB', 'LR'], refitted\n")
    result = run_step_13(tmp_path / "spy", company=company, stubs=spy_stubs(), after=after)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "ok"
    # Only the splits come from the real functions; the stubs print nothing.
    assert result.stdout.splitlines()[0].startswith("Step 13: stage 1 diagnostic at origin 4")
    assert not (tmp_path / "spy").exists()   # write_tables was stubbed


def test_step_13_spy_stubs_bite():
    # The spy checks fail when the recorded calls are wrong (so the data-flow test means
    # something): here the recorded calls are emptied before the checks run.
    result = run_step_13(REPO_ROOT / "never_written", stubs=spy_stubs(),
                         after="calls.clear()\n" + SPY_CHECKS)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


# --- U10.4d: step 13's calibration, by the spy run ---------------------------------------------

def test_step_13_refits_the_unweighted_setting_when_the_chosen_one_is_weighted(tmp_path):
    # Default tops: HGB's chosen 1 and LR's chosen 3 are balanced, so the best unweighted
    # (HGB 0, LR 2) is refitted on train without a class_weight (checked in SPY_CHECKS).
    after = SPY_CHECKS + ("assert refitted == ['HGB', 'LR']\n"
                          "assert [c[4] for c in calls if c[0] == 'fit'] == [\n"
                          "    hgb_candidates[1].settings, hgb_candidates[0].settings,\n"
                          "    lr_candidates[3].settings, lr_candidates[2].settings]\n")
    result = run_step_13(tmp_path / "spy", stubs=spy_stubs(), after=after)
    assert result.returncode == 0, result.stderr


def test_step_13_reuses_the_importance_model_when_the_chosen_setting_is_unweighted(tmp_path):
    # HGB's top 0 and LR's top 2 are unweighted: no second fit, Platt and the p come from the
    # importance model itself.
    tops = {"HGB": [0, 1, 2, 3], "LR": [2, 3, 1, 0]}
    after = SPY_CHECKS + ("assert refitted == [], refitted\n"
                          "assert calibrated == ['HGB', 'LR']\n"
                          "fits = [c for c in calls if c[0] == 'fit']\n"
                          "assert len(fits) == 2, len(fits)\n"
                          "models = {c[1] for c in calls if c[0] == 'predict_paid'}\n"
                          "assert models == {('model', 'HGB fit', position(fits[0]) + 1),\n"
                          "                  ('model', 'LR fit', position(fits[1]) + 1)}\n")
    result = run_step_13(tmp_path / "spy", stubs=spy_stubs(tops), after=after)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "ok"


def test_step_13_mixed_reuse_and_refit(tmp_path):
    # HGB's chosen setting is unweighted (reuse), LR's is weighted (refit).
    tops = {"HGB": [2, 3, 1, 0], "LR": [1, 0, 2, 3]}
    after = SPY_CHECKS + "assert refitted == ['LR'], refitted\n"
    result = run_step_13(tmp_path / "spy", stubs=spy_stubs(tops), after=after)
    assert result.returncode == 0, result.stderr


def test_step_13_platt_labels_and_printout(tmp_path):
    after = SPY_CHECKS + (
        "assert list(stored['validation']) == ['HGB', 'HGB + Platt', 'LR', 'LR + Platt']\n"
        "assert list(stored['test']) == list(stored['validation'])\n")
    result = run_step_13(tmp_path / "spy", stubs=spy_stubs(), after=after)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    platt = [line for line in lines if "Platt correction" in line]
    assert platt == ["        HGB: Platt correction fitted on validation: a 1.5000, b -0.2500",
                     "        LR: Platt correction fitted on validation: a 0.7500, b 0.1250"]
    assert not any("calibration skipped" in line for line in lines)


def test_step_13_skips_the_calibration_of_a_model_without_an_unweighted_setting(tmp_path):
    # best_unweighted returns None for HGB only: HGB gets no Platt, no p, no extra fit; LR
    # still runs (the loop continues), and the references are still added for both parts.
    after = SPY_CHECKS + (
        "assert calibrated == ['LR'], calibrated\n"
        "assert list(stored['validation']) == ['LR', 'LR + Platt']\n"
        "assert [c[1] for c in calls if c[0] == 'grid_search'] == ['HGB', 'LR']\n"
        "assert [c[1] for c in calls if c[0] == 'fit'].count('HGB fit') == 1\n"
        "for part in ('validation', 'test'):\n"
        "    assert list(comparisons[part][3]) == ['LR', 'LR + Platt', 'base rate',\n"
        "                                          'base rate per h', 'own rate']\n")
    result = run_step_13(tmp_path / "spy", stubs=spy_stubs(skip_first=True), after=after)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "        HGB: the grid has no unweighted setting; calibration skipped" in lines
    assert not any(line.startswith("        LR: the grid has no unweighted") for line in lines)
    assert [line for line in lines if "Platt correction" in line] == [
        "        LR: Platt correction fitted on validation: a 1.5000, b -0.2500"]


def test_step_13_references_are_added_for_both_parts(tmp_path):
    after = SPY_CHECKS + (
        "refs = [c for c in calls if c[0] == 'reference_probabilities']\n"
        "assert len(refs) == 2\n"
        "assert [len(c[2]) for c in refs] == [len(splits.validation), len(splits.test)]\n"
        "for part in ('validation', 'test'):\n"
        "    assert list(comparisons[part][3])[-3:] == REFERENCES\n")
    result = run_step_13(tmp_path / "spy", stubs=spy_stubs(), after=after)
    assert result.returncode == 0, result.stderr


def test_step_13_the_calibration_and_reliability_tables_are_written(tmp_path):
    after = SPY_CHECKS + ("assert 'calibration' in tables and 'reliability' in tables\n"
                          "assert tables['calibration'] is show_calibration_call[1]\n")
    result = run_step_13(tmp_path / "spy", stubs=spy_stubs(), after=after)
    assert result.returncode == 0, result.stderr



STEP_13_WIRING = """
import ast
import inspect
import textwrap
import run_hurdle_pipeline as script
from collection_estimation.hurdle import diagnostic

for name in ("diagnostic_splits", "grid_search", "choose_and_test", "class_weight_argument",
             "hgb_candidates", "lr_candidates", "permutation_importances", "write_tables",
             "best_unweighted", "fit_platt", "apply_platt", "reference_probabilities",
             "calibration_comparison"):
    assert getattr(script, name) is getattr(diagnostic, name), name
inspect.signature(script.show_calibration).bind("calibration")
# The U10.4a demo's functions are no longer imported.
assert not hasattr(script, "best_threshold")
assert not hasattr(script, "classification_scores")
inspect.signature(script.show_diagnostic_splits).bind("splits", 40, 5)
inspect.signature(script.show_diagnostic_grid).bind("name", "grid", "importance", "metric")
inspect.signature(script.show_diagnostic_test).bind("test")

body = ast.parse(textwrap.dedent(inspect.getsource(script.main))).body[0].body
statements = [ast.unparse(node) for node in body]
blocks = [i for i, node in enumerate(body)
          if isinstance(node, ast.If) and ast.unparse(node.test) == "settings.diagnostic.run"]
assert len(blocks) == 1, blocks
block = body[blocks[0]]
scores = statements.index(
    "show_stage1_scores(forecast_rows, predictions, train_rows, fit_seconds)")
assert scores < blocks[0], (scores, blocks)
assert block.orelse == []
inner = [ast.unparse(node) for node in block.body]
assert inner[:2] == [
    "splits = diagnostic_splits(table, settings.diagnostic.origin, "
    "settings.receivables.horizons)",
    "show_diagnostic_splits(splits, settings.diagnostic.origin, settings.receivables.horizons)",
], inner[:2]
text = ast.unparse(block)
# The diagnostic uses its own origin and rows, never step 12's demo origin or predictions.
for name in ("demo_origin", "predictions", "forecast_rows", "train_rows", "forecast_x",
             "best_threshold", "classification_scores"):
    assert name not in text, name
# The test rows are used once, by choose_and_test; the importance is measured on validation.
assert text.count("inputs['test']") == 1, text
assert "permutation_importances(model, *inputs['validation']" in text
assert "settings.diagnostic.importance_repeats, settings.stage1.random_seed)" in text
assert "write_tables(settings.diagnostic.output_folder, tables)" in text
# The grid is ranked, and the printout labelled, by the settings' selection metric.
assert "*inputs['validation'], metric=settings.diagnostic.selection_metric)" in text, text
assert ("show_diagnostic_grid(name, grid, importance, settings.diagnostic.selection_metric)"
        in text), text
# The two models, in this order, with their fits and candidate lists.
loops = [node for node in ast.walk(block) if isinstance(node, ast.For)
         and ast.unparse(node.target) == "(name, fit, candidates)"]
assert len(loops) == 1, len(loops)
items = [[ast.unparse(e) for e in item.elts] for item in loops[0].iter.elts]
assert items == [
    ["'HGB'", "fit_stage1", "hgb_candidates(settings.stage1, settings.diagnostic)"],
    ["'LR'", "fit_stage1_baseline", "lr_candidates(settings.stage1, settings.diagnostic)"],
], items
print("ok")
"""


def test_step_13_is_wired_in_after_step_12s_scores():
    result = run_code(STEP_13_WIRING)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_step_13_each_diagnostic_function_is_called_once_in_main():
    tree = main_tree()
    for name in ("diagnostic_splits", "show_diagnostic_splits", "hgb_candidates",
                 "lr_candidates", "grid_search", "choose_and_test", "permutation_importances",
                 "show_diagnostic_grid", "show_diagnostic_test", "write_tables",
                 "best_unweighted", "fit_platt", "apply_platt", "reference_probabilities",
                 "calibration_comparison", "show_calibration"):
        assert len(calls_to(tree, name)) == 1, name
    for gone in ("best_threshold", "classification_scores"):
        assert calls_to(tree, gone) == [], gone


def test_step_13_old_too_few_outcomes_skip_is_gone():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "too few outcomes" not in source
    assert "at step 12's origin, best F1 threshold" not in source


# show_diagnostic_grid / show_diagnostic_test on tiny hand-made frames.
SHOW = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script

grid = pd.DataFrame({
    "candidate": [3, 0, 6, 1, 5, 2, 4][:N_GRID],
    "C": [10.0, 0.01, 1.0, 0.1, 100.0, 0.5, 2.0][:N_GRID],
    "class_weight": ["balanced", "none", "none", "balanced", "none", "balanced", "none"][:N_GRID],
    "avg precision": [0.987654321, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05][:N_GRID],
    "fit seconds": [1.23456, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0][:N_GRID],
})
importance = pd.DataFrame({"feature": [f"f{i}" for i in range(N_IMPORTANCE)],
                           "log loss rise": np.linspace(0.0123456789, -0.001, N_IMPORTANCE),
                           "std": np.full(N_IMPORTANCE, 0.000123456)})
copies = (grid.copy(deep=True), importance.copy(deep=True))
WHAT
pd.testing.assert_frame_equal(grid, copies[0])
pd.testing.assert_frame_equal(importance, copies[1])
print("ok")
"""


def run_show(what, n_grid=7, n_importance=20):
    return run_code(SHOW.replace("N_GRID", str(n_grid))
                    .replace("N_IMPORTANCE", str(n_importance)).replace("WHAT", what))


def expected_grid_printout(name, n_grid, n_importance, metric="avg precision"):
    import numpy as np
    import pandas as pd
    grid = pd.DataFrame({
        "C": [10.0, 0.01, 1.0, 0.1, 100.0, 0.5, 2.0][:n_grid],
        "class_weight": ["balanced", "none", "none", "balanced", "none", "balanced",
                         "none"][:n_grid],
        "avg precision": [0.987654321, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05][:n_grid],
        "fit seconds": [1.23456, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0][:n_grid],
    })
    importance = pd.DataFrame({"feature": [f"f{i}" for i in range(n_importance)],
                               "log loss rise": np.linspace(0.0123456789, -0.001, n_importance),
                               "std": np.full(n_importance, 0.000123456)})
    return ([f"        {name}: the 5 best of {n_grid} settings by validation {metric}"]
            + grid.head(5).round(4).to_string(index=False).splitlines()
            + [f"        {name}: the 15 most important features (rise in validation log loss "
               f"when shuffled)"]
            + importance.head(15).round(5).to_string(index=False).splitlines())


def test_show_diagnostic_grid_printout():
    result = run_show("script.show_diagnostic_grid('LR', grid, importance, 'avg precision')")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines == expected_grid_printout("LR", 7, 20) + ["ok"]


def test_show_diagnostic_grid_drops_the_candidate_column_and_rounds():
    result = run_show("script.show_diagnostic_grid('HGB', grid, importance, 'avg precision')")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    header = lines[1]
    assert "candidate" not in header
    assert header.split() == ["C", "class_weight", "avg", "precision", "fit", "seconds"]
    assert len(lines) == 1 + 1 + 5 + 1 + 1 + 15 + 1
    assert "0.9877" in lines[2] and "1.2346" in lines[2]
    assert "0.987654" not in result.stdout
    assert "0.01235" in lines[9]          # importance rounded to 5 places
    assert "f14" in lines[-2] and "f15" not in result.stdout


def test_show_diagnostic_grid_with_fewer_rows_than_shown_prints_them_all():
    result = run_show("script.show_diagnostic_grid('LR', grid, importance, 'avg precision')",
                      n_grid=2, n_importance=3)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines == expected_grid_printout("LR", 2, 3) + ["ok"]
    # The header still says "the 5 best" (pinned: it states the cap, not the count shown).
    assert lines[0] == "        LR: the 5 best of 2 settings by validation avg precision"


def test_show_diagnostic_test_printout():
    code = ("test = pd.DataFrame([{'model': 'HGB', 'max_iter': 300, 'F1': 0.123456, "
            "'threshold': 0.55555}, {'model': 'LR', 'max_iter': np.nan, 'F1': 0.5, "
            "'threshold': 0.25}])\n"
            "before = test.copy(deep=True)\n"
            "script.show_diagnostic_test(test)\n"
            "pd.testing.assert_frame_equal(test, before)\n"
            "print('EXPECTED')\n"
            "print(test.round(4).to_string(index=False))\n")
    result = run_show(code)
    assert result.returncode == 0, result.stderr
    out, expected = result.stdout.split("EXPECTED\n")
    lines = out.splitlines()
    assert lines[0] == ("        TEST (target weeks after the validation weeks), each model's "
                        "chosen setting at its validation threshold:")
    assert lines[1:] == expected.splitlines()[:-1]   # minus the final "ok"
    assert "0.1235" in lines[2] and "0.5556" in lines[2]
    assert lines[2].split()[0] == "HGB" and lines[3].split()[0] == "LR"


# --- U10.4b': step 13 ranks each grid by [diagnostic] selection_metric -----------------------

@pytest.mark.parametrize("metric", ["log loss", "brier"])
def test_show_diagnostic_grid_header_names_the_metric(metric):
    result = run_show(f"script.show_diagnostic_grid('HGB', grid, importance, {metric!r})")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == f"        HGB: the 5 best of 7 settings by validation {metric}"
    assert lines == expected_grid_printout("HGB", 7, 20, metric) + ["ok"]
    assert "average precision" not in lines[0]


def test_show_diagnostic_grid_metric_does_not_reorder_the_grid():
    # The printout shows the grid as given (grid_search already sorted it); the metric is only
    # named in the header. The hand-made grid has no log loss column at all.
    by_ap = run_show("script.show_diagnostic_grid('LR', grid, importance, 'avg precision')")
    by_ll = run_show("script.show_diagnostic_grid('LR', grid, importance, 'log loss')")
    assert by_ap.returncode == 0 and by_ll.returncode == 0, (by_ap.stderr, by_ll.stderr)
    assert by_ap.stdout.splitlines()[1:] == by_ll.stdout.splitlines()[1:]


def test_show_diagnostic_grid_needs_the_metric():
    result = run_show("script.show_diagnostic_grid('LR', grid, importance)")
    assert result.returncode != 0
    assert "TypeError" in result.stderr


@pytest.mark.parametrize("metric", ["log loss", "brier", "avg precision"])
def test_step_13_spy_passes_the_settings_metric_through(metric, tmp_path):
    # SPY_CHECKS assert grid_search got metric=settings.diagnostic.selection_metric by keyword
    # and show_diagnostic_grid got it as its 4th argument; here for each metric.
    after = SPY_CHECKS + (f"assert settings.diagnostic.selection_metric == {metric!r}\n"
                          f"assert [c[8] for c in calls if c[0] == 'grid_search'] == "
                          f"[{metric!r}] * 2\n"
                          f"assert [c[4] for c in calls if c[0] == 'show_diagnostic_grid'] == "
                          f"[{metric!r}] * 2\n")
    result = run_step_13(tmp_path / "spy", stubs=spy_stubs(), after=after, metric=metric)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "ok"


def test_step_13_spy_metric_check_bites(tmp_path):
    # The pass-through check fails when the settings say something else than was passed.
    after = SPY_CHECKS + "assert calls[0][8] == 'log loss', calls[0][8]\n"
    result = run_step_13(tmp_path / "spy", stubs=spy_stubs(), after=after, metric="brier")
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_step_13_an_unknown_metric_stops_the_real_block(tmp_path):
    # Settings normally refuse it first; if one slipped through, grid_search refuses it before
    # any fit, and nothing is written.
    output = tmp_path / "never"
    result = run_step_13(output, metric="AUC")
    assert result.returncode != 0
    assert ("HGB: cannot rank by 'AUC'; choose one of ['log loss', 'brier', 'avg precision']"
            in result.stderr)
    assert not output.exists()


@pytest.fixture(scope="module", params=["log loss", "brier"])
def metric_run(request, tmp_path_factory):
    """The real block on the tiny table, ranked by log loss or Brier."""
    output = tmp_path_factory.mktemp("step13_metric") / "hurdle_diagnostic"
    result = run_step_13(output, metric=request.param)
    return SimpleNamespace(result=result, output=output, metric=request.param,
                           lines=result.stdout.splitlines())


def test_step_13_real_run_by_metric_succeeds(metric_run):
    assert metric_run.result.returncode == 0, metric_run.result.stderr
    assert metric_run.lines[-1] == "ok"


def test_step_13_real_run_by_metric_header_names_it(metric_run):
    assert metric_run.result.returncode == 0, metric_run.result.stderr
    for name in ("HGB", "LR"):
        header = f"        {name}: the 5 best of 4 settings by validation {metric_run.metric}"
        assert metric_run.lines.count(header) == 1, (header, metric_run.lines)
    assert not any("by validation avg precision" in line for line in metric_run.lines)


def test_step_13_real_run_by_metric_progress_lines_show_it(metric_run):
    assert metric_run.result.returncode == 0, metric_run.result.stderr
    progress = [line for line in metric_run.lines if line.startswith("  ... ")]
    assert len(progress) == 8
    assert all(f"}}, {metric_run.metric} " in line for line in progress), progress
    assert not any("avg precision" in line for line in progress)


def test_step_13_real_run_by_metric_grids_are_ascending(metric_run):
    import pandas as pd
    assert metric_run.result.returncode == 0, metric_run.result.stderr
    for name in ("hgb_grid", "lr_grid"):
        grid = pd.read_csv(metric_run.output / f"{name}.csv")
        assert grid[metric_run.metric].is_monotonic_increasing, (name, grid)
        assert "brier" in grid.columns


def test_step_13_real_run_by_metric_tests_the_grid_winner(metric_run):
    import pandas as pd
    assert metric_run.result.returncode == 0, metric_run.result.stderr
    test = pd.read_csv(metric_run.output / "test.csv")
    assert test["model"].tolist() == ["HGB", "LR"]
    hgb = pd.read_csv(metric_run.output / "hgb_grid.csv").iloc[0]
    lr = pd.read_csv(metric_run.output / "lr_grid.csv").iloc[0]
    assert test.loc[0, "max_iter"] == hgb["max_iter"]
    assert test.loc[0, "class_weight"] == hgb["class_weight"]
    assert test.loc[0, "threshold"] == pytest.approx(hgb["threshold"], rel=1e-12)
    assert test.loc[1, "C"] == pytest.approx(lr["C"])
    assert test.loc[1, "class_weight"] == lr["class_weight"]
    assert test.loc[1, "threshold"] == pytest.approx(lr["threshold"], rel=1e-12)
    assert "brier" in test.columns


@pytest.mark.skipif(not (REPO_ROOT / "settings" / "synthetic.toml").exists(),
                    reason="settings/synthetic.toml not present")
def test_the_synthetic_settings_choose_the_diagnostic_by_log_loss():
    from collection_estimation.hurdle.settings import load_settings
    settings = load_settings(REPO_ROOT / "settings" / "synthetic.toml", REPO_ROOT)
    assert settings.diagnostic.selection_metric == "log loss"


# --- U10.4d: step 13's calibration, by the real run ---------------------------------------------
# The real block on the tiny table (real_run above): for each model the best unweighted setting
# fitted on train, its Platt correction from the validation p, then calibration_comparison of
# the models, the corrected models and the three references, on validation and on test.

import re  # noqa: E402

CALIBRATION_KEYS = ["mean p", "share paid", "ECE", "slope", "intercept", "log loss", "brier",
                    "calibration error", "resolution", "uncertainty", "decomposition gap",
                    "log loss skill", "brier skill", "AUC"]
CALIBRATION_LABELS = ["HGB", "HGB + Platt", "LR", "LR + Platt", "base rate", "base rate per h",
                      "own rate"]
PLATT_LINE = re.compile(r"^        (HGB|LR): Platt correction fitted on validation: "
                        r"a -?\d+\.\d{4}, b -?\d+\.\d{4}$")


def read_calibration(folder):
    import pandas as pd
    return pd.read_csv(folder / "calibration.csv", dtype={"h": str})


def test_step_13_real_run_prints_one_platt_line_per_model(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    platt = [line for line in real_run.lines if PLATT_LINE.match(line)]
    assert [PLATT_LINE.match(line).group(1) for line in platt] == ["HGB", "LR"], platt
    assert not any("calibration skipped" in line for line in real_run.lines)


def test_step_13_real_run_calibration_table_shape_and_order(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    calibration = read_calibration(real_run.output)
    assert list(calibration.columns) == ["part", "model", "h"] + CALIBRATION_KEYS
    expected = [(part, label, h) for part in ("validation", "test")
                for label in CALIBRATION_LABELS for h in ("1", "2", "all")]
    assert list(zip(calibration["part"], calibration["model"], calibration["h"])) == expected


def test_step_13_real_run_reliability_table(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    import pandas as pd
    reliability = pd.read_csv(real_run.output / "reliability.csv", dtype={"h": str})
    assert list(reliability.columns) == ["part", "model", "h", "group", "p_low", "p_high",
                                         "mean_p", "share_paid", "rows"]
    assert len(reliability) == 2 * 7 * 3 * 4
    for (part, label, h), block in reliability.groupby(["part", "model", "h"], sort=False):
        assert block["group"].tolist() == [1, 2, 3, 4], (part, label, h)
    all_rows = reliability[reliability["h"] == "all"]
    assert set(all_rows.groupby(["part", "model"])["rows"].sum()) == {54}


def test_step_13_real_run_platt_is_calibrated_in_sample_on_validation(real_run):
    # The correction was fitted on exactly these rows: its slope and intercept there are
    # 1 and 0, and its mean p is the share paid (up to the solver's tolerance).
    assert real_run.result.returncode == 0, real_run.result.stderr
    calibration = read_calibration(real_run.output)
    pooled = calibration[(calibration["part"] == "validation") & (calibration["h"] == "all")]
    for label in ("HGB + Platt", "LR + Platt"):
        row = pooled[pooled["model"] == label].iloc[0]
        assert row["slope"] == pytest.approx(1.0, abs=0.02), (label, row["slope"])
        assert row["intercept"] == pytest.approx(0.0, abs=0.02), (label, row["intercept"])
        assert row["mean p"] == pytest.approx(row["share paid"], abs=1e-3), label


def test_step_13_real_run_platt_does_not_change_the_ranking(real_run):
    # Platt is monotone (for a > 0), so each model's AUC is unchanged by it.
    assert real_run.result.returncode == 0, real_run.result.stderr
    calibration = read_calibration(real_run.output).set_index(["part", "model", "h"])
    for part in ("validation", "test"):
        for h in ("1", "2", "all"):
            for name in ("HGB", "LR"):
                assert calibration.loc[(part, f"{name} + Platt", h), "AUC"] == pytest.approx(
                    calibration.loc[(part, name, h), "AUC"], abs=1e-12), (part, name, h)


def test_step_13_real_run_the_base_rate_has_no_skill_against_itself(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    calibration = read_calibration(real_run.output)
    base = calibration[calibration["model"] == "base rate"]
    assert len(base) == 6
    assert base["log loss skill"].abs().max() < 1e-12
    assert base["brier skill"].abs().max() < 1e-12


def test_step_13_real_run_share_paid_is_the_same_for_every_label(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    calibration = read_calibration(real_run.output)
    for (part, h), block in calibration.groupby(["part", "h"]):
        assert block["share paid"].nunique() == 1, (part, h)


def test_step_13_real_run_printout_shows_the_test_calibration(real_run):
    assert real_run.result.returncode == 0, real_run.result.stderr
    lines = real_run.lines
    start = lines.index("        CALIBRATION on test, all horizons (train-only models; skill "
                        "against the base rate):")
    header = lines[start + 1].split()
    assert header[:3] == ["model", "mean", "p"] and header[-1] == "AUC"
    shown = lines[start + 2:start + 2 + 7]
    for line, label in zip(shown, CALIBRATION_LABELS):
        assert line.strip().startswith(label), (line, label)
    middle = lines.index("        log-loss skill per horizon, on test:")
    assert middle == start + 2 + 7


REFERENCE_CHECK = """
from collection_estimation.hurdle.diagnostic import (calibration_comparison,
                                                     diagnostic_splits, reference_probabilities)
calibration = pd.read_csv(settings.diagnostic.output_folder / "calibration.csv",
                          dtype={"h": str})
splits = diagnostic_splits(table, 4, 2)
for part in ("validation", "test"):
    rows = getattr(splits, part)
    refs = reference_probabilities(splits.train, rows, 13, 1.0)
    direct, _ = calibration_comparison(rows["paid"].astype(bool), rows["h"], dict(refs),
                                       refs["base rate"], 4)
    written = calibration[(calibration["part"] == part)
                          & calibration["model"].isin(list(refs.columns))]
    assert len(written) == len(direct) == 9
    for (_, want), (_, got) in zip(direct.iterrows(), written.iterrows()):
        assert want["model"] == got["model"] and str(want["h"]) == got["h"]
        for key in direct.columns[2:]:
            assert np.isclose(want[key], got[key], rtol=1e-9, atol=1e-12), (part, key)
base = splits.train["paid"].astype(bool).mean()
assert np.allclose(calibration.loc[calibration["model"] == "base rate", "mean p"], base)
"""


def test_step_13_real_run_reference_rows_equal_a_direct_comparison(tmp_path):
    result = run_step_13(tmp_path / "out", after=REFERENCE_CHECK)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "ok"


def test_step_13_real_run_with_only_weighted_settings_skips_both_calibrations(tmp_path):
    output = tmp_path / "weighted"
    result = run_step_13(output, weights=("balanced",))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    for name in ("HGB", "LR"):
        assert lines.count(f"        {name}: the grid has no unweighted setting; calibration "
                           f"skipped") == 1
    assert not any("Platt" in line for line in lines)
    assert sorted(path.name for path in output.iterdir()) == sorted(
        name + ".csv" for name in TABLE_NAMES)
    calibration = read_calibration(output)
    assert calibration["model"].unique().tolist() == ["base rate", "base rate per h",
                                                      "own rate"]
    assert calibration["part"].unique().tolist() == ["validation", "test"]
    # The skip line comes right before the next model's progress lines.
    skip = lines.index("        HGB: the grid has no unweighted setting; calibration skipped")
    assert lines[skip + 1].startswith("  ... LR 1 of 2"), lines[skip + 1]


# --- show_calibration ----------------------------------------------------------------------------

SHOW_CALIBRATION = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script

keys = ["mean p", "share paid", "ECE", "slope", "intercept", "log loss", "brier",
        "calibration error", "resolution", "uncertainty", "decomposition gap",
        "log loss skill", "brier skill", "AUC"]
rows = []
for part, offset in (("validation", 0.5), ("test", 0.0)):
    for m, label in enumerate(["Zeta", "Alpha + Platt", "base rate"]):
        for k, h in enumerate([1, 2, "all"]):
            values = {key: offset + 0.0123456789 * (i + 1) + 0.1 * m + 0.01 * k
                      for i, key in enumerate(keys)}
            rows.append({"part": part, "model": label, "h": h, **values})
calibration = pd.DataFrame(rows)
before = calibration.copy(deep=True)
script.show_calibration(calibration)
pd.testing.assert_frame_equal(calibration, before)
print("EXPECTED")
test = before[before["part"] == "test"]
columns = ["model", "mean p", "share paid", "ECE", "slope", "intercept", "log loss", "brier",
           "log loss skill", "brier skill", "AUC"]
print(test[test["h"] == "all"][columns].round(4).to_string(index=False))
print("PIVOT")
for label in ["Zeta", "Alpha + Platt", "base rate"]:
    print(label + " [" + ", ".join(
        str(round(v, 4)) for v in test[test["model"] == label]["log loss skill"]) + "]")
"""


def test_show_calibration_printout():
    result = run_code(SHOW_CALIBRATION)
    assert result.returncode == 0, result.stderr
    out, rest = result.stdout.split("EXPECTED\n")
    table, pivot_rows = rest.split("PIVOT\n")
    lines = out.splitlines()
    assert lines[0] == ("        CALIBRATION on test, all horizons (train-only models; skill "
                        "against the base rate):")
    shown = table.splitlines()
    assert len(shown) == 1 + 3
    assert lines[1:1 + len(shown)] == shown
    assert lines[1 + len(shown)] == "        log-loss skill per horizon, on test:"
    pivot = lines[2 + len(shown):]
    # Two header lines (h, then model), then one line per model in first-seen order (Zeta
    # before Alpha: not sorted), its log-loss skill for h 1, 2 and all.
    assert len(pivot) == 2 + 3, pivot
    assert pivot[0].split() == ["h", "1", "2", "all"]
    assert pivot[1].split() == ["model"]
    expected = {}
    for line in pivot_rows.splitlines():
        label, values = line.split(" [")
        expected[label] = [float(v) for v in values.rstrip("]").split(", ")]
    for line, label in zip(pivot[2:], ["Zeta", "Alpha + Platt", "base rate"]):
        assert line.startswith(label), (line, label)
        assert [float(v) for v in line[len(label):].split()] == expected[label]


def test_show_calibration_shows_only_test_rows_pooled_and_rounds():
    result = run_code(SHOW_CALIBRATION)
    assert result.returncode == 0, result.stderr
    out = result.stdout.split("EXPECTED\n")[0]
    lines = out.splitlines()
    table = lines[2:5]
    assert [line.split()[0] for line in table] == ["Zeta", "Alpha", "base"]
    # Validation values are 0.5 higher: Zeta's pooled mean p there is 0.5 + .0123 + .02 =
    # 0.5323; on test 0.0323 (rounded to 4 places).
    assert "0.5323" not in out
    assert table[0].split()[1] == "0.0323"
    assert "0.032345" not in out
    header = lines[1]
    for gone in ("calibration error", "resolution", "uncertainty", "decomposition gap"):
        assert gone not in header
    assert header.split()[0] == "model" and "part" not in header


def test_show_calibration_without_test_rows_does_not_fail():
    code = ("import pandas as pd\nimport run_hurdle_pipeline as script\n"
            "script.show_calibration(pd.DataFrame({'part': ['validation'], 'model': ['M'], "
            "'h': ['all'], 'mean p': [0.1], 'share paid': [0.1], 'ECE': [0.0], 'slope': [1.0], "
            "'intercept': [0.0], 'log loss': [0.3], 'brier': [0.1], 'log loss skill': [0.0], "
            "'brier skill': [0.0], 'AUC': [0.5]}))\nprint('ok')\n")
    result = run_code(code)
    # Pinned behaviour: with no test rows the printout's tables are empty, nothing fails.
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "ok"


# --- Step 14 (U11.1): fit stage 2 on the paid training rows at the demo origin -----------------
# The wiring is checked on main()'s syntax tree: step 14 is the last part of main(), at its top
# level after step 13's `if settings.diagnostic.run:` block (so it runs whether or not the
# diagnostic does), with stage 2's own feature blocks and settings, and stage 1's company
# switch. The printout runs on a hand-made frame, every number worked out by hand; the real
# chain runs on step 12's tiny STAGE1 table.

STEP_14_WIRING = """
import ast
import inspect
import textwrap
import run_hurdle_pipeline as script
from collection_estimation.hurdle import stage2

assert script.fit_stage2 is stage2.fit_stage2
assert script.predict_amount is stage2.predict_amount   # U11.2
assert script.total_ratio is stage2.total_ratio         # U11.2
inspect.signature(script.show_stage2_fit).bind("origin", "paid_rows", "train_rows", "paid_x",
                                               "loss", 1.0)
inspect.signature(script.show_stage2_forecast).bind("forecast_rows", "m", "p", 1.0)

body = ast.parse(textwrap.dedent(inspect.getsource(script.main))).body[0].body
statements = [ast.unparse(node) for node in body]
diagnostic = [i for i, node in enumerate(body)
              if isinstance(node, ast.If) and ast.unparse(node.test) == "settings.diagnostic.run"]
assert len(diagnostic) == 1, diagnostic
step_14 = statements[diagnostic[0] + 1:]
# U11.3: the stage 2 scores (checked below): two statements. U11.4 added the cash-by-p check
# after them (two statements) and U11.5 step 15's `if` last: five statements in all.
scores_part = body[-5:-3]
step_14 = step_14[:-5]
assert step_14 == [
    "stage2_columns = feature_columns(block_feature_columns(history, calendar, owed), "
    "settings.stage2.feature_blocks)",
    "paid_rows = train_rows[train_rows['paid'].astype(bool)]",
    "paid_x = stage1_features(paid_rows, stage2_columns, companies, "
    "settings.stage1.company_as_feature)",
    # U11.3b: E for the paid training rows and for the forecast rows, both from the training
    # rows and [stage2], before the clock starts.
    "paid_exposure = exposure(train_rows, paid_rows, settings.stage2)",
    "forecast_exposure = exposure(train_rows, forecast_rows, settings.stage2)",
    "started = time.perf_counter()",
    "stage2_model = fit_stage2(paid_x, paid_rows['amount_deflated'], paid_exposure, "
    "settings.stage2)",
    "show_stage2_fit(origin, paid_rows, train_rows, paid_x, settings.stage2.loss, "
    "time.perf_counter() - started)",
    "show_exposure(settings.stage2, paid_rows, paid_exposure, forecast_rows, forecast_exposure)",
    # U11.2: the training total, m on step 12's forecast rows (stage 2's columns), and p x m
    # against the cash, with stage 1's HGB p (not the baseline's). U11.3b: each with its E.
    "training_ratio = total_ratio(stage2_model, paid_x, paid_rows['amount_deflated'], "
    "paid_exposure)",
    "m = predict_amount(stage2_model, stage1_features(forecast_rows, stage2_columns, "
    "companies, settings.stage1.company_as_feature), forecast_exposure)",
    "show_stage2_forecast(forecast_rows, m, predictions['stage 1 (HGB)'], training_ratio)",
], step_14
# U11.3: the paid labelled forecast rows (NA paid as not paid), then either a fallback line
# or the references, the scores (stage 2's m on those rows, then the references, skill
# against the training mean) and their printout.
assert ast.unparse(scores_part[0]) == (
    "paid_forecast = forecast_rows[(forecast_rows['label_status'] == 'labelled') & "
    "forecast_rows['paid'].fillna(False).astype(bool)]"), ast.unparse(scores_part[0])
branch = scores_part[1]
assert isinstance(branch, ast.If) and ast.unparse(branch.test) == "paid_forecast.empty"
assert [ast.unparse(n) for n in branch.body] == [
    "print('        no paid forecast rows: stage 2 cannot be scored here')"]
assert [ast.unparse(n) for n in branch.orelse] == [
    "amount_guesses = amount_references(train_rows, paid_forecast, "
    "settings.diagnostic.amount_reference_window, settings.diagnostic.reference_prior_weeks)",
    "stage2_scores = amount_scores(paid_forecast['amount_deflated'], paid_forecast['h'], "
    "{'stage 2': m[paid_forecast.index], **amount_guesses}, amount_guesses['training mean'])",
    "show_stage2_scores(stage2_scores)",
    # U11.4: stage 2's reliability table on the same paid forecast rows.
    "show_amount_reliability(amount_reliability(paid_forecast['amount_deflated'], "
    "m[paid_forecast.index], settings.diagnostic.calibration_groups))",
], [ast.unparse(n) for n in branch.orelse]
# U11.4: the amount-weighted check of the whole model on the labelled forecast rows (paid or
# not), with stage 1's HGB p and stage 2's m.
assert ast.unparse(body[-3]) == "labelled = forecast_rows['label_status'] == 'labelled'"
cash_check = body[-2]
assert isinstance(cash_check, ast.If) and ast.unparse(cash_check.test) == "labelled.any()"
assert cash_check.orelse == []
assert [ast.unparse(n) for n in cash_check.body] == [
    "show_cash_by_p_group(cash_by_p_group(predictions['stage 1 (HGB)'][labelled], "
    "m[labelled], forecast_rows.loc[labelled, 'amount_deflated'], "
    "settings.diagnostic.calibration_groups))"], [ast.unparse(n) for n in cash_check.body]
# U11.5: step 15 closes main(), behind its own run switch.
step_15 = body[-1]
assert isinstance(step_15, ast.If) and ast.unparse(step_15.test) == "settings.joint_diagnostic.run"
assert step_15.orelse == []
assert [ast.unparse(n) for n in step_15.body] == [
    "run_joint_diagnostic(settings, table, columns, stage2_columns, companies, calendar)"]
from collection_estimation.hurdle import diagnostic as diagnostic_module
assert script.amount_reliability is diagnostic_module.amount_reliability
assert script.cash_by_p_group is diagnostic_module.cash_by_p_group
inspect.signature(script.show_amount_reliability).bind("table")
inspect.signature(script.show_cash_by_p_group).bind("table")
inspect.signature(script.run_joint_diagnostic).bind("settings", "table", "columns",
                                                    "stage2_columns", "companies", "calendar")
assert script.amount_references is stage2.amount_references
assert script.exposure is stage2.exposure   # U11.3b
inspect.signature(script.show_exposure).bind("stage2", "paid_rows", "paid_exposure",
                                             "forecast_rows", "forecast_exposure")
assert script.amount_scores is stage2.amount_scores
inspect.signature(script.show_stage2_scores).bind("scores")
# Step 14 uses step 12's origin, rows, predictions and companies: nothing reassigns them after
# step 12 (step 13 works on its own rows and origin).
def assigned(name):
    # Names inside tuple targets count too (predictions, fit_seconds = {}, {}).
    return [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
            and any(isinstance(n, ast.Name) and n.id == name
                    for t in node.targets for n in ast.walk(t))]
for name in ("origin", "train_rows", "forecast_rows", "predictions", "companies"):
    assert len(assigned(name)) == 1, name
    assert assigned(name)[0] < diagnostic[0], name
for node in ast.walk(body[diagnostic[0]]):
    if isinstance(node, (ast.Assign, ast.For)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        assert not names & {"origin", "train_rows", "forecast_rows", "predictions",
                            "companies", "history", "calendar", "owed"}, names
print("ok")
"""


def test_step_14_is_wired_in_last_after_step_13():
    result = run_code(STEP_14_WIRING)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_step_14_calls_fit_stage2_and_its_printout_once():
    tree = main_tree()
    for name in ("fit_stage2", "show_stage2_fit", "total_ratio", "predict_amount",
                 "show_stage2_forecast", "amount_references", "amount_scores",
                 "show_stage2_scores", "show_exposure"):
        assert len(calls_to(tree, name)) == 1, name
    assert len(calls_to(tree, "exposure")) == 2   # paid training rows, forecast rows


def test_step_14_imports_the_stage2_functions_in_one_parenthesised_import():
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = [node for node in module.body if isinstance(node, ast.ImportFrom)
               and node.module == "collection_estimation.hurdle.stage2"]
    assert len(imports) == 1
    assert [alias.name for alias in imports[0].names] == [
        "amount_references", "amount_scores", "exposure", "fit_stage2", "predict_amount",
        "total_ratio"]


def test_step_14_never_fits_stage_2_with_the_stage_1_settings():
    source = main_source()
    assert "settings.stage1)" not in source.split("stage2_model = fit_stage2(")[1][:120]
    assert 'paid_rows["amount"]' not in source   # the target is the deflated amount


# show_stage2_fit on a hand-made frame. Paid amounts 1,000, 2,000, 3,000 and 1,000,000:
# median (2,000 + 3,000) / 2 = 2,500; mean 1,006,000 / 4 = 251,500; 99th percentile (linear,
# position 0.99 x 3 = 2.97) 3,000 + 0.97 x 997,000 = 970,090; max 1,000,000. 1,234 training
# rows, 5 input columns, 1.25 s shown as 1.2 s (one decimal).
SHOW_STAGE2 = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script

paid_rows = pd.DataFrame({"amount_deflated": [1_000.0, 2_000.0, 3_000.0, 1_000_000.0],
                          "amount": [9.0, 9.0, 9.0, 9.0]}, index=[3, 7, 8, 20])
train_rows = pd.DataFrame({"paid": np.zeros(1234, dtype=bool)})
paid_x = pd.DataFrame(np.ones((4, 5)), columns=list("abcde"), index=paid_rows.index)
copies = (paid_rows.copy(deep=True), train_rows.copy(deep=True), paid_x.copy(deep=True))
returned = script.show_stage2_fit(40, paid_rows, train_rows, paid_x, LOSS, FIT_SECONDS)
assert returned is None
pd.testing.assert_frame_equal(paid_rows, copies[0])
pd.testing.assert_frame_equal(train_rows, copies[1])
pd.testing.assert_frame_equal(paid_x, copies[2])
print("ok")
"""


def run_show_stage2(loss="poisson", fit_seconds=1.25):
    return run_code(SHOW_STAGE2.replace("LOSS", repr(loss))
                    .replace("FIT_SECONDS", repr(fit_seconds)))


def test_show_stage2_fit_printout_by_hand():
    result = run_show_stage2()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "Step 14: stage 2 at forecast week 40: 4 paid training rows (of 1,234), 5 input "
        "columns, loss 'poisson', fitted in 1.2 s",
        "        deflated amount per paid row: median 2,500, mean 251,500, 99th percentile "
        "970,090, max 1,000,000",
        "ok",
    ]


@pytest.mark.parametrize("loss", ["gamma", "squared_error"])
def test_show_stage2_fit_names_the_loss_it_is_given(loss):
    result = run_show_stage2(loss=loss)
    assert result.returncode == 0, result.stderr
    assert f"loss {loss!r}, fitted in" in result.stdout.splitlines()[0]


@pytest.mark.parametrize("seconds, shown", [(0.04, "0.0"), (0.06, "0.1"), (12.345, "12.3")])
def test_show_stage2_fit_time_has_one_decimal(seconds, shown):
    result = run_show_stage2(fit_seconds=seconds)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0].endswith(f"fitted in {shown} s")


def test_show_stage2_fit_reads_the_deflated_amount_not_the_nominal_one():
    # The frame's `amount` column is 9 everywhere: none of the printed numbers is 9.
    result = run_show_stage2()
    assert result.returncode == 0, result.stderr
    assert " 9," not in result.stdout and " 9\n" not in result.stdout


# The real chain on step 12's tiny STAGE1 table at origin 2: 6 training rows, of which the 3
# TR02-A rows are paid, each with amount_deflated 100. Stage 2's blocks are ("history",) only,
# unlike stage 1's three: input h, mean_receipts_4 and (with the company) company_code.
STEP_14_CHAIN = """
import time
from collection_estimation.hurdle.settings import Stage2Settings
settings_stage2 = Stage2Settings(loss=LOSS, max_iter=5, learning_rate=0.1, max_leaf_nodes=2,
                                 min_samples_leaf=1, random_seed=1, feature_blocks=("history",),
                                 scale=SCALE, balance_floor="median", own_mean_window=4,
                                 own_mean_prior_weeks=1.0)
if SCALE == "open balance":
    # The tiny table has no open balance: 50 everywhere, below the floor (the median paid
    # training amount, 100), so every E is lifted to 100 and s = 1.
    train_rows = train_rows.assign(open_balance=50.0)
    forecast_rows = forecast_rows.assign(open_balance=50.0)
stage2_columns = script.feature_columns(script.block_feature_columns(history, calendar, owed),
                                        settings_stage2.feature_blocks)
assert stage2_columns == ["h", "mean_receipts_4"], stage2_columns
paid_rows = train_rows[train_rows["paid"].astype(bool)]
paid_x = script.stage1_features(paid_rows, stage2_columns, companies, COMPANY_AS_FEATURE)
paid_exposure = script.exposure(train_rows, paid_rows, settings_stage2)
forecast_exposure = script.exposure(train_rows, forecast_rows, settings_stage2)
assert (paid_exposure == (1.0 if SCALE == "none" else 100.0)).all(), paid_exposure
started = time.perf_counter()
stage2_model = script.fit_stage2(paid_x, paid_rows["amount_deflated"], paid_exposure,
                                 settings_stage2)
script.show_stage2_fit(origin, paid_rows, train_rows, paid_x, settings_stage2.loss,
                       time.perf_counter() - started)
script.show_exposure(settings_stage2, paid_rows, paid_exposure, forecast_rows,
                     forecast_exposure)
assert len(paid_rows) == 3 and paid_rows["key"].eq("TR02-A").all()
m = stage2_model.predict(paid_x) * paid_exposure.to_numpy()
assert abs(m - 100.0).max() < 1e-6, m
# U11.2: the training ratio, m on the 4 forecast rows, and p x m with a hand-made p of 0.5.
training_ratio = script.total_ratio(stage2_model, paid_x, paid_rows["amount_deflated"],
                                    paid_exposure)
assert abs(training_ratio - 1.0) < 1e-9, training_ratio
forecast_m = script.predict_amount(stage2_model, script.stage1_features(
    forecast_rows, stage2_columns, companies, COMPANY_AS_FEATURE), forecast_exposure)
assert forecast_m.index.equals(forecast_rows.index) and forecast_m.name == "m"
p = pd.Series(0.5, index=forecast_rows.index, name="p")
script.show_stage2_forecast(forecast_rows, forecast_m, p, training_ratio)
print("chain ok")
"""


def run_step_14_chain(company=True, loss="poisson", scale="none"):
    stage1_part = STAGE1.rsplit('print("ok")', 1)[0]
    code = (stage1_part + STEP_14_CHAIN).replace("COMPANY_AS_FEATURE", str(company)) \
        .replace("ORIGIN", "2").replace("LOSS", repr(loss)).replace("SCALE", repr(scale))
    return run_code(code)


EXPOSURE_LINES = {
    "none": "        scale 'none': E = 1, stage 2 learns the amount itself",
    "open balance": ("        scale 'open balance', floor 'median': stage 2 learns the share "
                     "of the balance paid; lifted to the floor: 3 of 3 paid training rows, "
                     "4 of 4 forecast rows"),
}


@pytest.mark.parametrize("scale", ["none", "open balance"])
@pytest.mark.parametrize("company", [True, False])
@pytest.mark.parametrize("loss", ["poisson", "gamma", "squared_error"])
def test_step_14_real_chain_on_the_tiny_table(company, loss, scale):
    result = run_step_14_chain(company=company, loss=loss, scale=scale)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "chain ok"
    step_14 = [line for line in lines if line.startswith("Step 14")]
    assert len(step_14) == 1
    columns = 3 if company else 2
    assert step_14[0].startswith(f"Step 14: stage 2 at forecast week 2: 3 paid training rows "
                                 f"(of 6), {columns} input columns, loss {loss!r}, fitted in ")
    start = lines.index(step_14[0])
    # At origin 2 the forecast targets are week 3 (h=1, labelled: TR02-A paid 100, TR04-B 0)
    # and week 4 (h=2, future, left out). m = 100 everywhere, p = 0.5: p x m = 50 + 50 = 100
    # against 100 actual.
    assert lines[start + 1:-1] == [
        "        deflated amount per paid row: median 100, mean 100, 99th percentile 100, "
        "max 100",
        EXPOSURE_LINES[scale],
        "        training total ratio (sum of m / sum of actual, paid rows): 1.0000",
        "        m on the 4 forecast rows: median 100, mean 100",
        "        per h, the whole model (sum of p x m) against the actual cash, deflated:",
        "     mean m  p x m  actual cash  ratio",
        "h                                     ",
        "1     100.0  100.0        100.0    1.0",
        "all   100.0  100.0        100.0    1.0",
    ]


# --- Step 14 (U11.2): show_stage2_forecast on a hand-made frame -----------------------------------
# Six forecast rows: h=1: labelled (actual 300), labelled (0), missing_week; h=2: labelled
# (100), labelled (0), future. m = 100, 200, [999], 400, 50, [777]; p = .5, .25, [.9], .5, .2,
# [.9] (bracketed rows are not labelled and must be left out of the table).
#   h=1: mean m (100 + 200) / 2 = 150; p x m 50 + 50 = 100; actual 300; ratio 0.333
#   h=2: mean m (400 + 50) / 2 = 225; p x m 200 + 10 = 210; actual 100; ratio 2.1
#   all: mean m 750 / 4 = 187.5, shown 188 (half to even); p x m 310; actual 400; ratio 0.775
# The second line uses every forecast row: m sorted 50, 100, 200, 400, 777, 999, median 300,
# mean 2,526 / 6 = 421.
SHOW_FORECAST = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script

nan = float("nan")
forecast_rows = pd.DataFrame({
    "h": np.array([1, 1, 1, 2, 2, 2], dtype="int64"),
    "label_status": LABELS,
    "amount_deflated": [300.0, 0.0, nan, 100.0, 0.0, nan],
    "amount": [7.0] * 6,
}, index=pd.Index([10, 11, 12, 13, 14, 15]))
m = pd.Series([100.0, 200.0, 999.0, 400.0, 50.0, 777.0], index=forecast_rows.index, name="m")
p = pd.Series([0.5, 0.25, 0.9, 0.5, 0.2, 0.9], index=forecast_rows.index, name="p")
copies = (forecast_rows.copy(deep=True), m.copy(), p.copy())
returned = script.show_stage2_forecast(forecast_rows, m, p, RATIO)
assert returned is None
pd.testing.assert_frame_equal(forecast_rows, copies[0])
pd.testing.assert_series_equal(m, copies[1])
pd.testing.assert_series_equal(p, copies[2])
print("ok")
"""

FORECAST_LABELS = ["labelled", "labelled", "missing_week", "labelled", "labelled", "future"]
FORECAST_HEAD = [
    "        training total ratio (sum of m / sum of actual, paid rows): 1.0000",
    "        m on the 6 forecast rows: median 300, mean 421",
]


def run_show_forecast(labels=None, ratio=1.00004):
    labels = FORECAST_LABELS if labels is None else labels
    return run_code(SHOW_FORECAST.replace("LABELS", repr(labels)).replace("RATIO", repr(ratio)))


def test_show_stage2_forecast_printout_by_hand():
    result = run_show_forecast()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == FORECAST_HEAD + [
        "        per h, the whole model (sum of p x m) against the actual cash, deflated:",
        "     mean m  p x m  actual cash  ratio",
        "h                                     ",
        "1     150.0  100.0        300.0  0.333",
        "2     225.0  210.0        100.0  2.100",
        "all   188.0  310.0        400.0  0.775",
        "ok",
    ]


def test_show_stage2_forecast_training_ratio_has_four_decimals():
    result = run_show_forecast(ratio=0.987654)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == (
        "        training total ratio (sum of m / sum of actual, paid rows): 0.9877")


def test_show_stage2_forecast_leaves_unlabelled_rows_out_of_the_table():
    # Labelling the missing week and the future row would bring in m 999 / 777 and p 0.9;
    # neither value appears in the hand-checked table above.
    result = run_show_forecast()
    assert result.returncode == 0, result.stdout
    table = result.stdout.splitlines()[3:-1]
    for gone in ("999", "777", "899", "699"):
        assert not any(gone in line for line in table), gone


def test_show_stage2_forecast_a_missing_week_is_not_a_zero():
    # If the missing week were counted as 0 cash, h=1's mean m would be (100+200+999)/3 = 433.
    result = run_show_forecast()
    assert result.returncode == 0, result.stderr
    assert "433" not in result.stdout


def test_show_stage2_forecast_an_h_with_one_labelled_row():
    # Only one h=2 row labelled: the table has h=2 from that row alone.
    labels = ["labelled", "labelled", "missing_week", "labelled", "future", "future"]
    result = run_show_forecast(labels=labels)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[5] == "1     150.0  100.0        300.0  0.333"
    assert lines[6].split() == ["2", "400.0", "200.0", "100.0", "2.000"]
    # all: mean m 700 / 3 = 233.3 -> 233; p x m 300; actual 400; ratio 0.75
    assert lines[7].split() == ["all", "233.0", "300.0", "400.0", "0.750"]


@pytest.mark.parametrize("labels", [
    ["missing_week", "missing_week", "missing_week", "future", "future", "future"],
    ["future"] * 6,
], ids=["missing-and-future", "all-future"])
def test_show_stage2_forecast_without_labelled_rows_prints_the_fallback_and_returns(labels):
    result = run_show_forecast(labels=labels)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == FORECAST_HEAD + [
        "        no labelled forecast rows: p x m cannot be compared with actual cash",
        "ok",
    ]


def test_show_stage2_forecast_with_labelled_rows_does_not_print_the_fallback():
    result = run_show_forecast()
    assert result.returncode == 0, result.stderr
    assert "no labelled forecast rows" not in result.stdout


def test_show_stage2_forecast_an_h_with_no_cash_shows_an_infinite_ratio():
    # Pinned behaviour: both labelled h=2 rows paid nothing (actual 0), so h=2's ratio is
    # p x m / 0 = inf; it is printed, nothing fails.
    code = SHOW_FORECAST.replace("LABELS", repr(FORECAST_LABELS)).replace("RATIO", "1.0") \
        .replace("[300.0, 0.0, nan, 100.0, 0.0, nan]", "[300.0, 0.0, nan, 0.0, 0.0, nan]")
    result = run_code(code)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[6].split() == ["2", "225.0", "210.0", "0.0", "inf"]


def test_show_stage2_forecast_uses_the_deflated_amount_not_the_nominal_one():
    # `amount` is 7 on every row; the actual cash column shows 300 / 100 / 400, never 14 or 28.
    result = run_show_forecast()
    assert result.returncode == 0, result.stderr
    assert "14.0" not in result.stdout and "28.0" not in result.stdout


# --- Step 14 (U11.3): show_stage2_scores and the paid forecast rows ----------------------------------
# show_stage2_scores on amount_scores' output for the hand case of test_stage2.py: rows 10..13,
# h 2, 1, 2, 1, actual 100, 200, 50, 400, stage 2 110, 180, 60, 300, training mean 150.
#   pooled, stage 2: deviance 8.749 -> 9; skill 1 - 8.749 / 102.195 = 0.914; R2 0.853;
#     WAPE 140 / 750 = 0.187; MAPE (.1 + .1 + .2 + .25) / 4 = 0.1625 -> 0.162; total 650 / 750
#   pooled, training mean: deviance 102.195 -> 102; skill 0; R2 -0.078; WAPE 450 / 750 = 0.6;
#     MAPE (.5 + .25 + 2 + .625) / 4 = 0.844; total 600 / 750 = 0.8
#   skill per h, stage 2: h=1 0.892, h=2 0.975
SHOW_SCORES = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle.stage2 import amount_scores

index = pd.Index([10, 11, 12, 13])
actual = pd.Series([100.0, 200.0, 50.0, 400.0], index=index)
h = pd.Series(np.array(HORIZONS, dtype="int64"), index=index)
reference = pd.Series(150.0, index=index)
guesses = {"stage 2": pd.Series([110.0, 180.0, 60.0, 300.0], index=index),
           "training mean": reference,
           "own mean": pd.Series([90.0, 250.0, 40.0, 350.0], index=index)}
predictions = {name: guesses[name] for name in MODELS}
scores = amount_scores(actual, h, predictions, reference)
copy = scores.copy(deep=True)
returned = script.show_stage2_scores(scores)
assert returned is None
pd.testing.assert_frame_equal(scores, copy)
print("ok")
"""

SCORES_HEADER = ("        STAGE 2 alone, on the paid forecast rows (skill against the training "
                 "mean; R2's baseline is the rows' own mean):")


def run_show_scores(models=("stage 2", "training mean"), h=(2, 1, 2, 1)):
    return run_code(SHOW_SCORES.replace("MODELS", repr(list(models))).replace("HORIZONS", repr(list(h))))


def test_show_stage2_scores_printout_by_hand():
    result = run_show_scores()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        SCORES_HEADER,
        "        model  rows  m <= 0  mean deviance  deviance skill  R2 hindsight  WAPE  MAPE  "
        "total ratio",
        "      stage 2     4       0            9.0           0.914         0.853 0.187 0.162   "
        "     0.867",
        "training mean     4       0          102.0           0.000        -0.078 0.600 0.844   "
        "     0.800",
        "        deviance skill per h:",
        "h                  1      2    all",
        "model                             ",
        "stage 2        0.892  0.975  0.914",
        "training mean  0.000  0.000  0.000",
        "ok",
    ]


def test_show_stage2_scores_pooled_table_has_no_h_column_and_only_the_all_rows():
    result = run_show_scores()
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[1].split()[:2] == ["model", "rows"]
    assert " h " not in f" {lines[1]} "
    assert [line.split()[-1] for line in lines[2:4]] == ["0.867", "0.800"]


def test_show_stage2_scores_keeps_the_models_order_not_alphabetical():
    # As in main(): stage 2, then the references in amount_references' order.
    result = run_show_scores(models=("stage 2", "training mean", "own mean"))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    # own mean: 90, 250, 40, 350 against 100, 200, 50, 400: WAPE 120 / 750 = 0.16, total
    # 730 / 750 = 0.973. Alphabetical would put own mean first.
    assert lines[2:5] == [
        "      stage 2     4       0            9.0           0.914         0.853 0.187 0.162 "
        "       0.867",
        "training mean     4       0          102.0           0.000        -0.078 0.600 0.844 "
        "       0.800",
        "     own mean     4       0            5.0           0.949         0.928 0.160 0.169 "
        "       0.973",
    ]
    assert lines[-4:-1] == ["stage 2        0.892  0.975  0.914",
                            "training mean  0.000  0.000  0.000",
                            "own mean       0.941  0.969  0.949"]


def test_show_stage2_scores_per_h_columns_are_the_sorted_h_then_all():
    result = run_show_scores(h=(5, 1, 3, 1))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    header = lines[lines.index("        deviance skill per h:") + 1]
    assert header.split() == ["h", "1", "3", "5", "all"]


def test_show_stage2_scores_shows_minus_inf_skill_for_a_guess_at_or_below_zero():
    code = SHOW_SCORES.replace("MODELS", repr(["stage 2", "training mean"])) \
        .replace("HORIZONS", repr([2, 1, 2, 1])) \
        .replace("[110.0, 180.0, 60.0, 300.0]", "[110.0, 180.0, 60.0, -300.0]")
    result = run_code(code)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    stage2 = lines[2].split()
    assert stage2[:5] == ["stage", "2", "4", "1", "inf"]
    assert stage2[5] == "-inf"


# The paid forecast rows filter, run from main()'s own statement on a hand frame: labelled and
# paid only; a missing week and a future row (paid NA) are left out, never an error.
PAID_FILTER = """
import ast
import inspect
import textwrap
import pandas as pd
import run_hurdle_pipeline as script

body = ast.parse(textwrap.dedent(inspect.getsource(script.main))).body[0].body
statement = [node for node in body if isinstance(node, ast.Assign)
             and ast.unparse(node.targets[0]) == "paid_forecast"]
assert len(statement) == 1
forecast_rows = pd.DataFrame({
    "label_status": LABELS,
    "paid": pd.array(PAID, dtype="boolean"),
}, index=[30, 31, 32, 33, 34])
namespace = {"forecast_rows": forecast_rows}
exec(compile(ast.Module(body=statement, type_ignores=[]), "<step 14>", "exec"), namespace)
print(list(namespace["paid_forecast"].index))
"""


def run_paid_filter(labels, paid):
    return run_code(PAID_FILTER.replace("LABELS", repr(labels)).replace("PAID", repr(paid)))


def test_paid_forecast_rows_are_the_labelled_paid_rows():
    result = run_paid_filter(["labelled", "labelled", "missing_week", "future", "labelled"],
                             [True, False, None, None, True])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[30, 34]"


def test_paid_forecast_rows_ignore_a_paid_flag_on_an_unlabelled_row():
    # A row marked paid but not labelled (should not happen) still stays out.
    result = run_paid_filter(["missing_week", "future", "labelled", "labelled", "labelled"],
                             [True, True, False, None, False])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_paid_forecast_rows_when_nothing_paid_is_empty_and_the_fallback_is_printed():
    # The fallback branch: main()'s own If statement, run with an empty paid_forecast.
    code = """
import ast
import inspect
import textwrap
import pandas as pd
import run_hurdle_pipeline as script

body = ast.parse(textwrap.dedent(inspect.getsource(script.main))).body[0].body
branch = [node for node in body if isinstance(node, ast.If)
          and ast.unparse(node.test) == "paid_forecast.empty"]
assert len(branch) == 1
def refuse(*args, **kwargs):
    raise AssertionError("scored although no row paid")
namespace = {"paid_forecast": pd.DataFrame({"h": []}), "amount_references": refuse,
             "amount_scores": refuse, "show_stage2_scores": refuse}
exec(compile(ast.Module(body=branch, type_ignores=[]), "<step 14>", "exec"), namespace)
print("ok")
"""
    result = run_code(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "        no paid forecast rows: stage 2 cannot be scored here", "ok"]


# The real chain on the tiny table, continued through the scores: at origin 2 the one paid
# labelled forecast row is TR02-A, h=1, amount 100; m = 100 there. The tiny table has no
# paying/observed weeks, so they are added here: window 4, 1 paying week of 2 observed, and
# mean_receipts_4 = 1.0 (from the table): own mean (2 + 100) / 2 = 51; training mean 100.
STEP_14_SCORES = """
paid_forecast = forecast_rows[(forecast_rows["label_status"] == "labelled")
                              & forecast_rows["paid"].fillna(False).astype(bool)]
assert list(paid_forecast["key"]) == ["TR02-A"] and list(paid_forecast["h"]) == [1]
paid_forecast = paid_forecast.assign(paying_weeks_4=1, observed_weeks_4=2)
amount_guesses = script.amount_references(train_rows, paid_forecast, 4, 1.0)
assert list(amount_guesses.iloc[0]) == [100.0, 51.0], amount_guesses
stage2_scores = script.amount_scores(paid_forecast["amount_deflated"], paid_forecast["h"],
                                     {"stage 2": forecast_m[paid_forecast.index],
                                      **amount_guesses},
                                     amount_guesses["training mean"])
script.show_stage2_scores(stage2_scores)
print("scores ok")
"""


def test_step_14_real_chain_through_the_scores():
    stage1_part = STAGE1.rsplit('print("ok")', 1)[0]
    code = (stage1_part + STEP_14_CHAIN + STEP_14_SCORES).replace(
        "COMPANY_AS_FEATURE", "True").replace("ORIGIN", "2").replace("LOSS", "'poisson'") \
        .replace("SCALE", "'none'")
    result = run_code(code)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "scores ok"
    start = lines.index(SCORES_HEADER)
    # m = actual = training mean = 100: the reference's deviance is 0, so every skill is
    # undefined (NaN, printed, nothing fails); one row, so R2 is NaN too. Own mean 51 is off
    # by 49: deviance 2 (100 ln(100 / 51) - 49) = 36.7, WAPE = MAPE = 0.49, total 0.51.
    assert lines[start + 1:-1] == [
        "        model  rows  m <= 0  mean deviance  deviance skill  R2 hindsight  WAPE  MAPE  "
        "total ratio",
        "      stage 2     1       0            0.0             NaN           NaN  0.00  0.00  "
        "       1.00",
        "training mean     1       0            0.0             NaN           NaN  0.00  0.00  "
        "       1.00",
        "     own mean     1       0           37.0             NaN           NaN  0.49  0.49  "
        "       0.51",
        "        deviance skill per h:",
        "h               1  all",
        "model                 ",
        "stage 2       NaN  NaN",
        "training mean NaN  NaN",
        "own mean      NaN  NaN",
    ]


# --- Step 14 (U11.3b): show_exposure on hand-made frames -------------------------------------------
# Paid training rows: balances 50, 200, -10, 1,500 with E 100, 200, 100, 1,500: lifted where
# open_balance < E, i.e. 50 and -10 (200 = E is not lifted): 2 of 4. Forecast rows: balances
# 0, 300, 99.5 with E 100, 300, 100: 2 of 3. 1,234 forecast rows would show a comma: the
# forecast frame is padded with 1,231 rows at their floor (balance 100 = E, not lifted).
SHOW_EXPOSURE = """
import pandas as pd
from types import SimpleNamespace
import run_hurdle_pipeline as script

stage2 = SimpleNamespace(scale=SCALE, balance_floor=FLOOR)
paid_rows = pd.DataFrame({"open_balance": [50.0, 200.0, -10.0, 1500.0]}, index=[1, 2, 3, 4])
paid_exposure = pd.Series([100.0, 200.0, 100.0, 1500.0], index=paid_rows.index)
balances = [0.0, 300.0, 99.5] + [100.0] * 1231
forecast_rows = pd.DataFrame({"open_balance": balances})
forecast_exposure = pd.Series([100.0, 300.0, 100.0] + [100.0] * 1231)
if SCALE == "none":
    # "none" reads nothing from the rows: no open_balance column at all.
    paid_rows = paid_rows.drop(columns="open_balance")
    forecast_rows = forecast_rows.drop(columns="open_balance")
copies = (paid_rows.copy(deep=True), paid_exposure.copy(), forecast_rows.copy(deep=True),
          forecast_exposure.copy())
returned = script.show_exposure(stage2, paid_rows, paid_exposure, forecast_rows,
                                forecast_exposure)
assert returned is None
pd.testing.assert_frame_equal(paid_rows, copies[0])
pd.testing.assert_series_equal(paid_exposure, copies[1])
pd.testing.assert_frame_equal(forecast_rows, copies[2])
pd.testing.assert_series_equal(forecast_exposure, copies[3])
print("ok")
"""


def run_show_exposure(scale, floor="own mean"):
    return run_code(SHOW_EXPOSURE.replace("SCALE", repr(scale)).replace("FLOOR", repr(floor)))


def test_show_exposure_scale_none_prints_one_line_and_returns():
    result = run_show_exposure("none")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "        scale 'none': E = 1, stage 2 learns the amount itself", "ok"]


def test_show_exposure_open_balance_counts_the_rows_lifted_to_the_floor_by_hand():
    result = run_show_exposure("open balance")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "        scale 'open balance', floor 'own mean': stage 2 learns the share of the "
        "balance paid; lifted to the floor: 2 of 4 paid training rows, 2 of 1,234 forecast "
        "rows",
        "ok",
    ]


@pytest.mark.parametrize("floor", ["median", "mean", "own mean"])
def test_show_exposure_names_the_floor_it_is_given(floor):
    result = run_show_exposure("open balance", floor=floor)
    assert result.returncode == 0, result.stderr
    assert f"floor {floor!r}: stage 2 learns" in result.stdout.splitlines()[0]


# --- Step 14 (U11.4): show_amount_reliability and show_cash_by_p_group -----------------------------
# The printouts run on tables from the real functions (hand-made inputs); the header lines are
# exact, the numbers rounded as documented (amounts to whole TRY, ratios to 3 decimals).

SHOW_U11_4 = """
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle.diagnostic import amount_reliability, cash_by_p_group

actual = pd.Series([22.4, 12.0, 35.0, 24.0, 44.0, 35.0])
m = pd.Series([50.0, 10.0, 40.0, 20.0, 60.0, 30.0])
table = amount_reliability(actual, m, 3)
copy = table.copy(deep=True)
assert script.show_amount_reliability(table) is None
pd.testing.assert_frame_equal(table, copy)
p = pd.Series([0.1, 0.4, 0.2, 0.3, 0.8, 0.6])
cash = cash_by_p_group(p, pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
                       pd.Series([0.0, 500.0, 0.0, 0.0, 1000.0, 0.0]), 3)
copy = cash.copy(deep=True)
assert script.show_cash_by_p_group(cash) is None
pd.testing.assert_frame_equal(cash, copy)
print("ok")
"""


def test_show_amount_reliability_and_cash_by_p_group_printouts():
    result = run_code(SHOW_U11_4)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    assert lines[0] == ("        stage 2 reliability, paid forecast rows in equal groups by m "
                        "(calibrated: ratio = mean actual / mean m near 1 in every group):")
    header = lines[1].split()
    assert header == ["group", "m_low", "m_high", "mean_m", "mean_actual", "ratio", "rows"]
    # Group 1: m 10, 20; actual 12, 24 -> mean 18, ratio 1.2. Group 3: actual 22.4 and 44
    # -> mean 33.2 shown as 33, ratio 33.2 / 55 = 0.6036 shown as 0.604.
    assert lines[2].split() == ["1", "10.0", "20.0", "15.0", "18.0", "1.200", "2"] or \
        lines[2].split() == ["1", "10.0", "20.0", "15.0", "18.0", "1.2", "2"]
    assert lines[4].split()[4] == "33.0" and lines[4].split()[5] == "0.604"
    cash_start = lines.index("        whole model, labelled forecast rows in equal groups by p: "
                             "expected cash (sum of p x m) vs actual cash, deflated (ratio = "
                             "expected / actual):")
    assert lines[cash_start + 1].split() == ["group", "p_low", "p_high", "mean_p",
                                             "expected_cash", "actual_cash", "ratio", "rows"]
    first = lines[cash_start + 2].split()
    assert first[0] == "1" and first[4] == "70.0" and first[5] == "0.0" and first[6] == "NaN"
    assert lines[cash_start + 4].split()[6] == "0.76"


# --- Step 15 (U11.5): the joint diagnostic ------------------------------------------------------
# Wiring on the script's syntax tree and module; then run_joint_diagnostic itself, end to end,
# on a tiny hand-made table (2 companies x 8 customers, origins 0..33, h 1..3; T = 20, H = 3)
# with tiny grids, writing its CSVs to tmp_path only.

STEP_15_WIRING = """
import ast
import inspect
import run_hurdle_pipeline as script
from collection_estimation.hurdle import benchmark, diagnostic, diagnostic_joint

for name in ("benchmark_forecasts", "cash_totals"):
    assert getattr(script, name) is getattr(benchmark, name), name
for name in ("paired_bootstrap", "score_pairs", "scores_by_h", "stage1_grid", "stage2_grid",
             "stage1_gates", "stage2_gates"):
    assert getattr(script, name) is getattr(diagnostic_joint, name), name
inspect.signature(script.with_params).bind("gates", "grid", "stage1")
inspect.signature(script.show_gates).bind("stage 1", "gates", "log loss", ascending=True)
# total_codes was removed: score_pairs sums p x m per total with cash_totals itself.
assert not hasattr(diagnostic_joint, "total_codes")
assert not hasattr(script, "total_codes")
for name in ("amount_reliability", "cash_by_p_group", "calibration_summary"):
    assert getattr(script, name) is getattr(diagnostic, name), name
inspect.signature(script.show_joint_grid).bind("pairs", "mape")
inspect.signature(script.show_joint_results).bind("tables", "chosen", 10)
print("ok")
"""


def test_step_15_imports_and_signatures():
    result = run_code(STEP_15_WIRING)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_step_15_run_joint_diagnostic_is_called_once_in_main():
    tree = main_tree()
    assert len(calls_to(tree, "run_joint_diagnostic")) == 1
    for name in ("score_pairs", "benchmark_forecasts", "paired_bootstrap"):
        assert calls_to(tree, name) == [], name


def run_joint_diagnostic_source():
    text = SCRIPT.read_text(encoding="utf-8")
    nodes = [node for node in ast.parse(text).body
             if isinstance(node, ast.FunctionDef) and node.name == "run_joint_diagnostic"]
    assert len(nodes) == 1
    return " ".join(ast.get_source_segment(text, nodes[0]).split())


def test_step_15_scores_the_pairs_on_the_validation_rows_with_series_predictions():
    # score_pairs now takes the validation rows and the predictions as Series (with the rows'
    # index, so p x m aligns by label): no .to_numpy(), no separate totals, no total_codes.
    # Since the progress lines, the grid predictions go through a local predict_grid helper:
    # one prediction per grid setting, in grid order, each fitted on train and predicted on
    # the validation rows (still Series: no .to_numpy()).
    source = run_joint_diagnostic_source()
    for line in ("def predict_grid(stage, grid, predict):",
                 "for done, candidate in enumerate(grid, start=1):",
                 'predictions.append(predict(candidate.settings, "train", validation))',
                 "return predictions",
                 'p_by_stage1 = predict_grid("stage 1", grid1, predict_p)',
                 'm_by_stage2 = predict_grid("stage 2", grid2, predict_m)',
                 # U11.5b: only the settings that passed the eligibility gates are paired.
                 "pairs = score_pairs(validation, [p_by_stage1[i] for i in eligible1], "
                 "[m_by_stage2[j] for j in eligible2], joint.selection_metric)"):
        assert line in source, line
    assert source.count("score_pairs(") == 1
    assert source.count("predictions.append(") == 1
    assert source.count("predict_grid(") == 3  # the definition and the two calls
    assert (source.index("def predict_grid(") < source.index("p_by_stage1 = predict_grid(")
            < source.index("m_by_stage2 = predict_grid(") < source.index("pairs = score_pairs("))
    for gone in ("total_codes", "validation_totals",
                 'predict(candidate.settings, "train", validation).to_numpy()',
                 'p_by_stage1 = [', 'm_by_stage2 = ['):
        assert gone not in source, gone
    assert "total_codes" not in SCRIPT.read_text(encoding="utf-8")


def test_step_15_runs_after_step_14s_checks():
    source = main_source()
    assert (source.index("show_stage2_scores(stage2_scores)")
            < source.index("show_cash_by_p_group(")
            < source.index("if settings.joint_diagnostic.run:"))


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STEP_15_RUN = """
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle.settings import (JointDiagnosticSettings, Stage1Settings,
                                                   Stage2Settings)

OUT = Path(OUTPUT_FOLDER)
rng = np.random.default_rng(3)
rows = []
for company in ("TR02", "TR04"):
    for c in range(8):
        size = 100.0 * (1 + c)
        for origin in range(0, 34):
            for h in (1, 2, 3):
                x = rng.random()
                paid = rng.random() < 0.2 + 0.6 * x
                amount = size * (0.5 + x) if paid else 0.0
                rows.append({"key": f"{company}-{c}", "company_code": company,
                             "origin_week": origin, "h": h, "target_week": origin + h,
                             "x": x, "amount": amount, "paid": paid,
                             "open_balance": size * 2 * rng.random(),
                             "mean_receipts_4": size, "paying_weeks_4": int(rng.integers(0, 5)),
                             "observed_weeks_4": 4})
table = pd.DataFrame(rows)
table["label_status"] = np.where(table["target_week"] <= 33, "labelled", "future").astype(object)
future = table["label_status"] == "future"
table["amount_deflated"] = table["amount"].where(~future, np.nan)
table["paid"] = pd.array([None if f else bool(p) for f, p in zip(future, table["paid"])],
                         dtype="boolean")
weeks = np.arange(0, 40)
calendar = pd.DataFrame({"week": weeks, "business_days": 5 - (weeks % 6 == 0),
                         "has_month_end": weeks % 4 == 3})
table = table.merge(calendar.rename(columns={"week": "target_week"}), on="target_week")

stage1 = Stage1Settings(max_iter=5, learning_rate=0.3, max_leaf_nodes=4, min_samples_leaf=5,
                        random_seed=1, feature_blocks=("history",), company_as_feature=True,
                        baseline_c=1.0)
stage2 = Stage2Settings(loss="poisson", max_iter=5, learning_rate=0.3, max_leaf_nodes=4,
                        min_samples_leaf=5, random_seed=1, feature_blocks=("history",),
                        scale="none", balance_floor="median", own_mean_window=4,
                        own_mean_prior_weeks=1.0)
# U11.5b: gates wide open (every setting of this tiny grid passes them), so the run scores
# every pair; the gate tests replace this line with tighter gates.
GATES = dict(stage1_log_loss_tolerance=1e9, stage1_slope_band=(1e-9, 1e9), stage2_skill_tolerance=1e9, stage2_ratio_band=(1e-9, 1e9))
joint = JointDiagnosticSettings(
    run=True, stage1_max_iter=(3, 8), stage1_learning_rate=(0.3,), stage1_max_leaf_nodes=(4,),
    stage2_loss=("poisson", "gamma"), stage2_max_iter=(4,), stage2_learning_rate=(0.3,),
    stage2_max_leaf_nodes=(4,), stage2_scale=("none", "open balance"),
    stage2_balance_floor=("median", "own mean"), selection_metric=METRIC, bootstrap_samples=50,
    output_folder=OUT, **GATES)
settings = SimpleNamespace(
    stage1=stage1, stage2=stage2, joint_diagnostic=joint,
    receivables=SimpleNamespace(horizons=3),
    diagnostic=SimpleNamespace(origin=20, reference_window=4, reference_prior_weeks=1.0,
                               calibration_groups=3, amount_reference_window=4))
columns = ["h", "x", "business_days", "has_month_end"]
stage2_columns = ["h", "x"]
copy = table.copy(deep=True)
script.run_joint_diagnostic(settings, table, columns, stage2_columns, ["TR02", "TR04"],
                            calendar)
pd.testing.assert_frame_equal(table, copy)
print("run ok")
"""


@pytest.fixture(scope="module", params=["mape", "wape"])
def step_15_run(request, tmp_path_factory):
    folder = tmp_path_factory.mktemp(f"joint_{request.param}")
    code = (STEP_15_RUN.replace("OUTPUT_FOLDER", repr(str(folder)))
            .replace("METRIC", repr(request.param)))
    return request.param, folder, run_code(code)


def test_step_15_real_run_succeeds_and_does_not_change_the_table(step_15_run):
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "run ok"


def test_step_15_real_run_header_names_the_origin_and_the_split_weeks(step_15_run):
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == (
        "Step 15: joint diagnostic of stage 1 and stage 2 at origin 20 (diagnostic only); "
        "validation target weeks 21-23, test 24-26")


def test_step_15_real_run_writes_its_tables(step_15_run):
    _, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    names = sorted(path.name for path in folder.iterdir())
    assert names == sorted(f"{name}.csv" for name in (
        "pairs", "stage1_gates", "stage2_gates",
        "totals_validation_consolidated", "totals_validation_per_company",
        "totals_test_consolidated", "totals_test_per_company", "scores", "paired",
        "stage1_test", "stage2_test", "cash_by_p_group", "amount_reliability"))


def test_step_15_real_run_tries_every_pair(step_15_run):
    _, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    pairs = pd.read_csv(folder / "pairs.csv")
    # stage 1: 2 settings; stage 2: 2 losses x (1 "none" + 2 floors) = 6 settings.
    assert len(pairs) == 12
    assert len(pairs.drop_duplicates([c for c in pairs.columns if c.startswith("s")])) == 12


def test_step_15_real_run_pairs_are_sorted_by_the_selection_metric(step_15_run):
    metric, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    pairs = pd.read_csv(folder / "pairs.csv")
    assert pairs[metric.upper()].is_monotonic_increasing


def test_step_15_real_run_chosen_validation_score_equals_the_best_pair(step_15_run):
    # The chosen pair, fitted on train and scored on the validation totals, is the grid's best
    # pair: the same fits, the same totals, so the same number.
    metric, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    pairs = pd.read_csv(folder / "pairs.csv")
    scores = pd.read_csv(folder / "scores.csv")
    chosen = scores[(scores["part"] == "validation") & (scores["level"] == "consolidated")
                    & (scores["model"] == "chosen") & (scores["h"] == "all")]
    assert len(chosen) == 1
    assert chosen[metric.upper()].iloc[0] == pytest.approx(pairs[metric.upper()].iloc[0],
                                                           rel=1e-9)


def test_step_15_real_run_totals_add_up(step_15_run):
    _, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    for part in ("validation", "test"):
        consolidated = pd.read_csv(folder / f"totals_{part}_consolidated.csv")
        per_company = pd.read_csv(folder / f"totals_{part}_per_company.csv")
        assert set(per_company["company_code"]) == {"TR02", "TR04"}
        for column in ("actual", "chosen", "default"):
            summed = per_company.groupby(["origin_week", "h"])[column].sum()
            mine = consolidated.set_index(["origin_week", "h"])[column]
            np.testing.assert_allclose(summed.loc[mine.index], mine, rtol=1e-9)
        weeks = consolidated["target_week"]
        if part == "validation":
            assert weeks.between(21, 23).all()
        else:
            assert weeks.between(24, 26).all()
        assert (consolidated["target_week"]
                == consolidated["origin_week"] + consolidated["h"]).all()


def test_step_15_real_run_scores_cover_every_model_level_part_and_h(step_15_run):
    _, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    scores = pd.read_csv(folder / "scores.csv")
    assert set(scores["model"]) == {"chosen", "default", "deflated", "deflated_nolevel"}
    assert set(scores["part"]) == {"validation", "test"}
    assert set(scores["level"]) == {"consolidated", "per company"}
    assert set(scores["h"].astype(str)) == {"1", "2", "3", "all"}


def test_step_15_real_run_paired_comparisons(step_15_run):
    _, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    paired = pd.read_csv(folder / "paired.csv")
    assert paired[["part", "A", "B"]].values.tolist() == [
        [part, a, b] for part in ("validation", "test")
        for a in ("chosen", "default") for b in ("deflated", "deflated_nolevel")]
    scores = pd.read_csv(folder / "scores.csv")
    pooled = scores[(scores["level"] == "consolidated") & (scores["h"].astype(str) == "all")]
    for _, row in paired.iterrows():
        mape = pooled[pooled["part"] == row["part"]].set_index("model")["MAPE"]
        assert row["mean difference"] == pytest.approx(mape[row["A"]] - mape[row["B"]],
                                                       abs=1e-12)


def test_step_15_real_run_deflated_benchmark_does_not_fall_back_with_enough_weeks(step_15_run):
    # Training target weeks 1..20 (validation) and 1..23 (test): at least 12, so a model.
    _, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    for part in ("validation", "test"):
        totals = pd.read_csv(folder / f"totals_{part}_consolidated.csv")
        assert not totals["deflated_nolevel fallback"].any()
        assert np.isfinite(totals["deflated"]).all()


def test_step_15_real_run_prints_the_results(step_15_run):
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for text in ("the 10 best of 12 pairs by validation", "chosen pair: stage 1 ",
                 "VALIDATION, consolidated weekly totals", "TEST, consolidated weekly totals",
                 "PAIRED comparison on the consolidated totals",
                 "TEST, the chosen pair's stage 1 alone",
                 "TEST, the chosen pair's stage 2 alone on the paid rows, all h:",
                 "whole model, labelled forecast rows in equal groups by p",
                 "stage 2 reliability, paid forecast rows in equal groups by m",
                 "tables written to "):
        assert text in out, text


GRID_PROGRESS = re.compile(r"^  (stage [12]) grid: (\d+) of (\d+) done \((.*)\); "
                           r"(\d+) s so far, about (\d+) s left$")


def grid_progress_lines(stdout):
    return [GRID_PROGRESS.match(line) for line in stdout.splitlines()
            if GRID_PROGRESS.match(line)]


def test_step_15_real_run_prints_one_progress_line_per_grid_setting(step_15_run):
    # The fixture's grids: stage 1 has 2 settings, stage 2 has 6 (see tries_every_pair).
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    found = [(m.group(1), int(m.group(2)), int(m.group(3))) for m in
             grid_progress_lines(result.stdout)]
    assert found == ([("stage 1", done, 2) for done in (1, 2)]
                     + [("stage 2", done, 6) for done in range(1, 7)])


def test_step_15_real_run_progress_lines_name_distinct_settings(step_15_run):
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    for stage, size in (("stage 1", 2), ("stage 2", 6)):
        params = [m.group(4) for m in grid_progress_lines(result.stdout) if m.group(1) == stage]
        assert len(params) == size
        assert len(set(params)) == size, params
        assert all(p.startswith("{") and p.endswith("}") for p in params), params


def test_step_15_real_run_last_progress_line_of_each_grid_has_nothing_left(step_15_run):
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    for stage in ("stage 1", "stage 2"):
        lines = [m for m in grid_progress_lines(result.stdout) if m.group(1) == stage]
        assert lines[-1].group(0).endswith("s so far, about 0 s left"), lines[-1].group(0)
        elapsed = [int(m.group(5)) for m in lines]
        assert elapsed == sorted(elapsed)  # time so far never goes back


def test_step_15_real_run_progress_lines_come_before_the_grid_table(step_15_run):
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    progress = [i for i, line in enumerate(lines) if GRID_PROGRESS.match(line)]
    table = next(i for i, line in enumerate(lines) if "the 10 best of 12 pairs by validation"
                 in line)
    assert progress and max(progress) < table


def test_step_15_real_run_with_every_pair_scored_prints_no_not_scored_line(step_15_run):
    metric, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    assert "NOT scored" not in result.stdout
    pairs = pd.read_csv(folder / "pairs.csv")
    assert pairs.columns[-1] == "non-finite rows"
    assert (pairs["non-finite rows"] == 0).all()


# Step 15 with stage-2 settings whose m is NaN on one row: the same tiny run, with
# fit_stage2 / predict_amount wrapped (in the child process only) so a model remembers its
# settings and the poisoned ones predict a NaN m on one row of every prediction.
#
# Since U11.5b, WHERE matters: the stage-2 gates look at the paid validation rows only, so a NaN
# m on a PAID row fails that setting's m gate (it is never paired), while a NaN m on an UNPAID
# row passes the gates and still reaches score_pairs, which ranks the pair last (the
# "NOT scored" path). The first validation row is a paid one in this table.
STEP_15_POISON = """
_real_fit_stage2, _real_predict_amount = script.fit_stage2, script.predict_amount

def _fit_stage2(x, amount, exposure, settings):
    return (_real_fit_stage2(x, amount, exposure, settings), settings)

def _poisoned_position(m):
    if WHERE == "first row":
        return 0
    paid = table.loc[m.index, "paid"].astype("boolean").fillna(True).to_numpy(dtype=bool)
    return int(np.flatnonzero(~paid)[0])  # the first unpaid row

def _predict_amount(model, x, exposure):
    model, settings = model
    m = _real_predict_amount(model, x, exposure)
    if POISONED(settings):
        m = m.copy()
        m.iloc[_poisoned_position(m)] = float("nan")
    return m

script.fit_stage2, script.predict_amount = _fit_stage2, _predict_amount
copy = table.copy(deep=True)
"""


def run_poisoned_step_15(folder, poisoned, where="first unpaid row"):
    code = (STEP_15_RUN.replace("OUTPUT_FOLDER", repr(str(folder)))
            .replace("METRIC", repr("mape"))
            .replace("copy = table.copy(deep=True)\n",
                     STEP_15_POISON.replace("POISONED", poisoned)
                     .replace("WHERE", repr(where))))
    assert "_predict_amount" in code  # the patch went in
    return run_code(code)


@pytest.fixture(scope="module")
def step_15_gamma_poisoned(tmp_path_factory):
    # The 3 gamma settings (x 2 stage-1 settings = 6 pairs) give a NaN m on an unpaid row;
    # poisson ones, including the default [stage2], do not.
    folder = tmp_path_factory.mktemp("joint_gamma_nan")
    return folder, run_poisoned_step_15(folder, '(lambda s: s.loss == "gamma")')


NOT_SCORED_LINE = ("        6 pairs NOT scored (a NaN or infinite p x m on some rows; ranked last "
                   "with infinite scores):")


def test_step_15_some_non_finite_pairs_the_run_still_succeeds(step_15_gamma_poisoned):
    _, result = step_15_gamma_poisoned
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "run ok"


def test_step_15_some_non_finite_pairs_are_ranked_last_with_infinite_scores(
        step_15_gamma_poisoned):
    folder, result = step_15_gamma_poisoned
    assert result.returncode == 0, result.stderr
    pairs = pd.read_csv(folder / "pairs.csv")
    assert len(pairs) == 12
    assert (pairs["s2 loss"].iloc[:6] == "poisson").all()
    assert (pairs["s2 loss"].iloc[6:] == "gamma").all()
    assert (pairs["non-finite rows"].iloc[:6] == 0).all()
    assert (pairs["non-finite rows"].iloc[6:] == 1).all()
    for column in ("MAPE", "bias", "WAPE"):
        assert np.isfinite(pairs[column].iloc[:6]).all(), column
        assert (pairs[column].iloc[6:] == np.inf).all(), column


def test_step_15_some_non_finite_pairs_the_chosen_pair_is_a_scored_one(step_15_gamma_poisoned):
    folder, result = step_15_gamma_poisoned
    assert result.returncode == 0, result.stderr
    chosen = next(line for line in result.stdout.splitlines() if "chosen pair:" in line)
    assert "'loss': 'poisson'" in chosen and "gamma" not in chosen
    pairs = pd.read_csv(folder / "pairs.csv")
    scores = pd.read_csv(folder / "scores.csv")
    best = scores[(scores["part"] == "validation") & (scores["level"] == "consolidated")
                  & (scores["model"] == "chosen") & (scores["h"] == "all")]
    assert best["MAPE"].iloc[0] == pytest.approx(pairs["MAPE"].iloc[0], rel=1e-9)


def test_step_15_some_non_finite_pairs_prints_the_not_scored_block(step_15_gamma_poisoned):
    _, result = step_15_gamma_poisoned
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines.count(NOT_SCORED_LINE) == 1
    start = lines.index(NOT_SCORED_LINE)
    # After the 10 best (header + 10 rows), then a header and one line per failed pair.
    best = next(i for i, line in enumerate(lines) if "the 10 best of 12 pairs" in line)
    assert start == best + 12
    header, failed = lines[start + 1], lines[start + 2:start + 8]
    assert "non-finite rows" in header and "s2 loss" in header
    assert all("gamma" in line and "inf" in line for line in failed), failed
    assert all(line.split()[-1] == "1" for line in failed), failed
    # Exactly 6 failed rows: the next line is no longer part of the table.
    assert "gamma" not in lines[start + 8] and "inf" not in lines[start + 8]


def test_step_15_every_pair_non_finite_raises_and_writes_nothing(tmp_path):
    folder = tmp_path / "joint_all_nan"
    result = run_poisoned_step_15(folder, "(lambda s: True)")
    assert result.returncode != 0
    assert ("ValueError: step 15: every pair of the grid gave a NaN or infinite forecast on "
            "some validation rows, so none can be chosen") in result.stderr
    assert "run ok" not in result.stdout
    # Raised right after the scoring: no grid printout, no tables.
    assert "the 10 best of" not in result.stdout
    assert "NOT scored" not in result.stdout
    assert not folder.exists() or list(folder.iterdir()) == []


# --- Step 15, U11.5b: the eligibility gates ------------------------------------------------------
# The same tiny run. With the wide-open gates of STEP_15_RUN every setting passes (the tests
# above); here the GATES line is replaced to make some settings fail. Measured on this table:
# stage 1's two settings have slopes of about 1.69 (max_iter 3) and 1.12 (max_iter 8); stage 2's
# deviance skills run from about 0.17 to 0.59 with total ratios of 1.03 to 1.07.

import ast as _ast  # noqa: E402

WIDE_OPEN_GATES = ("GATES = dict(stage1_log_loss_tolerance=1e9, stage1_slope_band=(1e-9, 1e9), "
                   "stage2_skill_tolerance=1e9, stage2_ratio_band=(1e-9, 1e9))")
WIDE_OPEN = dict(stage1_log_loss_tolerance=1e9, stage1_slope_band=(1e-9, 1e9),
                 stage2_skill_tolerance=1e9, stage2_ratio_band=(1e-9, 1e9))
# The synthetic.toml values: on this table they keep one stage-1 setting and some stage-2 ones.
TIGHT_GATES = dict(stage1_log_loss_tolerance=0.10, stage1_slope_band=(0.8, 1.25),
                   stage2_skill_tolerance=0.05, stage2_ratio_band=(0.9, 1.1))
STAGE1_GATES_CSV_COLUMNS = ["stage1", "max_iter", "learning_rate", "max_leaf_nodes", "bad p",
                            "log loss", "log loss skill", "slope", "log loss gate", "skill gate",
                            "slope gate", "eligible"]
STAGE2_GATES_CSV_COLUMNS = ["stage2", "loss", "max_iter", "learning_rate", "max_leaf_nodes",
                            "scale", "balance_floor", "bad m", "deviance skill", "total ratio",
                            "m gate", "skill gate", "ratio gate", "eligible"]
STAGE1_PARAMS = ["max_iter", "learning_rate", "max_leaf_nodes"]
STAGE2_PARAMS = ["loss", "max_iter", "learning_rate", "max_leaf_nodes", "scale", "balance_floor"]


def run_step_15_with_gates(folder, gates, metric="mape"):
    assert WIDE_OPEN_GATES in STEP_15_RUN
    code = (STEP_15_RUN.replace("OUTPUT_FOLDER", repr(str(folder)))
            .replace("METRIC", repr(metric))
            .replace(WIDE_OPEN_GATES, f"GATES = {gates!r}"))
    return run_code(code)


def gate_csv(folder, stage):
    # keep_default_na=False: the "-" and "none" texts of the stage-2 params stay texts.
    return pd.read_csv(folder / f"stage{stage}_gates.csv", keep_default_na=False)


def progress_params(stdout, stage):
    """The params of each grid setting, from its progress line, in grid order."""
    return [_ast.literal_eval(m.group(4)) for m in grid_progress_lines(stdout)
            if m.group(1) == f"stage {stage}"]


def gate_line(stdout, stage):
    lines = [line for line in stdout.splitlines()
             if line.startswith(f"        stage {stage} gates: ")]
    assert len(lines) == 1, lines
    return lines[0]


def expected_gate_line(stage, gates):
    gate_columns = [c for c in gates.columns if c.endswith(" gate")]
    failing = {c: int((~gates[c].astype(bool)).sum()) for c in gate_columns}
    return (f"        stage {stage} gates: {int(gates['eligible'].sum())} of {len(gates)} "
            f"settings eligible; settings failing each gate: {failing}")


# wide-open gates (the step_15_run fixture: every setting eligible)

def test_step_15_real_run_prints_both_gate_lines_with_everything_eligible(step_15_run):
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    assert gate_line(result.stdout, 1) == (
        "        stage 1 gates: 2 of 2 settings eligible; settings failing each gate: "
        "{'log loss gate': 0, 'skill gate': 0, 'slope gate': 0}")
    assert gate_line(result.stdout, 2) == (
        "        stage 2 gates: 6 of 6 settings eligible; settings failing each gate: "
        "{'m gate': 0, 'skill gate': 0, 'ratio gate': 0}")


def test_step_15_real_run_gate_lines_come_after_the_grids_and_before_the_pairs(step_15_run):
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    last_progress = max(i for i, line in enumerate(lines) if GRID_PROGRESS.match(line))
    stage1 = lines.index(gate_line(result.stdout, 1))
    stage2 = lines.index(gate_line(result.stdout, 2))
    pairs = next(i for i, line in enumerate(lines) if "the 10 best of" in line)
    assert last_progress < stage1 < stage2 < pairs


def test_step_15_real_run_gate_tables_print_the_best_eligible_settings_without_gate_columns(
        step_15_run):
    _, _, result = step_15_run
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    stage1 = lines.index(gate_line(result.stdout, 1))
    assert lines[stage1 + 1].split() == ["stage1", "max_iter", "learning_rate", "max_leaf_nodes",
                                         "bad", "p", "log", "loss", "log", "loss", "skill",
                                         "slope"]
    # Both stage-1 settings, best (lowest) log loss first; neither has a bad p.
    rows1 = [line.split() for line in lines[stage1 + 2:stage1 + 4]]
    assert sorted(int(row[0]) for row in rows1) == [0, 1]
    assert [int(row[4]) for row in rows1] == [0, 0]
    assert float(rows1[0][5]) <= float(rows1[1][5])
    assert lines[stage1 + 4] == gate_line(result.stdout, 2)
    stage2 = stage1 + 4
    assert lines[stage2 + 1].split() == ["stage2", "loss", "max_iter", "learning_rate",
                                         "max_leaf_nodes", "scale", "balance_floor", "bad", "m",
                                         "deviance", "skill", "total", "ratio"]
    # The 5 best of 6, best (highest) deviance skill first; then the pairs table starts.
    rows2 = lines[stage2 + 2:stage2 + 7]
    skills = [float(line.split()[-2]) for line in rows2]
    assert skills == sorted(skills, reverse=True)
    assert "the 10 best of" in lines[stage2 + 7]
    for line in lines[stage1:stage2 + 7]:
        assert "True" not in line and "False" not in line, line


def test_step_15_real_run_gate_csvs_have_one_row_per_setting_with_its_grid_params(step_15_run):
    _, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    for stage, columns, params in ((1, STAGE1_GATES_CSV_COLUMNS, STAGE1_PARAMS),
                                   (2, STAGE2_GATES_CSV_COLUMNS, STAGE2_PARAMS)):
        gates = gate_csv(folder, stage)
        assert list(gates.columns) == columns
        grid = progress_params(result.stdout, stage)
        assert gates[f"stage{stage}"].tolist() == list(range(len(grid)))
        assert gates[params].to_dict("records") == grid


def test_step_15_real_run_gate_csvs_eligible_is_every_gate_passed(step_15_run):
    _, folder, result = step_15_run
    assert result.returncode == 0, result.stderr
    for stage in (1, 2):
        gates = gate_csv(folder, stage)
        gate_columns = [c for c in gates.columns if c.endswith(" gate")]
        assert len(gate_columns) == 3
        assert gates["eligible"].tolist() == gates[gate_columns].all(axis=1).tolist()
        assert gates["eligible"].all()
        assert gate_line(result.stdout, stage) == expected_gate_line(stage, gates)


# tight gates: some settings drop out

@pytest.fixture(scope="module")
def step_15_tight(tmp_path_factory):
    folder = tmp_path_factory.mktemp("joint_tight_gates")
    return folder, run_step_15_with_gates(folder, TIGHT_GATES)


def eligible_params(folder, stage, params):
    gates = gate_csv(folder, stage)
    return gates.loc[gates["eligible"].astype(bool), [f"stage{stage}", *params]]


def test_step_15_tight_gates_the_run_succeeds(step_15_tight):
    _, result = step_15_tight
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "run ok"


def test_step_15_tight_gates_drop_some_settings_of_each_stage_and_keep_some(step_15_tight):
    folder, result = step_15_tight
    assert result.returncode == 0, result.stderr
    eligible1 = eligible_params(folder, 1, STAGE1_PARAMS)["stage1"].tolist()
    eligible2 = eligible_params(folder, 2, STAGE2_PARAMS)["stage2"].tolist()
    assert 0 < len(eligible1) < 2 and 0 < len(eligible2) < 6, (eligible1, eligible2)
    # The grid numbers of the eligible settings are not simply 0, 1, ...: so a pair numbered
    # from 0 by score_pairs must be mapped back to its grid number to name the right setting.
    assert (eligible1 != list(range(len(eligible1)))
            or eligible2 != list(range(len(eligible2)))), (eligible1, eligible2)


def test_step_15_tight_gates_every_setting_is_still_fitted_and_gated(step_15_tight):
    folder, result = step_15_tight
    assert result.returncode == 0, result.stderr
    assert len(progress_params(result.stdout, 1)) == 2
    assert len(progress_params(result.stdout, 2)) == 6
    assert len(gate_csv(folder, 1)) == 2 and len(gate_csv(folder, 2)) == 6


def test_step_15_tight_gates_the_gate_lines_match_the_csvs(step_15_tight):
    folder, result = step_15_tight
    assert result.returncode == 0, result.stderr
    for stage in (1, 2):
        assert gate_line(result.stdout, stage) == expected_gate_line(stage, gate_csv(folder, stage))


def test_step_15_tight_gates_the_printed_best_settings_are_eligible_ones(step_15_tight):
    folder, result = step_15_tight
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    for stage in (1, 2):
        eligible = set(eligible_params(folder, stage, [])[f"stage{stage}"])
        start = lines.index(gate_line(result.stdout, stage)) + 2
        printed = [int(line.split()[0]) for line in lines[start:start + min(5, len(eligible))]]
        assert set(printed) == eligible, (stage, printed, eligible)


def test_step_15_tight_gates_pairs_are_exactly_the_eligible_by_eligible_settings(step_15_tight):
    folder, result = step_15_tight
    assert result.returncode == 0, result.stderr
    pairs = pd.read_csv(folder / "pairs.csv", keep_default_na=False)
    s1 = eligible_params(folder, 1, STAGE1_PARAMS)
    s2 = eligible_params(folder, 2, STAGE2_PARAMS)
    assert len(pairs) == len(s1) * len(s2)
    found = {(tuple(row[f"s1 {p}"] for p in STAGE1_PARAMS),
              tuple(row[f"s2 {p}"] for p in STAGE2_PARAMS)) for _, row in pairs.iterrows()}
    expected = {(tuple(a[p] for p in STAGE1_PARAMS), tuple(b[p] for p in STAGE2_PARAMS))
                for _, a in s1.iterrows() for _, b in s2.iterrows()}
    assert found == expected
    assert f"the 10 best of {len(pairs)} pairs by validation" in result.stdout


def test_step_15_tight_gates_the_chosen_pair_is_the_best_eligible_pair(step_15_tight):
    # The chosen pair is refitted from its grid numbers: if they were not mapped back from
    # score_pairs' 0-based numbers, the chosen pair would be another setting and its validation
    # score would not equal the best pair's.
    folder, result = step_15_tight
    assert result.returncode == 0, result.stderr
    pairs = pd.read_csv(folder / "pairs.csv", keep_default_na=False)
    scores = pd.read_csv(folder / "scores.csv")
    chosen = scores[(scores["part"] == "validation") & (scores["level"] == "consolidated")
                    & (scores["model"] == "chosen") & (scores["h"] == "all")]
    assert chosen["MAPE"].iloc[0] == pytest.approx(pairs["MAPE"].iloc[0], rel=1e-9)
    line = next(line for line in result.stdout.splitlines() if "chosen pair:" in line)
    s1 = {p: pairs[f"s1 {p}"].iloc[0].item() for p in STAGE1_PARAMS}
    s2 = {p: pairs[f"s2 {p}"].tolist()[0] for p in STAGE2_PARAMS}
    assert f"stage 1 {s1}" in line and f"stage 2 {s2}" in line, line


def test_step_15_tight_gates_writes_the_same_tables(step_15_tight):
    folder, result = step_15_tight
    assert result.returncode == 0, result.stderr
    assert {"pairs.csv", "stage1_gates.csv", "stage2_gates.csv"} <= {
        path.name for path in folder.iterdir()}


# no setting eligible at one stage: the run stops before scoring any pair

@pytest.mark.parametrize("changes, counts, failing_line", [
    ({"stage1_slope_band": (50.0, 100.0)}, (0, 6),
     "        stage 1 gates: 0 of 2 settings eligible; settings failing each gate: "
     "{'log loss gate': 0, 'skill gate': 0, 'slope gate': 2}"),
    ({"stage2_ratio_band": (50.0, 100.0)}, (2, 0),
     "        stage 2 gates: 0 of 6 settings eligible; settings failing each gate: "
     "{'m gate': 0, 'skill gate': 0, 'ratio gate': 6}"),
], ids=["no stage 1", "no stage 2"])
def test_step_15_no_eligible_setting_raises_and_writes_nothing(tmp_path, changes, counts,
                                                               failing_line):
    folder = tmp_path / "joint_none_eligible"
    result = run_step_15_with_gates(folder, {**WIDE_OPEN, **changes})
    assert result.returncode != 0
    assert (f"ValueError: step 15: {counts[0]} stage-1 and {counts[1]} stage-2 settings passed "
            f"the eligibility gates, so no pair can be scored") in result.stderr
    assert "run ok" not in result.stdout
    # The gates are printed (they say why), then nothing is scored or written.
    assert failing_line in result.stdout.splitlines()
    assert "the 10 best of" not in result.stdout
    assert "chosen pair:" not in result.stdout
    assert not folder.exists() or list(folder.iterdir()) == []


# a NaN m on a PAID validation row: the m gate catches it

@pytest.fixture(scope="module")
def step_15_gamma_poisoned_on_a_paid_row(tmp_path_factory):
    folder = tmp_path_factory.mktemp("joint_gamma_nan_paid")
    return folder, run_poisoned_step_15(folder, '(lambda s: s.loss == "gamma")',
                                        where="first row")


def test_step_15_nan_m_on_a_paid_row_fails_the_m_gate_and_the_run_succeeds(
        step_15_gamma_poisoned_on_a_paid_row):
    folder, result = step_15_gamma_poisoned_on_a_paid_row
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "run ok"
    gates = gate_csv(folder, 2)
    gamma = gates["loss"] == "gamma"
    assert gates.loc[gamma, "bad m"].tolist() == [1, 1, 1]
    assert gates.loc[~gamma, "bad m"].tolist() == [0, 0, 0]
    assert gates["eligible"].tolist() == (~gamma).tolist()
    assert gate_line(result.stdout, 2) == (
        "        stage 2 gates: 3 of 6 settings eligible; settings failing each gate: "
        "{'m gate': 3, 'skill gate': 3, 'ratio gate': 3}")


def test_step_15_nan_m_on_a_paid_row_its_settings_are_never_paired(
        step_15_gamma_poisoned_on_a_paid_row):
    folder, result = step_15_gamma_poisoned_on_a_paid_row
    assert result.returncode == 0, result.stderr
    pairs = pd.read_csv(folder / "pairs.csv")
    assert len(pairs) == 2 * 3
    assert set(pairs["s2 loss"]) == {"poisson"}
    assert (pairs["non-finite rows"] == 0).all()
    assert "NOT scored" not in result.stdout


def test_step_15_nan_m_on_a_paid_row_in_every_setting_raises_the_gate_error(tmp_path):
    folder = tmp_path / "joint_all_nan_paid"
    result = run_poisoned_step_15(folder, "(lambda s: True)", where="first row")
    assert result.returncode != 0
    assert ("ValueError: step 15: 2 stage-1 and 0 stage-2 settings passed the eligibility "
            "gates, so no pair can be scored") in result.stderr
    assert not folder.exists() or list(folder.iterdir()) == []


# Step 11, U7b: the holiday positions of every target week, with the [calendar] settings, over
# the same weeks as the calendar features, merged into the calendar block (on week, one to
# one), and their printout. Run on the real settings/holidays_tr.toml (read only), in a
# separate process like the checks above.
HOLIDAY_POSITIONS = """
import inspect
from datetime import date, timedelta
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle.features_calendar import CALENDAR_COLUMNS

assert script.holiday_position_features.__module__ == (
    "collection_estimation.hurdle.features_calendar")
inspect.signature(script.show_holiday_positions).bind("positions", "week_zero", "holidays")
source = " ".join(inspect.getsource(script.main).split())
call = ("positions = holiday_position_features( week_zero, 0, last_week + "
        "settings.receivables.horizons, holidays, settings.calendar.holiday_encodings, "
        "settings.calendar.drain_window, settings.calendar.catch_up_window)")
assert call in source, "holiday_position_features is not called as expected"
merge = 'calendar = calendar.merge(positions, on="week", how="left", validate="one_to_one")'
assert merge in source, "the positions are not merged into the calendar as expected"
show = "show_holiday_positions(positions, week_zero, holidays)"
assert show in source, "show_holiday_positions is not called as expected"
assert source.count("holiday_position_features(") == 1
assert source.count(show) == 1
order = [source.index("show_calendar(calendar, week_zero, holidays)"),
         source.index("holiday_position_features("), source.index(merge), source.index(show),
         source.index("receivables_features("),
         source.index("table = join_features(panel, history, calendar, owed)")]
assert order == sorted(order), order

holidays = script.load_holidays(script.REPO_ROOT / "settings" / "holidays_tr.toml")
week_zero = date(2025, 6, 2)
calendar = script.calendar_features(week_zero, 0, 69, holidays)
for encodings in (("drain",), ("days",), ("drain", "days"), ()):
    positions = script.holiday_position_features(week_zero, 0, 69, holidays, encodings, 3, 5)
    merged = calendar.merge(positions, on="week", how="left", validate="one_to_one")
    added = [column for column in positions.columns if column != "week"]
    assert list(merged.columns) == CALENDAR_COLUMNS + added, list(merged.columns)
    assert len(merged) == 70
    assert merged[added].notna().all().all()
    assert all(merged[column].dtype == "int64" for column in added)
    pd.testing.assert_frame_equal(merged[CALENDAR_COLUMNS], calendar)
    # The number of unusual weeks, counted here independently of the printout.
    if added:
        normal = pd.Series({c: 0 if c in ("drain_days", "catch_up_days") else 1 for c in added})
        unusual = int((positions[added] != normal).any(axis=1).sum())
    print(f"=== {encodings} unusual {unusual if added else '-'}")
    script.show_holiday_positions(positions, week_zero, holidays)
print("ok")
"""


@pytest.fixture(scope="module")
def holiday_positions_run():
    result = subprocess.run([sys.executable, "-c", HOLIDAY_POSITIONS], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    # The printout of each encoding, keyed by the encodings' text.
    parts = {}
    for chunk in result.stdout.strip().removesuffix("ok").split("=== ")[1:]:
        header, _, body = chunk.partition("\n")
        parts[header.split(" unusual ")[0]] = (header.split(" unusual ")[1], body)
    return parts


def test_holiday_positions_are_wired_in_merged_and_printed(holiday_positions_run):
    assert set(holiday_positions_run) == {"('drain',)", "('days',)", "('drain', 'days')", "()"}


def test_holiday_positions_printout_drain_counts_the_unusual_weeks(holiday_positions_run):
    unusual, body = holiday_positions_run["('drain',)"]
    assert unusual == "17"
    assert ("holiday positions: 2 columns; 17 of 70 weeks differ from a normal week:"
            in body)
    assert "np.int64" not in body


def test_holiday_positions_printout_drain_shows_the_known_weeks(holiday_positions_run):
    _, body = holiday_positions_run["('drain',)"]
    for line in ("week 41 (Monday 2026-03-16): drain_days 3, catch_up_days 0",
                 "week 42 (Monday 2026-03-23): drain_days 0, catch_up_days 3",
                 "week 50 (Monday 2026-05-18): drain_days 1, catch_up_days 3",
                 "week 51 (Monday 2026-05-25): drain_days 2, catch_up_days 0"):
        assert line in body, line
    # Victory Day, Sat 30 Aug 2025, changes nothing: weeks 12 and 13 are normal.
    assert "week 12 (" not in body and "week 13 (" not in body
    assert "Sat 30 Aug" not in body


def test_holiday_positions_printout_names_the_weekday_holidays_around_a_week(
        holiday_positions_run):
    _, body = holiday_positions_run["('drain',)"]
    line = next(line for line in body.splitlines() if "week 41 (" in line)
    assert "Fri 20 Mar" in line
    assert "Sat 21 Mar" not in line and "Sun 22 Mar" not in line


def test_holiday_positions_printout_days_counts_the_unusual_weeks(holiday_positions_run):
    unusual, body = holiday_positions_run["('days',)"]
    assert ("holiday positions: 15 columns; " + unusual + " of 70 weeks differ from a normal "
            "week:") in body
    assert unusual == "26"
    # No drain columns to show; the week is listed with its holidays.
    assert "drain_days" not in body
    assert "week 49 (Monday 2026-05-11)" in body  # next_tue = 0 (Tue 19 May)


def test_holiday_positions_printout_both_encodings(holiday_positions_run):
    unusual, body = holiday_positions_run["('drain', 'days')"]
    assert ("holiday positions: 17 columns; " + unusual + " of 70 weeks differ from a normal "
            "week:") in body
    assert "week 51 (Monday 2026-05-25): drain_days 2, catch_up_days 0" in body


def test_holiday_positions_printout_without_encodings(holiday_positions_run):
    _, body = holiday_positions_run["()"]
    assert body.strip() == "holiday positions: none ([calendar] holiday_encodings is empty)"


def test_holiday_positions_printout_lists_a_weeks_holidays_in_date_order(holiday_positions_run):
    # Week 50 sees Tue 19 May and the Kurban Bayramı Wed 27 .. Fri 29 May 2026: a date list
    # should read in date order, not sorted by the weekday's name.
    _, body = holiday_positions_run["('drain',)"]
    line = next(line for line in body.splitlines() if "week 50 (" in line)
    assert "Tue 19 May, Wed 27 May, Thu 28 May, Fri 29 May" in line, line


# U12.2: the top-ups at the diagnostic origin, by the trailing mean of TOPUP_MEAN_WINDOW weeks
# and by the ridge with and without the recent level, printed after the deflated label. The
# wiring is checked on main()'s source; the printout is run on hand-built side tables in a
# separate process. Two companies, origin 24, horizons 2, price levels that move every week (so
# deflating by the wrong week shows). TR04's G starts at week 17, too few weeks to fit, so its
# ridge falls back at every h and the fallbacks column counts it; TR02's G is all zero, forecast
# 0 and not counted. The trailing means and the actuals are computed in
# the script by plain loops over dicts, not by the module; the ridge columns come from
# `topup_forecasts` itself (tested in test_topups.py), summed over the companies.
TOPUPS = """
import inspect
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle import prices, topups

source = " ".join(inspect.getsource(script.main).split())
call = ("show_topups(new_cash, refunds, unidentified, levels, calendar, "
        "settings.diagnostic.origin, settings.receivables.horizons)")
assert call in source, "show_topups is not called as expected"
assert (source.index("show_deflated_label(panel, levels)") < source.index(call)
        < source.index("table = join_features(panel, history, calendar, owed)"))
for built in ("new_cash = new_customer_cash(", "refunds = refunds_by_week(",
              "unidentified = unidentified_cash_by_week(", "levels = published_levels(",
              "calendar = calendar.merge(positions"):
    assert source.index(built) < source.index(call), built
assert "show_trailing_means" not in source
assert not hasattr(script, "show_trailing_means")
assert not hasattr(script, "TOPUP_WINDOWS")
assert script.TOPUP_MEAN_WINDOW == 8
assert script.deflate is prices.deflate
assert script.topup_forecasts is topups.topup_forecasts
inspect.signature(script.show_topups).bind(
    "new_cash", "refunds", "unidentified", "levels", "calendar", 24, 2)

companies = ["TR02", "TR04"]
origin, horizons, window = 24, 2, script.TOPUP_MEAN_WINDOW
last = 30
level_of = {w: 1.0 + 0.1 * w + 0.05 * (w % 3) for w in range(0, last + 1)}  # by hand
levels = pd.Series(level_of)
calendar = pd.DataFrame({"week": range(0, last + 1),
                         "cycle": [float(w % 4) for w in range(0, last + 1)],
                         "odd": [float(w % 2) for w in range(0, last + 1)]})

new_rows = []
for company, scale in (("TR02", 1e6), ("TR04", 2.5e6)):
    for o in range(0, last):
        for h in (1, 2):
            if o + h <= last:
                new_rows.append({"company_code": company, "origin_week": o, "h": h,
                                 "target_week": o + h,
                                 "new_customer_cash": scale * (1 + (o + h) % 5) * h})
new_cash = pd.DataFrame(new_rows)
refund_rows = []
for company, scale in (("TR02", -1e5), ("TR04", -7e5)):
    for w in range(0, last + 1):
        refund_rows.append({"company_code": company, "week": w,
                            "refunds": scale * (2 + w % 3)})
refunds = pd.DataFrame(refund_rows)
g_rows = []
# TR02: G all zero (forecast 0, not flagged). TR04: G only from week 17, so 8 weeks up to
# the origin, fewer than MIN_TRAINING_WEEKS: its ridge falls back to the level, flagged.
for company, scale, first in (("TR02", 0.0, 0), ("TR04", 4e5, 17)):
    for w in range(first, last + 1):
        g_rows.append({"company_code": company, "week": w,
                       "unidentified_cash": scale * (1 + w % 4)})
unidentified = pd.DataFrame(g_rows)
copies = [t.copy(deep=True) for t in (new_cash, refunds, unidentified, calendar)]
levels_copy = levels.copy(deep=True)


def mean_by_hand(rows, week_key, value_key, h):
    total = 0.0
    for company in companies:
        values = [r[value_key] / level_of[r[week_key]] for r in rows
                  if r["company_code"] == company and (h is None or r["h"] == h)
                  and origin - window < r[week_key] <= origin]
        total += sum(values) / len(values)
    return total


def actual(rows, week_key, value_key, h, target):
    return sum(r[value_key] / level_of[r[week_key]] for r in rows
               if r[week_key] == target and (h is None or r["h"] == h))


expected = {}
for h in (1, 2):
    target = origin + h
    a_mean = mean_by_hand(new_rows, "target_week", "new_customer_cash", h)
    r_mean = mean_by_hand(refund_rows, "week", "refunds", None)
    g_mean = mean_by_hand(g_rows, "week", "unidentified_cash", None)
    a_actual = actual(new_rows, "target_week", "new_customer_cash", h, target)
    other_actual = (actual(refund_rows, "week", "refunds", None, target)
                    + actual(g_rows, "week", "unidentified_cash", None, target))
    line = {"A mean 8": a_mean / 1e6, "top-up mean 8": (a_mean + r_mean + g_mean) / 1e6,
            "A actual": a_actual / 1e6, "top-up actual": (a_actual + other_actual) / 1e6}
    for name, with_level in (("ridge", True), ("ridge no level", False)):
        found = topups.topup_forecasts(new_cash, refunds, unidentified, levels, calendar,
                                       [origin], horizons, "ridge", window, with_level)
        mine = found[found["h"] == h]
        line["A " + name] = float(mine["new_customer_cash"].sum()) / 1e6
        line["top-up " + name] = float(mine["topup"].sum()) / 1e6
        line["fallbacks " + name] = int(mine["fallback"].sum())
        # TR04's short G falls back; TR02's all-zero G is forecast 0 and not flagged, and
        # every other series keeps its fit: the count is exactly 1.
        assert mine.set_index("company_code")["fallback"].to_dict() == \
            {"TR02": False, "TR04": True}, (name, h, mine)
        by_company = mine.set_index("company_code")["unidentified_cash"]
        assert by_company["TR02"] == 0.0
        # TR04's fallback: the mean of its deflated G over the 4 weeks up to the origin.
        level = sum(r["unidentified_cash"] / level_of[r["week"]] for r in g_rows
                    if r["company_code"] == "TR04" and origin - 4 < r["week"] <= origin) / 4
        assert abs(by_company["TR04"] - level) < 1e-6 * level, (by_company["TR04"], level)
    expected[h] = line

script.show_topups(new_cash, refunds, unidentified, levels, calendar, origin, horizons)
for table, copy in zip((new_cash, refunds, unidentified, calendar), copies):
    pd.testing.assert_frame_equal(table, copy)
pd.testing.assert_series_equal(levels, levels_copy)
print("EXPECTED " + repr(expected))
print("ok")
"""

TOPUP_PRINTOUT_COLUMNS = ["A mean 8", "A ridge", "A ridge no level", "A actual",
                          "top-up mean 8", "top-up ridge", "top-up ridge no level",
                          "top-up actual", "fallbacks ridge", "fallbacks ridge no level"]


@pytest.fixture(scope="module")
def topups_run():
    result = subprocess.run([sys.executable, "-c", TOPUPS], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    lines = [line.strip() for line in result.stdout.splitlines()]
    expected = eval(next(line for line in lines if line.startswith("EXPECTED "))[9:])
    title = ("top-ups at origin 24, all companies, million TRY of week 0 (actual = what came, "
             "hindsight; fallbacks = companies whose ridge fell back to the level):")
    assert title in lines, result.stdout
    table = lines[lines.index(title) + 1:]
    end = next(i for i, line in enumerate(table) if line.startswith("EXPECTED "))
    return {"header": table[0], "index_name": table[1],
            "rows": [row.split() for row in table[2:4]], "after": table[4:end],
            "expected": expected}


def test_topups_are_wired_in_after_the_deflated_label(topups_run):
    # The source checks (call text, order, imports, TOPUP_MEAN_WINDOW, the old printout gone)
    # ran in the fixture.
    assert len(topups_run["rows"]) == 2


def test_topups_printout_columns_in_order(topups_run):
    assert topups_run["header"].split() == " ".join(TOPUP_PRINTOUT_COLUMNS).split()
    assert topups_run["index_name"] == "h"


def test_topups_printout_one_row_per_h(topups_run):
    assert [row[0] for row in topups_run["rows"]] == ["1", "2"]
    assert all(len(row) == 1 + len(TOPUP_PRINTOUT_COLUMNS) for row in topups_run["rows"])
    # Nothing else is printed after the two rows.
    assert topups_run["after"] == []


def test_topups_printout_matches_the_expected_values(topups_run):
    # The trailing means and actuals by hand (each company's own mean, deflated by the week
    # the cash came, summed over the companies), the ridge from topup_forecasts; million TRY
    # rounded to 3 places.
    for row, h in zip(topups_run["rows"], (1, 2)):
        printed = dict(zip(TOPUP_PRINTOUT_COLUMNS, (float(value) for value in row[1:])))
        for name in TOPUP_PRINTOUT_COLUMNS:
            assert printed[name] == pytest.approx(topups_run["expected"][h][name], abs=6e-4), \
                (h, name, printed[name], topups_run["expected"][h][name])


def test_topups_printout_counts_the_companies_that_fell_back(topups_run):
    # TR04's short G falls back in both ridges; TR02's all-zero G is not counted: one per h.
    for row in topups_run["rows"]:
        printed = dict(zip(TOPUP_PRINTOUT_COLUMNS, row[1:]))
        assert printed["fallbacks ridge"] == "1"
        assert printed["fallbacks ridge no level"] == "1"


def test_topups_printout_top_up_is_below_a_since_refunds_outweigh_g(topups_run):
    # R (negative) outweighs G here, so every top-up is below its A: the refunds' sign is kept.
    for row in topups_run["rows"]:
        printed = dict(zip(TOPUP_PRINTOUT_COLUMNS, (float(value) for value in row[1:])))
        for name in ("mean 8", "ridge", "ridge no level", "actual"):
            assert printed["top-up " + name] < printed["A " + name], name


def test_topups_printout_is_in_million_try(topups_run):
    value = float(topups_run["rows"][0][1 + TOPUP_PRINTOUT_COLUMNS.index("A mean 8")])
    assert 0.1 < abs(value) < 100.0
    assert value == pytest.approx(topups_run["expected"][1]["A mean 8"], abs=6e-4)
