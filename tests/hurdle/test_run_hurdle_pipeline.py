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
