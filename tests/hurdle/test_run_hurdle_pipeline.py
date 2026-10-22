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
DIAGNOSTIC_END = "settings.diagnostic.origin + 2 * settings.receivables.horizons"
DIAGNOSTIC_GUARD_TEST = "settings.diagnostic.run and diagnostic_end > int(weeks['week'].max())"


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
                         weeks=None, with_demo_guard=False):
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
    guard = source.index("if settings.diagnostic.run and diagnostic_end")
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
inspect.signature(script.show_stage2_fit).bind("origin", "paid_rows", "train_rows", "paid_x",
                                               "loss", 1.0)

body = ast.parse(textwrap.dedent(inspect.getsource(script.main))).body[0].body
statements = [ast.unparse(node) for node in body]
diagnostic = [i for i, node in enumerate(body)
              if isinstance(node, ast.If) and ast.unparse(node.test) == "settings.diagnostic.run"]
assert len(diagnostic) == 1, diagnostic
step_14 = statements[diagnostic[0] + 1:]
assert step_14 == [
    "stage2_columns = feature_columns(block_feature_columns(history, calendar, owed), "
    "settings.stage2.feature_blocks)",
    "paid_rows = train_rows[train_rows['paid'].astype(bool)]",
    "paid_x = stage1_features(paid_rows, stage2_columns, companies, "
    "settings.stage1.company_as_feature)",
    "started = time.perf_counter()",
    "stage2_model = fit_stage2(paid_x, paid_rows['amount_deflated'], settings.stage2)",
    "show_stage2_fit(origin, paid_rows, train_rows, paid_x, settings.stage2.loss, "
    "time.perf_counter() - started)",
], step_14
# Step 14 uses step 12's origin, training rows and companies: nothing reassigns them after
# step 12 (step 13 works on its own rows and origin).
def assigned(name):
    return [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
for name in ("origin", "train_rows", "companies"):
    assert len(assigned(name)) == 1, name
    assert assigned(name)[0] < diagnostic[0], name
for node in ast.walk(body[diagnostic[0]]):
    if isinstance(node, (ast.Assign, ast.For)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        assert not names & {"origin", "train_rows", "companies", "history", "calendar",
                            "owed"}, names
print("ok")
"""


def test_step_14_is_wired_in_last_after_step_13():
    result = run_code(STEP_14_WIRING)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_step_14_calls_fit_stage2_and_its_printout_once():
    tree = main_tree()
    assert len(calls_to(tree, "fit_stage2")) == 1
    assert len(calls_to(tree, "show_stage2_fit")) == 1


def test_step_14_never_fits_stage_2_with_the_stage_1_settings():
    source = main_source()
    assert "fit_stage2(paid_x, paid_rows['amount_deflated'], settings.stage1)" not in source
    assert 'fit_stage2(paid_x, paid_rows["amount_deflated"], settings.stage1)' not in source
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
                                 min_samples_leaf=1, random_seed=1, feature_blocks=("history",))
stage2_columns = script.feature_columns(script.block_feature_columns(history, calendar, owed),
                                        settings_stage2.feature_blocks)
assert stage2_columns == ["h", "mean_receipts_4"], stage2_columns
paid_rows = train_rows[train_rows["paid"].astype(bool)]
paid_x = script.stage1_features(paid_rows, stage2_columns, companies, COMPANY_AS_FEATURE)
started = time.perf_counter()
stage2_model = script.fit_stage2(paid_x, paid_rows["amount_deflated"], settings_stage2)
script.show_stage2_fit(origin, paid_rows, train_rows, paid_x, settings_stage2.loss,
                       time.perf_counter() - started)
assert len(paid_rows) == 3 and paid_rows["key"].eq("TR02-A").all()
m = stage2_model.predict(paid_x)
assert abs(m - 100.0).max() < 1e-6, m
print("chain ok")
"""


def run_step_14_chain(company=True, loss="poisson"):
    stage1_part = STAGE1.rsplit('print("ok")', 1)[0]
    code = (stage1_part + STEP_14_CHAIN).replace("COMPANY_AS_FEATURE", str(company)) \
        .replace("ORIGIN", "2").replace("LOSS", repr(loss))
    return run_code(code)


@pytest.mark.parametrize("company", [True, False])
@pytest.mark.parametrize("loss", ["poisson", "gamma", "squared_error"])
def test_step_14_real_chain_on_the_tiny_table(company, loss):
    result = run_step_14_chain(company=company, loss=loss)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "chain ok"
    step_14 = [line for line in lines if line.startswith("Step 14")]
    assert len(step_14) == 1
    columns = 3 if company else 2
    assert step_14[0].startswith(f"Step 14: stage 2 at forecast week 2: 3 paid training rows "
                                 f"(of 6), {columns} input columns, loss {loss!r}, fitted in ")
    following = lines[lines.index(step_14[0]) + 1]
    assert following == ("        deflated amount per paid row: median 100, mean 100, 99th "
                         "percentile 100, max 100")
