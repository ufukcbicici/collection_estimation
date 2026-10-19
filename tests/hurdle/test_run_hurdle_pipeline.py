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


# Step 13 (U10.4a): the diagnostic's splits, and best_threshold + classification_scores on
# step 12's predictions. Wiring checked on main()'s syntax tree; the block itself is run in a
# separate process (exec of main()'s own `if settings.diagnostic.run:` statement) on hand-made
# frames.
# The table: T = 2, H = 2. Target weeks 0, 1, 2 (paid T, F, T) -> train 3 rows, 66.7% paid;
# 3 (T) and 4 (F) -> validation 2 rows, 50.0% (a missing_week row at week 3 is left out);
# refit 5 rows, 60.0%; 5, 6 (T, T) -> test 2 rows, 100.0%; week 7 is after the test weeks.
# The forecast rows and predictions are step 12's SCORES frame (four labelled rows, one
# missing_week row and one future row whose p would move every score if used):
#   model A labelled p .8 T, .4 F, .6 T, .2 F -> best threshold .6, F1 1; AUC 1, AP 1,
#     log loss -(ln .8 + ln .6)/2 = 0.367.
#   model B labelled p .3 T, .9 F, .6 T, .4 F -> thresholds .3: F1 2/3, .4: 0.4, .6: 0.5,
#     .9: 0 -> best .3, precision .5, recall 1; AUC 1 of 4 pairs = .25; AP .5*.5 + .5*.5 = .5;
#     log loss -(ln .3 + ln .1 + ln .6 + ln .6)/4 = 1.132.
STEP_13 = """
import ast
import inspect
import textwrap
from types import SimpleNamespace
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script

tree = ast.parse(textwrap.dedent(inspect.getsource(script.main)))
blocks = [node for node in tree.body[0].body
          if isinstance(node, ast.If) and ast.unparse(node.test) == "settings.diagnostic.run"]
assert len(blocks) == 1, len(blocks)
code = compile(ast.fix_missing_locations(ast.Module(body=blocks, type_ignores=[])),
               "step13", "exec")

weeks = [0, 1, 2, 3, 3, 4, 5, 6, 7]
statuses = ["labelled"] * 4 + ["missing_week"] + ["labelled"] * 4
table = pd.DataFrame({
    "key": [f"TR02-{i}" for i in range(len(weeks))],
    "target_week": np.array(weeks, dtype="int64"),
    "label_status": np.array(statuses, dtype=object),
    "paid": pd.array([True, False, True, True, None, False, True, True, False],
                     dtype="boolean"),
}, index=pd.Index(range(100, 100 + len(weeks))))
forecast_rows = pd.DataFrame({
    "key": ["TR02-A", "TR04-B", "TR02-C", "TR02-A", "TR04-B", "TR02-C"],
    "h": np.array([1, 1, 1, 2, 2, 2], dtype="int64"),
    "label_status": ["labelled", "labelled", "missing_week", "labelled", "labelled", "future"],
    "paid": pd.array([True, False, None, True, False, None], dtype="boolean"),
}, index=pd.Index([10, 11, 12, 13, 14, 15]))
predictions = {name: pd.Series(values, index=forecast_rows.index, name="p")
               for name, values in MODELS}
settings = SimpleNamespace(diagnostic=SimpleNamespace(run=RUN, origin=2),
                           receivables=SimpleNamespace(horizons=2))
copies = (table.copy(deep=True), forecast_rows.copy(deep=True),
          {n: p.copy() for n, p in predictions.items()})
namespace = dict(vars(script))
namespace.update(settings=settings, table=table, forecast_rows=forecast_rows,
                 predictions=predictions)
STUBS
exec(code, namespace)
pd.testing.assert_frame_equal(table, copies[0])
pd.testing.assert_frame_equal(forecast_rows, copies[1])
for name in predictions:
    pd.testing.assert_series_equal(predictions[name], copies[2][name])
print("ok")
"""

NO_CALLS = """
def refuse(*args, **kwargs):
    raise AssertionError("called although [diagnostic] run is false")
namespace.update(diagnostic_splits=refuse, show_diagnostic_splits=refuse,
                 best_threshold=refuse, classification_scores=refuse)
"""

STEP_13_SPLIT_LINES = [
    "Step 13: stage 1 diagnostic at origin 2 (diagnostic only, never the forecast)",
    "        train              3 rows,  66.7% paid (target <= 2)",
    "        validation         2 rows,  50.0% paid (2 < target <= 4)",
    "        refit              5 rows,  60.0% paid (target <= 4)",
    "        test               2 rows, 100.0% paid (4 < target <= 6)",
]
STEP_13_MODEL_A = ("        stage 1 (HGB) at step 12's origin, best F1 threshold 0.600: F1 1.000, "
                   "precision 1.000, recall 1.000, AUC 1.000, avg precision 1.000, "
                   "log loss 0.367")
STEP_13_MODEL_B = ("        baseline (LR) at step 12's origin, best F1 threshold 0.300: F1 0.667, "
                   "precision 0.500, recall 1.000, AUC 0.250, avg precision 0.500, "
                   "log loss 1.132")


def run_step_13(models=None, run=True, stubs=""):
    models = TWO_MODELS if models is None else models
    models_text = "[" + ", ".join(f"({name!r}, {values})" for name, values in models) + "]"
    script = (STEP_13.replace("MODELS", models_text).replace("RUN", str(run))
              .replace("STUBS", stubs))
    return subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True,
                          text=True, encoding="utf-8", timeout=120)


def test_step_13_prints_the_splits_then_one_line_per_model_by_hand():
    result = run_step_13()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == STEP_13_SPLIT_LINES + [STEP_13_MODEL_A,
                                                                STEP_13_MODEL_B, "ok"]


def test_step_13_one_model():
    result = run_step_13(ONE_MODEL)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == STEP_13_SPLIT_LINES + [STEP_13_MODEL_A, "ok"]


def test_step_13_models_follow_the_order_of_the_predictions_dict():
    result = run_step_13(list(reversed(TWO_MODELS)))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[5:] == [STEP_13_MODEL_B, STEP_13_MODEL_A, "ok"]


def test_step_13_scores_each_model_on_its_own_p():
    result = run_step_13([("stage 1 (HGB)", MODEL_B), ("baseline (LR)", MODEL_A)])
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[5] == STEP_13_MODEL_B.replace("baseline (LR)", "stage 1 (HGB)")
    assert lines[6] == STEP_13_MODEL_A.replace("stage 1 (HGB)", "baseline (LR)")


def test_step_13_does_not_print_the_threshold_twice():
    result = run_step_13()
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines()[5:7]:
        assert line.count("threshold") == 1, line


def test_step_13_with_run_false_prints_and_calls_nothing():
    result = run_step_13(run=False, stubs=NO_CALLS)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["ok"]


def test_step_13_with_run_true_calls_go_through_the_stubs_check():
    # The NO_CALLS stubs do bite when run is true (so the run-false test is meaningful).
    result = run_step_13(run=True, stubs=NO_CALLS)
    assert result.returncode != 0
    assert "called although [diagnostic] run is false" in result.stderr


STEP_13_WIRING = """
import ast
import inspect
import textwrap
import run_hurdle_pipeline as script
from collection_estimation.hurdle import diagnostic

assert script.diagnostic_splits is diagnostic.diagnostic_splits
assert script.best_threshold is diagnostic.best_threshold
assert script.classification_scores is diagnostic.classification_scores
inspect.signature(script.show_diagnostic_splits).bind("splits", 40, 5)

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
assert inner[:4] == [
    "splits = diagnostic_splits(table, settings.diagnostic.origin, "
    "settings.receivables.horizons)",
    "show_diagnostic_splits(splits, settings.diagnostic.origin, settings.receivables.horizons)",
    "labelled = forecast_rows['label_status'] == 'labelled'",
    "actual = forecast_rows.loc[labelled, 'paid'].astype(bool)",
], inner[:4]
assert len(block.body) == 5 and isinstance(block.body[4], ast.For)
loop = block.body[4]
assert ast.unparse(loop.target) == "(name, p)"
assert ast.unparse(loop.iter) == "predictions.items()"
loop_body = [ast.unparse(node) for node in loop.body]
# First, the skip: fewer than two outcomes among the labelled rows -> print, next model.
skip = loop.body[0]
assert isinstance(skip, ast.If), loop_body
assert ast.unparse(skip.test) == "actual.nunique() < 2", ast.unparse(skip.test)
assert skip.orelse == []
assert len(skip.body) == 2, [ast.unparse(node) for node in skip.body]
assert isinstance(skip.body[0], ast.Expr) and isinstance(skip.body[0].value, ast.Call)
assert ast.unparse(skip.body[0].value.func) == "print"
assert isinstance(skip.body[1], ast.Continue)
assert "best_threshold" not in ast.unparse(skip)
assert "classification_scores" not in ast.unparse(skip)
assert loop_body[1] == "threshold, f1 = best_threshold(actual, p[labelled])", loop_body
assert loop_body[2] == ("scores = classification_scores(actual, p[labelled], threshold)"), \\
    loop_body
# The diagnostic uses its own origin, never step 12's demo origin.
assert "demo_origin" not in ast.unparse(block)
print("ok")
"""


def test_step_13_is_wired_in_after_step_12s_scores():
    result = subprocess.run([sys.executable, "-c", STEP_13_WIRING], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_step_13_block_is_the_only_use_of_the_diagnostic_functions():
    tree = main_tree()
    for name in ("diagnostic_splits", "show_diagnostic_splits", "best_threshold",
                 "classification_scores"):
        assert len(calls_to(tree, name)) == 1, name


# Step 13's skip: when step 12's labelled forecast rows hold one outcome only (all paid, all
# unpaid) or none at all (no labelled rows, so actual is empty), best_threshold cannot choose a
# threshold. The loop prints one skip line per model, in model order, and goes on; it calls
# neither best_threshold nor classification_scores (stubbed here to raise). The splits are
# still printed (they come from the table, which is unchanged). The forecast rows are changed
# in place before the block runs, and the no-mutation check's copy is taken again after that.
ONE_OUTCOME_ROWS = {
    "all paid": (
        '["labelled", "labelled", "missing_week", "labelled", "labelled", "future"]',
        "[True, True, None, True, True, None]"),
    "all unpaid": (
        '["labelled", "labelled", "missing_week", "labelled", "labelled", "future"]',
        "[False, False, None, False, False, None]"),
    "no labelled rows": (
        '["missing_week", "future", "missing_week", "future", "missing_week", "future"]',
        "[True, False, None, True, False, None]"),
}

REFUSE_SCORING = """
def refuse_scoring(*args, **kwargs):
    raise AssertionError("scoring called although there are too few outcomes")
namespace.update(best_threshold=refuse_scoring, classification_scores=refuse_scoring)
"""


def one_outcome_stubs(case, refuse=True):
    statuses, paid = ONE_OUTCOME_ROWS[case]
    stubs = (f"forecast_rows['label_status'] = {statuses}\n"
             f"forecast_rows['paid'] = pd.array({paid}, dtype='boolean')\n"
             "copies = (copies[0], forecast_rows.copy(deep=True), copies[2])\n")
    return stubs + (REFUSE_SCORING if refuse else "")


def skip_line(name):
    return f"        {name} at step 12's origin: too few outcomes to choose a threshold"


@pytest.mark.parametrize("case", list(ONE_OUTCOME_ROWS))
def test_step_13_too_few_outcomes_does_not_raise(case):
    result = run_step_13(stubs=one_outcome_stubs(case))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("case", list(ONE_OUTCOME_ROWS))
def test_step_13_too_few_outcomes_prints_the_splits_then_one_skip_line_per_model(case):
    result = run_step_13(stubs=one_outcome_stubs(case))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == STEP_13_SPLIT_LINES + [
        skip_line("stage 1 (HGB)"), skip_line("baseline (LR)"), "ok"]


@pytest.mark.parametrize("case", list(ONE_OUTCOME_ROWS))
def test_step_13_too_few_outcomes_skip_lines_follow_the_model_order(case):
    result = run_step_13(list(reversed(TWO_MODELS)), stubs=one_outcome_stubs(case))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[5:] == [
        skip_line("baseline (LR)"), skip_line("stage 1 (HGB)"), "ok"]


@pytest.mark.parametrize("case", list(ONE_OUTCOME_ROWS))
def test_step_13_too_few_outcomes_one_model_prints_one_skip_line(case):
    result = run_step_13(ONE_MODEL, stubs=one_outcome_stubs(case))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[5:] == [skip_line("stage 1 (HGB)"), "ok"]


@pytest.mark.parametrize("case", list(ONE_OUTCOME_ROWS))
def test_step_13_too_few_outcomes_prints_no_threshold_or_score(case):
    result = run_step_13(stubs=one_outcome_stubs(case))
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines()[5:]:
        assert "best F1 threshold" not in line, line
        assert "log loss" not in line, line
    assert result.stdout.count("too few outcomes") == 2


@pytest.mark.parametrize("case", list(ONE_OUTCOME_ROWS))
def test_step_13_too_few_outcomes_without_stubs_gives_the_same_output(case):
    # The real best_threshold / classification_scores are not reached either.
    stubbed = run_step_13(stubs=one_outcome_stubs(case))
    real = run_step_13(stubs=one_outcome_stubs(case, refuse=False))
    assert real.returncode == 0, real.stderr
    assert real.stdout == stubbed.stdout


def test_step_13_refuse_scoring_stubs_bite_with_two_outcomes():
    # The stubs do raise once the loop reaches best_threshold (so the skip tests mean something).
    result = run_step_13(stubs=REFUSE_SCORING)
    assert result.returncode != 0
    assert "scoring called although there are too few outcomes" in result.stderr


def test_step_13_two_outcomes_prints_no_skip_line():
    result = run_step_13()
    assert result.returncode == 0, result.stderr
    assert "too few outcomes" not in result.stdout
    assert result.stdout.splitlines() == STEP_13_SPLIT_LINES + [STEP_13_MODEL_A,
                                                                STEP_13_MODEL_B, "ok"]


def test_step_13_one_paid_and_one_unpaid_labelled_row_is_scored_not_skipped():
    # Two outcomes is enough: labelled rows 10 (paid, p .8) and 11 (unpaid, p .4) only.
    stubs = ("forecast_rows['label_status'] = "
             "['labelled', 'labelled', 'missing_week', 'future', 'future', 'future']\n"
             "copies = (copies[0], forecast_rows.copy(deep=True), copies[2])\n")
    result = run_step_13(ONE_MODEL, stubs=stubs)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "too few outcomes" not in result.stdout
    assert lines[5].startswith("        stage 1 (HGB) at step 12's origin, best F1 threshold "
                               "0.800: F1 1.000"), lines[5]
