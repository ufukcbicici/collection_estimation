"""Tests of `hurdle/reports/preparation.py`: `show_preparation` and the printouts of steps 1-11.

The printouts are run on small hand-built tables, or on what the package's own functions make
of them, in a separate process (`run_snippet`); `show_preparation`'s wiring is checked on its
syntax tree, read from the file without importing it. What `prepare_inputs` computes is tested
on real tiny data in tests/hurdle/test_inputs.py.
"""
import ast

import pytest

from tests.hurdle.code_checks import HURDLE, calls_to, function_source, function_tree, run_snippet

PREPARATION = HURDLE / "reports" / "preparation.py"
INPUTS_MODULE = HURDLE / "inputs.py"

# The printouts of steps 1-11, in the order main() used to make them. The top-ups
# (show_topups, compare_topups) are experiments of the script, called by main() itself.
PREPARATION_PRINTOUTS = [
    "show_daily_files", "show_one_day", "show_collections", "show_keys", "show_ledger",
    "show_tie", "show_filled", "print", "show_weeks", "show_weekly", "show_rates",
    "show_receivables_by_week", "show_known", "show_panel_rows", "show_labels",
    "show_new_customer_cash", "show_refunds", "print", "print", "show_prices",
    "show_projected_levels", "show_early_receipts", "show_matrix", "show_history",
    "show_calendar", "show_holiday_positions", "show_owed", "show_deflated_label",
    "show_table"]


def show_preparation_tree():
    return function_tree(PREPARATION, "show_preparation")


def show_preparation_source():
    return function_source(PREPARATION, "show_preparation")


def prepare_tree():
    return function_tree(INPUTS_MODULE, "prepare_inputs")


def prepare_source():
    return function_source(INPUTS_MODULE, "prepare_inputs")


def printouts_in_order(tree):
    """The names of the show_* / print calls in `tree`, in source order."""
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and (node.func.id.startswith("show_") or node.func.id == "print")]
    return [call.func.id for call in sorted(calls, key=lambda c: (c.lineno, c.col_offset))]


def call_args(tree, name):
    calls = calls_to(tree, name)
    assert len(calls) == 1, (name, len(calls))
    return [ast.unparse(arg) for arg in calls[0].args]


# Step 6, now in `inputs.prepare_inputs` (steps 1-11 moved there from main()), with every
# reader before it replaced by a stub in the inputs module, so that no file is read and
# nothing is written (no corpus, no ledger, no data/interim cache). prepare_inputs is stopped
# at week_calendar, the first call after step 6 that receives the filled days. The step-6
# printout is then made from what prepare_inputs passed on, with show_filled;
# that show_preparation passes the same three values is checked on its syntax tree below.
STEP_6 = """
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from collection_estimation.hurdle.reports import preparation as reports
from collection_estimation.hurdle import inputs as prep

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
    settings_file=Path("settings") / "stub.toml",
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

def fake_add_week_index(collections, week_zero):
    seen["week_index_input"] = collections
    return collections.assign(week=0)

def fake_week_calendar(days, week_zero, holidays, filled_days=None):
    seen["calendar_filled_days"] = filled_days
    raise Stop

prep.list_daily_files = lambda folder, fmt: {{day: f"file-{{day}}" for day in DAYS}}
prep.load_collections = lambda files, layout, cache: COLLECTIONS
prep.add_keys = lambda table, companies: table
prep.load_ledger = lambda path, as_of, layout, cache: LEDGER
prep.check_ledger_against_collections = lambda ledger, collections, days: {{}}
prep.fill_missing_days = fake_fill
prep.add_week_index = fake_add_week_index
prep.load_holidays = lambda path: []
prep.week_calendar = fake_week_calendar

try:
    prep.prepare_inputs(settings, with_details=True)
except Stop:
    pass
else:
    raise AssertionError("prepare_inputs() did not reach week_calendar")

collections, filled_days = seen["week_index_input"], seen["calendar_filled_days"]
reports.show_filled(collections, filled_days, settings.data.fill_missing_days_from_ledger)
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
    return run_snippet(STEP_6.format(switch=switch))


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


def test_every_step_1_to_11_printout_is_a_function_of_the_module():
    # They used to be functions of run_hurdle_pipeline.py; show_preparation calls them here.
    from collection_estimation.hurdle.reports import preparation
    # U18.1 added show_cleared_invoice_delays after show_receivables_by_week: it is called by
    # main(), not by show_preparation.
    names = [node.name for node in ast.parse(PREPARATION.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef)]
    expected = ["show_preparation"] + list(dict.fromkeys(
        name for name in PREPARATION_PRINTOUTS if name != "print"))
    expected.insert(expected.index("show_receivables_by_week") + 1,
                    "show_cleared_invoice_delays")
    assert names == expected
    for name in names:
        assert callable(getattr(preparation, name)), name


def test_show_preparation_reads_the_one_day_file_and_the_price_weeks_as_it_calls_them():
    import inspect
    from collection_estimation.hurdle import prices, read_collections
    from collection_estimation.hurdle.reports import preparation
    assert preparation.read_one_file is read_collections.read_one_file
    assert preparation.price_weeks is prices.price_weeks
    inspect.signature(preparation.read_one_file).bind("path", "layout")
    inspect.signature(preparation.price_weeks).bind("week_zero", -32, 64)


def test_show_preparation_has_three_parameters():
    tree = show_preparation_tree()
    assert [a.arg for a in tree.args.args] == ["inputs", "details", "settings"]
    assert tree.args.defaults == [] and tree.args.vararg is None
    assert tree.args.kwonlyargs == []


def test_show_preparation_printouts_in_the_old_step_order():
    assert printouts_in_order(show_preparation_tree()) == PREPARATION_PRINTOUTS


def test_show_preparation_step_6_printout_gets_the_filled_collections_and_days():
    assert call_args(show_preparation_tree(), "show_filled") == [
        "details.collections", "details.filled_days",
        "settings.data.fill_missing_days_from_ledger"]


def test_show_preparation_steps_3_and_4_printouts_get_the_keyed_lines():
    tree = show_preparation_tree()
    assert call_args(tree, "show_collections") == ["details.collections_keyed"]
    assert call_args(tree, "show_keys") == ["details.collections_keyed"]


def test_show_preparation_step_9_known_printout_keeps_the_placeholders():
    # details.known is every key known, placeholders too (as main() printed it before the
    # placeholders were split off); the labels use the identified weekly amounts.
    assert call_args(show_preparation_tree(), "show_known") == [
        "details.known", "details.collections", "inputs.last_week"]
    assert call_args(show_preparation_tree(), "show_labels") == [
        "details.panel", "details.identified_weekly"]
    assert "known=known_all" in prepare_source()
    assert "identified_weekly=identified_weekly" in prepare_source()


def test_show_preparation_step_2_reads_the_one_day_file_itself():
    tree = show_preparation_tree()
    assert call_args(tree, "read_one_file") == ["details.daily_files[day]",
                                                "settings.collections_layout"]
    assert calls_to(prepare_tree(), "read_one_file") == []
    guard = [node for node in ast.walk(tree) if isinstance(node, ast.If)
             and ast.unparse(node.test) == "day not in details.daily_files"]
    assert len(guard) == 1
    assert isinstance(guard[0].body[0], ast.Raise)
    assert ast.unparse(guard[0].body[0].exc.func) == "ValueError"
    shows = show_preparation_source()
    assert shows.index("if day not in details.daily_files") < shows.index("read_one_file(")


def test_show_preparation_step_10_inflation_figures_are_made_for_the_printout_only():
    tree = show_preparation_tree()
    for name in ("weekly_price_level", "published_price_level", "projected_price_level"):
        found = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == f"inflation.{name}"]
        assert len(found) == 1, name
        assert [ast.unparse(a) for a in found[0].args] == ["weeks_for_prices"], name
    assert "inflation" not in prepare_source()
    assert call_args(tree, "show_prices") == ["weeks_for_prices", "actual_level",
                                              "published_level", "projected_level",
                                              "inputs.levels"]


def test_show_preparation_projected_levels_at_the_demo_origin_and_the_last_week():
    loops = [node for node in ast.walk(show_preparation_tree()) if isinstance(node, ast.For)]
    assert len(loops) == 1
    assert ast.unparse(loops[0].iter) == "(settings.debug.demo_origin, inputs.last_week)"
    assert [ast.unparse(n) for n in loops[0].body] == [
        "show_projected_levels(inputs.week_zero, forecast_week, settings.receivables.horizons)"]


SHOW_PREPARATION_ONE_DAY = """
from datetime import date
from types import SimpleNamespace
from pathlib import Path
from collection_estimation.hurdle.reports import preparation as reports

shown = []
reports.show_daily_files = lambda files: shown.append(files)
details = SimpleNamespace(daily_files={date(2026, 3, 2): Path("a.xlsx")})
settings = SimpleNamespace(settings_file=Path("settings") / "stub.toml",
                           debug=SimpleNamespace(one_day=date(2026, 3, 3)))
try:
    reports.show_preparation(SimpleNamespace(), details, settings)
except ValueError as error:
    print("ERROR " + str(error))
assert shown == [details.daily_files], shown
print("ok")
"""


def test_show_preparation_one_day_without_a_file_raises_after_listing_the_files():
    result = run_snippet(SHOW_PREPARATION_ONE_DAY)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "ERROR stub.toml [debug]: one_day 2026-03-03 has no daily file", "ok"]


# Step 11, U6.2: the history features, from the weekly matrix, the known keys, the last week
# and the windows from the settings; and the printout of them. Run on a small hand-built
# matrix, in a separate process.
HISTORY = """
import inspect
import pandas as pd
from collection_estimation.hurdle.reports import preparation as reports
from collection_estimation.hurdle import inputs as prep

inspect.signature(prep.history_features).bind("matrix", "known", 64, (4, 13, 52))
try:
    inspect.signature(prep.history_features).bind("matrix", "known", 64)
except TypeError:
    pass
else:
    raise AssertionError("history_features accepts a call without the windows")
assert callable(reports.show_history)
calc = " ".join(inspect.getsource(prep.prepare_inputs).split())
shows = " ".join(inspect.getsource(reports.show_preparation).split())
assert "history = history_features(matrix, known, last_week, settings.history.windows)" in calc
assert "show_history(details.history, details.matrix)" in shows
assert calc.index("matrix = weekly_matrix(") < calc.index("history_features(")
assert shows.index("show_matrix(details.matrix)") < shows.index("show_history(")

matrix = pd.DataFrame(0.0, index=pd.Index(["A", "B"], name="key", dtype=object),
                      columns=pd.Index(list(range(0, 14)), name="week"))
matrix.loc["A", 2] = 20.0
matrix.loc["A", 8] = 50.0
matrix.loc["B", 13] = 30.0
known = pd.DataFrame({"key": pd.Series(["A", "B"], dtype=object), "known_week": [-5, 9]})
history = prep.history_features(matrix, known, 13, (4, 13))
reports.show_history(history, matrix)
print("ok")
"""


def test_history_features_are_wired_in_and_printed():
    result = run_snippet(HISTORY)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    # A: origin weeks 0..13 (14 rows), B: 9..13 (5 rows); 12 columns less key and origin.
    assert "history features: 19 rows (key x origin week), 10 features" in result.stdout
    assert "share NaN" in result.stdout
    assert "the key with the most receipts at origin weeks 0, 13 and 52" in result.stdout
    assert "median_payment_13" in result.stdout


# Step 11, U7: the calendar features of every target week, from week zero, week 0 to the
# last week + the horizons, and the holidays; and the printout of them. Run on the real
# settings/holidays_tr.toml (read only), in a separate process.
CALENDAR = """
import inspect
from datetime import date
from pathlib import Path
from collection_estimation.hurdle.reports import preparation as reports
from collection_estimation.hurdle import inputs as prep

inspect.signature(prep.calendar_features).bind("week_zero", 0, 69, "holidays")
try:
    inspect.signature(prep.calendar_features).bind("week_zero", 0, 69)
except TypeError:
    pass
else:
    raise AssertionError("calendar_features accepts a call without the holidays")
inspect.signature(reports.show_calendar).bind("calendar", "week_zero", "holidays")
calc = " ".join(inspect.getsource(prep.prepare_inputs).split())
shows = " ".join(inspect.getsource(reports.show_preparation).split())
assert "horizons = settings.receivables.horizons" in calc
call = "calendar = calendar_features(week_zero, 0, last_week + horizons, holidays)"
assert call in calc, "calendar_features is not called as expected"
assert "show_calendar(inputs.calendar, inputs.week_zero, details.holidays)" in shows
assert calc.index("history = history_features(") < calc.index("calendar_features(")
assert shows.index("show_history(details.history") < shows.index("show_calendar(")

holidays = prep.load_holidays(Path("settings") / "holidays_tr.toml")
week_zero = date(2025, 6, 2)
calendar = prep.calendar_features(week_zero, 0, 69, holidays)
reports.show_calendar(calendar, week_zero, holidays)
print("ok")
"""


def test_calendar_features_are_wired_in_and_printed():
    # The holiday names are Turkish: run_snippet reads the output as UTF-8.
    result = run_snippet(CALENDAR)
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
# process.
RECEIVABLES = """
import inspect
import pandas as pd
from collection_estimation.hurdle.reports import preparation as reports
from collection_estimation.hurdle import inputs as prep
from collection_estimation.hurdle.receivables_asof import receivables_columns

calc = " ".join(inspect.getsource(prep.prepare_inputs).split())
shows = " ".join(inspect.getsource(reports.show_preparation).split())
assert ("leak_check_weeks=settings.receivables.leak_check_weeks, "
        "age_edges=settings.receivables.overdue_age_days)") in calc, "step 8 call"
assert ("show_receivables_by_week(details.receivables, settings.receivables.leak_check_weeks, "
        "settings.debug.sample_week, settings.receivables.overdue_age_days)") in shows
call = ("owed = receivables_features(panel, receivables, levels, horizons, "
        "settings.receivables.overdue_age_days)")
assert call in calc, "receivables_features is not called as expected"
show = ("show_owed(details.owed, int(before_first_ledger_week(details.panel, "
        "details.receivables).sum()))")
assert show in shows, "show_owed is not called with the not-yet-in-ledger count"
assert calc.index("calendar = calendar_features(") < calc.index("receivables_features(")
assert shows.index("show_calendar(") < shows.index("show_owed(details.owed")
assert callable(reports.before_first_ledger_week)
inspect.signature(reports.show_owed).bind("owed", 0)
assert calc.index("receivables_by_week(") < calc.index("receivables_features(")
inspect.signature(prep.receivables_features).bind("panel", "receivables", "levels", 5, (30, 90))
try:
    inspect.signature(prep.receivables_features).bind("panel", "receivables", "levels", 5)
except TypeError:
    pass
else:
    raise AssertionError("receivables_features accepts a call without the age edges")
inspect.signature(reports.show_receivables_by_week).bind("r", (0,), 30, (30, 90))
assert callable(reports.show_owed)

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
reports.show_receivables_by_week(receivables, (0, 16), 8, (30, 90))

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
owed = prep.receivables_features(panel, receivables, levels, 5, (30, 90))
filled = int(reports.before_first_ledger_week(panel, receivables).sum())
assert filled == 2, filled
reports.show_owed(owed, filled)
# A key with no receivables row at all is refused, not filled.
absent = pd.concat([panel, pd.DataFrame({"key": ["D"], "origin_week": [3], "h": [1]})],
                   ignore_index=True)
try:
    prep.receivables_features(absent, receivables, levels, 5, (30, 90))
except ValueError as error:
    assert str(error) == "1 panel keys have no receivables row at all (first: ['D'])", error
else:
    raise AssertionError("a key with no receivables row was filled")
print("ok")
"""


def test_receivables_aging_and_features_are_wired_in_and_printed():
    result = run_snippet(RECEIVABLES)
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
from collection_estimation.hurdle.reports import preparation as reports
from collection_estimation.hurdle import inputs as prep

week_zero = date(2025, 6, 2)
weeks = reports.price_weeks(week_zero, -32, 64)
actual = reports.inflation.weekly_price_level(weeks)
published = reports.inflation.published_price_level(weeks)
projected = reports.inflation.projected_price_level(weeks)
if {old_basis}:
    levels = pd.Series(published, index=weeks["week"].to_numpy(), name="price_level")
else:
    levels = prep.published_levels(week_zero, -32, 64)
reports.show_prices(weeks, actual, published, projected, levels)

matrix = pd.DataFrame(0.0, index=pd.Index(["A", "B"], name="key", dtype=object),
                      columns=pd.Index(list(range(-2, 4)), name="week"))
matrix.loc["A", 1] = 20.0
matrix.loc["A", 3] = 5.0
reports.show_matrix(matrix)
print("ok")
"""


def run_prices(old_basis):
    return run_snippet(PRICES.format(old_basis=old_basis))


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


def test_step_10_price_weeks_of_the_printout_take_the_floored_week():
    # The copied inflation module is called only for the printout (show_preparation), on
    # the same weeks as the levels: the floored week prepare_inputs kept in the details.
    calls = calls_to(show_preparation_tree(), "price_weeks")
    assert len(calls) == 1
    args = [ast.unparse(arg) for arg in calls[0].args]
    assert args == ["inputs.week_zero", "details.first_price_week", "inputs.last_week"], args
    assert ("weeks_for_prices = price_weeks(inputs.week_zero, details.first_price_week, "
            "inputs.last_week)") in show_preparation_source()
    assert calls_to(prepare_tree(), "price_weeks") == []


def test_step_10_order_ledger_week_then_floor_then_prices():
    source = prepare_source()
    assert (source.index("first_ledger_week = ")
            < source.index("first_price_week = min(first_ledger_week, 0)")
            < source.index("published_levels(week_zero, first_price_week"))
    # The floor comes after last_week is known and after the ledger is loaded.
    assert source.index("ledger = load_ledger(") < source.index("first_ledger_week = ")
    assert source.index("last_week = int(weeks[\"week\"].max())") < source.index(
        "first_price_week = ")
    shows = show_preparation_source()
    assert shows.index("price_weeks(inputs.week_zero") < shows.index("show_prices(")


# Step 11: the deflated label, added to the panel after the receivables features and printed.
# The wiring is checked on main()'s source; the printout is run on a hand-built panel in a
# separate process: target weeks 0, 16, 17, 32 labelled, one missing-week and one future row,
# and one h=2 row (target 17) that the h=1 table must leave out.
DEFLATED = """
import inspect
import pandas as pd
from collection_estimation.hurdle.reports import preparation as reports
from collection_estimation.hurdle import inputs as prep

calc = " ".join(inspect.getsource(prep.prepare_inputs).split())
shows = " ".join(inspect.getsource(reports.show_preparation).split())
assert "panel = add_deflated_label(panel, levels)" in calc, "add_deflated_label not called"
assert "show_deflated_label(details.panel, inputs.levels)" in shows, (
    "show_deflated_label not called")
assert calc.index("owed = receivables_features(") < calc.index(
    "panel = add_deflated_label(panel, levels)")
assert shows.index("show_owed(details.owed") < shows.index("show_deflated_label(")
assert calc.index("levels = published_levels(") < calc.index("add_deflated_label(")
assert calc.index("panel = add_labels(") < calc.index("add_deflated_label(")
inspect.signature(prep.add_deflated_label).bind("panel", "levels")
inspect.signature(reports.show_deflated_label).bind("panel", "levels")

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
panel = prep.add_deflated_label(panel, levels)
assert panel["amount_deflated"].tolist()[:6] == [1e6, 1e6, 4e6, 1.5e6, 1e6, 5e6]
reports.show_deflated_label(panel, levels)
print("ok")
"""


def run_deflated():
    return run_snippet(DEFLATED)


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
from collection_estimation.hurdle.reports import preparation as reports
from collection_estimation.hurdle import inputs as prep

calc = " ".join(inspect.getsource(prep.prepare_inputs).split())
shows = " ".join(inspect.getsource(reports.show_preparation).split())
assert "table = join_features(panel, history, calendar, owed)" in calc, "join_features"
assert "show_table(inputs.table, details.panel)" in shows, "show_table not called"
assert calc.index("panel = add_deflated_label(panel, levels)") < calc.index(
    "table = join_features(panel, history, calendar, owed)")
assert shows.index("show_deflated_label(") < shows.index("show_table(inputs.table, details.panel)")
# show_table is the last printout of steps 1-11.
assert shows.rstrip().endswith("show_table(inputs.table, details.panel)")
# Every block is built before the join, and the panel has its deflated label by then.
for built in ("history = history_features(", "calendar = calendar_features(",
              "calendar = calendar.merge(positions", "owed = receivables_features(",
              "panel = add_deflated_label("):
    assert calc.index(built) < calc.index("table = join_features("), built
inspect.signature(prep.join_features).bind("panel", "history", "calendar", "owed")
inspect.signature(reports.show_table).bind("table", "panel")

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
table = prep.join_features(panel, history, calendar, owed)
assert len(table) == 4 and table.shape[1] == 11, table.shape
reports.show_table(table, panel)
print("ok")
"""


def run_table():
    return run_snippet(TABLE)


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
from collection_estimation.hurdle.reports import preparation as reports

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
reports.show_table(table, table)
print("ok")
"""


def run_no_paid(variant):
    return run_snippet(NO_PAID.format(variant=variant))


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


# Step 11, U7b: the holiday positions of every target week, with the [calendar] settings, over
# the same weeks as the calendar features, merged into the calendar block (on week, one to
# one), and their printout. Run on the real settings/holidays_tr.toml (read only), in a
# separate process.
HOLIDAY_POSITIONS = """
import inspect
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from collection_estimation.hurdle.reports import preparation as reports
from collection_estimation.hurdle import inputs as prep
from collection_estimation.hurdle.features_calendar import CALENDAR_COLUMNS

assert prep.holiday_position_features.__module__ == (
    "collection_estimation.hurdle.features_calendar")
inspect.signature(reports.show_holiday_positions).bind("positions", "week_zero", "holidays")
calc = " ".join(inspect.getsource(prep.prepare_inputs).split())
shows = " ".join(inspect.getsource(reports.show_preparation).split())
call = ("positions = holiday_position_features( week_zero, 0, last_week + horizons, holidays, "
        "settings.calendar.holiday_encodings, settings.calendar.drain_window, "
        "settings.calendar.catch_up_window)")
assert call in calc, "holiday_position_features is not called as expected"
merge = 'calendar = calendar.merge(positions, on="week", how="left", validate="one_to_one")'
assert merge in calc, "the positions are not merged into the calendar as expected"
show = "show_holiday_positions(details.positions, inputs.week_zero, details.holidays)"
assert show in shows, "show_holiday_positions is not called as expected"
assert calc.count("holiday_position_features(") == 1
assert shows.count("show_holiday_positions(") == 1
order = [calc.index("calendar = calendar_features("),
         calc.index("holiday_position_features("), calc.index(merge),
         calc.index("receivables_features("),
         calc.index("table = join_features(panel, history, calendar, owed)")]
assert order == sorted(order), order
order = [shows.index("show_calendar(inputs.calendar, inputs.week_zero, details.holidays)"),
         shows.index(show), shows.index("show_owed(")]
assert order == sorted(order), order
# The details keep the positions alone (not merged), for the printout.
assert "positions=positions" in calc

holidays = prep.load_holidays(Path("settings") / "holidays_tr.toml")
week_zero = date(2025, 6, 2)
calendar = prep.calendar_features(week_zero, 0, 69, holidays)
for encodings in (("drain",), ("days",), ("drain", "days"), ()):
    positions = prep.holiday_position_features(week_zero, 0, 69, holidays, encodings, 3, 5)
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
    reports.show_holiday_positions(positions, week_zero, holidays)
print("ok")
"""


@pytest.fixture(scope="module")
def holiday_positions_run():
    result = run_snippet(HOLIDAY_POSITIONS)
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


# U18.1: the printout of every cleared invoice's days late (`show_cleared_invoice_delays`),
# run in a separate process on a small hand-built table, on what `cleared_invoice_delays`
# makes of a tiny ledger, on a table of more than a thousand rows (thousands separators) and
# on an empty table. Each case's output follows a "== name" marker line.
CLEARED_DELAYS = """
import inspect
import pandas as pd
from collection_estimation.hurdle.reports import preparation as reports
from collection_estimation.hurdle.receivables_asof import (CLEARED_DELAY_COLUMNS,
                                                           cleared_invoice_delays)

inspect.signature(reports.show_cleared_invoice_delays).bind("delays")

def table(keys, days_late, lags):
    money = pd.Timestamp("2026-04-01")
    rows = []
    for n, (key, late, lag) in enumerate(zip(keys, days_late, lags)):
        rows.append({"key": key, "company_code": key[:4], "document_no": f"INV{n}",
                     "net_due_date": money - pd.Timedelta(days=late), "money_arrived": money,
                     "days_late": late, "known_on": money + pd.Timedelta(days=lag)})
    return pd.DataFrame(rows, columns=CLEARED_DELAY_COLUMNS)

print("== small")
reports.show_cleared_invoice_delays(table(
    ["TR02-A", "TR02-A", "TR02-B", "TR04-C", "TR04-C"], [-4, -2, 0, 6, 10], [0, 0, 3, 7, 30]))

print("== real")
ts = pd.Timestamp
ledger = pd.DataFrame({
    "key": ["TR02-100"] * 4, "company_code": ["TR02"] * 4,
    "document_no": ["INV1", "PAY1", "INV2", "PAY2"],
    "document_role": ["invoice", "payment", "invoice", "payment"],
    "invoice_reference": [None, None, None, None],
    "posting_date": [ts("2026-03-01"), ts("2026-03-20"), ts("2026-03-01"), ts("2026-04-12")],
    "net_due_date": [ts("2026-03-31"), ts("2026-03-20"), ts("2026-04-10"), ts("2026-04-12")],
    "clearing_date": [ts("2026-04-15"), ts("2026-04-15"), ts("2026-04-12"), ts("2026-04-12")],
    "clearing_document": ["CLR-1", "CLR-1", "CLR-2", "CLR-2"],
    "source_row": [2, 3, 4, 5]})
for column in ["invoice_reference", "clearing_document"]:
    ledger[column] = ledger[column].astype(object)
reports.show_cleared_invoice_delays(cleared_invoice_delays(ledger))

print("== many")
reports.show_cleared_invoice_delays(table(
    [f"TR02-{n % 1100}" for n in range(1200)], [5] * 1200, [0] * 1200))

print("== empty")
reports.show_cleared_invoice_delays(table(["TR02-A"], [1], [1]).iloc[0:0])

print("== empty real")
reports.show_cleared_invoice_delays(cleared_invoice_delays(ledger[ledger["document_role"] == "payment"]))
print("ok")
"""


@pytest.fixture(scope="module")
def cleared_delays_run():
    result = run_snippet(CLEARED_DELAYS)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    parts = {}
    for chunk in result.stdout.split("== ")[1:]:
        name, _, body = chunk.partition("\n")
        parts[name.strip()] = [line for line in body.splitlines() if line and line != "ok"]
    return parts


def test_cleared_delays_printout_is_a_function_of_the_module():
    assert function_tree(PREPARATION, "show_cleared_invoice_delays").args.args[0].arg == "delays"


def test_cleared_delays_printout_is_two_lines(cleared_delays_run):
    assert len(cleared_delays_run["small"]) == 2
    assert len(cleared_delays_run["real"]) == 2


def test_cleared_delays_printout_counts_quartiles_and_share_early(cleared_delays_run):
    # Days late -4, -2, 0, 6, 10: quartiles -2 / 0 / 6; 2 of 5 negative = 40%.
    assert cleared_delays_run["small"][0] == (
        "        cleared invoices with a payment: 5 from 3 keys; days late: quartiles "
        "-2 / 0 / 6, 40% paid early")


def test_cleared_delays_printout_clearing_lag(cleared_delays_run):
    # Lags 0, 0, 3, 7, 30: median 3; 90th percentile 7 + 0.6 * 23 = 20.8 -> 21; max 30.
    assert cleared_delays_run["small"][1] == (
        "        known after the money arrived (clearing lag), days: median 3, "
        "90th percentile 21, max 30")


def test_cleared_delays_printout_on_the_functions_own_output(cleared_delays_run):
    # INV1: paid 20 Mar, due 31 Mar (-11), cleared 15 Apr (lag 26); INV2: paid 12 Apr, due
    # 10 Apr (2), cleared 12 Apr (lag 0). Quartiles of (-11, 2): -7.75 / -4.5 / -1.25.
    first, second = cleared_delays_run["real"]
    assert first.startswith("        cleared invoices with a payment: 2 from 1 keys; ")
    assert "quartiles -8 / -4 / -1, 50% paid early" in first
    assert "median 13, 90th percentile 23, max 26" in second


def test_cleared_delays_printout_uses_thousands_separators(cleared_delays_run):
    assert cleared_delays_run["many"][0].startswith(
        "        cleared invoices with a payment: 1,200 from 1,100 keys; ")
    assert "quartiles 5 / 5 / 5, 0% paid early" in cleared_delays_run["many"][0]


def test_cleared_delays_printout_on_an_empty_table_is_one_line(cleared_delays_run):
    # No quartiles of nothing ("nan"): one line, and nothing else.
    assert cleared_delays_run["empty"] == ["        no cleared invoices with a payment"]


def test_cleared_delays_printout_on_the_functions_empty_output_is_one_line(cleared_delays_run):
    # A ledger of payments only: cleared_invoice_delays gives no rows.
    assert cleared_delays_run["empty real"] == ["        no cleared invoices with a payment"]


def test_cleared_delays_printout_is_not_called_by_show_preparation():
    # U18.1 is main()'s own step, after the inputs are unpacked.
    assert calls_to(show_preparation_tree(), "show_cleared_invoice_delays") == []
    assert calls_to(show_preparation_tree(), "cleared_invoice_delays") == []
