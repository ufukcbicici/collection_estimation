"""Tests of `run_hurdle_pipeline.py` (the step-by-step script): main() and the script's own
functions (step 15's joint diagnostic and the top-up experiments).

Running the real `main()` reads the whole synthetic corpus and writes to data/interim, so it
is not run here: main() is checked on its syntax tree, and parts of it (the guards, step 13's
and step 17's blocks, step 18's `if`) are run on their own with fakes. Code that imports the
script runs in a separate Python process: the script reconfigures sys.stdout and puts src/ on
sys.path, which should not leak into the rest of the test session.

The printouts the script imports are tested in tests/hurdle/reports/, step 18 (`run_grid`) in
tests/hurdle/test_grid_run.py, and the source of prepare_inputs in test_inputs_source.py.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.hurdle.code_checks import module_functions
from tests.hurdle.test_inputs_source import STEPS_1_TO_11

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "run_hurdle_pipeline.py"

CHECK = """
import inspect
import run_hurdle_pipeline as script
from collection_estimation.hurdle import inputs as prep
assert callable(script.main)
assert script.DEFAULT_SETTINGS_FILE.name == "synthetic.toml"
# Steps 1-11 moved to inputs.prepare_inputs; the script calls that one.
assert script.prepare_inputs is prep.prepare_inputs
# Every printout comes from hurdle/reports/ (the script defines only show_topups).
from collection_estimation.hurdle.reports import preparation
assert script.show_preparation is preparation.show_preparation
# The names the script calls must have the signatures it uses after L.2.
inspect.signature(prep.list_daily_files).bind("folder", "%d%m%Y")
inspect.signature(prep.load_collections).bind({}, "layout", "cache")
# After L.3: the ledger layout comes from the settings file.
inspect.signature(prep.load_ledger).bind("path", "as_of", "layout", "cache")
try:
    inspect.signature(prep.load_ledger).bind("path", "as_of", "cache")
except TypeError:
    pass
else:
    raise AssertionError("load_ledger still accepts the old three-argument call")
inspect.signature(prep.check_ledger_against_collections).bind("ledger", "collections", [])
# After L.4: the holidays come from the holidays file named in the settings.
inspect.signature(prep.load_holidays).bind("path")
inspect.signature(prep.week_calendar).bind([], "week_zero", "holidays")
try:
    inspect.signature(prep.week_calendar).bind([], "week_zero")
except TypeError:
    pass
else:
    raise AssertionError("week_calendar still accepts the old two-argument call")
# After U9b.5: step 7 builds every week's receivables in one call.
inspect.signature(prep.exchange_rates).bind("collections", "ledger")
inspect.signature(prep.receivables_by_week).bind(
    "ledger", "rates", "week_zero", last_week=64, as_of="as_of", horizons=5, min_cleared=3,
    leak_check_weeks=(0, 30, 64), age_edges=(30, 90))
# After U9c.1: the aging-bucket edges are required.
try:
    inspect.signature(prep.receivables_by_week).bind(
        "ledger", "rates", "week_zero", last_week=64, as_of="as_of", horizons=5,
        min_cleared=3, leak_check_weeks=(0, 30, 64))
except TypeError:
    pass
else:
    raise AssertionError("receivables_by_week accepts a call without the age edges")
# Step 8: when each key becomes known, from the daily files and the ledger.
inspect.signature(prep.known_weeks).bind("collections", "ledger", "week_zero")
# Step 9: the panel's rows, from the known keys, the last week and the horizons.
inspect.signature(prep.panel_rows).bind("known", 64, 5)
# Step 9, U4.2: the labels, from the panel, the weekly amounts and the week calendar.
inspect.signature(prep.add_labels).bind("panel", "weekly", "weeks")
# Step 9, side table A: new-customer cash, from the weekly amounts, the known keys, the
# calendar, the horizons and the companies (required: the grid is built from them).
inspect.signature(prep.new_customer_cash).bind("weekly", "known", "weeks", 5, ["TR02"])
try:
    inspect.signature(prep.new_customer_cash).bind("weekly", "known", "weeks", 5)
except TypeError:
    pass
else:
    raise AssertionError("new_customer_cash still accepts the old four-argument call")
# Side table R: refunds per company and week, from the weekly amounts, the calendar and
# the companies.
inspect.signature(prep.refunds_by_week).bind("weekly", "weeks", ["TR02"])
# Side table G: placeholder customers are split off by customer_id, and their gross
# receipts summed per company and week.
inspect.signature(prep.is_unidentified).bind("table", ("99999",))
inspect.signature(prep.unidentified_cash_by_week).bind("unidentified", "weeks", ["TR02"])
# The tie-out: the panel and the side tables A, G and R against all the weekly amounts.
inspect.signature(prep.check_side_tables).bind("panel", "new_cash", "unidentified",
                                                 "refunds", "weekly", ["TR02"])
# Step 6: days without a file filled from the ledger; the calendar takes the filled days.
inspect.signature(prep.fill_missing_days).bind("collections", "ledger", [])
try:
    inspect.signature(prep.fill_missing_days).bind("collections", "ledger")
except TypeError:
    pass
else:
    raise AssertionError("fill_missing_days accepts a call without the days with a file")
inspect.signature(prep.week_calendar).bind([], "week_zero", "holidays", [])
inspect.signature(prep.week_calendar).bind([], "week_zero", "holidays", filled_days=[])
# Step 10: TÜFE price levels by week, from week zero and the first and last week numbers.
inspect.signature(prep.published_levels).bind("week_zero", -32, 64)
# Step 11: the ledger's payments before the first daily file, as payment history.
inspect.signature(prep.first_history_week).bind("ledger", "week_zero")
inspect.signature(prep.ledger_receipts_before).bind("ledger", "first_file_day", "week_zero")
# Step 11, U6.1b: the weekly matrix, from the weekly amounts, the early receipts, the keys,
# the calendar, the price levels and the first and last weeks.
inspect.signature(prep.weekly_matrix).bind("weekly", "early", "keys", "weeks", "levels",
                                             -31, 64)
try:
    inspect.signature(prep.weekly_matrix).bind("weekly", "early", "keys", "weeks", "levels")
except TypeError:
    pass
else:
    raise AssertionError("weekly_matrix accepts a call without the first and last weeks")
print("ok")
"""


# Steps 1-11 as one call (`inputs.prepare_inputs`) and their printouts (`show_preparation`):
# main()'s side of the move, checked on its syntax tree (helpers further down: main_tree,
# prepare_tree). show_preparation is tested in reports/test_preparation.py, what
# prepare_inputs computes on real tiny data in test_inputs.py.

# main()'s names after the move, each unpacked from the inputs field of the same name.
UNPACKED = ("companies", "week_zero", "weeks", "weekly", "ledger", "rates", "levels",
            "calendar", "table", "block_columns", "new_cash", "refunds", "unidentified")

# The printouts of steps 1-11, show_preparation's (the top-ups are main()'s own, below).
PREPARATION_PRINTOUTS = [
    "show_daily_files", "show_one_day", "show_collections", "show_keys", "show_ledger",
    "show_tie", "show_filled", "print", "show_weeks", "show_weekly", "show_rates",
    "show_receivables_by_week", "show_known", "show_panel_rows", "show_labels",
    "show_new_customer_cash", "show_refunds", "print", "print", "show_prices",
    "show_projected_levels", "show_early_receipts", "show_matrix", "show_history",
    "show_calendar", "show_holiday_positions", "show_owed", "show_deflated_label",
    "show_table"]


def printouts_in_order(tree):
    """The names of the show_* / compare_topups / print calls in `tree`, in source order."""
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and (node.func.id.startswith("show_") or node.func.id in ("compare_topups",
                                                                         "print"))]
    return [call.func.id for call in sorted(calls, key=lambda c: (c.lineno, c.col_offset))]


def test_main_calls_prepare_inputs_once_with_details_then_show_preparation():
    body = main_tree().body
    calls = calls_to(main_tree(), "prepare_inputs")
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == ["settings"]
    assert keywords_of(calls[0]) == {"with_details": "True"}
    at = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
          and ast.unparse(node) == "inputs, details = prepare_inputs(settings, "
                                   "with_details=True)"]
    assert len(at) == 1
    assert ast.unparse(body[at[0] + 1]) == "show_preparation(inputs, details, settings)"
    # Only the settings come before it (and their two lines of printout).
    assert ast.unparse(body[0]) == "settings = load_settings(SETTINGS_FILE, REPO_ROOT)"
    assert all(isinstance(node, ast.Expr) for node in body[1:at[0]])


def test_main_calls_none_of_the_step_1_to_11_functions_itself():
    tree = main_tree()
    for name in STEPS_1_TO_11:
        assert calls_to(tree, name) == [], name


def test_the_script_no_longer_imports_the_step_1_to_11_functions():
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {alias.asname or alias.name for node in ast.walk(module)
                if isinstance(node, ast.ImportFrom) for alias in node.names}
    for name in STEPS_1_TO_11:
        assert name not in imported, name
    assert "prepare_inputs" in imported


def test_main_unpacks_each_input_from_the_field_of_its_name():
    tree = main_tree()
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        targets = target.elts if isinstance(target, ast.Tuple) else [target]
        values = value.elts if isinstance(value, ast.Tuple) else [value]
        if len(targets) != len(values):
            continue
        for t, v in zip(targets, values):
            if isinstance(t, ast.Name) and ast.unparse(v).startswith("inputs."):
                found.setdefault(t.id, []).append(ast.unparse(v))
    assert set(found) == set(UNPACKED), sorted(found)
    for name in UNPACKED:
        assert found[name] == [f"inputs.{name}"], (name, found[name])


def test_main_unpacking_comes_before_any_other_use_of_the_names():
    # history, owed, panel and the other intermediates of steps 1-11 are not main() names.
    tree = main_tree()
    stored = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Store)}
    for name in ("history", "owed", "panel", "known", "matrix", "positions", "receivables",
                 "collections", "daily_files", "identified_weekly", "first_price_week",
                 "history_start", "early_receipts", "holidays"):
        assert name not in stored, name


def test_main_never_rebinds_inputs_after_prepare_inputs():
    # The ModelInputs object stays bound to `inputs` for the whole of main(): no other
    # assignment, loop or with target, walrus or del of the name anywhere
    # (step 13's split dict is `split_inputs`, not `inputs`).
    tree = main_tree()
    stores = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
              and node.id == "inputs" and isinstance(node.ctx, (ast.Store, ast.Del))]
    assert len(stores) == 1, [getattr(n, "lineno", None) for n in stores]
    owner = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(n is stores[0] for t in node.targets for n in ast.walk(t))]
    assert len(owner) == 1
    assert ast.unparse(owner[0]) == ("inputs, details = prepare_inputs(settings, "
                                     "with_details=True)")
    # No nested function or class of main() takes `inputs` as a parameter either.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)) and node is not tree:
            assert "inputs" not in [a.arg for a in node.args.args + node.args.kwonlyargs]
    assert "split_inputs = {part:" in main_source()


def test_show_preparation_is_the_only_place_of_the_step_1_to_11_printouts():
    main_printouts = set(printouts_in_order(main_tree()))
    for name in set(PREPARATION_PRINTOUTS) - {"print"}:
        assert name not in main_printouts, name


def test_the_pipeline_script_exists():
    assert SCRIPT.is_file()


def test_the_pipeline_script_imports_without_running():
    result = subprocess.run([sys.executable, "-c", CHECK], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


# Steps 1-11 now live in hurdle/inputs.py (`prepare_inputs`), their printouts in
# hurdle/reports/preparation.py (`show_preparation`); main() is checked on its syntax tree,
# read from the file without importing or running it. The behaviour itself is tested in
# test_ledger_starting_late.py and test_inputs.py.
import ast  # noqa: E402

INPUTS_MODULE = REPO_ROOT / "src" / "collection_estimation" / "hurdle" / "inputs.py"


def function_tree(path, name):
    """The top-level `def name` of the file at `path`, as a syntax tree."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    found = [node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, name
    return found[0]


def function_source(path, name):
    """The source of `def name` in `path`, with every run of whitespace made one space."""
    return " ".join(ast.get_source_segment(path.read_text(encoding="utf-8"),
                                           function_tree(path, name)).split())


def main_tree():
    return function_tree(SCRIPT, "main")


def main_source():
    return function_source(SCRIPT, "main")


def prepare_tree():
    return function_tree(INPUTS_MODULE, "prepare_inputs")


def prepare_source():
    return function_source(INPUTS_MODULE, "prepare_inputs")


def assignments_to(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]


def calls_to(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == name]


# Step 12 (U10.1): stage 1's rows and model input at [debug] demo_origin. The wiring is
# checked on main()'s source; the guard on demo_origin (right after steps 1-11, i.e. after
# prepare_inputs, the unpacking of its inputs and the top-ups) is run from main()'s own syntax
# tree. The printout is tested in reports/test_stages.py.
STAGE1 = """
import inspect
import run_hurdle_pipeline as script
from collection_estimation.hurdle import inputs as prep

source = " ".join(inspect.getsource(script.main).split())
steps = [
    "inputs, details = prepare_inputs(settings, with_details=True)",
    "show_preparation(inputs, details, settings)",
    "levels, calendar, table = inputs.levels, inputs.calendar, inputs.table",
    "block_columns = inputs.block_columns",
    "origin = settings.debug.demo_origin",
    "columns = feature_columns(block_columns, settings.stage1.feature_blocks)",
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
assert source.index("companies, week_zero = inputs.companies, inputs.week_zero") < source.index(
    "train_x = stage1_features(")
# The block columns are made once, in prepare_inputs, not again in main().
assert "block_feature_columns(" not in source
inspect.signature(script.show_stage1_rows).bind("origin", "train", "forecast", "train_x")
inspect.signature(script.feature_columns).bind({}, ())
inspect.signature(prep.block_feature_columns).bind("history", "calendar", "owed")
inspect.signature(script.rows_known_at).bind("table", 40)
inspect.signature(script.rows_at_origin).bind("table", 40)
inspect.signature(script.stage1_features).bind("rows", [], [], True)

print("ok")
"""


def test_step_12_is_wired_in_after_the_training_table():
    result = run_code(STAGE1)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


GUARD_TEST = "settings.debug.demo_origin + settings.receivables.horizons > inputs.last_week"


def is_demo_origin_guard(node):
    return isinstance(node, ast.If) and ast.unparse(node.test) == GUARD_TEST


def demo_origin_guard():
    """The `if settings.debug.demo_origin + ... > inputs.last_week:` statement of main(),
    right after steps 1-11 (prepare_inputs) and the unpacking of their inputs."""
    found = [node for node in ast.walk(main_tree()) if is_demo_origin_guard(node)]
    assert len(found) == 1
    return found[0]


def last_week_statement():
    """prepare_inputs' own `last_week = ...` statement (step 7), the last week the guards
    compare with (as ModelInputs.last_week)."""
    found = assignments_to(prepare_tree(), "last_week")
    assert len(found) == 1
    return found[0]


def last_week_of(weeks):
    """Run prepare_inputs' `last_week = ...` statement on a week calendar."""
    module = ast.Module(body=[last_week_statement()], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(INPUTS_MODULE), "exec")
    names = {"weeks": weeks}
    exec(code, names)
    return names["last_week"]


def week_table(first_week, last_week, week_dtype="int64"):
    import pandas as pd
    return pd.DataFrame({"week": pd.Series(range(first_week, last_week + 1), dtype=week_dtype)})


def run_guard(origin, horizons, last_week, first_week=0, week_dtype="int64", weeks=None):
    """Run main()'s guard statement on its own, with the two names it reads: settings and the
    inputs, whose last_week is what prepare_inputs' own statement makes of a week calendar
    with weeks first_week..last_week (or of `weeks`)."""
    from types import SimpleNamespace
    if weeks is None:
        weeks = week_table(first_week, last_week, week_dtype)
    module = ast.Module(body=[demo_origin_guard()], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(SCRIPT), "exec")
    settings = SimpleNamespace(debug=SimpleNamespace(demo_origin=origin),
                               receivables=SimpleNamespace(horizons=horizons))
    inputs = SimpleNamespace(last_week=last_week_of(weeks))
    exec(code, {"settings": settings, "inputs": inputs})


def test_step_12_origin_is_read_from_debug_demo_origin():
    found = assignments_to(main_tree(), "origin")
    assert len(found) == 1
    assert ast.unparse(found[0].value) == "settings.debug.demo_origin"


def test_demo_origin_guard_sits_right_after_the_inputs_are_unpacked():
    # Steps 1-11 are one call now: the guard follows it, its printouts, the unpacking of the
    # inputs into main()'s names, the U18.1 printout of the cleared invoices' days late and
    # the top-ups (U12.2, U12.3), and nothing else. (U18.1 added its two statements, U18.2
    # its three: the heading, expected_dues and show_expected_dues.)
    body = main_tree().body
    calls = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
             and ast.unparse(node) == "inputs, details = prepare_inputs(settings, "
                                      "with_details=True)"]
    assert len(calls) == 1
    guards = [i for i, node in enumerate(body) if is_demo_origin_guard(node)]
    assert len(guards) == 1
    between = body[calls[0] + 1:guards[0]]
    assert ast.unparse(between[0]) == "show_preparation(inputs, details, settings)"
    assert [ast.unparse(node) for node in between[-7:]] == [
        U18_1_PRINT, U18_1_SHOW, U18_2_PRINT, U18_2_DUES, U18_2_SHOW, SHOW_TOPUPS,
        COMPARE_TOPUPS_IF]
    for node in between[1:-7]:
        assert isinstance(node, ast.Assign), ast.unparse(node)
        values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        for value in values:
            assert ast.unparse(value).startswith("inputs."), ast.unparse(node)


def test_last_week_is_the_largest_week_as_a_plain_int():
    # The guards read ModelInputs.last_week, made once in prepare_inputs (step 7).
    assert ast.unparse(last_week_statement()) == "last_week = int(weeks['week'].max())"
    assert "last_week=last_week" in prepare_source()


def test_demo_origin_guard_is_a_top_level_statement_of_main():
    # Not nested in a branch: it runs on every call of main().
    assert sum(is_demo_origin_guard(node) for node in main_tree().body) == 1


def test_demo_origin_guard_comes_right_after_steps_1_to_11():
    # It used to run in step 7, before the receivables; steps 1-11 are one call now
    # (prepare_inputs), so it runs after that call and its printouts, before step 12.
    source = main_source()
    guard = source.index("if settings.debug.demo_origin + settings.receivables.horizons")
    assert source.index("inputs, details = prepare_inputs(") < guard
    assert source.index("show_preparation(inputs, details, settings)") < guard


def test_demo_origin_guard_comes_before_any_stage_1_work():
    source = main_source()
    guard = source.index("if settings.debug.demo_origin + settings.receivables.horizons")
    for later in ("origin = settings.debug.demo_origin",
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
    import pandas as pd
    weeks = pd.DataFrame({"week": [64, 0, 10]})
    assert last_week_of(weeks) == 64
    run_guard(origin=59, horizons=5, last_week=None, weeks=weeks)
    with pytest.raises(ValueError, match="after the last week 64"):
        run_guard(origin=60, horizons=5, last_week=None, weeks=weeks)


@pytest.mark.parametrize("origin, horizons, last_week", [(60, 5, 64), (64, 5, 64),
                                                          (100, 5, 64), (0, 2, 1)])
def test_demo_origin_guard_raises_once_a_target_is_after_the_last_week(origin, horizons,
                                                                       last_week):
    with pytest.raises(ValueError) as error:
        run_guard(origin, horizons, last_week)
    assert str(error.value) == (f"[debug] demo_origin {origin}: its target weeks reach week "
                                f"{origin + horizons}, after the last week {last_week}, so "
                                f"step 12 would have no answers")


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
import run_hurdle_pipeline as script

assert script.time is time
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
from collection_estimation.hurdle import inputs as prep

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


# The second guard (U10.4a): when [diagnostic] run is on, its test weeks (up to origin +
# 2 horizons) must have answers. Run from main()'s own syntax tree, like the demo_origin guard.
# U11.5: step 15's joint diagnostic uses the same splits, so its run switch arms the guard too.
# Since steps 1-11 moved to prepare_inputs, both guards read inputs.last_week.
DIAGNOSTIC_END = "settings.diagnostic.origin + 2 * settings.receivables.horizons"
DIAGNOSTIC_GUARD_TEST = ("(settings.diagnostic.run or settings.joint_diagnostic.run) and "
                         "diagnostic_end > inputs.last_week")


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
    settings and the inputs, whose last_week is what prepare_inputs' own statement makes of a
    week calendar whose weeks run 0..last_week (or of `weeks`)."""
    from types import SimpleNamespace
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
        weeks = week_table(0, last_week, week_dtype)
    inputs = SimpleNamespace(last_week=last_week_of(weeks))
    exec(code, {"settings": settings, "inputs": inputs})


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
    # It used to run in step 7, before the receivables; steps 1-11 are one call now
    # (prepare_inputs), so it runs after that call, before step 12 and the diagnostics.
    source = main_source()
    guard = source.index("and diagnostic_end > inputs.last_week")
    assert source.index("inputs, details = prepare_inputs(") < guard
    assert source.index("show_preparation(inputs, details, settings)") < guard
    for later in ("origin = settings.debug.demo_origin", "train_rows = rows_known_at(",
                  "diagnostic_splits(", "run_joint_diagnostic(", "forecast(table, origin"):
        assert guard < source.index(later), later


def run_code(code):
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                          text=True, encoding="utf-8", timeout=300)


# Step 13 (U10.4b/c): the stage 1 diagnostic. For HGB, then LR: grid_search on validation,
# choose_and_test on the test rows, the chosen setting refitted on train for the permutation
# importance on validation, then show_diagnostic_grid; after both, show_diagnostic_test and
# write_tables. (The U10.4a demo on step 12's predictions, with its "too few outcomes" skip,
# is gone; its tests went with it.) The printouts are tested in reports/test_stages.py.
#
# Wiring is checked on main()'s syntax tree. The block itself is run in a separate process
# (exec of main()'s own `if settings.diagnostic.run:` statement) on a tiny hand-made table:
# T = 4, H = 2, 30 rows per target week 0..9, every tenth row (i % 10 == 9) a missing_week
# row -> 27 labelled rows per week: train weeks 0..4 = 135 rows, validation 5..6 = 54, refit
# 189, test 7..8 = 54; week 9 is after the test weeks. Grids: HGB 2 x 1 x 1 x 2 = 4 settings
# of at most 10 trees, LR 2 x 2 = 4 settings. CSV output goes to pytest's tmp_path only.
STEP_13 = """
import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle import inputs as prep
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
from collection_estimation.hurdle import inputs as prep
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
assert text.count("split_inputs['test']") == 1, text
assert "permutation_importances(model, *split_inputs['validation']" in text
assert "settings.diagnostic.importance_repeats, settings.stage1.random_seed)" in text
assert "write_tables(settings.diagnostic.output_folder, tables)" in text
# The grid is ranked, and the printout labelled, by the settings' selection metric.
assert "*split_inputs['validation'], metric=settings.diagnostic.selection_metric)" in text, text
# The splits' model inputs have their own name: `inputs` stays main()'s ModelInputs.
assert "split_inputs = {part:" in text, text
assert "inputs[" not in text.replace("split_inputs[", ""), text
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


# --- Step 14 (U11.1): fit stage 2 on the paid training rows at the demo origin -----------------
# The wiring is checked on main()'s syntax tree: step 14 is the last part of main(), at its top
# level after step 13's `if settings.diagnostic.run:` block (so it runs whether or not the
# diagnostic does), with stage 2's own feature blocks and settings, and stage 1's company
# switch. The printouts and the real chain on a tiny table are tested in
# reports/test_stages.py.

STEP_14_WIRING = """
import ast
import inspect
import textwrap
import run_hurdle_pipeline as script
from collection_estimation.hurdle import inputs as prep
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
# Step 14 is found by content, not by its distance from the end of main(): it runs from just
# after step 13's `if settings.diagnostic.run:` up to (not including) step 15's
# `if settings.joint_diagnostic.run:` (steps 16 and 17 follow since U13 and U14).
joint = [i for i, node in enumerate(body)
         if isinstance(node, ast.If) and ast.unparse(node.test) == "settings.joint_diagnostic.run"]
assert len(joint) == 1, joint
assert diagnostic[0] < joint[0], (diagnostic, joint)
# U11.3: the stage 2 scores (checked below): two statements. U11.4 added the cash-by-p check
# after them (two statements), just before step 15's `if`: four statements in all.
scores_part = body[joint[0] - 4:joint[0] - 2]
step_14 = statements[diagnostic[0] + 1:joint[0] - 4]
assert step_14 == [
    "stage2_columns = feature_columns(block_columns, settings.stage2.feature_blocks)",
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
assert (ast.unparse(body[joint[0] - 2])
        == "labelled = forecast_rows['label_status'] == 'labelled'"), ast.unparse(body[joint[0] - 2])
cash_check = body[joint[0] - 1]
assert isinstance(cash_check, ast.If) and ast.unparse(cash_check.test) == "labelled.any()"
assert cash_check.orelse == []
assert [ast.unparse(n) for n in cash_check.body] == [
    "show_cash_by_p_group(cash_by_p_group(predictions['stage 1 (HGB)'][labelled], "
    "m[labelled], forecast_rows.loc[labelled, 'amount_deflated'], "
    "settings.diagnostic.calibration_groups))"], [ast.unparse(n) for n in cash_check.body]
# U11.5: step 15 follows step 14's checks, behind its own run switch.
step_15 = body[joint[0]]
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


def test_step_14_is_wired_in_after_step_13_and_before_step_15():
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


# The paid forecast rows filter, run from main()'s own statement on a hand frame: labelled and
# paid only; a missing week and a future row (paid NA) are left out, never an error.
PAID_FILTER = """
import ast
import inspect
import textwrap
import pandas as pd
import run_hurdle_pipeline as script
from collection_estimation.hurdle import inputs as prep

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
from collection_estimation.hurdle import inputs as prep

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


# --- Step 15 (U11.5): the joint diagnostic ------------------------------------------------------
# Wiring on the script's syntax tree and module; then run_joint_diagnostic itself, end to end,
# on a tiny hand-made table (2 companies x 8 customers, origins 0..33, h 1..3; T = 20, H = 3)
# with tiny grids, writing its CSVs to tmp_path only.

STEP_15_WIRING = """
import ast
import inspect
import run_hurdle_pipeline as script
from collection_estimation.hurdle import inputs as prep
from collection_estimation.hurdle import benchmark, diagnostic, diagnostic_joint

for name in ("benchmark_forecasts", "cash_totals"):
    assert getattr(script, name) is getattr(benchmark, name), name
for name in ("paired_bootstrap", "score_pairs", "scores_by_h", "stage1_grid", "stage2_grid",
             "stage1_gates", "stage2_gates"):
    assert getattr(script, name) is getattr(diagnostic_joint, name), name
inspect.signature(script.with_params).bind("gates", "grid", "stage1")
# U16.4 moved with_params into hurdle/grid.py; step 15 imports it from there.
from collection_estimation.hurdle import grid as grid_module
assert script.with_params is grid_module.with_params
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
from collection_estimation.hurdle import inputs as prep
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


# U12.2: the top-ups at the diagnostic origin, by the trailing mean of [topups] window weeks
# (passed in as `window`, since U12.3; it was the constant TOPUP_MEAN_WINDOW) and by the ridge
# with and without the recent level, printed after the deflated label. The wiring is checked on
# main()'s source; the printout is run on hand-built side tables in a separate process. Two
# companies, origin 24, horizons 2, window 8, price levels that move every week (so deflating
# by the wrong week shows). TR04's G starts at week 17, too few weeks to fit, so its ridge
# falls back at every h and the fallbacks column counts it; TR02's G is all zero, forecast 0
# and not counted. The trailing means and the actuals are computed in the script by plain
# loops over dicts, not by the module; the ridge columns come from `topup_forecasts` itself
# (tested in test_topups.py), summed over the companies. Two more runs follow: window 5 (the
# argument, not a constant, names and sets the mean) and one with TR04's G of week 26 missing
# (`topup_actuals` drops TR04's row of h 2, and the forecast columns of h 2 are added up over
# the same rows: TR02 alone).
TOPUP_DATA = """
import pandas as pd

companies = ["TR02", "TR04"]
origin, horizons = 24, 2
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


def mean_by_hand(rows, week_key, value_key, h, origin, window, over=None):
    total = 0.0
    for company in (companies if over is None else over):
        values = [r[value_key] / level_of[r[week_key]] for r in rows
                  if r["company_code"] == company and (h is None or r["h"] == h)
                  and origin - window < r[week_key] <= origin]
        total += sum(values) / len(values)
    return total


def actual(rows, week_key, value_key, h, target, company=None):
    return sum(r[value_key] / level_of[r[week_key]] for r in rows
               if r[week_key] == target and (h is None or r["h"] == h)
               and (company is None or r["company_code"] == company))
"""

TOPUPS = TOPUP_DATA + """
import inspect
import run_hurdle_pipeline as script
from collection_estimation.hurdle import inputs as prep
from collection_estimation.hurdle import prices, topups

calc = " ".join(inspect.getsource(prep.prepare_inputs).split())
shows = " ".join(inspect.getsource(script.show_preparation).split())
source = " ".join(inspect.getsource(script.main).split())
call = ("show_topups(new_cash, refunds, unidentified, levels, calendar, "
        "settings.diagnostic.origin, settings.receivables.horizons, settings.topups.window)")
assert call in source, "show_topups is not called as expected"
assert source.count("show_topups(") == 1
assert "show_topups(" not in shows and "show_topups(" not in calc
unpacked = ("new_cash, refunds, unidentified = inputs.new_cash, inputs.refunds, "
            "inputs.unidentified")
assert (source.index(unpacked) < source.index(call)
        < source.index("if settings.debug.demo_origin + settings.receivables.horizons"))
# The tables it is given are the inputs' (the calendar with the holiday positions merged).
for built in ("new_cash = new_customer_cash(", "refunds = refunds_by_week(",
              "unidentified = unidentified_cash_by_week(", "levels = published_levels(",
              "calendar = calendar.merge(positions"):
    assert calc.index(built) < calc.index("inputs = ModelInputs("), built
assert "show_trailing_means" not in source and "show_trailing_means" not in shows
assert not hasattr(script, "show_trailing_means")
assert not hasattr(script, "TOPUP_WINDOWS")
# U12.3: the window comes from [topups]; the constant is gone.
assert not hasattr(script, "TOPUP_MEAN_WINDOW")
assert "TOPUP_MEAN_WINDOW" not in inspect.getsource(script.show_topups)
assert script.topup_forecasts is topups.topup_forecasts
assert script.topup_actuals is topups.topup_actuals
inspect.signature(script.show_topups).bind(
    "new_cash", "refunds", "unidentified", "levels", "calendar", 24, 2, 8)
try:
    inspect.signature(script.show_topups).bind(
        "new_cash", "refunds", "unidentified", "levels", "calendar", 24, 2)
except TypeError:
    pass
else:
    raise AssertionError("show_topups still accepts a call without the window")
# The actual columns come from topup_actuals, not from a loop of the script's own.
assert "topup_actuals(" in inspect.getsource(script.show_topups)

copies = [t.copy(deep=True) for t in (new_cash, refunds, unidentified, calendar)]
levels_copy = levels.copy(deep=True)
window = 8

expected = {}
for h in (1, 2):
    target = origin + h
    a_mean = mean_by_hand(new_rows, "target_week", "new_customer_cash", h, origin, window)
    r_mean = mean_by_hand(refund_rows, "week", "refunds", None, origin, window)
    g_mean = mean_by_hand(g_rows, "week", "unidentified_cash", None, origin, window)
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
        assert mine.set_index("company_code")["fallback"].to_dict() == \\
            {"TR02": False, "TR04": True}, (name, h, mine)
        by_company = mine.set_index("company_code")["unidentified_cash"]
        assert by_company["TR02"] == 0.0
        # TR04's fallback: the mean of its deflated G over the 4 weeks up to the origin.
        level = sum(r["unidentified_cash"] / level_of[r["week"]] for r in g_rows
                    if r["company_code"] == "TR04" and origin - 4 < r["week"] <= origin) / 4
        assert abs(by_company["TR04"] - level) < 1e-6 * level, (by_company["TR04"], level)
    expected[h] = line

script.show_topups(new_cash, refunds, unidentified, levels, calendar, origin, horizons, window)
for table, copy in zip((new_cash, refunds, unidentified, calendar), copies):
    pd.testing.assert_frame_equal(table, copy)
pd.testing.assert_series_equal(levels, levels_copy)
print("EXPECTED " + repr(expected))

# Window 5: the mean's columns are named and computed with the window passed in.
print("=== WINDOW5 " + repr({h: mean_by_hand(new_rows, "target_week", "new_customer_cash", h,
                                             origin, 5) / 1e6 for h in (1, 2)}))
script.show_topups(new_cash, refunds, unidentified, levels, calendar, origin, horizons, 5)

# TR04's G of week 26 (the target week of h 2) missing: topup_actuals drops TR04's row of h 2.
missing = unidentified[~((unidentified["company_code"] == "TR04")
                         & (unidentified["week"] == 26))]
g_missing = [r for r in g_rows if not (r["company_code"] == "TR04" and r["week"] == 26)]
missing_expected = {}
for h in (1, 2):
    target = origin + h
    kept = companies if h == 1 else ["TR02"]
    a = sum(actual(new_rows, "target_week", "new_customer_cash", h, target, c) for c in kept)
    other = sum(actual(refund_rows, "week", "refunds", None, target, c)
                + actual(g_missing, "week", "unidentified_cash", None, target, c) for c in kept)
    line = {"A actual": a / 1e6, "top-up actual": (a + other) / 1e6}
    # The forecasts (made at origin 24, so week 26 is not read) over the kept companies only.
    a_mean = mean_by_hand(new_rows, "target_week", "new_customer_cash", h, origin, window, kept)
    r_mean = mean_by_hand(refund_rows, "week", "refunds", None, origin, window, kept)
    g_mean = mean_by_hand(g_missing, "week", "unidentified_cash", None, origin, window, kept)
    line["A mean 8"] = a_mean / 1e6
    line["top-up mean 8"] = (a_mean + r_mean + g_mean) / 1e6
    for name, with_level in (("ridge", True), ("ridge no level", False)):
        found = topups.topup_forecasts(new_cash, refunds, missing, levels, calendar,
                                       [origin], horizons, "ridge", window, with_level)
        mine = found[(found["h"] == h) & found["company_code"].isin(kept)]
        line["A " + name] = float(mine["new_customer_cash"].sum()) / 1e6
        line["top-up " + name] = float(mine["topup"].sum()) / 1e6
        line["fallbacks " + name] = int(mine["fallback"].sum())
    missing_expected[h] = line
print("=== MISSING " + repr(missing_expected))
script.show_topups(new_cash, refunds, missing, levels, calendar, origin, horizons, window)
print("ok")
"""

TOPUP_PRINTOUT_COLUMNS = ["A mean 8", "A ridge", "A ridge no level", "A actual",
                          "top-up mean 8", "top-up ridge", "top-up ridge no level",
                          "top-up actual", "fallbacks ridge", "fallbacks ridge no level"]
TOPUP_TITLE = ("top-ups at origin 24, all companies, million TRY of week 0 (actual = what came, "
               "hindsight; fallbacks = companies whose ridge fell back to the level):")


def topup_table(lines):
    """The two rows of a show_topups printout: lines after its title."""
    return {"header": lines[0], "index_name": lines[1],
            "rows": [row.split() for row in lines[2:4]]}


@pytest.fixture(scope="module")
def topups_run():
    result = subprocess.run([sys.executable, "-c", TOPUPS], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    lines = [line.strip() for line in result.stdout.splitlines()]
    expected = eval(next(line for line in lines if line.startswith("EXPECTED "))[9:])
    assert TOPUP_TITLE in lines, result.stdout
    table = lines[lines.index(TOPUP_TITLE) + 1:]
    end = next(i for i, line in enumerate(table) if line.startswith("EXPECTED "))
    extra = {}
    for marker in ("WINDOW5", "MISSING"):
        start = next(i for i, line in enumerate(lines) if line.startswith(f"=== {marker} "))
        assert lines[start + 1] == TOPUP_TITLE, lines[start + 1]
        extra[marker] = {"expected": eval(lines[start][len(marker) + 5:]),
                         **topup_table(lines[start + 2:])}
    return {**topup_table(table), "after": table[4:end], "expected": expected,
            "window5": extra["WINDOW5"], "missing": extra["MISSING"]}


def test_topups_are_wired_in_main_after_the_inputs_are_unpacked(topups_run):
    # The source checks (call text with [topups] window, order, imports, TOPUP_MEAN_WINDOW
    # gone, the old printout gone) ran in the fixture. The top-ups used to be printed by
    # show_preparation, after the deflated label; main() prints them now, after show_table.
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


def test_topups_printout_names_and_computes_the_mean_with_the_window_passed_in(topups_run):
    run5 = topups_run["window5"]
    columns = [name.replace("mean 8", "mean 5") for name in TOPUP_PRINTOUT_COLUMNS]
    assert run5["header"].split() == " ".join(columns).split()
    assert "mean 8" not in run5["header"]
    for row, h in zip(run5["rows"], (1, 2)):
        printed = dict(zip(columns, (float(value) for value in row[1:])))
        assert printed["A mean 5"] == pytest.approx(run5["expected"][h], abs=6e-4)
        # The window changes the mean only: the ridge and the actual columns are those of 8.
        for name in ("A ridge", "A ridge no level", "A actual", "top-up actual"):
            assert printed[name] == pytest.approx(topups_run["expected"][h][name], abs=6e-4)
    # And the two windows really give different means here.
    assert run5["expected"][1] != pytest.approx(topups_run["expected"][1]["A mean 8"],
                                                abs=1e-3)


def test_topups_printout_actual_of_an_h_with_a_missing_week_leaves_that_company_out(
        topups_run):
    # topup_actuals drops TR04's row of h 2 (its G of week 26 is missing), and show_topups
    # adds the forecasts up over the same rows: at h 2 every column, forecast and actual,
    # holds TR02 alone, and TR04's fallback is no longer counted. h 1 is untouched.
    missing = topups_run["missing"]
    assert missing["header"] == topups_run["header"]
    for row, h in zip(missing["rows"], (1, 2)):
        printed = dict(zip(TOPUP_PRINTOUT_COLUMNS, (float(value) for value in row[1:])))
        for name in TOPUP_PRINTOUT_COLUMNS:
            assert printed[name] == pytest.approx(missing["expected"][h][name], abs=6e-4), \
                (h, name, printed[name], missing["expected"][h][name])
    # h 1: the full run's numbers; h 2: smaller A in every column, and no fallback.
    for name in TOPUP_PRINTOUT_COLUMNS:
        assert missing["expected"][1][name] == pytest.approx(topups_run["expected"][1][name])
    for name in ("A mean 8", "A ridge", "A ridge no level", "A actual"):
        assert missing["expected"][2][name] < topups_run["expected"][2][name] - 1e-3, name
    assert missing["expected"][2]["fallbacks ridge"] == 0
    assert missing["expected"][2]["fallbacks ridge no level"] == 0
    printed_h2 = dict(zip(TOPUP_PRINTOUT_COLUMNS, missing["rows"][1][1:]))
    assert printed_h2["fallbacks ridge"] == printed_h2["fallbacks ridge no level"] == "0"


def test_topups_printout_forecast_and_actual_columns_cover_the_same_companies(topups_run):
    # With every week present the forecast columns add up both companies, as the actuals do.
    full = topups_run["expected"][2]
    missing = topups_run["missing"]["expected"][2]
    assert full["A mean 8"] > missing["A mean 8"] + 1e-3
    assert full["A actual"] > missing["A actual"] + 1e-3


# U12.3: the choice of the top-up method. `topup_candidates` gives the candidates by name;
# `compare_topups` forecasts every one at every origin of [topups] compare_first_origin ..
# compare_last_origin, scores the consolidated top-up against what came (score_topups,
# by_company=False) and prints one line per candidate, then the lowest error and the method
# set in [topups]. The wiring is checked on main()'s source. The printout is run on the same
# hand-built side tables as above (origins 20..24, horizons 2, compare windows 2 and 4); the
# means' errors are worked out in the script by plain loops (each company's own trailing mean,
# added up, against the cash that came), the ridges' from topup_forecasts consolidated by hand.
COMPARE_TOPUPS = TOPUP_DATA + """
import dataclasses
import inspect
import run_hurdle_pipeline as script
from collection_estimation.hurdle import inputs as prep
from collection_estimation.hurdle import topups
from collection_estimation.hurdle.settings import TopupSettings

calc = " ".join(inspect.getsource(prep.prepare_inputs).split())
shows = " ".join(inspect.getsource(script.show_preparation).split())
source = " ".join(inspect.getsource(script.main).split())
call = ("compare_topups(new_cash, refunds, unidentified, levels, calendar, settings.topups, "
        "settings.receivables.horizons)")
guarded = "if settings.topups.compare: " + call
assert guarded in source, "compare_topups is not called, guarded by [topups] compare"
assert source.count("compare_topups(") == 1
assert "compare_topups(" not in shows
assert "compare_topups(" not in calc
show = "show_topups(new_cash, refunds, unidentified, levels, calendar,"
assert (source.index(show) < source.index(guarded)
        < source.index("if settings.debug.demo_origin + settings.receivables.horizons"))
assert script.score_topups is topups.score_topups
assert script.topup_actuals is topups.topup_actuals
inspect.signature(script.compare_topups).bind(
    "new_cash", "refunds", "unidentified", "levels", "calendar", "topups", 2)
inspect.signature(script.topup_candidates).bind((4, 8))

# topup_candidates
print("CANDIDATES " + repr([list(script.topup_candidates(w).items())
                            for w in ((4, 8, 13), (13,), (8, 4), ())]))

settings = TopupSettings(method="mean", window=4, with_level=True, compare=True,
                         compare_first_origin=20, compare_last_origin=24,
                         compare_windows=(2, 4))
origins = list(range(20, 25))


def consolidated_errors(forecast_of):
    # forecast_of(company, o, h) -> (A, R, G) forecast; errors of A, R and the top-up per h.
    errors = {h: {"A": [], "R": [], "topup": []} for h in (1, 2)}
    for o in origins:
        for h in (1, 2):
            target = o + h
            fa = fr = fg = 0.0
            for company in companies:
                a, r, g = forecast_of(company, o, h)
                fa, fr, fg = fa + a, fr + r, fg + g
            aa = actual(new_rows, "target_week", "new_customer_cash", h, target)
            ar = actual(refund_rows, "week", "refunds", None, target)
            ag = actual(g_rows, "week", "unidentified_cash", None, target)
            errors[h]["A"].append(fa - aa)
            errors[h]["R"].append(fr - ar)
            errors[h]["topup"].append(fa + fr + fg - aa - ar - ag)
    return errors


def mean_abs(values):
    return sum(abs(v) for v in values) / len(values)


def line_of(errors, fallbacks):
    line = {f"h{h}": mean_abs(errors[h]["topup"]) / 1e6 for h in (1, 2)}
    line["all h"] = (line["h1"] + line["h2"]) / 2
    line["A"] = sum(mean_abs(errors[h]["A"]) for h in (1, 2)) / 2 / 1e6
    line["R"] = sum(mean_abs(errors[h]["R"]) for h in (1, 2)) / 2 / 1e6
    line["bias"] = sum(sum(errors[h]["topup"]) / len(origins) for h in (1, 2)) / 2 / 1e6
    line["fallbacks"] = fallbacks
    return line


def one_company_mean(rows, week_key, value_key, company, h, o, window):
    values = [r[value_key] / level_of[r[week_key]] for r in rows
              if r["company_code"] == company and (h is None or r["h"] == h)
              and o - window < r[week_key] <= o]
    return sum(values) / len(values)


expected = {}
for window in (2, 4):
    def mean_forecast(company, o, h, window=window):
        return (one_company_mean(new_rows, "target_week", "new_customer_cash", company, h, o,
                                 window),
                one_company_mean(refund_rows, "week", "refunds", company, None, o, window),
                one_company_mean(g_rows, "week", "unidentified_cash", company, None, o, window))
    expected[f"mean {window}"] = line_of(consolidated_errors(mean_forecast), 0)
for name, with_level in (("ridge", True), ("ridge no level", False)):
    found = topups.topup_forecasts(new_cash, refunds, unidentified, levels, calendar, origins,
                                   2, "ridge", 1, with_level)
    by_key = found.set_index(["company_code", "origin_week", "h"])

    def ridge_forecast(company, o, h, by_key=by_key):
        got = by_key.loc[(company, o, h)]
        return got["new_customer_cash"], got["refunds"], got["unidentified_cash"]
    expected[name] = line_of(consolidated_errors(ridge_forecast), int(found["fallback"].sum()))
print("EXPECTED " + repr(expected))

copies = [t.copy(deep=True) for t in (new_cash, refunds, unidentified, calendar)]
levels_copy = levels.copy(deep=True)
for label, chosen in (("mean 4", settings),
                      ("ridge", dataclasses.replace(settings, method="ridge")),
                      ("ridge no level", dataclasses.replace(settings, method="ridge",
                                                             with_level=False)),
                      ("mean 13", dataclasses.replace(settings, window=13))):
    print("=== RUN " + label)
    script.compare_topups(new_cash, refunds, unidentified, levels, calendar, chosen, 2)
for table, copy in zip((new_cash, refunds, unidentified, calendar), copies):
    pd.testing.assert_frame_equal(table, copy)
pd.testing.assert_series_equal(levels, levels_copy)
print("ok")
"""

COMPARE_TITLE = ("top-up methods compared at origins 20-24 (5 origins), consolidated top-up: "
                 "mean absolute error per h and over all h, A and R on their own, bias (above "
                 "0: too high), million TRY of week 0; fallbacks = forecasts (company, origin, "
                 "h) whose ridge fell back to the level:")
COMPARE_CANDIDATES = ["mean 2", "mean 4", "ridge", "ridge no level"]
COMPARE_COLUMNS = ["h1", "h2", "all h", "A", "R", "bias", "fallbacks"]


def compare_rows(lines):
    """The candidate rows of a compare_topups printout: name -> the printed numbers."""
    rows = {}
    for line in lines:
        # The longest name first: "ridge no level" before "ridge".
        name = next((n for n in sorted(COMPARE_CANDIDATES, key=len, reverse=True)
                     if line.startswith(n + " ")), None)
        if name is None:
            break
        rows[name] = dict(zip(COMPARE_COLUMNS, line[len(name):].split()))
    return rows


@pytest.fixture(scope="module")
def compare_topups_run():
    result = subprocess.run([sys.executable, "-c", COMPARE_TOPUPS], cwd=REPO_ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    lines = [line.strip() for line in result.stdout.splitlines()]
    candidates = eval(next(line for line in lines if line.startswith("CANDIDATES "))[11:])
    expected = eval(next(line for line in lines if line.startswith("EXPECTED "))[9:])
    runs = {}
    starts = [i for i, line in enumerate(lines) if line.startswith("=== RUN ")]
    for start, stop in zip(starts, starts[1:] + [len(lines) - 1]):
        body = lines[start + 1:stop]
        runs[lines[start][len("=== RUN "):]] = {
            "title": body[0], "header": body[1], "index_name": body[2],
            "rows": compare_rows(body[3:]), "last": body[-1],
            "after_rows": body[3 + len(COMPARE_CANDIDATES):]}
    return {"candidates": candidates, "expected": expected, "runs": runs}


def test_topup_candidates_names_and_methods(compare_topups_run):
    first = compare_topups_run["candidates"][0]
    assert first == [("mean 4", ("mean", 4, True)), ("mean 8", ("mean", 8, True)),
                     ("mean 13", ("mean", 13, True)), ("ridge", ("ridge", 1, True)),
                     ("ridge no level", ("ridge", 1, False))]


def test_topup_candidates_keep_the_windows_order_and_put_the_ridges_last(compare_topups_run):
    _, one, unsorted, none = compare_topups_run["candidates"]
    assert [name for name, _ in one] == ["mean 13", "ridge", "ridge no level"]
    assert [name for name, _ in unsorted] == ["mean 8", "mean 4", "ridge", "ridge no level"]
    # No window: only the two ridges.
    assert none == [("ridge", ("ridge", 1, True)), ("ridge no level", ("ridge", 1, False))]


def test_compare_topups_is_wired_in_after_show_topups_guarded_by_compare(compare_topups_run):
    # The source checks (the guarded call, its order between show_topups and the demo-origin
    # guard, the imports and signatures) ran in the fixture.
    assert set(compare_topups_run["runs"]) == {"mean 4", "ridge", "ridge no level", "mean 13"}


def test_compare_topups_printout_title_header_and_one_row_per_candidate(compare_topups_run):
    for label, run in compare_topups_run["runs"].items():
        assert run["title"] == COMPARE_TITLE, label
        assert run["header"].split() == " ".join(COMPARE_COLUMNS).split(), label
        assert run["index_name"] == "candidate", label
        assert list(run["rows"]) == COMPARE_CANDIDATES, label
        assert all(len(row) == len(COMPARE_COLUMNS) for row in run["rows"].values()), label


def test_compare_topups_printout_matches_the_errors_worked_out_by_hand(compare_topups_run):
    expected = compare_topups_run["expected"]
    rows = compare_topups_run["runs"]["mean 4"]["rows"]
    for name in COMPARE_CANDIDATES:
        for column in COMPARE_COLUMNS[:-1]:
            assert float(rows[name][column]) == pytest.approx(expected[name][column],
                                                              abs=6e-4), (name, column)
        assert int(rows[name]["fallbacks"]) == expected[name]["fallbacks"], name


def test_compare_topups_all_h_is_the_mean_of_the_h_columns(compare_topups_run):
    for name, row in compare_topups_run["runs"]["mean 4"]["rows"].items():
        assert float(row["all h"]) == pytest.approx((float(row["h1"]) + float(row["h2"])) / 2,
                                                    abs=1.1e-3), name


def test_compare_topups_counts_the_ridges_fallbacks_and_none_for_the_means(compare_topups_run):
    # TR04's G (from week 17) has too few weeks at every origin 20..24 for a fit: one fallback
    # per origin and h, 5 x 2 = 10, in each ridge. The means never fall back.
    rows = compare_topups_run["runs"]["mean 4"]["rows"]
    assert rows["mean 2"]["fallbacks"] == rows["mean 4"]["fallbacks"] == "0"
    assert rows["ridge"]["fallbacks"] == rows["ridge no level"]["fallbacks"] == "10"
    assert compare_topups_run["expected"]["ridge"]["fallbacks"] == 10


def test_compare_topups_the_means_differ_by_window(compare_topups_run):
    rows = compare_topups_run["runs"]["mean 4"]["rows"]
    assert rows["mean 2"]["all h"] != rows["mean 4"]["all h"]


def test_compare_topups_last_line_names_the_lowest_error_and_the_chosen_method(
        compare_topups_run):
    expected = compare_topups_run["expected"]
    lowest = min(COMPARE_CANDIDATES, key=lambda name: expected[name]["all h"])
    for label, run in compare_topups_run["runs"].items():
        assert run["last"] == f"lowest error over all h: {lowest}; set in [topups]: {label}", \
            (label, run["last"])


def test_compare_topups_the_printed_table_does_not_depend_on_the_chosen_method(
        compare_topups_run):
    runs = compare_topups_run["runs"]
    for label in ("ridge", "ridge no level", "mean 13"):
        assert runs[label]["rows"] == runs["mean 4"]["rows"], label


def test_compare_topups_a_mean_window_outside_the_comparison_is_still_named(compare_topups_run):
    # read_topups now refuses method "mean" with a window outside compare_windows, so the
    # settings file cannot produce this. compare_topups itself takes any TopupSettings and
    # only labels it: window 13 against compare_windows (2, 4) still prints "mean 13", a
    # method the table above it does not hold. Pinned so the label stays a plain label.
    run = compare_topups_run["runs"]["mean 13"]
    assert run["last"].endswith("; set in [topups]: mean 13")
    assert "mean 13" not in run["rows"]


def test_compare_topups_prints_nothing_between_the_rows_and_the_last_line(compare_topups_run):
    for label, run in compare_topups_run["runs"].items():
        assert run["after_rows"] == [run["last"]], label


# Steps 16 (U13.4 + U14.1) and 17 (U14.2): the wiring is checked on main()'s syntax tree, read
# from the file; the names the script imports, in a separate process (as for the other steps).
# The printouts run on tiny hand-made frames, every number worked out by hand, also in a
# separate process (importing the script reconfigures sys.stdout and sys.path).

FORECAST_KEYWORDS = {
    "block_columns": "block_columns",
    "companies": "companies", "new_cash": "new_cash", "refunds": "refunds",
    "unidentified": "unidentified", "levels": "levels", "calendar": "calendar",
    "week_zero": "week_zero"}


def main_body():
    return main_tree().body


def index_of_if(body, test):
    found = [i for i, node in enumerate(body)
             if isinstance(node, ast.If) and ast.unparse(node.test) == test]
    assert len(found) == 1, (test, found)
    return found[0]


def step_16_body():
    """main()'s statements after step 15's `if` and before step 17's, without the
    `rolled = None` right before step 17's `if` (since U16.3 it sits there, outside step 17,
    so that step 18 can read it when step 17 is switched off; it is not part of step 16)."""
    body = main_body()
    at = index_of_if(body, "settings.evaluation.run")
    assert ast.unparse(body[at - 1]) == "rolled = None"
    return body[index_of_if(body, "settings.joint_diagnostic.run") + 1:at - 1]


def step_17_block():
    body = main_body()
    return body[index_of_if(body, "settings.evaluation.run")]


def keywords_of(call):
    assert all(k.arg is not None for k in call.keywords), ast.unparse(call)   # no **kwargs
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def calls_in(nodes, name):
    return [call for node in nodes for call in calls_to(node, name)]


def test_step_16_comes_after_step_15_and_before_step_17():
    body = main_body()
    assert (index_of_if(body, "settings.joint_diagnostic.run")
            < index_of_if(body, "settings.evaluation.run"))
    assert step_16_body(), "nothing between step 15 and step 17"
    # Step 16 runs whatever the switches say: none of its statements is an `if`.
    assert not any(isinstance(node, ast.If) for node in step_16_body())


def test_step_16_calls_forecast_once_with_the_origin_and_its_keyword_arguments():
    assert len(calls_to(main_tree(), "forecast")) == 1
    step_16 = step_16_body()   # one parse: the call is matched by identity below
    calls = calls_in(step_16, "forecast")
    assert len(calls) == 1
    call = calls[0]
    assert [ast.unparse(a) for a in call.args] == ["table", "origin", "settings"]
    assert keywords_of(call) == FORECAST_KEYWORDS
    assigned = [node for node in step_16 if isinstance(node, ast.Assign)
                and node.value is call]
    assert len(assigned) == 1 and [ast.unparse(t) for t in assigned[0].targets] == ["result"]


def test_step_16_statements_in_order():
    assert [ast.unparse(node) for node in step_16_body()] == [
        "started = time.perf_counter()",
        "result = forecast(table, origin, settings, block_columns=block_columns, "
        "companies=companies, new_cash=new_cash, refunds=refunds, "
        "unidentified=unidentified, levels=levels, calendar=calendar, week_zero=week_zero)",
        "show_customer_forecasts(result.customers, forecast_rows, time.perf_counter() - "
        "started, predictions['stage 1 (HGB)'] * m)",
        "show_forecast_totals(result.totals)",
        "actual = actual_totals(weekly, weeks, levels, companies, [origin], "
        "settings.receivables.horizons)",
        "show_actual_totals(actual, weekly, weeks)",
        "show_forecast_vs_actual(result.totals, actual)",
    ]


def test_step_16_customer_forecasts_are_checked_against_step_14s_p_times_m():
    calls = calls_in(step_16_body(), "show_customer_forecasts")
    assert len(calls) == 1 and len(calls_to(main_tree(), "show_customer_forecasts")) == 1
    args = [ast.unparse(a) for a in calls[0].args]
    assert args[:2] == ["result.customers", "forecast_rows"]
    # Stage 1's HGB p (not the LR baseline's) times step 14's m.
    assert args[3] == "predictions['stage 1 (HGB)'] * m"


def test_step_16_actual_totals_at_the_one_origin():
    calls = calls_in(step_16_body(), "actual_totals")
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == [
        "weekly", "weeks", "levels", "companies", "[origin]", "settings.receivables.horizons"]
    assert calls[0].keywords == []


def test_step_16_shows_the_totals_and_the_forecast_beside_the_actual_once():
    tree = main_tree()
    for name, args in (("show_forecast_totals", ["result.totals"]),
                       ("show_actual_totals", ["actual", "weekly", "weeks"]),
                       ("show_forecast_vs_actual", ["result.totals", "actual"])):
        assert len(calls_to(tree, name)) == 1, name
        call = calls_in(step_16_body(), name)
        assert len(call) == 1, name
        assert [ast.unparse(a) for a in call[0].args] == args, name


def test_step_16_and_17_imports():
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forecast_imports = [node for node in module.body if isinstance(node, ast.ImportFrom)
                        and node.module == "collection_estimation.hurdle.forecast"]
    assert len(forecast_imports) == 1
    # U16.1 added stage1_predictions and stage2_predictions (step 18); U16.3 removed them
    # (step 18 now fits through grid_predictions). CONSOLIDATED went with the printouts to
    # hurdle/reports/.
    assert [(a.name, a.asname) for a in forecast_imports[0].names] == [("forecast", None)]
    evaluate_imports = [node for node in module.body if isinstance(node, ast.ImportFrom)
                        and node.module == "collection_estimation.hurdle.evaluate"]
    assert len(evaluate_imports) == 1
    # U14.3 added REFERENCES and deflated_reference to the one import of evaluate; U14.5
    # added HORIZON_TOTAL, POOLED, evaluation_scores and stack_forecasts; the cache of step 17
    # added rolling_fingerprint; U14.5c added HURDLE and paired_wape_bootstrap; U14.5d added
    # KNOWN_CUSTOMERS, RULE, component_bias and payer_split; U14.6 added balance_coherence;
    # U16.5 added STACK_COLUMNS (show_grid_scores stacks step 17's totals with it). The
    # constants only the printouts read (HORIZON_TOTAL, KNOWN_CUSTOMERS, POOLED, REFERENCES,
    # RULE, STACK_COLUMNS) went with them to hurdle/reports/; main() still reads HURDLE.
    assert [(a.name, a.asname) for a in evaluate_imports[0].names] == [
        ("HURDLE", None), ("actual_totals", None), ("balance_coherence", None),
        ("component_bias", None), ("deflated_reference", None), ("evaluation_scores", None),
        ("paired_wape_bootstrap", None), ("payer_split", None), ("rolling_fingerprint", None),
        ("rolling_forecasts", None), ("rule_reference", None), ("stack_forecasts", None)]
    # The cache's one function, in an import of its own, right after evaluate's.
    cache_imports = [node for node in module.body if isinstance(node, ast.ImportFrom)
                     and node.module == "collection_estimation.hurdle.cache"]
    assert len(cache_imports) == 1
    assert [(a.name, a.asname) for a in cache_imports[0].names] == [("load_with_cache", None)]
    # U15 put the import of calibrate between them.
    assert module.body.index(cache_imports[0]) == module.body.index(evaluate_imports[0]) + 2
    calibrate_imports = [node for node in module.body if isinstance(node, ast.ImportFrom)
                         and node.module == "collection_estimation.hurdle.calibrate"]
    assert len(calibrate_imports) == 1
    # U16.7 added candidate_name (show_grid_test names the frozen corrections with it); it
    # went with show_grid_test to hurdle/reports/grid.py.
    assert [(a.name, a.asname) for a in calibrate_imports[0].names] == [
        ("calibration_reliability", None), ("choose_calibration", None)]
    assert module.body.index(calibrate_imports[0]) == module.body.index(evaluate_imports[0]) + 1


STEP_16_17_NAMES = """
import inspect
import run_hurdle_pipeline as script
from collection_estimation.hurdle import inputs as prep
from collection_estimation.hurdle import evaluate
from collection_estimation.hurdle import forecast as forecast_module

from collection_estimation.hurdle.reports import evaluation as reports
assert script.forecast is forecast_module.forecast
# Every printout of steps 16 and 17 is the evaluation report module's own.
for name in ("show_customer_forecasts", "show_forecast_totals", "show_actual_totals",
             "show_forecast_vs_actual", "show_rolling_forecasts", "show_deflated_reference",
             "show_rule_reference", "show_evaluation_scores", "show_paired_bootstrap",
             "show_component_bias", "show_payer_split", "show_balance_coherence",
             "show_calibration_reliability", "show_calibration_choice"):
    assert getattr(script, name) is getattr(reports, name), name
assert script.actual_totals is evaluate.actual_totals
assert script.rolling_forecasts is evaluate.rolling_forecasts
inspect.signature(script.show_customer_forecasts).bind("customers", "forecast_rows", 1.0,
                                                       "expected")
inspect.signature(script.show_forecast_totals).bind("totals")
inspect.signature(script.show_actual_totals).bind("actual", "weekly", "weeks")
inspect.signature(script.show_forecast_vs_actual).bind("totals", "actual")
inspect.signature(script.show_rolling_forecasts).bind("rolled", "actual")
# The keyword arguments steps 16 and 17 pass are the ones forecast and rolling_forecasts take.
KEYWORDS = dict(block_columns={}, companies=[], new_cash=None, refunds=None, unidentified=None,
                levels=None, calendar=None, week_zero=None)
inspect.signature(script.forecast).bind("table", 0, "settings", **KEYWORDS)
inspect.signature(script.rolling_forecasts).bind("table", [0], "settings", **KEYWORDS,
                                                 on_origin_done=print)
# The cache of step 17: the fingerprint takes the fit's arguments (no callback).
from collection_estimation.hurdle import cache as cache_module
assert script.rolling_fingerprint is evaluate.rolling_fingerprint
assert script.load_with_cache is cache_module.load_with_cache
inspect.signature(script.rolling_fingerprint).bind("table", [0], "settings", **KEYWORDS)
inspect.signature(script.load_with_cache).bind("name", [], print, "folder")
inspect.signature(script.actual_totals).bind("weekly", "weeks", "levels", [], [0], 5)
# U14.3.
assert script.deflated_reference is evaluate.deflated_reference
inspect.signature(script.deflated_reference).bind("weekly", "weeks", "levels", "calendar", [],
                                                  [0], 5, None)
inspect.signature(script.show_deflated_reference).bind("reference", "rolled", "actual")
# U14.4.
assert script.rule_reference is evaluate.rule_reference
inspect.signature(script.rule_reference).bind(
    "ledger", "rates", [0], "settings", companies=[], new_cash=None, refunds=None,
    unidentified=None, levels=None, calendar=None, week_zero=None)
inspect.signature(script.show_rule_reference).bind("rule", "actual")
# U14.5.
assert script.stack_forecasts is evaluate.stack_forecasts
assert script.evaluation_scores is evaluate.evaluation_scores
inspect.signature(script.stack_forecasts).bind(None, "reference", "rule")
inspect.signature(script.evaluation_scores).bind("forecasts", "actual", 5)
inspect.signature(script.show_evaluation_scores).bind("scores", 5)
# U14.5c.
assert script.HURDLE is evaluate.HURDLE and script.HURDLE == "hurdle"
assert script.paired_wape_bootstrap is evaluate.paired_wape_bootstrap
inspect.signature(script.paired_wape_bootstrap).bind("forecasts", "actual", [], 5, 2000, 0)
inspect.signature(script.show_paired_bootstrap).bind("bootstrap")
# U14.5d.
assert script.component_bias is evaluate.component_bias
assert script.payer_split is evaluate.payer_split
inspect.signature(script.component_bias).bind(None, "rule", "actual", "new_cash", "refunds",
                                              "unidentified", "levels", 5)
inspect.signature(script.payer_split).bind("customers", "weekly", "weeks", "levels", 5)
inspect.signature(script.show_component_bias).bind("bias", 5)
inspect.signature(script.show_payer_split).bind("split", 5)
# U14.6.
assert script.balance_coherence is evaluate.balance_coherence
inspect.signature(script.balance_coherence).bind("customers", "table", "weekly", "weeks",
                                                 "levels", 5, 1.5)
inspect.signature(script.show_balance_coherence).bind("coherence", 1.5)
# U15.
from collection_estimation.hurdle import calibrate
assert script.calibration_reliability is calibrate.calibration_reliability
assert script.choose_calibration is calibrate.choose_calibration
inspect.signature(script.calibration_reliability).bind("customers", "weekly", "weeks", "levels", 10)
inspect.signature(script.choose_calibration).bind("rolled", "weekly", "weeks", "levels", "actual",
                                                  "settings")
inspect.signature(script.show_calibration_reliability).bind("reliability")
inspect.signature(script.show_calibration_choice).bind("choice", 0.005)
print("ok")
"""


def test_step_16_and_17_names_are_the_modules_functions():
    result = run_code(STEP_16_17_NAMES)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_step_17_is_the_last_statement_of_main_behind_its_run_switch():
    # Since U16.1 only step 18 follows step 17's `if`; since U16.2 step 18 is one `if` of its
    # own, on [grid] run; since step 18 moved to grid_run.py, its one statement is run_grid.
    body = main_body()
    at = index_of_if(body, "settings.evaluation.run")
    block = body[at]
    assert block.orelse == []
    assert len(body[at + 1:]) == 1
    assert ast.unparse(body[at + 1].test) == "settings.grid.run"
    assert [ast.unparse(node) for node in body[at + 1].body] == [RUN_GRID]
    assert len(main_through_step_17()) == at + 1


def main_through_step_17():
    """main()'s statements up to and including step 17's `if` (step 18 follows it)."""
    body = main_body()
    return body[:index_of_if(body, "settings.evaluation.run") + 1]


def test_step_17_uses_the_validation_origins_only():
    block = step_17_block()
    origins = [node for node in block.body if isinstance(node, ast.Assign)
               and [ast.unparse(t) for t in node.targets] == ["origins"]]
    assert len(origins) == 1
    assert ast.unparse(origins[0].value) == "list(settings.evaluation.validation_origins)"
    assert block.body.index(origins[0]) == 0
    # The test origins are scored only in U16 (the v1 test, inside run_grid): main()'s code
    # never names them, nor run_test (its comments may).
    code = ast.unparse(main_tree())
    assert "test_origins" not in code
    assert "run_test" not in code


def test_step_17_calls_rolling_forecasts_once_with_a_progress_callback():
    # Since with_model (2026-11-03) the call and its callback sit in the with_model branch.
    assert len(calls_to(main_tree(), "rolling_forecasts")) == 1
    branch = with_model_if()
    calls = calls_to(branch, "rolling_forecasts")
    assert len(calls) == 1
    call = calls[0]
    assert [ast.unparse(a) for a in call.args] == ["table", "origins", "settings"]
    # Since the cache (2026-11-03): block_columns is a name assigned in the branch, and the
    # call is the one statement of fit_every_origin, which load_with_cache calls on a miss.
    assert keywords_of(call) == {**FORECAST_KEYWORDS, "block_columns": "block_columns",
                                 "on_origin_done": "progress"}
    fits = [node for node in branch.body
            if isinstance(node, ast.FunctionDef) and node.name == "fit_every_origin"]
    assert len(fits) == 1
    assert len(fits[0].body) == 1 and isinstance(fits[0].body[0], ast.Return)
    assert fits[0].body[0].value is call
    # progress is defined inside the branch, before the fit, with rolling_forecasts' four
    # callback arguments (done, total, origin, seconds).
    defs = [(i, node) for i, node in enumerate(branch.body)
            if isinstance(node, ast.FunctionDef) and node.name == "progress"]
    assert len(defs) == 1
    index, progress = defs[0]
    assert [a.arg for a in progress.args.args] == ["done", "total", "at", "seconds"]
    assert index < branch.body.index(fits[0])
    assert all(calls_to(node, "rolling_forecasts") == [] for node in branch.orelse)


def test_step_17_actual_totals_at_the_validation_origins_then_the_printout():
    block = step_17_block()
    calls = calls_to(block, "actual_totals")
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == [
        "weekly", "weeks", "levels", "companies", "origins", "settings.receivables.horizons"]
    # Step 16's and step 17's; step 18's two (the grid's and the v1 test's) are run_grid's.
    assert len(calls_to(main_tree(), "actual_totals")) == 2
    shows = calls_to(block, "show_rolling_forecasts")
    assert len(shows) == 1 and len(calls_to(main_tree(), "show_rolling_forecasts")) == 1
    assert [ast.unparse(a) for a in shows[0].args] == ["rolled", "rolled_actual"]
    statements = [ast.unparse(node) for node in block.body]
    # Since with_model: the actual cash comes first (right after the header), then the
    # with_model branch, whose last statement is the printout. (rolled = None was between
    # them until U16.3; it now sits right before step 17's `if`, so step 18 can read it.)
    assert statements[2] == ("rolled_actual = actual_totals(weekly, weeks, levels, companies, "
                             "origins, settings.receivables.horizons)")
    assert ast.unparse(step_17_block().body[3].test) == "settings.evaluation.with_model"
    assert "rolled = None" not in statements
    assert ast.unparse(with_model_if().body[-1]) == "show_rolling_forecasts(rolled, rolled_actual)"


# Step 17, U14.3: the `deflated` reference at the validation origins, right after the rolling
# forecasts' printout, and its printout. Wiring on main()'s syntax tree, as above.

DEFLATED_REFERENCE_ARGS = ["weekly", "weeks", "levels", "calendar", "companies", "origins",
                           "settings.receivables.horizons", "week_zero"]


def test_step_17_calls_deflated_reference_once_with_its_arguments():
    assert len(calls_to(main_tree(), "deflated_reference")) == 1
    block = step_17_block()
    calls = calls_to(block, "deflated_reference")
    assert len(calls) == 1
    call = calls[0]
    assert [ast.unparse(a) for a in call.args] == DEFLATED_REFERENCE_ARGS
    assert call.keywords == []
    assigned = [node for node in block.body if isinstance(node, ast.Assign)
                and node.value is call]
    assert len(assigned) == 1 and [ast.unparse(t) for t in assigned[0].targets] == ["reference"]


def test_step_17_deflated_reference_is_a_direct_statement_of_the_evaluation_block():
    # Directly in the `if settings.evaluation.run:` body, not nested deeper, not in an else.
    block = step_17_block()
    direct = [node for node in block.body if isinstance(node, ast.Assign)
              and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
              and node.value.func.id == "deflated_reference"]
    assert len(direct) == 1
    assert block.orelse == []


def test_step_17_deflated_reference_comes_after_show_rolling_forecasts():
    # Right after the with_model branch (which ends with show_rolling_forecasts).
    block = step_17_block()
    body = block.body
    branch = body.index(with_model_if(block))
    assert ast.unparse(body[branch + 1]) == (
        "reference = deflated_reference(weekly, weeks, levels, calendar, companies, origins, "
        "settings.receivables.horizons, week_zero)")


def test_step_17_shows_the_deflated_reference_once_with_the_reference_rolled_and_actual():
    assert len(calls_to(main_tree(), "show_deflated_reference")) == 1
    calls = calls_to(step_17_block(), "show_deflated_reference")
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == ["reference", "rolled", "rolled_actual"]
    assert calls[0].keywords == []


def test_step_17_ends_with_the_deflated_reference_and_its_printout():
    # U14.4 appended the financial rule's two statements after the reference's two,
    # U14.5 the scores' three after those, U14.5c the bootstrap's four after those, and U14.5d
    # the three of the bias and the payer split after those.
    statements = [ast.unparse(node) for node in step_17_block().body]
    assert statements[-14:-12] == [
        "reference = deflated_reference(weekly, weeks, levels, calendar, companies, origins, "
        "settings.receivables.horizons, week_zero)",
        "show_deflated_reference(reference, rolled, rolled_actual)"]


def test_step_17_deflated_reference_uses_the_same_origins_as_the_rolling_forecasts():
    block = step_17_block()
    rolled = calls_to(block, "rolling_forecasts")[0]
    reference = calls_to(block, "deflated_reference")[0]
    assert ast.unparse(rolled.args[1]) == "origins" == ast.unparse(reference.args[5])
    # origins is assigned once in the block, before both calls.
    assigned = [node for node in block.body if isinstance(node, ast.Assign)
                and "origins" in [ast.unparse(t) for t in node.targets]]
    assert len(assigned) == 1


def test_step_17_is_still_the_last_step_after_the_deflated_reference():
    # Since U16.1 step 18 follows step 17's `if`: main() up to that `if`.
    body = main_through_step_17()
    assert isinstance(body[-1], ast.If)
    assert ast.unparse(body[-1].test) == "settings.evaluation.run"
    # Since U14.5d the deflated reference's printout is thirteenth from last, before the rule's
    # two statements, the scores' three, the bootstrap's four and the three of the bias.
    assert ast.unparse(body[-1].body[-13]) == (
        "show_deflated_reference(reference, rolled, rolled_actual)")



# Step 17, U14.4: the financial-rule reference at the validation origins, right after the
# deflated reference's printout, and its printout. Wiring on main()'s syntax tree, as above.

RULE_REFERENCE_KEYWORDS = {
    "companies": "companies", "new_cash": "new_cash", "refunds": "refunds",
    "unidentified": "unidentified", "levels": "levels", "calendar": "calendar",
    "week_zero": "week_zero"}
RULE_STATEMENT = ("rule = rule_reference(ledger, rates, origins, settings, companies=companies, "
                  "new_cash=new_cash, refunds=refunds, unidentified=unidentified, "
                  "levels=levels, calendar=calendar, week_zero=week_zero)")
SHOW_RULE_STATEMENT = "show_rule_reference(rule, rolled_actual)"


def test_step_17_calls_rule_reference_once_with_its_arguments():
    assert len(calls_to(main_tree(), "rule_reference")) == 1
    block = step_17_block()
    calls = calls_to(block, "rule_reference")
    assert len(calls) == 1
    call = calls[0]
    assert [ast.unparse(a) for a in call.args] == ["ledger", "rates", "origins", "settings"]
    assert keywords_of(call) == RULE_REFERENCE_KEYWORDS
    # The keywords in the order written.
    assert [k.arg for k in call.keywords] == list(RULE_REFERENCE_KEYWORDS)
    assigned = [node for node in block.body if isinstance(node, ast.Assign)
                and node.value is call]
    assert len(assigned) == 1 and [ast.unparse(t) for t in assigned[0].targets] == ["rule"]


def test_step_17_rule_reference_is_a_direct_statement_of_the_evaluation_block():
    block = step_17_block()
    direct = [node for node in block.body if isinstance(node, ast.Assign)
              and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
              and node.value.func.id == "rule_reference"]
    assert len(direct) == 1
    shown = [node for node in block.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "show_rule_reference"]
    assert len(shown) == 1
    assert block.orelse == []


def test_step_17_rule_reference_comes_right_after_show_deflated_reference():
    statements = [ast.unparse(node) for node in step_17_block().body]
    shown = statements.index("show_deflated_reference(reference, rolled, rolled_actual)")
    assert statements.index(RULE_STATEMENT) == shown + 1
    assert statements.index(SHOW_RULE_STATEMENT) == shown + 2


def test_step_17_shows_the_rule_reference_once_with_the_rule_and_the_actual():
    assert len(calls_to(main_tree(), "show_rule_reference")) == 1
    calls = calls_to(step_17_block(), "show_rule_reference")
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == ["rule", "rolled_actual"]
    assert calls[0].keywords == []


def test_step_17_ends_with_the_rule_reference_and_its_printout():
    # Since U14.5 the scores' three statements follow the rule's two, since U14.5c the
    # bootstrap's four follow those, and since U14.5d the three of the bias those.
    statements = [ast.unparse(node) for node in step_17_block().body]
    assert statements[-12:-10] == [RULE_STATEMENT, SHOW_RULE_STATEMENT]


def test_step_17_is_still_the_last_step_after_the_rule_reference():
    # Since U16.1 step 18 follows step 17's `if`: main() up to that `if`.
    body = main_through_step_17()
    assert isinstance(body[-1], ast.If)
    assert ast.unparse(body[-1].test) == "settings.evaluation.run"
    assert body[-1].orelse == []
    # Since U14.5d the rule's printout is eleventh from last, before the scores' three, the
    # bootstrap's four and the three of the bias.
    assert ast.unparse(body[-1].body[-11]) == SHOW_RULE_STATEMENT


def test_step_17_rule_reference_uses_the_same_origins_as_the_rolling_forecasts():
    block = step_17_block()
    rolled = calls_to(block, "rolling_forecasts")[0]
    rule = calls_to(block, "rule_reference")[0]
    assert ast.unparse(rolled.args[1]) == "origins" == ast.unparse(rule.args[2])
    assigned = [node for node in block.body if isinstance(node, ast.Assign)
                and "origins" in [ast.unparse(t) for t in node.targets]]
    assert len(assigned) == 1
    rule_statement = [node for node in block.body if isinstance(node, ast.Assign)
                      and [ast.unparse(t) for t in node.targets] == ["rule"]]
    assert block.body.index(assigned[0]) < block.body.index(rule_statement[0])


def test_step_17_rule_reference_is_shown_against_the_rolled_actual_made_before_it():
    statements = [ast.unparse(node) for node in step_17_block().body]
    made = [i for i, s in enumerate(statements) if s.startswith("rolled_actual = ")]
    assert len(made) == 1
    assert made[0] < statements.index(SHOW_RULE_STATEMENT)


def test_rule_reference_inputs_are_assigned_in_main_before_step_17():
    # ledger and rates, the two inputs only the rule needs, exist before the evaluation block.
    # Since steps 1-11 moved to prepare_inputs they are unpacked from its inputs, several
    # names per statement (a tuple target).
    body = main_body()
    step_17 = index_of_if(body, "settings.evaluation.run")
    for name in ("ledger", "rates"):
        assigned = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                    and name in [n.id for t in node.targets for n in ast.walk(t)
                                 if isinstance(n, ast.Name)]]
        assert assigned and max(assigned) < step_17, name
        assert ast.unparse(body[max(assigned)]).count(f"inputs.{name}") == 1, name


def test_rule_reference_is_imported_once_from_evaluate():
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    found = [(node.module, a.name, a.asname) for node in ast.walk(module)
             if isinstance(node, ast.ImportFrom) for a in node.names
             if a.name == "rule_reference"]
    assert found == [("collection_estimation.hurdle.evaluate", "rule_reference", None)]


from types import SimpleNamespace as SimpleNamespace  # noqa: E402


# Step 17, [evaluation] with_model (2026-11-03): the hurdle model's forecasts run only if
# with_model is true; the references and the rule always run. Wiring on main()'s syntax tree;
# then both branches run for real by executing step 17's block with recording fakes.

SKIP_LINE = "        the hurdle model's forecasts are skipped ([evaluation] with_model = false)"


def with_model_if(block=None):
    """The `if settings.evaluation.with_model:` statement directly in step 17's block (of
    `block`, if given: main_tree() parses afresh each time, so nodes compare by identity only
    within one parse)."""
    block = block if block is not None else step_17_block()
    found = [node for node in block.body if isinstance(node, ast.If)
             and ast.unparse(node.test) == "settings.evaluation.with_model"]
    assert len(found) == 1
    return found[0]


def test_step_17_has_one_with_model_branch_and_only_the_payer_split_guard_besides():
    # U14.5d added the guard of payer_split, on `rolled` and not on the setting.
    ifs = [node for node in step_17_block().body if isinstance(node, ast.If)]
    assert len(ifs) == 2
    assert ast.unparse(ifs[0].test) == "settings.evaluation.with_model"
    assert ast.unparse(ifs[1].test) == "rolled is not None"
    assert "with_model" in ast.unparse(step_17_block())
    assert ast.unparse(main_tree()).count("settings.evaluation.with_model") == 1


def test_step_17_statement_order_with_the_with_model_branch():
    block = step_17_block()
    body = block.body
    shapes = [type(node).__name__ for node in body]
    # U14.5 appended stack_forecasts, evaluation_scores (two Assign) and the printout (Expr);
    # U14.5c subject, pairs, the bootstrap (three Assign) and its printout (Expr);
    # U14.5d the bias (Assign), its printout (Expr) and the guarded payer split (If).
    # U16.3 moved rolled = None (the Assign at 3) out, right before step 17's `if`.
    assert shapes == ["Assign", "Expr", "Assign", "If", "Assign", "Expr", "Assign",
                      "Expr", "Assign", "Assign", "Expr", "Assign", "Assign", "Assign", "Expr",
                      "Assign", "Expr", "If"]
    assert ast.unparse(body[0]) == "origins = list(settings.evaluation.validation_origins)"
    assert ast.unparse(body[2]).startswith("rolled_actual = actual_totals(")
    assert body[3] is with_model_if(block)
    assert ast.unparse(body[4]).startswith("reference = deflated_reference(")
    assert ast.unparse(body[5]) == "show_deflated_reference(reference, rolled, rolled_actual)"
    assert ast.unparse(body[6]).startswith("rule = rule_reference(")
    assert ast.unparse(body[7]) == "show_rule_reference(rule, rolled_actual)"
    assert ast.unparse(body[8]) == "forecasts = stack_forecasts(rolled, reference, rule)"
    assert ast.unparse(body[9]) == ("scores = evaluation_scores(forecasts, rolled_actual, "
                                    "settings.receivables.horizons)")
    assert ast.unparse(body[10]) == ("show_evaluation_scores(scores, "
                                     "settings.receivables.horizons)")
    assert ast.unparse(body[11]) == BOOT_SUBJECT_STATEMENT
    assert ast.unparse(body[12]) == BOOT_PAIRS_STATEMENT
    assert ast.unparse(body[13]) == BOOT_CALL_STATEMENT
    assert ast.unparse(body[14]) == BOOT_SHOW_STATEMENT
    assert ast.unparse(body[15]) == BIAS_CALL_STATEMENT
    assert ast.unparse(body[16]) == BIAS_SHOW_STATEMENT
    assert ast.unparse(body[17].test) == "rolled is not None"


def test_step_17_with_model_branch_body_in_order():
    branch = with_model_if().body
    # Since the cache (2026-11-03): the fit as a function, the fingerprint, and the forecasts
    # from load_with_cache. The block columns are no longer made here: they come with the
    # inputs of steps 1-11 (inputs.block_columns, unpacked at the top of main()).
    assert [type(node).__name__ for node in branch] == ["Assign", "FunctionDef",
                                                        "FunctionDef", "Assign", "Assign",
                                                        "Expr"]
    assert ast.unparse(branch[0]) == "started = time.perf_counter()"
    assert branch[1].name == "progress"
    assert branch[2].name == "fit_every_origin"
    assert ast.unparse(branch[3]).startswith("fingerprint = rolling_fingerprint(")
    assert ast.unparse(branch[4]) == ("rolled = load_with_cache('hurdle_rolling_forecasts', "
                                      "fingerprint, fit_every_origin, "
                                      "settings.data.cache_folder)")
    assert ast.unparse(branch[5]) == "show_rolling_forecasts(rolled, rolled_actual)"


def test_step_17_the_else_branch_prints_the_skip_line_only():
    orelse = with_model_if().orelse
    assert len(orelse) == 1
    call = orelse[0].value
    assert isinstance(orelse[0], ast.Expr) and ast.unparse(call.func) == "print"
    assert len(call.args) == 1 and call.keywords == []
    assert ast.literal_eval(call.args[0]) == SKIP_LINE


def test_step_17_the_skip_line_and_rolled_none_are_outside_nothing_else():
    # Nothing of the model (rolling_forecasts, show_rolling_forecasts, progress, started) is
    # outside the with_model branch.
    block = step_17_block()
    branch = with_model_if(block)
    outside = [node for node in block.body if node is not branch]
    for name in ("rolling_forecasts", "show_rolling_forecasts"):
        assert calls_in(outside, name) == [], name
    text = "\n".join(ast.unparse(node) for node in outside)
    assert "progress" not in text and "started" not in text


def test_step_17_header_no_longer_says_two_fits_each():
    header = step_17_block().body[1]
    assert ast.unparse(header.value.func) == "print"
    text = ast.unparse(header)
    assert "two fits" not in text
    assert text.endswith("({len(origins)} origins)')")


def test_step_17_rolled_none_is_assigned_once_right_before_the_branch():
    # Since U16.3 rolled = None is main()'s statement right before step 17's `if` (step 18
    # reads rolled even when step 17 is off); step 17's own statements no longer assign it.
    main = main_body()
    at = index_of_if(main, "settings.evaluation.run")
    assert ast.unparse(main[at - 1]) == "rolled = None"
    body = main[at].body
    rolled = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
              and [ast.unparse(t) for t in node.targets] == ["rolled"]]
    assert rolled == []
    # Nothing between it and the branch reads or sets rolled.
    before_branch = body[:body.index(with_model_if(main[at]))]
    assert not any(isinstance(n, ast.Name) and n.id == "rolled"
                   for node in before_branch for n in ast.walk(node))


def test_step_17_the_references_get_rolled_whatever_the_branch():
    calls = calls_to(step_17_block(), "show_deflated_reference")
    assert [ast.unparse(a) for a in calls[0].args] == ["reference", "rolled", "rolled_actual"]
    # The rule never sees the model.
    calls = calls_to(step_17_block(), "show_rule_reference")
    assert [ast.unparse(a) for a in calls[0].args] == ["rule", "rolled_actual"]


# Running step 17's block (both branches) with fakes: every name it reads is a recorder.

# main()'s names that step 17 reads, as unpacked from prepare_inputs' inputs (history and owed
# stay inside prepare_inputs; block_columns is given below as "<blocks>").
STEP_17_NAMES = ["weekly", "weeks", "levels", "companies", "table", "calendar", "new_cash",
                 "refunds", "unidentified", "week_zero", "ledger", "rates"]


STEP_17_FINGERPRINT = ["<fingerprint>", 1]
# U14.5c: the fake stack's methods, and the fake [evaluation] bootstrap settings.
STEP_17_HURDLE = "hurdle"
STEP_17_OTHER_METHODS = ["deflated", "deflated_nolevel", "deflated_trend", "rule"]
STEP_17_SAMPLES = 2000
# U14.6: the fake [evaluation] coherence_factor.
STEP_17_FACTOR = 1.75
STEP_17_SEED = 7
# U15: the fake [diagnostic] calibration_groups and [calibration] min_gain.
STEP_17_GROUPS = 12
STEP_17_MIN_GAIN = 0.0125


def step_17_rolled(**fields):
    """A fake of the rolling forecasts: since U14.5d step 17 reads their `customers`. A plain
    namespace, so that the real cache can pickle it and two of them compare by their fields."""
    return SimpleNamespace(customers="<rolled customers>", **fields)


def run_step_17(with_model, cache_folder="<cache folder>", load_with_cache=None,
                rolled_result=None, horizons=5):
    """Execute step 17's `if` with fake inputs and functions; return (calls, printed,
    rolled_result, actual_result).

    By default `load_with_cache` is a recorder that always misses: it records its call, then
    calls the fit function it was given (so rolling_forecasts is recorded right after it). A
    real `load_with_cache` (and a real `cache_folder`) can be passed instead; `rolled_result`
    is what the fake rolling_forecasts returns (a fresh `step_17_rolled()` by default)."""
    calls, printed = [], []

    def recorder(name, result):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result
        return call

    def always_miss(*args, **kwargs):
        calls.append(("load_with_cache", args, kwargs))
        return args[2]()

    def fake_stack(rolled, reference, rule):
        # U14.5c reads the stacked methods, so the fake stack holds them, as stack_forecasts
        # would: the hurdle model's only if rolled is not None.
        calls.append(("stack_forecasts", (rolled, reference, rule), {}))
        methods = ([STEP_17_HURDLE] if rolled is not None else []) + STEP_17_OTHER_METHODS
        return {"method": [method for method in methods for _ in range(2)]}

    rolled_result = step_17_rolled() if rolled_result is None else rolled_result
    actual_result = object()
    namespace = {name: f"<{name}>" for name in STEP_17_NAMES}
    settings = SimpleNamespace(
        evaluation=SimpleNamespace(run=True, with_model=with_model,
                                   validation_origins=(33, 34, 35),
                                   bootstrap_samples=STEP_17_SAMPLES,
                                   bootstrap_seed=STEP_17_SEED,
                                   coherence_factor=STEP_17_FACTOR),
        receivables=SimpleNamespace(horizons=horizons),
        diagnostic=SimpleNamespace(calibration_groups=STEP_17_GROUPS),
        calibration=SimpleNamespace(min_gain=STEP_17_MIN_GAIN),
        data=SimpleNamespace(cache_folder=cache_folder))
    namespace.update(
        settings=settings, time=SimpleNamespace(perf_counter=lambda: 0.0),
        print=lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)),
        actual_totals=recorder("actual_totals", actual_result),
        rolling_forecasts=recorder("rolling_forecasts", rolled_result),
        # The cache of the rolling forecasts (2026-11-03).
        rolling_fingerprint=recorder("rolling_fingerprint", STEP_17_FINGERPRINT),
        load_with_cache=always_miss if load_with_cache is None else load_with_cache,
        # The block columns come with the inputs (made once, in prepare_inputs).
        block_columns="<blocks>",
        show_rolling_forecasts=recorder("show_rolling_forecasts", None),
        deflated_reference=recorder("deflated_reference", "<reference>"),
        show_deflated_reference=recorder("show_deflated_reference", None),
        rule_reference=recorder("rule_reference", "<rule>"),
        show_rule_reference=recorder("show_rule_reference", None),
        # U14.5.
        stack_forecasts=fake_stack,
        evaluation_scores=recorder("evaluation_scores", "<scores>"),
        show_evaluation_scores=recorder("show_evaluation_scores", None),
        # U14.5c.
        HURDLE=STEP_17_HURDLE,
        paired_wape_bootstrap=recorder("paired_wape_bootstrap", "<bootstrap>"),
        show_paired_bootstrap=recorder("show_paired_bootstrap", None),
        # U14.5d.
        component_bias=recorder("component_bias", "<bias>"),
        show_component_bias=recorder("show_component_bias", None),
        payer_split=recorder("payer_split", "<split>"),
        show_payer_split=recorder("show_payer_split", None),
        # U14.6.
        balance_coherence=recorder("balance_coherence", "<coherence>"),
        show_balance_coherence=recorder("show_balance_coherence", None),
        # U15.
        calibration_reliability=recorder("calibration_reliability", "<reliability>"),
        show_calibration_reliability=recorder("show_calibration_reliability", None),
        choose_calibration=recorder("choose_calibration", "<choice>"),
        show_calibration_choice=recorder("show_calibration_choice", None))
    # Since U16.3 rolled = None is main()'s statement right before step 17's `if`: run both.
    body = main_body()
    at = index_of_if(body, "settings.evaluation.run")
    assert ast.unparse(body[at - 1]) == "rolled = None"
    module = ast.Module(body=body[at - 1:at + 1], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<step 17>", "exec"), namespace)
    return calls, printed, rolled_result, actual_result


def names_called(calls):
    return [name for name, _, _ in calls if name != "block_feature_columns"]


def test_step_17_with_model_true_calls_everything_in_order():
    calls, printed, _, _ = run_step_17(True)
    # Since the cache (2026-11-03): the fingerprint, then load_with_cache, which (on a miss)
    # calls the fit, i.e. rolling_forecasts.
    assert names_called(calls) == ["actual_totals", "rolling_fingerprint", "load_with_cache",
                                   "rolling_forecasts",
                                   "show_rolling_forecasts", "deflated_reference",
                                   "show_deflated_reference", "rule_reference",
                                   "show_rule_reference", "stack_forecasts",
                                   "evaluation_scores", "show_evaluation_scores",
                                   "paired_wape_bootstrap", "show_paired_bootstrap",
                                   "component_bias", "show_component_bias",
                                   "payer_split", "show_payer_split",
                                   "balance_coherence", "show_balance_coherence",
                                   "calibration_reliability", "show_calibration_reliability",
                                   "choose_calibration", "show_calibration_choice"]
    assert SKIP_LINE not in printed


def test_step_17_with_model_false_does_not_call_rolling_forecasts():
    calls, printed, _, _ = run_step_17(False)
    assert names_called(calls) == ["actual_totals", "deflated_reference",
                                   "show_deflated_reference", "rule_reference",
                                   "show_rule_reference", "stack_forecasts",
                                   "evaluation_scores", "show_evaluation_scores",
                                   "paired_wape_bootstrap", "show_paired_bootstrap",
                                   "component_bias", "show_component_bias"]
    assert "block_feature_columns" not in [name for name, _, _ in calls]


def test_step_17_with_model_false_prints_the_skip_line_once_after_the_header():
    _, printed, _, _ = run_step_17(False)
    assert printed == ["Step 17: evaluation over the validation origins 33-35 (3 origins)",
                       SKIP_LINE]


def test_step_17_with_model_true_prints_only_the_header():
    _, printed, _, _ = run_step_17(True)
    assert printed == ["Step 17: evaluation over the validation origins 33-35 (3 origins)"]


def test_step_17_with_model_false_passes_none_as_rolled():
    calls, _, _, actual = run_step_17(False)
    shown = [args for name, args, _ in calls if name == "show_deflated_reference"]
    assert len(shown) == 1
    assert shown[0][0] == "<reference>"
    assert shown[0][1] is None
    assert shown[0][2] is actual


def test_step_17_with_model_true_passes_the_rolled_forecasts_on():
    calls, _, rolled, actual = run_step_17(True)
    by_name = {name: (args, kwargs) for name, args, kwargs in calls}
    assert by_name["show_rolling_forecasts"][0] == (rolled, actual)
    assert by_name["show_deflated_reference"][0][1] is rolled
    args, kwargs = by_name["rolling_forecasts"]
    assert args[0] == "<table>" and list(args[1]) == [33, 34, 35]
    assert kwargs["block_columns"] == "<blocks>"
    assert callable(kwargs["on_origin_done"])


@pytest.mark.parametrize("with_model", [True, False])
def test_step_17_the_rule_and_the_references_run_alike_in_both_branches(with_model):
    calls, _, _, actual = run_step_17(with_model)
    by_name = {name: (args, kwargs) for name, args, kwargs in calls}
    assert by_name["actual_totals"][0] == ("<weekly>", "<weeks>", "<levels>", "<companies>",
                                           [33, 34, 35], 5)
    assert by_name["deflated_reference"][0] == ("<weekly>", "<weeks>", "<levels>",
                                                "<calendar>", "<companies>", [33, 34, 35], 5,
                                                "<week_zero>")
    args, kwargs = by_name["rule_reference"]
    assert args[:3] == ("<ledger>", "<rates>", [33, 34, 35])
    assert kwargs == {"companies": "<companies>", "new_cash": "<new_cash>",
                      "refunds": "<refunds>", "unidentified": "<unidentified>",
                      "levels": "<levels>", "calendar": "<calendar>",
                      "week_zero": "<week_zero>"}
    assert by_name["show_rule_reference"][0] == ("<rule>", actual)


def test_step_17_the_progress_callback_prints_one_line():
    calls, printed, _, _ = run_step_17(True)
    progress = [kwargs for name, _, kwargs in calls if name == "rolling_forecasts"][0]
    progress["on_origin_done"](1, 3, 33, 95.0)
    assert printed[-1] == "  ... origin 33: 1 of 3 done (95 s), about 0 min left"



# Step 17, U14.5: every method's forecasts stacked, scored against the actual cash, and the
# scores' printout, right after the rule's printout. Wiring on main()'s syntax tree and by
# running step 17's block with fakes (run_step_17 above); the printout on a hand-made scores
# table, in a separate process.

EVAL_STACK_STATEMENT = "forecasts = stack_forecasts(rolled, reference, rule)"
EVAL_SCORES_STATEMENT = ("scores = evaluation_scores(forecasts, rolled_actual, "
                    "settings.receivables.horizons)")
EVAL_SHOW_STATEMENT = "show_evaluation_scores(scores, settings.receivables.horizons)"


def test_step_17_has_the_three_score_statements_right_before_the_bootstrap():
    # Since U14.5c the bootstrap's four statements follow the scores' three, and since U14.5d
    # the three of the bias and the payer split follow those.
    statements = [ast.unparse(node) for node in step_17_block().body]
    assert statements[-10:-7] == [EVAL_STACK_STATEMENT, EVAL_SCORES_STATEMENT,
                                  EVAL_SHOW_STATEMENT]


def test_step_17_scores_come_right_after_show_rule_reference():
    statements = [ast.unparse(node) for node in step_17_block().body]
    shown = statements.index(SHOW_RULE_STATEMENT)
    assert statements[shown + 1:shown + 4] == [EVAL_STACK_STATEMENT, EVAL_SCORES_STATEMENT,
                                               EVAL_SHOW_STATEMENT]


def test_step_17_is_still_the_last_step_after_the_scores():
    # Since U16.1 step 18 follows step 17's `if`: main() up to that `if`.
    body = main_through_step_17()
    assert isinstance(body[-1], ast.If)
    assert ast.unparse(body[-1].test) == "settings.evaluation.run"
    assert body[-1].orelse == []
    # Since U14.5d the scores' printout is eighth from last, before the bootstrap's four and
    # the three of the bias and the payer split.
    assert ast.unparse(body[-1].body[-8]) == EVAL_SHOW_STATEMENT


@pytest.mark.parametrize("name", ["stack_forecasts", "evaluation_scores",
                                  "show_evaluation_scores"])
def test_step_17_calls_each_score_function_once_in_main(name):
    assert len(calls_to(main_tree(), name)) == 1, name
    assert len(calls_to(step_17_block(), name)) == 1, name


@pytest.mark.parametrize("name", ["stack_forecasts", "evaluation_scores",
                                  "show_evaluation_scores"])
def test_step_17_score_statements_are_direct_statements_of_the_evaluation_block(name):
    # Directly in the `if settings.evaluation.run:` body: not in the with_model branch, so
    # they run with or without the model.
    block = step_17_block()
    direct = [node for node in block.body if isinstance(node, (ast.Assign, ast.Expr))
              and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
              and node.value.func.id == name]
    assert len(direct) == 1, name
    assert calls_to(with_model_if(block), name) == [], name


def test_step_17_score_calls_take_no_keywords():
    block = step_17_block()
    for name in ("stack_forecasts", "evaluation_scores", "show_evaluation_scores"):
        assert calls_to(block, name)[0].keywords == [], name


def test_step_17_scores_use_the_horizons_setting_twice():
    block = step_17_block()
    scored = calls_to(block, "evaluation_scores")[0]
    shown = calls_to(block, "show_evaluation_scores")[0]
    assert ast.unparse(scored.args[2]) == "settings.receivables.horizons"
    assert ast.unparse(shown.args[1]) == "settings.receivables.horizons"


def test_step_17_stacks_the_rolled_reference_and_rule_made_before_it():
    # rolled (None or the model's), reference and rule are all assigned before the stack.
    # Since U16.3 rolled = None sits right before step 17's `if` (main()'s own statement).
    body = main_body()
    at = index_of_if(body, "settings.evaluation.run")
    assert ast.unparse(body[at - 1]) == "rolled = None"
    block = body[at]
    statements = [ast.unparse(node) for node in block.body]
    stack = statements.index(EVAL_STACK_STATEMENT)
    for prefix in ("reference = deflated_reference(", "rule = rule_reference("):
        made = [i for i, s in enumerate(statements) if s.startswith(prefix)]
        assert len(made) == 1 and made[0] < stack, prefix


def test_step_17_scores_against_the_rolled_actual_not_step_16s_actual():
    scored = calls_to(step_17_block(), "evaluation_scores")[0]
    assert [ast.unparse(a) for a in scored.args][:2] == ["forecasts", "rolled_actual"]


@pytest.mark.parametrize("with_model", [True, False])
def test_step_17_passes_the_stack_on_to_the_scores_and_the_printout(with_model):
    calls, _, rolled, actual = run_step_17(with_model)
    by_name = {name: (args, kwargs) for name, args, kwargs in calls}
    # 1. The stack gets the model's forecasts (or None), the reference and the rule.
    expected_rolled = rolled if with_model else None
    args, kwargs = by_name["stack_forecasts"]
    assert args[0] is expected_rolled
    assert args[1:] == ("<reference>", "<rule>") and kwargs == {}
    # 2. The scores get the stack, the rolled actual and the horizons setting.
    args, kwargs = by_name["evaluation_scores"]
    # (The fake stack's result: a table of its methods, for U14.5c to read.)
    assert isinstance(args[0], dict) and "method" in args[0]
    assert args[1] is actual and args[2] == 5 and kwargs == {}
    # 3. The printout gets the scores and the horizons setting.
    assert by_name["show_evaluation_scores"] == (("<scores>", 5), {})


@pytest.mark.parametrize("with_model", [True, False])
def test_step_17_each_score_function_runs_once(with_model):
    calls, _, _, _ = run_step_17(with_model)
    names = [name for name, _, _ in calls]
    for name in ("stack_forecasts", "evaluation_scores", "show_evaluation_scores"):
        assert names.count(name) == 1, name
    # And they come in that order, right before the bootstrap's two calls (U14.5c).
    start = names.index("stack_forecasts")
    assert names[start:start + 5] == ["stack_forecasts", "evaluation_scores",
                                      "show_evaluation_scores", "paired_wape_bootstrap",
                                      "show_paired_bootstrap"]


# Step 17, the cache of the rolling forecasts (2026-11-03): inside the with_model branch the
# fit is a function (fit_every_origin), the fingerprint is made from the same arguments
# (rolling_fingerprint, without the progress callback), and load_with_cache either reads the
# forecasts or calls the fit. Wiring on main()'s syntax tree; then step 17's block run with
# fakes (run_step_17 above), and with the real load_with_cache over a tmp_path cache folder.

from collection_estimation.hurdle import cache as step17_cache  # noqa: E402

CACHED_ROLLING_NAME = "hurdle_rolling_forecasts"


def fit_every_origin_def(branch=None):
    branch = branch if branch is not None else with_model_if()
    found = [node for node in branch.body
             if isinstance(node, ast.FunctionDef) and node.name == "fit_every_origin"]
    assert len(found) == 1
    return found[0]


def test_step_17_fit_every_origin_takes_no_arguments_and_returns_the_fit():
    fit = fit_every_origin_def()
    arguments = fit.args
    assert arguments.args == [] and arguments.kwonlyargs == [] and arguments.posonlyargs == []
    assert arguments.vararg is None and arguments.kwarg is None
    assert len(fit.body) == 1 and isinstance(fit.body[0], ast.Return)
    assert ast.unparse(fit.body[0].value.func) == "rolling_forecasts"


def test_step_17_fingerprint_and_fit_get_identical_arguments_but_the_callback():
    # 1. The two calls, both inside the with_model branch.
    branch = with_model_if()
    fit_call = calls_to(branch, "rolling_forecasts")[0]
    print_calls = calls_to(branch, "rolling_fingerprint")
    assert len(print_calls) == 1
    print_call = print_calls[0]
    # 2. The same positional arguments, and the same keywords apart from on_origin_done.
    assert ([ast.unparse(a) for a in print_call.args]
            == [ast.unparse(a) for a in fit_call.args] == ["table", "origins", "settings"])
    fit_keywords = keywords_of(fit_call)
    assert fit_keywords.pop("on_origin_done") == "progress"
    assert keywords_of(print_call) == fit_keywords
    # 3. In the same order too (readability: one can be checked against the other).
    assert [k.arg for k in print_call.keywords] == [k.arg for k in fit_call.keywords
                                                    if k.arg != "on_origin_done"]


def test_step_17_fingerprint_is_assigned_once_before_load_with_cache():
    branch = with_model_if()
    fingerprints = [i for i, node in enumerate(branch.body) if isinstance(node, ast.Assign)
                    and [ast.unparse(t) for t in node.targets] == ["fingerprint"]]
    loads = [i for i, node in enumerate(branch.body) if isinstance(node, ast.Assign)
             and calls_to(node, "load_with_cache")]
    assert len(fingerprints) == 1 and len(loads) == 1
    assert fingerprints[0] < loads[0]
    assert fit_every_origin_def(branch) in branch.body[:loads[0]]


def test_step_17_load_with_cache_gets_the_name_fingerprint_fit_and_cache_folder():
    branch = with_model_if()
    loads = calls_to(branch, "load_with_cache")
    assert len(loads) == 1
    load = loads[0]
    assert load.keywords == []
    assert [ast.unparse(a) for a in load.args] == [
        repr(CACHED_ROLLING_NAME), "fingerprint", "fit_every_origin",
        "settings.data.cache_folder"]
    # The fit function itself is passed (not called): load_with_cache calls it on a miss.
    assert isinstance(load.args[2], ast.Name)
    # Its result is the forecasts the rest of step 17 uses.
    assigned = [node for node in branch.body if isinstance(node, ast.Assign)
                and node.value is load]
    assert len(assigned) == 1 and [ast.unparse(t) for t in assigned[0].targets] == ["rolled"]


def test_step_17_block_columns_are_made_once_and_used_by_both():
    # Made once, in prepare_inputs (inputs.block_columns), unpacked at the top of main(), and
    # passed as they are to both the fingerprint and the fit.
    branch = with_model_if()
    tree = main_tree()
    assert calls_to(tree, "block_feature_columns") == []
    assert assignments_to(branch, "block_columns") == []
    made = assignments_to(tree, "block_columns")
    assert len(made) == 1 and any(node is made[0] for node in tree.body)
    assert ast.unparse(made[0].value) == "inputs.block_columns"
    assert len(calls_to(prepare_tree(), "block_feature_columns")) == 1
    assert "block_columns=block_feature_columns(history, calendar, owed)" in prepare_source()
    for name in ("rolling_forecasts", "rolling_fingerprint"):
        assert keywords_of(calls_to(branch, name)[0])["block_columns"] == "block_columns", name


def test_step_17_the_cache_calls_appear_once_in_main_and_only_in_the_branch():
    tree = main_tree()
    block = step_17_block()
    branch = with_model_if(block)
    outside = [node for node in block.body if node is not branch]
    for name in ("rolling_fingerprint", "load_with_cache"):
        assert len(calls_to(tree, name)) == 1, name
        assert calls_in(outside, name) == [], name
        assert calls_in(branch.orelse, name) == [], name
    assert CACHED_ROLLING_NAME in ast.unparse(branch)
    assert ast.unparse(tree).count(CACHED_ROLLING_NAME) == 1


# Running step 17's block with the fakes of run_step_17 (load_with_cache always misses).

def by_name_of(calls):
    found = {}
    for name, args, kwargs in calls:
        assert name not in found, f"{name} called twice"
        found[name] = (args, kwargs)
    return found


def test_step_17_run_fingerprint_and_fit_receive_the_same_arguments():
    calls, _, _, _ = run_step_17(True)
    by_name = by_name_of(calls)
    fit_args, fit_kwargs = by_name["rolling_forecasts"]
    print_args, print_kwargs = by_name["rolling_fingerprint"]
    # The very same objects, positionally and by keyword, apart from the callback.
    assert len(print_args) == len(fit_args) == 3
    for found, wanted in zip(print_args, fit_args):
        assert found is wanted
    fit_kwargs = dict(fit_kwargs)
    assert callable(fit_kwargs.pop("on_origin_done"))
    assert set(print_kwargs) == set(fit_kwargs)
    for name in fit_kwargs:
        assert print_kwargs[name] is fit_kwargs[name], name
    assert print_kwargs["block_columns"] == "<blocks>"


def test_step_17_run_load_with_cache_gets_name_fingerprint_fit_and_folder():
    calls, _, rolled, _ = run_step_17(True, cache_folder="<the cache folder>")
    args, kwargs = by_name_of(calls)["load_with_cache"]
    assert kwargs == {}
    assert args[0] == CACHED_ROLLING_NAME
    assert args[1] is STEP_17_FINGERPRINT          # what rolling_fingerprint returned
    assert callable(args[2])
    assert args[3] == "<the cache folder>"


def test_step_17_run_the_fit_function_is_what_calls_rolling_forecasts():
    # 1. In the run, rolling_forecasts is called only from inside load_with_cache.
    calls, _, rolled, _ = run_step_17(True)
    names = [name for name, _, _ in calls]
    assert names.index("rolling_forecasts") == names.index("load_with_cache") + 1
    # 2. Calling the fit function again calls rolling_forecasts again, with the same inputs.
    fit = by_name_of(calls)["load_with_cache"][0][2]
    before = len(calls)
    assert fit() is rolled
    assert [name for name, _, _ in calls[before:]] == ["rolling_forecasts"]


def test_step_17_run_the_result_of_load_with_cache_is_the_rolled_forecasts():
    cached = step_17_rolled()

    def always_hit(*args, **kwargs):
        return cached

    calls, _, _, actual = run_step_17(True, load_with_cache=always_hit)
    by_name = by_name_of(calls)
    assert "rolling_forecasts" not in by_name
    assert by_name["show_rolling_forecasts"][0] == (cached, actual)
    assert by_name["show_deflated_reference"][0][1] is cached
    assert by_name["stack_forecasts"][0][0] is cached


def test_step_17_run_with_model_false_calls_none_of_the_cache():
    loaded = []
    calls, _, _, _ = run_step_17(False, load_with_cache=lambda *a, **k: loaded.append(a))
    names = [name for name, _, _ in calls]
    for name in ("rolling_fingerprint", "load_with_cache", "rolling_forecasts",
                 "block_feature_columns"):
        assert name not in names, name
    assert loaded == []


# Running step 17's block with the REAL load_with_cache over a tmp_path cache folder (the
# fingerprint is still the fake's fixed STEP_17_FINGERPRINT).

def real_cache_run(cache_folder, rolled_result):
    return run_step_17(True, cache_folder=cache_folder,
                       load_with_cache=step17_cache.load_with_cache,
                       rolled_result=rolled_result)


def test_step_17_with_a_prefilled_cache_does_not_call_rolling_forecasts(tmp_path, capsys):
    # 1. A cache written under the fingerprint step 17 will make.
    cached = step_17_rolled(forecasts=[1.5, 2.5])
    step17_cache.load_with_cache(CACHED_ROLLING_NAME, STEP_17_FINGERPRINT, lambda: cached,
                                 tmp_path)
    capsys.readouterr()
    # 2. Step 17 reads it: no fit, and the cached forecasts go on to the printout and stack.
    calls, _, _, actual = real_cache_run(tmp_path, rolled_result={"never": "used"})
    by_name = by_name_of(calls)
    assert "rolling_forecasts" not in by_name
    assert by_name["show_rolling_forecasts"][0][0] == cached
    assert by_name["stack_forecasts"][0][0] == cached
    # 3. The real cache said where it read from.
    pickle_file = tmp_path / f"{CACHED_ROLLING_NAME}.pkl"
    assert capsys.readouterr().out == (f"{CACHED_ROLLING_NAME}: read from the cache; delete "
                                       f"{pickle_file} to make it again.\n")


def test_step_17_with_an_empty_cache_fits_once_then_reads(tmp_path):
    # 1. No cache: the fit runs once and the cache files are written.
    fitted = step_17_rolled(forecasts=[3.0])
    calls, _, _, _ = real_cache_run(tmp_path, rolled_result=fitted)
    assert [name for name, _, _ in calls].count("rolling_forecasts") == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"{CACHED_ROLLING_NAME}.pkl", f"{CACHED_ROLLING_NAME}_fingerprint.json"]
    # 2. The next run of step 17 reads the cache: no fit, the same forecasts.
    calls, _, _, _ = real_cache_run(tmp_path, rolled_result={"other": 1})
    by_name = by_name_of(calls)
    assert "rolling_forecasts" not in by_name
    assert by_name["show_rolling_forecasts"][0][0] == fitted


def test_step_17_with_a_cache_under_another_fingerprint_fits_again(tmp_path):
    # A cache made from other inputs (another fingerprint) is not used.
    step17_cache.load_with_cache(CACHED_ROLLING_NAME, ["older inputs"], lambda: {"old": 1},
                                 tmp_path)
    fitted = step_17_rolled(forecasts=[4.0])
    calls, _, _, _ = real_cache_run(tmp_path, rolled_result=fitted)
    by_name = by_name_of(calls)
    assert "rolling_forecasts" in by_name
    assert by_name["show_rolling_forecasts"][0][0] == fitted


def test_step_17_the_cache_goes_to_the_settings_cache_folder(tmp_path):
    folder = tmp_path / "nested" / "cache"
    real_cache_run(folder, rolled_result=step_17_rolled(forecasts=[1.0]))
    assert (folder / f"{CACHED_ROLLING_NAME}.pkl").is_file()
    assert not (tmp_path / f"{CACHED_ROLLING_NAME}.pkl").exists()



# Step 17, U14.5c: the paired bootstrap of WAPE, right after the scores' printout. The subject
# is the hurdle model if it ran, else deflated_trend; it is paired with every other method in
# the stack, in the stack's order. Wiring on main()'s syntax tree, then step 17's block run with
# fakes (run_step_17 above), then show_paired_bootstrap's exact printout on a hand-made table,
# in a separate process.

BOOT_SUBJECT_STATEMENT = "subject = HURDLE if rolled is not None else 'deflated_trend'"
BOOT_PAIRS_STATEMENT = ("pairs = [(subject, other) for other in dict.fromkeys(forecasts['method']) "
                        "if other != subject]")
BOOT_CALL_STATEMENT = ("bootstrap = paired_wape_bootstrap(forecasts, rolled_actual, pairs, "
                       "settings.receivables.horizons, settings.evaluation.bootstrap_samples, "
                       "settings.evaluation.bootstrap_seed)")
BOOT_SHOW_STATEMENT = "show_paired_bootstrap(bootstrap)"
BOOT_STATEMENTS = [BOOT_SUBJECT_STATEMENT, BOOT_PAIRS_STATEMENT, BOOT_CALL_STATEMENT,
                   BOOT_SHOW_STATEMENT]
# U14.5d: the three statements of the bias and the payer split follow the bootstrap's four.
BIAS_CALL_STATEMENT = ("bias = component_bias(rolled, rule, rolled_actual, new_cash, refunds, "
                       "unidentified, levels, settings.receivables.horizons)")
BIAS_SHOW_STATEMENT = "show_component_bias(bias, settings.receivables.horizons)"


def test_step_17_has_the_four_bootstrap_statements_right_before_the_bias():
    # Since U14.5d the three of the bias and the payer split follow them.
    statements = [ast.unparse(node) for node in step_17_block().body]
    assert statements[-7:-3] == BOOT_STATEMENTS


def test_step_17_bootstrap_comes_right_after_show_evaluation_scores():
    statements = [ast.unparse(node) for node in step_17_block().body]
    shown = statements.index(EVAL_SHOW_STATEMENT)
    assert statements[shown + 1:shown + 5] == BOOT_STATEMENTS


def test_step_17_is_still_the_last_step_after_the_bootstrap():
    # Since U16.1 step 18 follows step 17's `if`: main() up to that `if`.
    body = main_through_step_17()
    assert isinstance(body[-1], ast.If)
    assert ast.unparse(body[-1].test) == "settings.evaluation.run"
    assert body[-1].orelse == []
    # Since U14.5d the bootstrap's printout is fourth from last.
    assert ast.unparse(body[-1].body[-4]) == BOOT_SHOW_STATEMENT


@pytest.mark.parametrize("name", ["paired_wape_bootstrap", "show_paired_bootstrap"])
def test_step_17_calls_each_bootstrap_function_once_in_main(name):
    assert len(calls_to(main_tree(), name)) == 1, name
    assert len(calls_to(step_17_block(), name)) == 1, name


@pytest.mark.parametrize("name", ["paired_wape_bootstrap", "show_paired_bootstrap"])
def test_step_17_bootstrap_statements_are_outside_the_with_model_branch(name):
    # Directly in the `if settings.evaluation.run:` body: they run with or without the model.
    block = step_17_block()
    direct = [node for node in block.body if isinstance(node, (ast.Assign, ast.Expr))
              and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
              and node.value.func.id == name]
    assert len(direct) == 1, name
    assert calls_to(with_model_if(block), name) == [], name


def test_step_17_bootstrap_calls_take_no_keywords_and_the_settings_by_name():
    block = step_17_block()
    call = calls_to(block, "paired_wape_bootstrap")[0]
    assert call.keywords == []
    assert [ast.unparse(a) for a in call.args] == [
        "forecasts", "rolled_actual", "pairs", "settings.receivables.horizons",
        "settings.evaluation.bootstrap_samples", "settings.evaluation.bootstrap_seed"]
    # Not [joint_diagnostic]'s bootstrap_samples, a different setting.
    assert "joint_diagnostic" not in ast.unparse(call)
    shown = calls_to(block, "show_paired_bootstrap")[0]
    assert shown.keywords == [] and [ast.unparse(a) for a in shown.args] == ["bootstrap"]


def test_step_17_subject_pairs_and_bootstrap_are_assigned_once():
    for name in ("subject", "pairs", "bootstrap"):
        assert len(assignments_to(step_17_block(), name)) == 1, name
    assert len(assignments_to(main_tree(), "subject")) == 1


def test_step_17_the_subject_uses_the_hurdle_constant_not_a_literal():
    subject = assignments_to(step_17_block(), "subject")[0].value
    assert isinstance(subject, ast.IfExp)
    assert ast.unparse(subject.body) == "HURDLE"
    assert ast.unparse(subject.test) == "rolled is not None"
    assert ast.literal_eval(subject.orelse) == "deflated_trend"


# Running step 17's block with the fakes of run_step_17.

def test_step_17_run_with_the_model_pairs_the_hurdle_with_every_other_method():
    calls, _, _, actual = run_step_17(True)
    by_name = by_name_of(calls)
    args, kwargs = by_name["paired_wape_bootstrap"]
    assert kwargs == {}
    # The stack's methods in order: hurdle, then the four others, each once.
    assert args[2] == [("hurdle", "deflated"), ("hurdle", "deflated_nolevel"),
                       ("hurdle", "deflated_trend"), ("hurdle", "rule")]


def test_step_17_run_without_the_model_pairs_deflated_trend_with_the_others():
    calls, _, _, _ = run_step_17(False)
    args, _ = by_name_of(calls)["paired_wape_bootstrap"]
    assert args[2] == [("deflated_trend", "deflated"), ("deflated_trend", "deflated_nolevel"),
                       ("deflated_trend", "rule")]


@pytest.mark.parametrize("with_model", [True, False])
def test_step_17_run_bootstrap_gets_the_stack_the_rolled_actual_and_the_settings(with_model):
    calls, _, _, actual = run_step_17(with_model)
    by_name = by_name_of(calls)
    args, _ = by_name["paired_wape_bootstrap"]
    # 1. The very table the scores were made from, and the rolled actual cash.
    assert args[0] is by_name["evaluation_scores"][0][0]
    assert args[1] is actual
    # 2. horizons, then [evaluation]'s samples and seed.
    assert args[3:] == (5, STEP_17_SAMPLES, STEP_17_SEED)
    assert len(args) == 6


@pytest.mark.parametrize("with_model", [True, False])
def test_step_17_run_the_printout_gets_the_bootstrap_table(with_model):
    calls, _, _, _ = run_step_17(with_model)
    assert by_name_of(calls)["show_paired_bootstrap"] == (("<bootstrap>",), {})


def test_step_17_run_a_cached_model_is_still_the_subject():
    # rolled from the cache (not fitted) is not None: the hurdle is the subject.
    calls, _, _, _ = run_step_17(True, load_with_cache=lambda *args, **kwargs: step_17_rolled())
    pairs = by_name_of(calls)["paired_wape_bootstrap"][0][2]
    assert {method for method, _ in pairs} == {"hurdle"}


@pytest.mark.parametrize("with_model", [True, False])
def test_step_17_run_the_subject_is_never_paired_with_itself(with_model):
    calls, _, _, _ = run_step_17(with_model)
    pairs = by_name_of(calls)["paired_wape_bootstrap"][0][2]
    assert all(method != against for method, against in pairs)
    # Every other stacked method exactly once.
    assert len({against for _, against in pairs}) == len(pairs)


def test_step_17_run_the_bootstrap_settings_follow_the_settings_object():
    # 1. Run step 17's block with other bootstrap settings in its fake settings.
    calls, printed = [], []

    def recorder(name, result):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result
        return call

    namespace = {name: f"<{name}>" for name in STEP_17_NAMES}
    namespace.update(
        settings=SimpleNamespace(
            evaluation=SimpleNamespace(run=True, with_model=False, validation_origins=(40,),
                                       bootstrap_samples=13, bootstrap_seed=0),
            receivables=SimpleNamespace(horizons=3), data=SimpleNamespace(cache_folder="c")),
        time=SimpleNamespace(perf_counter=lambda: 0.0),
        print=lambda *args, **kwargs: printed.append(args),
        HURDLE="hurdle",
        stack_forecasts=lambda *args: {"method": ["deflated", "deflated_trend", "rule"]},
        **{name: recorder(name, f"<{name} result>") for name in (
            "actual_totals", "deflated_reference", "show_deflated_reference", "rule_reference",
            "show_rule_reference", "evaluation_scores", "show_evaluation_scores",
            "paired_wape_bootstrap", "show_paired_bootstrap", "component_bias",
            "show_component_bias")})
    # Since U16.3 rolled = None is main()'s statement right before step 17's `if`: run both.
    main = main_body()
    at = index_of_if(main, "settings.evaluation.run")
    assert ast.unparse(main[at - 1]) == "rolled = None"
    module = ast.Module(body=main[at - 1:at + 1], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<step 17>", "exec"), namespace)
    # 2. The bootstrap got horizons 3, 13 samples and seed 0.
    args, _ = by_name_of(calls)["paired_wape_bootstrap"]
    assert args[2] == [("deflated_trend", "deflated"), ("deflated_trend", "rule")]
    assert args[3:] == (3, 13, 0)


# Step 17, U14.5d: where the bias comes from, right after the bootstrap's printout. Wiring on
# main()'s syntax tree, then step 17's block run with the fakes of run_step_17, then the exact
# printouts of show_component_bias and show_payer_split on hand-made tables, in a separate
# process.

BIAS_PAYER_GUARD = "rolled is not None"
BIAS_PAYER_STATEMENTS = [
    ("split = payer_split(rolled.customers, weekly, weeks, levels, "
     "settings.receivables.horizons)"),
    "show_payer_split(split, settings.receivables.horizons)",
    # U14.6: the balance check, right after the payer split, under the same guard.
    ("coherence = balance_coherence(rolled.customers, table, weekly, weeks, levels, "
     "settings.receivables.horizons, settings.evaluation.coherence_factor)"),
    "show_balance_coherence(coherence, settings.evaluation.coherence_factor)",
    # U15: the reliability and the choice, last, under the same guard.
    ("reliability = calibration_reliability(rolled.customers, weekly, weeks, levels, "
     "settings.diagnostic.calibration_groups)"),
    "show_calibration_reliability(reliability)",
    "choice = choose_calibration(rolled, weekly, weeks, levels, rolled_actual, settings)",
    "show_calibration_choice(choice, settings.calibration.min_gain)"]


def test_step_17_ends_with_the_bias_and_the_guarded_payer_split():
    # 1. The last three statements of the block.
    body = step_17_block().body
    assert ast.unparse(body[-3]) == BIAS_CALL_STATEMENT
    assert ast.unparse(body[-2]) == BIAS_SHOW_STATEMENT
    assert isinstance(body[-1], ast.If)
    # 2. The guard holds exactly the payer split and its printout, with no else.
    assert ast.unparse(body[-1].test) == BIAS_PAYER_GUARD
    assert [ast.unparse(node) for node in body[-1].body] == BIAS_PAYER_STATEMENTS
    assert body[-1].orelse == []


def test_step_17_the_bias_comes_right_after_the_bootstraps_printout():
    statements = [ast.unparse(node) for node in step_17_block().body]
    shown = statements.index(BOOT_SHOW_STATEMENT)
    assert statements[shown + 1:shown + 3] == [BIAS_CALL_STATEMENT, BIAS_SHOW_STATEMENT]


def test_step_17_is_still_the_last_step_after_the_bias():
    # Since U16.1 step 18 follows step 17's `if`: main() up to that `if`.
    body = main_through_step_17()
    assert isinstance(body[-1], ast.If)
    assert ast.unparse(body[-1].test) == "settings.evaluation.run"
    assert body[-1].orelse == []
    assert ast.unparse(body[-1].body[-1].test) == BIAS_PAYER_GUARD


@pytest.mark.parametrize("name", ["component_bias", "show_component_bias", "payer_split",
                                  "show_payer_split"])
def test_step_17_calls_each_bias_function_once_in_main(name):
    assert len(calls_to(main_tree(), name)) == 1, name
    assert len(calls_to(step_17_block(), name)) == 1, name


@pytest.mark.parametrize("name", ["component_bias", "show_component_bias"])
def test_step_17_the_bias_is_outside_the_with_model_branch_and_the_guard(name):
    # 1. Not in the one branch nor in the other.
    block = step_17_block()
    assert calls_to(with_model_if(block), name) == []
    assert calls_to(block.body[-1], name) == []
    # 2. So a direct statement of the block.
    direct = [node for node in block.body if not isinstance(node, ast.If)
              and calls_to(node, name)]
    assert len(direct) == 1


@pytest.mark.parametrize("name", ["payer_split", "show_payer_split"])
def test_step_17_the_payer_split_is_only_inside_the_guard(name):
    block = step_17_block()
    assert calls_to(with_model_if(block), name) == []
    assert len(calls_to(block.body[-1], name)) == 1


def test_step_17_bias_calls_take_no_keywords():
    block = step_17_block()
    for name in ("component_bias", "show_component_bias", "payer_split", "show_payer_split"):
        assert calls_to(block, name)[0].keywords == [], name


def test_step_17_bias_and_split_are_assigned_once():
    assert len(assignments_to(step_17_block(), "bias")) == 1
    assert len(assignments_to(step_17_block(), "split")) == 1


# Running step 17's block with the fakes of run_step_17.

def test_step_17_run_component_bias_gets_its_arguments_in_order():
    # rolled, rule, the rolled actual, A, R, G, the levels, the settings' horizons.
    calls, _, rolled, actual = run_step_17(True)
    args, kwargs = by_name_of(calls)["component_bias"]
    assert kwargs == {}
    assert len(args) == 8
    assert args[0] is rolled
    assert args[1] == "<rule>"
    assert args[2] is actual
    assert args[3:] == ("<new_cash>", "<refunds>", "<unidentified>", "<levels>", 5)


def test_step_17_run_payer_split_gets_its_arguments_in_order():
    # The customers of the rolled forecasts, the weekly amounts, the calendar, the levels.
    calls, _, _, _ = run_step_17(True)
    args, kwargs = by_name_of(calls)["payer_split"]
    assert kwargs == {}
    assert args == ("<rolled customers>", "<weekly>", "<weeks>", "<levels>", 5)


def test_step_17_run_the_printouts_get_the_tables_and_the_horizons():
    calls, _, _, _ = run_step_17(True)
    by_name = by_name_of(calls)
    assert by_name["show_component_bias"] == (("<bias>", 5), {})
    assert by_name["show_payer_split"] == (("<split>", 5), {})


def test_step_17_run_the_horizons_follow_the_settings():
    # 1. Another [receivables] horizons.
    calls, _, _, _ = run_step_17(True, horizons=3)
    by_name = by_name_of(calls)
    # 2. All four calls get it.
    assert by_name["component_bias"][0][-1] == 3
    assert by_name["show_component_bias"][0][-1] == 3
    assert by_name["payer_split"][0][-1] == 3
    assert by_name["show_payer_split"][0][-1] == 3


def test_step_17_run_without_the_model_the_bias_still_runs_and_the_payer_split_does_not():
    # 1. with_model false.
    calls, _, _, actual = run_step_17(False)
    by_name = by_name_of(calls)
    names = [name for name, _, _ in calls]
    # 2. component_bias gets None for rolled and the rest as ever; its printout follows.
    args, _ = by_name["component_bias"]
    assert args[0] is None
    assert args[1] == "<rule>"
    assert args[2] is actual
    assert args[3:] == ("<new_cash>", "<refunds>", "<unidentified>", "<levels>", 5)
    assert by_name["show_component_bias"] == (("<bias>", 5), {})
    # 3. No payer split: it needs the model's customer rows.
    assert "payer_split" not in names
    assert "show_payer_split" not in names
    assert names[-2:] == ["component_bias", "show_component_bias"]


def test_step_17_run_the_guard_reads_rolled_not_the_setting():
    # with_model true, but load_with_cache gives None: `rolled` decides.
    calls, _, _, _ = run_step_17(True, load_with_cache=lambda *args, **kwargs: None)
    names = [name for name, _, _ in calls]
    assert "component_bias" in names
    assert "payer_split" not in names


def test_step_17_run_a_cached_model_gives_the_payer_split_its_customers():
    # From the cache, not fitted: the customers of what the cache gave.
    cached = SimpleNamespace(customers="<cached customers>")
    calls, _, _, _ = run_step_17(True, load_with_cache=lambda *args, **kwargs: cached)
    by_name = by_name_of(calls)
    assert by_name["payer_split"][0][0] == "<cached customers>"
    assert by_name["component_bias"][0][0] is cached


@pytest.mark.parametrize("with_model", [True, False])
def test_step_17_run_the_bias_comes_after_the_bootstraps_printout(with_model):
    calls, _, _, _ = run_step_17(with_model)
    names = [name for name, _, _ in calls]
    start = names.index("show_paired_bootstrap")
    assert names[start + 1:start + 3] == ["component_bias", "show_component_bias"]


def test_step_17_run_the_payer_split_comes_after_the_bias_with_its_printout():
    # U14.6's balance check and U15's calibration now follow it; see their tests.
    calls, _, _, _ = run_step_17(True)
    names = [name for name, _, _ in calls]
    start = names.index("show_component_bias")
    assert names[start + 1:start + 3] == ["payer_split", "show_payer_split"]


# Step 17, U14.6: the balance check, right after the payer split, only when the model ran. Wiring
# on main()'s syntax tree, then step 17's block run with the fakes of run_step_17, then the exact
# printout of show_balance_coherence on a hand-made table, in a separate process.

COHERENCE_CALL_STATEMENT = BIAS_PAYER_STATEMENTS[2]
COHERENCE_SHOW_STATEMENT = BIAS_PAYER_STATEMENTS[3]


def test_step_17_the_guard_has_the_balance_check_right_after_the_payer_split():
    # U15's calibration now ends the guard; see its tests.
    guarded = [ast.unparse(node) for node in step_17_block().body[-1].body]
    start = guarded.index(BIAS_PAYER_STATEMENTS[1])
    assert guarded[start + 1:start + 3] == [COHERENCE_CALL_STATEMENT, COHERENCE_SHOW_STATEMENT]


@pytest.mark.parametrize("name", ["balance_coherence", "show_balance_coherence"])
def test_step_17_calls_each_coherence_function_once_in_main(name):
    assert len(calls_to(main_tree(), name)) == 1, name
    assert len(calls_to(step_17_block(), name)) == 1, name


@pytest.mark.parametrize("name", ["balance_coherence", "show_balance_coherence"])
def test_step_17_the_balance_check_is_only_inside_the_guard(name):
    block = step_17_block()
    assert ast.unparse(block.body[-1].test) == "rolled is not None"
    assert len(calls_to(block.body[-1], name)) == 1
    assert calls_to(with_model_if(block), name) == []


def test_step_17_the_balance_check_calls_take_no_keywords():
    block = step_17_block()
    for name in ("balance_coherence", "show_balance_coherence"):
        assert calls_to(block, name)[0].keywords == [], name


def test_step_17_coherence_is_assigned_once():
    assert len(assignments_to(step_17_block(), "coherence")) == 1


def test_step_17_run_the_balance_check_gets_its_arguments_in_order():
    # The customers of the rolled forecasts, the training table, the weekly amounts, the
    # calendar, the levels, the horizons, the factor.
    calls, _, _, _ = run_step_17(True)
    args, kwargs = by_name_of(calls)["balance_coherence"]
    assert kwargs == {}
    assert args == ("<rolled customers>", "<table>", "<weekly>", "<weeks>", "<levels>", 5,
                    STEP_17_FACTOR)


def test_step_17_run_the_printout_gets_the_table_and_the_factor():
    calls, _, _, _ = run_step_17(True)
    assert by_name_of(calls)["show_balance_coherence"] == (("<coherence>", STEP_17_FACTOR), {})


def test_step_17_run_the_balance_check_comes_right_after_the_payer_split():
    # U15's calibration now comes last; see its tests.
    calls, _, _, _ = run_step_17(True)
    names = [name for name, _, _ in calls]
    start = names.index("payer_split")
    assert names[start:start + 4] == ["payer_split", "show_payer_split", "balance_coherence",
                                      "show_balance_coherence"]


def test_step_17_run_the_horizons_reach_the_balance_check():
    calls, _, _, _ = run_step_17(True, horizons=3)
    assert by_name_of(calls)["balance_coherence"][0][5] == 3


@pytest.mark.parametrize("name", ["balance_coherence", "show_balance_coherence"])
def test_step_17_run_without_the_model_there_is_no_balance_check(name):
    calls, _, _, _ = run_step_17(False)
    assert name not in [called for called, _, _ in calls]


@pytest.mark.parametrize("name", ["balance_coherence", "show_balance_coherence"])
def test_step_17_run_the_balance_check_reads_rolled_not_the_setting(name):
    # with_model true, but load_with_cache gives None: `rolled` decides.
    calls, _, _, _ = run_step_17(True, load_with_cache=lambda *args, **kwargs: None)
    assert name not in [called for called, _, _ in calls]


def test_step_17_run_a_cached_model_gives_the_balance_check_its_customers():
    cached = SimpleNamespace(customers="<cached customers>")
    calls, _, _, _ = run_step_17(True, load_with_cache=lambda *args, **kwargs: cached)
    assert by_name_of(calls)["balance_coherence"][0][0] == "<cached customers>"


# Step 17, U15: the reliability and the choice of corrections, after the balance check, only when
# the model ran. Wiring on main()'s syntax tree, then step 17's block run with the fakes of
# run_step_17, then the exact printouts, in a separate process.

CALIBRATION_STATEMENTS = BIAS_PAYER_STATEMENTS[4:]
CALIBRATION_NAMES = ["calibration_reliability", "show_calibration_reliability",
                     "choose_calibration", "show_calibration_choice"]


def test_step_17_the_guard_ends_with_the_calibration_right_after_the_balance_check():
    guarded = [ast.unparse(node) for node in step_17_block().body[-1].body]
    assert guarded[-4:] == CALIBRATION_STATEMENTS
    assert guarded[-6:-4] == [COHERENCE_CALL_STATEMENT, COHERENCE_SHOW_STATEMENT]


@pytest.mark.parametrize("name", CALIBRATION_NAMES)
def test_step_17_calls_each_calibration_function_once_in_main(name):
    assert len(calls_to(main_tree(), name)) == 1, name
    assert len(calls_to(step_17_block(), name)) == 1, name


@pytest.mark.parametrize("name", CALIBRATION_NAMES)
def test_step_17_the_calibration_is_only_inside_the_guard(name):
    block = step_17_block()
    assert ast.unparse(block.body[-1].test) == "rolled is not None"
    assert len(calls_to(block.body[-1], name)) == 1
    assert calls_to(with_model_if(block), name) == []


def test_step_17_the_calibration_calls_take_no_keywords():
    block = step_17_block()
    for name in CALIBRATION_NAMES:
        assert calls_to(block, name)[0].keywords == [], name


@pytest.mark.parametrize("name", ["reliability", "choice"])
def test_step_17_the_calibration_results_are_assigned_once(name):
    assert len(assignments_to(step_17_block(), name)) == 1


def test_step_17_applies_no_correction_to_the_forecasts_it_learned_from():
    # The corrections are not applied in step 17: scored on the weeks it learned from, the
    # chosen pair would look too good.
    assert calls_to(main_tree(), "corrected_forecast") == []
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = [alias.name for node in module.body if isinstance(node, ast.ImportFrom)
             for alias in node.names]
    assert "corrected_forecast" not in names


def test_step_17_run_the_calibration_comes_last_right_after_the_balance_check():
    calls, _, _, _ = run_step_17(True)
    assert [name for name, _, _ in calls][-6:] == ["balance_coherence", "show_balance_coherence",
                                                   *CALIBRATION_NAMES]


def test_step_17_run_the_reliability_gets_its_arguments_in_order():
    # The customers of the rolled forecasts, the weekly amounts, the calendar, the levels, the
    # groups of [diagnostic].
    calls, _, _, _ = run_step_17(True)
    assert by_name_of(calls)["calibration_reliability"] == (
        ("<rolled customers>", "<weekly>", "<weeks>", "<levels>", STEP_17_GROUPS), {})


def test_step_17_run_the_choice_gets_the_rolled_forecasts_their_actual_and_the_settings():
    calls, _, rolled, actual = run_step_17(True)
    args, kwargs = by_name_of(calls)["choose_calibration"]
    assert kwargs == {}
    assert len(args) == 6
    assert args[0] is rolled
    assert args[1:4] == ("<weekly>", "<weeks>", "<levels>")
    assert args[4] is actual
    assert args[5].calibration.min_gain == STEP_17_MIN_GAIN


def test_step_17_run_the_printouts_get_the_results():
    calls, _, _, _ = run_step_17(True)
    by_name = by_name_of(calls)
    assert by_name["show_calibration_reliability"] == (("<reliability>",), {})
    assert by_name["show_calibration_choice"] == (("<choice>", STEP_17_MIN_GAIN), {})


@pytest.mark.parametrize("name", CALIBRATION_NAMES)
def test_step_17_run_without_the_model_there_is_no_calibration(name):
    calls, _, _, _ = run_step_17(False)
    assert name not in [called for called, _, _ in calls]


def test_step_17_run_a_cached_model_gives_the_calibration_its_forecasts():
    cached = SimpleNamespace(customers="<cached customers>")
    calls, _, _, _ = run_step_17(True, load_with_cache=lambda *args, **kwargs: cached)
    by_name = by_name_of(calls)
    assert by_name["calibration_reliability"][0][0] == "<cached customers>"
    assert by_name["choose_calibration"][0][0] is cached


# --- Step 18: one call of grid_run.run_grid behind [grid] run -------------------------------
# Step 18 (U16) moved to hurdle/grid_run.py, which run_b.py runs alone too; what run_grid does
# is tested in test_grid_run.py. Here: main()'s one statement for it, after step 17's `if`,
# with step 17's forecasts (`rolled`, None unless step 17 made them), and the names main()
# no longer calls itself.
RUN_GRID = "run_grid(settings, inputs, rolled=rolled)"
STEP_18_IF = f"if settings.grid.run:\n    {RUN_GRID}"
# The grid's functions and printouts, which only run_grid calls now.
GRID_NAMES = ("grid_candidates", "grid_predictions", "grid_gates", "grid_scores", "grid_choice",
              "grid_test", "show_grid_candidates", "show_grid_predictions", "show_grid_gates",
              "show_grid_scores", "show_grid_choice", "show_grid_test")


def test_step_18_is_one_if_on_grid_run_right_after_step_17_and_last_in_main():
    body = main_body()
    block = body[index_of_if(body, "settings.evaluation.run") + 1]
    assert ast.unparse(block) == STEP_18_IF
    assert block.orelse == []
    assert block is body[-1]


def test_step_18_switch_is_read_once_in_main():
    # [grid] run gates step 18 only; [grid] run_test is run_grid's to read.
    assert len(re.findall(r"settings\.grid\.run\b", main_source())) == 1
    assert index_of_if(main_body(), "settings.grid.run") == len(main_body()) - 1
    assert "run_test" not in ast.unparse(main_tree())


def test_main_calls_run_grid_once_and_only_inside_step_18():
    calls = calls_to(main_tree(), "run_grid")
    assert len(calls) == 1
    assert [ast.unparse(c) for c in calls_to(main_body()[-1], "run_grid")] == [RUN_GRID]
    assert [ast.unparse(a) for a in calls[0].args] == ["settings", "inputs"]
    assert keywords_of(calls[0]) == {"rolled": "rolled"}
    # main() does not use what run_grid returns: the printouts are run_grid's.
    assert isinstance(main_body()[-1].body[0], ast.Expr)


def test_main_calls_none_of_the_grid_functions_or_printouts_itself():
    tree = main_tree()
    for name in GRID_NAMES:
        assert calls_to(tree, name) == [], name
    for name in ("fit_progress", "grid_origins", "grid_actual", "grid_table", "test_actual"):
        assert name not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}, name


STEP_18_RUN = """
from types import SimpleNamespace
import ast
import inspect
import textwrap
import run_hurdle_pipeline as script

body = ast.parse(textwrap.dedent(inspect.getsource(script.main))).body[0].body
block = body[-1]
assert ast.unparse(block.test) == "settings.grid.run"
calls = []
def run_grid(*args, **kwargs):
    calls.append((args, kwargs))
    return "<choice>", "<test>"
for run in (True, False):
    settings = SimpleNamespace(grid=SimpleNamespace(run=run))
    namespace = {"settings": settings, "inputs": "<inputs>", "rolled": "<rolled>",
                 "run_grid": run_grid}
    exec(compile(ast.Module(body=[block], type_ignores=[]), "<step 18>", "exec"), namespace)
    print(run, len(calls), calls[-1][0][1:] if calls else None,
          calls[-1][1] if calls else None, calls[-1][0][0] is settings if calls else None)
print("ok")
"""


def test_step_18_run_calls_run_grid_with_the_settings_inputs_and_rolled_only_when_on():
    # main()'s own `if` run with a fake run_grid: switched on, one call with main()'s
    # settings, inputs and rolled; switched off, none (the count stays 1, the last call is
    # still the first run's, made with the other settings object).
    result = run_code(STEP_18_RUN)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "True 1 ('<inputs>',) {'rolled': '<rolled>'} True",
        "False 1 ('<inputs>',) {'rolled': '<rolled>'} False",
        "ok"]


def test_the_script_imports_run_grid_from_grid_run():
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = [node for node in module.body if isinstance(node, ast.ImportFrom)
               and node.module == "collection_estimation.hurdle.grid_run"]
    assert len(imports) == 1
    assert [(a.name, a.asname) for a in imports[0].names] == [("run_grid", None)]


def test_grid_is_imported_once_with_with_params_only():
    # Step 15 still calls with_params; every other name of grid is run_grid's now (it was
    # "test_grid_is_imported_once_with_its_names", eight names).
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = [node for node in module.body if isinstance(node, ast.ImportFrom)
               and node.module == "collection_estimation.hurdle.grid"]
    assert len(imports) == 1
    assert [(a.name, a.asname) for a in imports[0].names] == [("with_params", None)]


def test_step_18_the_u16_1_check_is_gone_from_the_script():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    defined = {node.name for node in module_functions(SCRIPT)}
    imported = {a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for a in node.names}
    for name in ("stage1_predictions", "stage2_predictions", "show_stage_predictions", "p_alone",
                 "m_alone", "p_seconds", "m_seconds"):
        assert name not in names | defined | imported, name


# --- The script's own functions and its printouts ---------------------------------------------

REPORT_MODULES = ("collection_estimation.hurdle.reports.preparation",
                  "collection_estimation.hurdle.reports.stages",
                  "collection_estimation.hurdle.reports.evaluation")


def script_imports():
    """{module: [names]} of the script's top-level `from ... import`s."""
    module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    found = {}
    for node in module.body:
        if isinstance(node, ast.ImportFrom):
            found.setdefault(node.module, []).extend(a.asname or a.name for a in node.names)
    return found


def shows_called():
    """The show_* names called anywhere in the script."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    return {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id.startswith("show_")}


def test_the_script_defines_main_and_its_experiments_only():
    # Every printout moved to hurdle/reports/; what stays is main() and what only it runs.
    assert [node.name for node in module_functions(SCRIPT)] == [
        "main", "run_joint_diagnostic", "topup_candidates", "show_topups", "compare_topups"]


def test_the_script_defines_no_show_function_but_show_topups():
    defined = [node.name for node in module_functions(SCRIPT) if node.name.startswith("show_")]
    assert defined == ["show_topups"]


def test_each_printout_the_script_calls_is_imported_from_a_report_module():
    imports = script_imports()
    from_reports = {name for module in REPORT_MODULES for name in imports.get(module, [])}
    assert shows_called() - {"show_topups"} == from_reports
    # Imported from those modules only, and each from one of them only.
    for module, names in imports.items():
        if module not in REPORT_MODULES:
            assert not [name for name in names if name.startswith("show_")], module
    counted = [name for module in REPORT_MODULES for name in imports.get(module, [])]
    assert len(counted) == len(set(counted))


def test_the_script_imports_from_each_report_module_the_printouts_of_its_steps():
    imports = script_imports()
    # U18.1 added show_cleared_invoice_delays, U18.2 show_expected_dues.
    assert imports["collection_estimation.hurdle.reports.preparation"] == [
        "show_cleared_invoice_delays", "show_expected_dues", "show_preparation"]
    assert imports["collection_estimation.hurdle.reports.stages"] == [
        "show_amount_reliability", "show_calibration", "show_cash_by_p_group",
        "show_diagnostic_grid", "show_diagnostic_splits", "show_diagnostic_test",
        "show_exposure", "show_gates", "show_joint_grid", "show_joint_results",
        "show_stage1_rows", "show_stage1_scores", "show_stage2_fit", "show_stage2_forecast",
        "show_stage2_scores"]
    assert imports["collection_estimation.hurdle.reports.evaluation"] == [
        "show_actual_totals", "show_balance_coherence", "show_calibration_choice",
        "show_calibration_reliability", "show_component_bias", "show_customer_forecasts",
        "show_deflated_reference", "show_evaluation_scores", "show_forecast_totals",
        "show_forecast_vs_actual", "show_paired_bootstrap", "show_payer_split",
        "show_rolling_forecasts", "show_rule_reference"]
    assert "collection_estimation.hurdle.reports.grid" not in imports


def test_every_name_the_script_imports_is_used():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for module, names in script_imports().items():
        for name in names:
            assert name in used, (module, name)


# --- U12.2, U12.3: main() runs the top-ups right after unpacking the inputs ---------------------

SHOW_TOPUPS = ("show_topups(new_cash, refunds, unidentified, levels, calendar, "
               "settings.diagnostic.origin, settings.receivables.horizons, "
               "settings.topups.window)")
COMPARE_TOPUPS_IF = ("if settings.topups.compare:\n"
                     "    compare_topups(new_cash, refunds, unidentified, levels, calendar, "
                     "settings.topups, settings.receivables.horizons)")


U18_1_PRINT = "print('U18.1: days late of every cleared invoice')"
U18_1_SHOW = "show_cleared_invoice_delays(cleared_invoice_delays(ledger))"


def unpacking_ends():
    """The index in main()'s body of the last statement unpacking the inputs."""
    body = main_body()
    found = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
             and ast.unparse(node.value).startswith(("inputs.", "(inputs."))]
    assert found == list(range(found[0], found[-1] + 1)), found
    return found[-1]


def test_main_calls_show_topups_right_after_the_inputs_are_unpacked():
    # Right after the U18.1 and U18.2 printouts, which follow the unpacking (U18.1 moved it
    # by two, U18.2 by three more).
    body = main_body()
    at = unpacking_ends()
    assert ast.unparse(body[at + 6]) == SHOW_TOPUPS
    assert len(calls_to(main_tree(), "show_topups")) == 1


def test_main_calls_compare_topups_right_after_show_topups_behind_topups_compare():
    body = main_body()
    at = unpacking_ends()
    block = body[at + 7]
    assert ast.unparse(block) == COMPARE_TOPUPS_IF
    assert block.orelse == []
    assert len(calls_to(main_tree(), "compare_topups")) == 1


def test_the_top_ups_read_main_names_unpacked_before_them():
    body = main_body()
    at = unpacking_ends()
    stored = {n.id for node in body[:at + 1] for n in ast.walk(node)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for node in body[at + 6:at + 8]:
        read = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)} - {"settings", "show_topups",
                                                     "compare_topups"}
        assert read <= stored, read - stored


def test_the_top_ups_come_before_the_demo_origin_guard():
    body = main_body()
    at = unpacking_ends()
    assert is_demo_origin_guard(body[at + 8])


def test_show_preparation_no_longer_runs_the_top_ups():
    from tests.hurdle.code_checks import HURDLE
    tree = function_tree(HURDLE / "reports" / "preparation.py", "show_preparation")
    assert calls_to(tree, "show_topups") == [] and calls_to(tree, "compare_topups") == []


def test_rolled_is_set_to_none_once_right_before_step_17():
    tree = main_tree()   # one parse: nodes are matched by identity below
    body = tree.body
    at = index_of_if(body, "settings.evaluation.run")
    assert ast.unparse(body[at - 1]) == "rolled = None"
    nones = [node for node in assignments_to(tree, "rolled")
             if ast.unparse(node.value) == "None"]
    assert len(nones) == 1 and nones[0] is body[at - 1]


def test_rolled_is_not_reset_inside_step_17():
    tree = main_tree()   # one parse: nodes are matched by identity below
    block = tree.body[index_of_if(tree.body, "settings.evaluation.run")]
    assert not any(ast.unparse(node) == "rolled = None" for node in ast.walk(block)
                   if isinstance(node, ast.Assign))
    # Its only other assignment is the cached rolling forecasts, in the with_model branch.
    others = [node for node in assignments_to(tree, "rolled")
              if ast.unparse(node.value) != "None"]
    assert len(others) == 1
    assert any(node is others[0] for node in ast.walk(with_model_if(block)))
    assert ast.unparse(others[0].value).startswith("load_with_cache('hurdle_rolling_forecasts'")


def test_rolled_is_read_by_step_18_and_step_17_only_after_it_is_set():
    # Step 18 reads rolled even when step 17 is switched off: it is set before both ifs.
    body = main_body()
    set_at = [i for i, node in enumerate(body) if ast.unparse(node) == "rolled = None"][0]
    for i, node in enumerate(body[:set_at]):
        reads = [n for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == "rolled"]
        assert reads == [], ast.unparse(node)


def test_module_docstring_lists_step_18_after_the_forecast_line():
    lines = ast.get_docstring(ast.parse(SCRIPT.read_text(encoding="utf-8"))).splitlines()
    step_lines = [line for line in lines if re.match(r"^\s+\d+\s{2}", line)]
    assert step_lines[-2] == ("   13  forecast, evaluation, calibration                    "
                              "       (U13-U15)")
    assert step_lines[-1] == ("   18  settings grid                                        "
                              "       (U16)  <- in progress")


def test_step_15_still_calls_with_params_on_both_stages():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count('with_params(gates1, grid1, "stage1")') == 1
    assert source.count('with_params(gates2, grid2, "stage2")') == 1
    tree = ast.parse(source)
    defined = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert "with_params" not in defined


# --- U18.1: main() prints every cleared invoice's days late right after the unpacking --------

def test_main_prints_the_u18_1_heading_right_after_the_inputs_are_unpacked():
    body = main_body()
    at = unpacking_ends()
    assert ast.unparse(body[at + 1]) == U18_1_PRINT


def test_main_shows_the_cleared_invoice_delays_of_the_ledger_right_after_the_heading():
    body = main_body()
    at = unpacking_ends()
    assert ast.unparse(body[at + 2]) == U18_1_SHOW


def test_the_u18_1_calls_are_made_once_in_the_whole_script():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for name in ("cleared_invoice_delays", "show_cleared_invoice_delays"):
        found = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == name]
        assert len(found) == 1, name
    assert len(calls_to(main_tree(), "show_cleared_invoice_delays")) == 1


def test_the_u18_1_printout_reads_the_ledger_unpacked_from_the_inputs():
    body = main_body()
    at = unpacking_ends()
    owner = [node for node in body[:at + 1] if isinstance(node, ast.Assign)
             and any(isinstance(n, ast.Name) and n.id == "ledger" and isinstance(n.ctx, ast.Store)
                     for t in node.targets for n in ast.walk(t))]
    assert len(owner) == 1
    assert "inputs.ledger" in ast.unparse(owner[0].value)
    # The only binding of `ledger` in main(): not rebound before the call.
    stores = [n for n in ast.walk(main_tree()) if isinstance(n, ast.Name) and n.id == "ledger"
              and isinstance(n.ctx, (ast.Store, ast.Del))]
    assert len(stores) == 1


def test_the_script_imports_cleared_invoice_delays_from_receivables_asof():
    imports = script_imports()
    assert imports["collection_estimation.hurdle.receivables_asof"] == ["cleared_invoice_delays"]
    assert "show_cleared_invoice_delays" in imports[
        "collection_estimation.hurdle.reports.preparation"]


# --- U18.2: main() builds the delay spread right after the U18.1 printout ----------------------

U18_2_PRINT = "print('U18.2: the delay spread of the open invoices')"
U18_2_DUES = ("dues = expected_dues(table[['key', 'company_code', 'origin_week', 'h']], ledger, "
              "rates, levels, week_zero, details.holidays, "
              "horizons=settings.receivables.horizons, "
              "min_cleared=settings.receivables.min_cleared_for_delay, "
              "delays_kept=settings.rhythm.delays_kept, "
              "company_delays_kept=settings.rhythm.company_delays_kept, "
              "leak_check_weeks=settings.receivables.leak_check_weeks)")
U18_2_SHOW = ("show_expected_dues(dues, table, settings.evaluation.validation_origins, "
              "settings.receivables.horizons)")
U18_2_KEYWORDS = {"horizons": "settings.receivables.horizons",
                  "min_cleared": "settings.receivables.min_cleared_for_delay",
                  "delays_kept": "settings.rhythm.delays_kept",
                  "company_delays_kept": "settings.rhythm.company_delays_kept",
                  "leak_check_weeks": "settings.receivables.leak_check_weeks"}


def expected_dues_call():
    found = calls_to(main_tree(), "expected_dues")
    assert len(found) == 1
    return found[0]


def test_main_prints_the_u18_2_heading_right_after_the_u18_1_printout():
    body = main_body()
    at = unpacking_ends()
    assert ast.unparse(body[at + 2]) == U18_1_SHOW
    assert ast.unparse(body[at + 3]) == U18_2_PRINT


def test_main_builds_the_expected_dues_right_after_the_u18_2_heading():
    body = main_body()
    at = unpacking_ends()
    assert ast.unparse(body[at + 4]) == U18_2_DUES


def test_main_shows_the_expected_dues_right_after_building_them():
    body = main_body()
    at = unpacking_ends()
    assert ast.unparse(body[at + 5]) == U18_2_SHOW
    assert ast.unparse(body[at + 6]) == SHOW_TOPUPS


def test_expected_dues_gets_the_four_panel_columns_of_the_table():
    call = expected_dues_call()
    first = call.args[0]
    assert isinstance(first, ast.Subscript)
    assert ast.unparse(first.value) == "table"
    assert ast.literal_eval(first.slice) == ["key", "company_code", "origin_week", "h"]


def test_expected_dues_positional_arguments_in_order():
    call = expected_dues_call()
    assert [ast.unparse(arg) for arg in call.args] == [
        "table[['key', 'company_code', 'origin_week', 'h']]", "ledger", "rates", "levels",
        "week_zero", "details.holidays"]


def test_expected_dues_keyword_arguments_come_from_the_settings():
    call = expected_dues_call()
    assert {kw.arg: ast.unparse(kw.value) for kw in call.keywords} == U18_2_KEYWORDS


def test_expected_dues_call_matches_the_function_signature():
    import inspect
    from collection_estimation.hurdle.features_rhythm import expected_dues
    call = expected_dues_call()
    parameters = list(inspect.signature(expected_dues).parameters)
    assert parameters[:len(call.args)] == ["panel_rows", "ledger", "rates", "levels",
                                           "week_zero", "holidays"]
    assert [kw.arg for kw in call.keywords] == parameters[len(call.args):]


def test_the_settings_read_by_u18_2_exist_in_the_synthetic_settings():
    from collection_estimation.hurdle.settings import load_settings
    settings_file = SCRIPT.parent / "settings" / "synthetic.toml"
    if not settings_file.exists():
        pytest.skip("settings/synthetic.toml absent")
    settings = load_settings(settings_file, SCRIPT.parent)
    for text in list(U18_2_KEYWORDS.values()) + ["settings.evaluation.validation_origins"]:
        value = settings
        for part in text.split(".")[1:]:
            value = getattr(value, part)
        assert value is not None, text


def test_show_expected_dues_reads_the_dues_and_the_whole_table():
    found = calls_to(main_tree(), "show_expected_dues")
    assert len(found) == 1
    assert [ast.unparse(arg) for arg in found[0].args] == [
        "dues", "table", "settings.evaluation.validation_origins",
        "settings.receivables.horizons"]
    assert found[0].keywords == []


def test_dues_is_bound_once_in_main():
    stores = [n for n in ast.walk(main_tree()) if isinstance(n, ast.Name) and n.id == "dues"
              and isinstance(n.ctx, (ast.Store, ast.Del))]
    assert len(stores) == 1


def test_the_u18_2_calls_are_made_once_in_the_whole_script():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for name in ("expected_dues", "show_expected_dues"):
        found = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == name]
        assert len(found) == 1, name


def test_the_u18_2_statements_read_main_names_bound_before_them():
    body = main_body()
    at = unpacking_ends()
    stored = {n.id for node in body[:at + 1] for n in ast.walk(node)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    stored.add("dues")   # bound by the statement before show_expected_dues
    for node in body[at + 4:at + 6]:
        read = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)} - {"settings", "expected_dues",
                                                     "show_expected_dues"}
        assert read <= stored, read - stored
    for name in ("table", "ledger", "rates", "levels", "week_zero", "details"):
        assert name in stored, name


def test_the_script_imports_expected_dues_from_features_rhythm():
    imports = script_imports()
    assert imports["collection_estimation.hurdle.features_rhythm"] == ["expected_dues"]
    assert "show_expected_dues" in imports["collection_estimation.hurdle.reports.preparation"]
