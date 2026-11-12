"""Tests of `hurdle/grid_run.py`: `run_grid`, step 18 (the settings grid, U16) from end to end.

run_grid is what run_hurdle_pipeline.py (step 18) and run_hurdle_grid.py both call. It is
checked on its syntax tree (what it calls, in which order, with which arguments), then run with every grid
function and printout replaced in grid_run's namespace by a recorder, on a ModelInputs whose
fields are named placeholders, so that each argument can be traced to the input it came from.
The grid functions themselves are tested in test_grid.py, the printouts in
reports/test_grid.py; main()'s call of run_grid in test_run_hurdle_pipeline.py.
"""
import ast
import contextlib
import dataclasses
import inspect
import io
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from collection_estimation.hurdle import grid_run
from collection_estimation.hurdle.inputs import ModelInputs
from tests.hurdle.code_checks import HURDLE, REPO_ROOT, calls_to, function_tree, keywords_of

GRID_RUN = HURDLE / "grid_run.py"
SCRIPT = REPO_ROOT / "run_hurdle_pipeline.py"

# The statements of run_grid after its docstring. U16.2: the settings to try
# (show_grid_candidates). U16.3: every candidate at every validation origin (grid_predictions,
# each fit cached under hurdle_grid, with a progress callback), then show_grid_predictions,
# which checks candidate 0 against step 17's rolled forecasts (or says it cannot, without
# them). U16.4: grid_gates, and show_grid_gates once per stage. U16.5: the actual cash at the
# grid's origins, every combination scored (grid_scores) and show_grid_scores. U16.6:
# grid_choice and show_grid_choice. U16.7: without [grid] run_test, a line saying the v1 test
# is not run and (choice, None); with it, the v1 test on the test origins and (choice, test).
FIT_PROGRESS = (
    "def fit_progress(done, total, stage, at, number, seconds):\n"
    "    left = (time.perf_counter() - started) / done * (total - done)\n"
    "    print(f'  ... {stage} setting {number}, origin {at}: {done} of {total} done "
    "({seconds:.0f} s), about {left / 60:.0f} min left')")
GRID_CALL = (
    "predictions = grid_predictions(inputs.table, origins, candidates, settings, "
    "block_columns=inputs.block_columns, companies=inputs.companies, "
    "new_cash=inputs.new_cash, refunds=inputs.refunds, unidentified=inputs.unidentified, "
    "levels=inputs.levels, calendar=inputs.calendar, cache_folder=cache_folder, "
    "on_fit_done=fit_progress)")
ACTUAL = ("actual = actual_totals(inputs.weekly, inputs.weeks, inputs.levels, inputs.companies, "
          "origins, horizons)")
SCORES = ("table = grid_scores(predictions, candidates, gates, actual, settings, "
          "companies=inputs.companies, week_zero=inputs.week_zero)")
SHOW_SCORES = "show_grid_scores(table, candidates, rolled, actual, horizons)"
CHOICE = ("choice = grid_choice(predictions, candidates, table, actual, settings, "
          "companies=inputs.companies, week_zero=inputs.week_zero, weekly=inputs.weekly, "
          "weeks=inputs.weeks, levels=inputs.levels, calendar=inputs.calendar, "
          "ledger=inputs.ledger, rates=inputs.rates, new_cash=inputs.new_cash, "
          "refunds=inputs.refunds, unidentified=inputs.unidentified)")
SHOW_CHOICE = "show_grid_choice(choice, candidates, table, settings)"
NOT_RUN = ("if not settings.grid.run_test:\n"
           "    print('        the v1 test is not run ([grid] run_test = false)')\n"
           "    return (choice, None)")
TEST_ORIGINS = "test_origins = list(settings.evaluation.test_origins)"
TEST_BANNER = (
    "print(f'        V1 TEST over the test origins {test_origins[0]}-{test_origins[-1]} "
    "({len(test_origins)} origins): a report, not a choice; do not change the settings "
    "because of it')")
TEST_ACTUAL = ("test_actual = actual_totals(inputs.weekly, inputs.weeks, inputs.levels, "
               "inputs.companies, test_origins, horizons)")
TEST_CALL = (
    "test = grid_test(choice, test_actual, settings, table=inputs.table, "
    "block_columns=inputs.block_columns, companies=inputs.companies, "
    "week_zero=inputs.week_zero, weekly=inputs.weekly, weeks=inputs.weeks, "
    "levels=inputs.levels, calendar=inputs.calendar, ledger=inputs.ledger, rates=inputs.rates, "
    "new_cash=inputs.new_cash, refunds=inputs.refunds, unidentified=inputs.unidentified, "
    "cache_folder=cache_folder, on_fit_done=fit_progress)")
SHOW_TEST = "show_grid_test(test, choice, settings)"
TEST_PART = [TEST_ORIGINS, TEST_BANNER, "started = time.perf_counter()", TEST_ACTUAL, TEST_CALL,
             SHOW_TEST, "return (choice, test)"]
STATEMENTS = [
    "origins = list(settings.evaluation.validation_origins)",
    "horizons = settings.receivables.horizons",
    "cache_folder = settings.data.cache_folder / 'hurdle_grid'",
    "candidates = grid_candidates(settings)",
    "print(f'Step 18: settings grid (U16) over the validation origins "
    "{origins[0]}-{origins[-1]} ({len(origins)} origins)')",
    "show_grid_candidates(candidates, len(origins))",
    "started = time.perf_counter()",
    FIT_PROGRESS,
    GRID_CALL,
    "show_grid_predictions(predictions, rolled)",
    "gates = grid_gates(predictions, candidates, settings)",
    "show_grid_gates('stage 1', gates.stage1, 'log loss', ascending=True)",
    "show_grid_gates('stage 2', gates.stage2, 'deviance skill', ascending=False)",
    ACTUAL,
    SCORES,
    SHOW_SCORES,
    CHOICE,
    SHOW_CHOICE,
    NOT_RUN,
] + TEST_PART
NAMES = ("grid_candidates", "show_grid_candidates", "grid_predictions", "show_grid_predictions",
         "grid_gates", "grid_scores", "show_grid_scores", "grid_choice", "show_grid_choice",
         "grid_test", "show_grid_test")
CHOICE_KEYWORDS = {name: f"inputs.{name}" for name in (
    "companies", "week_zero", "weekly", "weeks", "levels", "calendar", "ledger", "rates",
    "new_cash", "refunds", "unidentified")}
TEST_NOT_RUN_LINE = "        the v1 test is not run ([grid] run_test = false)"


def run_grid_tree():
    return function_tree(GRID_RUN, "run_grid")


def body():
    """run_grid's statements after its docstring."""
    statements = run_grid_tree().body
    assert isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant)
    return statements[1:]


def statements():
    return [ast.unparse(node) for node in body()]


def not_run_at():
    """The index of the `if not settings.grid.run_test:` statement in body()."""
    return statements().index(NOT_RUN)


def calls_in(nodes, name):
    return [call for node in nodes for call in calls_to(node, name)]


def main_tree():
    return function_tree(SCRIPT, "main")


# --- the module and run_grid's signature ------------------------------------------------------

def test_run_grid_is_the_modules_only_function():
    module = ast.parse(GRID_RUN.read_text(encoding="utf-8"))
    assert [node.name for node in module.body if isinstance(node, ast.FunctionDef)] == [
        "run_grid"]
    assert ast.get_docstring(module)


def test_run_grid_takes_the_settings_the_inputs_and_rolled_by_keyword_only():
    parameters = inspect.signature(grid_run.run_grid).parameters
    assert list(parameters) == ["settings", "inputs", "rolled"]
    assert parameters["rolled"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["rolled"].default is None
    assert parameters["inputs"].annotation is ModelInputs
    for name in ("settings", "inputs"):
        assert parameters[name].default is inspect.Parameter.empty, name


def test_grid_run_imports_its_names_once_from_their_modules():
    module = ast.parse(GRID_RUN.read_text(encoding="utf-8"))
    imports = {node.module: [(a.name, a.asname) for a in node.names] for node in module.body
               if isinstance(node, ast.ImportFrom)}
    assert imports == {
        "evaluate": [("actual_totals", None)],
        "grid": [(name, None) for name in ("GridChoice", "GridTest", "grid_candidates",
                                           "grid_choice", "grid_gates", "grid_predictions",
                                           "grid_scores", "grid_test")],
        "inputs": [("ModelInputs", None)],
        "reports.grid": [(name, None) for name in (
            "show_grid_candidates", "show_grid_choice", "show_grid_gates",
            "show_grid_predictions", "show_grid_scores", "show_grid_test")],
    }
    assert [node.names[0].name for node in module.body if isinstance(node, ast.Import)] == [
        "time"]


def test_grid_run_names_are_the_modules_functions_with_the_signatures_it_uses():
    # Was GRID_NAMES on run_hurdle_pipeline.py (the script imported these itself before).
    from collection_estimation.hurdle import evaluate, grid
    from collection_estimation.hurdle.reports import grid as grid_reports
    for name in ("grid_candidates", "grid_predictions", "grid_gates", "grid_scores",
                 "grid_choice", "grid_test", "GridChoice", "GridTest"):
        assert getattr(grid_run, name) is getattr(grid, name), name
    for name in ("show_grid_candidates", "show_grid_predictions", "show_grid_gates",
                 "show_grid_scores", "show_grid_choice", "show_grid_test"):
        assert getattr(grid_run, name) is getattr(grid_reports, name), name
    assert grid_run.actual_totals is evaluate.actual_totals
    inspect.signature(grid_run.grid_candidates).bind("settings")
    inspect.signature(grid_run.grid_predictions).bind(
        "table", [1], "candidates", "settings", block_columns={}, companies=[], new_cash=None,
        refunds=None, unidentified=None, levels=None, calendar=None, cache_folder="c",
        on_fit_done=None)
    inspect.signature(grid_run.show_grid_candidates).bind("candidates", 3)
    inspect.signature(grid_run.show_grid_predictions).bind("predictions", None)
    inspect.signature(grid_run.grid_gates).bind("predictions", "candidates", "settings")
    inspect.signature(grid_run.show_grid_gates).bind("stage 1", "gates", "log loss",
                                                     ascending=True)
    inspect.signature(grid_run.grid_scores).bind("predictions", "candidates", "gates",
                                                 "actual", "settings", companies=[],
                                                 week_zero=None)
    inspect.signature(grid_run.show_grid_scores).bind("table", "candidates", None, "actual", 5)
    inspect.signature(grid_run.grid_choice).bind(
        "predictions", "candidates", "table", "actual", "settings", companies=[],
        week_zero=None, weekly=None, weeks=None, levels=None, calendar=None, ledger=None,
        rates=None, new_cash=None, refunds=None, unidentified=None)
    inspect.signature(grid_run.show_grid_choice).bind("choice", "candidates", "table",
                                                      "settings")
    inspect.signature(grid_run.grid_test).bind(
        "choice", "actual", "settings", table=None, block_columns={}, companies=[],
        week_zero=None, weekly=None, weeks=None, levels=None, calendar=None, ledger=None,
        rates=None, new_cash=None, refunds=None, unidentified=None, cache_folder="c",
        on_fit_done=None)
    inspect.signature(grid_run.show_grid_test).bind("test", "choice", "settings")
    inspect.signature(grid_run.actual_totals).bind("weekly", "weeks", "levels", [], [1], 5)


# --- run_grid's syntax tree -----------------------------------------------------------------

def test_run_grid_statements_in_order():
    assert statements() == STATEMENTS


def test_run_grid_calls_each_grid_function_and_printout_once():
    tree = run_grid_tree()
    for name in NAMES:
        assert len(calls_to(tree, name)) == 1, name
    assert len(calls_to(tree, "show_grid_gates")) == 2
    assert len(calls_to(tree, "actual_totals")) == 2      # the grid's, then the v1 test's


def test_run_grid_has_one_if_the_run_test_check_and_no_loop():
    ifs = [node for node in ast.walk(run_grid_tree()) if isinstance(node, ast.If)]
    assert len(ifs) == 1 and ast.unparse(ifs[0].test) == "not settings.grid.run_test"
    assert ifs[0].orelse == []
    assert not [node for node in ast.walk(run_grid_tree())
                if isinstance(node, (ast.For, ast.While))]


def test_run_grid_never_reads_the_grid_run_switch():
    # [grid] run is the pipeline main()'s to read (run_hurdle_grid.py runs the grid without
    # reading it); run_grid reads run_test only, once.
    source = " ".join(ast.unparse(run_grid_tree()).split())
    assert not re.search(r"settings\.grid\.run\b", source)
    assert source.count("settings.grid.run_test") == 1


def test_run_grid_returns_choice_and_none_or_choice_and_test():
    returns = [ast.unparse(node) for node in sorted(
        (node for node in ast.walk(run_grid_tree()) if isinstance(node, ast.Return)),
        key=lambda node: node.lineno)]
    assert returns == ["return (choice, None)", "return (choice, test)"]
    assert ast.unparse(run_grid_tree().returns) == "tuple[GridChoice, GridTest | None]"


def test_run_grid_reads_the_validation_origins_before_the_check_and_the_test_origins_after():
    before, after = body()[:not_run_at()], body()[not_run_at() + 1:]
    assert ast.unparse(before[0]) == "origins = list(settings.evaluation.validation_origins)"
    assert ast.unparse(after[0]) == TEST_ORIGINS

    def names(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    for node in before:
        assert "test_origins" not in ast.unparse(node), ast.unparse(node)
    for node in after:
        assert "validation_origins" not in ast.unparse(node), ast.unparse(node)
        assert "origins" not in names(node), ast.unparse(node)


def test_run_grid_cache_folder_is_a_subfolder_of_the_data_cache_made_once():
    assignment = body()[2]
    assert ast.unparse(assignment.targets[0]) == "cache_folder"
    folder = assignment.value
    assert isinstance(folder, ast.BinOp) and isinstance(folder.op, ast.Div)
    assert ast.unparse(folder.left) == "settings.data.cache_folder"
    assert isinstance(folder.right, ast.Constant) and folder.right.value == "hurdle_grid"
    stores = [node for node in ast.walk(run_grid_tree()) if isinstance(node, ast.Name)
              and node.id == "cache_folder" and isinstance(node.ctx, ast.Store)]
    assert len(stores) == 1


def test_run_grid_grid_predictions_and_grid_test_share_the_cache_folder_and_callback():
    predictions = keywords_of(calls_to(run_grid_tree(), "grid_predictions")[0])
    test = keywords_of(calls_to(run_grid_tree(), "grid_test")[0])
    for call in (predictions, test):
        assert call["cache_folder"] == "cache_folder"
        assert call["on_fit_done"] == "fit_progress"


def test_run_grid_fit_progress_is_defined_after_the_clock_and_before_the_grid():
    nodes = body()
    defs = [i for i, node in enumerate(nodes)
            if isinstance(node, ast.FunctionDef) and node.name == "fit_progress"]
    assert len(defs) == 1
    progress = nodes[defs[0]]
    assert [a.arg for a in progress.args.args] == ["done", "total", "stage", "at", "number",
                                                   "seconds"]
    assert progress.args.defaults == [] and progress.args.vararg is None
    assert statements().index("started = time.perf_counter()") < defs[0]
    assert defs[0] < statements().index(GRID_CALL)


def test_run_grid_the_test_restarts_the_clock_after_the_banner_and_before_the_fits():
    after = statements()[not_run_at() + 1:]
    clock = after.index("started = time.perf_counter()")
    assert after.index(TEST_BANNER) < clock < after.index(TEST_ACTUAL) < after.index(TEST_CALL)


def test_run_grid_grid_predictions_gets_the_table_pieces_without_week_zero():
    # The same table pieces rolling_forecasts gets in step 17, from the inputs (no week_zero:
    # the grid does not re-inflate), its own cache folder and its own callback.
    call = calls_to(run_grid_tree(), "grid_predictions")[0]
    assert [ast.unparse(a) for a in call.args] == ["inputs.table", "origins", "candidates",
                                                   "settings"]
    expected = {name: f"inputs.{name}" for name in (
        "block_columns", "companies", "new_cash", "refunds", "unidentified", "levels",
        "calendar")}
    expected.update(cache_folder="cache_folder", on_fit_done="fit_progress")
    assert keywords_of(call) == expected


def test_run_grid_grid_predictions_gets_what_main_gives_rolling_forecasts():
    # main() unpacks each of its names from the inputs field of the same name, so `companies`
    # there is `inputs.companies` here.
    rolling = keywords_of(calls_to(main_tree(), "rolling_forecasts")[0])
    call = keywords_of(calls_to(run_grid_tree(), "grid_predictions")[0])
    for name in ("block_columns", "companies", "new_cash", "refunds", "unidentified", "levels",
                 "calendar"):
        assert call[name] == f"inputs.{rolling[name]}", name


def test_run_grid_the_actual_cash_of_the_grid_and_of_the_test_differ_by_the_origins_only():
    grid_actual, test_actual = calls_to(run_grid_tree(), "actual_totals")
    grid_args = [ast.unparse(a) for a in grid_actual.args]
    test_args = [ast.unparse(a) for a in test_actual.args]
    assert grid_actual.keywords == [] and test_actual.keywords == []
    assert grid_args == ["inputs.weekly", "inputs.weeks", "inputs.levels", "inputs.companies",
                         "origins", "horizons"]
    assert test_args[4] == "test_origins"
    assert grid_args[:4] + grid_args[5:] == test_args[:4] + test_args[5:]


def test_run_grid_actual_cash_reads_what_main_gives_step_17s_actual():
    # The current combination can then be compared with step 17's model on the same weeks.
    found = [[ast.unparse(a) for a in call.args] for call in calls_to(main_tree(), "actual_totals")]
    step_17 = [args for args in found if args[4] == "origins"]
    assert len(step_17) == 1
    step_17 = step_17[0]
    grid = [ast.unparse(a) for a in calls_to(run_grid_tree(), "actual_totals")[0].args]
    assert step_17[4] == "origins" and grid[4] == "origins"
    assert [f"inputs.{name}" for name in step_17[:4]] == grid[:4]
    assert step_17[5] == "settings.receivables.horizons" and grid[5] == "horizons"


def test_run_grid_grid_gates_and_their_printouts():
    call = calls_to(run_grid_tree(), "grid_gates")[0]
    assert [ast.unparse(a) for a in call.args] == ["predictions", "candidates", "settings"]
    assert call.keywords == []
    found = [(ast.unparse(c.args[0]), ast.unparse(c.args[1]), ast.unparse(c.args[2]),
              keywords_of(c)) for c in calls_to(run_grid_tree(), "show_grid_gates")]
    assert found == [("'stage 1'", "gates.stage1", "'log loss'", {"ascending": "True"}),
                     ("'stage 2'", "gates.stage2", "'deviance skill'", {"ascending": "False"})]


def test_run_grid_grid_choice_keywords_are_the_inputs():
    call = calls_to(run_grid_tree(), "grid_choice")[0]
    assert [ast.unparse(a) for a in call.args] == ["predictions", "candidates", "table",
                                                   "actual", "settings"]
    assert keywords_of(call) == CHOICE_KEYWORDS


def test_run_grid_grid_choice_reads_what_main_gives_step_17s_references():
    # The references are made inside grid_choice from the same inputs step 17 gives them.
    call = keywords_of(calls_to(run_grid_tree(), "grid_choice")[0])
    rule = calls_to(main_tree(), "rule_reference")[0]
    assert [f"inputs.{ast.unparse(a)}" for a in rule.args][:2] == [call["ledger"],
                                                                   call["rates"]]
    for name, value in keywords_of(rule).items():
        assert call[name] == f"inputs.{value}", name
    reference = [ast.unparse(a) for a in calls_to(main_tree(), "deflated_reference")[0].args]
    assert [f"inputs.{name}" for name in reference[:5]] == [
        call["weekly"], call["weeks"], call["levels"], call["calendar"], call["companies"]]
    assert f"inputs.{reference[-1]}" == call["week_zero"]


def test_run_grid_grid_choice_reads_none_of_step_17s_results():
    source = ast.unparse(calls_to(run_grid_tree(), "grid_choice")[0])
    for name in ("rolled", "reference", "rule", "rolled_actual", "scores", "bootstrap",
                 "forecasts", "origins"):
        assert not re.search(rf"\b{name}\b", source), name


def test_run_grid_grid_test_keywords_are_the_grids_and_the_choices():
    call = calls_to(run_grid_tree(), "grid_test")[0]
    assert [ast.unparse(a) for a in call.args] == ["choice", "test_actual", "settings"]
    keywords = keywords_of(call)
    predictions = keywords_of(calls_to(run_grid_tree(), "grid_predictions")[0])
    choice = keywords_of(calls_to(run_grid_tree(), "grid_choice")[0])
    assert keywords["table"] == "inputs.table"
    for name in ("block_columns", "companies", "new_cash", "refunds", "unidentified", "levels",
                 "calendar", "cache_folder", "on_fit_done"):
        assert keywords[name] == predictions[name], name
    for name, value in choice.items():
        assert keywords[name] == value, name
    assert set(keywords) == {"table"} | set(choice) | {"block_columns", "cache_folder",
                                                        "on_fit_done"}


def test_run_grid_printouts_get_their_arguments():
    tree = run_grid_tree()
    expected = {"show_grid_candidates": ["candidates", "len(origins)"],
                "show_grid_predictions": ["predictions", "rolled"],
                "show_grid_scores": ["table", "candidates", "rolled", "actual", "horizons"],
                "show_grid_choice": ["choice", "candidates", "table", "settings"],
                "show_grid_test": ["test", "choice", "settings"]}
    for name, arguments in expected.items():
        call = calls_to(tree, name)[0]
        assert [ast.unparse(a) for a in call.args] == arguments, name
        assert call.keywords == [], name


def test_run_grid_reads_rolled_only_in_the_two_printouts_that_check_it():
    tree = run_grid_tree()   # one parse: nodes are matched by identity below
    reads = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
             and node.id == "rolled"]
    assert len(reads) == 2
    assert {ast.unparse(call.func) for call in ast.walk(tree)
            if isinstance(call, ast.Call) and any(a is r for a in call.args for r in reads)} == {
        "show_grid_predictions", "show_grid_scores"}


def test_run_grid_has_none_of_the_u16_1_names():
    names = {node.id for node in ast.walk(ast.parse(GRID_RUN.read_text(encoding="utf-8")))
             if isinstance(node, ast.Name)}
    for name in ("stage1_predictions", "stage2_predictions", "show_stage_predictions", "p_alone",
                 "m_alone", "p_seconds", "m_seconds"):
        assert name not in names, name


# --- run_grid run with recorders -------------------------------------------------------------

INPUTS = ModelInputs(
    companies="<companies>", week_zero="<week zero>", last_week="<last week>",
    weeks="<weeks>", weekly="<weekly>", ledger="<ledger>", rates="<rates>", levels="<levels>",
    calendar="<calendar>", table="<table>", block_columns="<blocks>", new_cash="<new cash>",
    refunds="<refunds>", unidentified="<unidentified>")

# Every field but last_week, as each grid function receives it.
INPUT_KEYWORDS = {"table": "<table>", "block_columns": "<blocks>", "companies": "<companies>",
                  "week_zero": "<week zero>", "weekly": "<weekly>", "weeks": "<weeks>",
                  "levels": "<levels>", "calendar": "<calendar>", "ledger": "<ledger>",
                  "rates": "<rates>", "new_cash": "<new cash>", "refunds": "<refunds>",
                  "unidentified": "<unidentified>"}


class Run:
    """run_grid with recorders for every function it calls, in grid_run's namespace."""

    def __init__(self, monkeypatch, capsys, *, run_test=False, validation_origins=(37, 40, 44),
                 test_origins=None, ticks=(100.0,), progress_calls=(), test_progress_calls=(),
                 horizons=5, cache_root=Path("cache_root")):
        self.calls = []
        self.monkeypatch, self.capsys = monkeypatch, capsys
        clock = iter(ticks)
        actual = iter(["<grid actual>", "<test actual>"])

        def recorder(name, result):
            def call(*args, **kwargs):
                self.calls.append((name, args, kwargs))
                return result() if callable(result) else result
            return call

        def fake_grid(*args, **kwargs):
            self.calls.append(("grid_predictions", args, kwargs))
            for item in progress_calls:
                kwargs["on_fit_done"](*item)
            return "<predictions>"

        def fake_test(*args, **kwargs):
            self.calls.append(("grid_test", args, kwargs))
            for item in test_progress_calls:
                kwargs["on_fit_done"](*item)
            return "<test>"

        fakes = {
            "grid_candidates": recorder("grid_candidates", "<candidates>"),
            "show_grid_candidates": recorder("show_grid_candidates", None),
            "grid_predictions": fake_grid,
            "show_grid_predictions": recorder("show_grid_predictions", None),
            "grid_gates": recorder("grid_gates", SimpleNamespace(stage1="<gates 1>",
                                                                 stage2="<gates 2>")),
            "show_grid_gates": recorder("show_grid_gates", None),
            "actual_totals": recorder("actual_totals", lambda: next(actual)),
            "grid_scores": recorder("grid_scores", "<grid table>"),
            "show_grid_scores": recorder("show_grid_scores", None),
            "grid_choice": recorder("grid_choice", "<choice>"),
            "show_grid_choice": recorder("show_grid_choice", None),
            "grid_test": fake_test,
            "show_grid_test": recorder("show_grid_test", None),
        }
        for name, fake in fakes.items():
            monkeypatch.setattr(grid_run, name, fake)
        monkeypatch.setattr(grid_run, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
        # No [grid] run and, without run_test, no test origins: run_grid must read neither.
        evaluation = SimpleNamespace(validation_origins=validation_origins)
        if test_origins is not None:
            evaluation.test_origins = test_origins
        self.settings = SimpleNamespace(grid=SimpleNamespace(run_test=run_test),
                                        evaluation=evaluation,
                                        data=SimpleNamespace(cache_folder=cache_root),
                                        receivables=SimpleNamespace(horizons=horizons))

    def __call__(self, inputs=INPUTS, **kwargs):
        self.capsys.readouterr()
        self.returned = grid_run.run_grid(self.settings, inputs, **kwargs)
        self.printed = self.capsys.readouterr().out.splitlines()
        return self

    def names(self):
        return [name for name, _, _ in self.calls]

    def only(self, name):
        found = [(args, kwargs) for called, args, kwargs in self.calls if called == name]
        assert len(found) == 1, (name, len(found))
        return found[0]

    def all(self, name):
        return [(args, kwargs) for called, args, kwargs in self.calls if called == name]


@pytest.fixture
def run(monkeypatch, capsys):
    """run(**options)(**run_grid keywords): a Run, called."""
    def make(**options):
        return Run(monkeypatch, capsys, **options)
    return make


def test_run_grid_calls_in_order_without_the_test(run):
    found = run()(rolled="<rolled>")
    assert found.names() == ["grid_candidates", "show_grid_candidates", "grid_predictions",
                             "show_grid_predictions", "grid_gates", "show_grid_gates",
                             "show_grid_gates", "actual_totals", "grid_scores",
                             "show_grid_scores", "grid_choice", "show_grid_choice"]


def test_run_grid_calls_in_order_with_the_test(run):
    found = run(run_test=True, test_origins=(49, 60), ticks=(100.0, 500.0))(rolled="<rolled>")
    assert found.names() == ["grid_candidates", "show_grid_candidates", "grid_predictions",
                             "show_grid_predictions", "grid_gates", "show_grid_gates",
                             "show_grid_gates", "actual_totals", "grid_scores",
                             "show_grid_scores", "grid_choice", "show_grid_choice",
                             "actual_totals", "grid_test", "show_grid_test"]


def test_run_grid_without_the_test_returns_the_choice_and_none(run):
    found = run()()
    assert found.returned == ("<choice>", None)
    assert type(found.returned) is tuple


def test_run_grid_with_the_test_returns_the_choice_and_the_test(run):
    found = run(run_test=True, test_origins=(49, 60), ticks=(100.0, 500.0))()
    assert found.returned == ("<choice>", "<test>")
    assert type(found.returned) is tuple


@pytest.mark.parametrize("origins, header", [
    ((37, 40, 44), "over the validation origins 37-44 (3 origins)"),
    ((40,), "over the validation origins 40-40 (1 origins)"),
    (tuple(range(20, 45)), "over the validation origins 20-44 (25 origins)"),
])
def test_run_grid_prints_the_header_then_says_the_test_is_not_run(run, origins, header):
    found = run(validation_origins=origins)()
    assert found.printed == [f"Step 18: settings grid (U16) {header}", TEST_NOT_RUN_LINE]


def test_run_grid_rolled_defaults_to_none(run):
    found = run()()
    assert found.only("show_grid_predictions") == (("<predictions>", None), {})
    assert found.only("show_grid_scores")[0][2] is None


@pytest.mark.parametrize("rolled", ["<rolled>", None])
def test_run_grid_forwards_rolled_to_the_predictions_and_scores_printouts(run, rolled):
    found = run()(rolled=rolled)
    assert found.only("show_grid_predictions") == (("<predictions>", rolled), {})
    assert found.only("show_grid_scores") == (
        ("<grid table>", "<candidates>", rolled, "<grid actual>", 5), {})
    for name, args, kwargs in found.calls:
        if name not in ("show_grid_predictions", "show_grid_scores"):
            assert "<rolled>" not in list(args) + list(kwargs.values()), name


def test_run_grid_grid_candidates_get_the_settings_themselves(run):
    found = run()()
    (settings,), kwargs = found.only("grid_candidates")
    assert settings is found.settings and kwargs == {}


@pytest.mark.parametrize("origins, count", [((37, 40, 44), 3), ((40,), 1),
                                            (tuple(range(20, 45)), 25)])
def test_run_grid_shows_the_candidates_with_the_origin_count(run, origins, count):
    found = run(validation_origins=origins)()
    assert found.only("show_grid_candidates") == (("<candidates>", count), {})


def test_run_grid_grid_predictions_gets_the_inputs_fields_and_the_cache_folder(run):
    found = run(validation_origins=(40, 37, 44))()
    args, kwargs = found.only("grid_predictions")
    assert args[0] == "<table>"
    # The origins as the settings list them (grid_predictions sorts them itself), as a list.
    assert args[1] == [40, 37, 44] and type(args[1]) is list
    assert args[2] == "<candidates>" and args[3] is found.settings
    on_fit_done = kwargs.pop("on_fit_done")
    assert callable(on_fit_done)
    assert kwargs == {name: INPUT_KEYWORDS[name] for name in (
        "block_columns", "companies", "new_cash", "refunds", "unidentified", "levels",
        "calendar")} | {"cache_folder": Path("cache_root") / "hurdle_grid"}


def test_run_grid_cache_folder_follows_the_settings(run, tmp_path):
    found = run(run_test=True, test_origins=(49,), ticks=(1.0, 2.0), cache_root=tmp_path)()
    assert found.only("grid_predictions")[1]["cache_folder"] == tmp_path / "hurdle_grid"
    assert found.only("grid_test")[1]["cache_folder"] == tmp_path / "hurdle_grid"
    assert not (tmp_path / "hurdle_grid").exists()   # made by the grid's fits, not here


def test_run_grid_gates_get_the_predictions_candidates_and_settings(run):
    found = run()()
    args, kwargs = found.only("grid_gates")
    assert args == ("<predictions>", "<candidates>", found.settings) and kwargs == {}
    assert found.all("show_grid_gates") == [
        (("stage 1", "<gates 1>", "log loss"), {"ascending": True}),
        (("stage 2", "<gates 2>", "deviance skill"), {"ascending": False})]


def test_run_grid_reads_the_actual_cash_at_the_validation_origins(run):
    found = run(validation_origins=(40, 37, 44), horizons=4)()
    assert found.all("actual_totals") == [
        (("<weekly>", "<weeks>", "<levels>", "<companies>", [40, 37, 44], 4), {})]


def test_run_grid_grid_scores_gets_the_gates_the_actual_and_the_inputs_fields(run):
    found = run()()
    (predictions, candidates, gates, actual, settings), kwargs = found.only("grid_scores")
    assert (predictions, candidates) == ("<predictions>", "<candidates>")
    assert (gates.stage1, gates.stage2) == ("<gates 1>", "<gates 2>")
    assert actual == "<grid actual>" and settings is found.settings
    assert kwargs == {"companies": "<companies>", "week_zero": "<week zero>"}


@pytest.mark.parametrize("horizons", [5, 3])
def test_run_grid_shows_the_scores_with_the_horizons_of_the_settings(run, horizons):
    found = run(horizons=horizons)(rolled="<rolled>")
    assert found.only("show_grid_scores") == (
        ("<grid table>", "<candidates>", "<rolled>", "<grid actual>", horizons), {})


def test_run_grid_grid_choice_gets_the_scores_and_the_inputs_fields(run):
    found = run()()
    (predictions, candidates, table, actual, settings), kwargs = found.only("grid_choice")
    assert (predictions, candidates, table, actual) == ("<predictions>", "<candidates>",
                                                        "<grid table>", "<grid actual>")
    assert settings is found.settings
    assert kwargs == {name: INPUT_KEYWORDS[name] for name in (
        "companies", "week_zero", "weekly", "weeks", "levels", "calendar", "ledger", "rates",
        "new_cash", "refunds", "unidentified")}


def test_run_grid_shows_the_choice_with_the_candidates_table_and_settings(run):
    found = run()(rolled="<rolled>")
    assert found.only("show_grid_choice") == (
        ("<choice>", "<candidates>", "<grid table>", found.settings), {})


def test_run_grid_without_the_test_never_reads_the_test_origins(run):
    # The settings have no test_origins at all: reading them would raise AttributeError.
    found = run()()
    assert not hasattr(found.settings.evaluation, "test_origins")
    assert found.names().count("actual_totals") == 1
    assert "grid_test" not in found.names() and "show_grid_test" not in found.names()


@pytest.mark.parametrize("origins, text", [
    ((49, 60), "49-60 (2 origins)"), ((57,), "57-57 (1 origins)"),
    (tuple(range(49, 61)), "49-60 (12 origins)")])
def test_run_grid_with_the_test_prints_the_v1_banner_and_not_the_not_run_line(run, origins,
                                                                              text):
    found = run(run_test=True, test_origins=origins, ticks=(100.0, 500.0))()
    assert found.printed == [
        "Step 18: settings grid (U16) over the validation origins 37-44 (3 origins)",
        f"        V1 TEST over the test origins {text}: a report, not a choice; do not change "
        f"the settings because of it"]


def test_run_grid_the_test_actual_is_at_the_test_origins_from_the_settings(run):
    found = run(run_test=True, test_origins=(60, 49), ticks=(100.0, 500.0))()
    assert found.all("actual_totals") == [
        (("<weekly>", "<weeks>", "<levels>", "<companies>", [37, 40, 44], 5), {}),
        (("<weekly>", "<weeks>", "<levels>", "<companies>", [60, 49], 5), {})]
    assert type(found.all("actual_totals")[1][0][4]) is list


def test_run_grid_grid_test_gets_the_choice_the_test_actual_and_the_inputs_fields(run):
    found = run(run_test=True, test_origins=(49, 60), ticks=(100.0, 500.0))()
    (choice, actual, settings), kwargs = found.only("grid_test")
    assert (choice, actual) == ("<choice>", "<test actual>") and settings is found.settings
    on_fit_done = kwargs.pop("on_fit_done")
    assert callable(on_fit_done)
    assert kwargs == INPUT_KEYWORDS | {"cache_folder": Path("cache_root") / "hurdle_grid"}


def test_run_grid_grid_test_gets_the_same_settings_callback_and_cache_as_the_grid(run):
    found = run(run_test=True, test_origins=(49, 60), ticks=(100.0, 500.0))()
    grid_args, grid_kwargs = found.only("grid_predictions")
    test_args, test_kwargs = found.only("grid_test")
    assert test_args[2] is grid_args[3]
    assert test_kwargs["on_fit_done"] is grid_kwargs["on_fit_done"]
    assert test_kwargs["cache_folder"] == grid_kwargs["cache_folder"]


def test_run_grid_shows_the_test_with_the_choice_and_settings(run):
    found = run(run_test=True, test_origins=(49, 60), ticks=(100.0, 500.0))()
    assert found.only("show_grid_test") == (("<test>", "<choice>", found.settings), {})


def test_run_grid_passes_on_the_inputs_objects_themselves(run):
    # Not copies: each grid function gets the very object the inputs hold.
    tables = {name: object() for name in INPUT_KEYWORDS}
    inputs = ModelInputs(last_week=64, **tables)
    found = run(run_test=True, test_origins=(49,), ticks=(1.0, 2.0))(inputs=inputs)
    _, kwargs = found.only("grid_test")
    for name, table in tables.items():
        assert kwargs[name] is table, name
    _, kwargs = found.only("grid_choice")
    for name in CHOICE_KEYWORDS:
        assert kwargs[name] is tables[name], name
    assert found.only("grid_predictions")[0][0] is tables["table"]


def test_run_grid_reads_the_inputs_fields_it_passes_on_and_never_last_week(run):
    read = []

    class Watched:
        def __getattr__(self, name):
            read.append(name)
            return getattr(INPUTS, name)

    run(run_test=True, test_origins=(49,), ticks=(1.0, 2.0))(inputs=Watched())
    assert set(read) == set(INPUT_KEYWORDS)
    assert "last_week" not in read


def test_run_grid_progress_lines(run):
    # The clock: 100 s at the start; the fits end at 160 s and 400 s (one tick per call).
    found = run(ticks=(100.0, 160.0, 400.0),
                progress_calls=[(1, 8, "stage 1", 37, 0, 59.6), (2, 8, "stage 2", 37, 3, 0.4)])()
    assert found.printed[1:] == [
        # 60 s per fit so far x 7 left = 420 s = 7 min.
        "  ... stage 1 setting 0, origin 37: 1 of 8 done (60 s), about 7 min left",
        # 300 s / 2 x 6 = 900 s = 15 min.
        "  ... stage 2 setting 3, origin 37: 2 of 8 done (0 s), about 15 min left",
        TEST_NOT_RUN_LINE,
    ]


def test_run_grid_last_progress_line_has_nothing_left(run):
    found = run(ticks=(0.0, 3000.0), progress_calls=[(4, 4, "stage 2", 44, 1, 12.0)])()
    assert found.printed[-2:] == [
        "  ... stage 2 setting 1, origin 44: 4 of 4 done (12 s), about 0 min left",
        TEST_NOT_RUN_LINE]


def test_run_grid_test_progress_counts_from_the_restarted_clock(run):
    # The grid's clock starts at 100 s; the test's restarts at 500 s; its one fit of 4 ends at
    # 560 s: 60 s per fit x 3 left = 180 s = 3 min (from the grid's start it would be 6 min).
    found = run(run_test=True, test_origins=(49, 60), ticks=(100.0, 500.0, 560.0),
                test_progress_calls=[(1, 4, "stage 1", 49, 0, 58.0)])()
    assert found.printed[-1] == ("  ... stage 1 setting 0, origin 49: 1 of 4 done (58 s), "
                                 "about 3 min left")
    assert found.printed[-2].startswith("        V1 TEST over the test origins 49-60")


def test_run_grid_reads_the_clock_once_without_the_test_and_twice_with_it(run):
    # Fewer ticks than reads would raise StopIteration: one read each for the grid's start
    # and the test's restart, none for the printouts.
    run(ticks=(1.0,))()
    run(run_test=True, test_origins=(49,), ticks=(1.0, 2.0))()


def test_run_grid_a_failing_grid_function_stops_it_before_the_choice(run, monkeypatch):
    found = run()

    def broken(*args, **kwargs):
        raise ValueError("gates failed")

    monkeypatch.setattr(grid_run, "grid_gates", broken)
    with pytest.raises(ValueError, match="gates failed"):
        found()
    assert "grid_scores" not in found.names() and "grid_choice" not in found.names()


def test_run_grid_changes_nothing_in_the_settings(run):
    found = run(run_test=True, test_origins=(49, 60), ticks=(100.0, 500.0))
    before = repr(found.settings)
    found()
    assert repr(found.settings) == before
    assert dataclasses.asdict(INPUTS) == {"last_week": "<last week>", **INPUT_KEYWORDS}


# --- run_grid from end to end on test_grid.py's tiny grid ------------------------------------
# Everything real but grid_candidates, which gives test_grid.py's two candidates per stage
# (the real one reads a full [grid] section of lists): the fits on test_forecast.py's small
# table, cached in tmp_path; the gates (wide open); the scores; the choice with its real
# calibration and references; and the v1 test. The settings are a frozen dataclass, as
# grid_choice and grid_test call `dataclasses.replace` on them.

E2E_VALIDATION = (8, 11, 14)     # the last forecasts up to week 19, before the test's
E2E_TEST = (22, 26)


@dataclasses.dataclass(frozen=True)
class E2ESettings:
    receivables: object
    topups: object
    grid: object
    evaluation: object
    calibration: object
    diagnostic: object
    stage1: object
    stage2: object
    data: object


def e2e_settings(cache_root, run_test=True):
    from tests.hurdle.test_grid import GRID, fit_settings
    base = fit_settings()
    grid = dataclasses.replace(GRID, run_test=run_test, stage1_log_loss_tolerance=1e9,
                               stage1_slope_band=(1e-9, 1e9), stage2_skill_tolerance=1e9,
                               stage2_ratio_band=(1e-9, 1e9), min_gain=0.005)
    evaluation = SimpleNamespace(validation_origins=E2E_VALIDATION, test_origins=E2E_TEST,
                                 bootstrap_samples=40, bootstrap_seed=5, rule_doubtful_days=90)
    return E2ESettings(receivables=base.receivables, topups=base.topups, grid=grid,
                       evaluation=evaluation, calibration=base.calibration,
                       diagnostic=base.diagnostic, stage1=base.stage1, stage2=base.stage2,
                       data=SimpleNamespace(cache_folder=cache_root))


def e2e_inputs():
    from tests.hurdle.test_evaluate import make_weeks, rule_rates, running_ledger
    from tests.hurdle.test_forecast import (BLOCK_COLUMNS, F_COMPANIES, WEEK_ZERO, f_calendar,
                                            f_levels, f_side_tables, make_table)
    from tests.hurdle.test_grid import fit_weekly
    new_cash, refunds, unidentified = f_side_tables()
    return ModelInputs(companies=list(F_COMPANIES), week_zero=WEEK_ZERO, last_week=40,
                       weeks=make_weeks(40), weekly=fit_weekly(), ledger=running_ledger(),
                       rates=rule_rates(), levels=f_levels(), calendar=f_calendar(),
                       table=make_table(), block_columns=BLOCK_COLUMNS, new_cash=new_cash,
                       refunds=refunds, unidentified=unidentified)


def e2e_run(settings, inputs, **kwargs):
    """run_grid with test_grid.py's candidates; (what it returned, its printed lines)."""
    from tests.hurdle.test_grid import g_candidates
    printed = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(printed):
        patch.setattr(grid_run, "grid_candidates", g_candidates)
        returned = grid_run.run_grid(settings, inputs, **kwargs)
    return returned, printed.getvalue().splitlines()


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    from collection_estimation.hurdle.evaluate import rolling_forecasts
    cache_root = tmp_path_factory.mktemp("run_grid")
    settings, inputs = e2e_settings(cache_root), e2e_inputs()
    rolled = rolling_forecasts(inputs.table, list(E2E_VALIDATION), settings,
                               block_columns=inputs.block_columns, companies=inputs.companies,
                               new_cash=inputs.new_cash, refunds=inputs.refunds,
                               unidentified=inputs.unidentified, levels=inputs.levels,
                               calendar=inputs.calendar, week_zero=inputs.week_zero)
    returned, printed = e2e_run(settings, inputs, rolled=rolled)
    return SimpleNamespace(settings=settings, inputs=inputs, rolled=rolled, returned=returned,
                           printed=printed, cache_root=cache_root)


def test_run_grid_end_to_end_returns_a_grid_choice_and_a_grid_test(e2e):
    from collection_estimation.hurdle.grid import GridChoice, GridTest
    choice, test = e2e.returned
    assert isinstance(choice, GridChoice) and isinstance(test, GridTest)


def test_run_grid_end_to_end_candidate_0_is_step_17s_model(e2e):
    # show_grid_predictions and show_grid_scores check the current settings against rolled.
    assert "        candidate 0 the same as step 17, row by row: p yes, m yes" in e2e.printed
    assert any(line.startswith("        current WAPE the same as step 17's model: yes (")
               for line in e2e.printed), e2e.printed


def test_run_grid_end_to_end_prints_the_steps_in_order(e2e):
    lines = e2e.printed
    banner = ("        V1 TEST over the test origins 22-26 (2 origins): a report, not a choice; "
              "do not change the settings because of it")
    assert lines[0] == "Step 18: settings grid (U16) over the validation origins 8-14 (3 origins)"
    assert lines.count(banner) == 1
    assert TEST_NOT_RUN_LINE not in lines
    gates = [i for i, line in enumerate(lines) if line.startswith("        stage 1 gates:")]
    assert len(gates) == 1 and gates[0] < lines.index(banner)


def test_run_grid_end_to_end_the_choice_is_grid_choices_on_the_same_inputs(e2e):
    from collection_estimation.hurdle.evaluate import actual_totals
    from collection_estimation.hurdle.grid import (grid_choice, grid_gates, grid_predictions,
                                                   grid_scores)
    from tests.hurdle.test_grid import g_candidates
    settings, inputs = e2e.settings, e2e.inputs
    candidates = g_candidates(settings)
    # The same cache: nothing is fitted again.
    predictions = grid_predictions(
        inputs.table, list(E2E_VALIDATION), candidates, settings,
        block_columns=inputs.block_columns, companies=inputs.companies,
        new_cash=inputs.new_cash, refunds=inputs.refunds, unidentified=inputs.unidentified,
        levels=inputs.levels, calendar=inputs.calendar,
        cache_folder=e2e.cache_root / "hurdle_grid")
    actual = actual_totals(inputs.weekly, inputs.weeks, inputs.levels, inputs.companies,
                           list(E2E_VALIDATION), 5)
    gates = grid_gates(predictions, candidates, settings)
    table = grid_scores(predictions, candidates, gates, actual, settings,
                        companies=inputs.companies, week_zero=inputs.week_zero)
    by_hand = grid_choice(predictions, candidates, table, actual, settings,
                          companies=inputs.companies, week_zero=inputs.week_zero,
                          weekly=inputs.weekly, weeks=inputs.weeks, levels=inputs.levels,
                          calendar=inputs.calendar, ledger=inputs.ledger, rates=inputs.rates,
                          new_cash=inputs.new_cash, refunds=inputs.refunds,
                          unidentified=inputs.unidentified)
    choice, _ = e2e.returned
    pd.testing.assert_series_equal(choice.winner, by_hand.winner)
    assert (choice.stage1, choice.stage2, choice.topups) == (by_hand.stage1, by_hand.stage2,
                                                             by_hand.topups)
    pd.testing.assert_frame_equal(choice.scores, by_hand.scores)


def test_run_grid_end_to_end_the_test_is_grid_tests_at_the_test_origins(e2e):
    from collection_estimation.hurdle.evaluate import actual_totals
    from collection_estimation.hurdle.grid import grid_test
    settings, inputs = e2e.settings, e2e.inputs
    choice, test = e2e.returned
    actual = actual_totals(inputs.weekly, inputs.weeks, inputs.levels, inputs.companies,
                           list(E2E_TEST), 5)
    by_hand = grid_test(choice, actual, settings, table=inputs.table,
                        block_columns=inputs.block_columns, companies=inputs.companies,
                        week_zero=inputs.week_zero, weekly=inputs.weekly, weeks=inputs.weeks,
                        levels=inputs.levels, calendar=inputs.calendar, ledger=inputs.ledger,
                        rates=inputs.rates, new_cash=inputs.new_cash, refunds=inputs.refunds,
                        unidentified=inputs.unidentified,
                        cache_folder=e2e.cache_root / "hurdle_grid")
    pd.testing.assert_frame_equal(test.scores, by_hand.scores)
    assert sorted(set(test.forecast.totals["origin_week"])) == list(E2E_TEST)


def test_run_grid_end_to_end_caches_every_fit_under_hurdle_grid(e2e):
    folder = e2e.cache_root / "hurdle_grid"
    assert folder.is_dir()
    assert list(e2e.cache_root.iterdir()) == [folder]
    # 2 stage-1 and 2 stage-2 candidates at 3 validation origins, then the chosen pair at the
    # 2 test origins: 16 fits, each one cache file pair (when the chosen pair is not one the
    # grid fitted at those origins already; the grid fits validation origins only).
    assert len(list(folder.iterdir())) == 2 * (3 * (2 + 2) + 2 * (1 + 1))


def test_run_grid_end_to_end_without_run_test_returns_none_for_the_test(e2e, tmp_path):
    # The grid's fits are read from a copy of the cache of the run above: nothing new.
    shutil.copytree(e2e.cache_root / "hurdle_grid", tmp_path / "hurdle_grid")
    before = sorted(p.name for p in (tmp_path / "hurdle_grid").iterdir())
    (choice, test), lines = e2e_run(e2e_settings(tmp_path, run_test=False), e2e.inputs)
    assert test is None
    pd.testing.assert_series_equal(choice.winner, e2e.returned[0].winner)
    assert lines[-1] == TEST_NOT_RUN_LINE
    for line in ("        candidate 0 against step 17: not checked (step 17 did not forecast "
                 "with the model)",
                 "        current against step 17: not checked (step 17 did not forecast with "
                 "the model)"):
        assert line in lines, line
    assert sorted(p.name for p in (tmp_path / "hurdle_grid").iterdir()) == before
