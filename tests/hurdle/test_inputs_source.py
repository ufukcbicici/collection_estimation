"""Source checks of `hurdle/inputs.py`: what `prepare_inputs` calls, and in which order.

Steps 1-11 of the pipeline moved from run_hurdle_pipeline.py's main() into prepare_inputs; these
checks came with them (from test_run_hurdle_pipeline.py). They read prepare_inputs as a syntax
tree, without running it; what it computes is tested end to end in test_inputs.py and
test_ledger_starting_late.py.
"""
import ast

from tests.hurdle.code_checks import HURDLE, calls_to, function_source, function_tree

INPUTS_MODULE = HURDLE / "inputs.py"

# Every step 1-11 function main() used to call itself; now only prepare_inputs calls them.
STEPS_1_TO_11 = ("list_daily_files", "load_collections", "add_keys", "load_ledger",
                 "check_ledger_against_collections", "fill_missing_days", "add_week_index",
                 "load_holidays", "week_calendar", "weekly_amounts", "exchange_rates",
                 "receivables_by_week", "known_weeks", "is_unidentified", "panel_rows",
                 "add_labels", "new_customer_cash", "refunds_by_week",
                 "unidentified_cash_by_week", "check_side_tables", "week_numbers",
                 "published_levels", "ledger_receipts_before", "first_history_week",
                 "weekly_matrix", "history_features", "calendar_features",
                 "holiday_position_features", "receivables_features", "add_deflated_label",
                 "join_features", "block_feature_columns")


def prepare_tree():
    return function_tree(INPUTS_MODULE, "prepare_inputs")


def prepare_source():
    return function_source(INPUTS_MODULE, "prepare_inputs")


def assignments_to(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]


def is_min_with_zero(value, first_arg_check):
    """value is `min(<first>, 0)` with exactly two positional arguments."""
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "min" and len(value.args) == 2 and not value.keywords
            and first_arg_check(value.args[0])
            and isinstance(value.args[1], ast.Constant) and value.args[1].value == 0)




def test_prepare_inputs_calls_each_step_1_to_11_function():
    tree = prepare_tree()
    for name in STEPS_1_TO_11:
        assert calls_to(tree, name), name


def test_inputs_module_prints_nothing():
    module = ast.parse(INPUTS_MODULE.read_text(encoding="utf-8"))
    assert [node for node in ast.walk(module) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "print"] == []


# Steps 10 and 11 with a ledger that starts AFTER the first daily file (week 0): the price
# range and the history start are floored at week 0, so published_levels() (normalised to
# week 0) and weekly_matrix() (which must hold the daily files' weeks from 0) do not raise.


def test_step_10_first_ledger_week_is_the_week_of_the_first_posting():
    source = prepare_source()
    assert ('first_ledger_week = int(week_numbers(pd.Series([ledger["posting_date"].min()]), '
            'week_zero).iloc[0])') in source
    assert len(assignments_to(prepare_tree(), "first_ledger_week")) == 1


def test_step_10_first_price_week_is_floored_at_week_zero():
    found = assignments_to(prepare_tree(), "first_price_week")
    assert len(found) == 1, "first_price_week must be assigned exactly once"
    assert is_min_with_zero(found[0].value,
                            lambda arg: isinstance(arg, ast.Name)
                            and arg.id == "first_ledger_week"), ast.dump(found[0].value)
    assert "first_price_week = min(first_ledger_week, 0)" in prepare_source()


def test_step_10_published_levels_takes_the_floored_week():
    calls = calls_to(prepare_tree(), "published_levels")
    assert len(calls) == 1
    args = [ast.unparse(arg) for arg in calls[0].args]
    assert args == ["week_zero", "first_price_week", "last_week"], args
    assert ("levels = published_levels(week_zero, first_price_week, last_week)"
            in prepare_source())


def test_step_10_details_keep_the_floored_first_price_week():
    assert "first_price_week=first_price_week" in prepare_source()


def history_switch():
    """The `if settings.history.use_ledger_before_first_file:` statement in prepare_inputs."""
    found = [node for node in ast.walk(prepare_tree()) if isinstance(node, ast.If)
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
    assert "history_start = min(first_history_week(ledger, week_zero), 0)" in prepare_source()


def test_step_11_switched_off_history_start_is_still_zero():
    switch = history_switch()
    found = [node for statement in switch.orelse for node in assignments_to(statement,
                                                                            "history_start")]
    assert len(found) == 1
    assert isinstance(found[0].value, ast.Constant) and found[0].value.value == 0


def test_step_11_history_start_is_assigned_only_in_the_switch():
    assert len(assignments_to(prepare_tree(), "history_start")) == 2


def test_step_11_switched_off_early_receipts_are_emptied():
    switch = history_switch()
    found = [node for statement in switch.orelse for node in assignments_to(statement,
                                                                            "early_receipts")]
    assert len(found) == 1
    assert ast.unparse(found[0].value) == "early_receipts.iloc[0:0]"


def test_step_11_early_receipts_are_read_once_before_the_switch():
    calls = calls_to(prepare_tree(), "ledger_receipts_before")
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args] == ["ledger", "min(daily_files)",
                                                          "week_zero"]
    source = prepare_source()
    assert source.index("early_receipts = ledger_receipts_before(") < source.index(
        "if settings.history.use_ledger_before_first_file:")


def test_step_11_early_receipts_are_cut_at_the_history_start_and_the_matrix_starts_there():
    source = prepare_source()
    assert 'early_receipts = early_receipts[early_receipts["week"] >= history_start]' in source
    calls = calls_to(prepare_tree(), "weekly_matrix")
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
