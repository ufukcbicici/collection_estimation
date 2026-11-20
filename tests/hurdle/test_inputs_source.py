"""Source checks of `hurdle/inputs.py`: what `prepare_inputs` calls, and in which order.

Steps 1-11 of the pipeline moved from run_hurdle_pipeline.py's main() into prepare_inputs; these
checks came with them (from test_run_hurdle_pipeline.py). They read prepare_inputs as a syntax
tree, without running it; what it computes is tested end to end in test_inputs.py and
test_ledger_starting_late.py.
"""
import ast

from tests.hurdle.code_checks import (
    HURDLE,
    calls_to,
    function_source,
    function_tree,
    keywords_of,
)

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
                 "expected_dues", "payment_habits", "series_features", "check_series_no_leak",
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


# --- step 11b (U18.4): the rhythm blocks, dues and habits ----------------------------------
# Built from the panel with the [rhythm] / [receivables] / [history] settings, after the
# receivables features and before the join, and passed to join_features and
# block_feature_columns as their last two arguments.

def only_call(name):
    calls = calls_to(prepare_tree(), name)
    assert len(calls) == 1, (name, len(calls))
    return calls[0]


def test_step_11b_inputs_module_imports_both_from_features_rhythm():
    module = ast.parse(INPUTS_MODULE.read_text(encoding="utf-8"))
    found = [node for node in module.body if isinstance(node, ast.ImportFrom)
             and node.module == "features_rhythm" and node.level == 1]
    assert len(found) == 1
    assert sorted(alias.name for alias in found[0].names) == ["expected_dues", "payment_habits"]


def test_step_11b_expected_dues_positional_arguments():
    call = only_call("expected_dues")
    assert [ast.unparse(arg) for arg in call.args] == [
        "panel[['key', 'company_code', 'origin_week', 'h']]", "ledger", "rates", "levels",
        "week_zero", "holidays"]


def test_step_11b_expected_dues_keywords_come_from_the_settings():
    assert keywords_of(only_call("expected_dues")) == {
        "horizons": "horizons",
        "min_cleared": "settings.receivables.min_cleared_for_delay",
        "delays_kept": "settings.rhythm.delays_kept",
        "company_delays_kept": "settings.rhythm.company_delays_kept",
        "leak_check_weeks": "settings.receivables.leak_check_weeks",
    }


def test_step_11b_payment_habits_positional_arguments():
    call = only_call("payment_habits")
    assert [ast.unparse(arg) for arg in call.args] == [
        "panel[['key', 'origin_week', 'h']]", "collections", "ledger", "week_zero",
        "holidays", "min(daily_files)"]


def test_step_11b_payment_habits_keywords_come_from_the_settings():
    assert keywords_of(only_call("payment_habits")) == {
        "use_ledger_before_first_file": "settings.history.use_ledger_before_first_file",
        "horizons": "horizons",
        "payment_days_kept": "settings.rhythm.payment_days_kept",
        "min_payment_days": "settings.rhythm.min_payment_days",
        "leak_check_weeks": "settings.receivables.leak_check_weeks",
    }


def test_step_11b_calls_match_the_function_signatures():
    import inspect

    from collection_estimation.hurdle.features_rhythm import expected_dues, payment_habits
    for name, function in (("expected_dues", expected_dues),
                           ("payment_habits", payment_habits)):
        call = only_call(name)
        inspect.signature(function).bind(*[ast.unparse(a) for a in call.args],
                                         **keywords_of(call))


def test_step_11b_results_are_bound_to_dues_and_habits_once():
    for name, function in (("dues", "expected_dues"), ("habits", "payment_habits")):
        found = assignments_to(prepare_tree(), name)
        assert len(found) == 1, name
        value = found[0].value
        assert isinstance(value, ast.Call) and ast.unparse(value.func) == function, name


def test_step_11b_is_built_after_the_receivables_and_before_the_join():
    source = prepare_source()
    order = [source.index("owed = receivables_features("),
             source.index("dues = expected_dues("),
             source.index("habits = payment_habits("),
             source.index("panel = add_deflated_label(panel, levels)"),
             source.index("table = join_features(")]
    assert order == sorted(order), order
    # Both use the panel before its deflated label is added (the panel's key columns only),
    # and the collections after the fill and the week index.
    assert source.index("collections = add_week_index(") < source.index(
        "habits = payment_habits(")


def test_step_11b_join_features_gets_the_two_blocks_then_the_series():
    # U19.2: the series block (step 11c) comes after the two rhythm blocks.
    call = only_call("join_features")
    assert [ast.unparse(arg) for arg in call.args] == [
        "panel", "history", "calendar", "owed", "dues", "habits", "series"]
    assert call.keywords == []


def test_step_11b_block_feature_columns_gets_the_two_blocks_then_the_series():
    call = only_call("block_feature_columns")
    assert [ast.unparse(arg) for arg in call.args] == [
        "history", "calendar", "owed", "dues", "habits", "series"]
    assert call.keywords == []


def test_step_11b_details_keep_the_two_blocks():
    found = [node for node in calls_to(prepare_tree(), "PreparationDetails")]
    assert len(found) == 1
    keywords = keywords_of(found[0])
    assert keywords["dues"] == "dues"
    assert keywords["habits"] == "habits"


def test_step_11b_model_inputs_block_columns_come_from_block_feature_columns():
    found = calls_to(prepare_tree(), "ModelInputs")
    assert len(found) == 1
    assert keywords_of(found[0])["block_columns"] == (
        "block_feature_columns(history, calendar, owed, dues, habits, series)")


# --- step 11c (U19.2): the series block -------------------------------------------------------
# Built at every panel origin with [series] mean_weeks, checked against a rebuild from the
# data cut off at the [receivables] leak_check_weeks, then joined last.

def test_step_11c_inputs_module_imports_both_from_features_series():
    module = ast.parse(INPUTS_MODULE.read_text(encoding="utf-8"))
    found = [node for node in module.body if isinstance(node, ast.ImportFrom)
             and node.module == "features_series" and node.level == 1]
    assert len(found) == 1
    assert sorted(alias.name for alias in found[0].names) == ["check_series_no_leak",
                                                              "series_features"]


def test_step_11c_origins_are_every_panel_origin_sorted():
    found = assignments_to(prepare_tree(), "origins")
    assert len(found) == 1
    assert ast.unparse(found[0].value) == "sorted(set(panel['origin_week']))"


def test_step_11c_series_features_arguments():
    call = only_call("series_features")
    assert [ast.unparse(arg) for arg in call.args] == [
        "weekly", "weeks", "levels", "calendar", "companies", "origins", "horizons",
        "settings.series.mean_weeks"]
    assert call.keywords == []


def test_step_11c_check_series_no_leak_arguments():
    call = only_call("check_series_no_leak")
    assert [ast.unparse(arg) for arg in call.args] == [
        "series", "weekly", "weeks", "levels", "calendar", "companies", "horizons",
        "settings.series.mean_weeks", "settings.receivables.leak_check_weeks"]
    assert call.keywords == []


def test_step_11c_calls_match_the_function_signatures():
    import inspect

    from collection_estimation.hurdle.features_series import (
        check_series_no_leak, series_features)
    for name, function in (("series_features", series_features),
                           ("check_series_no_leak", check_series_no_leak)):
        call = only_call(name)
        inspect.signature(function).bind(*[ast.unparse(a) for a in call.args],
                                         **keywords_of(call))


def test_step_11c_series_is_bound_once_to_series_features():
    found = assignments_to(prepare_tree(), "series")
    assert len(found) == 1
    value = found[0].value
    assert isinstance(value, ast.Call) and ast.unparse(value.func) == "series_features"


def test_step_11c_is_built_and_checked_after_the_rhythm_blocks_and_before_the_join():
    source = prepare_source()
    order = [source.index("habits = payment_habits("),
             source.index("origins = sorted(set(panel["),
             source.index("series = series_features("),
             source.index("check_series_no_leak("),
             source.index("panel = add_deflated_label(panel, levels)"),
             source.index("table = join_features(")]
    assert order == sorted(order), order


def test_step_11c_details_keep_the_series():
    found = calls_to(prepare_tree(), "PreparationDetails")
    assert len(found) == 1
    assert keywords_of(found[0])["series"] == "series"


def test_step_11c_preparation_details_has_a_series_field_last():
    import dataclasses

    from collection_estimation.hurdle.inputs import PreparationDetails
    names = [field.name for field in dataclasses.fields(PreparationDetails)]
    assert names[-1] == "series"
    assert names.index("series") == names.index("habits") + 1
