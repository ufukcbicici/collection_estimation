"""Tests of `hurdle/reports/stages.py`: the printouts of steps 12-15 (stage 1, the stage 1
diagnostic, stage 2, the joint diagnostic's gates).

Each printout runs in a separate process (`run_snippet`) on a hand-made frame, every number
worked out by hand, or on what the package's own functions make of a tiny table. How main()
calls them is tested in tests/hurdle/test_run_hurdle_pipeline.py.
"""
import pytest

from tests.hurdle.code_checks import run_snippet

# Step 12 (U10.1): stage 1's rows and model input, on a small hand-built training table.
# The table: keys TR02-A (always paid) and TR04-B (never paid), origins 0..3, h = 1, 2, last
# week 3 (targets 4 and 5 are future). At origin 2 the training rows are (0,1), (0,2), (1,1)
# per key = 6 rows, half of them paid; the forecast rows are 2 keys x 2 h = 4 rows. Features:
# h, mean_receipts_4 (NaN for B), has_month_end (bool), open_invoices (int).
STAGE1 = """
import pandas as pd
from collection_estimation.hurdle import inputs as prep
from collection_estimation.hurdle.reports import stages as reports
from collection_estimation.hurdle.stage1 import (feature_columns, fit_stage1, fit_stage1_baseline,
                                                 predict_paid, stage1_features)
from collection_estimation.hurdle.stage2 import (amount_references, amount_scores, exposure,
                                                 fit_stage2, predict_amount, total_ratio)
from collection_estimation.hurdle.training_table import rows_at_origin, rows_known_at

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
columns = feature_columns(prep.block_feature_columns(history, calendar, owed),
                                 ("history", "calendar", "receivables"))
assert columns == ["h", "mean_receipts_4", "has_month_end", "open_invoices"], columns
origin = ORIGIN
train_rows = rows_known_at(table, origin)
forecast_rows = rows_at_origin(table, origin)
companies = ["TR02", "TR04", "TR05"]
train_x = stage1_features(train_rows, columns, companies, COMPANY_AS_FEATURE)
reports.show_stage1_rows(origin, train_rows, forecast_rows, train_x)
print("ok")
"""


def run_stage1(company=True, origin=2):
    # str.replace, not str.format: the code holds dict literals.
    code = STAGE1.replace("COMPANY_AS_FEATURE", str(company)).replace("ORIGIN", str(origin))
    return run_snippet(code)


def test_step_12_printout_runs_on_the_tiny_table():
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


def test_step_12_printout_with_no_training_rows_does_not_crash():
    # Origin 0: the earliest target is week 1, so nothing is known yet. The printout must
    # still run (main()'s guard only bounds demo_origin from above).
    result = run_stage1(origin=0)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("Step 12: stage 1 at forecast week 0: 0 training rows ")
    assert "4 forecast rows (origin 0, 2 customers x h)" in lines[0]
    assert lines[-1] == "ok"

MODEL_A = "[0.8, 0.4, 0.99, 0.6, 0.2, 0.01]"
MODEL_B = "[0.3, 0.9, 0.01, 0.6, 0.4, 0.99]"

SCORES = """
import numpy as np
import pandas as pd
from collection_estimation.hurdle.reports import stages as reports

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
returned = reports.show_stage1_scores(forecast_rows, predictions, train_rows, fit_seconds)
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
    code = SCORES.replace("MODELS", models_text).replace("FIT_SECONDS", repr(fit_seconds))
    return run_snippet(code)


def score_table(lines, n_rows):
    """The header words of the score table and its `n_rows` rows as {name: [log loss, Brier,
    AUC]}, in printed order."""
    header = lines[1].split()
    table = {}
    for line in lines[2:2 + n_rows]:
        parts = line.rsplit(maxsplit=3)
        table[parts[0].strip()] = [float(n) for n in parts[1:]]
    return header, table


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
forecast_x = stage1_features(forecast_rows, columns, companies, COMPANY_AS_FEATURE)
predictions = {}
for name, fit in (("stage 1 (HGB)", fit_stage1),
                  ("baseline (LR)", fit_stage1_baseline)):
    model = fit(train_x, train_rows["paid"], settings)
    predictions[name] = predict_paid(model, forecast_x)
    assert list(predictions[name].index) == list(forecast_rows.index)
    assert predictions[name].between(0, 1).all()
reports.show_stage1_scores(forecast_rows, predictions, train_rows,
                          {"stage 1 (HGB)": 0.5, "baseline (LR)": 0.25})
print("ok")
""")


@pytest.mark.parametrize("company", [True, False])
def test_step_12_real_fit_predict_and_scores_on_the_tiny_table(company):
    code = STAGE1_FIT.replace("COMPANY_AS_FEATURE", str(company)).replace("ORIGIN", "2")
    result = run_snippet(code)
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
from collection_estimation.hurdle.reports import stages as reports

forecast_rows = pd.DataFrame({"key": list("abcd"), "h": np.array([1, 1, 2, 2]),
                              "label_status": ["labelled"] * 3 + ["missing_week"],
                              "paid": pd.array([True, True, True, None], dtype="boolean")},
                             index=[5, 6, 7, 8])
predictions = {"stage 1 (HGB)": pd.Series([0.7, 0.3, 0.6, 0.2], index=forecast_rows.index),
               "baseline (LR)": pd.Series([0.5, 0.5, 0.5, 0.5], index=forecast_rows.index)}
train_rows = pd.DataFrame({"paid": pd.array([True, False], dtype="boolean")})
reports.show_stage1_scores(forecast_rows, predictions, train_rows,
                          {"stage 1 (HGB)": 0.1, "baseline (LR)": 0.1})
print("ok")
"""


def test_step_12_scores_with_one_class_among_labelled_rows_shows_nan_auc_and_runs():
    # sklearn 1.9's roc_auc_score warns and returns NaN for a single class; the printout
    # must still finish, for every model.
    result = run_snippet(ONE_CLASS)
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
from collection_estimation.hurdle.reports import stages as reports

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
returned = reports.show_stage1_scores(forecast_rows, predictions, train_rows, fit_seconds)
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
    code = (NO_LABELLED.replace("STATUSES", repr(NO_LABELLED_CASES[case]))
            .replace("FIT_SECONDS", repr(fit_seconds)))
    return run_snippet(code)


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
    code = NO_LABELLED.replace(
        "STATUSES", repr(["missing_week", "labelled", "future"])).replace(
        "FIT_SECONDS", repr({"stage 1 (HGB)": 1.0}))
    code = code.replace('"paid": pd.array([None] * len(statuses), dtype="boolean")',
                        '"paid": pd.array([None, True, None], dtype="boolean")')
    result = run_snippet(code)
    assert result.returncode == 0, result.stderr
    assert "no labelled forecast rows to score" not in result.stdout
    assert result.stdout.splitlines()[0] == (
        "        fitted in stage 1 (HGB) 1.0 s; scored on 1 forecast rows (100.0% paid)")


# Step 13 (U10.4b/c): the stage 1 diagnostic's printouts: the splits, the grid and its
# importances, the test table and the calibration.
STEP_13_SPLIT_TABLE = """
import numpy as np
import pandas as pd
from collection_estimation.hurdle.reports import stages as reports
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
reports.show_diagnostic_splits(diagnostic_splits(table, 2, 2), 2, 2)
print("ok")
"""

STEP_13_SPLIT_LINES = [
    "Step 13: stage 1 diagnostic at origin 2 (diagnostic only, never the forecast)",
    "        train              3 rows,  66.7% paid (target <= 2)",
    "        validation         2 rows,  50.0% paid (2 < target <= 4)",
    "        refit              5 rows,  60.0% paid (target <= 4)",
    "        test               2 rows, 100.0% paid (4 < target <= 6)",
]


def test_step_13_split_printout_by_hand():
    result = run_snippet(STEP_13_SPLIT_TABLE)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == STEP_13_SPLIT_LINES + ["ok"]


# show_diagnostic_grid / show_diagnostic_test on tiny hand-made frames.
SHOW = """
import numpy as np
import pandas as pd
from collection_estimation.hurdle.reports import stages as reports

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
    return run_snippet(SHOW.replace("N_GRID", str(n_grid))
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
    result = run_show("reports.show_diagnostic_grid('LR', grid, importance, 'avg precision')")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines == expected_grid_printout("LR", 7, 20) + ["ok"]


def test_show_diagnostic_grid_drops_the_candidate_column_and_rounds():
    result = run_show("reports.show_diagnostic_grid('HGB', grid, importance, 'avg precision')")
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
    result = run_show("reports.show_diagnostic_grid('LR', grid, importance, 'avg precision')",
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
            "reports.show_diagnostic_test(test)\n"
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
    result = run_show(f"reports.show_diagnostic_grid('HGB', grid, importance, {metric!r})")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == f"        HGB: the 5 best of 7 settings by validation {metric}"
    assert lines == expected_grid_printout("HGB", 7, 20, metric) + ["ok"]
    assert "average precision" not in lines[0]


def test_show_diagnostic_grid_metric_does_not_reorder_the_grid():
    # The printout shows the grid as given (grid_search already sorted it); the metric is only
    # named in the header. The hand-made grid has no log loss column at all.
    by_ap = run_show("reports.show_diagnostic_grid('LR', grid, importance, 'avg precision')")
    by_ll = run_show("reports.show_diagnostic_grid('LR', grid, importance, 'log loss')")
    assert by_ap.returncode == 0 and by_ll.returncode == 0, (by_ap.stderr, by_ll.stderr)
    assert by_ap.stdout.splitlines()[1:] == by_ll.stdout.splitlines()[1:]


def test_show_diagnostic_grid_needs_the_metric():
    result = run_show("reports.show_diagnostic_grid('LR', grid, importance)")
    assert result.returncode != 0
    assert "TypeError" in result.stderr


# --- show_calibration ----------------------------------------------------------------------------

SHOW_CALIBRATION = """
import numpy as np
import pandas as pd
from collection_estimation.hurdle.reports import stages as reports

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
reports.show_calibration(calibration)
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
    result = run_snippet(SHOW_CALIBRATION)
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
    result = run_snippet(SHOW_CALIBRATION)
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
    code = ("import pandas as pd\n"
            "from collection_estimation.hurdle.reports import stages as reports\n"
            "reports.show_calibration(pd.DataFrame({'part': ['validation'], 'model': ['M'], "
            "'h': ['all'], 'mean p': [0.1], 'share paid': [0.1], 'ECE': [0.0], 'slope': [1.0], "
            "'intercept': [0.0], 'log loss': [0.3], 'brier': [0.1], 'log loss skill': [0.0], "
            "'brier skill': [0.0], 'AUC': [0.5]}))\nprint('ok')\n")
    result = run_snippet(code)
    # Pinned behaviour: with no test rows the printout's tables are empty, nothing fails.
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "ok"


# show_stage2_fit on a hand-made frame. Paid amounts 1,000, 2,000, 3,000 and 1,000,000:
# median (2,000 + 3,000) / 2 = 2,500; mean 1,006,000 / 4 = 251,500; 99th percentile (linear,
# position 0.99 x 3 = 2.97) 3,000 + 0.97 x 997,000 = 970,090; max 1,000,000. 1,234 training
# rows, 5 input columns, 1.25 s shown as 1.2 s (one decimal).
SHOW_STAGE2 = """
import numpy as np
import pandas as pd
from collection_estimation.hurdle.reports import stages as reports

paid_rows = pd.DataFrame({"amount_deflated": [1_000.0, 2_000.0, 3_000.0, 1_000_000.0],
                          "amount": [9.0, 9.0, 9.0, 9.0]}, index=[3, 7, 8, 20])
train_rows = pd.DataFrame({"paid": np.zeros(1234, dtype=bool)})
paid_x = pd.DataFrame(np.ones((4, 5)), columns=list("abcde"), index=paid_rows.index)
copies = (paid_rows.copy(deep=True), train_rows.copy(deep=True), paid_x.copy(deep=True))
returned = reports.show_stage2_fit(40, paid_rows, train_rows, paid_x, LOSS, FIT_SECONDS)
assert returned is None
pd.testing.assert_frame_equal(paid_rows, copies[0])
pd.testing.assert_frame_equal(train_rows, copies[1])
pd.testing.assert_frame_equal(paid_x, copies[2])
print("ok")
"""


def run_show_stage2(loss="poisson", fit_seconds=1.25):
    return run_snippet(SHOW_STAGE2.replace("LOSS", repr(loss))
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
stage2_columns = feature_columns(prep.block_feature_columns(history, calendar, owed),
                                        settings_stage2.feature_blocks)
assert stage2_columns == ["h", "mean_receipts_4"], stage2_columns
paid_rows = train_rows[train_rows["paid"].astype(bool)]
paid_x = stage1_features(paid_rows, stage2_columns, companies, COMPANY_AS_FEATURE)
paid_exposure = exposure(train_rows, paid_rows, settings_stage2)
forecast_exposure = exposure(train_rows, forecast_rows, settings_stage2)
assert (paid_exposure == (1.0 if SCALE == "none" else 100.0)).all(), paid_exposure
started = time.perf_counter()
stage2_model = fit_stage2(paid_x, paid_rows["amount_deflated"], paid_exposure,
                                 settings_stage2)
reports.show_stage2_fit(origin, paid_rows, train_rows, paid_x, settings_stage2.loss,
                       time.perf_counter() - started)
reports.show_exposure(settings_stage2, paid_rows, paid_exposure, forecast_rows,
                     forecast_exposure)
assert len(paid_rows) == 3 and paid_rows["key"].eq("TR02-A").all()
m = stage2_model.predict(paid_x) * paid_exposure.to_numpy()
assert abs(m - 100.0).max() < 1e-6, m
# U11.2: the training ratio, m on the 4 forecast rows, and p x m with a hand-made p of 0.5.
training_ratio = total_ratio(stage2_model, paid_x, paid_rows["amount_deflated"],
                                    paid_exposure)
assert abs(training_ratio - 1.0) < 1e-9, training_ratio
forecast_m = predict_amount(stage2_model, stage1_features(
    forecast_rows, stage2_columns, companies, COMPANY_AS_FEATURE), forecast_exposure)
assert forecast_m.index.equals(forecast_rows.index) and forecast_m.name == "m"
p = pd.Series(0.5, index=forecast_rows.index, name="p")
reports.show_stage2_forecast(forecast_rows, forecast_m, p, training_ratio)
print("chain ok")
"""


def run_step_14_chain(company=True, loss="poisson", scale="none"):
    stage1_part = STAGE1.rsplit('print("ok")', 1)[0]
    code = (stage1_part + STEP_14_CHAIN).replace("COMPANY_AS_FEATURE", str(company)) \
        .replace("ORIGIN", "2").replace("LOSS", repr(loss)).replace("SCALE", repr(scale))
    return run_snippet(code)


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
from collection_estimation.hurdle.reports import stages as reports

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
returned = reports.show_stage2_forecast(forecast_rows, m, p, RATIO)
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
    return run_snippet(SHOW_FORECAST.replace("LABELS", repr(labels)).replace("RATIO", repr(ratio)))


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
    result = run_snippet(code)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[6].split() == ["2", "225.0", "210.0", "0.0", "inf"]


def test_show_stage2_forecast_uses_the_deflated_amount_not_the_nominal_one():
    # `amount` is 7 on every row; the actual cash column shows 300 / 100 / 400, never 14 or 28.
    result = run_show_forecast()
    assert result.returncode == 0, result.stderr
    assert "14.0" not in result.stdout and "28.0" not in result.stdout


# --- Step 14 (U11.3): show_stage2_scores -------------------------------------------------------
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
from collection_estimation.hurdle.reports import stages as reports
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
returned = reports.show_stage2_scores(scores)
assert returned is None
pd.testing.assert_frame_equal(scores, copy)
print("ok")
"""

SCORES_HEADER = ("        STAGE 2 alone, on the paid forecast rows (skill against the training "
                 "mean; R2's baseline is the rows' own mean):")


def run_show_scores(models=("stage 2", "training mean"), h=(2, 1, 2, 1)):
    return run_snippet(SHOW_SCORES.replace("MODELS", repr(list(models)))
                       .replace("HORIZONS", repr(list(h))))


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
    result = run_snippet(code)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    stage2 = lines[2].split()
    assert stage2[:5] == ["stage", "2", "4", "1", "inf"]
    assert stage2[5] == "-inf"


# The real chain on the tiny table, continued through the scores: at origin 2 the one paid
# labelled forecast row is TR02-A, h=1, amount 100; m = 100 there. The tiny table has no
# paying/observed weeks, so they are added here: window 4, 1 paying week of 2 observed, and
# mean_receipts_4 = 1.0 (from the table): own mean (2 + 100) / 2 = 51; training mean 100.
STEP_14_SCORES = """
paid_forecast = forecast_rows[(forecast_rows["label_status"] == "labelled")
                              & forecast_rows["paid"].fillna(False).astype(bool)]
assert list(paid_forecast["key"]) == ["TR02-A"] and list(paid_forecast["h"]) == [1]
paid_forecast = paid_forecast.assign(paying_weeks_4=1, observed_weeks_4=2)
amount_guesses = amount_references(train_rows, paid_forecast, 4, 1.0)
assert list(amount_guesses.iloc[0]) == [100.0, 51.0], amount_guesses
stage2_scores = amount_scores(paid_forecast["amount_deflated"], paid_forecast["h"],
                                     {"stage 2": forecast_m[paid_forecast.index],
                                      **amount_guesses},
                                     amount_guesses["training mean"])
reports.show_stage2_scores(stage2_scores)
print("scores ok")
"""


def test_step_14_real_chain_through_the_scores():
    stage1_part = STAGE1.rsplit('print("ok")', 1)[0]
    code = (stage1_part + STEP_14_CHAIN + STEP_14_SCORES).replace(
        "COMPANY_AS_FEATURE", "True").replace("ORIGIN", "2").replace("LOSS", "'poisson'") \
        .replace("SCALE", "'none'")
    result = run_snippet(code)
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
from collection_estimation.hurdle.reports import stages as reports

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
returned = reports.show_exposure(stage2, paid_rows, paid_exposure, forecast_rows,
                                forecast_exposure)
assert returned is None
pd.testing.assert_frame_equal(paid_rows, copies[0])
pd.testing.assert_series_equal(paid_exposure, copies[1])
pd.testing.assert_frame_equal(forecast_rows, copies[2])
pd.testing.assert_series_equal(forecast_exposure, copies[3])
print("ok")
"""


def run_show_exposure(scale, floor="own mean"):
    return run_snippet(SHOW_EXPOSURE.replace("SCALE", repr(scale)).replace("FLOOR", repr(floor)))


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
from collection_estimation.hurdle.reports import stages as reports
from collection_estimation.hurdle.diagnostic import amount_reliability, cash_by_p_group

actual = pd.Series([22.4, 12.0, 35.0, 24.0, 44.0, 35.0])
m = pd.Series([50.0, 10.0, 40.0, 20.0, 60.0, 30.0])
table = amount_reliability(actual, m, 3)
copy = table.copy(deep=True)
assert reports.show_amount_reliability(table) is None
pd.testing.assert_frame_equal(table, copy)
p = pd.Series([0.1, 0.4, 0.2, 0.3, 0.8, 0.6])
cash = cash_by_p_group(p, pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
                       pd.Series([0.0, 500.0, 0.0, 0.0, 1000.0, 0.0]), 3)
copy = cash.copy(deep=True)
assert reports.show_cash_by_p_group(cash) is None
pd.testing.assert_frame_equal(cash, copy)
print("ok")
"""


def test_show_amount_reliability_and_cash_by_p_group_printouts():
    result = run_snippet(SHOW_U11_4)
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


# --- The module itself --------------------------------------------------------------------------

STAGE_PRINTOUTS = ["show_stage1_rows", "show_stage1_scores", "show_diagnostic_splits",
                   "show_diagnostic_grid", "show_calibration", "show_diagnostic_test",
                   "show_stage2_fit", "show_exposure", "show_stage2_forecast",
                   "show_stage2_scores", "show_amount_reliability", "show_cash_by_p_group",
                   "show_gates", "show_joint_grid", "show_joint_results"]


def test_every_step_12_to_15_printout_is_a_function_of_the_module_in_step_order():
    # They used to be functions of run_hurdle_pipeline.py, in this order.
    from collection_estimation.hurdle.reports import stages
    from tests.hurdle.code_checks import HURDLE, module_functions
    assert [node.name for node in module_functions(HURDLE / "reports" / "stages.py")] == (
        STAGE_PRINTOUTS)
    for name in STAGE_PRINTOUTS:
        assert callable(getattr(stages, name)), name


def test_stage1_scores_are_sklearns_metrics():
    import sklearn.metrics
    from collection_estimation.hurdle.reports import stages
    assert stages.log_loss is sklearn.metrics.log_loss
    assert stages.brier_score_loss is sklearn.metrics.brier_score_loss
    assert stages.roc_auc_score is sklearn.metrics.roc_auc_score


def test_step_15_printouts_take_the_arguments_run_joint_diagnostic_gives_them():
    import inspect
    from collection_estimation.hurdle.reports import stages
    inspect.signature(stages.show_gates).bind("stage 1", "gates", "log loss", ascending=True)
    inspect.signature(stages.show_joint_grid).bind("pairs", "mape")
    inspect.signature(stages.show_joint_results).bind("tables", "chosen", 2000)
