"""Tests of `hurdle/reports/grid.py`: the printouts of step 18 (the settings grid, U16).

The printouts are called in this process on hand-made objects (they print only, and the
module does not reconfigure sys.stdout); show_grid_choice also runs through the real module
in a separate process (`run_snippet`) on a real GridChoice. Their definitions are checked on
the module's syntax tree. How run_grid calls them is tested in tests/hurdle/test_grid_run.py.
"""
import ast
from types import SimpleNamespace

import pytest

from collection_estimation.hurdle.reports.grid import (
    show_grid_candidates,
    show_grid_gates,
    show_grid_predictions,
    show_grid_scores,
)
from tests.hurdle.code_checks import HURDLE, module_functions, run_snippet

GRID_REPORTS = HURDLE / "reports" / "grid.py"




def grid_functions():
    return module_functions(GRID_REPORTS)


def module_imports():
    module = ast.parse(GRID_REPORTS.read_text(encoding="utf-8"))
    return {a.asname or a.name for node in module.body if isinstance(node, ast.ImportFrom)
            for a in node.names}


def test_the_printouts_are_the_modules_functions_in_step_order():
    # Step 18's printouts, in the order run_grid calls them; nothing else is defined here.
    assert [node.name for node in grid_functions()] == [
        "show_grid_candidates", "show_grid_predictions", "show_grid_gates", "show_grid_scores",
        "show_grid_choice", "show_grid_test"]


def test_show_grid_predictions_is_a_module_level_function_with_two_parameters():
    found = [node for node in grid_functions() if node.name == "show_grid_predictions"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["predictions", "rolled"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None


def test_show_grid_candidates_is_a_module_level_function_with_two_parameters():
    found = [node for node in grid_functions() if node.name == "show_grid_candidates"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["candidates", "origins"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None


# show_grid_predictions, the function alone.

def grid_predictions_example(n=1500, origins=(37, 40)):
    """GridPredictions over `origins` with n rows each: 2 stage-1, 3 stage-2 and 2 top-up
    candidates, and rolled forecasts whose customers equal candidate 0 row by row."""
    import numpy as np
    import pandas as pd
    from collection_estimation.hurdle.grid import GridPredictions
    rows_count = n * len(origins)
    index = pd.Index(5000 + np.arange(rows_count)[::-1])
    rows = pd.DataFrame({"origin_week": np.repeat(origins, n)}, index=index)
    p = pd.DataFrame({0: np.full(rows_count, 0.25), 1: np.full(rows_count, 0.5)}, index=index)
    p.iloc[0, 0] = 0.85
    m = pd.DataFrame({0: np.full(rows_count, 400000.0), 1: np.full(rows_count, 1e6),
                      2: np.full(rows_count, 0.0)}, index=index)
    m.iloc[0, 0] = 35367890.0
    topups = [pd.DataFrame({"topup": [1.5e6, 2.5e6, -1.0e6]}),
              pd.DataFrame({"topup": [123456.0, 0.0]})]
    predictions = GridPredictions(rows=rows, p=p, m=m, topups=topups)
    customers = pd.DataFrame({"p": p[0].to_numpy(), "m": m[0].to_numpy()}, index=index.copy())
    return predictions, SimpleNamespace(customers=customers)


GRID_HEAD = [
    "        3,000 customer rows at 2 origins",
    "        stage 1 setting 0: mean p 0.2502",
    "        stage 1 setting 1: mean p 0.5000",
    # (2999 x 400000 + 35367890) / 1e6 = 1,235.0 M
    "        stage 2 setting 0: sum of m 1,235.0 M (TRY of week 0)",
    "        stage 2 setting 1: sum of m 3,000.0 M (TRY of week 0)",
    "        stage 2 setting 2: sum of m 0.0 M (TRY of week 0)",
    "        top-ups setting 0: sum 3.0 M (TRY of week 0)",
    "        top-ups setting 1: sum 0.1 M (TRY of week 0)",
]
GRID_SAME = "        candidate 0 the same as step 17, row by row: p {}, m {}"
GRID_NOT_CHECKED = ("        candidate 0 against step 17: not checked (step 17 did not forecast "
                    "with the model)")


def printed_predictions(capsys, predictions, rolled):
    capsys.readouterr()
    assert show_grid_predictions(predictions, rolled) is None
    return capsys.readouterr().out.splitlines()


def test_show_grid_predictions_exact_printout_when_equal(capsys):
    predictions, rolled = grid_predictions_example()
    assert printed_predictions(capsys, predictions, rolled) == GRID_HEAD + [
        GRID_SAME.format("yes", "yes")]


def test_show_grid_predictions_exact_printout_when_rolled_is_none(capsys):
    predictions, _ = grid_predictions_example()
    assert printed_predictions(capsys, predictions, None) == GRID_HEAD + [GRID_NOT_CHECKED]


def next_up(frame, row, column):
    import numpy as np
    frame = frame.copy()
    frame.iloc[row, column] = np.nextafter(frame.iloc[row, column], np.inf)
    return frame


@pytest.mark.parametrize("change, p, m", [
    ("p one ulp off", "NO", "yes"),
    ("m one ulp off", "yes", "NO"),
    ("both off", "NO", "NO"),
    ("customers index shifted", "NO", "NO"),
    ("customers index reordered", "NO", "NO"),
    ("customers one row short", "NO", "NO"),
    ("customers one row more", "NO", "NO"),
])
def test_show_grid_predictions_exact_printout_when_different(capsys, change, p, m):
    # Exact, row by row: one ulp is a difference, and any other index (labels, order or
    # length) is "NO" for both, without raising.
    import pandas as pd
    predictions, rolled = grid_predictions_example()
    customers = rolled.customers
    if change == "p one ulp off":
        customers = next_up(customers, 700, 0)
    elif change == "m one ulp off":
        customers = next_up(customers, 3, 1)
    elif change == "both off":
        customers = next_up(next_up(customers, 1, 0), 2, 1)
    elif change == "customers index shifted":
        customers = customers.set_axis(customers.index + 1)
    elif change == "customers index reordered":
        customers = customers.iloc[::-1]
    elif change == "customers one row short":
        customers = customers.iloc[:-1]
    elif change == "customers one row more":
        customers = pd.concat([customers, customers.iloc[:1].set_axis([1])])
    lines = printed_predictions(capsys, predictions, SimpleNamespace(customers=customers))
    assert lines == GRID_HEAD + [GRID_SAME.format(p, m)]


def test_show_grid_predictions_compares_candidate_0_only(capsys):
    # Other candidates differ from rolled everywhere; only column 0 is compared.
    predictions, rolled = grid_predictions_example()
    assert not (predictions.p[1] == rolled.customers["p"]).all()
    assert printed_predictions(capsys, predictions, rolled)[-1] == GRID_SAME.format("yes", "yes")


def test_show_grid_predictions_extra_columns_of_customers_do_not_matter(capsys):
    predictions, rolled = grid_predictions_example()
    customers = rolled.customers.assign(expected=1.0, key="k")
    lines = printed_predictions(capsys, predictions, SimpleNamespace(customers=customers))
    assert lines[-1] == GRID_SAME.format("yes", "yes")


def test_show_grid_predictions_with_one_candidate_each_and_no_topup(capsys):
    import pandas as pd
    from collection_estimation.hurdle.grid import GridPredictions
    index = pd.Index([7, 3])
    predictions = GridPredictions(rows=pd.DataFrame({"origin_week": [40, 40]}, index=index),
                                  p=pd.DataFrame({0: [0.1, 0.3]}, index=index),
                                  m=pd.DataFrame({0: [2e6, 3e6]}, index=index), topups=[])
    assert printed_predictions(capsys, predictions, None) == [
        "        2 customer rows at 1 origins",
        "        stage 1 setting 0: mean p 0.2000",
        "        stage 2 setting 0: sum of m 5.0 M (TRY of week 0)",
        GRID_NOT_CHECKED]


def test_show_grid_predictions_shows_a_nan_p_without_raising(capsys):
    # p and m are not checked by the grid (the gates do it): the printout shows them as they are.
    import numpy as np
    predictions, rolled = grid_predictions_example(n=2)
    predictions.p[1] = np.nan
    lines = printed_predictions(capsys, predictions, rolled)
    assert lines[2] == "        stage 1 setting 1: mean p nan"
    assert lines[-1] == GRID_SAME.format("yes", "yes")


def test_show_grid_predictions_changes_nothing(capsys):
    predictions, rolled = grid_predictions_example(n=3)
    before = (predictions.rows.copy(), predictions.p.copy(), predictions.m.copy(),
              [t.copy() for t in predictions.topups], rolled.customers.copy())
    printed_predictions(capsys, predictions, rolled)
    import pandas as pd
    pd.testing.assert_frame_equal(predictions.rows, before[0])
    pd.testing.assert_frame_equal(predictions.p, before[1])
    pd.testing.assert_frame_equal(predictions.m, before[2])
    for a, b in zip(predictions.topups, before[3]):
        pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(rolled.customers, before[4])


# show_grid_candidates, the function alone.

def grid_printer():
    return show_grid_candidates


def printed_grid(capsys, candidates, origins):
    capsys.readouterr()
    grid_printer()(candidates, origins)
    return capsys.readouterr().out.splitlines()


def test_show_grid_candidates_exact_printout_of_the_worked_example(capsys):
    from collection_estimation.hurdle.grid import grid_candidates
    from tests.hurdle.test_grid import make
    lines = printed_grid(capsys, grid_candidates(make()), 3)
    assert lines == [
        "        stage 1: 4 settings",
        "          0: max_iter 300, learning_rate 0.05, max_leaf_nodes 31   <- current",
        "          1: max_iter 300, learning_rate 0.01, max_leaf_nodes 31",
        "          2: max_iter 900, learning_rate 0.01, max_leaf_nodes 31",
        "          3: max_iter 900, learning_rate 0.05, max_leaf_nodes 31",
        "        stage 2: 4 settings",
        "          0: loss poisson, max_iter 300, learning_rate 0.05, max_leaf_nodes 31, "
        "scale open balance, balance_floor own mean   <- current",
        "          1: loss poisson, max_iter 300, learning_rate 0.1, max_leaf_nodes 31, "
        "scale open balance, balance_floor own mean",
        "          2: loss gamma, max_iter 300, learning_rate 0.05, max_leaf_nodes 31, "
        "scale open balance, balance_floor own mean",
        "          3: loss gamma, max_iter 300, learning_rate 0.1, max_leaf_nodes 31, "
        "scale open balance, balance_floor own mean",
        "        top-ups: 2 settings",
        "          0: mean of 13 weeks   <- current",
        "          1: mean of 8 weeks",
        # 4 x 4 x 2 = 32 pairs; 3 x (4 + 4) = 24 fits.
        "        32 combinations to score; per origin 4 stage-1 and 4 stage-2 fits; 3 "
        "origins: 24 fits (those made before are read from the cache)",
    ]


def test_show_grid_candidates_shows_a_ridge_and_a_scale_none_floor_as_a_dash(capsys):
    import dataclasses
    from collection_estimation.hurdle.grid import grid_candidates
    from tests.hurdle.test_grid import STAGE2, TOPUPS, make
    ridge = dataclasses.replace(TOPUPS, method="ridge")
    stage2 = dataclasses.replace(STAGE2, scale="none")
    candidates = grid_candidates(make(stage2=stage2, topups=ridge, topup_windows=(13,),
                                      stage1_max_iter=(300,), stage1_learning_rate=(0.05,),
                                      stage2_loss=("poisson",), stage2_learning_rate=(0.05,)))
    lines = printed_grid(capsys, candidates, 1)
    assert lines == [
        "        stage 1: 1 settings",
        "          0: max_iter 300, learning_rate 0.05, max_leaf_nodes 31   <- current",
        "        stage 2: 2 settings",
        "          0: loss poisson, max_iter 300, learning_rate 0.05, max_leaf_nodes 31, "
        "scale none, balance_floor -   <- current",
        "          1: loss poisson, max_iter 300, learning_rate 0.05, max_leaf_nodes 31, "
        "scale open balance, balance_floor own mean",
        "        top-ups: 2 settings",
        "          0: ridge with the level   <- current",
        "          1: mean of 13 weeks",
        "        4 combinations to score; per origin 1 stage-1 and 2 stage-2 fits; 1 "
        "origins: 3 fits (those made before are read from the cache)",
    ]


def test_show_grid_candidates_top_up_lines_with_both_ridges(capsys):
    # The synthetic.toml shape: current mean 13, means 8 and 13, both ridges: 4 top-ups.
    from collection_estimation.hurdle.grid import grid_candidates
    from tests.hurdle.test_grid import make
    candidates = grid_candidates(make(topup_windows=(8, 13), topup_ridge_with_level=(True, False)))
    lines = printed_grid(capsys, candidates, 3)
    start = lines.index("        top-ups: 4 settings")
    assert lines[start:] == [
        "        top-ups: 4 settings",
        "          0: mean of 13 weeks   <- current",
        "          1: mean of 8 weeks",
        "          2: ridge with the level",
        "          3: ridge without the level",
        # 4 x 4 x 4 = 64 pairs; the ridges add combinations but no fit: still 24 fits.
        "        64 combinations to score; per origin 4 stage-1 and 4 stage-2 fits; 3 "
        "origins: 24 fits (those made before are read from the cache)",
    ]


@pytest.mark.parametrize("with_level, text", [(True, "ridge with the level"),
                                              (False, "ridge without the level")])
def test_show_grid_candidates_a_current_ridge_line(capsys, with_level, text):
    candidates = SimpleNamespace(
        stage1=[SimpleNamespace(params={"k": 0})], stage2=[SimpleNamespace(params={"k": 0})],
        topups=[SimpleNamespace(method="ridge", window=8, with_level=with_level),
                SimpleNamespace(method="ridge", window=8, with_level=not with_level)])
    lines = printed_grid(capsys, candidates, 1)
    other = "ridge without the level" if with_level else "ridge with the level"
    assert lines[-3:-1] == [f"          0: {text}   <- current", f"          1: {other}"]
    # The ridge's window is not shown.
    assert not any("8 weeks" in line for line in lines)


def fake_candidates(n1, n2, n3):
    """GridCandidates-like lists of the given sizes (only what the printout reads)."""
    stage = [SimpleNamespace(params={"k": i}) for i in range(max(n1, n2))]
    topups = [SimpleNamespace(method="mean", window=i + 1) for i in range(n3)]
    return SimpleNamespace(stage1=stage[:n1], stage2=stage[:n2], topups=topups)


@pytest.mark.parametrize("n1, n2, n3, origins, combos, fits", [
    (1, 1, 1, 1, 1, 2),
    (4, 8, 2, 12, 64, 144),
    (2, 3, 4, 5, 24, 25),
    (5, 1, 3, 2, 15, 12),
    (1, 1, 1, 0, 1, 0),
])
def test_show_grid_candidates_counts(capsys, n1, n2, n3, origins, combos, fits):
    last = printed_grid(capsys, fake_candidates(n1, n2, n3), origins)[-1]
    assert last == (f"        {combos} combinations to score; per origin {n1} stage-1 and {n2} "
                    f"stage-2 fits; {origins} origins: {fits} fits (those made before are read "
                    f"from the cache)")


def test_show_grid_candidates_fits_count_fits_not_combinations(capsys):
    # The top-ups cost no fit: more of them change the combinations, not the fits.
    a = printed_grid(capsys, fake_candidates(2, 2, 1), 3)[-1]
    b = printed_grid(capsys, fake_candidates(2, 2, 6), 3)[-1]
    assert "origins: 12 fits" in a and "origins: 12 fits" in b
    assert "4 combinations" in a and "24 combinations" in b


def test_show_grid_candidates_no_longer_shows_a_time(capsys):
    lines = printed_grid(capsys, fake_candidates(2, 2, 2), 3)
    assert not any("min" in line.split() for line in lines)
    assert "about" not in lines[-1]


def test_show_grid_candidates_marks_only_slot_0_in_each_list(capsys):
    lines = printed_grid(capsys, fake_candidates(3, 2, 4), 1)
    marked = [line for line in lines if line.endswith("   <- current")]
    assert marked == ["          0: k 0   <- current", "          0: k 0   <- current",
                      "          0: mean of 1 weeks   <- current"]


def test_show_grid_candidates_returns_none_and_changes_nothing(capsys):
    candidates = fake_candidates(2, 2, 2)
    before = (list(candidates.stage1), list(candidates.stage2), list(candidates.topups))
    capsys.readouterr()
    assert grid_printer()(candidates, 2) is None
    assert (candidates.stage1, candidates.stage2, candidates.topups) == before


def test_show_grid_gates_is_a_module_level_function_with_four_parameters():
    found = [node for node in grid_functions() if node.name == "show_grid_gates"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["stage", "gates", "score", "ascending"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None


# show_grid_gates, the function alone.

def gates_printer():
    return show_grid_gates


def printed_gates(capsys, stage, gates, score, ascending):
    capsys.readouterr()
    assert gates_printer()(stage, gates, score, ascending) is None
    return capsys.readouterr().out.splitlines()


def stage1_table(eligible_rows=(1, 2), current=None, n=4):
    """A stage-1 gate table as grid_gates returns it: candidate 0 fails all three gates (or
    the gates in `current`, a dict of gate -> passed), 1 and 2 pass, 3 has a bad p."""
    import numpy as np
    import pandas as pd
    base = [
        # stage1, max_iter, bad p, log loss, skill, slope
        (0, 300, 0, 0.693147, -0.000072, 0.0),
        (1, 100, 0, 0.582705, 0.159274, 0.971141),
        (2, 200, 0, 0.583829, 0.157652, 1.120210),
        (3, 400, 1, np.nan, np.nan, np.nan),
    ][:n]
    table = pd.DataFrame(base, columns=["stage1", "max_iter", "bad p", "log loss",
                                        "log loss skill", "slope"])
    for gate in ("log loss gate", "skill gate", "slope gate"):
        table[gate] = [i in eligible_rows for i in table["stage1"]]
    if current is not None:
        for gate, passed in current.items():
            table.loc[table["stage1"] == 0, gate] = passed
    table["eligible"] = table["log loss gate"] & table["skill gate"] & table["slope gate"]
    return table


def stage2_table():
    import numpy as np
    import pandas as pd
    table = pd.DataFrame(
        [(0, "poisson", 50, 0, 0.0, 1.0), (1, "poisson", 100, 0, 0.532846, 0.960442),
         (2, "gamma", 150, 0, 0.479335, 0.99328), (3, "poisson", 200, 2, np.nan, np.nan)],
        columns=["stage2", "loss", "max_iter", "bad m", "deviance skill", "total ratio"])
    table["m gate"] = [True, True, True, False]
    table["skill gate"] = [False, True, True, False]
    table["ratio gate"] = [True, True, True, False]
    table["eligible"] = table["m gate"] & table["skill gate"] & table["ratio gate"]
    return table


def gates_text(table, score, ascending):
    return table.sort_values(score, ascending=ascending).round(4).to_string(
        index=False).splitlines()


def first_tokens(lines):
    return [line.split()[0] for line in lines[1:]]     # skip the table's header line


STAGE1_HEAD = ("        stage 1 gates: 2 of 4 candidates eligible; candidates failing each "
               "gate: {'log loss gate': 2, 'skill gate': 2, 'slope gate': 2}")


def test_show_grid_gates_exact_printout_when_the_current_settings_fail(capsys):
    table = stage1_table()
    lines = printed_gates(capsys, "stage 1", table, "log loss", True)
    assert lines == [
        STAGE1_HEAD,
        "        the current stage 1 settings FAIL log loss gate, skill gate, slope gate; they "
        "are still scored, as the settings to beat",
    ] + gates_text(table, "log loss", True)


def test_show_grid_gates_table_sorted_by_the_score_with_unscored_candidates_last(capsys):
    lines = printed_gates(capsys, "stage 1", stage1_table(), "log loss", True)
    assert first_tokens(lines[2:]) == ["1", "2", "0", "3"]
    assert lines[2].split()[:4] == ["stage1", "max_iter", "bad", "p"]


def test_show_grid_gates_rounds_the_table_to_four_decimals(capsys):
    lines = printed_gates(capsys, "stage 1", stage1_table(), "log loss", True)
    assert "0.5827" in lines[3] and "0.582705" not in lines[3]


def test_show_grid_gates_names_only_the_gates_the_current_settings_miss(capsys):
    table = stage1_table(current={"log loss gate": True, "skill gate": True})
    lines = printed_gates(capsys, "stage 1", table, "log loss", True)
    assert lines[0] == ("        stage 1 gates: 2 of 4 candidates eligible; candidates failing "
                        "each gate: {'log loss gate': 1, 'skill gate': 1, 'slope gate': 2}")
    assert lines[1] == ("        the current stage 1 settings FAIL slope gate; they are still "
                        "scored, as the settings to beat")


def test_show_grid_gates_eligible_current_and_others_prints_no_warning(capsys):
    table = stage1_table(eligible_rows=(0, 1, 2))
    lines = printed_gates(capsys, "stage 1", table, "log loss", True)
    assert lines == [
        "        stage 1 gates: 3 of 4 candidates eligible; candidates failing each gate: "
        "{'log loss gate': 1, 'skill gate': 1, 'slope gate': 1}",
    ] + gates_text(table, "log loss", True)


def test_show_grid_gates_eligible_current_and_no_other_says_only_the_current_goes_on(capsys):
    table = stage1_table(eligible_rows=(0,))
    lines = printed_gates(capsys, "stage 1", table, "log loss", True)
    assert lines == [
        "        stage 1 gates: 1 of 4 candidates eligible; candidates failing each gate: "
        "{'log loss gate': 3, 'skill gate': 3, 'slope gate': 3}",
        "        no other stage 1 candidate is eligible: only the current settings go on",
    ] + gates_text(table, "log loss", True)


def test_show_grid_gates_nothing_eligible_prints_both_lines(capsys):
    table = stage1_table(eligible_rows=())
    lines = printed_gates(capsys, "stage 1", table, "log loss", True)
    assert lines[:3] == [
        "        stage 1 gates: 0 of 4 candidates eligible; candidates failing each gate: "
        "{'log loss gate': 4, 'skill gate': 4, 'slope gate': 4}",
        "        the current stage 1 settings FAIL log loss gate, skill gate, slope gate; they "
        "are still scored, as the settings to beat",
        "        no other stage 1 candidate is eligible: only the current settings go on",
    ]
    assert lines[3:] == gates_text(table, "log loss", True)


def test_show_grid_gates_a_lone_eligible_current_prints_no_no_other_line(capsys):
    # Only the current settings in the grid: there are no others to be ineligible.
    table = stage1_table(eligible_rows=(0,), n=1)
    lines = printed_gates(capsys, "stage 1", table, "log loss", True)
    assert lines == [
        "        stage 1 gates: 1 of 1 candidates eligible; candidates failing each gate: "
        "{'log loss gate': 0, 'skill gate': 0, 'slope gate': 0}",
    ] + gates_text(table, "log loss", True)
    assert not any(line.startswith("        no other") for line in lines)


def test_show_grid_gates_a_lone_failing_current_prints_the_fail_line_only(capsys):
    table = stage1_table(eligible_rows=(), n=1)
    lines = printed_gates(capsys, "stage 1", table, "log loss", True)
    assert lines == [
        "        stage 1 gates: 0 of 1 candidates eligible; candidates failing each gate: "
        "{'log loss gate': 1, 'skill gate': 1, 'slope gate': 1}",
        "        the current stage 1 settings FAIL log loss gate, skill gate, slope gate; they "
        "are still scored, as the settings to beat",
    ] + gates_text(table, "log loss", True)


def test_show_grid_gates_one_other_eligible_is_enough_for_no_no_other_line(capsys):
    table = stage1_table(eligible_rows=(0, 3))
    lines = printed_gates(capsys, "stage 1", table, "log loss", True)
    assert not any(line.startswith("        no other") for line in lines)


def test_show_grid_gates_finds_candidate_0_by_number_not_position(capsys):
    # Candidate 0 is the third row; it is still the one reported as the current settings.
    table = stage1_table(eligible_rows=(1, 2, 3)).iloc[[2, 1, 0, 3]].reset_index(drop=True)
    lines = printed_gates(capsys, "stage 1", table, "log loss", True)
    assert lines[1] == ("        the current stage 1 settings FAIL log loss gate, skill gate, "
                        "slope gate; they are still scored, as the settings to beat")
    eligible_current = stage1_table(eligible_rows=(0,)).iloc[[3, 1, 0, 2]]
    lines = printed_gates(capsys, "stage 1", eligible_current, "log loss", True)
    assert lines[1] == ("        no other stage 1 candidate is eligible: only the current "
                        "settings go on")


def test_show_grid_gates_exact_printout_for_stage_2_descending(capsys):
    table = stage2_table()
    lines = printed_gates(capsys, "stage 2", table, "deviance skill", False)
    assert lines == [
        "        stage 2 gates: 2 of 4 candidates eligible; candidates failing each gate: "
        "{'m gate': 1, 'skill gate': 2, 'ratio gate': 1}",
        "        the current stage 2 settings FAIL skill gate; they are still scored, as the "
        "settings to beat",
    ] + gates_text(table, "deviance skill", False)
    # Highest skill first; the candidate without a score last.
    assert first_tokens(lines[2:]) == ["1", "2", "0", "3"]


def test_show_grid_gates_counts_only_columns_ending_in_gate(capsys):
    # "eligible" and the params are not gates, even a param whose name holds "gate".
    table = stage2_table().assign(gates_used=1)
    head = printed_gates(capsys, "stage 2", table, "deviance skill", False)[0]
    assert head.endswith("{'m gate': 1, 'skill gate': 2, 'ratio gate': 1}")


def test_show_grid_gates_changes_nothing(capsys):
    import pandas as pd
    table = stage1_table()
    before = table.copy()
    printed_gates(capsys, "stage 1", table, "log loss", True)
    pd.testing.assert_frame_equal(table, before)


def test_show_grid_gates_on_grid_gates_own_output(capsys):
    # The worked example of test_grid.py (stage 1 {1, 2} and stage 2 {1, 2} eligible;
    # candidate 0 fails all three stage-1 gates and stage 2's skill gate).
    from collection_estimation.hurdle.grid import grid_gates
    from tests.hurdle.test_grid import gate_candidates, gate_example, gate_settings
    gates = grid_gates(gate_example(), gate_candidates(), gate_settings())
    one = printed_gates(capsys, "stage 1", gates.stage1, "log loss", True)
    two = printed_gates(capsys, "stage 2", gates.stage2, "deviance skill", False)
    assert one[0].startswith("        stage 1 gates: 2 of 4 candidates eligible;")
    assert one[1] == ("        the current stage 1 settings FAIL log loss gate, skill gate, "
                      "slope gate; they are still scored, as the settings to beat")
    assert first_tokens(one[2:]) == ["1", "2", "0", "3"]
    assert two[0].startswith("        stage 2 gates: 2 of 4 candidates eligible;")
    assert two[1] == ("        the current stage 2 settings FAIL skill gate; they are still "
                      "scored, as the settings to beat")
    assert first_tokens(two[2:]) == ["1", "2", "0", "3"]
    assert not any(line.startswith("        no other") for line in one + two)


def test_show_grid_gates_prints_bad_elsewhere_and_does_not_count_it_as_a_gate(capsys):
    # U16.5 (F2) gave both gate tables a "bad elsewhere" column; the printout shows it in the
    # table and leaves it out of the gate counts (its name does not end in "gate").
    from collection_estimation.hurdle.grid import grid_gates
    from tests.hurdle.test_grid import gate_candidates, gate_example, gate_settings
    gates = grid_gates(gate_example(poison_unlabelled=True), gate_candidates(), gate_settings())
    one = printed_gates(capsys, "stage 1", gates.stage1, "log loss", True)
    two = printed_gates(capsys, "stage 2", gates.stage2, "deviance skill", False)
    # Candidate 1 is refused (bad elsewhere): only candidate 2 passes stage 1.
    assert one[0] == ("        stage 1 gates: 1 of 4 candidates eligible; candidates failing "
                      "each gate: {'log loss gate': 3, 'skill gate': 3, 'slope gate': 3}")
    assert one[-5:] == gates_text(gates.stage1, "log loss", True)
    assert two[-5:] == gates_text(gates.stage2, "deviance skill", False)
    assert "elsewhere" in one[-5] and "elsewhere" in two[-5]


def test_show_grid_scores_is_a_module_level_function_with_five_parameters():
    found = [node for node in grid_functions() if node.name == "show_grid_scores"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["table", "candidates", "rolled", "actual",
                                                   "horizons"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None


SHOW_SCORES_GLOBALS = {"CONSOLIDATED", "HURDLE", "POOLED", "STACK_COLUMNS", "evaluation_scores"}


def test_show_grid_scores_reads_only_names_the_module_imports():
    import builtins
    function = [node for node in grid_functions() if node.name == "show_grid_scores"][0]
    loaded = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Load)}
    stored = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Store)}
    params = {a.arg for a in function.args.args}
    free = loaded - stored - params - set(dir(builtins))
    assert free == SHOW_SCORES_GLOBALS
    assert SHOW_SCORES_GLOBALS <= module_imports()


def test_show_grid_scores_reads_evaluates_and_forecasts_own_objects():
    from collection_estimation.hurdle import evaluate, forecast
    from collection_estimation.hurdle.reports import grid as grid_reports
    assert grid_reports.CONSOLIDATED is forecast.CONSOLIDATED
    assert grid_reports.HURDLE is evaluate.HURDLE and grid_reports.POOLED is evaluate.POOLED
    assert grid_reports.STACK_COLUMNS is evaluate.STACK_COLUMNS
    assert grid_reports.evaluation_scores is evaluate.evaluation_scores


# show_grid_scores, the function alone.

def scores_printer():
    return show_grid_scores


def grid_score_table(lines=None):
    """A grid_scores table (sorted by WAPE, index reset); the current settings are rank 2."""
    import pandas as pd
    from collection_estimation.hurdle.grid import GRID_SCORE_COLUMNS
    lines = lines or [(1, 0, 1, False, 0.150004, 0.101234, -0.020001, 0.200049),
                      (0, 0, 0, True, 0.1, 0.12, 0.031, 0.22),
                      (0, 1, 1, False, 0.170001, float("nan"), 0.0, 0.25)]
    return pd.DataFrame(lines, columns=GRID_SCORE_COLUMNS)


def score_candidates(n1=2, n2=3, n3=2):
    return SimpleNamespace(stage1=["s1"] * n1, stage2=["s2"] * n2, topups=["t"] * n3)


def score_actual():
    """Origin 10, h 1 and 2, one company: TR02 and ALL each 100 then 200 TRY."""
    import pandas as pd
    from collection_estimation.hurdle.evaluate import ACTUAL_COLUMNS
    rows = [(company, 10, h, 10 + h, actual, actual) for company in ("TR02", "ALL")
            for h, actual in ((1, 100.0), (2, 200.0))]
    return pd.DataFrame(rows, columns=ACTUAL_COLUMNS)


def score_rolled(first=110.0, second=180.0):
    """Step 17's totals: |110 - 100| + |180 - 200| = 30 over 300, WAPE 0.1."""
    import pandas as pd
    rows = [(company, 10, h, 10 + h, total, 99.0) for company in ("TR02", "ALL")
            for h, total in ((1, first), (2, second))]
    totals = pd.DataFrame(rows, columns=["company_code", "origin_week", "h", "target_week",
                                         "total", "customers"])
    return SimpleNamespace(totals=totals, customers="<not read>")


def printed_scores(capsys, table, candidates, rolled, actual=None, horizons=2):
    capsys.readouterr()
    assert scores_printer()(table, candidates, rolled,
                            score_actual() if actual is None else actual, horizons) is None
    return capsys.readouterr().out.splitlines()


SCORES_HEAD = ("        3 of 12 combinations scored (stage-1 and stage-2 candidates eligible "
               "or current, every top-up); the 10 best by WAPE:")
CURRENT_LINE = ("        current settings: rank 2 of 3, WAPE 0.1000, 5-week WAPE 0.1200, bias "
                "+0.0310, company WAPE 0.2200")
NOT_CHECKED = ("        current against step 17: not checked (step 17 did not forecast with "
               "the model)")


def table_text(table, rows=10):
    return table.head(rows).round(4).to_string(index=False).splitlines()


def test_show_grid_scores_exact_printout_when_rolled_is_none(capsys):
    table = grid_score_table()
    lines = printed_scores(capsys, table, score_candidates(), None)
    assert lines == [SCORES_HEAD] + table_text(table) + [CURRENT_LINE, NOT_CHECKED]


def test_show_grid_scores_exact_printout_when_the_current_equals_step_17(capsys):
    table = grid_score_table()
    lines = printed_scores(capsys, table, score_candidates(), score_rolled())
    assert lines == [SCORES_HEAD] + table_text(table) + [
        CURRENT_LINE,
        "        current WAPE the same as step 17's model: yes (step 17 0.100000, grid "
        "0.100000)"]


def test_show_grid_scores_a_different_step_17_says_no(capsys):
    lines = printed_scores(capsys, grid_score_table(), score_candidates(), score_rolled(first=130.0))
    # |130 - 100| + |180 - 200| = 50 over 300.
    assert lines[-1] == ("        current WAPE the same as step 17's model: NO (step 17 "
                         "0.166667, grid 0.100000)")


def test_show_grid_scores_compares_exactly_not_to_the_printed_digits(capsys):
    # A difference far below the 6 printed decimals is still NO: the check is exact.
    table = grid_score_table()
    table.loc[1, "wape"] = 0.1 + 1e-12
    lines = printed_scores(capsys, table, score_candidates(), score_rolled())
    assert lines[-1] == ("        current WAPE the same as step 17's model: NO (step 17 "
                         "0.100000, grid 0.100000)")


def test_show_grid_scores_compares_the_consolidated_pooled_wape_only(capsys):
    # A company-only error (TR02 h 1 off by 50, ALL right on its own) does not move the check:
    # step 17's consolidated WAPE is scored on ALL's rows, as the grid's is.
    totals = score_rolled().totals.copy()
    totals.loc[(totals["company_code"] == "TR02") & (totals["h"] == 1), "total"] = 160.0
    lines = printed_scores(capsys, grid_score_table(), score_candidates(),
                           SimpleNamespace(totals=totals))
    assert lines[-1].startswith("        current WAPE the same as step 17's model: yes")


def test_show_grid_scores_reads_only_the_total_of_step_17(capsys):
    rolled = score_rolled()
    rolled.totals["customers"] = float("nan")     # every other column is not read
    lines = printed_scores(capsys, grid_score_table(), score_candidates(), rolled)
    assert lines[-1].startswith("        current WAPE the same as step 17's model: yes")


def test_show_grid_scores_counts_every_combination_of_the_candidates(capsys):
    lines = printed_scores(capsys, grid_score_table(), score_candidates(4, 5, 3), None)
    assert lines[0] == ("        3 of 60 combinations scored (stage-1 and stage-2 candidates "
                        "eligible or current, every top-up); the 10 best by WAPE:")


def many_lines(n, current_at):
    lines = [(i + 1, 0, 0, False, 0.1 + 0.01 * i, 0.1, 0.0, 0.2) for i in range(n)]
    lines[current_at] = (0, 0, 0, True) + lines[current_at][4:]
    return lines


def test_show_grid_scores_shows_the_10_best_only(capsys):
    table = grid_score_table(many_lines(14, current_at=12))
    lines = printed_scores(capsys, table, score_candidates(5, 3, 1), None)
    assert lines[1:12] == table_text(table)
    assert len(table_text(table)) == 11            # the header and 10 rows
    assert lines[12].startswith("        current settings: rank 13 of 14, WAPE 0.2200")
    assert len(lines) == 14


def test_show_grid_scores_rank_1_when_the_current_is_best(capsys):
    table = grid_score_table(many_lines(4, current_at=0))
    lines = printed_scores(capsys, table, score_candidates(2, 2, 1), None)
    assert lines[-2].startswith("        current settings: rank 1 of 4, WAPE 0.1000")


def test_show_grid_scores_a_lone_current_combination(capsys):
    table = grid_score_table([(0, 0, 0, True, 0.25, 0.3, -0.05, 0.4)])
    lines = printed_scores(capsys, table, score_candidates(1, 1, 1), None)
    assert lines[0] == ("        1 of 1 combinations scored (stage-1 and stage-2 candidates "
                        "eligible or current, every top-up); the 10 best by WAPE:")
    assert lines[-2] == ("        current settings: rank 1 of 1, WAPE 0.2500, 5-week WAPE "
                         "0.3000, bias -0.0500, company WAPE 0.4000")


def test_show_grid_scores_a_missing_5_week_wape_prints_nan(capsys):
    table = grid_score_table([(0, 0, 0, True, 0.25, float("nan"), 0.0, 0.4)])
    lines = printed_scores(capsys, table, score_candidates(1, 1, 1), None)
    assert lines[-2] == ("        current settings: rank 1 of 1, WAPE 0.2500, 5-week WAPE "
                         "nan, bias +0.0000, company WAPE 0.4000")


def test_show_grid_scores_rounds_the_table_to_four_decimals(capsys):
    lines = printed_scores(capsys, grid_score_table(), score_candidates(), None)
    body = "\n".join(lines[1:5])
    assert "0.15" in body and "0.150004" not in body and "0.101234" not in body


def test_show_grid_scores_rolled_missing_a_row_with_actual_cash_raises(capsys):
    # Step 17's totals at another origin than the grid's: evaluation_scores refuses them.
    rolled = score_rolled()
    rolled.totals["origin_week"] = 11
    rolled.totals["target_week"] = 11 + rolled.totals["h"]
    with pytest.raises(ValueError, match="has no forecast for"):
        scores_printer()(grid_score_table(), score_candidates(), rolled, score_actual(), 2)


def test_show_grid_scores_changes_nothing(capsys):
    import pandas as pd
    table, rolled, actual = grid_score_table(), score_rolled(), score_actual()
    before = (table.copy(), rolled.totals.copy(), actual.copy())
    printed_scores(capsys, table, score_candidates(), rolled, actual)
    pd.testing.assert_frame_equal(table, before[0])
    pd.testing.assert_frame_equal(rolled.totals, before[1])
    pd.testing.assert_frame_equal(actual, before[2])


def test_show_grid_scores_on_grid_scores_own_output(capsys):
    # test_grid.py's hand-made example: 3 x 3 x 2 combinations, all scored.
    from tests.hurdle.test_grid import sc_candidates, sc_run
    table = sc_run()
    lines = printed_scores(capsys, table, sc_candidates(), None, horizons=5)
    assert lines[0] == ("        18 of 18 combinations scored (stage-1 and stage-2 candidates "
                        "eligible or current, every top-up); the 10 best by WAPE:")
    assert lines[1].split()[:4] == ["stage1", "stage2", "topups", "current"]
    rank = int(table.index[table["current"]][0]) + 1
    current = table[table["current"]].iloc[0]
    assert lines[-2] == (f"        current settings: rank {rank} of 18, WAPE "
                         f"{current['wape']:.4f}, 5-week WAPE {current['total_wape']:.4f}, bias "
                         f"{current['bias']:+.4f}, company WAPE {current['company_wape']:.4f}")
    assert lines[-1] == NOT_CHECKED


def test_show_grid_choice_is_a_module_level_function_with_four_parameters():
    found = [node for node in grid_functions() if node.name == "show_grid_choice"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["choice", "candidates", "table", "settings"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None


SHOW_CHOICE_GLOBALS = {"HURDLE", "show_calibration_choice", "show_evaluation_scores",
                       "show_paired_bootstrap"}


def test_show_grid_choice_reads_only_names_the_module_imports():
    import builtins
    function = [node for node in grid_functions() if node.name == "show_grid_choice"][0]
    loaded = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Load)}
    stored = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Store)}
    params = {a.arg for a in function.args.args}
    free = loaded - stored - params - set(dir(builtins))
    assert free == SHOW_CHOICE_GLOBALS
    # The three printers it calls are step 17's, imported from reports/evaluation.py.
    assert SHOW_CHOICE_GLOBALS <= module_imports()


def test_show_grid_choice_and_test_call_step_17s_own_printers():
    from collection_estimation.hurdle.reports import evaluation
    from collection_estimation.hurdle.reports import grid as grid_reports
    for name in ("show_calibration_choice", "show_evaluation_scores", "show_paired_bootstrap"):
        assert getattr(grid_reports, name) is getattr(evaluation, name), name


def test_show_grid_test_is_a_module_level_function_with_three_parameters():
    found = [node for node in grid_functions() if node.name == "show_grid_test"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["test", "choice", "settings"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None


def test_show_grid_test_sits_right_after_show_grid_choice():
    names = [node.name for node in grid_functions()]
    assert names[names.index("show_grid_choice") + 1] == "show_grid_test"


SHOW_TEST_GLOBALS = {"HURDLE", "UNCORRECTED", "candidate_name", "show_evaluation_scores",
                     "show_paired_bootstrap"}


def test_show_grid_test_reads_only_names_the_module_imports():
    import builtins
    function = [node for node in grid_functions() if node.name == "show_grid_test"][0]
    loaded = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Load)}
    stored = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Store)}
    params = {a.arg for a in function.args.args}
    assert loaded - stored - params - set(dir(builtins)) == SHOW_TEST_GLOBALS
    assert SHOW_TEST_GLOBALS <= module_imports()


# show_grid_test, the function alone: compiled from the module's source, with the two printers
# it calls replaced by fakes that print what they were given (each is tested on its own in
# test_evaluation.py).

def grid_test_printer(printed):
    from collection_estimation.hurdle import calibrate, evaluate, grid
    found = [node for node in grid_functions() if node.name == "show_grid_test"]
    assert len(found) == 1

    def fake(name):
        def show(*args):
            printed.append((name, args))
            print(f"<{name}>")
        return show

    namespace = {"HURDLE": evaluate.HURDLE, "UNCORRECTED": grid.UNCORRECTED,
                 "candidate_name": calibrate.candidate_name,
                 "show_evaluation_scores": fake("show_evaluation_scores"),
                 "show_paired_bootstrap": fake("show_paired_bootstrap")}
    exec(compile(ast.Module(body=found, type_ignores=[]), "<show_grid_test>", "exec"), namespace)
    return namespace["show_grid_test"]


def hand_test_choice(p_kind="none", m_kind="scale"):
    from collection_estimation.hurdle.calibrate import MCorrection, PCorrection
    from collection_estimation.hurdle.settings import TopupSettings
    topups = TopupSettings(method="mean", window=13, with_level=True, compare=False,
                           compare_first_origin=0, compare_last_origin=0, compare_windows=(4,))
    calibration = SimpleNamespace(p_correction=PCorrection(p_kind),
                                  m_correction=MCorrection(m_kind, 1.036))
    return SimpleNamespace(stage1=SimpleNamespace(params={"max_iter": 900, "learning_rate": 0.01}),
                           stage2=SimpleNamespace(params={"loss": "poisson",
                                                          "learning_rate": 0.05}),
                           topups=topups, calibration=calibration)


def printed_test(capsys, test, choice, settings):
    printed = []
    capsys.readouterr()
    assert grid_test_printer(printed)(test, choice, settings) is None
    return capsys.readouterr().out.splitlines(), printed


def test_show_grid_test_exact_printout(capsys):
    test = SimpleNamespace(scores="<sc>", bootstrap="<bs>")
    lines, _ = printed_test(capsys, test, hand_test_choice(),
                            SimpleNamespace(receivables=SimpleNamespace(horizons=5)))
    assert lines == [
        "        frozen on validation: stage 1 {'max_iter': 900, 'learning_rate': 0.01}, "
        "stage 2 {'loss': 'poisson', 'learning_rate': 0.05}, top-ups mean (window 13, "
        "with_level True); corrections p none, m scale",
        "        on the test origins (hurdle = corrected; hurdle uncorrected for information "
        "only):",
        "<show_evaluation_scores>",
        "<show_paired_bootstrap>"]


def test_show_grid_test_names_the_corrections_with_candidate_name(capsys):
    from collection_estimation.hurdle.calibrate import candidate_name
    choice = hand_test_choice(p_kind="per h", m_kind="by size")
    lines, _ = printed_test(capsys, SimpleNamespace(scores=1, bootstrap=2), choice,
                            SimpleNamespace(receivables=SimpleNamespace(horizons=5)))
    name = candidate_name(choice.calibration.p_correction, choice.calibration.m_correction)
    assert lines[0].endswith(f"; corrections {name}")
    assert name == "p per h, m by size"


def test_show_grid_test_passes_the_test_scores_horizons_and_bootstrap(capsys):
    test = SimpleNamespace(scores="<sc>", bootstrap="<bs>", forecast="<f>", uncorrected="<u>")
    _, printed = printed_test(capsys, test, hand_test_choice(),
                              SimpleNamespace(receivables=SimpleNamespace(horizons=4)))
    assert printed == [("show_evaluation_scores", ("<sc>", 4)),
                       ("show_paired_bootstrap", ("<bs>",))]


# show_grid_choice, the function alone: compiled from the module's source, with the printers it
# calls replaced by fakes that print what they were given (each is tested on its own in
# test_evaluation.py).

def choice_printer(printed):
    from collection_estimation.hurdle import evaluate
    found = [node for node in grid_functions() if node.name == "show_grid_choice"]
    assert len(found) == 1

    def fake(name):
        def show(*args):
            printed.append((name, args))
            print(f"<{name}>")
        return show

    namespace = {"HURDLE": evaluate.HURDLE,
                 "show_calibration_choice": fake("show_calibration_choice"),
                 "show_evaluation_scores": fake("show_evaluation_scores"),
                 "show_paired_bootstrap": fake("show_paired_bootstrap")}
    exec(compile(ast.Module(body=found, type_ignores=[]), "<show_grid_choice>", "exec"),
         namespace)
    return namespace["show_grid_choice"]


def choice_candidates():
    from collection_estimation.hurdle.settings import TopupSettings
    topups = TopupSettings(method="mean", window=13, with_level=True, compare=False,
                           compare_first_origin=0, compare_last_origin=0, compare_windows=(4,))
    return SimpleNamespace(
        stage1=[SimpleNamespace(params={"max_iter": 300, "learning_rate": 0.05}),
                SimpleNamespace(params={"max_iter": 900, "learning_rate": 0.05})],
        stage2=[SimpleNamespace(params={"loss": "poisson", "learning_rate": 0.05}),
                SimpleNamespace(params={"loss": "gamma", "learning_rate": 0.1})],
        topups=[topups, SimpleNamespace(method="mean", window=8),
                SimpleNamespace(method="ridge", with_level=False)])


def choice_scores_table(lines=None):
    """A grid_scores table: current (0, 0, 0) WAPE 0.1, rank 2 by WAPE."""
    import pandas as pd
    from collection_estimation.hurdle.grid import GRID_SCORE_COLUMNS
    lines = lines or [(1, 0, 1, False, 0.098, 0.1, 0.0, 0.2),
                      (0, 0, 0, True, 0.1, 0.12, 0.031, 0.22),
                      (0, 1, 2, False, 0.17, 0.2, 0.0, 0.25)]
    return pd.DataFrame(lines, columns=GRID_SCORE_COLUMNS)


def choice_settings(grid_min_gain=0.005, calibration_min_gain=0.01, horizons=5):
    return SimpleNamespace(grid=SimpleNamespace(min_gain=grid_min_gain),
                           calibration=SimpleNamespace(min_gain=calibration_min_gain),
                           receivables=SimpleNamespace(horizons=horizons))


def hand_choice(table, row, candidates):
    winner = table.iloc[row]
    i, j, k = int(winner["stage1"]), int(winner["stage2"]), int(winner["topups"])
    return SimpleNamespace(winner=winner, stage1=candidates.stage1[i],
                           stage2=candidates.stage2[j], topups=candidates.topups[k],
                           forecast="<forecast>", calibration="<calibration>",
                           scores="<scores>", bootstrap="<bootstrap>")


def printed_choice(capsys, choice, candidates, table, settings):
    printed = []
    capsys.readouterr()
    assert choice_printer(printed)(choice, candidates, table, settings) is None
    return capsys.readouterr().out.splitlines(), printed


CHOICE_TAIL = ["        its calibration (U15.4 on its forecast):",
               "<show_calibration_choice>",
               "        on the same origins, uncorrected (hurdle = the combination kept):",
               "<show_evaluation_scores>",
               "<show_paired_bootstrap>"]


def test_show_grid_choice_exact_printout_when_the_current_settings_are_kept(capsys):
    candidates, table = choice_candidates(), choice_scores_table()
    choice = hand_choice(table, 1, candidates)
    lines, _ = printed_choice(capsys, choice, candidates, table, choice_settings())
    assert lines == ["        kept the current settings: no combination beats their WAPE "
                     "0.1000 by more than 0.005 (best 0.0980)"] + CHOICE_TAIL


def test_show_grid_choice_kept_and_best_shows_its_own_wape_as_the_best(capsys):
    candidates = choice_candidates()
    table = choice_scores_table([(0, 0, 0, True, 0.1, 0.12, 0.031, 0.22),
                                 (1, 0, 1, False, 0.13, 0.1, 0.0, 0.2)])
    lines, _ = printed_choice(capsys, hand_choice(table, 0, candidates), candidates, table,
                              choice_settings(grid_min_gain=0.02))
    assert lines[0] == ("        kept the current settings: no combination beats their WAPE "
                        "0.1000 by more than 0.02 (best 0.1000)")


def test_show_grid_choice_exact_printout_of_new_settings(capsys):
    candidates, table = choice_candidates(), choice_scores_table(
        [(1, 0, 1, False, 0.08, 0.1, 0.0, 0.2), (0, 0, 0, True, 0.1, 0.12, 0.031, 0.22),
         (0, 1, 2, False, 0.17, 0.2, 0.0, 0.25)])
    choice = hand_choice(table, 0, candidates)
    lines, _ = printed_choice(capsys, choice, candidates, table, choice_settings())
    assert lines == [
        "        NEW settings: WAPE 0.0800 against 0.1000 (gain 0.0200, more than 0.005):",
        "          stage 1: {'max_iter': 900, 'learning_rate': 0.05} (was {'max_iter': 300, "
        "'learning_rate': 0.05})",
        "          stage 2: {'loss': 'poisson', 'learning_rate': 0.05}",
        f"          top-ups: {candidates.topups[1]} (was {candidates.topups[0]})",
    ] + CHOICE_TAIL


def test_show_grid_choice_new_settings_mark_only_the_stages_that_changed(capsys):
    candidates, table = choice_candidates(), choice_scores_table(
        [(0, 1, 0, False, 0.05, 0.1, 0.0, 0.2), (0, 0, 0, True, 0.1, 0.12, 0.031, 0.22)])
    lines, _ = printed_choice(capsys, hand_choice(table, 0, candidates), candidates, table,
                              choice_settings())
    stage_lines = lines[1:4]
    assert stage_lines[0] == "          stage 1: {'max_iter': 300, 'learning_rate': 0.05}"
    assert stage_lines[1].endswith("(was {'loss': 'poisson', 'learning_rate': 0.05})")
    assert stage_lines[1].startswith("          stage 2: {'loss': 'gamma', ")
    assert stage_lines[2] == f"          top-ups: {candidates.topups[0]}"
    assert "(was" not in stage_lines[0] and "(was" not in stage_lines[2]


def test_show_grid_choice_the_was_test_compares_values_not_identity(capsys):
    # A winner whose stage-1 params equal candidate 0's (another dict, same values) is not
    # marked as changed.
    candidates, table = choice_candidates(), choice_scores_table(
        [(1, 1, 2, False, 0.05, 0.1, 0.0, 0.2), (0, 0, 0, True, 0.1, 0.12, 0.031, 0.22)])
    candidates.stage1[1] = SimpleNamespace(params=dict(candidates.stage1[0].params))
    lines, _ = printed_choice(capsys, hand_choice(table, 0, candidates), candidates, table,
                              choice_settings())
    assert lines[1] == "          stage 1: {'max_iter': 300, 'learning_rate': 0.05}"


def test_show_grid_choice_the_gain_is_current_minus_winner_rounded_to_four(capsys):
    candidates, table = choice_candidates(), choice_scores_table(
        [(1, 0, 1, False, 0.123456, 0.1, 0.0, 0.2), (0, 0, 0, True, 0.2, 0.12, 0.031, 0.22)])
    lines, _ = printed_choice(capsys, hand_choice(table, 0, candidates), candidates, table,
                              choice_settings(grid_min_gain=0.05))
    assert lines[0] == ("        NEW settings: WAPE 0.1235 against 0.2000 (gain 0.0765, more "
                        "than 0.05):")


def test_show_grid_choice_finds_the_current_by_its_flag_not_its_position(capsys):
    candidates = choice_candidates()
    table = choice_scores_table([(1, 0, 1, False, 0.07, 0.1, 0.0, 0.2),
                                 (0, 1, 2, False, 0.08, 0.2, 0.0, 0.25),
                                 (0, 0, 0, True, 0.3, 0.12, 0.031, 0.22)])
    lines, _ = printed_choice(capsys, hand_choice(table, 0, candidates), candidates, table,
                              choice_settings())
    assert lines[0].startswith("        NEW settings: WAPE 0.0700 against 0.3000 (gain 0.2300")


def test_show_grid_choice_the_current_kept_prints_no_stage_lines(capsys):
    candidates, table = choice_candidates(), choice_scores_table()
    lines, _ = printed_choice(capsys, hand_choice(table, 1, candidates), candidates, table,
                              choice_settings())
    assert not any("stage 1:" in line or "top-ups:" in line for line in lines)


@pytest.mark.parametrize("row", [0, 1])
def test_show_grid_choice_passes_the_choice_parts_to_the_printers(capsys, row):
    candidates, table = choice_candidates(), choice_scores_table(
        [(1, 0, 1, False, 0.08, 0.1, 0.0, 0.2), (0, 0, 0, True, 0.1, 0.12, 0.031, 0.22)])
    choice = hand_choice(table, row, candidates)
    choice.calibration, choice.scores, choice.bootstrap = "<cal>", "<sc>", "<bs>"
    _, printed = printed_choice(capsys, choice, candidates, table,
                                choice_settings(grid_min_gain=0.005, calibration_min_gain=0.03,
                                                horizons=4))
    # The calibration's own min_gain, not the grid's.
    assert printed == [("show_calibration_choice", ("<cal>", 0.03)),
                       ("show_evaluation_scores", ("<sc>", 4)),
                       ("show_paired_bootstrap", ("<bs>",))]


def test_show_grid_choice_changes_nothing(capsys):
    import pandas as pd
    candidates, table = choice_candidates(), choice_scores_table()
    before = table.copy()
    params = [dict(c.params) for c in candidates.stage1 + candidates.stage2]
    printed_choice(capsys, hand_choice(table, 0, candidates), candidates, table,
                   choice_settings())
    pd.testing.assert_frame_equal(table, before)
    assert [c.params for c in candidates.stage1 + candidates.stage2] == params


# show_grid_choice through the real module, with real printers, on a real GridChoice made by
# grid_choice on test_grid.py's hand-made example (the references are fakes there; the
# calibration a hand-made CalibrationChoice).
SHOW_GRID_CHOICE = """
from types import SimpleNamespace
import pandas as pd
from collection_estimation.hurdle.reports import grid as reports
from collection_estimation.hurdle import grid as grid_module
from collection_estimation.hurdle.calibrate import (CANDIDATE_COLUMNS, CalibrationChoice,
                                                    MCorrection, PCorrection)
from tests.hurdle import test_grid as tg

candidates_table = pd.DataFrame(
    [("p none, m none", "none", "none", 0, 0.17034, 0.11119, -0.05, 0.22221)],
    columns=CANDIDATE_COLUMNS)
calibration = CalibrationChoice(p_correction=PCorrection("none"),
                                m_correction=MCorrection("none"), split_week=21,
                                judged_origins=[22], candidates=candidates_table,
                                winner="p none, m none",
                                bootstrap=pd.DataFrame(columns=["horizon"]))
predictions, actual = tg.sc_predictions(), tg.sc_actual()
table = tg.sc_run(predictions=predictions, actual=actual)
candidates = SimpleNamespace(
    stage1=[SimpleNamespace(params={"max_iter": 10 * (i + 1)}) for i in range(3)],
    stage2=[SimpleNamespace(params={"learning_rate": 0.1 * (j + 1)}) for j in range(3)],
    topups=[f"<topups {k}>" for k in range(2)])
grid_module.choose_calibration = lambda *args, **kwargs: calibration
grid_module.deflated_reference = lambda *args, **kwargs: tg.fake_reference(actual)
grid_module.rule_reference = lambda *args, **kwargs: SimpleNamespace(
    totals=tg.fake_rule_totals(actual))
MODE
choice = tg.ch_run(predictions, scores, actual, settings=settings, candidates=candidates)
shown = SimpleNamespace(grid=settings.grid, calibration=settings.calibration,
                        receivables=settings.receivables)
reports.show_grid_choice(choice, candidates, scores, shown)
print("ok")
"""
KEPT_MODE = "scores = table\nsettings = tg.ch_settings(min_gain=10.0)"
NEW_MODE = "scores = tg.without_current(table)\nsettings = tg.ch_settings(min_gain=0.0)"


def real_choice_lines(mode):
    result = run_snippet(SHOW_GRID_CHOICE.replace("MODE", mode))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_show_grid_choice_real_printers_when_the_current_is_kept():
    lines = real_choice_lines(KEPT_MODE)
    assert lines[0].startswith("        kept the current settings: no combination beats their "
                               "WAPE ")
    assert lines[0].endswith(")") and "by more than 10.0 (best " in lines[0]
    assert lines[1] == "        its calibration (U15.4 on its forecast):"
    assert "        chosen: p none, m none" in lines
    at = lines.index("        on the same origins, uncorrected (hurdle = the combination kept):")
    rest = "\n".join(lines[at + 1:])
    assert "scores (U14.5), consolidated" in rest
    assert "paired bootstrap (U14.5c)" in rest
    assert "current settings" not in rest
    for method in ("hurdle", "deflated", "deflated_nolevel", "deflated_trend", "rule"):
        assert method in rest, method


def test_show_grid_choice_real_printers_with_new_settings():
    lines = real_choice_lines(NEW_MODE)
    assert lines[0].startswith("        NEW settings: WAPE ")
    assert lines[0].endswith("more than 0.0):")
    assert lines[1].startswith("          stage 1: ")
    assert lines[2].startswith("          stage 2: ")
    assert lines[3].startswith("          top-ups: ")
    at = lines.index("        on the same origins, uncorrected (hurdle = the combination kept):")
    rest = "\n".join(lines[at + 1:])
    assert "hurdle vs current settings" in rest
