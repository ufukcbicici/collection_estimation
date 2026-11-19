"""Tests of `grid_scores`' batches: the combinations forecast and scored `batch_size` at a time.

The table must not depend on the batch size (a combination's score reads its own rows only),
each combination is forecast and scored whole within one batch, `on_batch_done(done, total)`
is called once per batch, and a batch size below 1 is refused. Run on test_grid.py's hand-made
GridPredictions (every candidate eligible; since `shortlisted_numbers` only the shortlists are
scored, stage 1 {0, 1} x stage 2 {0, 2} x 2 top-ups = 8 combinations) and on
test_grid_extra_blocks.py's labelled candidates (U18.5: only same-subset pairs of the
shortlists). The progress lines run_grid prints from
`on_batch_done` are tested in test_grid_run.py.
"""
import inspect
import itertools
import math

import pandas as pd
import pytest

from collection_estimation.hurdle import grid as grid_module
from collection_estimation.hurdle.forecast import CONSOLIDATED
from collection_estimation.hurdle.grid import SCORE_BATCH, combination_name, grid_scores
from tests.hurdle.test_forecast import WEEK_ZERO
from tests.hurdle.test_grid import (
    SC_COMBINATIONS,
    SC_COMPANIES,
    identical_candidates,
    sc_actual,
    sc_candidates,
    sc_gates,
    sc_predictions,
    sc_settings,
)
from tests.hurdle.test_grid_extra_blocks import (
    NAMES1, NAMES2, SAME_SUBSET_SHORT, labelled, labelled_gates)

# The combinations of sc_predictions() with every candidate eligible: the shortlists
# {0, 1} x {0, 2} x 2 top-ups (was 3 x 3 x 2 = 18 before shortlisted_numbers).
N = 2 * 2 * 2
assert len(SC_COMBINATIONS) == N
BATCH_MESSAGE = "batch_size is {n}; at least 1 combination is needed"


def scores(predictions=None, candidates=None, gates=None, actual=None, **kwargs):
    """grid_scores on test_grid.py's stand-ins; `kwargs` go to it (batch_size, on_batch_done)."""
    return grid_scores(predictions if predictions is not None else sc_predictions(),
                       candidates or sc_candidates(), gates or sc_gates(),
                       actual if actual is not None else sc_actual(), sc_settings(),
                       companies=list(SC_COMPANIES), week_zero=WEEK_ZERO, **kwargs)


@pytest.fixture(scope="module")
def case():
    predictions, actual = sc_predictions(), sc_actual()
    return predictions, actual, scores(predictions, actual=actual)


def recorder():
    calls = []

    def on_batch_done(done, total):
        calls.append((done, total))
    return calls, on_batch_done


def watch_scoring(monkeypatch):
    """Records the methods (and their row counts) of every evaluation_scores call."""
    batches = []
    real = grid_module.evaluation_scores

    def watched(forecasts, actual, horizons):
        batches.append(forecasts["method"].value_counts().to_dict())
        return real(forecasts, actual, horizons)

    monkeypatch.setattr(grid_module, "evaluation_scores", watched)
    return batches


# --- the constant and the signature ---------------------------------------------------------

def test_score_batch_is_500():
    assert SCORE_BATCH == 500
    assert type(SCORE_BATCH) is int


def test_batch_size_and_on_batch_done_are_keyword_only_with_their_defaults():
    parameters = inspect.signature(grid_scores).parameters
    assert list(parameters)[-2:] == ["batch_size", "on_batch_done"]
    for name in ("batch_size", "on_batch_done"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert parameters["batch_size"].default == SCORE_BATCH
    assert parameters["on_batch_done"].default is None


# --- the table does not depend on the batch size -------------------------------------------

@pytest.mark.parametrize("batch_size", [
    1, 2,
    3,              # 8 = 2 x 3 + 2: a smaller last batch
    4,              # 8 = 2 x 4: two whole batches
    5,              # 8 = 5 + 3
    7,              # a last batch of 1
    N,              # exactly the combinations: one batch
    N + 1,          # more than the combinations: one batch
    10 ** 6,
])
def test_the_table_is_the_same_for_any_batch_size(case, batch_size):
    predictions, actual, table = case
    found = scores(predictions, actual=actual, batch_size=batch_size)
    pd.testing.assert_frame_equal(found, table, check_exact=True)


def test_the_default_batch_size_is_score_batch(case):
    predictions, actual, table = case
    pd.testing.assert_frame_equal(
        scores(predictions, actual=actual, batch_size=SCORE_BATCH), table, check_exact=True)
    assert len(table) == N < SCORE_BATCH


def test_the_table_of_one_batch_per_combination_equals_one_batch_of_all(case):
    predictions, actual, _ = case
    pd.testing.assert_frame_equal(scores(predictions, actual=actual, batch_size=1),
                                  scores(predictions, actual=actual, batch_size=N),
                                  check_exact=True)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 12])
def test_ties_keep_the_candidates_order_whatever_the_batch_size(batch_size):
    # Every candidate the same: every WAPE ties, so the order is the product order itself
    # (of the shortlists: stage 1 {0, 1}, stage 2 {0, 2}).
    predictions = identical_candidates()
    candidates, gates = sc_candidates(2, 3, 2), sc_gates((True, True), (True, True, True))
    found = scores(predictions, candidates=candidates, gates=gates, batch_size=batch_size)
    assert found["wape"].nunique() == 1
    assert list(zip(found["stage1"], found["stage2"], found["topups"])) == list(
        itertools.product([0, 1], [0, 2], range(2)))
    pd.testing.assert_frame_equal(found, scores(predictions, candidates=candidates,
                                                gates=gates), check_exact=True)


@pytest.mark.parametrize("batch_size", [1, 3, 4])
def test_ineligible_candidates_are_left_out_whatever_the_batch_size(case, batch_size):
    predictions, actual, _ = case
    gates = sc_gates((True, False, True), (False, True, True))
    found = scores(predictions, gates=gates, actual=actual, batch_size=batch_size)
    pd.testing.assert_frame_equal(found, scores(predictions, gates=gates, actual=actual),
                                  check_exact=True)
    # Stage-2 candidate 1 is eligible but not shortlisted (skill 0.2 < candidate 2's 0.3).
    assert set(found["stage1"]) == {0, 2} and set(found["stage2"]) == {0, 2}
    assert len(found) == 2 * 2 * 2


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_a_lone_combination_is_scored_whatever_the_batch_size(batch_size):
    predictions = sc_predictions(n1=1, n2=1, n3=1)
    candidates, gates = sc_candidates(1, 1, 1), sc_gates((True,), (True,))
    calls, on_batch_done = recorder()
    found = scores(predictions, candidates=candidates, gates=gates, batch_size=batch_size,
                   on_batch_done=on_batch_done)
    assert len(found) == 1 and bool(found.loc[0, "current"]) is True
    assert calls == [(1, 1)]


def test_the_batches_change_no_input(case):
    predictions, actual, _ = case
    before = (predictions.rows.copy(), predictions.p.copy(), predictions.m.copy(),
              [t.copy() for t in predictions.topups], actual.copy())
    scores(predictions, actual=actual, batch_size=4)
    for now, then in zip((predictions.rows, predictions.p, predictions.m, actual),
                         (before[0], before[1], before[2], before[4])):
        pd.testing.assert_frame_equal(now, then)
    for now, then in zip(predictions.topups, before[3]):
        pd.testing.assert_frame_equal(now, then)


# --- each combination whole within one batch -----------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 4, 7, N, N + 3])
def test_each_batch_is_scored_once_holding_whole_combinations(case, monkeypatch, batch_size):
    predictions, actual, _ = case
    batches = watch_scoring(monkeypatch)
    scores(predictions, actual=actual, batch_size=batch_size)
    assert len(batches) == math.ceil(N / batch_size)
    sizes = [len(batch) for batch in batches]
    assert sizes == [batch_size] * (N // batch_size) + ([N % batch_size] if N % batch_size else [])
    # Every combination in exactly one batch, with as many rows as when scored all at once.
    methods = [method for batch in batches for method in batch]
    assert len(methods) == len(set(methods)) == N
    rows = {method: count for batch in batches for method, count in batch.items()}
    assert len(set(rows.values())) == 1     # every combination the same full set of rows


def test_the_batches_follow_the_product_order(case, monkeypatch):
    predictions, actual, _ = case
    batches = watch_scoring(monkeypatch)
    scores(predictions, actual=actual, batch_size=5)
    expected = [combination_name(i, j, k) for i, j, k in SC_COMBINATIONS]
    found = [method for batch in batches for method in batch]
    assert found == expected
    assert [list(batch) for batch in batches] == [expected[s:s + 5] for s in range(0, N, 5)]


def test_one_batch_holds_the_same_rows_as_the_whole_grid(case, monkeypatch):
    predictions, actual, _ = case
    one = watch_scoring(monkeypatch)
    scores(predictions, actual=actual, batch_size=N)
    assert len(one) == 1
    many = watch_scoring(monkeypatch)
    scores(predictions, actual=actual, batch_size=1)
    assert {m: c for batch in many for m, c in batch.items()} == one[0]


def test_each_combination_is_forecast_once_whatever_the_batch_size(case, monkeypatch):
    predictions, actual, _ = case
    made = []
    real = grid_module.combination_forecast

    def counted(predictions, i, j, k, **kwargs):
        made.append((i, j, k))
        return real(predictions, i, j, k, **kwargs)

    monkeypatch.setattr(grid_module, "combination_forecast", counted)
    scores(predictions, actual=actual, batch_size=4)
    assert made == SC_COMBINATIONS


# --- on_batch_done -------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size, expected", [
    (1, [(d, N) for d in range(1, N + 1)]),
    (3, [(3, N), (6, N), (8, N)]),
    (4, [(4, N), (8, N)]),
    (5, [(5, N), (8, N)]),
    (N, [(N, N)]),
    (N + 1, [(N, N)]),
    (SCORE_BATCH, [(N, N)]),
])
def test_on_batch_done_once_per_batch_with_cumulative_counts(case, batch_size, expected):
    predictions, actual, _ = case
    calls, on_batch_done = recorder()
    scores(predictions, actual=actual, batch_size=batch_size, on_batch_done=on_batch_done)
    assert calls == expected
    assert len(calls) == math.ceil(N / batch_size)
    assert calls[-1] == (N, N)


def test_on_batch_done_with_the_default_batch_size_is_called_once(case):
    predictions, actual, _ = case
    calls, on_batch_done = recorder()
    scores(predictions, actual=actual, on_batch_done=on_batch_done)
    assert calls == [(N, N)]


def test_on_batch_done_counts_are_plain_ints(case):
    predictions, actual, _ = case
    calls, on_batch_done = recorder()
    scores(predictions, actual=actual, batch_size=7, on_batch_done=on_batch_done)
    for done, total in calls:
        assert type(done) is int and type(total) is int


def test_on_batch_done_total_counts_the_scored_combinations_only(case):
    # With ineligible candidates the total is what is scored, not the full product: stage 1
    # {0} (nothing else eligible), stage 2 {0, 1} (1 the only other eligible one).
    predictions, actual, _ = case
    calls, on_batch_done = recorder()
    found = scores(predictions, gates=sc_gates((True, False, False), (True, True, False)),
                   actual=actual, batch_size=3, on_batch_done=on_batch_done)
    assert len(found) == 1 * 2 * 2
    assert calls == [(3, 4), (4, 4)]


def test_on_batch_done_runs_after_its_batch_is_forecast_and_scored(case, monkeypatch):
    predictions, actual, _ = case
    batches = watch_scoring(monkeypatch)
    made = []
    real = grid_module.combination_forecast

    def counted(*args, **kwargs):
        made.append(args[1:4])
        return real(*args, **kwargs)

    monkeypatch.setattr(grid_module, "combination_forecast", counted)
    seen = []
    scores(predictions, actual=actual, batch_size=4,
           on_batch_done=lambda done, total: seen.append((done, len(made), len(batches))))
    # When told `done`, exactly `done` combinations have been forecast, and one more batch
    # has been scored each time: never ahead of the work, never behind it.
    assert seen == [(4, 4, 1), (8, 8, 2)]


@pytest.mark.parametrize("given", [{}, {"on_batch_done": None}])
def test_on_batch_done_is_not_required(case, given):
    predictions, actual, table = case
    pd.testing.assert_frame_equal(scores(predictions, actual=actual, batch_size=4, **given),
                                  table, check_exact=True)


def test_on_batch_done_that_raises_stops_the_scores(case, monkeypatch):
    predictions, actual, _ = case
    batches = watch_scoring(monkeypatch)

    def stop(done, total):
        raise RuntimeError(f"stopped at {done} of {total}")

    with pytest.raises(RuntimeError, match="stopped at 4 of 8"):
        scores(predictions, actual=actual, batch_size=4, on_batch_done=stop)
    assert len(batches) == 1


def test_on_batch_done_is_called_positionally_with_done_then_total(case):
    predictions, actual, _ = case
    seen = []

    def only_positional(a, b, /):
        seen.append((a, b))

    scores(predictions, actual=actual, batch_size=3, on_batch_done=only_positional)
    assert seen == [(3, N), (6, N), (N, N)]


# --- a batch size below 1 -----------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [0, -1, -500])
def test_a_batch_size_below_1_raises(case, batch_size):
    predictions, actual, _ = case
    with pytest.raises(ValueError) as error:
        scores(predictions, actual=actual, batch_size=batch_size)
    assert str(error.value) == BATCH_MESSAGE.format(n=batch_size)


def test_a_batch_size_below_1_raises_before_anything_is_forecast_or_told(case, monkeypatch):
    predictions, actual, _ = case

    def never(*args, **kwargs):
        raise AssertionError("forecast with a batch size of 0")

    monkeypatch.setattr(grid_module, "combination_forecast", never)
    calls, on_batch_done = recorder()
    with pytest.raises(ValueError, match="^batch_size is 0; at least 1 combination is needed$"):
        scores(predictions, actual=actual, batch_size=0, on_batch_done=on_batch_done)
    assert calls == []


def test_the_candidate_counts_are_checked_before_the_batch_size(case):
    predictions, actual, _ = case
    with pytest.raises(ValueError) as error:
        scores(predictions, gates=sc_gates((True, True), (True, True, True)), actual=actual,
               batch_size=0)
    assert "candidates" in str(error.value)
    assert "batch_size" not in str(error.value)


def test_the_batch_size_is_checked_before_the_origins(case):
    predictions, _, _ = case
    with pytest.raises(ValueError) as error:
        scores(predictions, actual=sc_actual(origins=(18,)), batch_size=0)
    assert str(error.value) == BATCH_MESSAGE.format(n=0)
    # With a good batch size the same actual cash is refused by the origin check.
    with pytest.raises(ValueError, match="the actual cash is for origins"):
        scores(predictions, actual=sc_actual(origins=(18,)), batch_size=1)


def test_a_batch_size_of_1_is_accepted(case):
    predictions, actual, table = case
    assert len(scores(predictions, actual=actual, batch_size=1)) == len(table)


# --- no consolidated cash, with small batches ----------------------------------------------

NO_CASH = "there is no consolidated cash at the grid's origins to score on"


def company_rows_only(actual):
    return actual[actual["company_code"] != CONSOLIDATED]


@pytest.mark.parametrize("batch_size", [1, 2, 4, N, SCORE_BATCH])
def test_no_consolidated_cash_raises_whatever_the_batch_size(case, batch_size):
    predictions, actual, _ = case
    with pytest.raises(ValueError) as error:
        scores(predictions, actual=company_rows_only(actual), batch_size=batch_size)
    assert str(error.value) == NO_CASH


@pytest.mark.parametrize("batch_size", [1, 3])
def test_no_consolidated_cash_raises_at_the_first_batch_before_it_is_told(case, monkeypatch,
                                                                         batch_size):
    predictions, actual, _ = case
    batches = watch_scoring(monkeypatch)
    calls, on_batch_done = recorder()
    with pytest.raises(ValueError, match=NO_CASH.replace("'", ".")):
        scores(predictions, actual=company_rows_only(actual), batch_size=batch_size,
               on_batch_done=on_batch_done)
    assert len(batches) == 1 and len(batches[0]) == batch_size
    assert calls == []


# --- U18.5: only same-subset pairs of the shortlists, whatever the batch size ------------

def subset_scores(**kwargs):
    return scores(sc_predictions(n1=4, n2=4, n3=2), candidates=labelled(NAMES1, NAMES2),
                  gates=labelled_gates(NAMES1, NAMES2), **kwargs)


# test_grid_extra_blocks.py's worked shortlists: stage 1 {0, 1, 2, 3}, stage 2 {0, 2, 3}; the
# same-subset pairs (0, 0), (1, 0), (2, 2), (3, 3), times 2 top-ups (was every same-subset
# pair, 10, before shortlisted_numbers).
SAME_SUBSET = SAME_SUBSET_SHORT


@pytest.fixture(scope="module")
def subset_table():
    return subset_scores()


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 7, len(SAME_SUBSET), len(SAME_SUBSET) + 1])
def test_subsets_the_table_is_the_same_for_any_batch_size(subset_table, batch_size):
    found = subset_scores(batch_size=batch_size)
    pd.testing.assert_frame_equal(found, subset_table, check_exact=True)
    assert set(zip(found["stage1"], found["stage2"], found["topups"])) == set(SAME_SUBSET)


@pytest.mark.parametrize("batch_size", [1, 3, 4])
def test_subsets_only_same_subset_pairs_are_forecast_and_scored(monkeypatch, batch_size):
    batches = watch_scoring(monkeypatch)
    calls, on_batch_done = recorder()
    subset_scores(batch_size=batch_size, on_batch_done=on_batch_done)
    scored = [method for batch in batches for method in batch]
    assert scored == [combination_name(i, j, k) for i, j, k in SAME_SUBSET]
    for i, j, _ in SAME_SUBSET:
        assert NAMES1[i] == NAMES2[j]
    total = len(SAME_SUBSET)
    assert total == 8
    assert len(calls) == math.ceil(total / batch_size)
    assert [t for _, t in calls] == [total] * len(calls)
    assert calls[-1] == (total, total)
    assert [d for d, _ in calls] == [min(total, batch_size * (b + 1)) for b in range(len(calls))]
