"""Tests for `hurdle/diagnostic.py`: `DiagnosticSplits`, `diagnostic_splits`, `best_threshold`
and `classification_scores`.

Every table is hand-made; nothing is read from data/ (no corpus, no ledger, no cache).
Hypothesis is not installed in this env, so the brute-force checks use seeded random trials.
"""
import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.diagnostic import (
    DiagnosticSplits,
    best_threshold,
    classification_scores,
    diagnostic_splits,
)

T, H = 10, 3   # origin and horizons of most split tests: train <= 10 < val <= 13 < test <= 16


def split_table(target_weeks, statuses=None, index=None):
    """A training table with one row per target week; `paid` alternates on labelled rows and
    is NA elsewhere. Extra columns check that every column is kept."""
    n = len(target_weeks)
    statuses = ["labelled"] * n if statuses is None else statuses
    paid = [bool(i % 2) if s == "labelled" else None for i, s in enumerate(statuses)]
    return pd.DataFrame({
        "key": [f"TR02-{i}" for i in range(n)],
        "origin_week": np.array([w - 1 for w in target_weeks], dtype="int64"),
        "h": np.ones(n, dtype="int64"),
        "target_week": np.array(target_weeks, dtype="int64"),
        "label_status": np.array(statuses, dtype=object),
        "paid": pd.array(paid, dtype="boolean"),
        "feature": np.arange(n, dtype="float64") / 10,
    }, index=index)


def weeks_of(frame):
    return sorted(frame["target_week"].tolist())


# --- DiagnosticSplits ------------------------------------------------------------------------

def test_diagnostic_splits_fields_in_order():
    assert [field.name for field in dataclasses.fields(DiagnosticSplits)] == [
        "train", "validation", "refit", "test"]


def test_diagnostic_splits_is_frozen():
    splits = diagnostic_splits(split_table(list(range(0, 20))), T, H)
    for name in ("train", "validation", "refit", "test"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(splits, name, None)


def test_diagnostic_splits_returns_dataframes():
    splits = diagnostic_splits(split_table(list(range(0, 20))), T, H)
    assert isinstance(splits, DiagnosticSplits)
    for name in ("train", "validation", "refit", "test"):
        assert isinstance(getattr(splits, name), pd.DataFrame)


# --- diagnostic_splits: boundaries -----------------------------------------------------------

def test_split_boundaries_exactly():
    # Target weeks T-1, T, T+1, T+H, T+H+1, T+2H, T+2H+1.
    weeks = [T - 1, T, T + 1, T + H, T + H + 1, T + 2 * H, T + 2 * H + 1]
    splits = diagnostic_splits(split_table(weeks), T, H)
    assert weeks_of(splits.train) == [T - 1, T]
    assert weeks_of(splits.validation) == [T + 1, T + H]
    assert weeks_of(splits.refit) == [T - 1, T, T + 1, T + H]
    assert weeks_of(splits.test) == [T + H + 1, T + 2 * H]


def test_origin_week_is_in_train_not_validation():
    splits = diagnostic_splits(split_table([T, T + 1, T + H + 1]), T, H)
    assert weeks_of(splits.train) == [T]
    assert T not in weeks_of(splits.validation)


def test_origin_plus_h_is_in_validation_not_test():
    splits = diagnostic_splits(split_table([T, T + H, T + H + 1]), T, H)
    assert weeks_of(splits.validation) == [T + H]
    assert T + H not in weeks_of(splits.test)


def test_origin_plus_2h_is_in_test():
    splits = diagnostic_splits(split_table([T, T + 1, T + 2 * H]), T, H)
    assert weeks_of(splits.test) == [T + 2 * H]


def test_after_origin_plus_2h_is_nowhere():
    weeks = [T, T + 1, T + H + 1, T + 2 * H + 1, T + 2 * H + 5, T + 100]
    splits = diagnostic_splits(split_table(weeks), T, H)
    for name in ("train", "validation", "refit", "test"):
        part = weeks_of(getattr(splits, name))
        assert all(week <= T + 2 * H for week in part), (name, part)


def test_every_week_from_0_to_30_lands_in_the_right_part():
    weeks = list(range(0, 31))
    splits = diagnostic_splits(split_table(weeks), T, H)
    assert weeks_of(splits.train) == list(range(0, T + 1))
    assert weeks_of(splits.validation) == list(range(T + 1, T + H + 1))
    assert weeks_of(splits.refit) == list(range(0, T + H + 1))
    assert weeks_of(splits.test) == list(range(T + H + 1, T + 2 * H + 1))


@pytest.mark.parametrize("origin, horizons", [(0, 1), (1, 1), (5, 2), (40, 5), (3, 10)])
def test_boundaries_for_several_origins_and_horizons(origin, horizons):
    weeks = list(range(0, origin + 3 * horizons + 3))
    splits = diagnostic_splits(split_table(weeks), origin, horizons)
    assert weeks_of(splits.train) == list(range(0, origin + 1))
    assert weeks_of(splits.validation) == list(range(origin + 1, origin + horizons + 1))
    assert weeks_of(splits.test) == list(range(origin + horizons + 1, origin + 2 * horizons + 1))


def test_several_rows_per_target_week_all_go_to_the_same_part():
    weeks = [T, T, T, T + 1, T + 1, T + H + 1, T + H + 1]
    splits = diagnostic_splits(split_table(weeks), T, H)
    assert len(splits.train) == 3 and len(splits.validation) == 2 and len(splits.test) == 2


# --- diagnostic_splits: labels ---------------------------------------------------------------

def test_unlabelled_rows_are_nowhere():
    weeks = [T - 2, T, T + 1, T + 2, T + H + 1, T + H + 2]
    statuses = ["labelled", "missing_week", "labelled", "future", "labelled", "missing_week"]
    table = split_table(weeks, statuses, index=[100, 101, 102, 103, 104, 105])
    splits = diagnostic_splits(table, T, H)
    unlabelled = {101, 103, 105}
    for name in ("train", "validation", "refit", "test"):
        part = getattr(splits, name)
        assert not unlabelled & set(part.index), name
        assert (part["label_status"] == "labelled").all(), name
        assert part["paid"].notna().all(), name


def test_unknown_status_values_count_as_unlabelled():
    # Only the exact text "labelled" is a label.
    weeks = [T, T, T + 1, T + H + 1]
    statuses = ["labelled", "Labelled", "labelled", "labelled"]
    splits = diagnostic_splits(split_table(weeks, statuses), T, H)
    assert len(splits.train) == 1


def test_each_label_is_in_exactly_one_of_train_validation_test():
    rng = np.random.default_rng(7)
    weeks = rng.integers(0, 2 * T, size=300).tolist()
    statuses = rng.choice(["labelled", "missing_week", "future"], size=300).tolist()
    table = split_table(weeks, statuses)
    splits = diagnostic_splits(table, T, H)
    seen = pd.concat([splits.train, splits.validation, splits.test]).index
    assert seen.is_unique
    in_range = table[(table["label_status"] == "labelled")
                     & (table["target_week"] <= T + 2 * H)]
    assert sorted(seen) == sorted(in_range.index)


def test_refit_is_train_and_validation_together():
    rng = np.random.default_rng(11)
    weeks = rng.integers(0, 2 * T, size=200).tolist()
    statuses = rng.choice(["labelled", "missing_week"], size=200).tolist()
    splits = diagnostic_splits(split_table(weeks, statuses), T, H)
    together = pd.concat([splits.train, splits.validation])
    assert len(splits.refit) == len(splits.train) + len(splits.validation)
    pd.testing.assert_frame_equal(splits.refit, together.loc[sorted(together.index)])


def test_refit_and_test_do_not_overlap():
    splits = diagnostic_splits(split_table(list(range(0, 20))), T, H)
    assert not set(splits.refit.index) & set(splits.test.index)


# --- diagnostic_splits: index, columns, inputs -----------------------------------------------

def test_the_index_and_columns_are_kept():
    weeks = [T - 1, T, T + 1, T + H + 1]
    table = split_table(weeks, index=pd.Index([907, 3, 55, 12], name="row"))
    splits = diagnostic_splits(table, T, H)
    assert list(splits.train.index) == [907, 3]
    assert list(splits.validation.index) == [55]
    assert list(splits.refit.index) == [907, 3, 55]
    assert list(splits.test.index) == [12]
    for name in ("train", "validation", "refit", "test"):
        part = getattr(splits, name)
        assert part.index.name == "row"
        assert list(part.columns) == list(table.columns)
        pd.testing.assert_frame_equal(part, table.loc[part.index])


def test_rows_keep_the_table_order_not_target_week_order():
    weeks = [T, T - 5, T - 1, T + 2, T + 1]
    table = split_table(weeks + [T + H + 1])
    splits = diagnostic_splits(table, T, H)
    assert splits.train["target_week"].tolist() == [T, T - 5, T - 1]
    assert splits.validation["target_week"].tolist() == [T + 2, T + 1]


def test_the_dtypes_are_kept():
    table = split_table(list(range(0, 20)))
    splits = diagnostic_splits(table, T, H)
    pd.testing.assert_series_equal(splits.train.dtypes, table.dtypes)


def test_the_table_is_not_changed():
    weeks = [T - 1, T, T + 1, T + 2, T + H + 1, T + 2 * H + 1]
    statuses = ["labelled", "missing_week", "labelled", "future", "labelled", "labelled"]
    table = split_table(weeks, statuses)
    before = table.copy(deep=True)
    diagnostic_splits(table, T, H)
    pd.testing.assert_frame_equal(table, before)


def test_changing_a_split_does_not_change_the_table():
    table = split_table(list(range(0, 20)))
    before = table.copy(deep=True)
    splits = diagnostic_splits(table, T, H)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splits.train.loc[splits.train.index[0], "feature"] = -99.0
    pd.testing.assert_frame_equal(table, before)


def test_same_input_same_splits():
    table = split_table(list(range(0, 20)))
    one, two = diagnostic_splits(table, T, H), diagnostic_splits(table, T, H)
    for name in ("train", "validation", "refit", "test"):
        pd.testing.assert_frame_equal(getattr(one, name), getattr(two, name))


# --- diagnostic_splits: empty parts ----------------------------------------------------------

def empty_message(origin, horizons, parts):
    return (f"the diagnostic at origin {origin} with {horizons} horizons has no labelled rows "
            f"for {parts}")


def test_no_train_rows_raises_naming_train():
    with pytest.raises(ValueError) as error:
        diagnostic_splits(split_table([T + 1, T + H + 1]), T, H)
    assert str(error.value) == empty_message(T, H, ["train"])


def test_no_validation_rows_raises_naming_validation():
    with pytest.raises(ValueError) as error:
        diagnostic_splits(split_table([T, T + H + 1]), T, H)
    assert str(error.value) == empty_message(T, H, ["validation"])


def test_no_test_rows_raises_naming_test():
    with pytest.raises(ValueError) as error:
        diagnostic_splits(split_table([T, T + 1, T + 2 * H + 1]), T, H)
    assert str(error.value) == empty_message(T, H, ["test"])


def test_no_validation_and_no_test_names_both_in_order():
    with pytest.raises(ValueError) as error:
        diagnostic_splits(split_table([T - 1, T]), T, H)
    assert str(error.value) == empty_message(T, H, ["validation", "test"])


def test_no_train_and_no_test_names_both_in_order():
    with pytest.raises(ValueError) as error:
        diagnostic_splits(split_table([T + 1, T + 2]), T, H)
    assert str(error.value) == empty_message(T, H, ["train", "test"])


def test_no_labelled_rows_names_all_three_and_not_refit():
    table = split_table([T, T + 1, T + H + 1], ["missing_week", "future", "missing_week"])
    with pytest.raises(ValueError) as error:
        diagnostic_splits(table, T, H)
    assert str(error.value) == empty_message(T, H, ["train", "validation", "test"])
    assert "refit" not in str(error.value)


def test_an_empty_table_names_all_three():
    table = split_table([]).iloc[0:0]
    with pytest.raises(ValueError) as error:
        diagnostic_splits(table, 4, 2)
    assert str(error.value) == empty_message(4, 2, ["train", "validation", "test"])


def test_rows_only_in_unlabelled_status_do_not_fill_a_part():
    # Validation's weeks hold only a missing_week row: validation is still empty.
    table = split_table([T, T + 1, T + H + 1], ["labelled", "missing_week", "labelled"])
    with pytest.raises(ValueError) as error:
        diagnostic_splits(table, T, H)
    assert str(error.value) == empty_message(T, H, ["validation"])


def test_test_weeks_after_the_last_labelled_week_raise():
    # The typical failure: T + 2H after the last labelled week.
    with pytest.raises(ValueError) as error:
        diagnostic_splits(split_table(list(range(0, 14))), 12, 5)
    assert str(error.value) == empty_message(12, 5, ["test"])


def test_zero_horizons_leaves_validation_and_test_empty():
    with pytest.raises(ValueError) as error:
        diagnostic_splits(split_table(list(range(0, 20))), T, 0)
    assert str(error.value) == empty_message(T, 0, ["validation", "test"])


# --- best_threshold --------------------------------------------------------------------------

def series(values, index=None, dtype=None):
    return pd.Series(values, index=index, dtype=dtype)


def brute_force(actual, p):
    """{threshold: F1} at every distinct p, predicting pays when p >= threshold."""
    actual = np.asarray(actual, dtype=bool)
    p = np.asarray(p, dtype=float)
    out = {}
    for t in np.unique(p):
        predicted = p >= t
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[float(t)] = (2 * precision * recall / (precision + recall)
                         if precision + recall else 0.0)
    return out


def test_best_threshold_small_example_by_hand():
    # p .9 T, .7 T, .6 F, .4 T, .1 F. At .4: TP 3, FP 1 -> P .75, R 1, F1 6/7 = 0.857 (best).
    actual = series([True, True, False, True, False])
    p = series([0.9, 0.7, 0.6, 0.4, 0.1])
    threshold, f1 = best_threshold(actual, p)
    assert threshold == 0.4
    assert f1 == pytest.approx(6 / 7)


@pytest.mark.parametrize("seed", range(200))
def test_best_threshold_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 40))
    actual = rng.random(n) < rng.uniform(0.1, 0.9)
    if actual.all() or not actual.any():
        actual[0] = not actual[0]
    # Rounded p values make ties common.
    p = np.round(rng.random(n), int(rng.integers(1, 4)))
    threshold, f1 = best_threshold(series(actual), series(p))
    expected = brute_force(actual, p)
    best = max(expected.values())
    assert f1 == pytest.approx(best, abs=1e-12)
    assert threshold in expected
    assert expected[threshold] == pytest.approx(best, abs=1e-12)


@pytest.mark.parametrize("seed", range(50))
def test_best_threshold_f1_is_the_f1_at_its_threshold(seed):
    rng = np.random.default_rng(1000 + seed)
    n = 30
    actual = np.array([True, False] + list(rng.random(n - 2) < 0.4))
    p = rng.random(n)
    threshold, f1 = best_threshold(series(actual), series(p))
    assert f1 == pytest.approx(brute_force(actual, p)[threshold], abs=1e-12)


def test_perfect_separation_gives_f1_one_at_the_lowest_payer():
    actual = series([False, True, False, True, True])
    p = series([0.1, 0.8, 0.3, 0.6, 0.95])
    threshold, f1 = best_threshold(actual, p)
    assert f1 == 1.0
    assert threshold == 0.6


def test_perfect_separation_with_a_single_payer():
    actual = series([False, False, True, False])
    p = series([0.2, 0.4, 0.41, 0.0])
    assert best_threshold(actual, p) == (0.41, 1.0)


def test_reversed_separation_still_gives_a_threshold():
    # Payers have the lowest p: the best is to predict everyone pays (recall 1, P = 2/5).
    actual = series([True, True, False, False, False])
    p = series([0.1, 0.2, 0.7, 0.8, 0.9])
    threshold, f1 = best_threshold(actual, p)
    expected = brute_force(actual, p)
    assert f1 == pytest.approx(max(expected.values()))
    assert threshold == 0.1
    assert f1 == pytest.approx(2 * 0.4 * 1 / 1.4)


def test_ties_in_p_are_split_together():
    # Two rows at p 0.5, one payer and one not: a threshold cannot separate them.
    actual = series([True, False, True, False])
    p = series([0.5, 0.5, 0.9, 0.1])
    threshold, f1 = best_threshold(actual, p)
    expected = brute_force(actual, p)
    assert f1 == pytest.approx(max(expected.values()))
    # At 0.9: P 1, R .5, F1 2/3. At 0.5: P 2/3, R 1, F1 0.8 (best).
    assert threshold == 0.5
    assert f1 == pytest.approx(0.8)


def test_all_same_p_predicts_everyone_and_f1_is_from_the_base_rate():
    actual = series([True, False, False, False])
    p = series([0.3, 0.3, 0.3, 0.3])
    threshold, f1 = best_threshold(actual, p)
    assert threshold == 0.3
    assert f1 == pytest.approx(2 * 0.25 * 1 / 1.25)   # P .25, R 1


# p .8 T, .6 F, .4 F, .2 T. At .8: P 1, R 1/2, F1 2/3. At .6: P 1/2, R 1/2, F1 1/2.
# At .4: P 1/3, R 1/2, F1 2/5. At .2: P 1/2, R 1, F1 2/3. A tie between .8 and .2.
TIE_ACTUAL = [True, False, False, True]
TIE_P = [0.8, 0.6, 0.4, 0.2]


def test_equal_f1_at_two_thresholds_returns_one_of_them():
    threshold, f1 = best_threshold(series(TIE_ACTUAL), series(TIE_P))
    assert f1 == pytest.approx(2 / 3)
    assert threshold in (0.8, 0.2)
    assert brute_force(TIE_ACTUAL, TIE_P)[0.8] == pytest.approx(brute_force(TIE_ACTUAL, TIE_P)[0.2])


def test_equal_f1_tie_goes_to_the_lowest_threshold():
    # Pins the current behaviour (np.argmax over ascending thresholds takes the first).
    assert best_threshold(series(TIE_ACTUAL), series(TIE_P))[0] == 0.2


@pytest.mark.parametrize("seed", range(30))
def test_threshold_is_one_of_the_p_values(seed):
    rng = np.random.default_rng(500 + seed)
    actual = np.array([True, False] + list(rng.random(18) < 0.5))
    p = rng.random(20)
    threshold, _ = best_threshold(series(actual), series(p))
    assert threshold in set(p.tolist())


def test_returns_python_floats():
    threshold, f1 = best_threshold(series([True, False, True]), series([0.9, 0.1, 0.4]))
    assert type(threshold) is float
    assert type(f1) is float


def test_returns_python_floats_for_float32_p():
    threshold, f1 = best_threshold(series([True, False, True]),
                                   series([0.9, 0.1, 0.4], dtype="float32"))
    assert type(threshold) is float and type(f1) is float


def test_f1_is_between_0_and_1():
    rng = np.random.default_rng(3)
    for _ in range(50):
        actual = np.array([True, False] + list(rng.random(10) < 0.3))
        _, f1 = best_threshold(series(actual), series(rng.random(12)))
        assert 0.0 < f1 <= 1.0


def test_p_of_exactly_0_and_1_are_fine():
    actual = series([True, False, True, False])
    p = series([1.0, 0.0, 1.0, 0.0])
    assert best_threshold(actual, p) == (1.0, 1.0)


def test_does_not_warn():
    actual = series([True, False, True, False, False])
    p = series([0.9, 0.9, 0.2, 0.1, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        best_threshold(actual, p)


def test_accepts_nullable_boolean_actual():
    actual = series([True, False, True, False], dtype="boolean")
    p = series([0.8, 0.3, 0.6, 0.2])
    assert best_threshold(actual, p) == (0.6, 1.0)


def test_accepts_int_actual():
    actual = series([1, 0, 1, 0])
    p = series([0.8, 0.3, 0.6, 0.2])
    assert best_threshold(actual, p) == (0.6, 1.0)


def test_works_with_a_non_default_index():
    index = pd.Index([40, 7, 99, 3])
    actual = series([True, False, True, False], index=index)
    p = series([0.8, 0.3, 0.6, 0.2], index=index)
    assert best_threshold(actual, p) == (0.6, 1.0)


def test_does_not_change_its_inputs():
    actual = series([1, 0, 1, 0], index=[5, 6, 7, 8])
    p = series([0.8, 0.3, 0.6, 0.2], index=[5, 6, 7, 8])
    before = (actual.copy(), p.copy())
    best_threshold(actual, p)
    pd.testing.assert_series_equal(actual, before[0])
    pd.testing.assert_series_equal(p, before[1])


def test_is_deterministic():
    rng = np.random.default_rng(9)
    actual = series(np.array([True, False] + list(rng.random(48) < 0.3)))
    p = series(rng.random(50))
    assert best_threshold(actual, p) == best_threshold(actual, p)


MISALIGNED = "actual and p must have the same rows (the same index)"
ONE_CLASS = "actual must hold both outcomes to choose a threshold"


def test_best_threshold_different_index_raises():
    with pytest.raises(ValueError) as error:
        best_threshold(series([True, False], index=[0, 1]), series([0.9, 0.1], index=[1, 2]))
    assert str(error.value) == MISALIGNED


def test_best_threshold_same_labels_in_another_order_raises():
    with pytest.raises(ValueError) as error:
        best_threshold(series([True, False], index=[0, 1]), series([0.9, 0.1], index=[1, 0]))
    assert str(error.value) == MISALIGNED


def test_best_threshold_different_length_raises():
    with pytest.raises(ValueError) as error:
        best_threshold(series([True, False, True]), series([0.9, 0.1]))
    assert str(error.value) == MISALIGNED


@pytest.mark.parametrize("values", [[True, True, True], [False, False], [1, 1], [True]])
def test_best_threshold_a_single_class_raises(values):
    with pytest.raises(ValueError) as error:
        best_threshold(series(values), series([0.5] * len(values)))
    assert str(error.value) == ONE_CLASS


def test_best_threshold_empty_raises_the_one_class_message():
    with pytest.raises(ValueError) as error:
        best_threshold(series([], dtype=bool), series([], dtype=float))
    assert str(error.value) == ONE_CLASS


def test_best_threshold_index_is_checked_before_the_classes():
    with pytest.raises(ValueError) as error:
        best_threshold(series([True, True], index=[0, 1]), series([0.9, 0.1], index=[5, 6]))
    assert str(error.value) == MISALIGNED


# --- classification_scores -------------------------------------------------------------------

KEYS = ["F1", "precision", "recall", "threshold", "AUC", "avg precision", "log loss"]

# The worked example: actual T, T, F, F, T; p .9, .4, .6, .1, .7; threshold .5.
# Predicted T, F, T, F, T: TP 2, FP 1, FN 1 -> precision 2/3, recall 2/3, F1 2/3.
# AUC: payers {.9, .4, .7} vs non-payers {.6, .1}: 5 of 6 pairs ordered right -> 5/6.
# Average precision: p sorted down .9 T, .7 T, .6 F, .4 T, .1 F; recall steps 1/3 at P 1,
# 1/3 at P 1, 1/3 at P 3/4 -> 1/3 + 1/3 + 1/4 = 11/12.
# Log loss: -(ln .9 + ln .4 + ln(1-.6) + ln(1-.1) + ln .7) / 5.
EX_ACTUAL = [True, True, False, False, True]
EX_P = [0.9, 0.4, 0.6, 0.1, 0.7]
EX_LOG_LOSS = -(math.log(0.9) + math.log(0.4) + math.log(0.4) + math.log(0.9)
                + math.log(0.7)) / 5


def example_scores(threshold=0.5):
    return classification_scores(series(EX_ACTUAL), series(EX_P), threshold)


def test_scores_keys_in_order():
    assert list(example_scores()) == KEYS


def test_scores_by_hand():
    scores = example_scores()
    assert scores["F1"] == pytest.approx(2 / 3)
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["threshold"] == 0.5
    assert scores["AUC"] == pytest.approx(5 / 6)
    assert scores["avg precision"] == pytest.approx(11 / 12)
    assert scores["log loss"] == pytest.approx(EX_LOG_LOSS)


def test_scores_a_second_example_with_unequal_precision_and_recall():
    # Threshold .65: predicted .9 T, .7 T -> TP 2, FP 0, FN 1: P 1, R 2/3, F1 0.8.
    scores = example_scores(0.65)
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["F1"] == pytest.approx(0.8)


def test_a_p_equal_to_the_threshold_is_predicted_to_pay():
    # Threshold .6 exactly: the .6 row (did not pay) is predicted to pay, so FP 1.
    at = example_scores(0.6)
    above = example_scores(0.6000001)
    assert at["precision"] == pytest.approx(2 / 3)    # .9 T, .6 F, .7 T
    assert above["precision"] == pytest.approx(1.0)   # .9 T, .7 T


def test_threshold_free_scores_do_not_depend_on_the_threshold():
    a, b = example_scores(0.05), example_scores(0.95)
    for key in ("AUC", "avg precision", "log loss"):
        assert a[key] == b[key], key


def test_threshold_is_returned_as_a_python_float():
    for threshold in (1, np.float32(0.5), np.float64(0.25), 0):
        value = example_scores(threshold)["threshold"]
        assert type(value) is float
        assert value == float(threshold)


def test_threshold_above_every_p_gives_zero_recall_and_precision_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = example_scores(0.99)
    assert scores["recall"] == 0.0
    assert scores["precision"] == 0.0
    assert scores["F1"] == 0.0
    assert scores["AUC"] == pytest.approx(5 / 6)


def test_threshold_at_or_below_every_p_gives_recall_one_and_the_base_rate_precision():
    scores = example_scores(0.1)
    assert scores["recall"] == 1.0
    assert scores["precision"] == pytest.approx(3 / 5)
    assert scores["F1"] == pytest.approx(2 * 0.6 / 1.6)


def test_perfect_scores():
    scores = classification_scores(series([True, False, True, False]),
                                   series([0.9, 0.2, 0.7, 0.1]), 0.5)
    assert (scores["F1"], scores["precision"], scores["recall"]) == (1.0, 1.0, 1.0)
    assert scores["AUC"] == 1.0
    assert scores["avg precision"] == 1.0


def test_log_loss_with_p_of_0_and_1_stays_finite():
    scores = classification_scores(series([True, False]), series([0.0, 1.0]), 0.5)
    assert math.isfinite(scores["log loss"])
    assert scores["log loss"] > 10


def test_scores_with_nullable_boolean_and_int_actual_are_the_same():
    plain = example_scores()
    for actual in (series(EX_ACTUAL, dtype="boolean"), series([int(a) for a in EX_ACTUAL])):
        scores = classification_scores(actual, series(EX_P), 0.5)
        assert scores == plain


def test_scores_work_with_a_non_default_index():
    index = pd.Index([90, 3, 17, 4, 55])
    scores = classification_scores(series(EX_ACTUAL, index=index), series(EX_P, index=index),
                                   0.5)
    assert scores == example_scores()


def test_scores_do_not_change_their_inputs():
    actual, p = series(EX_ACTUAL, dtype="boolean"), series(EX_P)
    before = (actual.copy(), p.copy())
    classification_scores(actual, p, 0.5)
    pd.testing.assert_series_equal(actual, before[0])
    pd.testing.assert_series_equal(p, before[1])


def test_scores_misaligned_index_raises():
    with pytest.raises(ValueError) as error:
        classification_scores(series([True, False], index=[0, 1]),
                              series([0.9, 0.1], index=[0, 2]), 0.5)
    assert str(error.value) == MISALIGNED


def test_scores_same_labels_in_another_order_raises():
    with pytest.raises(ValueError) as error:
        classification_scores(series([True, False], index=[0, 1]),
                              series([0.9, 0.1], index=[1, 0]), 0.5)
    assert str(error.value) == MISALIGNED


# --- classification_scores and best_threshold together ---------------------------------------

@pytest.mark.parametrize("seed", range(100))
def test_scores_at_the_best_threshold_give_the_same_f1(seed):
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(3, 60))
    actual = np.array([True, False] + list(rng.random(n - 2) < rng.uniform(0.1, 0.9)))
    p = np.round(rng.random(n), int(rng.integers(1, 4)))
    threshold, f1 = best_threshold(series(actual), series(p))
    scores = classification_scores(series(actual), series(p), threshold)
    assert scores["F1"] == pytest.approx(f1, abs=1e-12)
    assert scores["threshold"] == threshold


def test_no_threshold_beats_the_best_one_in_classification_scores():
    rng = np.random.default_rng(42)
    actual = series(np.array([True, False] + list(rng.random(38) < 0.35)))
    p = series(np.round(rng.random(40), 2))
    _, f1 = best_threshold(actual, p)
    for t in sorted(set(p.tolist())) + [0.0, 1.0, 0.555]:
        assert classification_scores(actual, p, t)["F1"] <= f1 + 1e-12, t
