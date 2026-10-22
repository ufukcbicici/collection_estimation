"""Tests for `hurdle/diagnostic.py`: `DiagnosticSplits`, `diagnostic_splits`, `best_threshold`,
`classification_scores` (U10.4a), and `Candidate`, `class_weight_argument`, `hgb_candidates`,
`lr_candidates`, `grid_search`, `choose_and_test`, `permutation_importances` and
`write_tables` (U10.4b/c).

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

KEYS = ["F1", "precision", "recall", "threshold", "AUC", "avg precision", "log loss",
        "brier"]

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
    for key in ("AUC", "avg precision", "log loss", "brier"):
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


# =============================================================================================
# U10.4b / U10.4c: Candidate, class_weight_argument, hgb_candidates, lr_candidates,
# grid_search, choose_and_test, permutation_importances, write_tables.
#
# Every table is tiny (a few hundred rows) and every real grid has a few small fits, so these
# run in seconds. Spy fits record their arguments; fake models return p chosen by hand.
# =============================================================================================

import io  # noqa: E402
import itertools  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from collection_estimation.hurdle.diagnostic import (  # noqa: E402
    Candidate,
    choose_and_test,
    class_weight_argument,
    grid_search,
    hgb_candidates,
    lr_candidates,
    permutation_importances,
    write_tables,
)
from collection_estimation.hurdle.settings import (  # noqa: E402
    DiagnosticSettings,
    Stage1Settings,
)
from collection_estimation.hurdle.stage1 import (  # noqa: E402
    fit_stage1,
    fit_stage1_baseline,
    predict_paid,
)


def stage1_settings(**changes):
    base = Stage1Settings(max_iter=7, learning_rate=0.3, max_leaf_nodes=5, min_samples_leaf=4,
                          random_seed=11, feature_blocks=("history", "calendar"),
                          company_as_feature=True, baseline_c=0.5)
    return dataclasses.replace(base, **changes)


def diagnostic_settings(**changes):
    base = DiagnosticSettings(run=True, origin=40, hgb_max_iter=(10, 30),
                              hgb_learning_rate=(0.05, 0.1, 0.2), hgb_max_leaf_nodes=(3, 7),
                              lr_c=(0.01, 1.0, 100.0), class_weights=("none", "balanced"),
                              selection_metric="log loss", calibration_groups=4,
                              reference_window=13, reference_prior_weeks=1.0,
                              amount_reference_window=52,
                              importance_repeats=2, output_folder=Path("unused"))
    return dataclasses.replace(base, **changes)


# --- Candidate -------------------------------------------------------------------------------

def test_candidate_fields_in_order():
    assert [field.name for field in dataclasses.fields(Candidate)] == [
        "params", "settings", "class_weight"]


def test_candidate_is_frozen():
    candidate = Candidate(params={"C": 1.0}, settings=stage1_settings(), class_weight="none")
    for name in ("params", "settings", "class_weight"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(candidate, name, None)


# --- class_weight_argument -------------------------------------------------------------------

def test_class_weight_none_is_none():
    assert class_weight_argument("none") is None


def test_class_weight_balanced_is_balanced():
    assert class_weight_argument("balanced") == "balanced"


@pytest.mark.parametrize("value", ["None", "Balanced", "BALANCED", "", " none", "balanced ",
                                   "auto", None, 0])
def test_class_weight_anything_else_raises_the_exact_message(value):
    with pytest.raises(ValueError) as error:
        class_weight_argument(value)
    assert str(error.value) == f"class weight {value!r} is not 'none' or 'balanced'"


# --- hgb_candidates --------------------------------------------------------------------------

HGB_KEYS = ["max_iter", "learning_rate", "max_leaf_nodes", "class_weight"]


def test_hgb_candidates_count_is_the_product_of_the_grid():
    candidates = hgb_candidates(stage1_settings(), diagnostic_settings())
    assert len(candidates) == 2 * 3 * 2 * 2


def test_hgb_candidates_follow_itertools_product_order():
    diagnostic = diagnostic_settings()
    candidates = hgb_candidates(stage1_settings(), diagnostic)
    expected = list(itertools.product(diagnostic.hgb_max_iter, diagnostic.hgb_learning_rate,
                                      diagnostic.hgb_max_leaf_nodes, diagnostic.class_weights))
    assert [tuple(c.params[k] for k in HGB_KEYS) for c in candidates] == expected
    # The last axis (class weight) varies fastest.
    assert [c.class_weight for c in candidates[:4]] == ["none", "balanced", "none", "balanced"]


def test_hgb_candidates_params_keys_in_order():
    for candidate in hgb_candidates(stage1_settings(), diagnostic_settings()):
        assert list(candidate.params) == HGB_KEYS


def test_hgb_candidates_class_weight_matches_the_params():
    for candidate in hgb_candidates(stage1_settings(), diagnostic_settings()):
        assert candidate.class_weight == candidate.params["class_weight"]


def test_hgb_candidates_settings_have_the_grid_values_in_place():
    for candidate in hgb_candidates(stage1_settings(), diagnostic_settings()):
        assert isinstance(candidate.settings, Stage1Settings)
        assert candidate.settings.max_iter == candidate.params["max_iter"]
        assert candidate.settings.learning_rate == candidate.params["learning_rate"]
        assert candidate.settings.max_leaf_nodes == candidate.params["max_leaf_nodes"]


def test_hgb_candidates_every_other_setting_is_unchanged():
    stage1 = stage1_settings()
    for candidate in hgb_candidates(stage1, diagnostic_settings()):
        assert candidate.settings == dataclasses.replace(
            stage1, max_iter=candidate.params["max_iter"],
            learning_rate=candidate.params["learning_rate"],
            max_leaf_nodes=candidate.params["max_leaf_nodes"])
        for name in ("min_samples_leaf", "random_seed", "feature_blocks",
                     "company_as_feature", "baseline_c"):
            assert getattr(candidate.settings, name) == getattr(stage1, name), name


def test_hgb_candidates_do_not_change_the_forecast_settings():
    stage1, diagnostic = stage1_settings(), diagnostic_settings()
    before = (dataclasses.replace(stage1), dataclasses.replace(diagnostic))
    candidates = hgb_candidates(stage1, diagnostic)
    assert stage1 == before[0] and diagnostic == before[1]
    assert all(candidate.settings is not stage1 for candidate in candidates)
    assert stage1.max_iter == 7 and stage1.learning_rate == 0.3 and stage1.max_leaf_nodes == 5


def test_hgb_candidates_one_value_per_axis_gives_one_candidate():
    diagnostic = diagnostic_settings(hgb_max_iter=(5,), hgb_learning_rate=(0.1,),
                                     hgb_max_leaf_nodes=(4,), class_weights=("balanced",))
    candidates = hgb_candidates(stage1_settings(), diagnostic)
    assert len(candidates) == 1
    assert candidates[0].params == {"max_iter": 5, "learning_rate": 0.1, "max_leaf_nodes": 4,
                                    "class_weight": "balanced"}
    assert candidates[0].class_weight == "balanced"


@pytest.mark.parametrize("axis", ["hgb_max_iter", "hgb_learning_rate", "hgb_max_leaf_nodes",
                                  "class_weights"])
def test_hgb_candidates_an_empty_axis_gives_no_candidates(axis):
    assert hgb_candidates(stage1_settings(), diagnostic_settings(**{axis: ()})) == []


def test_hgb_candidates_ignore_the_lr_grid():
    one = hgb_candidates(stage1_settings(), diagnostic_settings(lr_c=(0.1,)))
    two = hgb_candidates(stage1_settings(), diagnostic_settings(lr_c=(1.0, 2.0, 3.0)))
    assert [c.params for c in one] == [c.params for c in two]
    assert [c.settings for c in one] == [c.settings for c in two]


# --- lr_candidates ---------------------------------------------------------------------------

def test_lr_candidates_count_and_order():
    diagnostic = diagnostic_settings()
    candidates = lr_candidates(stage1_settings(), diagnostic)
    assert len(candidates) == 3 * 2
    assert [(c.params["C"], c.params["class_weight"]) for c in candidates] == list(
        itertools.product(diagnostic.lr_c, diagnostic.class_weights))


def test_lr_candidates_params_keys_in_order():
    for candidate in lr_candidates(stage1_settings(), diagnostic_settings()):
        assert list(candidate.params) == ["C", "class_weight"]
        assert candidate.class_weight == candidate.params["class_weight"]


def test_lr_candidates_replace_only_baseline_c():
    stage1 = stage1_settings()
    for candidate in lr_candidates(stage1, diagnostic_settings()):
        assert isinstance(candidate.settings, Stage1Settings)
        assert candidate.settings.baseline_c == candidate.params["C"]
        assert candidate.settings == dataclasses.replace(stage1,
                                                         baseline_c=candidate.params["C"])
        for name in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                     "random_seed", "feature_blocks", "company_as_feature"):
            assert getattr(candidate.settings, name) == getattr(stage1, name), name


def test_lr_candidates_do_not_change_the_forecast_settings():
    stage1, diagnostic = stage1_settings(), diagnostic_settings()
    before = (dataclasses.replace(stage1), dataclasses.replace(diagnostic))
    lr_candidates(stage1, diagnostic)
    assert stage1 == before[0] and diagnostic == before[1]
    assert stage1.baseline_c == 0.5


def test_lr_candidates_ignore_the_hgb_grid():
    one = lr_candidates(stage1_settings(), diagnostic_settings(hgb_max_iter=(1,)))
    two = lr_candidates(stage1_settings(), diagnostic_settings(hgb_max_iter=(1, 2, 3, 4)))
    assert [c.params for c in one] == [c.params for c in two]


@pytest.mark.parametrize("axis", ["lr_c", "class_weights"])
def test_lr_candidates_an_empty_axis_gives_no_candidates(axis):
    assert lr_candidates(stage1_settings(), diagnostic_settings(**{axis: ()})) == []


# --- tiny data for the real fits -------------------------------------------------------------

def tiny_data(n=300, seed=0, company=False, index_start=0):
    """x with an informative column `signal`, a pure-noise column `noise` and h; paid depends
    on signal only (and, with `company`, on the company: TR04 pays more often)."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    x = pd.DataFrame({"h": rng.integers(1, 6, size=n).astype("float64"),
                      "signal": signal, "noise": noise},
                     index=pd.RangeIndex(index_start, index_start + n))
    logit = 2.5 * signal - 0.8
    if company:
        codes = rng.choice(["TR02", "TR04", "TR91"], size=n)
        x["company_code"] = pd.Categorical(codes, categories=["TR02", "TR04", "TR91"])
        logit = logit + np.where(codes == "TR04", 2.0, 0.0)
    paid = pd.Series(rng.random(n) < 1 / (1 + np.exp(-logit)), index=x.index, name="paid")
    return x, paid


@pytest.fixture(scope="module")
def data():
    train_x, train_paid = tiny_data(300, seed=1)
    validation_x, validation_paid = tiny_data(200, seed=2, index_start=1000)
    test_x, test_paid = tiny_data(200, seed=3, index_start=5000)
    return SimpleNamespace(train_x=train_x, train_paid=train_paid, validation_x=validation_x,
                           validation_paid=validation_paid, test_x=test_x, test_paid=test_paid)


def run_quietly(function, *args, **kwargs):
    with redirect_stdout(io.StringIO()) as out:
        result = function(*args, **kwargs)
    return result, out.getvalue()


SMALL_HGB = diagnostic_settings(hgb_max_iter=(3, 15), hgb_learning_rate=(0.3,),
                                hgb_max_leaf_nodes=(3,), class_weights=("none", "balanced"))
SMALL_LR = diagnostic_settings(lr_c=(0.001, 1.0, 100.0), class_weights=("none", "balanced"))


# --- fakes -----------------------------------------------------------------------------------

class FakeModel:
    """A model predict_paid accepts: p is given by hand, as a function of x."""

    def __init__(self, columns, p_of_x):
        self.feature_names_in_ = np.array(list(columns), dtype=object)
        self.classes_ = np.array([False, True])
        self.p_of_x = p_of_x
        self.predicted_on = []

    def predict_proba(self, x):
        self.predicted_on.append(x)
        p = np.asarray(self.p_of_x(x), dtype=float)
        return np.column_stack([1 - p, p])


class SpyFit:
    """A fit that records each call and returns a FakeModel whose p comes from `p_for`
    (a function of the settings and class_weight passed in, and of the x predicted)."""

    def __init__(self, p_for):
        self.p_for = p_for
        self.calls = []
        self.models = []

    def __call__(self, train_x, paid, settings, class_weight="NOT GIVEN"):
        self.calls.append(SimpleNamespace(train_x=train_x, paid=paid, settings=settings,
                                          class_weight=class_weight))
        model = FakeModel(train_x.columns,
                          lambda x, s=settings, w=class_weight: self.p_for(s, w, x))
        self.models.append(model)
        return model


def fake_candidates(n, weights=None):
    weights = weights or ["none"] * n
    return [Candidate(params={"k": i, "class_weight": weights[i]},
                      settings=SimpleNamespace(tag=i), class_weight=weights[i])
            for i in range(n)]


# Hand-made validation rows: 8 rows, 4 payers.
VAL_X = pd.DataFrame({"a": np.arange(8, dtype="float64")}, index=pd.Index(range(20, 28)))
VAL_PAID = pd.Series([True, True, False, True, False, False, True, False], index=VAL_X.index)
TRAIN_X = pd.DataFrame({"a": np.arange(6, dtype="float64")}, index=pd.Index(range(6)))
TRAIN_PAID = pd.Series([True, False, True, False, True, False], index=TRAIN_X.index)
# Three p vectors with different average precision on VAL_PAID: GOOD > MID > BAD.
P_GOOD = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.3, 0.6, 0.15])   # perfect ranking
P_MID = np.array([0.9, 0.3, 0.8, 0.7, 0.1, 0.2, 0.25, 0.6])
P_BAD = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.7, 0.4, 0.6])


def test_the_hand_made_p_vectors_rank_good_mid_bad():
    ap = [average_precision_score(VAL_PAID, p) for p in (P_GOOD, P_MID, P_BAD)]
    assert ap[0] > ap[1] > ap[2]


SCORE_KEYS = ["F1", "precision", "recall", "threshold", "AUC", "avg precision", "log loss",
              "brier"]


# --- grid_search: shape ----------------------------------------------------------------------

def fake_grid(assignment, weights=None, name="M", metric="avg precision"):
    """grid_search over len(assignment) fake candidates; candidate i predicts assignment[i].
    Ranked by average precision unless `metric` says otherwise (the hand-made P_GOOD > P_MID
    > P_BAD order is an average-precision order)."""
    candidates = fake_candidates(len(assignment), weights)
    fit = SpyFit(lambda s, w, x: assignment[s.tag])
    grid, out = run_quietly(grid_search, name, fit, candidates, TRAIN_X, TRAIN_PAID, VAL_X,
                            VAL_PAID, metric=metric)
    return grid, out, fit, candidates


def test_grid_search_one_row_per_candidate():
    grid, _, _, _ = fake_grid([P_GOOD, P_MID, P_BAD, P_MID])
    assert len(grid) == 4
    assert sorted(grid["candidate"]) == [0, 1, 2, 3]


def test_grid_search_columns_in_order():
    grid, _, _, _ = fake_grid([P_GOOD, P_MID])
    assert list(grid.columns) == ["candidate", "k", "class_weight"] + SCORE_KEYS + [
        "fit seconds"]


def test_grid_search_hgb_params_columns_in_order():
    candidates = hgb_candidates(stage1_settings(), SMALL_HGB)
    fit = SpyFit(lambda s, w, x: P_MID)
    grid, _ = run_quietly(grid_search, "HGB", fit, candidates, TRAIN_X, TRAIN_PAID, VAL_X,
                          VAL_PAID, metric="log loss")
    assert list(grid.columns) == ["candidate"] + HGB_KEYS + SCORE_KEYS + ["fit seconds"]


def test_grid_search_has_a_fresh_index():
    grid, _, _, _ = fake_grid([P_BAD, P_MID, P_GOOD])
    assert isinstance(grid.index, pd.RangeIndex)
    assert list(grid.index) == [0, 1, 2]


def test_grid_search_fit_seconds_are_non_negative_floats():
    grid, _, _, _ = fake_grid([P_BAD, P_MID, P_GOOD])
    assert (grid["fit seconds"] >= 0).all()
    assert grid["fit seconds"].dtype == np.float64


def test_grid_search_candidate_column_holds_ints():
    grid, _, _, _ = fake_grid([P_BAD, P_MID, P_GOOD])
    assert pd.api.types.is_integer_dtype(grid["candidate"])


# --- grid_search: sort -----------------------------------------------------------------------

def test_grid_search_sorted_by_avg_precision_best_first():
    grid, _, _, _ = fake_grid([P_BAD, P_MID, P_GOOD])
    assert list(grid["candidate"]) == [2, 1, 0]
    assert grid["avg precision"].is_monotonic_decreasing


def test_grid_search_ties_keep_grid_order():
    # Candidates 1 and 3 tie at the top, 0 and 2 at the bottom, 4 in the middle.
    grid, _, _, _ = fake_grid([P_BAD, P_GOOD, P_BAD, P_GOOD, P_MID])
    assert list(grid["candidate"]) == [1, 3, 4, 0, 2]


def test_grid_search_all_tied_keeps_grid_order_exactly():
    grid, _, _, _ = fake_grid([P_MID] * 7)
    assert list(grid["candidate"]) == list(range(7))


def test_grid_search_ties_keep_grid_order_with_many_interleaved_ties():
    # Many ties interleaved: a non-stable sort (quicksort) could swap them.
    assignment = [P_GOOD if i % 3 == 0 else (P_MID if i % 3 == 1 else P_BAD)
                  for i in range(30)]
    grid, _, _, _ = fake_grid(assignment)
    expected = ([i for i in range(30) if i % 3 == 0] + [i for i in range(30) if i % 3 == 1]
                + [i for i in range(30) if i % 3 == 2])
    assert list(grid["candidate"]) == expected


def test_grid_search_candidate_maps_back_to_its_candidate():
    assignment = [P_BAD, P_GOOD, P_MID, P_GOOD]
    grid, _, _, candidates = fake_grid(assignment, weights=["none", "balanced", "none",
                                                            "balanced"])
    for _, row in grid.iterrows():
        candidate = candidates[int(row["candidate"])]
        assert row["k"] == candidate.params["k"]
        assert row["class_weight"] == candidate.params["class_weight"]
        expected = average_precision_score(VAL_PAID, assignment[int(row["candidate"])])
        assert row["avg precision"] == pytest.approx(expected)


# --- grid_search: scores and thresholds per candidate ----------------------------------------

def test_grid_search_each_row_has_its_own_best_threshold_and_scores():
    assignment = [P_BAD, P_GOOD, P_MID]
    grid, _, _, _ = fake_grid(assignment)
    for _, row in grid.iterrows():
        p = pd.Series(assignment[int(row["candidate"])], index=VAL_X.index)
        threshold, _ = best_threshold(VAL_PAID, p)
        assert row["threshold"] == threshold
        expected = classification_scores(VAL_PAID, p, threshold)
        for key in SCORE_KEYS:
            assert row[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-15), key
    # The hand-made p give different thresholds: they are chosen per candidate.
    assert grid["threshold"].nunique() == 3


def test_grid_search_passes_the_class_weight_argument_to_fit():
    weights = ["none", "balanced", "balanced", "none"]
    _, _, fit, candidates = fake_grid([P_MID] * 4, weights=weights)
    assert [call.class_weight for call in fit.calls] == [None, "balanced", "balanced", None]
    for call, candidate in zip(fit.calls, candidates):
        assert call.settings is candidate.settings


def test_grid_search_rejects_a_bad_class_weight_with_the_exact_message():
    candidates = [Candidate(params={"k": 0}, settings=SimpleNamespace(tag=0),
                            class_weight="Balanced")]
    fit = SpyFit(lambda s, w, x: P_MID)
    with pytest.raises(ValueError) as error:
        run_quietly(grid_search, "M", fit, candidates, TRAIN_X, TRAIN_PAID, VAL_X, VAL_PAID,
                    metric="log loss")
    assert str(error.value) == "class weight 'Balanced' is not 'none' or 'balanced'"
    assert fit.calls == []


def test_grid_search_fits_on_train_rows_only():
    _, _, fit, _ = fake_grid([P_MID, P_GOOD, P_BAD])
    assert len(fit.calls) == 3
    for call in fit.calls:
        assert call.train_x is TRAIN_X
        assert call.paid is TRAIN_PAID


def test_grid_search_predicts_and_scores_validation_rows_only():
    _, _, fit, _ = fake_grid([P_MID, P_GOOD, P_BAD])
    for model in fit.models:
        assert len(model.predicted_on) == 1
        assert model.predicted_on[0] is VAL_X


def test_grid_search_scores_against_validation_paid():
    # P_GOOD ranks VAL_PAID perfectly: AUC 1 and average precision 1.
    grid, _, _, _ = fake_grid([P_GOOD])
    assert grid.loc[0, "AUC"] == 1.0
    assert grid.loc[0, "avg precision"] == 1.0


def test_grid_search_fits_once_per_candidate_in_grid_order():
    _, _, fit, _ = fake_grid([P_BAD, P_GOOD, P_MID, P_MID, P_GOOD])
    assert [call.settings.tag for call in fit.calls] == [0, 1, 2, 3, 4]


def test_grid_search_does_not_change_its_inputs():
    train_x, train_paid, val_x, val_paid = (TRAIN_X.copy(), TRAIN_PAID.copy(), VAL_X.copy(),
                                            VAL_PAID.copy())
    candidates = fake_candidates(3)
    params_before = [dict(c.params) for c in candidates]
    fit = SpyFit(lambda s, w, x: [P_BAD, P_GOOD, P_MID][s.tag])
    run_quietly(grid_search, "M", fit, candidates, train_x, train_paid, val_x, val_paid,
                metric="brier")
    pd.testing.assert_frame_equal(train_x, TRAIN_X)
    pd.testing.assert_series_equal(train_paid, TRAIN_PAID)
    pd.testing.assert_frame_equal(val_x, VAL_X)
    pd.testing.assert_series_equal(val_paid, VAL_PAID)
    assert [c.params for c in candidates] == params_before


# --- grid_search: printout and errors --------------------------------------------------------

def test_grid_search_prints_one_progress_line_per_candidate_in_grid_order():
    assignment = [P_BAD, P_GOOD, P_MID]
    _, out, _, candidates = fake_grid(assignment, name="HGB")
    lines = out.splitlines()
    assert len(lines) == 3
    for number, (line, candidate, p) in enumerate(zip(lines, candidates, assignment), start=1):
        ap = average_precision_score(VAL_PAID, p)
        prefix = f"  ... HGB {number} of 3: {candidate.params}, avg precision {ap:.4f} ("
        assert line.startswith(prefix), line
        assert line.endswith(" s)"), line


def test_grid_search_empty_candidate_list_raises_the_exact_message():
    fit = SpyFit(lambda s, w, x: P_MID)
    with pytest.raises(ValueError) as error:
        grid_search("LR", fit, [], TRAIN_X, TRAIN_PAID, VAL_X, VAL_PAID, metric="log loss")
    assert str(error.value) == "LR: the grid has no settings to try"
    assert fit.calls == []


def test_grid_search_one_class_validation_raises_from_best_threshold():
    one_class = pd.Series([True] * 8, index=VAL_X.index)
    fit = SpyFit(lambda s, w, x: P_MID)
    with pytest.raises(ValueError) as error:
        run_quietly(grid_search, "M", fit, fake_candidates(1), TRAIN_X, TRAIN_PAID, VAL_X,
                    one_class, metric="log loss")
    assert str(error.value) == ONE_CLASS


# --- grid_search: real fits ------------------------------------------------------------------

@pytest.mark.parametrize("model", ["HGB", "LR"])
def test_grid_search_real_thresholds_equal_an_independent_best_threshold(model, data):
    if model == "HGB":
        fit, candidates = fit_stage1, hgb_candidates(stage1_settings(), SMALL_HGB)
    else:
        fit, candidates = fit_stage1_baseline, lr_candidates(stage1_settings(), SMALL_LR)
    grid, _ = run_quietly(grid_search, model, fit, candidates, data.train_x, data.train_paid,
                          data.validation_x, data.validation_paid, metric="avg precision")
    assert len(grid) == len(candidates)
    thresholds = {}
    for _, row in grid.iterrows():
        candidate = candidates[int(row["candidate"])]
        for key, value in candidate.params.items():
            assert row[key] == value, key
        fitted = fit(data.train_x, data.train_paid, candidate.settings,
                     class_weight=class_weight_argument(candidate.class_weight))
        p = predict_paid(fitted, data.validation_x)
        threshold, _ = best_threshold(data.validation_paid, p)
        assert row["threshold"] == threshold
        expected = classification_scores(data.validation_paid, p, threshold)
        for key in SCORE_KEYS:
            assert row[key] == pytest.approx(expected[key], rel=1e-9), key
        thresholds[int(row["candidate"])] = threshold
    # Balanced class weights move p, so the per-candidate thresholds differ.
    assert len(set(thresholds.values())) > 1
    assert grid["avg precision"].is_monotonic_decreasing


# --- choose_and_test -------------------------------------------------------------------------

TEST_X = pd.DataFrame({"a": np.arange(6, dtype="float64")}, index=pd.Index(range(50, 56)))
TEST_PAID = pd.Series([True, False, True, False, True, False], index=TEST_X.index)
REFIT_X = pd.DataFrame({"a": np.arange(10, dtype="float64")}, index=pd.Index(range(100, 110)))
REFIT_PAID = pd.Series([True, False] * 5, index=REFIT_X.index)
# On the test rows, the best F1 threshold is 0.35 (F1 1); at 0.65 only one payer is caught.
P_TEST = np.array([0.7, 0.3, 0.4, 0.1, 0.35, 0.2])


def hand_grid(rows):
    """A grid as grid_search returns it: candidate, k, class_weight, the scores."""
    return pd.DataFrame([{"candidate": c, "k": c, "class_weight": w, "F1": 0.5,
                          "precision": 0.5, "recall": 0.5, "threshold": t, "AUC": 0.5,
                          "avg precision": ap, "log loss": 0.7, "brier": 0.25,
                          "fit seconds": 0.0}
                         for c, w, t, ap in rows])


def test_choose_and_test_refits_the_top_rows_candidate_on_the_refit_rows():
    candidates = fake_candidates(3, ["none", "balanced", "none"])
    grid = hand_grid([(1, "balanced", 0.65, 0.9), (0, "none", 0.2, 0.8), (2, "none", 0.5, 0.1)])
    fit = SpyFit(lambda s, w, x: P_TEST)
    choose_and_test("M", fit, candidates, grid, REFIT_X, REFIT_PAID, TEST_X, TEST_PAID)
    assert len(fit.calls) == 1
    call = fit.calls[0]
    assert call.settings is candidates[1].settings
    assert call.class_weight == "balanced"
    assert call.train_x is REFIT_X
    assert call.paid is REFIT_PAID
    assert len(fit.models[0].predicted_on) == 1
    assert fit.models[0].predicted_on[0] is TEST_X


def test_choose_and_test_takes_the_first_row_not_the_highest_avg_precision():
    # Pins "grid.iloc[0]": the grid is trusted to be sorted already.
    candidates = fake_candidates(3)
    grid = hand_grid([(2, "none", 0.5, 0.1), (0, "none", 0.2, 0.9), (1, "none", 0.3, 0.8)])
    fit = SpyFit(lambda s, w, x: P_TEST)
    result = choose_and_test("M", fit, candidates, grid, REFIT_X, REFIT_PAID, TEST_X,
                             TEST_PAID)
    assert fit.calls[0].settings is candidates[2].settings
    assert result["k"] == 2


def test_choose_and_test_uses_the_validation_threshold_unchanged():
    candidates = fake_candidates(2)
    grid = hand_grid([(0, "none", 0.65, 0.9), (1, "none", 0.35, 0.5)])
    fit = SpyFit(lambda s, w, x: P_TEST)
    result = choose_and_test("M", fit, candidates, grid, REFIT_X, REFIT_PAID, TEST_X,
                             TEST_PAID)
    p = pd.Series(P_TEST, index=TEST_X.index)
    test_best, test_best_f1 = best_threshold(TEST_PAID, p)
    assert test_best == 0.35 and test_best_f1 == 1.0
    assert result["threshold"] == 0.65
    expected = classification_scores(TEST_PAID, p, 0.65)
    for key in SCORE_KEYS:
        assert result[key] == pytest.approx(expected[key]), key
    assert result["F1"] < test_best_f1   # not re-tuned on the test rows
    assert result["recall"] == pytest.approx(1 / 3)


def test_choose_and_test_returns_model_then_params_then_scores():
    candidates = fake_candidates(1, ["balanced"])
    grid = hand_grid([(0, "balanced", 0.35, 0.9)])
    fit = SpyFit(lambda s, w, x: P_TEST)
    result = choose_and_test("LR", fit, candidates, grid, REFIT_X, REFIT_PAID, TEST_X,
                             TEST_PAID)
    assert list(result) == ["model", "k", "class_weight"] + SCORE_KEYS
    p = pd.Series(P_TEST, index=TEST_X.index)
    assert result == {"model": "LR", "k": 0, "class_weight": "balanced",
                      **classification_scores(TEST_PAID, p, 0.35)}


def test_choose_and_test_threshold_is_a_float():
    candidates = fake_candidates(1)
    grid = hand_grid([(0, "none", 0.35, 0.9)])
    fit = SpyFit(lambda s, w, x: P_TEST)
    result = choose_and_test("M", fit, candidates, grid, REFIT_X, REFIT_PAID, TEST_X,
                             TEST_PAID)
    assert type(result["threshold"]) is float


def test_choose_and_test_accepts_a_float_candidate_column():
    # e.g. a grid read back from CSV, or a row of mixed dtypes.
    candidates = fake_candidates(3)
    grid = hand_grid([(2, "none", 0.35, 0.9)]).astype({"candidate": "float64"})
    fit = SpyFit(lambda s, w, x: P_TEST)
    choose_and_test("M", fit, candidates, grid, REFIT_X, REFIT_PAID, TEST_X, TEST_PAID)
    assert fit.calls[0].settings is candidates[2].settings


def test_choose_and_test_does_not_change_the_grid():
    candidates = fake_candidates(2)
    grid = hand_grid([(1, "none", 0.65, 0.9), (0, "none", 0.35, 0.5)])
    before = grid.copy(deep=True)
    choose_and_test("M", SpyFit(lambda s, w, x: P_TEST), candidates, grid, REFIT_X,
                    REFIT_PAID, TEST_X, TEST_PAID)
    pd.testing.assert_frame_equal(grid, before)


def test_choose_and_test_after_a_real_grid_search(data):
    candidates = lr_candidates(stage1_settings(), SMALL_LR)
    grid, _ = run_quietly(grid_search, "LR", fit_stage1_baseline, candidates, data.train_x,
                          data.train_paid, data.validation_x, data.validation_paid,
                          metric="avg precision")
    refit_x = pd.concat([data.train_x, data.validation_x])
    refit_paid = pd.concat([data.train_paid, data.validation_paid])
    result = choose_and_test("LR", fit_stage1_baseline, candidates, grid, refit_x, refit_paid,
                             data.test_x, data.test_paid)
    best = candidates[int(grid.iloc[0]["candidate"])]
    model = fit_stage1_baseline(refit_x, refit_paid, best.settings,
                                class_weight=class_weight_argument(best.class_weight))
    p = predict_paid(model, data.test_x)
    expected = classification_scores(data.test_paid, p, float(grid.iloc[0]["threshold"]))
    assert result["model"] == "LR"
    assert result["C"] == best.params["C"]
    assert result["class_weight"] == best.params["class_weight"]
    assert result["threshold"] == grid.iloc[0]["threshold"]
    for key in SCORE_KEYS:
        assert result[key] == pytest.approx(expected[key], rel=1e-9), key


# --- permutation_importances -----------------------------------------------------------------

def importance_of(table, feature):
    return float(table.loc[table["feature"] == feature, "log loss rise"].iloc[0])


@pytest.fixture(scope="module")
def lr_model(data):
    return fit_stage1_baseline(data.train_x, data.train_paid, stage1_settings(baseline_c=1.0))


@pytest.fixture(scope="module")
def company_data():
    train_x, train_paid = tiny_data(400, seed=21, company=True)
    validation_x, validation_paid = tiny_data(300, seed=22, company=True, index_start=900)
    return SimpleNamespace(train_x=train_x, train_paid=train_paid, validation_x=validation_x,
                           validation_paid=validation_paid)


@pytest.fixture(scope="module")
def hgb_company_model(company_data):
    return fit_stage1(company_data.train_x, company_data.train_paid,
                      stage1_settings(max_iter=30, learning_rate=0.1, max_leaf_nodes=4,
                                      min_samples_leaf=10))


def test_permutation_importances_columns_and_one_row_per_feature(lr_model, data):
    table = permutation_importances(lr_model, data.validation_x, data.validation_paid, 3, 0)
    assert list(table.columns) == ["feature", "log loss rise", "std"]
    assert sorted(table["feature"]) == sorted(data.validation_x.columns)
    assert len(table) == data.validation_x.shape[1]


def test_permutation_importances_sorted_descending_with_a_fresh_index(lr_model, data):
    table = permutation_importances(lr_model, data.validation_x, data.validation_paid, 3, 0)
    assert table["log loss rise"].is_monotonic_decreasing
    assert isinstance(table.index, pd.RangeIndex)
    assert list(table.index) == list(range(len(table)))


def test_permutation_importances_informative_column_rises_noise_does_not(lr_model, data):
    table = permutation_importances(lr_model, data.validation_x, data.validation_paid, 5, 0)
    assert table.iloc[0]["feature"] == "signal"
    assert importance_of(table, "signal") > 0.2
    assert abs(importance_of(table, "noise")) < 0.03
    assert abs(importance_of(table, "h")) < 0.03


def test_permutation_importances_std_is_non_negative(lr_model, data):
    table = permutation_importances(lr_model, data.validation_x, data.validation_paid, 4, 0)
    assert (table["std"] >= 0).all()


def test_permutation_importances_match_sklearn_on_neg_log_loss(lr_model, data):
    # The rise is sklearn's importances_mean for scoring="neg_log_loss": the mean fall of
    # minus the log loss, i.e. the mean rise of the log loss after shuffling.
    from sklearn.inspection import permutation_importance
    table = permutation_importances(lr_model, data.validation_x, data.validation_paid, 3, 7)
    result = permutation_importance(lr_model, data.validation_x, data.validation_paid,
                                    scoring="neg_log_loss", n_repeats=3, random_state=7)
    for column, mean, std in zip(data.validation_x.columns, result.importances_mean,
                                 result.importances_std):
        row = table[table["feature"] == column].iloc[0]
        assert row["log loss rise"] == pytest.approx(mean)
        assert row["std"] == pytest.approx(std)


def test_permutation_importances_a_column_the_model_ignores_rises_exactly_zero(data):
    # A fixed model that reads only `signal`: shuffling any other column changes nothing,
    # so its rise is exactly 0; shuffling `signal` raises the log loss.

    class ByHand(ClassifierMixin, BaseEstimator):
        """A fixed model: p = sigmoid(2.5 * signal - 0.8), the true model of tiny_data."""
        classes_ = np.array([False, True])

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            p = 1 / (1 + np.exp(-(2.5 * np.asarray(x["signal"]) - 0.8)))
            return np.column_stack([1 - p, p])

    x, paid = data.validation_x, data.validation_paid
    table = permutation_importances(ByHand(), x, paid, 1, 0)
    assert importance_of(table, "signal") > 0.3
    assert importance_of(table, "noise") == 0.0
    assert importance_of(table, "h") == 0.0


def test_permutation_importances_reproducible_with_the_same_seed(lr_model, data):
    one = permutation_importances(lr_model, data.validation_x, data.validation_paid, 3, 42)
    two = permutation_importances(lr_model, data.validation_x, data.validation_paid, 3, 42)
    pd.testing.assert_frame_equal(one, two)


def test_permutation_importances_do_not_change_their_inputs(lr_model, data):
    x, paid = data.validation_x.copy(), data.validation_paid.copy()
    permutation_importances(lr_model, x, paid, 2, 0)
    pd.testing.assert_frame_equal(x, data.validation_x)
    pd.testing.assert_series_equal(paid, data.validation_paid)


def test_permutation_importances_accept_nullable_boolean_paid(lr_model, data):
    plain = permutation_importances(lr_model, data.validation_x, data.validation_paid, 2, 0)
    nullable = permutation_importances(lr_model, data.validation_x,
                                       data.validation_paid.astype("boolean"), 2, 0)
    pd.testing.assert_frame_equal(plain, nullable)


def test_permutation_importances_hgb_with_a_categorical_company(hgb_company_model,
                                                                 company_data):
    x = company_data.validation_x
    table = permutation_importances(hgb_company_model, x, company_data.validation_paid, 3, 0)
    assert sorted(table["feature"]) == sorted(["h", "signal", "noise", "company_code"])
    assert table.iloc[0]["feature"] == "signal"
    assert importance_of(table, "signal") > 0.1
    assert importance_of(table, "company_code") > 0.01
    assert abs(importance_of(table, "noise")) < 0.05
    assert x["company_code"].dtype == "category"


def test_permutation_importances_lr_pipeline_with_a_categorical_company(company_data):
    model = fit_stage1_baseline(company_data.train_x, company_data.train_paid,
                                stage1_settings(baseline_c=1.0))
    table = permutation_importances(model, company_data.validation_x,
                                    company_data.validation_paid, 3, 0)
    assert sorted(table["feature"]) == sorted(["h", "signal", "noise", "company_code"])
    assert importance_of(table, "signal") > importance_of(table, "company_code") > 0.01


def test_permutation_importances_hgb_reproducible(hgb_company_model, company_data):
    one = permutation_importances(hgb_company_model, company_data.validation_x,
                                  company_data.validation_paid, 2, 5)
    two = permutation_importances(hgb_company_model, company_data.validation_x,
                                  company_data.validation_paid, 2, 5)
    pd.testing.assert_frame_equal(one, two)


def test_permutation_importances_ties_keep_column_order():
    # A model that ignores every column: every rise is 0, so the order is x's column order.
    x = pd.DataFrame({"z": np.arange(40.0), "a": np.arange(40.0)[::-1], "m": np.ones(40)})
    paid = pd.Series([True, False] * 20)

    class Constant(ClassifierMixin, BaseEstimator):
        classes_ = np.array([False, True])

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            return np.tile([0.4, 0.6], (len(x), 1))

    table = permutation_importances(Constant(), x, paid, 2, 0)
    assert list(table["feature"]) == ["z", "a", "m"]
    assert (table["log loss rise"] == 0).all()


# --- write_tables ----------------------------------------------------------------------------

def tables_to_write():
    return {
        "hgb_grid": pd.DataFrame({"candidate": [1, 0], "max_iter": [30, 10],
                                  "class_weight": ["balanced", "none"],
                                  "avg precision": [0.75, 0.5]}),
        "test": pd.DataFrame({"model": ["HGB", "LR"], "F1": [0.25, 0.125]},
                             index=pd.Index([7, 9], name="row")),
    }


def test_write_tables_writes_one_csv_per_table(tmp_path):
    paths = write_tables(tmp_path, tables_to_write())
    assert paths == [tmp_path / "hgb_grid.csv", tmp_path / "test.csv"]
    assert all(path.is_file() for path in paths)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["hgb_grid.csv", "test.csv"]


def test_write_tables_returns_path_objects(tmp_path):
    for path in write_tables(tmp_path, tables_to_write()):
        assert isinstance(path, Path)


def test_write_tables_read_back_equal_without_the_index(tmp_path):
    tables = tables_to_write()
    write_tables(tmp_path, tables)
    for name, table in tables.items():
        back = pd.read_csv(tmp_path / f"{name}.csv")
        pd.testing.assert_frame_equal(back, table.reset_index(drop=True))


def test_write_tables_writes_no_index_column(tmp_path):
    write_tables(tmp_path, tables_to_write())
    header = (tmp_path / "test.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "model,F1"


def test_write_tables_creates_missing_parent_folders(tmp_path):
    folder = tmp_path / "a" / "b" / "hurdle_diagnostic"
    paths = write_tables(folder, tables_to_write())
    assert folder.is_dir()
    assert all(path.parent == folder for path in paths)


def test_write_tables_takes_a_string_folder(tmp_path):
    paths = write_tables(str(tmp_path / "out"), tables_to_write())
    assert paths[0] == tmp_path / "out" / "hgb_grid.csv"
    assert paths[0].is_file()


def test_write_tables_into_an_existing_folder_overwrites_the_files(tmp_path):
    (tmp_path / "test.csv").write_text("old", encoding="utf-8")
    write_tables(tmp_path, tables_to_write())
    assert pd.read_csv(tmp_path / "test.csv")["model"].tolist() == ["HGB", "LR"]


def test_write_tables_with_no_tables_creates_the_folder_and_returns_nothing(tmp_path):
    folder = tmp_path / "empty"
    assert write_tables(folder, {}) == []
    assert folder.is_dir()
    assert list(folder.iterdir()) == []


def test_write_tables_paths_follow_the_dict_order(tmp_path):
    tables = {name: pd.DataFrame({"x": [1]}) for name in ("z", "a", "m")}
    assert [p.name for p in write_tables(tmp_path, tables)] == ["z.csv", "a.csv", "m.csv"]


def test_write_tables_does_not_change_the_tables(tmp_path):
    tables = tables_to_write()
    before = {name: table.copy(deep=True) for name, table in tables.items()}
    write_tables(tmp_path, tables)
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], before[name])


# =============================================================================================
# U10.4b': the Brier score in classification_scores, LOWER_IS_BETTER, and grid_search ranked
# by a chosen selection metric (log loss, brier or avg precision).
# =============================================================================================

from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

from collection_estimation.hurdle.diagnostic import LOWER_IS_BETTER  # noqa: E402
from collection_estimation.hurdle.settings import SELECTION_METRICS  # noqa: E402

METRICS = ["log loss", "brier", "avg precision"]
ALLOWED = "['log loss', 'brier', 'avg precision']"


# --- classification_scores: brier ------------------------------------------------------------

def test_scores_brier_by_hand():
    # (p - outcome)^2: .1^2, .6^2, .6^2, .1^2, .3^2 = .01 + .36 + .36 + .01 + .09 = .83; / 5.
    assert example_scores()["brier"] == pytest.approx(0.166)


def test_scores_brier_comes_right_after_log_loss_and_last():
    keys = list(example_scores())
    assert keys[-2:] == ["log loss", "brier"]


def test_scores_brier_is_zero_for_certain_right_answers():
    scores = classification_scores(series([True, False, True]), series([1.0, 0.0, 1.0]), 0.5)
    assert scores["brier"] == 0.0


def test_scores_brier_is_one_for_certain_wrong_answers_unlike_log_loss():
    # Brier is bounded by 1; log loss blows up (clipped) on the same rows.
    scores = classification_scores(series([True, False]), series([0.0, 1.0]), 0.5)
    assert scores["brier"] == 1.0
    assert scores["log loss"] > 10


def test_scores_brier_of_a_constant_half_is_a_quarter():
    scores = classification_scores(series([True, False, False, True, False]),
                                   series([0.5] * 5), 0.5)
    assert scores["brier"] == pytest.approx(0.25)


def test_scores_brier_does_not_depend_on_the_threshold():
    assert example_scores(0.05)["brier"] == example_scores(0.95)["brier"]


def test_scores_brier_is_a_float():
    assert isinstance(example_scores()["brier"], float)


@pytest.mark.parametrize("seed", range(30))
def test_scores_brier_equals_the_mean_squared_error_of_p(seed):
    rng = np.random.default_rng(7000 + seed)
    n = int(rng.integers(2, 50))
    actual = np.array([True, False] + list(rng.random(n - 2) < 0.4))
    p = rng.random(n)
    scores = classification_scores(series(actual), series(p), 0.5)
    assert scores["brier"] == pytest.approx(float(np.mean((p - actual) ** 2)), abs=1e-12)
    assert scores["brier"] == pytest.approx(brier_score_loss(actual, p), abs=1e-15)


def test_scores_brier_with_nullable_boolean_and_int_actual_is_the_same():
    plain = example_scores()["brier"]
    for actual in (series(EX_ACTUAL, dtype="boolean"), series([int(a) for a in EX_ACTUAL])):
        assert classification_scores(actual, series(EX_P), 0.5)["brier"] == plain


# --- LOWER_IS_BETTER -------------------------------------------------------------------------

def test_lower_is_better_constant():
    assert LOWER_IS_BETTER == {"log loss": True, "brier": True, "avg precision": False}
    assert list(LOWER_IS_BETTER) == METRICS


def test_lower_is_better_keys_are_scores_of_classification_scores():
    assert set(LOWER_IS_BETTER) <= set(example_scores())


def test_lower_is_better_matches_the_settings_selection_metrics():
    # The settings accept exactly the metrics grid_search can rank by, in the same order.
    assert tuple(LOWER_IS_BETTER) == SELECTION_METRICS


# --- grid_search: ranked by the selection metric ----------------------------------------------
# Four hand-made p vectors on VAL_PAID (payers at rows 0, 1, 3, 6) on which the metrics disagree:
#   SHY       perfect ranking, every p near .5        log loss .645  brier .226  avg prec 1
#   SURE      confident, one payer below a non-payer  log loss .212  brier .071  avg prec .95
#   RECKLESS  near-certain, one payer at p = .001     log loss .864  brier .125  avg prec .875
#   HALF      every p .5                              log loss .693  brier .250  avg prec .5
# In the grid order [HALF, RECKLESS, SURE, SHY] (candidates 0..3):
#   log loss, ascending:       SURE, SHY, HALF, RECKLESS   -> [2, 3, 0, 1]
#   brier, ascending:          SURE, RECKLESS, SHY, HALF   -> [2, 1, 3, 0]
#   avg precision, descending: SHY, SURE, RECKLESS, HALF   -> [3, 2, 1, 0]
P_SHY = np.array([0.54, 0.53, 0.48, 0.52, 0.47, 0.46, 0.51, 0.49])
P_SURE = np.array([0.97, 0.97, 0.03, 0.97, 0.03, 0.45, 0.40, 0.03])
P_RECKLESS = np.array([0.999, 0.999, 0.001, 0.999, 0.001, 0.001, 0.001, 0.001])
P_HALF = np.full(8, 0.5)
DISAGREE = [P_HALF, P_RECKLESS, P_SURE, P_SHY]
EXPECTED_ORDER = {"log loss": [2, 3, 0, 1], "brier": [2, 1, 3, 0],
                  "avg precision": [3, 2, 1, 0]}


def test_the_disagreeing_p_vectors_have_the_stated_scores():
    def scores(p):
        return (log_loss(VAL_PAID, p, labels=[False, True]), brier_score_loss(VAL_PAID, p),
                average_precision_score(VAL_PAID, p))
    assert scores(P_SHY) == pytest.approx((0.6446, 0.22575, 1.0), abs=1e-4)
    assert scores(P_SURE) == pytest.approx((0.2121, 0.0709875, 0.95), abs=1e-4)
    assert scores(P_RECKLESS) == pytest.approx((0.8643, 0.124751, 0.875), abs=1e-4)
    assert scores(P_HALF) == pytest.approx((0.6931, 0.25, 0.5), abs=1e-4)


@pytest.mark.parametrize("metric", METRICS)
def test_grid_search_ranks_by_the_chosen_metric(metric):
    grid, _, _, _ = fake_grid(DISAGREE, metric=metric)
    assert list(grid["candidate"]) == EXPECTED_ORDER[metric]


def test_grid_search_by_log_loss_is_ascending():
    grid, _, _, _ = fake_grid(DISAGREE, metric="log loss")
    assert grid["log loss"].is_monotonic_increasing
    assert grid.loc[0, "log loss"] == grid["log loss"].min()


def test_grid_search_by_brier_is_ascending():
    grid, _, _, _ = fake_grid(DISAGREE, metric="brier")
    assert grid["brier"].is_monotonic_increasing
    assert grid.loc[0, "brier"] == grid["brier"].min()


def test_grid_search_by_avg_precision_is_descending():
    grid, _, _, _ = fake_grid(DISAGREE, metric="avg precision")
    assert grid["avg precision"].is_monotonic_decreasing
    assert grid.loc[0, "avg precision"] == grid["avg precision"].max()


def test_grid_search_log_loss_and_avg_precision_pick_different_winners():
    # The confident but slightly mis-ranked SURE wins on log loss; the perfectly ranked but
    # timid SHY wins on average precision.
    by_log_loss, _, _, _ = fake_grid(DISAGREE, metric="log loss")
    by_ap, _, _, _ = fake_grid(DISAGREE, metric="avg precision")
    assert by_log_loss.loc[0, "candidate"] == 2
    assert by_ap.loc[0, "candidate"] == 3


def test_grid_search_log_loss_and_brier_pick_different_winners():
    # One confident mistake: log loss punishes RECKLESS without bound, Brier by at most 1/8.
    by_log_loss, _, _, _ = fake_grid([P_HALF, P_RECKLESS], metric="log loss")
    by_brier, _, _, _ = fake_grid([P_HALF, P_RECKLESS], metric="brier")
    assert list(by_log_loss["candidate"]) == [0, 1]
    assert list(by_brier["candidate"]) == [1, 0]


def test_grid_search_rows_are_the_same_whatever_the_metric():
    # The metric changes only the order of the rows, never their content.
    grids = [fake_grid(DISAGREE, metric=metric)[0] for metric in METRICS]
    tidy = [g.drop(columns="fit seconds").sort_values("candidate", ignore_index=True)
            for g in grids]
    for other in tidy[1:]:
        pd.testing.assert_frame_equal(other, tidy[0])


def test_grid_search_fits_each_candidate_once_whatever_the_metric():
    for metric in METRICS:
        _, _, fit, _ = fake_grid(DISAGREE, metric=metric)
        assert [call.settings.tag for call in fit.calls] == [0, 1, 2, 3], metric


@pytest.mark.parametrize("metric", ["log loss", "brier"])
def test_grid_search_ties_keep_grid_order_when_lower_is_better(metric):
    # SURE beats SHY on both: candidates 1 and 3 tie on top, 0 and 2 below, in grid order.
    grid, _, _, _ = fake_grid([P_SHY, P_SURE, P_SHY, P_SURE], metric=metric)
    assert list(grid["candidate"]) == [1, 3, 0, 2]


@pytest.mark.parametrize("metric", METRICS)
def test_grid_search_all_tied_keeps_grid_order_for_every_metric(metric):
    grid, _, _, _ = fake_grid([P_MID] * 6, metric=metric)
    assert list(grid["candidate"]) == list(range(6))


@pytest.mark.parametrize("metric", ["log loss", "brier"])
def test_grid_search_many_interleaved_ties_stay_stable_when_lower_is_better(metric):
    # Ascending order with many ties: a non-stable sort could swap equal rows.
    vectors = [P_SURE, P_SHY, P_HALF]   # better to worse on both log loss and brier
    assignment = [vectors[i % 3] for i in range(30)]
    grid, _, _, _ = fake_grid(assignment, metric=metric)
    expected = [i for k in range(3) for i in range(30) if i % 3 == k]
    assert list(grid["candidate"]) == expected


@pytest.mark.parametrize("metric", METRICS)
def test_grid_search_progress_line_names_the_metric_and_its_score(metric):
    _, out, _, candidates = fake_grid(DISAGREE, name="LR", metric=metric)
    lines = out.splitlines()
    assert len(lines) == 4
    for number, (line, candidate, p) in enumerate(zip(lines, candidates, DISAGREE), start=1):
        score = classification_scores(VAL_PAID, pd.Series(p, index=VAL_X.index), 0.5)[metric]
        prefix = f"  ... LR {number} of 4: {candidate.params}, {metric} {score:.4f} ("
        assert line.startswith(prefix), line
        assert line.endswith(" s)"), line


@pytest.mark.parametrize("metric", ["log loss", "brier"])
def test_grid_search_progress_line_does_not_show_avg_precision_for_other_metrics(metric):
    _, out, _, _ = fake_grid(DISAGREE, metric=metric)
    assert "avg precision" not in out


# --- grid_search: the metric argument --------------------------------------------------------

@pytest.mark.parametrize("bad", ["AUC", "F1", "threshold", "Log loss", "log_loss", "logloss",
                                 " log loss", "Brier", "average precision", "", None, 0])
def test_grid_search_unknown_metric_raises_the_exact_message(bad):
    fit = SpyFit(lambda s, w, x: P_MID)
    with pytest.raises(ValueError) as error:
        run_quietly(grid_search, "HGB", fit, fake_candidates(2), TRAIN_X, TRAIN_PAID, VAL_X,
                    VAL_PAID, metric=bad)
    assert str(error.value) == f"HGB: cannot rank by {bad!r}; choose one of {ALLOWED}"
    assert fit.calls == []


def test_grid_search_unknown_metric_message_starts_with_the_model_name():
    with pytest.raises(ValueError) as error:
        run_quietly(grid_search, "LR", SpyFit(lambda s, w, x: P_MID), fake_candidates(1),
                    TRAIN_X, TRAIN_PAID, VAL_X, VAL_PAID, metric="AUC")
    assert str(error.value).startswith("LR: ")


def test_grid_search_unknown_metric_prints_nothing():
    fit = SpyFit(lambda s, w, x: P_MID)
    with redirect_stdout(io.StringIO()) as out:
        with pytest.raises(ValueError):
            grid_search("M", fit, fake_candidates(3), TRAIN_X, TRAIN_PAID, VAL_X, VAL_PAID,
                        metric="AUC")
    assert out.getvalue() == ""


def test_grid_search_unknown_metric_is_reported_before_an_empty_grid():
    fit = SpyFit(lambda s, w, x: P_MID)
    with pytest.raises(ValueError) as error:
        grid_search("LR", fit, [], TRAIN_X, TRAIN_PAID, VAL_X, VAL_PAID, metric="AUC")
    assert str(error.value) == f"LR: cannot rank by 'AUC'; choose one of {ALLOWED}"


@pytest.mark.parametrize("metric", METRICS)
def test_grid_search_empty_grid_with_a_good_metric_has_the_empty_message(metric):
    with pytest.raises(ValueError) as error:
        grid_search("LR", SpyFit(lambda s, w, x: P_MID), [], TRAIN_X, TRAIN_PAID, VAL_X,
                    VAL_PAID, metric=metric)
    assert str(error.value) == "LR: the grid has no settings to try"


def test_grid_search_metric_is_required():
    with pytest.raises(TypeError):
        run_quietly(grid_search, "M", SpyFit(lambda s, w, x: P_MID), fake_candidates(1),
                    TRAIN_X, TRAIN_PAID, VAL_X, VAL_PAID)


def test_grid_search_metric_is_keyword_only():
    with pytest.raises(TypeError):
        run_quietly(grid_search, "M", SpyFit(lambda s, w, x: P_MID), fake_candidates(1),
                    TRAIN_X, TRAIN_PAID, VAL_X, VAL_PAID, "log loss")


# --- choose_and_test follows the metric that sorted the grid ---------------------------------

@pytest.mark.parametrize("metric", METRICS)
def test_choose_and_test_refits_the_winner_of_the_metric(metric):
    grid, _, _, candidates = fake_grid(DISAGREE, metric=metric)
    fit = SpyFit(lambda s, w, x: P_TEST)
    result = choose_and_test("M", fit, candidates, grid, REFIT_X, REFIT_PAID, TEST_X, TEST_PAID)
    winner = EXPECTED_ORDER[metric][0]
    assert len(fit.calls) == 1
    assert fit.calls[0].settings is candidates[winner].settings
    assert fit.calls[0].train_x is REFIT_X
    assert result["k"] == winner
    assert result["threshold"] == grid.loc[0, "threshold"]


def test_choose_and_test_after_a_log_loss_grid_is_not_the_avg_precision_winner():
    grid, _, _, candidates = fake_grid(DISAGREE, metric="log loss")
    fit = SpyFit(lambda s, w, x: P_TEST)
    choose_and_test("M", fit, candidates, grid, REFIT_X, REFIT_PAID, TEST_X, TEST_PAID)
    assert fit.calls[0].settings is candidates[2].settings          # SURE, the log-loss best
    assert fit.calls[0].settings is not candidates[3].settings      # SHY, the AP best


def test_choose_and_test_uses_the_log_loss_winners_own_validation_threshold():
    grid, _, _, candidates = fake_grid(DISAGREE, metric="log loss")
    sure_threshold, _ = best_threshold(VAL_PAID, pd.Series(P_SURE, index=VAL_X.index))
    result = choose_and_test("M", SpyFit(lambda s, w, x: P_TEST), candidates, grid, REFIT_X,
                             REFIT_PAID, TEST_X, TEST_PAID)
    assert result["threshold"] == sure_threshold
    expected = classification_scores(TEST_PAID, pd.Series(P_TEST, index=TEST_X.index),
                                     sure_threshold)
    for key in SCORE_KEYS:
        assert result[key] == pytest.approx(expected[key]), key


def test_choose_and_test_result_carries_the_brier_score():
    grid, _, _, candidates = fake_grid([P_SURE], metric="brier")
    result = choose_and_test("M", SpyFit(lambda s, w, x: P_TEST), candidates, grid, REFIT_X,
                             REFIT_PAID, TEST_X, TEST_PAID)
    assert list(result)[-2:] == ["log loss", "brier"]


# --- grid_search: real fits by each metric ---------------------------------------------------

@pytest.mark.parametrize("metric", METRICS)
def test_grid_search_real_lr_grid_is_sorted_by_the_metric(metric, data):
    candidates = lr_candidates(stage1_settings(), SMALL_LR)
    grid, out = run_quietly(grid_search, "LR", fit_stage1_baseline, candidates, data.train_x,
                            data.train_paid, data.validation_x, data.validation_paid,
                            metric=metric)
    if LOWER_IS_BETTER[metric]:
        assert grid[metric].is_monotonic_increasing
    else:
        assert grid[metric].is_monotonic_decreasing
    assert sorted(grid["candidate"]) == list(range(len(candidates)))
    assert all(f", {metric} " in line for line in out.splitlines())


def test_grid_search_real_hgb_brier_column_matches_an_independent_fit(data):
    candidates = hgb_candidates(stage1_settings(), SMALL_HGB)
    grid, _ = run_quietly(grid_search, "HGB", fit_stage1, candidates, data.train_x,
                          data.train_paid, data.validation_x, data.validation_paid,
                          metric="brier")
    assert grid["brier"].is_monotonic_increasing
    for _, row in grid.iterrows():
        candidate = candidates[int(row["candidate"])]
        model = fit_stage1(data.train_x, data.train_paid, candidate.settings,
                           class_weight=class_weight_argument(candidate.class_weight))
        p = predict_paid(model, data.validation_x)
        assert row["brier"] == pytest.approx(brier_score_loss(data.validation_paid, p),
                                             rel=1e-9)


# =============================================================================================
# U10.4d: calibration of stage 1. check_outcomes, reliability_table, calibration_slope_intercept,
# clipped_logit, calibration_summary (U10.4d.1); fit_platt, apply_platt (U10.4d.2);
# reference_probabilities (U10.4d.3); best_unweighted, calibration_comparison (U10.4d.4).
#
# The hand-worked example HAND (6 rows, 3 groups of 2):
#   row     0    1    2    3    4    5
#   p      .1   .4   .2   .3   .8   .6
#   paid    F    T    F    T    T    F
# Sorted by p: rows 0, 2 | 3, 1 | 5, 4 -> groups 1, 2, 3 with
#   mean p .15, .35, .70; share paid 0, 1, .5; weights 1/3 each; overall share .5.
#   ECE = (.15 + .65 + .20) / 3 = 1/3
#   calibration error = (.0225 + .4225 + .04) / 3 = .485 / 3 = 97/600
#   resolution = (.25 + .25 + 0) / 3 = 1/6 = 100/600; uncertainty = .25 = 150/600
#   brier = (.01 + .36 + .04 + .49 + .04 + .36) / 6 = 1.3 / 6 = 130/600
#   decomposition gap = 130/600 - (97 - 100 + 150)/600 = -17/600
#   AUC: payers .4, .3, .8 against non-payers .1, .2, .6: 2 + 2 + 3 = 7 of 9 pairs
#   against a constant .5 reference: brier skill = 1 - (130/600)/.25 = 2/15
# =============================================================================================

from sklearn.metrics import roc_auc_score  # noqa: E402

from collection_estimation.hurdle.diagnostic import (  # noqa: E402
    LOGIT_CLIP,
    apply_platt,
    best_unweighted,
    calibration_comparison,
    calibration_slope_intercept,
    calibration_summary,
    check_outcomes,
    clipped_logit,
    fit_platt,
    reference_probabilities,
    reliability_table,
)

HAND_P = pd.Series([0.1, 0.4, 0.2, 0.3, 0.8, 0.6])
HAND_PAID = pd.Series([False, True, False, True, True, False])
HALF = pd.Series(0.5, index=HAND_P.index)

SUMMARY_KEYS = ["mean p", "share paid", "ECE", "slope", "intercept", "log loss", "brier",
                "calibration error", "resolution", "uncertainty", "decomposition gap",
                "log loss skill", "brier skill", "AUC"]
RELIABILITY_COLUMNS = ["group", "p_low", "p_high", "mean_p", "share_paid", "rows"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def simulated(n, seed, scale=1.5):
    """True logits z, the true p = sigmoid(z) and outcomes drawn from it."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=scale, size=n)
    paid = pd.Series(rng.random(n) < sigmoid(z))
    return z, paid


# --- LOGIT_CLIP and clipped_logit -------------------------------------------------------------

def test_logit_clip_constant():
    assert LOGIT_CLIP == 1e-6


def test_clipped_logit_values_by_hand():
    p = pd.Series([0.5, 0.75, 0.25, 0.9], index=list("abcd"))
    logit = clipped_logit(p)
    assert list(logit.index) == list("abcd")
    assert logit.tolist() == pytest.approx([0.0, math.log(3), -math.log(3), math.log(9)],
                                           abs=1e-12)


def test_clipped_logit_clips_the_edges_and_beyond():
    p = pd.Series([0.0, 1.0, -0.5, 1.5, 1e-9, 1 - 1e-9])
    edge = math.log(LOGIT_CLIP / (1 - LOGIT_CLIP))
    logit = clipped_logit(p)
    assert np.isfinite(logit).all()
    assert logit.tolist() == pytest.approx([edge, -edge, edge, -edge, edge, -edge], rel=1e-9)


def test_clipped_logit_is_antisymmetric_and_leaves_p_unchanged():
    p = pd.Series([0.01, 0.3, 0.7, 0.99])
    before = p.copy()
    logit = clipped_logit(p)
    assert logit.iloc[0] == pytest.approx(-logit.iloc[3]) and logit.iloc[1] == pytest.approx(
        -logit.iloc[2])
    pd.testing.assert_series_equal(p, before)


# --- check_outcomes --------------------------------------------------------------------------

def test_check_outcomes_returns_actual_as_bool():
    actual = pd.Series([1, 0, 1], index=[5, 6, 7])
    result = check_outcomes(actual, pd.Series([0.2, 0.3, 0.4], index=[5, 6, 7]))
    assert result.dtype == bool
    assert result.tolist() == [True, False, True]
    assert list(result.index) == [5, 6, 7]


def test_check_outcomes_accepts_the_nullable_boolean_dtype():
    actual = pd.Series(pd.array([True, False], dtype="boolean"))
    assert check_outcomes(actual, pd.Series([0.1, 0.2])).dtype == bool


@pytest.mark.parametrize("p_index", [[0, 1, 2, 4], [1, 0, 2, 3], [0, 1, 2], ["0", "1", "2", "3"]],
                         ids=["other label", "reordered", "shorter", "strings"])
def test_check_outcomes_refuses_a_different_index(p_index):
    actual = pd.Series([True, False, True, False])
    with pytest.raises(ValueError) as error:
        check_outcomes(actual, pd.Series(0.5, index=p_index))
    assert str(error.value) == "actual and p must have the same rows (the same index)"


def test_check_outcomes_refuses_no_rows():
    with pytest.raises(ValueError) as error:
        check_outcomes(pd.Series([], dtype=bool), pd.Series([], dtype=float))
    assert str(error.value) == "there are no rows to measure"


@pytest.mark.parametrize("values", [[True, True, True], [False, False], [True]])
def test_check_outcomes_refuses_one_outcome(values):
    with pytest.raises(ValueError) as error:
        check_outcomes(pd.Series(values), pd.Series(0.5, index=range(len(values))))
    assert str(error.value) == "actual must hold both outcomes (payers and non-payers)"


def test_check_outcomes_does_not_change_actual():
    actual = pd.Series([1, 0])
    check_outcomes(actual, pd.Series([0.5, 0.5]))
    assert actual.dtype == np.int64 and actual.tolist() == [1, 0]


# --- reliability_table -----------------------------------------------------------------------

def test_reliability_table_by_hand():
    table = reliability_table(HAND_PAID, HAND_P, 3)
    assert list(table.columns) == RELIABILITY_COLUMNS
    assert table["group"].tolist() == [1, 2, 3]
    assert table["p_low"].tolist() == pytest.approx([0.1, 0.3, 0.6])
    assert table["p_high"].tolist() == pytest.approx([0.2, 0.4, 0.8])
    assert table["mean_p"].tolist() == pytest.approx([0.15, 0.35, 0.7])
    assert table["share_paid"].tolist() == pytest.approx([0.0, 1.0, 0.5])
    assert table["rows"].tolist() == [2, 2, 2]


def test_reliability_table_does_not_depend_on_row_order():
    shuffled = [4, 0, 5, 2, 1, 3]
    a = reliability_table(HAND_PAID, HAND_P, 3)
    b = reliability_table(HAND_PAID[shuffled], HAND_P[shuffled], 3)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("n, groups, sizes", [(7, 3, [3, 2, 2]), (10, 4, [3, 2, 3, 2]),
                                              (6, 6, [1] * 6), (5, 2, [3, 2]), (8, 2, [4, 4])])
def test_reliability_table_group_sizes_follow_position_times_groups_over_n(n, groups, sizes):
    # group = position * groups // n + 1, worked out by hand for each case.
    p = pd.Series(np.linspace(0.05, 0.95, n))
    actual = pd.Series([i % 2 == 0 for i in range(n)])
    table = reliability_table(actual, p, groups)
    assert table["rows"].tolist() == sizes
    assert table["group"].tolist() == list(range(1, groups + 1))
    assert int(table["rows"].sum()) == n


def test_reliability_table_ties_keep_row_order():
    # Every p ties: the stable sort keeps rows 0, 1 in group 1 and rows 2, 3 in group 2.
    table = reliability_table(pd.Series([True, True, False, False]), pd.Series([0.5] * 4), 2)
    assert table["share_paid"].tolist() == [1.0, 0.0]
    assert table["mean_p"].tolist() == [0.5, 0.5]


def test_reliability_table_ties_keep_row_order_with_a_shuffled_index():
    # Row order, not index order: index 9 comes first.
    actual = pd.Series([False, True, True, False], index=[9, 3, 7, 1])
    table = reliability_table(actual, pd.Series(0.3, index=actual.index), 2)
    assert table["share_paid"].tolist() == [0.5, 0.5]
    actual = pd.Series([False, False, True, True], index=[9, 3, 7, 1])
    table = reliability_table(actual, pd.Series(0.3, index=actual.index), 2)
    assert table["share_paid"].tolist() == [0.0, 1.0]


def test_reliability_table_groups_ordered_by_p():
    rng = np.random.default_rng(4)
    p = pd.Series(rng.random(200))
    actual = pd.Series(rng.random(200) < p)
    table = reliability_table(actual, p, 10)
    assert table["mean_p"].is_monotonic_increasing
    assert (table["p_high"].iloc[:-1].to_numpy() <= table["p_low"].iloc[1:].to_numpy()).all()
    assert (table["p_low"] <= table["mean_p"]).all() and (table["mean_p"] <= table["p_high"]).all()
    assert table["rows"].tolist() == [20] * 10


def test_reliability_table_does_not_change_its_inputs():
    p, actual = HAND_P.copy(), HAND_PAID.copy()
    reliability_table(actual, p, 3)
    pd.testing.assert_series_equal(p, HAND_P)
    pd.testing.assert_series_equal(actual, HAND_PAID)


@pytest.mark.parametrize("groups", [1, 0, -2])
def test_reliability_table_refuses_fewer_than_two_groups(groups):
    with pytest.raises(ValueError) as error:
        reliability_table(HAND_PAID, HAND_P, groups)
    assert str(error.value) == f"groups is {groups}; at least 2 are needed"


@pytest.mark.parametrize("groups", [7, 100])
def test_reliability_table_refuses_more_groups_than_rows(groups):
    with pytest.raises(ValueError) as error:
        reliability_table(HAND_PAID, HAND_P, groups)
    assert str(error.value) == f"6 rows cannot be cut into {groups} groups"


def test_reliability_table_checks_the_outcomes_first():
    with pytest.raises(ValueError, match="both outcomes"):
        reliability_table(pd.Series([True] * 6), HAND_P, 1)
    with pytest.raises(ValueError, match="same rows"):
        reliability_table(HAND_PAID, HAND_P.reset_index(drop=True).set_axis(range(1, 7)), 3)


# --- calibration_slope_intercept -------------------------------------------------------------

def test_slope_intercept_are_python_floats():
    slope, intercept = calibration_slope_intercept(HAND_PAID, HAND_P)
    assert type(slope) is float and type(intercept) is float


def test_slope_intercept_solve_the_unpenalised_score_equations():
    # The unpenalised maximum-likelihood fit has zero gradient: sum(y - s) = 0 and
    # sum(x (y - s)) = 0, with x = logit(p) and s = sigmoid(a x + b). A penalty would break this.
    slope, intercept = calibration_slope_intercept(HAND_PAID, HAND_P)
    x = np.log(HAND_P / (1 - HAND_P)).to_numpy()
    residual = HAND_PAID.to_numpy().astype(float) - sigmoid(slope * x + intercept)
    # Up to the solver's tolerance; a penalty of C = 1 would leave a gradient of about slope.
    assert abs(residual.sum()) < 1e-3
    assert abs((x * residual).sum()) < 1e-3


def test_slope_intercept_is_unpenalised_on_a_larger_sample():
    z, paid = simulated(3000, seed=21)
    p = pd.Series(sigmoid(0.6 * z + 0.3))
    slope, intercept = calibration_slope_intercept(paid, p)
    x = np.log(p / (1 - p)).to_numpy()
    residual = paid.to_numpy().astype(float) - sigmoid(slope * x + intercept)
    assert abs(residual.mean()) < 1e-4 and abs((x * residual).mean()) < 1e-4


def test_slope_intercept_of_calibrated_p_is_about_one_and_zero():
    z, paid = simulated(40000, seed=1)
    slope, intercept = calibration_slope_intercept(paid, pd.Series(sigmoid(z)))
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_slope_of_over_confident_p_is_below_one():
    # p = sigmoid(2 z): too extreme, the slope is about 1/2.
    z, paid = simulated(40000, seed=2)
    slope, intercept = calibration_slope_intercept(paid, pd.Series(sigmoid(2 * z)))
    assert slope == pytest.approx(0.5, abs=0.04)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_slope_of_under_confident_p_is_above_one():
    # p = sigmoid(z / 2): too timid, the slope is about 2.
    z, paid = simulated(40000, seed=3)
    slope, intercept = calibration_slope_intercept(paid, pd.Series(sigmoid(0.5 * z)))
    assert slope == pytest.approx(2.0, abs=0.12)
    assert intercept == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("shift", [-1.0, 0.7])
def test_intercept_of_shifted_p_undoes_the_shift(shift):
    # p = sigmoid(z + shift): every p moved the same way; slope 1, intercept -shift.
    z, paid = simulated(40000, seed=4)
    slope, intercept = calibration_slope_intercept(paid, pd.Series(sigmoid(z + shift)))
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(-shift, abs=0.06)


def test_slope_intercept_clip_p_of_zero_and_one():
    # p of exactly 0 and 1 would give an infinite logit; the clip keeps the fit finite.
    actual = pd.Series([False, True, False, True, True, False])
    p = pd.Series([0.0, 1.0, 0.3, 0.6, 0.5, 0.4])
    slope, intercept = calibration_slope_intercept(actual, p)
    assert math.isfinite(slope) and math.isfinite(intercept)


@pytest.mark.parametrize("actual, p, match", [
    (pd.Series([True, True]), pd.Series([0.5, 0.6]), "both outcomes"),
    (pd.Series([True, False]), pd.Series([0.5, 0.6], index=[1, 2]), "same rows"),
    (pd.Series([], dtype=bool), pd.Series([], dtype=float), "no rows"),
])
def test_slope_intercept_refusals(actual, p, match):
    with pytest.raises(ValueError, match=match):
        calibration_slope_intercept(actual, p)


# --- calibration_summary ---------------------------------------------------------------------

def test_calibration_summary_keys_in_order():
    assert list(calibration_summary(HAND_PAID, HAND_P, HALF, 3)) == SUMMARY_KEYS


def test_calibration_summary_by_hand():
    summary = calibration_summary(HAND_PAID, HAND_P, HALF, 3)
    log_loss_by_hand = -(math.log(0.9) + math.log(0.4) + math.log(0.8) + math.log(0.3)
                         + math.log(0.8) + math.log(0.4)) / 6
    expected = {
        "mean p": 0.4, "share paid": 0.5, "ECE": 1 / 3, "log loss": log_loss_by_hand,
        "brier": 130 / 600, "calibration error": 97 / 600, "resolution": 100 / 600,
        "uncertainty": 0.25, "decomposition gap": -17 / 600,
        "log loss skill": 1 - log_loss_by_hand / math.log(2), "brier skill": 2 / 15,
        "AUC": 7 / 9,
    }
    for key, value in expected.items():
        assert summary[key] == pytest.approx(value, abs=1e-12), key


def test_calibration_summary_slope_and_intercept_are_the_regressions():
    summary = calibration_summary(HAND_PAID, HAND_P, HALF, 3)
    slope, intercept = calibration_slope_intercept(HAND_PAID, HAND_P)
    assert summary["slope"] == slope and summary["intercept"] == intercept


def test_calibration_summary_values_are_python_floats():
    summary = calibration_summary(HAND_PAID, HAND_P, HALF, 3)
    for key, value in summary.items():
        assert isinstance(value, float), (key, type(value))


def test_calibration_summary_a_second_hand_example_with_other_groups():
    # p .1 .2 .3 .4 (paid F F T F): 2 groups. Group 1 mean .15 share 0; group 2 mean .35
    # share .5; overall .25. ECE = (.15 + .15)/2 = .15; calibration error = (.0225 +
    # .0225)/2 = .0225; resolution = (.0625 + .0625)/2 = .0625; uncertainty = .1875;
    # brier = (.01 + .04 + .49 + .16)/4 = .175; gap = .175 - (.0225 - .0625 + .1875) = .0275.
    actual = pd.Series([False, False, True, False])
    p = pd.Series([0.1, 0.2, 0.3, 0.4])
    summary = calibration_summary(actual, p, pd.Series(0.25, index=p.index), 2)
    for key, value in {"ECE": 0.15, "calibration error": 0.0225, "resolution": 0.0625,
                       "uncertainty": 0.1875, "brier": 0.175, "decomposition gap": 0.0275,
                       "mean p": 0.25, "share paid": 0.25,
                       "brier skill": 1 - 0.175 / 0.1875, "AUC": 2 / 3}.items():
        assert summary[key] == pytest.approx(value, abs=1e-12), key


def test_decomposition_gap_is_zero_when_p_is_constant_in_each_group():
    # Grouping loses nothing when every group holds one p: brier = CE - RES + UNC exactly.
    actual = pd.Series([False, True, True, False, True, True])
    p = pd.Series([0.2, 0.2, 0.5, 0.5, 0.9, 0.9])
    summary = calibration_summary(actual, p, pd.Series(0.5, index=p.index), 3)
    assert summary["decomposition gap"] == pytest.approx(0.0, abs=1e-12)
    assert summary["brier"] == pytest.approx(
        summary["calibration error"] - summary["resolution"] + summary["uncertainty"],
        abs=1e-12)


def test_calibration_summary_of_a_perfectly_grouped_p():
    # Each group's mean p equals its share paid: ECE and calibration error 0, no resolution.
    actual = pd.Series([True, False, True, False])
    p = pd.Series([0.5] * 4)
    summary = calibration_summary(actual, p, p, 2)
    assert summary["ECE"] == 0 and summary["calibration error"] == 0
    assert summary["resolution"] == 0 and summary["uncertainty"] == 0.25
    assert summary["brier"] == 0.25 and summary["decomposition gap"] == pytest.approx(0)
    assert summary["AUC"] == 0.5


def test_calibration_summary_skill_against_itself_is_zero():
    summary = calibration_summary(HAND_PAID, HAND_P, HAND_P, 3)
    assert summary["log loss skill"] == pytest.approx(0.0, abs=1e-12)
    assert summary["brier skill"] == pytest.approx(0.0, abs=1e-12)


def test_calibration_summary_skill_against_a_better_reference_is_negative():
    better = pd.Series([0.05, 0.9, 0.1, 0.8, 0.9, 0.2])
    summary = calibration_summary(HAND_PAID, HAND_P, better, 3)
    assert summary["log loss skill"] < 0 and summary["brier skill"] < 0


def test_calibration_summary_matches_sklearn_scores_on_random_rows():
    rng = np.random.default_rng(8)
    p = pd.Series(rng.uniform(0.02, 0.98, 500), index=pd.RangeIndex(100, 600))
    actual = pd.Series(rng.random(500) < p)
    reference = pd.Series(actual.mean(), index=p.index)
    summary = calibration_summary(actual, p, reference, 10)
    table = reliability_table(actual, p, 10)
    weights = table["rows"] / 500
    assert summary["log loss"] == pytest.approx(log_loss(actual, p))
    assert summary["brier"] == pytest.approx(brier_score_loss(actual, p))
    assert summary["AUC"] == pytest.approx(roc_auc_score(actual, p))
    assert summary["ECE"] == pytest.approx(
        float((weights * (table["mean_p"] - table["share_paid"]).abs()).sum()))
    assert summary["log loss skill"] == pytest.approx(
        1 - log_loss(actual, p) / log_loss(actual, reference))
    assert summary["brier skill"] == pytest.approx(
        1 - brier_score_loss(actual, p) / brier_score_loss(actual, reference))
    assert 0 <= summary["ECE"] <= 1 and summary["calibration error"] >= 0
    assert summary["resolution"] >= 0


def test_calibration_summary_uses_the_groups_argument():
    rng = np.random.default_rng(9)
    p = pd.Series(rng.random(120))
    actual = pd.Series(rng.random(120) < p)
    few = calibration_summary(actual, p, pd.Series(0.5, index=p.index), 2)
    many = calibration_summary(actual, p, pd.Series(0.5, index=p.index), 12)
    assert few["ECE"] != many["ECE"]
    assert few["brier"] == many["brier"] and few["log loss"] == many["log loss"]


def test_calibration_summary_refuses_a_misaligned_reference():
    with pytest.raises(ValueError) as error:
        calibration_summary(HAND_PAID, HAND_P, HALF.set_axis(range(1, 7)), 3)
    assert str(error.value) == "reference_p and p must have the same rows (the same index)"


def test_calibration_summary_refuses_a_reordered_reference():
    with pytest.raises(ValueError, match="reference_p and p must have the same rows"):
        calibration_summary(HAND_PAID, HAND_P, HALF[[1, 0, 2, 3, 4, 5]], 3)


@pytest.mark.parametrize("actual, p, groups, match", [
    (pd.Series([True] * 6), HAND_P, 3, "both outcomes"),
    (HAND_PAID, HAND_P, 1, "at least 2"),
    (HAND_PAID, HAND_P, 7, "cannot be cut"),
])
def test_calibration_summary_passes_on_the_refusals(actual, p, groups, match):
    with pytest.raises(ValueError, match=match):
        calibration_summary(actual, p, pd.Series(0.5, index=p.index), groups)


# --- fit_platt and apply_platt ---------------------------------------------------------------

def test_fit_platt_is_calibration_slope_intercept():
    z, paid = simulated(2000, seed=11)
    p = pd.Series(sigmoid(1.7 * z - 0.4))
    assert fit_platt(paid, p) == calibration_slope_intercept(paid, p)
    assert fit_platt(HAND_PAID, HAND_P) == calibration_slope_intercept(HAND_PAID, HAND_P)


def test_fit_platt_has_the_same_refusals():
    with pytest.raises(ValueError, match="both outcomes"):
        fit_platt(pd.Series([False, False]), pd.Series([0.1, 0.2]))
    with pytest.raises(ValueError, match="same rows"):
        fit_platt(HAND_PAID, HAND_P.set_axis(range(10, 16)))


def test_apply_platt_by_hand():
    # a = 2, b = 1: logit(.5) = 0 -> sigmoid(1); logit(.75) = log 3 -> sigmoid(2 log 3 + 1).
    p = pd.Series([0.5, 0.75, 0.25], index=["x", "y", "z"])
    corrected = apply_platt(p, 2.0, 1.0)
    expected = [sigmoid(1.0), sigmoid(2 * math.log(3) + 1), sigmoid(-2 * math.log(3) + 1)]
    assert corrected.tolist() == pytest.approx(expected, abs=1e-12)
    assert list(corrected.index) == ["x", "y", "z"]


def test_apply_platt_identity_leaves_p_unchanged():
    p = pd.Series(np.linspace(0.001, 0.999, 50), index=pd.RangeIndex(7, 57))
    pd.testing.assert_series_equal(apply_platt(p, 1.0, 0.0), p, check_exact=False, rtol=1e-9)


def test_apply_platt_accepts_ints_and_numpy_floats():
    p = pd.Series([0.2, 0.6])
    expected = apply_platt(p, 1.0, 0.0)
    pd.testing.assert_series_equal(apply_platt(p, 1, 0), expected)
    pd.testing.assert_series_equal(apply_platt(p, np.float64(1.0), np.float64(0.0)), expected)


def test_apply_platt_does_not_change_p():
    p = pd.Series([0.2, 0.6])
    apply_platt(p, 3.0, -1.0)
    assert p.tolist() == [0.2, 0.6]


def test_apply_platt_p_of_zero_and_one_stays_strictly_inside():
    p = pd.Series([0.0, 1.0, -0.1, 1.1, 0.5])
    corrected = apply_platt(p, 1.0, 0.0)
    assert ((corrected > 0) & (corrected < 1)).all()
    assert corrected.tolist()[:4] == pytest.approx(
        [LOGIT_CLIP, 1 - LOGIT_CLIP, LOGIT_CLIP, 1 - LOGIT_CLIP], rel=1e-6)


@pytest.mark.parametrize("a, b", [(5.0, 0.0), (1.0, 40.0), (1.0, -800.0), (3.0, 0.0)],
                         ids=["steep", "shifted up", "shifted far down", "a=3"])
def test_apply_platt_stays_strictly_inside_for_large_corrections(a, b):
    # The contract: every value strictly between 0 and 1, whatever finite a and b.
    p = pd.Series([0.0, 1e-6, 0.5, 1 - 1e-6, 1.0])
    corrected = apply_platt(p, a, b)
    assert ((corrected > 0) & (corrected < 1)).all(), corrected.tolist()


@pytest.mark.parametrize("bad", [True, False, "1", None, [1.0], float("nan"), float("inf"),
                                 float("-inf"), np.nan, complex(1, 0)])
@pytest.mark.parametrize("which", ["a", "b"])
def test_apply_platt_refuses_a_non_finite_or_non_number(bad, which):
    args = {"a": 1.0, "b": 0.0}
    args[which] = bad
    with pytest.raises(ValueError) as error:
        apply_platt(pd.Series([0.2, 0.6]), args["a"], args["b"])
    assert str(error.value) == f"Platt's {which} is {bad!r}, not a finite number"


def test_platt_recovers_a_known_distortion_out_of_sample():
    # The model's p = sigmoid(2 z + .5) (too extreme, shifted); the truth is sigmoid(z). So
    # logit(p) = 2 z + .5 and z = .5 logit(p) - .25: Platt should find a ~ .5, b ~ -.25 on
    # one sample and correct a second, independent one back to the truth.
    z_fit, paid_fit = simulated(40000, seed=31)
    a, b = fit_platt(paid_fit, pd.Series(sigmoid(2 * z_fit + 0.5)))
    assert a == pytest.approx(0.5, abs=0.03)
    assert b == pytest.approx(-0.25, abs=0.04)

    z_new, paid_new = simulated(20000, seed=32)
    distorted = pd.Series(sigmoid(2 * z_new + 0.5))
    corrected = apply_platt(distorted, a, b)
    truth = sigmoid(z_new)
    assert np.abs(corrected.to_numpy() - truth).max() < 0.03
    assert log_loss(paid_new, corrected) < log_loss(paid_new, distorted)
    slope, intercept = calibration_slope_intercept(paid_new, corrected)
    assert slope == pytest.approx(1.0, abs=0.06) and intercept == pytest.approx(0.0, abs=0.06)


def test_platt_corrected_p_is_exactly_calibrated_in_sample():
    # Refitting on the corrected p of the same rows gives slope 1 and intercept 0 (the
    # correction is itself a logistic regression on logit(p)).
    z, paid = simulated(5000, seed=33)
    p = pd.Series(sigmoid(1.6 * z - 0.7))
    a, b = fit_platt(paid, p)
    slope, intercept = calibration_slope_intercept(paid, apply_platt(p, a, b))
    assert slope == pytest.approx(1.0, abs=1e-3) and intercept == pytest.approx(0.0, abs=1e-3)


# --- reference_probabilities -----------------------------------------------------------------
# Train: h 1 paid T F F F (1/4), h 2 paid T T T F F F (1/2) -> 4 of 10, base rate .4.
# Rows a..d: h 2, 1, 2, 1; window 13 history (paying, observed) (0, 0), (3, 4), (13, 13),
# (2, 10). Own rate with prior 1: .4/1 = .4, 3.4/5 = .68, 13.4/14, 2.4/11.
# With prior 2: .8/2 = .4, 3.8/6, 13.8/15 = .92, 2.8/12.

def ref_train(paid=None, h=None):
    paid = [True, False, False, False, True, True, True, False, False, False] if paid is None \
        else paid
    h = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2] if h is None else h
    return pd.DataFrame({"h": np.array(h, dtype="int64"), "paid": paid,
                         "paying_weeks_13": 99, "observed_weeks_13": 99},
                        index=pd.RangeIndex(500, 500 + len(h)))


def ref_rows(**changes):
    columns = {"h": np.array([2, 1, 2, 1], dtype="int64"),
               "paying_weeks_4": [0, 1, 4, 0], "observed_weeks_4": [0, 4, 4, 3],
               "paying_weeks_13": [0, 3, 13, 2], "observed_weeks_13": [0, 4, 13, 10],
               "other": ["w", "x", "y", "z"]}
    columns.update(changes)
    return pd.DataFrame(columns, index=list("abcd"))


def test_reference_probabilities_columns_index_and_dtypes():
    refs = reference_probabilities(ref_train(), ref_rows(), 13, 1.0)
    assert list(refs.columns) == ["base rate", "base rate per h", "own rate"]
    assert list(refs.index) == list("abcd")
    assert all(dtype == np.float64 for dtype in refs.dtypes)


def test_reference_probabilities_by_hand_prior_one():
    refs = reference_probabilities(ref_train(), ref_rows(), 13, 1.0)
    assert refs["base rate"].tolist() == pytest.approx([0.4] * 4, abs=1e-15)
    assert refs["base rate per h"].tolist() == pytest.approx([0.5, 0.25, 0.5, 0.25], abs=1e-15)
    assert refs["own rate"].tolist() == pytest.approx([0.4, 0.68, 13.4 / 14, 2.4 / 11],
                                                      abs=1e-15)


def test_reference_probabilities_by_hand_prior_two():
    refs = reference_probabilities(ref_train(), ref_rows(), 13, 2.0)
    assert refs["own rate"].tolist() == pytest.approx([0.4, 3.8 / 6, 0.92, 2.8 / 12], abs=1e-15)
    # The prior does not touch the other two references.
    assert refs["base rate per h"].tolist() == pytest.approx([0.5, 0.25, 0.5, 0.25])


def test_reference_probabilities_reads_the_window_given():
    # Window 4: (0, 0), (1, 4), (4, 4), (0, 3) -> .4, 1.4/5, 4.4/5, .4/4.
    refs = reference_probabilities(ref_train(), ref_rows(), 4, 1.0)
    assert refs["own rate"].tolist() == pytest.approx([0.4, 0.28, 0.88, 0.1], abs=1e-15)


def test_reference_probabilities_ignore_the_training_rows_history():
    # The train rows' own history columns (99, 99) play no part; only their paid and h.
    refs = reference_probabilities(ref_train(), ref_rows(), 13, 1.0)
    assert refs["own rate"].iloc[1] == pytest.approx(0.68)


@pytest.mark.parametrize("prior", [0.5, 1.0, 2.0, 3.0, 7.0, 1e-9, 1e6])
@pytest.mark.parametrize("paid_count", [2, 3, 7, 13])
def test_no_observed_week_gives_exactly_the_base_rate(prior, paid_count):
    # The contract: a row with no observed week gets EXACTLY the base rate, whatever prior.
    # 20 train rows, h alternating 1, 2; rows 0-3 give each h both outcomes; then
    # paid_count - 2 more payers -> base rate paid_count / 20 (.1, .15, .35, .65).
    paid = [True, True, False, False] + [True] * (paid_count - 2) + [False] * (18 - paid_count)
    train = ref_train(paid=paid, h=[1, 2] * 10)
    base = paid_count / 20
    assert train["paid"].mean() == base
    rows = ref_rows(paying_weeks_13=[0, 0, 0, 0], observed_weeks_13=[0, 0, 0, 0])
    refs = reference_probabilities(train, rows, 13, prior)
    assert refs["own rate"].tolist() == [base] * 4, (refs["own rate"].tolist(), base)
    assert (refs["own rate"] == refs["base rate"]).all()


def test_own_rate_never_reaches_zero_or_one():
    rows = ref_rows(paying_weeks_13=[0, 52, 0, 13], observed_weeks_13=[52, 52, 13, 13])
    refs = reference_probabilities(ref_train(), rows, 13, 1.0)
    assert ((refs["own rate"] > 0) & (refs["own rate"] < 1)).all()
    assert refs["own rate"].tolist() == pytest.approx([0.4 / 53, 52.4 / 53, 0.4 / 14, 13.4 / 14])


def test_own_rate_shrinks_towards_the_base_rate_as_the_prior_grows():
    rows = ref_rows()
    small = reference_probabilities(ref_train(), rows, 13, 0.01)["own rate"]
    large = reference_probabilities(ref_train(), rows, 13, 1e6)["own rate"]
    raw = pd.Series([np.nan, 0.75, 1.0, 0.2], index=rows.index)
    assert small.iloc[1:].tolist() == pytest.approx(raw.iloc[1:].tolist(), abs=0.01)
    assert large.tolist() == pytest.approx([0.4] * 4, abs=1e-4)


def test_reference_probabilities_accept_fractional_counts_and_nullable_paid():
    train = ref_train()
    train["paid"] = pd.array(train["paid"], dtype="boolean")
    rows = ref_rows(paying_weeks_13=[0.0, 1.5, 2.0, 0.0], observed_weeks_13=[0.0, 3.0, 2.0, 1.0])
    refs = reference_probabilities(train, rows, 13, 1.0)
    assert refs["own rate"].tolist() == pytest.approx([0.4, 1.9 / 4, 2.4 / 3, 0.4 / 2])


def test_reference_probabilities_do_not_change_their_inputs():
    train, rows = ref_train(), ref_rows()
    before = (train.copy(deep=True), rows.copy(deep=True))
    reference_probabilities(train, rows, 13, 1.0)
    pd.testing.assert_frame_equal(train, before[0])
    pd.testing.assert_frame_equal(rows, before[1])


def test_reference_probabilities_on_no_rows_is_empty():
    rows = ref_rows().iloc[:0]
    refs = reference_probabilities(ref_train(), rows, 13, 1.0)
    assert refs.empty and list(refs.columns) == ["base rate", "base rate per h", "own rate"]


@pytest.mark.parametrize("prior", [0, 0.0, -1.0, float("nan")])
def test_reference_probabilities_refuse_a_prior_not_above_zero(prior):
    with pytest.raises(ValueError) as error:
        reference_probabilities(ref_train(), ref_rows(), 13, prior)
    assert str(error.value) == f"prior_weeks is {prior!r}; it must be above 0"


@pytest.mark.parametrize("drop, missing", [
    (["paying_weeks_13"], "['paying_weeks_13']"),
    (["observed_weeks_13"], "['observed_weeks_13']"),
    (["paying_weeks_13", "observed_weeks_13"], "['paying_weeks_13', 'observed_weeks_13']"),
])
def test_reference_probabilities_refuse_missing_history_columns(drop, missing):
    with pytest.raises(ValueError) as error:
        reference_probabilities(ref_train(), ref_rows().drop(columns=drop), 13, 1.0)
    assert str(error.value) == f"the rows have no history columns {missing} for window 13"


def test_reference_probabilities_refuse_a_window_without_columns():
    with pytest.raises(ValueError) as error:
        reference_probabilities(ref_train(), ref_rows(), 52, 1.0)
    assert str(error.value) == ("the rows have no history columns ['paying_weeks_52', "
                                "'observed_weeks_52'] for window 52")


@pytest.mark.parametrize("paying, observed, count", [
    ([-1, 3, 13, 2], [0, 4, 13, 10], 1),
    ([0, 5, 13, 2], [0, 4, 13, 10], 1),
    ([np.nan, 3, 13, 2], [0, 4, 13, 10], 1),
    ([0, 3, 13, 2], [0, np.nan, 13, 10], 1),
    ([0, 3, 14, 11], [-1, 4, 13, 10], 3),
], ids=["negative paying", "paying above observed", "NaN paying", "NaN observed", "three"])
def test_reference_probabilities_refuse_impossible_counts(paying, observed, count):
    rows = ref_rows(paying_weeks_13=paying, observed_weeks_13=observed)
    with pytest.raises(ValueError) as error:
        reference_probabilities(ref_train(), rows, 13, 1.0)
    assert str(error.value) == (f"{count} rows have impossible history counts for window 13 "
                                f"(negative, NaN, or more paying than observed weeks)")


@pytest.mark.parametrize("value", [True, False])
def test_reference_probabilities_refuse_train_with_one_outcome(value):
    with pytest.raises(ValueError) as error:
        reference_probabilities(ref_train(paid=[value] * 10), ref_rows(), 13, 1.0)
    assert str(error.value) == "the training rows must hold both outcomes for a base rate"


def test_reference_probabilities_refuse_an_h_train_lacks():
    rows = ref_rows(h=np.array([2, 3, 5, 1], dtype="int64"))
    with pytest.raises(ValueError) as error:
        reference_probabilities(ref_train(), rows, 13, 1.0)
    assert str(error.value) == "the training rows have no rows with h [3, 5]"


@pytest.mark.parametrize("paid, bad", [
    ([True, True, True, True, True, True, True, False, False, False], "[1]"),
    ([True, False, False, False, False, False, False, False, False, False], "[2]"),
], ids=["h 1 all paid", "h 2 none paid"])
def test_reference_probabilities_refuse_an_h_with_one_outcome(paid, bad):
    with pytest.raises(ValueError) as error:
        reference_probabilities(ref_train(paid=paid), ref_rows(), 13, 1.0)
    assert str(error.value) == f"the training rows hold one outcome only for h {bad}"


def test_reference_probabilities_one_outcome_h_absent_from_rows_still_refused():
    # Train's h 3 holds one outcome; rows have no h 3. Pinned: the check covers every train h.
    train = ref_train(paid=[True, False, False, False, True, True, True, False, False, False,
                            True], h=[1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3])
    with pytest.raises(ValueError, match=r"one outcome only for h \[3\]"):
        reference_probabilities(train, ref_rows(), 13, 1.0)


def test_reference_probabilities_prior_is_checked_before_the_columns():
    with pytest.raises(ValueError, match="prior_weeks"):
        reference_probabilities(ref_train(), ref_rows().drop(columns="paying_weeks_13"), 13, 0)


# --- best_unweighted -------------------------------------------------------------------------

def weighted_candidates(weights):
    return [Candidate(params={"k": i}, settings=SimpleNamespace(tag=i), class_weight=w)
            for i, w in enumerate(weights)]


def test_best_unweighted_is_the_first_unweighted_row_in_grid_order():
    candidates = weighted_candidates(["none", "balanced", "none", "balanced"])
    grid = pd.DataFrame({"candidate": [3, 1, 2, 0], "log loss": [0.1, 0.2, 0.3, 0.4]})
    assert best_unweighted(grid, candidates) is candidates[2]


def test_best_unweighted_follows_the_grid_order_not_the_score_or_candidate_number():
    candidates = weighted_candidates(["none", "none", "balanced"])
    grid = pd.DataFrame({"candidate": [2, 1, 0], "log loss": [0.9, 0.5, 0.1]})
    assert best_unweighted(grid, candidates) is candidates[1]


def test_best_unweighted_top_row_unweighted_is_the_chosen_object():
    candidates = weighted_candidates(["balanced", "none"])
    grid = pd.DataFrame({"candidate": [1, 0]})
    chosen = candidates[int(grid.iloc[0]["candidate"])]
    assert best_unweighted(grid, candidates) is chosen


def test_best_unweighted_reads_float_candidate_numbers():
    candidates = weighted_candidates(["balanced", "none"])
    assert best_unweighted(pd.DataFrame({"candidate": [0.0, 1.0]}), candidates) is candidates[1]


def test_best_unweighted_none_when_every_setting_is_weighted():
    candidates = weighted_candidates(["balanced"] * 3)
    assert best_unweighted(pd.DataFrame({"candidate": [2, 0, 1]}), candidates) is None


def test_best_unweighted_none_for_an_empty_grid():
    grid = pd.DataFrame({"candidate": pd.Series([], dtype="int64")})
    assert best_unweighted(grid, weighted_candidates(["none"])) is None


def test_best_unweighted_only_looks_at_the_grids_candidates():
    # Candidate 1 is unweighted but not in the grid.
    candidates = weighted_candidates(["balanced", "none", "balanced"])
    assert best_unweighted(pd.DataFrame({"candidate": [2, 0]}), candidates) is None


def test_best_unweighted_on_real_candidate_lists():
    candidates = lr_candidates(stage1_settings(), diagnostic_settings())   # C x (none, balanced)
    grid = pd.DataFrame({"candidate": [5, 3, 2, 0, 1, 4]})
    assert candidates[5].class_weight == "balanced" and candidates[3].class_weight == "balanced"
    assert best_unweighted(grid, candidates) is candidates[2]


# --- calibration_comparison ------------------------------------------------------------------

def comparison_data(n=90, seed=12):
    """Rows with h 3, 1, 2 (first appearance unsorted), two models and a reference."""
    rng = np.random.default_rng(seed)
    index = pd.Index([f"r{i}" for i in range(n)])
    h = pd.Series(np.array([3, 1, 2] * (n // 3), dtype="int64"), index=index)
    p = pd.Series(rng.uniform(0.05, 0.95, n), index=index)
    actual = pd.Series(rng.random(n) < p, index=index)
    other = pd.Series(rng.uniform(0.05, 0.95, n), index=index)
    reference = pd.Series(actual.mean(), index=index)
    for horizon in (1, 2, 3):
        assert actual[h == horizon].nunique() == 2
    return actual, h, {"Model": p, "Another": other}, reference


def test_calibration_comparison_summary_shape_and_order():
    actual, h, predictions, reference = comparison_data()
    summary, _ = calibration_comparison(actual, h, predictions, reference, 3)
    assert list(summary.columns) == ["model", "h"] + SUMMARY_KEYS
    assert list(zip(summary["model"], summary["h"])) == [
        ("Model", 1), ("Model", 2), ("Model", 3), ("Model", "all"),
        ("Another", 1), ("Another", 2), ("Another", 3), ("Another", "all")]
    assert list(summary.index) == list(range(8))


def test_calibration_comparison_all_rows_equal_a_direct_summary():
    actual, h, predictions, reference = comparison_data()
    summary, _ = calibration_comparison(actual, h, predictions, reference, 4)
    for label, p in predictions.items():
        row = summary[(summary["model"] == label) & (summary["h"] == "all")].iloc[0]
        direct = calibration_summary(actual, p, reference, 4)
        for key in SUMMARY_KEYS:
            assert row[key] == direct[key], (label, key)


def test_calibration_comparison_per_h_rows_equal_a_direct_summary_on_those_rows():
    actual, h, predictions, reference = comparison_data()
    summary, _ = calibration_comparison(actual, h, predictions, reference, 3)
    for label, p in predictions.items():
        for horizon in (1, 2, 3):
            mask = h == horizon
            row = summary[(summary["model"] == label) & (summary["h"] == horizon)].iloc[0]
            direct = calibration_summary(actual[mask], p[mask], reference[mask], 3)
            for key in SUMMARY_KEYS:
                assert row[key] == direct[key], (label, horizon, key)


def test_calibration_comparison_reliability_shape_and_content():
    actual, h, predictions, reference = comparison_data()
    _, reliability = calibration_comparison(actual, h, predictions, reference, 5)
    assert list(reliability.columns) == ["model", "h"] + RELIABILITY_COLUMNS
    assert len(reliability) == 2 * 4 * 5
    assert list(reliability.index) == list(range(40))
    labels = [(m, x) for m in ("Model", "Another") for x in (1, 2, 3, "all")]
    assert list(dict.fromkeys(zip(reliability["model"], reliability["h"]))) == labels
    part = reliability[(reliability["model"] == "Another") & (reliability["h"] == 2)]
    mask = h == 2
    direct = reliability_table(actual[mask], predictions["Another"][mask], 5)
    pd.testing.assert_frame_equal(part.drop(columns=["model", "h"]).reset_index(drop=True),
                                  direct)
    for (_, _), block in reliability.groupby(["model", reliability["h"].astype(str)]):
        assert block["group"].tolist() == [1, 2, 3, 4, 5]
    all_rows = reliability[reliability["h"] == "all"]
    assert all_rows.groupby("model")["rows"].sum().tolist() == [90, 90]


def test_calibration_comparison_one_horizon_still_has_an_all_row():
    actual, _, predictions, reference = comparison_data()
    h = pd.Series(1, index=actual.index)
    summary, _ = calibration_comparison(actual, h, {"Model": predictions["Model"]}, reference, 3)
    assert summary["h"].tolist() == [1, "all"]
    one, everything = summary.iloc[0], summary.iloc[1]
    for key in SUMMARY_KEYS:
        assert one[key] == everything[key], key


def test_calibration_comparison_does_not_change_its_inputs():
    actual, h, predictions, reference = comparison_data()
    before = [s.copy() for s in (actual, h, reference, *predictions.values())]
    calibration_comparison(actual, h, predictions, reference, 3)
    for old, new in zip(before, (actual, h, reference, *predictions.values())):
        pd.testing.assert_series_equal(old, new)


@pytest.mark.parametrize("what", ["h", "reference_p", "Model", "Another"])
@pytest.mark.parametrize("how", ["relabelled", "reordered", "shorter"])
def test_calibration_comparison_refuses_misaligned_series(what, how):
    actual, h, predictions, reference = comparison_data()
    series = {"h": h, "reference_p": reference, **predictions}[what]
    if how == "relabelled":
        bad = series.set_axis([f"s{i}" for i in range(len(series))])
    elif how == "reordered":
        bad = series.iloc[::-1]
    else:
        bad = series.iloc[:-1]
    if what == "h":
        h = bad
    elif what == "reference_p":
        reference = bad
    else:
        predictions = {**predictions, what: bad}
    with pytest.raises(ValueError) as error:
        calibration_comparison(actual, h, predictions, reference, 3)
    assert str(error.value) == f"{what} must have the rows of actual (the same index)"


def test_calibration_comparison_checks_alignment_before_measuring():
    # A misaligned second model is refused even though the first could be measured.
    actual, h, predictions, reference = comparison_data()
    predictions["Another"] = predictions["Another"].iloc[1:]
    with pytest.raises(ValueError, match="Another must have the rows"):
        calibration_comparison(actual, h, predictions, reference, 3)


def test_calibration_comparison_refuses_a_horizon_with_one_outcome():
    actual, h, predictions, reference = comparison_data()
    actual = actual.copy()
    actual[h == 2] = True
    with pytest.raises(ValueError, match="both outcomes"):
        calibration_comparison(actual, h, predictions, reference, 3)


def test_calibration_comparison_refuses_too_many_groups_for_a_horizon():
    actual, h, predictions, reference = comparison_data()   # 30 rows per h
    with pytest.raises(ValueError, match="30 rows cannot be cut into 31 groups"):
        calibration_comparison(actual, h, predictions, reference, 31)


def test_calibration_comparison_with_references_as_predictions():
    # As the pipeline does: the references are scored like models, beside them.
    actual, h, predictions, reference = comparison_data()
    labels = {"Model": predictions["Model"], "base rate": reference}
    summary, _ = calibration_comparison(actual, h, labels, reference, 3)
    base_all = summary[(summary["model"] == "base rate") & (summary["h"] == "all")].iloc[0]
    assert base_all["log loss skill"] == pytest.approx(0.0, abs=1e-12)
    assert base_all["brier skill"] == pytest.approx(0.0, abs=1e-12)
    assert base_all["AUC"] == 0.5


# === U11.4: equal_groups, cash_by_p_group, amount_reliability ================================
# equal_groups is the shared grouping (reliability_table now uses it; its tests above are
# unchanged). cash_by_p_group checks p x m against the cash on all rows grouped by p;
# amount_reliability checks m against the paid amounts grouped by m.

from collection_estimation.hurdle.diagnostic import (  # noqa: E402
    amount_reliability,
    cash_by_p_group,
    equal_groups,
)

NAN, INF = float("nan"), float("inf")
CASH_COLUMNS = ["group", "p_low", "p_high", "mean_p", "expected_cash", "actual_cash", "ratio",
                "rows"]
AMOUNT_COLUMNS = ["group", "m_low", "m_high", "mean_m", "mean_actual", "ratio", "rows"]


# --- equal_groups --------------------------------------------------------------------------------

def test_equal_groups_by_hand():
    # Sorted: 0.1 (row 0), 0.2 (2), 0.3 (3), 0.4 (1), 0.6 (5), 0.8 (4).
    group = equal_groups(HAND_P, 3)
    assert isinstance(group, np.ndarray)
    assert group.tolist() == [1, 2, 1, 2, 3, 3]


@pytest.mark.parametrize("n, groups, sizes", [(7, 3, [3, 2, 2]), (10, 4, [3, 2, 3, 2]),
                                              (6, 6, [1] * 6), (5, 2, [3, 2]), (8, 2, [4, 4])])
def test_equal_groups_sizes_follow_position_times_groups_over_n(n, groups, sizes):
    group = equal_groups(pd.Series(np.linspace(0.0, 1.0, n)), groups)
    assert np.bincount(group)[1:].tolist() == sizes


def test_equal_groups_lowest_predictions_are_group_one_and_groups_are_ordered():
    rng = np.random.default_rng(7)
    prediction = pd.Series(rng.normal(size=97))
    group = equal_groups(prediction, 5)
    assert set(group) == {1, 2, 3, 4, 5}
    for g in range(1, 5):
        assert prediction[group == g].max() <= prediction[group == g + 1].min()


def test_equal_groups_ties_keep_row_order():
    assert equal_groups(pd.Series([0.5] * 4), 2).tolist() == [1, 1, 2, 2]


def test_equal_groups_ties_keep_row_order_not_index_order():
    prediction = pd.Series([0.3, 0.3, 0.3, 0.3], index=[9, 3, 7, 1])
    assert equal_groups(prediction, 2).tolist() == [1, 1, 2, 2]


def test_equal_groups_follow_row_positions_on_a_shuffled_index():
    prediction = pd.Series([5.0, 1.0, 3.0, 2.0], index=["d", "a", "c", "b"])
    assert equal_groups(prediction, 2).tolist() == [2, 1, 2, 1]


def test_equal_groups_does_not_change_its_input():
    prediction = HAND_P.copy()
    equal_groups(prediction, 3)
    pd.testing.assert_series_equal(prediction, HAND_P)


def test_equal_groups_accepts_negative_and_large_predictions():
    group = equal_groups(pd.Series([-5.0, 1e12, 0.0, -1e12]), 2)
    assert group.tolist() == [1, 2, 2, 1]


def test_equal_groups_accepts_ints():
    assert equal_groups(pd.Series([3, 1, 2, 4]), 2).tolist() == [2, 1, 1, 2]


@pytest.mark.parametrize("groups", [1, 0, -2])
def test_equal_groups_refuses_fewer_than_two_groups(groups):
    with pytest.raises(ValueError) as error:
        equal_groups(HAND_P, groups)
    assert str(error.value) == f"groups is {groups}; at least 2 are needed"


@pytest.mark.parametrize("groups", [7, 100])
def test_equal_groups_refuses_more_groups_than_rows(groups):
    with pytest.raises(ValueError) as error:
        equal_groups(HAND_P, groups)
    assert str(error.value) == f"6 rows cannot be cut into {groups} groups"


def test_equal_groups_refuses_empty():
    with pytest.raises(ValueError, match="0 rows cannot be cut into 2 groups"):
        equal_groups(pd.Series([], dtype=float), 2)


def test_equal_groups_refuses_nan_and_counts_them():
    with pytest.raises(ValueError) as error:
        equal_groups(pd.Series([0.1, NAN, 0.3, NAN]), 2)
    assert str(error.value) == "the prediction is NaN in 2 rows"


def test_equal_groups_refuses_nullable_na():
    with pytest.raises(ValueError, match="NaN in 1 rows"):
        equal_groups(pd.Series(pd.array([0.1, None, 0.3], dtype="Float64")), 2)


def test_equal_groups_checks_groups_before_rows_before_nan():
    with pytest.raises(ValueError, match="at least 2"):
        equal_groups(pd.Series([NAN]), 1)
    with pytest.raises(ValueError, match="cannot be cut"):
        equal_groups(pd.Series([NAN]), 2)


def test_reliability_table_uses_equal_groups():
    rng = np.random.default_rng(13)
    p = pd.Series(rng.random(50), index=rng.permutation(50))
    actual = pd.Series(rng.random(50) < p.to_numpy(), index=p.index)
    table = reliability_table(actual, p, 4)
    group = equal_groups(p, 4)
    for g in range(1, 5):
        row = table[table["group"] == g].iloc[0]
        assert row["rows"] == (group == g).sum()
        assert row["mean_p"] == pytest.approx(p[group == g].mean())
        assert row["share_paid"] == pytest.approx(actual[group == g].mean())


# --- cash_by_p_group -----------------------------------------------------------------------------
# Six rows, p as HAND_P (groups of 3: rows 0, 2 | 3, 1 | 5, 4). m = 100 to 600; actual cash
# 0, 500, 0, 0, 1000, 0.
#   group 1: rows 0, 2: p x m = 10 + 60 = 70, actual 0 -> ratio NaN
#   group 2: rows 3, 1: p x m = 120 + 80 = 200, actual 500 -> 0.4
#   group 3: rows 5, 4: p x m = 360 + 400 = 760, actual 1000 -> 0.76

CASH_M = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
CASH_ACTUAL = pd.Series([0.0, 500.0, 0.0, 0.0, 1000.0, 0.0])


def test_cash_by_p_group_by_hand():
    table = cash_by_p_group(HAND_P, CASH_M, CASH_ACTUAL, 3)
    assert list(table.columns) == CASH_COLUMNS
    assert table["group"].tolist() == [1, 2, 3]
    assert table["p_low"].tolist() == pytest.approx([0.1, 0.3, 0.6])
    assert table["p_high"].tolist() == pytest.approx([0.2, 0.4, 0.8])
    assert table["mean_p"].tolist() == pytest.approx([0.15, 0.35, 0.7])
    assert table["expected_cash"].tolist() == pytest.approx([70.0, 200.0, 760.0])
    assert table["actual_cash"].tolist() == pytest.approx([0.0, 500.0, 1000.0])
    assert np.isnan(table["ratio"].iloc[0])
    assert table["ratio"].iloc[1:].tolist() == pytest.approx([0.4, 0.76])
    assert table["rows"].tolist() == [2, 2, 2]


def test_cash_by_p_group_totals_add_up_to_the_whole():
    rng = np.random.default_rng(2)
    p = pd.Series(rng.random(103))
    m = pd.Series(rng.random(103) * 1000)
    actual = pd.Series(np.where(rng.random(103) < p, m * rng.random(103) * 2, 0.0))
    table = cash_by_p_group(p, m, actual, 7)
    assert table["expected_cash"].sum() == pytest.approx(float((p * m).sum()))
    assert table["actual_cash"].sum() == pytest.approx(float(actual.sum()))
    assert int(table["rows"].sum()) == 103


def test_cash_by_p_group_groups_are_the_equal_groups_of_p():
    rng = np.random.default_rng(3)
    p = pd.Series(rng.random(40), index=rng.permutation(40) + 100)
    m = pd.Series(rng.random(40) * 10, index=p.index)
    actual = pd.Series(rng.random(40) * 10, index=p.index)
    table = cash_by_p_group(p, m, actual, 4)
    group = equal_groups(p, 4)
    for g in range(1, 5):
        mine = group == g
        row = table.iloc[g - 1]
        assert row["expected_cash"] == pytest.approx(float((p * m)[mine].sum()))
        assert row["actual_cash"] == pytest.approx(float(actual[mine].sum()))


def test_cash_by_p_group_a_calibrated_model_has_ratios_near_one():
    rng = np.random.default_rng(4)
    n = 40_000
    p = pd.Series(rng.random(n))
    m = pd.Series(rng.lognormal(mean=5, sigma=0.3, size=n))
    actual = pd.Series(np.where(rng.random(n) < p, m, 0.0))
    table = cash_by_p_group(p, m, actual, 5)
    np.testing.assert_allclose(table["ratio"], 1.0, atol=0.05)


def test_cash_by_p_group_does_not_depend_on_row_order():
    shuffled = [4, 0, 5, 2, 1, 3]
    a = cash_by_p_group(HAND_P, CASH_M, CASH_ACTUAL, 3)
    b = cash_by_p_group(HAND_P[shuffled], CASH_M[shuffled], CASH_ACTUAL[shuffled], 3)
    pd.testing.assert_frame_equal(a, b)


def test_cash_by_p_group_works_on_a_non_default_index():
    index = ["f", "e", "d", "c", "b", "a"]
    table = cash_by_p_group(HAND_P.set_axis(index), CASH_M.set_axis(index),
                            CASH_ACTUAL.set_axis(index), 3)
    assert table["expected_cash"].tolist() == pytest.approx([70.0, 200.0, 760.0])


def test_cash_by_p_group_accepts_int_and_nullable_actuals():
    table = cash_by_p_group(HAND_P, CASH_M, CASH_ACTUAL.astype("int64"), 3)
    assert table["actual_cash"].dtype == "float64"
    assert table["actual_cash"].tolist() == [0.0, 500.0, 1000.0]
    table = cash_by_p_group(HAND_P, CASH_M, CASH_ACTUAL.astype("Float64"), 3)
    assert table["actual_cash"].tolist() == [0.0, 500.0, 1000.0]


def test_cash_by_p_group_a_negative_m_is_accepted():
    # Stage 2 with squared error can predict m below 0; the check does not hide it.
    table = cash_by_p_group(HAND_P, CASH_M.where(CASH_M.index != 0, -100.0), CASH_ACTUAL, 3)
    assert table["expected_cash"].iloc[0] == pytest.approx(-10.0 + 60.0)


def test_cash_by_p_group_all_actual_zero_gives_all_nan_ratios():
    table = cash_by_p_group(HAND_P, CASH_M, CASH_ACTUAL * 0, 3)
    assert table["ratio"].isna().all()


def test_cash_by_p_group_does_not_change_its_inputs():
    p, m, actual = HAND_P.copy(), CASH_M.copy(), CASH_ACTUAL.copy()
    cash_by_p_group(p, m, actual, 3)
    pd.testing.assert_series_equal(p, HAND_P)
    pd.testing.assert_series_equal(m, CASH_M)
    pd.testing.assert_series_equal(actual, CASH_ACTUAL)


@pytest.mark.parametrize("which", ["m", "actual_amount"])
@pytest.mark.parametrize("change", [lambda s: s.set_axis(range(1, 7)),
                                    lambda s: s.iloc[::-1], lambda s: s.iloc[:-1]],
                         ids=["shifted", "reversed", "shorter"])
def test_cash_by_p_group_refuses_misaligned_inputs(which, change):
    inputs = {"m": CASH_M, "actual_amount": CASH_ACTUAL}
    inputs[which] = change(inputs[which])
    with pytest.raises(ValueError) as error:
        cash_by_p_group(HAND_P, inputs["m"], inputs["actual_amount"], 3)
    assert str(error.value) == f"{which} must have the rows of p (the same index)"


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
@pytest.mark.parametrize("which", ["m", "actual_amount"])
def test_cash_by_p_group_refuses_nan_and_infinite(which, bad):
    inputs = {"m": CASH_M.copy(), "actual_amount": CASH_ACTUAL.copy()}
    inputs[which].iloc[2] = bad
    with pytest.raises(ValueError) as error:
        cash_by_p_group(HAND_P, inputs["m"], inputs["actual_amount"], 3)
    assert str(error.value) == f"{which} holds a NaN or infinite value"


def test_cash_by_p_group_refuses_a_negative_actual():
    with pytest.raises(ValueError) as error:
        cash_by_p_group(HAND_P, CASH_M, CASH_ACTUAL.where(CASH_ACTUAL.index != 3, -1.0), 3)
    assert str(error.value) == ("actual_amount is below 0 in some rows; it is the cash "
                                "received")


def test_cash_by_p_group_refuses_bad_groups_and_nan_p_as_equal_groups_does():
    with pytest.raises(ValueError, match="at least 2 are needed"):
        cash_by_p_group(HAND_P, CASH_M, CASH_ACTUAL, 1)
    with pytest.raises(ValueError, match="6 rows cannot be cut into 7 groups"):
        cash_by_p_group(HAND_P, CASH_M, CASH_ACTUAL, 7)
    with pytest.raises(ValueError, match="the prediction is NaN in 1 rows"):
        cash_by_p_group(HAND_P.where(HAND_P.index != 0, NAN), CASH_M, CASH_ACTUAL, 3)


# --- amount_reliability --------------------------------------------------------------------------
# Six paid rows, m = 50, 10, 40, 20, 60, 30 (sorted rows 1, 3, 5, 2, 0, 4); groups of 3:
#   group 1: rows 1, 3: m 10, 20 (mean 15); actual 12, 24 (mean 18) -> 1.2
#   group 2: rows 5, 2: m 30, 40 (mean 35); actual 35, 35 (mean 35) -> 1.0
#   group 3: rows 0, 4: m 50, 60 (mean 55); actual 22, 44 (mean 33) -> 0.6

AMOUNT_M = pd.Series([50.0, 10.0, 40.0, 20.0, 60.0, 30.0])
AMOUNT_ACTUAL = pd.Series([22.0, 12.0, 35.0, 24.0, 44.0, 35.0])


def test_amount_reliability_by_hand():
    table = amount_reliability(AMOUNT_ACTUAL, AMOUNT_M, 3)
    assert list(table.columns) == AMOUNT_COLUMNS
    assert table["group"].tolist() == [1, 2, 3]
    assert table["m_low"].tolist() == [10.0, 30.0, 50.0]
    assert table["m_high"].tolist() == [20.0, 40.0, 60.0]
    assert table["mean_m"].tolist() == pytest.approx([15.0, 35.0, 55.0])
    assert table["mean_actual"].tolist() == pytest.approx([18.0, 35.0, 33.0])
    assert table["ratio"].tolist() == pytest.approx([1.2, 1.0, 0.6])
    assert table["rows"].tolist() == [2, 2, 2]


def test_amount_reliability_groups_by_m_not_by_the_actual():
    group = equal_groups(AMOUNT_M, 3)
    table = amount_reliability(AMOUNT_ACTUAL, AMOUNT_M, 3)
    for g in range(1, 4):
        assert table.iloc[g - 1]["mean_actual"] == pytest.approx(
            AMOUNT_ACTUAL[group == g].mean())


def test_amount_reliability_a_perfect_mean_model_is_calibrated_in_every_group():
    # Regression to the mean: grouped by m (the true mean), the mean actual matches m.
    rng = np.random.default_rng(6)
    n = 60_000
    m = pd.Series(rng.lognormal(mean=6, sigma=1.0, size=n))
    actual = pd.Series(m * rng.gamma(shape=4.0, scale=0.25, size=n))   # mean 1 multiplier
    table = amount_reliability(actual, m, 5)
    np.testing.assert_allclose(table["ratio"], 1.0, atol=0.03)


def test_amount_reliability_does_not_depend_on_row_order():
    shuffled = [4, 0, 5, 2, 1, 3]
    a = amount_reliability(AMOUNT_ACTUAL, AMOUNT_M, 3)
    b = amount_reliability(AMOUNT_ACTUAL[shuffled], AMOUNT_M[shuffled], 3)
    pd.testing.assert_frame_equal(a, b)


def test_amount_reliability_accepts_int_actuals():
    table = amount_reliability(AMOUNT_ACTUAL.astype("int64"), AMOUNT_M, 3)
    assert table["mean_actual"].tolist() == pytest.approx([18.0, 35.0, 33.0])


def test_amount_reliability_does_not_change_its_inputs():
    actual, m = AMOUNT_ACTUAL.copy(), AMOUNT_M.copy()
    amount_reliability(actual, m, 3)
    pd.testing.assert_series_equal(actual, AMOUNT_ACTUAL)
    pd.testing.assert_series_equal(m, AMOUNT_M)


@pytest.mark.parametrize("change", [lambda s: s.set_axis(range(1, 7)),
                                    lambda s: s.iloc[::-1], lambda s: s.iloc[:-1]],
                         ids=["shifted", "reversed", "shorter"])
def test_amount_reliability_refuses_a_misaligned_m(change):
    with pytest.raises(ValueError) as error:
        amount_reliability(AMOUNT_ACTUAL, change(AMOUNT_M), 3)
    assert str(error.value) == "m must have the rows of actual (the same index)"


@pytest.mark.parametrize("bad, message", [
    (0.0, "amount is 0 or below in 1 rows"),
    (-5.0, "amount is 0 or below in 1 rows"),
    (NAN, "amount has 1 missing values"),
    (INF, "amount holds an infinite value"),
], ids=["zero", "negative", "nan", "inf"])
def test_amount_reliability_refuses_actuals_as_check_amounts_does(bad, message):
    actual = AMOUNT_ACTUAL.copy()
    actual.iloc[2] = bad
    with pytest.raises(ValueError) as error:
        amount_reliability(actual, AMOUNT_M, 3)
    assert str(error.value).startswith(message)


def test_amount_reliability_refuses_no_rows():
    with pytest.raises(ValueError, match="there are no paid rows"):
        amount_reliability(pd.Series([], dtype=float), pd.Series([], dtype=float), 2)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_amount_reliability_refuses_nan_and_infinite_m(bad):
    m = AMOUNT_M.copy()
    m.iloc[0] = bad
    with pytest.raises(ValueError) as error:
        amount_reliability(AMOUNT_ACTUAL, m, 3)
    assert str(error.value) == "m holds a NaN or infinite value"


def test_amount_reliability_checks_the_index_then_the_actual_then_m():
    bad_actual = AMOUNT_ACTUAL.where(AMOUNT_ACTUAL.index != 0, 0.0)
    bad_m = AMOUNT_M.where(AMOUNT_M.index != 0, NAN)
    with pytest.raises(ValueError, match="^m must have the rows"):
        amount_reliability(bad_actual, bad_m.iloc[:-1], 3)
    with pytest.raises(ValueError, match="^amount is 0 or below"):
        amount_reliability(bad_actual, bad_m, 3)
    with pytest.raises(ValueError, match="^m holds a NaN"):
        amount_reliability(AMOUNT_ACTUAL, bad_m, 3)


def test_amount_reliability_refuses_bad_groups_as_equal_groups_does():
    with pytest.raises(ValueError, match="at least 2 are needed"):
        amount_reliability(AMOUNT_ACTUAL, AMOUNT_M, 1)
    with pytest.raises(ValueError, match="6 rows cannot be cut into 7 groups"):
        amount_reliability(AMOUNT_ACTUAL, AMOUNT_M, 7)
