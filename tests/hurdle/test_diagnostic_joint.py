"""Tests for `hurdle/diagnostic_joint.py` (U11.5): `stage1_grid`, `stage2_grid`,
`total_scores`, `scores_by_h`, `score_pairs` (on the panel rows, with p and m as Series;
`total_codes` was removed) and `paired_bootstrap`; and (U11.5b) the eligibility gates
`stage1_gates` and `stage2_gates`, at the end of the file.

Every table is hand-made; nothing is read from data/ and nothing is fitted. Hypothesis is not
installed in this env, so the brute-force checks use seeded random trials.
"""
import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.diagnostic import Candidate
from collection_estimation.hurdle.diagnostic_joint import (
    check_actual_totals,
    paired_bootstrap,
    score_pairs,
    scores_by_h,
    stage1_grid,
    stage2_grid,
    total_scores,
)
from collection_estimation.hurdle.settings import (
    FEATURE_BLOCKS,
    JointDiagnosticSettings,
    Stage1Settings,
    Stage2Settings,
)

nan, inf = float("nan"), float("inf")
SCORE_KEYS = ["totals", "zero actual", "MAPE", "bias", "WAPE"]


def stage1_settings(**changes):
    base = Stage1Settings(max_iter=7, learning_rate=0.3, max_leaf_nodes=5, min_samples_leaf=4,
                          random_seed=11, feature_blocks=("history", "calendar"),
                          company_as_feature=True, baseline_c=0.5)
    return dataclasses.replace(base, **changes)


def stage2_settings(**changes):
    base = Stage2Settings(loss="poisson", max_iter=10, learning_rate=0.1, max_leaf_nodes=7,
                          min_samples_leaf=5, random_seed=3, feature_blocks=FEATURE_BLOCKS,
                          scale="none", balance_floor="mean", own_mean_window=4,
                          own_mean_prior_weeks=1.0)
    return dataclasses.replace(base, **changes)


def joint_settings(**changes):
    base = JointDiagnosticSettings(
        run=True, stage1_max_iter=(100, 300), stage1_learning_rate=(0.05,),
        stage1_max_leaf_nodes=(15, 31, 63), stage2_loss=("poisson", "gamma"),
        stage2_max_iter=(200,), stage2_learning_rate=(0.05, 0.1), stage2_max_leaf_nodes=(31,),
        stage2_scale=("none", "open balance"), stage2_balance_floor=("median", "own mean"),
        stage1_log_loss_tolerance=0.10, stage1_slope_band=(0.8, 1.25),
        stage2_skill_tolerance=0.05, stage2_ratio_band=(0.9, 1.1),
        selection_metric="mape", bootstrap_samples=200, output_folder=Path("unused"))
    return dataclasses.replace(base, **changes)


# --- stage1_grid -----------------------------------------------------------------------------

def test_stage1_grid_every_combination_in_product_order():
    grid = stage1_grid(stage1_settings(), joint_settings())
    assert all(isinstance(c, Candidate) for c in grid)
    assert [c.params for c in grid] == [
        {"max_iter": t, "learning_rate": r, "max_leaf_nodes": n}
        for t, r, n in itertools.product((100, 300), (0.05,), (15, 31, 63))]


def test_stage1_grid_settings_carry_the_params_and_keep_everything_else():
    stage1 = stage1_settings()
    for candidate in stage1_grid(stage1, joint_settings()):
        s = candidate.settings
        assert (s.max_iter, s.learning_rate, s.max_leaf_nodes) == (
            candidate.params["max_iter"], candidate.params["learning_rate"],
            candidate.params["max_leaf_nodes"])
        assert dataclasses.replace(s, max_iter=7, learning_rate=0.3, max_leaf_nodes=5) == stage1


def test_stage1_grid_has_no_class_weights():
    assert {c.class_weight for c in stage1_grid(stage1_settings(), joint_settings())} == {"none"}


def test_stage1_grid_does_not_change_the_stage1_settings():
    stage1 = stage1_settings()
    stage1_grid(stage1, joint_settings())
    assert stage1 == stage1_settings()


def test_stage1_grid_of_one_value_each_is_one_candidate():
    joint = joint_settings(stage1_max_iter=(900,), stage1_learning_rate=(0.01,),
                           stage1_max_leaf_nodes=(63,))
    grid = stage1_grid(stage1_settings(), joint)
    assert len(grid) == 1
    assert grid[0].params == {"max_iter": 900, "learning_rate": 0.01, "max_leaf_nodes": 63}


# --- stage2_grid -----------------------------------------------------------------------------

def test_stage2_grid_size_counts_floors_only_under_open_balance():
    # 2 losses x 1 x 2 rates x 1 x (1 for "none" + 2 floors for "open balance") = 12.
    assert len(stage2_grid(stage2_settings(), joint_settings())) == 12


def test_stage2_grid_order_and_params():
    grid = stage2_grid(stage2_settings(), joint_settings())
    expected = []
    for loss, rate in itertools.product(("poisson", "gamma"), (0.05, 0.1)):
        common = {"loss": loss, "max_iter": 200, "learning_rate": rate, "max_leaf_nodes": 31}
        expected.append({**common, "scale": "none", "balance_floor": "-"})
        expected += [{**common, "scale": "open balance", "balance_floor": f}
                     for f in ("median", "own mean")]
    assert [c.params for c in grid] == expected


def test_stage2_grid_under_none_the_floor_is_the_stage2_settings_own():
    grid = stage2_grid(stage2_settings(balance_floor="mean"), joint_settings())
    nones = [c for c in grid if c.params["scale"] == "none"]
    assert nones and all(c.settings.balance_floor == "mean" for c in nones)
    assert all(c.params["balance_floor"] == "-" for c in nones)


def test_stage2_grid_under_open_balance_the_floor_is_the_grids():
    grid = stage2_grid(stage2_settings(balance_floor="mean"), joint_settings())
    for c in grid:
        if c.params["scale"] == "open balance":
            assert c.settings.balance_floor == c.params["balance_floor"]
            assert c.settings.scale == "open balance"


def test_stage2_grid_settings_carry_the_params_and_keep_everything_else():
    stage2 = stage2_settings()
    for c in stage2_grid(stage2, joint_settings()):
        s = c.settings
        assert (s.loss, s.max_iter, s.learning_rate, s.max_leaf_nodes, s.scale) == (
            c.params["loss"], c.params["max_iter"], c.params["learning_rate"],
            c.params["max_leaf_nodes"], c.params["scale"])
        assert (s.min_samples_leaf, s.random_seed, s.feature_blocks, s.own_mean_window,
                s.own_mean_prior_weeks) == (5, 3, FEATURE_BLOCKS, 4, 1.0)
        assert c.class_weight == "none"


def test_stage2_grid_only_none_scale_gives_one_candidate_per_combination():
    grid = stage2_grid(stage2_settings(), joint_settings(stage2_scale=("none",)))
    assert len(grid) == 4
    assert {c.params["balance_floor"] for c in grid} == {"-"}


def test_stage2_grid_only_open_balance():
    grid = stage2_grid(stage2_settings(), joint_settings(stage2_scale=("open balance",),
                                                         stage2_balance_floor=("mean",)))
    assert len(grid) == 4
    assert {c.settings.balance_floor for c in grid} == {"mean"}


def test_stage2_grid_settings_are_distinct_and_hashable():
    grid = stage2_grid(stage2_settings(), joint_settings())
    # The pipeline caches fits keyed by the settings: equal settings would share a fit.
    assert len({c.settings for c in grid}) == len(grid)


def test_stage2_grid_does_not_change_the_stage2_settings():
    stage2 = stage2_settings()
    stage2_grid(stage2, joint_settings())
    assert stage2 == stage2_settings()


# --- total_scores ----------------------------------------------------------------------------

def test_total_scores_by_hand():
    actual = pd.Series([100.0, 200.0, 0.0, 50.0])
    forecast = pd.Series([110.0, 150.0, 20.0, 50.0])
    scores = total_scores(actual, forecast)
    assert list(scores) == SCORE_KEYS
    assert scores["totals"] == 4
    assert scores["zero actual"] == 1
    # MAPE over the 3 positive totals: (0.1 + 0.25 + 0) / 3
    assert scores["MAPE"] == pytest.approx(0.35 / 3)
    # bias: (330 - 350) / 350; WAPE: (10 + 50 + 20 + 0) / 350
    assert scores["bias"] == pytest.approx(-20 / 350)
    assert scores["WAPE"] == pytest.approx(80 / 350)


def test_total_scores_perfect_forecast_is_all_zero():
    actual = pd.Series([1.0, 2.0, 3.0])
    scores = total_scores(actual, actual.copy())
    assert (scores["MAPE"], scores["bias"], scores["WAPE"]) == (0.0, 0.0, 0.0)


def test_total_scores_types():
    scores = total_scores(pd.Series([1.0, 2.0]), pd.Series([1.5, 2.5]))
    assert type(scores["totals"]) is int and type(scores["zero actual"]) is int
    assert all(type(scores[k]) is float for k in ("MAPE", "bias", "WAPE"))


def test_total_scores_under_and_over_forecasts_cancel_in_bias_not_in_wape():
    scores = total_scores(pd.Series([100.0, 100.0]), pd.Series([50.0, 150.0]))
    assert scores["bias"] == 0.0
    assert scores["WAPE"] == pytest.approx(0.5)
    assert scores["MAPE"] == pytest.approx(0.5)


def test_total_scores_mape_weighs_weeks_equally_wape_by_size():
    scores = total_scores(pd.Series([10.0, 1000.0]), pd.Series([20.0, 1000.0]))
    assert scores["MAPE"] == pytest.approx(0.5)
    assert scores["WAPE"] == pytest.approx(10 / 1010)


def test_total_scores_works_on_a_shuffled_index():
    actual = pd.Series([100.0, 200.0], index=["b", "a"])
    forecast = pd.Series([110.0, 100.0], index=["b", "a"])
    assert total_scores(actual, forecast)["MAPE"] == pytest.approx((0.1 + 0.5) / 2)


def test_total_scores_accepts_int_actuals():
    scores = total_scores(pd.Series([100, 0, 200]), pd.Series([100.0, 5.0, 100.0]))
    assert scores["zero actual"] == 1
    assert scores["MAPE"] == pytest.approx(0.25)


def test_total_scores_single_total():
    scores = total_scores(pd.Series([4.0]), pd.Series([5.0]))
    assert scores["totals"] == 1 and scores["MAPE"] == pytest.approx(0.25)


@pytest.mark.parametrize("forecast", [
    pd.Series([1.0, 2.0], index=[1, 2]),
    pd.Series([1.0]),
    pd.Series([2.0, 1.0], index=[1, 0]),
], ids=["shifted", "shorter", "reversed"])
def test_total_scores_refuses_misaligned_totals(forecast):
    with pytest.raises(ValueError) as error:
        total_scores(pd.Series([1.0, 2.0]), forecast)
    assert str(error.value) == "forecast and actual must have the same totals (the same index)"


def test_total_scores_refuses_empty():
    with pytest.raises(ValueError) as error:
        total_scores(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert str(error.value) == "there are no totals to score"


@pytest.mark.parametrize("bad", [nan, inf, -inf])
@pytest.mark.parametrize("which", ["actual", "forecast"])
def test_total_scores_refuses_nan_and_infinite(bad, which):
    series = {"actual": pd.Series([1.0, 2.0]), "forecast": pd.Series([1.0, 2.0])}
    series[which] = pd.Series([1.0, bad])
    with pytest.raises(ValueError) as error:
        total_scores(series["actual"], series["forecast"])
    assert str(error.value) == f"{which} holds a NaN or infinite value"


def test_total_scores_refuses_a_negative_actual():
    with pytest.raises(ValueError) as error:
        total_scores(pd.Series([1.0, -0.5]), pd.Series([1.0, 1.0]))
    assert str(error.value) == "an actual total is below 0; the totals are cash received"


def test_total_scores_accepts_a_negative_forecast():
    scores = total_scores(pd.Series([100.0]), pd.Series([-100.0]))
    assert scores["MAPE"] == pytest.approx(2.0) and scores["bias"] == pytest.approx(-2.0)


def test_total_scores_refuses_all_zero_actuals():
    with pytest.raises(ValueError) as error:
        total_scores(pd.Series([0.0, 0.0]), pd.Series([1.0, 2.0]))
    assert str(error.value) == "every actual total is 0: no error relative to the cash exists"


def test_total_scores_does_not_change_its_inputs():
    actual, forecast = pd.Series([1.0, 0.0, 3.0]), pd.Series([2.0, 1.0, 3.0])
    copies = actual.copy(), forecast.copy()
    total_scores(actual, forecast)
    pd.testing.assert_series_equal(actual, copies[0])
    pd.testing.assert_series_equal(forecast, copies[1])


# --- scores_by_h -----------------------------------------------------------------------------

def h_totals(companies=False):
    frame = pd.DataFrame({
        "origin_week": [40, 41, 40, 41, 40, 41],
        "h": [1, 1, 2, 2, 3, 3],
        "actual": [100.0, 200.0, 100.0, 0.0, 400.0, 100.0],
        "a": [110.0, 180.0, 50.0, 10.0, 400.0, 150.0],
        "b": [100.0, 200.0, 100.0, 0.0, 300.0, 100.0],
    })
    if companies:
        frame = pd.concat([frame.assign(company_code="TR02"),
                           frame.assign(company_code="TR04",
                                        a=frame["a"] * 2, actual=frame["actual"] * 3)],
                          ignore_index=True)
    return frame


def test_scores_by_h_consolidated_rows_and_columns():
    table = scores_by_h(h_totals(), ["a", "b"])
    assert list(table.columns) == ["model", "h", *SCORE_KEYS]
    assert table[["model", "h"]].values.tolist() == [
        ["a", 1], ["a", 2], ["a", 3], ["a", "all"], ["b", 1], ["b", 2], ["b", 3], ["b", "all"]]


def test_scores_by_h_each_row_is_total_scores_of_its_subset():
    totals = h_totals()
    table = scores_by_h(totals, ["a", "b"])
    for _, row in table.iterrows():
        part = totals if row["h"] == "all" else totals[totals["h"] == row["h"]]
        expected = total_scores(part["actual"], part[row["model"]])
        for key in SCORE_KEYS:
            assert row[key] == pytest.approx(expected[key]), (row["model"], row["h"], key)


def test_scores_by_h_by_hand_for_h_2():
    table = scores_by_h(h_totals(), ["a"])
    row = table[table["h"] == 2].iloc[0]
    # actual 100, 0; forecast 50, 10: MAPE 0.5 over the one positive total; WAPE 60/100.
    assert row["zero actual"] == 1
    assert row["MAPE"] == pytest.approx(0.5)
    assert row["WAPE"] == pytest.approx(0.6)
    assert row["bias"] == pytest.approx(-0.4)


def test_scores_by_h_per_company():
    totals = h_totals(companies=True)
    table = scores_by_h(totals, ["a"])
    assert list(table.columns) == ["model", "company_code", "h", *SCORE_KEYS]
    assert table[["company_code", "h"]].values.tolist() == [
        ["TR02", 1], ["TR02", 2], ["TR02", 3], ["TR02", "all"],
        ["TR04", 1], ["TR04", 2], ["TR04", 3], ["TR04", "all"]]
    for _, row in table.iterrows():
        part = totals[totals["company_code"] == row["company_code"]]
        if row["h"] != "all":
            part = part[part["h"] == row["h"]]
        assert row["MAPE"] == pytest.approx(total_scores(part["actual"], part["a"])["MAPE"])


def test_scores_by_h_company_codes_are_plain_values_not_tuples():
    table = scores_by_h(h_totals(companies=True), ["a"])
    assert set(table["company_code"]) == {"TR02", "TR04"}


def test_scores_by_h_categorical_companies_with_an_unused_category():
    totals = h_totals(companies=True)
    totals["company_code"] = pd.Categorical(totals["company_code"],
                                            categories=["TR02", "TR04", "TR05"])
    table = scores_by_h(totals, ["a"])
    assert set(table["company_code"].astype(str)) == {"TR02", "TR04"}


def test_scores_by_h_h_values_are_sorted():
    totals = h_totals().iloc[::-1].reset_index(drop=True)
    table = scores_by_h(totals, ["a"])
    assert table["h"].tolist() == [1, 2, 3, "all"]


def test_scores_by_h_refuses_an_h_whose_actuals_are_all_zero():
    totals = h_totals()
    totals.loc[totals["h"] == 1, "actual"] = 0.0
    with pytest.raises(ValueError, match="every actual total is 0"):
        scores_by_h(totals, ["a"])


def test_scores_by_h_no_models_is_an_empty_table():
    assert len(scores_by_h(h_totals(), [])) == 0


# --- score_pairs -----------------------------------------------------------------------------
# score_pairs(rows, p_by_stage1, m_by_stage2, metric): the totals are summed per (origin_week,
# h) from `rows` itself (actual = amount_deflated, forecast = p x m), all companies together.

def pair_rows(seed=0, n=40, origins=(40, 41, 42), horizons=(1, 2), index=None):
    """Panel-like rows: every (origin_week, h) has at least one row; amounts >= 0, some 0."""
    rng = np.random.default_rng(seed)
    keys = list(itertools.product(origins, horizons))
    chosen = keys + [keys[k] for k in rng.integers(0, len(keys), size=n - len(keys))]
    order = rng.permutation(n)
    origin = np.array([chosen[k][0] for k in order])
    h = np.array([chosen[k][1] for k in order])
    paid = rng.random(n) < 0.6
    amount = np.where(paid, rng.random(n) * 900 + 100, 0.0)
    frame = pd.DataFrame({"origin_week": origin, "h": h, "target_week": origin + h,
                          "company_code": rng.choice(["TR02", "TR04"], size=n),
                          "amount_deflated": amount})
    # Each (origin_week, h) total must be above 0 somewhere: give the first row of each one.
    for o, hh in keys:
        first = frame.index[(frame["origin_week"] == o) & (frame["h"] == hh)][0]
        frame.loc[first, "amount_deflated"] = 50.0 + o + hh
    if index is not None:
        frame.index = index
    return frame


def pair_data(seed=0, n=40, index=None, stage1=3, stage2=2):
    rows = pair_rows(seed, n, index=index)
    rng = np.random.default_rng(1000 + seed)
    ps = [pd.Series(rng.random(n), index=rows.index) for _ in range(stage1)]
    ms = [pd.Series(rng.random(n) * 500, index=rows.index) for _ in range(stage2)]
    return rows, ps, ms


def by_hand(rows, p, m):
    """total_scores of the pair, every total summed with a plain loop over the rows."""
    keys = sorted({(int(o), int(h)) for o, h in zip(rows["origin_week"], rows["h"])})
    actual, forecast = [], []
    for o, h in keys:
        a = f = 0.0
        for label in rows.index:
            if rows.at[label, "origin_week"] == o and rows.at[label, "h"] == h:
                a += rows.at[label, "amount_deflated"]
                f += p[label] * m[label]
        actual.append(a)
        forecast.append(f)
    return total_scores(pd.Series(actual), pd.Series(forecast))


def test_score_pairs_one_row_per_pair_with_its_columns():
    rows, ps, ms = pair_data()
    table = score_pairs(rows, ps, ms, "mape")
    assert list(table.columns) == ["stage1", "stage2", *SCORE_KEYS, "non-finite rows"]
    assert len(table) == 6
    # Every pair of a clean grid is scored: no non-finite row anywhere, an integer count.
    assert (table["non-finite rows"] == 0).all()
    assert pd.api.types.is_integer_dtype(table["non-finite rows"])
    assert np.isfinite(table[["MAPE", "bias", "WAPE"]].to_numpy()).all()
    assert sorted(zip(table["stage1"], table["stage2"])) == list(
        itertools.product(range(3), range(2)))
    assert table.index.tolist() == list(range(6))


def test_score_pairs_counts_one_total_per_origin_week_and_h():
    rows, ps, ms = pair_data()
    table = score_pairs(rows, ps, ms, "mape")
    # 3 origins x 2 horizons, all companies together.
    assert (table["totals"] == 6).all()
    assert (table["zero actual"] == 0).all()


@pytest.mark.parametrize("seed", range(6))
def test_score_pairs_equal_summing_p_times_m_per_total_by_hand(seed):
    rows, ps, ms = pair_data(seed)
    table = score_pairs(rows, ps, ms, "wape")
    for _, row in table.iterrows():
        expected = by_hand(rows, ps[int(row["stage1"])], ms[int(row["stage2"])])
        for key in SCORE_KEYS:
            assert row[key] == pytest.approx(expected[key], rel=1e-12), (row["stage1"],
                                                                         row["stage2"], key)


def test_score_pairs_by_hand_small_example():
    rows = pd.DataFrame({"origin_week": [40, 40, 40, 41], "h": [1, 1, 2, 1],
                         "target_week": [41, 41, 42, 42],
                         "amount_deflated": [100.0, 0.0, 50.0, 200.0]})
    p = pd.Series([0.5, 1.0, 1.0, 0.25])
    m = pd.Series([100.0, 60.0, 50.0, 400.0])
    table = score_pairs(rows, [p], [m], "mape")
    # (40, 1): actual 100, forecast 50 + 60 = 110 (10%); (40, 2): 50 vs 50 (0%);
    # (41, 1): 200 vs 100 (50%). MAPE 0.2; bias (260 - 350) / 350; WAPE (10 + 0 + 100) / 350.
    row = table.iloc[0]
    assert row["totals"] == 3 and row["zero actual"] == 0
    assert row["MAPE"] == pytest.approx(0.2)
    assert row["bias"] == pytest.approx(-90 / 350)
    assert row["WAPE"] == pytest.approx(110 / 350)


def test_score_pairs_companies_are_summed_together():
    rows = pd.DataFrame({"origin_week": [40, 40], "h": [1, 1], "target_week": [41, 41],
                         "company_code": ["TR02", "TR04"], "amount_deflated": [10.0, 30.0]})
    p, m = pd.Series([1.0, 1.0]), pd.Series([30.0, 10.0])
    table = score_pairs(rows, [p], [m], "mape")
    # Per company both are wrong; the consolidated total (40 vs 40) is right.
    assert table["totals"].iloc[0] == 1
    assert table["MAPE"].iloc[0] == pytest.approx(0.0)


def test_score_pairs_a_total_with_zero_actual_is_counted_and_left_out_of_the_mape():
    rows = pd.DataFrame({"origin_week": [40, 41], "h": [1, 1], "target_week": [41, 42],
                         "amount_deflated": [100.0, 0.0]})
    p, m = pd.Series([1.0, 1.0]), pd.Series([80.0, 30.0])
    row = score_pairs(rows, [p], [m], "mape").iloc[0]
    assert row["zero actual"] == 1
    assert row["MAPE"] == pytest.approx(0.2)
    assert row["WAPE"] == pytest.approx(0.5)


@pytest.mark.parametrize("metric, column", [("mape", "MAPE"), ("wape", "WAPE")])
@pytest.mark.parametrize("seed", range(3))
def test_score_pairs_sorted_by_the_metric_best_first(metric, column, seed):
    rows, ps, ms = pair_data(seed, stage1=4, stage2=3)
    table = score_pairs(rows, ps, ms, metric)
    assert table[column].is_monotonic_increasing
    assert table.index.tolist() == list(range(len(table)))


def test_score_pairs_ties_keep_the_grid_order():
    rows, ps, ms = pair_data(1)
    same = [ps[0], ps[0].copy(), ps[0].copy()]
    for metric in ("mape", "wape"):
        table = score_pairs(rows, same, [ms[0], ms[0].copy()], metric)
        assert list(zip(table["stage1"], table["stage2"])) == list(
            itertools.product(range(3), range(2)))


def test_score_pairs_ties_among_others_keep_the_grid_order():
    rows, ps, ms = pair_data(2)
    # Pairs (0, j) and (2, j) are identical; (1, j) differ. Within equal scores, i order.
    table = score_pairs(rows, [ps[0], ps[1], ps[0].copy()], ms, "mape")
    for j in range(2):
        positions = [k for k, (a, b) in enumerate(zip(table["stage1"], table["stage2"]))
                     if b == j and a in (0, 2)]
        assert [int(table["stage1"].iloc[k]) for k in positions] == [0, 2]
        assert positions[1] == positions[0] + 1


@pytest.mark.parametrize("metric", ["mape", "wape"])
def test_score_pairs_the_perfect_pair_wins(metric):
    rows, _, _ = pair_data(4)
    good_p = pd.Series(1.0, index=rows.index)
    good_m = rows["amount_deflated"].copy()
    table = score_pairs(rows, [good_p * 0.5, good_p, good_p * 0.9],
                        [good_m * 3, good_m + 1.0, good_m], metric)
    assert (table["stage1"].iloc[0], table["stage2"].iloc[0]) == (1, 2)
    assert table["MAPE"].iloc[0] == pytest.approx(0.0, abs=1e-15)
    assert table["WAPE"].iloc[0] == pytest.approx(0.0, abs=1e-15)
    assert table["bias"].iloc[0] == pytest.approx(0.0, abs=1e-15)


def test_score_pairs_one_setting_each():
    rows, ps, ms = pair_data(5, stage1=1, stage2=1)
    table = score_pairs(rows, ps, ms, "wape")
    assert table[["stage1", "stage2"]].values.tolist() == [[0, 0]]


@pytest.mark.parametrize("index", [
    pd.RangeIndex(100, 140),
    pd.Index(np.arange(40)[::-1] * 7 + 3),
    pd.Index([f"r{k}" for k in range(40)]),
])
def test_score_pairs_rows_with_a_non_default_index(index):
    rows, ps, ms = pair_data(6, index=index)
    table = score_pairs(rows, ps, ms, "mape")
    plain_rows, plain_ps, plain_ms = (rows.reset_index(drop=True),
                                      [p.reset_index(drop=True) for p in ps],
                                      [m.reset_index(drop=True) for m in ms])
    pd.testing.assert_frame_equal(table, score_pairs(plain_rows, plain_ps, plain_ms, "mape"))
    for _, row in table.iterrows():
        expected = by_hand(rows, ps[int(row["stage1"])], ms[int(row["stage2"])])
        assert row["MAPE"] == pytest.approx(expected["MAPE"], rel=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_score_pairs_shuffled_rows_give_the_same_table(seed):
    rows, ps, ms = pair_data(seed)
    order = np.random.default_rng(seed).permutation(len(rows))
    shuffled = rows.iloc[order]
    table = score_pairs(shuffled, [p.iloc[order] for p in ps], [m.iloc[order] for m in ms],
                        "mape")
    expected = score_pairs(rows, ps, ms, "mape")
    pd.testing.assert_frame_equal(table, expected, check_exact=False, rtol=1e-12)


def test_score_pairs_predictions_in_another_order_are_refused_not_misaligned():
    # Same labels, another order: p x m would align by label into a new order, which no longer
    # matches the rows; cash_totals refuses it rather than summing the wrong rows.
    rows, ps, ms = pair_data(7, index=pd.Index(np.arange(40)[::-1]))
    with pytest.raises(ValueError, match=r"^values must have the rows of rows \(the same "
                                         r"index\)$"):
        score_pairs(rows, ps, [ms[0].sort_index(), ms[1]], "mape")


def test_score_pairs_predictions_with_other_labels_are_refused():
    rows, ps, ms = pair_data(8)
    other = ps[0].copy()
    other.index = other.index + 1000
    with pytest.raises(ValueError):
        score_pairs(rows, [other], ms, "mape")


@pytest.mark.parametrize("metric", ["MAPE", "WAPE", "bias", "log loss", "", None])
def test_score_pairs_refuses_an_unknown_metric(metric):
    rows, ps, ms = pair_data()
    with pytest.raises(ValueError) as error:
        score_pairs(rows, ps, ms, metric)
    assert str(error.value) == f"cannot rank pairs by {metric!r}; choose 'mape' or 'wape'"


def test_score_pairs_checks_the_metric_before_anything_else():
    # Even with rows it could not total, an unknown metric is the error reported.
    rows = pd.DataFrame({"origin_week": [40], "h": [1], "target_week": [41]})
    with pytest.raises(ValueError, match="^cannot rank pairs by 'r2'"):
        score_pairs(rows, [pd.Series([1.0])], [pd.Series([1.0])], "r2")


# A NaN or infinite p x m on any row: the pair is not scored (infinite MAPE, bias and WAPE,
# ranked last, "non-finite rows" counts the rows) and the rest of the grid still runs.

def poisoned(series, positions, bad):
    """A copy of `series` with `bad` at the given positions."""
    series = series.copy()
    for k in positions:
        series.iloc[k] = bad
    return series


def without(items, k):
    return [x for i, x in enumerate(items) if i != k]


def scored_part(table):
    """The scored pairs of a score_pairs table, keyed by (stage1, stage2)."""
    good = table[table["non-finite rows"] == 0]
    return {(int(r["stage1"]), int(r["stage2"])): tuple(r[k] for k in SCORE_KEYS)
            for _, r in good.iterrows()}


@pytest.mark.parametrize("metric", ["mape", "wape"])
@pytest.mark.parametrize("bad", [nan, inf, -inf])
@pytest.mark.parametrize("which", ["p", "m"])
def test_score_pairs_a_non_finite_prediction_is_not_scored(bad, which, metric):
    rows, ps, ms = pair_data()
    if which == "p":
        ps[1] = poisoned(ps[1], [3], bad)
        bad_pairs = {(1, 0), (1, 1)}
    else:
        ms[1] = poisoned(ms[1], [3], bad)
        bad_pairs = {(0, 1), (1, 1), (2, 1)}
    table = score_pairs(rows, ps, ms, metric)
    assert len(table) == 6
    failed = table[table["non-finite rows"] > 0]
    assert set(zip(failed["stage1"], failed["stage2"])) == bad_pairs
    assert (failed["non-finite rows"] == 1).all()
    for column in ("MAPE", "bias", "WAPE"):
        assert (failed[column] == np.inf).all(), column  # +inf, never NaN or -inf
    # The totals are still counted: 3 origins x 2 horizons, none with a zero actual.
    assert (failed["totals"] == 6).all() and (failed["zero actual"] == 0).all()


@pytest.mark.parametrize("metric", ["mape", "wape"])
@pytest.mark.parametrize("which", ["p", "m"])
def test_score_pairs_non_finite_pairs_rank_last_in_grid_order(which, metric):
    rows, ps, ms = pair_data(3, stage1=4, stage2=3)
    if which == "p":
        ps[2] = poisoned(ps[2], [0], nan)
    else:
        ms[0] = poisoned(ms[0], [0], inf)
    table = score_pairs(rows, ps, ms, metric)
    flags = (table["non-finite rows"] > 0).tolist()
    n_bad = 3 if which == "p" else 4
    assert flags == [False] * (12 - n_bad) + [True] * n_bad
    # The scored part is still sorted best first, the failed tail keeps the grid order.
    assert table[metric.upper()].iloc[:12 - n_bad].is_monotonic_increasing
    tail = list(zip(table["stage1"].iloc[12 - n_bad:], table["stage2"].iloc[12 - n_bad:]))
    assert tail == sorted(tail)
    assert table.index.tolist() == list(range(12))


@pytest.mark.parametrize("metric", ["mape", "wape"])
@pytest.mark.parametrize("which", ["p", "m"])
@pytest.mark.parametrize("seed", range(3))
def test_score_pairs_a_non_finite_setting_leaves_the_other_pairs_scores_unchanged(
        which, metric, seed):
    rows, ps, ms = pair_data(seed, stage1=3, stage2=3)
    k = 1
    if which == "p":
        bad_ps, bad_ms = [*ps[:k], poisoned(ps[k], [5, 9], nan), *ps[k + 1:]], ms
        clean = score_pairs(rows, without(ps, k), ms, metric)
        remap = {(i, j): (i + (i >= k), j) for i, j in scored_part(clean)}
    else:
        bad_ps, bad_ms = ps, [*ms[:k], poisoned(ms[k], [5, 9], -inf), *ms[k + 1:]]
        clean = score_pairs(rows, ps, without(ms, k), metric)
        remap = {(i, j): (i, j + (j >= k)) for i, j in scored_part(clean)}
    table = score_pairs(rows, bad_ps, bad_ms, metric)
    expected = {remap[key]: value for key, value in scored_part(clean).items()}
    assert scored_part(table) == expected  # exactly the same numbers
    # And in the same order, with the failed pairs after them.
    n_good = len(expected)
    order = list(zip(table["stage1"].iloc[:n_good], table["stage2"].iloc[:n_good]))
    clean_order = [remap[(int(i), int(j))] for i, j in zip(clean["stage1"], clean["stage2"])]
    assert order == clean_order
    assert (table["non-finite rows"].iloc[n_good:] == 2).all()


def test_score_pairs_several_non_finite_settings():
    rows, ps, ms = pair_data(4, stage1=3, stage2=2)
    ps[0] = poisoned(ps[0], [2], nan)
    ms[1] = poisoned(ms[1], [7, 8], inf)
    table = score_pairs(rows, ps, ms, "mape")
    counts = {(int(i), int(j)): int(n) for i, j, n in
              zip(table["stage1"], table["stage2"], table["non-finite rows"])}
    # p0 bad on row 2; m1 bad on rows 7, 8: (0, 1) is bad on all three rows.
    assert counts == {(0, 0): 1, (0, 1): 3, (1, 0): 0, (1, 1): 2, (2, 0): 0, (2, 1): 2}
    assert list(zip(table["stage1"], table["stage2"]))[2:] == [(0, 0), (0, 1), (1, 1), (2, 1)]
    assert np.isfinite(table["MAPE"].iloc[:2]).all()
    assert (table["MAPE"].iloc[2:] == np.inf).all()


def test_score_pairs_a_row_bad_in_both_p_and_m_counts_once():
    rows, ps, ms = pair_data(5, stage1=1, stage2=1)
    table = score_pairs(rows, [poisoned(ps[0], [4], nan)], [poisoned(ms[0], [4], inf)],
                        "mape")
    assert table["non-finite rows"].tolist() == [1]


def test_score_pairs_counts_every_non_finite_row():
    rows, ps, ms = pair_data(6, stage1=1, stage2=1)
    table = score_pairs(rows, ps, [poisoned(ms[0], range(0, 40, 4), nan)], "wape")
    assert table["non-finite rows"].tolist() == [10]
    assert table["WAPE"].tolist() == [np.inf]


def test_score_pairs_every_pair_non_finite_keeps_the_grid_order():
    rows, ps, ms = pair_data(7, stage1=2, stage2=2)
    ms = [poisoned(m, [0], nan) for m in ms]
    for metric in ("mape", "wape"):
        table = score_pairs(rows, ps, ms, metric)
        assert list(zip(table["stage1"], table["stage2"])) == list(
            itertools.product(range(2), range(2)))
        assert (table["non-finite rows"] == 1).all()
        assert (table[["MAPE", "bias", "WAPE"]] == np.inf).all().all()


@pytest.mark.parametrize("p_value, m_value", [(0.0, inf), (inf, 0.0), (0.0, -inf),
                                              (-inf, 0.0)])
def test_score_pairs_zero_times_infinity_is_non_finite(p_value, m_value):
    # 0 x inf is NaN: the pair is not scored, even though one of the two is a plain 0.
    rows, ps, ms = pair_data(8, stage1=1, stage2=1)
    p, m = poisoned(ps[0], [6], p_value), poisoned(ms[0], [6], m_value)
    table = score_pairs(rows, [p], [m], "mape")
    assert table["non-finite rows"].tolist() == [1]
    assert table["MAPE"].tolist() == [np.inf]


def test_score_pairs_a_finite_p_and_m_whose_product_overflows_is_non_finite():
    rows, ps, ms = pair_data(9, stage1=1, stage2=1)
    p, m = poisoned(ps[0], [1], 1e200), poisoned(ms[0], [1], 1e200)
    assert np.isfinite(p).all() and np.isfinite(m).all()
    table = score_pairs(rows, [p], [m], "mape")
    assert table["non-finite rows"].tolist() == [1]
    assert table["bias"].tolist() == [np.inf]


@pytest.mark.parametrize("which", ["p", "m"])
def test_score_pairs_an_all_zero_finite_prediction_is_scored_normally(which):
    rows, ps, ms = pair_data(10, stage1=2, stage2=2)
    zero = pd.Series(0.0, index=rows.index)
    if which == "p":
        ps[1] = zero
        zero_pairs = {(1, 0), (1, 1)}
    else:
        ms[0] = zero
        zero_pairs = {(0, 0), (1, 0)}
    table = score_pairs(rows, ps, ms, "mape")
    assert (table["non-finite rows"] == 0).all()
    for _, row in table.iterrows():
        if (int(row["stage1"]), int(row["stage2"])) in zero_pairs:
            # A forecast of 0 everywhere: every error is the whole actual.
            assert row["MAPE"] == pytest.approx(1.0)
            assert row["bias"] == pytest.approx(-1.0)
            assert row["WAPE"] == pytest.approx(1.0)
    # Being finite, they rank by their score like any other pair: last only if worst.
    assert table["MAPE"].is_monotonic_increasing


def test_score_pairs_an_all_zero_forecast_beats_a_non_finite_one():
    rows, ps, ms = pair_data(11, stage1=1, stage2=2)
    zero = pd.Series(0.0, index=rows.index)
    table = score_pairs(rows, ps, [poisoned(ms[0], [0], nan), zero], "mape")
    assert list(zip(table["stage1"], table["stage2"])) == [(0, 1), (0, 0)]
    assert table["non-finite rows"].tolist() == [0, 1]


def test_score_pairs_a_non_finite_pair_counts_the_zero_actual_totals_like_a_scored_one():
    rows = pd.DataFrame({"origin_week": [40, 41], "h": [1, 1], "target_week": [41, 42],
                         "amount_deflated": [100.0, 0.0]})
    p = pd.Series([1.0, 1.0])
    table = score_pairs(rows, [p], [pd.Series([80.0, 30.0]), pd.Series([nan, 30.0])],
                        "mape")
    assert table["totals"].tolist() == [2, 2]
    assert table["zero actual"].tolist() == [1, 1]
    assert table["non-finite rows"].tolist() == [0, 1]


def test_score_pairs_int_predictions_are_scored():
    rows, _, _ = pair_data(12)
    p = pd.Series(1, index=rows.index, dtype="int64")
    m = pd.Series(np.arange(40), index=rows.index, dtype="int64")
    table = score_pairs(rows, [p], [m], "mape")
    assert table["non-finite rows"].tolist() == [0]
    expected = by_hand(rows, p.astype(float), m.astype(float))
    assert table["MAPE"].iloc[0] == pytest.approx(expected["MAPE"], rel=1e-12)


# The index check comes before any pair is formed: misaligned series are refused, never
# turned into a NaN pair.

@pytest.mark.parametrize("which", ["p", "m"])
@pytest.mark.parametrize("change", ["shifted", "shorter", "longer", "reordered"])
def test_score_pairs_refuses_a_series_without_the_rows_index(which, change):
    rows, ps, ms = pair_data(13)
    target = ps[2] if which == "p" else ms[1]
    if change == "shifted":
        target = target.set_axis(target.index + 1)
    elif change == "shorter":
        target = target.iloc[:-1]
    elif change == "longer":
        target = pd.concat([target, pd.Series([1.0], index=[999])])
    else:
        target = target.iloc[::-1]
    if which == "p":
        ps[2] = target
    else:
        ms[1] = target
    with pytest.raises(ValueError, match=r"^values must have the rows of rows \(the same "
                                         r"index\)$"):
        score_pairs(rows, ps, ms, "mape")


def test_score_pairs_index_check_comes_before_the_non_finite_handling():
    # A grid holding both a NaN setting and a misaligned one is refused, not partly scored.
    rows, ps, ms = pair_data(14)
    ms[0] = poisoned(ms[0], [0], nan)
    ps[1] = ps[1].set_axis(ps[1].index + 1000)
    with pytest.raises(ValueError, match="^values must have the rows of rows"):
        score_pairs(rows, ps, ms, "mape")


def test_score_pairs_unknown_metric_is_reported_before_the_index_check():
    rows, ps, ms = pair_data(15)
    ps[0] = ps[0].set_axis(ps[0].index + 1000)
    with pytest.raises(ValueError, match="^cannot rank pairs by 'r2'"):
        score_pairs(rows, ps, ms, "r2")


def test_score_pairs_with_a_non_finite_setting_does_not_change_its_inputs():
    rows, ps, ms = pair_data(16)
    ms[1] = poisoned(ms[1], [3], nan)
    copies = rows.copy(deep=True), [p.copy() for p in ps], [m.copy() for m in ms]
    score_pairs(rows, ps, ms, "mape")
    pd.testing.assert_frame_equal(rows, copies[0])
    for a, b in zip(ps + ms, copies[1] + copies[2]):
        pd.testing.assert_series_equal(a, b)


def test_score_pairs_refuses_a_nan_actual_amount():
    rows, ps, ms = pair_data()
    rows.loc[rows.index[0], "amount_deflated"] = nan
    with pytest.raises(ValueError, match="^values holds a NaN or infinite value$"):
        score_pairs(rows, ps, ms, "mape")


def test_score_pairs_refuses_actual_totals_all_zero_through_total_scores():
    rows, ps, ms = pair_data()
    rows["amount_deflated"] = 0.0
    with pytest.raises(ValueError, match="every actual total is 0"):
        score_pairs(rows, ps, ms, "mape")


def test_score_pairs_refuses_a_negative_actual_total_through_total_scores():
    rows, ps, ms = pair_data()
    rows["amount_deflated"] = -1.0
    with pytest.raises(ValueError, match="an actual total is below 0"):
        score_pairs(rows, ps, ms, "mape")


def test_score_pairs_accepts_a_zero_forecast():
    rows, _, _ = pair_data()
    zero = pd.Series(0.0, index=rows.index)
    row = score_pairs(rows, [zero], [zero + 5.0], "mape").iloc[0]
    assert row["MAPE"] == pytest.approx(1.0)
    assert row["bias"] == pytest.approx(-1.0)


def test_score_pairs_does_not_change_its_inputs():
    rows, ps, ms = pair_data()
    copies = rows.copy(deep=True), [p.copy() for p in ps], [m.copy() for m in ms]
    lists = list(ps), list(ms)
    score_pairs(rows, ps, ms, "mape")
    pd.testing.assert_frame_equal(rows, copies[0])
    for a, b in zip(ps + ms, copies[1] + copies[2]):
        pd.testing.assert_series_equal(a, b)
    assert all(a is b for a, b in zip(ps, lists[0])) and len(ps) == len(lists[0])
    assert all(a is b for a, b in zip(ms, lists[1])) and len(ms) == len(lists[1])


def test_score_pairs_is_deterministic():
    rows, ps, ms = pair_data(9)
    pd.testing.assert_frame_equal(score_pairs(rows, ps, ms, "wape"),
                                  score_pairs(rows, ps, ms, "wape"))


# check_actual_totals: the actual-total faults, checked on their own, by total_scores and once
# by score_pairs before any pair (so they cannot hide behind "every pair is non-finite").

MSG_EMPTY = "there are no totals to score"
MSG_NONFINITE = "actual holds a NaN or infinite value"
MSG_NEGATIVE = "an actual total is below 0; the totals are cash received"
MSG_ALL_ZERO = "every actual total is 0: no error relative to the cash exists"


def raises_exactly(message, call, *args):
    with pytest.raises(ValueError) as error:
        call(*args)
    assert str(error.value) == message


def test_check_actual_totals_refuses_empty():
    raises_exactly(MSG_EMPTY, check_actual_totals, pd.Series([], dtype=float))


@pytest.mark.parametrize("bad", [nan, inf, -inf])
def test_check_actual_totals_refuses_nan_and_infinite(bad):
    raises_exactly(MSG_NONFINITE, check_actual_totals, pd.Series([1.0, bad, 2.0]))


def test_check_actual_totals_refuses_a_none_in_an_object_series():
    raises_exactly(MSG_NONFINITE, check_actual_totals, pd.Series([1.0, None], dtype=object))


@pytest.mark.parametrize("values", [[-0.5], [1.0, -1e-12], [-3.0, -1.0]])
def test_check_actual_totals_refuses_a_negative_total(values):
    raises_exactly(MSG_NEGATIVE, check_actual_totals, pd.Series(values))


@pytest.mark.parametrize("values", [[0.0], [0.0, 0.0, 0.0], [0, 0]])
def test_check_actual_totals_refuses_all_zero(values):
    raises_exactly(MSG_ALL_ZERO, check_actual_totals, pd.Series(values))


def test_check_actual_totals_negative_is_reported_even_when_the_sum_is_zero():
    # -1 + 1 = 0: the negative check comes first, so it is not mistaken for "all zero".
    raises_exactly(MSG_NEGATIVE, check_actual_totals, pd.Series([-1.0, 1.0]))


def test_check_actual_totals_nan_is_reported_before_a_negative():
    raises_exactly(MSG_NONFINITE, check_actual_totals, pd.Series([-1.0, nan]))


def test_check_actual_totals_minus_zero_is_not_below_zero():
    assert check_actual_totals(pd.Series([-0.0, 5.0])) is None


@pytest.mark.parametrize("values", [
    [0.0, 3.0, 0.0], [1.0], [0, 7], [1e-300, 0.0], [0.0] * 9 + [0.01],
], ids=["zeros-around", "single", "ints", "tiny", "one-nonzero"])
def test_check_actual_totals_accepts_valid_totals_with_some_zeros(values):
    assert check_actual_totals(pd.Series(values)) is None


def test_check_actual_totals_accepts_a_non_default_index():
    assert check_actual_totals(pd.Series([0.0, 2.0], index=["b", "a"])) is None


def test_check_actual_totals_does_not_change_its_input():
    actual = pd.Series([0.0, 2.0, 0.0], index=[5, 3, 1])
    copy = actual.copy()
    check_actual_totals(actual)
    pd.testing.assert_series_equal(actual, copy)


# total_scores: the order of the checks is index, actual totals, then the forecast.

@pytest.mark.parametrize("actual, message", [
    ([1.0, nan], MSG_NONFINITE),
    ([1.0, inf], MSG_NONFINITE),
    ([1.0, -2.0], MSG_NEGATIVE),
    ([0.0, 0.0], MSG_ALL_ZERO),
], ids=["nan", "inf", "negative", "all-zero"])
@pytest.mark.parametrize("bad_forecast", [nan, inf, -inf])
def test_total_scores_reports_a_bad_actual_before_a_bad_forecast(actual, message,
                                                                  bad_forecast):
    raises_exactly(message, total_scores, pd.Series(actual), pd.Series([bad_forecast, 1.0]))


def test_total_scores_empty_actual_and_forecast_reports_no_totals():
    raises_exactly(MSG_EMPTY, total_scores, pd.Series([], dtype=float),
                   pd.Series([], dtype=float))


def test_total_scores_index_check_still_comes_before_a_bad_actual():
    raises_exactly("forecast and actual must have the same totals (the same index)",
                   total_scores, pd.Series([-1.0, nan]), pd.Series([nan], index=[7]))


def test_total_scores_a_good_actual_with_a_bad_forecast_reports_the_forecast():
    raises_exactly("forecast holds a NaN or infinite value", total_scores,
                   pd.Series([0.0, 2.0]), pd.Series([1.0, nan]))


# score_pairs: the actual totals are checked once, before the loop.

def every_pair_non_finite(rows, ps, ms):
    """The grid of ps x ms with every m poisoned, so no pair would reach total_scores."""
    return ps, [poisoned(m, [0], nan) for m in ms]


@pytest.mark.parametrize("metric", ["mape", "wape"])
def test_score_pairs_every_pair_non_finite_with_a_negative_actual_raises(metric):
    rows, ps, ms = pair_data(20, stage1=2, stage2=2)
    rows.loc[rows.index[0], "amount_deflated"] = -1e6  # its total goes below 0
    ps, ms = every_pair_non_finite(rows, ps, ms)
    raises_exactly(MSG_NEGATIVE, score_pairs, rows, ps, ms, metric)


@pytest.mark.parametrize("metric", ["mape", "wape"])
def test_score_pairs_every_pair_non_finite_with_all_zero_actuals_raises(metric):
    rows, ps, ms = pair_data(21, stage1=2, stage2=2)
    rows["amount_deflated"] = 0.0
    ps, ms = every_pair_non_finite(rows, ps, ms)
    raises_exactly(MSG_ALL_ZERO, score_pairs, rows, ps, ms, metric)


@pytest.mark.parametrize("bad", ["p", "m", "overflow"])
def test_score_pairs_single_non_finite_pair_with_a_negative_actual_raises(bad):
    rows, ps, ms = pair_data(22, stage1=1, stage2=1)
    rows["amount_deflated"] = -1.0
    p, m = ps[0], ms[0]
    if bad == "p":
        p = poisoned(p, [3], inf)
    elif bad == "m":
        m = poisoned(m, [3], -inf)
    else:
        p, m = poisoned(p, [3], 1e200), poisoned(m, [3], 1e200)
    raises_exactly(MSG_NEGATIVE, score_pairs, rows, [p], [m], "mape")


def test_score_pairs_a_negative_row_whose_total_stays_positive_is_scored():
    # The check is on the totals, not the rows: a refund row inside a positive total is fine.
    rows, ps, ms = pair_data(23, stage1=1, stage2=1)
    first = rows.index[0]
    key = rows.loc[first, ["origin_week", "h"]]
    same = rows.index[(rows["origin_week"] == key["origin_week"]) & (rows["h"] == key["h"])]
    rows.loc[same, "amount_deflated"] = 10.0
    rows.loc[first, "amount_deflated"] = -1.0
    assert len(same) > 1
    table = score_pairs(rows, ps, ms, "mape")
    assert table["non-finite rows"].tolist() == [0]
    assert np.isfinite(table["MAPE"]).all()


@pytest.mark.parametrize("stage1, stage2", [(0, 2), (2, 0), (0, 0)])
def test_score_pairs_an_empty_grid_still_refuses_bad_actuals(stage1, stage2):
    rows, ps, ms = pair_data(24, stage1=stage1, stage2=stage2)
    rows["amount_deflated"] = 0.0
    raises_exactly(MSG_ALL_ZERO, score_pairs, rows, ps, ms, "mape")


def test_score_pairs_no_rows_reports_no_totals():
    rows = pair_rows(25).iloc[0:0]
    empty = pd.Series([], dtype=float, index=rows.index)
    raises_exactly(MSG_EMPTY, score_pairs, rows, [empty], [empty], "mape")


def test_score_pairs_metric_check_comes_before_the_actual_totals_check():
    rows, ps, ms = pair_data(26)
    rows["amount_deflated"] = 0.0
    with pytest.raises(ValueError, match="^cannot rank pairs by 'r2'"):
        score_pairs(rows, ps, ms, "r2")


def test_score_pairs_index_check_comes_before_the_actual_totals_check():
    rows, ps, ms = pair_data(27)
    rows["amount_deflated"] = -1.0
    ps[0] = ps[0].set_axis(ps[0].index + 1000)
    with pytest.raises(ValueError, match="^values must have the rows of rows"):
        score_pairs(rows, ps, ms, "mape")


def test_score_pairs_valid_actuals_with_a_zero_total_and_every_pair_non_finite_is_a_table():
    rows = pd.DataFrame({"origin_week": [40, 41], "h": [1, 1], "target_week": [41, 42],
                         "amount_deflated": [100.0, 0.0]})
    table = score_pairs(rows, [pd.Series([1.0, 1.0])], [pd.Series([nan, 1.0])], "mape")
    assert table["non-finite rows"].tolist() == [1]
    assert table["zero actual"].tolist() == [1]
    assert table["MAPE"].tolist() == [np.inf]


def test_total_codes_was_removed():
    from collection_estimation.hurdle import diagnostic_joint
    assert not hasattr(diagnostic_joint, "total_codes")


# --- paired_bootstrap ------------------------------------------------------------------------

def bootstrap_data(seed=0, n=30):
    rng = np.random.default_rng(seed)
    actual = pd.Series(rng.random(n) * 1000 + 100, index=pd.RangeIndex(10, 10 + n))
    a = actual * (1 + rng.normal(scale=0.1, size=n))
    b = actual * (1 + rng.normal(scale=0.3, size=n))
    return actual, a, b


def test_paired_bootstrap_keys():
    actual, a, b = bootstrap_data()
    result = paired_bootstrap(actual, a, b, 100, 1)
    assert list(result) == ["mean difference", "low", "high", "A better"]
    assert all(type(v) is float for v in result.values())


@pytest.mark.parametrize("seed", range(6))
def test_paired_bootstrap_mean_difference_is_mape_a_minus_mape_b(seed):
    actual, a, b = bootstrap_data(seed)
    result = paired_bootstrap(actual, a, b, 50, seed)
    expected = total_scores(actual, a)["MAPE"] - total_scores(actual, b)["MAPE"]
    assert result["mean difference"] == pytest.approx(expected, abs=1e-12)


def test_paired_bootstrap_by_hand():
    actual = pd.Series([100.0, 200.0])
    a = pd.Series([110.0, 200.0])      # APE 0.1, 0
    b = pd.Series([100.0, 300.0])      # APE 0, 0.5
    result = paired_bootstrap(actual, a, b, 10, 0)
    assert result["mean difference"] == pytest.approx((0.1 - 0.5) / 2)
    assert result["A better"] == 0.5
    # Every resample mean lies between the two differences 0.1 and -0.5.
    assert -0.5 - 1e-12 <= result["low"] <= result["high"] <= 0.1 + 1e-12


def test_paired_bootstrap_is_reproducible_with_the_seed():
    actual, a, b = bootstrap_data(2)
    assert paired_bootstrap(actual, a, b, 500, 42) == paired_bootstrap(actual, a, b, 500, 42)


def test_paired_bootstrap_another_seed_gives_another_interval():
    actual, a, b = bootstrap_data(2)
    one, two = paired_bootstrap(actual, a, b, 500, 1), paired_bootstrap(actual, a, b, 500, 2)
    assert one["mean difference"] == two["mean difference"]
    assert (one["low"], one["high"]) != (two["low"], two["high"])


def test_paired_bootstrap_interval_brackets_the_mean_for_a_clear_winner():
    actual, a, b = bootstrap_data(4, n=200)
    result = paired_bootstrap(actual, a, b, 2000, 0)
    assert result["low"] <= result["mean difference"] <= result["high"]
    assert result["high"] < 0       # A (10% noise) is clearly better than B (30%)
    assert result["A better"] > 0.5


def test_paired_bootstrap_swapping_a_and_b_mirrors_the_result():
    actual, a, b = bootstrap_data(5)
    ab, ba = paired_bootstrap(actual, a, b, 300, 9), paired_bootstrap(actual, b, a, 300, 9)
    assert ab["mean difference"] == pytest.approx(-ba["mean difference"])
    assert ab["low"] == pytest.approx(-ba["high"])
    assert ab["high"] == pytest.approx(-ba["low"])


def test_paired_bootstrap_identical_forecasts_give_zero_and_no_a_better():
    actual, a, _ = bootstrap_data(1)
    result = paired_bootstrap(actual, a, a.copy(), 100, 0)
    assert result == {"mean difference": 0.0, "low": 0.0, "high": 0.0, "A better": 0.0}


def test_paired_bootstrap_one_sample_works():
    actual, a, b = bootstrap_data(1)
    result = paired_bootstrap(actual, a, b, 1, 0)
    assert result["low"] == pytest.approx(result["high"])


def test_paired_bootstrap_a_better_counts_strict_wins_only():
    actual = pd.Series([10.0, 10.0, 10.0, 10.0])
    a = pd.Series([10.0, 11.0, 12.0, 9.0])     # APE 0, .1, .2, .1
    b = pd.Series([11.0, 11.0, 10.0, 8.0])     # APE .1, .1, 0, .2
    assert paired_bootstrap(actual, a, b, 10, 0)["A better"] == 0.5


@pytest.mark.parametrize("which", ["forecast_a", "forecast_b"])
def test_paired_bootstrap_refuses_misaligned_forecasts(which):
    actual, a, b = bootstrap_data()
    series = {"forecast_a": a, "forecast_b": b}
    series[which] = series[which].reset_index(drop=True)
    with pytest.raises(ValueError) as error:
        paired_bootstrap(actual, series["forecast_a"], series["forecast_b"], 10, 0)
    assert str(error.value) == f"{which} must have the totals of actual (the same index)"


@pytest.mark.parametrize("bad", [0.0, -1.0, nan])
def test_paired_bootstrap_refuses_an_actual_not_above_zero(bad):
    actual, a, b = bootstrap_data()
    actual = actual.copy()
    actual.iloc[3] = bad
    with pytest.raises(ValueError) as error:
        paired_bootstrap(actual, a, b, 10, 0)
    assert str(error.value) == "every actual total must be above 0 for a percentage error"


@pytest.mark.parametrize("samples", [0, -5])
def test_paired_bootstrap_refuses_fewer_than_one_sample(samples):
    actual, a, b = bootstrap_data()
    with pytest.raises(ValueError) as error:
        paired_bootstrap(actual, a, b, samples, 0)
    assert str(error.value) == f"samples is {samples}; at least 1 is needed"


def test_paired_bootstrap_does_not_change_its_inputs():
    actual, a, b = bootstrap_data()
    copies = actual.copy(), a.copy(), b.copy()
    paired_bootstrap(actual, a, b, 50, 0)
    pd.testing.assert_series_equal(actual, copies[0])
    pd.testing.assert_series_equal(a, copies[1])
    pd.testing.assert_series_equal(b, copies[2])


# --- paired_bootstrap: FINDINGS (fail against the current source) --------------------------------
# total_scores refuses NaN/inf forecasts and an empty set of totals; paired_bootstrap does not.
# With a NaN forecast it returns NaN for the mean and the interval but "A better" = 0.0, a
# plausible-looking number; with no totals it returns all NaN (plus numpy RuntimeWarnings).
# Neither case can arise from step 15 today (the forecasts are sums of finite p x m and the
# consolidated totals are never empty), so this is a missing guard, not a wrong result in the
# pipeline. These tests state the behaviour its sibling total_scores already has.

@pytest.mark.parametrize("bad", [nan, inf])
def test_paired_bootstrap_refuses_a_nan_or_infinite_forecast(bad):
    actual = pd.Series([100.0, 200.0, 300.0])
    good = pd.Series([110.0, 190.0, 300.0])
    broken = pd.Series([bad, 190.0, 300.0])
    with pytest.raises(ValueError):
        paired_bootstrap(actual, broken, good, 10, 0)


def test_paired_bootstrap_refuses_no_totals():
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        paired_bootstrap(empty, empty.copy(), empty.copy(), 10, 0)


# --- the eligibility gates (U11.5b) ---------------------------------------------------------------
# Each setting of a grid is checked on its own on validation before any pair is scored.
# stage1_gates: log loss within (1 + tolerance) of the grid's best, log-loss skill above 0,
# calibration slope inside the band (ends included). stage2_gates: every m finite and above 0,
# deviance skill within `tolerance` of the best m-gate passer, total ratio inside the band
# (ends included).

from sklearn.metrics import log_loss  # noqa: E402

from collection_estimation.hurdle.diagnostic import calibration_slope_intercept  # noqa: E402
from collection_estimation.hurdle.diagnostic_joint import (  # noqa: E402
    stage1_gates,
    stage2_gates,
)
from collection_estimation.hurdle.stage2 import amount_summary  # noqa: E402

STAGE1_GATE_COLUMNS = ["stage1", "bad p", "log loss", "log loss skill", "slope",
                       "log loss gate", "skill gate", "slope gate", "eligible"]
STAGE2_GATE_COLUMNS = ["stage2", "bad m", "deviance skill", "total ratio", "m gate",
                       "skill gate", "ratio gate", "eligible"]
S1_GATES = ["log loss gate", "skill gate", "slope gate", "eligible"]
S2_GATES = ["m gate", "skill gate", "ratio gate", "eligible"]


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def stage1_data():
    """4000 validation rows whose outcome follows p = sigmoid(z), and five stage-1 settings
    built from z. Measured on these rows (seed 0):

        setting   log loss   / true's   slope
        true      0.5240     1           1.01   (the right p)
        shift     0.5375     1.026       1.01   (every logit + 0.4: worse, slope still ~1)
        under     0.5585     1.066       2.02   (too timid)
        over      0.5937     1.133       0.51   (too confident)
        anti      1.326      2.53       -1.01   (backwards: worse than the base rate)
    """
    rng = np.random.default_rng(0)
    index = pd.RangeIndex(100, 4100, name="row")  # not 0..n-1, so rows are matched by label
    z = pd.Series(rng.normal(0, 1.5, len(index)), index=index)
    paid = pd.Series(rng.random(len(index)) < sigmoid(z), index=index)
    base_rate = pd.Series(paid.mean(), index=index)
    p = {"true": sigmoid(z), "shift": sigmoid(z + 0.4), "under": sigmoid(0.5 * z),
         "over": sigmoid(2 * z), "anti": sigmoid(-z)}
    return paid, p, base_rate


def gates1(names, tolerance=0.10, band=(0.8, 1.25), base_rate=None):
    """stage1_gates on the named settings of stage1_data, indexed by name."""
    paid, p, default_base = stage1_data()
    table = stage1_gates(paid, [p[name] for name in names],
                         default_base if base_rate is None else base_rate, tolerance, band)
    table.insert(0, "name", list(names))
    return table.set_index("name")


def gate_flags(row, columns):
    return {column: bool(row[column]) for column in columns}


# stage1_gates: shape and values

def test_stage1_gates_one_row_per_setting_numbered_in_grid_order():
    paid, p, base_rate = stage1_data()
    table = stage1_gates(paid, [p["true"], p["shift"], p["under"]], base_rate, 0.1, (0.8, 1.25))
    assert list(table.columns) == STAGE1_GATE_COLUMNS
    assert table["stage1"].tolist() == [0, 1, 2]
    assert list(table.index) == [0, 1, 2]
    for column in S1_GATES:
        assert table[column].dtype == bool, column


def test_stage1_gates_measures_log_loss_skill_and_slope_like_the_diagnostic():
    paid, p, base_rate = stage1_data()
    table = stage1_gates(paid, [p["true"], p["over"]], base_rate, 0.1, (0.8, 1.25))
    base_loss = log_loss(paid, base_rate, labels=[False, True])
    for i, name in enumerate(("true", "over")):
        loss = log_loss(paid, p[name], labels=[False, True])
        assert table.loc[i, "log loss"] == pytest.approx(loss, rel=1e-12)
        assert table.loc[i, "log loss skill"] == pytest.approx(1 - loss / base_loss, rel=1e-12)
        slope, _ = calibration_slope_intercept(paid, p[name])
        assert table.loc[i, "slope"] == pytest.approx(slope, rel=1e-9)


def test_stage1_gates_the_right_p_is_eligible():
    row = gates1(["true"]).loc["true"]
    assert gate_flags(row, S1_GATES) == {"log loss gate": True, "skill gate": True,
                                         "slope gate": True, "eligible": True}


# stage1_gates: each gate alone

def test_stage1_log_loss_gate_alone_fails_a_setting_too_far_from_the_best():
    # shift's log loss is 2.6% above true's: outside a 1% tolerance, while its skill is above
    # 0 and its slope inside the band.
    row = gates1(["true", "shift"], tolerance=0.01).loc["shift"]
    assert gate_flags(row, S1_GATES) == {"log loss gate": False, "skill gate": True,
                                         "slope gate": True, "eligible": False}


def test_stage1_log_loss_gate_passes_the_same_setting_with_a_wider_tolerance():
    row = gates1(["true", "shift"], tolerance=0.05).loc["shift"]
    assert gate_flags(row, S1_GATES) == {"log loss gate": True, "skill gate": True,
                                         "slope gate": True, "eligible": True}


def test_stage1_skill_gate_alone_fails_a_skill_of_exactly_zero():
    # With the right p itself as the base rate, the right p's skill is exactly 0: not above 0.
    _, p, _ = stage1_data()
    row = gates1(["true"], base_rate=p["true"]).loc["true"]
    assert row["log loss skill"] == 0.0
    assert gate_flags(row, S1_GATES) == {"log loss gate": True, "skill gate": False,
                                         "slope gate": True, "eligible": False}


def test_stage1_skill_gate_alone_fails_a_setting_worse_than_the_base_rate():
    # shift judged against the right p as the "base rate": negative skill; alone in its grid
    # (so the best log loss); slope 1.01.
    _, p, _ = stage1_data()
    row = gates1(["shift"], base_rate=p["true"]).loc["shift"]
    assert row["log loss skill"] < 0
    assert gate_flags(row, S1_GATES) == {"log loss gate": True, "skill gate": False,
                                         "slope gate": True, "eligible": False}


def test_stage1_slope_gate_alone_fails_a_slope_above_the_band():
    row = gates1(["true", "under"], tolerance=0.10).loc["under"]
    assert row["slope"] > 1.25
    assert gate_flags(row, S1_GATES) == {"log loss gate": True, "skill gate": True,
                                         "slope gate": False, "eligible": False}


def test_stage1_slope_gate_alone_fails_a_slope_below_the_band():
    row = gates1(["true", "over"], tolerance=0.20).loc["over"]
    assert row["slope"] < 0.8
    assert gate_flags(row, S1_GATES) == {"log loss gate": True, "skill gate": True,
                                         "slope gate": False, "eligible": False}


def test_stage1_a_backwards_setting_fails_skill_and_slope():
    row = gates1(["true", "anti"], tolerance=10.0).loc["anti"]
    assert row["log loss skill"] < 0 and row["slope"] < 0
    assert gate_flags(row, S1_GATES) == {"log loss gate": True, "skill gate": False,
                                         "slope gate": False, "eligible": False}


# stage1_gates: ends and tolerance

def test_stage1_slope_band_low_end_is_included():
    slope = gates1(["true"])["slope"].iloc[0]
    assert gates1(["true"], band=(slope, 2.0))["slope gate"].iloc[0]
    assert not gates1(["true"], band=(np.nextafter(slope, np.inf), 2.0))["slope gate"].iloc[0]


def test_stage1_slope_band_high_end_is_included():
    slope = gates1(["true"])["slope"].iloc[0]
    assert gates1(["true"], band=(0.5, slope))["slope gate"].iloc[0]
    assert not gates1(["true"], band=(0.5, np.nextafter(slope, -np.inf)))["slope gate"].iloc[0]


def test_stage1_a_setting_tied_with_the_best_passes_any_tolerance():
    paid, p, base_rate = stage1_data()
    table = stage1_gates(paid, [p["true"], p["true"].copy()], base_rate, 1e-12, (0.8, 1.25))
    assert table["log loss gate"].tolist() == [True, True]


def test_stage1_log_loss_tolerance_just_below_and_just_above_the_needed_one():
    table = gates1(["true", "shift"])
    needed = table.loc["shift", "log loss"] / table.loc["true", "log loss"] - 1
    assert not gates1(["true", "shift"], tolerance=needed * 0.999).loc["shift", "log loss gate"]
    assert gates1(["true", "shift"], tolerance=needed * 1.001).loc["shift", "log loss gate"]


def test_stage1_tolerance_is_relative_to_the_grids_best_not_a_running_best():
    # shift comes first: a running best would let it pass (it is the best "so far").
    assert not gates1(["shift", "true"], tolerance=0.01).loc["shift", "log loss gate"]
    # Alone, shift is its grid's best.
    assert gates1(["shift"], tolerance=0.01).loc["shift", "log loss gate"]


def test_stage1_the_order_of_the_settings_does_not_change_any_settings_result():
    names = ["true", "shift", "under", "over", "anti"]
    reference = gates1(names, tolerance=0.05).drop(columns="stage1")
    for order in (names[::-1], ["under", "anti", "true", "over", "shift"]):
        table = gates1(order, tolerance=0.05)
        assert table["stage1"].tolist() == list(range(len(order)))
        pd.testing.assert_frame_equal(table.drop(columns="stage1").loc[names], reference,
                                      rtol=1e-9)


def test_stage1_a_bad_setting_does_not_move_the_others_gates():
    # anti's huge log loss is not the best, so it cannot loosen or tighten the others' gates.
    with_anti = gates1(["true", "shift", "anti"], tolerance=0.01)
    without = gates1(["true", "shift"], tolerance=0.01)
    pd.testing.assert_frame_equal(with_anti.loc[["true", "shift"], S1_GATES],
                                  without[S1_GATES])


def test_stage1_gates_accepts_a_nullable_boolean_paid_column():
    # The pipeline passes validation["paid"], a pandas "boolean" column.
    paid, p, base_rate = stage1_data()
    plain = stage1_gates(paid, [p["true"], p["shift"]], base_rate, 0.1, (0.8, 1.25))
    nullable = stage1_gates(paid.astype("boolean"), [p["true"], p["shift"]], base_rate, 0.1,
                            (0.8, 1.25))
    pd.testing.assert_frame_equal(plain, nullable)


def test_stage1_gates_does_not_change_its_inputs():
    paid, p, base_rate = stage1_data()
    copies = paid.copy(), p["true"].copy(), base_rate.copy()
    stage1_gates(paid, [p["true"]], base_rate, 0.1, (0.8, 1.25))
    pd.testing.assert_series_equal(paid, copies[0])
    pd.testing.assert_series_equal(p["true"], copies[1])
    pd.testing.assert_series_equal(base_rate, copies[2])


# stage1_gates: errors

def test_stage1_gates_refuses_an_empty_grid():
    paid, _, base_rate = stage1_data()
    with pytest.raises(ValueError, match="^there are no stage-1 settings to check$"):
        stage1_gates(paid, [], base_rate, 0.1, (0.8, 1.25))


def test_stage1_gates_refuses_a_p_with_other_rows():
    paid, p, base_rate = stage1_data()
    shifted = p["true"].set_axis(p["true"].index + 1)
    with pytest.raises(ValueError,
                       match=r"^actual and p must have the same rows \(the same index\)$"):
        stage1_gates(paid, [p["true"], shifted], base_rate, 0.1, (0.8, 1.25))


def test_stage1_gates_refuses_a_p_with_the_rows_in_another_order():
    paid, p, base_rate = stage1_data()
    with pytest.raises(ValueError, match="must have the same rows"):
        stage1_gates(paid, [p["true"].iloc[::-1]], base_rate, 0.1, (0.8, 1.25))


def test_stage1_gates_refuses_a_base_rate_with_other_rows():
    paid, p, base_rate = stage1_data()
    with pytest.raises(ValueError, match="must have the same rows"):
        stage1_gates(paid, [p["true"]], base_rate.iloc[:-1], 0.1, (0.8, 1.25))


def test_stage1_gates_refuses_a_single_outcome():
    paid = pd.Series([True] * 4)
    p = pd.Series([0.5, 0.6, 0.7, 0.8])
    with pytest.raises(ValueError,
                       match=r"^actual must hold both outcomes \(payers and non-payers\)$"):
        stage1_gates(paid, [p], p.copy(), 0.1, (0.8, 1.25))


def test_stage1_gates_refuses_no_rows():
    empty_paid = pd.Series([], dtype=bool)
    empty_p = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="^there are no rows to measure$"):
        stage1_gates(empty_paid, [empty_p], empty_p.copy(), 0.1, (0.8, 1.25))


# stage1_gates: a bad p (not a finite number in [0, 1]) fails only its own setting. Fixed
# after the U11.5b test report: a NaN p used to raise scikit-learn's "Input contains NaN."
# from inside log_loss and stop all of step 15.

S1_SCORES = ["log loss", "log loss skill", "slope"]
BAD_P_VALUES = [nan, inf, -inf, -1e-9, np.nextafter(0.0, -1.0), 1 + 1e-9,
                np.nextafter(1.0, 2.0), -1.0, 2.0]


def with_bad_p(series, positions, value):
    broken = series.copy()
    broken.iloc[list(positions)] = value
    return broken


def test_stage1_a_nan_p_fails_that_setting_instead_of_stopping_the_grid():
    paid, p, base_rate = stage1_data()
    broken = p["true"].copy()
    broken.iloc[0] = nan
    table = stage1_gates(paid, [p["true"], broken], base_rate, 0.1, (0.8, 1.25))
    assert table["eligible"].tolist() == [True, False]


@pytest.mark.parametrize("bad", BAD_P_VALUES)
def test_stage1_a_bad_p_has_no_scores_and_fails_every_gate(bad):
    paid, p, base_rate = stage1_data()
    table = stage1_gates(paid, [p["true"], with_bad_p(p["true"], [7], bad)], base_rate, 0.1,
                         (0.8, 1.25))
    row = table.loc[1]
    assert row["bad p"] == 1
    assert all(np.isnan(row[column]) for column in S1_SCORES)
    assert gate_flags(row, S1_GATES) == {"log loss gate": False, "skill gate": False,
                                         "slope gate": False, "eligible": False}
    assert table.loc[0, "bad p"] == 0
    assert bool(table.loc[0, "eligible"])


@pytest.mark.parametrize("bad", BAD_P_VALUES)
def test_stage1_a_bad_p_does_not_change_the_other_settings_rows(bad):
    paid, p, base_rate = stage1_data()
    good = [p["true"], p["shift"], p["under"]]
    alone = stage1_gates(paid, good, base_rate, 0.01, (0.8, 1.25))
    mixed = stage1_gates(paid, [with_bad_p(p["true"], [0], bad)] + good, base_rate, 0.01,
                         (0.8, 1.25))
    pd.testing.assert_frame_equal(mixed.iloc[1:].drop(columns="stage1").reset_index(drop=True),
                                  alone.drop(columns="stage1"))


def test_stage1_exact_zero_and_one_are_good_p():
    paid, p, base_rate = stage1_data()
    edged = p["true"].copy()
    payer, non_payer = paid[paid].index[0], paid[~paid].index[0]
    edged[payer] = 1.0
    edged[non_payer] = 0.0
    row = stage1_gates(paid, [edged], base_rate, 0.1, (0.8, 1.25)).loc[0]
    assert row["bad p"] == 0
    assert all(np.isfinite(row[column]) for column in S1_SCORES)
    assert row["log loss"] == pytest.approx(log_loss(paid, edged, labels=[False, True]),
                                            rel=1e-12)
    assert bool(row["eligible"])


def test_stage1_a_p_of_only_zeros_and_ones_is_good_and_measured():
    # Five right rows and one confidently wrong row: still a probability, so measured.
    paid = pd.Series([True, False, True, False, True, False])
    p = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    base_rate = pd.Series(0.5, index=paid.index)
    row = stage1_gates(paid, [p], base_rate, 0.1, (0.8, 1.25)).loc[0]
    assert row["bad p"] == 0
    assert np.isfinite(row["log loss"])


def test_stage1_bad_p_counts_every_bad_row_of_every_kind():
    paid, p, base_rate = stage1_data()
    broken = p["true"].copy()
    for position, value in zip([0, 5, 9, 20, 33, 40], [nan, inf, -inf, -0.1, 1.1, nan]):
        broken.iloc[position] = value
    broken.iloc[50] = 0.0   # good
    broken.iloc[51] = 1.0   # good
    table = stage1_gates(paid, [p["true"], broken], base_rate, 0.1, (0.8, 1.25))
    assert table["bad p"].tolist() == [0, 6]


def test_stage1_bad_p_column_is_an_integer_count():
    paid, p, base_rate = stage1_data()
    table = stage1_gates(paid, [p["true"], with_bad_p(p["true"], [1, 2], nan)], base_rate, 0.1,
                         (0.8, 1.25))
    assert pd.api.types.is_integer_dtype(table["bad p"])
    assert table["bad p"].tolist() == [0, 2]


def test_stage1_a_none_in_an_object_p_counts_as_bad():
    paid, p, base_rate = stage1_data()
    broken = p["true"].astype(object)
    broken.iloc[3] = None
    table = stage1_gates(paid, [p["true"], broken], base_rate, 0.1, (0.8, 1.25))
    assert table["bad p"].tolist() == [0, 1]
    assert table["eligible"].tolist() == [True, False]


def test_stage1_the_best_log_loss_is_taken_among_good_settings_only():
    # true is the best p, but with one NaN it is not measured: shift becomes the grid's best,
    # so a 1% tolerance (which shift fails beside a good true) now passes it.
    paid, p, base_rate = stage1_data()
    table = stage1_gates(paid, [with_bad_p(p["true"], [0], nan), p["shift"]], base_rate, 0.01,
                         (0.8, 1.25))
    assert table["eligible"].tolist() == [False, True]
    assert bool(table.loc[1, "log loss gate"])


@pytest.mark.parametrize("bad", [nan, inf, -1.0, 2.0])
def test_stage1_a_bad_p_anywhere_in_the_order_gives_the_same_good_rows(bad):
    paid, p, base_rate = stage1_data()
    broken = with_bad_p(p["anti"], [4], bad)
    first = stage1_gates(paid, [broken, p["true"], p["shift"]], base_rate, 0.05, (0.8, 1.25))
    middle = stage1_gates(paid, [p["true"], broken, p["shift"]], base_rate, 0.05, (0.8, 1.25))
    last = stage1_gates(paid, [p["true"], p["shift"], broken], base_rate, 0.05, (0.8, 1.25))
    rows = [table.drop(columns="stage1").reset_index(drop=True)
            for table in (first.iloc[[1, 2, 0]], middle.iloc[[0, 2, 1]], last)]
    pd.testing.assert_frame_equal(rows[0], rows[1])
    pd.testing.assert_frame_equal(rows[0], rows[2])
    assert last["eligible"].tolist() == [True, True, False]


def test_stage1_every_setting_bad_makes_every_setting_ineligible_without_error():
    paid, p, base_rate = stage1_data()
    settings = [with_bad_p(p["true"], [0], nan), with_bad_p(p["shift"], [1], inf),
                with_bad_p(p["under"], [2], -0.5), with_bad_p(p["over"], [3], 1.5)]
    table = stage1_gates(paid, settings, base_rate, 0.1, (0.8, 1.25))
    assert list(table.columns) == STAGE1_GATE_COLUMNS
    assert table["bad p"].tolist() == [1, 1, 1, 1]
    assert table[S1_SCORES].isna().all().all()
    for column in S1_GATES:
        assert table[column].dtype == bool, column
        assert not table[column].any(), column


def test_stage1_a_p_bad_on_every_row_is_counted_and_fails():
    paid, p, base_rate = stage1_data()
    all_nan = pd.Series(nan, index=paid.index)
    table = stage1_gates(paid, [p["true"], all_nan], base_rate, 0.1, (0.8, 1.25))
    assert table["bad p"].tolist() == [0, len(paid)]
    assert table["eligible"].tolist() == [True, False]


def test_stage1_a_bad_p_does_not_change_its_input():
    paid, p, base_rate = stage1_data()
    broken = with_bad_p(p["true"], [0, 1], nan)
    broken.iloc[2] = inf
    copy = broken.copy()
    stage1_gates(paid, [broken], base_rate, 0.1, (0.8, 1.25))
    pd.testing.assert_series_equal(broken, copy)


def test_stage1_a_bad_p_with_other_rows_still_raises_the_index_error():
    paid, p, base_rate = stage1_data()
    broken = with_bad_p(p["true"], [0], nan).set_axis(p["true"].index + 1)
    with pytest.raises(ValueError, match="must have the same rows"):
        stage1_gates(paid, [p["true"], broken], base_rate, 0.1, (0.8, 1.25))


def test_stage1_a_bad_p_with_a_single_outcome_still_raises():
    paid = pd.Series([True] * 4)
    p = pd.Series([0.5, nan, 0.7, 2.0])
    with pytest.raises(ValueError, match="must hold both outcomes"):
        stage1_gates(paid, [p], pd.Series(0.5, index=paid.index), 0.1, (0.8, 1.25))


# stage2_gates: shape and values

ROWS = pd.Index([3, 5, 8, 13], name="row")  # not 0..n-1, so rows are matched by label
AMOUNT = pd.Series([10.0, 20.0, 30.0, 40.0], index=ROWS)   # what was paid; sum 100
REFERENCE = pd.Series(25.0, index=ROWS)                     # the training mean


def m_of(*values):
    return pd.Series([float(v) for v in values], index=ROWS)


# The stage-2 settings, measured against AMOUNT and REFERENCE:
#   name       m                  deviance skill   total ratio
#   perfect    10, 20, 30, 40     1.0              1.0
#   high       11, 22, 33, 44     0.956            1.1 (exactly)
#   low        9, 18, 27, 36      0.950            0.9 (exactly)
#   too high   12, 24, 36, 48     0.834            1.2
#   noisy      14, 16, 34, 36     0.854            1.0
#   noisy2     15, 15, 35, 35     0.773            1.0
#   flat       25, 25, 25, 25     0.0              1.0 (the reference itself)
M = {"perfect": m_of(10, 20, 30, 40), "high": m_of(11, 22, 33, 44), "low": m_of(9, 18, 27, 36),
     "too high": m_of(12, 24, 36, 48), "noisy": m_of(14, 16, 34, 36),
     "noisy2": m_of(15, 15, 35, 35), "flat": m_of(25, 25, 25, 25)}


def gates2(names, tolerance=0.05, band=(0.9, 1.1), m=None, reference=REFERENCE):
    """stage2_gates on the named settings (from M, or from `m` if named there), by name."""
    settings = {**M, **(m or {})}
    table = stage2_gates(AMOUNT, [settings[name] for name in names], reference, tolerance,
                         band)
    table.insert(0, "name", list(names))
    return table.set_index("name")


def test_stage2_gates_one_row_per_setting_numbered_in_grid_order():
    table = stage2_gates(AMOUNT, [M["perfect"], M["noisy"], M["high"]], REFERENCE, 0.05,
                         (0.9, 1.1))
    assert list(table.columns) == STAGE2_GATE_COLUMNS
    assert table["stage2"].tolist() == [0, 1, 2]
    assert list(table.index) == [0, 1, 2]
    assert table["bad m"].tolist() == [0, 0, 0]
    for column in S2_GATES:
        assert table[column].dtype == bool, column


def test_stage2_gates_scores_like_amount_summary():
    table = gates2(["perfect", "noisy", "too high"])
    for name in ("perfect", "noisy", "too high"):
        scores = amount_summary(AMOUNT, M[name], REFERENCE)
        assert table.loc[name, "deviance skill"] == pytest.approx(scores["deviance skill"],
                                                                  rel=1e-12)
        assert table.loc[name, "total ratio"] == pytest.approx(scores["total ratio"], rel=1e-12)


def test_stage2_gates_the_right_m_is_eligible():
    row = gates2(["perfect"]).loc["perfect"]
    assert gate_flags(row, S2_GATES) == {"m gate": True, "skill gate": True,
                                         "ratio gate": True, "eligible": True}


# stage2_gates: each gate alone

@pytest.mark.parametrize("bad", [0.0, -5.0, nan, inf, -inf])
def test_stage2_m_gate_fails_one_bad_m_and_that_setting_has_no_scores(bad):
    broken = m_of(10, 20, 30, 40)
    broken.iloc[1] = bad
    table = gates2(["perfect", "broken"], m={"broken": broken})
    row = table.loc["broken"]
    assert row["bad m"] == 1
    assert np.isnan(row["deviance skill"]) and np.isnan(row["total ratio"])
    assert gate_flags(row, S2_GATES) == {"m gate": False, "skill gate": False,
                                         "ratio gate": False, "eligible": False}
    assert table.loc["perfect", "eligible"]


def test_stage2_bad_m_counts_every_bad_row():
    assert gates2(["broken"], m={"broken": m_of(nan, inf, 0, -1)}).loc["broken", "bad m"] == 4
    assert gates2(["broken"], m={"broken": m_of(nan, 20, 0, 40)}).loc["broken", "bad m"] == 2


def test_stage2_a_tiny_positive_m_passes_the_m_gate():
    row = gates2(["tiny"], tolerance=10.0, band=(0.01, 10.0),
                 m={"tiny": m_of(1e-12, 20, 30, 40)}).loc["tiny"]
    assert row["bad m"] == 0 and row["m gate"]


def test_stage2_skill_gate_alone_fails_a_setting_too_far_from_the_best():
    # noisy: skill 0.854 against the best's 1.0, tolerance 0.05; total ratio exactly 1.
    row = gates2(["perfect", "noisy"]).loc["noisy"]
    assert gate_flags(row, S2_GATES) == {"m gate": True, "skill gate": False,
                                         "ratio gate": True, "eligible": False}


def test_stage2_skill_gate_passes_the_same_setting_with_a_wider_tolerance():
    row = gates2(["perfect", "noisy"], tolerance=0.2).loc["noisy"]
    assert gate_flags(row, S2_GATES) == {"m gate": True, "skill gate": True,
                                         "ratio gate": True, "eligible": True}


def test_stage2_ratio_gate_alone_fails_a_total_ratio_above_the_band():
    row = gates2(["perfect", "too high"], tolerance=0.5).loc["too high"]
    assert row["total ratio"] == pytest.approx(1.2)
    assert gate_flags(row, S2_GATES) == {"m gate": True, "skill gate": True,
                                         "ratio gate": False, "eligible": False}


def test_stage2_ratio_gate_alone_fails_a_total_ratio_below_the_band():
    row = gates2(["perfect", "low"], tolerance=0.5, band=(0.95, 1.1)).loc["low"]
    assert gate_flags(row, S2_GATES) == {"m gate": True, "skill gate": True,
                                         "ratio gate": False, "eligible": False}


# stage2_gates: ends and tolerance

def test_stage2_ratio_band_ends_are_included():
    # 110 / 100 and 90 / 100 are exactly the floats 1.1 and 0.9.
    table = gates2(["perfect", "high", "low"], tolerance=0.5)
    assert table.loc["high", "total ratio"] == 1.1
    assert table.loc["low", "total ratio"] == 0.9
    assert table["ratio gate"].tolist() == [True, True, True]


def test_stage2_ratio_just_outside_the_band_fails():
    table = gates2(["just above", "just below"], tolerance=0.5,
                   m={"just above": m_of(11, 22, 33, 44.001),
                      "just below": m_of(9, 18, 27, 35.999)})
    assert table["ratio gate"].tolist() == [False, False]


def test_stage2_a_setting_tied_with_the_best_passes_any_tolerance():
    table = gates2(["noisy", "noisy copy"], tolerance=1e-12,
                   m={"noisy copy": M["noisy"].copy()})
    assert table["skill gate"].tolist() == [True, True]


def test_stage2_tolerance_is_relative_to_the_grids_best_not_a_running_best():
    # noisy comes first: a running best would let it pass.
    assert not gates2(["noisy", "perfect"]).loc["noisy", "skill gate"]
    # Alone, noisy is its grid's best; beside the worse noisy2 it is still the best.
    assert gates2(["noisy"]).loc["noisy", "skill gate"]
    assert gates2(["noisy2", "noisy"])["skill gate"].tolist() == [False, True]


def test_stage2_the_order_of_the_settings_does_not_change_any_settings_result():
    names = ["perfect", "high", "low", "too high", "noisy", "noisy2", "flat"]
    reference = gates2(names, tolerance=0.1).drop(columns="stage2")
    for order in (names[::-1], ["flat", "noisy", "perfect", "low", "noisy2", "too high", "high"]):
        table = gates2(order, tolerance=0.1)
        assert table["stage2"].tolist() == list(range(len(order)))
        pd.testing.assert_frame_equal(table.drop(columns="stage2").loc[names], reference)


def test_stage2_the_best_skill_is_taken_among_m_gate_passers_only():
    # broken would have been perfect but for one m of 0: it cannot set the best, so noisy
    # (the only usable setting) is the best and passes the skill gate.
    broken = m_of(10, 20, 30, 40)
    broken.iloc[0] = 0.0
    table = gates2(["broken", "noisy"], m={"broken": broken})
    assert table.loc["noisy", "skill gate"] and table.loc["noisy", "eligible"]
    assert not table.loc["broken", "eligible"]


def test_stage2_no_setting_passes_the_m_gate_then_every_setting_fails_and_nothing_raises():
    table = gates2(["a", "b"], m={"a": m_of(0, 20, 30, 40), "b": m_of(10, nan, 30, 40)})
    assert table["m gate"].tolist() == [False, False]
    assert table["skill gate"].tolist() == [False, False]
    assert table["eligible"].tolist() == [False, False]


def test_stage2_a_reference_exactly_right_gives_nan_skill_and_every_setting_fails():
    # amount_summary: no deviance left to improve on, so the skill is undefined (NaN); a NaN
    # skill compares False with the best, so the skill gate fails everywhere.
    table = gates2(["perfect", "noisy"], reference=AMOUNT.copy())
    assert table["deviance skill"].isna().all()
    assert table["m gate"].tolist() == [True, True]
    assert table["skill gate"].tolist() == [False, False]
    assert table["eligible"].tolist() == [False, False]


def test_stage2_an_m_equal_to_the_reference_has_skill_zero_and_fails_the_skill_gate():
    row = gates2(["perfect", "flat"]).loc["flat"]
    assert row["deviance skill"] == pytest.approx(0.0, abs=1e-12)
    assert gate_flags(row, S2_GATES) == {"m gate": True, "skill gate": False,
                                         "ratio gate": True, "eligible": False}


def test_stage2_gates_does_not_change_its_inputs():
    amount, m, reference = AMOUNT.copy(), M["noisy"].copy(), REFERENCE.copy()
    stage2_gates(amount, [m], reference, 0.05, (0.9, 1.1))
    pd.testing.assert_series_equal(amount, AMOUNT)
    pd.testing.assert_series_equal(m, M["noisy"])
    pd.testing.assert_series_equal(reference, REFERENCE)


# stage2_gates: errors

def test_stage2_gates_refuses_an_empty_grid():
    with pytest.raises(ValueError, match="^there are no stage-2 settings to check$"):
        stage2_gates(AMOUNT, [], REFERENCE, 0.05, (0.9, 1.1))


def test_stage2_gates_refuses_an_m_with_other_rows():
    other = M["noisy"].set_axis([3, 5, 8, 14])
    with pytest.raises(ValueError,
                       match=r"^m must have the rows of amount \(the same index\)$"):
        stage2_gates(AMOUNT, [M["perfect"], other], REFERENCE, 0.05, (0.9, 1.1))


def test_stage2_gates_refuses_an_m_with_the_rows_in_another_order():
    with pytest.raises(ValueError, match="^m must have the rows of amount"):
        stage2_gates(AMOUNT, [M["noisy"].iloc[::-1]], REFERENCE, 0.05, (0.9, 1.1))


def test_stage2_gates_refuses_an_m_with_fewer_rows():
    with pytest.raises(ValueError, match="^m must have the rows of amount"):
        stage2_gates(AMOUNT, [M["noisy"].iloc[:-1]], REFERENCE, 0.05, (0.9, 1.1))


def test_stage2_gates_refuses_a_reference_with_other_rows():
    with pytest.raises(ValueError,
                       match=r"^reference must have the rows of amount \(the same index\)$"):
        stage2_gates(AMOUNT, [M["noisy"]], REFERENCE.iloc[:-1], 0.05, (0.9, 1.1))


@pytest.mark.parametrize("bad", [0.0, -25.0, nan, inf])
def test_stage2_gates_refuses_a_reference_that_is_not_finite_and_above_zero(bad):
    reference = REFERENCE.copy()
    reference.iloc[2] = bad
    with pytest.raises(ValueError, match="^reference must be finite and above 0 in every row$"):
        stage2_gates(AMOUNT, [M["noisy"]], reference, 0.05, (0.9, 1.1))


@pytest.mark.parametrize("bad, message", [
    (0.0, r"^amount is 0 or below in 1 rows: stage 2 learns from paid rows only"),
    (-3.0, r"^amount is 0 or below in 1 rows"),
    (nan, r"^amount has 1 missing values"),
    (inf, r"^amount holds an infinite value$"),
])
def test_stage2_gates_refuses_an_amount_that_is_not_a_paid_amount(bad, message):
    amount = AMOUNT.copy()
    amount.iloc[0] = bad
    with pytest.raises(ValueError, match=message):
        stage2_gates(amount, [M["noisy"]], REFERENCE, 0.05, (0.9, 1.1))


def test_stage2_gates_refuses_no_paid_rows():
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="^there are no paid rows to learn the amount from$"):
        stage2_gates(empty, [empty.copy()], empty.copy(), 0.05, (0.9, 1.1))
