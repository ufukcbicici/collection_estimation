"""Tests for `hurdle/grid.py` (U16.2): `GridCandidates` and `grid_candidates`.

`grid_candidates` reads only `settings.stage1`, `.stage2`, `.topups` and `.grid`, so most
tests pass a small stand-in (a SimpleNamespace holding real settings dataclasses). One test
at the end loads a real `Settings` through `load_settings`. Nothing is fitted.

U16.3, at the end: `ROW_COLUMNS`, `GridPredictions`, `setting_hash`, `cached_fit` and
`grid_predictions`, fitted on test_forecast.py's small training table with a tmp_path cache.

U16.4, at the very end: `GridGates`, `with_params` and `grid_gates`, mostly on hand-made
`GridPredictions` (seeded random rows; nothing fitted), once on the U16.3 grid itself.

U16.5, after that: `GRID_SCORE_COLUMNS`, `scored_numbers`, `combination_name` and
`grid_scores`, on hand-made predictions checked against a computation by hand, then on the
U16.3 grid against `rolling_forecasts` (the current combination must score as step 17's model).
"""
import dataclasses
import itertools
from types import SimpleNamespace

import pytest

from collection_estimation.hurdle.diagnostic import Candidate
from collection_estimation.hurdle.diagnostic_joint import stage1_grid, stage2_grid
from collection_estimation.hurdle.grid import GridCandidates, grid_candidates
from collection_estimation.hurdle.settings import (
    GridSettings,
    Stage1Settings,
    Stage2Settings,
    TopupSettings,
)

STAGE1 = Stage1Settings(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                        min_samples_leaf=20, random_seed=1,
                        feature_blocks=("history", "calendar", "receivables"),
                        company_as_feature=True, baseline_c=1.0)
STAGE2 = Stage2Settings(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                        min_samples_leaf=10, random_seed=7, feature_blocks=("history", "calendar"),
                        scale="open balance", balance_floor="own mean", own_mean_window=52,
                        own_mean_prior_weeks=1.0)
TOPUPS = TopupSettings(method="mean", window=13, with_level=True, compare=False,
                       compare_first_origin=13, compare_last_origin=40,
                       compare_windows=(4, 8, 13))
GRID = GridSettings(run=True, run_test=False,
                    stage1_max_iter=(300, 900), stage1_learning_rate=(0.01, 0.05),
                    stage1_max_leaf_nodes=(31,),
                    stage2_loss=("poisson", "gamma"), stage2_max_iter=(300,),
                    stage2_learning_rate=(0.05, 0.1), stage2_max_leaf_nodes=(31,),
                    stage2_scale=("open balance",), stage2_balance_floor=("own mean",),
                    stage1_log_loss_tolerance=0.1, stage1_slope_band=(0.8, 1.25),
                    stage2_skill_tolerance=0.05, stage2_ratio_band=(0.9, 1.1),
                    topup_windows=(8, 13),
                    # No ridge by default, so the worked counts stay small; the ridge tests set
                    # it themselves.
                    topup_ridge_with_level=(), min_gain=0.005)

STAGE1_KEYS = ["max_iter", "learning_rate", "max_leaf_nodes"]
STAGE2_KEYS = ["loss", "max_iter", "learning_rate", "max_leaf_nodes", "scale", "balance_floor"]


def make(stage1=STAGE1, stage2=STAGE2, topups=TOPUPS, **grid_changes):
    """A stand-in for `Settings` with the four sections grid_candidates reads."""
    return SimpleNamespace(stage1=stage1, stage2=stage2, topups=topups,
                           grid=dataclasses.replace(GRID, **grid_changes))


def stage1_triples(candidates):
    return [(c.params["max_iter"], c.params["learning_rate"], c.params["max_leaf_nodes"])
            for c in candidates]


def stage2_tuples(candidates):
    return [tuple(c.params[key] for key in STAGE2_KEYS) for c in candidates]


# --- the result's shape ------------------------------------------------------------------

def test_returns_grid_candidates_with_three_lists():
    found = grid_candidates(make())
    assert isinstance(found, GridCandidates)
    assert type(found.stage1) is list and type(found.stage2) is list
    assert type(found.topups) is list
    assert all(isinstance(c, Candidate) for c in found.stage1 + found.stage2)
    assert all(isinstance(t, TopupSettings) for t in found.topups)


def test_grid_candidates_is_frozen():
    found = grid_candidates(make())
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.stage1 = []


def test_grid_candidates_fields_in_order():
    assert [f.name for f in dataclasses.fields(GridCandidates)] == ["stage1", "stage2", "topups"]


def test_same_settings_give_equal_candidates_and_inputs_are_untouched():
    settings = make()
    before = (settings.stage1, settings.stage2, settings.topups, settings.grid)
    assert grid_candidates(settings) == grid_candidates(settings)
    assert (settings.stage1, settings.stage2, settings.topups, settings.grid) == before
    assert settings.stage1 == STAGE1 and settings.stage2 == STAGE2 and settings.topups == TOPUPS


# --- stage 1 -------------------------------------------------------------------------------

def test_stage1_slot_0_is_the_current_stage1():
    first = grid_candidates(make()).stage1[0]
    assert first.settings == STAGE1
    assert first.class_weight == "none"
    assert first.params == {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31}
    assert list(first.params) == STAGE1_KEYS


def test_stage1_params_keys_are_the_same_as_stage1_grids():
    # The printout lists params in key order: the current one must read like the others.
    grid_keys = list(stage1_grid(STAGE1, GRID)[0].params)
    for candidate in grid_candidates(make()).stage1:
        assert list(candidate.params) == grid_keys


def test_stage1_current_in_the_grid_is_listed_once_and_first():
    # Grid 300/900 x 0.01/0.05 x 31 holds the current (300, 0.05, 31).
    found = grid_candidates(make()).stage1
    assert stage1_triples(found) == [(300, 0.05, 31), (300, 0.01, 31), (900, 0.01, 31),
                                     (900, 0.05, 31)]
    assert len(found) == 4
    assert sum(c.settings == STAGE1 for c in found) == 1


def test_stage1_current_not_in_the_grid_is_still_first():
    found = grid_candidates(make(stage1_max_iter=(100, 600))).stage1
    assert stage1_triples(found) == [(300, 0.05, 31), (100, 0.01, 31), (100, 0.05, 31),
                                     (600, 0.01, 31), (600, 0.05, 31)]
    assert found[0].settings == STAGE1


def test_stage1_rest_is_stage1_grid_in_its_order_without_the_current():
    settings = make(stage1_max_iter=(900, 300, 100), stage1_learning_rate=(0.2, 0.05),
                    stage1_max_leaf_nodes=(63, 31, 15))
    expected = [c for c in stage1_grid(STAGE1, settings.grid) if c.settings != STAGE1]
    found = grid_candidates(settings).stage1
    assert found[1:] == expected
    assert len(found) == 3 * 2 * 3


def test_stage1_rest_order_follows_itertools_product_of_the_grid():
    settings = make(stage1_max_iter=(900, 100), stage1_learning_rate=(0.2, 0.01),
                    stage1_max_leaf_nodes=(63, 15))
    expected = list(itertools.product((900, 100), (0.2, 0.01), (63, 15)))
    assert stage1_triples(grid_candidates(settings).stage1[1:]) == expected


def test_stage1_a_one_point_grid_equal_to_the_current_gives_one_candidate():
    found = grid_candidates(make(stage1_max_iter=(300,), stage1_learning_rate=(0.05,),
                                 stage1_max_leaf_nodes=(31,))).stage1
    assert len(found) == 1
    assert found[0].settings == STAGE1


def test_stage1_a_one_point_grid_other_than_the_current_gives_two():
    found = grid_candidates(make(stage1_max_iter=(900,), stage1_learning_rate=(0.01,),
                                 stage1_max_leaf_nodes=(63,))).stage1
    assert stage1_triples(found) == [(300, 0.05, 31), (900, 0.01, 63)]


def test_stage1_no_two_candidates_have_equal_settings():
    settings = make(stage1_max_iter=(100, 300, 900), stage1_learning_rate=(0.01, 0.05, 0.1),
                    stage1_max_leaf_nodes=(15, 31))
    found = grid_candidates(settings).stage1
    assert len({c.settings for c in found}) == len(found) == 18


def test_stage1_every_candidate_has_no_class_weights():
    for candidate in grid_candidates(make()).stage1:
        assert candidate.class_weight == "none"


def test_stage1_candidates_keep_every_other_stage1_setting():
    # Only max_iter, learning_rate and max_leaf_nodes may differ from [stage1].
    for candidate in grid_candidates(make(stage1_max_iter=(100, 900))).stage1:
        same = dataclasses.replace(candidate.settings, max_iter=STAGE1.max_iter,
                                   learning_rate=STAGE1.learning_rate,
                                   max_leaf_nodes=STAGE1.max_leaf_nodes)
        assert same == STAGE1
        assert candidate.settings.company_as_feature is True
        assert candidate.settings.feature_blocks == ("history", "calendar", "receivables")


def test_stage1_params_match_the_settings_they_carry():
    for candidate in grid_candidates(make(stage1_max_iter=(100, 900))).stage1:
        assert candidate.params == {"max_iter": candidate.settings.max_iter,
                                    "learning_rate": candidate.settings.learning_rate,
                                    "max_leaf_nodes": candidate.settings.max_leaf_nodes}


@pytest.mark.parametrize("company_as_feature", [True, False])
def test_stage1_company_as_feature_is_kept_in_every_candidate(company_as_feature):
    stage1 = dataclasses.replace(STAGE1, company_as_feature=company_as_feature)
    for candidate in grid_candidates(make(stage1=stage1)).stage1:
        assert candidate.settings.company_as_feature is company_as_feature


def test_stage1_current_with_a_rate_the_grid_could_not_hold_stays_first():
    # [stage1] may hold 0.5 (rate_setting); the grid refuses it, but the current is kept.
    stage1 = dataclasses.replace(STAGE1, learning_rate=0.5)
    found = grid_candidates(make(stage1=stage1)).stage1
    assert found[0].settings == stage1
    assert found[0].params["learning_rate"] == 0.5
    assert len(found) == 5


# --- stage 2 -------------------------------------------------------------------------------

def test_stage2_slot_0_is_the_current_stage2():
    first = grid_candidates(make()).stage2[0]
    assert first.settings == STAGE2
    assert first.class_weight == "none"
    assert first.params == {"loss": "poisson", "max_iter": 300, "learning_rate": 0.05,
                            "max_leaf_nodes": 31, "scale": "open balance",
                            "balance_floor": "own mean"}
    assert list(first.params) == STAGE2_KEYS


def test_stage2_params_keys_are_the_same_as_stage2_grids():
    grid_keys = list(stage2_grid(STAGE2, GRID)[0].params)
    for candidate in grid_candidates(make()).stage2:
        assert list(candidate.params) == grid_keys


@pytest.mark.parametrize("floor", ["median", "mean", "own mean"])
def test_stage2_current_scale_none_shows_the_floor_as_a_dash(floor):
    stage2 = dataclasses.replace(STAGE2, scale="none", balance_floor=floor)
    first = grid_candidates(make(stage2=stage2)).stage2[0]
    assert first.params["scale"] == "none"
    assert first.params["balance_floor"] == "-"
    # The settings themselves keep the floor they hold.
    assert first.settings == stage2
    assert first.settings.balance_floor == floor


@pytest.mark.parametrize("floor", ["median", "mean", "own mean"])
def test_stage2_current_open_balance_shows_its_floor(floor):
    stage2 = dataclasses.replace(STAGE2, balance_floor=floor)
    assert grid_candidates(make(stage2=stage2)).stage2[0].params["balance_floor"] == floor


def test_stage2_current_in_the_grid_is_listed_once_and_first():
    # Grid poisson/gamma x 300 x 0.05/0.1 x 31 x open balance x own mean holds the current.
    found = grid_candidates(make()).stage2
    assert stage2_tuples(found) == [
        ("poisson", 300, 0.05, 31, "open balance", "own mean"),
        ("poisson", 300, 0.1, 31, "open balance", "own mean"),
        ("gamma", 300, 0.05, 31, "open balance", "own mean"),
        ("gamma", 300, 0.1, 31, "open balance", "own mean")]
    assert sum(c.settings == STAGE2 for c in found) == 1


def test_stage2_current_not_in_the_grid_is_still_first():
    found = grid_candidates(make(stage2_max_iter=(900,))).stage2
    assert len(found) == 1 + 4
    assert found[0].settings == STAGE2
    assert all(c.settings.max_iter == 900 for c in found[1:])


def test_stage2_rest_is_stage2_grid_in_its_order_without_the_current():
    settings = make(stage2_loss=("gamma", "poisson"), stage2_max_iter=(900, 300),
                    stage2_learning_rate=(0.1, 0.05), stage2_max_leaf_nodes=(63, 31),
                    stage2_scale=("open balance", "none"),
                    stage2_balance_floor=("own mean", "median"))
    expected = [c for c in stage2_grid(STAGE2, settings.grid) if c.settings != STAGE2]
    found = grid_candidates(settings).stage2
    assert found[1:] == expected
    # 16 (loss, iter, rate, leaves) points x (2 floors under open balance + 1 under none),
    # the current among them.
    assert len(found) == 16 * 3


def test_stage2_current_scale_none_in_the_grid_is_not_listed_twice():
    # Under "none" the grid's candidate keeps the current floor, so it equals the current.
    stage2 = dataclasses.replace(STAGE2, scale="none", balance_floor="median")
    settings = make(stage2=stage2, stage2_loss=("poisson",), stage2_learning_rate=(0.05,),
                    stage2_scale=("none", "open balance"),
                    stage2_balance_floor=("median", "own mean"))
    found = grid_candidates(settings).stage2
    assert stage2_tuples(found) == [
        ("poisson", 300, 0.05, 31, "none", "-"),
        ("poisson", 300, 0.05, 31, "open balance", "median"),
        ("poisson", 300, 0.05, 31, "open balance", "own mean")]
    assert sum(c.settings == stage2 for c in found) == 1


def test_stage2_current_scale_none_with_a_floor_outside_the_grid_is_still_deduplicated():
    # The floor does not matter under "none": the grid's "none" candidate is the current.
    stage2 = dataclasses.replace(STAGE2, scale="none", balance_floor="mean")
    settings = make(stage2=stage2, stage2_loss=("poisson",), stage2_learning_rate=(0.05,),
                    stage2_scale=("none",), stage2_balance_floor=("own mean",))
    found = grid_candidates(settings).stage2
    assert len(found) == 1
    assert found[0].settings == stage2


def test_stage2_same_hyperparameters_other_floor_is_a_different_candidate():
    # Current open balance + "mean"; the grid's open balance + "own mean" is another setting.
    stage2 = dataclasses.replace(STAGE2, balance_floor="mean")
    settings = make(stage2=stage2, stage2_loss=("poisson",), stage2_learning_rate=(0.05,))
    found = grid_candidates(settings).stage2
    assert stage2_tuples(found) == [
        ("poisson", 300, 0.05, 31, "open balance", "mean"),
        ("poisson", 300, 0.05, 31, "open balance", "own mean")]


def test_stage2_current_squared_error_stays_first_and_the_grid_adds_none():
    stage2 = dataclasses.replace(STAGE2, loss="squared_error")
    found = grid_candidates(make(stage2=stage2)).stage2
    assert found[0].settings == stage2
    assert found[0].params["loss"] == "squared_error"
    assert [c.params["loss"] for c in found[1:]] == ["poisson", "poisson", "gamma", "gamma"]
    assert len(found) == 5


def test_stage2_no_two_candidates_have_equal_settings():
    settings = make(stage2_max_iter=(300, 900), stage2_scale=("none", "open balance"),
                    stage2_balance_floor=("median", "own mean"))
    found = grid_candidates(settings).stage2
    assert len({c.settings for c in found}) == len(found)


def test_stage2_every_candidate_has_no_class_weights():
    for candidate in grid_candidates(make()).stage2:
        assert candidate.class_weight == "none"


def test_stage2_candidates_keep_every_other_stage2_setting():
    settings = make(stage2_scale=("none", "open balance"),
                    stage2_balance_floor=("median", "own mean"))
    for candidate in grid_candidates(settings).stage2:
        s = candidate.settings
        assert s.feature_blocks == ("history", "calendar")
        assert (s.min_samples_leaf, s.random_seed) == (10, 7)
        assert (s.own_mean_window, s.own_mean_prior_weeks) == (52, 1.0)


def test_stage2_params_match_the_settings_they_carry():
    settings = make(stage2_scale=("none", "open balance"),
                    stage2_balance_floor=("median", "own mean"))
    for candidate in grid_candidates(settings).stage2:
        s = candidate.settings
        floor = s.balance_floor if s.scale == "open balance" else "-"
        assert candidate.params == {"loss": s.loss, "max_iter": s.max_iter,
                                    "learning_rate": s.learning_rate,
                                    "max_leaf_nodes": s.max_leaf_nodes, "scale": s.scale,
                                    "balance_floor": floor}


@pytest.mark.parametrize("blocks", [("history",), ("calendar", "receivables"),
                                    ("history", "calendar", "receivables")])
def test_feature_blocks_of_each_stage_are_kept_in_every_candidate(blocks):
    stage1 = dataclasses.replace(STAGE1, feature_blocks=blocks)
    stage2 = dataclasses.replace(STAGE2, feature_blocks=blocks[::-1])
    found = grid_candidates(make(stage1=stage1, stage2=stage2))
    assert all(c.settings.feature_blocks == blocks for c in found.stage1)
    assert all(c.settings.feature_blocks == blocks[::-1] for c in found.stage2)


def test_stage2_candidates_are_stage2_settings_and_stage1_candidates_stage1_settings():
    found = grid_candidates(make())
    assert all(type(c.settings) is Stage1Settings for c in found.stage1)
    assert all(type(c.settings) is Stage2Settings for c in found.stage2)


def test_the_stage_lists_do_not_depend_on_each_other():
    a = grid_candidates(make())
    b = grid_candidates(make(stage2_max_iter=(100, 900), stage2_loss=("gamma",)))
    assert a.stage1 == b.stage1
    c = grid_candidates(make(stage1_max_leaf_nodes=(15, 63)))
    assert a.stage2 == c.stage2
    assert a.topups == b.topups == c.topups


# --- top-ups -------------------------------------------------------------------------------

def test_topups_slot_0_is_the_current_topups_itself():
    found = grid_candidates(make()).topups
    assert found[0] is TOPUPS


def test_topups_current_window_in_the_grid_is_not_listed_twice():
    found = grid_candidates(make(topup_windows=(8, 13))).topups
    assert [(t.method, t.window) for t in found] == [("mean", 13), ("mean", 8)]


def test_topups_current_window_not_in_the_grid_is_still_first():
    found = grid_candidates(make(topup_windows=(4, 8))).topups
    assert [(t.method, t.window) for t in found] == [("mean", 13), ("mean", 4), ("mean", 8)]


def test_topups_follow_the_order_of_topup_windows():
    found = grid_candidates(make(topup_windows=(26, 4, 52))).topups
    assert [t.window for t in found] == [13, 26, 4, 52]


def test_topups_a_single_window_equal_to_the_current_gives_one():
    assert grid_candidates(make(topup_windows=(13,))).topups == [TOPUPS]


def test_topups_candidates_change_only_the_method_and_the_window():
    for topups in grid_candidates(make(topup_windows=(4, 8))).topups[1:]:
        assert dataclasses.replace(topups, method=TOPUPS.method, window=TOPUPS.window) == TOPUPS
        assert topups.with_level is True
        assert topups.compare_windows == (4, 8, 13)


def test_topups_current_ridge_stays_first_and_every_window_is_added():
    ridge = dataclasses.replace(TOPUPS, method="ridge", window=13, with_level=False)
    found = grid_candidates(make(topups=ridge, topup_windows=(8, 13))).topups
    assert found[0] is ridge
    # The ridge's own window is not a mean of it: both means are added.
    assert [(t.method, t.window) for t in found] == [("ridge", 13), ("mean", 8), ("mean", 13)]
    assert found[1] == dataclasses.replace(ridge, method="mean", window=8)
    assert found[2].with_level is False


def test_topups_an_empty_ridge_list_adds_no_ridge():
    found = grid_candidates(make(topup_windows=(4, 8, 13, 26), topup_ridge_with_level=())).topups
    assert all(t.method == "mean" for t in found)
    ridge = dataclasses.replace(TOPUPS, method="ridge")
    found = grid_candidates(make(topups=ridge, topup_windows=(4, 8),
                                 topup_ridge_with_level=())).topups
    assert [t.method for t in found] == ["ridge", "mean", "mean"]


def tops(found):
    return [(t.method, t.window, t.with_level) for t in found]


def test_topups_order_is_current_then_means_then_ridges():
    # Current mean 13 (with the level); means 8 (13 is the current); then both ridges.
    found = grid_candidates(make(topup_windows=(8, 13),
                                 topup_ridge_with_level=(True, False))).topups
    assert tops(found) == [("mean", 13, True), ("mean", 8, True), ("ridge", 13, True),
                           ("ridge", 13, False)]


def test_topups_ridges_follow_the_order_of_topup_ridge_with_level():
    found = grid_candidates(make(topup_windows=(13,),
                                 topup_ridge_with_level=(False, True))).topups
    assert tops(found) == [("mean", 13, True), ("ridge", 13, False), ("ridge", 13, True)]


@pytest.mark.parametrize("with_level", [True, False])
def test_topups_a_single_ridge_value_adds_one_ridge(with_level):
    found = grid_candidates(make(topup_windows=(13,), topup_ridge_with_level=(with_level,))).topups
    assert tops(found) == [("mean", 13, True), ("ridge", 13, with_level)]


def test_topups_ridge_candidates_change_only_the_method_and_with_level():
    found = grid_candidates(make(topup_ridge_with_level=(True, False))).topups
    ridges = [t for t in found if t.method == "ridge"]
    assert len(ridges) == 2
    for ridge in ridges:
        # The ridge keeps [topups]' window (unused by the ridge) and every compare setting.
        assert dataclasses.replace(ridge, method=TOPUPS.method,
                                   with_level=TOPUPS.with_level) == TOPUPS
        assert ridge.window == 13


@pytest.mark.parametrize("with_level", [True, False])
def test_topups_a_current_ridge_with_the_same_level_is_not_listed_twice(with_level):
    current = dataclasses.replace(TOPUPS, method="ridge", window=8, with_level=with_level)
    found = grid_candidates(make(topups=current, topup_windows=(8, 13),
                                 topup_ridge_with_level=(True, False))).topups
    assert found[0] is current
    # Both means (the ridge's window is not a mean), then only the other ridge.
    assert tops(found) == [("ridge", 8, with_level), ("mean", 8, with_level),
                           ("mean", 13, with_level), ("ridge", 8, not with_level)]
    assert sum(t == current for t in found) == 1


def test_topups_a_current_ridge_and_only_its_own_level_adds_no_ridge():
    current = dataclasses.replace(TOPUPS, method="ridge", with_level=False)
    found = grid_candidates(make(topups=current, topup_windows=(4,),
                                 topup_ridge_with_level=(False,))).topups
    assert tops(found) == [("ridge", 13, False), ("mean", 4, False)]


def test_topups_with_ridges_no_two_candidates_are_equal():
    for current in (TOPUPS, dataclasses.replace(TOPUPS, method="ridge"),
                    dataclasses.replace(TOPUPS, method="ridge", with_level=False)):
        found = grid_candidates(make(topups=current, topup_windows=(4, 8, 13),
                                     topup_ridge_with_level=(True, False))).topups
        assert len(set(found)) == len(found), current


def test_topups_the_ridge_list_does_not_change_the_stage_lists():
    a = grid_candidates(make())
    b = grid_candidates(make(topup_ridge_with_level=(True, False)))
    assert (a.stage1, a.stage2) == (b.stage1, b.stage2)
    assert b.topups[:len(a.topups)] == a.topups


def test_topups_no_two_candidates_are_equal():
    found = grid_candidates(make(topup_windows=(4, 8, 13, 26, 52))).topups
    assert len(set(found)) == len(found) == 5


def test_the_topups_do_not_change_the_stage_lists():
    a = grid_candidates(make())
    b = grid_candidates(make(topups=dataclasses.replace(TOPUPS, method="ridge"),
                             topup_windows=(4,)))
    assert (a.stage1, a.stage2) == (b.stage1, b.stage2)


# --- counts --------------------------------------------------------------------------------

def test_the_counts_of_the_example():
    found = grid_candidates(make())
    assert (len(found.stage1), len(found.stage2), len(found.topups)) == (4, 4, 2)


def test_the_gate_settings_and_min_gain_do_not_change_the_candidates():
    a = grid_candidates(make())
    b = grid_candidates(make(stage1_log_loss_tolerance=0.5, stage1_slope_band=(0.5, 2.0),
                             stage2_skill_tolerance=0.3, stage2_ratio_band=(0.5, 1.5),
                             min_gain=0.1, run=False, run_test=True))
    assert a == b


# --- a real Settings ---------------------------------------------------------------------

def test_a_loaded_settings_gives_its_current_sections_first(tmp_path):
    from collection_estimation.hurdle.settings import load_settings
    from tests.hurdle.test_settings import write_settings
    repo = tmp_path / "repo"
    repo.mkdir()
    settings = load_settings(write_settings(tmp_path), repo)
    found = grid_candidates(settings)
    assert found.stage1[0].settings == settings.stage1
    assert found.stage2[0].settings == settings.stage2
    assert found.topups[0] is settings.topups
    assert found.stage1[1:] == [c for c in stage1_grid(settings.stage1, settings.grid)
                                if c.settings != settings.stage1]
    assert found.stage2[1:] == [c for c in stage2_grid(settings.stage2, settings.grid)
                                if c.settings != settings.stage2]
    means = [dataclasses.replace(settings.topups, method="mean", window=w)
             for w in settings.grid.topup_windows]
    ridges = [dataclasses.replace(settings.topups, method="ridge", with_level=level)
              for level in settings.grid.topup_ridge_with_level]
    assert found.topups[1:] == [t for t in means + ridges if t != settings.topups]
    # test_settings' VALID: [topups] is a ridge without the level, the grid tries [false, true]
    # ridges, so only the ridge with the level is added.
    assert [(t.method, t.with_level) for t in found.topups if t.method == "ridge"] == [
        ("ridge", False), ("ridge", True)]
    for candidate in found.stage1:
        assert candidate.settings.company_as_feature == settings.stage1.company_as_feature
        assert candidate.settings.feature_blocks == settings.stage1.feature_blocks
    for candidate in found.stage2:
        assert candidate.settings.feature_blocks == settings.stage2.feature_blocks


# =============================================================================================
# U16.3: `ROW_COLUMNS`, `GridPredictions`, `setting_hash`, `cached_fit` and `grid_predictions`.
#
# The small training table, settings and side tables are test_forecast.py's (8 customers,
# companies TR02 and TR04, origins 0..30, h 1..5, ten trees), so one fit takes well under a
# second. Every cache folder is a pytest tmp_path; nothing reads or writes data/.
# =============================================================================================

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from collection_estimation.hurdle import grid as grid_module  # noqa: E402
from collection_estimation.hurdle.diagnostic import reference_probabilities  # noqa: E402
from collection_estimation.hurdle.forecast import (  # noqa: E402
    CUSTOMER_COLUMNS, customer_forecasts, stage1_predictions, stage2_predictions)
from collection_estimation.hurdle.grid import (  # noqa: E402
    ROW_COLUMNS, GridPredictions, cached_fit, grid_predictions, setting_hash)
from collection_estimation.hurdle.stage2 import amount_references  # noqa: E402
from collection_estimation.hurdle.topups import topup_forecasts  # noqa: E402
from collection_estimation.hurdle.training_table import (  # noqa: E402
    rows_at_origin, rows_known_at)
from tests.hurdle.test_forecast import (  # noqa: E402
    BLOCK_COLUMNS, F_COMPANIES, f_calendar, f_levels, f_settings, f_side_tables, make_table,
    stage1_settings)

G_ORIGINS = [18, 22]
G_WINDOW = 4            # the table's history columns are for a 4-week window


def g_settings(horizons=5, company_as_feature=True, reference_window=G_WINDOW,
               amount_reference_window=G_WINDOW, reference_prior_weeks=1.0):
    """f_settings plus the [diagnostic] reference windows grid_predictions reads."""
    settings = f_settings(horizons=horizons,
                          stage1=stage1_settings(company_as_feature=company_as_feature))
    settings.diagnostic = SimpleNamespace(reference_window=reference_window,
                                          amount_reference_window=amount_reference_window,
                                          reference_prior_weeks=reference_prior_weeks)
    return settings


def g_candidates(settings=None, n1=2, n2=2, topups=None):
    """Candidate 0 of each stage is the current [stage1] / [stage2]; the others change
    max_iter (stage 1) or learning_rate (stage 2). Top-ups: the current, then a 4-week mean."""
    settings = settings or g_settings()
    s1, s2 = settings.stage1, settings.stage2
    stage1 = [Candidate(params={"max_iter": s1.max_iter}, settings=s1, class_weight="none")]
    for i in range(1, n1):
        changed = dataclasses.replace(s1, max_iter=s1.max_iter + 3 * i)
        stage1.append(Candidate(params={"max_iter": changed.max_iter}, settings=changed,
                                class_weight="none"))
    stage2 = [Candidate(params={"learning_rate": s2.learning_rate}, settings=s2,
                        class_weight="none")]
    for j in range(1, n2):
        changed = dataclasses.replace(s2, learning_rate=0.1 + 0.1 * j)
        stage2.append(Candidate(params={"learning_rate": changed.learning_rate},
                                settings=changed, class_weight="none"))
    if topups is None:
        topups = [settings.topups, dataclasses.replace(settings.topups, window=4)]
    return GridCandidates(stage1=stage1, stage2=stage2, topups=topups)


def g_inputs(side=None):
    new_cash, refunds, unidentified = side if side is not None else f_side_tables()
    return dict(block_columns=BLOCK_COLUMNS, companies=list(F_COMPANIES), new_cash=new_cash,
                refunds=refunds, unidentified=unidentified, levels=f_levels(),
                calendar=f_calendar())


def g_run(table, cache_folder, origins=G_ORIGINS, candidates=None, settings=None,
          on_fit_done=None, **overrides):
    settings = settings or g_settings()
    kwargs = g_inputs()
    kwargs.update(overrides)
    return grid_predictions(table, origins, candidates or g_candidates(settings), settings,
                            cache_folder=cache_folder, on_fit_done=on_fit_done, **kwargs)


def count_fits(monkeypatch, fail=False):
    """Wrap the two stage functions (as grid sees them): record each call, then fit (or, with
    `fail`, raise). Returns the list of ("stage1" / "stage2", origin of the forecast rows)."""
    calls = []

    def wrap(name, real):
        def step(train_rows, forecast_rows, *args, **kwargs):
            calls.append((name, int(forecast_rows["origin_week"].iloc[0])))
            if fail:
                raise AssertionError(f"{name} must not be called")
            return real(train_rows, forecast_rows, *args, **kwargs)
        return step

    monkeypatch.setattr(grid_module, "stage1_predictions", wrap("stage1", stage1_predictions))
    monkeypatch.setattr(grid_module, "stage2_predictions", wrap("stage2", stage2_predictions))
    return calls


def cache_files(folder):
    folder = Path(folder)
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


@pytest.fixture(scope="module")
def g_table():
    return make_table()


@pytest.fixture(scope="module")
def g_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("grid_cache")


@pytest.fixture(scope="module")
def g_result(g_table, g_cache):
    progress = []
    result = g_run(g_table, g_cache, on_fit_done=lambda *args: progress.append(args))
    return result, progress


def copied_cache(g_cache, tmp_path):
    """A private copy of the module's filled cache (ask for g_result first)."""
    target = tmp_path / "cache"
    shutil.copytree(g_cache, target)
    return target


# --- constants and the result type -------------------------------------------------------

def test_row_columns_are_the_documented_list():
    assert ROW_COLUMNS == ["key", "company_code", "customer_id", "origin_week", "h",
                           "target_week", "label_status", "paid", "amount_deflated",
                           "base_rate", "training_mean"]
    assert ROW_COLUMNS[:6] == CUSTOMER_COLUMNS[:6]


def test_grid_predictions_fields_in_order_and_frozen(g_result):
    assert [f.name for f in dataclasses.fields(GridPredictions)] == ["rows", "p", "m", "topups"]
    result, _ = g_result
    assert isinstance(result, GridPredictions)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.p = None


# --- setting_hash ----------------------------------------------------------------------------

def test_setting_hash_is_the_first_16_hex_of_sha256_of_the_sorted_json():
    text = json.dumps(dataclasses.asdict(STAGE1), sort_keys=True)
    assert setting_hash(STAGE1) == hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_setting_hash_is_16_lowercase_hex_characters():
    found = setting_hash(STAGE2)
    assert len(found) == 16
    assert all(c in "0123456789abcdef" for c in found)


def test_setting_hash_is_deterministic_and_equal_for_equal_settings():
    assert setting_hash(STAGE1) == setting_hash(dataclasses.replace(STAGE1))
    assert setting_hash(TOPUPS) == setting_hash(TOPUPS)


@pytest.mark.parametrize("change", [dict(max_iter=301), dict(learning_rate=0.051),
                                    dict(company_as_feature=False),
                                    dict(feature_blocks=("history",)), dict(random_seed=2)])
def test_setting_hash_changes_with_any_field(change):
    assert setting_hash(dataclasses.replace(STAGE1, **change)) != setting_hash(STAGE1)


def test_setting_hash_of_different_stages_differ():
    assert setting_hash(STAGE1) != setting_hash(STAGE2)


def test_setting_hash_a_tuple_and_a_list_field_hash_the_same():
    # json.dumps writes both as a JSON list: the hash is of what the cache saves.
    as_list = dataclasses.replace(STAGE1, feature_blocks=list(STAGE1.feature_blocks))
    assert setting_hash(as_list) == setting_hash(STAGE1)


def test_setting_hash_refuses_a_non_dataclass():
    with pytest.raises(TypeError):
        setting_hash(SimpleNamespace(max_iter=1))


# --- cached_fit --------------------------------------------------------------------------

def fake_fit(calls, value=1.5):
    def fit():
        calls.append(value)
        return pd.Series([value, 2 * value], index=[10, 11])
    return fit


def test_cached_fit_first_call_fits_and_writes_named_files(tmp_path):
    calls = []
    found = cached_fit("stage1", 18, STAGE1, ["a", 1], tmp_path, fake_fit(calls))
    assert calls == [1.5]
    pd.testing.assert_series_equal(found, pd.Series([1.5, 3.0], index=[10, 11]))
    name = f"stage1_origin18_{setting_hash(STAGE1)}"
    assert cache_files(tmp_path) == [f"{name}.pkl", f"{name}_fingerprint.json"]


def test_cached_fit_fingerprint_is_stage_origin_setting_then_inputs(tmp_path):
    cached_fit("stage2", 7, STAGE2, [{"x": [1, 2]}, "z"], tmp_path, fake_fit([]))
    name = f"stage2_origin7_{setting_hash(STAGE2)}"
    saved = json.loads((tmp_path / f"{name}_fingerprint.json").read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(["stage2", 7, dataclasses.asdict(STAGE2), {"x": [1, 2]},
                                      "z"]))
    assert saved == expected


def test_cached_fit_second_call_reads_without_fitting(tmp_path):
    calls = []
    first = cached_fit("stage1", 18, STAGE1, ["a"], tmp_path, fake_fit(calls, 1.5))
    second = cached_fit("stage1", 18, STAGE1, ["a"], tmp_path, fake_fit(calls, 9.0))
    assert calls == [1.5]
    pd.testing.assert_series_equal(first, second)


def test_cached_fit_changed_input_refits_and_overwrites(tmp_path):
    calls = []
    cached_fit("stage1", 18, STAGE1, ["a"], tmp_path, fake_fit(calls, 1.5))
    found = cached_fit("stage1", 18, STAGE1, ["b"], tmp_path, fake_fit(calls, 9.0))
    assert calls == [1.5, 9.0]
    assert found.iloc[0] == 9.0
    # Same name (the inputs are not in it): the file is replaced, not added.
    assert len(cache_files(tmp_path)) == 2
    again = cached_fit("stage1", 18, STAGE1, ["b"], tmp_path, fake_fit(calls, 0.0))
    assert calls == [1.5, 9.0] and again.iloc[0] == 9.0


def test_cached_fit_changed_setting_refits_into_another_file(tmp_path):
    calls = []
    other = dataclasses.replace(STAGE1, max_iter=900)
    cached_fit("stage1", 18, STAGE1, ["a"], tmp_path, fake_fit(calls, 1.5))
    cached_fit("stage1", 18, other, ["a"], tmp_path, fake_fit(calls, 2.5))
    assert calls == [1.5, 2.5]
    assert len(cache_files(tmp_path)) == 4
    # Both are kept: going back to the first setting reads its file.
    back = cached_fit("stage1", 18, STAGE1, ["a"], tmp_path, fake_fit(calls, 7.0))
    assert calls == [1.5, 2.5] and back.iloc[0] == 1.5


@pytest.mark.parametrize("stage, origin", [("stage2", 18), ("stage1", 19)])
def test_cached_fit_other_stage_or_origin_is_another_fit(tmp_path, stage, origin):
    calls = []
    cached_fit("stage1", 18, STAGE1, ["a"], tmp_path, fake_fit(calls, 1.5))
    cached_fit(stage, origin, STAGE1, ["a"], tmp_path, fake_fit(calls, 2.5))
    assert calls == [1.5, 2.5]
    assert f"{stage}_origin{origin}_{setting_hash(STAGE1)}.pkl" in cache_files(tmp_path)


def test_cached_fit_creates_a_missing_folder(tmp_path):
    folder = tmp_path / "a" / "b"
    cached_fit("stage1", 1, STAGE1, [], folder, fake_fit([]))
    assert folder.is_dir() and len(cache_files(folder)) == 2


def test_cached_fit_a_damaged_pickle_is_refitted(tmp_path):
    calls = []
    cached_fit("stage1", 18, STAGE1, ["a"], tmp_path, fake_fit(calls, 1.5))
    (tmp_path / f"stage1_origin18_{setting_hash(STAGE1)}.pkl").write_bytes(b"not a pickle")
    found = cached_fit("stage1", 18, STAGE1, ["a"], tmp_path, fake_fit(calls, 3.0))
    assert calls == [1.5, 3.0] and found.iloc[0] == 3.0


def test_cached_fit_a_fit_that_raises_leaves_no_cache(tmp_path):
    def broken():
        raise ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        cached_fit("stage1", 18, STAGE1, [], tmp_path, broken)
    assert cache_files(tmp_path) == []


# --- grid_predictions: the shapes -----------------------------------------------------------

def expected_rows_index(table, origins):
    return pd.concat([rows_at_origin(table, o) for o in sorted(set(origins))]).index


def test_grid_rows_have_row_columns_and_the_tables_index_in_origin_order(g_table, g_result):
    result, _ = g_result
    assert list(result.rows.columns) == ROW_COLUMNS
    assert result.rows.index.equals(expected_rows_index(g_table, G_ORIGINS))
    assert list(result.rows["origin_week"].unique()) == G_ORIGINS


def test_grid_rows_copy_the_table_columns(g_table, g_result):
    result, _ = g_result
    expected = g_table.loc[result.rows.index, ROW_COLUMNS[:9]]
    pd.testing.assert_frame_equal(result.rows[ROW_COLUMNS[:9]], expected)


def test_grid_rows_hold_the_labels_of_future_rows_as_nan(g_result):
    # Origin 22 predicts target weeks 23..27; past 23 the table has no label yet.
    result, _ = g_result
    future = result.rows[result.rows["target_week"] > 23]
    assert len(future) > 0
    assert (future["label_status"] == "future").all()
    assert future["amount_deflated"].isna().all()


def test_grid_p_and_m_have_one_column_per_candidate_and_the_rows_index(g_result):
    result, _ = g_result
    assert list(result.p.columns) == [0, 1]
    assert list(result.m.columns) == [0, 1]
    assert result.p.index.equals(result.rows.index)
    assert result.m.index.equals(result.rows.index)
    assert [str(d) for d in result.p.dtypes] == ["float64", "float64"]
    assert [str(d) for d in result.m.dtypes] == ["float64", "float64"]


def test_grid_candidates_give_different_p_and_m(g_result):
    # Candidate 1 is another setting: its column is not a copy of candidate 0's.
    result, _ = g_result
    assert not np.array_equal(result.p[0].to_numpy(), result.p[1].to_numpy())
    assert not np.array_equal(result.m[0].to_numpy(), result.m[1].to_numpy())


def test_grid_topups_one_table_per_candidate(g_result):
    result, _ = g_result
    assert type(result.topups) is list and len(result.topups) == 2
    assert all(isinstance(t, pd.DataFrame) for t in result.topups)


@pytest.mark.parametrize("n1, n2, n3", [(1, 1, 1), (3, 1, 2), (1, 3, 1)])
def test_grid_column_counts_follow_the_candidate_counts(g_table, tmp_path, n1, n2, n3):
    settings = g_settings()
    topups = [dataclasses.replace(settings.topups, window=w) for w in (8, 4, 2)[:n3]]
    result = g_run(g_table, tmp_path, origins=[18], settings=settings,
                   candidates=g_candidates(settings, n1, n2, topups))
    assert list(result.p.columns) == list(range(n1))
    assert list(result.m.columns) == list(range(n2))
    assert len(result.topups) == n3


# --- grid_predictions: the values -----------------------------------------------------------

def test_grid_candidate_0_equals_customer_forecasts_exactly(g_table, g_result):
    # The current settings at each origin give customer_forecasts' p and m, bit for bit.
    result, _ = g_result
    expected = pd.concat([customer_forecasts(g_table, o, g_settings(), BLOCK_COLUMNS,
                                             list(F_COMPANIES)) for o in G_ORIGINS])
    assert result.p.index.equals(expected.index)
    assert np.array_equal(result.p[0].to_numpy(), expected["p"].to_numpy())
    assert np.array_equal(result.m[0].to_numpy(), expected["m"].to_numpy())


def test_grid_every_candidate_equals_the_stage_functions_by_hand(g_table, g_result):
    result, _ = g_result
    settings = g_settings()
    candidates = g_candidates(settings)
    for origin in G_ORIGINS:
        train, at = rows_known_at(g_table, origin), rows_at_origin(g_table, origin)
        for i, candidate in enumerate(candidates.stage1):
            p = stage1_predictions(train, at, candidate.settings, BLOCK_COLUMNS, F_COMPANIES)
            assert np.array_equal(result.p.loc[at.index, i].to_numpy(), p.to_numpy()), (origin, i)
        for j, candidate in enumerate(candidates.stage2):
            m = stage2_predictions(train, at, candidate.settings, BLOCK_COLUMNS, F_COMPANIES,
                                   settings.stage1.company_as_feature)
            assert np.array_equal(result.m.loc[at.index, j].to_numpy(), m.to_numpy()), (origin, j)


def test_grid_stage2_reads_the_current_company_as_feature(g_table, tmp_path):
    # Stage-2 candidates get [stage1] company_as_feature of the settings.
    settings = g_settings(company_as_feature=False)
    result = g_run(g_table, tmp_path, origins=[18], settings=settings,
                   candidates=g_candidates(settings, 1, 1))
    train, at = rows_known_at(g_table, 18), rows_at_origin(g_table, 18)
    m = stage2_predictions(train, at, settings.stage2, BLOCK_COLUMNS, F_COMPANIES, False)
    assert np.array_equal(result.m[0].to_numpy(), m.to_numpy())
    with_company = stage2_predictions(train, at, settings.stage2, BLOCK_COLUMNS, F_COMPANIES,
                                      True)
    assert not np.array_equal(result.m[0].to_numpy(), with_company.to_numpy())


def test_grid_topups_equal_topup_forecasts_per_candidate(g_result):
    result, _ = g_result
    settings = g_settings()
    new_cash, refunds, unidentified = f_side_tables()
    for k, topup in enumerate(g_candidates(settings).topups):
        expected = topup_forecasts(new_cash, refunds, unidentified, f_levels(), f_calendar(),
                                   G_ORIGINS, 5, topup.method, topup.window, topup.with_level)
        pd.testing.assert_frame_equal(result.topups[k], expected)


def test_grid_topup_candidates_differ(g_result):
    result, _ = g_result
    assert not result.topups[0]["topup"].equals(result.topups[1]["topup"])


def test_grid_topups_cover_every_origin(g_result):
    result, _ = g_result
    for topups in result.topups:
        assert sorted(topups["origin_week"].unique()) == G_ORIGINS


def test_grid_references_equal_the_reference_functions(g_table, g_result):
    result, _ = g_result
    for origin in G_ORIGINS:
        train, at = rows_known_at(g_table, origin), rows_at_origin(g_table, origin)
        base = reference_probabilities(train, at, G_WINDOW, 1.0)["base rate"]
        mean = amount_references(train, at, G_WINDOW, 1.0)["training mean"]
        rows = result.rows.loc[at.index]
        assert np.array_equal(rows["base_rate"].to_numpy(), base.to_numpy()), origin
        assert np.array_equal(rows["training_mean"].to_numpy(), mean.to_numpy()), origin


def test_grid_references_are_one_value_per_origin_learned_from_rows_known_there(g_table,
                                                                                 g_result):
    result, _ = g_result
    for origin in G_ORIGINS:
        rows = result.rows[result.rows["origin_week"] == origin]
        train = rows_known_at(g_table, origin)
        assert rows["base_rate"].nunique() == 1
        assert rows["base_rate"].iloc[0] == pytest.approx(train["paid"].astype(bool).mean())
        paid = train[train["paid"].astype(bool)]
        assert rows["training_mean"].iloc[0] == pytest.approx(paid["amount_deflated"].mean())
    # Different origins learn from different rows.
    assert result.rows.groupby("origin_week")["base_rate"].first().nunique() == 2


def test_grid_references_read_the_diagnostic_settings(g_table, tmp_path, monkeypatch):
    seen = []
    real_p, real_m = reference_probabilities, amount_references

    def spy_p(train, rows, window, prior):
        seen.append(("p", window, prior))
        return real_p(train, rows, window, prior)

    def spy_m(train, rows, window, prior):
        seen.append(("m", window, prior))
        return real_m(train, rows, window, prior)
    monkeypatch.setattr(grid_module, "reference_probabilities", spy_p)
    monkeypatch.setattr(grid_module, "amount_references", spy_m)
    settings = g_settings(reference_window=4, amount_reference_window=4,
                          reference_prior_weeks=2.5)
    # Distinct windows are not possible on this table (one window, 4), so the spy proves
    # which field goes where by a second run below with a fake window.
    g_run(g_table, tmp_path, origins=[18], settings=settings,
          candidates=g_candidates(settings, 1, 1, [settings.topups]))
    assert seen == [("p", 4, 2.5), ("m", 4, 2.5)]


def test_grid_reference_window_and_amount_window_go_to_their_own_reference(g_table, tmp_path,
                                                                           monkeypatch):
    seen = []

    def fake_p(train, rows, window, prior):
        seen.append(("p", window, prior))
        return pd.DataFrame({"base rate": 0.5}, index=rows.index)

    def fake_m(train, rows, window, prior):
        seen.append(("m", window, prior))
        return pd.DataFrame({"training mean": 7.0}, index=rows.index)
    monkeypatch.setattr(grid_module, "reference_probabilities", fake_p)
    monkeypatch.setattr(grid_module, "amount_references", fake_m)
    settings = g_settings(reference_window=13, amount_reference_window=52,
                          reference_prior_weeks=0.75)
    result = g_run(g_table, tmp_path, origins=[18], settings=settings,
                   candidates=g_candidates(settings, 1, 1, [settings.topups]))
    assert seen == [("p", 13, 0.75), ("m", 52, 0.75)]
    assert (result.rows["base_rate"] == 0.5).all()
    assert (result.rows["training_mean"] == 7.0).all()


@pytest.mark.parametrize("field", ["reference_window", "amount_reference_window"])
def test_grid_a_reference_window_the_table_lacks_raises(g_table, tmp_path, field):
    settings = g_settings(**{field: 13})
    with pytest.raises(ValueError, match="no history columns"):
        g_run(g_table, tmp_path, origins=[18], settings=settings,
              candidates=g_candidates(settings, 1, 1, [settings.topups]))


# --- the references come before each origin's fits ------------------------------------------

@pytest.mark.parametrize("field", ["reference_window", "amount_reference_window"])
def test_grid_a_reference_window_the_table_lacks_refused_before_any_fit(g_table, tmp_path,
                                                                        monkeypatch, field):
    # Both origins, the full candidate lists: no fit, no callback, no cache folder.
    settings = g_settings(**{field: 13})
    message = refused(monkeypatch, g_table, tmp_path, settings=settings,
                      candidates=g_candidates(settings))
    assert "no history columns" in message


def test_grid_a_raising_amount_reference_alone_stops_before_any_fit(g_table, tmp_path,
                                                                    monkeypatch):
    # reference_probabilities is fine; amount_references (called second) raises.
    def broken(*args, **kwargs):
        raise ValueError("amount reference broken")
    monkeypatch.setattr(grid_module, "amount_references", broken)
    assert refused(monkeypatch, g_table, tmp_path) == "amount reference broken"


def test_grid_a_raising_base_rate_reference_stops_before_any_fit(g_table, tmp_path,
                                                                 monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("base rate broken")
    monkeypatch.setattr(grid_module, "reference_probabilities", broken)
    assert refused(monkeypatch, g_table, tmp_path) == "base rate broken"


def test_grid_references_are_made_before_the_fits_at_each_origin(g_table, tmp_path,
                                                                 monkeypatch):
    events = []
    real_p, real_m, real_cached = reference_probabilities, amount_references, \
        grid_module.cached_fit

    def spy_p(train, rows, *args):
        events.append(("p", int(rows["origin_week"].iloc[0])))
        return real_p(train, rows, *args)

    def spy_m(train, rows, *args):
        events.append(("m", int(rows["origin_week"].iloc[0])))
        return real_m(train, rows, *args)

    def spy_fit(stage, origin, *args, **kwargs):
        events.append((stage, origin))
        return real_cached(stage, origin, *args, **kwargs)
    monkeypatch.setattr(grid_module, "reference_probabilities", spy_p)
    monkeypatch.setattr(grid_module, "amount_references", spy_m)
    monkeypatch.setattr(grid_module, "cached_fit", spy_fit)
    g_run(g_table, tmp_path)
    per_origin = [("p", None), ("m", None), ("stage1", None), ("stage1", None),
                  ("stage2", None), ("stage2", None)]
    assert events == [(name, origin) for origin in G_ORIGINS for name, _ in per_origin]


def test_grid_a_reference_failing_at_the_second_origin_keeps_the_first_origins_fits(
        g_table, tmp_path, monkeypatch):
    # The references are made per origin, before that origin's fits: origin 18 is fitted
    # (and cached) in full, origin 22 not at all.
    real = reference_probabilities

    def broken_at_22(train, rows, *args):
        if int(rows["origin_week"].iloc[0]) == 22:
            raise ValueError("no reference at 22")
        return real(train, rows, *args)
    monkeypatch.setattr(grid_module, "reference_probabilities", broken_at_22)
    calls = count_fits(monkeypatch)
    progress = []
    with pytest.raises(ValueError, match="no reference at 22"):
        g_run(g_table, tmp_path / "cache", on_fit_done=lambda *a: progress.append(a))
    assert calls == [("stage1", 18), ("stage1", 18), ("stage2", 18), ("stage2", 18)]
    assert [args[:5] for args in progress] == [
        (1, 8, "stage 1", 18, 0), (2, 8, "stage 1", 18, 1),
        (3, 8, "stage 2", 18, 0), (4, 8, "stage 2", 18, 1)]
    files = cache_files(tmp_path / "cache")
    assert len(files) == 8 and all("_origin18_" in name for name in files)


def test_grid_a_reference_failure_leaves_an_existing_cache_untouched(g_table, g_cache, g_result,
                                                                     tmp_path, monkeypatch):
    cache = copied_cache(g_cache, tmp_path)
    before = {name: (cache / name).read_bytes() for name in cache_files(cache)}
    calls = count_fits(monkeypatch, fail=True)
    settings = g_settings(amount_reference_window=13)
    with pytest.raises(ValueError, match="no history columns"):
        g_run(g_table, cache, settings=settings, candidates=g_candidates(g_settings()))
    assert calls == []
    assert {name: (cache / name).read_bytes() for name in cache_files(cache)} == before


# --- numpy-int origins ---------------------------------------------------------------------

@pytest.mark.parametrize("origins", [
    np.array([22, 18, 22], dtype=np.int64),
    [np.int64(22), np.int64(18)],
    np.array([18, 22], dtype=np.int32),
    [np.int64(18), 22],
], ids=["int64 array", "np.int64 list", "int32 array", "mixed"])
def test_grid_numpy_origins_read_the_cache_of_plain_ones(g_table, g_cache, g_result, tmp_path,
                                                        monkeypatch, origins):
    # Same fingerprint as plain ints: every fit is read back, the output is the same.
    result, first = g_result
    calls = count_fits(monkeypatch, fail=True)
    progress = []
    found = g_run(g_table, copied_cache(g_cache, tmp_path), origins=origins,
                  on_fit_done=lambda *args: progress.append(args))
    assert calls == []
    pd.testing.assert_frame_equal(found.rows, result.rows)
    pd.testing.assert_frame_equal(found.p, result.p)
    pd.testing.assert_frame_equal(found.m, result.m)
    assert len(found.topups) == len(result.topups)
    for a, b in zip(found.topups, result.topups):
        pd.testing.assert_frame_equal(a, b)
    assert [args[:5] for args in progress] == [args[:5] for args in first]


def test_grid_numpy_origins_on_a_fresh_cache_fit_and_write_plain_fingerprints(g_table,
                                                                             tmp_path):
    # Before the fix the cache's json.dumps raised TypeError on an np.int64 origin.
    settings = g_settings()
    progress = []
    found = g_run(g_table, tmp_path, origins=np.array([18], dtype=np.int64), settings=settings,
                  candidates=g_candidates(settings, 1, 1, [settings.topups]),
                  on_fit_done=lambda *args: progress.append(args))
    name = f"stage1_origin18_{setting_hash(settings.stage1)}"
    assert f"{name}.pkl" in cache_files(tmp_path)
    saved = json.loads((tmp_path / f"{name}_fingerprint.json").read_text(encoding="utf-8"))
    assert saved[:2] == ["stage1", 18]
    assert [type(args[3]) for args in progress] == [int, int]
    assert list(found.rows["origin_week"].unique()) == [18]
    expected = customer_forecasts(g_table, 18, settings, BLOCK_COLUMNS, list(F_COMPANIES))
    assert np.array_equal(found.p[0].to_numpy(), expected["p"].to_numpy())
    assert np.array_equal(found.m[0].to_numpy(), expected["m"].to_numpy())


def test_grid_numpy_origins_reach_the_topups_as_plain_ints(g_table, tmp_path, monkeypatch):
    seen = []

    def spy(*args, **kwargs):
        seen.append(list(args[5]))
        raise ValueError("stop here")
    monkeypatch.setattr(grid_module, "topup_forecasts", spy)
    with pytest.raises(ValueError, match="stop here"):
        g_run(g_table, tmp_path, origins=np.array([22, 18], dtype=np.int64))
    assert seen == [[18, 22]]
    assert [type(origin) for origin in seen[0]] == [int, int]


def test_grid_unsorted_repeated_origins_equal_sorted_unique(g_table, g_cache, g_result,
                                                            tmp_path):
    result, _ = g_result
    found = g_run(g_table, copied_cache(g_cache, tmp_path), origins=[22, 18, 22])
    pd.testing.assert_frame_equal(found.rows, result.rows)
    pd.testing.assert_frame_equal(found.p, result.p)
    pd.testing.assert_frame_equal(found.m, result.m)
    for a, b in zip(found.topups, result.topups):
        pd.testing.assert_frame_equal(a, b)


def test_grid_each_origins_slice_equals_the_grid_at_that_origin_alone(g_table, g_result,
                                                                      tmp_path):
    result, _ = g_result
    alone = g_run(g_table, tmp_path, origins=[22])
    at = (result.rows["origin_week"] == 22).to_numpy()
    pd.testing.assert_frame_equal(alone.rows, result.rows[at])
    pd.testing.assert_frame_equal(alone.p, result.p[at])
    pd.testing.assert_frame_equal(alone.m, result.m[at])


def test_grid_does_not_change_its_inputs(g_table, tmp_path):
    table = g_table.copy()
    kwargs = g_inputs()
    before = {name: value.copy() for name, value in kwargs.items()
              if isinstance(value, (pd.DataFrame, pd.Series))}
    blocks = {k: list(v) for k, v in BLOCK_COLUMNS.items()}
    settings = g_settings()
    candidates = g_candidates(settings)
    candidates_before = (list(candidates.stage1), list(candidates.stage2),
                         list(candidates.topups))
    grid_predictions(table, [18], candidates, settings, cache_folder=tmp_path, **kwargs)
    pd.testing.assert_frame_equal(table, g_table)
    for name, value in before.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(kwargs[name], value)
        else:
            pd.testing.assert_series_equal(kwargs[name], value)
    assert BLOCK_COLUMNS == blocks and kwargs["companies"] == list(F_COMPANIES)
    assert (candidates.stage1, candidates.stage2, candidates.topups) == candidates_before


def test_grid_rows_are_not_a_view_of_the_table(g_table, tmp_path):
    table = g_table.copy()
    result = g_run(table, tmp_path, origins=[18],
                   candidates=g_candidates(n1=1, n2=1, topups=[g_settings().topups]))
    result.rows.loc[result.rows.index[0], "amount_deflated"] = -123.0
    assert (table["amount_deflated"] != -123.0).all()


# --- grid_predictions: the progress callback -------------------------------------------------

def test_grid_callback_once_per_fit_in_order(g_result):
    _, progress = g_result
    # 2 origins x (2 stage-1 + 2 stage-2) = 8 fits.
    assert [args[:5] for args in progress] == [
        (1, 8, "stage 1", 18, 0), (2, 8, "stage 1", 18, 1),
        (3, 8, "stage 2", 18, 0), (4, 8, "stage 2", 18, 1),
        (5, 8, "stage 1", 22, 0), (6, 8, "stage 1", 22, 1),
        (7, 8, "stage 2", 22, 0), (8, 8, "stage 2", 22, 1)]


def test_grid_callback_seconds_are_non_negative_floats(g_result):
    _, progress = g_result
    assert all(len(args) == 6 for args in progress)
    assert all(isinstance(args[5], float) and args[5] >= 0 for args in progress)


def test_grid_callback_total_is_origins_times_both_stages(g_table, tmp_path):
    settings = g_settings()
    progress = []
    g_run(g_table, tmp_path, origins=[18, 20, 22], settings=settings,
          candidates=g_candidates(settings, 3, 1, [settings.topups]),
          on_fit_done=lambda *args: progress.append(args))
    assert len(progress) == 12
    assert {args[1] for args in progress} == {3 * (3 + 1)}
    assert [args[0] for args in progress] == list(range(1, 13))
    assert [args[2] for args in progress[:4]] == ["stage 1"] * 3 + ["stage 2"]


def test_grid_callback_is_called_for_cached_fits_too(g_table, g_cache, g_result, tmp_path,
                                                     monkeypatch):
    _, first = g_result
    calls = count_fits(monkeypatch, fail=True)
    progress = []
    g_run(g_table, copied_cache(g_cache, tmp_path),
          on_fit_done=lambda *args: progress.append(args))
    assert calls == []
    assert [args[:5] for args in progress] == [args[:5] for args in first]


def test_grid_callback_runs_after_its_fit(g_table, tmp_path, monkeypatch):
    events = []
    real = stage1_predictions

    def stage1(*args, **kwargs):
        events.append("fit")
        return real(*args, **kwargs)
    monkeypatch.setattr(grid_module, "stage1_predictions", stage1)
    settings = g_settings()
    g_run(g_table, tmp_path, origins=[18], settings=settings,
          candidates=g_candidates(settings, 2, 1, [settings.topups]),
          on_fit_done=lambda *args: events.append(args[2]))
    assert events == ["fit", "stage 1", "fit", "stage 1", "stage 2"]


def test_grid_no_callback_is_the_default(g_table, g_cache, g_result, tmp_path):
    result, _ = g_result
    kwargs = g_inputs()
    settings = g_settings()
    found = grid_predictions(g_table, G_ORIGINS, g_candidates(settings), settings,
                             cache_folder=copied_cache(g_cache, tmp_path), **kwargs)
    pd.testing.assert_frame_equal(found.p, result.p)
    pd.testing.assert_frame_equal(found.m, result.m)
    pd.testing.assert_frame_equal(found.rows, result.rows)


# --- grid_predictions: the cache --------------------------------------------------------------

def test_grid_writes_one_cache_file_pair_per_fit(g_cache, g_result):
    settings = g_settings()
    candidates = g_candidates(settings)
    expected = []
    for origin in G_ORIGINS:
        for stage, listed in (("stage1", candidates.stage1), ("stage2", candidates.stage2)):
            for candidate in listed:
                name = f"{stage}_origin{origin}_{setting_hash(candidate.settings)}"
                expected += [f"{name}.pkl", f"{name}_fingerprint.json"]
    assert cache_files(g_cache) == sorted(expected)


def test_grid_second_run_reads_every_fit_from_the_cache(g_table, g_cache, g_result, tmp_path,
                                                        monkeypatch):
    result, _ = g_result
    calls = count_fits(monkeypatch, fail=True)
    found = g_run(g_table, copied_cache(g_cache, tmp_path))
    assert calls == []
    pd.testing.assert_frame_equal(found.p, result.p)
    pd.testing.assert_frame_equal(found.m, result.m)
    pd.testing.assert_frame_equal(found.rows, result.rows)


def test_grid_a_fresh_cache_fits_everything_once(g_table, tmp_path, monkeypatch):
    calls = count_fits(monkeypatch)
    g_run(g_table, tmp_path)
    assert calls == [("stage1", 18), ("stage1", 18), ("stage2", 18), ("stage2", 18),
                     ("stage1", 22), ("stage1", 22), ("stage2", 22), ("stage2", 22)]


def test_grid_one_more_stage1_candidate_fits_only_that_one(g_table, g_cache, g_result,
                                                          tmp_path, monkeypatch):
    result, _ = g_result
    calls = count_fits(monkeypatch)
    settings = g_settings()
    found = g_run(g_table, copied_cache(g_cache, tmp_path), settings=settings,
                  candidates=g_candidates(settings, n1=3, n2=2))
    assert calls == [("stage1", 18), ("stage1", 22)]
    pd.testing.assert_frame_equal(found.p[[0, 1]], result.p)
    pd.testing.assert_frame_equal(found.m, result.m)


def test_grid_one_more_stage2_candidate_fits_only_that_one(g_table, g_cache, g_result,
                                                          tmp_path, monkeypatch):
    calls = count_fits(monkeypatch)
    settings = g_settings()
    g_run(g_table, copied_cache(g_cache, tmp_path), settings=settings,
          candidates=g_candidates(settings, n1=2, n2=3))
    assert calls == [("stage2", 18), ("stage2", 22)]


def test_grid_a_reordered_candidate_list_reads_the_cache(g_table, g_cache, g_result, tmp_path,
                                                         monkeypatch):
    # The cache is keyed by setting, not by candidate number.
    result, _ = g_result
    calls = count_fits(monkeypatch, fail=True)
    candidates = g_candidates()
    swapped = GridCandidates(stage1=candidates.stage1[::-1], stage2=candidates.stage2,
                             topups=candidates.topups)
    found = g_run(g_table, copied_cache(g_cache, tmp_path), candidates=swapped)
    assert calls == []
    assert np.array_equal(found.p[0].to_numpy(), result.p[1].to_numpy())
    assert np.array_equal(found.p[1].to_numpy(), result.p[0].to_numpy())


def test_grid_one_more_origin_fits_only_there(g_table, g_cache, g_result, tmp_path,
                                             monkeypatch):
    calls = count_fits(monkeypatch)
    g_run(g_table, copied_cache(g_cache, tmp_path), origins=[18, 20, 22])
    assert calls == [("stage1", 20), ("stage1", 20), ("stage2", 20), ("stage2", 20)]


def test_grid_more_topup_candidates_fit_nothing(g_table, g_cache, g_result, tmp_path,
                                                monkeypatch):
    calls = count_fits(monkeypatch, fail=True)
    settings = g_settings()
    topups = [settings.topups, dataclasses.replace(settings.topups, window=4),
              dataclasses.replace(settings.topups, window=2)]
    found = g_run(g_table, copied_cache(g_cache, tmp_path), settings=settings,
                  candidates=g_candidates(settings, topups=topups))
    assert calls == [] and len(found.topups) == 3


def test_grid_a_changed_table_refits_everything(g_table, g_cache, g_result, tmp_path,
                                                monkeypatch):
    calls = count_fits(monkeypatch)
    table = g_table.copy()
    # A change no fit at 18 or 22 reads (a later origin's feature) still refits: the
    # fingerprint holds the whole table.
    late = table.index[table["origin_week"] == 30][0]
    table.loc[late, "business_days"] = 1.0
    g_run(table, copied_cache(g_cache, tmp_path))
    assert len(calls) == 8


@pytest.mark.parametrize("overrides", [
    dict(block_columns={**BLOCK_COLUMNS, "calendar": []}),
    dict(companies=["TR02", "TR04", "TR05"]),
], ids=["blocks", "companies"])
def test_grid_changed_blocks_or_companies_refit_everything(g_table, g_cache, g_result, tmp_path,
                                                           monkeypatch, overrides):
    calls = count_fits(monkeypatch)
    g_run(g_table, copied_cache(g_cache, tmp_path), **overrides)
    assert len(calls) == 8


def test_grid_companies_in_another_order_read_the_cache(g_table, g_cache, g_result, tmp_path,
                                                        monkeypatch):
    # The fingerprint holds the sorted list, and the fits sort it too (stage1_features).
    result, _ = g_result
    calls = count_fits(monkeypatch, fail=True)
    found = g_run(g_table, copied_cache(g_cache, tmp_path),
                  companies=list(reversed(F_COMPANIES)))
    assert calls == []
    pd.testing.assert_frame_equal(found.p, result.p)


def test_grid_reversed_companies_really_give_the_same_fit(g_table):
    # The premise of the test above, without the cache.
    train, at = rows_known_at(g_table, 18), rows_at_origin(g_table, 18)
    a = stage1_predictions(train, at, g_settings().stage1, BLOCK_COLUMNS, ["TR02", "TR04"])
    b = stage1_predictions(train, at, g_settings().stage1, BLOCK_COLUMNS, ["TR04", "TR02"])
    pd.testing.assert_series_equal(a, b)


def test_grid_changed_horizons_setting_refits(g_table, g_cache, g_result, tmp_path,
                                              monkeypatch):
    # [receivables] is in every fit's inputs. With 4 horizons the table must have h 1..4.
    calls = count_fits(monkeypatch)
    settings = g_settings(horizons=4)
    table = g_table[g_table["h"] <= 4]
    g_run(table, copied_cache(g_cache, tmp_path), settings=settings,
          candidates=g_candidates(settings))
    assert len(calls) == 8


def test_grid_changed_company_as_feature_refits_stage2_only(g_table, g_cache, g_result,
                                                            tmp_path, monkeypatch):
    # The stage-1 candidates are kept as they were; only settings.stage1 (which stage 2
    # reads) changes: stage 1 reads the cache, every stage-2 candidate is refitted.
    calls = count_fits(monkeypatch)
    candidates = g_candidates(g_settings())
    new = g_settings(company_as_feature=False)
    g_run(g_table, copied_cache(g_cache, tmp_path), settings=new, candidates=candidates)
    assert calls == [("stage2", 18), ("stage2", 18), ("stage2", 22), ("stage2", 22)]


def test_grid_changed_diagnostic_settings_refit_nothing(g_table, g_cache, g_result, tmp_path,
                                                        monkeypatch):
    calls = count_fits(monkeypatch, fail=True)
    settings = g_settings(reference_prior_weeks=5.0)
    g_run(g_table, copied_cache(g_cache, tmp_path), settings=settings,
          candidates=g_candidates(g_settings()))
    assert calls == []


def test_grid_changed_topup_side_tables_refit_nothing(g_table, g_cache, g_result, tmp_path,
                                                      monkeypatch):
    result, _ = g_result
    calls = count_fits(monkeypatch, fail=True)
    new_cash, refunds, unidentified = f_side_tables(seed=9)
    found = g_run(g_table, copied_cache(g_cache, tmp_path), new_cash=new_cash,
                  refunds=refunds, unidentified=unidentified)
    assert calls == []
    assert not found.topups[0].equals(result.topups[0])


def test_grid_cache_files_name_stage_origin_and_setting(g_table, tmp_path):
    settings = g_settings()
    g_run(g_table, tmp_path, origins=[18], settings=settings,
          candidates=g_candidates(settings, 1, 1, [settings.topups]))
    assert cache_files(tmp_path) == sorted([
        f"stage1_origin18_{setting_hash(settings.stage1)}.pkl",
        f"stage1_origin18_{setting_hash(settings.stage1)}_fingerprint.json",
        f"stage2_origin18_{setting_hash(settings.stage2)}.pkl",
        f"stage2_origin18_{setting_hash(settings.stage2)}_fingerprint.json"])


def test_grid_a_stopped_run_keeps_the_fits_it_finished(g_table, tmp_path, monkeypatch):
    # The first stage-2 fit fails: the two stage-1 fits at origin 18 are cached and are not
    # fitted again on the next run.
    settings = g_settings()
    candidates = g_candidates(settings)

    def stage2(*args, **kwargs):
        raise RuntimeError("stopped")
    monkeypatch.setattr(grid_module, "stage2_predictions", stage2)
    with pytest.raises(RuntimeError, match="stopped"):
        g_run(g_table, tmp_path, settings=settings, candidates=candidates)
    assert len(cache_files(tmp_path)) == 4
    calls = count_fits(monkeypatch)
    g_run(g_table, tmp_path, settings=settings, candidates=candidates)
    assert calls[:2] == [("stage2", 18), ("stage2", 18)]
    assert ("stage1", 18) not in calls
    assert len(calls) == 6


# --- grid_predictions: refused before any fit --------------------------------------------------

def refused(monkeypatch, table, tmp_path, **kwargs):
    """Run with the fits spied on; return the ValueError's message (no fit, no file)."""
    calls = count_fits(monkeypatch, fail=True)
    progress = []
    with pytest.raises(ValueError) as error:
        g_run(table, tmp_path / "cache", on_fit_done=lambda *a: progress.append(a), **kwargs)
    assert calls == [] and progress == []
    assert not (tmp_path / "cache").exists()
    return str(error.value)


@pytest.mark.parametrize("origins", [[], ()], ids=["list", "tuple"])
def test_grid_no_origins_refused_before_any_fit(g_table, tmp_path, monkeypatch, origins):
    assert refused(monkeypatch, g_table, tmp_path, origins=origins) == (
        "there are no origins to forecast at")


def test_grid_duplicate_index_refused_before_any_fit(g_table, tmp_path, monkeypatch):
    table = g_table.copy()
    labels = list(table.index)
    labels[1] = labels[0]
    table.index = pd.Index(labels)
    assert "more than once" in refused(monkeypatch, table, tmp_path)


def test_grid_all_in_companies_refused_before_any_fit(g_table, tmp_path, monkeypatch):
    assert refused(monkeypatch, g_table, tmp_path, companies=["TR02", "TR04", "ALL"]) == (
        "'ALL' is the code of the consolidated rows; it cannot be a company")


def test_grid_a_last_origin_with_no_rows_refused_before_any_fit(g_table, tmp_path, monkeypatch):
    assert refused(monkeypatch, g_table, tmp_path, origins=[18, 22, 99]) == (
        "the table has no rows at origin week 99")


def test_grid_horizons_unlike_the_table_refused_before_any_fit(g_table, tmp_path, monkeypatch):
    settings = g_settings(horizons=4)
    message = refused(monkeypatch, g_table, tmp_path, settings=settings,
                      candidates=g_candidates(settings))
    assert message == ("the table's rows at origin week 18 are for h [1, 2, 3, 4, 5], but "
                       "[receivables] horizons is 4")


def test_grid_a_bad_last_topup_candidate_refused_before_any_fit(g_table, tmp_path, monkeypatch):
    settings = g_settings()
    topups = [settings.topups, dataclasses.replace(settings.topups, method="median")]
    message = refused(monkeypatch, g_table, tmp_path, settings=settings,
                      candidates=g_candidates(settings, topups=topups))
    assert message == "method is 'median'; it must be one of ('mean', 'ridge')"


def test_grid_topup_window_below_one_refused_before_any_fit(g_table, tmp_path, monkeypatch):
    settings = g_settings()
    topups = [dataclasses.replace(settings.topups, window=0)]
    message = refused(monkeypatch, g_table, tmp_path, settings=settings,
                      candidates=g_candidates(settings, topups=topups))
    assert message == "window is 0; the trailing mean needs at least 1 week"


def test_grid_topup_data_missing_for_the_last_origin_refused_before_any_fit(g_table, tmp_path,
                                                                            monkeypatch):
    # As in test_evaluate: no new-customer cash in target weeks 23..26, so the 4-week mean
    # of candidate 1 has no data at origin 26 (18 is fine). Candidate 1 is last.
    new_cash, refunds, unidentified = f_side_tables()
    new_cash = new_cash[~new_cash["target_week"].between(23, 26)]
    message = refused(monkeypatch, g_table, tmp_path, origins=[18, 26], new_cash=new_cash,
                      refunds=refunds, unidentified=unidentified)
    assert message == ("h 1, origin 26: no week with data in the 4 weeks up to the origin, so "
                       "no trailing mean")


def test_grid_origin_checks_come_before_the_topups(g_table, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(grid_module, "topup_forecasts",
                        lambda *a, **k: called.append("topups"))
    with pytest.raises(ValueError, match="no rows at origin week 99"):
        g_run(g_table, tmp_path, origins=[99])
    assert called == []


def test_grid_calls_check_forecast_origins_with_the_horizons(g_table, tmp_path, monkeypatch):
    seen = []

    def spy(table, origins, companies, horizons):
        seen.append((table is g_table, list(origins), list(companies), horizons))
        raise ValueError("stop here")
    monkeypatch.setattr(grid_module, "check_forecast_origins", spy)
    with pytest.raises(ValueError, match="stop here"):
        g_run(g_table, tmp_path, origins=[22, 18])
    assert seen == [(True, [22, 18], list(F_COMPANIES), 5)]


def test_grid_every_topup_candidate_made_once_for_all_origins_up_front(g_table, tmp_path,
                                                                       monkeypatch):
    events = []
    real_topups, real_cached = topup_forecasts, grid_module.cached_fit

    def topups(*args, **kwargs):
        events.append(("topups", list(args[5]), args[6], args[7], args[8]))
        return real_topups(*args, **kwargs)

    def cached(*args, **kwargs):
        events.append(("fit",))
        return real_cached(*args, **kwargs)
    monkeypatch.setattr(grid_module, "topup_forecasts", topups)
    monkeypatch.setattr(grid_module, "cached_fit", cached)
    g_run(g_table, tmp_path, origins=[22, 18])
    assert events[:2] == [("topups", [18, 22], 5, "mean", 8), ("topups", [18, 22], 5, "mean", 4)]
    assert events[2:] == [("fit",)] * 8


# --- grid_predictions: p and m are not checked ---------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.5, -0.2])
def test_grid_a_bad_p_is_passed_through(g_table, tmp_path, monkeypatch, bad):
    def stage1(train_rows, forecast_rows, *args, **kwargs):
        return pd.Series(bad, index=forecast_rows.index)
    monkeypatch.setattr(grid_module, "stage1_predictions", stage1)
    settings = g_settings()
    result = g_run(g_table, tmp_path, origins=[18], settings=settings,
                   candidates=g_candidates(settings, 1, 1, [settings.topups]))
    assert np.array_equal(result.p[0].to_numpy(), np.full(len(result.rows), bad),
                          equal_nan=True)


@pytest.mark.parametrize("bad", [np.nan, -np.inf, -1.0])
def test_grid_a_bad_m_is_passed_through(g_table, tmp_path, monkeypatch, bad):
    def stage2(train_rows, forecast_rows, *args, **kwargs):
        return pd.Series(bad, index=forecast_rows.index)
    monkeypatch.setattr(grid_module, "stage2_predictions", stage2)
    settings = g_settings()
    result = g_run(g_table, tmp_path, origins=[18], settings=settings,
                   candidates=g_candidates(settings, 1, 1, [settings.topups]))
    assert np.array_equal(result.m[0].to_numpy(), np.full(len(result.rows), bad),
                          equal_nan=True)


def test_grid_one_broken_candidate_does_not_stop_the_others(g_table, tmp_path, monkeypatch):
    real = stage1_predictions

    def stage1(train_rows, forecast_rows, setting, *args, **kwargs):
        if setting.max_iter != 10:     # candidate 1
            return pd.Series(np.nan, index=forecast_rows.index)
        return real(train_rows, forecast_rows, setting, *args, **kwargs)
    monkeypatch.setattr(grid_module, "stage1_predictions", stage1)
    result = g_run(g_table, tmp_path)
    assert result.p[1].isna().all()
    assert result.p[0].between(0, 1).all()


# --- grid_predictions on grid_candidates' own output ----------------------------------------

def test_grid_runs_on_grid_candidates_output(g_table, tmp_path):
    # Candidates from grid_candidates itself (on the small settings): candidate 0 is still
    # customer_forecasts.
    settings = g_settings()
    settings.grid = dataclasses.replace(
        GRID, stage1_max_iter=(10, 5), stage1_learning_rate=(0.1,), stage1_max_leaf_nodes=(7,),
        stage2_loss=("poisson",), stage2_max_iter=(10,), stage2_learning_rate=(0.1,),
        stage2_max_leaf_nodes=(7,), stage2_scale=("none",), stage2_balance_floor=("median",),
        topup_windows=(8,), topup_ridge_with_level=())
    candidates = grid_candidates(settings)
    assert (len(candidates.stage1), len(candidates.stage2), len(candidates.topups)) == (2, 1, 1)
    result = g_run(g_table, tmp_path, origins=[18], settings=settings, candidates=candidates)
    expected = customer_forecasts(g_table, 18, settings, BLOCK_COLUMNS, list(F_COMPANIES))
    assert np.array_equal(result.p[0].to_numpy(), expected["p"].to_numpy())
    assert np.array_equal(result.m[0].to_numpy(), expected["m"].to_numpy())


def test_grid_a_fractional_origin_is_refused_before_any_fit(g_table, tmp_path, monkeypatch):
    # See test_evaluate: int(21.5) is 21, a week with rows; it must not be fitted at 21.
    assert refused(monkeypatch, g_table, tmp_path, origins=[18, 21.5]) == (
        "the table has no rows at origin week 21.5")


def test_grid_integral_float_origins_read_the_cache_of_plain_ones(g_table, g_cache, g_result,
                                                                 tmp_path, monkeypatch):
    result, first = g_result
    calls = count_fits(monkeypatch, fail=True)
    progress = []
    found = g_run(g_table, copied_cache(g_cache, tmp_path), origins=[22.0, np.float64(18.0)],
                  on_fit_done=lambda *args: progress.append(args))
    assert calls == []
    pd.testing.assert_frame_equal(found.rows, result.rows)
    pd.testing.assert_frame_equal(found.p, result.p)
    pd.testing.assert_frame_equal(found.m, result.m)
    assert [args[:5] for args in progress] == [args[:5] for args in first]
    assert all(type(args[3]) is int for args in progress)


# =============================================================================================
# U16.4: `GridGates`, `with_params` and `grid_gates`.
#
# A hand-made `GridPredictions` (seeded, no fit): two origins of 300 labelled rows each plus 30
# missing_week and 30 future rows, whose labels are NaN. True pay probability q per row;
# stage-1 candidates: 0 a flat 0.5 (the current settings, made bad on purpose), 1 q itself,
# 2 q shrunk 10% toward 0.5, 3 an over-confident q. Stage-2 candidates on the paid rows: 0 the
# training mean (no skill), 1 the true mean, 2 the true mean with noise, 3 twice the true mean.
# With the [grid] tolerances below, the eligible candidates are stage 1 {1, 2} and stage 2
# {1, 2}; [joint_diagnostic]'s (deliberately stricter) would leave stage 1 {1} and stage 2 {}.
# =============================================================================================

from scipy.special import expit, logit  # noqa: E402

from collection_estimation.hurdle.diagnostic_joint import (  # noqa: E402
    stage1_gates, stage2_gates)
from collection_estimation.hurdle.grid import GridGates, grid_gates, with_params  # noqa: E402
from collection_estimation.hurdle.panel import FUTURE, LABELLED, MISSING_WEEK  # noqa: E402

GATE_GRID = dataclasses.replace(GRID, stage1_log_loss_tolerance=0.1,
                                stage1_slope_band=(0.8, 1.25), stage2_skill_tolerance=0.1,
                                stage2_ratio_band=(0.9, 1.1))
# Not read by grid_gates: stricter, so using them would change the result.
GATE_JOINT = SimpleNamespace(stage1_log_loss_tolerance=0.0, stage1_slope_band=(0.95, 1.05),
                             stage2_skill_tolerance=0.01, stage2_ratio_band=(0.99, 1.01))
GATE_ORIGINS = (37, 40)
GATE_N = 300            # labelled rows per origin
GATE_UNLABELLED = 30    # missing_week rows and future rows, each, per origin


def gate_settings(**grid_changes):
    return SimpleNamespace(grid=dataclasses.replace(GATE_GRID, **grid_changes),
                           joint_diagnostic=GATE_JOINT)


def gate_candidates(n1=4, n2=4):
    stage1 = [Candidate(params={"max_iter": 100 * (i + 1), "learning_rate": 0.05},
                        settings=f"<stage1 {i}>", class_weight="none") for i in range(n1)]
    stage2 = [Candidate(params={"loss": "poisson", "max_iter": 50 * (j + 1)},
                        settings=f"<stage2 {j}>", class_weight="none") for j in range(n2)]
    return GridCandidates(stage1=stage1, stage2=stage2, topups=["<topups 0>"])


def gate_example(seed=5, poison_unlabelled=False):
    """GridPredictions as described above. With `poison_unlabelled`, the unlabelled rows carry
    a broken p and m for candidate 1 (p NaN, m -1). Since the "bad elsewhere" rule (F2 of
    U16.5) a p that is not a number in [0, 1] on a row the gates do not judge refuses the
    candidate, so the worked example is clean by default (it was poisoned by default before,
    when unlabelled rows were simply ignored)."""
    rng = np.random.default_rng(seed)
    frames, p_rows, m_rows = [], [], []
    start = 5000
    for origin in GATE_ORIGINS:
        n, u = GATE_N, GATE_UNLABELLED
        total = n + 2 * u
        index = pd.Index(start + np.arange(total))
        start += total
        q = rng.uniform(0.1, 0.9, total)
        paid = (rng.uniform(size=total) < q).astype(float)
        mu = rng.uniform(1e5, 1e6, total)
        amount = np.where(paid == 1, rng.gamma(4.0, mu / 4.0), 0.0)
        status = np.array([LABELLED] * n + [MISSING_WEEK] * u + [FUTURE] * u, dtype=object)
        paid[n:] = np.nan
        amount[n:] = np.nan
        labelled_paid = (status == LABELLED) & (paid == 1)
        base_rate = float(np.nanmean(paid[:n])) + (0.01 if origin == GATE_ORIGINS[0] else -0.01)
        training_mean = float(amount[labelled_paid].mean()) * (1.02 if origin == 40 else 0.98)
        frames.append(pd.DataFrame({
            "key": [f"C{i % 9}" for i in range(total)], "company_code": "TR02",
            "customer_id": [f"C{i % 9}" for i in range(total)], "origin_week": origin,
            "h": 1 + np.arange(total) % 5, "target_week": origin + 1 + np.arange(total) % 5,
            "label_status": status, "paid": paid, "amount_deflated": amount,
            "base_rate": base_rate, "training_mean": training_mean}, index=index))
        p = pd.DataFrame({0: np.full(total, 0.5), 1: q, 2: 0.5 + 0.9 * (q - 0.5),
                          3: expit(2.5 * logit(q))}, index=index)
        m = pd.DataFrame({0: np.full(total, training_mean), 1: mu,
                          2: mu * rng.uniform(0.7, 1.4, total), 3: 2 * mu}, index=index)
        if poison_unlabelled:
            p.iloc[n:, 1] = np.nan
            m.iloc[n:, 1] = -1.0
        p_rows.append(p)
        m_rows.append(m)
    rows = pd.concat(frames)
    return GridPredictions(rows=rows, p=pd.concat(p_rows), m=pd.concat(m_rows),
                           topups=[pd.DataFrame({"topup": [1.0]})])


def by_hand(predictions, s1_tol=0.1, s1_band=(0.8, 1.25), s2_tol=0.1, s2_band=(0.9, 1.1)):
    """stage1_gates / stage2_gates called directly on the pooled labelled (and paid) rows,
    with the "bad elsewhere" rule: a candidate with a p outside [0, 1] or not finite (an m not
    finite) on a row its stage does not judge is given NaN on every judged row, and the count
    of such rows is put right after "bad p" ("bad m")."""
    rows = predictions.rows
    lab = rows[rows["label_status"] == "labelled"]
    paid = lab[lab["paid"] == 1]
    other1 = rows.index.difference(lab.index)
    other2 = rows.index.difference(paid.index)
    p_values, p_bad, m_values, m_bad = [], [], [], []
    for i in predictions.p.columns:
        p = predictions.p.loc[other1, i].astype(float)
        bad = int(((~np.isfinite(p)) | (p < 0) | (p > 1)).sum())
        judged = predictions.p.loc[lab.index, i]
        p_values.append(judged * np.nan if bad else judged)
        p_bad.append(bad)
    for j in predictions.m.columns:
        bad = int((~np.isfinite(predictions.m.loc[other2, j].astype(float))).sum())
        judged = predictions.m.loc[paid.index, j]
        m_values.append(judged * np.nan if bad else judged)
        m_bad.append(bad)
    g1 = stage1_gates(lab["paid"].astype(bool), p_values, lab["base_rate"], s1_tol, s1_band)
    g2 = stage2_gates(paid["amount_deflated"], m_values, paid["training_mean"], s2_tol, s2_band)
    g1.insert(g1.columns.get_loc("bad p") + 1, "bad elsewhere", p_bad)
    g2.insert(g2.columns.get_loc("bad m") + 1, "bad elsewhere", m_bad)
    return g1, g2


def without_params(table, candidates):
    return table.drop(columns=list(candidates[0].params))


def eligible(table, number_column):
    return table.loc[table["eligible"], number_column].tolist()


def remade(predictions, rows=None, p=None, m=None, topups=None):
    """A copy of `predictions` with some parts replaced."""
    return GridPredictions(rows=predictions.rows if rows is None else rows,
                           p=predictions.p if p is None else p,
                           m=predictions.m if m is None else m,
                           topups=predictions.topups if topups is None else topups)


def only_rows(predictions, keep):
    return remade(predictions, rows=predictions.rows.loc[keep], p=predictions.p.loc[keep],
                  m=predictions.m.loc[keep])


@pytest.fixture(scope="module")
def gate_result():
    predictions = gate_example()
    candidates = gate_candidates()
    return predictions, candidates, grid_gates(predictions, candidates, gate_settings())


# --- GridGates -------------------------------------------------------------------------------

def test_grid_gates_type_fields_in_order_and_frozen(gate_result):
    assert [f.name for f in dataclasses.fields(GridGates)] == ["stage1", "stage2"]
    _, _, gates = gate_result
    assert isinstance(gates, GridGates)
    with pytest.raises(dataclasses.FrozenInstanceError):
        gates.stage1 = None


# --- with_params -----------------------------------------------------------------------------

def small_gates(numbers=(0, 1, 2)):
    return pd.DataFrame({"stage1": list(numbers), "log loss": [0.6, 0.5, 0.7][:len(numbers)],
                         "eligible": [True, False, True][:len(numbers)]})


def test_with_params_puts_the_params_right_after_the_number_column():
    found = with_params(small_gates(), gate_candidates(3, 1).stage1, "stage1")
    assert list(found.columns) == ["stage1", "max_iter", "learning_rate", "log loss", "eligible"]
    assert found["max_iter"].tolist() == [100, 200, 300]
    assert found["learning_rate"].tolist() == [0.05, 0.05, 0.05]
    assert found["log loss"].tolist() == [0.6, 0.5, 0.7]
    assert found["eligible"].tolist() == [True, False, True]


def test_with_params_reads_each_rows_own_candidate_by_number():
    # Numbers out of order: row k gets candidates[number of row k].
    gates = pd.DataFrame({"stage2": [2, 0], "score": [1.0, 2.0]})
    found = with_params(gates, gate_candidates(1, 3).stage2, "stage2")
    assert found["max_iter"].tolist() == [150, 50]
    assert found["stage2"].tolist() == [2, 0]
    assert found["loss"].tolist() == ["poisson", "poisson"]


def test_with_params_numpy_integer_numbers_work():
    gates = small_gates()
    gates["stage1"] = gates["stage1"].astype("int32")
    found = with_params(gates, gate_candidates(3, 1).stage1, "stage1")
    assert found["max_iter"].tolist() == [100, 200, 300]


def test_with_params_does_not_change_the_gate_table():
    gates = small_gates()
    before = gates.copy()
    with_params(gates, gate_candidates(3, 1).stage1, "stage1")
    pd.testing.assert_frame_equal(gates, before)


def test_with_params_a_number_past_the_candidate_list_raises():
    with pytest.raises(IndexError):
        with_params(small_gates((0, 5, 1)), gate_candidates(3, 1).stage1, "stage1")


def test_with_params_is_defined_here_not_in_the_pipeline_script():
    # Moved out of run_hurdle_pipeline.py (step 15) in U16.4: one definition, here.
    source = (Path(__file__).resolve().parents[2] / "run_hurdle_pipeline.py").read_text(
        encoding="utf-8")
    assert "def with_params(" not in source
    assert "    with_params,\n" in source


def test_with_params_keeps_each_rows_params_when_the_table_is_reordered():
    # A sorted gate table (index 2, 1, 0) keeps each row's own params (fixed in U16.4: the
    # params frame takes the gate table's index).
    gates = small_gates().iloc[::-1]
    found = with_params(gates, gate_candidates(3, 1).stage1, "stage1")
    pairs = list(zip(found["stage1"], found["max_iter"]))
    assert pairs == [(2, 300), (1, 200), (0, 100)]
    assert found.index.tolist() == [2, 1, 0]
    assert found["log loss"].tolist() == [0.7, 0.5, 0.6]
    assert len(found) == 3 and not found.isna().any().any()


def test_with_params_keeps_each_rows_params_when_the_table_is_filtered():
    # Only the eligible rows of a 6-candidate table: a non-contiguous index (1, 3, 4).
    gates = pd.DataFrame({"stage1": [0, 1, 2, 3, 4, 5],
                          "log loss": [0.6, 0.5, 0.7, 0.55, 0.52, 0.9],
                          "eligible": [False, True, False, True, True, False]})
    kept = gates[gates["eligible"]]
    found = with_params(kept, gate_candidates(6, 1).stage1, "stage1")
    assert found.index.tolist() == [1, 3, 4]
    assert list(zip(found["stage1"], found["max_iter"])) == [(1, 200), (3, 400), (4, 500)]
    assert found["log loss"].tolist() == [0.5, 0.55, 0.52]
    assert list(found.columns) == ["stage1", "max_iter", "learning_rate", "log loss",
                                   "eligible"]
    assert len(found) == 3 and not found.isna().any().any()


def test_with_params_filtered_and_sorted_table_keeps_its_index_and_params():
    gates = pd.DataFrame({"stage2": [0, 1, 2, 3], "deviance skill": [0.0, 0.53, 0.48, -0.5],
                          "eligible": [False, True, True, False]})
    kept = gates[gates["eligible"]].sort_values("deviance skill")      # index 2, 1
    found = with_params(kept, gate_candidates(1, 4).stage2, "stage2")
    assert found.index.tolist() == [2, 1]
    assert list(zip(found["stage2"], found["max_iter"])) == [(2, 150), (1, 100)]
    assert found["deviance skill"].tolist() == [0.48, 0.53]


def test_with_params_a_non_integer_index_is_kept():
    gates = small_gates().set_axis(["a", "b", "c"])
    found = with_params(gates, gate_candidates(3, 1).stage1, "stage1")
    assert found.index.tolist() == ["a", "b", "c"]
    assert found["max_iter"].tolist() == [100, 200, 300]


def test_with_params_on_grid_gates_output_filtered_to_the_eligible_rows(gate_result):
    # grid_gates' own table, filtered: the same params as in the full table, row by row.
    predictions, candidates, gates = gate_result
    g1, _ = by_hand(predictions)
    kept = g1[g1["eligible"]]
    found = with_params(kept, candidates.stage1, "stage1")
    pd.testing.assert_frame_equal(found, gates.stage1[gates.stage1["eligible"]])


# --- grid_gates: the result ----------------------------------------------------------------

def test_grid_gates_the_worked_example_eligibility(gate_result):
    _, _, gates = gate_result
    assert eligible(gates.stage1, "stage1") == [1, 2]
    assert eligible(gates.stage2, "stage2") == [1, 2]
    # The current settings (candidate 0) fail in both stages.
    assert not gates.stage1.loc[0, "eligible"] and not gates.stage2.loc[0, "eligible"]


def test_grid_gates_tables_have_one_row_per_candidate_in_number_order(gate_result):
    _, _, gates = gate_result
    assert gates.stage1["stage1"].tolist() == [0, 1, 2, 3]
    assert gates.stage2["stage2"].tolist() == [0, 1, 2, 3]


def test_grid_gates_columns_are_number_params_then_the_gate_columns(gate_result):
    _, _, gates = gate_result
    assert list(gates.stage1.columns) == [
        "stage1", "max_iter", "learning_rate", "bad p", "bad elsewhere", "log loss",
        "log loss skill", "slope", "log loss gate", "skill gate", "slope gate", "eligible"]
    assert list(gates.stage2.columns) == [
        "stage2", "loss", "max_iter", "bad m", "bad elsewhere", "deviance skill", "total ratio",
        "m gate", "skill gate", "ratio gate", "eligible"]
    # The worked example is clean: nothing is broken outside the judged rows.
    assert (gates.stage1["bad elsewhere"] == 0).all() and (gates.stage2["bad elsewhere"] == 0).all()


def test_grid_gates_params_are_each_candidates_own(gate_result):
    _, candidates, gates = gate_result
    for table, listed, column in ((gates.stage1, candidates.stage1, "stage1"),
                                  (gates.stage2, candidates.stage2, "stage2")):
        for _, row in table.iterrows():
            params = listed[int(row[column])].params
            assert {key: row[key] for key in params} == params


def test_grid_gates_equal_the_gate_functions_on_the_pooled_rows(gate_result):
    predictions, candidates, gates = gate_result
    g1, g2 = by_hand(predictions)
    pd.testing.assert_frame_equal(without_params(gates.stage1, candidates.stage1), g1)
    pd.testing.assert_frame_equal(without_params(gates.stage2, candidates.stage2), g2)


def test_grid_gates_equal_with_params_of_the_gate_functions(gate_result):
    predictions, candidates, gates = gate_result
    g1, g2 = by_hand(predictions)
    pd.testing.assert_frame_equal(gates.stage1, with_params(g1, candidates.stage1, "stage1"))
    pd.testing.assert_frame_equal(gates.stage2, with_params(g2, candidates.stage2, "stage2"))


def test_grid_gates_pool_the_origins_rather_than_judging_one(gate_result):
    # The pooled table differs from either origin alone (its scores are over all rows).
    predictions, candidates, gates = gate_result
    for origin in GATE_ORIGINS:
        keep = predictions.rows.index[predictions.rows["origin_week"] == origin]
        one = grid_gates(only_rows(predictions, keep), candidates, gate_settings())
        assert not np.allclose(one.stage1["log loss"], gates.stage1["log loss"])
        assert not np.allclose(one.stage2["deviance skill"], gates.stage2["deviance skill"])


def test_grid_gates_stage2_is_judged_on_the_paid_labelled_rows_only(gate_result):
    # Candidate 0's m is the training mean: its total ratio is sum(training mean) / sum(paid
    # amount) over the labelled rows that paid, nothing else.
    predictions, _, gates = gate_result
    rows = predictions.rows
    paid = rows[(rows["label_status"] == "labelled") & (rows["paid"] == 1)]
    expected = paid["training_mean"].sum() / paid["amount_deflated"].sum()
    assert gates.stage2.loc[0, "total ratio"] == pytest.approx(expected, rel=1e-12)


def test_grid_gates_stage1_reads_each_rows_own_base_rate(gate_result):
    # The two origins carry different base rates; skill is against each row's own.
    predictions, _, gates = gate_result
    rows = predictions.rows
    lab = rows[rows["label_status"] == "labelled"]
    assert lab["base_rate"].nunique() == 2
    flat = stage1_gates(lab["paid"].astype(bool), [predictions.p.loc[lab.index, 1]],
                        pd.Series(lab["base_rate"].mean(), index=lab.index), 0.1, (0.8, 1.25))
    assert gates.stage1.loc[1, "log loss skill"] != pytest.approx(flat.loc[0, "log loss skill"],
                                                                  rel=1e-12)


# --- grid_gates: "the best" is the grid's best, not the current settings --------------------

def test_grid_gates_stage1_log_loss_limit_is_the_grids_best_not_candidate_0(gate_result):
    _, _, gates = gate_result
    table = gates.stage1
    best = table["log loss"].min()
    assert table["log loss"].idxmin() != 0
    assert table["log loss gate"].tolist() == (table["log loss"] <= best * 1.1).tolist()
    # Measured against candidate 0 (log loss 0.69), candidate 3 (0.72) would pass; it fails.
    assert table.loc[3, "log loss"] <= table.loc[0, "log loss"] * 1.1
    assert not table.loc[3, "log loss gate"] and not table.loc[0, "log loss gate"]


def test_grid_gates_stage2_skill_limit_is_the_grids_best_not_candidate_0(gate_result):
    _, _, gates = gate_result
    table = gates.stage2
    best = table["deviance skill"].max()
    assert table["deviance skill"].idxmax() != 0
    assert table["skill gate"].tolist() == (table["deviance skill"] >= best - 0.1).tolist()
    # Candidate 0's skill is about 0 (it is nearly its own reference); against itself as the
    # best it would pass the skill gate. It fails.
    assert abs(table.loc[0, "deviance skill"]) < 0.01
    assert not table.loc[0, "skill gate"]


def test_grid_gates_reordering_the_candidates_reorders_the_results_only():
    # The best is found wherever it is: candidate lists and columns reversed together give
    # the same verdict per candidate.
    predictions, candidates = gate_example(), gate_candidates()
    order = [3, 2, 1, 0]
    flipped = remade(predictions, p=predictions.p[order].set_axis(range(4), axis=1),
                     m=predictions.m[order].set_axis(range(4), axis=1))
    flipped_candidates = GridCandidates(stage1=[candidates.stage1[i] for i in order],
                                        stage2=[candidates.stage2[i] for i in order],
                                        topups=candidates.topups)
    straight = grid_gates(predictions, candidates, gate_settings())
    reverse = grid_gates(flipped, flipped_candidates, gate_settings())
    for name in ("stage1", "stage2"):
        a = getattr(straight, name).drop(columns=name).set_index("max_iter").sort_index()
        b = getattr(reverse, name).drop(columns=name).set_index("max_iter").sort_index()
        pd.testing.assert_frame_equal(a, b)


# --- grid_gates: the [grid] tolerances, not [joint_diagnostic]'s --------------------------

def test_grid_gates_use_the_grid_tolerances(gate_result):
    predictions, _, gates = gate_result
    grid_way = by_hand(predictions)
    joint_way = by_hand(predictions, 0.0, (0.95, 1.05), 0.01, (0.99, 1.01))
    # The two settings really disagree on this example ...
    assert eligible(joint_way[0], "stage1") == [1] and eligible(joint_way[1], "stage2") == []
    # ... and grid_gates follows [grid].
    assert eligible(gates.stage1, "stage1") == eligible(grid_way[0], "stage1") == [1, 2]
    assert eligible(gates.stage2, "stage2") == eligible(grid_way[1], "stage2") == [1, 2]


@pytest.mark.parametrize("field, value, stage, expected", [
    ("stage1_log_loss_tolerance", 0.0, "stage1", [1]),
    ("stage1_slope_band", (0.95, 1.05), "stage1", [1]),
    ("stage1_slope_band", (1.0, 1.25), "stage1", [2]),
    ("stage2_skill_tolerance", 0.01, "stage2", [1]),
    ("stage2_ratio_band", (0.99, 1.01), "stage2", [2]),
    ("stage2_ratio_band", (0.5, 2.5), "stage2", [1, 2]),
])
def test_grid_gates_each_grid_tolerance_reaches_its_gate(field, value, stage, expected):
    gates = grid_gates(gate_example(), gate_candidates(), gate_settings(**{field: value}))
    assert eligible(getattr(gates, stage), stage) == expected


def test_grid_gates_read_only_settings_grid():
    # A stand-in with nothing but .grid is enough.
    gates = grid_gates(gate_example(), gate_candidates(), SimpleNamespace(grid=GATE_GRID))
    assert eligible(gates.stage1, "stage1") == [1, 2]
    assert eligible(gates.stage2, "stage2") == [1, 2]


# --- grid_gates: rows that are not labelled are ignored ---------------------------------------

def labelled_only(predictions):
    return only_rows(predictions,
                     predictions.rows.index[predictions.rows["label_status"] == "labelled"])


def test_grid_gates_unlabelled_rows_change_nothing(gate_result):
    predictions, candidates, gates = gate_result
    assert (predictions.rows["label_status"] != "labelled").sum() == 4 * GATE_UNLABELLED
    alone = grid_gates(labelled_only(predictions), candidates, gate_settings())
    pd.testing.assert_frame_equal(gates.stage1, alone.stage1)
    pd.testing.assert_frame_equal(gates.stage2, alone.stage2)


def test_grid_gates_a_nan_p_on_unlabelled_rows_refuses_the_candidate():
    # Reversed in U16.5 (F2): candidate 1's p is NaN on every unlabelled row (120 of them).
    # It used to be eligible (unlabelled rows were not read); now it fails every gate, since
    # grid_scores forecasts those rows too.
    predictions = gate_example(poison_unlabelled=True)
    unlabelled = predictions.rows["label_status"] != "labelled"
    assert predictions.p.loc[unlabelled, 1].isna().all()
    gates = grid_gates(predictions, gate_candidates(), gate_settings()).stage1
    row = gates.loc[1]
    assert row["bad elsewhere"] == 4 * GATE_UNLABELLED
    assert not row["log loss gate"] and not row["skill gate"] and not row["slope gate"]
    assert not row["eligible"]
    # Judged as NaN on every labelled row: no score.
    assert row["bad p"] == 2 * GATE_N and np.isnan(row["log loss"])
    # The others: no row broken elsewhere.
    assert (gates.drop(index=1)["bad elsewhere"] == 0).all()


def test_grid_gates_a_negative_m_on_unlabelled_rows_is_not_bad_elsewhere():
    # Stage 2's rule is "not finite" only: candidate 1's m of -1 on the unlabelled rows is not
    # counted, and it stays eligible (as the U16.5 rule is written; see the report).
    predictions = gate_example(poison_unlabelled=True)
    unlabelled = predictions.rows["label_status"] != "labelled"
    assert (predictions.m.loc[unlabelled, 1] == -1).all()
    gates = grid_gates(predictions, gate_candidates(), gate_settings()).stage2
    assert gates.loc[1, "bad elsewhere"] == 0 and gates.loc[1, "bad m"] == 0
    assert gates.loc[1, "eligible"]


def test_grid_gates_a_poisoned_example_equals_the_rule_by_hand():
    predictions = gate_example(poison_unlabelled=True)
    candidates = gate_candidates()
    gates = grid_gates(predictions, candidates, gate_settings())
    g1, g2 = by_hand(predictions)
    pd.testing.assert_frame_equal(gates.stage1, with_params(g1, candidates.stage1, "stage1"))
    pd.testing.assert_frame_equal(gates.stage2, with_params(g2, candidates.stage2, "stage2"))
    # Candidate 1, the best of stage 1, is refused: the best passes to candidate 2.
    assert eligible(gates.stage1, "stage1") == [2]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1.5, -0.2])
@pytest.mark.parametrize("status", [MISSING_WEEK, FUTURE])
def test_grid_gates_one_bad_p_on_one_unlabelled_row_refuses_only_its_candidate(bad, status):
    predictions, candidates = gate_example(), gate_candidates()
    rows = predictions.rows
    target = rows.index[rows["label_status"] == status][0]
    predictions.p.loc[target, 3] = bad
    gates = grid_gates(predictions, candidates, gate_settings()).stage1
    assert gates.loc[3, "bad elsewhere"] == 1 and not gates.loc[3, "eligible"]
    assert not gates.loc[3, ["log loss gate", "skill gate", "slope gate"]].any()
    # Candidate 3 was not eligible anyway; the others are untouched.
    clean = grid_gates(gate_example(), candidates, gate_settings()).stage1
    pd.testing.assert_frame_equal(gates.drop(index=3), clean.drop(index=3))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_grid_gates_a_p_of_exactly_0_or_1_on_an_unlabelled_row_is_not_bad_elsewhere(value):
    predictions = gate_example()
    unlabelled = predictions.rows.index[predictions.rows["label_status"] != "labelled"]
    predictions.p.loc[unlabelled, 2] = value
    gates = grid_gates(predictions, gate_candidates(), gate_settings()).stage1
    assert gates.loc[2, "bad elsewhere"] == 0 and gates.loc[2, "eligible"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_grid_gates_a_non_finite_m_on_an_unlabelled_row_refuses_its_candidate(bad):
    predictions = gate_example()
    target = predictions.rows.index[predictions.rows["label_status"] == FUTURE][0]
    predictions.m.loc[target, 2] = bad
    gates = grid_gates(predictions, gate_candidates(), gate_settings()).stage2
    assert gates.loc[2, "bad elsewhere"] == 1
    assert not gates.loc[2, ["m gate", "skill gate", "ratio gate"]].any()
    assert eligible(gates, "stage2") == [1]


def test_grid_gates_a_broken_best_elsewhere_does_not_set_the_best():
    # Candidate 1 is the best of both stages; broken on one future row in each stage, it is
    # refused and candidate 2 becomes the best, as when it is broken on a judged row.
    predictions = gate_example()
    target = predictions.rows.index[predictions.rows["label_status"] == FUTURE][0]
    predictions.p.loc[target, 1] = np.nan
    predictions.m.loc[target, 1] = np.inf
    gates = grid_gates(predictions, gate_candidates(), gate_settings())
    assert eligible(gates.stage1, "stage1") == [2]
    assert eligible(gates.stage2, "stage2") == [2]


def test_grid_gates_the_current_broken_elsewhere_fails_its_gates_and_is_still_scored():
    predictions = gate_example()
    target = predictions.rows.index[predictions.rows["label_status"] == MISSING_WEEK][0]
    predictions.p.loc[target, 0] = np.nan
    gates = grid_gates(predictions, gate_candidates(), gate_settings())
    assert gates.stage1.loc[0, "bad elsewhere"] == 1 and not gates.stage1.loc[0, "eligible"]
    assert scored_numbers(gates.stage1, "stage1") == [0, 1, 2]


def test_grid_gates_bad_p_counts_judged_rows_and_bad_elsewhere_the_others():
    # One NaN on a labelled row and two on unlabelled rows of candidate 2.
    predictions = gate_example()
    rows = predictions.rows
    predictions.p.loc[first_labelled(predictions), 2] = np.nan
    predictions.p.loc[rows.index[rows["label_status"] != "labelled"][:2], 2] = np.nan
    row = grid_gates(predictions, gate_candidates(), gate_settings()).stage1.loc[2]
    assert row["bad elsewhere"] == 2 and not row["eligible"]


@pytest.mark.parametrize("status", [MISSING_WEEK, FUTURE])
def test_grid_gates_each_unlabelled_status_alone_is_ignored(status):
    predictions = gate_example()
    rows = predictions.rows
    part = only_rows(predictions, rows.index[rows["label_status"].isin([LABELLED, status])])
    a = grid_gates(part, gate_candidates(), gate_settings())
    b = grid_gates(labelled_only(predictions), gate_candidates(), gate_settings())
    pd.testing.assert_frame_equal(a.stage1, b.stage1)
    pd.testing.assert_frame_equal(a.stage2, b.stage2)


def test_grid_gates_a_labelled_row_turned_unlabelled_drops_out():
    # Relabelling one paid row as missing_week is the same as removing it.
    predictions = gate_example()
    rows = predictions.rows.copy()
    target = rows.index[(rows["label_status"] == "labelled") & (rows["paid"] == 1)][0]
    rows.loc[target, "label_status"] = MISSING_WEEK
    a = grid_gates(remade(predictions, rows=rows), gate_candidates(), gate_settings())
    b = grid_gates(only_rows(predictions, predictions.rows.index.drop(target)),
                   gate_candidates(), gate_settings())
    pd.testing.assert_frame_equal(a.stage1, b.stage1)
    pd.testing.assert_frame_equal(a.stage2, b.stage2)


def test_grid_gates_a_bad_reference_on_an_unlabelled_row_is_ignored(gate_result):
    predictions, candidates, gates = gate_result
    rows = predictions.rows.copy()
    rows.loc[rows["label_status"] != "labelled", ["training_mean", "base_rate"]] = np.nan
    found = grid_gates(remade(predictions, rows=rows), candidates, gate_settings())
    pd.testing.assert_frame_equal(found.stage1, gates.stage1)
    pd.testing.assert_frame_equal(found.stage2, gates.stage2)


# --- grid_gates: a broken candidate fails its gates, the others go on -----------------------

def drop_candidate(predictions, candidates, stage, number):
    """predictions and candidates without one candidate of `stage`, renumbered."""
    frame = predictions.p if stage == "stage1" else predictions.m
    kept = [c for c in frame.columns if c != number]
    frame = frame[kept].set_axis(range(len(kept)), axis=1)
    listed = [c for i, c in enumerate(getattr(candidates, stage)) if i != number]
    if stage == "stage1":
        return (remade(predictions, p=frame),
                GridCandidates(stage1=listed, stage2=candidates.stage2,
                               topups=candidates.topups))
    return (remade(predictions, m=frame),
            GridCandidates(stage1=candidates.stage1, stage2=listed, topups=candidates.topups))


def first_labelled(predictions, paid=None):
    rows = predictions.rows
    mask = rows["label_status"] == "labelled"
    if paid is not None:
        mask &= rows["paid"] == (1 if paid else 0)
    return rows.index[mask][0]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1.5, -0.2])
@pytest.mark.parametrize("number", [0, 1, 3])
def test_grid_gates_a_bad_p_fails_only_its_own_candidate(bad, number):
    predictions, candidates = gate_example(), gate_candidates()
    predictions.p.loc[first_labelled(predictions), number] = bad
    gates = grid_gates(predictions, candidates, gate_settings()).stage1
    row = gates.loc[number]
    assert row["bad p"] == 1
    assert not row["log loss gate"] and not row["skill gate"] and not row["slope gate"]
    assert not row["eligible"]
    # The others are judged as if the broken one were not in the grid at all.
    others = gates.drop(index=number).reset_index(drop=True)
    p_less, c_less = drop_candidate(predictions, candidates, "stage1", number)
    expected = grid_gates(p_less, c_less, gate_settings()).stage1
    pd.testing.assert_frame_equal(others.drop(columns="stage1"),
                                  expected.drop(columns="stage1"))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -1.0])
@pytest.mark.parametrize("number", [0, 1, 3])
def test_grid_gates_a_bad_m_fails_only_its_own_candidate(bad, number):
    predictions, candidates = gate_example(), gate_candidates()
    predictions.m.loc[first_labelled(predictions, paid=True), number] = bad
    gates = grid_gates(predictions, candidates, gate_settings()).stage2
    row = gates.loc[number]
    assert row["bad m"] == 1
    assert not row["m gate"] and not row["skill gate"] and not row["ratio gate"]
    assert not row["eligible"]
    others = gates.drop(index=number).reset_index(drop=True)
    p_less, c_less = drop_candidate(predictions, candidates, "stage2", number)
    expected = grid_gates(p_less, c_less, gate_settings()).stage2
    pd.testing.assert_frame_equal(others.drop(columns="stage2"),
                                  expected.drop(columns="stage2"))


def test_grid_gates_a_broken_best_candidate_hands_the_best_to_the_next():
    # Candidate 1 (the best of both stages) broken: candidate 2 becomes the best and passes.
    predictions = gate_example()
    predictions.p.loc[first_labelled(predictions), 1] = np.nan
    predictions.m.loc[first_labelled(predictions, paid=True), 1] = np.nan
    gates = grid_gates(predictions, gate_candidates(), gate_settings())
    assert eligible(gates.stage1, "stage1") == [2]
    assert eligible(gates.stage2, "stage2") == [2]


def test_grid_gates_a_nan_m_on_a_labelled_unpaid_row_is_bad_elsewhere():
    # Reversed in U16.5 (F2): stage 2 judges the paid rows only, so a NaN m on a labelled row
    # that did not pay is "elsewhere"; it used to be ignored, now it refuses the candidate.
    predictions = gate_example()
    rows = predictions.rows
    unpaid = rows.index[(rows["label_status"] == "labelled") & (rows["paid"] == 0)]
    predictions.m.loc[unpaid, 2] = np.nan
    gates = grid_gates(predictions, gate_candidates(), gate_settings()).stage2
    assert gates.loc[2, "bad elsewhere"] == len(unpaid)
    assert not gates.loc[2, "eligible"]
    assert eligible(gates, "stage2") == [1]


def test_grid_gates_a_zero_m_on_a_labelled_unpaid_row_is_not_bad():
    # 0 is a finite m: neither "bad m" (not judged there) nor "bad elsewhere".
    predictions = gate_example()
    rows = predictions.rows
    unpaid = rows.index[(rows["label_status"] == "labelled") & (rows["paid"] == 0)]
    predictions.m.loc[unpaid, 2] = 0.0
    gates = grid_gates(predictions, gate_candidates(), gate_settings()).stage2
    assert gates.loc[2, "bad m"] == 0 and gates.loc[2, "bad elsewhere"] == 0
    assert gates.loc[2, "eligible"]


def test_grid_gates_a_p_broken_on_a_labelled_row_is_bad_p_not_bad_elsewhere():
    # Stage 1 judges every labelled row, paid or not.
    predictions = gate_example()
    predictions.p.loc[first_labelled(predictions, paid=False), 2] = np.nan
    row = grid_gates(predictions, gate_candidates(), gate_settings()).stage1.loc[2]
    assert row["bad p"] == 1 and row["bad elsewhere"] == 0


def test_grid_gates_every_candidate_broken_still_returns_tables():
    predictions = gate_example()
    predictions.p.loc[first_labelled(predictions)] = np.nan
    predictions.m.loc[first_labelled(predictions, paid=True)] = 0.0
    gates = grid_gates(predictions, gate_candidates(), gate_settings())
    assert not gates.stage1["eligible"].any() and not gates.stage2["eligible"].any()
    assert (gates.stage1["bad p"] == 1).all() and (gates.stage2["bad m"] == 1).all()


def test_grid_gates_a_p_of_exactly_0_or_1_is_not_bad():
    # The ends of [0, 1] are probabilities: not "bad p".
    predictions = gate_example()
    predictions.p.loc[first_labelled(predictions, paid=True), 2] = 1.0
    predictions.p.loc[first_labelled(predictions, paid=False), 2] = 0.0
    gates = grid_gates(predictions, gate_candidates(), gate_settings()).stage1
    assert gates.loc[2, "bad p"] == 0
    assert np.isfinite(gates.loc[2, "log loss"])


# --- grid_gates: sizes, single candidates, top-ups -------------------------------------------

@pytest.mark.parametrize("n1, n2", [(1, 1), (1, 4), (4, 1), (2, 3)])
def test_grid_gates_one_row_per_candidate_for_any_counts(n1, n2):
    predictions = gate_example()
    predictions = remade(predictions, p=predictions.p[list(range(n1))],
                         m=predictions.m[list(range(n2))])
    gates = grid_gates(predictions, gate_candidates(n1, n2), gate_settings())
    assert gates.stage1["stage1"].tolist() == list(range(n1))
    assert gates.stage2["stage2"].tolist() == list(range(n2))


def test_grid_gates_a_lone_candidate_is_its_own_best():
    # Only candidate 1 in each stage: the log-loss and skill limits are its own scores.
    predictions = gate_example()
    lone = remade(predictions, p=predictions.p[[1]].set_axis([0], axis=1),
                  m=predictions.m[[1]].set_axis([0], axis=1), topups=[])
    gates = grid_gates(lone, gate_candidates(1, 1), gate_settings())
    assert gates.stage1.loc[0, "log loss gate"] and gates.stage1.loc[0, "eligible"]
    assert gates.stage2.loc[0, "skill gate"] and gates.stage2.loc[0, "eligible"]


@pytest.mark.parametrize("topups", [[], ["a"], ["a", "b", "c"]])
def test_grid_gates_ignore_the_topups(gate_result, topups):
    # Top-ups have no gate: their count and tables do not matter.
    predictions, candidates, gates = gate_result
    other = GridCandidates(stage1=candidates.stage1, stage2=candidates.stage2, topups=topups)
    found = grid_gates(remade(predictions, topups=[]), other, gate_settings())
    assert [f.name for f in dataclasses.fields(found)] == ["stage1", "stage2"]
    pd.testing.assert_frame_equal(found.stage1, gates.stage1)
    pd.testing.assert_frame_equal(found.stage2, gates.stage2)


def test_grid_gates_paid_given_as_bool_or_float_is_the_same(gate_result):
    predictions, candidates, gates = gate_result
    rows = predictions.rows.copy()
    lab = rows["label_status"] == "labelled"
    rows["paid"] = rows["paid"].astype(object)
    rows.loc[lab, "paid"] = [bool(v) for v in rows.loc[lab, "paid"]]
    found = grid_gates(remade(predictions, rows=rows), candidates, gate_settings())
    pd.testing.assert_frame_equal(found.stage1, gates.stage1)
    pd.testing.assert_frame_equal(found.stage2, gates.stage2)


# --- grid_gates: purity --------------------------------------------------------------------

def test_grid_gates_are_deterministic_and_change_nothing():
    predictions, candidates, settings = gate_example(), gate_candidates(), gate_settings()
    before = (predictions.rows.copy(), predictions.p.copy(), predictions.m.copy(),
              [dict(c.params) for c in candidates.stage1 + candidates.stage2], settings.grid)
    a = grid_gates(predictions, candidates, settings)
    b = grid_gates(predictions, candidates, settings)
    pd.testing.assert_frame_equal(a.stage1, b.stage1)
    pd.testing.assert_frame_equal(a.stage2, b.stage2)
    pd.testing.assert_frame_equal(predictions.rows, before[0])
    pd.testing.assert_frame_equal(predictions.p, before[1])
    pd.testing.assert_frame_equal(predictions.m, before[2])
    assert [c.params for c in candidates.stage1 + candidates.stage2] == before[3]
    assert settings.grid is before[4]


def test_grid_gates_tables_are_not_views_of_the_predictions():
    predictions = gate_example()
    gates = grid_gates(predictions, gate_candidates(), gate_settings())
    p_before, m_before = predictions.p.copy(), predictions.m.copy()
    gates.stage1.loc[:, "log loss"] = -1.0
    gates.stage2.loc[:, "deviance skill"] = -1.0
    pd.testing.assert_frame_equal(predictions.p, p_before)
    pd.testing.assert_frame_equal(predictions.m, m_before)


# --- grid_gates: refused --------------------------------------------------------------------

@pytest.mark.parametrize("n1", [3, 5])
def test_grid_gates_stage1_count_mismatch_raises(n1):
    with pytest.raises(ValueError) as error:
        grid_gates(gate_example(), gate_candidates(n1, 4), gate_settings())
    assert str(error.value) == f"the predictions hold 4 stage-1 candidates, the candidates {n1}"


@pytest.mark.parametrize("n2", [3, 5])
def test_grid_gates_stage2_count_mismatch_raises(n2):
    with pytest.raises(ValueError) as error:
        grid_gates(gate_example(), gate_candidates(4, n2), gate_settings())
    assert str(error.value) == f"the predictions hold 4 stage-2 candidates, the candidates {n2}"


def test_grid_gates_stage1_count_is_checked_before_stage2():
    with pytest.raises(ValueError, match="stage-1 candidates"):
        grid_gates(gate_example(), gate_candidates(2, 2), gate_settings())


def test_grid_gates_count_check_comes_before_the_label_check():
    predictions = gate_example()
    rows = predictions.rows.assign(label_status=FUTURE)
    with pytest.raises(ValueError, match="stage-2 candidates"):
        grid_gates(remade(predictions, rows=rows), gate_candidates(4, 3), gate_settings())


@pytest.mark.parametrize("statuses", [[FUTURE], [MISSING_WEEK], [FUTURE, MISSING_WEEK]])
def test_grid_gates_no_labelled_row_raises(statuses):
    predictions = gate_example()
    rows = predictions.rows.copy()
    rows["label_status"] = [statuses[i % len(statuses)] for i in range(len(rows))]
    with pytest.raises(ValueError) as error:
        grid_gates(remade(predictions, rows=rows), gate_candidates(), gate_settings())
    assert str(error.value) == ("no row of the predictions is labelled: the gates have "
                                "nothing to judge")


def test_grid_gates_empty_predictions_raise_no_labelled_row():
    predictions = gate_example()
    with pytest.raises(ValueError, match="no row of the predictions is labelled"):
        grid_gates(only_rows(predictions, predictions.rows.index[:0]), gate_candidates(),
                   gate_settings())


def test_grid_gates_no_labelled_row_paid_raises():
    # Rows marked paid exist, but only among the unlabelled ones: the paid rows are taken
    # from the labelled ones.
    predictions = gate_example()
    rows = predictions.rows.copy()
    lab = rows["label_status"] == "labelled"
    rows.loc[lab, "paid"] = 0.0
    rows.loc[~lab, "paid"] = 1.0
    with pytest.raises(ValueError) as error:
        grid_gates(remade(predictions, rows=rows), gate_candidates(), gate_settings())
    assert str(error.value) == ("no labelled row of the predictions paid: stage 2 cannot be "
                                "judged")


def test_grid_gates_every_labelled_row_paid_raises_as_stage1_gates_does():
    # One outcome only: stage 1 cannot be judged; stage1_gates' own error comes through.
    predictions = gate_example()
    rows = predictions.rows.copy()
    lab = rows["label_status"] == "labelled"
    rows.loc[lab, "paid"] = 1.0
    rows.loc[lab & ~(rows["amount_deflated"] > 0), "amount_deflated"] = 1000.0
    with pytest.raises(ValueError, match="both outcomes"):
        grid_gates(remade(predictions, rows=rows), gate_candidates(), gate_settings())


def test_grid_gates_a_bad_training_mean_on_a_paid_row_raises_as_stage2_gates_does():
    predictions = gate_example()
    rows = predictions.rows.copy()
    rows.loc[first_labelled(predictions, paid=True), "training_mean"] = 0.0
    with pytest.raises(ValueError, match="reference must be finite and above 0"):
        grid_gates(remade(predictions, rows=rows), gate_candidates(), gate_settings())


# --- grid_gates on the U16.3 grid itself ----------------------------------------------------

def test_grid_gates_run_on_grid_predictions_output(g_result):
    result, _ = g_result
    settings = g_settings()
    settings.grid = GATE_GRID
    candidates = g_candidates(settings)
    gates = grid_gates(result, candidates, settings)
    assert gates.stage1["stage1"].tolist() == [0, 1]
    assert gates.stage2["stage2"].tolist() == [0, 1]
    assert gates.stage1.columns[1] == "max_iter"
    assert gates.stage2.columns[1] == "learning_rate"
    g1, g2 = by_hand(result)
    pd.testing.assert_frame_equal(gates.stage1, with_params(g1, candidates.stage1, "stage1"))
    pd.testing.assert_frame_equal(gates.stage2, with_params(g2, candidates.stage2, "stage2"))
    # The U16.3 grid holds future rows (labels NaN); they were left out without trouble.
    assert (result.rows["label_status"] == "future").any()
    assert (gates.stage1["bad p"] == 0).all() and (gates.stage2["bad m"] == 0).all()


# =============================================================================================
# U16.5: `GRID_SCORE_COLUMNS`, `scored_numbers`, `combination_name` and `grid_scores`.
#
# Mostly on a hand-made `GridPredictions` (seeded, nothing fitted): companies TR02 and TR04,
# origins 18 and 22, h 1..5, three customers per company; p and m columns per candidate; top-up
# tables in `topup_forecasts`' shape. The actual cash is `actual_totals` on a seeded weekly
# table. The gates are hand-made tables (`scored_numbers` reads only the number column and
# "eligible"). The scores are checked against an independent computation by hand (sums, the
# projected level, WAPE / bias formulas). At the end, on the U16.3 grid itself: combination
# (0, 0, 0) scores exactly as `rolling_forecasts` with the same settings.
# =============================================================================================

from datetime import date  # noqa: E402

from collection_estimation.hurdle.evaluate import (  # noqa: E402
    HORIZON_TOTAL, HURDLE, POOLED, STACK_COLUMNS, actual_totals, evaluation_scores,
    rolling_forecasts)
from collection_estimation.hurdle.forecast import CONSOLIDATED, forecast_totals  # noqa: E402
from collection_estimation.hurdle.grid import (  # noqa: E402
    GRID_SCORE_COLUMNS, combination_name, grid_scores, scored_numbers)
from collection_estimation.hurdle.prices import projected_levels  # noqa: E402
from collection_estimation.hurdle.topups import TOPUP_COLUMNS  # noqa: E402
from tests.hurdle.test_evaluate import make_weekly, make_weeks  # noqa: E402
from tests.hurdle.test_forecast import WEEK_ZERO  # noqa: E402

SC_COMPANIES = ["TR02", "TR04"]
SC_ORIGINS = (18, 22)
SC_H = 5
SC_CUSTOMERS = 3        # per company


def sc_settings(horizons=SC_H):
    return SimpleNamespace(receivables=SimpleNamespace(horizons=horizons))


def sc_topups(seed, origins=SC_ORIGINS, companies=SC_COMPANIES, horizons=SC_H):
    rng = np.random.default_rng(seed)
    rows = []
    for origin in origins:
        for company in companies:
            for h in range(1, horizons + 1):
                new, refunds, unid = (rng.uniform(100, 400), -rng.uniform(10, 80),
                                      rng.uniform(20, 120))
                rows.append({"company_code": company, "origin_week": origin, "h": h,
                             "target_week": origin + h, "new_customer_cash": new,
                             "refunds": refunds, "unidentified_cash": unid,
                             "topup": new + refunds + unid, "fallback": False})
    return pd.DataFrame(rows, columns=TOPUP_COLUMNS)


def sc_predictions(n1=3, n2=3, n3=2, seed=11, origins=SC_ORIGINS, index_offset=900):
    """A hand-made GridPredictions; every candidate distinct, p in (0, 1), m above 0."""
    rng = np.random.default_rng(seed)
    rows = []
    for origin in origins:
        for number, company in enumerate(SC_COMPANIES):
            for c in range(SC_CUSTOMERS):
                for h in range(1, SC_H + 1):
                    customer = f"C{number}{c}"
                    rows.append({"key": f"{company}|{customer}", "company_code": company,
                                 "customer_id": customer, "origin_week": origin, "h": h,
                                 "target_week": origin + h, "label_status": "labelled",
                                 "paid": 1.0, "amount_deflated": 500.0, "base_rate": 0.5,
                                 "training_mean": 400.0})
    frame = pd.DataFrame(rows, columns=ROW_COLUMNS)
    # A shuffled index that is not a range, as the training table's.
    frame.index = pd.Index(index_offset + rng.permutation(len(frame)))
    n = len(frame)
    p = pd.DataFrame({i: rng.uniform(0.05, 0.95, n) for i in range(n1)}, index=frame.index)
    m = pd.DataFrame({j: rng.uniform(200, 2000, n) for j in range(n2)}, index=frame.index)
    topups = [sc_topups(seed + 100 + k, origins=origins) for k in range(n3)]
    return GridPredictions(rows=frame, p=p, m=m, topups=topups)


def sc_candidates(n1=3, n2=3, n3=2):
    return GridCandidates(stage1=[f"<s1 {i}>" for i in range(n1)],
                          stage2=[f"<s2 {j}>" for j in range(n2)],
                          topups=[f"<t {k}>" for k in range(n3)])


def sc_gate(number_column, eligible):
    return pd.DataFrame({number_column: list(range(len(eligible))), "eligible": list(eligible)})


def sc_gates(eligible1=(True, True, True), eligible2=(True, True, True)):
    return GridGates(stage1=sc_gate("stage1", eligible1), stage2=sc_gate("stage2", eligible2))


def sc_actual(origins=SC_ORIGINS, missing=(), seed=3, horizons=SC_H, zero_company=None):
    """`actual_totals` on seeded weekly cash; `zero_company` gets no cash at all (rows of 0)."""
    rng = np.random.default_rng(seed)
    lines = [(f"{company}|K", company, "K", week, rng.uniform(500, 6000), -rng.uniform(0, 100))
             for company in SC_COMPANIES for week in range(0, 41)]
    lines = [line for line in lines if line[1] != zero_company]
    return actual_totals(make_weekly(lines), make_weeks(40, missing=missing), f_levels(),
                         list(SC_COMPANIES), list(origins), horizons)


def sc_run(predictions=None, candidates=None, gates=None, actual=None, settings=None,
           companies=SC_COMPANIES, week_zero=WEEK_ZERO):
    predictions = predictions if predictions is not None else sc_predictions()
    return grid_scores(predictions, candidates or sc_candidates(), gates or sc_gates(),
                       actual if actual is not None else sc_actual(), settings or sc_settings(),
                       companies=list(companies), week_zero=week_zero)


def sc_by_hand(predictions, i, j, k, actual, horizons=SC_H, week_zero=WEEK_ZERO):
    """One combination's scores computed from scratch: no forecast_totals, no
    evaluation_scores. Returns (wape, total_wape, bias, company_wape)."""
    rows = predictions.rows
    expected = predictions.p[i] * predictions.m[j]
    topups = predictions.topups[k]
    forecast_rows = []
    for origin in sorted(set(rows["origin_week"])):
        level = projected_levels(week_zero, origin, [origin + h for h in range(1, horizons + 1)])
        for h in range(1, horizons + 1):
            consolidated = 0.0
            for company in SC_COMPANIES:
                mine = (rows["origin_week"] == origin) & (rows["h"] == h) & (
                    rows["company_code"] == company)
                top = topups[(topups["origin_week"] == origin) & (topups["h"] == h)
                             & (topups["company_code"] == company)].iloc[0]
                real = (expected[mine].sum() + top["new_customer_cash"]
                        + top["unidentified_cash"] + top["refunds"])
                value = real * level[origin + h]
                consolidated += value
                forecast_rows.append((company, origin, h, value))
            forecast_rows.append((CONSOLIDATED, origin, h, consolidated))
    made = pd.DataFrame(forecast_rows, columns=["company_code", "origin_week", "h", "forecast"])
    both = made.merge(actual[["company_code", "origin_week", "h", "actual"]],
                      on=["company_code", "origin_week", "h"])

    def wape(frame):
        return float((frame["forecast"] - frame["actual"]).abs().sum() / frame["actual"].sum())

    everything = both[both["company_code"] == CONSOLIDATED]
    per_origin = everything.groupby("origin_week").agg(
        weeks=("h", "size"), forecast=("forecast", "sum"), actual=("actual", "sum"))
    complete = per_origin[per_origin["weeks"] == horizons]
    total_wape = wape(complete) if not complete.empty else float("nan")
    bias = float(everything["forecast"].sum() / everything["actual"].sum() - 1)
    company_wape = float(np.mean([wape(both[both["company_code"] == c]) for c in SC_COMPANIES]))
    return wape(everything), total_wape, bias, company_wape


def sc_line(table, i, j, k):
    found = table[(table["stage1"] == i) & (table["stage2"] == j) & (table["topups"] == k)]
    assert len(found) == 1
    return found.iloc[0]


@pytest.fixture(scope="module")
def sc_result():
    predictions = sc_predictions()
    actual = sc_actual()
    return predictions, actual, sc_run(predictions=predictions, actual=actual)


# --- the constants and the small helpers --------------------------------------------------

def test_grid_score_columns_are_the_documented_list():
    assert GRID_SCORE_COLUMNS == ["stage1", "stage2", "topups", "current", "wape", "total_wape",
                                  "bias", "company_wape"]


def test_combination_name_exact_format():
    assert combination_name(0, 0, 0) == "stage1 0, stage2 0, top-ups 0"
    assert combination_name(3, 12, 1) == "stage1 3, stage2 12, top-ups 1"


def test_combination_names_of_different_combinations_differ():
    names = {combination_name(i, j, k) for i, j, k in itertools.product(range(12), repeat=3)}
    assert len(names) == 12 ** 3


def test_scored_numbers_are_the_eligible_ones_plus_0_sorted():
    gates = pd.DataFrame({"stage1": [0, 1, 2, 3, 4], "eligible": [False, True, False, True, True]})
    assert scored_numbers(gates, "stage1") == [0, 1, 3, 4]


def test_scored_numbers_an_eligible_current_is_listed_once():
    gates = pd.DataFrame({"stage2": [0, 1, 2], "eligible": [True, False, True]})
    assert scored_numbers(gates, "stage2") == [0, 2]


def test_scored_numbers_nothing_eligible_gives_only_the_current():
    gates = pd.DataFrame({"stage1": [0, 1, 2], "eligible": [False, False, False]})
    assert scored_numbers(gates, "stage1") == [0]


@pytest.mark.parametrize("flag", [True, False])
def test_scored_numbers_a_lone_current_is_scored_eligible_or_not(flag):
    assert scored_numbers(pd.DataFrame({"stage1": [0], "eligible": [flag]}), "stage1") == [0]


def test_scored_numbers_are_sorted_whatever_the_order_of_the_gate_table():
    gates = pd.DataFrame({"stage1": [3, 0, 4, 1, 2],
                          "eligible": [True, False, True, True, False]})
    assert scored_numbers(gates, "stage1") == [0, 1, 3, 4]


def test_scored_numbers_find_the_current_by_number_not_position():
    # Candidate 0 is the last row, ineligible; the first row (candidate 2) is not current.
    gates = pd.DataFrame({"stage1": [2, 1, 0], "eligible": [False, True, False]})
    assert scored_numbers(gates, "stage1") == [0, 1]


def test_scored_numbers_are_plain_ints():
    gates = pd.DataFrame({"stage1": np.array([0, 1, 2], dtype=np.int64),
                          "eligible": [False, True, True]})
    found = scored_numbers(gates, "stage1")
    assert found == [0, 1, 2] and all(type(n) is int for n in found)


def test_scored_numbers_read_the_named_column_only():
    gates = pd.DataFrame({"stage1": [5, 6, 7], "stage2": [0, 1, 2],
                          "eligible": [False, True, False]})
    assert scored_numbers(gates, "stage2") == [0, 1]
    assert scored_numbers(gates, "stage1") == [6]


def test_scored_numbers_do_not_change_the_gate_table():
    gates = pd.DataFrame({"stage1": [2, 0, 1], "eligible": [True, False, True]})
    before = gates.copy()
    scored_numbers(gates, "stage1")
    pd.testing.assert_frame_equal(gates, before)


def test_scored_numbers_on_grid_gates_output(gate_result):
    # The U16.4 worked example: eligible stage 1 {1, 2} and stage 2 {1, 2}; 0 fails both.
    _, _, gates = gate_result
    assert scored_numbers(gates.stage1, "stage1") == [0, 1, 2]
    assert scored_numbers(gates.stage2, "stage2") == [0, 1, 2]


# --- grid_scores: the table ---------------------------------------------------------------

def test_grid_scores_columns_and_a_fresh_range_index(sc_result):
    _, _, table = sc_result
    assert list(table.columns) == GRID_SCORE_COLUMNS
    assert table.index.equals(pd.RangeIndex(len(table)))


def test_grid_scores_one_row_per_combination_of_every_candidate_when_all_eligible(sc_result):
    _, _, table = sc_result
    assert len(table) == 3 * 3 * 2
    found = set(zip(table["stage1"], table["stage2"], table["topups"]))
    assert found == set(itertools.product(range(3), range(3), range(2)))


def test_grid_scores_only_the_current_combination_is_current(sc_result):
    _, _, table = sc_result
    current = table[table["current"]]
    assert len(current) == 1
    assert current[["stage1", "stage2", "topups"]].iloc[0].tolist() == [0, 0, 0]
    assert table["current"].dtype == bool


def test_grid_scores_number_columns_are_integers(sc_result):
    _, _, table = sc_result
    for column in ("stage1", "stage2", "topups"):
        assert pd.api.types.is_integer_dtype(table[column]), column


def test_grid_scores_sorted_by_wape_ascending(sc_result):
    _, _, table = sc_result
    assert table["wape"].is_monotonic_increasing
    # The premise: the example's combinations really do score differently.
    assert table["wape"].nunique() == len(table)


def test_grid_scores_scores_are_finite_numbers(sc_result):
    _, _, table = sc_result
    for column in ("wape", "total_wape", "bias", "company_wape"):
        assert np.isfinite(table[column].astype(float)).all(), column
    assert (table["wape"] >= 0).all() and (table["company_wape"] >= 0).all()


def test_grid_scores_every_combination_equals_the_computation_by_hand(sc_result):
    predictions, actual, table = sc_result
    for i, j, k in itertools.product(range(3), range(3), range(2)):
        line = sc_line(table, i, j, k)
        wape, total_wape, bias, company_wape = sc_by_hand(predictions, i, j, k, actual)
        assert line["wape"] == pytest.approx(wape, rel=1e-12), (i, j, k)
        assert line["total_wape"] == pytest.approx(total_wape, rel=1e-12), (i, j, k)
        assert line["bias"] == pytest.approx(bias, rel=1e-9, abs=1e-12), (i, j, k)
        assert line["company_wape"] == pytest.approx(company_wape, rel=1e-12), (i, j, k)


def alone_scores(predictions, i, j, k, actual, method="alone"):
    """One combination through forecast_totals and evaluation_scores on its own."""
    rows = predictions.rows
    keys = ["company_code", "origin_week", "h", "target_week"]
    customers = rows[keys].assign(expected=predictions.p[i] * predictions.m[j])
    topups = predictions.topups[k]
    parts = []
    for origin in sorted(set(rows["origin_week"])):
        projected = projected_levels(WEEK_ZERO, origin, [origin + h for h in range(1, SC_H + 1)])
        totals = forecast_totals(customers[rows["origin_week"] == origin],
                                 topups[topups["origin_week"] == origin], projected,
                                 list(SC_COMPANIES), SC_H)
        parts.append(totals[keys + ["total"]].rename(columns={"total": "forecast"}))
    stacked = pd.concat(parts, ignore_index=True).assign(method=method)
    return evaluation_scores(stacked[STACK_COLUMNS], actual, SC_H)


def pooled(scores, company, column="wape", horizon=POOLED):
    return scores.loc[(scores["company_code"] == company) & (scores["horizon"] == horizon),
                      column].iloc[0]


def test_grid_scores_each_combination_equals_scoring_it_alone_exactly(sc_result):
    # Being scored beside the others changes nothing: the same numbers, to the last bit.
    predictions, actual, table = sc_result
    for i, j, k in [(0, 0, 0), (2, 1, 1), (1, 2, 0)]:
        scores = alone_scores(predictions, i, j, k, actual)
        line = sc_line(table, i, j, k)
        assert line["wape"] == pooled(scores, CONSOLIDATED)
        assert line["total_wape"] == pooled(scores, CONSOLIDATED, horizon=HORIZON_TOTAL)
        assert line["bias"] == pooled(scores, CONSOLIDATED, "bias")
        companies = [pooled(scores, c) for c in SC_COMPANIES]
        assert line["company_wape"] == pytest.approx(np.mean(companies), rel=1e-15)


def test_grid_scores_the_stage_candidates_and_topups_each_move_the_score(sc_result):
    _, _, table = sc_result
    base = sc_line(table, 0, 0, 0)["wape"]
    assert sc_line(table, 1, 0, 0)["wape"] != base
    assert sc_line(table, 0, 1, 0)["wape"] != base
    assert sc_line(table, 0, 0, 1)["wape"] != base


def test_grid_scores_company_wape_is_the_plain_mean_of_the_companies_pooled_wape(sc_result):
    predictions, actual, table = sc_result
    scores = alone_scores(predictions, 1, 1, 1, actual)
    tr02, tr04 = pooled(scores, "TR02"), pooled(scores, "TR04")
    line = sc_line(table, 1, 1, 1)
    assert line["company_wape"] == pytest.approx((tr02 + tr04) / 2, rel=1e-15)
    # Not the consolidated WAPE: errors of the companies do not cancel in it.
    assert line["company_wape"] != line["wape"]


def test_grid_scores_are_deterministic_and_change_no_input():
    predictions, actual, gates = sc_predictions(), sc_actual(), sc_gates((True, False, True))
    before = (predictions.rows.copy(), predictions.p.copy(), predictions.m.copy(),
              [t.copy() for t in predictions.topups], actual.copy(), gates.stage1.copy(),
              gates.stage2.copy())
    first = sc_run(predictions=predictions, actual=actual, gates=gates)
    second = sc_run(predictions=predictions, actual=actual, gates=gates)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(predictions.rows, before[0])
    pd.testing.assert_frame_equal(predictions.p, before[1])
    pd.testing.assert_frame_equal(predictions.m, before[2])
    for found, kept in zip(predictions.topups, before[3]):
        pd.testing.assert_frame_equal(found, kept)
    pd.testing.assert_frame_equal(actual, before[4])
    pd.testing.assert_frame_equal(gates.stage1, before[5])
    pd.testing.assert_frame_equal(gates.stage2, before[6])


def test_grid_scores_the_row_order_of_the_predictions_does_not_matter(sc_result):
    predictions, actual, table = sc_result
    order = np.random.default_rng(1).permutation(len(predictions.rows))
    shuffled = remade(predictions, rows=predictions.rows.iloc[order],
                      p=predictions.p.iloc[order[::-1]], m=predictions.m.iloc[order])
    found = sc_run(predictions=shuffled, actual=actual)
    pd.testing.assert_frame_equal(found[GRID_SCORE_COLUMNS[:4]], table[GRID_SCORE_COLUMNS[:4]])
    for column in ("wape", "total_wape", "bias", "company_wape"):
        np.testing.assert_allclose(found[column], table[column], rtol=1e-12, atol=1e-15)


# --- grid_scores: ties keep the candidates' order -------------------------------------------

def identical_candidates(n1=2, n2=3, n3=2):
    predictions = sc_predictions(n1=1, n2=1, n3=1)
    p = pd.concat([predictions.p[0]] * n1, axis=1, keys=range(n1))
    m = pd.concat([predictions.m[0]] * n2, axis=1, keys=range(n2))
    return GridPredictions(rows=predictions.rows, p=p, m=m, topups=predictions.topups * n3)


def test_grid_scores_ties_keep_the_order_of_the_candidates():
    # Every candidate the same: every WAPE ties, so the table is the product order itself.
    table = sc_run(predictions=identical_candidates(), candidates=sc_candidates(2, 3, 2),
                   gates=sc_gates((True, True), (True, True, True)))
    assert table["wape"].nunique() == 1
    assert list(zip(table["stage1"], table["stage2"], table["topups"])) == list(
        itertools.product(range(2), range(3), range(2)))
    assert bool(table.loc[0, "current"]) is True


def test_grid_scores_a_tie_with_the_current_settings_puts_the_current_first():
    # Candidate 1 of stage 2 equals the current one: they tie and (0, 0, 0) stays first.
    predictions = sc_predictions(n1=1, n2=2, n3=1)
    m = predictions.m.copy()
    m[1] = m[0]
    table = sc_run(predictions=remade(predictions, m=m), candidates=sc_candidates(1, 2, 1),
                   gates=sc_gates((True,), (True, True)))
    assert table["wape"].iloc[0] == table["wape"].iloc[1]
    assert table[["stage1", "stage2", "topups"]].values.tolist() == [[0, 0, 0], [0, 1, 0]]


# --- grid_scores: which candidates are scored ----------------------------------------------

def test_grid_scores_ineligible_candidates_other_than_the_current_are_not_scored(sc_result):
    predictions, actual, table = sc_result
    found = sc_run(predictions=predictions, actual=actual,
                   gates=sc_gates((True, False, True), (True, True, False)))
    assert set(found["stage1"]) == {0, 2} and set(found["stage2"]) == {0, 1}
    assert len(found) == 2 * 2 * 2
    # Each kept combination scores as in the full grid.
    for _, line in found.iterrows():
        full = sc_line(table, line["stage1"], line["stage2"], line["topups"])
        assert line["wape"] == full["wape"] and line["bias"] == full["bias"]


def test_grid_scores_an_ineligible_current_is_still_scored_with_the_eligible_others(sc_result):
    predictions, actual, table = sc_result
    found = sc_run(predictions=predictions, actual=actual,
                   gates=sc_gates((False, True, False), (False, False, True)))
    assert set(zip(found["stage1"], found["stage2"], found["topups"])) == set(
        itertools.product([0, 1], [0, 2], [0, 1]))
    assert found["current"].sum() == 1
    current = found[found["current"]].iloc[0]
    assert current["wape"] == sc_line(table, 0, 0, 0)["wape"]


def test_grid_scores_nothing_eligible_scores_the_current_with_every_topup(sc_result):
    predictions, actual, _ = sc_result
    found = sc_run(predictions=predictions, actual=actual,
                   gates=sc_gates((False,) * 3, (False,) * 3))
    assert sorted(zip(found["stage1"], found["stage2"], found["topups"])) == [(0, 0, 0),
                                                                                (0, 0, 1)]


@pytest.mark.parametrize("n3", [1, 2, 4])
def test_grid_scores_every_topup_candidate_is_scored(n3):
    predictions = sc_predictions(n1=2, n2=1, n3=n3)
    table = sc_run(predictions=predictions, candidates=sc_candidates(2, 1, n3),
                   gates=sc_gates((False, False), (False,)))
    # Top-ups have no gate: all of them, beside the current stage settings only.
    assert sorted(table["topups"]) == list(range(n3))
    assert set(table["stage1"]) == {0} and set(table["stage2"]) == {0}


def test_grid_scores_combinations_are_the_product_of_the_scored_numbers(sc_result):
    predictions, actual, _ = sc_result
    gates = sc_gates((True, False, True), (False, True, True))
    found = sc_run(predictions=predictions, actual=actual, gates=gates)
    expected = set(itertools.product(scored_numbers(gates.stage1, "stage1"),
                                     scored_numbers(gates.stage2, "stage2"), range(2)))
    assert set(zip(found["stage1"], found["stage2"], found["topups"])) == expected


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_grid_scores_a_broken_ineligible_p_is_never_touched(sc_result, bad):
    predictions, actual, _ = sc_result
    p = predictions.p.copy()
    p[1] = bad
    found = sc_run(predictions=remade(predictions, p=p), actual=actual,
                   gates=sc_gates((True, False, True)))
    assert 1 not in set(found["stage1"])
    assert np.isfinite(found["wape"]).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_grid_scores_a_broken_ineligible_m_is_never_touched(sc_result, bad):
    predictions, actual, _ = sc_result
    m = predictions.m.copy()
    m[2] = bad
    found = sc_run(predictions=remade(predictions, m=m), actual=actual,
                   gates=sc_gates(eligible2=(True, True, False)))
    assert 2 not in set(found["stage2"])
    assert np.isfinite(found["wape"]).all()


# --- grid_scores: the inputs it reads -------------------------------------------------------

def test_grid_scores_project_the_levels_once_per_origin(sc_result, monkeypatch):
    predictions, actual, _ = sc_result
    calls = []

    def spy(week_zero, origin, targets, *args, **kwargs):
        calls.append((week_zero, origin, list(targets), args, kwargs))
        return projected_levels(week_zero, origin, targets, *args, **kwargs)

    monkeypatch.setattr(grid_module, "projected_levels", spy)
    sc_run(predictions=predictions, actual=actual)
    assert calls == [(WEEK_ZERO, 18, [19, 20, 21, 22, 23], (), {}),
                     (WEEK_ZERO, 22, [23, 24, 25, 26, 27], (), {})]


def test_grid_scores_another_week_zero_changes_the_nominal_scores(sc_result):
    predictions, actual, table = sc_result
    later = sc_run(predictions=predictions, actual=actual, week_zero=date(2025, 1, 6))
    merged = table.merge(later, on=["stage1", "stage2", "topups"], suffixes=("", "_later"))
    assert (merged["wape"] != merged["wape_later"]).all()


def test_grid_scores_a_week_zero_not_a_monday_raises_as_projected_levels_does(sc_result):
    predictions, actual, _ = sc_result
    with pytest.raises(ValueError, match="not a Monday"):
        sc_run(predictions=predictions, actual=actual, week_zero=date(2025, 6, 3))


def test_grid_scores_each_topup_candidate_reads_its_own_table_at_its_own_origin(sc_result):
    # Change top-up candidate 1 at origin 22 only: only k = 1 rows move.
    predictions, actual, table = sc_result
    changed = predictions.topups[1].copy()
    changed.loc[changed["origin_week"] == 22, "new_customer_cash"] += 5000.0
    found = sc_run(predictions=remade(predictions, topups=[predictions.topups[0], changed]),
                   actual=actual)
    merged = table.merge(found, on=["stage1", "stage2", "topups"], suffixes=("", "_new"))
    k0, k1 = merged[merged["topups"] == 0], merged[merged["topups"] == 1]
    assert (k0["wape"] == k0["wape_new"]).all()
    assert (k1["wape"] != k1["wape_new"]).all()


def test_grid_scores_the_topup_column_topup_is_not_what_is_added(sc_result):
    # forecast_totals adds new_customer_cash, unidentified_cash and refunds; the `topup`
    # column is not read.
    predictions, actual, table = sc_result
    topups = [t.assign(topup=t["topup"] * 99.0) for t in predictions.topups]
    found = sc_run(predictions=remade(predictions, topups=topups), actual=actual)
    pd.testing.assert_frame_equal(found, table)


def test_grid_scores_the_label_columns_of_the_rows_are_not_read(sc_result):
    predictions, actual, table = sc_result
    rows = predictions.rows.assign(paid=np.nan, amount_deflated=np.nan, label_status="future",
                                   base_rate=np.nan, training_mean=np.nan)
    found = sc_run(predictions=remade(predictions, rows=rows), actual=actual)
    pd.testing.assert_frame_equal(found, table)


def test_grid_scores_the_origins_come_from_the_rows():
    # A grid at origin 18 alone, scored on the actual cash at 18.
    predictions = sc_predictions(origins=(18,))
    actual = sc_actual(origins=(18,))
    table = sc_run(predictions=predictions, actual=actual)
    wape, _, _, _ = sc_by_hand(predictions, 0, 0, 0, actual)
    assert sc_line(table, 0, 0, 0)["wape"] == pytest.approx(wape, rel=1e-12)


def test_grid_scores_total_wape_is_nan_when_no_origin_has_all_its_weeks():
    # Week 23 has no data: origin 18 lacks h 5, origin 22 lacks h 1; no 5-week total.
    predictions = sc_predictions()
    actual = sc_actual(missing=(23,))
    assert not (actual["target_week"] == 23).any()
    table = sc_run(predictions=predictions, actual=actual)
    assert table["total_wape"].isna().all()
    assert np.isfinite(table["wape"]).all() and np.isfinite(table["bias"]).all()
    wape, total_wape, _, _ = sc_by_hand(predictions, 0, 0, 0, actual)
    assert np.isnan(total_wape)
    assert sc_line(table, 0, 0, 0)["wape"] == pytest.approx(wape, rel=1e-12)


def test_grid_scores_total_wape_uses_only_the_complete_origins():
    # Week 27 missing: origin 22 lacks h 5, origin 18 is whole; the total is origin 18's alone.
    predictions = sc_predictions()
    actual = sc_actual(missing=(27,))
    table = sc_run(predictions=predictions, actual=actual)
    _, total_wape, _, _ = sc_by_hand(predictions, 0, 0, 0, actual)
    assert np.isfinite(total_wape)
    assert sc_line(table, 0, 0, 0)["total_wape"] == pytest.approx(total_wape, rel=1e-12)


# --- grid_scores: refused -------------------------------------------------------------------

@pytest.mark.parametrize("stage, eligible1, eligible2, found", [
    ("stage-1", (True, True), (True, True, True), 2),
    ("stage-1", (True,) * 4, (True, True, True), 4),
    ("stage-2", (True, True, True), (True,), 1),
    ("stage-2", (True, True, True), (True,) * 5, 5),
])
def test_grid_scores_a_gate_table_of_the_wrong_length_raises(sc_result, stage, eligible1,
                                                             eligible2, found):
    predictions, actual, _ = sc_result
    with pytest.raises(ValueError) as error:
        sc_run(predictions=predictions, actual=actual, gates=sc_gates(eligible1, eligible2))
    assert str(error.value) == f"there are {found} {stage} gates for 3 candidates"


@pytest.mark.parametrize("n3", [1, 3])
def test_grid_scores_a_topup_list_of_the_wrong_length_raises(sc_result, n3):
    # The candidates list n3 top-ups, the predictions hold 2 top-up forecasts.
    predictions, actual, _ = sc_result
    with pytest.raises(ValueError) as error:
        sc_run(predictions=predictions, actual=actual, candidates=sc_candidates(3, 3, n3))
    assert str(error.value) == f"there are 2 top-up forecasts for {n3} candidates"


def test_grid_scores_the_stage_1_count_is_checked_first(sc_result):
    predictions, actual, _ = sc_result
    with pytest.raises(ValueError) as error:
        sc_run(predictions=predictions, actual=actual, candidates=sc_candidates(2, 2, 5))
    assert str(error.value) == "there are 3 stage-1 gates for 2 candidates"


def test_grid_scores_count_checks_in_their_order(sc_result):
    # stage-1 gates, stage-1 p columns, stage-2 gates, stage-2 m columns, top-up forecasts:
    # with every later one also wrong, each check in turn is the one that speaks.
    predictions, actual, _ = sc_result
    p2 = predictions.p.drop(columns=[2])
    m2 = predictions.m.drop(columns=[2])
    bad_all = remade(predictions, p=p2, m=m2, topups=predictions.topups[:1])
    cases = [
        (sc_gates((True,) * 2, (True,) * 2), "there are 2 stage-1 gates for 3 candidates"),
        (sc_gates((True,) * 3, (True,) * 2), "there are 2 stage-1 p columns for 3 candidates"),
    ]
    for gates, message in cases:
        with pytest.raises(ValueError) as error:
            sc_run(predictions=bad_all, actual=actual, gates=gates)
        assert str(error.value) == message
    later = [
        (remade(predictions, m=m2, topups=predictions.topups[:1]), sc_gates(eligible2=(True,) * 2),
         "there are 2 stage-2 gates for 3 candidates"),
        (remade(predictions, m=m2, topups=predictions.topups[:1]), sc_gates(),
         "there are 2 stage-2 m columns for 3 candidates"),
        (remade(predictions, topups=predictions.topups[:1]), sc_gates(),
         "there are 1 top-up forecasts for 2 candidates"),
    ]
    for found, gates, message in later:
        with pytest.raises(ValueError) as error:
            sc_run(predictions=found, actual=actual, gates=gates)
        assert str(error.value) == message


def test_grid_scores_the_counts_are_checked_before_anything_is_forecast(sc_result, monkeypatch):
    predictions, actual, _ = sc_result

    def never(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(grid_module, "forecast_totals", never)
    monkeypatch.setattr(grid_module, "projected_levels", never)
    with pytest.raises(ValueError, match="top-up forecasts"):
        sc_run(predictions=predictions, actual=actual, candidates=sc_candidates(3, 3, 1))
    with pytest.raises(ValueError, match="stage-1 p columns"):
        sc_run(predictions=remade(predictions, p=predictions.p.drop(columns=[1])), actual=actual)


@pytest.mark.parametrize("part", ["p", "m"])
def test_grid_scores_a_nan_p_or_m_of_the_current_settings_raises(sc_result, part):
    predictions, actual, _ = sc_result
    broken = getattr(predictions, part).copy()
    broken.iloc[4, 0] = np.nan
    with pytest.raises(ValueError, match="the customers have no expected in 1 rows"):
        sc_run(predictions=remade(predictions, **{part: broken}), actual=actual,
               gates=sc_gates((False,) * 3, (False,) * 3))


def test_grid_scores_an_infinite_m_of_the_current_settings_raises(sc_result):
    predictions, actual, _ = sc_result
    m = predictions.m.copy()
    m.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        sc_run(predictions=remade(predictions, m=m), actual=actual)


def test_grid_scores_a_nan_p_of_an_eligible_candidate_raises(sc_result):
    # Eligible means scored: a broken eligible candidate is not skipped silently.
    predictions, actual, _ = sc_result
    p = predictions.p.copy()
    p.iloc[7, 2] = np.nan
    with pytest.raises(ValueError, match="no expected"):
        sc_run(predictions=remade(predictions, p=p), actual=actual)


def gated_scores(predictions):
    """grid_gates then grid_scores on a U16.4 example (one company TR02, zero top-ups)."""
    gates = grid_gates(predictions, gate_candidates(), gate_settings())
    topups = pd.DataFrame([{"company_code": "TR02", "origin_week": origin, "h": h,
                            "target_week": origin + h, "new_customer_cash": 0.0,
                            "refunds": 0.0, "unidentified_cash": 0.0, "topup": 0.0,
                            "fallback": False}
                           for origin in GATE_ORIGINS for h in range(1, 6)],
                          columns=TOPUP_COLUMNS)
    candidates = GridCandidates(stage1=gate_candidates().stage1,
                                stage2=gate_candidates().stage2, topups=["<t>"])
    lines = [("TR02|K", "TR02", "K", week, 1e7, 0.0) for week in range(0, 50)]
    actual = actual_totals(make_weekly(lines), make_weeks(49), f_levels(), ["TR02"],
                           list(GATE_ORIGINS), 5)
    return gates, grid_scores(remade(predictions, topups=[topups]), candidates, gates, actual,
                              sc_settings(), companies=["TR02"], week_zero=WEEK_ZERO)


def test_grid_scores_a_candidate_broken_only_on_unlabelled_rows_is_refused_not_fatal():
    # F2 of U16.5: candidate 1's p is NaN on the unlabelled rows only. grid_gates now refuses
    # it ("bad elsewhere"), so grid_scores never forecasts with it and scores the others.
    predictions = gate_example(poison_unlabelled=True)
    gates, table = gated_scores(predictions)
    assert gates.stage1.loc[1, "bad elsewhere"] > 0
    assert 1 not in scored_numbers(gates.stage1, "stage1")
    assert set(table["stage1"]) == {0, 2}
    assert np.isfinite(table["wape"]).all()


def test_grid_scores_the_current_broken_on_unlabelled_rows_fails_its_gates_and_still_raises():
    # Candidate 0 is scored even when it fails its gates, so a NaN p of the current settings
    # on an unlabelled row reaches forecast_totals, which refuses it.
    predictions = gate_example()
    target = predictions.rows.index[predictions.rows["label_status"] == FUTURE][0]
    predictions.p.loc[target, 0] = np.nan
    with pytest.raises(ValueError, match="the customers have no expected in 1 rows"):
        gated_scores(predictions)


def test_grid_scores_on_the_clean_gate_example_scores_the_eligible_and_the_current():
    _, table = gated_scores(gate_example())
    assert set(zip(table["stage1"], table["stage2"])) == set(
        itertools.product([0, 1, 2], [0, 1, 2]))
    assert table["current"].sum() == 1


@pytest.mark.parametrize("origins", [(18, 22, 26), (18,), (22, 26), (17, 18, 22)])
def test_grid_scores_actual_for_other_origins_than_the_predictions_raises(sc_result, origins):
    # F4 of U16.5: fewer or more origins are refused, with the two lists in the message.
    predictions, _, _ = sc_result
    with pytest.raises(ValueError) as error:
        sc_run(predictions=predictions, actual=sc_actual(origins=origins))
    assert str(error.value) == (f"the actual cash is for origins {sorted(origins)}, the "
                                f"predictions for [18, 22]")


def test_grid_scores_the_origin_check_comes_before_any_forecast(sc_result, monkeypatch):
    predictions, _, _ = sc_result

    def never(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(grid_module, "forecast_totals", never)
    monkeypatch.setattr(grid_module, "projected_levels", never)
    with pytest.raises(ValueError, match="the actual cash is for origins"):
        sc_run(predictions=predictions, actual=sc_actual(origins=(18,)))


def test_grid_scores_the_counts_are_checked_before_the_origins(sc_result):
    predictions, _, _ = sc_result
    with pytest.raises(ValueError, match="top-up forecasts"):
        sc_run(predictions=predictions, actual=sc_actual(origins=(18,)),
               candidates=sc_candidates(3, 3, 1))


def test_grid_scores_actual_origins_compare_as_numbers_not_types(sc_result):
    # numpy integer origins in `actual` are the same origins.
    predictions, actual, table = sc_result
    found = sc_run(predictions=predictions,
                   actual=actual.assign(origin_week=actual["origin_week"].astype("int32")))
    pd.testing.assert_frame_equal(found, table)


def test_grid_scores_company_wape_leaves_out_a_company_with_no_cash():
    # F3 of U16.5, documented as intended: TR04 has no cash at the grid's origins (its WAPE is
    # a share of nothing, NaN); company_wape is TR02's alone, not NaN and not halved.
    predictions = sc_predictions()
    actual = sc_actual(zero_company="TR04")
    tr04 = actual[actual["company_code"] == "TR04"]
    assert len(tr04) > 0 and (tr04["actual"] == 0).all()
    table = sc_run(predictions=predictions, actual=actual)
    scores = alone_scores(predictions, 0, 0, 0, actual)
    assert np.isnan(pooled(scores, "TR04"))
    current = table[table["current"]].iloc[0]
    assert current["company_wape"] == pooled(scores, "TR02")
    assert np.isfinite(table["company_wape"]).all()


def test_grid_scores_empty_actual_raises_as_actual_for_no_origin():
    # Since F4, an empty actual is refused by the origin check first.
    with pytest.raises(ValueError) as error:
        sc_run(actual=sc_actual().iloc[0:0])
    assert str(error.value) == ("the actual cash is for origins [], the predictions for "
                                "[18, 22]")


def test_grid_scores_actual_without_consolidated_rows_raises():
    # Company rows only, at the right origins: past the origin check, nothing to score on.
    actual = sc_actual()
    actual = actual[actual["company_code"] != CONSOLIDATED]
    assert sorted(set(actual["origin_week"])) == [18, 22]
    with pytest.raises(ValueError) as error:
        sc_run(actual=actual)
    assert str(error.value) == "there is no consolidated cash at the grid's origins to score on"


def test_grid_scores_a_company_missing_from_the_rows_raises_as_forecast_totals_does():
    # The company list names TR05, which has no customers nor top-ups: never a silent zero.
    with pytest.raises(ValueError, match="have no rows for"):
        sc_run(companies=["TR02", "TR04", "TR05"])


def test_grid_scores_a_company_list_without_a_company_of_the_rows_raises():
    with pytest.raises(ValueError, match="not in the company list"):
        sc_run(companies=["TR02"])


def test_grid_scores_horizons_other_than_the_rows_raise():
    with pytest.raises(ValueError):
        sc_run(settings=sc_settings(horizons=4))


def test_grid_scores_a_topup_table_missing_an_origin_raises(sc_result):
    predictions, actual, _ = sc_result
    short = predictions.topups[1][predictions.topups[1]["origin_week"] == 18]
    with pytest.raises(ValueError):
        sc_run(predictions=remade(predictions, topups=[predictions.topups[0], short]),
               actual=actual)


# --- grid_scores: the docstring's promise on the sizes of `predictions` --------------------

P_COLUMNS_MESSAGE = "there are 2 stage-1 p columns for 3 candidates"


def test_grid_scores_p_lacking_an_eligible_candidates_column_raises_value_error(sc_result):
    # The docstring: "Raises ValueError if `gates` or `predictions` do not have one entry per
    # candidate". The gates are the right size; p lacks the column of candidate 2 (F1: this
    # was a raw KeyError before).
    predictions, actual, _ = sc_result
    with pytest.raises(ValueError) as error:
        sc_run(predictions=remade(predictions, p=predictions.p.drop(columns=[2])),
               actual=actual)
    assert str(error.value) == P_COLUMNS_MESSAGE


def test_grid_scores_p_lacking_an_ineligible_candidates_column_raises_value_error(sc_result):
    # F1: was silent before (the column of an ineligible candidate is never read).
    predictions, actual, _ = sc_result
    with pytest.raises(ValueError) as error:
        sc_run(predictions=remade(predictions, p=predictions.p.drop(columns=[2])),
               actual=actual, gates=sc_gates((True, True, False)))
    assert str(error.value) == P_COLUMNS_MESSAGE


def test_grid_scores_m_with_more_columns_than_candidates_raises_value_error(sc_result):
    predictions, actual, _ = sc_result
    m = predictions.m.copy()
    m[3] = m[0]
    with pytest.raises(ValueError) as error:
        sc_run(predictions=remade(predictions, m=m), actual=actual)
    assert str(error.value) == "there are 4 stage-2 m columns for 3 candidates"


# --- grid_scores on the U16.3 grid itself: the current combination is step 17's model --------

def g_actual(origins=G_ORIGINS):
    rng = np.random.default_rng(8)
    lines = [(f"{company}|K", company, "K", week, rng.uniform(2000, 20000), -rng.uniform(0, 300))
             for company in F_COMPANIES for week in range(0, 41)]
    return actual_totals(make_weekly(lines), make_weeks(40), f_levels(), list(F_COMPANIES),
                         list(origins), 5)


def g_gates(candidates, eligible1=None, eligible2=None):
    n1, n2 = len(candidates.stage1), len(candidates.stage2)
    return GridGates(stage1=sc_gate("stage1", eligible1 or (True,) * n1),
                     stage2=sc_gate("stage2", eligible2 or (True,) * n2))


@pytest.fixture(scope="module")
def g_scores(g_result):
    result, _ = g_result
    settings = g_settings()
    candidates = g_candidates(settings)
    actual = g_actual()
    table = grid_scores(result, candidates, g_gates(candidates), actual, settings,
                        companies=list(F_COMPANIES), week_zero=WEEK_ZERO)
    return result, candidates, settings, actual, table


@pytest.fixture(scope="module")
def g_rolled(g_table):
    return rolling_forecasts(g_table, G_ORIGINS, g_settings(), week_zero=WEEK_ZERO,
                             **g_inputs())


def hurdle_scores(rolled, actual):
    """rolling_forecasts' totals scored as step 17 (and show_grid_scores) score them."""
    hurdle = rolled.totals[["company_code", "origin_week", "h", "target_week", "total"]]
    hurdle = hurdle.rename(columns={"total": "forecast"}).assign(method=HURDLE)
    return evaluation_scores(hurdle[STACK_COLUMNS], actual, 5)


def test_grid_scores_on_the_u16_3_grid_has_every_combination(g_scores):
    _, _, _, _, table = g_scores
    assert set(zip(table["stage1"], table["stage2"], table["topups"])) == set(
        itertools.product(range(2), range(2), range(2)))


def test_grid_scores_current_combination_equals_step_17s_rolling_forecasts_exactly(g_scores,
                                                                                     g_rolled):
    # The check the pipeline prints: (0, 0, 0) scored as step 17 scores rolling_forecasts.
    _, _, _, actual, table = g_scores
    scores = hurdle_scores(g_rolled, actual)
    current = table[table["current"]].iloc[0]
    assert current["wape"] == pooled(scores, CONSOLIDATED)
    assert current["bias"] == pooled(scores, CONSOLIDATED, "bias")
    assert current["total_wape"] == pooled(scores, CONSOLIDATED, horizon=HORIZON_TOTAL)
    assert current["company_wape"] == pytest.approx(
        np.mean([pooled(scores, c) for c in F_COMPANIES]), rel=1e-15)


def test_grid_scores_current_combination_totals_equal_rolling_forecasts_totals(g_scores,
                                                                                g_rolled):
    # Under the scores: the forecast of (0, 0, 0) is rolling_forecasts' total, row by row.
    result, _, _, _, _ = g_scores
    rows = result.rows
    keys = ["company_code", "origin_week", "h", "target_week"]
    customers = rows[keys].assign(expected=result.p[0] * result.m[0])
    parts = []
    for origin in G_ORIGINS:
        projected = projected_levels(WEEK_ZERO, origin, [origin + h for h in range(1, 6)])
        parts.append(forecast_totals(customers[rows["origin_week"] == origin],
                                     result.topups[0][result.topups[0]["origin_week"] == origin],
                                     projected, list(F_COMPANIES), 5))
    mine = pd.concat(parts, ignore_index=True)
    pd.testing.assert_series_equal(mine["total"], g_rolled.totals["total"])


def test_grid_scores_another_combination_is_not_step_17s(g_scores, g_rolled):
    _, _, _, actual, table = g_scores
    step17 = pooled(hurdle_scores(g_rolled, actual), CONSOLIDATED)
    assert (table.loc[~table["current"], "wape"] != step17).all()


def test_grid_scores_on_the_u16_3_grid_ineligible_gates_drop_only_the_others(g_scores):
    result, candidates, settings, actual, table = g_scores
    found = grid_scores(result, candidates, g_gates(candidates, (False, True), (False, True)),
                        actual, settings, companies=list(F_COMPANIES), week_zero=WEEK_ZERO)
    pd.testing.assert_frame_equal(found, table)
    only_current = grid_scores(result, candidates,
                               g_gates(candidates, (False, False), (False, False)), actual,
                               settings, companies=list(F_COMPANIES), week_zero=WEEK_ZERO)
    assert sorted(zip(only_current["stage1"], only_current["stage2"],
                      only_current["topups"])) == [(0, 0, 0), (0, 0, 1)]
    current = only_current[only_current["current"]].iloc[0]
    assert current["wape"] == table[table["current"]].iloc[0]["wape"]


# =============================================================================================
# U16.6: `projected_at`, `combination_forecast`, `choose_combination`, `CURRENT_SETTINGS`,
# `GridChoice` and `grid_choice`.
#
# `projected_at` and `combination_forecast` on the hand-made U16.5 predictions (sc_*), checked
# against `projected_levels` and `forecast_totals` called by hand. `choose_combination` on
# small hand-made score tables, with exactly representable WAPEs (0.75 - 0.25 == 0.5) so the
# "more than min_gain" boundary is exact. `grid_choice` twice: on the sc_* predictions with
# the calibration and the references replaced by recording fakes (what each step is given,
# how the comparison is stacked), and on a fitted grid at three origins with everything real
# (the current combination is step 17's model, the rule uses the chosen top-ups, the
# calibration is `choose_calibration` on the winner's forecast). `grid_choice` calls
# `dataclasses.replace` on the settings, so its settings here are a small frozen dataclass.
# =============================================================================================

from collection_estimation.hurdle.calibrate import (  # noqa: E402
    CalibrationChoice, choose_calibration)
from collection_estimation.hurdle.evaluate import (  # noqa: E402
    REFERENCE_COLUMNS, REFERENCES, RULE, deflated_reference, paired_wape_bootstrap,
    rule_reference, stack_forecasts)
from collection_estimation.hurdle.forecast import Forecast  # noqa: E402
from collection_estimation.hurdle.grid import (  # noqa: E402
    CURRENT_SETTINGS, GridChoice, choose_combination, combination_forecast, grid_choice,
    projected_at)
from tests.hurdle.test_evaluate import rule_rates, running_ledger  # noqa: E402

CH_KEYS = ["company_code", "origin_week", "h", "target_week"]


# --- projected_at ----------------------------------------------------------------------------

def test_projected_at_equals_projected_levels_per_origin():
    found = projected_at([18, 22], 5, WEEK_ZERO)
    assert list(found) == [18, 22]
    for origin in (18, 22):
        expected = projected_levels(WEEK_ZERO, origin, [origin + h for h in range(1, 6)])
        pd.testing.assert_series_equal(found[origin], expected)


def test_projected_at_covers_the_target_weeks_origin_plus_1_to_horizons():
    found = projected_at([10, 30], 3, WEEK_ZERO)
    assert list(found[10].index) == [11, 12, 13]
    assert list(found[30].index) == [31, 32, 33]


def test_projected_at_one_horizon_gives_one_week():
    found = projected_at([20], 1, WEEK_ZERO)
    assert list(found[20].index) == [21]


def test_projected_at_keeps_the_order_given():
    assert list(projected_at([22, 18], 2, WEEK_ZERO)) == [22, 18]


def test_projected_at_no_origins_gives_an_empty_dict():
    assert projected_at([], 5, WEEK_ZERO) == {}


def test_projected_at_a_week_zero_not_a_monday_raises_as_projected_levels_does():
    with pytest.raises(ValueError):
        projected_levels(date(2025, 6, 3), 18, [19])
    with pytest.raises(ValueError):
        projected_at([18], 5, date(2025, 6, 3))


# --- combination_forecast on the hand-made predictions ----------------------------------------

def sc_projected(origins=SC_ORIGINS):
    return projected_at(sorted(origins), SC_H, WEEK_ZERO)


def sc_combination(predictions, i, j, k):
    return combination_forecast(predictions, i, j, k, projected=sc_projected(),
                                companies=list(SC_COMPANIES), horizons=SC_H)


def by_hand_totals(predictions, i, j, k, horizons=SC_H, companies=SC_COMPANIES,
                   week_zero=WEEK_ZERO):
    rows = predictions.rows
    customers = rows[CUSTOMER_COLUMNS[:6]].assign(p=predictions.p[i], m=predictions.m[j])
    customers["expected"] = customers["p"] * customers["m"]
    topups = predictions.topups[k]
    parts = []
    for origin in sorted(set(rows["origin_week"])):
        levels = projected_levels(week_zero, origin, [origin + h for h in range(1, horizons + 1)])
        parts.append(forecast_totals(customers[customers["origin_week"] == origin],
                                     topups[topups["origin_week"] == origin], levels,
                                     list(companies), horizons))
    return pd.concat(parts, ignore_index=True)


def test_combination_forecast_returns_a_forecast():
    found = sc_combination(sc_predictions(), 0, 0, 0)
    assert isinstance(found, Forecast)


def test_combination_forecast_customers_have_the_customer_columns_and_the_rows_index():
    predictions = sc_predictions()
    customers = sc_combination(predictions, 1, 2, 0).customers
    assert list(customers.columns) == CUSTOMER_COLUMNS
    pd.testing.assert_index_equal(customers.index, predictions.rows.index)


def test_combination_forecast_customers_copy_the_six_row_columns():
    predictions = sc_predictions()
    customers = sc_combination(predictions, 1, 2, 0).customers
    pd.testing.assert_frame_equal(customers[CUSTOMER_COLUMNS[:6]],
                                  predictions.rows[CUSTOMER_COLUMNS[:6]])


@pytest.mark.parametrize("i, j", [(0, 0), (1, 2), (2, 1)])
def test_combination_forecast_p_m_and_expected_are_the_chosen_candidates(i, j):
    predictions = sc_predictions()
    customers = sc_combination(predictions, i, j, 1).customers
    np.testing.assert_array_equal(customers["p"].to_numpy(), predictions.p[i].to_numpy())
    np.testing.assert_array_equal(customers["m"].to_numpy(), predictions.m[j].to_numpy())
    np.testing.assert_array_equal(customers["expected"].to_numpy(),
                                  (predictions.p[i] * predictions.m[j]).to_numpy())


def test_combination_forecast_p_and_m_are_joined_by_index_not_position():
    # p and m with their rows in reverse order: the same labels, the same values per label.
    predictions = sc_predictions()
    found = sc_combination(remade(predictions, p=predictions.p.iloc[::-1],
                                  m=predictions.m.iloc[::-1]), 1, 1, 0)
    expected = sc_combination(predictions, 1, 1, 0)
    pd.testing.assert_frame_equal(found.customers, expected.customers)
    pd.testing.assert_frame_equal(found.totals, expected.totals)


@pytest.mark.parametrize("i, j, k", [(0, 0, 0), (2, 1, 1), (1, 2, 0)])
def test_combination_forecast_totals_equal_forecast_totals_by_hand(i, j, k):
    predictions = sc_predictions()
    found = sc_combination(predictions, i, j, k).totals
    pd.testing.assert_frame_equal(found, by_hand_totals(predictions, i, j, k))


def test_combination_forecast_totals_have_a_fresh_range_index_in_origin_order():
    totals = sc_combination(sc_predictions(), 0, 0, 0).totals
    assert list(totals.index) == list(range(len(totals)))
    origins = list(totals["origin_week"])
    assert origins == sorted(origins)
    # One row per company and h, then the consolidated ones, at each origin.
    assert len(totals) == len(SC_ORIGINS) * (len(SC_COMPANIES) + 1) * SC_H


def test_combination_forecast_totals_are_what_grid_scores_scores(sc_result):
    predictions, actual, table = sc_result
    for i, j, k in [(0, 0, 0), (2, 1, 1)]:
        totals = sc_combination(predictions, i, j, k).totals
        stacked = totals[CH_KEYS + ["total"]].rename(columns={"total": "forecast"})
        stacked = stacked.assign(method="x")[STACK_COLUMNS]
        scores = evaluation_scores(stacked, actual, SC_H)
        assert sc_line(table, i, j, k)["wape"] == pooled(scores, CONSOLIDATED)


def test_combination_forecast_each_topup_candidate_reads_its_own_table():
    predictions = sc_predictions()
    first = sc_combination(predictions, 0, 0, 0).totals
    second = sc_combination(predictions, 0, 0, 1).totals
    assert (first["total"] != second["total"]).all()
    pd.testing.assert_series_equal(first["customers"], second["customers"])


def test_combination_forecast_uses_the_projected_levels_given():
    predictions = sc_predictions()
    doubled = {origin: levels * 2 for origin, levels in sc_projected().items()}
    found = combination_forecast(predictions, 0, 0, 0, projected=doubled,
                                 companies=list(SC_COMPANIES), horizons=SC_H).totals
    plain = sc_combination(predictions, 0, 0, 0).totals
    np.testing.assert_allclose(found["total"], 2 * plain["total"], rtol=1e-15)
    pd.testing.assert_series_equal(found["real_total"], plain["real_total"])


def test_combination_forecast_origins_come_from_the_rows_sorted():
    # Rows of origin 22 first: the totals still start with origin 18.
    predictions = sc_predictions()
    rows = predictions.rows.sort_values("origin_week", ascending=False, kind="stable")
    found = sc_combination(remade(predictions, rows=rows), 0, 0, 0)
    assert list(dict.fromkeys(found.totals["origin_week"])) == [18, 22]
    pd.testing.assert_frame_equal(found.totals, sc_combination(predictions, 0, 0, 0).totals)


def test_combination_forecast_does_not_change_the_predictions():
    predictions = sc_predictions()
    before = (predictions.rows.copy(), predictions.p.copy(), predictions.m.copy(),
              [t.copy() for t in predictions.topups])
    forecast = sc_combination(predictions, 1, 1, 1)
    forecast.customers["p"] = -1.0
    forecast.customers["key"] = "changed"
    pd.testing.assert_frame_equal(predictions.rows, before[0])
    pd.testing.assert_frame_equal(predictions.p, before[1])
    pd.testing.assert_frame_equal(predictions.m, before[2])
    for found, kept in zip(predictions.topups, before[3]):
        pd.testing.assert_frame_equal(found, kept)


def test_combination_forecast_is_deterministic():
    predictions = sc_predictions()
    first, second = sc_combination(predictions, 2, 0, 1), sc_combination(predictions, 2, 0, 1)
    pd.testing.assert_frame_equal(first.totals, second.totals)
    pd.testing.assert_frame_equal(first.customers, second.customers)


@pytest.mark.parametrize("part, bad", [("p", np.nan), ("m", np.nan), ("m", np.inf)])
def test_combination_forecast_a_bad_p_or_m_raises_value_error(part, bad):
    predictions = sc_predictions()
    frame = getattr(predictions, part).copy()
    frame.iloc[3, 1] = bad
    broken = remade(predictions, **{part: frame})
    with pytest.raises(ValueError):
        sc_combination(broken, 1, 1, 0)


def test_combination_forecast_a_company_missing_from_the_list_raises():
    with pytest.raises(ValueError):
        combination_forecast(sc_predictions(), 0, 0, 0, projected=sc_projected(),
                             companies=["TR02"], horizons=SC_H)


# --- choose_combination ------------------------------------------------------------------------

def choice_table(lines):
    """A grid_scores-like table from (stage1, stage2, topups, current, wape)."""
    frame = pd.DataFrame([(i, j, k, current, wape, 0.1, 0.0, 0.2)
                          for i, j, k, current, wape in lines], columns=GRID_SCORE_COLUMNS)
    frame["current"] = frame["current"].astype(bool)
    return frame


def test_choose_combination_returns_a_series_the_row_of_the_scores():
    table = choice_table([(0, 0, 0, True, 0.75), (1, 0, 0, False, 0.3)])
    found = choose_combination(table, 0.25)
    assert isinstance(found, pd.Series)
    pd.testing.assert_series_equal(found, table.iloc[1])


def test_choose_combination_nothing_better_keeps_the_current_row():
    table = choice_table([(0, 0, 0, True, 0.5), (1, 0, 0, False, 0.6), (0, 1, 0, False, 0.7)])
    found = choose_combination(table, 0.0)
    pd.testing.assert_series_equal(found, table.iloc[0])
    assert bool(found["current"])


def test_choose_combination_exactly_min_gain_better_is_not_enough():
    # 0.75 - 0.25 == 0.5 exactly in binary floating point.
    assert 0.75 - 0.25 == 0.5
    table = choice_table([(1, 0, 0, False, 0.5), (0, 0, 0, True, 0.75)])
    found = choose_combination(table, 0.25)
    assert bool(found["current"]) and found["wape"] == 0.75


def test_choose_combination_just_more_than_min_gain_better_wins():
    better = float(np.nextafter(0.5, 0.0))
    table = choice_table([(1, 0, 0, False, better), (0, 0, 0, True, 0.75)])
    found = choose_combination(table, 0.25)
    assert not bool(found["current"]) and found["wape"] == better


def test_choose_combination_min_gain_zero_an_equal_wape_keeps_the_current():
    table = choice_table([(1, 0, 0, False, 0.5), (0, 0, 0, True, 0.5)])
    assert bool(choose_combination(table, 0.0)["current"])


def test_choose_combination_min_gain_zero_any_strictly_lower_wape_wins():
    table = choice_table([(1, 0, 0, False, float(np.nextafter(0.5, 0.0))),
                          (0, 0, 0, True, 0.5)])
    assert not bool(choose_combination(table, 0.0)["current"])


def test_choose_combination_a_large_min_gain_keeps_even_the_worst_current():
    table = choice_table([(1, 0, 0, False, 0.1), (0, 1, 0, False, 0.2), (0, 0, 0, True, 0.9)])
    assert bool(choose_combination(table, 1.0)["current"])


def test_choose_combination_of_several_better_the_lowest_wape_wins():
    table = choice_table([(0, 0, 0, True, 0.75), (1, 0, 0, False, 0.4),
                          (0, 1, 0, False, 0.125), (1, 1, 1, False, 0.3)])
    pd.testing.assert_series_equal(choose_combination(table, 0.25), table.iloc[2])


def test_choose_combination_a_tie_among_the_better_gives_the_first_row():
    table = choice_table([(0, 0, 0, True, 0.75), (1, 1, 0, False, 0.375),
                          (0, 1, 1, False, 0.25), (1, 0, 1, False, 0.25)])
    found = choose_combination(table, 0.25)
    pd.testing.assert_series_equal(found, table.iloc[2])
    assert (found["stage1"], found["stage2"], found["topups"]) == (0, 1, 1)


def test_choose_combination_a_tie_keeps_the_table_order_whatever_the_index():
    table = choice_table([(0, 0, 0, True, 0.75), (2, 0, 0, False, 0.25),
                          (1, 0, 0, False, 0.25)])
    table.index = [30, 10, 20]
    found = choose_combination(table, 0.0)
    assert found.name == 10 and found["stage1"] == 2


def test_choose_combination_finds_the_current_by_its_flag_not_its_position():
    table = choice_table([(1, 0, 0, False, 0.6), (0, 1, 0, False, 0.7), (0, 0, 0, True, 0.5)])
    found = choose_combination(table, 0.0)
    pd.testing.assert_series_equal(found, table.iloc[2])


def test_choose_combination_the_returned_row_keeps_its_index_label():
    table = choice_table([(0, 0, 0, True, 0.75), (1, 0, 0, False, 0.25)])
    table.index = [7, 4]
    assert choose_combination(table, 0.25).name == 4
    assert choose_combination(table, 0.6).name == 7


def test_choose_combination_a_lone_current_row_is_kept():
    table = choice_table([(0, 0, 0, True, 0.5)])
    pd.testing.assert_series_equal(choose_combination(table, 0.005), table.iloc[0])


def test_choose_combination_a_nan_in_another_score_column_does_not_matter():
    # total_wape is NaN when no origin has all its weeks (grid_scores); only wape decides.
    table = choice_table([(0, 0, 0, True, 0.75), (1, 0, 0, False, 0.25)])
    table["total_wape"] = np.nan
    assert choose_combination(table, 0.25)["stage1"] == 1


def test_choose_combination_does_not_change_the_scores():
    table = choice_table([(0, 0, 0, True, 0.75), (1, 0, 0, False, 0.25), (0, 1, 0, False, 0.3)])
    before = table.copy()
    choose_combination(table, 0.0)
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("lines", [
    [(0, 0, 0, False, 0.5), (1, 0, 0, False, 0.4)],
    [(0, 0, 0, True, 0.5), (1, 0, 0, True, 0.4)],
    [],
])
def test_choose_combination_not_exactly_one_current_row_raises(lines):
    table = choice_table(lines) if lines else (
        pd.DataFrame(columns=GRID_SCORE_COLUMNS).astype({"current": bool, "wape": float}))
    with pytest.raises(ValueError, match="one is needed"):
        choose_combination(table, 0.005)


@pytest.mark.parametrize("nan_at", [0, 1])
def test_choose_combination_a_nan_wape_raises(nan_at):
    lines = [(0, 0, 0, True, 0.5), (1, 0, 0, False, 0.1)]
    i, j, k, current, _ = lines[nan_at]
    lines[nan_at] = (i, j, k, current, float("nan"))
    with pytest.raises(ValueError, match="no WAPE"):
        choose_combination(choice_table(lines), 0.005)


def test_choose_combination_a_nan_wape_raises_even_when_the_current_would_be_kept():
    table = choice_table([(0, 0, 0, True, 0.1), (1, 0, 0, False, float("nan"))])
    with pytest.raises(ValueError, match="no WAPE"):
        choose_combination(table, 10.0)


def test_choose_combination_the_current_check_comes_before_the_nan_check():
    table = choice_table([(0, 0, 0, False, float("nan")), (1, 0, 0, False, 0.1)])
    with pytest.raises(ValueError, match="one is needed"):
        choose_combination(table, 0.005)


def test_choose_combination_on_grid_scores_output(sc_result):
    _, _, table = sc_result
    current = table[table["current"]].iloc[0]
    pd.testing.assert_series_equal(choose_combination(table, 10.0), current)
    pd.testing.assert_series_equal(choose_combination(table, -1.0), table.iloc[0])


# --- CURRENT_SETTINGS and GridChoice -----------------------------------------------------------

def test_current_settings_name():
    assert CURRENT_SETTINGS == "current settings"


def test_current_settings_is_no_other_methods_name():
    assert CURRENT_SETTINGS not in {HURDLE, RULE, *REFERENCES}


def test_grid_choice_type_fields_in_order_and_frozen():
    assert [f.name for f in dataclasses.fields(GridChoice)] == [
        "winner", "stage1", "stage2", "topups", "forecast", "calibration", "scores",
        "bootstrap"]
    choice = GridChoice(*[f"<{n}>" for n in range(8)])
    with pytest.raises(dataclasses.FrozenInstanceError):
        choice.winner = "other"


# --- grid_choice with recording fakes, on the hand-made predictions ----------------------------

@dataclasses.dataclass(frozen=True)
class ChoiceSettings:
    """What grid_choice and the steps it calls read; a dataclass, as `dataclasses.replace`
    needs one."""
    receivables: object
    topups: object
    grid: object
    evaluation: object
    calibration: object
    diagnostic: object
    stage1: object = None
    stage2: object = None


def ch_settings(min_gain=0.005, samples=40, seed=9, topups="<settings topups>"):
    return ChoiceSettings(
        receivables=SimpleNamespace(horizons=SC_H, min_cleared_for_delay=3),
        topups=topups,
        grid=SimpleNamespace(min_gain=min_gain),
        evaluation=SimpleNamespace(bootstrap_samples=samples, bootstrap_seed=seed,
                                   rule_doubtful_days=90),
        calibration=SimpleNamespace(company_shrink_rows=200, min_gain=0.005),
        diagnostic=SimpleNamespace(calibration_groups=2))


def fake_reference(actual):
    """deflated_reference's shape: each variant a fixed multiple of the actual cash."""
    frames = []
    for factor, name in zip((1.05, 1.10, 0.95), REFERENCES):
        frame = actual[CH_KEYS].copy()
        frame["reference"] = name
        frame["real_forecast"] = actual["actual_real"] * factor
        frame["price_level"] = 1.0
        frame["forecast"] = actual["actual"] * factor
        frame["fallback"] = False
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[REFERENCE_COLUMNS]


def fake_rule_totals(actual):
    return actual[CH_KEYS].assign(forecast=actual["actual"] * 0.9)


class Recorder:
    """Replaces choose_calibration, deflated_reference and rule_reference as grid sees them."""

    def __init__(self, monkeypatch, actual, fail=None):
        self.calls = []
        self.fail = fail
        self.actual = actual
        monkeypatch.setattr(grid_module, "choose_calibration", self.calibration)
        monkeypatch.setattr(grid_module, "deflated_reference", self.reference)
        monkeypatch.setattr(grid_module, "rule_reference", self.rule)

    def check(self, name):
        if self.fail == name:
            raise ValueError(f"{name} failed")

    def calibration(self, *args, **kwargs):
        self.calls.append(("calibration", args, kwargs))
        self.check("calibration")
        return "<calibration>"

    def reference(self, *args, **kwargs):
        self.calls.append(("reference", args, kwargs))
        self.check("reference")
        return fake_reference(self.actual)

    def rule(self, *args, **kwargs):
        self.calls.append(("rule", args, kwargs))
        self.check("rule")
        return SimpleNamespace(totals=fake_rule_totals(self.actual), invoices="<invoices>")

    def only(self, name):
        found = [(args, kwargs) for called, args, kwargs in self.calls if called == name]
        assert len(found) == 1, name
        return found[0]


CH_SIDE = dict(weekly="<weekly>", weeks="<weeks>", levels="<levels>", calendar="<calendar>",
               ledger="<ledger>", rates="<rates>", new_cash="<new cash>", refunds="<refunds>",
               unidentified="<unidentified>")


def ch_run(predictions, scores, actual, settings=None, candidates=None, **overrides):
    kwargs = dict(CH_SIDE, companies=list(SC_COMPANIES), week_zero=WEEK_ZERO)
    kwargs.update(overrides)
    return grid_choice(predictions, candidates or sc_candidates(), scores, actual,
                       settings or ch_settings(), **kwargs)


def without_current(table, also=None):
    """`table` with the current settings' WAPE (and those of rows matching `also`) made 1
    worse, so that another combination is chosen; every other row keeps grid_scores' WAPE."""
    table = table.copy()
    worse = table["current"].copy()
    if also is not None:
        worse |= also(table)
    table.loc[worse, "wape"] += 1.0
    return table


@pytest.fixture
def ch_case(sc_result, monkeypatch):
    predictions, actual, table = sc_result
    return predictions, actual, table, Recorder(monkeypatch, actual)


def stacked_by_hand(predictions, i, j, k, actual, with_current):
    forecast = sc_combination(predictions, i, j, k)
    rule = SimpleNamespace(totals=fake_rule_totals(actual))
    stacked = stack_forecasts(forecast, fake_reference(actual), rule)
    if with_current:
        current = sc_combination(predictions, 0, 0, 0).totals
        current = current[CH_KEYS + ["total"]].rename(columns={"total": "forecast"})
        current = current.assign(method=CURRENT_SETTINGS)[STACK_COLUMNS]
        stacked = pd.concat([stacked, current], ignore_index=True)
        stacked["company_code"] = stacked["company_code"].astype(object)
    return stacked


def winner_numbers(choice):
    return tuple(int(choice.winner[c]) for c in ("stage1", "stage2", "topups"))


def test_grid_choice_returns_a_grid_choice(ch_case):
    predictions, actual, table, _ = ch_case
    assert isinstance(ch_run(predictions, table, actual), GridChoice)


def test_grid_choice_current_wins_the_winner_is_the_current_row(ch_case):
    predictions, actual, table, _ = ch_case
    choice = ch_run(predictions, table, actual, settings=ch_settings(min_gain=10.0))
    pd.testing.assert_series_equal(choice.winner, table[table["current"]].iloc[0])
    candidates = sc_candidates()
    assert (choice.stage1, choice.stage2, choice.topups) == (
        candidates.stage1[0], candidates.stage2[0], candidates.topups[0])


@pytest.mark.parametrize("min_gain", [10.0, 0.0, -1.0])
def test_grid_choice_winner_is_choose_combination_with_grid_min_gain(ch_case, min_gain):
    predictions, actual, table, _ = ch_case
    choice = ch_run(predictions, table, actual, settings=ch_settings(min_gain=min_gain))
    pd.testing.assert_series_equal(choice.winner, choose_combination(table, min_gain))


def test_grid_choice_new_winner_gives_its_candidates(ch_case):
    predictions, actual, table, _ = ch_case
    doctored = without_current(table)
    choice = ch_run(predictions, doctored, actual, settings=ch_settings(min_gain=0.0))
    expected = choose_combination(doctored, 0.0)
    assert not bool(choice.winner["current"])
    pd.testing.assert_series_equal(choice.winner, expected)
    i, j, k = winner_numbers(choice)
    assert (choice.stage1, choice.stage2, choice.topups) == (f"<s1 {i}>", f"<s2 {j}>",
                                                             f"<t {k}>")


def test_grid_choice_forecast_is_the_winners_combination_forecast(ch_case):
    predictions, actual, table, _ = ch_case
    choice = ch_run(predictions, without_current(table), actual,
                    settings=ch_settings(min_gain=0.0))
    expected = sc_combination(predictions, *winner_numbers(choice))
    pd.testing.assert_frame_equal(choice.forecast.totals, expected.totals)
    pd.testing.assert_frame_equal(choice.forecast.customers, expected.customers)


def test_grid_choice_calibration_is_choose_calibration_on_the_winners_forecast(ch_case):
    predictions, actual, table, recorder = ch_case
    settings = ch_settings(min_gain=0.0)
    choice = ch_run(predictions, without_current(table), actual, settings=settings)
    assert choice.calibration == "<calibration>"
    args, kwargs = recorder.only("calibration")
    assert kwargs == {}
    forecast, weekly, weeks, levels, given_actual, given_settings = args
    assert forecast is choice.forecast
    assert (weekly, weeks, levels) == ("<weekly>", "<weeks>", "<levels>")
    assert given_actual is actual
    # The settings themselves, not the copy with the chosen top-ups.
    assert given_settings is settings


def test_grid_choice_deflated_reference_at_the_predictions_origins(ch_case):
    predictions, actual, table, recorder = ch_case
    ch_run(predictions, table, actual)
    args, kwargs = recorder.only("reference")
    assert kwargs == {}
    assert args == ("<weekly>", "<weeks>", "<levels>", "<calendar>", list(SC_COMPANIES),
                    sorted(SC_ORIGINS), SC_H, WEEK_ZERO)


def test_grid_choice_rule_reference_gets_its_inputs(ch_case):
    predictions, actual, table, recorder = ch_case
    ch_run(predictions, table, actual)
    args, kwargs = recorder.only("rule")
    assert args[:3] == ("<ledger>", "<rates>", sorted(SC_ORIGINS))
    assert kwargs == {"companies": list(SC_COMPANIES), "new_cash": "<new cash>",
                      "refunds": "<refunds>", "unidentified": "<unidentified>",
                      "levels": "<levels>", "calendar": "<calendar>", "week_zero": WEEK_ZERO}


def test_grid_choice_origins_are_sorted_and_once_each(ch_case):
    predictions, actual, table, recorder = ch_case
    rows = predictions.rows.sort_values("origin_week", ascending=False, kind="stable")
    ch_run(remade(predictions, rows=rows), table, actual)
    assert recorder.only("reference")[0][5] == [18, 22]
    assert recorder.only("rule")[0][2] == [18, 22]


@pytest.mark.parametrize("k", [0, 1])
def test_grid_choice_rule_uses_the_chosen_topups_not_the_settings_topups(ch_case, k):
    predictions, actual, table, recorder = ch_case
    settings = ch_settings(min_gain=0.0)
    doctored = without_current(table, also=lambda t: t["topups"] != k)
    choice = ch_run(predictions, doctored, actual, settings=settings)
    assert int(choice.winner["topups"]) == k
    given = recorder.only("rule")[0][3]
    assert given.topups == f"<t {k}>"
    assert given.topups == choice.topups
    # Only the top-ups are replaced; the settings given stay as they were.
    assert dataclasses.replace(given, topups=settings.topups) == settings
    assert settings.topups == "<settings topups>"


def test_grid_choice_rule_uses_candidate_0s_topups_when_the_current_wins(ch_case):
    predictions, actual, table, recorder = ch_case
    choice = ch_run(predictions, table, actual, settings=ch_settings(min_gain=10.0))
    assert recorder.only("rule")[0][3].topups == "<t 0>" == choice.topups


def test_grid_choice_each_step_is_called_once(ch_case):
    predictions, actual, table, recorder = ch_case
    ch_run(predictions, table, actual)
    assert sorted(name for name, _, _ in recorder.calls) == ["calibration", "reference", "rule"]


def test_grid_choice_current_wins_the_methods_have_no_current_settings(ch_case):
    predictions, actual, table, _ = ch_case
    choice = ch_run(predictions, table, actual, settings=ch_settings(min_gain=10.0))
    assert list(dict.fromkeys(choice.scores["method"])) == [HURDLE, *REFERENCES, RULE]
    assert CURRENT_SETTINGS not in set(choice.bootstrap["against"])


def test_grid_choice_new_winner_adds_the_current_settings_last(ch_case):
    predictions, actual, table, _ = ch_case
    choice = ch_run(predictions, without_current(table), actual,
                    settings=ch_settings(min_gain=0.0))
    assert list(dict.fromkeys(choice.scores["method"])) == [HURDLE, *REFERENCES, RULE,
                                                            CURRENT_SETTINGS]


@pytest.mark.parametrize("new", [False, True])
def test_grid_choice_scores_equal_evaluation_scores_of_the_stack_by_hand(ch_case, new):
    predictions, actual, table, _ = ch_case
    scores = without_current(table) if new else table
    choice = ch_run(predictions, scores, actual,
                    settings=ch_settings(min_gain=0.0 if new else 10.0))
    stacked = stacked_by_hand(predictions, *winner_numbers(choice), actual, with_current=new)
    pd.testing.assert_frame_equal(choice.scores, evaluation_scores(stacked, actual, SC_H))


@pytest.mark.parametrize("new", [False, True])
def test_grid_choice_the_winners_scores_equal_its_grid_scores_row(ch_case, new):
    predictions, actual, table, _ = ch_case
    scores = without_current(table) if new else table
    choice = ch_run(predictions, scores, actual,
                    settings=ch_settings(min_gain=0.0 if new else 10.0))
    line = sc_line(table, *winner_numbers(choice))          # the untouched grid_scores row
    hurdle = choice.scores[choice.scores["method"] == HURDLE]
    assert pooled(hurdle, CONSOLIDATED) == line["wape"]
    assert pooled(hurdle, CONSOLIDATED, "bias") == line["bias"]
    assert pooled(hurdle, CONSOLIDATED, horizon=HORIZON_TOTAL) == line["total_wape"]
    assert np.mean([pooled(hurdle, c) for c in SC_COMPANIES]) == pytest.approx(
        line["company_wape"], rel=1e-15)


def test_grid_choice_current_settings_scores_equal_the_current_grid_scores_row(ch_case):
    predictions, actual, table, _ = ch_case
    choice = ch_run(predictions, without_current(table), actual,
                    settings=ch_settings(min_gain=0.0))
    current = choice.scores[choice.scores["method"] == CURRENT_SETTINGS]
    line = sc_line(table, 0, 0, 0)
    assert pooled(current, CONSOLIDATED) == line["wape"]
    assert pooled(current, CONSOLIDATED, "bias") == line["bias"]
    assert pooled(current, CONSOLIDATED, horizon=HORIZON_TOTAL) == line["total_wape"]


@pytest.mark.parametrize("new", [False, True])
def test_grid_choice_bootstrap_is_hurdle_against_every_other_method(ch_case, new):
    predictions, actual, table, _ = ch_case
    scores = without_current(table) if new else table
    choice = ch_run(predictions, scores, actual,
                    settings=ch_settings(min_gain=0.0 if new else 10.0, samples=40, seed=9))
    others = [*REFERENCES, RULE] + ([CURRENT_SETTINGS] if new else [])
    stacked = stacked_by_hand(predictions, *winner_numbers(choice), actual, with_current=new)
    expected = paired_wape_bootstrap(stacked, actual, [(HURDLE, o) for o in others], SC_H,
                                     40, 9)
    pd.testing.assert_frame_equal(choice.bootstrap, expected)
    assert set(choice.bootstrap["method"]) == {HURDLE}
    assert list(dict.fromkeys(choice.bootstrap["against"])) == others


def test_grid_choice_bootstrap_reads_the_evaluation_samples_and_seed(ch_case):
    predictions, actual, table, _ = ch_case
    first = ch_run(predictions, table, actual, settings=ch_settings(samples=30, seed=1))
    assert set(first.bootstrap["samples"]) == {30}
    second = ch_run(predictions, table, actual, settings=ch_settings(samples=30, seed=2))
    assert not first.bootstrap[["low", "high"]].equals(second.bootstrap[["low", "high"]])


def test_grid_choice_does_not_change_its_inputs(ch_case):
    predictions, actual, table, _ = ch_case
    doctored = without_current(table)
    before = (predictions.rows.copy(), predictions.p.copy(), predictions.m.copy(),
              doctored.copy(), actual.copy())
    settings = ch_settings(min_gain=0.0)
    ch_run(predictions, doctored, actual, settings=settings)
    pd.testing.assert_frame_equal(predictions.rows, before[0])
    pd.testing.assert_frame_equal(predictions.p, before[1])
    pd.testing.assert_frame_equal(predictions.m, before[2])
    pd.testing.assert_frame_equal(doctored, before[3])
    pd.testing.assert_frame_equal(actual, before[4])
    assert settings == ch_settings(min_gain=0.0)


def test_grid_choice_is_deterministic(ch_case):
    predictions, actual, table, _ = ch_case
    first = ch_run(predictions, table, actual)
    second = ch_run(predictions, table, actual)
    pd.testing.assert_frame_equal(first.scores, second.scores)
    pd.testing.assert_frame_equal(first.bootstrap, second.bootstrap)
    pd.testing.assert_series_equal(first.winner, second.winner)


@pytest.mark.parametrize("step", ["calibration", "reference", "rule"])
def test_grid_choice_a_failing_step_passes_its_error_through(sc_result, monkeypatch, step):
    predictions, actual, table = sc_result
    Recorder(monkeypatch, actual, fail=step)
    with pytest.raises(ValueError, match=f"{step} failed"):
        ch_run(predictions, table, actual)


def test_grid_choice_two_current_rows_raise_before_any_step(ch_case):
    predictions, actual, table, recorder = ch_case
    broken = table.copy()
    broken.loc[broken.index[-1], "current"] = True
    with pytest.raises(ValueError, match="one is needed"):
        ch_run(predictions, broken, actual)
    assert recorder.calls == []


def test_grid_choice_a_nan_wape_raises_before_any_step(ch_case):
    predictions, actual, table, recorder = ch_case
    broken = table.copy()
    broken.loc[broken.index[-1], "wape"] = np.nan
    with pytest.raises(ValueError, match="no WAPE"):
        ch_run(predictions, broken, actual)
    assert recorder.calls == []


def test_grid_choice_a_broken_winner_p_raises_as_forecast_totals_does(ch_case):
    predictions, actual, table, _ = ch_case
    p = predictions.p.copy()
    p.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        ch_run(remade(predictions, p=p), table, actual, settings=ch_settings(min_gain=10.0))


def test_grid_choice_a_reference_without_a_variant_raises_as_stack_forecasts_does(
        sc_result, monkeypatch):
    predictions, actual, table = sc_result
    Recorder(monkeypatch, actual)
    reference = fake_reference(actual)
    monkeypatch.setattr(grid_module, "deflated_reference", lambda *args, **kwargs: reference[
        reference["reference"] != "deflated_trend"])
    with pytest.raises(ValueError, match="deflated_trend"):
        ch_run(predictions, table, actual)


# --- grid_choice on a fitted grid at three origins, everything real ----------------------------
#
# Origins 18, 22, 26 (the calibration judges at the origins after the middle target week, 25;
# two origins would leave none). The weekly cash holds the table's own customer keys, so the
# calibration rows see who paid; actual, deflated and calibration all read it.

FIT_ORIGINS = [18, 22, 26]


def fit_settings(min_gain=0.005):
    base = g_settings()
    return ChoiceSettings(
        receivables=base.receivables, topups=base.topups, stage1=base.stage1, stage2=base.stage2,
        grid=SimpleNamespace(min_gain=min_gain),
        evaluation=SimpleNamespace(bootstrap_samples=40, bootstrap_seed=5,
                                   rule_doubtful_days=90),
        calibration=SimpleNamespace(company_shrink_rows=50, min_gain=0.005),
        diagnostic=SimpleNamespace(reference_window=G_WINDOW, amount_reference_window=G_WINDOW,
                                   reference_prior_weeks=1.0, calibration_groups=2))


def fit_weekly():
    rng = np.random.default_rng(21)
    lines = []
    for c in range(8):
        company = F_COMPANIES[c % 2]
        for week in range(0, 41):
            if rng.random() < 0.55:
                lines.append((f"{company}|C{c}", company, f"C{c}", week,
                              rng.uniform(200, 900) * (c + 1), -rng.uniform(0, 30)))
    for company in F_COMPANIES:
        for week in range(0, 41):
            lines.append((f"{company}|U", company, None, week, rng.uniform(300, 900), 0.0))
    return make_weekly(lines)


@pytest.fixture(scope="module")
def fit_case(g_table, tmp_path_factory):
    settings = fit_settings()
    candidates = g_candidates(settings)
    predictions = grid_predictions(g_table, FIT_ORIGINS, candidates, settings,
                                   cache_folder=tmp_path_factory.mktemp("choice_cache"),
                                   **g_inputs())
    weekly, weeks, levels = fit_weekly(), make_weeks(40), f_levels()
    actual = actual_totals(weekly, weeks, levels, list(F_COMPANIES), FIT_ORIGINS, 5)
    table = grid_scores(predictions, candidates, g_gates(candidates), actual, settings,
                        companies=list(F_COMPANIES), week_zero=WEEK_ZERO)
    rolled = rolling_forecasts(g_table, FIT_ORIGINS, settings, week_zero=WEEK_ZERO,
                               **g_inputs())
    new_cash, refunds, unidentified = f_side_tables()
    side = dict(weekly=weekly, weeks=weeks, levels=levels, calendar=f_calendar(),
                ledger=running_ledger(), rates=rule_rates(), new_cash=new_cash,
                refunds=refunds, unidentified=unidentified)
    return SimpleNamespace(settings=settings, candidates=candidates, predictions=predictions,
                           actual=actual, table=table, rolled=rolled, side=side)


def fit_run(case, scores=None, min_gain=0.005):
    return grid_choice(case.predictions, case.candidates,
                       case.table if scores is None else scores, case.actual,
                       fit_settings(min_gain), companies=list(F_COMPANIES), week_zero=WEEK_ZERO,
                       **case.side)


def fit_projected():
    return projected_at(FIT_ORIGINS, 5, WEEK_ZERO)


@pytest.fixture(scope="module")
def fit_kept(fit_case):
    return fit_run(fit_case, min_gain=10.0)


@pytest.fixture(scope="module")
def fit_new(fit_case):
    # The current WAPE and every row with candidate-0 top-ups made worse: a combination with
    # the other top-ups (window 4, not [topups]' 8) wins with its real grid_scores WAPE.
    scores = without_current(fit_case.table, also=lambda t: t["topups"] == 0)
    return fit_run(fit_case, scores=scores, min_gain=0.0)


def assert_same_calibration(found, expected):
    assert isinstance(found, CalibrationChoice)
    assert found.p_correction == expected.p_correction
    assert found.m_correction == expected.m_correction
    assert found.split_week == expected.split_week
    assert found.judged_origins == expected.judged_origins
    assert found.winner == expected.winner
    pd.testing.assert_frame_equal(found.candidates, expected.candidates)
    pd.testing.assert_frame_equal(found.bootstrap, expected.bootstrap)


def test_fitted_combination_0_0_0_equals_rolling_forecasts_totals(fit_case):
    found = combination_forecast(fit_case.predictions, 0, 0, 0, projected=fit_projected(),
                                 companies=list(F_COMPANIES), horizons=5)
    pd.testing.assert_frame_equal(found.totals, fit_case.rolled.totals)


def test_fitted_combination_0_0_0_equals_rolling_forecasts_customers(fit_case):
    found = combination_forecast(fit_case.predictions, 0, 0, 0, projected=fit_projected(),
                                 companies=list(F_COMPANIES), horizons=5)
    pd.testing.assert_frame_equal(found.customers, fit_case.rolled.customers)


def test_fitted_kept_the_current_settings(fit_case, fit_kept):
    assert bool(fit_kept.winner["current"])
    assert fit_kept.stage1 is fit_case.candidates.stage1[0]
    assert fit_kept.stage2 is fit_case.candidates.stage2[0]
    assert fit_kept.topups is fit_case.candidates.topups[0]


def test_fitted_kept_forecast_is_step_17s(fit_case, fit_kept):
    pd.testing.assert_frame_equal(fit_kept.forecast.totals, fit_case.rolled.totals)
    pd.testing.assert_frame_equal(fit_kept.forecast.customers, fit_case.rolled.customers)


def test_fitted_kept_calibration_is_step_17s_choice(fit_case, fit_kept):
    case = fit_case
    step_17 = choose_calibration(case.rolled, case.side["weekly"], case.side["weeks"],
                                 case.side["levels"], case.actual, case.settings)
    assert_same_calibration(fit_kept.calibration, step_17)


def test_fitted_kept_hurdle_scores_equal_the_current_grid_row_and_step_17(fit_case, fit_kept):
    case = fit_case
    hurdle = fit_kept.scores[fit_kept.scores["method"] == HURDLE].reset_index(drop=True)
    line = case.table[case.table["current"]].iloc[0]
    assert pooled(hurdle, CONSOLIDATED) == line["wape"]
    assert pooled(hurdle, CONSOLIDATED, "bias") == line["bias"]
    pd.testing.assert_frame_equal(hurdle, hurdle_scores(case.rolled, case.actual)
                                  .reset_index(drop=True))


def test_fitted_kept_has_no_current_settings_method(fit_kept):
    assert CURRENT_SETTINGS not in set(fit_kept.scores["method"])
    assert list(dict.fromkeys(fit_kept.bootstrap["against"])) == [*REFERENCES, RULE]


def test_fitted_references_equal_the_reference_functions_by_hand(fit_case, fit_kept):
    case = fit_case
    side = case.side
    reference = deflated_reference(side["weekly"], side["weeks"], side["levels"],
                                   side["calendar"], list(F_COMPANIES), FIT_ORIGINS, 5,
                                   WEEK_ZERO)
    rule = rule_reference(side["ledger"], side["rates"], FIT_ORIGINS, case.settings,
                          companies=list(F_COMPANIES), new_cash=side["new_cash"],
                          refunds=side["refunds"], unidentified=side["unidentified"],
                          levels=side["levels"], calendar=side["calendar"], week_zero=WEEK_ZERO)
    stacked = stack_forecasts(case.rolled, reference, rule)
    pd.testing.assert_frame_equal(fit_kept.scores, evaluation_scores(stacked, case.actual, 5))


def test_fitted_new_winner_has_the_other_topups(fit_case, fit_new):
    assert not bool(fit_new.winner["current"])
    assert int(fit_new.winner["topups"]) == 1
    assert fit_new.topups is fit_case.candidates.topups[1]
    assert fit_new.topups != fit_case.settings.topups


def test_fitted_new_winner_scores_equal_its_grid_scores_row(fit_case, fit_new):
    i, j, k = winner_numbers(fit_new)
    table = fit_case.table
    line = table[(table["stage1"] == i) & (table["stage2"] == j) & (table["topups"] == k)]
    line = line.iloc[0]
    hurdle = fit_new.scores[fit_new.scores["method"] == HURDLE]
    assert pooled(hurdle, CONSOLIDATED) == line["wape"]
    assert pooled(hurdle, CONSOLIDATED, "bias") == line["bias"]
    assert pooled(hurdle, CONSOLIDATED, horizon=HORIZON_TOTAL) == line["total_wape"]
    assert fit_new.winner["wape"] == line["wape"]


def test_fitted_new_current_settings_scores_are_step_17s_model(fit_case, fit_new):
    case = fit_case
    current = fit_new.scores[fit_new.scores["method"] == CURRENT_SETTINGS]
    step_17 = hurdle_scores(case.rolled, case.actual)
    pd.testing.assert_frame_equal(current.drop(columns="method").reset_index(drop=True),
                                  step_17.drop(columns="method").reset_index(drop=True))


def test_fitted_new_rule_uses_the_chosen_topups(fit_case, fit_new):
    case = fit_case
    side = case.side

    def rule_scores(topups):
        rule = rule_reference(side["ledger"], side["rates"], FIT_ORIGINS,
                              dataclasses.replace(case.settings, topups=topups),
                              companies=list(F_COMPANIES), new_cash=side["new_cash"],
                              refunds=side["refunds"], unidentified=side["unidentified"],
                              levels=side["levels"], calendar=side["calendar"],
                              week_zero=WEEK_ZERO)
        stacked = rule.totals[CH_KEYS + ["forecast"]].assign(method=RULE)[STACK_COLUMNS]
        return evaluation_scores(stacked, case.actual, 5).reset_index(drop=True)

    found = fit_new.scores[fit_new.scores["method"] == RULE].reset_index(drop=True)
    pd.testing.assert_frame_equal(found, rule_scores(case.candidates.topups[1]))
    with_settings_topups = rule_scores(case.settings.topups)
    assert pooled(found, CONSOLIDATED) != pooled(with_settings_topups, CONSOLIDATED)


def test_fitted_new_calibration_is_choose_calibration_on_its_forecast(fit_case, fit_new):
    case = fit_case
    forecast = combination_forecast(case.predictions, *winner_numbers(fit_new),
                                    projected=fit_projected(), companies=list(F_COMPANIES),
                                    horizons=5)
    pd.testing.assert_frame_equal(fit_new.forecast.totals, forecast.totals)
    by_hand = choose_calibration(forecast, case.side["weekly"], case.side["weeks"],
                                 case.side["levels"], case.actual, case.settings)
    assert_same_calibration(fit_new.calibration, by_hand)


def test_fitted_new_bootstrap_pairs_hurdle_with_every_other_method(fit_new):
    assert set(fit_new.bootstrap["method"]) == {HURDLE}
    assert list(dict.fromkeys(fit_new.bootstrap["against"])) == [*REFERENCES, RULE,
                                                                 CURRENT_SETTINGS]
    assert set(fit_new.bootstrap["samples"]) == {40}


@pytest.mark.parametrize("min_gain", [0.0, 0.005, 10.0])
def test_fitted_on_the_natural_scores_the_winner_follows_the_min_gain_rule(fit_case, min_gain):
    # Whatever this grid's numbers: kept unless some WAPE is more than min_gain lower.
    case = fit_case
    choice = fit_run(case, min_gain=min_gain)
    pd.testing.assert_series_equal(choice.winner, choose_combination(case.table, min_gain))
    current = case.table[case.table["current"]].iloc[0]["wape"]
    better = (case.table["wape"] < current - min_gain).any()
    assert bool(choice.winner["current"]) == (not better)


# =============================================================================================
# U16.7: `method_rows`, `UNCORRECTED`, `GridTest` and `grid_test`.
#
# `method_rows` on the hand-made U16.5 predictions, against the inline code `grid_choice`
# used before it (the U16.6 tests above still pin grid_choice's output). `grid_test` on the
# U16.3 table, really fitted at the test origins 18 and 22 (G_ORIGINS, so the U16.3 grid and
# step 17's rolling forecasts at the same origins are the references to compare with), with a
# hand-made GridChoice: its validation forecast only needs `customers["origin_week"]`, its
# calibration a hand-made CalibrationChoice whose corrections are distinctive, so that applying
# them unchanged (and not learning them again) can be told apart. Every cache is a tmp_path.
# =============================================================================================

from collection_estimation.hurdle import calibrate as calibrate_module  # noqa: E402
from collection_estimation.hurdle.calibrate import (  # noqa: E402
    ALL_H, ONE_MAP, PER_H, SCALE, MCorrection, PCorrection, corrected_forecast)
from collection_estimation.hurdle.grid import (  # noqa: E402
    UNCORRECTED, GridTest, grid_test, method_rows)

T_ORIGINS = [18, 22]
T_VALIDATION = (9, 13)          # the last one forecasts up to week 18: just before week 19


# --- UNCORRECTED and method_rows ---------------------------------------------------------------

def test_uncorrected_name():
    assert UNCORRECTED == "hurdle uncorrected"


def test_uncorrected_is_no_other_methods_name():
    assert UNCORRECTED not in {HURDLE, CURRENT_SETTINGS, RULE, *REFERENCES}


def old_inline_rows(forecast, method):
    """What grid_choice did inline before U16.7 moved it into method_rows."""
    current = forecast.totals
    current = current[CH_KEYS + ["total"]].rename(columns={"total": "forecast"})
    current.insert(0, "method", method)
    current["company_code"] = current["company_code"].astype(object)
    return current[STACK_COLUMNS]


@pytest.mark.parametrize("ijk", [(0, 0, 0), (2, 1, 1)])
def test_method_rows_equals_grid_choices_old_inline_code(sc_result, ijk):
    predictions, _, _ = sc_result
    forecast = sc_combination(predictions, *ijk)
    pd.testing.assert_frame_equal(method_rows(forecast, CURRENT_SETTINGS),
                                  old_inline_rows(forecast, CURRENT_SETTINGS))


def test_method_rows_shape_columns_and_values(sc_result):
    predictions, _, _ = sc_result
    forecast = sc_combination(predictions, 1, 2, 0)
    rows = method_rows(forecast, "some method")
    assert list(rows.columns) == STACK_COLUMNS
    assert len(rows) == len(forecast.totals)
    assert set(rows["method"]) == {"some method"}
    assert rows["company_code"].dtype == object
    pd.testing.assert_index_equal(rows.index, forecast.totals.index)
    for column in CH_KEYS:
        assert list(rows[column]) == list(forecast.totals[column]), column
    assert rows["forecast"].to_numpy().tolist() == forecast.totals["total"].to_numpy().tolist()


def test_method_rows_company_code_is_object_even_from_a_category(sc_result):
    predictions, _, _ = sc_result
    forecast = sc_combination(predictions, 0, 0, 0)
    totals = forecast.totals.assign(company_code=forecast.totals["company_code"].astype("category"))
    rows = method_rows(Forecast(totals=totals, customers=forecast.customers), "x")
    assert rows["company_code"].dtype == object
    assert list(rows["company_code"]) == list(forecast.totals["company_code"].astype(str))


def test_method_rows_does_not_change_the_forecast(sc_result):
    predictions, _, _ = sc_result
    forecast = sc_combination(predictions, 0, 0, 0)
    before = forecast.totals.copy()
    rows = method_rows(forecast, "x")
    rows.loc[rows.index[0], "forecast"] = -1.0
    pd.testing.assert_frame_equal(forecast.totals, before)


def test_method_rows_stacks_with_stack_forecasts_without_dtype_trouble(sc_result):
    predictions, actual, _ = sc_result
    forecast = sc_combination(predictions, 0, 0, 0)
    stacked = stack_forecasts(forecast, fake_reference(actual),
                              SimpleNamespace(totals=fake_rule_totals(actual)))
    both = pd.concat([stacked, method_rows(forecast, UNCORRECTED)], ignore_index=True)
    assert both["company_code"].dtype == object
    hurdle = both[both["method"] == HURDLE].drop(columns="method").reset_index(drop=True)
    plain = both[both["method"] == UNCORRECTED].drop(columns="method").reset_index(drop=True)
    pd.testing.assert_frame_equal(hurdle, plain)


# --- GridTest ----------------------------------------------------------------------------------

def test_grid_test_type_fields_in_order_and_frozen():
    assert [f.name for f in dataclasses.fields(GridTest)] == ["forecast", "uncorrected",
                                                              "scores", "bootstrap"]
    found = GridTest("<f>", "<u>", "<s>", "<b>")
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.forecast = "other"


# --- grid_test: the case ---------------------------------------------------------------------

def t_settings(test_origins=tuple(T_ORIGINS), samples=40, seed=5, topups=None):
    """fit_settings with [evaluation] test_origins; a frozen dataclass (grid_test calls
    `dataclasses.replace` on it)."""
    base = fit_settings()
    evaluation = SimpleNamespace(bootstrap_samples=samples, bootstrap_seed=seed,
                                 rule_doubtful_days=90, test_origins=tuple(test_origins),
                                 validation_origins="<never read>")
    changes = dict(evaluation=evaluation)
    if topups is not None:
        changes["topups"] = topups
    return dataclasses.replace(base, **changes)


def t_calibration(p=None, m=None):
    return CalibrationChoice(p_correction=p or PCorrection("none"),
                             m_correction=m or MCorrection("none"), split_week=11,
                             judged_origins=[13], candidates=pd.DataFrame(),
                             winner="<winner>", bootstrap=pd.DataFrame())


def t_choice(ijk=(0, 0, 0), p=None, m=None, validation=T_VALIDATION, settings=None):
    """A GridChoice as grid_choice gives it, with g_candidates' candidates `ijk`; its forecast
    holds only the validation origins of its customers (all grid_test reads of it)."""
    candidates = g_candidates(settings or t_settings())
    i, j, k = ijk
    forecast = SimpleNamespace(customers=pd.DataFrame({"origin_week": list(validation)}),
                               totals="<validation totals>")
    return GridChoice(winner="<winner>", stage1=candidates.stage1[i],
                      stage2=candidates.stage2[j], topups=candidates.topups[k],
                      forecast=forecast, calibration=t_calibration(p, m),
                      scores="<validation scores>", bootstrap="<validation bootstrap>")


def t_actual(origins=T_ORIGINS):
    return actual_totals(fit_weekly(), make_weeks(40), f_levels(), list(F_COMPANIES),
                         list(origins), 5)


def t_side():
    new_cash, refunds, unidentified = f_side_tables()
    return dict(weekly=fit_weekly(), weeks=make_weeks(40), levels=f_levels(),
                calendar=f_calendar(), ledger=running_ledger(), rates=rule_rates(),
                new_cash=new_cash, refunds=refunds, unidentified=unidentified)


def t_run(table, choice, cache_folder, actual=None, settings=None, on_fit_done=None,
          **overrides):
    kwargs = dict(t_side(), table=table, block_columns=BLOCK_COLUMNS,
                  companies=list(F_COMPANIES), week_zero=WEEK_ZERO, cache_folder=cache_folder,
                  on_fit_done=on_fit_done)
    kwargs.update(overrides)
    return grid_test(choice, t_actual() if actual is None else actual,
                     settings or t_settings(), **kwargs)


def t_stacked(found, choice, settings):
    """The stack grid_test scores, made by hand with the real references."""
    side = t_side()
    reference = deflated_reference(side["weekly"], side["weeks"], side["levels"],
                                   side["calendar"], list(F_COMPANIES), T_ORIGINS, 5, WEEK_ZERO)
    rule = rule_reference(side["ledger"], side["rates"], T_ORIGINS,
                          dataclasses.replace(settings, topups=choice.topups),
                          companies=list(F_COMPANIES), new_cash=side["new_cash"],
                          refunds=side["refunds"], unidentified=side["unidentified"],
                          levels=side["levels"], calendar=side["calendar"], week_zero=WEEK_ZERO)
    return pd.concat([stack_forecasts(found.forecast, reference, rule),
                      method_rows(found.uncorrected, UNCORRECTED)], ignore_index=True)


# A distinctive m scale and a p map, applied as frozen on validation.
T_SCALE = MCorrection(SCALE, common=1.2345)
T_MAP = PCorrection(ONE_MAP, {ALL_H: (0.8, 0.3)})


@pytest.fixture(scope="module")
def t_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("test_cache")


@pytest.fixture(scope="module")
def t_plain(g_table, t_cache):
    """grid_test with the current settings and no corrections, on a fresh cache."""
    progress = []
    found = t_run(g_table, t_choice(), t_cache,
                  on_fit_done=lambda *args: progress.append(args))
    return found, progress


@pytest.fixture(scope="module")
def t_corrected(g_table, t_cache, t_plain):
    """grid_test with the current settings and the distinctive corrections (fits read back)."""
    return t_run(g_table, t_choice(p=T_MAP, m=T_SCALE), t_cache)


@pytest.fixture(scope="module")
def t_other(g_table, tmp_path_factory):
    """grid_test with candidates (1, 1, 1): not the current settings, other top-ups."""
    choice = t_choice((1, 1, 1), m=T_SCALE)
    return choice, t_run(g_table, choice, tmp_path_factory.mktemp("test_cache_other"))


# --- grid_test: what it returns --------------------------------------------------------------

def test_grid_test_returns_a_grid_test(t_plain):
    found, _ = t_plain
    assert isinstance(found, GridTest)
    assert isinstance(found.forecast, Forecast) and isinstance(found.uncorrected, Forecast)


def test_grid_test_forecasts_are_at_the_test_origins_only(t_plain):
    found, _ = t_plain
    for forecast in (found.forecast, found.uncorrected):
        assert sorted(set(forecast.totals["origin_week"])) == T_ORIGINS
        assert sorted(set(forecast.customers["origin_week"])) == T_ORIGINS


def test_grid_test_uncorrected_is_grid_predictions_and_combination_forecast_by_hand(
        g_table, t_plain, tmp_path):
    found, _ = t_plain
    choice = t_choice()
    chosen = GridCandidates(stage1=[choice.stage1], stage2=[choice.stage2],
                            topups=[choice.topups])
    predictions = grid_predictions(g_table, T_ORIGINS, chosen, t_settings(),
                                   cache_folder=tmp_path, **g_inputs())
    expected = combination_forecast(predictions, 0, 0, 0,
                                    projected=projected_at(T_ORIGINS, 5, WEEK_ZERO),
                                    companies=list(F_COMPANIES), horizons=5)
    pd.testing.assert_frame_equal(found.uncorrected.totals, expected.totals)
    pd.testing.assert_frame_equal(found.uncorrected.customers, expected.customers)


def test_grid_test_uncorrected_with_the_current_settings_equals_rolling_forecasts(t_plain,
                                                                                 g_rolled):
    # Step 17's model at the same origins: the current settings fitted again at each origin.
    found, _ = t_plain
    pd.testing.assert_frame_equal(found.uncorrected.totals, g_rolled.totals)
    pd.testing.assert_frame_equal(found.uncorrected.customers, g_rolled.customers)


def test_grid_test_uncorrected_of_other_candidates_is_the_grids_combination(t_other, g_result):
    # g_result is the U16.3 grid (g_candidates) at the same origins: (1, 1, 1) must match.
    _, found = t_other
    predictions, _ = g_result
    expected = combination_forecast(predictions, 1, 1, 1,
                                    projected=projected_at(T_ORIGINS, 5, WEEK_ZERO),
                                    companies=list(F_COMPANIES), horizons=5)
    pd.testing.assert_frame_equal(found.uncorrected.totals, expected.totals)
    pd.testing.assert_frame_equal(found.uncorrected.customers, expected.customers)


def test_grid_test_other_candidates_give_another_forecast_than_the_current(t_other, t_plain):
    _, other = t_other
    plain, _ = t_plain
    assert not np.allclose(other.uncorrected.totals["total"], plain.uncorrected.totals["total"])


# --- grid_test: the corrections, applied and not learned ---------------------------------------

def test_grid_test_with_no_corrections_the_forecast_equals_the_uncorrected(t_plain):
    found, _ = t_plain
    pd.testing.assert_frame_equal(found.forecast.totals, found.uncorrected.totals)
    pd.testing.assert_frame_equal(found.forecast.customers, found.uncorrected.customers)


def test_grid_test_forecast_is_corrected_forecast_of_the_uncorrected(t_corrected):
    expected = corrected_forecast(t_corrected.uncorrected, T_MAP, T_SCALE)
    pd.testing.assert_frame_equal(t_corrected.forecast.totals, expected.totals)
    pd.testing.assert_frame_equal(t_corrected.forecast.customers, expected.customers)


def test_grid_test_corrections_do_not_change_the_uncorrected_forecast(t_corrected, t_plain):
    plain, _ = t_plain
    pd.testing.assert_frame_equal(t_corrected.uncorrected.totals, plain.uncorrected.totals)
    pd.testing.assert_frame_equal(t_corrected.uncorrected.customers, plain.uncorrected.customers)


def test_grid_test_the_m_scale_is_applied_exactly_as_frozen(g_table, tmp_path):
    found = t_run(g_table, t_choice(m=T_SCALE), tmp_path)
    customers, plain = found.forecast.customers, found.uncorrected.customers
    pd.testing.assert_series_equal(customers["m"], plain["m"] * 1.2345)
    pd.testing.assert_series_equal(customers["p"], plain["p"])
    pd.testing.assert_series_equal(customers["expected"], customers["p"] * customers["m"],
                                   check_names=False)


def test_grid_test_the_corrected_total_differs_from_the_uncorrected(t_corrected):
    assert not np.allclose(t_corrected.forecast.totals["total"],
                           t_corrected.uncorrected.totals["total"])


def test_grid_test_never_learns_a_correction_again(g_table, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a correction must not be learned on the test origins")
    monkeypatch.setattr(grid_module, "choose_calibration", refuse)
    for name in ("choose_calibration", "fit_p_correction", "fit_m_correction", "fit_platt",
                 "calibration_rows"):
        monkeypatch.setattr(calibrate_module, name, refuse)
    found = t_run(g_table, t_choice(p=T_MAP, m=T_SCALE), tmp_path)
    expected = corrected_forecast(found.uncorrected, T_MAP, T_SCALE)
    pd.testing.assert_frame_equal(found.forecast.totals, expected.totals)


def test_grid_test_does_not_read_the_rest_of_the_calibration_choice(g_table, tmp_path):
    # Only the two corrections are used: the rest of the choice may be anything.
    choice = t_choice(m=T_SCALE)
    calibration = SimpleNamespace(p_correction=PCorrection("none"), m_correction=T_SCALE)
    choice = dataclasses.replace(choice, calibration=calibration)
    found = t_run(g_table, choice, tmp_path)
    pd.testing.assert_series_equal(found.forecast.customers["m"],
                                   found.uncorrected.customers["m"] * 1.2345)


def test_grid_test_a_per_h_correction_missing_an_h_raises_as_corrected_forecast_does(
        g_table, tmp_path):
    p = PCorrection(PER_H, {1: (1.0, 0.0), 2: (1.0, 0.0)})
    with pytest.raises(ValueError, match="learned nothing for h"):
        t_run(g_table, t_choice(p=p), tmp_path)


# --- grid_test: the comparison -----------------------------------------------------------------

def test_grid_test_methods_are_hurdle_the_references_the_rule_then_uncorrected(t_plain):
    found, _ = t_plain
    assert list(dict.fromkeys(found.scores["method"])) == [HURDLE, *REFERENCES, RULE,
                                                           UNCORRECTED]
    assert CURRENT_SETTINGS not in set(found.scores["method"])


def test_grid_test_scores_equal_evaluation_scores_of_the_stack_by_hand(t_corrected):
    stacked = t_stacked(t_corrected, t_choice(p=T_MAP, m=T_SCALE), t_settings())
    pd.testing.assert_frame_equal(t_corrected.scores,
                                  evaluation_scores(stacked, t_actual(), 5))


def test_grid_test_hurdle_scores_are_the_corrected_forecasts(t_corrected):
    rows = t_corrected.forecast.totals[CH_KEYS + ["total"]].rename(columns={"total": "forecast"})
    rows = rows.assign(method=HURDLE)[STACK_COLUMNS]
    expected = evaluation_scores(rows, t_actual(), 5).reset_index(drop=True)
    found = t_corrected.scores[t_corrected.scores["method"] == HURDLE].reset_index(drop=True)
    pd.testing.assert_frame_equal(found, expected)


def test_grid_test_uncorrected_scores_are_the_uncorrected_forecasts(t_corrected):
    expected = evaluation_scores(method_rows(t_corrected.uncorrected, UNCORRECTED), t_actual(),
                                 5).reset_index(drop=True)
    found = t_corrected.scores[t_corrected.scores["method"] == UNCORRECTED]
    pd.testing.assert_frame_equal(found.reset_index(drop=True), expected)


def test_grid_test_with_no_corrections_hurdle_and_uncorrected_score_the_same(t_plain):
    found, _ = t_plain
    scores = found.scores
    hurdle = scores[scores["method"] == HURDLE].drop(columns="method").reset_index(drop=True)
    plain = scores[scores["method"] == UNCORRECTED].drop(columns="method").reset_index(drop=True)
    pd.testing.assert_frame_equal(hurdle, plain)


def test_grid_test_with_corrections_hurdle_and_uncorrected_score_differently(t_corrected):
    scores = t_corrected.scores
    assert pooled(scores[scores["method"] == HURDLE], CONSOLIDATED) != pooled(
        scores[scores["method"] == UNCORRECTED], CONSOLIDATED)


def test_grid_test_bootstrap_is_hurdle_against_every_other_method(t_corrected):
    stacked = t_stacked(t_corrected, t_choice(p=T_MAP, m=T_SCALE), t_settings())
    others = [*REFERENCES, RULE, UNCORRECTED]
    expected = paired_wape_bootstrap(stacked, t_actual(), [(HURDLE, o) for o in others], 5,
                                     40, 5)
    pd.testing.assert_frame_equal(t_corrected.bootstrap, expected)
    assert set(t_corrected.bootstrap["method"]) == {HURDLE}
    assert list(dict.fromkeys(t_corrected.bootstrap["against"])) == others


def test_grid_test_bootstrap_with_no_corrections_still_pairs_with_uncorrected(t_plain):
    found, _ = t_plain
    assert list(dict.fromkeys(found.bootstrap["against"])) == [*REFERENCES, RULE, UNCORRECTED]


def test_grid_test_bootstrap_reads_the_evaluation_samples_and_seed(g_table, t_cache, t_plain):
    first = t_run(g_table, t_choice(m=T_SCALE), t_cache, settings=t_settings(samples=30, seed=1))
    assert set(first.bootstrap["samples"]) == {30}
    second = t_run(g_table, t_choice(m=T_SCALE), t_cache,
                   settings=t_settings(samples=30, seed=2))
    assert not first.bootstrap[["low", "high"]].equals(second.bootstrap[["low", "high"]])
    pd.testing.assert_frame_equal(first.scores, second.scores)


def test_grid_test_references_equal_the_reference_functions_by_hand(t_corrected):
    side = t_side()
    reference = deflated_reference(side["weekly"], side["weeks"], side["levels"],
                                   side["calendar"], list(F_COMPANIES), T_ORIGINS, 5, WEEK_ZERO)
    stacked = stack_forecasts(None, reference, SimpleNamespace(
        totals=pd.DataFrame(columns=CH_KEYS + ["forecast"])))
    expected = evaluation_scores(stacked[stacked["method"].isin(list(REFERENCES))], t_actual(),
                                 5).reset_index(drop=True)
    found = t_corrected.scores[t_corrected.scores["method"].isin(list(REFERENCES))]
    pd.testing.assert_frame_equal(found.reset_index(drop=True), expected)


def test_grid_test_rule_uses_the_chosen_topups_not_the_settings_topups(t_other):
    choice, found = t_other
    settings = t_settings()
    assert choice.topups != settings.topups
    side = t_side()

    def rule_scores(topups):
        rule = rule_reference(side["ledger"], side["rates"], T_ORIGINS,
                              dataclasses.replace(settings, topups=topups),
                              companies=list(F_COMPANIES), new_cash=side["new_cash"],
                              refunds=side["refunds"], unidentified=side["unidentified"],
                              levels=side["levels"], calendar=side["calendar"],
                              week_zero=WEEK_ZERO)
        stacked = rule.totals[CH_KEYS + ["forecast"]].assign(method=RULE)[STACK_COLUMNS]
        return evaluation_scores(stacked, t_actual(), 5).reset_index(drop=True)

    rule = found.scores[found.scores["method"] == RULE].reset_index(drop=True)
    pd.testing.assert_frame_equal(rule, rule_scores(choice.topups))
    assert pooled(rule, CONSOLIDATED) != pooled(rule_scores(settings.topups), CONSOLIDATED)


class GridSpy:
    """Wraps the references (and grid_predictions) as grid sees them: records, then calls
    the real function."""

    def __init__(self, monkeypatch):
        self.calls = {}
        for name in ("deflated_reference", "rule_reference", "grid_predictions"):
            real = getattr(grid_module, name)
            monkeypatch.setattr(grid_module, name, self.wrap(name, real))

    def wrap(self, name, real):
        def step(*args, **kwargs):
            self.calls.setdefault(name, []).append((args, kwargs))
            return real(*args, **kwargs)
        return step


def test_grid_test_rule_gets_the_settings_with_only_the_topups_replaced(g_table, tmp_path,
                                                                       monkeypatch):
    spy = GridSpy(monkeypatch)
    settings = t_settings()
    choice = t_choice((0, 0, 1), settings=settings)
    t_run(g_table, choice, tmp_path, settings=settings)
    (args, kwargs), = spy.calls["rule_reference"]
    given = args[3]
    assert given.topups is choice.topups
    assert dataclasses.replace(given, topups=settings.topups) == settings
    assert settings.topups != choice.topups          # the settings themselves are untouched
    assert settings == t_settings()


def test_grid_test_the_references_get_their_inputs_at_the_test_origins(g_table, tmp_path,
                                                                       monkeypatch):
    spy = GridSpy(monkeypatch)
    side = t_side()
    t_run(g_table, t_choice(), tmp_path, settings=t_settings(test_origins=(22, 18, 22)))
    (args, kwargs), = spy.calls["deflated_reference"]
    assert kwargs == {}
    assert args[5] == T_ORIGINS and args[6] == 5 and args[7] == WEEK_ZERO
    assert list(args[4]) == list(F_COMPANIES)
    pd.testing.assert_frame_equal(args[0], side["weekly"])
    (args, kwargs), = spy.calls["rule_reference"]
    assert args[2] == T_ORIGINS
    assert sorted(kwargs) == ["calendar", "companies", "levels", "new_cash", "refunds",
                              "unidentified", "week_zero"]
    assert kwargs["week_zero"] == WEEK_ZERO


def test_grid_test_fits_at_the_test_origins_sorted_once_each_with_one_candidate_each(
        g_table, tmp_path, monkeypatch):
    spy = GridSpy(monkeypatch)
    choice = t_choice((1, 0, 1))
    cache = tmp_path / "grid cache"
    t_run(g_table, choice, cache, settings=t_settings(test_origins=(22, 18, 22)))
    (args, kwargs), = spy.calls["grid_predictions"]
    assert args[1] == T_ORIGINS
    candidates = args[2]
    assert isinstance(candidates, GridCandidates)
    assert candidates.stage1 == [choice.stage1]
    assert candidates.stage2 == [choice.stage2]
    assert candidates.topups == [choice.topups]
    assert kwargs["cache_folder"] == cache


def test_grid_test_unsorted_test_origins_give_the_same_result(g_table, t_cache, t_corrected):
    found = t_run(g_table, t_choice(p=T_MAP, m=T_SCALE), t_cache,
                  settings=t_settings(test_origins=(22, 18, 22)))
    pd.testing.assert_frame_equal(found.forecast.totals, t_corrected.forecast.totals)
    pd.testing.assert_frame_equal(found.scores, t_corrected.scores)
    pd.testing.assert_frame_equal(found.bootstrap, t_corrected.bootstrap)


# --- grid_test: only the chosen candidates are fitted, through the cache --------------------

def test_grid_test_fits_each_stage_once_per_test_origin(t_plain):
    _, progress = t_plain
    assert len(progress) == 2 * len(T_ORIGINS)
    assert [(done, total) for done, total, *_ in progress] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert [(stage, origin, number) for _, _, stage, origin, number, _ in progress] == [
        ("stage 1", 18, 0), ("stage 2", 18, 0), ("stage 1", 22, 0), ("stage 2", 22, 0)]


def test_grid_test_fresh_cache_fits_only_the_chosen_settings(g_table, tmp_path, monkeypatch):
    calls = count_fits(monkeypatch)
    t_run(g_table, t_choice((1, 1, 0)), tmp_path)
    assert calls == [("stage1", 18), ("stage2", 18), ("stage1", 22), ("stage2", 22)]


def test_grid_test_writes_its_fits_into_the_cache_folder_named_by_the_chosen_settings(
        g_table, tmp_path):
    choice = t_choice((1, 1, 0))
    t_run(g_table, choice, tmp_path)
    names = cache_files(tmp_path)
    for stage, setting in (("stage1", choice.stage1.settings), ("stage2", choice.stage2.settings)):
        for origin in T_ORIGINS:
            prefix = f"{stage}_origin{origin}_{setting_hash(setting)}"
            assert any(name.startswith(prefix) for name in names), (prefix, names)
    for origin in (9, 13):
        assert not any(f"origin{origin}_" in name for name in names)


def test_grid_test_a_second_call_reads_every_fit_from_the_cache(g_table, t_cache, t_plain,
                                                                monkeypatch):
    found, _ = t_plain
    count_fits(monkeypatch, fail=True)       # any fit would raise
    progress = []
    again = t_run(g_table, t_choice(), t_cache, on_fit_done=lambda *args: progress.append(args))
    assert len(progress) == 4
    pd.testing.assert_frame_equal(again.uncorrected.totals, found.uncorrected.totals)
    pd.testing.assert_frame_equal(again.scores, found.scores)


def test_grid_test_reads_the_fits_the_grid_made_at_the_same_origins(g_table, g_cache, g_result,
                                                                    monkeypatch, tmp_path):
    # The U16.3 grid fitted g_candidates at 18 and 22 in g_cache: grid_test with those
    # settings refits nothing (on a private copy of that cache).
    cache = copied_cache(g_cache, tmp_path)
    count_fits(monkeypatch, fail=True)
    t_run(g_table, t_choice((1, 1, 1)), cache)


def test_grid_test_without_on_fit_done_runs(t_corrected):
    # t_corrected passes on_fit_done=None.
    assert isinstance(t_corrected, GridTest)


# --- grid_test: the checks --------------------------------------------------------------------

@pytest.mark.parametrize("validation", [(9, 14), (14, 9), (17,), (18,), (22,), (30, 3)])
def test_grid_test_validation_reaching_the_test_weeks_raises_before_any_fit(
        g_table, tmp_path, monkeypatch, validation):
    count_fits(monkeypatch, fail=True)
    last = max(validation)
    with pytest.raises(ValueError) as error:
        t_run(g_table, t_choice(validation=validation), tmp_path)
    message = str(error.value)
    assert f"the last validation origin {last} forecasts up to week {last + 5}" in message
    assert "first test target week 19" in message
    assert cache_files(tmp_path) == []


@pytest.mark.parametrize("validation", [(13,), (13, 9), (2, 13, 7), (0,)])
def test_grid_test_validation_ending_just_before_the_test_weeks_is_accepted(
        g_table, t_cache, t_plain, validation):
    # The last validation origin 13 forecasts up to week 18; the first test target is 19.
    found = t_run(g_table, t_choice(validation=validation), t_cache)
    pd.testing.assert_frame_equal(found.forecast.totals, t_plain[0].forecast.totals)


def test_grid_test_the_overlap_check_reads_the_horizons(g_table, t_cache, t_plain):
    # With 5 weeks, 14 overlaps (week 19); the check is origin + horizons, not a fixed gap.
    with pytest.raises(ValueError, match="forecasts up to week 19"):
        t_run(g_table, t_choice(validation=(14,)), t_cache)


@pytest.mark.parametrize("origins", [[18], [22], [18, 22, 26], [17, 22], [14, 18]])
def test_grid_test_actual_for_other_origins_raises_before_any_fit(g_table, tmp_path,
                                                                  monkeypatch, origins):
    count_fits(monkeypatch, fail=True)
    with pytest.raises(ValueError) as error:
        t_run(g_table, t_choice(), tmp_path, actual=t_actual(origins))
    assert str(error.value) == (f"the actual cash is for origins {sorted(origins)}, the test "
                                f"origins are {T_ORIGINS}")
    assert cache_files(tmp_path) == []


def test_grid_test_actual_with_rows_in_any_order_is_accepted(g_table, t_cache, t_corrected):
    actual = t_actual().iloc[::-1]
    found = t_run(g_table, t_choice(p=T_MAP, m=T_SCALE), t_cache, actual=actual)
    pd.testing.assert_frame_equal(found.forecast.totals, t_corrected.forecast.totals)


def test_grid_test_the_overlap_check_comes_before_the_actual_check(g_table, tmp_path):
    with pytest.raises(ValueError, match="last validation origin"):
        t_run(g_table, t_choice(validation=(20,)), tmp_path, actual=t_actual([18]))


@pytest.mark.parametrize("step", ["deflated_reference", "rule_reference", "grid_predictions",
                                  "corrected_forecast", "evaluation_scores",
                                  "paired_wape_bootstrap", "combination_forecast"])
def test_grid_test_a_failing_step_passes_its_error_through(g_table, tmp_path, monkeypatch,
                                                           step):
    def fail(*args, **kwargs):
        raise ValueError(f"{step} failed")
    monkeypatch.setattr(grid_module, step, fail)
    with pytest.raises(ValueError, match=f"{step} failed"):
        t_run(g_table, t_choice(), tmp_path)


def test_grid_test_a_test_origin_outside_the_table_raises_as_grid_predictions_does(
        g_table, tmp_path):
    settings = t_settings(test_origins=(18, 40))
    with pytest.raises(ValueError):
        t_run(g_table, t_choice(), tmp_path, settings=settings, actual=t_actual([18, 40]))


# --- grid_test: purity -------------------------------------------------------------------------

def test_grid_test_does_not_change_its_inputs(g_table, t_cache, t_plain):
    table, actual, settings = g_table.copy(), t_actual(), t_settings()
    choice = t_choice(p=T_MAP, m=T_SCALE)
    before_actual = actual.copy()
    before_customers = choice.forecast.customers.copy()
    t_run(g_table, choice, t_cache, actual=actual, settings=settings)
    pd.testing.assert_frame_equal(g_table, table)
    pd.testing.assert_frame_equal(actual, before_actual)
    pd.testing.assert_frame_equal(choice.forecast.customers, before_customers)
    assert settings == t_settings()
    assert choice.calibration.m_correction == T_SCALE and choice.calibration.p_correction == T_MAP


def test_grid_test_is_deterministic(g_table, t_cache, t_corrected):
    again = t_run(g_table, t_choice(p=T_MAP, m=T_SCALE), t_cache)
    pd.testing.assert_frame_equal(again.forecast.totals, t_corrected.forecast.totals)
    pd.testing.assert_frame_equal(again.scores, t_corrected.scores)
    pd.testing.assert_frame_equal(again.bootstrap, t_corrected.bootstrap)
