"""Tests for `hurdle/grid.py` (U16.2): `GridCandidates` and `grid_candidates`.

`grid_candidates` reads only `settings.stage1`, `.stage2`, `.topups` and `.grid`, so most
tests pass a small stand-in (a SimpleNamespace holding real settings dataclasses). One test
at the end loads a real `Settings` through `load_settings`. Nothing is fitted.
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
