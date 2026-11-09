"""Tests for `hurdle/grid.py` (U16.2): `GridCandidates` and `grid_candidates`.

`grid_candidates` reads only `settings.stage1`, `.stage2`, `.topups` and `.grid`, so most
tests pass a small stand-in (a SimpleNamespace holding real settings dataclasses). One test
at the end loads a real `Settings` through `load_settings`. Nothing is fitted.

U16.3, at the end: `ROW_COLUMNS`, `GridPredictions`, `setting_hash`, `cached_fit` and
`grid_predictions`, fitted on test_forecast.py's small training table with a tmp_path cache.
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
