"""Tests for U18.5 in `hurdle/grid.py`: the feature-block subsets of the settings grid.

`block_subsets`, `subset_name`, `with_extra_blocks`, `grid_candidates` with [grid]
extra_blocks, `same_extra_blocks`, `combination_count`, `ALWAYS_HASHED_BLOCKS`,
`hashed_table`, `grid_scores` (only same-subset pairs, the "extra_blocks" column) and the
cache of `grid_predictions` (each fit hashed on the table without the extra blocks it does not
use).

The candidate tests use test_grid.py's stand-in settings (`make`, extra_blocks () there). The
cache tests fit test_forecast.py's small table (ten trees, well under a second a fit) with two
extra columns, one "dues" and one "habits", into a tmp_path cache; nothing reads or writes
data/.
"""
import dataclasses
import itertools
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import grid as grid_module
from collection_estimation.hurdle.diagnostic import Candidate
from collection_estimation.hurdle.diagnostic_joint import stage1_grid, stage2_grid
from collection_estimation.hurdle.forecast import stage1_predictions, stage2_predictions
from collection_estimation.hurdle.grid import (
    ALWAYS_HASHED_BLOCKS,
    GRID_SCORE_COLUMNS,
    GridCandidates,
    block_subsets,
    choose_combination,
    combination_count,
    grid_candidates,
    grid_predictions,
    hashed_table,
    same_extra_blocks,
    setting_hash,
    subset_name,
    with_extra_blocks,
)
from collection_estimation.hurdle.settings import FEATURE_BLOCKS, Stage1Settings, Stage2Settings
from tests.hurdle.test_forecast import BLOCK_COLUMNS, F_COMPANIES, make_table
from tests.hurdle.test_grid import (
    GRID,
    STAGE1,
    STAGE2,
    TOPUPS,
    g_inputs,
    g_settings,
    make,
    sc_actual,
    sc_gates,
    sc_predictions,
    sc_run,
)

BOTH = ("dues", "habits")


def make_both(**changes):
    """test_grid's stand-in settings with [grid] extra_blocks = ("dues", "habits")."""
    return make(extra_blocks=BOTH, **changes)


# --- block_subsets ---------------------------------------------------------------------------

def test_block_subsets_of_dues_and_habits_smallest_first():
    assert block_subsets(BOTH) == [(), ("dues",), ("habits",), ("dues", "habits")]


def test_block_subsets_of_nothing_is_the_empty_subset_alone():
    assert block_subsets(()) == [()]


def test_block_subsets_of_one_block():
    assert block_subsets(("habits",)) == [(), ("habits",)]


def test_block_subsets_within_a_size_follow_the_order_given():
    assert block_subsets(("habits", "dues")) == [(), ("habits",), ("dues",), ("habits", "dues")]


def test_block_subsets_of_three_blocks_are_all_eight_by_size():
    found = block_subsets(("receivables", "dues", "habits"))
    assert found == [(), ("receivables",), ("dues",), ("habits",), ("receivables", "dues"),
                     ("receivables", "habits"), ("dues", "habits"),
                     ("receivables", "dues", "habits")]
    assert [len(s) for s in found] == sorted(len(s) for s in found)
    assert len(set(found)) == 8


def test_block_subsets_returns_a_list_of_tuples():
    found = block_subsets(BOTH)
    assert type(found) is list and all(type(s) is tuple for s in found)


# --- subset_name -----------------------------------------------------------------------------

@pytest.mark.parametrize("subset, name", [((), "none"), (("dues",), "dues"),
                                          (("habits",), "habits"),
                                          (("dues", "habits"), "dues + habits"),
                                          (("a", "b", "c"), "a + b + c")])
def test_subset_name(subset, name):
    assert subset_name(subset) == name


def test_subset_names_of_the_subsets_differ():
    names = [subset_name(s) for s in block_subsets(BOTH)]
    assert names == ["none", "dues", "habits", "dues + habits"]
    assert len(set(names)) == 4


# --- with_extra_blocks -----------------------------------------------------------------------

@pytest.mark.parametrize("subset, expected", [
    ((), ("history", "calendar", "receivables")),
    (("dues",), ("history", "calendar", "receivables", "dues")),
    (("habits",), ("history", "calendar", "receivables", "habits")),
    (("dues", "habits"), ("history", "calendar", "receivables", "dues", "habits")),
    (("habits", "dues"), ("history", "calendar", "receivables", "dues", "habits")),
])
def test_with_extra_blocks_adds_the_subset_in_feature_blocks_order(subset, expected):
    found = with_extra_blocks(STAGE1, subset)
    assert found.feature_blocks == expected
    assert type(found.feature_blocks) is tuple


def test_with_extra_blocks_changes_nothing_else_and_keeps_the_type():
    for stage in (STAGE1, STAGE2):
        found = with_extra_blocks(stage, BOTH)
        assert type(found) is type(stage)
        assert dataclasses.replace(found, feature_blocks=stage.feature_blocks) == stage


def test_with_extra_blocks_of_the_empty_subset_equals_the_settings():
    # Both stand-ins hold their blocks in FEATURE_BLOCKS order, as load_settings gives them.
    assert with_extra_blocks(STAGE1, ()) == STAGE1
    assert with_extra_blocks(STAGE2, ()) == STAGE2


def test_with_extra_blocks_puts_a_block_between_the_stages_own():
    stage = dataclasses.replace(STAGE2, feature_blocks=("history", "dues"))
    assert with_extra_blocks(stage, ("calendar", "habits")).feature_blocks == (
        "history", "calendar", "dues", "habits")


def test_with_extra_blocks_a_block_the_stage_has_is_not_repeated():
    found = with_extra_blocks(STAGE1, ("history", "dues"))
    assert found.feature_blocks == ("history", "calendar", "receivables", "dues")


def test_with_extra_blocks_does_not_change_its_input():
    before = dataclasses.replace(STAGE1)
    with_extra_blocks(STAGE1, BOTH)
    assert STAGE1 == before


def test_with_extra_blocks_reorders_blocks_given_out_of_order():
    # Documented as found: a stage whose own blocks are not in FEATURE_BLOCKS order (never
    # the case after load_settings) comes back reordered, even with no extra block.
    stage = dataclasses.replace(STAGE2, feature_blocks=("calendar", "history"))
    assert with_extra_blocks(stage, ()).feature_blocks == ("history", "calendar")


# --- grid_candidates with extra_blocks () : as before U18.5 -------------------------------------

def pre_u18_5(settings):
    """grid_candidates' two stage lists as written before U18.5 (copied from its diff)."""
    stage1, stage2, grid = settings.stage1, settings.stage2, settings.grid
    current1 = Candidate(params={"max_iter": stage1.max_iter,
                                 "learning_rate": stage1.learning_rate,
                                 "max_leaf_nodes": stage1.max_leaf_nodes},
                         settings=stage1, class_weight="none")
    floor = stage2.balance_floor if stage2.scale == "open balance" else "-"
    current2 = Candidate(params={"loss": stage2.loss, "max_iter": stage2.max_iter,
                                 "learning_rate": stage2.learning_rate,
                                 "max_leaf_nodes": stage2.max_leaf_nodes, "scale": stage2.scale,
                                 "balance_floor": floor},
                         settings=stage2, class_weight="none")
    first = [current1] + [c for c in stage1_grid(stage1, grid) if c.settings != stage1]
    second = [current2] + [c for c in stage2_grid(stage2, grid) if c.settings != stage2]
    return first, second


PRE_SHAPES = [
    dict(),
    dict(stage1_max_iter=(100, 600)),
    dict(stage1_max_iter=(900, 300, 100), stage1_learning_rate=(0.2, 0.05),
         stage1_max_leaf_nodes=(63, 31, 15)),
    dict(stage2_scale=("none", "open balance"), stage2_balance_floor=("median", "own mean")),
    dict(stage2=dataclasses.replace(STAGE2, scale="none", balance_floor="median"),
         stage2_scale=("none", "open balance"), stage2_balance_floor=("median", "own mean")),
    dict(stage1=dataclasses.replace(STAGE1, learning_rate=0.5), topup_windows=(4, 8),
         topup_ridge_with_level=(True, False)),
]


@pytest.mark.parametrize("changes", PRE_SHAPES)
def test_no_extra_block_gives_the_pre_u18_5_candidates_plus_the_none_name(changes):
    settings = make(**changes)
    assert settings.grid.extra_blocks == ()
    found = grid_candidates(settings)
    old1, old2 = pre_u18_5(settings)
    for listed, old in ((found.stage1, old1), (found.stage2, old2)):
        assert [c.settings for c in listed] == [c.settings for c in old]
        assert [c.class_weight for c in listed] == [c.class_weight for c in old]
        assert [c.params for c in listed] == [{**c.params, "extra_blocks": "none"} for c in old]
        assert all(list(c.params)[-1] == "extra_blocks" for c in listed)


@pytest.mark.parametrize("changes", PRE_SHAPES)
def test_no_extra_block_the_combinations_are_the_old_product(changes):
    found = grid_candidates(make(**changes))
    assert combination_count(found) == len(found.stage1) * len(found.stage2) * len(found.topups)


# --- grid_candidates with extra_blocks ("dues", "habits") ---------------------------------------

def by_subset(listed):
    """{subset name: the candidates with it, in order}, and the names in order of appearance."""
    names = list(dict.fromkeys(c.params["extra_blocks"] for c in listed))
    return {name: [c for c in listed if c.params["extra_blocks"] == name] for name in names}


def test_candidate_0_of_each_stage_is_still_the_current_settings_exactly():
    found = grid_candidates(make_both())
    assert found.stage1[0].settings == STAGE1
    assert found.stage2[0].settings == STAGE2
    assert found.stage1[0].params["extra_blocks"] == "none"
    assert found.stage2[0].params["extra_blocks"] == "none"


def test_the_subsets_come_in_block_subsets_order_each_in_one_run():
    found = grid_candidates(make_both())
    for listed in (found.stage1, found.stage2):
        names = [c.params["extra_blocks"] for c in listed]
        # One contiguous run per subset, in the order of block_subsets.
        runs = [name for name, _ in itertools.groupby(names)]
        assert runs == ["none", "dues", "habits", "dues + habits"]


def test_each_subset_is_the_whole_no_extra_list_with_its_blocks_added():
    plain = grid_candidates(make())
    found = grid_candidates(make_both())
    for listed, base in ((found.stage1, plain.stage1), (found.stage2, plain.stage2)):
        groups = by_subset(listed)
        for subset in block_subsets(BOTH):
            name = subset_name(subset)
            assert [c.settings for c in groups[name]] == [
                with_extra_blocks(c.settings, subset) for c in base], name
            assert [c.params for c in groups[name]] == [
                {**c.params, "extra_blocks": name} for c in base], name


def test_each_subset_starts_with_the_current_settings_with_its_blocks():
    found = grid_candidates(make_both())
    for listed, stage in ((found.stage1, STAGE1), (found.stage2, STAGE2)):
        for name, group in by_subset(listed).items():
            subset = () if name == "none" else tuple(name.split(" + "))
            assert group[0].settings == with_extra_blocks(stage, subset)
            assert group[0].settings.max_iter == stage.max_iter
            assert group[0].settings.learning_rate == stage.learning_rate


def test_the_rest_of_a_subset_is_the_stage_grid_of_its_current_without_it():
    settings = make_both(stage1_max_iter=(900, 300, 100), stage2_loss=("gamma", "poisson"))
    found = grid_candidates(settings)
    for listed, stage, grid_of in ((found.stage1, STAGE1, stage1_grid),
                                   (found.stage2, STAGE2, stage2_grid)):
        for subset in block_subsets(BOTH):
            name = subset_name(subset)
            current = with_extra_blocks(stage, subset)
            expected = [c for c in grid_of(current, settings.grid) if c.settings != current]
            group = by_subset(listed)[name]
            assert [c.settings for c in group[1:]] == [c.settings for c in expected]
            assert [c.params for c in group[1:]] == [
                {**c.params, "extra_blocks": name} for c in expected]


def test_the_counts_grow_by_the_number_of_subsets():
    plain = grid_candidates(make())
    found = grid_candidates(make_both())
    assert len(found.stage1) == 4 * len(plain.stage1) == 16
    assert len(found.stage2) == 4 * len(plain.stage2) == 16


def test_a_grid_point_equal_to_the_current_with_a_subset_is_listed_once():
    # GRID holds the current stage-1 point (300, 0.05, 31): each subset lists it only first.
    found = grid_candidates(make_both())
    for subset in block_subsets(BOTH):
        current = with_extra_blocks(STAGE1, subset)
        assert sum(c.settings == current for c in found.stage1) == 1
        current = with_extra_blocks(STAGE2, subset)
        assert sum(c.settings == current for c in found.stage2) == 1


def test_no_two_stage_candidates_have_equal_settings_across_subsets():
    found = grid_candidates(make_both(stage2_scale=("none", "open balance"),
                                      stage2_balance_floor=("median", "own mean")))
    for listed in (found.stage1, found.stage2):
        assert len({c.settings for c in listed}) == len(listed)


def test_every_candidates_blocks_are_its_stage_blocks_plus_its_subset():
    found = grid_candidates(make_both())
    for listed, stage in ((found.stage1, STAGE1), (found.stage2, STAGE2)):
        for c in listed:
            name = c.params["extra_blocks"]
            subset = () if name == "none" else tuple(name.split(" + "))
            assert set(c.settings.feature_blocks) == set(stage.feature_blocks) | set(subset)
            assert c.settings.feature_blocks == tuple(
                b for b in FEATURE_BLOCKS if b in c.settings.feature_blocks)


def test_the_stage_candidates_keep_their_types_and_no_class_weight():
    found = grid_candidates(make_both())
    assert all(type(c.settings) is Stage1Settings for c in found.stage1)
    assert all(type(c.settings) is Stage2Settings for c in found.stage2)
    assert all(c.class_weight == "none" for c in found.stage1 + found.stage2)


def test_the_topups_are_listed_once_and_unchanged():
    for changes in ({}, dict(topup_windows=(4, 8), topup_ridge_with_level=(True, False))):
        assert grid_candidates(make_both(**changes)).topups == grid_candidates(
            make(**changes)).topups
    assert grid_candidates(make_both()).topups[0] is TOPUPS


def test_one_extra_block_gives_two_subsets():
    found = grid_candidates(make(extra_blocks=("habits",)))
    assert list(by_subset(found.stage1)) == ["none", "habits"]
    assert len(found.stage1) == 8 and len(found.stage2) == 8


def test_grid_candidates_with_extra_blocks_is_deterministic_and_leaves_its_input():
    settings = make_both()
    before = (settings.stage1, settings.stage2, settings.topups, settings.grid)
    assert grid_candidates(settings) == grid_candidates(settings)
    assert (settings.stage1, settings.stage2, settings.topups, settings.grid) == before


def test_a_loaded_settings_with_extra_blocks(tmp_path):
    from collection_estimation.hurdle.settings import load_settings
    from tests.hurdle.test_settings import write_settings
    repo = tmp_path / "repo"
    repo.mkdir()
    path = write_settings(tmp_path, changes={("grid", "extra_blocks"): '["habits", "dues"]'})
    settings = load_settings(path, repo)
    found = grid_candidates(settings)
    assert found.stage1[0].settings == settings.stage1
    assert found.stage2[0].settings == settings.stage2
    assert list(by_subset(found.stage1)) == ["none", "dues", "habits", "dues + habits"]
    assert list(by_subset(found.stage2)) == ["none", "dues", "habits", "dues + habits"]


# --- same_extra_blocks and combination_count -------------------------------------------------

def stand_in(name):
    return SimpleNamespace(params={"extra_blocks": name})


@pytest.mark.parametrize("a, b, same", [("none", "none", True), ("dues", "dues", True),
                                        ("none", "dues", False), ("dues", "habits", False),
                                        ("dues + habits", "dues + habits", True),
                                        ("dues + habits", "habits", False)])
def test_same_extra_blocks(a, b, same):
    assert same_extra_blocks(stand_in(a), stand_in(b)) is same


def test_same_extra_blocks_reads_only_the_params_name():
    first = Candidate(params={"max_iter": 1, "extra_blocks": "dues"}, settings="x",
                      class_weight="none")
    second = Candidate(params={"loss": "gamma", "extra_blocks": "dues"}, settings="y",
                       class_weight="balanced")
    assert same_extra_blocks(first, second) is True


def test_same_extra_blocks_without_the_name_raises_key_error():
    with pytest.raises(KeyError):
        same_extra_blocks(SimpleNamespace(params={}), stand_in("none"))


def test_combination_count_of_the_worked_example():
    # 4 subsets x (4 stage-1 x 4 stage-2 of that subset) x 2 top-ups.
    assert combination_count(grid_candidates(make_both())) == 4 * 4 * 4 * 2


@pytest.mark.parametrize("names1, names2, n3, expected", [
    (["none"], ["none"], 1, 1),
    (["none", "dues"], ["none", "dues"], 3, 6),
    (["none", "none", "dues"], ["none", "dues", "dues"], 2, (2 * 1 + 1 * 2) * 2),
    (["none", "dues"], ["habits"], 4, 0),
    (["none"], ["none"], 0, 0),
])
def test_combination_count_counts_only_same_subset_pairs(names1, names2, n3, expected):
    candidates = SimpleNamespace(stage1=[stand_in(n) for n in names1],
                                 stage2=[stand_in(n) for n in names2], topups=["t"] * n3)
    assert combination_count(candidates) == expected


def test_combination_count_is_far_below_the_full_product_with_subsets():
    found = grid_candidates(make_both())
    full = len(found.stage1) * len(found.stage2) * len(found.topups)
    assert combination_count(found) * 4 == full


# --- grid_scores with subsets ----------------------------------------------------------------

def labelled(names1, names2, n3=2):
    """sc_candidates-like candidates whose stage candidates carry the given subset names."""
    return GridCandidates(
        stage1=[Candidate(params={"extra_blocks": n}, settings=f"<s1 {i}>", class_weight="none")
                for i, n in enumerate(names1)],
        stage2=[Candidate(params={"extra_blocks": n}, settings=f"<s2 {j}>", class_weight="none")
                for j, n in enumerate(names2)],
        topups=[f"<t {k}>" for k in range(n3)])


NAMES1 = ["none", "none", "dues", "habits"]
NAMES2 = ["none", "dues", "dues", "habits"]


@pytest.fixture(scope="module")
def subset_case():
    predictions = sc_predictions(n1=4, n2=4, n3=2)
    actual = sc_actual()
    gates = sc_gates((True,) * 4, (True,) * 4)
    table = sc_run(predictions=predictions, candidates=labelled(NAMES1, NAMES2), gates=gates,
                   actual=actual)
    every = sc_run(predictions=predictions, candidates=labelled(["none"] * 4, ["none"] * 4),
                   gates=gates, actual=actual)
    return predictions, actual, table, every


def test_grid_scores_columns_include_extra_blocks_after_topups(subset_case):
    _, _, table, _ = subset_case
    assert list(table.columns) == GRID_SCORE_COLUMNS
    assert GRID_SCORE_COLUMNS.index("extra_blocks") == GRID_SCORE_COLUMNS.index("topups") + 1


def test_grid_scores_scores_only_same_subset_pairs(subset_case):
    _, _, table, _ = subset_case
    expected = {(i, j, k) for i, j, k in itertools.product(range(4), range(4), range(2))
                if NAMES1[i] == NAMES2[j]}
    found = set(zip(table["stage1"], table["stage2"], table["topups"]))
    assert found == expected
    assert len(table) == len(expected) == (2 * 1 + 1 * 2 + 1 * 1) * 2


def test_grid_scores_never_scores_a_cross_subset_pair(subset_case):
    _, _, table, _ = subset_case
    for i, j in zip(table["stage1"], table["stage2"]):
        assert NAMES1[i] == NAMES2[j], (i, j)


def test_grid_scores_extra_blocks_column_is_the_pairs_subset_name(subset_case):
    _, _, table, _ = subset_case
    for _, line in table.iterrows():
        assert line["extra_blocks"] == NAMES1[line["stage1"]] == NAMES2[line["stage2"]]
    assert set(table["extra_blocks"]) == {"none", "dues", "habits"}


def test_grid_scores_a_pair_scores_as_it_does_without_subsets(subset_case):
    # Labelling the candidates only drops pairs: the scores of a kept pair do not move.
    _, _, table, every = subset_case
    keys = ["stage1", "stage2", "topups"]
    merged = table.merge(every, on=keys, suffixes=("", "_all"))
    assert len(merged) == len(table)
    for column in ("wape", "total_wape", "bias", "company_wape"):
        np.testing.assert_array_equal(merged[column].to_numpy(),
                                      merged[f"{column}_all"].to_numpy())


def test_grid_scores_the_current_is_0_0_0_only(subset_case):
    _, _, table, _ = subset_case
    current = table[table["current"]]
    assert len(current) == 1
    assert (current.iloc[0]["stage1"], current.iloc[0]["stage2"],
            current.iloc[0]["topups"]) == (0, 0, 0)
    assert current.iloc[0]["extra_blocks"] == "none"


def test_grid_scores_all_none_is_the_full_product(subset_case):
    _, _, _, every = subset_case
    assert len(every) == 4 * 4 * 2
    assert set(every["extra_blocks"]) == {"none"}


def test_grid_scores_sorted_by_wape_with_subsets(subset_case):
    _, _, table, _ = subset_case
    assert list(table["wape"]) == sorted(table["wape"])
    assert list(table.index) == list(range(len(table)))


def test_grid_scores_a_subset_with_no_partner_in_the_other_stage_is_not_scored():
    predictions = sc_predictions(n1=3, n2=2, n3=1)
    table = sc_run(predictions=predictions,
                   candidates=labelled(["none", "habits", "dues"], ["none", "dues"], n3=1),
                   gates=sc_gates((True,) * 3, (True,) * 2))
    assert 1 not in set(table["stage1"])     # habits has no stage-2 partner
    assert set(zip(table["stage1"], table["stage2"])) == {(0, 0), (2, 1)}


def test_grid_scores_an_ineligible_candidate_drops_its_pairs_only():
    predictions = sc_predictions(n1=4, n2=4, n3=1)
    table = sc_run(predictions=predictions, candidates=labelled(NAMES1, NAMES2, n3=1),
                   gates=sc_gates((True, True, False, True), (True,) * 4))
    assert set(zip(table["stage1"], table["stage2"])) == {(0, 0), (1, 0), (3, 3)}


def test_choose_combination_reads_the_table_with_its_extra_blocks_column(subset_case):
    _, _, table, _ = subset_case
    keys = ["stage1", "stage2", "topups", "current", "wape"]
    for min_gain in (10.0, 0.0):
        chosen = choose_combination(table, min_gain)
        same = choose_combination(table.drop(columns="extra_blocks"), min_gain)
        assert chosen.name == same.name
        assert [chosen[k] for k in keys] == [same[k] for k in keys]
    winner = choose_combination(table, 0.0)
    assert winner["extra_blocks"] == NAMES1[winner["stage1"]]


# --- ALWAYS_HASHED_BLOCKS and hashed_table ---------------------------------------------------

def test_always_hashed_blocks_are_the_three_base_blocks():
    assert ALWAYS_HASHED_BLOCKS == ("history", "calendar", "receivables")
    assert set(ALWAYS_HASHED_BLOCKS) < set(FEATURE_BLOCKS)


BLOCKS = {**BLOCK_COLUMNS, "dues": ["due_share", "due_gap"], "habits": ["habit_day"]}


def small_table():
    columns = (["key", "origin_week"] + [c for cs in BLOCKS.values() for c in cs]
               + ["amount_deflated"])
    return pd.DataFrame([[f"r{i}", i] + [float(i * 10 + n) for n in range(len(columns) - 2)]
                         for i in range(3)], columns=columns)


@pytest.mark.parametrize("blocks, dropped", [
    (("history", "calendar", "receivables"), ["due_share", "due_gap", "habit_day"]),
    (("history", "calendar", "receivables", "dues"), ["habit_day"]),
    (("history", "calendar", "receivables", "habits"), ["due_share", "due_gap"]),
    (("history", "calendar", "receivables", "dues", "habits"), []),
    # The base blocks are hashed even when a stage does not use them.
    (("history",), ["due_share", "due_gap", "habit_day"]),
    (("dues",), ["habit_day"]),
])
def test_hashed_table_drops_only_the_unused_extra_blocks(blocks, dropped):
    table = small_table()
    found = hashed_table(table, BLOCKS, blocks)
    assert list(found.columns) == [c for c in table.columns if c not in dropped]
    pd.testing.assert_frame_equal(found, table.drop(columns=dropped))


def test_hashed_table_keeps_columns_of_no_block():
    found = hashed_table(small_table(), BLOCKS, ("history",))
    assert {"key", "origin_week", "amount_deflated"} <= set(found.columns)


def test_hashed_table_does_not_change_the_table():
    table = small_table()
    before = table.copy()
    hashed_table(table, BLOCKS, ("history",))
    pd.testing.assert_frame_equal(table, before)


def test_hashed_table_without_extra_blocks_in_block_columns_is_the_table():
    table = small_table().drop(columns=["due_share", "due_gap", "habit_day"])
    pd.testing.assert_frame_equal(hashed_table(table, BLOCK_COLUMNS, ("history",)), table)


def test_hashed_table_a_block_column_the_table_lacks_raises_key_error():
    table = small_table().drop(columns="habit_day")
    with pytest.raises(KeyError):
        hashed_table(table, BLOCKS, ("history",))


# --- grid_predictions: the cache per set of feature blocks -----------------------------------

ORIGINS = [18, 22]
EXT_BLOCKS = {**BLOCK_COLUMNS, "dues": ["due_x"], "habits": ["habit_x"]}


def ext_table(seed=4):
    """test_forecast's table with one dues and one habits column (seeded values)."""
    table = make_table()
    rng = np.random.default_rng(seed)
    return table.assign(due_x=rng.normal(size=len(table)), habit_x=rng.normal(size=len(table)))


def ext_candidates(settings, subsets=None):
    """One stage-1 and one stage-2 candidate per subset of ("dues", "habits") (or `subsets`):
    the current settings with its blocks, as grid_candidates lists them."""
    subsets = block_subsets(BOTH) if subsets is None else subsets

    def listed(stage):
        return [Candidate(params={"extra_blocks": subset_name(s)},
                          settings=with_extra_blocks(stage, s), class_weight="none")
                for s in subsets]

    return GridCandidates(stage1=listed(settings.stage1), stage2=listed(settings.stage2),
                          topups=[settings.topups])


def record_fits(monkeypatch):
    """Wrap the two stage functions as grid sees them; record (stage, origin, blocks)."""
    calls = []

    def wrap(name, real):
        def step(train_rows, forecast_rows, setting, *args):
            calls.append((name, int(forecast_rows["origin_week"].iloc[0]),
                          setting.feature_blocks))
            return real(train_rows, forecast_rows, setting, *args)
        return step

    monkeypatch.setattr(grid_module, "stage1_predictions", wrap("stage1", stage1_predictions))
    monkeypatch.setattr(grid_module, "stage2_predictions", wrap("stage2", stage2_predictions))
    return calls


def ext_run(table, cache_folder, candidates=None, block_columns=EXT_BLOCKS):
    settings = g_settings()
    kwargs = g_inputs()
    kwargs["block_columns"] = block_columns
    return grid_predictions(table, ORIGINS, candidates or ext_candidates(settings), settings,
                            cache_folder=cache_folder, **kwargs)


def all_fits(candidates, origins=ORIGINS):
    return sorted((stage, origin, c.settings.feature_blocks) for origin in origins
                  for stage, listed in (("stage1", candidates.stage1),
                                        ("stage2", candidates.stage2))
                  for c in listed)


@pytest.fixture(scope="module")
def ext_cache(tmp_path_factory):
    folder = tmp_path_factory.mktemp("extra_blocks_cache")
    result = ext_run(ext_table(), folder)
    return folder, result


def copied(ext_cache, tmp_path):
    target = tmp_path / "cache"
    shutil.copytree(ext_cache[0], target)
    return target


def test_every_subset_is_fitted_once_per_origin_with_its_own_blocks(tmp_path, monkeypatch):
    calls = record_fits(monkeypatch)
    ext_run(ext_table(), tmp_path)
    candidates = ext_candidates(g_settings())
    assert sorted(calls) == all_fits(candidates)
    assert len(calls) == 2 * 2 * 4
    assert {blocks for _, _, blocks in calls} == {
        ("history", "calendar", "receivables"),
        ("history", "calendar", "receivables", "dues"),
        ("history", "calendar", "receivables", "habits"),
        ("history", "calendar", "receivables", "dues", "habits")}


def test_candidates_with_different_blocks_never_share_a_cache_file(ext_cache):
    folder, _ = ext_cache
    candidates = ext_candidates(g_settings())
    names = [f"{stage}_origin{origin}_{setting_hash(c.settings)}"
             for origin in ORIGINS
             for stage, listed in (("stage1", candidates.stage1), ("stage2", candidates.stage2))
             for c in listed]
    assert len(set(names)) == len(names) == 16
    files = sorted(p.name for p in Path(folder).iterdir())
    assert files == sorted([f"{n}.pkl" for n in names] + [f"{n}_fingerprint.json" for n in names])


def test_a_second_run_reads_every_subset_from_the_cache(ext_cache, tmp_path, monkeypatch):
    _, result = ext_cache
    calls = record_fits(monkeypatch)
    found = ext_run(ext_table(), copied(ext_cache, tmp_path))
    assert calls == []
    pd.testing.assert_frame_equal(found.p, result.p)
    pd.testing.assert_frame_equal(found.m, result.m)


def changed(column, rows=None):
    """ext_table with `column` changed on every row (or on `rows`)."""
    table = ext_table()
    where = table.index if rows is None else rows
    table.loc[where, column] = table.loc[where, column] * 3.0 + 1.0
    return table


def uses(block):
    return lambda blocks: block in blocks


@pytest.mark.parametrize("column, refit", [("habit_x", uses("habits")), ("due_x", uses("dues"))],
                         ids=["habits", "dues"])
def test_a_change_in_one_extra_block_refits_only_the_fits_that_use_it(ext_cache, tmp_path,
                                                                     monkeypatch, column, refit):
    _, result = ext_cache
    calls = record_fits(monkeypatch)
    found = ext_run(changed(column), copied(ext_cache, tmp_path))
    candidates = ext_candidates(g_settings())
    expected = [fit for fit in all_fits(candidates) if refit(fit[2])]
    assert sorted(calls) == expected
    assert len(calls) == 8
    # The fits that do not use the block are read back unchanged.
    for number, c in enumerate(candidates.stage1):
        if not refit(c.settings.feature_blocks):
            pd.testing.assert_series_equal(found.p[number], result.p[number])
    for number, c in enumerate(candidates.stage2):
        if not refit(c.settings.feature_blocks):
            pd.testing.assert_series_equal(found.m[number], result.m[number])


def test_a_change_in_the_extra_columns_leaves_the_base_only_fits_valid(ext_cache, tmp_path,
                                                                     monkeypatch):
    calls = record_fits(monkeypatch)
    table = changed("habit_x")
    table["due_x"] = 0.0
    ext_run(table, copied(ext_cache, tmp_path))
    base = ("history", "calendar", "receivables")
    assert all(blocks != base for _, _, blocks in calls)
    assert len(calls) == 12


@pytest.mark.parametrize("column", ["business_days", "open_balance", "mean_receipts_4"])
def test_a_change_in_a_base_column_refits_everything(ext_cache, tmp_path, monkeypatch, column):
    calls = record_fits(monkeypatch)
    table = ext_table()
    late = table.index[table["origin_week"] == 30][:3]
    table.loc[late, column] = 1.0
    ext_run(table, copied(ext_cache, tmp_path))
    assert sorted(calls) == all_fits(ext_candidates(g_settings()))


def test_a_change_in_a_column_of_no_block_refits_everything(ext_cache, tmp_path, monkeypatch):
    # Only the unused extra blocks are left out of the hash: any other column still counts.
    calls = record_fits(monkeypatch)
    table = ext_table()
    table.loc[table.index[:2], "amount"] = 12345.0
    ext_run(table, copied(ext_cache, tmp_path))
    assert len(calls) == 16


def test_a_cache_made_before_the_extra_columns_existed_still_serves_the_base_fits(
        tmp_path, monkeypatch):
    # The pre-U18.5 shape: a table and block_columns without dues or habits. Its base fits'
    # cache stays valid once the table gains the rhythm columns (they are not hashed for them).
    settings = g_settings()
    base_only = ext_candidates(settings, subsets=[()])
    ext_run(make_table(), tmp_path, candidates=base_only, block_columns=BLOCK_COLUMNS)
    calls = record_fits(monkeypatch)
    ext_run(ext_table(), tmp_path)
    base = ("history", "calendar", "receivables")
    assert not any(blocks == base for _, _, blocks in calls)
    assert len(calls) == 12


def test_the_base_fits_equal_a_run_on_the_table_without_the_extra_columns(ext_cache, tmp_path):
    _, result = ext_cache
    settings = g_settings()
    alone = ext_run(make_table(), tmp_path, candidates=ext_candidates(settings, subsets=[()]),
                    block_columns=BLOCK_COLUMNS)
    pd.testing.assert_series_equal(alone.p[0], result.p[0])
    pd.testing.assert_series_equal(alone.m[0], result.m[0])


def record_inputs(monkeypatch):
    seen = []
    real = grid_module.cached_fit

    def wrapped(stage, origin, setting, inputs, cache_folder, fit):
        seen.append((stage, origin, setting.feature_blocks, inputs))
        return real(stage, origin, setting, inputs, cache_folder, fit)

    monkeypatch.setattr(grid_module, "cached_fit", wrapped)
    return seen


def test_the_fit_inputs_hold_the_hashed_tables_fingerprint_and_its_blocks(tmp_path,
                                                                        monkeypatch):
    seen = record_inputs(monkeypatch)
    table = ext_table()
    settings = g_settings()
    ext_run(table, tmp_path)
    assert len(seen) == 16
    for stage, _, blocks, inputs in seen:
        expected_columns = {b: cs for b, cs in EXT_BLOCKS.items()
                            if b in blocks or b in ALWAYS_HASHED_BLOCKS}
        assert inputs[0] == asdict(settings.receivables)
        assert inputs[1] == expected_columns
        assert list(inputs[1]) == [b for b in EXT_BLOCKS if b in expected_columns]
        assert inputs[2] == sorted(F_COMPANIES)
        assert inputs[3] == grid_module.table_fingerprint(hashed_table(table, EXT_BLOCKS, blocks))
        if stage == "stage2":
            assert len(inputs) == 6 and inputs[5] is settings.stage1.company_as_feature
        else:
            assert len(inputs) == 5


def test_the_base_fits_inputs_are_those_of_the_table_without_extra_columns(tmp_path,
                                                                         monkeypatch):
    seen = record_inputs(monkeypatch)
    ext_run(ext_table(), tmp_path)
    base = ("history", "calendar", "receivables")
    inputs = [i for _, _, blocks, i in seen if blocks == base]
    assert inputs
    for found in inputs:
        assert found[1] == BLOCK_COLUMNS
        assert found[3] == grid_module.table_fingerprint(make_table())


def test_the_same_blocks_share_one_set_of_inputs_across_the_stages(tmp_path, monkeypatch):
    seen = record_inputs(monkeypatch)
    ext_run(ext_table(), tmp_path)
    by_blocks = {}
    for stage, _, blocks, inputs in seen:
        by_blocks.setdefault(blocks, set()).add(repr(inputs[:5]))
    assert len(by_blocks) == 4
    assert all(len(found) == 1 for found in by_blocks.values())
    assert len({next(iter(found)) for found in by_blocks.values()}) == 4


def test_the_chosen_candidates_alone_read_the_grids_cache(ext_cache, tmp_path, monkeypatch):
    # grid_test reuses grid_predictions with the chosen candidates, feature_blocks included:
    # a chosen "dues + habits" pair fitted in the grid is read back, not refitted.
    _, result = ext_cache
    candidates = ext_candidates(g_settings())
    chosen = GridCandidates(stage1=[candidates.stage1[3]], stage2=[candidates.stage2[3]],
                            topups=candidates.topups)
    calls = record_fits(monkeypatch)
    found = ext_run(ext_table(), copied(ext_cache, tmp_path), candidates=chosen)
    assert calls == []
    np.testing.assert_array_equal(found.p[0].to_numpy(), result.p[3].to_numpy())
    np.testing.assert_array_equal(found.m[0].to_numpy(), result.m[3].to_numpy())


def test_grid_predictions_on_grid_candidates_with_extra_blocks(tmp_path, monkeypatch):
    # The real candidate lists, cut to one grid point per stage and subset (four fits each).
    settings = g_settings()
    real = grid_candidates(SimpleNamespace(
        stage1=settings.stage1, stage2=settings.stage2, topups=settings.topups,
        grid=dataclasses.replace(GRID, extra_blocks=BOTH, stage1_max_iter=(10,),
                                 stage1_learning_rate=(0.1,), stage1_max_leaf_nodes=(7,),
                                 stage2_loss=("poisson",), stage2_max_iter=(10,),
                                 stage2_learning_rate=(0.1,), stage2_max_leaf_nodes=(7,),
                                 stage2_scale=("none",), stage2_balance_floor=("median",),
                                 topup_windows=(8,))))
    assert len(real.stage1) == len(real.stage2) == 4
    calls = record_fits(monkeypatch)
    found = ext_run(ext_table(), tmp_path, candidates=real)
    assert list(found.p.columns) == [0, 1, 2, 3] and list(found.m.columns) == [0, 1, 2, 3]
    assert sorted(calls) == all_fits(real)
