"""The settings grid (U16): which settings of the whole model are tried, over many origins.

U16.2, `grid_candidates`: the settings to try, from [grid]. Three lists, one per part of the
model, each with the current settings ([stage1], [stage2], [topups]) first:

  stage 1   the current [stage1], then every other combination of the [grid] stage-1 values
  stage 2   the current [stage2], then every other combination of the [grid] stage-2 values
  top-ups   the current [topups], then the trailing mean over each of [grid] topup_windows,
            then the ridge with and/or without the recent level ([grid] topup_ridge_with_level)

The current settings are always tried, even when the grid does not list them: a new pair must
beat them by [grid] min_gain WAPE to replace them (U16.6). Every candidate keeps
[stage1] company_as_feature and the feature blocks of its stage: stage 2 reads stage 1's
settings only through company_as_feature, so each stage-1 and each stage-2 setting can be
fitted once per origin and paired afterwards (U16.1).
"""
import dataclasses
from dataclasses import dataclass

from .diagnostic import Candidate
from .diagnostic_joint import stage1_grid, stage2_grid


@dataclass(frozen=True)
class GridCandidates:
    """The settings the grid tries; in each list the current settings come first."""
    stage1: list[Candidate]     # each with a copy of [stage1]
    stage2: list[Candidate]     # each with a copy of [stage2]
    topups: list                # each a copy of [topups] (`TopupSettings`)


def grid_candidates(settings) -> GridCandidates:
    """The settings to try (see the module docstring), from `settings` (a `Settings`).

    `stage1_grid` and `stage2_grid` read the [grid] values, which have the names of the
    [joint_diagnostic] ones. A combination equal to the current settings is not listed twice.
    A top-up candidate is a copy of [topups] with the method and its own setting replaced (the
    mean's window, the ridge's with_level); the setting the method does not read is kept.
    """
    stage1, stage2, topups = settings.stage1, settings.stage2, settings.topups
    grid = settings.grid

    current_stage1 = Candidate(
        params={"max_iter": stage1.max_iter, "learning_rate": stage1.learning_rate,
                "max_leaf_nodes": stage1.max_leaf_nodes},
        settings=stage1, class_weight="none")
    floor = stage2.balance_floor if stage2.scale == "open balance" else "-"
    current_stage2 = Candidate(
        params={"loss": stage2.loss, "max_iter": stage2.max_iter,
                "learning_rate": stage2.learning_rate, "max_leaf_nodes": stage2.max_leaf_nodes,
                "scale": stage2.scale, "balance_floor": floor},
        settings=stage2, class_weight="none")

    stage1_candidates = [current_stage1] + [
        candidate for candidate in stage1_grid(stage1, grid) if candidate.settings != stage1]
    stage2_candidates = [current_stage2] + [
        candidate for candidate in stage2_grid(stage2, grid) if candidate.settings != stage2]

    topup_candidates = [topups]
    for window in grid.topup_windows:
        mean = dataclasses.replace(topups, method="mean", window=window)
        if mean != topups:
            topup_candidates.append(mean)
    for with_level in grid.topup_ridge_with_level:
        ridge = dataclasses.replace(topups, method="ridge", with_level=with_level)
        if ridge != topups:
            topup_candidates.append(ridge)

    return GridCandidates(stage1=stage1_candidates, stage2=stage2_candidates,
                          topups=topup_candidates)
