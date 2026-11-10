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

U16.3, `grid_predictions`: every candidate's forecast at every origin, nothing scored yet.
Each stage-1 and each stage-2 candidate is fitted at each origin on the rows known there and
predicts the rows at it (its p or m); each top-up candidate is forecast at all origins at
once. Beside them, per row, what happened and the two references the gates (U16.4) need.
Every fit is cached on its own (`cache.load_with_cache`), keyed by its stage, origin and
setting: a grid with one more candidate fits only that one, and a run stopped after hours
keeps every fit it finished.

U16.4, `grid_gates`: which stage-1 and stage-2 candidates make sense on their own, judged on
the labelled rows of every validation origin pooled (`diagnostic_joint.stage1_gates` and
`stage2_gates`, with the [grid] tolerances). "The best" a gate measures against is the best
candidate of the grid, whichever it is. Top-ups have no gate: they are judged in the totals
only (U16.5). The current settings are scored even if they fail a gate (U16.6): they are what
is kept unless beaten.
"""
import dataclasses
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .cache import load_with_cache, source_fingerprint, table_fingerprint
from .diagnostic import Candidate, reference_probabilities
from .diagnostic_joint import stage1_gates, stage1_grid, stage2_gates, stage2_grid
from .evaluate import MODEL_CODE, check_forecast_origins
from .forecast import CUSTOMER_COLUMNS, stage1_predictions, stage2_predictions
from .stage2 import amount_references
from .topups import topup_forecasts
from .training_table import rows_at_origin, rows_known_at

# The columns of `GridPredictions.rows`, in order: the customer row, what happened (labels
# known only in hindsight), and the references of the gates: the base-rate p (stage 1's
# log-loss skill) and the training-mean amount (stage 2's deviance skill).
ROW_COLUMNS = CUSTOMER_COLUMNS[:6] + ["label_status", "paid", "amount_deflated", "base_rate",
                                      "training_mean"]


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


@dataclass(frozen=True)
class GridPredictions:
    """Every candidate's forecast at every origin (U16.3). Candidate numbers are those of
    `GridCandidates` (0 = the current settings)."""
    rows: pd.DataFrame          # one row per customer, origin and h: ROW_COLUMNS; the table's
                                # index
    p: pd.DataFrame             # column i = stage-1 candidate i's p; the index of `rows`
    m: pd.DataFrame             # column j = stage-2 candidate j's m; the index of `rows`
    topups: list[pd.DataFrame]  # [k] = top-up candidate k (`topup_forecasts`' table)


def setting_hash(setting) -> str:
    """A short hash of one stage's settings (a dataclass), for the name of its cache files."""
    text = json.dumps(asdict(setting), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cached_fit(stage: str, origin: int, setting, inputs: list, cache_folder: Path,
               fit: Callable[[], pd.Series]) -> pd.Series:
    """`fit()`, one stage's prediction with `setting` at `origin`, read from `cache_folder`
    if it was made before from the same `inputs` (everything else it is made from), else
    made and saved there. The file name holds the stage, the origin and `setting_hash`."""
    name = f"{stage}_origin{origin}_{setting_hash(setting)}"
    fingerprint = [stage, origin, asdict(setting), *inputs]
    return load_with_cache(name, fingerprint, fit, cache_folder)


def grid_predictions(table: pd.DataFrame, origins: list[int], candidates: GridCandidates,
                     settings, *, block_columns: dict[str, list[str]], companies: list[str],
                     new_cash: pd.DataFrame, refunds: pd.DataFrame, unidentified: pd.DataFrame,
                     levels: pd.Series, calendar: pd.DataFrame, cache_folder: Path,
                     on_fit_done: Callable[[int, int, str, int, int, float], None] | None = None
                     ) -> GridPredictions:
    """Every candidate of `candidates` at every origin of `origins` (see the module
    docstring). `table`, `block_columns`, `companies` and the top-up inputs are those of
    `evaluate.rolling_forecasts`; `settings` the whole settings ([receivables] horizons,
    [stage1] company_as_feature, and the [diagnostic] reference windows are read).

    Before the first fit: the origins are checked as `rolling_forecasts` checks them
    (`check_forecast_origins`) and every top-up candidate is forecast; at each origin the
    references are made before its fits. A wrong setting thus fails in seconds. Each fit is read from `cache_folder` if it was made before from the
    same inputs, and fitted and saved there if not. p and m are not checked: a candidate
    with a p that is not a number or an m below 0 is refused by the gates (U16.4), so one
    broken candidate does not stop the grid.

    `on_fit_done(done, total, stage, origin, candidate, seconds)`, if given, is called after
    each fit, read from the cache or not (e.g. to print progress).

    Raises ValueError as `check_forecast_origins`, the top-ups, the references and the fits
    do, and if an origin has no rows.
    """
    horizons = settings.receivables.horizons
    origins = check_forecast_origins(table, origins, companies, horizons)
    topups = [topup_forecasts(new_cash, refunds, unidentified, levels, calendar, origins,
                              horizons, topup.method, topup.window, topup.with_level)
              for topup in candidates.topups]

    # What every fit is made from besides its own setting and origin; the table's hash takes
    # seconds, so it is made once.
    here = Path(__file__).parent
    inputs = [asdict(settings.receivables), block_columns, sorted(companies),
              table_fingerprint(table), source_fingerprint(*(here / name for name in MODEL_CODE))]
    company_as_feature = settings.stage1.company_as_feature
    diagnostic = settings.diagnostic

    rows, p_parts, m_parts = [], [], []
    total = len(origins) * (len(candidates.stage1) + len(candidates.stage2))
    done = 0
    for origin in origins:
        train_rows = rows_known_at(table, origin)
        forecast_rows = rows_at_origin(table, origin)
        if forecast_rows.empty:
            raise ValueError(f"the table has no rows at origin week {origin}")

        # What happened, and the gates' references, learned from the rows known at T. Made
        # before the fits, so a wrong [diagnostic] window fails in seconds.
        base_rate = reference_probabilities(train_rows, forecast_rows,
                                            diagnostic.reference_window,
                                            diagnostic.reference_prior_weeks)["base rate"]
        training_mean = amount_references(train_rows, forecast_rows,
                                          diagnostic.amount_reference_window,
                                          diagnostic.reference_prior_weeks)["training mean"]
        rows.append(forecast_rows[ROW_COLUMNS[:9]].assign(base_rate=base_rate,
                                                          training_mean=training_mean))

        # Stage 1: each candidate fitted once at this origin (or read back).
        p_at = {}
        for number, candidate in enumerate(candidates.stage1):
            started = time.perf_counter()
            p_at[number] = cached_fit(
                "stage1", origin, candidate.settings, inputs, cache_folder,
                lambda setting=candidate.settings: stage1_predictions(
                    train_rows, forecast_rows, setting, block_columns, companies))
            done += 1
            if on_fit_done is not None:
                on_fit_done(done, total, "stage 1", origin, number,
                            time.perf_counter() - started)

        # Stage 2 the same way. company_as_feature is the one stage-1 setting it reads.
        m_at = {}
        for number, candidate in enumerate(candidates.stage2):
            started = time.perf_counter()
            m_at[number] = cached_fit(
                "stage2", origin, candidate.settings, inputs + [company_as_feature],
                cache_folder,
                lambda setting=candidate.settings: stage2_predictions(
                    train_rows, forecast_rows, setting, block_columns, companies,
                    company_as_feature))
            done += 1
            if on_fit_done is not None:
                on_fit_done(done, total, "stage 2", origin, number,
                            time.perf_counter() - started)

        p_parts.append(pd.DataFrame(p_at, index=forecast_rows.index))
        m_parts.append(pd.DataFrame(m_at, index=forecast_rows.index))

    return GridPredictions(rows=pd.concat(rows), p=pd.concat(p_parts), m=pd.concat(m_parts),
                           topups=topups)


@dataclass(frozen=True)
class GridGates:
    """The gates of every stage-1 and stage-2 candidate (U16.4): `stage1_gates`' and
    `stage2_gates`' tables, with each candidate's grid settings after its number."""
    stage1: pd.DataFrame        # one row per stage-1 candidate; column "stage1" its number
    stage2: pd.DataFrame        # one row per stage-2 candidate; column "stage2" its number


def with_params(gates: pd.DataFrame, candidates: list[Candidate],
                number_column: str) -> pd.DataFrame:
    """A gate table with each candidate's grid settings (`params`) added after its number,
    row by row (the gate table may be sorted or filtered)."""
    params = pd.DataFrame([candidates[int(i)].params for i in gates[number_column]],
                          index=gates.index)
    return pd.concat([gates[[number_column]], params, gates.drop(columns=number_column)],
                     axis=1)


def grid_gates(predictions: GridPredictions, candidates: GridCandidates,
               settings) -> GridGates:
    """The gates of every candidate (see the module docstring), on the rows of `predictions`
    whose label is known, all origins together; the tolerances are [grid]'s.

    Stage 1 is judged on every labelled row (paid or not, against the base rate); stage 2 on
    the labelled rows that paid (the deflated amount, against the training mean).

    Raises ValueError if `predictions` and `candidates` do not have the same number of
    stage-1 or stage-2 candidates, if no row is labelled or none of them paid, and as the
    gate functions do.
    """
    for stage, columns, listed in (("stage-1", predictions.p.columns, candidates.stage1),
                                   ("stage-2", predictions.m.columns, candidates.stage2)):
        if len(columns) != len(listed):
            raise ValueError(f"the predictions hold {len(columns)} {stage} candidates, the "
                             f"candidates {len(listed)}")
    rows = predictions.rows
    labelled = rows[rows["label_status"] == "labelled"]
    if labelled.empty:
        raise ValueError("no row of the predictions is labelled: the gates have nothing to "
                         "judge")
    paid = labelled[labelled["paid"].astype(bool)]
    if paid.empty:
        raise ValueError("no labelled row of the predictions paid: stage 2 cannot be judged")

    grid = settings.grid
    gates1 = stage1_gates(labelled["paid"].astype(bool),
                          [predictions.p.loc[labelled.index, i] for i in predictions.p.columns],
                          labelled["base_rate"], grid.stage1_log_loss_tolerance,
                          grid.stage1_slope_band)
    gates2 = stage2_gates(paid["amount_deflated"],
                          [predictions.m.loc[paid.index, j] for j in predictions.m.columns],
                          paid["training_mean"], grid.stage2_skill_tolerance,
                          grid.stage2_ratio_band)
    return GridGates(stage1=with_params(gates1, candidates.stage1, "stage1"),
                     stage2=with_params(gates2, candidates.stage2, "stage2"))
