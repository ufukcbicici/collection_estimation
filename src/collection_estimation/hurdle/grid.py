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

U18.5, the feature-block subsets: [grid] extra_blocks (e.g. ["dues", "habits"]) are tried on
top of each stage's own feature_blocks, every subset of them (none, dues, habits, both), and
each stage list above is made once per subset (`block_subsets`). A combination uses the same
subset in both stages; its params name it ("extra_blocks"). Candidate 0 stays the current
settings with no extra block, so a block is kept only if it beats them by min_gain.

U16.3, `grid_predictions`: every candidate's forecast at every origin, nothing scored yet.
Each stage-1 and each stage-2 candidate is fitted at each origin on the rows known there and
predicts the rows at it (its p or m); each top-up candidate is forecast at all origins at
once. Beside them, per row, what happened and the two references the gates (U16.4) need.
Every fit is cached on its own (`cache.load_with_cache`), keyed by its stage, origin and
setting: a grid with one more candidate fits only that one, and a run stopped after hours
keeps every fit it finished. The table is hashed without the columns of the extra blocks a
fit does not use, so a change to how a block is built ([rhythm]) refits only the fits that
use it. With [grid] workers above 1, the fits of an origin run that many at a time, each in
its own process with its share of the threads (`fit_on_worker`); the predictions are the same.

U16.4, `grid_gates`: which stage-1 and stage-2 candidates make sense on their own, judged on
the labelled rows of every validation origin pooled (`diagnostic_joint.stage1_gates` and
`stage2_gates`, with the [grid] tolerances). "The best" a gate measures against is the best
candidate of the grid, whichever it is. Top-ups have no gate: they are judged in the totals
only (U16.5). The current settings are scored even if they fail a gate (U16.6): they are what
is kept unless beaten.

U16.5, `grid_scores`: every combination of a scored stage-1 candidate (eligible, or the
current settings), a scored stage-2 candidate (the same) and a top-up candidate, forecast at
every origin as `forecast.forecast` does (p x m added up per company and h, plus the top-ups,
times the price level projected at the origin; nominal TRY) and scored against the cash that
came by `evaluate.evaluation_scores`, as step 17 scores the model. Nothing is chosen here
(U16.6), and p and m are not calibrated. The combinations are forecast and scored in
batches of SCORE_BATCH, so the forecasts of a whole grid are never held at once.

U16.6, `grid_choice`: the combination kept (the current settings unless beaten by more than
[grid] min_gain WAPE), its calibration, and, on the same validation origins, how it compares
with the current settings and with the references (`deflated` variants, financial rule), which
are made here so the grid does not need step 17.

U16.7, `grid_test`: the v1 test, only with [grid] run_test. What `grid_choice` froze (the
three settings and the corrections of p and m) is judged on the [evaluation] test origins
against the same references. The settings are fitted again at each test origin on the rows
known there; the corrections are not learned again. A report, nothing is chosen from it.
"""
import dataclasses
import hashlib
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .cache import load_with_cache, source_fingerprint, table_fingerprint
from .diagnostic import Candidate, reference_probabilities
from .diagnostic_joint import stage1_gates, stage1_grid, stage2_gates, stage2_grid
from .calibrate import CalibrationChoice, choose_calibration, corrected_forecast
from .evaluate import (HORIZON_TOTAL, HURDLE, MODEL_CODE, POOLED, STACK_COLUMNS,
                       check_forecast_origins, deflated_reference, evaluation_scores,
                       paired_wape_bootstrap, rule_reference, stack_forecasts)
from .forecast import (CONSOLIDATED, CUSTOMER_COLUMNS, Forecast, forecast_totals,
                       stage1_predictions, stage2_predictions)
from .prices import projected_levels
from .settings import FEATURE_BLOCKS
from .stage2 import amount_references
from .topups import topup_forecasts
from .training_table import rows_at_origin, rows_known_at

# The columns of `GridPredictions.rows`, in order: the customer row, what happened (labels
# known only in hindsight), and the references of the gates: the base-rate p (stage 1's
# log-loss skill) and the training-mean amount (stage 2's deviance skill).
ROW_COLUMNS = CUSTOMER_COLUMNS[:6] + ["label_status", "paid", "amount_deflated", "base_rate",
                                      "training_mean"]

# The columns of `grid_scores`' table, in order: the candidate numbers, the extra feature
# blocks they use (U18.5), whether it is the current settings, then the scores of the
# consolidated totals (WAPE over every (origin, h), WAPE of each origin's total over its h
# weeks, bias) and the mean of the companies' WAPE.
GRID_SCORE_COLUMNS = ["stage1", "stage2", "topups", "extra_blocks", "current", "wape",
                      "total_wape", "bias", "company_wape"]

# How many combinations `grid_scores` forecasts and scores at a time. A combination's score
# reads its own rows only, so the batch size changes the memory used (about 32 MB of
# forecasts per batch), not the scores.
SCORE_BATCH = 500

# The feature blocks every fit's cache name hashes, used or not: stage 2 reads the open balance
# (receivables) and the history means for its scale, whatever its feature_blocks.
ALWAYS_HASHED_BLOCKS = ("history", "calendar", "receivables")


@dataclass(frozen=True)
class GridCandidates:
    """The settings the grid tries; in each list the current settings come first."""
    stage1: list[Candidate]     # each with a copy of [stage1]
    stage2: list[Candidate]     # each with a copy of [stage2]
    topups: list                # each a copy of [topups] (`TopupSettings`)


def block_subsets(extra_blocks: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Every subset of `extra_blocks`, smallest first; within a size, in the order of
    `extra_blocks`. E.g. ("dues", "habits") -> (), ("dues",), ("habits",), ("dues", "habits")."""
    subsets = []
    for size in range(len(extra_blocks) + 1):
        subsets.extend(itertools.combinations(extra_blocks, size))
    return subsets


def subset_name(subset: tuple[str, ...]) -> str:
    """The name of a subset of extra blocks in the params and tables: "none" or
    "dues + habits"."""
    return " + ".join(subset) if subset else "none"


def with_extra_blocks(stage_settings, subset: tuple[str, ...]):
    """A copy of one stage's settings with the blocks of `subset` added to its feature_blocks,
    in the order of FEATURE_BLOCKS (as [stage1] and [stage2] are read)."""
    blocks = [block for block in FEATURE_BLOCKS
              if block in stage_settings.feature_blocks or block in subset]
    return dataclasses.replace(stage_settings, feature_blocks=tuple(blocks))


def grid_candidates(settings) -> GridCandidates:
    """The settings to try (see the module docstring), from `settings` (a `Settings`).

    For each subset of [grid] extra_blocks (`block_subsets`), the current [stage1] with those
    blocks added, then every combination of the [grid] stage-1 values with them; stage 2 the
    same way. `stage1_grid` and `stage2_grid` read the [grid] values, which have the names of
    the [joint_diagnostic] ones. A combination equal to the current settings is not listed
    twice. Every stage candidate's params end with "extra_blocks" (`subset_name`).

    A top-up candidate is a copy of [topups] with the method and its own setting replaced (the
    mean's window, the ridge's with_level); the setting the method does not read is kept. The
    top-ups do not depend on the feature blocks, so they are listed once.
    """
    stage1, stage2, topups = settings.stage1, settings.stage2, settings.topups
    grid = settings.grid

    stage1_candidates, stage2_candidates = [], []
    for subset in block_subsets(grid.extra_blocks):
        name = subset_name(subset)

        # Stage 1: the current settings with this subset, then the grid's others.
        current1 = with_extra_blocks(stage1, subset)
        stage1_candidates.append(Candidate(
            params={"max_iter": stage1.max_iter, "learning_rate": stage1.learning_rate,
                    "max_leaf_nodes": stage1.max_leaf_nodes, "extra_blocks": name},
            settings=current1, class_weight="none"))
        for candidate in stage1_grid(current1, grid):
            if candidate.settings != current1:
                stage1_candidates.append(Candidate(
                    params={**candidate.params, "extra_blocks": name},
                    settings=candidate.settings, class_weight=candidate.class_weight))

        # Stage 2 the same way.
        current2 = with_extra_blocks(stage2, subset)
        floor = stage2.balance_floor if stage2.scale == "open balance" else "-"
        stage2_candidates.append(Candidate(
            params={"loss": stage2.loss, "max_iter": stage2.max_iter,
                    "learning_rate": stage2.learning_rate,
                    "max_leaf_nodes": stage2.max_leaf_nodes, "scale": stage2.scale,
                    "balance_floor": floor, "extra_blocks": name},
            settings=current2, class_weight="none"))
        for candidate in stage2_grid(current2, grid):
            if candidate.settings != current2:
                stage2_candidates.append(Candidate(
                    params={**candidate.params, "extra_blocks": name},
                    settings=candidate.settings, class_weight=candidate.class_weight))

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


def same_extra_blocks(stage1_candidate: Candidate, stage2_candidate: Candidate) -> bool:
    """True if a stage-1 and a stage-2 candidate use the same subset of extra blocks: only
    such a pair is a combination (one subset for both stages)."""
    return stage1_candidate.params["extra_blocks"] == stage2_candidate.params["extra_blocks"]


def combination_count(candidates: GridCandidates) -> int:
    """How many combinations the candidates make: each stage-1 and stage-2 pair with the same
    extra blocks, times every top-up."""
    pairs = 0
    for first in candidates.stage1:
        for second in candidates.stage2:
            if same_extra_blocks(first, second):
                pairs += 1
    return pairs * len(candidates.topups)


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


def hashed_table(table: pd.DataFrame, block_columns: dict[str, list[str]],
                 blocks: tuple[str, ...]) -> pd.DataFrame:
    """The part of `table` a fit with feature blocks `blocks` is hashed on: all of it but the
    columns of the blocks it does not use, other than the ALWAYS_HASHED_BLOCKS."""
    unused = [column for block, columns in block_columns.items()
              if block not in blocks and block not in ALWAYS_HASHED_BLOCKS
              for column in columns]
    return table.drop(columns=unused)


def cached_fit(stage: str, origin: int, setting, inputs: list, cache_folder: Path,
               fit: Callable[[], pd.Series]) -> pd.Series:
    """`fit()`, one stage's prediction with `setting` at `origin`, read from `cache_folder`
    if it was made before from the same `inputs` (everything else it is made from), else
    made and saved there. The file name holds the stage, the origin and `setting_hash`."""
    name = f"{stage}_origin{origin}_{setting_hash(setting)}"
    fingerprint = [stage, origin, asdict(setting), *inputs]
    return load_with_cache(name, fingerprint, fit, cache_folder)


def available_cpus(cgroup_folder: Path = Path("/sys/fs/cgroup")) -> int:
    """How many CPUs this process may use, at least 1: the machine's (`os.cpu_count`, or those
    this process is pinned to, where the system says), but no more than a CPU limit set on it.

    In a container (e.g. a Kubernetes pod with a CPU limit of 12000m), `os.cpu_count` counts
    every core of the machine, not the limit; giving each worker its share of those would run
    far more threads than the pod may use, and the pod would be held back. The limit is read
    from `cgroup_folder` (Linux): cgroup v2's cpu.max ("1200000 100000" = 12 CPUs, "max ..." =
    none), else cgroup v1's cpu/cpu.cfs_quota_us over cpu/cpu.cfs_period_us (-1 = none). A
    fraction of a CPU is rounded down. Where neither file is there (e.g. Windows), or a file
    cannot be read, does not hold numbers or gives a period of 0 or less, no limit.
    """
    if hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1

    limit = None
    v2 = cgroup_folder / "cpu.max"
    v1_quota = cgroup_folder / "cpu" / "cpu.cfs_quota_us"
    v1_period = cgroup_folder / "cpu" / "cpu.cfs_period_us"
    try:
        if v2.is_file():
            quota, period = v2.read_text().split()[:2]
            if quota != "max" and int(period) > 0:
                limit = int(quota) // int(period)
        elif v1_quota.is_file() and v1_period.is_file():
            quota, period = int(v1_quota.read_text()), int(v1_period.read_text())
            if quota > 0 and period > 0:
                limit = quota // period
    except (OSError, ValueError):
        # A file that cannot be read or does not hold numbers: no limit known.
        limit = None
    if limit is not None:
        cpus = min(cpus, limit)
    return max(1, cpus)


def fit_on_worker(stage: str, number: int, setting, origin: int, train_rows: pd.DataFrame,
                  forecast_rows: pd.DataFrame, block_columns: dict[str, list[str]],
                  companies: list[str], company_as_feature: bool, inputs: list,
                  cache_folder: Path, threads: int | None):
    """One fit of `grid_predictions`: stage `stage` ("stage1" or "stage2") with `setting` at
    `origin`, read from or saved to `cache_folder` (`cached_fit`, `inputs` as there; stage 2
    also hashes `company_as_feature`). With at most `threads` threads (None: no limit).

    Run in a worker process when [grid] workers is above 1, so it gets only plain data and
    builds the fit itself here (a function made in the caller cannot be sent to another
    process). Returns (stage, number, the prediction, the seconds it took), `number` being
    the candidate's, so the caller knows which fit came back.

    Raises ValueError if `stage` is neither, before anything is read or fitted.
    """
    if stage not in ("stage1", "stage2"):
        raise ValueError(f"stage is {stage!r}, not 'stage1' or 'stage2'")
    started = time.perf_counter()
    with threadpool_limits(threads):
        if stage == "stage1":
            prediction = cached_fit(
                "stage1", origin, setting, inputs, cache_folder,
                lambda: stage1_predictions(train_rows, forecast_rows, setting, block_columns,
                                           companies))
        else:   # "stage2"
            prediction = cached_fit(
                "stage2", origin, setting, inputs + [company_as_feature], cache_folder,
                lambda: stage2_predictions(train_rows, forecast_rows, setting, block_columns,
                                           companies, company_as_feature))
    return stage, number, prediction, time.perf_counter() - started


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

    With [grid] workers above 1, the fits of each origin (stage 1's first, the long ones,
    then stage 2's) run that many at a time in worker processes, each with the CPUs this
    process may use shared out (`available_cpus`, `fit_on_worker`); the p and m columns are still in the order of the
    candidates. With 1, one after another here, with every thread.

    `on_fit_done(done, total, stage, origin, candidate, seconds)`, if given, is called after
    each fit, read from the cache or not (e.g. to print progress); with several workers, in
    the order the fits finish.

    Raises ValueError as `check_forecast_origins`, the top-ups, the references and the fits
    do, and if an origin has no rows.
    """
    horizons = settings.receivables.horizons
    origins = check_forecast_origins(table, origins, companies, horizons)
    topups = [topup_forecasts(new_cash, refunds, unidentified, levels, calendar, origins,
                              horizons, topup.method, topup.window, topup.with_level)
              for topup in candidates.topups]

    # What every fit is made from besides its own setting and origin, per set of feature
    # blocks: the table without the extra blocks the fit does not use, and the columns of the
    # blocks hashed. A table's hash takes seconds, so each is made once.
    here = Path(__file__).parent
    code = source_fingerprint(*(here / name for name in MODEL_CODE))
    inputs_of = {}
    for candidate in candidates.stage1 + candidates.stage2:
        blocks = candidate.settings.feature_blocks
        if blocks in inputs_of:
            continue
        hashed = hashed_table(table, block_columns, blocks)
        hashed_columns = {block: columns for block, columns in block_columns.items()
                          if block in blocks or block in ALWAYS_HASHED_BLOCKS}
        inputs_of[blocks] = [asdict(settings.receivables), hashed_columns, sorted(companies),
                             table_fingerprint(hashed), code]
    company_as_feature = settings.stage1.company_as_feature
    diagnostic = settings.diagnostic
    # With several workers, each fit gets its share of the threads; alone, all of them.
    workers = settings.grid.workers
    threads = None if workers == 1 else max(1, available_cpus() // workers)

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

        # Every candidate fitted once at this origin (or read back): stage 1's first, the
        # long ones, so that stage 2's short fits fill the gaps at the end.
        fits = ([("stage1", number, candidate.settings)
                 for number, candidate in enumerate(candidates.stage1)]
                + [("stage2", number, candidate.settings)
                   for number, candidate in enumerate(candidates.stage2)])

        def arguments(stage, number, setting):
            return (stage, number, setting, origin, train_rows, forecast_rows, block_columns,
                    companies, company_as_feature, inputs_of[setting.feature_blocks],
                    cache_folder, threads)

        if workers == 1:
            # Here, one after another.
            results = (fit_on_worker(*arguments(*fit)) for fit in fits)
        else:
            # In the worker processes, each result as soon as it is done; the loop below
            # runs here, in this process.
            results = Parallel(n_jobs=workers, backend="loky",
                               return_as="generator_unordered")(
                delayed(fit_on_worker)(*arguments(*fit)) for fit in fits)
        p_at, m_at = {}, {}
        for stage, number, prediction, seconds in results:
            if stage == "stage1":
                p_at[number] = prediction
            else:
                m_at[number] = prediction
            done += 1
            if on_fit_done is not None:
                on_fit_done(done, total, "stage 1" if stage == "stage1" else "stage 2",
                            origin, number, seconds)

        # In the order of the candidates, whatever order the fits finished in.
        p_parts.append(pd.DataFrame({number: p_at[number]
                                     for number in range(len(candidates.stage1))},
                                    index=forecast_rows.index))
        m_parts.append(pd.DataFrame({number: m_at[number]
                                     for number in range(len(candidates.stage2))},
                                    index=forecast_rows.index))

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

    Every row is forecast, though, not only those judged. A candidate whose p is not a
    number in [0, 1], or whose m is not a number, on a row the gates do not judge is judged as
    if its p (or m) were unusable on every judged row: it fails every gate and does not set
    the best. Column "bad elsewhere" counts those rows, so a broken candidate does not stop
    the scores (U16.5).

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

    # Each candidate's values on the judged rows, and how many rows outside them are broken.
    p_judged, p_elsewhere = [], []
    for i in predictions.p.columns:
        p = predictions.p[i].astype("float64")
        broken = ~np.isfinite(p) | (p < 0) | (p > 1)
        p_elsewhere.append(int(broken.drop(labelled.index).sum()))
        p_judged.append(refused_if(p.loc[labelled.index], p_elsewhere[-1] > 0))
    m_judged, m_elsewhere = [], []
    for j in predictions.m.columns:
        m = predictions.m[j].astype("float64")
        broken = ~np.isfinite(m)
        m_elsewhere.append(int(broken.drop(paid.index).sum()))
        m_judged.append(refused_if(m.loc[paid.index], m_elsewhere[-1] > 0))

    grid = settings.grid
    gates1 = stage1_gates(labelled["paid"].astype(bool), p_judged, labelled["base_rate"],
                          grid.stage1_log_loss_tolerance, grid.stage1_slope_band)
    gates2 = stage2_gates(paid["amount_deflated"], m_judged, paid["training_mean"],
                          grid.stage2_skill_tolerance, grid.stage2_ratio_band)
    gates1.insert(gates1.columns.get_loc("bad p") + 1, "bad elsewhere", p_elsewhere)
    gates2.insert(gates2.columns.get_loc("bad m") + 1, "bad elsewhere", m_elsewhere)
    return GridGates(stage1=with_params(gates1, candidates.stage1, "stage1"),
                     stage2=with_params(gates2, candidates.stage2, "stage2"))


def refused_if(values: pd.Series, refused: bool) -> pd.Series:
    """`values`, or, if `refused`, NaN in every row: a candidate the gates must fail."""
    if refused:
        return pd.Series(float("nan"), index=values.index, name=values.name)
    return values


def scored_numbers(gates: pd.DataFrame, number_column: str) -> list[int]:
    """The candidates of one stage that are scored: the eligible ones, and the current
    settings (number 0) always."""
    scored = gates["eligible"] | (gates[number_column] == 0)
    return sorted(int(number) for number in gates.loc[scored, number_column])


def combination_name(stage1: int, stage2: int, topups: int) -> str:
    """The name of a combination of candidates in the stacked forecasts."""
    return f"stage1 {stage1}, stage2 {stage2}, top-ups {topups}"


def projected_at(origins: list[int], horizons: int, week_zero) -> dict[int, pd.Series]:
    """Per origin, the price levels of its target weeks projected at it
    (`prices.projected_levels` from `week_zero`), as `forecast.forecast` uses them."""
    return {origin: projected_levels(week_zero, origin,
                                     [origin + h for h in range(1, horizons + 1)])
            for origin in origins}


def combination_forecast(predictions: GridPredictions, stage1: int, stage2: int, topups: int,
                         *, projected: dict[int, pd.Series], companies: list[str],
                         horizons: int) -> Forecast:
    """The forecast of one combination of candidates at every origin of `predictions`, as
    `evaluate.rolling_forecasts` returns it: `customers` (p of stage-1 candidate `stage1`, m of
    stage-2 candidate `stage2`, expected = p x m) and `totals` (`forecast_totals` at each
    origin, with top-up candidate `topups` and the price levels `projected` (`projected_at`)).
    Raises ValueError as `forecast_totals` does (e.g. a NaN p x m)."""
    rows = predictions.rows
    customers = rows[CUSTOMER_COLUMNS[:6]].assign(p=predictions.p[stage1],
                                                  m=predictions.m[stage2])
    customers["expected"] = customers["p"] * customers["m"]
    topup_table = predictions.topups[topups]
    totals = [forecast_totals(customers[customers["origin_week"] == origin],
                              topup_table[topup_table["origin_week"] == origin],
                              projected[origin], companies, horizons)
              for origin in sorted(set(rows["origin_week"]))]
    return Forecast(totals=pd.concat(totals, ignore_index=True), customers=customers)


def grid_scores(predictions: GridPredictions, candidates: GridCandidates, gates: GridGates,
                actual: pd.DataFrame, settings, *, companies: list[str], week_zero,
                batch_size: int = SCORE_BATCH,
                on_batch_done: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """The scores of every combination (see the module docstring): one row per combination,
    `GRID_SCORE_COLUMNS`, sorted by WAPE (on a tie, in the order of the candidates, so the
    current settings first). `actual` is `evaluate.actual_totals` at the origins of
    `predictions`; `companies` the settings' company list; `week_zero` the start of week 0,
    from which the price levels are projected.

    company_wape is the mean over the companies that have cash at the grid's origins: a
    company with none has no WAPE (a share of nothing) and is left out of the mean.

    The combinations are forecast and scored `batch_size` at a time, each one whole within
    its batch; the scores do not depend on it. `on_batch_done(done, total)`, if given, is
    called after each batch with the combinations scored so far and in all (e.g. to print
    progress).

    Raises ValueError if `gates` or `predictions` do not have one entry per candidate, if
    `actual` is not for the origins of `predictions` (scored on fewer origins, the grid would
    be judged on other weeks than asked), if `batch_size` is below 1, if there is no
    consolidated cash to score on, and
    as `forecast_totals` (e.g. a NaN p x m of the current settings) and `evaluation_scores`
    do.
    """
    for what, found, listed in (("stage-1 gates", len(gates.stage1), candidates.stage1),
                                ("stage-1 p columns", len(predictions.p.columns),
                                 candidates.stage1),
                                ("stage-2 gates", len(gates.stage2), candidates.stage2),
                                ("stage-2 m columns", len(predictions.m.columns),
                                 candidates.stage2),
                                ("top-up forecasts", len(predictions.topups),
                                 candidates.topups)):
        if found != len(listed):
            raise ValueError(f"there are {found} {what} for {len(listed)} candidates")
    if batch_size < 1:
        raise ValueError(f"batch_size is {batch_size}; at least 1 combination is needed")
    horizons = settings.receivables.horizons
    origins = sorted(set(predictions.rows["origin_week"]))
    actual_origins = sorted(set(actual["origin_week"]))
    if actual_origins != origins:
        raise ValueError(f"the actual cash is for origins {actual_origins}, the predictions "
                         f"for {origins}")
    projected = projected_at(origins, horizons, week_zero)
    stage1 = scored_numbers(gates.stage1, "stage1")
    stage2 = scored_numbers(gates.stage2, "stage2")
    combinations = []
    for i, j, k in itertools.product(stage1, stage2, range(len(candidates.topups))):
        # One subset of the extra blocks for both stages (U18.5).
        if same_extra_blocks(candidates.stage1[i], candidates.stage2[j]):
            combinations.append((i, j, k))

    keys = ["company_code", "origin_week", "h", "target_week"]
    lines = []
    for start in range(0, len(combinations), batch_size):
        batch = combinations[start:start + batch_size]

        # 1. Each combination's totals at each origin, as `forecast` makes them: all of a
        # combination's rows in its own batch.
        parts = []
        for i, j, k in batch:
            totals = combination_forecast(predictions, i, j, k, projected=projected,
                                          companies=companies, horizons=horizons).totals
            part = totals[keys + ["total"]].rename(columns={"total": "forecast"})
            part.insert(0, "method", combination_name(i, j, k))
            parts.append(part)
        forecasts = pd.concat(parts, ignore_index=True)
        forecasts["company_code"] = forecasts["company_code"].astype(object)

        # 2. The batch scored as step 17 scores the model.
        scores = evaluation_scores(forecasts[STACK_COLUMNS], actual, horizons)
        if scores[(scores["company_code"] == CONSOLIDATED)
                  & (scores["horizon"] == POOLED)].empty:
            raise ValueError("there is no consolidated cash at the grid's origins to score on")

        # 3. One line per combination; the batch's forecasts and scores are then dropped.
        for i, j, k in batch:
            mine = scores[scores["method"] == combination_name(i, j, k)]
            consolidated = mine[mine["company_code"] == CONSOLIDATED].set_index("horizon")
            by_company = mine[(mine["company_code"] != CONSOLIDATED)
                              & (mine["horizon"] == POOLED)]
            lines.append({"stage1": i, "stage2": j, "topups": k,
                          "extra_blocks": candidates.stage1[i].params["extra_blocks"],
                          "current": (i, j, k) == (0, 0, 0),
                          "wape": consolidated.loc[POOLED, "wape"],
                          "total_wape": consolidated["wape"].get(HORIZON_TOTAL, float("nan")),
                          "bias": consolidated.loc[POOLED, "bias"],
                          "company_wape": by_company["wape"].mean()})
        if on_batch_done is not None:
            on_batch_done(len(lines), len(combinations))
    table = pd.DataFrame(lines, columns=GRID_SCORE_COLUMNS)
    # "stable" keeps the candidates' order on a tie.
    return table.sort_values("wape", kind="stable").reset_index(drop=True)


def choose_combination(scores: pd.DataFrame, min_gain: float) -> pd.Series:
    """The row of `scores` (`grid_scores`) to keep: the current settings, unless a combination's
    WAPE is lower than theirs by more than `min_gain`; then the lowest WAPE (on a tie, the
    first row, the earlier candidates).

    So a new combination must beat the current settings by more than the noise of choosing
    among many. Raises ValueError if there is not one current row or a WAPE is NaN (it could
    not be compared).
    """
    current = scores[scores["current"]]
    if len(current) != 1:
        raise ValueError(f"the scores hold {len(current)} rows of the current settings; one "
                         f"is needed")
    if scores["wape"].isna().any():
        missing = scores.loc[scores["wape"].isna(), ["stage1", "stage2", "topups"]]
        raise ValueError(f"no WAPE for the combinations {missing.values.tolist()}: is there no "
                         f"consolidated cash to judge on?")
    better = scores[scores["wape"] < current["wape"].iloc[0] - min_gain]
    if better.empty:
        return current.iloc[0]
    return better.sort_values("wape", kind="stable").iloc[0]


# The method names of the comparison on validation (`grid_choice`): the chosen combination is
# `evaluate.HURDLE`; the current settings, if they were not chosen, this.
CURRENT_SETTINGS = "current settings"
# On the v1 test (`grid_test`) the chosen combination with its corrections is `evaluate.HURDLE`;
# the same forecast without them, shown for information only, this.
UNCORRECTED = "hurdle uncorrected"


def method_rows(forecast: Forecast, method: str) -> pd.DataFrame:
    """The totals of `forecast` as one method's rows of the stacked forecasts
    (`evaluate.STACK_COLUMNS`), named `method`."""
    keys = ["company_code", "origin_week", "h", "target_week"]
    rows = forecast.totals[keys + ["total"]].rename(columns={"total": "forecast"})
    rows.insert(0, "method", method)
    rows["company_code"] = rows["company_code"].astype(object)
    return rows[STACK_COLUMNS]


@dataclass(frozen=True)
class GridChoice:
    """What the grid chose (U16.6), on the validation origins."""
    winner: pd.Series            # its row of `grid_scores`
    stage1: Candidate            # its stage-1 candidate
    stage2: Candidate            # its stage-2 candidate
    topups: object               # its top-up candidate (`TopupSettings`)
    forecast: Forecast           # its forecast at every origin (`combination_forecast`)
    calibration: CalibrationChoice  # `choose_calibration` on that forecast
    scores: pd.DataFrame         # `evaluation_scores` of it, the current settings (if not
                                 # chosen), the `deflated` variants and the rule
    bootstrap: pd.DataFrame      # `paired_wape_bootstrap`: it against each of the others


def grid_choice(predictions: GridPredictions, candidates: GridCandidates, scores: pd.DataFrame,
                actual: pd.DataFrame, settings, *, companies: list[str], week_zero,
                weekly: pd.DataFrame, weeks: pd.DataFrame, levels: pd.Series,
                calendar: pd.DataFrame, ledger: pd.DataFrame, rates: pd.DataFrame,
                new_cash: pd.DataFrame, refunds: pd.DataFrame,
                unidentified: pd.DataFrame) -> GridChoice:
    """The grid's choice on the validation origins of `predictions` (U16.6):

      1. the combination kept (`choose_combination` on `scores`, `grid_scores`' table, with
         [grid] min_gain) and its forecast (`combination_forecast`);
      2. its corrections of p and m (`calibrate.choose_calibration`, no new fits);
      3. the references made here at the same origins, so step 17 is not needed: the
         `deflated` variants (`evaluate.deflated_reference`) and the financial rule
         (`evaluate.rule_reference`, with the chosen top-ups, so that it differs from the
         model only in the known customers);
      4. the chosen combination, uncorrected, scored beside the current settings (if not
         chosen) and the references (`evaluation_scores`), and compared with each of them
         (`paired_wape_bootstrap`, [evaluation] samples and seed). Uncorrected: the kept
         corrections are learned on these very weeks, so the corrected forecast would be
         flattered here; its fair comparison is the v1 test (U16.7).

    `actual` is `evaluate.actual_totals` at the origins of `predictions`; the other arguments
    are those of the steps named. Raises ValueError as they do.
    """
    horizons = settings.receivables.horizons
    origins = sorted(set(predictions.rows["origin_week"]))
    projected = projected_at(origins, horizons, week_zero)

    # 1. The combination kept, and its forecast.
    winner = choose_combination(scores, settings.grid.min_gain)
    i, j, k = int(winner["stage1"]), int(winner["stage2"]), int(winner["topups"])
    forecast = combination_forecast(predictions, i, j, k, projected=projected,
                                    companies=companies, horizons=horizons)

    # 2. Its corrections.
    calibration = choose_calibration(forecast, weekly, weeks, levels, actual, settings)

    # 3. The references at the same origins.
    reference = deflated_reference(weekly, weeks, levels, calendar, companies, origins,
                                   horizons, week_zero)
    with_chosen_topups = dataclasses.replace(settings, topups=candidates.topups[k])
    rule = rule_reference(ledger, rates, origins, with_chosen_topups, companies=companies,
                          new_cash=new_cash, refunds=refunds, unidentified=unidentified,
                          levels=levels, calendar=calendar, week_zero=week_zero)

    # 4. Everything scored together, and the chosen one against each of the others.
    forecasts = stack_forecasts(forecast, reference, rule)
    if not winner["current"]:
        current = combination_forecast(predictions, 0, 0, 0, projected=projected,
                                       companies=companies, horizons=horizons)
        forecasts = pd.concat([forecasts, method_rows(current, CURRENT_SETTINGS)],
                              ignore_index=True)
    method_scores = evaluation_scores(forecasts, actual, horizons)
    pairs = [(HURDLE, other) for other in dict.fromkeys(forecasts["method"]) if other != HURDLE]
    bootstrap = paired_wape_bootstrap(forecasts, actual, pairs, horizons,
                                      settings.evaluation.bootstrap_samples,
                                      settings.evaluation.bootstrap_seed)

    return GridChoice(winner=winner, stage1=candidates.stage1[i], stage2=candidates.stage2[j],
                      topups=candidates.topups[k], forecast=forecast, calibration=calibration,
                      scores=method_scores, bootstrap=bootstrap)


@dataclass(frozen=True)
class GridTest:
    """The v1 test (U16.7): the grid's choice on the test origins, with nothing chosen there."""
    forecast: Forecast           # the chosen combination with its corrections, every test origin
    uncorrected: Forecast        # the same without the corrections (information only)
    scores: pd.DataFrame         # `evaluation_scores` of both, the `deflated` variants and the
                                 # rule
    bootstrap: pd.DataFrame      # `paired_wape_bootstrap`: the corrected one against each other


def grid_test(choice: GridChoice, actual: pd.DataFrame, settings, *, table: pd.DataFrame,
              block_columns: dict[str, list[str]], companies: list[str], week_zero,
              weekly: pd.DataFrame, weeks: pd.DataFrame, levels: pd.Series,
              calendar: pd.DataFrame, ledger: pd.DataFrame, rates: pd.DataFrame,
              new_cash: pd.DataFrame, refunds: pd.DataFrame, unidentified: pd.DataFrame,
              cache_folder: Path,
              on_fit_done: Callable[[int, int, str, int, int, float], None] | None = None
              ) -> GridTest:
    """The v1 test (U16.7) on the [evaluation] test origins: what `choice` (`grid_choice`)
    froze on validation, judged on weeks it never saw. A report: nothing is chosen from it.

      1. the chosen stage-1, stage-2 and top-up settings fitted at each test origin on the rows
         known there (`grid_predictions` with those three alone, each fit cached in
         `cache_folder`; `on_fit_done` as there) and forecast (`combination_forecast`);
      2. its corrections of p and m, as learned on validation, applied unchanged
         (`calibrate.corrected_forecast`);
      3. the references at the test origins: the `deflated` variants and the financial rule
         with the chosen top-ups, as in `grid_choice`;
      4. the corrected forecast (`evaluate.HURDLE`), the uncorrected one (`UNCORRECTED`) and the
         references scored together (`evaluation_scores`), and the corrected one compared with
         each of the others (`paired_wape_bootstrap`, [evaluation] samples and seed).

    `actual` is `evaluate.actual_totals` at the test origins; the other arguments are those of
    `grid_predictions` and `grid_choice`. Raises ValueError if a test target week is not after
    every validation target week of `choice` (validation would have seen the test), if
    `actual` is not for the test origins, and as the steps named do.
    """
    horizons = settings.receivables.horizons
    origins = sorted(set(settings.evaluation.test_origins))
    validation = sorted(set(choice.forecast.customers["origin_week"]))
    if validation[-1] + horizons > origins[0]:
        raise ValueError(f"the last validation origin {validation[-1]} forecasts up to week "
                         f"{validation[-1] + horizons}, which is not before the first test "
                         f"target week {origins[0] + 1}")
    actual_origins = sorted(set(actual["origin_week"]))
    if actual_origins != origins:
        raise ValueError(f"the actual cash is for origins {actual_origins}, the test origins are "
                         f"{origins}")

    # 1. The chosen settings fitted at every test origin, and their forecast.
    chosen = GridCandidates(stage1=[choice.stage1], stage2=[choice.stage2],
                            topups=[choice.topups])
    predictions = grid_predictions(table, origins, chosen, settings,
                                   block_columns=block_columns, companies=companies,
                                   new_cash=new_cash, refunds=refunds,
                                   unidentified=unidentified, levels=levels, calendar=calendar,
                                   cache_folder=cache_folder, on_fit_done=on_fit_done)
    uncorrected = combination_forecast(predictions, 0, 0, 0,
                                       projected=projected_at(origins, horizons, week_zero),
                                       companies=companies, horizons=horizons)

    # 2. The corrections learned on validation, applied unchanged.
    calibration = choice.calibration
    forecast = corrected_forecast(uncorrected, calibration.p_correction,
                                  calibration.m_correction)

    # 3. The references at the test origins.
    reference = deflated_reference(weekly, weeks, levels, calendar, companies, origins,
                                   horizons, week_zero)
    with_chosen_topups = dataclasses.replace(settings, topups=choice.topups)
    rule = rule_reference(ledger, rates, origins, with_chosen_topups, companies=companies,
                          new_cash=new_cash, refunds=refunds, unidentified=unidentified,
                          levels=levels, calendar=calendar, week_zero=week_zero)

    # 4. Everything scored together, and the corrected forecast against each of the others.
    forecasts = pd.concat([stack_forecasts(forecast, reference, rule),
                           method_rows(uncorrected, UNCORRECTED)], ignore_index=True)
    method_scores = evaluation_scores(forecasts, actual, horizons)
    pairs = [(HURDLE, other) for other in dict.fromkeys(forecasts["method"]) if other != HURDLE]
    bootstrap = paired_wape_bootstrap(forecasts, actual, pairs, horizons,
                                      settings.evaluation.bootstrap_samples,
                                      settings.evaluation.bootstrap_seed)
    return GridTest(forecast=forecast, uncorrected=uncorrected, scores=method_scores,
                    bootstrap=bootstrap)
