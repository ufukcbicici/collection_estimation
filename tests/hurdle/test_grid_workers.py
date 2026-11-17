"""Tests for [grid] workers: the setting (`read_grid` with `count_setting`), `fit_on_worker`, and
`grid_predictions` with one worker (here, in order) and with several (loky worker processes).

Most `grid_predictions` tests with workers above 1 replace `grid.Parallel` with an in-process
stand-in that runs the fits in REVERSED order, so the "results come back in finishing order"
logic is tested deterministically and without starting processes. A monkeypatch does not reach
a loky worker process, so the few tests with real worker processes (`real_runs`) use real fits
on test_forecast.py's small table and compare against one worker.
"""
import contextlib
import dataclasses
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import threadpoolctl

from collection_estimation.hurdle import grid as grid_module
from collection_estimation.hurdle.diagnostic import Candidate
from collection_estimation.hurdle.forecast import stage1_predictions, stage2_predictions
from collection_estimation.hurdle.grid import (
    GridCandidates,
    cached_fit,
    fit_on_worker,
    grid_predictions,
    setting_hash,
)
from collection_estimation.hurdle.settings import GridSettings, load_settings, read_grid
from collection_estimation.hurdle.training_table import rows_at_origin, rows_known_at
from tests.hurdle.test_forecast import BLOCK_COLUMNS, F_COMPANIES, make_table
from tests.hurdle.test_grid import (
    G_ORIGINS,
    cache_files,
    count_fits,
    g_candidates,
    g_inputs,
    g_settings,
)
from tests.hurdle.test_settings import (
    FILE_NAME,
    GRID_SECTION,
    GRID_SECTION_KEYS,
    REMOVE,
    REPO_ROOT,
    SYNTHETIC_SETTINGS,
    grid_table,
    write_settings,
)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    return root


def load(tmp_path, repo, changes=None):
    return load_settings(write_settings(tmp_path, changes=changes), repo)


def grid_error(tmp_path, repo, changes):
    with pytest.raises(ValueError) as error:
        load(tmp_path, repo, {("grid", key): value for key, value in changes.items()})
    return str(error.value)


def read_grid_error(**changes):
    with pytest.raises(ValueError) as error:
        read_grid(grid_table(**changes), "here")
    return str(error.value)


# =============================================================================================
# The setting
# =============================================================================================

def test_workers_is_the_last_grid_field_and_a_required_key():
    names = [f.name for f in dataclasses.fields(GridSettings)]
    assert names[-1] == "workers"
    assert "workers" in GRID_SECTION_KEYS


def test_workers_has_no_default():
    field = {f.name: f for f in dataclasses.fields(GridSettings)}["workers"]
    assert field.default is dataclasses.MISSING
    assert field.default_factory is dataclasses.MISSING


@pytest.mark.parametrize("text, expected", [("1", 1), ("3", 3), ("64", 64)])
def test_workers_good_values_are_read_as_int(tmp_path, repo, text, expected):
    grid = load(tmp_path, repo, {("grid", "workers"): text}).grid
    assert grid.workers == expected
    assert type(grid.workers) is int


@pytest.mark.parametrize("text, shown", [
    ("0", "0"), ("-1", "-1"), ("2.5", "2.5"), ("3.0", "3.0"), ('"3"', "'3'"),
    ("true", "True"), ("false", "False"), ("[3]", "[3]"),
], ids=["zero", "negative", "fraction", "integral-float", "text", "true", "false", "list"])
def test_a_bad_workers_has_the_exact_message(tmp_path, repo, text, shown):
    assert grid_error(tmp_path, repo, {"workers": text}) == (
        f"{GRID_SECTION}: workers is {shown}, not a whole number of 1 or more")


def test_a_missing_workers_has_the_exact_message(tmp_path, repo):
    message = grid_error(tmp_path, repo, {"workers": REMOVE})
    assert message == (f"{GRID_SECTION}: missing ['workers']; expected exactly "
                       f"{GRID_SECTION_KEYS}")


def test_a_workers_typo_names_the_missing_and_the_unknown_key(tmp_path, repo):
    message = grid_error(tmp_path, repo, {"workers": REMOVE, "worker": "3"})
    assert message == (f"{GRID_SECTION}: missing ['workers']; unknown ['worker']; "
                       f"expected exactly {GRID_SECTION_KEYS}")


def test_workers_in_another_section_is_unknown_there(tmp_path, repo):
    with pytest.raises(ValueError) as error:
        load(tmp_path, repo, {("stage1", "workers"): "3"})
    assert f"{FILE_NAME} [stage1]" in str(error.value)
    assert "unknown ['workers']" in str(error.value)


def test_a_bad_workers_is_refused_with_the_grid_switched_off(tmp_path, repo):
    message = grid_error(tmp_path, repo, {"run": "false", "run_test": "false", "workers": "0"})
    assert message == f"{GRID_SECTION}: workers is 0, not a whole number of 1 or more"


def test_workers_is_checked_after_every_other_grid_field(tmp_path, repo):
    # Field order: a bad min_gain, then a bad extra_blocks, are reported before a bad workers.
    assert grid_error(tmp_path, repo, {"min_gain": "0", "workers": "0"}).startswith(
        f"{GRID_SECTION}: min_gain")
    assert grid_error(tmp_path, repo, {"extra_blocks": '["x"]', "workers": "0"}).startswith(
        f"{GRID_SECTION}: extra_blocks")
    assert grid_error(tmp_path, repo, {"run": "false", "workers": "0"}).startswith(
        f"{GRID_SECTION}: run_test = true needs run = true")


def test_a_bad_workers_wins_over_the_extra_blocks_overlap_error(tmp_path, repo):
    # The overlap check runs once Settings is built, so a [grid] field error comes first.
    message = grid_error(tmp_path, repo, {"extra_blocks": '["history"]', "workers": "0"})
    assert message == f"{GRID_SECTION}: workers is 0, not a whole number of 1 or more"


def test_workers_does_not_change_the_other_grid_fields(tmp_path, repo):
    (tmp_path / "a").mkdir()
    one = load(tmp_path / "a", repo, {("grid", "workers"): "1"})
    five = load(tmp_path, repo, {("grid", "workers"): "5"})
    assert dataclasses.replace(one.grid, workers=5) == five.grid
    assert one.stage1 == five.stage1 and one.stage2 == five.stage2


@pytest.mark.parametrize("value", [1, 2, 12])
def test_read_grid_reads_workers(value):
    assert read_grid(grid_table(workers=value), "here").workers == value


@pytest.mark.parametrize("bad", [0, -3, 2.5, 3.0, "3", True, False, None])
def test_read_grid_refuses_a_bad_workers_naming_where(bad):
    assert read_grid_error(workers=bad) == (
        f"here: workers is {bad!r}, not a whole number of 1 or more")


def test_read_grid_without_workers_raises_key_error():
    table = grid_table()
    del table["workers"]
    with pytest.raises(KeyError):
        read_grid(table, "here")


def test_read_grid_checks_extra_blocks_before_workers():
    assert read_grid_error(extra_blocks="x", workers=0).startswith("here: extra_blocks")


@pytest.mark.skipif(not SYNTHETIC_SETTINGS.exists(), reason="settings/synthetic.toml absent")
def test_the_synthetic_toml_has_a_whole_number_of_workers():
    # Only the type is pinned; the value is the owner's to change.
    workers = load_settings(SYNTHETIC_SETTINGS, REPO_ROOT).grid.workers
    assert type(workers) is int and workers >= 1


# =============================================================================================
# fit_on_worker
# =============================================================================================

ORIGIN = 18


@pytest.fixture(scope="module")
def w_table():
    return make_table()


@pytest.fixture(scope="module")
def w_rows(w_table):
    return rows_known_at(w_table, ORIGIN), rows_at_origin(w_table, ORIGIN)


def worker_call(rows, stage, setting, cache_folder, number=0, company_as_feature=True,
                inputs=None, threads=None):
    train, at = rows
    return fit_on_worker(stage, number, setting, ORIGIN, train, at, BLOCK_COLUMNS,
                         list(F_COMPANIES), company_as_feature,
                         ["in-a", 1] if inputs is None else inputs, cache_folder, threads)


def record_stages(monkeypatch, rows):
    """Replace both stage functions in grid with recorders returning a marked Series."""
    calls = []
    _, at = rows

    def stage1(*args):
        calls.append(("stage1", args))
        return pd.Series(0.25, index=at.index)

    def stage2(*args):
        calls.append(("stage2", args))
        return pd.Series(7.0, index=at.index)
    monkeypatch.setattr(grid_module, "stage1_predictions", stage1)
    monkeypatch.setattr(grid_module, "stage2_predictions", stage2)
    return calls


def test_fit_on_worker_stage1_calls_stage1_predictions_with_the_rows_and_setting(
        w_rows, tmp_path, monkeypatch):
    calls = record_stages(monkeypatch, w_rows)
    settings = g_settings()
    found = worker_call(w_rows, "stage1", settings.stage1, tmp_path, number=4)
    assert [name for name, _ in calls] == ["stage1"]
    train, at, setting, blocks, companies = calls[0][1]
    assert train is w_rows[0] and at is w_rows[1]
    assert setting == settings.stage1 and blocks == BLOCK_COLUMNS
    assert companies == list(F_COMPANIES)
    stage, number, prediction, seconds = found
    assert (stage, number) == ("stage1", 4)
    pd.testing.assert_series_equal(prediction, pd.Series(0.25, index=w_rows[1].index))
    assert isinstance(seconds, float) and seconds >= 0


@pytest.mark.parametrize("company_as_feature", [True, False])
def test_fit_on_worker_stage2_calls_stage2_predictions_with_company_as_feature(
        w_rows, tmp_path, monkeypatch, company_as_feature):
    calls = record_stages(monkeypatch, w_rows)
    settings = g_settings()
    found = worker_call(w_rows, "stage2", settings.stage2, tmp_path, number=2,
                        company_as_feature=company_as_feature)
    assert [name for name, _ in calls] == ["stage2"]
    train, at, setting, blocks, companies, caf = calls[0][1]
    assert train is w_rows[0] and at is w_rows[1]
    assert setting == settings.stage2 and blocks == BLOCK_COLUMNS
    assert companies == list(F_COMPANIES) and caf is company_as_feature
    assert found[:2] == ("stage2", 2)
    pd.testing.assert_series_equal(found[2], pd.Series(7.0, index=w_rows[1].index))


@pytest.mark.parametrize("stage", ["stage3", "Stage1", "stage 1", "", None])
def test_fit_on_worker_refuses_an_unknown_stage_before_anything(w_rows, tmp_path, monkeypatch,
                                                                stage):
    calls = record_stages(monkeypatch, w_rows)
    events = []
    record_limits(monkeypatch, events)
    cached = record_cached_fit(monkeypatch)
    with pytest.raises(ValueError) as error:
        worker_call(w_rows, stage, g_settings().stage1, tmp_path, threads=2)
    assert str(error.value) == f"stage is {stage!r}, not 'stage1' or 'stage2'"
    assert calls == [] and events == [] and cached == []
    assert cache_files(tmp_path) == []


def test_fit_on_worker_an_unknown_stage_writes_no_cache_with_the_real_cached_fit(
        w_rows, tmp_path, monkeypatch):
    calls = record_stages(monkeypatch, w_rows)
    with pytest.raises(ValueError, match="^stage is 'stage3', not 'stage1' or 'stage2'$"):
        worker_call(w_rows, "stage3", g_settings().stage1, tmp_path / "cache")
    assert calls == []
    assert not (tmp_path / "cache").exists()


def test_fit_on_worker_returns_a_four_tuple(w_rows, tmp_path, monkeypatch):
    record_stages(monkeypatch, w_rows)
    found = worker_call(w_rows, "stage1", g_settings().stage1, tmp_path)
    assert type(found) is tuple and len(found) == 4


def test_fit_on_worker_real_fits_equal_the_stage_functions(w_rows, tmp_path):
    settings = g_settings()
    train, at = w_rows
    p = worker_call(w_rows, "stage1", settings.stage1, tmp_path / "1")[2]
    m = worker_call(w_rows, "stage2", settings.stage2, tmp_path / "2")[2]
    pd.testing.assert_series_equal(
        p, stage1_predictions(train, at, settings.stage1, BLOCK_COLUMNS, F_COMPANIES))
    pd.testing.assert_series_equal(
        m, stage2_predictions(train, at, settings.stage2, BLOCK_COLUMNS, F_COMPANIES, True))


def record_cached_fit(monkeypatch):
    calls = []

    def fake(stage, origin, setting, inputs, cache_folder, fit):
        calls.append((stage, origin, setting, list(inputs), cache_folder))
        return fit()
    monkeypatch.setattr(grid_module, "cached_fit", fake)
    return calls


def test_fit_on_worker_stage1_hashes_the_inputs_as_given(w_rows, tmp_path, monkeypatch):
    record_stages(monkeypatch, w_rows)
    calls = record_cached_fit(monkeypatch)
    setting = g_settings().stage1
    worker_call(w_rows, "stage1", setting, tmp_path, inputs=["x", {"y": 1}])
    assert calls == [("stage1", ORIGIN, setting, ["x", {"y": 1}], tmp_path)]


@pytest.mark.parametrize("company_as_feature", [True, False])
def test_fit_on_worker_stage2_hashes_the_inputs_then_company_as_feature(
        w_rows, tmp_path, monkeypatch, company_as_feature):
    record_stages(monkeypatch, w_rows)
    calls = record_cached_fit(monkeypatch)
    setting = g_settings().stage2
    inputs = ["x", {"y": 1}]
    worker_call(w_rows, "stage2", setting, tmp_path, inputs=inputs,
                company_as_feature=company_as_feature)
    assert calls == [("stage2", ORIGIN, setting, ["x", {"y": 1}, company_as_feature], tmp_path)]
    assert inputs == ["x", {"y": 1}]       # the caller's list is not changed


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_fit_on_worker_writes_the_same_cache_files_as_cached_fit_did(w_rows, tmp_path,
                                                                     monkeypatch, stage):
    # The fingerprint file holds [stage, origin, the setting, *inputs] (+ company_as_feature
    # for stage 2), exactly as the per-stage loops wrote it before [grid] workers.
    record_stages(monkeypatch, w_rows)
    setting = getattr(g_settings(), stage)
    inputs = [{"horizons": 5}, "fingerprint", ["a", "b"]]
    worker_call(w_rows, stage, setting, tmp_path, inputs=inputs, company_as_feature=False)
    name = f"{stage}_origin{ORIGIN}_{setting_hash(setting)}"
    assert cache_files(tmp_path) == [f"{name}.pkl", f"{name}_fingerprint.json"]
    saved = json.loads((tmp_path / f"{name}_fingerprint.json").read_text(encoding="utf-8"))
    expected = [stage, ORIGIN, asdict(setting), *inputs] + ([False] if stage == "stage2" else [])
    assert saved == json.loads(json.dumps(expected))


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_fit_on_worker_reads_a_fit_cached_by_cached_fit(w_rows, tmp_path, monkeypatch, stage):
    setting = getattr(g_settings(), stage)
    inputs = ["in"]
    hashed = inputs + ([True] if stage == "stage2" else [])
    stored = pd.Series(np.arange(len(w_rows[1]), dtype=float), index=w_rows[1].index)
    cached_fit(stage, ORIGIN, setting, hashed, tmp_path, lambda: stored)
    calls = record_stages(monkeypatch, w_rows)
    found = worker_call(w_rows, stage, setting, tmp_path, inputs=inputs)
    assert calls == []
    pd.testing.assert_series_equal(found[2], stored)


def test_fit_on_worker_a_raising_fit_propagates_and_caches_nothing(w_rows, tmp_path,
                                                                  monkeypatch):
    def broken(*args):
        raise RuntimeError("fit failed")
    monkeypatch.setattr(grid_module, "stage1_predictions", broken)
    with pytest.raises(RuntimeError, match="fit failed"):
        worker_call(w_rows, "stage1", g_settings().stage1, tmp_path)
    assert cache_files(tmp_path) == []


def record_limits(monkeypatch, events):
    @contextlib.contextmanager
    def limits(threads):
        events.append(("enter", threads))
        yield
        events.append(("exit", threads))
    monkeypatch.setattr(grid_module, "threadpool_limits", limits)


@pytest.mark.parametrize("threads", [None, 1, 4])
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_fit_on_worker_fits_inside_threadpool_limits_of_threads(w_rows, tmp_path, monkeypatch,
                                                                stage, threads):
    events = []
    record_limits(monkeypatch, events)
    _, at = w_rows

    def fit(*args):
        events.append("fit")
        return pd.Series(1.0, index=at.index)
    monkeypatch.setattr(grid_module, f"{stage}_predictions", fit)
    worker_call(w_rows, stage, getattr(g_settings(), stage), tmp_path, threads=threads)
    assert events == [("enter", threads), "fit", ("exit", threads)]


def test_fit_on_worker_a_cached_read_is_inside_threadpool_limits_too(w_rows, tmp_path,
                                                                    monkeypatch):
    setting = g_settings().stage1
    record_stages(monkeypatch, w_rows)
    worker_call(w_rows, "stage1", setting, tmp_path)
    events = []
    record_limits(monkeypatch, events)
    worker_call(w_rows, "stage1", setting, tmp_path, threads=2)
    assert events == [("enter", 2), ("exit", 2)]


def test_fit_on_worker_really_limits_the_threads_during_the_fit(w_rows, tmp_path, monkeypatch):
    # The real threadpool_limits: during the fit every thread pool runs on 1 thread.
    seen = []
    _, at = w_rows

    def fit(*args):
        seen.extend(pool["num_threads"] for pool in threadpoolctl.threadpool_info())
        return pd.Series(1.0, index=at.index)
    monkeypatch.setattr(grid_module, "stage1_predictions", fit)
    worker_call(w_rows, "stage1", g_settings().stage1, tmp_path, threads=1)
    if not seen:
        pytest.skip("no thread pool loaded")
    assert set(seen) == {1}


# =============================================================================================
# grid_predictions: workers == 1, and workers > 1 with an in-process stand-in for Parallel
# =============================================================================================

def w_run(table, cache_folder, workers, origins=G_ORIGINS, candidates=None, on_fit_done=None):
    settings = g_settings(workers=workers)
    return grid_predictions(table, origins, candidates or g_candidates(settings), settings,
                            cache_folder=cache_folder, on_fit_done=on_fit_done, **g_inputs())


def fake_fit_on_worker(monkeypatch, calls):
    """Replace grid.fit_on_worker: record its arguments, return a p / m that names the fit
    (100 x origin + number; stage 2 + 0.5), taking no time."""
    def fake(stage, number, setting, origin, train_rows, forecast_rows, block_columns,
             companies, company_as_feature, inputs, cache_folder, threads):
        calls.append(dict(stage=stage, number=number, setting=setting, origin=origin,
                          threads=threads, inputs=inputs, cache_folder=cache_folder,
                          company_as_feature=company_as_feature))
        value = 100 * origin + number + (0.5 if stage == "stage2" else 0.0)
        return stage, number, pd.Series(value, index=forecast_rows.index), 0.0
    monkeypatch.setattr(grid_module, "fit_on_worker", fake)


class ReversedParallel:
    """Stands in for joblib.Parallel: records its options, runs the delayed calls here in
    REVERSED order (as if the last one finished first) and yields each result."""
    made = []

    def __init__(self, **options):
        ReversedParallel.made.append(options)

    def __call__(self, tasks):
        tasks = list(tasks)
        return (function(*args, **kwargs) for function, args, kwargs in reversed(tasks))


@pytest.fixture
def reversed_parallel(monkeypatch):
    ReversedParallel.made = []
    monkeypatch.setattr(grid_module, "Parallel", ReversedParallel)
    return ReversedParallel


@pytest.fixture
def no_parallel(monkeypatch):
    def refuse(**options):
        raise AssertionError("Parallel must not be used with one worker")
    monkeypatch.setattr(grid_module, "Parallel", refuse)


def test_one_worker_never_makes_a_parallel(w_table, tmp_path, no_parallel):
    w_run(w_table, tmp_path, workers=1, origins=[18])


def test_one_worker_passes_threads_none(w_table, tmp_path, monkeypatch, no_parallel):
    calls = []
    fake_fit_on_worker(monkeypatch, calls)
    w_run(w_table, tmp_path, workers=1)
    assert len(calls) == 8 and {c["threads"] for c in calls} == {None}


def test_one_worker_does_not_ask_for_the_cpu_count(w_table, tmp_path, monkeypatch, no_parallel):
    calls = []
    fake_fit_on_worker(monkeypatch, calls)
    monkeypatch.setattr(grid_module.os, "cpu_count", lambda: None)
    w_run(w_table, tmp_path, workers=1)
    assert {c["threads"] for c in calls} == {None}


def test_one_worker_calls_the_fits_stage1_then_stage2_per_origin(w_table, tmp_path, monkeypatch,
                                                                  no_parallel):
    calls = []
    fake_fit_on_worker(monkeypatch, calls)
    settings = g_settings()
    candidates = g_candidates(settings, 3, 2)
    progress = []
    w_run(w_table, tmp_path, workers=1, candidates=candidates,
          on_fit_done=lambda *args: progress.append(args))
    order = [(c["origin"], c["stage"], c["number"]) for c in calls]
    expected = [(o, s, n) for o in G_ORIGINS
                for s, count in (("stage1", 3), ("stage2", 2)) for n in range(count)]
    assert order == expected
    assert [args[:5] for args in progress] == [
        (i + 1, 10, "stage 1" if s == "stage1" else "stage 2", o, n)
        for i, (o, s, n) in enumerate(expected)]
    assert [c["setting"] for c in calls[:5]] == (
        [c.settings for c in candidates.stage1] + [c.settings for c in candidates.stage2])


@pytest.mark.parametrize("cpus, workers, threads", [(12, 3, 4), (13, 2, 6), (12, 5, 2),
                                                    (2, 3, 1), (1, 2, 1), (4, 4, 1)])
def test_several_workers_get_cpu_count_over_workers_threads_at_least_one(
        w_table, tmp_path, monkeypatch, reversed_parallel, cpus, workers, threads):
    calls = []
    fake_fit_on_worker(monkeypatch, calls)
    monkeypatch.setattr(grid_module.os, "cpu_count", lambda: cpus)
    w_run(w_table, tmp_path, workers=workers)
    assert len(calls) == 8 and {c["threads"] for c in calls} == {threads}


@pytest.mark.parametrize("workers", [2, 3])
def test_several_workers_with_an_unknown_cpu_count_get_one_thread_each_and_run(
        w_table, tmp_path, monkeypatch, reversed_parallel, workers):
    # os.cpu_count() is None when the machine cannot tell: 1 thread per fit, no TypeError.
    calls = []
    fake_fit_on_worker(monkeypatch, calls)
    monkeypatch.setattr(grid_module.os, "cpu_count", lambda: None)
    result = w_run(w_table, tmp_path, workers=workers)
    assert len(calls) == 8 and {c["threads"] for c in calls} == {1}
    assert list(result.p.columns) == [0, 1] and list(result.m.columns) == [0, 1]


def test_several_workers_make_one_loky_unordered_parallel_per_origin(
        w_table, tmp_path, monkeypatch, reversed_parallel):
    fake_fit_on_worker(monkeypatch, [])
    w_run(w_table, tmp_path, workers=3)
    assert reversed_parallel.made == [
        dict(n_jobs=3, backend="loky", return_as="generator_unordered")] * len(G_ORIGINS)


def test_several_workers_send_every_fit_with_the_one_worker_arguments(
        w_table, tmp_path, monkeypatch, reversed_parallel):
    one, several = [], []
    fake_fit_on_worker(monkeypatch, one)
    with monkeypatch.context() as patch:
        patch.setattr(grid_module, "Parallel", lambda **o: pytest.fail("no Parallel"))
        w_run(w_table, tmp_path / "a", workers=1)
    fake_fit_on_worker(monkeypatch, several)
    w_run(w_table, tmp_path / "a", workers=2)

    def key(call):
        return (call["origin"], call["stage"], call["number"])
    strip = [{k: v for k, v in c.items() if k != "threads"} for c in one]
    assert sorted(strip, key=key) == sorted(
        [{k: v for k, v in c.items() if k != "threads"} for c in several], key=key)


def test_several_workers_columns_are_in_candidate_order_whatever_the_finishing_order(
        w_table, tmp_path, monkeypatch, reversed_parallel):
    fake_fit_on_worker(monkeypatch, [])
    settings = g_settings()
    candidates = g_candidates(settings, 3, 2)
    result = w_run(w_table, tmp_path, workers=2, candidates=candidates)
    assert list(result.p.columns) == [0, 1, 2]
    assert list(result.m.columns) == [0, 1]
    for origin in G_ORIGINS:
        at = result.rows["origin_week"] == origin
        for number in range(3):
            assert (result.p.loc[at, number] == 100 * origin + number).all()
        for number in range(2):
            assert (result.m.loc[at, number] == 100 * origin + number + 0.5).all()
    assert result.p.index.equals(result.rows.index) and result.m.index.equals(result.rows.index)


def test_several_workers_equal_one_worker_with_the_same_fits(w_table, tmp_path, monkeypatch,
                                                             reversed_parallel):
    fake_fit_on_worker(monkeypatch, [])
    candidates = g_candidates(g_settings(), 3, 2)
    several = w_run(w_table, tmp_path, workers=2, candidates=candidates)
    with monkeypatch.context() as patch:
        patch.setattr(grid_module, "Parallel", lambda **o: pytest.fail("no Parallel"))
        one = w_run(w_table, tmp_path, workers=1, candidates=candidates)
    pd.testing.assert_frame_equal(several.p, one.p)
    pd.testing.assert_frame_equal(several.m, one.m)
    pd.testing.assert_frame_equal(several.rows, one.rows)
    for a, b in zip(several.topups, one.topups):
        pd.testing.assert_frame_equal(a, b)


def test_several_workers_report_each_fit_in_finishing_order_counting_up(
        w_table, tmp_path, monkeypatch, reversed_parallel):
    fake_fit_on_worker(monkeypatch, [])
    candidates = g_candidates(g_settings(), 3, 2)
    progress = []
    w_run(w_table, tmp_path, workers=2, candidates=candidates,
          on_fit_done=lambda *args: progress.append(args))
    # Reversed per origin: stage 2's last first, stage 1's first last.
    per_origin = [("stage 2", 1), ("stage 2", 0), ("stage 1", 2), ("stage 1", 1),
                  ("stage 1", 0)]
    assert [args[:5] for args in progress] == [
        (i + 1, 10, stage, origin, number)
        for i, (origin, (stage, number)) in enumerate(
            [(o, fit) for o in G_ORIGINS for fit in per_origin])]
    assert progress[-1][0] == progress[-1][1] == 10


def test_several_workers_a_fit_that_raises_is_raised_in_the_caller(w_table, tmp_path,
                                                                   monkeypatch,
                                                                   reversed_parallel):
    def broken(*args):
        raise RuntimeError("worker fit failed")
    monkeypatch.setattr(grid_module, "fit_on_worker", broken)
    with pytest.raises(RuntimeError, match="worker fit failed"):
        w_run(w_table, tmp_path, workers=2)


# =============================================================================================
# grid_predictions with real loky worker processes (real fits on the small table)
# =============================================================================================

def bytes_and_times(folder):
    folder = Path(folder)
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(folder.iterdir())}


@pytest.fixture(scope="module")
def real_runs(w_table, tmp_path_factory):
    """One grid at G_ORIGINS with 2 + 2 candidates: once with 1 worker, once with 2 (in
    worker processes), each into its own fresh cache."""
    out = {}
    for workers in (1, 2):
        folder = tmp_path_factory.mktemp(f"workers{workers}")
        progress = []
        result = w_run(w_table, folder, workers=workers,
                       on_fit_done=lambda *args, progress=progress: progress.append(args))
        out[workers] = SimpleNamespace(result=result, progress=progress, folder=folder)
    return out


def test_real_two_workers_give_exactly_the_one_worker_p_and_m(real_runs):
    one, two = real_runs[1].result, real_runs[2].result
    pd.testing.assert_frame_equal(two.p, one.p, check_exact=True)
    pd.testing.assert_frame_equal(two.m, one.m, check_exact=True)
    pd.testing.assert_frame_equal(two.rows, one.rows, check_exact=True)
    assert len(two.topups) == len(one.topups)
    for a, b in zip(two.topups, one.topups):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_real_two_workers_fit_in_other_processes(w_table, tmp_path, monkeypatch):
    # A recorder patched in here is not seen by the worker processes: with a fresh cache every
    # fit still happens, but none of them in this process.
    calls = count_fits(monkeypatch)
    found = w_run(w_table, tmp_path, workers=2, origins=[18])
    assert calls == []
    assert found.p.shape[1] == 2 and found.m.shape[1] == 2
    assert len(cache_files(tmp_path)) == 8


def test_real_two_workers_columns_are_in_candidate_order(real_runs):
    two = real_runs[2].result
    assert list(two.p.columns) == [0, 1] and list(two.m.columns) == [0, 1]
    # The candidates really differ, so a swapped column would show.
    assert not two.p[0].equals(two.p[1]) and not two.m[0].equals(two.m[1])


def test_real_two_workers_report_one_call_per_fit(real_runs):
    one, two = real_runs[1].progress, real_runs[2].progress
    assert len(two) == 8
    assert [args[0] for args in two] == list(range(1, 9))
    assert {args[1] for args in two} == {8}
    assert two[-1][0] == two[-1][1]
    assert {args[2:5] for args in two} == {args[2:5] for args in one}
    assert len({args[2:5] for args in two}) == 8
    assert all(isinstance(args[5], float) and args[5] >= 0 for args in two)
    # The origins are done one after another: all of 18's fits before any of 22's.
    assert [args[3] for args in two] == [18] * 4 + [22] * 4


def test_real_one_worker_reports_stage1_then_stage2_per_origin(real_runs):
    assert [args[:5] for args in real_runs[1].progress] == [
        (1, 8, "stage 1", 18, 0), (2, 8, "stage 1", 18, 1),
        (3, 8, "stage 2", 18, 0), (4, 8, "stage 2", 18, 1),
        (5, 8, "stage 1", 22, 0), (6, 8, "stage 1", 22, 1),
        (7, 8, "stage 2", 22, 0), (8, 8, "stage 2", 22, 1)]


def test_real_two_workers_write_the_same_cache_files(real_runs):
    one, two = real_runs[1].folder, real_runs[2].folder
    assert cache_files(two) == cache_files(one)
    assert len(cache_files(two)) == 16
    for name in cache_files(two):
        if name.endswith("_fingerprint.json"):
            assert (two / name).read_text(encoding="utf-8") == (
                (one / name).read_text(encoding="utf-8")), name


def test_real_two_workers_second_run_reads_its_own_cache(real_runs, w_table, tmp_path):
    folder = tmp_path / "cache"
    shutil.copytree(real_runs[2].folder, folder)
    before = bytes_and_times(folder)
    again = w_run(w_table, folder, workers=2)
    assert bytes_and_times(folder) == before          # nothing refitted or rewritten
    pd.testing.assert_frame_equal(again.p, real_runs[2].result.p, check_exact=True)
    pd.testing.assert_frame_equal(again.m, real_runs[2].result.m, check_exact=True)


def test_real_one_worker_reads_the_two_worker_cache_without_fitting(real_runs, w_table,
                                                                    tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    shutil.copytree(real_runs[2].folder, folder)
    calls = count_fits(monkeypatch, fail=True)
    found = w_run(w_table, folder, workers=1)
    assert calls == []
    pd.testing.assert_frame_equal(found.p, real_runs[1].result.p, check_exact=True)
    pd.testing.assert_frame_equal(found.m, real_runs[1].result.m, check_exact=True)


def test_real_a_fit_raising_in_a_worker_is_raised_in_the_caller(w_table, tmp_path):
    # max_iter 0 is refused by scikit-learn inside the fit, so in the worker process.
    settings = g_settings()
    s1 = settings.stage1
    bad = dataclasses.replace(s1, max_iter=0)
    candidates = GridCandidates(
        stage1=[Candidate(params={"max_iter": s1.max_iter, "extra_blocks": "none"}, settings=s1,
                          class_weight="none"),
                Candidate(params={"max_iter": 0, "extra_blocks": "none"}, settings=bad,
                          class_weight="none")],
        stage2=g_candidates(settings, 1, 1).stage2, topups=[settings.topups])
    with pytest.raises(ValueError, match="max_iter"):
        w_run(w_table, tmp_path, workers=2, origins=[18], candidates=candidates)
    # The broken fit leaves no cache of its own.
    assert not any(setting_hash(bad) in name for name in cache_files(tmp_path))


def test_real_the_same_bad_fit_raises_the_same_with_one_worker(w_table, tmp_path):
    settings = g_settings()
    bad = dataclasses.replace(settings.stage1, max_iter=0)
    candidates = GridCandidates(
        stage1=[Candidate(params={"max_iter": 0, "extra_blocks": "none"}, settings=bad,
                          class_weight="none")],
        stage2=g_candidates(settings, 1, 1).stage2, topups=[settings.topups])
    with pytest.raises(ValueError, match="max_iter"):
        w_run(w_table, tmp_path, workers=1, origins=[18], candidates=candidates)
