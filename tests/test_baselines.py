"""Tests for the target series, the metrics and the baseline table.

The load-bearing test is `test_reproduces_the_design_table`. It is not really a test of
this module — it checks that the corpus copy, the ingest, the week spine, the aggregation
and the calendar block all line up with the generator that produced the published figures.
If it fails, something upstream is wrong.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from collection_estimation import config                            # noqa: E402
from collection_estimation.baselines import (                       # noqa: E402
    bias,
    mape,
    mape_table,
    r2,
    rolling_origin_baselines,
    score,
    wape,
    weekly_totals,
    weeks_affected_by_missing_files,
)
from collection_estimation.calendar_features import (               # noqa: E402
    build_calendar_features,
)
from collection_estimation.ingest import read_corpus                # noqa: E402
from collection_estimation.panel import (                           # noqa: E402
    build_week_index,
    to_customer_weeks,
)


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return read_corpus(config.CORPUS_DIR)


@pytest.fixture(scope="module")
def weeks(corpus) -> pd.DataFrame:
    return build_week_index(corpus)


@pytest.fixture(scope="module")
def series(corpus, weeks) -> pd.Series:
    return weekly_totals(to_customer_weeks(corpus, weeks), weeks)


@pytest.fixture(scope="module")
def calendar(weeks) -> pd.DataFrame:
    return build_calendar_features(weeks)


@pytest.fixture(scope="module")
def evaluations(series, calendar) -> pd.DataFrame:
    return rolling_origin_baselines(series, calendar)


#: The v1 synthetic corpus (generator seed 20260908), archived byte-identical when the
#: installed corpus was regenerated as v2. Some findings below were measured on it and did
#: not survive regeneration; they are pinned here so they stay reproducible as a record
#: rather than silently re-measured on whatever corpus happens to be installed.
V1_CORPUS_DIR = os.path.join(config.RAW, "_archive", "v1_seed20260908",
                             "SYNTHETIC_ey_tahsilat_20250602_20260828")


@pytest.fixture(scope="module")
def v1_evaluations() -> pd.DataFrame:
    """The same chain as `evaluations`, run on the archived v1 corpus."""
    if not os.path.isdir(V1_CORPUS_DIR):
        pytest.skip(f"v1 corpus archive not found at {V1_CORPUS_DIR}")
    v1 = read_corpus(V1_CORPUS_DIR)
    v1_weeks = build_week_index(v1)
    v1_series = weekly_totals(to_customer_weeks(v1, v1_weeks), v1_weeks)
    return rolling_origin_baselines(v1_series, build_calendar_features(v1_weeks))


# ---------------------------------------------------------------- the target series

def test_series_covers_every_week(series, weeks):
    assert len(series) == len(weeks)
    assert list(series.index) == list(weeks["week"])


def test_series_is_the_net_total(series, corpus):
    """Net, not positives only — the design's table was computed this way."""
    assert abs(series.sum() - corpus["amount_local"].sum()) < 0.01


def test_series_differs_from_a_positives_only_total(corpus, weeks):
    """If these were identical the choice would not matter. They are not."""
    cw = to_customer_weeks(corpus, weeks)
    net = weekly_totals(cw, weeks)
    positives = cw.groupby("week", observed=True)["amount_positive"].sum().reindex(
        weeks["week"], fill_value=0.0)
    assert not np.allclose(net.to_numpy(), positives.to_numpy())
    assert net.sum() < positives.sum(), "refunds must reduce the net"


# ---------------------------------------------------------------- metrics

def test_perfect_prediction_scores_zero():
    y = np.array([100.0, 200.0, 300.0])
    assert mape(y, y) == 0.0
    assert wape(y, y) == 0.0
    assert bias(y, y) == 0.0


def test_bias_keeps_its_sign():
    y = np.array([100.0, 100.0])
    assert bias(y, y + 10) == 10.0
    assert bias(y, y - 10) == -10.0


def test_mape_and_wape_hide_the_sign_that_bias_keeps():
    """Why all three are reported together."""
    y = np.array([100.0, 100.0])
    high, low = y + 10, y - 10
    assert mape(y, high) == mape(y, low)
    assert wape(y, high) == wape(y, low)
    assert bias(y, high) == -bias(y, low)


def test_mape_is_asymmetric_across_week_sizes():
    """The same absolute error costs more in a light week — which rewards under-predicting.

    This is the property that makes MAPE alone misleading, and the reason WAPE is reported
    beside it.
    """
    actual = np.array([10.0, 1000.0])
    error_on_light = np.array([20.0, 1000.0])     # +10 on the light week
    error_on_heavy = np.array([10.0, 1010.0])     # +10 on the heavy week
    assert mape(actual, error_on_light) > mape(actual, error_on_heavy)
    # WAPE has one denominator, so it charges the same for both.
    assert wape(actual, error_on_light) == pytest.approx(
        wape(actual, error_on_heavy))


def test_zero_actuals_are_skipped_by_mape_not_infinite():
    actual = np.array([0.0, 100.0])
    assert np.isfinite(mape(actual, np.array([5.0, 110.0])))


# ---------------------------------------------------------------- R², the fourth metric

def test_r2_of_a_perfect_prediction_is_one():
    y = np.array([100.0, 200.0, 300.0])
    assert r2(y, y) == 1.0


def test_r2_of_the_actuals_own_mean_is_zero():
    """Zero means "no better than a flat line through the weeks being forecast".

    That line is HINDSIGHT — its height is the mean of the very weeks being predicted,
    which nobody could have known at the origin. So zero is the anchor the scale is read
    against, not a bar any deployable model was asked to clear. The deployable bars are
    the baseline models.
    """
    y = np.array([100.0, 200.0, 600.0])
    flat = np.full(len(y), y.mean())
    assert r2(y, flat) == pytest.approx(0.0)


def test_r2_is_nan_when_the_actuals_are_constant():
    """SST == 0, so the ratio is undefined and NaN says so.

    Note that even a PERFECT prediction of a constant series comes back NaN rather than
    1.0. Nothing was explained because there was no variance to explain — and a 1.0 here
    would read as a flawless model down a results column.
    """
    flat = np.array([100.0, 100.0, 100.0])
    assert np.isnan(r2(flat, flat))
    assert np.isnan(r2(flat, np.array([90.0, 110.0, 100.0])))


def test_r2_can_be_negative_and_that_is_an_ordinary_result():
    """Negative says "worse than that flat hindsight line", not "broken".

    Here the model nails four weeks out of five and misses one heavy week completely —
    which on a series whose variance sits almost entirely in that week is enough to land
    below zero while still scoring a perfectly respectable 16% MAPE. Anyone reading a
    negative R² as a bug would be throwing away a usable model.

        SST = 4·80² + 320² = 128_000      SSE = 400² = 160_000      1 − 1.25 = −0.25
    """
    actual = np.array([100.0, 100.0, 100.0, 100.0, 500.0])
    quiet_level = np.full(5, 100.0)
    assert r2(actual, quiet_level) == pytest.approx(-0.25)
    assert mape(actual, quiet_level) == pytest.approx(16.0)


def test_r2_keeps_the_zero_weeks_that_mape_drops():
    """`mape` filters ``actual != 0``; `r2` does not, and the two answers differ.

    A week with no collections has no percentage error, so MAPE is blind to whatever was
    predicted for it. It is still part of the variance, so R² charges for it.

        SST = (0−100)² + 0 + (200−100)² = 20_000       SSE = 50² = 2_500
    """
    actual = np.array([0.0, 100.0, 200.0])
    exact = np.array([0.0, 100.0, 200.0])
    wrong_on_the_zero_week = np.array([50.0, 100.0, 200.0])

    assert mape(actual, exact) == mape(actual, wrong_on_the_zero_week) == 0.0
    assert r2(actual, exact) == 1.0
    assert r2(actual, wrong_on_the_zero_week) == pytest.approx(0.875)


def test_mape_and_r2_can_rank_two_models_the_opposite_way():
    """The disagreement that justifies reporting a squared-error metric at all.

    Four quiet weeks of 100 and one heavy week of 500. Two models:

        steady_low        10% low every week          absolute error 90 in total
        misses_the_peak   exact on the quiet weeks,
                          90 short on the heavy one   absolute error 90 in total

    The two carry the SAME total absolute error, so WAPE cannot separate them at all —
    which isolates the cause of the disagreement below as the squaring, not a change of
    denominator.

    MAPE prefers `misses_the_peak` (3.6% against 10%) because its single miss lands on
    the largest denominator in the set. R² prefers `steady_low` (0.977 against 0.937)
    because squaring makes one 90 TRY miss cost more than five small ones. The model that
    "wins" depends entirely on which metric is quoted, and a cash plan cares about the
    heavy week.

        SST = 128_000       SSE = 2_900 (steady_low)  against  8_100 (misses_the_peak)
    """
    actual = np.array([100.0, 100.0, 100.0, 100.0, 500.0])
    steady_low = np.array([90.0, 90.0, 90.0, 90.0, 450.0])
    misses_the_peak = np.array([100.0, 100.0, 100.0, 100.0, 410.0])

    assert wape(actual, steady_low) == pytest.approx(wape(actual, misses_the_peak))

    assert mape(actual, steady_low) == pytest.approx(10.0)
    assert mape(actual, misses_the_peak) == pytest.approx(3.6)
    assert r2(actual, steady_low) == pytest.approx(0.97734375)
    assert r2(actual, misses_the_peak) == pytest.approx(0.93671875)

    assert mape(actual, misses_the_peak) < mape(actual, steady_low)
    assert r2(actual, misses_the_peak) < r2(actual, steady_low)


# ---------------------------------------------------------------- the harness

def test_every_origin_only_uses_its_own_past(series, calendar):
    """The leakage check: a change to a LATE week must not move an EARLY prediction."""
    baseline = rolling_origin_baselines(series, calendar)

    tampered = series.copy()
    tampered.iloc[-1] = tampered.iloc[-1] * 100
    after = rolling_origin_baselines(tampered, calendar)

    early = baseline[baseline["target_week"] < len(series) - 1]
    early_after = after[after["target_week"] < len(series) - 1]
    merged = early.merge(early_after, on=["model", "horizon", "origin"],
                         suffixes=("", "_after"))
    assert np.allclose(merged["predicted"], merged["predicted_after"]), (
        "predictions for earlier weeks changed when a later week was altered")


def test_target_is_always_origin_plus_horizon(evaluations):
    assert (evaluations["target_week"]
            == evaluations["origin"] + evaluations["horizon"]).all()


def test_all_models_and_horizons_are_evaluated(evaluations):
    assert set(evaluations["model"]) == {
        "naive", "ma4", "ma8", "calendar_aware", "calendar_only"}
    assert set(evaluations["horizon"]) == {1, 2, 3, 4, 5}


def test_naive_predicts_the_previous_observed_week(series, calendar):
    got = rolling_origin_baselines(series, calendar)
    naive = got[got["model"] == "naive"]
    expected = series.to_numpy()[naive["origin"].to_numpy()]
    assert np.allclose(naive["predicted"], expected)


# ---------------------------------------------------------------- the bar

def test_reproduces_the_generators_benchmark(evaluations):
    """The corpus generator runs its own benchmark. Ours must agree with it.

    This is a check on the whole chain rather than on this module: ingest, week spine,
    customer-week aggregation and the calendar block all have to line up with the
    generator for these to match.

    **The expected figures are read from the installed corpus's oracle**, not written
    here. They belong to one corpus: regenerate with a different seed, or change anything
    upstream of `missing = rng_py.sample(...)` in the generator, and they move. An earlier
    version hardcoded the v1 corpus's table (seed 20260908: naive 28.4, ma4 25.4, ma8
    23.5, calendar_aware 20.2) and broke on the first regeneration without anything in
    the chain being wrong. Reading the oracle keeps the test about the CHAIN — which is
    what it is for — and lets the corpus change underneath it.

    Only h=1, and a 1.5-point tolerance: the generator computes its benchmark with its own
    code, not this module's, so agreement to the decimal is not expected and not the
    claim. Agreement to within a point and a half is.
    """
    import json
    with open(config.ORACLE, encoding="utf-8") as fh:
        benchmark = json.load(fh)["forecasting"]["benchmark"]["mape"]

    table = mape_table(evaluations)
    for model in ("naive", "ma4", "ma8", "calendar_aware"):
        expected = benchmark[model]["1"]            # JSON keys: horizons are strings
        got = table.loc[model, 1]
        assert abs(got - expected) < 1.5, (
            f"{model} at h=1: got {got:.1f}, the generator's oracle says {expected}. "
            f"Something in the chain from workbooks to weekly series disagrees with the "
            f"generator that wrote {config.ORACLE}.")


def test_calendar_wins_clearly_at_short_horizons(evaluations):
    """The calendar advantage is real but it is a SHORT-HORIZON advantage.

    At h=1 and h=2 both calendar models beat the 8-week average by about three points.
    Beyond that the margin closes, and at h=5 the moving average is ahead of both.

    The design originally claimed the calendar wins at every horizon. It does not — that
    claim came from a calendar model whose level term was mis-specified for h>1 and
    happened to score well by luck. See the module docstring in `baselines.py`.
    """
    table = mape_table(evaluations)
    for horizon in (1, 2):
        for model in ("calendar_aware", "calendar_only"):
            assert table.loc[model, horizon] < table.loc["ma8", horizon] - 2.0, (
                f"{model} should beat ma8 clearly at h={horizon}")


def test_no_model_dominates_at_every_horizon(v1_evaluations):
    """Recorded because it is easy to assume otherwise and act on it.

        h=1,2  calendar models win by ~3 points
        h=3    calendar_only wins; calendar_aware loses to ma8
        h=4    calendar_aware wins; calendar_only loses to ma8
        h=5    ma8 beats both

    **A record of a finding on the v1 synthetic corpus (seed 20260908), pinned to its
    archive.** It did not survive regeneration: on v2 (seed 58068921) `calendar_aware`
    wins at every horizon. So it is a fact about one random draw, not about the models,
    and is kept only as a reproducible record of what was measured. Skips if the archive
    is absent.
    """
    table = mape_table(v1_evaluations)
    winners = {int(h): table[h].idxmin() for h in (1, 2, 3, 4, 5)}
    assert len(set(winners.values())) > 1, (
        f"expected different models to win at different horizons, got {winners}")
    assert winners[5] == "ma8", (
        "at h=5 the 8-week average is the one to beat, not the calendar")


def test_the_level_term_does_not_earn_its_place(v1_evaluations):
    """calendar_only has one parameter fewer and is better on average.

    The weekly series is barely autocorrelated (+0.10 at lag 1, -0.13 at lag 3), so a
    horizon-specific level coefficient fitted on ~40 observations adds variance rather
    than signal.

    **A record of a finding on the v1 synthetic corpus (seed 20260908), pinned to its
    archive.** It did not survive regeneration — on v2 (seed 58068921) the ordering
    reverses, `calendar_only` 22.19 against `calendar_aware` 20.85 mean MAPE — and it was
    ALSO reversed on the real EY data, where the recent level turned out to be
    horizon-dependent rather than worthless. Do not cite it as a property of the models.
    Kept only as a reproducible record of what was measured on v1. Skips if the archive
    is absent.
    """
    table = mape_table(v1_evaluations)
    assert table.loc["calendar_only"].mean() < table.loc["calendar_aware"].mean()


def test_moving_averages_beat_naive_at_horizon_one(evaluations):
    table = mape_table(evaluations)
    assert table.loc["ma8", 1] < table.loc["naive", 1]


def test_score_reports_all_three_metrics(evaluations):
    scored = score(evaluations)
    assert {"mape", "wape", "bias"} <= set(scored.columns)
    assert (scored["n"] > 20).all(), "too few origins to trust any of these"


def _evaluations(rows) -> pd.DataFrame:
    """A hand-made evaluations frame from ``(model, horizon, origin, actual, predicted)``.

    The corpus fixtures above are module-scoped and slow, and a metric test wants numbers
    small enough to check by eye rather than real ones.
    """
    return pd.DataFrame(
        [{"model": m, "horizon": int(h), "origin": int(o), "target_week": int(o) + int(h),
          "actual": float(a), "predicted": float(p)} for m, h, o, a, p in rows])


def test_score_reports_r2_beside_the_other_three(evaluations):
    """The fourth column, added because the other three all average ABSOLUTE errors."""
    scored = score(evaluations)
    assert {"mape", "wape", "bias", "r2"} <= set(scored.columns)
    assert scored["r2"].notna().all(), "no model/horizon should score an undefined R²"
    assert (scored["r2"] < 1.0).all(), "a perfect forecast would be a leak, not a result"


def test_r2_ranks_models_within_one_horizon_by_squared_error():
    """Why the R² column can be read straight DOWN, which is what the pipeline claims.

    Inside one horizon every model is scored on the same target weeks, so SST is shared
    and ``1 − SSE/SST`` is a monotone transform of the sum of squared error. Ranking by
    R² is therefore exactly ranking by SSE — no hindsight enters the COMPARISON even
    though it sits in the number.
    """
    actual = [100.0, 100.0, 100.0, 100.0, 500.0]
    predictions = {                                     # SSE, by hand:
        "tight": [100.0, 100.0, 100.0, 100.0, 480.0],   #     20²  =    400
        "middle": [90.0, 90.0, 90.0, 90.0, 450.0],      #  4·10² + 50²  =  2_900
        "loose": [100.0, 100.0, 100.0, 100.0, 300.0],   #    200²  = 40_000
    }
    rows = [(model, 1, origin, a, p)
            for model, predicted in predictions.items()
            for origin, (a, p) in enumerate(zip(actual, predicted))]

    scored = score(_evaluations(rows)).set_index("model")
    sse = {model: sum((a - p) ** 2 for a, p in zip(actual, predicted))
           for model, predicted in predictions.items()}

    assert list(scored["r2"].sort_values(ascending=False).index) == sorted(sse, key=sse.get)


def test_r2_is_not_comparable_ACROSS_horizons():
    """Each horizon scores a different set of weeks, so each has its own denominator.

    Both rows below make exactly the same errors — every prediction is 50 high — but the
    h=2 target weeks vary less, so the same errors explain away a bigger share of a
    smaller variance and the R² falls from 0.902 to 0.609. Reading the columns across and
    concluding "the model degrades with horizon" would be reading the denominator.

        h=1 actuals 100,100,100,100,500 → SST 128_000     SSE 5·50² = 12_500 → 0.902
        h=2 actuals 200,200,200,200,400 → SST  32_000     SSE 5·50² = 12_500 → 0.609
    """
    rows = [("m", 1, origin, a, a + 50.0)
            for origin, a in enumerate([100.0, 100.0, 100.0, 100.0, 500.0])]
    rows += [("m", 2, origin, a, a + 50.0)
             for origin, a in enumerate([200.0, 200.0, 200.0, 200.0, 400.0])]

    scored = score(_evaluations(rows)).set_index("horizon")
    assert scored.loc[1, "r2"] == pytest.approx(0.902)
    assert scored.loc[2, "r2"] == pytest.approx(0.609)


# ---------------------------------------------------------------- missing files

def test_missing_file_weeks_are_reported_not_dropped(series, weeks):
    """They stay in the evaluation; dropping them would flatter every model."""
    import json
    with open(config.ORACLE, encoding="utf-8") as fh:
        missing = json.load(fh)["ey_spec_alignment"]["missing_files"]

    affected = weeks_affected_by_missing_files(series, weeks, missing)
    assert len(affected) == len(missing), (
        "each missing day should fall in its own week here")
    assert len(series) == len(weeks), "no week was removed from the series"
