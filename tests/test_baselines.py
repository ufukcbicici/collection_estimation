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
    assert set(evaluations["model"]) == {"naive", "ma4", "ma8", "calendar_aware"}
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

        generator, seed 20260908, h=1:
        naive 28.4   ma4 25.4   ma8 23.5   calendar_aware 20.2

    **These figures belong to one corpus.** Regenerate with a different seed, or change
    anything upstream of `missing = rng_py.sample(...)` in the generator, and they move —
    see `test_a_missing_file_on_a_month_end_hurts_the_calendar_model` for why they move
    unevenly. If this fails after a regeneration, re-read the generator's own printed
    table before assuming the code is wrong.
    """
    table = mape_table(evaluations)
    generator = {"naive": 28.4, "ma4": 25.4, "ma8": 23.5, "calendar_aware": 20.2}
    for model, expected in generator.items():
        got = table.loc[model, 1]
        assert abs(got - expected) < 1.5, (
            f"{model} at h=1: got {got:.1f}, the generator printed {expected}. Either "
            f"something in the chain from workbooks to weekly series disagrees with the "
            f"generator, or the corpus was regenerated.")


def test_a_missing_file_on_a_month_end_hurts_the_calendar_model():
    """Why the baseline table is sensitive to WHERE the missing files land.

    Six business days have no file. One of them — 2025-09-30 — is the last business day
    of September, and a calendar model flags that week as a month end and predicts high.
    The actual is missing the month's biggest day, so the model takes the full error on
    precisely the week it is most confident about. A moving average barely notices.

    This is not an artefact of synthetic data: a real extract with a day missing at month
    end would do the same thing to a real model.
    """
    from datetime import date
    from collection_estimation.calendar_features import business_days

    work = business_days(date(2025, 6, 2), date(2026, 8, 30))
    last_of_month = {}
    for day in work:
        last_of_month[(day.year, day.month)] = day
    assert date(2025, 9, 30) in set(last_of_month.values())

    import json
    with open(config.ORACLE, encoding="utf-8") as fh:
        missing = json.load(fh)["ey_spec_alignment"]["missing_files"]
    assert "2025-09-30" in missing


def test_calendar_beats_the_moving_averages_at_every_horizon(evaluations):
    """The finding the design leads with: the calendar carries most of the signal.

    The margin is narrower than the design first reported — about 3 points at h=1 rather
    than 8 — but the ordering is unchanged and it holds at every horizon.
    """
    table = mape_table(evaluations)
    for horizon in (1, 2, 3, 4, 5):
        assert table.loc["calendar_aware", horizon] < table.loc["ma8", horizon]
        assert table.loc["calendar_aware", horizon] < table.loc["ma4", horizon]


def test_moving_averages_beat_naive_at_horizon_one(evaluations):
    table = mape_table(evaluations)
    assert table.loc["ma8", 1] < table.loc["naive", 1]


def test_score_reports_all_three_metrics(evaluations):
    scored = score(evaluations)
    assert {"mape", "wape", "bias"} <= set(scored.columns)
    assert (scored["n"] > 20).all(), "too few origins to trust any of these"


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
