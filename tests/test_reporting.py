"""Tests for the review layer: the per-forecast log, cash error, and five-week risk.

Two things in `reporting.py` are easy to get backwards, and most of what follows exists to
hold them down rather than to cover lines.

**The sign convention.** ``error = predicted - actual``. Positive is an OVER-forecast:
cash that was planned against and did not arrive, a liquidity exposure. Negative is an
UNDER-forecast: a buffer held or credit drawn that turned out to be unnecessary, a
carrying cost. Flipping this would not crash anything — it would relabel one risk as the
other in a file a treasurer reads.

**The exclusion rule in `cumulative_by_origin`.** An origin that does not carry every one
of h=1..5 is DROPPED, not totalled over a shorter window. A four-week total sitting in a
column labelled five-week is a silent failure: the error looks smaller than it is and the
origins stop being comparable with each other.

Everything here is built from hand-made frames whose numbers can be checked by eye. The
corpus fixtures in `test_baselines.py` are module-scoped and slow, and nothing in this
module needs real data — except the two end-to-end tests at the bottom, which use a real
`rolling_origin_baselines` frame because only that can catch the harness and the reporting
layer disagreeing about a column name.
"""
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from collection_estimation import reporting                          # noqa: E402
from collection_estimation.baselines import (                        # noqa: E402
    rolling_origin_baselines,
)
from collection_estimation.panel import build_week_index             # noqa: E402
from collection_estimation.reporting import (                        # noqa: E402
    CUMULATIVE_HORIZONS,
    cash_error,
    code_version,
    cumulative_by_origin,
    cumulative_risk,
    forecast_log,
    write_forecast_log,
)

#: A Monday, so the spine starts on a whole ISO week.
FIRST_MONDAY = pd.Timestamp("2026-01-05")

#: The columns `forecast_log` promises, in order.
LOG_COLUMNS = [
    "model", "horizon",
    "origin", "origin_week", "origin_week_start", "train_cutoff",
    "target_week", "target_iso_week", "target_week_start", "target_week_end",
    "actual_try", "predicted_try", "error_try", "abs_error_try", "ape_pct",
    "direction", "code_version",
]


# ---------------------------------------------------------------- builders

def week_spine(n_weeks: int) -> pd.DataFrame:
    """`n_weeks` contiguous weeks, built by the production spine builder.

    Built through `panel.build_week_index` rather than assembled here, so a change to the
    spine's columns or its week numbering fails these tests instead of silently leaving
    them testing a spine the pipeline no longer produces.
    """
    collections = pd.DataFrame({"document_date": [
        FIRST_MONDAY, FIRST_MONDAY + pd.Timedelta(days=7 * (n_weeks - 1) + 6)]})
    spine = build_week_index(collections)
    assert len(spine) == n_weeks, "the helper itself is wrong, not the code under test"
    return spine


def evaluations(rows) -> pd.DataFrame:
    """An evaluations frame from ``(model, horizon, origin, actual, predicted)`` tuples.

    Same columns `rolling_origin_baselines` emits, so anything downstream that reads a
    column by name is genuinely exercised.
    """
    return pd.DataFrame(
        [{"model": m, "horizon": int(h), "origin": int(o),
          "target_week": int(o) + int(h), "actual": float(a), "predicted": float(p)}
         for m, h, o, a, p in rows])


def one_model(errors, *, actual=1000.0, model="m", horizon=1) -> pd.DataFrame:
    """One model at one horizon with the SIGNED errors given, one per origin.

    ``predicted = actual + error``, so a positive entry is an over-forecast.
    """
    return evaluations([(model, horizon, origin, actual, actual + error)
                        for origin, error in enumerate(errors)])


def cumulative_frame(errors_by_model, *, weekly_actual=100.0) -> pd.DataFrame:
    """A cumulative frame built the way the pipeline builds one.

    ``errors_by_model`` maps a model to its per-origin FIVE-WEEK signed error, spread
    evenly over the five horizons. Run through `cumulative_by_origin` rather than
    fabricated, so `cumulative_risk` is tested on the shape it is actually handed.
    """
    rows = []
    for model, errors in errors_by_model.items():
        for origin, total in enumerate(errors):
            rows += [(model, h, origin, weekly_actual, weekly_actual + total / 5)
                     for h in CUMULATIVE_HORIZONS]
    return cumulative_by_origin(evaluations(rows))


def _fake_git(sha: str, status: str):
    """Stand in for `subprocess.run`, answering the two git calls `code_version` makes."""
    def run(args, **kwargs):
        text = sha if "rev-parse" in args else status
        return subprocess.CompletedProcess(args, 0, stdout=text + "\n", stderr="")
    return run


# ================================================================ code_version

def test_code_version_of_this_checkout_is_a_sha_or_unknown():
    """The contract is the SHAPE of the string, not its value.

    The sha changes with every commit and the ``-dirty`` suffix with every uncommitted
    edit — including the ones that add these tests — so asserting a value would be
    asserting the state of the working tree.
    """
    got = code_version()
    assert isinstance(got, str) and got
    assert got == "unknown" or re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?", got), got


def test_code_version_returns_unknown_when_the_path_is_not_a_checkout(tmp_path):
    assert code_version(tmp_path / "does" / "not" / "exist") == "unknown"


@pytest.mark.parametrize("failure", [
    FileNotFoundError("git"),                        # git is not installed at all
    subprocess.TimeoutExpired("git", 10),            # a hung git, e.g. a network remote
    subprocess.CalledProcessError(128, "git"),       # not a repository
    OSError("something else entirely"),
], ids=["no-git", "timeout", "not-a-repo", "other-os-error"])
def test_code_version_never_raises_whatever_git_does(monkeypatch, failure):
    """A missing version is a nuisance; an exception is a lost run.

    `code_version` is called at the top of step 5, after the pipeline has already done
    every expensive thing it does. Any failure mode has to come back as a string.
    """
    def boom(*args, **kwargs):
        raise failure

    monkeypatch.setattr(reporting.subprocess, "run", boom)
    assert code_version() == "unknown"


def test_code_version_marks_a_dirty_working_tree(monkeypatch):
    """``-dirty`` says the sha alone does not describe what ran — treat it as provisional."""
    monkeypatch.setattr(reporting.subprocess, "run", _fake_git("abc1234", " M src/x.py"))
    assert code_version() == "abc1234-dirty"


def test_code_version_is_the_bare_sha_when_the_tree_is_clean(monkeypatch):
    monkeypatch.setattr(reporting.subprocess, "run", _fake_git("abc1234", ""))
    assert code_version() == "abc1234"


def test_code_version_is_unknown_when_git_prints_no_sha(monkeypatch):
    """An empty sha is not a version. Returning it would put a blank column in the file."""
    monkeypatch.setattr(reporting.subprocess, "run", _fake_git("", ""))
    assert code_version() == "unknown"


# ================================================================ forecast_log

def test_forecast_log_emits_the_documented_columns_in_order():
    log = forecast_log(one_model([10.0]), week_spine(6), version="v")
    assert list(log.columns) == LOG_COLUMNS


def test_forecast_log_is_one_row_per_forecast():
    frame = evaluations([("a", h, o, 100.0, 110.0) for h in (1, 2) for o in (3, 4)])
    log = forecast_log(frame, week_spine(10), version="v")
    assert len(log) == len(frame) == 4


def test_forecast_log_resolves_week_indices_to_dates_from_the_spine():
    """"Origin week 4" is not a claim anyone can audit; a date is."""
    spine = week_spine(10)
    log = forecast_log(evaluations([("a", 3, 4, 100.0, 100.0)]), spine, version="v")
    row = log.iloc[0]

    assert row["origin_week"] == spine.loc[4, "iso_week"]
    assert row["origin_week_start"] == spine.loc[4, "week_start"]
    assert row["target_week"] == 7
    assert row["target_iso_week"] == spine.loc[7, "iso_week"]
    assert row["target_week_start"] == spine.loc[7, "week_start"]
    assert row["target_week_end"] == spine.loc[7, "week_end"]


def test_train_cutoff_is_the_origin_weeks_sunday_not_its_monday():
    """The last day whose collections the model was allowed to see.

    Quoting the Monday instead would understate what the model had seen by five days,
    which is the difference between an honest backtest and a leaking one — and the file
    is the artefact somebody would check that claim against.
    """
    spine = week_spine(10)
    log = forecast_log(evaluations([("a", 1, 4, 100.0, 100.0)]), spine, version="v")
    row = log.iloc[0]

    assert row["train_cutoff"] == spine.loc[4, "week_end"]
    assert row["train_cutoff"] == row["origin_week_start"] + pd.Timedelta(days=6)
    assert row["train_cutoff"] < row["target_week_start"]


def test_error_is_predicted_minus_actual_so_positive_is_an_over_forecast():
    """The sign convention, pinned.

    An over-forecast is cash that was expected and did not arrive — a liquidity exposure.
    An under-forecast is a carrying cost. Getting the sign backwards would swap the two
    everywhere downstream, including in `direction`, and nothing would fail loudly.
    """
    log = forecast_log(evaluations([
        ("a", 1, 0, 100.0, 130.0),          # predicted 30 more than arrived
        ("a", 1, 1, 100.0, 70.0),           # predicted 30 less than arrived
        ("a", 1, 2, 100.0, 100.0),
    ]), week_spine(6), version="v")

    assert list(log["error_try"]) == [30.0, -30.0, 0.0]
    assert list(log["abs_error_try"]) == [30.0, 30.0, 0.0]
    assert list(log["direction"]) == ["over", "under", "exact"]


def test_ape_is_a_percentage_of_the_actual():
    log = forecast_log(evaluations([("a", 1, 0, 200.0, 250.0)]),
                       week_spine(6), version="v")
    assert log.iloc[0]["ape_pct"] == pytest.approx(25.0)


def test_ape_is_nan_for_a_zero_week_rather_than_infinite():
    """A week with no collections has no percentage error — but it still has a cash one.

    NaN drops out of any mean taken over the column. An inf would poison it.
    """
    log = forecast_log(evaluations([("a", 1, 0, 0.0, 500.0)]),
                       week_spine(6), version="v")
    assert np.isnan(log.iloc[0]["ape_pct"])
    assert log.iloc[0]["error_try"] == 500.0
    assert log.iloc[0]["direction"] == "over"


def test_forecast_log_is_sorted_by_model_then_origin_then_horizon():
    """The reading order of the file: one model's run, in time, five horizons per origin."""
    frame = evaluations([("b", 2, 1, 100.0, 100.0), ("a", 2, 0, 100.0, 100.0),
                         ("a", 1, 0, 100.0, 100.0), ("a", 1, 1, 100.0, 100.0)])
    log = forecast_log(frame, week_spine(8), version="v")

    assert list(zip(log["model"], log["origin"], log["horizon"])) == [
        ("a", 0, 1), ("a", 0, 2), ("a", 1, 1), ("b", 1, 2)]
    assert list(log.index) == [0, 1, 2, 3], "the index must be reset, not left scrambled"


def test_forecast_log_stamps_the_version_it_is_given():
    log = forecast_log(one_model([1.0, 2.0]), week_spine(6), version="deadbee-dirty")
    assert set(log["code_version"]) == {"deadbee-dirty"}


def test_forecast_log_asks_for_the_code_version_when_it_is_not_given(monkeypatch):
    """A log without a version cannot be tied to what produced it, so there is no path
    through this function that leaves the column blank."""
    monkeypatch.setattr(reporting, "code_version", lambda *a, **k: "stubbed")
    log = forecast_log(one_model([1.0]), week_spine(6))
    assert set(log["code_version"]) == {"stubbed"}


def test_forecast_log_does_not_mutate_the_evaluations_it_is_given():
    """The pipeline scores, logs, and then goes on using the same frame."""
    frame = one_model([10.0, -10.0])
    before = frame.copy(deep=True)
    forecast_log(frame, week_spine(6), version="v")
    pd.testing.assert_frame_equal(frame, before)


# ================================================================ write_forecast_log

def test_write_forecast_log_writes_utf8_with_a_bom(tmp_path):
    """Without the BOM, Excel on a Turkish Windows machine decodes the file as cp1254.

    Three bytes, and they remove an entire class of "the file you sent is broken" reply.
    The model names are the part that mangles.
    """
    log = forecast_log(one_model([10.0], model="tüfe_deflated"),
                       week_spine(6), version="v")
    path = write_forecast_log(log, tmp_path / "log.csv")

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    text = path.read_text(encoding="utf-8-sig")
    assert "tüfe_deflated" in text
    assert not text.startswith("﻿"), "utf-8-sig must consume the BOM on the way back"


def test_write_forecast_log_creates_the_parent_directory(tmp_path):
    log = forecast_log(one_model([10.0]), week_spine(6), version="v")
    path = write_forecast_log(log, tmp_path / "interim" / "runs" / "log.csv")
    assert path.exists()
    assert path == tmp_path / "interim" / "runs" / "log.csv"


def test_write_forecast_log_writes_iso_dates_not_timestamps(tmp_path):
    """``2026-01-05``, not ``2026-01-05 00:00:00`` — a midnight in a date column is noise
    and Excel-TR will reformat it on open."""
    spine = week_spine(6)
    log = forecast_log(evaluations([("a", 1, 0, 100.0, 110.0)]), spine, version="v")
    path = write_forecast_log(log, tmp_path / "log.csv")

    text = path.read_text(encoding="utf-8-sig")
    assert "00:00:00" not in text
    back = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    for column, week in (("origin_week_start", 0), ("train_cutoff", 0),
                         ("target_week_start", 1), ("target_week_end", 1)):
        expected = spine.loc[week, "week_start" if "start" in column else "week_end"]
        assert back.loc[0, column] == expected.strftime("%Y-%m-%d")


def test_write_forecast_log_does_not_mutate_the_log(tmp_path):
    """It stringifies four date columns for the file. The caller's frame keeps datetimes —
    the pipeline returns that frame for inspection after writing it."""
    log = forecast_log(one_model([10.0]), week_spine(6), version="v")
    before = log.copy(deep=True)
    write_forecast_log(log, tmp_path / "log.csv")
    pd.testing.assert_frame_equal(log, before)


# ================================================================ cash_error

def test_cash_error_reports_the_error_in_lira():
    """Errors +30, −10, +20, −60 on weeks of 1000: mae 30, bias −5."""
    got = cash_error(one_model([30.0, -10.0, 20.0, -60.0])).iloc[0]
    assert got["model"] == "m" and got["horizon"] == 1 and got["n"] == 4
    assert got["mae_try"] == 30.0
    assert got["bias_try"] == -5.0


def test_a_bias_of_zero_does_not_mean_the_forecast_was_right():
    """Why `mae_try` is reported beside it: bias is a NET, and two big errors cancel."""
    got = cash_error(one_model([50.0, -50.0])).iloc[0]
    assert got["bias_try"] == 0.0
    assert got["mae_try"] == 50.0


def test_mean_over_is_averaged_over_the_over_forecasts_only():
    """Not diluted by the weeks that went the other way — the property that makes it usable.

    With +30, −10, +20, −60 the average over-forecast is (30+20)/2 = 25. Divided by all
    four it would read 12.5, and the mean error reads −5. Anyone sizing a liquidity buffer
    on either of those would size it at half, or at nothing.
    """
    got = cash_error(one_model([30.0, -10.0, 20.0, -60.0])).iloc[0]
    assert got["mean_over_try"] == 25.0
    assert got["mean_under_try"] == 35.0                 # (10 + 60) / 2, not (10 + 60) / 4


def test_worst_over_and_worst_under_are_the_tails_and_both_read_positive():
    """`worst_under_try` is a SIZE, not a signed error, so the two columns cannot be told
    apart by sign — only by name. Worth knowing before summing them."""
    got = cash_error(one_model([30.0, -10.0, 20.0, -60.0])).iloc[0]
    assert got["worst_over_try"] == 30.0
    assert got["worst_under_try"] == 60.0
    assert got["mean_under_try"] > 0 and got["mean_over_try"] > 0


def test_over_rate_counts_the_forecasts_that_came_in_above_the_actual():
    got = cash_error(one_model([30.0, -10.0, 20.0, -60.0])).iloc[0]
    assert got["over_rate_pct"] == 50.0


def test_an_exact_forecast_counts_in_neither_direction():
    """It is still in `n` and in the over_rate denominator, which is why the rate here is
    33.3% and not 50%: over and under do not have to add to 100."""
    got = cash_error(one_model([10.0, -10.0, 0.0])).iloc[0]
    assert got["n"] == 3
    assert got["over_rate_pct"] == pytest.approx(33.3)
    assert got["mean_over_try"] == 10.0
    assert got["mean_under_try"] == 10.0


def test_a_model_that_never_over_forecasts_zeroes_its_over_columns():
    """0.0 unambiguously means "there were none": an over-forecast is strictly greater
    than zero, so a non-empty set of them can never average to 0.0."""
    got = cash_error(one_model([-10.0, -20.0, -30.0])).iloc[0]
    assert got["over_rate_pct"] == 0.0
    assert got["mean_over_try"] == 0.0
    assert got["worst_over_try"] == 0.0
    assert got["mean_under_try"] == 20.0
    assert got["worst_under_try"] == 30.0


def test_bias_pct_is_nan_without_a_mean_weekly():
    """The percentage needs a denominator the function is not given. NaN, not a guess."""
    got = cash_error(one_model([100.0, 100.0])).iloc[0]
    assert got["bias_try"] == 100.0
    assert np.isnan(got["bias_pct"])


def test_bias_pct_is_the_bias_as_a_share_of_the_mean_week():
    got = cash_error(one_model([100.0, 100.0]), mean_weekly=1000.0).iloc[0]
    assert got["bias_pct"] == 10.0


def test_bias_pct_is_nan_rather_than_infinite_for_a_zero_mean_week():
    """A degenerate denominator gives no percentage rather than a division by zero."""
    got = cash_error(one_model([100.0]), mean_weekly=0.0).iloc[0]
    assert np.isnan(got["bias_pct"])


def test_cash_error_has_one_sorted_row_per_model_and_horizon():
    frame = pd.concat([
        one_model([10.0], model="b", horizon=2),
        one_model([10.0], model="a", horizon=2),
        one_model([10.0], model="a", horizon=1),
    ], ignore_index=True)
    got = cash_error(frame)
    assert list(zip(got["model"], got["horizon"])) == [("a", 1), ("a", 2), ("b", 2)]


def test_cash_error_does_not_mutate_the_evaluations_it_is_given():
    frame = one_model([30.0, -60.0])
    before = frame.copy(deep=True)
    cash_error(frame, mean_weekly=1000.0)
    pd.testing.assert_frame_equal(frame, before)


# ================================================================ cumulative_by_origin

def test_only_origins_carrying_every_horizon_are_kept():
    """THE exclusion rule. Origin 1 here is the tail of the series: its h=5 target week
    does not exist yet, so it has four forecasts and not five."""
    rows = [("m", h, 0, 100.0, 100.0) for h in (1, 2, 3, 4, 5)]
    rows += [("m", h, 1, 100.0, 100.0) for h in (1, 2, 3, 4)]
    got = cumulative_by_origin(evaluations(rows))
    assert list(got["origin"]) == [0]


def test_a_short_origin_is_dropped_rather_than_totalled_over_a_short_window():
    """The failure this guards against is silent: a four-week total in a column labelled
    five-week, carrying an error that looks smaller than it is and an origin that is no
    longer comparable with the ones above it.

    Every week here is 100 actual against 120 predicted, so a complete origin totals 500
    with an error of 100 and a truncated one would total 400 with an error of 80.
    """
    rows = [("m", h, 0, 100.0, 120.0) for h in (1, 2, 3, 4, 5)]
    rows += [("m", h, 1, 100.0, 120.0) for h in (1, 2, 3, 4)]
    got = cumulative_by_origin(evaluations(rows))

    assert list(got["actual_try"]) == [500.0]
    assert list(got["error_try"]) == [100.0]
    assert 400.0 not in set(got["actual_try"])
    assert 80.0 not in set(got["error_try"])


def test_the_cumulative_error_is_the_sum_of_the_five_SIGNED_errors():
    """Errors from one origin offset across the five weeks or compound across them, and
    which happens is a property of the model — anything flat offsets, anything that
    extrapolates a trend compounds. Summing the signed errors is what lets that show;
    summing absolute ones would report both models as identical.

    Both models below are wrong by 20 TRY in four weeks out of five.
    """
    alternating = [20.0, -20.0, 20.0, -20.0, 10.0]      # nets to +10
    one_way = [20.0, 20.0, 20.0, 20.0, 20.0]            # nets to +100
    rows = [("offsetting", h, 0, 100.0, 100.0 + e)
            for h, e in zip(CUMULATIVE_HORIZONS, alternating)]
    rows += [("compounding", h, 0, 100.0, 100.0 + e)
             for h, e in zip(CUMULATIVE_HORIZONS, one_way)]

    got = cumulative_by_origin(evaluations(rows)).set_index("model")
    assert got.loc["offsetting", "error_try"] == 10.0
    assert got.loc["compounding", "error_try"] == 100.0
    assert got.loc["offsetting", "actual_try"] == got.loc["compounding", "actual_try"]
    assert got.loc["offsetting", "ape_pct"] == pytest.approx(2.0)
    assert got.loc["compounding", "ape_pct"] == pytest.approx(20.0)


def test_horizons_outside_the_window_are_left_out_of_the_total():
    """An h=6 row must not inflate a five-week total."""
    rows = [("m", h, 0, 100.0, 110.0) for h in (1, 2, 3, 4, 5, 6)]
    got = cumulative_by_origin(evaluations(rows))
    assert list(got["actual_try"]) == [500.0]
    assert list(got["predicted_try"]) == [550.0]


def test_the_horizon_window_is_configurable_and_the_rule_follows_it():
    """Narrowing to h=1..2 makes a two-horizon origin complete, and totals over two weeks."""
    rows = [("m", h, 0, 100.0, 110.0) for h in (1, 2, 3, 4, 5)]
    rows += [("m", h, 9, 100.0, 110.0) for h in (1, 2)]
    got = cumulative_by_origin(evaluations(rows), horizons=(1, 2))
    assert list(got["origin"]) == [0, 9]
    assert list(got["actual_try"]) == [200.0, 200.0]


def test_cumulative_direction_follows_the_same_sign_convention():
    rows = []
    for origin, total in ((0, 50.0), (1, -50.0), (2, 0.0)):
        rows += [("m", h, origin, 100.0, 100.0 + total / 5) for h in CUMULATIVE_HORIZONS]
    got = cumulative_by_origin(evaluations(rows))
    assert list(got["direction"]) == ["over", "under", "exact"]


def test_cumulative_ape_is_nan_when_the_five_week_actual_totals_zero():
    """Weekly totals are NET of refunds, so a window summing to zero is arithmetically
    possible. It yields no percentage rather than a division by zero — and the cash error
    is still reported, because that one is well defined."""
    actuals = [300.0, -300.0, 100.0, -100.0, 0.0]
    rows = [("m", h, 0, a, a + 10.0) for h, a in zip(CUMULATIVE_HORIZONS, actuals)]
    got = cumulative_by_origin(evaluations(rows)).iloc[0]

    assert got["actual_try"] == 0.0
    assert np.isnan(got["ape_pct"])
    assert got["error_try"] == 50.0


def test_each_model_is_totalled_separately():
    rows = [(m, h, 0, 100.0, 100.0) for m in ("a", "b") for h in CUMULATIVE_HORIZONS]
    got = cumulative_by_origin(evaluations(rows))
    assert list(got["model"]) == ["a", "b"]
    assert list(got["actual_try"]) == [500.0, 500.0]


# ================================================================ cumulative_risk

def test_cumulative_risk_reports_the_five_week_totals_per_model():
    """Five-week actual is 500 per origin, so errors of +50, −50, +100 and 0 are APEs of
    10%, 10%, 20% and 0% — mean 10 — with a signed total of +100 over 2000 actual."""
    got = cumulative_risk(cumulative_frame({"m": [50.0, -50.0, 100.0, 0.0]})).iloc[0]
    assert got["n_origins"] == 4
    assert got["mape_5w"] == pytest.approx(10.0)
    assert got["bias_5w_pct"] == pytest.approx(5.0)
    assert got["over_rate_pct"] == 50.0


def test_p95_over_is_taken_over_the_over_forecasts_only():
    """The liquidity number: a bad-but-not-unprecedented five-week shortfall.

    Under-forecasts are excluded outright — folding them in would drag the percentile
    toward zero and size a buffer against the wrong distribution. Here two of the four
    origins over-forecast, by 100 and 200, and numpy interpolates the 95th percentile
    linearly between them: 100 + 0.95·100.

    With a handful of over-forecasts this is an indication, not a reliable quantile.
    """
    got = cumulative_risk(
        cumulative_frame({"m": [100.0, 200.0, -1000.0, -2000.0]})).iloc[0]
    assert got["p95_over_try"] == pytest.approx(195.0)
    assert got["worst_over_try"] == 200.0
    assert got["mean_over_try"] == 150.0
    assert got["over_rate_pct"] == 50.0


def test_a_model_that_never_over_forecasts_zeroes_its_over_columns():
    got = cumulative_risk(cumulative_frame({"m": [-100.0, -200.0]})).iloc[0]
    assert got["over_rate_pct"] == 0.0
    assert got["p95_over_try"] == 0.0
    assert got["worst_over_try"] == 0.0
    assert got["mean_over_try"] == 0.0
    assert got["bias_5w_pct"] == pytest.approx(-30.0)      # −300 over 1000 actual


def test_mape_5w_and_bias_5w_tell_different_stories():
    """A model whose five-week totals alternate direction has a bias of exactly zero and a
    MAPE of 20%. Reporting the bias alone would describe it as unbiased and accurate; it
    is only the first."""
    got = cumulative_risk(
        cumulative_frame({"m": [100.0, -100.0, 100.0, -100.0]})).iloc[0]
    assert got["bias_5w_pct"] == 0.0
    assert got["mape_5w"] == pytest.approx(20.0)


def test_cumulative_risk_is_sorted_best_first():
    """The frame is printed as-is in the pipeline, so the order is part of the output."""
    got = cumulative_risk(cumulative_frame({"worse": [250.0, 250.0],
                                            "better": [50.0, 50.0]}))
    assert list(got["model"]) == ["better", "worse"]
    assert got["mape_5w"].is_monotonic_increasing


def test_cumulative_risk_bias_is_nan_when_the_actual_total_is_zero():
    actuals = [300.0, -300.0, 100.0, -100.0, 0.0]
    rows = [("m", h, 0, a, a + 10.0) for h, a in zip(CUMULATIVE_HORIZONS, actuals)]
    got = cumulative_risk(cumulative_by_origin(evaluations(rows))).iloc[0]
    assert np.isnan(got["bias_5w_pct"])
    assert got["over_rate_pct"] == 100.0


# ================================================================ end to end

def synthetic_run(n_weeks: int = 24):
    """A real `rolling_origin_baselines` frame over a hand-made series and calendar.

    No corpus: the point is to exercise the column names and the week arithmetic the
    harness and the reporting layer have to agree on, not the data.
    """
    spine = week_spine(n_weeks)
    series = pd.Series([1000.0 + 100.0 * (i % 4) for i in range(n_weeks)],
                       index=pd.Index(spine["week"], name="week"))
    calendar = pd.DataFrame({
        "week": spine["week"],
        "n_business_days": [5] * n_weeks,
        "has_month_end": [1 if i % 4 == 3 else 0 for i in range(n_weeks)],
        "has_quarter_end": [0] * n_weeks,
        "n_holidays": [0] * n_weeks,
    })
    return spine, rolling_origin_baselines(series, calendar, warmup=8)


def test_the_log_covers_every_evaluation_the_harness_produces():
    """The one place a column renamed in `baselines.py` and not in `reporting.py` shows up.

    Hand-made frames agree with `forecast_log` by construction; this one does not.
    """
    spine, frame = synthetic_run()
    log = forecast_log(frame, spine, version="v")

    assert len(log) == len(frame)
    assert log[["origin_week_start", "train_cutoff",
                "target_week_start", "target_week_end"]].notna().all().all()
    assert (log["train_cutoff"] < log["target_week_start"]).all()
    assert log["abs_error_try"].sum() == pytest.approx(
        (frame["predicted"] - frame["actual"]).abs().sum())


def test_origins_at_the_end_of_the_series_are_dropped_from_the_cumulative():
    """Where the exclusion rule earns its place: the harness genuinely emits short origins.

    The last origin able to carry h=1..5 is ``len(series) - 1 - 5``; every origin after it
    runs out of target weeks and must not appear in a five-week total.
    """
    n_weeks = 24
    _, frame = synthetic_run(n_weeks)
    complete = cumulative_by_origin(frame)

    last_full_origin = n_weeks - 1 - max(CUMULATIVE_HORIZONS)
    expected = sorted(o for o in frame["origin"].unique() if o <= last_full_origin)
    assert expected, "the fixture must produce at least one complete origin"
    assert set(frame["origin"]) - set(complete["origin"]), (
        "the fixture must also produce at least one SHORT origin, or this proves nothing")

    for model in sorted(frame["model"].unique()):
        got = sorted(complete.loc[complete["model"] == model, "origin"])
        assert got == expected, f"{model} kept the wrong origins"
