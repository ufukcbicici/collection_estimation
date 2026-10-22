"""Tests for `hurdle/benchmark.py` (U11.5): `cash_totals`, `recent_level`, `deflated_forecasts`
and `benchmark_forecasts`, the `deflated` benchmark the hurdle model is compared with.

Every series and calendar is hand-made; nothing is read from data/. The properties that matter
most are tested directly: no week after the origin is ever read for a forecast (and no target
week after `train_until` for the fit), a week without data is skipped and never counted as 0,
a series driven only by the calendar is recovered, and each company's series is its own.
"""
import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.benchmark import (
    ALPHAS,
    LEVEL_WEEKS,
    MIN_TRAINING_WEEKS,
    benchmark_forecasts,
    cash_totals,
    deflated_forecasts,
    recent_level,
)

FORECAST_COLUMNS = ["origin_week", "h", "target_week", "forecast", "fallback"]


def calendar(first=0, last=80):
    """A week calendar shaped like `calendar_features`: ints and bools, one row per week."""
    weeks = np.arange(first, last + 1)
    return pd.DataFrame({
        "week": weeks.astype("int64"),
        "business_days": (5 - (weeks % 7 == 3)).astype("int64"),
        "has_month_end": weeks % 4 == 3,
        "has_quarter_end": weeks % 13 == 12,
        "has_mid_month": weeks % 4 == 1,
        "holidays_week_before": (weeks % 7 == 4).astype("int64"),
    })


def calendar_driven(cal, weeks):
    """Cash that is an exact linear function of the calendar features of its week."""
    rows = cal.set_index("week").loc[list(weeks)]
    values = (1000.0 + 50.0 * rows["business_days"] + 400.0 * rows["has_month_end"]
              + 300.0 * rows["has_quarter_end"] + 120.0 * rows["has_mid_month"]
              - 80.0 * rows["holidays_week_before"])
    return pd.Series(values.to_numpy(dtype=float), index=pd.Index(list(weeks)))


def noisy_series(weeks, seed=0, level=1000.0):
    rng = np.random.default_rng(seed)
    return pd.Series(level + rng.normal(scale=100.0, size=len(weeks)),
                     index=pd.Index(list(weeks)))


# --- constants -------------------------------------------------------------------------------

def test_constants():
    assert LEVEL_WEEKS == 4
    assert MIN_TRAINING_WEEKS == 12
    assert ALPHAS == (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


# --- cash_totals -----------------------------------------------------------------------------

def panel(index=None):
    return pd.DataFrame({
        "origin_week": [10, 10, 10, 10, 11, 10],
        "h": [1, 1, 2, 1, 1, 1],
        "target_week": [11, 11, 12, 11, 12, 11],
        "company_code": ["TR04", "TR02", "TR02", "TR02", "TR02", "TR04"],
        "key": list("abcdef"),
    }, index=index)


def test_cash_totals_consolidated_by_hand():
    values = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    totals = cash_totals(panel(), values, by_company=False)
    assert list(totals.columns) == ["origin_week", "h", "target_week", "total"]
    assert totals[["origin_week", "h", "target_week"]].values.tolist() == [
        [10, 1, 11], [10, 2, 12], [11, 1, 12]]
    assert totals["total"].tolist() == [1.0 + 2.0 + 8.0 + 32.0, 4.0, 16.0]


def test_cash_totals_by_company_by_hand():
    values = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    totals = cash_totals(panel(), values, by_company=True)
    assert list(totals.columns) == ["origin_week", "h", "target_week", "company_code", "total"]
    assert totals.drop(columns="total").values.tolist() == [
        [10, 1, 11, "TR02"], [10, 1, 11, "TR04"], [10, 2, 12, "TR02"], [11, 1, 12, "TR02"]]
    assert totals["total"].tolist() == [10.0, 33.0, 4.0, 16.0]


def test_cash_totals_company_totals_add_up_to_the_consolidated_ones():
    rng = np.random.default_rng(3)
    values = pd.Series(rng.random(6) * 100)
    by_company = cash_totals(panel(), values, by_company=True)
    consolidated = cash_totals(panel(), values, by_company=False)
    summed = by_company.groupby(["origin_week", "h", "target_week"], as_index=False)["total"].sum()
    np.testing.assert_allclose(summed["total"], consolidated["total"])


def test_cash_totals_works_on_a_shuffled_non_default_index():
    index = [50, 3, 9, 1, 77, 2]
    values = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], index=index)
    totals = cash_totals(panel(index=index), values, by_company=False)
    assert totals["total"].tolist() == [43.0, 4.0, 16.0]


def test_cash_totals_ints_and_bools_are_summed_as_floats():
    totals = cash_totals(panel(), pd.Series([1, 0, 1, 1, 0, 1]), by_company=False)
    assert totals["total"].dtype == "float64"
    assert totals["total"].tolist() == [3.0, 1.0, 0.0]


def test_cash_totals_categorical_company_leaves_unused_categories_out():
    rows = panel().assign(company_code=pd.Categorical(
        panel()["company_code"], categories=["TR02", "TR04", "TR05"]))
    totals = cash_totals(rows, pd.Series(np.ones(6)), by_company=True)
    assert "TR05" not in set(totals["company_code"])
    assert len(totals) == 4


def test_cash_totals_does_not_change_its_inputs():
    rows, values = panel(), pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    copies = rows.copy(deep=True), values.copy(deep=True)
    cash_totals(rows, values, by_company=True)
    pd.testing.assert_frame_equal(rows, copies[0])
    pd.testing.assert_series_equal(values, copies[1])


@pytest.mark.parametrize("values", [
    pd.Series(np.ones(6)).set_axis(range(1, 7)),
    pd.Series(np.ones(5)),
    pd.Series(np.ones(6))[::-1],
], ids=["shifted", "shorter", "reversed"])
def test_cash_totals_refuses_misaligned_values(values):
    with pytest.raises(ValueError) as error:
        cash_totals(panel(), values, by_company=False)
    assert str(error.value) == "values must have the rows of rows (the same index)"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_cash_totals_refuses_nan_and_infinite_values(bad):
    values = pd.Series([1.0, 2.0, bad, 8.0, 16.0, 32.0])
    with pytest.raises(ValueError) as error:
        cash_totals(panel(), values, by_company=True)
    assert str(error.value) == "values holds a NaN or infinite value"


# --- recent_level ----------------------------------------------------------------------------

def test_recent_level_is_the_mean_of_the_four_weeks_ending_at_the_origin():
    series = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], index=[5, 6, 7, 8, 9, 10])
    assert recent_level(series, 9) == pytest.approx((2 + 4 + 8 + 16) / 4)
    assert recent_level(series, 10) == pytest.approx((4 + 8 + 16 + 32) / 4)


def test_recent_level_skips_missing_weeks_rather_than_counting_them_as_zero():
    series = pd.Series([100.0, 300.0], index=[37, 39])
    assert recent_level(series, 40) == pytest.approx(200.0)


def test_recent_level_with_one_week_of_data():
    assert recent_level(pd.Series([7.0], index=[40]), 40) == 7.0
    assert recent_level(pd.Series([7.0], index=[37]), 40) == 7.0


def test_recent_level_is_nan_when_no_week_in_the_window_has_data():
    series = pd.Series([1.0, 2.0], index=[30, 36])
    assert np.isnan(recent_level(series, 40))
    assert np.isnan(recent_level(pd.Series([], dtype=float), 40))


def test_recent_level_window_edges():
    # Week origin - LEVEL_WEEKS is just outside; the origin itself is inside.
    series = pd.Series([1000.0, 1.0, 2.0], index=[36, 37, 40])
    assert recent_level(series, 40) == pytest.approx(1.5)


def test_recent_level_never_reads_a_week_after_the_origin():
    base = pd.Series(np.arange(1.0, 31.0), index=range(0, 30))
    origin = 15
    expected = recent_level(base, origin)
    for week in range(origin + 1, 30):
        changed = base.copy()
        changed[week] = 1e9
        assert recent_level(changed, origin) == expected, week


def test_recent_level_does_not_need_a_sorted_index():
    series = pd.Series([16.0, 2.0, 8.0, 4.0], index=[9, 6, 8, 7])
    assert recent_level(series, 9) == pytest.approx(7.5)


# --- deflated_forecasts: shape ---------------------------------------------------------------

def test_deflated_forecasts_one_row_per_origin_in_the_given_order():
    cal = calendar()
    series = noisy_series(range(0, 50))
    result = deflated_forecasts(series, cal, 2, 40, [45, 41, 43], with_level=True)
    assert list(result.columns) == FORECAST_COLUMNS
    assert result["origin_week"].tolist() == [45, 41, 43]
    assert result["h"].tolist() == [2, 2, 2]
    assert result["target_week"].tolist() == [47, 43, 45]
    assert result["fallback"].dtype == bool


def test_deflated_forecasts_with_no_origins_is_empty():
    result = deflated_forecasts(noisy_series(range(0, 50)), calendar(), 1, 40, [], True)
    assert len(result) == 0


def test_deflated_forecasts_does_not_change_its_inputs():
    cal, series = calendar(), noisy_series(range(0, 50))
    copies = cal.copy(deep=True), series.copy(deep=True)
    origins = [41, 42]
    deflated_forecasts(series, cal, 3, 40, origins, True)
    pd.testing.assert_frame_equal(cal, copies[0])
    pd.testing.assert_series_equal(series, copies[1])
    assert origins == [41, 42]


def test_deflated_forecasts_is_deterministic():
    cal, series = calendar(), noisy_series(range(0, 50), seed=9)
    a = deflated_forecasts(series, cal, 2, 40, [40, 42, 44], True)
    b = deflated_forecasts(series, cal, 2, 40, [40, 42, 44], True)
    pd.testing.assert_frame_equal(a, b)


# --- deflated_forecasts: what it learns ------------------------------------------------------

def test_deflated_forecasts_recovers_a_calendar_driven_series_without_the_level():
    cal = calendar()
    series = calendar_driven(cal, range(0, 61))
    origins = list(range(40, 55))
    # The whole series is passed (each origin needs a recent level, even when unused).
    result = deflated_forecasts(series, cal, 1, 40, origins, with_level=False)
    truth = calendar_driven(cal, [o + 1 for o in origins]).to_numpy()
    assert not result["fallback"].any()
    np.testing.assert_allclose(result["forecast"], truth, rtol=0.01)


def test_deflated_forecasts_recovers_a_calendar_driven_series_with_the_level():
    cal = calendar()
    series = calendar_driven(cal, range(0, 61))
    origins = list(range(40, 55))
    result = deflated_forecasts(series, cal, 3, 40, origins, with_level=True)
    truth = calendar_driven(cal, [o + 3 for o in origins]).to_numpy()
    assert not result["fallback"].any()
    np.testing.assert_allclose(result["forecast"], truth, rtol=0.05)


def test_deflated_forecasts_beats_the_recent_level_on_a_calendar_driven_series():
    cal = calendar()
    series = calendar_driven(cal, range(0, 61))
    origins = list(range(40, 55))
    result = deflated_forecasts(series, cal, 1, 40, origins, with_level=False)
    truth = calendar_driven(cal, [o + 1 for o in origins]).to_numpy()
    levels = np.array([recent_level(series, o) for o in origins])
    assert (np.abs(result["forecast"] - truth).mean()
            < 0.2 * np.abs(levels - truth).mean())


def test_deflated_forecasts_a_constant_series_is_forecast_as_that_constant():
    series = pd.Series(500.0, index=range(0, 50))
    result = deflated_forecasts(series, calendar(), 2, 40, [40, 45], True)
    np.testing.assert_allclose(result["forecast"], 500.0, rtol=1e-6)


# --- deflated_forecasts: fallback ------------------------------------------------------------

def test_deflated_forecasts_falls_back_to_the_level_with_too_few_training_weeks():
    series = noisy_series(range(0, 20), seed=1)
    train_until = MIN_TRAINING_WEEKS - 2       # weeks 0..10: 11 training weeks
    origins = [10, 12, 15]
    result = deflated_forecasts(series, calendar(), 1, train_until, origins, False)
    assert result["fallback"].all()
    assert result["forecast"].tolist() == pytest.approx(
        [recent_level(series, o) for o in origins])


def test_deflated_forecasts_exactly_min_training_weeks_fits_a_model():
    series = calendar_driven(calendar(), range(0, 20))
    train_until = MIN_TRAINING_WEEKS - 1       # weeks 0..11: 12 training weeks
    result = deflated_forecasts(series, calendar(), 1, train_until, [11], False)
    assert not result["fallback"].any()


def test_deflated_forecasts_weeks_without_a_level_are_not_training_weeks():
    # 12 weeks of data, h = 5: weeks 0 to 4 have their origin before any data (no level), so
    # with the level only 7 training weeks remain -> fallback; without it, 12 -> a model.
    series = calendar_driven(calendar(), range(0, 12))
    with_level = deflated_forecasts(series, calendar(), 5, 11, [11], True)
    without = deflated_forecasts(series, calendar(), 5, 11, [11], False)
    assert with_level["fallback"].tolist() == [True]
    assert with_level["forecast"].iloc[0] == pytest.approx(recent_level(series, 11))
    assert without["fallback"].tolist() == [False]


def test_deflated_forecasts_a_missing_week_is_not_a_training_week():
    # Weeks 0..12 with week 6 missing: 12 weeks of data -> a model. With weeks 6 and 7
    # missing, 11 -> the level. A missing week counted as 0 would have given 13 rows.
    weeks = [w for w in range(0, 13) if w != 6]
    series = calendar_driven(calendar(), weeks)
    assert not deflated_forecasts(series, calendar(), 1, 12, [12], False)["fallback"].any()
    fewer = series.drop(7)
    assert deflated_forecasts(fewer, calendar(), 1, 12, [12], False)["fallback"].all()


def test_deflated_forecasts_a_missing_week_differs_from_a_week_of_zero_cash():
    weeks = [w for w in range(0, 50) if w not in (20, 33)]
    series = noisy_series(weeks, seed=4)
    zeros = pd.concat([series, pd.Series(0.0, index=[20, 33])]).sort_index()
    a = deflated_forecasts(series, calendar(), 1, 40, [41, 42], True)
    b = deflated_forecasts(zeros, calendar(), 1, 40, [41, 42], True)
    assert not np.allclose(a["forecast"], b["forecast"])


def test_deflated_forecasts_falls_back_when_the_prediction_is_not_above_zero():
    # One calendar feature x, 0 or 1 in the training weeks; cash = 1,000 - 900 x. At week 46
    # x is 5, never seen: the fitted line gives about 1,000 - 4,500 < 0, so the level is used.
    weeks = np.arange(0, 50)
    x = np.where(weeks % 3 == 0, 1.0, 0.0)
    x[weeks == 46] = 5.0
    cal = pd.DataFrame({"week": weeks, "x": x})
    series = pd.Series(1000.0 - 900.0 * x[:46], index=pd.Index(weeks[:46]))
    result = deflated_forecasts(series, cal, 1, 40, [40, 45], False)
    assert result["fallback"].tolist() == [False, True]
    assert result["forecast"].iloc[1] == pytest.approx(recent_level(series, 45))
    assert result["forecast"].iloc[0] > 0


def test_deflated_forecasts_falls_back_on_a_negative_prediction_by_monkeypatch(monkeypatch):
    from sklearn.pipeline import Pipeline
    monkeypatch.setattr(Pipeline, "predict", lambda self, x: np.full(len(x), -5.0))
    series = noisy_series(range(0, 50))
    result = deflated_forecasts(series, calendar(), 1, 40, [40, 45], True)
    assert result["fallback"].all()
    assert result["forecast"].tolist() == pytest.approx(
        [recent_level(series, 40), recent_level(series, 45)])


@pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
def test_deflated_forecasts_falls_back_on_zero_nan_and_inf_predictions(monkeypatch, value):
    from sklearn.pipeline import Pipeline
    monkeypatch.setattr(Pipeline, "predict", lambda self, x: np.full(len(x), value))
    series = noisy_series(range(0, 50))
    result = deflated_forecasts(series, calendar(), 1, 40, [44], True)
    assert result["fallback"].tolist() == [True]
    assert result["forecast"].iloc[0] == pytest.approx(recent_level(series, 44))


def test_deflated_forecasts_a_positive_prediction_is_not_a_fallback(monkeypatch):
    from sklearn.pipeline import Pipeline
    monkeypatch.setattr(Pipeline, "predict", lambda self, x: np.full(len(x), 1e-9))
    result = deflated_forecasts(noisy_series(range(0, 50)), calendar(), 1, 40, [44], True)
    assert result["fallback"].tolist() == [False]
    assert result["forecast"].iloc[0] == 1e-9


# --- deflated_forecasts: no leakage ----------------------------------------------------------

@pytest.mark.parametrize("with_level", [True, False])
@pytest.mark.parametrize("h", [1, 3, 5])
def test_deflated_forecasts_never_reads_a_week_after_the_origin(with_level, h):
    # Origins at or after train_until: every value after each origin is scrambled and the
    # forecast must not move.
    series = noisy_series(range(0, 70), seed=h)
    train_until = 40
    rng = np.random.default_rng(100 + h)
    for origin in (40, 43, 50):
        base = deflated_forecasts(series, calendar(), h, train_until, [origin], with_level)
        changed = series.copy()
        after = changed.index > origin
        changed[after] = rng.random(int(after.sum())) * 1e6
        again = deflated_forecasts(changed, calendar(), h, train_until, [origin], with_level)
        pd.testing.assert_frame_equal(base, again)


@pytest.mark.parametrize("h", [1, 4])
def test_deflated_forecasts_dropping_weeks_after_the_origin_changes_nothing(h):
    series = noisy_series(range(0, 70), seed=11)
    for origin in (40, 47):
        base = deflated_forecasts(series, calendar(), h, 40, [origin], True)
        cut = deflated_forecasts(series.loc[:origin], calendar(), h, 40, [origin], True)
        pd.testing.assert_frame_equal(base, cut)


def test_deflated_forecasts_the_model_only_learns_from_weeks_up_to_train_until():
    # Without the level, weeks after train_until (but at or before the origin) must not move
    # the forecast at all: they are neither training weeks nor inputs.
    series = noisy_series(range(0, 70), seed=5)
    base = deflated_forecasts(series, calendar(), 2, 40, [50, 55], False)
    changed = series.copy()
    changed[(changed.index > 40)] = 1e7
    again = deflated_forecasts(changed, calendar(), 2, 40, [50, 55], False)
    pd.testing.assert_frame_equal(base, again)


def test_deflated_forecasts_weeks_between_train_until_and_origin_do_enter_the_level():
    # Legitimate: the level at the origin reads the weeks up to the origin, after train_until.
    series = noisy_series(range(0, 70), seed=5)
    base = deflated_forecasts(series, calendar(), 2, 40, [50], True)
    changed = series.copy()
    changed[48] = 1e6
    again = deflated_forecasts(changed, calendar(), 2, 40, [50], True)
    assert again["forecast"].iloc[0] != base["forecast"].iloc[0]


def test_deflated_forecasts_the_target_weeks_own_cash_is_never_an_input():
    series = noisy_series(range(0, 70), seed=8)
    for origin in (40, 45):
        base = deflated_forecasts(series, calendar(), 1, 40, [origin], True)
        changed = series.copy()
        changed[origin + 1] = -1e9
        again = deflated_forecasts(changed, calendar(), 1, 40, [origin], True)
        pd.testing.assert_frame_equal(base, again)


# --- deflated_forecasts: refusals ------------------------------------------------------------

def test_deflated_forecasts_refuses_a_calendar_without_a_training_week():
    cal = calendar()
    cal = cal[~cal["week"].isin([7, 9])]
    with pytest.raises(ValueError) as error:
        deflated_forecasts(noisy_series(range(0, 50)), cal, 1, 40, [41], True)
    assert str(error.value) == "the calendar has no rows for weeks [7, 9]"


def test_deflated_forecasts_refuses_a_calendar_without_a_target_week():
    with pytest.raises(ValueError) as error:
        deflated_forecasts(noisy_series(range(0, 50)), calendar(0, 44), 3, 40, [40, 42], True)
    assert str(error.value) == "the calendar has no rows for weeks [45]"


def test_deflated_forecasts_does_not_need_calendar_weeks_after_train_until_that_are_not_targets():
    # Series weeks 41..49 are neither training weeks nor targets of origin 40 with h = 1.
    result = deflated_forecasts(noisy_series(range(0, 50)), calendar(0, 41), 1, 40, [40], True)
    assert len(result) == 1


@pytest.mark.parametrize("with_level", [True, False])
def test_deflated_forecasts_refuses_an_origin_without_a_recent_level(with_level):
    weeks = [w for w in range(0, 50) if not 41 <= w <= 44]
    with pytest.raises(ValueError) as error:
        deflated_forecasts(noisy_series(weeks), calendar(), 2, 40, [40, 44], with_level)
    assert str(error.value) == (
        f"h 2, origin 44: no week with data in the {LEVEL_WEEKS} weeks up to the origin, so "
        f"no recent level")


# --- benchmark_forecasts ---------------------------------------------------------------------

def totals_frame(weeks, horizons=(1, 2), companies=None, seed=0):
    """Totals shaped like `cash_totals`: for each h, one total per target week (origin =
    target - h), per company if given."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in horizons:
        for company in (companies or [None]):
            for week in weeks:
                row = {"origin_week": week - h, "h": h, "target_week": week}
                if company is not None:
                    row["company_code"] = company
                row["total"] = 1000.0 + 100.0 * h + rng.normal(scale=50.0)
                rows.append(row)
    return pd.DataFrame(rows)


def series_of(totals, h, company=None):
    mine = totals["h"] == h
    if company is not None:
        mine &= totals["company_code"] == company
    return totals[mine].set_index("target_week")["total"].sort_index()


def test_benchmark_forecasts_consolidated_equals_deflated_forecasts_per_h():
    totals = totals_frame(range(2, 60))
    wanted = pd.DataFrame({"origin_week": [45, 41, 43, 41, 44], "h": [1, 2, 1, 1, 2]})
    result = benchmark_forecasts(totals, calendar(), 40, wanted, True)
    assert list(result.columns) == ["origin_week", "h", "target_week", "forecast", "fallback"]
    assert result[["origin_week", "h"]].values.tolist() == wanted.values.tolist()
    for _, row in result.iterrows():
        direct = deflated_forecasts(series_of(totals, row["h"]), calendar(), int(row["h"]), 40,
                                    [int(row["origin_week"])], True)
        assert row["forecast"] == pytest.approx(direct["forecast"].iloc[0], rel=1e-12)
        assert row["target_week"] == row["origin_week"] + row["h"]
        assert row["fallback"] == direct["fallback"].iloc[0]


def test_benchmark_forecasts_keeps_wanteds_row_order_and_extra_columns():
    totals = totals_frame(range(2, 60))
    wanted = pd.DataFrame({"origin_week": [44, 40, 42], "h": [2, 1, 2],
                           "note": ["x", "y", "z"]})
    result = benchmark_forecasts(totals, calendar(), 40, wanted, False)
    assert result["note"].tolist() == ["x", "y", "z"]
    assert result["target_week"].tolist() == [46, 41, 44]


def test_benchmark_forecasts_a_repeated_wanted_row_gets_the_same_forecast():
    totals = totals_frame(range(2, 60))
    wanted = pd.DataFrame({"origin_week": [42, 42], "h": [1, 1]})
    result = benchmark_forecasts(totals, calendar(), 40, wanted, True)
    assert len(result) == 2
    assert result["forecast"].iloc[0] == result["forecast"].iloc[1]


def test_benchmark_forecasts_per_company_equals_its_own_series():
    totals = totals_frame(range(2, 60), companies=["TR02", "TR04"], seed=2)
    wanted = pd.DataFrame({"origin_week": [41, 41, 42, 42], "h": [1, 1, 2, 2],
                           "company_code": ["TR04", "TR02", "TR02", "TR04"]})
    result = benchmark_forecasts(totals, calendar(), 40, wanted, True)
    assert result["company_code"].tolist() == ["TR04", "TR02", "TR02", "TR04"]
    for _, row in result.iterrows():
        direct = deflated_forecasts(series_of(totals, row["h"], row["company_code"]),
                                    calendar(), int(row["h"]), 40,
                                    [int(row["origin_week"])], True)
        assert row["forecast"] == pytest.approx(direct["forecast"].iloc[0], rel=1e-12)


def test_benchmark_forecasts_companies_are_independent():
    totals = totals_frame(range(2, 60), companies=["TR02", "TR04"], seed=2)
    wanted = pd.DataFrame({"origin_week": [41, 43, 41, 43], "h": [1, 2, 1, 2],
                           "company_code": ["TR02", "TR02", "TR04", "TR04"]})
    base = benchmark_forecasts(totals, calendar(), 40, wanted, True)
    changed = totals.copy()
    other = changed["company_code"] == "TR04"
    changed.loc[other, "total"] = changed.loc[other, "total"] * 7 + 5000
    again = benchmark_forecasts(changed, calendar(), 40, wanted, True)
    tr02 = base["company_code"] == "TR02"
    pd.testing.assert_series_equal(base.loc[tr02, "forecast"], again.loc[tr02, "forecast"])
    assert not np.allclose(base.loc[~tr02, "forecast"], again.loc[~tr02, "forecast"])


def test_benchmark_forecasts_horizons_are_independent():
    totals = totals_frame(range(2, 60), seed=6)
    wanted = pd.DataFrame({"origin_week": [41, 43], "h": [1, 2]})
    base = benchmark_forecasts(totals, calendar(), 40, wanted, True)
    changed = totals.copy()
    changed.loc[changed["h"] == 2, "total"] *= 3
    again = benchmark_forecasts(changed, calendar(), 40, wanted, True)
    assert again["forecast"].iloc[0] == base["forecast"].iloc[0]
    assert again["forecast"].iloc[1] != base["forecast"].iloc[1]


def test_benchmark_forecasts_categorical_company_codes():
    totals = totals_frame(range(2, 60), companies=["TR02", "TR04"], seed=2)
    categories = ["TR02", "TR04", "TR05"]
    totals_cat = totals.assign(company_code=pd.Categorical(totals["company_code"],
                                                           categories=categories))
    wanted = pd.DataFrame({"origin_week": [41, 42], "h": [1, 2],
                           "company_code": pd.Categorical(["TR04", "TR02"],
                                                          categories=categories)})
    result = benchmark_forecasts(totals_cat, calendar(), 40, wanted, True)
    plain = benchmark_forecasts(totals, calendar(), 40,
                                wanted.assign(company_code=["TR04", "TR02"]), True)
    assert result["company_code"].astype(str).tolist() == ["TR04", "TR02"]
    np.testing.assert_allclose(result["forecast"], plain["forecast"])
    assert not result["forecast"].isna().any()


def test_benchmark_forecasts_never_reads_totals_after_the_latest_origin():
    totals = totals_frame(range(2, 70), seed=3)
    wanted = pd.DataFrame({"origin_week": [40, 44, 45], "h": [1, 2, 1]})
    base = benchmark_forecasts(totals, calendar(), 40, wanted, True)
    changed = totals.copy()
    changed.loc[changed["target_week"] > 45, "total"] = 1e9
    again = benchmark_forecasts(changed, calendar(), 40, wanted, True)
    pd.testing.assert_frame_equal(base, again)


def test_benchmark_forecasts_does_not_change_its_inputs():
    totals = totals_frame(range(2, 60), companies=["TR02"])
    wanted = pd.DataFrame({"origin_week": [41], "h": [1], "company_code": ["TR02"]})
    cal = calendar()
    copies = totals.copy(deep=True), wanted.copy(deep=True), cal.copy(deep=True)
    benchmark_forecasts(totals, cal, 40, wanted, True)
    pd.testing.assert_frame_equal(totals, copies[0])
    pd.testing.assert_frame_equal(wanted, copies[1])
    pd.testing.assert_frame_equal(cal, copies[2])


def test_benchmark_forecasts_on_cash_totals_of_a_panel_end_to_end():
    # The pipeline's chain: panel rows -> cash_totals -> benchmark_forecasts.
    rng = np.random.default_rng(12)
    rows = []
    for origin in range(0, 55):
        for h in (1, 2):
            for customer in range(4):
                rows.append({"origin_week": origin, "h": h, "target_week": origin + h,
                             "company_code": "TR02" if customer < 2 else "TR04",
                             "amount_deflated": float(rng.integers(0, 3) * 100)})
    frame = pd.DataFrame(rows)
    totals = cash_totals(frame, frame["amount_deflated"], by_company=True)
    wanted = totals[totals["origin_week"].between(41, 45)].drop(columns=["target_week",
                                                                         "total"])
    result = benchmark_forecasts(totals, calendar(), 40, wanted, True)
    assert len(result) == len(wanted)
    assert np.isfinite(result["forecast"]).all() and (result["forecast"] > 0).all()
