"""Tests for `hurdle/topups.py` (U12.1): `trailing_mean_forecasts`, the trailing-mean forecast
of a top-up series (A_h new-customer cash, R refunds, G unidentified cash) per company.

Every series is hand-made; nothing is read from data/. The properties that matter most are
tested directly: the forecast is the mean of the `window` target weeks ending at the origin,
no week after the origin is ever read, a week without data is left out (never counted as 0)
while a week of 0.0 cash counts, and the forecast is flat over h for one series.
"""
import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.benchmark import recent_level
from collection_estimation.hurdle.topups import FORECAST_COLUMNS, trailing_mean_forecasts


def powers_of_two(first=0, last=40):
    """Week w holds 2**w: every subset of weeks has its own mean, so a wrong window shows."""
    weeks = list(range(first, last + 1))
    return pd.Series([float(2 ** w) for w in weeks], index=weeks)


def noisy(weeks, seed=0, level=1000.0):
    rng = np.random.default_rng(seed)
    return pd.Series(level + rng.normal(scale=100.0, size=len(weeks)),
                     index=pd.Index(list(weeks)))


# --- constants -------------------------------------------------------------------------------

def test_forecast_columns():
    assert FORECAST_COLUMNS == ["origin_week", "h", "target_week", "forecast"]


# --- the mean --------------------------------------------------------------------------------

@pytest.mark.parametrize("window", [1, 4, 8, 13])
def test_forecast_is_the_mean_of_the_window_weeks_ending_at_the_origin(window):
    series = powers_of_two()
    origin = 30
    result = trailing_mean_forecasts(series, 1, [origin], window)
    expected = sum(2.0 ** w for w in range(origin - window + 1, origin + 1)) / window
    assert result["forecast"].iloc[0] == pytest.approx(expected, rel=1e-15)


def test_forecast_by_hand():
    series = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], index=[5, 6, 7, 8, 9, 10])
    result = trailing_mean_forecasts(series, 2, [9, 10], 4)
    assert result["forecast"].tolist() == pytest.approx([(2 + 4 + 8 + 16) / 4,
                                                         (4 + 8 + 16 + 32) / 4])


@pytest.mark.parametrize("window", [1, 4, 8, 13])
def test_forecast_equals_recent_level_with_the_window(window):
    series = noisy(range(0, 60), seed=window)
    origins = [20, 33, 47, 59]
    result = trailing_mean_forecasts(series, 3, origins, window)
    assert result["forecast"].tolist() == [recent_level(series, o, weeks=window)
                                           for o in origins]


@pytest.mark.parametrize("window", [1, 8, 13])
def test_forecast_window_edges(window):
    origin = 40
    values = {origin - window: 1e9, origin - window + 1: 3.0, origin: 5.0}  # one key if 1
    series = pd.Series(values)
    result = trailing_mean_forecasts(series, 1, [origin], window)
    assert result["forecast"].iloc[0] == pytest.approx(5.0 if window == 1 else 4.0)


# --- shape -----------------------------------------------------------------------------------

def test_output_columns_and_order():
    result = trailing_mean_forecasts(powers_of_two(), 2, [20, 25], 4)
    assert list(result.columns) == FORECAST_COLUMNS


def test_one_row_per_origin_in_the_given_order():
    origins = [35, 12, 27, 12]
    result = trailing_mean_forecasts(noisy(range(0, 40)), 3, origins, 8)
    assert result["origin_week"].tolist() == origins
    assert result["h"].tolist() == [3, 3, 3, 3]
    assert result["target_week"].tolist() == [38, 15, 30, 15]
    # A repeated origin gets the same forecast.
    assert result["forecast"].iloc[1] == result["forecast"].iloc[3]


@pytest.mark.parametrize("h", [1, 2, 5, 13])
def test_target_week_is_origin_plus_h(h):
    origins = [10, 20, 30]
    result = trailing_mean_forecasts(noisy(range(0, 40)), h, origins, 4)
    assert result["target_week"].tolist() == [o + h for o in origins]
    assert (result["h"] == h).all()


def test_output_dtypes():
    result = trailing_mean_forecasts(noisy(range(0, 40)), 2, [20, 30], 4)
    assert result["forecast"].dtype == "float64"
    for column in ("origin_week", "h", "target_week"):
        assert pd.api.types.is_integer_dtype(result[column]), column


def test_empty_origins_give_an_empty_table_with_the_columns():
    result = trailing_mean_forecasts(noisy(range(0, 40)), 1, [], 4)
    assert len(result) == 0
    assert list(result.columns) == FORECAST_COLUMNS


def test_the_target_week_need_not_be_in_the_series():
    # The forecast is for a week the series does not hold yet (the future).
    series = noisy(range(0, 21))
    result = trailing_mean_forecasts(series, 5, [20], 4)
    assert result["target_week"].iloc[0] == 25
    assert np.isfinite(result["forecast"].iloc[0])


# --- flat over h (R and G) -------------------------------------------------------------------

@pytest.mark.parametrize("window", [4, 8, 13])
def test_forecast_does_not_depend_on_h_but_the_target_week_does(window):
    series = noisy(range(0, 60), seed=2)
    origins = [20, 40, 55]
    results = {h: trailing_mean_forecasts(series, h, origins, window) for h in range(1, 6)}
    for h in range(2, 6):
        assert results[h]["forecast"].tolist() == results[1]["forecast"].tolist()
        assert results[h]["target_week"].tolist() == [o + h for o in origins]
        assert results[h]["target_week"].tolist() != results[1]["target_week"].tolist()


# --- no leakage ------------------------------------------------------------------------------

@pytest.mark.parametrize("window", [1, 4, 8, 13])
@pytest.mark.parametrize("h", [1, 3, 5])
def test_changing_a_week_after_the_origin_does_not_change_the_forecast(h, window):
    series = noisy(range(0, 70), seed=h * 100 + window)
    for origin in (20, 41, 60):
        base = trailing_mean_forecasts(series, h, [origin], window)
        for week in range(origin + 1, 70):
            changed = series.copy()
            changed[week] = -1e12
            again = trailing_mean_forecasts(changed, h, [origin], window)
            pd.testing.assert_frame_equal(base, again)


@pytest.mark.parametrize("window", [4, 13])
@pytest.mark.parametrize("h", [1, 5])
def test_scrambling_or_dropping_every_week_after_the_origin_changes_nothing(h, window):
    series = noisy(range(0, 70), seed=7)
    rng = np.random.default_rng(1)
    origins = [25, 40]
    for origin in origins:
        base = trailing_mean_forecasts(series, h, [origin], window)
        scrambled = series.copy()
        after = scrambled.index > origin
        scrambled[after] = rng.random(int(after.sum())) * 1e9
        pd.testing.assert_frame_equal(base, trailing_mean_forecasts(scrambled, h, [origin],
                                                                    window))
        pd.testing.assert_frame_equal(base, trailing_mean_forecasts(series.loc[:origin], h,
                                                                    [origin], window))


def test_the_target_weeks_own_value_is_never_read():
    series = noisy(range(0, 70), seed=8)
    for h in (1, 4):
        base = trailing_mean_forecasts(series, h, [40], 8)
        changed = series.copy()
        changed[40 + h] = 1e15
        pd.testing.assert_frame_equal(base, trailing_mean_forecasts(changed, h, [40], 8))


def test_a_week_inside_the_window_does_move_the_forecast():
    # The other side of the leak test: the window is really read.
    series = noisy(range(0, 70), seed=8)
    base = trailing_mean_forecasts(series, 1, [40], 8)["forecast"].iloc[0]
    changed = series.copy()
    changed[33] += 800.0
    again = trailing_mean_forecasts(changed, 1, [40], 8)["forecast"].iloc[0]
    assert again == pytest.approx(base + 100.0)


# --- missing weeks, zero weeks, negative values ----------------------------------------------

@pytest.mark.parametrize("window", [4, 8, 13])
def test_missing_weeks_are_left_out_not_counted_as_zero(window):
    series = pd.Series([100.0, 300.0], index=[40 - window + 1, 40])
    result = trailing_mean_forecasts(series, 1, [40], window)
    assert result["forecast"].iloc[0] == pytest.approx(200.0)


def test_a_week_of_zero_cash_counts():
    # Week 38 holds 0.0 (no cash that week): it is in the mean. Missing, it would not be.
    with_zero = pd.Series([90.0, 0.0, 30.0], index=[37, 38, 40])
    without = with_zero.drop(38)
    assert trailing_mean_forecasts(with_zero, 1, [40], 4)["forecast"].iloc[0] == \
        pytest.approx(40.0)
    assert trailing_mean_forecasts(without, 1, [40], 4)["forecast"].iloc[0] == \
        pytest.approx(60.0)


def test_a_window_of_only_zero_weeks_is_a_forecast_of_zero_not_an_error():
    series = pd.Series(0.0, index=range(0, 20))
    result = trailing_mean_forecasts(series, 2, [10, 19], 4)
    assert result["forecast"].tolist() == [0.0, 0.0]


def test_negative_values_refunds_are_averaged_as_they_are():
    refunds = pd.Series([-100.0, 0.0, -50.0, -250.0], index=[7, 8, 9, 10])
    result = trailing_mean_forecasts(refunds, 3, [10], 4)
    assert result["forecast"].iloc[0] == pytest.approx(-100.0)


def test_mixed_signs_are_averaged_as_they_are():
    series = pd.Series([300.0, -100.0, -200.0], index=[8, 9, 10])
    result = trailing_mean_forecasts(series, 1, [10], 4)
    assert result["forecast"].iloc[0] == pytest.approx(0.0)


# --- index -----------------------------------------------------------------------------------

def test_an_unsorted_series_index_works():
    series = pd.Series([16.0, 2.0, 8.0, 4.0, 1e6, 1e9], index=[9, 6, 8, 7, 1, 12])
    result = trailing_mean_forecasts(series, 1, [9], 4)
    assert result["forecast"].iloc[0] == pytest.approx(7.5)
    sorted_result = trailing_mean_forecasts(series.sort_index(), 1, [9], 4)
    pd.testing.assert_frame_equal(result, sorted_result)


def test_a_series_indexed_by_a_named_week_column_works():
    # The pipeline passes table.set_index("week")["deflated"].
    table = pd.DataFrame({"week": [38, 39, 40], "deflated": [1.0, 2.0, 6.0]})
    series = table.set_index("week")["deflated"]
    assert trailing_mean_forecasts(series, 1, [40], 4)["forecast"].iloc[0] == \
        pytest.approx(3.0)


def test_an_integer_series_gives_a_float_forecast():
    series = pd.Series([1, 2], index=[39, 40], dtype="int64")
    result = trailing_mean_forecasts(series, 1, [40], 4)
    assert result["forecast"].iloc[0] == 1.5
    assert result["forecast"].dtype == "float64"


# --- inputs untouched, deterministic ---------------------------------------------------------

def test_does_not_change_its_inputs():
    series = noisy(range(0, 40))
    copy = series.copy(deep=True)
    origins = [30, 20]
    trailing_mean_forecasts(series, 2, origins, 8)
    pd.testing.assert_series_equal(series, copy)
    assert origins == [30, 20]


def test_is_deterministic():
    series = noisy(range(0, 40), seed=3)
    a = trailing_mean_forecasts(series, 2, [20, 30, 39], 8)
    b = trailing_mean_forecasts(series, 2, [20, 30, 39], 8)
    pd.testing.assert_frame_equal(a, b)


# --- refusals --------------------------------------------------------------------------------

@pytest.mark.parametrize("h", [0, -1, -5])
def test_refuses_h_below_one(h):
    with pytest.raises(ValueError) as error:
        trailing_mean_forecasts(noisy(range(0, 40)), h, [20], 4)
    assert str(error.value) == f"h is {h}; horizons start at 1"


@pytest.mark.parametrize("h", [0, -1])
def test_refuses_h_below_one_even_without_origins(h):
    with pytest.raises(ValueError) as error:
        trailing_mean_forecasts(noisy(range(0, 40)), h, [], 4)
    assert str(error.value) == f"h is {h}; horizons start at 1"


@pytest.mark.parametrize("window", [0, -1, -13])
def test_refuses_window_below_one(window):
    with pytest.raises(ValueError) as error:
        trailing_mean_forecasts(noisy(range(0, 40)), 1, [20], window)
    assert str(error.value) == f"window is {window}; the trailing mean needs at least 1 week"


@pytest.mark.parametrize("window", [0, -1])
def test_refuses_window_below_one_even_without_origins(window):
    # The docstring: "Raises ValueError if `h` or `window` is below 1" -- unconditionally,
    # as it does for h.
    with pytest.raises(ValueError) as error:
        trailing_mean_forecasts(noisy(range(0, 40)), 1, [], window)
    assert str(error.value) == f"window is {window}; the trailing mean needs at least 1 week"


def test_a_bad_h_is_reported_before_a_bad_window():
    with pytest.raises(ValueError) as error:
        trailing_mean_forecasts(noisy(range(0, 40)), 0, [20], 0)
    assert str(error.value) == "h is 0; horizons start at 1"


def test_a_bad_window_is_reported_before_an_origin_without_data():
    with pytest.raises(ValueError) as error:
        trailing_mean_forecasts(pd.Series([], dtype=float), 1, [20], 0)
    assert str(error.value) == "window is 0; the trailing mean needs at least 1 week"


def test_refuses_an_origin_with_no_data_in_its_window():
    series = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 30])
    with pytest.raises(ValueError) as error:
        trailing_mean_forecasts(series, 2, [12, 25], 8)
    assert str(error.value) == ("h 2, origin 25: no week with data in the 8 weeks up to the "
                                "origin, so no trailing mean")


def test_refuses_an_origin_whose_only_data_is_just_outside_or_after_the_window():
    series = pd.Series([1.0, 2.0], index=[36, 41])
    with pytest.raises(ValueError) as error:
        trailing_mean_forecasts(series, 1, [40], 4)
    assert str(error.value) == ("h 1, origin 40: no week with data in the 4 weeks up to the "
                                "origin, so no trailing mean")


def test_refuses_an_empty_series():
    with pytest.raises(ValueError) as error:
        trailing_mean_forecasts(pd.Series([], dtype=float), 3, [5], 13)
    assert str(error.value) == ("h 3, origin 5: no week with data in the 13 weeks up to the "
                                "origin, so no trailing mean")


def test_an_origin_before_any_data_is_refused_the_first_one_with_data_is_not():
    series = pd.Series([5.0, 7.0], index=[10, 11])
    with pytest.raises(ValueError):
        trailing_mean_forecasts(series, 1, [9], 4)
    assert trailing_mean_forecasts(series, 1, [10], 4)["forecast"].iloc[0] == 5.0
