"""Tests for `hurdle/topups.py`: `trailing_mean_forecasts` (U12.1), the trailing-mean forecast
of a top-up series (A_h new-customer cash, R refunds, G unidentified cash) per company, and
`topup_forecasts` (U12.2), A, R and G of every company by the mean or the ridge (second half),
and `topup_actuals` and `score_topups` (U12.3), what came and how far a method was from it
(last part).

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


# =============================================================================================
# U12.2: `topup_forecasts`, A, R and G of every company by the trailing mean or the ridge.
#
# The side tables are hand-built in the shape `side_tables.py` makes them (A per origin week,
# h and target week; R per week, negative; G per week), in nominal TRY, with price levels
# that move every week, so deflating by the wrong week shows. The calendar has a `week`
# column and two numeric features covering every week a forecast needs.
# =============================================================================================

from collection_estimation.hurdle.benchmark import (  # noqa: E402
    MIN_TRAINING_WEEKS,
    deflated_forecasts,
)
from collection_estimation.hurdle.prices import deflate  # noqa: E402
from collection_estimation.hurdle.topups import (  # noqa: E402
    METHODS,
    TOPUP_COLUMNS,
    topup_forecasts,
)

COMPANIES = ("TR02", "TR04")


def side_tables(companies=COMPANIES, last=40, horizons=3, seed=0):
    """A (every origin 0..last-1 and h with target <= last), R and G (weeks 0..last), per
    company, nominal TRY, each with a calendar pattern, noise and a company scale."""
    rng = np.random.default_rng(seed)
    new_rows, refund_rows, g_rows = [], [], []
    for number, company in enumerate(companies):
        scale = 1.0 + number
        for origin in range(0, last):
            for h in range(1, horizons + 1):
                target = origin + h
                if target <= last:
                    value = scale * (1000.0 + 200.0 * (target % 4) + 50.0 * h
                                     + rng.normal(scale=40.0))
                    new_rows.append({"company_code": company, "origin_week": origin, "h": h,
                                     "target_week": target, "new_customer_cash": value})
        for week in range(0, last + 1):
            refund_rows.append({"company_code": company, "week": week,
                                "refunds": -scale * (300.0 + 60.0 * (week % 4)
                                                     + abs(rng.normal(scale=20.0)))})
            g_rows.append({"company_code": company, "week": week,
                           "unidentified_cash": scale * (500.0 + 100.0 * np.cos(week)
                                                         + rng.normal(scale=20.0))})
    return pd.DataFrame(new_rows), pd.DataFrame(refund_rows), pd.DataFrame(g_rows)


def price_levels(last=60):
    return pd.Series([1.0 + 0.02 * w + 0.01 * (w % 3) for w in range(0, last + 1)],
                     index=range(0, last + 1))


def calendar_of(last=60):
    weeks = list(range(0, last + 1))
    return pd.DataFrame({"week": weeks, "cycle": [float(w % 4) for w in weeks],
                         "wave": [float(np.cos(w)) for w in weeks]})


def run(tables=None, levels=None, calendar=None, origins=(30,), horizons=3, method="ridge",
        window=8, with_level=True):
    new_cash, refunds, unidentified = tables if tables is not None else side_tables()
    return topup_forecasts(new_cash, refunds, unidentified,
                           price_levels() if levels is None else levels,
                           calendar_of() if calendar is None else calendar, list(origins),
                           horizons, method, window, with_level)


def deflated_series(table, company, week_column, value_column, levels, h=None):
    """One company's series deflated by the week its cash came, sorted by week."""
    own = table[table["company_code"] == company]
    if h is not None:
        own = own[own["h"] == h]
    values = deflate(own[value_column], own[week_column], levels)
    return pd.Series(values.to_numpy(), index=own[week_column].to_numpy()).sort_index()


def row(result, company, origin, h):
    found = result[(result["company_code"] == company) & (result["origin_week"] == origin)
                   & (result["h"] == h)]
    assert len(found) == 1
    return found.iloc[0]


# --- constants -------------------------------------------------------------------------------

def test_topup_columns():
    assert TOPUP_COLUMNS == ["company_code", "origin_week", "h", "target_week",
                             "new_customer_cash", "refunds", "unidentified_cash", "topup",
                             "fallback"]


def test_methods():
    assert METHODS == ("mean", "ridge")


# --- the mean, by hand, with deflation -------------------------------------------------------

def hand_tables():
    """One company, weeks 6..10, price levels that are not 1 (and not monotone):
    6 -> 2.0, 7 -> 4.0, 8 -> 0.5, 9 -> 1.25, 10 -> 2.5."""
    levels = pd.Series({6: 2.0, 7: 4.0, 8: 0.5, 9: 1.25, 10: 2.5})
    a_rows = [
        # h = 1: targets 7, 8, 9, 10 -> deflated 8/4 = 2, 1/0.5 = 2, 5/1.25 = 4, 10/2.5 = 4.
        (6, 1, 7, 8.0), (7, 1, 8, 1.0), (8, 1, 9, 5.0), (9, 1, 10, 10.0),
        # h = 2: targets 8, 9, 10 -> deflated 2/0.5 = 4, 2.5/1.25 = 2, 25/2.5 = 10.
        (6, 2, 8, 2.0), (7, 2, 9, 2.5), (8, 2, 10, 25.0),
    ]
    new_cash = pd.DataFrame([{"company_code": "TR02", "origin_week": o, "h": h,
                              "target_week": t, "new_customer_cash": v}
                             for o, h, t, v in a_rows])
    # R, weeks 6..10: nominal -4, -4, -1, -5, -5 -> deflated -2, -1, -2, -4, -2.
    refunds = pd.DataFrame({"company_code": "TR02", "week": [6, 7, 8, 9, 10],
                            "refunds": [-4.0, -4.0, -1.0, -5.0, -5.0]})
    # G, weeks 6..10: nominal 6, 4, 3, 0, 7.5 -> deflated 3, 1, 6, 0, 3.
    unidentified = pd.DataFrame({"company_code": "TR02", "week": [6, 7, 8, 9, 10],
                                 "unidentified_cash": [6.0, 4.0, 3.0, 0.0, 7.5]})
    return (new_cash, refunds, unidentified), levels


NO_CALENDAR = pd.DataFrame({"week": []})


def test_mean_by_hand_with_deflation():
    tables, levels = hand_tables()
    result = run(tables, levels, NO_CALENDAR, origins=[10], horizons=2, method="mean",
                 window=4)
    assert len(result) == 2
    h1, h2 = row(result, "TR02", 10, 1), row(result, "TR02", 10, 2)
    # A h=1: weeks 7..10 -> (2 + 2 + 4 + 4) / 4; A h=2: weeks 8, 9, 10 only -> 16 / 3.
    assert h1["new_customer_cash"] == pytest.approx(3.0)
    assert h2["new_customer_cash"] == pytest.approx(16.0 / 3.0)
    # R weeks 7..10: (-1 - 2 - 4 - 2) / 4; G weeks 7..10: (1 + 6 + 0 + 3) / 4. Flat over h.
    for line in (h1, h2):
        assert line["refunds"] == pytest.approx(-2.25)
        assert line["unidentified_cash"] == pytest.approx(2.5)
        assert line["topup"] == pytest.approx(line["new_customer_cash"] - 2.25 + 2.5)
        assert not line["fallback"]
    assert h1["target_week"] == 11 and h2["target_week"] == 12


def test_mean_by_hand_a_shorter_window_and_an_earlier_origin():
    tables, levels = hand_tables()
    result = run(tables, levels, NO_CALENDAR, origins=[9, 10], horizons=2, method="mean",
                 window=2)
    # Origin 10, window 2: A h=1 weeks 9, 10 -> 4; A h=2 -> (2 + 10) / 2; R -> -3; G -> 1.5.
    assert row(result, "TR02", 10, 1)["new_customer_cash"] == pytest.approx(4.0)
    assert row(result, "TR02", 10, 2)["new_customer_cash"] == pytest.approx(6.0)
    assert row(result, "TR02", 10, 1)["refunds"] == pytest.approx(-3.0)
    assert row(result, "TR02", 10, 1)["unidentified_cash"] == pytest.approx(1.5)
    # Origin 9, window 2: A h=1 weeks 8, 9 -> 3; A h=2 -> (4 + 2) / 2; R -> -3; G -> 3.
    assert row(result, "TR02", 9, 1)["new_customer_cash"] == pytest.approx(3.0)
    assert row(result, "TR02", 9, 2)["new_customer_cash"] == pytest.approx(3.0)
    assert row(result, "TR02", 9, 2)["refunds"] == pytest.approx(-3.0)
    assert row(result, "TR02", 9, 2)["unidentified_cash"] == pytest.approx(3.0)


def test_mean_deflates_a_by_its_target_week_not_its_origin_week():
    tables, levels = hand_tables()
    result = run(tables, levels, NO_CALENDAR, origins=[10], horizons=1, method="mean",
                 window=1)
    # Target week 10 (level 2.5): 10 / 2.5 = 4. By origin week 9 (level 1.25) it would be 8.
    assert row(result, "TR02", 10, 1)["new_customer_cash"] == pytest.approx(4.0)


def test_mean_does_not_need_the_calendar():
    tables, levels = hand_tables()
    result = run(tables, levels, NO_CALENDAR, origins=[10], horizons=1, method="mean",
                 window=4)
    assert len(result) == 1


# --- the mean equals trailing_mean_forecasts per series --------------------------------------

@pytest.mark.parametrize("window", [1, 4, 8, 13])
def test_mean_equals_trailing_mean_forecasts_of_each_deflated_series(window):
    tables = side_tables()
    new_cash, refunds, unidentified = tables
    levels = price_levels()
    origins = [20, 27, 35]
    result = run(tables, levels, origins=origins, method="mean", window=window)
    for company in COMPANIES:
        r = deflated_series(refunds, company, "week", "refunds", levels)
        g = deflated_series(unidentified, company, "week", "unidentified_cash", levels)
        for h in (1, 2, 3):
            a = deflated_series(new_cash, company, "target_week", "new_customer_cash", levels,
                                h=h)
            mine = result[(result["company_code"] == company) & (result["h"] == h)]
            for series, column in ((a, "new_customer_cash"), (r, "refunds"),
                                   (g, "unidentified_cash")):
                expected = trailing_mean_forecasts(series, h, origins, window)
                assert mine["origin_week"].tolist() == expected["origin_week"].tolist()
                assert mine["target_week"].tolist() == expected["target_week"].tolist()
                assert mine[column].to_numpy() == pytest.approx(
                    expected["forecast"].to_numpy(), rel=1e-12), (company, h, column)


def test_mean_r_and_g_are_flat_over_h_and_a_is_not():
    result = run(origins=[30], method="mean", window=8)
    for company in COMPANIES:
        mine = result[result["company_code"] == company]
        assert mine["refunds"].nunique() == 1
        assert mine["unidentified_cash"].nunique() == 1
        assert mine["new_customer_cash"].nunique() == 3


def test_mean_refunds_stay_negative():
    result = run(origins=[10, 20, 30, 40], method="mean", window=4)
    assert (result["refunds"] < 0).all()


# --- the ridge equals deflated_forecasts per series, refitted at every origin -----------------

@pytest.mark.parametrize("with_level", [True, False])
def test_ridge_equals_deflated_forecasts_refitted_at_each_origin(with_level):
    tables = side_tables()
    new_cash, refunds, unidentified = tables
    levels, calendar = price_levels(), calendar_of()
    origins = [22, 30, 37]
    result = run(tables, levels, calendar, origins=origins, with_level=with_level)
    for company in COMPANIES:
        # R goes into the ridge as -R (positive) and comes back negated.
        minus_r = -deflated_series(refunds, company, "week", "refunds", levels)
        g = deflated_series(unidentified, company, "week", "unidentified_cash", levels)
        for h in (1, 2, 3):
            a = deflated_series(new_cash, company, "target_week", "new_customer_cash", levels,
                                h=h)
            for origin in origins:
                got = row(result, company, origin, h)
                fallbacks = []
                for series, column, sign in ((a, "new_customer_cash", 1.0),
                                             (minus_r, "refunds", -1.0),
                                             (g, "unidentified_cash", 1.0)):
                    expected = deflated_forecasts(series, calendar, h, origin, [origin],
                                                  with_level).iloc[0]
                    assert got[column] == pytest.approx(sign * expected["forecast"],
                                                        rel=1e-12), (company, h, origin, column)
                    fallbacks.append(bool(expected["fallback"]))
                assert bool(got["fallback"]) == any(fallbacks)


def test_ridge_is_refitted_at_each_origin_not_once():
    # A forecast at origin 25 made alone equals the one made alongside a later origin, and a
    # model fitted once up to the latest origin would give something else at origin 25.
    tables, levels, calendar = side_tables(), price_levels(), calendar_of()
    together = run(tables, levels, calendar, origins=[25, 38])
    alone = run(tables, levels, calendar, origins=[25])
    pd.testing.assert_frame_equal(
        together[together["origin_week"] == 25].reset_index(drop=True), alone)
    g = deflated_series(tables[2], "TR02", "week", "unidentified_cash", levels)
    fitted_once = deflated_forecasts(g, calendar, 1, 38, [25], True)["forecast"].iloc[0]
    assert row(alone, "TR02", 25, 1)["unidentified_cash"] != pytest.approx(fitted_once,
                                                                           rel=1e-9)


# --- refunds: the sign round trip ------------------------------------------------------------

@pytest.mark.parametrize("with_level", [True, False])
def test_ridge_refunds_come_back_negative_and_do_not_fall_back(with_level):
    # Regular negative refunds: -R is positive, the ridge predicts above 0 and keeps its fit.
    tables, levels, calendar = side_tables(), price_levels(), calendar_of()
    refunds = tables[1]
    result = run(tables, levels, calendar, origins=[20, 30, 40], with_level=with_level)
    assert (result["refunds"] < 0).all()
    for company in COMPANIES:
        minus_r = -deflated_series(refunds, company, "week", "refunds", levels)
        for h in (1, 2, 3):
            found = deflated_forecasts(minus_r, calendar, h, 30, [30], with_level).iloc[0]
            assert not found["fallback"]
            assert row(result, company, 30, h)["refunds"] == pytest.approx(-found["forecast"])


def test_ridge_refunds_are_not_fed_in_negative():
    # Fed in as they are (negative), every prediction is "not above 0": the ridge would fall
    # back to the level. The function must not do that.
    tables, levels, calendar = side_tables(), price_levels(), calendar_of()
    r = deflated_series(tables[1], "TR02", "week", "refunds", levels)
    as_they_are = deflated_forecasts(r, calendar, 1, 30, [30], True).iloc[0]
    assert as_they_are["fallback"]  # what feeding R in negative would give
    result = run(tables, levels, calendar, origins=[30])
    got = row(result, "TR02", 30, 1)
    assert got["refunds"] != pytest.approx(as_they_are["forecast"], rel=1e-9)
    assert not got["fallback"]


def test_ridge_refunds_fallback_is_the_negative_recent_level():
    # Too few weeks for a fit: every forecast is the recent level; R's is negative.
    levels = pd.Series(2.0, index=range(0, 20))
    refunds = pd.DataFrame({"company_code": "TR02", "week": [5, 6, 7, 8],
                            "refunds": [-2.0, -4.0, -6.0, -16.0]})
    new_cash = pd.DataFrame({"company_code": "TR02", "origin_week": [7], "h": [1],
                             "target_week": [8], "new_customer_cash": [10.0]})
    unidentified = pd.DataFrame({"company_code": "TR02", "week": [8],
                                 "unidentified_cash": [6.0]})
    result = run((new_cash, refunds, unidentified), levels, calendar_of(20), origins=[8],
                 horizons=1)
    got = row(result, "TR02", 8, 1)
    # Deflated R weeks 5..8: -1, -2, -3, -8 -> level -3.5. A: 5; G: 3.
    assert got["refunds"] == pytest.approx(-3.5)
    assert got["new_customer_cash"] == pytest.approx(5.0)
    assert got["unidentified_cash"] == pytest.approx(3.0)
    assert got["topup"] == pytest.approx(4.5)
    assert got["fallback"]


# --- one ridge fit per h ---------------------------------------------------------------------

def same_calendar_rows(calendar, week, like):
    """`calendar` with the features of `week` set to those of `like`."""
    changed = calendar.copy()
    features = [c for c in changed.columns if c != "week"]
    changed.loc[changed["week"] == week, features] = \
        changed.loc[changed["week"] == like, features].to_numpy()
    return changed


def random_walk_tables(seed=3, last=45):
    """R and G that wander (so the recent level carries information), A as usual."""
    rng = np.random.default_rng(seed)
    new_cash, _, _ = side_tables(companies=("TR02",), last=last, horizons=2, seed=seed)
    walk_r = 400.0 + np.cumsum(rng.normal(scale=30.0, size=last + 1))
    walk_g = 900.0 + np.cumsum(rng.normal(scale=50.0, size=last + 1))
    refunds = pd.DataFrame({"company_code": "TR02", "week": range(last + 1),
                            "refunds": -np.abs(walk_r)})
    unidentified = pd.DataFrame({"company_code": "TR02", "week": range(last + 1),
                                 "unidentified_cash": np.abs(walk_g)})
    return new_cash, refunds, unidentified


def test_ridge_without_level_r_and_g_depend_only_on_the_target_weeks_calendar():
    # Without the level, every h's model is fitted on the same rows (week w's calendar ->
    # its cash), so two target weeks with the same calendar get the same R and G.
    origin = 30
    calendar = same_calendar_rows(calendar_of(), origin + 2, origin + 1)
    result = run(random_walk_tables(), price_levels(), calendar, origins=[origin],
                 horizons=2, with_level=False)
    h1, h2 = row(result, "TR02", origin, 1), row(result, "TR02", origin, 2)
    assert h1["refunds"] == pytest.approx(h2["refunds"], rel=1e-12)
    assert h1["unidentified_cash"] == pytest.approx(h2["unidentified_cash"], rel=1e-12)


def test_ridge_with_level_r_and_g_have_one_fit_per_h():
    # Same target-week calendar and the same level at the origin, yet h = 1 and h = 2 differ:
    # each h has its own fit (the training rows' level is taken h weeks before each week).
    origin = 30
    calendar = same_calendar_rows(calendar_of(), origin + 2, origin + 1)
    result = run(random_walk_tables(), price_levels(), calendar, origins=[origin],
                 horizons=2, with_level=True)
    h1, h2 = row(result, "TR02", origin, 1), row(result, "TR02", origin, 2)
    assert not h1["fallback"] and not h2["fallback"]
    assert h1["refunds"] != pytest.approx(h2["refunds"], rel=1e-9)
    assert h1["unidentified_cash"] != pytest.approx(h2["unidentified_cash"], rel=1e-9)


def test_ridge_a_uses_its_own_series_per_h():
    # A_1 and A_2 are different series: A_2 doubled changes h = 2 only.
    tables = side_tables()
    new_cash, refunds, unidentified = tables
    base = run(tables, origins=[30])
    doubled = new_cash.copy()
    doubled.loc[doubled["h"] == 2, "new_customer_cash"] *= 2.0
    again = run((doubled, refunds, unidentified), origins=[30])
    for h in (1, 3):
        pd.testing.assert_frame_equal(base[base["h"] == h], again[again["h"] == h])
    assert not np.allclose(base.loc[base["h"] == 2, "new_customer_cash"],
                           again.loc[again["h"] == 2, "new_customer_cash"])


# --- no leakage ------------------------------------------------------------------------------

def after_origin_changed(tables, levels, calendar, origin, horizons, which, seed=11):
    """Copies with everything after `origin` scrambled in `which` ('A', 'R', 'G', 'levels'
    or 'calendar'; the calendar only after the last target week, origin + horizons)."""
    rng = np.random.default_rng(seed)
    new_cash, refunds, unidentified = (t.copy() for t in tables)
    levels, calendar = levels.copy(), calendar.copy()
    if which == "A":
        late = new_cash["target_week"] > origin
        new_cash.loc[late, "new_customer_cash"] = rng.random(int(late.sum())) * 1e9
    elif which == "R":
        late = refunds["week"] > origin
        refunds.loc[late, "refunds"] = -rng.random(int(late.sum())) * 1e9
    elif which == "G":
        late = unidentified["week"] > origin
        unidentified.loc[late, "unidentified_cash"] = rng.random(int(late.sum())) * 1e9
    elif which == "levels":
        late = levels.index > origin
        levels[late] = 1.0 + rng.random(int(late.sum())) * 50.0
    elif which == "calendar":
        late = calendar["week"] > origin + horizons
        calendar.loc[late, ["cycle", "wave"]] = rng.random((int(late.sum()), 2)) * 1e6
    return (new_cash, refunds, unidentified), levels, calendar


@pytest.mark.parametrize("which", ["A", "R", "G", "levels", "calendar"])
@pytest.mark.parametrize("method,with_level", [("mean", True), ("ridge", True),
                                               ("ridge", False)])
def test_changing_anything_after_the_origin_does_not_change_the_forecast(which, method,
                                                                         with_level):
    tables, levels, calendar = side_tables(), price_levels(), calendar_of()
    for origin in (20, 31):
        base = run(tables, levels, calendar, origins=[origin], method=method,
                   with_level=with_level)
        changed = after_origin_changed(tables, levels, calendar, origin, 3, which)
        again = run(*changed, origins=[origin], method=method, with_level=with_level)
        pd.testing.assert_frame_equal(base, again)


@pytest.mark.parametrize("method,with_level", [("mean", True), ("ridge", True),
                                               ("ridge", False)])
def test_dropping_every_row_after_the_origin_does_not_change_the_forecast(method, with_level):
    tables = side_tables()
    origin = 28
    base = run(tables, origins=[origin], method=method, with_level=with_level)
    new_cash, refunds, unidentified = tables
    cut = (new_cash[new_cash["target_week"] <= origin], refunds[refunds["week"] <= origin],
           unidentified[unidentified["week"] <= origin])
    pd.testing.assert_frame_equal(base, run(cut, origins=[origin], method=method,
                                            with_level=with_level))


@pytest.mark.parametrize("which,column", [("A", "new_customer_cash"), ("R", "refunds"),
                                          ("G", "unidentified_cash")])
@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_a_week_up_to_the_origin_does_move_the_forecast(which, column, method):
    # The other side of the leak test: the data up to the origin is really read.
    tables = side_tables()
    origin = 30
    base = run(tables, origins=[origin], method=method)
    new_cash, refunds, unidentified = (t.copy() for t in tables)
    if which == "A":
        new_cash.loc[new_cash["target_week"] == origin, "new_customer_cash"] *= 3.0
    elif which == "R":
        refunds.loc[refunds["week"] == origin, "refunds"] *= 3.0
    else:
        unidentified.loc[unidentified["week"] == origin, "unidentified_cash"] *= 3.0
    again = run((new_cash, refunds, unidentified), origins=[origin], method=method)
    assert not np.allclose(base[column], again[column])


# --- per company -----------------------------------------------------------------------------

@pytest.mark.parametrize("method,with_level", [("mean", True), ("ridge", True),
                                               ("ridge", False)])
def test_each_company_is_forecast_on_its_own_series(method, with_level):
    tables = side_tables(companies=("TR02", "TR04", "TR05"))
    together = run(tables, origins=[25, 33], method=method, with_level=with_level)
    alone = run(tuple(t[t["company_code"] == "TR04"] for t in tables), origins=[25, 33],
                method=method, with_level=with_level)
    pd.testing.assert_frame_equal(
        together[together["company_code"] == "TR04"].reset_index(drop=True), alone)


@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_another_companys_data_does_not_move_a_forecast(method):
    tables = side_tables()
    base = run(tables, origins=[30], method=method)
    scaled = []
    for table, column in zip(tables, ("new_customer_cash", "refunds", "unidentified_cash")):
        table = table.copy()
        table.loc[table["company_code"] == "TR04", column] *= 1e6
        scaled.append(table)
    again = run(tuple(scaled), origins=[30], method=method)
    pd.testing.assert_frame_equal(base[base["company_code"] == "TR02"],
                                  again[again["company_code"] == "TR02"])
    assert not np.allclose(base.loc[base["company_code"] == "TR04", "topup"],
                           again.loc[again["company_code"] == "TR04", "topup"])


def test_companies_are_the_union_of_the_three_tables():
    new_cash, refunds, unidentified = side_tables(companies=("TR02",))
    extra = side_tables(companies=("TR09",), seed=4)
    tables = tuple(pd.concat([mine, theirs], ignore_index=True)
                   for mine, theirs in zip((new_cash, refunds, unidentified), extra))
    result = run(tables, origins=[30], method="mean")
    assert result["company_code"].unique().tolist() == ["TR02", "TR09"]


def test_a_company_missing_from_one_table_is_refused():
    # side_tables.py gives every company rows in all three tables. A company with refunds
    # but no A rows gets an empty A series, and its forecast is refused, not filled with 0.
    new_cash, refunds, unidentified = side_tables()
    new_cash = new_cash[new_cash["company_code"] != "TR04"]
    for method in ("mean", "ridge"):
        with pytest.raises(ValueError, match="no week with data"):
            run((new_cash, refunds, unidentified), origins=[30], method=method)


# --- fallback --------------------------------------------------------------------------------

@pytest.mark.parametrize("with_level", [True, False])
def test_ridge_an_all_zero_series_is_forecast_zero_and_not_flagged(with_level):
    # The ridge itself falls back on it (a prediction of 0 is not above 0), but 0 is the right
    # answer: the flag is not set.
    new_cash, refunds, unidentified = side_tables()
    unidentified = unidentified.copy()
    unidentified.loc[unidentified["company_code"] == "TR04", "unidentified_cash"] = 0.0
    result = run((new_cash, refunds, unidentified), origins=[25, 35], with_level=with_level)
    mine = result[result["company_code"] == "TR04"]
    assert (mine["unidentified_cash"] == 0.0).all()
    assert not mine["fallback"].any()
    # deflated_forecasts on its own does report the fallback: the unflagging is topups'.
    levels, calendar = price_levels(), calendar_of()
    g = deflated_series(unidentified, "TR04", "week", "unidentified_cash", levels)
    for h in (1, 2, 3):
        assert deflated_forecasts(g, calendar, h, 35, [35], with_level)["fallback"].iloc[0]
    assert not result.loc[result["company_code"] == "TR02", "fallback"].any()


def test_ridge_all_zero_refunds_are_forecast_zero_not_negative_and_not_flagged():
    new_cash, refunds, unidentified = side_tables(companies=("TR02",))
    refunds = refunds.assign(refunds=0.0)
    result = run((new_cash, refunds, unidentified), origins=[30])
    assert (result["refunds"] == 0.0).all()
    assert not result["fallback"].any()


def test_ridge_all_zero_a_is_forecast_zero_and_not_flagged():
    new_cash, refunds, unidentified = side_tables(companies=("TR02",))
    new_cash = new_cash.assign(new_customer_cash=0.0)
    result = run((new_cash, refunds, unidentified), origins=[30])
    assert (result["new_customer_cash"] == 0.0).all()
    assert not result["fallback"].any()


def test_ridge_zero_up_to_the_origin_and_nonzero_after_is_not_flagged():
    # Only the weeks up to the origin count: large cash after it changes neither the forecast
    # (0) nor the flag.
    new_cash, refunds, unidentified = side_tables(companies=("TR02",))
    unidentified = unidentified.assign(
        unidentified_cash=np.where(unidentified["week"] > 30, 5e6, 0.0))
    result = run((new_cash, refunds, unidentified), origins=[30])
    assert (result["unidentified_cash"] == 0.0).all()
    assert not result["fallback"].any()
    # A later origin sees the nonzero weeks: no longer a zero series.
    later = run((new_cash, refunds, unidentified), origins=[38])
    assert (later["unidentified_cash"] > 0).all()


def test_ridge_nonzero_only_before_the_level_window_is_still_flagged():
    # G is nonzero in weeks 0..2 and zero in 3..10: the recent level at origin 10 is 0, so the
    # forecast is 0, but the series is not all zero up to the origin, and its fallback (11
    # weeks, fewer than MIN_TRAINING_WEEKS) is a real one: flagged.
    assert 11 < MIN_TRAINING_WEEKS
    new_cash, refunds, _ = side_tables(companies=("TR02",), last=30, horizons=1)
    unidentified = pd.DataFrame({"company_code": "TR02", "week": range(0, 11),
                                 "unidentified_cash": [7.0, 3.0, 9.0] + [0.0] * 8})
    result = run((new_cash, refunds, unidentified), origins=[10], horizons=1)
    got = row(result, "TR02", 10, 1)
    assert got["unidentified_cash"] == 0.0
    assert got["fallback"]


def test_ridge_a_prediction_not_above_zero_is_flagged_even_when_the_level_is_zero():
    # G = 1000 in weeks with cycle 1..3 and -3000 in cycle-0 weeks: the ridge predicts about
    # -3000 for target week 32 (cycle 0), not above 0, so it falls back to the level of weeks
    # 28..31, (-3000 + 3 * 1000) / 4 = 0. A real fallback on a nonzero series: flagged.
    weeks = list(range(0, 41))
    levels = pd.Series(1.0, index=weeks + list(range(41, 61)))
    unidentified = pd.DataFrame({"company_code": "TR02", "week": weeks,
                                 "unidentified_cash": [-3000.0 if w % 4 == 0 else 1000.0
                                                       for w in weeks]})
    new_cash, refunds, _ = side_tables(companies=("TR02",), horizons=1)
    result = run((new_cash, refunds, unidentified), levels, origins=[31], horizons=1,
                 with_level=False)
    got = row(result, "TR02", 31, 1)
    assert got["unidentified_cash"] == pytest.approx(0.0, abs=1e-9)
    assert got["fallback"]
    # A and R keep their fit: the flag comes from G alone (the OR over the three).
    a = deflated_series(new_cash, "TR02", "target_week", "new_customer_cash", levels, h=1)
    minus_r = -deflated_series(refunds, "TR02", "week", "refunds", levels)
    assert not deflated_forecasts(a, calendar_of(), 1, 31, [31], False)["fallback"].iloc[0]
    assert not deflated_forecasts(minus_r, calendar_of(), 1, 31, [31], False)["fallback"].iloc[0]


def test_the_row_flag_is_the_or_over_a_r_and_g():
    # TR02: only its G falls back for a real reason (8 weeks of data); TR04: an all-zero G,
    # not flagged; every A and R keeps its fit.
    new_cash, refunds, unidentified = side_tables()
    unidentified = unidentified.copy()
    unidentified = unidentified[(unidentified["company_code"] != "TR02")
                                | (unidentified["week"] > 22)]
    unidentified.loc[unidentified["company_code"] == "TR04", "unidentified_cash"] = 0.0
    result = run((new_cash, refunds, unidentified), origins=[30])
    assert result.loc[result["company_code"] == "TR02", "fallback"].all()
    assert not result.loc[result["company_code"] == "TR04", "fallback"].any()


def test_ridge_too_few_training_weeks_falls_back_to_the_recent_level():
    # MIN_TRAINING_WEEKS - 1 weeks of G: no fit, the forecast is the mean of the last 4 weeks.
    weeks = list(range(0, MIN_TRAINING_WEEKS - 1))
    levels = pd.Series(4.0, index=range(0, 40))
    unidentified = pd.DataFrame({"company_code": "TR02", "week": weeks,
                                 "unidentified_cash": [4.0 * (w + 1) for w in weeks]})
    new_cash, refunds, _ = side_tables(companies=("TR02",), last=30, horizons=1)
    origin = weeks[-1]
    result = run((new_cash, refunds, unidentified), levels, origins=[origin], horizons=1)
    got = row(result, "TR02", origin, 1)
    # Deflated G = w + 1; the last 4 weeks are origin-3 .. origin: mean origin - 0.5.
    assert got["unidentified_cash"] == pytest.approx(origin - 0.5)
    assert got["fallback"]


def test_mean_never_falls_back_even_on_an_all_zero_series():
    new_cash, refunds, unidentified = side_tables()
    unidentified = unidentified.assign(unidentified_cash=0.0)
    refunds = refunds.assign(refunds=0.0)
    result = run((new_cash, refunds, unidentified), origins=[20, 30], method="mean")
    assert (result["unidentified_cash"] == 0.0).all()
    assert (result["refunds"] == 0.0).all()
    assert result["fallback"].dtype == bool
    assert not result["fallback"].any()


def test_fallback_is_a_bool_column_for_the_ridge():
    assert run(origins=[30])["fallback"].dtype == bool


# --- the top-up, columns, order --------------------------------------------------------------

@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_topup_is_the_sum_of_the_three(method):
    result = run(origins=[20, 30, 40], method=method)
    assert result["topup"].to_numpy() == pytest.approx(
        (result["new_customer_cash"] + result["refunds"]
         + result["unidentified_cash"]).to_numpy(), rel=1e-15)
    # Refunds are negative, so the top-up is below A + G.
    assert (result["topup"] < result["new_customer_cash"] + result["unidentified_cash"]).all()


@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_columns_order_and_sorting(method):
    tables = side_tables(companies=("TR05", "TR02", "TR04"))
    # Shuffled rows and unsorted origins: the output must not depend on the input order.
    rng = np.random.default_rng(5)
    shuffled = tuple(t.iloc[rng.permutation(len(t))] for t in tables)
    result = run(shuffled, origins=[33, 21, 27], horizons=3, method=method)
    assert list(result.columns) == TOPUP_COLUMNS
    assert len(result) == 3 * 3 * 3
    expected_keys = [(c, o, h) for c in ("TR02", "TR04", "TR05") for o in (21, 27, 33)
                     for h in (1, 2, 3)]
    assert list(zip(result["company_code"], result["origin_week"], result["h"])) == \
        expected_keys
    assert (result["target_week"] == result["origin_week"] + result["h"]).all()
    assert result.index.tolist() == list(range(len(result)))
    pd.testing.assert_frame_equal(result, run(tables, origins=[21, 27, 33], method=method))


def test_output_dtypes():
    result = run(origins=[25, 30])
    for column in ("origin_week", "h", "target_week"):
        assert pd.api.types.is_integer_dtype(result[column]), column
    for column in ("new_customer_cash", "refunds", "unidentified_cash", "topup"):
        assert result[column].dtype == "float64", column


@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_no_origins_give_an_empty_table_with_the_columns(method):
    result = run(origins=[], method=method)
    assert len(result) == 0
    assert list(result.columns) == TOPUP_COLUMNS


# --- inputs untouched, deterministic ---------------------------------------------------------

@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_topup_forecasts_does_not_change_its_inputs(method):
    tables, levels, calendar = side_tables(), price_levels(), calendar_of()
    copies = [t.copy(deep=True) for t in tables]
    levels_copy, calendar_copy = levels.copy(deep=True), calendar.copy(deep=True)
    origins = [30, 20]
    topup_forecasts(*tables, levels, calendar, origins, 3, method, 8, True)
    for table, copy in zip(tables, copies):
        pd.testing.assert_frame_equal(table, copy)
    pd.testing.assert_series_equal(levels, levels_copy)
    pd.testing.assert_frame_equal(calendar, calendar_copy)
    assert origins == [30, 20]


def test_topup_forecasts_is_deterministic():
    pd.testing.assert_frame_equal(run(origins=[25, 35]), run(origins=[25, 35]))


# --- refusals --------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["Mean", "RIDGE", "", "trailing", "deflated"])
def test_refuses_an_unknown_method(method):
    with pytest.raises(ValueError) as error:
        run(method=method)
    assert str(error.value) == f"method is {method!r}; it must be one of ('mean', 'ridge')"


def test_refuses_an_unknown_method_even_without_origins():
    with pytest.raises(ValueError, match="method is 'x'"):
        run(origins=[], method="x")


def test_an_unknown_method_is_reported_before_a_bad_window():
    with pytest.raises(ValueError) as error:
        run(method="x", window=0)
    assert str(error.value).startswith("method is 'x'")


@pytest.mark.parametrize("window", [0, -1, -8])
@pytest.mark.parametrize("origins", [(30,), ()])
def test_mean_refuses_a_window_below_one_even_without_origins(window, origins):
    with pytest.raises(ValueError) as error:
        run(origins=origins, method="mean", window=window)
    assert str(error.value) == f"window is {window}; the trailing mean needs at least 1 week"


@pytest.mark.parametrize("window", [0, -1])
def test_ridge_ignores_the_window(window):
    pd.testing.assert_frame_equal(run(window=window), run(window=8))


@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_a_week_without_a_price_level_is_refused(method):
    levels = price_levels().drop(index=[17])
    with pytest.raises(ValueError, match=r"no price level for weeks \[17\]"):
        run(levels=levels, method=method)


def test_ridge_refuses_a_calendar_without_the_target_week():
    calendar = calendar_of()
    calendar = calendar[calendar["week"] != 32]
    with pytest.raises(ValueError, match=r"the calendar has no rows for weeks \[32\]"):
        run(calendar=calendar, origins=[30], horizons=2)


def test_ridge_refuses_a_calendar_without_a_training_week():
    calendar = calendar_of()
    calendar = calendar[calendar["week"] != 10]
    with pytest.raises(ValueError, match="the calendar has no rows for weeks"):
        run(calendar=calendar, origins=[30])


def late_only(tables, first):
    return tuple(t[t[c] >= first] for t, c in zip(tables, ("target_week", "week", "week")))


def test_mean_refuses_an_origin_without_data_in_its_window():
    with pytest.raises(ValueError, match="no week with data in the 4 weeks up to the origin"):
        run(late_only(side_tables(), 30), origins=[20], method="mean", window=4)


def test_ridge_refuses_an_origin_without_a_recent_level():
    with pytest.raises(ValueError, match="so no recent level"):
        run(late_only(side_tables(), 30), origins=[20])


@pytest.mark.parametrize("horizons", [0, -1, -5])
@pytest.mark.parametrize("method", ["mean", "ridge"])
@pytest.mark.parametrize("origins", [(30,), ()])
def test_refuses_horizons_below_one_even_without_origins(horizons, method, origins):
    with pytest.raises(ValueError) as error:
        run(origins=origins, horizons=horizons, method=method)
    assert str(error.value) == f"horizons is {horizons}; horizons start at 1"


def test_an_unknown_method_is_reported_before_bad_horizons():
    with pytest.raises(ValueError) as error:
        run(method="x", horizons=0)
    assert str(error.value).startswith("method is 'x'")


def test_bad_horizons_are_reported_before_a_bad_window():
    with pytest.raises(ValueError) as error:
        run(method="mean", horizons=0, window=0)
    assert str(error.value) == "horizons is 0; horizons start at 1"


def no_company_tables():
    return tuple(t.iloc[0:0] for t in side_tables())


@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_refuses_side_tables_with_no_company(method):
    with pytest.raises(ValueError) as error:
        run(no_company_tables(), origins=[30], method=method)
    assert str(error.value) == "the side tables hold no company, so there is nothing to forecast"


@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_no_company_and_no_origins_give_the_empty_table(method):
    result = run(no_company_tables(), origins=[], method=method)
    assert len(result) == 0
    assert list(result.columns) == TOPUP_COLUMNS


# =============================================================================================
# U12.3: `topup_actuals`, what the top-ups really were (hindsight), and `score_topups`, how far
# a method's forecasts were from them.
#
# The actuals reuse the hand-built side tables above (`hand_tables`, `side_tables`). The scores
# are checked on small forecast and actual tables written out by hand, so every mean absolute
# error and bias can be worked out on paper.
# =============================================================================================

from collection_estimation.hurdle.topups import (  # noqa: E402
    SCORE_COLUMNS,
    SCORED_SERIES,
    score_topups,
    topup_actuals,
)

ACTUAL_COLUMNS = ["company_code", "origin_week", "h", "target_week", "new_customer_cash",
                  "refunds", "unidentified_cash", "topup"]
KEYS = ["company_code", "origin_week", "h"]


def actuals_of(tables=None, levels=None, origins=(30,), horizons=3):
    new_cash, refunds, unidentified = tables if tables is not None else side_tables()
    return topup_actuals(new_cash, refunds, unidentified,
                         price_levels() if levels is None else levels, list(origins), horizons)


def keys_of(table):
    return list(zip(table["company_code"], table["origin_week"], table["h"]))


# --- topup_actuals: constants and shape ------------------------------------------------------

def test_actuals_columns_are_the_topup_columns_without_fallback():
    assert TOPUP_COLUMNS[:-1] == ACTUAL_COLUMNS
    result = actuals_of(origins=[20, 30])
    assert list(result.columns) == ACTUAL_COLUMNS
    assert "fallback" not in result.columns


def test_actuals_output_dtypes():
    result = actuals_of(origins=[20, 30])
    for column in ("origin_week", "h", "target_week"):
        assert pd.api.types.is_integer_dtype(result[column]), column
    for column in ("new_customer_cash", "refunds", "unidentified_cash", "topup"):
        assert result[column].dtype == "float64", column


def test_actuals_one_row_per_company_origin_and_h_sorted():
    tables = side_tables(companies=("TR05", "TR02", "TR04"))
    rng = np.random.default_rng(9)
    shuffled = tuple(t.iloc[rng.permutation(len(t))] for t in tables)
    result = actuals_of(shuffled, origins=[33, 21, 27], horizons=3)
    assert keys_of(result) == [(c, o, h) for c in ("TR02", "TR04", "TR05")
                               for o in (21, 27, 33) for h in (1, 2, 3)]
    assert (result["target_week"] == result["origin_week"] + result["h"]).all()
    assert result.index.tolist() == list(range(len(result)))
    pd.testing.assert_frame_equal(result, actuals_of(tables, origins=[21, 27, 33]))


def test_actuals_no_origins_give_an_empty_table_with_the_columns():
    result = actuals_of(origins=[])
    assert len(result) == 0
    assert list(result.columns) == ACTUAL_COLUMNS


def test_actuals_an_origin_not_in_the_side_tables_gives_no_row():
    # Origin 200 is far past the data: no A row, so no actual (not a row of zeros).
    result = actuals_of(origins=[30, 200])
    assert set(result["origin_week"]) == {30}


def test_actuals_a_repeated_origin_gives_its_rows_once():
    once = actuals_of(origins=[30])
    pd.testing.assert_frame_equal(actuals_of(origins=[30, 30]), once)


# --- topup_actuals: by hand, with deflation by the target week -------------------------------

def test_actuals_by_hand_with_deflation():
    # hand_tables: levels 6 -> 2, 7 -> 4, 8 -> 0.5, 9 -> 1.25, 10 -> 2.5.
    tables, levels = hand_tables()
    result = actuals_of(tables, levels, origins=[7, 8], horizons=2)
    expected = pd.DataFrame({
        "company_code": "TR02",
        "origin_week": [7, 7, 8, 8], "h": [1, 2, 1, 2], "target_week": [8, 9, 9, 10],
        # A: origin 7 h 1 -> 1 / 0.5; 7 h 2 -> 2.5 / 1.25; 8 h 1 -> 5 / 1.25; 8 h 2 -> 25 / 2.5.
        "new_customer_cash": [2.0, 2.0, 4.0, 10.0],
        # R of the target week: 8 -> -1 / 0.5; 9 -> -5 / 1.25; 10 -> -5 / 2.5.
        "refunds": [-2.0, -4.0, -4.0, -2.0],
        # G of the target week: 8 -> 3 / 0.5; 9 -> 0 / 1.25; 10 -> 7.5 / 2.5.
        "unidentified_cash": [6.0, 0.0, 0.0, 3.0],
        "topup": [6.0, -2.0, 0.0, 11.0],
    })
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_actuals_deflate_by_the_target_week_not_the_origin_week():
    tables, levels = hand_tables()
    got = actuals_of(tables, levels, origins=[9], horizons=1).iloc[0]
    # Target week 10 (level 2.5): A 10 / 2.5 = 4. By origin week 9 (1.25) it would be 8.
    assert got["target_week"] == 10
    assert got["new_customer_cash"] == pytest.approx(4.0)
    assert got["refunds"] == pytest.approx(-2.0)
    assert got["unidentified_cash"] == pytest.approx(3.0)
    assert got["topup"] == pytest.approx(5.0)


def test_actuals_equal_deflate_of_each_row():
    tables = side_tables()
    new_cash, refunds, unidentified = tables
    levels = price_levels()
    result = actuals_of(tables, levels, origins=[12, 25, 37], horizons=3)
    assert len(result) == 2 * 3 * 3
    for _, got in result.iterrows():
        company, origin, h, target = (got["company_code"], got["origin_week"], got["h"],
                                      got["target_week"])
        a = new_cash[(new_cash["company_code"] == company) & (new_cash["origin_week"] == origin)
                     & (new_cash["h"] == h)]["new_customer_cash"].item()
        r = refunds[(refunds["company_code"] == company)
                    & (refunds["week"] == target)]["refunds"].item()
        g = unidentified[(unidentified["company_code"] == company)
                         & (unidentified["week"] == target)]["unidentified_cash"].item()
        level = levels[target]
        assert got["new_customer_cash"] == pytest.approx(a / level, rel=1e-14)
        assert got["refunds"] == pytest.approx(r / level, rel=1e-14)
        assert got["unidentified_cash"] == pytest.approx(g / level, rel=1e-14)
        assert got["topup"] == pytest.approx((a + r + g) / level, rel=1e-12)


def test_actuals_refunds_stay_negative():
    result = actuals_of(origins=list(range(5, 38)))
    assert (result["refunds"] < 0).all()


def test_actuals_r_and_g_are_the_target_weeks_whatever_the_origin():
    # R and G do not depend on the origin: (origin 29, h 2) and (origin 30, h 1) share target
    # week 31, and so their R and G; their A differ (A_2 and A_1).
    result = actuals_of(origins=[29, 30], horizons=2)
    for company in COMPANIES:
        first, second = row(result, company, 29, 2), row(result, company, 30, 1)
        assert first["target_week"] == second["target_week"] == 31
        assert first["refunds"] == second["refunds"]
        assert first["unidentified_cash"] == second["unidentified_cash"]
        assert first["new_customer_cash"] != second["new_customer_cash"]


def test_actuals_join_r_and_g_by_company_and_week():
    # TR04's R and G are far larger than TR02's: a join on the week alone would mix them.
    result = actuals_of(origins=[30], horizons=1)
    tr02, tr04 = row(result, "TR02", 30, 1), row(result, "TR04", 30, 1)
    _, refunds, unidentified = side_tables()
    level = price_levels()[31]
    for company, got in (("TR02", tr02), ("TR04", tr04)):
        r = refunds[(refunds["company_code"] == company) & (refunds["week"] == 31)]
        g = unidentified[(unidentified["company_code"] == company)
                         & (unidentified["week"] == 31)]
        assert got["refunds"] == pytest.approx(r["refunds"].item() / level)
        assert got["unidentified_cash"] == pytest.approx(g["unidentified_cash"].item() / level)


def test_actuals_h_above_horizons_is_left_out():
    result = actuals_of(origins=[30], horizons=2)
    assert sorted(set(result["h"])) == [1, 2]
    assert len(actuals_of(origins=[30], horizons=3)) == 2 * 3


def test_actuals_horizons_beyond_the_side_tables_give_only_the_h_they_hold():
    # The side tables hold h 1..3; asking for 5 gives those three (nothing is invented).
    result = actuals_of(origins=[30], horizons=5)
    assert sorted(set(result["h"])) == [1, 2, 3]


# --- topup_actuals: no data is not 0 ---------------------------------------------------------

def test_actuals_a_target_week_without_refunds_drops_the_row_not_counts_zero():
    tables, levels = hand_tables()
    new_cash, refunds, unidentified = tables
    refunds = refunds[refunds["week"] != 9]
    result = actuals_of((new_cash, refunds, unidentified), levels, origins=[7, 8], horizons=2)
    # Target week 9 is (7, 2) and (8, 1): both gone; (7, 1) and (8, 2) are untouched.
    assert keys_of(result) == [("TR02", 7, 1), ("TR02", 8, 2)]
    assert result["topup"].tolist() == pytest.approx([6.0, 11.0])


def test_actuals_a_target_week_without_unidentified_cash_drops_the_row():
    tables, levels = hand_tables()
    new_cash, refunds, unidentified = tables
    unidentified = unidentified[unidentified["week"] != 10]
    result = actuals_of((new_cash, refunds, unidentified), levels, origins=[8, 9], horizons=2)
    assert 10 not in set(result["target_week"])
    assert keys_of(result) == [("TR02", 8, 1)]


def test_actuals_a_target_week_without_new_customer_cash_drops_the_row():
    tables, levels = hand_tables()
    new_cash, refunds, unidentified = tables
    new_cash = new_cash[~((new_cash["origin_week"] == 8) & (new_cash["h"] == 2))]
    result = actuals_of((new_cash, refunds, unidentified), levels, origins=[8], horizons=2)
    assert keys_of(result) == [("TR02", 8, 1)]


def test_actuals_a_zero_week_is_kept_as_zero():
    # G of week 9 is 0.0 (no cash that week): the row is kept, with G 0. Missing, it is dropped.
    tables, levels = hand_tables()
    result = actuals_of(tables, levels, origins=[8], horizons=1)
    assert keys_of(result) == [("TR02", 8, 1)]
    assert result["unidentified_cash"].iloc[0] == 0.0
    new_cash, refunds, unidentified = tables
    new_cash = new_cash.assign(new_customer_cash=0.0)
    zero_a = actuals_of((new_cash, refunds, unidentified), levels, origins=[8], horizons=1)
    assert zero_a["new_customer_cash"].tolist() == [0.0]
    assert zero_a["topup"].iloc[0] == pytest.approx(-4.0)


def test_actuals_a_week_missing_for_one_company_drops_only_its_rows():
    new_cash, refunds, unidentified = side_tables()
    refunds = refunds[~((refunds["company_code"] == "TR04") & (refunds["week"] == 31))]
    result = actuals_of((new_cash, refunds, unidentified), origins=[30], horizons=2)
    assert keys_of(result) == [("TR02", 30, 1), ("TR02", 30, 2), ("TR04", 30, 2)]


def test_actuals_past_the_last_week_are_dropped():
    # side_tables' last week is 40: origin 39 has only h 1 (target 40).
    result = actuals_of(origins=[38, 39], horizons=3)
    assert set(zip(result["origin_week"], result["h"])) == {(38, 1), (38, 2), (39, 1)}


# --- topup_actuals agrees with topup_forecasts' rows -----------------------------------------

@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_actuals_have_the_forecasts_keys_when_every_target_week_has_data(method):
    origins = [15, 22, 30, 37]  # 37 + 3 = 40, the last week
    forecasts = run(origins=origins, method=method)
    actuals = actuals_of(origins=origins, horizons=3)
    assert keys_of(actuals) == keys_of(forecasts)
    assert actuals["target_week"].tolist() == forecasts["target_week"].tolist()
    assert list(actuals.columns) == list(forecasts.columns[:-1])


def test_actuals_keys_are_a_subset_of_the_forecasts_keys_past_the_last_week():
    origins = [30, 38, 39, 40]
    forecasts = run(origins=origins, method="mean")
    actuals = actuals_of(origins=origins, horizons=3)
    assert set(keys_of(actuals)) < set(keys_of(forecasts))
    assert len(forecasts) == 2 * 4 * 3
    assert len(actuals) == 2 * (3 + 2 + 1)


def test_actuals_differ_from_the_forecasts_made_without_them():
    # A sanity check that the actuals are not the forecasts under another name.
    forecasts = run(origins=[30], method="mean")
    actuals = actuals_of(origins=[30])
    assert not np.allclose(forecasts["topup"], actuals["topup"])


# --- topup_actuals: inputs untouched, deterministic, refusals --------------------------------

def test_topup_actuals_does_not_change_its_inputs():
    tables, levels = side_tables(), price_levels()
    copies = [t.copy(deep=True) for t in tables]
    levels_copy = levels.copy(deep=True)
    origins = [30, 20]
    topup_actuals(*tables, levels, origins, 3)
    for table, copy in zip(tables, copies):
        pd.testing.assert_frame_equal(table, copy)
    pd.testing.assert_series_equal(levels, levels_copy)
    assert origins == [30, 20]


def test_topup_actuals_is_deterministic():
    pd.testing.assert_frame_equal(actuals_of(origins=[20, 30]), actuals_of(origins=[20, 30]))


@pytest.mark.parametrize("horizons", [0, -1, -5])
@pytest.mark.parametrize("origins", [(30,), ()])
def test_actuals_refuse_horizons_below_one_even_without_origins(horizons, origins):
    with pytest.raises(ValueError) as error:
        actuals_of(origins=origins, horizons=horizons)
    assert str(error.value) == f"horizons is {horizons}; horizons start at 1"


def test_actuals_refuse_a_target_week_without_a_price_level():
    levels = price_levels().drop(index=[31])
    with pytest.raises(ValueError, match=r"no price level for weeks \[31\]"):
        actuals_of(levels=levels, origins=[30])


def test_actuals_do_not_need_price_levels_of_weeks_they_do_not_use():
    levels = price_levels().drop(index=[5, 17])
    assert len(actuals_of(levels=levels, origins=[30])) == 2 * 3


def test_actuals_refuse_a_refunds_table_with_a_week_twice():
    new_cash, refunds, unidentified = side_tables()
    refunds = pd.concat([refunds, refunds.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        actuals_of((new_cash, refunds, unidentified))


def test_actuals_refuse_an_unidentified_table_with_a_week_twice():
    new_cash, refunds, unidentified = side_tables()
    unidentified = pd.concat([unidentified, unidentified.iloc[[5]]], ignore_index=True)
    with pytest.raises(ValueError):
        actuals_of((new_cash, refunds, unidentified))


# --- score_topups: constants ------------------------------------------------------------------

def test_scored_series():
    assert SCORED_SERIES == ["new_customer_cash", "refunds", "unidentified_cash", "topup"]


def test_score_columns():
    assert SCORE_COLUMNS == ["h", "series", "mean_abs_error", "bias", "forecasts"]


# --- score_topups: by hand --------------------------------------------------------------------

def scored_table(rows):
    """rows: (company, origin, h, A, R, G); the top-up is their sum. Columns as from
    topup_actuals (target week origin + h)."""
    table = pd.DataFrame(rows, columns=["company_code", "origin_week", "h",
                                        "new_customer_cash", "refunds", "unidentified_cash"])
    table.insert(3, "target_week", table["origin_week"] + table["h"])
    table["topup"] = table["new_customer_cash"] + table["refunds"] + table["unidentified_cash"]
    return table


# Two companies, origins 10 and 11, h 1. Top-ups: forecast TR02 5, 3 / TR04 1, 4; actual
# TR02 4, 6 / TR04 3, 1. Errors TR02 +1, -3; TR04 -2, +3; consolidated (6 - 7, 7 - 7) -1, 0.
HAND_FORECASTS = scored_table([
    ("TR02", 10, 1, 5.0, -1.0, 1.0), ("TR02", 11, 1, 2.0, -1.0, 2.0),
    ("TR04", 10, 1, 1.0, -2.0, 2.0), ("TR04", 11, 1, 4.0, -1.0, 1.0),
])
HAND_ACTUALS = scored_table([
    ("TR02", 10, 1, 4.0, -2.0, 2.0), ("TR02", 11, 1, 6.0, -1.0, 1.0),
    ("TR04", 10, 1, 2.0, -1.0, 2.0), ("TR04", 11, 1, 1.0, -2.0, 2.0),
])


def score_of(scores, series, company=None, h=1):
    found = scores[(scores["series"] == series) & (scores["h"] == h)]
    if company is not None:
        found = found[found["company_code"] == company]
    assert len(found) == 1
    return found.iloc[0]


def test_hand_topups_are_as_described():
    assert HAND_FORECASTS["topup"].tolist() == [5.0, 3.0, 1.0, 4.0]
    assert HAND_ACTUALS["topup"].tolist() == [4.0, 6.0, 3.0, 1.0]


def test_score_by_company_by_hand():
    scores = score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=True)
    # TR02: top-up errors +1, -3 -> MAE 2, bias -1. A errors +1, -4 -> MAE 2.5, bias -1.5.
    # R errors +1, 0 -> 0.5, 0.5. G errors -1, +1 -> MAE 1, bias 0.
    expected = {("TR02", "topup"): (2.0, -1.0), ("TR02", "new_customer_cash"): (2.5, -1.5),
                ("TR02", "refunds"): (0.5, 0.5), ("TR02", "unidentified_cash"): (1.0, 0.0),
                # TR04: top-up -2, +3 -> 2.5, 0.5. A -1, +3 -> 2, 1. R -1, +1 -> 1, 0.
                # G 0, -1 -> 0.5, -0.5.
                ("TR04", "topup"): (2.5, 0.5), ("TR04", "new_customer_cash"): (2.0, 1.0),
                ("TR04", "refunds"): (1.0, 0.0), ("TR04", "unidentified_cash"): (0.5, -0.5)}
    assert len(scores) == 8
    for (company, series), (mae, bias) in expected.items():
        got = score_of(scores, series, company)
        assert got["mean_abs_error"] == pytest.approx(mae), (company, series)
        assert got["bias"] == pytest.approx(bias), (company, series)
        assert got["forecasts"] == 2


def test_score_consolidated_by_hand():
    scores = score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=False)
    # Summed per origin: forecast top-up 6, 7; actual 7, 7 -> errors -1, 0: MAE 0.5, bias -0.5.
    # A: 6 - 6, 6 - 7 -> 0, -1. R: -3 + 3, -2 + 3 -> 0, +1. G: 3 - 4, 3 - 3 -> -1, 0.
    expected = {"topup": (0.5, -0.5), "new_customer_cash": (0.5, -0.5),
                "refunds": (0.5, 0.5), "unidentified_cash": (0.5, -0.5)}
    assert len(scores) == 4
    for series, (mae, bias) in expected.items():
        got = score_of(scores, series)
        assert got["mean_abs_error"] == pytest.approx(mae), series
        assert got["bias"] == pytest.approx(bias), series
        assert got["forecasts"] == 2


def test_consolidated_error_is_not_the_mean_or_sum_of_the_company_errors():
    # Errors of opposite signs in two companies cancel once the companies are added up.
    by_company = score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=True)
    consolidated = score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=False)
    company_mae = by_company.loc[by_company["series"] == "topup", "mean_abs_error"]
    total_mae = score_of(consolidated, "topup")["mean_abs_error"]
    assert company_mae.tolist() == pytest.approx([2.0, 2.5])
    assert total_mae == pytest.approx(0.5)
    assert total_mae != pytest.approx(company_mae.mean())
    assert total_mae != pytest.approx(company_mae.sum())
    assert total_mae < company_mae.min()


def test_consolidated_bias_is_the_sum_of_the_company_biases_when_every_row_matches():
    # The bias is linear: with every (company, origin, h) in both, consolidated = sum.
    by_company = score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=True)
    consolidated = score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=False)
    for series in SCORED_SERIES:
        company_bias = by_company.loc[by_company["series"] == series, "bias"].sum()
        assert score_of(consolidated, series)["bias"] == pytest.approx(company_bias), series


@pytest.mark.parametrize("by_company", [True, False])
def test_bias_is_above_zero_when_the_forecasts_are_too_high(by_company):
    high = HAND_ACTUALS.copy()
    for series in SCORED_SERIES:
        high[series] = high[series] + 2.5
    scores = score_topups(high, HAND_ACTUALS, by_company=by_company)
    expected = 2.5 if by_company else 5.0  # two companies of +2.5 add up
    assert scores["bias"].tolist() == pytest.approx([expected] * len(scores))
    assert scores["mean_abs_error"].tolist() == pytest.approx([expected] * len(scores))
    low = score_topups(HAND_ACTUALS, high, by_company=by_company)
    assert low["bias"].tolist() == pytest.approx([-expected] * len(low))


@pytest.mark.parametrize("by_company", [True, False])
def test_a_perfect_forecast_scores_zero(by_company):
    scores = score_topups(HAND_ACTUALS, HAND_ACTUALS, by_company=by_company)
    assert (scores["mean_abs_error"] == 0.0).all()
    assert (scores["bias"] == 0.0).all()


def test_the_mean_abs_error_is_at_least_the_absolute_bias():
    forecasts = run(origins=[20, 25, 30, 35], method="mean")
    actuals = actuals_of(origins=[20, 25, 30, 35])
    for by_company in (True, False):
        scores = score_topups(forecasts, actuals, by_company=by_company)
        assert (scores["mean_abs_error"] >= scores["bias"].abs() - 1e-9).all()


def test_refund_errors_are_scored_on_the_signed_values():
    # R forecast -1, actual -3: the forecast is too high (fewer refunds), error +2.
    forecasts = scored_table([("TR02", 10, 1, 0.0, -1.0, 0.0)])
    actuals = scored_table([("TR02", 10, 1, 0.0, -3.0, 0.0)])
    got = score_of(score_topups(forecasts, actuals, by_company=True), "refunds", "TR02")
    assert got["bias"] == pytest.approx(2.0)
    assert got["mean_abs_error"] == pytest.approx(2.0)


# --- score_topups: which rows are scored -----------------------------------------------------

def test_only_rows_in_both_tables_are_scored_and_counted():
    forecasts = pd.concat([HAND_FORECASTS, scored_table([("TR02", 12, 1, 99.0, 0.0, 0.0)])],
                          ignore_index=True)
    actuals = pd.concat([HAND_ACTUALS, scored_table([("TR04", 13, 1, -99.0, 0.0, 0.0)])],
                        ignore_index=True)
    scores = score_topups(forecasts, actuals, by_company=True)
    pd.testing.assert_frame_equal(scores,
                                  score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=True))


def test_forecasts_counts_the_matched_origins_per_company_and_h():
    # TR02 has two origins matched at h 1, TR04 one (its origin 11 actual is missing).
    actuals = HAND_ACTUALS[~((HAND_ACTUALS["company_code"] == "TR04")
                             & (HAND_ACTUALS["origin_week"] == 11))]
    scores = score_topups(HAND_FORECASTS, actuals, by_company=True)
    assert set(scores.loc[scores["company_code"] == "TR02", "forecasts"]) == {2}
    assert set(scores.loc[scores["company_code"] == "TR04", "forecasts"]) == {1}
    # TR04's one error: top-up 1 - 3 = -2.
    assert score_of(scores, "topup", "TR04")["mean_abs_error"] == pytest.approx(2.0)


def test_consolidated_adds_up_only_the_companies_matched_at_each_origin():
    # Origin 11 has only TR02 in the actuals: its consolidated forecast and actual are TR02's
    # alone (3 against 6), not TR02 + TR04's forecast against TR02's actual.
    actuals = HAND_ACTUALS[~((HAND_ACTUALS["company_code"] == "TR04")
                             & (HAND_ACTUALS["origin_week"] == 11))]
    got = score_of(score_topups(HAND_FORECASTS, actuals, by_company=False), "topup")
    # Origin 10: 6 - 7 = -1; origin 11: 3 - 6 = -3 -> MAE 2, bias -2, 2 origins.
    assert got["mean_abs_error"] == pytest.approx(2.0)
    assert got["bias"] == pytest.approx(-2.0)
    assert got["forecasts"] == 2


def test_the_target_week_and_other_columns_are_not_read():
    forecasts = HAND_FORECASTS.assign(target_week=-1, fallback=True, extra="x")
    actuals = HAND_ACTUALS.assign(target_week=999)
    for by_company in (True, False):
        pd.testing.assert_frame_equal(
            score_topups(forecasts, actuals, by_company=by_company),
            score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=by_company))


def test_the_row_order_of_the_inputs_does_not_matter():
    rng = np.random.default_rng(4)
    forecasts = HAND_FORECASTS.iloc[rng.permutation(len(HAND_FORECASTS))]
    actuals = HAND_ACTUALS.iloc[rng.permutation(len(HAND_ACTUALS))].reset_index(drop=True)
    for by_company in (True, False):
        pd.testing.assert_frame_equal(
            score_topups(forecasts, actuals, by_company=by_company),
            score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=by_company))


# --- score_topups: shape ---------------------------------------------------------------------

def test_score_columns_by_company_and_consolidated():
    assert list(score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=True).columns) == \
        ["company_code"] + SCORE_COLUMNS
    assert list(score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=False).columns) == \
        SCORE_COLUMNS


def test_scores_are_sorted_by_company_h_and_series_in_scored_order():
    origins = [20, 30]
    forecasts = run(tables=side_tables(companies=("TR05", "TR02")), origins=origins,
                    method="mean")
    actuals = actuals_of(side_tables(companies=("TR05", "TR02")), origins=origins)
    scores = score_topups(forecasts, actuals, by_company=True)
    assert list(zip(scores["company_code"], scores["h"], scores["series"])) == [
        (c, h, s) for c in ("TR02", "TR05") for h in (1, 2, 3) for s in SCORED_SERIES]
    consolidated = score_topups(forecasts, actuals, by_company=False)
    assert list(zip(consolidated["h"], consolidated["series"])) == [
        (h, s) for h in (1, 2, 3) for s in SCORED_SERIES]
    assert consolidated.index.tolist() == list(range(len(consolidated)))


def test_score_dtypes():
    for by_company in (True, False):
        scores = score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=by_company)
        assert pd.api.types.is_integer_dtype(scores["h"])
        assert pd.api.types.is_integer_dtype(scores["forecasts"])
        assert scores["mean_abs_error"].dtype == "float64"
        assert scores["bias"].dtype == "float64"


def test_one_company_consolidated_equals_by_company_without_the_company_column():
    forecasts = run(tables=side_tables(companies=("TR02",)), origins=[20, 30], method="mean")
    actuals = actuals_of(side_tables(companies=("TR02",)), origins=[20, 30])
    by_company = score_topups(forecasts, actuals, by_company=True)
    consolidated = score_topups(forecasts, actuals, by_company=False)
    pd.testing.assert_frame_equal(by_company.drop(columns="company_code"), consolidated)


# --- score_topups end to end, on topup_forecasts and topup_actuals ---------------------------

@pytest.mark.parametrize("method", ["mean", "ridge"])
def test_scores_equal_a_direct_computation(method):
    origins = list(range(20, 38))
    forecasts = run(origins=origins, method=method)
    actuals = actuals_of(origins=origins)
    both = forecasts.merge(actuals, on=KEYS, suffixes=("_f", "_a"))
    scores = score_topups(forecasts, actuals, by_company=True)
    for (company, h), rows in both.groupby(["company_code", "h"]):
        for series in SCORED_SERIES:
            error = rows[f"{series}_f"] - rows[f"{series}_a"]
            got = score_of(scores, series, company, h)
            assert got["mean_abs_error"] == pytest.approx(float(np.mean(np.abs(error))),
                                                          rel=1e-12)
            assert got["bias"] == pytest.approx(float(np.mean(error)), rel=1e-9, abs=1e-9)
            assert got["forecasts"] == len(origins)
    consolidated = score_topups(forecasts, actuals, by_company=False)
    totals = both.groupby(["origin_week", "h"])[[f"topup_f", f"topup_a"]].sum().reset_index()
    for h, rows in totals.groupby("h"):
        error = rows["topup_f"] - rows["topup_a"]
        got = score_of(consolidated, "topup", h=h)
        assert got["mean_abs_error"] == pytest.approx(float(np.mean(np.abs(error))), rel=1e-12)
        assert got["forecasts"] == len(origins)


def test_scores_past_the_last_week_count_fewer_origins_at_longer_h():
    origins = [36, 37, 38, 39]  # targets past 40 have no actual
    forecasts = run(origins=origins, method="mean")
    scores = score_topups(forecasts, actuals_of(origins=origins), by_company=False)
    counts = scores.groupby("h")["forecasts"].first().to_dict()
    assert counts == {1: 4, 2: 3, 3: 2}


def test_score_topups_does_not_change_its_inputs():
    forecasts, actuals = HAND_FORECASTS.copy(deep=True), HAND_ACTUALS.copy(deep=True)
    for by_company in (True, False):
        score_topups(forecasts, actuals, by_company=by_company)
    pd.testing.assert_frame_equal(forecasts, HAND_FORECASTS)
    pd.testing.assert_frame_equal(actuals, HAND_ACTUALS)


# --- score_topups: refusals ------------------------------------------------------------------

@pytest.mark.parametrize("by_company", [True, False])
def test_score_refuses_a_forecast_row_twice(by_company):
    forecasts = pd.concat([HAND_FORECASTS, HAND_FORECASTS.iloc[[2]]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        score_topups(forecasts, HAND_ACTUALS, by_company=by_company)
    assert str(error.value) == "forecasts holds a (company, origin, h) more than once"


@pytest.mark.parametrize("by_company", [True, False])
def test_score_refuses_an_actual_row_twice(by_company):
    actuals = pd.concat([HAND_ACTUALS, HAND_ACTUALS.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        score_topups(HAND_FORECASTS, actuals, by_company=by_company)
    assert str(error.value) == "actuals holds a (company, origin, h) more than once"


def test_score_refuses_a_duplicate_key_even_with_different_values():
    twice = scored_table([("TR02", 10, 1, 1.0, 0.0, 0.0), ("TR02", 10, 1, 2.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="forecasts holds"):
        score_topups(twice, HAND_ACTUALS, by_company=True)


def test_score_refuses_a_duplicate_even_if_it_would_not_be_matched():
    # A duplicate key the other table does not have is still refused.
    extra = scored_table([("TR09", 50, 1, 1.0, 0.0, 0.0), ("TR09", 50, 1, 1.0, 0.0, 0.0)])
    actuals = pd.concat([HAND_ACTUALS, extra], ignore_index=True)
    with pytest.raises(ValueError, match="actuals holds"):
        score_topups(HAND_FORECASTS, actuals, by_company=True)


def test_score_forecast_duplicates_are_reported_before_actual_duplicates():
    forecasts = pd.concat([HAND_FORECASTS, HAND_FORECASTS.iloc[[0]]], ignore_index=True)
    actuals = pd.concat([HAND_ACTUALS, HAND_ACTUALS.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        score_topups(forecasts, actuals, by_company=True)
    assert str(error.value).startswith("forecasts holds")


def test_score_the_same_origin_and_h_in_two_companies_is_not_a_duplicate():
    # Consolidated adds them up; that is not a duplicate key.
    scores = score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=False)
    assert len(scores) == 4


@pytest.mark.parametrize("by_company", [True, False])
def test_score_refuses_no_overlap(by_company):
    actuals = HAND_ACTUALS.assign(origin_week=HAND_ACTUALS["origin_week"] + 100)
    with pytest.raises(ValueError) as error:
        score_topups(HAND_FORECASTS, actuals, by_company=by_company)
    assert str(error.value) == "no (company, origin, h) is in both the forecasts and the actuals"


def test_score_refuses_no_overlap_when_only_the_companies_differ():
    actuals = HAND_ACTUALS.assign(
        company_code=HAND_ACTUALS["company_code"].map({"TR02": "TR98", "TR04": "TR99"}))
    with pytest.raises(ValueError, match="no \\(company, origin, h\\) is in both"):
        score_topups(HAND_FORECASTS, actuals, by_company=False)


@pytest.mark.parametrize("by_company", [True, False])
def test_score_refuses_empty_tables(by_company):
    with pytest.raises(ValueError, match="no \\(company, origin, h\\) is in both"):
        score_topups(HAND_FORECASTS.iloc[0:0], HAND_ACTUALS, by_company=by_company)
    with pytest.raises(ValueError, match="no \\(company, origin, h\\) is in both"):
        score_topups(run(origins=[]), actuals_of(origins=[]), by_company=by_company)


# --- score_topups: NaN and infinite values are refused ----------------------------------------
# Checked per table right after its duplicate check: forecasts' checks come before actuals'.

def with_value(table, row, series, value):
    changed = table.copy()
    changed.loc[changed.index[row], series] = value
    return changed


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("series", SCORED_SERIES)
@pytest.mark.parametrize("by_company", [True, False])
def test_score_refuses_a_non_finite_forecast(value, series, by_company):
    forecasts = with_value(HAND_FORECASTS, 1, series, value)
    with pytest.raises(ValueError) as error:
        score_topups(forecasts, HAND_ACTUALS, by_company=by_company)
    assert str(error.value) == "forecasts holds a NaN or infinite value"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("series", SCORED_SERIES)
@pytest.mark.parametrize("by_company", [True, False])
def test_score_refuses_a_non_finite_actual(value, series, by_company):
    actuals = with_value(HAND_ACTUALS, 2, series, value)
    with pytest.raises(ValueError) as error:
        score_topups(HAND_FORECASTS, actuals, by_company=by_company)
    assert str(error.value) == "actuals holds a NaN or infinite value"


@pytest.mark.parametrize("which", ["forecasts", "actuals"])
def test_score_refuses_a_non_finite_value_in_an_unmatched_row(which):
    extra = scored_table([("TR09", 50, 1, np.nan, 0.0, 0.0)])
    forecasts, actuals = HAND_FORECASTS, HAND_ACTUALS
    if which == "forecasts":
        forecasts = pd.concat([forecasts, extra], ignore_index=True)
    else:
        actuals = pd.concat([actuals, extra], ignore_index=True)
    with pytest.raises(ValueError) as error:
        score_topups(forecasts, actuals, by_company=True)
    assert str(error.value) == f"{which} holds a NaN or infinite value"


def test_score_non_finite_forecasts_are_reported_before_non_finite_actuals():
    with pytest.raises(ValueError) as error:
        score_topups(with_value(HAND_FORECASTS, 0, "topup", np.nan),
                     with_value(HAND_ACTUALS, 0, "topup", np.inf), by_company=False)
    assert str(error.value) == "forecasts holds a NaN or infinite value"


def test_score_forecast_duplicates_are_reported_before_forecast_non_finite_values():
    forecasts = pd.concat([HAND_FORECASTS, HAND_FORECASTS.iloc[[0]]], ignore_index=True)
    forecasts = with_value(forecasts, 1, "refunds", np.nan)
    with pytest.raises(ValueError) as error:
        score_topups(forecasts, HAND_ACTUALS, by_company=True)
    assert str(error.value) == "forecasts holds a (company, origin, h) more than once"


def test_score_non_finite_forecasts_are_reported_before_actual_duplicates():
    # Per table: forecasts' duplicate and value checks, then actuals'.
    actuals = pd.concat([HAND_ACTUALS, HAND_ACTUALS.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        score_topups(with_value(HAND_FORECASTS, 0, "unidentified_cash", np.inf), actuals,
                     by_company=True)
    assert str(error.value) == "forecasts holds a NaN or infinite value"


def test_score_actual_duplicates_are_reported_before_actual_non_finite_values():
    actuals = pd.concat([HAND_ACTUALS, HAND_ACTUALS.iloc[[0]]], ignore_index=True)
    actuals = with_value(actuals, 1, "topup", np.nan)
    with pytest.raises(ValueError) as error:
        score_topups(HAND_FORECASTS, actuals, by_company=True)
    assert str(error.value) == "actuals holds a (company, origin, h) more than once"


def test_score_a_non_finite_value_outside_the_scored_series_is_not_read():
    # Only SCORED_SERIES are checked: another column may hold anything.
    forecasts = HAND_FORECASTS.assign(extra=np.nan)
    actuals = HAND_ACTUALS.assign(extra=np.inf)
    pd.testing.assert_frame_equal(score_topups(forecasts, actuals, by_company=True),
                                  score_topups(HAND_FORECASTS, HAND_ACTUALS, by_company=True))


def test_score_large_finite_values_are_scored():
    big = 1e300
    forecasts = scored_table([("TR02", 10, 1, big, 0.0, 0.0)])
    actuals = scored_table([("TR02", 10, 1, 0.0, 0.0, 0.0)])
    got = score_of(score_topups(forecasts, actuals, by_company=True), "new_customer_cash",
                   "TR02")
    assert got["mean_abs_error"] == big


def test_score_refuses_a_non_finite_value_without_changing_the_inputs():
    forecasts = with_value(HAND_FORECASTS, 0, "topup", np.nan)
    copy = forecasts.copy(deep=True)
    with pytest.raises(ValueError):
        score_topups(forecasts, HAND_ACTUALS, by_company=True)
    pd.testing.assert_frame_equal(forecasts, copy)
