"""Tests for `hurdle/features_series.py` (U19.1): the `deflated_trend` forecasts of each
company's (and the consolidated) weekly cash, as features.

Every series and table is small and hand-made; nothing is read from data/. The most important
test is the leak test at the end: a row made at origin t must not change when everything after
week t is cut off or changed.

Numbers to keep in mind: the reference needs `benchmark.MIN_TRAINING_WEEKS` (12) weeks with
data up to the origin, and its recent level looks at the `benchmark.LEVEL_WEEKS` (4) weeks up
to the origin.
"""
import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.benchmark import (
    LEVEL_WEEKS, MIN_TRAINING_WEEKS, deflated_forecasts, recent_level)
from collection_estimation.hurdle.evaluate import real_weekly_cash
from collection_estimation.hurdle.features_series import (
    SERIES_COLUMNS, SERIES_KEYS, TREND_REFERENCE, WITH_LEVEL, WITH_TREND, series_features,
    trend_forecast_at, trend_ratio)
from collection_estimation.hurdle.forecast import CONSOLIDATED


# --- builders --------------------------------------------------------------------------------

def simple_calendar(last=60):
    """Two calendar features per week, weeks 0..last."""
    weeks = list(range(0, last + 1))
    return pd.DataFrame({"week": weeks, "cycle": [float(w % 4) for w in weeks],
                         "wave": [float(np.cos(w)) for w in weeks]})


def growing_series(weeks):
    """A cash series that grows and has a 4-week cycle: always well above 0."""
    return pd.Series([1000.0 + 20.0 * w + 100.0 * (w % 4) for w in weeks],
                     index=pd.Index(list(weeks)), dtype="float64")


def make_weekly(rows):
    """rows: (company_code, week, gross_receipts, refunds); one line each."""
    frame = pd.DataFrame(rows, columns=["company_code", "week", "gross_receipts", "refunds"])
    frame.insert(0, "key", [f"K{i}" for i in range(len(frame))])
    frame["week"] = frame["week"].astype("int64")
    frame["gross_receipts"] = frame["gross_receipts"].astype("float64")
    frame["refunds"] = frame["refunds"].astype("float64")
    return frame


def make_weeks(last_week, missing=()):
    weeks = list(range(0, last_week + 1))
    return pd.DataFrame({"week": weeks, "is_missing": [w in missing for w in weeks]})


def make_levels(first=0, last=60):
    weeks = list(range(first, last + 1))
    return pd.Series([1.0 + w / 100 for w in weeks], index=weeks)


COMPANIES = ["1000", "2000"]
LAST = 30
MISSING = {6}


def example_weekly(last=LAST):
    """Two companies, weeks 0..last; week 6 missing (a huge amount there, never read); company
    2000 has no line in week 10 (a real 0)."""
    rows = []
    for w in range(0, last + 1):
        rows.append(("1000", w, 1e9 if w in MISSING else 1000.0 + 20.0 * w + 100.0 * (w % 4),
                     -5.0))
        if w != 10:
            rows.append(("2000", w, 1e9 if w in MISSING else 400.0 + 50.0 * np.cos(w), 0.0))
    return make_weekly(rows)


def example_inputs(**overrides):
    kwargs = dict(weekly=example_weekly(), weeks=make_weeks(LAST, missing=MISSING),
                  levels=make_levels(), calendar=simple_calendar(), companies=list(COMPANIES),
                  origins=[5, 14, 20], horizons=2, mean_weeks=4)
    kwargs.update(overrides)
    return kwargs


def run(**overrides):
    return series_features(**example_inputs(**overrides))


def one_row(result, company, origin, h):
    found = result[(result["company_code"] == company) & (result["origin_week"] == origin)
                   & (result["h"] == h)]
    assert len(found) == 1, (company, origin, h, len(found))
    return found.iloc[0]


# --- constants -------------------------------------------------------------------------------

def test_keys_and_columns():
    assert SERIES_KEYS == ["company_code", "origin_week", "h", "target_week"]
    assert SERIES_COLUMNS == ["trend_forecast", "trend_ratio", "all_trend_forecast",
                              "all_trend_ratio"]


def test_the_reference_is_deflated_trend_without_level_with_trend():
    assert TREND_REFERENCE == "deflated_trend"
    assert (WITH_LEVEL, WITH_TREND) == (False, True)


# --- trend_forecast_at -----------------------------------------------------------------------

def test_trend_forecast_is_the_references_forecast_when_not_a_fallback():
    series, calendar = growing_series(range(0, 25)), simple_calendar()
    for origin, h in [(15, 1), (20, 3), (24, 5)]:
        made = deflated_forecasts(series, calendar, h, train_until=origin, origins=[origin],
                                  with_level=False, with_trend=True).iloc[0]
        assert not made["fallback"]
        assert trend_forecast_at(series, calendar, origin, h) == float(made["forecast"])


def test_trend_forecast_is_a_plain_float():
    value = trend_forecast_at(growing_series(range(0, 25)), simple_calendar(), 20, 1)
    assert type(value) is float
    assert value > 0


def test_trend_forecast_follows_the_growth():
    # The trend input lets the ridge extrapolate the straight line: the series is an exact
    # line plus the 4-week cycle, so week 28 (cycle 0) is about 1000 + 20 x 28 = 1560, far
    # above week 24's 1480 of the same cycle position.
    series = growing_series(range(0, 25))
    value = trend_forecast_at(series, simple_calendar(), 24, 4)
    assert value == pytest.approx(1560.0, abs=5.0)
    assert value > series.loc[24]


def test_trend_forecast_reads_only_the_weeks_up_to_the_origin():
    calendar = simple_calendar()
    short = growing_series(range(0, 21))
    longer = pd.concat([short, pd.Series([1e9, -1e9, 5.0], index=[21, 22, 23])])
    assert trend_forecast_at(longer, calendar, 20, 2) == trend_forecast_at(short, calendar, 20, 2)


def test_trend_forecast_is_nan_with_too_few_training_weeks():
    series, calendar = growing_series(range(0, MIN_TRAINING_WEEKS - 1)), simple_calendar()
    origin = MIN_TRAINING_WEEKS - 2          # 11 weeks with data up to the origin
    made = deflated_forecasts(series, calendar, 1, origin, [origin], False, True).iloc[0]
    assert made["fallback"]                  # the reference would give its recent level
    assert np.isnan(trend_forecast_at(series, calendar, origin, 1))


def test_trend_forecast_exactly_at_the_minimum_training_weeks_is_a_number():
    series, calendar = growing_series(range(0, MIN_TRAINING_WEEKS)), simple_calendar()
    origin = MIN_TRAINING_WEEKS - 1          # weeks 0..11: exactly 12 with data
    assert not np.isnan(trend_forecast_at(series, calendar, origin, 1))


def test_trend_forecast_is_nan_when_the_prediction_is_not_above_zero():
    # Falls by 80 a week: 1000 at week 0, 40 at week 12. The straight line goes below 0 soon.
    series = pd.Series([1000.0 - 80.0 * w for w in range(0, 13)], index=range(0, 13))
    calendar = simple_calendar()
    made = deflated_forecasts(series, calendar, 5, 12, [12], False, True).iloc[0]
    assert made["fallback"]                  # precondition: the ridge predicted <= 0
    assert np.isnan(trend_forecast_at(series, calendar, 12, 5))


def test_trend_forecast_is_nan_with_no_data_in_the_level_weeks():
    # Data in weeks 0..20, none in 21..24: the reference itself would raise at origin 24.
    series, calendar = growing_series(range(0, 21)), simple_calendar()
    origin = 20 + LEVEL_WEEKS
    with pytest.raises(ValueError, match="no recent level"):
        deflated_forecasts(series, calendar, 1, origin, [origin], False, True)
    assert np.isnan(trend_forecast_at(series, calendar, origin, 1))


def test_trend_forecast_with_one_week_of_data_in_the_level_weeks_is_a_number():
    # Data in weeks 0..20, none in 21..23: at origin 23 week 20 is still in the level weeks.
    series = growing_series(range(0, 21))
    assert not np.isnan(trend_forecast_at(series, simple_calendar(), 20 + LEVEL_WEEKS - 1, 1))


def test_trend_forecast_raises_when_the_calendar_lacks_the_target_week():
    series = growing_series(range(0, 25))
    with pytest.raises(ValueError, match="calendar has no rows"):
        trend_forecast_at(series, simple_calendar(last=24), 24, 1)


def test_trend_forecast_does_not_modify_the_series():
    series = growing_series(range(0, 25))
    before = series.copy()
    trend_forecast_at(series, simple_calendar(), 20, 2)
    pd.testing.assert_series_equal(series, before)


# --- trend_ratio -----------------------------------------------------------------------------

def test_ratio_is_forecast_over_the_trailing_mean():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=[0, 1, 2, 3, 4])
    # Mean over weeks 2, 3, 4 = 40.
    assert trend_ratio(60.0, series, 4, 3) == 60.0 / 40.0


def test_ratio_mean_reads_no_week_after_the_origin():
    series = pd.Series([10.0, 20.0, 30.0, 1e9], index=[0, 1, 2, 3])
    # Origin 2, window 2: weeks 1 and 2, mean 25.
    assert trend_ratio(50.0, series, 2, 2) == 2.0


def test_ratio_mean_leaves_out_weeks_without_data():
    series = pd.Series([10.0, 30.0], index=[0, 3])     # weeks 1, 2 have no data
    # Origin 3, window 4: weeks 0..3, of which 0 and 3 have data: mean 20.
    assert trend_ratio(40.0, series, 3, 4) == 2.0


def test_ratio_is_nan_when_the_forecast_is_nan():
    series = pd.Series([10.0, 20.0], index=[0, 1])
    assert np.isnan(trend_ratio(float("nan"), series, 1, 2))


def test_ratio_is_nan_with_no_data_in_the_mean_weeks():
    series = pd.Series([10.0, 20.0], index=[0, 1])
    assert np.isnan(trend_ratio(5.0, series, 10, 3))   # weeks 8..10: none with data


def test_ratio_is_nan_when_the_mean_is_zero():
    series = pd.Series([0.0, 0.0, 0.0], index=[0, 1, 2])
    assert np.isnan(trend_ratio(5.0, series, 2, 3))


def test_ratio_is_nan_when_the_mean_is_negative():
    series = pd.Series([-10.0, -30.0], index=[0, 1])
    assert np.isnan(trend_ratio(5.0, series, 1, 2))


def test_ratio_with_mean_weeks_below_one_raises():
    series = pd.Series([10.0], index=[0])
    with pytest.raises(ValueError):
        trend_ratio(5.0, series, 0, 0)


# --- series_features: shape, order, keys -----------------------------------------------------

def test_result_has_the_keys_then_the_features():
    assert list(run().columns) == SERIES_KEYS + SERIES_COLUMNS


def test_one_row_per_origin_h_and_company():
    # 3 origins x 2 h x 2 companies.
    assert len(run()) == 3 * 2 * 2


def test_rows_sorted_by_origin_then_h_then_company():
    result = run()
    expected = [(o, h, c) for o in [5, 14, 20] for h in [1, 2] for c in COMPANIES]
    assert list(zip(result["origin_week"], result["h"], result["company_code"])) == expected
    assert result.index.equals(pd.RangeIndex(len(result)))


def test_target_week_is_origin_plus_h():
    result = run()
    assert (result["target_week"] == result["origin_week"] + result["h"]).all()


def test_no_consolidated_rows_only_companies():
    assert set(run()["company_code"]) == set(COMPANIES)


def test_feature_columns_are_float():
    result = run()
    for column in SERIES_COLUMNS:
        assert result[column].dtype == np.float64, column


def test_duplicate_and_unsorted_origins_give_the_sorted_unique_rows():
    pd.testing.assert_frame_equal(run(origins=[20, 5, 14, 20, 5]), run(origins=[5, 14, 20]))


def test_duplicate_and_unsorted_companies_give_the_sorted_unique_rows():
    pd.testing.assert_frame_equal(run(companies=["2000", "1000", "2000"]),
                                  run(companies=["1000", "2000"]))


def test_a_company_without_any_cash_still_gets_rows():
    result = run(companies=["1000", "2000", "3000"])
    assert len(result) == 3 * 2 * 3
    # Its series is all 0: the ridge predicts 0 (not above 0), so no forecast and no ratio.
    mine = result[result["company_code"] == "3000"]
    assert mine["trend_forecast"].isna().all()
    assert mine["trend_ratio"].isna().all()


# --- series_features: values -----------------------------------------------------------------

def test_company_forecast_is_trend_forecast_at_on_its_own_series():
    inputs = example_inputs()
    real = real_weekly_cash(inputs["weekly"], inputs["weeks"], inputs["levels"], COMPANIES)
    result = run()
    for company in COMPANIES:
        for origin in [14, 20]:
            for h in [1, 2]:
                expected = trend_forecast_at(real[company], inputs["calendar"], origin, h)
                assert not np.isnan(expected)
                assert one_row(result, company, origin, h)["trend_forecast"] == expected


def test_company_ratio_is_its_forecast_over_its_trailing_mean():
    inputs = example_inputs()
    real = real_weekly_cash(inputs["weekly"], inputs["weeks"], inputs["levels"], COMPANIES)
    row = one_row(run(), "2000", 20, 2)
    assert row["trend_ratio"] == row["trend_forecast"] / recent_level(real["2000"], 20, 4)


def test_ratio_uses_mean_weeks_not_the_level_weeks():
    inputs = example_inputs()
    real = real_weekly_cash(inputs["weekly"], inputs["weeks"], inputs["levels"], COMPANIES)
    row = one_row(run(mean_weeks=10), "1000", 20, 1)
    assert row["trend_ratio"] == row["trend_forecast"] / recent_level(real["1000"], 20, 10)
    # Same forecast, other window, other ratio.
    assert one_row(run(mean_weeks=10), "1000", 20, 1)["trend_forecast"] == \
        one_row(run(mean_weeks=4), "1000", 20, 1)["trend_forecast"]
    assert one_row(run(mean_weeks=10), "1000", 20, 1)["trend_ratio"] != \
        one_row(run(mean_weeks=4), "1000", 20, 1)["trend_ratio"]


def test_all_columns_are_the_consolidated_series_forecast_and_ratio():
    inputs = example_inputs()
    real = real_weekly_cash(inputs["weekly"], inputs["weeks"], inputs["levels"], COMPANIES)
    result = run()
    for origin in [14, 20]:
        for h in [1, 2]:
            expected = trend_forecast_at(real[CONSOLIDATED], inputs["calendar"], origin, h)
            expected_ratio = trend_ratio(expected, real[CONSOLIDATED], origin, 4)
            for company in COMPANIES:
                row = one_row(result, company, origin, h)
                assert row["all_trend_forecast"] == expected
                assert row["all_trend_ratio"] == expected_ratio


def test_all_columns_repeat_on_every_company_row_of_an_origin_and_h():
    result = run()
    per_pair = result.groupby(["origin_week", "h"])[["all_trend_forecast", "all_trend_ratio"]]
    assert (per_pair.nunique(dropna=False) == 1).all().all()


def test_all_forecast_is_not_the_sum_of_the_company_forecasts():
    # The consolidated series is fitted on its own, so the two need not add up.
    row_pair = run()
    pair = row_pair[(row_pair["origin_week"] == 20) & (row_pair["h"] == 1)]
    assert pair["all_trend_forecast"].iloc[0] != pytest.approx(pair["trend_forecast"].sum(),
                                                               rel=1e-12)


def test_early_origin_has_nan_everywhere():
    # Origin 5: weeks 0..5 with data, too few to fit the ridge.
    result = run()
    early = result[result["origin_week"] == 5]
    assert early[SERIES_COLUMNS].isna().all().all()


def test_origins_with_enough_weeks_have_no_nan():
    result = run()
    later = result[result["origin_week"] >= 14]
    assert later[SERIES_COLUMNS].notna().all().all()


def test_origin_just_after_a_long_gap_has_nan_and_the_rest_is_kept():
    # Weeks 17..20 have no data: no recent level at origin 20 for anyone.
    weeks = make_weeks(LAST, missing=MISSING | {17, 18, 19, 20})
    result = run(weeks=weeks, origins=[16, 20, 21])
    assert result.loc[result["origin_week"] == 20, SERIES_COLUMNS].isna().all().all()
    assert result.loc[result["origin_week"] == 16, SERIES_COLUMNS].notna().all().all()
    assert result.loc[result["origin_week"] == 21, SERIES_COLUMNS].notna().all().all()


def test_a_row_does_not_depend_on_the_other_origins_asked_for():
    alone = run(origins=[14])
    together = run(origins=[5, 14, 20])
    pd.testing.assert_frame_equal(
        together[together["origin_week"] == 14].reset_index(drop=True), alone)


def test_a_row_does_not_depend_on_the_horizon_count():
    two = run(horizons=2)
    five = run(horizons=5)
    pd.testing.assert_frame_equal(five[five["h"] <= 2].reset_index(drop=True), two)


def test_same_inputs_give_the_same_result():
    pd.testing.assert_frame_equal(run(), run())


def test_inputs_are_not_modified():
    inputs = example_inputs()
    before = {name: inputs[name].copy() for name in ("weekly", "weeks", "levels", "calendar")}
    series_features(**inputs)
    for name, frame in before.items():
        if isinstance(frame, pd.Series):
            pd.testing.assert_series_equal(inputs[name], frame)
        else:
            pd.testing.assert_frame_equal(inputs[name], frame)


# --- series_features: errors -----------------------------------------------------------------

def test_empty_origins_raise():
    with pytest.raises(ValueError, match="no origins"):
        run(origins=[])


@pytest.mark.parametrize("horizons", [0, -1])
def test_horizons_below_one_raise(horizons):
    with pytest.raises(ValueError, match="horizons start at 1"):
        run(horizons=horizons)


@pytest.mark.parametrize("mean_weeks", [0, -3])
def test_mean_weeks_below_one_raise(mean_weeks):
    with pytest.raises(ValueError, match="mean_weeks"):
        run(mean_weeks=mean_weeks)


def test_empty_company_list_raises():
    with pytest.raises(ValueError, match="the company list is empty"):
        run(companies=[])


def test_all_in_the_company_list_raises():
    with pytest.raises(ValueError, match="consolidated rows"):
        run(companies=["1000", "2000", CONSOLIDATED])


def test_cash_of_a_company_outside_the_list_raises():
    with pytest.raises(ValueError, match=r"not in the list: \['2000'\]"):
        run(companies=["1000"])


def test_target_week_past_the_calendar_raises():
    with pytest.raises(ValueError, match="calendar has no rows"):
        run(calendar=simple_calendar(last=20), origins=[20])


def test_own_checks_come_before_the_company_checks():
    # Empty origins and an empty company list: the origins error first.
    with pytest.raises(ValueError, match="no origins"):
        run(origins=[], companies=[])
    with pytest.raises(ValueError, match="horizons start at 1"):
        run(horizons=0, companies=[])
    with pytest.raises(ValueError, match="mean_weeks"):
        run(mean_weeks=0, companies=[])


# --- THE LEAK TEST ---------------------------------------------------------------------------
# A row made at origin t reads only weeks up to t. Cutting every table to weeks <= t, or
# changing every amount and price level after t, must give exactly the same rows. The calendar
# is known ahead (a holiday is in the calendar before it comes), so it is never cut.

LEAK_ORIGINS = [12, 14, 17, 20, 24]


def rows_at(origin, weekly, weeks, levels, mean_weeks=4):
    return series_features(weekly, weeks, levels, simple_calendar(), list(COMPANIES), [origin],
                           3, mean_weeks)


@pytest.mark.parametrize("origin", LEAK_ORIGINS)
def test_leak_cutting_everything_after_the_origin_changes_nothing(origin):
    weekly, weeks, levels = example_weekly(), make_weeks(LAST, missing=MISSING), make_levels()
    full = rows_at(origin, weekly, weeks, levels)
    cut = rows_at(origin, weekly[weekly["week"] <= origin].reset_index(drop=True),
                  weeks[weeks["week"] <= origin].reset_index(drop=True),
                  levels[levels.index <= origin])
    pd.testing.assert_frame_equal(cut, full)


@pytest.mark.parametrize("origin", LEAK_ORIGINS)
def test_leak_changing_every_amount_after_the_origin_changes_nothing(origin):
    weekly, weeks, levels = example_weekly(), make_weeks(LAST, missing=MISSING), make_levels()
    full = rows_at(origin, weekly, weeks, levels)
    changed = weekly.copy()
    after = changed["week"] > origin
    changed.loc[after, "gross_receipts"] = changed.loc[after, "gross_receipts"] * 7.0 + 1e6
    changed.loc[after, "refunds"] = -3e5
    pd.testing.assert_frame_equal(rows_at(origin, changed, weeks, levels), full)


@pytest.mark.parametrize("origin", LEAK_ORIGINS)
def test_leak_changing_the_price_levels_after_the_origin_changes_nothing(origin):
    weekly, weeks, levels = example_weekly(), make_weeks(LAST, missing=MISSING), make_levels()
    full = rows_at(origin, weekly, weeks, levels)
    changed = levels.copy()
    changed[changed.index > origin] = 50.0
    pd.testing.assert_frame_equal(rows_at(origin, weekly, weeks, changed), full)


@pytest.mark.parametrize("origin", LEAK_ORIGINS)
def test_leak_which_weeks_after_the_origin_have_data_changes_nothing(origin):
    weekly, levels = example_weekly(), make_levels()
    full = rows_at(origin, weekly, make_weeks(LAST, missing=MISSING), levels)
    other_gaps = MISSING | {w for w in range(origin + 1, LAST + 1) if w % 2 == 0}
    pd.testing.assert_frame_equal(rows_at(origin, weekly, make_weeks(LAST, missing=other_gaps),
                                          levels), full)


def test_leak_with_a_long_mean_window_changes_nothing():
    weekly, weeks, levels = example_weekly(), make_weeks(LAST, missing=MISSING), make_levels()
    full = rows_at(20, weekly, weeks, levels, mean_weeks=13)
    cut = rows_at(20, weekly[weekly["week"] <= 20], weeks[weeks["week"] <= 20],
                  levels[levels.index <= 20], mean_weeks=13)
    pd.testing.assert_frame_equal(cut, full)


def test_leak_test_has_teeth_a_change_at_the_origin_does_show():
    # The same change, made at the origin week itself, must change the rows: otherwise the
    # tests above would pass for a function that read nothing at all.
    weekly, weeks, levels = example_weekly(), make_weeks(LAST, missing=MISSING), make_levels()
    full = rows_at(20, weekly, weeks, levels)
    changed = weekly.copy()
    at = changed["week"] == 20
    changed.loc[at, "gross_receipts"] = changed.loc[at, "gross_receipts"] * 7.0
    moved = rows_at(20, changed, weeks, levels)
    assert not moved["trend_forecast"].equals(full["trend_forecast"])
    assert not moved["all_trend_forecast"].equals(full["all_trend_forecast"])


# --- check_series_no_leak (U19.2) ------------------------------------------------------------
# prepare_inputs (step 11c) builds the series rows at every panel origin, then calls
# check_series_no_leak with the [receivables] leak_check_weeks: each check week that is one of
# the rows' origins is made again from the weekly amounts, the week calendar and the price
# levels cut off after it, and must equal the stored rows exactly. A week that is not an
# origin is skipped. Below, "stored" rows are made at origins 5, 14, 20 and 24 for h = 1, 2
# with a mean window of 4 weeks; origin 5 is too early for a trend (all NaN).

from collection_estimation.hurdle.evaluate import real_weekly_cash as _real  # noqa: E402
from collection_estimation.hurdle.features_series import check_series_no_leak  # noqa: E402

CHECK_ORIGINS = [5, 14, 20, 24]


def leak_message(week):
    return (f"leak check: the series features of week {week} differ when made again from the "
            f"data up to that week, so something from after it was used")


def stored(weekly=None, origins=CHECK_ORIGINS, horizons=2, mean_weeks=4):
    weekly = example_weekly() if weekly is None else weekly
    return series_features(weekly, make_weeks(LAST, missing=MISSING), make_levels(),
                           simple_calendar(), list(COMPANIES), origins, horizons, mean_weeks)


def check(series, check_weeks, horizons=2, mean_weeks=4, weekly=None):
    """check_series_no_leak against the example's honest data."""
    weekly = example_weekly() if weekly is None else weekly
    return check_series_no_leak(series, weekly, make_weeks(LAST, missing=MISSING),
                                make_levels(), simple_calendar(), list(COMPANIES), horizons,
                                mean_weeks, check_weeks)


def leaked_at(week):
    """The honest rows, except those of origin `week`: made at week + 1 (one week of data too
    many) and labelled as made at `week`, h and target week as if at `week`."""
    honest = stored()
    late = stored(origins=[week + 1])
    late = late.assign(origin_week=week, target_week=late["target_week"] - 1)
    rows = pd.concat([honest[honest["origin_week"] != week], late])
    return rows.sort_values(["origin_week", "h", "company_code"]).reset_index(drop=True)


# honest input passes

def test_check_passes_on_honest_rows_and_returns_none():
    assert check(stored(), (14, 20)) is None


def test_check_passes_at_every_origin():
    check(stored(), tuple(CHECK_ORIGINS))


def test_check_passes_at_an_early_origin_whose_rows_are_all_nan():
    series = stored()
    assert series.loc[series["origin_week"] == 5, SERIES_COLUMNS].isna().all().all()
    check(series, (5,))


def test_check_passes_with_a_long_mean_window_and_three_horizons():
    check(stored(horizons=3, mean_weeks=13), (14, 24), horizons=3, mean_weeks=13)


def test_check_passes_when_the_rows_have_a_non_default_index():
    series = stored()
    series.index = range(100, 100 + len(series))
    check(series, (14, 20))


def test_check_passes_when_the_data_after_the_week_differ_from_the_data_the_rows_came_from():
    # Rows made from data that differ only after week 20 are still honest at week 20.
    changed = example_weekly()
    after = changed["week"] > 20
    changed.loc[after, "gross_receipts"] = changed.loc[after, "gross_receipts"] * 9.0 + 1e6
    check(stored(weekly=changed), (20,))


# a real leak is caught

def test_rows_made_with_one_week_too_many_raise():
    with pytest.raises(ValueError) as error:
        check(leaked_at(20), (20,))
    assert str(error.value) == leak_message(20)


def test_the_one_week_late_rows_really_differ():
    # The leak above is a real one: the late rows are not the honest rows by chance.
    honest = stored()
    late = leaked_at(20)
    at_20 = honest["origin_week"] == 20
    assert not honest.loc[at_20, "trend_forecast"].reset_index(drop=True).equals(
        late.loc[late["origin_week"] == 20, "trend_forecast"].reset_index(drop=True))


def test_a_leak_at_a_week_not_checked_is_not_caught():
    # Only the check weeks are rebuilt: a leak at 20 passes a check of 14 and 24.
    check(leaked_at(20), (14, 24))


def test_the_first_leaking_check_week_is_named():
    with pytest.raises(ValueError) as error:
        check(leaked_at(24), (14, 20, 24))
    assert str(error.value) == leak_message(24)


def test_a_ratio_over_the_whole_periods_mean_raises():
    # Another leak: the trend ratio divided by the company's mean over every week, the weeks
    # after the origin among them.
    series = stored()
    real = _real(example_weekly(), make_weeks(LAST, missing=MISSING), make_levels(),
                 list(COMPANIES))
    at_20 = series["origin_week"] == 20
    for company in COMPANIES:
        rows = at_20 & (series["company_code"] == company)
        series.loc[rows, "trend_ratio"] = (series.loc[rows, "trend_forecast"]
                                           / real[company].mean())
    with pytest.raises(ValueError, match=r"^leak check: the series features of week 20"):
        check(series, (20,))


def test_rows_made_from_other_data_up_to_the_week_raise():
    # The stored rows came from a week-20 amount seven times larger than the data checked.
    changed = example_weekly()
    at_20 = changed["week"] == 20
    changed.loc[at_20, "gross_receipts"] = changed.loc[at_20, "gross_receipts"] * 7.0
    with pytest.raises(ValueError) as error:
        check(stored(weekly=changed), (20,))
    assert str(error.value) == leak_message(20)


def test_one_changed_value_in_one_row_raises():
    series = stored()
    row = series.index[(series["origin_week"] == 14) & (series["h"] == 2)
                       & (series["company_code"] == "2000")][0]
    series.loc[row, "all_trend_ratio"] = series.loc[row, "all_trend_ratio"] + 1e-9
    with pytest.raises(ValueError, match=r"^leak check: the series features of week 14"):
        check(series, (14,))


def test_a_number_where_the_rebuild_gives_nan_raises():
    series = stored()
    series.loc[series["origin_week"] == 5, "trend_forecast"] = 1.0
    with pytest.raises(ValueError, match=r"^leak check: the series features of week 5"):
        check(series, (5,))


def test_a_missing_row_at_a_check_week_raises():
    series = stored()
    drop = series.index[(series["origin_week"] == 20) & (series["h"] == 2)
                        & (series["company_code"] == "1000")]
    with pytest.raises(ValueError, match=r"^leak check: the series features of week 20"):
        check(series.drop(index=drop), (20,))


def test_rows_made_with_another_mean_window_raise():
    with pytest.raises(ValueError, match=r"^leak check: the series features of week 20"):
        check(stored(mean_weeks=4), (20,), mean_weeks=13)


def test_rows_made_with_another_horizon_count_raise():
    with pytest.raises(ValueError, match=r"^leak check: the series features of week 20"):
        check(stored(horizons=2), (20,), horizons=3)


# weeks not among the origins, and no check weeks

def test_check_weeks_not_among_the_origins_are_skipped():
    # Every row is wrong, but none of the check weeks is an origin of the rows.
    series = stored().assign(trend_forecast=-1.0)
    assert check(series, (0, 3, 15, 99)) is None


def test_empty_check_weeks_check_nothing():
    series = stored().assign(trend_forecast=-1.0)
    assert check(series, ()) is None


def test_a_check_week_among_the_origins_is_checked_beside_skipped_ones():
    series = stored().assign(trend_forecast=-1.0)
    with pytest.raises(ValueError) as error:
        check(series, (3, 14, 99))
    assert str(error.value) == leak_message(14)


# it changes nothing

def test_check_does_not_modify_its_inputs():
    series, weekly = stored(), example_weekly()
    weeks, levels, calendar = make_weeks(LAST, missing=MISSING), make_levels(), simple_calendar()
    companies = list(COMPANIES)
    copies = (series.copy(deep=True), weekly.copy(deep=True), weeks.copy(deep=True),
              levels.copy(deep=True), calendar.copy(deep=True), list(companies))
    check_series_no_leak(series, weekly, weeks, levels, calendar, companies, 2, 4, (14, 20))
    pd.testing.assert_frame_equal(series, copies[0])
    pd.testing.assert_frame_equal(weekly, copies[1])
    pd.testing.assert_frame_equal(weeks, copies[2])
    pd.testing.assert_series_equal(levels, copies[3])
    pd.testing.assert_frame_equal(calendar, copies[4])
    assert companies == copies[5]
