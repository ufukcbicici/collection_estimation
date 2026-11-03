"""Tests for `hurdle/evaluate.py`: `actual_totals` (U14.1), the cash that really came per company
and consolidated ("ALL") in the target week of every origin and h, nominal and deflated.

Every table is hand-made; nothing is read from data/. The properties that matter most are
tested directly: actual = gross receipts + refunds of the TARGET week, deflated by the TARGET
week's level; a missing week or a week past the calendar has no row (no data is not 0), while
a company with no line in a week with data gets a real 0.0; "ALL" is the sum of the companies.

U14.2, at the end: `rolling_forecasts`, `forecast.forecast` at every origin, stacked, on the
small training table and side tables of test_forecast.py. Every cheap check must fire before
the first fit, which the tests prove by replacing the fits with spies that fail if called.
"""
import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.evaluate import ACTUAL_COLUMNS, actual_totals
from collection_estimation.hurdle.forecast import CONSOLIDATED


# --- builders --------------------------------------------------------------------------------

def make_weekly(rows):
    """rows: (key, company_code, customer_id, week, gross_receipts, refunds)."""
    frame = pd.DataFrame(rows, columns=["key", "company_code", "customer_id", "week",
                                        "gross_receipts", "refunds"])
    frame["week"] = frame["week"].astype("int64")
    frame["gross_receipts"] = frame["gross_receipts"].astype("float64")
    frame["refunds"] = frame["refunds"].astype("float64")
    frame["unapplied_lines"] = 0
    frame["lines"] = 1
    return frame


def make_weeks(last_week, missing=()):
    weeks = list(range(0, last_week + 1))
    missing_days = [1 if w in missing else 0 for w in weeks]
    return pd.DataFrame({"week": weeks,
                         "business_days": 5,
                         "days_with_file": [5 - m for m in missing_days],
                         "filled_days": 0,
                         "missing_days": missing_days,
                         "is_missing": [m > 0 for m in missing_days]})


def make_levels(first=0, last=30):
    """Level of week w is 1 + w / 10: distinct per week, so a wrong week shows."""
    weeks = list(range(first, last + 1))
    return pd.Series([1.0 + w / 10 for w in weeks], index=weeks)


def level(week):
    return 1.0 + week / 10


def row(result, company, origin, h):
    found = result[(result["company_code"] == company) & (result["origin_week"] == origin)
                   & (result["h"] == h)]
    assert len(found) == 1, f"expected one row for {company}, {origin}, {h}, got {len(found)}"
    return found.iloc[0]


def pairs(result, company):
    part = result[result["company_code"] == company]
    return list(zip(part["origin_week"], part["h"]))


@pytest.fixture
def basic():
    """Two companies, weeks 0-10, week 6 missing."""
    weekly = make_weekly([
        ("K1", "1000", "C1", 3, 100.0, 0.0),
        ("K2", "1000", "C2", 3, 50.0, -20.0),
        ("K3", "1000", "C3", 4, 10.0, 0.0),
        ("K4", "2000", "C4", 3, 7.0, -1.0),
        ("K5", "2000", "C5", 5, 200.0, -50.0),
        ("K6", "2000", "C6", 6, 999.0, 0.0),      # in a missing week: never read
        ("K7", "1000", None, 4, 5.0, -2.5),       # unidentified cash counts too
    ])
    return weekly, make_weeks(10, missing={6}), make_levels()


# --- constants -------------------------------------------------------------------------------

def test_actual_columns():
    assert ACTUAL_COLUMNS == ["company_code", "origin_week", "h", "target_week", "actual",
                              "actual_real"]


def test_result_has_exactly_the_actual_columns(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 3)
    assert list(result.columns) == ACTUAL_COLUMNS


# --- hand-computed values --------------------------------------------------------------------

def test_several_keys_of_one_company_in_one_week_are_summed(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 1)
    assert row(result, "1000", 2, 1)["actual"] == pytest.approx(100.0 + 50.0 - 20.0)


def test_refunds_are_subtracted(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 3)
    assert row(result, "2000", 2, 1)["actual"] == pytest.approx(7.0 - 1.0)
    assert row(result, "2000", 2, 3)["actual"] == pytest.approx(200.0 - 50.0)
    assert row(result, "1000", 2, 2)["actual"] == pytest.approx(10.0 + 5.0 - 2.5)


def test_all_values_by_hand(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 3)
    expected = {
        ("1000", 1): 130.0, ("1000", 2): 12.5, ("1000", 3): 0.0,
        ("2000", 1): 6.0, ("2000", 2): 0.0, ("2000", 3): 150.0,
        (CONSOLIDATED, 1): 136.0, (CONSOLIDATED, 2): 12.5, (CONSOLIDATED, 3): 150.0,
    }
    for (company, h), actual in expected.items():
        found = row(result, company, 2, h)
        assert found["target_week"] == 2 + h
        assert found["actual"] == pytest.approx(actual)
        assert found["actual_real"] == pytest.approx(actual / level(2 + h))


def test_actual_real_uses_the_target_weeks_level_not_the_origins(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 1)
    found = row(result, "1000", 2, 1)
    assert found["actual_real"] == pytest.approx(130.0 / level(3), rel=1e-15)
    assert found["actual_real"] != pytest.approx(130.0 / level(2))


def test_actual_real_with_a_level_series_that_is_not_a_simple_formula():
    weekly = make_weekly([("K1", "A", "C1", 2, 300.0, 0.0), ("K2", "A", "C2", 3, 300.0, 0.0)])
    levels = pd.Series([1.0, 1.5, 2.0, 3.0], index=[0, 1, 2, 3])
    result = actual_totals(weekly, make_weeks(3), levels, ["A"], [1], 2)
    assert row(result, "A", 1, 1)["actual_real"] == pytest.approx(150.0)
    assert row(result, "A", 1, 2)["actual_real"] == pytest.approx(100.0)


# --- missing weeks and weeks beyond the calendar ---------------------------------------------

def test_missing_target_week_has_no_row_for_any_company(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [4], 3)
    for company in ["1000", "2000", CONSOLIDATED]:
        assert pairs(result, company) == [(4, 1), (4, 3)]
    assert 6 not in set(result["target_week"])


def test_target_week_beyond_the_calendar_has_no_row(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [8], 5)
    for company in ["1000", "2000", CONSOLIDATED]:
        assert pairs(result, company) == [(8, 1), (8, 2)]
    assert result["target_week"].max() == 10


def test_origin_all_of_whose_targets_lack_data_contributes_nothing(basic):
    weekly, weeks, levels = basic
    # Origin 10: targets 11, 12 are past the calendar. Origin 5, h=1: target 6 is missing.
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [5, 10, 2], 1)
    assert set(result["origin_week"]) == {2}
    alone = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 1)
    pd.testing.assert_frame_equal(result, alone)


def test_no_target_with_data_gives_an_empty_table_with_the_columns(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [10, 20], 3)
    assert result.empty
    assert list(result.columns) == ACTUAL_COLUMNS


def test_no_origins_gives_an_empty_table_with_the_columns(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [], 3)
    assert result.empty
    assert list(result.columns) == ACTUAL_COLUMNS


def test_missing_week_without_a_price_level_does_not_raise():
    # The missing week 2 has no level, but it gets no row, so it is never deflated.
    weekly = make_weekly([("K1", "A", "C1", 1, 10.0, 0.0), ("K2", "A", "C2", 3, 20.0, 0.0)])
    levels = pd.Series([1.0, 1.0, 2.0], index=[0, 1, 3])
    result = actual_totals(weekly, make_weeks(3, missing={2}), levels, ["A"], [0], 3)
    assert pairs(result, "A") == [(0, 1), (0, 3)]
    assert row(result, "A", 0, 3)["actual_real"] == pytest.approx(10.0)


def test_cash_in_a_missing_week_is_never_counted(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], list(range(0, 10)), 5)
    assert not np.isclose(result["actual"], 999.0).any()
    assert 6 not in set(result["target_week"])


# --- real zeros ------------------------------------------------------------------------------

def test_company_with_no_line_in_a_week_with_data_gets_zero(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [3], 2)
    found = row(result, "2000", 3, 1)        # week 4: only company 1000 has lines
    assert found["actual"] == 0.0
    assert found["actual_real"] == 0.0


def test_week_with_data_and_no_line_at_all_gives_zero_for_every_company(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [6], 1)  # week 7
    for company in ["1000", "2000", CONSOLIDATED]:
        found = row(result, company, 6, 1)
        assert found["actual"] == 0.0
        assert found["actual_real"] == 0.0


def test_company_with_no_cash_at_all_still_gets_rows(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000", "3000"], [2], 3)
    assert pairs(result, "3000") == [(2, 1), (2, 2), (2, 3)]
    assert (result.loc[result["company_code"] == "3000", "actual"] == 0.0).all()
    assert (result.loc[result["company_code"] == "3000", "actual_real"] == 0.0).all()


def test_actual_has_no_nan(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000", "3000"],
                           list(range(0, 10)), 5)
    assert not result["actual"].isna().any()
    assert not result["actual_real"].isna().any()


def test_every_company_has_the_same_origin_h_pairs(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000", "3000"],
                           list(range(0, 10)), 5)
    reference = pairs(result, CONSOLIDATED)
    for company in ["1000", "2000", "3000"]:
        assert pairs(result, company) == reference


def test_empty_weekly_gives_zeros_everywhere():
    weekly = make_weekly([])
    result = actual_totals(weekly, make_weeks(5), make_levels(), ["A", "B"], [1], 2)
    assert len(result) == 6
    assert (result["actual"] == 0.0).all()
    assert (result["actual_real"] == 0.0).all()


# --- consolidated ----------------------------------------------------------------------------

def test_all_is_the_sum_of_the_companies_per_origin_and_h(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000", "3000"],
                           list(range(0, 10)), 5)
    companies = result[result["company_code"] != CONSOLIDATED]
    summed = companies.groupby(["origin_week", "h"])[["actual", "actual_real"]].sum()
    consolidated = result[result["company_code"] == CONSOLIDATED].set_index(
        ["origin_week", "h"])[["actual", "actual_real"]]
    pd.testing.assert_frame_equal(consolidated.sort_index(), summed.sort_index(),
                                  check_exact=False, rtol=1e-12)


def test_consolidated_ties_to_the_weekly_table_in_every_week_with_data(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], list(range(0, 10)), 5)
    consolidated = result[result["company_code"] == CONSOLIDATED]
    for _, found in consolidated.iterrows():
        in_week = weekly[weekly["week"] == found["target_week"]]
        assert found["actual"] == pytest.approx(
            (in_week["gross_receipts"] + in_week["refunds"]).sum())


def test_same_target_week_from_different_origins_gives_the_same_actual(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [3, 4, 2], 3)
    for company in ["1000", "2000", CONSOLIDATED]:
        a = row(result, company, 3, 2)
        b = row(result, company, 4, 1)
        c = row(result, company, 2, 3)
        assert a["target_week"] == b["target_week"] == c["target_week"] == 5
        assert a["actual"] == b["actual"] == c["actual"]
        assert a["actual_real"] == b["actual_real"] == c["actual_real"]


def test_actual_does_not_depend_on_the_horizon_count(basic):
    weekly, weeks, levels = basic
    short = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 2)
    long = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 5)
    trimmed = long[long["h"] <= 2].reset_index(drop=True)
    pd.testing.assert_frame_equal(short, trimmed)


# --- order, deduplication, dtypes ------------------------------------------------------------

def test_duplicate_and_unsorted_origins_are_deduplicated_and_sorted(basic):
    weekly, weeks, levels = basic
    messy = actual_totals(weekly, weeks, levels, ["1000", "2000"], [4, 2, 4, 0, 2], 2)
    clean = actual_totals(weekly, weeks, levels, ["1000", "2000"], [0, 2, 4], 2)
    pd.testing.assert_frame_equal(messy, clean)
    assert not messy.duplicated(["company_code", "origin_week", "h"]).any()


def test_duplicate_and_unsorted_companies_are_deduplicated_and_sorted(basic):
    weekly, weeks, levels = basic
    messy = actual_totals(weekly, weeks, levels, ["2000", "1000", "2000"], [2], 2)
    clean = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 2)
    pd.testing.assert_frame_equal(messy, clean)


def test_row_order_companies_sorted_then_all_then_origin_and_h(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["2000", "3000", "1000"], [4, 0, 2], 3)
    order = list(dict.fromkeys(result["company_code"]))
    assert order == ["1000", "2000", "3000", CONSOLIDATED]
    for company in order:
        part = pairs(result, company)
        assert part == sorted(part)
    assert list(result.index) == list(range(len(result)))


def test_all_comes_last_even_when_a_company_code_sorts_after_it():
    # "ZZZ" > "ALL" alphabetically, but "ALL" must still come last.
    weekly = make_weekly([("K1", "ZZZ", "C1", 1, 10.0, 0.0), ("K2", "AAA", "C2", 1, 5.0, 0.0)])
    result = actual_totals(weekly, make_weeks(3), make_levels(), ["ZZZ", "AAA"], [0], 1)
    assert list(result["company_code"]) == ["AAA", "ZZZ", CONSOLIDATED]


def test_dtypes(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2, 3], 3)
    assert result["company_code"].dtype == object
    assert result["actual"].dtype == np.float64
    assert result["actual_real"].dtype == np.float64
    for column in ["origin_week", "h", "target_week"]:
        assert pd.api.types.is_integer_dtype(result[column]), column


def test_target_week_is_origin_plus_h(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], list(range(0, 10)), 5)
    assert (result["target_week"] == result["origin_week"] + result["h"]).all()
    assert result["h"].between(1, 5).all()


def test_number_of_rows(basic):
    weekly, weeks, levels = basic
    # Origins 0..9, h 1..5: targets 1..14; with data: 1..10 except 6.
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], list(range(0, 10)), 5)
    expected_pairs = sum(1 for o in range(0, 10) for h in range(1, 6)
                         if o + h <= 10 and o + h != 6)
    assert len(result) == 3 * expected_pairs


def test_integer_company_codes_work_and_sort_numerically():
    weekly = make_weekly([("K1", 20, "C1", 1, 10.0, 0.0), ("K2", 3, "C2", 1, 5.0, 0.0)])
    result = actual_totals(weekly, make_weeks(3), make_levels(), [20, 3], [0], 1)
    assert list(result["company_code"]) == [3, 20, CONSOLIDATED]
    assert result["company_code"].dtype == object
    assert list(result["actual"]) == pytest.approx([5.0, 10.0, 15.0])


# --- negative net ----------------------------------------------------------------------------

def test_negative_net_is_passed_through(basic):
    weekly = make_weekly([("K1", "A", "C1", 2, 10.0, -40.0), ("K2", "B", "C2", 2, 5.0, 0.0)])
    result = actual_totals(weekly, make_weeks(4), make_levels(), ["A", "B"], [1], 1)
    assert row(result, "A", 1, 1)["actual"] == pytest.approx(-30.0)
    assert row(result, "A", 1, 1)["actual_real"] == pytest.approx(-30.0 / level(2))
    assert row(result, CONSOLIDATED, 1, 1)["actual"] == pytest.approx(-25.0)


def test_refund_only_week_is_negative():
    weekly = make_weekly([("K1", "A", "C1", 2, 0.0, -12.0)])
    result = actual_totals(weekly, make_weeks(4), make_levels(), ["A"], [1], 1)
    assert row(result, "A", 1, 1)["actual"] == pytest.approx(-12.0)


# --- errors ----------------------------------------------------------------------------------

@pytest.mark.parametrize("horizons", [0, -1, -5])
def test_horizons_below_one_raise(basic, horizons):
    weekly, weeks, levels = basic
    with pytest.raises(ValueError, match="horizons"):
        actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], horizons)


def test_empty_company_list_raises(basic):
    weekly, weeks, levels = basic
    with pytest.raises(ValueError, match="company list is empty"):
        actual_totals(weekly, weeks, levels, [], [2], 3)


def test_all_in_the_company_list_raises(basic):
    weekly, weeks, levels = basic
    with pytest.raises(ValueError, match="consolidated"):
        actual_totals(weekly, weeks, levels, ["1000", "2000", CONSOLIDATED], [2], 3)


def test_weekly_with_a_company_outside_the_list_raises_and_names_it(basic):
    weekly, weeks, levels = basic
    with pytest.raises(ValueError, match="not in the list") as error:
        actual_totals(weekly, weeks, levels, ["1000"], [2], 3)
    assert "2000" in str(error.value)


def test_company_outside_the_list_raises_even_if_its_cash_is_only_in_a_missing_week(basic):
    weekly, weeks, levels = basic
    extra = pd.concat([weekly, make_weekly([("K9", "9999", "C9", 6, 1.0, 0.0)])],
                      ignore_index=True)
    with pytest.raises(ValueError, match="9999"):
        actual_totals(extra, weeks, levels, ["1000", "2000"], [2], 3)


def test_target_week_with_no_price_level_raises(basic):
    weekly, weeks, _ = basic
    levels = make_levels(0, 3)     # no level for week 4 and on
    with pytest.raises(ValueError, match=r"no price level for weeks \[4, 5\]"):
        actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 3)


# --- purity ----------------------------------------------------------------------------------

def test_inputs_are_not_modified(basic):
    weekly, weeks, levels = basic
    companies = ["2000", "1000"]
    origins = [4, 2, 4]
    copies = (weekly.copy(deep=True), weeks.copy(deep=True), levels.copy(deep=True))
    actual_totals(weekly, weeks, levels, companies, origins, 3)
    pd.testing.assert_frame_equal(weekly, copies[0])
    pd.testing.assert_frame_equal(weeks, copies[1])
    pd.testing.assert_series_equal(levels, copies[2])
    assert companies == ["2000", "1000"]
    assert origins == [4, 2, 4]


def test_same_inputs_give_the_same_result(basic):
    weekly, weeks, levels = basic
    first = actual_totals(weekly, weeks, levels, ["1000", "2000"], [0, 2, 4], 4)
    second = actual_totals(weekly, weeks, levels, ["1000", "2000"], [0, 2, 4], 4)
    pd.testing.assert_frame_equal(first, second)


def test_order_of_the_weekly_rows_does_not_matter(basic):
    weekly, weeks, levels = basic
    shuffled = weekly.sample(frac=1.0, random_state=3).reset_index(drop=True)
    first = actual_totals(weekly, weeks, levels, ["1000", "2000"], [0, 2, 4], 4)
    second = actual_totals(shuffled, weeks, levels, ["1000", "2000"], [0, 2, 4], 4)
    pd.testing.assert_frame_equal(first, second)


# --- dtypes of an empty result ---------------------------------------------------------------

def assert_actual_dtypes(result):
    assert result["company_code"].dtype == object
    for column in ["origin_week", "h", "target_week"]:
        assert result[column].dtype == np.int64, column
    assert result["actual"].dtype == np.float64
    assert result["actual_real"].dtype == np.float64


def test_empty_result_with_no_usable_target_week_has_the_right_dtypes(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [10, 20], 3)
    assert result.empty
    assert list(result.columns) == ACTUAL_COLUMNS
    assert_actual_dtypes(result)


def test_empty_result_with_only_missing_target_weeks_has_the_right_dtypes(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [5], 1)   # week 6 missing
    assert result.empty
    assert_actual_dtypes(result)


def test_empty_result_with_no_origins_has_the_right_dtypes(basic):
    weekly, weeks, levels = basic
    result = actual_totals(weekly, weeks, levels, ["1000", "2000"], [], 3)
    assert result.empty
    assert list(result.columns) == ACTUAL_COLUMNS
    assert_actual_dtypes(result)


def test_empty_result_with_empty_weekly_and_no_origins_has_the_right_dtypes():
    result = actual_totals(make_weekly([]), make_weeks(5), make_levels(), ["A"], [], 2)
    assert result.empty
    assert_actual_dtypes(result)


def test_empty_companies_is_still_an_error_even_with_no_origins(basic):
    weekly, weeks, levels = basic
    with pytest.raises(ValueError, match="company list is empty"):
        actual_totals(weekly, weeks, levels, [], [], 3)


def test_non_empty_result_has_the_same_dtypes_as_an_empty_one(basic):
    weekly, weeks, levels = basic
    full = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2], 3)
    empty = actual_totals(weekly, weeks, levels, ["1000", "2000"], [], 3)
    assert_actual_dtypes(full)
    assert full.dtypes.equals(empty.dtypes)


@pytest.mark.parametrize("empty_first", [True, False])
def test_concatenating_an_empty_result_with_a_non_empty_one_keeps_the_dtypes(basic,
                                                                            empty_first):
    weekly, weeks, levels = basic
    full = actual_totals(weekly, weeks, levels, ["1000", "2000"], [2, 3], 3)
    empty = actual_totals(weekly, weeks, levels, ["1000", "2000"], [10], 3)
    parts = [empty, full] if empty_first else [full, empty]
    joined = pd.concat(parts, ignore_index=True)
    assert_actual_dtypes(joined)
    pd.testing.assert_frame_equal(joined, full)


# =============================================================================================
# U14.2: `rolling_forecasts`, `forecast.forecast` at every origin, stacked.
#
# The small training table, settings and side tables are the ones test_forecast.py builds for
# `forecast` (8 customers, companies TR02 and TR04, origins 0..30, h 1..5, ten trees), so one
# origin takes well under a second. Nothing reads data/.
# =============================================================================================

import time  # noqa: E402
from datetime import date, timedelta  # noqa: E402

from collection_estimation.hurdle import evaluate as evaluate_module  # noqa: E402
from collection_estimation.hurdle import forecast as forecast_module  # noqa: E402
from collection_estimation.hurdle.evaluate import rolling_forecasts  # noqa: E402
from collection_estimation.hurdle.forecast import Forecast, forecast  # noqa: E402
from collection_estimation.hurdle.inflation import InflationCoverageError  # noqa: E402
from collection_estimation.hurdle.prices import projected_levels  # noqa: E402
from tests.hurdle.test_forecast import (  # noqa: E402
    F_COMPANIES, WEEK_ZERO, f_inputs, f_settings, f_side_tables, make_table)

R_ORIGINS = [18, 22, 26]


@pytest.fixture(scope="module")
def r_table():
    return make_table()


def r_run(table, origins=R_ORIGINS, settings=None, on_origin_done=None, **overrides):
    kwargs = f_inputs()
    kwargs.update(overrides)
    return rolling_forecasts(table, origins, settings or f_settings(),
                             on_origin_done=on_origin_done, **kwargs)


def r_by_hand(table, origins, settings=None, **overrides):
    """`forecast` called by hand at each origin and stacked as the docstring says."""
    kwargs = f_inputs()
    kwargs.update(overrides)
    results = [forecast(table, origin, settings or f_settings(), **kwargs) for origin in origins]
    return (pd.concat([r.totals for r in results], ignore_index=True),
            pd.concat([r.customers for r in results]))


@pytest.fixture(scope="module")
def r_result(r_table):
    return r_run(r_table)


def spy_on_the_fits(monkeypatch):
    """Replace `forecast` (as evaluate sees it) and the stage fits with recorders that fail."""
    called = []

    def recorder(name):
        def step(*args, **kwargs):
            called.append(name)
            raise AssertionError(f"{name} must not be called")
        return step

    monkeypatch.setattr(evaluate_module, "forecast", recorder("forecast"))
    for name in ("fit_stage1", "fit_stage2", "customer_forecasts"):
        monkeypatch.setattr(forecast_module, name, recorder(name))
    return called


def refused_before_any_fit(monkeypatch, table, origins=R_ORIGINS, settings=None, **overrides):
    """Run with every fit spied on; return the ValueError's message (no fit may have run)."""
    called = spy_on_the_fits(monkeypatch)
    with pytest.raises(ValueError) as error:
        r_run(table, origins, settings, **overrides)
    assert called == [], called
    return str(error.value)


def with_label_repeated(table, pairs):
    """A copy of `table` whose index gives row `copy_to` the label of row `copy_from`."""
    table = table.copy()
    labels = list(table.index)
    for copy_from, copy_to in pairs:
        labels[copy_to] = labels[copy_from]
    table.index = pd.Index(labels)
    return table


# --- the result is forecast() at every origin, stacked -----------------------------------

def test_rolling_returns_a_forecast(r_result):
    assert isinstance(r_result, Forecast)


def test_rolling_equals_forecast_by_hand_per_origin_stacked(r_table, r_result):
    totals, customers = r_by_hand(r_table, R_ORIGINS)
    pd.testing.assert_frame_equal(r_result.totals, totals)
    pd.testing.assert_frame_equal(r_result.customers, customers)


def test_rolling_totals_have_a_fresh_range_index(r_result):
    assert list(r_result.totals.index) == list(range(len(r_result.totals)))


def test_rolling_customers_keep_the_tables_index(r_table, r_result):
    index = r_result.customers.index
    assert not isinstance(index, pd.RangeIndex)
    assert index.is_unique
    assert set(index) <= set(r_table.index)
    # Each customer row is the table's row with that label.
    at = r_table.loc[index]
    for column in ("origin_week", "h", "key"):
        assert (at[column].to_numpy() == r_result.customers[column].to_numpy()).all(), column


def test_rolling_customers_are_every_row_at_every_origin(r_table, r_result):
    wanted = r_table.index[r_table["origin_week"].isin(R_ORIGINS)]
    assert set(r_result.customers.index) == set(wanted)
    assert len(r_result.customers) == len(wanted)


def test_rolling_stacks_in_origin_order(r_result):
    for frame in (r_result.totals, r_result.customers):
        origins = frame["origin_week"].to_numpy()
        assert (np.diff(origins) >= 0).all()
        assert sorted(set(origins)) == R_ORIGINS


@pytest.mark.parametrize("origin", R_ORIGINS)
def test_each_origins_slice_equals_a_single_forecast(r_table, r_result, origin):
    alone = forecast(r_table, origin, f_settings(), **f_inputs())
    totals = r_result.totals[r_result.totals["origin_week"] == origin].reset_index(drop=True)
    customers = r_result.customers[r_result.customers["origin_week"] == origin]
    pd.testing.assert_frame_equal(totals, alone.totals)
    pd.testing.assert_frame_equal(customers, alone.customers)


@pytest.mark.parametrize("origin", R_ORIGINS)
def test_each_origins_slice_equals_rolling_at_that_origin_alone(r_table, r_result, origin):
    alone = r_run(r_table, [origin])
    totals = r_result.totals[r_result.totals["origin_week"] == origin].reset_index(drop=True)
    customers = r_result.customers[r_result.customers["origin_week"] == origin]
    pd.testing.assert_frame_equal(totals, alone.totals)
    pd.testing.assert_frame_equal(customers, alone.customers)


def test_a_single_origin_equals_forecast(r_table):
    alone = forecast(r_table, 22, f_settings(), **f_inputs())
    rolled = r_run(r_table, [22])
    pd.testing.assert_frame_equal(rolled.totals, alone.totals.reset_index(drop=True))
    pd.testing.assert_frame_equal(rolled.customers, alone.customers)


@pytest.mark.parametrize("given", [[26, 18, 22], [22, 26, 18, 22, 18], (26, 22, 18),
                                   {18, 22, 26}],
                         ids=["unsorted", "repeated", "tuple", "set"])
def test_unsorted_or_repeated_origins_equal_sorted_unique(r_table, r_result, given):
    found = r_run(r_table, given)
    pd.testing.assert_frame_equal(found.totals, r_result.totals)
    pd.testing.assert_frame_equal(found.customers, r_result.customers)


def test_each_origin_is_forecast_once_in_sorted_order(r_table, monkeypatch):
    seen = []
    real = evaluate_module.forecast

    def recording(table, origin, settings, **kwargs):
        seen.append(origin)
        return real(table, origin, settings, **kwargs)

    monkeypatch.setattr(evaluate_module, "forecast", recording)
    r_run(r_table, [26, 18, 26, 22])
    assert seen == [18, 22, 26]


def test_forecast_gets_the_documented_arguments(r_table, monkeypatch):
    calls = []
    real = evaluate_module.forecast

    def recording(table, origin, settings, **kwargs):
        calls.append((table, origin, settings, kwargs))
        return real(table, origin, settings, **kwargs)

    monkeypatch.setattr(evaluate_module, "forecast", recording)
    settings = f_settings()
    kwargs = f_inputs()
    rolling_forecasts(r_table, [18, 22], settings, **kwargs)
    assert [c[1] for c in calls] == [18, 22]
    for table, _, given_settings, given in calls:
        assert table is r_table
        assert given_settings is settings
        assert set(given) == set(kwargs)
        for name, value in kwargs.items():
            assert given[name] is value, name


def test_rolling_is_deterministic(r_table, r_result):
    again = r_run(r_table)
    pd.testing.assert_frame_equal(again.totals, r_result.totals)
    pd.testing.assert_frame_equal(again.customers, r_result.customers)


def test_rolling_with_other_settings_equals_by_hand(r_table):
    settings = f_settings(horizons=3, method="ridge", with_level=False)
    table = r_table[r_table["h"] <= 3]
    found = r_run(table, [26, 20], settings)
    totals, customers = r_by_hand(table, [20, 26], settings)
    pd.testing.assert_frame_equal(found.totals, totals)
    pd.testing.assert_frame_equal(found.customers, customers)


# --- the callback --------------------------------------------------------------------------

def test_callback_is_called_once_per_origin_in_order(r_table):
    calls = []
    r_run(r_table, [26, 18, 22, 18], on_origin_done=lambda *args: calls.append(args))
    assert [(done, total, origin) for done, total, origin, _ in calls] == [
        (1, 3, 18), (2, 3, 22), (3, 3, 26)]


def test_callback_seconds_are_non_negative_floats(r_table):
    calls = []
    r_run(r_table, on_origin_done=lambda *args: calls.append(args))
    assert len(calls) == 3
    for _, _, _, seconds in calls:
        assert isinstance(seconds, float)
        assert seconds >= 0.0


def test_callback_gets_four_positional_arguments(r_table):
    calls = []

    def callback(*args, **kwargs):
        assert kwargs == {}
        calls.append(args)

    r_run(r_table, [22], on_origin_done=callback)
    assert len(calls) == 1
    assert len(calls[0]) == 4
    assert calls[0][:3] == (1, 1, 22)


def test_callback_runs_after_each_origins_forecast_before_the_next(r_table, monkeypatch):
    events = []
    real = evaluate_module.forecast

    def recording(table, origin, settings, **kwargs):
        events.append(("forecast", origin))
        return real(table, origin, settings, **kwargs)

    monkeypatch.setattr(evaluate_module, "forecast", recording)
    r_run(r_table, on_origin_done=lambda done, total, origin, s: events.append(("done", origin)))
    assert events == [("forecast", 18), ("done", 18), ("forecast", 22), ("done", 22),
                      ("forecast", 26), ("done", 26)]


def test_callback_seconds_time_the_forecast_of_that_origin(r_table, monkeypatch):
    # A forecast that sleeps 0.05 s: the reported time covers it.
    real = evaluate_module.forecast

    def slow(table, origin, settings, **kwargs):
        time.sleep(0.05)
        return real(table, origin, settings, **kwargs)

    monkeypatch.setattr(evaluate_module, "forecast", slow)
    calls = []
    r_run(r_table, [22], on_origin_done=lambda *args: calls.append(args))
    assert calls[0][3] >= 0.05


def test_callback_does_not_change_the_result(r_table, r_result):
    with_callback = r_run(r_table, on_origin_done=lambda *args: None)
    pd.testing.assert_frame_equal(with_callback.totals, r_result.totals)
    pd.testing.assert_frame_equal(with_callback.customers, r_result.customers)


def test_no_callback_is_the_default(r_table, r_result):
    found = rolling_forecasts(r_table, R_ORIGINS, f_settings(), **f_inputs())
    pd.testing.assert_frame_equal(found.totals, r_result.totals)


def test_callback_is_not_called_when_refused_up_front(r_table):
    calls = []
    with pytest.raises(ValueError):
        r_run(r_table, [18, 99], on_origin_done=lambda *args: calls.append(args))
    assert calls == []


# --- refused before any fit -------------------------------------------------------------

@pytest.mark.parametrize("origins", [[], (), set()], ids=["list", "tuple", "set"])
def test_no_origins_exact_message(r_table, monkeypatch, origins):
    message = refused_before_any_fit(monkeypatch, r_table, origins)
    assert message == "there are no origins to forecast at"


def test_duplicate_index_label_exact_message(r_table, monkeypatch):
    table = with_label_repeated(r_table, [(0, 1), (2, 3)])
    first = [int(r_table.index[0]), int(r_table.index[2])]
    message = refused_before_any_fit(monkeypatch, table)
    assert message == (f"the table's index holds 2 labels more than once (first: {first}); "
                       f"the customer rows are joined by index")


def test_duplicate_index_message_shows_at_most_five_labels(r_table, monkeypatch):
    table = with_label_repeated(r_table, [(2 * i, 2 * i + 1) for i in range(7)])
    message = refused_before_any_fit(monkeypatch, table)
    assert message.startswith("the table's index holds 7 labels more than once (first: [")
    shown = message.split("(first: [")[1].split("]")[0].split(", ")
    assert len(shown) == 5


def test_a_label_three_times_counts_once(r_table, monkeypatch):
    table = with_label_repeated(r_table, [(0, 1), (0, 2)])
    message = refused_before_any_fit(monkeypatch, table)
    assert message.startswith("the table's index holds 1 labels more than once")


def test_duplicate_index_outside_the_origins_is_still_refused(r_table, monkeypatch):
    # The two rows sharing a label are at origin 0, not one of the origins forecast.
    at_zero = np.flatnonzero(r_table["origin_week"].to_numpy() == 0)
    table = with_label_repeated(r_table, [(at_zero[0], at_zero[1])])
    message = refused_before_any_fit(monkeypatch, table)
    assert "more than once" in message


def test_all_in_companies_exact_message(r_table, monkeypatch):
    message = refused_before_any_fit(monkeypatch, r_table, companies=["TR02", "TR04", "ALL"])
    assert message == "'ALL' is the code of the consolidated rows; it cannot be a company"


def test_origin_with_no_rows_exact_message(r_table, monkeypatch):
    message = refused_before_any_fit(monkeypatch, r_table, [99])
    assert message == "the table has no rows at origin week 99"


def test_only_the_last_origin_missing_is_found_before_any_fit(r_table, monkeypatch):
    # 18 and 22 are fine; the missing one sorts last, so a check inside the loop would fit
    # 18 and 22 first.
    message = refused_before_any_fit(monkeypatch, r_table, [22, 99, 18])
    assert message == "the table has no rows at origin week 99"


def test_an_origin_between_good_ones_with_no_rows(r_table, monkeypatch):
    table = r_table[r_table["origin_week"] != 22]
    message = refused_before_any_fit(monkeypatch, table, [18, 22, 26])
    assert message == "the table has no rows at origin week 22"


def test_the_first_missing_origin_in_sorted_order_is_named(r_table, monkeypatch):
    message = refused_before_any_fit(monkeypatch, r_table, [120, 99, 18])
    assert message == "the table has no rows at origin week 99"


@pytest.mark.parametrize("horizons", [3, 4, 6])
def test_horizons_setting_unlike_the_table_exact_message(r_table, monkeypatch, horizons):
    message = refused_before_any_fit(monkeypatch, r_table, settings=f_settings(horizons))
    assert message == (f"the table's rows at origin week 18 are for h [1, 2, 3, 4, 5], but "
                       f"[receivables] horizons is {horizons}")


def test_only_the_last_origin_missing_an_h_is_found_before_any_fit(r_table, monkeypatch):
    table = r_table[~((r_table["origin_week"] == 26) & (r_table["h"] == 5))]
    message = refused_before_any_fit(monkeypatch, table)
    assert message == ("the table's rows at origin week 26 are for h [1, 2, 3, 4], but "
                       "[receivables] horizons is 5")


def test_only_the_last_origin_with_an_extra_h_is_found_before_any_fit(r_table, monkeypatch):
    extra = r_table[(r_table["origin_week"] == 26) & (r_table["h"] == 5)].copy()
    extra["h"] = 6
    extra["target_week"] = 32
    extra.index = extra.index + 100_000
    table = pd.concat([r_table, extra])
    message = refused_before_any_fit(monkeypatch, table)
    assert message == ("the table's rows at origin week 26 are for h [1, 2, 3, 4, 5, 6], but "
                       "[receivables] horizons is 5")


def test_h_check_lists_plain_ints_for_a_float_h_column(r_table, monkeypatch):
    table = r_table[~((r_table["origin_week"] == 26) & (r_table["h"] == 5))].copy()
    table["h"] = table["h"].astype("float64")
    message = refused_before_any_fit(monkeypatch, table)
    assert message.endswith("are for h [1, 2, 3, 4], but [receivables] horizons is 5")


def test_a_float_h_column_that_matches_passes_the_check(r_table):
    table = r_table.copy()
    table["h"] = table["h"].astype("float64")
    totals, customers = r_by_hand(table, [22])
    found = r_run(table, [22])
    pd.testing.assert_frame_equal(found.totals, totals)
    pd.testing.assert_frame_equal(found.customers, customers)


def test_a_mismatch_at_an_origin_not_forecast_is_not_refused(r_table):
    # Origin 10 lacks h 5; only origin 26 is forecast, so the check passes. (The rows of
    # origin 10 are still training rows, so the forecast is compared with forecast() on the
    # same table, not with the full table's.)
    table = r_table[~((r_table["origin_week"] == 10) & (r_table["h"] == 5))]
    found = r_run(table, [26])
    totals, customers = r_by_hand(table, [26])
    pd.testing.assert_frame_equal(found.totals, totals)
    pd.testing.assert_frame_equal(found.customers, customers)


def test_unknown_topup_method_is_refused_before_any_fit(r_table, monkeypatch):
    message = refused_before_any_fit(monkeypatch, r_table, settings=f_settings(method="median"))
    assert message == "method is 'median'; it must be one of ('mean', 'ridge')"


def test_topup_window_below_one_is_refused_before_any_fit(r_table, monkeypatch):
    message = refused_before_any_fit(monkeypatch, r_table, settings=f_settings(window=0))
    assert message == "window is 0; the trailing mean needs at least 1 week"


def test_topup_data_missing_only_for_the_last_origin_is_found_before_any_fit(r_table,
                                                                              monkeypatch):
    # The new-customer cash of target weeks 23..26 removed: origin 26 has no week with data in
    # its 4-week window, 18 and 22 do (their windows end at 18 and 22).
    new_cash, refunds, unidentified = f_side_tables()
    new_cash = new_cash[~new_cash["target_week"].between(23, 26)]
    message = refused_before_any_fit(monkeypatch, r_table, settings=f_settings(window=4),
                                     new_cash=new_cash, refunds=refunds,
                                     unidentified=unidentified)
    assert message == ("h 1, origin 26: no week with data in the 4 weeks up to the origin, so "
                       "no trailing mean")


def test_non_monday_week_zero_is_refused_before_any_fit(r_table, monkeypatch):
    message = refused_before_any_fit(monkeypatch, r_table, week_zero=date(2025, 6, 3))
    assert message == ("week_zero 2025-06-03 is a Tuesday, not a Monday; weeks run Monday to "
                       "Sunday")


def a_week_zero_where_only_the_later_origin_lacks_tufe(early, late):
    """A Monday from which `projected_levels` works at `early` but raises at `late`."""
    day = date(2015, 1, 5)   # a Monday
    while day < date(2040, 1, 1):
        try:
            projected_levels(day, early, [early + 1])
        except InflationCoverageError:
            day += timedelta(days=7)
            continue
        try:
            projected_levels(day, late, [late + 1])
        except InflationCoverageError:
            return day
        day += timedelta(days=7)
    raise AssertionError("no such week_zero: the TUFE table never ends?")


def test_price_projection_failing_only_for_the_last_origin_is_found_before_any_fit(
        r_table, monkeypatch):
    week_zero = a_week_zero_where_only_the_later_origin_lacks_tufe(22, 26)
    called = spy_on_the_fits(monkeypatch)
    with pytest.raises(InflationCoverageError) as error:
        r_run(r_table, week_zero=week_zero)
    assert called == []
    assert str(error.value).startswith("week 26 ")


# --- the order of the up-front checks ---------------------------------------------------

def test_order_no_origins_before_duplicate_index(r_table, monkeypatch):
    table = with_label_repeated(r_table, [(0, 1)])
    message = refused_before_any_fit(monkeypatch, table, [])
    assert message == "there are no origins to forecast at"


def test_order_duplicate_index_before_all(r_table, monkeypatch):
    table = with_label_repeated(r_table, [(0, 1)])
    message = refused_before_any_fit(monkeypatch, table, companies=["ALL"])
    assert "more than once" in message


def test_order_all_before_missing_origin(r_table, monkeypatch):
    message = refused_before_any_fit(monkeypatch, r_table, [99], companies=["TR02", "ALL"])
    assert message.startswith("'ALL' is the code")


def test_order_missing_origin_before_h_mismatch_of_a_later_origin(r_table, monkeypatch):
    table = r_table[~((r_table["origin_week"] == 26) & (r_table["h"] == 5))]
    message = refused_before_any_fit(monkeypatch, table, [26, 20, 21.5])
    # 20 is fine, 21.5 has no rows and sorts before 26.
    assert message == "the table has no rows at origin week 21.5"


def test_order_table_checks_before_the_topups(r_table, monkeypatch):
    message = refused_before_any_fit(monkeypatch, r_table, [18, 99],
                                     settings=f_settings(method="median"))
    assert message == "the table has no rows at origin week 99"


def test_order_topups_before_the_price_projection(r_table, monkeypatch):
    message = refused_before_any_fit(monkeypatch, r_table, settings=f_settings(window=0),
                                     week_zero=date(2025, 6, 3))
    assert message == "window is 0; the trailing mean needs at least 1 week"


def test_topups_are_made_once_for_all_origins_up_front(r_table, monkeypatch):
    calls = []
    real = evaluate_module.topup_forecasts

    def recording(*args, **kwargs):
        calls.append(list(args[5]))
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "topup_forecasts", recording)
    spy_on_the_fits(monkeypatch)
    with pytest.raises(AssertionError, match="must not be called"):
        r_run(r_table, [26, 18, 22])
    assert calls == [[18, 22, 26]]


def test_levels_are_projected_for_every_origin_up_front(r_table, monkeypatch):
    calls = []
    real = evaluate_module.projected_levels

    def recording(week_zero, origin, targets, *args, **kwargs):
        calls.append((week_zero, origin, list(targets)))
        return real(week_zero, origin, targets, *args, **kwargs)

    monkeypatch.setattr(evaluate_module, "projected_levels", recording)
    spy_on_the_fits(monkeypatch)
    with pytest.raises(AssertionError, match="must not be called"):
        r_run(r_table, [26, 18, 22])
    assert calls == [(WEEK_ZERO, o, [o + h for h in range(1, 6)]) for o in [18, 22, 26]]


# --- inputs are not changed -------------------------------------------------------------

def test_rolling_does_not_change_its_inputs(r_table):
    table = r_table.copy(deep=True)
    kwargs = f_inputs()
    frames = {name: kwargs[name].copy(deep=True)
              for name in ("new_cash", "refunds", "unidentified", "calendar", "levels")}
    block_columns = {key: list(value) for key, value in kwargs["block_columns"].items()}
    origins = [26, 18, 22, 18]
    settings = f_settings()
    rolling_forecasts(table, origins, settings, **kwargs)
    pd.testing.assert_frame_equal(table, r_table)
    for name in ("new_cash", "refunds", "unidentified", "calendar"):
        pd.testing.assert_frame_equal(kwargs[name], frames[name])
    pd.testing.assert_series_equal(kwargs["levels"], frames["levels"])
    assert kwargs["block_columns"] == block_columns
    assert kwargs["companies"] == list(F_COMPANIES)
    assert kwargs["week_zero"] == WEEK_ZERO
    assert origins == [26, 18, 22, 18]
    assert settings == f_settings()


def test_rolling_result_is_not_a_view_of_the_table(r_table):
    table = r_table.copy(deep=True)
    found = rolling_forecasts(table, [22], f_settings(), **f_inputs())
    before = found.customers.copy(deep=True)
    table.loc[:, "origin_week"] = -1
    pd.testing.assert_frame_equal(found.customers, before)


# =============================================================================================
# U14.3: `deflated_reference`, `benchmark.deflated_forecasts` per company and consolidated at
# every origin, h and variant, re-inflated with the price level projected at the origin.
#
# Two companies, weeks 0..40 with data except 6 and 25 (each holds a huge amount that must
# never be read); company 2000 has no line in weeks 8, 19 and 27 (a real 0 there). The level of
# week w is 1 + w / 10 (make_levels), the calendar has two features; origins 20 and 30, h 1..3.
# Every expected value is rebuilt by hand from the weekly rows and compared with a direct call
# of `deflated_forecasts`. Nothing reads data/.
# =============================================================================================

from collection_estimation.hurdle import benchmark  # noqa: E402
from collection_estimation.hurdle.evaluate import (  # noqa: E402
    REFERENCE_COLUMNS, REFERENCES, deflated_reference)

D_COMPANIES = ["1000", "2000"]
D_ORIGINS = [20, 30]
D_HORIZONS = 3
D_LAST = 40
D_MISSING = {6, 25}
D_NO_LINE_2000 = {8, 19, 27}


def d_weekly(last=D_LAST, seed=3):
    rng = np.random.default_rng(seed)
    rows, n = [], 0
    for w in range(0, last + 1):
        huge = w in D_MISSING
        for customer in ("C1", None):     # two lines of company 1000, one unidentified
            n += 1
            gross = 1e9 if huge else 1000.0 + 150.0 * (w % 4) + rng.normal(scale=60.0)
            rows.append((f"K{n}", "1000", customer, w, gross, -abs(rng.normal(scale=20.0))))
        if w not in D_NO_LINE_2000:
            n += 1
            gross = 1e9 if huge else 400.0 + 90.0 * np.cos(w) + rng.normal(scale=30.0)
            rows.append((f"K{n}", "2000", "C2", w, gross, -abs(rng.normal(scale=10.0))))
    return make_weekly(rows)


def d_calendar(last=60):
    weeks = list(range(0, last + 1))
    return pd.DataFrame({"week": weeks, "cycle": [float(w % 4) for w in weeks],
                         "wave": [float(np.cos(w)) for w in weeks]})


def d_inputs(**overrides):
    kwargs = dict(weekly=d_weekly(), weeks=make_weeks(D_LAST, missing=D_MISSING),
                  levels=make_levels(0, 60), calendar=d_calendar(), companies=list(D_COMPANIES),
                  origins=list(D_ORIGINS), horizons=D_HORIZONS, week_zero=WEEK_ZERO)
    kwargs.update(overrides)
    return kwargs


def d_run(**overrides):
    return deflated_reference(**d_inputs(**overrides))


def d_series(weekly, weeks, levels, company=None):
    """The deflated series by hand: per week with data, the (company's or all) net cash,
    0.0 where there is no line, divided by the week's level; missing weeks left out."""
    with_data = sorted(weeks.loc[~weeks["is_missing"], "week"])
    values = {}
    for w in with_data:
        mine = weekly[weekly["week"] == w]
        if company is not None:
            mine = mine[mine["company_code"] == company]
        values[w] = float((mine["gross_receipts"] + mine["refunds"]).sum()) / float(levels[w])
    return pd.Series(values, dtype="float64")


def d_row(result, company, origin, h, reference):
    found = result[(result["company_code"] == company) & (result["origin_week"] == origin)
                   & (result["h"] == h) & (result["reference"] == reference)]
    assert len(found) == 1, (company, origin, h, reference, len(found))
    return found.iloc[0]


@pytest.fixture(scope="module")
def d_result():
    return d_run()


# --- constants -------------------------------------------------------------------------------

def test_reference_columns():
    assert REFERENCE_COLUMNS == ["company_code", "origin_week", "h", "target_week", "reference",
                                 "real_forecast", "price_level", "forecast", "fallback"]


def test_references_are_the_two_variants_with_level_first():
    assert REFERENCES == {"deflated": True, "deflated_nolevel": False}
    assert list(REFERENCES) == ["deflated", "deflated_nolevel"]


# --- shape, order, dtypes -------------------------------------------------------------------

def test_reference_has_exactly_the_reference_columns(d_result):
    assert list(d_result.columns) == REFERENCE_COLUMNS


def test_reference_row_count(d_result):
    assert len(d_result) == (len(D_COMPANIES) + 1) * len(D_ORIGINS) * D_HORIZONS * 2


def test_reference_row_order(d_result):
    expected = [(c, o, h, r) for c in D_COMPANIES + [CONSOLIDATED] for o in D_ORIGINS
                for h in range(1, D_HORIZONS + 1) for r in REFERENCES]
    found = list(zip(d_result["company_code"], d_result["origin_week"], d_result["h"],
                     d_result["reference"]))
    assert found == expected
    assert d_result.index.equals(pd.RangeIndex(len(d_result)))


def test_reference_code_columns_are_object_as_in_actual_totals(d_result):
    # actual_totals casts company_code to object explicitly; the reference is joined to it
    # (and to the forecast totals) on company_code, so the same dtype is expected here.
    assert d_result["company_code"].dtype == object
    assert d_result["reference"].dtype == object


def test_reference_code_columns_hold_the_codes_as_strings(d_result):
    assert all(isinstance(v, str) for v in d_result["company_code"])
    assert all(isinstance(v, str) for v in d_result["reference"])


def test_reference_dtypes(d_result):
    for name in ("origin_week", "h", "target_week"):
        assert d_result[name].dtype == "int64", name
    for name in ("real_forecast", "price_level", "forecast"):
        assert d_result[name].dtype == "float64", name
    assert d_result["fallback"].dtype == bool


def test_reference_target_week_is_origin_plus_h(d_result):
    assert (d_result["target_week"] == d_result["origin_week"] + d_result["h"]).all()


def test_reference_values_are_finite(d_result):
    for name in ("real_forecast", "price_level", "forecast"):
        assert np.isfinite(d_result[name]).all(), name


def test_reference_duplicate_and_unsorted_origins_collapse(d_result):
    found = d_run(origins=[30, 20, 30, 20, 20])
    pd.testing.assert_frame_equal(found, d_result)


def test_reference_duplicate_and_unsorted_companies_collapse(d_result):
    found = d_run(companies=["2000", "1000", "2000"])
    pd.testing.assert_frame_equal(found, d_result)


def test_reference_all_comes_last_even_when_a_company_sorts_after_it():
    weekly = d_weekly()
    weekly["company_code"] = weekly["company_code"].map({"1000": "ZZZ", "2000": "AAA"})
    found = d_run(weekly=weekly, companies=["ZZZ", "AAA"], origins=[20], horizons=1)
    assert list(found["company_code"]) == ["AAA", "AAA", "ZZZ", "ZZZ", "ALL", "ALL"]


def test_reference_integer_company_codes_work_and_sort_numerically():
    weekly = d_weekly()
    weekly["company_code"] = weekly["company_code"].map({"1000": 10, "2000": 9})
    found = d_run(weekly=weekly, companies=[10, 9], origins=[20], horizons=1)
    assert list(found["company_code"]) == [9, 9, 10, 10, "ALL", "ALL"]


def test_reference_company_with_no_cash_at_all_gets_zero_and_falls_back():
    found = d_run(companies=["1000", "2000", "3000"], origins=[20], horizons=2)
    part = found[found["company_code"] == "3000"]
    assert len(part) == 4
    assert (part["real_forecast"] == 0.0).all()
    assert (part["forecast"] == 0.0).all()
    assert part["fallback"].all()     # a prediction of 0 is not above 0


def test_reference_single_origin_single_horizon():
    found = d_run(origins=[20], horizons=1)
    assert len(found) == 3 * 1 * 1 * 2
    assert set(found["h"]) == {1}


# --- the values: by hand against deflated_forecasts -------------------------------------------

@pytest.mark.parametrize("company", ["1000", "2000", CONSOLIDATED])
def test_reference_equals_deflated_forecasts_on_the_hand_built_series(d_result, company):
    kw = d_inputs()
    series = d_series(kw["weekly"], kw["weeks"], kw["levels"],
                      None if company == CONSOLIDATED else company)
    for origin in D_ORIGINS:
        for h in range(1, D_HORIZONS + 1):
            for reference, with_level in REFERENCES.items():
                made = benchmark.deflated_forecasts(series, kw["calendar"], h, origin, [origin],
                                                    with_level).iloc[0]
                found = d_row(d_result, company, origin, h, reference)
                assert found["real_forecast"] == pytest.approx(made["forecast"], rel=1e-9), \
                    (company, origin, h, reference)
                assert bool(found["fallback"]) == bool(made["fallback"])


def test_hand_built_series_holds_the_zero_weeks_and_leaves_the_missing_weeks_out():
    kw = d_inputs()
    series = d_series(kw["weekly"], kw["weeks"], kw["levels"], "2000")
    assert not set(series.index) & D_MISSING
    for w in D_NO_LINE_2000:
        assert series[w] == 0.0
    assert len(series) == D_LAST + 1 - len(D_MISSING)


def test_reference_the_zero_filled_week_is_in_the_series_not_left_out():
    # Origin 8, h 1: under 12 training weeks, so the forecast is the recent level, the mean of
    # weeks 5, 7, 8 (6 is missing). Company 2000 has no line in week 8: a real 0 in the mean.
    found = d_run(origins=[8], horizons=1)
    weekly = d_weekly()
    net = weekly.assign(net=weekly["gross_receipts"] + weekly["refunds"])

    def real(w):
        return net[(net["company_code"] == "2000") & (net["week"] == w)]["net"].sum() / level(w)

    expected = (real(5) + real(7) + 0.0) / 3
    for reference in REFERENCES:
        row = d_row(found, "2000", 8, 1, reference)
        assert row["fallback"]
        assert row["real_forecast"] == pytest.approx(expected, rel=1e-12)
        # Leaving week 8 out would give the mean of two weeks instead.
        assert row["real_forecast"] != pytest.approx((real(5) + real(7)) / 2, rel=1e-6)


def test_reference_cash_in_a_missing_week_is_never_read(d_result):
    # The missing weeks hold 1e9; they must not reach any forecast, nor the level.
    assert d_result["real_forecast"].abs().max() < 1e5
    weekly = d_weekly()
    weekly.loc[weekly["week"].isin(D_MISSING), "gross_receipts"] = -5e12
    pd.testing.assert_frame_equal(d_run(weekly=weekly), d_result)


def test_reference_a_missing_week_without_a_price_level_does_not_raise(d_result):
    levels = make_levels(0, 60).drop(list(D_MISSING))
    pd.testing.assert_frame_equal(d_run(levels=levels), d_result)


def test_reference_several_lines_of_a_company_in_a_week_are_summed(d_result):
    # Split every line in two halves: the series, and so the result, must not change.
    halves = d_weekly()
    halves["gross_receipts"] = halves["gross_receipts"] / 2
    halves["refunds"] = halves["refunds"] / 2
    doubled = pd.concat([halves, halves.assign(key=halves["key"] + "y")], ignore_index=True)
    found = d_run(weekly=doubled)
    pd.testing.assert_frame_equal(found, d_result, check_exact=False, rtol=1e-9)


def test_reference_refunds_count_in_the_series():
    no_refunds = d_weekly().assign(refunds=0.0)
    found = d_run(weekly=no_refunds, origins=[8], horizons=1)
    with_refunds = d_run(origins=[8], horizons=1)
    # Recent level (fallback): refunds are negative, so dropping them raises every forecast.
    assert with_refunds["fallback"].all()
    assert (found["real_forecast"] > with_refunds["real_forecast"]).all()


def test_reference_forecast_is_real_forecast_times_price_level(d_result):
    assert (d_result["forecast"] == d_result["real_forecast"] * d_result["price_level"]).all()


def test_reference_price_level_is_projected_at_the_origin(d_result):
    for origin in D_ORIGINS:
        projected = projected_levels(WEEK_ZERO, origin,
                                     [origin + h for h in range(1, D_HORIZONS + 1)])
        part = d_result[d_result["origin_week"] == origin]
        expected = part["target_week"].map(projected)
        assert (part["price_level"].to_numpy() == expected.to_numpy()).all(), origin


def test_reference_price_level_is_not_the_published_level(d_result):
    # make_levels' 1 + w / 10 is nothing like the TÜFE: re-inflation must not use `levels`.
    assert not np.allclose(d_result["price_level"], d_result["target_week"].map(level))


def test_reference_price_level_is_the_same_for_every_company_and_variant(d_result):
    per_target = d_result.groupby(["origin_week", "h"])["price_level"].nunique()
    assert (per_target == 1).all()


def test_reference_same_target_from_two_origins_uses_each_origins_projection():
    found = d_run(origins=[20, 22], horizons=3)
    a = d_row(found, "1000", 20, 3, "deflated")["price_level"]     # target 23, seen at 20
    b = d_row(found, "1000", 22, 1, "deflated")["price_level"]     # target 23, seen at 22
    assert a == float(projected_levels(WEEK_ZERO, 20, [23]).loc[23])
    assert b == float(projected_levels(WEEK_ZERO, 22, [23]).loc[23])


def test_reference_the_two_variants_differ_somewhere(d_result):
    wide = d_result.pivot(index=["company_code", "origin_week", "h"], columns="reference",
                          values="real_forecast")
    assert not np.allclose(wide["deflated"], wide["deflated_nolevel"])


# --- consolidated ------------------------------------------------------------------------------

def test_reference_all_is_fitted_on_the_summed_series(d_result):
    kw = d_inputs()
    total = d_series(kw["weekly"], kw["weeks"], kw["levels"])
    by_company = sum(d_series(kw["weekly"], kw["weeks"], kw["levels"], c) for c in D_COMPANIES)
    pd.testing.assert_series_equal(total, by_company, check_exact=False, rtol=1e-12)
    for origin in D_ORIGINS:
        for h in range(1, D_HORIZONS + 1):
            for reference, with_level in REFERENCES.items():
                made = benchmark.deflated_forecasts(total, kw["calendar"], h, origin, [origin],
                                                    with_level).iloc[0]
                found = d_row(d_result, CONSOLIDATED, origin, h, reference)
                assert found["real_forecast"] == pytest.approx(made["forecast"], rel=1e-9)


def test_reference_all_is_not_the_sum_of_the_company_forecasts(d_result):
    keys = ["origin_week", "h", "reference"]
    companies = (d_result[d_result["company_code"] != CONSOLIDATED]
                 .groupby(keys)["real_forecast"].sum())
    consolidated = (d_result[d_result["company_code"] == CONSOLIDATED]
                    .set_index(keys)["real_forecast"])
    gap = (consolidated - companies.reindex(consolidated.index)).abs()
    assert not d_result[d_result["company_code"] == CONSOLIDATED]["fallback"].all()
    assert gap.max() > 1e-6


def test_reference_all_equals_the_sum_when_everything_falls_back():
    # Fallback = the recent level, a mean, which is additive across companies.
    found = d_run(origins=[8], horizons=2)
    assert found["fallback"].all()
    keys = ["origin_week", "h", "reference"]
    companies = found[found["company_code"] != CONSOLIDATED].groupby(keys)["real_forecast"].sum()
    consolidated = found[found["company_code"] == CONSOLIDATED].set_index(keys)["real_forecast"]
    # Compared by key tuples and values: groupby turns the object `reference` column into a
    # str index level under pandas 3, while set_index keeps it object; only the dtype differs.
    summed = dict(zip(companies.index.tolist(), companies.to_numpy()))
    assert sorted(summed) == sorted(consolidated.index.tolist())
    for key, value in zip(consolidated.index.tolist(), consolidated.to_numpy()):
        assert value == pytest.approx(summed[key], rel=1e-12), key


# --- fallback ----------------------------------------------------------------------------------

def test_reference_too_few_training_weeks_gives_the_recent_level_and_fallback():
    origin = 8      # weeks 0..8 minus week 6: 8 training weeks, below MIN_TRAINING_WEEKS
    assert 8 < benchmark.MIN_TRAINING_WEEKS
    found = d_run(origins=[origin], horizons=2)
    kw = d_inputs()
    for company in D_COMPANIES + [CONSOLIDATED]:
        series = d_series(kw["weekly"], kw["weeks"], kw["levels"],
                          None if company == CONSOLIDATED else company)
        expected = benchmark.recent_level(series, origin)
        for h in (1, 2):
            for reference in REFERENCES:
                row = d_row(found, company, origin, h, reference)
                assert bool(row["fallback"]) is True
                assert row["real_forecast"] == pytest.approx(expected, rel=1e-12)
                assert row["forecast"] == row["real_forecast"] * row["price_level"]


def test_reference_enough_training_weeks_gives_a_fitted_forecast(d_result):
    # At origin 30 there are well over 12 training weeks for every series and variant.
    part = d_result[d_result["origin_week"] == 30]
    assert not part["fallback"].any()


# --- target weeks without data ---------------------------------------------------------------

def test_reference_a_missing_target_week_still_gets_a_row(d_result):
    # Week 22 missing, origin 20, h 2: the target has no data but is still forecast.
    weeks = make_weeks(D_LAST, missing=D_MISSING | {22})
    found = d_run(weeks=weeks)
    for company in D_COMPANIES + [CONSOLIDATED]:
        for reference in REFERENCES:
            row = d_row(found, company, 20, 2, reference)
            assert row["target_week"] == 22 and np.isfinite(row["forecast"])
    assert len(found) == len(d_result)


def test_reference_target_week_past_the_weeks_table_still_gets_a_row():
    # Weeks run to 40; origin 39, h 3 forecasts week 42, which the calendar covers.
    found = d_run(origins=[39], horizons=3)
    assert list(found[found["company_code"] == "1000"]["target_week"]) == [40, 40, 41, 41,
                                                                          42, 42]


# --- no leak -----------------------------------------------------------------------------------

def at_origin(result, origin):
    return result[result["origin_week"] == origin].reset_index(drop=True)


@pytest.mark.parametrize("origin", D_ORIGINS)
def test_reference_no_leak_removing_weekly_rows_after_the_origin(d_result, origin):
    weekly = d_weekly()
    found = d_run(weekly=weekly[weekly["week"] <= origin], origins=[origin])
    pd.testing.assert_frame_equal(found, at_origin(d_result, origin))


@pytest.mark.parametrize("origin", D_ORIGINS)
def test_reference_no_leak_perturbing_weekly_rows_after_the_origin(d_result, origin):
    weekly = d_weekly()
    later = weekly["week"] > origin
    weekly.loc[later, "gross_receipts"] = weekly.loc[later, "gross_receipts"] * 13.0 + 7e5
    weekly.loc[later, "refunds"] = -9e4
    found = d_run(weekly=weekly, origins=[origin])
    pd.testing.assert_frame_equal(found, at_origin(d_result, origin))


@pytest.mark.parametrize("origin", D_ORIGINS)
def test_reference_no_leak_changing_the_weeks_after_the_origin(d_result, origin):
    later_missing = {w for w in range(origin + 1, D_LAST + 1) if w % 3 == 0}
    weeks = make_weeks(D_LAST, missing={w for w in D_MISSING if w <= origin} | later_missing)
    found = d_run(weeks=weeks, origins=[origin])
    pd.testing.assert_frame_equal(found, at_origin(d_result, origin))


@pytest.mark.parametrize("origin", D_ORIGINS)
def test_reference_no_leak_dropping_the_weeks_after_the_origin(d_result, origin):
    weeks = make_weeks(D_LAST, missing=D_MISSING)
    found = d_run(weeks=weeks[weeks["week"] <= origin], origins=[origin])
    pd.testing.assert_frame_equal(found, at_origin(d_result, origin))


@pytest.mark.parametrize("origin", D_ORIGINS)
def test_reference_no_leak_changing_the_price_levels_after_the_origin(d_result, origin):
    levels = make_levels(0, 60)
    levels[levels.index > origin] = levels[levels.index > origin] * 3.0
    found = d_run(levels=levels, origins=[origin])
    pd.testing.assert_frame_equal(found, at_origin(d_result, origin))


def test_reference_each_origin_is_independent_of_the_others(d_result):
    for origin in D_ORIGINS:
        pd.testing.assert_frame_equal(d_run(origins=[origin]), at_origin(d_result, origin))


def test_reference_horizon_count_does_not_change_the_shorter_horizons(d_result):
    found = d_run(horizons=1)
    expected = d_result[d_result["h"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(found, expected)


# --- refusals ----------------------------------------------------------------------------------

def test_reference_empty_origins_raise():
    with pytest.raises(ValueError, match="there are no origins to forecast at"):
        d_run(origins=[])


@pytest.mark.parametrize("horizons", [0, -1])
def test_reference_horizons_below_one_raise(horizons):
    with pytest.raises(ValueError, match=f"horizons is {horizons}; horizons start at 1"):
        d_run(horizons=horizons)


def test_reference_empty_company_list_raises():
    with pytest.raises(ValueError, match="the company list is empty"):
        d_run(companies=[])


def test_reference_all_in_the_company_list_raises():
    with pytest.raises(ValueError, match="'ALL' is the code of the consolidated rows"):
        d_run(companies=["1000", "2000", "ALL"])


def test_reference_weekly_cash_of_a_company_outside_the_list_raises_and_names_it():
    with pytest.raises(ValueError, match=r"companies not in the list: \['2000'\]"):
        d_run(companies=["1000"])


def test_reference_company_outside_the_list_raises_even_if_only_in_a_missing_week():
    weekly = pd.concat([d_weekly(), make_weekly([("KX", "9000", "C9", 6, 1.0, 0.0)])],
                       ignore_index=True)
    with pytest.raises(ValueError, match=r"companies not in the list: \['9000'\]"):
        d_run(weekly=weekly)


def test_reference_week_with_data_and_no_price_level_raises():
    levels = make_levels(0, 60).drop([12])
    with pytest.raises(ValueError, match=r"no price level for weeks \[12\]"):
        d_run(levels=levels)


def test_reference_target_week_without_a_calendar_row_raises():
    with pytest.raises(ValueError, match=r"the calendar has no rows for weeks \[33\]"):
        d_run(calendar=d_calendar(last=32), origins=[30], horizons=3)


def test_reference_training_week_without_a_calendar_row_raises():
    calendar = d_calendar()
    calendar = calendar[calendar["week"] != 10]
    with pytest.raises(ValueError, match=r"the calendar has no rows for weeks \[10\]"):
        d_run(calendar=calendar)


def test_reference_no_data_in_the_recent_level_weeks_raises():
    # Weeks 17..20 all missing: at origin 20 there is no recent level.
    weeks = make_weeks(D_LAST, missing=D_MISSING | {17, 18, 19, 20})
    with pytest.raises(ValueError, match="origin 20: no week with data in the 4 weeks up to "
                                         "the origin, so no recent level"):
        d_run(weeks=weeks)


def test_reference_three_of_four_level_weeks_missing_does_not_raise():
    weeks = make_weeks(D_LAST, missing=D_MISSING | {17, 18, 19})
    found = d_run(weeks=weeks, origins=[20], horizons=1)
    assert len(found) == 6


def test_reference_week_zero_not_a_monday_raises():
    with pytest.raises(ValueError, match="not a Monday"):
        d_run(week_zero=WEEK_ZERO + timedelta(days=1))


def test_reference_price_projection_outside_the_tufe_table_raises():
    with pytest.raises(InflationCoverageError):
        d_run(week_zero=date(2040, 1, 2))   # a Monday far past the TÜFE table


# --- purity ----------------------------------------------------------------------------------

def test_reference_inputs_are_not_modified():
    kw = d_inputs(origins=[30, 20, 20], companies=["2000", "1000"])
    copies = {name: kw[name].copy(deep=True) for name in ("weekly", "weeks", "calendar")}
    levels = kw["levels"].copy(deep=True)
    deflated_reference(**kw)
    for name, frame in copies.items():
        pd.testing.assert_frame_equal(kw[name], frame)
    pd.testing.assert_series_equal(kw["levels"], levels)
    assert kw["origins"] == [30, 20, 20] and kw["companies"] == ["2000", "1000"]


def test_reference_same_inputs_give_the_same_result(d_result):
    pd.testing.assert_frame_equal(d_run(), d_result)


def test_reference_order_of_the_weekly_rows_does_not_matter(d_result):
    weekly = d_weekly().sample(frac=1.0, random_state=11)
    found = d_run(weekly=weekly)
    pd.testing.assert_frame_equal(found, d_result, check_exact=False, rtol=1e-9)


def test_reference_empty_weekly_gives_zero_forecasts_that_fall_back():
    found = d_run(weekly=d_weekly().iloc[0:0], origins=[20], horizons=1)
    assert len(found) == 6
    assert (found["real_forecast"] == 0.0).all() and (found["forecast"] == 0.0).all()
    assert found["fallback"].all()



# =============================================================================================
# U14.4: the financial-rule reference: `invoice_delays`, `rule_invoices`, `rule_totals`,
# `RuleReference` and `rule_reference`.
#
# Hand-made AR ledgers in the shape `open_invoices_at` and `payment_delays_at` read (as in
# test_receivables_at.py). Week zero is test_forecast's Monday 2025-06-02, so the origin T = 20
# ends on Sunday 2025-10-26 (CUT below). [receivables] horizons 5, min_cleared 3 and
# [evaluation] rule_doubtful_days 90 unless a test says otherwise.
#
# The running example at T = 20 (CUT = Sunday 2025-10-26), delays by hand:
#   TR02 customer A: three cleared invoices 4, 6, 8 days late -> its own usual delay 6.
#   TR02 customer C: three cleared invoices 0, 2, 2 days late -> 2 (no open invoice).
#   TR02 customer B: ONE cleared invoice, 10 days late -> too few (NaN) -> TR02's median over
#        its customers WITH a usual delay, median(6, 2) = 4. (An invoice-level median over
#        0, 2, 2, 4, 6, 8 would be 3; counting B's 10 would give median(6, 2, 10) = 6.)
#   TR04 customer E: ONE cleared invoice -> NaN; TR04 has no usual delay at all, so its
#        customer D gets 0 days ("none"). TR02's median does not leak into TR04.
# The open invoices at T (status, h, outstanding TRY):
#   A1  100 TRY due CUT-6   -> expected CUT (the cutoff Sunday)        LATE, h 1
#   A2 1000 TRY due CUT-5, 300 paid before T -> expected CUT+1 (Monday) EXPECTED h 1, 700
#   A3   50 TRY due CUT+29  -> expected CUT+35 = Sunday of week 25      EXPECTED h 5 (= horizons)
#        (a 50 TRY payment for it is posted only after T, in the leak tests)
#   A4   20 TRY due CUT+30  -> expected CUT+36 = Monday of week 26      AFTER_HORIZON, h 6
#   A5   30 TRY due CUT-90  -> 90 days past due (not > 90), expected CUT-84   LATE, h 1
#   A6  999 TRY due CUT-91  -> 91 days past due                          DOUBTFUL
#   B1  100 EUR due CUT+3, rate known at T 36 (35 before, 99 the day after) -> 3600 TRY,
#        expected CUT+7 (Sunday of week 21)                            EXPECTED h 1, company
#   B2  200 TRY due CUT+10, cleared only on CUT+2 (still open at T) -> expected CUT+14
#        (Sunday of week 22)                                            EXPECTED h 2, company
#   D1  400 TRY (TR04) due CUT+8 (Monday of week 22), 0 days           EXPECTED h 2, none
#   D2 5000 TRY (TR04) due CUT-100                                       DOUBTFUL, none
# =============================================================================================

from types import SimpleNamespace  # noqa: E402

from collection_estimation.hurdle.evaluate import (  # noqa: E402
    AFTER_HORIZON, DOUBTFUL, EXPECTED, FROM_COMPANY, FROM_CUSTOMER, FROM_NOTHING, LATE,
    RULE_COLUMNS, RULE_INVOICE_COLUMNS, RuleReference, invoice_delays, rule_invoices,
    rule_reference, rule_totals)
from collection_estimation.hurdle.receivables_asof import (  # noqa: E402
    end_of_week, ledger_as_known_at, open_invoices_at)
from collection_estimation.hurdle.topups import topup_forecasts  # noqa: E402
from tests.hurdle.test_forecast import f_calendar, f_levels  # noqa: E402

RULE_ORIGIN = 20
CUT = end_of_week(WEEK_ZERO, RULE_ORIGIN)          # Sunday 2025-10-26
RULE_HORIZONS = 5
RULE_MIN_CLEARED = 3
RULE_DOUBTFUL = 90
RULE_COMPANIES = ["TR02", "TR04"]
RULE_LEDGER_COLUMNS = ["key", "company_code", "customer_id", "document_no", "document_role",
                       "invoice_reference", "posting_date", "net_due_date", "clearing_date",
                       "clearing_document", "amount_document", "document_currency",
                       "source_row"]


def days(n):
    return timedelta(days=n)


def rl(role, document_no, amount, posted, due, *, customer, company="TR02", reference=None,
       cleared=None, clearing_document=None, currency="TRY"):
    """One ledger line; `amount` as in the ledger (payments negative)."""
    def ts(day):
        return pd.Timestamp(day) if day is not None else pd.NaT
    return {"key": f"{company}-{customer}", "company_code": company, "customer_id": customer,
            "document_no": document_no, "document_role": role, "invoice_reference": reference,
            "posting_date": ts(posted), "net_due_date": ts(due), "clearing_date": ts(cleared),
            "clearing_document": clearing_document, "amount_document": float(amount),
            "document_currency": currency}


def rl_ledger(lines):
    """A ledger table with ledger-like dtypes; source_row counts from 2."""
    if not lines:
        table = pd.DataFrame({c: pd.Series(dtype=object) for c in RULE_LEDGER_COLUMNS})
    else:
        table = pd.DataFrame(lines, columns=RULE_LEDGER_COLUMNS[:-1])
        table["source_row"] = range(2, 2 + len(table))
    for column in ["posting_date", "net_due_date", "clearing_date"]:
        table[column] = pd.to_datetime(table[column])
    table["amount_document"] = table["amount_document"].astype("float64")
    for column in ["invoice_reference", "clearing_document"]:
        table[column] = table[column].astype(object)
    table["source_row"] = table["source_row"].astype("int64")
    return table


def history(customer, delays_late, company="TR02", back=120):
    """Cleared invoices of `customer`, each paid `delays_late[i]` days after its due date,
    every one cleared (on the payment day) well before CUT."""
    lines = []
    for i, late in enumerate(delays_late):
        due = CUT - days(back + 7 * i)
        paid = due + days(late)
        assert paid <= CUT
        document, clearing = f"H-{company}-{customer}-{i}", f"CL-{company}-{customer}-{i}"
        lines.append(rl("invoice", document, 10.0, due - days(30), due, customer=customer,
                        company=company, cleared=paid, clearing_document=clearing))
        lines.append(rl("payment", f"P-{company}-{customer}-{i}", -10.0, paid, paid,
                        customer=customer, company=company, reference=document, cleared=paid,
                        clearing_document=clearing))
    return lines


def rule_rates(extra=()):
    rows = [(CUT - days(10), "EUR", 35.0), (CUT, "EUR", 36.0), (CUT + days(1), "EUR", 99.0)]
    rows += list(extra)
    return pd.DataFrame({"day": pd.Series([r[0] for r in rows], dtype=object),
                         "currency": pd.Series([r[1] for r in rows], dtype=object),
                         "rate": pd.Series([r[2] for r in rows], dtype="float64")})


def running_lines():
    """The running example's lines, as known at and before T (see the header)."""
    return (history("A", [4, 6, 8]) + history("C", [0, 2, 2]) + history("B", [10])
            + history("E", [5], company="TR04") + [
                rl("invoice", "A1", 100, CUT - days(40), CUT - days(6), customer="A"),
                rl("invoice", "A2", 1000, CUT - days(40), CUT - days(5), customer="A"),
                rl("payment", "PA2", -300, CUT - days(1), CUT - days(1), customer="A",
                   reference="A2"),
                rl("invoice", "A3", 50, CUT - days(10), CUT + days(29), customer="A"),
                rl("invoice", "A4", 20, CUT - days(10), CUT + days(30), customer="A"),
                rl("invoice", "A5", 30, CUT - days(120), CUT - days(90), customer="A"),
                rl("invoice", "A6", 999, CUT - days(120), CUT - days(91), customer="A"),
                rl("invoice", "B1", 100, CUT - days(5), CUT + days(3), customer="B",
                   currency="EUR"),
                rl("invoice", "B2", 200, CUT - days(5), CUT + days(10), customer="B",
                   cleared=CUT + days(2), clearing_document="CL-B2"),
                rl("payment", "PB2", -200, CUT + days(2), CUT + days(2), customer="B",
                   reference="B2", cleared=CUT + days(2), clearing_document="CL-B2"),
                rl("invoice", "D1", 400, CUT - days(5), CUT + days(8), customer="D",
                   company="TR04"),
                rl("invoice", "D2", 5000, CUT - days(130), CUT - days(100), customer="D",
                   company="TR04"),
            ])


def running_ledger():
    return rl_ledger(running_lines())


def later_lines():
    """Lines posted after T, and clearings made after T: none may change the rows at T."""
    return [
        # A1 paid and cleared the day after T (its own line gets a clearing date in run_full).
        rl("payment", "PA1", -100, CUT + days(1), CUT + days(1), customer="A", reference="A1",
           cleared=CUT + days(1), clearing_document="CL-A1"),
        # A payment for A3 posted after T: A3 stays 50 outstanding at T.
        rl("payment", "PA3", -50, CUT + days(3), CUT + days(3), customer="A", reference="A3"),
        # A new invoice posted the Monday after T.
        rl("invoice", "N1", 777, CUT + days(1), CUT + days(5), customer="A"),
        # D's first cleared invoices, all after T: D still has no usual delay at T.
        *[line for i in range(3) for line in (
            rl("invoice", f"HD{i}", 10, CUT + days(1), CUT + days(2), customer="D",
               company="TR04", cleared=CUT + days(3 + i), clearing_document=f"CL-D{i}"),
            rl("payment", f"PD{i}", -10, CUT + days(3 + i), CUT + days(3 + i), customer="D",
               company="TR04", reference=f"HD{i}", cleared=CUT + days(3 + i),
               clearing_document=f"CL-D{i}"))],
    ]


def full_ledger():
    """The running ledger as extracted later: the later lines, and A1 cleared after T."""
    lines = running_lines()
    for line in lines:
        if line["document_no"] == "A1":
            line["clearing_date"] = pd.Timestamp(CUT + days(1))
            line["clearing_document"] = "CL-A1"
    return rl_ledger(lines + later_lines())


def run_rule_invoices(ledger=None, rates=None, origin=RULE_ORIGIN, horizons=RULE_HORIZONS,
                      min_cleared=RULE_MIN_CLEARED, doubtful_days=RULE_DOUBTFUL):
    return rule_invoices(running_ledger() if ledger is None else ledger,
                         rule_rates() if rates is None else rates, WEEK_ZERO, origin, horizons,
                         min_cleared, doubtful_days)


def by_document(invoices):
    assert not invoices["document_no"].duplicated().any()
    return invoices.set_index("document_no")


@pytest.fixture(scope="module")
def running():
    return run_rule_invoices()


# (status, h or None if not asserted, outstanding_try, delay_days, delay_from, days_past_due,
#  expected_date)
RUNNING = {
    "A1": (LATE, 1, 100.0, 6, FROM_CUSTOMER, 6, CUT),
    "A2": (EXPECTED, 1, 700.0, 6, FROM_CUSTOMER, 5, CUT + days(1)),
    "A3": (EXPECTED, 5, 50.0, 6, FROM_CUSTOMER, -29, CUT + days(35)),
    "A4": (AFTER_HORIZON, 6, 20.0, 6, FROM_CUSTOMER, -30, CUT + days(36)),
    "A5": (LATE, 1, 30.0, 6, FROM_CUSTOMER, 90, CUT - days(84)),
    "A6": (DOUBTFUL, None, 999.0, 6, FROM_CUSTOMER, 91, CUT - days(85)),
    "B1": (EXPECTED, 1, 3600.0, 4, FROM_COMPANY, -3, CUT + days(7)),
    "B2": (EXPECTED, 2, 200.0, 4, FROM_COMPANY, -10, CUT + days(14)),
    "D1": (EXPECTED, 2, 400.0, 0, FROM_NOTHING, -8, CUT + days(8)),
    "D2": (DOUBTFUL, None, 5000.0, 0, FROM_NOTHING, 100, CUT - days(100)),
}


# --- constants -----------------------------------------------------------------------------

def test_rule_status_constants():
    assert (DOUBTFUL, LATE, EXPECTED, AFTER_HORIZON) == ("doubtful", "late", "expected",
                                                         "after horizon")
    assert len({DOUBTFUL, LATE, EXPECTED, AFTER_HORIZON}) == 4


def test_rule_delay_source_constants():
    assert (FROM_CUSTOMER, FROM_COMPANY, FROM_NOTHING) == ("customer", "company", "none")


def test_rule_invoice_columns():
    assert RULE_INVOICE_COLUMNS == ["key", "company_code", "document_no", "origin_week",
                                    "net_due_date", "outstanding_try", "days_past_due",
                                    "delay_days", "delay_from", "expected_date", "status", "h"]


def test_rule_columns():
    assert RULE_COLUMNS == ["company_code", "origin_week", "h", "target_week", "invoice_cash",
                            "late_cash", "fallback_delay_cash", "topup_cash", "price_level",
                            "forecast"]


def test_rule_premise_the_cutoff_is_the_sunday_ending_week_20():
    assert CUT == date(2025, 10, 26)
    assert CUT.weekday() == 6


# --- rule_invoices: the running example by hand ----------------------------------------------

def test_rule_invoices_has_exactly_the_columns(running):
    assert list(running.columns) == RULE_INVOICE_COLUMNS


def test_rule_invoices_one_row_per_open_invoice(running):
    # History invoices are cleared before T; B2 (cleared after T) is still open.
    assert sorted(running["document_no"]) == sorted(RUNNING)


@pytest.mark.parametrize("document", sorted(RUNNING))
def test_rule_invoices_each_invoice_by_hand(running, document):
    status, h, outstanding, delay, source, past_due, expected = RUNNING[document]
    found = by_document(running).loc[document]
    assert found["status"] == status
    if h is not None:
        assert found["h"] == h
    assert found["outstanding_try"] == pytest.approx(outstanding, rel=1e-12)
    assert found["delay_days"] == delay
    assert found["delay_from"] == source
    assert found["days_past_due"] == past_due
    assert found["expected_date"] == pd.Timestamp(expected)
    assert found["origin_week"] == RULE_ORIGIN


def test_rule_invoices_keys_and_companies(running):
    table = by_document(running)
    for document in RUNNING:
        company = "TR04" if document.startswith("D") else "TR02"
        assert table.loc[document, "company_code"] == company
        assert table.loc[document, "key"] == f"{company}-{document[0]}"


def test_rule_invoices_expected_date_is_due_date_plus_delay(running):
    assert (running["expected_date"]
            == running["net_due_date"] + pd.to_timedelta(running["delay_days"], unit="D")).all()


def test_rule_invoices_days_past_due_is_cutoff_minus_due_date(running):
    assert ((pd.Timestamp(CUT) - running["net_due_date"]).dt.days
            == running["days_past_due"]).all()


def test_rule_invoices_h_of_counted_rows_is_the_week_of_the_expected_date(running):
    not_late = running[running["status"].isin([EXPECTED, AFTER_HORIZON])]
    weeks = week_numbers_of(not_late["expected_date"])
    assert (not_late["h"] == weeks - RULE_ORIGIN).all()


def week_numbers_of(dates):
    from collection_estimation.hurdle.series import week_numbers
    return week_numbers(dates, WEEK_ZERO)


def test_rule_invoices_every_late_row_has_h_1(running):
    late = running[running["status"] == LATE]
    assert len(late) == 2
    assert (late["h"] == 1).all()


def test_rule_invoices_expected_rows_lie_in_the_horizon(running):
    expected = running[running["status"] == EXPECTED]
    assert expected["h"].between(1, RULE_HORIZONS).all()
    assert (expected["expected_date"] > pd.Timestamp(CUT)).all()


def test_rule_invoices_after_horizon_rows_lie_after_it(running):
    after = running[running["status"] == AFTER_HORIZON]
    assert (after["h"] > RULE_HORIZONS).all()


def test_rule_invoices_statuses_are_only_the_four(running):
    assert set(running["status"]) <= {DOUBTFUL, LATE, EXPECTED, AFTER_HORIZON}
    assert set(running["delay_from"]) <= {FROM_CUSTOMER, FROM_COMPANY, FROM_NOTHING}


def test_rule_invoices_dtypes(running):
    assert running["origin_week"].dtype == "int64"
    assert running["h"].dtype == "int64"
    assert running["days_past_due"].dtype == "int64"
    assert running["delay_days"].dtype == "int64"
    assert running["outstanding_try"].dtype == "float64"
    # The unit (s or ns) is the ledger's; both dates are datetimes of the same unit.
    assert pd.api.types.is_datetime64_dtype(running["net_due_date"])
    assert running["expected_date"].dtype == running["net_due_date"].dtype
    # Text columns: object or pandas' str dtype (the ledger's; pandas 3 infers str).
    for column in ("key", "company_code", "document_no", "delay_from", "status"):
        assert (pd.api.types.is_object_dtype(running[column])
                or pd.api.types.is_string_dtype(running[column])), column


# --- rule_invoices: the boundaries -------------------------------------------------------------

def test_expected_date_on_the_cutoff_sunday_is_late_and_monday_after_is_h_1_expected(running):
    table = by_document(running)
    assert table.loc["A1", "expected_date"] == pd.Timestamp(CUT)
    assert table.loc["A1", "status"] == LATE
    assert table.loc["A2", "expected_date"] == pd.Timestamp(CUT + days(1))
    assert table.loc["A2", "status"] == EXPECTED and table.loc["A2", "h"] == 1


def test_h_equal_to_horizons_is_expected_and_one_more_is_after_horizon(running):
    table = by_document(running)
    assert (table.loc["A3", "status"], table.loc["A3", "h"]) == (EXPECTED, RULE_HORIZONS)
    assert (table.loc["A4", "status"], table.loc["A4", "h"]) == (AFTER_HORIZON,
                                                                 RULE_HORIZONS + 1)


def test_more_horizons_makes_the_after_horizon_invoice_expected():
    table = by_document(run_rule_invoices(horizons=6))
    assert (table.loc["A4", "status"], table.loc["A4", "h"]) == (EXPECTED, 6)


def test_fewer_horizons_makes_the_last_week_invoice_after_horizon():
    table = by_document(run_rule_invoices(horizons=4))
    assert (table.loc["A3", "status"], table.loc["A3", "h"]) == (AFTER_HORIZON, 5)
    assert table.loc["A2", "status"] == EXPECTED   # h 1 unchanged


def test_days_past_due_equal_to_doubtful_days_is_not_doubtful_one_more_is(running):
    table = by_document(running)
    assert (table.loc["A5", "days_past_due"], table.loc["A5", "status"]) == (90, LATE)
    assert (table.loc["A6", "days_past_due"], table.loc["A6", "status"]) == (91, DOUBTFUL)


@pytest.mark.parametrize("doubtful_days, a5, a6", [(89, DOUBTFUL, DOUBTFUL),
                                                  (90, LATE, DOUBTFUL),
                                                  (91, LATE, LATE)])
def test_the_doubtful_threshold_moves_with_doubtful_days(doubtful_days, a5, a6):
    table = by_document(run_rule_invoices(doubtful_days=doubtful_days))
    assert (table.loc["A5", "status"], table.loc["A6", "status"]) == (a5, a6)


def test_doubtful_beats_late(running):
    # A6 and D2 are past their expected date too; doubtful wins.
    table = by_document(running)
    for document in ("A6", "D2"):
        assert table.loc[document, "expected_date"] <= pd.Timestamp(CUT)
        assert table.loc[document, "status"] == DOUBTFUL


def test_doubtful_days_is_counted_on_the_due_date_not_the_expected_date():
    # Customer F pays 200 days late: F1 is 91 days past due but expected 109 days after T;
    # it is still doubtful (priority over AFTER_HORIZON). F2 is 50 days past due, expected
    # 150 days after T: AFTER_HORIZON, though overdue.
    ledger = rl_ledger(history("F", [200, 200, 200], back=320) + [
        rl("invoice", "F1", 10, CUT - days(120), CUT - days(91), customer="F"),
        rl("invoice", "F2", 20, CUT - days(80), CUT - days(50), customer="F")])
    table = by_document(run_rule_invoices(ledger))
    assert table.loc["F1", "delay_days"] == 200
    assert table.loc["F1", "expected_date"] == pd.Timestamp(CUT + days(109))
    assert table.loc["F1", "status"] == DOUBTFUL
    assert table.loc["F2", "expected_date"] == pd.Timestamp(CUT + days(150))
    assert table.loc["F2", "status"] == AFTER_HORIZON
    assert table.loc["F2", "h"] == week_numbers_of(pd.Series([pd.Timestamp(CUT + days(150))]))[0] - 20


def test_a_negative_usual_delay_moves_the_expected_date_before_the_due_date():
    # G pays 3 days early: an invoice due Tuesday CUT + 2 is expected Saturday CUT - 1, LATE.
    ledger = rl_ledger(history("G", [-3, -3, -3]) + [
        rl("invoice", "G1", 10, CUT - days(10), CUT + days(2), customer="G"),
        rl("invoice", "G2", 10, CUT - days(10), CUT + days(4), customer="G")])
    table = by_document(run_rule_invoices(ledger))
    assert table.loc["G1", "delay_days"] == -3
    assert table.loc["G1", "expected_date"] == pd.Timestamp(CUT - days(1))
    assert (table.loc["G1", "status"], table.loc["G1", "h"]) == (LATE, 1)
    # Due Thursday CUT + 4, expected Monday CUT + 1: h 1, expected.
    assert (table.loc["G2", "status"], table.loc["G2", "h"]) == (EXPECTED, 1)
    assert table.loc["G1", "days_past_due"] == -2


# --- the three delay sources -------------------------------------------------------------------

def test_delay_from_the_customer_is_its_median(running):
    assert by_document(running).loc["A1", "delay_days"] == 6


def test_delay_from_the_company_is_the_median_over_its_customers_with_a_usual_delay(running):
    # median(6, 2) = 4: not B's own single 10, not the invoice-level median 3, not 6.
    table = by_document(running)
    assert table.loc["B1", "delay_from"] == FROM_COMPANY
    assert table.loc["B1", "delay_days"] == 4


def test_delay_from_nothing_is_zero_and_the_other_companys_median_does_not_leak(running):
    table = by_document(running)
    assert table.loc["D1", "delay_from"] == FROM_NOTHING
    assert table.loc["D1", "delay_days"] == 0
    assert table.loc["D1", "expected_date"] == table.loc["D1", "net_due_date"]


def test_min_cleared_1_makes_every_customer_with_one_cleared_invoice_its_own_source():
    table = by_document(run_rule_invoices(min_cleared=1))
    assert (table.loc["B1", "delay_from"], table.loc["B1", "delay_days"]) == (FROM_CUSTOMER, 10)
    # E's own invoice gives TR04 a usual delay (5); D still has none of its own.
    assert (table.loc["D1", "delay_from"], table.loc["D1", "delay_days"]) == (FROM_COMPANY, 5)


def test_min_cleared_above_everyones_count_falls_back_to_none_everywhere():
    table = run_rule_invoices(min_cleared=4)
    assert set(table["delay_from"]) == {FROM_NOTHING}
    assert (table["delay_days"] == 0).all()


def test_a_customers_half_day_median_is_rounded_half_to_even():
    # A half goes to the even number, as invoice_delays' docstring states (approved by Ufuk,
    # 2026-11-03): 2.5 -> 2, 3.5 -> 4, -2.5 -> -2.
    ledger = rl_ledger(history("X", [2, 3]) + history("Y", [3, 4]) + history("W", [-3, -2])
                       + [rl("invoice", f"{c}1", 10, CUT - days(5), CUT + days(20), customer=c)
                          for c in ("X", "Y", "W")])
    table = by_document(run_rule_invoices(ledger, min_cleared=2))
    assert table.loc["X1", "delay_days"] == 2
    assert table.loc["Y1", "delay_days"] == 4
    assert table.loc["W1", "delay_days"] == -2
    assert set(table["delay_from"]) == {FROM_CUSTOMER}


def test_a_companys_half_day_median_is_rounded_half_to_even():
    # TR04: customers P (1) and Q (2) -> 1.5 -> 2. TR05: customers S (2) and U (3) -> 2.5 -> 2.
    ledger = rl_ledger(history("P", [1], company="TR04") + history("Q", [2], company="TR04")
                       + history("S", [2], company="TR05") + history("U", [3], company="TR05")
                       + [rl("invoice", "R4", 10, CUT - days(5), CUT + days(20), customer="R",
                             company="TR04"),
                          rl("invoice", "R5", 10, CUT - days(5), CUT + days(20), customer="R",
                             company="TR05")])
    table = by_document(run_rule_invoices(ledger, min_cleared=1))
    assert (table.loc["R4", "delay_from"], table.loc["R4", "delay_days"]) == (FROM_COMPANY, 2)
    assert (table.loc["R5", "delay_from"], table.loc["R5", "delay_days"]) == (FROM_COMPANY, 2)


def test_invoice_delays_directly_adds_the_two_columns_and_keeps_the_rest():
    ledger = running_ledger()
    invoices = open_invoices_at(ledger, CUT)
    before = invoices.copy()
    found = invoice_delays(invoices, ledger, CUT, RULE_MIN_CLEARED)
    pd.testing.assert_frame_equal(invoices, before)               # not changed in place
    assert list(found.columns) == list(invoices.columns) + ["delay_from", "delay_days"]
    pd.testing.assert_frame_equal(found[list(invoices.columns)], invoices)
    assert found["delay_days"].dtype == "int64"
    table = found.set_index("document_no")
    assert table.loc["A1", "delay_days"] == 6 and table.loc["B1", "delay_days"] == 4
    assert table.loc["D1", "delay_days"] == 0


def test_invoice_delays_on_no_invoices_is_empty():
    ledger = running_ledger()
    invoices = open_invoices_at(ledger, CUT).iloc[0:0]
    found = invoice_delays(invoices, ledger, CUT, RULE_MIN_CLEARED)
    assert found.empty
    assert {"delay_from", "delay_days"} <= set(found.columns)


# --- amounts: partial payments, foreign currency, cleared after T -----------------------------

def test_a_partial_payment_before_t_reduces_the_outstanding(running):
    assert by_document(running).loc["A2", "outstanding_try"] == pytest.approx(700.0)


def test_two_partial_payments_both_count():
    ledger = rl_ledger(running_lines() + [
        rl("payment", "PA2b", -150, CUT, CUT, customer="A", reference="A2")])   # on T itself
    assert by_document(run_rule_invoices(ledger)).loc["A2", "outstanding_try"] == pytest.approx(
        550.0)


def test_a_foreign_currency_invoice_uses_the_rate_known_at_t(running):
    # 100 EUR x 36 (the rate on the cutoff day itself), not 35 (before) or 99 (after).
    assert by_document(running).loc["B1", "outstanding_try"] == pytest.approx(3600.0)


def test_a_foreign_currency_invoice_at_an_older_rate_when_none_on_t():
    rates = rule_rates()
    rates = rates[rates["day"] != CUT]
    assert by_document(run_rule_invoices(rates=rates)).loc["B1", "outstanding_try"] == (
        pytest.approx(3500.0))


def test_a_lower_case_currency_is_converted_too():
    lines = running_lines()
    for line in lines:
        if line["document_no"] == "B1":
            line["document_currency"] = " eur "
    table = by_document(run_rule_invoices(rl_ledger(lines)))
    assert table.loc["B1", "outstanding_try"] == pytest.approx(3600.0)


def test_an_invoice_cleared_after_t_is_still_open_at_t(running):
    table = by_document(running)
    assert "B2" in table.index
    assert table.loc["B2", "outstanding_try"] == pytest.approx(200.0)   # paid only after T


def test_an_invoice_cleared_on_t_itself_is_not_open():
    lines = running_lines()
    for line in lines:
        if line["document_no"] == "B2":
            line["clearing_date"] = pd.Timestamp(CUT)
    assert "B2" not in set(run_rule_invoices(rl_ledger(lines))["document_no"])


def test_an_invoice_posted_on_t_is_in_and_one_posted_the_day_after_is_not():
    ledger = rl_ledger(running_lines() + [
        rl("invoice", "ON", 5, CUT, CUT + days(20), customer="A"),
        rl("invoice", "AFTER", 5, CUT + days(1), CUT + days(20), customer="A")])
    found = set(run_rule_invoices(ledger)["document_no"])
    assert "ON" in found and "AFTER" not in found


def test_a_fully_paid_open_invoice_is_left_out():
    ledger = rl_ledger(running_lines() + [
        rl("payment", "PA4", -20, CUT - days(1), CUT - days(1), customer="A", reference="A4")])
    assert "A4" not in set(run_rule_invoices(ledger)["document_no"])


# --- no leak ---------------------------------------------------------------------------------

def test_no_leak_lines_posted_or_cleared_after_t_do_not_change_the_rows_at_t(running):
    pd.testing.assert_frame_equal(run_rule_invoices(full_ledger()), running)


def test_no_leak_premise_the_later_lines_do_change_a_later_origin():
    # Teeth: at T + 1 the later lines are known, and the rows differ.
    later = run_rule_invoices(full_ledger(), origin=RULE_ORIGIN + 1)
    without = run_rule_invoices(running_ledger(), origin=RULE_ORIGIN + 1)
    assert not later.reset_index(drop=True).equals(without.reset_index(drop=True))
    assert "N1" in set(later["document_no"])
    assert "A1" not in set(later["document_no"])


def test_no_leak_equals_the_ledger_cut_at_t_and_the_rates_cut_at_t(running):
    rates = rule_rates()
    cut = run_rule_invoices(ledger_as_known_at(full_ledger(), CUT),
                            rates[rates["day"] <= CUT])
    pd.testing.assert_frame_equal(cut, running)


def test_no_leak_a_rate_after_t_is_never_used():
    rates = rule_rates(extra=[(CUT + days(1), "USD", 30.0)])
    lines = running_lines() + [rl("invoice", "U1", 10, CUT - days(5), CUT + days(20),
                                  customer="A", currency="USD")]
    with pytest.raises(ValueError) as error:
        run_rule_invoices(rl_ledger(lines), rates)
    assert str(error.value) == f"no exchange rate known on or before {CUT} for ['USD']"


def test_no_leak_customer_delays_cleared_after_t_are_not_used(running):
    # D's three cleared invoices after T would give it its own delay; at T it has none.
    found = by_document(run_rule_invoices(full_ledger()))
    assert (found.loc["D1", "delay_from"], found.loc["D1", "delay_days"]) == (FROM_NOTHING, 0)
    later = by_document(run_rule_invoices(full_ledger(), origin=RULE_ORIGIN + 1))
    assert later.loc["D1", "delay_from"] == FROM_CUSTOMER


# --- rule_invoices: refusals and empty -------------------------------------------------------

def test_an_open_invoice_without_a_due_date_raises_with_the_exact_message():
    ledger = rl_ledger(running_lines() + [
        rl("invoice", "NODUE", 10, CUT - days(5), None, customer="A")])
    with pytest.raises(ValueError) as error:
        run_rule_invoices(ledger)
    assert str(error.value) == (f"at {CUT}, 1 open invoices have no net due date "
                                f"(first: ['NODUE'])")


def test_an_invoice_without_a_due_date_posted_after_t_does_not_raise(running):
    ledger = rl_ledger(running_lines() + [
        rl("invoice", "NODUE", 10, CUT + days(1), None, customer="A")])
    pd.testing.assert_frame_equal(run_rule_invoices(ledger), running)


def test_a_currency_with_no_rate_at_all_raises():
    ledger = rl_ledger(running_lines() + [
        rl("invoice", "G1", 10, CUT - days(5), CUT + days(20), customer="A", currency="GBP")])
    with pytest.raises(ValueError) as error:
        run_rule_invoices(ledger)
    assert str(error.value) == f"no exchange rate known on or before {CUT} for ['GBP']"


def test_no_line_known_at_t_gives_an_empty_table_with_the_columns():
    found = run_rule_invoices(origin=-30)     # before every line of the running ledger
    assert found.empty
    assert list(found.columns) == RULE_INVOICE_COLUMNS


def test_an_empty_ledger_gives_an_empty_table_with_the_columns():
    found = run_rule_invoices(rl_ledger([]))
    assert found.empty
    assert list(found.columns) == RULE_INVOICE_COLUMNS


def test_rule_invoices_does_not_change_its_inputs():
    ledger, rates = running_ledger(), rule_rates()
    ledger_before, rates_before = ledger.copy(), rates.copy()
    run_rule_invoices(ledger, rates)
    pd.testing.assert_frame_equal(ledger, ledger_before)
    pd.testing.assert_frame_equal(rates, rates_before)


def test_rule_invoices_is_deterministic(running):
    pd.testing.assert_frame_equal(run_rule_invoices(), running)


def test_rule_invoices_does_not_depend_on_the_ledger_row_order(running):
    shuffled = running_ledger().sample(frac=1.0, random_state=7).reset_index(drop=True)
    found = run_rule_invoices(shuffled)
    key = ["document_no"]
    pd.testing.assert_frame_equal(found.sort_values(key, ignore_index=True),
                                  running.sort_values(key, ignore_index=True))


# --- rule_totals ------------------------------------------------------------------------------

TOTALS_COMPANIES = ["TR02", "TR04", "TR05"]     # TR05 has no invoice at all
TOPUP_OFFSET = {"TR02": 0.0, "TR04": 1000.0, "TR05": 2000.0}


def level_at(week):
    return 1.0 + (week - RULE_ORIGIN) / 10.0      # 1.1 at T + 1 ... 1.5 at T + 5


def rule_topups(companies=TOTALS_COMPANIES, horizons=RULE_HORIZONS):
    """topup = 10 h + an offset per company, deflated."""
    return pd.DataFrame([{"company_code": c, "origin_week": RULE_ORIGIN, "h": h,
                          "target_week": RULE_ORIGIN + h, "topup": 10.0 * h + TOPUP_OFFSET[c]}
                         for c in companies for h in range(1, horizons + 1)])


def rule_projected(horizons=RULE_HORIZONS):
    weeks = [RULE_ORIGIN + h for h in range(1, horizons + 1)]
    return pd.Series([level_at(w) for w in weeks], index=weeks)


# (company, h) -> (invoice_cash, late_cash, fallback_delay_cash), by hand from the header.
RUNNING_TOTALS = {
    ("TR02", 1): (100.0 + 700.0 + 30.0 + 3600.0, 100.0 + 30.0, 3600.0),
    ("TR02", 2): (200.0, 0.0, 200.0),
    ("TR02", 5): (50.0, 0.0, 0.0),
    ("TR04", 2): (400.0, 0.0, 400.0),
}


def expected_part(company, h):
    return RUNNING_TOTALS.get((company, h), (0.0, 0.0, 0.0))


@pytest.fixture(scope="module")
def running_totals(running):
    return rule_totals(running, rule_topups(), rule_projected(), TOTALS_COMPANIES, RULE_ORIGIN,
                       RULE_HORIZONS)


def total_row(totals, company, h, origin=RULE_ORIGIN):
    found = totals[(totals["company_code"] == company) & (totals["h"] == h)
                   & (totals["origin_week"] == origin)]
    assert len(found) == 1, (company, h, origin, len(found))
    return found.iloc[0]


def test_rule_totals_has_exactly_the_columns(running_totals):
    assert list(running_totals.columns) == RULE_COLUMNS


def test_rule_totals_row_count_and_order(running_totals):
    assert list(zip(running_totals["company_code"], running_totals["h"])) == [
        (c, h) for c in TOTALS_COMPANIES + [CONSOLIDATED] for h in range(1, RULE_HORIZONS + 1)]


@pytest.mark.parametrize("company", TOTALS_COMPANIES)
@pytest.mark.parametrize("h", range(1, RULE_HORIZONS + 1))
def test_rule_totals_each_company_and_h_by_hand(running_totals, company, h):
    found = total_row(running_totals, company, h)
    invoice, late, fallback = expected_part(company, h)
    topup = (10.0 * h + TOPUP_OFFSET[company]) * level_at(RULE_ORIGIN + h)
    assert found["target_week"] == RULE_ORIGIN + h
    assert found["invoice_cash"] == pytest.approx(invoice, rel=1e-12)
    assert found["late_cash"] == pytest.approx(late, rel=1e-12)
    assert found["fallback_delay_cash"] == pytest.approx(fallback, rel=1e-12)
    assert found["price_level"] == pytest.approx(level_at(RULE_ORIGIN + h), rel=1e-15)
    assert found["topup_cash"] == pytest.approx(topup, rel=1e-12)
    assert found["forecast"] == pytest.approx(invoice + topup, rel=1e-12)


@pytest.mark.parametrize("h", range(1, RULE_HORIZONS + 1))
def test_rule_totals_all_rows_by_hand(running_totals, h):
    found = total_row(running_totals, CONSOLIDATED, h)
    parts = [expected_part(c, h) for c in TOTALS_COMPANIES]
    topup = sum(10.0 * h + TOPUP_OFFSET[c] for c in TOTALS_COMPANIES) * level_at(RULE_ORIGIN + h)
    invoice = sum(p[0] for p in parts)
    assert found["invoice_cash"] == pytest.approx(invoice, rel=1e-12)
    assert found["late_cash"] == pytest.approx(sum(p[1] for p in parts), rel=1e-12)
    assert found["fallback_delay_cash"] == pytest.approx(sum(p[2] for p in parts), rel=1e-12)
    assert found["topup_cash"] == pytest.approx(topup, rel=1e-12)
    assert found["forecast"] == pytest.approx(invoice + topup, rel=1e-12)
    assert found["price_level"] == pytest.approx(level_at(RULE_ORIGIN + h), rel=1e-15)
    assert found["target_week"] == RULE_ORIGIN + h


def test_rule_totals_h1_all_by_hand_numbers(running_totals):
    # 4430 (TR02) + 0 + 0; top-ups (10 + 1010 + 2010) x 1.1 = 3333; forecast 7763.
    found = total_row(running_totals, CONSOLIDATED, 1)
    assert found["invoice_cash"] == pytest.approx(4430.0)
    assert found["late_cash"] == pytest.approx(130.0)
    assert found["fallback_delay_cash"] == pytest.approx(3600.0)
    assert found["topup_cash"] == pytest.approx(3333.0)
    assert found["forecast"] == pytest.approx(7763.0)


def test_rule_totals_all_is_the_sum_of_the_companies_for_every_amount(running_totals):
    amounts = ["invoice_cash", "late_cash", "fallback_delay_cash", "topup_cash", "forecast"]
    companies = running_totals[running_totals["company_code"] != CONSOLIDATED]
    consolidated = running_totals[running_totals["company_code"] == CONSOLIDATED]
    summed = companies.groupby("h")[amounts].sum()
    np.testing.assert_allclose(consolidated.set_index("h")[amounts].to_numpy(),
                               summed.to_numpy(), rtol=1e-14)


def test_rule_totals_a_company_with_no_invoice_gets_zero_invoice_rows(running_totals):
    rows = running_totals[running_totals["company_code"] == "TR05"]
    assert len(rows) == RULE_HORIZONS
    for column in ("invoice_cash", "late_cash", "fallback_delay_cash"):
        assert (rows[column] == 0.0).all(), column
    assert (rows["forecast"] == rows["topup_cash"]).all()


def test_rule_totals_doubtful_and_after_horizon_invoices_count_nowhere(running_totals):
    # 999 + 5000 doubtful and 20 after the horizon: the invoice cash over everything is the
    # outstanding of the EXPECTED and LATE rows only.
    all_rows = running_totals[running_totals["company_code"] == CONSOLIDATED]
    assert all_rows["invoice_cash"].sum() == pytest.approx(4430.0 + 200.0 + 50.0 + 400.0)


def test_rule_totals_late_cash_only_in_h_1(running_totals):
    assert (running_totals.loc[running_totals["h"] > 1, "late_cash"] == 0.0).all()


def test_rule_totals_parts_never_exceed_the_invoice_cash(running_totals):
    assert (running_totals["late_cash"] <= running_totals["invoice_cash"] + 1e-9).all()
    assert (running_totals["fallback_delay_cash"]
            <= running_totals["invoice_cash"] + 1e-9).all()


def test_rule_totals_dtypes(running_totals):
    assert running_totals["company_code"].dtype == object
    for column in ("origin_week", "h", "target_week"):
        assert running_totals[column].dtype == "int64", column
    for column in ("invoice_cash", "late_cash", "fallback_delay_cash", "topup_cash",
                   "price_level", "forecast"):
        assert running_totals[column].dtype == "float64", column


def test_rule_totals_index_is_a_fresh_range(running_totals):
    assert list(running_totals.index) == list(range(len(running_totals)))


def test_rule_totals_on_no_invoices_are_the_top_ups_alone(running):
    found = rule_totals(running.iloc[0:0], rule_topups(), rule_projected(), TOTALS_COMPANIES,
                        RULE_ORIGIN, RULE_HORIZONS)
    assert len(found) == (len(TOTALS_COMPANIES) + 1) * RULE_HORIZONS
    assert (found["invoice_cash"] == 0.0).all()
    np.testing.assert_allclose(found["forecast"], found["topup_cash"], rtol=0)


def test_rule_totals_on_an_empty_rule_invoices_result(running):
    empty = run_rule_invoices(origin=-30)
    found = rule_totals(empty, rule_topups(), rule_projected(), TOTALS_COMPANIES, RULE_ORIGIN,
                        RULE_HORIZONS)
    assert (found["invoice_cash"] == 0.0).all()
    assert list(found.columns) == RULE_COLUMNS


def test_rule_totals_unsorted_and_repeated_companies_collapse(running, running_totals):
    found = rule_totals(running, rule_topups(), rule_projected(), ["TR05", "TR02", "TR04", "TR02"],
                        RULE_ORIGIN, RULE_HORIZONS)
    pd.testing.assert_frame_equal(found, running_totals)


def test_rule_totals_a_negative_top_up_lowers_the_forecast(running):
    topups = rule_topups()
    topups["topup"] = -topups["topup"]
    found = rule_totals(running, topups, rule_projected(), TOTALS_COMPANIES, RULE_ORIGIN,
                        RULE_HORIZONS)
    row = total_row(found, "TR02", 1)
    assert row["forecast"] == pytest.approx(4430.0 - 10.0 * 1.1)


def test_rule_totals_an_invoice_of_a_company_outside_the_list_raises(running):
    with pytest.raises(ValueError) as error:
        rule_totals(running, rule_topups(), rule_projected(), ["TR02"], RULE_ORIGIN,
                    RULE_HORIZONS)
    assert str(error.value) == ("the open invoices at origin week 20 have companies not in the "
                                "list: ['TR04']")


def test_rule_totals_a_doubtful_invoice_outside_the_list_still_raises(running):
    # Only TR04's doubtful D2 is left: it counts in no total, but its company is still checked.
    only_d2 = running[running["document_no"].isin(["A1", "D2"])]
    with pytest.raises(ValueError, match=r"not in the list: \['TR04'\]"):
        rule_totals(only_d2, rule_topups(["TR02"]), rule_projected(), ["TR02"], RULE_ORIGIN,
                    RULE_HORIZONS)


def test_rule_totals_a_missing_top_up_row_raises_with_the_exact_message(running):
    topups = rule_topups()
    topups = topups[~((topups["company_code"] == "TR04") & (topups["h"] == 3))]
    with pytest.raises(ValueError) as error:
        rule_totals(running, topups, rule_projected(), TOTALS_COMPANIES, RULE_ORIGIN,
                    RULE_HORIZONS)
    assert str(error.value) == ("the top-ups have 0 rows for company 'TR04', h 3 at origin "
                                "week 20; expected 1")


def test_rule_totals_a_repeated_top_up_row_raises_with_the_exact_message(running):
    topups = rule_topups()
    topups = pd.concat([topups, topups[(topups["company_code"] == "TR02")
                                       & (topups["h"] == 5)]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        rule_totals(running, topups, rule_projected(), TOTALS_COMPANIES, RULE_ORIGIN,
                    RULE_HORIZONS)
    assert str(error.value) == ("the top-ups have 2 rows for company 'TR02', h 5 at origin "
                                "week 20; expected 1")


def test_rule_totals_top_ups_of_a_company_with_no_invoices_must_be_there(running):
    with pytest.raises(ValueError, match="0 rows for company 'TR05', h 1"):
        rule_totals(running, rule_topups(["TR02", "TR04"]), rule_projected(), TOTALS_COMPANIES,
                    RULE_ORIGIN, RULE_HORIZONS)


def test_rule_totals_does_not_change_its_inputs(running):
    invoices, topups, projected = running.copy(), rule_topups(), rule_projected()
    before = (invoices.copy(), topups.copy(), projected.copy())
    rule_totals(invoices, topups, projected, TOTALS_COMPANIES, RULE_ORIGIN, RULE_HORIZONS)
    pd.testing.assert_frame_equal(invoices, before[0])
    pd.testing.assert_frame_equal(topups, before[1])
    pd.testing.assert_series_equal(projected, before[2])


# --- RuleReference ----------------------------------------------------------------------------

def test_rule_reference_dataclass_fields_and_frozen():
    import dataclasses
    assert [f.name for f in dataclasses.fields(RuleReference)] == ["totals", "invoices"]
    made = RuleReference(totals=pd.DataFrame(), invoices=pd.DataFrame())
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.totals = None


# --- rule_reference over many origins -----------------------------------------------------------
#
# The running ledger at origins 18, 20 and 22, with test_forecast's side tables (companies TR02
# and TR04), price levels and calendar, the mean top-up over 8 weeks.

RR_ORIGINS = [18, 20, 22]


def rr_settings(horizons=RULE_HORIZONS, min_cleared=RULE_MIN_CLEARED,
                doubtful_days=RULE_DOUBTFUL):
    settings = f_settings(horizons=horizons)
    receivables = SimpleNamespace(horizons=horizons, min_cleared_for_delay=min_cleared)
    return SimpleNamespace(receivables=receivables, topups=settings.topups,
                           evaluation=SimpleNamespace(rule_doubtful_days=doubtful_days))


def rr_inputs(**overrides):
    new_cash, refunds, unidentified = f_side_tables(companies=RULE_COMPANIES)
    kwargs = dict(companies=list(RULE_COMPANIES), new_cash=new_cash, refunds=refunds,
                  unidentified=unidentified, levels=f_levels(), calendar=f_calendar(),
                  week_zero=WEEK_ZERO)
    kwargs.update(overrides)
    return kwargs


def rr_run(ledger=None, rates=None, origins=RR_ORIGINS, settings=None, **overrides):
    return rule_reference(running_ledger() if ledger is None else ledger,
                          rule_rates() if rates is None else rates, origins,
                          settings or rr_settings(), **rr_inputs(**overrides))


def rr_by_hand(origins, settings=None, ledger=None, rates=None, **overrides):
    """Every step called by hand at each origin, then sorted as the docstring says."""
    settings = settings or rr_settings()
    kwargs = rr_inputs(**overrides)
    ledger = running_ledger() if ledger is None else ledger
    rates = rule_rates() if rates is None else rates
    horizons = settings.receivables.horizons
    totals, invoices = [], []
    for origin in origins:
        made = rule_invoices(ledger, rates, WEEK_ZERO, origin, horizons,
                             settings.receivables.min_cleared_for_delay,
                             settings.evaluation.rule_doubtful_days)
        topups = topup_forecasts(kwargs["new_cash"], kwargs["refunds"], kwargs["unidentified"],
                                 kwargs["levels"], kwargs["calendar"], [origin], horizons,
                                 settings.topups.method, settings.topups.window,
                                 settings.topups.with_level)
        projected = projected_levels(WEEK_ZERO, origin,
                                     [origin + h for h in range(1, horizons + 1)])
        totals.append(rule_totals(made, topups, projected, kwargs["companies"], origin,
                                  horizons))
        invoices.append(made)
    stacked = pd.concat(totals, ignore_index=True)
    order = sorted(set(kwargs["companies"])) + [CONSOLIDATED]
    parts = [stacked[stacked["company_code"] == c].sort_values(["origin_week", "h"])
             for c in order]
    return (pd.concat(parts, ignore_index=True), pd.concat(invoices, ignore_index=True))


@pytest.fixture(scope="module")
def rr_result():
    return rr_run()


def test_rule_reference_returns_a_rule_reference(rr_result):
    assert isinstance(rr_result, RuleReference)
    assert list(rr_result.totals.columns) == RULE_COLUMNS
    assert list(rr_result.invoices.columns) == RULE_INVOICE_COLUMNS


def test_rule_reference_equals_the_steps_by_hand(rr_result):
    totals, invoices = rr_by_hand(RR_ORIGINS)
    pd.testing.assert_frame_equal(rr_result.totals, totals)
    pd.testing.assert_frame_equal(rr_result.invoices, invoices)


def test_rule_reference_totals_sorted_companies_then_all_then_origin_and_h(rr_result):
    expected = [(c, o, h) for c in ["TR02", "TR04", CONSOLIDATED] for o in RR_ORIGINS
                for h in range(1, RULE_HORIZONS + 1)]
    found = list(zip(rr_result.totals["company_code"], rr_result.totals["origin_week"],
                     rr_result.totals["h"]))
    assert found == expected
    assert list(rr_result.totals.index) == list(range(len(expected)))


def test_rule_reference_all_comes_last_even_when_a_company_sorts_after_it():
    lines = running_lines()
    for line in lines:
        if line["company_code"] == "TR04":
            line["company_code"] = "ZZ"
            line["key"] = line["key"].replace("TR04", "ZZ")
    new_cash, refunds, unidentified = f_side_tables(companies=["AA", "ZZ"])
    # TR02's lines are renamed AA (sorts before ALL), TR04's ZZ (sorts after it).
    for line in lines:
        if line["company_code"] == "TR02":
            line["company_code"] = "AA"
            line["key"] = line["key"].replace("TR02", "AA")
    found = rule_reference(rl_ledger(lines), rule_rates(), [20], rr_settings(),
                           **rr_inputs(companies=["ZZ", "AA"], new_cash=new_cash,
                                       refunds=refunds, unidentified=unidentified)).totals
    assert list(dict.fromkeys(found["company_code"])) == ["AA", "ZZ", CONSOLIDATED]


def test_rule_reference_invoices_are_stacked_in_origin_order(rr_result):
    origins = list(rr_result.invoices["origin_week"])
    assert origins == sorted(origins)
    assert set(origins) == set(RR_ORIGINS)
    assert list(rr_result.invoices.index) == list(range(len(rr_result.invoices)))


@pytest.mark.parametrize("origin", RR_ORIGINS)
def test_rule_reference_each_origins_invoices_equal_rule_invoices_alone(rr_result, origin):
    found = rr_result.invoices[rr_result.invoices["origin_week"] == origin]
    alone = run_rule_invoices(origin=origin)
    pd.testing.assert_frame_equal(found.reset_index(drop=True), alone)


def test_rule_reference_at_t_holds_the_running_example(rr_result):
    at_t = rr_result.invoices[rr_result.invoices["origin_week"] == RULE_ORIGIN]
    table = by_document(at_t)
    for document, (status, h, outstanding, delay, source, *_rest) in RUNNING.items():
        assert table.loc[document, "status"] == status, document
        assert table.loc[document, "outstanding_try"] == pytest.approx(outstanding), document
    totals = rr_result.totals
    for (company, h), (invoice, late, fallback) in RUNNING_TOTALS.items():
        row = total_row(totals, company, h)
        assert row["invoice_cash"] == pytest.approx(invoice)
        assert row["late_cash"] == pytest.approx(late)
        assert row["fallback_delay_cash"] == pytest.approx(fallback)


def test_rule_reference_invoice_cash_is_the_expected_and_late_outstanding(rr_result):
    counted = rr_result.invoices[rr_result.invoices["status"].isin([EXPECTED, LATE])]
    sums = counted.groupby(["company_code", "origin_week", "h"])["outstanding_try"].sum()
    companies = rr_result.totals[rr_result.totals["company_code"] != CONSOLIDATED]
    for _, row in companies.iterrows():
        expected = sums.get((row["company_code"], row["origin_week"], row["h"]), 0.0)
        assert row["invoice_cash"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_rule_reference_topup_cash_is_the_topup_times_the_level_projected_at_the_origin(
        rr_result):
    kwargs = rr_inputs()
    settings = rr_settings()
    for origin in RR_ORIGINS:
        topups = topup_forecasts(kwargs["new_cash"], kwargs["refunds"], kwargs["unidentified"],
                                 kwargs["levels"], kwargs["calendar"], [origin], RULE_HORIZONS,
                                 settings.topups.method, settings.topups.window,
                                 settings.topups.with_level)
        projected = projected_levels(WEEK_ZERO, origin,
                                     [origin + h for h in range(1, RULE_HORIZONS + 1)])
        for company in RULE_COMPANIES:
            for h in range(1, RULE_HORIZONS + 1):
                topup = topups[(topups["company_code"] == company) & (topups["h"] == h)]
                row = total_row(rr_result.totals, company, h, origin)
                level_now = float(projected.loc[origin + h])
                assert row["price_level"] == pytest.approx(level_now, rel=1e-15)
                assert row["topup_cash"] == pytest.approx(
                    float(topup["topup"].iloc[0]) * level_now, rel=1e-12)
                assert row["forecast"] == pytest.approx(row["invoice_cash"] + row["topup_cash"],
                                                        rel=1e-12)


def test_rule_reference_all_rows_are_the_sums_with_the_same_price_level(rr_result):
    amounts = ["invoice_cash", "late_cash", "fallback_delay_cash", "topup_cash", "forecast"]
    totals = rr_result.totals
    companies = totals[totals["company_code"] != CONSOLIDATED]
    consolidated = totals[totals["company_code"] == CONSOLIDATED].set_index(["origin_week", "h"])
    summed = companies.groupby(["origin_week", "h"])[amounts].sum()
    np.testing.assert_allclose(consolidated.loc[summed.index, amounts].to_numpy(),
                               summed.to_numpy(), rtol=1e-14)
    levels = companies.groupby(["origin_week", "h"])["price_level"].agg(["min", "max"])
    assert (levels["min"] == levels["max"]).all()
    np.testing.assert_array_equal(consolidated.loc[levels.index, "price_level"].to_numpy(),
                                  levels["min"].to_numpy())


def test_rule_reference_target_week_is_origin_plus_h(rr_result):
    totals = rr_result.totals
    assert (totals["target_week"] == totals["origin_week"] + totals["h"]).all()


def test_rule_reference_dtypes(rr_result):
    totals = rr_result.totals
    assert totals["company_code"].dtype == object
    for column in ("origin_week", "h", "target_week"):
        assert totals[column].dtype == "int64", column
    for column in ("invoice_cash", "late_cash", "fallback_delay_cash", "topup_cash",
                   "price_level", "forecast"):
        assert totals[column].dtype == "float64", column
    assert rr_result.invoices["h"].dtype == "int64"
    assert rr_result.invoices["origin_week"].dtype == "int64"


def test_rule_reference_values_are_finite(rr_result):
    numbers = rr_result.totals.select_dtypes("number")
    assert np.isfinite(numbers.to_numpy()).all()


@pytest.mark.parametrize("given", [[22, 18, 20], [20, 22, 18, 20, 22]])
def test_rule_reference_unsorted_and_repeated_origins_equal_sorted_unique(rr_result, given):
    found = rr_run(origins=given)
    pd.testing.assert_frame_equal(found.totals, rr_result.totals)
    pd.testing.assert_frame_equal(found.invoices, rr_result.invoices)


def test_rule_reference_unsorted_and_repeated_companies_equal_sorted(rr_result):
    found = rr_run(companies=["TR04", "TR02", "TR04"])
    pd.testing.assert_frame_equal(found.totals, rr_result.totals)


def test_rule_reference_a_single_origin_equals_its_slice(rr_result):
    alone = rr_run(origins=[RULE_ORIGIN])
    sliced = rr_result.totals[rr_result.totals["origin_week"] == RULE_ORIGIN]
    pd.testing.assert_frame_equal(alone.totals, sliced.reset_index(drop=True))


def test_rule_reference_uses_the_doubtful_days_setting():
    found = rr_run(origins=[RULE_ORIGIN], settings=rr_settings(doubtful_days=89))
    table = by_document(found.invoices)
    assert table.loc["A5", "status"] == DOUBTFUL
    # A5's 30 TRY leaves TR02's h 1 invoice cash and late cash.
    row = total_row(found.totals, "TR02", 1)
    assert row["invoice_cash"] == pytest.approx(4430.0 - 30.0)
    assert row["late_cash"] == pytest.approx(100.0)


def test_rule_reference_uses_the_min_cleared_setting():
    found = rr_run(origins=[RULE_ORIGIN], settings=rr_settings(min_cleared=1))
    assert by_document(found.invoices).loc["B1", "delay_from"] == FROM_CUSTOMER
    assert total_row(found.totals, "TR02", 1)["fallback_delay_cash"] == pytest.approx(0.0)


def test_rule_reference_uses_the_horizons_setting():
    new_cash, refunds, unidentified = f_side_tables(companies=RULE_COMPANIES, horizons=6)
    found = rr_run(origins=[RULE_ORIGIN], settings=rr_settings(horizons=6), new_cash=new_cash,
                   refunds=refunds, unidentified=unidentified)
    assert set(found.totals["h"]) == set(range(1, 7))
    assert total_row(found.totals, "TR02", 6)["invoice_cash"] == pytest.approx(20.0)   # A4


def test_rule_reference_no_leak_from_the_ledger_after_each_origin(rr_result):
    found = rr_run(ledger=full_ledger(), origins=[RULE_ORIGIN])
    sliced_totals = rr_result.totals[rr_result.totals["origin_week"] == RULE_ORIGIN]
    sliced_invoices = rr_result.invoices[rr_result.invoices["origin_week"] == RULE_ORIGIN]
    pd.testing.assert_frame_equal(found.totals, sliced_totals.reset_index(drop=True))
    pd.testing.assert_frame_equal(found.invoices, sliced_invoices.reset_index(drop=True))


def test_rule_reference_is_deterministic(rr_result):
    again = rr_run()
    pd.testing.assert_frame_equal(again.totals, rr_result.totals)
    pd.testing.assert_frame_equal(again.invoices, rr_result.invoices)


def test_rule_reference_does_not_change_its_inputs():
    ledger, rates = running_ledger(), rule_rates()
    kwargs = rr_inputs()
    copies = {name: value.copy() for name, value in kwargs.items()
              if isinstance(value, (pd.DataFrame, pd.Series))}
    companies = list(kwargs["companies"])
    ledger_before, rates_before = ledger.copy(), rates.copy()
    origins = [22, 18, 20]
    rule_reference(ledger, rates, origins, rr_settings(), **kwargs)
    pd.testing.assert_frame_equal(ledger, ledger_before)
    pd.testing.assert_frame_equal(rates, rates_before)
    for name, copy in copies.items():
        if isinstance(copy, pd.DataFrame):
            pd.testing.assert_frame_equal(kwargs[name], copy)
        else:
            pd.testing.assert_series_equal(kwargs[name], copy)
    assert kwargs["companies"] == companies
    assert origins == [22, 18, 20]


# --- rule_reference: refusals ---------------------------------------------------------------

@pytest.mark.parametrize("origins", [[], ()])
def test_rule_reference_no_origins_exact_message(origins):
    with pytest.raises(ValueError) as error:
        rr_run(origins=origins)
    assert str(error.value) == "there are no origins to forecast at"


def test_rule_reference_all_in_companies_exact_message():
    with pytest.raises(ValueError) as error:
        rr_run(companies=["TR02", CONSOLIDATED, "TR04"])
    assert str(error.value) == ("'ALL' is the code of the consolidated rows; it cannot be a "
                                "company")


def test_rule_reference_no_origins_is_checked_before_all():
    with pytest.raises(ValueError, match="no origins"):
        rr_run(origins=[], companies=[CONSOLIDATED])


def test_rule_reference_an_invoice_of_a_company_outside_the_list_raises():
    new_cash, refunds, unidentified = f_side_tables(companies=["TR02"])
    with pytest.raises(ValueError) as error:
        rr_run(origins=[RULE_ORIGIN], companies=["TR02"], new_cash=new_cash, refunds=refunds,
               unidentified=unidentified)
    assert str(error.value) == ("the open invoices at origin week 20 have companies not in the "
                                "list: ['TR04']")


def test_rule_reference_an_open_invoice_without_a_due_date_raises():
    ledger = rl_ledger(running_lines() + [
        rl("invoice", "NODUE", 10, CUT - days(5), None, customer="A")])
    with pytest.raises(ValueError, match=r"open invoices have no net due date"):
        rr_run(ledger=ledger, origins=[RULE_ORIGIN])


def test_rule_reference_a_top_up_row_missing_for_a_company_raises():
    # Side tables of TR02 only, but TR04 in the list: its top-ups have no row.
    new_cash, refunds, unidentified = f_side_tables(companies=["TR02"])
    with pytest.raises(ValueError, match="0 rows for company 'TR04', h 1 at origin week 18"):
        rr_run(new_cash=new_cash, refunds=refunds, unidentified=unidentified)


def test_rule_reference_week_zero_not_a_monday_raises():
    with pytest.raises(ValueError):
        rr_run(week_zero=WEEK_ZERO + timedelta(days=1))



# --- 2026-11-03 follow-up: the two new refusals --------------------------------------------
# rule_reference refuses an empty company list (after empty origins, before "ALL");
# rule_totals refuses top-ups of a company outside the list (after the invoices' check, before
# the per-(company, h) row count).

def test_rule_reference_empty_company_list_exact_message():
    with pytest.raises(ValueError) as error:
        rr_run(companies=[])
    assert str(error.value) == "the company list is empty"


def test_rule_reference_empty_company_list_with_no_invoices_is_refused_too():
    # Before the fix this returned 0 total rows and no error.
    with pytest.raises(ValueError) as error:
        rr_run(ledger=rl_ledger([]), origins=[RULE_ORIGIN], companies=[])
    assert str(error.value) == "the company list is empty"


def test_rule_reference_empty_origins_is_checked_before_the_empty_company_list():
    with pytest.raises(ValueError) as error:
        rr_run(origins=[], companies=[])
    assert str(error.value) == "there are no origins to forecast at"


def test_rule_reference_empty_company_list_is_checked_before_any_step():
    # A ledger that rule_invoices would refuse: the company list is found empty first.
    ledger = rl_ledger(running_lines() + [
        rl("invoice", "NODUE", 10, CUT - days(5), None, customer="A")])
    with pytest.raises(ValueError) as error:
        rr_run(ledger=ledger, companies=[])
    assert str(error.value) == "the company list is empty"


def test_rule_reference_all_check_still_follows_the_empty_list_check():
    with pytest.raises(ValueError) as error:
        rr_run(companies=[CONSOLIDATED])
    assert str(error.value) == ("'ALL' is the code of the consolidated rows; it cannot be a "
                                "company")


def test_rule_totals_top_ups_of_a_company_outside_the_list_exact_message(running):
    with pytest.raises(ValueError) as error:
        rule_totals(running, rule_topups(["TR02", "TR04", "TR05"]), rule_projected(),
                    ["TR02", "TR04"], RULE_ORIGIN, RULE_HORIZONS)
    assert str(error.value) == ("the top-ups at origin week 20 have companies not in the "
                                "list: ['TR05']")


def test_rule_totals_top_ups_outside_the_list_are_listed_sorted(running):
    topups = pd.concat([rule_topups(["TR02", "TR04"]),
                        rule_topups(["TR05"]).assign(company_code="ZZ"),
                        rule_topups(["TR05"]).assign(company_code="AB")], ignore_index=True)
    with pytest.raises(ValueError) as error:
        rule_totals(running, topups, rule_projected(), ["TR02", "TR04"], RULE_ORIGIN,
                    RULE_HORIZONS)
    assert str(error.value) == ("the top-ups at origin week 20 have companies not in the "
                                "list: ['AB', 'ZZ']")


def test_rule_totals_a_single_outside_top_up_row_is_refused(running):
    extra = pd.DataFrame([{"company_code": "TR05", "origin_week": RULE_ORIGIN, "h": 3,
                           "target_week": RULE_ORIGIN + 3, "topup": 0.0}])
    topups = pd.concat([rule_topups(["TR02", "TR04"]), extra], ignore_index=True)
    with pytest.raises(ValueError, match=r"top-ups at origin week 20 have companies not in "
                                         r"the list: \['TR05'\]"):
        rule_totals(running, topups, rule_projected(), ["TR02", "TR04"], RULE_ORIGIN,
                    RULE_HORIZONS)


def test_rule_totals_the_invoices_check_comes_before_the_top_ups_check(running):
    # Both outside: TR04's invoices (list [TR02]) and TR05's top-ups.
    with pytest.raises(ValueError) as error:
        rule_totals(running, rule_topups(["TR02", "TR05"]), rule_projected(), ["TR02"],
                    RULE_ORIGIN, RULE_HORIZONS)
    assert str(error.value) == ("the open invoices at origin week 20 have companies not in the "
                                "list: ['TR04']")


def test_rule_totals_the_top_ups_check_comes_before_the_row_count(running):
    # TR05 outside, and TR02 h 3 missing: the outside company is reported.
    topups = rule_topups(["TR02", "TR04", "TR05"])
    topups = topups[~((topups["company_code"] == "TR02") & (topups["h"] == 3))]
    with pytest.raises(ValueError) as error:
        rule_totals(running, topups, rule_projected(), ["TR02", "TR04"], RULE_ORIGIN,
                    RULE_HORIZONS)
    assert str(error.value) == ("the top-ups at origin week 20 have companies not in the "
                                "list: ['TR05']")


def test_rule_totals_top_ups_of_exactly_the_listed_companies_pass(running, running_totals):
    found = rule_totals(running, rule_topups(), rule_projected(), TOTALS_COMPANIES,
                        RULE_ORIGIN, RULE_HORIZONS)
    pd.testing.assert_frame_equal(found, running_totals)


def test_rule_totals_no_invoices_and_outside_top_ups_is_still_refused(running):
    with pytest.raises(ValueError, match=r"top-ups at origin week 20 .*\['TR05'\]"):
        rule_totals(running.iloc[0:0], rule_topups(["TR02", "TR05"]), rule_projected(),
                    ["TR02"], RULE_ORIGIN, RULE_HORIZONS)


def test_rule_reference_side_tables_of_a_company_outside_the_list_are_refused():
    # Side tables of TR02, TR04 and TR05; the list (and the invoices) only TR02 and TR04.
    new_cash, refunds, unidentified = f_side_tables(companies=["TR02", "TR04", "TR05"])
    with pytest.raises(ValueError) as error:
        rr_run(new_cash=new_cash, refunds=refunds, unidentified=unidentified)
    assert str(error.value) == ("the top-ups at origin week 18 have companies not in the "
                                "list: ['TR05']")


def test_rule_reference_invoices_outside_are_reported_before_top_ups_outside():
    new_cash, refunds, unidentified = f_side_tables(companies=["TR02", "TR05"])
    with pytest.raises(ValueError) as error:
        rr_run(origins=[RULE_ORIGIN], companies=["TR02"], new_cash=new_cash, refunds=refunds,
               unidentified=unidentified)
    assert str(error.value) == ("the open invoices at origin week 20 have companies not in the "
                                "list: ['TR04']")
