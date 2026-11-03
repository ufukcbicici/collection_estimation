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
