"""Tests for `hurdle/evaluate.py`: `actual_totals` (U14.1), the cash that really came per company
and consolidated ("ALL") in the target week of every origin and h, nominal and deflated.

Every table is hand-made; nothing is read from data/. The properties that matter most are
tested directly: actual = gross receipts + refunds of the TARGET week, deflated by the TARGET
week's level; a missing week or a week past the calendar has no row (no data is not 0), while
a company with no line in a week with data gets a real 0.0; "ALL" is the sum of the companies.

U14.2, at the end: `rolling_forecasts`, `forecast.forecast` at every origin, stacked, on the
small training table and side tables of test_forecast.py. Every cheap check must fire before
the first fit, which the tests prove by replacing the fits with spies that fail if called.

U14.5, at the very end: `stack_forecasts`, `score_row` and `evaluation_scores`, on hand-made
tables whose every score is worked out by hand in the comments.
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


def test_references_are_the_three_variants_in_order():
    # Each variant is (with_level, with_trend); the order is the row order of the output.
    assert REFERENCES == {"deflated": (True, False),
                          "deflated_nolevel": (False, False),
                          "deflated_trend": (False, True)}
    assert list(REFERENCES) == ["deflated", "deflated_nolevel", "deflated_trend"]


def test_references_variants_are_distinct_settings():
    # Two names with the same (with_level, with_trend) would be the same forecast twice.
    settings = list(REFERENCES.values())
    assert len(set(settings)) == len(settings)
    for with_level, with_trend in settings:
        assert isinstance(with_level, bool) and isinstance(with_trend, bool)


# --- shape, order, dtypes -------------------------------------------------------------------

def test_reference_has_exactly_the_reference_columns(d_result):
    assert list(d_result.columns) == REFERENCE_COLUMNS


def test_reference_row_count(d_result):
    # (companies + ALL) x origins x horizons x the three variants.
    assert len(d_result) == (len(D_COMPANIES) + 1) * len(D_ORIGINS) * D_HORIZONS * 3


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
    # One row per variant (three) for each company, ALL last.
    assert list(found["company_code"]) == ["AAA"] * 3 + ["ZZZ"] * 3 + ["ALL"] * 3


def test_reference_integer_company_codes_work_and_sort_numerically():
    weekly = d_weekly()
    weekly["company_code"] = weekly["company_code"].map({"1000": 10, "2000": 9})
    found = d_run(weekly=weekly, companies=[10, 9], origins=[20], horizons=1)
    assert list(found["company_code"]) == [9] * 3 + [10] * 3 + ["ALL"] * 3


def test_reference_company_with_no_cash_at_all_gets_zero_and_falls_back():
    found = d_run(companies=["1000", "2000", "3000"], origins=[20], horizons=2)
    part = found[found["company_code"] == "3000"]
    assert len(part) == 2 * 3          # two horizons x three variants
    assert (part["real_forecast"] == 0.0).all()
    assert (part["forecast"] == 0.0).all()
    assert part["fallback"].all()     # a prediction of 0 is not above 0


def test_reference_single_origin_single_horizon():
    found = d_run(origins=[20], horizons=1)
    assert len(found) == 3 * 1 * 1 * 3
    assert set(found["h"]) == {1}


# --- the values: by hand against deflated_forecasts -------------------------------------------

@pytest.mark.parametrize("company", ["1000", "2000", CONSOLIDATED])
def test_reference_equals_deflated_forecasts_on_the_hand_built_series(d_result, company):
    kw = d_inputs()
    series = d_series(kw["weekly"], kw["weeks"], kw["levels"],
                      None if company == CONSOLIDATED else company)
    for origin in D_ORIGINS:
        for h in range(1, D_HORIZONS + 1):
            for reference, (with_level, with_trend) in REFERENCES.items():
                made = benchmark.deflated_forecasts(series, kw["calendar"], h, origin, [origin],
                                                    with_level, with_trend).iloc[0]
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


def test_reference_the_trend_variant_differs_from_the_no_level_one(d_result):
    # deflated_trend is deflated_nolevel plus the week number: on this series it must move.
    wide = d_result.pivot(index=["company_code", "origin_week", "h"], columns="reference",
                          values="real_forecast")
    assert not np.allclose(wide["deflated_trend"], wide["deflated_nolevel"])


@pytest.mark.parametrize("company", ["1000", "2000", CONSOLIDATED])
def test_reference_deflated_trend_is_the_trend_forecast_times_the_projected_level(d_result,
                                                                                  company):
    kw = d_inputs()
    # Step 1: the deflated series by hand (company or consolidated).
    series = d_series(kw["weekly"], kw["weeks"], kw["levels"],
                      None if company == CONSOLIDATED else company)
    for origin in D_ORIGINS:
        # Step 2: the price levels projected at this origin for its target weeks.
        targets = [origin + h for h in range(1, D_HORIZONS + 1)]
        projected = projected_levels(WEEK_ZERO, origin, targets)
        for h in range(1, D_HORIZONS + 1):
            # Step 3: the direct call, no level, with the trend, refitted at the origin.
            made = benchmark.deflated_forecasts(series, kw["calendar"], h, origin, [origin],
                                                with_level=False, with_trend=True).iloc[0]
            expected_level = float(projected.loc[origin + h])
            # Step 4: the reference row must be that forecast, re-inflated at the origin.
            found = d_row(d_result, company, origin, h, "deflated_trend")
            assert found["real_forecast"] == pytest.approx(made["forecast"], rel=1e-9)
            assert found["price_level"] == expected_level
            assert found["forecast"] == pytest.approx(made["forecast"] * expected_level,
                                                      rel=1e-9)
            assert bool(found["fallback"]) == bool(made["fallback"])


# --- consolidated ------------------------------------------------------------------------------

def test_reference_all_is_fitted_on_the_summed_series(d_result):
    kw = d_inputs()
    total = d_series(kw["weekly"], kw["weeks"], kw["levels"])
    by_company = sum(d_series(kw["weekly"], kw["weeks"], kw["levels"], c) for c in D_COMPANIES)
    pd.testing.assert_series_equal(total, by_company, check_exact=False, rtol=1e-12)
    for origin in D_ORIGINS:
        for h in range(1, D_HORIZONS + 1):
            for reference, (with_level, with_trend) in REFERENCES.items():
                made = benchmark.deflated_forecasts(total, kw["calendar"], h, origin, [origin],
                                                    with_level, with_trend).iloc[0]
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
    # Three variants per target week.
    assert list(found[found["company_code"] == "1000"]["target_week"]) == (
        [40] * 3 + [41] * 3 + [42] * 3)


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
    assert len(found) == 3 * 3          # (two companies + ALL) x three variants


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
    assert len(found) == 3 * 3          # (two companies + ALL) x three variants
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



# =============================================================================================
# U14.5a, `stack_forecasts`: every method's forecasts in one long table (the hurdle model's
# `total`, if it ran; each variant of the `deflated` reference; the financial rule).
# U14.5b, `score_row` and `evaluation_scores`: WAPE, bias, mean absolute error and (for "ALL"
# only) MAPE, per h, pooled over every (origin, h), and for each origin's h-week total.
#
# Every table is hand-made and every expected number is worked out by hand in the comments.
# =============================================================================================

from collection_estimation.hurdle.evaluate import (  # noqa: E402
    HORIZON_TOTAL, HURDLE, POOLED, RULE, SCORE_COLUMNS, STACK_COLUMNS, RuleReference,
    evaluation_scores, score_row, stack_forecasts)

S_KEYS = ["company_code", "origin_week", "h", "target_week"]
NAN = float("nan")


# --- builders --------------------------------------------------------------------------------

def s_forecasts(rows):
    """rows: (method, company_code, origin_week, h, target_week, forecast)."""
    frame = pd.DataFrame(rows, columns=STACK_COLUMNS)
    for column in ("origin_week", "h", "target_week"):
        frame[column] = frame[column].astype("int64")
    frame["forecast"] = frame["forecast"].astype("float64")
    return frame


def s_actual(rows):
    """rows: (company_code, origin_week, h, target_week, actual); actual_real is a decoy."""
    frame = pd.DataFrame(rows, columns=S_KEYS + ["actual"])
    for column in ("origin_week", "h", "target_week"):
        frame[column] = frame[column].astype("int64")
    frame["actual"] = frame["actual"].astype("float64")
    frame["actual_real"] = -1.0          # never scored: scoring it would make every WAPE NaN
    return frame


# Origins 10 and 11, h = 1, 2, so target weeks 11, 12 (origin 10) and 12, 13 (origin 11).
S_PAIRS = [(10, 1, 11), (10, 2, 12), (11, 1, 12), (11, 2, 13)]


def s_rolled(total_by_company, real_total=-5.0):
    """A Forecast whose totals have `total` (company -> 4 values in S_PAIRS order) and a
    `real_total` that is deliberately different (-5.0 everywhere)."""
    rows = []
    for company, totals in total_by_company.items():
        for (origin, h, target), total in zip(S_PAIRS, totals):
            rows.append({"company_code": company, "origin_week": origin, "h": h,
                         "target_week": target, "customers": 0.0, "new_customer_cash": 0.0,
                         "unidentified_cash": 0.0, "refunds": 0.0, "real_total": real_total,
                         "price_level": 2.0, "total": total, "fallback": False})
    return Forecast(totals=pd.DataFrame(rows), customers=pd.DataFrame({"key": ["K0"]}))


def s_reference(value_by_variant, companies=("TR02", "ALL")):
    """A deflated_reference-shaped table: per variant, per company, the four S_PAIRS rows;
    forecast = base + 10 x pair index (real_forecast is half of it, a decoy)."""
    rows = []
    for name, base in value_by_variant.items():
        for company in companies:
            for i, (origin, h, target) in enumerate(S_PAIRS):
                forecast = base + 10.0 * i
                rows.append({"company_code": company, "origin_week": origin, "h": h,
                             "target_week": target, "reference": name,
                             "real_forecast": forecast / 2.0, "price_level": 2.0,
                             "forecast": forecast, "fallback": False})
    return pd.DataFrame(rows)


def s_rule(forecast_by_company):
    """A RuleReference whose totals have `forecast` (company -> 4 values in S_PAIRS order)."""
    rows = []
    for company, forecasts in forecast_by_company.items():
        for (origin, h, target), forecast in zip(S_PAIRS, forecasts):
            rows.append({"company_code": company, "origin_week": origin, "h": h,
                         "target_week": target, "invoice_cash": forecast / 2,
                         "late_cash": 0.0, "fallback_delay_cash": 0.0,
                         "topup_cash": forecast / 2, "price_level": 1.0,
                         "forecast": forecast})
    return RuleReference(totals=pd.DataFrame(rows), invoices=pd.DataFrame({"key": ["K0"]}))


ALL_THREE = {"deflated": 1000.0, "deflated_nolevel": 2000.0, "deflated_trend": 3000.0}


@pytest.fixture
def stack_inputs():
    rolled = s_rolled({"TR02": [1.0, 2.0, 3.0, 4.0], "ALL": [5.0, 6.0, 7.0, 8.0]})
    reference = s_reference(ALL_THREE)
    rule = s_rule({"TR02": [11.0, 12.0, 13.0, 14.0], "ALL": [15.0, 16.0, 17.0, 18.0]})
    return rolled, reference, rule


def score_of(scores, method, company, horizon):
    found = scores[(scores["method"] == method) & (scores["company_code"] == company)
                   & (scores["horizon"] == horizon)]
    assert len(found) == 1, f"expected one row for {method}, {company}, {horizon}: {len(found)}"
    return found.iloc[0]


# --- constants -------------------------------------------------------------------------------

def test_stack_and_score_constants():
    assert HURDLE == "hurdle"
    assert RULE == "rule"
    assert POOLED == "pooled"
    assert HORIZON_TOTAL == "total"
    assert STACK_COLUMNS == ["method", "company_code", "origin_week", "h", "target_week",
                             "forecast"]
    assert SCORE_COLUMNS == ["method", "company_code", "horizon", "rows", "forecast", "actual",
                             "abs_error", "wape", "bias", "mean_abs_error", "mape"]


def test_method_names_do_not_clash_with_the_reference_names():
    # A clash would merge two methods' rows into one group in evaluation_scores.
    assert HURDLE not in REFERENCES and RULE not in REFERENCES and HURDLE != RULE


# --- stack_forecasts ---------------------------------------------------------------------------

def test_stack_has_exactly_the_stack_columns_and_a_fresh_index(stack_inputs):
    stacked = stack_forecasts(*stack_inputs)
    assert list(stacked.columns) == STACK_COLUMNS
    assert list(stacked.index) == list(range(len(stacked)))


def test_stack_with_the_hurdle_methods_in_order_and_row_counts(stack_inputs):
    stacked = stack_forecasts(*stack_inputs)
    # 8 rows per method: 2 companies x 4 (origin, h); the methods one block after the other.
    expected_order = [HURDLE, "deflated", "deflated_nolevel", "deflated_trend", RULE]
    assert list(stacked["method"]) == [m for m in expected_order for _ in range(8)]


def test_stack_without_the_hurdle_has_no_hurdle_rows(stack_inputs):
    _, reference, rule = stack_inputs
    stacked = stack_forecasts(None, reference, rule)
    expected_order = ["deflated", "deflated_nolevel", "deflated_trend", RULE]
    assert list(stacked["method"]) == [m for m in expected_order for _ in range(8)]
    assert list(stacked.columns) == STACK_COLUMNS
    assert list(stacked.index) == list(range(32))


def test_stack_the_hurdle_forecast_is_its_total_not_its_real_total(stack_inputs):
    stacked = stack_forecasts(*stack_inputs)
    hurdle = stacked[stacked["method"] == HURDLE]
    # real_total is -5.0 everywhere in s_rolled; total runs 1 ... 8.
    assert list(hurdle["forecast"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert -5.0 not in set(stacked["forecast"])


def test_stack_the_hurdle_keys_come_from_its_totals(stack_inputs):
    stacked = stack_forecasts(*stack_inputs)
    hurdle = stacked[stacked["method"] == HURDLE].reset_index(drop=True)
    expected = stack_inputs[0].totals[S_KEYS].reset_index(drop=True)
    pd.testing.assert_frame_equal(hurdle[S_KEYS], expected, check_dtype=False)


def test_stack_each_reference_variant_takes_its_own_forecast_column(stack_inputs):
    stacked = stack_forecasts(*stack_inputs)
    for name, base in ALL_THREE.items():
        mine = stacked[stacked["method"] == name]
        # TR02 then ALL, each base + 0, 10, 20, 30 (never real_forecast, half of it).
        steps = [base + 10.0 * i for i in range(4)]
        assert list(mine["forecast"]) == steps + steps, name


def test_stack_the_rule_forecast_is_its_forecast_column(stack_inputs):
    stacked = stack_forecasts(*stack_inputs)
    rule = stacked[stacked["method"] == RULE]
    assert list(rule["forecast"]) == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    assert list(rule["company_code"]) == ["TR02"] * 4 + ["ALL"] * 4


def test_stack_each_methods_rows_keep_their_input_order():
    # 1. The reference's 24 rows in a scrambled, interleaved order.
    reference = s_reference(ALL_THREE)
    scrambled = reference.iloc[[23, 0, 13, 5, 2, 17, 8, 11, 1, 19, 4, 22, 6, 15, 3, 9, 12,
                                21, 7, 14, 18, 10, 20, 16]].reset_index(drop=True)
    # 2. The hurdle's totals reversed, so its forecasts read 1, 2, 3, 4.
    rolled = s_rolled({"ALL": [4.0, 3.0, 2.0, 1.0]})
    rolled = Forecast(totals=rolled.totals.iloc[::-1].reset_index(drop=True),
                      customers=rolled.customers)
    # 3. The rule's totals in the order 2, 0, 3, 1.
    rule = s_rule({"ALL": [1.0, 2.0, 3.0, 4.0]})
    rule = RuleReference(totals=rule.totals.iloc[[2, 0, 3, 1]].reset_index(drop=True),
                         invoices=rule.invoices)
    stacked = stack_forecasts(rolled, scrambled, rule)
    # 4. Each method's rows as they came: nothing is re-sorted.
    assert list(stacked.loc[stacked["method"] == HURDLE, "forecast"]) == [1.0, 2.0, 3.0, 4.0]
    for name in REFERENCES:
        expected = scrambled.loc[scrambled["reference"] == name, "forecast"]
        found = stacked.loc[stacked["method"] == name, "forecast"]
        assert list(found) == list(expected), name
    assert list(stacked.loc[stacked["method"] == RULE, "forecast"]) == [3.0, 1.0, 4.0, 2.0]


def test_stack_values_carry_over_row_by_row(stack_inputs):
    # Each stacked row is the input row it came from: the stack rebuilt by hand.
    rolled, reference, rule = stack_inputs
    stacked = stack_forecasts(rolled, reference, rule)
    by_hand = [rolled.totals[S_KEYS + ["total"]].rename(columns={"total": "forecast"})
               .assign(method=HURDLE)]
    for name in REFERENCES:
        by_hand.append(reference.loc[reference["reference"] == name, S_KEYS + ["forecast"]]
                       .assign(method=name))
    by_hand.append(rule.totals[S_KEYS + ["forecast"]].assign(method=RULE))
    expected = pd.concat(by_hand, ignore_index=True)[STACK_COLUMNS]
    pd.testing.assert_frame_equal(stacked, expected, check_dtype=False)


def test_stack_method_and_company_code_are_object_dtype(stack_inputs):
    rolled, reference, rule = stack_inputs
    # Even if the inputs hold the company code as a category or a pandas string.
    reference = reference.assign(company_code=reference["company_code"].astype("category"))
    rule = RuleReference(totals=rule.totals.assign(
        company_code=rule.totals["company_code"].astype("string")), invoices=rule.invoices)
    stacked = stack_forecasts(rolled, reference, rule)
    assert stacked["method"].dtype == object
    assert stacked["company_code"].dtype == object
    assert set(stacked["company_code"]) == {"TR02", "ALL"}


def test_stack_ignores_a_reference_variant_not_in_references(stack_inputs):
    rolled, reference, rule = stack_inputs
    extra = s_reference({"deflated_other": 7.0})
    stacked = stack_forecasts(rolled, pd.concat([reference, extra], ignore_index=True), rule)
    assert "deflated_other" not in set(stacked["method"])
    assert len(stacked) == 5 * 8


@pytest.mark.parametrize("missing", list(REFERENCES))
def test_stack_a_reference_variant_with_no_rows_raises_exact_message(stack_inputs, missing):
    rolled, reference, rule = stack_inputs
    without = reference[reference["reference"] != missing]
    with pytest.raises(ValueError) as error:
        stack_forecasts(rolled, without, rule)
    assert str(error.value) == f"the reference table has no {missing!r} rows"


def test_stack_an_empty_reference_names_the_first_variant(stack_inputs):
    _, reference, rule = stack_inputs
    with pytest.raises(ValueError) as error:
        stack_forecasts(None, reference.iloc[0:0], rule)
    assert str(error.value) == "the reference table has no 'deflated' rows"


def test_stack_does_not_change_its_inputs(stack_inputs):
    rolled, reference, rule = stack_inputs
    before = (rolled.totals.copy(), reference.copy(), rule.totals.copy())
    stack_forecasts(rolled, reference, rule)
    pd.testing.assert_frame_equal(rolled.totals, before[0])
    pd.testing.assert_frame_equal(reference, before[1])
    pd.testing.assert_frame_equal(rule.totals, before[2])


def test_stack_is_deterministic(stack_inputs):
    pd.testing.assert_frame_equal(stack_forecasts(*stack_inputs), stack_forecasts(*stack_inputs))


# --- score_row ---------------------------------------------------------------------------------

def test_score_row_by_hand_with_mape():
    # forecast 110, 150, 140 against actual 100, 200, 200: errors 10, 50, 60 (sum 120).
    forecast = pd.Series([110.0, 150.0, 140.0])
    actual = pd.Series([100.0, 200.0, 200.0])
    found = score_row("m", "ALL", "pooled", forecast, actual, True)
    assert found["rows"] == 3
    assert found["forecast"] == pytest.approx(400.0)
    assert found["actual"] == pytest.approx(500.0)
    assert found["abs_error"] == pytest.approx(120.0)
    assert found["wape"] == pytest.approx(120.0 / 500.0)                 # 0.24
    assert found["bias"] == pytest.approx(400.0 / 500.0 - 1)             # -0.2
    assert found["mean_abs_error"] == pytest.approx(40.0)
    assert found["mape"] == pytest.approx((0.1 + 0.25 + 0.3) / 3)        # 0.216667


def test_score_row_keys_are_the_score_columns_in_order():
    found = score_row("m", "TR02", "1", pd.Series([1.0]), pd.Series([2.0]), False)
    assert list(found) == SCORE_COLUMNS
    assert (found["method"], found["company_code"], found["horizon"]) == ("m", "TR02", "1")


def test_score_row_mape_is_nan_without_with_mape():
    found = score_row("m", "TR02", "1", pd.Series([110.0]), pd.Series([100.0]), False)
    assert np.isnan(found["mape"])
    assert found["wape"] == pytest.approx(0.1)


def test_score_row_mape_is_nan_if_any_actual_is_zero():
    # Sum of actual 100 > 0: WAPE and bias are defined; MAPE is not (a share of 0).
    found = score_row("m", "ALL", "1", pd.Series([5.0, 110.0]), pd.Series([0.0, 100.0]), True)
    assert np.isnan(found["mape"])
    assert found["wape"] == pytest.approx(15.0 / 100.0)
    assert found["bias"] == pytest.approx(115.0 / 100.0 - 1)


def test_score_row_mape_is_nan_if_any_actual_is_negative():
    # Errors 6 and 10 over actual -1 + 100 = 99.
    found = score_row("m", "ALL", "1", pd.Series([5.0, 110.0]), pd.Series([-1.0, 100.0]), True)
    assert np.isnan(found["mape"])
    assert found["wape"] == pytest.approx(16.0 / 99.0)


@pytest.mark.parametrize("actual", [[0.0, 0.0], [-10.0, 10.0], [-5.0, 1.0]])
def test_score_row_wape_and_bias_are_nan_if_actual_sums_to_zero_or_less(actual):
    found = score_row("m", "ALL", "1", pd.Series([3.0, 4.0]), pd.Series(actual), True)
    assert np.isnan(found["wape"]) and np.isnan(found["bias"]) and np.isnan(found["mape"])
    # The TRY numbers are still there.
    errors = [abs(3.0 - actual[0]), abs(4.0 - actual[1])]
    assert found["abs_error"] == pytest.approx(sum(errors))
    assert found["mean_abs_error"] == pytest.approx(sum(errors) / 2)
    assert found["actual"] == pytest.approx(sum(actual))


def test_score_row_a_perfect_forecast_scores_zero():
    found = score_row("m", "ALL", "1", pd.Series([7.0, 9.0]), pd.Series([7.0, 9.0]), True)
    assert found["wape"] == 0.0 and found["bias"] == 0.0 and found["mape"] == 0.0
    assert found["mean_abs_error"] == 0.0


def test_score_row_errors_cancel_in_bias_but_not_in_wape():
    # +50 and -50: bias 0, WAPE 100 / 200.
    found = score_row("m", "ALL", "1", pd.Series([150.0, 50.0]), pd.Series([100.0, 100.0]), True)
    assert found["bias"] == pytest.approx(0.0)
    assert found["wape"] == pytest.approx(0.5)
    assert found["mape"] == pytest.approx(0.5)


def test_score_row_numbers_are_python_floats_and_rows_an_int():
    found = score_row("m", "ALL", "1", pd.Series([1.0, 2.0]), pd.Series([2.0, 2.0]), True)
    for name in ("forecast", "actual", "abs_error", "wape", "bias", "mean_abs_error", "mape"):
        assert type(found[name]) is float, name
    assert found["rows"] == 2 and isinstance(found["rows"], int)


# --- evaluation_scores: the running example ----------------------------------------------------
# Companies TR02 and ALL, origins 10 and 11, h = 1, 2 (horizons 2). Actual: (11, 2) -> week 13
# has no data (no row).
#   ALL  actual: (10,1) 100  (10,2) 200  (11,1) 200
#   TR02 actual: (10,1)  40  (10,2)  80  (11,1)  80
# Method "hurdle":
#   ALL  forecast: (10,1) 110  (10,2) 150  (11,1) 140  (11,2) 999 (no actual: never scored)
#   TR02 forecast: (10,1)  50  (10,2)  60  (11,1)  80  (11,2) 999
# By hand, ALL:
#   h1: f 110+140 = 250, a 100+200 = 300, |e| 10+60 = 70: WAPE 70/300, bias 250/300-1 = -1/6,
#       mean |e| 35, MAPE (0.1 + 0.3)/2 = 0.2
#   h2: f 150, a 200, |e| 50: WAPE 0.25, bias -0.25, mean |e| 50, MAPE 0.25
#   pooled: f 400, a 500, |e| 120: WAPE 0.24, bias -0.2, mean |e| 40, MAPE 0.65/3
#   total: only origin 10 has both weeks: f 260, a 300, |e| 40: WAPE 40/300, bias -40/300,
#       mean |e| 40, MAPE 40/300; rows = 1 (one origin)
# By hand, TR02 (no MAPE):
#   h1: f 130, a 120, |e| 10: WAPE 1/12, bias 1/12, mean |e| 5
#   h2: f 60, a 80, |e| 20: WAPE 0.25, bias -0.25, mean |e| 20
#   pooled: f 190, a 200, |e| 30: WAPE 0.15, bias -0.05, mean |e| 10
#   total: origin 10, f 110, a 120, |e| 10: WAPE 1/12, bias -1/12, mean |e| 10
# Method "rule": the actual itself wherever there is one (every error 0), 999 at (11, 2).

E_HORIZONS = 2


def e_actual():
    return s_actual([
        ("TR02", 10, 1, 11, 40.0), ("TR02", 10, 2, 12, 80.0), ("TR02", 11, 1, 12, 80.0),
        ("ALL", 10, 1, 11, 100.0), ("ALL", 10, 2, 12, 200.0), ("ALL", 11, 1, 12, 200.0)])


def e_forecasts(unscored=999.0):
    hurdle = {"TR02": [50.0, 60.0, 80.0, unscored], "ALL": [110.0, 150.0, 140.0, unscored]}
    rule = {"TR02": [40.0, 80.0, 80.0, unscored], "ALL": [100.0, 200.0, 200.0, unscored]}
    rows = []
    for method, by_company in ((HURDLE, hurdle), (RULE, rule)):
        for company, values in by_company.items():
            for (origin, h, target), value in zip(S_PAIRS, values):
                rows.append((method, company, origin, h, target, value))
    return s_forecasts(rows)


@pytest.fixture
def e_scores():
    return evaluation_scores(e_forecasts(), e_actual(), E_HORIZONS)


def assert_score(row, rows, forecast, actual, abs_error, wape, bias, mean_abs_error, mape):
    """mape None means: must be NaN."""
    assert row["rows"] == rows
    assert row["forecast"] == pytest.approx(forecast)
    assert row["actual"] == pytest.approx(actual)
    assert row["abs_error"] == pytest.approx(abs_error)
    assert row["wape"] == pytest.approx(wape)
    assert row["bias"] == pytest.approx(bias)
    assert row["mean_abs_error"] == pytest.approx(mean_abs_error)
    if mape is None:
        assert np.isnan(row["mape"])
    else:
        assert row["mape"] == pytest.approx(mape)


def test_scores_have_exactly_the_score_columns(e_scores):
    assert list(e_scores.columns) == SCORE_COLUMNS
    assert list(e_scores.index) == list(range(len(e_scores)))


def test_scores_row_order_methods_companies_then_h_pooled_total(e_scores):
    # Per (method, company) in order of first appearance: "1", "2", pooled, total.
    found = list(zip(e_scores["method"], e_scores["company_code"], e_scores["horizon"]))
    expected = [(method, company, horizon)
                for method in (HURDLE, RULE) for company in ("TR02", "ALL")
                for horizon in ("1", "2", POOLED, HORIZON_TOTAL)]
    assert found == expected


def test_scores_consolidated_h1_by_hand(e_scores):
    row = score_of(e_scores, HURDLE, CONSOLIDATED, "1")
    assert_score(row, 2, 250.0, 300.0, 70.0, 70.0 / 300.0, -1.0 / 6.0, 35.0, 0.2)


def test_scores_consolidated_h2_by_hand(e_scores):
    row = score_of(e_scores, HURDLE, CONSOLIDATED, "2")
    assert_score(row, 1, 150.0, 200.0, 50.0, 0.25, -0.25, 50.0, 0.25)


def test_scores_consolidated_pooled_by_hand(e_scores):
    row = score_of(e_scores, HURDLE, CONSOLIDATED, POOLED)
    assert_score(row, 3, 400.0, 500.0, 120.0, 0.24, -0.2, 40.0, 0.65 / 3)


def test_scores_consolidated_total_by_hand_only_the_complete_origin(e_scores):
    # Origin 11 has data only for h1, so its total is left out: one origin scored.
    row = score_of(e_scores, HURDLE, CONSOLIDATED, HORIZON_TOTAL)
    assert_score(row, 1, 260.0, 300.0, 40.0, 40.0 / 300.0, -40.0 / 300.0, 40.0, 40.0 / 300.0)


def test_scores_company_rows_by_hand_and_without_mape(e_scores):
    assert_score(score_of(e_scores, HURDLE, "TR02", "1"), 2, 130.0, 120.0, 10.0, 1 / 12, 1 / 12,
                 5.0, None)
    assert_score(score_of(e_scores, HURDLE, "TR02", "2"), 1, 60.0, 80.0, 20.0, 0.25, -0.25,
                 20.0, None)
    assert_score(score_of(e_scores, HURDLE, "TR02", POOLED), 3, 190.0, 200.0, 30.0, 0.15, -0.05,
                 10.0, None)
    assert_score(score_of(e_scores, HURDLE, "TR02", HORIZON_TOTAL), 1, 110.0, 120.0, 10.0,
                 1 / 12, -1 / 12, 10.0, None)


def test_scores_mape_only_for_the_consolidated_rows(e_scores):
    consolidated = e_scores[e_scores["company_code"] == CONSOLIDATED]
    companies = e_scores[e_scores["company_code"] != CONSOLIDATED]
    assert consolidated["mape"].notna().all()
    assert companies["mape"].isna().all()


def test_scores_a_perfect_method_scores_zero_everywhere(e_scores):
    rule = e_scores[e_scores["method"] == RULE]
    assert (rule["wape"] == 0.0).all() and (rule["bias"] == 0.0).all()
    assert (rule["abs_error"] == 0.0).all() and (rule["mean_abs_error"] == 0.0).all()
    assert (rule[rule["company_code"] == CONSOLIDATED]["mape"] == 0.0).all()


def test_scores_a_target_week_without_actual_is_left_out_everywhere():
    # (11, 2) -> week 13 has no actual: its forecast, whatever it is, changes no score.
    small = evaluation_scores(e_forecasts(unscored=999.0), e_actual(), E_HORIZONS)
    huge = evaluation_scores(e_forecasts(unscored=1e12), e_actual(), E_HORIZONS)
    pd.testing.assert_frame_equal(small, huge)
    assert not (small["forecast"] >= 999.0).any()


def test_scores_the_pooled_rows_count_only_the_rows_with_actual(e_scores):
    # 4 forecasts per method and company, 3 with an actual.
    assert set(e_scores.loc[e_scores["horizon"] == POOLED, "rows"]) == {3}


def test_scores_use_the_actual_column_not_actual_real(e_scores):
    # actual_real is -1.0 everywhere in s_actual; were it scored, every WAPE would be NaN.
    assert e_scores["wape"].notna().all()
    assert score_of(e_scores, HURDLE, CONSOLIDATED, POOLED)["actual"] == pytest.approx(500.0)


def test_scores_consolidated_is_scored_on_its_own_rows_not_the_sum_of_companies(e_scores):
    # 1. ALL's pooled forecast is ALL's own 110 + 150 + 140, not TR02's 190 plus anything.
    assert score_of(e_scores, HURDLE, CONSOLIDATED, POOLED)["forecast"] == pytest.approx(400.0)
    # 2. Changing TR02's forecasts leaves ALL's scores as they are.
    forecasts = e_forecasts()
    tr02 = (forecasts["method"] == HURDLE) & (forecasts["company_code"] == "TR02")
    forecasts.loc[tr02, "forecast"] = 1e6
    changed = evaluation_scores(forecasts, e_actual(), E_HORIZONS)
    consolidated_after = changed[changed["company_code"] == CONSOLIDATED].reset_index(drop=True)
    consolidated_before = e_scores[e_scores["company_code"] == CONSOLIDATED].reset_index(
        drop=True)
    pd.testing.assert_frame_equal(consolidated_after, consolidated_before)


def test_scores_pooled_equals_score_row_on_the_rows_with_actual(e_scores):
    # The pooled row is score_row on the three (origin, h) with data, nothing else.
    expected = score_row(HURDLE, CONSOLIDATED, POOLED, pd.Series([110.0, 150.0, 140.0]),
                         pd.Series([100.0, 200.0, 200.0]), True)
    found = score_of(e_scores, HURDLE, CONSOLIDATED, POOLED)
    for name, value in expected.items():
        if isinstance(value, float):
            assert found[name] == pytest.approx(value), name
        else:
            assert found[name] == value, name


def test_scores_horizon_labels_are_strings(e_scores):
    assert set(e_scores["horizon"]) == {"1", "2", POOLED, HORIZON_TOTAL}
    assert all(isinstance(h, str) for h in e_scores["horizon"])


def test_scores_order_of_first_appearance_not_sorted():
    # Methods "z" then "a"; companies ALL then TR02 for "z", TR02 then ALL for "a".
    actual = s_actual([("TR02", 10, 1, 11, 1.0), ("ALL", 10, 1, 11, 2.0)])
    forecasts = s_forecasts([
        ("z", "ALL", 10, 1, 11, 2.0), ("z", "TR02", 10, 1, 11, 1.0),
        ("a", "TR02", 10, 1, 11, 1.0), ("a", "ALL", 10, 1, 11, 2.0)])
    scores = evaluation_scores(forecasts, actual, 1)
    pairs_found = list(dict.fromkeys(zip(scores["method"], scores["company_code"])))
    assert pairs_found == [("z", "ALL"), ("z", "TR02"), ("a", "TR02"), ("a", "ALL")]


def test_scores_an_h_with_no_actual_has_no_row():
    # Horizons 2, but actual only at h = 1: no "2" row, and no origin is complete, so no
    # total row either.
    actual = s_actual([("ALL", 10, 1, 11, 100.0), ("ALL", 11, 1, 12, 100.0)])
    forecasts = s_forecasts([("m", "ALL", o, h, o + h, 90.0) for o in (10, 11) for h in (1, 2)])
    scores = evaluation_scores(forecasts, actual, 2)
    assert list(scores["horizon"]) == ["1", POOLED]
    assert score_of(scores, "m", "ALL", POOLED)["rows"] == 2


def test_scores_no_complete_origin_means_no_total_row():
    # Each origin misses one of its two target weeks.
    actual = s_actual([("ALL", 10, 1, 11, 100.0), ("ALL", 11, 2, 13, 100.0)])
    forecasts = s_forecasts([("m", "ALL", o, h, o + h, 90.0) for o in (10, 11) for h in (1, 2)])
    scores = evaluation_scores(forecasts, actual, 2)
    assert list(scores["horizon"]) == ["1", "2", POOLED]


def test_scores_total_sums_each_origin_before_scoring():
    # Origin 10: f 150 + 50 = 200 vs a 100 + 100 = 200: the errors (+50, -50) cancel in the
    # origin's total, so the total WAPE is 0 while the pooled WAPE is 100 / 200.
    actual = s_actual([("ALL", 10, 1, 11, 100.0), ("ALL", 10, 2, 12, 100.0)])
    forecasts = s_forecasts([("m", "ALL", 10, 1, 11, 150.0), ("m", "ALL", 10, 2, 12, 50.0)])
    scores = evaluation_scores(forecasts, actual, 2)
    assert score_of(scores, "m", "ALL", HORIZON_TOTAL)["wape"] == pytest.approx(0.0)
    assert score_of(scores, "m", "ALL", POOLED)["wape"] == pytest.approx(0.5)


def test_scores_total_rows_count_the_origins():
    # Three complete origins (10, 11, 12) and one incomplete (13 misses h = 2).
    actual = s_actual([("ALL", o, h, o + h, 10.0) for o in (10, 11, 12) for h in (1, 2)]
                      + [("ALL", 13, 1, 14, 10.0)])
    forecasts = s_forecasts([("m", "ALL", o, h, o + h, 12.0) for o in (10, 11, 12, 13)
                             for h in (1, 2)])
    scores = evaluation_scores(forecasts, actual, 2)
    # Each complete origin: f 24 vs a 20, |e| 4. Three of them: f 72, a 60, |e| 12.
    total = score_of(scores, "m", "ALL", HORIZON_TOTAL)
    assert_score(total, 3, 72.0, 60.0, 12.0, 0.2, 0.2, 4.0, 0.2)
    # Pooled: all 7 (origin, h) with data, origin 13's included.
    assert score_of(scores, "m", "ALL", POOLED)["rows"] == 7


def test_scores_with_one_horizon_pooled_and_total_equal_h1():
    actual = s_actual([("ALL", 10, 1, 11, 100.0), ("ALL", 11, 1, 12, 50.0)])
    forecasts = s_forecasts([("m", "ALL", 10, 1, 11, 80.0), ("m", "ALL", 11, 1, 12, 70.0)])
    scores = evaluation_scores(forecasts, actual, 1)
    assert list(scores["horizon"]) == ["1", POOLED, HORIZON_TOTAL]
    numbers = ["rows", "forecast", "actual", "abs_error", "wape", "bias", "mean_abs_error",
               "mape"]
    h1 = score_of(scores, "m", "ALL", "1")[numbers].astype(float)
    for horizon in (POOLED, HORIZON_TOTAL):
        found = score_of(scores, "m", "ALL", horizon)[numbers].astype(float)
        pd.testing.assert_series_equal(found, h1, check_names=False)


def test_scores_a_company_whose_actual_sums_to_zero_has_nan_wape_and_bias():
    actual = s_actual([("TR02", 10, 1, 11, 0.0), ("ALL", 10, 1, 11, 100.0)])
    forecasts = s_forecasts([("m", "TR02", 10, 1, 11, 5.0), ("m", "ALL", 10, 1, 11, 105.0)])
    scores = evaluation_scores(forecasts, actual, 1)
    tr02 = score_of(scores, "m", "TR02", POOLED)
    assert np.isnan(tr02["wape"]) and np.isnan(tr02["bias"])
    assert tr02["mean_abs_error"] == pytest.approx(5.0)
    assert score_of(scores, "m", "ALL", POOLED)["wape"] == pytest.approx(0.05)


def test_scores_a_consolidated_zero_actual_week_gives_nan_mape_but_a_wape():
    actual = s_actual([("ALL", 10, 1, 11, 0.0), ("ALL", 10, 2, 12, 100.0)])
    forecasts = s_forecasts([("m", "ALL", 10, 1, 11, 10.0), ("m", "ALL", 10, 2, 12, 110.0)])
    scores = evaluation_scores(forecasts, actual, 2)
    # 1. Pooled: an actual of 0 among the rows, so no MAPE; WAPE (10 + 10) / 100.
    pooled = score_of(scores, "m", "ALL", POOLED)
    assert np.isnan(pooled["mape"])
    assert pooled["wape"] == pytest.approx(0.2)
    # 2. h1 alone: actual sums to 0, so WAPE, bias and MAPE are all NaN.
    h1 = score_of(scores, "m", "ALL", "1")
    assert np.isnan(h1["wape"]) and np.isnan(h1["bias"]) and np.isnan(h1["mape"])
    # 3. The origin's total has actual 100 > 0, so its MAPE is defined: |120 - 100| / 100.
    assert score_of(scores, "m", "ALL", HORIZON_TOTAL)["mape"] == pytest.approx(0.2)


def test_scores_a_forecast_company_without_any_actual_has_no_rows():
    # TR09 is forecast, but the actual has no row for it: nothing to score, no refusal.
    actual = s_actual([("ALL", 10, 1, 11, 100.0)])
    forecasts = s_forecasts([("m", "ALL", 10, 1, 11, 90.0), ("m", "TR09", 10, 1, 11, 5.0)])
    scores = evaluation_scores(forecasts, actual, 1)
    assert set(scores["company_code"]) == {"ALL"}


def test_scores_with_and_without_the_hurdle_agree_on_the_other_methods(stack_inputs):
    # The pipeline's two branches: the references' and the rule's scores do not depend on
    # whether the hurdle model ran.
    rolled, reference, rule = stack_inputs
    actual = s_actual([(c, o, h, t, 1000.0) for c in ("TR02", "ALL")
                       for o, h, t in S_PAIRS[:3]])
    with_model = evaluation_scores(stack_forecasts(rolled, reference, rule), actual, 2)
    without = evaluation_scores(stack_forecasts(None, reference, rule), actual, 2)
    assert list(dict.fromkeys(with_model["method"])) == [HURDLE, *REFERENCES, RULE]
    assert list(dict.fromkeys(without["method"])) == [*REFERENCES, RULE]
    others = with_model[with_model["method"] != HURDLE].reset_index(drop=True)
    pd.testing.assert_frame_equal(others, without)


def test_scores_end_to_end_the_hurdle_is_scored_on_its_total(stack_inputs):
    # The hurdle's ALL totals at the three pairs with data are 5, 6, 7 (real_total -5 unused)
    # against an actual of 2 each: f 18, a 6, |e| 3 + 4 + 5 = 12: WAPE 2.0, bias 2.0,
    # mean |e| 4, MAPE (1.5 + 2.0 + 2.5) / 3 = 2.0.
    rolled, reference, rule = stack_inputs
    actual = s_actual([(c, o, h, t, 2.0) for c in ("TR02", "ALL") for o, h, t in S_PAIRS[:3]])
    scores = evaluation_scores(stack_forecasts(rolled, reference, rule), actual, 2)
    row = score_of(scores, HURDLE, CONSOLIDATED, POOLED)
    assert_score(row, 3, 18.0, 6.0, 12.0, 2.0, 2.0, 4.0, 2.0)


def test_scores_do_not_change_their_inputs():
    forecasts, actual = e_forecasts(), e_actual()
    before = (forecasts.copy(), actual.copy())
    evaluation_scores(forecasts, actual, E_HORIZONS)
    pd.testing.assert_frame_equal(forecasts, before[0])
    pd.testing.assert_frame_equal(actual, before[1])


def test_scores_are_deterministic():
    first = evaluation_scores(e_forecasts(), e_actual(), E_HORIZONS)
    second = evaluation_scores(e_forecasts(), e_actual(), E_HORIZONS)
    pd.testing.assert_frame_equal(first, second)


def test_scores_a_nan_forecast_is_never_scored_as_a_perfect_one():
    # A NaN forecast would vanish from the sums and score its week as perfect: refused, with
    # the method and keys of the first such row named.
    actual = s_actual([("ALL", 10, 1, 11, 2.0), ("ALL", 10, 2, 12, 4.0)])
    forecasts = s_forecasts([("m", "ALL", 10, 1, 11, 1.0), ("m", "ALL", 10, 2, 12, NAN)])
    with pytest.raises(ValueError) as error:
        evaluation_scores(forecasts, actual, 2)
    assert str(error.value) == (
        "1 forecasts are NaN or infinite (first: {'method': 'm', 'company_code': 'ALL', "
        "'origin_week': 10, 'h': 2, 'target_week': 12})")


# --- evaluation_scores: NaN or infinite forecasts (refused before the join) ---------------------

NONFINITE = [NAN, float("inf"), float("-inf")]
NONFINITE_IDS = ["nan", "inf", "-inf"]


def nonfinite_forecasts(value, method=RULE, company="ALL", origin=10, h=2):
    """The running example's forecasts with one row's forecast set to `value`."""
    forecasts = e_forecasts()
    forecasts.loc[is_row(forecasts, method, company, origin, h), "forecast"] = value
    return forecasts


@pytest.mark.parametrize("bad", NONFINITE, ids=NONFINITE_IDS)
def test_scores_a_nonfinite_forecast_exact_message(bad):
    # rule's ALL (10, 2) forecast is NaN, inf or -inf: refused, the row named.
    forecasts = nonfinite_forecasts(bad)
    with pytest.raises(ValueError) as error:
        evaluation_scores(forecasts, e_actual(), E_HORIZONS)
    assert str(error.value) == (
        "1 forecasts are NaN or infinite (first: {'method': 'rule', 'company_code': 'ALL', "
        "'origin_week': 10, 'h': 2, 'target_week': 12})")


@pytest.mark.parametrize("bad", NONFINITE, ids=NONFINITE_IDS)
def test_scores_a_nonfinite_forecast_in_a_week_without_actual_is_refused_too(bad):
    # (11, 2) aims at week 13, which has no actual cash: still refused (the check runs on
    # every forecast, before the join). Four such rows; hurdle's TR02 row comes first.
    forecasts = e_forecasts(unscored=bad)
    with pytest.raises(ValueError) as error:
        evaluation_scores(forecasts, e_actual(), E_HORIZONS)
    assert str(error.value) == (
        "4 forecasts are NaN or infinite (first: {'method': 'hurdle', 'company_code': 'TR02', "
        "'origin_week': 11, 'h': 2, 'target_week': 13})")


def test_scores_nonfinite_forecasts_are_counted_across_kinds_and_methods():
    # One NaN, one inf, one -inf in different methods: counted together, the first in the
    # table's order named (hurdle's rows come before rule's).
    forecasts = e_forecasts()
    forecasts.loc[is_row(forecasts, RULE, "TR02", 11, 1), "forecast"] = NAN
    forecasts.loc[is_row(forecasts, HURDLE, "ALL", 10, 1), "forecast"] = float("inf")
    forecasts.loc[is_row(forecasts, RULE, "ALL", 10, 2), "forecast"] = float("-inf")
    with pytest.raises(ValueError) as error:
        evaluation_scores(forecasts, e_actual(), E_HORIZONS)
    assert str(error.value) == (
        "3 forecasts are NaN or infinite (first: {'method': 'hurdle', 'company_code': 'ALL', "
        "'origin_week': 10, 'h': 1, 'target_week': 11})")


def test_scores_a_very_large_finite_forecast_is_scored_not_refused():
    # The edge just inside: 1e300 is finite, so it is scored (a huge WAPE, not a refusal).
    forecasts = nonfinite_forecasts(1e300, method=HURDLE)
    scores = evaluation_scores(forecasts, e_actual(), E_HORIZONS)
    h2 = score_of(scores, HURDLE, "ALL", "2")
    assert np.isfinite(h2["wape"]) and h2["wape"] > 1e290


def test_scores_a_nonfinite_forecast_does_not_change_the_inputs():
    # The refusal leaves the caller's tables as they were.
    forecasts, actual = nonfinite_forecasts(NAN), e_actual()
    before = (forecasts.copy(), actual.copy())
    with pytest.raises(ValueError):
        evaluation_scores(forecasts, actual, E_HORIZONS)
    pd.testing.assert_frame_equal(forecasts, before[0])
    pd.testing.assert_frame_equal(actual, before[1])


def test_scores_the_horizons_check_comes_before_the_nonfinite_check():
    with pytest.raises(ValueError, match=r"^horizons is 0; horizons start at 1$"):
        evaluation_scores(nonfinite_forecasts(NAN), e_actual(), 0)


def test_scores_the_h_check_comes_before_the_nonfinite_check():
    # An h outside 1 ... horizons whose forecast is also NaN: the h message.
    extra = s_forecasts([(HURDLE, "ALL", 12, 4, 16, NAN)])
    forecasts = pd.concat([e_forecasts(), extra], ignore_index=True)
    with pytest.raises(ValueError, match=r"^forecasts for h \[4\]; horizons is 2$"):
        evaluation_scores(forecasts, e_actual(), E_HORIZONS)


def test_scores_the_repeat_check_comes_before_the_nonfinite_check():
    # A repeated row and, elsewhere, a NaN forecast: the repeat message.
    forecasts = nonfinite_forecasts(NAN)
    repeated = pd.concat([forecasts, forecasts.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"^1 forecasts are there twice"):
        evaluation_scores(repeated, e_actual(), E_HORIZONS)


def test_scores_a_repeated_nan_row_is_reported_as_a_repeat():
    # The same NaN row twice: the repeat check runs first.
    forecasts = nonfinite_forecasts(NAN)
    repeated = pd.concat([forecasts, forecasts[forecasts["forecast"].isna()]],
                         ignore_index=True)
    with pytest.raises(ValueError, match=r"^1 forecasts are there twice"):
        evaluation_scores(repeated, e_actual(), E_HORIZONS)


def test_scores_the_nonfinite_check_comes_before_the_missing_check():
    # rule misses ALL (10, 1) and hurdle's ALL (10, 2) is inf: the NaN-or-infinite message.
    forecasts = nonfinite_forecasts(float("inf"), method=HURDLE)
    missing_one = forecasts[~is_row(forecasts, RULE, "ALL", 10, 1)]
    with pytest.raises(ValueError, match=r"^1 forecasts are NaN or infinite"):
        evaluation_scores(missing_one, e_actual(), E_HORIZONS)


# --- evaluation_scores: refusals ---------------------------------------------------------------

@pytest.mark.parametrize("horizons", [0, -1])
def test_scores_horizons_below_one_exact_message(horizons):
    with pytest.raises(ValueError) as error:
        evaluation_scores(e_forecasts(), e_actual(), horizons)
    assert str(error.value) == f"horizons is {horizons}; horizons start at 1"


def test_scores_horizons_check_comes_first():
    # Even with a repeated forecast, the horizons message comes first.
    forecasts = e_forecasts()
    forecasts = pd.concat([forecasts, forecasts.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"^horizons is 0; horizons start at 1$"):
        evaluation_scores(forecasts, e_actual(), 0)


@pytest.mark.parametrize("bad_h", [0, 3, 7])
def test_scores_a_forecast_h_outside_one_to_horizons_exact_message(bad_h):
    extra = s_forecasts([(HURDLE, "ALL", 12, bad_h, 12 + bad_h, 1.0)])
    forecasts = pd.concat([e_forecasts(), extra], ignore_index=True)
    with pytest.raises(ValueError) as error:
        evaluation_scores(forecasts, e_actual(), E_HORIZONS)
    assert str(error.value) == f"forecasts for h [{bad_h}]; horizons is 2"


def test_scores_several_h_outside_are_listed_sorted():
    extra = s_forecasts([(HURDLE, "ALL", 12, 5, 17, 1.0), (HURDLE, "ALL", 12, 0, 12, 1.0),
                         (RULE, "ALL", 12, 4, 16, 1.0)])
    forecasts = pd.concat([e_forecasts(), extra], ignore_index=True)
    with pytest.raises(ValueError) as error:
        evaluation_scores(forecasts, e_actual(), E_HORIZONS)
    assert str(error.value) == "forecasts for h [0, 4, 5]; horizons is 2"


def test_scores_an_h_outside_is_refused_even_without_actual_for_it():
    # The forecast at h = 3 aims at a week with no actual: still refused.
    extra = s_forecasts([(RULE, "ALL", 50, 3, 53, 1.0)])
    forecasts = pd.concat([e_forecasts(), extra], ignore_index=True)
    with pytest.raises(ValueError, match=r"^forecasts for h \[3\]; horizons is 2$"):
        evaluation_scores(forecasts, e_actual(), E_HORIZONS)


def test_scores_a_repeated_forecast_exact_message():
    forecasts = e_forecasts()
    repeated = pd.concat([forecasts, forecasts.iloc[[5]]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        evaluation_scores(repeated, e_actual(), E_HORIZONS)
    assert str(error.value) == (
        "1 forecasts are there twice (first: {'method': 'hurdle', 'company_code': 'ALL', "
        "'origin_week': 10, 'h': 2, 'target_week': 12})")


def test_scores_a_repeat_with_a_different_value_is_still_refused():
    forecasts = e_forecasts()
    again = forecasts.iloc[[0]].assign(forecast=-1.0)
    with pytest.raises(ValueError, match=r"^1 forecasts are there twice"):
        evaluation_scores(pd.concat([forecasts, again], ignore_index=True), e_actual(),
                          E_HORIZONS)


def test_scores_repeats_are_counted_and_the_first_named():
    forecasts = e_forecasts()
    repeated = pd.concat([forecasts, forecasts.iloc[[0, 1, 9]]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        evaluation_scores(repeated, e_actual(), E_HORIZONS)
    assert str(error.value) == (
        "3 forecasts are there twice (first: {'method': 'hurdle', 'company_code': 'TR02', "
        "'origin_week': 10, 'h': 1, 'target_week': 11})")


def test_scores_the_same_row_in_two_methods_is_not_a_repeat(e_scores):
    # Every hurdle row has a twin in rule with the same keys: allowed, 2 x 2 x 4 score rows.
    assert len(e_scores) == 16


def test_scores_a_repeat_in_a_week_without_actual_is_still_refused():
    forecasts = e_forecasts()
    unscored = forecasts[(forecasts["origin_week"] == 11) & (forecasts["h"] == 2)].iloc[[0]]
    with pytest.raises(ValueError, match=r"^1 forecasts are there twice"):
        evaluation_scores(pd.concat([forecasts, unscored], ignore_index=True), e_actual(),
                          E_HORIZONS)


def test_scores_the_h_check_comes_before_the_repeat_check():
    bad = s_forecasts([(HURDLE, "ALL", 12, 9, 21, 1.0)] * 2)
    forecasts = pd.concat([e_forecasts(), bad], ignore_index=True)
    with pytest.raises(ValueError, match=r"^forecasts for h \[9\]; horizons is 2$"):
        evaluation_scores(forecasts, e_actual(), E_HORIZONS)


def is_row(forecasts, method, company, origin, h):
    return ((forecasts["method"] == method) & (forecasts["company_code"] == company)
            & (forecasts["origin_week"] == origin) & (forecasts["h"] == h))


def test_scores_a_method_missing_a_forecast_for_a_row_with_actual_exact_message():
    # Drop rule's ALL (10, 2) forecast: week 12 has actual cash for ALL.
    forecasts = e_forecasts()
    dropped = forecasts[~is_row(forecasts, RULE, "ALL", 10, 2)]
    with pytest.raises(ValueError) as error:
        evaluation_scores(dropped, e_actual(), E_HORIZONS)
    assert str(error.value) == (
        "'rule' has no forecast for 1 rows with actual cash (first: {'company_code': 'ALL', "
        "'origin_week': 10, 'h': 2, 'target_week': 12})")


def test_scores_a_method_missing_a_whole_company_is_refused_with_the_count():
    forecasts = e_forecasts()
    dropped = forecasts[~((forecasts["method"] == HURDLE) & (forecasts["company_code"] == "TR02"))]
    with pytest.raises(ValueError) as error:
        evaluation_scores(dropped, e_actual(), E_HORIZONS)
    assert str(error.value) == (
        "'hurdle' has no forecast for 3 rows with actual cash (first: {'company_code': 'TR02', "
        "'origin_week': 10, 'h': 1, 'target_week': 11})")


def test_scores_a_forecast_for_the_wrong_target_week_counts_as_missing():
    # rule's ALL (10, 1) aims at week 99 instead of 11: the row with actual has no forecast.
    forecasts = e_forecasts()
    forecasts.loc[is_row(forecasts, RULE, "ALL", 10, 1), "target_week"] = 99
    with pytest.raises(ValueError, match=r"^'rule' has no forecast for 1 rows"):
        evaluation_scores(forecasts, e_actual(), E_HORIZONS)


def test_scores_a_missing_forecast_in_a_week_without_actual_is_fine(e_scores):
    # Dropping every forecast at (11, 2) (week 13, no actual) is not refused, same scores.
    forecasts = e_forecasts()
    kept = forecasts[~((forecasts["origin_week"] == 11) & (forecasts["h"] == 2))]
    pd.testing.assert_frame_equal(evaluation_scores(kept, e_actual(), E_HORIZONS), e_scores)


def test_scores_the_first_method_missing_a_row_is_named():
    # Both methods miss ALL (11, 1): hurdle comes first in the forecasts.
    forecasts = e_forecasts()
    gone = ((forecasts["company_code"] == "ALL") & (forecasts["origin_week"] == 11)
            & (forecasts["h"] == 1))
    with pytest.raises(ValueError, match=r"^'hurdle' has no forecast"):
        evaluation_scores(forecasts[~gone], e_actual(), E_HORIZONS)


def test_scores_the_repeat_check_comes_before_the_missing_check():
    # rule misses a row and hurdle has a repeat: the repeat is reported.
    forecasts = e_forecasts()
    missing_one = forecasts[~is_row(forecasts, RULE, "ALL", 10, 1)]
    broken = pd.concat([missing_one, forecasts.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"^1 forecasts are there twice"):
        evaluation_scores(broken, e_actual(), E_HORIZONS)


def test_scores_a_repeated_actual_row_is_refused():
    # Not one of the documented refusals, but a repeated actual row must not count a week
    # twice: the many-to-one merge refuses it (pandas' MergeError is a ValueError).
    actual = e_actual()
    repeated = pd.concat([actual, actual.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        evaluation_scores(e_forecasts(), repeated, E_HORIZONS)


# =============================================================================================
# The cache of step 17: `rolling_fingerprint` (everything `rolling_forecasts` makes its result
# from, as a fingerprint for `cache.load_with_cache`) and `MODEL_CODE` (the source files whose
# hash it carries). Run with the real Settings of settings/synthetic.toml, changed one field
# at a time with dataclasses.replace; the tables are small hand-made ones (only their hash
# goes in, so their content does not need to make sense to the model).
# =============================================================================================

import ast as fp_ast  # noqa: E402
import inspect as fp_inspect  # noqa: E402
import json as fp_json  # noqa: E402
import shutil as fp_shutil  # noqa: E402
from dataclasses import asdict as fp_asdict, replace as fp_replace  # noqa: E402
from pathlib import Path as FpPath  # noqa: E402

from collection_estimation.hurdle.cache import (  # noqa: E402
    load_with_cache, source_fingerprint, table_fingerprint)
from collection_estimation.hurdle.evaluate import MODEL_CODE, rolling_fingerprint  # noqa: E402
from collection_estimation.hurdle.settings import load_settings  # noqa: E402

FP_REPO_ROOT = FpPath(__file__).resolve().parents[2]
FP_SETTINGS_FILE = FP_REPO_ROOT / "settings" / "synthetic.toml"
FP_HURDLE_FOLDER = FpPath(evaluate_module.__file__).parent
FP_TABLE_NAMES = ["table", "new_cash", "refunds", "unidentified", "levels", "calendar"]


@pytest.fixture(scope="module")
def fp_settings():
    if not FP_SETTINGS_FILE.exists():
        pytest.skip("settings/synthetic.toml not present")
    return load_settings(FP_SETTINGS_FILE, FP_REPO_ROOT)


def fp_table():
    """A tiny stand-in training table: two customers, origins 20 and 21, h 1 and 2."""
    rows = [{"key": f"K{c}", "company_code": "TR02", "origin_week": origin, "h": h,
             "target_week": origin + h, "paid": float((c + h) % 2), "amount": 10.0 * c + h}
            for c in (1, 2) for origin in (20, 21) for h in (1, 2)]
    return pd.DataFrame(rows)


def fp_arguments(settings):
    """Every argument of rolling_fingerprint, fresh each call: (table, origins, settings,
    the keyword arguments)."""
    new_cash, refunds, unidentified = f_side_tables(last=30)
    keywords = dict(block_columns={"history": ["mean_receipts_4"], "calendar": ["cycle"]},
                    companies=["TR02", "TR04"], new_cash=new_cash, refunds=refunds,
                    unidentified=unidentified, levels=f_levels(), calendar=f_calendar(),
                    week_zero=WEEK_ZERO)
    return fp_table(), [20, 21], settings, keywords


def fp_of(table, origins, settings, keywords):
    return rolling_fingerprint(table, origins, settings, **keywords)


def fp_with(settings, section, **changes):
    """`settings` with one section's fields changed (Settings and its sections are frozen)."""
    changed_section = fp_replace(getattr(settings, section), **changes)
    return fp_replace(settings, **{section: changed_section})


# --- shape and encoding ----------------------------------------------------------------------

def test_fingerprint_is_json_serialisable_with_the_real_settings(fp_settings):
    # load_with_cache encodes it as JSON before anything else: it must encode, and come back
    # equal (after the JSON round trip, tuples become lists).
    fingerprint = fp_of(*fp_arguments(fp_settings))
    text = fp_json.dumps(fingerprint, indent=1)
    assert fp_json.loads(text) == fp_json.loads(fp_json.dumps(fingerprint))
    assert isinstance(fingerprint, list)


def test_fingerprint_items_in_order(fp_settings):
    # 1. Build it, and each item by hand as the docstring says.
    table, origins, settings, keywords = fp_arguments(fp_settings)
    fingerprint = fp_of(table, origins, settings, keywords)
    topups = settings.topups
    expected = [
        [20, 21],
        fp_asdict(settings.receivables),
        fp_asdict(settings.stage1),
        fp_asdict(settings.stage2),
        [topups.method, topups.window, topups.with_level],
        keywords["block_columns"],
        ["TR02", "TR04"],
        WEEK_ZERO.isoformat(),
        table_fingerprint(table),
        table_fingerprint(keywords["new_cash"]),
        table_fingerprint(keywords["refunds"]),
        table_fingerprint(keywords["unidentified"]),
        table_fingerprint(keywords["levels"]),
        table_fingerprint(keywords["calendar"]),
        source_fingerprint(*(FP_HURDLE_FOLDER / name for name in MODEL_CODE)),
    ]
    # 2. Item by item, so a failure names the item.
    assert len(fingerprint) == len(expected)
    for index, (found, wanted) in enumerate(zip(fingerprint, expected)):
        assert found == wanted, index


def test_fingerprint_is_deterministic_and_equal_for_fresh_copies_of_the_inputs(fp_settings):
    # Two builds of the same inputs (fresh tables each time) give the same fingerprint.
    first = fp_of(*fp_arguments(fp_settings))
    second = fp_of(*fp_arguments(fp_settings))
    assert first == second


def test_fingerprint_with_the_real_settings_hits_the_cache_on_the_second_call(fp_settings,
                                                                               tmp_path):
    # 1. The first call fits (here: a stand-in result) and writes the cache.
    fitted = Forecast(totals=pd.DataFrame({"total": [1.5, 2.5]}),
                      customers=pd.DataFrame({"key": ["K1"]}))
    calls = []

    def fit():
        calls.append(1)
        return fitted

    load_with_cache("hurdle_rolling_forecasts", fp_of(*fp_arguments(fp_settings)), fit,
                    tmp_path)
    # 2. The second call, with a fingerprint built afresh from the same inputs, reads it.
    again = load_with_cache("hurdle_rolling_forecasts", fp_of(*fp_arguments(fp_settings)), fit,
                            tmp_path)
    assert calls == [1]
    assert isinstance(again, Forecast)
    pd.testing.assert_frame_equal(again.totals, fitted.totals)
    pd.testing.assert_frame_equal(again.customers, fitted.customers)


def test_fingerprint_does_not_change_its_inputs(fp_settings):
    table, origins, settings, keywords = fp_arguments(fp_settings)
    before_table, before_origins = table.copy(), list(origins)
    before_blocks = {name: list(columns) for name, columns in keywords["block_columns"].items()}
    before_companies = list(keywords["companies"])
    fp_of(table, origins, settings, keywords)
    pd.testing.assert_frame_equal(table, before_table)
    assert origins == before_origins
    assert keywords["block_columns"] == before_blocks
    assert keywords["companies"] == before_companies


def test_fingerprint_takes_rolling_forecasts_arguments_without_the_callback():
    # Step 17 passes both the same arguments; only on_origin_done is the fit's alone.
    fit_parameters = dict(fp_inspect.signature(rolling_forecasts).parameters)
    del fit_parameters["on_origin_done"]
    print_parameters = dict(fp_inspect.signature(rolling_fingerprint).parameters)
    assert list(print_parameters) == list(fit_parameters)
    for name, parameter in print_parameters.items():
        assert parameter.kind == fit_parameters[name].kind, name


# --- what changes it: one input at a time ----------------------------------------------------

def fp_change_table_value(table, origins, settings, keywords):
    table.loc[0, "amount"] += 0.01
    return table, origins, settings, keywords


def fp_change_table_index(table, origins, settings, keywords):
    return table.set_axis(table.index + 100), origins, settings, keywords


def fp_change_table_column_name(table, origins, settings, keywords):
    return table.rename(columns={"amount": "amount_nominal"}), origins, settings, keywords


def fp_change_one_keyword_table(name):
    """Change the first float of one keyword table (or series) by 0.001."""
    def change(table, origins, settings, keywords):
        changed = keywords[name].copy()
        if isinstance(changed, pd.Series):
            changed.iloc[0] += 0.001
        else:
            column = [c for c in changed.columns if changed[c].dtype == "float64"][0]
            changed.loc[changed.index[0], column] += 0.001
        return table, origins, settings, {**keywords, name: changed}
    return change


def fp_change_origins(table, origins, settings, keywords):
    return table, origins + [22], settings, keywords


def fp_change_settings_by(section, field, how):
    """Change one field relative to its current value (so the test holds whatever the toml)."""
    def change(table, origins, settings, keywords):
        value = getattr(getattr(settings, section), field)
        return table, origins, fp_with(settings, section, **{field: how(value)}), keywords
    return change


def fp_change_keyword(name, how):
    def change(table, origins, settings, keywords):
        return table, origins, settings, {**keywords, name: how(keywords[name])}
    return change


FP_CHANGES = {
    # The origins.
    "an origin added": fp_change_origins,
    # [receivables], [stage1], [stage2] and the method of [topups].
    "receivables horizons": fp_change_settings_by("receivables", "horizons", lambda v: v + 1),
    "receivables min_cleared_for_delay": fp_change_settings_by(
        "receivables", "min_cleared_for_delay", lambda v: v + 1),
    "receivables overdue_age_days": fp_change_settings_by(
        "receivables", "overdue_age_days", lambda v: tuple(v) + (999,)),
    "stage1 learning_rate": fp_change_settings_by("stage1", "learning_rate", lambda v: v / 2),
    "stage1 max_iter": fp_change_settings_by("stage1", "max_iter", lambda v: v + 1),
    "stage1 random_seed": fp_change_settings_by("stage1", "random_seed", lambda v: v + 1),
    "stage1 company_as_feature": fp_change_settings_by("stage1", "company_as_feature",
                                                       lambda v: not v),
    "stage1 feature_blocks": fp_change_settings_by("stage1", "feature_blocks",
                                                   lambda v: tuple(v)[:-1]),
    "stage2 loss": fp_change_settings_by(
        "stage2", "loss", lambda v: "gamma" if v != "gamma" else "poisson"),
    "stage2 learning_rate": fp_change_settings_by("stage2", "learning_rate", lambda v: v / 2),
    "stage2 own_mean_prior_weeks": fp_change_settings_by("stage2", "own_mean_prior_weeks",
                                                         lambda v: v + 0.5),
    "topups method": fp_change_settings_by(
        "topups", "method", lambda v: "ridge" if v != "ridge" else "mean"),
    "topups window": fp_change_settings_by("topups", "window", lambda v: v + 1),
    "topups with_level": fp_change_settings_by("topups", "with_level", lambda v: not v),
    # The other arguments.
    "a block column added": fp_change_keyword(
        "block_columns", lambda v: {**v, "history": v["history"] + ["paying_weeks_4"]}),
    "a block removed": fp_change_keyword(
        "block_columns", lambda v: {"history": v["history"]}),
    "a company added": fp_change_keyword("companies", lambda v: v + ["TR05"]),
    "week zero a week later": fp_change_keyword("week_zero", lambda v: v + timedelta(days=7)),
    # The tables.
    "a table value": fp_change_table_value,
    "the table's index": fp_change_table_index,
    "a table column name": fp_change_table_column_name,
    **{f"a {name} value": fp_change_one_keyword_table(name) for name in FP_TABLE_NAMES[1:]},
}


@pytest.mark.parametrize("change", list(FP_CHANGES.values()), ids=list(FP_CHANGES))
def test_fingerprint_changes_when_an_input_changes(fp_settings, change):
    # 1. The fingerprint of the inputs as they are.
    before = fp_of(*fp_arguments(fp_settings))
    # 2. One input changed: the fingerprint is different (after the JSON round trip too, as
    #    load_with_cache compares it).
    after = fp_of(*change(*fp_arguments(fp_settings)))
    assert after != before
    assert fp_json.loads(fp_json.dumps(after)) != fp_json.loads(fp_json.dumps(before))


def test_fingerprint_changes_when_a_table_dtype_changes(fp_settings):
    # The same numbers as float instead of int: a different fingerprint (dtypes count).
    table, origins, settings, keywords = fp_arguments(fp_settings)
    before = fp_of(table, origins, settings, keywords)
    changed = table.astype({"origin_week": "float64"})
    assert fp_of(changed, origins, settings, keywords) != before


def test_fingerprint_changes_when_the_table_rows_are_reordered(fp_settings):
    table, origins, settings, keywords = fp_arguments(fp_settings)
    before = fp_of(table, origins, settings, keywords)
    reordered = table.iloc[::-1]
    assert fp_of(reordered, origins, settings, keywords) != before


# --- what does not change it -----------------------------------------------------------------

FP_NO_CHANGES = {
    # [diagnostic] and [joint_diagnostic] never change a forecast.
    "diagnostic run": fp_change_settings_by("diagnostic", "run", lambda v: not v),
    "diagnostic origin": fp_change_settings_by("diagnostic", "origin", lambda v: v + 1),
    "diagnostic hgb_max_iter": fp_change_settings_by("diagnostic", "hgb_max_iter",
                                                     lambda v: tuple(v) + (7,)),
    "joint_diagnostic run": fp_change_settings_by("joint_diagnostic", "run", lambda v: not v),
    # [evaluation]: which origins come from the argument, not from the settings.
    "evaluation run": fp_change_settings_by("evaluation", "run", lambda v: not v),
    "evaluation with_model": fp_change_settings_by("evaluation", "with_model",
                                                   lambda v: not v),
    "evaluation validation_origins": fp_change_settings_by(
        "evaluation", "validation_origins", lambda v: tuple(v) + (99,)),
    "evaluation rule_doubtful_days": fp_change_settings_by(
        "evaluation", "rule_doubtful_days", lambda v: v + 1),
    # U14.5c: the paired bootstrap scores the forecasts; it never makes one.
    "evaluation bootstrap_samples": fp_change_settings_by(
        "evaluation", "bootstrap_samples", lambda v: v + 1),
    "evaluation bootstrap_seed": fp_change_settings_by(
        "evaluation", "bootstrap_seed", lambda v: v + 1),
    # U14.6: the balance check is a report; it never makes a forecast.
    "evaluation coherence_factor": fp_change_settings_by(
        "evaluation", "coherence_factor", lambda v: v + 1),
    # [topups]: only the comparison's fields.
    "topups compare": fp_change_settings_by("topups", "compare", lambda v: not v),
    "topups compare_first_origin": fp_change_settings_by(
        "topups", "compare_first_origin", lambda v: v - 1),
    "topups compare_last_origin": fp_change_settings_by(
        "topups", "compare_last_origin", lambda v: v - 1),
    "topups compare_windows": fp_change_settings_by(
        "topups", "compare_windows", lambda v: tuple(v) + (26,)),
    # Printing, and where the cache goes.
    "debug demo_origin": fp_change_settings_by("debug", "demo_origin", lambda v: v + 1),
    "data cache_folder": fp_change_settings_by(
        "data", "cache_folder", lambda v: FpPath(v) / "elsewhere"),
    # The origins' order and repeats; the companies' order.
    "origins reversed": lambda t, o, s, k: (t, list(reversed(o)), s, k),
    "origins repeated": lambda t, o, s, k: (t, o + o + [o[0]], s, k),
    "origins as a tuple": lambda t, o, s, k: (t, tuple(o), s, k),
    "companies reversed": fp_change_keyword("companies", lambda v: list(reversed(v))),
    # A copy of each table.
    "copies of the tables": lambda t, o, s, k: (
        t.copy(), o, s, {**k, **{n: k[n].copy() for n in FP_TABLE_NAMES[1:]}}),
}


@pytest.mark.parametrize("change", list(FP_NO_CHANGES.values()), ids=list(FP_NO_CHANGES))
def test_fingerprint_does_not_change_for_what_no_forecast_reads(fp_settings, change):
    before = fp_of(*fp_arguments(fp_settings))
    after = fp_of(*change(*fp_arguments(fp_settings)))
    assert after == before


# --- MODEL_CODE: the source files it hashes --------------------------------------------------

def test_model_code_files_all_exist_beside_evaluate():
    for name in MODEL_CODE:
        assert (FP_HURDLE_FOLDER / name).is_file(), name


def test_model_code_is_a_tuple_of_distinct_python_file_names_without_evaluate():
    assert isinstance(MODEL_CODE, tuple)
    assert len(set(MODEL_CODE)) == len(MODEL_CODE)
    assert all(name.endswith(".py") and "/" not in name and "\\" not in name
               for name in MODEL_CODE)
    assert "evaluate.py" not in MODEL_CODE


def test_model_code_holds_the_modules_one_forecast_runs():
    # forecast() and the stages, top-ups and price levels it calls.
    for name in ("forecast.py", "stage1.py", "stage2.py", "topups.py", "prices.py",
                 "training_table.py", "benchmark.py"):
        assert name in MODEL_CODE, name


def fp_hurdle_imports(file_name):
    """The hurdle modules `file_name` imports (`from .x import`, `from . import x` or
    `from collection_estimation.hurdle.x import`, anywhere in the file), as file names."""
    tree = fp_ast.parse((FP_HURDLE_FOLDER / file_name).read_text(encoding="utf-8"))
    found = set()
    for node in fp_ast.walk(tree):
        if not isinstance(node, fp_ast.ImportFrom):
            continue
        if node.level == 1 and node.module:
            found.add(node.module.split(".")[0] + ".py")
        elif node.level == 1:
            found.update(alias.name + ".py" for alias in node.names)
        elif node.module and node.module.startswith("collection_estimation.hurdle."):
            found.add(node.module.split(".")[2] + ".py")
    return found


def test_model_code_covers_every_hurdle_module_one_forecast_imports():
    # A change to any module forecast.py reaches (by import, transitively) can change a
    # forecast; if its file is not hashed, the cache would serve forecasts made by the old
    # code. Walk the imports from forecast.py.
    reached, waiting = set(), ["forecast.py"]
    while waiting:
        name = waiting.pop()
        if name in reached:
            continue
        reached.add(name)
        waiting.extend(fp_hurdle_imports(name) - reached)
    missing = sorted(reached - set(MODEL_CODE))
    assert missing == [], f"forecast.py reaches {missing}, which MODEL_CODE does not hash"


@pytest.fixture
def fp_code_copy(tmp_path, monkeypatch):
    """The MODEL_CODE files (and evaluate.py) copied to tmp_path, and evaluate's __file__
    pointed there, so editing a copy is like editing the code."""
    for name in MODEL_CODE + ("evaluate.py",):
        fp_shutil.copyfile(FP_HURDLE_FOLDER / name, tmp_path / name)
    monkeypatch.setattr(evaluate_module, "__file__", str(tmp_path / "evaluate.py"))
    return tmp_path


@pytest.mark.parametrize("name", MODEL_CODE)
def test_fingerprint_changes_when_a_model_code_file_is_edited(fp_settings, fp_code_copy, name):
    # 1. With the copies as they are, the fingerprint as before the edit.
    before = fp_of(*fp_arguments(fp_settings))
    # 2. One copied file gets one more line: the fingerprint changes, in the code's item only.
    with open(fp_code_copy / name, "a", encoding="utf-8") as edited:
        edited.write("# edited\n")
    after = fp_of(*fp_arguments(fp_settings))
    assert after != before
    assert after[:-1] == before[:-1]


def test_fingerprint_with_the_copied_code_equals_the_real_one(fp_settings, fp_code_copy):
    # Identical bytes in another folder: the same fingerprint (a hash of content, not paths).
    copied = fp_of(*fp_arguments(fp_settings))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluate_module, "__file__", str(FP_HURDLE_FOLDER / "evaluate.py"))
        real = fp_of(*fp_arguments(fp_settings))
    assert copied == real


def test_fingerprint_does_not_change_when_evaluate_itself_is_edited(fp_settings, fp_code_copy):
    # evaluate.py is left out on purpose (it changes often).
    before = fp_of(*fp_arguments(fp_settings))
    with open(fp_code_copy / "evaluate.py", "a", encoding="utf-8") as edited:
        edited.write("# edited\n")
    assert fp_of(*fp_arguments(fp_settings)) == before


def test_fingerprint_fails_if_a_model_code_file_is_missing(fp_settings, fp_code_copy):
    # A missing file is an error, not a silently shorter fingerprint.
    (fp_code_copy / MODEL_CODE[0]).unlink()
    with pytest.raises(FileNotFoundError):
        fp_of(*fp_arguments(fp_settings))


# =============================================================================================
# U14.5c: `scored_rows` (the checks and the join `evaluation_scores` used to do itself), and the
# paired bootstrap of WAPE: `weighted_wape`, `drawn_counts`, `resampled_weeks`,
# `resampled_origins` (circular blocks), `bootstrap_row` and `paired_wape_bootstrap`.
#
# The tables are built with s_forecasts / s_actual above. Where an interval is checked
# exactly, the expected resamples are rebuilt here from the documented recipe (the same
# default_rng(seed), the same draws), independently of the code under test.
# =============================================================================================

from collection_estimation.hurdle.evaluate import (  # noqa: E402
    BOOTSTRAP_COLUMNS, bootstrap_row, drawn_counts, paired_wape_bootstrap, resampled_origins,
    resampled_weeks, scored_rows, weighted_wape)


# --- scored_rows -------------------------------------------------------------------------------

def test_scored_rows_columns_are_the_forecasts_plus_actual():
    rows = scored_rows(e_forecasts(), e_actual(), E_HORIZONS)
    assert list(rows.columns) == STACK_COLUMNS + ["actual"]
    # actual_real (a decoy in s_actual) is not carried over.
    assert "actual_real" not in rows.columns


def test_scored_rows_is_an_inner_join_on_the_rows_with_actual():
    # 2 methods x 2 companies x 3 (origin, h) with actual; (11, 2) -> week 13 has none.
    rows = scored_rows(e_forecasts(), e_actual(), E_HORIZONS)
    assert len(rows) == 12
    assert not ((rows["origin_week"] == 11) & (rows["h"] == 2)).any()
    assert not (rows["forecast"] == 999.0).any()


def test_scored_rows_puts_each_forecast_beside_the_actual_of_its_own_row():
    rows = scored_rows(e_forecasts(), e_actual(), E_HORIZONS)
    hurdle_all = rows[(rows["method"] == HURDLE) & (rows["company_code"] == "ALL")]
    by_key = {(o, h): (f, a) for o, h, f, a in zip(hurdle_all["origin_week"], hurdle_all["h"],
                                                   hurdle_all["forecast"], hurdle_all["actual"])}
    assert by_key == {(10, 1): (110.0, 100.0), (10, 2): (150.0, 200.0), (11, 1): (140.0, 200.0)}


def test_scored_rows_an_actual_row_for_a_company_nobody_forecasts_is_refused():
    # An actual row for a company nobody forecasts is checked: every method must have it.
    actual = pd.concat([e_actual(), s_actual([("TR09", 10, 1, 11, 5.0)])], ignore_index=True)
    with pytest.raises(ValueError, match=r"^'hurdle' has no forecast for 1 rows"):
        scored_rows(e_forecasts(), actual, E_HORIZONS)


@pytest.mark.parametrize("horizons", [0, -3])
def test_scored_rows_refuses_horizons_below_one_with_the_exact_message(horizons):
    with pytest.raises(ValueError) as error:
        scored_rows(e_forecasts(), e_actual(), horizons)
    assert str(error.value) == f"horizons is {horizons}; horizons start at 1"


def test_scored_rows_refuses_an_h_outside_with_the_exact_message():
    extra = s_forecasts([(HURDLE, "ALL", 12, 3, 15, 1.0)])
    with pytest.raises(ValueError) as error:
        scored_rows(pd.concat([e_forecasts(), extra], ignore_index=True), e_actual(), E_HORIZONS)
    assert str(error.value) == "forecasts for h [3]; horizons is 2"


def test_scored_rows_refuses_a_repeat_a_nonfinite_and_a_missing_forecast():
    # 1. A repeat.
    forecasts = e_forecasts()
    with pytest.raises(ValueError, match=r"^1 forecasts are there twice"):
        scored_rows(pd.concat([forecasts, forecasts.iloc[[0]]], ignore_index=True), e_actual(),
                    E_HORIZONS)
    # 2. A NaN.
    with pytest.raises(ValueError, match=r"^1 forecasts are NaN or infinite"):
        scored_rows(nonfinite_forecasts(NAN), e_actual(), E_HORIZONS)
    # 3. A missing forecast for a row with actual.
    dropped = forecasts[~is_row(forecasts, RULE, "ALL", 10, 2)]
    with pytest.raises(ValueError, match=r"^'rule' has no forecast for 1 rows"):
        scored_rows(dropped, e_actual(), E_HORIZONS)


def test_scored_rows_and_evaluation_scores_refuse_with_the_same_message():
    # evaluation_scores delegates its checks: the very same message for the same input.
    broken = nonfinite_forecasts(float("inf"))
    with pytest.raises(ValueError) as from_rows:
        scored_rows(broken, e_actual(), E_HORIZONS)
    with pytest.raises(ValueError) as from_scores:
        evaluation_scores(broken, e_actual(), E_HORIZONS)
    assert str(from_rows.value) == str(from_scores.value)


def test_scored_rows_does_not_change_its_inputs():
    forecasts, actual = e_forecasts(), e_actual()
    before = (forecasts.copy(), actual.copy())
    scored_rows(forecasts, actual, E_HORIZONS)
    pd.testing.assert_frame_equal(forecasts, before[0])
    pd.testing.assert_frame_equal(actual, before[1])


# --- weighted_wape -------------------------------------------------------------------------------

def test_weighted_wape_with_unit_weights_is_the_plain_wape():
    # errors 10, 50, 60 over actual 100, 200, 200: 120 / 500.
    found = weighted_wape(np.array([10.0, 50.0, 60.0]), np.array([100.0, 200.0, 200.0]),
                          np.ones(3))
    assert found == pytest.approx(0.24)
    assert type(found) is float


def test_weighted_wape_by_hand_with_weights():
    # weights 2, 0, 1: (2*10 + 0*50 + 1*60) / (2*100 + 0 + 1*200) = 80 / 400.
    found = weighted_wape(np.array([10.0, 50.0, 60.0]), np.array([100.0, 200.0, 200.0]),
                          np.array([2.0, 0.0, 1.0]))
    assert found == pytest.approx(0.2)


def test_weighted_wape_with_whole_weights_equals_repeating_the_rows():
    errors, actual = np.array([3.0, 7.0, 1.0]), np.array([10.0, 20.0, 5.0])
    weights = np.array([3.0, 1.0, 2.0])
    repeated_errors = np.repeat(errors, weights.astype(int))
    repeated_actual = np.repeat(actual, weights.astype(int))
    assert weighted_wape(errors, actual, weights) == pytest.approx(
        weighted_wape(repeated_errors, repeated_actual, np.ones(len(repeated_actual))))


def test_weighted_wape_does_not_depend_on_the_scale_of_the_weights():
    errors, actual = np.array([3.0, 7.0]), np.array([10.0, 20.0])
    assert weighted_wape(errors, actual, np.array([1.0, 2.0])) == pytest.approx(
        weighted_wape(errors, actual, np.array([5.0, 10.0])))


# --- drawn_counts --------------------------------------------------------------------------------

def test_drawn_counts_by_hand():
    found = drawn_counts(np.array([3, 5, 7]), [5, 5, 3])
    assert list(found) == [1.0, 2.0, 0.0]
    assert found.dtype == np.float64


def test_drawn_counts_repeated_values_each_get_the_count():
    # Two rows of week 5 (two origins aim at it): both count as often as week 5 was drawn.
    found = drawn_counts(np.array([5, 5, 3, 9]), [5, 5, 5, 9])
    assert list(found) == [3.0, 3.0, 0.0, 1.0]


def test_drawn_counts_nothing_drawn_gives_zeros():
    assert list(drawn_counts(np.array([1, 2]), [])) == [0.0, 0.0]


def test_drawn_counts_ignores_drawn_values_not_among_the_values():
    assert list(drawn_counts(np.array([1, 2]), [2, 8, 8])) == [0.0, 1.0]


def test_drawn_counts_sum_equals_the_number_drawn_when_values_are_distinct():
    values = np.array([10, 11, 12, 13])
    drawn = [12, 10, 12, 13, 13, 13]
    assert drawn_counts(values, drawn).sum() == len(drawn)


# --- resampled_weeks -----------------------------------------------------------------------------

def test_resampled_weeks_is_a_list_as_long_as_the_weeks_drawn_from_them():
    weeks = [11, 12, 13, 14, 15]
    drawn = resampled_weeks(weeks, np.random.default_rng(3))
    assert isinstance(drawn, list)
    assert len(drawn) == len(weeks)
    assert set(drawn) <= set(weeks)


def test_resampled_weeks_is_rng_choice_with_replacement():
    weeks = [11, 12, 13, 14, 15]
    expected = np.random.default_rng(42).choice(weeks, size=5, replace=True)
    assert resampled_weeks(weeks, np.random.default_rng(42)) == list(expected)


def test_resampled_weeks_draws_with_replacement():
    # Over many draws of 5 weeks, some resample repeats a week (without replacement none would).
    rng = np.random.default_rng(0)
    assert any(len(set(resampled_weeks([1, 2, 3, 4, 5], rng))) < 5 for _ in range(50))


def test_resampled_weeks_of_one_week_is_that_week():
    assert resampled_weeks([7], np.random.default_rng(1)) == [7]


# --- resampled_origins: circular blocks ----------------------------------------------------------

def test_resampled_origins_length_and_members():
    origins = [33, 34, 35, 36, 37, 38, 39]
    rng = np.random.default_rng(5)
    for _ in range(200):
        drawn = resampled_origins(origins, 3, rng)
        assert len(drawn) == len(origins)
        assert set(drawn) <= set(origins)


def test_resampled_origins_by_hand_from_the_starts():
    # 1. The starts the same rng gives (one per block): blocks of 2 out of 5 origins need 3
    #    blocks (6 drawn, cut to 5).
    origins = [10, 11, 12, 13, 14]
    starter = np.random.default_rng(9)
    starts = [int(starter.integers(0, 5)) for _ in range(3)]
    # 2. Each block runs on from its start, wrapping past the last origin.
    expected = [origins[(start + step) % 5] for start in starts for step in range(2)][:5]
    assert resampled_origins(origins, 2, np.random.default_rng(9)) == expected


def test_resampled_origins_a_block_past_the_end_wraps_to_the_first():
    # Find a seed whose first start is the last origin: its block of 3 is last, first, second.
    origins = [20, 21, 22, 23]
    for seed in range(1000):
        if int(np.random.default_rng(seed).integers(0, 4)) == 3:
            break
    drawn = resampled_origins(origins, 3, np.random.default_rng(seed))
    assert drawn[:3] == [23, 20, 21]


def test_resampled_origins_with_block_one_is_a_plain_resample():
    origins = [1, 2, 3, 4, 5, 6]
    starter = np.random.default_rng(17)
    expected = [origins[int(starter.integers(0, 6))] for _ in range(6)]
    assert resampled_origins(origins, 1, np.random.default_rng(17)) == expected


def test_resampled_origins_with_block_equal_to_the_count_is_a_rotation():
    # One block covers every origin once, starting anywhere: each origin exactly once.
    origins = [40, 41, 42, 43, 44]
    rng = np.random.default_rng(2)
    for _ in range(50):
        drawn = resampled_origins(origins, 5, rng)
        assert sorted(drawn) == origins
        start = origins.index(drawn[0])
        assert drawn == origins[start:] + origins[:start]


def test_resampled_origins_consecutive_draws_within_a_block_are_neighbours():
    # Inside each block of 3, each origin is the next one (circularly) after the previous.
    origins = [10, 11, 12, 13, 14, 15, 16]
    rng = np.random.default_rng(8)
    for _ in range(100):
        drawn = resampled_origins(origins, 3, rng)
        for block_start in range(0, len(drawn), 3):
            block = drawn[block_start:block_start + 3]
            for before, after in zip(block, block[1:]):
                assert origins.index(after) == (origins.index(before) + 1) % len(origins)


@pytest.mark.parametrize("count, block", [(6, 3), (7, 3), (12, 5), (4, 4), (5, 2)])
def test_resampled_origins_every_origin_is_drawn_equally_often_in_the_long_run(count, block):
    # The circular block bootstrap draws every origin, the first and last included, as often
    # as the middle ones (a non-circular version under-draws the ends).
    origins = list(range(30, 30 + count))
    rng = np.random.default_rng(20261104)
    tally = dict.fromkeys(origins, 0)
    resamples = 20000
    for _ in range(resamples):
        for origin in resampled_origins(origins, block, rng):
            tally[origin] += 1
    expected = resamples          # n draws per resample over n origins: once each on average
    for origin, seen in tally.items():
        assert abs(seen / expected - 1) < 0.03, (origin, seen, expected)
    # The ends against the middle, explicitly.
    middle = tally[origins[count // 2]]
    assert abs(tally[origins[0]] / middle - 1) < 0.05
    assert abs(tally[origins[-1]] / middle - 1) < 0.05


def test_resampled_origins_of_one_origin_is_that_origin():
    assert resampled_origins([50], 1, np.random.default_rng(0)) == [50]


# --- bootstrap_row -------------------------------------------------------------------------------

def bs_cycling_weights(vectors):
    """A draw_weights that ignores the rng and hands out `vectors` in turn (so the expected
    differences can be worked out by hand)."""
    state = {"next": 0}

    def draw(rng):
        found = np.asarray(vectors[state["next"] % len(vectors)], dtype="float64")
        state["next"] += 1
        return found
    return draw


def test_bootstrap_row_keys_are_the_bootstrap_columns():
    found = bootstrap_row("a", "b", POOLED, np.array([1.0]), np.array([2.0]), np.array([10.0]),
                          lambda rng: np.ones(1), 3, 0)
    assert list(found) == BOOTSTRAP_COLUMNS
    assert (found["method"], found["against"], found["horizon"]) == ("a", "b", POOLED)
    assert found["rows"] == 1 and found["samples"] == 3


def test_bootstrap_row_observed_numbers_by_hand():
    # errors 10, 20 vs 30, 30 over actual 100, 100: wape 30/200 = 0.15, against
    # 60/200 = 0.3, difference -0.15.
    found = bootstrap_row("a", "b", POOLED, np.array([10.0, 20.0]), np.array([30.0, 30.0]),
                          np.array([100.0, 100.0]), lambda rng: np.ones(2), 5, 0)
    assert found["wape"] == pytest.approx(0.15)
    assert found["wape_against"] == pytest.approx(0.3)
    assert found["difference"] == pytest.approx(-0.15)


def test_bootstrap_row_low_high_and_share_by_hand_from_known_weights():
    # 1. errors a = 10, 40, b = 20, 20, actual 100, 100 (observed: 50/200 - 40/200 = +0.05).
    error, error_against, actual = (np.array([10.0, 40.0]), np.array([20.0, 20.0]),
                                    np.array([100.0, 100.0]))
    # 2. Four resamples with weights (2, 0), (0, 2), (1, 1), (2, 0): differences
    #    10/100 - 20/100 = -0.1;  40/100 - 20/100 = +0.2;  +0.05;  -0.1.
    vectors = [[2, 0], [0, 2], [1, 1], [2, 0]]
    differences = np.array([-0.1, 0.2, 0.05, -0.1])
    found = bootstrap_row("a", "b", POOLED, error, error_against, actual,
                          bs_cycling_weights(vectors), 4, 0)
    assert found["difference"] == pytest.approx(0.05)
    assert found["low"] == pytest.approx(np.quantile(differences, 0.05))
    assert found["high"] == pytest.approx(np.quantile(differences, 0.95))
    # 3. Two of four below 0.
    assert found["share_better"] == pytest.approx(0.5)


def test_bootstrap_row_identical_methods_give_zero_everywhere():
    error = np.array([5.0, 9.0, 1.0])
    found = bootstrap_row("a", "b", POOLED, error, error.copy(), np.array([10.0, 20.0, 30.0]),
                          lambda rng: rng.integers(1, 4, size=3).astype(float), 200, 7)
    assert found["difference"] == 0.0
    assert found["low"] == 0.0 and found["high"] == 0.0
    # A tie is never "better".
    assert found["share_better"] == 0.0


def test_bootstrap_row_better_in_every_row_is_better_in_every_resample():
    found = bootstrap_row("a", "b", POOLED, np.array([1.0, 2.0, 3.0]),
                          np.array([2.0, 3.0, 4.0]), np.array([10.0, 10.0, 10.0]),
                          lambda rng: rng.integers(0, 3, size=3).astype(float) + 0.5, 300, 1)
    assert found["share_better"] == 1.0
    assert found["high"] < 0.0 and found["difference"] < 0.0


def test_bootstrap_row_with_one_sample_has_its_one_difference_as_both_ends():
    found = bootstrap_row("a", "b", POOLED, np.array([10.0, 40.0]), np.array([20.0, 20.0]),
                          np.array([100.0, 100.0]), bs_cycling_weights([[0, 2]]), 1, 0)
    assert found["low"] == pytest.approx(0.2) and found["high"] == pytest.approx(0.2)
    assert found["share_better"] == 0.0
    assert found["samples"] == 1


def test_bootstrap_row_calls_draw_weights_once_per_sample_with_a_seeded_rng():
    # 1. Record the first number each call's rng gives.
    seen = []

    def draw(rng):
        seen.append(float(rng.random()))
        return np.ones(2)
    bootstrap_row("a", "b", POOLED, np.ones(2), np.ones(2), np.ones(2), draw, 6, 123)
    # 2. Six calls, all on one rng seeded with 123, in sequence.
    reference = np.random.default_rng(123)
    assert seen == [float(reference.random()) for _ in range(6)]


def test_bootstrap_row_numbers_are_python_floats():
    found = bootstrap_row("a", "b", POOLED, np.array([1.0, 3.0]), np.array([2.0, 1.0]),
                          np.array([5.0, 5.0]), lambda rng: rng.integers(1, 3, 2).astype(float),
                          20, 0)
    for name in ("wape", "wape_against", "difference", "low", "high", "share_better"):
        assert isinstance(found[name], float), name


# --- paired_wape_bootstrap: the running example ---------------------------------------------------
# Horizons 2; origins 10 ... 13; consolidated actual by target week: 11: 100, 12: 200,
# 13: 150, 14: 120; week 15 has none, so (13, 2) is not scored and origin 13 is incomplete.
#   (origin, h, target):  (10,1,11) (10,2,12) (11,1,12) (11,2,13) (12,1,13) (12,2,14) (13,1,14)
#   actual                   100       200       200       150       150       120       120
#   hurdle                   110       190       230       140       160       100       130
#   deflated                  80       240       150       180       120       150        90
#   twin                  = hurdle everywhere
# hurdle |e|: 10 10 30 10 10 20 10 = 100;  deflated |e|: 20 40 50 30 30 30 30 = 230;
# actual sum 1040: pooled WAPE hurdle 100/1040, deflated 230/1040; hurdle better in every row.
# Totals of the complete origins 10, 11, 12 (actual 300, 350, 270 = 920):
#   hurdle 300, 370, 260 -> |e| 0, 20, 10 = 30;  deflated 320, 330, 270 -> |e| 20, 20, 0 = 40.
# Every (origin, h) of origin 13 at h = 2 forecasts 999 (never scored). TR02 is a decoy
# (forecast 5, actual 4 everywhere): only "ALL" is bootstrapped.

BS_HORIZONS = 2
BS_PAIRS = [(10, 1, 11), (10, 2, 12), (11, 1, 12), (11, 2, 13), (12, 1, 13), (12, 2, 14),
            (13, 1, 14), (13, 2, 15)]
BS_ACTUAL_BY_WEEK = {11: 100.0, 12: 200.0, 13: 150.0, 14: 120.0}
BS_VALUES = {
    HURDLE: [110.0, 190.0, 230.0, 140.0, 160.0, 100.0, 130.0, 999.0],
    "deflated": [80.0, 240.0, 150.0, 180.0, 120.0, 150.0, 90.0, 999.0],
    "twin": [110.0, 190.0, 230.0, 140.0, 160.0, 100.0, 130.0, 999.0],
}


def bs_forecasts(values=None):
    values = BS_VALUES if values is None else values
    rows = []
    for method, by_pair in values.items():
        for company in ("TR02", "ALL"):
            for (origin, h, target), value in zip(BS_PAIRS, by_pair):
                rows.append((method, company, origin, h, target,
                             value if company == "ALL" else 5.0))
    return s_forecasts(rows)


def bs_actual(by_week=None):
    by_week = BS_ACTUAL_BY_WEEK if by_week is None else by_week
    rows = []
    for company in ("TR02", "ALL"):
        for origin, h, target in BS_PAIRS:
            if target in by_week:
                rows.append((company, origin, h, target,
                             by_week[target] if company == "ALL" else 4.0))
    return s_actual(rows)


def bs_run(pairs=((HURDLE, "deflated"),), samples=300, seed=11, forecasts=None, actual=None,
           horizons=BS_HORIZONS):
    forecasts = bs_forecasts() if forecasts is None else forecasts
    actual = bs_actual() if actual is None else actual
    return paired_wape_bootstrap(forecasts, actual, list(pairs), horizons, samples, seed)


def bs_row(result, method, against, horizon):
    found = result[(result["method"] == method) & (result["against"] == against)
                   & (result["horizon"] == horizon)]
    assert len(found) == 1, (method, against, horizon, len(found))
    return found.iloc[0]


def bs_expected_differences(errors, errors_against, actual, units, kind, samples, seed,
                            horizons):
    """The resampled differences rebuilt from the recipe: `units` is each value's target week
    (pooled) or origin (total)."""
    rng = np.random.default_rng(seed)
    distinct = sorted(set(units))
    count = len(distinct)
    differences = []
    for _ in range(samples):
        # 1. Draw the target weeks (pooled) or circular blocks of origins (total).
        if kind == POOLED:
            drawn = list(rng.choice(distinct, size=count, replace=True))
        else:
            block = min(horizons, count)
            drawn = []
            while len(drawn) < count:
                start = int(rng.integers(0, count))
                drawn += [distinct[(start + step) % count] for step in range(block)]
            drawn = drawn[:count]
        # 2. Each value counts as often as its unit was drawn.
        weights = np.array([drawn.count(unit) for unit in units], dtype=float)
        mine = (weights * errors).sum() / (weights * actual).sum()
        theirs = (weights * errors_against).sum() / (weights * actual).sum()
        differences.append(mine - theirs)
    return np.array(differences)


def test_bootstrap_has_exactly_the_bootstrap_columns_and_a_fresh_index():
    result = bs_run()
    assert BOOTSTRAP_COLUMNS == ["method", "against", "horizon", "rows", "wape", "wape_against",
                                 "difference", "low", "high", "share_better", "samples"]
    assert list(result.columns) == BOOTSTRAP_COLUMNS
    assert list(result.index) == list(range(len(result)))


def test_bootstrap_one_pooled_and_one_total_row_per_pair_in_pair_order():
    result = bs_run(pairs=[(HURDLE, "deflated"), ("deflated", "twin"), (HURDLE, "twin")])
    assert list(zip(result["method"], result["against"], result["horizon"])) == [
        (HURDLE, "deflated", POOLED), (HURDLE, "deflated", HORIZON_TOTAL),
        ("deflated", "twin", POOLED), ("deflated", "twin", HORIZON_TOTAL),
        (HURDLE, "twin", POOLED), (HURDLE, "twin", HORIZON_TOTAL)]


def test_bootstrap_rows_count_the_origin_h_pooled_and_the_complete_origins_total():
    result = bs_run()
    # 7 consolidated (origin, h) with actual; 3 complete origins (13 misses week 15).
    assert bs_row(result, HURDLE, "deflated", POOLED)["rows"] == 7
    assert bs_row(result, HURDLE, "deflated", HORIZON_TOTAL)["rows"] == 3


def test_bootstrap_observed_wapes_by_hand():
    result = bs_run()
    pooled = bs_row(result, HURDLE, "deflated", POOLED)
    assert pooled["wape"] == pytest.approx(100.0 / 1040.0)
    assert pooled["wape_against"] == pytest.approx(230.0 / 1040.0)
    assert pooled["difference"] == pytest.approx(-130.0 / 1040.0)
    total = bs_row(result, HURDLE, "deflated", HORIZON_TOTAL)
    assert total["wape"] == pytest.approx(30.0 / 920.0)
    assert total["wape_against"] == pytest.approx(40.0 / 920.0)
    assert total["difference"] == pytest.approx(-10.0 / 920.0)


def test_bootstrap_observed_wapes_equal_evaluation_scores_for_all():
    # The observed numbers are the consolidated pooled and total WAPEs of evaluation_scores.
    forecasts, actual = bs_forecasts(), bs_actual()
    scores = evaluation_scores(forecasts, actual, BS_HORIZONS)
    result = bs_run(pairs=[(HURDLE, "deflated"), ("deflated", "twin")])
    for method, against in [(HURDLE, "deflated"), ("deflated", "twin")]:
        for horizon in (POOLED, HORIZON_TOTAL):
            row = bs_row(result, method, against, horizon)
            assert row["wape"] == pytest.approx(
                score_of(scores, method, CONSOLIDATED, horizon)["wape"])
            assert row["wape_against"] == pytest.approx(
                score_of(scores, against, CONSOLIDATED, horizon)["wape"])
            assert row["rows"] == score_of(scores, method, CONSOLIDATED, horizon)["rows"]


def test_bootstrap_pooled_interval_matches_the_recipe_exactly():
    # 1. The seven scored rows of the running example, in (origin, h) order.
    scored = [(o, h, t) for o, h, t in BS_PAIRS if t in BS_ACTUAL_BY_WEEK]
    weeks = [t for _, _, t in scored]
    actual = np.array([BS_ACTUAL_BY_WEEK[t] for t in weeks])
    hurdle = np.array([BS_VALUES[HURDLE][BS_PAIRS.index(p)] for p in scored])
    deflated = np.array([BS_VALUES["deflated"][BS_PAIRS.index(p)] for p in scored])
    # 2. The differences over the same resamples, rebuilt.
    differences = bs_expected_differences(np.abs(hurdle - actual), np.abs(deflated - actual),
                                          actual, weeks, POOLED, 300, 11, BS_HORIZONS)
    row = bs_row(bs_run(samples=300, seed=11), HURDLE, "deflated", POOLED)
    assert row["low"] == pytest.approx(np.quantile(differences, 0.05), rel=1e-9, abs=1e-12)
    assert row["high"] == pytest.approx(np.quantile(differences, 0.95), rel=1e-9, abs=1e-12)
    assert row["share_better"] == pytest.approx((differences < 0).mean())


def test_bootstrap_total_interval_matches_the_recipe_exactly():
    # 1. The complete origins' totals (10, 11, 12), by hand from the table above.
    actual = np.array([300.0, 350.0, 270.0])
    hurdle_errors = np.array([0.0, 20.0, 10.0])
    deflated_errors = np.array([20.0, 20.0, 0.0])
    # 2. Circular blocks of min(2, 3) = 2 origins, rebuilt.
    differences = bs_expected_differences(hurdle_errors, deflated_errors, actual, [10, 11, 12],
                                          HORIZON_TOTAL, 300, 11, BS_HORIZONS)
    row = bs_row(bs_run(samples=300, seed=11), HURDLE, "deflated", HORIZON_TOTAL)
    assert row["low"] == pytest.approx(np.quantile(differences, 0.05), rel=1e-9, abs=1e-12)
    assert row["high"] == pytest.approx(np.quantile(differences, 0.95), rel=1e-9, abs=1e-12)
    assert row["share_better"] == pytest.approx((differences < 0).mean())


def test_bootstrap_an_identical_method_gives_zero_and_never_better():
    result = bs_run(pairs=[(HURDLE, "twin")])
    for horizon in (POOLED, HORIZON_TOTAL):
        row = bs_row(result, HURDLE, "twin", horizon)
        assert row["difference"] == 0.0, horizon
        assert (row["low"], row["high"]) == (0.0, 0.0), horizon
        assert row["share_better"] == 0.0, horizon
        assert row["wape"] == row["wape_against"], horizon


def test_bootstrap_better_in_every_week_is_better_in_every_pooled_resample():
    # hurdle's error is below deflated's in every scored (origin, h).
    row = bs_row(bs_run(samples=500), HURDLE, "deflated", POOLED)
    assert row["share_better"] == 1.0
    assert row["low"] <= row["high"] < 0.0


def test_bootstrap_the_reversed_pair_mirrors_the_pair():
    # Same seed, same resamples: the differences change sign, so the interval mirrors and the
    # shares of "better" add up to 1 minus the share of ties.
    result = bs_run(pairs=[(HURDLE, "deflated"), ("deflated", HURDLE)])
    for horizon in (POOLED, HORIZON_TOTAL):
        forward = bs_row(result, HURDLE, "deflated", horizon)
        backward = bs_row(result, "deflated", HURDLE, horizon)
        assert backward["difference"] == pytest.approx(-forward["difference"])
        assert backward["low"] == pytest.approx(-forward["high"])
        assert backward["high"] == pytest.approx(-forward["low"])
        assert backward["wape"] == pytest.approx(forward["wape_against"])
        assert forward["share_better"] + backward["share_better"] <= 1.0 + 1e-12


def test_bootstrap_interval_holds_the_observed_difference_side_and_is_ordered():
    result = bs_run(pairs=[(HURDLE, "deflated"), ("deflated", HURDLE)], samples=400)
    for _, row in result.iterrows():
        assert row["low"] <= row["high"]
        assert 0.0 <= row["share_better"] <= 1.0
        assert row["samples"] == 400


def test_bootstrap_samples_one_gives_one_difference_as_both_ends():
    result = bs_run(samples=1)
    for _, row in result.iterrows():
        assert row["low"] == row["high"]
        assert row["share_better"] in (0.0, 1.0)
        assert row["samples"] == 1


def test_bootstrap_one_target_week_gives_a_point_interval():
    # Horizons 1, one origin: one (origin, h), one week, one origin. Every resample draws the
    # same week (and the same origin), so the interval is the observed difference.
    forecasts = s_forecasts([("a", "ALL", 10, 1, 11, 90.0), ("b", "ALL", 10, 1, 11, 130.0)])
    actual = s_actual([("ALL", 10, 1, 11, 100.0)])
    result = paired_wape_bootstrap(forecasts, actual, [("a", "b")], 1, 50, 3)
    assert list(result["horizon"]) == [POOLED, HORIZON_TOTAL]
    for _, row in result.iterrows():
        # a: 10/100, b: 30/100, difference -0.2.
        assert row["rows"] == 1
        assert row["difference"] == pytest.approx(-0.2)
        assert row["low"] == pytest.approx(-0.2) and row["high"] == pytest.approx(-0.2)
        assert row["share_better"] == 1.0


def test_bootstrap_is_deterministic():
    pd.testing.assert_frame_equal(bs_run(pairs=[(HURDLE, "deflated"), ("deflated", "twin")]),
                                  bs_run(pairs=[(HURDLE, "deflated"), ("deflated", "twin")]))


def test_bootstrap_another_seed_changes_the_interval_not_the_observed_numbers():
    first = bs_row(bs_run(seed=1, samples=200), HURDLE, "deflated", POOLED)
    second = bs_row(bs_run(seed=2, samples=200), HURDLE, "deflated", POOLED)
    for name in ("wape", "wape_against", "difference", "rows"):
        assert first[name] == second[name], name
    # Over four target weeks and 200 resamples, the two seeds' 5% and 95% points differ.
    assert (first["low"], first["high"]) != (second["low"], second["high"])


def test_bootstrap_a_pairs_rows_do_not_depend_on_the_other_pairs_or_their_order():
    # 1. The pair alone.
    alone = bs_run(pairs=[(HURDLE, "deflated")])
    # 2. The same pair after another, and before another.
    after = bs_run(pairs=[("deflated", "twin"), (HURDLE, "deflated")])
    before = bs_run(pairs=[(HURDLE, "deflated"), ("twin", "deflated")])
    mine_after = after[after["method"] == HURDLE].reset_index(drop=True)
    mine_before = before[before["method"] == HURDLE].reset_index(drop=True)
    pd.testing.assert_frame_equal(mine_after, alone)
    pd.testing.assert_frame_equal(mine_before, alone)


def test_bootstrap_swapping_the_pairs_swaps_the_blocks_of_rows():
    one = bs_run(pairs=[(HURDLE, "deflated"), ("deflated", "twin")])
    two = bs_run(pairs=[("deflated", "twin"), (HURDLE, "deflated")])
    pd.testing.assert_frame_equal(one.iloc[[2, 3, 0, 1]].reset_index(drop=True), two)


def test_bootstrap_the_same_pair_twice_gives_the_same_rows_twice():
    result = bs_run(pairs=[(HURDLE, "deflated"), (HURDLE, "deflated")])
    first = result.iloc[:2].reset_index(drop=True)
    second = result.iloc[2:].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_bootstrap_empty_pairs_give_an_empty_table_with_the_columns():
    result = bs_run(pairs=[])
    assert result.empty
    assert list(result.columns) == BOOTSTRAP_COLUMNS


def test_bootstrap_an_incomplete_origin_is_left_out_of_the_total_only():
    # Origin 13's h = 1 forecast changed: the pooled row moves, the total row does not.
    values = {name: list(v) for name, v in BS_VALUES.items()}
    values[HURDLE][6] = 5000.0                     # (13, 1, 14)
    changed = bs_run(forecasts=bs_forecasts(values))
    base = bs_run()
    assert (bs_row(changed, HURDLE, "deflated", POOLED)["wape"]
            != bs_row(base, HURDLE, "deflated", POOLED)["wape"])
    pd.testing.assert_series_equal(bs_row(changed, HURDLE, "deflated", HORIZON_TOTAL),
                                   bs_row(base, HURDLE, "deflated", HORIZON_TOTAL))


def test_bootstrap_no_complete_origin_means_no_total_row():
    # Only week 11 and 13 have actual: every origin misses one of its two weeks.
    actual = bs_actual({11: 100.0, 13: 150.0})
    result = bs_run(actual=actual)
    assert list(result["horizon"]) == [POOLED]
    assert result.iloc[0]["rows"] == 3            # (10,1), (11,2), (12,1)


def test_bootstrap_total_block_is_cut_to_the_number_of_complete_origins():
    # Horizons 3 but only origins 10 and 11 complete: blocks of min(3, 2) = 2, so every
    # resample is a rotation of the two origins and the total interval is a point.
    pairs3 = [(o, h, o + h) for o in (10, 11, 12) for h in (1, 2, 3)]
    weeks = {11: 100.0, 12: 100.0, 13: 100.0, 14: 100.0}       # origin 12 misses week 15
    rows_a = [("a", "ALL", o, h, t, 110.0 + o) for o, h, t in pairs3]
    rows_b = [("b", "ALL", o, h, t, 80.0) for o, h, t in pairs3]
    forecasts = s_forecasts(rows_a + rows_b)
    actual = s_actual([("ALL", o, h, t, weeks[t]) for o, h, t in pairs3 if t in weeks])
    result = paired_wape_bootstrap(forecasts, actual, [("a", "b")], 3, 100, 4)
    total = bs_row(result, "a", "b", HORIZON_TOTAL)
    assert total["rows"] == 2
    assert total["low"] == pytest.approx(total["difference"])
    assert total["high"] == pytest.approx(total["difference"])


def test_bootstrap_a_week_without_actual_is_left_out():
    # (13, 2) -> week 15 has no actual: its forecast, whatever it is, changes nothing.
    values = {name: list(v) for name, v in BS_VALUES.items()}
    values[HURDLE][7] = 1e12
    pd.testing.assert_frame_equal(bs_run(forecasts=bs_forecasts(values)), bs_run())


def test_bootstrap_uses_the_consolidated_rows_only():
    # TR02's forecasts and actual changed: the same result.
    forecasts, actual = bs_forecasts(), bs_actual()
    forecasts.loc[forecasts["company_code"] == "TR02", "forecast"] = 1e6
    actual.loc[actual["company_code"] == "TR02", "actual"] = 1.0
    pd.testing.assert_frame_equal(bs_run(forecasts=forecasts, actual=actual), bs_run())


def test_bootstrap_a_company_actual_of_zero_is_not_refused():
    # The "above 0" check is on the consolidated actual only.
    actual = bs_actual()
    actual.loc[actual["company_code"] == "TR02", "actual"] = 0.0
    result = bs_run(actual=actual)
    assert len(result) == 2


def test_bootstrap_does_not_change_its_inputs():
    forecasts, actual = bs_forecasts(), bs_actual()
    pairs = [(HURDLE, "deflated")]
    before = (forecasts.copy(), actual.copy(), list(pairs))
    paired_wape_bootstrap(forecasts, actual, pairs, BS_HORIZONS, 50, 0)
    pd.testing.assert_frame_equal(forecasts, before[0])
    pd.testing.assert_frame_equal(actual, before[1])
    assert pairs == before[2]


def test_bootstrap_does_not_depend_on_the_row_order_of_its_inputs():
    forecasts, actual = bs_forecasts(), bs_actual()
    shuffled_forecasts = forecasts.sample(frac=1.0, random_state=5).reset_index(drop=True)
    shuffled_actual = actual.sample(frac=1.0, random_state=6).reset_index(drop=True)
    shuffled = paired_wape_bootstrap(shuffled_forecasts, shuffled_actual, [(HURDLE, "deflated")],
                                     BS_HORIZONS, 300, 11)
    pd.testing.assert_frame_equal(shuffled, bs_run(), check_exact=False, rtol=1e-12)


def test_bootstrap_does_not_depend_on_the_scale_of_the_money():
    # WAPE is a ratio: every forecast and actual times 1000 gives the same numbers.
    forecasts, actual = bs_forecasts(), bs_actual()
    forecasts["forecast"] *= 1000.0
    actual["actual"] *= 1000.0
    pd.testing.assert_frame_equal(bs_run(forecasts=forecasts, actual=actual), bs_run(),
                                  check_exact=False, rtol=1e-9)


def test_bootstrap_end_to_end_on_stacked_forecasts(stack_inputs):
    # The pipeline's shape: stack_forecasts' table, the hurdle against each other method.
    rolled, reference, rule = stack_inputs
    forecasts = stack_forecasts(rolled, reference, rule)
    actual = s_actual([(c, o, h, t, 1000.0) for c in ("TR02", "ALL")
                       for o, h, t in S_PAIRS[:3]])
    pairs = [(HURDLE, other) for other in dict.fromkeys(forecasts["method"]) if other != HURDLE]
    result = paired_wape_bootstrap(forecasts, actual, pairs, 2, 100, 0)
    assert list(dict.fromkeys(result["against"])) == [*REFERENCES, RULE]
    assert set(result["method"]) == {HURDLE}
    assert len(result) == 2 * len(pairs)


# --- paired_wape_bootstrap: refusals --------------------------------------------------------------

@pytest.mark.parametrize("samples", [0, -1, -2000])
def test_bootstrap_samples_below_one_exact_message(samples):
    with pytest.raises(ValueError) as error:
        bs_run(samples=samples)
    assert str(error.value) == f"samples is {samples}; at least 1 is needed"


def test_bootstrap_a_pair_with_the_same_method_twice_exact_message():
    with pytest.raises(ValueError) as error:
        bs_run(pairs=[(HURDLE, "deflated"), ("twin", "twin")])
    assert str(error.value) == "the pair ('twin', 'twin') compares a method with itself"


def test_bootstrap_a_pair_naming_an_unknown_method_exact_message():
    with pytest.raises(ValueError) as error:
        bs_run(pairs=[(HURDLE, "arima")])
    assert str(error.value) == ("the pair ('hurdle', 'arima') names methods with no forecasts: "
                                "['arima']")


def test_bootstrap_a_pair_naming_two_unknown_methods_lists_both_sorted():
    with pytest.raises(ValueError) as error:
        bs_run(pairs=[("zeta", "alpha")])
    assert str(error.value) == ("the pair ('zeta', 'alpha') names methods with no forecasts: "
                                "['alpha', 'zeta']")


def test_bootstrap_the_same_unknown_method_twice_is_a_self_pair_first():
    with pytest.raises(ValueError, match=r"compares a method with itself$"):
        bs_run(pairs=[("arima", "arima")])


def test_bootstrap_the_samples_check_comes_before_the_pair_checks():
    with pytest.raises(ValueError, match=r"^samples is 0; at least 1 is needed$"):
        bs_run(pairs=[("x", "x")], samples=0)


def test_bootstrap_the_first_bad_pair_is_named():
    with pytest.raises(ValueError, match=r"^the pair \('hurdle', 'nope'\)"):
        bs_run(pairs=[(HURDLE, "deflated"), (HURDLE, "nope"), ("twin", "twin")])


def test_bootstrap_the_pair_checks_come_before_scored_rows_checks():
    # horizons 0 would be refused by scored_rows; the bad pair is reported first.
    with pytest.raises(ValueError, match=r"^the pair \('hurdle', 'nope'\)"):
        bs_run(pairs=[(HURDLE, "nope")], horizons=0)


def test_bootstrap_scored_rows_refusals_apply():
    # 1. Horizons below 1.
    with pytest.raises(ValueError) as error:
        bs_run(horizons=0)
    assert str(error.value) == "horizons is 0; horizons start at 1"
    # 2. A NaN forecast, even for a company the bootstrap does not use.
    forecasts = bs_forecasts()
    forecasts.loc[forecasts.index[0], "forecast"] = NAN
    with pytest.raises(ValueError, match=r"^1 forecasts are NaN or infinite"):
        bs_run(forecasts=forecasts)
    # 3. A method missing a row with actual, in a method no pair uses.
    forecasts = bs_forecasts()
    gone = (forecasts["method"] == "twin") & (forecasts["origin_week"] == 10)
    with pytest.raises(ValueError, match=r"^'twin' has no forecast for 4 rows"):
        bs_run(forecasts=forecasts[~gone])


@pytest.mark.parametrize("bad", [0.0, -1.0, -500.0])
def test_bootstrap_a_consolidated_actual_not_above_zero_exact_message(bad):
    actual = bs_actual()
    week_12 = (actual["company_code"] == "ALL") & (actual["target_week"] == 12)
    actual.loc[week_12, "actual"] = bad
    with pytest.raises(ValueError) as error:
        bs_run(actual=actual)
    assert str(error.value) == "every consolidated actual must be above 0 for a WAPE"


def test_bootstrap_scored_rows_checks_come_before_the_actual_check():
    actual = bs_actual()
    actual.loc[actual["company_code"] == "ALL", "actual"] = 0.0
    with pytest.raises(ValueError, match=r"^horizons is 0"):
        bs_run(actual=actual, horizons=0)


BS_NO_CONSOLIDATED = "there are no consolidated ('ALL') rows with actual cash to compare"


def test_bootstrap_with_no_consolidated_rows_refuses_rather_than_returning_nan():
    # Only TR02 has forecasts and actual: nothing consolidated to bootstrap. (Before the fix
    # this returned a NaN row with share_better 0.0, shown as "first better in 0%".)
    forecasts = s_forecasts([("a", "TR02", 10, 1, 11, 5.0), ("b", "TR02", 10, 1, 11, 6.0)])
    actual = s_actual([("TR02", 10, 1, 11, 4.0)])
    with pytest.raises(ValueError) as error:
        paired_wape_bootstrap(forecasts, actual, [("a", "b")], 1, 5, 0)
    assert str(error.value) == BS_NO_CONSOLIDATED


def test_bootstrap_with_no_consolidated_rows_raises_no_numpy_warning():
    # The refusal comes before any WAPE is divided out.
    import warnings
    forecasts = s_forecasts([("a", "TR02", 10, 1, 11, 5.0), ("b", "TR02", 10, 1, 11, 6.0)])
    actual = s_actual([("TR02", 10, 1, 11, 4.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"^there are no consolidated"):
            paired_wape_bootstrap(forecasts, actual, [("a", "b")], 1, 5, 0)


def test_bootstrap_refusal_order_scored_rows_then_no_consolidated_then_actual():
    # 1. Only TR02 rows and horizons 0: scored_rows' horizons message comes first.
    forecasts = s_forecasts([("a", "TR02", 10, 1, 11, 5.0), ("b", "TR02", 10, 1, 11, 6.0)])
    actual = s_actual([("TR02", 10, 1, 11, 0.0)])
    with pytest.raises(ValueError, match=r"^horizons is 0; horizons start at 1$"):
        paired_wape_bootstrap(forecasts, actual, [("a", "b")], 0, 5, 0)
    # 2. Valid horizons, no ALL rows, a TR02 actual of 0: the no-consolidated message, not
    #    the actual > 0 one.
    with pytest.raises(ValueError) as error:
        paired_wape_bootstrap(forecasts, actual, [("a", "b")], 1, 5, 0)
    assert str(error.value) == BS_NO_CONSOLIDATED
    # 3. ALL rows present but their actual 0: the actual > 0 message.
    zero_week = bs_actual()
    zero_week.loc[zero_week["company_code"] == "ALL", "actual"] = 0.0
    with pytest.raises(ValueError) as error:
        bs_run(actual=zero_week)
    assert str(error.value) == "every consolidated actual must be above 0 for a WAPE"


# =============================================================================================
# U14.5d, `part_actuals`, `part_rows`, `component_bias`: where the bias comes from, each part of
# the forecast against its own actual cash. `payer_row`, `payer_split`: for the known customers,
# how many pay against how much each pays.
#
# Every table is hand-made and every expected number is worked out by hand below. The worked
# example of `component_bias` has two companies, origins 10 and 11, h = 1, 2:
#
#   (origin, h, target)   projected level   published level of the target week
#   (10, 1, 11)           2.0               2.5
#   (10, 2, 12)           3.0               3.5
#   (11, 1, 12)           4.0               3.5
#   (11, 2, 13)           5.0               4.5
#
# So week 12 is reached from two origins, at two different projected levels but one published
# level. A wrong level, or an A read by target week instead of by (origin, h), shows.
# =============================================================================================

import warnings  # noqa: E402

from pandas.errors import MergeError  # noqa: E402

from collection_estimation.hurdle.evaluate import (  # noqa: E402
    ADDING_PARTS, COMPONENT_COLUMNS, KNOWN_CUSTOMERS, NEW_CUSTOMERS, PART_TOTAL, PAYER_COLUMNS,
    PRICE_LEVEL, REFUND_PART, TIE_TOLERANCE, UNIDENTIFIED, component_bias, part_actuals,
    part_rows, payer_row, payer_split, ratio_or_nan)
from collection_estimation.hurdle.forecast import CUSTOMER_COLUMNS, TOTAL_COLUMNS  # noqa: E402

CB_KEYS = ["company_code", "origin_week", "h", "target_week"]
CB_PAIRS = [(10, 1, 11), (10, 2, 12), (11, 1, 12), (11, 2, 13)]
CB_COMPANIES = ["TR02", "TR04"]
CB_HORIZONS = 2
CB_PROJECTED = {(10, 11): 2.0, (10, 12): 3.0, (11, 12): 4.0, (11, 13): 5.0}
CB_PUBLISHED = {11: 2.5, 12: 3.5, 13: 4.5}

# All the actual cash, by company and target week.
CB_CASH = {"TR02": {11: 1000.0, 12: 2000.0, 13: 3000.0},
           "TR04": {11: 100.0, 12: 200.0, 13: 300.0}}
# Table A, by company, origin and h: 0 at h = 1 in this example.
CB_NEW = {"TR02": {(10, 1): 0.0, (10, 2): 50.0, (11, 1): 0.0, (11, 2): 70.0},
          "TR04": {(10, 1): 0.0, (10, 2): 5.0, (11, 1): 0.0, (11, 2): 7.0}}
# Table G, by company and target week: TR04 has none at all.
CB_UNIDENTIFIED = {"TR02": {11: 10.0, 12: 20.0, 13: 30.0}, "TR04": {11: 0.0, 12: 0.0, 13: 0.0}}
# Table R, by company and target week: negative.
CB_REFUNDS = {"TR02": {11: -1.0, 12: -2.0, 13: -3.0}, "TR04": {11: -4.0, 12: 0.0, 13: -6.0}}

# What is left, by hand: actual - A - G - R.
#   TR02 (10,1,11): 1000 -  0 - 10 + 1 =  991     TR04 (10,1,11): 100 - 0 - 0 + 4 = 104
#   TR02 (10,2,12): 2000 - 50 - 20 + 2 = 1932     TR04 (10,2,12): 200 - 5 - 0 + 0 = 195
#   TR02 (11,1,12): 2000 -  0 - 20 + 2 = 1982     TR04 (11,1,12): 200 - 0 - 0 + 0 = 200
#   TR02 (11,2,13): 3000 - 70 - 30 + 3 = 2903     TR04 (11,2,13): 300 - 7 - 0 + 6 = 299
CB_KNOWN = {"TR02": {(10, 1): 991.0, (10, 2): 1932.0, (11, 1): 1982.0, (11, 2): 2903.0},
            "TR04": {(10, 1): 104.0, (10, 2): 195.0, (11, 1): 200.0, (11, 2): 299.0}}

# The forecast's parts in TRY of week 0: (customers, new customers, unidentified, refunds), in
# the order of CB_PAIRS. "ALL" is the two companies added up.
CB_REAL_PARTS = {
    "TR02": [(400.0, 1.0, 2.0, -1.0), (500.0, 10.0, 5.0, -2.0),
             (300.0, 1.0, 5.0, -1.0), (600.0, 10.0, 5.0, -3.0)],
    "TR04": [(40.0, 1.0, 0.0, -1.0), (50.0, 2.0, 0.0, -1.0),
             (60.0, 1.0, 0.0, -2.0), (70.0, 2.0, 0.0, -2.0)],
    "ALL": [(440.0, 2.0, 2.0, -2.0), (550.0, 12.0, 5.0, -3.0),
            (360.0, 2.0, 5.0, -3.0), (670.0, 12.0, 5.0, -5.0)],
}
# The rule's open invoices (nominal), in the order of CB_PAIRS.
CB_RULE_INVOICES = {"TR02": [900.0, 1000.0, 800.0, 1100.0], "TR04": [90.0, 100.0, 80.0, 110.0],
                    "ALL": [990.0, 1100.0, 880.0, 1210.0]}

# Every number component_bias must give: (company, horizon) -> (forecast, actual, gap, ratio),
# None where the actual is 0 and so the ratio is NaN. Worked out from the tables above:
#   TR02 known, h1: 400 x 2 + 300 x 4 = 2000 against 991 + 1982 = 2973.
#   TR02 total, h1: 402 x 2 + 305 x 4 = 2024 against 1000 + 2000 = 3000.
#   TR02 price level, h1: 2024 against 402 x 2.5 + 305 x 3.5 = 1005 + 1067.5 = 2072.5.
#   TR02 h2 price level: 513 x 3.5 + 612 x 4.5 = 1795.5 + 2754 = 4549.5.
#   TR04 h1 price level: 40 x 2.5 + 59 x 3.5 = 306.5; h2: 51 x 3.5 + 70 x 4.5 = 493.5.
#   ALL  h1 price level: 442 x 2.5 + 364 x 3.5 = 2379; h2: 564 x 3.5 + 682 x 4.5 = 5043.
CB_EXPECTED = {
    KNOWN_CUSTOMERS: {
        ("TR02", "1"): (2000.0, 2973.0, -973.0, 2000 / 2973),
        ("TR02", "2"): (4500.0, 4835.0, -335.0, 4500 / 4835),
        ("TR02", "pooled"): (6500.0, 7808.0, -1308.0, 6500 / 7808),
        ("TR04", "1"): (320.0, 304.0, 16.0, 320 / 304),
        ("TR04", "2"): (500.0, 494.0, 6.0, 500 / 494),
        ("TR04", "pooled"): (820.0, 798.0, 22.0, 820 / 798),
        ("ALL", "1"): (2320.0, 3277.0, -957.0, 2320 / 3277),
        ("ALL", "2"): (5000.0, 5329.0, -329.0, 5000 / 5329),
        ("ALL", "pooled"): (7320.0, 8606.0, -1286.0, 7320 / 8606),
    },
    NEW_CUSTOMERS: {
        ("TR02", "1"): (6.0, 0.0, 6.0, None),
        ("TR02", "2"): (80.0, 120.0, -40.0, 80 / 120),
        ("TR02", "pooled"): (86.0, 120.0, -34.0, 86 / 120),
        ("TR04", "1"): (6.0, 0.0, 6.0, None),
        ("TR04", "2"): (16.0, 12.0, 4.0, 16 / 12),
        ("TR04", "pooled"): (22.0, 12.0, 10.0, 22 / 12),
        ("ALL", "1"): (12.0, 0.0, 12.0, None),
        ("ALL", "2"): (96.0, 132.0, -36.0, 96 / 132),
        ("ALL", "pooled"): (108.0, 132.0, -24.0, 108 / 132),
    },
    UNIDENTIFIED: {
        ("TR02", "1"): (24.0, 30.0, -6.0, 0.8),
        ("TR02", "2"): (40.0, 50.0, -10.0, 0.8),
        ("TR02", "pooled"): (64.0, 80.0, -16.0, 0.8),
        ("TR04", "1"): (0.0, 0.0, 0.0, None),
        ("TR04", "2"): (0.0, 0.0, 0.0, None),
        ("TR04", "pooled"): (0.0, 0.0, 0.0, None),
        ("ALL", "1"): (24.0, 30.0, -6.0, 0.8),
        ("ALL", "2"): (40.0, 50.0, -10.0, 0.8),
        ("ALL", "pooled"): (64.0, 80.0, -16.0, 0.8),
    },
    REFUND_PART: {
        ("TR02", "1"): (-6.0, -3.0, -3.0, 2.0),
        ("TR02", "2"): (-21.0, -5.0, -16.0, 4.2),
        ("TR02", "pooled"): (-27.0, -8.0, -19.0, 27 / 8),
        ("TR04", "1"): (-10.0, -4.0, -6.0, 2.5),
        ("TR04", "2"): (-13.0, -6.0, -7.0, 13 / 6),
        ("TR04", "pooled"): (-23.0, -10.0, -13.0, 2.3),
        ("ALL", "1"): (-16.0, -7.0, -9.0, 16 / 7),
        ("ALL", "2"): (-34.0, -11.0, -23.0, 34 / 11),
        ("ALL", "pooled"): (-50.0, -18.0, -32.0, 50 / 18),
    },
    PART_TOTAL: {
        ("TR02", "1"): (2024.0, 3000.0, -976.0, 2024 / 3000),
        ("TR02", "2"): (4599.0, 5000.0, -401.0, 4599 / 5000),
        ("TR02", "pooled"): (6623.0, 8000.0, -1377.0, 6623 / 8000),
        ("TR04", "1"): (316.0, 300.0, 16.0, 316 / 300),
        ("TR04", "2"): (503.0, 500.0, 3.0, 503 / 500),
        ("TR04", "pooled"): (819.0, 800.0, 19.0, 819 / 800),
        ("ALL", "1"): (2340.0, 3300.0, -960.0, 2340 / 3300),
        ("ALL", "2"): (5102.0, 5500.0, -398.0, 5102 / 5500),
        ("ALL", "pooled"): (7442.0, 8800.0, -1358.0, 7442 / 8800),
    },
    PRICE_LEVEL: {
        ("TR02", "1"): (2024.0, 2072.5, -48.5, 2024 / 2072.5),
        ("TR02", "2"): (4599.0, 4549.5, 49.5, 4599 / 4549.5),
        ("TR02", "pooled"): (6623.0, 6622.0, 1.0, 6623 / 6622),
        ("TR04", "1"): (316.0, 306.5, 9.5, 316 / 306.5),
        ("TR04", "2"): (503.0, 493.5, 9.5, 503 / 493.5),
        ("TR04", "pooled"): (819.0, 800.0, 19.0, 819 / 800),
        ("ALL", "1"): (2340.0, 2379.0, -39.0, 2340 / 2379),
        ("ALL", "2"): (5102.0, 5043.0, 59.0, 5102 / 5043),
        ("ALL", "pooled"): (7442.0, 7422.0, 20.0, 7442 / 7422),
    },
}
# The rule's open invoices against the same known customers' cash.
CB_EXPECTED_RULE = {
    ("TR02", "1"): (1700.0, 2973.0, -1273.0, 1700 / 2973),
    ("TR02", "2"): (2100.0, 4835.0, -2735.0, 2100 / 4835),
    ("TR02", "pooled"): (3800.0, 7808.0, -4008.0, 3800 / 7808),
    ("TR04", "1"): (170.0, 304.0, -134.0, 170 / 304),
    ("TR04", "2"): (210.0, 494.0, -284.0, 210 / 494),
    ("TR04", "pooled"): (380.0, 798.0, -418.0, 380 / 798),
    ("ALL", "1"): (1870.0, 3277.0, -1407.0, 1870 / 3277),
    ("ALL", "2"): (2310.0, 5329.0, -3019.0, 2310 / 5329),
    ("ALL", "pooled"): (4180.0, 8606.0, -4426.0, 4180 / 8606),
}
CB_ORDER_COMPANIES = ["TR02", "TR04", "ALL"]
CB_ORDER_HORIZONS = ["1", "2", "pooled"]
CB_PARTS_IN_ORDER = [KNOWN_CUSTOMERS, NEW_CUSTOMERS, UNIDENTIFIED, REFUND_PART, PART_TOTAL,
                     PRICE_LEVEL]


# --- builders --------------------------------------------------------------------------------

def cb_actual():
    """The rows of `actual_totals` for the worked example, ALL included; actual_real a decoy."""
    records = []
    for origin, h, target in CB_PAIRS:
        for company in CB_COMPANIES:
            records.append((company, origin, h, target, CB_CASH[company][target], -1.0))
        consolidated_cash = sum(CB_CASH[company][target] for company in CB_COMPANIES)
        records.append((CONSOLIDATED, origin, h, target, consolidated_cash, -1.0))
    return pd.DataFrame(records, columns=ACTUAL_COLUMNS)


def cb_new_cash():
    """Table A, with its target_week as side_tables.new_customer_cash gives it."""
    records = [(company, origin, h, target, CB_NEW[company][(origin, h)])
               for company in CB_COMPANIES for origin, h, target in CB_PAIRS]
    return pd.DataFrame(records, columns=["company_code", "origin_week", "h", "target_week",
                                          "new_customer_cash"])


def cb_by_week(amounts, column):
    records = [(company, week, amount) for company, by_week in amounts.items()
               for week, amount in by_week.items()]
    return pd.DataFrame(records, columns=["company_code", "week", column])


def cb_refunds():
    return cb_by_week(CB_REFUNDS, "refunds")


def cb_unidentified():
    return cb_by_week(CB_UNIDENTIFIED, "unidentified_cash")


def cb_levels():
    return pd.Series(CB_PUBLISHED)


def cb_rolled(real_parts=None, projected=None):
    """A Forecast whose `totals` have the real parts, the projected level of each row, their
    sum as real_total and real_total x level as total."""
    real_parts = CB_REAL_PARTS if real_parts is None else real_parts
    projected = CB_PROJECTED if projected is None else projected
    records = []
    for company, parts in real_parts.items():
        for (origin, h, target), (customers, new, unidentified, refund) in zip(CB_PAIRS, parts):
            real_total = customers + new + unidentified + refund
            projected_level = projected[(origin, target)]
            records.append((company, origin, h, target, customers, new, unidentified, refund,
                            real_total, projected_level, real_total * projected_level, False))
    return Forecast(totals=pd.DataFrame(records, columns=TOTAL_COLUMNS),
                    customers=pd.DataFrame(columns=CUSTOMER_COLUMNS))


def cb_rule(invoices=None):
    """A RuleReference whose `forecast` is a decoy: the part is `invoice_cash`."""
    invoices = CB_RULE_INVOICES if invoices is None else invoices
    records = []
    for company, amounts in invoices.items():
        for (origin, h, target), amount in zip(CB_PAIRS, amounts):
            records.append({"company_code": company, "origin_week": origin, "h": h,
                            "target_week": target, "invoice_cash": amount, "late_cash": 7.0,
                            "fallback_delay_cash": 8.0, "topup_cash": 9.0, "price_level": -3.0,
                            "forecast": -1234.0})
    return RuleReference(totals=pd.DataFrame(records),
                         invoices=pd.DataFrame({"key": ["K0"]}))


def cb_run(**changes):
    """component_bias on the worked example, with some arguments replaced."""
    arguments = dict(rolled=cb_rolled(), rule=cb_rule(), actual=cb_actual(),
                     new_cash=cb_new_cash(), refunds=cb_refunds(),
                     unidentified=cb_unidentified(), levels=cb_levels(), horizons=CB_HORIZONS)
    arguments.update(changes)
    return component_bias(**arguments)


def cb_parts_run(**changes):
    """part_actuals on the worked example, with some tables replaced."""
    arguments = dict(actual=cb_actual(), new_cash=cb_new_cash(), refunds=cb_refunds(),
                     unidentified=cb_unidentified())
    arguments.update(changes)
    return part_actuals(**arguments)


def cb_part_row(found, company, origin, h):
    selected = found[(found["company_code"] == company) & (found["origin_week"] == origin)
                     & (found["h"] == h)]
    assert len(selected) == 1, (company, origin, h, len(selected))
    return selected.iloc[0]


def cb_bias_row(bias, method, part, company, horizon):
    selected = bias[(bias["method"] == method) & (bias["part"] == part)
                    & (bias["company_code"] == company) & (bias["horizon"] == horizon)]
    assert len(selected) == 1, (method, part, company, horizon, len(selected))
    return selected.iloc[0]


def cb_assert_numbers(found, expected):
    """expected: (forecast, actual, gap, ratio), ratio None for NaN."""
    made, came, gap, ratio = expected
    assert found["forecast"] == pytest.approx(made)
    assert found["actual"] == pytest.approx(came)
    assert found["gap"] == pytest.approx(gap)
    if ratio is None:
        assert np.isnan(found["ratio"])
    else:
        assert found["ratio"] == pytest.approx(ratio)


def cb_refused(expected_message, **changes):
    """component_bias must refuse with exactly this message."""
    with pytest.raises(ValueError) as error:
        cb_run(**changes)
    assert str(error.value) == expected_message


def cb_totals_edited(column, company, origin, h, change):
    """The worked example's Forecast, with `change` added to one row's `column`."""
    rolled = cb_rolled()
    chosen = ((rolled.totals["company_code"] == company)
              & (rolled.totals["origin_week"] == origin) & (rolled.totals["h"] == h))
    assert chosen.sum() == 1
    rolled.totals.loc[chosen, column] += change
    return rolled


# --- constants -------------------------------------------------------------------------------

def test_component_bias_part_names():
    assert KNOWN_CUSTOMERS == "known customers"
    assert NEW_CUSTOMERS == "new customers"
    assert UNIDENTIFIED == "unidentified"
    assert REFUND_PART == "refunds"
    assert PART_TOTAL == "total"
    assert PRICE_LEVEL == "price level"


def test_adding_parts_map_each_part_to_its_column_of_the_totals_in_order():
    assert list(ADDING_PARTS.items()) == [(KNOWN_CUSTOMERS, "customers"),
                                          (NEW_CUSTOMERS, "new_customer_cash"),
                                          (UNIDENTIFIED, "unidentified_cash"),
                                          (REFUND_PART, "refunds")]


def test_price_level_and_total_are_not_adding_parts():
    assert PRICE_LEVEL not in ADDING_PARTS
    assert PART_TOTAL not in ADDING_PARTS


def test_component_columns():
    assert COMPONENT_COLUMNS == ["method", "company_code", "horizon", "part", "forecast",
                                 "actual", "gap", "ratio"]


def test_payer_columns():
    assert PAYER_COLUMNS == ["company_code", "horizon", "rows", "expected_payers",
                             "actual_payers", "p_ratio", "expected_per_payer",
                             "actual_per_payer", "m_ratio", "pxm_ratio"]


def test_tie_tolerance_is_one_kurus():
    assert TIE_TOLERANCE == 0.01


# --- ratio_or_nan ----------------------------------------------------------------------------

@pytest.mark.parametrize("top, bottom, expected", [
    (6.0, 3.0, 2.0), (0.0, 3.0, 0.0), (-6.0, 3.0, -2.0), (-6.0, -3.0, 2.0), (1, 4, 0.25)])
def test_ratio_or_nan_divides(top, bottom, expected):
    assert ratio_or_nan(top, bottom) == expected


@pytest.mark.parametrize("top, bottom", [(1.0, 0.0), (0.0, 0.0), (-1.0, 0.0), (1.0, -0.0),
                                         (1.0, 0), (float("nan"), 0.0), (1.0, float("nan")),
                                         (float("nan"), 2.0)])
def test_ratio_or_nan_is_nan_without_a_bottom_or_a_top(top, bottom):
    assert np.isnan(ratio_or_nan(top, bottom))


def test_ratio_or_nan_gives_a_plain_float_from_numpy_numbers():
    assert type(ratio_or_nan(np.float64(1.0), np.int64(2))) is float


# --- part_actuals ----------------------------------------------------------------------------

def test_part_actuals_columns_and_row_order():
    # 1. The keys, then the amounts; the companies' rows as they came, then ALL by origin and h.
    found = cb_parts_run()
    assert list(found.columns) == CB_KEYS + ["actual", "actual_new", "actual_unidentified",
                                             "actual_refunds", "actual_known"]
    companies_first = [(company, origin, h, target) for origin, h, target in CB_PAIRS
                       for company in CB_COMPANIES]
    consolidated_last = [(CONSOLIDATED, origin, h, target) for origin, h, target in CB_PAIRS]
    assert list(found[CB_KEYS].itertuples(index=False, name=None)) == (companies_first
                                                                       + consolidated_last)


def test_part_actuals_a_is_read_by_origin_and_h_not_by_target_week():
    # Week 12 is reached from (10, 2) and from (11, 1): one target week, two different A.
    found = cb_parts_run()
    assert cb_part_row(found, "TR02", 10, 2)["actual_new"] == 50.0
    assert cb_part_row(found, "TR02", 11, 1)["actual_new"] == 0.0


@pytest.mark.parametrize("origin, h", [(10, 2), (11, 1)])
def test_part_actuals_g_and_r_are_read_by_target_week(origin, h):
    # Week 12 from either origin: the same G and R.
    found = cb_parts_run()
    assert cb_part_row(found, "TR02", origin, h)["actual_unidentified"] == 20.0
    assert cb_part_row(found, "TR02", origin, h)["actual_refunds"] == -2.0


@pytest.mark.parametrize("company", CB_COMPANIES)
@pytest.mark.parametrize("origin, h, target", CB_PAIRS)
def test_part_actuals_known_is_what_is_left(company, origin, h, target):
    # By hand in the table above: actual - A - G - R, with R negative so it adds.
    found = cb_part_row(cb_parts_run(), company, origin, h)
    assert found["target_week"] == target
    assert found["actual"] == CB_CASH[company][target]
    assert found["actual_known"] == pytest.approx(CB_KNOWN[company][(origin, h)])


@pytest.mark.parametrize("origin, h, new, unidentified, refund, known", [
    (10, 1, 0.0, 10.0, -5.0, 991.0 + 104.0),
    (10, 2, 55.0, 20.0, -2.0, 1932.0 + 195.0),
    (11, 1, 0.0, 20.0, -2.0, 1982.0 + 200.0),
    (11, 2, 77.0, 30.0, -9.0, 2903.0 + 299.0)])
def test_part_actuals_all_is_the_sum_of_the_companies(origin, h, new, unidentified, refund, known):
    consolidated = cb_part_row(cb_parts_run(), CONSOLIDATED, origin, h)
    assert consolidated["actual_new"] == pytest.approx(new)
    assert consolidated["actual_unidentified"] == pytest.approx(unidentified)
    assert consolidated["actual_refunds"] == pytest.approx(refund)
    assert consolidated["actual_known"] == pytest.approx(known)


def test_part_actuals_all_is_not_read_from_the_all_rows_of_the_actual():
    # 1. The ALL rows of `actual` hold rubbish.
    actual = cb_actual()
    actual.loc[actual["company_code"] == CONSOLIDATED, "actual"] = -999999.0
    # 2. ALL is still the companies added up: week 11 from origin 10 is 1000 + 100.
    consolidated = cb_part_row(cb_parts_run(actual=actual), CONSOLIDATED, 10, 1)
    assert consolidated["actual"] == 1100.0


def test_part_actuals_the_parts_add_up_to_the_actual_in_every_row():
    found = cb_parts_run()
    added_up = (found["actual_known"] + found["actual_new"] + found["actual_unidentified"]
                + found["actual_refunds"])
    assert added_up.to_numpy() == pytest.approx(found["actual"].to_numpy())


def test_part_actuals_company_code_is_plain_object():
    assert cb_parts_run()["company_code"].dtype == object


def test_part_actuals_inputs_unchanged():
    # 1. Copies of every table.
    actual, new_cash, refunds, unidentified = (cb_actual(), cb_new_cash(), cb_refunds(),
                                               cb_unidentified())
    copies = [table.copy() for table in (actual, new_cash, refunds, unidentified)]
    # 2. Called, and each table is as it was.
    part_actuals(actual, new_cash, refunds, unidentified)
    for table, copied in zip((actual, new_cash, refunds, unidentified), copies):
        pd.testing.assert_frame_equal(table, copied)


# Refusals: a missing amount must not land in the known customers' cash.

CB_FIRST_10_2 = "{'company_code': 'TR02', 'origin_week': 10, 'h': 2, 'target_week': 12}"


def test_part_actuals_refuses_a_missing_a_row():
    # 1. A has no row for TR02 at (10, 2).
    new_cash = cb_new_cash()
    new_cash = new_cash[~((new_cash["company_code"] == "TR02") & (new_cash["origin_week"] == 10)
                          & (new_cash["h"] == 2))]
    # 2. Refused, not counted as known cash.
    with pytest.raises(ValueError) as error:
        cb_parts_run(new_cash=new_cash)
    assert str(error.value) == ("the side tables have no row for 1 rows of the actual cash "
                                f"(first: {CB_FIRST_10_2})")


def test_part_actuals_refuses_a_missing_g_row_and_counts_every_row_it_touches():
    # 1. G has no row for TR02 in week 12, which (10, 2) and (11, 1) both target.
    unidentified = cb_unidentified()
    unidentified = unidentified[~((unidentified["company_code"] == "TR02")
                                  & (unidentified["week"] == 12))]
    # 2. Two rows of the actual are short of it.
    with pytest.raises(ValueError) as error:
        cb_parts_run(unidentified=unidentified)
    assert str(error.value) == ("the side tables have no row for 2 rows of the actual cash "
                                f"(first: {CB_FIRST_10_2})")


def test_part_actuals_refuses_a_missing_r_row():
    # R has no row for TR04 in week 13, targeted only by (11, 2).
    refunds = cb_refunds()
    refunds = refunds[~((refunds["company_code"] == "TR04") & (refunds["week"] == 13))]
    with pytest.raises(ValueError) as error:
        cb_parts_run(refunds=refunds)
    assert str(error.value) == (
        "the side tables have no row for 1 rows of the actual cash "
        "(first: {'company_code': 'TR04', 'origin_week': 11, 'h': 2, 'target_week': 13})")


def test_part_actuals_refuses_a_company_with_no_side_rows_at_all():
    # 1. Table A lacks TR04: its four rows have nothing.
    new_cash = cb_new_cash()
    with pytest.raises(ValueError) as error:
        cb_parts_run(new_cash=new_cash[new_cash["company_code"] != "TR04"])
    # 2. The message counts the four rows and names the first.
    assert str(error.value) == (
        "the side tables have no row for 4 rows of the actual cash "
        "(first: {'company_code': 'TR04', 'origin_week': 10, 'h': 1, 'target_week': 11})")


def test_part_actuals_treats_a_nan_amount_like_a_missing_row():
    # The row is there, its amount is not.
    new_cash = cb_new_cash()
    new_cash.loc[(new_cash["company_code"] == "TR02") & (new_cash["origin_week"] == 10)
                 & (new_cash["h"] == 2), "new_customer_cash"] = float("nan")
    with pytest.raises(ValueError) as error:
        cb_parts_run(new_cash=new_cash)
    assert str(error.value) == ("the side tables have no row for 1 rows of the actual cash "
                                f"(first: {CB_FIRST_10_2})")


def test_part_actuals_a_real_zero_in_a_side_table_is_not_missing():
    # TR04's G is 0.0 in every week and TR04's R is 0.0 in week 12: no refusal, known unchanged.
    found = cb_part_row(cb_parts_run(), "TR04", 11, 1)
    assert found["actual_unidentified"] == 0.0
    assert found["actual_refunds"] == 0.0
    assert found["actual_known"] == 200.0


def test_part_actuals_ignores_side_rows_outside_the_actual():
    # 1. Side rows for a week, an origin and a company the actual does not have.
    new_cash = pd.concat([cb_new_cash(), pd.DataFrame(
        [("TR02", 10, 3, 13, 5555.0), ("TR99", 10, 1, 11, 5555.0)],
        columns=cb_new_cash().columns)], ignore_index=True)
    refunds = pd.concat([cb_refunds(), pd.DataFrame(
        [("TR02", 99, -5555.0), ("TR99", 11, -5555.0)], columns=cb_refunds().columns)],
        ignore_index=True)
    unidentified = pd.concat([cb_unidentified(), pd.DataFrame(
        [("TR02", 99, 5555.0), ("TR99", 11, 5555.0)], columns=cb_unidentified().columns)],
        ignore_index=True)
    # 2. Same result as without them.
    found = cb_parts_run(new_cash=new_cash, refunds=refunds, unidentified=unidentified)
    pd.testing.assert_frame_equal(found, cb_parts_run())


@pytest.mark.parametrize("table_name", ["new_cash", "refunds", "unidentified"])
def test_part_actuals_a_repeated_side_row_is_refused_not_counted_twice(table_name):
    # 1. Each side table with its first row twice.
    tables = {"new_cash": cb_new_cash(), "refunds": cb_refunds(),
              "unidentified": cb_unidentified()}
    table = tables[table_name]
    # 2. pandas' merge check refuses it (a repeat would double the rows of the actual).
    with pytest.raises(MergeError):
        cb_parts_run(**{table_name: pd.concat([table, table.iloc[[0]]], ignore_index=True)})


def test_part_actuals_the_order_of_the_side_rows_does_not_matter():
    # Side tables reversed: the same amounts on the same rows.
    found = cb_parts_run(new_cash=cb_new_cash().iloc[::-1], refunds=cb_refunds().iloc[::-1],
                         unidentified=cb_unidentified().iloc[::-1])
    pd.testing.assert_frame_equal(found, cb_parts_run())


# --- part_rows -------------------------------------------------------------------------------

def cb_small_rows():
    """Two companies, h 1 and 2; TR04 has only h = 2."""
    return pd.DataFrame({"company_code": ["TR02", "TR02", "TR04"], "h": [1, 2, 2],
                         "made": [10.0, 20.0, 5.0], "came": [4.0, 0.0, 0.0]})


def test_part_rows_exact_rows():
    # 1. TR02 h1 10/4; h2 20/0 so no ratio; pooled 30/4. TR04 has no h = 1, so no such row.
    found = part_rows("a method", "a part", cb_small_rows(), "made", "came", 2)
    # 2. NaN replaced by None so that the rows compare.
    cleaned = [{name: (None if isinstance(value, float) and np.isnan(value) else value)
                for name, value in made_row.items()} for made_row in found]
    assert cleaned == [
        {"method": "a method", "company_code": "TR02", "horizon": "1", "part": "a part",
         "forecast": 10.0, "actual": 4.0, "gap": 6.0, "ratio": 2.5},
        {"method": "a method", "company_code": "TR02", "horizon": "2", "part": "a part",
         "forecast": 20.0, "actual": 0.0, "gap": 20.0, "ratio": None},
        {"method": "a method", "company_code": "TR02", "horizon": "pooled", "part": "a part",
         "forecast": 30.0, "actual": 4.0, "gap": 26.0, "ratio": 7.5},
        {"method": "a method", "company_code": "TR04", "horizon": "2", "part": "a part",
         "forecast": 5.0, "actual": 0.0, "gap": 5.0, "ratio": None},
        {"method": "a method", "company_code": "TR04", "horizon": "pooled", "part": "a part",
         "forecast": 5.0, "actual": 0.0, "gap": 5.0, "ratio": None},
    ]


def test_part_rows_keeps_the_companies_in_the_order_they_come():
    found = part_rows("m", "p", cb_small_rows().iloc[::-1], "made", "came", 2)
    assert found[0]["company_code"] == "TR04"


def test_part_rows_empty_gives_no_rows():
    assert part_rows("m", "p", cb_small_rows().iloc[0:0], "made", "came", 2) == []


# --- component_bias: the exact numbers -------------------------------------------------------

CB_EVERY_CASE = [(part, company, horizon) for part in CB_PARTS_IN_ORDER
                 for company in CB_ORDER_COMPANIES for horizon in CB_ORDER_HORIZONS]


@pytest.fixture(scope="module")
def cb_bias():
    return cb_run()


@pytest.mark.parametrize("part, company, horizon", CB_EVERY_CASE)
def test_component_bias_hurdle_exact_numbers(cb_bias, part, company, horizon):
    found = cb_bias_row(cb_bias, HURDLE, part, company, horizon)
    cb_assert_numbers(found, CB_EXPECTED[part][(company, horizon)])


@pytest.mark.parametrize("company", CB_ORDER_COMPANIES)
@pytest.mark.parametrize("horizon", CB_ORDER_HORIZONS)
def test_component_bias_rule_exact_numbers(cb_bias, company, horizon):
    found = cb_bias_row(cb_bias, RULE, KNOWN_CUSTOMERS, company, horizon)
    cb_assert_numbers(found, CB_EXPECTED_RULE[(company, horizon)])


@pytest.mark.parametrize("company", CB_ORDER_COMPANIES)
@pytest.mark.parametrize("horizon", CB_ORDER_HORIZONS)
def test_component_bias_the_four_gaps_add_up_to_the_total_gap(cb_bias, company, horizon):
    # 1. The gaps of the four adding parts.
    gaps = [cb_bias_row(cb_bias, HURDLE, part, company, horizon)["gap"] for part in ADDING_PARTS]
    # 2. Their sum is the total's gap; the price level's is NOT part of the sum.
    total = cb_bias_row(cb_bias, HURDLE, PART_TOTAL, company, horizon)
    assert sum(gaps) == pytest.approx(total["gap"])


@pytest.mark.parametrize("company", CB_ORDER_COMPANIES)
@pytest.mark.parametrize("horizon", CB_ORDER_HORIZONS)
def test_component_bias_the_forecasts_and_actuals_add_up_too(cb_bias, company, horizon):
    made = [cb_bias_row(cb_bias, HURDLE, part, company, horizon)["forecast"]
            for part in ADDING_PARTS]
    came = [cb_bias_row(cb_bias, HURDLE, part, company, horizon)["actual"]
            for part in ADDING_PARTS]
    total = cb_bias_row(cb_bias, HURDLE, PART_TOTAL, company, horizon)
    assert sum(made) == pytest.approx(total["forecast"])
    assert sum(came) == pytest.approx(total["actual"])


@pytest.mark.parametrize("seed", range(12))
def test_component_bias_gaps_add_up_on_random_tables(seed):
    # 1. Random parts, levels and cash; the adding-up holds whatever the amounts are.
    generator = np.random.default_rng(seed)

    def random_parts():
        return [tuple(float(value) for value in generator.uniform(-500.0, 500.0, size=4))
                for _ in CB_PAIRS]

    by_company = {company: random_parts() for company in CB_COMPANIES}
    by_company["ALL"] = [tuple(a + b for a, b in zip(first, second)) for first, second in
                         zip(by_company["TR02"], by_company["TR04"])]
    projected = {key: float(generator.uniform(1.0, 9.0)) for key in CB_PROJECTED}
    actual = cb_actual()
    actual["actual"] = generator.uniform(0.0, 5000.0, size=len(actual))
    # 2. The sum of the four gaps is the total's gap in every row of the result.
    bias = cb_run(rolled=cb_rolled(by_company, projected), actual=actual)
    for company in CB_ORDER_COMPANIES:
        for horizon in CB_ORDER_HORIZONS:
            gaps = [cb_bias_row(bias, HURDLE, part, company, horizon)["gap"]
                    for part in ADDING_PARTS]
            total = cb_bias_row(bias, HURDLE, PART_TOTAL, company, horizon)["gap"]
            assert sum(gaps) == pytest.approx(total)


def test_component_bias_nominal_uses_the_projected_level_of_each_row(cb_bias):
    # TR02 known, h = 1: 400 x 2 (origin 10) + 300 x 4 (origin 11) = 2000. At the published
    # levels it would be 400 x 2.5 + 300 x 3.5 = 2050.
    assert cb_bias_row(cb_bias, HURDLE, KNOWN_CUSTOMERS, "TR02", "1")["forecast"] == 2000.0


def test_component_bias_price_level_forecast_is_the_total(cb_bias):
    for company in CB_ORDER_COMPANIES:
        for horizon in CB_ORDER_HORIZONS:
            price = cb_bias_row(cb_bias, HURDLE, PRICE_LEVEL, company, horizon)
            total = cb_bias_row(cb_bias, HURDLE, PART_TOTAL, company, horizon)
            assert price["forecast"] == total["forecast"]


def test_component_bias_price_level_actual_is_real_total_at_the_published_level(cb_bias):
    # TR02, h = 1: 402 x 2.5 + 305 x 3.5 = 1005 + 1067.5. Not any part of the actual cash.
    found = cb_bias_row(cb_bias, HURDLE, PRICE_LEVEL, "TR02", "1")
    assert found["actual"] == pytest.approx(402 * 2.5 + 305 * 3.5)


def test_component_bias_price_level_does_not_depend_on_the_actual_cash():
    # 1. All the actual cash doubled.
    actual = cb_actual()
    actual["actual"] = actual["actual"] * 2
    bias = cb_run(actual=actual)
    # 2. The price level row is what it was; the total's actual has doubled.
    found = cb_bias_row(bias, HURDLE, PRICE_LEVEL, "TR02", "1")
    cb_assert_numbers(found, CB_EXPECTED[PRICE_LEVEL][("TR02", "1")])
    assert cb_bias_row(bias, HURDLE, PART_TOTAL, "TR02", "1")["actual"] == 6000.0


def test_component_bias_price_level_gap_is_zero_when_projected_equals_published():
    # Each row's projected level is the published level of its target week.
    projected = {(10, 11): 2.5, (10, 12): 3.5, (11, 12): 3.5, (11, 13): 4.5}
    bias = cb_run(rolled=cb_rolled(projected=projected))
    gaps = bias.loc[bias["part"] == PRICE_LEVEL, "gap"].to_numpy()
    assert gaps == pytest.approx(np.zeros(9))


def test_component_bias_the_price_level_gap_is_only_part_of_the_total_gap(cb_bias):
    # -48.5 of -976 for TR02 at h = 1.
    price = cb_bias_row(cb_bias, HURDLE, PRICE_LEVEL, "TR02", "1")["gap"]
    total = cb_bias_row(cb_bias, HURDLE, PART_TOTAL, "TR02", "1")["gap"]
    assert price == pytest.approx(-48.5)
    assert total == pytest.approx(-976.0)


def test_component_bias_refunds_keep_their_negative_sign(cb_bias):
    refund_rows = cb_bias[cb_bias["part"] == REFUND_PART]
    assert (refund_rows["forecast"] < 0).all()
    assert (refund_rows["actual"] < 0).all()
    # Too many refunds forecast: the gap is negative, and it lowers the total.
    assert (refund_rows["gap"] < 0).all()
    # Two negatives: the ratio stays positive.
    assert (refund_rows["ratio"] > 0).all()


def test_component_bias_ratio_is_nan_where_g_is_all_zero_and_the_gap_stays():
    # 1. TR04's G is 0 in every week, but 1 of it is forecast at (10, 1): 2 at level 2.
    parts = {company: list(values) for company, values in CB_REAL_PARTS.items()}
    parts["TR04"][0] = (40.0, 1.0, 1.0, -1.0)
    parts["ALL"][0] = (440.0, 2.0, 3.0, -2.0)
    found = cb_bias_row(cb_run(rolled=cb_rolled(parts)), HURDLE, UNIDENTIFIED, "TR04", "1")
    # 2. No ratio, but the gap is there.
    assert np.isnan(found["ratio"])
    assert found["gap"] == 2.0


def test_component_bias_ratio_is_nan_for_new_customers_at_h_1_and_the_gap_stays(cb_bias):
    # 2 + 4 forecast against an A of 0 at h = 1, for each company.
    found = cb_bias_row(cb_bias, HURDLE, NEW_CUSTOMERS, "TR02", "1")
    assert found["actual"] == 0.0
    assert np.isnan(found["ratio"])
    assert found["gap"] == 6.0
    # h = 2 and pooled have an actual, so a ratio.
    assert not np.isnan(cb_bias_row(cb_bias, HURDLE, NEW_CUSTOMERS, "TR02", "2")["ratio"])
    assert not np.isnan(cb_bias_row(cb_bias, HURDLE, NEW_CUSTOMERS, "TR02", "pooled")["ratio"])


def test_component_bias_no_ratio_is_infinite(cb_bias):
    assert not np.isinf(cb_bias["ratio"].to_numpy()).any()


# --- component_bias: rolled None, the rule, the order ----------------------------------------

def test_component_bias_without_the_model_gives_only_the_rule_rows():
    # 1. No hurdle model.
    bias = cb_run(rolled=None)
    # 2. One part of one method, three companies, three horizons.
    assert set(bias["method"]) == {RULE}
    assert set(bias["part"]) == {KNOWN_CUSTOMERS}
    assert len(bias) == 9
    assert list(bias.columns) == COMPONENT_COLUMNS
    # 3. The same rows as with the model.
    with_model = cb_run()
    expected = with_model[with_model["method"] == RULE].reset_index(drop=True)
    pd.testing.assert_frame_equal(bias.reset_index(drop=True), expected)


def test_component_bias_without_the_model_does_not_read_the_levels():
    # Only the hurdle's price level row needs them.
    bias = cb_run(rolled=None, levels=pd.Series(dtype="float64"))
    assert len(bias) == 9


def test_component_bias_the_rule_part_is_invoice_cash_not_the_rules_forecast(cb_bias):
    # The rule's `forecast` is -1234 in every row of the builder.
    rule_rows = cb_bias[cb_bias["method"] == RULE]
    assert (rule_rows["forecast"] > 0).all()


def test_component_bias_the_rule_is_compared_with_the_known_customers_cash(cb_bias):
    # Not all the actual cash: the same actual as the hurdle's known customers' row.
    for company in CB_ORDER_COMPANIES:
        for horizon in CB_ORDER_HORIZONS:
            rule_row = cb_bias_row(cb_bias, RULE, KNOWN_CUSTOMERS, company, horizon)
            hurdle_row = cb_bias_row(cb_bias, HURDLE, KNOWN_CUSTOMERS, company, horizon)
            assert rule_row["actual"] == hurdle_row["actual"]


def test_component_bias_all_of_the_rule_is_its_own_all_rows():
    # An ALL row of the rule that is not the sum of its companies: read as it is.
    invoices = dict(CB_RULE_INVOICES)
    invoices["ALL"] = [1.0, 2.0, 3.0, 4.0]
    bias = cb_run(rolled=None, rule=cb_rule(invoices))
    assert cb_bias_row(bias, RULE, KNOWN_CUSTOMERS, "ALL", "pooled")["forecast"] == 10.0


def test_component_bias_all_of_the_hurdle_is_its_own_all_rows():
    # The ALL rows of the totals are read, not the companies added up: raise one ALL row's
    # customers and its total together (level 2 at (10, 1), so +1 real is +2 nominal).
    rolled = cb_totals_edited("customers", "ALL", 10, 1, 1.0)
    chosen = ((rolled.totals["company_code"] == "ALL") & (rolled.totals["origin_week"] == 10)
              & (rolled.totals["h"] == 1))
    rolled.totals.loc[chosen, "total"] += 2.0
    bias = cb_run(rolled=rolled)
    assert cb_bias_row(bias, HURDLE, KNOWN_CUSTOMERS, "ALL", "1")["forecast"] == 2322.0
    assert cb_bias_row(bias, HURDLE, KNOWN_CUSTOMERS, "TR02", "1")["forecast"] == 2000.0


def test_component_bias_column_order(cb_bias):
    assert list(cb_bias.columns) == COMPONENT_COLUMNS


def test_component_bias_row_order(cb_bias):
    # The hurdle's parts in order, each by company (TR02, TR04, ALL) and horizon; then the rule.
    expected = [(HURDLE, company, horizon, part) for part in CB_PARTS_IN_ORDER
                for company in CB_ORDER_COMPANIES for horizon in CB_ORDER_HORIZONS]
    expected += [(RULE, company, horizon, KNOWN_CUSTOMERS) for company in CB_ORDER_COMPANIES
                 for horizon in CB_ORDER_HORIZONS]
    columns = ["method", "company_code", "horizon", "part"]
    assert list(cb_bias[columns].itertuples(index=False, name=None)) == expected


def test_component_bias_the_horizon_column_is_text(cb_bias):
    assert all(isinstance(value, str) for value in cb_bias["horizon"])


def test_component_bias_weeks_without_actual_are_absent():
    # 1. Week 13 has no data: the actual has no row for (11, 2, 13). The forecast does, and its
    #    side rows and published level are there too; none of it may count.
    actual = cb_actual()
    actual = actual[actual["target_week"] != 13]
    bias = cb_run(actual=actual)
    # 2. TR02's known customers at h = 2 are only (10, 2, 12): 500 x 3 against 1932.
    found = cb_bias_row(bias, HURDLE, KNOWN_CUSTOMERS, "TR02", "2")
    cb_assert_numbers(found, (1500.0, 1932.0, -432.0, 1500 / 1932))
    # 3. Pooled: 2000 + 1500 against 2973 + 1932.
    found = cb_bias_row(bias, HURDLE, KNOWN_CUSTOMERS, "TR02", "pooled")
    cb_assert_numbers(found, (3500.0, 4905.0, -1405.0, 3500 / 4905))
    # 4. The rule's too: 1000 against 1932.
    found = cb_bias_row(bias, RULE, KNOWN_CUSTOMERS, "TR02", "2")
    cb_assert_numbers(found, (1000.0, 1932.0, -932.0, 1000 / 1932))


def test_component_bias_an_h_without_any_row_has_no_row_in_the_result():
    # Only h = 1 in the actual, horizons even 5: no row for h = 2, pooled is h = 1's.
    actual = cb_actual()
    bias = cb_run(actual=actual[actual["h"] == 1], horizons=5)
    assert set(bias["horizon"]) == {"1", "pooled"}
    one = cb_bias_row(bias, HURDLE, PART_TOTAL, "TR02", "1")
    pooled = cb_bias_row(bias, HURDLE, PART_TOTAL, "TR02", "pooled")
    assert one["forecast"] == pooled["forecast"] == 2024.0
    assert one["actual"] == pooled["actual"] == 3000.0


def test_component_bias_row_order_of_the_inputs_does_not_matter(cb_bias):
    # Every table reversed except the actual, which sets the order of the result.
    rolled = cb_rolled()
    rule = cb_rule()
    bias = cb_run(rolled=Forecast(totals=rolled.totals.iloc[::-1], customers=rolled.customers),
                  rule=RuleReference(totals=rule.totals.iloc[::-1], invoices=rule.invoices),
                  new_cash=cb_new_cash().iloc[::-1], refunds=cb_refunds().iloc[::-1],
                  unidentified=cb_unidentified().iloc[::-1], levels=cb_levels().iloc[::-1])
    pd.testing.assert_frame_equal(bias, cb_bias)


def test_component_bias_forecasts_beyond_the_actual_are_ignored():
    # 1. Forecast and rule rows for week 14, past the actual, with no published level.
    rolled = cb_rolled()
    extra_total = rolled.totals.iloc[[0]].assign(origin_week=11, h=3, target_week=14)
    rule = cb_rule()
    extra_rule = rule.totals.iloc[[0]].assign(origin_week=11, h=3, target_week=14)
    # 2. The result is the same as without them.
    bias = cb_run(
        rolled=Forecast(totals=pd.concat([rolled.totals, extra_total], ignore_index=True),
                        customers=rolled.customers),
        rule=RuleReference(totals=pd.concat([rule.totals, extra_rule], ignore_index=True),
                           invoices=rule.invoices))
    pd.testing.assert_frame_equal(bias, cb_run())


def test_component_bias_inputs_unchanged():
    # 1. The inputs, and a copy of each table in them.
    rolled, rule, actual, new_cash, refunds, unidentified, levels = (
        cb_rolled(), cb_rule(), cb_actual(), cb_new_cash(), cb_refunds(), cb_unidentified(),
        cb_levels())
    frames = [rolled.totals, rolled.customers, rule.totals, rule.invoices, actual, new_cash,
              refunds, unidentified]
    copies = [frame.copy() for frame in frames]
    levels_copy = levels.copy()
    # 2. Called, and each is as it was.
    component_bias(rolled, rule, actual, new_cash, refunds, unidentified, levels, CB_HORIZONS)
    for frame, copied in zip(frames, copies):
        pd.testing.assert_frame_equal(frame, copied)
    pd.testing.assert_series_equal(levels, levels_copy)


def test_component_bias_works_on_the_real_actual_totals():
    # 1. actual_totals on a small weekly table: week 11 has TR02 100 and TR04 10 less a refund
    #    of 3; week 12 is missing.
    weekly = make_weekly([("K1", "TR02", "C1", 11, 100.0, 0.0),
                          ("K2", "TR04", "C2", 11, 10.0, -3.0),
                          ("K1", "TR02", "C1", 12, 777.0, 0.0)])
    actual = actual_totals(weekly, make_weeks(13, missing={12}), make_levels(), CB_COMPANIES,
                           [10], 2)
    # 2. Only h = 1 has data; its parts are zero but for R of TR04 in week 11.
    new_cash = pd.DataFrame([("TR02", 10, 1, 11, 0.0), ("TR04", 10, 1, 11, 0.0)],
                            columns=["company_code", "origin_week", "h", "target_week",
                                     "new_customer_cash"])
    refunds = pd.DataFrame([("TR02", 11, 0.0), ("TR04", 11, -3.0)],
                           columns=["company_code", "week", "refunds"])
    unidentified = pd.DataFrame([("TR02", 11, 0.0), ("TR04", 11, 0.0)],
                                columns=["company_code", "week", "unidentified_cash"])
    rule = RuleReference(totals=pd.DataFrame(
        [("TR02", 10, 1, 11, 80.0), ("TR04", 10, 1, 11, 5.0), ("ALL", 10, 1, 11, 85.0)],
        columns=CB_KEYS + ["invoice_cash"]), invoices=pd.DataFrame())
    # 3. The known customers' cash is the gross receipts: 100 and 10, 110 together.
    bias = component_bias(None, rule, actual, new_cash, refunds, unidentified, make_levels(), 2)
    cb_assert_numbers(cb_bias_row(bias, RULE, KNOWN_CUSTOMERS, "TR04", "1"),
                      (5.0, 10.0, -5.0, 0.5))
    cb_assert_numbers(cb_bias_row(bias, RULE, KNOWN_CUSTOMERS, "ALL", "pooled"),
                      (85.0, 110.0, -25.0, 85 / 110))


# --- component_bias: the refusals ------------------------------------------------------------

@pytest.mark.parametrize("horizons", [0, -1])
def test_component_bias_refuses_horizons_below_one(horizons):
    cb_refused(f"horizons is {horizons}; horizons start at 1", horizons=horizons)


def test_component_bias_refuses_horizons_below_one_before_anything_else():
    # Nothing else is looked at: every table is None.
    with pytest.raises(ValueError) as error:
        component_bias(None, None, None, None, None, None, None, 0)
    assert str(error.value) == "horizons is 0; horizons start at 1"


def test_component_bias_refuses_an_h_above_horizons():
    cb_refused("the actual cash has rows for h [2]; horizons is 1", horizons=1)


def test_component_bias_refuses_an_h_below_one_and_sorts_the_list():
    # 1. Two rows of the actual with h 0 and 7.
    actual = cb_actual()
    actual.loc[0, "h"] = 7
    actual.loc[1, "h"] = 0
    # 2. Both named, sorted.
    cb_refused("the actual cash has rows for h [0, 7]; horizons is 2", actual=actual)


def test_component_bias_refuses_a_row_with_no_hurdle_forecast():
    # 1. The Forecast has no row for TR02 at (10, 2).
    rolled = cb_rolled()
    kept = ~((rolled.totals["company_code"] == "TR02") & (rolled.totals["origin_week"] == 10)
             & (rolled.totals["h"] == 2))
    rolled = Forecast(totals=rolled.totals[kept], customers=rolled.customers)
    # 2. Refused, with the count and the first row.
    cb_refused("the hurdle model has no forecast for 1 rows of the actual cash "
               f"(first: {CB_FIRST_10_2})", rolled=rolled)


def test_component_bias_refuses_a_nan_total_as_no_forecast():
    rolled = cb_totals_edited("total", "TR02", 10, 2, float("nan"))
    cb_refused("the hurdle model has no forecast for 1 rows of the actual cash "
               f"(first: {CB_FIRST_10_2})", rolled=rolled)


def test_component_bias_refuses_a_row_with_no_rule_forecast():
    # 1. The rule has no row for TR02 at (10, 2).
    rule = cb_rule()
    kept = ~((rule.totals["company_code"] == "TR02") & (rule.totals["origin_week"] == 10)
             & (rule.totals["h"] == 2))
    rule = RuleReference(totals=rule.totals[kept], invoices=rule.invoices)
    # 2. Refused, with or without the hurdle model.
    expected = ("the financial rule has no forecast for 1 rows of the actual cash "
                f"(first: {CB_FIRST_10_2})")
    cb_refused(expected, rule=rule)
    cb_refused(expected, rule=rule, rolled=None)


def test_component_bias_refuses_a_nan_invoice_cash_as_no_forecast():
    rule = cb_rule()
    rule.totals.loc[(rule.totals["company_code"] == "ALL") & (rule.totals["origin_week"] == 11)
                    & (rule.totals["h"] == 2), "invoice_cash"] = float("nan")
    cb_refused("the financial rule has no forecast for 1 rows of the actual cash "
               "(first: {'company_code': 'ALL', 'origin_week': 11, 'h': 2, 'target_week': 13})",
               rule=rule)


def test_component_bias_refuses_a_side_table_gap_with_its_message():
    new_cash = cb_new_cash()
    cb_refused("the side tables have no row for 4 rows of the actual cash "
               "(first: {'company_code': 'TR04', 'origin_week': 10, 'h': 1, 'target_week': 11})",
               new_cash=new_cash[new_cash["company_code"] != "TR04"])


CB_NOT_ADDING_UP = ("the parts do not add up to the total in 1 rows "
                    "(first: {'company_code': 'TR02', 'origin_week': 10, 'h': 1, "
                    "'target_week': 11})")


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_component_bias_a_difference_just_within_the_tolerance_is_let_through(sign):
    # 1. The total of one row a shade less than TIE_TOLERANCE off its parts.
    rolled = cb_totals_edited("total", "TR02", 10, 1, sign * TIE_TOLERANCE * 0.99)
    # 2. No refusal; and the total's row holds the total as it was given.
    found = cb_bias_row(cb_run(rolled=rolled), HURDLE, PART_TOTAL, "TR02", "1")
    assert found["forecast"] == pytest.approx(2024.0 + sign * TIE_TOLERANCE * 0.99)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_component_bias_a_difference_just_beyond_the_tolerance_is_refused(sign):
    rolled = cb_totals_edited("total", "TR02", 10, 1, sign * TIE_TOLERANCE * 1.01)
    cb_refused(CB_NOT_ADDING_UP, rolled=rolled)


def test_component_bias_the_parts_must_add_up_row_by_row_not_in_sum():
    # Two rows off in opposite directions: their sum is right, each row is wrong.
    rolled = cb_totals_edited("total", "TR02", 10, 1, 5.0)
    chosen = ((rolled.totals["company_code"] == "TR02") & (rolled.totals["origin_week"] == 11)
              & (rolled.totals["h"] == 1))
    rolled.totals.loc[chosen, "total"] -= 5.0
    cb_refused("the parts do not add up to the total in 2 rows "
               "(first: {'company_code': 'TR02', 'origin_week': 10, 'h': 1, "
               "'target_week': 11})", rolled=rolled)


@pytest.mark.parametrize("column", ["customers", "new_customer_cash", "unidentified_cash",
                                    "refunds", "price_level"])
def test_component_bias_any_column_off_breaks_the_adding_up(column):
    # One real part or the level of one row moved; total unchanged.
    cb_refused(CB_NOT_ADDING_UP, rolled=cb_totals_edited(column, "TR02", 10, 1, 1.0))


def test_component_bias_real_total_is_not_part_of_the_adding_up():
    # real_total only feeds the price level's actual: moving it moves that, nothing else.
    bias = cb_run(rolled=cb_totals_edited("real_total", "TR02", 10, 1, 10.0))
    # +10 real at week 11's published 2.5 is +25.
    found = cb_bias_row(bias, HURDLE, PRICE_LEVEL, "TR02", "1")
    assert found["actual"] == pytest.approx(2072.5 + 25.0)
    cb_assert_numbers(cb_bias_row(bias, HURDLE, PART_TOTAL, "TR02", "1"),
                      CB_EXPECTED[PART_TOTAL][("TR02", "1")])


def test_component_bias_refuses_a_target_week_with_no_published_level():
    cb_refused("no published price level for weeks [13]", levels=cb_levels().drop(13))


def test_component_bias_names_every_week_with_no_published_level_sorted():
    cb_refused("no published price level for weeks [11, 13]", levels=cb_levels().drop([13, 11]))


def test_component_bias_a_nan_published_level_counts_as_none():
    levels = cb_levels()
    levels[12] = float("nan")
    cb_refused("no published price level for weeks [12]", levels=levels)


def test_component_bias_a_week_without_actual_needs_no_published_level():
    # Week 13 has neither a level nor a row in the actual: no refusal.
    actual = cb_actual()
    bias = cb_run(actual=actual[actual["target_week"] != 13], levels=cb_levels().drop(13))
    assert not bias.empty


def test_component_bias_refusal_order_h_then_side_tables_then_hurdle():
    # 1. An h outside and a side table short: the h comes first.
    actual = cb_actual()
    actual.loc[0, "h"] = 7
    new_cash = cb_new_cash()
    no_tr04_new_cash = new_cash[new_cash["company_code"] != "TR04"]
    cb_refused("the actual cash has rows for h [7]; horizons is 2", actual=actual,
               new_cash=no_tr04_new_cash)
    # 2. A side table short and no forecast: the side table comes first.
    rolled = cb_rolled()
    no_tr04_rolled = Forecast(totals=rolled.totals[rolled.totals["company_code"] != "TR04"],
                              customers=rolled.customers)
    with pytest.raises(ValueError, match="^the side tables have no row"):
        cb_run(new_cash=no_tr04_new_cash, rolled=no_tr04_rolled)


def test_component_bias_refusal_order_hurdle_then_adding_up_then_level_then_rule():
    # 1. All of them wrong at once: no hurdle forecast first.
    rolled = cb_totals_edited("customers", "TR02", 10, 1, 1.0)
    without_row = Forecast(
        totals=rolled.totals[~((rolled.totals["company_code"] == "TR04")
                               & (rolled.totals["origin_week"] == 11))],
        customers=rolled.customers)
    rule = cb_rule()
    no_rule = RuleReference(totals=rule.totals.iloc[1:], invoices=rule.invoices)
    no_level = cb_levels().drop(13)
    with pytest.raises(ValueError, match="^the hurdle model has no forecast"):
        cb_run(rolled=without_row, rule=no_rule, levels=no_level)
    # 2. Then the parts not adding up.
    with pytest.raises(ValueError, match="^the parts do not add up"):
        cb_run(rolled=rolled, rule=no_rule, levels=no_level)
    # 3. Then the published level.
    with pytest.raises(ValueError, match="^no published price level"):
        cb_run(rule=no_rule, levels=no_level)
    # 4. Then the rule.
    with pytest.raises(ValueError, match="^the financial rule has no forecast"):
        cb_run(rule=no_rule)


@pytest.mark.parametrize("repeated", ["rolled", "rule"])
def test_component_bias_a_repeated_forecast_row_is_refused(repeated):
    # A row twice would add its amount twice; pandas' merge check refuses it.
    rolled, rule = cb_rolled(), cb_rule()
    if repeated == "rolled":
        rolled = Forecast(totals=pd.concat([rolled.totals, rolled.totals.iloc[[0]]],
                                           ignore_index=True), customers=rolled.customers)
    else:
        rule = RuleReference(totals=pd.concat([rule.totals, rule.totals.iloc[[0]]],
                                              ignore_index=True), invoices=rule.invoices)
    with pytest.raises(MergeError):
        cb_run(rolled=rolled, rule=rule)


# --- payer_row -------------------------------------------------------------------------------

def ps_row_table(p, expected, paid, actual_real):
    return pd.DataFrame({"p": p, "expected": expected, "paid": paid, "actual_real": actual_real})


def test_payer_row_exact_numbers():
    # 1. p 0.5 + 0.5 = 1 expected against 2 who paid; 30 expected against 80 that came:
    #    per payer 30 against 40.
    found = payer_row("TR02", "1", ps_row_table([0.5, 0.5, 0.0], [10.0, 20.0, 0.0],
                                                [True, True, False], [30.0, 50.0, 0.0]))
    # 2. p ratio 1 / 2, m ratio 30 / 40, p x m ratio 30 / 80 = 1 / 2 x 3 / 4.
    assert found == {"company_code": "TR02", "horizon": "1", "rows": 3,
                     "expected_payers": 1.0, "actual_payers": 2, "p_ratio": 0.5,
                     "expected_per_payer": 30.0, "actual_per_payer": 40.0, "m_ratio": 0.75,
                     "pxm_ratio": 0.375}


def test_payer_row_the_counts_are_plain_ints():
    found = payer_row("A", "1", ps_row_table([0.5], [1.0], [True], [1.0]))
    assert type(found["actual_payers"]) is int
    assert type(found["rows"]) is int


def test_payer_row_nobody_paid_gives_nan_not_infinity():
    # 1. Nobody paid: nothing to divide by in p, in the actual per payer and in p x m.
    found = payer_row("A", "1", ps_row_table([0.5], [10.0], [False], [0.0]))
    # 2. What can be computed is there.
    assert found["expected_payers"] == 0.5
    assert found["expected_per_payer"] == 20.0
    assert found["actual_payers"] == 0
    # 3. The ratios are NaN.
    for name in ("p_ratio", "actual_per_payer", "m_ratio", "pxm_ratio"):
        assert np.isnan(found[name]), name


def test_payer_row_nobody_expected_gives_a_zero_p_ratio_and_no_m_ratio():
    # p is 0 in every row, one customer paid all the same.
    found = payer_row("A", "1", ps_row_table([0.0], [0.0], [True], [5.0]))
    assert found["p_ratio"] == 0.0
    assert np.isnan(found["expected_per_payer"])
    assert np.isnan(found["m_ratio"])
    assert found["pxm_ratio"] == 0.0


def test_payer_row_on_no_rows_gives_nans():
    found = payer_row("A", "1", ps_row_table([], [], [], []))
    assert found["rows"] == 0
    assert found["expected_payers"] == 0.0
    for name in ("p_ratio", "expected_per_payer", "actual_per_payer", "m_ratio", "pxm_ratio"):
        assert np.isnan(found[name]), name


# --- payer_split: the worked example ---------------------------------------------------------
#
# Horizons 2; published levels: week 11 2.0, week 12 4.0, week 13 5.0, and week 13 has no data.
#
#   key company  origin h target   p     m   p x m   gross   real paid?
#   A   TR02     10     1 11       0.5   100 50      300     150  yes
#   B   TR02     10     1 11       0.25  40  10      (none)  0    no: no weekly row
#   C   TR04     10     1 11       0.5   20  10      0       0    no: refund only
#   A   TR02     10     2 12       0.5   60  30      400     100  yes
#   B   TR02     10     2 12       0.75  40  30      40      10   yes
#   C   TR04     10     2 12       1.0   20  20      80      20   yes
#   A   TR02     11     1 12       1.0   50  50      400     100  yes
#   B   TR02     11     1 12       0.25  40  10      40      10   yes
#   C   TR04     11     1 12       0.5   40  20      80      20   yes
#   A   TR02     11     2 13  left out: week 13 has no data, whatever the weekly amounts say.
#
# TR02 h1: p 2 / 3 payers; p x m 120, per payer 60; real 260, per payer 260 / 3.
# TR02 h2: p 1.25 / 2; p x m 60, per payer 48; real 110, per payer 55.
# TR04 h1: p 1 / 1; p x m 30, per payer 30; real 20, per payer 20.
# TR04 h2: p 1 / 1; p x m 20, per payer 20; real 20, per payer 20.
# ALL  h1: p 3 / 4; p x m 150, per payer 50; real 280, per payer 70.
# ALL  h2: p 2.25 / 3; p x m 80, per payer 80 / 2.25; real 130, per payer 130 / 3.

PS_HORIZONS = 2
PS_ROWS = [("A", "TR02", 10, 1, 11, 0.5, 100.0), ("B", "TR02", 10, 1, 11, 0.25, 40.0),
           ("C", "TR04", 10, 1, 11, 0.5, 20.0), ("A", "TR02", 10, 2, 12, 0.5, 60.0),
           ("B", "TR02", 10, 2, 12, 0.75, 40.0), ("C", "TR04", 10, 2, 12, 1.0, 20.0),
           ("A", "TR02", 11, 1, 12, 1.0, 50.0), ("B", "TR02", 11, 1, 12, 0.25, 40.0),
           ("C", "TR04", 11, 1, 12, 0.5, 40.0), ("A", "TR02", 11, 2, 13, 0.9, 1000.0)]
PS_LEVELS = {11: 2.0, 12: 4.0, 13: 5.0}

# (company, horizon) -> rows, expected payers, actual payers, p ratio, expected per payer,
# actual per payer, m ratio, p x m ratio.
PS_EXPECTED = {
    ("TR02", "1"): (4, 2.0, 3, 2 / 3, 60.0, 260 / 3, 180 / 260, 120 / 260),
    ("TR02", "2"): (2, 1.25, 2, 0.625, 48.0, 55.0, 48 / 55, 60 / 110),
    ("TR02", "pooled"): (6, 3.25, 5, 0.65, 180 / 3.25, 74.0, (180 / 3.25) / 74, 180 / 370),
    ("TR04", "1"): (2, 1.0, 1, 1.0, 30.0, 20.0, 1.5, 1.5),
    ("TR04", "2"): (1, 1.0, 1, 1.0, 20.0, 20.0, 1.0, 1.0),
    ("TR04", "pooled"): (3, 2.0, 2, 1.0, 25.0, 20.0, 1.25, 1.25),
    ("ALL", "1"): (6, 3.0, 4, 0.75, 50.0, 70.0, 5 / 7, 150 / 280),
    ("ALL", "2"): (3, 2.25, 3, 0.75, 80 / 2.25, 130 / 3, (80 / 2.25) / (130 / 3), 80 / 130),
    ("ALL", "pooled"): (9, 5.25, 7, 0.75, 230 / 5.25, 410 / 7, (230 / 5.25) / (410 / 7),
                        230 / 410),
}
PS_ORDER = [("TR02", "1"), ("TR02", "2"), ("TR02", "pooled"), ("TR04", "1"), ("TR04", "2"),
            ("TR04", "pooled"), ("ALL", "1"), ("ALL", "2"), ("ALL", "pooled")]


def ps_customers(records=None):
    """records: (key, company_code, origin_week, h, target_week, p, m); expected is p x m."""
    records = PS_ROWS if records is None else records
    frame = pd.DataFrame(records, columns=["key", "company_code", "origin_week", "h",
                                           "target_week", "p", "m"])
    frame["customer_id"] = frame["key"].str.lower()
    frame["expected"] = frame["p"] * frame["m"]
    return frame[CUSTOMER_COLUMNS]


def ps_weekly():
    """B has no row in week 11; C's row there is a refund only; week 13 has a decoy; other keys
    and other weeks would show if they were read."""
    return make_weekly([("A", "TR02", "a", 11, 300.0, 0.0),
                        ("C", "TR04", "c", 11, 0.0, -5.0),
                        ("A", "TR02", "a", 12, 400.0, -7.0),
                        ("B", "TR02", "b", 12, 40.0, 0.0),
                        ("C", "TR04", "c", 12, 80.0, 0.0),
                        ("A", "TR02", "a", 13, 99999.0, 0.0),
                        ("A", "TR02", "a", 10, 88888.0, 0.0),
                        ("B", "TR02", "b", 14, 77777.0, 0.0),
                        ("not known", "TR02", None, 11, 66666.0, 0.0)])


def ps_weeks():
    return make_weeks(14, missing={13})


def ps_levels():
    return pd.Series(PS_LEVELS)


def ps_run(**changes):
    arguments = dict(customers=ps_customers(), weekly=ps_weekly(), weeks=ps_weeks(),
                     levels=ps_levels(), horizons=PS_HORIZONS)
    arguments.update(changes)
    return payer_split(**arguments)


def ps_row(split, company, horizon):
    selected = split[(split["company_code"] == company) & (split["horizon"] == horizon)]
    assert len(selected) == 1, (company, horizon, len(selected))
    return selected.iloc[0]


def ps_refused(expected_message, **changes):
    with pytest.raises(ValueError) as error:
        ps_run(**changes)
    assert str(error.value) == expected_message


@pytest.fixture(scope="module")
def ps_split():
    return ps_run()


@pytest.mark.parametrize("company, horizon", PS_ORDER)
def test_payer_split_exact_numbers(ps_split, company, horizon):
    found = ps_row(ps_split, company, horizon)
    rows_count, expected_payers, actual_payers, p_ratio, expected_each, actual_each, m_ratio, \
        pxm_ratio = PS_EXPECTED[(company, horizon)]
    assert found["rows"] == rows_count
    assert found["expected_payers"] == pytest.approx(expected_payers)
    assert found["actual_payers"] == actual_payers
    assert found["p_ratio"] == pytest.approx(p_ratio)
    assert found["expected_per_payer"] == pytest.approx(expected_each)
    assert found["actual_per_payer"] == pytest.approx(actual_each)
    assert found["m_ratio"] == pytest.approx(m_ratio)
    assert found["pxm_ratio"] == pytest.approx(pxm_ratio)


@pytest.mark.parametrize("company, horizon", PS_ORDER)
def test_payer_split_p_ratio_times_m_ratio_is_pxm_ratio(ps_split, company, horizon):
    found = ps_row(ps_split, company, horizon)
    assert found["p_ratio"] * found["m_ratio"] == pytest.approx(found["pxm_ratio"])


@pytest.mark.parametrize("seed", range(12))
def test_payer_split_the_identity_holds_on_random_tables(seed):
    # 1. Random p, m and receipts for 40 customers, all in week 11; about half pay, and the
    #    first of each company always does.
    generator = np.random.default_rng(seed)
    records = [(f"K{i}", "TR02" if i % 2 else "TR04", 10, 1, 11,
                float(generator.uniform(0.01, 1.0)), float(generator.uniform(1.0, 500.0)))
               for i in range(40)]
    weekly_rows = [(f"K{i}", "TR02" if i % 2 else "TR04", f"k{i}", 11,
                    float(generator.uniform(1.0, 900.0)), 0.0)
                   for i in range(40) if generator.random() < 0.5 or i < 2]
    # 2. The identity in every row of the result.
    split = ps_run(customers=ps_customers(records), weekly=make_weekly(weekly_rows), horizons=1)
    products = split["p_ratio"] * split["m_ratio"]
    assert products.to_numpy() == pytest.approx(split["pxm_ratio"].to_numpy())


def test_payer_split_columns_and_row_order(ps_split):
    assert list(ps_split.columns) == PAYER_COLUMNS
    assert list(zip(ps_split["company_code"], ps_split["horizon"])) == PS_ORDER


def test_payer_split_companies_are_sorted_whatever_order_they_come_in():
    # TR04 first in the customers: TR02 still first in the result, ALL last.
    split = ps_run(customers=ps_customers().iloc[::-1])
    assert list(zip(split["company_code"], split["horizon"])) == PS_ORDER


def test_payer_split_customer_order_and_index_do_not_matter(ps_split):
    # The rows reversed, on an index that is nothing like a range.
    customers = ps_customers().iloc[::-1]
    customers.index = range(1000, 1000 + 10 * len(customers), 10)
    pd.testing.assert_frame_equal(ps_run(customers=customers), ps_split)


def test_payer_split_a_customer_with_no_weekly_row_paid_zero_and_is_still_in_the_rows():
    # 1. Only customer B at (10, 1, 11): week 11 has data, B has no weekly row there.
    split = ps_run(customers=ps_customers([("B", "TR02", 10, 1, 11, 0.25, 40.0)]))
    found = ps_row(split, "TR02", "1")
    # 2. In the rows and in the expected payers, not among those who paid.
    assert found["rows"] == 1
    assert found["expected_payers"] == 0.25
    assert found["actual_payers"] == 0


def test_payer_split_a_refund_only_key_week_is_not_a_payer():
    # C's row in week 11 has gross receipts 0 and refunds -5.
    split = ps_run(customers=ps_customers([("C", "TR04", 10, 1, 11, 0.5, 20.0)]))
    found = ps_row(split, "TR04", "1")
    assert found["rows"] == 1
    assert found["actual_payers"] == 0
    assert np.isnan(found["actual_per_payer"])


def test_payer_split_refunds_are_not_in_the_receipts():
    # A's -7 refund in week 12 does not touch the 400 of gross receipts: 400 / 4 = 100.
    split = ps_run(customers=ps_customers([("A", "TR02", 10, 2, 12, 0.5, 60.0)]))
    assert ps_row(split, "TR02", "2")["actual_per_payer"] == 100.0


def test_payer_split_only_the_target_week_of_each_key_is_read():
    # Other weeks of A (10, 13) and an unknown key in week 11 hold decoys of tens of thousands.
    split = ps_run(customers=ps_customers([("A", "TR02", 10, 1, 11, 0.5, 100.0)]))
    assert ps_row(split, "TR02", "1")["actual_per_payer"] == 150.0


def test_payer_split_receipts_are_deflated_by_the_target_weeks_level():
    # 300 in week 11 at level 2, not at the origin's week or at week 12's level.
    split = ps_run(customers=ps_customers([("A", "TR02", 10, 1, 11, 0.5, 100.0)]),
                   levels=pd.Series({10: 10.0, 11: 2.0, 12: 7.0, 13: 1.0}))
    assert ps_row(split, "TR02", "1")["actual_per_payer"] == 150.0


def test_payer_split_expected_is_not_deflated_again():
    # p x m is already in TRY of week 0: 0.5 x 100 = 50, per expected payer 100, at any level.
    split = ps_run(customers=ps_customers([("A", "TR02", 10, 1, 11, 0.5, 100.0)]),
                   levels=pd.Series({11: 50.0}))
    assert ps_row(split, "TR02", "1")["expected_per_payer"] == 100.0


def test_payer_split_uses_the_expected_column_not_p_times_m():
    # A decoy m: `expected` says 50, p x m would say 4.
    customers = ps_customers([("A", "TR02", 10, 1, 11, 0.5, 100.0)])
    customers["m"] = 8.0
    split = ps_run(customers=customers)
    assert ps_row(split, "TR02", "1")["expected_per_payer"] == 100.0


def test_payer_split_a_missing_target_week_is_left_out_not_nobody_paid(ps_split):
    # 1. The result equals the one without the row for the missing week 13.
    without = ps_run(customers=ps_customers(PS_ROWS[:-1]))
    pd.testing.assert_frame_equal(without, ps_split)
    # 2. Its p of 0.9 is not in the expected payers; had it counted as nobody paid, it would be.
    assert ps_row(ps_split, "TR02", "2")["expected_payers"] == 1.25


def test_payer_split_a_target_week_past_the_calendar_is_left_out(ps_split):
    # Week 15 is past the calendar: no data, no level either, and no refusal.
    customers = ps_customers(PS_ROWS + [("A", "TR02", 13, 2, 15, 0.9, 1000.0)])
    pd.testing.assert_frame_equal(ps_run(customers=customers), ps_split)


def test_payer_split_a_missing_week_needs_no_published_level(ps_split):
    # Week 13 is left out before the deflation, so its level is not asked for.
    pd.testing.assert_frame_equal(ps_run(levels=ps_levels().drop(13)), ps_split)


def test_payer_split_no_payers_gives_nan_not_infinity_or_an_exception():
    # 1. Nobody in week 11 paid, in either company, with warnings turned into errors.
    customers = ps_customers([("B", "TR02", 10, 1, 11, 0.25, 40.0),
                              ("C", "TR04", 10, 1, 11, 0.5, 20.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        split = ps_run(customers=customers)
    # 2. NaN in every row of the result.
    for name in ("p_ratio", "actual_per_payer", "m_ratio", "pxm_ratio"):
        assert split[name].isna().all(), name
    assert not np.isinf(split.select_dtypes("number").to_numpy(dtype="float64")).any()
    assert (split["actual_payers"] == 0).all()


def test_payer_split_an_h_without_rows_has_no_row_but_the_pooled_one():
    # Only h = 2 rows, horizons 3: no row for h = 1 or 3, in any company.
    customers = ps_customers([record for record in PS_ROWS if record[3] == 2])
    split = ps_run(customers=customers, horizons=3)
    assert set(split["horizon"]) == {"2", "pooled"}
    two = ps_row(split, "ALL", "2")
    pooled = ps_row(split, "ALL", "pooled")
    assert two["rows"] == pooled["rows"] == 3
    assert two["expected_payers"] == pooled["expected_payers"]


def test_payer_split_a_company_without_rows_at_some_h_keeps_the_others():
    # TR04 has only h = 1, TR02 both.
    customers = ps_customers([record for record in PS_ROWS
                              if record[1] == "TR02" or record[3] == 1])
    split = ps_run(customers=customers)
    assert list(zip(split["company_code"], split["horizon"])) == [
        ("TR02", "1"), ("TR02", "2"), ("TR02", "pooled"), ("TR04", "1"), ("TR04", "pooled"),
        ("ALL", "1"), ("ALL", "2"), ("ALL", "pooled")]


def test_payer_split_all_counts_every_company(ps_split):
    # ALL's counts are the companies added up, in each horizon.
    for horizon in ("1", "2", "pooled"):
        first = ps_row(ps_split, "TR02", horizon)
        second = ps_row(ps_split, "TR04", horizon)
        consolidated = ps_row(ps_split, "ALL", horizon)
        assert consolidated["rows"] == first["rows"] + second["rows"]
        assert consolidated["actual_payers"] == first["actual_payers"] + second["actual_payers"]
        assert consolidated["expected_payers"] == pytest.approx(
            first["expected_payers"] + second["expected_payers"])


def test_payer_split_the_same_key_week_serves_two_origins(ps_split):
    # A's 400 in week 12 counts at (10, 2) and at (11, 1): once per forecast of it.
    assert ps_row(ps_split, "TR02", "1")["actual_payers"] == 3
    assert ps_row(ps_split, "TR02", "2")["actual_payers"] == 2


def test_payer_split_the_counts_are_whole_numbers(ps_split):
    assert ps_split["actual_payers"].dtype.kind == "i"
    assert ps_split["rows"].dtype.kind == "i"


def test_payer_split_inputs_unchanged():
    # 1. The inputs and a copy of each.
    customers, weekly, weeks, levels = ps_customers(), ps_weekly(), ps_weeks(), ps_levels()
    frames = [customers, weekly, weeks]
    copies = [frame.copy() for frame in frames]
    levels_copy = levels.copy()
    # 2. Called, and each is as it was.
    payer_split(customers, weekly, weeks, levels, PS_HORIZONS)
    for frame, copied in zip(frames, copies):
        pd.testing.assert_frame_equal(frame, copied)
    pd.testing.assert_series_equal(levels, levels_copy)


# --- payer_split: the refusals ---------------------------------------------------------------

@pytest.mark.parametrize("horizons", [0, -1])
def test_payer_split_refuses_horizons_below_one(horizons):
    ps_refused(f"horizons is {horizons}; horizons start at 1", horizons=horizons)


def test_payer_split_refuses_an_h_above_horizons():
    ps_refused("customer forecasts for h [2]; horizons is 1", horizons=1)


def test_payer_split_refuses_an_h_below_one_and_sorts_the_list():
    customers = ps_customers()
    customers.loc[0, "h"] = 9
    customers.loc[1, "h"] = 0
    ps_refused("customer forecasts for h [0, 9]; horizons is 2", customers=customers)


def test_payer_split_refuses_an_h_outside_even_in_a_week_without_data():
    # The checks read every row, before the weeks without data are left out.
    customers = ps_customers(PS_ROWS[:-1] + [("A", "TR02", 11, 3, 13, 0.9, 1000.0)])
    ps_refused("customer forecasts for h [3]; horizons is 2", customers=customers)


def test_payer_split_refuses_a_repeated_key_origin_h():
    # 1. The third row of the table, twice.
    customers = ps_customers(PS_ROWS + [PS_ROWS[2]])
    # 2. Named by its key, origin and h.
    ps_refused("1 customer forecasts are there twice "
               "(first: {'key': 'C', 'origin_week': 10, 'h': 1})", customers=customers)


def test_payer_split_counts_every_repeat():
    customers = ps_customers(PS_ROWS + [PS_ROWS[0], PS_ROWS[0], PS_ROWS[1]])
    ps_refused("3 customer forecasts are there twice "
               "(first: {'key': 'A', 'origin_week': 10, 'h': 1})", customers=customers)


def test_payer_split_refuses_no_rows_in_a_week_with_data():
    # Every row is in week 13, which has no data.
    customers = ps_customers([("A", "TR02", 11, 2, 13, 0.9, 1000.0)])
    ps_refused("there are no customer forecasts in a week with data", customers=customers)


def test_payer_split_refuses_an_empty_customer_table():
    ps_refused("there are no customer forecasts in a week with data",
               customers=ps_customers([]))


def test_payer_split_refuses_a_week_with_data_and_no_level():
    # As `deflate` does.
    ps_refused("no price level for weeks [11]", levels=ps_levels().drop(11))


def test_payer_split_refusal_order():
    # Horizons first, then the h outside, then the repeat, and only then no rows with data.
    repeated = ps_customers([("A", "TR02", 11, 2, 13, 0.9, 1000.0)] * 2)
    ps_refused("horizons is 0; horizons start at 1", customers=repeated, horizons=0)
    ps_refused("customer forecasts for h [2]; horizons is 1", customers=repeated, horizons=1)
    ps_refused("1 customer forecasts are there twice "
               "(first: {'key': 'A', 'origin_week': 11, 'h': 2})", customers=repeated)


def test_payer_split_the_same_key_at_another_origin_or_h_is_no_repeat(ps_split):
    # The worked example has A at (10, 1), (10, 2), (11, 1), (11, 2): no refusal.
    assert ps_row(ps_split, "ALL", "pooled")["rows"] == 9


def test_payer_split_a_repeated_key_week_of_the_weekly_table_is_refused():
    # Receipts twice would count the customer twice; pandas' merge check refuses it.
    weekly = ps_weekly()
    with pytest.raises(MergeError):
        ps_run(weekly=pd.concat([weekly, weekly.iloc[[0]]], ignore_index=True))


# --- component_bias: the shape of what is not there ------------------------------------------

def test_component_bias_an_empty_actual_gives_an_empty_table_with_the_columns():
    bias = cb_run(actual=cb_actual().iloc[0:0])
    assert bias.empty
    assert list(bias.columns) == COMPONENT_COLUMNS


def test_component_bias_a_repeated_row_of_the_actual_is_refused():
    actual = cb_actual()
    with pytest.raises(MergeError):
        cb_run(actual=pd.concat([actual, actual.iloc[[0]]], ignore_index=True))


# =============================================================================================
# U14.6, `with_receipts`, `coherence_row`, `balance_coherence`: does each customer's forecast over
# the h weeks of an origin fit what it owes at that origin, with the same measures on the actual
# cash? A report only.
#
# The worked example. Horizons 2, factor 1.5; published levels: week 10 1.0, week 11 2.0, week 12
# 4.0, week 13 5.0, week 14 8.0; week 13 has no data, so origin 11 (targets 12, 13) is dropped
# whole. Balances are read at h = 1; the h = 2 rows of the table hold a decoy, balance + 1000.
#
#   key co.  origin balance limit  forecast        actual (gross / level)        over?
#   A   TR02 10     100     150    60 + 90 = 150   200/2 + 200/4 = 150           neither: exact
#   B   TR02 10     100     150    100 + 60 = 160  40/2 + none = 20              forecast
#   C   TR02 10     40      60     5 + 5 = 10      240/2 + 40/4 = 130            actual
#   D   TR02 10     0       -      8 + 2 = 10      6/2 + 8/4 = 5                 nothing open
#   E   TR02 10     -50     -      1 + 3 = 4       no weekly row = 0             nothing open
#   A   TR02 9      50      75     30 + 30 = 60    15/1 + 200/2 = 115            actual
#   F   TR04 10     200     300    400 + 0 = 400   1000/2 + refund only = 500    both
#   G   TR04 10     -10     -      20 + 10 = 30    20/2 + 40/4 = 20              nothing open
#   A, F at origin 11: forecast 2000 and 4000, balance 7777, in no total.
#
# TR02: forecast 394, actual 420, balance 290 (not 290 - 50), 4 with a balance of 6 rows.
# TR04: forecast 430, actual 520, balance 200, 1 with a balance of 2 rows.
# ALL: forecast 824, actual 940, balance 490, 5 with a balance of 8 rows.
# =============================================================================================

from collection_estimation.hurdle.evaluate import (  # noqa: E402
    COHERENCE_COLUMNS, balance_coherence, coherence_row, with_receipts)

BC_HORIZONS = 2
BC_FACTOR = 1.5
BC_LEVELS = {10: 1.0, 11: 2.0, 12: 4.0, 13: 5.0, 14: 8.0}
BC_COMPANY = {"A": "TR02", "B": "TR02", "C": "TR02", "D": "TR02", "E": "TR02", "F": "TR04",
              "G": "TR04", "Z": "TR02"}
# (key, origin): (forecast at h = 1, forecast at h = 2, balance).
BC_ROWS = {("A", 9): (30.0, 30.0, 50.0),
           ("A", 10): (60.0, 90.0, 100.0),
           ("B", 10): (100.0, 60.0, 100.0),
           ("C", 10): (5.0, 5.0, 40.0),
           ("D", 10): (8.0, 2.0, 0.0),
           ("E", 10): (1.0, 3.0, -50.0),
           ("F", 10): (400.0, 0.0, 200.0),
           ("G", 10): (20.0, 10.0, -10.0),
           ("A", 11): (1000.0, 1000.0, 7777.0),
           ("F", 11): (2000.0, 2000.0, 7777.0)}
# One row per company and "ALL": every column of COHERENCE_COLUMNS, worked out by hand above.
BC_EXPECTED = {
    "TR02": dict(rows=6, forecast=394.0, actual=420.0, open_balance=290.0,
                 forecast_share=394 / 290, actual_share=420 / 290, rows_with_balance=4,
                 share_forecast_over=0.25, share_actual_over=0.5,
                 rows_over=1, forecast_over=160.0, actual_over=20.0,
                 rows_nothing_open=2, forecast_nothing_open=14.0, actual_nothing_open=5.0),
    "TR04": dict(rows=2, forecast=430.0, actual=520.0, open_balance=200.0,
                 forecast_share=2.15, actual_share=2.6, rows_with_balance=1,
                 share_forecast_over=1.0, share_actual_over=1.0,
                 rows_over=1, forecast_over=400.0, actual_over=500.0,
                 rows_nothing_open=1, forecast_nothing_open=30.0, actual_nothing_open=20.0),
    "ALL": dict(rows=8, forecast=824.0, actual=940.0, open_balance=490.0,
                forecast_share=824 / 490, actual_share=940 / 490, rows_with_balance=5,
                share_forecast_over=0.4, share_actual_over=0.6,
                rows_over=2, forecast_over=560.0, actual_over=520.0,
                rows_nothing_open=3, forecast_nothing_open=44.0, actual_nothing_open=25.0),
}
BC_AMOUNT_COLUMNS = ["forecast", "actual", "open_balance", "forecast_over", "actual_over",
                     "forecast_nothing_open", "actual_nothing_open"]
BC_COUNT_COLUMNS = ["rows", "rows_with_balance", "rows_over", "rows_nothing_open"]
BC_RATIO_COLUMNS = ["forecast_share", "actual_share", "share_forecast_over",
                    "share_actual_over"]


def bc_customers(rows=None, horizons=BC_HORIZONS):
    """One row per (key, origin) x h. p and m are decoys: p x m is not `expected`."""
    rows = BC_ROWS if rows is None else rows
    records = []
    for (key, origin), values in rows.items():
        for h in range(1, horizons + 1):
            records.append({"key": key, "company_code": BC_COMPANY[key],
                            "customer_id": key.lower(), "origin_week": origin, "h": h,
                            "target_week": origin + h, "p": 0.5, "m": 3.0,
                            "expected": values[h - 1]})
    return pd.DataFrame(records, columns=CUSTOMER_COLUMNS)


def bc_table(rows=None, horizons=BC_HORIZONS):
    """The balances at h = 1; later h hold a decoy. Customer Z and origin 8 are never forecast."""
    rows = BC_ROWS if rows is None else rows
    extra = {("Z", 10): 5000.0, ("A", 8): 3333.0}
    records = []
    for (key, origin), balance in {**{k: v[-1] for k, v in rows.items()}, **extra}.items():
        for h in range(1, horizons + 1):
            records.append({"key": key, "origin_week": origin, "h": h, "extra": -1.0,
                            "open_balance": balance if h == 1 else balance + 1000.0})
    return pd.DataFrame(records)


def bc_weekly():
    """Refunds are never read. Weeks 9, 13, 14 and the unknown customer are decoys."""
    return make_weekly([("A", "TR02", "a", 10, 15.0, 0.0),
                        ("A", "TR02", "a", 11, 200.0, -7.0),
                        ("A", "TR02", "a", 12, 200.0, 0.0),
                        ("B", "TR02", "b", 11, 40.0, 0.0),
                        ("C", "TR02", "c", 11, 240.0, 0.0),
                        ("C", "TR02", "c", 12, 40.0, 0.0),
                        ("D", "TR02", "d", 11, 6.0, 0.0),
                        ("D", "TR02", "d", 12, 8.0, 0.0),
                        ("F", "TR04", "f", 11, 1000.0, 0.0),
                        ("F", "TR04", "f", 12, 0.0, -9.0),
                        ("G", "TR04", "g", 11, 20.0, 0.0),
                        ("G", "TR04", "g", 12, 40.0, 0.0),
                        ("A", "TR02", "a", 13, 99999.0, 0.0),
                        ("A", "TR02", "a", 14, 88888.0, 0.0),
                        ("E", "TR02", "e", 9, 77777.0, 0.0),
                        ("not known", "TR02", None, 11, 66666.0, 0.0)])


def bc_weeks():
    return make_weeks(14, missing={13})


def bc_levels():
    return pd.Series(BC_LEVELS)


def bc_run(**changes):
    arguments = dict(customers=bc_customers(), table=bc_table(), weekly=bc_weekly(),
                     weeks=bc_weeks(), levels=bc_levels(), horizons=BC_HORIZONS,
                     factor=BC_FACTOR)
    arguments.update(changes)
    return balance_coherence(**arguments)


def bc_row(result, company="ALL"):
    selected = result[result["company_code"] == company]
    assert len(selected) == 1, (company, len(selected))
    return selected.iloc[0]


def bc_refused(expected_message, **changes):
    with pytest.raises(ValueError) as error:
        bc_run(**changes)
    assert str(error.value) == expected_message


def bc_subset(*wanted):
    """The worked example with only these (key, origin) rows, for customers and table alike."""
    rows = {pair: BC_ROWS[pair] for pair in wanted}
    return dict(customers=bc_customers(rows), table=bc_table(rows))


def bc_one(balance, expected, gross, factor=BC_FACTOR):
    """One customer, one origin (10), horizons 1, target week 11 at level 2: actual = gross / 2.
    """
    customers = bc_customers({("A", 10): (expected, balance)}, horizons=1)
    table = pd.DataFrame({"key": ["A"], "origin_week": [10], "h": [1],
                          "open_balance": [balance]})
    weekly = make_weekly([("A", "TR02", "a", 11, gross, 0.0)])
    return bc_row(bc_run(customers=customers, table=table, weekly=weekly, horizons=1,
                         factor=factor))


@pytest.fixture(scope="module")
def bc_result():
    return bc_run()


# --- with_receipts ---------------------------------------------------------------------------

def wr_rows(pairs=(("A", 11), ("B", 11), ("A", 12))):
    return pd.DataFrame({"key": [key for key, _ in pairs],
                         "target_week": [week for _, week in pairs],
                         "kept": [float(i) for i in range(len(pairs))]})


def test_with_receipts_adds_the_two_columns_and_keeps_the_rest():
    found = with_receipts(wr_rows(), bc_weekly(), bc_levels())
    assert list(found.columns) == ["key", "target_week", "kept", "gross_receipts", "actual_real"]


def test_with_receipts_exact_numbers():
    found = with_receipts(wr_rows(), bc_weekly(), bc_levels())
    assert found["gross_receipts"].tolist() == [200.0, 40.0, 200.0]
    # Each by the level of its own target week: 200 / 2, 40 / 2, 200 / 4.
    assert found["actual_real"].tolist() == [100.0, 20.0, 50.0]


def test_with_receipts_keeps_the_rows_in_their_order():
    pairs = (("G", 12), ("A", 11), ("C", 12), ("B", 11))
    found = with_receipts(wr_rows(pairs), bc_weekly(), bc_levels())
    assert list(zip(found["key"], found["target_week"])) == list(pairs)
    assert found["kept"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_with_receipts_no_weekly_row_is_a_paid_zero():
    found = with_receipts(wr_rows((("E", 11), ("B", 12))), bc_weekly(), bc_levels())
    assert found["gross_receipts"].tolist() == [0.0, 0.0]
    assert found["actual_real"].tolist() == [0.0, 0.0]


def test_with_receipts_empty_weekly_table_gives_zeros():
    found = with_receipts(wr_rows(), bc_weekly().iloc[0:0], bc_levels())
    assert found["gross_receipts"].tolist() == [0.0, 0.0, 0.0]
    assert found["actual_real"].tolist() == [0.0, 0.0, 0.0]


def test_with_receipts_refunds_are_not_in_it():
    # A pays 200 and refunds 7 in week 11; F only refunds in week 12.
    found = with_receipts(wr_rows((("A", 11), ("F", 12))), bc_weekly(), bc_levels())
    assert found["gross_receipts"].tolist() == [200.0, 0.0]
    assert found["actual_real"].tolist() == [100.0, 0.0]


def test_with_receipts_reads_only_the_target_week_of_the_key():
    # A has other weeks, and other customers have week 12: none of it leaks in.
    found = with_receipts(wr_rows((("D", 12),)), bc_weekly(), bc_levels())
    assert found["gross_receipts"].tolist() == [8.0]


def test_with_receipts_the_same_key_week_on_two_rows_is_served_twice():
    # Two origins reach the same target week: the many of many_to_one.
    found = with_receipts(wr_rows((("A", 12), ("A", 12))), bc_weekly(), bc_levels())
    assert found["gross_receipts"].tolist() == [200.0, 200.0]


def test_with_receipts_does_not_take_the_other_weekly_columns():
    found = with_receipts(wr_rows(), bc_weekly(), bc_levels())
    for column in ("refunds", "company_code", "customer_id", "week", "lines"):
        assert column not in found.columns


def test_with_receipts_does_not_change_its_inputs():
    rows, weekly, levels = wr_rows(), bc_weekly(), bc_levels()
    rows_before, weekly_before, levels_before = rows.copy(), weekly.copy(), levels.copy()
    found = with_receipts(rows, weekly, levels)
    assert found is not rows
    pd.testing.assert_frame_equal(rows, rows_before)
    pd.testing.assert_frame_equal(weekly, weekly_before)
    pd.testing.assert_series_equal(levels, levels_before)


def test_with_receipts_ignores_the_index_of_the_rows():
    rows = wr_rows()
    rows.index = [7, 7, 3]
    found = with_receipts(rows, bc_weekly(), bc_levels())
    assert found["actual_real"].tolist() == [100.0, 20.0, 50.0]


def test_with_receipts_empty_rows_give_an_empty_table_with_the_columns():
    found = with_receipts(wr_rows().iloc[0:0], bc_weekly(), bc_levels())
    assert found.empty
    assert list(found.columns) == ["key", "target_week", "kept", "gross_receipts", "actual_real"]


def test_with_receipts_a_target_week_with_no_level_is_refused():
    levels = bc_levels().drop(11)
    with pytest.raises(ValueError) as error:
        with_receipts(wr_rows(), bc_weekly(), levels)
    assert str(error.value) == "no price level for weeks [11]"


def test_with_receipts_a_target_week_with_no_level_is_refused_even_with_no_cash():
    # The customer paid nothing that week; 0 / unknown is still refused, not a quiet 0.
    with pytest.raises(ValueError, match=r"no price level for weeks \[12\]"):
        with_receipts(wr_rows((("B", 12),)), bc_weekly(), bc_levels().drop(12))


def test_with_receipts_a_repeated_key_week_of_the_weekly_table_is_refused():
    weekly = bc_weekly()
    with pytest.raises(MergeError):
        with_receipts(wr_rows(), pd.concat([weekly, weekly.iloc[[0]]], ignore_index=True),
                      bc_levels())


def test_payer_split_gets_its_receipts_from_with_receipts(monkeypatch):
    # The extraction: payer_split still goes through the helper, and reads what it returns.
    seen = []
    real = evaluate_module.with_receipts

    def spy(rows, weekly, levels):
        seen.append(len(rows))
        return real(rows, weekly, levels)

    monkeypatch.setattr(evaluate_module, "with_receipts", spy)
    split = ps_run()
    assert seen == [9]
    assert ps_row(split, "ALL", "pooled")["rows"] == 9


# --- balance_coherence: the worked example ---------------------------------------------------

def test_coherence_columns_are_pinned():
    assert COHERENCE_COLUMNS == ["company_code", "rows", "forecast", "actual", "open_balance",
                                 "forecast_share", "actual_share", "rows_with_balance",
                                 "share_forecast_over", "share_actual_over", "rows_over",
                                 "forecast_over", "actual_over", "rows_nothing_open",
                                 "forecast_nothing_open", "actual_nothing_open"]


def test_balance_coherence_result_has_exactly_the_columns_and_rows_in_order(bc_result):
    assert list(bc_result.columns) == COHERENCE_COLUMNS
    assert bc_result["company_code"].tolist() == ["TR02", "TR04", "ALL"]
    assert list(bc_result.index) == [0, 1, 2]


@pytest.mark.parametrize("company", list(BC_EXPECTED))
@pytest.mark.parametrize("column", COHERENCE_COLUMNS[1:])
def test_balance_coherence_exact_numbers(bc_result, company, column):
    assert bc_row(bc_result, company)[column] == pytest.approx(BC_EXPECTED[company][column])


def test_the_worked_example_pins_every_column():
    for values in BC_EXPECTED.values():
        assert sorted(values) == sorted(COHERENCE_COLUMNS[1:])


def test_balance_coherence_dtypes(bc_result):
    for column in BC_COUNT_COLUMNS:
        assert bc_result[column].dtype == "int64", column
    for column in BC_AMOUNT_COLUMNS + BC_RATIO_COLUMNS:
        assert bc_result[column].dtype == "float64", column


def test_balance_coherence_companies_are_sorted_whatever_order_they_come_in():
    customers = bc_customers().iloc[::-1]
    assert bc_run(customers=customers)["company_code"].tolist() == ["TR02", "TR04", "ALL"]


def test_balance_coherence_all_comes_last_even_when_a_code_sorts_after_it():
    customers = bc_customers()
    customers["company_code"] = customers["company_code"].map({"TR02": "AAA", "TR04": "ZZZ"})
    assert bc_run(customers=customers)["company_code"].tolist() == ["AAA", "ZZZ", "ALL"]


def test_balance_coherence_one_company_gives_it_and_an_equal_all():
    found = bc_run(**bc_subset(("A", 10), ("B", 10)))
    assert found["company_code"].tolist() == ["TR02", "ALL"]
    for column in COHERENCE_COLUMNS[1:]:
        assert found.iloc[0][column] == found.iloc[1][column], column


# --- ALL is the companies added up -----------------------------------------------------------

@pytest.mark.parametrize("column", BC_AMOUNT_COLUMNS + BC_COUNT_COLUMNS)
def test_balance_coherence_all_is_the_sum_of_the_companies(bc_result, column):
    companies = bc_result[bc_result["company_code"] != "ALL"]
    assert bc_row(bc_result)[column] == pytest.approx(companies[column].sum())


def test_balance_coherence_all_shares_come_from_the_all_totals_not_a_mean(bc_result):
    # The mean of the companies' forecast shares would be (394/290 + 2.15) / 2 = 1.754.
    total = bc_row(bc_result)
    assert total["forecast_share"] == pytest.approx(total["forecast"] / total["open_balance"])
    assert total["forecast_share"] == pytest.approx(824 / 490)
    # The mean of the companies' over-shares would be (0.25 + 1) / 2 = 0.625, not 2 of 5.
    assert total["share_forecast_over"] == pytest.approx(0.4)


def test_balance_coherence_groups_partition_the_rows(bc_result):
    assert (bc_result["rows"]
            == bc_result["rows_with_balance"] + bc_result["rows_nothing_open"]).all()


def bc_by_hand(customers, table, weekly, weeks, levels, horizons, factor, company):
    """The whole report by plain loops, as a second opinion on random tables."""
    with_data = {w for w, gone in zip(weeks["week"], weeks["is_missing"]) if not gone}
    paid = {(k, w): g for k, w, g in zip(weekly["key"], weekly["week"], weekly["gross_receipts"])}
    balances = {(k, o): b for k, o, h, b in zip(table["key"], table["origin_week"], table["h"],
                                                table["open_balance"]) if h == 1}
    good_origins = [origin for origin in set(customers["origin_week"])
                    if all(origin + h in with_data for h in range(1, horizons + 1))]
    found = dict.fromkeys(COHERENCE_COLUMNS[1:], 0.0)
    over_forecast = over_actual = 0
    for (key, origin), rows in customers.groupby(["key", "origin_week"]):
        if origin not in good_origins:
            continue
        if company != "ALL" and rows["company_code"].iloc[0] != company:
            continue
        forecast = rows["expected"].sum()
        actual = sum(paid.get((key, week), 0.0) / levels[week] for week in rows["target_week"])
        balance = balances[(key, origin)]
        found["rows"] += 1
        found["forecast"] += forecast
        found["actual"] += actual
        if balance > 0:
            found["rows_with_balance"] += 1
            found["open_balance"] += balance
            over_actual += actual > factor * balance
            if forecast > factor * balance:
                over_forecast += 1
                found["rows_over"] += 1
                found["forecast_over"] += forecast
                found["actual_over"] += actual
        else:
            found["rows_nothing_open"] += 1
            found["forecast_nothing_open"] += forecast
            found["actual_nothing_open"] += actual
    found["forecast_share"] = found["forecast"] / found["open_balance"]
    found["actual_share"] = found["actual"] / found["open_balance"]
    found["share_forecast_over"] = over_forecast / found["rows_with_balance"]
    found["share_actual_over"] = over_actual / found["rows_with_balance"]
    return found


def bc_random(seed):
    """Three companies, four origins of which one is incomplete, balances of every sign."""
    rng = np.random.default_rng(seed)
    customers, table, weekly = [], [], []
    for number in range(12):
        key, company = f"K{number}", ["TR01", "TR02", "TR03"][number % 3]
        for origin in (5, 6, 7, 8):
            balance = float(rng.choice([-100.0, 0.0, 50.0, 200.0, 1000.0]))
            for h in (1, 2, 3):
                customers.append({"key": key, "company_code": company,
                                  "customer_id": key.lower(), "origin_week": origin, "h": h,
                                  "target_week": origin + h, "p": 0.5, "m": 1.0,
                                  "expected": float(rng.uniform(0, 400))})
                table.append({"key": key, "origin_week": origin, "h": h,
                              "open_balance": balance})
        for week in range(5, 13):
            if rng.random() < 0.6:
                weekly.append((key, company, key.lower(), week, float(rng.uniform(0, 600)),
                               -float(rng.uniform(0, 30))))
    return dict(customers=pd.DataFrame(customers, columns=CUSTOMER_COLUMNS),
                table=pd.DataFrame(table), weekly=make_weekly(weekly),
                weeks=make_weeks(20, missing={11}), levels=make_levels(0, 20), horizons=3,
                factor=float(rng.choice([0.5, 1.0, 1.5, 2.5])))


@pytest.mark.parametrize("seed", range(25))
def test_balance_coherence_equals_plain_loops_on_random_tables(seed):
    arguments = bc_random(seed)
    found = balance_coherence(**arguments)
    assert found["company_code"].tolist() == ["TR01", "TR02", "TR03", "ALL"]
    for company in found["company_code"]:
        by_hand = bc_by_hand(**arguments, company=company)
        got = bc_row(found, company)
        for column in COHERENCE_COLUMNS[1:]:
            assert got[column] == pytest.approx(by_hand[column], abs=1e-9), (company, column)


# --- the limit: strictly above factor x balance ----------------------------------------------

def test_a_forecast_of_exactly_the_limit_is_not_over():
    found = bc_one(100.0, 150.0, 0.0)
    assert found["share_forecast_over"] == 0.0
    assert found["rows_over"] == 0
    assert found["forecast_over"] == 0.0


def test_a_forecast_just_above_the_limit_is_over():
    expected = float(np.nextafter(150.0, np.inf))
    found = bc_one(100.0, expected, 0.0)
    assert found["share_forecast_over"] == 1.0
    assert found["rows_over"] == 1
    assert found["forecast_over"] == expected


def test_a_forecast_just_below_the_limit_is_not_over():
    assert bc_one(100.0, float(np.nextafter(150.0, 0.0)), 0.0)["rows_over"] == 0


def test_actual_cash_of_exactly_the_limit_is_not_over():
    # Gross 300 at level 2 is 150.
    assert bc_one(100.0, 0.0, 300.0)["share_actual_over"] == 0.0


def test_actual_cash_just_above_the_limit_is_over():
    gross = 2.0 * float(np.nextafter(150.0, np.inf))
    assert bc_one(100.0, 0.0, gross)["share_actual_over"] == 1.0


def test_the_worked_example_customer_a_sits_exactly_on_the_limit_in_both():
    # A at origin 10: forecast 150 and actual 150 against the limit of 1.5 x 100.
    found = bc_row(bc_run(**bc_subset(("A", 10))))
    assert (found["forecast"], found["actual"], found["open_balance"]) == (150.0, 150.0, 100.0)
    assert found["share_forecast_over"] == 0.0
    assert found["share_actual_over"] == 0.0


def test_the_factor_is_the_one_given():
    # Factor 1: limits 100, 100, 40, 50, 200.
    #   forecast over: A10 (150), B10 (160), A9 (60), F10 (400); actual there 150, 20, 115, 500.
    #   actual over: A10 (150), C10 (130), A9 (115), F10 (500).
    found = bc_row(bc_run(factor=1.0))
    assert found["rows_over"] == 4
    assert found["forecast_over"] == 770.0
    assert found["actual_over"] == 785.0
    assert found["share_forecast_over"] == pytest.approx(0.8)
    assert found["share_actual_over"] == pytest.approx(0.8)


def test_a_large_factor_gives_real_zeros_not_nan():
    found = bc_row(bc_run(factor=100.0))
    assert found["rows_over"] == 0
    assert (found["forecast_over"], found["actual_over"]) == (0.0, 0.0)
    assert (found["share_forecast_over"], found["share_actual_over"]) == (0.0, 0.0)


def test_the_factor_changes_only_the_over_columns():
    a, b = bc_run(factor=1.0), bc_run(factor=100.0)
    moved = {"share_forecast_over", "share_actual_over", "rows_over", "forecast_over",
             "actual_over"}
    for column in set(COHERENCE_COLUMNS) - moved:
        pd.testing.assert_series_equal(a[column], b[column])


def test_the_over_columns_describe_the_customers_whose_forecast_is_over():
    # C's actual is over its limit and its forecast is not: C is in no over amount, though it
    # is in the actual's share.
    found = bc_row(bc_run(**bc_subset(("C", 10))))
    assert found["share_actual_over"] == 1.0
    assert found["share_forecast_over"] == 0.0
    assert found["rows_over"] == 0
    assert found["actual_over"] == 0.0
    # B's forecast is over and its actual is not: its actual is what actual_over adds.
    found = bc_row(bc_run(**bc_subset(("B", 10))))
    assert found["share_actual_over"] == 0.0
    assert found["rows_over"] == 1
    assert found["actual_over"] == 20.0


# --- negative and zero balances --------------------------------------------------------------

@pytest.mark.parametrize("balance", [0.0, -0.0, -50.0, -1e9])
def test_no_positive_balance_means_nothing_open(balance):
    found = bc_one(balance, 10.0, 8.0)
    assert found["open_balance"] == 0.0
    assert found["rows_with_balance"] == 0
    assert found["rows_nothing_open"] == 1
    assert found["forecast_nothing_open"] == 10.0
    assert found["actual_nothing_open"] == 4.0


def test_the_smallest_positive_balance_is_a_balance():
    found = bc_one(1e-9, 0.0, 0.0)
    assert found["rows_with_balance"] == 1
    assert found["rows_nothing_open"] == 0


@pytest.mark.parametrize("column", ["forecast_share", "actual_share", "share_forecast_over",
                                    "share_actual_over"])
@pytest.mark.parametrize("balance", [0.0, -50.0])
def test_with_no_balance_at_all_the_ratios_are_nan(column, balance):
    assert np.isnan(bc_one(balance, 10.0, 8.0)[column])


def test_with_no_balance_at_all_the_amounts_are_still_reported():
    found = bc_one(-50.0, 10.0, 8.0)
    assert (found["rows"], found["forecast"], found["actual"]) == (1, 10.0, 4.0)
    assert (found["rows_over"], found["forecast_over"], found["actual_over"]) == (0, 0.0, 0.0)


def test_a_credit_does_not_lower_the_balance_total():
    # B 100 and E -50: the total is 100, not 50.
    found = bc_row(bc_run(**bc_subset(("B", 10), ("E", 10))))
    assert found["open_balance"] == 100.0


def test_the_shares_divide_everyones_cash_by_the_balance_total():
    # B (160 and 20, balance 100) and D (10 and 5, nothing open): 170 / 100 and 25 / 100, not
    # 160 / 100 and 20 / 100.
    found = bc_row(bc_run(**bc_subset(("B", 10), ("D", 10))))
    assert found["forecast_share"] == pytest.approx(1.7)
    assert found["actual_share"] == pytest.approx(0.25)


def test_the_over_shares_count_customers_with_a_balance_only():
    # B over and D nothing open: 1 of 1, not 1 of 2. D's huge forecast has no limit to pass.
    rows = {("B", 10): BC_ROWS[("B", 10)], ("D", 10): (1e9, 1e9, 0.0)}
    found = bc_row(bc_run(customers=bc_customers(rows), table=bc_table(rows)))
    assert found["share_forecast_over"] == 1.0
    assert found["rows_over"] == 1
    assert found["forecast_over"] == 160.0


def test_one_company_with_nothing_open_does_not_touch_the_others_ratios():
    found = bc_run(**bc_subset(("A", 10), ("G", 10)))
    assert np.isnan(bc_row(found, "TR04")["forecast_share"])
    assert bc_row(found, "TR02")["forecast_share"] == 1.5
    # ALL has a balance: A's 100. Forecast 150 + 30.
    assert bc_row(found, "ALL")["forecast_share"] == 1.8


# --- where the balance comes from ------------------------------------------------------------

def test_the_balance_is_read_at_h_1_not_at_a_later_h(bc_result):
    # bc_table's later h hold balance + 1000; a sum over h would be far above 490.
    assert bc_row(bc_result)["open_balance"] == 490.0


def test_the_balance_is_read_at_h_1_whatever_the_order_of_the_table():
    pd.testing.assert_frame_equal(bc_run(table=bc_table().iloc[::-1]), bc_run())


def test_each_origin_has_its_own_balance_of_the_same_customer():
    # A at origin 9 owes 50, at origin 10 owes 100: both count, each once.
    found = bc_row(bc_run(**bc_subset(("A", 9), ("A", 10))))
    assert found["open_balance"] == 150.0
    assert found["rows"] == 2
    # A's limits are 75 and 150; its forecasts 60 and 150; neither is over.
    assert found["rows_over"] == 0


def test_a_customer_is_one_row_per_origin_not_one_per_h(bc_result):
    # 8 (customer, origin) pairs against 16 rows of customers.
    assert bc_row(bc_result)["rows"] == 8


def test_a_balance_in_the_table_with_no_forecast_adds_nothing(bc_result):
    # Z (5000, origin 10) and A at origin 8 (3333) are in the table and were never forecast.
    assert bc_row(bc_result)["open_balance"] == 490.0
    assert bc_row(bc_result, "TR02")["rows"] == 6


def test_the_balance_is_not_deflated_again():
    # Target week level 2: the balance stays 100.
    assert bc_one(100.0, 0.0, 0.0)["open_balance"] == 100.0


def test_a_balance_from_a_float_h_column_is_still_read():
    table = bc_table().astype({"h": "float64"})
    pd.testing.assert_frame_equal(bc_run(table=table), bc_run())


# --- the actual cash -------------------------------------------------------------------------

def test_the_forecast_is_all_h_of_the_origin_summed():
    # B at origin 10: 100 + 60.
    assert bc_row(bc_run(**bc_subset(("B", 10))))["forecast"] == 160.0


def test_the_forecast_is_the_expected_column_not_p_times_m(bc_result):
    # p x m is 1.5 on every row, 24.0 over the 16 rows.
    assert bc_row(bc_result)["forecast"] == 824.0


def test_the_forecast_is_not_deflated():
    assert bc_one(100.0, 80.0, 0.0)["forecast"] == 80.0


def test_the_actual_is_deflated_by_the_target_weeks_level_not_the_origins():
    # C at origin 10: 240 / level of 11 (2) + 40 / level of 12 (4); by the origin's level (1)
    # it would be 280.
    assert bc_row(bc_run(**bc_subset(("C", 10))))["actual"] == 130.0


def test_a_customer_with_no_weekly_row_paid_zero_and_is_still_counted():
    found = bc_row(bc_run(**bc_subset(("E", 10))))
    assert found["rows"] == 1
    assert found["actual"] == 0.0
    assert found["forecast"] == 4.0


def test_a_customer_with_only_one_of_its_weeks_in_weekly_has_that_week():
    # B has a row in week 11 and none in week 12.
    assert bc_row(bc_run(**bc_subset(("B", 10))))["actual"] == 20.0


def test_an_empty_weekly_table_gives_zero_actuals_not_nan():
    found = bc_run(weekly=bc_weekly().iloc[0:0])
    assert (found["actual"] == 0.0).all()
    assert (found["actual_share"] == 0.0).all()
    assert (found["share_actual_over"] == 0.0).all()
    assert (found["forecast"] == bc_run()["forecast"]).all()


def test_refunds_are_not_in_the_actual():
    weekly = bc_weekly()
    weekly["refunds"] = -123456.0
    pd.testing.assert_frame_equal(bc_run(weekly=weekly), bc_run())


def test_cash_of_customers_nobody_forecast_is_not_in_the_actual(bc_result):
    # "not known" paid 66666 in week 11, with data.
    assert bc_row(bc_result)["actual"] == 940.0


def test_the_same_key_week_serves_two_origins(bc_result):
    # A's week 11 (gross 200) is in origin 9 and in origin 10, each deflated to 100.
    found = bc_row(bc_run(**bc_subset(("A", 9), ("A", 10))))
    assert found["actual"] == 115.0 + 150.0


# --- only the complete origins ---------------------------------------------------------------

def test_an_origin_with_a_missing_target_week_is_dropped_whole():
    # A and F at origin 11 are out: their h = 1 (week 12, with data) as well.
    found = bc_run()
    without = {pair: values for pair, values in BC_ROWS.items() if pair[1] != 11}
    pd.testing.assert_frame_equal(found, bc_run(customers=bc_customers(without)))


def test_dropping_an_incomplete_origin_does_not_touch_the_others(bc_result):
    # Origin 10 shares week 12 with the dropped origin 11, and is whole.
    assert bc_row(bc_result)["forecast"] == 824.0
    assert bc_row(bc_result)["actual"] == 940.0
    assert bc_row(bc_result)["open_balance"] == 490.0


def test_a_missing_week_at_h_1_drops_the_origin_too():
    # Week 11 missing: origins 9 (targets 10, 11) and 10 (targets 11, 12) go; origin 11 (12, 13)
    # is whole once week 13 has data.
    found = bc_run(weeks=make_weeks(14, missing={11}))
    total = bc_row(found)
    assert total["rows"] == 2
    assert total["forecast"] == 6000.0
    assert total["open_balance"] == 2 * 7777.0


def test_an_origin_whose_target_week_is_past_the_calendar_is_dropped():
    # The calendar ends at week 12, so origin 11 cannot be whole, week 13 being unknown.
    pd.testing.assert_frame_equal(bc_run(weeks=make_weeks(12)), bc_run())


def test_a_week_with_data_decides_not_the_cash_in_it():
    # Week 13 has data now: A's forecast at origin 11 is complete, with its gross of 99999.
    total = bc_row(bc_run(weeks=make_weeks(14)))
    assert total["rows"] == 10
    assert total["forecast"] == 824.0 + 6000.0
    assert total["open_balance"] == 490.0 + 2 * 7777.0
    # A: 200 / 4 + 99999 / 5; F: 0 + 0.
    assert total["actual"] == pytest.approx(940.0 + 50.0 + 99999.0 / 5.0)


def test_balance_coherence_cash_in_a_missing_week_is_never_counted(bc_result):
    # A paid 99999 in the missing week 13, which only the dropped origin reaches.
    assert bc_row(bc_result)["actual"] == 940.0


def test_a_dropped_origin_needs_no_price_level():
    # Week 13 has no data and no level; only the dropped origin 11 reaches it.
    pd.testing.assert_frame_equal(bc_run(levels=bc_levels().drop(13)), bc_run())


def test_a_dropped_origin_needs_no_balance():
    # Origin 11 has no row in the table: no refusal, since it is not used.
    table = bc_table()
    pd.testing.assert_frame_equal(bc_run(table=table[table["origin_week"] != 11]), bc_run())


def test_an_origin_with_rows_for_only_some_h_is_dropped():
    # Origin 10 has no row at h = 2 at all: it cannot cover the same weeks as the actual.
    customers = bc_customers()
    customers = customers[~((customers["origin_week"] == 10) & (customers["h"] == 2))]
    total = bc_row(bc_run(customers=customers))
    assert total["rows"] == 1
    assert total["forecast"] == 60.0


def test_horizons_above_the_rows_h_leave_no_complete_origin():
    bc_refused("no origin has data in all its target weeks", horizons=3)


def test_horizons_one_needs_only_the_first_week():
    # Origin 12 reaches 13 (missing) and goes; origins 9, 10, 11 reach weeks with data.
    rows = {("A", 9): (30.0, 50.0), ("A", 12): (500.0, 700.0), ("B", 11): (70.0, 30.0)}
    found = bc_row(bc_run(customers=bc_customers(rows, horizons=1),
                          table=bc_table(rows, horizons=1), horizons=1))
    assert found["rows"] == 2
    assert found["forecast"] == 100.0
    assert found["open_balance"] == 80.0
    # A's 15 / 1 in week 10; B has no row in week 12.
    assert found["actual"] == 15.0


def test_a_week_is_incomplete_if_any_day_is_missing_as_the_calendar_says():
    # is_missing decides, not days_with_file: flip it by hand on an otherwise full week.
    weeks = make_weeks(14)
    weeks.loc[weeks["week"] == 13, "is_missing"] = True
    pd.testing.assert_frame_equal(bc_run(weeks=weeks), bc_run())


# --- the order and index of the input do not matter ------------------------------------------

def test_the_order_of_the_customers_does_not_matter(bc_result):
    shuffled = bc_customers().sample(frac=1.0, random_state=3)
    pd.testing.assert_frame_equal(bc_run(customers=shuffled), bc_result)


def test_the_index_of_the_customers_does_not_matter(bc_result):
    # rolling_forecasts keeps the table's index; here also with repeated labels.
    customers = bc_customers()
    customers.index = [5000 + i % 3 for i in range(len(customers))]
    pd.testing.assert_frame_equal(bc_run(customers=customers), bc_result)


def test_the_index_of_the_table_does_not_matter(bc_result):
    table = bc_table()
    table.index = [7] * len(table)
    pd.testing.assert_frame_equal(bc_run(table=table), bc_result)


def test_the_order_of_the_weekly_rows_does_not_matter(bc_result):
    weekly = bc_weekly().sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(bc_run(weekly=weekly), bc_result)


def test_the_extra_columns_of_the_table_are_not_taken(bc_result):
    assert "extra" not in bc_result.columns


def test_same_inputs_give_the_same_report(bc_result):
    pd.testing.assert_frame_equal(bc_run(), bc_result)


def test_the_inputs_are_not_changed():
    customers, table, weekly, weeks, levels = (bc_customers(), bc_table(), bc_weekly(),
                                               bc_weeks(), bc_levels())
    before = customers.copy(), table.copy(), weekly.copy(), weeks.copy(), levels.copy()
    balance_coherence(customers, table, weekly, weeks, levels, BC_HORIZONS, BC_FACTOR)
    pd.testing.assert_frame_equal(customers, before[0])
    pd.testing.assert_frame_equal(table, before[1])
    pd.testing.assert_frame_equal(weekly, before[2])
    pd.testing.assert_frame_equal(weeks, before[3])
    pd.testing.assert_series_equal(levels, before[4])


def test_the_report_raises_no_warning(bc_result):
    # Nothing-open companies included: a NaN by ratio_or_nan, not by a division.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bc_run()
        bc_one(-50.0, 10.0, 8.0)


# --- the refusals ----------------------------------------------------------------------------

@pytest.mark.parametrize("horizons", [0, -1, -5])
def test_balance_coherence_refuses_horizons_below_one(horizons):
    bc_refused(f"horizons is {horizons}; horizons start at 1", horizons=horizons)


@pytest.mark.parametrize("factor", [0, 0.0, -0.0, -1.5, -1e-12])
def test_balance_coherence_refuses_a_factor_not_above_zero(factor):
    bc_refused(f"factor is {factor}; it must be above 0", factor=factor)


def test_balance_coherence_takes_a_very_small_factor():
    assert bc_row(bc_run(factor=1e-12))["rows_over"] == 5


def test_balance_coherence_refuses_an_h_above_horizons():
    bc_refused("customer forecasts for h [2]; horizons is 1", horizons=1)


def test_balance_coherence_refuses_an_h_below_one_and_sorts_the_list():
    customers = bc_customers()
    customers.loc[0, "h"] = 7
    customers.loc[1, "h"] = 0
    bc_refused("customer forecasts for h [0, 7]; horizons is 2", customers=customers)


def test_balance_coherence_refuses_an_h_outside_even_at_a_dropped_origin():
    customers = bc_customers()
    customers.loc[customers["origin_week"] == 11, "h"] = 3
    bc_refused("customer forecasts for h [3]; horizons is 2", customers=customers)


def test_balance_coherence_refuses_a_repeated_key_origin_h():
    customers = bc_customers()
    repeated = pd.concat([customers, customers.iloc[[2]]], ignore_index=True)
    bc_refused("1 customer forecasts are there twice "
               "(first: {'key': 'A', 'origin_week': 10, 'h': 1})", customers=repeated)


def test_balance_coherence_counts_every_repeat():
    customers = bc_customers()
    repeated = pd.concat([customers, customers.iloc[[2, 2, 5]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"^3 customer forecasts are there twice"):
        bc_run(customers=repeated)


def test_balance_coherence_refuses_a_repeat_even_at_a_dropped_origin():
    customers = bc_customers()
    at_11 = customers[customers["origin_week"] == 11].iloc[[0]]
    with pytest.raises(ValueError, match=r"^1 customer forecasts are there twice"):
        bc_run(customers=pd.concat([customers, at_11], ignore_index=True))


def test_balance_coherence_the_same_key_at_another_origin_or_h_is_no_repeat(bc_result):
    # A is at origins 9, 10, 11 and at h 1, 2: no refusal.
    assert list(bc_result.columns) == COHERENCE_COLUMNS


def test_balance_coherence_refuses_when_no_origin_is_complete():
    bc_refused("no origin has data in all its target weeks",
               weeks=make_weeks(14, missing={10, 11, 12}))


def test_balance_coherence_refuses_a_calendar_that_ends_too_soon():
    bc_refused("no origin has data in all its target weeks", weeks=make_weeks(5))


def test_balance_coherence_refuses_an_empty_customer_table():
    bc_refused("no origin has data in all its target weeks", customers=bc_customers().iloc[0:0])


def test_balance_coherence_refuses_a_forecast_with_no_balance():
    table = bc_table()
    gone = (table["key"] == "A") & (table["origin_week"] == 9)
    bc_refused("1 customer forecasts have no balance in the table "
               "(first: {'key': 'A', 'origin_week': 9})", table=table[~gone])


def test_balance_coherence_counts_every_forecast_with_no_balance():
    # Per customer and origin, not per h: B and C at origin 10.
    table = bc_table()
    bc_refused("2 customer forecasts have no balance in the table "
               "(first: {'key': 'B', 'origin_week': 10})",
               table=table[~table["key"].isin(["B", "C"])])


def test_a_balance_of_another_origin_does_not_serve():
    # A has balances at origins 8 and 10, and none at 9.
    table = bc_table()
    table = table[~((table["key"] == "A") & (table["origin_week"] == 9))]
    with pytest.raises(ValueError, match="have no balance"):
        bc_run(table=table)


def test_a_balance_only_at_a_later_h_does_not_serve():
    # h = 1 is what the report reads; the decoy at h = 2 is not a fallback.
    table = bc_table()
    gone = (table["key"] == "B") & (table["origin_week"] == 10) & (table["h"] == 1)
    bc_refused("1 customer forecasts have no balance in the table "
               "(first: {'key': 'B', 'origin_week': 10})", table=table[~gone])


def test_a_nan_balance_is_no_balance_not_nothing_open():
    table = bc_table()
    table.loc[(table["key"] == "C") & (table["origin_week"] == 10) & (table["h"] == 1),
              "open_balance"] = np.nan
    bc_refused("1 customer forecasts have no balance in the table "
               "(first: {'key': 'C', 'origin_week': 10})", table=table)


def test_balance_coherence_a_used_week_with_no_level_is_refused():
    bc_refused("no price level for weeks [11]", levels=bc_levels().drop(11))


def test_the_refusals_come_in_order():
    customers = bc_customers()
    twice = pd.concat([customers, customers.iloc[[2]]], ignore_index=True)
    outside = twice.copy()
    outside.loc[0, "h"] = 9
    bc_refused("horizons is 0; horizons start at 1", customers=outside, horizons=0, factor=0)
    bc_refused("factor is 0; it must be above 0", customers=outside, factor=0)
    bc_refused("customer forecasts for h [9]; horizons is 2", customers=outside)
    # h before the repeat; the repeat before the complete origins.
    bc_refused("1 customer forecasts are there twice "
               "(first: {'key': 'A', 'origin_week': 10, 'h': 1})", customers=twice,
               weeks=make_weeks(5))
    # The complete origins before the balances.
    bc_refused("no origin has data in all its target weeks", weeks=make_weeks(5),
               table=bc_table().iloc[0:0])


# --- repeated rows that would count twice: pandas' merge checks refuse them ------------------

def test_a_repeated_balance_row_fails_rather_than_counts_twice():
    table = bc_table()
    at_h_1 = table[(table["key"] == "B") & (table["origin_week"] == 10) & (table["h"] == 1)]
    with pytest.raises(MergeError):
        bc_run(table=pd.concat([table, at_h_1], ignore_index=True))


def test_two_different_balances_for_one_customer_and_origin_fail():
    table = bc_table()
    at_h_1 = table[(table["key"] == "B") & (table["origin_week"] == 10)
                   & (table["h"] == 1)].assign(open_balance=1.0)
    with pytest.raises(MergeError):
        bc_run(table=pd.concat([table, at_h_1], ignore_index=True))


def test_a_repeated_balance_of_a_customer_nobody_forecast_fails_too():
    # validate looks at the whole right-hand table, not only at the keys that match: Z has no
    # forecast and still stops the report.
    table = bc_table()
    at_h_1 = table[(table["key"] == "Z") & (table["h"] == 1)]
    with pytest.raises(MergeError):
        bc_run(table=pd.concat([table, at_h_1], ignore_index=True))


def test_a_repeated_row_at_a_later_h_is_not_read_at_all(bc_result):
    table = bc_table()
    later = table[(table["key"] == "B") & (table["origin_week"] == 10) & (table["h"] == 2)]
    pd.testing.assert_frame_equal(bc_run(table=pd.concat([table, later], ignore_index=True)),
                                  bc_result)


def test_one_customer_under_two_companies_at_one_origin_fails():
    # Summed per (key, company, origin): two rows for one balance, not a split of it.
    customers = bc_customers()
    customers.loc[(customers["key"] == "B") & (customers["h"] == 2), "company_code"] = "TR04"
    with pytest.raises(MergeError):
        bc_run(customers=customers)


def test_a_repeated_key_week_of_the_weekly_table_fails():
    weekly = bc_weekly()
    with pytest.raises(MergeError):
        bc_run(weekly=pd.concat([weekly, weekly.iloc[[0]]], ignore_index=True))


# --- coherence_row called directly -----------------------------------------------------------

def cr_rows(records):
    """records: (expected, actual_real, open_balance)."""
    return pd.DataFrame(records, columns=["expected", "actual_real", "open_balance"])


def test_coherence_row_keys_are_the_columns_in_order():
    found = coherence_row("TR02", cr_rows([(1.0, 2.0, 3.0)]), 1.5)
    assert list(found) == COHERENCE_COLUMNS
    assert found["company_code"] == "TR02"


def test_coherence_row_by_hand():
    found = coherence_row("X", cr_rows([(16.0, 1.0, 10.0), (1.0, 16.0, 10.0), (2.0, 3.0, 0.0),
                                        (4.0, 5.0, -7.0)]), 1.5)
    assert found == {"company_code": "X", "rows": 4, "forecast": 23.0, "actual": 25.0,
                     "open_balance": 20.0, "forecast_share": 1.15, "actual_share": 1.25,
                     "rows_with_balance": 2, "share_forecast_over": 0.5,
                     "share_actual_over": 0.5, "rows_over": 1, "forecast_over": 16.0,
                     "actual_over": 1.0, "rows_nothing_open": 2, "forecast_nothing_open": 6.0,
                     "actual_nothing_open": 8.0}


def test_coherence_row_values_are_plain_python_numbers():
    found = coherence_row("X", cr_rows([(16.0, 1.0, 10.0), (2.0, 3.0, 0.0)]), 1.5)
    for column, value in found.items():
        if column == "company_code":
            continue
        assert type(value) in (int, float), (column, type(value))


def test_coherence_row_on_no_rows_gives_zeros_and_nan():
    found = coherence_row("X", cr_rows([]).astype("float64"), 1.5)
    assert (found["rows"], found["forecast"], found["open_balance"]) == (0, 0.0, 0.0)
    assert np.isnan(found["forecast_share"])
    assert np.isnan(found["share_forecast_over"])


# --- with the real rolling forecasts: the table's shape is what the report expects ---------------

def test_balance_coherence_takes_the_real_rolling_forecasts(r_table, r_result):
    # 1. rolling_forecasts' customers against its own training table (8 customers, h 1..5, 3
    #    origins, index shuffled); no weekly cash, so every actual is a real 0.
    found = balance_coherence(r_result.customers, r_table, make_weekly([]), make_weeks(60),
                              make_levels(0, 60), 5, 1.5)
    assert found["company_code"].tolist()[-1] == "ALL"
    total = bc_row(found)
    assert total["rows"] == 8 * len(R_ORIGINS)
    # 2. The forecast is the customers' expected, and the balance the table's at h = 1, clipped.
    assert total["forecast"] == pytest.approx(r_result.customers["expected"].sum())
    at_1 = r_table[(r_table["h"] == 1) & r_table["origin_week"].isin(R_ORIGINS)]
    assert total["open_balance"] == pytest.approx(at_1["open_balance"].clip(lower=0).sum())
    assert total["actual"] == 0.0
