"""Tests for `hurdle/evaluate.py`: `actual_totals` (U14.1), the cash that really came per company
and consolidated ("ALL") in the target week of every origin and h, nominal and deflated.

Every table is hand-made; nothing is read from data/. The properties that matter most are
tested directly: actual = gross receipts + refunds of the TARGET week, deflated by the TARGET
week's level; a missing week or a week past the calendar has no row (no data is not 0), while
a company with no line in a week with data gets a real 0.0; "ALL" is the sum of the companies.
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
