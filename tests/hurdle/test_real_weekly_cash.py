"""Tests for `evaluate.real_weekly_cash` (U19.1): each company's deflated weekly cash and the
consolidated "ALL" series, the code moved out of `deflated_reference`.

Every table is small and hand-made. The level of week w is 1 + w / 10, so a wrong week's level
shows in the numbers. At the end, `deflated_reference` is checked against a copy of the code
as it was before the move (`series_before_the_move`), so the move changed nothing.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.benchmark import deflated_forecasts
from collection_estimation.hurdle.evaluate import REFERENCES, deflated_reference, real_weekly_cash
from collection_estimation.hurdle.forecast import CONSOLIDATED
from collection_estimation.hurdle.prices import deflate, projected_levels


# --- builders --------------------------------------------------------------------------------

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
    return pd.Series([1.0 + w / 10 for w in weeks], index=weeks)


def small_example():
    """Two companies, weeks 0-3, week 2 missing (its 999 must never be read).

    week 0: 1000 has 100 - 10 = 90, 2000 has 20
    week 1: 1000 has two lines, 50 and 30 = 80; 2000 has no line (a real 0)
    week 2: missing
    week 3: 1000 has 40 - 5 = 35, 2000 has 60 - 60 = 0
    """
    weekly = make_weekly([
        ("1000", 0, 100.0, -10.0),
        ("2000", 0, 20.0, 0.0),
        ("1000", 1, 50.0, 0.0),
        ("1000", 1, 30.0, 0.0),
        ("1000", 2, 999.0, 0.0),
        ("2000", 2, 999.0, 0.0),
        ("1000", 3, 40.0, -5.0),
        ("2000", 3, 60.0, -60.0),
    ])
    return weekly, make_weeks(3, missing={2}), make_levels()


# --- values ----------------------------------------------------------------------------------

def test_keys_are_the_companies_sorted_then_all():
    weekly, weeks, levels = small_example()
    result = real_weekly_cash(weekly, weeks, levels, ["2000", "1000"])
    assert list(result) == ["1000", "2000", CONSOLIDATED]


def test_company_values_by_hand_deflated_by_their_own_weeks_level():
    weekly, weeks, levels = small_example()
    result = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    assert list(result["1000"].index) == [0, 1, 3]
    assert result["1000"].tolist() == pytest.approx([90.0 / 1.0, 80.0 / 1.1, 35.0 / 1.3])


def test_company_without_a_line_in_a_week_with_data_gets_zero():
    weekly, weeks, levels = small_example()
    result = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    assert result["2000"].loc[1] == 0.0
    assert result["2000"].tolist() == pytest.approx([20.0, 0.0, 0.0])


def test_missing_week_is_not_in_any_series():
    weekly, weeks, levels = small_example()
    result = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    for series in result.values():
        assert 2 not in series.index
        assert list(series.index) == [0, 1, 3]


def test_all_is_the_sum_of_the_companies_each_week():
    weekly, weeks, levels = small_example()
    result = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    assert result[CONSOLIDATED].tolist() == pytest.approx(
        [(90.0 + 20.0) / 1.0, 80.0 / 1.1, 35.0 / 1.3])
    pd.testing.assert_series_equal(result[CONSOLIDATED], result["1000"] + result["2000"],
                                   check_names=False)


def test_week_with_data_and_no_line_at_all_is_zero_for_everyone():
    weekly, _, levels = small_example()
    weeks = make_weeks(4, missing={2})          # week 4 has data but no line at all
    result = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    for series in result.values():
        assert series.loc[4] == 0.0


def test_a_line_in_a_week_outside_the_weeks_table_is_not_read():
    weekly, weeks, levels = small_example()
    weekly = pd.concat([weekly, make_weekly([("1000", 9, 5e6, 0.0)])], ignore_index=True)
    result = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    assert 9 not in result["1000"].index
    assert result["1000"].tolist() == pytest.approx([90.0, 80.0 / 1.1, 35.0 / 1.3])


def test_company_with_no_cash_at_all_gets_zeros():
    weekly, weeks, levels = small_example()
    result = real_weekly_cash(weekly, weeks, levels, ["1000", "2000", "3000"])
    assert list(result) == ["1000", "2000", "3000", CONSOLIDATED]
    assert result["3000"].tolist() == [0.0, 0.0, 0.0]


def test_empty_weekly_gives_zero_series():
    _, weeks, levels = small_example()
    empty = make_weekly([])
    result = real_weekly_cash(empty, weeks, levels, ["1000"])
    assert list(result) == ["1000", CONSOLIDATED]
    assert list(result["1000"].index) == [0, 1, 3]
    assert result["1000"].tolist() == [0.0, 0.0, 0.0]
    assert result[CONSOLIDATED].tolist() == [0.0, 0.0, 0.0]


def test_duplicate_companies_are_counted_once():
    weekly, weeks, levels = small_example()
    once = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    twice = real_weekly_cash(weekly, weeks, levels, ["2000", "1000", "2000"])
    assert list(twice) == list(once)
    for company in once:
        pd.testing.assert_series_equal(twice[company], once[company])


def test_integer_company_codes_sort_numerically():
    weekly = make_weekly([(10, 0, 1.0, 0.0), (9, 0, 2.0, 0.0)])
    result = real_weekly_cash(weekly, make_weeks(0), make_levels(), [10, 9])
    assert list(result) == [9, 10, CONSOLIDATED]


def test_series_are_float_and_indexed_by_week():
    weekly, weeks, levels = small_example()
    for series in real_weekly_cash(weekly, weeks, levels, ["1000", "2000"]).values():
        assert series.dtype == np.float64
        assert list(series.index) == [0, 1, 3]


def test_missing_week_without_a_price_level_does_not_raise():
    weekly, weeks, _ = small_example()
    levels = make_levels().drop(2)               # week 2 is missing anyway
    result = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    assert list(result["1000"].index) == [0, 1, 3]


def test_inputs_are_not_modified():
    weekly, weeks, levels = small_example()
    before = (weekly.copy(), weeks.copy(), levels.copy())
    real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    pd.testing.assert_frame_equal(weekly, before[0])
    pd.testing.assert_frame_equal(weeks, before[1])
    pd.testing.assert_series_equal(levels, before[2])


def test_order_of_the_weekly_rows_does_not_matter():
    weekly, weeks, levels = small_example()
    shuffled = weekly.iloc[::-1].reset_index(drop=True)
    first = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    second = real_weekly_cash(shuffled, weeks, levels, ["1000", "2000"])
    for company in first:
        pd.testing.assert_series_equal(first[company], second[company])


# --- errors ----------------------------------------------------------------------------------

def test_empty_company_list_raises():
    weekly, weeks, levels = small_example()
    with pytest.raises(ValueError, match="the company list is empty"):
        real_weekly_cash(weekly, weeks, levels, [])


def test_all_in_the_company_list_raises():
    weekly, weeks, levels = small_example()
    with pytest.raises(ValueError, match="consolidated rows"):
        real_weekly_cash(weekly, weeks, levels, ["1000", "2000", CONSOLIDATED])


def test_cash_of_a_company_outside_the_list_raises_and_names_it():
    weekly, weeks, levels = small_example()
    with pytest.raises(ValueError, match=r"not in the list: \['2000'\]"):
        real_weekly_cash(weekly, weeks, levels, ["1000"])


def test_company_outside_the_list_raises_even_with_cash_only_in_a_missing_week():
    weekly, weeks, levels = small_example()
    weekly = pd.concat([weekly, make_weekly([("5000", 2, 1.0, 0.0)])], ignore_index=True)
    with pytest.raises(ValueError, match=r"\['5000'\]"):
        real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])


def test_week_with_data_and_no_price_level_raises():
    weekly, weeks, _ = small_example()
    levels = make_levels().drop(3)
    with pytest.raises(ValueError, match="no price level"):
        real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])


# --- the move changed nothing ----------------------------------------------------------------

def series_before_the_move(weekly, weeks, levels, companies):
    """The lines of `deflated_reference` before U19.1, copied unchanged (without the checks)."""
    with_data = sorted(set(weeks.loc[~weeks["is_missing"], "week"]))
    net = weekly.assign(actual=weekly["gross_receipts"] + weekly["refunds"])
    cash = (net[net["week"].isin(with_data)]
            .pivot_table(index="week", columns="company_code", values="actual", aggfunc="sum")
            .reindex(index=with_data, columns=sorted(set(companies)), fill_value=0.0)
            .fillna(0.0))
    cash[CONSOLIDATED] = cash.sum(axis=1)
    week_numbers = pd.Series(cash.index, index=cash.index)
    return {company: deflate(cash[company], week_numbers, levels) for company in cash.columns}


def longer_example():
    """Two companies, weeks 0-30, week 6 missing (huge amount), 2000 has no line in week 10."""
    rows = []
    for w in range(0, 31):
        rows.append(("1000", w, 1e9 if w == 6 else 1000.0 + 20.0 * w + 100.0 * (w % 4), -5.0))
        if w != 10:
            rows.append(("2000", w, 1e9 if w == 6 else 400.0 + 50.0 * np.cos(w), 0.0))
    return make_weekly(rows), make_weeks(30, missing={6}), make_levels()


def simple_calendar(last=60):
    weeks = list(range(0, last + 1))
    return pd.DataFrame({"week": weeks, "cycle": [float(w % 4) for w in weeks],
                         "wave": [float(np.cos(w)) for w in weeks]})


def test_real_weekly_cash_is_the_code_before_the_move():
    weekly, weeks, levels = longer_example()
    now = real_weekly_cash(weekly, weeks, levels, ["1000", "2000"])
    before = series_before_the_move(weekly, weeks, levels, ["1000", "2000"])
    assert list(now) == list(before)
    for company in before:
        pd.testing.assert_series_equal(now[company], before[company])


def test_deflated_reference_gives_what_the_code_before_the_move_gave():
    weekly, weeks, levels = longer_example()
    calendar, week_zero = simple_calendar(), date(2025, 6, 2)
    origins, horizons = [15, 20], 2
    found = deflated_reference(weekly, weeks, levels, calendar, ["1000", "2000"], origins,
                               horizons, week_zero)
    before = series_before_the_move(weekly, weeks, levels, ["1000", "2000"])
    expected = []
    for company, series in before.items():
        for origin in origins:
            projected = projected_levels(week_zero, origin, [origin + h for h in
                                                             range(1, horizons + 1)])
            for h in range(1, horizons + 1):
                for with_level, with_trend in REFERENCES.values():
                    made = deflated_forecasts(series, calendar, h, origin, [origin],
                                              with_level, with_trend).iloc[0]
                    expected.append((company, origin, h, float(made["forecast"]),
                                     float(made["forecast"]) * float(projected.loc[origin + h]),
                                     bool(made["fallback"])))
    assert list(zip(found["company_code"], found["origin_week"], found["h"],
                    found["real_forecast"], found["forecast"], found["fallback"])) == expected


def test_deflated_reference_still_checks_origins_before_the_companies():
    weekly, weeks, levels = longer_example()
    with pytest.raises(ValueError, match="no origins"):
        deflated_reference(weekly, weeks, levels, simple_calendar(), [], [], 2,
                           date(2025, 6, 2))


def test_deflated_reference_still_checks_horizons_before_the_companies():
    weekly, weeks, levels = longer_example()
    with pytest.raises(ValueError, match="horizons start at 1"):
        deflated_reference(weekly, weeks, levels, simple_calendar(), [CONSOLIDATED], [15], 0,
                           date(2025, 6, 2))


def test_deflated_reference_raises_the_company_errors_of_real_weekly_cash():
    weekly, weeks, levels = longer_example()
    with pytest.raises(ValueError, match="the company list is empty"):
        deflated_reference(weekly, weeks, levels, simple_calendar(), [], [15], 2,
                           date(2025, 6, 2))
    with pytest.raises(ValueError, match=r"not in the list: \['2000'\]"):
        deflated_reference(weekly, weeks, levels, simple_calendar(), ["1000"], [15], 2,
                           date(2025, 6, 2))
