"""Tests for `hurdle/calibrate.py` (U15): the reliability tables, the corrections of p and m, the
corrected forecast and the choice among the corrections.

Every table is hand-made; nothing is read from data/. Two small worlds carry most of the tests:

`worked()`, for `corrected_forecast`: two companies, two origins, two weeks ahead, with the same
target week priced differently at the two origins.

`validation_case()`, for `choose_calibration`: origins 10-17, horizons 2, so the target weeks are
11-19, the split is week 15 and the judged origins are 15, 16, 17. Four customers:

    key  company  p    m    pays in target week w
    K1   TR02     0.8  100  unless w is divisible by 4 (not 12, not 16)
    K2   TR02     0.5  200  even weeks
    K3   TR04     0.5  400  odd weeks
    K4   TR04     0.1  800  weeks 13 and 17 only

Whoever pays, pays 2 x m in the weeks up to the split and 5 x m after it (in TRY of week 0), so a
factor learned on the early rows is exactly 2, and anything else shows that later rows got in.
The price level of week w is 1 + w / 10, at every origin. Top-ups per (origin, h): TR02 +5,
TR04 +1, 6 in all.

The m of the rows that paid, target weeks 11-15 (rows aiming at each week: 1, 2, 2, 2, 2):
    K1 100 x (1 + 2 + 2 + 2) = 700    K2 200 x (2 + 2) = 800
    K3 400 x (1 + 2 + 2) = 2000       K4 800 x 2 = 1600                 together 5100
and target weeks 16-19 (rows: 2, 2, 2, 1):
    K1 100 x (2 + 2 + 1) = 500        K2 200 x (2 + 2) = 800
    K3 400 x (2 + 1) = 1200           K4 800 x 2 = 1600                 together 4100
So the frozen `scale`, learned on every row, is (2 x 5100 + 5 x 4100) / 9200 = 30700 / 9200.

The consolidated actual cash (nominal) in the judged weeks, m of the payers x 5 x level:
    week 16: 200 x 5 x 2.6 = 2600                 week 17: 1300 x 5 x 2.7 = 17550
    week 18: 300 x 5 x 2.8 = 4200                 week 19: 500 x 5 x 2.9 = 7250
"""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import calibrate
from collection_estimation.hurdle.calibrate import (
    ALL_H,
    BY_SIZE,
    CANDIDATE_COLUMNS,
    M_KINDS,
    NONE,
    ONE_MAP,
    P_KINDS,
    PER_COMPANY,
    PER_H,
    SCALE,
    CalibrationChoice,
    MCorrection,
    PCorrection,
    actual_over_m,
    apply_m_correction,
    apply_p_correction,
    calibration_reliability,
    calibration_rows,
    calibration_split,
    candidate_forecasts,
    candidate_name,
    candidate_scores,
    choose_calibration,
    corrected_forecast,
    fit_m_correction,
    fit_p_correction,
    log_of_m,
    numbers_learned,
    simplest_good_enough,
)
from collection_estimation.hurdle.evaluate import (
    BOOTSTRAP_COLUMNS,
    HORIZON_TOTAL,
    POOLED,
    STACK_COLUMNS,
    actual_totals,
)
from collection_estimation.hurdle.forecast import (
    CONSOLIDATED,
    CUSTOMER_COLUMNS,
    TOTAL_COLUMNS,
    Forecast,
    forecast_totals,
)


# --- builders --------------------------------------------------------------------------------

def make_customers(rows):
    """rows: (key, company_code, origin_week, h, p, m)."""
    frame = pd.DataFrame(rows, columns=["key", "company_code", "origin_week", "h", "p", "m"])
    frame["customer_id"] = frame["key"]
    frame["target_week"] = frame["origin_week"] + frame["h"]
    frame["p"] = frame["p"].astype("float64")
    frame["m"] = frame["m"].astype("float64")
    frame["expected"] = frame["p"] * frame["m"]
    return frame[CUSTOMER_COLUMNS]


def make_weekly(rows):
    """rows: (key, company_code, week, gross_receipts, refunds)."""
    frame = pd.DataFrame(rows, columns=["key", "company_code", "week", "gross_receipts",
                                        "refunds"])
    frame["customer_id"] = frame["key"]
    frame["week"] = frame["week"].astype("int64")
    frame["gross_receipts"] = frame["gross_receipts"].astype("float64")
    frame["refunds"] = frame["refunds"].astype("float64")
    frame["unapplied_lines"] = 0
    frame["lines"] = 1
    return frame


def make_weeks(last_week=30, missing=()):
    weeks = list(range(0, last_week + 1))
    return pd.DataFrame({"week": weeks, "is_missing": [w in missing for w in weeks]})


def level(week):
    return 1.0 + week / 10


def make_levels(last_week=30):
    weeks = list(range(0, last_week + 1))
    return pd.Series([level(w) for w in weeks], index=weeks)


def make_rows(rows):
    """Rows as `calibration_rows` returns them: (company_code, h, p, m, actual_real)."""
    frame = pd.DataFrame(rows, columns=["company_code", "h", "p", "m", "actual_real"])
    frame["paid"] = frame["actual_real"] > 0
    return frame


def platt_rows(h=1, low=(20, 15), high=(20, 18)):
    """`low`[0] rows at p = 0.5 of which `low`[1] paid, `high`[0] rows at p = 0.8 of which
    `high`[1] paid. Two values of logit(p), so the fit goes exactly through both shares:
    b = logit(share at 0.5), as logit(0.5) = 0; a = (logit(share at 0.8) - b) / logit(0.8)."""
    rows = []
    for p, (count, paid) in ((0.5, low), (0.8, high)):
        rows += [("TR02", h, p, 10.0, 10.0 if number < paid else 0.0) for number in range(count)]
    return make_rows(rows)


def logit(p):
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def make_rolled(customers, topup_of, level_of, companies, horizons):
    """A `Forecast` as `rolling_forecasts` stacks it, from `forecast_totals` per origin.
    topup_of(company, origin, h) -> (new_customer_cash, unidentified_cash, refunds, fallback);
    level_of(origin, target_week) -> the price level projected at that origin."""
    totals = []
    for origin in sorted(set(customers["origin_week"])):
        topups = pd.DataFrame(
            [(company, origin, h, origin + h, *topup_of(company, origin, h))
             for company in companies for h in range(1, horizons + 1)],
            columns=["company_code", "origin_week", "h", "target_week", "new_customer_cash",
                     "unidentified_cash", "refunds", "fallback"])
        levels = pd.Series({origin + h: level_of(origin, origin + h)
                            for h in range(1, horizons + 1)})
        totals.append(forecast_totals(customers[customers["origin_week"] == origin], topups,
                                      levels, companies, horizons))
    return Forecast(totals=pd.concat(totals, ignore_index=True), customers=customers)


def row_of(totals, company, origin, h):
    found = totals[(totals["company_code"] == company) & (totals["origin_week"] == origin)
                   & (totals["h"] == h)]
    assert len(found) == 1
    return found.iloc[0]


# The world of `corrected_forecast`. Target week 12 is priced 1.2 at origin 10 and 1.5 at origin
# 11. TR02's top-ups are +10, +1, -2 (net 9); TR04's +20, +2, -4 (net 18); TR04 fell back at
# origin 11, h 1.
WORKED_LEVELS = {(10, 11): 1.1, (10, 12): 1.2, (11, 12): 1.5, (11, 13): 1.6}


def worked_topups(company, origin, h):
    if company == "TR02":
        return 10.0, 1.0, -2.0, False
    return 20.0, 2.0, -4.0, (origin, h) == (11, 1)


def worked(index=None):
    customers = make_customers([
        ("K1", "TR02", 10, 1, 0.5, 100.0), ("K2", "TR04", 10, 1, 0.2, 50.0),
        ("K1", "TR02", 10, 2, 0.4, 200.0), ("K2", "TR04", 10, 2, 0.1, 300.0),
        ("K1", "TR02", 11, 1, 0.6, 10.0), ("K2", "TR04", 11, 1, 0.3, 20.0),
        ("K1", "TR02", 11, 2, 0.5, 40.0), ("K2", "TR04", 11, 2, 0.25, 80.0)])
    if index is not None:
        customers.index = index
    return make_rolled(customers, worked_topups, lambda o, t: WORKED_LEVELS[(o, t)],
                       ["TR02", "TR04"], 2)


# The world of `choose_calibration`; see the top of this file.
CUSTOMERS = {"K1": ("TR02", 0.8, 100.0), "K2": ("TR02", 0.5, 200.0),
             "K3": ("TR04", 0.5, 400.0), "K4": ("TR04", 0.1, 800.0)}
COMPANIES = ["TR02", "TR04"]
SPLIT = 15
JUDGED = [15, 16, 17]
ORIGINS = list(range(10, 18))
HORIZONS = 2


def pays(key, week):
    return {"K1": week % 4 != 0, "K2": week % 2 == 0, "K3": week % 2 == 1,
            "K4": week in (13, 17)}[key]


def case_topups(company, origin, h):
    return (5.0, 1.0, -1.0, False) if company == "TR02" else (3.0, 0.0, -2.0, False)


def case_customers():
    return make_customers([(key, company, origin, h, p, m)
                           for origin in ORIGINS for h in range(1, HORIZONS + 1)
                           for key, (company, p, m) in CUSTOMERS.items()])


def case_weekly():
    rows = []
    for week in range(11, 20):
        factor = 2.0 if week <= SPLIT else 5.0
        for key, (company, _, m) in CUSTOMERS.items():
            if pays(key, week):
                rows.append((key, company, week, factor * m * level(week), 0.0))
    return make_weekly(rows)


def validation_case(missing=()):
    """The world: rolled, weekly, weeks, levels, actual."""
    customers = case_customers()
    rolled = make_rolled(customers, case_topups, lambda _, target: level(target), COMPANIES,
                         HORIZONS)
    weekly = case_weekly()
    weeks = make_weeks(missing=missing)
    levels = make_levels()
    actual = actual_totals(weekly, weeks, levels, COMPANIES, ORIGINS, HORIZONS)
    return SimpleNamespace(rolled=rolled, weekly=weekly, weeks=weeks, levels=levels,
                           actual=actual)


def make_settings(min_gain=0.005, shrink_rows=200, groups=2, horizons=HORIZONS, samples=50,
                  seed=7):
    return SimpleNamespace(
        receivables=SimpleNamespace(horizons=horizons),
        calibration=SimpleNamespace(company_shrink_rows=shrink_rows, min_gain=min_gain),
        diagnostic=SimpleNamespace(calibration_groups=groups),
        evaluation=SimpleNamespace(bootstrap_samples=samples, bootstrap_seed=seed))


def choose(case, **settings):
    return choose_calibration(case.rolled, case.weekly, case.weeks, case.levels, case.actual,
                              make_settings(**settings))


def force_winner(monkeypatch, name):
    """Make the choice fall on the candidate `name`, to see what is frozen for it."""
    def chosen(candidates, min_gain):
        return candidates[candidates["candidate"] == name].iloc[0]
    monkeypatch.setattr(calibrate, "simplest_good_enough", chosen)


def candidate_row(choice, p_kind, m_kind):
    found = choice.candidates[(choice.candidates["p"] == p_kind)
                              & (choice.candidates["m"] == m_kind)]
    assert len(found) == 1
    return found.iloc[0]


# The two by-hand WAPEs of the judged origins (see the top of the file). With p and m as they
# are, the customers' expected cash is 0.8 x 100 + 0.5 x 200 + 0.5 x 400 + 0.1 x 800 = 460 in
# every (origin, h); with `scale` learned on the early rows, 2 x 460 = 920. Plus 6 of top-ups.
# The judged (origin, h) aim at weeks 16, 17, 17, 18, 18, 19.
ACTUAL_OF_WEEK = {16: 2600.0, 17: 17550.0, 18: 4200.0, 19: 7250.0}
JUDGED_WEEKS = [16, 17, 17, 18, 18, 19]


def wape_by_hand(real_total, weeks=tuple(JUDGED_WEEKS)):
    error = sum(abs(real_total * level(week) - ACTUAL_OF_WEEK[week]) for week in weeks)
    return error / sum(ACTUAL_OF_WEEK[week] for week in weeks)


# --- calibration_rows ------------------------------------------------------------------------

def rows_case():
    customers = make_customers([("K1", "TR02", 10, 1, 0.5, 100.0),    # week 11
                                ("K2", "TR02", 10, 1, 0.5, 100.0),    # week 11, no weekly row
                                ("K3", "TR04", 10, 1, 0.5, 100.0),    # week 11, only a refund
                                ("K1", "TR02", 10, 2, 0.5, 100.0),    # week 12, missing
                                ("K1", "TR02", 11, 2, 0.5, 100.0)])   # week 13, refund and cash
    weekly = make_weekly([("K1", "TR02", 11, 210.0, 0.0),
                          ("K3", "TR04", 11, 0.0, -50.0),
                          ("K1", "TR02", 12, 999.0, 0.0),
                          ("K1", "TR02", 13, 230.0, -40.0),
                          ("K2", "TR02", 14, 777.0, 0.0)])            # no row aims at week 14
    return customers, weekly, make_weeks(missing=(12,)), make_levels()


def test_calibration_rows_keeps_only_the_weeks_with_data():
    found = calibration_rows(*rows_case())
    assert sorted(found["target_week"]) == [11, 11, 11, 13]


def test_calibration_rows_a_customer_with_no_weekly_row_paid_a_real_zero():
    found = calibration_rows(*rows_case())
    row = found[found["key"] == "K2"].iloc[0]
    assert row["gross_receipts"] == 0.0
    assert row["actual_real"] == 0.0
    assert not row["paid"]


def test_calibration_rows_a_refund_is_not_a_receipt():
    found = calibration_rows(*rows_case())
    # A week with only a refund did not pay; a week with cash keeps it gross.
    only_refund = found[found["key"] == "K3"].iloc[0]
    assert only_refund["gross_receipts"] == 0.0
    assert not only_refund["paid"]
    both = found[found["target_week"] == 13].iloc[0]
    assert both["gross_receipts"] == 230.0
    assert both["paid"]


def test_calibration_rows_deflates_by_the_target_weeks_level():
    found = calibration_rows(*rows_case())
    paid_11 = found[(found["key"] == "K1") & (found["target_week"] == 11)].iloc[0]
    assert paid_11["actual_real"] == pytest.approx(210.0 / 2.1)
    paid_13 = found[found["target_week"] == 13].iloc[0]
    assert paid_13["actual_real"] == pytest.approx(230.0 / 2.3)


def test_calibration_rows_two_origins_aiming_at_a_week_both_get_its_cash():
    customers = make_customers([("K1", "TR02", 10, 2, 0.5, 1.0), ("K1", "TR02", 11, 1, 0.5, 2.0)])
    weekly = make_weekly([("K1", "TR02", 12, 60.0, 0.0)])
    found = calibration_rows(customers, weekly, make_weeks(), make_levels())
    assert list(found["gross_receipts"]) == [60.0, 60.0]


def test_calibration_rows_keeps_the_forecast_columns_and_adds_three():
    customers, weekly, weeks, levels = rows_case()
    found = calibration_rows(customers, weekly, weeks, levels)
    assert list(found.columns) == CUSTOMER_COLUMNS + ["gross_receipts", "actual_real", "paid"]
    assert found["paid"].dtype == bool


def test_calibration_rows_does_not_change_its_inputs():
    customers, weekly, weeks, levels = rows_case()
    before = customers.copy(), weekly.copy(), weeks.copy(), levels.copy()
    calibration_rows(customers, weekly, weeks, levels)
    pd.testing.assert_frame_equal(customers, before[0])
    pd.testing.assert_frame_equal(weekly, before[1])
    pd.testing.assert_frame_equal(weeks, before[2])
    pd.testing.assert_series_equal(levels, before[3])


def test_calibration_rows_with_no_week_with_data_raises():
    customers, weekly, _, levels = rows_case()
    weeks = make_weeks(missing=(11, 12, 13))
    with pytest.raises(ValueError, match="no customer forecasts in a week with data"):
        calibration_rows(customers, weekly, weeks, levels)


def test_calibration_rows_a_target_week_past_the_calendar_is_not_a_week_with_data():
    customers, weekly, _, levels = rows_case()
    found = calibration_rows(customers, weekly, make_weeks(last_week=11), levels)
    assert set(found["target_week"]) == {11}


def test_calibration_rows_a_week_with_no_price_level_raises():
    customers, weekly, weeks, _ = rows_case()
    with pytest.raises(ValueError, match="no price level"):
        calibration_rows(customers, weekly, weeks, make_levels(last_week=12))


# --- calibration_reliability -----------------------------------------------------------------

def reliability_case():
    """Week 11 (h 1): p 0.2 did not pay, 0.3 paid, 0.6 did not, 0.7 paid.
    Week 12 (h 2): p 0.1 did not pay, 0.4 did not, 0.5 paid, 0.9 paid. The cash is nominal, level
    2.1 in week 11 and 2.2 in week 12, so the real amounts are 100 and 300, 10 and 30."""
    customers = make_customers([("K1", "TR02", 10, 1, 0.2, 1.0), ("K2", "TR02", 10, 1, 0.3, 90.0),
                                ("K3", "TR02", 10, 1, 0.6, 2.0), ("K4", "TR02", 10, 1, 0.7, 150.0),
                                ("K1", "TR02", 10, 2, 0.1, 3.0), ("K2", "TR02", 10, 2, 0.4, 4.0),
                                ("K3", "TR02", 10, 2, 0.5, 5.0), ("K4", "TR02", 10, 2, 0.9, 20.0)])
    weekly = make_weekly([("K2", "TR02", 11, 100.0 * 2.1, 0.0), ("K4", "TR02", 11, 300.0 * 2.1, 0.0),
                          ("K3", "TR02", 12, 10.0 * 2.2, 0.0), ("K4", "TR02", 12, 30.0 * 2.2, 0.0)])
    return customers, weekly, make_weeks(), make_levels()


def test_reliability_p_has_each_h_then_pooled():
    found = calibration_reliability(*reliability_case(), groups=2)
    assert list(found.p["horizon"]) == ["1", "1", "2", "2", POOLED, POOLED]
    assert list(found.p.columns[:2]) == ["horizon", "group"]


def test_reliability_p_by_hand():
    found = calibration_reliability(*reliability_case(), groups=2)
    at_h1 = found.p[found.p["horizon"] == "1"]
    assert list(at_h1["mean_p"]) == pytest.approx([0.25, 0.65])
    assert list(at_h1["share_paid"]) == pytest.approx([0.5, 0.5])
    at_h2 = found.p[found.p["horizon"] == "2"]
    assert list(at_h2["mean_p"]) == pytest.approx([0.25, 0.7])
    assert list(at_h2["share_paid"]) == pytest.approx([0.0, 1.0])
    # Pooled, p sorted: 0.1 0.2 0.3 0.4 | 0.5 0.6 0.7 0.9.
    pooled = found.p[found.p["horizon"] == POOLED]
    assert list(pooled["mean_p"]) == pytest.approx([0.25, 2.7 / 4])
    assert list(pooled["share_paid"]) == pytest.approx([0.25, 0.75])
    assert list(pooled["rows"]) == [4, 4]


def test_reliability_m_only_on_the_rows_that_paid_by_hand():
    found = calibration_reliability(*reliability_case(), groups=2)
    # Paid rows by m: 5 (real 10), 20 (real 30) | 90 (real 100), 150 (real 300).
    assert list(found.m["rows"]) == [2, 2]
    assert list(found.m["mean_m"]) == pytest.approx([12.5, 120.0])
    assert list(found.m["mean_actual"]) == pytest.approx([20.0, 200.0])
    assert list(found.m["ratio"]) == pytest.approx([20.0 / 12.5, 200.0 / 120.0])


def test_reliability_leaves_out_a_missing_week():
    customers, weekly, _, levels = reliability_case()
    found = calibration_reliability(customers, weekly, make_weeks(missing=(12,)), levels, groups=2)
    assert list(found.p["horizon"]) == ["1", "1", POOLED, POOLED]
    assert found.m["rows"].sum() == 2


def test_reliability_too_many_groups_raises():
    with pytest.raises(ValueError, match="cannot be cut"):
        calibration_reliability(*reliability_case(), groups=5)


def test_reliability_an_h_where_everybody_paid_raises():
    customers = make_customers([("K1", "TR02", 10, 1, 0.2, 1.0), ("K2", "TR02", 10, 1, 0.3, 1.0)])
    weekly = make_weekly([("K1", "TR02", 11, 1.0, 0.0), ("K2", "TR02", 11, 1.0, 0.0)])
    with pytest.raises(ValueError, match="both outcomes"):
        calibration_reliability(customers, weekly, make_weeks(), make_levels(), groups=2)


# --- fit_p_correction and apply_p_correction -------------------------------------------------

def test_fit_p_none_learns_nothing():
    correction = fit_p_correction(platt_rows(), NONE)
    assert correction == PCorrection(NONE, {})


def test_apply_p_none_returns_equal_values_in_a_new_series():
    rows = pd.DataFrame({"p": [0.1, 0.7], "h": [1, 2]}, index=[5, 9])
    found = apply_p_correction(rows, PCorrection(NONE))
    pd.testing.assert_series_equal(found, rows["p"])
    found.iloc[0] = 0.9
    assert rows["p"].iloc[0] == 0.1


def test_fit_p_one_map_goes_through_the_two_shares_by_hand():
    # Share 15 / 20 = 0.75 at p = 0.5, 18 / 20 = 0.9 at p = 0.8.
    correction = fit_p_correction(platt_rows(), ONE_MAP)
    assert correction.kind == ONE_MAP
    assert list(correction.maps) == [ALL_H]
    a, b = correction.maps[ALL_H]
    assert b == pytest.approx(logit(0.75), abs=5e-3)
    assert a == pytest.approx((logit(0.9) - logit(0.75)) / logit(0.8), abs=5e-3)


def test_apply_p_one_map_by_hand():
    rows = pd.DataFrame({"p": [0.5, 0.2], "h": [1, 2]}, index=[5, 9])
    # Slope 2, intercept log 3: 0.5 has logit 0, so odds 3; 0.2 has odds 1/4, squared 1/16, x 3.
    found = apply_p_correction(rows, PCorrection(ONE_MAP, {ALL_H: (2.0, math.log(3.0))}))
    assert list(found.index) == [5, 9]
    assert list(found) == pytest.approx([0.75, (3 / 16) / (1 + 3 / 16)])


def test_one_map_is_one_map_for_all_h():
    rows = pd.concat([platt_rows(h=1, low=(20, 15), high=(20, 18)),
                      platt_rows(h=2, low=(20, 5), high=(20, 12))], ignore_index=True)
    correction = fit_p_correction(rows, ONE_MAP)
    # Pooled shares: 20 / 40 at p = 0.5 and 30 / 40 at p = 0.8.
    a, b = correction.maps[ALL_H]
    assert b == pytest.approx(logit(0.5), abs=5e-3)
    assert a == pytest.approx(logit(0.75) / logit(0.8), abs=5e-3)


def test_fit_p_per_h_learns_each_h_on_its_own_rows_by_hand():
    rows = pd.concat([platt_rows(h=1, low=(20, 15), high=(20, 18)),
                      platt_rows(h=2, low=(20, 5), high=(20, 12))], ignore_index=True)
    correction = fit_p_correction(rows, PER_H)
    assert correction.kind == PER_H
    assert list(correction.maps) == [1, 2]
    assert all(type(h) is int for h in correction.maps)
    a1, b1 = correction.maps[1]
    assert b1 == pytest.approx(logit(0.75), abs=5e-3)
    assert a1 == pytest.approx((logit(0.9) - logit(0.75)) / logit(0.8), abs=5e-3)
    a2, b2 = correction.maps[2]
    assert b2 == pytest.approx(logit(0.25), abs=5e-3)
    assert a2 == pytest.approx((logit(0.6) - logit(0.25)) / logit(0.8), abs=5e-3)


def test_apply_p_per_h_uses_each_rows_own_map_and_keeps_the_index():
    rows = pd.DataFrame({"p": [0.5, 0.5, 0.2], "h": [2, 1, 2]}, index=[30, 10, 20])
    # h 1: odds x 1. h 2: odds x 3.
    found = apply_p_correction(rows, PCorrection(PER_H, {1: (1.0, 0.0), 2: (1.0, math.log(3.0))}))
    assert list(found.index) == [30, 10, 20]
    assert list(found) == pytest.approx([0.75, 0.5, (3 / 4) / (1 + 3 / 4)])


def test_apply_p_per_h_may_have_maps_for_an_h_with_no_rows():
    rows = pd.DataFrame({"p": [0.5], "h": [1]})
    found = apply_p_correction(rows, PCorrection(PER_H, {1: (1.0, 0.0), 2: (1.0, 5.0)}))
    assert list(found) == pytest.approx([0.5])


def test_apply_p_per_h_raises_for_an_unseen_h():
    rows = pd.DataFrame({"p": [0.5, 0.5, 0.5], "h": [1, 3, 4]})
    with pytest.raises(ValueError, match=r"learned nothing for h \[3, 4\]"):
        apply_p_correction(rows, PCorrection(PER_H, {1: (1.0, 0.0)}))


def test_apply_p_does_not_change_the_rows():
    rows = pd.DataFrame({"p": [0.5, 0.2], "h": [1, 2]})
    before = rows.copy()
    apply_p_correction(rows, PCorrection(PER_H, {1: (2.0, 1.0), 2: (2.0, 1.0)}))
    apply_p_correction(rows, PCorrection(ONE_MAP, {ALL_H: (2.0, 1.0)}))
    pd.testing.assert_frame_equal(rows, before)


@pytest.mark.parametrize("kind", ["scale", "per_h", "", None])
def test_fit_p_another_kind_raises(kind):
    with pytest.raises(ValueError, match="the correction of p is"):
        fit_p_correction(platt_rows(), kind)


@pytest.mark.parametrize("kind", [ONE_MAP, PER_H])
def test_fit_p_with_one_outcome_raises(kind):
    rows = platt_rows(low=(20, 20), high=(20, 20))
    with pytest.raises(ValueError, match="both outcomes"):
        fit_p_correction(rows, kind)


def test_fit_p_per_h_raises_if_any_h_has_one_outcome():
    rows = pd.concat([platt_rows(h=1), platt_rows(h=2, low=(20, 0), high=(20, 0))],
                     ignore_index=True)
    with pytest.raises(ValueError, match="both outcomes"):
        fit_p_correction(rows, PER_H)
    # The rows together have both, so one map is fine.
    assert fit_p_correction(rows, ONE_MAP).kind == ONE_MAP


# --- actual_over_m and log_of_m --------------------------------------------------------------

def test_actual_over_m_is_a_ratio_of_sums_not_a_mean_of_ratios():
    paid = pd.DataFrame({"actual_real": [10.0, 300.0], "m": [100.0, 100.0]})
    assert actual_over_m(paid) == pytest.approx(1.55)
    assert type(actual_over_m(paid)) is float


def test_log_of_m_by_hand():
    found = log_of_m(pd.Series([math.e, 1.0, 100.0], index=[7, 8, 9]))
    assert list(found.index) == [7, 8, 9]
    assert list(found) == pytest.approx([1.0, 0.0, math.log(100.0)])


@pytest.mark.parametrize("m", [0.0, 1e-9, 0.5, 0.999])
def test_log_of_m_below_1_counts_as_1(m):
    found = log_of_m(pd.Series([m]))
    assert found.iloc[0] == 0.0


# --- fit_m_correction and apply_m_correction -------------------------------------------------

def m_rows():
    """TR02: two paid rows, m together 100, actual 300 (own factor 3). TR04: four paid rows, m
    together 100, actual 100 (own factor 1). Together m 200, actual 400 (common factor 2). The
    unpaid rows would pull everything down if they counted."""
    return make_rows([("TR02", 1, 0.5, 40.0, 200.0), ("TR02", 2, 0.5, 60.0, 100.0),
                      ("TR04", 1, 0.5, 10.0, 25.0), ("TR04", 1, 0.5, 20.0, 25.0),
                      ("TR04", 2, 0.5, 30.0, 25.0), ("TR04", 2, 0.5, 40.0, 25.0),
                      ("TR02", 1, 0.5, 5000.0, 0.0), ("TR04", 2, 0.5, 7000.0, 0.0)])


def test_fit_m_none():
    assert fit_m_correction(m_rows(), NONE, 200, 2) == MCorrection(NONE)


def test_fit_m_none_needs_no_paid_rows():
    rows = m_rows()
    assert fit_m_correction(rows[~rows["paid"]], NONE, 200, 2) == MCorrection(NONE)


def test_apply_m_none_leaves_m_as_it_is():
    rows = pd.DataFrame({"m": [1.5, 0.0, 3e7], "company_code": ["A", "B", "C"]}, index=[3, 1, 2])
    found = apply_m_correction(rows, MCorrection(NONE))
    pd.testing.assert_series_equal(found, rows["m"])


def test_fit_m_scale_by_hand_ignores_the_rows_that_did_not_pay():
    correction = fit_m_correction(m_rows(), SCALE, 200, 2)
    assert correction.kind == SCALE
    assert correction.common == pytest.approx(2.0)
    assert correction.per_company == {} and correction.factors == ()


def test_scale_makes_the_sums_match_on_the_paid_rows():
    rows = m_rows()
    paid = rows[rows["paid"]]
    corrected = apply_m_correction(paid, fit_m_correction(rows, SCALE, 200, 2))
    assert corrected.sum() == pytest.approx(paid["actual_real"].sum())
    assert list(corrected) == pytest.approx(list(paid["m"] * 2.0))


def test_fit_m_per_company_shrinkage_by_hand():
    correction = fit_m_correction(m_rows(), PER_COMPANY, 200, 2)
    assert correction.kind == PER_COMPANY
    assert correction.common == pytest.approx(2.0)
    assert sorted(correction.per_company) == ["TR02", "TR04"]
    # (n x own + shrink_rows x common) / (n + shrink_rows); n counts the paid rows only.
    assert correction.per_company["TR02"] == pytest.approx((2 * 3.0 + 200 * 2.0) / 202)
    assert correction.per_company["TR04"] == pytest.approx((4 * 1.0 + 200 * 2.0) / 204)


def test_fit_m_per_company_a_small_shrink_stays_near_the_own_factor():
    correction = fit_m_correction(m_rows(), PER_COMPANY, 1, 2)
    assert correction.per_company["TR02"] == pytest.approx((2 * 3.0 + 2.0) / 3)
    assert correction.per_company["TR04"] == pytest.approx((4 * 1.0 + 2.0) / 5)


def test_apply_m_per_company_uses_each_companys_factor_and_the_common_for_an_unknown_one():
    rows = pd.DataFrame({"m": [10.0, 10.0, 10.0], "company_code": ["TR04", "TR99", "TR02"]},
                        index=[8, 6, 7])
    correction = MCorrection(PER_COMPANY, common=2.0, per_company={"TR02": 3.0, "TR04": 0.5})
    found = apply_m_correction(rows, correction)
    assert list(found.index) == [8, 6, 7]
    assert list(found) == pytest.approx([5.0, 20.0, 30.0])


def by_size_rows():
    """Groups by m: (10, 100), m together 110, actual 220, factor 2, median log m = the mean of
    log 10 and log 100; (1000, 10000), m together 11000, actual 5500, factor 0.5."""
    return make_rows([("TR02", 1, 0.5, 1000.0, 500.0), ("TR02", 1, 0.5, 10.0, 20.0),
                      ("TR02", 1, 0.5, 10000.0, 5000.0), ("TR02", 1, 0.5, 100.0, 200.0),
                      ("TR02", 1, 0.5, 1e6, 0.0)])


def test_fit_m_by_size_by_hand():
    correction = fit_m_correction(by_size_rows(), BY_SIZE, 200, 2)
    assert correction.kind == BY_SIZE
    assert correction.log_m == pytest.approx((1.5 * math.log(10), 3.5 * math.log(10)))
    assert correction.factors == pytest.approx((2.0, 0.5))
    assert correction.common == pytest.approx(5720.0 / 11110.0)
    assert type(correction.log_m) is tuple and type(correction.factors) is tuple


def test_apply_m_by_size_interpolates_in_log_m():
    correction = MCorrection(BY_SIZE, log_m=(math.log(10), math.log(1000)), factors=(2.0, 1.0))
    rows = pd.DataFrame({"m": [10.0, 100.0, 1000.0, 10 ** 2.5]}, index=[4, 3, 2, 1])
    found = apply_m_correction(rows, correction)
    assert list(found.index) == [4, 3, 2, 1]
    # m = 100 lies half way in log m, not half way in m: the factor is 1.5, not about 1.9.
    # 10 ** 2.5 lies three quarters of the way: 1.25.
    assert list(found) == pytest.approx([20.0, 150.0, 1000.0, 1.25 * 10 ** 2.5])


def test_apply_m_by_size_is_flat_beyond_the_ends():
    correction = MCorrection(BY_SIZE, log_m=(math.log(10), math.log(1000)), factors=(2.0, 1.0))
    rows = pd.DataFrame({"m": [2.0, 9.0, 1001.0, 1e7]})
    found = apply_m_correction(rows, correction)
    assert list(found) == pytest.approx([4.0, 18.0, 1001.0, 1e7])


def test_apply_m_by_size_with_m_below_1():
    # Below 1, m counts as 1 for finding the factor, which stays flat; the amount stays small.
    correction = MCorrection(BY_SIZE, log_m=(math.log(10), math.log(1000)), factors=(2.0, 1.0))
    rows = pd.DataFrame({"m": [0.5, 0.0]})
    found = apply_m_correction(rows, correction)
    assert list(found) == pytest.approx([1.0, 0.0])
    assert np.isfinite(found).all()


def test_fit_m_by_size_does_not_learn_from_m_below_1():
    # Two paid rows with m below 1 and a huge actual: if they counted, the first factor would be
    # 2000 / 0.7. They are left out of the groups (not of the common factor); the rest is
    # `by_size_rows`: factors 2 and 0.5.
    rows = pd.concat([by_size_rows(),
                      make_rows([("TR02", 1, 0.5, 0.2, 1000.0), ("TR02", 1, 0.5, 0.5, 1000.0)])],
                     ignore_index=True)
    correction = fit_m_correction(rows, BY_SIZE, 200, 2)
    assert correction.log_m == pytest.approx((1.5 * math.log(10), 3.5 * math.log(10)))
    assert correction.factors == pytest.approx((2.0, 0.5))
    assert correction.common == pytest.approx(7720.0 / 11110.7)


def test_by_size_m_below_1_is_corrected_with_the_first_factor():
    rows = pd.concat([by_size_rows(),
                      make_rows([("TR02", 1, 0.5, 0.2, 1000.0), ("TR02", 1, 0.5, 0.5, 1000.0)])],
                     ignore_index=True)
    correction = fit_m_correction(rows, BY_SIZE, 200, 2)
    found = apply_m_correction(pd.DataFrame({"m": [0.0, 0.2, 0.5, 0.999]}), correction)
    assert list(found) == pytest.approx([0.0, 0.4, 1.0, 1.998])


def test_fit_m_by_size_with_only_m_below_1_raises():
    rows = make_rows([("TR02", 1, 0.5, 0.2, 1.0), ("TR02", 1, 0.5, 0.5, 1.0),
                      ("TR02", 1, 0.5, 0.9, 1.0)])
    with pytest.raises(ValueError, match="0 rows cannot be cut into 2 groups"):
        fit_m_correction(rows, BY_SIZE, 200, 2)


def test_fit_m_by_size_counts_only_m_of_1_or_more_against_the_groups():
    # Four paid rows, but only two with m of at least 1: three groups cannot be cut.
    rows = make_rows([("TR02", 1, 0.5, 0.2, 1.0), ("TR02", 1, 0.5, 0.5, 1.0),
                      ("TR02", 1, 0.5, 10.0, 1.0), ("TR02", 1, 0.5, 20.0, 1.0)])
    with pytest.raises(ValueError, match="2 rows cannot be cut into 3 groups"):
        fit_m_correction(rows, BY_SIZE, 200, 3)


def test_fit_m_by_size_merges_groups_with_the_same_median():
    # The repro of the tied medians: groups (100, 100) and (100, 100) share the median log 100.
    # Merged, one factor from all four rows: 800 / 400 = 2, so the corrected m add up to 800.
    rows = make_rows([("TR02", 1, 0.5, 100.0, 100.0), ("TR02", 1, 0.5, 100.0, 100.0),
                      ("TR02", 1, 0.5, 100.0, 300.0), ("TR02", 1, 0.5, 100.0, 300.0)])
    correction = fit_m_correction(rows, BY_SIZE, 200, 2)
    assert correction.log_m == pytest.approx((math.log(100.0),))
    assert correction.factors == pytest.approx((2.0,))
    corrected = apply_m_correction(rows, correction)
    assert corrected.sum() == pytest.approx(800.0)
    assert numbers_learned(PCorrection(NONE), correction) == 1


def test_fit_m_by_size_merges_only_the_groups_that_tie():
    # Three groups by m: (100, 100), (100, 100), (500, 900). The first two merge: 600 / 400 =
    # 1.5 at log 100; the third stays: 700 / 1400 = 0.5 at the mean of log 500 and log 900.
    rows = make_rows([("TR02", 1, 0.5, 100.0, 100.0), ("TR02", 1, 0.5, 100.0, 200.0),
                      ("TR02", 1, 0.5, 100.0, 100.0), ("TR02", 1, 0.5, 100.0, 200.0),
                      ("TR02", 1, 0.5, 500.0, 300.0), ("TR02", 1, 0.5, 900.0, 400.0)])
    correction = fit_m_correction(rows, BY_SIZE, 200, 3)
    assert correction.log_m == pytest.approx(
        (math.log(100.0), (math.log(500.0) + math.log(900.0)) / 2))
    assert correction.factors == pytest.approx((1.5, 0.5))
    assert numbers_learned(PCorrection(NONE), correction) == 2
    # The merged rows add up to their actual, 600.
    corrected = apply_m_correction(rows.iloc[:4], correction)
    assert corrected.sum() == pytest.approx(600.0)


def test_fit_m_by_size_positions_are_strictly_increasing_on_random_rows():
    rng = np.random.default_rng(20261108)
    for _ in range(200):
        size = int(rng.integers(4, 40))
        # Few distinct values, so that ties are common; some below 1.
        m = rng.choice([0.3, 1.0, 5.0, 5.0, 50.0, 50.0, 50.0, 400.0], size=size)
        actual = rng.uniform(1.0, 100.0, size=size)
        rows = make_rows([("TR02", 1, 0.5, float(a), float(b)) for a, b in zip(m, actual)])
        groups = int(rng.integers(2, 6))
        try:
            correction = fit_m_correction(rows, BY_SIZE, 200, groups)
        except ValueError as error:
            assert "cannot be cut" in str(error)
            continue
        assert len(correction.log_m) == len(correction.factors) <= groups
        assert all(b > a for a, b in zip(correction.log_m, correction.log_m[1:]))
        assert all(np.isfinite(correction.factors))
        # One position: the rows of m >= 1 add up to their actual.
        if len(correction.factors) == 1:
            sized = rows[rows["m"] >= 1.0]
            corrected = apply_m_correction(sized, correction)
            assert corrected.sum() == pytest.approx(sized["actual_real"].sum())


def test_actual_over_m_raises_if_m_adds_up_to_0():
    with pytest.raises(ValueError, match="m adds up to 0.0 on 2 paid rows"):
        actual_over_m(pd.DataFrame({"actual_real": [10.0, 5.0], "m": [0.0, 0.0]}))


def test_actual_over_m_raises_if_m_adds_up_below_0():
    with pytest.raises(ValueError, match="no factor can be learned"):
        actual_over_m(pd.DataFrame({"actual_real": [10.0], "m": [-3.0]}))


@pytest.mark.parametrize("kind", [SCALE, PER_COMPANY, BY_SIZE])
def test_fit_m_with_every_paid_m_0_raises_clearly(kind):
    rows = make_rows([("TR02", 1, 0.5, 0.0, 10.0), ("TR04", 1, 0.5, 0.0, 20.0),
                      ("TR02", 1, 0.5, 0.0, 30.0)])
    with pytest.raises(ValueError, match="m adds up to 0.0 on 3 paid rows"):
        fit_m_correction(rows, kind, 200, 2)


def test_fit_m_per_company_with_one_company_of_m_0_raises():
    # The common factor is fine (m adds up to 40); TR04's own is not.
    rows = make_rows([("TR02", 1, 0.5, 40.0, 10.0), ("TR04", 1, 0.5, 0.0, 20.0)])
    with pytest.raises(ValueError, match="m adds up to 0.0 on 1 paid rows"):
        fit_m_correction(rows, PER_COMPANY, 200, 2)
    assert fit_m_correction(rows, SCALE, 200, 2).common == pytest.approx(30.0 / 40.0)


def test_fit_m_by_size_too_few_paid_rows_raises():
    with pytest.raises(ValueError, match="cannot be cut"):
        fit_m_correction(by_size_rows(), BY_SIZE, 200, 5)


@pytest.mark.parametrize("kind", [SCALE, PER_COMPANY, BY_SIZE])
def test_fit_m_with_no_paid_rows_raises(kind):
    rows = m_rows()
    with pytest.raises(ValueError, match="no paid rows"):
        fit_m_correction(rows[~rows["paid"]], kind, 200, 2)


@pytest.mark.parametrize("kind", ["one map", "per h", "", None])
def test_fit_m_another_kind_raises(kind):
    with pytest.raises(ValueError, match="the correction of m is"):
        fit_m_correction(m_rows(), kind, 200, 2)


def test_fit_m_another_kind_is_named_before_the_paid_rows_are_looked_at():
    rows = m_rows()
    with pytest.raises(ValueError, match="the correction of m is"):
        fit_m_correction(rows[~rows["paid"]], "scales", 200, 2)


@pytest.mark.parametrize("kind", M_KINDS)
def test_apply_m_does_not_change_the_rows(kind):
    rows = m_rows()
    before = rows.copy()
    apply_m_correction(rows, fit_m_correction(rows, kind, 200, 2))
    pd.testing.assert_frame_equal(rows, before)


@pytest.mark.parametrize("kind", ["typo", "", None, "per_h", "scale"])
def test_apply_p_an_unknown_kind_raises(kind):
    rows = pd.DataFrame({"p": [0.5], "h": [1]})
    with pytest.raises(ValueError, match="the correction of p is"):
        apply_p_correction(rows, PCorrection(kind, {1: (1.0, 1.0), ALL_H: (1.0, 1.0)}))


@pytest.mark.parametrize("kind", ["typo", "", None, "scales", "one map"])
def test_apply_m_an_unknown_kind_raises(kind):
    rows = pd.DataFrame({"m": [10.0], "company_code": ["TR02"]})
    with pytest.raises(ValueError, match="the correction of m is"):
        apply_m_correction(rows, MCorrection(kind, common=3.0))


def test_corrected_forecast_an_unknown_kind_raises():
    with pytest.raises(ValueError, match="the correction of m is 'typo'"):
        corrected_forecast(worked(), PCorrection(NONE), MCorrection("typo", common=3.0))
    with pytest.raises(ValueError, match="the correction of p is 'typo'"):
        corrected_forecast(worked(), PCorrection("typo", {1: (1.0, 0.0), 2: (1.0, 0.0)}),
                           MCorrection(NONE))


def test_corrections_are_frozen():
    with pytest.raises(AttributeError):
        PCorrection(NONE).kind = "x"
    with pytest.raises(AttributeError):
        MCorrection(NONE).common = 2.0


# --- corrected_forecast ----------------------------------------------------------------------

def test_corrected_forecast_with_none_for_both_equals_the_input_exactly():
    rolled = worked(index=[50, 40, 30, 20, 10, 0, 70, 60])
    found = corrected_forecast(rolled, PCorrection(NONE), MCorrection(NONE))
    pd.testing.assert_frame_equal(found.totals, rolled.totals, check_exact=True)
    pd.testing.assert_frame_equal(found.customers, rolled.customers, check_exact=True)


def test_corrected_forecast_with_none_for_both_on_the_validation_world_equals_it_exactly():
    rolled = validation_case().rolled
    found = corrected_forecast(rolled, PCorrection(NONE), MCorrection(NONE))
    pd.testing.assert_frame_equal(found.totals, rolled.totals, check_exact=True)
    pd.testing.assert_frame_equal(found.customers, rolled.customers, check_exact=True)


def test_corrected_forecast_only_customers_real_total_and_total_change():
    rolled = worked()
    found = corrected_forecast(rolled, PCorrection(NONE), MCorrection(SCALE, 2.0)).totals
    assert list(found.columns) == TOTAL_COLUMNS
    unchanged = ["company_code", "origin_week", "h", "target_week", "new_customer_cash",
                 "unidentified_cash", "refunds", "price_level", "fallback"]
    pd.testing.assert_frame_equal(found[unchanged], rolled.totals[unchanged], check_exact=True)
    assert (found["customers"] == 2 * rolled.totals["customers"]).all()
    assert (found["real_total"] != rolled.totals["real_total"]).all()
    assert (found["total"] != rolled.totals["total"]).all()


def test_corrected_forecast_keeps_each_fallback():
    rolled = worked()
    found = corrected_forecast(rolled, PCorrection(NONE), MCorrection(SCALE, 2.0)).totals
    assert row_of(found, "TR04", 11, 1)["fallback"]
    assert row_of(found, CONSOLIDATED, 11, 1)["fallback"]
    assert not row_of(found, "TR04", 10, 1)["fallback"]
    assert not row_of(found, "TR02", 11, 1)["fallback"]


def test_corrected_forecast_each_origin_keeps_its_own_price_levels_by_hand():
    found = corrected_forecast(worked(), PCorrection(NONE), MCorrection(SCALE, 2.0)).totals
    # Target week 12 from origin 10, h 2: TR02's 0.4 x 200 x 2 = 160, + 9 = 169, level 1.2.
    early = row_of(found, "TR02", 10, 2)
    assert early["target_week"] == 12
    assert early["price_level"] == 1.2
    assert early["customers"] == pytest.approx(160.0)
    assert early["real_total"] == pytest.approx(169.0)
    assert early["total"] == pytest.approx(169.0 * 1.2)
    # The same target week from origin 11, h 1: TR02's 0.6 x 10 x 2 = 12, + 9 = 21, level 1.5.
    late = row_of(found, "TR02", 11, 1)
    assert late["target_week"] == 12
    assert late["price_level"] == 1.5
    assert late["customers"] == pytest.approx(12.0)
    assert late["real_total"] == pytest.approx(21.0)
    assert late["total"] == pytest.approx(21.0 * 1.5)


def test_corrected_forecast_the_all_rows_are_the_sum_of_the_companies_by_hand():
    found = corrected_forecast(worked(), PCorrection(NONE), MCorrection(SCALE, 2.0)).totals
    # Origin 11, h 1: TR02 12 and 21; TR04 0.3 x 20 x 2 = 12 and 12 + 18 = 30. Level 1.5.
    row = row_of(found, CONSOLIDATED, 11, 1)
    assert row["customers"] == pytest.approx(24.0)
    assert row["real_total"] == pytest.approx(51.0)
    assert row["total"] == pytest.approx(51.0 * 1.5)
    assert row["new_customer_cash"] == 30.0
    assert row["unidentified_cash"] == 3.0
    assert row["refunds"] == -6.0
    assert row["price_level"] == 1.5


def test_corrected_forecast_every_all_row_adds_up():
    found = corrected_forecast(worked(), PCorrection(ONE_MAP, {ALL_H: (1.2, 0.3)}),
                               MCorrection(PER_COMPANY, 1.5, {"TR02": 2.0, "TR04": 0.5})).totals
    companies = found[found["company_code"] != CONSOLIDATED]
    summed = companies.groupby(["origin_week", "h"])[["customers", "real_total", "total"]].sum()
    consolidated = found[found["company_code"] == CONSOLIDATED].set_index(["origin_week", "h"])
    for column in ("customers", "real_total", "total"):
        pd.testing.assert_series_equal(consolidated[column].sort_index(),
                                       summed[column].sort_index())


def test_corrected_forecast_the_all_rows_are_rebuilt_not_kept():
    # Garbage in the consolidated rows of the input must not reach the output.
    rolled = worked()
    good = corrected_forecast(rolled, PCorrection(NONE), MCorrection(NONE)).totals
    totals = rolled.totals.copy()
    is_all = totals["company_code"] == CONSOLIDATED
    for column in ("customers", "new_customer_cash", "unidentified_cash", "refunds", "real_total",
                   "price_level", "total"):
        totals.loc[is_all, column] = -999.0
    found = corrected_forecast(Forecast(totals=totals, customers=rolled.customers),
                               PCorrection(NONE), MCorrection(NONE)).totals
    pd.testing.assert_frame_equal(found, good, check_exact=True)


def test_corrected_forecast_expected_is_the_corrected_p_times_the_corrected_m():
    rolled = worked()
    # Odds x 3 for every p; m x 2.
    found = corrected_forecast(rolled, PCorrection(ONE_MAP, {ALL_H: (1.0, math.log(3.0))}),
                               MCorrection(SCALE, 2.0)).customers
    old_p = rolled.customers["p"]
    wanted_p = 3 * old_p / (1 - old_p) / (1 + 3 * old_p / (1 - old_p))
    assert list(found["p"]) == pytest.approx(list(wanted_p))
    assert list(found["m"]) == pytest.approx(list(rolled.customers["m"] * 2))
    assert (found["expected"] == found["p"] * found["m"]).all()


def test_corrected_forecast_customers_keep_their_rows_columns_and_index():
    index = [50, 40, 30, 20, 10, 0, 70, 60]
    rolled = worked(index=index)
    found = corrected_forecast(rolled, PCorrection(ONE_MAP, {ALL_H: (1.0, 1.0)}),
                               MCorrection(SCALE, 2.0)).customers
    assert list(found.index) == index
    assert list(found.columns) == CUSTOMER_COLUMNS
    same = ["key", "company_code", "customer_id", "origin_week", "h", "target_week"]
    pd.testing.assert_frame_equal(found[same], rolled.customers[same], check_exact=True)


def test_corrected_forecast_the_totals_follow_the_corrected_expected():
    rolled = worked()
    found = corrected_forecast(rolled, PCorrection(PER_H, {1: (1.0, 0.5), 2: (0.9, -0.2)}),
                               MCorrection(BY_SIZE, log_m=(2.0, 5.0), factors=(1.5, 0.7)))
    sums = found.customers.groupby(["company_code", "origin_week", "h"])["expected"].sum()
    for (company, origin, h), expected in sums.items():
        assert row_of(found.totals, company, origin, h)["customers"] == pytest.approx(expected)


def test_corrected_forecast_does_not_change_its_input():
    rolled = worked()
    totals, customers = rolled.totals.copy(), rolled.customers.copy()
    corrected_forecast(rolled, PCorrection(ONE_MAP, {ALL_H: (2.0, 1.0)}), MCorrection(SCALE, 2.0))
    pd.testing.assert_frame_equal(rolled.totals, totals, check_exact=True)
    pd.testing.assert_frame_equal(rolled.customers, customers, check_exact=True)


def test_corrected_forecast_customers_of_an_origin_with_no_totals_raises():
    rolled = worked()
    totals = rolled.totals[rolled.totals["origin_week"] == 10]
    with pytest.raises(ValueError, match=r"customers are for origins \[10, 11\].*\[10\]"):
        corrected_forecast(Forecast(totals=totals, customers=rolled.customers),
                           PCorrection(NONE), MCorrection(NONE))


def test_corrected_forecast_totals_of_an_origin_with_no_customers_raises():
    rolled = worked()
    customers = rolled.customers[rolled.customers["origin_week"] == 11]
    with pytest.raises(ValueError, match=r"customers are for origins \[11\].*\[10, 11\]"):
        corrected_forecast(Forecast(totals=rolled.totals, customers=customers),
                           PCorrection(NONE), MCorrection(NONE))


def test_corrected_forecast_a_company_with_no_customer_row_raises():
    rolled = worked()
    customers = rolled.customers[~((rolled.customers["company_code"] == "TR04")
                                   & (rolled.customers["origin_week"] == 11))]
    with pytest.raises(ValueError, match="no rows for"):
        corrected_forecast(Forecast(totals=rolled.totals, customers=customers),
                           PCorrection(NONE), MCorrection(NONE))


def test_corrected_forecast_per_h_without_an_h_of_the_rows_raises():
    with pytest.raises(ValueError, match=r"learned nothing for h \[2\]"):
        corrected_forecast(worked(), PCorrection(PER_H, {1: (1.0, 0.0)}), MCorrection(NONE))


# --- calibration_split -----------------------------------------------------------------------

def test_split_of_the_validation_origins():
    # First target 34, last 44 + 5 = 49, middle (34 + 49) // 2 = 41.
    split_week, judged = calibration_split(list(range(33, 45)), 5)
    assert split_week == 41
    assert judged == [41, 42, 43, 44]


def test_split_returns_plain_ints():
    split_week, judged = calibration_split(list(np.arange(33, 45)), 5)
    assert type(split_week) is int
    assert type(judged) is list
    assert all(type(origin) is int for origin in judged)


def test_split_of_unsorted_and_repeated_origins_is_the_same():
    origins = [44, 33, 41, 41, 37, 42, 44, 43, 34, 35, 36, 38, 39, 40]
    assert calibration_split(origins, 5) == (41, [41, 42, 43, 44])


def test_split_does_not_change_the_list():
    origins = [44, 33, 44]
    calibration_split(origins, 5)
    assert origins == [44, 33, 44]


def test_split_an_origin_exactly_at_the_middle_is_judged():
    # First target 11, last 13, middle 12: origin 12's targets are all after the middle.
    assert calibration_split([10, 11, 12], 1) == (12, [12])


def test_split_of_two_origins_with_one_horizon():
    assert calibration_split([10, 11], 1) == (11, [11])


def test_split_a_judged_origin_has_every_target_week_after_the_split():
    for horizons in (1, 2, 5, 13):
        for last in range(11, 60, 7):
            origins = list(range(10, last))
            try:
                split_week, judged = calibration_split(origins, horizons)
            except ValueError:
                continue
            assert min(judged) + 1 > split_week
            assert set(judged) <= set(origins)


def test_split_rounds_the_middle_down():
    # First target 11, last 14: (11 + 14) // 2 = 12, not 13.
    assert calibration_split([10, 11, 12, 13], 1) == (12, [12, 13])


@pytest.mark.parametrize("origins, horizons", [([10], 5), ([10], 1), ([10, 11], 5),
                                               (list(range(10, 13)), 13)])
def test_split_with_no_origin_past_the_middle_raises(origins, horizons):
    with pytest.raises(ValueError, match="nothing to judge"):
        calibration_split(origins, horizons)


@pytest.mark.parametrize("horizons", [0, -1])
def test_split_with_horizons_below_1_raises(horizons):
    with pytest.raises(ValueError, match="horizons start at 1"):
        calibration_split([10, 11, 12], horizons)


def test_split_with_no_origins_raises():
    with pytest.raises(ValueError, match="no origins"):
        calibration_split([], 5)


# --- candidate_name and numbers_learned ------------------------------------------------------

def test_candidate_name():
    assert candidate_name(PCorrection(NONE), MCorrection(NONE)) == "p none, m none"
    assert (candidate_name(PCorrection(PER_H), MCorrection(BY_SIZE))
            == "p per h, m by size")


FIVE_H = {h: (1.0, 0.0) for h in range(1, 6)}
FIVE_COMPANIES = {f"TR0{n}": 1.0 for n in range(1, 6)}
P_OF_KIND = {NONE: PCorrection(NONE),
             ONE_MAP: PCorrection(ONE_MAP, {ALL_H: (1.0, 0.0)}),
             PER_H: PCorrection(PER_H, FIVE_H)}
M_OF_KIND = {NONE: MCorrection(NONE),
             SCALE: MCorrection(SCALE, 1.1),
             # `per company` and `by size` hold a common factor too: it is not counted.
             PER_COMPANY: MCorrection(PER_COMPANY, 1.1, per_company=FIVE_COMPANIES),
             BY_SIZE: MCorrection(BY_SIZE, 1.1, log_m=tuple(float(n) for n in range(10)),
                                  factors=tuple(1.0 for _ in range(10)))}
P_NUMBERS = {NONE: 0, ONE_MAP: 2, PER_H: 10}
M_NUMBERS = {NONE: 0, SCALE: 1, PER_COMPANY: 5, BY_SIZE: 10}


@pytest.mark.parametrize("p_kind", P_KINDS)
@pytest.mark.parametrize("m_kind", M_KINDS)
def test_numbers_learned_is_the_sum_of_both(p_kind, m_kind):
    found = numbers_learned(P_OF_KIND[p_kind], M_OF_KIND[m_kind])
    assert found == P_NUMBERS[p_kind] + M_NUMBERS[m_kind]


def test_numbers_learned_of_a_scale_of_exactly_one_is_still_one():
    assert numbers_learned(PCorrection(NONE), MCorrection(SCALE, 1.0)) == 1


def test_numbers_learned_on_real_fits():
    rows = pd.concat([platt_rows(h=1), platt_rows(h=2)], ignore_index=True)
    # Every m different, so the three groups have three medians.
    rows["m"] = 1.0 + np.arange(len(rows), dtype="float64")
    p_correction = fit_p_correction(rows, PER_H)
    m_correction = fit_m_correction(rows, BY_SIZE, 200, 3)
    assert len(m_correction.factors) == 3
    assert numbers_learned(p_correction, m_correction) == 2 * 2 + 3


def test_numbers_learned_counts_the_merged_groups_of_a_real_fit():
    # platt_rows has m = 10 on every row: the three groups are one.
    rows = pd.concat([platt_rows(h=1), platt_rows(h=2)], ignore_index=True)
    m_correction = fit_m_correction(rows, BY_SIZE, 200, 3)
    assert len(m_correction.factors) == 1
    assert numbers_learned(fit_p_correction(rows, PER_H), m_correction) == 2 * 2 + 1


# --- simplest_good_enough --------------------------------------------------------------------

def table(*rows):
    return pd.DataFrame(rows, columns=["candidate", "numbers_learned", "wape"])


# The real example: the first rows of the table the pipeline printed.
REAL = table(("none+none", 0, 0.1703), ("none+scale", 1, 0.1630), ("none+company", 5, 0.1630),
             ("none+size", 10, 0.1627), ("one+none", 2, 0.1800), ("one+scale", 3, 0.1640),
             ("one+company", 7, 0.1640), ("one+size", 12, 0.1635), ("h+none", 10, 0.1810),
             ("h+scale", 11, 0.1650), ("h+company", 15, 0.1650), ("h+size", 20, 0.1645))


@pytest.mark.parametrize("min_gain, winner", [(0.005, "none+scale"), (0.0001, "none+size"),
                                              (0.01, "none+none")])
def test_simplest_good_enough_the_real_example(min_gain, winner):
    assert simplest_good_enough(REAL, min_gain)["candidate"] == winner


def test_simplest_good_enough_returns_the_whole_row():
    found = simplest_good_enough(REAL, 0.005)
    assert isinstance(found, pd.Series)
    assert found["numbers_learned"] == 1
    assert found["wape"] == 0.1630


def test_simplest_good_enough_the_lowest_wape_is_taken_over_all_rows_not_the_simplest():
    # The lowest is the last and largest; only the row within the margin of it counts.
    rows = table(("a", 0, 0.9), ("b", 5, 0.8), ("c", 50, 0.1))
    assert simplest_good_enough(rows, 0.05)["candidate"] == "c"


def test_simplest_good_enough_fewest_numbers_not_the_lowest_wape_in_the_margin():
    rows = table(("complex", 9, 0.100), ("middle", 4, 0.101), ("simple", 1, 0.102),
                 ("out", 0, 0.200))
    assert simplest_good_enough(rows, 0.01)["candidate"] == "simple"


def test_simplest_good_enough_a_tie_in_numbers_goes_to_the_first_row():
    rows = table(("first", 2, 0.11), ("second", 2, 0.10), ("third", 2, 0.105), ("big", 9, 0.10))
    assert simplest_good_enough(rows, 0.05)["candidate"] == "first"


def test_simplest_good_enough_a_tie_goes_to_the_first_row_whatever_the_index():
    rows = table(("first", 2, 0.11), ("second", 2, 0.10))
    rows.index = [9, 3]
    assert simplest_good_enough(rows, 0.05)["candidate"] == "first"


def test_simplest_good_enough_exactly_at_the_margin_is_kept():
    # Numbers a binary float holds exactly: 0.25 + 0.125 is exactly 0.375.
    rows = table(("complex", 9, 0.25), ("simple", 1, 0.375))
    assert simplest_good_enough(rows, 0.125)["candidate"] == "simple"


def test_simplest_good_enough_just_past_the_margin_is_out():
    rows = table(("complex", 9, 0.25), ("simple", 1, np.nextafter(0.375, 1.0)))
    assert simplest_good_enough(rows, 0.125)["candidate"] == "complex"


def test_simplest_good_enough_just_inside_the_margin_is_kept():
    rows = table(("complex", 9, 0.25), ("simple", 1, np.nextafter(0.375, 0.0)))
    assert simplest_good_enough(rows, 0.125)["candidate"] == "simple"


def test_simplest_good_enough_the_margin_is_a_difference_not_a_share():
    # 0.56 is 0.06 above 0.5: inside a margin of 0.1, outside one of 0.05.
    rows = table(("complex", 9, 0.5), ("simple", 1, 0.56))
    assert simplest_good_enough(rows, 0.1)["candidate"] == "simple"
    assert simplest_good_enough(rows, 0.05)["candidate"] == "complex"


@pytest.mark.parametrize("position", [0, 1, 2])
def test_simplest_good_enough_a_nan_wape_raises_naming_it(position):
    wapes = [0.1, 0.2, 0.3]
    wapes[position] = float("nan")
    rows = table(("a", 0, wapes[0]), ("b", 1, wapes[1]), ("c", 2, wapes[2]))
    with pytest.raises(ValueError, match="no WAPE for .*" + "abc"[position]):
        simplest_good_enough(rows, 0.01)


def test_simplest_good_enough_does_not_change_the_table():
    before = REAL.copy()
    simplest_good_enough(REAL, 0.005)
    pd.testing.assert_frame_equal(REAL, before)


def test_simplest_good_enough_on_random_tables_keeps_its_rule():
    rng = np.random.default_rng(20261107)
    for _ in range(200):
        size = int(rng.integers(1, 15))
        rows = table(*[(str(number), int(rng.integers(0, 6)), float(rng.uniform(0.1, 0.3)))
                       for number in range(size)])
        min_gain = float(rng.uniform(0.0001, 0.1))
        found = simplest_good_enough(rows, min_gain)
        good = rows[rows["wape"] <= rows["wape"].min() + min_gain]
        assert found["candidate"] in set(good["candidate"])
        assert found["numbers_learned"] == good["numbers_learned"].min()
        # The first of the good rows of that size.
        same_size = good[good["numbers_learned"] == found["numbers_learned"]]
        assert found["candidate"] == same_size.iloc[0]["candidate"]


# --- candidate_forecasts and candidate_scores ------------------------------------------------

def two_by_two():
    return ([PCorrection(NONE), PCorrection(ONE_MAP, {ALL_H: (1.0, math.log(3.0))})],
            [MCorrection(NONE), MCorrection(SCALE, 2.0)])


def test_candidate_forecasts_has_every_pair_in_the_shape_of_a_stack():
    p_corrections, m_corrections = two_by_two()
    found = candidate_forecasts(worked(), p_corrections, m_corrections)
    assert list(found.columns) == STACK_COLUMNS
    assert list(dict.fromkeys(found["method"])) == ["p none, m none", "p none, m scale",
                                                    "p one map, m none", "p one map, m scale"]
    # 2 origins x 2 h x (2 companies + ALL) rows per pair.
    assert list(found.groupby("method", sort=False).size()) == [12, 12, 12, 12]


def test_candidate_forecasts_forecast_is_the_nominal_total_of_each_pair():
    rolled = worked()
    p_corrections, m_corrections = two_by_two()
    found = candidate_forecasts(rolled, p_corrections, m_corrections)
    mine = found[found["method"] == "p one map, m scale"]
    totals = corrected_forecast(rolled, p_corrections[1], m_corrections[1]).totals
    assert list(mine["forecast"]) == list(totals["total"])
    plain = found[found["method"] == "p none, m none"]
    assert list(plain["forecast"]) == list(rolled.totals["total"])


def scores_case():
    case = validation_case()
    judged = Forecast(totals=case.rolled.totals[case.rolled.totals["origin_week"].isin(JUDGED)],
                      customers=case.rolled.customers[
                          case.rolled.customers["origin_week"].isin(JUDGED)])
    actual = case.actual[case.actual["origin_week"].isin(JUDGED)]
    p_corrections = [PCorrection(NONE)]
    m_corrections = [MCorrection(NONE), MCorrection(SCALE, 2.0)]
    forecasts = candidate_forecasts(judged, p_corrections, m_corrections)
    return forecasts, actual, p_corrections, m_corrections


def test_candidate_scores_keeps_the_columns_and_the_order_of_the_lists():
    forecasts, actual, p_corrections, m_corrections = scores_case()
    found = candidate_scores(forecasts, actual, p_corrections, m_corrections, HORIZONS)
    assert list(found.columns) == CANDIDATE_COLUMNS
    assert list(found["candidate"]) == ["p none, m none", "p none, m scale"]
    assert list(found["p"]) == [NONE, NONE]
    assert list(found["m"]) == [NONE, SCALE]
    assert list(found["numbers_learned"]) == [0, 1]


def test_candidate_scores_the_wape_is_the_consolidated_pooled_one_by_hand():
    found = candidate_scores(*scores_case(), HORIZONS)
    assert found["wape"].iloc[0] == pytest.approx(wape_by_hand(466.0))
    assert found["wape"].iloc[1] == pytest.approx(wape_by_hand(926.0))
    # Worked out on paper: 45661 / 53350 and 38071 / 53350.
    assert found["wape"].iloc[0] == pytest.approx(45661.0 / 53350.0)
    assert found["wape"].iloc[1] == pytest.approx(38071.0 / 53350.0)


def test_candidate_scores_total_wape_bias_and_company_wape_by_hand():
    row = candidate_scores(*scores_case(), HORIZONS).iloc[1]    # p none, m scale
    made = sum(926.0 * level(week) for week in JUDGED_WEEKS)
    assert row["bias"] == pytest.approx(made / 53350.0 - 1)
    # Per origin, forecast against actual: 15: 4907.8 against 20150; 16: 5093.0 against 21750;
    # 17: 5278.2 against 11450.
    assert row["total_wape"] == pytest.approx((15242.2 + 16657.0 + 6171.8) / 53350.0)
    # TR02, 2 x 180 + 5 = 365 a week, against 2600, 1350, 1350, 4200, 4200, 1450.
    # TR04, 2 x 280 + 1 = 561 a week, against 0, 16200, 16200, 0, 0, 5800.
    tr02 = 9127.5 / 15150.0
    tr04 = 38143.9 / 38200.0
    assert row["company_wape"] == pytest.approx((tr02 + tr04) / 2)


def test_candidate_scores_total_wape_is_nan_without_a_complete_origin():
    forecasts, actual, p_corrections, m_corrections = scores_case()
    # Only h = 1: no origin has both its weeks.
    actual = actual[actual["h"] == 1]
    found = candidate_scores(forecasts, actual, p_corrections, m_corrections, HORIZONS)
    assert found["total_wape"].isna().all()
    assert found["wape"].notna().all()


def test_candidate_scores_a_wape_is_nan_if_no_cash_came():
    forecasts, actual, p_corrections, m_corrections = scores_case()
    actual = actual.assign(actual=0.0)
    found = candidate_scores(forecasts, actual, p_corrections, m_corrections, HORIZONS)
    assert found["wape"].isna().all()


# --- choose_calibration: the shape of the choice ---------------------------------------------

def test_choose_returns_a_choice_with_the_split():
    found = choose(validation_case())
    assert isinstance(found, CalibrationChoice)
    assert found.split_week == SPLIT
    assert found.judged_origins == JUDGED
    assert type(found.split_week) is int
    assert all(type(origin) is int for origin in found.judged_origins)


def test_choose_scores_all_twelve_pairs_in_the_order_of_the_kinds():
    candidates = choose(validation_case()).candidates
    assert list(candidates.columns) == CANDIDATE_COLUMNS
    assert len(candidates) == 12
    wanted = [(p, m) for p in P_KINDS for m in M_KINDS]
    assert list(zip(candidates["p"], candidates["m"])) == wanted
    assert candidates["candidate"].is_unique
    assert candidates["wape"].notna().all()


def test_choose_counts_the_numbers_of_each_pair():
    candidates = choose(validation_case(), groups=3).candidates
    # Two h, two companies, three groups.
    p_numbers = {NONE: 0, ONE_MAP: 2, PER_H: 4}
    m_numbers = {NONE: 0, SCALE: 1, PER_COMPANY: 2, BY_SIZE: 3}
    for _, row in candidates.iterrows():
        assert row["numbers_learned"] == p_numbers[row["p"]] + m_numbers[row["m"]]


def test_choose_the_wapes_of_the_candidates_by_hand():
    choice = choose(validation_case())
    assert candidate_row(choice, NONE, NONE)["wape"] == pytest.approx(45661.0 / 53350.0)
    # The factor learned on the early rows is 2 in every company and every group, so these
    # three are one forecast.
    for kind in (SCALE, PER_COMPANY, BY_SIZE):
        assert candidate_row(choice, NONE, kind)["wape"] == pytest.approx(38071.0 / 53350.0)


def test_choose_does_not_change_its_inputs():
    case = validation_case()
    before = {name: getattr(case, name).copy() for name in ("weekly", "weeks", "levels", "actual")}
    totals, customers = case.rolled.totals.copy(), case.rolled.customers.copy()
    choose(case)
    pd.testing.assert_frame_equal(case.rolled.totals, totals, check_exact=True)
    pd.testing.assert_frame_equal(case.rolled.customers, customers, check_exact=True)
    pd.testing.assert_frame_equal(case.weekly, before["weekly"])
    pd.testing.assert_frame_equal(case.weeks, before["weeks"])
    pd.testing.assert_frame_equal(case.actual, before["actual"])
    pd.testing.assert_series_equal(case.levels, before["levels"])


def test_choose_is_repeatable():
    first, second = choose(validation_case()), choose(validation_case())
    pd.testing.assert_frame_equal(first.candidates, second.candidates, check_exact=True)
    pd.testing.assert_frame_equal(first.bootstrap, second.bootstrap, check_exact=True)
    assert first.p_correction == second.p_correction
    assert first.m_correction == second.m_correction


# --- choose_calibration: the winner and the bootstrap ----------------------------------------

def test_choose_a_large_min_gain_means_no_correction_and_an_empty_bootstrap():
    choice = choose(validation_case(), min_gain=10.0)
    assert choice.winner == "p none, m none"
    assert choice.p_correction == PCorrection(NONE)
    assert choice.m_correction == MCorrection(NONE)
    assert choice.bootstrap.empty
    assert list(choice.bootstrap.columns) == BOOTSTRAP_COLUMNS


def test_choose_a_correction_that_wins_has_only_the_pooled_bootstrap_row(monkeypatch):
    force_winner(monkeypatch, "p none, m scale")
    choice = choose(validation_case())
    assert list(choice.bootstrap["horizon"]) == [POOLED]
    row = choice.bootstrap.iloc[0]
    assert row["method"] == "p none, m scale"
    assert row["against"] == "p none, m none"
    assert row["wape"] == pytest.approx(38071.0 / 53350.0)
    assert row["wape_against"] == pytest.approx(45661.0 / 53350.0)
    assert row["samples"] == 50


def test_choose_the_real_bootstrap_has_both_measures_before_the_filter(monkeypatch):
    # So the filter matters: the bootstrap is asked for the rows that it then leaves out.
    seen = {}
    real = calibrate.paired_wape_bootstrap

    def spy(*args, **kwargs):
        seen["table"] = real(*args, **kwargs)
        return seen["table"]
    monkeypatch.setattr(calibrate, "paired_wape_bootstrap", spy)
    force_winner(monkeypatch, "p none, m scale")
    choice = choose(validation_case())
    assert set(seen["table"]["horizon"]) == {POOLED, HORIZON_TOTAL}
    assert list(choice.bootstrap["horizon"]) == [POOLED]


def test_choose_the_bootstrap_gets_the_settings_and_only_the_judged_rows(monkeypatch):
    seen = {}
    real = calibrate.paired_wape_bootstrap

    def spy(forecasts, actual, pairs, horizons, samples, seed):
        seen.update(forecasts=forecasts, actual=actual, pairs=pairs, horizons=horizons,
                    samples=samples, seed=seed)
        return real(forecasts, actual, pairs, horizons, samples, seed)
    monkeypatch.setattr(calibrate, "paired_wape_bootstrap", spy)
    force_winner(monkeypatch, "p one map, m by size")
    choose(validation_case(), samples=17, seed=23)
    assert seen["pairs"] == [("p one map, m by size", "p none, m none")]
    assert (seen["horizons"], seen["samples"], seen["seed"]) == (HORIZONS, 17, 23)
    assert set(seen["actual"]["origin_week"]) == set(JUDGED)
    assert set(seen["forecasts"]["origin_week"]) == set(JUDGED)


# The WAPEs on this world, to three places. No correction of m: 0.856, 0.828 (one map), 0.827 (per
# h). Any correction of m: 0.714, 0.668 (one map), 0.663 (per h). The three corrections of m are
# one forecast here, as all their factors are 2.
@pytest.mark.parametrize("min_gain, winner, frozen_kinds", [
    (0.06, "p none, m scale", (NONE, SCALE)),        # 0.714 is within 0.06 of 0.663
    (0.02, "p one map, m scale", (ONE_MAP, SCALE)),  # 0.668 is, 0.714 is not
    (0.001, "p per h, m scale", (PER_H, SCALE)),     # only the lowest three; scale is smallest
])
def test_choose_a_natural_winner_is_the_simplest_within_the_margin(min_gain, winner, frozen_kinds):
    choice = choose(validation_case(), min_gain=min_gain)
    assert choice.winner == winner
    assert (choice.p_correction.kind, choice.m_correction.kind) == frozen_kinds
    assert list(choice.bootstrap["horizon"]) == [POOLED]
    assert choice.bootstrap.iloc[0]["method"] == winner


def test_choose_a_natural_scale_winner_freezes_the_scale_of_all_the_rows_by_hand():
    choice = choose(validation_case(), min_gain=0.06)
    assert choice.winner == "p none, m scale"
    assert choice.m_correction.common == pytest.approx(30700.0 / 9200.0)
    assert choice.p_correction == PCorrection(NONE)


@pytest.mark.parametrize("p_kind", P_KINDS)
@pytest.mark.parametrize("m_kind", M_KINDS)
def test_choose_the_frozen_kinds_are_the_winners(monkeypatch, p_kind, m_kind):
    name = f"p {p_kind}, m {m_kind}"
    force_winner(monkeypatch, name)
    choice = choose(validation_case())
    assert choice.winner == name
    assert choice.p_correction.kind == p_kind
    assert choice.m_correction.kind == m_kind


# --- choose_calibration: no leak -------------------------------------------------------------

def forced_scale(monkeypatch, case, **settings):
    force_winner(monkeypatch, "p none, m scale")
    return choose(case, **settings)


def frame_with_receipt(weekly, key, week, receipts):
    """`weekly` with the receipts of one key in one week replaced, or added."""
    found = weekly.copy()
    here = (found["key"] == key) & (found["week"] == week)
    if here.any():
        found.loc[here, "gross_receipts"] = receipts
        return found
    company = CUSTOMERS[key][0]
    return pd.concat([found, make_weekly([(key, company, week, receipts, 0.0)])],
                     ignore_index=True)


def test_choose_the_frozen_scale_is_learned_on_all_the_rows_by_hand(monkeypatch):
    choice = forced_scale(monkeypatch, validation_case())
    assert choice.m_correction.common == pytest.approx(30700.0 / 9200.0)


def test_choose_the_compared_scale_is_learned_on_the_early_rows_only(monkeypatch):
    # The score of `p none, m scale` is that of a factor of exactly 2, not 3.34.
    choice = forced_scale(monkeypatch, validation_case())
    found = candidate_row(choice, NONE, SCALE)["wape"]
    assert found == pytest.approx(wape_by_hand(2 * 460.0 + 6))
    assert found != pytest.approx(wape_by_hand(30700.0 / 9200.0 * 460.0 + 6))


def test_choose_the_frozen_per_company_is_learned_on_all_the_rows_by_hand(monkeypatch):
    force_winner(monkeypatch, "p none, m per company")
    choice = choose(validation_case(), shrink_rows=10)
    # TR02's paid rows. K1: 7 early, 5 late, m 100. K2: 4 early, 4 late, m 200. Early m 1500,
    # late m 1300, 20 rows; own factor (2 x 1500 + 5 x 1300) / 2800.
    # TR04's. K3: 5 early, 3 late, m 400. K4: 2 early, 2 late, m 800. Early m 3600, late 2800,
    # 12 rows; own factor (2 x 3600 + 5 x 2800) / 6400.
    common = 30700.0 / 9200.0
    tr02 = (2 * 1500 + 5 * 1300) / 2800
    tr04 = (2 * 3600 + 5 * 2800) / 6400
    assert choice.m_correction.common == pytest.approx(common)
    assert choice.m_correction.per_company["TR02"] == pytest.approx((20 * tr02 + 10 * common) / 30)
    assert choice.m_correction.per_company["TR04"] == pytest.approx((12 * tr04 + 10 * common) / 22)


def test_choose_the_frozen_one_map_is_learned_on_all_the_rows(monkeypatch):
    force_winner(monkeypatch, "p one map, m none")
    case = validation_case()
    choice = choose(case)
    rows = calibration_rows(case.rolled.customers, case.weekly, case.weeks, case.levels)
    assert choice.p_correction == fit_p_correction(rows, ONE_MAP)
    early = fit_p_correction(rows[rows["target_week"] <= SPLIT], ONE_MAP)
    assert choice.p_correction != early


def test_choose_the_compared_one_map_is_learned_on_the_early_rows_only():
    case = validation_case()
    choice = choose(case)
    rows = calibration_rows(case.rolled.customers, case.weekly, case.weeks, case.levels)
    early = fit_p_correction(rows[rows["target_week"] <= SPLIT], ONE_MAP)
    # Worked out from that map: every customer's p goes through it, m stays, top-ups stay.
    a, b = early.maps[ALL_H]
    real_total = 6.0
    for _, p, m in CUSTOMERS.values():
        real_total += sigmoid(a * logit(p) + b) * m
    assert candidate_row(choice, ONE_MAP, NONE)["wape"] == pytest.approx(wape_by_hand(real_total))


@pytest.mark.parametrize("week", [16, 17, 18, 19])
def test_choose_receipts_after_the_split_do_not_move_any_candidate(week):
    # `actual` stays as it was: only what the corrections could learn from changes. K2 pays in
    # even weeks, K3 in odd weeks, and a receipt appears for K4, who did not pay: p would see it.
    case = validation_case()
    baseline = choose(case).candidates
    weekly = frame_with_receipt(case.weekly, "K2" if week % 2 == 0 else "K3", week, 1e9)
    if week in (16, 18, 19):
        weekly = frame_with_receipt(weekly, "K4", week, 1e9)
    case.weekly = weekly
    pd.testing.assert_frame_equal(choose(case).candidates, baseline, check_exact=True)


def test_choose_receipts_after_the_split_do_move_the_frozen_scale(monkeypatch):
    # The control of the test above: the frozen correction does see those weeks. K2's receipts
    # in week 16, aimed at by two rows, become 5 x 200 x level + 1000 x level.
    case = validation_case()
    extra = 1000.0
    case.weekly = frame_with_receipt(case.weekly, "K2", 16, (5 * 200.0 + extra) * level(16))
    choice = forced_scale(monkeypatch, case)
    assert choice.m_correction.common == pytest.approx((30700.0 + 2 * extra) / 9200.0)
    assert candidate_row(choice, NONE, SCALE)["wape"] == pytest.approx(38071.0 / 53350.0)


def test_choose_receipts_exactly_at_the_split_do_move_the_candidates():
    # The first week on the other side: K3 pays in week 15, aimed at by two rows.
    case = validation_case()
    baseline = choose(case).candidates
    # 2 x 400 x level becomes 3 x 400 x level, so the paid m stays 5100, and the actual 800 more.
    case.weekly = frame_with_receipt(case.weekly, "K3", 15, 3 * 400.0 * level(15))
    found = choose(case).candidates
    factor = (2 * 5100.0 + 800.0) / 5100.0
    wanted = wape_by_hand(factor * 460.0 + 6)
    assert candidate_row(SimpleNamespace(candidates=found), NONE, SCALE)["wape"] == pytest.approx(
        wanted)
    assert not found["wape"].equals(baseline["wape"])


def with_m_of_row(case, origin, h, key, m):
    customers = case.rolled.customers.copy()
    here = ((customers["origin_week"] == origin) & (customers["h"] == h)
            & (customers["key"] == key))
    assert here.sum() == 1
    customers.loc[here, "m"] = m
    customers.loc[here, "expected"] = customers.loc[here, "p"] * m
    case.rolled = Forecast(totals=case.rolled.totals, customers=customers)
    return case


def test_choose_a_row_of_an_earlier_origin_aiming_past_the_split_is_used_for_neither(monkeypatch):
    # Origin 14, h 2 aims at week 16: origin 14 is not judged, week 16 is not learned. K2 paid.
    baseline = forced_scale(monkeypatch, validation_case())
    case = with_m_of_row(validation_case(), 14, 2, "K2", 1e6)
    found = forced_scale(monkeypatch, case)
    pd.testing.assert_frame_equal(found.candidates, baseline.candidates, check_exact=True)
    # But the frozen correction, learned on every row, does see it.
    wanted = 30700.0 / (9200.0 - 200.0 + 1e6)
    assert found.m_correction.common == pytest.approx(wanted)


def test_choose_a_row_of_an_earlier_origin_aiming_up_to_the_split_is_learned_from():
    # Origin 13, h 2 aims at week 15: K3 paid. m 400 becomes 4400: the early m 5100 becomes 9100,
    # with the actual unchanged, 2 x 5100.
    baseline = choose(validation_case()).candidates
    found = choose(with_m_of_row(validation_case(), 13, 2, "K3", 4400.0)).candidates
    assert not found["wape"].equals(baseline["wape"])
    factor = 2 * 5100.0 / 9100.0
    wanted = wape_by_hand(factor * 460.0 + 6)
    choice = SimpleNamespace(candidates=found)
    assert candidate_row(choice, NONE, SCALE)["wape"] == pytest.approx(wanted)


def test_choose_a_row_of_a_judged_origin_is_not_learned_from():
    # Origin 16, h 1 aims at week 17: K3 paid. It is judged: its forecast changes, as only that
    # (origin, h) moves. The factor stays 2. The customers are changed, and the totals follow.
    case = with_m_of_row(validation_case(), 16, 1, "K3", 500.0)
    # Consistent totals for the changed customers: they are rebuilt from them in any case.
    choice = choose(case)
    # Expected 460 + 0.5 x 100 = 510 at this (origin, h), so 2 x 510 + 6 = 1026 against week 17.
    errors = (abs(926 * 2.6 - 2600) + abs(926 * 2.7 - 17550) + abs(1026 * 2.7 - 17550)
              + 2 * abs(926 * 2.8 - 4200) + abs(926 * 2.9 - 7250))
    assert candidate_row(choice, NONE, SCALE)["wape"] == pytest.approx(errors / 53350.0)


def test_choose_a_different_receipt_in_the_future_with_a_new_actual_changes_only_the_score():
    # Hindsight may change the score; it may not change what was learned.
    case = validation_case()
    case.weekly = frame_with_receipt(case.weekly, "K2", 16, 2000.0 * level(16))
    case.actual = actual_totals(case.weekly, case.weeks, case.levels, COMPANIES, ORIGINS, HORIZONS)
    choice = choose(case)
    # Week 16's actual goes from 1000 x 2.6 to 2000 x 2.6 = 5200; the factor stays 2.
    error = (abs(926 * 2.6 - 5200) + 2 * abs(926 * 2.7 - 17550) + 2 * abs(926 * 2.8 - 4200)
             + abs(926 * 2.9 - 7250))
    assert candidate_row(choice, NONE, SCALE)["wape"] == pytest.approx(error / (53350.0 + 2600.0))


# --- choose_calibration: missing weeks and impossible inputs ---------------------------------

def test_choose_a_missing_week_among_the_judged_is_not_scored():
    choice = choose(validation_case(missing=(18,)))
    weeks = (16, 17, 17, 19)
    assert candidate_row(choice, NONE, SCALE)["wape"] == pytest.approx(wape_by_hand(926.0, weeks))
    assert candidate_row(choice, NONE, NONE)["wape"] == pytest.approx(wape_by_hand(466.0, weeks))
    # Only origin 15 still has both weeks (16 and 17): 4907.8 against 20150.
    assert candidate_row(choice, NONE, SCALE)["total_wape"] == pytest.approx(15242.2 / 20150.0)


def test_choose_a_missing_week_among_the_early_ones_is_not_learned_from(monkeypatch):
    # Week 14 is missing: whatever is in it does not count. It held K1 and K2, 2 rows each,
    # m 600 in all, so the frozen scale loses those: early 4500, late 4100.
    case = validation_case(missing=(14,))
    case.weekly = frame_with_receipt(case.weekly, "K1", 14, 1e9)
    choice = forced_scale(monkeypatch, case)
    assert candidate_row(choice, NONE, SCALE)["wape"] == pytest.approx(38071.0 / 53350.0)
    assert choice.m_correction.common == pytest.approx((2 * 4500.0 + 5 * 4100.0) / 8600.0)


def test_choose_with_no_row_to_learn_from_raises():
    with pytest.raises(ValueError, match="no rows with data up to target week 15"):
        choose(validation_case(missing=(11, 12, 13, 14, 15)))


def test_choose_with_too_few_origins_raises():
    case = validation_case()
    totals = case.rolled.totals
    customers = case.rolled.customers
    case.rolled = Forecast(totals=totals[totals["origin_week"] == 10],
                           customers=customers[customers["origin_week"] == 10])
    with pytest.raises(ValueError, match="nothing to judge"):
        choose(case)


def test_choose_with_customers_of_other_origins_than_the_totals_raises():
    case = validation_case()
    customers = case.rolled.customers
    case.rolled = Forecast(totals=case.rolled.totals,
                           customers=customers[customers["origin_week"] != 16])
    with pytest.raises(ValueError, match="customers are for origins"):
        choose(case)


def test_choose_with_a_nan_wape_raises():
    # No cash came at all in the judged weeks: a WAPE of nothing.
    case = validation_case()
    case.actual = case.actual.assign(actual=0.0)
    with pytest.raises(ValueError, match="no WAPE for"):
        choose(case)


def test_candidate_scores_with_no_consolidated_cash_raises_a_value_error():
    forecasts, actual, p_corrections, m_corrections = scores_case()
    with pytest.raises(ValueError, match="no consolidated cash in the judged weeks"):
        candidate_scores(forecasts, actual.iloc[:0], p_corrections, m_corrections, HORIZONS)


def test_candidate_scores_with_only_company_rows_raises_a_value_error():
    forecasts, actual, p_corrections, m_corrections = scores_case()
    companies = actual[actual["company_code"] != CONSOLIDATED]
    with pytest.raises(ValueError, match="no consolidated cash in the judged weeks"):
        candidate_scores(forecasts, companies, p_corrections, m_corrections, HORIZONS)


def test_choose_with_every_judged_week_missing_raises_a_value_error():
    with pytest.raises(ValueError, match="no consolidated cash in the judged weeks"):
        choose(validation_case(missing=(16, 17, 18, 19)))


def zero_week_case(value=0.0):
    """The world with every actual of week 16 set to 0, and the consolidated one to `value`."""
    case = validation_case()
    actual = case.actual.copy()
    actual.loc[actual["target_week"] == 16, "actual"] = 0.0
    actual.loc[(actual["target_week"] == 16) & (actual["company_code"] == CONSOLIDATED),
               "actual"] = value
    case.actual = actual
    return case


@pytest.mark.parametrize("value", [0.0, -50.0])
def test_choose_no_correction_with_a_judged_actual_of_0_or_less_does_not_raise(monkeypatch,
                                                                              value):
    case = zero_week_case(value)
    called = []
    monkeypatch.setattr(calibrate, "paired_wape_bootstrap",
                        lambda *args, **kwargs: called.append(1))
    choice = choose(case, min_gain=10.0)
    assert choice.winner == "p none, m none"
    assert called == []
    assert choice.bootstrap.empty
    assert list(choice.bootstrap.columns) == BOOTSTRAP_COLUMNS
    assert choice.p_correction == PCorrection(NONE) and choice.m_correction == MCorrection(NONE)


def test_choose_no_correction_with_a_judged_actual_of_0_and_the_real_bootstrap():
    # The same without the spy: the real path does not raise either.
    choice = choose(zero_week_case(), min_gain=10.0)
    assert choice.winner == "p none, m none"
    assert choice.bootstrap.empty


def test_choose_no_correction_never_calls_the_bootstrap(monkeypatch):
    called = []
    monkeypatch.setattr(calibrate, "paired_wape_bootstrap",
                        lambda *args, **kwargs: called.append(1))
    choose(validation_case(), min_gain=10.0)
    assert called == []


def test_choose_a_correction_with_a_judged_actual_of_0_still_raises_in_the_bootstrap(monkeypatch):
    # The other side: when a correction wins the bootstrap runs, and it needs every
    # consolidated actual above 0.
    force_winner(monkeypatch, "p none, m scale")
    with pytest.raises(ValueError, match="every consolidated actual must be above 0"):
        choose(zero_week_case())


def test_choose_with_too_many_groups_for_the_early_rows_raises():
    # 18 paid rows up to the split (7 + 4 + 5 + 2); more groups than rows cannot be cut.
    with pytest.raises(ValueError, match="cannot be cut"):
        choose(validation_case(), groups=19)
