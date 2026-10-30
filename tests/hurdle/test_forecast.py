"""Tests for `hurdle/forecast.py` (U13.2): `customer_forecasts`.

`customer_forecasts(table, origin, settings, block_columns, companies)` fits stage 1 on
`rows_known_at(table, origin)` and stage 2 on the paid rows among them (exposure learned from
the known rows), then predicts p and m for `rows_at_origin(table, origin)`. It returns the
columns CUSTOMER_COLUMNS with the table's index and expected = p x m, and raises ValueError
for no rows at the origin, a NaN / infinite p, m or expected, an m below 0, and as the fits do.

A small hand-made training table (8 customers in two companies, origins 0..30, h 1..5; target
weeks after 23 unlabelled) and few trees, so every forecast takes well under a second.
Nothing reads the corpus.

U13.3, at the end: `forecast_totals` (with `_check_pairs`, `TOTAL_COLUMNS`, `CONSOLIDATED`),
pure bookkeeping, tested on small hand-made customers / top-ups / levels tables.
"""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import forecast as forecast_module
from collection_estimation.hurdle.forecast import (
    CONSOLIDATED, CUSTOMER_COLUMNS, TOTAL_COLUMNS, _check_pairs, customer_forecasts,
    forecast_totals)
from collection_estimation.hurdle.prices import projected_levels
from collection_estimation.hurdle.topups import TOPUP_COLUMNS
from collection_estimation.hurdle.settings import FEATURE_BLOCKS, Stage1Settings, Stage2Settings
from collection_estimation.hurdle.stage1 import (
    feature_columns, fit_stage1, predict_paid, stage1_features)
from collection_estimation.hurdle.stage2 import exposure, fit_stage2, predict_amount
from collection_estimation.hurdle.training_table import rows_at_origin, rows_known_at

nan = float("nan")
inf = float("inf")

COMPANIES = ["TR02", "TR04", "TR05"]
BLOCK_COLUMNS = {
    "history": ["mean_receipts_4", "paying_weeks_4", "observed_weeks_4"],
    "calendar": ["business_days"],
    "receivables": ["open_balance", "due_in_target_week"],
}
ORIGIN = 20          # the default origin of the tests
LAST_LABELLED = 23   # target weeks after this are "future" (not yet known when built)
HORIZONS = 5


def stage1_settings(blocks=FEATURE_BLOCKS, company_as_feature=True, max_iter=10, seed=3):
    return Stage1Settings(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=7,
                          min_samples_leaf=5, random_seed=seed, feature_blocks=blocks,
                          company_as_feature=company_as_feature, baseline_c=1.0)


def stage2_settings(blocks=FEATURE_BLOCKS, loss="poisson", scale="none",
                    balance_floor="median", max_iter=10, seed=3):
    return Stage2Settings(loss=loss, max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=7,
                          min_samples_leaf=5, random_seed=seed, feature_blocks=blocks,
                          scale=scale, balance_floor=balance_floor, own_mean_window=4,
                          own_mean_prior_weeks=1.0)


def make_settings(stage1=None, stage2=None):
    """Only `[stage1]` and `[stage2]` are read, so a namespace with those two will do."""
    return SimpleNamespace(stage1=stage1 or stage1_settings(),
                           stage2=stage2 or stage2_settings())


def make_table(n_customers=8, last_origin=30, seed=0, index_offset=5000):
    """A training table: one row per customer, origin t (0..last_origin) and h (1..5).

    paid depends on open_balance and due_in_target_week; amount_deflated is above 0 exactly
    when paid. Rows with target week after LAST_LABELLED are "future", their labels NaN.
    The index starts at `index_offset` and is shuffled, so it is not a RangeIndex from 0."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        company = COMPANIES[c % 2]
        size = 1000.0 * (c + 1)
        for t in range(last_origin + 1):
            for h in range(1, HORIZONS + 1):
                open_balance = max(0.0, rng.normal(size, size / 3))
                due = open_balance * rng.random() if rng.random() < 0.5 else 0.0
                score = -1.0 + 2.5 * (due > 0) + 0.3 * rng.normal()
                paid = bool(score > 0) if rng.random() > 0.15 else bool(rng.random() < 0.5)
                amount = float(rng.lognormal(np.log(0.3 * size + 50.0), 0.4)) if paid else 0.0
                rows.append({
                    "key": f"{company}|C{c}", "company_code": company, "customer_id": f"C{c}",
                    "origin_week": t, "h": h, "target_week": t + h,
                    "label_status": "labelled", "amount": amount * 1.2,
                    "amount_deflated": amount, "paid": paid,
                    # A window with no observed week has no mean (NaN), as in the real table.
                    "mean_receipts_4": (np.nan if t == 0 else float(rng.lognormal(6, 1))),
                    "paying_weeks_4": float(rng.integers(0, min(t, 4) + 1)),
                    "observed_weeks_4": float(min(t, 4)),
                    "business_days": float(5 - (t % 7 == 3)),
                    "open_balance": open_balance, "due_in_target_week": due,
                })
    table = pd.DataFrame(rows)
    future = table["target_week"] > LAST_LABELLED
    table.loc[future, "label_status"] = "future"
    table["paid"] = table["paid"].astype(object)
    table.loc[future, ["amount", "amount_deflated", "paid"]] = np.nan
    table.index = pd.Index(index_offset + rng.permutation(len(table)))
    return table


@pytest.fixture(scope="module")
def table():
    return make_table()


def run(table, origin=ORIGIN, settings=None, block_columns=BLOCK_COLUMNS, companies=COMPANIES):
    return customer_forecasts(table, origin, settings or make_settings(), block_columns,
                              companies)


def by_hand(table, origin, settings, block_columns=BLOCK_COLUMNS, companies=COMPANIES):
    """The same steps, called one by one: what customer_forecasts must equal."""
    s1, s2 = settings.stage1, settings.stage2
    train = rows_known_at(table, origin)
    at = rows_at_origin(table, origin)
    c1 = feature_columns(block_columns, s1.feature_blocks)
    model1 = fit_stage1(stage1_features(train, c1, companies, s1.company_as_feature),
                        train["paid"], s1)
    p = predict_paid(model1, stage1_features(at, c1, companies, s1.company_as_feature))
    c2 = feature_columns(block_columns, s2.feature_blocks)
    paid = train[train["paid"].astype(bool)]
    model2 = fit_stage2(stage1_features(paid, c2, companies, s1.company_as_feature),
                        paid["amount_deflated"], exposure(train, paid, s2), s2)
    m = predict_amount(model2, stage1_features(at, c2, companies, s1.company_as_feature),
                       exposure(train, at, s2))
    return p, m, model1, model2


@pytest.fixture(scope="module")
def result(table):
    return run(table)


# --- the premises of the hand-made table ------------------------------------------------------

def test_table_premises(table):
    known = rows_known_at(table, ORIGIN)
    at = rows_at_origin(table, ORIGIN)
    assert len(at) == 8 * HORIZONS
    assert known["paid"].astype(bool).nunique() == 2
    assert (known["target_week"] <= ORIGIN).all()
    # Some rows at the origin are "future" (target week > 23) and carry NaN labels.
    assert (at["label_status"] == "future").any()
    assert at.loc[at["label_status"] == "future", "paid"].isna().all()


# --- CUSTOMER_COLUMNS -----------------------------------------------------------------------------

def test_customer_columns_are_the_documented_list():
    assert CUSTOMER_COLUMNS == ["key", "company_code", "customer_id", "origin_week", "h",
                                "target_week", "p", "m", "expected"]


# --- shape ------------------------------------------------------------------------------------

def test_result_has_the_customer_columns_in_order(result):
    assert list(result.columns) == CUSTOMER_COLUMNS


def test_result_has_one_row_per_row_at_the_origin_with_its_index(table, result):
    at = rows_at_origin(table, ORIGIN)
    assert result.index.equals(at.index)
    assert len(result) == len(at)


def test_result_index_is_the_tables_not_a_new_range(table, result):
    assert result.index.min() >= 5000
    assert set(result.index) <= set(table.index)


def test_result_origin_week_is_the_origin_everywhere(result):
    assert (result["origin_week"] == ORIGIN).all()


def test_result_identity_columns_are_copied_from_the_table(table, result):
    at = rows_at_origin(table, ORIGIN)
    pd.testing.assert_frame_equal(result[CUSTOMER_COLUMNS[:6]], at[CUSTOMER_COLUMNS[:6]])


def test_result_target_week_is_origin_plus_h(result):
    assert (result["target_week"] == result["origin_week"] + result["h"]).all()


def test_result_every_customer_and_h_once(result):
    assert not result.duplicated(["key", "h"]).any()
    for _, horizons in result.groupby("key")["h"]:
        assert sorted(horizons) == list(range(1, HORIZONS + 1))


def test_result_expected_is_p_times_m_exactly(result):
    assert (result["expected"] == result["p"] * result["m"]).all()


def test_result_p_is_a_probability(result):
    assert ((result["p"] >= 0) & (result["p"] <= 1)).all()


def test_result_m_is_not_below_zero(result):
    assert (result["m"] >= 0).all()


def test_result_numbers_are_float64_and_finite(result):
    for column in ("p", "m", "expected"):
        assert result[column].dtype == np.float64
        assert np.isfinite(result[column]).all()


def test_result_holds_no_label_columns(result):
    for column in ("paid", "amount", "amount_deflated", "label_status"):
        assert column not in result.columns


# --- equivalence with the steps called by hand -----------------------------------------------

@pytest.mark.parametrize("origin", [10, ORIGIN, 25, 28])
def test_p_and_m_equal_the_steps_called_by_hand(table, origin):
    settings = make_settings()
    got = run(table, origin, settings)
    p, m, _, _ = by_hand(table, origin, settings)
    np.testing.assert_array_equal(got["p"].to_numpy(), p.to_numpy())
    np.testing.assert_array_equal(got["m"].to_numpy(), m.to_numpy())
    assert got.index.equals(p.index) and got.index.equals(m.index)


@pytest.mark.parametrize("scale,floor", [("open balance", "median"), ("open balance", "mean"),
                                         ("open balance", "own mean")])
def test_p_and_m_equal_the_steps_by_hand_with_open_balance_scale(table, scale, floor):
    settings = make_settings(stage2=stage2_settings(scale=scale, balance_floor=floor))
    got = run(table, ORIGIN, settings)
    p, m, _, _ = by_hand(table, ORIGIN, settings)
    np.testing.assert_array_equal(got["p"].to_numpy(), p.to_numpy())
    np.testing.assert_array_equal(got["m"].to_numpy(), m.to_numpy())


def test_p_and_m_equal_the_steps_by_hand_without_company(table):
    settings = make_settings(stage1=stage1_settings(company_as_feature=False))
    got = run(table, ORIGIN, settings)
    p, m, _, _ = by_hand(table, ORIGIN, settings)
    np.testing.assert_array_equal(got["p"].to_numpy(), p.to_numpy())
    np.testing.assert_array_equal(got["m"].to_numpy(), m.to_numpy())


def test_forecast_is_deterministic(table):
    pd.testing.assert_frame_equal(run(table), run(table))


def test_a_different_seed_changes_nothing_structural(table):
    settings = make_settings(stage1_settings(seed=11), stage2_settings(seed=11))
    got = run(table, ORIGIN, settings)
    assert list(got.columns) == CUSTOMER_COLUMNS and len(got) == 8 * HORIZONS


# --- no leak: rows with target week after T do not matter -------------------------------------

def test_labels_of_rows_with_target_week_after_the_origin_do_not_matter(table, result):
    changed = table.copy()
    later = (changed["target_week"] > ORIGIN) & (changed["label_status"] == "labelled")
    assert later.any()
    flipped = ~changed.loc[later, "paid"].astype(bool)
    changed.loc[later, "paid"] = flipped
    changed.loc[later, "amount_deflated"] = np.where(flipped, 9.9e6, 0.0)
    changed.loc[later, "amount"] = changed.loc[later, "amount_deflated"] * 3
    pd.testing.assert_frame_equal(run(changed), result)


def test_features_of_later_origins_do_not_matter(table, result):
    changed = table.copy()
    later = changed["origin_week"] > ORIGIN
    changed.loc[later, ["open_balance", "due_in_target_week", "mean_receipts_4"]] *= 1000.0
    pd.testing.assert_frame_equal(run(changed), result)


def test_dropping_every_row_after_the_origin_changes_nothing(table, result):
    kept = table[(table["origin_week"] <= ORIGIN)]
    pd.testing.assert_frame_equal(run(kept), result)


def test_forecast_rows_need_no_labels(table, result):
    # Every row at T loses its labels and becomes "future": same p and m.
    changed = table.copy()
    at = changed["origin_week"] == ORIGIN
    changed["paid"] = changed["paid"].astype(object)
    changed.loc[at, ["paid", "amount", "amount_deflated"]] = np.nan
    changed.loc[at, "label_status"] = "future"
    pd.testing.assert_frame_equal(run(changed), result)


def test_unlabelled_rows_before_the_origin_are_not_learned_from(table, result):
    # A row whose target week has ended but is not "labelled" is left out of the fit.
    changed = table.copy()
    extra = changed[changed["target_week"] <= ORIGIN].iloc[:30].copy()
    extra.index = extra.index + 100000
    extra["label_status"] = "unknown"
    extra["paid"] = True
    extra["amount_deflated"] = 1e9
    pd.testing.assert_frame_equal(run(pd.concat([changed, extra])), result)


def test_leak_check_has_teeth(table, result):
    # Changing a label whose target week HAS ended does move the forecast.
    changed = table.copy()
    known = (changed["target_week"] <= ORIGIN) & changed["paid"].astype(bool)
    changed.loc[known, "amount_deflated"] *= 10.0
    assert not run(changed)["m"].equals(result["m"])


# --- settings honoured --------------------------------------------------------------------------

def test_each_stage_is_fitted_on_its_own_feature_blocks(table, monkeypatch):
    seen = {}
    real1, real2 = forecast_module.fit_stage1, forecast_module.fit_stage2

    def spy1(x, paid, settings):
        seen["stage1"] = list(x.columns)
        return real1(x, paid, settings)

    def spy2(x, amount, exp, settings):
        seen["stage2"] = list(x.columns)
        return real2(x, amount, exp, settings)

    monkeypatch.setattr(forecast_module, "fit_stage1", spy1)
    monkeypatch.setattr(forecast_module, "fit_stage2", spy2)
    settings = make_settings(stage1_settings(blocks=("history",)),
                             stage2_settings(blocks=("calendar", "receivables")))
    run(table, ORIGIN, settings)
    assert seen["stage1"] == ["h", "mean_receipts_4", "paying_weeks_4", "observed_weeks_4",
                              "company_code"]
    assert seen["stage2"] == ["h", "business_days", "open_balance", "due_in_target_week",
                              "company_code"]


def test_stage2_company_follows_the_stage1_setting(table, monkeypatch):
    seen = {}
    real2 = forecast_module.fit_stage2

    def spy2(x, amount, exp, settings):
        seen["stage2"] = list(x.columns)
        return real2(x, amount, exp, settings)

    monkeypatch.setattr(forecast_module, "fit_stage2", spy2)
    run(table, ORIGIN, make_settings(stage1_settings(company_as_feature=False)))
    assert "company_code" not in seen["stage2"]


def test_stage2_with_a_block_stage1_lacks_runs_and_differs(table, result):
    settings = make_settings(stage1_settings(blocks=("history",)),
                             stage2_settings(blocks=FEATURE_BLOCKS))
    got = run(table, ORIGIN, settings)
    p, m, model1, model2 = by_hand(table, ORIGIN, settings)
    assert list(model1.feature_names_in_) == ["h", "mean_receipts_4", "paying_weeks_4",
                                              "observed_weeks_4", "company_code"]
    assert "open_balance" in model2.feature_names_in_
    np.testing.assert_array_equal(got["p"].to_numpy(), p.to_numpy())
    np.testing.assert_array_equal(got["m"].to_numpy(), m.to_numpy())
    # Stage 1 lost the receivables block: p moves; stage 2 kept every block: m is unchanged.
    assert not got["p"].equals(result["p"])
    np.testing.assert_array_equal(got["m"].to_numpy(), result["m"].to_numpy())


def test_no_blocks_at_all_fits_on_h_and_company_only(table):
    settings = make_settings(stage1_settings(blocks=()), stage2_settings(blocks=()))
    got = run(table, ORIGIN, settings)
    assert np.isfinite(got["expected"]).all()


def test_open_balance_scale_changes_m(table, result):
    got = run(table, ORIGIN, make_settings(stage2=stage2_settings(scale="open balance")))
    np.testing.assert_array_equal(got["p"].to_numpy(), result["p"].to_numpy())
    assert not np.allclose(got["m"].to_numpy(), result["m"].to_numpy())


def test_open_balance_scale_m_is_s_times_the_exposure(table):
    settings = make_settings(stage2=stage2_settings(scale="open balance"))
    got = run(table, ORIGIN, settings)
    _, _, _, model2 = by_hand(table, ORIGIN, settings)
    at = rows_at_origin(table, ORIGIN)
    train = rows_known_at(table, ORIGIN)
    x2 = stage1_features(at, feature_columns(BLOCK_COLUMNS, FEATURE_BLOCKS), COMPANIES, True)
    e = exposure(train, at, settings.stage2)
    assert (e != 1.0).any()
    np.testing.assert_array_equal(got["m"].to_numpy(), model2.predict(x2) * e.to_numpy())


@pytest.mark.parametrize("loss", ["poisson", "gamma"])
def test_log_link_losses_give_m_above_zero(table, loss):
    got = run(table, ORIGIN, make_settings(stage2=stage2_settings(loss=loss)))
    assert (got["m"] > 0).all()


def test_stage2_settings_do_not_change_p(table, result):
    got = run(table, ORIGIN, make_settings(stage2=stage2_settings(loss="gamma", max_iter=3)))
    np.testing.assert_array_equal(got["p"].to_numpy(), result["p"].to_numpy())


def test_stage1_settings_change_p(table, result):
    got = run(table, ORIGIN, make_settings(stage1=stage1_settings(max_iter=3)))
    assert not np.allclose(got["p"].to_numpy(), result["p"].to_numpy())


# --- errors -------------------------------------------------------------------------------

@pytest.mark.parametrize("origin", [31, 99, -1])
def test_no_rows_at_the_origin_exact_message(table, origin):
    with pytest.raises(ValueError) as error:
        run(table, origin)
    assert type(error.value) is ValueError
    assert str(error.value) == f"the table has no rows at origin week {origin}"


def test_no_rows_at_the_origin_is_checked_before_fitting(table, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("fit_stage1 must not be called")
    monkeypatch.setattr(forecast_module, "fit_stage1", boom)
    with pytest.raises(ValueError, match="no rows at origin week 99"):
        run(table, 99)


def fake_series(values_for):
    """A predict function returning a Series built from `values_for(index)`."""
    def predict(model, x, *args):
        return pd.Series(values_for(x.index), index=x.index)
    return predict


@pytest.mark.parametrize("bad", [nan, inf, -inf])
def test_p_nan_or_infinite_raises_counting_the_rows(table, monkeypatch, bad):
    def values(index):
        v = np.full(len(index), 0.5)
        v[:3] = bad
        return v
    monkeypatch.setattr(forecast_module, "predict_paid", fake_series(values))
    with pytest.raises(ValueError) as error:
        run(table)
    assert type(error.value) is ValueError
    assert str(error.value) == f"origin week {ORIGIN}: p is NaN or infinite in 3 rows"


@pytest.mark.parametrize("bad", [nan, inf])
def test_m_nan_or_infinite_raises_counting_the_rows(table, monkeypatch, bad):
    def values(index):
        v = np.full(len(index), 100.0)
        v[-2:] = bad
        return v
    monkeypatch.setattr(forecast_module, "predict_amount", fake_series(values))
    with pytest.raises(ValueError) as error:
        run(table)
    assert str(error.value) == f"origin week {ORIGIN}: m is NaN or infinite in 2 rows"


def test_expected_nan_from_zero_times_infinite_is_reported_for_p_first(table, monkeypatch):
    # p infinite is reported as p, before expected.
    monkeypatch.setattr(forecast_module, "predict_paid",
                        fake_series(lambda i: np.where(np.arange(len(i)) == 0, inf, 0.5)))
    with pytest.raises(ValueError, match="p is NaN or infinite in 1 rows"):
        run(table)


def test_expected_overflow_is_reported_for_expected(table, monkeypatch):
    # p and m finite, but p x m overflows to infinity: caught on expected.
    monkeypatch.setattr(forecast_module, "predict_paid", fake_series(lambda i: np.full(len(i), 1e300)))
    monkeypatch.setattr(forecast_module, "predict_amount",
                        fake_series(lambda i: np.full(len(i), 1e300)))
    with pytest.raises(ValueError) as error:
        run(table)
    assert str(error.value) == (f"origin week {ORIGIN}: expected is NaN or infinite in "
                                f"{8 * HORIZONS} rows")


def test_m_below_zero_raises_naming_the_loss(table, monkeypatch):
    monkeypatch.setattr(forecast_module, "predict_amount",
                        fake_series(lambda i: np.where(np.arange(len(i)) < 4, -1.0, 10.0)))
    settings = make_settings(stage2=stage2_settings(loss="squared_error"))
    with pytest.raises(ValueError) as error:
        run(table, ORIGIN, settings)
    assert type(error.value) is ValueError
    message = str(error.value)
    assert message.startswith(f"origin week {ORIGIN}: m is below 0 in 4 rows")
    assert "'squared_error'" in message
    assert "'poisson' and 'gamma' never predict one" in message


def test_m_of_exactly_zero_is_allowed(table, monkeypatch):
    monkeypatch.setattr(forecast_module, "predict_amount",
                        fake_series(lambda i: np.zeros(len(i))))
    got = run(table)
    assert (got["m"] == 0).all() and (got["expected"] == 0).all()


def test_nan_check_comes_before_the_negative_check(table, monkeypatch):
    monkeypatch.setattr(forecast_module, "predict_amount",
                        fake_series(lambda i: np.where(np.arange(len(i)) == 0, nan, -5.0)))
    with pytest.raises(ValueError, match="m is NaN or infinite in 1 rows"):
        run(table)


def squared_error_case():
    """A table where a real squared-error stage 2 predicts m below 0, with no monkeypatch.

    Stage 2 sees only h and business_days. On the paid training rows the amount is 1000 when
    h <= 2 and business_days is 5, and 1 when exactly one of "h >= 3" and "business_days 4"
    holds; no training row has both. Stumps (2 leaves a tree) make the model additive, so it
    learns about 1000 - 999 - 999 for that unseen corner, and the rows at the origin are put
    in it (business_days 4)."""
    table = make_table()
    known_paid = (table["target_week"] <= ORIGIN) & table["paid"].eq(True)
    train = table["target_week"] <= ORIGIN
    rng = np.random.default_rng(2)
    late = table["h"] >= 3
    short = pd.Series(rng.random(len(table)) < 0.5, index=table.index) & ~late
    table.loc[train, "business_days"] = np.where(short[train], 4.0, 5.0)
    table.loc[known_paid, "amount_deflated"] = np.where(
        late[known_paid] | short[known_paid], 1.0, 1000.0)
    table.loc[table["origin_week"] == ORIGIN, "business_days"] = 4.0
    settings = make_settings(
        stage1_settings(blocks=("history",), company_as_feature=False),
        dataclasses.replace(stage2_settings(blocks=("calendar",), loss="squared_error",
                                            max_iter=100), max_leaf_nodes=2))
    return table, settings


def test_squared_error_case_premise_the_real_fit_goes_below_zero():
    table, settings = squared_error_case()
    _, m, _, _ = by_hand(table, ORIGIN, settings)
    assert (m < 0).any()
    assert (m[rows_at_origin(table, ORIGIN)["h"] >= 3] < 0).all()


def test_squared_error_really_predicting_below_zero_raises_naming_the_loss():
    table, settings = squared_error_case()
    _, m, _, _ = by_hand(table, ORIGIN, settings)
    with pytest.raises(ValueError) as error:
        run(table, ORIGIN, settings)
    assert type(error.value) is ValueError
    assert str(error.value).startswith(
        f"origin week {ORIGIN}: m is below 0 in {int((m < 0).sum())} rows")
    assert "[stage2] loss 'squared_error'" in str(error.value)


def test_the_same_case_with_poisson_does_not_go_below_zero():
    table, settings = squared_error_case()
    settings = make_settings(settings.stage1, dataclasses.replace(settings.stage2,
                                                                  loss="poisson"))
    assert (run(table, ORIGIN, settings)["m"] > 0).all()


def test_one_class_in_paid_propagates_the_stage1_error(table):
    changed = table.copy()
    known = changed["label_status"] == "labelled"
    changed.loc[known, "paid"] = True
    changed.loc[known, "amount_deflated"] = 100.0
    with pytest.raises(ValueError, match="the classifier needs both outcomes"):
        run(changed)


def test_unknown_block_propagates_the_feature_columns_error(table):
    settings = make_settings(stage2=stage2_settings(blocks=("nope",)))
    with pytest.raises(ValueError, match="unknown feature blocks"):
        run(table, ORIGIN, settings)


def test_unknown_company_propagates_the_features_error(table):
    with pytest.raises(ValueError, match="not in the settings' company list"):
        run(table, companies=["TR02"])


def test_a_paid_row_with_no_amount_propagates_the_stage2_error(table):
    changed = table.copy()
    row = changed.index[(changed["target_week"] <= ORIGIN) & changed["paid"].eq(True)][0]
    changed.loc[row, "amount_deflated"] = 0.0
    with pytest.raises(ValueError, match="amount is 0 or below in 1 rows"):
        run(changed)


def test_unknown_stage2_loss_propagates(table):
    with pytest.raises(ValueError, match="stage 2 loss 'tweedie'"):
        run(table, ORIGIN, make_settings(stage2=stage2_settings(loss="tweedie")))


def test_missing_open_balance_at_the_origin_propagates_the_exposure_error(table):
    changed = table.copy()
    row = changed.index[changed["origin_week"] == ORIGIN][0]
    changed.loc[row, "open_balance"] = np.nan
    settings = make_settings(stage1_settings(blocks=("history",)),
                             stage2_settings(blocks=("history",), scale="open balance"))
    with pytest.raises(ValueError, match="open_balance is missing in 1 rows"):
        run(changed, ORIGIN, settings)


def test_origin_with_no_known_rows_raises(table):
    # At origin 0 no target week has ended: nothing to learn from.
    with pytest.raises(ValueError):
        run(table, 0)


# --- inputs are not changed --------------------------------------------------------------------

def test_does_not_change_the_table(table):
    before = table.copy()
    run(table, ORIGIN, make_settings(stage2=stage2_settings(scale="open balance")))
    pd.testing.assert_frame_equal(table, before)


def test_does_not_change_block_columns_or_companies(table):
    blocks = {name: list(columns) for name, columns in BLOCK_COLUMNS.items()}
    companies = list(COMPANIES)
    run(table, ORIGIN, make_settings(), blocks, companies)
    assert blocks == BLOCK_COLUMNS and companies == COMPANIES


def test_result_is_not_a_view_of_the_table(table):
    before = table.copy()
    got = run(table)
    got["customer_id"] = "X"
    got["h"] = -1
    pd.testing.assert_frame_equal(table, before)


def test_settings_are_frozen_dataclasses_and_left_alone():
    s1 = stage1_settings()
    assert dataclasses.is_dataclass(s1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s1.max_iter = 99


# =========================================================================================
# U13.3: forecast_totals
#
# A hand-made example whose answer is known: companies TR02 and TR04 (passed unsorted),
# horizons 3, origin 40, so target weeks 41-43, price levels 2.0, 4.0 and 0.5 (exact in
# binary, so every total is exact).
#
#   customers (expected per customer)     top-ups (new, refunds, unidentified, fallback)
#   TR02 h1: 100 + 50 = 150               TR02: (1, -4, 8, F) (2, -5, 16, T) (3, -6, 32, F)
#   TR02 h2: 10                           TR04: (10, -1, 0, F) (20, -1, 0, F) (30, -1, 0, F)
#   TR02 h3: 1 + 2 + 3 = 6
#   TR04 h1: 7    h2: 20 + 30 = 50    h3: 0
# =========================================================================================

T_ORIGIN = 40
T_COMPANIES = ["TR04", "TR02"]
T_HORIZONS = 3

CUSTOMER_EXPECTED = {("TR02", 1): [100.0, 50.0], ("TR02", 2): [10.0],
                     ("TR02", 3): [1.0, 2.0, 3.0], ("TR04", 1): [7.0],
                     ("TR04", 2): [20.0, 30.0], ("TR04", 3): [0.0]}
TOPUP_VALUES = {"TR02": [(1.0, -4.0, 8.0, False), (2.0, -5.0, 16.0, True),
                         (3.0, -6.0, 32.0, False)],
                "TR04": [(10.0, -1.0, 0.0, False), (20.0, -1.0, 0.0, False),
                         (30.0, -1.0, 0.0, False)]}
T_LEVELS = pd.Series({41: 2.0, 42: 4.0, 43: 0.5}, name="price_level")

# (company, h) -> customers, real_total, total
EXPECTED_TOTALS = {
    ("TR02", 1): (150.0, 155.0, 310.0), ("TR02", 2): (10.0, 23.0, 92.0),
    ("TR02", 3): (6.0, 35.0, 17.5), ("TR04", 1): (7.0, 16.0, 32.0),
    ("TR04", 2): (50.0, 69.0, 276.0), ("TR04", 3): (0.0, 29.0, 14.5),
    ("ALL", 1): (157.0, 171.0, 342.0), ("ALL", 2): (60.0, 92.0, 368.0),
    ("ALL", 3): (6.0, 64.0, 32.0),
}


def hand_customers(expected=CUSTOMER_EXPECTED, origin=T_ORIGIN, index_offset=700):
    """A customers table shaped like `customer_forecasts`' output (CUSTOMER_COLUMNS)."""
    rows = []
    for (company, h), values in expected.items():
        for i, value in enumerate(values):
            rows.append({"key": f"{company}|K{i}", "company_code": company,
                         "customer_id": f"K{i}", "origin_week": origin, "h": h,
                         "target_week": origin + h, "p": 0.5, "m": 2 * value,
                         "expected": value})
    frame = pd.DataFrame(rows, columns=CUSTOMER_COLUMNS)
    frame.index = pd.RangeIndex(index_offset, index_offset + len(frame))
    return frame


def hand_topups(values=TOPUP_VALUES, origin=T_ORIGIN):
    """A top-ups table shaped like `topups.topup_forecasts`' output (TOPUP_COLUMNS)."""
    rows = []
    for company, per_h in values.items():
        for h, (new, refunds, unidentified, fallback) in enumerate(per_h, start=1):
            rows.append({"company_code": company, "origin_week": origin, "h": h,
                         "target_week": origin + h, "new_customer_cash": new,
                         "refunds": refunds, "unidentified_cash": unidentified,
                         "topup": new + refunds + unidentified, "fallback": fallback})
    return pd.DataFrame(rows, columns=TOPUP_COLUMNS)


def totals(customers=None, topups=None, levels=None, companies=None, horizons=T_HORIZONS):
    return forecast_totals(hand_customers() if customers is None else customers,
                           hand_topups() if topups is None else topups,
                           T_LEVELS if levels is None else levels,
                           T_COMPANIES if companies is None else companies, horizons)


def totals_error(**kwargs):
    with pytest.raises(ValueError) as error:
        totals(**kwargs)
    assert type(error.value) is ValueError
    return str(error.value)


@pytest.fixture
def hand_totals():
    return totals()


def row(frame, company, h):
    found = frame[(frame["company_code"] == company) & (frame["h"] == h)]
    assert len(found) == 1
    return found.iloc[0]


# --- constants ---------------------------------------------------------------------------

def test_total_columns_are_the_documented_list():
    assert TOTAL_COLUMNS == ["company_code", "origin_week", "h", "target_week", "customers",
                             "new_customer_cash", "unidentified_cash", "refunds",
                             "real_total", "price_level", "total", "fallback"]


def test_consolidated_code_is_all():
    assert CONSOLIDATED == "ALL"


# --- _check_pairs -----------------------------------------------------------------------

def test_check_pairs_equal_sets_pass():
    _check_pairs("things", {("A", 1), ("B", 2)}, {("B", 2), ("A", 1)})


def test_check_pairs_missing_exact_message_sorted():
    with pytest.raises(ValueError) as error:
        _check_pairs("things", {("A", 1)}, {("A", 1), ("B", 2), ("A", 2)})
    assert str(error.value) == "the things have no rows for (company, h) [('A', 2), ('B', 2)]"


def test_check_pairs_extra_exact_message():
    with pytest.raises(ValueError) as error:
        _check_pairs("things", {("A", 1), ("Z", 9)}, {("A", 1)})
    assert str(error.value) == ("the things have rows for (company, h) [('Z', 9)], which are "
                                "not in the company list or the horizons")


def test_check_pairs_missing_is_reported_before_extra():
    with pytest.raises(ValueError, match="have no rows for"):
        _check_pairs("things", {("Z", 9)}, {("A", 1)})


def test_check_pairs_lists_at_most_ten():
    wanted = {("A", h) for h in range(1, 16)}
    with pytest.raises(ValueError) as error:
        _check_pairs("things", set(), wanted)
    assert str(error.value).count("('A',") == 10


def test_check_pairs_extra_with_mixed_types_does_not_crash_sorting():
    # sorted(..., key=str): a None company beside text must not raise TypeError.
    with pytest.raises(ValueError, match="which are not in"):
        _check_pairs("things", {("A", 1), (None, 1), ("B", 1.5)}, {("A", 1)})


# --- arithmetic ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", list(EXPECTED_TOTALS))
def test_totals_by_hand(hand_totals, pair):
    customers, real_total, total = EXPECTED_TOTALS[pair]
    found = row(hand_totals, *pair)
    assert found["customers"] == customers
    assert found["real_total"] == real_total
    assert found["total"] == total


def test_real_total_is_the_sum_of_the_four_parts(hand_totals):
    parts = (hand_totals["customers"] + hand_totals["new_customer_cash"]
             + hand_totals["unidentified_cash"] + hand_totals["refunds"])
    np.testing.assert_array_equal(hand_totals["real_total"].to_numpy(), parts.to_numpy())


def test_refunds_are_added_negative_not_subtracted(hand_totals):
    found = row(hand_totals, "TR04", 3)
    assert found["refunds"] == -1.0
    assert found["real_total"] == 0.0 + 30.0 + 0.0 - 1.0


def test_total_is_real_total_times_the_price_level(hand_totals):
    np.testing.assert_array_equal(hand_totals["total"].to_numpy(),
                                  (hand_totals["real_total"]
                                   * hand_totals["price_level"]).to_numpy())


def test_price_level_is_the_level_of_the_target_week(hand_totals):
    np.testing.assert_array_equal(hand_totals["price_level"].to_numpy(),
                                  hand_totals["target_week"].map(T_LEVELS).to_numpy())


def test_top_up_amounts_are_carried_unchanged(hand_totals):
    for company, per_h in TOPUP_VALUES.items():
        for h, (new, refunds, unidentified, fallback) in enumerate(per_h, start=1):
            found = row(hand_totals, company, h)
            assert (found["new_customer_cash"], found["refunds"],
                    found["unidentified_cash"]) == (new, refunds, unidentified)
            assert bool(found["fallback"]) is fallback


def test_customers_column_ignores_p_and_m(hand_totals):
    customers = hand_customers()
    customers["p"], customers["m"] = 0.99, 1e9
    pd.testing.assert_frame_equal(totals(customers=customers), hand_totals)


def test_levels_may_hold_more_weeks_than_needed(hand_totals):
    levels = pd.concat([pd.Series({30: 9.0, 99: 9.0}), T_LEVELS])
    pd.testing.assert_frame_equal(totals(levels=levels), hand_totals)


def test_negative_real_total_is_kept(hand_totals):
    # Large refunds can make a week negative: kept, not clipped.
    values = {**TOPUP_VALUES, "TR04": [(0.0, -500.0, 0.0, False)] + TOPUP_VALUES["TR04"][1:]}
    found = row(totals(topups=hand_topups(values)), "TR04", 1)
    assert found["real_total"] == 7.0 - 500.0
    assert found["total"] == (7.0 - 500.0) * 2.0


def test_float_customers_are_summed(hand_totals):
    # To rounding only: a groupby sum and Series.sum may add in a different order.
    expected = {**CUSTOMER_EXPECTED, ("TR02", 1): [0.1, 0.2, 0.3]}
    found = row(totals(customers=hand_customers(expected)), "TR02", 1)
    assert found["customers"] == pytest.approx(0.6, rel=1e-15)


# --- consolidated rows ------------------------------------------------------------------------

def test_consolidated_rows_one_per_h(hand_totals):
    consolidated = hand_totals[hand_totals["company_code"] == CONSOLIDATED]
    assert list(consolidated["h"]) == [1, 2, 3]


@pytest.mark.parametrize("column", ["customers", "new_customer_cash", "unidentified_cash",
                                    "refunds", "real_total", "total"])
def test_consolidated_amounts_are_the_sum_over_companies(hand_totals, column):
    companies = hand_totals[hand_totals["company_code"] != CONSOLIDATED]
    for h in range(1, T_HORIZONS + 1):
        assert row(hand_totals, CONSOLIDATED, h)[column] == (
            companies.loc[companies["h"] == h, column].sum())


def test_consolidated_price_level_and_weeks_are_the_companies(hand_totals):
    for h in range(1, T_HORIZONS + 1):
        found = row(hand_totals, CONSOLIDATED, h)
        assert found["price_level"] == T_LEVELS[T_ORIGIN + h]
        assert found["target_week"] == T_ORIGIN + h
        assert found["origin_week"] == T_ORIGIN


def test_consolidated_total_is_also_its_real_total_times_the_level(hand_totals):
    consolidated = hand_totals[hand_totals["company_code"] == CONSOLIDATED]
    np.testing.assert_allclose(consolidated["total"],
                               consolidated["real_total"] * consolidated["price_level"],
                               rtol=1e-15)


def test_consolidated_fallback_is_any_company(hand_totals):
    assert [bool(row(hand_totals, CONSOLIDATED, h)["fallback"]) for h in (1, 2, 3)] == [
        False, True, False]


def test_consolidated_fallback_true_when_every_company_falls_back():
    values = {company: [(n, r, u, True) for n, r, u, _ in per_h]
              for company, per_h in TOPUP_VALUES.items()}
    found = totals(topups=hand_topups(values))
    assert found["fallback"].astype(bool).all()


def test_one_company_consolidated_equals_it():
    customers = hand_customers({k: v for k, v in CUSTOMER_EXPECTED.items() if k[0] == "TR02"})
    topups = hand_topups({"TR02": TOPUP_VALUES["TR02"]})
    found = totals(customers=customers, topups=topups, companies=["TR02"])
    company = found[found["company_code"] == "TR02"].drop(columns="company_code")
    consolidated = found[found["company_code"] == CONSOLIDATED].drop(columns="company_code")
    pd.testing.assert_frame_equal(company.reset_index(drop=True),
                                  consolidated.reset_index(drop=True), check_dtype=False)


# --- shape and order ----------------------------------------------------------------------------

def test_columns_are_total_columns(hand_totals):
    assert list(hand_totals.columns) == TOTAL_COLUMNS


def test_topup_column_is_not_in_the_output(hand_totals):
    assert "topup" not in hand_totals.columns


def test_rows_companies_sorted_then_all_h_ascending(hand_totals):
    assert list(zip(hand_totals["company_code"], hand_totals["h"])) == [
        ("TR02", 1), ("TR02", 2), ("TR02", 3), ("TR04", 1), ("TR04", 2), ("TR04", 3),
        ("ALL", 1), ("ALL", 2), ("ALL", 3)]


def test_row_count_is_companies_plus_one_times_horizons(hand_totals):
    assert len(hand_totals) == (len(T_COMPANIES) + 1) * T_HORIZONS


def test_index_is_a_fresh_range(hand_totals):
    assert list(hand_totals.index) == list(range(len(hand_totals)))


def test_all_is_last_even_when_a_company_sorts_after_it():
    # "AAA" sorts before "ALL", "ZZZ" after; ALL still comes last.
    rename = {"TR02": "ZZZ", "TR04": "AAA"}
    customers = hand_customers().replace({"company_code": rename})
    topups = hand_topups().replace({"company_code": rename})
    found = totals(customers=customers, topups=topups, companies=["ZZZ", "AAA"])
    assert list(found["company_code"]) == ["AAA"] * 3 + ["ZZZ"] * 3 + ["ALL"] * 3


def test_origin_and_target_weeks_carried_from_the_top_ups(hand_totals):
    assert (hand_totals["origin_week"] == T_ORIGIN).all()
    assert (hand_totals["target_week"] == T_ORIGIN + hand_totals["h"]).all()


def test_company_list_order_does_not_matter(hand_totals):
    pd.testing.assert_frame_equal(totals(companies=["TR02", "TR04"]), hand_totals)


def test_input_row_order_does_not_matter(hand_totals):
    customers = hand_customers().sample(frac=1.0, random_state=3)
    topups = hand_topups().sample(frac=1.0, random_state=4)
    pd.testing.assert_frame_equal(totals(customers=customers, topups=topups), hand_totals)


def test_horizons_one_gives_one_row_per_company_and_all():
    customers = hand_customers({k: v for k, v in CUSTOMER_EXPECTED.items() if k[1] == 1})
    topups = hand_topups({c: v[:1] for c, v in TOPUP_VALUES.items()})
    found = totals(customers=customers, topups=topups, horizons=1)
    assert list(found["company_code"]) == ["TR02", "TR04", "ALL"]
    assert row(found, "ALL", 1)["total"] == 342.0


def test_amount_columns_are_float(hand_totals):
    for column in ("customers", "new_customer_cash", "unidentified_cash", "refunds",
                   "real_total", "price_level", "total"):
        assert hand_totals[column].dtype == np.float64


def test_inputs_are_not_changed():
    customers, topups, levels = hand_customers(), hand_topups(), T_LEVELS.copy()
    companies = list(T_COMPANIES)
    before = customers.copy(), topups.copy(), levels.copy()
    forecast_totals(customers, topups, levels, companies, T_HORIZONS)
    pd.testing.assert_frame_equal(customers, before[0])
    pd.testing.assert_frame_equal(topups, before[1])
    pd.testing.assert_series_equal(levels, before[2])
    assert companies == T_COMPANIES


# --- errors ----------------------------------------------------------------------------------

@pytest.mark.parametrize("horizons", [0, -1])
def test_horizons_below_one_exact_message(horizons):
    assert totals_error(horizons=horizons) == f"horizons is {horizons}; horizons start at 1"


def test_all_in_companies_exact_message():
    assert totals_error(companies=["TR02", "ALL"]) == (
        "'ALL' is the code of the consolidated rows; it cannot be a company")


def test_two_origins_across_customers_and_topups_exact_message():
    message = totals_error(customers=hand_customers(origin=41))
    assert message == ("the customers and top-ups must be for one origin week; they hold "
                       "[40, 41]")


def test_two_origins_within_the_customers():
    customers = hand_customers()
    customers.loc[customers.index[0], "origin_week"] = 39
    assert "they hold [39, 40]" in totals_error(customers=customers)


def test_two_origins_within_the_topups():
    topups = hand_topups()
    topups.loc[0, "origin_week"] = 41
    assert "they hold [40, 41]" in totals_error(topups=topups)


def test_topup_pair_twice_exact_message():
    topups = pd.concat([hand_topups(), hand_topups().iloc[[0]]], ignore_index=True)
    assert totals_error(topups=topups) == "the top-ups hold a company and h more than once"


def test_topups_missing_a_company():
    topups = hand_topups({"TR02": TOPUP_VALUES["TR02"]})
    message = totals_error(topups=topups)
    assert message.startswith("the top-ups have no rows for (company, h) [('TR04', 1)")


def test_topups_missing_an_h():
    topups = hand_topups({c: v[:2] for c, v in TOPUP_VALUES.items()})
    assert totals_error(topups=topups) == (
        "the top-ups have no rows for (company, h) [('TR02', 3), ('TR04', 3)]")


def test_topups_extra_company():
    topups = pd.concat([hand_topups(), hand_topups({"TR99": TOPUP_VALUES["TR02"]})],
                       ignore_index=True)
    message = totals_error(topups=topups)
    assert "the top-ups have rows for (company, h) [('TR99', 1)" in message
    assert "not in the company list or the horizons" in message


@pytest.mark.parametrize("bad_h", [0, 4])
def test_topups_extra_h_outside_the_horizons(bad_h):
    extra = hand_topups({"TR02": [(1.0, -1.0, 1.0, False)]})
    extra["h"], extra["target_week"] = bad_h, T_ORIGIN + bad_h
    topups = pd.concat([hand_topups(), extra], ignore_index=True)
    assert f"[('TR02', {bad_h})]" in totals_error(topups=topups)


def test_customers_missing_a_whole_company_raises_not_zero():
    customers = hand_customers({k: v for k, v in CUSTOMER_EXPECTED.items() if k[0] == "TR02"})
    message = totals_error(customers=customers)
    assert message == ("the customers have no rows for (company, h) "
                       "[('TR04', 1), ('TR04', 2), ('TR04', 3)]")


def test_customers_missing_one_h():
    expected = {k: v for k, v in CUSTOMER_EXPECTED.items() if k != ("TR04", 2)}
    assert totals_error(customers=hand_customers(expected)) == (
        "the customers have no rows for (company, h) [('TR04', 2)]")


def test_customers_with_no_rows_at_all():
    customers = hand_customers().iloc[0:0]
    # No customer rows: the origins come from the top-ups alone, then every pair is missing.
    assert totals_error(customers=customers).startswith("the customers have no rows for")


def test_customers_extra_company():
    expected = {**CUSTOMER_EXPECTED, ("TR99", 1): [5.0]}
    assert "the customers have rows for (company, h) [('TR99', 1)]" in totals_error(
        customers=hand_customers(expected))


def test_customers_extra_h():
    expected = {**CUSTOMER_EXPECTED, ("TR02", 4): [5.0]}
    assert "[('TR02', 4)]" in totals_error(customers=hand_customers(expected))


def test_a_target_week_missing_from_levels_exact_message():
    assert totals_error(levels=T_LEVELS.drop(42)) == "no price level for target weeks [42]"


def test_nan_level_reads_as_missing():
    levels = T_LEVELS.copy()
    levels[43] = nan
    assert totals_error(levels=levels) == "no price level for target weeks [43]"


def test_infinite_level_is_reported():
    levels = T_LEVELS.copy()
    levels[41] = inf
    # total is checked before price_level, and holds the company rows and ALL.
    assert totals_error(levels=levels) == "total is NaN or infinite in 3 rows"


@pytest.mark.parametrize("column", ["new_customer_cash", "refunds", "unidentified_cash"])
@pytest.mark.parametrize("bad", [nan, inf])
def test_nan_or_infinite_top_up_is_reported(column, bad):
    topups = hand_topups()
    topups.loc[0, column] = bad
    message = totals_error(topups=topups)
    assert "is NaN or infinite in" in message


def test_nan_top_up_names_its_column_first():
    topups = hand_topups()
    topups.loc[0, "new_customer_cash"] = nan
    # The company row is NaN; the ALL row's sum skips the NaN, so it counts 1 row.
    assert totals_error(topups=topups) == "new_customer_cash is NaN or infinite in 1 rows"


def test_infinite_top_up_counts_the_company_and_all_rows():
    topups = hand_topups()
    topups.loc[0, "unidentified_cash"] = inf
    assert totals_error(topups=topups) == "unidentified_cash is NaN or infinite in 2 rows"


def test_infinite_customer_expected_is_reported():
    customers = hand_customers()
    customers.loc[customers.index[0], "expected"] = inf
    assert totals_error(customers=customers) == "customers is NaN or infinite in 2 rows"


def test_nan_customer_expected_is_reported():
    # groupby's sum would skip it (TR02 h1 would read 50, not 150): refused up front.
    customers = hand_customers()
    customers.loc[customers.index[0], "expected"] = nan
    assert totals_error(customers=customers) == "the customers have no expected in 1 rows"


def test_customer_row_with_a_missing_company_is_not_dropped_silently():
    # groupby would leave the row out: refused up front.
    customers = hand_customers()
    customers.loc[customers.index[0], "company_code"] = None
    assert totals_error(customers=customers) == "the customers have no company_code in 1 rows"


def test_customer_row_with_a_nan_company_is_refused_too():
    customers = hand_customers()
    customers.loc[customers.index[[0, 3]], "company_code"] = nan
    assert totals_error(customers=customers) == "the customers have no company_code in 2 rows"


def test_customer_row_with_a_missing_h_is_not_dropped_silently():
    customers = hand_customers()
    customers["h"] = customers["h"].astype("float64")
    customers.loc[customers.index[0], "h"] = nan
    assert totals_error(customers=customers) == "the customers have no h in 1 rows"


def test_customer_missing_values_count_every_row():
    customers = hand_customers()
    customers.loc[customers.index[:4], "expected"] = nan
    assert totals_error(customers=customers) == "the customers have no expected in 4 rows"


def test_customer_missing_values_checked_company_then_h_then_expected():
    customers = hand_customers()
    customers["h"] = customers["h"].astype("float64")
    customers.loc[customers.index[0], "expected"] = nan
    customers.loc[customers.index[1], "h"] = nan
    assert totals_error(customers=customers) == "the customers have no h in 1 rows"
    customers.loc[customers.index[2], "company_code"] = None
    assert totals_error(customers=customers) == "the customers have no company_code in 1 rows"


def test_customer_missing_h_is_reported_as_missing_not_as_a_wrong_target_week():
    # A NaN h also makes target_week != origin + h; the missing-value check comes first.
    customers = hand_customers()
    customers["h"] = customers["h"].astype("float64")
    customers.loc[customers.index[0], "h"] = nan
    assert "target week" not in totals_error(customers=customers)


def test_a_customer_with_an_infinite_expected_is_not_a_missing_value():
    customers = hand_customers()
    customers.loc[customers.index[0], "expected"] = inf
    assert totals_error(customers=customers) == "customers is NaN or infinite in 2 rows"


# --- target weeks: origin + h in both tables ----------------------------------------------

def test_topup_target_week_other_than_origin_plus_h_exact_message():
    # The probe: TR02 h1 at week 42 used to give week 42's level and two "ALL" rows for h 1.
    topups = hand_topups()
    topups.loc[0, "target_week"] = 42
    assert totals_error(topups=topups) == (
        "the top-ups have a target week other than origin 40 + h in 1 rows")


def test_topup_target_weeks_counted():
    topups = hand_topups()
    topups["target_week"] = topups["target_week"] + 1
    assert totals_error(topups=topups) == (
        "the top-ups have a target week other than origin 40 + h in 6 rows")


def test_customer_target_week_other_than_origin_plus_h_exact_message():
    customers = hand_customers()
    customers.loc[customers.index[[0, 1]], "target_week"] = 99
    assert totals_error(customers=customers) == (
        "the customers have a target week other than origin 40 + h in 2 rows")


def test_customer_target_weeks_checked_before_topup_target_weeks():
    customers, topups = hand_customers(), hand_topups()
    customers.loc[customers.index[0], "target_week"] = 99
    topups.loc[0, "target_week"] = 99
    assert totals_error(customers=customers, topups=topups).startswith(
        "the customers have a target week")


def test_missing_target_week_is_a_wrong_target_week():
    topups = hand_topups()
    topups["target_week"] = topups["target_week"].astype("float64")
    topups.loc[0, "target_week"] = nan
    assert totals_error(topups=topups) == (
        "the top-ups have a target week other than origin 40 + h in 1 rows")


def test_topup_with_a_nan_h_trips_the_target_week_check():
    # Pinned: a NaN h in the top-ups is caught by the target-week check, before the pairs.
    topups = hand_topups()
    topups["h"] = topups["h"].astype("float64")
    topups.loc[0, "h"] = nan
    assert totals_error(topups=topups) == (
        "the top-ups have a target week other than origin 40 + h in 1 rows")


def test_topup_with_a_missing_company_is_reported_by_the_pair_check():
    # Top-ups have no missing-value check: a None company leaves (TR02, 1) missing.
    topups = hand_topups()
    topups.loc[0, "company_code"] = None
    assert totals_error(topups=topups) == (
        "the top-ups have no rows for (company, h) [('TR02', 1)]")


def test_no_duplicate_consolidated_row_can_appear():
    # Every target week is origin + h, so the consolidated rows are exactly one per h.
    found = totals()
    consolidated = found[found["company_code"] == CONSOLIDATED]
    assert list(consolidated["h"]) == [1, 2, 3]
    assert not found.duplicated(["company_code", "h"]).any()
    assert (found["target_week"] == found["origin_week"] + found["h"]).all()


def test_float_h_with_matching_target_weeks_passes():
    customers = hand_customers()
    customers["h"] = customers["h"].astype("float64")
    assert row(totals(customers=customers), "ALL", 1)["total"] == 342.0


# --- order of the checks ------------------------------------------------------------------------

def test_order_horizons_before_all():
    assert totals_error(horizons=0, companies=["ALL"]).startswith("horizons is 0")


def test_order_all_before_origins():
    assert totals_error(companies=["ALL"], customers=hand_customers(origin=41)).startswith(
        "'ALL'")


def test_order_origins_before_topup_duplicates():
    topups = pd.concat([hand_topups(), hand_topups().iloc[[0]]], ignore_index=True)
    message = totals_error(topups=topups, customers=hand_customers(origin=41))
    assert message.startswith("the customers and top-ups must be for one origin week")


def test_order_topup_duplicates_before_topup_pairs():
    topups = hand_topups({"TR02": TOPUP_VALUES["TR02"]})
    topups = pd.concat([topups, topups.iloc[[0]]], ignore_index=True)
    assert totals_error(topups=topups) == "the top-ups hold a company and h more than once"


def test_order_topup_pairs_before_customer_pairs():
    customers = hand_customers({k: v for k, v in CUSTOMER_EXPECTED.items() if k[0] == "TR02"})
    topups = hand_topups({"TR02": TOPUP_VALUES["TR02"]})
    assert totals_error(customers=customers, topups=topups).startswith("the top-ups")


def test_order_customer_pairs_before_levels():
    customers = hand_customers({k: v for k, v in CUSTOMER_EXPECTED.items() if k[0] == "TR02"})
    assert totals_error(customers=customers, levels=T_LEVELS.drop(42)).startswith(
        "the customers")


def test_order_levels_before_finite():
    topups = hand_topups()
    topups.loc[0, "refunds"] = inf
    assert totals_error(topups=topups, levels=T_LEVELS.drop(43)) == (
        "no price level for target weeks [43]")


# The full order: horizons -> ALL -> origins -> customer missing values -> target weeks
# (customers, then top-ups) -> top-up duplicates -> top-up pairs -> customer pairs ->
# levels -> finite.

def nan_expected_customers(origin=T_ORIGIN):
    customers = hand_customers(origin=origin)
    customers.loc[customers.index[0], "expected"] = nan
    return customers


def test_order_all_before_customer_missing_values():
    assert totals_error(customers=nan_expected_customers(),
                        companies=["TR02", "ALL"]).startswith("'ALL'")


def test_order_origins_before_customer_missing_values():
    assert totals_error(customers=nan_expected_customers(origin=41)).startswith(
        "the customers and top-ups must be for one origin week")


def test_order_customer_missing_values_before_target_weeks():
    customers = nan_expected_customers()
    customers.loc[customers.index[1], "target_week"] = 99
    topups = hand_topups()
    topups.loc[0, "target_week"] = 99
    assert totals_error(customers=customers, topups=topups) == (
        "the customers have no expected in 1 rows")


def test_order_topup_target_weeks_before_topup_duplicates():
    topups = pd.concat([hand_topups(), hand_topups().iloc[[0]]], ignore_index=True)
    topups.loc[1, "target_week"] = 99
    assert totals_error(topups=topups).startswith("the top-ups have a target week")


def test_order_target_weeks_before_pairs():
    topups = hand_topups({"TR02": TOPUP_VALUES["TR02"]})       # TR04 missing
    topups.loc[0, "target_week"] = 99
    assert totals_error(topups=topups).startswith("the top-ups have a target week")


def test_order_customer_missing_values_before_customer_pairs():
    customers = hand_customers({k: v for k, v in CUSTOMER_EXPECTED.items() if k[0] == "TR02"})
    customers.loc[customers.index[0], "expected"] = nan
    assert totals_error(customers=customers) == "the customers have no expected in 1 rows"


def test_order_customer_missing_values_before_levels_and_finite():
    topups = hand_topups()
    topups.loc[0, "refunds"] = inf
    assert totals_error(customers=nan_expected_customers(), topups=topups,
                        levels=T_LEVELS.drop(43)) == "the customers have no expected in 1 rows"


# --- end to end with the real projected_levels ---------------------------------------------

def test_with_real_projected_levels():
    # Week zero Monday 2025-06-02, origin 40: the levels a real forecast would use.
    from datetime import date
    levels = projected_levels(date(2025, 6, 2), T_ORIGIN, [T_ORIGIN + h for h in (1, 2, 3)])
    found = totals(levels=levels)
    for company, h in EXPECTED_TOTALS:
        got = row(found, company, h)
        assert got["price_level"] == levels[T_ORIGIN + h]
        # Company rows: exactly real_total x level. ALL: the sum of the company totals, so
        # equal to its real_total x level to rounding only.
        assert got["total"] == pytest.approx(got["real_total"] * levels[T_ORIGIN + h],
                                             rel=0 if company != CONSOLIDATED else 1e-14)
    all_rows = found[found["company_code"] == CONSOLIDATED]
    company_rows = found[found["company_code"] != CONSOLIDATED]
    np.testing.assert_allclose(all_rows["total"].to_numpy(),
                               company_rows.groupby("h")["total"].sum().to_numpy(), rtol=1e-14)
