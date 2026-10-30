"""Tests for `hurdle/forecast.py` (U13.2): `customer_forecasts`.

`customer_forecasts(table, origin, settings, block_columns, companies)` fits stage 1 on
`rows_known_at(table, origin)` and stage 2 on the paid rows among them (exposure learned from
the known rows), then predicts p and m for `rows_at_origin(table, origin)`. It returns the
columns CUSTOMER_COLUMNS with the table's index and expected = p x m, and raises ValueError
for no rows at the origin, a NaN / infinite p, m or expected, an m below 0, and as the fits do.

A small hand-made training table (8 customers in two companies, origins 0..30, h 1..5; target
weeks after 23 unlabelled) and
few trees, so every forecast takes well under a second. Nothing reads the corpus.
"""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import forecast as forecast_module
from collection_estimation.hurdle.forecast import CUSTOMER_COLUMNS, customer_forecasts
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
