"""Tests for `hurdle/stage2.py` (U11.1): `check_amounts` and `fit_stage2`.

`check_amounts(train_x, amount)` returns `amount` as float64 after checking, in this order:
the index equals train_x's (else "amount and train_x must have the same rows (the same
index)"), it is not empty, it has no NaN, every value is finite, every value is above 0. Each
refusal is a ValueError with its own message.

`fit_stage2(train_x, amount, exposure, settings)` runs `check_amounts` and `check_exposure`,
then fits a HistGradientBoostingRegressor with the `[stage2]` loss, max_iter, learning_rate,
max_leaf_nodes, min_samples_leaf and random_seed, early_stopping=False and
categorical_features="from_dtype" (a pandas categorical `company_code` is used as a category),
on s = amount / exposure with weights exposure ** (2 - LOSS_POWER[loss]) (U11.3b).

The U11.1-U11.3 tests pass an exposure of 1 everywhere (`ones(x)`): with it the fit is the
plain fit of the amount, bit for bit (pinned in the U11.3b section at the end).

Small synthetic X / amount and few trees, so every fit takes well under a second.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from collection_estimation.hurdle.settings import FEATURE_BLOCKS, STAGE2_LOSSES, Stage2Settings
from collection_estimation.hurdle.stage2 import check_amounts, fit_stage2

nan = float("nan")
inf = float("inf")

SAME_ROWS = "amount and train_x must have the same rows (the same index)"
NO_ROWS = "there are no paid rows to learn the amount from"


def stage2_settings(loss="poisson", max_iter=10, learning_rate=0.1, max_leaf_nodes=7,
                    min_samples_leaf=5, random_seed=3, scale="none", balance_floor="median",
                    own_mean_window=4, own_mean_prior_weeks=1.0):
    return Stage2Settings(loss=loss, max_iter=max_iter, learning_rate=learning_rate,
                          max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                          random_seed=random_seed, feature_blocks=FEATURE_BLOCKS,
                          scale=scale, balance_floor=balance_floor,
                          own_mean_window=own_mean_window,
                          own_mean_prior_weeks=own_mean_prior_weeks)


def ones(x):
    """An exposure of 1 for every row of `x`: m = s x 1, the plain fit of the amount."""
    return pd.Series(1.0, index=x.index, name="exposure")


def synthetic_xa(n=400, seed=0, company=True, index_offset=1000):
    """x: h, a signal column s, a noise column z (with NaN), and the company code; amount is
    a positive, skewed amount that grows with s (exp(1 + s) times a lognormal noise). The
    index starts at `index_offset`, not 0."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    x = pd.DataFrame({"h": rng.integers(1, 6, size=n).astype("float64"),
                      "s": s,
                      "z": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n))})
    if company:
        x["company_code"] = pd.Categorical(rng.choice(["TR02", "TR04"], size=n),
                                           categories=["TR02", "TR04", "TR05"])
    amount = np.exp(1.0 + s) * rng.lognormal(sigma=0.5, size=n) * 1000.0
    x.index = pd.RangeIndex(index_offset, index_offset + n)
    return x, pd.Series(amount, index=x.index, name="amount_deflated")


def small_x(index=(10, 11, 12)):
    return pd.DataFrame({"h": [1.0] * len(index)}, index=pd.Index(list(index)))


def refusal(x, amount):
    with pytest.raises(ValueError) as error:
        check_amounts(x, amount)
    return str(error.value)


# --- check_amounts: what it accepts and returns -----------------------------------------------

def test_check_amounts_returns_float64_values_unchanged():
    x = small_x()
    amount = pd.Series([1.5, 200.0, 1e9], index=x.index, name="amount_deflated")
    result = check_amounts(x, amount)
    assert result.dtype == "float64"
    pd.testing.assert_series_equal(result, amount)


def test_check_amounts_keeps_the_index_and_the_name():
    x = small_x((7, 3, 5))
    amount = pd.Series([1.0, 2.0, 3.0], index=x.index, name="amount_deflated")
    result = check_amounts(x, amount)
    assert result.index.equals(x.index)
    assert result.name == "amount_deflated"


def test_check_amounts_converts_ints_to_float():
    x = small_x()
    amount = pd.Series([1, 20, 300], index=x.index, dtype="int64")
    result = check_amounts(x, amount)
    assert result.dtype == "float64"
    assert result.tolist() == [1.0, 20.0, 300.0]


def test_check_amounts_converts_true_to_one():
    x = small_x()
    result = check_amounts(x, pd.Series([True, True, True], index=x.index))
    assert result.dtype == "float64"
    assert result.tolist() == [1.0, 1.0, 1.0]


def test_check_amounts_a_false_is_an_amount_of_zero_and_is_refused():
    x = small_x()
    message = refusal(x, pd.Series([True, False, True], index=x.index))
    assert message.startswith("amount is 0 or below in 1 rows")


def test_check_amounts_accepts_nullable_float_without_na():
    x = small_x()
    result = check_amounts(x, pd.Series([1.0, 2.0, 3.0], index=x.index, dtype="Float64"))
    assert result.dtype == "float64"
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_check_amounts_accepts_float32():
    x = small_x()
    result = check_amounts(x, pd.Series([1.0, 2.0, 3.0], index=x.index, dtype="float32"))
    assert result.dtype == "float64"


def test_check_amounts_accepts_tiny_and_huge_positive_amounts():
    x = small_x()
    amount = pd.Series([5e-324, 1e-300, 1e300], index=x.index)
    pd.testing.assert_series_equal(check_amounts(x, amount), amount)


def test_check_amounts_accepts_a_single_row():
    x = small_x((42,))
    result = check_amounts(x, pd.Series([7.0], index=x.index))
    assert result.tolist() == [7.0]


def test_check_amounts_accepts_a_duplicated_index_that_matches():
    # Duplicate index labels are train_x's business; equal indexes line up row by row.
    x = small_x((1, 1, 2))
    result = check_amounts(x, pd.Series([1.0, 2.0, 3.0], index=x.index))
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_check_amounts_does_not_look_at_train_x_columns():
    x = pd.DataFrame(index=pd.Index([1, 2]))
    assert check_amounts(x, pd.Series([1.0, 2.0], index=x.index)).tolist() == [1.0, 2.0]


def test_check_amounts_does_not_change_its_inputs():
    x = small_x()
    amount = pd.Series([1, 2, 3], index=x.index, dtype="int64", name="a")
    x_copy, amount_copy = x.copy(deep=True), amount.copy(deep=True)
    check_amounts(x, amount)
    pd.testing.assert_frame_equal(x, x_copy)
    pd.testing.assert_series_equal(amount, amount_copy)


def test_check_amounts_result_is_not_a_view_that_writes_back():
    x = small_x()
    amount = pd.Series([1.0, 2.0, 3.0], index=x.index)
    result = check_amounts(x, amount)
    result.iloc[0] = 99.0
    assert amount.iloc[0] == 1.0


# --- check_amounts: index -----------------------------------------------------------------------

def test_check_amounts_index_mismatch_exact_message():
    x = small_x()
    assert refusal(x, pd.Series([1.0, 2.0, 3.0], index=[20, 21, 22])) == SAME_ROWS


def test_check_amounts_index_mismatch_with_equal_length_is_refused():
    # Same length, other labels: lining up by position would pair rows wrongly.
    x = small_x((10, 11, 12))
    assert refusal(x, pd.Series([1.0, 2.0, 3.0])) == SAME_ROWS   # RangeIndex 0..2


def test_check_amounts_same_labels_in_another_order_is_a_mismatch():
    x = small_x((10, 11, 12))
    assert refusal(x, pd.Series([1.0, 2.0, 3.0], index=[12, 11, 10])) == SAME_ROWS


def test_check_amounts_a_shorter_amount_is_a_mismatch():
    x = small_x((10, 11, 12))
    assert refusal(x, pd.Series([1.0, 2.0], index=[10, 11])) == SAME_ROWS


def test_check_amounts_a_longer_amount_is_a_mismatch():
    x = small_x((10, 11))
    assert refusal(x, pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])) == SAME_ROWS


def test_check_amounts_index_is_checked_before_the_values():
    # A bad value and a bad index: the index is named.
    x = small_x((10, 11, 12))
    assert refusal(x, pd.Series([nan, -1.0, inf], index=[0, 1, 2])) == SAME_ROWS


def test_check_amounts_an_empty_amount_against_a_non_empty_x_is_an_index_mismatch():
    x = small_x((10, 11))
    assert refusal(x, pd.Series([], dtype="float64")) == SAME_ROWS


# --- check_amounts: empty -------------------------------------------------------------------------

def test_check_amounts_empty_exact_message():
    x = small_x(())
    assert refusal(x, pd.Series([], index=x.index, dtype="float64")) == NO_ROWS


def test_check_amounts_empty_object_series_says_no_rows():
    x = pd.DataFrame(index=pd.RangeIndex(0))
    assert refusal(x, pd.Series([], index=x.index, dtype=object)) == NO_ROWS


# --- check_amounts: missing, infinite, zero or negative -----------------------------------------

def test_check_amounts_one_nan_exact_message():
    x = small_x()
    assert refusal(x, pd.Series([1.0, nan, 3.0], index=x.index)) == (
        "amount has 1 missing values: stage 2 learns from paid rows, which all have an amount")


def test_check_amounts_counts_every_nan():
    x = small_x((1, 2, 3, 4))
    message = refusal(x, pd.Series([nan, 2.0, nan, nan], index=x.index))
    assert message.startswith("amount has 3 missing values")


def test_check_amounts_all_nan_is_refused_as_missing():
    x = small_x()
    message = refusal(x, pd.Series([nan, nan, nan], index=x.index))
    assert message.startswith("amount has 3 missing values")


def test_check_amounts_nullable_na_is_refused_as_missing():
    x = small_x()
    message = refusal(x, pd.Series([1.0, None, 3.0], index=x.index, dtype="Float64"))
    assert message.startswith("amount has 1 missing values")


def test_check_amounts_object_none_is_refused_as_missing():
    x = small_x()
    message = refusal(x, pd.Series([1.0, None, 3.0], index=x.index, dtype=object))
    assert message.startswith("amount has 1 missing values")


@pytest.mark.parametrize("bad", [inf, -inf], ids=["inf", "minus-inf"])
def test_check_amounts_infinite_exact_message(bad):
    x = small_x()
    assert refusal(x, pd.Series([1.0, bad, 3.0], index=x.index)) == (
        "amount holds an infinite value")


def test_check_amounts_nan_is_reported_before_infinite():
    x = small_x()
    message = refusal(x, pd.Series([nan, inf, 3.0], index=x.index))
    assert message.startswith("amount has 1 missing values")


def test_check_amounts_infinite_is_reported_before_negative():
    x = small_x()
    assert refusal(x, pd.Series([-1.0, inf, 3.0], index=x.index)) == (
        "amount holds an infinite value")


def test_check_amounts_one_zero_exact_message():
    x = small_x()
    assert refusal(x, pd.Series([1.0, 0.0, 3.0], index=x.index)) == (
        "amount is 0 or below in 1 rows: stage 2 learns from paid rows only (amount above 0)")


@pytest.mark.parametrize("values, count", [
    ([0.0, 0.0, 3.0], 2), ([-1.0, 2.0, 3.0], 1), ([-0.0, -5.0, -1e-300], 3),
    ([-5e-324, 1.0, 1.0], 1),
], ids=["two-zeros", "negative", "all-not-above-zero", "tiniest-negative"])
def test_check_amounts_counts_rows_at_or_below_zero(values, count):
    x = small_x()
    message = refusal(x, pd.Series(values, index=x.index))
    assert message.startswith(f"amount is 0 or below in {count} rows")


def test_check_amounts_negative_int_is_refused():
    x = small_x()
    message = refusal(x, pd.Series([1, -2, 3], index=x.index, dtype="int64"))
    assert message.startswith("amount is 0 or below in 1 rows")


def test_check_amounts_refusals_are_value_errors_not_other_types():
    x = small_x()
    for values in ([nan, 1.0, 1.0], [inf, 1.0, 1.0], [0.0, 1.0, 1.0]):
        with pytest.raises(ValueError):
            check_amounts(x, pd.Series(values, index=x.index))


# --- fit_stage2: the estimator's parameters -----------------------------------------------------

def test_fit_stage2_returns_a_fitted_regressor():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    assert isinstance(model, HistGradientBoostingRegressor)
    assert list(model.feature_names_in_) == ["h", "s", "z", "company_code"]
    assert model.n_iter_ == 10


@pytest.mark.parametrize("loss", STAGE2_LOSSES)
def test_fit_stage2_fits_with_each_loss(loss):
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings(loss=loss))
    assert model.get_params()["loss"] == loss
    prediction = model.predict(x)
    assert prediction.shape == (len(x),)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("loss", ["poisson", "gamma"])
def test_fit_stage2_log_link_losses_predict_above_zero(loss):
    x, amount = synthetic_xa()
    assert (fit_stage2(x, amount, ones(x), stage2_settings(loss=loss)).predict(x) > 0).all()


def test_fit_stage2_parameters_come_from_the_settings():
    x, amount = synthetic_xa()
    settings = stage2_settings(loss="gamma", max_iter=12, learning_rate=0.2, max_leaf_nodes=5,
                               min_samples_leaf=7, random_seed=11)
    params = fit_stage2(x, amount, ones(x), settings).get_params()
    assert params["loss"] == "gamma"
    assert params["max_iter"] == 12
    assert params["learning_rate"] == 0.2
    assert params["max_leaf_nodes"] == 5
    assert params["min_samples_leaf"] == 7
    assert params["random_state"] == 11


def test_fit_stage2_fixed_parameters():
    x, amount = synthetic_xa()
    params = fit_stage2(x, amount, ones(x), stage2_settings()).get_params()
    assert params["early_stopping"] is False
    assert params["categorical_features"] == "from_dtype"


def test_fit_stage2_other_parameters_are_sklearns_defaults():
    # Nothing beyond the settings and the two fixed choices is set.
    x, amount = synthetic_xa()
    params = fit_stage2(x, amount, ones(x), stage2_settings()).get_params()
    defaults = HistGradientBoostingRegressor().get_params()
    set_here = {"loss", "max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                "random_state", "early_stopping", "categorical_features"}
    for name, value in defaults.items():
        if name not in set_here:
            assert params[name] == value, name


def test_fit_stage2_does_not_read_the_feature_blocks():
    # feature_blocks picks the input columns before the fit; the fit itself ignores it.
    x, amount = synthetic_xa()
    one = fit_stage2(x, amount, ones(x), stage2_settings())
    other = dataclasses.replace(stage2_settings(), feature_blocks=("history",))
    two = fit_stage2(x, amount, ones(x), other)
    np.testing.assert_array_equal(one.predict(x), two.predict(x))


def test_fit_stage2_no_early_stopping_with_more_than_10000_rows():
    # sklearn's early_stopping="auto" switches on above 10,000 rows; off, every tree is built.
    x, amount = synthetic_xa(n=12_000, company=False)
    model = fit_stage2(x, amount, ones(x), stage2_settings(max_iter=15, learning_rate=0.01))
    assert model.n_iter_ == 15
    assert model.do_early_stopping_ is False


# --- fit_stage2: the company as a category ------------------------------------------------------

def test_fit_stage2_company_code_is_treated_as_categorical():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    assert model.is_categorical_.tolist() == [False, False, False, True]


def test_fit_stage2_without_company_nothing_is_categorical():
    x, amount = synthetic_xa(company=False)
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    assert model.is_categorical_ is None or not model.is_categorical_.any()


def test_fit_stage2_learns_an_amount_carried_only_by_the_company():
    # TR02 pays about 10,000, TR05 about 100; the other columns are noise.
    rng = np.random.default_rng(7)
    n = 600
    codes = rng.choice(["TR02", "TR05"], size=n)
    x = pd.DataFrame({"h": rng.integers(1, 6, size=n).astype("float64"),
                      "z": rng.normal(size=n),
                      "company_code": pd.Categorical(codes,
                                                     categories=["TR02", "TR04", "TR05"])})
    amount = pd.Series(np.where(codes == "TR02", 10_000.0, 100.0)
                       * rng.lognormal(sigma=0.2, size=n))
    model = fit_stage2(x, amount, ones(x), stage2_settings(max_iter=50, learning_rate=0.2))
    m = model.predict(x)
    assert np.median(m[codes == "TR02"]) > 5_000
    assert np.median(m[codes == "TR05"]) < 300


def test_fit_stage2_a_company_is_a_category_not_a_number():
    # The amount is high for the 1st and 3rd category and low for the 2nd: no single split on
    # the codes 0 < 1 < 2 separates them, a categorical split does at once. One tree, two
    # leaves: only a categorical split can get both groups right.
    codes = np.array(["TR02", "TR04", "TR05"] * 100)
    x = pd.DataFrame({"company_code": pd.Categorical(codes,
                                                     categories=["TR02", "TR04", "TR05"])})
    amount = pd.Series(np.where(codes == "TR04", 10.0, 1000.0))
    model = fit_stage2(x, amount, ones(x), stage2_settings(loss="squared_error", max_iter=1,
                                                  learning_rate=1.0, max_leaf_nodes=2,
                                                  min_samples_leaf=1))
    m = pd.Series(model.predict(x))
    assert m[codes == "TR04"].max() < m[codes != "TR04"].min()


# --- fit_stage2: what it learns -----------------------------------------------------------------

def test_fit_stage2_poisson_keeps_the_training_total():
    x, amount = synthetic_xa(n=2000)
    model = fit_stage2(x, amount, ones(x), stage2_settings(max_iter=100, learning_rate=0.05))
    ratio = model.predict(x).sum() / amount.sum()
    assert ratio == pytest.approx(1.0, abs=1e-3)


def test_fit_stage2_poisson_keeps_the_total_per_company_roughly():
    # The company is a category: each company's total is kept too, not only the grand total.
    x, amount = synthetic_xa(n=2000)
    model = fit_stage2(x, amount, ones(x), stage2_settings(max_iter=100, learning_rate=0.05))
    m = pd.Series(model.predict(x), index=x.index)
    for code in ("TR02", "TR04"):
        rows = x["company_code"] == code
        assert m[rows].sum() / amount[rows].sum() == pytest.approx(1.0, abs=0.02), code


def test_fit_stage2_poisson_with_one_tree_already_keeps_the_total_roughly():
    # The starting point is the mean amount (log link): the total holds from the first tree.
    x, amount = synthetic_xa(n=1000)
    model = fit_stage2(x, amount, ones(x), stage2_settings(max_iter=1))
    assert model.predict(x).sum() / amount.sum() == pytest.approx(1.0, abs=0.02)


def test_fit_stage2_learns_the_signal():
    x, amount = synthetic_xa(n=2000)
    model = fit_stage2(x, amount, ones(x), stage2_settings(max_iter=100, learning_rate=0.1))
    m = model.predict(x)
    # The amount grows with s: the prediction ranks the rows the same way, mostly.
    assert np.corrcoef(np.log(m), x["s"])[0, 1] > 0.9


def test_fit_stage2_constant_amount_predicts_that_amount():
    x, _ = synthetic_xa(n=200)
    amount = pd.Series(250.0, index=x.index)
    for loss in STAGE2_LOSSES:
        m = fit_stage2(x, amount, ones(x), stage2_settings(loss=loss)).predict(x)
        np.testing.assert_allclose(m, 250.0, rtol=1e-6, err_msg=loss)


def test_fit_stage2_a_single_paid_row_fits():
    x = pd.DataFrame({"h": [1.0]}, index=[5])
    model = fit_stage2(x, pd.Series([123.0], index=[5]), ones(x), stage2_settings(min_samples_leaf=1))
    assert model.predict(x)[0] == pytest.approx(123.0)


def test_fit_stage2_same_seed_gives_identical_predictions():
    x, amount = synthetic_xa()
    one = fit_stage2(x, amount, ones(x), stage2_settings(random_seed=5)).predict(x)
    two = fit_stage2(x, amount, ones(x), stage2_settings(random_seed=5)).predict(x)
    np.testing.assert_array_equal(one, two)


# --- fit_stage2: the inputs it accepts ----------------------------------------------------------

def test_fit_stage2_accepts_nan_features():
    x, amount = synthetic_xa()
    x.loc[x.index[:50], "s"] = np.nan
    assert np.isfinite(fit_stage2(x, amount, ones(x), stage2_settings()).predict(x)).all()


def test_fit_stage2_int_amounts_fit_like_their_floats():
    x, amount = synthetic_xa()
    amount = amount.round().astype("int64")
    plain = fit_stage2(x, amount.astype("float64"), ones(x), stage2_settings()).predict(x)
    ints = fit_stage2(x, amount, ones(x), stage2_settings()).predict(x)
    np.testing.assert_array_equal(plain, ints)


def test_fit_stage2_does_not_change_its_inputs():
    x, amount = synthetic_xa()
    amount = amount.round().astype("int64")
    x_copy, amount_copy = x.copy(deep=True), amount.copy(deep=True)
    fit_stage2(x, amount, ones(x), stage2_settings())
    pd.testing.assert_frame_equal(x, x_copy)
    pd.testing.assert_series_equal(amount, amount_copy)


def test_fit_stage2_does_not_change_the_settings():
    x, amount = synthetic_xa()
    settings = stage2_settings()
    copy = Stage2Settings(**vars(settings))
    fit_stage2(x, amount, ones(x), settings)
    assert settings == copy


# --- fit_stage2: the guards, before any fitting ---------------------------------------------------

@pytest.fixture
def no_fit(monkeypatch):
    """Make any HistGradientBoostingRegressor.fit raise, to show the guards run first."""
    def refuse(self, *args, **kwargs):
        raise AssertionError("fit was called")
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", refuse)


@pytest.mark.parametrize("change, message", [
    (lambda a: a.reset_index(drop=True), SAME_ROWS),
    (lambda a: a.iloc[::-1], SAME_ROWS),
    (lambda a: a.iloc[:-1], SAME_ROWS),
    (lambda a: a.where(a.index != a.index[3], nan), "amount has 1 missing values"),
    (lambda a: a.where(a.index != a.index[3], inf), "amount holds an infinite value"),
    (lambda a: a.where(a.index != a.index[3], 0.0), "amount is 0 or below in 1 rows"),
    (lambda a: a.where(a.index != a.index[3], -10.0), "amount is 0 or below in 1 rows"),
], ids=["new-index", "reversed", "shorter", "nan", "inf", "zero", "negative"])
def test_fit_stage2_refuses_bad_amounts_before_fitting(no_fit, change, message):
    x, amount = synthetic_xa()
    with pytest.raises(ValueError) as error:
        fit_stage2(x, change(amount), ones(x), stage2_settings())
    assert str(error.value).startswith(message)


def test_fit_stage2_refuses_empty_amounts_before_fitting(no_fit):
    x, amount = synthetic_xa()
    with pytest.raises(ValueError) as error:
        fit_stage2(x.iloc[:0], amount.iloc[:0], ones(x.iloc[:0]), stage2_settings())
    assert str(error.value) == NO_ROWS


def test_fit_stage2_no_fit_fixture_bites(no_fit):
    x, amount = synthetic_xa()
    with pytest.raises(AssertionError, match="fit was called"):
        fit_stage2(x, amount, ones(x), stage2_settings())


def test_fit_stage2_unknown_loss_is_not_checked_again():
    # The settings file only lets STAGE2_LOSSES through; fit_stage2 does not check again.
    # Pinned behaviour since U11.3b: the loss's power is looked up in LOSS_POWER before sklearn
    # sees the loss, so an unknown loss is a bare KeyError, not sklearn's ValueError.
    x, amount = synthetic_xa()
    with pytest.raises(KeyError, match="tweedie"):
        fit_stage2(x, amount, ones(x), stage2_settings(loss="tweedie"))


# --- U11.2: predict_amount ----------------------------------------------------------------------
# m for each row: model.predict(x) as a Series named "m" on x's index, nothing clipped. Refuses
# (ValueError, predict_paid's message) unless x's columns are the fitted ones, in order.

from collection_estimation.hurdle.stage2 import predict_amount, total_ratio  # noqa: E402

FITTED = ["h", "s", "z", "company_code"]


def columns_message(given, fitted):
    return f"x has columns {given}, but the model was fitted on {fitted}"


def interaction_x_amount(repeats=25):
    """Two 0/1 columns a, b; amount 100 at (0, 0) and 1 elsewhere, every cell `repeats` times.
    Every amount is above 0, but the best additive fit (what stumps converge to) is
    25.75 - 24.75 - 24.75 = -23.75 at (1, 1): a squared-error model predicts below 0 there."""
    cells = [(0, 0, 100.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)] * repeats
    x = pd.DataFrame({"a": [c[0] for c in cells], "b": [c[1] for c in cells]},
                     dtype="float64")
    return x, pd.Series([c[2] for c in cells])


def test_predict_amount_is_the_models_prediction_named_m_on_xs_index():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    m = predict_amount(model, x, ones(x))
    assert isinstance(m, pd.Series)
    assert m.name == "m"
    assert m.index.equals(x.index)
    assert m.dtype == "float64"
    np.testing.assert_array_equal(m.to_numpy(), model.predict(x))


def test_predict_amount_on_new_rows_with_their_own_index():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    new_x, _ = synthetic_xa(n=30, seed=9)
    new_x.index = pd.Index([f"row-{i}" for i in range(30)])
    m = predict_amount(model, new_x, ones(new_x))
    assert m.index.equals(new_x.index)
    assert len(m) == 30


def test_predict_amount_keeps_a_shuffled_index_row_by_row():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    shuffled = x.sample(frac=1.0, random_state=0)
    m = predict_amount(model, x, ones(x))
    pd.testing.assert_series_equal(predict_amount(model, shuffled, ones(shuffled)), m.loc[shuffled.index])


def test_predict_amount_on_zero_rows_raises_sklearns_value_error():
    # Pinned behaviour (not in the spec): sklearn refuses an empty x, so predict_amount does
    # not return an empty m. Step 14 always has forecast rows at the demo origin.
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    with pytest.raises(ValueError, match="0 sample"):
        predict_amount(model, x.iloc[:0], ones(x.iloc[:0]))


def test_predict_amount_squared_error_below_zero_is_not_clipped():
    x, amount = interaction_x_amount()
    model = fit_stage2(x, amount, ones(x), stage2_settings(loss="squared_error", max_iter=200,
                                                  learning_rate=0.5, max_leaf_nodes=2,
                                                  min_samples_leaf=1))
    m = predict_amount(model, x, ones(x))
    corner = (x["a"] == 1) & (x["b"] == 1)
    assert (m[corner] < -20).all(), m[corner].iloc[0]
    np.testing.assert_allclose(m[corner], -23.75, atol=0.01)
    np.testing.assert_array_equal(m.to_numpy(), model.predict(x))


@pytest.mark.parametrize("loss", ["poisson", "gamma"])
def test_predict_amount_log_link_losses_stay_above_zero_on_the_same_data(loss):
    x, amount = interaction_x_amount()
    model = fit_stage2(x, amount, ones(x), stage2_settings(loss=loss, max_iter=200, learning_rate=0.5,
                                                  max_leaf_nodes=2, min_samples_leaf=1))
    assert (predict_amount(model, x, ones(x)) > 0).all()


class NegativeModel:
    """A stand-in model whose predictions include values at and below 0."""
    feature_names_in_ = np.array(["h", "s"], dtype=object)

    def predict(self, x):
        return np.array([-5.0, 0.0, 3.0])[:len(x)]


def test_predict_amount_passes_negative_and_zero_predictions_through():
    x = pd.DataFrame({"h": [1.0, 2.0, 3.0], "s": [0.0, 0.0, 0.0]}, index=[7, 8, 9])
    m = predict_amount(NegativeModel(), x, ones(x))
    pd.testing.assert_series_equal(m, pd.Series([-5.0, 0.0, 3.0], index=[7, 8, 9], name="m"))


def test_predict_amount_columns_in_another_order_exact_message():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    with pytest.raises(ValueError) as error:
        predict_amount(model, x[["s", "h", "z", "company_code"]], ones(x[["s", "h", "z", "company_code"]]))
    assert str(error.value) == columns_message(["s", "h", "z", "company_code"], FITTED)


def test_predict_amount_a_missing_column_exact_message():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    with pytest.raises(ValueError) as error:
        predict_amount(model, x.drop(columns="company_code"), ones(x.drop(columns="company_code")))
    assert str(error.value) == columns_message(["h", "s", "z"], FITTED)


def test_predict_amount_an_extra_column_exact_message():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    with pytest.raises(ValueError) as error:
        predict_amount(model, x.assign(extra=1.0), ones(x.assign(extra=1.0)))
    assert str(error.value) == columns_message(FITTED + ["extra"], FITTED)


def test_predict_amount_a_renamed_column_is_refused():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    with pytest.raises(ValueError, match="but the model was fitted on"):
        predict_amount(model, x.rename(columns={"z": "Z"}), ones(x.rename(columns={"z": "Z"})))


def test_predict_amount_message_matches_predict_paids():
    from collection_estimation.hurdle.stage1 import predict_paid
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    with pytest.raises(ValueError) as from_m:
        predict_amount(model, x.drop(columns="z"), ones(x.drop(columns="z")))
    # predict_paid's check runs before it touches classes_, so the regressor reaches it too.
    with pytest.raises(ValueError) as from_p:
        predict_paid(model, x.drop(columns="z"))
    assert str(from_m.value) == str(from_p.value)


def test_predict_amount_does_not_change_its_input():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    x_copy = x.copy(deep=True)
    predict_amount(model, x, ones(x))
    pd.testing.assert_frame_equal(x, x_copy)


def test_predict_amount_on_x_with_nan_features_gives_finite_m():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    x.loc[x.index[:20], ["s", "z"]] = np.nan
    assert np.isfinite(predict_amount(model, x, ones(x))).all()


# --- U11.2: total_ratio -----------------------------------------------------------------------
# check_amounts first (paid rows only), then sum(predict_amount) / sum(amount), as a float.

class FixedModel:
    """A stand-in model that predicts the given values, on columns ["h"]; counts its calls."""
    feature_names_in_ = np.array(["h"], dtype=object)

    def __init__(self, values):
        self.values = np.asarray(values, dtype="float64")
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        return self.values[:len(x)]


def test_total_ratio_by_hand():
    # m = 10, 20, 70 (sum 100); amount = 50, 30, 120 (sum 200): ratio 0.5.
    x = pd.DataFrame({"h": [1.0, 2.0, 3.0]}, index=[4, 5, 6])
    ratio = total_ratio(FixedModel([10.0, 20.0, 70.0]), x,
                        pd.Series([50.0, 30.0, 120.0], index=x.index), ones(x))
    assert ratio == 0.5
    assert type(ratio) is float


def test_total_ratio_above_one_by_hand():
    x = pd.DataFrame({"h": [1.0, 2.0]})
    assert total_ratio(FixedModel([30.0, 30.0]), x, pd.Series([10.0, 30.0]), ones(x)) == 1.5


def test_total_ratio_is_a_ratio_of_totals_not_a_mean_of_ratios():
    # Row ratios 2.0 and 0.5 (mean 1.25); totals 52 over 101.
    x = pd.DataFrame({"h": [1.0, 2.0]})
    ratio = total_ratio(FixedModel([2.0, 50.0]), x, pd.Series([1.0, 100.0]), ones(x))
    assert ratio == pytest.approx(52.0 / 101.0)


def test_total_ratio_counts_negative_predictions():
    # m = -10, 30 sums to 20; amount 10 + 10 = 20: ratio 1.0, the negative m is not clipped.
    x = pd.DataFrame({"h": [1.0, 2.0]})
    assert total_ratio(FixedModel([-10.0, 30.0]), x, pd.Series([10.0, 10.0]), ones(x)) == 1.0


def test_total_ratio_accepts_int_amounts():
    x = pd.DataFrame({"h": [1.0, 2.0]})
    assert total_ratio(FixedModel([5.0, 5.0]), x, pd.Series([4, 6], dtype="int64"), ones(x)) == 1.0


def test_total_ratio_poisson_on_its_training_rows_is_about_one():
    x, amount = synthetic_xa(n=2000)
    model = fit_stage2(x, amount, ones(x), stage2_settings(max_iter=100, learning_rate=0.05))
    ratio = total_ratio(model, x, amount, ones(x))
    assert ratio == pytest.approx(1.0, abs=1e-3)
    assert ratio == pytest.approx(float(model.predict(x).sum() / amount.sum()))


def test_total_ratio_squared_error_on_its_training_rows_is_about_one():
    # Squared error starts at the mean and every leaf is a mean of residuals: the total holds.
    x, amount = synthetic_xa(n=1000)
    model = fit_stage2(x, amount, ones(x), stage2_settings(loss="squared_error", max_iter=50))
    assert total_ratio(model, x, amount, ones(x)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("loss", STAGE2_LOSSES)
def test_total_ratio_is_finite_and_positive_for_each_loss(loss):
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings(loss=loss))
    ratio = total_ratio(model, x, amount, ones(x))
    assert np.isfinite(ratio) and ratio > 0


def test_total_ratio_on_other_rows_than_the_training_rows():
    # It is a check of any paid rows, not only the training ones.
    x, amount = synthetic_xa(n=1000)
    model = fit_stage2(x, amount, ones(x), stage2_settings(max_iter=50))
    other_x, other_amount = synthetic_xa(n=300, seed=4)
    ratio = total_ratio(model, other_x, other_amount, ones(other_x))
    assert ratio == pytest.approx(float(model.predict(other_x).sum() / other_amount.sum()))


@pytest.mark.parametrize("change, message", [
    (lambda a: a.reset_index(drop=True), SAME_ROWS),
    (lambda a: a.iloc[::-1], SAME_ROWS),
    (lambda a: a.where(a.index != a.index[0], nan), "amount has 1 missing values"),
    (lambda a: a.where(a.index != a.index[0], inf), "amount holds an infinite value"),
    (lambda a: a.where(a.index != a.index[0], 0.0), "amount is 0 or below in 1 rows"),
    (lambda a: a.where(a.index != a.index[0], -1.0), "amount is 0 or below in 1 rows"),
], ids=["new-index", "reversed", "nan", "inf", "zero", "negative"])
def test_total_ratio_refuses_bad_amounts_before_predicting(change, message):
    x = pd.DataFrame({"h": [1.0, 2.0, 3.0]}, index=[4, 5, 6])
    model = FixedModel([1.0, 1.0, 1.0])
    with pytest.raises(ValueError) as error:
        total_ratio(model, x, change(pd.Series([1.0, 2.0, 3.0], index=x.index)), ones(x))
    assert str(error.value).startswith(message)
    assert model.calls == 0


def test_total_ratio_refuses_empty_rows_before_predicting():
    x = pd.DataFrame({"h": pd.Series([], dtype="float64")})
    model = FixedModel([])
    with pytest.raises(ValueError) as error:
        total_ratio(model, x, pd.Series([], index=x.index, dtype="float64"), ones(x))
    assert str(error.value) == NO_ROWS
    assert model.calls == 0


def test_total_ratio_fixed_model_counts_calls():
    # The call counter used above does count: one prediction per good call.
    x = pd.DataFrame({"h": [1.0]})
    model = FixedModel([1.0])
    total_ratio(model, x, pd.Series([1.0]), ones(x))
    assert model.calls == 1


def test_total_ratio_refuses_wrong_columns_like_predict_amount():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    with pytest.raises(ValueError) as error:
        total_ratio(model, x[["s", "h", "z", "company_code"]], amount, ones(x[["s", "h", "z", "company_code"]]))
    assert str(error.value) == columns_message(["s", "h", "z", "company_code"], FITTED)


def test_total_ratio_amounts_are_checked_before_the_columns():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    bad = amount.where(amount.index != amount.index[0], 0.0)
    with pytest.raises(ValueError) as error:
        total_ratio(model, x.drop(columns="z"), bad, ones(x.drop(columns="z")))
    assert str(error.value).startswith("amount is 0 or below in 1 rows")


def test_total_ratio_does_not_change_its_inputs():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    amount = amount.round().astype("int64")
    x_copy, amount_copy = x.copy(deep=True), amount.copy(deep=True)
    total_ratio(model, x, amount, ones(x))
    pd.testing.assert_frame_equal(x, x_copy)
    pd.testing.assert_series_equal(amount, amount_copy)


# --- U11.3: amount_references ---------------------------------------------------------------
# "training mean": mean amount_deflated of the paid training rows, on every row. "own mean":
# (receipts + prior_weeks x training mean) / (paying weeks + prior_weeks), receipts =
# mean_receipts_w x observed_weeks_w (0 where observed is 0), exactly the training mean where
# paying weeks is 0.

from collection_estimation.hurdle.stage2 import (  # noqa: E402
    amount_references,
    amount_scores,
    amount_summary,
)

REFERENCE_COLUMNS = ["training mean", "own mean"]


def train_table(paid=(True, False, True), amounts=(100.0, 0.0, 300.0)):
    """Paid rows 100 and 300: training mean 200. The unpaid row's 0 is not counted."""
    return pd.DataFrame({"paid": pd.array(list(paid), dtype="boolean"),
                         "amount_deflated": list(amounts)}, index=[0, 1, 2])


def history_rows(window=4, mean=(50.0, nan, 0.0, 250.0), paying=(2, 0, 0, 4),
                 observed=(4, 0, 4, 4), index=(20, 21, 22, 23)):
    """Four customers, window 4:
      20: receipts 50 x 4 = 200 in 2 paying weeks   own mean (200 + 200) / 3 = 133.33
      21: no data (observed 0, mean NaN)             exactly the training mean 200
      22: data, never paid (receipts 0)              exactly 200
      23: receipts 250 x 4 = 1,000 in 4 paying weeks (1,000 + 200) / 5 = 240
    (own means for prior_weeks 1)."""
    return pd.DataFrame({f"mean_receipts_{window}": list(mean),
                         f"paying_weeks_{window}": np.array(paying, dtype="int64"),
                         f"observed_weeks_{window}": np.array(observed, dtype="int64"),
                         "other": ["a", "b", "c", "d"][:len(index)]},
                        index=pd.Index(list(index)))


def references_refusal(train, rows, window=4, prior_weeks=1.0):
    with pytest.raises(ValueError) as error:
        amount_references(train, rows, window, prior_weeks)
    return str(error.value)


def test_amount_references_by_hand():
    result = amount_references(train_table(), history_rows(), 4, 1.0)
    assert list(result.columns) == REFERENCE_COLUMNS
    assert result.index.equals(pd.Index([20, 21, 22, 23]))
    assert (result["training mean"] == 200.0).all()
    np.testing.assert_allclose(result["own mean"], [400 / 3, 200.0, 200.0, 240.0])


def test_amount_references_dtypes_are_float64():
    result = amount_references(train_table(), history_rows(), 4, 1.0)
    assert result["training mean"].dtype == "float64"
    assert result["own mean"].dtype == "float64"


def test_amount_references_shrinkage_grows_with_prior_weeks():
    # Customer 20: (200 + k x 200) / (2 + k); k = 3: 800 / 5 = 160; k = 0.5: 300 / 2.5 = 120.
    rows = history_rows()
    assert amount_references(train_table(), rows, 4, 3.0)["own mean"][20] == pytest.approx(160)
    assert amount_references(train_table(), rows, 4, 0.5)["own mean"][20] == pytest.approx(120)
    assert amount_references(train_table(), rows, 4, 3)["own mean"][23] == pytest.approx(
        (1000 + 600) / 7)


def test_amount_references_a_tiny_prior_gives_the_customers_own_mean_per_paying_week():
    # Customer 20 paid 200 in 2 paying weeks: 100 per paying week.
    own = amount_references(train_table(), history_rows(), 4, 1e-9)["own mean"]
    assert own[20] == pytest.approx(100.0, rel=1e-6)
    assert own[23] == pytest.approx(250.0, rel=1e-6)


def test_amount_references_a_huge_prior_gives_the_training_mean():
    own = amount_references(train_table(), history_rows(), 4, 1e12)["own mean"]
    np.testing.assert_allclose(own, 200.0, rtol=1e-6)


@pytest.mark.parametrize("prior_weeks", [1.0, 3.0, 0.1, 7.3, 1e-6, 2])
def test_amount_references_no_paying_week_is_exactly_the_training_mean(prior_weeks):
    # A training mean with an awkward last digit: (0.1 + 0.2) / 2.
    train = train_table(paid=(True, True, False), amounts=(0.1, 0.2, 0.0))
    training_mean = float(pd.Series([0.1, 0.2]).mean())
    own = amount_references(train, history_rows(), 4, prior_weeks)["own mean"]
    assert own[21] == training_mean
    assert own[22] == training_mean


def test_amount_references_uses_the_window_it_is_given():
    rows = history_rows(window=13).join(history_rows(window=4, mean=(10.0, nan, 0.0, 10.0))
                                        .drop(columns="other"))
    four = amount_references(train_table(), rows, 4, 1.0)["own mean"]
    thirteen = amount_references(train_table(), rows, 13, 1.0)["own mean"]
    assert four[20] == pytest.approx((40 + 200) / 3)
    assert thirteen[20] == pytest.approx(400 / 3)


def test_amount_references_ignores_unpaid_training_amounts():
    # The unpaid row's amount (here 1,000,000) does not move the training mean.
    train = train_table(amounts=(100.0, 1_000_000.0, 300.0))
    assert (amount_references(train, history_rows(), 4, 1.0)["training mean"] == 200.0).all()


def test_amount_references_accepts_plain_bool_and_int_paid():
    for paid in ([True, False, True], [1, 0, 1]):
        train = pd.DataFrame({"paid": paid, "amount_deflated": [100.0, 0.0, 300.0]})
        assert (amount_references(train, history_rows(), 4, 1.0)["training mean"]
                == 200.0).all()


def test_amount_references_training_mean_is_the_same_on_every_row_and_a_python_mean():
    train = train_table(paid=(True, True, True), amounts=(1.0, 2.0, 4.0))
    result = amount_references(train, history_rows(), 4, 1.0)
    assert result["training mean"].nunique() == 1
    assert result["training mean"].iloc[0] == pytest.approx(7.0 / 3.0)


def test_amount_references_keeps_rows_order_and_index():
    rows = history_rows(index=("z", "a", "m", "b"))
    result = amount_references(train_table(), rows, 4, 1.0)
    assert list(result.index) == ["z", "a", "m", "b"]
    assert result.loc["z", "own mean"] == pytest.approx(400 / 3)


def test_amount_references_on_zero_rows_is_empty_with_the_columns():
    result = amount_references(train_table(), history_rows().iloc[:0], 4, 1.0)
    assert len(result) == 0
    assert list(result.columns) == REFERENCE_COLUMNS


def test_amount_references_does_not_change_its_inputs():
    train, rows = train_table(), history_rows()
    copies = train.copy(deep=True), rows.copy(deep=True)
    amount_references(train, rows, 4, 1.0)
    pd.testing.assert_frame_equal(train, copies[0])
    pd.testing.assert_frame_equal(rows, copies[1])


def test_amount_references_accepts_numpy_and_int_prior_weeks():
    for prior in (np.float64(1.0), np.int64(1), 1):
        own = amount_references(train_table(), history_rows(), 4, prior)["own mean"]
        assert own[20] == pytest.approx(400 / 3)


# amount_references: refusals

@pytest.mark.parametrize("bad", [0, 0.0, -1.0, float("inf"), nan, True, False, "1", None,
                                 [1.0]],
                         ids=["zero", "zero-float", "negative", "inf", "nan", "true", "false",
                              "string", "none", "list"])
def test_amount_references_refuses_bad_prior_weeks_with_the_exact_message(bad):
    message = references_refusal(train_table(), history_rows(), prior_weeks=bad)
    assert message == f"prior_weeks is {bad!r}; it must be a number above 0"


@pytest.mark.parametrize("drop, shown", [
    (["mean_receipts_4"], "['mean_receipts_4']"),
    (["paying_weeks_4"], "['paying_weeks_4']"),
    (["observed_weeks_4"], "['observed_weeks_4']"),
    (["mean_receipts_4", "observed_weeks_4"], "['mean_receipts_4', 'observed_weeks_4']"),
], ids=["mean", "paying", "observed", "two"])
def test_amount_references_refuses_missing_history_columns_with_the_exact_message(drop, shown):
    message = references_refusal(train_table(), history_rows().drop(columns=drop))
    assert message == f"the rows have no history columns {shown} for window 4"


def test_amount_references_another_window_is_missing():
    message = references_refusal(train_table(), history_rows(), window=13)
    assert message == ("the rows have no history columns ['mean_receipts_13', "
                       "'paying_weeks_13', 'observed_weeks_13'] for window 13")


IMPOSSIBLE = ("rows have impossible history values for window 4 (paying weeks negative, NaN "
              "or above the observed weeks, or receipts NaN or below 0)")


@pytest.mark.parametrize("change, count", [
    (lambda r: r.assign(paying_weeks_4=[-1, 0, 0, 4]), 1),
    (lambda r: r.assign(paying_weeks_4=[2.0, nan, 0.0, 4.0]), 1),
    (lambda r: r.assign(paying_weeks_4=[5, 1, 0, 4]), 2),        # above observed (4 and 0)
    (lambda r: r.assign(mean_receipts_4=[nan, nan, 0.0, 250.0]), 1),   # NaN with data
    (lambda r: r.assign(mean_receipts_4=[-1.0, nan, 0.0, 250.0]), 1),  # negative receipts
    (lambda r: r.assign(observed_weeks_4=[4.0, nan, 4.0, 4.0]), 1),    # observed NaN
], ids=["paying-negative", "paying-nan", "paying-above-observed", "receipts-nan",
        "receipts-negative", "observed-nan"])
def test_amount_references_refuses_impossible_history_with_the_exact_message(change, count):
    message = references_refusal(train_table(), change(history_rows()))
    assert message == f"{count} {IMPOSSIBLE}"


def test_amount_references_nan_mean_without_data_is_not_impossible():
    # Customer 21 has observed 0 and mean NaN: that is what a window without data looks like.
    assert amount_references(train_table(), history_rows(), 4, 1.0)["own mean"][21] == 200.0


def test_amount_references_negative_mean_without_data_is_ignored():
    # With observed 0 the receipts are 0 whatever mean_receipts holds.
    rows = history_rows(mean=(50.0, -7.0, 0.0, 250.0))
    assert amount_references(train_table(), rows, 4, 1.0)["own mean"][21] == 200.0


@pytest.mark.parametrize("train", [
    lambda: train_table(paid=(False, False, False)),
    lambda: train_table().iloc[:0],
], ids=["none-paid", "empty"])
def test_amount_references_refuses_no_paid_training_row(train):
    assert references_refusal(train(), history_rows()) == (
        "the training rows have no paid row to take a mean amount from")


@pytest.mark.parametrize("amounts", [(0.0, 0.0, 300.0), (-5.0, 0.0, 300.0), (nan, 0.0, 300.0)],
                         ids=["zero", "negative", "nan"])
def test_amount_references_refuses_a_paid_amount_not_above_0(amounts):
    assert references_refusal(train_table(amounts=amounts), history_rows()) == (
        "a paid training row has an amount that is not above 0")


def test_amount_references_an_unpaid_zero_amount_is_fine():
    train = train_table(paid=(True, False, True), amounts=(100.0, 0.0, 300.0))
    assert amount_references(train, history_rows(), 4, 1.0)["training mean"].iloc[0] == 200.0


def test_amount_references_refusal_order():
    # Everything broken: prior_weeks first, then the missing columns, then the impossible
    # values, then no paid row, then a paid amount not above 0.
    broken_rows = history_rows().assign(paying_weeks_4=[-1, 0, 0, 4])
    no_paid = train_table(paid=(False, False, False))
    assert references_refusal(no_paid, broken_rows.drop(columns="mean_receipts_4"),
                              prior_weeks=0).startswith("prior_weeks is 0")
    assert references_refusal(no_paid, broken_rows.drop(columns="mean_receipts_4")
                              ).startswith("the rows have no history columns")
    assert references_refusal(no_paid, broken_rows).endswith(IMPOSSIBLE)
    assert references_refusal(no_paid, history_rows()).startswith(
        "the training rows have no paid row")
    assert references_refusal(train_table(amounts=(0.0, 0.0, 300.0)), history_rows()
                              ).startswith("a paid training row")


# --- U11.3: amount_scores and amount_summary --------------------------------------------------
# A hand-checkable case, four paid rows (index 10..13) in two horizons, h given out of order:
#   row  h  actual  stage 2  reference (training mean)
#   10   2   100     110       150
#   11   1   200     180       150
#   12   2    50      60       150
#   13   1   400     300       150
# Poisson deviance of one row: 2 (y ln(y / g) - y + g).

SCORE_COLUMNS = ["model", "h", "rows", "m <= 0", "mean deviance", "deviance skill",
                 "R2 hindsight", "WAPE", "MAPE", "total ratio"]


def deviance(y, g):
    y, g = np.asarray(y, dtype="float64"), np.asarray(g, dtype="float64")
    return float(np.mean(2 * (y * np.log(y / g) - y + g)))


def r2(y, g):
    y, g = np.asarray(y, dtype="float64"), np.asarray(g, dtype="float64")
    return float(1 - ((y - g) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def score_case():
    index = pd.Index([10, 11, 12, 13])
    actual = pd.Series([100.0, 200.0, 50.0, 400.0], index=index, name="amount_deflated")
    h = pd.Series(np.array([2, 1, 2, 1], dtype="int64"), index=index, name="h")
    stage2 = pd.Series([110.0, 180.0, 60.0, 300.0], index=index, name="m")
    reference = pd.Series(150.0, index=index, name="training mean")
    return actual, h, {"stage 2": stage2, "training mean": reference}, reference


def scores_of(case=None):
    actual, h, predictions, reference = case or score_case()
    return amount_scores(actual, h, predictions, reference)


def row(scores, model, h):
    found = scores[(scores["model"] == model) & (scores["h"] == h)]
    assert len(found) == 1, (model, h)
    return found.iloc[0]


def test_amount_scores_columns_and_row_order():
    scores = scores_of()
    assert list(scores.columns) == SCORE_COLUMNS
    assert list(scores["model"]) == ["stage 2"] * 3 + ["training mean"] * 3
    assert list(scores["h"]) == [1, 2, "all"] * 2
    assert list(scores.index) == list(range(6))


def test_amount_scores_h1_by_hand():
    s = row(scores_of(), "stage 2", 1)
    assert s["rows"] == 2 and s["m <= 0"] == 0
    assert s["mean deviance"] == pytest.approx(deviance([200, 400], [180, 300]))
    assert s["mean deviance"] == pytest.approx(16.14494, abs=1e-4)
    reference_deviance = deviance([200, 400], [150, 150])
    assert s["deviance skill"] == pytest.approx(1 - s["mean deviance"] / reference_deviance)
    # R2: mean 300, SS_res 20^2 + 100^2 = 10,400, SS_tot 2 x 100^2 = 20,000: 0.48.
    assert s["R2 hindsight"] == pytest.approx(0.48)
    assert s["WAPE"] == pytest.approx(120 / 600)            # (20 + 100) / 600
    assert s["MAPE"] == pytest.approx((0.1 + 0.25) / 2)     # 20/200, 100/400
    assert s["total ratio"] == pytest.approx(480 / 600)


def test_amount_scores_h2_by_hand():
    s = row(scores_of(), "stage 2", 2)
    assert s["rows"] == 2
    assert s["mean deviance"] == pytest.approx(deviance([100, 50], [110, 60]))
    assert s["R2 hindsight"] == pytest.approx(r2([100, 50], [110, 60]))   # 1 - 200/1250
    assert s["R2 hindsight"] == pytest.approx(0.84)
    assert s["WAPE"] == pytest.approx(20 / 150)
    assert s["MAPE"] == pytest.approx((0.1 + 0.2) / 2)
    assert s["total ratio"] == pytest.approx(170 / 150)


def test_amount_scores_all_row_pools_every_row_not_the_horizons_means():
    s = row(scores_of(), "stage 2", "all")
    y, g = [100, 200, 50, 400], [110, 180, 60, 300]
    assert s["rows"] == 4
    assert s["mean deviance"] == pytest.approx(deviance(y, g))
    assert s["deviance skill"] == pytest.approx(1 - deviance(y, g) / deviance(y, [150] * 4))
    assert s["R2 hindsight"] == pytest.approx(r2(y, g))
    assert s["WAPE"] == pytest.approx(140 / 750)
    assert s["MAPE"] == pytest.approx((0.1 + 0.1 + 0.2 + 0.25) / 4)
    assert s["total ratio"] == pytest.approx(650 / 750)


def test_amount_scores_the_reference_has_zero_skill_against_itself():
    scores = scores_of()
    assert (scores.loc[scores["model"] == "training mean", "deviance skill"] == 0.0).all()


def test_amount_scores_reference_by_hand():
    s = row(scores_of(), "training mean", "all")
    assert s["total ratio"] == pytest.approx(600 / 750)
    assert s["WAPE"] == pytest.approx((50 + 50 + 100 + 250) / 750)


def test_amount_scores_values_are_python_numbers_of_the_right_kind():
    s = row(scores_of(), "stage 2", "all")
    assert int(s["rows"]) == 4 and int(s["m <= 0"]) == 0
    scores = scores_of()
    for column in ("mean deviance", "deviance skill", "R2 hindsight", "WAPE", "MAPE",
                   "total ratio"):
        assert scores[column].dtype == "float64", column


def test_amount_scores_h_is_sorted_whatever_the_row_order():
    actual, h, predictions, reference = score_case()
    h = pd.Series(np.array([5, 3, 5, 1], dtype="int64"), index=h.index)
    scores = amount_scores(actual, h, predictions, reference)
    assert list(scores["h"]) == [1, 3, 5, "all"] * 2
    assert list(row(scores, "stage 2", 3)[["rows"]]) == [1]


def test_amount_scores_models_follow_the_dict_order():
    actual, h, predictions, reference = score_case()
    reordered = {"training mean": predictions["training mean"],
                 "stage 2": predictions["stage 2"]}
    scores = amount_scores(actual, h, reordered, reference)
    assert list(scores["model"].unique()) == ["training mean", "stage 2"]


def test_amount_scores_one_row_in_an_h_has_nan_r2_and_the_rest():
    actual, h, predictions, reference = score_case()
    h = pd.Series(np.array([2, 1, 1, 1], dtype="int64"), index=h.index)
    s = row(amount_scores(actual, h, predictions, reference), "stage 2", 2)
    assert s["rows"] == 1
    assert np.isnan(s["R2 hindsight"])
    assert s["MAPE"] == pytest.approx(0.1) and s["WAPE"] == pytest.approx(0.1)
    assert s["total ratio"] == pytest.approx(1.1)
    assert s["mean deviance"] == pytest.approx(deviance([100], [110]))


def test_amount_scores_a_single_h_gives_that_h_and_all_with_the_same_numbers():
    actual, h, predictions, reference = score_case()
    h = pd.Series(np.ones(4, dtype="int64"), index=h.index)
    scores = amount_scores(actual, h, predictions, reference)
    assert list(scores["h"]) == [1, "all", 1, "all"]
    one, everything = row(scores, "stage 2", 1), row(scores, "stage 2", "all")
    pd.testing.assert_series_equal(one.drop("h"), everything.drop("h"), check_names=False)


@pytest.mark.parametrize("bad", [0.0, -20.0], ids=["zero", "negative"])
def test_amount_scores_a_guess_at_or_below_0_gives_infinite_deviance_and_minus_inf_skill(bad):
    actual, h, predictions, reference = score_case()
    guess = predictions["stage 2"].copy()
    guess[13] = bad                               # h = 1
    scores = amount_scores(actual, h, {"stage 2": guess}, reference)
    for horizon in (1, "all"):
        s = row(scores, "stage 2", horizon)
        assert s["m <= 0"] == 1
        assert s["mean deviance"] == float("inf")
        assert s["deviance skill"] == float("-inf")
    # h = 2 has no bad guess: finite.
    s = row(scores, "stage 2", 2)
    assert s["m <= 0"] == 0 and np.isfinite(s["mean deviance"])


def test_amount_scores_a_guess_below_0_still_counts_in_the_other_scores():
    actual, h, predictions, reference = score_case()
    guess = predictions["stage 2"].copy()
    guess[13] = -100.0                            # actual 400: error 500
    s = row(amount_scores(actual, h, {"stage 2": guess}, reference), "stage 2", 1)
    assert s["WAPE"] == pytest.approx((20 + 500) / 600)
    assert s["total ratio"] == pytest.approx((180 - 100) / 600)
    assert s["R2 hindsight"] == pytest.approx(r2([200, 400], [180, -100]))


def test_amount_scores_counts_every_guess_at_or_below_0():
    actual, h, predictions, reference = score_case()
    guess = pd.Series([0.0, -1.0, -5.0, 300.0], index=actual.index)
    scores = amount_scores(actual, h, {"bad": guess}, reference)
    assert row(scores, "bad", 1)["m <= 0"] == 1
    assert row(scores, "bad", 2)["m <= 0"] == 2
    assert row(scores, "bad", "all")["m <= 0"] == 3


def test_amount_scores_a_perfect_reference_leaves_the_skill_undefined():
    actual, h, predictions, _ = score_case()
    scores = amount_scores(actual, h, {"stage 2": predictions["stage 2"], "exact": actual},
                           actual)
    assert scores["deviance skill"].isna().all()
    assert (row(scores, "exact", "all")[["mean deviance"]] == 0.0).all()


def test_amount_scores_a_perfect_guess():
    actual, h, _, reference = score_case()
    s = row(amount_scores(actual, h, {"exact": actual.copy()}, reference), "exact", "all")
    assert s["mean deviance"] == pytest.approx(0.0, abs=1e-12)
    assert s["deviance skill"] == pytest.approx(1.0)
    assert s["R2 hindsight"] == 1.0
    assert s["WAPE"] == 0.0 and s["MAPE"] == 0.0 and s["total ratio"] == 1.0


def test_amount_scores_mape_weighs_small_payments_like_large_ones_wape_does_not():
    index = pd.Index([0, 1])
    actual = pd.Series([1.0, 1000.0], index=index)
    h = pd.Series([1, 1], index=index)
    guess = pd.Series([2.0, 1000.0], index=index)    # 100% off on the small one only
    s = row(amount_scores(actual, h, {"g": guess}, pd.Series(10.0, index=index)), "g", "all")
    assert s["MAPE"] == pytest.approx(0.5)
    assert s["WAPE"] == pytest.approx(1 / 1001)


def test_amount_scores_does_not_change_its_inputs():
    actual, h, predictions, reference = score_case()
    copies = (actual.copy(), h.copy(), {k: v.copy() for k, v in predictions.items()},
              reference.copy())
    amount_scores(actual, h, predictions, reference)
    pd.testing.assert_series_equal(actual, copies[0])
    pd.testing.assert_series_equal(h, copies[1])
    assert list(predictions) == list(copies[2])
    for name in predictions:
        pd.testing.assert_series_equal(predictions[name], copies[2][name])
    pd.testing.assert_series_equal(reference, copies[3])


def test_amount_scores_accepts_int_actual_amounts():
    actual, h, predictions, reference = score_case()
    as_float = amount_scores(actual, h, predictions, reference)
    as_int = amount_scores(actual.astype("int64"), h, predictions, reference)
    pd.testing.assert_frame_equal(as_float, as_int)


def test_amount_summary_matches_one_row_of_amount_scores():
    actual, h, predictions, reference = score_case()
    summary = amount_summary(actual, predictions["stage 2"], reference)
    expected = row(scores_of(), "stage 2", "all")
    assert list(summary) == SCORE_COLUMNS[2:]
    for key, value in summary.items():
        assert value == pytest.approx(expected[key]), key


# amount_scores: refusals, in order

def scores_refusal(actual, h, predictions, reference):
    with pytest.raises(ValueError) as error:
        amount_scores(actual, h, predictions, reference)
    return str(error.value)


@pytest.mark.parametrize("target, name", [("h", "h"), ("reference", "reference"),
                                          ("stage 2", "stage 2"),
                                          ("training mean", "training mean")])
@pytest.mark.parametrize("change", [
    lambda s: s.reset_index(drop=True), lambda s: s.iloc[::-1], lambda s: s.iloc[:-1],
], ids=["new-index", "reversed", "shorter"])
def test_amount_scores_refuses_a_series_without_actuals_rows(target, name, change):
    actual, h, predictions, reference = score_case()
    if target == "h":
        h = change(h)
    elif target == "reference":
        reference = change(reference)
    else:
        predictions = dict(predictions, **{target: change(predictions[target])})
    assert scores_refusal(actual, h, predictions, reference) == (
        f"{name} must have the rows of actual (the same index)")


def test_amount_scores_index_checks_run_h_then_reference_then_predictions_in_order():
    actual, h, predictions, reference = score_case()
    bad = {name: s.reset_index(drop=True) for name, s in predictions.items()}
    assert scores_refusal(actual, h.reset_index(drop=True), bad,
                          reference.reset_index(drop=True)).startswith("h must")
    assert scores_refusal(actual, h, bad, reference.reset_index(drop=True)).startswith(
        "reference must")
    assert scores_refusal(actual, h, bad, reference).startswith("stage 2 must")


@pytest.mark.parametrize("values, message", [
    ([nan, 200.0, 50.0, 400.0], "amount has 1 missing values"),
    ([inf, 200.0, 50.0, 400.0], "amount holds an infinite value"),
    ([0.0, 200.0, 50.0, 400.0], "amount is 0 or below in 1 rows"),
    ([-1.0, -2.0, 50.0, 400.0], "amount is 0 or below in 2 rows"),
], ids=["nan", "inf", "zero", "negative"])
def test_amount_scores_refuses_actual_amounts_as_check_amounts_does(values, message):
    actual, h, predictions, reference = score_case()
    actual = pd.Series(values, index=actual.index)
    assert scores_refusal(actual, h, predictions, reference).startswith(message)


def test_amount_scores_refuses_no_rows():
    actual, h, predictions, reference = score_case()
    empty = {name: s.iloc[:0] for name, s in predictions.items()}
    assert scores_refusal(actual.iloc[:0], h.iloc[:0], empty, reference.iloc[:0]) == NO_ROWS


@pytest.mark.parametrize("bad", [nan, inf, -inf], ids=["nan", "inf", "minus-inf"])
def test_amount_scores_refuses_a_guess_that_is_not_finite(bad):
    actual, h, predictions, reference = score_case()
    guess = predictions["stage 2"].copy()
    guess[12] = bad
    assert scores_refusal(actual, h, dict(predictions, **{"stage 2": guess}), reference) == (
        "stage 2 holds a NaN or infinite amount")


@pytest.mark.parametrize("bad", [nan, inf], ids=["nan", "inf"])
def test_amount_scores_refuses_a_reference_that_is_not_finite(bad):
    actual, h, predictions, reference = score_case()
    reference = reference.copy()
    reference[12] = bad
    assert scores_refusal(actual, h, {"stage 2": predictions["stage 2"]}, reference) == (
        "reference holds a NaN or infinite amount")


@pytest.mark.parametrize("bad", [0.0, -3.0], ids=["zero", "negative"])
def test_amount_scores_refuses_a_reference_not_above_0(bad):
    actual, h, predictions, reference = score_case()
    reference = reference.copy()
    reference[12] = bad
    assert scores_refusal(actual, h, {"stage 2": predictions["stage 2"]}, reference) == (
        "reference must be above 0 in every row")


def test_amount_scores_value_checks_come_after_the_index_checks_and_in_order():
    actual, h, predictions, reference = score_case()
    bad_actual = pd.Series([0.0, 200.0, 50.0, 400.0], index=actual.index)
    nan_guess = predictions["stage 2"].where(predictions["stage 2"].index != 12, nan)
    nan_reference = reference.where(reference.index != 12, nan)
    zero_reference = reference.where(reference.index != 12, 0.0)
    # index before actual
    assert scores_refusal(bad_actual, h.reset_index(drop=True), predictions,
                          reference).startswith("h must")
    # actual before the guesses' values
    assert scores_refusal(bad_actual, h, {"stage 2": nan_guess}, nan_reference).startswith(
        "amount is 0 or below")
    # the reference's finiteness before the guesses'
    assert scores_refusal(actual, h, {"stage 2": nan_guess}, nan_reference).startswith(
        "reference holds")
    # every guess's finiteness before the reference's sign
    assert scores_refusal(actual, h, {"stage 2": nan_guess}, zero_reference).startswith(
        "stage 2 holds")
    assert scores_refusal(actual, h, {"stage 2": predictions["stage 2"]},
                          zero_reference).startswith("reference must be above 0")


# --- U11.3 fix: amount_scores refuses repeated labels in actual's index, first ---------------------

REPEATED = "actual's index has repeated labels; each row needs its own label"


def test_amount_scores_refuses_repeated_labels_in_actual():
    # Before the fix, [1, 1] with h 1, 2 reported 2 rows for h=1 and 4 for "all".
    index = pd.Index([1, 1])
    with pytest.raises(ValueError) as error:
        amount_scores(pd.Series([10.0, 20.0], index=index), pd.Series([1, 2], index=index),
                      {"g": pd.Series([10.0, 20.0], index=index)},
                      pd.Series([15.0, 15.0], index=index))
    assert str(error.value) == REPEATED


def test_amount_scores_repeated_labels_are_checked_before_everything_else():
    # h, reference and the guess all have other rows, and the amounts are bad too.
    index = pd.Index([3, 3, 4])
    other = pd.Index([7, 8, 9])
    with pytest.raises(ValueError) as error:
        amount_scores(pd.Series([0.0, nan, -1.0], index=index), pd.Series([1, 1, 1], index=other),
                      {"g": pd.Series([1.0, 1.0, 1.0], index=other)},
                      pd.Series([nan, 0.0, 1.0], index=other))
    assert str(error.value) == REPEATED


def test_amount_scores_unique_labels_in_any_order_are_fine():
    actual, h, predictions, reference = score_case()
    order = [13, 10, 12, 11]
    scores = amount_scores(actual[order], h[order],
                           {k: v[order] for k, v in predictions.items()}, reference[order])
    pd.testing.assert_frame_equal(scores, scores_of())


# --- U11.3b: check_exposure -------------------------------------------------------------------------

from collection_estimation.hurdle.stage2 import (  # noqa: E402
    LOSS_POWER,
    check_exposure,
    exposure,
    paid_training_amounts,
)

EXPOSURE_ROWS = "exposure and x must have the same rows (the same index)"


def exposure_refusal(x, values):
    with pytest.raises(ValueError) as error:
        check_exposure(x, values)
    return str(error.value)


def test_check_exposure_returns_float64_values_unchanged():
    x = small_x()
    e = pd.Series([0.5, 2.0, 1e9], index=x.index, name="exposure")
    result = check_exposure(x, e)
    assert result.dtype == "float64"
    pd.testing.assert_series_equal(result, e)


def test_check_exposure_converts_ints():
    x = small_x()
    result = check_exposure(x, pd.Series([1, 2, 3], index=x.index, dtype="int64"))
    assert result.dtype == "float64" and result.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("index", [[20, 21, 22], [12, 11, 10], [10, 11]],
                         ids=["other-labels", "reversed", "shorter"])
def test_check_exposure_index_mismatch_exact_message(index):
    x = small_x((10, 11, 12))
    assert exposure_refusal(x, pd.Series([1.0] * len(index), index=index)) == EXPOSURE_ROWS


def test_check_exposure_index_mismatch_with_equal_length_is_refused():
    x = small_x((10, 11, 12))
    assert exposure_refusal(x, pd.Series([1.0, 1.0, 1.0])) == EXPOSURE_ROWS


@pytest.mark.parametrize("values, count", [
    ([nan, 1.0, 1.0], 1), ([inf, 1.0, 1.0], 1), ([-inf, 1.0, 1.0], 1), ([0.0, 1.0, 1.0], 1),
    ([-1.0, -2.0, 1.0], 2), ([nan, inf, 0.0], 3), ([-0.0, 1.0, 1.0], 1),
], ids=["nan", "inf", "minus-inf", "zero", "two-negative", "all-three", "minus-zero"])
def test_check_exposure_refuses_unusable_values_with_their_count(values, count):
    x = small_x()
    assert exposure_refusal(x, pd.Series(values, index=x.index)) == (
        f"exposure is NaN, infinite, or 0 or below in {count} rows")


def test_check_exposure_index_is_checked_before_the_values():
    x = small_x((10, 11, 12))
    assert exposure_refusal(x, pd.Series([nan, 0.0, -1.0])) == EXPOSURE_ROWS


def test_check_exposure_accepts_tiny_positive_values():
    x = small_x()
    result = check_exposure(x, pd.Series([5e-324, 1e-300, 1.0], index=x.index))
    assert (result > 0).all()


def test_check_exposure_does_not_change_its_input():
    x = small_x()
    e = pd.Series([1, 2, 3], index=x.index, dtype="int64")
    copy = e.copy()
    check_exposure(x, e)
    pd.testing.assert_series_equal(e, copy)


# --- U11.3b: fit_stage2 with an exposure ------------------------------------------------------------

def test_loss_power_is_the_tweedie_power_of_each_loss():
    assert LOSS_POWER == {"poisson": 1, "gamma": 2, "squared_error": 0}
    assert set(LOSS_POWER) == set(STAGE2_LOSSES)


@pytest.mark.parametrize("loss", STAGE2_LOSSES)
def test_fit_stage2_with_exposure_one_is_bit_for_bit_the_plain_fit(loss):
    # The key property of U11.3b: with E = 1 the target is amount / 1 and every weight is
    # 1 ** k = 1, and the fit (hence m) equals a plain, unweighted fit of the amount exactly.
    # Kept under 200,000 rows on purpose: above that, sklearn bins a random subsample, where
    # passing sample_weight might change which rows are drawn.
    x, amount = synthetic_xa(n=3000)
    settings = stage2_settings(loss=loss, max_iter=50, max_leaf_nodes=15)
    ours = predict_amount(fit_stage2(x, amount, ones(x), settings), x, ones(x)).to_numpy()
    plain = HistGradientBoostingRegressor(
        loss=loss, max_iter=50, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=5,
        early_stopping=False, categorical_features="from_dtype", random_state=3,
    ).fit(x, amount).predict(x)
    assert np.array_equal(ours, plain)


@pytest.fixture
def captured_fit(monkeypatch):
    """Replace HistGradientBoostingRegressor.fit by one that records X, y and sample_weight."""
    calls = []

    def record(self, X, y, sample_weight=None):
        calls.append({"X": X, "y": y, "sample_weight": sample_weight})
        return self
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", record)
    return calls


@pytest.mark.parametrize("loss, weights", [
    ("poisson", [10.0, 4.0, 0.5]),            # E ** 1
    ("gamma", [1.0, 1.0, 1.0]),               # E ** 0
    ("squared_error", [100.0, 16.0, 0.25]),   # E ** 2
])
def test_fit_stage2_fits_the_share_with_the_weights_of_its_loss(captured_fit, loss, weights):
    x = small_x((10, 11, 12))
    amount = pd.Series([50.0, 2.0, 3.0], index=x.index)
    e = pd.Series([10.0, 4.0, 0.5], index=x.index)
    fit_stage2(x, amount, e, stage2_settings(loss=loss))
    (call,) = captured_fit
    assert call["X"] is x
    np.testing.assert_array_equal(np.asarray(call["y"]), [5.0, 0.5, 6.0])   # amount / E
    np.testing.assert_allclose(np.asarray(call["sample_weight"]), weights)
    assert list(call["y"].index) == [10, 11, 12]


def test_fit_stage2_with_exposure_one_passes_weights_of_one(captured_fit):
    x, amount = synthetic_xa(n=50)
    fit_stage2(x, amount, ones(x), stage2_settings())
    (call,) = captured_fit
    assert (np.asarray(call["sample_weight"]) == 1.0).all()
    np.testing.assert_array_equal(np.asarray(call["y"]), amount.to_numpy())


@pytest.mark.parametrize("change, message", [
    (lambda e: e.reset_index(drop=True), EXPOSURE_ROWS),
    (lambda e: e.iloc[::-1], EXPOSURE_ROWS),
    (lambda e: e.where(e.index != e.index[3], nan), "exposure is NaN, infinite, or 0 or below in 1"),
    (lambda e: e.where(e.index != e.index[3], 0.0), "exposure is NaN, infinite, or 0 or below in 1"),
    (lambda e: e.where(e.index != e.index[3], -5.0), "exposure is NaN, infinite, or 0 or below in 1"),
], ids=["new-index", "reversed", "nan", "zero", "negative"])
def test_fit_stage2_refuses_bad_exposure_before_fitting(no_fit, change, message):
    x, amount = synthetic_xa()
    with pytest.raises(ValueError) as error:
        fit_stage2(x, amount, change(ones(x)), stage2_settings())
    assert str(error.value).startswith(message)


def test_fit_stage2_checks_the_amount_before_the_exposure(no_fit):
    x, amount = synthetic_xa()
    bad_amount = amount.where(amount.index != amount.index[0], 0.0)
    with pytest.raises(ValueError) as error:
        fit_stage2(x, bad_amount, ones(x).reset_index(drop=True), stage2_settings())
    assert str(error.value).startswith("amount is 0 or below in 1 rows")


def test_fit_stage2_exposure_is_a_required_argument():
    x, amount = synthetic_xa()
    with pytest.raises(TypeError):
        fit_stage2(x, amount, stage2_settings())


def balance_case(n=3000, seed=2):
    """x as synthetic_xa; an exposure E (a lognormal 'open balance') and an amount that is
    roughly a share of it: amount = E x share, share from s."""
    x, share = synthetic_xa(n=n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    e = pd.Series(np.exp(rng.normal(size=n)) * 5_000.0, index=x.index, name="exposure")
    return x, share / 5_000.0 * e, e


def test_fit_stage2_poisson_with_a_balance_exposure_keeps_the_training_total():
    x, amount, e = balance_case()
    model = fit_stage2(x, amount, e, stage2_settings(max_iter=100, learning_rate=0.05,
                                                     max_leaf_nodes=15))
    assert total_ratio(model, x, amount, e) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("loss", STAGE2_LOSSES)
def test_fit_stage2_a_constant_exposure_only_sets_the_scale(loss):
    # E = 10 everywhere: s is a tenth of the amount and m = s x 10 is the amount again; the
    # fitted m equals the E = 1 fit up to rounding (the weights scale every row alike).
    x, amount = synthetic_xa(n=1000)
    settings = stage2_settings(loss=loss, max_iter=30)
    ten = pd.Series(10.0, index=x.index)
    with_ten = predict_amount(fit_stage2(x, amount, ten, settings), x, ten)
    with_one = predict_amount(fit_stage2(x, amount, ones(x), settings), x, ones(x))
    np.testing.assert_allclose(with_ten, with_one, rtol=1e-6)


def test_fit_stage2_with_exposure_does_not_change_its_inputs():
    x, amount, e = balance_case(n=200)
    copies = x.copy(deep=True), amount.copy(), e.copy()
    fit_stage2(x, amount, e, stage2_settings())
    pd.testing.assert_frame_equal(x, copies[0])
    pd.testing.assert_series_equal(amount, copies[1])
    pd.testing.assert_series_equal(e, copies[2])


# --- U11.3b: predict_amount and total_ratio with an exposure ------------------------------------------

class ShareModel:
    """A stand-in model that predicts the shares s it is given, on columns ["h"]."""
    feature_names_in_ = np.array(["h"], dtype=object)

    def __init__(self, shares):
        self.shares = np.asarray(shares, dtype="float64")

    def predict(self, x):
        return self.shares[:len(x)]


def test_predict_amount_is_s_times_e_by_hand():
    x = pd.DataFrame({"h": [1.0, 2.0, 3.0]}, index=[7, 8, 9])
    e = pd.Series([1000.0, 50.0, 2.0], index=x.index)
    m = predict_amount(ShareModel([0.1, 2.0, -3.0]), x, e)
    pd.testing.assert_series_equal(m, pd.Series([100.0, 100.0, -6.0], index=[7, 8, 9], name="m"))


def test_predict_amount_checks_the_columns_before_the_exposure():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    with pytest.raises(ValueError) as error:
        predict_amount(model, x.drop(columns="z"), ones(x).reset_index(drop=True))
    assert str(error.value).startswith("x has columns")


@pytest.mark.parametrize("change, message", [
    (lambda e: e.reset_index(drop=True), EXPOSURE_ROWS),
    (lambda e: e.where(e.index != e.index[0], 0.0), "exposure is NaN, infinite, or 0 or below in 1"),
    (lambda e: e.where(e.index != e.index[0], inf), "exposure is NaN, infinite, or 0 or below in 1"),
], ids=["new-index", "zero", "inf"])
def test_predict_amount_refuses_bad_exposure(change, message):
    x = pd.DataFrame({"h": [1.0, 2.0, 3.0]}, index=[7, 8, 9])
    with pytest.raises(ValueError) as error:
        predict_amount(ShareModel([1.0, 1.0, 1.0]), x, change(pd.Series(2.0, index=x.index)))
    assert str(error.value).startswith(message)


def test_predict_amount_exposure_goes_row_by_row_on_a_shuffled_index():
    x = pd.DataFrame({"h": [1.0, 2.0, 3.0]}, index=["c", "a", "b"])
    e = pd.Series([1.0, 10.0, 100.0], index=["c", "a", "b"])
    m = predict_amount(ShareModel([1.0, 1.0, 1.0]), x, e)
    assert m.to_dict() == {"c": 1.0, "a": 10.0, "b": 100.0}


def test_total_ratio_with_exposure_by_hand():
    # s = 0.1, 0.2 and E = 1,000, 500: m = 100, 100, sum 200; amount 150 + 250 = 400: 0.5.
    x = pd.DataFrame({"h": [1.0, 2.0]}, index=[0, 1])
    ratio = total_ratio(ShareModel([0.1, 0.2]), x, pd.Series([150.0, 250.0]),
                        pd.Series([1000.0, 500.0]))
    assert ratio == pytest.approx(0.5)


def test_total_ratio_checks_amounts_then_columns_then_exposure():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, ones(x), stage2_settings())
    bad_amount = amount.where(amount.index != amount.index[0], 0.0)
    bad_e = ones(x).reset_index(drop=True)
    with pytest.raises(ValueError, match="^amount is 0 or below"):
        total_ratio(model, x.drop(columns="z"), bad_amount, bad_e)
    with pytest.raises(ValueError, match="^x has columns"):
        total_ratio(model, x.drop(columns="z"), amount, bad_e)
    with pytest.raises(ValueError, match="^exposure and x must have the same rows"):
        total_ratio(model, x, amount, bad_e)


# --- U11.3b: paid_training_amounts --------------------------------------------------------------------

def test_paid_training_amounts_are_the_paid_rows_amounts():
    result = paid_training_amounts(train_table())
    pd.testing.assert_series_equal(result, pd.Series([100.0, 300.0], index=[0, 2],
                                                     name="amount_deflated"))


def test_paid_training_amounts_accepts_plain_bool_and_int_paid():
    for paid in ([True, False, True], [1, 0, 1]):
        train = pd.DataFrame({"paid": paid, "amount_deflated": [100.0, 0.0, 300.0]})
        assert paid_training_amounts(train).tolist() == [100.0, 300.0]


@pytest.mark.parametrize("train, message", [
    (lambda: train_table(paid=(False, False, False)),
     "the training rows have no paid row to take a mean amount from"),
    (lambda: train_table().iloc[:0],
     "the training rows have no paid row to take a mean amount from"),
    (lambda: train_table(amounts=(0.0, 0.0, 300.0)),
     "a paid training row has an amount that is not above 0"),
    (lambda: train_table(amounts=(nan, 0.0, 300.0)),
     "a paid training row has an amount that is not above 0"),
], ids=["none-paid", "empty", "zero", "nan"])
def test_paid_training_amounts_refusals(train, message):
    with pytest.raises(ValueError) as error:
        paid_training_amounts(train())
    assert str(error.value) == message


def test_paid_training_amounts_does_not_change_its_input():
    train = train_table()
    copy = train.copy(deep=True)
    paid_training_amounts(train)
    pd.testing.assert_frame_equal(train, copy)


def test_amount_references_uses_paid_training_amounts_messages():
    # Factored out: amount_references refuses with the same two messages as before.
    with pytest.raises(ValueError, match="^the training rows have no paid row"):
        amount_references(train_table(paid=(False, False, False)), history_rows(), 4, 1.0)
    with pytest.raises(ValueError, match="^a paid training row has an amount"):
        amount_references(train_table(amounts=(-1.0, 0.0, 300.0)), history_rows(), 4, 1.0)


# --- U11.3b: exposure ---------------------------------------------------------------------------------
# floor_train's paid amounts are 100, 200 and 900, so the median (200) and the mean (400)
# differ and a test can tell them apart. Own mean (window 4, prior 1, training mean 400):
#   20: (200 + 400) / 3 = 200;   21, 22: 400 (no paying week);   23: (1000 + 400) / 5 = 280.

def floor_train():
    return pd.DataFrame({"paid": pd.array([True, True, False, True], dtype="boolean"),
                         "amount_deflated": [100.0, 200.0, 0.0, 900.0]}, index=[0, 1, 2, 3])


def balance_rows(balances=(150.0, 1000.0, -50.0, 280.0)):
    return history_rows().assign(open_balance=list(balances))


def exposure_settings(scale="open balance", floor="median", window=4, prior=1.0):
    return stage2_settings(scale=scale, balance_floor=floor, own_mean_window=window,
                           own_mean_prior_weeks=prior)


def test_exposure_none_is_one_on_every_row_named_exposure():
    rows = balance_rows()
    e = exposure(floor_train(), rows, exposure_settings(scale="none"))
    pd.testing.assert_series_equal(e, pd.Series(1.0, index=rows.index, name="exposure"))


def test_exposure_none_needs_nothing_from_the_rows_or_the_training_rows():
    # No open_balance, no history, no paid training row: "none" checks nothing.
    rows = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    e = exposure(train_table(paid=(False, False, False)), rows, exposure_settings(scale="none"))
    assert e.tolist() == [1.0, 1.0] and list(e.index) == ["x", "y"]
    assert e.dtype == "float64"


@pytest.mark.parametrize("floor, value", [("median", 200.0), ("mean", 400.0)])
def test_exposure_open_balance_with_a_training_floor_by_hand(floor, value):
    # Balances 150, 1,000, -50, 280: below the floor -> the floor; at or above -> kept.
    e = exposure(floor_train(), balance_rows(), exposure_settings(floor=floor))
    expected = [max(b, value) for b in (150.0, 1000.0, -50.0, 280.0)]
    pd.testing.assert_series_equal(e, pd.Series(expected, index=[20, 21, 22, 23],
                                                name="exposure"))


def test_exposure_open_balance_with_the_own_mean_floor_by_hand():
    # Own means 200, 400, 400, 280: 150 -> 200; 1,000 kept; -50 -> 400; 280 = its floor, kept.
    e = exposure(floor_train(), balance_rows(), exposure_settings(floor="own mean"))
    np.testing.assert_allclose(e, [200.0, 1000.0, 400.0, 280.0])
    assert e.name == "exposure" and list(e.index) == [20, 21, 22, 23]


def test_exposure_own_mean_floor_follows_its_window_and_prior():
    rows = balance_rows(balances=(0.0, 0.0, 0.0, 0.0))
    e = exposure(floor_train(), rows, exposure_settings(floor="own mean", prior=3.0))
    # 20: (200 + 1200) / 5 = 280; 23: (1000 + 1200) / 7 = 314.29
    np.testing.assert_allclose(e, [280.0, 400.0, 400.0, 2200 / 7])
    with pytest.raises(ValueError, match="no history columns"):
        exposure(floor_train(), rows, exposure_settings(floor="own mean", window=13))


def test_exposure_a_balance_equal_to_the_floor_is_kept():
    e = exposure(floor_train(), balance_rows(balances=(200.0, 200.0, 200.0, 200.0)),
                 exposure_settings(floor="median"))
    assert (e == 200.0).all()


def test_exposure_zero_and_negative_balances_get_the_floor():
    e = exposure(floor_train(), balance_rows(balances=(0.0, -1e9, -0.0, 1e-9)),
                 exposure_settings(floor="mean"))
    assert (e == 400.0).all()


def test_exposure_is_always_above_zero_with_a_balance():
    e = exposure(floor_train(), balance_rows(), exposure_settings(floor="own mean"))
    assert (e > 0).all()


def test_exposure_int_balances_are_read_as_floats():
    rows = balance_rows().assign(open_balance=np.array([150, 1000, -50, 280], dtype="int64"))
    e = exposure(floor_train(), rows, exposure_settings())
    assert e.dtype == "float64" and e.tolist() == [200.0, 1000.0, 200.0, 280.0]


def test_exposure_without_an_open_balance_column_exact_message():
    with pytest.raises(ValueError) as error:
        exposure(floor_train(), history_rows(), exposure_settings())
    assert str(error.value) == "the rows have no open_balance column to scale by"


def test_exposure_with_missing_open_balances_exact_message():
    with pytest.raises(ValueError) as error:
        exposure(floor_train(), balance_rows(balances=(nan, 1.0, nan, 3.0)),
                 exposure_settings())
    assert str(error.value) == "open_balance is missing in 2 rows"


@pytest.mark.parametrize("floor", ["median", "mean", "own mean"])
def test_exposure_refuses_no_paid_training_row_for_every_floor(floor):
    with pytest.raises(ValueError, match="^the training rows have no paid row"):
        exposure(train_table(paid=(False, False, False)), balance_rows(),
                 exposure_settings(floor=floor))


def test_exposure_refusal_order_column_then_missing_then_floor():
    no_paid = train_table(paid=(False, False, False))
    with pytest.raises(ValueError, match="no open_balance column"):
        exposure(no_paid, history_rows(), exposure_settings())
    with pytest.raises(ValueError, match="open_balance is missing in 1 rows"):
        exposure(no_paid, balance_rows(balances=(nan, 1.0, 1.0, 1.0)), exposure_settings())
    with pytest.raises(ValueError, match="no paid row"):
        exposure(no_paid, balance_rows(), exposure_settings())


def test_exposure_own_mean_refuses_impossible_history_like_amount_references():
    rows = balance_rows().assign(paying_weeks_4=[-1, 0, 0, 4])
    with pytest.raises(ValueError, match="impossible history values"):
        exposure(floor_train(), rows, exposure_settings(floor="own mean"))


def test_exposure_on_zero_rows_is_empty():
    e = exposure(floor_train(), balance_rows().iloc[:0], exposure_settings())
    assert len(e) == 0 and e.name == "exposure"


def test_exposure_does_not_change_its_inputs():
    train, rows = floor_train(), balance_rows()
    copies = train.copy(deep=True), rows.copy(deep=True)
    exposure(train, rows, exposure_settings(floor="own mean"))
    pd.testing.assert_frame_equal(train, copies[0])
    pd.testing.assert_frame_equal(rows, copies[1])


def test_exposure_feeds_fit_and_predict_end_to_end():
    # The chain step 14 runs: E for the paid training rows and for new rows, then fit and m.
    x, amount = synthetic_xa(n=300)
    rng = np.random.default_rng(5)
    rows = pd.DataFrame({"open_balance": rng.normal(loc=500, scale=800, size=300),
                         "paid": True, "amount_deflated": amount.to_numpy()}, index=x.index)
    settings = exposure_settings(floor="median")
    e = exposure(rows, rows, settings)
    assert (e >= float(amount.median())).all()
    model = fit_stage2(x, amount, e, settings)
    m = predict_amount(model, x, e)
    assert np.isfinite(m).all() and (m > 0).all()
    assert total_ratio(model, x, amount, e) == pytest.approx(1.0, abs=0.02)
