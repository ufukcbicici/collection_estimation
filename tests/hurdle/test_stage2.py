"""Tests for `hurdle/stage2.py` (U11.1): `check_amounts` and `fit_stage2`.

`check_amounts(train_x, amount)` returns `amount` as float64 after checking, in this order:
the index equals train_x's (else "amount and train_x must have the same rows (the same
index)"), it is not empty, it has no NaN, every value is finite, every value is above 0. Each
refusal is a ValueError with its own message.

`fit_stage2(train_x, amount, settings)` runs `check_amounts`, then fits a
HistGradientBoostingRegressor with the `[stage2]` loss, max_iter, learning_rate,
max_leaf_nodes, min_samples_leaf and random_seed, early_stopping=False and
categorical_features="from_dtype" (a pandas categorical `company_code` is used as a category).

Small synthetic X / amount and few trees, so every fit takes well under a second.
"""
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
                    min_samples_leaf=5, random_seed=3):
    return Stage2Settings(loss=loss, max_iter=max_iter, learning_rate=learning_rate,
                          max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                          random_seed=random_seed, feature_blocks=FEATURE_BLOCKS)


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
    model = fit_stage2(x, amount, stage2_settings())
    assert isinstance(model, HistGradientBoostingRegressor)
    assert list(model.feature_names_in_) == ["h", "s", "z", "company_code"]
    assert model.n_iter_ == 10


@pytest.mark.parametrize("loss", STAGE2_LOSSES)
def test_fit_stage2_fits_with_each_loss(loss):
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings(loss=loss))
    assert model.get_params()["loss"] == loss
    prediction = model.predict(x)
    assert prediction.shape == (len(x),)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("loss", ["poisson", "gamma"])
def test_fit_stage2_log_link_losses_predict_above_zero(loss):
    x, amount = synthetic_xa()
    assert (fit_stage2(x, amount, stage2_settings(loss=loss)).predict(x) > 0).all()


def test_fit_stage2_parameters_come_from_the_settings():
    x, amount = synthetic_xa()
    settings = stage2_settings(loss="gamma", max_iter=12, learning_rate=0.2, max_leaf_nodes=5,
                               min_samples_leaf=7, random_seed=11)
    params = fit_stage2(x, amount, settings).get_params()
    assert params["loss"] == "gamma"
    assert params["max_iter"] == 12
    assert params["learning_rate"] == 0.2
    assert params["max_leaf_nodes"] == 5
    assert params["min_samples_leaf"] == 7
    assert params["random_state"] == 11


def test_fit_stage2_fixed_parameters():
    x, amount = synthetic_xa()
    params = fit_stage2(x, amount, stage2_settings()).get_params()
    assert params["early_stopping"] is False
    assert params["categorical_features"] == "from_dtype"


def test_fit_stage2_other_parameters_are_sklearns_defaults():
    # Nothing beyond the settings and the two fixed choices is set.
    x, amount = synthetic_xa()
    params = fit_stage2(x, amount, stage2_settings()).get_params()
    defaults = HistGradientBoostingRegressor().get_params()
    set_here = {"loss", "max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                "random_state", "early_stopping", "categorical_features"}
    for name, value in defaults.items():
        if name not in set_here:
            assert params[name] == value, name


def test_fit_stage2_does_not_read_the_feature_blocks():
    # feature_blocks picks the input columns before the fit; the fit itself ignores it.
    x, amount = synthetic_xa()
    one = fit_stage2(x, amount, stage2_settings())
    other = Stage2Settings(loss="poisson", max_iter=10, learning_rate=0.1, max_leaf_nodes=7,
                           min_samples_leaf=5, random_seed=3, feature_blocks=("history",))
    two = fit_stage2(x, amount, other)
    np.testing.assert_array_equal(one.predict(x), two.predict(x))


def test_fit_stage2_no_early_stopping_with_more_than_10000_rows():
    # sklearn's early_stopping="auto" switches on above 10,000 rows; off, every tree is built.
    x, amount = synthetic_xa(n=12_000, company=False)
    model = fit_stage2(x, amount, stage2_settings(max_iter=15, learning_rate=0.01))
    assert model.n_iter_ == 15
    assert model.do_early_stopping_ is False


# --- fit_stage2: the company as a category ------------------------------------------------------

def test_fit_stage2_company_code_is_treated_as_categorical():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    assert model.is_categorical_.tolist() == [False, False, False, True]


def test_fit_stage2_without_company_nothing_is_categorical():
    x, amount = synthetic_xa(company=False)
    model = fit_stage2(x, amount, stage2_settings())
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
    model = fit_stage2(x, amount, stage2_settings(max_iter=50, learning_rate=0.2))
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
    model = fit_stage2(x, amount, stage2_settings(loss="squared_error", max_iter=1,
                                                  learning_rate=1.0, max_leaf_nodes=2,
                                                  min_samples_leaf=1))
    m = pd.Series(model.predict(x))
    assert m[codes == "TR04"].max() < m[codes != "TR04"].min()


# --- fit_stage2: what it learns -----------------------------------------------------------------

def test_fit_stage2_poisson_keeps_the_training_total():
    x, amount = synthetic_xa(n=2000)
    model = fit_stage2(x, amount, stage2_settings(max_iter=100, learning_rate=0.05))
    ratio = model.predict(x).sum() / amount.sum()
    assert ratio == pytest.approx(1.0, abs=1e-3)


def test_fit_stage2_poisson_keeps_the_total_per_company_roughly():
    # The company is a category: each company's total is kept too, not only the grand total.
    x, amount = synthetic_xa(n=2000)
    model = fit_stage2(x, amount, stage2_settings(max_iter=100, learning_rate=0.05))
    m = pd.Series(model.predict(x), index=x.index)
    for code in ("TR02", "TR04"):
        rows = x["company_code"] == code
        assert m[rows].sum() / amount[rows].sum() == pytest.approx(1.0, abs=0.02), code


def test_fit_stage2_poisson_with_one_tree_already_keeps_the_total_roughly():
    # The starting point is the mean amount (log link): the total holds from the first tree.
    x, amount = synthetic_xa(n=1000)
    model = fit_stage2(x, amount, stage2_settings(max_iter=1))
    assert model.predict(x).sum() / amount.sum() == pytest.approx(1.0, abs=0.02)


def test_fit_stage2_learns_the_signal():
    x, amount = synthetic_xa(n=2000)
    model = fit_stage2(x, amount, stage2_settings(max_iter=100, learning_rate=0.1))
    m = model.predict(x)
    # The amount grows with s: the prediction ranks the rows the same way, mostly.
    assert np.corrcoef(np.log(m), x["s"])[0, 1] > 0.9


def test_fit_stage2_constant_amount_predicts_that_amount():
    x, _ = synthetic_xa(n=200)
    amount = pd.Series(250.0, index=x.index)
    for loss in STAGE2_LOSSES:
        m = fit_stage2(x, amount, stage2_settings(loss=loss)).predict(x)
        np.testing.assert_allclose(m, 250.0, rtol=1e-6, err_msg=loss)


def test_fit_stage2_a_single_paid_row_fits():
    x = pd.DataFrame({"h": [1.0]}, index=[5])
    model = fit_stage2(x, pd.Series([123.0], index=[5]), stage2_settings(min_samples_leaf=1))
    assert model.predict(x)[0] == pytest.approx(123.0)


def test_fit_stage2_same_seed_gives_identical_predictions():
    x, amount = synthetic_xa()
    one = fit_stage2(x, amount, stage2_settings(random_seed=5)).predict(x)
    two = fit_stage2(x, amount, stage2_settings(random_seed=5)).predict(x)
    np.testing.assert_array_equal(one, two)


# --- fit_stage2: the inputs it accepts ----------------------------------------------------------

def test_fit_stage2_accepts_nan_features():
    x, amount = synthetic_xa()
    x.loc[x.index[:50], "s"] = np.nan
    assert np.isfinite(fit_stage2(x, amount, stage2_settings()).predict(x)).all()


def test_fit_stage2_int_amounts_fit_like_their_floats():
    x, amount = synthetic_xa()
    amount = amount.round().astype("int64")
    plain = fit_stage2(x, amount.astype("float64"), stage2_settings()).predict(x)
    ints = fit_stage2(x, amount, stage2_settings()).predict(x)
    np.testing.assert_array_equal(plain, ints)


def test_fit_stage2_does_not_change_its_inputs():
    x, amount = synthetic_xa()
    amount = amount.round().astype("int64")
    x_copy, amount_copy = x.copy(deep=True), amount.copy(deep=True)
    fit_stage2(x, amount, stage2_settings())
    pd.testing.assert_frame_equal(x, x_copy)
    pd.testing.assert_series_equal(amount, amount_copy)


def test_fit_stage2_does_not_change_the_settings():
    x, amount = synthetic_xa()
    settings = stage2_settings()
    copy = Stage2Settings(**vars(settings))
    fit_stage2(x, amount, settings)
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
        fit_stage2(x, change(amount), stage2_settings())
    assert str(error.value).startswith(message)


def test_fit_stage2_refuses_empty_amounts_before_fitting(no_fit):
    x, amount = synthetic_xa()
    with pytest.raises(ValueError) as error:
        fit_stage2(x.iloc[:0], amount.iloc[:0], stage2_settings())
    assert str(error.value) == NO_ROWS


def test_fit_stage2_no_fit_fixture_bites(no_fit):
    x, amount = synthetic_xa()
    with pytest.raises(AssertionError, match="fit was called"):
        fit_stage2(x, amount, stage2_settings())


def test_fit_stage2_unknown_loss_is_left_to_sklearn():
    # The settings file only lets STAGE2_LOSSES through; fit_stage2 does not check again.
    x, amount = synthetic_xa()
    with pytest.raises(ValueError):
        fit_stage2(x, amount, stage2_settings(loss="tweedie"))


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
    model = fit_stage2(x, amount, stage2_settings())
    m = predict_amount(model, x)
    assert isinstance(m, pd.Series)
    assert m.name == "m"
    assert m.index.equals(x.index)
    assert m.dtype == "float64"
    np.testing.assert_array_equal(m.to_numpy(), model.predict(x))


def test_predict_amount_on_new_rows_with_their_own_index():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    new_x, _ = synthetic_xa(n=30, seed=9)
    new_x.index = pd.Index([f"row-{i}" for i in range(30)])
    m = predict_amount(model, new_x)
    assert m.index.equals(new_x.index)
    assert len(m) == 30


def test_predict_amount_keeps_a_shuffled_index_row_by_row():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    shuffled = x.sample(frac=1.0, random_state=0)
    m = predict_amount(model, x)
    pd.testing.assert_series_equal(predict_amount(model, shuffled), m.loc[shuffled.index])


def test_predict_amount_on_zero_rows_raises_sklearns_value_error():
    # Pinned behaviour (not in the spec): sklearn refuses an empty x, so predict_amount does
    # not return an empty m. Step 14 always has forecast rows at the demo origin.
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    with pytest.raises(ValueError, match="0 sample"):
        predict_amount(model, x.iloc[:0])


def test_predict_amount_squared_error_below_zero_is_not_clipped():
    x, amount = interaction_x_amount()
    model = fit_stage2(x, amount, stage2_settings(loss="squared_error", max_iter=200,
                                                  learning_rate=0.5, max_leaf_nodes=2,
                                                  min_samples_leaf=1))
    m = predict_amount(model, x)
    corner = (x["a"] == 1) & (x["b"] == 1)
    assert (m[corner] < -20).all(), m[corner].iloc[0]
    np.testing.assert_allclose(m[corner], -23.75, atol=0.01)
    np.testing.assert_array_equal(m.to_numpy(), model.predict(x))


@pytest.mark.parametrize("loss", ["poisson", "gamma"])
def test_predict_amount_log_link_losses_stay_above_zero_on_the_same_data(loss):
    x, amount = interaction_x_amount()
    model = fit_stage2(x, amount, stage2_settings(loss=loss, max_iter=200, learning_rate=0.5,
                                                  max_leaf_nodes=2, min_samples_leaf=1))
    assert (predict_amount(model, x) > 0).all()


class NegativeModel:
    """A stand-in model whose predictions include values at and below 0."""
    feature_names_in_ = np.array(["h", "s"], dtype=object)

    def predict(self, x):
        return np.array([-5.0, 0.0, 3.0])[:len(x)]


def test_predict_amount_passes_negative_and_zero_predictions_through():
    x = pd.DataFrame({"h": [1.0, 2.0, 3.0], "s": [0.0, 0.0, 0.0]}, index=[7, 8, 9])
    m = predict_amount(NegativeModel(), x)
    pd.testing.assert_series_equal(m, pd.Series([-5.0, 0.0, 3.0], index=[7, 8, 9], name="m"))


def test_predict_amount_columns_in_another_order_exact_message():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    with pytest.raises(ValueError) as error:
        predict_amount(model, x[["s", "h", "z", "company_code"]])
    assert str(error.value) == columns_message(["s", "h", "z", "company_code"], FITTED)


def test_predict_amount_a_missing_column_exact_message():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    with pytest.raises(ValueError) as error:
        predict_amount(model, x.drop(columns="company_code"))
    assert str(error.value) == columns_message(["h", "s", "z"], FITTED)


def test_predict_amount_an_extra_column_exact_message():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    with pytest.raises(ValueError) as error:
        predict_amount(model, x.assign(extra=1.0))
    assert str(error.value) == columns_message(FITTED + ["extra"], FITTED)


def test_predict_amount_a_renamed_column_is_refused():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    with pytest.raises(ValueError, match="but the model was fitted on"):
        predict_amount(model, x.rename(columns={"z": "Z"}))


def test_predict_amount_message_matches_predict_paids():
    from collection_estimation.hurdle.stage1 import predict_paid
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    with pytest.raises(ValueError) as from_m:
        predict_amount(model, x.drop(columns="z"))
    # predict_paid's check runs before it touches classes_, so the regressor reaches it too.
    with pytest.raises(ValueError) as from_p:
        predict_paid(model, x.drop(columns="z"))
    assert str(from_m.value) == str(from_p.value)


def test_predict_amount_does_not_change_its_input():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    x_copy = x.copy(deep=True)
    predict_amount(model, x)
    pd.testing.assert_frame_equal(x, x_copy)


def test_predict_amount_on_x_with_nan_features_gives_finite_m():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    x.loc[x.index[:20], ["s", "z"]] = np.nan
    assert np.isfinite(predict_amount(model, x)).all()


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
                        pd.Series([50.0, 30.0, 120.0], index=x.index))
    assert ratio == 0.5
    assert type(ratio) is float


def test_total_ratio_above_one_by_hand():
    x = pd.DataFrame({"h": [1.0, 2.0]})
    assert total_ratio(FixedModel([30.0, 30.0]), x, pd.Series([10.0, 30.0])) == 1.5


def test_total_ratio_is_a_ratio_of_totals_not_a_mean_of_ratios():
    # Row ratios 2.0 and 0.5 (mean 1.25); totals 52 over 101.
    x = pd.DataFrame({"h": [1.0, 2.0]})
    ratio = total_ratio(FixedModel([2.0, 50.0]), x, pd.Series([1.0, 100.0]))
    assert ratio == pytest.approx(52.0 / 101.0)


def test_total_ratio_counts_negative_predictions():
    # m = -10, 30 sums to 20; amount 10 + 10 = 20: ratio 1.0, the negative m is not clipped.
    x = pd.DataFrame({"h": [1.0, 2.0]})
    assert total_ratio(FixedModel([-10.0, 30.0]), x, pd.Series([10.0, 10.0])) == 1.0


def test_total_ratio_accepts_int_amounts():
    x = pd.DataFrame({"h": [1.0, 2.0]})
    assert total_ratio(FixedModel([5.0, 5.0]), x, pd.Series([4, 6], dtype="int64")) == 1.0


def test_total_ratio_poisson_on_its_training_rows_is_about_one():
    x, amount = synthetic_xa(n=2000)
    model = fit_stage2(x, amount, stage2_settings(max_iter=100, learning_rate=0.05))
    ratio = total_ratio(model, x, amount)
    assert ratio == pytest.approx(1.0, abs=1e-3)
    assert ratio == pytest.approx(float(model.predict(x).sum() / amount.sum()))


def test_total_ratio_squared_error_on_its_training_rows_is_about_one():
    # Squared error starts at the mean and every leaf is a mean of residuals: the total holds.
    x, amount = synthetic_xa(n=1000)
    model = fit_stage2(x, amount, stage2_settings(loss="squared_error", max_iter=50))
    assert total_ratio(model, x, amount) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("loss", STAGE2_LOSSES)
def test_total_ratio_is_finite_and_positive_for_each_loss(loss):
    x, amount = synthetic_xa()
    ratio = total_ratio(fit_stage2(x, amount, stage2_settings(loss=loss)), x, amount)
    assert np.isfinite(ratio) and ratio > 0


def test_total_ratio_on_other_rows_than_the_training_rows():
    # It is a check of any paid rows, not only the training ones.
    x, amount = synthetic_xa(n=1000)
    model = fit_stage2(x, amount, stage2_settings(max_iter=50))
    other_x, other_amount = synthetic_xa(n=300, seed=4)
    ratio = total_ratio(model, other_x, other_amount)
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
        total_ratio(model, x, change(pd.Series([1.0, 2.0, 3.0], index=x.index)))
    assert str(error.value).startswith(message)
    assert model.calls == 0


def test_total_ratio_refuses_empty_rows_before_predicting():
    x = pd.DataFrame({"h": pd.Series([], dtype="float64")})
    model = FixedModel([])
    with pytest.raises(ValueError) as error:
        total_ratio(model, x, pd.Series([], index=x.index, dtype="float64"))
    assert str(error.value) == NO_ROWS
    assert model.calls == 0


def test_total_ratio_fixed_model_counts_calls():
    # The call counter used above does count: one prediction per good call.
    x = pd.DataFrame({"h": [1.0]})
    model = FixedModel([1.0])
    total_ratio(model, x, pd.Series([1.0]))
    assert model.calls == 1


def test_total_ratio_refuses_wrong_columns_like_predict_amount():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    with pytest.raises(ValueError) as error:
        total_ratio(model, x[["s", "h", "z", "company_code"]], amount)
    assert str(error.value) == columns_message(["s", "h", "z", "company_code"], FITTED)


def test_total_ratio_amounts_are_checked_before_the_columns():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    bad = amount.where(amount.index != amount.index[0], 0.0)
    with pytest.raises(ValueError) as error:
        total_ratio(model, x.drop(columns="z"), bad)
    assert str(error.value).startswith("amount is 0 or below in 1 rows")


def test_total_ratio_does_not_change_its_inputs():
    x, amount = synthetic_xa()
    model = fit_stage2(x, amount, stage2_settings())
    amount = amount.round().astype("int64")
    x_copy, amount_copy = x.copy(deep=True), amount.copy(deep=True)
    total_ratio(model, x, amount)
    pd.testing.assert_frame_equal(x, x_copy)
    pd.testing.assert_series_equal(amount, amount_copy)
