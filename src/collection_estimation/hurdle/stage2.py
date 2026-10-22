"""Stage 2: if the customer pays in the target week, how much? A regressor for m.

Fitted on the paid rows only, applied to every row: m answers "how much, if it pays", which
every row has an answer to, and is always multiplied by stage 1's p. The input is the same as
stage 1's (`stage1.stage1_features`); choosing the rows happens in `training_table`.

m = s x E. E, the exposure, sets the scale (`exposure`): 1 for every row (`[stage2] scale =
"none"`: s is the amount itself) or the customer's open balance, lifted to a floor
(`"open balance"`: s is the share of the balance paid). The trees learn s.
"""
import numbers

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance, r2_score


def check_amounts(train_x: pd.DataFrame, amount: pd.Series) -> pd.Series:
    """`amount` as floats, after checking it can be learned from: it lines up with `train_x`,
    holds at least one row, and every amount is finite and above 0 (stage 2 learns from paid
    rows only, and paid means an amount above 0). Raises ValueError otherwise."""
    if not amount.index.equals(train_x.index):
        raise ValueError("amount and train_x must have the same rows (the same index)")
    if amount.empty:
        raise ValueError("there are no paid rows to learn the amount from")
    amount = amount.astype("float64")
    if amount.isna().any():
        raise ValueError(f"amount has {int(amount.isna().sum())} missing values: stage 2 "
                         f"learns from paid rows, which all have an amount")
    if not np.isfinite(amount).all():
        raise ValueError("amount holds an infinite value")
    if (amount <= 0).any():
        raise ValueError(f"amount is 0 or below in {int((amount <= 0).sum())} rows: stage 2 "
                         f"learns from paid rows only (amount above 0)")
    return amount


def check_exposure(x: pd.DataFrame, exposure: pd.Series) -> pd.Series:
    """`exposure` as floats, after checking that it lines up with `x` and is finite and above
    0 in every row (m = s x E). Raises ValueError otherwise."""
    if not exposure.index.equals(x.index):
        raise ValueError("exposure and x must have the same rows (the same index)")
    exposure = exposure.astype("float64")
    usable = np.isfinite(exposure) & (exposure > 0)
    if not usable.all():
        raise ValueError(f"exposure is NaN, infinite, or 0 or below in "
                         f"{int((~usable).sum())} rows")
    return exposure


# The Tweedie power of each stage-2 loss. Fitting s = amount / E with row weights
# E ** (2 - power) is the same as fitting the amount with log E as a fixed offset: the model is
# judged on the amount, and E only sets the scale.
LOSS_POWER = {"poisson": 1, "gamma": 2, "squared_error": 0}


def fit_stage2(train_x: pd.DataFrame, amount: pd.Series, exposure: pd.Series,
               settings) -> HistGradientBoostingRegressor:
    """Fit the "how much, if it pays?" regressor on the paid rows: `train_x` (from
    `stage1_features`), their `amount` (deflated, TRY of week 0) and their `exposure` (from
    `exposure`), with the `[stage2]` settings. The model learns s = amount / exposure, each
    row weighted by exposure ** (2 - power of the loss), so it is judged on the amount; with
    an exposure of 1 everywhere this is the plain fit of the amount, unchanged.

    The loss is `settings.loss`: "poisson" (the default) keeps the predicted total close to
    the actual total on the training rows, which is what p x m needs when it is summed. Early
    stopping is off: sklearn would otherwise hold out a random tenth of the rows, mixing weeks.
    The company code, if present, is used as a category.

    Raises ValueError for an unknown loss, and as `check_amounts` and `check_exposure` do.
    """
    if settings.loss not in LOSS_POWER:
        raise ValueError(f"stage 2 loss {settings.loss!r} is not one of {list(LOSS_POWER)}")
    amount = check_amounts(train_x, amount)
    exposure = check_exposure(train_x, exposure)
    model = HistGradientBoostingRegressor(
        loss=settings.loss,
        max_iter=settings.max_iter,
        learning_rate=settings.learning_rate,
        max_leaf_nodes=settings.max_leaf_nodes,
        min_samples_leaf=settings.min_samples_leaf,
        early_stopping=False,
        categorical_features="from_dtype",
        random_state=settings.random_seed,
    )
    model.fit(train_x, amount / exposure,
              sample_weight=exposure ** (2 - LOSS_POWER[settings.loss]))
    return model


def predict_amount(model, x: pd.DataFrame, exposure: pd.Series) -> pd.Series:
    """m for each row of `x`: the amount the customer pays in the target week if it pays
    (deflated, TRY of week 0), = the model's s x `exposure` (from `exposure`, with the
    settings the model was fitted with). `model` is from `fit_stage2`. Nothing is clipped: a
    squared-error model can predict below 0, and that should stay visible. Raises ValueError
    unless `x` has the columns the model was fitted on, in order, or as `check_exposure`
    does."""
    fitted = list(model.feature_names_in_)
    if list(x.columns) != fitted:
        raise ValueError(f"x has columns {list(x.columns)}, but the model was fitted on "
                         f"{fitted}")
    exposure = check_exposure(x, exposure)
    return pd.Series(model.predict(x) * exposure.to_numpy(), index=x.index, name="m")


def total_ratio(model, x: pd.DataFrame, amount: pd.Series, exposure: pd.Series) -> float:
    """The predicted total over the actual total on paid rows: sum of m / sum of `amount`.
    Stage 2 alone (no p): 1.0 means that, given the customers paid, no cash was lost or
    invented in total. Raises ValueError as `check_amounts` and `predict_amount` do."""
    amount = check_amounts(x, amount)
    return float(predict_amount(model, x, exposure).sum() / amount.sum())


def paid_training_amounts(train_rows: pd.DataFrame) -> pd.Series:
    """The deflated amounts of the paid rows of `train_rows`. Raises ValueError if there is no
    paid row, or a paid row's amount is not above 0."""
    paid_amounts = train_rows.loc[train_rows["paid"].astype(bool), "amount_deflated"]
    if paid_amounts.empty:
        raise ValueError("the training rows have no paid row to take a mean amount from")
    if not (paid_amounts > 0).all():
        raise ValueError("a paid training row has an amount that is not above 0")
    return paid_amounts


def amount_references(train_rows: pd.DataFrame, rows: pd.DataFrame, window: int,
                      prior_weeks: float) -> pd.DataFrame:
    """Two simple guesses of the amount paid, if paid, for each of `rows`; one column each,
    with the same index as `rows`; both known at fit time, scored like a model:

      training mean   the mean deflated amount of the paid rows of `train_rows`, for every row
      own mean        the customer's own mean payment over its last `window` weeks, pulled
                      towards the training mean: (receipts + prior_weeks x training mean) /
                      (paying weeks + prior_weeks). Receipts = mean_receipts_<window> x
                      observed_weeks_<window>, the sum over the window (weeks up to the origin
                      t, so known at t). A customer with few paying weeks gets a mean close to
                      the training mean, one with none gets exactly the training mean.

    Raises ValueError if `prior_weeks` is not a number above 0, the history columns of
    `window` are missing or impossible (paying weeks negative, NaN or above the observed
    weeks; receipts NaN or below 0 in a window with data), or `train_rows` has no paid row
    (or a paid row whose amount is not above 0).
    """
    if (isinstance(prior_weeks, bool) or not isinstance(prior_weeks, numbers.Real)
            or not 0 < prior_weeks < float("inf")):
        raise ValueError(f"prior_weeks is {prior_weeks!r}; it must be a number above 0")
    mean, paying, observed = (f"mean_receipts_{window}", f"paying_weeks_{window}",
                              f"observed_weeks_{window}")
    missing = [column for column in (mean, paying, observed) if column not in rows.columns]
    if missing:
        raise ValueError(f"the rows have no history columns {missing} for window {window}")
    has_data = rows[observed] > 0
    # mean_receipts is NaN in a window without data; such a window holds no receipts.
    receipts = (rows[mean] * rows[observed]).where(has_data, 0.0)
    impossible = (~((rows[paying] >= 0) & (rows[paying] <= rows[observed]))
                  | ~(receipts >= 0))
    if impossible.any():
        raise ValueError(f"{int(impossible.sum())} rows have impossible history values for "
                         f"window {window} (paying weeks negative, NaN or above the observed "
                         f"weeks, or receipts NaN or below 0)")
    training_mean = float(paid_training_amounts(train_rows).mean())
    own_mean = (receipts + prior_weeks * training_mean) / (rows[paying] + prior_weeks)
    # With no paying week the formula is the training mean, but floating point can miss it
    # by one last digit; set it exactly.
    own_mean = own_mean.where(rows[paying] > 0, training_mean)
    return pd.DataFrame({"training mean": training_mean,
                         "own mean": own_mean.astype("float64")}, index=rows.index)


def exposure(train_rows: pd.DataFrame, rows: pd.DataFrame, settings) -> pd.Series:
    """E for each of `rows` (m = s x E), from the `[stage2]` settings; learned from
    `train_rows` only, so known at fit time:

      scale "none"           1.0 for every row: s is the amount itself
      scale "open balance"   the row's open_balance, lifted to the floor where it is lower:
                             s is the share of the balance paid. balance_floor "median" /
                             "mean": the median / mean deflated amount of the paid training
                             rows; "own mean": the customer's own mean payment over
                             own_mean_window weeks (see `amount_references`), for customers
                             paying with little or nothing open (advances, invoices issued
                             after the origin).

    Raises ValueError for an unknown scale or floor, if `rows` has no open_balance or a
    missing one, and as `paid_training_amounts` and `amount_references` do.
    """
    if settings.scale not in ("none", "open balance"):
        raise ValueError(f"stage 2 scale {settings.scale!r} is not 'none' or 'open balance'")
    if settings.scale == "none":
        return pd.Series(1.0, index=rows.index, name="exposure")
    if settings.balance_floor not in ("median", "mean", "own mean"):
        raise ValueError(f"stage 2 balance_floor {settings.balance_floor!r} is not 'median', "
                         f"'mean' or 'own mean'")
    if "open_balance" not in rows.columns:
        raise ValueError("the rows have no open_balance column to scale by")
    missing = int(rows["open_balance"].isna().sum())
    if missing:
        raise ValueError(f"open_balance is missing in {missing} rows")
    if settings.balance_floor == "own mean":
        floor = amount_references(train_rows, rows, settings.own_mean_window,
                                  settings.own_mean_prior_weeks)["own mean"]
    elif settings.balance_floor == "median":
        floor = float(paid_training_amounts(train_rows).median())
    else:
        floor = float(paid_training_amounts(train_rows).mean())
    balance = rows["open_balance"].astype("float64")
    return balance.where(balance >= floor, floor).rename("exposure")


def amount_scores(actual: pd.Series, h: pd.Series, predictions: dict[str, pd.Series],
                  reference: pd.Series) -> pd.DataFrame:
    """How close several guesses of the amount (`predictions`: name -> m, e.g. stage 2 and
    the references) come to the `actual` amounts of paid rows, for each horizon in `h` and for
    all rows together (h = "all"). One row per (model, h):

      rows             paid rows scored
      m <= 0           rows whose guess is 0 or below (squared error only)
      mean deviance    the mean Poisson deviance: errors judged relative to the guess, so
                       the few largest payments do not decide. Infinite if a guess is 0 or
                       below: a row that paid was given no chance of paying anything
      deviance skill   1 - mean deviance / `reference`'s mean deviance (D², the Poisson
                       version of R²): 0 = no better than the reference, 1 = perfect
      R2 hindsight     the usual R², whose baseline is the scored rows' own mean amount (not
                       known in advance); NaN with fewer than 2 rows
      WAPE             sum of |guess - actual| / sum of actual
      MAPE             the mean of |guess - actual| / actual over the rows: every payment
                       counts the same, so small payments weigh as much as large ones
      total ratio      sum of guesses / sum of actual; 1.0 = no cash lost or invented

    Raises ValueError if `actual`'s index repeats a label, if `h`, `reference` or a guess does
    not have the rows of `actual`, a guess is NaN or infinite, `reference` is not above 0, or
    `actual` is not a valid paid amount (see `check_amounts`).
    """
    # Rows are picked by label per horizon: a repeated label would be picked more than once.
    if not actual.index.is_unique:
        raise ValueError("actual's index has repeated labels; each row needs its own label")
    for name, series in (("h", h), ("reference", reference), *predictions.items()):
        if not series.index.equals(actual.index):
            raise ValueError(f"{name} must have the rows of actual (the same index)")
    actual = check_amounts(h.to_frame(), actual)
    for name, guess in (("reference", reference), *predictions.items()):
        if not np.isfinite(guess.astype("float64")).all():
            raise ValueError(f"{name} holds a NaN or infinite amount")
    if not (reference > 0).all():
        raise ValueError("reference must be above 0 in every row")
    scores = []
    for name, guess in predictions.items():
        for horizon in [*sorted(h.unique()), "all"]:
            rows = h.index if horizon == "all" else h.index[h == horizon]
            scores.append({"model": name, "h": horizon,
                           **amount_summary(actual[rows], guess[rows], reference[rows])})
    return pd.DataFrame(scores)


def amount_summary(actual: pd.Series, guess: pd.Series, reference: pd.Series) -> dict:
    """The scores of `amount_scores` for one set of rows (already checked there)."""
    non_positive = int((guess <= 0).sum())
    deviance = (float("inf") if non_positive
                else mean_poisson_deviance(actual, guess))
    reference_deviance = mean_poisson_deviance(actual, reference)
    return {
        "rows": len(actual),
        "m <= 0": non_positive,
        "mean deviance": deviance,
        # A reference that is exactly right leaves nothing to improve on: skill undefined.
        "deviance skill": (1 - deviance / reference_deviance if reference_deviance > 0
                           else float("nan")),
        "R2 hindsight": r2_score(actual, guess) if len(actual) >= 2 else float("nan"),
        "WAPE": float((guess - actual).abs().sum() / actual.sum()),
        "MAPE": float(((guess - actual).abs() / actual).mean()),
        "total ratio": float(guess.sum() / actual.sum()),
    }
