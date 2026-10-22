"""Stage 2: if the customer pays in the target week, how much? A regressor for m.

Fitted on the paid rows only, applied to every row: m answers "how much, if it pays", which
every row has an answer to, and is always multiplied by stage 1's p. The input is the same as
stage 1's (`stage1.stage1_features`); choosing the rows happens in `training_table`.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


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


def fit_stage2(train_x: pd.DataFrame, amount: pd.Series, settings) -> HistGradientBoostingRegressor:
    """Fit the "how much, if it pays?" regressor on the paid rows: `train_x` (from
    `stage1_features`) and their `amount` (deflated, TRY of week 0), with the `[stage2]`
    settings.

    The loss is `settings.loss`: "poisson" (the default) keeps the predicted total close to
    the actual total on the training rows, which is what p x m needs when it is summed. Early
    stopping is off: sklearn would otherwise hold out a random tenth of the rows, mixing weeks.
    The company code, if present, is used as a category.

    Raises ValueError as `check_amounts` does.
    """
    amount = check_amounts(train_x, amount)
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
    model.fit(train_x, amount)
    return model
