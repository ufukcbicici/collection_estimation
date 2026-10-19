"""Stage 1: will the customer pay anything in the target week? A classifier for p.

This module turns rows of the training table into the model's input, fits the classifier
and predicts p. Choosing the rows (which forecast week, which rows are known) happens in
`training_table`.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Columns that must never be features: identities, absolute time, and the labels.
NOT_FEATURES = ["key", "company_code", "customer_id", "origin_week", "target_week",
                "label_status", "amount", "amount_deflated", "paid"]


def feature_columns(block_columns: dict[str, list[str]], blocks: tuple[str, ...]) -> list[str]:
    """The feature columns: the horizon h, then the columns of each block in `blocks`
    (`block_columns` from `training_table.block_feature_columns`). The company code is added
    separately (see `stage1_features`).

    Raises ValueError for an unknown block, or if a block holds a column that must never be
    a feature (an identity, an absolute week or a label).
    """
    unknown = [block for block in blocks if block not in block_columns]
    if unknown:
        raise ValueError(f"unknown feature blocks {unknown}; known: {list(block_columns)}")
    columns = ["h"] + [column for block in blocks for column in block_columns[block]]
    forbidden = [column for column in columns if column in NOT_FEATURES]
    if forbidden:
        raise ValueError(f"columns {forbidden} must never be features")
    return columns


def stage1_features(rows: pd.DataFrame, columns: list[str], companies: list[str],
                    company_as_feature: bool) -> pd.DataFrame:
    """The model's input for `rows`: the `columns` as numbers (True/False as 1.0/0.0, NaN
    kept), plus, if `company_as_feature`, the company code as a categorical column whose
    categories are `companies` (the same for every fit, so a code always means the same).

    Raises ValueError if a column is missing, or a row's company is not in `companies`.
    """
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(f"the rows have no columns {missing}")
    features = rows[columns].astype("float64")
    if company_as_feature:
        # key=str: a missing code (None) cannot be ordered beside text otherwise.
        unknown = sorted(set(rows["company_code"]) - set(companies), key=str)
        if unknown:
            raise ValueError(f"companies {unknown} are not in the settings' company list")
        features["company_code"] = pd.Categorical(rows["company_code"],
                                                  categories=sorted(set(companies)))
    return features


def fit_stage1(train_x: pd.DataFrame, paid: pd.Series, settings,
               class_weight: str | None = None) -> HistGradientBoostingClassifier:
    """Fit the "will it pay?" classifier on `train_x` (from `stage1_features`) and `paid`
    (True/False per row), with the `[stage1]` settings.

    Log loss with no class weights, so that the predictions are probabilities: they will be
    multiplied by amounts. `class_weight` exists only for the diagnostic; the forecast never
    sets it. Early stopping is off: sklearn would otherwise hold out a random tenth of the
    rows, mixing weeks. The company code, if present, is used as a category.

    Raises ValueError if `paid` has a missing value, does not line up with `train_x`, or
    holds only one class.
    """
    if not paid.index.equals(train_x.index):
        raise ValueError("paid and train_x must have the same rows (the same index)")
    if paid.isna().any():
        raise ValueError(f"paid has {int(paid.isna().sum())} missing values: training rows must "
                         f"all be labelled")
    paid = paid.astype(bool)
    if paid.nunique() < 2:
        raise ValueError(f"paid is {bool(paid.iloc[0]) if len(paid) else 'empty'} in every "
                         f"row; the classifier needs both outcomes")
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=settings.max_iter,
        learning_rate=settings.learning_rate,
        max_leaf_nodes=settings.max_leaf_nodes,
        min_samples_leaf=settings.min_samples_leaf,
        early_stopping=False,
        categorical_features="from_dtype",
        class_weight=class_weight,
        random_state=settings.random_seed,
    )
    model.fit(train_x, paid)
    return model


def predict_paid(model: HistGradientBoostingClassifier, x: pd.DataFrame) -> pd.Series:
    """p for each row of `x`: the probability that the customer pays anything in the target
    week. Raises ValueError unless `x` has the columns the model was fitted on, in order."""
    fitted = list(model.feature_names_in_)
    if list(x.columns) != fitted:
        raise ValueError(f"x has columns {list(x.columns)}, but the model was fitted on "
                         f"{fitted}")
    paid_column = list(model.classes_).index(True)
    return pd.Series(model.predict_proba(x)[:, paid_column], index=x.index, name="p")
