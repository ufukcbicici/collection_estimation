"""Stage 1: will the customer pay anything in the target week? A classifier for p.

This module turns rows of the training table into the model's input, fits the classifier
and predicts p. Choosing the rows (which forecast week, which rows are known) happens in
`training_table`.
"""
import pandas as pd

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
