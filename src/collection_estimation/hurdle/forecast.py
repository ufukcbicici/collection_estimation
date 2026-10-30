"""One forecast (U13): the cash expected in each of the next weeks, made at the end of one
week T (the origin), per company and consolidated.

U13.2, the customers: both stages fitted on the rows whose answers were known at T, then, for
every customer known at T and every h, p (stage 1, "will it pay?"), m (stage 2, "how much,
if it pays?") and the expected cash p x m, deflated (TRY of week 0). The same steps as
pipeline steps 12 and 14, in one call, so that one forecast, the evaluation over many
origins (U14) and the production forecast (U17) all take the same path.
"""
import numpy as np
import pandas as pd

from .stage1 import feature_columns, fit_stage1, predict_paid, stage1_features
from .stage2 import exposure, fit_stage2, predict_amount
from .training_table import rows_at_origin, rows_known_at

# The columns of the table `customer_forecasts` returns, in order.
CUSTOMER_COLUMNS = ["key", "company_code", "customer_id", "origin_week", "h", "target_week",
                    "p", "m", "expected"]


def customer_forecasts(table: pd.DataFrame, origin: int, settings,
                       block_columns: dict[str, list[str]], companies: list[str]) -> pd.DataFrame:
    """The forecast made at the end of week `origin` (T) for every customer known at T and
    every h: p, m and expected = p x m, deflated (TRY of week 0).

    `table` is the training table, `settings` the whole settings (its `[stage1]` and
    `[stage2]` are used), `block_columns` the columns of each feature block
    (`training_table.block_feature_columns`) and `companies` the settings' company list.

    Stage 1 is fitted on the rows known at T (`rows_known_at`), stage 2 on those of them that
    paid, with the exposure learned from the rows known at T; both predict the rows at T
    (`rows_at_origin`), whose labels are not needed. One row per customer and h, with the
    columns `CUSTOMER_COLUMNS` and the table's index, so the actual cash can be joined on.

    Raises ValueError if the table has no rows at T, if p, m or expected is NaN or infinite,
    or if m is below 0 (a customer cannot pay a negative amount; `[stage2] loss`
    "squared_error" can predict one, "poisson" and "gamma" cannot), and as the fits do.
    """
    train_rows = rows_known_at(table, origin)
    forecast_rows = rows_at_origin(table, origin)
    if forecast_rows.empty:
        raise ValueError(f"the table has no rows at origin week {origin}")
    stage1, stage2 = settings.stage1, settings.stage2

    # Stage 1, "will it pay?", on every row known at T.
    columns1 = feature_columns(block_columns, stage1.feature_blocks)
    stage1_model = fit_stage1(
        stage1_features(train_rows, columns1, companies, stage1.company_as_feature),
        train_rows["paid"], stage1)
    p = predict_paid(stage1_model, stage1_features(forecast_rows, columns1, companies,
                                                   stage1.company_as_feature))

    # Stage 2, "how much, if it pays?", on the rows known at T that paid; m = s x exposure.
    columns2 = feature_columns(block_columns, stage2.feature_blocks)
    paid_rows = train_rows[train_rows["paid"].astype(bool)]
    stage2_model = fit_stage2(
        stage1_features(paid_rows, columns2, companies, stage1.company_as_feature),
        paid_rows["amount_deflated"], exposure(train_rows, paid_rows, stage2), stage2)
    m = predict_amount(stage2_model,
                       stage1_features(forecast_rows, columns2, companies,
                                       stage1.company_as_feature),
                       exposure(train_rows, forecast_rows, stage2))

    forecasts = forecast_rows[CUSTOMER_COLUMNS[:6]].assign(p=p, m=m, expected=p * m)
    for column in ("p", "m", "expected"):
        bad = int((~np.isfinite(forecasts[column].astype("float64"))).sum())
        if bad:
            raise ValueError(f"origin week {origin}: {column} is NaN or infinite in {bad} rows")
    negative = int((forecasts["m"] < 0).sum())
    if negative:
        raise ValueError(f"origin week {origin}: m is below 0 in {negative} rows; a customer "
                         f"cannot pay a negative amount ([stage2] loss {stage2.loss!r}; "
                         f"'poisson' and 'gamma' never predict one)")
    return forecasts
