"""One forecast (U13): the cash expected in each of the next weeks, made at the end of one
week T (the origin), per company and consolidated.

U13.2, the customers: both stages fitted on the rows whose answers were known at T, then, for
every customer known at T and every h, p (stage 1, "will it pay?"), m (stage 2, "how much,
if it pays?") and the expected cash p x m, deflated (TRY of week 0). The same steps as
pipeline steps 12 and 14, in one call, so that one forecast, the evaluation over many
origins (U14) and the production forecast (U17) all take the same path.

U13.3, the totals: per company and h, the customers' p x m added up, plus the top-ups (U12:
new customers A, unidentified cash G, refunds R, negative), deflated; then times the price
level projected at T (U13.1), in nominal TRY. Plus the consolidated rows, company "ALL".

U13.4, `forecast`: the steps above in one call, the one path every forecast takes.
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .prices import projected_levels
from .stage1 import feature_columns, fit_stage1, predict_paid, stage1_features
from .stage2 import exposure, fit_stage2, predict_amount
from .topups import topup_forecasts
from .training_table import rows_at_origin, rows_known_at

# The columns of the table `customer_forecasts` returns, in order.
CUSTOMER_COLUMNS = ["key", "company_code", "customer_id", "origin_week", "h", "target_week",
                    "p", "m", "expected"]

# The columns of the table `forecast_totals` returns, in order. All amounts are deflated (TRY
# of week 0) except `total`, which is nominal TRY: real_total x price_level.
TOTAL_COLUMNS = ["company_code", "origin_week", "h", "target_week", "customers",
                 "new_customer_cash", "unidentified_cash", "refunds", "real_total",
                 "price_level", "total", "fallback"]
# The company code of the consolidated rows: all companies added up.
CONSOLIDATED = "ALL"


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


def _check_pairs(what: str, found: set, wanted: set) -> None:
    """Raise ValueError unless `found` (company, h) pairs are exactly the `wanted` ones."""
    missing, extra = sorted(wanted - found), sorted(found - wanted, key=str)
    if missing:
        raise ValueError(f"the {what} have no rows for (company, h) {missing[:10]}")
    if extra:
        raise ValueError(f"the {what} have rows for (company, h) {extra[:10]}, which are not in "
                         f"the company list or the horizons")


def forecast_totals(customers: pd.DataFrame, topups: pd.DataFrame, levels: pd.Series,
                    companies: list[str], horizons: int) -> pd.DataFrame:
    """The forecast per company and h, and consolidated: `customers` (from
    `customer_forecasts`) added up per company and h, plus `topups` (from
    `topups.topup_forecasts`, made at the same origin), real_total = customers +
    new_customer_cash + unidentified_cash + refunds (refunds are negative), deflated; and
    total = real_total x price_level, nominal TRY, with the level of each target week from
    `levels` (`prices.projected_levels` at the origin).

    One row per company in `companies` and h = 1 to `horizons`, then the consolidated rows
    (company "ALL": every amount added up over the companies, the same price level, fallback
    True if any company's is), with the columns `TOTAL_COLUMNS`.

    Raises ValueError if `horizons` is below 1, the company list holds "ALL", the customers
    and top-ups are not for one single origin week, a customer row lacks its company, h or
    expected (adding up would drop it without a word), a target week is not origin + h, the
    top-ups hold a company and h twice, either table lacks a (company, h) of the list or holds
    one outside it (a missing company is an error, never a zero), `levels` lacks a target
    week, or a value is NaN or infinite.
    """
    if horizons < 1:
        raise ValueError(f"horizons is {horizons}; horizons start at 1")
    if CONSOLIDATED in companies:
        raise ValueError(f"{CONSOLIDATED!r} is the code of the consolidated rows; it cannot be "
                         f"a company")
    origins = set(customers["origin_week"]) | set(topups["origin_week"])
    if len(origins) != 1:
        raise ValueError(f"the customers and top-ups must be for one origin week; they hold "
                         f"{sorted(origins)}")
    origin = origins.pop()
    # groupby would leave out a row with no company or h, and sum() would skip a NaN amount:
    # cash lost without an error. So they are refused here, before adding up.
    for column in ("company_code", "h", "expected"):
        missing = int(customers[column].isna().sum())
        if missing:
            raise ValueError(f"the customers have no {column} in {missing} rows")
    for what, table in (("customers", customers), ("top-ups", topups)):
        wrong = int((table["target_week"] != origin + table["h"]).sum())
        if wrong:
            raise ValueError(f"the {what} have a target week other than origin {origin} + h in "
                             f"{wrong} rows")

    wanted = {(company, h) for company in companies for h in range(1, horizons + 1)}
    topup_pairs = list(zip(topups["company_code"], topups["h"]))
    if len(topup_pairs) != len(set(topup_pairs)):
        raise ValueError("the top-ups hold a company and h more than once")
    _check_pairs("top-ups", set(topup_pairs), wanted)
    customer_sums = (customers.groupby(["company_code", "h"])["expected"].sum()
                     .rename("customers").reset_index())
    _check_pairs("customers", set(zip(customer_sums["company_code"], customer_sums["h"])),
                 wanted)

    totals = topups.merge(customer_sums, on=["company_code", "h"], validate="one_to_one")
    totals["price_level"] = totals["target_week"].map(levels)
    if totals["price_level"].isna().any():
        missing = sorted(set(totals.loc[totals["price_level"].isna(), "target_week"]))
        raise ValueError(f"no price level for target weeks {missing}")
    totals["real_total"] = (totals["customers"] + totals["new_customer_cash"]
                            + totals["unidentified_cash"] + totals["refunds"])
    totals["total"] = totals["real_total"] * totals["price_level"]
    totals = totals.sort_values(["company_code", "h"], ignore_index=True)

    amounts = ["customers", "new_customer_cash", "unidentified_cash", "refunds",
               "real_total", "total"]
    consolidated = totals.groupby(["origin_week", "h", "target_week"], as_index=False).agg(
        {**{column: "sum" for column in amounts}, "price_level": "first", "fallback": "any"})
    consolidated["company_code"] = CONSOLIDATED

    found = pd.concat([totals[TOTAL_COLUMNS], consolidated[TOTAL_COLUMNS]], ignore_index=True)
    for column in amounts + ["price_level"]:
        bad = int((~np.isfinite(found[column].astype("float64"))).sum())
        if bad:
            raise ValueError(f"{column} is NaN or infinite in {bad} rows")
    return found


@dataclass(frozen=True)
class Forecast:
    """One forecast made at one origin: `totals` (`forecast_totals`, per company and h plus
    the consolidated rows) and `customers` (`customer_forecasts`, the p, m and p x m behind
    the totals' `customers` column)."""
    totals: pd.DataFrame
    customers: pd.DataFrame


def forecast(table: pd.DataFrame, origin: int, settings, *,
             block_columns: dict[str, list[str]], companies: list[str],
             new_cash: pd.DataFrame, refunds: pd.DataFrame, unidentified: pd.DataFrame,
             levels: pd.Series, calendar: pd.DataFrame, week_zero: date) -> Forecast:
    """The forecast made at the end of week `origin` (T) for h = 1 to `[receivables]
    horizons`, per company and consolidated, deflated and nominal.

    The customers (`customer_forecasts` on `table`, `block_columns` and `companies`), the
    top-ups (`topups.topup_forecasts` on the side tables `new_cash`, `refunds` and
    `unidentified`, with `levels` the published price levels and `calendar` the calendar
    features, by the `[topups]` settings), the price levels projected at T
    (`prices.projected_levels` from `week_zero`), and the totals (`forecast_totals`).

    Raises ValueError if the table's rows at T are not for h = 1 to `[receivables] horizons`
    (checked before the fits, so the error names the setting); otherwise each step raises
    ValueError on what it cannot use.
    """
    horizons = settings.receivables.horizons
    found = set(table.loc[table["origin_week"] == origin, "h"])
    wanted = set(range(1, horizons + 1))
    # No rows at T is left to customer_forecasts, which says so.
    if found and found != wanted:
        raise ValueError(f"the table's rows at origin week {origin} are for h {sorted(found)}, "
                         f"but [receivables] horizons is {horizons}")
    customers = customer_forecasts(table, origin, settings, block_columns, companies)
    topups = topup_forecasts(new_cash, refunds, unidentified, levels, calendar, [origin],
                             horizons, settings.topups.method, settings.topups.window,
                             settings.topups.with_level)
    projected = projected_levels(week_zero, origin,
                                 [origin + h for h in range(1, horizons + 1)])
    totals = forecast_totals(customers, topups, projected, companies, horizons)
    return Forecast(totals=totals, customers=customers)
