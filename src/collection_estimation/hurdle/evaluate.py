"""The evaluation (U14): the forecast made at many origins, scored against the cash that really
came and against two references, `deflated` and the simple financial rule.

U14.1, the actual cash: per company and consolidated, for every origin and h, the cash that
came in the target week, nominal and deflated (TRY of week 0). It is read straight from the
weekly amounts, not rebuilt from the model's parts: gross receipts + refunds (refunds are
negative), which `side_tables.check_side_tables` proves equals the panel's cash + A + G + R.

U14.2, the forecasts: `forecast.forecast` at every origin, stacked. Everything that can be
checked cheaply is checked for every origin before the first (slow) fit.
"""
import time
from datetime import date
from typing import Callable

import pandas as pd

from .forecast import CONSOLIDATED, Forecast, forecast
from .prices import deflate, projected_levels
from .topups import topup_forecasts

# The columns of the table `actual_totals` returns, in order.
ACTUAL_COLUMNS = ["company_code", "origin_week", "h", "target_week", "actual", "actual_real"]


def actual_totals(weekly: pd.DataFrame, weeks: pd.DataFrame, levels: pd.Series,
                  companies: list[str], origins: list[int], horizons: int) -> pd.DataFrame:
    """The cash that really came (hindsight) in each target week of each origin, in the shape
    of `forecast.forecast_totals`: per company in `companies` and consolidated (company
    "ALL"), for every origin in `origins` and h = 1 to `horizons`.

    `weekly` is ALL the weekly amounts (`series.weekly_amounts`, identified customers and
    not), `weeks` the week calendar (`series.week_calendar`, with `is_missing`) and `levels`
    the published price levels (`prices.published_levels`). actual = gross receipts + refunds
    of the target week, nominal TRY; actual_real = actual divided by the target week's price
    level, TRY of week 0 (as the top-ups and the panel's label are deflated).

    A target week that is missing or not in the calendar (after the last week) has no row: no
    data is not 0. In a week with data, a company with no line gets 0, a real 0. Columns
    `ACTUAL_COLUMNS`, sorted by company (as in `forecast_totals`: the companies, then "ALL"),
    origin and h.

    Raises ValueError if `horizons` is below 1, the company list holds "ALL" or is empty, or
    the weekly amounts hold cash of a company outside the list (it would be in "ALL" but in
    no company), and as `deflate` does for a week with no price level.
    """
    if horizons < 1:
        raise ValueError(f"horizons is {horizons}; horizons start at 1")
    if not companies:
        raise ValueError("the company list is empty")
    if CONSOLIDATED in companies:
        raise ValueError(f"{CONSOLIDATED!r} is the code of the consolidated rows; it cannot be "
                         f"a company")
    outside = sorted(set(weekly["company_code"]) - set(companies), key=str)
    if outside:
        raise ValueError(f"the weekly amounts have cash of companies not in the list: {outside}")

    # Every (origin, h) whose target week has data; the others get no row.
    with_data = set(weeks.loc[~weeks["is_missing"], "week"])
    targets = pd.DataFrame([{"origin_week": origin, "h": h, "target_week": origin + h}
                            for origin in sorted(set(origins)) for h in range(1, horizons + 1)
                            if origin + h in with_data],
                           columns=["origin_week", "h", "target_week"]).astype("int64")

    net = weekly.assign(actual=weekly["gross_receipts"] + weekly["refunds"])
    by_week = net.groupby(["company_code", "week"], as_index=False)["actual"].sum()
    by_week = by_week.rename(columns={"week": "target_week"})
    grid = pd.DataFrame({"company_code": sorted(set(companies))}).merge(targets, how="cross")
    per_company = grid.merge(by_week, on=["company_code", "target_week"], how="left",
                             validate="many_to_one")
    per_company["actual"] = per_company["actual"].fillna(0.0).astype("float64")
    per_company = per_company.sort_values(["company_code", "origin_week", "h"],
                                          ignore_index=True)

    consolidated = per_company.groupby(["origin_week", "h", "target_week"],
                                       as_index=False)["actual"].sum()
    consolidated["company_code"] = CONSOLIDATED
    found = pd.concat([per_company, consolidated], ignore_index=True)
    found["company_code"] = found["company_code"].astype(object)
    found["actual_real"] = deflate(found["actual"], found["target_week"], levels)
    return found[ACTUAL_COLUMNS]


def rolling_forecasts(table: pd.DataFrame, origins: list[int], settings, *,
                      block_columns: dict[str, list[str]], companies: list[str],
                      new_cash: pd.DataFrame, refunds: pd.DataFrame, unidentified: pd.DataFrame,
                      levels: pd.Series, calendar: pd.DataFrame, week_zero: date,
                      on_origin_done: Callable[[int, int, int, float], None] | None = None
                      ) -> Forecast:
    """`forecast.forecast` at every origin in `origins` (sorted, each once), with its
    `totals` and `customers` stacked in origin order. The other arguments are `forecast`'s.
    Each origin is fitted afresh on the rows known at it; nothing passes between origins.

    Before the first fit, for every origin: the table has rows at it, for h = 1 to
    `[receivables] horizons`; the top-ups (`topup_forecasts`, all origins at once) and the
    projected price levels (`projected_levels`) can be made. A mistake in the settings or the
    inputs then fails in seconds, not after the fits of the origins before it.

    `on_origin_done(done, total, origin, seconds)`, if given, is called after each origin
    (e.g. to print progress).

    Raises ValueError if `origins` is empty, the table's index holds a label twice (the
    actual cash is joined to the customer rows by index), the company list holds "ALL", an
    origin has no rows or rows for other h, and as the cheap steps and `forecast` do.
    """
    origins = sorted(set(origins))
    if not origins:
        raise ValueError("there are no origins to forecast at")
    if table.index.has_duplicates:
        repeated = table.index[table.index.duplicated()].unique()
        raise ValueError(f"the table's index holds {len(repeated)} labels more than once "
                         f"(first: {list(repeated[:5])}); the customer rows are joined by index")
    if CONSOLIDATED in companies:
        raise ValueError(f"{CONSOLIDATED!r} is the code of the consolidated rows; it cannot be "
                         f"a company")
    horizons = settings.receivables.horizons
    wanted = set(range(1, horizons + 1))
    h_at = table.groupby("origin_week")["h"].unique()
    for origin in origins:
        if origin not in h_at.index:
            raise ValueError(f"the table has no rows at origin week {origin}")
        found = {int(h) for h in h_at[origin]}
        if found != wanted:
            raise ValueError(f"the table's rows at origin week {origin} are for h "
                             f"{sorted(found)}, but [receivables] horizons is {horizons}")
    # Dry runs, only to fail in seconds: forecast() makes both again for its own origin.
    topup_forecasts(new_cash, refunds, unidentified, levels, calendar, origins, horizons,
                    settings.topups.method, settings.topups.window, settings.topups.with_level)
    for origin in origins:
        projected_levels(week_zero, origin, [origin + h for h in range(1, horizons + 1)])

    results = []
    for done, origin in enumerate(origins, start=1):
        started = time.perf_counter()
        results.append(forecast(table, origin, settings, block_columns=block_columns,
                                companies=companies, new_cash=new_cash, refunds=refunds,
                                unidentified=unidentified, levels=levels, calendar=calendar,
                                week_zero=week_zero))
        if on_origin_done is not None:
            on_origin_done(done, len(origins), origin, time.perf_counter() - started)
    return Forecast(totals=pd.concat([r.totals for r in results], ignore_index=True),
                    customers=pd.concat([r.customers for r in results]))
