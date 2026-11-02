"""The evaluation (U14): the forecast made at many origins, scored against the cash that really
came and against two references, `deflated` and the simple financial rule.

U14.1, the actual cash: per company and consolidated, for every origin and h, the cash that
came in the target week, nominal and deflated (TRY of week 0). It is read straight from the
weekly amounts, not rebuilt from the model's parts: gross receipts + refunds (refunds are
negative), which `side_tables.check_side_tables` proves equals the panel's cash + A + G + R.
"""
import pandas as pd

from .forecast import CONSOLIDATED
from .prices import deflate

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
