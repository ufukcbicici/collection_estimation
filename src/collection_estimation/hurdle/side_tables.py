"""The cash the panel does not hold, per company: new customers (A), refunds (R) and
unidentified cash (G).

The panel forecasts the gross receipts of keys known at the forecast week. A company's net
cash in week t+h is that, plus the gross receipts of keys not yet known at t (A), plus cash
with no identifiable customer (G), plus its refunds (R, negative). These tables hold the
actual values for weeks with data; forecasting them is a later step (the top-ups).
"""
import pandas as pd

from .keys import normalise_customer_id

# The columns of the table `new_customer_cash` returns, in order.
NEW_CASH_COLUMNS = ["company_code", "origin_week", "h", "target_week", "new_customer_cash"]

# The columns of the tables `refunds_by_week` and `unidentified_cash_by_week` return.
REFUND_COLUMNS = ["company_code", "week", "refunds"]
UNIDENTIFIED_COLUMNS = ["company_code", "week", "unidentified_cash"]


def new_customer_cash(weekly: pd.DataFrame, known: pd.DataFrame, weeks: pd.DataFrame,
                      horizons: int, companies: list[str]) -> pd.DataFrame:
    """Table A: per company, forecast week t and h, the gross receipts in week t+h of the
    company's keys that were not yet known at the end of t (their known_week > t).

    Such a key may become known before t+h; its cash still counts here, because the panel's
    rows for forecast week t cannot contain it. One row for every company in `companies`
    (each once), every t from 0 to the last week and every h = 1 ... `horizons` whose target
    week has data (not after the last week, not missing); 0.0 when no new customer paid. The
    companies are given, not taken from `known`, so that a company with no known key (e.g.
    only a placeholder of unidentified cash) still gets its rows, as in tables R and G.

    `weekly` is from `weekly_amounts`, `known` from `known_weeks`, `weeks` the week calendar.
    Raises ValueError if a key appears twice in `known`, or a key in `weekly` is not in it.
    """
    last_week = weeks["week"].max()
    weeks_with_data = set(weeks.loc[~weeks["is_missing"], "week"])

    repeated = known.loc[known["key"].duplicated(), "key"]
    if not repeated.empty:
        raise ValueError(f"{repeated.nunique()} keys appear more than once in the known table "
                         f"(first: {sorted(repeated.unique())[:5]})")
    known_week = known.set_index("key")["known_week"]
    unknown_keys = sorted(set(weekly["key"]) - set(known_week.index))
    if unknown_keys:
        raise ValueError(f"{len(unknown_keys)} keys in the weekly amounts are not in the known "
                         f"table (first: {unknown_keys[:10]})")

    # Every row the table must have, with 0.0 to start with.
    grid = pd.DataFrame({"company_code": sorted(set(companies))}).merge(
        pd.DataFrame({"origin_week": range(last_week + 1)}), how="cross").merge(
        pd.DataFrame({"h": range(1, horizons + 1)}), how="cross")
    grid["target_week"] = grid["origin_week"] + grid["h"]
    grid = grid[grid["target_week"].isin(weeks_with_data)]

    # Each key-week of receipts, once for every h, as seen from forecast week t = week - h.
    receipts = weekly[["key", "company_code", "week", "gross_receipts"]].merge(
        pd.DataFrame({"h": range(1, horizons + 1)}), how="cross")
    receipts["origin_week"] = receipts["week"] - receipts["h"]
    receipts["known_week"] = receipts["key"].map(known_week)
    is_new = receipts["known_week"] > receipts["origin_week"]
    new_cash = (receipts[is_new]
                .groupby(["company_code", "origin_week", "h"], as_index=False)["gross_receipts"]
                .sum()
                .rename(columns={"gross_receipts": "new_customer_cash"}))

    table = grid.merge(new_cash, on=["company_code", "origin_week", "h"], how="left")
    table["new_customer_cash"] = table["new_customer_cash"].fillna(0.0)
    table["company_code"] = table["company_code"].astype(object)
    for column in ("origin_week", "h", "target_week"):
        table[column] = table[column].astype("int64")
    table = table.sort_values(["company_code", "origin_week", "h"], ignore_index=True)
    return table[NEW_CASH_COLUMNS]


def refunds_by_week(weekly: pd.DataFrame, weeks: pd.DataFrame,
                    companies: list[str]) -> pd.DataFrame:
    """Table R: per company and week, the refunds paid out to customers (a negative amount).

    All keys count, known or not, so a week's refunds do not depend on the forecast week:
    one row per company in `companies` and per week with data (from 0 to the last week, not
    missing); 0.0 in a week without refunds. `weekly` is from `weekly_amounts`, `weeks` the
    week calendar.
    """
    return company_week_totals(weekly, weeks, companies, "refunds", "refunds")


def is_unidentified(table: pd.DataFrame, unidentified_customers: tuple[str, ...]) -> pd.Series:
    """True for the rows of `table` whose customer_id is one of the placeholder customers
    of unidentified cash (normalised like any customer ID, so "0099999" matches "99999")."""
    placeholders = {normalise_customer_id(customer) for customer in unidentified_customers}
    return table["customer_id"].isin(placeholders)


def unidentified_cash_by_week(unidentified: pd.DataFrame, weeks: pd.DataFrame,
                              companies: list[str]) -> pd.DataFrame:
    """Table G: per company and week, the gross receipts with no identifiable customer.

    `unidentified` is the weekly amounts of the placeholder customers only (see
    `is_unidentified`). One row per company in `companies` and per week with data; 0.0 in a
    week without any.
    """
    return company_week_totals(unidentified, weeks, companies, "gross_receipts",
                               "unidentified_cash")


def company_week_totals(weekly: pd.DataFrame, weeks: pd.DataFrame, companies: list[str],
                        column: str, name: str) -> pd.DataFrame:
    """Per company and week with data, the sum of `column` of `weekly`, called `name`.

    One row per company in `companies` (each once) and per week in the calendar that is not
    missing; 0.0 where `weekly` has nothing. Sorted by company, then week.
    """
    weeks_with_data = weeks.loc[~weeks["is_missing"], "week"]
    grid = pd.DataFrame({"company_code": sorted(set(companies))}).merge(
        pd.DataFrame({"week": weeks_with_data.to_numpy()}), how="cross")

    totals = weekly.groupby(["company_code", "week"], as_index=False)[column].sum()
    totals = totals.rename(columns={column: name})
    table = grid.merge(totals, on=["company_code", "week"], how="left")
    table[name] = table[name].fillna(0.0).astype("float64")
    table["company_code"] = table["company_code"].astype(object)
    table["week"] = table["week"].astype("int64")
    table = table.sort_values(["company_code", "week"], ignore_index=True)
    return table[["company_code", "week", name]]


# Panel + side tables must match the receipts and refunds within this (TRY).
TIE_OUT_TOLERANCE = 0.01


def check_side_tables(panel: pd.DataFrame, new_cash: pd.DataFrame, unidentified: pd.DataFrame,
                      refunds: pd.DataFrame, weekly: pd.DataFrame, companies: list[str]) -> int:
    """Check that every lira of every company's cash is in exactly one place.

    For every company, forecast week t and h with a labelled target week: the panel's
    labelled amounts (that company's keys) + new-customer cash A + unidentified cash G must
    equal the company's gross receipts in week t+h; and for every company and week, R must
    equal the company's refunds. `weekly` is ALL the weekly amounts (identified and not).
    Also checks that every company in `companies` is in A, G and R, that G has a row for
    every (company, target week) of A, and that no cash belongs to a company outside
    `companies`. Raises ValueError naming the first breaks; returns the
    number of company-week-horizon checks made.
    """
    expected = set(companies)
    outside = sorted(set(weekly["company_code"]) - expected)
    if outside:
        raise ValueError(f"the weekly amounts have cash of companies not in the list: {outside}")
    for name, table in (("A", new_cash), ("G", unidentified), ("R", refunds)):
        absent = sorted(expected - set(table["company_code"]))
        if absent:
            raise ValueError(f"table {name} has no rows for the companies {absent}")

    keys = ["company_code", "origin_week", "h"]
    labelled = panel[panel["label_status"] == "labelled"]
    in_panel = labelled.groupby(keys, as_index=False)["amount"].sum()
    # Every labelled (company, t, h) of the panel must have its row in table A.
    unmatched = in_panel.merge(new_cash[keys], on=keys, how="left", indicator=True)
    unmatched = unmatched[unmatched["_merge"] == "left_only"]
    if not unmatched.empty:
        raise ValueError(f"{len(unmatched)} labelled (company, t, h) of the panel have no row in "
                         f"table A (first: {unmatched[keys].head(5).values.tolist()})")

    checks = new_cash.merge(in_panel, on=keys, how="left")
    checks["amount"] = checks["amount"].fillna(0.0)
    g_by_week = unidentified.rename(columns={"week": "target_week"})
    checks = checks.merge(g_by_week, on=["company_code", "target_week"], how="left")
    # G has a row for every company and week with data, as A has: a missing one means the
    # side tables were built inconsistently, so it raises rather than counting it as 0.
    no_g_row = checks["unidentified_cash"].isna()
    if no_g_row.any():
        # Several A rows (one per h) share a target week: count each missing pair once.
        missing = (checks.loc[no_g_row, ["company_code", "target_week"]]
                   .drop_duplicates().sort_values(["company_code", "target_week"]))
        raise ValueError(f"table G has no row for {len(missing)} (company, target week) of "
                         f"table A (first: {missing.head(5).values.tolist()})")
    receipts = weekly.groupby(["company_code", "week"], as_index=False)["gross_receipts"].sum()
    receipts = receipts.rename(columns={"week": "target_week"})
    checks = checks.merge(receipts, on=["company_code", "target_week"], how="left")
    checks["gross_receipts"] = checks["gross_receipts"].fillna(0.0)

    accounted = checks["amount"] + checks["new_customer_cash"] + checks["unidentified_cash"]
    checks["difference"] = accounted - checks["gross_receipts"]
    breaks = checks[checks["difference"].abs() > TIE_OUT_TOLERANCE]
    if not breaks.empty:
        first = breaks[keys + ["difference"]].head(5).round(2).values.tolist()
        raise ValueError(f"for {len(breaks)} (company, t, h) the panel + A + G differ from the "
                         f"gross receipts (first, with the difference in TRY: {first})")

    weekly_refunds = weekly.groupby(["company_code", "week"], as_index=False)["refunds"].sum()
    refund_checks = refunds.merge(weekly_refunds, on=["company_code", "week"], how="left",
                                  suffixes=("", "_weekly"))
    refund_checks["refunds_weekly"] = refund_checks["refunds_weekly"].fillna(0.0)
    refund_breaks = refund_checks[(refund_checks["refunds"] - refund_checks["refunds_weekly"])
                                  .abs() > TIE_OUT_TOLERANCE]
    if not refund_breaks.empty:
        first = refund_breaks[["company_code", "week"]].head(5).values.tolist()
        raise ValueError(f"for {len(refund_breaks)} (company, week) table R differs from the "
                         f"weekly refunds (first: {first})")
    return len(checks)
