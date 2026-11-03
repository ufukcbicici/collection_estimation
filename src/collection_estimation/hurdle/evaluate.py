"""The evaluation (U14): the forecast made at many origins, scored against the cash that really
came and against two references, `deflated` and the simple financial rule.

U14.1, the actual cash: per company and consolidated, for every origin and h, the cash that
came in the target week, nominal and deflated (TRY of week 0). It is read straight from the
weekly amounts, not rebuilt from the model's parts: gross receipts + refunds (refunds are
negative), which `side_tables.check_side_tables` proves equals the panel's cash + A + G + R.

U14.2, the forecasts: `forecast.forecast` at every origin, stacked. Everything that can be
checked cheaply is checked for every origin before the first (slow) fit.

U14.3, the `deflated` reference: `benchmark.deflated_forecasts` on all the cash of each company
(and of all of them, fitted on its own), deflated, refitted at every origin on the weeks known
at it, then times the price level projected at the origin, as the model's total is. In three
variants: with the recent level, without it, and without it but with a trend.

U14.4, the financial-rule reference: what a finance person would forecast from the open
invoices, with no model. Each invoice open at the origin is expected on its due date plus the
customer's usual delay; a date already passed means "late, expected next week"; an invoice
long past due is doubtful and left out. Plus the same top-ups as the model.
"""
import time
from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from .benchmark import deflated_forecasts
from .forecast import CONSOLIDATED, Forecast, forecast
from .prices import deflate, projected_levels
from .receivables_asof import (end_of_week, open_invoices_at, payment_delays_at,
                               rates_known_at, to_try)
from .series import week_numbers
from .topups import topup_forecasts

# The columns of the table `actual_totals` returns, in order.
ACTUAL_COLUMNS = ["company_code", "origin_week", "h", "target_week", "actual", "actual_real"]
# The columns of the table `deflated_reference` returns, in order. real_forecast is deflated
# (TRY of week 0), forecast nominal TRY: real_forecast x price_level.
REFERENCE_COLUMNS = ["company_code", "origin_week", "h", "target_week", "reference",
                     "real_forecast", "price_level", "forecast", "fallback"]
# The variants of the `deflated` reference, each (with_level, with_trend): the recent level
# among the inputs or not, and the week number (a straight-line trend) or not. deflated_trend
# was the best setting of a scratch grid on the validation origins (2026-11-03).
REFERENCES = {"deflated": (True, False),
              "deflated_nolevel": (False, False),
              "deflated_trend": (False, True)}

# What the financial rule decides for each open invoice (the `status` column).
DOUBTFUL = "doubtful"                # too long past due: left out
LATE = "late"                        # its expected date has passed: expected in week T+1
EXPECTED = "expected"                # expected in one of the weeks T+1 ... T+horizons
AFTER_HORIZON = "after horizon"      # expected after the last of those weeks
# Where an invoice's delay came from (the `delay_from` column).
FROM_CUSTOMER = "customer"           # the customer's own usual delay
FROM_COMPANY = "company"             # too few cleared invoices: its company's median delay
FROM_NOTHING = "none"                # no delay known in the company either: 0 days
# The columns of the table `rule_invoices` returns, in order.
RULE_INVOICE_COLUMNS = ["key", "company_code", "document_no", "origin_week", "net_due_date",
                        "outstanding_try", "days_past_due", "delay_days", "delay_from",
                        "expected_date", "status", "h"]
# The columns of the `totals` of `rule_reference`, in order. All amounts nominal TRY.
RULE_COLUMNS = ["company_code", "origin_week", "h", "target_week", "invoice_cash", "late_cash",
                "fallback_delay_cash", "topup_cash", "price_level", "forecast"]


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


def deflated_reference(weekly: pd.DataFrame, weeks: pd.DataFrame, levels: pd.Series,
                       calendar: pd.DataFrame, companies: list[str], origins: list[int],
                       horizons: int, week_zero: date) -> pd.DataFrame:
    """The `deflated` reference made at every origin in `origins`, for h = 1 to `horizons`,
    per company in `companies` and consolidated (company "ALL"), in both variants of
    `REFERENCES`.

    The series of a company is its cash (gross receipts + refunds, as in `actual_totals`) in
    each week with data, divided by the week's published price level (`levels`), TRY of week
    0; a week with data where the company has no line is a real 0, a week without data is not
    in the series. The consolidated series is all the companies' cash, fitted on its own (so
    the companies' forecasts need not add up to it). At each origin T and h,
    `benchmark.deflated_forecasts` is fitted on the series' weeks up to T, with `calendar`
    (the calendar features of every week), and forecasts week T + h; then times the price
    level of T + h projected at T (`prices.projected_levels` from `week_zero`).

    Columns `REFERENCE_COLUMNS`, sorted by company (the companies, then "ALL"), origin, h and
    reference (in the order of `REFERENCES`: "deflated", "deflated_nolevel", "deflated_trend").

    Raises ValueError if `origins` is empty, `horizons` is below 1, the company list holds
    "ALL" or is empty, or the weekly amounts hold cash of a company outside the list, as
    `deflate` does for a week with no price level, and as `deflated_forecasts` (no calendar
    row for a week, no data in the weeks of the recent level) and `projected_levels` do.
    """
    origins = sorted(set(origins))
    if not origins:
        raise ValueError("there are no origins to forecast at")
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

    # The cash per week with data (rows) and company (columns), 0 where a company has no line.
    with_data = sorted(set(weeks.loc[~weeks["is_missing"], "week"]))
    net = weekly.assign(actual=weekly["gross_receipts"] + weekly["refunds"])
    cash = (net[net["week"].isin(with_data)]
            .pivot_table(index="week", columns="company_code", values="actual", aggfunc="sum")
            .reindex(index=with_data, columns=sorted(set(companies)), fill_value=0.0)
            .fillna(0.0))
    cash[CONSOLIDATED] = cash.sum(axis=1)
    week_numbers = pd.Series(cash.index, index=cash.index)
    real = {company: deflate(cash[company], week_numbers, levels) for company in cash.columns}

    projected = {origin: projected_levels(week_zero, origin,
                                          [origin + h for h in range(1, horizons + 1)])
                 for origin in origins}
    rows = []
    for company, series in real.items():
        for origin in origins:
            for h in range(1, horizons + 1):
                for reference, (with_level, with_trend) in REFERENCES.items():
                    made = deflated_forecasts(series, calendar, h, origin, [origin],
                                              with_level, with_trend).iloc[0]
                    level = float(projected[origin].loc[origin + h])
                    rows.append({"company_code": company, "origin_week": origin, "h": h,
                                 "target_week": origin + h, "reference": reference,
                                 "real_forecast": float(made["forecast"]), "price_level": level,
                                 "forecast": float(made["forecast"]) * level,
                                 "fallback": bool(made["fallback"])})
    found = pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
    found[["company_code", "reference"]] = found[["company_code", "reference"]].astype(object)
    return found


def invoice_delays(invoices: pd.DataFrame, ledger: pd.DataFrame, cutoff: date,
                   min_cleared: int) -> pd.DataFrame:
    """`invoices` (from `open_invoices_at`) with two columns added: delay_days, the days late
    the rule expects each invoice to be paid, and delay_from, where that number came from.

    The customer's usual delay (`payment_delays_at`: the median days late of its invoices
    cleared by `cutoff`, if it has at least `min_cleared`); if it has none, the median of the
    usual delays of its company's customers; if the company has none either, 0. Rounded to
    whole days, a half to the even number (pandas' rounding: 2.5 -> 2, 3.5 -> 4); it moves
    an expected date by at most one day.
    """
    delays = payment_delays_at(ledger, cutoff, min_cleared)
    customer_delay = invoices["key"].map(delays.set_index("key")["usual_delay_days"])

    # The company of each customer that has a usual delay, then the median per company.
    company_of_key = ledger[["key", "company_code"]].drop_duplicates()
    known_delays = delays.dropna(subset=["usual_delay_days"])
    known_delays = known_delays.merge(company_of_key, on="key", how="left")
    company_median = known_delays.groupby("company_code")["usual_delay_days"].median()
    company_delay = invoices["company_code"].map(company_median)

    invoices = invoices.copy()
    invoices["delay_from"] = FROM_CUSTOMER
    invoices.loc[customer_delay.isna(), "delay_from"] = FROM_COMPANY
    invoices.loc[customer_delay.isna() & company_delay.isna(), "delay_from"] = FROM_NOTHING

    delay = customer_delay.fillna(company_delay).fillna(0.0)
    invoices["delay_days"] = delay.round().astype("int64")
    return invoices


def rule_invoices(ledger: pd.DataFrame, rates: pd.DataFrame, week_zero: date, origin: int,
                  horizons: int, min_cleared: int, doubtful_days: int) -> pd.DataFrame:
    """Every invoice open at the end of week `origin` (T), and what the financial rule
    expects of it.

    1. The open invoices at T (`open_invoices_at`, the ledger as it stood then), each
       outstanding amount in TRY at the exchange rate known at T.
    2. The expected payment date: the net due date + the usual delay (`invoice_delays`).
    3. The status (one of the constants above): DOUBTFUL if more than `doubtful_days` days
       past due at T (left out); otherwise LATE if the expected date is not after T (expected
       in week T + 1); otherwise EXPECTED if it falls in week T + 1 ... T + `horizons`;
       otherwise AFTER_HORIZON.

    One row per invoice, columns `RULE_INVOICE_COLUMNS`; h is the week ahead the rule expects
    it in (1 for LATE; for a DOUBTFUL row it means nothing, as the row counts nowhere). Only
    the EXPECTED and LATE rows count in the forecast. Raises
    ValueError if an open invoice has no net due date, and as `to_try` does for a currency
    with no exchange rate known at T.
    """
    cutoff = end_of_week(week_zero, origin)
    cutoff_day = pd.Timestamp(cutoff)

    # 1. The open invoices at T, in TRY at the rates known at T.
    invoices = open_invoices_at(ledger, cutoff)
    no_due_date = invoices["net_due_date"].isna()
    if no_due_date.any():
        first = invoices.loc[no_due_date, "document_no"].head(5).tolist()
        raise ValueError(f"at {cutoff}, {int(no_due_date.sum())} open invoices have no net due "
                         f"date (first: {first})")
    invoices["outstanding_try"] = to_try(invoices["outstanding"], invoices["currency"],
                                         rates_known_at(rates, cutoff), cutoff)
    invoices["origin_week"] = origin

    # 2. When the rule expects each to be paid.
    invoices = invoice_delays(invoices, ledger, cutoff, min_cleared)
    invoices["days_past_due"] = (cutoff_day - invoices["net_due_date"]).dt.days
    invoices["expected_date"] = (invoices["net_due_date"]
                                 + pd.to_timedelta(invoices["delay_days"], unit="D"))
    invoices["h"] = week_numbers(invoices["expected_date"], week_zero) - origin

    # 3. The status. Set from the weakest to the strongest, so a later line wins.
    is_doubtful = invoices["days_past_due"] > doubtful_days
    is_late = invoices["expected_date"] <= cutoff_day
    invoices["status"] = EXPECTED
    invoices.loc[invoices["h"] > horizons, "status"] = AFTER_HORIZON
    invoices.loc[is_late, "status"] = LATE
    invoices.loc[is_doubtful, "status"] = DOUBTFUL
    # A late invoice is expected now: in the first week ahead.
    invoices.loc[is_late, "h"] = 1

    return invoices[RULE_INVOICE_COLUMNS]


def rule_totals(invoices: pd.DataFrame, topups: pd.DataFrame, projected: pd.Series,
                companies: list[str], origin: int, horizons: int) -> pd.DataFrame:
    """The financial rule's forecast at `origin` per company and h, then the consolidated
    rows (company "ALL", every amount added up). All amounts nominal TRY.

    invoice_cash: the outstanding TRY of the EXPECTED and LATE `invoices` (from
    `rule_invoices`) in that week; late_cash and fallback_delay_cash: the parts of it that
    are LATE, and whose delay did not come from the customer. topup_cash: the top-ups
    (`topups`, from `topup_forecasts` at the same origin, deflated) times the price level of
    the target week projected at the origin (`projected`). forecast = invoice_cash +
    topup_cash. Columns `RULE_COLUMNS`.

    Raises ValueError if an invoice or a top-up belongs to a company outside the list (its
    cash would be in no row), or the top-ups do not have exactly one row for a company and h.
    """
    outside = sorted(set(invoices["company_code"]) - set(companies), key=str)
    if outside:
        raise ValueError(f"the open invoices at origin week {origin} have companies not in the "
                         f"list: {outside}")
    outside = sorted(set(topups["company_code"]) - set(companies), key=str)
    if outside:
        raise ValueError(f"the top-ups at origin week {origin} have companies not in the "
                         f"list: {outside}")
    counted = invoices[invoices["status"].isin([EXPECTED, LATE])]

    rows = []
    for company in sorted(set(companies)):
        for h in range(1, horizons + 1):
            in_week = counted[(counted["company_code"] == company) & (counted["h"] == h)]
            is_late = in_week["status"] == LATE
            is_fallback = in_week["delay_from"] != FROM_CUSTOMER

            topup = topups[(topups["company_code"] == company) & (topups["h"] == h)]
            if len(topup) != 1:
                raise ValueError(f"the top-ups have {len(topup)} rows for company {company!r}, "
                                 f"h {h} at origin week {origin}; expected 1")
            price_level = float(projected.loc[origin + h])
            topup_cash = float(topup["topup"].iloc[0]) * price_level

            invoice_cash = float(in_week["outstanding_try"].sum())
            rows.append({"company_code": company, "origin_week": origin, "h": h,
                         "target_week": origin + h, "invoice_cash": invoice_cash,
                         "late_cash": float(in_week.loc[is_late, "outstanding_try"].sum()),
                         "fallback_delay_cash":
                             float(in_week.loc[is_fallback, "outstanding_try"].sum()),
                         "topup_cash": topup_cash, "price_level": price_level,
                         "forecast": invoice_cash + topup_cash})
    totals = pd.DataFrame(rows, columns=RULE_COLUMNS)

    # The consolidated rows: every amount added up over the companies; one price level.
    amounts = ["invoice_cash", "late_cash", "fallback_delay_cash", "topup_cash", "forecast"]
    by_week = totals.groupby(["origin_week", "h", "target_week"], as_index=False)
    consolidated = by_week[amounts].sum()
    consolidated["price_level"] = [float(projected.loc[week])
                                   for week in consolidated["target_week"]]
    consolidated["company_code"] = CONSOLIDATED

    found = pd.concat([totals, consolidated[RULE_COLUMNS]], ignore_index=True)
    found["company_code"] = found["company_code"].astype(object)
    return found


@dataclass(frozen=True)
class RuleReference:
    """The financial-rule reference over many origins: `totals` (`rule_totals` of every
    origin) and `invoices` (`rule_invoices` of every origin: each open invoice and what the
    rule expected of it)."""
    totals: pd.DataFrame
    invoices: pd.DataFrame


def rule_reference(ledger: pd.DataFrame, rates: pd.DataFrame, origins: list[int], settings, *,
                   companies: list[str], new_cash: pd.DataFrame, refunds: pd.DataFrame,
                   unidentified: pd.DataFrame, levels: pd.Series, calendar: pd.DataFrame,
                   week_zero: date) -> RuleReference:
    """The financial rule's forecast at every origin in `origins` (sorted, each once), for
    h = 1 to `[receivables] horizons`, per company and consolidated, nominal TRY.

    At each origin: the open invoices and what the rule expects of them (`rule_invoices`,
    with `[receivables] min_cleared_for_delay` and `[evaluation] rule_doubtful_days`), the
    same top-ups as the model (`topup_forecasts` with the `[topups]` settings, on the side
    tables `new_cash`, `refunds`, `unidentified`, the published `levels` and the
    `calendar`), and the price levels projected at the origin (`projected_levels`).

    Raises ValueError if `origins` is empty, the company list is empty or holds "ALL", and
    as the steps do.
    """
    origins = sorted(set(origins))
    if not origins:
        raise ValueError("there are no origins to forecast at")
    if not companies:
        raise ValueError("the company list is empty")
    if CONSOLIDATED in companies:
        raise ValueError(f"{CONSOLIDATED!r} is the code of the consolidated rows; it cannot be "
                         f"a company")
    horizons = settings.receivables.horizons
    topup_settings = settings.topups

    all_totals = []
    all_invoices = []
    for origin in origins:
        invoices = rule_invoices(ledger, rates, week_zero, origin, horizons,
                                 settings.receivables.min_cleared_for_delay,
                                 settings.evaluation.rule_doubtful_days)
        topups = topup_forecasts(new_cash, refunds, unidentified, levels, calendar, [origin],
                                 horizons, topup_settings.method, topup_settings.window,
                                 topup_settings.with_level)
        target_weeks = [origin + h for h in range(1, horizons + 1)]
        projected = projected_levels(week_zero, origin, target_weeks)

        all_totals.append(rule_totals(invoices, topups, projected, companies, origin, horizons))
        all_invoices.append(invoices)

    totals = pd.concat(all_totals, ignore_index=True)
    # Sorted like the other tables: the companies, then "ALL"; then origin and h.
    company_order = {company: i for i, company in enumerate(sorted(set(companies)))}
    company_order[CONSOLIDATED] = len(company_order)
    totals["order"] = totals["company_code"].map(company_order)
    totals = totals.sort_values(["order", "origin_week", "h"], ignore_index=True)
    totals = totals.drop(columns="order")
    return RuleReference(totals=totals, invoices=pd.concat(all_invoices, ignore_index=True))
