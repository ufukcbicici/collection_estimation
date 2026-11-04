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

U14.5, the scores: every method's forecasts in one long table (`stack_forecasts`), then per
method and company the error against the actual cash (`evaluation_scores`): WAPE, bias, the
mean absolute error in TRY, and MAPE for the consolidated rows; per h, pooled over all h, and
for the total of the h weeks of each origin. Then, consolidated, whether one method is really
more accurate than another or only luckier on these weeks (`paired_wape_bootstrap`).
"""
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .benchmark import deflated_forecasts
from .cache import source_fingerprint, table_fingerprint
from .forecast import CONSOLIDATED, Forecast, forecast
from .prices import deflate, projected_levels
from .receivables_asof import (end_of_week, open_invoices_at, payment_delays_at,
                               rates_known_at, to_try)
from .series import week_numbers
from .topups import topup_forecasts

# The code a forecast at one origin runs (beside this file), for `rolling_fingerprint`. Not
# evaluate.py itself: it changes often, and `rolling_forecasts` only loops over the origins
# (after a change to that loop, delete the cache by hand).
MODEL_CODE = ("forecast.py", "stage1.py", "stage2.py", "topups.py", "benchmark.py", "prices.py",
              "inflation.py", "training_table.py")

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

# The method names of the stacked forecasts; the references are named as in `REFERENCES`.
HURDLE = "hurdle"
RULE = "rule"
# The columns of the table `stack_forecasts` returns, in order. forecast is nominal TRY.
STACK_COLUMNS = ["method", "company_code", "origin_week", "h", "target_week", "forecast"]
# The `horizon` of a score row: "1" ... "5" for one h, or one of these two.
POOLED = "pooled"                    # every (origin, h) together
HORIZON_TOTAL = "total"              # per origin the sum over its h weeks, then scored
# The columns of the table `evaluation_scores` returns, in order. rows is the number of
# (origin, h) scored (of origins, for HORIZON_TOTAL); forecast, actual and abs_error are sums,
# nominal TRY.
SCORE_COLUMNS = ["method", "company_code", "horizon", "rows", "forecast", "actual", "abs_error",
                 "wape", "bias", "mean_abs_error", "mape"]
# The columns of the table `paired_wape_bootstrap` returns, in order. horizon is POOLED or
# HORIZON_TOTAL; difference = wape - wape_against (below 0: method is better); low and high
# the 5% and 95% points of the difference over the resamples (a 90% interval).
BOOTSTRAP_COLUMNS = ["method", "against", "horizon", "rows", "wape", "wape_against",
                     "difference", "low", "high", "share_better", "samples"]


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


def rolling_fingerprint(table: pd.DataFrame, origins: list[int], settings, *,
                        block_columns: dict[str, list[str]], companies: list[str],
                        new_cash: pd.DataFrame, refunds: pd.DataFrame,
                        unidentified: pd.DataFrame, levels: pd.Series, calendar: pd.DataFrame,
                        week_zero: date) -> list:
    """Everything `rolling_forecasts` makes its result from, with its arguments, as a
    fingerprint for `cache.load_with_cache`: if any of it changes, the cached forecasts are
    made again. The fits are seeded, so a cached result equals a new one.

    The origins; the settings the fits read (`[receivables]`, `[stage1]`, `[stage2]`, and the
    method of `[topups]`; not e.g. `[diagnostic]`, which does not change a forecast); the
    feature lists, companies and week 0; a hash of every input table; and a hash of the
    model's code (`MODEL_CODE`).
    """
    topups = settings.topups
    here = Path(__file__).parent
    return [
        sorted(set(origins)),
        # 1. The settings the fits read.
        asdict(settings.receivables),
        asdict(settings.stage1),
        asdict(settings.stage2),
        [topups.method, topups.window, topups.with_level],
        # 2. The other arguments.
        block_columns,
        sorted(companies),
        week_zero.isoformat(),
        # 3. The input tables.
        table_fingerprint(table),
        table_fingerprint(new_cash),
        table_fingerprint(refunds),
        table_fingerprint(unidentified),
        table_fingerprint(levels),
        table_fingerprint(calendar),
        # 4. The model's code.
        source_fingerprint(*(here / name for name in MODEL_CODE)),
    ]


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


def stack_forecasts(rolled: Forecast | None, reference: pd.DataFrame,
                    rule: RuleReference) -> pd.DataFrame:
    """Every method's forecasts in one long table, one row per method, company, origin and h,
    nominal TRY: the hurdle model's `total` (from `rolling_forecasts`; left out if `rolled`
    is None, i.e. step 17 ran without the model), each variant of the `deflated` reference
    (from `deflated_reference`, named as in `REFERENCES`) and the financial rule (from
    `rule_reference`).

    Columns `STACK_COLUMNS`, the methods in that order (hurdle, the references, rule), each
    method's rows in the order they came. Raises ValueError if the reference table lacks one
    of the variants.
    """
    keys = ["company_code", "origin_week", "h", "target_week"]
    parts = []

    # 1. The hurdle model, if it ran.
    if rolled is not None:
        hurdle = rolled.totals[keys + ["total"]].rename(columns={"total": "forecast"})
        hurdle.insert(0, "method", HURDLE)
        parts.append(hurdle)

    # 2. The `deflated` references, one variant after the other.
    for name in REFERENCES:
        mine = reference.loc[reference["reference"] == name, keys + ["forecast"]].copy()
        if mine.empty:
            raise ValueError(f"the reference table has no {name!r} rows")
        mine.insert(0, "method", name)
        parts.append(mine)

    # 3. The financial rule.
    by_rule = rule.totals[keys + ["forecast"]].copy()
    by_rule.insert(0, "method", RULE)
    parts.append(by_rule)

    stacked = pd.concat(parts, ignore_index=True)
    stacked["method"] = stacked["method"].astype(object)
    stacked["company_code"] = stacked["company_code"].astype(object)
    return stacked[STACK_COLUMNS]


def score_row(method: str, company: str, horizon: str, forecast: pd.Series, actual: pd.Series,
              with_mape: bool) -> dict:
    """One row of `evaluation_scores`: the forecasts against the actual cash, both nominal
    TRY, one value per (origin, h) (or per origin, for a total).

    WAPE = sum |forecast - actual| / sum actual; bias = sum forecast / sum actual - 1; both
    NaN if the actual cash adds up to 0 or less (a share of nothing). mean_abs_error is the
    mean |forecast - actual| in TRY. MAPE = mean of |forecast - actual| / actual, only if
    `with_mape` and every actual is above 0, else NaN.
    """
    errors = (forecast - actual).abs()
    came = float(actual.sum())
    made = float(forecast.sum())

    wape = float("nan")
    bias = float("nan")
    if came > 0:
        wape = float(errors.sum()) / came
        bias = made / came - 1

    mape = float("nan")
    if with_mape and (actual > 0).all():
        mape = float((errors / actual).mean())

    return {"method": method, "company_code": company, "horizon": horizon,
            "rows": len(actual), "forecast": made, "actual": came,
            "abs_error": float(errors.sum()), "wape": wape, "bias": bias,
            "mean_abs_error": float(errors.mean()), "mape": mape}


def scored_rows(forecasts: pd.DataFrame, actual: pd.DataFrame, horizons: int) -> pd.DataFrame:
    """The forecasts (from `stack_forecasts`) beside the actual cash of their week (column
    actual, from `actual_totals`): only the (company, origin, h) with actual cash, checked
    first. Shared by `evaluation_scores` and `paired_wape_bootstrap`.

    Raises ValueError if `horizons` is below 1, a forecast's h is not 1 to `horizons`, a
    method has two forecasts for one (company, origin, h), a forecast is NaN or infinite, or
    a method has no forecast for a row with actual cash (the methods would be scored on
    different weeks).
    """
    if horizons < 1:
        raise ValueError(f"horizons is {horizons}; horizons start at 1")
    keys = ["company_code", "origin_week", "h", "target_week"]

    # 1. Each forecast is for h 1 to horizons, and each method has each row once.
    outside = sorted(set(forecasts["h"]) - set(range(1, horizons + 1)))
    if outside:
        raise ValueError(f"forecasts for h {outside}; horizons is {horizons}")
    repeated = forecasts.duplicated(["method"] + keys)
    if repeated.any():
        first = forecasts.loc[repeated, ["method"] + keys].iloc[0].to_dict()
        raise ValueError(f"{int(repeated.sum())} forecasts are there twice (first: {first})")
    # A forecast with no value: the sums would skip it and score that week as perfect.
    no_value = ~np.isfinite(forecasts["forecast"].astype("float64"))
    if no_value.any():
        first = forecasts.loc[no_value, ["method"] + keys].iloc[0].to_dict()
        raise ValueError(f"{int(no_value.sum())} forecasts are NaN or infinite (first: {first})")

    # 2. Every method has a forecast for every row with actual cash.
    wanted = actual[keys]
    methods = list(dict.fromkeys(forecasts["method"]))
    for method in methods:
        mine = forecasts.loc[forecasts["method"] == method, keys]
        found = wanted.merge(mine, on=keys, how="left", indicator=True)
        missing = found[found["_merge"] == "left_only"]
        if not missing.empty:
            first = missing[keys].iloc[0].to_dict()
            raise ValueError(f"{method!r} has no forecast for {len(missing)} rows with actual "
                             f"cash (first: {first})")

    # 3. The forecasts beside the actual cash of their week.
    return forecasts.merge(actual[keys + ["actual"]], on=keys, how="inner",
                           validate="many_to_one")


def evaluation_scores(forecasts: pd.DataFrame, actual: pd.DataFrame,
                      horizons: int) -> pd.DataFrame:
    """The scores of every method (`forecasts`, from `stack_forecasts`) against the actual cash
    (`actual`, from `actual_totals`), per method and company (the companies and "ALL"),
    nominal TRY. Only the (company, origin, h) with actual cash are scored.

    For each method and company, rows with `horizon`:
      - "1" ... str(`horizons`): the (origin, h) of that h;
      - POOLED: every (origin, h) together (a target week counts once per h aiming at it);
      - HORIZON_TOTAL: per origin, the forecasts and the actual cash added up over its h
        weeks, then scored as one value per origin. Only origins with actual cash in all
        `horizons` target weeks; no row if there is none.
    The numbers are `score_row`'s; MAPE only for "ALL". "ALL" is scored on its own rows, not
    as the sum of the companies (the `deflated` references fit it on its own).

    Columns `SCORE_COLUMNS`; the methods and companies in the order they come in `forecasts`,
    per company the h, then POOLED, then HORIZON_TOTAL.

    Raises ValueError as `scored_rows` does (the checks on the forecasts).
    """
    # 1. The rows scored, checked.
    rows = scored_rows(forecasts, actual, horizons)

    # 2. The scores, method by method and company by company.
    scores = []
    for (method, company), mine in rows.groupby(["method", "company_code"], sort=False):
        with_mape = company == CONSOLIDATED

        # Per h.
        for h in range(1, horizons + 1):
            at_h = mine[mine["h"] == h]
            if not at_h.empty:
                scores.append(score_row(method, company, str(h), at_h["forecast"],
                                        at_h["actual"], with_mape))

        # Every (origin, h) together.
        scores.append(score_row(method, company, POOLED, mine["forecast"], mine["actual"],
                                with_mape))

        # Per origin the sum over its h weeks; only origins with all of them.
        per_origin = mine.groupby("origin_week").agg(weeks=("h", "size"),
                                                     forecast=("forecast", "sum"),
                                                     actual=("actual", "sum"))
        complete = per_origin[per_origin["weeks"] == horizons]
        if not complete.empty:
            scores.append(score_row(method, company, HORIZON_TOTAL, complete["forecast"],
                                    complete["actual"], with_mape))

    return pd.DataFrame(scores, columns=SCORE_COLUMNS)


def weighted_wape(abs_error: np.ndarray, actual: np.ndarray, weights: np.ndarray) -> float:
    """WAPE with each value counted `weights` times: sum w |error| / sum w actual."""
    return float((weights * abs_error).sum() / (weights * actual).sum())


def drawn_counts(values: np.ndarray, drawn: list) -> np.ndarray:
    """For each of `values`, how often it is in `drawn` (0 if not)."""
    times = pd.Series(drawn).value_counts()
    return pd.Series(values).map(times).fillna(0).to_numpy(dtype="float64")


def resampled_weeks(weeks: list[int], rng: np.random.Generator) -> list[int]:
    """One resample of the target weeks: as many weeks as there are, drawn with replacement.
    All the forecasts aiming at one week go together, because they share its actual cash."""
    return list(rng.choice(weeks, size=len(weeks), replace=True))


def resampled_origins(origins: list[int], block: int, rng: np.random.Generator) -> list[int]:
    """One resample of the origins, in blocks of `block` consecutive origins (of the sorted
    list), each block starting at a random origin, until there are as many origins as before
    (the last block cut short). Nearby origins share target weeks, so they go together.

    A block that runs past the last origin goes on from the first (a circular block), so that
    every origin is drawn equally often; without it the first and last origins would be in
    few blocks and the resamples would lean on the middle ones."""
    drawn = []
    while len(drawn) < len(origins):
        start = int(rng.integers(0, len(origins)))
        for step in range(block):
            drawn.append(origins[(start + step) % len(origins)])
    return drawn[:len(origins)]


def bootstrap_row(method: str, against: str, horizon: str, error: np.ndarray,
                  error_against: np.ndarray, actual: np.ndarray,
                  draw_weights: Callable[[np.random.Generator], np.ndarray], samples: int,
                  seed: int) -> dict:
    """One row of `paired_wape_bootstrap`: the two methods' absolute errors and the actual
    cash, one value each per row (an (origin, h) or an origin's total), and
    `draw_weights(rng)`, how often each row counts in one resample."""
    wape = weighted_wape(error, actual, np.ones(len(actual)))
    wape_against = weighted_wape(error_against, actual, np.ones(len(actual)))

    # The same resamples for both methods.
    rng = np.random.default_rng(seed)
    differences = np.empty(samples)
    for i in range(samples):
        weights = draw_weights(rng)
        differences[i] = (weighted_wape(error, actual, weights)
                          - weighted_wape(error_against, actual, weights))

    return {"method": method, "against": against, "horizon": horizon, "rows": len(actual),
            "wape": wape, "wape_against": wape_against, "difference": wape - wape_against,
            "low": float(np.quantile(differences, 0.05)),
            "high": float(np.quantile(differences, 0.95)),
            "share_better": float((differences < 0).mean()), "samples": samples}


def paired_wape_bootstrap(forecasts: pd.DataFrame, actual: pd.DataFrame,
                          pairs: list[tuple[str, str]], horizons: int, samples: int,
                          seed: int) -> pd.DataFrame:
    """For each pair (method, against) of methods in `forecasts` (from `stack_forecasts`): is
    `method` more accurate than `against` on the consolidated cash, or only luckier on these
    weeks? Both are scored on the same resamples (paired), so the luck they share cancels.

    Two rows per pair, as in `evaluation_scores`:
      - POOLED: the WAPE over every (origin, h). A resample draws the target weeks with
        replacement (`resampled_weeks`); every forecast aiming at a drawn week counts as
        often as the week was drawn.
      - HORIZON_TOTAL: the WAPE of the per-origin totals over the h weeks (only origins with
        actual cash in all of them; no row if there is none). A resample draws the origins
        in blocks of `horizons` consecutive origins (`resampled_origins`; fewer if there are
        fewer origins).
    wape and wape_against are measured on the weeks as they are; difference = wape -
    wape_against; low and high its 5% and 95% points over `samples` resamples;
    share_better the share of resamples in which `method` has the lower WAPE. Each row
    starts from `seed`, so a pair's result does not depend on the other pairs.

    Few weeks give wide intervals: a guide, not a proof. Columns `BOOTSTRAP_COLUMNS`, the
    pairs in their order. Raises ValueError if `samples` is below 1, a pair names a method not
    in `forecasts` or the same method twice, there is no consolidated row with actual cash or
    a consolidated actual is not above 0, and as `scored_rows` does.
    """
    if samples < 1:
        raise ValueError(f"samples is {samples}; at least 1 is needed")
    methods = set(forecasts["method"])
    for method, against in pairs:
        if method == against:
            raise ValueError(f"the pair ({method!r}, {against!r}) compares a method with itself")
        unknown = sorted({method, against} - methods)
        if unknown:
            raise ValueError(f"the pair ({method!r}, {against!r}) names methods with no "
                             f"forecasts: {unknown}")

    # 1. The consolidated rows with actual cash, checked as for the scores.
    rows = scored_rows(forecasts, actual, horizons)
    rows = rows[rows["company_code"] == CONSOLIDATED]
    if rows.empty:
        raise ValueError(f"there are no consolidated ({CONSOLIDATED!r}) rows with actual cash "
                         f"to compare")
    if not (rows["actual"] > 0).all():
        raise ValueError("every consolidated actual must be above 0 for a WAPE")

    keys = ["origin_week", "h", "target_week"]
    results = []
    for method, against in pairs:
        # 2. The two methods side by side: one row per (origin, h).
        mine = rows.loc[rows["method"] == method, keys + ["forecast", "actual"]]
        theirs = rows.loc[rows["method"] == against, keys + ["forecast"]]
        both = mine.merge(theirs, on=keys, suffixes=("", "_against"), validate="one_to_one")

        # 3. Pooled: resample the target weeks.
        weeks = sorted(set(both["target_week"]))
        row_weeks = both["target_week"].to_numpy()

        def week_weights(rng):
            return drawn_counts(row_weeks, resampled_weeks(weeks, rng))

        results.append(bootstrap_row(
            method, against, POOLED,
            (both["forecast"] - both["actual"]).abs().to_numpy(),
            (both["forecast_against"] - both["actual"]).abs().to_numpy(),
            both["actual"].to_numpy(), week_weights, samples, seed))

        # 4. The per-origin totals: resample blocks of consecutive origins.
        per_origin = both.groupby("origin_week").agg(
            weeks=("h", "size"), forecast=("forecast", "sum"),
            forecast_against=("forecast_against", "sum"), actual=("actual", "sum"))
        complete = per_origin[per_origin["weeks"] == horizons]
        if complete.empty:
            continue
        origins = sorted(complete.index)
        block = min(horizons, len(origins))

        def origin_weights(rng):
            return drawn_counts(complete.index.to_numpy(),
                                resampled_origins(origins, block, rng))

        results.append(bootstrap_row(
            method, against, HORIZON_TOTAL,
            (complete["forecast"] - complete["actual"]).abs().to_numpy(),
            (complete["forecast_against"] - complete["actual"]).abs().to_numpy(),
            complete["actual"].to_numpy(), origin_weights, samples, seed))

    return pd.DataFrame(results, columns=BOOTSTRAP_COLUMNS)
