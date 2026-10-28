"""The top-ups (U12): forecasts of the cash the panel does not hold, per company: new
customers (A), refunds (R) and unidentified cash (G), from the side tables (`side_tables.py`).

Each is a weekly series indexed by target week, deflated (TRY of week 0, like the panel's
label); re-inflation comes with the forecast (U13). A depends on h: it is the cash of the
customers not yet known at the origin t = target week - h, so there is one series A_h per h.
R and G do not depend on the origin: one series each, the same for every h.

A forecast made at origin T reads only target weeks up to T. For A_h these are the weeks
whose own origin is T - h or earlier, so all their cash was known at T.

U12.1, the trailing mean: the mean of the series over the `window` target weeks ending at T
(weeks without data left out, never counted as 0). For R and G it is flat over h; for A each
h has its own.

U12.2, the ridge: the `deflated` benchmark's model (`benchmark.deflated_forecasts`) on each
top-up series, refitted at every origin on the target weeks up to it, with one fit per h
(for R and G too: the same series, but predicting h weeks ahead is a different task).
`topup_forecasts` gives A, R and G of every company with either method in one table.

U12.3, the choice of method: `topup_actuals` is what really came, in the same shape, and
`score_topups` the error of a method's forecasts over many origins ([topups] in the settings).
"""
import numpy as np
import pandas as pd

from .benchmark import deflated_forecasts, recent_level
from .prices import deflate

# The columns of the table `trailing_mean_forecasts` returns, in order.
FORECAST_COLUMNS = ["origin_week", "h", "target_week", "forecast"]
# The columns of the table `topup_forecasts` returns, in order.
TOPUP_COLUMNS = ["company_code", "origin_week", "h", "target_week", "new_customer_cash",
                 "refunds", "unidentified_cash", "topup", "fallback"]
# The forecasting methods `topup_forecasts` knows.
METHODS = ("mean", "ridge")


def trailing_mean_forecasts(series: pd.Series, h: int, origins: list[int],
                            window: int) -> pd.DataFrame:
    """The trailing-mean forecast of `series` (one top-up of one company: cash per target
    week, deflated, indexed by week, weeks without data absent) for horizon `h`, made at each
    of `origins`: the mean of the `window` target weeks ending at the origin.

    For R and G pass the one series with any h; for A pass A_h, the series of that h.

    Returns one row per origin: origin_week, h, target_week (origin + h) and forecast.
    Raises ValueError if `h` or `window` is below 1, or an origin has no week with data in
    its window.
    """
    if h < 1:
        raise ValueError(f"h is {h}; horizons start at 1")
    # Checked here too, not only in recent_level: with no origins it would never be called.
    if window < 1:
        raise ValueError(f"window is {window}; the trailing mean needs at least 1 week")
    forecasts = []
    for origin in origins:
        forecast = recent_level(series, origin, weeks=window)
        if np.isnan(forecast):
            raise ValueError(f"h {h}, origin {origin}: no week with data in the {window} "
                             f"weeks up to the origin, so no trailing mean")
        forecasts.append({"origin_week": origin, "h": h, "target_week": origin + h,
                          "forecast": forecast})
    return pd.DataFrame(forecasts, columns=FORECAST_COLUMNS)


def topup_forecasts(new_cash: pd.DataFrame, refunds: pd.DataFrame, unidentified: pd.DataFrame,
                    levels: pd.Series, calendar: pd.DataFrame, origins: list[int],
                    horizons: int, method: str, window: int,
                    with_level: bool) -> pd.DataFrame:
    """The top-up forecasts of every company, made at each of `origins`, for h = 1 to
    `horizons`, deflated (TRY of week 0).

    The inputs are the side tables as `side_tables.py` makes them, in nominal TRY:
    `new_cash` (A, per origin week, h and target week), `refunds` (R, per week, negative) and
    `unidentified` (G, per week); `levels` the price levels (`prices.published_levels`), which
    deflate each amount by the week its cash came. Each company is forecast on its own series.

    `method` "mean": the trailing mean of the last `window` target weeks (U12.1).
    `method` "ridge": `deflated_forecasts` on the calendar features of the target week
    (`calendar`) and, if `with_level`, the recent level, refitted at every origin on the target
    weeks up to it. R goes in as -R, positive, since the ridge falls back to the recent level
    for any forecast not above 0; its forecast is turned back to negative.

    Returns one row per company, origin and h: company_code, origin_week, h, target_week, the
    forecasts new_customer_cash, refunds and unidentified_cash, topup (their sum; refunds are
    negative) and fallback (True if any of the three is the ridge's fallback to the recent
    level, not counting a series of only zeros up to the origin, whose forecast is 0 either
    way; always False for "mean"). Raises ValueError for an unknown method, `horizons` below
    1, a `window` below 1 with "mean", side tables with no company (with at least one
    origin), or (from the methods) an origin without the data its forecast needs.
    """
    if method not in METHODS:
        raise ValueError(f"method is {method!r}; it must be one of {METHODS}")
    if horizons < 1:
        raise ValueError(f"horizons is {horizons}; horizons start at 1")
    # Checked here too, not only in trailing_mean_forecasts: with no origins it is never called.
    if method == "mean" and window < 1:
        raise ValueError(f"window is {window}; the trailing mean needs at least 1 week")
    new_cash = new_cash.assign(deflated=deflate(new_cash["new_customer_cash"],
                                                new_cash["target_week"], levels))
    refunds = refunds.assign(deflated=-deflate(refunds["refunds"], refunds["week"], levels))
    unidentified = unidentified.assign(deflated=deflate(unidentified["unidentified_cash"],
                                                        unidentified["week"], levels))

    def series_of(table: pd.DataFrame, company: str, week_column: str) -> pd.Series:
        """One company's deflated series, indexed by the week its cash came."""
        own = table[table["company_code"] == company]
        return own.set_index(week_column)["deflated"].sort_index()

    def forecast(series: pd.Series, h: int) -> pd.DataFrame:
        """The forecasts of `series` for horizon `h` at every origin, with a fallback column."""
        if method == "mean":
            return trailing_mean_forecasts(series, h, origins, window).assign(fallback=False)
        parts = []
        for origin in origins:
            part = deflated_forecasts(series, calendar, h, origin, [origin], with_level)
            # A series of only zeros up to the origin is forecast 0 either way: its fallback to
            # the level (0) is the right answer, not a warning, so it is not flagged.
            seen = series[series.index <= origin]
            part["fallback"] &= bool((seen != 0).any())
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    if not origins:
        return pd.DataFrame(columns=TOPUP_COLUMNS)
    companies = sorted({*new_cash["company_code"], *refunds["company_code"],
                        *unidentified["company_code"]})
    if not companies:
        raise ValueError("the side tables hold no company, so there is nothing to forecast")
    parts = []
    for company in companies:
        own_refunds = series_of(refunds, company, "week")
        own_unidentified = series_of(unidentified, company, "week")
        for h in range(1, horizons + 1):
            own_new_cash = series_of(new_cash[new_cash["h"] == h], company, "target_week")
            a = forecast(own_new_cash, h)
            r = forecast(own_refunds, h)
            g = forecast(own_unidentified, h)
            part = a[["origin_week", "h", "target_week"]].assign(
                company_code=company,
                new_customer_cash=a["forecast"],
                refunds=-r["forecast"],
                unidentified_cash=g["forecast"],
                fallback=a["fallback"] | r["fallback"] | g["fallback"])
            parts.append(part)
    found = pd.concat(parts, ignore_index=True)
    found["topup"] = found["new_customer_cash"] + found["refunds"] + found["unidentified_cash"]
    found = found.sort_values(["company_code", "origin_week", "h"], ignore_index=True)
    return found[TOPUP_COLUMNS]


def topup_actuals(new_cash: pd.DataFrame, refunds: pd.DataFrame, unidentified: pd.DataFrame,
                  levels: pd.Series, origins: list[int], horizons: int) -> pd.DataFrame:
    """What the top-ups really were (hindsight), in the shape of `topup_forecasts`: per
    company, origin and h, the cash of A, R and G in the target week (origin + h), deflated by
    that week's price level, and their sum. The side tables and `levels` as for
    `topup_forecasts`.

    A row is left out when its target week has no data in one of the tables (missing, or
    after the last week): no data is not 0. Returns the columns of TOPUP_COLUMNS without
    fallback, sorted by company, origin and h. Raises ValueError if `horizons` is below 1.
    """
    if horizons < 1:
        raise ValueError(f"horizons is {horizons}; horizons start at 1")
    keys = ["company_code", "origin_week", "h", "target_week"]
    wanted = new_cash["origin_week"].isin(origins) & (new_cash["h"] <= horizons)
    found = new_cash.loc[wanted, keys + ["new_customer_cash"]]
    # R and G do not depend on the origin: each target week's value, joined by company and
    # week; a target week a table has no row for drops out here.
    for table, column in ((refunds, "refunds"), (unidentified, "unidentified_cash")):
        by_week = table[["company_code", "week", column]].rename(columns={"week": "target_week"})
        found = found.merge(by_week, on=["company_code", "target_week"], how="inner",
                            validate="many_to_one")
    for column in ("new_customer_cash", "refunds", "unidentified_cash"):
        found[column] = deflate(found[column], found["target_week"], levels)
    found["topup"] = found["new_customer_cash"] + found["refunds"] + found["unidentified_cash"]
    found = found.sort_values(["company_code", "origin_week", "h"], ignore_index=True)
    return found[TOPUP_COLUMNS[:-1]]


# The series `score_topups` scores, in order: the three top-ups and their sum.
SCORED_SERIES = ["new_customer_cash", "refunds", "unidentified_cash", "topup"]
# The columns of the table `score_topups` returns, after company_code (if by company).
SCORE_COLUMNS = ["h", "series", "mean_abs_error", "bias", "forecasts"]


def score_topups(forecasts: pd.DataFrame, actuals: pd.DataFrame,
                 by_company: bool) -> pd.DataFrame:
    """How far `forecasts` (from `topup_forecasts`) were from `actuals` (from
    `topup_actuals`), per h and series (SCORED_SERIES), and per company if `by_company`;
    otherwise the companies are first added up, so each origin and h is scored on its
    consolidated top-up.

    Only (company, origin, h) found in both tables are scored; `forecasts` counts them.
    The scores, in TRY of week 0 like the inputs: mean_abs_error, the mean of |forecast -
    actual| over the origins, and bias, the mean of forecast - actual (above 0: forecasts too
    high). Not MAPE: refunds are negative and A is often 0. Returns one row per (company,) h
    and series, sorted by them. Raises ValueError if a (company, origin, h) appears twice in
    either table, either holds a NaN or infinite value, or no row is found in both.
    """
    keys = ["company_code", "origin_week", "h"]
    for name, table in (("forecasts", forecasts), ("actuals", actuals)):
        if table.duplicated(keys).any():
            raise ValueError(f"{name} holds a (company, origin, h) more than once")
        if not np.isfinite(table[SCORED_SERIES].astype("float64")).all().all():
            raise ValueError(f"{name} holds a NaN or infinite value")
    both = forecasts[keys + SCORED_SERIES].merge(actuals[keys + SCORED_SERIES], on=keys,
                                                 suffixes=("_forecast", "_actual"))
    if both.empty:
        raise ValueError("no (company, origin, h) is in both the forecasts and the actuals")
    if not by_company:
        both = both.drop(columns="company_code").groupby(["origin_week", "h"],
                                                         as_index=False).sum()
    groups = ["company_code", "h"] if by_company else ["h"]
    scores = []
    for group, rows in both.groupby(groups):
        group = group if isinstance(group, tuple) else (group,)
        for series in SCORED_SERIES:
            error = rows[f"{series}_forecast"] - rows[f"{series}_actual"]
            scores.append({**dict(zip(groups, group)), "series": series,
                           "mean_abs_error": float(error.abs().mean()),
                           "bias": float(error.mean()), "forecasts": len(rows)})
    columns = (["company_code"] if by_company else []) + SCORE_COLUMNS
    return pd.DataFrame(scores)[columns]
