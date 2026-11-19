"""The series block (U19): where each company's weekly cash as a whole is heading, handed to
its customers' rows as features. Everything is as known at the end of the origin week t.

The customer model sees each customer, but not the company's week as a whole. The
`deflated_trend` reference does (a ridge on the company's deflated weekly cash, with the
calendar and a straight-line trend; `benchmark.deflated_forecasts`), and it forecasts the
totals best at h1 and h2. So at each origin t it is fitted on the weeks up to t only, per
company and for all companies together, and its forecast of the target week t + h goes to
every row of that company, origin and h.

Example: origin 40, h 1, company TR02. Fitted on TR02's weeks up to 40, the ridge forecasts
52 M TRY (of week 0) for week 41. TR02's mean over the 13 weeks up to 40 is 40 M, so the
ratio is 52 / 40 = 1.30: a strong week ahead for the whole company. Every TR02 row at origin
40 and h 1 gets trend_forecast 52 M and trend_ratio 1.30, and the same two for the
consolidated series (all_trend_forecast, all_trend_ratio).

  - The ratio says "a strong or a weak week, against the company's usual", whatever the
    company's size; the forecast itself is kept beside it, since trees cannot divide.
  - Where the reference would fall back to the recent level (fewer than
    `benchmark.MIN_TRAINING_WEEKS` training weeks, or a prediction not above 0), or there is
    no week with data just before t, the forecast and its ratio are NaN: the early origins
    carry no trend at all rather than a level dressed up as one.
  - The forecasts, not the ridge's coefficients: those are one value per company and origin,
    shared by all of its customers, and the forecast already holds what they say for the
    target week.
"""
import numpy as np
import pandas as pd

from .benchmark import deflated_forecasts, recent_level
from .evaluate import REFERENCES, real_weekly_cash
from .forecast import CONSOLIDATED

# The keys of a row of `series_features`, then its features.
SERIES_KEYS = ["company_code", "origin_week", "h", "target_week"]
SERIES_COLUMNS = ["trend_forecast", "trend_ratio", "all_trend_forecast", "all_trend_ratio"]

# The reference whose forecasts are the features, and its two switches: no recent level, a
# straight-line trend.
TREND_REFERENCE = "deflated_trend"
WITH_LEVEL, WITH_TREND = REFERENCES[TREND_REFERENCE]


def trend_forecast_at(series: pd.Series, calendar: pd.DataFrame, origin: int, h: int) -> float:
    """The `deflated_trend` forecast of week `origin` + `h`, fitted on the weeks of `series`
    (deflated cash per week with data, indexed by week) up to `origin` only.

    NaN where the reference would not use its ridge: no week with data in the weeks of the
    recent level up to `origin`, too few training weeks, or a prediction not above 0.
    """
    if np.isnan(recent_level(series, origin)):
        return float("nan")
    made = deflated_forecasts(series, calendar, h, train_until=origin, origins=[origin],
                              with_level=WITH_LEVEL, with_trend=WITH_TREND).iloc[0]
    if made["fallback"]:
        return float("nan")
    return float(made["forecast"])


def trend_ratio(forecast: float, series: pd.Series, origin: int, mean_weeks: int) -> float:
    """`forecast` divided by the mean of `series` over the `mean_weeks` weeks up to `origin`
    (weeks with data only). NaN if the forecast is NaN, or that mean is NaN or not above 0."""
    mean = recent_level(series, origin, mean_weeks)
    if np.isnan(forecast) or np.isnan(mean) or mean <= 0:
        return float("nan")
    return forecast / mean


def series_features(weekly: pd.DataFrame, weeks: pd.DataFrame, levels: pd.Series,
                    calendar: pd.DataFrame, companies: list[str], origins: list[int],
                    horizons: int, mean_weeks: int) -> pd.DataFrame:
    """The series features (see the module docstring) at every origin in `origins`, for
    h = 1 to `horizons`, per company in `companies`.

    `weekly`, `weeks` and `levels` give the deflated weekly cash (`evaluate.real_weekly_cash`,
    the series the references fit on); `calendar` the calendar features of every week, as
    for the references. `mean_weeks` is the window of the ratios' mean.

    Returns one row per company, origin and h: `SERIES_KEYS` then `SERIES_COLUMNS`, sorted
    by origin, h and company. Each row reads only the weeks up to its origin.

    Raises ValueError if `origins` is empty, `horizons` or `mean_weeks` is below 1, and as
    `real_weekly_cash` and `deflated_forecasts` do.
    """
    origins = sorted(set(origins))
    if not origins:
        raise ValueError("there are no origins to make the series features at")
    if horizons < 1:
        raise ValueError(f"horizons is {horizons}; horizons start at 1")
    if mean_weeks < 1:
        raise ValueError(f"mean_weeks is {mean_weeks}; the ratio's mean needs at least 1 week")
    real = real_weekly_cash(weekly, weeks, levels, companies)
    consolidated = real[CONSOLIDATED]

    # About 60 origins x 5 h x (5 companies + all of them): a few thousand small ridge fits.
    # With no recent level among its inputs, the ridge is the same for every h at one origin,
    # but fitting it once per h keeps this the reference's own code; it takes seconds.
    rows = []
    for origin in origins:
        for h in range(1, horizons + 1):
            all_forecast = trend_forecast_at(consolidated, calendar, origin, h)
            all_ratio = trend_ratio(all_forecast, consolidated, origin, mean_weeks)
            for company in sorted(set(companies)):
                forecast = trend_forecast_at(real[company], calendar, origin, h)
                rows.append({
                    "company_code": company,
                    "origin_week": origin,
                    "h": h,
                    "target_week": origin + h,
                    "trend_forecast": forecast,
                    "trend_ratio": trend_ratio(forecast, real[company], origin, mean_weeks),
                    "all_trend_forecast": all_forecast,
                    "all_trend_ratio": all_ratio,
                })
    return pd.DataFrame(rows, columns=SERIES_KEYS + SERIES_COLUMNS)
