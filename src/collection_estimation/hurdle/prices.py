"""The price level by week number: the bridge between our weeks and the copied `inflation.py`.

`inflation.py` works on a table of weeks with `week_start` and `week_end` columns; this module
builds that table from our week numbers, adds the guard the copied module lacks (it quietly
falls back to its earliest month for a week whose published month is not in the table), and
deflates amounts by the price level published in their own week.
"""
from datetime import date, timedelta

import pandas as pd

from . import inflation


def price_weeks(week_zero: date, first_week: int, last_week: int) -> pd.DataFrame:
    """The weeks `first_week` ... `last_week` as `inflation.py` expects them: columns week,
    week_start (the Monday) and week_end (the Sunday), as Timestamps."""
    if last_week < first_week:
        raise ValueError(f"last_week {last_week} is before first_week {first_week}")
    weeks = pd.DataFrame({"week": range(first_week, last_week + 1)})
    weeks["week_start"] = [pd.Timestamp(week_zero + timedelta(days=7 * week))
                           for week in weeks["week"]]
    weeks["week_end"] = weeks["week_start"] + pd.Timedelta(days=6)
    weeks["week"] = weeks["week"].astype("int64")
    return weeks


def check_price_coverage(weeks: pd.DataFrame,
                         lag_months: int = inflation.PUBLICATION_LAG_MONTHS) -> None:
    """Raise ValueError unless the TÜFE table has every month these weeks need.

    That is every month from the first week to the last (as `inflation.check_coverage`
    checks), and also the month PUBLISHED as of the first week, `lag_months` earlier: the
    copied module would otherwise use its earliest month for it without saying so.
    """
    first_start = weeks["week_start"].iloc[0]
    inflation.check_coverage(first_start.date(), weeks["week_end"].iloc[-1].date())
    year, month = first_start.year, first_start.month - lag_months
    while month < 1:
        year, month = year - 1, month + 12
    if (year, month) not in inflation.TUFE:
        raise inflation.InflationCoverageError(
            f"the week starting {first_start.date()} needs the TÜFE published by then, for "
            f"{year}-{month:02d}, which is not in the table; extend `inflation.TUFE`")


def published_levels(week_zero: date, first_week: int, last_week: int) -> pd.Series:
    """The price level PUBLISHED as of each week (a monthly step), indexed by week number and
    normalised to 1.0 at the price level of `last_week`, so deflated amounts read as TRY of
    the last week. Raises if the TÜFE table does not cover the weeks (see
    `check_price_coverage`)."""
    weeks = price_weeks(week_zero, first_week, last_week)
    check_price_coverage(weeks)
    levels = inflation.published_price_level(weeks)
    return pd.Series(levels, index=weeks["week"].to_numpy(), name="price_level")


def deflate(amounts: pd.Series, weeks: pd.Series, levels: pd.Series) -> pd.Series:
    """Each amount divided by the price level of its own week (`levels` from
    `published_levels`). `amounts` and `weeks` must have the same index (e.g. two columns of
    one table). Raises ValueError if they do not, or for a week that has no price level."""
    # pandas divides by matching index labels: different indexes would give NaN for every
    # row without any error.
    if not amounts.index.equals(weeks.index):
        raise ValueError("amounts and weeks must have the same index (e.g. two columns of one "
                         "table)")
    level = weeks.map(levels)
    if level.isna().any():
        missing = sorted(set(weeks[level.isna()]))
        raise ValueError(f"no price level for weeks {missing[:10]}")
    return amounts / level
