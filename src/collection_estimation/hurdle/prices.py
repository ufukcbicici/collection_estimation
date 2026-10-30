"""The price level by week number: the bridge between our weeks and the copied `inflation.py`.

`inflation.py` works on a table of weeks with `week_start` and `week_end` columns; this module
builds that table from our week numbers, adds the guard the copied module lacks (it quietly
falls back to its earliest month for a week whose published month is not in the table), and
deflates amounts by the price level published in their own week. A forecast is re-inflated
with `projected_levels`: the same published level, as known at the forecast week.
"""
import math
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
    normalised to 1.0 at week 0, so deflated amounts read as TRY of week 0.

    Week 0, not the last week: scaling to the last week would multiply every amount by a
    number not known until then. It is one constant and carries no information, but week 0
    is known from the start, so nothing from the future appears at all. Raises ValueError
    unless the weeks include week 0, and (see `check_price_coverage`) if the TÜFE table does
    not cover them.
    """
    if not first_week <= 0 <= last_week:
        raise ValueError(f"weeks {first_week} to {last_week} do not include week 0, the week "
                         f"the price levels are normalised to")
    weeks = price_weeks(week_zero, first_week, last_week)
    check_price_coverage(weeks)
    # The copied module normalises to the last week of the range; rescale to week 0.
    levels = pd.Series(inflation.published_price_level(weeks), index=weeks["week"].to_numpy(),
                       name="price_level")
    return levels / levels.loc[0]


def published_month(week_start: date,
                    lag_months: int = inflation.PUBLICATION_LAG_MONTHS) -> tuple[int, int]:
    """The TÜFE month published as of a week starting `week_start`, as (year, month):
    `lag_months` before the month of the week's Monday (a week in July has June's)."""
    months = week_start.year * 12 + (week_start.month - 1) - lag_months
    return months // 12, months % 12 + 1


def _month_number(month: tuple[int, int]) -> int:
    """(year, month) as a count of months, so two months can be subtracted."""
    return month[0] * 12 + month[1] - 1


def projected_levels(week_zero: date, origin: int, target_weeks: list[int],
                     window: int = inflation.PROJECTION_WINDOW,
                     lag_months: int = inflation.PUBLICATION_LAG_MONTHS) -> pd.Series:
    """The price level a forecast made at the end of week `origin` re-inflates each target
    week with: the TÜFE that WILL BE published as of the target week, as known at the origin.
    Indexed by target week, normalised to 1.0 at week 0 like `published_levels`.

    The labels were deflated by the level published as of their own week, so the forecast
    is re-inflated by the same figure:

      published by the origin   the figure itself, from the table (exact; for a target week
                                not after the origin this equals `published_levels`)
      not published yet         the last figure published by the origin, grown at its mean
                                monthly rate over the last `window` months, for as many
                                months as the target's figure lies past it

    Reads no TÜFE month published after the origin, except, for an origin before week 0,
    week 0's own figure, which every level is divided by (a constant scale; the labels are
    divided by it too). Raises InflationCoverageError if the table lacks the month published
    by the origin or by week 0, or a month to look up, and ValueError for no target weeks, a
    `week_zero` that is not a Monday (a week's TÜFE month is its Monday's month, as when the
    labels were deflated), a window below 1, a negative lag (it would read months not yet
    published), or a projection with fewer than two months to take the rate from.
    """
    if not target_weeks:
        raise ValueError("no target weeks to project the price level for")
    if week_zero.weekday() != 0:
        raise ValueError(f"week_zero {week_zero} is a {week_zero:%A}, not a Monday; weeks run "
                         f"Monday to Sunday")
    if window < 1:
        raise ValueError(f"window is {window}; the rate of inflation needs at least 1 month")
    if lag_months < 0:
        raise ValueError(f"lag_months is {lag_months}; a negative lag reads TÜFE months not "
                         f"yet published")

    def week_start(week: int) -> date:
        return week_zero + timedelta(days=7 * week)

    known = published_month(week_start(origin), lag_months)
    base = published_month(week_start(0), lag_months)
    for week, month in ((origin, known), (0, base)):
        if month not in inflation.TUFE:
            raise inflation.InflationCoverageError(
                f"week {week} (starting {week_start(week)}) needs the TÜFE for "
                f"{month[0]}-{month[1]:02d}, published by then, which is not in the table; "
                f"extend `inflation.TUFE`")

    growth = None   # the mean monthly log growth, worked out only if a target needs it
    levels = []
    for week in target_weeks:
        needed = published_month(week_start(week), lag_months)
        if needed <= known:
            if needed not in inflation.TUFE:
                raise inflation.InflationCoverageError(
                    f"target week {week} needs the TÜFE for {needed[0]}-{needed[1]:02d}, "
                    f"which is not in the table; extend `inflation.TUFE`")
            level = inflation.TUFE[needed]
        else:
            if growth is None:
                history = [month for month in sorted(inflation.TUFE)
                           if month <= known][-(window + 1):]
                if len(history) < 2:
                    raise ValueError(f"origin week {origin}: the table has fewer than two "
                                     f"months up to {known[0]}-{known[1]:02d} to take the "
                                     f"rate of inflation from")
                growth = ((math.log(inflation.TUFE[history[-1]])
                           - math.log(inflation.TUFE[history[0]]))
                          / (_month_number(history[-1]) - _month_number(history[0])))
            months_ahead = _month_number(needed) - _month_number(known)
            level = inflation.TUFE[known] * math.exp(growth * months_ahead)
        levels.append(level / inflation.TUFE[base])
    return pd.Series(levels, index=list(target_weeks), name="price_level")


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
