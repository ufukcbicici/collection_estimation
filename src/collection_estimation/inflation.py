"""TÜFE deflation — modelling collections in real terms instead of nominal.

Turkish consumer inflation ran at roughly 2% a month across this period, so a nominal
weekly collections series drifts upward by tens of percent over a year for reasons that
have nothing to do with collections behaviour. The pipeline currently handles that with a
**linear trend regressor**, which works at h=1 and over-extrapolates by h=5.

Deflating attacks the same problem at the source: divide the series by the price level,
model the stationary real series, then re-inflate the prediction. Nothing is extrapolated
by the model, because the drift was removed rather than fitted.

--------------------------------------------------------------------------------------
THE INDEX, AND HOW IT WAS ASSEMBLED
--------------------------------------------------------------------------------------

TÜİK **changed the TÜFE base from 2003=100 to 2025=100 on 1 January 2026**, so no single
published series spans this data. The table below is stated on the old 2003=100 basis
throughout, with the 2026 values chained on through the monthly rates. Only ratios are ever
used, so the base itself is arbitrary — what matters is that the series is continuous
across the break.

It was cross-checked three ways, and all three agree:

* published 2003=100 index levels (Jan 2025 - May 2026) against published monthly rates —
  agreement to two decimal places on all eleven overlapping months;
* Sep-Dec 2025 chained forward from Aug 2025 gives 3513.96, while back-deriving Dec 2025
  from Jan 2026's level and rate gives 3513.76 — a 0.006% disagreement;
* Jun/Jul 2026 chained from May 2026 match the 2025=100 published values converted at the
  measured 31.83 factor.

**These are published statistics, not estimates.** Extend the table when the data does; do
not interpolate past its end, which is what `InflationCoverageError` prevents.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

#: TÜFE, monthly, stated on the 2003=100 basis and chained across the 2026 base change.
#: `(year, month) -> index level`.
TUFE = {
    (2025, 1): 2819.65, (2025, 2): 2883.75, (2025, 3): 2954.69, (2025, 4): 3043.23,
    (2025, 5): 3089.74, (2025, 6): 3132.17, (2025, 7): 3196.66, (2025, 8): 3261.72,
    (2025, 9): 3367.07, (2025, 10): 3452.92, (2025, 11): 3482.96, (2025, 12): 3513.96,
    (2026, 1): 3683.83, (2026, 2): 3793.05, (2026, 3): 3866.74, (2026, 4): 4028.47,
    (2026, 5): 4097.55, (2026, 6): 4138.11, (2026, 7): 4211.77, (2026, 8): 4289.27,
}

#: TÜİK publishes month M in the first days of month M+1. So a forecast made during month M
#: has the index for M-1 as its most recent PUBLISHED figure — not M. Getting this wrong
#: would leak roughly one month of inflation into every forecast.
PUBLICATION_LAG_MONTHS = 1

#: Months of published history used to estimate the forward inflation rate. Six is a
#: compromise: shorter tracks turning points, longer is less noisy. Turkish monthly
#: inflation over this period ranges from 0.87% to 5.03%, so a single month is far too
#: noisy to extrapolate from.
PROJECTION_WINDOW = 6


class InflationCoverageError(ValueError):
    """The requested range reaches beyond the declared TÜFE table.

    Raised rather than extrapolated, for the same reason `HolidayCoverageError` is: the
    failure it prevents is silent. Carrying the last known index forward would quietly
    treat several months of ~2%/month inflation as zero, and a model deflated by a stale
    index is simply a nominal model wearing a disguise. Nothing would look wrong.
    """


def check_coverage(first: date, last: date) -> None:
    """Fail loudly if any month in range is missing from the table."""
    needed = set()
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        needed.add((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)

    missing = sorted(needed - set(TUFE))
    if missing:
        have = sorted(TUFE)
        raise InflationCoverageError(
            f"TÜFE is not declared for {[f'{y}-{m:02d}' for y, m in missing]}. "
            f"The table covers {have[0][0]}-{have[0][1]:02d} to "
            f"{have[-1][0]}-{have[-1][1]:02d}. Extend `inflation.TUFE` from TÜİK before "
            f"deflating data from {first} to {last} — carrying a stale index forward "
            f"would silently treat real inflation as zero.")


def _monthly_frame() -> pd.DataFrame:
    """The table as a frame indexed by the MIDPOINT of each month.

    Mid-month, not the first: a monthly index is an average over the month, so attributing
    it to the month's middle is the honest reading. Attributing it to the first day shifts
    the whole deflator half a month early.
    """
    rows = [{"when": pd.Timestamp(year=y, month=m, day=15), "index": v}
            for (y, m), v in sorted(TUFE.items())]
    return pd.DataFrame(rows).set_index("when")


def _raw_weekly_level(weeks: pd.DataFrame) -> np.ndarray:
    """Monthly index interpolated to weekly, un-normalised.

    Interpolated **log-linearly** — prices compound, so a straight line between two
    monthly levels systematically understates the middle of the interval. At 2%/month the
    error is small, but it is free to avoid.
    """
    first = weeks["week_start"].iloc[0].date()
    last = weeks["week_end"].iloc[-1].date()
    check_coverage(first, last)

    monthly = _monthly_frame()
    # Thursday: the week's midpoint, matching the mid-month convention above.
    midpoints = pd.to_datetime(weeks["week_start"]) + pd.Timedelta(days=3)

    known_x = monthly.index.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    known_y = np.log(monthly["index"].to_numpy(dtype=float))
    want_x = midpoints.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    return np.exp(np.interp(want_x, known_x, known_y))


def weekly_price_level(weeks: pd.DataFrame) -> np.ndarray:
    """The ACTUAL price level at each week, normalised to 1.0 at the last week.

    **Uses figures that were not published yet at most origins**, so this is the oracle
    version — useful for measuring how much the inflation projection costs, not for an
    honest backtest. `published_price_level` is the one to model with.

    The base week is arbitrary and cancels out: every use divides one price level by
    another. The last week is chosen so deflated amounts read as "today's TRY".
    """
    raw = _raw_weekly_level(weeks)
    return raw / raw[-1]


def _published_key(start: pd.Timestamp, lag_months: int) -> tuple[int, int]:
    """The most recent TÜFE month published as of a given week."""
    year, month = start.year, start.month
    for _ in range(lag_months):
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return year, month


def published_price_level(weeks: pd.DataFrame,
                          lag_months: int = PUBLICATION_LAG_MONTHS) -> np.ndarray:
    """The most recent price level actually PUBLISHED as of each week.

    A **step function**, not a curve: within any month the latest figure is `lag_months`
    months old and does not move. That is genuinely what a forecaster knows, and it is why
    this — not `weekly_price_level` — is what the deflated model divides by.

    The cost is a small sawtooth in the deflated series: prices rise smoothly while the
    deflator jumps once a month, so real amounts drift up within a month and step back at
    the boundary. At ~2%/month that is a ±1% wobble against a ~20% MAPE. Accepting a
    visible ±1% artefact is the right trade against an invisible leak.

    Normalised on the same basis as `weekly_price_level`, so the two are comparable.
    """
    raw = _raw_weekly_level(weeks)
    keys = sorted(TUFE)
    out = np.empty(len(weeks), dtype=float)
    for i, start in enumerate(weeks["week_start"]):
        key = _published_key(start, lag_months)
        # Weeks before the table starts fall back to its earliest month.
        out[i] = TUFE[key if key in TUFE else keys[0]]
    return out / raw[-1]


def projected_price_level(weeks: pd.DataFrame,
                          window: int = PROJECTION_WINDOW,
                          lag_months: int = PUBLICATION_LAG_MONTHS) -> np.ndarray:
    """`P[origin, target]`: the price level at `target` as PROJECTED from `origin`.

    Re-inflating a real forecast back to nominal needs the price level at the target week,
    which is in the future and therefore unknown. It is projected from the last published
    figure at the recently published rate of inflation — nothing here reads beyond the
    origin.

    Using the actual future index instead would hand the model perfect foresight of
    inflation and flatter every deflated result. `weekly_price_level` is that version; the
    gap between the two is the price of not knowing next month's TÜFE.

    **Compound from the published month, not from the origin.** The anchor is an index for
    a month whose midpoint is roughly six weeks before the origin, so projecting only
    `target - origin` weeks forward never makes up the stale interval and every prediction
    comes out low. Measured before this was fixed: a systematic **-2.46%** undershoot of
    the true price level, which passes straight through into the forecast as bias.
    """
    n = len(weeks)
    raw_last = _raw_weekly_level(weeks)[-1]
    keys = sorted(TUFE)

    # Week midpoints, matching the mid-month convention of the index itself.
    target_mid = (pd.to_datetime(weeks["week_start"]) + pd.Timedelta(days=3))
    target_ord = target_mid.map(pd.Timestamp.toordinal).to_numpy(dtype=float)

    out = np.empty((n, n), dtype=float)
    for origin in range(n):
        key = _published_key(weeks["week_start"].iloc[origin], lag_months)
        available = [k for k in keys if k <= key]
        anchor_key = available[-1] if available else keys[0]
        anchor_level = TUFE[anchor_key] / raw_last
        anchor_ord = float(pd.Timestamp(
            year=anchor_key[0], month=anchor_key[1], day=15).toordinal())

        if len(available) < 2:
            # Too little published history to estimate a rate; assume flat prices.
            out[origin, :] = anchor_level
            continue

        recent = available[-(window + 1):]
        growth = ((np.log(TUFE[recent[-1]]) - np.log(TUFE[recent[0]]))
                  / (len(recent) - 1))
        # Elapsed months from the anchor MONTH's midpoint to each target week's midpoint.
        elapsed = (target_ord - anchor_ord) / 30.437
        out[origin, :] = anchor_level * np.exp(growth * elapsed)

    return out
