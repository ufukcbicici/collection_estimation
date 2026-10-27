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
"""
import numpy as np
import pandas as pd

from .benchmark import recent_level

# The columns of the table `trailing_mean_forecasts` returns, in order.
FORECAST_COLUMNS = ["origin_week", "h", "target_week", "forecast"]


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
