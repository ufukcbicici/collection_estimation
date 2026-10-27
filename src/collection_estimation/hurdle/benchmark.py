"""The `deflated` benchmark: a ridge regression on a weekly cash total, in real terms.

The hurdle model is compared with it. Rebuilt from the old `weekly_models.make_deflated`,
which this package never imports; the method is the same:

  - the cash is deflated (TRY of week 0), so the inflation drift is divided out and no trend
    is needed;
  - one model per horizon h: the target week's cash from the target week's calendar features
    and, optionally, the recent level (the mean cash of the LEVEL_WEEKS weeks ending at the
    origin, i.e. h weeks before the target: the level a forecast made at the origin knows);
  - ridge regression on standardised inputs, its penalty chosen from ALPHAS by RidgeCV (about
    40 training weeks against 5 or 6 inputs is too few for plain least squares);
  - a fallback to the recent level when there are fewer than MIN_TRAINING_WEEKS training weeks
    or the prediction is not above 0.

What it forecasts: the same totals as the hurdle model. For horizon h the series is y_h(w),
the cash in week w from the customers known at w - h (`cash_totals` of the panel rows with
that h), so both models are scored on exactly the same numbers. Without the new customers'
cash, which comes in U12 (top-ups). Weeks without data are not in the series: they are left
out of the training rows and of the level, never counted as 0.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The weeks the recent level averages: the origin and the 3 weeks before it.
LEVEL_WEEKS = 4
# The ridge penalties RidgeCV chooses from (by leave-one-out error on the training weeks).
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
# Fewer training weeks than this: no regression, the forecast is the recent level.
MIN_TRAINING_WEEKS = 12


def cash_totals(rows: pd.DataFrame, values: pd.Series, by_company: bool) -> pd.DataFrame:
    """`values` (one per row of `rows`, e.g. the deflated amount or p x m) summed per origin
    week, h and target week, and per company if `by_company`. One row per total, with the
    columns origin_week, h, target_week, (company_code,) total; sorted by them.

    Raises ValueError if `values` does not have the rows of `rows` or holds a NaN or
    infinite value.
    """
    if not values.index.equals(rows.index):
        raise ValueError("values must have the rows of rows (the same index)")
    if not np.isfinite(values.astype("float64")).all():
        raise ValueError("values holds a NaN or infinite value")
    keys = ["origin_week", "h", "target_week"] + (["company_code"] if by_company else [])
    frame = rows[keys].assign(total=values.astype("float64"))
    return frame.groupby(keys, as_index=False, observed=True)["total"].sum()


def recent_level(series: pd.Series, origin: int, weeks: int = LEVEL_WEEKS) -> float:
    """The mean of `series` (cash per week, indexed by week, weeks without data absent) over
    the `weeks` weeks ending at `origin`; weeks without data are left out. NaN if none of
    them has data. Reads no week after `origin`. Raises ValueError if `weeks` is below 1."""
    if weeks < 1:
        raise ValueError(f"weeks is {weeks}; the level needs at least 1 week")
    window = series[(series.index > origin - weeks) & (series.index <= origin)]
    return float(window.mean()) if len(window) else float("nan")


def deflated_forecasts(series: pd.Series, calendar: pd.DataFrame, h: int, train_until: int,
                       origins: list[int], with_level: bool) -> pd.DataFrame:
    """The `deflated` forecast of `series` (y_h: cash per target week, deflated, indexed by
    week, weeks without data absent) for horizon `h`, made at each of `origins` (target week
    origin + h).

    The model is fitted once, on the target weeks up to `train_until`; each training week w
    is described by the calendar features of w (`calendar`: one row per week, a `week` column
    and the features) and, if `with_level`, by the recent level at its origin w - h. A week
    whose level has no data is not used. Each forecast uses the target week's calendar and
    the level at its origin (weeks up to the origin only).

    Returns one row per origin: origin_week, h, target_week, forecast and fallback (True when
    the forecast is the recent level: too few training weeks, or a prediction that is not
    above 0 or not finite). Raises ValueError if a week needed is not in `calendar`, or a
    forecast needs a recent level that has no data.
    """
    features = calendar.set_index("week")
    needed = sorted({*series.index[series.index <= train_until], *(o + h for o in origins)})
    missing = [week for week in needed if week not in features.index]
    if missing:
        raise ValueError(f"the calendar has no rows for weeks {missing}")

    def inputs(target_week: int, origin: int) -> list[float] | None:
        row = [float(value) for value in features.loc[target_week]]
        if with_level:
            level = recent_level(series, origin)
            if np.isnan(level):
                return None
            row.insert(0, level)
        return row

    x, y = [], []
    for week, cash in series[series.index <= train_until].items():
        row = inputs(week, week - h)
        if row is not None:
            x.append(row)
            y.append(cash)
    model = None
    if len(y) >= MIN_TRAINING_WEEKS:
        # RidgeCV writes into its alphas while fitting, so it gets its own array.
        model = make_pipeline(StandardScaler(),
                              RidgeCV(alphas=np.array(ALPHAS, dtype=float)))
        model.fit(np.array(x), np.array(y))

    forecasts = []
    for origin in origins:
        level = recent_level(series, origin)
        if np.isnan(level):
            raise ValueError(f"h {h}, origin {origin}: no week with data in the "
                             f"{LEVEL_WEEKS} weeks up to the origin, so no recent level")
        value = float("nan")
        if model is not None:
            value = float(model.predict(np.array([inputs(origin + h, origin)]))[0])
        fallback = not (np.isfinite(value) and value > 0)
        forecasts.append({"origin_week": origin, "h": h, "target_week": origin + h,
                          "forecast": level if fallback else value, "fallback": fallback})
    return pd.DataFrame(forecasts)


def benchmark_forecasts(totals: pd.DataFrame, calendar: pd.DataFrame, train_until: int,
                        wanted: pd.DataFrame, with_level: bool) -> pd.DataFrame:
    """`deflated_forecasts` for every row of `wanted` (origin_week, h, and company_code if
    the totals are per company), each (h, company) with its own series and model.

    `totals`: the actual cash totals from `cash_totals` (origin_week, h, target_week,
    (company_code,) total), at least up to the latest origin wanted: the series y_h are taken
    from them, and the training weeks are those up to `train_until`. Returns `wanted` with
    the columns target_week, forecast and fallback added, in the same row order.
    """
    by_company = "company_code" in wanted.columns
    groups = ["h", "company_code"] if by_company else ["h"]
    parts = []
    for group, rows in wanted.groupby(groups, observed=True):
        group = group if isinstance(group, tuple) else (group,)
        h = int(group[0])
        mine = totals["h"] == h
        if by_company:
            mine &= totals["company_code"] == group[1]
        series = totals[mine].set_index("target_week")["total"].sort_index()
        part = deflated_forecasts(series, calendar, h, train_until,
                                  sorted(set(rows["origin_week"])), with_level)
        if by_company:
            part["company_code"] = group[1]
        parts.append(part)
    keys = ["origin_week", "h"] + (["company_code"] if by_company else [])
    found = pd.concat(parts, ignore_index=True)
    return wanted.merge(found, on=keys, how="left", validate="many_to_one")
