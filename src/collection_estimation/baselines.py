"""The weekly target series, the metrics, and the baseline models.

This is the bar. Nothing here knows about individual customers; it exists to be beaten.

**Two calendar models, and the difference between them is a finding.**

`calendar_aware` regresses on the recent level plus the calendar. Its design matrix is
horizon-specific: at prediction the level is the last four observed weeks, which end `h`
weeks before the target, so each training row's level must end `h` weeks before its own
target too. An earlier version used `y[t-4:t]` for every horizon — a one-step-ahead
relationship applied to a level `h` weeks stale — which is correct at h=1 by coincidence
and wrong beyond it.

`calendar_only` drops the level entirely: intercept plus calendar. One parameter fewer and
nothing that can be mis-specified.

Measured, the level term earns nothing. The weekly series is barely autocorrelated —
+0.10 at lag 1, **−0.13 at lag 3**, +0.43 at lag 4 where the monthly cycle shows — so a
horizon-specific level coefficient is estimated from ~40 noisy observations and adds
variance rather than signal. Averaged over horizons `calendar_only` (21.0% MAPE) beats
`calendar_aware` (21.3%).

Worth knowing before building anything more elaborate: on this data the calendar carries
the signal and the recent level does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Weeks of history required before the first forecast origin.
WARMUP = 12

#: The columns of the calendar block the calendar-aware model is allowed to see. All are
#: knowable in advance for any target week — that is what makes the model honest.
CALENDAR_COLUMNS = ("n_business_days", "has_month_end", "has_quarter_end", "n_holidays")


# --------------------------------------------------------------------------------------
# The target series
# --------------------------------------------------------------------------------------

def weekly_totals(customer_weeks: pd.DataFrame, weeks: pd.DataFrame) -> pd.Series:
    """`Y[t]`: total money collected each week, indexed by the integer week.

    **The NET total**, positives plus refunds. Customer-weeks keep the two apart, so this
    is a real choice: the design's published table was computed on net totals — the
    generator's benchmark summed each day's grand total, which includes refunds — and
    section 8.1 composes the forecast as ``sum(p*a) + Nhat - Rhat``, also net. Using
    positives alone would produce numbers that cannot be compared with the table this
    module exists to reproduce.

    Weeks with no collections appear as 0 rather than being dropped, so the index stays
    contiguous and ``t + h`` keeps working.
    """
    totals = customer_weeks.groupby("week", observed=True)["amount_net"].sum()
    return totals.reindex(weeks["week"], fill_value=0.0).astype(float)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error.

    **Asymmetric, and that matters.** The denominator is the actual, so the same absolute
    error costs more in a light week than a heavy one — which quietly rewards
    under-prediction. It is the metric the design quotes, so it must be reported, but
    never on its own.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = actual != 0
    if not keep.any():
        return float("nan")
    return float(np.mean(np.abs((actual[keep] - predicted[keep]) / actual[keep])) * 100)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error: one denominator for the whole period.

    The companion to `mape`. Because the denominator is a total rather than each week's
    own value, a light week cannot dominate, and a model cannot look good by simply
    predicting low.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / denominator * 100)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean signed error. Positive means over-prediction.

    The only one of the three that keeps a direction. Equation (6.9) predicts a
    systematic term here, and both MAPE and WAPE hide its sign — a model 10% high and a
    model 10% low score identically on those.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(predicted - actual))


# --------------------------------------------------------------------------------------
# The models
# --------------------------------------------------------------------------------------

def _fit_calendar(history: np.ndarray, design: np.ndarray,
                  target_row: np.ndarray) -> float:
    """Ordinary least squares on the recent level plus the target week's calendar.

    Five parameters, nothing privileged: every input is either already observed or a fact
    about a future week that any calendar gives you.

    `numpy.linalg.lstsq` rather than scikit-learn, deliberately. It is what the corpus
    generator's own benchmark used, so a disagreement between that table and this one
    means a data or pipeline difference rather than a solver difference.
    """
    if len(history) < 12:
        return float(history[-4:].mean())
    try:
        coefficients, *_ = np.linalg.lstsq(design, history, rcond=None)
    except np.linalg.LinAlgError:
        return float(history[-4:].mean())
    value = float(target_row @ coefficients)
    return value if value > 0 else float(history[-4:].mean())


def calendar_matrix(series: pd.Series, calendar: pd.DataFrame) -> np.ndarray:
    """The calendar block as a plain array aligned to the series index."""
    block = calendar.set_index("week").loc[series.index, list(CALENDAR_COLUMNS)]
    return block.to_numpy(dtype=float)


def rolling_origin(series: pd.Series, design: np.ndarray,
                   models: dict, horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
                   warmup: int = WARMUP) -> pd.DataFrame:
    """Algorithm 4, for any set of models: one row per (model, horizon, origin).

    Each model is a callable ``fn(y, design, origin, target_index, h) -> float``. It may
    read `y` only up to and including `origin`, and any row of `design`, because the
    design matrix here holds calendar facts that are knowable in advance. Anything derived
    from collections must be passed as a separate as-of-origin structure, not smuggled in
    here — see the cross-customer block for why.

    Models are refit from scratch at every origin. Affordable at this size, and it removes
    any question of state leaking between origins.
    """
    y = series.to_numpy(dtype=float)
    n = len(y)

    records = []
    for origin in range(warmup, n - 1):
        for h in horizons:
            target_index = origin + h
            if target_index >= n:
                continue
            for name, fn in models.items():
                records.append({
                    "model": name, "horizon": h, "origin": origin,
                    "target_week": target_index, "actual": y[target_index],
                    "predicted": float(fn(y, design, origin, target_index, h)),
                })
    return pd.DataFrame(records)


def rolling_origin_baselines(
        series: pd.Series,
        calendar: pd.DataFrame,
        horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
        warmup: int = WARMUP) -> pd.DataFrame:
    """Algorithm 4 for the baselines: one row per (model, horizon, origin).

    At each origin only weeks up to and including it are used. The models are refit from
    scratch every time, which is affordable here and removes any question of state
    leaking between origins.
    """
    y = series.to_numpy(dtype=float)
    n = len(y)
    cal = calendar_matrix(series, calendar)

    records = []
    for origin in range(warmup, n - 1):
        history = y[:origin + 1]                 # everything up to and including origin

        for h in horizons:
            target_index = origin + h
            if target_index >= n:
                continue
            actual = y[target_index]

            # The design matrix is HORIZON-SPECIFIC, and this matters.
            #
            # At prediction time the level is the last four observed weeks, which end `h`
            # weeks before the target. So each training row's level must also end `h`
            # weeks before its own target — `y[t-h-3 : t-h+1]`, not `y[t-4 : t]`.
            #
            # Using `y[t-4 : t]` fits a one-step-ahead relationship and then applies it to
            # a level that is `h` weeks stale. Since weekly autocorrelation decays, the
            # coefficient comes out too large for the longer horizons and the model leans
            # on a number it should have discounted. Correct at h=1 by coincidence, wrong
            # at h=2 and beyond.
            rows, targets = [], []
            for t in range(h + 3, origin + 1):
                level = y[t - h - 3:t - h + 1].mean()
                rows.append(np.concatenate(([1.0, level], cal[t])))
                targets.append(y[t])

            # And the same model without the level at all: intercept plus calendar. One
            # parameter fewer and nothing that can be mis-specified across horizons. It
            # turns out to be as good — see the module docstring.
            plain_rows = [np.concatenate(([1.0], cal[t]))
                          for t in range(origin + 1)]

            predictions = {
                "naive": history[-1],
                "ma4": history[-4:].mean(),
                "ma8": history[-8:].mean(),
                "calendar_aware": _fit_calendar(
                    np.asarray(targets), np.asarray(rows),
                    np.concatenate(([1.0, history[-4:].mean()], cal[target_index]))),
                "calendar_only": _fit_calendar(
                    y[:origin + 1], np.asarray(plain_rows),
                    np.concatenate(([1.0], cal[target_index]))),
            }
            for name, predicted in predictions.items():
                records.append({"model": name, "horizon": h, "origin": origin,
                                "target_week": target_index,
                                "actual": actual, "predicted": float(predicted)})

    return pd.DataFrame(records)


def score(evaluations: pd.DataFrame) -> pd.DataFrame:
    """MAPE, WAPE and Bias per model and horizon.

    All three, always. See the docstrings above for why any one of them alone is
    misleading.
    """
    out = []
    for (model, horizon), group in evaluations.groupby(["model", "horizon"],
                                                       observed=True):
        actual = group["actual"].to_numpy()
        predicted = group["predicted"].to_numpy()
        # Standard error of the MAPE across origins. With ~50 evaluation points a
        # difference of under about 2 points is not distinguishable from noise, and a
        # single missing file has already moved a model by 3.7. Reporting a point
        # estimate alone is how a model gets "improved" into noise.
        ape = np.abs((actual - predicted) / np.where(actual == 0, np.nan, actual)) * 100
        ape = ape[np.isfinite(ape)]
        out.append({
            "model": model, "horizon": int(horizon), "n": len(group),
            "mape": round(mape(actual, predicted), 2),
            "mape_se": round(float(ape.std(ddof=1) / np.sqrt(len(ape))), 2)
            if len(ape) > 1 else float("nan"),
            "wape": round(wape(actual, predicted), 2),
            "bias": round(bias(actual, predicted), 0),
        })
    return pd.DataFrame(out).sort_values(["model", "horizon"]).reset_index(drop=True)


def mape_table(evaluations: pd.DataFrame) -> pd.DataFrame:
    """The section 11.2 layout: models down, horizons across, MAPE in the cells."""
    scored = score(evaluations)
    table = scored.pivot(index="model", columns="horizon", values="mape")
    order = [m for m in ("naive", "ma4", "ma8", "calendar_aware", "calendar_only") if m in table.index]
    return table.loc[order]


def weeks_affected_by_missing_files(
        series: pd.Series, weeks: pd.DataFrame,
        missing_dates: list[str]) -> list[int]:
    """Which evaluation weeks contain a day whose file is absent.

    Those weeks show a full business-day count and depressed collections, and nothing in
    the features can explain the gap. Dropping them would flatter every model by removing
    errors it genuinely makes, so they stay in — but the count is reported, because an
    unexplainable week should be visible rather than absorbed silently into the error.
    """
    if not missing_dates:
        return []
    stamps = pd.to_datetime(pd.Series(missing_dates))
    affected = set()
    for _, week in weeks.iterrows():
        inside = stamps.between(week["week_start"], week["week_end"])
        if inside.any():
            affected.add(int(week["week"]))
    return sorted(affected)
