"""The weekly target series, the metrics, and the baseline models.

This is the bar. Section 11.2 of the design quotes a table produced by the corpus
generator's own benchmark; reproducing it here is a check on everything built so far —
the corpus copy, the ingest, the week spine, the aggregation and the calendar block all
have to line up for the numbers to match.

Nothing in this module knows about individual customers. It exists to be beaten.
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
    cal = calendar.set_index("week").loc[series.index, list(CALENDAR_COLUMNS)]
    cal = cal.to_numpy(dtype=float)

    records = []
    for origin in range(warmup, n - 1):
        history = y[:origin + 1]                 # everything up to and including origin

        # Design matrix for the calendar model: intercept, the level four weeks back from
        # each row, and that row's own calendar. Built from history only.
        rows, targets = [], []
        for t in range(4, origin + 1):
            rows.append(np.concatenate(([1.0, y[t - 4:t].mean()], cal[t])))
            targets.append(y[t])
        design = np.asarray(rows)
        fitted = np.asarray(targets)

        for h in horizons:
            target_index = origin + h
            if target_index >= n:
                continue
            actual = y[target_index]

            predictions = {
                "naive": history[-1],
                "ma4": history[-4:].mean(),
                "ma8": history[-8:].mean(),
                "calendar_aware": _fit_calendar(
                    fitted, design,
                    np.concatenate(([1.0, history[-4:].mean()], cal[target_index]))),
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
        out.append({
            "model": model, "horizon": int(horizon), "n": len(group),
            "mape": round(mape(actual, predicted), 2),
            "wape": round(wape(actual, predicted), 2),
            "bias": round(bias(actual, predicted), 0),
        })
    return pd.DataFrame(out).sort_values(["model", "horizon"]).reset_index(drop=True)


def mape_table(evaluations: pd.DataFrame) -> pd.DataFrame:
    """The section 11.2 layout: models down, horizons across, MAPE in the cells."""
    scored = score(evaluations)
    table = scored.pivot(index="model", columns="horizon", values="mape")
    order = [m for m in ("naive", "ma4", "ma8", "calendar_aware") if m in table.index]
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
