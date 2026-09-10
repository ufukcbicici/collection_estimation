"""The recommended weekly model.

**Ridge on the calendar block + a linear trend + the recent level**, refit at every origin,
a separate fit per horizon. Six features:

    level              mean of the 4 weeks ending `h` weeks before the target
    n_business_days    0-5
    has_month_end      the week contains the last BUSINESS day of a month
    has_quarter_end    ... of March, June, September or December
    n_holidays         weekday public holidays in the week
    trend              the integer week index

Measured on the synthetic corpus at h=1: `MAPE 19.83 +/- 2.49, WAPE 19.70, bias -1.0%`.

**Chosen for the BIAS, not the accuracy.** Every candidate's MAPE sat inside one standard
error of every other, so this was never an accuracy choice. What separates it is that it is
the only competitive model not systematically 10-19% low, which for a cash forecast matters
more than a point of MAPE. Beyond h=3 an 8-week moving average is just as good.

The variants that lost — log target, Tweedie, the derived cadence regressors — are in
`parked/variants.py` with their numbers. `parked/variants.compare_weekly_models` re-runs
the whole set in seconds and is worth calling once on real data.

--------------------------------------------------------------------------------------
Why the trend term is here
--------------------------------------------------------------------------------------

Every model under-predicted by ~10% of the weekly mean. The hypothesis was a loss/metric
mismatch — squared error targets the conditional MEAN while MAPE is minimised nearer the
MEDIAN. **That hypothesis was wrong.** Fitting log(y) made both MAPE and bias worse, and a
Tweedie GLM barely moved the bias; meanwhile `ma8`, which has no loss subtlety at all, was
the least biased model in the set. If the crudest model is the least biased, the cause is
not the loss.

It was a **trend**. The weekly total grows across the corpus and a model fit on all history
regresses toward a historical mean that, in a rising series, sits below the present level.
One linear time regressor takes the bias from -10.9% to -1.0%.

    ridge          MAPE 20.04   bias -10.9%
    ridge+trend    MAPE 19.83   bias  -1.0%

Note what that does and does not buy: the MAPE change is far inside the standard error, so
the trend term is justified by BIAS, not accuracy.

**On real data this is the first thing to re-examine.** The drift in the synthetic corpus is
a generator parameter (3%/month) rather than an EY measurement. Turkish inflation makes
nominal weekly totals non-stationary in reality too, so *some* drift handling is needed —
but a linear trend is the crude answer. Deflating by a price index is the better one, and
needs a series we do not have. See `weekly_design`.

--------------------------------------------------------------------------------------
NEGATIVE RESULT: the level term earns nothing, in any parameterisation.
--------------------------------------------------------------------------------------

Measured 2026-09-09. The question was whether `_rows` should pass the four recent weeks
SEPARATELY instead of compressing them into one mean — a mean cannot weight last week above
four weeks ago, and ridge could. It can, and it gains nothing.

Compared PAIRED on the same (origin, horizon) pairs. That matters: the ~2.5 MAPE error bar
is dominated by origin-to-origin variation which is COMMON to both models and cancels on
differencing. Two independent standard errors can only say "cannot distinguish"; the paired
CI can say "equal", which is the stronger claim actually needed here.

    vs mean4                h=1     avg h=1..5    95% CI (all h)
    four separate lags     -0.05      -0.32      [-1.49, +0.85]
    last week only         +1.82      -0.50      [-1.79, +0.78]
    eight lags             +1.75      +1.72         -
    NO LEVEL AT ALL        +0.89      -0.00      [-1.37, +1.36]

The last row is the real finding: `make_ridge(with_level=False)` scores identically to
`with_level=True`. **The working model is calendar + trend.**

(One cell reached significance — last-week-only at h=3, -3.25 [-6.36, -0.15]. Discount it:
1 hit in 15 comparisons at 95% is fewer than the 0.75 expected by chance.)

The cause is that the series has no persistence to exploit. Autocorrelation, n=65 so the se
is about 0.124:

    lag      raw     detrended
     1     +0.098    -0.144
     2     +0.207    -0.015
     3     -0.128    -0.435
     4     +0.427    +0.250
     5     +0.229    +0.018

Detrended, only lags 3 and 4 clear noise, and together they are a ~4-week rhythm — the
month-end cycle, which the calendar block already holds in `has_month_end`. The rest is
white noise. No parameterisation of an empty feature can rescue it.

**`with_level=True` is kept anyway, and that is a judgement, not a measurement.** A
near-white weekly series is a property of the GENERATOR; real collections plausibly carry
persistence (a large payer slipping a week, a backlog clearing). One parameter that costs
measurably nothing is cheap insurance. Do not cite the level term as doing work — and on
real data, re-run the comparison rather than assuming the result carries over.

--------------------------------------------------------------------------------------
NEGATIVE RESULT: do not model daily and sum to weekly.
--------------------------------------------------------------------------------------

Rejected on measurement, 2026-09-09. The argument for it was that daily gives 306
observations instead of 65, and that month-end is a DAY effect the weekly model averages
away. Both parts fail: the day-level calendar fit summed to weeks gives R^2 = 0.185
**in-sample** against the weekly model's 0.160, because **day-of-week dominates the
day-level signal and cancels exactly on aggregation** (every week has one Monday and one
Friday). The sample-size argument was overstated too — daily CV / weekly CV is 1.62 where
independence would give 2.24. Full write-up in section 12b of the design document.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

#: Ridge penalties searched by `RidgeCV`. Spread wide because ~50 training points against
#: a dozen regressors can want a lot of shrinkage.
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

#: Minimum training rows before a fitted model is trusted at all.
MIN_TRAIN = 12

#: The name the recommended model is reported under.
RECOMMENDED = "ridge+trend"


def _rows(y: np.ndarray, design: np.ndarray, origin: int, h: int,
          with_level: bool) -> tuple[np.ndarray, np.ndarray]:
    """Training rows as they would have been known at an origin `h` weeks earlier.

    The level column, when present, ends `h` weeks before its own target — the same
    horizon-aware rule the calendar baseline needed. Pairing `y[t]` with the four weeks
    immediately before it fits a one-step relationship and then applies it to a level `h`
    weeks stale.
    """
    start = h + 3 if with_level else 0
    features, targets = [], []
    for t in range(start, origin + 1):
        row = list(design[t])
        if with_level:
            row.insert(0, float(y[t - h - 3:t - h + 1].mean()))
        features.append(row)
        targets.append(y[t])
    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


def _predict_row(y: np.ndarray, design: np.ndarray, origin: int,
                 target_index: int, with_level: bool) -> np.ndarray:
    row = list(design[target_index])
    if with_level:
        row.insert(0, float(y[origin - 3:origin + 1].mean()))
    return np.asarray(row, dtype=float)


def _fallback(y: np.ndarray, origin: int) -> float:
    """What to return when a fit is impossible or produces nonsense."""
    return float(y[max(0, origin - 3):origin + 1].mean())


def make_ridge(log_target: bool = False, with_level: bool = False,
               alphas: tuple[float, ...] = ALPHAS):
    """Ridge on the design matrix, optionally on the log of the target.

    Ridge rather than OLS because ~50 training points against a dozen regressors is
    already at the edge — the same variance problem that made the *correct* horizon-aware
    level term score worse than the mis-specified one.

    With ``log_target`` the prediction is ``exp(fitted)``, which is the conditional
    MEDIAN, not the mean. **No smearing correction is applied, and that is deliberate.**
    Section 7.3 wants smearing because the hurdle model sums thousands of predictions and
    needs each to be a mean; here a single number is scored on relative error, which the
    median serves better. Same transform, opposite correction. (Measured, `log_target`
    loses — it is kept only so `parked/variants.py` can reproduce that result.)
    """
    def predict(y, design, origin, target_index, h):
        X, target = _rows(y, design, origin, h, with_level)
        if len(target) < MIN_TRAIN:
            return _fallback(y, origin)
        scaler = StandardScaler().fit(X)
        # np.asarray, not the tuple: RidgeCV writes back into `alphas[0]` during fit and
        # a tuple raises "does not support item assignment".
        fitted = RidgeCV(alphas=np.asarray(alphas, dtype=float)).fit(
            scaler.transform(X), np.log(target) if log_target else target)
        row = _predict_row(y, design, origin, target_index, with_level)
        value = float(fitted.predict(scaler.transform(row.reshape(1, -1)))[0])
        if log_target:
            value = float(np.exp(value))
        return value if np.isfinite(value) and value > 0 else _fallback(y, origin)
    return predict


def make_lags(n_lags: int = 5, alphas: tuple[float, ...] = ALPHAS):
    """Ridge on `n_lags` INDIVIDUAL past weeks, instead of their mean.

    `make_ridge(with_level=True)` compresses the recent past into one number. A mean cannot
    weight last week above four weeks ago; separate lags let ridge do that. This is the
    version that asks whether the compression was throwing anything away.

    Horizon-aware, like the mean: at training row `t` for horizon `h` the observable weeks
    are `y[t-h], y[t-h-1], ...`, so the earliest usable row is `t = h + n_lags - 1`.

    **Measured on the synthetic corpus, this does not help** — the paired difference against
    the mean-of-four was -0.32 MAPE points averaged over horizons, 95% CI [-1.49, +0.85],
    and dropping the level entirely scored the same again. It is kept in the pipeline
    because the real EY series behaves differently from the synthetic one (its baselines are
    far worse, meaning even less week-to-week persistence), so the question is worth asking
    of each new data set rather than settled once. See the negative result above.
    """
    def predict(y, design, origin, target_index, h):
        start = h + n_lags - 1
        rows, targets = [], []
        for t in range(start, origin + 1):
            lags = [y[t - h - k] for k in range(n_lags)]
            rows.append(np.concatenate((lags, design[t])))
            targets.append(y[t])
        if len(targets) < MIN_TRAIN:
            return _fallback(y, origin)

        X = np.asarray(rows, dtype=float)
        target = np.asarray(targets, dtype=float)
        scaler = StandardScaler().fit(X)
        fitted = RidgeCV(alphas=np.asarray(alphas, dtype=float)).fit(
            scaler.transform(X), target)

        row = np.concatenate(([y[origin - k] for k in range(n_lags)],
                              design[target_index]))
        value = float(fitted.predict(scaler.transform(row.reshape(1, -1)))[0])
        return value if np.isfinite(value) and value > 0 else _fallback(y, origin)
    return predict


def make_deflated(deflator: np.ndarray, projected: np.ndarray,
                  with_level: bool = True, alphas: tuple[float, ...] = ALPHAS):
    """Fit in REAL terms, predict, then re-inflate the prediction to nominal.

    The alternative to a trend regressor. Rather than fitting the drift and extrapolating
    it — which works at h=1 and over-extrapolates by h=5 — the drift is divided out of the
    target before modelling, and put back afterwards.

    ``deflator``   price level per week, leak-free (`inflation.published_price_level`)
    ``projected``  `[origin, target]` price level as projected from each origin
                   (`inflation.projected_price_level`)

    **The design matrix must NOT carry a trend.** Deflating already removes the drift;
    fitting a trend on top would model it twice.

    Two things are deliberately separated here. The model predicts a real amount, which is
    a statement about collections. Converting that to a nominal amount is a statement about
    inflation, and it uses only the rate published as of the origin. Mixing them — using the
    actual future index — would let the model score well on inflation foresight it would
    not have in production.
    """
    def predict(y, design, origin, target_index, h):
        real = y / deflator
        X, target = _rows(real, design, origin, h, with_level)
        if len(target) < MIN_TRAIN:
            return _fallback(y, origin)

        scaler = StandardScaler().fit(X)
        fitted = RidgeCV(alphas=np.asarray(alphas, dtype=float)).fit(
            scaler.transform(X), target)
        row = _predict_row(real, design, origin, target_index, with_level)
        value = float(fitted.predict(scaler.transform(row.reshape(1, -1)))[0])
        if not np.isfinite(value) or value <= 0:
            return _fallback(y, origin)

        # Back to nominal, at the price level projected from THIS origin.
        return value * float(projected[origin, target_index])
    return predict


def weekly_design(series, calendar, with_trend: bool = True) -> np.ndarray:
    """Calendar block, optionally with a linear time index appended.

    The trend is a *model* regressor rather than a calendar feature — it describes the
    series' drift, not the week's calendar — so it is assembled here.

    On real EY data the equivalent question is whether to model in nominal or real terms.
    Turkish inflation makes nominal weekly totals non-stationary, and the synthetic corpus
    carries a drift by construction. A trend regressor is the crude answer; deflating by a
    price index is the better one, and needs a series we do not have.
    """
    from .baselines import calendar_matrix

    block = calendar_matrix(series, calendar)
    if not with_trend:
        return block
    trend = np.arange(len(series), dtype=float).reshape(-1, 1)
    return np.hstack([block, trend])


def evaluate_recommended(series, calendar, weeks=None):
    """Run the recommended model and its close variants over every rolling origin.

    The single call the pipeline makes. Returns the same shape as
    `baselines.rolling_origin_baselines`, so the two concatenate and score together.

    Four models, and each answers one question:

    ``ridge+trend``       the recommendation: calendar + trend + the recent level
    ``ridge+trend_nolevel``   does the level term earn anything? (measured: no)
    ``lags5+trend``       do five separate lags beat their mean? (measured: no)
    ``deflated``          does dividing the drift out beat extrapolating it?

    All four sit on the same calendar block, so the comparison isolates one change each.
    `deflated` is the exception that proves it: it uses the design matrix WITHOUT the trend
    column, because deflating already removes the drift.

    `weeks` is required for `deflated` — the deflator needs real dates. Without it the
    other three still run, so a data set outside the TÜFE table is not blocked.
    """
    from .baselines import rolling_origin

    design = weekly_design(series, calendar, with_trend=True)
    models = {
        RECOMMENDED:            make_ridge(with_level=True),
        "ridge+trend_nolevel":  make_ridge(with_level=False),
        "lags5+trend":          make_lags(5),
    }
    frame = rolling_origin(series, design, models)

    if weeks is None:
        return frame

    from .inflation import (InflationCoverageError, projected_price_level,
                            published_price_level)
    try:
        deflator = published_price_level(weeks)
        projected = projected_price_level(weeks)
    except InflationCoverageError as exc:
        print(f"\n  ! deflated model skipped: {exc}")
        return frame

    import pandas as pd

    # No trend column: the deflation has already taken the drift out.
    flat = weekly_design(series, calendar, with_trend=False)
    deflated = rolling_origin(series, flat, {
        "deflated":         make_deflated(deflator, projected, with_level=True),
        "deflated_nolevel": make_deflated(deflator, projected, with_level=False),
    })
    return pd.concat([frame, deflated], ignore_index=True)
