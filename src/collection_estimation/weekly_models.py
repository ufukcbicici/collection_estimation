"""Candidate weekly models, to be compared against the baselines.

Kept out of `baselines.py` on purpose: that module is the bar, and a candidate should not
live in it.

**What these were built to settle, and what they actually found.**

Every baseline under-predicts by roughly 10% of the weekly mean. The hypothesis was a
loss/metric mismatch — squared error targets the conditional MEAN while MAPE is minimised
nearer the MEDIAN — so two remedies were tried: fitting ``log(y)`` (targets the median) and
a Tweedie GLM with a log link (targets the mean without a back-transform).

**Both failed, and the hypothesis was wrong.** Measured at h=1:

    ridge            MAPE 20.04   bias  -10.9%
    ridge_log        MAPE 20.82   bias  -12.5%     log made both worse
    tweedie_1.3      MAPE 20.90   bias   -9.4%     barely moved the bias
    ma8              MAPE 23.78   bias   -4.5%     the SMALLEST bias, with no calendar

That last row is the tell. A moving average has no loss subtlety at all, so if it is the
least biased the cause cannot be the loss. It is a **trend**: the weekly total grows
**43.2%** from the first quarter of the corpus to the last, and a model fit on all history
regresses toward the historical mean, which in a rising series sits below the present
level. `naive` is least biased precisely because it reports the most recent value.

Adding a linear time regressor removes it:

    ridge+trend      MAPE 19.83   bias   -1.0%

**Note what that does and does not buy.** The MAPE change (20.04 -> 19.83) is far inside
the ~2.5 standard error, so the trend term is justified by BIAS, not accuracy. Indeed no
model here separates from another on MAPE: the whole spread is about one point against a
2.5 error bar. On this series the loss and the link do not matter; the trend does.

**Tweedie, not Gamma, if a mean-targeting model is ever wanted.** Gamma assumes
``Var ∝ mu^2`` — constant coefficient of variation — right for an individual payment but
not for a weekly TOTAL, which is a sum of ~315 of them. Measured, the Tweedie power is
**1.10** by residual regression and **1.26** by level quartile, so `p = 2` is unsupported.
Tweedie combined with a trend over-extrapolates (bias +2.8%, MAPE 22.22), because the log
link compounds with the linear term.

The design matrix is calendar plus an optional trend, and nothing else. That is deliberate:
it isolates the loss, the link and the trend from the effect of adding regressors.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV, TweedieRegressor
from sklearn.preprocessing import StandardScaler

#: Ridge penalties searched by `RidgeCV`. Spread wide because ~50 training points against
#: a dozen regressors can want a lot of shrinkage.
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

#: L2 penalty for the Tweedie GLM. Fixed rather than searched: an inner split on ~50
#: points is thinner than the difference it would resolve.
TWEEDIE_ALPHA = 1e-4

#: Minimum training rows before a fitted model is trusted at all.
MIN_TRAIN = 12


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
    """Ridge on the calendar block, optionally on the log of the target.

    Ridge rather than OLS because ~50 training points against a dozen regressors is
    already at the edge — the same variance problem that made the *correct* horizon-aware
    level term score worse than the mis-specified one.

    With ``log_target`` the prediction is ``exp(fitted)``, which is the conditional
    MEDIAN, not the mean. **No smearing correction is applied, and that is deliberate.**
    Section 7.3 wants smearing because the hurdle model sums thousands of predictions and
    needs each to be a mean; here a single number is scored on relative error, which the
    median serves better. Same transform, opposite correction.
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


def make_tweedie(power: float, with_level: bool = False,
                 alpha: float = TWEEDIE_ALPHA):
    """Tweedie GLM with a log link — targets the conditional MEAN on the natural scale.

    `power` is the variance-mean exponent: 1 is Poisson-like, 2 is Gamma. Measured on this
    series it is about 1.1 to 1.3, so values in that region are the supported ones and
    `power=2` is offered only as a contrast.

    The log link keeps predictions positive without a transform, so unlike the log-target
    ridge there is nothing to back-transform and no median/mean confusion.
    """
    def predict(y, design, origin, target_index, h):
        X, target = _rows(y, design, origin, h, with_level)
        if len(target) < MIN_TRAIN or (target <= 0).any():
            return _fallback(y, origin)
        scaler = StandardScaler().fit(X)
        try:
            fitted = TweedieRegressor(power=power, alpha=alpha, link="log",
                                      max_iter=2000).fit(scaler.transform(X), target)
        except Exception:                       # noqa: BLE001 — a failed fit is not fatal
            return _fallback(y, origin)
        row = _predict_row(y, design, origin, target_index, with_level)
        value = float(fitted.predict(scaler.transform(row.reshape(1, -1)))[0])
        return value if np.isfinite(value) and value > 0 else _fallback(y, origin)
    return predict


def weekly_design(series, calendar, with_trend: bool) -> np.ndarray:
    """Calendar block, optionally with a linear time index appended.

    The trend is a *model* regressor rather than a calendar feature — it describes the
    series' drift, not the week's calendar — so it is assembled here.

    On real EY data the equivalent question is whether to model in nominal or real terms.
    Turkish inflation makes nominal weekly totals non-stationary, and this corpus carries
    a drift by construction. A trend regressor is the crude answer; deflating by a price
    index is the better one, and needs a series we do not have.
    """
    from .baselines import calendar_matrix

    block = calendar_matrix(series, calendar)
    if not with_trend:
        return block
    trend = np.arange(len(series), dtype=float).reshape(-1, 1)
    return np.hstack([block, trend])


def candidate_models(with_trend: bool) -> dict:
    """The set compared in the pipeline, suffixed so both design matrices can be merged.

    Both the trend and the no-trend variants are kept, because the contrast between them
    is the finding — the bias is a trend problem, not a loss problem, and a table showing
    only the fixed version would hide that.
    """
    suffix = "+trend" if with_trend else ""
    return {
        f"ols_level{suffix}": make_ridge(with_level=True, alphas=(1e-8,)),
        f"ridge{suffix}":     make_ridge(with_level=True),
        f"ridge_log{suffix}": make_ridge(log_target=True, with_level=True),
        f"tweedie_1.3{suffix}": make_tweedie(power=1.3, with_level=True),
    }


def compare_weekly_models(series, calendar) -> "object":
    """Run every candidate, with and without the trend, on one harness."""
    import pandas as pd

    from .baselines import rolling_origin

    frames = []
    for with_trend in (False, True):
        design = weekly_design(series, calendar, with_trend)
        frames.append(rolling_origin(series, design, candidate_models(with_trend)))
    return pd.concat(frames, ignore_index=True)
