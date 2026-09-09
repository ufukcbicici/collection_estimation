"""Weekly model variants that were tried and lost.

PARKED, and kept for the record rather than for use. Each one tested a specific hypothesis
about why every model under-predicted, and each is a measured negative result written up in
section 12b of the design document.

**Re-run these on the real EY data before trusting the conclusions.** Every number below
was measured on a 65-week SYNTHETIC corpus whose dynamics we invented. Two of the four
findings depend on properties of that corpus which real data need not share:

* the Tweedie result depends on the variance-mean power of the weekly total (measured
  1.1-1.3 here, where Gamma assumes 2.0);
* the derived-regressor result depends on the customer population still growing at the end
  of the series, which is a 65-week artefact.

`compare_weekly_models` runs the whole set against both design matrices in one pass. It is
the cheapest way to check whether these conclusions survive contact with real files.

--------------------------------------------------------------------------------------
What was measured, at h=1 on the synthetic corpus
--------------------------------------------------------------------------------------

    ridge            MAPE 20.04   bias  -10.9%
    ridge_log        MAPE 20.82   bias  -12.5%     log made both worse
    tweedie_1.3      MAPE 20.90   bias   -9.4%     barely moved the bias
    ma8              MAPE 23.78   bias   -4.5%     the SMALLEST bias, with no calendar
    ridge+trend      MAPE 19.83   bias   -1.0%     <- the recommendation

The `ma8` row is the tell. A moving average has no loss subtlety at all, so if it is the
least biased the cause cannot be the loss function. It was a **trend** all along.

Tweedie combined with a trend over-extrapolates (bias +2.8%, MAPE 22.22), because the log
link compounds with the linear term. Gamma (`power=2`) is unsupported here regardless: it
assumes a constant coefficient of variation, right for one payment and wrong for a weekly
total that sums ~315 of them.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV, TweedieRegressor
from sklearn.preprocessing import StandardScaler

from ..weekly_models import (ALPHAS, MIN_TRAIN, _fallback, _predict_row, _rows,
                             make_ridge, weekly_design)

#: L2 penalty for the Tweedie GLM. Fixed rather than searched: an inner split on ~50
#: points is thinner than the difference it would resolve.
TWEEDIE_ALPHA = 1e-4


def make_tweedie(power: float, with_level: bool = False,
                 alpha: float = TWEEDIE_ALPHA):
    """Tweedie GLM with a log link — targets the conditional MEAN on the natural scale.

    `power` is the variance-mean exponent: 1 is Poisson-like, 2 is Gamma. Measured on the
    synthetic series it is about 1.1 to 1.3, so values in that region are the supported
    ones and `power=2` is offered only as a contrast.

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


def make_ridge_with_derived(derived: dict, alphas: tuple[float, ...] = ALPHAS):
    """Ridge on calendar + trend + the derived regressors of `derived_regressors.py`.

    The derived block is indexed by ``(origin, horizon)`` rather than by week, because it
    is computed from customer history as of the origin. So it cannot ride in the design
    matrix the harness passes around — it is closed over here, and looked up for the
    training rows at ``(t - h, h)`` and for the prediction at ``(origin, h)``.

    That lookup is the same horizon-consistency rule as the level term: a training row
    must carry the regressors that would have been known when forecasting its own target
    from `h` weeks out.
    """
    from .derived_regressors import derived_matrix

    def predict(y, design, origin, target_index, h):
        start = h + 3
        rows, targets = [], []
        for t in range(start, origin + 1):
            level = float(y[t - h - 3:t - h + 1].mean())
            rows.append(np.concatenate(
                ([level], design[t], derived_matrix(derived, t - h, h))))
            targets.append(y[t])
        if len(targets) < MIN_TRAIN:
            return _fallback(y, origin)

        X = np.asarray(rows, dtype=float)
        target = np.asarray(targets, dtype=float)
        scaler = StandardScaler().fit(X)
        fitted = RidgeCV(alphas=np.asarray(alphas, dtype=float)).fit(
            scaler.transform(X), target)

        row = np.concatenate(([float(y[origin - 3:origin + 1].mean())],
                              design[target_index],
                              derived_matrix(derived, origin, h)))
        value = float(fitted.predict(scaler.transform(row.reshape(1, -1)))[0])
        return value if np.isfinite(value) and value > 0 else _fallback(y, origin)
    return predict


def candidate_models(with_trend: bool) -> dict:
    """The set compared in the original pipeline, suffixed so both matrices can merge.

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


def compare_weekly_models(series, calendar):
    """Run every candidate, with and without the trend, on one harness.

    This is the function to call on real EY data to check whether the four negative
    results above survive. It costs seconds.
    """
    import pandas as pd

    from ..baselines import rolling_origin

    frames = []
    for with_trend in (False, True):
        design = weekly_design(series, calendar, with_trend)
        frames.append(rolling_origin(series, design, candidate_models(with_trend)))
    return pd.concat(frames, ignore_index=True)
