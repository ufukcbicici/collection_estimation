"""The joint diagnostic of stage 1 and stage 2 (U11.5): how good is the whole model's cash?

DIAGNOSTIC ONLY. Nothing from it reaches the forecast, whose settings are [stage1] and
[stage2].

A row-level error is impossible here (most rows paid 0), so the model is judged on weekly
cash totals. For each origin t and horizon h whose target week t + h is in a split: the
forecast total = the sum of p x m over every customer known at t, the actual total = the sum
of the deflated amounts they paid. The errors of these totals:

  MAPE   the mean of |forecast - actual| / actual over the totals: every week counts the same
  bias   (sum of forecasts - sum of actuals) / sum of actuals: below 0 = forecasts too low
  WAPE   sum of |forecast - actual| / sum of actuals: weeks weighed by their size

The splits are the stage 1 diagnostic's (`diagnostic.diagnostic_splits`). Each stage-1 and
each stage-2 setting is fitted once on the training rows; every pair is then scored by
multiplying their stored p and m, so trying every pair costs no more fits than two separate
grids. The best pair on validation (by MAPE or WAPE of the consolidated totals) is refitted
on train + validation and tested once, beside the default pair and the `deflated` benchmark
(`benchmark.py`).
"""
import dataclasses
import itertools

import numpy as np
import pandas as pd

from .benchmark import cash_totals
from .diagnostic import Candidate


def stage1_grid(stage1, joint) -> list[Candidate]:
    """Every combination of the [joint_diagnostic] stage-1 grid, each a copy of the [stage1]
    settings with those values in place; no class weights (p must be a probability: it is
    multiplied by m)."""
    return [
        Candidate(params={"max_iter": trees, "learning_rate": rate, "max_leaf_nodes": leaves},
                  settings=dataclasses.replace(stage1, max_iter=trees, learning_rate=rate,
                                               max_leaf_nodes=leaves),
                  class_weight="none")
        for trees, rate, leaves in itertools.product(
            joint.stage1_max_iter, joint.stage1_learning_rate, joint.stage1_max_leaf_nodes)
    ]


def stage2_grid(stage2, joint) -> list[Candidate]:
    """Every combination of the [joint_diagnostic] stage-2 grid, each a copy of the [stage2]
    settings with those values in place. The balance floor matters only under the scale
    "open balance", so under "none" each setting appears once (its floor shown as "-")."""
    candidates = []
    cartesian = itertools.product(
            joint.stage2_loss, joint.stage2_max_iter, joint.stage2_learning_rate,
            joint.stage2_max_leaf_nodes, joint.stage2_scale)
    cartesian = [tpl for tpl in cartesian]
    for loss, trees, rate, leaves, scale in cartesian:
        floors = joint.stage2_balance_floor if scale == "open balance" else (None,)
        for floor in floors:
            settings = dataclasses.replace(stage2, loss=loss, max_iter=trees,
                                           learning_rate=rate, max_leaf_nodes=leaves,
                                           scale=scale,
                                           balance_floor=floor or stage2.balance_floor)
            candidates.append(Candidate(
                params={"loss": loss, "max_iter": trees, "learning_rate": rate,
                        "max_leaf_nodes": leaves, "scale": scale, "balance_floor": floor or "-"},
                settings=settings, class_weight="none"))
    return candidates


def total_scores(actual: pd.Series, forecast: pd.Series) -> dict:
    """The errors of `forecast` against `actual` cash totals (one value per total, same
    index): totals, zero actual (totals with an actual of 0, left out of the MAPE: no
    percentage error exists), MAPE, bias and WAPE (see the module's description).

    Raises ValueError if the two do not line up, are empty, hold a NaN or infinite value, or
    an actual total is below 0 or all are 0.
    """
    if not forecast.index.equals(actual.index):
        raise ValueError("forecast and actual must have the same totals (the same index)")
    if actual.empty:
        raise ValueError("there are no totals to score")
    for name, series in (("actual", actual), ("forecast", forecast)):
        if not np.isfinite(series.astype("float64")).all():
            raise ValueError(f"{name} holds a NaN or infinite value")
    if (actual < 0).any():
        raise ValueError("an actual total is below 0; the totals are cash received")
    if actual.sum() == 0:
        raise ValueError("every actual total is 0: no error relative to the cash exists")
    error = forecast - actual
    positive = actual > 0
    return {
        "totals": len(actual),
        "zero actual": int((~positive).sum()),
        "MAPE": float((error[positive].abs() / actual[positive]).mean()),
        "bias": float(error.sum() / actual.sum()),
        "WAPE": float(error.abs().sum() / actual.sum()),
    }


def scores_by_h(totals: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """`total_scores` of each forecast column in `models` against the column `actual` of
    `totals` (one row per total, with an `h` column and, if per company, `company_code`), per
    h and for all h together (h = "all"), per company if the totals have companies. One row
    per (model, (company_code,) h)."""
    groups = ["company_code"] if "company_code" in totals.columns else []
    rows = []
    for model in models:
        for company, part in (totals.groupby(groups) if groups else [(None, totals)]):
            for h in [*sorted(part["h"].unique()), "all"]:
                chosen = part if h == "all" else part[part["h"] == h]
                row = {"model": model}
                if groups:
                    row["company_code"] = company[0] if isinstance(company, tuple) else company
                row["h"] = h
                rows.append({**row, **total_scores(chosen["actual"], chosen[model])})
    return pd.DataFrame(rows)


def score_pairs(rows: pd.DataFrame, p_by_stage1: list[pd.Series],
                m_by_stage2: list[pd.Series], metric: str) -> pd.DataFrame:
    """Every pair of a stage-1 and a stage-2 setting, scored on the weekly cash totals of
    `rows` (panel rows, all companies together).

    p_by_stage1[i] is stage-1 setting i's p for each of `rows`, m_by_stage2[j] stage-2
    setting j's m (both with the index of `rows`). For a pair (i, j), each row's forecast
    cash is p x m; the forecast total of each (origin week, h) is the sum of that over its
    rows, and the actual total the sum of the rows' deflated amounts. Both totals come from
    `cash_totals` on the same rows, so they line up total by total.

    Returns one row per pair: stage1 = i, stage2 = j, then `total_scores`, sorted by
    `metric` ("mape" or "wape"), best first.
    """
    if metric not in ("mape", "wape"):
        raise ValueError(f"cannot rank pairs by {metric!r}; choose 'mape' or 'wape'")
    actual = cash_totals(rows, rows["amount_deflated"], by_company=False)
    scored = []
    for i, p in enumerate(p_by_stage1):
        for j, m in enumerate(m_by_stage2):
            forecast = cash_totals(rows, p * m, by_company=False)
            scored.append({"stage1": i, "stage2": j,
                           **total_scores(actual["total"], forecast["total"])})
    table = pd.DataFrame(scored)
    # mergesort is stable: pairs with the same score keep their grid order.
    return table.sort_values(metric.upper(), kind="mergesort", ignore_index=True)


def paired_bootstrap(actual: pd.Series, forecast_a: pd.Series, forecast_b: pd.Series,
                     samples: int, seed: int) -> dict:
    """Is forecast A more accurate than forecast B on the same totals? The difference of their
    absolute percentage errors, total by total (A's minus B's: below 0 = A better):

      mean difference     the mean of the differences (= MAPE of A - MAPE of B)
      low, high           the 2.5% and 97.5% points of that mean over `samples` resamples of
                          the totals (drawn with replacement)
      A better            the share of totals where A's error is smaller

    Totals from nearby origins overlap (they share target weeks), so their differences are
    correlated and the interval is narrower than it should be. Raises ValueError if the
    three do not line up, there are no totals, a forecast is NaN or infinite, an actual total
    is not above 0, or `samples` < 1.
    """
    for name, series in (("forecast_a", forecast_a), ("forecast_b", forecast_b)):
        if not series.index.equals(actual.index):
            raise ValueError(f"{name} must have the totals of actual (the same index)")
    if actual.empty:
        raise ValueError("there are no totals to compare")
    for name, series in (("forecast_a", forecast_a), ("forecast_b", forecast_b)):
        if not np.isfinite(series.astype("float64")).all():
            raise ValueError(f"{name} holds a NaN or infinite value")
    if not (actual > 0).all():
        raise ValueError("every actual total must be above 0 for a percentage error")
    if samples < 1:
        raise ValueError(f"samples is {samples}; at least 1 is needed")
    difference = (((forecast_a - actual).abs() - (forecast_b - actual).abs()) / actual
                  ).to_numpy()
    draws = np.random.default_rng(seed).integers(0, len(difference),
                                                 size=(samples, len(difference)))
    means = difference[draws].mean(axis=1)
    return {
        "mean difference": float(difference.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "A better": float((difference < 0).mean()),
    }
