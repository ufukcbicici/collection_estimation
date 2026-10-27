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
grids. Before pairing, each setting must pass its stage's eligibility gates on validation
(U11.5b: `stage1_gates`, `stage2_gates`), so a pair cannot win the totals by luck while one
half is broken on its own. The best eligible pair on validation (by MAPE or WAPE of the
consolidated totals) is refitted
on train + validation and tested once, beside the default pair and the `deflated` benchmark
(`benchmark.py`).
"""
import dataclasses
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .benchmark import cash_totals
from .diagnostic import Candidate, calibration_slope_intercept, check_outcomes
from .stage2 import amount_summary, check_amounts


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


def stage1_gates(paid: pd.Series, p_by_stage1: list[pd.Series], base_rate: pd.Series,
                 log_loss_tolerance: float, slope_band: tuple[float, float]) -> pd.DataFrame:
    """Which stage-1 settings make sense on their own (U11.5b), judged on validation rows:
    `paid` is what happened (True/False), p_by_stage1[i] setting i's p and `base_rate` the
    base-rate reference p, all with the same rows.

    A setting is eligible if all three hold:
      - its log loss is at most the best setting's x (1 + `log_loss_tolerance`), the best
        being the lowest log loss of this grid;
      - its log-loss skill against the base rate is above 0;
      - its calibration slope is inside `slope_band` (low, high), ends included.

    Returns one row per setting: stage1 = i, bad p (rows whose p is not a finite number in
    [0, 1]), log loss, log loss skill and slope (NaN when bad p is above 0), then one
    True/False column per gate (True = passed): "log loss gate", "skill gate", "slope gate",
    and eligible (all three passed; a setting with a bad p fails all three). The best log
    loss is taken among the settings without a bad p. Raises ValueError for an empty grid.
    """
    if not p_by_stage1:
        raise ValueError("there are no stage-1 settings to check")
    paid = check_outcomes(paid, base_rate)
    base_loss = log_loss(paid, base_rate, labels=[False, True])
    measured = []
    for i, p in enumerate(p_by_stage1):
        check_outcomes(paid, p)
        p = p.astype("float64")
        # A setting with a p that is not a probability is not measured (it fails every
        # gate), so one broken setting does not stop the grid.
        bad = int((~np.isfinite(p) | (p < 0) | (p > 1)).sum())
        loss = skill = slope = float("nan")
        if bad == 0:
            loss = log_loss(paid, p, labels=[False, True])
            skill = 1 - loss / base_loss
            slope, _ = calibration_slope_intercept(paid, p)
        measured.append({"stage1": i, "bad p": bad, "log loss": loss, "log loss skill": skill,
                         "slope": slope})
    table = pd.DataFrame(measured)

    # NaN scores compare False below, so a setting with a bad p fails every gate.
    low, high = slope_band
    loss_limit = table["log loss"].min() * (1 + log_loss_tolerance)
    table["log loss gate"] = table["log loss"] <= loss_limit
    table["skill gate"] = table["log loss skill"] > 0
    table["slope gate"] = table["slope"].between(low, high)
    table["eligible"] = table["log loss gate"] & table["skill gate"] & table["slope gate"]
    return table


def stage2_gates(amount: pd.Series, m_by_stage2: list[pd.Series], reference: pd.Series,
                 skill_tolerance: float, ratio_band: tuple[float, float]) -> pd.DataFrame:
    """Which stage-2 settings make sense on their own (U11.5b), judged on the paid validation
    rows: `amount` is what they paid (deflated), m_by_stage2[j] setting j's m and `reference`
    the reference guess for the deviance skill (the training mean), all with the same rows.

    A setting is eligible if all three hold:
      - every m is finite and above 0 (a paying customer cannot pay nothing or minus);
      - its deviance skill is at least the best setting's - `skill_tolerance`, the best being
        the highest skill of this grid among settings that pass the first gate;
      - its total ratio (sum of m / sum of amount) is inside `ratio_band` (low, high), ends
        included.

    Returns one row per setting: stage2 = j, bad m (rows whose m is not finite or not above
    0), deviance skill and total ratio (NaN when bad m is above 0), then one True/False column
    per gate (True = passed): "m gate", "skill gate", "ratio gate", and eligible (all three
    passed; a setting failing the m gate fails the other two as well, having no scores).
    Raises ValueError for an empty grid, a
    series that does not have the rows of `amount`, an amount that is not a valid paid
    amount, or a reference that is not finite and above 0.
    """
    if not m_by_stage2:
        raise ValueError("there are no stage-2 settings to check")
    amount = check_amounts(amount.to_frame(), amount)
    for name, series in (("reference", reference), *(("m", m) for m in m_by_stage2)):
        if not series.index.equals(amount.index):
            raise ValueError(f"{name} must have the rows of amount (the same index)")
    if not (np.isfinite(reference.astype("float64")) & (reference > 0)).all():
        raise ValueError("reference must be finite and above 0 in every row")
    measured = []
    for j, m in enumerate(m_by_stage2):
        m = m.astype("float64")
        bad = int((~np.isfinite(m) | (m <= 0)).sum())
        skill = ratio = float("nan")
        if bad == 0:
            scores = amount_summary(amount, m, reference)
            skill, ratio = scores["deviance skill"], scores["total ratio"]
        measured.append({"stage2": j, "bad m": bad, "deviance skill": skill,
                         "total ratio": ratio})
    table = pd.DataFrame(measured)

    low, high = ratio_band
    table["m gate"] = table["bad m"] == 0
    # The best among the settings with usable m; NaN scores compare False, so they fail.
    best_skill = table.loc[table["m gate"], "deviance skill"].max()
    table["skill gate"] = table["deviance skill"] >= best_skill - skill_tolerance
    table["ratio gate"] = table["total ratio"].between(low, high)
    table["eligible"] = table["m gate"] & table["skill gate"] & table["ratio gate"]
    return table


def check_actual_totals(actual: pd.Series) -> None:
    """Raises ValueError if the actual cash totals cannot be scored against: empty, a NaN or
    infinite value, a total below 0, or every total 0."""
    if actual.empty:
        raise ValueError("there are no totals to score")
    if not np.isfinite(actual.astype("float64")).all():
        raise ValueError("actual holds a NaN or infinite value")
    if (actual < 0).any():
        raise ValueError("an actual total is below 0; the totals are cash received")
    if actual.sum() == 0:
        raise ValueError("every actual total is 0: no error relative to the cash exists")


def total_scores(actual: pd.Series, forecast: pd.Series) -> dict:
    """The errors of `forecast` against `actual` cash totals (one value per total, same
    index): totals, zero actual (totals with an actual of 0, left out of the MAPE: no
    percentage error exists), MAPE, bias and WAPE (see the module's description).

    Raises ValueError if the two do not line up, `check_actual_totals` refuses the actual
    totals, or the forecast holds a NaN or infinite value.
    """
    if not forecast.index.equals(actual.index):
        raise ValueError("forecast and actual must have the same totals (the same index)")
    check_actual_totals(actual)
    if not np.isfinite(forecast.astype("float64")).all():
        raise ValueError("forecast holds a NaN or infinite value")
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

    A pair whose p x m is NaN or infinite on any row (e.g. a stage-2 setting whose m
    overflows) is not scored: its MAPE, bias and WAPE are infinite, so it ranks last and the
    rest of the grid still runs.

    Returns one row per pair: stage1 = i, stage2 = j, then `total_scores` and "non-finite
    rows" (how many rows had a NaN or infinite p x m; 0 for a scored pair), sorted by
    `metric` ("mape" or "wape"), best first.
    """
    if metric not in ("mape", "wape"):
        raise ValueError(f"cannot rank pairs by {metric!r}; choose 'mape' or 'wape'")
    # Checked here, not left to cash_totals: p x m of misaligned series is NaN wherever the
    # labels differ, which would pass as a failed setting instead of a mistake.
    for predictions in (*p_by_stage1, *m_by_stage2):
        if not predictions.index.equals(rows.index):
            raise ValueError("values must have the rows of rows (the same index)")
    actual = cash_totals(rows, rows["amount_deflated"], by_company=False)
    # Checked once here: an unscored pair never reaches total_scores, so a fault in the
    # actual totals would otherwise hide behind "every pair is NaN".
    check_actual_totals(actual["total"])
    scored = []
    for i, p in enumerate(p_by_stage1):
        for j, m in enumerate(m_by_stage2):
            cash = (p * m).astype("float64")
            non_finite = int((~np.isfinite(cash)).sum())
            if non_finite:
                scores = {"totals": len(actual),
                          "zero actual": int((actual["total"] == 0).sum()),
                          "MAPE": np.inf, "bias": np.inf, "WAPE": np.inf}
            else:
                forecast = cash_totals(rows, cash, by_company=False)
                scores = total_scores(actual["total"], forecast["total"])
            scored.append({"stage1": i, "stage2": j, **scores, "non-finite rows": non_finite})
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
