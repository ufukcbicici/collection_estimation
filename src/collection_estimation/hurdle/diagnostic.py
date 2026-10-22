"""The stage 1 diagnostic: how good can stage 1 be, tuned on its own?

DIAGNOSTIC ONLY. Nothing from it reaches the forecast, whose setting is [stage1].

Labelled rows are split by target week, with H the number of horizons and T the origin:
training (target <= T), validation (T < target <= T+H), refit (target <= T+H) and test
(T+H < target <= T+2H). Each model's setting is chosen on validation by the SELECTION METRIC
of the settings: log loss or Brier score (which reward honest probabilities: calibration and
separation together), or average precision (threshold-free, focused on the rare payers;
class-weighted settings, whose p is not a probability, usually win only under it). Every
setting also gets its own best-F1 threshold
on validation. The chosen setting is then refitted and tested once, with the validation
threshold. (The forecast's own setting, both stages together, is chosen elsewhere by the
pooled WAPE of the total cash forecast.)
"""
import dataclasses
import itertools
import numbers
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from .stage1 import predict_paid
from .stage2 import check_amounts


@dataclass(frozen=True)
class DiagnosticSplits:
    """The four sets of labelled rows of the diagnostic."""
    train: pd.DataFrame        # target week <= T
    validation: pd.DataFrame   # T < target week <= T + H
    refit: pd.DataFrame        # target week <= T + H (train and validation together)
    test: pd.DataFrame         # T + H < target week <= T + 2H


def diagnostic_splits(table: pd.DataFrame, origin: int, horizons: int) -> DiagnosticSplits:
    """The labelled rows of `table` (the training table) split by target week around
    `origin` (T) with `horizons` (H). A label is in exactly one of train, validation and test.

    Raises ValueError if a part has no rows (e.g. T + 2H after the last labelled week).
    """
    labelled = table[table["label_status"] == "labelled"]
    target = labelled["target_week"]
    splits = DiagnosticSplits(
        train=labelled[target <= origin],
        validation=labelled[(target > origin) & (target <= origin + horizons)],
        refit=labelled[target <= origin + horizons],
        test=labelled[(target > origin + horizons) & (target <= origin + 2 * horizons)],
    )
    empty = [name for name in ("train", "validation", "test")
             if getattr(splits, name).empty]
    if empty:
        raise ValueError(f"the diagnostic at origin {origin} with {horizons} horizons has no "
                         f"labelled rows for {empty}")
    return splits


def best_threshold(actual: pd.Series, p: pd.Series) -> tuple[float, float]:
    """The threshold on p that gives the highest F1 for `actual` (True/False), and that F1.
    A row is predicted "pays" when p >= threshold.

    F1 is the harmonic mean of precision (of the rows predicted to pay, the share that paid)
    and recall (of the rows that paid, the share predicted to pay). Raises ValueError if
    `actual` and `p` do not line up, or `actual` does not hold both outcomes.
    """
    if not actual.index.equals(p.index):
        raise ValueError("actual and p must have the same rows (the same index)")
    actual = actual.astype(bool)
    if actual.nunique() < 2:
        raise ValueError("actual must hold both outcomes to choose a threshold")
    precision, recall, thresholds = precision_recall_curve(actual, p)
    # The last precision/recall pair (recall 0) has no threshold; leave it out.
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def classification_scores(actual: pd.Series, p: pd.Series, threshold: float) -> dict:
    """The diagnostic's scores of p against `actual` (True/False): F1, precision and recall
    when rows with p >= `threshold` are predicted to pay, plus the threshold-free AUC, average
    precision (the area under the precision-recall curve), log loss and Brier score (the mean
    of (p - outcome)^2)."""
    if not actual.index.equals(p.index):
        raise ValueError("actual and p must have the same rows (the same index)")
    actual = actual.astype(bool)
    predicted = p >= threshold
    return {
        "F1": f1_score(actual, predicted, zero_division=0.0),
        "precision": precision_score(actual, predicted, zero_division=0.0),
        "recall": recall_score(actual, predicted, zero_division=0.0),
        "threshold": float(threshold),
        "AUC": roc_auc_score(actual, p),
        "avg precision": average_precision_score(actual, p),
        "log loss": log_loss(actual, p, labels=[False, True]),
        "brier": brier_score_loss(actual, p),
    }


# The scores a grid can be ranked by (keys of `classification_scores`), and whether lower is
# better for each.
LOWER_IS_BETTER = {"log loss": True, "brier": True, "avg precision": False}


@dataclass(frozen=True)
class Candidate:
    """One setting of a diagnostic grid: what is tried, and how to fit it."""
    params: dict            # the values that vary, shown in the tables, e.g. {"C": 0.1, ...}
    settings: object        # a copy of the [stage1] settings with those values in place
    class_weight: str       # "none" or "balanced"


def class_weight_argument(class_weight: str) -> str | None:
    """The class_weight sklearn takes: None for "none", "balanced" for "balanced"."""
    if class_weight not in ("none", "balanced"):
        raise ValueError(f"class weight {class_weight!r} is not 'none' or 'balanced'")
    return None if class_weight == "none" else "balanced"


def hgb_candidates(stage1, diagnostic) -> list[Candidate]:
    """Every combination of the [diagnostic] HGB grid and class weights, each a copy of the
    [stage1] settings with those values replaced (the forecast's settings are not changed)."""
    return [
        Candidate(params={"max_iter": trees, "learning_rate": rate, "max_leaf_nodes": leaves,
                          "class_weight": weight},
                  settings=dataclasses.replace(stage1, max_iter=trees, learning_rate=rate,
                                               max_leaf_nodes=leaves),
                  class_weight=weight)
        for trees, rate, leaves, weight in itertools.product(
            diagnostic.hgb_max_iter, diagnostic.hgb_learning_rate,
            diagnostic.hgb_max_leaf_nodes, diagnostic.class_weights)
    ]


def lr_candidates(stage1, diagnostic) -> list[Candidate]:
    """Every combination of the [diagnostic] logistic-regression C values and class weights."""
    return [
        Candidate(params={"C": c, "class_weight": weight},
                  settings=dataclasses.replace(stage1, baseline_c=c), class_weight=weight)
        for c, weight in itertools.product(diagnostic.lr_c, diagnostic.class_weights)
    ]


def grid_search(name: str, fit, candidates: list[Candidate], train_x: pd.DataFrame,
                train_paid: pd.Series, validation_x: pd.DataFrame,
                validation_paid: pd.Series, *, metric: str) -> pd.DataFrame:
    """Fit each candidate on the training rows, predict the validation rows, find its own
    best-F1 threshold there and score it. One row per candidate (its params, the validation
    scores at its threshold, the fit time and its position `candidate` in `candidates`),
    sorted by `metric` (a key of LOWER_IS_BETTER), best first. `fit` is `fit_stage1` or
    `fit_stage1_baseline`.
    """
    if metric not in LOWER_IS_BETTER:
        raise ValueError(f"{name}: cannot rank by {metric!r}; choose one of "
                         f"{list(LOWER_IS_BETTER)}")
    if not candidates:
        raise ValueError(f"{name}: the grid has no settings to try")
    rows = []
    for number, candidate in enumerate(candidates, start=1):
        started = time.perf_counter()
        model = fit(train_x, train_paid, candidate.settings,
                    class_weight=class_weight_argument(candidate.class_weight))
        seconds = time.perf_counter() - started
        p = predict_paid(model, validation_x)
        threshold, _ = best_threshold(validation_paid, p)
        scores = classification_scores(validation_paid, p, threshold)
        rows.append({"candidate": number - 1, **candidate.params, **scores,
                     "fit seconds": seconds})
        print(f"  ... {name} {number} of {len(candidates)}: {candidate.params}, "
              f"{metric} {scores[metric]:.4f} ({seconds:.0f} s)")
    grid = pd.DataFrame(rows)
    # mergesort is stable: settings with the same score keep their grid order.
    return grid.sort_values(metric, ascending=LOWER_IS_BETTER[metric], kind="mergesort",
                            ignore_index=True)


def choose_and_test(name: str, fit, candidates: list[Candidate], grid: pd.DataFrame,
                    refit_x: pd.DataFrame, refit_paid: pd.Series, test_x: pd.DataFrame,
                    test_paid: pd.Series) -> dict:
    """The test row of the best setting of `grid` (from `grid_search`): refit it on the refit
    rows (train and validation), predict the test rows and score them with the threshold it
    got on validation, unchanged. Returns {"model": name, the setting's params, the test
    scores}."""
    best = grid.iloc[0]
    candidate = candidates[int(best["candidate"])]
    model = fit(refit_x, refit_paid, candidate.settings,
                class_weight=class_weight_argument(candidate.class_weight))
    p = predict_paid(model, test_x)
    scores = classification_scores(test_paid, p, float(best["threshold"]))
    return {"model": name, **candidate.params, **scores}


def permutation_importances(model, x: pd.DataFrame, paid: pd.Series, repeats: int,
                            seed: int) -> pd.DataFrame:
    """How much each feature matters to `model` on the rows `x`: shuffle the feature's
    column `repeats` times and measure how much the log loss rises (0 or less: the model does
    not use it). One row per feature, most important first, with the mean rise and its
    standard deviation over the shuffles."""
    result = permutation_importance(model, x, paid.astype(bool), scoring="neg_log_loss",
                                    n_repeats=repeats, random_state=seed)
    # The score is minus the log loss, so a fall in the score is a rise in the log loss.
    table = pd.DataFrame({"feature": list(x.columns),
                          "log loss rise": result.importances_mean,
                          "std": result.importances_std})
    return table.sort_values("log loss rise", ascending=False, kind="mergesort",
                             ignore_index=True)


def write_tables(folder: Path, tables: dict[str, pd.DataFrame]) -> list[Path]:
    """Write each table as `<name>.csv` in `folder` (created if missing); returns the paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = folder / f"{name}.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    return paths


# p is kept this far from 0 and 1 before taking logit(p), which is infinite at the edges.
LOGIT_CLIP = 1e-6


def check_outcomes(actual: pd.Series, p: pd.Series) -> pd.Series:
    """`actual` as True/False, after checking that it lines up with `p`, is not empty and holds
    both outcomes (the measures below compare payers with non-payers)."""
    if not actual.index.equals(p.index):
        raise ValueError("actual and p must have the same rows (the same index)")
    if actual.empty:
        raise ValueError("there are no rows to measure")
    actual = actual.astype(bool)
    if actual.nunique() < 2:
        raise ValueError("actual must hold both outcomes (payers and non-payers)")
    return actual


def equal_groups(prediction: pd.Series, groups: int) -> np.ndarray:
    """The group (1 to `groups`) of each row when the rows are sorted by `prediction` and cut
    into `groups` groups of (nearly) equal size: the lowest predictions are group 1. In the
    order of `prediction`'s rows. Tied predictions keep their row order (a stable sort).

    Equal-size groups, not equal-width ones: most predictions are small, so equal widths would
    leave the top groups nearly empty. Raises ValueError if `groups` < 2, there are fewer rows
    than groups, or a prediction is NaN (it cannot be sorted).
    """
    if groups < 2:
        raise ValueError(f"groups is {groups}; at least 2 are needed")
    if len(prediction) < groups:
        raise ValueError(f"{len(prediction)} rows cannot be cut into {groups} groups")
    if prediction.isna().any():
        raise ValueError(f"the prediction is NaN in {int(prediction.isna().sum())} rows")
    # Position of each row in the sorted order, then its group: the first len/groups
    # positions are group 1, and so on.
    order = np.argsort(prediction.to_numpy(), kind="mergesort")
    group = np.empty(len(prediction), dtype=int)
    group[order] = np.arange(len(prediction)) * groups // len(prediction) + 1
    return group


def reliability_table(actual: pd.Series, p: pd.Series, groups: int) -> pd.DataFrame:
    """How well p matches what happened: the rows in `groups` equal-size groups by p
    (`equal_groups`), one row per group with its lowest and highest p, its mean p, the share
    of its rows that actually paid, and its row count. A calibrated model has mean p close to
    the share paid in every group.

    Raises ValueError as `equal_groups` and `check_outcomes` do.
    """
    actual = check_outcomes(actual, p)
    group = equal_groups(p, groups)
    rows = pd.DataFrame({"group": group, "p": p.to_numpy(), "paid": actual.to_numpy()})
    table = rows.groupby("group").agg(p_low=("p", "min"), p_high=("p", "max"),
                                      mean_p=("p", "mean"), share_paid=("paid", "mean"),
                                      rows=("p", "size"))
    return table.reset_index()


def cash_by_p_group(p: pd.Series, m: pd.Series, actual_amount: pd.Series,
                    groups: int) -> pd.DataFrame:
    """The amount-weighted calibration check of the whole model: the rows (paid or not) in
    `groups` equal-size groups by p, one row per group with its lowest and highest p, its mean
    p, the expected cash (the sum of p x m), the actual cash (the sum of `actual_amount`, 0
    for rows that did not pay), their ratio (expected / actual; 1.0 = the cash adds up) and
    the row count.

    It shows whether m, learned on payers, adds cash on rows unlike any payer: on the low-p
    groups p x m should still match the little cash that came. Raises ValueError if `m` or
    `actual_amount` does not have the rows of `p`, either holds a NaN or infinite value, an
    actual amount is below 0, and as `equal_groups` does.
    """
    for name, series in (("m", m), ("actual_amount", actual_amount)):
        if not series.index.equals(p.index):
            raise ValueError(f"{name} must have the rows of p (the same index)")
        if not np.isfinite(series.astype("float64")).all():
            raise ValueError(f"{name} holds a NaN or infinite value")
    if (actual_amount < 0).any():
        raise ValueError("actual_amount is below 0 in some rows; it is the cash received")
    rows = pd.DataFrame({"group": equal_groups(p, groups), "p": p.to_numpy(),
                         "expected": (p * m).to_numpy(),
                         "actual": actual_amount.astype("float64").to_numpy()})
    table = rows.groupby("group").agg(p_low=("p", "min"), p_high=("p", "max"),
                                      mean_p=("p", "mean"), expected_cash=("expected", "sum"),
                                      actual_cash=("actual", "sum"), rows=("p", "size"))
    # A group that received no cash has no ratio.
    table.insert(5, "ratio", (table["expected_cash"] / table["actual_cash"])
                 .where(table["actual_cash"] > 0))
    return table.reset_index()


def amount_reliability(actual: pd.Series, m: pd.Series, groups: int) -> pd.DataFrame:
    """Stage 2's reliability table: the paid rows in `groups` equal-size groups by m, one row
    per group with its lowest and highest m, its mean m, the mean `actual` amount, their ratio
    (mean actual / mean m; 1.0 = calibrated) and the row count. Calibrated m: the mean actual
    equals the mean m in every group.

    Grouped by m, not by the actual amount: grouped by the actual, small payments always look
    over-predicted and large ones under-predicted, even for a perfect model of the mean
    (regression to the mean). Raises ValueError if `m` does not have the rows of `actual`, is
    NaN or infinite, `actual` is not a valid paid amount (`check_amounts`), and as
    `equal_groups` does.
    """
    if not m.index.equals(actual.index):
        raise ValueError("m must have the rows of actual (the same index)")
    actual = check_amounts(actual.to_frame(), actual)
    if not np.isfinite(m.astype("float64")).all():
        raise ValueError("m holds a NaN or infinite value")
    rows = pd.DataFrame({"group": equal_groups(m, groups), "m": m.to_numpy(),
                         "actual": actual.to_numpy()})
    table = rows.groupby("group").agg(m_low=("m", "min"), m_high=("m", "max"),
                                      mean_m=("m", "mean"), mean_actual=("actual", "mean"),
                                      rows=("m", "size"))
    table.insert(4, "ratio", table["mean_actual"] / table["mean_m"])
    return table.reset_index()


def calibration_slope_intercept(actual: pd.Series, p: pd.Series) -> tuple[float, float]:
    """The slope a and intercept b of an unpenalised logistic regression of the outcome on
    logit(p). Calibrated: a = 1, b = 0. a below 1: p is too extreme (too confident); b
    shifts every p up (b > 0) or down (b < 0). p is first clipped to [LOGIT_CLIP,
    1 - LOGIT_CLIP]."""
    actual = check_outcomes(actual, p)
    logit = clipped_logit(p).to_numpy().reshape(-1, 1)
    # C = infinity: no penalty (scikit-learn 1.8 deprecates penalty=None for this).
    fit = LogisticRegression(C=np.inf).fit(logit, actual.to_numpy())
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def clipped_logit(p: pd.Series) -> pd.Series:
    """logit(p) = log(p / (1 - p)), with p first clipped to [LOGIT_CLIP, 1 - LOGIT_CLIP]."""
    clipped = p.clip(LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(clipped / (1 - clipped))


def fit_platt(actual: pd.Series, p: pd.Series) -> tuple[float, float]:
    """The Platt correction learned from p and what happened: (a, b) such that
    sigmoid(a x logit(p) + b) matches the outcomes best (see `apply_platt`). It is the same fit
    as `calibration_slope_intercept`; this name is for learning a correction, that one for
    measuring calibration."""
    return calibration_slope_intercept(actual, p)


def apply_platt(p: pd.Series, a: float, b: float) -> pd.Series:
    """p corrected by Platt's (a, b) from `fit_platt`: sigmoid(a x logit(p) + b), with the same
    index as p, every value within [LOGIT_CLIP, 1 - LOGIT_CLIP], so strictly between 0 and 1
    even for a large correction. Raises ValueError unless a and b are finite numbers (numpy
    numbers included)."""
    for name, value in (("a", a), ("b", b)):
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not np.isfinite(value)):
            raise ValueError(f"Platt's {name} is {value!r}, not a finite number")
    # expit is the sigmoid without overflow; the clip keeps 0 and 1 out, which floating point
    # reaches for |a x logit(p) + b| above about 37.
    return expit(a * clipped_logit(p) + b).clip(LOGIT_CLIP, 1 - LOGIT_CLIP)


def calibration_summary(actual: pd.Series, p: pd.Series, reference_p: pd.Series,
                        groups: int) -> dict:
    """One row of calibration measures for p against `actual` (True/False):

      mean p, share paid    calibration in the large
      ECE                   expected calibration error: the size-weighted mean of
                            |mean p - share paid| over the `groups` groups of
                            `reliability_table`; 0 = calibrated
      slope, intercept      see `calibration_slope_intercept`
      log loss, brier       the two proper scores (lower is better)
      calibration error, resolution, uncertainty
                            the Brier decomposition over the same groups: how far p is from
                            the share paid (lower is better), how much the groups' shares
                            differ from the overall share (higher is better: payers are
                            separated), and the data's own difficulty
      decomposition gap     brier - (calibration error - resolution + uncertainty): not 0,
                            because grouping p loses its spread within each group
      log loss skill, brier skill
                            1 - score / `reference_p`'s score: 0 = no better than the
                            reference, 1 = perfect, below 0 = worse
      AUC                   separation of payers from non-payers, beside calibration
    """
    actual = check_outcomes(actual, p)
    if not reference_p.index.equals(p.index):
        raise ValueError("reference_p and p must have the same rows (the same index)")
    table = reliability_table(actual, p, groups)
    weights = table["rows"] / table["rows"].sum()
    overall = actual.mean()
    slope, intercept = calibration_slope_intercept(actual, p)
    scores = {"log loss": log_loss(actual, p, labels=[False, True]),
              "brier": brier_score_loss(actual, p)}
    reference = {"log loss": log_loss(actual, reference_p, labels=[False, True]),
                 "brier": brier_score_loss(actual, reference_p)}
    calibration_error = float((weights * (table["mean_p"] - table["share_paid"]) ** 2).sum())
    resolution = float((weights * (table["share_paid"] - overall) ** 2).sum())
    uncertainty = float(overall * (1 - overall))
    return {
        "mean p": float(p.mean()),
        "share paid": float(overall),
        "ECE": float((weights * (table["mean_p"] - table["share_paid"]).abs()).sum()),
        "slope": slope,
        "intercept": intercept,
        "log loss": scores["log loss"],
        "brier": scores["brier"],
        "calibration error": calibration_error,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "decomposition gap": scores["brier"] - (calibration_error - resolution + uncertainty),
        "log loss skill": 1 - scores["log loss"] / reference["log loss"],
        "brier skill": 1 - scores["brier"] / reference["brier"],
        "AUC": roc_auc_score(actual, p),
    }


def reference_probabilities(train_rows: pd.DataFrame, rows: pd.DataFrame, window: int,
                            prior_weeks: float) -> pd.DataFrame:
    """Three simple references for p, one column each, with the same index as `rows`; all
    learned from `train_rows` (known at fit time), scored like a model:

      base rate         the share paid in `train_rows`, for every row
      base rate per h   the share paid in `train_rows` with the row's horizon h
      own rate          the customer's own recent pay rate: (paying weeks + prior_weeks x base
                        rate) / (observed weeks + prior_weeks), from the row's history columns
                        paying_weeks_<window> and observed_weeks_<window> (weeks up to the
                        origin t, so known at t). The prior weeks keep a rate of exactly 0 or 1
                        away from the edges (log loss is infinite there); a customer with no
                        observed week gets exactly the base rate.

    Raises ValueError if `rows` has an h that `train_rows` lacks, the history columns of
    `window` are missing or impossible (negative, NaN, or more paying than observed weeks),
    `prior_weeks` is not above 0, or `train_rows` holds one outcome only, overall or for an h
    (a rate of 0 or 1).
    """
    if not prior_weeks > 0:
        raise ValueError(f"prior_weeks is {prior_weeks!r}; it must be above 0")
    paying, observed = f"paying_weeks_{window}", f"observed_weeks_{window}"
    missing = [column for column in (paying, observed) if column not in rows.columns]
    if missing:
        raise ValueError(f"the rows have no history columns {missing} for window {window}")
    impossible = ~((rows[paying] >= 0) & (rows[paying] <= rows[observed]))
    if impossible.any():
        raise ValueError(f"{int(impossible.sum())} rows have impossible history counts for "
                         f"window {window} (negative, NaN, or more paying than observed weeks)")
    train_paid = train_rows["paid"].astype(bool)
    if train_paid.nunique() < 2:
        raise ValueError("the training rows must hold both outcomes for a base rate")
    base_rate = float(train_paid.mean())
    rate_per_h = train_paid.groupby(train_rows["h"]).mean()
    unknown = sorted(set(rows["h"]) - set(rate_per_h.index))
    if unknown:
        raise ValueError(f"the training rows have no rows with h {unknown}")
    one_outcome = sorted(h for h, rate in rate_per_h.items() if rate in (0.0, 1.0))
    if one_outcome:
        raise ValueError(f"the training rows hold one outcome only for h {one_outcome}")
    own_rate = ((rows[paying] + prior_weeks * base_rate)
                / (rows[observed] + prior_weeks))
    # With no observed week the formula is the base rate, but floating point can miss it by
    # one last digit; set it exactly.
    own_rate = own_rate.where(rows[observed] > 0, base_rate)
    return pd.DataFrame({
        "base rate": base_rate,
        "base rate per h": rows["h"].map(rate_per_h).astype("float64"),
        "own rate": own_rate.astype("float64"),
    }, index=rows.index)


def best_unweighted(grid: pd.DataFrame, candidates: list[Candidate]) -> Candidate | None:
    """The highest-ranked setting of `grid` (from `grid_search`) without class weights, or None
    if the grid has none. Calibration is measured on it: a class-weighted model's p is not a
    probability."""
    for number in grid["candidate"]:
        candidate = candidates[int(number)]
        if candidate.class_weight == "none":
            return candidate
    return None


def calibration_comparison(actual: pd.Series, h: pd.Series, predictions: dict[str, pd.Series],
                           reference_p: pd.Series,
                           groups: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The calibration of several sets of p (`predictions`: label -> p, e.g. models, models
    after Platt, references) on the same rows, for each horizon in `h` and for all rows
    together (h = "all"). Returns two long tables:

      summary       one row per (model, h): `calibration_summary` against `reference_p`
      reliability   one row per (model, h, group): `reliability_table`

    Raises ValueError if `h`, `reference_p` or a prediction does not have the rows of `actual`
    (and as `calibration_summary` does, e.g. for a horizon whose rows hold one outcome).
    """
    for name, series in (("h", h), ("reference_p", reference_p), *predictions.items()):
        if not series.index.equals(actual.index):
            raise ValueError(f"{name} must have the rows of actual (the same index)")
    summary, reliability = [], []
    for label, p in predictions.items():
        for horizon in [*sorted(h.unique()), "all"]:
            rows = h.index if horizon == "all" else h.index[h == horizon]
            summary.append({"model": label, "h": horizon,
                            **calibration_summary(actual[rows], p[rows], reference_p[rows],
                                                  groups)})
            table = reliability_table(actual[rows], p[rows], groups)
            table.insert(0, "h", horizon)
            table.insert(0, "model", label)
            reliability.append(table)
    return pd.DataFrame(summary), pd.concat(reliability, ignore_index=True)
