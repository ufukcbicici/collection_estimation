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
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from .stage1 import predict_paid


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
