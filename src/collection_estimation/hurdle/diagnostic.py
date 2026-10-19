"""The stage 1 diagnostic: how well does stage 1 classify payers when tuned for it?

DIAGNOSTIC ONLY. The forecast needs honest probabilities (log loss, no class weights, the
[stage1] setting); this module instead tunes each model for F1, class weights allowed, to
see how good a classifier it can be. Nothing from it reaches the forecast.

Labelled rows are split by target week, with H the number of horizons and T the origin:
training (target <= T), validation (T < target <= T+H), refit (target <= T+H) and test
(T+H < target <= T+2H). Each model's setting is chosen on validation by AVERAGE PRECISION
(threshold-free, and focused on the rare payers), with its log loss shown beside it; the F1
threshold is chosen on validation too. The chosen setting is then refitted and tested once,
with the validation threshold. (The forecast's own setting, both stages together, is chosen
elsewhere by the pooled WAPE of the total cash forecast.)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


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
    precision (the area under the precision-recall curve) and log loss."""
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
    }
