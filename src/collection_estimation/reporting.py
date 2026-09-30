"""Review reporting: the per-forecast log, cash-denominated error, and the over-forecast
risk that the percentage metrics hide.

Added 2026-09-11 in response to Reyzi's review of the model deck. He asked for four things
and one correction, and this module is all five:

* **A per-forecast record.** Origin date, training cutoff, target week, actual, prediction,
  one row each, so two models can be checked to have been scored over the same periods
  rather than taken on trust. `forecast_log`.
* **The code version beside it**, so a result can be tied to what produced it.
  `code_version`.
* **TL-denominated error and WAPE beside MAPE.** A percentage cannot be added up, cannot be
  compared across weeks of different size, and cannot be put in front of a treasurer.
  `cash_error`.
* **The five-week cumulative error per origin.** The h=5 MAPE describes week five on its
  own; nobody plans against week five on its own. `cumulative_by_origin`.
* **Over- and under-forecasting reported separately.** `cash_error` and `cumulative_risk`.

--------------------------------------------------------------------------------------
THE SIGN CONVENTION, AND WHY IT IS THE POINT
--------------------------------------------------------------------------------------

Throughout this module ``error = predicted - actual``, so:

    error > 0   OVER-forecast    cash was expected that did not arrive
    error < 0   UNDER-forecast   more cash arrived than was expected

**These are different risks and neither is "the" risk.** An over-forecast is a *liquidity*
exposure: obligations were planned against money that did not come. An under-forecast is a
*carrying cost*: a buffer was held, or credit drawn, that turned out to be unnecessary.
Which one is more expensive is a treasury policy question and not a modelling one.

The deck previously said under-forecasting was "the error that matters for a cash
forecast". That was one-sided, and this module exists so the question is settled by
measurement rather than by wording: both directions are reported, always, with the
over-forecast tail called out separately because it is the one with a hard consequence.

--------------------------------------------------------------------------------------
ON POOLING HORIZONS
--------------------------------------------------------------------------------------

The five horizons from one origin are **not independent**. They share the origin, the
training window and, at h=1..5, four of the same five target weeks' worth of drift. Two
consequences, both of which this module is built around:

1. Counting "all five horizons point the same way" as five independent agreements and
   deriving a p-value from it is invalid. No function here does that.
2. The five-week cumulative error cannot be inferred from the five per-horizon errors; it
   has to be measured. `cumulative_by_origin` measures it directly, and the cumulative
   error is exactly the sum of the five signed errors, so offsetting and compounding both
   show up where they actually happen.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

#: The horizons a cumulative figure is built over. A cumulative row is emitted only for an
#: origin that has every one of them, so the totals are comparable across origins.
CUMULATIVE_HORIZONS = (1, 2, 3, 4, 5)


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------

def code_version(repo_root: str | Path | None = None) -> str:
    """The commit that produced a result, as ``<sha>`` or ``<sha>-dirty``.

    Reyzi asked for the code version alongside the results file, and the reason is
    ordinary reproducibility: a results file without one cannot be re-derived six weeks
    later, when the only question anybody asks is "which run was this".

    **Never raises.** Returns ``"unknown"`` if git is absent, if this is not a checkout, or
    if the call fails for any other reason. A missing version is a nuisance; an exception
    here would abort a pipeline run that had already done all its work.

    ``-dirty`` means the working tree had uncommitted changes, which makes the sha alone
    an incomplete description of what ran. Treat a dirty result as provisional.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        if not sha:
            return "unknown"
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return f"{sha}-dirty" if status else sha
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------
# The per-forecast log
# --------------------------------------------------------------------------------------

def forecast_log(evaluations: pd.DataFrame, weeks: pd.DataFrame, *,
                 version: str | None = None) -> pd.DataFrame:
    """One row per forecast, with real dates instead of week indices.

    `rolling_origin` already emits the right rows; what it does not carry is anything a
    reader can check against a calendar. This joins the week spine back on and names the
    columns for someone who did not write the pipeline.

    ``train_cutoff`` is the **last day whose collections the model was allowed to see**,
    which is the origin week's Sunday. It is stated explicitly because "origin week 41" is
    not a claim anyone can audit and "trained on data to 2026-03-15" is.

    ``direction`` splits over from under rather than leaving only a signed number, so the
    file can be grouped on it without every reader re-deriving the convention.
    """
    spine = weeks.set_index("week")[["iso_week", "week_start", "week_end"]]

    log = evaluations.copy()
    origin = log["origin"].map(spine["week_start"])
    log["origin_week"] = log["origin"].map(spine["iso_week"])
    log["origin_week_start"] = origin
    log["train_cutoff"] = log["origin"].map(spine["week_end"])
    log["target_iso_week"] = log["target_week"].map(spine["iso_week"])
    log["target_week_start"] = log["target_week"].map(spine["week_start"])
    log["target_week_end"] = log["target_week"].map(spine["week_end"])

    log = log.rename(columns={"actual": "actual_try", "predicted": "predicted_try"})
    log["error_try"] = log["predicted_try"] - log["actual_try"]
    log["abs_error_try"] = log["error_try"].abs()

    actual = log["actual_try"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(log["error_try"].to_numpy(dtype=float) /
                     np.where(actual == 0, np.nan, actual)) * 100
    log["ape_pct"] = ape

    log["direction"] = np.where(log["error_try"] > 0, "over",
                                np.where(log["error_try"] < 0, "under", "exact"))
    log["code_version"] = version if version is not None else code_version()

    return log[[
        "model", "horizon",
        "origin", "origin_week", "origin_week_start", "train_cutoff",
        "target_week", "target_iso_week", "target_week_start", "target_week_end",
        "actual_try", "predicted_try", "error_try", "abs_error_try", "ape_pct",
        "direction", "code_version",
    ]].sort_values(["model", "origin", "horizon"]).reset_index(drop=True)


def write_forecast_log(log: pd.DataFrame, path: str | Path) -> Path:
    """Write the log as UTF-8 CSV with ISO dates.

    UTF-8 **with BOM** (``utf-8-sig``): the intended reader opens this in Excel on a
    Turkish Windows machine, and without the BOM Excel decodes it as cp1254 and mangles
    every model name and header it does not expect. The BOM costs three bytes and removes
    an entire class of "the file is broken" replies.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = log.copy()
    for column in ("origin_week_start", "train_cutoff",
                   "target_week_start", "target_week_end"):
        if column in out:
            out[column] = pd.to_datetime(out[column]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# --------------------------------------------------------------------------------------
# Error in cash, and in both directions
# --------------------------------------------------------------------------------------

def cash_error(evaluations: pd.DataFrame, *, mean_weekly: float | None = None
               ) -> pd.DataFrame:
    """Per model and horizon: percentage error, cash error, and the two directions apart.

    `baselines.score` reports MAPE, WAPE, R² and a bias in TRY. This adds what a reader
    needs in order to act on them:

    ``mae_try``        mean absolute error in lira. The number to put in front of a
                       treasurer, because a percentage of a week whose size varies by a
                       factor of three is not a quantity anyone can plan against.
    ``bias_pct``       the signed bias as a share of the mean week, which is how the deck
                       quotes it. Requires `mean_weekly`; otherwise NaN.
    ``over_rate_pct``  share of forecasts that came in ABOVE the actual.
    ``mean_over_try``  average size of an over-forecast, counted over the over-forecasts
                       only — not diluted by the weeks that went the other way.
    ``worst_over_try`` the largest single over-forecast. A tail number, reported because a
                       liquidity problem is caused by the worst week and not the average
                       one.

    ``mean_under_try`` and ``worst_under_try`` mirror them, positive-signed, because the
    cost of holding an unnecessary buffer is real even though it is not a shortfall.
    """
    out = []
    for (model, horizon), group in evaluations.groupby(["model", "horizon"],
                                                       observed=True):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        error = predicted - actual

        over = error[error > 0]
        under = -error[error < 0]

        bias_try = float(error.mean())
        out.append({
            "model": model,
            "horizon": int(horizon),
            "n": len(group),
            "mae_try": round(float(np.abs(error).mean()), 0),
            "bias_try": round(bias_try, 0),
            "bias_pct": round(bias_try / mean_weekly * 100, 1)
            if mean_weekly else float("nan"),
            "over_rate_pct": round(float(len(over) / len(error) * 100), 1),
            "mean_over_try": round(float(over.mean()), 0) if len(over) else 0.0,
            "worst_over_try": round(float(over.max()), 0) if len(over) else 0.0,
            "mean_under_try": round(float(under.mean()), 0) if len(under) else 0.0,
            "worst_under_try": round(float(under.max()), 0) if len(under) else 0.0,
        })
    return pd.DataFrame(out).sort_values(["model", "horizon"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# The five-week cumulative
# --------------------------------------------------------------------------------------

def cumulative_by_origin(evaluations: pd.DataFrame,
                         horizons: tuple[int, ...] = CUMULATIVE_HORIZONS
                         ) -> pd.DataFrame:
    """Per model and origin: total cash forecast over h=1..5 against what arrived.

    This is the quantity a rolling cash plan is actually built on, and it is **not**
    recoverable from the per-horizon table. Errors from one origin can offset across the
    five weeks or compound across them, and which one happens is a property of the model:
    anything that extrapolates a trend compounds, anything flat offsets.

    Only origins carrying every horizon are kept, so the totals are comparable. An origin
    near the end of the series has fewer than five target weeks available and is dropped
    rather than totalled over a shorter window, which would understate its error.
    """
    wanted = set(horizons)
    frame = evaluations[evaluations["horizon"].isin(wanted)]

    out = []
    for (model, origin), group in frame.groupby(["model", "origin"], observed=True):
        # Length as well as membership: a set comparison cannot see a duplicated
        # horizon, which would be summed twice into a total that still looks
        # plausible. `rolling_origin_baselines` cannot produce one today.
        if (set(group["horizon"].astype(int)) != wanted
                or len(group) != len(wanted)):
            continue
        actual = float(group["actual"].sum())
        predicted = float(group["predicted"].sum())
        error = predicted - actual
        out.append({
            "model": model,
            "origin": int(origin),
            "actual_try": actual,
            "predicted_try": predicted,
            "error_try": error,
            "ape_pct": abs(error) / actual * 100 if actual else float("nan"),
            "direction": "over" if error > 0 else ("under" if error < 0 else "exact"),
        })
    # Named columns so that "no origin carried every horizon" returns an empty frame of
    # the right shape rather than raising on the sort. Reachable whenever the evaluation
    # window is shorter than the cumulative window.
    table = pd.DataFrame(out, columns=["model", "origin", "actual_try", "predicted_try",
                                       "error_try", "ape_pct", "direction"])
    return table.sort_values(["model", "origin"]).reset_index(drop=True)


def cumulative_risk(cumulative: pd.DataFrame) -> pd.DataFrame:
    """Per model: how wrong the five-week cash plan was, and how badly on the high side.

    ``n_scored``       origins that ``mape_5w`` was averaged over. Equal to ``n_origins``
                       unless a five-week window netted to zero, which has a cash error
                       but no percentage error.
    ``mape_5w``        mean absolute percentage error of the five-week total.
    ``bias_5w_pct``    signed, as a share of the five-week actual. Positive is over.
    ``over_rate_pct``  share of origins whose five-week plan over-forecast cash.
    ``p95_over_try``   the 95th percentile of the over-forecast shortfall.

    **`p95_over_try` is the liquidity number.** It answers "in a bad but not
    unprecedented five-week window, how much cash did the plan expect that did not
    arrive", which is the question a buffer is sized against. The mean cannot answer it,
    and MAPE cannot even see the direction.

    With ~50 origins the 95th percentile sits among the top two or three observations, so
    it is an indication rather than a reliable quantile. Report it as such.
    """
    out = []
    for model, group in cumulative.groupby("model", observed=True):
        error = group["error_try"].to_numpy(dtype=float)
        actual = group["actual_try"].to_numpy(dtype=float)
        over = error[error > 0]
        out.append({
            "model": model,
            "n_origins": len(group),
            # `mape_5w` is a skipna mean: an origin whose five weeks net to zero has no
            # percentage error and drops out of it, while still being a real origin with
            # a real cash error. Reporting both counts keeps `n_origins` from being read
            # as the MAPE's sample size when the two differ.
            "n_scored": int(group["ape_pct"].notna().sum()),
            "mape_5w": round(float(group["ape_pct"].mean()), 2),
            "bias_5w_pct": round(float(error.sum() / actual.sum() * 100), 1)
            if actual.sum() else float("nan"),
            "over_rate_pct": round(float(len(over) / len(error) * 100), 1),
            "mean_over_try": round(float(over.mean()), 0) if len(over) else 0.0,
            "p95_over_try": round(float(np.percentile(over, 95)), 0) if len(over) else 0.0,
            "worst_over_try": round(float(over.max()), 0) if len(over) else 0.0,
        })
    return pd.DataFrame(out).sort_values("mape_5w").reset_index(drop=True)
