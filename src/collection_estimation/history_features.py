"""Section 9.1 — the customer-history block.

One row per ``(customer, origin week)``. The history at origin `t` does not depend on the
horizon, so it is computed once here and joined to all five panel rows.

That is a correctness property as much as an efficiency one: this module never sees
``target_week``, so it cannot use it by accident. 185k rows instead of 857k is the bonus.

**Missing values are NaN, never a sentinel.** A customer with one payment has no
inter-payment gap, and encoding that as 0 or -1 is a lie a tree will happily learn. NaN is
supported by every estimator in play — scikit-learn 1.9's RandomForest and
HistGradientBoosting both accept it, and LightGBM splits on missingness natively. Paired
indicators (`has_gap`, `has_paid`) let a model separate "unknown" from "small".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import first_week

#: Rolling window lengths, in weeks.
#:
#: The design lists 52 as well. It is dropped: with 65 weeks a 52-week window is only
#: fully populated from week 52 onward, so for most origins it is simply "all history so
#: far" wearing a fixed-window label. The `lifetime_*` columns say that honestly instead.
WINDOWS = (4, 8, 13, 26)


def _dense_grid(customer_weeks: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """One row per (customer, week) from the customer's first appearance to the end.

    **Dense, not sparse, and this is load-bearing.** `n_4` means "how many of the last four
    weeks had a payment", which needs the silent weeks present. Rolling over the sparse
    table would count four *payments* rather than four *weeks* — silently wrong for
    precisely the inactive customers who dominate the panel.

    It starts at the first appearance rather than at week 0, so weeks before a customer
    existed are not counted as weeks they failed to pay.
    """
    first = first_week(customer_weeks)
    starts = first.to_numpy()
    span = n_weeks - starts

    grid = pd.DataFrame({
        "customer": np.repeat(first.index.to_numpy(), span),
        "week": np.concatenate([np.arange(s, n_weeks) for s in starts]),
    })
    grid = grid.merge(
        customer_weeks[["customer", "week", "amount_positive", "amount_negative"]],
        on=["customer", "week"], how="left")
    grid[["amount_positive", "amount_negative"]] = grid[
        ["amount_positive", "amount_negative"]].fillna(0.0)
    return grid.sort_values(["customer", "week"]).reset_index(drop=True)


def build_history_features(customer_weeks: pd.DataFrame,
                           weeks: pd.DataFrame) -> pd.DataFrame:
    """Per-customer history, as known at the end of each origin week.

    Every window is ``(t-W, t]`` — **inclusive of week t**, which is observed at the
    origin. This is where leakage would enter: `(t-W, t)` throws away a week, and
    `(t-W, t+1]` leaks the target. `test_no_feature_at_t_depends_on_week_t_plus_one`
    pins it.

    Columns, all `float64` so NaN is representable:

    ``n_W, sum_W, mean_W, max_W``   over the last W weeks, W in {4, 8, 13, 26}
    ``lifetime_n / sum / mean / max``
    ``rec``        weeks since the last POSITIVE payment; NaN before the first one
    ``has_paid``   0/1 — has this customer made a positive payment yet
    ``age``        weeks since first seen
    ``active``     lifetime_n / (age + 1), the share of weeks ever paid in
    ``gap_mean / gap_sd / gap_median``   inter-payment gaps seen so far; NaN if none
    ``has_gap``    0/1 — are there at least two payments to form a gap from
    ``phase``      ``rec mod gap_median``: position in the payment cycle
    ``trend``      ``mean_8`` over ``mean_8`` eight weeks ago; NaN if that was zero
    ``n_refund_26 / sum_refund_26 / lifetime_n_refunds``

    Windows are computed over whatever history exists rather than blanked while short.
    `age` is there so a model can learn to discount a mean built from three weeks; dropping
    early origins instead would throw away most of the panel.
    """
    grid = _dense_grid(customer_weeks, len(weeks))
    grid["paid"] = (grid["amount_positive"] > 0).astype(float)
    grid["refunded"] = (grid["amount_negative"] < 0).astype(float)

    by_customer = grid.groupby("customer", observed=True, sort=False)

    def rolled(column: str, window: int, how: str) -> pd.Series:
        series = by_customer[column].rolling(window, min_periods=1)
        return getattr(series, how)().reset_index(level=0, drop=True)

    def expanded(column: str, how: str) -> pd.Series:
        series = by_customer[column].expanding(min_periods=1)
        return getattr(series, how)().reset_index(level=0, drop=True)

    out = grid[["customer", "week"]].copy()

    for window in WINDOWS:
        out[f"n_{window}"] = rolled("paid", window, "sum")
        out[f"sum_{window}"] = rolled("amount_positive", window, "sum")
        out[f"mean_{window}"] = rolled("amount_positive", window, "mean")
        out[f"max_{window}"] = rolled("amount_positive", window, "max")

    out["lifetime_n"] = expanded("paid", "sum")
    out["lifetime_sum"] = expanded("amount_positive", "sum")
    out["lifetime_mean"] = expanded("amount_positive", "mean")
    out["lifetime_max"] = expanded("amount_positive", "max")

    # Weeks since the last positive payment. NaN until the first one, which is possible
    # because a customer's first appearance may be a refund-only week.
    pay_week = grid["week"].where(grid["paid"] > 0)
    last_pay = pay_week.groupby(grid["customer"], observed=True, sort=False).ffill()
    out["rec"] = grid["week"] - last_pay
    out["has_paid"] = last_pay.notna().astype(float)

    out["age"] = grid["week"] - by_customer["week"].transform("min")
    out["active"] = out["lifetime_n"] / (out["age"] + 1.0)

    # Inter-payment gaps. At a paying week, the previous paying week is the value
    # `last_pay` held on the ROW BEFORE — shifting the ffilled column gives exactly that.
    previous_pay = last_pay.groupby(grid["customer"], observed=True,
                                    sort=False).shift()
    grid["gap"] = np.where(grid["paid"] > 0, grid["week"] - previous_pay, np.nan)

    out["gap_mean"] = expanded("gap", "mean")
    out["gap_sd"] = expanded("gap", "std")
    out["gap_median"] = expanded("gap", "median")
    out["has_gap"] = out["gap_median"].notna().astype(float)

    # Where in its own cycle is this customer? Without this a tree has to reconstruct
    # periodicity from `rec` alone, which it does badly.
    cycle = out["gap_median"].where(out["gap_median"] > 0)
    out["phase"] = out["rec"] % cycle

    previous_mean_8 = by_customer["amount_positive"].transform(
        lambda s: s.rolling(8, min_periods=1).mean().shift(8))
    out["trend"] = np.where(previous_mean_8 > 0,
                            out["mean_8"] / previous_mean_8, np.nan)

    out["n_refund_26"] = rolled("refunded", 26, "sum")
    out["sum_refund_26"] = rolled("amount_negative", 26, "sum")
    out["lifetime_n_refunds"] = expanded("refunded", "sum")

    numeric = [c for c in out.columns if c not in ("customer", "week")]
    out[numeric] = out[numeric].astype("float64")
    out["week"] = out["week"].astype("int16")
    return out


def attach_to_panel(panel: pd.DataFrame,
                    history: pd.DataFrame) -> pd.DataFrame:
    """Join history onto the panel by ``(customer, week)`` — the ORIGIN week.

    Never by `target_week`. Joining on the target would hand the model the history of the
    week it is trying to predict, which is the most direct leak available here.
    """
    merged = panel.merge(history, on=["customer", "week"], how="left",
                         validate="many_to_one")
    if len(merged) != len(panel):
        raise ValueError("the history join changed the panel's row count")
    return merged
