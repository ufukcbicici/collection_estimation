"""Weekly regressors derived from the per-customer blocks.

The bridge from the parked hurdle model: collapse per-customer cadence into a handful of
weekly numbers, without fitting 4,500 models.

``n_due``            customers whose cadence says a payment is due in the target week
``expected_amount``  those customers' typical payment sizes, summed
``active_base``      customers with any payment in the last 8 weeks

**Indexed by (origin, horizon), not by week.** These describe the TARGET week but are
computed from customer history, so they must be computed as of the origin. Building them
once over all weeks would let a model see who paid when it is trying to predict whether
they will — the same trap as the cross-customer block, and the same fix.

--------------------------------------------------------------------------------------
NEGATIVE RESULT: these do not improve the weekly forecast. Kept for the record.
--------------------------------------------------------------------------------------

Measured 2026-09-09. They carry real signal — `n_due` correlates +0.45 with the target
week's total and `expected_amount` +0.47 — but the signal is **the trend**, not cadence:

    correlation with the week index
      n_due              +0.942
      expected_amount    +0.975
      active_base        +0.624

They SUBSTITUTE for a linear trend rather than adding to it, and using both is worse:

    h=1                          MAPE   se     bias
    calendar + trend            19.83  2.49   -1.0%
    calendar + derived          19.66  2.49   -2.8%   equivalent, within noise
    calendar + trend + derived  20.98  2.60   -0.6%   collinear, worse

The cause is in the construction. `n_due` counts customers *eligible* to be due, and
eligibility requires three payments, so the count grows monotonically as history
accumulates. It measures corpus maturity, not this week's cadence.

Dividing by the active base does not fix it — `due_share` still correlates +0.943 with
time, because the eligible population grows faster than the active base does. Three
parameterisations were tried (counts, values, shares) and all three say the same thing.

**Caveat on how far this generalises.** Sixty-five weeks is short enough that the customer
population never stabilises; on a longer real series the eligible base would flatten and
cadence might separate from trend. This is a result about this corpus, not a proof that
cadence is useless.

Given a linear trend is one regressor with no per-customer machinery behind it, prefer the
trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Tolerance, in weeks, on "due". A customer with a four-week cadence last seen three
#: weeks ago counts as due now. Zero tolerance makes the count almost always zero and the
#: regressor useless; a wide one makes everybody due and the regressor constant.
DUE_TOLERANCE = 1

#: A customer needs this many payments before its cadence means anything. Same reasoning
#: as the cross-customer block, where a three-payment customer produced a lift of 6.
MIN_PAYMENTS = 3

#: Window for `active_base`, in weeks.
ACTIVE_WINDOW = 8

REGRESSOR_COLUMNS = ("n_due", "expected_amount", "active_base")


def build_derived_regressors(customer_weeks: pd.DataFrame, weeks: pd.DataFrame,
                             horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
                             tolerance: int = DUE_TOLERANCE,
                             min_payments: int = MIN_PAYMENTS
                             ) -> dict[tuple[int, int], np.ndarray]:
    """``{(origin, horizon): array of regressors}``, using only weeks <= origin.

    "Due" is modular rather than one-shot: a customer with a four-week cadence last seen
    at week 10 is due at 14, 18, 22 and so on, so a five-week horizon can contain a second
    cycle for frequent payers.
    """
    n_weeks = len(weeks)
    positive = customer_weeks[customer_weeks["amount_positive"] > 0]

    # Per customer: the weeks it paid in, and its typical payment.
    by_customer = {
        customer: (group["week"].to_numpy(), group["amount_positive"].to_numpy())
        for customer, group in positive.groupby("customer", observed=True)
    }

    out: dict[tuple[int, int], np.ndarray] = {}
    for origin in range(n_weeks):
        # Cadence as known at this origin.
        cadence = []
        active = 0
        for customer, (weeks_paid, amounts) in by_customer.items():
            seen = weeks_paid <= origin
            count = int(seen.sum())
            if count == 0:
                continue
            last = int(weeks_paid[seen].max())
            if last > origin - ACTIVE_WINDOW:
                active += 1
            if count < min_payments:
                continue
            gaps = np.diff(weeks_paid[seen])
            gap = float(np.median(gaps))
            if gap <= 0:
                continue
            cadence.append((last, gap, float(np.mean(amounts[seen]))))

        for h in horizons:
            target = origin + h
            if target >= n_weeks:
                continue
            n_due = 0
            expected = 0.0
            for last, gap, typical in cadence:
                elapsed = target - last
                if elapsed <= 0:
                    continue
                # Distance to the nearest multiple of the cadence.
                offset = elapsed % gap
                distance = min(offset, gap - offset)
                if distance <= tolerance:
                    n_due += 1
                    expected += typical
            out[(origin, h)] = np.array([float(n_due), expected, float(active)])

    return out


def derived_matrix(derived: dict, origin: int, h: int) -> np.ndarray:
    """Lookup with a zero fallback, so a missing (origin, horizon) cannot crash a fit."""
    return derived.get((origin, h), np.zeros(len(REGRESSOR_COLUMNS)))
