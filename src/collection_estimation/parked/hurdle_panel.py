"""The dense (customer, week, horizon) panel — Algorithm 1, lines 5-11.

PARKED. Split out of `panel.py` when the pipeline was trimmed for handover: the weekly
model needs only the week spine and the customer-week aggregation, both of which stayed.
This is the part only the per-customer hurdle model uses.

Nothing on the production path imports this. It is not dead — it is unrun.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Forecast horizons, in weeks. The design predicts t+1 .. t+5.
HORIZONS = (1, 2, 3, 4, 5)


def first_week(customer_weeks: pd.DataFrame) -> pd.Series:
    """Each customer's first observed week, indexed by customer.

    "Observed" means any collection row, positive or negative — the week we first have
    evidence the customer exists. A customer whose first appearance happens to be a
    refund still becomes predictable from that point.
    """
    return customer_weeks.groupby("customer", observed=True)["week"].min()


def build_panel(customer_weeks: pd.DataFrame, weeks: pd.DataFrame,
                horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Algorithm 1, lines 5-11: one row per (customer, origin week, horizon).

    The **skeleton only** — keys and targets. Features (sections 9.1 to 9.3) are joined
    on afterwards, by ``(customer, week)`` for the history block and by ``target_week``
    for the calendar block. Computing them here would fuse two concerns and make both
    harder to test.

    Columns:

    ``customer``, ``week``   the customer and the ORIGIN week t
    ``horizon``              h, in weeks
    ``target_week``          t + h
    ``company_code``         carried through for grouping and diagnostics
    ``y``                    positive amount collected at (customer, t+h); 0 if none
    ``z``                    1 if y > 0, else 0 — the stage 1 target
    ``y_negative``           refunds at t+h, kept negative; 0 if none

    Three decisions worth knowing:

    **Rows start at each customer's first observed week** (section 5). A customer never
    seen has no history to build features from, and a row existing before their first
    appearance leaks the fact that they will eventually appear. This takes the panel from
    292,955 customer-weeks to 184,860 and lifts the base rate from 6.99% to 11.08%.
    Genuinely new customers are the separate ``Nhat`` term of (3.2), not a panel row.

    **Churned customers keep their rows to the end.** We cannot know at time t that a
    customer has stopped for good, and their zeros are real training signal.

    **The panel is RAGGED**, and this departs from the written line 6. The algorithm says
    ``t <= T - max(H)``, which stops every origin five weeks early so all horizons exist
    for each. Instead any ``(t, h)`` with ``t + h <= T`` is kept, because Algorithm 2
    already filters to ``t + h <= tj`` at each origin — enforcing it twice would discard
    rows the evaluation would have used, and h=1 would lose four usable origins for
    nothing. Horizon 1 therefore has slightly more rows than horizon 5, which does not
    distort evaluation because each horizon is scored on its own origins.

    **The target is the POSITIVE amount, not the net.** Section 3.1 defines ``z`` on
    positive cells and stage 2 needs strictly positive support. With the net, a
    customer-week holding a 1000 collection and a 1500 refund would carry ``z = 0`` —
    recorded as "did not pay" when they plainly did.

    **A consequence worth stating: the panel's base rate is LOWER than the cell base
    rate**, ~8.9% against the 11.08% the design quotes for customer-weeks. That is not a
    defect. A customer's first appearance can never be a target, because the origin would
    have to precede it — and 42% of this roster appears exactly once, so for those
    customers the single positive cell sits at ``first[c]`` and is unreachable. Predicting
    a customer's first-ever payment from their own history is impossible by construction;
    it is what the separate ``Nhat`` term exists for.

    For the same reason a customer first seen in the very last week gets **no rows at
    all** — no origin at or after their first appearance has a target inside the data.
    """
    n_weeks = len(weeks)
    first = first_week(customer_weeks)
    customers = first.index.to_numpy()
    starts = first.to_numpy()

    # Dense (customer, origin week) grid, built with repeat rather than a Python loop:
    # ~4,500 customers x ~40 weeks x 5 horizons is a few hundred thousand iterations of
    # pandas indexing otherwise, which takes minutes instead of about a second.
    span = n_weeks - starts                       # weeks from first appearance to the end
    grid = pd.DataFrame({
        "customer": np.repeat(customers, span),
        "week": np.concatenate([np.arange(s, n_weeks) for s in starts]),
    })

    # Cross with the horizons, then drop targets beyond the end of the observed span.
    panel = grid.loc[grid.index.repeat(len(horizons))].reset_index(drop=True)
    panel["horizon"] = np.tile(horizons, len(grid))
    panel["target_week"] = panel["week"] + panel["horizon"]
    panel = panel[panel["target_week"] < n_weeks].reset_index(drop=True)

    # Targets, by joining the sparse aggregation onto the target week. Anything with no
    # matching customer-week collected nothing, which is a real zero and not missing.
    targets = customer_weeks[["customer", "week", "amount_positive",
                              "amount_negative"]].rename(
        columns={"week": "target_week", "amount_positive": "y",
                 "amount_negative": "y_negative"})
    panel = panel.merge(targets, on=["customer", "target_week"], how="left")
    panel[["y", "y_negative"]] = panel[["y", "y_negative"]].fillna(0.0)
    panel["z"] = (panel["y"] > 0).astype("int8")

    entity = customer_weeks.groupby("customer", observed=True)["company_code"].first()
    panel["company_code"] = panel["customer"].map(entity).astype("string")

    panel["customer"] = panel["customer"].astype("category")
    panel["week"] = panel["week"].astype("int16")
    panel["target_week"] = panel["target_week"].astype("int16")
    panel["horizon"] = panel["horizon"].astype("int8")

    return panel[["customer", "week", "horizon", "target_week", "company_code",
                  "y", "y_negative", "z"]]
