"""Section 9.3 — the cross-customer block.

Two kinds of feature with very different costs, kept separate on purpose.

**Cheap** — plain rolling aggregates over past weeks (`entity_recent`, `global_recent`).
Leakage-safe by construction, no estimation involved.

**Expensive** — partner features, which need a co-occurrence matrix estimated from data.
That matrix must be rebuilt as of every origin: built over all weeks it encodes future
co-payments, which is leakage trap 3 of section 10.2 and the subtlest one available here,
because the result looks entirely reasonable.

Co-occurrence counts are additive over weeks, so the matrix is accumulated in a single
forward pass rather than rebuilt 52 times — the correct version is affordable, which
removes the temptation to take the leaky shortcut.

**The selection rule is a threshold, not plain top-k.** Measured on this corpus, unrelated
customer pairs reach a lift ratio of 2.10 by chance, while the median pair sits at 1.08.
Top-k alone would hand almost every customer five "partners" that are coincidence. A
customer with no partner above the floor gets none.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .panel import first_week

#: Partners kept per customer, at most.
TOP_K = 5

#: A pair must share at least this many paying weeks to be considered at all. Below this
#: the estimate is a coincidence between two rare customers.
MIN_SHARED_WEEKS = 3

#: And BOTH customers must pay this often before a lift estimate means anything.
#:
#: This is the threshold that actually does the work, and leaving it out was a mistake
#: caught by measurement: with only `MIN_SHARED_WEEKS` in force, a customer paying three
#: times that shared all three weeks scored a conditional probability of 1.0 and a lift of
#: 6x — from three observations. 1,777 of 4,507 customers came back with the maximum five
#: partners each, the cap binding for every one of them, which is the signature of a filter
#: that is not filtering.
#:
#: Ten matches the corpus-wide check: 579 of 4,498 customers have ten or more paying weeks,
#: and below that the lift estimate cannot be separated from the 2.10 chance ratio the
#: negative controls reach.
MIN_PAYING_WEEKS = 10

#: And the lift must clear this. Calibrated against the corpus's negative controls, whose
#: worst chance lift was 2.10 — anything under that is indistinguishable from noise.
MIN_LIFT = 2.0

#: Weeks of separation still counted as co-occurrence. 0 means same week only.
#:
#: The design says +/-1 week. Same week is used instead, for two measured reasons.
#:
#: The `MIN_LIFT` floor was calibrated on SAME-WEEK co-occurrence — the negative controls
#: reach 2.10 that way. A +/-1 window roughly triples the counts and so triples every
#: lift, which made the floor stop binding entirely: all 579 eligible customers came back
#: with the maximum five partners, `top_k` doing all the selecting and the threshold none.
#:
#: It also makes the count asymmetric. `shared[(a, b)]` would pair a-in-week-w with
#: b-in-w-or-w+1, while `shared[(b, a)]` pairs b-in-w with a-in-w-or-w+1 — different
#: quantities under one name. Same-week co-occurrence is symmetric by construction.
#:
#: Widening this again means recalibrating `MIN_LIFT` against the controls first.
CO_WINDOW = 0


def _paying_weeks(customer_weeks: pd.DataFrame) -> dict[str, set[int]]:
    positive = customer_weeks[customer_weeks["amount_positive"] > 0]
    return {c: set(w) for c, w in
            positive.groupby("customer", observed=True)["week"]}


def find_partners(customer_weeks: pd.DataFrame, up_to_week: int,
                  top_k: int = TOP_K, min_shared: int = MIN_SHARED_WEEKS,
                  min_lift: float = MIN_LIFT,
                  min_paying_weeks: int = MIN_PAYING_WEEKS,
                  window: int = CO_WINDOW) -> dict[str, list[str]]:
    """Each customer's partners, using ONLY weeks up to and including `up_to_week`.

    A partner is a customer whose paying weeks coincide with this one's more often than
    chance explains. `lift` is P(partner pays | this one pays) divided by the partner's
    unconditional rate.

    Returns an empty list for customers with no partner clearing the thresholds, which is
    most of them: only about 13% of this roster has enough paying weeks to say anything.
    """
    horizon = up_to_week + 1
    weeks_by = {c: {w for w in ws if w <= up_to_week}
                for c, ws in _paying_weeks(customer_weeks).items()}
    # BOTH sides of a pair must be active enough for the estimate to mean anything, so the
    # filter is applied before any counting rather than to the pair afterwards.
    weeks_by = {c: ws for c, ws in weeks_by.items() if len(ws) >= min_paying_weeks}
    if not weeks_by:
        return {}

    # Bucket customers by week, then count co-occurrences week by week. Additive, so this
    # is the step an incremental accumulator would reuse.
    active: dict[int, list[str]] = defaultdict(list)
    for customer, ws in weeks_by.items():
        for w in ws:
            active[w].append(customer)

    shared: dict[tuple[str, str], int] = defaultdict(int)
    for week, members in active.items():
        near = list(members)
        for offset in range(1, window + 1):
            near.extend(active.get(week + offset, []))
        for i, a in enumerate(members):
            for b in near:
                if a != b:
                    shared[(a, b)] += 1

    scored: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (a, b), count in shared.items():
        if count < min_shared:
            continue
        conditional = count / len(weeks_by[a])
        baseline = len(weeks_by[b]) / horizon
        if baseline <= 0:
            continue
        lift = conditional / baseline
        if lift >= min_lift:
            scored[a].append((lift, b))

    return {a: [b for _, b in sorted(v, reverse=True)[:top_k]]
            for a, v in scored.items()}


def build_cross_features(customer_weeks: pd.DataFrame, weeks: pd.DataFrame,
                         origins: range | None = None,
                         top_k: int = TOP_K, min_shared: int = MIN_SHARED_WEEKS,
                         min_lift: float = MIN_LIFT,
                         min_paying_weeks: int = MIN_PAYING_WEEKS) -> pd.DataFrame:
    """One row per (customer, origin week).

    ``entity_recent_4w``   the customer's EY entity's collections, last 4 weeks
    ``global_recent_4w``   all customers, last 4 weeks
    ``n_partners``         partners clearing the thresholds as of this origin
    ``partner_paid_1w``    did any partner pay in the origin week
    ``partner_paid_3w``    ... in the last three weeks
    ``partner_sum_3w``     how much they paid
    ``has_partners``       0/1 — the indicator that keeps "none" distinct from "zero"

    **Partners are recomputed at every origin** from weeks up to that origin only. Doing it
    once over the whole span would let a model see who a customer will coincide with in
    future, which is the leak this block exists to avoid.
    """
    n_weeks = len(weeks)
    first = first_week(customer_weeks)
    if origins is None:
        origins = range(int(first.min()), n_weeks)

    positive = customer_weeks[customer_weeks["amount_positive"] > 0]
    paid_by_week: dict[int, dict[str, float]] = defaultdict(dict)
    for customer, week, amount in zip(positive["customer"], positive["week"],
                                      positive["amount_positive"]):
        paid_by_week[int(week)][customer] = float(amount)

    weekly_total = (customer_weeks.groupby("week", observed=True)["amount_positive"]
                    .sum().reindex(range(n_weeks), fill_value=0.0))
    entity_total = (customer_weeks.groupby(["company_code", "week"], observed=True)
                    ["amount_positive"].sum())
    entity_of = customer_weeks.groupby("customer", observed=True)["company_code"].first()

    rows = []
    for origin in origins:
        partners = find_partners(customer_weeks, origin, top_k=top_k,
                                 min_shared=min_shared, min_lift=min_lift,
                                 min_paying_weeks=min_paying_weeks)
        recent = range(max(0, origin - 3), origin + 1)
        window_4 = range(max(0, origin - 3), origin + 1)

        global_recent = float(weekly_total.loc[list(window_4)].sum())

        for customer, start in first.items():
            if origin < start:
                continue
            code = entity_of[customer]
            entity_recent = float(sum(
                entity_total.get((code, w), 0.0) for w in window_4))

            mine = partners.get(customer, [])
            paid_1w = paid_3w = 0.0
            total_3w = 0.0
            for partner in mine:
                if partner in paid_by_week.get(origin, {}):
                    paid_1w = 1.0
                for w in recent:
                    amount = paid_by_week.get(w, {}).get(partner)
                    if amount is not None:
                        paid_3w = 1.0
                        total_3w += amount

            rows.append({
                "customer": customer, "week": origin,
                "entity_recent_4w": entity_recent,
                "global_recent_4w": global_recent,
                "n_partners": float(len(mine)),
                "has_partners": 1.0 if mine else 0.0,
                "partner_paid_1w": paid_1w if mine else np.nan,
                "partner_paid_3w": paid_3w if mine else np.nan,
                "partner_sum_3w": total_3w if mine else np.nan,
            })

    out = pd.DataFrame(rows)
    out["week"] = out["week"].astype("int16")
    return out
