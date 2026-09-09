"""The week spine and the customer-week aggregation.

Algorithm 1 of the design begins "aggregate R to customer-weeks". This module is that
line, plus the week index everything downstream is addressed by.

See `weekly_collection_forecast_design.md` in the cockpit repo, sections 1, 3.1 and 4.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .config import STATUS_NOTES


def build_week_index(collections: pd.DataFrame) -> pd.DataFrame:
    """The contiguous week spine every later step is indexed on.

    One row per ISO week between the first and last collection, with:

    ``week``        integer, 0-based, contiguous — the index we do arithmetic on
    ``iso_week``    label such as ``2025-W23`` — for printing, never for arithmetic
    ``week_start``  the Monday
    ``week_end``    the Sunday

    **Why an integer and not the ISO label.** The design writes ``t + h`` throughout, and
    ISO labels do not add. `2025-W52` plus one is not `2025-W53`: some ISO years have 52
    weeks and some 53, and the label rolls over to `2026-W01`. Code doing arithmetic on
    labels is wrong at exactly one point in this corpus and silently right everywhere
    else, which is the worst way to be wrong.

    **Why it is built from the date range, not from the observed weeks.** A week in which
    nothing at all was collected still exists, and dropping it would shift every ``t + h``
    after it by one. Building the spine from ``collections["document_date"].unique()``
    would do exactly that.
    """
    if collections.empty:
        raise ValueError("cannot build a week index from an empty table")

    first = pd.Timestamp(collections["document_date"].min())
    last = pd.Timestamp(collections["document_date"].max())

    # Back up to the Monday of the first week and forward to the Sunday of the last, so
    # the spine covers whole ISO weeks at both ends.
    start = (first - timedelta(days=int(first.dayofweek))).normalize()
    end = (last + timedelta(days=6 - int(last.dayofweek))).normalize()

    starts = pd.date_range(start, end, freq="W-MON" if start.dayofweek == 0 else "7D")
    if starts.empty or starts[0] != start:
        starts = pd.date_range(start, end, freq="7D")

    weeks = pd.DataFrame({"week_start": starts})
    weeks["week_end"] = weeks["week_start"] + timedelta(days=6)
    iso = weeks["week_start"].dt.isocalendar()
    weeks["iso_week"] = (iso["year"].astype(str) + "-W"
                         + iso["week"].astype(int).astype(str).str.zfill(2))
    weeks.insert(0, "week", range(len(weeks)))
    return weeks[["week", "iso_week", "week_start", "week_end"]]


def assign_week(collections: pd.DataFrame, weeks: pd.DataFrame) -> pd.Series:
    """The integer week index of each collection row.

    Computed as whole weeks since the spine's first Monday, so it cannot disagree with
    `build_week_index` and needs no join.
    """
    origin = weeks["week_start"].iloc[0]
    days = (collections["document_date"] - origin).dt.days
    if (days < 0).any():
        raise ValueError("a collection falls before the start of the week index")
    index = days // 7
    if (index >= len(weeks)).any():
        raise ValueError("a collection falls after the end of the week index")
    return index.astype("int64")


def to_customer_weeks(collections: pd.DataFrame,
                      weeks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Line 1 of Algorithm 1: aggregate collections to one row per customer-week.

    **Sparse** — only customer-weeks in which something happened. Expanding to the dense
    panel is a later step, and doing it here would produce a table two orders of magnitude
    larger for no gain.

    Columns:

    ``customer``, ``week``      the key
    ``company_code``            the customer's entity (see the note below)
    ``amount_positive``         sum of the positive rows
    ``amount_negative``         sum of the negative rows, kept NEGATIVE
    ``amount_net``              the two added
    ``n_rows``                  how many collection rows, positive or negative
    ``n_unapplied``             how many were unapplied cash

    **Positives and negatives are kept apart, not netted.** Section 3.1 defines the
    classification target as ``z = 1[y > 0]`` on positive cells and handles refunds as
    their own term, because they break the positive support the amount model needs.
    Netting first destroys that distinction irreversibly — and it is not hypothetical: 92
    customer-weeks in this corpus have a negative net, so ``amount_net > 0`` is not the
    same question as "did this customer collect anything".

    ``n_unapplied`` counts rows carrying a Turkish status note instead of an invoice
    number — money received but not yet matched to a receivable. They are ordinary money
    and are included in the amounts; the count is kept because a customer-week that is
    entirely unapplied cash may behave differently, and we cannot tell without measuring.
    """
    if weeks is None:
        weeks = build_week_index(collections)

    working = collections.copy()
    working["week"] = assign_week(working, weeks)

    amount = working["amount_local"]
    working["amount_positive"] = amount.where(amount > 0, 0.0)
    working["amount_negative"] = amount.where(amount < 0, 0.0)
    working["is_unapplied"] = working["government_invoice"].isin(STATUS_NOTES)

    grouped = working.groupby(["customer", "week"], observed=True, sort=True).agg(
        company_code=("company_code", "first"),
        amount_positive=("amount_positive", "sum"),
        amount_negative=("amount_negative", "sum"),
        n_rows=("amount_local", "size"),
        n_unapplied=("is_unapplied", "sum"),
    ).reset_index()

    grouped["amount_net"] = grouped["amount_positive"] + grouped["amount_negative"]
    grouped["n_unapplied"] = grouped["n_unapplied"].astype("int64")

    return grouped[["customer", "week", "company_code", "amount_positive",
                    "amount_negative", "amount_net", "n_rows", "n_unapplied"]]


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


def weeks_without_collections(customer_weeks: pd.DataFrame,
                              weeks: pd.DataFrame) -> list[int]:
    """Weeks in the spine that no customer collected in.

    Reported rather than hidden. A week with no collections at all is either a genuine
    lull or a sign that the corpus is missing files for all of its days — and those are
    different facts. EY's own generation notes insist a missing source day must not be
    read as zero cash flow.
    """
    seen = set(customer_weeks["week"].unique())
    return [int(w) for w in weeks["week"] if w not in seen]
