"""The week spine and the customer-week aggregation.

Algorithm 1 of the design begins "aggregate R to customer-weeks". This module is that
line, plus the week index everything downstream is addressed by.

The dense (customer, week, horizon) panel that used to live here is **parked** — only the
per-customer hurdle model needs it. See `parked/hurdle_panel.py`.

See `weekly_collection_forecast_design.md` in the cockpit repo, sections 1, 3.1 and 4.
"""
from __future__ import annotations

from datetime import timedelta

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
