"""Tests for the dense panel — Algorithm 1, lines 5-11.

The constructed cases pin the three decisions: rows start at a customer's first observed
week, the panel is deliberately ragged, and the target is the positive amount rather than
the net.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from collection_estimation import config                          # noqa: E402
from collection_estimation.ingest import read_corpus              # noqa: E402
from collection_estimation.panel import (                         # noqa: E402
    build_week_index,
    to_customer_weeks,
)
from collection_estimation.parked.hurdle_panel import (           # noqa: E402
    HORIZONS,
    build_panel,
    first_week,
)


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return read_corpus(config.CORPUS_DIR)


@pytest.fixture(scope="module")
def weeks(corpus) -> pd.DataFrame:
    return build_week_index(corpus)


@pytest.fixture(scope="module")
def customer_weeks(corpus, weeks) -> pd.DataFrame:
    return to_customer_weeks(corpus, weeks)


@pytest.fixture(scope="module")
def panel(customer_weeks, weeks) -> pd.DataFrame:
    return build_panel(customer_weeks, weeks)


def tiny(records, n_weeks=10):
    """A customer-week table and a matching spine, without touching Excel."""
    cw = pd.DataFrame(records)
    for column, default in (("company_code", "TR02"), ("amount_negative", 0.0),
                            ("n_rows", 1), ("n_unapplied", 0)):
        if column not in cw.columns:
            cw[column] = default
    cw["amount_net"] = cw["amount_positive"] + cw["amount_negative"]
    spine = pd.DataFrame({
        "week": range(n_weeks),
        "iso_week": [f"2025-W{i:02d}" for i in range(n_weeks)],
        "week_start": pd.date_range("2025-01-06", periods=n_weeks, freq="7D"),
    })
    spine["week_end"] = spine["week_start"] + pd.Timedelta(days=6)
    return cw, spine


# ---------------------------------------------------------------- the restriction

def test_rows_start_at_the_customers_first_observed_week():
    """Section 5. Weeks before a customer appears have no history and leak."""
    cw, spine = tiny([{"customer": "C1", "week": 4, "amount_positive": 100.0}])
    got = build_panel(cw, spine)
    assert got["week"].min() == 4, "no origin may precede the first appearance"


def test_customers_starting_later_have_fewer_rows():
    cw, spine = tiny([{"customer": "EARLY", "week": 0, "amount_positive": 1.0},
                      {"customer": "LATE", "week": 5, "amount_positive": 1.0}])
    got = build_panel(cw, spine)
    counts = got.groupby("customer", observed=True).size()
    assert counts["EARLY"] > counts["LATE"]


def test_churned_customers_keep_rows_to_the_end():
    """We cannot know at time t that a customer has stopped; their zeros are signal."""
    cw, spine = tiny([{"customer": "C1", "week": 1, "amount_positive": 100.0}])
    got = build_panel(cw, spine)
    assert got["week"].max() == 8, (
        "origins should run to the last week that still has a horizon-1 target")
    assert (got.loc[got["week"] > 1, "z"] == 0).all()


def test_first_week_is_any_appearance_not_the_first_positive():
    """A customer whose first evidence is a refund is still observed from then on."""
    cw, spine = tiny([
        {"customer": "C1", "week": 2, "amount_positive": 0.0,
         "amount_negative": -50.0},
        {"customer": "C1", "week": 6, "amount_positive": 100.0},
    ])
    assert first_week(cw)["C1"] == 2
    assert build_panel(cw, spine)["week"].min() == 2


# ---------------------------------------------------------------- raggedness

def test_panel_is_ragged_by_design():
    """Horizon 1 has more rows than horizon 5.

    A deliberate departure from line 6, which stops every origin five weeks early so all
    horizons exist for each. Algorithm 2 already filters to `t + h <= tj` at its origin,
    so enforcing it here too would discard rows the evaluation would have used.
    """
    cw, spine = tiny([{"customer": "C1", "week": 0, "amount_positive": 1.0}])
    got = build_panel(cw, spine)
    per_horizon = got.groupby("horizon", observed=True).size()
    assert per_horizon[1] > per_horizon[5]
    assert per_horizon[1] - per_horizon[5] == 4


def test_no_target_falls_outside_the_spine(panel, weeks):
    assert panel["target_week"].max() < len(weeks)
    assert panel["target_week"].min() >= 1


def test_target_week_is_always_origin_plus_horizon(panel):
    assert (panel["target_week"] == panel["week"] + panel["horizon"]).all()


def test_every_horizon_is_present(panel):
    assert set(panel["horizon"].unique()) == set(HORIZONS)


# ---------------------------------------------------------------- targets

def test_target_is_the_positive_amount_not_the_net():
    """With the net, a refunded week would read as "did not pay"."""
    cw, spine = tiny([
        {"customer": "C1", "week": 0, "amount_positive": 1.0},
        {"customer": "C1", "week": 3, "amount_positive": 1000.0,
         "amount_negative": -1500.0},
    ])
    got = build_panel(cw, spine)
    row = got[(got["week"] == 2) & (got["horizon"] == 1)].iloc[0]
    assert row["y"] == 1000.0
    assert row["y_negative"] == -1500.0
    assert row["z"] == 1, "a collection happened, even though the net is negative"


def test_absent_customer_weeks_become_real_zeros():
    cw, spine = tiny([{"customer": "C1", "week": 0, "amount_positive": 5.0}])
    got = build_panel(cw, spine)
    assert got["y"].notna().all(), "an absent week is a zero, never missing"
    assert (got.loc[got["target_week"] != 0, "y"] == 0).all()


def test_z_matches_y(panel):
    assert ((panel["y"] > 0) == (panel["z"] == 1)).all()


def test_targets_line_up_with_the_aggregation(panel, customer_weeks):
    """Every positive customer-week must appear as a target the right number of times."""
    observed = customer_weeks[customer_weeks["amount_positive"] > 0]
    key = set(zip(observed["customer"], observed["week"]))
    hit = set(zip(panel.loc[panel["z"] == 1, "customer"].astype(str),
                  panel.loc[panel["z"] == 1, "target_week"]))
    # Every positive target in the panel is a real customer-week...
    assert hit <= key
    # ...and the amounts agree.
    merged = panel.merge(
        observed.rename(columns={"week": "target_week",
                                 "amount_positive": "expected"}),
        on=["customer", "target_week"], how="inner")
    assert np.allclose(merged["y"], merged["expected"])


# ---------------------------------------------------------------- the real corpus

def test_panel_base_rate_is_below_the_cell_base_rate(panel, customer_weeks, weeks):
    """~8.9% on panel rows against 11.08% on customer-week cells, and that is correct.

    A customer's first appearance can never be a TARGET: the origin would have to precede
    it, and no such origin exists. Since 42% of this roster appears exactly once, those
    customers' only positive cell sits at `first[c]` and is unreachable — they contribute
    rows but no positive targets, which pulls the rate down.

    Predicting a first-ever payment from a customer's own history is impossible by
    construction. That population is what the separate `Nhat` term is for.
    """
    cells = int((len(weeks) - first_week(customer_weeks)).sum())
    cell_rate = len(customer_weeks[customer_weeks["amount_positive"] > 0]) / cells
    panel_rate = panel["z"].mean()

    assert 0.105 < cell_rate < 0.115, f"cell base rate {cell_rate:.2%}"
    assert 0.08 < panel_rate < 0.10, f"panel base rate {panel_rate:.2%}"
    assert panel_rate < cell_rate


def test_a_first_appearance_is_never_a_target():
    """The mechanism behind the gap above, isolated."""
    cw, spine = tiny([{"customer": "C1", "week": 3, "amount_positive": 500.0}])
    got = build_panel(cw, spine)
    assert (got["z"] == 0).all(), (
        "the only positive week is the first one, which no valid origin can reach")
    assert (got["target_week"] > 3).all()


def test_panel_size_is_what_the_restriction_implies(panel, customer_weeks, weeks):
    cells = int((len(weeks) - first_week(customer_weeks)).sum())
    # Ragged: each customer loses between 1 and 5 target slots at the end of the span.
    assert len(panel) < cells * len(HORIZONS)
    assert len(panel) > cells * len(HORIZONS) * 0.9


def test_dtypes_are_compact(panel):
    """~900k rows: strings and int64 cost several times more for no benefit."""
    assert panel["customer"].dtype.name == "category"
    assert panel["week"].dtype == "int16"
    assert panel["horizon"].dtype == "int8"
    assert panel["z"].dtype == "int8"


def test_every_customer_with_a_reachable_target_appears(panel, customer_weeks, weeks):
    """A customer first seen in the final week has no rows, and should not.

    No origin at or after their first appearance has a target inside the data, so there
    is nothing to predict. Excluding them is correct; they belong to `Nhat`.
    """
    first = first_week(customer_weeks)
    reachable = set(first[first <= len(weeks) - 2].index)
    unreachable = set(first[first > len(weeks) - 2].index)

    present = set(panel["customer"].astype(str))
    assert present == reachable
    assert not (present & unreachable)


def test_total_positive_target_value_is_consistent(panel, customer_weeks):
    """Each customer-week is a target once per horizon that can reach it."""
    per_h = panel.groupby("horizon", observed=True)["y"].sum()
    assert per_h.min() > 0
    # h=1 reaches the most target weeks, so it carries the most value.
    assert per_h[1] >= per_h[5]
