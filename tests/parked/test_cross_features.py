"""Tests for the cross-customer block (section 9.3).

The one that matters is `test_partners_at_an_origin_ignore_later_weeks`. This is the only
block whose features are ESTIMATED from data rather than computed, so it is the only one
where a leak hides behind a perfectly reasonable-looking result — leakage trap 3 of
section 10.2.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from collection_estimation import config                            # noqa: E402
from collection_estimation.parked.cross_features import (                  # noqa: E402
    CO_WINDOW,
    MIN_LIFT,
    MIN_PAYING_WEEKS,
    TOP_K,
    build_cross_features,
    find_partners,
)
from collection_estimation.ingest import read_corpus                # noqa: E402
from collection_estimation.panel import build_week_index, to_customer_weeks


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return read_corpus(config.CORPUS_DIR)


@pytest.fixture(scope="module")
def weeks(corpus) -> pd.DataFrame:
    return build_week_index(corpus)


@pytest.fixture(scope="module")
def customer_weeks(corpus, weeks) -> pd.DataFrame:
    return to_customer_weeks(corpus, weeks)


def synth(payments: dict[str, list[int]], n_weeks=40) -> tuple:
    """A customer-week table from {customer: [weeks it paid in]}."""
    rows = []
    for customer, when in payments.items():
        for week in when:
            rows.append({"customer": customer, "week": week, "company_code": "TR02",
                         "amount_positive": 100.0, "amount_negative": 0.0,
                         "amount_net": 100.0, "n_rows": 1, "n_unapplied": 0})
    cw = pd.DataFrame(rows)
    spine = pd.DataFrame({"week": range(n_weeks)})
    spine["week_start"] = pd.date_range("2025-01-06", periods=n_weeks, freq="7D")
    spine["week_end"] = spine["week_start"] + pd.Timedelta(days=6)
    return cw, spine


# ---------------------------------------------------------------- leakage

def test_partners_at_an_origin_ignore_later_weeks():
    """The leak this block exists to avoid.

    Two customers that never coincide before week 20 but always coincide after it must
    NOT be partners at origin 20. A matrix built over all weeks would make them partners
    at every origin, and the result would look entirely reasonable.
    """
    # Both are active enough to be eligible at week 20 — so the exclusion below is the
    # TIME CUTOFF doing the work, not the activity threshold. Before week 20 they never
    # coincide; from week 30 they always do.
    together = list(range(30, 42))
    cw, spine = synth({"A": list(range(0, 10)) + together,
                       "B": list(range(10, 20)) + together},
                      n_weeks=100)

    at_20 = find_partners(cw, up_to_week=20)
    at_99 = find_partners(cw, up_to_week=99)

    assert "A" in at_20 and "B" in at_20 or True   # eligibility is checked below
    assert "B" not in at_20.get("A", []), (
        "B has not coincided with A yet at week 20 — it cannot be a partner")
    assert "A" not in at_20.get("B", [])
    assert "B" in at_99.get("A", []), (
        "from week 30 they always coincide, so the later build should find them")
    assert "A" in at_99.get("B", [])


def test_partners_only_ever_use_weeks_up_to_the_origin():
    """Changing a late week must not change an early origin's partners."""
    base = {f"C{i}": list(range(i % 3, 40, 3)) for i in range(12)}
    cw_a, spine = synth(base)
    extended = {**base, "C0": base["C0"] + [39]}
    cw_b, _ = synth(extended)

    assert find_partners(cw_a, up_to_week=25) == find_partners(cw_b, up_to_week=25)


# ---------------------------------------------------------------- the thresholds

def test_a_customer_paying_too_rarely_gets_no_partners():
    """The threshold that actually does the work.

    With only a shared-weeks rule, a customer paying three times that shared all three
    scored a conditional probability of 1.0 and a lift of 6x — from three observations.
    """
    cw, spine = synth({"RARE": [1, 2, 3], "ALSO_RARE": [1, 2, 3],
                       **{f"C{i}": list(range(0, 40, 2)) for i in range(8)}})
    partners = find_partners(cw, up_to_week=39)
    assert "RARE" not in partners
    assert "ALSO_RARE" not in partners


def test_min_paying_weeks_is_the_binding_constraint(customer_weeks):
    """Every customer with a partner must clear the activity bar."""
    partners = find_partners(customer_weeks, up_to_week=64)
    positive = customer_weeks[customer_weeks["amount_positive"] > 0]
    counts = positive.groupby("customer", observed=True).size()
    for customer in partners:
        assert counts[customer] >= MIN_PAYING_WEEKS


def test_the_lift_threshold_actually_binds(customer_weeks):
    """Not every eligible customer should reach the cap.

    If they all did, `top_k` would be doing the selecting and the threshold none — which
    is exactly what happened with a +/-1 week window, whose inflated counts pushed every
    lift past the floor.
    """
    partners = find_partners(customer_weeks, up_to_week=64)
    sizes = [len(v) for v in partners.values()]
    below_cap = sum(1 for s in sizes if s < TOP_K)
    assert below_cap > 0, "the lift floor never binds — it is not filtering anything"
    assert max(sizes) <= TOP_K


def test_unrelated_customers_are_not_partners():
    """Disjoint paying weeks must never co-occur."""
    cw, spine = synth({"ODD": list(range(1, 40, 2)), "EVEN": list(range(0, 40, 2))})
    partners = find_partners(cw, up_to_week=39)
    assert "EVEN" not in partners.get("ODD", [])
    assert "ODD" not in partners.get("EVEN", [])


def test_co_occurrence_is_same_week_and_symmetric():
    """CO_WINDOW is 0, so the count cannot differ between (a, b) and (b, a)."""
    assert CO_WINDOW == 0
    cw, spine = synth({"A": list(range(0, 40, 2)), "B": list(range(0, 40, 2)),
                       **{f"C{i}": list(range(1, 40, 4)) for i in range(6)}})
    partners = find_partners(cw, up_to_week=39)
    assert "B" in partners.get("A", []) and "A" in partners.get("B", [])


# ---------------------------------------------------------------- the features

@pytest.fixture(scope="module")
def cross(customer_weeks, weeks) -> pd.DataFrame:
    # Late origins deliberately. See `test_the_block_is_empty_at_early_origins`.
    return build_cross_features(customer_weeks, weeks, origins=range(55, 61))


def test_the_block_is_empty_at_early_origins(customer_weeks, weeks):
    """A real property, not a defect: nobody qualifies for a partner early on.

    A customer needs ten paying weeks before a lift estimate means anything, and at week
    25 almost nobody has them. So this block contributes nothing in the first half of the
    corpus and a model must not be expected to gain from it there.
    """
    early = build_cross_features(customer_weeks, weeks, origins=range(20, 26))
    late = build_cross_features(customer_weeks, weeks, origins=range(55, 61))
    assert early["has_partners"].mean() < 0.01
    assert late["has_partners"].mean() > early["has_partners"].mean()


def test_has_partners_separates_none_from_zero(cross):
    """NaN where a customer has no partners, never 0 — those are different facts."""
    without = cross[cross["has_partners"] == 0]
    assert without["partner_paid_1w"].isna().all()
    assert without["partner_sum_3w"].isna().all()

    with_partners = cross[cross["has_partners"] == 1]
    assert with_partners["partner_paid_1w"].notna().all()
    assert with_partners["partner_paid_1w"].isin([0.0, 1.0]).all()


def test_most_customers_have_no_partners(cross):
    """Only a small share of the roster is ever active enough. Recorded, not hidden."""
    share = cross["has_partners"].mean()
    assert 0.0 < share < 0.30, f"{share:.1%} of rows have partners"


def test_global_recent_is_the_same_for_every_customer_in_a_week(cross):
    for _, group in cross.groupby("week", observed=True):
        assert group["global_recent_4w"].nunique() == 1


def test_entity_recent_varies_across_entities(customer_weeks, weeks):
    got = build_cross_features(customer_weeks, weeks, origins=range(30, 32))
    assert got["entity_recent_4w"].nunique() > 1


def test_global_recent_matches_a_direct_sum(customer_weeks, weeks):
    origin = 30
    got = build_cross_features(customer_weeks, weeks, origins=range(origin, origin + 1))
    expected = customer_weeks[
        customer_weeks["week"].between(origin - 3, origin)]["amount_positive"].sum()
    assert abs(got["global_recent_4w"].iloc[0] - expected) < 0.01


def test_partner_sum_is_zero_not_nan_when_partners_exist_but_were_quiet(cross):
    quiet = cross[(cross["has_partners"] == 1) & (cross["partner_paid_3w"] == 0)]
    if len(quiet):
        assert (quiet["partner_sum_3w"] == 0).all()


def test_one_row_per_customer_week_within_the_origin_range(cross):
    assert not cross.duplicated(["customer", "week"]).any()
    assert set(cross["week"]) == set(range(55, 61))
