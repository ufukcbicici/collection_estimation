"""Tests for the customer-history block (section 9.1).

The one that matters most is `test_no_feature_at_t_depends_on_week_t_plus_one`. Every
other property here is a convenience; that one is the correctness of the whole exercise.
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
from collection_estimation.parked.history_features import (                # noqa: E402
    WINDOWS,
    attach_to_panel,
    build_history_features,
)
from collection_estimation.ingest import read_corpus                # noqa: E402
from collection_estimation.panel import (                           # noqa: E402
    build_week_index,
    to_customer_weeks,
)
from collection_estimation.parked.hurdle_panel import build_panel   # noqa: E402


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
def history(customer_weeks, weeks) -> pd.DataFrame:
    return build_history_features(customer_weeks, weeks)


def tiny(records, n_weeks=20):
    """A customer-week table and matching spine, without touching Excel."""
    cw = pd.DataFrame(records)
    for column, default in (("company_code", "TR02"), ("amount_negative", 0.0),
                            ("n_rows", 1), ("n_unapplied", 0)):
        if column not in cw.columns:
            cw[column] = default
    cw["amount_net"] = cw["amount_positive"] + cw["amount_negative"]
    spine = pd.DataFrame({"week": range(n_weeks)})
    spine["week_start"] = pd.date_range("2025-01-06", periods=n_weeks, freq="7D")
    spine["week_end"] = spine["week_start"] + pd.Timedelta(days=6)
    return cw, spine


# ---------------------------------------------------------------- leakage

def test_no_feature_at_t_depends_on_week_t_plus_one():
    """The correctness of the whole exercise.

    Change what happens at week 10 and nothing computed at weeks 0..9 may move. A window
    written `(t-W, t+1]` instead of `(t-W, t]` would fail here and pass everything else.
    """
    base, spine = tiny([{"customer": "C1", "week": w, "amount_positive": 100.0}
                        for w in (0, 3, 6, 9)])
    changed = pd.concat([base, pd.DataFrame([{
        "customer": "C1", "week": 10, "amount_positive": 999_999.0,
        "amount_negative": 0.0, "amount_net": 999_999.0, "company_code": "TR02",
        "n_rows": 1, "n_unapplied": 0}])], ignore_index=True)

    before = build_history_features(base, spine)
    after = build_history_features(changed, spine)

    early_before = before[before["week"] <= 9].reset_index(drop=True)
    early_after = after[after["week"] <= 9].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_before, early_after)


def test_window_includes_the_origin_week_itself():
    """`(t-W, t]` — week t is observed at the origin, so it counts."""
    cw, spine = tiny([{"customer": "C1", "week": 0, "amount_positive": 10.0},
                      {"customer": "C1", "week": 5, "amount_positive": 500.0}])
    got = build_history_features(cw, spine).set_index("week")
    assert got.loc[5, "sum_4"] == 500.0, "the payment in week 5 must be inside week 5's window"
    assert got.loc[4, "sum_4"] == 0.0


def test_attach_joins_on_the_origin_not_the_target(customer_weeks, weeks, history):
    panel = build_panel(customer_weeks, weeks).head(5000)
    merged = attach_to_panel(panel, history)
    assert len(merged) == len(panel)
    # A row's features must match its ORIGIN week's history, not its target's.
    lookup = history.set_index(["customer", "week"])["rec"]
    sample = merged.head(200)
    expected = lookup.reindex(
        list(zip(sample["customer"].astype(str), sample["week"]))).to_numpy()
    assert np.allclose(sample["rec"].to_numpy(), expected, equal_nan=True)


# ---------------------------------------------------------------- density

def test_counts_are_over_weeks_not_over_payments():
    """`n_4` counts weeks with a payment out of four, not the last four payments.

    Rolling over the sparse table instead of a dense grid would give 4 here, silently
    wrong for exactly the inactive customers who dominate the panel.
    """
    cw, spine = tiny([{"customer": "C1", "week": w, "amount_positive": 1.0}
                      for w in (0, 1, 10, 11)])
    got = build_history_features(cw, spine).set_index("week")
    assert got.loc[11, "n_4"] == 2.0, "weeks 8..11 contain two paying weeks"


def test_history_starts_at_the_customers_first_appearance():
    """Weeks before a customer existed are not weeks they failed to pay."""
    cw, spine = tiny([{"customer": "C1", "week": 7, "amount_positive": 1.0}])
    got = build_history_features(cw, spine)
    assert got["week"].min() == 7
    assert got[got["week"] == 7]["age"].iloc[0] == 0


def test_one_row_per_customer_week(history, customer_weeks, weeks):
    from collection_estimation.parked.hurdle_panel import first_week
    expected = int((len(weeks) - first_week(customer_weeks)).sum())
    assert len(history) == expected
    assert not history.duplicated(["customer", "week"]).any()


# ---------------------------------------------------------------- the features

def test_recency_counts_weeks_since_the_last_payment():
    cw, spine = tiny([{"customer": "C1", "week": 2, "amount_positive": 5.0}])
    got = build_history_features(cw, spine).set_index("week")
    assert got.loc[2, "rec"] == 0
    assert got.loc[5, "rec"] == 3


def test_recency_is_nan_before_a_customers_first_positive_payment():
    """A first appearance can be a refund-only week."""
    cw, spine = tiny([
        {"customer": "C1", "week": 1, "amount_positive": 0.0,
         "amount_negative": -50.0},
        {"customer": "C1", "week": 4, "amount_positive": 100.0},
    ])
    got = build_history_features(cw, spine).set_index("week")
    assert np.isnan(got.loc[1, "rec"])
    assert got.loc[1, "has_paid"] == 0.0
    assert got.loc[4, "rec"] == 0
    assert got.loc[4, "has_paid"] == 1.0


def test_gap_statistics_are_nan_until_there_are_two_payments():
    """NaN, not a sentinel. A single payment gives no gap, and 0 would be a lie."""
    cw, spine = tiny([{"customer": "C1", "week": 1, "amount_positive": 1.0},
                      {"customer": "C1", "week": 6, "amount_positive": 1.0}])
    got = build_history_features(cw, spine).set_index("week")
    assert np.isnan(got.loc[1, "gap_median"])
    assert got.loc[1, "has_gap"] == 0.0
    assert got.loc[6, "gap_median"] == 5.0
    assert got.loc[6, "has_gap"] == 1.0


def test_phase_tracks_position_in_a_regular_payment_cycle():
    """The feature that recognises a customer paying like clockwork."""
    cw, spine = tiny([{"customer": "C1", "week": w, "amount_positive": 1.0}
                      for w in (0, 4, 8, 12)])
    got = build_history_features(cw, spine).set_index("week")
    assert got.loc[12, "gap_median"] == 4.0
    assert got.loc[12, "phase"] == 0.0, "on a payment week, phase resets"
    assert got.loc[14, "phase"] == 2.0, "two weeks into a four-week cycle"


def test_trend_is_nan_when_the_earlier_window_was_empty():
    cw, spine = tiny([{"customer": "C1", "week": 12, "amount_positive": 100.0}],
                     n_weeks=25)
    got = build_history_features(cw, spine).set_index("week")
    assert np.isnan(got.loc[12, "trend"]), "no earlier activity to compare against"


def test_refund_features_are_present_and_negative():
    cw, spine = tiny([{"customer": "C1", "week": 2, "amount_positive": 100.0,
                       "amount_negative": -30.0}])
    got = build_history_features(cw, spine).set_index("week")
    assert got.loc[2, "n_refund_26"] == 1.0
    assert got.loc[2, "sum_refund_26"] == -30.0
    assert got.loc[5, "lifetime_n_refunds"] == 1.0


def test_active_is_a_share_between_zero_and_one(history):
    assert history["active"].between(0, 1).all()


# ---------------------------------------------------------------- the real corpus

def test_all_expected_columns_exist(history):
    for window in WINDOWS:
        for prefix in ("n", "sum", "mean", "max"):
            assert f"{prefix}_{window}" in history.columns
    for column in ("rec", "age", "active", "gap_mean", "gap_sd", "gap_median",
                   "has_gap", "has_paid", "phase", "trend",
                   "n_refund_26", "sum_refund_26", "lifetime_n_refunds"):
        assert column in history.columns


def test_no_infinities_anywhere(history):
    numeric = history.drop(columns=["customer"]).to_numpy(dtype=float)
    assert not np.isinf(numeric).any(), "an infinity means a division was not guarded"


def test_nan_appears_only_where_it_should(history):
    """NaN is deliberate and confined to the genuinely undefined."""
    allowed = {"rec", "gap_mean", "gap_sd", "gap_median", "phase", "trend"}
    for column in history.columns:
        if column in allowed or column == "customer":
            continue
        assert history[column].notna().all(), f"{column} should never be NaN"


def test_gap_nan_agrees_with_the_has_gap_indicator(history):
    assert (history["gap_median"].isna() == (history["has_gap"] == 0)).all()


def test_a_single_appearance_customer_has_no_gap(history, customer_weeks):
    counts = customer_weeks.groupby("customer", observed=True).size()
    once = counts[counts == 1].index[:20]
    rows = history[history["customer"].isin(once)]
    assert (rows["has_gap"] == 0).all()


def test_features_are_usable_by_the_estimators_we_plan_to_use(history):
    """NaN must not be a problem for the models. Verified, not assumed."""
    from sklearn.ensemble import RandomForestClassifier

    sample = history.head(4000)
    X = sample.drop(columns=["customer", "week"]).to_numpy(dtype=float)
    y = (np.arange(len(sample)) % 2).astype(int)
    RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
