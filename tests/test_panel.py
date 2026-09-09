"""Tests for the week spine and the customer-week aggregation.

The constructed cases target the two decisions that carry risk: an integer week index
that must stay contiguous across a year boundary, and positives and negatives that must
not be netted together.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from collection_estimation import config                          # noqa: E402
from collection_estimation.ingest import read_corpus              # noqa: E402
from collection_estimation.panel import (                         # noqa: E402
    assign_week,
    build_week_index,
    to_customer_weeks,
    weeks_without_collections,
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


def rows(records) -> pd.DataFrame:
    """A minimal collections table."""
    frame = pd.DataFrame(records)
    frame["document_date"] = pd.to_datetime(frame["document_date"]).astype(
        "datetime64[s]")
    for column, default in (("company_code", "TR02"), ("government_invoice", "X"),
                            ("customer", "C1")):
        if column not in frame.columns:
            frame[column] = default
    return frame


# ---------------------------------------------------------------- the week spine

def test_week_index_is_contiguous_and_zero_based(weeks):
    assert list(weeks["week"]) == list(range(len(weeks)))


def test_weeks_start_on_monday_and_are_seven_days(weeks):
    assert (weeks["week_start"].dt.dayofweek == 0).all()
    assert ((weeks["week_end"] - weeks["week_start"]).dt.days == 6).all()


def test_consecutive_weeks_are_exactly_seven_days_apart(weeks):
    assert (weeks["week_start"].diff().dropna().dt.days == 7).all()


def test_spine_covers_the_whole_corpus(corpus, weeks):
    assert weeks["week_start"].iloc[0] <= corpus["document_date"].min()
    assert weeks["week_end"].iloc[-1] >= corpus["document_date"].max()


def test_integer_index_stays_contiguous_across_a_year_boundary():
    """The reason `week` is an integer and `iso_week` is only a label.

    Note what is and is not true here. ISO labels *sort* fine — the year leads and the
    week number is zero-padded, so `2025-W52` precedes `2026-W01` lexically as well as
    chronologically. What they do not support is ARITHMETIC: there is no operation on the
    label that advances one week, because 2025 has 52 ISO weeks and other years have 53,
    so the successor of `2025-W52` is `2026-W01` rather than `2025-W53`.

    The design writes `t + h` throughout. That is why `week` exists.
    """
    table = rows([{"document_date": "2025-12-20", "amount_local": 1.0},
                  {"document_date": "2026-01-10", "amount_local": 1.0}])
    spine = build_week_index(table)

    assert list(spine["week"]) == list(range(len(spine)))

    labels = list(spine["iso_week"])
    assert "2025-W52" in labels and "2026-W01" in labels

    # The two are adjacent by one step in the integer index...
    at = {lab: int(w) for lab, w in zip(labels, spine["week"])}
    assert at["2026-W01"] - at["2025-W52"] == 1

    # ...while the week NUMBER inside the label jumps backwards from 52 to 1. Nothing can
    # be added to the label to cross that.
    weeks_no = [int(lab.split("-W")[1]) for lab in labels]
    assert any(b < a for a, b in zip(weeks_no, weeks_no[1:])), (
        "the week number resets at the year boundary, so label arithmetic is undefined")

    # Consecutive integer indices are always exactly seven days apart, boundary included.
    assert (spine["week_start"].diff().dropna().dt.days == 7).all()


def test_week_assignment_agrees_with_the_spine(corpus, weeks):
    assigned = assign_week(corpus, weeks)
    lookup = weeks.set_index("week")
    starts = lookup.loc[assigned, "week_start"].to_numpy()
    ends = lookup.loc[assigned, "week_end"].to_numpy()
    dates = corpus["document_date"].to_numpy()
    assert (dates >= starts).all() and (dates <= ends).all()


def test_a_collection_outside_the_spine_raises(corpus, weeks):
    stray = corpus.head(1).copy()
    stray["document_date"] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="before the start"):
        assign_week(stray, weeks)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        build_week_index(pd.DataFrame({"document_date": []}))


# ---------------------------------------------------------------- aggregation

def test_positives_and_negatives_are_not_netted():
    """A refund must not cancel a collection into invisibility.

    Netting first would make `amount_net > 0` stand in for "did this customer collect
    anything", and those are different questions — 92 customer-weeks in the real corpus
    have a negative net despite containing positive collections.
    """
    table = rows([
        {"document_date": "2026-07-02", "amount_local": 1000.0},
        {"document_date": "2026-07-03", "amount_local": -1500.0},
    ])
    got = to_customer_weeks(table)
    assert len(got) == 1
    assert got["amount_positive"].iloc[0] == 1000.0
    assert got["amount_negative"].iloc[0] == -1500.0
    assert got["amount_net"].iloc[0] == -500.0
    assert got["n_rows"].iloc[0] == 2


def test_rows_in_the_same_week_are_summed():
    table = rows([{"document_date": d, "amount_local": 100.0}
                  for d in ("2026-06-29", "2026-06-30", "2026-07-03")])
    got = to_customer_weeks(table)
    assert len(got) == 1, "Monday to Friday of one week is one customer-week"
    assert got["amount_positive"].iloc[0] == 300.0


def test_rows_in_different_weeks_stay_apart():
    table = rows([{"document_date": "2026-06-26", "amount_local": 100.0},
                  {"document_date": "2026-06-29", "amount_local": 200.0}])
    got = to_customer_weeks(table).sort_values("week")
    assert len(got) == 2, "Friday and the following Monday are different weeks"
    assert list(got["amount_positive"]) == [100.0, 200.0]


def test_customers_are_kept_apart():
    table = rows([{"document_date": "2026-07-02", "amount_local": 1.0,
                   "customer": "C1"},
                  {"document_date": "2026-07-02", "amount_local": 2.0,
                   "customer": "C2"}])
    got = to_customer_weeks(table)
    assert set(got["customer"]) == {"C1", "C2"}


def test_unapplied_cash_is_counted_but_still_included_in_the_amount():
    table = rows([{"document_date": "2026-07-02", "amount_local": 500.0,
                   "government_invoice": config.STATUS_NOTES[0]},
                  {"document_date": "2026-07-02", "amount_local": 100.0,
                   "government_invoice": "GNF2026000000001"}])
    got = to_customer_weeks(table)
    assert got["n_unapplied"].iloc[0] == 1
    assert got["amount_positive"].iloc[0] == 600.0, (
        "unapplied cash is real money and belongs in the total")


# ---------------------------------------------------------------- against the corpus

def test_amounts_reconcile_with_the_source(corpus, customer_weeks):
    assert abs(customer_weeks["amount_net"].sum()
               - corpus["amount_local"].sum()) < 0.01
    assert customer_weeks["n_rows"].sum() == len(corpus)


def test_every_customer_survives_the_aggregation(corpus, customer_weeks):
    assert set(customer_weeks["customer"]) == set(corpus["customer"])


def test_the_sparse_table_is_much_smaller_than_the_dense_panel(
        corpus, weeks, customer_weeks):
    dense = corpus["customer"].nunique() * len(weeks)
    assert len(customer_weeks) < dense / 10, (
        "the aggregation must stay sparse; expanding to the dense panel is a later step")


def test_negative_net_customer_weeks_exist(customer_weeks):
    """The case that makes netting-first wrong, measured rather than assumed."""
    negative_net = (customer_weeks["amount_net"] < 0).sum()
    assert negative_net > 0
    both = ((customer_weeks["amount_net"] < 0)
            & (customer_weeks["amount_positive"] > 0)).sum()
    assert both >= 0


def test_no_week_is_entirely_without_collections(customer_weeks, weeks):
    """If this ever fails it is a finding, not a bug: report it, do not paper over it."""
    empty = weeks_without_collections(customer_weeks, weeks)
    assert empty == [], f"weeks with no collections at all: {empty}"


def test_company_code_is_carried_through(corpus, customer_weeks):
    assert set(customer_weeks["company_code"]) <= set(config.ENTITY_LABEL)
