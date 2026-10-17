"""Tests for `hurdle/side_tables.py`: `new_customer_cash` (table A), `refunds_by_week`
(table R), `is_unidentified`, `unidentified_cash_by_week` (table G), `company_week_totals`,
`check_side_tables`, NEW_CASH_COLUMNS, REFUND_COLUMNS, UNIDENTIFIED_COLUMNS and
TIE_OUT_TOLERANCE.

Hand-built `weekly` (shaped like `series.weekly_amounts`), `known` (like `known_weeks`) and
`weeks` (like `series.week_calendar`) tables; one slow integration block against the
synthetic corpus and ledger in data/raw (skipped if absent; never uses data/interim).
"""
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.panel import add_labels, panel_rows
from collection_estimation.hurdle.side_tables import (
    NEW_CASH_COLUMNS,
    REFUND_COLUMNS,
    TIE_OUT_TOLERANCE,
    UNIDENTIFIED_COLUMNS,
    check_side_tables,
    company_week_totals,
    is_unidentified,
    new_customer_cash,
    refunds_by_week,
    unidentified_cash_by_week,
)
from tests.hurdle.test_panel import known_table
from tests.hurdle.test_panel_labels import weekly_table, weeks_table


def companies_of(known):
    """The companies of a hand-built known table, in order of first appearance: the list
    `new_customer_cash` is given when a test's companies are just those of `known`."""
    return list(known["company_code"].unique())


def cash(table, company, origin, h):
    hit = table[(table["company_code"] == company) & (table["origin_week"] == origin)
                & (table["h"] == h)]
    assert len(hit) == 1, (company, origin, h, len(hit))
    return hit.iloc[0]["new_customer_cash"]


def pairs(table, company):
    rows = table[table["company_code"] == company]
    return set(zip(rows["origin_week"], rows["h"]))


# --- the worked example: one company, K known at week 3, pays 100 in week 4 --------------

@pytest.fixture
def worked():
    known = known_table([("TR02", "K", 3)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0)])
    weeks = weeks_table(6)
    return new_customer_cash(weekly, known, weeks, 4, companies_of(known))


def test_constant():
    assert NEW_CASH_COLUMNS == ["company_code", "origin_week", "h", "target_week",
                                "new_customer_cash"]


def test_worked_example_counts_k_before_it_is_known(worked):
    assert cash(worked, "TR02", 0, 4) == 100.0
    assert cash(worked, "TR02", 1, 3) == 100.0
    assert cash(worked, "TR02", 2, 2) == 100.0


def test_worked_example_key_known_at_origin_is_not_new(worked):
    assert cash(worked, "TR02", 3, 1) == 0.0


def test_worked_example_origins_from_known_week_never_count_k(worked):
    assert (worked.loc[worked["origin_week"] >= 3, "new_customer_cash"] == 0.0).all()


def test_worked_example_total_is_three_times_100(worked):
    assert worked["new_customer_cash"].sum() == 300.0
    assert (worked["new_customer_cash"] > 0).sum() == 3


def test_worked_example_rows_other_target_weeks_are_zero(worked):
    other = worked[worked["target_week"] != 4]
    assert (other["new_customer_cash"] == 0.0).all()


# --- the grid ---------------------------------------------------------------------------

def test_rows_are_every_origin_and_h_with_target_in_calendar(worked):
    expected = {(t, h) for t in range(7) for h in range(1, 5) if t + h <= 6}
    assert pairs(worked, "TR02") == expected
    assert len(worked) == len(expected)


def test_target_week_is_origin_plus_h(worked):
    assert (worked["target_week"] == worked["origin_week"] + worked["h"]).all()


def test_rows_beyond_last_week_are_absent(worked):
    assert worked["target_week"].max() == 6
    assert worked["origin_week"].max() == 5  # origin 6 has no target within the calendar


def test_missing_target_weeks_are_absent():
    known = known_table([("TR02", "K", 3)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(6, missing={4}), 3, companies_of(known))
    assert 4 not in set(table["target_week"])
    expected = {(t, h) for t in range(7) for h in range(1, 4) if t + h <= 6 and t + h != 4}
    assert pairs(table, "TR02") == expected
    assert (table["new_customer_cash"] == 0.0).all()  # K's only receipt is in week 4


def test_origin_in_missing_week_still_has_rows():
    known = known_table([("TR02", "K", 3)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(6, missing={2}), 2, companies_of(known))
    assert cash(table, "TR02", 2, 2) == 100.0  # origin 2 is missing, target 4 is not


def test_every_company_in_the_list_gets_rows_even_without_receipts():
    known = known_table([("TR02", "K", 3), ("TR05", "Q", 0)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(5), 2, companies_of(known))
    assert pairs(table, "TR05") == pairs(table, "TR02")
    assert (table.loc[table["company_code"] == "TR05", "new_customer_cash"] == 0.0).all()


def test_companies_come_from_the_list_only():
    # The companies are those of `companies`, not those of `known`.
    known = known_table([("TR02", "K", 1)])
    table = new_customer_cash(weekly_table([]), known, weeks_table(3), 2, ["TR04"])
    assert set(table["company_code"]) == {"TR04"}


def test_company_in_the_list_with_no_known_key_gets_all_zero_rows():
    known = known_table([("TR02", "K", 3)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(5), 2, ["TR02", "TR06"])
    assert set(table["company_code"]) == {"TR02", "TR06"}
    assert pairs(table, "TR06") == pairs(table, "TR02")
    assert len(pairs(table, "TR06")) == 9  # origins 0-4 x h 1-2 with target <= 5
    assert (table.loc[table["company_code"] == "TR06", "new_customer_cash"] == 0.0).all()
    assert cash(table, "TR02", 2, 2) == 100.0  # TR02 unchanged


def test_company_in_known_but_not_in_the_list_gets_no_rows():
    # TR04's new-customer cash is not counted in A when TR04 is not in the list.
    known = known_table([("TR02", "K", 3), ("TR04", "B", 3)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0), ("TR04-B", 4, 55.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(5), 2, ["TR02"])
    assert set(table["company_code"]) == {"TR02"}
    assert table["new_customer_cash"].sum() == 100.0  # only origin 2, h = 2; no 55 anywhere
    assert 55.0 not in set(table["new_customer_cash"])


def test_duplicate_companies_counted_once():
    known = known_table([("TR02", "K", 3), ("TR04", "B", 3)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0), ("TR04-B", 4, 55.0, 0.0)])
    once = new_customer_cash(weekly, known, weeks_table(5), 2, ["TR02", "TR04"])
    twice = new_customer_cash(weekly, known, weeks_table(5), 2,
                              ["TR04", "TR02", "TR02", "TR04", "TR02"])
    pd.testing.assert_frame_equal(once, twice)
    assert not twice.duplicated(["company_code", "origin_week", "h"]).any()


def test_companies_are_required():
    known = known_table([("TR02", "K", 3)])
    with pytest.raises(TypeError):
        new_customer_cash(weekly_table([]), known, weeks_table(5), 2)


def test_empty_weekly_gives_all_zero():
    known = known_table([("TR02", "K", 1), ("TR04", "B", 5)])
    table = new_customer_cash(weekly_table([]), known, weeks_table(3), 2, companies_of(known))
    assert len(table) == 2 * 5  # (0,1)(0,2)(1,1)(1,2)(2,1) per company
    assert (table["new_customer_cash"] == 0.0).all()
    assert table["new_customer_cash"].dtype == "float64"


def test_horizons_one():
    known = known_table([("TR02", "K", 3)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0), ("TR02-K", 3, 40.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(5), 1, companies_of(known))
    assert set(table["h"]) == {1}
    assert list(table["origin_week"]) == [0, 1, 2, 3, 4]
    assert cash(table, "TR02", 2, 1) == 40.0  # week 3 paid, known 3 > origin 2
    assert cash(table, "TR02", 3, 1) == 0.0   # known at 3 = origin
    assert table["new_customer_cash"].sum() == 40.0


# --- what counts as new -----------------------------------------------------------------

def test_key_known_next_week_paying_two_weeks_later():
    known = known_table([("TR02", "K", 1)])
    weekly = weekly_table([("TR02-K", 2, 70.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(4), 3, companies_of(known))
    assert cash(table, "TR02", 0, 2) == 70.0  # new for origin 0, h=2
    assert cash(table, "TR02", 1, 1) == 0.0   # known at 1: not new for origin 1


def test_key_known_before_week_zero_is_never_new():
    known = known_table([("TR02", "A", -5)])
    weekly = weekly_table([("TR02-A", w, 10.0, 0.0) for w in range(1, 6)])
    table = new_customer_cash(weekly, known, weeks_table(5), 3, companies_of(known))
    assert (table["new_customer_cash"] == 0.0).all()


def test_key_known_after_last_week_counts_everywhere_it_pays():
    # Its known_week is later than every origin, so all its cash is new.
    known = known_table([("TR02", "Z", 99)])
    weekly = weekly_table([("TR02-Z", 2, 5.0, 0.0), ("TR02-Z", 3, 7.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(3), 2, companies_of(known))
    assert cash(table, "TR02", 1, 1) == 5.0
    assert cash(table, "TR02", 0, 2) == 5.0
    assert cash(table, "TR02", 2, 1) == 7.0
    assert cash(table, "TR02", 1, 2) == 7.0
    assert table["new_customer_cash"].sum() == 24.0


def test_refunds_are_not_included():
    known = known_table([("TR02", "K", 3)])
    weekly = weekly_table([("TR02-K", 4, 100.0, -30.0), ("TR02-K", 5, 0.0, -20.0)])
    table = new_customer_cash(weekly, known, weeks_table(6), 3, companies_of(known))
    assert cash(table, "TR02", 2, 2) == 100.0
    assert cash(table, "TR02", 2, 3) == 0.0  # week 5: refund only, gross 0
    assert (table["new_customer_cash"] >= 0).all()


def test_keys_of_another_company_do_not_count():
    known = known_table([("TR02", "K", 3), ("TR04", "K", 3)])
    weekly = weekly_table([("TR04-K", 4, 100.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(5), 2, companies_of(known))
    assert (table.loc[table["company_code"] == "TR02", "new_customer_cash"] == 0.0).all()
    assert cash(table, "TR04", 2, 2) == 100.0
    assert cash(table, "TR04", 3, 1) == 0.0


def test_several_new_keys_in_one_week_add_up_and_known_ones_do_not():
    known = known_table([("TR02", "A", -1), ("TR02", "B", 3), ("TR02", "C", 4)])
    weekly = weekly_table([("TR02-A", 4, 1000.0, 0.0), ("TR02-B", 4, 10.0, 0.0),
                           ("TR02-C", 4, 1.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(5), 3, companies_of(known))
    assert cash(table, "TR02", 1, 3) == 11.0
    assert cash(table, "TR02", 2, 2) == 11.0
    assert cash(table, "TR02", 3, 1) == 1.0   # B known at 3, C not yet


def test_same_customer_id_in_two_companies_judged_on_key():
    known = known_table([("TR02", "A", 0), ("TR05", "A", 3)])
    weekly = weekly_table([("TR02-A", 2, 10.0, 0.0), ("TR05-A", 2, 20.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(3), 2, companies_of(known))
    assert cash(table, "TR02", 1, 1) == 0.0
    assert cash(table, "TR05", 1, 1) == 20.0
    assert cash(table, "TR02", 0, 2) == 0.0
    assert cash(table, "TR05", 0, 2) == 20.0


# --- shape and types --------------------------------------------------------------------

@pytest.fixture
def two_companies():
    known = known_table([("TR05", "C", 2), ("TR02", "K", 3), ("TR04", "B", 0)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0), ("TR05-C", 3, 25.5, 0.0)])
    return new_customer_cash(weekly, known, weeks_table(6, missing={5}), 3, companies_of(known))


def test_columns_exactly(two_companies):
    assert list(two_companies.columns) == NEW_CASH_COLUMNS


def test_dtypes(two_companies):
    for column in ("origin_week", "h", "target_week"):
        assert two_companies[column].dtype == "int64", column
    assert two_companies["new_customer_cash"].dtype == "float64"
    assert two_companies["company_code"].dtype == object


def test_sorted_by_company_origin_h(two_companies):
    ordered = two_companies.sort_values(["company_code", "origin_week", "h"],
                                        ignore_index=True)
    pd.testing.assert_frame_equal(two_companies, ordered)
    assert list(two_companies["company_code"].unique()) == ["TR02", "TR04", "TR05"]


def test_index_is_range(two_companies):
    assert list(two_companies.index) == list(range(len(two_companies)))


def test_no_nan(two_companies):
    assert not two_companies.isna().any().any()


def test_values_in_two_company_example(two_companies):
    assert cash(two_companies, "TR05", 1, 2) == 25.5
    assert cash(two_companies, "TR05", 2, 1) == 0.0
    # horizons 3: K (known 3, pays 4) from origins 1, 2; C (known 2, pays 3) from 0, 1.
    assert two_companies["new_customer_cash"].sum() == 25.5 * 2 + 100.0 * 2


# --- errors -----------------------------------------------------------------------------

def test_key_not_in_known_raises():
    known = known_table([("TR02", "K", 3)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0), ("TR02-X", 4, 1.0, 0.0)])
    with pytest.raises(ValueError, match=re.escape(
            "1 keys in the weekly amounts are not in the known table (first: ['TR02-X'])")):
        new_customer_cash(weekly, known, weeks_table(5), 2, companies_of(known))


def test_unknown_keys_listed_at_most_ten_sorted():
    known = known_table([("TR02", "K", 3)])
    missing = [f"X{i:02d}" for i in range(12)]
    weekly = weekly_table([(f"TR02-{c}", 1, 1.0, 0.0) for c in reversed(missing)])
    with pytest.raises(ValueError) as caught:
        new_customer_cash(weekly, known, weeks_table(5), 2, companies_of(known))
    first_ten = [f"TR02-{c}" for c in missing[:10]]
    assert str(caught.value) == (f"12 keys in the weekly amounts are not in the known table "
                                 f"(first: {first_ten})")


def test_unknown_key_counted_once_across_weeks():
    known = known_table([("TR02", "K", 3)])
    weekly = weekly_table([("TR02-X", 1, 1.0, 0.0), ("TR02-X", 2, 1.0, 0.0)])
    with pytest.raises(ValueError, match=r"^1 keys"):
        new_customer_cash(weekly, known, weeks_table(5), 2, companies_of(known))


# --- errors: a key repeated in the known table ------------------------------------------

def test_key_twice_in_known_raises():
    known = known_table([("TR02", "K", 3), ("TR02", "K", 5)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0)])
    with pytest.raises(ValueError) as caught:
        new_customer_cash(weekly, known, weeks_table(6), 2, companies_of(known))
    assert str(caught.value) == ("1 keys appear more than once in the known table "
                                 "(first: ['TR02-K'])")


def test_key_three_times_in_known_counted_once():
    known = known_table([("TR02", "K", 3), ("TR02", "K", 5), ("TR02", "K", 1)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0)])
    with pytest.raises(ValueError) as caught:
        new_customer_cash(weekly, known, weeks_table(6), 2, companies_of(known))
    assert str(caught.value) == ("1 keys appear more than once in the known table "
                                 "(first: ['TR02-K'])")


def test_several_repeated_keys_counted_and_first_five_sorted():
    repeated = ["G", "B", "F", "A", "E", "C", "D"]  # seven repeated keys, not in order
    rows = [("TR02", c, 1) for c in repeated] + [("TR02", c, 2) for c in repeated]
    rows += [("TR02", c, 4) for c in ("A", "D")]  # a third time for two of them
    rows += [("TR02", "U", 0), ("TR04", "A", 0)]  # unique keys, not counted
    known = known_table(rows)
    with pytest.raises(ValueError) as caught:
        new_customer_cash(weekly_table([]), known, weeks_table(4), 2, companies_of(known))
    assert str(caught.value) == ("7 keys appear more than once in the known table "
                                 "(first: ['TR02-A', 'TR02-B', 'TR02-C', 'TR02-D', 'TR02-E'])")


def test_repeated_keys_fewer_than_five_all_listed():
    known = known_table([("TR05", "Z", 1), ("TR05", "Z", 2), ("TR02", "Y", 1),
                         ("TR02", "Y", 3), ("TR04", "X", 0)])
    with pytest.raises(ValueError) as caught:
        new_customer_cash(weekly_table([]), known, weeks_table(4), 2, companies_of(known))
    assert str(caught.value) == ("2 keys appear more than once in the known table "
                                 "(first: ['TR02-Y', 'TR05-Z'])")


def test_repeated_key_checked_before_unknown_keys():
    # Both faults are present; the repeated key is reported first.
    known = known_table([("TR02", "K", 3), ("TR02", "K", 4)])
    weekly = weekly_table([("TR02-X", 4, 1.0, 0.0)])
    with pytest.raises(ValueError, match=r"^1 keys appear more than once in the known table"):
        new_customer_cash(weekly, known, weeks_table(5), 2, companies_of(known))


def test_repeated_key_raises_in_any_row_order():
    known = known_table([("TR02", "K", 3), ("TR02", "A", 1), ("TR02", "K", 4)])
    shuffled = known.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(ValueError, match=re.escape("(first: ['TR02-K'])")):
        new_customer_cash(weekly_table([]), shuffled, weeks_table(5), 2, companies_of(shuffled))


def test_same_customer_id_in_two_companies_is_not_repeated():
    known = known_table([("TR02", "K", 3), ("TR04", "K", 3), ("TR05", "K", 1)])
    table = new_customer_cash(weekly_table([("TR04-K", 4, 5.0, 0.0)]), known,
                              weeks_table(5), 2, companies_of(known))
    assert set(table["company_code"]) == {"TR02", "TR04", "TR05"}


def test_unique_keys_still_pass():
    known = known_table([("TR02", "A", 0), ("TR02", "B", 2), ("TR04", "C", 1)])
    weekly = weekly_table([("TR02-B", 3, 10.0, 0.0), ("TR04-C", 2, 4.0, 0.0)])
    table = new_customer_cash(weekly, known, weeks_table(5), 2, companies_of(known))
    assert cash(table, "TR02", 1, 2) == 10.0
    assert cash(table, "TR04", 0, 2) == 4.0


def test_repeated_key_inputs_not_modified():
    known = known_table([("TR02", "K", 3), ("TR02", "K", 5), ("TR04", "B", 0)])
    weekly = weekly_table([("TR02-K", 4, 100.0, -3.0)])
    weeks = weeks_table(6, missing={5})
    copies = (known.copy(deep=True), weekly.copy(deep=True), weeks.copy(deep=True))
    with pytest.raises(ValueError):
        new_customer_cash(weekly, known, weeks, 3, companies_of(known))
    pd.testing.assert_frame_equal(known, copies[0])
    pd.testing.assert_frame_equal(weekly, copies[1])
    pd.testing.assert_frame_equal(weeks, copies[2])


# --- purity -----------------------------------------------------------------------------

def test_inputs_not_modified():
    known = known_table([("TR02", "K", 3), ("TR04", "B", 0)])
    weekly = weekly_table([("TR02-K", 4, 100.0, -3.0), ("TR04-B", 2, 9.0, 0.0)])
    weeks = weeks_table(6, missing={5})
    copies = (known.copy(deep=True), weekly.copy(deep=True), weeks.copy(deep=True))
    new_customer_cash(weekly, known, weeks, 3, companies_of(known))
    pd.testing.assert_frame_equal(known, copies[0])
    pd.testing.assert_frame_equal(weekly, copies[1])
    pd.testing.assert_frame_equal(weeks, copies[2])


def test_deterministic():
    known = known_table([("TR02", "K", 3), ("TR04", "B", 1)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0), ("TR04-B", 2, 9.0, 0.0)])
    weeks = weeks_table(6)
    pd.testing.assert_frame_equal(new_customer_cash(weekly, known, weeks, 3, companies_of(known)),
                                  new_customer_cash(weekly, known, weeks, 3, companies_of(known)))


def test_row_order_of_inputs_does_not_matter():
    known = known_table([("TR02", "K", 3), ("TR04", "B", 1)])
    weekly = weekly_table([("TR02-K", 4, 100.0, 0.0), ("TR04-B", 2, 9.0, 0.0),
                           ("TR02-K", 5, 3.0, 0.0)])
    weeks = weeks_table(6)
    a = new_customer_cash(weekly, known, weeks, 3, companies_of(known))
    b = new_customer_cash(weekly.iloc[::-1].reset_index(drop=True),
                          known.iloc[::-1].reset_index(drop=True),
                          weeks.iloc[::-1].reset_index(drop=True), 3,
                          companies_of(known)[::-1])
    pd.testing.assert_frame_equal(a, b)


# === refunds_by_week (table R) ==========================================================

def refund(table, company, week):
    hit = table[(table["company_code"] == company) & (table["week"] == week)]
    assert len(hit) == 1, (company, week, len(hit))
    return hit.iloc[0]["refunds"]


def test_refund_constant():
    assert REFUND_COLUMNS == ["company_code", "week", "refunds"]


@pytest.fixture
def refund_example():
    # Weeks 0-5, week 3 missing. TR02: K refunds in weeks 1 and 4, L (unknown anywhere)
    # in week 4; TR04 in week 2; TR05 has no lines at all.
    weekly = weekly_table([("TR02-K", 1, 100.0, -10.0), ("TR02-K", 4, 0.0, -5.5),
                           ("TR02-L", 4, 50.0, -2.25), ("TR02-L", 2, 80.0, 0.0),
                           ("TR04-B", 2, 0.0, -7.0)])
    return refunds_by_week(weekly, weeks_table(5, missing={3}), ["TR05", "TR02", "TR04"])


def test_refund_values(refund_example):
    assert refund(refund_example, "TR02", 1) == -10.0
    assert refund(refund_example, "TR02", 4) == -7.75  # K and L together
    assert refund(refund_example, "TR04", 2) == -7.0


def test_refund_zero_in_week_without_refunds(refund_example):
    assert refund(refund_example, "TR02", 0) == 0.0
    assert refund(refund_example, "TR02", 2) == 0.0  # receipts but no refund
    assert refund(refund_example, "TR04", 4) == 0.0


def test_refund_company_without_lines_all_zero(refund_example):
    rows = refund_example[refund_example["company_code"] == "TR05"]
    assert list(rows["week"]) == [0, 1, 2, 4, 5]
    assert (rows["refunds"] == 0.0).all()


def test_refund_rows_every_company_and_week_with_data(refund_example):
    expected = {(c, w) for c in ("TR02", "TR04", "TR05") for w in (0, 1, 2, 4, 5)}
    assert set(zip(refund_example["company_code"], refund_example["week"])) == expected
    assert len(refund_example) == len(expected)


def test_refund_missing_week_has_no_row(refund_example):
    assert 3 not in set(refund_example["week"])


def test_refund_in_missing_week_is_dropped():
    weekly = weekly_table([("TR02-K", 3, 0.0, -9.0), ("TR02-K", 2, 0.0, -1.0)])
    table = refunds_by_week(weekly, weeks_table(4, missing={3}), ["TR02"])
    assert list(table["week"]) == [0, 1, 2, 4]
    assert table["refunds"].sum() == -1.0


def test_refund_weeks_after_last_week_ignored():
    weekly = weekly_table([("TR02-K", 2, 0.0, -1.0), ("TR02-K", 7, 0.0, -100.0)])
    table = refunds_by_week(weekly, weeks_table(4), ["TR02"])
    assert list(table["week"]) == [0, 1, 2, 3, 4]
    assert table["refunds"].sum() == -1.0


def test_refund_weeks_before_week_zero_ignored():
    weekly = weekly_table([("TR02-K", -1, 0.0, -4.0), ("TR02-K", 0, 0.0, -1.0)])
    table = refunds_by_week(weekly, weeks_table(2), ["TR02"])
    assert list(table["week"]) == [0, 1, 2]
    assert table["refunds"].sum() == -1.0


def test_refund_duplicate_companies_counted_once():
    weekly = weekly_table([("TR02-K", 1, 0.0, -3.0)])
    table = refunds_by_week(weekly, weeks_table(2), ["TR02", "TR04", "TR02", "TR04"])
    assert len(table) == 2 * 3
    assert refund(table, "TR02", 1) == -3.0


def test_refund_company_not_in_list_ignored():
    weekly = weekly_table([("TR02-K", 1, 0.0, -3.0), ("TR06-K", 1, 0.0, -50.0)])
    table = refunds_by_week(weekly, weeks_table(2), ["TR02"])
    assert set(table["company_code"]) == {"TR02"}
    assert table["refunds"].sum() == -3.0


def test_refund_all_keys_count_one_week_one_company():
    weekly = weekly_table([(f"TR02-C{i}", 2, 0.0, -1.0) for i in range(6)])
    table = refunds_by_week(weekly, weeks_table(3), ["TR02"])
    assert refund(table, "TR02", 2) == -6.0


def test_refund_gross_receipts_not_involved():
    base = [("TR02-K", 1, 0.0, -3.0), ("TR02-L", 2, 0.0, 0.0)]
    rich = [("TR02-K", 1, 9999.0, -3.0), ("TR02-L", 2, 123.0, 0.0)]
    weeks = weeks_table(3)
    pd.testing.assert_frame_equal(refunds_by_week(weekly_table(base), weeks, ["TR02"]),
                                  refunds_by_week(weekly_table(rich), weeks, ["TR02"]))


def test_refund_empty_weekly_gives_all_zero():
    table = refunds_by_week(weekly_table([]), weeks_table(3, missing={1}), ["TR02", "TR04"])
    assert len(table) == 2 * 3
    assert (table["refunds"] == 0.0).all()
    assert table["refunds"].dtype == "float64"
    assert list(table.columns) == REFUND_COLUMNS


def test_refund_columns_exactly(refund_example):
    assert list(refund_example.columns) == REFUND_COLUMNS


def test_refund_dtypes(refund_example):
    assert refund_example["week"].dtype == "int64"
    assert refund_example["refunds"].dtype == "float64"
    assert refund_example["company_code"].dtype == object


def test_refund_sorted_by_company_then_week(refund_example):
    ordered = refund_example.sort_values(["company_code", "week"], ignore_index=True)
    pd.testing.assert_frame_equal(refund_example, ordered)
    assert list(refund_example["company_code"].unique()) == ["TR02", "TR04", "TR05"]


def test_refund_index_is_range(refund_example):
    assert list(refund_example.index) == list(range(len(refund_example)))


def test_refund_no_nan(refund_example):
    assert not refund_example.isna().any().any()


def test_refund_never_positive(refund_example):
    assert (refund_example["refunds"] <= 0).all()


def test_refund_inputs_not_modified():
    weekly = weekly_table([("TR02-K", 1, 100.0, -10.0), ("TR04-B", 7, 0.0, -7.0)])
    weeks = weeks_table(5, missing={3})
    companies = ["TR04", "TR02", "TR04"]
    copies = (weekly.copy(deep=True), weeks.copy(deep=True), list(companies))
    refunds_by_week(weekly, weeks, companies)
    pd.testing.assert_frame_equal(weekly, copies[0])
    pd.testing.assert_frame_equal(weeks, copies[1])
    assert companies == copies[2]


def test_refund_row_order_of_inputs_does_not_matter():
    weekly = weekly_table([("TR02-K", 1, 0.0, -10.0), ("TR04-B", 2, 0.0, -7.0),
                           ("TR02-L", 1, 0.0, -1.0)])
    weeks = weeks_table(4, missing={2})
    a = refunds_by_week(weekly, weeks, ["TR02", "TR04"])
    b = refunds_by_week(weekly.iloc[::-1].reset_index(drop=True),
                        weeks.iloc[::-1].reset_index(drop=True), ["TR04", "TR02"])
    pd.testing.assert_frame_equal(a, b)


def test_refund_deterministic():
    weekly = weekly_table([("TR02-K", 1, 0.0, -10.0)])
    weeks = weeks_table(3)
    pd.testing.assert_frame_equal(refunds_by_week(weekly, weeks, ["TR02"]),
                                  refunds_by_week(weekly, weeks, ["TR02"]))


# === is_unidentified ===================================================================

def ids_table(customer_ids, company="TR02"):
    """A table with a customer_id column (like `known` or `weekly`), a custom index."""
    return pd.DataFrame({"company_code": [company] * len(customer_ids),
                         "customer_id": pd.Series(customer_ids, dtype=object).to_numpy()},
                        index=[10 + 3 * i for i in range(len(customer_ids))])


def test_is_unidentified_exact_match():
    table = ids_table(["99999", "12345", "99999"])
    assert list(is_unidentified(table, ("99999",))) == [True, False, True]


@pytest.mark.parametrize("placeholder", ["0099999", "0000099999", " 99999 ", "99999\t",
                                         "​99999"],
                         ids=["leading-zeros", "ten-digit-sap", "spaces", "tab", "zero-width"])
def test_is_unidentified_normalises_the_placeholder(placeholder):
    table = ids_table(["99999", "12345"])
    assert list(is_unidentified(table, (placeholder,))) == [True, False]


@pytest.mark.parametrize("placeholder", ["suspense", " Suspense ", "SUSPENSE"],
                         ids=["lower", "mixed-spaces", "upper"])
def test_is_unidentified_letter_ids_ignore_case_and_spaces(placeholder):
    table = ids_table(["SUSPENSE", "AB12"])
    assert list(is_unidentified(table, (placeholder,))) == [True, False]


def test_is_unidentified_leading_zeros_kept_for_letter_ids():
    # Only all-digit IDs lose their leading zeros, so "00AB" is not "AB".
    table = ids_table(["00AB", "AB"])
    assert list(is_unidentified(table, ("00ab",))) == [True, False]


def test_is_unidentified_does_not_match_a_prefix_or_substring():
    table = ids_table(["999990", "9999", "199999"])
    assert not is_unidentified(table, ("99999",)).any()


def test_is_unidentified_zero_matches_zero():
    table = ids_table(["0", "10"])
    assert list(is_unidentified(table, ("0000",))) == [True, False]


def test_is_unidentified_several_placeholders():
    table = ids_table(["99999", "SUSPENSE", "12345", "88888"])
    assert list(is_unidentified(table, ("0099999", "suspense"))) == [True, True, False, False]


def test_is_unidentified_empty_tuple_is_all_false():
    table = ids_table(["99999", "12345"])
    result = is_unidentified(table, ())
    assert not result.any()
    assert len(result) == 2


def test_is_unidentified_empty_list_is_all_false():
    assert not is_unidentified(ids_table(["99999"]), []).any()


def test_is_unidentified_empty_table():
    result = is_unidentified(ids_table([]), ("99999",))
    assert len(result) == 0


def test_is_unidentified_matches_customer_id_in_every_company():
    # Pinned: matching is on customer_id, not key, so a placeholder of one company matches
    # the same customer number in every company.
    weekly = weekly_table([("TR02-99999", 1, 10.0, 0.0), ("TR04-99999", 1, 20.0, 0.0),
                           ("TR05-12345", 1, 30.0, 0.0)])
    assert list(is_unidentified(weekly, ("99999",))) == [True, True, False]


def test_is_unidentified_ignores_the_key_column():
    # A key that looks like a placeholder does not matter; only customer_id is read.
    table = pd.DataFrame({"key": ["99999", "TR02-1"], "customer_id": ["1", "99999"]})
    assert list(is_unidentified(table, ("99999",))) == [False, True]


def test_is_unidentified_works_on_known_and_weekly_tables():
    known = known_table([("TR02", "99999", 0), ("TR02", "K", 0)])
    weekly = weekly_table([("TR02-K", 1, 5.0, 0.0), ("TR02-99999", 1, 7.0, 0.0)])
    assert list(known.loc[is_unidentified(known, ("99999",)), "key"]) == ["TR02-99999"]
    assert list(weekly.loc[is_unidentified(weekly, ("99999",)), "key"]) == ["TR02-99999"]


def test_is_unidentified_returns_a_bool_series_on_the_table_index():
    table = ids_table(["99999", "12345", "SUSPENSE"])
    result = is_unidentified(table, ("99999",))
    assert isinstance(result, pd.Series)
    assert result.dtype == bool
    assert list(result.index) == list(table.index)


def test_is_unidentified_does_not_normalise_the_table_side():
    # The table's customer_id is assumed normalised already (by add_keys); a raw spelling
    # in the table is not matched.
    table = ids_table(["0099999", "suspense"])
    assert not is_unidentified(table, ("99999", "SUSPENSE")).any()


@pytest.mark.parametrize("bad", ["", "   ", 99999], ids=["empty", "blank", "number"])
def test_is_unidentified_bad_placeholder_raises(bad):
    with pytest.raises(ValueError):
        is_unidentified(ids_table(["99999"]), (bad,))


def test_is_unidentified_input_not_modified():
    table = ids_table(["99999", "12345"])
    copy = table.copy(deep=True)
    placeholders = ("0099999",)
    is_unidentified(table, placeholders)
    pd.testing.assert_frame_equal(table, copy)
    assert placeholders == ("0099999",)


def test_is_unidentified_masks_split_the_table_without_overlap():
    weekly = weekly_table([("TR02-99999", 1, 10.0, 0.0), ("TR02-K", 1, 20.0, -1.0),
                           ("TR04-99999", 2, 5.0, -2.0)])
    mask = is_unidentified(weekly, ("99999",))
    assert len(weekly[mask]) + len(weekly[~mask]) == len(weekly)
    assert weekly[mask]["gross_receipts"].sum() + weekly[~mask]["gross_receipts"].sum() == 35.0


# === unidentified_cash_by_week (table G) ===============================================

def unidentified(table, company, week):
    hit = table[(table["company_code"] == company) & (table["week"] == week)]
    assert len(hit) == 1, (company, week, len(hit))
    return hit.iloc[0]["unidentified_cash"]


def test_unidentified_constant():
    assert UNIDENTIFIED_COLUMNS == ["company_code", "week", "unidentified_cash"]


@pytest.fixture
def unidentified_example():
    # Weeks 0-5, week 3 missing. TR02's placeholder pays in weeks 1 and 4 (with a refund in
    # 4), TR04's in week 2 and week 3 (missing); TR05 has none.
    weekly = weekly_table([("TR02-99999", 1, 100.0, 0.0), ("TR02-99999", 4, 12.5, -40.0),
                           ("TR04-99999", 2, 7.0, 0.0), ("TR04-99999", 3, 1000.0, 0.0)])
    return unidentified_cash_by_week(weekly, weeks_table(5, missing={3}),
                                     ["TR05", "TR02", "TR04"])


def test_unidentified_values(unidentified_example):
    assert unidentified(unidentified_example, "TR02", 1) == 100.0
    assert unidentified(unidentified_example, "TR02", 4) == 12.5
    assert unidentified(unidentified_example, "TR04", 2) == 7.0


def test_unidentified_sums_gross_receipts_not_refunds(unidentified_example):
    # Week 4: gross 12.5, refund -40; G is the gross receipts only.
    assert unidentified(unidentified_example, "TR02", 4) == 12.5
    assert (unidentified_example["unidentified_cash"] >= 0).all()


def test_unidentified_refunds_do_not_change_it():
    weeks = weeks_table(3)
    base = weekly_table([("TR02-99999", 1, 10.0, 0.0)])
    with_refunds = weekly_table([("TR02-99999", 1, 10.0, -999.0), ("TR02-99999", 2, 0.0, -5.0)])
    pd.testing.assert_frame_equal(unidentified_cash_by_week(base, weeks, ["TR02"]),
                                  unidentified_cash_by_week(with_refunds, weeks, ["TR02"]))


def test_unidentified_zero_elsewhere(unidentified_example):
    assert unidentified(unidentified_example, "TR02", 0) == 0.0
    assert unidentified(unidentified_example, "TR02", 2) == 0.0
    assert unidentified(unidentified_example, "TR04", 4) == 0.0
    tr05 = unidentified_example[unidentified_example["company_code"] == "TR05"]
    assert (tr05["unidentified_cash"] == 0.0).all()


def test_unidentified_grid_is_every_company_and_week_with_data(unidentified_example):
    expected = {(c, w) for c in ("TR02", "TR04", "TR05") for w in (0, 1, 2, 4, 5)}
    assert set(zip(unidentified_example["company_code"], unidentified_example["week"])) == expected
    assert len(unidentified_example) == len(expected)


def test_unidentified_missing_week_absent_and_its_cash_dropped(unidentified_example):
    assert 3 not in set(unidentified_example["week"])
    assert unidentified_example["unidentified_cash"].sum() == 119.5  # not the 1000 of week 3


def test_unidentified_weeks_outside_the_calendar_ignored():
    weekly = weekly_table([("TR02-99999", -1, 5.0, 0.0), ("TR02-99999", 1, 3.0, 0.0),
                           ("TR02-99999", 9, 50.0, 0.0)])
    table = unidentified_cash_by_week(weekly, weeks_table(2), ["TR02"])
    assert list(table["week"]) == [0, 1, 2]
    assert table["unidentified_cash"].sum() == 3.0


def test_unidentified_several_placeholders_in_one_company_week_add_up():
    weekly = weekly_table([("TR02-99999", 1, 3.0, 0.0), ("TR02-SUSPENSE", 1, 4.25, 0.0)])
    table = unidentified_cash_by_week(weekly, weeks_table(2), ["TR02"])
    assert unidentified(table, "TR02", 1) == 7.25


def test_unidentified_company_not_in_list_dropped():
    weekly = weekly_table([("TR02-99999", 1, 3.0, 0.0), ("TR06-99999", 1, 80.0, 0.0)])
    table = unidentified_cash_by_week(weekly, weeks_table(2), ["TR02"])
    assert set(table["company_code"]) == {"TR02"}
    assert table["unidentified_cash"].sum() == 3.0


def test_unidentified_duplicate_companies_counted_once():
    weekly = weekly_table([("TR02-99999", 1, 3.0, 0.0)])
    table = unidentified_cash_by_week(weekly, weeks_table(2), ["TR02", "TR02", "TR04"])
    assert len(table) == 2 * 3
    assert unidentified(table, "TR02", 1) == 3.0


def test_unidentified_empty_input_gives_all_zeros():
    table = unidentified_cash_by_week(weekly_table([]), weeks_table(3, missing={1}),
                                      ["TR02", "TR04"])
    assert len(table) == 2 * 3
    assert (table["unidentified_cash"] == 0.0).all()
    assert table["unidentified_cash"].dtype == "float64"
    assert list(table.columns) == UNIDENTIFIED_COLUMNS


def test_unidentified_empty_filtered_weekly_gives_all_zeros():
    # As in the pipeline with unidentified_customers = (): an empty slice of a real table.
    weekly = weekly_table([("TR02-K", 1, 3.0, 0.0)])
    table = unidentified_cash_by_week(weekly[is_unidentified(weekly, ())], weeks_table(2),
                                      ["TR02"])
    assert list(table["unidentified_cash"]) == [0.0, 0.0, 0.0]


def test_unidentified_columns_exactly(unidentified_example):
    assert list(unidentified_example.columns) == UNIDENTIFIED_COLUMNS


def test_unidentified_dtypes_like_the_refunds_table(unidentified_example):
    assert unidentified_example["company_code"].dtype == object
    assert unidentified_example["week"].dtype == "int64"
    assert unidentified_example["unidentified_cash"].dtype == "float64"
    refunds = refunds_by_week(weekly_table([]), weeks_table(5, missing={3}),
                              ["TR05", "TR02", "TR04"])
    assert list(unidentified_example.dtypes) == list(refunds.dtypes)
    pd.testing.assert_frame_equal(unidentified_example[["company_code", "week"]],
                                  refunds[["company_code", "week"]])


def test_unidentified_sorted_by_company_then_week(unidentified_example):
    ordered = unidentified_example.sort_values(["company_code", "week"], ignore_index=True)
    pd.testing.assert_frame_equal(unidentified_example, ordered)
    assert list(unidentified_example["company_code"].unique()) == ["TR02", "TR04", "TR05"]


def test_unidentified_index_is_range(unidentified_example):
    assert list(unidentified_example.index) == list(range(len(unidentified_example)))


def test_unidentified_no_nan(unidentified_example):
    assert not unidentified_example.isna().any().any()


def test_unidentified_inputs_not_modified():
    weekly = weekly_table([("TR02-99999", 1, 10.0, -1.0), ("TR04-99999", 7, 2.0, 0.0)])
    weeks = weeks_table(4, missing={2})
    companies = ["TR04", "TR02", "TR04"]
    copies = (weekly.copy(deep=True), weeks.copy(deep=True), list(companies))
    unidentified_cash_by_week(weekly, weeks, companies)
    pd.testing.assert_frame_equal(weekly, copies[0])
    pd.testing.assert_frame_equal(weeks, copies[1])
    assert companies == copies[2]


def test_unidentified_row_order_of_inputs_does_not_matter():
    weekly = weekly_table([("TR02-99999", 1, 10.0, 0.0), ("TR04-99999", 3, 2.0, 0.0),
                           ("TR02-99999", 2, 1.0, 0.0)])
    weeks = weeks_table(4, missing={2})
    a = unidentified_cash_by_week(weekly, weeks, ["TR02", "TR04"])
    b = unidentified_cash_by_week(weekly.iloc[::-1].reset_index(drop=True),
                                  weeks.iloc[::-1].reset_index(drop=True), ["TR04", "TR02"])
    pd.testing.assert_frame_equal(a, b)


# === company_week_totals ===============================================================

def test_company_week_totals_column_and_name():
    weekly = weekly_table([("TR02-K", 1, 10.0, -1.0), ("TR02-L", 1, 5.0, -2.0)])
    weeks = weeks_table(2)
    gross = company_week_totals(weekly, weeks, ["TR02"], "gross_receipts", "total")
    assert list(gross.columns) == ["company_code", "week", "total"]
    assert list(gross["total"]) == [0.0, 15.0, 0.0]
    neg = company_week_totals(weekly, weeks, ["TR02"], "refunds", "out")
    assert list(neg.columns) == ["company_code", "week", "out"]
    assert list(neg["out"]) == [0.0, -3.0, 0.0]


def test_company_week_totals_other_numeric_column():
    weekly = weekly_table([("TR02-K", 1, 10.0, -1.0), ("TR02-L", 1, 5.0, 0.0)])
    table = company_week_totals(weekly, weeks_table(1), ["TR02"], "lines", "lines_total")
    assert list(table["lines_total"]) == [0.0, 3.0]  # K has 2 lines (refund), L has 1
    assert table["lines_total"].dtype == "float64"


def test_company_week_totals_name_may_equal_the_column():
    weekly = weekly_table([("TR02-K", 0, 4.0, 0.0)])
    table = company_week_totals(weekly, weeks_table(0), ["TR02"], "gross_receipts",
                                "gross_receipts")
    assert list(table.columns) == ["company_code", "week", "gross_receipts"]
    assert list(table["gross_receipts"]) == [4.0]


def test_company_week_totals_companies_deduplicated_and_sorted():
    weekly = weekly_table([("TR04-K", 0, 4.0, 0.0)])
    table = company_week_totals(weekly, weeks_table(1), ["TR06", "TR04", "TR06", "TR04", "TR02"],
                                "gross_receipts", "g")
    assert list(table["company_code"]) == ["TR02", "TR02", "TR04", "TR04", "TR06", "TR06"]
    assert list(table["week"]) == [0, 1] * 3
    assert table["g"].sum() == 4.0


def test_company_week_totals_accepts_a_tuple_of_companies():
    table = company_week_totals(weekly_table([]), weeks_table(1), ("TR02", "TR04"),
                                "gross_receipts", "g")
    assert len(table) == 4


def test_company_week_totals_no_companies_gives_empty_table():
    table = company_week_totals(weekly_table([("TR02-K", 0, 4.0, 0.0)]), weeks_table(1), [],
                                "gross_receipts", "g")
    assert len(table) == 0
    assert list(table.columns) == ["company_code", "week", "g"]


def test_company_week_totals_all_weeks_missing_gives_empty_table():
    weeks = weeks_table(2, missing={0, 1, 2})
    table = company_week_totals(weekly_table([("TR02-K", 0, 4.0, 0.0)]), weeks, ["TR02"],
                                "gross_receipts", "g")
    assert len(table) == 0


def test_company_week_totals_unknown_column_raises():
    with pytest.raises(KeyError):
        company_week_totals(weekly_table([("TR02-K", 0, 4.0, 0.0)]), weeks_table(1), ["TR02"],
                            "no_such_column", "g")


def test_refunds_by_week_is_company_week_totals_of_refunds():
    weekly = weekly_table([("TR02-K", 1, 100.0, -10.0), ("TR04-B", 2, 0.0, -7.0),
                           ("TR02-L", 4, 50.0, -2.25)])
    weeks = weeks_table(5, missing={3})
    companies = ["TR04", "TR02", "TR05"]
    pd.testing.assert_frame_equal(
        refunds_by_week(weekly, weeks, companies),
        company_week_totals(weekly, weeks, companies, "refunds", "refunds"))


def test_unidentified_cash_by_week_is_company_week_totals_of_gross_receipts():
    weekly = weekly_table([("TR02-99999", 1, 100.0, -10.0), ("TR04-99999", 2, 3.0, 0.0)])
    weeks = weeks_table(3, missing={1})
    companies = ["TR04", "TR02"]
    expected = company_week_totals(weekly, weeks, companies, "gross_receipts",
                                   "unidentified_cash")
    pd.testing.assert_frame_equal(unidentified_cash_by_week(weekly, weeks, companies), expected)


# === tie-out by hand: panel + A + G = the company's gross receipts =====================

PLACEHOLDER = "99999"


@pytest.fixture
def tie_out():
    """TR02 with K (known from week 0), N (new, known at week 3) and the placeholder P
    (known 0, in `known` like any key); TR04 with B (known 0) and its own placeholder.
    Weeks 0-5, week 4 missing; horizons 2."""
    known_all = known_table([("TR02", "K", 0), ("TR02", "N", 3), ("TR02", PLACEHOLDER, 0),
                             ("TR04", "B", 0), ("TR04", PLACEHOLDER, 1)])
    weekly = weekly_table([
        ("TR02-K", 1, 100.0, 0.0), ("TR02-99999", 1, 40.0, -3.0), ("TR02-N", 1, 25.0, 0.0),
        ("TR02-K", 2, 10.0, -1.0), ("TR02-99999", 3, 7.0, 0.0), ("TR02-N", 3, 5.0, 0.0),
        ("TR02-99999", 4, 900.0, 0.0), ("TR02-K", 5, 1.0, 0.0), ("TR02-99999", 5, 2.0, 0.0),
        ("TR04-B", 2, 60.0, 0.0), ("TR04-99999", 2, 6.0, 0.0), ("TR04-99999", 5, 9.5, 0.0),
    ])
    weeks = weeks_table(5, missing={4})
    placeholders = ("0099999",)  # as an SAP export might spell it

    known = known_all[~is_unidentified(known_all, placeholders)]
    identified = weekly[~is_unidentified(weekly, placeholders)]
    unidentified_weekly = weekly[is_unidentified(weekly, placeholders)]
    companies = list(known["company_code"].unique())

    labelled = add_labels(panel_rows(known, 5, 2), identified, weeks)
    new_cash = new_customer_cash(identified, known, weeks, 2, companies)
    g = unidentified_cash_by_week(unidentified_weekly, weeks, companies)
    return weekly, known_all, known, labelled, new_cash, g


def test_tie_out_placeholder_has_no_panel_rows(tie_out):
    _, known_all, known, labelled, _, _ = tie_out
    assert PLACEHOLDER not in set(labelled["customer_id"])
    assert len(known) == len(known_all) - 2
    with_placeholder = panel_rows(known_all, 5, 2)
    assert len(labelled) < len(with_placeholder)


def test_tie_out_panel_plus_new_plus_unidentified_is_gross_receipts(tie_out):
    weekly, _, _, labelled, new_cash, g = tie_out
    in_panel = (labelled[labelled["label_status"] == "labelled"]
                .groupby(["company_code", "origin_week", "h"])["amount"].sum()
                .rename("in_panel").reset_index())
    receipts = (weekly.groupby(["company_code", "week"])["gross_receipts"].sum()
                .rename("receipts").reset_index().rename(columns={"week": "target_week"}))
    g = g.rename(columns={"week": "target_week"})
    check = (new_cash.merge(in_panel, on=["company_code", "origin_week", "h"], how="left")
             .merge(g, on=["company_code", "target_week"], how="left")
             .merge(receipts, on=["company_code", "target_week"], how="left")
             .fillna({"in_panel": 0.0, "receipts": 0.0}))
    assert not check["unidentified_cash"].isna().any()  # G has every target week of A
    total = check["in_panel"] + check["new_customer_cash"] + check["unidentified_cash"]
    assert ((total - check["receipts"]).abs() <= 1e-9).all(), check.to_dict("records")
    assert len(check) == len(new_cash)  # one check per row of A


def test_tie_out_named_points(tie_out):
    _, _, _, labelled, new_cash, g = tie_out

    def parts(company, origin, h):
        rows = labelled[(labelled["company_code"] == company)
                        & (labelled["origin_week"] == origin) & (labelled["h"] == h)]
        panel = rows["amount"].sum()
        new = new_cash[(new_cash["company_code"] == company)
                       & (new_cash["origin_week"] == origin)
                       & (new_cash["h"] == h)]["new_customer_cash"].item()
        return panel, new, unidentified(g, company, origin + h)

    # TR02, target week 1 (receipts 165): K 100 in the panel, N 25 new, P 40 unidentified.
    assert parts("TR02", 0, 1) == (100.0, 25.0, 40.0)
    # TR02, target week 3 from origin 2 (receipts 12): N still new (known 3 > 2).
    assert parts("TR02", 2, 1) == (0.0, 5.0, 7.0)
    # ... and from origin 3? No: target 4 is missing. From origin 1, h = 2: same split.
    assert parts("TR02", 1, 2) == (0.0, 5.0, 7.0)
    # TR04, target week 5 (receipts 9.5): only the placeholder paid.
    assert parts("TR04", 3, 2) == (0.0, 0.0, 9.5)
    assert parts("TR04", 1, 1) == (60.0, 0.0, 6.0)


def test_tie_out_missing_week_cash_is_in_no_table(tie_out):
    _, _, _, labelled, new_cash, g = tie_out
    assert 4 not in set(g["week"])
    assert 4 not in set(new_cash["target_week"])
    assert labelled.loc[labelled["target_week"] == 4, "amount"].isna().all()


def test_tie_out_refunds_use_all_weekly_amounts(tie_out):
    weekly, _, known, _, _, _ = tie_out
    refunds = refunds_by_week(weekly, weeks_table(5, missing={4}),
                              list(known["company_code"].unique()))
    # The placeholder's -3 in week 1 is in R (R takes every key).
    assert refund(refunds, "TR02", 1) == -3.0
    assert refunds["refunds"].sum() == -4.0


def placeholder_only_company(companies=None):
    """TR02 with an identified key K; TR06 whose only key is the placeholder. Built as the
    pipeline does, with `companies` the full list (default) or the old filtered one."""
    known_all = known_table([("TR02", "K", 0), ("TR06", PLACEHOLDER, 0)])
    weekly = weekly_table([("TR02-K", 1, 10.0, 0.0), ("TR06-99999", 1, 50.0, -5.0)])
    weeks = weeks_table(2)
    placeholders = (PLACEHOLDER,)
    known = known_all[~is_unidentified(known_all, placeholders)]
    identified = weekly[~is_unidentified(weekly, placeholders)]
    if companies is None:
        companies = list(known["company_code"].unique())
    g = unidentified_cash_by_week(weekly[is_unidentified(weekly, placeholders)], weeks,
                                  companies)
    r = refunds_by_week(weekly, weeks, companies)
    panel = add_labels(panel_rows(known, 2, 2), identified, weeks)
    new_cash = new_customer_cash(identified, known, weeks, 2, companies)
    return panel, new_cash, g, r, weekly, companies


def test_tie_out_placeholder_only_company_has_g_and_r_rows_with_the_full_list():
    # The pipeline takes the companies from the settings ([companies] known), not from the
    # known table after the placeholders are removed.
    _, _, g, r, _, _ = placeholder_only_company(["TR02", "TR06"])
    assert unidentified(g, "TR06", 1) == 50.0
    assert refund(r, "TR06", 1) == -5.0


def test_tie_out_placeholder_only_company_passes_with_the_full_list():
    # TR06's 50 is in G, its -5 in R, and A has its rows (all 0.0), so the tie-out holds.
    panel, new_cash, g, r, weekly, companies = placeholder_only_company(["TR02", "TR06"])
    assert check_side_tables(panel, new_cash, g, r, weekly, companies) == len(new_cash)


def test_tie_out_placeholder_only_company_a_has_its_zero_rows():
    # A follows the company list, not `known`: TR06 (no identified key) gets the same rows
    # as TR02, all 0.0.
    _, new_cash, _, _, _, _ = placeholder_only_company(["TR02", "TR06"])
    assert set(new_cash["company_code"]) == {"TR02", "TR06"}
    tr02 = new_cash[new_cash["company_code"] == "TR02"].reset_index(drop=True)
    tr06 = new_cash[new_cash["company_code"] == "TR06"].reset_index(drop=True)
    expected = tr02.assign(company_code="TR06", new_customer_cash=0.0)
    expected["company_code"] = expected["company_code"].astype(object)
    pd.testing.assert_frame_equal(tr06, expected)
    assert len(tr06) == 3  # (0,1)(0,2)(1,1)


def test_tie_out_placeholder_only_company_fails_with_the_old_filtered_list():
    panel, new_cash, g, r, weekly, companies = placeholder_only_company()
    assert companies == ["TR02"]  # the old rule drops TR06
    assert g["unidentified_cash"].sum() == 0.0  # TR06's 50 is in no table
    with pytest.raises(ValueError, match=re.escape(
            "the weekly amounts have cash of companies not in the list: ['TR06']")):
        check_side_tables(panel, new_cash, g, r, weekly, companies)


def test_tie_out_old_filtered_g_and_r_fail_against_the_full_list():
    panel, new_cash, g, r, weekly, _ = placeholder_only_company()
    with pytest.raises(ValueError, match=re.escape("has no rows for the companies ['TR06']")):
        check_side_tables(panel, new_cash, g, r, weekly, ["TR02", "TR06"])


# === check_side_tables ==================================================================

@pytest.fixture
def side(tie_out):
    """The six inputs of `check_side_tables` from the tie-out example: 7 rows of A per
    company (targets 1, 2, 3, 5; week 4 missing), so 14 checks."""
    weekly, _, _, labelled, new_cash, g = tie_out
    r = refunds_by_week(weekly, weeks_table(5, missing={4}), ["TR02", "TR04"])
    return labelled, new_cash, g, r, weekly, ["TR02", "TR04"]


def bump(table, column, company, origin, h, delta):
    out = table.copy(deep=True)
    hit = ((out["company_code"] == company) & (out["origin_week"] == origin)
           & (out["h"] == h))
    assert hit.sum() == 1
    out.loc[hit, column] += delta
    return out


def test_tie_out_tolerance_constant():
    assert TIE_OUT_TOLERANCE == 0.01


def test_check_passes_and_returns_rows_of_a(side):
    assert len(side[1]) == 14
    assert check_side_tables(*side) == 14


def test_check_return_is_an_int(side):
    assert isinstance(check_side_tables(*side), int)


def test_check_zero_amount_panel_rows_do_not_matter(side):
    panel, new_cash, g, r, weekly, companies = side
    zero_rows = (panel["amount"] == 0.0)
    assert zero_rows.any()
    assert check_side_tables(panel[~zero_rows], new_cash, g, r, weekly, companies) == 14


def test_check_inputs_not_modified(side):
    copies = [t.copy(deep=True) for t in side[:5]]
    companies = list(side[5])
    check_side_tables(*side)
    for original, copy in zip(side[:5], copies):
        pd.testing.assert_frame_equal(original, copy)
    assert side[5] == companies


def test_check_inputs_not_modified_on_failure(side):
    panel, new_cash, g, r, weekly, companies = side
    bad = bump(new_cash, "new_customer_cash", "TR02", 0, 1, 100.0)
    tables = (panel, bad, g, r, weekly)
    copies = [t.copy(deep=True) for t in tables]
    with pytest.raises(ValueError):
        check_side_tables(*tables, companies)
    for original, copy in zip(tables, copies):
        pd.testing.assert_frame_equal(original, copy)


def test_check_accepts_a_tuple_of_companies(side):
    panel, new_cash, g, r, weekly, _ = side
    assert check_side_tables(panel, new_cash, g, r, weekly, ("TR04", "TR02")) == 14


def test_check_duplicate_companies_in_the_list_are_fine(side):
    panel, new_cash, g, r, weekly, _ = side
    assert check_side_tables(panel, new_cash, g, r, weekly, ["TR02", "TR04", "TR02"]) == 14


def test_check_row_order_of_inputs_does_not_matter(side):
    shuffled = [t.iloc[::-1].reset_index(drop=True) for t in side[:5]]
    assert check_side_tables(*shuffled, side[5]) == 14


# --- coverage --------------------------------------------------------------------------

def test_check_cash_of_an_unlisted_company_raises(side):
    panel, new_cash, g, r, weekly, companies = side
    extra = pd.concat([weekly, weekly_table([("TR06-K", 1, 5.0, 0.0)])], ignore_index=True)
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, g, r, extra, companies)
    assert str(caught.value) == ("the weekly amounts have cash of companies not in the list: "
                                 "['TR06']")


def test_check_unlisted_companies_sorted_and_each_once(side):
    panel, new_cash, g, r, weekly, companies = side
    extra = pd.concat([weekly, weekly_table([("TR91-K", 1, 5.0, 0.0), ("TR06-K", 1, 5.0, 0.0),
                                             ("TR91-L", 2, 1.0, 0.0)])], ignore_index=True)
    with pytest.raises(ValueError, match=re.escape("not in the list: ['TR06', 'TR91']")):
        check_side_tables(panel, new_cash, g, r, extra, companies)


def test_check_unlisted_company_with_zero_receipts_still_raises(side):
    # The check is on the company codes present in `weekly`, not on the amounts.
    panel, new_cash, g, r, weekly, companies = side
    extra = pd.concat([weekly, weekly_table([("TR06-K", 1, 0.0, -2.0)])], ignore_index=True)
    with pytest.raises(ValueError, match=re.escape("not in the list: ['TR06']")):
        check_side_tables(panel, new_cash, g, r, extra, companies)


def test_check_company_dropped_from_the_list_raises(side):
    panel, new_cash, g, r, weekly, _ = side
    with pytest.raises(ValueError, match=re.escape("not in the list: ['TR04']")):
        check_side_tables(panel, new_cash, g, r, weekly, ["TR02"])


@pytest.mark.parametrize("name, position", [("A", 1), ("G", 2), ("R", 3)])
def test_check_company_missing_from_a_side_table_raises(side, name, position):
    tables = list(side)
    table = tables[position]
    tables[position] = table[table["company_code"] != "TR04"].reset_index(drop=True)
    with pytest.raises(ValueError) as caught:
        check_side_tables(*tables)
    assert str(caught.value) == f"table {name} has no rows for the companies ['TR04']"


def test_check_listed_company_with_no_cash_must_still_be_in_every_table(side):
    panel, new_cash, g, r, weekly, _ = side
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, g, r, weekly, ["TR02", "TR04", "TR05"])
    assert str(caught.value) == "table A has no rows for the companies ['TR05']"


def test_check_missing_companies_listed_sorted(side):
    panel, new_cash, g, r, weekly, _ = side
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, g.iloc[0:0], r, weekly, ["TR04", "TR02"])
    assert str(caught.value) == "table G has no rows for the companies ['TR02', 'TR04']"


def test_check_coverage_order_a_before_g_before_r(side):
    panel, new_cash, g, r, weekly, companies = side
    a2, g2, r2 = (t[t["company_code"] != "TR04"] for t in (new_cash, g, r))
    with pytest.raises(ValueError, match=r"^table A "):
        check_side_tables(panel, a2, g2, r2, weekly, companies)
    with pytest.raises(ValueError, match=r"^table G "):
        check_side_tables(panel, new_cash, g2, r2, weekly, companies)
    with pytest.raises(ValueError, match=r"^table R "):
        check_side_tables(panel, new_cash, g, r2, weekly, companies)


def test_check_unlisted_cash_reported_before_missing_tables(side):
    panel, new_cash, g, r, weekly, companies = side
    extra = pd.concat([weekly, weekly_table([("TR06-K", 1, 5.0, 0.0)])], ignore_index=True)
    with pytest.raises(ValueError, match=r"^the weekly amounts have cash of companies"):
        check_side_tables(panel, new_cash.iloc[0:0], g, r, extra, companies)


# --- every labelled (company, t, h) of the panel has its row in A ----------------------

def test_check_labelled_panel_cell_without_a_row_raises(side):
    panel, new_cash, g, r, weekly, companies = side
    hit = ((new_cash["company_code"] == "TR02") & (new_cash["origin_week"] == 0)
           & (new_cash["h"] == 1))
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash[~hit], g, r, weekly, companies)
    assert str(caught.value) == ("1 labelled (company, t, h) of the panel have no row in "
                                 "table A (first: [['TR02', 0, 1]])")


def test_check_labelled_cells_without_a_row_counted_and_first_five(side):
    panel, new_cash, g, r, weekly, companies = side
    keep = ((new_cash["company_code"] != "TR02")
            | ((new_cash["origin_week"] == 4) & (new_cash["h"] == 1)))
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash[keep], g, r, weekly, companies)
    assert str(caught.value) == (
        "6 labelled (company, t, h) of the panel have no row in table A (first: "
        "[['TR02', 0, 1], ['TR02', 0, 2], ['TR02', 1, 1], ['TR02', 1, 2], ['TR02', 2, 1]])")


def test_check_missing_week_and_future_panel_rows_are_ignored(side):
    panel, new_cash, g, r, weekly, companies = side
    unlabelled = panel["label_status"] != "labelled"
    assert set(panel.loc[unlabelled, "label_status"]) == {"missing_week", "future"}
    doctored = panel.copy()
    doctored.loc[unlabelled, "amount"] = 1e6  # would break the sums if they were read
    assert check_side_tables(doctored, new_cash, g, r, weekly, companies) == 14


def test_check_unlabelled_panel_rows_need_no_row_in_a(side):
    # Panel rows at target week 4 (missing) and 6 (future) have no row in A; no error.
    panel, new_cash, g, r, weekly, companies = side
    assert {4, 6} <= set(panel["target_week"])
    assert not {4, 6} & set(new_cash["target_week"])
    assert check_side_tables(panel, new_cash, g, r, weekly, companies) == 14


def test_check_empty_panel_loses_the_known_keys_cash(side):
    panel, new_cash, g, r, weekly, companies = side
    with pytest.raises(ValueError, match=r"the panel \+ A \+ G differ"):
        check_side_tables(panel.iloc[0:0], new_cash, g, r, weekly, companies)


# --- panel + A + G = gross receipts ----------------------------------------------------

def test_check_known_key_cash_also_in_a_raises(side):
    # K's 100 in week 1 is in the panel; adding it to A counts it twice.
    panel, new_cash, g, r, weekly, companies = side
    bad = bump(new_cash, "new_customer_cash", "TR02", 0, 1, 100.0)
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, bad, g, r, weekly, companies)
    assert str(caught.value) == ("for 1 (company, t, h) the panel + A + G differ from the "
                                 "gross receipts (first, with the difference in TRY: "
                                 "[['TR02', 0, 1, 100.0]])")


def test_check_cash_missing_from_a_gives_a_negative_difference(side):
    # N's 25 (new for origin 0) removed from A: the cash is lost.
    panel, new_cash, g, r, weekly, companies = side
    bad = bump(new_cash, "new_customer_cash", "TR02", 0, 1, -25.0)
    with pytest.raises(ValueError, match=re.escape("[['TR02', 0, 1, -25.0]]")):
        check_side_tables(panel, bad, g, r, weekly, companies)


def test_check_placeholder_cash_in_both_g_and_panel_raises(tie_out):
    # The placeholder kept in the panel (built from known_all and all weekly amounts) while
    # its cash is also in G: counted twice wherever it paid and was known at t.
    weekly, known_all, _, _, new_cash, g = tie_out
    weeks = weeks_table(5, missing={4})
    panel = add_labels(panel_rows(known_all, 5, 2), weekly, weeks)
    r = refunds_by_week(weekly, weeks, ["TR02", "TR04"])
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, g, r, weekly, ["TR02", "TR04"])
    # TR02's placeholder (known 0) pays in weeks 1, 3, 5: (0,1) (1,2) (2,1) (3,2) (4,1);
    # TR04's (known 1) in weeks 2 and 5: (1,1) (3,2) (4,1); (0,2) has origin 0 < 1.
    assert str(caught.value) == (
        "for 8 (company, t, h) the panel + A + G differ from the gross receipts (first, "
        "with the difference in TRY: [['TR02', 0, 1, 40.0], ['TR02', 1, 2, 7.0], "
        "['TR02', 2, 1, 7.0], ['TR02', 3, 2, 2.0], ['TR02', 4, 1, 2.0]])")


def test_check_g_wrong_value_raises(side):
    panel, new_cash, g, r, weekly, companies = side
    bad = g.copy()
    bad.loc[(bad["company_code"] == "TR04") & (bad["week"] == 5), "unidentified_cash"] = 0.0
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, bad, r, weekly, companies)
    # Target week 5 of TR04: (3, 2) and (4, 1), each short by 9.5.
    assert str(caught.value).startswith("for 2 (company, t, h)")
    assert "[['TR04', 3, 2, -9.5], ['TR04', 4, 1, -9.5]]" in str(caught.value)


def drop_g_rows(g, pairs):
    """`g` without the rows of the given (company, week) pairs."""
    hit = pd.Series(False, index=g.index)
    for company, week in pairs:
        hit |= (g["company_code"] == company) & (g["week"] == week)
    assert hit.sum() == len(pairs)
    return g[~hit]


def test_check_g_missing_a_target_week_raises(side):
    # TR02's G row of week 1 is gone; A has one row with target week 1 for TR02: (0, 1).
    panel, new_cash, g, r, weekly, companies = side
    bad = drop_g_rows(g, [("TR02", 1)])
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, bad, r, weekly, companies)
    assert str(caught.value) == ("table G has no row for 1 (company, target week) of table A "
                                 "(first: [['TR02', 1]])")


def test_check_g_missing_row_of_a_zero_week_still_raises(side):
    # TR02's G is 0.0 in week 2: dropping the row changes no sum, but it is still a fault.
    panel, new_cash, g, r, weekly, companies = side
    assert unidentified(g, "TR02", 2) == 0.0
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, drop_g_rows(g, [("TR02", 2)]), r, weekly, companies)
    # A's rows of TR02 with target week 2, (0, 2) and (1, 1), are one missing pair.
    assert str(caught.value) == ("table G has no row for 1 (company, target week) of table A "
                                 "(first: [['TR02', 2]])")


def test_check_g_several_missing_rows_counted_and_first_five(side):
    # TR02 loses weeks 2, 3, 5 (two rows of A each), TR04 week 1 (one row): 7 rows of A
    # but 4 distinct (company, target week) pairs, each listed once.
    panel, new_cash, g, r, weekly, companies = side
    bad = drop_g_rows(g, [("TR02", 2), ("TR02", 3), ("TR02", 5), ("TR04", 1)])
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, bad, r, weekly, companies)
    assert str(caught.value) == (
        "table G has no row for 4 (company, target week) of table A (first: "
        "[['TR02', 2], ['TR02', 3], ['TR02', 5], ['TR04', 1]])")


def test_check_g_missing_rows_fewer_than_five_all_listed(side):
    panel, new_cash, g, r, weekly, companies = side
    bad = drop_g_rows(g, [("TR04", 1), ("TR04", 5)])
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, bad, r, weekly, companies)
    # TR04: target 1 is (0, 1); target 5 is (3, 2) and (4, 1), listed once.
    assert str(caught.value) == (
        "table G has no row for 2 (company, target week) of table A (first: "
        "[['TR04', 1], ['TR04', 5]])")


def test_check_g_pair_used_by_several_rows_of_a_listed_once(side):
    # TR02's target week 3 is the target of two rows of A, (1, 2) and (2, 1).
    panel, new_cash, g, r, weekly, companies = side
    rows = new_cash[(new_cash["company_code"] == "TR02") & (new_cash["target_week"] == 3)]
    assert sorted(zip(rows["origin_week"], rows["h"])) == [(1, 2), (2, 1)]
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, drop_g_rows(g, [("TR02", 3)]), r, weekly, companies)
    assert str(caught.value) == ("table G has no row for 1 (company, target week) of table A "
                                 "(first: [['TR02', 3]])")


def test_check_g_more_than_five_missing_pairs_counts_distinct_and_lists_five_sorted(side):
    # Every target week of A (1, 2, 3, 5) dropped for both companies, given in a scrambled
    # order: 14 rows of A, 8 distinct pairs; five listed, by company then week.
    panel, new_cash, g, r, weekly, companies = side
    pairs = [("TR04", 5), ("TR02", 3), ("TR04", 1), ("TR02", 5),
             ("TR04", 3), ("TR02", 1), ("TR04", 2), ("TR02", 2)]
    assert set(pairs) == set(zip(new_cash["company_code"], new_cash["target_week"]))
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, drop_g_rows(g, pairs), r, weekly, companies)
    assert str(caught.value) == (
        "table G has no row for 8 (company, target week) of table A (first: "
        "[['TR02', 1], ['TR02', 2], ['TR02', 3], ['TR02', 5], ['TR04', 1]])")


def test_check_g_every_row_of_a_present_passes(side):
    panel, new_cash, g, r, weekly, companies = side
    a_pairs = set(zip(new_cash["company_code"], new_cash["target_week"]))
    assert a_pairs <= set(zip(g["company_code"], g["week"]))
    assert check_side_tables(panel, new_cash, g, r, weekly, companies) == 14


def test_check_g_rows_beyond_those_of_a_are_not_needed(side):
    # G's week 0 rows are the target of no row of A; dropping them is fine.
    panel, new_cash, g, r, weekly, companies = side
    assert 0 not in set(new_cash["target_week"])
    bad = drop_g_rows(g, [("TR02", 0), ("TR04", 0)])
    assert check_side_tables(panel, new_cash, bad, r, weekly, companies) == 14


def test_check_g_company_missing_entirely_reports_coverage_first(side):
    # TR04 gone from G and a row of TR02 gone too: the coverage message wins.
    panel, new_cash, g, r, weekly, companies = side
    bad = drop_g_rows(g[g["company_code"] != "TR04"], [("TR02", 1)])
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, bad, r, weekly, companies)
    assert str(caught.value) == "table G has no rows for the companies ['TR04']"


def test_check_r_coverage_reported_before_missing_g_rows(side):
    panel, new_cash, g, r, weekly, companies = side
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, drop_g_rows(g, [("TR02", 1)]),
                          r[r["company_code"] != "TR04"], weekly, companies)
    assert str(caught.value) == "table R has no rows for the companies ['TR04']"


def test_check_panel_cell_without_a_row_reported_before_missing_g_rows(side):
    panel, new_cash, g, r, weekly, companies = side
    hit = ((new_cash["company_code"] == "TR02") & (new_cash["origin_week"] == 0)
           & (new_cash["h"] == 1))
    with pytest.raises(ValueError, match=r"^1 labelled \(company, t, h\) of the panel have no "):
        check_side_tables(panel, new_cash[~hit], drop_g_rows(g, [("TR02", 2)]), r, weekly,
                          companies)


def test_check_missing_g_rows_reported_before_the_tie_out(side):
    panel, new_cash, g, r, weekly, companies = side
    bad_a = bump(new_cash, "new_customer_cash", "TR04", 1, 1, 50.0)
    with pytest.raises(ValueError, match=r"^table G has no row for 1 \(company, target week\)"):
        check_side_tables(panel, bad_a, drop_g_rows(g, [("TR02", 1)]), r, weekly, companies)


def test_check_missing_g_rows_inputs_not_modified(side):
    panel, new_cash, g, r, weekly, companies = side
    bad = drop_g_rows(g, [("TR02", 1)])
    tables = (panel, new_cash, bad, r, weekly)
    copies = [t.copy(deep=True) for t in tables]
    with pytest.raises(ValueError):
        check_side_tables(*tables, companies)
    for original, copy in zip(tables, copies):
        pd.testing.assert_frame_equal(original, copy)


def test_check_weekly_with_identified_rows_only_breaks_the_tie(side):
    # `weekly` must be ALL the weekly amounts; given the identified rows only, G's cash has
    # nothing to match.
    panel, new_cash, g, r, weekly, companies = side
    identified = weekly[~is_unidentified(weekly, (PLACEHOLDER,))]
    with pytest.raises(ValueError, match=r"^for \d+ \(company, t, h\) the panel \+ A \+ G"):
        check_side_tables(panel, new_cash, g, r, identified, companies)


@pytest.mark.parametrize("delta", [0.005, -0.005, 0.0099])
def test_check_difference_within_tolerance_passes(side, delta):
    panel, new_cash, g, r, weekly, companies = side
    bad = bump(new_cash, "new_customer_cash", "TR04", 1, 1, delta)
    assert check_side_tables(panel, bad, g, r, weekly, companies) == 14


@pytest.mark.parametrize("delta", [0.02, -0.02, 0.0101])
def test_check_difference_beyond_tolerance_raises(side, delta):
    panel, new_cash, g, r, weekly, companies = side
    bad = bump(new_cash, "new_customer_cash", "TR04", 1, 1, delta)
    with pytest.raises(ValueError, match=r"^for 1 \(company, t, h\)"):
        check_side_tables(panel, bad, g, r, weekly, companies)


def test_check_difference_rounded_to_two_decimals(side):
    panel, new_cash, g, r, weekly, companies = side
    bad = bump(new_cash, "new_customer_cash", "TR04", 1, 1, 0.123456)
    with pytest.raises(ValueError, match=re.escape("[['TR04', 1, 1, 0.12]]")):
        check_side_tables(panel, bad, g, r, weekly, companies)


def test_check_breaks_counted_all_listed_first_five(side):
    panel, new_cash, g, r, weekly, companies = side
    bad = new_cash.copy()
    bad["new_customer_cash"] += 1.0
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, bad, g, r, weekly, companies)
    assert str(caught.value) == (
        "for 14 (company, t, h) the panel + A + G differ from the gross receipts (first, "
        "with the difference in TRY: [['TR02', 0, 1, 1.0], ['TR02', 0, 2, 1.0], "
        "['TR02', 1, 1, 1.0], ['TR02', 1, 2, 1.0], ['TR02', 2, 1, 1.0]])")


def test_check_extra_cash_in_a_week_with_no_receipts_raises(side):
    # TR04 has no receipts in week 3: A must be 0 there.
    panel, new_cash, g, r, weekly, companies = side
    bad = bump(new_cash, "new_customer_cash", "TR04", 2, 1, 3.0)
    with pytest.raises(ValueError, match=re.escape("[['TR04', 2, 1, 3.0]]")):
        check_side_tables(panel, bad, g, r, weekly, companies)


def test_check_refunds_play_no_part_in_the_receipts_tie(side):
    panel, new_cash, g, _, weekly, companies = side
    more = weekly.copy()
    more["refunds"] = more["refunds"] - 50.0
    r = refunds_by_week(more, weeks_table(5, missing={4}), companies)
    assert check_side_tables(panel, new_cash, g, r, more, companies) == 14


# --- R = the weekly refunds ------------------------------------------------------------

def test_check_doctored_refund_raises(side):
    panel, new_cash, g, r, weekly, companies = side
    bad = r.copy()
    bad.loc[(bad["company_code"] == "TR02") & (bad["week"] == 1), "refunds"] = -3.5
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, g, bad, weekly, companies)
    assert str(caught.value) == ("for 1 (company, week) table R differs from the weekly "
                                 "refunds (first: [['TR02', 1]])")


@pytest.mark.parametrize("delta, fails", [(0.005, False), (-0.005, False), (0.02, True),
                                          (-0.02, True)])
def test_check_refund_tolerance(side, delta, fails):
    panel, new_cash, g, r, weekly, companies = side
    bad = r.copy()
    bad.loc[(bad["company_code"] == "TR02") & (bad["week"] == 1), "refunds"] += delta
    if fails:
        with pytest.raises(ValueError, match=r"^for 1 \(company, week\) table R"):
            check_side_tables(panel, new_cash, g, bad, weekly, companies)
    else:
        assert check_side_tables(panel, new_cash, g, bad, weekly, companies) == 14


def test_check_refund_in_a_week_without_weekly_refunds_raises(side):
    panel, new_cash, g, r, weekly, companies = side
    bad = r.copy()
    bad.loc[(bad["company_code"] == "TR04") & (bad["week"] == 3), "refunds"] = -1.0
    with pytest.raises(ValueError, match=re.escape("(first: [['TR04', 3]])")):
        check_side_tables(panel, new_cash, g, bad, weekly, companies)


def test_check_refund_breaks_counted_and_first_five(side):
    panel, new_cash, g, r, weekly, companies = side
    bad = r.copy()
    bad["refunds"] -= 1.0
    with pytest.raises(ValueError) as caught:
        check_side_tables(panel, new_cash, g, bad, weekly, companies)
    assert str(caught.value) == (
        "for 10 (company, week) table R differs from the weekly refunds (first: "
        "[['TR02', 0], ['TR02', 1], ['TR02', 2], ['TR02', 3], ['TR02', 5]])")


def test_check_refunds_of_a_missing_week_are_not_compared(side):
    panel, new_cash, g, r, weekly, companies = side
    assert 4 not in set(r["week"])
    more = pd.concat([weekly, weekly_table([("TR02-Q", 4, 0.0, -77.0)])], ignore_index=True)
    assert check_side_tables(panel, new_cash, g, r, more, companies) == 14


def test_check_receipts_tie_checked_before_refunds(side):
    panel, new_cash, g, r, weekly, companies = side
    bad_a = bump(new_cash, "new_customer_cash", "TR02", 0, 1, 1.0)
    bad_r = r.copy()
    bad_r["refunds"] -= 1.0
    with pytest.raises(ValueError, match=r"the panel \+ A \+ G differ"):
        check_side_tables(panel, bad_a, g, bad_r, weekly, companies)


def test_check_placeholder_refunds_must_be_in_r(side):
    # The placeholder's -3 in week 1 is in weekly: an R built from identified rows only
    # fails the check.
    panel, new_cash, g, _, weekly, companies = side
    identified = weekly[~is_unidentified(weekly, (PLACEHOLDER,))]
    r = refunds_by_week(identified, weeks_table(5, missing={4}), companies)
    with pytest.raises(ValueError, match=re.escape("(first: [['TR02', 1]])")):
        check_side_tables(panel, new_cash, g, r, weekly, companies)


# --- integration with the synthetic corpus and ledger (SLOW) ---------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
HOLIDAYS_FILE = REPO_ROOT / "settings" / "holidays_tr.toml"
AS_OF = date(2026, 8, 28)
WEEK_ZERO = date(2025, 6, 2)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]


@pytest.fixture(scope="module")
def synthetic_sources():
    """The keyed daily-file lines, the keyed ledger, the days with a file and the holidays,
    read once for the module."""
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    from collection_estimation.hurdle.holidays import load_holidays
    from collection_estimation.hurdle.keys import add_keys
    from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
    from collection_estimation.hurdle.read_ledger import read_ledger
    from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

    daily_files = list_daily_files(CORPUS_FOLDER, SYNTHETIC_LAYOUT.file_name_date_format)
    days = list(daily_files)
    # Direct reads, never the data/interim cache.
    collections = add_keys(read_all_files(daily_files, SYNTHETIC_LAYOUT), KNOWN_COMPANIES)
    ledger = add_keys(read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT),
                      KNOWN_COMPANIES)
    return collections, ledger, days, load_holidays(HOLIDAYS_FILE)


@pytest.fixture(scope="module")
def synthetic(synthetic_sources):
    from collection_estimation.hurdle.fill_from_ledger import fill_missing_days
    from collection_estimation.hurdle.known import known_weeks
    from collection_estimation.hurdle.panel import add_labels, panel_rows
    from collection_estimation.hurdle.series import add_week_index, week_calendar, weekly_amounts

    collections, ledger, days, holidays = synthetic_sources
    collections, filled_days = fill_missing_days(collections, ledger, days)
    collections = add_week_index(collections, WEEK_ZERO)
    weeks = week_calendar(days, WEEK_ZERO, holidays, filled_days)
    weekly = weekly_amounts(collections)
    known = known_weeks(collections, ledger, WEEK_ZERO)
    new_cash = new_customer_cash(weekly, known, weeks, 5, KNOWN_COMPANIES)
    labelled = add_labels(panel_rows(known, 64, 5), weekly, weeks)
    return weekly, known, weeks, new_cash, labelled


@pytest.fixture(scope="module")
def synthetic_refunds(synthetic):
    weekly, known, weeks, _, _ = synthetic
    return refunds_by_week(weekly, weeks, list(known["company_code"].unique()))


@pytest.fixture(scope="module")
def synthetic_unfilled(synthetic_sources):
    """As the pipeline does with [data] fill_missing_days_from_ledger = false."""
    from collection_estimation.hurdle.series import add_week_index, week_calendar, weekly_amounts

    collections, _, days, holidays = synthetic_sources
    collections = add_week_index(collections.assign(filled_from_ledger=False), WEEK_ZERO)
    weeks = week_calendar(days, WEEK_ZERO, holidays, [])
    weekly = weekly_amounts(collections)
    return weekly, weeks, refunds_by_week(weekly, weeks, KNOWN_COMPANIES)


def test_slow_synthetic_row_count(synthetic):
    _, known, weeks, new_cash, _ = synthetic
    assert int(weeks["week"].max()) == 64
    assert len(new_cash) == 1_550
    assert set(new_cash["company_code"]) == set(known["company_code"])
    assert list(new_cash.columns) == NEW_CASH_COLUMNS


def test_slow_synthetic_share_by_h(synthetic):
    weekly, _, _, new_cash, _ = synthetic
    week_receipts = weekly.groupby("week")["gross_receipts"].sum()
    one_per_target = new_cash.drop_duplicates(["origin_week", "h"])
    denominator = (one_per_target["target_week"].map(week_receipts).fillna(0.0)
                   .groupby(one_per_target["h"]).sum())
    numerator = new_cash.groupby("h")["new_customer_cash"].sum()
    shares = {h: round(100 * numerator[h] / denominator[h], 2) for h in range(1, 6)}
    assert shares == {1: 0.03, 2: 0.06, 3: 0.33, 4: 1.54, 5: 3.61}


def test_slow_synthetic_panel_plus_new_cash_is_company_receipts(synthetic):
    weekly, _, _, new_cash, labelled = synthetic
    in_panel = (labelled.groupby(["company_code", "origin_week", "h"])["amount"]
                .sum().rename("in_panel").reset_index())
    receipts = (weekly.groupby(["company_code", "week"])["gross_receipts"].sum()
                .rename("receipts").reset_index().rename(columns={"week": "target_week"}))
    check = (new_cash.merge(in_panel, on=["company_code", "origin_week", "h"], how="left")
             .merge(receipts, on=["company_code", "target_week"], how="left")
             .fillna({"in_panel": 0.0, "receipts": 0.0}))
    gap = (check["in_panel"] + check["new_customer_cash"] - check["receipts"]).abs()
    assert gap.max() <= 0.01, check.loc[gap.idxmax()].to_dict()
    # A few named points, spelled out.
    for company, origin, h in [("TR02", 0, 5), ("TR04", 30, 3), ("TR05", 63, 1),
                               ("TR06", 10, 4), ("TR91", 59, 5)]:
        point = check[(check["company_code"] == company) & (check["origin_week"] == origin)
                      & (check["h"] == h)]
        assert len(point) == 1
        point = point.iloc[0]
        assert abs(point["in_panel"] + point["new_customer_cash"] - point["receipts"]) <= 0.01


def test_slow_synthetic_new_cash_never_negative(synthetic):
    *_, new_cash, _ = synthetic
    assert (new_cash["new_customer_cash"] >= 0).all()


def test_slow_synthetic_refunds_row_count(synthetic_refunds, synthetic):
    _, known, weeks, _, _ = synthetic
    assert not weeks["is_missing"].any()  # filling on: every week has data
    assert len(synthetic_refunds) == 325  # 5 companies x 65 weeks
    assert set(synthetic_refunds["company_code"]) == set(known["company_code"])
    assert sorted(synthetic_refunds["company_code"].unique()) == KNOWN_COMPANIES
    assert list(synthetic_refunds.columns) == REFUND_COLUMNS


def test_slow_synthetic_refunds_total(synthetic_refunds, synthetic):
    weekly = synthetic[0]
    total = synthetic_refunds["refunds"].sum()
    assert abs(total - (-31_043_453.84)) <= 0.01
    assert abs(total - weekly["refunds"].sum()) <= 0.01


def test_slow_synthetic_refunds_weeks_with_refunds(synthetic_refunds):
    assert synthetic_refunds.loc[synthetic_refunds["refunds"] < 0, "week"].nunique() == 55
    assert (synthetic_refunds["refunds"] <= 0).all()


def test_slow_synthetic_refunds_per_company(synthetic_refunds, synthetic):
    weekly = synthetic[0]
    ours = synthetic_refunds.groupby("company_code")["refunds"].sum()
    theirs = weekly.groupby("company_code")["refunds"].sum().reindex(ours.index,
                                                                      fill_value=0.0)
    assert ((ours - theirs).abs() <= 0.01).all(), (ours - theirs).to_dict()


def test_slow_synthetic_refunds_per_company_and_week(synthetic_refunds, synthetic):
    weekly = synthetic[0]
    theirs = weekly.groupby(["company_code", "week"])["refunds"].sum()
    ours = synthetic_refunds.set_index(["company_code", "week"])["refunds"]
    gap = (ours - theirs.reindex(ours.index, fill_value=0.0)).abs()
    assert gap.max() <= 1e-6


def test_slow_synthetic_refunds_unfilled_missing_weeks_have_no_rows(synthetic_unfilled):
    _, weeks, refunds = synthetic_unfilled
    missing = set(weeks.loc[weeks["is_missing"], "week"])
    assert len(missing) == 6
    assert not missing & set(refunds["week"])
    assert set(refunds["week"]) == set(weeks.loc[~weeks["is_missing"], "week"])
    assert len(refunds) == 5 * (65 - 6)


def test_slow_synthetic_refunds_unfilled_match_weekly_in_weeks_with_data(synthetic_unfilled):
    weekly, weeks, refunds = synthetic_unfilled
    with_data = weeks.loc[~weeks["is_missing"], "week"]
    expected = weekly.loc[weekly["week"].isin(with_data), "refunds"].sum()
    assert abs(refunds["refunds"].sum() - expected) <= 0.01


# --- unidentified cash on the synthetic data (SLOW) -------------------------------------
# The synthetic data has no placeholder customer, so one is made: the customer_id with the
# largest total gross receipts, computed from the data (never hard-coded).

@pytest.fixture(scope="module")
def synthetic_placeholder(synthetic):
    weekly, known, weeks, _, _ = synthetic
    totals = weekly.groupby("customer_id")["gross_receipts"].sum()
    top = totals.idxmax()
    placeholders = (top,)
    companies = list(known["company_code"].unique())
    known_without = known[~is_unidentified(known, placeholders)]
    identified = weekly[~is_unidentified(weekly, placeholders)]
    unidentified_weekly = weekly[is_unidentified(weekly, placeholders)]
    g = unidentified_cash_by_week(unidentified_weekly, weeks, companies)
    return top, known_without, identified, unidentified_weekly, g


def test_slow_synthetic_placeholder_is_a_real_customer(synthetic_placeholder, synthetic):
    top, _, _, unidentified_weekly, _ = synthetic_placeholder
    weekly = synthetic[0]
    assert len(unidentified_weekly) > 0
    assert set(unidentified_weekly["customer_id"]) == {top}
    assert (weekly.groupby("customer_id")["gross_receipts"].sum().max()
            == weekly.loc[weekly["customer_id"] == top, "gross_receipts"].sum())


def test_slow_synthetic_g_total_is_the_placeholders_receipts(synthetic_placeholder, synthetic):
    top, _, _, _, g = synthetic_placeholder
    weekly, _, weeks, _, _ = synthetic
    with_data = set(weeks.loc[~weeks["is_missing"], "week"])
    rows = weekly[(weekly["customer_id"] == top) & weekly["week"].isin(with_data)]
    assert abs(g["unidentified_cash"].sum() - rows["gross_receipts"].sum()) <= 0.01
    assert g["unidentified_cash"].sum() > 0


def test_slow_synthetic_g_per_company_and_week(synthetic_placeholder, synthetic):
    top, _, _, _, g = synthetic_placeholder
    weekly = synthetic[0]
    theirs = (weekly[weekly["customer_id"] == top]
              .groupby(["company_code", "week"])["gross_receipts"].sum())
    ours = g.set_index(["company_code", "week"])["unidentified_cash"]
    gap = (ours - theirs.reindex(ours.index, fill_value=0.0)).abs()
    assert gap.max() <= 1e-6
    assert len(g) == 325 and list(g.columns) == UNIDENTIFIED_COLUMNS


def test_slow_synthetic_panel_without_placeholder_has_fewer_rows(synthetic_placeholder, synthetic):
    top, known_without, _, _, _ = synthetic_placeholder
    _, known, _, _, labelled = synthetic
    removed = known[known["customer_id"] == top]
    assert len(known_without) == len(known) - len(removed) and len(removed) >= 1
    panel_without = panel_rows(known_without, 64, 5)
    assert len(panel_without) < len(labelled)
    assert top not in set(panel_without["customer_id"])
    # Exactly the placeholder's rows are gone.
    assert len(labelled) - len(panel_without) == (labelled["customer_id"] == top).sum()


def test_slow_synthetic_tie_out_with_placeholder(synthetic_placeholder, synthetic):
    _, known_without, identified, _, g = synthetic_placeholder
    weekly, _, weeks, _, _ = synthetic
    labelled = add_labels(panel_rows(known_without, 64, 5), identified, weeks)
    new_cash = new_customer_cash(identified, known_without, weeks, 5, KNOWN_COMPANIES)
    in_panel = (labelled.groupby(["company_code", "origin_week", "h"])["amount"]
                .sum().rename("in_panel").reset_index())
    receipts = (weekly.groupby(["company_code", "week"])["gross_receipts"].sum()
                .rename("receipts").reset_index().rename(columns={"week": "target_week"}))
    check = (new_cash.merge(in_panel, on=["company_code", "origin_week", "h"], how="left")
             .merge(g.rename(columns={"week": "target_week"}),
                    on=["company_code", "target_week"], how="left")
             .merge(receipts, on=["company_code", "target_week"], how="left")
             .fillna({"in_panel": 0.0, "receipts": 0.0}))
    assert not check["unidentified_cash"].isna().any()
    gap = (check["in_panel"] + check["new_customer_cash"] + check["unidentified_cash"]
           - check["receipts"]).abs()
    assert gap.max() <= 0.01, check.loc[gap.idxmax()].to_dict()


def test_slow_synthetic_no_placeholders_g_is_all_zero(synthetic):
    weekly, known, weeks, _, _ = synthetic
    mask = is_unidentified(weekly, ())
    assert not mask.any()
    assert not is_unidentified(known, ()).any()
    g = unidentified_cash_by_week(weekly[mask], weeks, list(known["company_code"].unique()))
    assert len(g) == 325
    assert (g["unidentified_cash"] == 0.0).all()


# --- check_side_tables on the synthetic data (SLOW) -------------------------------------

def test_slow_synthetic_check_side_tables_filling_on(synthetic):
    weekly, _, weeks, new_cash, labelled = synthetic
    g = unidentified_cash_by_week(weekly[is_unidentified(weekly, ())], weeks, KNOWN_COMPANIES)
    r = refunds_by_week(weekly, weeks, KNOWN_COMPANIES)
    assert check_side_tables(labelled, new_cash, g, r, weekly, KNOWN_COMPANIES) == 1_550


def test_slow_synthetic_check_side_tables_filling_off(synthetic_sources, synthetic_unfilled):
    from collection_estimation.hurdle.known import known_weeks
    from collection_estimation.hurdle.series import add_week_index

    collections, ledger, _, _ = synthetic_sources
    weekly, weeks, refunds = synthetic_unfilled
    collections = add_week_index(collections.assign(filled_from_ledger=False), WEEK_ZERO)
    known = known_weeks(collections, ledger, WEEK_ZERO)
    labelled = add_labels(panel_rows(known, int(weeks["week"].max()), 5), weekly, weeks)
    new_cash = new_customer_cash(weekly, known, weeks, 5, KNOWN_COMPANIES)
    g = unidentified_cash_by_week(weekly[is_unidentified(weekly, ())], weeks, KNOWN_COMPANIES)
    assert check_side_tables(labelled, new_cash, g, refunds, weekly, KNOWN_COMPANIES) == 1_400


def test_slow_synthetic_check_side_tables_with_placeholder(synthetic_placeholder, synthetic):
    _, known_without, identified, _, g = synthetic_placeholder
    weekly, _, weeks, _, _ = synthetic
    labelled = add_labels(panel_rows(known_without, 64, 5), identified, weeks)
    new_cash = new_customer_cash(identified, known_without, weeks, 5, KNOWN_COMPANIES)
    r = refunds_by_week(weekly, weeks, KNOWN_COMPANIES)
    assert check_side_tables(labelled, new_cash, g, r, weekly, KNOWN_COMPANIES) == 1_550


def test_slow_synthetic_check_side_tables_placeholder_also_in_panel_fails(
        synthetic_placeholder, synthetic):
    # The full panel still holds the placeholder while G holds its cash too.
    *_, g = synthetic_placeholder
    weekly, _, weeks, new_cash, labelled = synthetic
    r = refunds_by_week(weekly, weeks, KNOWN_COMPANIES)
    with pytest.raises(ValueError, match=r"the panel \+ A \+ G differ"):
        check_side_tables(labelled, new_cash, g, r, weekly, KNOWN_COMPANIES)
