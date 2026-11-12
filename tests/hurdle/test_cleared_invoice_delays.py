"""Tests for `cleared_invoice_delays` in `hurdle/receivables_asof.py` (U18.1), and for
`payment_delays_at` rebuilt on it.

`cleared_invoice_delays(ledger)` gives one row per cleared invoice that has at least one
payment in its clearing group (same company code and `clearing_document`): the money arrived
on the latest posting date of those payments, days late = that date - the net due date, and
known_on = the latest of the invoice's posting date, its clearing date and the arrival. It
raises ValueError on an invoice in the ledger twice, a cleared invoice with no clearing
document, and a payment of a group posted after the invoice's clearing date.

`payment_delays_at(ledger, cutoff, min_cleared)` filters those rows to known_on <= cutoff.
Its behaviour must equal the old definition (invoices posted and cleared by the cutoff,
payments posted by the cutoff) on any ledger with no payment after its clearing; that is
checked here against an independent re-computation of the old definition, on seeded random
ledgers.

Ledgers are built with the line builders of test_payment_delays.py. The SLOW tests at the end
read the synthetic ledger in `data/raw/` once and are skipped when it is absent; the repo's
`data/interim/` cache is never used.
"""
import itertools
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.keys import add_keys
from collection_estimation.hurdle.read_ledger import read_ledger
from collection_estimation.hurdle.receivables_asof import (
    CLEARED_DELAY_COLUMNS,
    DELAY_COLUMNS,
    cleared_invoice_delays,
    ledger_as_known_at,
    payment_delays_at,
)
from tests.hurdle.test_payment_delays import (
    credit_memo,
    invoice,
    ledger,
    payment,
    running_example,
)
from tests.hurdle.workbooks import SYNTHETIC_LEDGER_LAYOUT

CUTOFF = date(2026, 4, 30)


def row_of(table, document_no, key=None):
    rows = table[table["document_no"] == document_no]
    if key is not None:
        rows = rows[rows["key"] == key]
    assert len(rows) == 1, (document_no, key, len(rows))
    return rows.iloc[0]


def one_invoice(posted, due, cleared, paid_on, group="CLR-1", **kwargs):
    """One invoice cleared on `cleared` by payments posted on each day of `paid_on`."""
    lines = [invoice("INV1", posted, due, cleared=cleared, clearing_document=group, **kwargs)]
    for n, day in enumerate(paid_on):
        lines.append(payment(f"PAY{n}", day, cleared=cleared, clearing_document=group,
                             **kwargs))
    return ledger(lines)


# --- the constant --------------------------------------------------------------------------

def test_cleared_delay_columns_are_as_documented():
    assert CLEARED_DELAY_COLUMNS == ["key", "company_code", "document_no", "net_due_date",
                                     "money_arrived", "days_late", "known_on"]


# --- the running example -------------------------------------------------------------------

def test_running_example_gives_one_row_with_every_value():
    # Invoice posted 1 Mar, due 31 Mar; payments 20 Mar and 15 Apr; cleared 15 Apr.
    table = cleared_invoice_delays(running_example())
    assert len(table) == 1
    row = table.iloc[0]
    assert row["key"] == "TR02-100"
    assert row["company_code"] == "TR02"
    assert row["document_no"] == "INV1"
    assert row["net_due_date"] == pd.Timestamp("2026-03-31")
    assert row["money_arrived"] == pd.Timestamp("2026-04-15")
    assert row["days_late"] == 15
    assert row["known_on"] == pd.Timestamp("2026-04-15")


# --- days late -----------------------------------------------------------------------------

def test_late_payment_is_positive_days():
    table = cleared_invoice_delays(one_invoice("2026-03-01", "2026-03-31", "2026-04-20",
                                               ["2026-04-07"]))
    assert table["days_late"].tolist() == [7]


def test_early_payment_is_negative_days():
    table = cleared_invoice_delays(one_invoice("2026-03-01", "2026-03-31", "2026-04-20",
                                               ["2026-03-21"]))
    assert table["days_late"].tolist() == [-10]


def test_payment_on_the_due_date_is_zero():
    table = cleared_invoice_delays(one_invoice("2026-03-01", "2026-03-31", "2026-04-20",
                                               ["2026-03-31"]))
    assert table["days_late"].tolist() == [0]


def test_days_late_are_whole_days_across_february():
    table = cleared_invoice_delays(one_invoice("2026-01-01", "2026-02-27", "2026-03-02",
                                               ["2026-03-02"]))
    assert table["days_late"].tolist() == [3]


def test_days_late_counts_from_the_due_date_not_the_posting_or_clearing_date():
    # Posted 1 Mar, due 31 Mar, paid 5 Apr, cleared 25 Apr: 5, not 35 or 25 or 55.
    table = cleared_invoice_delays(one_invoice("2026-03-01", "2026-03-31", "2026-04-25",
                                               ["2026-04-05"]))
    assert table["days_late"].tolist() == [5]
    assert table["money_arrived"].tolist() == [pd.Timestamp("2026-04-05")]


# --- the latest payment of the group wins --------------------------------------------------

def test_instalments_the_latest_payment_is_the_arrival():
    table = cleared_invoice_delays(one_invoice(
        "2026-03-01", "2026-03-31", "2026-04-20", ["2026-03-20", "2026-04-10", "2026-04-02"]))
    assert table["money_arrived"].tolist() == [pd.Timestamp("2026-04-10")]
    assert table["days_late"].tolist() == [10]


def test_instalments_give_one_row_not_one_per_payment():
    table = cleared_invoice_delays(one_invoice(
        "2026-03-01", "2026-03-31", "2026-04-20",
        ["2026-03-20", "2026-04-02", "2026-04-10", "2026-04-11"]))
    assert len(table) == 1


def test_the_order_of_the_payments_in_the_table_does_not_matter():
    forward = one_invoice("2026-03-01", "2026-03-31", "2026-04-20",
                          ["2026-03-20", "2026-04-10", "2026-04-02"])
    backward = forward.iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(cleared_invoice_delays(forward),
                                  cleared_invoice_delays(backward))


def test_unapplied_cash_posted_before_the_invoice_is_matched_through_the_group():
    # The money came on 15 Feb, before the invoice was posted (1 Mar), with no reference;
    # the clearing on 5 Mar matched it. Due 31 Mar: 44 days early.
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-03-05",
                clearing_document="CLR-9"),
        payment("PAY1", "2026-02-15", cleared="2026-03-05", clearing_document="CLR-9"),
    ]))
    row = table.iloc[0]
    assert row["money_arrived"] == pd.Timestamp("2026-02-15")
    assert row["days_late"] == -44
    assert row["known_on"] == pd.Timestamp("2026-03-05")


def test_unapplied_cash_can_be_the_latest_payment_of_the_group():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-20",
                clearing_document="CLR-9"),
        payment("PAY1", "2026-03-25", cleared="2026-04-20", clearing_document="CLR-9",
                reference="INV1"),
        payment("PAY2", "2026-04-08", cleared="2026-04-20", clearing_document="CLR-9"),
    ]))
    assert table["days_late"].tolist() == [8]


def test_a_payment_naming_the_invoice_in_another_group_is_not_used():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-03", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY2", "2026-04-12", cleared="2026-04-12", clearing_document="CLR-2",
                reference="INV1"),
    ]))
    assert table["days_late"].tolist() == [3]


def test_a_payment_without_a_clearing_document_is_not_used():
    # Even posted after the clearing date: it is in no group, so it is not "paid after".
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-03", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY2", "2026-04-28", reference="INV1"),
    ]))
    assert table["days_late"].tolist() == [3]


def test_one_clearing_document_clearing_two_invoices_gives_both_the_last_payment():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        invoice("INV2", "2026-03-01", "2026-04-10", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-01", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY2", "2026-04-12", cleared="2026-04-15", clearing_document="CLR-1"),
    ]))
    assert table["document_no"].tolist() == ["INV1", "INV2"]
    assert table["money_arrived"].tolist() == [pd.Timestamp("2026-04-12")] * 2
    assert table["days_late"].tolist() == [12, 2]


# --- invoices without a row ----------------------------------------------------------------

def test_an_invoice_cleared_by_a_credit_memo_alone_has_no_row():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-10",
                clearing_document="CLR-CM"),
        credit_memo("CM1", "2026-04-10", cleared="2026-04-10", clearing_document="CLR-CM"),
    ]))
    assert len(table) == 0
    assert list(table.columns) == CLEARED_DELAY_COLUMNS


def test_a_credit_memo_in_a_paid_group_is_not_a_payment():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-25",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-03", cleared="2026-04-25", clearing_document="CLR-1"),
        credit_memo("CM1", "2026-04-25", cleared="2026-04-25", clearing_document="CLR-1"),
    ]))
    assert table["money_arrived"].tolist() == [pd.Timestamp("2026-04-03")]


def test_a_credit_memo_cleared_invoice_drops_out_among_paid_ones():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-05",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-05", cleared="2026-04-05", clearing_document="CLR-1"),
        invoice("INV2", "2026-03-01", "2026-03-31", cleared="2026-04-10",
                clearing_document="CLR-CM"),
        credit_memo("CM1", "2026-04-10", cleared="2026-04-10", clearing_document="CLR-CM"),
        invoice("INV3", "2026-03-01", "2026-03-31", cleared="2026-04-08",
                clearing_document="CLR-3"),
        payment("PAY3", "2026-04-08", cleared="2026-04-08", clearing_document="CLR-3"),
    ]))
    assert table["document_no"].tolist() == ["INV1", "INV3"]


def test_an_open_invoice_has_no_row():
    table = cleared_invoice_delays(ledger([invoice("INV1", "2026-03-01", "2026-03-31")]))
    assert len(table) == 0


def test_an_open_invoice_with_a_partial_payment_has_no_row():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31"),
        payment("PAY1", "2026-03-20", reference="INV1"),
    ]))
    assert len(table) == 0


def test_only_the_cleared_invoices_of_a_mixed_ledger_have_rows():
    table = cleared_invoice_delays(ledger([
        invoice("OPEN1", "2026-03-01", "2026-03-31"),
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-05",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-05", cleared="2026-04-05", clearing_document="CLR-1"),
        invoice("OPEN2", "2026-03-02", "2026-04-01"),
    ]))
    assert table["document_no"].tolist() == ["INV1"]


def test_only_invoices_have_rows_not_payments_or_credit_memos():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-05",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-05", cleared="2026-04-05", clearing_document="CLR-1"),
        credit_memo("CM1", "2026-04-05", cleared="2026-04-05", clearing_document="CLR-1"),
    ]))
    assert table["document_no"].tolist() == ["INV1"]


# --- groups are keyed by company too -------------------------------------------------------

def test_the_same_clearing_document_in_two_companies_is_two_groups():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1", company="TR02"),
        payment("PAY1", "2026-04-02", cleared="2026-04-15", clearing_document="CLR-1",
                company="TR02"),
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-25",
                clearing_document="CLR-1", company="TR04"),
        payment("PAY1", "2026-04-20", cleared="2026-04-25", clearing_document="CLR-1",
                company="TR04"),
    ]))
    assert table["key"].tolist() == ["TR02-100", "TR04-100"]
    assert row_of(table, "INV1", "TR02-100")["days_late"] == 2
    assert row_of(table, "INV1", "TR04-100")["days_late"] == 20
    assert table["company_code"].tolist() == ["TR02", "TR04"]


def test_another_companys_later_payment_does_not_raise_paid_after_clearing():
    # TR04's payment on 25 Apr is after TR02's clearing on 15 Apr, but in another group.
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-02", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY9", "2026-04-25", cleared="2026-04-25", clearing_document="CLR-1",
                company="TR04"),
    ]))
    assert table["days_late"].tolist() == [2]


def test_an_invoice_whose_group_has_only_another_companys_payment_has_no_row():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY9", "2026-04-02", cleared="2026-04-15", clearing_document="CLR-1",
                company="TR04"),
    ]))
    assert len(table) == 0


def test_two_customers_of_one_company_sharing_a_clearing_document_share_the_group():
    # The group is (company, clearing_document), not the key.
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1", customer="100"),
        payment("PAY1", "2026-04-09", cleared="2026-04-15", clearing_document="CLR-1",
                customer="200"),
    ]))
    assert table["days_late"].tolist() == [9]


# --- known_on ------------------------------------------------------------------------------

THREE_DAYS = [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-10"),
              pd.Timestamp("2026-04-20")]


@pytest.mark.parametrize("posted, cleared, paid", list(itertools.permutations(THREE_DAYS)))
def test_known_on_is_the_latest_of_posting_clearing_and_arrival(posted, cleared, paid):
    built = one_invoice(str(posted.date()), "2026-04-30", str(cleared.date()),
                        [str(paid.date())])
    if paid > cleared:
        # A payment after the clearing is refused, so known_on is never the arrival alone.
        with pytest.raises(ValueError, match="posted after the clearing date"):
            cleared_invoice_delays(built)
        return
    table = cleared_invoice_delays(built)
    assert table["known_on"].tolist() == [max(posted, cleared, paid)]


def test_known_on_with_all_three_on_one_day():
    table = cleared_invoice_delays(one_invoice("2026-04-10", "2026-04-30", "2026-04-10",
                                               ["2026-04-10"]))
    assert table["known_on"].tolist() == [pd.Timestamp("2026-04-10")]


def test_known_on_is_the_arrival_when_it_ties_the_clearing_after_the_posting():
    table = cleared_invoice_delays(one_invoice("2026-04-01", "2026-04-30", "2026-04-15",
                                               ["2026-04-15"]))
    assert table["known_on"].tolist() == [pd.Timestamp("2026-04-15")]


def test_known_on_is_the_posting_date_for_a_late_posted_invoice():
    # Cleared and paid before it was posted (inconsistent but possible in an extract).
    table = cleared_invoice_delays(one_invoice("2026-05-02", "2026-05-31", "2026-04-15",
                                               ["2026-04-10"]))
    assert table["known_on"].tolist() == [pd.Timestamp("2026-05-02")]


def test_known_on_does_not_use_the_due_date():
    table = cleared_invoice_delays(one_invoice("2026-03-01", "2026-12-31", "2026-04-15",
                                               ["2026-04-10"]))
    assert table["known_on"].tolist() == [pd.Timestamp("2026-04-15")]


def test_known_on_is_never_before_the_arrival_or_the_clearing():
    table = cleared_invoice_delays(ledger([
        line for n, (posted, cleared, paid) in enumerate([
            ("2026-03-01", "2026-04-15", "2026-04-10"),
            ("2026-04-20", "2026-04-15", "2026-04-15"),
            ("2026-03-01", "2026-03-05", "2026-02-01")])
        for line in (invoice(f"INV{n}", posted, "2026-03-31", cleared=cleared,
                             clearing_document=f"CLR-{n}"),
                     payment(f"PAY{n}", paid, cleared=cleared, clearing_document=f"CLR-{n}"))
    ]))
    assert (table["known_on"] >= table["money_arrived"]).all()
    assert table["known_on"].tolist() == [pd.Timestamp("2026-04-15"),
                                          pd.Timestamp("2026-04-20"),
                                          pd.Timestamp("2026-03-05")]


# --- columns, dtypes, order ----------------------------------------------------------------

def test_columns_are_exactly_cleared_delay_columns():
    assert list(cleared_invoice_delays(running_example()).columns) == CLEARED_DELAY_COLUMNS


def test_dtypes_dates_are_datetimes_and_days_late_an_integer():
    table = cleared_invoice_delays(running_example())
    for column in ["net_due_date", "money_arrived", "known_on"]:
        assert pd.api.types.is_datetime64_any_dtype(table[column]), column
    assert pd.api.types.is_integer_dtype(table["days_late"])
    for column in ["key", "company_code", "document_no"]:
        assert not pd.api.types.is_numeric_dtype(table[column]), column


def test_extra_ledger_columns_do_not_leak_into_the_result():
    table = running_example()
    table["amount_document"] = 1000.0
    table["customer_id"] = "100"
    assert list(cleared_invoice_delays(table).columns) == CLEARED_DELAY_COLUMNS


def test_rows_follow_the_ledger_order_of_the_invoices():
    # Groups are found in a different order from the invoices; the invoices' order wins.
    table = cleared_invoice_delays(ledger([
        payment("PAY1", "2026-04-01", cleared="2026-04-05", clearing_document="CLR-1"),
        invoice("INV3", "2026-03-01", "2026-03-31", cleared="2026-04-07",
                clearing_document="CLR-3", customer="300"),
        payment("PAY2", "2026-04-02", cleared="2026-04-06", clearing_document="CLR-2"),
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-05",
                clearing_document="CLR-1", customer="100"),
        payment("PAY3", "2026-04-03", cleared="2026-04-07", clearing_document="CLR-3"),
        invoice("INV2", "2026-03-01", "2026-03-31", cleared="2026-04-06",
                clearing_document="CLR-2", customer="200"),
    ]))
    assert table["document_no"].tolist() == ["INV3", "INV1", "INV2"]
    assert table["days_late"].tolist() == [3, 1, 2]


def test_index_runs_from_0():
    table = cleared_invoice_delays(ledger([
        invoice("OPEN", "2026-03-01", "2026-03-31"),
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-05",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-05", cleared="2026-04-05", clearing_document="CLR-1"),
        invoice("INV2", "2026-03-01", "2026-03-31", cleared="2026-04-06",
                clearing_document="CLR-2"),
        payment("PAY2", "2026-04-06", cleared="2026-04-06", clearing_document="CLR-2"),
    ]))
    assert table.index.tolist() == [0, 1]


@pytest.mark.parametrize("lines", [
    [],
    [invoice("INV1", "2026-03-01", "2026-03-31")],
    [invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-10",
             clearing_document="CLR-CM"),
     credit_memo("CM1", "2026-04-10", cleared="2026-04-10", clearing_document="CLR-CM")],
    [payment("PAY1", "2026-04-01", cleared="2026-04-01", clearing_document="CLR-1")],
], ids=["empty ledger", "only open", "only credit-memo cleared", "only payments"])
def test_an_empty_result_keeps_the_columns_and_dtypes(lines):
    table = cleared_invoice_delays(ledger(lines))
    assert len(table) == 0
    assert list(table.columns) == CLEARED_DELAY_COLUMNS
    for column in ["net_due_date", "money_arrived", "known_on"]:
        assert pd.api.types.is_datetime64_any_dtype(table[column]), column
    assert pd.api.types.is_integer_dtype(table["days_late"])


def test_the_input_is_not_modified():
    original = running_example()
    copy = original.copy(deep=True)
    cleared_invoice_delays(original)
    pd.testing.assert_frame_equal(original, copy)


def test_calling_twice_gives_the_same_table():
    pd.testing.assert_frame_equal(cleared_invoice_delays(running_example()),
                                  cleared_invoice_delays(running_example()))


# --- the errors: an invoice twice ----------------------------------------------------------

def test_an_invoice_in_the_ledger_twice_raises():
    with pytest.raises(ValueError, match="Duplicate entries in the invoices table!"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31"),
            invoice("INV1", "2026-03-01", "2026-03-31"),
        ]))


def test_an_invoice_twice_raises_even_if_both_are_open():
    with pytest.raises(ValueError, match="Duplicate"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31"),
            invoice("INV1", "2026-06-01", "2026-06-30"),
        ]))


def test_the_same_document_no_under_two_keys_is_not_a_duplicate():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-05",
                clearing_document="CLR-1", customer="100"),
        payment("PAY1", "2026-04-05", cleared="2026-04-05", clearing_document="CLR-1",
                customer="100"),
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-06",
                clearing_document="CLR-2", customer="200"),
        payment("PAY2", "2026-04-06", cleared="2026-04-06", clearing_document="CLR-2",
                customer="200"),
    ]))
    assert table["key"].tolist() == ["TR02-100", "TR02-200"]


def test_a_payment_and_an_invoice_sharing_a_document_no_is_not_a_duplicate():
    table = cleared_invoice_delays(ledger([
        invoice("DOC1", "2026-03-01", "2026-03-31", cleared="2026-04-05",
                clearing_document="CLR-1"),
        payment("DOC1", "2026-04-05", cleared="2026-04-05", clearing_document="CLR-1"),
    ]))
    assert len(table) == 1


# --- the errors: a cleared invoice without a clearing document ------------------------------

def test_a_cleared_invoice_without_a_clearing_document_raises():
    with pytest.raises(ValueError, match="cleared invoices have no clearing document"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15"),
        ]))


def test_the_missing_clearing_document_message_counts_all_and_lists_the_first_five_rows():
    lines = [invoice(f"INV{n}", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                     source_row=100 + n) for n in range(7)]
    lines.append(invoice("INVOK", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                         clearing_document="CLR-1", source_row=50))
    with pytest.raises(ValueError) as error:
        cleared_invoice_delays(ledger(lines))
    assert str(error.value) == ("7 cleared invoices have no clearing document, so their "
                                "payments cannot be found (first rows: [100, 101, 102, 103, "
                                "104])")


def test_an_open_invoice_without_a_clearing_document_does_not_raise():
    assert len(cleared_invoice_delays(ledger([invoice("INV1", "2026-03-01",
                                                      "2026-03-31")]))) == 0


def test_a_cleared_payment_without_a_clearing_document_does_not_raise():
    assert len(cleared_invoice_delays(ledger([payment("PAY1", "2026-04-01",
                                                      cleared="2026-04-01")]))) == 0


def test_the_duplicate_check_comes_before_the_clearing_document_check():
    with pytest.raises(ValueError, match="Duplicate"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15"),
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15"),
        ]))


# --- the errors: a payment after the clearing ----------------------------------------------

def test_a_payment_posted_after_the_clearing_date_raises():
    with pytest.raises(ValueError, match="posted after the clearing date"):
        cleared_invoice_delays(one_invoice("2026-03-01", "2026-03-31", "2026-04-15",
                                           ["2026-04-10", "2026-04-16"]))


def test_a_payment_posted_on_the_clearing_date_is_fine():
    table = cleared_invoice_delays(one_invoice("2026-03-01", "2026-03-31", "2026-04-15",
                                               ["2026-04-10", "2026-04-15"]))
    assert table["money_arrived"].tolist() == [pd.Timestamp("2026-04-15")]


def test_a_payment_one_day_after_the_clearing_date_raises():
    with pytest.raises(ValueError, match="posted after the clearing date"):
        cleared_invoice_delays(one_invoice("2026-03-01", "2026-03-31", "2026-04-15",
                                           ["2026-04-16"]))


def test_the_paid_after_clearing_message_counts_the_invoices_and_names_their_rows():
    lines = []
    for n in range(7):
        lines += [invoice(f"INV{n}", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                          clearing_document=f"CLR-{n}", source_row=200 + n),
                  payment(f"PAY{n}", "2026-04-20", cleared="2026-04-15",
                          clearing_document=f"CLR-{n}", source_row=300 + n)]
    lines += [invoice("INVOK", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                      clearing_document="CLR-OK", source_row=60),
              payment("PAYOK", "2026-04-15", cleared="2026-04-15",
                      clearing_document="CLR-OK", source_row=61)]
    with pytest.raises(ValueError) as error:
        cleared_invoice_delays(ledger(lines))
    assert str(error.value) == ("7 cleared invoices have a payment in their clearing group "
                                "posted after the clearing date (first rows: [200, 201, 202, "
                                "203, 204])")


def test_one_late_payment_in_a_shared_group_flags_each_invoice_it_clears():
    with pytest.raises(ValueError, match=r"^2 cleared invoices have a payment"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-1"),
            invoice("INV2", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-1"),
            payment("PAY1", "2026-04-20", cleared="2026-04-15", clearing_document="CLR-1"),
        ]))


def test_a_late_payment_in_the_group_of_an_open_invoice_does_not_raise():
    # Only cleared invoices are matched to groups.
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31"),
        payment("PAY1", "2026-04-20", cleared="2026-04-15", clearing_document="CLR-1"),
    ]))
    assert len(table) == 0


def test_a_late_payment_in_a_group_without_a_cleared_invoice_does_not_raise():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-10", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY9", "2026-05-20", cleared="2026-05-01", clearing_document="CLR-9"),
    ]))
    assert table["days_late"].tolist() == [10]


def test_the_clearing_document_check_comes_before_the_paid_after_clearing_check():
    with pytest.raises(ValueError, match="no clearing document"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15"),
            invoice("INV2", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-2"),
            payment("PAY2", "2026-04-20", cleared="2026-04-15", clearing_document="CLR-2"),
        ]))


# --- the errors: a clearing group with more than one clearing date -------------------------

SEVERAL_DATES = "ledger lines are in a clearing group with more than one clearing date"


def test_a_payment_line_cleared_later_than_its_invoice_raises():
    # The case that broke the leak property: the payment's own clearing date (20 Apr) is
    # later than the invoice's (15 Apr), so a cutoff between them would cut the group.
    with pytest.raises(ValueError) as error:
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-1"),
            payment("PAY1", "2026-04-10", cleared="2026-04-20", clearing_document="CLR-1"),
        ]))
    assert str(error.value) == (f"2 {SEVERAL_DATES} (first rows: [2, 3])")


def test_a_group_with_two_clearing_dates_raises_whatever_the_order_of_its_lines():
    with pytest.raises(ValueError, match=f"^2 {SEVERAL_DATES}"):
        cleared_invoice_delays(ledger([
            payment("PAY1", "2026-04-10", cleared="2026-04-12", clearing_document="CLR-1"),
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-1"),
        ]))


def test_a_group_line_with_no_clearing_date_beside_a_dated_one_raises():
    # NaT counts as a value: a payment in CLR-1 without a clearing date.
    with pytest.raises(ValueError, match=f"^2 {SEVERAL_DATES}"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-1"),
            payment("PAY1", "2026-04-10", clearing_document="CLR-1"),
        ]))


def test_an_open_invoice_carrying_a_dated_groups_clearing_document_raises():
    # The invoice has the clearing document but no clearing date; its group is dated.
    with pytest.raises(ValueError, match=f"^2 {SEVERAL_DATES}"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", clearing_document="CLR-1"),
            payment("PAY1", "2026-04-10", cleared="2026-04-10", clearing_document="CLR-1"),
        ]))


def test_a_credit_memo_line_with_another_clearing_date_raises():
    # Every document role counts, not only invoices and payments.
    with pytest.raises(ValueError, match=f"^3 {SEVERAL_DATES}"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-1"),
            payment("PAY1", "2026-04-10", cleared="2026-04-15", clearing_document="CLR-1"),
            credit_memo("CM1", "2026-04-16", cleared="2026-04-16", clearing_document="CLR-1"),
        ]))


def test_a_group_whose_lines_all_lack_a_clearing_date_does_not_raise():
    # One value (NaT) only; the invoice is not cleared, so there is no row.
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", clearing_document="CLR-1"),
        payment("PAY1", "2026-04-10", clearing_document="CLR-1"),
    ]))
    assert len(table) == 0


def test_the_same_clearing_document_in_another_company_with_another_date_is_fine():
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1", company="TR02"),
        payment("PAY1", "2026-04-10", cleared="2026-04-15", clearing_document="CLR-1",
                company="TR02"),
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-25",
                clearing_document="CLR-1", company="TR04"),
        payment("PAY1", "2026-04-20", cleared="2026-04-25", clearing_document="CLR-1",
                company="TR04"),
    ]))
    assert table["days_late"].tolist() == [10, 20]


def test_lines_without_a_clearing_document_are_not_in_any_group():
    # Different clearing dates on lines with no clearing document are not a group.
    table = cleared_invoice_delays(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-10", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY2", "2026-04-11", cleared="2026-04-18"),
        payment("PAY3", "2026-04-12"),
    ]))
    assert table["days_late"].tolist() == [10]


def test_the_several_dates_message_counts_every_line_of_the_bad_groups_and_lists_five():
    lines = []
    for n in range(3):   # three bad groups of two lines each: 6 lines
        lines += [invoice(f"INV{n}", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                          clearing_document=f"CLR-{n}", source_row=100 + 2 * n),
                  payment(f"PAY{n}", "2026-04-10", cleared="2026-04-16",
                          clearing_document=f"CLR-{n}", source_row=101 + 2 * n)]
    lines += [invoice("INVOK", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                      clearing_document="CLR-OK", source_row=50),
              payment("PAYOK", "2026-04-10", cleared="2026-04-15",
                      clearing_document="CLR-OK", source_row=51)]
    with pytest.raises(ValueError) as error:
        cleared_invoice_delays(ledger(lines))
    assert str(error.value) == (f"6 {SEVERAL_DATES} (first rows: [100, 101, 102, 103, 104])")


def test_the_several_dates_message_lists_rows_in_ledger_order_not_group_order():
    with pytest.raises(ValueError) as error:
        cleared_invoice_delays(ledger([
            payment("PAYB", "2026-04-10", cleared="2026-04-16", clearing_document="CLR-B",
                    source_row=30),
            invoice("INVA", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-A", source_row=31),
            invoice("INVB", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-B", source_row=32),
            payment("PAYA", "2026-04-10", cleared="2026-04-17", clearing_document="CLR-A",
                    source_row=33),
        ]))
    assert str(error.value) == (f"4 {SEVERAL_DATES} (first rows: [30, 31, 32, 33])")


def test_the_clearing_document_check_comes_before_the_several_dates_check():
    with pytest.raises(ValueError, match="no clearing document"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15"),
            invoice("INV2", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-2"),
            payment("PAY2", "2026-04-10", cleared="2026-04-20", clearing_document="CLR-2"),
        ]))


def test_the_duplicate_check_comes_before_the_several_dates_check():
    with pytest.raises(ValueError, match="Duplicate"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31"),
            invoice("INV1", "2026-03-01", "2026-03-31"),
            invoice("INV2", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-2"),
            payment("PAY2", "2026-04-10", cleared="2026-04-20", clearing_document="CLR-2"),
        ]))


def test_the_several_dates_check_comes_before_the_paid_after_clearing_check():
    # CLR-1 has a payment after its clearing; CLR-2 has two clearing dates.
    with pytest.raises(ValueError, match=f"^2 {SEVERAL_DATES}"):
        cleared_invoice_delays(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-1"),
            payment("PAY1", "2026-04-20", cleared="2026-04-15", clearing_document="CLR-1"),
            invoice("INV2", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                    clearing_document="CLR-2"),
            payment("PAY2", "2026-04-10", cleared="2026-04-20", clearing_document="CLR-2"),
        ]))


def test_payment_delays_at_raises_on_two_clearing_dates_after_the_cutoff():
    with pytest.raises(ValueError, match=SEVERAL_DATES):
        payment_delays_at(ledger([
            invoice("INV1", "2026-06-01", "2026-06-30", cleared="2026-07-15",
                    clearing_document="CLR-1"),
            payment("PAY1", "2026-07-10", cleared="2026-07-20", clearing_document="CLR-1"),
        ]), CUTOFF, 1)


# --- payment_delays_at now checks the whole ledger -----------------------------------------

def test_payment_delays_at_raises_on_a_duplicate_posted_after_the_cutoff():
    with pytest.raises(ValueError, match="Duplicate"):
        payment_delays_at(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31"),
            invoice("INV1", "2026-06-01", "2026-06-30"),
        ]), CUTOFF, 1)


def test_payment_delays_at_raises_on_a_missing_clearing_document_after_the_cutoff():
    with pytest.raises(ValueError, match="cleared invoices have no clearing document"):
        payment_delays_at(ledger([
            invoice("INV1", "2026-06-01", "2026-06-30", cleared="2026-07-15"),
        ]), CUTOFF, 1)


def test_payment_delays_at_raises_on_a_payment_after_clearing_after_the_cutoff():
    with pytest.raises(ValueError, match="posted after the clearing date"):
        payment_delays_at(one_invoice("2026-06-01", "2026-06-30", "2026-07-15",
                                      ["2026-07-20"]), CUTOFF, 1)


def test_payment_delays_at_counts_an_invoice_only_once_its_last_payment_is_known():
    # Cleared 30 Apr by payments of 10 and 30 Apr: counted at 30 Apr, not at 29 Apr.
    built = one_invoice("2026-03-01", "2026-03-31", "2026-04-30", ["2026-04-10", "2026-04-30"])
    assert len(payment_delays_at(built, date(2026, 4, 29), 1)) == 0
    table = payment_delays_at(built, date(2026, 4, 30), 1)
    assert table["usual_delay_days"].tolist() == [30.0]


def test_payment_delays_at_aggregates_the_rows_known_by_the_cutoff():
    lines = []
    for n, (cleared, paid) in enumerate([("2026-04-05", "2026-04-02"),
                                         ("2026-04-20", "2026-04-08"),
                                         ("2026-05-10", "2026-05-01")]):
        lines += [invoice(f"INV{n}", "2026-03-01", "2026-03-31", cleared=cleared,
                          clearing_document=f"CLR-{n}"),
                  payment(f"PAY{n}", paid, cleared=cleared, clearing_document=f"CLR-{n}")]
    built = ledger(lines)
    delays = cleared_invoice_delays(built)
    for cutoff in [date(2026, 4, 4), date(2026, 4, 5), date(2026, 4, 30), date(2026, 5, 31)]:
        known = delays[delays["known_on"] <= pd.Timestamp(cutoff)]
        table = payment_delays_at(built, cutoff, 1)
        if len(known) == 0:
            assert len(table) == 0
            continue
        assert table["cleared_invoices"].tolist() == [len(known)]
        assert table["usual_delay_days"].tolist() == [float(known["days_late"].median())]


# --- equivalence with the old definition, and no leak -------------------------------------

def old_payment_delays_at(table, cutoff, min_cleared):
    """The old definition, re-computed independently with plain Python: invoices posted and
    cleared by `cutoff`; the arrival = the latest payment of the group posted by `cutoff`."""
    end = pd.Timestamp(cutoff)
    rows = table.to_dict("records")
    latest = {}
    for r in rows:
        if (r["document_role"] == "payment" and r["clearing_document"] is not None
                and not pd.isna(r["clearing_document"]) and r["posting_date"] <= end):
            group = (r["company_code"], r["clearing_document"])
            latest[group] = max(latest.get(group, r["posting_date"]), r["posting_date"])
    late = {}
    for r in rows:
        if (r["document_role"] != "invoice" or r["posting_date"] > end
                or pd.isna(r["clearing_date"]) or r["clearing_date"] > end):
            continue
        group = (r["company_code"], r["clearing_document"])
        if group in latest:
            late.setdefault(r["key"], []).append((latest[group] - r["net_due_date"]).days)
    result = {}
    for key, values in late.items():
        delay = float(np.median(values)) if len(values) >= min_cleared else math.nan
        result[key] = (delay, len(values))
    return result


def as_dict(table):
    return {row.key: (row.usual_delay_days, row.cleared_invoices)
            for row in table.itertuples()}


def same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        (da, na), (db, nb) = a[key], b[key]
        assert na == nb, key
        assert (math.isnan(da) and math.isnan(db)) or da == db, (key, da, db)


def random_ledger(seed):
    """A consistent random ledger: no payment of a group is posted after its clearing date.

    Two companies sharing clearing-document names, several customers, open invoices (some
    partly paid), invoices cleared by one or more payments (some posted before the invoice,
    as unapplied cash), groups clearing two invoices, credit-memo-only clearings, and
    payments with no clearing document.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-10-01")

    def day(offset):
        return str((start + pd.Timedelta(days=int(offset))).date())

    lines = []
    for company in ["TR02", "TR04"]:
        for customer in ["100", "200", "300", "400"]:
            for n in range(int(rng.integers(3, 9))):
                posted = int(rng.integers(0, 200))
                due = posted + int(rng.integers(0, 60))
                doc = f"INV-{customer}-{n}"
                kind = rng.choice(["open", "paid", "paid", "paid", "memo", "pair"])
                kw = dict(company=company, customer=customer)
                if kind == "open":
                    lines.append(invoice(doc, day(posted), day(due), **kw))
                    if rng.random() < 0.5:
                        lines.append(payment(f"P-{doc}", day(posted + 5), reference=doc, **kw))
                    continue
                group = f"CLR-{customer}-{n}"  # the same names in both companies
                cleared = posted + int(rng.integers(-5, 90))
                if kind == "memo":
                    lines += [invoice(doc, day(posted), day(due), cleared=day(cleared),
                                      clearing_document=group, **kw),
                              credit_memo(f"CM-{doc}", day(cleared), cleared=day(cleared),
                                          clearing_document=group, **kw)]
                    continue
                docs = [doc] if kind == "paid" else [doc, doc + "b"]
                for d in docs:
                    lines.append(invoice(d, day(posted), day(due + (d != doc) * 7),
                                         cleared=day(cleared), clearing_document=group, **kw))
                for k in range(int(rng.integers(1, 4))):
                    paid = cleared - int(rng.integers(0, 80))   # on or before the clearing
                    lines.append(payment(f"P{k}-{doc}", day(paid), cleared=day(cleared),
                                         clearing_document=group,
                                         reference=doc if rng.random() < 0.6 else None, **kw))
            lines.append(payment(f"LOOSE-{customer}", day(int(rng.integers(0, 250))),
                                 company=company, customer=customer))
    order = rng.permutation(len(lines))
    return ledger([lines[i] for i in order])


SEEDS = [1, 2, 3, 4, 5, 6]
CUTOFFS = [date(2025, 10, 1), date(2025, 11, 15), date(2026, 1, 4), date(2026, 3, 1),
           date(2026, 5, 20), date(2027, 1, 1)]


def test_random_ledgers_are_consistent_and_rich():
    # Premise: the generator makes rows, some cleared invoices are open at early cutoffs,
    # and unapplied cash (paid before the invoice was posted) occurs.
    for seed in SEEDS:
        table = random_ledger(seed)
        delays = cleared_invoice_delays(table)
        assert len(delays) > 10
        assert (delays["known_on"] > pd.Timestamp(CUTOFFS[2])).any()
        assert (delays["known_on"] <= pd.Timestamp(CUTOFFS[3])).any()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("min_cleared", [1, 3])
def test_payment_delays_at_equals_the_old_definition(seed, min_cleared):
    table = random_ledger(seed)
    for cutoff in CUTOFFS:
        same(as_dict(payment_delays_at(table, cutoff, min_cleared)),
             old_payment_delays_at(table, cutoff, min_cleared))


def test_the_old_definition_has_teeth():
    # Premise: the old re-computation sees different things at different cutoffs.
    table = random_ledger(1)
    counts = [sum(n for _, n in old_payment_delays_at(table, c, 1).values()) for c in CUTOFFS]
    assert counts == sorted(counts)
    assert len(set(counts)) > 2


@pytest.mark.parametrize("seed", SEEDS)
def test_rows_known_at_a_cutoff_equal_the_rows_of_the_ledger_as_known_then(seed):
    table = random_ledger(seed)
    full = cleared_invoice_delays(table)
    for cutoff in CUTOFFS:
        known = full[full["known_on"] <= pd.Timestamp(cutoff)].reset_index(drop=True)
        cut = cleared_invoice_delays(ledger_as_known_at(table, cutoff)).reset_index(drop=True)
        pd.testing.assert_frame_equal(known, cut, check_dtype=False)


def test_the_leak_property_on_the_running_example_at_every_day():
    table = running_example()
    full = cleared_invoice_delays(table)
    for cutoff in pd.date_range("2026-02-25", "2026-04-20"):
        known = full[full["known_on"] <= cutoff].reset_index(drop=True)
        cut = cleared_invoice_delays(ledger_as_known_at(table, cutoff.date()))
        pd.testing.assert_frame_equal(known, cut.reset_index(drop=True), check_dtype=False)


# --- integration with the synthetic ledger (SLOW) ----------------------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
AS_OF = date(2026, 8, 28)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]


@pytest.fixture(scope="module")
def synthetic_ledger():
    if not LEDGER_FILE.exists():
        pytest.skip("the synthetic ledger is not in data/raw")
    ledger_lines = read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT)  # direct, no cache
    return add_keys(ledger_lines, KNOWN_COMPANIES)


@pytest.fixture(scope="module")
def synthetic_delays(synthetic_ledger):
    return cleared_invoice_delays(synthetic_ledger)


def test_slow_synthetic_delays_shape(synthetic_delays):
    assert len(synthetic_delays) > 0
    assert list(synthetic_delays.columns) == CLEARED_DELAY_COLUMNS
    assert not synthetic_delays.duplicated(["key", "document_no"]).any()
    assert (synthetic_delays["known_on"] >= synthetic_delays["money_arrived"]).all()
    days = (synthetic_delays["money_arrived"] - synthetic_delays["net_due_date"]).dt.days
    assert synthetic_delays["days_late"].tolist() == days.tolist()


@pytest.mark.parametrize("cutoff", [date(2025, 6, 8), date(2026, 1, 4)])
def test_slow_synthetic_rows_known_at_a_cutoff_equal_the_ledger_as_known(synthetic_ledger,
                                                                         synthetic_delays,
                                                                         cutoff):
    known = synthetic_delays[synthetic_delays["known_on"] <= pd.Timestamp(cutoff)]
    cut = cleared_invoice_delays(ledger_as_known_at(synthetic_ledger, cutoff))
    assert len(known) > 0
    pd.testing.assert_frame_equal(known.reset_index(drop=True), cut.reset_index(drop=True),
                                  check_dtype=False)


@pytest.mark.parametrize("cutoff", [date(2025, 6, 8), date(2026, 1, 4)])
def test_slow_synthetic_payment_delays_at_equals_the_old_definition(synthetic_ledger, cutoff):
    same(as_dict(payment_delays_at(synthetic_ledger, cutoff, 3)),
         old_payment_delays_at(synthetic_ledger, cutoff, 3))
