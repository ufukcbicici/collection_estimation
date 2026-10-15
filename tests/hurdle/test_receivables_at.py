"""Tests for U9b.4 in `hurdle/receivables_asof.py`: `receivables_at` and its helpers.

`receivables_at(ledger, week, week_zero, rates, horizons, min_cleared)` gives one row per key
with each customer's receivables at the end of week t (its Sunday): the open invoices'
outstanding split by when it falls due (due_in_week_1 .. due_in_week_<horizons>, due_later,
overdue), the amount-weighted days past due, the open invoice count, unapplied cash, the
open balance of all open lines, and the usual delay from `payment_delays_at`.

Week zero is Monday 2026-01-05, so week t runs Monday week_zero + 7t to Sunday
week_zero + 7t + 6. Ledgers and rate tables are built by hand. The SLOW tests at the end read
the synthetic corpus and ledger in `data/raw/` once and are skipped when absent; the repo's
`data/interim/` cache is never used.
"""
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle import receivables_asof
from collection_estimation.hurdle.keys import add_keys
from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
from collection_estimation.hurdle.read_ledger import read_ledger
from collection_estimation.hurdle.receivables_asof import (
    TIE_OUT_TOLERANCE,
    due_buckets,
    due_in_week_columns,
    exchange_rates,
    open_invoices_at,
    open_lines_at,
    rates_known_at,
    receivables_at,
    receivables_columns,
    to_try,
)
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

WEEK_ZERO = date(2026, 1, 5)  # a Monday

LEDGER_COLUMNS = ["key", "company_code", "customer_id", "document_no", "document_role",
                  "invoice_reference", "posting_date", "net_due_date", "clearing_date",
                  "clearing_document", "amount_document", "document_currency", "source_row"]


def monday(week):
    return WEEK_ZERO + timedelta(days=7 * week)


def sunday(week):
    """The cutoff of week `week`: the Sunday that ends it."""
    return WEEK_ZERO + timedelta(days=7 * week + 6)


def ts(day):
    return pd.Timestamp(day) if day is not None else pd.NaT


def line(role, document_no, amount, posted, due=None, reference=None, cleared=None,
         clearing_document=None, customer="A", company="TR02", currency="TRY"):
    return {"key": f"{company}-{customer}", "company_code": company, "customer_id": customer,
            "document_no": document_no, "document_role": role, "invoice_reference": reference,
            "posting_date": ts(posted), "net_due_date": ts(due or posted),
            "clearing_date": ts(cleared), "clearing_document": clearing_document,
            "amount_document": float(amount), "document_currency": currency}


def invoice(document_no, amount, posted, due=None, **kwargs):
    return line("invoice", document_no, amount, posted, due=due, **kwargs)


def payment(document_no, amount, posted, reference=None, **kwargs):
    """`amount` as in the ledger: negative for money received, positive for a refund."""
    return line("payment", document_no, amount, posted, reference=reference, **kwargs)


def credit_memo(document_no, amount, posted, **kwargs):
    return line("credit_memo", document_no, amount, posted, **kwargs)


def ledger(lines):
    """A ledger table with ledger-like dtypes; source_row counts from 2."""
    if not lines:
        table = pd.DataFrame({column: pd.Series(dtype=object) for column in LEDGER_COLUMNS})
    else:
        table = pd.DataFrame(lines, columns=[c for c in LEDGER_COLUMNS if c != "source_row"])
        table["source_row"] = range(2, 2 + len(table))
    for column in ["posting_date", "net_due_date", "clearing_date"]:
        table[column] = pd.to_datetime(table[column])
    table["amount_document"] = table["amount_document"].astype("float64")
    table["invoice_reference"] = table["invoice_reference"].astype(object)
    table["clearing_document"] = table["clearing_document"].astype(object)
    table["source_row"] = table["source_row"].astype("int64")
    return table


def rates_table(rows):
    """rows: (day, currency, rate) triples."""
    return pd.DataFrame({
        "day": pd.Series([r[0] for r in rows], dtype=object),
        "currency": pd.Series([r[1] for r in rows], dtype=object),
        "rate": pd.Series([r[2] for r in rows], dtype="float64"),
    })


NO_RATES = rates_table([])
EUR_RATES = rates_table([
    (date(2026, 1, 2), "EUR", 35.0),
    (date(2026, 1, 23), "EUR", 36.0),   # the rate known at the end of week 2
    (date(2026, 2, 6), "EUR", 37.0),    # known from the end of week 4
])


# The running example, at the end of week 2 (Sunday 2026-01-25), customer A:
# INV-1 1,000 TRY due Wed 2026-02-04 (week 4 = t+2) with 300 paid -> due_in_week_2 700;
# INV-2 100 EUR due 2026-01-15 (10 days before the cutoff) -> overdue 100 * 36;
# 200 TRY of unapplied cash -> unapplied_cash 200; open balance 700 + 3,600 - 200 = 4,100.
WEEK = 2
CUTOFF = sunday(WEEK)


def running_example_lines():
    return [
        invoice("INV-1", 1000, "2026-01-10", due="2026-02-04"),
        payment("PAY-1", -300, "2026-01-20", reference="INV-1"),
        invoice("INV-2", 100, "2025-12-01", due="2026-01-15", currency="EUR"),
        payment("PAY-2", -200, "2026-01-22"),
    ]


def running_example():
    return ledger(running_example_lines())


def run(table, week=WEEK, rates=EUR_RATES, horizons=5, min_cleared=3):
    return receivables_at(table, week, WEEK_ZERO, rates, horizons, min_cleared)


def row_of(table, key="TR02-A"):
    rows = table[table["key"] == key]
    assert len(rows) == 1, table
    return rows.iloc[0]


def open_invoices_with_try(table, cutoff, rates=EUR_RATES):
    """`open_invoices_at` with the `outstanding_try` column `due_buckets` needs."""
    invoices = open_invoices_at(table, cutoff)
    invoices["outstanding_try"] = to_try(invoices["outstanding"], invoices["currency"],
                                         rates_known_at(rates, cutoff), cutoff)
    return invoices


def buckets(lines, week=WEEK, horizons=5, rates=EUR_RATES):
    cutoff = sunday(week)
    invoices = open_invoices_with_try(ledger(lines), cutoff, rates)
    return due_buckets(invoices, week, WEEK_ZERO, cutoff, horizons)


# --- the test calendar itself --------------------------------------------------------

def test_week_zero_is_a_monday_and_week_two_ends_on_sunday_25_january():
    assert WEEK_ZERO.weekday() == 0
    assert CUTOFF == date(2026, 1, 25)
    assert CUTOFF.weekday() == 6


# --- column helpers ------------------------------------------------------------------

def test_tie_out_tolerance_is_one_kurus():
    assert TIE_OUT_TOLERANCE == 0.01


def test_due_in_week_columns_for_five():
    assert due_in_week_columns(5) == ["due_in_week_1", "due_in_week_2", "due_in_week_3",
                                      "due_in_week_4", "due_in_week_5"]


def test_due_in_week_columns_for_one():
    assert due_in_week_columns(1) == ["due_in_week_1"]


def test_receivables_columns_for_three():
    assert receivables_columns(3) == [
        "key", "week", "due_in_week_1", "due_in_week_2", "due_in_week_3", "due_later",
        "overdue", "overdue_days", "open_invoices", "unapplied_cash", "open_balance",
        "usual_delay_days", "cleared_invoices"]


def test_receivables_columns_length_grows_with_horizons():
    assert len(receivables_columns(5)) == len(receivables_columns(1)) + 4


# --- to_try --------------------------------------------------------------------------

def test_to_try_multiplies_by_each_currency_rate():
    amounts = pd.Series([10.0, 2.0, 5.0])
    currencies = pd.Series(["EUR", "USD", "EUR"])
    result = to_try(amounts, currencies, {"EUR": 36.0, "USD": 32.0, "TRY": 1.0}, CUTOFF)
    assert result.tolist() == [360.0, 64.0, 180.0]


def test_to_try_keeps_try_at_rate_one():
    result = to_try(pd.Series([123.45]), pd.Series(["TRY"]), {"TRY": 1.0}, CUTOFF)
    assert result.tolist() == [123.45]


def test_to_try_normalises_currency_codes():
    result = to_try(pd.Series([1.0, 1.0, 1.0]), pd.Series([" eur ", "Eur", "try"]),
                    {"EUR": 36.0, "TRY": 1.0}, CUTOFF)
    assert result.tolist() == [36.0, 36.0, 1.0]


def test_to_try_keeps_negative_amounts_negative():
    result = to_try(pd.Series([-50.0]), pd.Series(["EUR"]), {"EUR": 36.0, "TRY": 1.0}, CUTOFF)
    assert result.tolist() == [-1800.0]


def test_to_try_keeps_the_index():
    amounts = pd.Series([1.0, 2.0], index=[7, 3])
    currencies = pd.Series(["EUR", "TRY"], index=[7, 3])
    result = to_try(amounts, currencies, {"EUR": 36.0, "TRY": 1.0}, CUTOFF)
    assert result.to_dict() == {7: 36.0, 3: 2.0}


def test_to_try_with_a_missing_rate_raises_naming_the_currency_and_the_cutoff():
    with pytest.raises(ValueError) as error:
        to_try(pd.Series([1.0, 1.0]), pd.Series(["EUR", "GBP"]), {"EUR": 36.0, "TRY": 1.0},
               CUTOFF)
    message = str(error.value)
    assert "GBP" in message
    assert str(CUTOFF) in message
    assert "EUR" not in message


def test_to_try_names_every_missing_currency():
    with pytest.raises(ValueError) as error:
        to_try(pd.Series([1.0, 1.0]), pd.Series(["USD", "GBP"]), {"TRY": 1.0}, CUTOFF)
    assert "GBP" in str(error.value) and "USD" in str(error.value)


def test_to_try_on_empty_series_is_empty():
    result = to_try(pd.Series([], dtype="float64"), pd.Series([], dtype=object), {"TRY": 1.0},
                    CUTOFF)
    assert len(result) == 0


def test_to_try_blank_currency_with_another_missing_rate_raises_value_error():
    """The blank currency is reported first, before the missing-rate check."""
    with pytest.raises(ValueError) as error:
        to_try(pd.Series([1.0, 1.0]), pd.Series([None, "GBP"]), {"TRY": 1.0}, CUTOFF)
    assert str(error.value) == f"1 amounts to convert at {CUTOFF} have no currency"


def test_to_try_blank_currency_alone_raises_value_error():
    with pytest.raises(ValueError) as error:
        to_try(pd.Series([1.0]), pd.Series([None]), {"TRY": 1.0}, CUTOFF)
    assert str(error.value) == f"1 amounts to convert at {CUTOFF} have no currency"


@pytest.mark.parametrize("blank", [None, "", "   ", float("nan")])
def test_to_try_counts_every_blank_currency(blank):
    with pytest.raises(ValueError) as error:
        to_try(pd.Series([1.0, 2.0, 3.0]), pd.Series([blank, "TRY", blank]), {"TRY": 1.0},
               CUTOFF)
    assert str(error.value) == f"2 amounts to convert at {CUTOFF} have no currency"


# --- due_buckets ---------------------------------------------------------------------

def bucket_row(lines, key="TR02-A", **kwargs):
    return row_of(buckets(lines, **kwargs), key)


@pytest.mark.parametrize("h", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("weekday", range(7))
def test_invoice_due_any_day_of_week_t_plus_h_goes_to_due_in_week_h(h, weekday):
    due = monday(WEEK + h) + timedelta(days=weekday)
    row = bucket_row([invoice("I", 100, "2026-01-06", due=due)])
    for other in range(1, 6):
        assert row[f"due_in_week_{other}"] == (100.0 if other == h else 0.0)
    assert row["due_later"] == 0.0
    assert row["overdue"] == 0.0


def test_due_on_the_cutoff_sunday_is_overdue_with_zero_days():
    row = bucket_row([invoice("I", 100, "2026-01-06", due=CUTOFF)])
    assert row["overdue"] == 100.0
    assert row["due_in_week_1"] == 0.0
    assert row["overdue_days"] == 0.0


def test_due_on_the_monday_after_the_cutoff_is_due_in_week_1():
    row = bucket_row([invoice("I", 100, "2026-01-06", due=CUTOFF + timedelta(days=1))])
    assert row["due_in_week_1"] == 100.0
    assert row["overdue"] == 0.0


def test_due_on_the_last_sunday_of_the_horizon_is_in_the_last_column():
    row = bucket_row([invoice("I", 100, "2026-01-06", due=sunday(WEEK + 5))])
    assert row["due_in_week_5"] == 100.0
    assert row["due_later"] == 0.0


def test_due_on_the_monday_after_the_horizon_is_due_later():
    row = bucket_row([invoice("I", 100, "2026-01-06", due=monday(WEEK + 6))])
    assert row["due_later"] == 100.0
    assert row["due_in_week_5"] == 0.0


def test_due_far_in_the_future_is_due_later():
    row = bucket_row([invoice("I", 100, "2026-01-06", due="2027-06-30")])
    assert row["due_later"] == 100.0


def test_due_earlier_in_week_t_is_overdue():
    row = bucket_row([invoice("I", 100, "2026-01-06", due=monday(WEEK))])
    assert row["overdue"] == 100.0
    assert row["overdue_days"] == 6.0


def test_due_long_ago_is_overdue_with_the_days_since():
    row = bucket_row([invoice("I", 100, "2025-10-01", due="2025-11-25")])
    assert row["overdue"] == 100.0
    assert row["overdue_days"] == float((CUTOFF - date(2025, 11, 25)).days)


def test_overdue_days_is_the_amount_weighted_average():
    row = bucket_row([
        invoice("I1", 100, "2025-12-01", due=CUTOFF - timedelta(days=10)),
        invoice("I2", 300, "2025-12-01", due=CUTOFF - timedelta(days=30)),
    ])
    assert row["overdue"] == 400.0
    assert row["overdue_days"] == pytest.approx((100 * 10 + 300 * 30) / 400)  # 25, not 20


def test_overdue_days_ignores_invoices_that_are_not_overdue():
    row = bucket_row([
        invoice("I1", 100, "2025-12-01", due=CUTOFF - timedelta(days=10)),
        invoice("I2", 900, "2025-12-01", due=CUTOFF + timedelta(days=3)),
    ])
    assert row["overdue_days"] == 10.0


def test_overdue_days_is_zero_when_nothing_is_overdue():
    row = bucket_row([invoice("I", 100, "2026-01-06", due=CUTOFF + timedelta(days=3))])
    assert row["overdue"] == 0.0
    assert row["overdue_days"] == 0.0
    assert not math.isnan(row["overdue_days"])


def test_overdue_days_weights_by_outstanding_not_by_invoice_amount():
    row = bucket_row([
        invoice("I1", 1000, "2025-12-01", due=CUTOFF - timedelta(days=10)),
        payment("P1", -900, "2026-01-02", reference="I1"),  # 100 left
        invoice("I2", 100, "2025-12-01", due=CUTOFF - timedelta(days=30)),
    ])
    assert row["overdue"] == 200.0
    assert row["overdue_days"] == pytest.approx(20.0)


def test_overdue_days_weights_in_try_across_currencies():
    row = bucket_row([
        invoice("I1", 100, "2025-12-01", due=CUTOFF - timedelta(days=10), currency="EUR"),
        invoice("I2", 3600, "2025-12-01", due=CUTOFF - timedelta(days=40)),
    ])
    assert row["overdue"] == pytest.approx(7200.0)
    assert row["overdue_days"] == pytest.approx(25.0)


def test_open_invoices_counts_every_open_invoice_of_the_key():
    row = bucket_row([
        invoice("I1", 100, "2025-12-01", due="2025-12-20"),
        invoice("I2", 100, "2026-01-06", due="2026-01-28"),
        invoice("I3", 100, "2026-01-06", due="2026-06-01"),
    ])
    assert row["open_invoices"] == 3


def test_due_buckets_parts_add_up_to_the_total_and_each_invoice_lands_once():
    lines = [invoice(f"I{n}", 100 + n, "2025-12-01", due=CUTOFF + timedelta(days=d))
             for n, d in enumerate([-40, -1, 0, 1, 8, 15, 22, 29, 35, 36, 200])]
    row = bucket_row(lines)
    parts = due_in_week_columns(5) + ["due_later", "overdue"]
    total = sum(100 + n for n in range(11))
    assert sum(row[part] for part in parts) == pytest.approx(total)
    assert row["overdue"] == 100 + 101 + 102           # -40, -1, 0 days
    assert row["due_in_week_1"] == 103                  # +1
    assert row["due_in_week_2"] == 104                  # +8
    assert row["due_in_week_3"] == 105
    assert row["due_in_week_4"] == 106
    assert row["due_in_week_5"] == 107 + 108            # +29, +35 (Sunday of week t+5)
    assert row["due_later"] == 109 + 110                # +36 (Monday), +200
    assert row["open_invoices"] == 11


def test_due_buckets_keeps_keys_apart():
    table = buckets([
        invoice("I1", 100, "2026-01-06", due="2026-01-28", customer="A"),
        invoice("I2", 200, "2026-01-06", due="2026-01-28", customer="B"),
        invoice("I3", 50, "2026-01-06", due="2026-01-28", company="TR04", customer="A"),
    ])
    assert table.set_index("key")["due_in_week_1"].to_dict() == {
        "TR02-A": 100.0, "TR02-B": 200.0, "TR04-A": 50.0}


def test_due_buckets_columns_and_one_row_per_key():
    table = buckets([invoice("I1", 100, "2026-01-06", due="2026-01-28"),
                     invoice("I2", 100, "2026-01-06", due="2026-01-29")], horizons=3)
    assert list(table.columns) == (["key"] + due_in_week_columns(3)
                                   + ["due_later", "overdue", "overdue_days", "open_invoices"])
    assert len(table) == 1


def test_due_buckets_with_horizons_1_sends_week_2_to_due_later():
    row = bucket_row([invoice("I", 100, "2026-01-06", due=monday(WEEK + 2))], horizons=1)
    assert row["due_later"] == 100.0
    assert "due_in_week_2" not in row.index


def test_due_buckets_uses_week_numbers_not_days_ahead():
    # Due Monday of week t+1 is 1 day ahead; due Sunday of week t+1 is 7 days ahead: both t+1.
    row = bucket_row([invoice("I1", 1, "2026-01-06", due=monday(WEEK + 1)),
                      invoice("I2", 2, "2026-01-06", due=sunday(WEEK + 1))])
    assert row["due_in_week_1"] == 3.0
    assert row["due_in_week_2"] == 0.0


def test_due_buckets_tie_out_raises_when_an_invoice_lands_in_no_bucket(monkeypatch):
    """A week number that is not a whole number (NaN) puts an invoice in no bucket."""
    invoices = open_invoices_with_try(ledger([invoice("I", 100, "2026-01-06", due="2026-01-28")]),
                                      CUTOFF)
    monkeypatch.setattr(receivables_asof, "week_numbers",
                        lambda dates, week_zero: pd.Series(float("nan"), index=dates.index))
    with pytest.raises(ValueError, match="does not add up") as error:
        due_buckets(invoices, WEEK, WEEK_ZERO, CUTOFF, 5)
    assert "TR02-A" in str(error.value)
    assert str(CUTOFF) in str(error.value)


def test_due_buckets_tie_out_raises_when_an_invoice_lands_twice(monkeypatch):
    """With one extra due-in-week column, week t+6 lands in it AND in due_later: caught."""
    monkeypatch.setattr(receivables_asof, "due_in_week_columns",
                        lambda horizons: [f"due_in_week_{h}" for h in range(1, horizons + 2)])
    invoices = open_invoices_with_try(
        ledger([invoice("I", 100, "2026-01-06", due=monday(WEEK + 6))]), CUTOFF)
    with pytest.raises(ValueError, match="does not add up"):
        due_buckets(invoices, WEEK, WEEK_ZERO, CUTOFF, 5)


def test_due_buckets_tolerance_is_honoured(monkeypatch):
    invoices = open_invoices_with_try(ledger([invoice("I", 100, "2026-01-06", due="2026-01-28")]),
                                      CUTOFF)
    monkeypatch.setattr(receivables_asof, "TIE_OUT_TOLERANCE", -1.0)
    with pytest.raises(ValueError, match="does not add up"):
        due_buckets(invoices, WEEK, WEEK_ZERO, CUTOFF, 5)


def test_due_buckets_with_a_missing_due_date_raises_value_error():
    """A NaT net due date is not silently bucketed: it is named before any week arithmetic."""
    table = ledger([invoice("I", 100, "2026-01-06", due="2026-01-28"),
                    invoice("J", 50, "2026-01-06", due="2026-01-28")])
    table.loc[0, "net_due_date"] = pd.NaT
    invoices = open_invoices_with_try(table, CUTOFF)
    with pytest.raises(ValueError) as error:
        due_buckets(invoices, WEEK, WEEK_ZERO, CUTOFF, 5)
    assert str(error.value) == (f"at {CUTOFF}, 1 open invoices have no net due date "
                                f"(first: ['I'])")


def test_due_buckets_missing_due_dates_list_at_most_five_documents():
    numbers = [f"I{n}" for n in range(1, 8)]
    table = ledger([invoice(n, 10, "2026-01-06", due="2026-01-28") for n in numbers])
    table["net_due_date"] = pd.NaT
    invoices = open_invoices_with_try(table, CUTOFF)
    with pytest.raises(ValueError) as error:
        due_buckets(invoices, WEEK, WEEK_ZERO, CUTOFF, 5)
    message = str(error.value)
    assert message.startswith(f"at {CUTOFF}, 7 open invoices have no net due date (first: ")
    listed = message[message.index("[") + 1:message.index("]")].replace("'", "").split(", ")
    assert len(listed) == 5
    assert set(listed) <= set(numbers)


def test_receivables_at_with_a_missing_due_date_raises_the_named_error():
    table = ledger([invoice("I", 100, "2026-01-06", due="2026-01-28")])
    table.loc[0, "net_due_date"] = pd.NaT
    with pytest.raises(ValueError, match=r"1 open invoices have no net due date \(first: \['I'\]\)"):
        run(table, rates=NO_RATES)


def test_due_buckets_does_not_modify_its_input():
    invoices = open_invoices_with_try(running_example(), CUTOFF)
    before = invoices.copy()
    due_buckets(invoices, WEEK, WEEK_ZERO, CUTOFF, 5)
    pd.testing.assert_frame_equal(invoices, before)


# --- open_lines_at -------------------------------------------------------------------

def balances(lines, cutoff=CUTOFF, rates=EUR_RATES):
    return open_lines_at(ledger(lines), cutoff, rates_known_at(rates, cutoff))


def balance_row(lines, key="TR02-A", **kwargs):
    return row_of(balances(lines, **kwargs), key)


def test_open_lines_invoice_with_open_partial_payment_is_700():
    row = balance_row([invoice("I", 1000, "2026-01-10"),
                       payment("P", -300, "2026-01-20", reference="I")])
    assert row["open_balance"] == 700.0
    assert row["unapplied_cash"] == 0.0


def test_open_lines_unapplied_cash_lowers_the_balance_and_shows_positive():
    row = balance_row([invoice("I", 1000, "2026-01-10"),
                       payment("P", -300, "2026-01-20", reference="I"),
                       payment("U", -200, "2026-01-22")])
    assert row["open_balance"] == 500.0
    assert row["unapplied_cash"] == 200.0


def test_open_lines_a_partial_payment_is_not_unapplied_cash():
    row = balance_row([payment("P", -300, "2026-01-20", reference="I")])
    assert row["unapplied_cash"] == 0.0
    assert row["open_balance"] == -300.0


def test_open_lines_open_credit_memo_counts():
    row = balance_row([invoice("I", 1000, "2026-01-10"), credit_memo("C", -150, "2026-01-12")])
    assert row["open_balance"] == 850.0
    assert row["unapplied_cash"] == 0.0


def test_open_lines_open_refund_counts():
    row = balance_row([credit_memo("C", -150, "2026-01-12"),
                       payment("R", 100, "2026-01-14", reference="C")])
    assert row["open_balance"] == -50.0
    assert row["unapplied_cash"] == 0.0


def test_open_lines_cleared_lines_do_not_count():
    row = balance_row([
        invoice("I1", 1000, "2026-01-10", cleared="2026-01-20", clearing_document="K1"),
        payment("P1", -1000, "2026-01-20", reference="I1", cleared="2026-01-20",
                clearing_document="K1"),
        invoice("I2", 50, "2026-01-10"),
    ])
    assert row["open_balance"] == 50.0


def test_open_lines_cleared_on_the_cutoff_counts_as_cleared():
    row = balance_row([invoice("I1", 1000, "2026-01-10", cleared=CUTOFF, clearing_document="K"),
                       invoice("I2", 50, "2026-01-10")])
    assert row["open_balance"] == 50.0


def test_open_lines_cleared_after_the_cutoff_is_still_open():
    row = balance_row([invoice("I1", 1000, "2026-01-10",
                               cleared=CUTOFF + timedelta(days=1), clearing_document="K"),
                       payment("U", -200, "2026-01-22", cleared=CUTOFF + timedelta(days=1),
                               clearing_document="K")])
    assert row["open_balance"] == 800.0
    assert row["unapplied_cash"] == 200.0


def test_open_lines_posted_after_the_cutoff_do_not_count():
    row = balance_row([invoice("I1", 1000, "2026-01-10"),
                       payment("U", -200, CUTOFF + timedelta(days=1))])
    assert row["open_balance"] == 1000.0
    assert row["unapplied_cash"] == 0.0


def test_open_lines_posted_on_the_cutoff_count():
    row = balance_row([invoice("I1", 1000, CUTOFF)])
    assert row["open_balance"] == 1000.0


def test_open_lines_foreign_lines_use_the_cutoff_rate():
    row = balance_row([invoice("I", 100, "2025-12-01", currency="EUR"),
                       payment("U", -10, "2026-01-02", currency="EUR")])
    assert row["open_balance"] == pytest.approx(90 * 36.0)
    assert row["unapplied_cash"] == pytest.approx(360.0)


def test_open_lines_foreign_line_with_no_rate_raises():
    with pytest.raises(ValueError, match="GBP"):
        balances([invoice("I", 100, "2025-12-01", currency="GBP")])


def test_open_lines_cleared_foreign_line_with_no_rate_does_not_raise():
    row = balance_row([invoice("I", 100, "2025-12-01", currency="GBP", cleared="2026-01-02",
                               clearing_document="K"),
                       invoice("J", 5, "2025-12-01")])
    assert row["open_balance"] == 5.0


def test_open_lines_one_row_per_key_with_the_columns():
    table = balances([invoice("I1", 10, "2026-01-10", customer="A"),
                      invoice("I2", 20, "2026-01-10", customer="A"),
                      invoice("I3", 30, "2026-01-10", customer="B")])
    assert list(table.columns) == ["key", "unapplied_cash", "open_balance"]
    assert table.set_index("key")["open_balance"].to_dict() == {"TR02-A": 30.0, "TR02-B": 30.0}


def test_open_lines_key_with_nothing_open_gets_no_row():
    table = balances([invoice("I1", 10, "2026-01-10", customer="A", cleared="2026-01-11",
                              clearing_document="K"),
                      invoice("I2", 20, "2026-01-10", customer="B")])
    assert table["key"].tolist() == ["TR02-B"]


# --- receivables_at: the running example ---------------------------------------------

def test_running_example_row():
    row = row_of(run(running_example()))
    assert row["week"] == WEEK
    assert row["due_in_week_1"] == 0.0
    assert row["due_in_week_2"] == 700.0
    assert row["due_in_week_3"] == 0.0
    assert row["due_later"] == 0.0
    assert row["overdue"] == pytest.approx(100 * 36.0)
    assert row["overdue_days"] == 10.0
    assert row["open_invoices"] == 2
    assert row["unapplied_cash"] == 200.0
    assert row["open_balance"] == pytest.approx(700 + 100 * 36.0 - 200)
    assert math.isnan(row["usual_delay_days"])
    assert row["cleared_invoices"] == 0


def test_running_example_columns_are_exactly_receivables_columns():
    assert list(run(running_example()).columns) == receivables_columns(5)


def test_running_example_dtypes():
    table = run(running_example())
    amounts = due_in_week_columns(5) + ["due_later", "overdue", "overdue_days",
                                        "unapplied_cash", "open_balance", "usual_delay_days"]
    for column in amounts:
        assert table[column].dtype == "float64", column
    for column in ["week", "open_invoices", "cleared_invoices"]:
        assert table[column].dtype == "int64", column


def test_horizons_3_gives_three_due_columns():
    table = run(running_example(), horizons=3)
    assert list(table.columns) == receivables_columns(3)
    assert row_of(table)["due_in_week_2"] == 700.0


def test_horizons_1_puts_week_t_plus_2_into_due_later():
    row = row_of(run(running_example(), horizons=1))
    assert row["due_in_week_1"] == 0.0
    assert row["due_later"] == 700.0


def test_a_later_week_revalues_the_eur_invoice_at_the_later_rate():
    week_4 = row_of(run(running_example(), week=4))
    # At the end of week 4 (Sun 2026-02-08) EUR is 37 and INV-1 (due Wed 2026-02-04) is overdue.
    assert week_4["overdue"] == pytest.approx(700 + 100 * 37.0)
    assert week_4["open_balance"] == pytest.approx(700 + 100 * 37.0 - 200)
    week_3 = row_of(run(running_example(), week=3))
    assert week_3["overdue"] == pytest.approx(100 * 36.0)
    assert week_3["due_in_week_1"] == 700.0


def test_the_same_week_ignores_a_rate_published_after_its_sunday():
    rates = rates_table([(date(2026, 1, 2), "EUR", 35.0), (CUTOFF + timedelta(days=1), "EUR", 99.0)])
    row = row_of(run(running_example(), rates=rates))
    assert row["overdue"] == pytest.approx(3500.0)


def test_a_currency_with_no_rate_yet_raises():
    rates = rates_table([(date(2026, 2, 1), "EUR", 36.0)])
    with pytest.raises(ValueError, match="EUR"):
        run(running_example(), rates=rates)


def test_a_currency_with_no_rate_at_all_raises_naming_the_cutoff():
    with pytest.raises(ValueError, match=str(CUTOFF)):
        run(running_example(), rates=NO_RATES)


def test_try_only_ledger_needs_no_rates():
    table = run(ledger([invoice("I", 100, "2026-01-10", due="2026-02-04")]), rates=NO_RATES)
    assert row_of(table)["due_in_week_2"] == 100.0


def test_input_ledger_and_rates_are_not_modified():
    table = running_example()
    before, rates_before = table.copy(), EUR_RATES.copy()
    run(table)
    pd.testing.assert_frame_equal(table, before)
    pd.testing.assert_frame_equal(EUR_RATES, rates_before)


def test_same_inputs_give_the_same_table_twice():
    pd.testing.assert_frame_equal(run(running_example()), run(running_example()))


# --- receivables_at: which keys get a row --------------------------------------------

def test_a_key_whose_only_lines_are_later_gets_no_row():
    lines = running_example_lines() + [invoice("LATE", 100, CUTOFF + timedelta(days=1),
                                               customer="B")]
    assert run(ledger(lines))["key"].tolist() == ["TR02-A"]


def test_a_key_with_a_line_posted_on_the_cutoff_gets_a_row():
    lines = running_example_lines() + [invoice("ON", 100, CUTOFF, due="2026-03-30",
                                               customer="B")]
    assert row_of(run(ledger(lines)), "TR02-B")["due_later"] == 100.0


def test_a_key_with_only_cleared_lines_gets_a_row_of_zeros():
    lines = running_example_lines() + [
        invoice("C1", 500, "2026-01-06", due="2026-01-10", customer="B", cleared="2026-01-12",
                clearing_document="K1"),
        payment("CP1", -500, "2026-01-12", reference="C1", customer="B", cleared="2026-01-12",
                clearing_document="K1"),
    ]
    row = row_of(run(ledger(lines)), "TR02-B")
    for column in due_in_week_columns(5) + ["due_later", "overdue", "overdue_days",
                                            "unapplied_cash", "open_balance"]:
        assert row[column] == 0.0, column
    assert row["open_invoices"] == 0
    assert row["cleared_invoices"] == 1
    assert math.isnan(row["usual_delay_days"])  # 1 cleared < min_cleared 3


def test_a_key_with_only_unapplied_cash_has_no_invoices_but_a_negative_balance():
    lines = running_example_lines() + [payment("U", -80, "2026-01-12", customer="B")]
    row = row_of(run(ledger(lines)), "TR02-B")
    assert row["open_invoices"] == 0
    assert row["unapplied_cash"] == 80.0
    assert row["open_balance"] == -80.0


def test_a_ledger_where_every_line_is_cleared_gives_zero_rows_not_an_error():
    lines = [
        invoice("C1", 500, "2026-01-06", due="2026-01-10", cleared="2026-01-12",
                clearing_document="K1"),
        payment("CP1", -500, "2026-01-12", reference="C1", cleared="2026-01-12",
                clearing_document="K1"),
    ]
    row = row_of(run(ledger(lines), rates=NO_RATES))
    assert row["open_invoices"] == 0
    assert row["open_balance"] == 0.0
    assert row["overdue_days"] == 0.0
    assert row["cleared_invoices"] == 1


def test_rows_are_sorted_by_key_and_index_is_fresh():
    lines = [invoice("I1", 1, "2026-01-06", customer="C"),
             invoice("I2", 1, "2026-01-06", customer="A"),
             invoice("I3", 1, "2026-01-06", customer="B")]
    table = run(ledger(lines), rates=NO_RATES)
    assert table["key"].tolist() == ["TR02-A", "TR02-B", "TR02-C"]
    assert table.index.tolist() == [0, 1, 2]


def test_an_empty_week_before_any_line_gives_an_empty_table_with_the_columns():
    table = run(ledger([invoice("I", 1, "2026-03-01")]), week=0, rates=NO_RATES)
    assert table.empty
    assert list(table.columns) == receivables_columns(5)


def test_an_empty_week_table_has_the_same_dtypes_as_a_full_one():
    table = run(ledger([invoice("I", 1, "2026-03-01")]), week=0, rates=NO_RATES)
    amounts = due_in_week_columns(5) + ["due_later", "overdue", "overdue_days",
                                        "unapplied_cash", "open_balance", "usual_delay_days"]
    for column in amounts:
        assert table[column].dtype == "float64", column
    for column in ["week", "open_invoices", "cleared_invoices"]:
        assert table[column].dtype == "int64", column
    assert table["key"].dtype == object


def test_an_empty_week_before_any_line_on_a_horizon_of_two():
    table = run(ledger([invoice("I", 1, "2026-03-01")]), week=0, rates=NO_RATES, horizons=2)
    assert table.empty
    assert list(table.columns) == receivables_columns(2)


def test_due_parts_add_up_to_the_open_invoices_outstanding_in_try():
    lines = running_example_lines() + [
        invoice("B1", 50, "2025-11-01", due="2025-11-30", customer="B", currency="EUR"),
        invoice("B2", 400, "2026-01-02", due="2026-04-01", customer="B"),
        payment("BP", -100, "2026-01-03", reference="B2", customer="B"),
    ]
    table = run(ledger(lines)).set_index("key")
    invoices = open_invoices_with_try(ledger(lines), CUTOFF)
    expected = invoices.groupby("key")["outstanding_try"].sum()
    parts = due_in_week_columns(5) + ["due_later", "overdue"]
    for key, total in expected.items():
        assert table.loc[key, parts].sum() == pytest.approx(total)


# --- receivables_at: the usual delay -------------------------------------------------

def cleared_history(customer, delays):
    """Invoices cleared before the cutoff, each paid `delay` days after its due date."""
    lines = []
    for n, delay in enumerate(delays):
        due = date(2025, 12, 1) + timedelta(days=n)
        paid = due + timedelta(days=delay)
        lines += [
            invoice(f"{customer}H{n}", 100, "2025-11-01", due=due, customer=customer,
                    cleared=paid, clearing_document=f"{customer}K{n}"),
            payment(f"{customer}HP{n}", -100, paid, reference=f"{customer}H{n}",
                    customer=customer, cleared=paid, clearing_document=f"{customer}K{n}"),
        ]
    return lines


def test_usual_delay_is_the_median_with_enough_history():
    lines = running_example_lines() + cleared_history("A", [2, 10, 4])
    row = row_of(run(ledger(lines)))
    assert row["usual_delay_days"] == 4.0
    assert row["cleared_invoices"] == 3
    assert row["open_invoices"] == 2  # the cleared invoices are not open


def test_usual_delay_is_nan_below_min_cleared_but_the_count_is_kept():
    lines = running_example_lines() + cleared_history("A", [2, 10])
    row = row_of(run(ledger(lines)))
    assert math.isnan(row["usual_delay_days"])
    assert row["cleared_invoices"] == 2


def test_min_cleared_is_passed_through():
    lines = running_example_lines() + cleared_history("A", [2, 10])
    row = row_of(run(ledger(lines), min_cleared=2))
    assert row["usual_delay_days"] == 6.0


def test_a_clearing_after_the_cutoff_is_not_history_yet():
    lines = running_example_lines() + [
        invoice("X", 100, "2026-01-06", due="2026-01-10", cleared=CUTOFF + timedelta(days=2),
                clearing_document="KX"),
        payment("XP", -100, CUTOFF + timedelta(days=2), reference="X",
                cleared=CUTOFF + timedelta(days=2), clearing_document="KX"),
    ]
    row = row_of(run(ledger(lines)))
    assert row["cleared_invoices"] == 0
    assert row["open_invoices"] == 3
    assert row["overdue"] == pytest.approx(3600 + 100)


# --- integration with the synthetic corpus and ledger (SLOW) -------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
AS_OF = date(2026, 8, 28)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]
SYNTHETIC_WEEK_ZERO = date(2025, 6, 2)


@pytest.fixture(scope="module")
def synthetic():
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    daily_files = list_daily_files(CORPUS_FOLDER, SYNTHETIC_LAYOUT.file_name_date_format)
    collections = read_all_files(daily_files, SYNTHETIC_LAYOUT)  # direct read, no cache
    ledger_lines = read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT)  # direct, no cache
    keyed = add_keys(ledger_lines, KNOWN_COMPANIES)
    rates = exchange_rates(collections, keyed)
    return collections, keyed, rates


@pytest.fixture(scope="module")
def week_30(synthetic):
    _, keyed, rates = synthetic
    return receivables_at(keyed, 30, SYNTHETIC_WEEK_ZERO, rates, 5, 3)


def test_slow_week_30_cutoff_is_4_january_2026():
    assert SYNTHETIC_WEEK_ZERO + timedelta(days=7 * 30 + 6) == date(2026, 1, 4)


def test_slow_week_30_has_3548_keys(week_30):
    assert len(week_30) == 3548
    assert week_30["key"].is_unique
    assert list(week_30.columns) == receivables_columns(5)


def test_slow_week_30_key_tr02_11248642(synthetic, week_30):
    _, _, rates = synthetic
    row = row_of(week_30, "TR02-11248642")
    eur = rates_known_at(rates, date(2026, 1, 4))["EUR"]
    assert row["due_in_week_1"] == pytest.approx(122225.68, abs=0.01)
    assert row["due_in_week_2"] == pytest.approx(22506.84 * eur, abs=0.01)
    assert row["due_later"] == pytest.approx(460278.77, abs=0.01)
    assert row["open_invoices"] == 3
    assert row["cleared_invoices"] == 2
    assert math.isnan(row["usual_delay_days"])


def test_slow_week_30_due_parts_tie_to_open_invoices_in_try(synthetic, week_30):
    _, keyed, rates = synthetic
    cutoff = date(2026, 1, 4)
    invoices = open_invoices_at(keyed, cutoff)
    invoices["outstanding_try"] = to_try(invoices["outstanding"], invoices["currency"],
                                         rates_known_at(rates, cutoff), cutoff)
    expected = invoices.groupby("key")["outstanding_try"].sum()
    table = week_30.set_index("key")
    parts = table[due_in_week_columns(5) + ["due_later", "overdue"]].sum(axis=1)
    # Keys with no open invoice have zero parts.
    expected = expected.reindex(table.index, fill_value=0.0)
    assert (parts - expected).abs().max() <= TIE_OUT_TOLERANCE
    counts = invoices.groupby("key").size().reindex(table.index, fill_value=0)
    assert (table["open_invoices"] == counts).all()


def test_slow_week_30_dtypes_and_no_missing_amounts(week_30):
    amounts = due_in_week_columns(5) + ["due_later", "overdue", "overdue_days",
                                        "unapplied_cash", "open_balance"]
    for column in amounts:
        assert week_30[column].dtype == "float64"
        assert week_30[column].notna().all(), column
    for column in ["week", "open_invoices", "cleared_invoices"]:
        assert week_30[column].dtype == "int64"
    assert (week_30["week"] == 30).all()
    assert (week_30[due_in_week_columns(5) + ["due_later", "overdue"]] >= 0).all().all()
    assert (week_30["overdue_days"] >= 0).all()


def known_at(collections, keyed, cutoff):
    """The corpus and ledger as they stood at `cutoff`, and the rates built from them only."""
    end = pd.Timestamp(cutoff)
    cut = keyed[keyed["posting_date"] <= end].copy()
    later = cut["clearing_date"] > end
    cut.loc[later, "clearing_date"] = pd.NaT
    cut["clearing_document"] = cut["clearing_document"].astype(object)
    cut.loc[later, "clearing_document"] = None
    daily = collections[collections["document_date"] <= end]
    return cut, exchange_rates(daily, cut)


@pytest.mark.parametrize("week", [0, 30])
def test_slow_no_leak_from_later_lines_clearings_or_rates(synthetic, week):
    collections, keyed, rates = synthetic
    cutoff = SYNTHETIC_WEEK_ZERO + timedelta(days=7 * week + 6)
    cut, cut_rates = known_at(collections, keyed, cutoff)
    full = receivables_at(keyed, week, SYNTHETIC_WEEK_ZERO, rates, 5, 3)
    as_known = receivables_at(cut, week, SYNTHETIC_WEEK_ZERO, cut_rates, 5, 3)
    assert len(full) > 0
    pd.testing.assert_frame_equal(full, as_known)
