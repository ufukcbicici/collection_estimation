"""Tests for `payment_delays_at` in `hurdle/receivables_asof.py`.

`payment_delays_at(ledger, cutoff, min_cleared)` says how late each key usually pays, from the
invoices posted and cleared by `cutoff`. The money is taken to arrive on the LATEST posting
date of the payments in the invoice's clearing group (same company code and
`clearing_document`); days late = that date - the net due date. Per key: the median days late
(NaN below `min_cleared` cleared invoices) and the count.

Ledgers are built by hand with `ledger` and the line builders below. The SLOW tests at the end
read the synthetic ledger in `data/raw/` once and are skipped when it is absent; the repo's
`data/interim/` cache is never used.
"""
import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.keys import add_keys
from collection_estimation.hurdle.read_ledger import read_ledger
from collection_estimation.hurdle.receivables_asof import DELAY_COLUMNS, payment_delays_at
from tests.hurdle.workbooks import SYNTHETIC_LEDGER_LAYOUT

LEDGER_COLUMNS = ["key", "company_code", "document_no", "document_role", "invoice_reference",
                  "posting_date", "net_due_date", "clearing_date", "clearing_document",
                  "source_row"]
CUTOFF = date(2026, 4, 30)


def ts(text):
    return pd.Timestamp(text) if text is not None else pd.NaT


def line(role, document_no, posted, due=None, cleared=None, clearing_document=None,
         reference=None, company="TR02", customer="100", source_row=None):
    return {"key": f"{company}-{customer}", "company_code": company,
            "document_no": document_no, "document_role": role,
            "invoice_reference": reference, "posting_date": ts(posted),
            "net_due_date": ts(due or posted), "clearing_date": ts(cleared),
            "clearing_document": clearing_document, "source_row": source_row}


def invoice(document_no, posted, due, cleared=None, clearing_document=None, **kwargs):
    return line("invoice", document_no, posted, due, cleared, clearing_document, **kwargs)


def payment(document_no, posted, cleared=None, clearing_document=None, reference=None,
            **kwargs):
    return line("payment", document_no, posted, None, cleared, clearing_document, reference,
                **kwargs)


def credit_memo(document_no, posted, cleared=None, clearing_document=None, reference=None,
                **kwargs):
    return line("credit_memo", document_no, posted, None, cleared, clearing_document,
                reference, **kwargs)


def ledger(lines):
    """A ledger table with the columns `payment_delays_at` needs, with ledger-like dtypes."""
    if not lines:
        table = pd.DataFrame({column: pd.Series(dtype=object) for column in LEDGER_COLUMNS})
    else:
        table = pd.DataFrame(lines, columns=LEDGER_COLUMNS)
    for column in ["posting_date", "net_due_date", "clearing_date"]:
        table[column] = pd.to_datetime(table[column])
    for column in ["invoice_reference", "clearing_document"]:
        table[column] = table[column].astype(object)
    rows = table["source_row"]
    table["source_row"] = pd.Series(
        [r if r is not None and not pd.isna(r) else i + 2 for i, r in enumerate(rows)],
        dtype="int64", index=table.index)
    return table


def running_example():
    """Invoice due 31 Mar; payments 20 Mar and 15 Apr, all in CLR-55, cleared 15 Apr."""
    return ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-55"),
        payment("PAY1", "2026-03-20", cleared="2026-04-15", clearing_document="CLR-55",
                reference="INV1"),
        payment("PAY2", "2026-04-15", cleared="2026-04-15", clearing_document="CLR-55",
                reference="INV1"),
    ])


def cleared_invoice(n, due, paid, company="TR02", customer="100"):
    """An invoice and its one payment, in their own clearing group; `paid` is the payment day."""
    group = f"CLR-{company}-{customer}-{n}"
    return [
        invoice(f"INV{n}", "2026-01-01", due, cleared=paid, clearing_document=group,
                company=company, customer=customer),
        payment(f"PAY{n}", paid, cleared=paid, clearing_document=group, company=company,
                customer=customer),
    ]


def invoices_late_by(days, customer="100", company="TR02"):
    """One cleared invoice per entry of `days`, each paid that many days after its due date."""
    lines = []
    due = pd.Timestamp("2026-02-01")
    for n, late in enumerate(days):
        paid = due + pd.Timedelta(days=late)
        lines += cleared_invoice(f"{customer}-{n}", str(due.date()), str(paid.date()),
                                 company=company, customer=customer)
    return lines


def delay_of(table, key):
    return table.set_index("key").loc[key, "usual_delay_days"]


def count_of(table, key):
    return table.set_index("key").loc[key, "cleared_invoices"]


# --- the constant --------------------------------------------------------------------------

def test_delay_columns_are_key_usual_delay_and_count():
    assert DELAY_COLUMNS == ["key", "usual_delay_days", "cleared_invoices"]


# --- the running example -------------------------------------------------------------------

def test_running_example_is_15_days_late():
    table = payment_delays_at(running_example(), CUTOFF, 1)
    assert table["key"].tolist() == ["TR02-100"]
    assert delay_of(table, "TR02-100") == 15.0
    assert count_of(table, "TR02-100") == 1


def test_running_example_uses_the_latest_payment_not_the_first():
    # The first payment (20 Mar) would give -11.
    assert delay_of(payment_delays_at(running_example(), CUTOFF, 1), "TR02-100") != -11.0


def test_payment_order_in_the_table_does_not_matter():
    reversed_ledger = running_example().iloc[::-1].reset_index(drop=True)
    assert delay_of(payment_delays_at(reversed_ledger, CUTOFF, 1), "TR02-100") == 15.0


# --- the cutoff ----------------------------------------------------------------------------

def test_an_invoice_cleared_on_the_cutoff_counts():
    table = payment_delays_at(running_example(), date(2026, 4, 15), 1)
    assert count_of(table, "TR02-100") == 1
    assert delay_of(table, "TR02-100") == 15.0


def test_an_invoice_cleared_after_the_cutoff_does_not_count():
    table = payment_delays_at(running_example(), date(2026, 4, 14), 1)
    assert len(table) == 0
    assert list(table.columns) == DELAY_COLUMNS


def test_an_open_invoice_does_not_count():
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31"),
        payment("PAY1", "2026-03-20", reference="INV1"),
    ]), CUTOFF, 1)
    assert len(table) == 0


def test_an_invoice_posted_after_the_cutoff_does_not_count():
    # Inconsistent on purpose: cleared before it was posted. Posting is checked first.
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-05-02", "2026-05-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-15", cleared="2026-04-15", clearing_document="CLR-1"),
    ]), CUTOFF, 1)
    assert len(table) == 0


def test_a_payment_posted_after_the_cutoff_is_not_used():
    # The group's later payment is posted after the cutoff, so the earlier one is the latest.
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-10", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY2", "2026-05-05", cleared="2026-04-15", clearing_document="CLR-1"),
    ]), date(2026, 4, 30), 1)
    assert delay_of(table, "TR02-100") == 10.0


def test_a_timestamp_cutoff_gives_the_same_table_as_a_date():
    by_date = payment_delays_at(running_example(), date(2026, 4, 15), 1)
    by_timestamp = payment_delays_at(running_example(), pd.Timestamp("2026-04-15"), 1)
    pd.testing.assert_frame_equal(by_date, by_timestamp)


# --- the clearing group --------------------------------------------------------------------

def test_unapplied_cash_in_the_clearing_group_is_found():
    # The only payment has no invoice_reference: it is found through the clearing document.
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-20",
                clearing_document="CLR-9"),
        payment("PAY1", "2026-04-05", cleared="2026-04-20", clearing_document="CLR-9"),
    ]), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 5.0


def test_unapplied_cash_can_be_the_latest_payment_of_the_group():
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-20",
                clearing_document="CLR-9"),
        payment("PAY1", "2026-03-25", cleared="2026-04-20", clearing_document="CLR-9",
                reference="INV1"),
        payment("PAY2", "2026-04-08", cleared="2026-04-20", clearing_document="CLR-9"),
    ]), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 8.0


def test_a_payment_of_another_company_with_the_same_clearing_document_does_not_count():
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-02", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY9", "2026-04-25", cleared="2026-04-25", clearing_document="CLR-1",
                company="TR04"),
    ]), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 2.0


def test_an_invoice_whose_group_has_only_another_companys_payment_does_not_count():
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY9", "2026-04-02", cleared="2026-04-15", clearing_document="CLR-1",
                company="TR04"),
    ]), CUTOFF, 1)
    assert "TR02-100" not in table["key"].tolist()


def test_a_payment_in_another_clearing_group_does_not_count_even_if_it_names_the_invoice():
    # The clearing group is the source, not invoice_reference.
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-03", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY2", "2026-04-28", cleared="2026-04-28", clearing_document="CLR-2",
                reference="INV1"),
    ]), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 3.0


def test_a_payment_naming_the_invoice_but_in_another_group_alone_does_not_count():
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY2", "2026-04-03", cleared="2026-04-15", clearing_document="CLR-2",
                reference="INV1"),
    ]), CUTOFF, 1)
    assert len(table) == 0


def test_a_payment_with_no_clearing_document_is_not_used():
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-03", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY2", "2026-04-20", reference="INV1"),
    ]), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 3.0


def test_a_credit_memo_in_the_group_is_not_a_payment():
    # The credit memo is later than the payment; only the payment's date is used.
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-25",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-03", cleared="2026-04-25", clearing_document="CLR-1"),
        credit_memo("CM1", "2026-04-25", cleared="2026-04-25", clearing_document="CLR-1"),
    ]), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 3.0


def test_one_clearing_document_clearing_two_invoices_gives_both_the_last_payment_date():
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        invoice("INV2", "2026-03-01", "2026-04-10", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-01", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY2", "2026-04-12", cleared="2026-04-15", clearing_document="CLR-1"),
    ]), CUTOFF, 1)
    # INV1: 12 Apr - 31 Mar = 12; INV2: 12 Apr - 10 Apr = 2; median of two = 7.
    assert count_of(table, "TR02-100") == 2
    assert delay_of(table, "TR02-100") == 7.0


def test_a_shared_clearing_document_does_not_multiply_the_invoices():
    # Several payments in a group must not make an invoice count more than once.
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-04-01", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY2", "2026-04-02", cleared="2026-04-15", clearing_document="CLR-1"),
        payment("PAY3", "2026-04-03", cleared="2026-04-15", clearing_document="CLR-1"),
    ]), CUTOFF, 1)
    assert count_of(table, "TR02-100") == 1


# --- days late -----------------------------------------------------------------------------

def test_early_payment_is_negative():
    table = payment_delays_at(ledger(cleared_invoice(1, "2026-03-31", "2026-03-21")), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == -10.0


def test_payment_on_the_due_date_is_zero():
    table = payment_delays_at(ledger(cleared_invoice(1, "2026-03-31", "2026-03-31")), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 0.0


def test_days_late_are_whole_days_across_a_month_end():
    table = payment_delays_at(ledger(cleared_invoice(1, "2026-02-27", "2026-03-02")), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 3.0


def test_the_clearing_date_itself_is_not_used():
    # Cleared 35 days after the last payment: days late still counts from the payment.
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-25",
                clearing_document="CLR-1"),
        payment("PAY1", "2026-03-21", cleared="2026-04-25", clearing_document="CLR-1"),
    ]), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == -10.0


def test_the_invoice_posting_date_is_not_used():
    early = ledger(cleared_invoice(1, "2026-03-31", "2026-04-05"))
    late = early.copy()
    late.loc[late["document_role"] == "invoice", "posting_date"] = pd.Timestamp("2026-03-30")
    assert (payment_delays_at(early, CUTOFF, 1)["usual_delay_days"].tolist()
            == payment_delays_at(late, CUTOFF, 1)["usual_delay_days"].tolist() == [5.0])


# --- per key: median and count -------------------------------------------------------------

def test_usual_delay_is_the_median_of_an_odd_count():
    table = payment_delays_at(ledger(invoices_late_by([1, 50, 4])), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 4.0
    assert count_of(table, "TR02-100") == 3


def test_usual_delay_of_an_even_count_is_the_mean_of_the_middle_two():
    table = payment_delays_at(ledger(invoices_late_by([1, 2, 5, 60])), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 3.5


def test_usual_delay_is_not_the_mean():
    table = payment_delays_at(ledger(invoices_late_by([0, 0, 30])), CUTOFF, 1)
    assert delay_of(table, "TR02-100") == 0.0


def test_usual_delay_is_float64_and_count_is_int():
    table = payment_delays_at(ledger(invoices_late_by([1, 2, 3])), CUTOFF, 1)
    assert table["usual_delay_days"].dtype == "float64"
    assert pd.api.types.is_integer_dtype(table["cleared_invoices"])


def test_min_cleared_3_with_2_invoices_gives_nan_but_keeps_the_count():
    table = payment_delays_at(ledger(invoices_late_by([1, 2])), CUTOFF, 3)
    assert math.isnan(delay_of(table, "TR02-100"))
    assert count_of(table, "TR02-100") == 2


def test_min_cleared_3_with_3_invoices_gives_a_value():
    table = payment_delays_at(ledger(invoices_late_by([1, 2, 9])), CUTOFF, 3)
    assert delay_of(table, "TR02-100") == 2.0


def test_min_cleared_1_gives_a_value_for_every_key():
    lines = (invoices_late_by([4], customer="100") + invoices_late_by([1, 3], customer="200")
             + invoices_late_by([7, 8, 9], customer="300"))
    table = payment_delays_at(ledger(lines), CUTOFF, 1)
    assert table["usual_delay_days"].notna().all()
    assert table.set_index("key")["usual_delay_days"].to_dict() == {
        "TR02-100": 4.0, "TR02-200": 2.0, "TR02-300": 8.0}


def test_min_cleared_applies_per_key():
    lines = invoices_late_by([1, 2], customer="100") + invoices_late_by([5, 6, 7],
                                                                         customer="200")
    table = payment_delays_at(ledger(lines), CUTOFF, 3)
    assert math.isnan(delay_of(table, "TR02-100"))
    assert delay_of(table, "TR02-200") == 6.0
    assert table.set_index("key")["cleared_invoices"].to_dict() == {"TR02-100": 2,
                                                                    "TR02-200": 3}


def test_keys_of_different_companies_are_kept_apart():
    lines = (invoices_late_by([1], customer="100", company="TR02")
             + invoices_late_by([9], customer="100", company="TR04"))
    table = payment_delays_at(ledger(lines), CUTOFF, 1)
    assert table.set_index("key")["usual_delay_days"].to_dict() == {"TR02-100": 1.0,
                                                                   "TR04-100": 9.0}


# --- invoices cleared without a payment ----------------------------------------------------

def test_an_invoice_cleared_by_a_credit_memo_alone_does_not_count():
    table = payment_delays_at(ledger(
        invoices_late_by([2, 4, 6])
        + [invoice("INVCM", "2026-03-01", "2026-03-31", cleared="2026-04-10",
                   clearing_document="CLR-CM"),
           credit_memo("CM1", "2026-04-10", cleared="2026-04-10", clearing_document="CLR-CM")]
    ), CUTOFF, 1)
    assert count_of(table, "TR02-100") == 3
    assert delay_of(table, "TR02-100") == 4.0


def test_a_key_whose_only_cleared_invoices_had_no_payment_gets_no_row():
    table = payment_delays_at(ledger(
        invoices_late_by([2], customer="100")
        + [invoice("INVCM", "2026-03-01", "2026-03-31", cleared="2026-04-10",
                   clearing_document="CLR-CM", customer="200"),
           credit_memo("CM1", "2026-04-10", cleared="2026-04-10", clearing_document="CLR-CM",
                       customer="200")]
    ), CUTOFF, 1)
    assert table["key"].tolist() == ["TR02-100"]


# --- a cleared invoice without a clearing document -----------------------------------------

def test_a_cleared_invoice_with_no_clearing_document_raises():
    with pytest.raises(ValueError):
        payment_delays_at(ledger([
            invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-04-15"),
        ]), CUTOFF, 1)


def test_the_error_names_the_count_the_cutoff_and_the_first_source_rows():
    lines = [invoice(f"INV{n}", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                     source_row=100 + n) for n in range(7)]
    lines.append(invoice("INVOK", "2026-03-01", "2026-03-31", cleared="2026-04-15",
                         clearing_document="CLR-1", source_row=50))
    with pytest.raises(ValueError) as error:
        payment_delays_at(ledger(lines), CUTOFF, 1)
    message = str(error.value)
    assert message.startswith("7 invoices")
    assert "2026-04-30" in message
    assert "[100, 101, 102, 103, 104]" in message
    assert "105" not in message
    assert "50" not in message


def test_an_open_invoice_with_no_clearing_document_does_not_raise():
    table = payment_delays_at(ledger([invoice("INV1", "2026-03-01", "2026-03-31")]), CUTOFF, 1)
    assert len(table) == 0


def test_an_invoice_cleared_after_the_cutoff_with_no_clearing_document_does_not_raise():
    table = payment_delays_at(ledger([
        invoice("INV1", "2026-03-01", "2026-03-31", cleared="2026-05-15"),
    ]), CUTOFF, 1)
    assert len(table) == 0


def test_a_payment_with_no_clearing_document_does_not_raise():
    table = payment_delays_at(ledger([payment("PAY1", "2026-04-01", cleared="2026-04-01")]),
                              CUTOFF, 1)
    assert len(table) == 0


# --- the shape of the result ---------------------------------------------------------------

def test_columns_are_exactly_delay_columns():
    table = payment_delays_at(running_example(), CUTOFF, 1)
    assert list(table.columns) == DELAY_COLUMNS


def test_index_runs_from_0():
    lines = (invoices_late_by([1], customer="300") + invoices_late_by([2], customer="100")
             + invoices_late_by([3], customer="200"))
    table = payment_delays_at(ledger(lines), CUTOFF, 1)
    assert table.index.tolist() == [0, 1, 2]


def test_one_row_per_key_with_a_counted_invoice():
    lines = (invoices_late_by([1, 2], customer="100") + invoices_late_by([3], customer="200")
             + [invoice("OPEN", "2026-03-01", "2026-03-31", customer="300")])
    table = payment_delays_at(ledger(lines), CUTOFF, 1)
    assert sorted(table["key"]) == ["TR02-100", "TR02-200"]
    assert table["key"].is_unique


def test_an_empty_ledger_gives_an_empty_table_with_the_columns():
    table = payment_delays_at(ledger([]), CUTOFF, 3)
    assert len(table) == 0
    assert list(table.columns) == DELAY_COLUMNS


def test_a_ledger_with_nothing_cleared_gives_an_empty_table_with_the_columns():
    table = payment_delays_at(ledger([invoice("INV1", "2026-03-01", "2026-03-31")]), CUTOFF, 3)
    assert len(table) == 0
    assert list(table.columns) == DELAY_COLUMNS


def test_the_input_is_not_modified():
    original = running_example()
    copy = original.copy(deep=True)
    payment_delays_at(original, CUTOFF, 1)
    pd.testing.assert_frame_equal(original, copy)


def test_calling_twice_gives_the_same_table():
    first = payment_delays_at(running_example(), CUTOFF, 1)
    second = payment_delays_at(running_example(), CUTOFF, 1)
    pd.testing.assert_frame_equal(first, second)


def test_extra_ledger_columns_do_not_leak_into_the_result():
    table = running_example()
    table["amount_document"] = 1000.0
    table["customer_id"] = "100"
    assert list(payment_delays_at(table, CUTOFF, 1).columns) == DELAY_COLUMNS


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


def known_at(full: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """The ledger as it stood at `cutoff`: later lines dropped, later clearings blanked."""
    end = pd.Timestamp(cutoff)
    cut = full[full["posting_date"] <= end].copy()
    later = cut["clearing_date"] > end
    cut.loc[later, "clearing_date"] = pd.NaT
    cut["clearing_document"] = cut["clearing_document"].astype(object)
    cut.loc[later, "clearing_document"] = None
    return cut


def test_slow_payment_delays_at_2026_01_04(synthetic_ledger):
    table = payment_delays_at(synthetic_ledger, date(2026, 1, 4), 3)
    assert len(table) == 3078
    assert table["usual_delay_days"].notna().sum() == 1606
    assert table["usual_delay_days"].isna().sum() == 1472
    assert table["usual_delay_days"].median() == 5.0


def test_slow_payment_delays_shape(synthetic_ledger):
    table = payment_delays_at(synthetic_ledger, date(2026, 1, 4), 3)
    assert list(table.columns) == DELAY_COLUMNS
    assert table.index.tolist() == list(range(len(table)))
    assert table["key"].is_unique
    assert (table["cleared_invoices"] >= 1).all()
    assert table["usual_delay_days"].isna().equals(table["cleared_invoices"] < 3)


@pytest.mark.parametrize("cutoff", [date(2025, 6, 8), date(2026, 1, 4)])
def test_slow_only_what_was_known_at_the_cutoff_is_used(synthetic_ledger, cutoff):
    full = payment_delays_at(synthetic_ledger, cutoff, 3)
    as_known = payment_delays_at(known_at(synthetic_ledger, cutoff), cutoff, 3)
    pd.testing.assert_frame_equal(full, as_known)
    assert len(full) > 0
