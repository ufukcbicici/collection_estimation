"""Tests for `open_invoices_at` in `hurdle/receivables_asof.py`.

`open_invoices_at(ledger, cutoff)` lists every invoice open at the end of `cutoff`, using only
the ledger lines posted by then: amount, paid so far (linked payments, positive), and what is
still outstanding, all in the invoice's own currency. Ledgers are built by hand with `ledger`
and the line builders below. The SLOW tests at the end read the synthetic ledger in
`data/raw/` once and are skipped when it (or the corpus beside it) is absent; the repo's
`data/interim/` cache is never used.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.keys import add_keys
from collection_estimation.hurdle.read_ledger import read_ledger
from collection_estimation.hurdle.receivables_asof import (
    NOTHING_LEFT,
    OPEN_INVOICE_COLUMNS,
    open_invoices_at,
)
from tests.hurdle.workbooks import SYNTHETIC_LEDGER_LAYOUT

LEDGER_COLUMNS = ["key", "company_code", "customer_id", "document_no", "document_role",
                  "invoice_reference", "posting_date", "net_due_date", "clearing_date",
                  "amount_document", "document_currency"]


def ts(text):
    return pd.Timestamp(text) if text is not None else pd.NaT


def invoice(document_no, amount, posted, due=None, cleared=None, company="TR02",
            customer="100", currency="TRY"):
    return {"key": f"{company}-{customer}", "company_code": company, "customer_id": customer,
            "document_no": document_no, "document_role": "invoice", "invoice_reference": None,
            "posting_date": ts(posted), "net_due_date": ts(due or posted),
            "clearing_date": ts(cleared), "amount_document": float(amount),
            "document_currency": currency}


def payment(document_no, amount, posted, reference=None, cleared=None, company="TR02",
            customer="100", currency="TRY"):
    """`amount` as in the ledger: negative for money received."""
    return {"key": f"{company}-{customer}", "company_code": company, "customer_id": customer,
            "document_no": document_no, "document_role": "payment",
            "invoice_reference": reference, "posting_date": ts(posted),
            "net_due_date": ts(posted), "clearing_date": ts(cleared),
            "amount_document": float(amount), "document_currency": currency}


def credit_memo(document_no, amount, posted, reference=None, cleared=None, company="TR02",
                customer="100", currency="TRY"):
    line = payment(document_no, amount, posted, reference, cleared, company, customer, currency)
    line["document_role"] = "credit_memo"
    return line


def ledger(lines):
    """A ledger table with the columns `open_invoices_at` needs, with ledger-like dtypes."""
    if not lines:
        table = pd.DataFrame({column: pd.Series(dtype=object) for column in LEDGER_COLUMNS})
    else:
        table = pd.DataFrame(lines, columns=LEDGER_COLUMNS)
    for column in ["posting_date", "net_due_date", "clearing_date"]:
        table[column] = pd.to_datetime(table[column])
    table["amount_document"] = table["amount_document"].astype("float64")
    table["invoice_reference"] = table["invoice_reference"].astype(object)
    return table


def running_example():
    """Invoice 1,000 posted 1 Mar due 31 Mar; paid 300 on 20 Mar and 700 on 15 Apr."""
    return ledger([
        invoice("INV1", 1000, "2026-03-01", due="2026-03-31", cleared="2026-04-15"),
        payment("PAY1", -300, "2026-03-20", reference="INV1", cleared="2026-04-15"),
        payment("PAY2", -700, "2026-04-15", reference="INV1", cleared="2026-04-15"),
    ])


def one_row(table):
    assert len(table) == 1, table
    return table.iloc[0]


# --- the running example ---------------------------------------------------------------

def test_running_example_at_22_march_has_one_row_paid_300_outstanding_700():
    row = one_row(open_invoices_at(running_example(), date(2026, 3, 22)))
    assert row["document_no"] == "INV1"
    assert row["amount"] == 1000.0
    assert row["paid"] == 300.0
    assert row["outstanding"] == 700.0
    assert row["posting_date"] == pd.Timestamp("2026-03-01")
    assert row["net_due_date"] == pd.Timestamp("2026-03-31")
    assert row["key"] == "TR02-100"
    assert row["company_code"] == "TR02"
    assert row["customer_id"] == "100"
    assert row["currency"] == "TRY"


def test_running_example_at_15_april_has_no_row():
    table = open_invoices_at(running_example(), date(2026, 4, 15))
    assert table.empty
    assert list(table.columns) == OPEN_INVOICE_COLUMNS


def test_running_example_on_the_day_before_the_second_payment():
    row = one_row(open_invoices_at(running_example(), date(2026, 4, 14)))
    assert (row["paid"], row["outstanding"]) == (300.0, 700.0)


def test_running_example_before_the_first_payment_is_unpaid():
    row = one_row(open_invoices_at(running_example(), date(2026, 3, 19)))
    assert (row["paid"], row["outstanding"]) == (0.0, 1000.0)


def test_running_example_on_the_first_payment_day_counts_it():
    row = one_row(open_invoices_at(running_example(), date(2026, 3, 20)))
    assert (row["paid"], row["outstanding"]) == (300.0, 700.0)


def test_running_example_after_clearing_has_no_row():
    assert open_invoices_at(running_example(), date(2026, 5, 1)).empty


# --- posting date ------------------------------------------------------------------------

def test_invoice_posted_on_the_cutoff_counts():
    table = open_invoices_at(ledger([invoice("I", 50, "2026-03-01")]), date(2026, 3, 1))
    assert one_row(table)["document_no"] == "I"


def test_invoice_posted_the_day_after_the_cutoff_does_not_count():
    table = open_invoices_at(ledger([invoice("I", 50, "2026-03-02")]), date(2026, 3, 1))
    assert table.empty


def test_payment_posted_after_the_cutoff_does_not_count():
    lines = ledger([invoice("I", 100, "2026-03-01"),
                    payment("P", -40, "2026-03-11", reference="I")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 10)))
    assert (row["paid"], row["outstanding"]) == (0.0, 100.0)


def test_payment_posted_on_the_cutoff_counts():
    lines = ledger([invoice("I", 100, "2026-03-01"),
                    payment("P", -40, "2026-03-10", reference="I")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 10)))
    assert (row["paid"], row["outstanding"]) == (40.0, 60.0)


# --- clearing date -----------------------------------------------------------------------

def test_never_cleared_invoice_is_open():
    table = open_invoices_at(ledger([invoice("I", 100, "2026-03-01")]), date(2026, 6, 1))
    assert one_row(table)["outstanding"] == 100.0


def test_invoice_cleared_on_the_cutoff_is_not_open():
    lines = ledger([invoice("I", 100, "2026-03-01", cleared="2026-03-10")])
    assert open_invoices_at(lines, date(2026, 3, 10)).empty


def test_invoice_cleared_before_the_cutoff_is_not_open():
    lines = ledger([invoice("I", 100, "2026-03-01", cleared="2026-03-05")])
    assert open_invoices_at(lines, date(2026, 3, 10)).empty


def test_invoice_cleared_after_the_cutoff_is_open():
    lines = ledger([invoice("I", 100, "2026-03-01", cleared="2026-03-11")])
    assert one_row(open_invoices_at(lines, date(2026, 3, 10)))["outstanding"] == 100.0


def test_invoice_cleared_after_the_cutoff_keeps_its_payments_so_far():
    lines = ledger([invoice("I", 100, "2026-03-01", cleared="2026-03-20"),
                    payment("P1", -30, "2026-03-05", reference="I", cleared="2026-03-20"),
                    payment("P2", -70, "2026-03-20", reference="I", cleared="2026-03-20")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 10)))
    assert (row["paid"], row["outstanding"]) == (30.0, 70.0)


def test_cleared_invoice_with_no_payment_record_is_still_not_open():
    # Cleared by something we cannot see (e.g. a write-off): the clearing date decides.
    lines = ledger([invoice("I", 100, "2026-03-01", cleared="2026-03-05")])
    assert open_invoices_at(lines, date(2026, 3, 6)).empty


# --- linking payments --------------------------------------------------------------------

def test_paid_is_positive_and_sums_several_payments():
    lines = ledger([invoice("I", 1000, "2026-03-01"),
                    payment("P1", -100, "2026-03-02", reference="I"),
                    payment("P2", -250.5, "2026-03-03", reference="I"),
                    payment("P3", -49.5, "2026-03-04", reference="I")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 31)))
    assert row["paid"] == pytest.approx(400.0)
    assert row["outstanding"] == pytest.approx(600.0)


def test_payment_with_the_same_reference_in_another_company_does_not_count():
    lines = ledger([invoice("I", 100, "2026-03-01", company="TR02"),
                    payment("P", -60, "2026-03-02", reference="I", company="TR04")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 31)))
    assert (row["paid"], row["outstanding"]) == (0.0, 100.0)


def test_same_document_number_in_two_companies_each_gets_its_own_payments():
    lines = ledger([invoice("I", 100, "2026-03-01", company="TR02"),
                    invoice("I", 500, "2026-03-01", company="TR04"),
                    payment("P", -10, "2026-03-02", reference="I", company="TR02"),
                    payment("P", -200, "2026-03-02", reference="I", company="TR04")])
    table = open_invoices_at(lines, date(2026, 3, 31)).set_index("company_code")
    assert table.loc["TR02", "paid"] == 10.0 and table.loc["TR02", "outstanding"] == 90.0
    assert table.loc["TR04", "paid"] == 200.0 and table.loc["TR04", "outstanding"] == 300.0


def test_payment_for_another_invoice_does_not_count():
    lines = ledger([invoice("I1", 100, "2026-03-01"), invoice("I2", 100, "2026-03-01"),
                    payment("P", -60, "2026-03-02", reference="I2")])
    table = open_invoices_at(lines, date(2026, 3, 31)).set_index("document_no")
    assert table.loc["I1", "paid"] == 0.0
    assert table.loc["I2", "paid"] == 60.0


def test_unapplied_payment_is_never_applied():
    lines = ledger([invoice("I", 100, "2026-03-01"),
                    payment("P", -100, "2026-03-02", reference=None)])
    row = one_row(open_invoices_at(lines, date(2026, 3, 31)))
    assert (row["paid"], row["outstanding"]) == (0.0, 100.0)


def test_unapplied_payment_with_nan_reference_is_never_applied():
    lines = ledger([invoice("I", 100, "2026-03-01"),
                    payment("P", -100, "2026-03-02", reference=float("nan"))])
    row = one_row(open_invoices_at(lines, date(2026, 3, 31)))
    assert row["paid"] == 0.0


def test_credit_memo_referencing_the_invoice_is_not_applied():
    lines = ledger([invoice("I", 100, "2026-03-01"),
                    credit_memo("M", -40, "2026-03-02", reference="I")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 31)))
    assert (row["paid"], row["outstanding"]) == (0.0, 100.0)


def test_refund_referencing_a_credit_memo_is_not_applied_to_the_invoice():
    lines = ledger([invoice("I", 100, "2026-03-01"),
                    credit_memo("M", -40, "2026-03-02"),
                    payment("R", 40, "2026-03-03", reference="M")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 31)))
    assert (row["paid"], row["outstanding"]) == (0.0, 100.0)


def test_credit_memos_and_payments_are_never_rows():
    lines = ledger([credit_memo("M", -40, "2026-03-02"),
                    payment("P", -100, "2026-03-02"),
                    payment("R", 40, "2026-03-03", reference="M")])
    assert open_invoices_at(lines, date(2026, 3, 31)).empty


def test_outstanding_is_in_the_invoice_currency_without_conversion():
    lines = ledger([invoice("I", 1000, "2026-03-01", currency="EUR"),
                    payment("P", -250, "2026-03-02", reference="I", currency="EUR")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 31)))
    assert row["currency"] == "EUR"
    assert (row["amount"], row["paid"], row["outstanding"]) == (1000.0, 250.0, 750.0)


def test_amount_is_amount_document():
    lines = ledger([invoice("I", 1234.56, "2026-03-01")])
    assert one_row(open_invoices_at(lines, date(2026, 3, 31)))["amount"] == 1234.56


@pytest.mark.parametrize("raw, expected", [(" eur ", "EUR"), ("usd", "USD"), ("TRY", "TRY"),
                                           ("Gbp ", "GBP")])
def test_currency_is_normalised(raw, expected):
    lines = ledger([invoice("I", 100, "2026-03-01", currency=raw)])
    assert one_row(open_invoices_at(lines, date(2026, 3, 31)))["currency"] == expected


# --- nothing left to pay -----------------------------------------------------------------

def test_fully_paid_but_not_cleared_is_left_out_with_a_note(capsys):
    lines = ledger([invoice("I", 100, "2026-03-01"),
                    payment("P", -100, "2026-03-02", reference="I")])
    table = open_invoices_at(lines, date(2026, 3, 31))
    assert table.empty
    assert list(table.columns) == OPEN_INVOICE_COLUMNS
    out = capsys.readouterr().out
    assert ("Note: at 2026-03-31, 1 open invoices have nothing left to pay and are left out"
            in out)


def test_over_paid_but_not_cleared_is_left_out(capsys):
    lines = ledger([invoice("I", 100, "2026-03-01"),
                    payment("P", -150, "2026-03-02", reference="I")])
    assert open_invoices_at(lines, date(2026, 3, 31)).empty
    assert "1 open invoices have nothing left to pay" in capsys.readouterr().out


def test_exactly_nothing_left_threshold_is_left_out(capsys):
    assert NOTHING_LEFT == 0.005
    lines = ledger([invoice("I", 0.005, "2026-03-01")])
    assert open_invoices_at(lines, date(2026, 3, 31)).empty
    assert "1 open invoices have nothing left to pay" in capsys.readouterr().out


def test_just_above_nothing_left_is_kept(capsys):
    lines = ledger([invoice("I", 0.006, "2026-03-01")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 31)))
    assert row["outstanding"] == pytest.approx(0.006)
    assert capsys.readouterr().out == ""


def test_kurus_left_after_payment_is_kept():
    lines = ledger([invoice("I", 100, "2026-03-01"),
                    payment("P", -99.99, "2026-03-02", reference="I")])
    row = one_row(open_invoices_at(lines, date(2026, 3, 31)))
    assert row["outstanding"] == pytest.approx(0.01)


def test_note_counts_every_left_out_invoice_and_others_are_kept(capsys):
    lines = ledger([invoice("A", 100, "2026-03-01"),
                    payment("PA", -100, "2026-03-02", reference="A"),
                    invoice("B", 50, "2026-03-01"),
                    payment("PB", -60, "2026-03-02", reference="B"),
                    invoice("C", 70, "2026-03-01")])
    table = open_invoices_at(lines, date(2026, 3, 31))
    assert table["document_no"].tolist() == ["C"]
    assert ("Note: at 2026-03-31, 2 open invoices have nothing left to pay and are left out"
            in capsys.readouterr().out)


def test_no_note_when_nothing_is_left_out(capsys):
    open_invoices_at(running_example(), date(2026, 3, 22))
    assert capsys.readouterr().out == ""


def test_no_note_for_a_paid_and_cleared_invoice(capsys):
    open_invoices_at(running_example(), date(2026, 4, 15))
    assert capsys.readouterr().out == ""


# --- shape of the result -----------------------------------------------------------------

def test_columns_are_exactly_open_invoice_columns_in_order():
    table = open_invoices_at(running_example(), date(2026, 3, 22))
    assert list(table.columns) == OPEN_INVOICE_COLUMNS


def test_open_invoice_columns_constant():
    assert OPEN_INVOICE_COLUMNS == ["key", "company_code", "customer_id", "document_no",
                                    "currency", "posting_date", "net_due_date", "amount",
                                    "paid", "outstanding"]


def test_index_is_0_to_n_minus_1_even_when_the_ledger_index_is_odd():
    lines = ledger([payment("P0", -5, "2026-03-01"),
                    invoice("A", 100, "2026-03-01"),
                    invoice("B", 100, "2026-03-01", cleared="2026-03-02"),
                    invoice("C", 100, "2026-03-01")])
    lines.index = [70, 50, 30, 10]
    table = open_invoices_at(lines, date(2026, 3, 31))
    assert table.index.tolist() == [0, 1]
    assert table["document_no"].tolist() == ["A", "C"]


def test_empty_ledger_gives_an_empty_table_with_the_columns():
    table = open_invoices_at(ledger([]), date(2026, 3, 31))
    assert table.empty
    assert list(table.columns) == OPEN_INVOICE_COLUMNS


def test_nothing_open_gives_an_empty_table_with_the_columns():
    lines = ledger([invoice("I", 100, "2026-03-01", cleared="2026-03-02"),
                    payment("P", -100, "2026-03-02", reference="I", cleared="2026-03-02")])
    table = open_invoices_at(lines, date(2026, 3, 31))
    assert table.empty
    assert list(table.columns) == OPEN_INVOICE_COLUMNS


def test_cutoff_before_every_line_gives_an_empty_table():
    table = open_invoices_at(running_example(), date(2025, 1, 1))
    assert table.empty
    assert list(table.columns) == OPEN_INVOICE_COLUMNS


def test_input_is_not_modified():
    lines = running_example()
    before = lines.copy(deep=True)
    open_invoices_at(lines, date(2026, 3, 22))
    pd.testing.assert_frame_equal(lines, before)
    assert list(lines.columns) == LEDGER_COLUMNS


def test_cutoff_as_date_and_as_timestamp_give_the_same_table():
    as_date = open_invoices_at(running_example(), date(2026, 3, 22))
    as_timestamp = open_invoices_at(running_example(), pd.Timestamp("2026-03-22"))
    pd.testing.assert_frame_equal(as_date, as_timestamp)


def test_timestamp_cutoff_on_the_clearing_day_is_not_open():
    assert open_invoices_at(running_example(), pd.Timestamp("2026-04-15")).empty


def test_same_inputs_give_the_same_table():
    first = open_invoices_at(running_example(), date(2026, 3, 22))
    second = open_invoices_at(running_example(), date(2026, 3, 22))
    pd.testing.assert_frame_equal(first, second)


def test_several_customers_and_companies_are_kept_apart():
    lines = ledger([
        invoice("A", 100, "2026-03-01", company="TR02", customer="1"),
        invoice("B", 200, "2026-03-01", company="TR02", customer="2", due="2026-04-30"),
        invoice("C", 300, "2026-03-01", company="TR05", customer="1", currency="USD"),
        payment("P", -50, "2026-03-02", reference="B", company="TR02", customer="2"),
    ])
    table = open_invoices_at(lines, date(2026, 3, 31)).set_index("document_no")
    assert table.loc["A", "key"] == "TR02-1"
    assert table.loc["B", "key"] == "TR02-2"
    assert table.loc["C", "key"] == "TR05-1"
    assert table.loc["B", "net_due_date"] == pd.Timestamp("2026-04-30")
    assert table["outstanding"].to_dict() == {"A": 100.0, "B": 150.0, "C": 300.0}
    assert table.loc["C", "currency"] == "USD"


# --- integration with the synthetic ledger (SLOW) ----------------------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
AS_OF = date(2026, 8, 28)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]


@pytest.fixture(scope="module")
def synthetic_ledger():
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    ledger_lines = read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT)  # direct, no cache
    return add_keys(ledger_lines, KNOWN_COMPANIES)


def known_at(full: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """The ledger as it stood at `cutoff`: later lines dropped, later clearings blanked."""
    end = pd.Timestamp(cutoff)
    cut = full[full["posting_date"] <= end].copy()
    cut.loc[cut["clearing_date"] > end, "clearing_date"] = pd.NaT
    return cut


def test_slow_open_invoices_at_2026_01_04(synthetic_ledger, capsys):
    table = open_invoices_at(synthetic_ledger, date(2026, 1, 4))
    cutoff = pd.Timestamp("2026-01-04")
    assert len(table) == 3801
    assert (table["paid"] > 0).sum() == 145
    assert (table["net_due_date"] < cutoff).sum() == 1460
    assert table["key"].nunique() == 1703
    assert "nothing left to pay" not in capsys.readouterr().out


def test_slow_open_invoices_at_2026_08_30(synthetic_ledger, capsys):
    table = open_invoices_at(synthetic_ledger, date(2026, 8, 30))
    assert len(table) == 4045
    assert "nothing left to pay" not in capsys.readouterr().out


@pytest.mark.parametrize("cutoff", [date(2025, 6, 8), date(2026, 1, 4), date(2026, 8, 30)])
def test_slow_amounts_are_consistent(synthetic_ledger, cutoff):
    table = open_invoices_at(synthetic_ledger, cutoff)
    assert list(table.columns) == OPEN_INVOICE_COLUMNS
    assert table.index.tolist() == list(range(len(table)))
    assert (table["outstanding"] > NOTHING_LEFT).all()
    assert (table["outstanding"] <= table["amount"]).all()
    assert (table["paid"] >= 0).all()
    assert (table["posting_date"] <= pd.Timestamp(cutoff)).all()
    assert table["currency"].notna().all()
    assert not table.duplicated(["company_code", "document_no"]).any()


@pytest.mark.parametrize("cutoff", [date(2025, 6, 8), date(2026, 1, 4)])
def test_slow_only_what_was_known_at_the_cutoff_is_used(synthetic_ledger, cutoff):
    full = open_invoices_at(synthetic_ledger, cutoff)
    as_known = open_invoices_at(known_at(synthetic_ledger, cutoff), cutoff)
    pd.testing.assert_frame_equal(full, as_known)
    assert len(full) > 0
