"""Tests for `hurdle/fill_from_ledger.py`: `fill_missing_days` and `payment_lines_from_ledger`.

The unit tests build small tables by hand: `collections_table` gives payment lines with the
columns `read_one_file` returns plus `customer_id` and `key` (as after `add_keys`), and
`ledger_table` gives ledger lines with the columns `read_ledger` returns plus `key` and
`customer_id`. Customers are fake.

The hand-built week (Monday 2026-03-02 to Tuesday 2026-03-10):
    Fri 27 Feb   ledger payment, before the first file        -> not filled
    Mon 02 Mar   file; one payment line, tied by the ledger
    Tue 03 Mar   file; two payment lines, tied by the ledger
    Wed 04 Mar   NO FILE; ledger: applied payment, unapplied USD cash, refund paid out,
                 an invoice and a credit memo                 -> the three payments filled
    Thu 05 Mar   file; one payment line
    Fri 06 Mar   file; no payment lines, no ledger payment
    Mon 09 Mar   NO FILE; ledger: only an invoice             -> nothing filled, not a filled day
    Tue 10 Mar   file (the last); one payment line
    Wed 11 Mar   ledger payment, after the last file          -> not filled

The slow tests at the end read the synthetic corpus and ledger in `data/raw/` directly (never
the `data/interim/` cache), once per module, and are skipped when they are absent.
"""
import copy
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import fill_from_ledger
from collection_estimation.hurdle.fill_from_ledger import (
    FILL_TOLERANCE,
    FROM_LEDGER,
    fill_missing_days,
    payment_lines_from_ledger,
)
from collection_estimation.hurdle.read_collections import LINE_COLUMNS, TEXT_COLUMNS
from collection_estimation.hurdle.read_ledger import (
    LEDGER_COLUMNS,
    LEDGER_TEXT_COLUMNS,
    check_ledger_against_collections,
)
from collection_estimation.hurdle.series import (
    add_week_index,
    check_weekly_totals,
    weekly_amounts,
)

COLLECTION_COLUMNS = LINE_COLUMNS + ["customer_id", "key"]
RESULT_COLUMNS = COLLECTION_COLUMNS + ["filled_from_ledger"]
RESULT_TEXT_COLUMNS = TEXT_COLUMNS + ["customer_id", "key"]

BEFORE = date(2026, 2, 27)
MON, TUE, WED, THU, FRI = (date(2026, 3, d) for d in (2, 3, 4, 5, 6))
NEXT_MON, NEXT_TUE, AFTER = date(2026, 3, 9), date(2026, 3, 10), date(2026, 3, 11)
DAYS_WITH_FILE = [MON, TUE, THU, FRI, NEXT_TUE]
WEEK_ZERO = MON


# --- builders --------------------------------------------------------------------------

def collection_line(day, amount_try, customer="0000100001", company="TR02", sap="1400000001",
                    source_row=3, **extra):
    """One daily-file payment line, as read_one_file + add_keys give it."""
    customer_id = customer.lstrip("0")
    line = {
        "company_code": company,
        "customer": customer,
        "customer_name": "Fake Customer",
        "document_date": day,
        "bank": "Fake Bank",
        "sap_document_no": sap,
        "invoice_no": "GIB2026000001",
        "amount_original": amount_try,
        "currency_original": "TRY",
        "amount_try": amount_try,
        "currency_local": "TRY",
        "source_file": f"{day:%d%m%Y} fake.xlsx",
        "source_row": source_row,
        "customer_id": customer_id,
        "key": f"{company}-{customer_id}",
    }
    line.update(extra)
    return line


def collections_table(lines):
    table = pd.DataFrame(lines, columns=COLLECTION_COLUMNS)
    table["document_date"] = pd.to_datetime(table["document_date"])
    for column in ("amount_original", "amount_try"):
        table[column] = table[column].astype("float64")
    table["source_row"] = table["source_row"].astype("int64")
    for column in RESULT_TEXT_COLUMNS:
        table[column] = table[column].astype(object).where(table[column].notna(), None)
    return table


def ledger_line(role, day, amount_local, *, amount_document=None, currency="TRY",
                document_no="1400000001", invoice_reference="9000000001",
                customer="0000100001", company="TR02", source_row=2):
    """One ledger line, as read_ledger + add_keys give it (SAP signs: payment received -)."""
    customer_id = customer.lstrip("0")
    type_of_role = {"invoice": "RV", "payment": "DZ", "credit_memo": "DG"}
    return {
        "company_code": company,
        "customer": customer,
        "customer_name": None,
        "document_no": document_no,
        "document_type": type_of_role[role],
        "reference": None,
        "invoice_reference": invoice_reference,
        "document_date": day,
        "posting_date": day,
        "baseline_date": None,
        "payment_terms": None,
        "net_due_date": day if role == "invoice" else None,
        "amount_document": amount_local if amount_document is None else amount_document,
        "document_currency": currency,
        "amount_local": amount_local,
        "clearing_date": None,
        "clearing_document": None,
        "document_role": role,
        "source_row": source_row,
        "key": f"{company}-{customer_id}",
        "customer_id": customer_id,
    }


def ledger_table(lines):
    table = pd.DataFrame(lines, columns=LEDGER_COLUMNS + ["key", "customer_id"])
    for column in ("document_date", "posting_date", "baseline_date", "net_due_date",
                   "clearing_date"):
        table[column] = pd.to_datetime(table[column])
    for column in ("amount_document", "amount_local"):
        table[column] = table[column].astype("float64")
    table["source_row"] = table["source_row"].astype("int64")
    # As read_ledger does: text columns are object with None (under pandas 3 they would
    # otherwise be inferred as the `str` dtype, with NaN for an empty value).
    for column in LEDGER_TEXT_COLUMNS + ["key", "customer_id"]:
        table[column] = table[column].astype(object).where(table[column].notna(), None)
    return table


def week_collections():
    return collections_table([
        collection_line(MON, 100.0, source_row=3),
        collection_line(TUE, 30.0, source_row=3, customer="0000100002", sap="1400000002"),
        collection_line(TUE, 20.0, source_row=4, customer="0000100003", sap=None),
        collection_line(THU, 200.0, source_row=3),
        collection_line(NEXT_TUE, 10.0, source_row=3, company="TR04"),
    ])


def week_ledger():
    return ledger_table([
        ledger_line("payment", BEFORE, -70.0, document_no="1400000100", source_row=2),
        ledger_line("payment", MON, -100.0, document_no="1400000101", source_row=3),
        ledger_line("payment", TUE, -30.0, document_no="1400000102", customer="0000100002",
                    source_row=4),
        ledger_line("payment", TUE, -20.0, document_no="1400000103", customer="0000100003",
                    invoice_reference=None, source_row=5),
        # The missing Wednesday: an applied payment, unapplied USD cash, a refund paid out,
        # an invoice and a credit memo.
        ledger_line("payment", WED, -300.0, document_no="1400000104", customer="0000100002",
                    invoice_reference="9000000002", source_row=6),
        ledger_line("payment", WED, -40.0, amount_document=-1.25, currency="USD",
                    document_no="1400000105", customer="0000100004", company="TR05",
                    invoice_reference=None, source_row=7),
        ledger_line("payment", WED, 25.0, document_no="1400000106", customer="0000100003",
                    invoice_reference="9000000003", source_row=8),
        ledger_line("invoice", WED, 1000.0, document_no="9000000010", invoice_reference=None,
                    source_row=9),
        ledger_line("credit_memo", WED, -10.0, document_no="9100000010",
                    invoice_reference="9000000010", source_row=10),
        ledger_line("payment", THU, -200.0, document_no="1400000107", source_row=11),
        # The missing next Monday has only an invoice.
        ledger_line("invoice", NEXT_MON, 500.0, document_no="9000000011",
                    invoice_reference=None, source_row=12),
        ledger_line("payment", NEXT_TUE, -10.0, document_no="1400000108", company="TR04",
                    source_row=13),
        ledger_line("payment", AFTER, -80.0, document_no="1400000109", source_row=14),
    ])


def fill(collections=None, ledger=None, days=None):
    return fill_missing_days(week_collections() if collections is None else collections,
                             week_ledger() if ledger is None else ledger,
                             DAYS_WITH_FILE if days is None else days)


def filled_part(result):
    return result[result["filled_from_ledger"]].reset_index(drop=True)


# --- constants -------------------------------------------------------------------------

def test_constants():
    assert FROM_LEDGER == "ledger"
    assert FILL_TOLERANCE == 0.01


# --- what is filled --------------------------------------------------------------------

def test_returns_the_lines_and_the_filled_days():
    result, filled_days = fill()
    assert isinstance(result, pd.DataFrame)
    assert filled_days == [WED]


def test_the_three_wednesday_payments_are_filled():
    filled = filled_part(fill()[0])
    assert len(filled) == 3
    assert set(filled["document_date"].dt.date) == {WED}
    assert filled["source_row"].tolist() == [6, 7, 8]


def test_line_count_is_originals_plus_filled():
    result, _ = fill()
    assert len(result) == len(week_collections()) + 3
    assert (~result["filled_from_ledger"]).sum() == len(week_collections())


def test_ledger_payment_before_the_first_file_is_not_filled():
    result, filled_days = fill()
    assert BEFORE not in set(result["document_date"].dt.date)
    assert BEFORE not in filled_days


def test_ledger_payment_after_the_last_file_is_not_filled():
    result, filled_days = fill()
    assert AFTER not in set(result["document_date"].dt.date)
    assert AFTER not in filled_days


def test_invoices_and_credit_memos_on_a_missing_day_are_not_filled():
    filled = filled_part(fill()[0])
    assert 9 not in filled["source_row"].tolist()   # the invoice
    assert 10 not in filled["source_row"].tolist()  # the credit memo
    assert not filled["sap_document_no"].isin(["9000000010", "9100000010"]).any()


def test_a_missing_day_with_only_an_invoice_is_not_a_filled_day():
    result, filled_days = fill()
    assert NEXT_MON not in filled_days
    assert NEXT_MON not in set(result["document_date"].dt.date)


def test_payments_on_days_with_a_file_are_not_duplicated():
    result, _ = fill()
    for day in (MON, TUE, THU, NEXT_TUE):
        on_day = result[result["document_date"].dt.date == day]
        assert not on_day["filled_from_ledger"].any()
    assert (result["source_file"] == FROM_LEDGER).sum() == 3


def test_the_first_and_last_day_with_a_file_bound_the_range_strictly():
    # A ledger payment on the first or last file day is compared, not filled.
    result, _ = fill()
    assert not result.loc[result["document_date"].dt.date.isin([MON, NEXT_TUE]),
                          "filled_from_ledger"].any()


def test_days_with_file_may_be_unsorted_with_duplicates():
    expected = fill()
    result = fill(days=[NEXT_TUE, THU, MON, FRI, TUE, MON])
    pd.testing.assert_frame_equal(result[0], expected[0])
    assert result[1] == expected[1]


@pytest.mark.parametrize("as_given", [
    lambda d: datetime(d.year, d.month, d.day),
    lambda d: datetime(d.year, d.month, d.day, 14, 30),
    pd.Timestamp,
], ids=["datetime", "datetime-with-time", "timestamp"])
def test_days_with_file_as_datetimes_or_timestamps_give_the_same_result(as_given):
    expected = fill()
    result = fill(days=[as_given(d) for d in DAYS_WITH_FILE])
    pd.testing.assert_frame_equal(result[0], expected[0])
    assert result[1] == expected[1]


def test_a_ledger_payment_with_a_time_of_day_is_filled_on_its_day():
    ledger = week_ledger()
    ledger.loc[ledger["source_row"] == 6, "posting_date"] = pd.Timestamp("2026-03-04 15:45")
    result, filled_days = fill(ledger=ledger)
    assert filled_days == [WED]
    assert len(filled_part(result)) == 3


def test_payments_on_several_missing_days_give_sorted_filled_days():
    ledger = pd.concat([
        week_ledger(),
        ledger_table([ledger_line("payment", NEXT_MON, -5.0, document_no="1400000200",
                                  source_row=20)]),
    ], ignore_index=True)
    result, filled_days = fill(ledger=ledger)
    assert filled_days == [WED, NEXT_MON]
    assert all(type(day) is date for day in filled_days)
    assert len(filled_part(result)) == 4


def test_filled_days_are_plain_dates_without_duplicates():
    _, filled_days = fill()
    assert len(filled_days) == len(set(filled_days))
    assert all(type(day) is date for day in filled_days)


def test_no_missing_day_with_payments_fills_nothing():
    ledger = week_ledger()
    ledger = ledger[~((ledger["posting_date"].dt.date == WED)
                      & (ledger["document_role"] == "payment"))].reset_index(drop=True)
    result, filled_days = fill(ledger=ledger)
    assert filled_days == []
    assert not result["filled_from_ledger"].any()
    expected = week_collections().sort_values("document_date", kind="stable",
                                              ignore_index=True)
    pd.testing.assert_frame_equal(result.drop(columns="filled_from_ledger"), expected)


def test_no_missing_day_keeps_the_column_types():
    ledger = week_ledger()
    ledger = ledger[ledger["posting_date"].dt.date != WED].reset_index(drop=True)
    result, _ = fill(ledger=ledger)
    assert result["amount_try"].dtype == "float64"
    assert result["amount_original"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(result["document_date"])
    assert result["filled_from_ledger"].dtype == bool


def test_a_weekend_payment_between_files_is_filled():
    # Filling is by "no file that day", not by business day; the week calendar ignores a
    # filled weekend day (see test_series.py).
    saturday = date(2026, 3, 7)
    ledger = pd.concat([
        week_ledger(),
        ledger_table([ledger_line("payment", saturday, -9.0, document_no="1400000300",
                                  source_row=30)]),
    ], ignore_index=True)
    _, filled_days = fill(ledger=ledger)
    assert filled_days == [WED, saturday]


# --- each filled line ------------------------------------------------------------------

def filled_by_row():
    filled = filled_part(fill()[0])
    return {int(row["source_row"]): row for _, row in filled.iterrows()}


def test_a_filled_applied_payment_line():
    line = filled_by_row()[6]
    assert line["company_code"] == "TR02"
    assert line["customer"] == "0000100002"
    assert line["customer_id"] == "100002"
    assert line["key"] == "TR02-100002"
    assert line["document_date"] == pd.Timestamp(WED)
    assert line["amount_try"] == 300.0
    assert line["amount_original"] == 300.0
    assert line["currency_original"] == "TRY"
    assert line["currency_local"] == "TRY"
    assert line["sap_document_no"] == "1400000104"
    assert line["customer_name"] is None
    assert line["bank"] is None
    assert line["invoice_no"] is None
    assert line["source_file"] == FROM_LEDGER
    assert line["source_row"] == 6
    assert bool(line["filled_from_ledger"]) is True


def test_a_filled_unapplied_payment_has_no_sap_document_number():
    line = filled_by_row()[7]
    assert line["sap_document_no"] is None
    assert line["company_code"] == "TR05"
    assert line["key"] == "TR05-100004"


def test_a_filled_foreign_currency_payment_keeps_both_amounts_sign_flipped():
    line = filled_by_row()[7]
    assert line["amount_try"] == 40.0
    assert line["amount_original"] == 1.25
    assert line["currency_original"] == "USD"
    assert line["currency_local"] == "TRY"


def test_a_refund_paid_out_becomes_a_negative_line():
    line = filled_by_row()[8]
    assert line["amount_try"] == -25.0
    assert line["amount_original"] == -25.0
    assert line["sap_document_no"] == "1400000106"


def test_filled_lines_take_customer_id_and_key_from_the_ledger_not_recomputed():
    ledger = week_ledger()
    ledger.loc[ledger["source_row"] == 6, "customer_id"] = "MASTER-1"
    ledger.loc[ledger["source_row"] == 6, "key"] = "TR02-MASTER-1"
    line = filled_part(fill(ledger=ledger)[0]).set_index("source_row").loc[6]
    assert line["customer_id"] == "MASTER-1"
    assert line["key"] == "TR02-MASTER-1"
    assert line["customer"] == "0000100002"


def test_document_date_is_the_posting_date_not_the_document_date():
    ledger = week_ledger()
    ledger.loc[ledger["source_row"] == 6, "document_date"] = pd.Timestamp("2026-02-20")
    line = filled_part(fill(ledger=ledger)[0]).set_index("source_row").loc[6]
    assert line["document_date"] == pd.Timestamp(WED)


def test_filled_total_equals_the_ledger_payments_on_the_missing_days():
    result, _ = fill()
    tie = check_ledger_against_collections(week_ledger(), week_collections(), DAYS_WITH_FILE)
    assert filled_part(result)["amount_try"].sum() == pytest.approx(
        tie["try_on_missing_file_days"], abs=1e-9)
    assert tie["try_on_missing_file_days"] == pytest.approx(315.0)


# --- the result table ------------------------------------------------------------------

def test_result_columns_are_the_collection_columns_plus_the_flag():
    result, _ = fill()
    assert list(result.columns) == RESULT_COLUMNS


def test_an_extra_collection_column_the_ledger_cannot_fill_raises_naming_it():
    collections = week_collections()
    collections.insert(3, "note", "x")
    ledger = week_ledger()
    with pytest.raises(ValueError, match="note"):
        fill_missing_days(collections, ledger, DAYS_WITH_FILE)


def test_every_original_line_is_kept_unchanged_and_flagged_false():
    result, _ = fill()
    originals = result[~result["filled_from_ledger"]].drop(columns="filled_from_ledger")
    expected = week_collections()
    pd.testing.assert_frame_equal(originals.reset_index(drop=True), expected)


def test_result_is_sorted_by_document_date():
    result, _ = fill()
    assert result["document_date"].is_monotonic_increasing


def test_result_index_is_a_plain_range():
    result, _ = fill()
    assert result.index.tolist() == list(range(len(result)))


def test_sort_is_stable_within_a_day_for_original_lines():
    collections = collections_table([
        collection_line(THU, 200.0, source_row=9),
        collection_line(MON, 100.0, source_row=3),
        collection_line(TUE, 20.0, source_row=7, customer="0000100003", sap=None),
        collection_line(TUE, 30.0, source_row=2, customer="0000100002", sap="1400000002"),
        collection_line(NEXT_TUE, 10.0, source_row=3, company="TR04"),
    ])
    result, _ = fill(collections=collections)
    tuesday = result[result["document_date"].dt.date == TUE]
    assert tuesday["source_row"].tolist() == [7, 2]


def test_filled_lines_keep_the_ledger_order_within_a_day():
    ledger = week_ledger()
    ledger = pd.concat([ledger[ledger["source_row"] == 8], ledger[ledger["source_row"] != 8]],
                       ignore_index=True)
    filled = filled_part(fill(ledger=ledger)[0])
    assert filled["source_row"].tolist() == [8, 6, 7]


def test_unsorted_collections_come_back_sorted():
    collections = week_collections().iloc[::-1].reset_index(drop=True)
    result, _ = fill(collections=collections)
    assert result["document_date"].is_monotonic_increasing
    assert result.index.tolist() == list(range(len(result)))


def test_text_columns_hold_none_not_nan():
    result, _ = fill()
    for column in RESULT_TEXT_COLUMNS:
        values = result[column].tolist()
        assert all(value is None or isinstance(value, str) for value in values), column
        assert result[column].dtype == object, column


def test_filled_text_columns_with_no_value_are_none():
    filled = filled_part(fill()[0])
    for column in ("customer_name", "bank", "invoice_no"):
        assert all(value is None for value in filled[column]), column


def test_amounts_are_float64_and_document_date_datetime64():
    result, _ = fill()
    assert result["amount_try"].dtype == "float64"
    assert result["amount_original"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(result["document_date"])


def test_source_row_holds_integers():
    result, _ = fill()
    assert pd.api.types.is_integer_dtype(result["source_row"])


def test_filled_from_ledger_is_bool():
    result, _ = fill()
    assert result["filled_from_ledger"].dtype == bool
    assert result["filled_from_ledger"].sum() == 3


def test_filling_twice_gives_equal_results():
    first = fill()
    second = fill()
    pd.testing.assert_frame_equal(first[0], second[0])
    assert first[1] == second[1]


# --- the tie is checked first ----------------------------------------------------------

def test_a_mismatch_on_a_day_with_a_file_raises_the_tie_error():
    ledger = week_ledger()
    ledger.loc[ledger["source_row"] == 3, "amount_local"] = -101.0  # Monday: 101 vs 100
    with pytest.raises(ValueError, match="differ from the daily files"):
        fill(ledger=ledger)


def test_a_daily_line_without_a_ledger_payment_raises_the_tie_error():
    collections = pd.concat([
        week_collections(), collections_table([collection_line(FRI, 5.0, source_row=3)]),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="differ from the daily files"):
        fill(collections=collections)


def test_the_tie_is_checked_before_anything_is_filled(monkeypatch):
    calls = []

    def failing_tie(ledger, collections, days):
        calls.append(days)
        raise ValueError("tie failed")

    def must_not_run(*args, **kwargs):
        raise AssertionError("payment_lines_from_ledger ran before the tie was checked")

    monkeypatch.setattr(fill_from_ledger, "check_ledger_against_collections", failing_tie)
    monkeypatch.setattr(fill_from_ledger, "payment_lines_from_ledger", must_not_run)
    with pytest.raises(ValueError, match="tie failed"):
        fill()
    assert len(calls) == 1


def test_the_tie_gets_the_same_three_inputs(monkeypatch):
    seen = {}
    real = fill_from_ledger.check_ledger_against_collections

    def spy(ledger, collections, days):
        seen["args"] = (ledger, collections, days)
        return real(ledger, collections, days)

    monkeypatch.setattr(fill_from_ledger, "check_ledger_against_collections", spy)
    collections, ledger = week_collections(), week_ledger()
    fill_missing_days(collections, ledger, DAYS_WITH_FILE)
    assert seen["args"][0] is ledger
    assert seen["args"][1] is collections
    assert list(seen["args"][2]) == DAYS_WITH_FILE


def test_empty_days_with_file_raises_value_error():
    with pytest.raises(ValueError):
        fill(days=[])


# --- the TRY total check ---------------------------------------------------------------

def tie_reporting(total):
    real = check_ledger_against_collections

    def fake(ledger, collections, days):
        result = real(ledger, collections, days)
        result["try_on_missing_file_days"] = total
        return result
    return fake


def test_the_total_check_passes_when_the_totals_agree():
    result, _ = fill()  # no error: 315 filled, 315 in the tie
    assert filled_part(result)["amount_try"].sum() == pytest.approx(315.0)


def test_a_filled_total_that_differs_from_the_tie_raises(monkeypatch):
    monkeypatch.setattr(fill_from_ledger, "check_ledger_against_collections",
                        tie_reporting(999.0))
    with pytest.raises(ValueError) as error:
        fill()
    message = str(error.value)
    assert "315.00" in message
    assert "999.00" in message


def test_a_difference_within_the_tolerance_passes(monkeypatch):
    monkeypatch.setattr(fill_from_ledger, "check_ledger_against_collections",
                        tie_reporting(315.0 + FILL_TOLERANCE / 2))
    result, _ = fill()
    assert filled_part(result)["amount_try"].sum() == pytest.approx(315.0)


def test_a_difference_just_over_the_tolerance_raises(monkeypatch):
    monkeypatch.setattr(fill_from_ledger, "check_ledger_against_collections",
                        tie_reporting(315.0 + 0.011))
    with pytest.raises(ValueError, match="filled lines add up"):
        fill()


def test_a_tie_claiming_money_on_missing_days_when_none_is_filled_raises(monkeypatch):
    ledger = week_ledger()
    ledger = ledger[ledger["posting_date"].dt.date != WED].reset_index(drop=True)
    monkeypatch.setattr(fill_from_ledger, "check_ledger_against_collections",
                        tie_reporting(50.0))
    with pytest.raises(ValueError, match="filled lines add up"):
        fill(ledger=ledger)


# --- inputs are not modified -----------------------------------------------------------

def test_inputs_are_not_modified():
    collections, ledger = week_collections(), week_ledger()
    days = list(DAYS_WITH_FILE)
    collections_before, ledger_before = collections.copy(deep=True), ledger.copy(deep=True)
    days_before = copy.copy(days)
    fill_missing_days(collections, ledger, days)
    pd.testing.assert_frame_equal(collections, collections_before)
    pd.testing.assert_frame_equal(ledger, ledger_before)
    assert days == days_before
    assert "filled_from_ledger" not in collections.columns


def test_changing_the_result_does_not_change_the_inputs():
    collections, ledger = week_collections(), week_ledger()
    result, _ = fill_missing_days(collections, ledger, DAYS_WITH_FILE)
    result.loc[:, "amount_try"] = 0.0
    result.loc[:, "customer"] = "changed"
    pd.testing.assert_frame_equal(collections, week_collections())
    pd.testing.assert_frame_equal(ledger, week_ledger())


# --- payment_lines_from_ledger ---------------------------------------------------------

def wednesday_payments():
    ledger = week_ledger()
    return ledger[(ledger["posting_date"].dt.date == WED)
                  & (ledger["document_role"] == "payment")]


def test_payment_lines_from_ledger_gives_the_requested_columns_in_order():
    lines = payment_lines_from_ledger(wednesday_payments(), pd.Index(COLLECTION_COLUMNS))
    assert list(lines.columns) == COLLECTION_COLUMNS
    assert lines.index.tolist() == [0, 1, 2]


def test_payment_lines_from_ledger_accepts_a_subset_in_any_order():
    lines = payment_lines_from_ledger(wednesday_payments(),
                                      pd.Index(["amount_try", "key", "source_file"]))
    assert list(lines.columns) == ["amount_try", "key", "source_file"]
    assert lines["amount_try"].tolist() == [300.0, 40.0, -25.0]
    assert set(lines["source_file"]) == {FROM_LEDGER}


def test_payment_lines_from_ledger_accepts_a_plain_list_of_columns():
    lines = payment_lines_from_ledger(wednesday_payments(), COLLECTION_COLUMNS)
    assert list(lines.columns) == COLLECTION_COLUMNS


def test_payment_lines_from_ledger_raises_naming_a_column_it_cannot_fill():
    with pytest.raises(ValueError) as error:
        payment_lines_from_ledger(wednesday_payments(),
                                  pd.Index(COLLECTION_COLUMNS + ["week"]))
    assert "week" in str(error.value)


def test_payment_lines_from_ledger_names_every_column_it_cannot_fill():
    with pytest.raises(ValueError) as error:
        payment_lines_from_ledger(wednesday_payments(),
                                  pd.Index(["amount_try", "week", "region"]))
    message = str(error.value)
    assert "week" in message and "region" in message
    assert "amount_try" not in message


def test_payment_lines_from_ledger_on_no_payments_gives_an_empty_table():
    lines = payment_lines_from_ledger(wednesday_payments().iloc[0:0],
                                      pd.Index(COLLECTION_COLUMNS))
    assert lines.empty
    assert list(lines.columns) == COLLECTION_COLUMNS


def test_payment_lines_from_ledger_does_not_modify_the_payments():
    payments = wednesday_payments()
    before = payments.copy(deep=True)
    payment_lines_from_ledger(payments, pd.Index(COLLECTION_COLUMNS))
    pd.testing.assert_frame_equal(payments, before)


def test_payment_lines_from_ledger_text_columns_hold_none_not_nan():
    lines = payment_lines_from_ledger(wednesday_payments(), pd.Index(COLLECTION_COLUMNS))
    for column in lines.columns:
        if lines[column].dtype == object:
            assert all(value is None or isinstance(value, str) for value in lines[column]), \
                column


def test_payment_lines_from_ledger_unapplied_is_decided_by_invoice_reference():
    payments = wednesday_payments().copy()
    payments["invoice_reference"] = None  # every line unapplied
    lines = payment_lines_from_ledger(payments, pd.Index(COLLECTION_COLUMNS))
    assert all(value is None for value in lines["sap_document_no"])


def test_a_ledger_with_pandas_str_text_columns_still_gives_none_for_unapplied_cash():
    # Under pandas 3 a ledger whose text columns have the `str` dtype (e.g. built with
    # pd.DataFrame, not read_ledger) holds NaN for an empty value; the unapplied line's SAP
    # document number must still come out as None.
    payments = wednesday_payments().copy()
    for column in ("document_no", "invoice_reference"):
        payments[column] = payments[column].astype("str")
    lines = payment_lines_from_ledger(payments, pd.Index(COLLECTION_COLUMNS))
    assert lines["sap_document_no"].tolist()[1] is None


def str_dtype_wednesday_payments():
    """The Wednesday payments with every ledger text column in pandas' `str` dtype."""
    payments = wednesday_payments().copy()
    for column in LEDGER_TEXT_COLUMNS + ["key", "customer_id"]:
        payments[column] = payments[column].astype("str")
    return payments


def test_str_dtype_text_columns_give_object_columns_with_none_and_typed_amounts_and_dates():
    payments = str_dtype_wednesday_payments()
    # The precondition, so the test is not vacuous: the text columns really are `str`, and
    # an empty one (invoice_reference of the unapplied cash) holds a missing value, not None.
    assert payments["invoice_reference"].dtype != object
    assert pd.api.types.is_string_dtype(payments["invoice_reference"])
    assert payments["invoice_reference"].isna().sum() == 1

    lines = payment_lines_from_ledger(payments, pd.Index(COLLECTION_COLUMNS))

    assert lines["amount_try"].dtype == "float64"
    assert lines["amount_original"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(lines["document_date"])
    text_columns = [column for column in lines.columns
                    if not pd.api.types.is_numeric_dtype(lines[column])
                    and not pd.api.types.is_datetime64_any_dtype(lines[column])]
    assert set(RESULT_TEXT_COLUMNS) <= set(text_columns)
    for column in text_columns:
        assert lines[column].dtype == object, column
        values = lines[column].tolist()
        assert all(value is None or isinstance(value, str) for value in values), column
    for column in ("customer_name", "bank", "invoice_no"):
        assert all(value is None for value in lines[column]), column
    assert lines["sap_document_no"].tolist() == ["1400000104", None, "1400000106"]
    assert lines["customer"].tolist() == ["0000100002", "0000100004", "0000100003"]
    assert lines["currency_original"].tolist() == ["TRY", "USD", "TRY"]


# --- weekly_amounts on the filled lines ------------------------------------------------

def weekly_of_filled():
    result, _ = fill()
    with_weeks = add_week_index(result, WEEK_ZERO)
    return with_weeks, weekly_amounts(with_weeks)


def test_weekly_amounts_runs_on_the_filled_lines_and_totals_tie():
    with_weeks, weekly = weekly_of_filled()
    check_weekly_totals(with_weeks, weekly)  # does not raise
    assert weekly["lines"].sum() == len(with_weeks)
    total = weekly["gross_receipts"].sum() + weekly["refunds"].sum()
    assert total == pytest.approx(with_weeks["amount_try"].sum())


def test_a_filled_unapplied_payment_counts_as_an_unapplied_line():
    _, weekly = weekly_of_filled()
    tr05 = weekly[weekly["key"] == "TR05-100004"]
    assert len(tr05) == 1
    assert tr05["unapplied_lines"].iloc[0] == 1
    assert tr05["gross_receipts"].iloc[0] == pytest.approx(40.0)


def test_a_filled_refund_counts_as_a_refund():
    _, weekly = weekly_of_filled()
    customer = weekly[(weekly["key"] == "TR02-100003") & (weekly["week"] == 0)]
    assert customer["refunds"].iloc[0] == pytest.approx(-25.0)
    assert customer["gross_receipts"].iloc[0] == pytest.approx(20.0)  # Tuesday's daily line
    assert customer["unapplied_lines"].iloc[0] == 1  # only Tuesday's (no SAP number)


def test_week_zero_gross_receipts_include_the_filled_day():
    _, weekly = weekly_of_filled()
    week0 = weekly[weekly["week"] == 0]
    # Daily files 100 + 30 + 20 + 200, filled 300 + 40; refund -25.
    assert week0["gross_receipts"].sum() == pytest.approx(690.0)
    assert week0["refunds"].sum() == pytest.approx(-25.0)
    # Daily: one unapplied line (Tuesday); filled: one (the USD cash).
    assert week0["unapplied_lines"].sum() == 2


# --- integration with the synthetic corpus and ledger (SLOW) ---------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
AS_OF = date(2026, 8, 28)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]
SYNTHETIC_WEEK_ZERO = date(2025, 6, 2)
EXPECTED_FILLED_DAYS = [date(2025, 9, 8), date(2025, 12, 17), date(2026, 2, 4),
                        date(2026, 3, 13), date(2026, 4, 20), date(2026, 6, 1)]


@pytest.fixture(scope="module")
def synthetic_filled():
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    from collection_estimation.hurdle.keys import add_keys
    from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
    from collection_estimation.hurdle.read_ledger import read_ledger
    from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

    daily_files = list_daily_files(CORPUS_FOLDER, SYNTHETIC_LAYOUT.file_name_date_format)
    # Direct reads, never the data/interim cache.
    collections = add_keys(read_all_files(daily_files, SYNTHETIC_LAYOUT), KNOWN_COMPANIES)
    ledger = add_keys(read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT),
                      KNOWN_COMPANIES)
    result, filled_days = fill_missing_days(collections, ledger, list(daily_files))
    return collections, ledger, result, filled_days


def test_slow_synthetic_six_filled_days(synthetic_filled):
    _, _, _, filled_days = synthetic_filled
    assert filled_days == EXPECTED_FILLED_DAYS


def test_slow_synthetic_726_filled_lines(synthetic_filled):
    _, _, result, _ = synthetic_filled
    assert result["filled_from_ledger"].sum() == 726
    assert (result["source_file"] == FROM_LEDGER).sum() == 726


def test_slow_synthetic_filled_amount(synthetic_filled):
    _, _, result, _ = synthetic_filled
    filled = result[result["filled_from_ledger"]]
    assert filled["amount_try"].sum() == pytest.approx(206_531_569.50, abs=0.01)


def test_slow_synthetic_total_lines(synthetic_filled):
    collections, _, result, _ = synthetic_filled
    assert len(collections) == 27_311
    assert len(result) == 27_311 + 726


def test_slow_synthetic_filled_lines_sit_on_the_filled_days_only(synthetic_filled):
    _, _, result, filled_days = synthetic_filled
    filled = result[result["filled_from_ledger"]]
    assert sorted(set(filled["document_date"].dt.date)) == filled_days


def test_slow_synthetic_result_is_sorted_with_plain_index(synthetic_filled):
    _, _, result, _ = synthetic_filled
    assert result["document_date"].is_monotonic_increasing
    assert result.index.tolist() == list(range(len(result)))


def test_slow_synthetic_originals_kept(synthetic_filled):
    collections, _, result, _ = synthetic_filled
    originals = result[~result["filled_from_ledger"]].drop(columns="filled_from_ledger")
    expected = collections.sort_values("document_date", kind="stable", ignore_index=True)
    pd.testing.assert_frame_equal(originals.reset_index(drop=True), expected)


def test_slow_synthetic_weekly_amounts(synthetic_filled):
    _, _, result, _ = synthetic_filled
    with_weeks = add_week_index(result, SYNTHETIC_WEEK_ZERO)
    weekly = weekly_amounts(with_weeks)
    check_weekly_totals(with_weeks, weekly)
    assert len(weekly) == 20_947
    assert weekly["gross_receipts"].sum() == pytest.approx(8_885_177_992.23, abs=0.01)
    assert weekly["refunds"].sum() == pytest.approx(-31_043_453.84, abs=0.01)
    assert weekly["unapplied_lines"].sum() == 842


def test_slow_synthetic_text_columns_hold_none_not_nan(synthetic_filled):
    _, _, result, _ = synthetic_filled
    for column in RESULT_TEXT_COLUMNS:
        assert not any(isinstance(value, float) and np.isnan(value)
                       for value in result[column]), column
