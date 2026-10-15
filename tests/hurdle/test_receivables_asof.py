"""Tests for the exchange-rate units of `hurdle/receivables_asof.py`.

`exchange_rates` turns every foreign-currency line of the daily files and the ledger into a
rate (TRY amount / own amount) and takes the median per (day, currency); `rate_lines` makes
those per-line rates for one table; `rates_known_at` picks each currency's last rate on or
before a cutoff. Inputs are small tables built by hand with `collections_table` and
`ledger_table`. The SLOW tests at the end read the synthetic corpus and ledger in `data/raw/`
(once each) and are skipped when absent. The repo's `data/interim/` cache is never used.
"""
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
from collection_estimation.hurdle.read_ledger import read_ledger
from collection_estimation.hurdle.receivables_asof import (
    LOCAL_CURRENCY,
    exchange_rates,
    first_rows,
    normalise_currency,
    rate_lines,
    rates_known_at,
)
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

D1, D2, D3 = date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)
COLUMNS = ["day", "currency", "rate"]


def collections_table(lines, source_rows=None):
    """lines: (document day, currency, amount_original, amount_try) tuples."""
    if source_rows is None:
        source_rows = list(range(2, 2 + len(lines)))
    return pd.DataFrame({
        "document_date": pd.to_datetime(pd.Series([l[0] for l in lines], dtype=object)),
        "currency_original": pd.Series([l[1] for l in lines], dtype=object),
        "amount_original": pd.Series([l[2] for l in lines], dtype="float64"),
        "amount_try": pd.Series([l[3] for l in lines], dtype="float64"),
        "source_row": pd.Series(source_rows, dtype="int64"),
    })


def ledger_table(lines, source_rows=None):
    """lines: (posting day, currency, amount_document, amount_local) tuples."""
    if source_rows is None:
        source_rows = list(range(2, 2 + len(lines)))
    return pd.DataFrame({
        "posting_date": pd.to_datetime(pd.Series([l[0] for l in lines], dtype=object)),
        "document_currency": pd.Series([l[1] for l in lines], dtype=object),
        "amount_document": pd.Series([l[2] for l in lines], dtype="float64"),
        "amount_local": pd.Series([l[3] for l in lines], dtype="float64"),
        "source_row": pd.Series(source_rows, dtype="int64"),
    })


def rates_of(result):
    """{(day, currency): rate} for easy comparison."""
    return {(row.day, row.currency): row.rate for row in result.itertuples()}


def rates_table(rows):
    """rows: (day, currency, rate) triples, in the given order."""
    return pd.DataFrame({
        "day": pd.Series([r[0] for r in rows], dtype=object),
        "currency": pd.Series([r[1] for r in rows], dtype=object),
        "rate": pd.Series([r[2] for r in rows], dtype="float64"),
    })


# --- constant ------------------------------------------------------------------------------

def test_local_currency_is_try():
    assert LOCAL_CURRENCY == "TRY"


# --- one line, one rate --------------------------------------------------------------------

def test_foreign_daily_line_rate_is_try_over_own_amount():
    result = exchange_rates(collections_table([(D1, "EUR", 100.0, 3550.0)]), ledger_table([]))
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.5)}


def test_foreign_ledger_line_rate_is_local_over_document_amount():
    result = exchange_rates(collections_table([]), ledger_table([(D1, "USD", 200.0, 6400.0)]))
    assert rates_of(result) == {(D1, "USD"): pytest.approx(32.0)}


def test_day_is_a_date_object_not_a_timestamp():
    result = exchange_rates(collections_table([(D1, "EUR", 1.0, 35.0)]),
                            ledger_table([(D2, "EUR", 1.0, 36.0)]))
    for day in result["day"]:
        assert type(day) is date


def test_daily_line_uses_document_date_with_time_dropped():
    collections = collections_table([(datetime(2026, 3, 2, 15, 30), "EUR", 1.0, 35.0)])
    result = exchange_rates(collections, ledger_table([]))
    assert list(result["day"]) == [D1]


def test_ledger_line_uses_posting_date_with_time_dropped():
    ledger = ledger_table([(datetime(2026, 3, 3, 23, 59), "EUR", 1.0, 35.0)])
    result = exchange_rates(collections_table([]), ledger)
    assert list(result["day"]) == [D2]


def test_ledger_uses_posting_date_not_another_date_column():
    ledger = ledger_table([(D2, "EUR", 1.0, 35.0)])
    ledger["document_date"] = pd.to_datetime(pd.Series([D1], dtype=object))
    result = exchange_rates(collections_table([]), ledger)
    assert list(result["day"]) == [D2]


def test_currency_is_kept_as_text():
    result = exchange_rates(collections_table([(D1, "GBP", 2.0, 80.0)]), ledger_table([]))
    assert list(result["currency"]) == ["GBP"]
    assert all(isinstance(c, str) for c in result["currency"])


def test_rate_column_is_float():
    result = exchange_rates(collections_table([(D1, "EUR", 2.0, 71.0)]), ledger_table([]))
    assert result["rate"].dtype == "float64"


def test_try_lines_give_no_rate_in_either_source():
    result = exchange_rates(collections_table([(D1, "TRY", 100.0, 100.0)]),
                            ledger_table([(D1, "TRY", 50.0, 50.0)]))
    assert len(result) == 0
    assert list(result.columns) == COLUMNS


def test_try_lines_are_dropped_among_foreign_ones():
    result = exchange_rates(collections_table([(D1, "TRY", 100.0, 100.0),
                                               (D1, "EUR", 1.0, 35.0)]),
                            ledger_table([(D1, "TRY", 7.0, 7.0)]))
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.0)}


def test_zero_own_amount_daily_line_is_skipped():
    result = exchange_rates(collections_table([(D1, "EUR", 0.0, 0.0),
                                               (D1, "EUR", 2.0, 70.0)]), ledger_table([]))
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.0)}


def test_zero_own_amount_ledger_line_is_skipped():
    result = exchange_rates(collections_table([]), ledger_table([(D1, "USD", 0.0, 0.0),
                                                                 (D1, "USD", 1.0, 32.0)]))
    assert rates_of(result) == {(D1, "USD"): pytest.approx(32.0)}


def test_zero_own_amount_with_nonzero_try_is_skipped_not_infinite():
    result = exchange_rates(collections_table([(D1, "EUR", 0.0, 5.0)]),
                            ledger_table([(D1, "EUR", 0.0, -5.0)]))
    assert len(result) == 0


def test_only_zero_amount_lines_give_an_empty_table():
    result = exchange_rates(collections_table([(D1, "EUR", 0.0, 0.0)]),
                            ledger_table([(D1, "USD", 0.0, 0.0)]))
    assert list(result.columns) == COLUMNS
    assert len(result) == 0


# --- median per day and currency -----------------------------------------------------------

def test_day_rate_is_median_of_three_lines_across_both_sources():
    collections = collections_table([(D1, "EUR", 1.0, 35.00), (D1, "EUR", 1.0, 35.20)])
    ledger = ledger_table([(D1, "EUR", 1.0, 35.10)])
    result = exchange_rates(collections, ledger)
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.10)}


def test_median_is_not_the_mean():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D1, "EUR", 1.0, 35.1)])
    ledger = ledger_table([(D1, "EUR", 1.0, 99.0)])
    result = exchange_rates(collections, ledger)
    assert rates_of(result)[(D1, "EUR")] == pytest.approx(35.1)


def test_median_of_two_lines_is_their_midpoint():
    result = exchange_rates(collections_table([(D1, "USD", 1.0, 32.0)]),
                            ledger_table([(D1, "USD", 1.0, 32.2)]))
    assert rates_of(result)[(D1, "USD")] == pytest.approx(32.1)


def test_median_uses_per_line_rates_not_summed_amounts():
    # Summed amounts would give (100*35 + 1*40) / 101 = 35.05; the median of rates is 37.5.
    result = exchange_rates(collections_table([(D1, "EUR", 100.0, 3500.0)]),
                            ledger_table([(D1, "EUR", 1.0, 40.0)]))
    assert rates_of(result)[(D1, "EUR")] == pytest.approx(37.5)


def test_different_currencies_on_the_same_day_stay_separate():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D1, "USD", 1.0, 32.0)])
    ledger = ledger_table([(D1, "GBP", 1.0, 41.0), (D1, "USD", 1.0, 32.2)])
    result = exchange_rates(collections, ledger)
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.0),
                                (D1, "GBP"): pytest.approx(41.0),
                                (D1, "USD"): pytest.approx(32.1)}


def test_same_currency_on_different_days_stays_separate():
    result = exchange_rates(collections_table([(D1, "EUR", 1.0, 35.0), (D2, "EUR", 1.0, 36.0)]),
                            ledger_table([]))
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.0),
                                (D2, "EUR"): pytest.approx(36.0)}


def test_one_row_per_day_and_currency():
    collections = collections_table([(D1, "EUR", 1.0, 35.0)] * 4)
    ledger = ledger_table([(D1, "EUR", 2.0, 70.0)] * 3)
    result = exchange_rates(collections, ledger)
    assert len(result) == 1
    assert not result.duplicated(["day", "currency"]).any()


def test_refund_with_both_amounts_negative_gives_a_positive_rate():
    result = exchange_rates(collections_table([(D1, "EUR", -10.0, -355.0)]),
                            ledger_table([(D2, "USD", -5.0, -160.0)]))
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.5),
                                (D2, "USD"): pytest.approx(32.0)}


def test_refund_and_payment_pool_into_one_median():
    result = exchange_rates(collections_table([(D1, "EUR", -10.0, -350.0),
                                               (D1, "EUR", 10.0, 352.0)]),
                            ledger_table([(D1, "EUR", 1.0, 35.4)]))
    assert rates_of(result)[(D1, "EUR")] == pytest.approx(35.2)


# --- shape of the result -------------------------------------------------------------------

def test_columns_are_exactly_day_currency_rate():
    result = exchange_rates(collections_table([(D1, "EUR", 1.0, 35.0)]), ledger_table([]))
    assert list(result.columns) == COLUMNS


def test_sorted_by_currency_then_day_with_a_fresh_index():
    collections = collections_table([(D3, "USD", 1.0, 33.0), (D1, "USD", 1.0, 32.0),
                                     (D2, "EUR", 1.0, 36.0)])
    ledger = ledger_table([(D3, "EUR", 1.0, 37.0), (D1, "GBP", 1.0, 41.0),
                           (D1, "EUR", 1.0, 35.0)])
    result = exchange_rates(collections, ledger)
    assert list(zip(result["currency"], result["day"])) == [
        ("EUR", D1), ("EUR", D2), ("EUR", D3), ("GBP", D1), ("USD", D1), ("USD", D3)]
    assert list(result.index) == list(range(len(result)))


def test_index_is_zero_based_even_when_inputs_have_an_odd_index():
    collections = collections_table([(D2, "EUR", 1.0, 36.0), (D1, "EUR", 1.0, 35.0)])
    collections.index = [10, 5]
    ledger = ledger_table([(D1, "USD", 1.0, 32.0)])
    ledger.index = [99]
    result = exchange_rates(collections, ledger)
    assert list(result.index) == [0, 1, 2]


def test_both_inputs_empty_give_an_empty_table_with_the_columns():
    result = exchange_rates(collections_table([]), ledger_table([]))
    assert list(result.columns) == COLUMNS
    assert len(result) == 0


def test_only_try_lines_give_an_empty_table_with_the_columns():
    result = exchange_rates(collections_table([(D1, "TRY", 5.0, 5.0)]),
                            ledger_table([(D2, "TRY", 9.0, 9.0)]))
    assert list(result.columns) == COLUMNS
    assert len(result) == 0


def test_empty_daily_files_with_a_foreign_ledger_still_give_rates():
    result = exchange_rates(collections_table([]), ledger_table([(D1, "EUR", 1.0, 35.0)]))
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.0)}


def test_inputs_are_not_modified():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D1, "TRY", 3.0, 3.0)])
    ledger = ledger_table([(D2, "USD", 0.0, 0.0), (D2, "USD", 2.0, 64.0)])
    before = collections.copy(deep=True), ledger.copy(deep=True)
    exchange_rates(collections, ledger)
    pd.testing.assert_frame_equal(collections, before[0], check_exact=True)
    pd.testing.assert_frame_equal(ledger, before[1], check_exact=True)


def test_same_inputs_give_the_same_result_twice():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D2, "USD", 1.0, 32.0)])
    ledger = ledger_table([(D1, "EUR", 1.0, 35.2)])
    pd.testing.assert_frame_equal(exchange_rates(collections, ledger),
                                  exchange_rates(collections, ledger), check_exact=True)


# --- opposite signs ------------------------------------------------------------------------

def test_opposite_signs_in_the_daily_files_raise_naming_the_source():
    collections = collections_table([(D1, "EUR", 10.0, -350.0)], source_rows=[17])
    with pytest.raises(ValueError, match="the daily files") as error:
        exchange_rates(collections, ledger_table([]))
    assert "opposite signs" in str(error.value)
    assert "17" in str(error.value)


def test_opposite_signs_in_the_ledger_raise_naming_the_source():
    ledger = ledger_table([(D1, "USD", -5.0, 160.0)], source_rows=[4242])
    with pytest.raises(ValueError, match="the ledger") as error:
        exchange_rates(collections_table([]), ledger)
    assert "4242" in str(error.value)
    assert "the daily files" not in str(error.value)


def test_good_ledger_does_not_hide_bad_daily_files():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D1, "EUR", 1.0, -35.0)],
                                    source_rows=[2, 3])
    with pytest.raises(ValueError, match="the daily files"):
        exchange_rates(collections, ledger_table([(D1, "EUR", 1.0, 35.0)]))


def test_opposite_signs_message_lists_only_the_bad_rows():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D1, "EUR", 1.0, -35.0),
                                     (D2, "USD", -1.0, 32.0)], source_rows=[101, 102, 103])
    with pytest.raises(ValueError) as error:
        exchange_rates(collections, ledger_table([]))
    message = str(error.value)
    assert "[102, 103]" in message
    assert "101" not in message


def test_opposite_signs_message_lists_the_first_five_rows_only():
    lines = [(D1, "EUR", 1.0, -35.0)] * 7
    collections = collections_table(lines, source_rows=[11, 12, 13, 14, 15, 16, 17])
    with pytest.raises(ValueError) as error:
        exchange_rates(collections, ledger_table([]))
    message = str(error.value)
    assert "[11, 12, 13, 14, 15]" in message
    assert "16" not in message and "17" not in message


def test_opposite_signs_on_a_try_line_do_not_raise():
    """A TRY line is never turned into a rate, so its signs are not checked here."""
    result = exchange_rates(collections_table([(D1, "TRY", 10.0, -10.0)]), ledger_table([]))
    assert len(result) == 0


def test_rate_lines_raises_with_the_given_source_name():
    table = ledger_table([(D1, "EUR", 1.0, -35.0)], source_rows=[9])
    with pytest.raises(ValueError, match="my source"):
        rate_lines(table, "posting_date", "document_currency", "amount_document",
                   "amount_local", "my source")


# --- rate_lines directly -------------------------------------------------------------------

def test_rate_lines_gives_one_row_per_foreign_nonzero_line_without_taking_medians():
    table = collections_table([(D1, "EUR", 1.0, 35.0), (D1, "EUR", 1.0, 35.2),
                               (D1, "TRY", 5.0, 5.0), (D1, "EUR", 0.0, 0.0)])
    lines = rate_lines(table, "document_date", "currency_original", "amount_original",
                       "amount_try", "the daily files")
    assert list(lines.columns) == COLUMNS
    assert len(lines) == 2
    assert sorted(lines["rate"]) == pytest.approx([35.0, 35.2])
    assert set(lines["day"]) == {D1}


def test_rate_lines_on_an_empty_table_gives_the_columns():
    lines = rate_lines(collections_table([]), "document_date", "currency_original",
                       "amount_original", "amount_try", "the daily files")
    assert list(lines.columns) == COLUMNS
    assert len(lines) == 0


# --- the ledger extends the range ----------------------------------------------------------

def test_currency_only_in_the_ledger_gets_rates():
    result = exchange_rates(collections_table([(D2, "EUR", 1.0, 35.0)]),
                            ledger_table([(D2, "CHF", 1.0, 38.0), (D3, "CHF", 1.0, 38.5)]))
    chf = result[result["currency"] == "CHF"]
    assert list(chf["day"]) == [D2, D3]
    assert list(chf["rate"]) == pytest.approx([38.0, 38.5])


def test_date_only_in_the_ledger_before_the_daily_files_start_gets_a_rate():
    early = date(2024, 10, 25)
    result = exchange_rates(collections_table([(D2, "EUR", 1.0, 35.0)]),
                            ledger_table([(early, "EUR", 1.0, 30.0)]))
    assert rates_of(result) == {(early, "EUR"): pytest.approx(30.0),
                                (D2, "EUR"): pytest.approx(35.0)}


def test_day_only_in_the_daily_files_gets_a_rate():
    result = exchange_rates(collections_table([(D3, "EUR", 1.0, 36.0)]),
                            ledger_table([(D1, "EUR", 1.0, 35.0)]))
    assert set(result["day"]) == {D1, D3}


# --- rates_known_at ------------------------------------------------------------------------

def test_known_at_uses_the_last_day_on_or_before_the_cutoff():
    rates = rates_table([(D1, "EUR", 35.0), (D2, "EUR", 36.0), (D3, "EUR", 37.0)])
    assert rates_known_at(rates, D2) == {"EUR": 36.0, "TRY": 1.0}


def test_known_at_a_day_equal_to_the_cutoff_counts():
    rates = rates_table([(D1, "EUR", 35.0), (D3, "EUR", 37.0)])
    assert rates_known_at(rates, D3)["EUR"] == 37.0


def test_known_at_a_later_day_is_ignored():
    rates = rates_table([(D1, "EUR", 35.0), (D3, "EUR", 37.0)])
    assert rates_known_at(rates, D2)["EUR"] == 35.0


def test_known_at_a_gap_carries_the_last_rate_forward():
    rates = rates_table([(D1, "USD", 32.0)])
    assert rates_known_at(rates, date(2026, 12, 31))["USD"] == 32.0


def test_known_at_each_currency_uses_its_own_last_day():
    rates = rates_table([(D1, "EUR", 35.0), (D3, "EUR", 37.0),
                         (D2, "USD", 32.5), (D1, "USD", 32.0)])
    assert rates_known_at(rates, D3) == {"EUR": 37.0, "USD": 32.5, "TRY": 1.0}


def test_known_at_try_is_one_with_no_rates_at_all():
    assert rates_known_at(rates_table([]), D1) == {"TRY": 1.0}


def test_known_at_try_is_one_when_every_rate_is_later():
    rates = rates_table([(D3, "EUR", 37.0)])
    assert rates_known_at(rates, D1) == {"TRY": 1.0}


def test_known_at_currency_with_no_rate_yet_is_absent():
    rates = rates_table([(D1, "EUR", 35.0), (D3, "GBP", 41.0)])
    result = rates_known_at(rates, D2)
    assert "GBP" not in result
    assert set(result) == {"EUR", "TRY"}


def test_known_at_try_is_one_even_if_rates_hold_a_try_row():
    rates = rates_table([(D1, "TRY", 2.0), (D1, "EUR", 35.0)])
    assert rates_known_at(rates, D2)["TRY"] == 1.0


def test_known_at_values_are_python_floats_and_keys_str():
    rates = exchange_rates(collections_table([(D1, "EUR", 1.0, 35.0)]),
                           ledger_table([(D1, "USD", 1.0, 32.0)]))
    result = rates_known_at(rates, D2)
    for currency, rate in result.items():
        assert type(currency) is str
        assert type(rate) is float


def test_known_at_works_when_rates_are_not_sorted():
    rates = rates_table([(D3, "EUR", 37.0), (D1, "EUR", 35.0), (D2, "EUR", 36.0),
                         (date(2026, 3, 10), "EUR", 40.0)])
    assert rates_known_at(rates, D3)["EUR"] == 37.0
    assert rates_known_at(rates, date(2026, 3, 9))["EUR"] == 37.0
    assert rates_known_at(rates, D1)["EUR"] == 35.0


def test_known_at_does_not_modify_the_input():
    rates = rates_table([(D3, "EUR", 37.0), (D1, "EUR", 35.0), (D2, "USD", 32.0)])
    before = rates.copy(deep=True)
    rates_known_at(rates, D2)
    pd.testing.assert_frame_equal(rates, before, check_exact=True)


def test_known_at_round_trip_with_exchange_rates():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D3, "EUR", 1.0, 37.0)])
    ledger = ledger_table([(D2, "EUR", 1.0, 36.0), (D2, "USD", 2.0, 64.4)])
    rates = exchange_rates(collections, ledger)
    assert rates_known_at(rates, D2) == {"EUR": pytest.approx(36.0),
                                         "USD": pytest.approx(32.2), "TRY": 1.0}


# --- currency codes are trimmed and upper-cased --------------------------------------------

def daily_rate_lines(table):
    return rate_lines(table, "document_date", "currency_original", "amount_original",
                      "amount_try", "the daily files")


def test_padded_lower_case_currency_comes_out_upper_case_and_trimmed():
    result = exchange_rates(collections_table([(D1, " eur ", 1.0, 35.0)]), ledger_table([]))
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.0)}


def test_rate_lines_output_currency_is_normalised():
    lines = daily_rate_lines(collections_table([(D1, " usd", 1.0, 32.0), (D1, "Gbp ", 1.0, 41.0)]))
    assert list(lines["currency"]) == ["USD", "GBP"]


@pytest.mark.parametrize("local", ["try", " TRY ", "Try", "TRY\t"])
def test_try_in_any_case_or_padding_is_local_and_gives_no_rate(local):
    result = exchange_rates(collections_table([(D1, local, 100.0, 100.0)]),
                            ledger_table([(D1, local, 5.0, 5.0)]))
    assert len(result) == 0


def test_lower_case_try_with_opposite_signs_does_not_raise():
    """Were 'try' taken as foreign, this line would raise for opposite signs."""
    result = exchange_rates(collections_table([(D1, " try", 10.0, -10.0)]), ledger_table([]))
    assert len(result) == 0


def test_eur_spellings_of_one_day_pool_into_one_median():
    collections = collections_table([(D1, "eur", 1.0, 35.0), (D1, "EUR", 1.0, 35.2)])
    ledger = ledger_table([(D1, " Eur ", 1.0, 35.1)])
    result = exchange_rates(collections, ledger)
    assert rates_of(result) == {(D1, "EUR"): pytest.approx(35.1)}
    assert len(result) == 1


def test_rate_lines_does_not_modify_the_currency_column():
    table = collections_table([(D1, " eur ", 1.0, 35.0)])
    before = table.copy(deep=True)
    daily_rate_lines(table)
    pd.testing.assert_frame_equal(table, before, check_exact=True)


# --- a line with no currency ---------------------------------------------------------------

@pytest.mark.parametrize("empty", [None, float("nan"), "", "  "])
def test_line_with_no_currency_raises(empty):
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D1, empty, 1.0, 35.0)],
                                    source_rows=[5, 6])
    with pytest.raises(ValueError) as error:
        exchange_rates(collections, ledger_table([]))
    message = str(error.value)
    assert message == "the daily files: 1 lines have no currency (first rows: [6])"


def test_no_currency_in_the_ledger_names_the_ledger():
    ledger = ledger_table([(D1, None, 1.0, 35.0)], source_rows=[77])
    with pytest.raises(ValueError, match=r"^the ledger: 1 lines have no currency") as error:
        exchange_rates(collections_table([]), ledger)
    assert "[77]" in str(error.value)


def test_no_currency_is_checked_on_lines_that_look_like_try():
    """Equal own and TRY amounts would be a TRY line, but a missing code is still an error."""
    with pytest.raises(ValueError, match="no currency"):
        daily_rate_lines(collections_table([(D1, None, 100.0, 100.0)]))


def test_no_currency_is_checked_on_zero_amount_lines():
    """A zero own amount would skip the line as a rate, but not the currency check."""
    with pytest.raises(ValueError, match="no currency"):
        daily_rate_lines(collections_table([(D1, "", 0.0, 0.0)]))


def test_no_currency_is_checked_before_amount_problems():
    collections = collections_table([(D1, "EUR", 1.0, -35.0), (D1, "EUR", float("nan"), 35.0),
                                     (D1, "  ", 1.0, 35.0)], source_rows=[2, 3, 4])
    with pytest.raises(ValueError) as error:
        daily_rate_lines(collections)
    message = str(error.value)
    assert "no currency" in message
    assert "first rows: [4]" in message
    assert "opposite signs" not in message and "missing amount" not in message


def test_no_currency_message_counts_all_and_lists_first_five():
    lines = [(D1, None, 1.0, 35.0)] * 3 + [(D1, "EUR", 1.0, 35.0)] + [(D1, "", 1.0, 35.0)] * 4
    collections = collections_table(lines, source_rows=[10, 11, 12, 13, 14, 15, 16, 17])
    with pytest.raises(ValueError) as error:
        daily_rate_lines(collections)
    assert str(error.value) == ("the daily files: 7 lines have no currency "
                                "(first rows: [10, 11, 12, 14, 15])")


# --- a foreign line with a missing amount --------------------------------------------------

def test_missing_try_amount_on_a_foreign_line_raises():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D1, "EUR", 2.0, None)],
                                    source_rows=[20, 21])
    with pytest.raises(ValueError) as error:
        daily_rate_lines(collections)
    assert str(error.value) == ("the daily files: 1 foreign-currency lines have a missing "
                                "amount (first rows: [21])")


def test_missing_own_amount_on_a_foreign_line_raises():
    ledger = ledger_table([(D1, "USD", None, 32.0)], source_rows=[8])
    with pytest.raises(ValueError) as error:
        exchange_rates(collections_table([]), ledger)
    assert str(error.value) == ("the ledger: 1 foreign-currency lines have a missing "
                                "amount (first rows: [8])")


def test_both_amounts_missing_on_a_foreign_line_raises_missing():
    with pytest.raises(ValueError, match="have a missing amount"):
        daily_rate_lines(collections_table([(D1, "EUR", None, None)]))


def test_missing_amounts_on_a_try_line_do_not_raise():
    """Only foreign lines are turned into rates, so a TRY line's amounts are not checked."""
    lines = daily_rate_lines(collections_table([(D1, "TRY", None, 5.0), (D1, "try", 5.0, None)]))
    assert len(lines) == 0


def test_missing_amount_lists_the_first_five_rows_and_counts_all():
    lines = [(D1, "EUR", 1.0, None)] * 6
    collections = collections_table(lines, source_rows=[31, 32, 33, 34, 35, 36])
    with pytest.raises(ValueError) as error:
        daily_rate_lines(collections)
    message = str(error.value)
    assert "6 foreign-currency lines have a missing amount" in message
    assert "[31, 32, 33, 34, 35]" in message
    assert "36" not in message


# --- a foreign line with a TRY amount of zero ----------------------------------------------

def test_zero_try_amount_with_nonzero_own_amount_raises_its_own_message():
    collections = collections_table([(D1, "EUR", 10.0, 0.0)], source_rows=[40])
    with pytest.raises(ValueError) as error:
        exchange_rates(collections, ledger_table([]))
    assert str(error.value) == ("the daily files: 1 foreign-currency lines have a TRY amount "
                                "of zero but a non-zero amount in their own currency "
                                "(first rows: [40])")


def test_zero_try_amount_with_negative_own_amount_is_zero_not_opposite_signs():
    """0 / -10 is -0.0: it must report as zero, not as opposite signs."""
    ledger = ledger_table([(D1, "USD", -10.0, 0.0)], source_rows=[41])
    with pytest.raises(ValueError) as error:
        exchange_rates(collections_table([]), ledger)
    message = str(error.value)
    assert "have a TRY amount of zero" in message
    assert "opposite signs" not in message
    assert "[41]" in message


def test_zero_on_both_sides_is_still_skipped_not_an_error():
    lines = daily_rate_lines(collections_table([(D1, "EUR", 0.0, 0.0), (D1, "EUR", 1.0, 35.0)]))
    assert list(lines["rate"]) == pytest.approx([35.0])


def test_zero_try_amount_on_a_try_line_does_not_raise():
    assert len(daily_rate_lines(collections_table([(D1, "TRY", 10.0, 0.0)]))) == 0


# --- which problem is reported first -------------------------------------------------------

def test_missing_wins_over_zero_and_opposite_signs():
    collections = collections_table([(D1, "EUR", 1.0, -35.0), (D1, "EUR", 1.0, 0.0),
                                     (D1, "EUR", 1.0, None)], source_rows=[2, 3, 4])
    with pytest.raises(ValueError) as error:
        daily_rate_lines(collections)
    assert str(error.value) == ("the daily files: 1 foreign-currency lines have a missing "
                                "amount (first rows: [4])")


def test_zero_wins_over_opposite_signs():
    collections = collections_table([(D1, "EUR", 1.0, -35.0), (D1, "EUR", 1.0, -36.0),
                                     (D1, "USD", 1.0, 0.0)], source_rows=[2, 3, 4])
    with pytest.raises(ValueError) as error:
        daily_rate_lines(collections)
    message = str(error.value)
    assert "1 foreign-currency lines have a TRY amount of zero" in message
    assert "first rows: [4]" in message


def test_opposite_signs_message_is_exact_when_alone():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D1, "eur", 1.0, -35.0),
                                     (D1, "EUR", -2.0, 70.0)], source_rows=[2, 3, 4])
    with pytest.raises(ValueError) as error:
        daily_rate_lines(collections)
    assert str(error.value) == ("the daily files: 2 foreign-currency lines have TRY and "
                                "own-currency amounts of opposite signs (first rows: [3, 4])")


def test_bad_daily_files_are_reported_before_the_ledger_is_read():
    collections = collections_table([(D1, "EUR", 1.0, 0.0)])
    ledger = ledger_table([(D1, None, 1.0, 35.0)])
    with pytest.raises(ValueError, match="^the daily files: .*TRY amount of zero"):
        exchange_rates(collections, ledger)


# --- normalise_currency directly -----------------------------------------------------------

@pytest.mark.parametrize("empty", [None, float("nan"), pd.NA, "", "  ", "\t\n"])
def test_normalise_currency_empty_is_none(empty):
    assert normalise_currency(empty) is None


@pytest.mark.parametrize("raw, expected", [(" gbp ", "GBP"), ("EUR", "EUR"), ("usd", "USD"),
                                           ("Try", "TRY"), ("\tchf\n", "CHF")])
def test_normalise_currency_trims_and_upper_cases(raw, expected):
    assert normalise_currency(raw) == expected


def test_normalise_currency_returns_str():
    assert type(normalise_currency(" eur ")) is str


# --- first_rows directly -------------------------------------------------------------------

def test_first_rows_gives_source_rows_of_flagged_lines_in_table_order():
    table = collections_table([(D1, "EUR", 1.0, 1.0)] * 4, source_rows=[90, 50, 70, 60])
    flags = pd.Series([False, True, True, True], index=table.index)
    assert first_rows(table, flags) == [50, 70, 60]


def test_first_rows_caps_at_five():
    table = collections_table([(D1, "EUR", 1.0, 1.0)] * 8, source_rows=list(range(100, 108)))
    flags = pd.Series([True] * 8, index=table.index)
    assert first_rows(table, flags) == [100, 101, 102, 103, 104]


def test_first_rows_none_flagged_is_empty():
    table = collections_table([(D1, "EUR", 1.0, 1.0)] * 2)
    assert first_rows(table, pd.Series([False, False], index=table.index)) == []


def test_first_rows_aligns_on_the_index_not_position():
    table = collections_table([(D1, "EUR", 1.0, 1.0)] * 3, source_rows=[7, 8, 9])
    table.index = [30, 10, 20]
    flags = pd.Series([True, False, False], index=[10, 20, 30])
    assert first_rows(table, flags) == [8]


def test_rate_lines_reports_source_rows_not_positions_with_an_odd_index():
    collections = collections_table([(D1, "EUR", 1.0, 35.0), (D1, "EUR", 1.0, 0.0)],
                                    source_rows=[500, 501])
    collections.index = [7, 3]
    with pytest.raises(ValueError, match=r"first rows: \[501\]"):
        daily_rate_lines(collections)


# --- integration with the synthetic ledger and corpus (SLOW) --------------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
AS_OF = date(2026, 8, 28)


@pytest.fixture(scope="module")
def synthetic():
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    ledger = read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT)  # direct, no cache
    daily_files = list_daily_files(CORPUS_FOLDER, SYNTHETIC_LAYOUT.file_name_date_format)
    collections = read_all_files(daily_files, SYNTHETIC_LAYOUT)  # direct read, no cache
    return collections, ledger, exchange_rates(collections, ledger)


def test_slow_synthetic_rates_cover_eur_gbp_usd(synthetic):
    _, _, rates = synthetic
    assert sorted(rates["currency"].unique()) == ["EUR", "GBP", "USD"]


def test_slow_synthetic_each_currency_runs_from_2024_10_25_to_2026_08_28(synthetic):
    _, _, rates = synthetic
    for currency, days in rates.groupby("currency")["day"]:
        assert (min(days), max(days)) == (date(2024, 10, 25), date(2026, 8, 28)), currency


def test_slow_synthetic_days_per_currency(synthetic):
    _, _, rates = synthetic
    assert rates.groupby("currency").size().to_dict() == {"EUR": 480, "GBP": 245, "USD": 470}


def test_slow_synthetic_all_rates_positive(synthetic):
    _, _, rates = synthetic
    assert (rates["rate"] > 0).all()
    assert rates["rate"].notna().all()


def test_slow_synthetic_daily_and_ledger_medians_agree_on_shared_days(synthetic):
    collections, ledger, _ = synthetic
    daily = rate_lines(collections, "document_date", "currency_original", "amount_original",
                       "amount_try", "the daily files").groupby(["day", "currency"])["rate"].median()
    from_ledger = rate_lines(ledger, "posting_date", "document_currency", "amount_document",
                             "amount_local", "the ledger").groupby(["day", "currency"])["rate"].median()
    shared = daily.index.intersection(from_ledger.index)
    assert len(shared) > 0
    relative = ((daily[shared] - from_ledger[shared]).abs() / from_ledger[shared]).max()
    assert relative <= 1e-4


def test_slow_synthetic_rates_known_at_2025_06_01_has_all_currencies(synthetic):
    _, _, rates = synthetic
    known = rates_known_at(rates, date(2025, 6, 1))
    assert set(known) == {"EUR", "GBP", "USD", "TRY"}
    assert known["TRY"] == 1.0
