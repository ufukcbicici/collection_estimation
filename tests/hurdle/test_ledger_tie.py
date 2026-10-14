"""Tests for `check_ledger_against_collections` in `hurdle/read_ledger.py`.

The ledger's payment lines (role 'payment', e.g. DZ, sign flipped) must tie, day by day, to the daily files' TRY
totals on every day that has a file. The inputs are small tables built by hand with
`ledger_table` and `collections_table`. Two SLOW tests at the end use the synthetic ledger and
corpus in `data/raw/` (each read once, ~30 s for the ledger) and are skipped when absent. The
repo's `data/interim/` cache is never used.
"""
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
from collection_estimation.hurdle.read_ledger import (
    check_ledger_against_collections,
    read_ledger,
)
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

DATE_FORMAT = SYNTHETIC_LAYOUT.file_name_date_format  # "%d%m%Y"

D1, D2, D3 = date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)

RESULT_KEYS = {"days_compared", "days_before_first_file", "try_before_first_file",
               "missing_file_days_in_ledger", "try_on_missing_file_days",
               "days_after_last_file", "try_after_last_file"}


# The role read_ledger gives each synthetic document type (see SYNTHETIC_LEDGER_LAYOUT).
ROLE_OF_TYPE = {code: role for role, codes in SYNTHETIC_LEDGER_LAYOUT.document_types.items()
                for code in codes}


def ledger_table(lines, roles=None):
    """lines: (document_type, posting day, amount_local) triples.

    `roles` gives each line's document_role; by default the synthetic role of its type
    (RV invoice, DZ payment, DG credit memo).
    """
    if roles is None:
        roles = [ROLE_OF_TYPE[t] for t, _, _ in lines]
    return pd.DataFrame({
        "document_type": pd.Series([t for t, _, _ in lines], dtype=object),
        "document_role": pd.Series(list(roles), dtype=object),
        "posting_date": pd.to_datetime(pd.Series([d for _, d, _ in lines], dtype=object)),
        "amount_local": pd.Series([a for _, _, a in lines], dtype="float64"),
    })


def collections_table(lines):
    """lines: (document day, amount_try) pairs."""
    return pd.DataFrame({
        "document_date": pd.to_datetime(pd.Series([d for d, _ in lines], dtype=object)),
        "amount_try": pd.Series([a for _, a in lines], dtype="float64"),
    })


def check(ledger_lines, collection_lines, days):
    return check_ledger_against_collections(ledger_table(ledger_lines),
                                            collections_table(collection_lines), days)


# --- exact tie -----------------------------------------------------------------------------

def test_exact_tie_returns_all_seven_keys():
    result = check([("DZ", D1, -100.0), ("DZ", D2, -250.5)],
                   [(D1, 100.0), (D2, 250.5)], [D1, D2])
    assert set(result) == RESULT_KEYS
    assert result == {"days_compared": 2, "days_before_first_file": 0,
                      "try_before_first_file": 0.0, "missing_file_days_in_ledger": 0,
                      "try_on_missing_file_days": 0.0, "days_after_last_file": 0,
                      "try_after_last_file": 0.0}


TRY_KEYS = ("try_before_first_file", "try_on_missing_file_days", "try_after_last_file")
COUNT_KEYS = ("days_compared", "days_before_first_file", "missing_file_days_in_ledger",
              "days_after_last_file")


def test_result_try_values_are_floats_and_counts_ints():
    result = check([("DZ", D1, -100.0)], [(D1, 100.0)], [D1])
    for key in TRY_KEYS:
        assert type(result[key]) is float
    for key in COUNT_KEYS:
        assert isinstance(result[key], int)


def test_result_types_hold_when_every_group_is_non_empty():
    result = check([("DZ", D1, -1.0), ("DZ", D2, -2.0), ("DZ", D3, -3.0),
                    ("DZ", date(2026, 3, 5), -4.0), ("DZ", date(2026, 3, 6), -5.0)],
                   [(D2, 2.0), (date(2026, 3, 5), 4.0)], [D2, date(2026, 3, 5)])
    for key in TRY_KEYS:
        assert type(result[key]) is float
    for key in COUNT_KEYS:
        assert isinstance(result[key], int)


def test_several_lines_per_day_are_summed_on_both_sides():
    result = check([("DZ", D1, -60.0), ("DZ", D1, -40.0), ("DZ", D1, -0.25)],
                   [(D1, 30.0), (D1, 70.25)], [D1])
    assert result["days_compared"] == 1


def test_lines_split_differently_but_same_day_total_tie():
    result = check([("DZ", D1, -100.0)], [(D1, 20.0), (D1, 30.0), (D1, 50.0)], [D1])
    assert result["days_compared"] == 1


def test_posting_date_with_a_time_of_day_counts_on_that_date():
    ledger = ledger_table([("DZ", datetime(2026, 3, 2, 15, 30), -100.0)])
    collections = collections_table([(D1, 100.0)])
    assert check_ledger_against_collections(ledger, collections, [D1])["days_compared"] == 1


# --- tolerance -----------------------------------------------------------------------------

def test_difference_of_half_a_cent_passes():
    assert check([("DZ", D1, -100.005)], [(D1, 100.0)], [D1])["days_compared"] == 1


def test_difference_of_exactly_one_cent_passes():
    assert check([("DZ", D1, -0.01)], [(D1, 0.0)], [D1])["days_compared"] == 1


def test_difference_just_over_one_cent_raises():
    with pytest.raises(ValueError, match="on 1 days"):
        check([("DZ", D1, -100.011)], [(D1, 100.0)], [D1])


# --- mismatches ----------------------------------------------------------------------------

def test_mismatch_on_one_day_raises_with_count_and_both_totals():
    with pytest.raises(ValueError) as error:
        check([("DZ", D1, -100.0), ("DZ", D2, -1234567.891)],
              [(D1, 100.0), (D2, 1234000.0)], [D1, D2])
    message = str(error.value)
    assert "differ from the daily files on 1 days" in message
    assert "2026-03-03" in message
    assert "1,234,000.00" in message
    assert "1,234,567.89" in message
    assert "2026-03-02" not in message


def test_mismatch_message_names_the_daily_and_ledger_side():
    with pytest.raises(ValueError) as error:
        check([("DZ", D1, -5.0)], [(D1, 7.0)], [D1])
    assert "2026-03-02: daily files 7.00 TRY, ledger 5.00 TRY" in str(error.value)


def seven_bad_days():
    return [date(2026, 3, day) for day in range(2, 9)]


def test_mismatch_on_seven_days_counts_seven_and_lists_the_first_five():
    days = seven_bad_days()
    with pytest.raises(ValueError) as error:
        check([("DZ", day, -1.0) for day in days], [(day, 2.0) for day in days], days)
    message = str(error.value)
    assert "on 7 days" in message
    for day in days[:5]:
        assert str(day) in message
    for day in days[5:]:
        assert str(day) not in message
    assert message.count("daily files 2.00 TRY, ledger 1.00 TRY") == 5


def test_first_five_listed_are_the_earliest_even_if_days_are_given_unsorted():
    days = seven_bad_days()
    shuffled = [days[6], days[2], days[5], days[0], days[4], days[1], days[3]]
    with pytest.raises(ValueError) as error:
        check([("DZ", day, -1.0) for day in days], [(day, 2.0) for day in days], shuffled)
    message = str(error.value)
    for day in days[:5]:
        assert str(day) in message
    for day in days[5:]:
        assert str(day) not in message


def test_good_days_among_bad_ones_are_not_counted_or_listed():
    with pytest.raises(ValueError) as error:
        check([("DZ", D1, -1.0), ("DZ", D2, -2.0), ("DZ", D3, -3.0)],
              [(D1, 1.0), (D2, 9.0), (D3, 3.0)], [D1, D2, D3])
    message = str(error.value)
    assert "on 1 days" in message
    assert "2026-03-03" in message
    assert "2026-03-02" not in message and "2026-03-04" not in message


# --- document types ------------------------------------------------------------------------

def test_invoices_and_credit_memos_are_ignored():
    result = check([("RV", D1, 5000.0), ("DG", D1, -300.0), ("DZ", D1, -100.0)],
                   [(D1, 100.0)], [D1])
    assert result["days_compared"] == 1


def test_rv_and_dg_on_days_without_a_file_are_not_counted_as_payment_days():
    result = check([("DZ", D2, -100.0), ("RV", D1, 5000.0), ("DG", D3, -300.0)],
                   [(D2, 100.0)], [D2])
    assert result["days_before_first_file"] == 0
    assert result["missing_file_days_in_ledger"] == 0
    assert result["days_after_last_file"] == 0
    assert result["try_before_first_file"] == 0.0
    assert result["try_on_missing_file_days"] == 0.0
    assert result["try_after_last_file"] == 0.0


def test_a_day_with_only_rv_and_dg_compares_as_zero():
    with pytest.raises(ValueError, match="ledger 0.00 TRY"):
        check([("RV", D1, 100.0), ("DG", D1, -100.0)], [(D1, 100.0)], [D1])


# --- document roles ------------------------------------------------------------------------

def test_lines_are_chosen_by_role_not_by_type_code():
    # DA is a payment code in this ledger; a DZ marked as a credit memo is not cash.
    ledger = ledger_table([("DA", D1, -60.0), ("DZ", D1, -40.0), ("DZ", D1, -999.0)],
                          roles=["payment", "payment", "credit_memo"])
    result = check_ledger_against_collections(ledger, collections_table([(D1, 100.0)]), [D1])
    assert result["days_compared"] == 1


def test_a_dz_line_whose_role_is_not_payment_is_ignored():
    ledger = ledger_table([("DZ", D1, -100.0)], roles=["invoice"])
    with pytest.raises(ValueError, match="daily files 100.00 TRY, ledger 0.00 TRY"):
        check_ledger_against_collections(ledger, collections_table([(D1, 100.0)]), [D1])


def test_payments_under_another_code_on_days_without_a_file_are_counted():
    ledger = ledger_table([("ZP", D1, -7.0), ("DZ", D2, -1.0)], roles=["payment", "payment"])
    result = check_ledger_against_collections(ledger, collections_table([(D2, 1.0)]), [D2])
    assert result["days_before_first_file"] == 1
    assert result["try_before_first_file"] == pytest.approx(7.0, abs=1e-9)


def test_a_ledger_without_document_role_raises_key_error():
    ledger = ledger_table([("DZ", D1, -1.0)]).drop(columns="document_role")
    with pytest.raises(KeyError, match="document_role"):
        check_ledger_against_collections(ledger, collections_table([(D1, 1.0)]), [D1])


# --- refunds -------------------------------------------------------------------------------

def test_refund_paid_out_counts_as_negative_cash():
    result = check([("DZ", D1, -500.0), ("DZ", D1, 120.0)],
                   [(D1, 500.0), (D1, -120.0)], [D1])
    assert result["days_compared"] == 1


def test_day_with_only_a_refund_ties_to_a_negative_daily_total():
    assert check([("DZ", D1, 75.0)], [(D1, -75.0)], [D1])["days_compared"] == 1


def test_refund_read_with_the_wrong_sign_raises():
    with pytest.raises(ValueError, match="daily files 75.00 TRY, ledger -75.00 TRY"):
        check([("DZ", D1, 75.0)], [(D1, 75.0)], [D1])


def test_refund_on_a_day_without_a_file_reduces_the_try_counted():
    result = check([("DZ", D1, -100.0), ("DZ", D1, 30.0), ("DZ", D2, -5.0)],
                   [(D2, 5.0)], [D2])
    assert result["days_before_first_file"] == 1
    assert result["try_before_first_file"] == pytest.approx(70.0, abs=1e-9)


# --- a day with a file and nothing on one side ---------------------------------------------

def test_day_with_a_file_but_no_ledger_payment_ties_if_daily_total_is_zero():
    result = check([("DZ", D1, -10.0)], [(D1, 10.0), (D2, 40.0), (D2, -40.0)], [D1, D2])
    assert result["days_compared"] == 2


def test_day_with_a_file_but_no_ledger_payment_raises_if_daily_total_is_not_zero():
    with pytest.raises(ValueError) as error:
        check([("DZ", D1, -10.0)], [(D1, 10.0), (D2, 40.0)], [D1, D2])
    assert "2026-03-03: daily files 40.00 TRY, ledger 0.00 TRY" in str(error.value)


def test_day_with_a_file_and_no_daily_lines_compares_the_ledger_against_zero():
    with pytest.raises(ValueError) as error:
        check([("DZ", D1, -10.0), ("DZ", D2, -40.0)], [(D1, 10.0)], [D1, D2])
    assert "2026-03-03: daily files 0.00 TRY, ledger 40.00 TRY" in str(error.value)


def test_day_with_a_file_and_nothing_on_either_side_ties():
    result = check([("DZ", D1, -10.0)], [(D1, 10.0)], [D1, D2])
    assert result["days_compared"] == 2


def test_ledger_payments_netting_to_zero_on_a_file_day_with_no_lines_tie():
    result = check([("DZ", D1, -10.0), ("DZ", D1, 10.0)], [], [D1])
    assert result["days_compared"] == 1


def test_no_ledger_lines_at_all_and_empty_daily_files_tie():
    result = check([], [], [D1, D2])
    assert result == {"days_compared": 2, "days_before_first_file": 0,
                      "try_before_first_file": 0.0, "missing_file_days_in_ledger": 0,
                      "try_on_missing_file_days": 0.0, "days_after_last_file": 0,
                      "try_after_last_file": 0.0}


def test_no_ledger_lines_at_all_against_a_nonzero_day_raises():
    with pytest.raises(ValueError, match="on 1 days"):
        check([], [(D1, 10.0)], [D1])


def test_daily_lines_on_a_day_not_in_days_with_file_are_not_compared():
    """Pinned: only days in days_with_file are compared; other daily lines are ignored."""
    result = check([("DZ", D1, -10.0)], [(D1, 10.0), (D3, 999.0)], [D1])
    assert result["days_compared"] == 1


# --- ledger payments on days without a file ------------------------------------------------

def test_ledger_days_before_the_first_file_are_counted_with_their_try():
    result = check([("DZ", date(2026, 1, 5), -100.0), ("DZ", date(2026, 1, 5), -1.5),
                    ("DZ", date(2026, 2, 9), -200.0), ("DZ", D2, -7.0)],
                   [(D2, 7.0)], [D2])
    assert result["days_before_first_file"] == 2
    assert result["try_before_first_file"] == pytest.approx(301.5, abs=1e-9)
    assert result["missing_file_days_in_ledger"] == 0
    assert result["try_on_missing_file_days"] == 0.0


def test_ledger_day_inside_the_range_without_a_file_is_a_missing_file_day():
    result = check([("DZ", D1, -1.0), ("DZ", D2, -55.25), ("DZ", D3, -3.0)],
                   [(D1, 1.0), (D3, 3.0)], [D1, D3])
    assert result["days_compared"] == 2
    assert result["missing_file_days_in_ledger"] == 1
    assert result["try_on_missing_file_days"] == pytest.approx(55.25, abs=1e-9)
    assert result["days_before_first_file"] == 0
    assert result["days_after_last_file"] == 0


def test_ledger_day_after_the_last_file_counts_after_and_not_as_missing():
    result = check([("DZ", D1, -1.0), ("DZ", D2, -2.0), ("DZ", date(2026, 4, 1), -42.0)],
                   [(D1, 1.0), (D2, 2.0)], [D1, D2])
    assert result["days_after_last_file"] == 1
    assert result["try_after_last_file"] == pytest.approx(42.0, abs=1e-9)
    assert result["missing_file_days_in_ledger"] == 0
    assert result["try_on_missing_file_days"] == 0.0
    assert result["days_before_first_file"] == 0


def test_several_ledger_days_after_the_last_file_are_counted_with_their_try():
    result = check([("DZ", D1, -1.0), ("DZ", D2, -10.0), ("DZ", D2, -0.5),
                    ("DZ", D3, -20.0)], [(D1, 1.0)], [D1])
    assert result["days_after_last_file"] == 2
    assert result["try_after_last_file"] == pytest.approx(30.5, abs=1e-9)
    assert result["missing_file_days_in_ledger"] == 0


def test_groups_split_on_the_first_and_the_last_file_day():
    result = check([("DZ", date(2026, 3, 1), -10.0),     # the day before the first file
                    ("DZ", D2, -1.0),                    # first file day
                    ("DZ", D3, -20.0),                   # missing, between files
                    ("DZ", date(2026, 3, 5), -2.0),      # last file day
                    ("DZ", date(2026, 3, 6), -30.0)],    # after the last file
                   [(D2, 1.0), (date(2026, 3, 5), 2.0)], [D2, date(2026, 3, 5)])
    assert result == {"days_compared": 2,
                      "days_before_first_file": 1,
                      "try_before_first_file": pytest.approx(10.0, abs=1e-9),
                      "missing_file_days_in_ledger": 1,
                      "try_on_missing_file_days": pytest.approx(20.0, abs=1e-9),
                      "days_after_last_file": 1,
                      "try_after_last_file": pytest.approx(30.0, abs=1e-9)}


def test_one_day_before_one_between_one_after_each_with_its_own_try():
    """Distinct amounts per group, so a TRY landing in the wrong group is caught."""
    result = check([("DZ", D1, -11.0), ("DZ", D1, -0.25),  # before the first file
                    ("DZ", D2, -1.0),                     # first file day
                    ("DZ", D3, -22.0),                    # between: a missing file
                    ("DZ", date(2026, 3, 5), -2.0),       # last file day
                    ("DZ", date(2026, 3, 6), -33.5)],     # after the last file
                   [(D2, 1.0), (date(2026, 3, 5), 2.0)], [D2, date(2026, 3, 5)])
    assert (result["days_before_first_file"], result["missing_file_days_in_ledger"],
            result["days_after_last_file"]) == (1, 1, 1)
    assert result["try_before_first_file"] == pytest.approx(11.25, abs=1e-9)
    assert result["try_on_missing_file_days"] == pytest.approx(22.0, abs=1e-9)
    assert result["try_after_last_file"] == pytest.approx(33.5, abs=1e-9)


def test_single_file_day_splits_before_and_after_and_has_no_missing_days():
    result = check([("DZ", D1, -4.0), ("DZ", D2, -5.0), ("DZ", D3, -6.0)],
                   [(D2, 5.0)], [D2])
    assert result["days_compared"] == 1
    assert result["days_before_first_file"] == 1
    assert result["try_before_first_file"] == pytest.approx(4.0, abs=1e-9)
    assert result["days_after_last_file"] == 1
    assert result["try_after_last_file"] == pytest.approx(6.0, abs=1e-9)
    assert result["missing_file_days_in_ledger"] == 0
    assert result["try_on_missing_file_days"] == 0.0


def test_first_and_last_file_days_are_in_no_without_file_group():
    result = check([("DZ", D1, -1.0), ("DZ", D3, -3.0)], [(D1, 1.0), (D3, 3.0)], [D1, D3])
    assert result["days_before_first_file"] == 0
    assert result["missing_file_days_in_ledger"] == 0
    assert result["days_after_last_file"] == 0


def test_three_groups_and_compared_days_account_for_every_ledger_payment_day():
    file_days = [date(2026, 3, 4), date(2026, 3, 9), date(2026, 3, 20)]
    ledger_days = [date(2026, 3, day) for day in range(1, 26)]  # before, between and after
    ledger_lines = [("DZ", day, -float(day.day)) for day in ledger_days]
    ledger_lines.append(("DZ", date(2026, 3, 12), 3.0))         # a refund on a missing day
    result = check(ledger_lines, [(day, float(day.day)) for day in file_days], file_days)

    groups = (result["days_before_first_file"], result["missing_file_days_in_ledger"],
              result["days_after_last_file"])
    assert groups == (3, 14, 5)
    assert result["days_compared"] + sum(groups) == len(ledger_days)

    total_received = sum(day.day for day in ledger_days) - 3.0
    on_file_days = sum(day.day for day in file_days)
    without_file = (result["try_before_first_file"] + result["try_on_missing_file_days"]
                    + result["try_after_last_file"])
    assert on_file_days + without_file == pytest.approx(total_received, abs=1e-9)


def test_refund_only_day_without_a_file_counts_as_a_day_with_negative_try():
    result = check([("DZ", D1, -1.0), ("DZ", D2, 9.0), ("DZ", D3, -3.0)],
                   [(D1, 1.0), (D3, 3.0)], [D1, D3])
    assert result["missing_file_days_in_ledger"] == 1
    assert result["try_on_missing_file_days"] == pytest.approx(-9.0, abs=1e-9)


def test_refund_only_day_after_the_last_file_gives_negative_try_after():
    result = check([("DZ", D1, -1.0), ("DZ", D2, 9.0)], [(D1, 1.0)], [D1])
    assert result["days_after_last_file"] == 1
    assert result["try_after_last_file"] == pytest.approx(-9.0, abs=1e-9)
    assert result["missing_file_days_in_ledger"] == 0
    assert result["try_on_missing_file_days"] == 0.0


# --- days_with_file: duplicates, order, kinds, empty ----------------------------------------

def test_duplicate_days_are_compared_once():
    result = check([("DZ", D1, -1.0), ("DZ", D2, -2.0)], [(D1, 1.0), (D2, 2.0)],
                   [D1, D2, D1, D2, D2])
    assert result["days_compared"] == 2


def test_unsorted_days_use_the_earliest_as_the_first_file_day():
    result = check([("DZ", D1, -5.0), ("DZ", D2, -2.0), ("DZ", D3, -3.0)],
                   [(D2, 2.0), (D3, 3.0)], [D3, D2])
    assert result["days_before_first_file"] == 1
    assert result["try_before_first_file"] == pytest.approx(5.0, abs=1e-9)
    assert result["missing_file_days_in_ledger"] == 0


@pytest.mark.parametrize("kind", ["datetime", "timestamp", "mixed"])
def test_datetimes_and_timestamps_are_converted_to_dates(kind):
    days = {
        "datetime": [datetime(2026, 3, 2), datetime(2026, 3, 3)],
        "timestamp": [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")],
        "mixed": [pd.Timestamp("2026-03-02"), datetime(2026, 3, 3), D1],
    }[kind]
    result = check([("DZ", D1, -1.0), ("DZ", D2, -2.0), ("DZ", D3, -3.0)],
                   [(D1, 1.0), (D2, 2.0)], days)
    assert result["days_compared"] == 2
    assert result["days_after_last_file"] == 1
    assert result["try_after_last_file"] == pytest.approx(3.0, abs=1e-9)
    assert result["missing_file_days_in_ledger"] == 0
    assert result["try_on_missing_file_days"] == 0.0


def test_datetime_with_a_time_of_day_counts_as_its_date():
    result = check([("DZ", D1, -1.0)], [(D1, 1.0)], [datetime(2026, 3, 2, 17, 45)])
    assert result["days_compared"] == 1
    assert result["missing_file_days_in_ledger"] == 0


def test_mismatch_still_detected_when_days_are_timestamps():
    with pytest.raises(ValueError, match="2026-03-02"):
        check([("DZ", D1, -1.0)], [(D1, 2.0)], [pd.Timestamp("2026-03-02")])


def test_empty_days_with_file_raises():
    with pytest.raises(ValueError, match="no days with a file"):
        check([("DZ", D1, -1.0)], [(D1, 1.0)], [])


def test_empty_days_with_file_raises_even_with_empty_tables():
    with pytest.raises(ValueError, match="no days with a file"):
        check([], [], [])


# --- inputs are not modified ---------------------------------------------------------------

def test_inputs_are_not_modified_on_success():
    ledger = ledger_table([("DZ", D1, -100.0), ("RV", D1, 100.0), ("DZ", D2, -3.0)])
    collections = collections_table([(D1, 100.0)])
    days = [D1, D1]
    ledger_before, collections_before = ledger.copy(deep=True), collections.copy(deep=True)
    check_ledger_against_collections(ledger, collections, days)
    pd.testing.assert_frame_equal(ledger, ledger_before, check_exact=True)
    pd.testing.assert_frame_equal(collections, collections_before, check_exact=True)
    assert days == [D1, D1]


def test_inputs_are_not_modified_on_failure():
    ledger = ledger_table([("DZ", D1, -100.0)])
    collections = collections_table([(D1, 1.0)])
    days = [pd.Timestamp("2026-03-02")]
    ledger_before, collections_before = ledger.copy(deep=True), collections.copy(deep=True)
    with pytest.raises(ValueError):
        check_ledger_against_collections(ledger, collections, days)
    pd.testing.assert_frame_equal(ledger, ledger_before, check_exact=True)
    pd.testing.assert_frame_equal(collections, collections_before, check_exact=True)
    assert days == [pd.Timestamp("2026-03-02")]


def test_same_inputs_give_the_same_result_twice():
    ledger = ledger_table([("DZ", date(2026, 1, 1), -9.0), ("DZ", D1, -1.0)])
    collections = collections_table([(D1, 1.0)])
    first = check_ledger_against_collections(ledger, collections, [D1])
    assert check_ledger_against_collections(ledger, collections, [D1]) == first


# --- integration with the synthetic ledger and corpus (SLOW) --------------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
AS_OF = date(2026, 8, 28)

EXPECTED_MISSING_FILE_DAYS = [date(2025, 9, 8), date(2025, 12, 17), date(2026, 2, 4),
                              date(2026, 3, 13), date(2026, 4, 20), date(2026, 6, 1)]


@pytest.fixture(scope="module")
def synthetic():
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    ledger = read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT)  # direct, no cache
    daily_files = list_daily_files(CORPUS_FOLDER, DATE_FORMAT)
    collections = read_all_files(daily_files, SYNTHETIC_LAYOUT)  # direct read, no cache
    return ledger, collections, list(daily_files)


def test_slow_synthetic_ledger_ties_to_the_daily_files(synthetic):
    ledger, collections, days = synthetic
    result = check_ledger_against_collections(ledger, collections, days)
    assert result["days_compared"] == 306
    assert result["days_before_first_file"] == 139
    assert result["missing_file_days_in_ledger"] == 6
    assert result["try_before_first_file"] == pytest.approx(2_240_423_149.13, abs=0.01)
    assert result["try_on_missing_file_days"] == pytest.approx(206_531_569.50, abs=0.01)
    assert result["days_after_last_file"] == 0
    assert result["try_after_last_file"] == 0.0


def test_slow_synthetic_missing_file_days_are_the_six_known_days(synthetic):
    ledger, collections, days = synthetic
    payment_days = set(ledger.loc[ledger["document_role"] == "payment",
                                  "posting_date"].dt.date)
    missing = sorted(day for day in payment_days - set(days) if min(days) < day < max(days))
    assert missing == EXPECTED_MISSING_FILE_DAYS
    result = check_ledger_against_collections(ledger, collections, days)
    assert result["missing_file_days_in_ledger"] == len(missing)
    payments = ledger[ledger["document_role"] == "payment"]
    on_missing = -payments.loc[payments["posting_date"].dt.date.isin(missing),
                               "amount_local"].sum()
    assert result["try_on_missing_file_days"] == pytest.approx(on_missing, abs=0.01)
