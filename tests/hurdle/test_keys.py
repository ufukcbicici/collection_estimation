"""Tests for `hurdle/keys.py`: `normalise_customer_id`, `add_keys` and `check_company_codes`.

One spelling per customer: numeric IDs lose their leading zeros, text IDs are trimmed and
upper-cased; the key is company code + '-' + customer ID. Customers are fake.
"""
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.keys import (
    INVISIBLE_CHARACTERS,
    add_keys,
    check_company_codes,
    normalise_customer_id,
)
from collection_estimation.hurdle.read_collections import list_daily_files, read_one_file
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, line, make_workbook

DATE_FORMAT = SYNTHETIC_LAYOUT.file_name_date_format  # "%d%m%Y"

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"


# --- numeric IDs -----------------------------------------------------------------------

def test_leading_zeros_are_dropped():
    assert normalise_customer_id("0011226452") == "11226452"


def test_sap_padded_and_unpadded_forms_give_the_same_id():
    assert normalise_customer_id("0011226452") == normalise_customer_id("11226452")


def test_surrounding_spaces_are_stripped():
    assert normalise_customer_id("  11226452 ") == "11226452"


def test_surrounding_tabs_and_newlines_are_stripped():
    assert normalise_customer_id("\t11226452\n") == "11226452"


def test_spaces_and_leading_zeros_together():
    assert normalise_customer_id("  0011226452 ") == "11226452"


def test_clean_numeric_id_is_unchanged():
    assert normalise_customer_id("11226452") == "11226452"


def test_all_zeros_becomes_a_single_zero():
    assert normalise_customer_id("0000") == "0"


def test_single_zero_stays_zero():
    assert normalise_customer_id("0") == "0"


def test_inner_zeros_are_kept():
    assert normalise_customer_id("00100200") == "100200"


# --- IDs with letters ------------------------------------------------------------------

def test_letter_id_is_trimmed_and_upper_cased():
    assert normalise_customer_id(" ab12 ") == "AB12"


def test_leading_zeros_kept_when_id_has_letters():
    assert normalise_customer_id("00AB12") == "00AB12"


def test_inner_characters_are_kept():
    assert normalise_customer_id("C-0012") == "C-0012"


def test_lower_and_upper_case_give_the_same_id():
    assert normalise_customer_id("c-0012") == normalise_customer_id("C-0012")


# --- Turkish letters (a decision) -----------------------------------------------------

def test_turkish_dotless_i_and_i_give_the_same_id_by_choice():
    # Kept on purpose: SAP customer numbers are numeric, so treating them alike is likelier
    # to fix a typo than to merge two customers.
    assert normalise_customer_id("ab\u0131" + "1") == "ABI1"  # 'abı1'
    assert normalise_customer_id("abi1") == "ABI1"


# --- invisible characters --------------------------------------------------------------
# Written as escapes, because the characters themselves cannot be seen in the source.

def test_invisible_characters_are_the_zero_width_space_and_the_byte_order_mark():
    assert INVISIBLE_CHARACTERS == ["\u200b", "\ufeff"]


def test_byte_order_mark_at_the_start_is_removed():
    assert normalise_customer_id("\ufeff0011226452") == "11226452"


def test_zero_width_space_at_the_start_is_removed():
    assert normalise_customer_id("\u200b0011") == "11"


def test_zero_width_space_inside_is_removed():
    assert normalise_customer_id("1122\u200b6452") == "11226452"


def test_byte_order_mark_inside_is_removed():
    assert normalise_customer_id("1122\ufeff6452") == "11226452"


def test_invisible_character_at_the_end_is_removed():
    assert normalise_customer_id("11226452\u200b") == "11226452"


def test_invisible_characters_between_spaces_and_zeros_are_removed():
    assert normalise_customer_id(" \ufeff 0011226452 \u200b ") == "11226452"


def test_byte_order_mark_and_clean_forms_give_the_same_id():
    assert normalise_customer_id("\ufeff0011226452") == normalise_customer_id("11226452")


def test_invisible_characters_in_a_letter_id_are_removed():
    assert normalise_customer_id("\ufeffab\u200b12") == "AB12"


@pytest.mark.parametrize("raw", ["\u200b", "\ufeff", "\ufeff\u200b", " \u200b \ufeff ",
                                 "\t\ufeff\n"])
def test_only_invisible_characters_and_spaces_is_rejected_as_empty(raw):
    with pytest.raises(ValueError, match="empty"):
        normalise_customer_id(raw)


# --- non-ASCII digits are text, not numbers (current behaviour) -------------------------

def test_superscript_digit_is_not_treated_as_a_number():
    assert normalise_customer_id("²3") == "²3"


def test_superscript_with_leading_zero_keeps_the_zero():
    assert normalise_customer_id("0²3") == "0²3"


def test_arabic_indic_digits_keep_their_zeros():
    assert normalise_customer_id("٠٠١") == "٠٠١"


def test_full_width_digits_keep_their_zeros():
    assert normalise_customer_id("００１") == "００１"


# --- inner spaces (current behaviour) ----------------------------------------------------

def test_inner_space_is_kept():
    assert normalise_customer_id("11 22") == "11 22"


def test_inner_space_stops_zero_stripping():
    assert normalise_customer_id("0011 22") == "0011 22"


# --- errors ----------------------------------------------------------------------------

def test_none_is_rejected_as_not_text():
    with pytest.raises(ValueError, match="not text"):
        normalise_customer_id(None)


def test_int_is_rejected_as_not_text():
    with pytest.raises(ValueError, match="not text"):
        normalise_customer_id(11226452)


def test_float_is_rejected_as_not_text():
    with pytest.raises(ValueError, match="not text"):
        normalise_customer_id(11226452.0)


def test_empty_string_is_rejected_as_empty():
    with pytest.raises(ValueError, match="empty"):
        normalise_customer_id("")


def test_spaces_only_is_rejected_as_empty():
    with pytest.raises(ValueError, match="empty"):
        normalise_customer_id("   ")


def test_whitespace_only_with_tabs_and_newlines_is_rejected_as_empty():
    with pytest.raises(ValueError, match="empty"):
        normalise_customer_id("\t\n ")


# --- idempotent ------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "  0011226452 ", "11226452", "0000", "0", " ab12 ", "00AB12", "C-0012", "²3", "٠٠١",
    "11 22", "\ufeff0011226452", "1122\u200b6452",
])
def test_applying_twice_gives_the_same_as_once(raw):
    once = normalise_customer_id(raw)
    assert normalise_customer_id(once) == once


# --- add_keys: helpers ------------------------------------------------------------------

KNOWN = ["TR02", "TR04"]


def payments(company_codes, customers, amounts=None, index=None):
    """A small table of payment lines: company code, raw customer, and an extra column."""
    if amounts is None:
        amounts = [100.0 * (number + 1) for number in range(len(customers))]
    return pd.DataFrame({"company_code": company_codes, "customer": customers,
                         "amount_try": amounts}, index=index)


# --- add_keys: the new columns -----------------------------------------------------------

def test_add_keys_adds_customer_id_and_key():
    with_keys = add_keys(payments(["TR02"], ["0011226452"]), KNOWN)
    assert with_keys["customer_id"].tolist() == ["11226452"]
    assert with_keys["key"].tolist() == ["TR02-11226452"]


def test_add_keys_adds_exactly_two_columns_at_the_end():
    table = payments(["TR02"], ["0011226452"])
    with_keys = add_keys(table, KNOWN)
    assert list(with_keys.columns) == list(table.columns) + ["customer_id", "key"]


def test_add_keys_keeps_every_original_column_unchanged():
    table = payments(["TR02", "TR04"], [" 0011226452 ", "ab12"], [10.5, -3.0])
    with_keys = add_keys(table, KNOWN)
    pd.testing.assert_frame_equal(with_keys[table.columns], table)


def test_add_keys_keeps_the_raw_customer_value():
    with_keys = add_keys(payments(["TR02"], ["  0011226452 "]), KNOWN)
    assert with_keys["customer"].tolist() == ["  0011226452 "]


def test_add_keys_keeps_row_count_order_and_index():
    table = payments(["TR04", "TR02", "TR02"], ["3", "1", "2"], index=[30, 10, 20])
    with_keys = add_keys(table, KNOWN)
    assert len(with_keys) == 3
    assert with_keys.index.tolist() == [30, 10, 20]
    assert with_keys["key"].tolist() == ["TR04-3", "TR02-1", "TR02-2"]


def test_add_keys_does_not_add_columns_to_the_input():
    table = payments(["TR02"], ["0011226452"])
    add_keys(table, KNOWN)
    assert list(table.columns) == ["company_code", "customer", "amount_try"]


def test_add_keys_does_not_change_the_input_values():
    table = payments(["TR02", "TR04"], ["0011226452", " ab12 "])
    before = table.copy()
    add_keys(table, KNOWN, customer_map={"11226452": "99"})
    pd.testing.assert_frame_equal(table, before)


def test_add_keys_returns_a_new_table():
    table = payments(["TR02"], ["1"])
    assert add_keys(table, KNOWN) is not table


# --- add_keys: what makes two keys equal or different -----------------------------------

def test_same_customer_in_two_companies_gets_two_keys():
    with_keys = add_keys(payments(["TR02", "TR04"], ["11226452", "11226452"]), KNOWN)
    assert with_keys["key"].tolist() == ["TR02-11226452", "TR04-11226452"]


def test_padded_and_unpadded_customer_in_one_company_get_one_key():
    with_keys = add_keys(payments(["TR02", "TR02"], ["0011226452", "11226452"]), KNOWN)
    assert with_keys["key"].nunique() == 1
    assert with_keys["key"].iloc[0] == "TR02-11226452"


def test_byte_order_mark_and_clean_customer_get_one_key():
    with_keys = add_keys(payments(["TR02", "TR02"], ["﻿0011226452", "11226452"]), KNOWN)
    assert with_keys["key"].tolist() == ["TR02-11226452", "TR02-11226452"]


# --- add_keys: company codes -------------------------------------------------------------

def test_unknown_company_code_is_rejected():
    with pytest.raises(ValueError, match="unknown company codes"):
        add_keys(payments(["TR02", "TR99"], ["1", "2"]), KNOWN)


def test_unknown_company_code_error_names_the_code_and_its_number_of_lines():
    table = payments(["TR99", "TR02", "TR99"], ["1", "2", "3"])
    with pytest.raises(ValueError) as error:
        add_keys(table, KNOWN)
    assert "'TR99': 2" in str(error.value)


def test_unknown_company_code_error_lists_the_known_codes():
    with pytest.raises(ValueError) as error:
        add_keys(payments(["TR99"], ["1"]), ["TR04", "TR02"])
    assert "['TR02', 'TR04']" in str(error.value)


def test_unknown_company_code_error_does_not_list_known_codes_found_in_the_data():
    with pytest.raises(ValueError) as error:
        add_keys(payments(["TR02", "TR99"], ["1", "2"]), KNOWN)
    lines_part = str(error.value).split("known codes are")[0]
    assert "TR02" not in lines_part


def test_several_unknown_company_codes_are_all_named():
    table = payments(["TR98", "TR99", "TR99", "TR02", "TR97"], ["1", "2", "3", "4", "5"])
    with pytest.raises(ValueError) as error:
        add_keys(table, KNOWN)
    message = str(error.value)
    assert "'TR98': 1" in message
    assert "'TR99': 2" in message
    assert "'TR97': 1" in message


@pytest.mark.parametrize("code", [" tr02", "tr02", "TR02 ", " TR02"])
def test_company_codes_are_compared_exactly(code):
    # U0 already strips company codes, so a different spelling here is a real anomaly.
    with pytest.raises(ValueError) as error:
        add_keys(payments([code], ["1"]), KNOWN)
    assert repr(code) in str(error.value)


def test_known_codes_absent_from_the_data_are_fine():
    with_keys = add_keys(payments(["TR02"], ["1"]), ["TR02", "TR04", "TR05", "TR06", "TR91"])
    assert with_keys["key"].tolist() == ["TR02-1"]


def test_company_code_is_checked_before_customers_are_normalised():
    # A bad company code and a bad customer on one table: the company code is reported.
    with pytest.raises(ValueError, match="unknown company codes"):
        add_keys(payments(["TR99"], [""]), KNOWN)


def test_missing_company_code_is_rejected():
    with pytest.raises(ValueError, match="unknown company codes"):
        add_keys(payments(["TR02", None], ["1", "2"]), KNOWN)


def test_missing_company_code_is_named_in_the_error():
    # value_counts() drops missing values by default, so the message could read '{}'.
    with pytest.raises(ValueError) as error:
        add_keys(payments(["TR02", None], ["1", "2"]), KNOWN)
    assert "{}" not in str(error.value)


# --- check_company_codes on its own -----------------------------------------------------

def test_check_company_codes_returns_none_when_all_known():
    assert check_company_codes(pd.Series(["TR02", "TR04", "TR02"]), KNOWN) is None


def test_check_company_codes_accepts_an_empty_series():
    assert check_company_codes(pd.Series([], dtype=object), KNOWN) is None


def test_check_company_codes_rejects_everything_when_no_codes_are_known():
    with pytest.raises(ValueError, match="'TR02': 1"):
        check_company_codes(pd.Series(["TR02"]), [])


# --- add_keys: customer_map --------------------------------------------------------------

def test_customer_map_merges_two_accounts_into_one_key():
    table = payments(["TR02", "TR02"], ["11226452", "11226453"])
    with_keys = add_keys(table, KNOWN, customer_map={"11226453": "11226452"})
    assert with_keys["key"].tolist() == ["TR02-11226452", "TR02-11226452"]
    assert with_keys["customer_id"].tolist() == ["11226452", "11226452"]


def test_customer_map_leaves_unmapped_ids_unchanged():
    table = payments(["TR02", "TR02"], ["0011226452", "ab12"])
    with_keys = add_keys(table, KNOWN, customer_map={"777": "888"})
    assert with_keys["customer_id"].tolist() == ["11226452", "AB12"]


def test_customer_map_keys_are_normalised():
    with_keys = add_keys(payments(["TR02"], ["11226452"]), KNOWN,
                         customer_map={"0011226452": "555"})
    assert with_keys["customer_id"].tolist() == ["555"]


def test_customer_map_values_are_normalised():
    with_keys = add_keys(payments(["TR02"], ["11226453"]), KNOWN,
                         customer_map={"11226453": "  0011226452 "})
    assert with_keys["customer_id"].tolist() == ["11226452"]
    assert with_keys["key"].tolist() == ["TR02-11226452"]


def test_customer_map_letter_keys_match_regardless_of_case():
    with_keys = add_keys(payments(["TR02"], ["AB12"]), KNOWN, customer_map={" ab12": "cd34"})
    assert with_keys["customer_id"].tolist() == ["CD34"]


def test_customer_map_keeps_the_raw_customer_value():
    with_keys = add_keys(payments(["TR02"], ["0011226453"]), KNOWN,
                         customer_map={"11226453": "11226452"})
    assert with_keys["customer"].tolist() == ["0011226453"]


def test_customer_map_applies_in_every_company():
    table = payments(["TR02", "TR04"], ["11226453", "11226453"])
    with_keys = add_keys(table, KNOWN, customer_map={"11226453": "11226452"})
    assert with_keys["key"].tolist() == ["TR02-11226452", "TR04-11226452"]


def test_customer_map_is_applied_once_not_chained():
    # 1 -> 2 and 2 -> 3: a line for customer 1 becomes 2, not 3.
    with_keys = add_keys(payments(["TR02"], ["1"]), KNOWN, customer_map={"1": "2", "2": "3"})
    assert with_keys["customer_id"].tolist() == ["2"]


@pytest.mark.parametrize("no_map", [None, {}])
def test_empty_or_missing_customer_map_means_no_mapping(no_map):
    table = payments(["TR02", "TR04"], ["0011226452", " ab12 "])
    with_keys = add_keys(table, KNOWN, customer_map=no_map)
    pd.testing.assert_frame_equal(with_keys, add_keys(table, KNOWN))
    assert with_keys["customer_id"].tolist() == ["11226452", "AB12"]


def test_bad_customer_map_entry_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        add_keys(payments(["TR02"], ["1"]), KNOWN, customer_map={"1": "   "})


def test_customer_map_is_not_modified():
    customer_map = {"0011226453": " 11226452 "}
    add_keys(payments(["TR02"], ["11226453"]), KNOWN, customer_map=customer_map)
    assert customer_map == {"0011226453": " 11226452 "}


# --- add_keys: empty and bad inputs ------------------------------------------------------

def test_empty_table_gives_an_empty_table_with_the_two_new_columns():
    empty = pd.DataFrame({"company_code": pd.Series([], dtype=object),
                          "customer": pd.Series([], dtype=object),
                          "amount_try": pd.Series([], dtype="float64")})
    with_keys = add_keys(empty, KNOWN)
    assert len(with_keys) == 0
    assert list(with_keys.columns) == ["company_code", "customer", "amount_try",
                                       "customer_id", "key"]


def test_empty_table_with_a_customer_map_gives_an_empty_table():
    # Text columns as object, as U0 gives them; an empty list alone would make them float64.
    empty = payments(pd.Series([], dtype=object), pd.Series([], dtype=object),
                     pd.Series([], dtype="float64"))
    with_keys = add_keys(empty, KNOWN, customer_map={"1": "2"})
    assert len(with_keys) == 0
    assert "key" in with_keys.columns


@pytest.mark.parametrize("customer", ["", "   ", "​"])
def test_blank_customer_is_rejected(customer):
    with pytest.raises(ValueError, match="empty"):
        add_keys(payments(["TR02", "TR02"], ["1", customer]), KNOWN)


@pytest.mark.parametrize("customer", [None, 11226452, 11226452.0, float("nan")])
def test_customer_that_is_not_text_is_rejected(customer):
    with pytest.raises(ValueError, match="not text"):
        add_keys(payments(["TR02", "TR02"], ["1", customer]), KNOWN)


# --- add_keys on a workbook read by U0 --------------------------------------------------

def test_add_keys_on_a_read_workbook(tmp_path):
    path = make_workbook(tmp_path, [
        line("0011226452", 100.0, company="TR02"),
        line("11226452", 50.0, company="TR02"),
        line("0011226452", 25.0, company="TR04"),
    ])
    lines = read_one_file(path, SYNTHETIC_LAYOUT)
    with_keys = add_keys(lines, ["TR02", "TR04", "TR05", "TR06", "TR91"])
    assert with_keys["key"].tolist() == ["TR02-11226452", "TR02-11226452", "TR04-11226452"]
    assert with_keys["customer"].tolist() == ["0011226452", "11226452", "0011226452"]
    pd.testing.assert_frame_equal(with_keys[lines.columns], lines)


def test_add_keys_on_a_read_workbook_with_an_unknown_company(tmp_path):
    path = make_workbook(tmp_path, [line("1", 100.0, company="TR02"),
                                    line("2", 50.0, company="TR77")])
    with pytest.raises(ValueError, match="'TR77': 1"):
        add_keys(read_one_file(path, SYNTHETIC_LAYOUT), ["TR02"])


# --- the synthetic corpus --------------------------------------------------------------

@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason="synthetic corpus not installed")
def test_synthetic_first_file_gets_one_key_per_company_and_customer():
    first_file = next(iter(list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT).values()))
    lines = read_one_file(first_file, SYNTHETIC_LAYOUT)
    with_keys = add_keys(lines, ["TR02", "TR04", "TR05", "TR06", "TR91"])
    pairs = lines[["company_code", "customer"]].drop_duplicates()
    assert len(pairs) > 0
    assert with_keys["key"].nunique() == len(pairs)

@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason="synthetic corpus not installed")
def test_synthetic_customer_ids_are_already_normal():
    first_file = next(iter(list_daily_files(SYNTHETIC_CORPUS, DATE_FORMAT).values()))
    customers = read_one_file(first_file, SYNTHETIC_LAYOUT)["customer"]
    assert len(customers) > 0
    for customer in customers:
        assert normalise_customer_id(customer) == customer
