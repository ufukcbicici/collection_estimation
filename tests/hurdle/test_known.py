"""Tests for `hurdle/known.py`: when each key becomes known.

Hand-built tables for the unit tests; one slow integration test against the synthetic
corpus and ledger in data/raw (skipped if they are absent; never uses data/interim).
"""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.known import (
    KNOWN_COLUMNS, check_daily_keys_in_ledger, known_weeks)
from collection_estimation.hurdle.series import week_numbers

WEEK_ZERO = date(2025, 6, 2)  # a Monday


def ts(text):
    return pd.Timestamp(text)


def collections_table(rows):
    """rows: (company_code, customer_id, document_date)."""
    frame = pd.DataFrame(rows, columns=["company_code", "customer_id", "document_date"])
    frame = frame.astype({"company_code": object, "customer_id": object})
    frame["document_date"] = pd.to_datetime(frame["document_date"])
    frame["key"] = (frame["company_code"] + "-" + frame["customer_id"]).astype(object)
    return frame


def ledger_table(rows):
    """rows: (company_code, customer_id, posting_date, document_role)."""
    frame = pd.DataFrame(rows, columns=["company_code", "customer_id", "posting_date",
                                        "document_role"])
    frame = frame.astype({"company_code": object, "customer_id": object,
                          "document_role": object})
    frame["posting_date"] = pd.to_datetime(frame["posting_date"])
    frame["key"] = (frame["company_code"] + "-" + frame["customer_id"]).astype(object)
    return frame


def row_of(known, key):
    rows = known[known["key"] == key]
    assert len(rows) == 1
    return rows.iloc[0]


# --- shape ------------------------------------------------------------------------------

@pytest.fixture
def mixed():
    collections = collections_table([
        ("TR02", "300", "2025-06-10"),
        ("TR02", "100", "2025-06-05"),
        ("TR02", "100", "2025-06-03"),
        ("TR04", "200", "2025-07-01"),
    ])
    ledger = ledger_table([
        ("TR04", "200", "2025-06-20", "invoice"),
        ("TR02", "300", "2025-06-12", "payment"),
        ("TR02", "100", "2025-05-01", "invoice"),
        ("TR05", "999", "2025-08-01", "credit_memo"),  # only in the ledger
        ("TR02", "100", "2025-06-15", "payment"),
    ])
    return collections, ledger


def test_columns_are_exactly_known_columns(mixed):
    known = known_weeks(*mixed, WEEK_ZERO)
    assert list(known.columns) == KNOWN_COLUMNS


def test_known_columns_constant():
    assert KNOWN_COLUMNS == ["key", "company_code", "customer_id", "first_seen",
                             "known_week", "first_seen_in"]


def test_one_row_per_key_union_of_both_sources(mixed):
    collections, ledger = mixed
    known = known_weeks(collections, ledger, WEEK_ZERO)
    assert known["key"].is_unique
    assert set(known["key"]) == set(collections["key"]) | set(ledger["key"])
    assert len(known) == 4


def test_sorted_by_key_with_default_index(mixed):
    known = known_weeks(*mixed, WEEK_ZERO)
    assert list(known["key"]) == sorted(known["key"])
    assert list(known.index) == list(range(len(known)))


def test_company_code_and_customer_id_carried(mixed):
    known = known_weeks(*mixed, WEEK_ZERO)
    row = row_of(known, "TR04-200")
    assert row["company_code"] == "TR04"
    assert row["customer_id"] == "200"


def test_dtypes(mixed):
    known = known_weeks(*mixed, WEEK_ZERO)
    assert pd.api.types.is_datetime64_any_dtype(known["first_seen"])
    assert known["known_week"].dtype == "int64"
    for column in ["key", "company_code", "customer_id", "first_seen_in"]:
        assert known[column].dtype == object, column


# --- first_seen and first_seen_in ---------------------------------------------------------

def test_ledger_earlier_gives_ledger(mixed):
    known = known_weeks(*mixed, WEEK_ZERO)
    row = row_of(known, "TR02-100")
    assert row["first_seen"] == ts("2025-05-01")
    assert row["first_seen_in"] == "ledger"


def test_daily_files_earlier_gives_daily_files(mixed):
    known = known_weeks(*mixed, WEEK_ZERO)
    row = row_of(known, "TR02-300")
    assert row["first_seen"] == ts("2025-06-10")
    assert row["first_seen_in"] == "daily files"


def test_key_only_in_ledger_is_ledger(mixed):
    known = known_weeks(*mixed, WEEK_ZERO)
    row = row_of(known, "TR05-999")
    assert row["first_seen"] == ts("2025-08-01")
    assert row["first_seen_in"] == "ledger"


def test_same_day_in_both_gives_ledger():
    collections = collections_table([("TR02", "1", "2025-06-04")])
    ledger = ledger_table([("TR02", "1", "2025-06-04", "payment")])
    row = row_of(known_weeks(collections, ledger, WEEK_ZERO), "TR02-1")
    assert row["first_seen"] == ts("2025-06-04")
    assert row["first_seen_in"] == "ledger"


def test_same_day_tie_gives_ledger_whatever_the_row_order():
    # Several rows on both sides, ledger rows in reverse order: still 'ledger'.
    collections = collections_table([("TR02", "1", "2025-06-09"), ("TR02", "1", "2025-06-04")])
    ledger = ledger_table([("TR02", "1", "2025-07-01", "invoice"),
                           ("TR02", "1", "2025-06-04", "payment")])
    row = row_of(known_weeks(collections, ledger, WEEK_ZERO), "TR02-1")
    assert row["first_seen_in"] == "ledger"


def test_daily_files_one_day_earlier_than_ledger():
    collections = collections_table([("TR02", "1", "2025-06-03")])
    ledger = ledger_table([("TR02", "1", "2025-06-04", "payment")])
    row = row_of(known_weeks(collections, ledger, WEEK_ZERO), "TR02-1")
    assert row["first_seen"] == ts("2025-06-03")
    assert row["first_seen_in"] == "daily files"


@pytest.mark.parametrize("role", ["invoice", "payment", "credit_memo"])
def test_any_ledger_role_counts_as_first_interaction(role):
    collections = collections_table([("TR02", "1", "2025-06-20")])
    ledger = ledger_table([("TR02", "1", "2025-06-25", "invoice"),
                           ("TR02", "1", "2025-06-10", role)])
    row = row_of(known_weeks(collections, ledger, WEEK_ZERO), "TR02-1")
    assert row["first_seen"] == ts("2025-06-10")
    assert row["first_seen_in"] == "ledger"


def test_first_seen_is_the_minimum_not_the_first_row():
    collections = collections_table([("TR02", "1", "2025-09-01"), ("TR02", "1", "2025-07-01"),
                                     ("TR02", "1", "2025-08-01")])
    ledger = ledger_table([("TR02", "1", "2025-10-01", "invoice"),
                           ("TR02", "1", "2025-07-15", "invoice")])
    row = row_of(known_weeks(collections, ledger, WEEK_ZERO), "TR02-1")
    assert row["first_seen"] == ts("2025-07-01")
    assert row["first_seen_in"] == "daily files"


def test_same_customer_in_two_companies_gives_two_keys():
    collections = collections_table([("TR02", "500", "2025-06-10"),
                                     ("TR04", "500", "2025-06-03")])
    ledger = ledger_table([("TR02", "500", "2025-01-15", "invoice"),
                           ("TR04", "500", "2025-07-01", "invoice")])
    known = known_weeks(collections, ledger, WEEK_ZERO)
    assert list(known["key"]) == ["TR02-500", "TR04-500"]
    tr02, tr04 = row_of(known, "TR02-500"), row_of(known, "TR04-500")
    assert (tr02["first_seen"], tr02["first_seen_in"]) == (ts("2025-01-15"), "ledger")
    assert (tr04["first_seen"], tr04["first_seen_in"]) == (ts("2025-06-03"), "daily files")
    assert tr02["company_code"] == "TR02" and tr04["company_code"] == "TR04"


# --- known_week -------------------------------------------------------------------------

@pytest.mark.parametrize("day", ["2025-06-02", "2025-06-03", "2025-06-06", "2025-06-07",
                                 "2025-06-08"])
def test_monday_to_sunday_of_week_zero_is_week_0(day):
    ledger = ledger_table([("TR02", "1", day, "invoice")])
    known = known_weeks(collections_table([]), ledger, WEEK_ZERO)
    assert known["known_week"].tolist() == [0]


@pytest.mark.parametrize("day, week", [("2025-06-09", 1), ("2025-06-01", -1),
                                       ("2025-05-26", -1), ("2025-05-25", -2),
                                       ("2026-08-28", 64)])
def test_known_week_boundaries(day, week):
    ledger = ledger_table([("TR02", "1", day, "invoice")])
    known = known_weeks(collections_table([]), ledger, WEEK_ZERO)
    assert known["known_week"].tolist() == [week]


def test_date_before_week_zero_is_the_right_negative_week():
    # 2024-10-25 is a Friday; its Monday is 2024-10-21, 224 days = 32 weeks before 2025-06-02.
    ledger = ledger_table([("TR02", "1", "2024-10-25", "invoice")])
    known = known_weeks(collections_table([]), ledger, WEEK_ZERO)
    assert known["known_week"].tolist() == [-32]


def test_known_week_matches_week_numbers(mixed):
    known = known_weeks(*mixed, WEEK_ZERO)
    expected = week_numbers(known["first_seen"], WEEK_ZERO)
    assert known["known_week"].tolist() == expected.tolist()


def test_week_zero_not_a_monday_raises(mixed):
    with pytest.raises(ValueError, match="Monday"):
        known_weeks(*mixed, date(2025, 6, 3))


# --- the daily-keys-in-ledger check -----------------------------------------------------

def test_daily_key_missing_from_ledger_raises():
    collections = collections_table([("TR02", "1", "2025-06-03"), ("TR02", "2", "2025-06-03")])
    ledger = ledger_table([("TR02", "1", "2025-06-01", "invoice")])
    with pytest.raises(ValueError) as error:
        known_weeks(collections, ledger, WEEK_ZERO)
    assert str(error.value) == ("1 keys in the daily files have no line in the ledger "
                                "(first: ['TR02-2'])")


def test_missing_keys_listed_sorted_and_at_most_ten():
    customers = [f"{n:02d}" for n in range(14, 0, -1)]  # 14 keys, unsorted, all missing
    collections = collections_table([("TR02", c, "2025-06-03") for c in customers])
    ledger = ledger_table([("TR04", "x", "2025-06-01", "invoice")])
    with pytest.raises(ValueError) as error:
        check_daily_keys_in_ledger(collections, ledger)
    expected_first = sorted(f"TR02-{c}" for c in customers)[:10]
    assert str(error.value) == (f"14 keys in the daily files have no line in the ledger "
                                f"(first: {expected_first})")


def test_missing_key_counted_once_even_with_many_lines():
    collections = collections_table([("TR02", "9", "2025-06-03")] * 5)
    ledger = ledger_table([("TR02", "1", "2025-06-01", "invoice")])
    with pytest.raises(ValueError, match=r"^1 keys in the daily files"):
        check_daily_keys_in_ledger(collections, ledger)


def test_check_passes_when_all_daily_keys_in_ledger(mixed):
    assert check_daily_keys_in_ledger(*mixed) is None


def test_check_passes_with_keys_only_in_ledger():
    collections = collections_table([("TR02", "1", "2025-06-03")])
    ledger = ledger_table([("TR02", "1", "2025-06-01", "invoice"),
                           ("TR06", "7", "2025-06-01", "invoice")])
    check_daily_keys_in_ledger(collections, ledger)


# --- purity and empty inputs ------------------------------------------------------------

def test_inputs_not_modified(mixed):
    collections, ledger = mixed
    collections_before, ledger_before = collections.copy(), ledger.copy()
    known_weeks(collections, ledger, WEEK_ZERO)
    pd.testing.assert_frame_equal(collections, collections_before)
    pd.testing.assert_frame_equal(ledger, ledger_before)


def test_deterministic(mixed):
    pd.testing.assert_frame_equal(known_weeks(*mixed, WEEK_ZERO), known_weeks(*mixed, WEEK_ZERO))


def test_empty_inputs_give_empty_table():
    known = known_weeks(collections_table([]), ledger_table([]), WEEK_ZERO)
    assert len(known) == 0
    assert list(known.columns) == KNOWN_COLUMNS


def test_empty_daily_files_with_a_ledger():
    ledger = ledger_table([("TR02", "1", "2025-06-01", "invoice")])
    known = known_weeks(collections_table([]), ledger, WEEK_ZERO)
    assert known["key"].tolist() == ["TR02-1"]
    assert known["first_seen_in"].tolist() == ["ledger"]
    assert known["known_week"].dtype == "int64"


def test_extra_columns_are_dropped(mixed):
    collections, ledger = mixed
    collections = collections.assign(amount=1.0, week=3)
    known = known_weeks(collections, ledger, WEEK_ZERO)
    assert list(known.columns) == KNOWN_COLUMNS


# --- integration with the synthetic corpus and ledger (SLOW) ---------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
AS_OF = date(2026, 8, 28)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]


@pytest.fixture(scope="module")
def synthetic_known():
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    from collection_estimation.hurdle.keys import add_keys
    from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
    from collection_estimation.hurdle.read_ledger import read_ledger
    from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

    daily_files = list_daily_files(CORPUS_FOLDER, SYNTHETIC_LAYOUT.file_name_date_format)
    collections = add_keys(read_all_files(daily_files, SYNTHETIC_LAYOUT), KNOWN_COMPANIES)
    ledger = add_keys(read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT),
                      KNOWN_COMPANIES)
    return collections, ledger, known_weeks(collections, ledger, WEEK_ZERO)


def test_slow_synthetic_has_4590_keys(synthetic_known):
    _, _, known = synthetic_known
    assert len(known) == 4590
    assert known["key"].is_unique


def test_slow_synthetic_known_week_split(synthetic_known):
    _, _, known = synthetic_known
    assert (known["known_week"] < 0).sum() == 2027
    assert known["known_week"].between(0, 64).sum() == 2563


def test_slow_synthetic_all_first_seen_in_ledger(synthetic_known):
    _, _, known = synthetic_known
    assert set(known["first_seen_in"]) == {"ledger"}


def test_slow_synthetic_ledger_only_keys_included(synthetic_known):
    collections, ledger, known = synthetic_known
    ledger_only = set(ledger["key"]) - set(collections["key"])
    assert len(ledger_only) == 144
    assert ledger_only <= set(known["key"])


def test_slow_synthetic_first_seen_not_after_first_daily_date(synthetic_known):
    collections, _, known = synthetic_known
    first_daily = collections.groupby("key")["document_date"].min()
    first_seen = known.set_index("key")["first_seen"].reindex(first_daily.index)
    assert first_seen.notna().all()
    assert (first_seen <= first_daily).all()
