"""A ledger that starts AFTER the first daily file (week 0): steps 10 and 11 of
`run_hurdle_pipeline.main()`, replayed on small hand-built tables with the real functions.

main() takes the price range from the ledger's first posting and the history start from
`first_history_week`, both floored at week 0 (`min(..., 0)`). These tests show why: without
the floor `published_levels` (normalised to week 0) and `weekly_matrix` (which must hold the
daily files' receipts from week 0) raise; with it the run goes on, the ledger simply gives no
history before the first file, and the matrix starts at week 0. For a ledger that starts
before week 0 the floor changes nothing.

Nothing here reads the corpus or reads or writes data/interim.
"""
from datetime import date

import pandas as pd
import pytest

from collection_estimation.hurdle.features_history import (
    first_history_week,
    history_features,
    ledger_receipts_before,
    weekly_matrix,
)
from collection_estimation.hurdle.prices import price_weeks, published_levels
from collection_estimation.hurdle.series import week_numbers

WEEK_ZERO = date(2025, 6, 2)        # a Monday, the week of the first daily file
FIRST_FILE_DAY = date(2025, 6, 2)
LAST_WEEK = 10
KEYS = ["TR02-1", "TR02-2"]


def ledger_table(rows):
    """rows: (key, document_role, posting_date text, amount_local)."""
    return pd.DataFrame({
        "key": pd.Series([row[0] for row in rows], dtype=object),
        "document_role": pd.Series([row[1] for row in rows], dtype=object),
        "posting_date": pd.to_datetime(pd.Series([row[2] for row in rows], dtype=object)),
        "amount_local": pd.Series([row[3] for row in rows], dtype="float64"),
    })


def weekly_table(rows):
    """rows: (key, week, gross_receipts), the daily files' receipts from `weekly_amounts`."""
    return pd.DataFrame({
        "key": pd.Series([row[0] for row in rows], dtype=object),
        "company_code": pd.Series([row[0].split("-")[0] for row in rows], dtype=object),
        "customer_id": pd.Series([row[0].split("-")[1] for row in rows], dtype=object),
        "week": pd.Series([row[1] for row in rows], dtype="int64"),
        "gross_receipts": pd.Series([row[2] for row in rows], dtype="float64"),
        "refunds": pd.Series([0.0] * len(rows), dtype="float64"),
        "unapplied_lines": pd.Series([0] * len(rows), dtype="int64"),
        "lines": pd.Series([1] * len(rows), dtype="int64"),
    })


def calendar(last_week=LAST_WEEK, missing=()):
    weeks = list(range(last_week + 1))
    is_missing = [week in missing for week in weeks]
    return pd.DataFrame({
        "week": pd.Series(weeks, dtype="int64"),
        "monday": [WEEK_ZERO + pd.Timedelta(weeks=week) for week in weeks],
        "business_days": pd.Series([5] * len(weeks), dtype="int64"),
        "days_with_file": pd.Series([4 if m else 5 for m in is_missing], dtype="int64"),
        "filled_days": pd.Series([0] * len(weeks), dtype="int64"),
        "missing_days": pd.Series([1 if m else 0 for m in is_missing], dtype="int64"),
        "is_missing": pd.Series(is_missing, dtype=bool),
    })


def known_table(pairs):
    return pd.DataFrame({"key": pd.Series(list(pairs), dtype=object),
                         "known_week": pd.Series(list(pairs.values()), dtype="int64")})


# The daily files have receipts in weeks 0, 1 and 2, before the late ledger starts.
DAILY = weekly_table([("TR02-1", 0, 100.0), ("TR02-1", 2, 50.0), ("TR02-2", 1, 30.0),
                      ("TR02-2", 5, 70.0), ("TR02-1", 9, 10.0)])

# Week 3 starts Monday 2025-06-23. This ledger's first posting is Wednesday 2025-06-25, in
# week 3, so its first fully covered week is 4.
LATE_LEDGER = ledger_table([
    ("TR02-1", "invoice", "2025-06-25", 500.0),
    ("TR02-1", "payment", "2025-06-27", -200.0),
    ("TR02-2", "payment", "2025-07-08", -70.0),
])

# A ledger whose first posting is Wednesday 2025-05-07 (week -4): it starts before week 0.
EARLY_LEDGER = ledger_table([
    ("TR02-1", "invoice", "2025-05-07", 300.0),
    ("TR02-1", "payment", "2025-05-09", -40.0),     # week -4, partly covered: dropped
    ("TR02-1", "payment", "2025-05-14", -60.0),     # week -3
    ("TR02-2", "payment", "2025-05-28", -25.0),     # week -1
    ("TR02-2", "payment", "2025-06-10", -70.0),     # after the first file: not early
])


def first_ledger_week(ledger):
    """As main() computes it (step 10)."""
    return int(week_numbers(pd.Series([ledger["posting_date"].min()]), WEEK_ZERO).iloc[0])


def history_start_and_early(ledger, floored=True):
    """As main() computes them in step 11 with use_ledger_before_first_file on."""
    start = first_history_week(ledger, WEEK_ZERO)
    if floored:
        start = min(start, 0)
    early = ledger_receipts_before(ledger, FIRST_FILE_DAY, WEEK_ZERO)
    return start, early[early["week"] >= start]


# --- the late ledger: the week numbers -----------------------------------------------------

def test_late_ledger_first_posting_is_in_week_3():
    assert first_ledger_week(LATE_LEDGER) == 3


def test_late_ledger_first_history_week_is_4():
    assert first_history_week(LATE_LEDGER, WEEK_ZERO) == 4


def test_late_ledger_starting_on_a_monday_of_week_3_gives_history_week_3():
    ledger = ledger_table([("TR02-1", "payment", "2025-06-23", -10.0)])
    assert first_ledger_week(ledger) == 3
    assert first_history_week(ledger, WEEK_ZERO) == 3
    assert min(first_history_week(ledger, WEEK_ZERO), 0) == 0


# --- step 10: price levels -------------------------------------------------------------------

def test_without_the_floor_published_levels_raises_for_a_late_ledger():
    with pytest.raises(ValueError, match="do not include week 0"):
        published_levels(WEEK_ZERO, first_ledger_week(LATE_LEDGER), LAST_WEEK)


def test_with_the_floor_published_levels_works_for_a_late_ledger():
    first_price_week = min(first_ledger_week(LATE_LEDGER), 0)
    assert first_price_week == 0
    levels = published_levels(WEEK_ZERO, first_price_week, LAST_WEEK)
    assert list(levels.index) == list(range(0, LAST_WEEK + 1))
    assert levels.loc[0] == pytest.approx(1.0)
    assert levels.notna().all()


def test_with_the_floor_price_weeks_start_at_week_zero_for_a_late_ledger():
    first_price_week = min(first_ledger_week(LATE_LEDGER), 0)
    weeks = price_weeks(WEEK_ZERO, first_price_week, LAST_WEEK)
    assert weeks["week"].iloc[0] == 0
    assert weeks["week_start"].iloc[0] == pd.Timestamp(WEEK_ZERO)
    assert weeks["week"].iloc[-1] == LAST_WEEK


def test_without_the_floor_the_late_ledger_price_range_would_miss_weeks_0_to_2():
    # price_weeks itself does not raise, but the range lacks the daily files' first weeks.
    weeks = price_weeks(WEEK_ZERO, first_ledger_week(LATE_LEDGER), LAST_WEEK)
    assert weeks["week"].iloc[0] == 3


# --- step 11: early receipts and the weekly matrix ------------------------------------------

def test_late_ledger_gives_no_early_receipts():
    start, early = history_start_and_early(LATE_LEDGER)
    assert start == 0
    assert early.empty
    assert list(early.columns) == ["key", "week", "gross_receipts"]


def test_late_ledger_gives_no_early_receipts_even_without_the_floor():
    # The ledger has no line before the first file, so there is nothing to cut.
    assert ledger_receipts_before(LATE_LEDGER, FIRST_FILE_DAY, WEEK_ZERO).empty


def test_without_the_floor_weekly_matrix_raises_for_a_late_ledger():
    start, early = history_start_and_early(LATE_LEDGER, floored=False)
    assert start == 4
    levels = published_levels(WEEK_ZERO, 0, LAST_WEEK)
    with pytest.raises(ValueError, match=r"receipts in weeks \[0, 1, 2\] fall outside weeks "
                                         r"4 to 10"):
        weekly_matrix(DAILY, early, pd.Series(KEYS, dtype=object), calendar(), levels,
                      start, LAST_WEEK)


def test_with_the_floor_weekly_matrix_starts_at_week_zero_for_a_late_ledger():
    start, early = history_start_and_early(LATE_LEDGER)
    levels = published_levels(WEEK_ZERO, min(first_ledger_week(LATE_LEDGER), 0), LAST_WEEK)
    matrix = weekly_matrix(DAILY, early, pd.Series(KEYS, dtype=object), calendar(), levels,
                           start, LAST_WEEK)
    assert list(matrix.columns) == list(range(0, LAST_WEEK + 1))
    assert list(matrix.index) == KEYS
    assert not any(week < 0 for week in matrix.columns)


def test_with_the_floor_the_late_ledger_matrix_is_the_daily_files_deflated():
    start, early = history_start_and_early(LATE_LEDGER)
    levels = published_levels(WEEK_ZERO, 0, LAST_WEEK)
    matrix = weekly_matrix(DAILY, early, pd.Series(KEYS, dtype=object), calendar(), levels,
                           start, LAST_WEEK)
    nominal = matrix.mul(levels, axis=1)
    assert nominal.loc["TR02-1", 0] == pytest.approx(100.0)
    assert nominal.loc["TR02-1", 2] == pytest.approx(50.0)
    assert nominal.loc["TR02-2", 1] == pytest.approx(30.0)
    assert nominal.loc["TR02-2", 5] == pytest.approx(70.0)
    assert nominal.to_numpy().sum() == pytest.approx(DAILY["gross_receipts"].sum())
    # The late ledger's own payments (week 3, week 5) are not added: they are not early.
    assert nominal.loc["TR02-1", 3] == 0.0


def test_with_the_floor_the_late_ledger_is_the_same_as_the_setting_switched_off():
    # Switched off, main() uses history_start 0 and no early receipts at all.
    levels = published_levels(WEEK_ZERO, 0, LAST_WEEK)
    keys = pd.Series(KEYS, dtype=object)
    start, early = history_start_and_early(LATE_LEDGER)
    on = weekly_matrix(DAILY, early, keys, calendar(missing=(6,)), levels, start, LAST_WEEK)
    none = ledger_receipts_before(LATE_LEDGER, FIRST_FILE_DAY, WEEK_ZERO).iloc[0:0]
    off = weekly_matrix(DAILY, none, keys, calendar(missing=(6,)), levels, 0, LAST_WEEK)
    pd.testing.assert_frame_equal(on, off)


def test_with_the_floor_history_features_work_for_a_late_ledger():
    start, early = history_start_and_early(LATE_LEDGER)
    levels = published_levels(WEEK_ZERO, 0, LAST_WEEK)
    matrix = weekly_matrix(DAILY, early, pd.Series(KEYS, dtype=object), calendar(), levels,
                           start, LAST_WEEK)
    known = known_table({"TR02-1": 0, "TR02-2": 1})
    history = history_features(matrix, known, LAST_WEEK, (4, 13))
    # TR02-1 at origins 0..10 (11 rows), TR02-2 at 1..10 (10 rows).
    assert len(history) == 21
    assert history["origin_week"].min() == 0
    first = history[(history["key"] == "TR02-1") & (history["origin_week"] == 0)].iloc[0]
    assert first["weeks_since_last_payment"] == 0
    assert first["observed_weeks_4"] == 1          # no weeks before 0 in the matrix
    assert first["observed_weeks_13"] == 1


# --- a ledger starting before week 0: the floor changes nothing -------------------------------

def test_early_ledger_week_numbers():
    assert first_ledger_week(EARLY_LEDGER) == -4
    assert first_history_week(EARLY_LEDGER, WEEK_ZERO) == -3


def test_early_ledger_floor_leaves_the_price_week_unchanged():
    week = first_ledger_week(EARLY_LEDGER)
    assert min(week, 0) == week


def test_early_ledger_floor_leaves_the_history_start_unchanged():
    week = first_history_week(EARLY_LEDGER, WEEK_ZERO)
    assert min(week, 0) == week


def test_early_ledger_price_levels_are_the_same_with_and_without_the_floor():
    week = first_ledger_week(EARLY_LEDGER)
    pd.testing.assert_series_equal(published_levels(WEEK_ZERO, min(week, 0), LAST_WEEK),
                                   published_levels(WEEK_ZERO, week, LAST_WEEK))
    pd.testing.assert_frame_equal(price_weeks(WEEK_ZERO, min(week, 0), LAST_WEEK),
                                  price_weeks(WEEK_ZERO, week, LAST_WEEK))


def test_early_ledger_matrix_is_the_same_with_and_without_the_floor():
    levels = published_levels(WEEK_ZERO, first_ledger_week(EARLY_LEDGER), LAST_WEEK)
    keys = pd.Series(KEYS, dtype=object)
    start_f, early_f = history_start_and_early(EARLY_LEDGER, floored=True)
    start_u, early_u = history_start_and_early(EARLY_LEDGER, floored=False)
    assert start_f == start_u == -3
    pd.testing.assert_frame_equal(early_f, early_u)
    floored = weekly_matrix(DAILY, early_f, keys, calendar(), levels, start_f, LAST_WEEK)
    unfloored = weekly_matrix(DAILY, early_u, keys, calendar(), levels, start_u, LAST_WEEK)
    pd.testing.assert_frame_equal(floored, unfloored)
    assert list(floored.columns) == list(range(-3, LAST_WEEK + 1))


def test_early_ledger_early_receipts_are_kept_from_the_history_start():
    start, early = history_start_and_early(EARLY_LEDGER)
    got = {(row.key, row.week): row.gross_receipts for row in early.itertuples()}
    # Week -4 (partly covered) is cut; the payment after the first file is not early.
    assert got == {("TR02-1", -3): 60.0, ("TR02-2", -1): 25.0}


def test_a_ledger_starting_in_week_zero_floor_is_a_no_op():
    ledger = ledger_table([("TR02-1", "payment", "2025-06-02", -10.0)])
    assert first_ledger_week(ledger) == 0
    assert first_history_week(ledger, WEEK_ZERO) == 0
    levels = published_levels(WEEK_ZERO, min(first_ledger_week(ledger), 0), LAST_WEEK)
    start, early = history_start_and_early(ledger)
    assert start == 0 and early.empty
    matrix = weekly_matrix(DAILY, early, pd.Series(KEYS, dtype=object), calendar(), levels,
                           start, LAST_WEEK)
    assert matrix.columns[0] == 0


def test_a_ledger_starting_mid_week_zero_gives_history_week_one_floored_to_zero():
    # First posting Wednesday of week 0: first full week is 1, after the first file. Without
    # the floor the matrix would start at week 1 and lose the daily files' week 0.
    ledger = ledger_table([("TR02-1", "payment", "2025-06-04", -10.0)])
    assert first_history_week(ledger, WEEK_ZERO) == 1
    levels = published_levels(WEEK_ZERO, 0, LAST_WEEK)
    keys = pd.Series(KEYS, dtype=object)
    with pytest.raises(ValueError, match=r"fall outside weeks 1 to 10"):
        weekly_matrix(DAILY, ledger_receipts_before(ledger, FIRST_FILE_DAY, WEEK_ZERO), keys,
                      calendar(), levels, first_history_week(ledger, WEEK_ZERO), LAST_WEEK)
    start, early = history_start_and_early(ledger)
    assert start == 0
    matrix = weekly_matrix(DAILY, early, keys, calendar(), levels, start, LAST_WEEK)
    assert matrix.columns[0] == 0
