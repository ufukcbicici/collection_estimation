"""Tests for the labels of `hurdle/panel.py` (U4.2): `add_labels`, `check_labels_within_receipts`
and the constants LABELLED, MISSING_WEEK, FUTURE, LABEL_TOLERANCE.

Hand-built panels (from `panel_rows` on a small known table), weekly tables shaped like
`series.weekly_amounts` and calendars shaped like `series.week_calendar`; one slow
integration block against the synthetic corpus and ledger in data/raw (skipped if absent;
never uses data/interim).
"""
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.panel import (
    FUTURE,
    LABEL_TOLERANCE,
    LABELLED,
    MISSING_WEEK,
    PANEL_ROW_COLUMNS,
    add_labels,
    check_labels_within_receipts,
    panel_rows,
)
from tests.hurdle.test_panel import known_table

WEEKLY_COLUMNS = ["key", "company_code", "customer_id", "week", "gross_receipts", "refunds",
                  "unapplied_lines", "lines"]
WEEKS_COLUMNS = ["week", "monday", "business_days", "days_with_file", "filled_days",
                 "missing_days", "is_missing"]
LABEL_COLUMNS = ["label_status", "amount", "paid"]


def weekly_table(rows):
    """rows: (key, week, gross_receipts, refunds). Shaped like `weekly_amounts`' output."""
    records = []
    for key, week, gross, refunds in rows:
        company, customer = key.split("-")
        records.append((key, company, customer, week, float(gross), float(refunds), 0,
                        1 + (refunds != 0)))
    frame = pd.DataFrame(records, columns=WEEKLY_COLUMNS)
    return frame.astype({"key": object, "company_code": object, "customer_id": object,
                         "week": "int64", "gross_receipts": "float64", "refunds": "float64",
                         "unapplied_lines": "int64", "lines": "int64"})


def weeks_table(last_week, missing=()):
    """Weeks 0 ... last_week, the weeks in `missing` with one missing day."""
    rows = []
    for week in range(last_week + 1):
        gap = 1 if week in missing else 0
        rows.append((week, date(2025, 6, 2) + timedelta(days=7 * week), 5, 5 - gap, 0, gap,
                     bool(gap)))
    frame = pd.DataFrame(rows, columns=WEEKS_COLUMNS)
    return frame.astype({"week": "int64", "business_days": "int64",
                         "days_with_file": "int64", "filled_days": "int64",
                         "missing_days": "int64", "is_missing": bool})


# The running example: weeks 0-4, week 3 missing; A known from the start, B from week 2.
# Panel = panel_rows(known, 4, 2): target weeks 1 ... 6, so 5 and 6 are FUTURE.
@pytest.fixture
def known():
    return known_table([("TR02", "A", -5), ("TR04", "B", 2)])


@pytest.fixture
def panel(known):
    return panel_rows(known, 4, 2)


@pytest.fixture
def weeks():
    return weeks_table(4, missing={3})


@pytest.fixture
def weekly():
    return weekly_table([
        ("TR02-A", 1, 100.0, 0.0),
        ("TR02-A", 2, 0.0, -10.0),     # only a refund: gross 0
        ("TR02-A", 3, 50.0, 0.0),      # in the missing week
        ("TR04-B", 4, 30.0, -5.0),
        ("TR02-A", 5, 70.0, 0.0),      # after the last week (doctored: cannot happen)
        ("TR09-NEW", 1, 20.0, 0.0),    # a key not in the panel (new-customer cash)
    ])


@pytest.fixture
def labelled(panel, weekly, weeks):
    return add_labels(panel, weekly, weeks)


def row(table, key, origin, h):
    hit = table[(table["key"] == key) & (table["origin_week"] == origin) & (table["h"] == h)]
    assert len(hit) == 1, (key, origin, h, len(hit))
    return hit.iloc[0]


# --- the constants ----------------------------------------------------------------------

def test_status_constants():
    assert LABELLED == "labelled"
    assert MISSING_WEEK == "missing_week"
    assert FUTURE == "future"
    assert len({LABELLED, MISSING_WEEK, FUTURE}) == 3


def test_label_tolerance_is_one_kurus():
    assert LABEL_TOLERANCE == 0.01


# --- shape ------------------------------------------------------------------------------

def test_columns_are_panel_columns_then_labels(labelled):
    assert list(labelled.columns) == PANEL_ROW_COLUMNS + LABEL_COLUMNS


def test_row_count_unchanged(panel, labelled):
    assert len(labelled) == len(panel)


def test_panel_columns_and_row_order_kept(panel, labelled):
    pd.testing.assert_frame_equal(labelled[PANEL_ROW_COLUMNS], panel)


def test_index_is_zero_to_n_minus_one(labelled):
    assert list(labelled.index) == list(range(len(labelled)))


def test_index_reset_even_when_panel_index_is_not(panel, weekly, weeks):
    shifted = panel.set_axis(range(100, 100 + len(panel)))
    result = add_labels(shifted, weekly, weeks)
    assert list(result.index) == list(range(len(panel)))
    pd.testing.assert_frame_equal(result[PANEL_ROW_COLUMNS],
                                  panel.reset_index(drop=True))


def test_row_order_kept_when_panel_is_not_sorted(panel, weekly, weeks):
    shuffled = panel.sample(frac=1, random_state=3).reset_index(drop=True)
    result = add_labels(shuffled, weekly, weeks)
    pd.testing.assert_frame_equal(result[PANEL_ROW_COLUMNS], shuffled)


def test_extra_panel_columns_kept(panel, weekly, weeks):
    extra = panel.assign(note=[f"n{i}" for i in range(len(panel))])
    result = add_labels(extra, weekly, weeks)
    assert list(result.columns) == PANEL_ROW_COLUMNS + ["note"] + LABEL_COLUMNS
    assert result["note"].tolist() == extra["note"].tolist()


def test_amount_dtype_float64(labelled):
    assert labelled["amount"].dtype == "float64"


def test_paid_dtype_nullable_boolean(labelled):
    assert labelled["paid"].dtype == "boolean"


def test_label_status_holds_strings(labelled):
    assert pd.api.types.is_string_dtype(labelled["label_status"])
    assert set(labelled["label_status"]) <= {LABELLED, MISSING_WEEK, FUTURE}


def test_label_status_dtype_is_object(labelled):
    assert labelled["label_status"].dtype == object


def test_label_status_dtype_is_object_on_an_empty_panel(weekly, weeks):
    empty = panel_rows(known_table([("TR02", "LATE", 9)]), 4, 2)
    result = add_labels(empty, weekly, weeks)
    assert len(result) == 0
    assert result["label_status"].dtype == object


# --- label_status ------------------------------------------------------------------------

def test_label_status_by_hand(labelled):
    expected = {1: LABELLED, 2: LABELLED, 3: MISSING_WEEK, 4: LABELLED, 5: FUTURE, 6: FUTURE}
    for target, status in expected.items():
        got = set(labelled.loc[labelled["target_week"] == target, "label_status"])
        assert got == {status}, target


def test_status_counts_by_hand(labelled):
    # Rows (key, origin, h): A at origins 0-4, B at 2-4, h = 1, 2 -> 16 rows.
    # Targets: 3 -> A(1,2), A(2,1), B(2,1); 5, 6 -> A(3,2), A(4,1), A(4,2), B(3,2), B(4,1),
    # B(4,2).
    counts = labelled["label_status"].value_counts().to_dict()
    assert counts == {LABELLED: 7, MISSING_WEEK: 3, FUTURE: 6}


def test_target_equal_to_last_week_is_labelled(labelled):
    assert set(labelled.loc[labelled["target_week"] == 4, "label_status"]) == {LABELLED}


def test_target_one_after_last_week_is_future(labelled):
    assert set(labelled.loc[labelled["target_week"] == 5, "label_status"]) == {FUTURE}


def test_missing_last_week_is_missing_week_not_future(panel, weekly):
    weeks = weeks_table(4, missing={4})
    result = add_labels(panel, weekly, weeks)
    assert set(result.loc[result["target_week"] == 4, "label_status"]) == {MISSING_WEEK}
    assert set(result.loc[result["target_week"] == 5, "label_status"]) == {FUTURE}


def test_missing_and_future_cannot_both_hold_on_a_calendar(weeks):
    # "A missing week after the last week is FUTURE": the last week is weeks["week"].max(),
    # so every week the calendar flags missing is <= it. The two rules never meet on any
    # calendar; the overwrite order in add_labels is unobservable. Pinned here as a fact
    # about the inputs, so a change to how "last week" is defined resurfaces the question.
    assert (weeks.loc[weeks["is_missing"], "week"] <= weeks["week"].max()).all()


def calendar_message(weeks_list):
    return re.escape(f"target weeks {weeks_list} are not in the week calendar")


def test_target_week_inside_the_calendar_range_but_absent_raises(panel, weekly):
    # Calendar weeks 0-4 plus week 6 (week 5 absent): the last week is 6, so target 5 is
    # up to the last week but not in the calendar.
    weeks = pd.concat([weeks_table(4), weeks_table(6).iloc[[6]]], ignore_index=True)
    with pytest.raises(ValueError, match=calendar_message([5])):
        add_labels(panel, weekly, weeks)


def test_calendar_gap_error_lists_weeks_sorted(panel, weekly):
    # Calendar = weeks 0, 2 and 6: targets 1, 3, 4, 5 are absent and <= 6.
    weeks = weeks_table(6).iloc[[0, 2, 6]].reset_index(drop=True)
    with pytest.raises(ValueError, match=calendar_message([1, 3, 4, 5])):
        add_labels(panel, weekly, weeks)


def test_calendar_gap_error_lists_at_most_ten_weeks(weekly):
    # One key known from the start, origins 0-20, h = 1: targets 1 ... 21. The calendar has
    # only weeks 0 and 21, so targets 1 ... 20 (twenty weeks) are absent.
    panel = panel_rows(known_table([("TR02", "A", -5)]), 20, 1)
    weeks = weeks_table(21).iloc[[0, 21]].reset_index(drop=True)
    with pytest.raises(ValueError, match=calendar_message(list(range(1, 11)))):
        add_labels(panel, weekly, weeks)


def test_target_weeks_after_the_last_week_need_not_be_in_the_calendar(labelled):
    # The running calendar stops at week 4; targets 5 and 6 are not in it and are FUTURE.
    assert set(labelled.loc[labelled["target_week"] > 4, "label_status"]) == {FUTURE}
    assert set(labelled.loc[labelled["target_week"] > 4, "target_week"]) == {5, 6}


def test_calendar_gap_check_ignores_missing_status(panel, weekly):
    # A week flagged missing is still in the calendar: no error, it is MISSING_WEEK.
    result = add_labels(panel, weekly, weeks_table(4, missing={2}))
    assert set(result.loc[result["target_week"] == 2, "label_status"]) == {MISSING_WEEK}


def test_no_missing_week_gives_no_missing_rows(panel, weekly):
    result = add_labels(panel, weekly, weeks_table(4))
    assert MISSING_WEEK not in set(result["label_status"])


def test_every_week_missing(panel, weekly):
    result = add_labels(panel, weekly, weeks_table(4, missing=set(range(5))))
    within = result["target_week"] <= 4
    assert set(result.loc[within, "label_status"]) == {MISSING_WEEK}
    assert set(result.loc[~within, "label_status"]) == {FUTURE}
    assert result["amount"].isna().all()
    assert result["paid"].isna().all()


def test_status_depends_only_on_target_week(labelled):
    per_target = labelled.groupby("target_week")["label_status"].nunique()
    assert (per_target == 1).all()


# --- amount ------------------------------------------------------------------------------

def test_amount_is_gross_receipts_of_the_target_week(labelled):
    assert row(labelled, "TR02-A", 0, 1)["amount"] == 100.0
    assert row(labelled, "TR04-B", 3, 1)["amount"] == 30.0
    assert row(labelled, "TR04-B", 2, 2)["amount"] == 30.0


def test_same_target_week_same_amount_at_every_origin(labelled):
    # A's week-1 receipts at origin 0 h=1; A's week-2 at origin 0 h=2 and origin 1 h=1.
    assert row(labelled, "TR02-A", 0, 2)["amount"] == row(labelled, "TR02-A", 1, 1)["amount"]


def test_no_weekly_row_gives_zero(labelled):
    # A has no weekly row in week 4: paid nothing, a real zero.
    for origin, h in ((2, 2), (3, 1)):
        result = row(labelled, "TR02-A", origin, h)
        assert result["label_status"] == LABELLED
        assert result["amount"] == 0.0


def test_refund_only_week_gives_zero_not_the_refund(labelled):
    result = row(labelled, "TR02-A", 1, 1)  # target week 2: gross 0, refunds -10
    assert result["label_status"] == LABELLED
    assert result["amount"] == 0.0


def test_refunds_are_not_netted_from_amount(labelled):
    # B's week 4: gross 30, refunds -5 -> amount 30, not 25.
    assert row(labelled, "TR04-B", 3, 1)["amount"] == 30.0


def test_missing_week_amount_is_nan_even_with_a_weekly_row(labelled):
    result = row(labelled, "TR02-A", 2, 1)  # target week 3 has A's 50.0
    assert result["label_status"] == MISSING_WEEK
    assert np.isnan(result["amount"])


def test_future_amount_is_nan_even_with_a_weekly_row(labelled):
    result = row(labelled, "TR02-A", 4, 1)  # target week 5 has A's 70.0
    assert result["label_status"] == FUTURE
    assert np.isnan(result["amount"])


def test_labelled_rows_never_nan(labelled):
    assert labelled.loc[labelled["label_status"] == LABELLED, "amount"].notna().all()


def test_unlabelled_rows_always_nan(labelled):
    assert labelled.loc[labelled["label_status"] != LABELLED, "amount"].isna().all()


def test_amount_matches_by_key_not_by_customer_id(panel, weeks):
    # Same customer id under another company: not A's money.
    weekly = weekly_table([("TR05-A", 1, 999.0, 0.0)])
    result = add_labels(panel, weekly, weeks)
    assert row(result, "TR02-A", 0, 1)["amount"] == 0.0


def test_empty_weekly_gives_all_zero_labels(panel, weeks):
    result = add_labels(panel, weekly_table([]), weeks)
    labelled_rows = result[result["label_status"] == LABELLED]
    assert len(labelled_rows) == 7
    assert (labelled_rows["amount"] == 0.0).all()
    assert not labelled_rows["paid"].any()


def test_weekly_rows_outside_the_panel_add_no_rows(panel, weekly, weeks):
    extra = pd.concat([weekly, weekly_table([("TR02-A", 40, 5.0, 0.0),
                                             ("TR77-X", 2, 5.0, 0.0)])], ignore_index=True)
    result = add_labels(panel, extra, weeks)
    assert len(result) == len(panel)


# --- paid --------------------------------------------------------------------------------

def test_paid_true_iff_amount_positive_on_labelled_rows(labelled):
    rows = labelled[labelled["label_status"] == LABELLED]
    assert rows["paid"].notna().all()
    assert (rows["paid"].astype(bool) == (rows["amount"] > 0)).all()


def test_paid_by_hand(labelled):
    assert row(labelled, "TR02-A", 0, 1)["paid"]
    assert not row(labelled, "TR02-A", 1, 1)["paid"]   # refund only
    assert not row(labelled, "TR02-A", 3, 1)["paid"]   # no weekly row
    assert row(labelled, "TR04-B", 3, 1)["paid"]


def test_paid_counts_by_hand(labelled):
    rows = labelled[labelled["label_status"] == LABELLED]
    # Paid: A->1 (origin 0 h1), B->4 (origin 2 h2, origin 3 h1). Seven labelled rows.
    assert int(rows["paid"].sum()) == 3
    assert int((~rows["paid"]).sum()) == 4


def test_paid_is_na_on_missing_and_future_rows(labelled):
    rows = labelled[labelled["label_status"] != LABELLED]
    assert len(rows) == 9
    assert rows["paid"].isna().all()


def test_negative_gross_would_not_be_paid(panel, weeks):
    # gross_receipts is never negative from weekly_amounts; if it were, paid is False.
    weekly = weekly_table([("TR02-A", 1, -3.0, 0.0)])
    result = add_labels(panel, weekly, weeks)
    assert not row(result, "TR02-A", 0, 1)["paid"]


def test_tiny_positive_amount_is_paid(panel, weeks):
    weekly = weekly_table([("TR02-A", 1, 0.001, 0.0)])
    assert row(add_labels(panel, weekly, weeks), "TR02-A", 0, 1)["paid"]


# --- purity ------------------------------------------------------------------------------

def test_inputs_not_modified(panel, weekly, weeks):
    before = (panel.copy(), weekly.copy(), weeks.copy())
    add_labels(panel, weekly, weeks)
    for after, original in zip((panel, weekly, weeks), before):
        pd.testing.assert_frame_equal(after, original)


def test_deterministic(panel, weekly, weeks):
    pd.testing.assert_frame_equal(add_labels(panel, weekly, weeks),
                                  add_labels(panel, weekly, weeks))


def test_empty_panel(known, weekly, weeks):
    empty = panel_rows(known_table([("TR02", "LATE", 9)]), 4, 2)
    result = add_labels(empty, weekly, weeks)
    assert len(result) == 0
    assert list(result.columns) == PANEL_ROW_COLUMNS + LABEL_COLUMNS


# --- check_labels_within_receipts ----------------------------------------------------------

def labelled_by_hand(rows):
    """rows: (key, origin_week, target_week, label_status, amount)."""
    frame = pd.DataFrame(rows, columns=["key", "origin_week", "target_week", "label_status",
                                        "amount"])
    return frame.astype({"origin_week": "int64", "target_week": "int64", "amount": "float64"})


def exceed_message(count, pairs):
    return re.escape(f"for {count} (origin, target) weeks the labelled amounts add up to more "
                     f"than the target week's receipts (first: {pairs})")


def test_check_passes_on_add_labels_output(labelled, weekly):
    check_labels_within_receipts(labelled, weekly)


def test_check_passes_when_labels_equal_receipts():
    weekly = weekly_table([("TR02-A", 1, 60.0, 0.0), ("TR02-B", 1, 40.0, 0.0)])
    labelled = labelled_by_hand([("TR02-A", 0, 1, LABELLED, 60.0),
                                 ("TR02-B", 0, 1, LABELLED, 40.0)])
    check_labels_within_receipts(labelled, weekly)


def test_check_passes_when_labels_fall_short():
    # Short by new-customer cash: allowed.
    weekly = weekly_table([("TR02-A", 1, 60.0, 0.0), ("TR02-NEW", 1, 40.0, 0.0)])
    check_labels_within_receipts(labelled_by_hand([("TR02-A", 0, 1, LABELLED, 60.0)]), weekly)


def test_check_passes_within_tolerance():
    weekly = weekly_table([("TR02-A", 1, 100.0, 0.0)])
    check_labels_within_receipts(labelled_by_hand([("TR02-A", 0, 1, LABELLED, 100.005)]),
                                 weekly)


def test_check_raises_just_above_tolerance():
    weekly = weekly_table([("TR02-A", 1, 100.0, 0.0)])
    with pytest.raises(ValueError, match=exceed_message(1, [[0, 1]])):
        check_labels_within_receipts(labelled_by_hand([("TR02-A", 0, 1, LABELLED, 100.02)]),
                                     weekly)


def test_check_raises_when_a_key_is_counted_twice():
    weekly = weekly_table([("TR02-A", 1, 60.0, 0.0), ("TR02-B", 1, 40.0, 0.0)])
    labelled = labelled_by_hand([("TR02-A", 0, 1, LABELLED, 60.0),
                                 ("TR02-A", 0, 1, LABELLED, 60.0),
                                 ("TR02-B", 0, 1, LABELLED, 40.0)])
    with pytest.raises(ValueError, match=exceed_message(1, [[0, 1]])):
        check_labels_within_receipts(labelled, weekly)


def test_check_groups_per_origin_not_across_origins():
    # The same key at two origins for the same target week: each origin alone is fine.
    weekly = weekly_table([("TR02-A", 2, 100.0, 0.0)])
    labelled = labelled_by_hand([("TR02-A", 0, 2, LABELLED, 100.0),
                                 ("TR02-A", 1, 2, LABELLED, 100.0)])
    check_labels_within_receipts(labelled, weekly)


def test_check_target_week_without_weekly_rows_has_zero_receipts():
    weekly = weekly_table([("TR02-A", 1, 100.0, 0.0)])
    check_labels_within_receipts(labelled_by_hand([("TR02-A", 0, 3, LABELLED, 0.0)]), weekly)
    with pytest.raises(ValueError, match=exceed_message(1, [[0, 3]])):
        check_labels_within_receipts(labelled_by_hand([("TR02-A", 0, 3, LABELLED, 5.0)]),
                                     weekly)


def test_check_ignores_missing_and_future_rows():
    weekly = weekly_table([("TR02-A", 1, 10.0, 0.0)])
    labelled = labelled_by_hand([("TR02-A", 0, 1, MISSING_WEEK, 1e9),
                                 ("TR02-A", 0, 2, FUTURE, 1e9),
                                 ("TR02-A", 1, 1, LABELLED, 10.0)])
    check_labels_within_receipts(labelled, weekly)


def test_check_counts_and_lists_the_first_five_pairs():
    weekly = weekly_table([("TR02-A", 1, 1.0, 0.0)])
    rows = [("TR02-A", origin, origin + h, LABELLED, 50.0)
            for origin in range(4) for h in (1, 2)]
    labelled = labelled_by_hand(rows)
    # All eight pairs exceed ((0, 1) too: receipts 1.0 vs 50.0); the first five are listed
    # in groupby order (origin, then target).
    expected_pairs = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]
    with pytest.raises(ValueError, match=exceed_message(8, expected_pairs)):
        check_labels_within_receipts(labelled, weekly)


def test_check_refunds_do_not_count_as_receipts():
    # Week receipts are gross_receipts only: a refund does not raise or lower the bound.
    weekly = weekly_table([("TR02-A", 1, 50.0, -40.0)])
    check_labels_within_receipts(labelled_by_hand([("TR02-A", 0, 1, LABELLED, 50.0)]), weekly)


def test_check_does_not_modify_inputs(labelled, weekly):
    before = (labelled.copy(), weekly.copy())
    check_labels_within_receipts(labelled, weekly)
    pd.testing.assert_frame_equal(labelled, before[0])
    pd.testing.assert_frame_equal(weekly, before[1])


def test_add_labels_raises_on_a_duplicated_panel_row(panel, weekly, weeks):
    # A doctored panel with A's (origin 0, h 1) row twice: A's 100.0 counts twice in week 1
    # (week 1 receipts: 100 + 20 from the new key = 120 < 200).
    doubled = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match=exceed_message(1, [[0, 1]])):
        add_labels(doubled, weekly, weeks)


def repeated_message(count, pairs):
    return re.escape(f"the weekly amounts have {count} repeated (key, week) rows "
                     f"(first: {pairs})")


def test_add_labels_raises_on_a_duplicated_weekly_row(panel, weekly, weeks):
    # A (key, week) twice in `weekly` would duplicate panel rows in the merge and double the
    # week's receipts in the check (both sides grow together); it is rejected up front.
    doubled = pd.concat([weekly, weekly.iloc[[0]]], ignore_index=True)  # A, week 1, 100.0
    with pytest.raises(ValueError, match=repeated_message(1, [["TR02-A", 1]])):
        add_labels(panel, doubled, weeks)


def test_duplicated_weekly_row_raises_even_with_different_amounts(panel, weekly, weeks):
    extra = pd.concat([weekly, weekly_table([("TR04-B", 4, 1.0, 0.0)])], ignore_index=True)
    with pytest.raises(ValueError, match=repeated_message(1, [["TR04-B", 4]])):
        add_labels(panel, extra, weeks)


def test_duplicated_weekly_row_raises_outside_the_panel(panel, weekly, weeks):
    # Checked on `weekly` itself, before any merge: a key not in the panel counts too.
    extra = pd.concat([weekly, weekly_table([("TR09-NEW", 1, 5.0, 0.0)])], ignore_index=True)
    with pytest.raises(ValueError, match=repeated_message(1, [["TR09-NEW", 1]])):
        add_labels(panel, extra, weeks)


def test_duplicated_weekly_error_counts_and_lists_the_first_five(panel, weeks):
    # Seven repeats: weeks 1-6 of A once more each, and week 1 of A a third time.
    rows = [("TR02-A", week, 1.0, 0.0) for week in range(1, 7)]
    weekly = weekly_table(rows + rows + [("TR02-A", 1, 1.0, 0.0)])
    first = [["TR02-A", week] for week in range(1, 6)]
    with pytest.raises(ValueError, match=repeated_message(7, first)):
        add_labels(panel, weekly, weeks)


def test_same_week_of_two_keys_is_not_a_repeat(panel, weeks):
    weekly = weekly_table([("TR02-A", 1, 10.0, 0.0), ("TR04-B", 1, 20.0, 0.0)])
    result = add_labels(panel, weekly, weeks)
    assert row(result, "TR02-A", 0, 1)["amount"] == 10.0


def test_two_weeks_of_one_key_is_not_a_repeat(panel, weeks):
    weekly = weekly_table([("TR02-A", 1, 10.0, 0.0), ("TR02-A", 2, 20.0, 0.0)])
    result = add_labels(panel, weekly, weeks)
    assert row(result, "TR02-A", 0, 1)["amount"] == 10.0
    assert row(result, "TR02-A", 0, 2)["amount"] == 20.0


def test_same_customer_id_in_two_companies_same_week_is_not_a_repeat(panel, weeks):
    # The repeat is judged on the key, not on customer_id.
    weekly = weekly_table([("TR02-A", 1, 10.0, 0.0), ("TR05-A", 1, 20.0, 0.0)])
    assert row(add_labels(panel, weekly, weeks), "TR02-A", 0, 1)["amount"] == 10.0


def test_check_alone_misses_a_duplicated_weekly_row(labelled, weekly):
    # The duplication seen by the check alone (add_labels now rejects it before the check),
    # on a correctly labelled table: the doubled receipts only loosen the bound.
    doubled = pd.concat([weekly, weekly.iloc[[0]]], ignore_index=True)
    check_labels_within_receipts(labelled, doubled)


# --- integration with the synthetic corpus and ledger (SLOW) ---------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
HOLIDAYS_FILE = REPO_ROOT / "settings" / "holidays_tr.toml"
AS_OF = date(2026, 8, 28)
WEEK_ZERO = date(2025, 6, 2)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]
UNFILLED_MISSING_WEEKS = {14, 28, 35, 40, 46, 52}


@pytest.fixture(scope="module")
def synthetic_inputs():
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    from collection_estimation.hurdle.holidays import load_holidays
    from collection_estimation.hurdle.keys import add_keys
    from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
    from collection_estimation.hurdle.read_ledger import read_ledger
    from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

    daily_files = list_daily_files(CORPUS_FOLDER, SYNTHETIC_LAYOUT.file_name_date_format)
    # Direct reads, never the data/interim cache.
    collections = add_keys(read_all_files(daily_files, SYNTHETIC_LAYOUT), KNOWN_COMPANIES)
    ledger = add_keys(read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT),
                      KNOWN_COMPANIES)
    return list(daily_files), collections, ledger, load_holidays(HOLIDAYS_FILE)


def build_labelled(synthetic_inputs, fill):
    from collection_estimation.hurdle.fill_from_ledger import fill_missing_days
    from collection_estimation.hurdle.known import known_weeks
    from collection_estimation.hurdle.series import add_week_index, week_calendar, weekly_amounts

    days, collections, ledger, holidays = synthetic_inputs
    if fill:
        collections, filled_days = fill_missing_days(collections, ledger, days)
    else:
        collections, filled_days = collections.assign(filled_from_ledger=False), []
    collections = add_week_index(collections, WEEK_ZERO)
    weeks = week_calendar(days, WEEK_ZERO, holidays, filled_days)
    weekly = weekly_amounts(collections)
    known = known_weeks(collections, ledger, WEEK_ZERO)
    panel = panel_rows(known, 64, 5)
    return panel, weekly, weeks, add_labels(panel, weekly, weeks)


@pytest.fixture(scope="module")
def synthetic_filled(synthetic_inputs):
    return build_labelled(synthetic_inputs, fill=True)


@pytest.fixture(scope="module")
def synthetic_unfilled(synthetic_inputs):
    return build_labelled(synthetic_inputs, fill=False)


def test_slow_synthetic_filled_status_counts(synthetic_filled):
    panel, _, weeks, labelled = synthetic_filled
    assert int(weeks["week"].max()) == 64
    assert not weeks["is_missing"].any()
    assert len(labelled) == len(panel)
    counts = labelled["label_status"].value_counts().to_dict()
    assert counts.get(LABELLED, 0) == 1_090_890
    assert counts.get(FUTURE, 0) == 68_850
    assert counts.get(MISSING_WEEK, 0) == 0


def test_slow_synthetic_filled_future_rows_are_past_week_64(synthetic_filled):
    *_, labelled = synthetic_filled
    future = labelled["label_status"] == FUTURE
    assert (labelled.loc[future, "target_week"] > 64).all()
    assert (labelled.loc[~future, "target_week"] <= 64).all()


def test_slow_synthetic_filled_paid_count(synthetic_filled):
    *_, labelled = synthetic_filled
    rows = labelled[labelled["label_status"] == LABELLED]
    assert int(rows["paid"].sum()) == 99_975
    assert labelled.loc[labelled["label_status"] != LABELLED, "paid"].isna().all()


def test_slow_synthetic_filled_h1_share_of_receipts(synthetic_filled):
    _, weekly, _, labelled = synthetic_filled
    one = labelled[(labelled["label_status"] == LABELLED) & (labelled["h"] == 1)]
    in_panel = one.groupby("target_week")["amount"].sum()
    receipts = weekly.groupby("week")["gross_receipts"].sum().reindex(in_panel.index,
                                                                       fill_value=0.0)
    assert round(100 * in_panel.sum() / receipts.sum(), 2) == 99.97


def test_slow_synthetic_filled_panel_columns_and_order_kept(synthetic_filled):
    panel, _, _, labelled = synthetic_filled
    pd.testing.assert_frame_equal(labelled[PANEL_ROW_COLUMNS], panel)
    assert labelled["paid"].dtype == "boolean"
    assert labelled["amount"].dtype == "float64"


def test_slow_synthetic_unfilled_missing_weeks(synthetic_unfilled):
    _, _, weeks, labelled = synthetic_unfilled
    assert set(weeks.loc[weeks["is_missing"], "week"]) == UNFILLED_MISSING_WEEKS
    missing = labelled[labelled["label_status"] == MISSING_WEEK]
    assert len(missing) > 0
    assert set(missing["target_week"]) == UNFILLED_MISSING_WEEKS
    assert missing["amount"].isna().all()
    assert missing["paid"].isna().all()


def test_slow_synthetic_unfilled_no_labelled_row_in_a_missing_week(synthetic_unfilled):
    *_, labelled = synthetic_unfilled
    rows = labelled[labelled["label_status"] == LABELLED]
    assert not rows["target_week"].isin(UNFILLED_MISSING_WEEKS).any()


# --- add_deflated_label on the synthetic panel (SLOW) ----------------------------------
# The real TÜFE levels for weeks -32 to 64, as the pipeline builds them. The unfilled panel
# has missing-week rows as well as future ones, so both kinds of NaN are exercised.

@pytest.fixture(scope="module")
def synthetic_levels():
    from collection_estimation.hurdle.prices import published_levels
    return published_levels(WEEK_ZERO, -32, 64)


@pytest.mark.parametrize("which", ["synthetic_filled", "synthetic_unfilled"])
def test_slow_synthetic_deflated_label(which, synthetic_levels, request):
    from collection_estimation.hurdle.panel import add_deflated_label

    *_, labelled = request.getfixturevalue(which)
    before = labelled.copy()
    result = add_deflated_label(labelled, synthetic_levels)
    pd.testing.assert_frame_equal(labelled, before)
    pd.testing.assert_frame_equal(result.drop(columns="amount_deflated"), labelled)
    assert result["amount_deflated"].dtype == "float64"

    is_labelled = result["label_status"] == LABELLED
    assert int(result["amount_deflated"].isna().sum()) == int((~is_labelled).sum())
    assert result.loc[is_labelled, "amount_deflated"].notna().all()
    assert ((result["amount"] == 0.0) == (result["amount_deflated"] == 0.0)).all()

    # amount / amount_deflated is the target week's level, row by row and per target week.
    paying = result[is_labelled & (result["amount"] > 0)]
    ratio = paying["amount"] / paying["amount_deflated"]
    np.testing.assert_allclose(ratio.to_numpy(),
                               paying["target_week"].map(synthetic_levels).to_numpy(),
                               rtol=1e-12, atol=0.0)
    per_week = paying.groupby("target_week")[["amount", "amount_deflated"]].sum()
    np.testing.assert_allclose((per_week["amount"] / per_week["amount_deflated"]).to_numpy(),
                               per_week.index.map(synthetic_levels).to_numpy(),
                               rtol=1e-12, atol=0.0)
