"""Tests for U9b.5 in `hurdle/receivables_asof.py`: `end_of_week`, `ledger_as_known_at`,
`receivables_by_week` and `check_no_leak`.

`receivables_by_week` stacks `receivables_at` for weeks 0 .. last_week into one key x week
table, refuses a last week that starts after the ledger's extract date (`as_of`), notes one
that ends after it, and then rebuilds the weeks in `leak_check_weeks` from a ledger cut off
at that week (`check_no_leak`).

The small ledgers reuse the hand-built helpers of `test_receivables_at.py` (week zero is
Monday 2026-01-05). The SLOW test at the end reads the synthetic corpus and ledger in
`data/raw/` once and is skipped when they are absent; `data/interim/` is never used.
"""
import contextlib
import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import receivables_asof
from collection_estimation.hurdle.keys import add_keys
from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
from collection_estimation.hurdle.read_ledger import read_ledger
from collection_estimation.hurdle.receivables_asof import (
    check_no_leak,
    end_of_week,
    exchange_rates,
    ledger_as_known_at,
    receivables_at,
    receivables_by_week,
    receivables_columns,
)
from tests.hurdle.test_receivables_at import (
    AGE_COLUMNS,
    AGE_EDGES,
    EUR_RATES,
    WEEK_ZERO,
    invoice,
    ledger,
    monday,
    payment,
    sunday,
)
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

FAR_AS_OF = date(2027, 1, 1)   # an extract date after every week used in the small tests


def small_lines():
    """Two customers over six weeks, with a TRY and a EUR invoice, a partial payment and a
    clearing in week 5."""
    return [
        invoice("INV-1", 1000, "2026-01-07", due="2026-02-04"),                 # week 0
        payment("PAY-1", -300, "2026-01-20", reference="INV-1"),                # week 2
        invoice("INV-2", 100, "2025-12-01", due="2026-01-15", currency="EUR",
                cleared="2026-02-10", clearing_document="K2"),                  # cleared wk 5
        payment("PAY-2", -100, "2026-02-10", reference="INV-2", currency="EUR",
                cleared="2026-02-10", clearing_document="K2"),                  # week 5
        invoice("INV-3", 500, "2026-01-28", due="2026-03-01", customer="B"),    # week 3
    ]


def small_ledger():
    return ledger(small_lines())


def by_week(table=None, last_week=5, as_of=FAR_AS_OF, horizons=5, min_cleared=3,
            leak_check_weeks=(), rates=EUR_RATES, age_edges=AGE_EDGES):
    table = small_ledger() if table is None else table
    return receivables_by_week(table, rates, WEEK_ZERO, last_week, as_of, horizons,
                               min_cleared, leak_check_weeks, age_edges=age_edges)


def stacked_receivables_at(table, last_week, rates=EUR_RATES, horizons=5, min_cleared=3,
                           age_edges=AGE_EDGES):
    blocks = [receivables_at(table, week, WEEK_ZERO, rates, horizons, min_cleared, age_edges)
              for week in range(last_week + 1)]
    return pd.concat(blocks, ignore_index=True)


# --- end_of_week ---------------------------------------------------------------------

def test_end_of_week_0_is_week_zero_plus_6_days_a_sunday():
    assert end_of_week(WEEK_ZERO, 0) == WEEK_ZERO + timedelta(days=6)
    assert end_of_week(WEEK_ZERO, 0) == date(2026, 1, 11)
    assert end_of_week(WEEK_ZERO, 0).weekday() == 6


def test_end_of_week_3_is_week_zero_plus_27_days():
    assert end_of_week(WEEK_ZERO, 3) == WEEK_ZERO + timedelta(days=27)
    assert end_of_week(WEEK_ZERO, 3).weekday() == 6


def test_end_of_week_returns_a_date():
    assert type(end_of_week(WEEK_ZERO, 2)) is date


def test_end_of_week_on_the_synthetic_calendar():
    assert end_of_week(date(2025, 6, 2), 30) == date(2026, 1, 4)
    assert end_of_week(date(2025, 6, 2), 64) == date(2026, 8, 30)


def test_receivables_at_uses_end_of_week_as_its_cutoff():
    # A line posted on Sunday of week 2 counts in week 2; one posted on Monday of week 3 not.
    table = ledger([invoice("ON", 10, sunday(2), due=sunday(9)),
                    invoice("AFTER", 20, monday(3), due=sunday(9), customer="B")])
    result = receivables_at(table, 2, WEEK_ZERO, EUR_RATES, 5, 3, AGE_EDGES)
    assert result["key"].tolist() == ["TR02-A"]


# --- ledger_as_known_at --------------------------------------------------------------

CUT = date(2026, 1, 25)   # sunday(2)


def known_at_lines():
    return [
        invoice("POSTED-BEFORE", 100, "2026-01-10"),
        invoice("POSTED-ON", 200, "2026-01-25", customer="B"),
        invoice("POSTED-AFTER", 300, "2026-01-26", customer="C"),
        invoice("CLEARED-AFTER", 400, "2026-01-12", cleared="2026-02-03",
                clearing_document="K-AFTER"),
        invoice("CLEARED-ON", 500, "2026-01-12", cleared="2026-01-25",
                clearing_document="K-ON"),
        invoice("CLEARED-BEFORE", 600, "2026-01-12", cleared="2026-01-20",
                clearing_document="K-BEFORE"),
    ]


def known_at():
    return ledger_as_known_at(ledger(known_at_lines()), CUT)


def row(table, document_no):
    rows = table[table["document_no"] == document_no]
    assert len(rows) == 1, table
    return rows.iloc[0]


def test_known_at_drops_lines_posted_after_the_cutoff():
    assert "POSTED-AFTER" not in set(known_at()["document_no"])


def test_known_at_keeps_lines_posted_on_and_before_the_cutoff():
    documents = set(known_at()["document_no"])
    assert {"POSTED-BEFORE", "POSTED-ON"} <= documents
    assert len(documents) == 5


def test_known_at_undoes_a_clearing_after_the_cutoff():
    line = row(known_at(), "CLEARED-AFTER")
    assert pd.isna(line["clearing_date"])
    assert line["clearing_document"] is None


def test_known_at_keeps_a_clearing_on_the_cutoff():
    line = row(known_at(), "CLEARED-ON")
    assert line["clearing_date"] == pd.Timestamp("2026-01-25")
    assert line["clearing_document"] == "K-ON"


def test_known_at_keeps_a_clearing_before_the_cutoff():
    line = row(known_at(), "CLEARED-BEFORE")
    assert line["clearing_date"] == pd.Timestamp("2026-01-20")
    assert line["clearing_document"] == "K-BEFORE"


def test_known_at_leaves_open_lines_open():
    line = row(known_at(), "POSTED-BEFORE")
    assert pd.isna(line["clearing_date"])
    assert pd.isna(line["clearing_document"])


def test_known_at_does_not_modify_its_input():
    original = ledger(known_at_lines())
    copy = original.copy(deep=True)
    ledger_as_known_at(original, CUT)
    pd.testing.assert_frame_equal(original, copy)


def test_known_at_keeps_columns_dtypes_and_other_values():
    original = ledger(known_at_lines())
    result = ledger_as_known_at(original, CUT)
    assert list(result.columns) == list(original.columns)
    assert result.dtypes.equals(original.dtypes)
    kept = original[original["posting_date"] <= pd.Timestamp(CUT)]
    untouched = [c for c in original.columns if c not in ("clearing_date", "clearing_document")]
    pd.testing.assert_frame_equal(result[untouched], kept[untouched])
    # The original row labels are kept, so a line can be traced back.
    assert result.index.tolist() == kept.index.tolist()


def test_known_at_a_cutoff_before_every_line_gives_an_empty_table_with_the_columns():
    original = ledger(known_at_lines())
    result = ledger_as_known_at(original, date(2025, 1, 1))
    assert len(result) == 0
    assert list(result.columns) == list(original.columns)


def test_known_at_a_cutoff_after_everything_changes_nothing():
    original = ledger(known_at_lines())
    pd.testing.assert_frame_equal(ledger_as_known_at(original, date(2030, 1, 1)), original)


# --- receivables_by_week: the stacked table ------------------------------------------

def test_by_week_equals_receivables_at_stacked_week_by_week():
    pd.testing.assert_frame_equal(by_week(), stacked_receivables_at(small_ledger(), 5))


def test_by_week_has_one_block_per_week_0_to_last_week():
    result = by_week(last_week=5)
    assert sorted(result["week"].unique()) == [0, 1, 2, 3, 4, 5]
    assert result["week"].is_monotonic_increasing


def test_by_week_key_and_week_are_unique_together():
    result = by_week()
    assert not result.duplicated(["key", "week"]).any()


def test_by_week_index_is_0_to_n_minus_1():
    result = by_week()
    assert result.index.tolist() == list(range(len(result)))


@pytest.mark.parametrize("horizons", [1, 3, 5])
def test_by_week_columns_are_receivables_columns(horizons):
    assert list(by_week(horizons=horizons).columns) == receivables_columns(horizons, AGE_EDGES)


def test_by_week_last_week_0_gives_week_0_only():
    result = by_week(last_week=0)
    pd.testing.assert_frame_equal(result, receivables_at(small_ledger(), 0, WEEK_ZERO,
                                                         EUR_RATES, 5, 3, AGE_EDGES))


def test_by_week_follows_the_ledger_over_time():
    result = by_week().set_index(["key", "week"])
    # Customer B appears in week 3, when INV-3 is posted.
    assert ("TR02-B", 2) not in result.index
    assert result.loc[("TR02-B", 3), "open_invoices"] == 1
    # A: INV-1 and INV-2 open until week 5, when INV-2 is cleared.
    assert result.loc[("TR02-A", 4), "open_invoices"] == 2
    assert result.loc[("TR02-A", 5), "open_invoices"] == 1
    assert result.loc[("TR02-A", 5), "cleared_invoices"] == 1


def test_by_week_an_empty_first_week_keeps_the_dtypes():
    # Nothing is posted until week 2: weeks 0 and 1 have no rows.
    table = ledger([invoice("LATE", 100, monday(2), due=sunday(4))])
    result = by_week(table, last_week=3)
    assert sorted(result["week"].unique()) == [2, 3]
    assert result["key"].dtype == object
    for column in ["week", "open_invoices", "cleared_invoices"]:
        assert result[column].dtype == "int64", column
    for column in ["due_in_week_1", "overdue", *AGE_COLUMNS, "open_balance",
                   "usual_delay_days"]:
        assert result[column].dtype == "float64", column


def test_by_week_does_not_modify_its_inputs():
    table = small_ledger()
    table_copy = table.copy(deep=True)
    rates = EUR_RATES.copy(deep=True)
    by_week(table, rates=rates, leak_check_weeks=(0, 3, 5))
    pd.testing.assert_frame_equal(table, table_copy)
    pd.testing.assert_frame_equal(rates, EUR_RATES)


# --- receivables_by_week: progress lines ---------------------------------------------

def test_by_week_prints_no_progress_for_fewer_than_10_weeks(capsys):
    by_week(last_week=8)   # 9 weeks
    assert "receivables for" not in capsys.readouterr().out


def test_by_week_prints_progress_at_10_weeks(capsys):
    by_week(last_week=9)   # exactly 10 weeks
    out = capsys.readouterr().out
    assert "  ... receivables for 10 of 10 weeks" in out
    assert out.count("receivables for") == 1


def test_by_week_prints_progress_every_10_weeks(capsys):
    by_week(last_week=24)  # 25 weeks
    lines = [line for line in capsys.readouterr().out.splitlines()
             if "receivables for" in line]
    assert lines == ["  ... receivables for 10 of 25 weeks",
                     "  ... receivables for 20 of 25 weeks"]


# --- receivables_by_week: the as_of rule ----------------------------------------------

def test_a_last_week_starting_after_as_of_raises_naming_week_monday_and_as_of():
    as_of = monday(5) - timedelta(days=1)   # the Sunday before week 5
    with pytest.raises(ValueError) as error:
        by_week(last_week=5, as_of=as_of)
    message = str(error.value)
    assert "week 5" in message
    assert str(monday(5)) in message
    assert str(as_of) in message


def test_a_last_week_starting_long_after_as_of_raises():
    with pytest.raises(ValueError, match="too old"):
        by_week(last_week=5, as_of=WEEK_ZERO)


def test_a_last_week_starting_on_as_of_prints_a_note(capsys):
    as_of = monday(5)
    by_week(last_week=5, as_of=as_of)
    out = capsys.readouterr().out
    assert (f"Note: week 5 ends on {sunday(5)}, after the ledger's extract date {as_of}"
            in out)


def test_a_last_week_ending_after_as_of_still_gives_every_week(capsys):
    as_of = sunday(5) - timedelta(days=2)    # Friday of week 5
    result = by_week(last_week=5, as_of=as_of)
    assert "Note: week 5 ends on" in capsys.readouterr().out
    pd.testing.assert_frame_equal(result, stacked_receivables_at(small_ledger(), 5))


def test_a_last_week_ending_on_as_of_prints_no_note(capsys):
    by_week(last_week=5, as_of=sunday(5))
    assert "Note: week" not in capsys.readouterr().out


def test_a_last_week_ending_before_as_of_prints_no_note(capsys):
    by_week(last_week=5, as_of=FAR_AS_OF)
    assert "Note: week" not in capsys.readouterr().out


def test_the_as_of_rule_is_checked_before_any_week_is_built(monkeypatch):
    calls = []
    monkeypatch.setattr(receivables_asof, "receivables_at",
                        lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError):
        by_week(last_week=5, as_of=WEEK_ZERO)
    assert calls == []


# --- check_no_leak --------------------------------------------------------------------

def leak_check(table, stored, weeks, last_week=5):
    check_no_leak(stored, table, EUR_RATES, weeks, last_week, WEEK_ZERO, 5, 3, AGE_EDGES)


def test_check_no_leak_passes_on_a_correct_table():
    table = small_ledger()
    leak_check(table, stacked_receivables_at(table, 5), (0, 1, 2, 3, 4, 5))


def test_by_week_runs_the_leak_check_on_every_named_week():
    by_week(leak_check_weeks=(0, 2, 5))   # no error


def test_by_week_calls_check_no_leak_with_its_table_and_weeks(monkeypatch):
    seen = {}

    def spy(stored, table, rates, weeks, last_week, week_zero, horizons, min_cleared,
            age_edges):
        seen.update(stored=stored, weeks=weeks, last_week=last_week, week_zero=week_zero,
                    horizons=horizons, min_cleared=min_cleared, age_edges=age_edges)

    monkeypatch.setattr(receivables_asof, "check_no_leak", spy)
    result = by_week(last_week=5, horizons=3, min_cleared=2, leak_check_weeks=(1, 4),
                     age_edges=(15, 45, 120))
    assert seen["stored"] is result
    assert seen["age_edges"] == (15, 45, 120)
    assert seen["weeks"] == (1, 4)
    assert seen["last_week"] == 5
    assert (seen["week_zero"], seen["horizons"], seen["min_cleared"]) == (WEEK_ZERO, 3, 2)


def test_check_no_leak_with_an_empty_week_list_checks_nothing():
    table = small_ledger()
    stored = stacked_receivables_at(table, 5)
    stored.loc[0, "due_in_week_1"] += 1.0   # wrong, but not checked
    leak_check(table, stored, ())
    leak_check(table, stored.iloc[0:0], ())   # not even an empty table is looked at


def test_check_no_leak_catches_an_altered_value_naming_the_week_and_cutoff():
    table = small_ledger()
    stored = stacked_receivables_at(table, 5)
    altered = stored.index[(stored["week"] == 3) & (stored["key"] == "TR02-A")][0]
    stored.loc[altered, "due_in_week_1"] += 0.01
    with pytest.raises(ValueError) as error:
        leak_check(table, stored, (3,))
    message = str(error.value)
    assert "week 3" in message
    assert str(sunday(3)) in message
    # The other weeks still pass.
    leak_check(table, stored, (0, 1, 2, 4, 5))


def test_check_no_leak_catches_a_missing_row():
    table = small_ledger()
    stored = stacked_receivables_at(table, 5)
    stored = stored.drop(stored.index[(stored["week"] == 4) & (stored["key"] == "TR02-B")])
    with pytest.raises(ValueError, match="week 4"):
        leak_check(table, stored, (4,))


def test_check_no_leak_raises_for_a_week_after_last_week():
    table = small_ledger()
    stored = stacked_receivables_at(table, 3)
    with pytest.raises(ValueError,
                       match=r"leak check: week 7 is outside the weeks built \(0 to 3\)"):
        leak_check(table, stored, (0, 7), last_week=3)


def test_check_no_leak_raises_for_last_week_plus_one():
    table = small_ledger()
    stored = stacked_receivables_at(table, 3)
    with pytest.raises(ValueError,
                       match=r"leak check: week 4 is outside the weeks built \(0 to 3\)"):
        leak_check(table, stored, (4,), last_week=3)


def test_check_no_leak_raises_for_a_negative_week():
    table = small_ledger()
    stored = stacked_receivables_at(table, 5)
    with pytest.raises(ValueError,
                       match=r"leak check: week -1 is outside the weeks built \(0 to 5\)"):
        leak_check(table, stored, (-1,))


def test_check_no_leak_accepts_week_0_and_last_week_at_the_bounds():
    table = small_ledger()
    leak_check(table, stacked_receivables_at(table, 3), (0, 3), last_week=3)   # no error


def test_by_week_raises_for_a_leak_check_week_after_last_week():
    with pytest.raises(ValueError,
                       match=r"leak check: week 6 is outside the weeks built \(0 to 5\)"):
        by_week(last_week=5, leak_check_weeks=(6,))


def test_by_week_raises_for_a_negative_leak_check_week():
    with pytest.raises(ValueError,
                       match=r"leak check: week -2 is outside the weeks built \(0 to 5\)"):
        by_week(last_week=5, leak_check_weeks=(-2,))


def late_ledger():
    """A ledger whose first line is posted in week 4: weeks 0 to 3 have no rows."""
    return ledger([invoice("LATE", 100, monday(4), due=sunday(6)),
                   invoice("LATE-B", 50, sunday(4), due=sunday(7), customer="B")])


def test_check_no_leak_passes_on_empty_weeks_inside_the_range():
    """A week before any ledger line has no rows; it is rebuilt as an empty table and
    passes, alongside a non-empty week."""
    table = late_ledger()
    stored = stacked_receivables_at(table, 5)
    assert not {0, 1, 2, 3} & set(stored["week"])
    assert 4 in set(stored["week"])
    leak_check(table, stored, (0, 2, 4))   # no error


def test_by_week_leak_checks_empty_weeks_inside_the_range():
    result = by_week(late_ledger(), last_week=5, leak_check_weeks=(0, 2, 5))
    assert sorted(result["week"].unique()) == [4, 5]


def test_check_no_leak_still_catches_a_row_planted_in_an_empty_week():
    """An empty week is compared, not skipped: a stored row where the rebuild has none
    is a difference."""
    table = late_ledger()
    stored = stacked_receivables_at(table, 5)
    planted = stored[stored["week"] == 4].iloc[[0]].copy()
    planted["week"] = 2
    stored = pd.concat([stored, planted], ignore_index=True)
    with pytest.raises(ValueError, match="leak check: week 2 differs"):
        leak_check(table, stored, (2,))


def with_a_future_invoice():
    """The small ledger plus an invoice for customer A posted in week 6."""
    return ledger(small_lines() + [invoice("FUTURE", 777, monday(6), due=sunday(8))])


def cutoff_blind(original):
    """A broken `open_invoices_at` that ignores its cutoff and sees the whole ledger."""
    return lambda ledger_lines, cutoff: original(ledger_lines, date(2099, 1, 1))


def test_check_no_leak_catches_a_genuinely_leaky_build(monkeypatch):
    """The cutoff-blind build puts week 6's invoice into week 1. The rebuild, from a ledger
    cut off at week 1, cannot see it, so the leak check fails."""
    monkeypatch.setattr(receivables_asof, "open_invoices_at",
                        cutoff_blind(receivables_asof.open_invoices_at))
    with pytest.raises(ValueError, match="leak check: week 1 differs"):
        by_week(with_a_future_invoice(), last_week=6, leak_check_weeks=(1,))


def test_the_same_leaky_build_passes_when_only_the_last_week_is_checked(monkeypatch):
    """The leak check only sees the weeks it is told to check: nothing comes after the last
    week, so a cutoff-blind build looks fine there."""
    monkeypatch.setattr(receivables_asof, "open_invoices_at",
                        cutoff_blind(receivables_asof.open_invoices_at))
    by_week(with_a_future_invoice(), last_week=6, leak_check_weeks=(6,))   # no error


# --- U9c.1: the aging buckets over the weeks ----------------------------------------

def aging_ledger():
    """One TRY invoice due Wednesday 2026-01-07 (week 0), never paid: at the end of week t it
    is 4 + 7t days past due. Plus one for B due Wednesday 2026-01-28 (week 3), half paid in
    week 1."""
    return ledger([invoice("OLD", 100, "2025-12-01", due="2026-01-07"),
                   invoice("B1", 60, "2026-01-02", due="2026-01-28", customer="B"),
                   payment("BP", -30, "2026-01-13", reference="B1", customer="B")])


def test_by_week_an_unpaid_invoice_moves_through_the_buckets():
    result = by_week(aging_ledger(), last_week=14).set_index(["key", "week"])
    for week in range(15):
        days = 4 + 7 * week
        expected = ("overdue_up_to_30" if days <= 30 else
                    "overdue_31_to_90" if days <= 90 else "overdue_over_90")
        row = result.loc[("TR02-A", week)]
        for column in AGE_COLUMNS:
            assert row[column] == (100.0 if column == expected else 0.0), (week, column)


def test_by_week_buckets_hold_the_outstanding_after_a_partial_payment():
    result = by_week(aging_ledger(), last_week=8).set_index(["key", "week"])
    assert result.loc[("TR02-B", 2), "overdue"] == 0.0            # not yet due
    assert result.loc[("TR02-B", 3), "overdue_up_to_30"] == 30.0  # due in week 3, half paid
    # Week 6 ends Sunday 2026-02-22: 25 days; week 7 ends 2026-03-01: 32 days.
    assert sunday(6) == date(2026, 2, 22) and sunday(7) == date(2026, 3, 1)
    assert result.loc[("TR02-B", 6), "overdue_up_to_30"] == 30.0
    assert result.loc[("TR02-B", 7), "overdue_31_to_90"] == 30.0
    assert result.loc[("TR02-B", 7), "overdue_up_to_30"] == 0.0


def test_by_week_passes_its_age_edges_on():
    result = by_week(aging_ledger(), last_week=3, age_edges=(10, 20))
    assert list(result.columns) == receivables_columns(5, (10, 20))
    row = result.set_index(["key", "week"]).loc[("TR02-A", 2)]   # 18 days
    assert row["overdue_11_to_20"] == 100.0


def test_by_week_buckets_add_up_to_overdue_in_every_row():
    result = by_week(aging_ledger(), last_week=20)
    assert ((result[AGE_COLUMNS].sum(axis=1) - result["overdue"]).abs() <= 1e-9).all()


def test_the_leak_check_passes_with_the_buckets_on_every_week():
    by_week(aging_ledger(), last_week=14, leak_check_weeks=tuple(range(15)))   # no error
    by_week(small_ledger(), last_week=5, leak_check_weeks=(0, 1, 2, 3, 4, 5),
            age_edges=(7, 14, 21))   # no error


def test_check_no_leak_catches_an_altered_bucket():
    table = aging_ledger()
    stored = stacked_receivables_at(table, 6)
    altered = stored.index[(stored["week"] == 5) & (stored["key"] == "TR02-A")][0]
    stored.loc[altered, "overdue_31_to_90"] += 0.01
    with pytest.raises(ValueError, match="leak check: week 5 differs"):
        check_no_leak(stored, table, EUR_RATES, (5,), 6, WEEK_ZERO, 5, 3, AGE_EDGES)


def test_check_no_leak_rebuilds_with_its_own_age_edges():
    """A table built with other edges has other columns: the rebuild does not match it."""
    table = aging_ledger()
    stored = stacked_receivables_at(table, 3, age_edges=(10, 20))
    with pytest.raises(ValueError, match="leak check: week 2 differs"):
        check_no_leak(stored, table, EUR_RATES, (2,), 3, WEEK_ZERO, 5, 3, AGE_EDGES)


@pytest.mark.parametrize("bad", [(90, 30), (30, 30), (0, 30), (30.0,), ()],
                         ids=["reversed", "repeated", "zero", "float", "empty"])
def test_by_week_refuses_bad_edges_with_the_exact_message(bad):
    with pytest.raises(ValueError) as error:
        by_week(last_week=3, age_edges=bad)
    assert str(error.value) == (f"age_edges are {bad}; give whole numbers of days (1 or more), "
                                f"from smallest to largest, each once")


def test_by_week_refuses_bad_edges_with_a_leak_check_too():
    with pytest.raises(ValueError, match=r"age_edges are \(30, 30\);"):
        by_week(last_week=3, leak_check_weeks=(1, 2), age_edges=(30, 30))


def test_by_week_accepts_numpy_integer_edges():
    plain = by_week(aging_ledger(), last_week=6, leak_check_weeks=(3,))
    numpy_edges = by_week(aging_ledger(), last_week=6, leak_check_weeks=(3,),
                          age_edges=(np.int64(30), np.int64(90)))
    pd.testing.assert_frame_equal(plain, numpy_edges)


def test_receivables_by_week_requires_age_edges():
    with pytest.raises(TypeError):
        receivables_by_week(small_ledger(), EUR_RATES, WEEK_ZERO, 5, FAR_AS_OF, 5, 3, ())


# --- integration with the synthetic corpus and ledger (SLOW, ~20 s) -------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
AS_OF = date(2026, 8, 28)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]
SYNTHETIC_WEEK_ZERO = date(2025, 6, 2)


@pytest.fixture(scope="module")
def synthetic_by_week():
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    daily_files = list_daily_files(CORPUS_FOLDER, SYNTHETIC_LAYOUT.file_name_date_format)
    collections = read_all_files(daily_files, SYNTHETIC_LAYOUT)  # direct read, no cache
    ledger_lines = read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT)  # direct, no cache
    keyed = add_keys(ledger_lines, KNOWN_COMPANIES)
    rates = exchange_rates(collections, keyed)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        table = receivables_by_week(keyed, rates, SYNTHETIC_WEEK_ZERO, last_week=64,
                                    as_of=AS_OF, horizons=5, min_cleared=3,
                                    leak_check_weeks=(0, 30, 64), age_edges=AGE_EDGES)
    return keyed, rates, table, printed.getvalue()


def test_slow_by_week_synthetic_has_231948_rows_and_4590_keys(synthetic_by_week):
    _, _, table, _ = synthetic_by_week
    assert len(table) == 231_948
    assert table["key"].nunique() == 4_590
    assert sorted(table["week"].unique()) == list(range(65))
    assert not table.duplicated(["key", "week"]).any()
    assert list(table.columns) == receivables_columns(5, AGE_EDGES)
    assert table.index.tolist() == list(range(len(table)))


def test_slow_by_week_synthetic_prints_the_week_64_note(synthetic_by_week):
    _, _, _, printed = synthetic_by_week
    assert ("Note: week 64 ends on 2026-08-30, after the ledger's extract date 2026-08-28"
            in printed)
    assert "  ... receivables for 60 of 65 weeks" in printed


def test_slow_by_week_synthetic_week_30_equals_receivables_at(synthetic_by_week):
    keyed, rates, table, _ = synthetic_by_week
    week_30 = table[table["week"] == 30].reset_index(drop=True)
    expected = receivables_at(keyed, 30, SYNTHETIC_WEEK_ZERO, rates, 5, 3, AGE_EDGES)
    assert len(week_30) == 3_548
    pd.testing.assert_frame_equal(week_30, expected)


def test_slow_by_week_synthetic_buckets_add_up_to_overdue(synthetic_by_week):
    _, _, table, _ = synthetic_by_week
    assert (table[AGE_COLUMNS].sum(axis=1) - table["overdue"]).abs().max() <= 0.01
    assert (table[AGE_COLUMNS] >= 0).all().all()
