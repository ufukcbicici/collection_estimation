"""Tests for U18.2, `hurdle/features_rhythm.py`: the delay spread of the open invoices.

Each invoice open at the end of week t (its Sunday) is spread over the weeks after t by the
customer's past delays (days late of its most recent cleared invoices known by t): arrival =
net due date + delay, moved to the next business day; only arrivals after t are possible and
share the outstanding amount equally. The single shifted date is due + the median delay
(half to even), moved the same way; one not after t goes to shifted_past and counts in no week.

The running example of the module docstring is pinned here: 100 TRY due Monday 2 March 2026,
delays 3, 5, 6, 12, 20, forecast at the end of that week (Sunday 8 March). Week zero is
Monday 2026-01-05, so 2-8 March is week 8. Arrivals Thu 5 (impossible), Mon 9, Mon 9, Mon 16,
Mon 23: 50 / 25 / 25 in weeks t+1, t+2, t+3.

Ledgers and rate tables are built with the builders of test_receivables_at.py. The holiday
table used by default has no holiday in the first months of the year; a few tests add fixed
holidays to test the move to the next business day. The SLOW tests at the end read the
synthetic ledger in `data/raw/` once and are skipped when it is absent; `data/interim/` is
never used.
"""
import contextlib
import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import features_rhythm as rhythm
from collection_estimation.hurdle.features_rhythm import (
    CALENDAR_MARGIN_DAYS,
    EXPECTED_DUE_COLUMNS,
    SPREAD_TOLERANCE,
    add_per_key,
    amounts_per_week,
    check_spread_adds_up,
    days_since_week_zero,
    delay_samples_at,
    delay_statistics,
    expected_dues,
    invoice_spread_at,
    key_spread_columns,
    next_business_days,
    open_invoices_in_try,
    shifted_columns,
    spread_columns,
    to_next_business_day,
    week_column,
)
from collection_estimation.hurdle.holidays import HolidayTable
from collection_estimation.hurdle.receivables_asof import (
    cleared_invoice_delays,
    ledger_as_known_at,
)
from tests.hurdle.test_receivables_at import invoice, ledger, payment, rates_table

WEEK_ZERO = date(2026, 1, 5)   # a Monday
ORIGIN = 8                     # Monday 2 March to Sunday 8 March 2026
CUTOFF = date(2026, 3, 8)
NO_RATES = rates_table([])
KEY = "TR02-A"


def bayrams(years=(2025, 2026, 2027)):
    """Both Bayrams of each year, in June and August (made up: none near the test dates)."""
    found = []
    for year in years:
        found += [(date(year, 6, 10), date(year, 6, 12), "Ramazan Bayramı"),
                  (date(year, 8, 20), date(year, 8, 23), "Kurban Bayramı")]
    return tuple(found)


NO_HOLIDAYS = HolidayTable(fixed={}, bayrams=bayrams())


def holidays_on(*month_days):
    """A table with fixed holidays on the given (month, day)s."""
    return HolidayTable(fixed={md: f"holiday {md}" for md in month_days}, bayrams=bayrams())


def business(holidays=NO_HOLIDAYS):
    """{day: next business day} from 1 Sep 2025 to 31 Dec 2026, days since WEEK_ZERO."""
    return next_business_days(date(2025, 9, 1), date(2026, 12, 31), holidays, WEEK_ZERO)


def day(text):
    """A date as days since WEEK_ZERO (Monday 5 January 2026)."""
    return int((date.fromisoformat(text) - WEEK_ZERO).days)


# Every day of the calendar a business day: arrivals are then never moved (boundary tests).
EVERY_DAY = {d: d for d in range(day("2025-09-01"), day("2026-12-31") + 1)}


def history(delays, customer="A", company="TR02", start="2025-11-03", step=14, tag="H",
            clearing_lag=0):
    """One cleared invoice per entry of `delays`, due `start` + step * i, paid `delay` days
    after its due date, cleared `clearing_lag` days after the payment. Arrivals are in the
    order of `delays` when step is larger than the spread of the delays."""
    lines = []
    for i, late in enumerate(delays):
        due = pd.Timestamp(start) + pd.Timedelta(days=step * i)
        paid = due + pd.Timedelta(days=late)
        cleared = paid + pd.Timedelta(days=clearing_lag)
        number = f"{tag}-{company}-{customer}-{i:03d}"
        group = f"CLR-{number}"
        posted = min(due, paid) - pd.Timedelta(days=30)
        lines.append(invoice(number, 10.0, str(posted.date()), str(due.date()),
                             cleared=str(cleared.date()), clearing_document=group,
                             customer=customer, company=company))
        lines.append(payment(f"P{number}", -10.0, str(paid.date()), reference=number,
                             cleared=str(cleared.date()), clearing_document=group,
                             customer=customer, company=company))
    return lines


EXAMPLE_DELAYS = [3, 5, 6, 12, 20]


def example_lines(amount=100.0, due="2026-03-02", **kwargs):
    return ([invoice("OPEN", amount, "2026-01-10", due, **kwargs)]
            + history(EXAMPLE_DELAYS))


def keys_of(*pairs):
    """keys from ("TR02-A", ...) or (key, company) pairs."""
    rows = [(p, p[:4]) if isinstance(p, str) else p for p in pairs]
    return pd.DataFrame({"key": [r[0] for r in rows], "company_code": [r[1] for r in rows]})


def spread(lines, keys=None, origin=ORIGIN, horizons=5, min_cleared=3, delays_kept=20,
           company_delays_kept=200, holidays=NO_HOLIDAYS, next_business=None, rates=NO_RATES):
    table = ledger(lines)
    return invoice_spread_at(table, cleared_invoice_delays(table), rates,
                             keys if keys is not None else keys_of(KEY), origin, WEEK_ZERO,
                             business(holidays) if next_business is None else next_business,
                             horizons, min_cleared, delays_kept, company_delays_kept)


def row(table, key=KEY):
    found = table[table["key"] == key]
    assert len(found) == 1
    return found.iloc[0]


def weeks(r, horizons=5, kind="spread"):
    return [float(r[f"{kind}_week_{k}"]) for k in range(1, horizons + 1)]


def panel(keys=(KEY,), origins=(ORIGIN,), horizons=5, companies=None):
    rows = []
    for origin in origins:
        for key in keys:
            for h in range(1, horizons + 1):
                company = (companies or {}).get(key, key[:4])
                rows.append({"key": key, "company_code": company, "origin_week": origin,
                             "h": h})
    return pd.DataFrame(rows)


def dues(lines, panel_rows=None, levels=None, horizons=5, min_cleared=3, delays_kept=20,
         company_delays_kept=200, leak_check_weeks=(), holidays=NO_HOLIDAYS, rates=NO_RATES):
    panel_rows = panel(horizons=horizons) if panel_rows is None else panel_rows
    if levels is None:
        levels = pd.Series({week: 1.0 for week in set(panel_rows["origin_week"])})
    return expected_dues(panel_rows, ledger(lines), rates, levels, WEEK_ZERO, holidays,
                         horizons, min_cleared, delays_kept, company_delays_kept,
                         leak_check_weeks)


# --- the column helpers and constants -----------------------------------------------------

def test_spread_columns():
    assert spread_columns(3) == ["spread_week_1", "spread_week_2", "spread_week_3"]
    assert spread_columns(1) == ["spread_week_1"]


def test_shifted_columns():
    assert shifted_columns(2) == ["shifted_week_1", "shifted_week_2"]


def test_key_spread_columns_in_order_with_shifted_past_before_week_1():
    assert key_spread_columns(2) == [
        "key", "origin_week", "open_outstanding", "spread_week_1", "spread_week_2",
        "spread_later", "overdue_beyond_habit", "without_delays", "shifted_past",
        "shifted_week_1", "shifted_week_2", "shifted_later", "delay_iqr_days", "delays_used",
        "delays_pooled"]


def test_expected_due_columns_in_order():
    assert EXPECTED_DUE_COLUMNS == [
        "spread_in_target", "spread_before_target", "spread_share_in_target",
        "overdue_beyond_habit", "shifted_due_in_target", "delay_iqr_days", "delays_used",
        "delays_pooled"]


def test_shifted_past_is_not_a_per_row_feature():
    assert "shifted_past" not in EXPECTED_DUE_COLUMNS


def test_constants():
    assert SPREAD_TOLERANCE == 0.01
    assert CALENDAR_MARGIN_DAYS == 21


# --- days_since_week_zero ------------------------------------------------------------------

def test_days_since_week_zero_counts_from_its_monday():
    found = days_since_week_zero(pd.Series([pd.Timestamp("2026-01-05"),
                                            pd.Timestamp("2026-01-06"),
                                            pd.Timestamp("2026-03-02")]), WEEK_ZERO)
    assert found.tolist() == [0, 1, 56]
    assert found.dtype == np.int64


def test_days_since_week_zero_week_is_day_floor_7():
    # Mon 2 March .. Sun 8 March 2026 are all week 8; Mon 9 March is week 9.
    days = days_since_week_zero(pd.Series(pd.date_range("2026-03-02", "2026-03-09")),
                                WEEK_ZERO)
    assert (days // 7).tolist() == [8] * 7 + [9]


def test_days_since_week_zero_is_negative_before_week_zero():
    found = days_since_week_zero(pd.Series([pd.Timestamp("2026-01-04"),
                                            pd.Timestamp("2025-12-29")]), WEEK_ZERO)
    assert found.tolist() == [-1, -7]
    assert (found // 7).tolist() == [-1, -1]


def test_days_since_week_zero_accept_dates_and_drop_the_time_of_day():
    found = days_since_week_zero(pd.Series([date(2026, 3, 2), pd.Timestamp("2026-03-02 23:59")]),
                                 WEEK_ZERO)
    assert found.tolist() == [day("2026-03-02")] * 2


def test_days_since_week_zero_time_of_day_before_week_zero_stays_on_its_day():
    # Midday Sunday 4 Jan is day -1, not 0.
    found = days_since_week_zero(pd.Series([pd.Timestamp("2026-01-04 12:00")]), WEEK_ZERO)
    assert found.tolist() == [-1]


def test_days_since_week_zero_differ_by_one_per_day():
    days = pd.Series(pd.date_range("2026-02-25", "2026-03-03"))
    assert np.diff(days_since_week_zero(days, WEEK_ZERO)).tolist() == [1] * 6


def test_days_since_week_zero_keeps_the_index():
    dates = pd.Series([pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")], index=[7, 3])
    assert days_since_week_zero(dates, WEEK_ZERO).index.tolist() == [7, 3]


def test_days_since_week_zero_of_an_empty_series_is_empty_int64():
    found = days_since_week_zero(pd.Series([], dtype="datetime64[ns]"), WEEK_ZERO)
    assert len(found) == 0
    assert found.dtype == np.int64


# --- next_business_days --------------------------------------------------------------------

def week_of_march(holidays=NO_HOLIDAYS):
    """Mon 2 to Sun 8 March 2026."""
    return next_business_days(date(2026, 3, 2), date(2026, 3, 8), holidays, WEEK_ZERO)


def two_weeks_of_march(holidays=NO_HOLIDAYS):
    """Mon 2 to Fri 13 March 2026."""
    return next_business_days(date(2026, 3, 2), date(2026, 3, 13), holidays, WEEK_ZERO)


def test_next_business_days_keep_a_weekday():
    found = two_weeks_of_march()
    for d in ("2026-03-02", "2026-03-04", "2026-03-06", "2026-03-13"):
        assert found[day(d)] == day(d)


def test_next_business_days_move_a_weekend_to_monday():
    found = two_weeks_of_march()
    assert found[day("2026-03-07")] == day("2026-03-09")
    assert found[day("2026-03-08")] == day("2026-03-09")


def test_next_business_days_skip_a_fixed_holiday():
    found = week_of_march(holidays_on((3, 4)))
    assert found[day("2026-03-04")] == day("2026-03-05")
    assert found[day("2026-03-03")] == day("2026-03-03")


def test_next_business_days_skip_a_holiday_and_the_weekend_after_it():
    found = two_weeks_of_march(holidays_on((3, 5), (3, 6)))
    assert found[day("2026-03-05")] == day("2026-03-09")
    assert found[day("2026-03-06")] == day("2026-03-09")
    assert found[day("2026-03-04")] == day("2026-03-04")


def test_next_business_days_skip_bayrams():
    # 10, 11, 12 June 2026 (Wed-Fri) are the made-up Ramazan Bayrami of NO_HOLIDAYS.
    found = next_business_days(date(2026, 6, 8), date(2026, 6, 19), NO_HOLIDAYS, WEEK_ZERO)
    assert found[day("2026-06-09")] == day("2026-06-09")
    for d in ("2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14"):
        assert found[day(d)] == day("2026-06-15"), d


def test_next_business_days_cover_every_day_up_to_the_last_business_day():
    assert sorted(two_weeks_of_march()) == list(range(day("2026-03-02"), day("2026-03-13") + 1))


def test_next_business_days_leave_out_days_after_the_last_business_day():
    # Sat 7 and Sun 8 have no business day on or after them inside the range.
    assert sorted(week_of_march()) == list(range(day("2026-03-02"), day("2026-03-06") + 1))


def test_next_business_days_leave_out_a_trailing_holiday_and_weekend():
    # Fri 6 a holiday as well: the range's last business day is Thu 5.
    assert sorted(week_of_march(holidays_on((3, 6)))) == list(range(day("2026-03-02"),
                                                                    day("2026-03-05") + 1))


def test_next_business_days_of_a_weekend_are_empty():
    assert next_business_days(date(2026, 3, 7), date(2026, 3, 8), NO_HOLIDAYS,
                              WEEK_ZERO) == {}


def test_next_business_days_start_on_a_weekend_moves_it_forward():
    found = next_business_days(date(2026, 3, 7), date(2026, 3, 9), NO_HOLIDAYS, WEEK_ZERO)
    assert found == {day("2026-03-07"): day("2026-03-09"), day("2026-03-08"): day("2026-03-09"),
                     day("2026-03-09"): day("2026-03-09")}


def test_next_business_days_values_are_business_days_on_or_after_the_day():
    found = business(holidays_on((3, 5), (3, 6), (5, 1)))
    for d, moved in found.items():
        assert moved >= d
        assert found[moved] == moved                     # a business day maps to itself
        assert (WEEK_ZERO + timedelta(days=moved)).weekday() < 5
        assert moved - d <= 6


def test_next_business_days_are_ints_counted_from_week_zero():
    other_zero = WEEK_ZERO + timedelta(days=14)
    base = two_weeks_of_march()
    shifted = next_business_days(date(2026, 3, 2), date(2026, 3, 13), NO_HOLIDAYS, other_zero)
    assert shifted == {d - 14: m - 14 for d, m in base.items()}
    assert all(isinstance(d, int) and isinstance(m, int) for d, m in base.items())


def test_next_business_days_before_week_zero_are_negative():
    # Thu 1 Jan .. Mon 5 Jan 2026; Sat 3 and Sun 4 move to Mon 5 (day 0). 1 Jan is not a
    # holiday in NO_HOLIDAYS.
    found = next_business_days(date(2026, 1, 1), date(2026, 1, 5), NO_HOLIDAYS, WEEK_ZERO)
    assert found == {-4: -4, -3: -3, -2: 0, -1: 0, 0: 0}


def test_next_business_days_raise_when_a_year_lacks_its_bayrams():
    with pytest.raises(ValueError, match="2028"):
        next_business_days(date(2027, 12, 1), date(2028, 1, 31), NO_HOLIDAYS, WEEK_ZERO)


def test_next_business_days_raise_when_start_is_after_end():
    with pytest.raises(ValueError, match="after the end"):
        next_business_days(date(2026, 3, 9), date(2026, 3, 2), NO_HOLIDAYS, WEEK_ZERO)


# --- to_next_business_day ------------------------------------------------------------------

MARCH = business(NO_HOLIDAYS)


def test_to_next_business_day_keeps_a_weekday():
    days = pd.Series([day("2026-03-02"), day("2026-03-06")])
    assert to_next_business_day(days, MARCH).tolist() == days.tolist()


def test_to_next_business_day_moves_a_weekend_to_monday():
    days = pd.Series([day("2026-03-07"), day("2026-03-08")])
    assert to_next_business_day(days, MARCH).tolist() == [day("2026-03-09")] * 2


def test_to_next_business_day_skips_a_holiday_and_the_weekend_after_it():
    calendar = business(holidays_on((3, 5), (3, 6)))
    days = pd.Series([day("2026-03-05"), day("2026-03-06"), day("2026-03-04")])
    assert to_next_business_day(days, calendar).tolist() == [day("2026-03-09"),
                                                             day("2026-03-09"),
                                                             day("2026-03-04")]


def test_to_next_business_day_leaves_days_outside_the_calendar_unchanged():
    calendar = next_business_days(date(2026, 3, 2), date(2026, 3, 15), NO_HOLIDAYS, WEEK_ZERO)
    # Sat 28 Feb is before the range; Sat 14 and Sun 15 after its last business day; 20 March
    # after its end; Fri 13 inside it.
    days = pd.Series([day("2026-02-28"), day("2026-03-14"), day("2026-03-15"),
                      day("2026-03-20"), day("2026-03-13")])
    assert to_next_business_day(days, calendar).tolist() == days.tolist()


def test_to_next_business_day_with_an_empty_calendar_returns_a_copy():
    days = pd.Series([day("2026-03-07")])
    moved = to_next_business_day(days, {})
    assert moved.tolist() == days.tolist()
    assert moved is not days


def test_to_next_business_day_does_not_change_its_input():
    days = pd.Series([day("2026-03-07"), day("2026-03-08")])
    before = days.copy()
    to_next_business_day(days, MARCH)
    pd.testing.assert_series_equal(days, before)


def test_to_next_business_day_is_int64_and_keeps_the_index():
    days = pd.Series([day("2026-03-07"), 10_000, day("2026-03-02")], index=[30, 10, 20])
    moved = to_next_business_day(days, MARCH)
    assert moved.dtype == np.int64
    assert moved.index.tolist() == [30, 10, 20]
    assert moved.tolist() == [day("2026-03-09"), 10_000, day("2026-03-02")]


def test_to_next_business_day_negative_days_are_moved_too():
    # Sun 4 Jan 2026 is day -1, Sat 3 Jan day -2: both Mon 5 Jan, day 0.
    assert to_next_business_day(pd.Series([-1, -2]), MARCH).tolist() == [0, 0]


def test_to_next_business_day_of_an_empty_series_is_empty_int64():
    moved = to_next_business_day(pd.Series([], dtype="int64"), MARCH)
    assert len(moved) == 0
    assert moved.dtype == np.int64


# --- delay_samples_at ----------------------------------------------------------------------

def samples(lines, keys=None, cutoff=CUTOFF, min_cleared=3, delays_kept=20,
            company_delays_kept=200):
    table = ledger(lines)
    return delay_samples_at(keys if keys is not None else keys_of(KEY),
                            cleared_invoice_delays(table), cutoff, min_cleared, delays_kept,
                            company_delays_kept)


def delays_of(table, key=KEY):
    return sorted(table.loc[table["key"] == key, "days_late"].tolist())


def test_samples_columns_and_dtypes():
    found = samples(history(EXAMPLE_DELAYS))
    assert list(found.columns) == ["key", "days_late", "delays_pooled"]
    assert found["days_late"].dtype == np.int64
    assert found["delays_pooled"].dtype == bool


def test_samples_own_delays_when_enough():
    found = samples(history(EXAMPLE_DELAYS))
    assert delays_of(found) == EXAMPLE_DELAYS
    assert not found["delays_pooled"].any()


def test_samples_keep_the_most_recent_by_arrival():
    # Arrivals in the order given; the oldest (10) is dropped.
    found = samples(history([10, 1, 2, 3]), delays_kept=3)
    assert delays_of(found) == [1, 2, 3]


def test_samples_most_recent_is_by_arrival_not_by_due_date():
    # Due 3 Nov (late 40 -> arrives 13 Dec) and due 17 Nov (late 1 -> arrives 18 Nov):
    # by arrival the first is the most recent.
    lines = history([40, 1], step=14)
    found = samples(lines, min_cleared=1, delays_kept=1)
    assert delays_of(found) == [40]


def test_samples_ties_on_arrival_keep_the_last_document_number():
    # Both arrive 2026-01-20; delays_kept 1 keeps document "b-..." (sorted after "a-...").
    lines = (history([5], start="2026-01-15", tag="a") + history([9], start="2026-01-11",
                                                                  tag="b"))
    found = samples(lines, min_cleared=1, delays_kept=1)
    assert delays_of(found) == [9]


def test_samples_ties_on_arrival_in_the_company_pool_keep_the_last_key():
    # Customer Z of TR02 has none of its own; A and B both arrive 2026-01-20: the pool of
    # 1 keeps B's (key TR02-B sorts after TR02-A).
    lines = (history([5], customer="A", start="2026-01-15")
             + history([9], customer="B", start="2026-01-11"))
    found = samples(lines, keys=keys_of("TR02-Z"), min_cleared=1, company_delays_kept=1)
    assert delays_of(found, "TR02-Z") == [9]
    assert found["delays_pooled"].all()


def test_samples_leave_out_delays_known_after_the_cutoff():
    # Paid 1 March (before the cutoff) but cleared 10 March: known on 10 March.
    lines = history(EXAMPLE_DELAYS) + history([0], start="2026-03-01", tag="late",
                                              clearing_lag=9)
    found = samples(lines)
    assert delays_of(found) == EXAMPLE_DELAYS
    assert delays_of(samples(lines, cutoff=date(2026, 3, 10))) == sorted(EXAMPLE_DELAYS + [0])


def test_samples_pool_the_company_below_min_cleared():
    lines = history([1, 2], customer="A") + history([7, 8, 9], customer="B")
    found = samples(lines, keys=keys_of("TR02-A"))
    assert delays_of(found, "TR02-A") == [1, 2, 7, 8, 9]   # the company's, its own included
    assert found["delays_pooled"].all()


def test_samples_with_exactly_min_cleared_use_their_own():
    lines = history([1, 2, 3], customer="A") + history([7, 8, 9], customer="B")
    found = samples(lines, keys=keys_of("TR02-A"))
    assert delays_of(found, "TR02-A") == [1, 2, 3]
    assert not found["delays_pooled"].any()


def test_samples_pool_keeps_the_company_most_recent():
    lines = history([1, 2], customer="A", start="2025-10-01") + history(
        [7, 8, 9], customer="B", start="2025-12-01")
    found = samples(lines, keys=keys_of("TR02-A"), company_delays_kept=2)
    assert delays_of(found, "TR02-A") == [8, 9]


def test_samples_pool_never_crosses_companies():
    lines = history([7, 8, 9], customer="B", company="TR04")
    found = samples(lines, keys=keys_of("TR02-A"))
    assert len(found) == 0


def test_samples_only_for_the_keys_asked():
    lines = history(EXAMPLE_DELAYS, customer="A") + history([1, 2, 3], customer="B")
    found = samples(lines, keys=keys_of("TR02-A"))
    assert set(found["key"]) == {"TR02-A"}


def test_samples_pooled_key_uses_its_own_company_code_from_keys():
    lines = history([7, 8, 9], customer="B", company="TR04")
    found = samples(lines, keys=keys_of(("TR02-A", "TR04")))
    assert delays_of(found, "TR02-A") == [7, 8, 9]


def test_samples_with_no_delays_at_all_are_empty_with_the_right_dtypes():
    found = samples([invoice("OPEN", 100.0, "2026-01-10", "2026-03-02")])
    assert len(found) == 0
    assert list(found.columns) == ["key", "days_late", "delays_pooled"]
    assert found["days_late"].dtype == np.int64
    assert found["delays_pooled"].dtype == bool


def test_samples_min_cleared_counts_all_own_delays_not_only_the_kept_ones():
    # FINDING candidate: the docstring says "a key with at least `min_cleared` of its own
    # uses its last `delays_kept`". Five of its own, min_cleared 3, delays_kept 2: it should
    # use its own last two (12, 20), not be pooled.
    lines = history(EXAMPLE_DELAYS) + history([90, 91, 92], customer="B")
    found = samples(lines, min_cleared=3, delays_kept=2)
    assert delays_of(found) == [12, 20]
    assert not found["delays_pooled"].any()


# --- open_invoices_in_try ------------------------------------------------------------------

def open_in_try(lines, keys=None, rates=NO_RATES, cutoff=CUTOFF):
    return open_invoices_in_try(ledger(lines), keys if keys is not None else keys_of(KEY),
                                rates, cutoff, WEEK_ZERO)


def test_open_invoices_in_try_running_example():
    found = open_in_try(example_lines())
    assert found["document_no"].tolist() == ["OPEN"]
    assert found["outstanding_try"].tolist() == [100.0]
    assert found["due_day"].tolist() == [day("2026-03-02")]     # 56: week 8
    assert found["invoice"].tolist() == [0]
    assert found["due_day"].dtype == np.int64


def test_open_invoices_in_try_numbers_invoices_from_zero_with_a_fresh_index():
    lines = (example_lines() + [invoice("OPEN-B", 40.0, "2026-01-10", "2026-03-09"),
                                invoice("OTHER", 70.0, "2026-01-10", "2026-03-02",
                                        customer="B")])
    found = open_in_try(lines, keys=keys_of("TR02-A", "TR02-B"))
    assert found["invoice"].tolist() == list(range(len(found)))
    assert found.index.tolist() == list(range(len(found)))
    assert sorted(found["document_no"]) == ["OPEN", "OPEN-B", "OTHER"]


def test_open_invoices_in_try_only_the_keys_asked():
    lines = example_lines() + [invoice("OTHER", 70.0, "2026-01-10", "2026-03-02",
                                       customer="B")]
    found = open_in_try(lines)
    assert set(found["key"]) == {KEY}
    assert found["document_no"].tolist() == ["OPEN"]


def test_open_invoices_in_try_is_net_of_partial_payments():
    lines = example_lines() + [payment("PART", -40.0, "2026-02-20", reference="OPEN")]
    assert open_in_try(lines)["outstanding_try"].tolist() == pytest.approx([60.0])


def test_open_invoices_in_try_leaves_out_invoices_posted_after_the_cutoff():
    lines = example_lines() + [invoice("LATER", 70.0, "2026-03-09", "2026-03-20")]
    assert open_in_try(lines)["document_no"].tolist() == ["OPEN"]


def test_open_invoices_in_try_converts_at_the_rate_known_at_the_cutoff():
    rates = rates_table([(date(2026, 1, 2), "EUR", 35.0), (date(2026, 3, 6), "EUR", 36.0),
                         (date(2026, 3, 9), "EUR", 99.0)])
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-02", currency="EUR")]
    assert open_in_try(lines, rates=rates)["outstanding_try"].tolist() == [3600.0]


def test_open_invoices_in_try_without_a_rate_raises():
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-02", currency="EUR")]
    with pytest.raises(ValueError, match="EUR"):
        open_in_try(lines)


def test_open_invoices_in_try_without_a_net_due_date_raises_with_count_and_document():
    table = ledger(example_lines() + [invoice("OPEN-B", 40.0, "2026-01-10", "2026-03-09")])
    table["net_due_date"] = table["net_due_date"].where(
        ~table["document_no"].isin(["OPEN", "OPEN-B"]))
    with pytest.raises(ValueError, match="2 open invoices have no net due date") as error:
        open_invoices_in_try(table, keys_of(KEY), NO_RATES, CUTOFF, WEEK_ZERO)
    assert "OPEN" in str(error.value) and "OPEN-B" in str(error.value)
    assert str(CUTOFF) in str(error.value)


def test_open_invoices_in_try_a_missing_due_date_of_another_key_does_not_raise():
    table = ledger(example_lines() + [invoice("OTHER", 70.0, "2026-01-10", "2026-03-02",
                                              customer="B")])
    table.loc[table["document_no"] == "OTHER", "net_due_date"] = pd.NaT
    found = open_invoices_in_try(table, keys_of(KEY), NO_RATES, CUTOFF, WEEK_ZERO)
    assert found["document_no"].tolist() == ["OPEN"]


def test_open_invoices_in_try_with_nothing_open_is_empty_with_its_columns():
    found = open_in_try(history(EXAMPLE_DELAYS))
    assert len(found) == 0
    for column in ("key", "outstanding_try", "due_day", "invoice"):
        assert column in found.columns


def test_open_invoices_in_try_does_not_change_its_inputs():
    table = ledger(example_lines())
    keys = keys_of(KEY)
    copies = (table.copy(), keys.copy())
    open_invoices_in_try(table, keys, NO_RATES, CUTOFF, WEEK_ZERO)
    pd.testing.assert_frame_equal(copies[0], table)
    pd.testing.assert_frame_equal(copies[1], keys)


# --- delay_statistics ----------------------------------------------------------------------

def sample_table(rows):
    """rows: (key, days_late, delays_pooled)."""
    return pd.DataFrame({"key": pd.Series([r[0] for r in rows], dtype=object),
                         "days_late": pd.Series([r[1] for r in rows], dtype="int64"),
                         "delays_pooled": pd.Series([r[2] for r in rows], dtype=bool)})


def test_delay_statistics_running_example():
    found = delay_statistics(sample_table([(KEY, d, False) for d in EXAMPLE_DELAYS]))
    assert list(found.columns) == ["key", "delays_used", "delays_pooled", "median_delay",
                                   "delay_iqr_days"]
    r = row(found)
    assert r["delays_used"] == 5
    assert r["median_delay"] == 6.0
    assert r["delay_iqr_days"] == 7.0     # 12 - 5
    assert not r["delays_pooled"]


def test_delay_statistics_one_row_per_key_with_its_own_values():
    found = delay_statistics(sample_table([("A", 1, False), ("A", 4, False), ("B", 3, True)]))
    assert sorted(found["key"]) == ["A", "B"]
    a, b = row(found, "A"), row(found, "B")
    assert (a["delays_used"], a["median_delay"], a["delay_iqr_days"]) == (2, 2.5, 1.5)
    assert not a["delays_pooled"]
    assert (b["delays_used"], b["median_delay"], b["delay_iqr_days"]) == (1, 3.0, 0.0)
    assert b["delays_pooled"]


@pytest.mark.parametrize("delays, iqr", [([1, 2, 4], 1.5), ([5], 0.0), ([5, 5, 5], 0.0),
                                         ([0, 10], 5.0), ([-4, -2, 0, 6, 10], 8.0)])
def test_delay_statistics_iqr_is_the_linear_q75_minus_q25(delays, iqr):
    found = delay_statistics(sample_table([(KEY, d, False) for d in delays]))
    assert row(found)["delay_iqr_days"] == pytest.approx(iqr)


def test_delay_statistics_keeps_the_half_of_an_even_median():
    # The rounding to whole days happens later, in invoice_spread_at.
    found = delay_statistics(sample_table([(KEY, 2, False), (KEY, 3, False)]))
    assert row(found)["median_delay"] == 2.5


def test_delay_statistics_dtypes():
    found = delay_statistics(sample_table([(KEY, 3, True), (KEY, 5, True)]))
    assert found["delays_used"].dtype == np.int64
    assert found["delays_pooled"].dtype == bool
    assert found["median_delay"].dtype == np.float64
    assert found["delay_iqr_days"].dtype == np.float64


def test_delay_statistics_of_no_samples_is_empty_with_its_columns():
    found = delay_statistics(sample_table([]))
    assert len(found) == 0
    assert list(found.columns) == ["key", "delays_used", "delays_pooled", "median_delay",
                                   "delay_iqr_days"]


def test_delay_statistics_from_delay_samples_at():
    found = delay_statistics(samples(history([1, 2]) + history([7, 8, 9], customer="B"),
                                     keys=keys_of("TR02-A", "TR02-B")))
    assert row(found, "TR02-A")["delays_pooled"]                    # 2 < min_cleared 3
    assert row(found, "TR02-A")["delays_used"] == 5
    assert not row(found, "TR02-B")["delays_pooled"]
    assert row(found, "TR02-B")["median_delay"] == 8.0


# --- week_column ---------------------------------------------------------------------------

@pytest.mark.parametrize("week, expected", [
    (-3, "spread_past"), (0, "spread_past"), (1, "spread_week_1"), (4, "spread_week_4"),
    (5, "spread_week_5"), (6, "spread_later"), (40, "spread_later")])
def test_week_column(week, expected):
    assert week_column(week, "spread", 5) == expected


def test_week_column_uses_the_prefix():
    assert week_column(0, "shifted", 2) == "shifted_past"
    assert week_column(2, "shifted", 2) == "shifted_week_2"
    assert week_column(3, "shifted", 2) == "shifted_later"


def test_week_column_names_match_the_key_spread_columns():
    for prefix in ("spread", "shifted"):
        names = {week_column(week, prefix, 3) for week in range(1, 10)}
        assert names <= set(key_spread_columns(3))
    assert week_column(0, "shifted", 3) in key_spread_columns(3)


# --- amounts_per_week ----------------------------------------------------------------------

def placed_table(rows):
    """rows: (key, week, amount)."""
    return pd.DataFrame({"key": [r[0] for r in rows],
                         "week": pd.Series([r[1] for r in rows], dtype="int64"),
                         "amount": pd.Series([r[2] for r in rows], dtype="float64")})


def test_amounts_per_week_sums_per_key_and_week():
    found = amounts_per_week(placed_table([("A", 1, 10.0), ("A", 1, 5.0), ("A", 3, 2.0),
                                           ("B", 2, 7.0)]), "spread", 5)
    assert row(found, "A")["spread_week_1"] == 15.0
    assert row(found, "A")["spread_week_3"] == 2.0
    assert row(found, "B")["spread_week_2"] == 7.0


def test_amounts_per_week_fills_zero_where_nothing_lands():
    found = amounts_per_week(placed_table([("A", 1, 10.0), ("B", 2, 7.0)]), "spread", 5)
    assert row(found, "A")["spread_week_2"] == 0.0
    assert row(found, "B")["spread_week_1"] == 0.0


def test_amounts_per_week_only_columns_of_weeks_that_occur():
    found = amounts_per_week(placed_table([("A", 1, 10.0), ("A", 9, 1.0), ("A", 0, 3.0)]),
                             "shifted", 5)
    assert sorted(found.columns) == ["key", "shifted_later", "shifted_past", "shifted_week_1"]
    assert row(found, "A")["shifted_past"] == 3.0
    assert row(found, "A")["shifted_later"] == 1.0


def test_amounts_per_week_weeks_past_the_horizons_add_up_in_later():
    found = amounts_per_week(placed_table([("A", 6, 1.0), ("A", 7, 2.0), ("A", 30, 4.0)]),
                             "spread", 5)
    assert row(found, "A")["spread_later"] == 7.0


def test_amounts_per_week_weeks_before_one_add_up_in_past():
    found = amounts_per_week(placed_table([("A", 0, 1.0), ("A", -2, 2.0)]), "shifted", 5)
    assert row(found, "A")["shifted_past"] == 3.0


def test_amounts_per_week_one_row_per_key_and_a_plain_column_index():
    found = amounts_per_week(placed_table([("B", 1, 1.0), ("A", 2, 1.0), ("B", 2, 1.0)]),
                             "spread", 5)
    assert sorted(found["key"]) == ["A", "B"]
    assert found["key"].is_unique
    assert found.columns.name is None
    assert found.index.tolist() == list(range(len(found)))


def test_amounts_per_week_of_nothing_is_a_table_with_only_a_key_column():
    found = amounts_per_week(placed_table([]), "spread", 5)
    assert list(found.columns) == ["key"]
    assert len(found) == 0
    assert found["key"].dtype == object


def test_amounts_per_week_does_not_change_its_input():
    placed = placed_table([("A", 1, 10.0), ("A", 9, 1.0)])
    before = placed.copy()
    amounts_per_week(placed, "spread", 5)
    pd.testing.assert_frame_equal(before, placed)


def test_amounts_per_week_is_float():
    found = amounts_per_week(placed_table([("A", 1, 10.0), ("B", 2, 7.0)]), "spread", 5)
    assert found["spread_week_1"].dtype == np.float64
    assert found["spread_week_2"].dtype == np.float64


# --- add_per_key ---------------------------------------------------------------------------

def test_add_per_key_adds_the_columns_and_fills_missing_keys():
    table = pd.DataFrame({"key": ["A", "B", "C"], "origin_week": [8, 8, 8]})
    per_key = pd.DataFrame({"key": ["C", "A"], "x": [3.0, 1.0], "y": [30.0, 10.0]})
    found = add_per_key(table, per_key, fill=0.0)
    assert found["key"].tolist() == ["A", "B", "C"]          # the order of `table`
    assert found["x"].tolist() == [1.0, 0.0, 3.0]
    assert found["y"].tolist() == [10.0, 0.0, 30.0]
    assert found["origin_week"].tolist() == [8, 8, 8]


def test_add_per_key_fill_nan_and_false():
    table = pd.DataFrame({"key": ["A", "B"]})
    found = add_per_key(table, pd.DataFrame({"key": ["A"], "x": [2.0]}), fill=float("nan"))
    assert found["x"].iloc[0] == 2.0 and np.isnan(found["x"].iloc[1])
    found = add_per_key(table, pd.DataFrame({"key": ["A"], "flag": [True]}), fill=False)
    assert found["flag"].tolist() == [True, False]


def test_add_per_key_with_an_empty_per_key_adds_nothing():
    table = pd.DataFrame({"key": ["A", "B"], "x": [1.0, 2.0]})
    found = add_per_key(table, pd.DataFrame({"key": pd.Series(dtype=object)}), fill=0.0)
    pd.testing.assert_frame_equal(found, table)


def test_add_per_key_with_an_empty_per_key_and_columns_fills_every_row():
    table = pd.DataFrame({"key": ["A", "B"]})
    per_key = pd.DataFrame({"key": pd.Series(dtype=object), "x": pd.Series(dtype="float64")})
    assert add_per_key(table, per_key, fill=0.0)["x"].tolist() == [0.0, 0.0]


def test_add_per_key_ignores_keys_not_in_the_table():
    table = pd.DataFrame({"key": ["A"]})
    found = add_per_key(table, pd.DataFrame({"key": ["A", "Z"], "x": [1.0, 9.0]}), fill=0.0)
    assert found["key"].tolist() == ["A"]
    assert found["x"].tolist() == [1.0]


def test_add_per_key_raises_on_a_repeated_key_in_per_key():
    table = pd.DataFrame({"key": ["A", "B"]})
    with pytest.raises(pd.errors.MergeError):
        add_per_key(table, pd.DataFrame({"key": ["A", "A"], "x": [1.0, 2.0]}), fill=0.0)


def test_add_per_key_raises_on_a_repeated_key_in_the_table():
    table = pd.DataFrame({"key": ["A", "A"]})
    with pytest.raises(pd.errors.MergeError):
        add_per_key(table, pd.DataFrame({"key": ["A"], "x": [1.0]}), fill=0.0)


def test_add_per_key_does_not_change_its_inputs():
    table = pd.DataFrame({"key": ["A", "B"]})
    per_key = pd.DataFrame({"key": ["A"], "x": [1.0]})
    copies = (table.copy(), per_key.copy())
    add_per_key(table, per_key, fill=0.0)
    pd.testing.assert_frame_equal(copies[0], table)
    pd.testing.assert_frame_equal(copies[1], per_key)


# --- invoice_spread_at: the running example -------------------------------------------------

def test_running_example_spread():
    r = row(spread(example_lines()))
    assert r["open_outstanding"] == 100.0
    assert weeks(r) == [50.0, 25.0, 25.0, 0.0, 0.0]
    assert r["spread_later"] == 0.0
    assert r["overdue_beyond_habit"] == 0.0
    assert r["without_delays"] == 0.0


def test_running_example_shifted_date_is_week_1():
    # Median 6: Sun 8 March, moved to Mon 9 = week t+1.
    r = row(spread(example_lines()))
    assert r["shifted_past"] == 0.0
    assert weeks(r, kind="shifted") == [100.0, 0.0, 0.0, 0.0, 0.0]
    assert r["shifted_later"] == 0.0


def test_running_example_delay_statistics():
    r = row(spread(example_lines()))
    assert r["delay_iqr_days"] == 7.0       # 12 - 5
    assert r["delays_used"] == 5
    assert not r["delays_pooled"]
    assert r["origin_week"] == ORIGIN


def test_running_example_columns_and_dtypes():
    table = spread(example_lines())
    assert list(table.columns) == key_spread_columns(5)
    assert table["delays_used"].dtype == np.int64
    assert table["delays_pooled"].dtype == bool
    for column in ["open_outstanding"] + spread_columns(5) + shifted_columns(5) + [
            "spread_later", "overdue_beyond_habit", "without_delays", "shifted_past",
            "shifted_later", "delay_iqr_days"]:
        assert table[column].dtype == np.float64, column


def test_two_delays_on_the_same_day_each_take_a_share():
    # 5 and 6 days late both land on Mon 9: two of the four possible arrivals = 50.
    r = row(spread(example_lines(), horizons=1))
    assert r["spread_week_1"] == 50.0
    assert r["spread_later"] == 50.0


def test_a_holiday_turns_the_impossible_arrival_into_a_possible_one():
    # Thu 5 and Fri 6 March holidays: the 3-day arrival moves to Mon 9, so all five are
    # possible: three on Mon 9 (60), Mon 16 (20), Mon 23 (20).
    r = row(spread(example_lines(), holidays=holidays_on((3, 5), (3, 6))))
    assert weeks(r) == [60.0, 20.0, 20.0, 0.0, 0.0]


def test_a_holiday_inside_a_week_keeps_the_week():
    # Mon 16 a holiday: the 12-day arrival moves to Tue 17, still week t+2.
    r = row(spread(example_lines(), holidays=holidays_on((3, 16))))
    assert weeks(r) == [50.0, 25.0, 25.0, 0.0, 0.0]


def test_a_holiday_on_the_last_friday_pushes_an_arrival_into_the_next_week():
    # Due Mon 9 March + 4 = Fri 13; a holiday that day: Mon 16, week t+2. Delays 4, 4, 4.
    lines = [invoice("OPEN", 90.0, "2026-01-10", "2026-03-09")] + history([4, 4, 4])
    assert weeks(row(spread(lines)))[:2] == [90.0, 0.0]
    assert weeks(row(spread(lines, holidays=holidays_on((3, 13)))))[:2] == [0.0, 90.0]


@pytest.mark.parametrize("horizons, expected, later", [
    (1, [50.0], 50.0), (2, [50.0, 25.0], 25.0), (3, [50.0, 25.0, 25.0], 0.0),
    (8, [50.0, 25.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0)])
def test_weeks_after_horizons_go_to_spread_later(horizons, expected, later):
    r = row(spread(example_lines(), horizons=horizons))
    assert weeks(r, horizons) == expected
    assert r["spread_later"] == later
    assert list(spread(example_lines(), horizons=horizons).columns) == key_spread_columns(
        horizons)


def test_an_arrival_on_the_sunday_of_t_is_impossible_and_the_monday_after_is_week_1():
    # Every day a business day, so nothing moves: due Mon 2 + 6 = Sun 8 (= t), + 7 = Mon 9.
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-02")] + history([6, 7])
    r = row(spread(lines, min_cleared=1, next_business=EVERY_DAY))
    assert weeks(r) == [100.0, 0.0, 0.0, 0.0, 0.0]


def test_an_arrival_on_the_friday_of_t_is_impossible():
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-02")] + history([4, 7])
    r = row(spread(lines, min_cleared=1))
    assert weeks(r) == [100.0, 0.0, 0.0, 0.0, 0.0]


# --- invoice_spread_at: overdue beyond habit, without delays, shifted_past --------------------

def test_an_invoice_older_than_every_delay_is_overdue_beyond_habit():
    # Due 3 Nov 2025: every arrival is long before 8 March.
    r = row(spread(example_lines(due="2025-11-03")))
    assert r["overdue_beyond_habit"] == 100.0
    assert weeks(r) == [0.0] * 5
    assert r["spread_later"] == 0.0


def test_an_invoice_overdue_past_its_shifted_date_goes_to_shifted_past():
    # Due Mon 16 Feb: median 6 -> Sun 22 Feb -> Mon 23 Feb, before t: shifted_past.
    # Its spread still has one possible arrival (+20 = Sun 8 March -> Mon 9): all 100 there.
    r = row(spread(example_lines(due="2026-02-16")))
    assert r["shifted_past"] == 100.0
    assert weeks(r, kind="shifted") == [0.0] * 5
    assert r["shifted_later"] == 0.0
    assert weeks(r) == [100.0, 0.0, 0.0, 0.0, 0.0]
    assert r["overdue_beyond_habit"] == 0.0


def test_a_shifted_date_on_the_sunday_of_t_is_past_and_the_monday_after_is_week_1():
    # Every day a business day: due Mon 2 March + median 6 = Sun 8 (t) is past; + 7 = Mon 9.
    on_sunday = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-02")] + history([6])
    r = row(spread(on_sunday, min_cleared=1, next_business=EVERY_DAY))
    assert r["shifted_past"] == 100.0
    assert r["shifted_week_1"] == 0.0
    on_monday = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-02")] + history([7])
    r = row(spread(on_monday, min_cleared=1, next_business=EVERY_DAY))
    assert r["shifted_past"] == 0.0
    assert r["shifted_week_1"] == 100.0


def test_a_shifted_date_on_the_sunday_of_t_moves_to_week_1_with_the_real_calendar():
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-02")] + history([6])
    r = row(spread(lines, min_cleared=1))
    assert r["shifted_past"] == 0.0
    assert r["shifted_week_1"] == 100.0


def test_a_shifted_date_beyond_the_horizons_goes_to_shifted_later():
    # Due Mon 27 April (week 16 = t+8), horizons 5.
    r = row(spread(example_lines(due="2026-04-27")))
    assert r["shifted_later"] == 100.0
    assert r["spread_later"] == 100.0
    assert weeks(r, kind="shifted") == [0.0] * 5


def test_a_key_without_any_delays_goes_to_without_delays():
    lines = example_lines() + [invoice("ZOPEN", 50.0, "2026-01-10", "2026-03-04",
                                       customer="Z", company="TR09")]
    r = row(spread(lines, keys=keys_of("TR02-A", "TR09-Z")), "TR09-Z")
    assert r["open_outstanding"] == 50.0
    assert r["without_delays"] == 50.0
    assert weeks(r) == [0.0] * 5 and weeks(r, kind="shifted") == [0.0] * 5
    assert r["shifted_past"] == 0.0 and r["shifted_later"] == 0.0
    assert r["spread_later"] == 0.0 and r["overdue_beyond_habit"] == 0.0
    assert np.isnan(r["delay_iqr_days"])
    assert r["delays_used"] == 0
    assert not r["delays_pooled"]


def test_a_key_with_delays_but_nothing_open_has_zero_amounts():
    lines = history(EXAMPLE_DELAYS)
    r = row(spread(lines))
    assert r["open_outstanding"] == 0.0
    assert weeks(r) == [0.0] * 5
    assert r["shifted_past"] == 0.0 and r["without_delays"] == 0.0
    assert r["delays_used"] == 5
    assert r["delay_iqr_days"] == 7.0


def test_a_key_absent_from_the_ledger_gets_a_row_of_zeros():
    r = row(spread(example_lines(), keys=keys_of("TR02-A", "TR02-Q")), "TR02-Q")
    # It is pooled from TR02 (A's delays), but has nothing open.
    assert r["open_outstanding"] == 0.0
    assert weeks(r) == [0.0] * 5
    assert r["delays_pooled"]
    assert r["delays_used"] == 5


def test_a_pooled_key_is_spread_by_its_company_delays():
    lines = history(EXAMPLE_DELAYS, customer="B") + [
        invoice("OPEN", 100.0, "2026-01-10", "2026-03-02", customer="A")]
    r = row(spread(lines, keys=keys_of("TR02-A")))
    assert weeks(r) == [50.0, 25.0, 25.0, 0.0, 0.0]
    assert r["delays_pooled"]
    assert r["delays_used"] == 5


def test_with_no_delays_anywhere_every_invoice_is_without_delays():
    # FINDING candidate: at an early origin no cleared invoice is known yet. The function
    # must still return a row: everything without_delays.
    r = row(spread([invoice("OPEN", 100.0, "2026-01-10", "2026-03-02")]))
    assert r["without_delays"] == 100.0
    assert r["delays_used"] == 0
    assert np.isnan(r["delay_iqr_days"])


# --- invoice_spread_at: amounts --------------------------------------------------------------

def test_each_invoice_is_shared_on_its_own():
    # A: 100 due Mon 2 March (50 / 25 / 25); B: 40 due Mon 9 March, all five arrivals possible
    # (Thu 12; Mon 16 twice; Mon 23; Mon 30): 8 / 16 / 8 / 8.
    lines = example_lines() + [invoice("OPEN-B", 40.0, "2026-01-10", "2026-03-09")]
    r = row(spread(lines))
    assert weeks(r) == pytest.approx([58.0, 41.0, 33.0, 8.0, 0.0])
    # Shifted: A on Mon 9 (week 1), B on 15 -> Mon 16 (week 2).
    assert weeks(r, kind="shifted") == [100.0, 40.0, 0.0, 0.0, 0.0]


def test_the_outstanding_is_net_of_partial_payments():
    lines = example_lines() + [payment("PART", -40.0, "2026-02-20", reference="OPEN")]
    r = row(spread(lines))
    assert r["open_outstanding"] == pytest.approx(60.0)
    assert weeks(r) == pytest.approx([30.0, 15.0, 15.0, 0.0, 0.0])


def test_a_foreign_invoice_is_converted_at_the_rate_known_at_t():
    rates = rates_table([(date(2026, 1, 2), "EUR", 35.0), (date(2026, 3, 6), "EUR", 36.0),
                         (date(2026, 3, 9), "EUR", 99.0)])
    lines = ([invoice("OPEN", 100.0, "2026-01-10", "2026-03-02", currency="EUR")]
             + history(EXAMPLE_DELAYS))
    r = row(spread(lines, rates=rates))
    assert r["open_outstanding"] == 3600.0
    assert weeks(r) == [1800.0, 900.0, 900.0, 0.0, 0.0]


def test_a_foreign_invoice_without_a_rate_raises():
    lines = ([invoice("OPEN", 100.0, "2026-01-10", "2026-03-02", currency="EUR")]
             + history(EXAMPLE_DELAYS))
    with pytest.raises(ValueError, match="EUR"):
        spread(lines)


def test_invoices_of_keys_not_asked_for_are_left_out():
    lines = example_lines() + [invoice("OTHER", 70.0, "2026-01-10", "2026-03-02",
                                       customer="B")]
    table = spread(lines)
    assert table["key"].tolist() == [KEY]
    assert row(table)["open_outstanding"] == 100.0


def test_invoices_posted_after_t_or_cleared_by_t_are_not_open():
    lines = history(EXAMPLE_DELAYS) + [
        invoice("LATER", 70.0, "2026-03-09", "2026-03-20"),
        invoice("DONE", 30.0, "2026-01-10", "2026-03-02", cleared="2026-03-08",
                clearing_document="CLR-DONE"),
        payment("PDONE", -30.0, "2026-03-08", reference="DONE", cleared="2026-03-08",
                clearing_document="CLR-DONE")]
    assert row(spread(lines))["open_outstanding"] == 0.0


def test_a_delay_known_after_t_is_not_used():
    lines = example_lines() + history([0], start="2026-03-01", tag="late", clearing_lag=9)
    r = row(spread(lines))
    assert r["delays_used"] == 5
    assert weeks(r) == [50.0, 25.0, 25.0, 0.0, 0.0]


def test_delays_kept_changes_the_spread():
    # The last two delays (12, 20): Mon 16 and Mon 23.
    r = row(spread(example_lines(), min_cleared=2, delays_kept=2))
    assert weeks(r) == [0.0, 50.0, 50.0, 0.0, 0.0]
    assert r["delays_used"] == 2
    assert r["delay_iqr_days"] == 4.0   # 18 - 14


# --- invoice_spread_at: the median and the IQR -------------------------------------------------

def test_the_median_rounds_a_half_down_to_even():
    # Delays 2 and 3: median 2.5 -> 2. Due Fri 6 March + 2 = Sun 8 (t): past. (Rounded up to
    # 3 it would be Mon 9, week t+1.) Every day a business day, so nothing moves.
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-06")] + history([2, 3])
    r = row(spread(lines, min_cleared=2, next_business=EVERY_DAY))
    assert r["shifted_past"] == 100.0
    assert r["shifted_week_1"] == 0.0


def test_the_median_rounds_a_half_up_to_even():
    # Delays 3 and 4: median 3.5 -> 4. Due Thu 5 March + 4 = Mon 9: week t+1. (Rounded down
    # to 3 it would be Sun 8, past.)
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-05")] + history([3, 4])
    r = row(spread(lines, min_cleared=2, next_business=EVERY_DAY))
    assert r["shifted_past"] == 0.0
    assert r["shifted_week_1"] == 100.0


def test_the_median_rounding_matches_evaluate_invoice_delays_rule():
    # evaluate.invoice_delays rounds with pandas' round (half to even): the same rule.
    assert pd.Series([2.5, 3.5, -2.5, 6.0]).round().astype("int64").tolist() == [2, 4, -2, 6]


@pytest.mark.parametrize("delays, iqr", [([1, 2, 4], 1.5), ([5], 0.0), ([5, 5, 5], 0.0),
                                         ([0, 10], 5.0), ([-4, -2, 0, 6, 10], 8.0)])
def test_delay_iqr_is_the_linear_q75_minus_q25(delays, iqr):
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-02")] + history(delays)
    r = row(spread(lines, min_cleared=1))
    assert r["delay_iqr_days"] == pytest.approx(iqr)
    assert r["delays_used"] == len(delays)


def test_negative_delays_pay_before_the_due_date():
    # Paid 3 days early: due Mon 9 March - 3 = Fri 6 (t): impossible; 0 -> Mon 9.
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-09")] + history([-3, 0])
    r = row(spread(lines, min_cleared=1))
    assert weeks(r) == [100.0, 0.0, 0.0, 0.0, 0.0]


# --- invoice_spread_at: errors ---------------------------------------------------------------

def test_repeated_keys_raise():
    with pytest.raises(ValueError, match="repeated"):
        spread(example_lines(), keys=keys_of("TR02-A", "TR02-A"))


def test_an_open_invoice_without_a_net_due_date_raises():
    table = ledger(example_lines())
    table.loc[table["document_no"] == "OPEN", "net_due_date"] = pd.NaT
    with pytest.raises(ValueError, match="no net due date") as error:
        invoice_spread_at(table, cleared_invoice_delays(table), NO_RATES, keys_of(KEY),
                          ORIGIN, WEEK_ZERO, business(), 5, 3, 20, 200)
    assert "OPEN" in str(error.value)


def test_invoice_spread_at_does_not_change_its_inputs():
    table = ledger(example_lines())
    delays = cleared_invoice_delays(table)
    keys = keys_of(KEY)
    copies = (table.copy(), delays.copy(), keys.copy())
    invoice_spread_at(table, delays, NO_RATES, keys, ORIGIN, WEEK_ZERO, business(), 5, 3, 20,
                      200)
    for before, after in zip(copies, (table, delays, keys)):
        pd.testing.assert_frame_equal(before, after)


def test_the_rows_follow_the_order_of_keys():
    lines = example_lines() + history([1, 2, 3], customer="B")
    table = spread(lines, keys=keys_of("TR02-B", "TR02-Q", "TR02-A"))
    assert table["key"].tolist() == ["TR02-B", "TR02-Q", "TR02-A"]
    assert table.index.tolist() == [0, 1, 2]


# --- the tie-out -------------------------------------------------------------------------------

def tie_table(**changes):
    values = {"key": ["K"], "origin_week": [8], "open_outstanding": [100.0],
              "spread_week_1": [50.0], "spread_week_2": [20.0], "spread_later": [10.0],
              "overdue_beyond_habit": [10.0], "without_delays": [10.0],
              "shifted_past": [30.0], "shifted_week_1": [40.0], "shifted_week_2": [0.0],
              "shifted_later": [20.0]}
    values.update({key: [value] for key, value in changes.items()})
    return pd.DataFrame(values)


def test_check_spread_adds_up_passes_on_a_tied_table():
    check_spread_adds_up(tie_table(), 2)


def test_check_spread_adds_up_passes_within_the_tolerance():
    check_spread_adds_up(tie_table(spread_week_1=50.005, shifted_week_1=39.995), 2)


def test_check_spread_adds_up_raises_on_the_spread():
    with pytest.raises(ValueError, match="the spread of 1 keys") as error:
        check_spread_adds_up(tie_table(spread_week_1=50.02), 2)
    assert "'K'" in str(error.value)


def test_check_spread_adds_up_raises_on_the_shifted_date():
    with pytest.raises(ValueError, match="the shifted date of 1 keys"):
        check_spread_adds_up(tie_table(shifted_later=20.02), 2)


def test_check_spread_adds_up_counts_shifted_past():
    with pytest.raises(ValueError, match="shifted date"):
        check_spread_adds_up(tie_table(shifted_past=0.0), 2)


def test_check_spread_adds_up_does_not_count_shifted_past_in_the_spread():
    # shifted_past is not part of the spread: moving amounts between them unties the spread.
    with pytest.raises(ValueError, match="the spread"):
        check_spread_adds_up(tie_table(overdue_beyond_habit=0.0, shifted_past=30.0), 2)


def test_check_spread_adds_up_counts_without_delays_in_both():
    with pytest.raises(ValueError, match="the spread"):
        check_spread_adds_up(tie_table(without_delays=0.0), 2)


def random_ledger(seed):
    """Three customers of TR02 and one of TR04: random cleared history, random open
    invoices (some partly paid, some long overdue, some due far ahead)."""
    rng = np.random.default_rng(seed)
    lines = []
    for customer, company in (("A", "TR02"), ("B", "TR02"), ("C", "TR02"), ("D", "TR04")):
        count = int(rng.integers(0, 8))
        delays = rng.integers(-10, 60, size=count).tolist()
        lines += history(delays, customer=customer, company=company,
                         start=str(date(2025, 9, 1) + timedelta(days=int(rng.integers(0, 30)))),
                         step=int(rng.integers(5, 20)))
        for n in range(int(rng.integers(0, 5))):
            due = date(2026, 3, 8) + timedelta(days=int(rng.integers(-150, 80)))
            number = f"O-{customer}-{n}"
            amount = float(rng.integers(1, 10_000)) + 0.37
            lines.append(invoice(number, amount, "2025-09-01", str(due), customer=customer,
                                 company=company))
            if rng.random() < 0.3:
                lines.append(payment(f"PP-{customer}-{n}", -round(amount * rng.random(), 2),
                                     "2026-02-01", reference=number, customer=customer,
                                     company=company))
    return lines


@pytest.mark.parametrize("seed", range(40))
def test_random_ledgers_tie_out_and_are_not_negative(seed):
    lines = random_ledger(seed)
    keys = keys_of("TR02-A", "TR02-B", "TR02-C", "TR04-D")
    for min_cleared, kept, company_kept in ((3, 20, 200), (1, 2, 3), (5, 5, 1)):
        for horizons in (1, 5):
            table = spread(lines, keys=keys, horizons=horizons, min_cleared=min_cleared,
                           delays_kept=kept, company_delays_kept=company_kept)
            spread_part = (table[spread_columns(horizons)].sum(axis=1) + table["spread_later"]
                           + table["overdue_beyond_habit"] + table["without_delays"])
            shifted_part = (table["shifted_past"] + table[shifted_columns(horizons)].sum(axis=1)
                            + table["shifted_later"] + table["without_delays"])
            assert (spread_part - table["open_outstanding"]).abs().max() <= 0.01
            assert (shifted_part - table["open_outstanding"]).abs().max() <= 0.01
            amounts = (spread_columns(horizons) + shifted_columns(horizons)
                       + ["spread_later", "overdue_beyond_habit", "without_delays",
                          "shifted_past", "shifted_later"])
            assert (table[amounts] >= -1e-9).all().all()
            assert (table["delays_used"] <= np.where(table["delays_pooled"], company_kept,
                                                     kept)).all()
            no_delays = table["delays_used"] == 0
            assert table.loc[no_delays, "delay_iqr_days"].isna().all()
            assert (table.loc[no_delays, "without_delays"]
                    == table.loc[no_delays, "open_outstanding"]).all()


# --- expected_dues -----------------------------------------------------------------------------

def at(features, h, key=KEY, origin=ORIGIN):
    found = features[(features["key"] == key) & (features["origin_week"] == origin)
                     & (features["h"] == h)]
    assert len(found) == 1
    return found.iloc[0]


def test_expected_dues_columns_and_dtypes():
    features = dues(example_lines())
    assert list(features.columns) == ["key", "origin_week", "h"] + EXPECTED_DUE_COLUMNS
    for column in ("spread_in_target", "spread_before_target", "spread_share_in_target",
                   "overdue_beyond_habit", "shifted_due_in_target", "delay_iqr_days"):
        assert features[column].dtype == np.float64, column
    assert features["delays_used"].dtype == np.int64
    assert features["delays_pooled"].dtype == bool


def test_expected_dues_running_example_per_h():
    features = dues(example_lines())
    assert features["spread_in_target"].tolist() == [50.0, 25.0, 25.0, 0.0, 0.0]
    assert features["spread_before_target"].tolist() == [0.0, 50.0, 75.0, 100.0, 100.0]
    assert features["spread_share_in_target"].tolist() == [0.5, 0.25, 0.25, 0.0, 0.0]
    assert features["shifted_due_in_target"].tolist() == [100.0, 0.0, 0.0, 0.0, 0.0]
    assert features["overdue_beyond_habit"].tolist() == [0.0] * 5
    assert features["delay_iqr_days"].tolist() == [7.0] * 5
    assert features["delays_used"].tolist() == [5] * 5
    assert not features["delays_pooled"].any()


def test_expected_dues_deflates_by_the_level_of_the_origin_week():
    features = dues(example_lines(), levels=pd.Series({ORIGIN: 2.0, ORIGIN + 1: 100.0}))
    assert features["spread_in_target"].tolist() == [25.0, 12.5, 12.5, 0.0, 0.0]
    assert features["shifted_due_in_target"].tolist() == [50.0, 0.0, 0.0, 0.0, 0.0]
    # Not amounts: unchanged by the level.
    assert features["spread_share_in_target"].tolist() == [0.5, 0.25, 0.25, 0.0, 0.0]
    assert features["delay_iqr_days"].tolist() == [7.0] * 5
    assert features["delays_used"].tolist() == [5] * 5


def test_expected_dues_deflates_overdue_beyond_habit():
    features = dues(example_lines(due="2025-11-03"), levels=pd.Series({ORIGIN: 4.0}))
    assert features["overdue_beyond_habit"].tolist() == [25.0] * 5
    assert features["spread_share_in_target"].tolist() == [0.0] * 5


def test_expected_dues_each_origin_has_its_own_level():
    # The same invoice at origins 7 and 8: at 7 (cutoff 1 March) all five arrivals are
    # possible; at 8 four. Levels 1 and 2.
    features = dues(example_lines(), panel_rows=panel(origins=(7, 8)),
                    levels=pd.Series({7: 1.0, 8: 2.0}))
    at_7 = features[features["origin_week"] == 7]["spread_in_target"].tolist()
    at_8 = features[features["origin_week"] == 8]["spread_in_target"].tolist()
    # At 7: Thu 5 (t+1), Mon 9 x2 (t+2), Mon 16 (t+3), Mon 23 (t+4): 20 / 40 / 20 / 20.
    assert at_7 == [20.0, 40.0, 20.0, 20.0, 0.0]
    assert at_8 == [25.0, 12.5, 12.5, 0.0, 0.0]


def test_expected_dues_shifted_past_counts_in_no_h():
    features = dues(example_lines(due="2026-02-16"))
    assert features["shifted_due_in_target"].tolist() == [0.0] * 5
    assert features["spread_in_target"].tolist() == [100.0, 0.0, 0.0, 0.0, 0.0]


def test_expected_dues_follows_the_order_of_the_panel_rows():
    lines = example_lines() + history([1, 2, 3], customer="B")
    rows = panel(keys=("TR02-A", "TR02-B")).sample(frac=1.0, random_state=3)
    rows.index = range(100, 100 + len(rows))
    features = dues(lines, panel_rows=rows)
    assert features[["key", "origin_week", "h"]].to_numpy().tolist() == rows[
        ["key", "origin_week", "h"]].to_numpy().tolist()
    assert features.index.tolist() == list(range(len(rows)))


def test_expected_dues_a_key_with_nothing_open_has_a_nan_share():
    lines = example_lines() + history([1, 2, 3], customer="B")
    features = dues(lines, panel_rows=panel(keys=("TR02-A", "TR02-B")))
    b = features[features["key"] == "TR02-B"]
    assert b["spread_in_target"].tolist() == [0.0] * 5
    assert b["spread_share_in_target"].isna().all()
    assert b["delays_used"].tolist() == [3] * 5


def test_expected_dues_a_key_without_delays():
    lines = [invoice("ZOPEN", 50.0, "2026-01-10", "2026-03-04", customer="Z",
                     company="TR09")] + history(EXAMPLE_DELAYS)
    features = dues(lines, panel_rows=panel(keys=("TR02-A", "TR09-Z")))
    z = features[features["key"] == "TR09-Z"]
    assert z["delay_iqr_days"].isna().all()
    assert z["delays_used"].tolist() == [0] * 5
    assert not z["delays_pooled"].any()
    assert z["spread_in_target"].tolist() == [0.0] * 5
    assert z["spread_share_in_target"].tolist() == [0.0] * 5   # open 50, nothing spread


def test_expected_dues_pools_by_the_panel_company_code():
    lines = history(EXAMPLE_DELAYS, customer="B", company="TR04") + [
        invoice("OPEN", 100.0, "2026-01-10", "2026-03-02", customer="A", company="TR02")]
    rows = panel(keys=("TR02-A",), companies={"TR02-A": "TR04"})
    features = dues(lines, panel_rows=rows)
    assert features["delays_pooled"].all()
    assert features["spread_in_target"].tolist() == [50.0, 25.0, 25.0, 0.0, 0.0]


def test_expected_dues_sum_over_h_is_the_spread_inside_the_horizons():
    lines = example_lines() + [invoice("OPEN-B", 40.0, "2026-01-10", "2026-03-09")]
    features = dues(lines, horizons=3)
    # Weeks 1-3: 58 + 41 + 33; week 4 (8) is spread_later.
    assert features["spread_in_target"].sum() == pytest.approx(132.0)
    last = features[features["h"] == 3].iloc[0]
    assert last["spread_before_target"] + last["spread_in_target"] == pytest.approx(132.0)


def test_expected_dues_a_foreign_invoice_at_the_rate_known_at_t():
    rates = rates_table([(date(2026, 1, 2), "EUR", 35.0), (date(2026, 3, 6), "EUR", 36.0),
                         (date(2026, 3, 9), "EUR", 99.0)])
    lines = ([invoice("OPEN", 100.0, "2026-01-10", "2026-03-02", currency="EUR")]
             + history(EXAMPLE_DELAYS))
    features = dues(lines, rates=rates, levels=pd.Series({ORIGIN: 2.0}),
                    leak_check_weeks=(ORIGIN,))
    assert features["spread_in_target"].tolist() == [900.0, 450.0, 450.0, 0.0, 0.0]
    assert features["shifted_due_in_target"].tolist() == [1800.0, 0.0, 0.0, 0.0, 0.0]


def test_expected_dues_no_price_level_raises():
    with pytest.raises(ValueError, match="no price level for weeks \\[8\\]"):
        dues(example_lines(), levels=pd.Series({7: 1.0}))


@pytest.mark.parametrize("level", [0.0, -1.0])
def test_expected_dues_a_level_of_zero_or_below_raises(level):
    with pytest.raises(ValueError, match="0 or below for weeks \\[8\\]"):
        dues(example_lines(), levels=pd.Series({ORIGIN: level}))


def test_expected_dues_a_nan_level_raises():
    with pytest.raises(ValueError, match="no price level"):
        dues(example_lines(), levels=pd.Series({ORIGIN: float("nan")}))


def test_expected_dues_is_deterministic_and_does_not_change_its_inputs():
    table = ledger(example_lines())
    rows = panel()
    levels = pd.Series({ORIGIN: 1.5})
    copies = (table.copy(), rows.copy(), levels.copy())
    first = expected_dues(rows, table, NO_RATES, levels, WEEK_ZERO, NO_HOLIDAYS, 5, 3, 20,
                          200, (ORIGIN,))
    second = expected_dues(rows, table, NO_RATES, levels, WEEK_ZERO, NO_HOLIDAYS, 5, 3, 20,
                           200, (ORIGIN,))
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(copies[0], table)
    pd.testing.assert_frame_equal(copies[1], rows)
    pd.testing.assert_series_equal(copies[2], levels)


def test_expected_dues_uses_the_holidays_given():
    features = dues(example_lines(), holidays=holidays_on((3, 5), (3, 6)))
    assert features["spread_in_target"].tolist() == [60.0, 20.0, 20.0, 0.0, 0.0]


# --- expected_dues: the leak check ----------------------------------------------------------

def leaky_lines():
    """The running example plus what happened after t: the open invoice paid and cleared on
    12 March (a new delay), a new invoice posted 10 March, and a new rate."""
    return example_lines() + [
        payment("PAY-OPEN", -100.0, "2026-03-12", reference="OPEN", cleared="2026-03-12",
                clearing_document="CLR-OPEN"),
        invoice("NEW", 55.0, "2026-03-10", "2026-03-11")]


def leaky_ledger():
    table = ledger(leaky_lines())
    # The open invoice is cleared on 12 March, with its payment.
    table.loc[table["document_no"] == "OPEN", ["clearing_date", "clearing_document"]] = [
        pd.Timestamp("2026-03-12"), "CLR-OPEN"]
    return table


def test_lines_after_t_do_not_change_the_features_of_t():
    full = leaky_ledger()
    rows = panel()
    levels = pd.Series({ORIGIN: 1.0})
    with_future = expected_dues(rows, full, NO_RATES, levels, WEEK_ZERO, NO_HOLIDAYS, 5, 3,
                                20, 200, (ORIGIN,))
    as_known = expected_dues(rows, ledger_as_known_at(full, CUTOFF), NO_RATES, levels,
                             WEEK_ZERO, NO_HOLIDAYS, 5, 3, 20, 200, ())
    pd.testing.assert_frame_equal(with_future, as_known)
    assert with_future["spread_in_target"].tolist() == [50.0, 25.0, 25.0, 0.0, 0.0]


def test_the_leak_check_passes_at_several_weeks():
    rows = panel(origins=(6, 7, 8, 9))
    levels = pd.Series({6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0})
    expected_dues(rows, leaky_ledger(), NO_RATES, levels, WEEK_ZERO, NO_HOLIDAYS, 5, 3, 20,
                  200, (6, 7, 8, 9))


def test_the_leak_check_skips_weeks_that_are_not_origins():
    expected_dues(panel(), leaky_ledger(), NO_RATES, pd.Series({ORIGIN: 1.0}), WEEK_ZERO,
                  NO_HOLIDAYS, 5, 3, 20, 200, (0, 99, -3))


def test_the_leak_check_catches_a_feature_that_used_the_future(monkeypatch):
    # A spread that peeks: it adds the number of ledger lines posted after t.
    real = rhythm.invoice_spread_at

    def peeking(ledger_then, delays, rates, keys, origin, *args):
        table = real(ledger_then, delays, rates, keys, origin, *args)
        cutoff = pd.Timestamp(rhythm.end_of_week(WEEK_ZERO, origin))
        later = int((ledger_then["posting_date"] > cutoff).sum())
        table["delay_iqr_days"] = table["delay_iqr_days"] + later
        return table

    monkeypatch.setattr(rhythm, "invoice_spread_at", peeking)
    with pytest.raises(ValueError, match="leak check: the delay spread of week 8 differs"):
        expected_dues(panel(), leaky_ledger(), NO_RATES, pd.Series({ORIGIN: 1.0}), WEEK_ZERO,
                      NO_HOLIDAYS, 5, 3, 20, 200, (ORIGIN,))


def test_the_leak_check_catches_rates_from_the_future(monkeypatch):
    # The stored run converts at the last rate of the whole table, the rebuild at the rate
    # known then: they differ, and the check says so.
    real = rhythm.rates_known_at
    monkeypatch.setattr(rhythm, "rates_known_at",
                        lambda rates, cutoff: real(rates, date(2030, 1, 1)))
    rates = rates_table([(date(2026, 1, 2), "EUR", 35.0), (date(2026, 3, 20), "EUR", 40.0)])
    lines = ([invoice("OPEN", 100.0, "2026-01-10", "2026-03-02", currency="EUR")]
             + history(EXAMPLE_DELAYS))
    with pytest.raises(ValueError, match="leak check"):
        dues(lines, rates=rates, leak_check_weeks=(ORIGIN,))


def test_the_leak_check_with_an_origin_that_has_nothing_open_and_one_that_has():
    # dtype stability across origins: week 3 (cutoff 25 Jan) has delays but no open invoice
    # (OPEN is posted 20 Feb); week 8 has both. The stored table is the concatenation of both.
    lines = [invoice("OPEN", 100.0, "2026-02-20", "2026-03-02")] + history(EXAMPLE_DELAYS)
    rows = panel(origins=(3, 8))
    features = dues(lines, panel_rows=rows, levels=pd.Series({3: 1.0, 8: 1.0}),
                    leak_check_weeks=(3, 8))
    assert features[features["origin_week"] == 3]["spread_share_in_target"].isna().all()


def test_the_leak_check_with_an_origin_before_any_delay_is_known():
    # FINDING candidate (same root as test_with_no_delays_anywhere_...): week 0 (cutoff
    # 11 Jan) knows no cleared invoice yet in this ledger, week 8 does.
    # History due 12 Jan + 7i: the first arrives Thu 15 Jan, after week 0's Sunday.
    lines = [invoice("OPEN", 100.0, "2026-01-10", "2026-03-02")] + history(
        EXAMPLE_DELAYS, start="2026-01-12", step=7)
    rows = panel(origins=(0, 8))
    features = dues(lines, panel_rows=rows, levels=pd.Series({0: 1.0, 8: 1.0}),
                    leak_check_weeks=(0, 8))
    week_0 = features[features["origin_week"] == 0]
    assert week_0["delays_used"].tolist() == [0] * 5


# --- integration with the synthetic ledger (SLOW) ------------------------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
HOLIDAYS_FILE = Path(__file__).resolve().parents[2] / "settings" / "holidays_tr.toml"
AS_OF = date(2026, 8, 28)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]
SYNTHETIC_WEEK_ZERO = date(2025, 6, 2)


@pytest.fixture(scope="module")
def synthetic():
    if not LEDGER_FILE.exists():
        pytest.skip("the synthetic ledger is not in data/raw")
    from collection_estimation.hurdle.holidays import load_holidays
    from collection_estimation.hurdle.keys import add_keys
    from collection_estimation.hurdle.read_ledger import read_ledger
    from collection_estimation.hurdle.receivables_asof import rate_lines
    from tests.hurdle.workbooks import SYNTHETIC_LEDGER_LAYOUT
    keyed = add_keys(read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT),
                     KNOWN_COMPANIES)   # direct read, no cache
    lines = rate_lines(keyed, "posting_date", "document_currency", "amount_document",
                       "amount_local", "the ledger")
    rates = lines.groupby(["day", "currency"], as_index=False)["rate"].median()
    holidays = load_holidays(HOLIDAYS_FILE)
    return keyed, rates, holidays


def test_slow_synthetic_spread_ties_out_and_passes_the_leak_check(synthetic):
    keyed, rates, holidays = synthetic
    keys = keyed[["key", "company_code"]].drop_duplicates("key")
    rows = pd.concat([keys.assign(origin_week=origin, h=h) for origin in (30, 31)
                      for h in range(1, 6)], ignore_index=True)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        features = expected_dues(rows, keyed, rates, pd.Series({30: 1.0, 31: 1.0}),
                                 SYNTHETIC_WEEK_ZERO, holidays, 5, 3, 20, 200, (30, 31))
    assert len(features) == len(rows)
    assert list(features.columns) == ["key", "origin_week", "h"] + EXPECTED_DUE_COLUMNS
    assert (features["spread_in_target"] >= 0).all()
    assert features["delays_used"].gt(0).mean() > 0.5
    share = features["spread_share_in_target"].dropna()
    assert ((share >= 0) & (share <= 1 + 1e-9)).all()


def test_slow_synthetic_key_table_ties_out_per_key(synthetic):
    keyed, rates, holidays = synthetic
    keys = keyed[["key", "company_code"]].drop_duplicates("key")
    calendar = next_business_days(date(2025, 5, 1), date(2026, 12, 31), holidays,
                                  SYNTHETIC_WEEK_ZERO)
    with contextlib.redirect_stdout(io.StringIO()):
        table = invoice_spread_at(keyed, cleared_invoice_delays(keyed), rates, keys, 30,
                                  SYNTHETIC_WEEK_ZERO, calendar, 5, 3, 20, 200)
    spread_part = (table[spread_columns(5)].sum(axis=1) + table["spread_later"]
                   + table["overdue_beyond_habit"] + table["without_delays"])
    shifted_part = (table["shifted_past"] + table[shifted_columns(5)].sum(axis=1)
                    + table["shifted_later"] + table["without_delays"])
    assert (spread_part - table["open_outstanding"]).abs().max() <= 0.01
    assert (shifted_part - table["open_outstanding"]).abs().max() <= 0.01
    assert table["open_outstanding"].sum() > 0
    assert len(table) == len(keys)
