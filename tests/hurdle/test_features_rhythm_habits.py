"""Tests for U18.3, `hurdle/features_rhythm.py`: the payment habits.

A customer's payment days are the days its money arrived: the daily files' lines
(`collections`: key, document_date, amount_try) and, before the first daily file, the
ledger's payment lines (sign of amount_local flipped). Negative amounts count as 0 per line;
a key and day whose amounts add up to 0 is not a payment day. From its last
`payment_days_kept` payment days known by the end of week t: the shares at the start, middle
and end of the month, the median gap and its IQR, the payment-due ratio, and per week after t
the month-place habit, the weekday habit and the share of possible next payments (last
payment day + a past gap, moved to the next business day, only those after t).

The module docstring's worked example is pinned here: payments on 31 Jan, 28 Feb, 31 Mar,
29 Apr, 30 May and 28 Jun, forecast at the end of Sunday 7 July. It needs a year that is not
a leap year (the docstring's gaps 28, 31, 29, 31, 29 need a 28-day February) with 7 July on a
Sunday: 2019, with week zero Monday 31 December 2018.

The other tests use the U18.2 tests' week zero (Monday 5 January 2026) and holiday helpers.
The test ledger builder of test_receivables_at.py has no amount_local; `ledger_lines` adds
it (equal to amount_document: TRY lines).
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import features_rhythm as rhythm
from collection_estimation.hurdle.features_rhythm import (
    HABIT_COLUMNS,
    LAST_DAY_FROM_START,
    MID_MONTH_DAYS,
    MONTH_END_LAST_DAYS,
    MONTH_START_DAYS,
    PAYMENT_DAY_COLUMNS,
    business_weekdays,
    habit_statistics,
    habit_week_columns,
    habits_at,
    key_habit_columns,
    month_place,
    next_business_days,
    next_payment_shares,
    payment_days,
    payment_habits,
    payment_samples_at,
    target_week_habits,
    week_dates,
)
from collection_estimation.hurdle.holidays import HolidayTable, public_holidays
from tests.hurdle.test_features_rhythm import (
    EVERY_DAY,
    NO_HOLIDAYS,
    WEEK_ZERO,
    bayrams,
    business,
    day,
    holidays_on,
)
from tests.hurdle.test_receivables_at import invoice, ledger, payment

NAN = float("nan")
ORIGIN = 8                       # Monday 2 March to Sunday 8 March 2026
CUTOFF_DAY = day("2026-03-08")
FIRST_FILE_DAY = date(2026, 1, 1)
PER_KEY_HABITS = ["month_start_share", "mid_month_share", "month_end_share",
                  "median_gap_days", "gap_iqr_days", "payment_due_ratio"]


def collections_of(rows):
    """Daily-file lines from (key, date text, amount_try) triples."""
    return pd.DataFrame({
        "key": pd.Series([r[0] for r in rows], dtype=object),
        "document_date": pd.to_datetime(pd.Series([r[1] for r in rows], dtype=object)),
        "amount_try": pd.Series([float(r[2]) for r in rows], dtype="float64"),
    })


def ledger_lines(lines=()):
    """A test ledger with amount_local (TRY lines: the same as amount_document)."""
    table = ledger(list(lines))
    table["amount_local"] = table["amount_document"].astype("float64")
    return table


NO_LEDGER = ledger_lines()

# A Monday payer: every Monday from 5 January to 2 March 2026 (9 payment days, gaps 7).
MONDAYS = ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26", "2026-02-02",
           "2026-02-09", "2026-02-16", "2026-02-23", "2026-03-02"]


def mondays(key="TR02-A", amount=100.0):
    return [(key, text, amount) for text in MONDAYS]


def days_of(rows, ledger_table=None, first_file_day=FIRST_FILE_DAY, use_ledger=True):
    return payment_days(collections_of(rows), NO_LEDGER if ledger_table is None else ledger_table,
                        first_file_day, WEEK_ZERO, use_ledger)


def samples_of(rows, keys=("TR02-A",), cutoff_day=CUTOFF_DAY, kept=24):
    return payment_samples_at(days_of(rows), pd.Series(list(keys), dtype=object), cutoff_day,
                              kept)


def panel(keys=("TR02-A",), origins=(ORIGIN,), horizons=5):
    rows = [{"key": key, "origin_week": origin, "h": h}
            for origin in origins for key in keys for h in range(1, horizons + 1)]
    return pd.DataFrame(rows)


def habits(rows, panel_rows=None, ledger_table=None, holidays=NO_HOLIDAYS,
           first_file_day=FIRST_FILE_DAY, horizons=5, kept=24, minimum=3, leak_check_weeks=(),
           week_zero=WEEK_ZERO, use_ledger=True):
    return payment_habits(panel() if panel_rows is None else panel_rows, collections_of(rows),
                          NO_LEDGER if ledger_table is None else ledger_table, week_zero,
                          holidays, first_file_day, use_ledger, horizons, kept, minimum,
                          leak_check_weeks)


def key_row(table, key="TR02-A"):
    found = table[table["key"] == key]
    assert len(found) == 1
    return found.iloc[0]


def at_h(features, h, key="TR02-A", origin=ORIGIN):
    found = features[(features["key"] == key) & (features["origin_week"] == origin)
                     & (features["h"] == h)]
    assert len(found) == 1
    return found.iloc[0]


def per_h(features, column, key="TR02-A", origin=ORIGIN, horizons=5):
    return [float(at_h(features, h, key, origin)[column]) for h in range(1, horizons + 1)]


def week_values(row, prefix, horizons=5):
    return [float(row[f"{prefix}_week_{k}"]) for k in range(1, horizons + 1)]


# --- the worked example of the module docstring (2019) ---------------------------------------

EXAMPLE_WEEK_ZERO = date(2018, 12, 31)          # a Monday
EXAMPLE_ORIGIN = 26                             # Monday 1 July to Sunday 7 July 2019
EXAMPLE_HOLIDAYS = HolidayTable(fixed={}, bayrams=bayrams((2018, 2019, 2020)))
EXAMPLE_DATES = ["2019-01-31", "2019-02-28", "2019-03-31", "2019-04-29", "2019-05-30",
                 "2019-06-28"]


def example_features():
    rows = [("TR02-A", text, 100.0) for text in EXAMPLE_DATES]
    return habits(rows, panel_rows=panel(origins=(EXAMPLE_ORIGIN,)),
                  holidays=EXAMPLE_HOLIDAYS, first_file_day=date(2019, 1, 1),
                  leak_check_weeks=(EXAMPLE_ORIGIN,), week_zero=EXAMPLE_WEEK_ZERO)


def test_example_origin_ends_on_sunday_7_july_2019():
    assert rhythm.end_of_week(EXAMPLE_WEEK_ZERO, EXAMPLE_ORIGIN) == date(2019, 7, 7)
    assert date(2019, 7, 7).weekday() == 6


def test_example_gaps_are_the_docstrings():
    days = payment_days(collections_of([("K", t, 1.0) for t in EXAMPLE_DATES]), NO_LEDGER,
                        date(2019, 1, 1), EXAMPLE_WEEK_ZERO, True)
    samples = payment_samples_at(days, pd.Series(["K"]), 10_000, 24)
    assert samples["gap"].tolist()[1:] == [28.0, 31.0, 29.0, 31.0, 29.0]
    assert np.isnan(samples["gap"].iloc[0])


def test_example_every_payment_is_at_the_month_end():
    r = at_h(example_features(), 1, origin=EXAMPLE_ORIGIN)
    assert r["payment_days_used"] == 6
    assert (r["month_start_share"], r["mid_month_share"], r["month_end_share"]) == (0.0, 0.0, 1.0)


def test_example_median_gap_iqr_and_due_ratio():
    r = at_h(example_features(), 1, origin=EXAMPLE_ORIGIN)
    assert r["median_gap_days"] == 29.0
    # Gaps 28, 29, 29, 31, 31: linear q75 31 - q25 29.
    assert r["gap_iqr_days"] == 2.0
    # 28 June to 7 July is 9 days, over 29.
    assert r["payment_due_ratio"] == pytest.approx(9 / 29)


def test_example_month_habit_is_all_in_the_week_of_29_july():
    # The places are -1 (31 Jan, 28 Feb, 31 Mar), -2 (29 Apr, 30 May) and -3 (28 Jun).
    # 22-28 July (t+3) has places -10 ... -4; 29 July - 4 August (t+4: places -3, -2, -1,
    # 1 ... 4) holds all of them, as the docstring says.
    assert per_h(example_features(), "month_habit_in_target",
                 origin=EXAMPLE_ORIGIN) == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_example_next_payments_each_count_one_fifth():
    # 26 (Fri), 29 (Mon), 27 (Sat -> Mon 29), 29, 27 (-> Mon 29) July: all after t.
    # Fri 26 is in t+3 (22-28 July), the four Mondays 29 July in t+4.
    shares = per_h(example_features(), "next_payment_share_in_target", origin=EXAMPLE_ORIGIN)
    assert shares == pytest.approx([0.0, 0.0, 0.2, 0.8, 0.0])
    assert sum(shares) == pytest.approx(1.0)


def test_example_weekday_habit():
    # Weekdays: Thu 31 Jan, Thu 28 Feb, Sun 31 Mar, Mon 29 Apr, Thu 30 May, Fri 28 Jun. The
    # Sunday is never a business day: 5 / 6 in every week without a holiday.
    assert per_h(example_features(), "weekday_habit_in_target",
                 origin=EXAMPLE_ORIGIN) == pytest.approx([5 / 6] * 5)


def test_example_per_key_values_are_the_same_for_every_h():
    features = example_features()
    for column in ["payment_days_used"] + PER_KEY_HABITS:
        assert len(set(features[column].tolist())) == 1, column


# --- columns and constants -------------------------------------------------------------------

def test_habit_columns_in_order():
    assert HABIT_COLUMNS == ["payment_days_used", "month_start_share", "mid_month_share",
                             "month_end_share", "median_gap_days", "gap_iqr_days",
                             "payment_due_ratio", "month_habit_in_target",
                             "weekday_habit_in_target", "next_payment_share_in_target"]


def test_payment_day_columns():
    assert PAYMENT_DAY_COLUMNS == ["key", "date", "day", "gross_receipts"]


def test_constants():
    assert LAST_DAY_FROM_START == 15
    assert MONTH_START_DAYS == (1, 5)
    assert MID_MONTH_DAYS == (11, 20)
    assert MONTH_END_LAST_DAYS == 5


def test_habit_week_columns():
    assert habit_week_columns("next_payment", 3) == [
        "next_payment_week_1", "next_payment_week_2", "next_payment_week_3"]
    assert habit_week_columns("x", 1) == ["x_week_1"]
    assert habit_week_columns("x", 0) == []


def test_key_habit_columns_in_order():
    assert key_habit_columns(2) == [
        "key", "origin_week", "payment_days_used", "month_start_share", "mid_month_share",
        "month_end_share", "median_gap_days", "gap_iqr_days", "payment_due_ratio",
        "month_habit_week_1", "month_habit_week_2", "weekday_habit_week_1",
        "weekday_habit_week_2", "next_payment_week_1", "next_payment_week_2"]


def test_key_habit_columns_hold_the_per_key_habit_columns_in_the_same_order():
    per_key = [c for c in key_habit_columns(5) if c in HABIT_COLUMNS]
    assert per_key == HABIT_COLUMNS[:7]


# --- month_place -----------------------------------------------------------------------------

@pytest.mark.parametrize("text, place", [
    ("2026-01-01", 1), ("2026-01-05", 5), ("2026-01-15", 15),
    ("2026-01-16", -16),              # the 16th of a 31-day month
    ("2026-01-31", -1), ("2026-01-27", -5), ("2026-01-26", -6),
    ("2026-02-27", -2),               # 27 February, not a leap year
    ("2026-03-30", -2),               # 30 March: lines up with 27 February
    ("2026-02-28", -1), ("2026-02-16", -13), ("2026-02-15", 15),
    ("2024-02-28", -2), ("2024-02-29", -1),   # a leap year
    ("2026-04-30", -1), ("2026-04-16", -15), ("2026-04-26", -5),
])
def test_month_place(text, place):
    assert month_place(pd.Series(pd.to_datetime([text]))).tolist() == [place]


def test_month_place_27_february_and_30_march_line_up():
    places = month_place(pd.Series(pd.to_datetime(["2026-02-27", "2026-03-30"])))
    assert places.tolist() == [-2, -2]


def test_month_place_over_a_whole_year_is_never_0_and_in_range():
    dates = pd.Series(pd.date_range("2024-01-01", "2025-12-31", freq="D"))
    places = month_place(dates)
    assert (places != 0).all()
    assert places.between(1, 15).sum() == 24 * 15
    assert places.between(-17, -1).sum() == len(dates) - 24 * 15
    # The last day of every month is -1, the first is +1.
    assert (places[dates.dt.is_month_end] == -1).all()
    assert (places[dates.dt.day == 1] == 1).all()


def test_month_place_is_int64_and_keeps_the_index():
    dates = pd.Series(pd.to_datetime(["2026-01-03", "2026-01-30"]), index=[7, 3])
    places = month_place(dates)
    assert places.dtype == "int64"
    assert places.index.tolist() == [7, 3]
    assert places.tolist() == [3, -2]


def test_month_place_of_an_empty_series_is_empty_int64():
    places = month_place(pd.Series([], dtype="datetime64[ns]"))
    assert len(places) == 0 and places.dtype == "int64"


# --- payment_days ----------------------------------------------------------------------------

def test_payment_days_columns_dtypes_and_day():
    days = days_of([("TR02-A", "2026-01-07", 50.0)])
    assert list(days.columns) == PAYMENT_DAY_COLUMNS
    assert days["key"].dtype == object
    assert days["date"].dtype == "datetime64[ns]" or str(days["date"].dtype).startswith(
        "datetime64")
    assert days["day"].dtype == "int64"
    assert days["gross_receipts"].dtype == "float64"
    assert days.iloc[0]["day"] == 2
    assert days.iloc[0]["gross_receipts"] == 50.0


def test_payment_days_sum_lines_of_a_key_and_day():
    days = days_of([("TR02-A", "2026-01-07", 50.0), ("TR02-A", "2026-01-07", 25.0),
                    ("TR02-B", "2026-01-07", 10.0)])
    assert days[["key", "gross_receipts"]].values.tolist() == [["TR02-A", 75.0],
                                                               ["TR02-B", 10.0]]


def test_payment_days_clip_negative_lines_before_summing():
    # A payment of 100 and a refund of 30 on the same day: 100, not the net 70.
    days = days_of([("TR02-A", "2026-01-07", 100.0), ("TR02-A", "2026-01-07", -30.0)])
    assert days["gross_receipts"].tolist() == [100.0]


def test_payment_days_drop_a_day_with_only_refunds():
    days = days_of([("TR02-A", "2026-01-07", -30.0), ("TR02-A", "2026-01-08", 10.0)])
    assert days["day"].tolist() == [3]


def test_payment_days_drop_a_day_of_zero_amounts():
    days = days_of([("TR02-A", "2026-01-07", 0.0), ("TR02-A", "2026-01-07", -0.0)])
    assert days.empty
    assert list(days.columns) == PAYMENT_DAY_COLUMNS


def test_payment_days_of_refunds_only_are_empty():
    days = days_of([("TR02-A", "2026-01-07", -5.0), ("TR02-B", "2026-01-09", -1.0)])
    assert days.empty


def test_payment_days_are_sorted_by_key_then_day():
    days = days_of([("TR02-B", "2026-01-09", 1.0), ("TR02-A", "2026-01-20", 1.0),
                    ("TR02-B", "2026-01-06", 1.0), ("TR02-A", "2026-01-05", 1.0)])
    assert days[["key", "day"]].values.tolist() == [["TR02-A", 0], ["TR02-A", 15],
                                                     ["TR02-B", 1], ["TR02-B", 4]]
    assert days.index.tolist() == [0, 1, 2, 3]


def test_payment_days_before_week_zero_are_negative():
    days = days_of([("TR02-A", "2026-01-04", 1.0)])
    assert days["day"].tolist() == [-1]


def test_payment_days_take_ledger_payments_before_the_first_file_with_the_sign_flipped():
    table = ledger_lines([payment("P1", -80.0, "2025-12-15"),
                          payment("P2", -20.0, "2025-12-15")])
    days = days_of([("TR02-A", "2026-01-07", 5.0)], table, first_file_day=date(2026, 1, 1))
    assert days[["day", "gross_receipts"]].values.tolist() == [[-21, 100.0], [2, 5.0]]


def test_payment_days_leave_out_ledger_payments_on_or_after_the_first_file():
    table = ledger_lines([payment("P1", -80.0, "2025-12-31"), payment("P2", -20.0, "2026-01-01"),
                          payment("P3", -20.0, "2026-01-10")])
    days = days_of([], table, first_file_day=date(2026, 1, 1))
    assert days["date"].tolist() == [pd.Timestamp("2025-12-31")]


def test_payment_days_ledger_refunds_are_not_payment_days():
    # A positive ledger payment line is money paid back: flipped, it is negative, clipped.
    table = ledger_lines([payment("R1", 40.0, "2025-12-15"), payment("P1", -10.0, "2025-12-16"),
                          payment("R2", 40.0, "2025-12-16")])
    days = days_of([], table, first_file_day=date(2026, 1, 1))
    assert days[["date", "gross_receipts"]].values.tolist() == [[pd.Timestamp("2025-12-16"),
                                                                 10.0]]


def test_payment_days_ignore_ledger_lines_that_are_not_payments():
    table = ledger_lines([invoice("I1", -500.0, "2025-12-15"),     # a negative invoice
                          invoice("I2", 500.0, "2025-12-16")])
    assert days_of([], table, first_file_day=date(2026, 1, 1)).empty


def test_payment_days_ledger_payments_use_amount_local_not_amount_document():
    table = ledger_lines([payment("P1", -10.0, "2025-12-15", currency="EUR")])
    table["amount_local"] = -350.0
    days = days_of([], table, first_file_day=date(2026, 1, 1))
    assert days["gross_receipts"].tolist() == [350.0]


def test_payment_days_ledger_keys_stay_apart():
    table = ledger_lines([payment("P1", -1.0, "2025-12-15", customer="A"),
                          payment("P2", -1.0, "2025-12-15", customer="B")])
    days = days_of([], table, first_file_day=date(2026, 1, 1))
    assert days["key"].tolist() == ["TR02-A", "TR02-B"]


def test_payment_days_of_no_lines_at_all_are_empty_with_their_columns():
    days = days_of([])
    assert days.empty
    assert list(days.columns) == PAYMENT_DAY_COLUMNS
    assert days["day"].dtype == "int64"
    assert days["gross_receipts"].dtype == "float64"


def test_payment_days_do_not_change_their_inputs():
    collections = collections_of([("TR02-A", "2026-01-07", -5.0), ("TR02-A", "2026-01-08", 3.0)])
    table = ledger_lines([payment("P1", -80.0, "2025-12-15")])
    before = (collections.copy(), table.copy())
    payment_days(collections, table, FIRST_FILE_DAY, WEEK_ZERO, True)
    pd.testing.assert_frame_equal(collections, before[0])
    pd.testing.assert_frame_equal(table, before[1])


# --- payment_samples_at ----------------------------------------------------------------------

def test_samples_columns():
    samples = samples_of(mondays())
    assert list(samples.columns) == PAYMENT_DAY_COLUMNS + ["day_of_month", "place", "weekday",
                                                           "gap"]


def test_samples_added_columns():
    samples = samples_of([("TR02-A", "2026-01-31", 1.0), ("TR02-A", "2026-02-07", 1.0)])
    assert samples["day_of_month"].tolist() == [31, 7]
    assert samples["place"].tolist() == [-1, 7]
    assert samples["weekday"].tolist() == [5, 5]      # Saturdays
    assert np.isnan(samples["gap"].iloc[0]) and samples["gap"].iloc[1] == 7.0


def test_samples_keep_the_last_n_on_or_before_the_cutoff():
    samples = samples_of(mondays(), kept=3)
    assert samples["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-02-16", "2026-02-23",
                                                                 "2026-03-02"]
    # The first kept day has no gap, even though an earlier payment day exists.
    assert np.isnan(samples["gap"].iloc[0])
    assert samples["gap"].tolist()[1:] == [7.0, 7.0]


def test_samples_cutoff_day_is_included_and_the_next_day_is_not():
    rows = [("TR02-A", "2026-03-01", 1.0), ("TR02-A", "2026-03-08", 1.0),
            ("TR02-A", "2026-03-09", 1.0)]
    samples = samples_of(rows, cutoff_day=day("2026-03-08"))
    assert samples["day"].tolist() == [day("2026-03-01"), day("2026-03-08")]


def test_samples_with_more_kept_than_history_keep_everything():
    assert len(samples_of(mondays(), kept=100)) == 9


def test_samples_only_for_the_keys_asked():
    samples = samples_of(mondays() + mondays("TR02-B") + mondays("TR04-C"),
                         keys=("TR02-B", "TR04-C"))
    assert sorted(set(samples["key"])) == ["TR02-B", "TR04-C"]


def test_samples_gaps_never_cross_keys():
    rows = [("TR02-A", "2026-01-05", 1.0), ("TR02-A", "2026-01-15", 1.0),
            ("TR02-B", "2026-01-20", 1.0), ("TR02-B", "2026-01-22", 1.0)]
    samples = samples_of(rows, keys=("TR02-A", "TR02-B"))
    assert samples["key"].tolist() == ["TR02-A", "TR02-A", "TR02-B", "TR02-B"]
    gaps = samples["gap"].tolist()
    assert np.isnan(gaps[0]) and gaps[1] == 10.0 and np.isnan(gaps[2]) and gaps[3] == 2.0


def test_samples_kept_counts_per_key():
    samples = samples_of(mondays() + mondays("TR02-B")[:2], keys=("TR02-A", "TR02-B"), kept=4)
    assert samples.groupby("key").size().to_dict() == {"TR02-A": 4, "TR02-B": 2}


def test_samples_of_a_key_without_days_are_empty():
    samples = samples_of(mondays(), keys=("TR09-Z",))
    assert samples.empty


# --- habit_statistics -----------------------------------------------------------------------

SHARE_DATES = ["2026-01-01", "2026-01-05",        # month start (1st-5th)
               "2026-01-06", "2026-01-10",        # neither
               "2026-01-11", "2026-01-20",        # mid month (11th-20th)
               "2026-01-21", "2026-01-26",        # neither (26 Jan is place -6)
               "2026-01-27", "2026-01-31"]        # month end (last 5 days: -5 ... -1)


def share_stats(dates, cutoff_day=day("2026-02-08"), key="TR02-A"):
    samples = samples_of([(key, text, 1.0) for text in dates], keys=(key,),
                         cutoff_day=cutoff_day)
    return key_row(habit_statistics(samples, cutoff_day), key)


def test_habit_statistics_columns():
    samples = samples_of([("TR02-A", t, 1.0) for t in SHARE_DATES])
    assert list(habit_statistics(samples, CUTOFF_DAY).columns) == ["key"] + PER_KEY_HABITS


def test_habit_statistics_month_shares_boundaries():
    r = share_stats(SHARE_DATES)
    assert (r["month_start_share"], r["mid_month_share"], r["month_end_share"]) == (
        0.2, 0.2, 0.2)


def test_habit_statistics_month_end_in_february_is_the_24th_to_the_28th():
    r = share_stats(["2026-02-23", "2026-02-24", "2026-02-28"], cutoff_day=day("2026-03-08"))
    assert r["month_end_share"] == pytest.approx(2 / 3)


def test_habit_statistics_month_end_in_a_leap_february_is_the_25th_to_the_29th():
    days = payment_days(collections_of([("K", "2024-02-24", 1.0), ("K", "2024-02-25", 1.0),
                                        ("K", "2024-02-29", 1.0)]), NO_LEDGER,
                        date(2024, 1, 1), WEEK_ZERO, True)
    samples = payment_samples_at(days, pd.Series(["K"]), 0, 24)
    assert key_row(habit_statistics(samples, 0), "K")["month_end_share"] == pytest.approx(2 / 3)


def test_habit_statistics_shares_need_not_add_up_to_1():
    # The 6th-10th and the 21st up to the last 5 days are in no share.
    r = share_stats(["2026-01-07", "2026-01-22"])
    assert (r["month_start_share"], r["mid_month_share"], r["month_end_share"]) == (0, 0, 0)


def test_habit_statistics_median_gap_iqr_and_due_ratio():
    # Gaps 4, 1, 4, 1, 9, 1, 5, 1, 4: sorted 1, 1, 1, 1, 4, 4, 4, 5, 9 -> median 4,
    # linear q25 1, q75 4 -> IQR 3. Cutoff 8 Feb, last payment 31 Jan: 8 days / 4 = 2.
    r = share_stats(SHARE_DATES)
    assert r["median_gap_days"] == 4.0
    assert r["gap_iqr_days"] == 3.0
    assert r["payment_due_ratio"] == 2.0


def test_habit_statistics_median_of_an_even_count_is_the_mean_of_the_middle_two():
    r = share_stats(["2026-01-01", "2026-01-04", "2026-01-11"])    # gaps 3, 7
    assert r["median_gap_days"] == 5.0
    assert r["gap_iqr_days"] == pytest.approx(2.0)                  # 6 - 4


def test_habit_statistics_regular_payer_has_iqr_0():
    r = share_stats(MONDAYS, cutoff_day=CUTOFF_DAY)
    assert r["median_gap_days"] == 7.0 and r["gap_iqr_days"] == 0.0
    assert r["payment_due_ratio"] == pytest.approx(6 / 7)            # 2 March to 8 March


def test_habit_statistics_due_ratio_of_a_payment_on_the_cutoff_day_is_0():
    r = share_stats(MONDAYS, cutoff_day=day("2026-03-02"))
    assert r["payment_due_ratio"] == 0.0


def test_habit_statistics_one_row_per_key_with_its_own_values():
    samples = samples_of(mondays("TR02-A") + [("TR02-B", t, 1.0) for t in SHARE_DATES],
                         keys=("TR02-A", "TR02-B"))
    stats = habit_statistics(samples, CUTOFF_DAY)
    assert sorted(stats["key"]) == ["TR02-A", "TR02-B"]
    assert key_row(stats, "TR02-A")["median_gap_days"] == 7.0
    assert key_row(stats, "TR02-B")["median_gap_days"] == 4.0


def test_habit_statistics_of_a_single_payment_day_has_nan_gaps():
    r = share_stats(["2026-01-31"])
    assert r["month_end_share"] == 1.0
    assert np.isnan(r["median_gap_days"]) and np.isnan(r["gap_iqr_days"])
    assert np.isnan(r["payment_due_ratio"])


def test_habit_statistics_of_no_samples_is_empty_with_its_columns():
    stats = habit_statistics(samples_of([]), CUTOFF_DAY)
    assert stats.empty
    assert list(stats.columns) == ["key"] + PER_KEY_HABITS
    assert stats["key"].dtype == object
    assert all(stats[c].dtype == "float64" for c in PER_KEY_HABITS)


def test_habit_statistics_do_not_change_the_samples():
    samples = samples_of([("TR02-A", t, 1.0) for t in SHARE_DATES])
    before = samples.copy()
    habit_statistics(samples, CUTOFF_DAY)
    pd.testing.assert_frame_equal(samples, before)


# --- week_dates and business_weekdays -------------------------------------------------------

def test_week_dates_are_monday_to_sunday_of_the_week():
    dates = week_dates(ORIGIN, WEEK_ZERO)
    assert [d.strftime("%Y-%m-%d") for d in dates] == [
        "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06", "2026-03-07",
        "2026-03-08"]
    assert dates.index.tolist() == list(range(7))


def test_week_dates_of_week_zero_and_a_negative_week():
    assert week_dates(0, WEEK_ZERO).iloc[0] == pd.Timestamp("2026-01-05")
    assert week_dates(-1, WEEK_ZERO).iloc[6] == pd.Timestamp("2026-01-04")


def test_business_weekdays_without_holidays_are_monday_to_friday():
    assert business_weekdays(week_dates(ORIGIN, WEEK_ZERO), {}) == [0, 1, 2, 3, 4]


def test_business_weekdays_a_holiday_removes_its_weekday():
    holiday_days = public_holidays(date(2026, 3, 1), date(2026, 3, 31), holidays_on((3, 4)))
    assert business_weekdays(week_dates(ORIGIN, WEEK_ZERO), holiday_days) == [0, 1, 3, 4]


def test_business_weekdays_a_holiday_on_a_weekend_changes_nothing():
    holiday_days = public_holidays(date(2026, 3, 1), date(2026, 3, 31), holidays_on((3, 7)))
    assert business_weekdays(week_dates(ORIGIN, WEEK_ZERO), holiday_days) == [0, 1, 2, 3, 4]


def test_business_weekdays_of_a_whole_holiday_week_are_empty():
    holiday_days = public_holidays(date(2026, 3, 1), date(2026, 3, 31),
                                   holidays_on((3, 2), (3, 3), (3, 4), (3, 5), (3, 6)))
    assert business_weekdays(week_dates(ORIGIN, WEEK_ZERO), holiday_days) == []


# --- target_week_habits ----------------------------------------------------------------------

def target(rows, holiday_days=None, origin=ORIGIN, horizons=5, keys=("TR02-A",),
           cutoff_day=CUTOFF_DAY):
    samples = samples_of(rows, keys=keys, cutoff_day=cutoff_day)
    return target_week_habits(samples, origin, WEEK_ZERO, holiday_days or {}, horizons)


def test_target_week_habits_columns():
    table = target(mondays(), horizons=2)
    assert list(table.columns) == ["key", "month_habit_week_1", "weekday_habit_week_1",
                                   "month_habit_week_2", "weekday_habit_week_2"]


def test_target_week_habits_monday_payer_weekday_habit_is_1_every_week():
    assert week_values(key_row(target(mondays())), "weekday_habit") == [1.0] * 5


def test_target_week_habits_a_holiday_monday_removes_the_weekday_habit_of_that_week():
    # Monday 16 March (t+2) a holiday.
    holiday_days = public_holidays(date(2026, 3, 1), date(2026, 4, 30), holidays_on((3, 16)))
    values = week_values(key_row(target(mondays(), holiday_days)), "weekday_habit")
    assert values == [1.0, 0.0, 1.0, 1.0, 1.0]


def test_target_week_habits_a_holiday_on_another_weekday_changes_nothing():
    holiday_days = public_holidays(date(2026, 3, 1), date(2026, 4, 30), holidays_on((3, 18)))
    assert week_values(key_row(target(mondays(), holiday_days)), "weekday_habit") == [1.0] * 5


def test_target_week_habits_weekend_payments_never_match_a_weekday():
    rows = [("TR02-A", t, 1.0) for t in ["2026-02-07", "2026-02-14", "2026-02-22"]]   # Sat, Sat, Sun
    assert week_values(key_row(target(rows)), "weekday_habit") == [0.0] * 5


def test_target_week_habits_month_habit_at_a_month_boundary():
    # t+4 is Mon 30 March - Sun 5 April: places -2, -1, 1, 2, 3, 4, 5. Payment days on
    # 31 Jan (-1), 27 Feb (-2), 3 Feb (3) and 10 Feb (10): 3 of 4 in t+4.
    rows = [("TR02-A", t, 1.0) for t in ["2026-01-31", "2026-02-03", "2026-02-10",
                                          "2026-02-27"]]
    values = week_values(key_row(target(rows)), "month_habit")
    assert values[3] == 0.75
    # t+1 9-15 March: places 9 ... 15 -> only the 10th.
    assert values[0] == 0.25
    # t+2 16-22 March: places -16 ... -10 -> none; t+3 23-29 March: -9 ... -3 -> none.
    assert values[1:3] == [0.0, 0.0]


def test_target_week_habits_month_habit_lines_up_month_ends_of_different_lengths():
    # 28 Feb (-1) and 30 April (-1) both match 31 March (-1) in t+4.
    rows = [("TR02-A", t, 1.0) for t in ["2025-04-30", "2026-02-28"]]
    assert week_values(key_row(target(rows)), "month_habit")[3] == 1.0


def test_target_week_habits_one_row_per_key_with_its_own_values():
    rows = mondays("TR02-A") + [("TR02-B", t, 1.0) for t in ["2026-02-07", "2026-02-14"]]
    table = target(rows, keys=("TR02-A", "TR02-B"))
    assert table["key"].tolist() == ["TR02-A", "TR02-B"]
    assert week_values(key_row(table, "TR02-B"), "weekday_habit") == [0.0] * 5


def test_target_week_habits_of_no_samples_has_no_rows():
    table = target([])
    assert table.empty


def test_target_week_habits_do_not_change_the_samples():
    samples = samples_of(mondays())
    before = samples.copy()
    target_week_habits(samples, ORIGIN, WEEK_ZERO, {}, 5)
    pd.testing.assert_frame_equal(samples, before)


# --- next_payment_shares ---------------------------------------------------------------------

def next_shares(rows, next_business=None, origin=ORIGIN, horizons=5, keys=("TR02-A",),
                cutoff_day=CUTOFF_DAY):
    samples = samples_of(rows, keys=keys, cutoff_day=cutoff_day)
    return next_payment_shares(samples, origin,
                               EVERY_DAY if next_business is None else next_business, horizons)


def dates_for(*texts, key="TR02-A"):
    return [(key, t, 1.0) for t in texts]


def test_next_payment_shares_columns():
    assert list(next_shares(mondays(), horizons=3).columns) == [
        "key", "next_payment_week_1", "next_payment_week_2", "next_payment_week_3"]


def test_next_payment_shares_monday_payer_all_in_week_1():
    # 2 March + 7 = Monday 9 March, t+1, for every one of the 8 gaps.
    assert week_values(key_row(next_shares(mondays())), "next_payment") == [1.0, 0, 0, 0, 0]


def test_next_payment_shares_equal_shares_and_repeated_gaps_count_each():
    # Last day 2 March.
    rows = dates_for("2026-01-05", "2026-01-12", "2026-01-26", "2026-02-02", "2026-02-23",
                     "2026-03-02")
    # gaps: 7, 14, 7, 21, 7 -> t+1 3/5, t+2 1/5, t+3 1/5
    values = week_values(key_row(next_shares(rows)), "next_payment")
    assert values == pytest.approx([0.6, 0.2, 0.2, 0.0, 0.0])


def test_next_payment_shares_only_candidates_after_t_count():
    # Last day Mon 2 March; gaps 2 (Wed 4 March, in t: impossible), 7 and 14.
    rows = dates_for("2026-02-12", "2026-02-26", "2026-02-28", "2026-03-02")
    # gaps 14, 2, 2: two impossible (4 March), one in t+2 (16 March) -> 1.0 in t+2.
    assert week_values(key_row(next_shares(rows)), "next_payment") == [0, 1.0, 0, 0, 0]


def test_next_payment_shares_all_on_or_before_t_give_0_every_week():
    # Gaps of 1 day, last payment Thu 5 March: every candidate is Fri 6 March, in t.
    rows = dates_for("2026-03-03", "2026-03-04", "2026-03-05")
    table = next_shares(rows)
    assert table["key"].tolist() == ["TR02-A"]
    assert week_values(key_row(table), "next_payment") == [0.0] * 5


def test_next_payment_shares_empty_possible_frame_for_every_key():
    # Both keys' only candidates (last day + a 1-day gap) fall in t.
    rows = (dates_for("2026-03-03", "2026-03-04", "2026-03-05")
            + dates_for("2026-03-04", "2026-03-05", key="TR02-B"))
    table = next_shares(rows, keys=("TR02-A", "TR02-B"))
    assert table["key"].tolist() == ["TR02-A", "TR02-B"]
    for key in ("TR02-A", "TR02-B"):
        assert week_values(key_row(table, key), "next_payment") == [0.0] * 5
    assert all(table[c].dtype == "float64" for c in table.columns if c != "key")


def test_next_payment_shares_a_candidate_on_the_sunday_of_t_moves_to_monday_and_counts():
    # Last day Wed 4 March, gap 4 -> Sun 8 March (t), moved to Mon 9 March (t+1).
    rows = dates_for("2026-02-28", "2026-03-04")
    assert week_values(key_row(next_shares(rows, next_business=business())),
                       "next_payment") == [1.0, 0, 0, 0, 0]
    # With every day a business day it stays on the Sunday of t: no candidate after t.
    assert week_values(key_row(next_shares(rows)), "next_payment") == [0.0] * 5


def test_next_payment_shares_a_holiday_friday_pushes_a_candidate_into_the_next_week():
    # Last day Fri 6 March, gap 7 -> Fri 13 March (t+1); a holiday on the 13th -> Mon 16 (t+2).
    rows = dates_for("2026-02-27", "2026-03-06")
    assert week_values(key_row(next_shares(rows, next_business=business())),
                       "next_payment") == [1.0, 0, 0, 0, 0]
    assert week_values(key_row(next_shares(rows, next_business=business(holidays_on((3, 13))))),
                       "next_payment") == [0, 1.0, 0, 0, 0]


def test_next_payment_shares_candidates_past_the_horizons_still_share():
    # Last day 2 March, gaps 7 (t+1) and 70 (11 May, t+10): each 1/2; only t+1 is shown.
    rows = dates_for("2025-12-15", "2025-12-22", "2026-03-02")   # gaps 7 and 70
    values = week_values(key_row(next_shares(rows)), "next_payment")
    assert values == [0.5, 0.0, 0.0, 0.0, 0.0]


def test_next_payment_shares_a_single_payment_day_has_no_candidate():
    table = next_shares(dates_for("2026-03-02"))
    assert week_values(key_row(table), "next_payment") == [0.0] * 5


def test_next_payment_shares_one_row_per_key_with_its_own_values():
    rows = mondays("TR02-A") + dates_for("2026-02-27", "2026-03-06", key="TR02-B")
    table = next_shares(rows, keys=("TR02-A", "TR02-B"))
    assert week_values(key_row(table, "TR02-A"), "next_payment") == [1.0, 0, 0, 0, 0]
    assert week_values(key_row(table, "TR02-B"), "next_payment") == [1.0, 0, 0, 0, 0]


def test_next_payment_shares_of_no_samples_has_no_rows():
    assert next_shares([]).empty


def test_next_payment_shares_sum_to_at_most_1():
    rows = dates_for("2026-01-05", "2026-01-09", "2026-01-20", "2026-02-10", "2026-02-11",
                     "2026-03-02")
    total = sum(week_values(key_row(next_shares(rows, next_business=business())),
                            "next_payment"))
    assert 0.0 <= total <= 1.0 + 1e-12


# --- habits_at -------------------------------------------------------------------------------

def at(rows, keys=("TR02-A",), origin=ORIGIN, kept=24, minimum=3, holiday_days=None,
       next_business=None, horizons=5):
    return habits_at(days_of(rows), pd.Series(list(keys), dtype=object), origin, WEEK_ZERO,
                     holiday_days or {}, EVERY_DAY if next_business is None else next_business,
                     horizons, kept, minimum)


def test_habits_at_columns_in_order():
    assert list(at(mondays()).columns) == key_habit_columns(5)


def test_habits_at_monday_payer():
    r = key_row(at(mondays()))
    assert r["origin_week"] == ORIGIN
    assert r["payment_days_used"] == 9
    assert r["median_gap_days"] == 7.0
    assert week_values(r, "next_payment") == [1.0, 0, 0, 0, 0]
    assert week_values(r, "weekday_habit") == [1.0] * 5


def test_habits_at_payment_days_used_is_an_int_count_and_0_without_days():
    table = at(mondays(), keys=("TR02-A", "TR09-Z"))
    assert table["payment_days_used"].dtype == "int64"
    assert table["payment_days_used"].tolist() == [9, 0]


def test_habits_at_below_min_payment_days_is_nan_in_every_habit_column():
    table = at(mondays()[:2], minimum=3)
    r = key_row(table)
    assert r["payment_days_used"] == 2
    habit_columns = [c for c in key_habit_columns(5)
                     if c not in ("key", "origin_week", "payment_days_used")]
    assert all(np.isnan(r[c]) for c in habit_columns)


def test_habits_at_exactly_min_payment_days_has_habits():
    r = key_row(at(mondays()[-3:], minimum=3))
    assert r["payment_days_used"] == 3 and r["median_gap_days"] == 7.0


def test_habits_at_min_counts_the_kept_days_not_the_whole_history():
    # 9 payment days, but only 2 kept: below a minimum of 3.
    r = key_row(at(mondays(), kept=2, minimum=3))
    assert r["payment_days_used"] == 2
    assert np.isnan(r["median_gap_days"])


def test_habits_at_kept_smaller_than_the_history_uses_the_most_recent():
    # Early gaps of 14, then weekly: kept 3 gives the weekly habit only.
    rows = dates_for("2026-01-05", "2026-01-19", "2026-02-02", "2026-02-16", "2026-02-23",
                     "2026-03-02")
    assert key_row(at(rows, kept=24))["median_gap_days"] == 14.0
    assert key_row(at(rows, kept=3))["median_gap_days"] == 7.0
    assert key_row(at(rows, kept=3))["payment_days_used"] == 3


def test_habits_at_payment_days_after_the_origin_are_not_used():
    later = mondays() + dates_for("2026-03-09", "2026-03-16")
    pd.testing.assert_frame_equal(at(later), at(mondays()))


def test_habits_at_keys_without_a_habit_beside_keys_with_one():
    table = at(mondays("TR02-A") + mondays("TR02-B")[:1], keys=("TR02-B", "TR02-A", "TR09-Z"))
    assert table["key"].tolist() == ["TR02-B", "TR02-A", "TR09-Z"]
    assert table["payment_days_used"].tolist() == [1, 9, 0]
    assert np.isnan(key_row(table, "TR02-B")["median_gap_days"])
    assert np.isnan(key_row(table, "TR09-Z")["next_payment_week_1"])
    assert key_row(table, "TR02-A")["next_payment_week_1"] == 1.0


def test_habits_at_with_no_habit_anywhere_has_every_column_as_float_nan():
    table = at(mondays()[:1], keys=("TR02-A", "TR02-B"))
    assert list(table.columns) == key_habit_columns(5)
    for column in key_habit_columns(5)[3:]:
        assert table[column].dtype == "float64", column
        assert table[column].isna().all(), column


def test_habits_at_with_no_payment_days_at_all():
    table = at([], keys=("TR02-A",))
    assert table["payment_days_used"].tolist() == [0]
    assert table[key_habit_columns(5)[3:]].isna().all().all()


def test_habits_at_repeated_keys_raise():
    with pytest.raises(ValueError, match="keys are repeated"):
        at(mondays(), keys=("TR02-A", "TR02-B", "TR02-A"))


def test_habits_at_with_no_keys_is_empty_with_its_columns():
    table = at(mondays(), keys=())
    assert table.empty and list(table.columns) == key_habit_columns(5)


def test_habits_at_does_not_change_its_inputs():
    days = days_of(mondays())
    keys = pd.Series(["TR02-A"], dtype=object)
    before = (days.copy(), keys.copy())
    habits_at(days, keys, ORIGIN, WEEK_ZERO, {}, EVERY_DAY, 5, 24, 3)
    pd.testing.assert_frame_equal(days, before[0])
    pd.testing.assert_series_equal(keys, before[1])


# --- payment_habits --------------------------------------------------------------------------

def test_payment_habits_columns_and_dtypes():
    features = habits(mondays())
    assert list(features.columns) == ["key", "origin_week", "h"] + HABIT_COLUMNS
    assert features["payment_days_used"].dtype == "int64"
    for column in HABIT_COLUMNS[1:]:
        assert features[column].dtype == "float64", column
    assert features.index.tolist() == list(range(len(features)))


def test_payment_habits_monday_payer_per_h():
    features = habits(mondays())
    assert per_h(features, "next_payment_share_in_target") == [1.0, 0, 0, 0, 0]
    assert per_h(features, "weekday_habit_in_target") == [1.0] * 5
    assert per_h(features, "payment_days_used") == [9] * 5


def test_payment_habits_each_row_takes_its_own_h():
    # Month habit of a key paying on the 10th: t+1 is 9-15 March, t+5 6-12 April; t+2 to
    # t+4 (16 March - 5 April) hold no 10th.
    rows = dates_for("2025-12-10", "2026-01-10", "2026-02-10")
    features = habits(rows)
    assert per_h(features, "month_habit_in_target") == [1.0, 0.0, 0.0, 0.0, 1.0]


def test_payment_habits_follow_the_order_of_the_panel_rows():
    rows = panel(keys=("TR02-A", "TR02-B"), origins=(7, 8))
    shuffled = rows.sample(frac=1.0, random_state=3).reset_index(drop=True)
    data = mondays("TR02-A") + mondays("TR02-B")[:4]
    features = habits(data, panel_rows=shuffled)
    assert features[["key", "origin_week", "h"]].values.tolist() == \
        shuffled[["key", "origin_week", "h"]].values.tolist()
    ordered = habits(data, panel_rows=rows)
    merged = shuffled.merge(ordered, on=["key", "origin_week", "h"])
    pd.testing.assert_frame_equal(features, merged[features.columns])


def test_payment_habits_a_panel_with_a_non_default_index():
    rows = panel()
    rows.index = [40, 30, 20, 10, 0]
    features = habits(mondays(), panel_rows=rows)
    assert features.index.tolist() == [0, 1, 2, 3, 4]
    assert features["h"].tolist() == [1, 2, 3, 4, 5]


def test_payment_habits_panel_rows_may_have_more_columns():
    rows = panel()
    rows["company_code"] = "TR02"
    features = habits(mondays(), panel_rows=rows)
    assert "company_code" not in features.columns


def test_payment_habits_each_origin_has_its_own_habits():
    features = habits(mondays(), panel_rows=panel(origins=(5, 8)))
    # By the end of week 5 (Sunday 15 Feb): 6 Mondays (5 Jan - 9 Feb); by week 8: 9.
    assert per_h(features, "payment_days_used", origin=5) == [6] * 5
    assert per_h(features, "payment_days_used", origin=8) == [9] * 5


def test_payment_habits_below_min_payment_days_are_nan_but_count():
    features = habits(mondays()[:2], minimum=3)
    assert per_h(features, "payment_days_used") == [2] * 5
    assert features[HABIT_COLUMNS[1:]].isna().all().all()


def test_payment_habits_a_customer_without_payment_days():
    features = habits(mondays(), panel_rows=panel(keys=("TR02-A", "TR09-Z")))
    z = features[features["key"] == "TR09-Z"]
    assert z["payment_days_used"].tolist() == [0] * 5
    assert z[HABIT_COLUMNS[1:]].isna().all().all()


def test_payment_habits_with_no_payment_days_at_all():
    features = habits([])
    assert features["payment_days_used"].tolist() == [0] * 5
    assert features[HABIT_COLUMNS[1:]].isna().all().all()
    assert list(features.columns) == ["key", "origin_week", "h"] + HABIT_COLUMNS


def test_payment_habits_refunds_only_give_no_habit():
    rows = [("TR02-A", t, -50.0) for t in MONDAYS]
    features = habits(rows)
    assert features["payment_days_used"].tolist() == [0] * 5
    assert features[HABIT_COLUMNS[1:]].isna().all().all()


@pytest.mark.parametrize("minimum", [1, 0, -2])
def test_payment_habits_min_payment_days_below_2_raises(minimum):
    with pytest.raises(ValueError, match=rf"min_payment_days is {minimum}; it must be 2 or more"):
        habits(mondays(), minimum=minimum)


def test_payment_habits_min_payment_days_of_2_is_accepted():
    features = habits(mondays()[-2:], minimum=2)
    assert per_h(features, "median_gap_days") == [7.0] * 5
    assert per_h(features, "gap_iqr_days") == [0.0] * 5


def test_payment_habits_payment_days_kept_smaller_than_the_history():
    rows = dates_for("2026-01-05", "2026-01-19", "2026-02-02", "2026-02-16", "2026-02-23",
                     "2026-03-02")
    assert per_h(habits(rows, kept=3), "median_gap_days") == [7.0] * 5
    assert per_h(habits(rows, kept=24), "median_gap_days") == [14.0] * 5


def test_payment_habits_a_key_whose_gaps_all_land_on_or_before_t():
    rows = dates_for("2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05")
    features = habits(rows)
    assert per_h(features, "next_payment_share_in_target") == [0.0] * 5
    assert per_h(features, "median_gap_days") == [1.0] * 5


def test_payment_habits_a_holiday_target_week_lowers_the_weekday_habit():
    # Monday 16 March (t+2) a holiday: a Monday payer's weekday habit there is 0.
    features = habits(mondays(), holidays=holidays_on((3, 16)))
    assert per_h(features, "weekday_habit_in_target") == [1.0, 0.0, 1.0, 1.0, 1.0]


def test_payment_habits_a_holiday_moves_the_next_payment():
    # Last Monday 2 March + 7 = 9 March; a holiday on 9 March moves it to Tue 10 (still t+1).
    features = habits(mondays(), holidays=holidays_on((3, 9)))
    assert per_h(features, "next_payment_share_in_target") == [1.0, 0, 0, 0, 0]
    assert per_h(features, "weekday_habit_in_target")[0] == 0.0


def test_payment_habits_a_weekend_next_payment_moves_to_monday():
    # A Wednesday payer with a gap of 4 days at the end: last Wed 4 March + 4 = Sun 8 March
    # (t), moved to Mon 9 March (t+1). Gaps: 7, 7, 3 (Wed -> Sat), 4 (Sat -> Wed).
    rows = dates_for("2026-02-11", "2026-02-18", "2026-02-25", "2026-02-28", "2026-03-04")
    features = habits(rows)
    # Candidates: +7 -> 11 Mar (t+1), +7 -> 11 Mar, +3 -> Sat 7 Mar -> Mon 9 (t+1),
    # +4 -> Sun 8 Mar -> Mon 9 (t+1).
    assert per_h(features, "next_payment_share_in_target") == [1.0, 0, 0, 0, 0]


def test_payment_habits_use_the_ledger_before_the_first_file():
    # The ledger alone gives the customer its habit: payments every Monday in January,
    # files from 2 February.
    table = ledger_lines([payment(f"P{i}", -10.0, text) for i, text in enumerate(MONDAYS[:4])])
    files = mondays()[4:]
    features = habits(files, ledger_table=table, first_file_day=date(2026, 2, 2))
    assert per_h(features, "payment_days_used") == [9] * 5
    only_files = habits(files, first_file_day=date(2026, 2, 2))
    assert per_h(only_files, "payment_days_used") == [5] * 5


def test_payment_habits_ledger_lines_after_the_first_file_do_not_add_days():
    table = ledger_lines([payment("P1", -10.0, "2026-02-04"), payment("P2", -10.0, "2026-02-05")])
    with_ledger = habits(mondays(), ledger_table=table, first_file_day=date(2026, 1, 5))
    pd.testing.assert_frame_equal(with_ledger, habits(mondays(), first_file_day=date(2026, 1, 5)))


def test_payment_habits_amounts_do_not_matter():
    small = [(k, t, 1.0) for k, t, _ in mondays()]
    large = [(k, t, 1e9) for k, t, _ in mondays()]
    pd.testing.assert_frame_equal(habits(small), habits(large))


def test_payment_habits_is_deterministic_and_does_not_change_its_inputs():
    rows = panel(keys=("TR02-A", "TR02-B"), origins=(6, 8))
    collections = collections_of(mondays() + mondays("TR02-B")[:5])
    table = ledger_lines([payment("P1", -10.0, "2025-12-15")])
    before = (rows.copy(), collections.copy(), table.copy())
    first = payment_habits(rows, collections, table, WEEK_ZERO, NO_HOLIDAYS, FIRST_FILE_DAY,
                           True, 5, 24, 3, (6, 8))
    second = payment_habits(rows, collections, table, WEEK_ZERO, NO_HOLIDAYS, FIRST_FILE_DAY,
                            True, 5, 24, 3, (6, 8))
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(rows, before[0])
    pd.testing.assert_frame_equal(collections, before[1])
    pd.testing.assert_frame_equal(table, before[2])


def test_payment_habits_horizons_of_1():
    features = habits(mondays(), panel_rows=panel(horizons=1), horizons=1)
    assert features["h"].tolist() == [1]
    assert features["next_payment_share_in_target"].tolist() == [1.0]


# --- payment_habits: the leak check ----------------------------------------------------------

def leaky_rows():
    """The Monday payer plus payment days after t (9 and 12 March)."""
    return mondays() + dates_for("2026-03-09", "2026-03-12")


def test_a_future_payment_day_does_not_change_the_features_of_an_earlier_origin():
    with_future = habits(leaky_rows(), leak_check_weeks=(ORIGIN,))
    as_known = habits(mondays())
    pd.testing.assert_frame_equal(with_future, as_known)


def test_a_future_ledger_payment_does_not_change_the_features_of_an_earlier_origin():
    # The ledger before the first file (15 March): a payment posted after t (10 March).
    early = ledger_lines([payment(f"P{i}", -10.0, text) for i, text in enumerate(MONDAYS)])
    later = ledger_lines([payment(f"P{i}", -10.0, text)
                          for i, text in enumerate(MONDAYS + ["2026-03-10"])])
    a = habits([], ledger_table=early, first_file_day=date(2026, 3, 15))
    b = habits([], ledger_table=later, first_file_day=date(2026, 3, 15),
               leak_check_weeks=(ORIGIN,))
    pd.testing.assert_frame_equal(a, b)
    assert per_h(b, "payment_days_used") == [9] * 5


def test_the_leak_check_passes_at_several_weeks():
    table = ledger_lines([payment("P1", -10.0, "2025-12-15"), payment("P2", -10.0, "2025-12-29"),
                          payment("P3", -10.0, "2026-01-20")])
    habits(leaky_rows(), panel_rows=panel(keys=("TR02-A", "TR02-B"), origins=(0, 3, 6, 8, 9)),
           ledger_table=table, first_file_day=date(2026, 1, 15),
           leak_check_weeks=(0, 3, 6, 8, 9))


def test_the_leak_check_skips_weeks_that_are_not_origins():
    habits(leaky_rows(), leak_check_weeks=(0, 99, -3))


def test_the_leak_check_with_an_origin_before_any_payment_day_and_one_after():
    # dtype stability across origins: week 0 knows no payment day, week 8 knows nine.
    rows = panel(origins=(0, 8))
    features = habits(mondays()[1:], panel_rows=rows, leak_check_weeks=(0, 8))
    assert per_h(features, "payment_days_used", origin=0) == [0] * 5


def test_the_leak_check_catches_a_habit_that_used_the_future(monkeypatch):
    # A habit that peeks: it adds the number of payment days after its cutoff.
    real = rhythm.habits_at

    def peeking(days, keys, origin, week_zero, *args):
        table = real(days, keys, origin, week_zero, *args)
        cutoff_day = (rhythm.end_of_week(week_zero, origin) - week_zero).days
        later = int((days["day"] > cutoff_day).sum())
        table["median_gap_days"] = table["median_gap_days"] + later
        return table

    monkeypatch.setattr(rhythm, "habits_at", peeking)
    with pytest.raises(ValueError, match="leak check: the payment habits of week 8 differ"):
        habits(leaky_rows(), leak_check_weeks=(ORIGIN,))


def test_the_leak_check_names_the_cutoff(monkeypatch):
    real = rhythm.habits_at

    def peeking(days, keys, origin, *args):
        table = real(days, keys, origin, *args)
        table["payment_days_used"] = len(days)
        return table

    monkeypatch.setattr(rhythm, "habits_at", peeking)
    with pytest.raises(ValueError, match="cut off at 2026-03-08"):
        habits(leaky_rows(), leak_check_weeks=(ORIGIN,))


# --- the ledger switch (use_ledger_before_first_file) ----------------------------------------

EARLY_LEDGER_ONLY = [payment(f"P{i}", -10.0, text, customer="L")
                     for i, text in enumerate(MONDAYS[:4])]      # TR02-L, January only


def test_payment_days_without_the_ledger_leave_out_the_early_ledger_payments():
    table = ledger_lines(EARLY_LEDGER_ONLY)
    days = days_of(mondays(), table, first_file_day=date(2026, 2, 2), use_ledger=False)
    assert "TR02-L" not in set(days["key"])


def test_payment_days_without_the_ledger_keep_the_daily_file_days_unchanged():
    table = ledger_lines(EARLY_LEDGER_ONLY + [payment("PA", -5.0, "2025-12-15")])
    without = days_of(mondays(), table, first_file_day=date(2026, 1, 1), use_ledger=False)
    files_only = days_of(mondays(), first_file_day=date(2026, 1, 1))
    pd.testing.assert_frame_equal(without, files_only)
    assert list(without.columns) == PAYMENT_DAY_COLUMNS


def test_payment_days_without_the_ledger_and_without_files_are_empty_with_their_columns():
    days = days_of([], ledger_lines(EARLY_LEDGER_ONLY), first_file_day=date(2026, 2, 2),
                   use_ledger=False)
    assert days.empty
    assert list(days.columns) == PAYMENT_DAY_COLUMNS
    assert days["day"].dtype == "int64"
    assert days["gross_receipts"].dtype == "float64"


def test_payment_days_with_the_ledger_keep_the_early_ledger_payments():
    days = days_of(mondays(), ledger_lines(EARLY_LEDGER_ONLY), first_file_day=date(2026, 2, 2),
                   use_ledger=True)
    assert days.loc[days["key"] == "TR02-L", "day"].tolist() == [day(t) for t in MONDAYS[:4]]
    assert len(days[days["key"] == "TR02-A"]) == 9


def test_payment_habits_without_the_ledger_a_ledger_only_key_has_no_habit():
    rows = panel(keys=("TR02-A", "TR02-L"))
    features = habits(mondays(), panel_rows=rows, ledger_table=ledger_lines(EARLY_LEDGER_ONLY),
                      first_file_day=date(2026, 2, 2), use_ledger=False,
                      leak_check_weeks=(ORIGIN,))
    ledger_key = features[features["key"] == "TR02-L"]
    assert ledger_key["payment_days_used"].tolist() == [0] * 5
    assert ledger_key[HABIT_COLUMNS[1:]].isna().all().all()


def test_payment_habits_with_the_ledger_a_ledger_only_key_has_its_habit():
    rows = panel(keys=("TR02-A", "TR02-L"))
    features = habits(mondays(), panel_rows=rows, ledger_table=ledger_lines(EARLY_LEDGER_ONLY),
                      first_file_day=date(2026, 2, 2), use_ledger=True,
                      leak_check_weeks=(ORIGIN,))
    assert per_h(features, "payment_days_used", key="TR02-L") == [4] * 5
    assert per_h(features, "median_gap_days", key="TR02-L") == [7.0] * 5


def test_payment_habits_without_the_ledger_the_daily_file_keys_are_unchanged():
    rows = panel(keys=("TR02-A", "TR02-L"))
    table = ledger_lines(EARLY_LEDGER_ONLY)
    without = habits(mondays(), panel_rows=rows, ledger_table=table,
                     first_file_day=date(2026, 2, 2), use_ledger=False)
    with_ledger = habits(mondays(), panel_rows=rows, ledger_table=table,
                         first_file_day=date(2026, 2, 2), use_ledger=True)
    a = without[without["key"] == "TR02-A"].reset_index(drop=True)
    b = with_ledger[with_ledger["key"] == "TR02-A"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_payment_habits_without_the_ledger_drop_the_early_days_of_a_mixed_key():
    # TR02-A paid by the ledger in January and in the files from 2 February.
    table = ledger_lines([payment(f"P{i}", -10.0, text) for i, text in enumerate(MONDAYS[:4])])
    files = mondays()[4:]
    without = habits(files, ledger_table=table, first_file_day=date(2026, 2, 2),
                     use_ledger=False)
    with_ledger = habits(files, ledger_table=table, first_file_day=date(2026, 2, 2),
                         use_ledger=True)
    assert per_h(without, "payment_days_used") == [5] * 5
    assert per_h(with_ledger, "payment_days_used") == [9] * 5


def test_payment_habits_without_the_ledger_is_the_same_as_with_an_empty_ledger():
    table = ledger_lines(EARLY_LEDGER_ONLY + [payment("PA", -5.0, "2025-12-15")])
    rows = panel(keys=("TR02-A", "TR02-L"), origins=(3, 8))
    without = habits(mondays(), panel_rows=rows, ledger_table=table,
                     first_file_day=date(2026, 2, 2), use_ledger=False)
    empty = habits(mondays(), panel_rows=rows, first_file_day=date(2026, 2, 2))
    pd.testing.assert_frame_equal(without, empty)


@pytest.mark.parametrize("use_ledger", [False, True])
def test_the_leak_check_passes_with_either_ledger_switch(use_ledger):
    table = ledger_lines(EARLY_LEDGER_ONLY + [payment("PA", -5.0, "2025-12-15"),
                                              payment("PB", -5.0, "2026-03-20")])
    habits(leaky_rows(), panel_rows=panel(keys=("TR02-A", "TR02-L"), origins=(0, 3, 6, 8, 9)),
           ledger_table=table, first_file_day=date(2026, 2, 2), use_ledger=use_ledger,
           leak_check_weeks=(0, 3, 6, 8, 9))


@pytest.mark.parametrize("use_ledger", [False, True])
def test_both_payment_days_calls_get_the_ledger_switch(monkeypatch, use_ledger):
    # The main call and the leak check's rebuild.
    real = rhythm.payment_days
    seen = []

    def recording(collections, ledger_table, first_file_day, week_zero, switch):
        seen.append(switch)
        return real(collections, ledger_table, first_file_day, week_zero, switch)

    monkeypatch.setattr(rhythm, "payment_days", recording)
    habits(mondays(), ledger_table=ledger_lines(EARLY_LEDGER_ONLY),
           first_file_day=date(2026, 2, 2), use_ledger=use_ledger, leak_check_weeks=(ORIGIN,))
    assert seen == [use_ledger, use_ledger]


def test_the_signatures_take_the_ledger_switch_without_a_default():
    import inspect
    assert list(inspect.signature(payment_habits).parameters) == [
        "panel_rows", "collections", "ledger", "week_zero", "holidays", "first_file_day",
        "use_ledger_before_first_file", "horizons", "payment_days_kept", "min_payment_days",
        "leak_check_weeks"]
    assert list(inspect.signature(payment_days).parameters) == [
        "collections", "ledger", "first_file_day", "week_zero", "use_ledger_before_first_file"]
    for function in (payment_habits, payment_days):
        parameter = inspect.signature(function).parameters["use_ledger_before_first_file"]
        assert parameter.default is inspect.Parameter.empty


def test_payment_days_without_the_switch_raise_type_error():
    with pytest.raises(TypeError):
        payment_days(collections_of([]), NO_LEDGER, FIRST_FILE_DAY, WEEK_ZERO)

# --- random data: the invariants hold, the leak check passes ----------------------------------

def random_rows(seed, keys=("TR02-A", "TR02-B", "TR04-C", "TR05-D")):
    rng = np.random.default_rng(seed)
    rows = []
    for key in keys:
        count = int(rng.integers(0, 30))
        offsets = rng.integers(-120, 120, size=count)
        for offset in offsets:
            amount = float(rng.choice([-50.0, 0.0, 10.0, 250.0]))
            text = (pd.Timestamp("2026-03-01") + pd.Timedelta(days=int(offset))).strftime(
                "%Y-%m-%d")
            rows.append((key, text, amount))
    return rows


@pytest.mark.parametrize("seed", range(8))
def test_random_payment_days_keep_the_invariants_and_pass_the_leak_check(seed):
    keys = ("TR02-A", "TR02-B", "TR04-C", "TR05-D", "TR06-E")
    rows = panel(keys=keys, origins=(2, 5, 8, 11, 14))
    features = habits(random_rows(seed), panel_rows=rows, holidays=holidays_on((3, 16), (4, 3)),
                      kept=10, minimum=3, leak_check_weeks=(2, 5, 8, 11, 14))
    assert len(features) == len(rows)
    assert (features["payment_days_used"].between(0, 10)).all()
    no_habit = features["payment_days_used"] < 3
    assert features.loc[no_habit, HABIT_COLUMNS[1:]].isna().all().all()
    with_habit = features[~no_habit]
    assert with_habit[HABIT_COLUMNS[1:]].notna().all().all()
    for column in ["month_start_share", "mid_month_share", "month_end_share",
                   "month_habit_in_target", "weekday_habit_in_target",
                   "next_payment_share_in_target"]:
        assert with_habit[column].between(0, 1).all(), column
    shares = (with_habit["month_start_share"] + with_habit["mid_month_share"]
              + with_habit["month_end_share"])
    assert (shares <= 1 + 1e-12).all()
    assert (with_habit["median_gap_days"] >= 1).all()
    assert (with_habit["gap_iqr_days"] >= 0).all()
    assert (with_habit["payment_due_ratio"] >= 0).all()
    # The next-payment shares over the horizons add up to at most 1 per key and origin.
    total = with_habit.groupby(["key", "origin_week"])["next_payment_share_in_target"].sum()
    assert (total <= 1 + 1e-9).all()


# --- integration with the synthetic ledger (SLOW) ---------------------------------------------
# The ledger's payments alone (no daily files: the first file day after the extract), so the
# corpus need not be read. Skipped when the synthetic ledger is not in data/raw.

from tests.hurdle.test_features_rhythm import (  # noqa: E402
    AS_OF,
    LEDGER_FILE,
    SYNTHETIC_WEEK_ZERO,
    synthetic,  # noqa: F401  (the module fixture)
)


def test_slow_synthetic_habits_from_the_ledger_pass_the_leak_check(synthetic):  # noqa: F811
    keyed, _rates, holidays = synthetic
    keys = keyed["key"].drop_duplicates()
    rows = pd.concat([pd.DataFrame({"key": keys.to_numpy(), "origin_week": origin, "h": h})
                      for origin in (30, 31) for h in range(1, 6)], ignore_index=True)
    no_files = collections_of([])
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        features = payment_habits(rows, no_files, keyed, SYNTHETIC_WEEK_ZERO, holidays,
                                  AS_OF + pd.Timedelta(days=1).to_pytimedelta(), True, 5, 24,
                                  3, (30, 31))
    assert len(features) == len(rows)
    assert list(features.columns) == ["key", "origin_week", "h"] + HABIT_COLUMNS
    assert features["median_gap_days"].notna().mean() > 0.3
    share = features["next_payment_share_in_target"].dropna()
    assert ((share >= 0) & (share <= 1)).all()
