"""Tests for `hurdle/inflation.py`: the TÜFE table, the coverage guard, and the three price
levels (`weekly_price_level`, `published_price_level`, `projected_price_level`).

`hurdle/inflation.py` is a deliberate copy of the older `collection_estimation/inflation.py`
with four TÜFE months added (2024-09 .. 2024-12). Only `test_copy_matches_old_module` imports
the old module, to pin that the copy behaves the same wherever both tables agree.

Every `weeks` frame is built by `make_weeks`: Monday starts, `week_end = week_start + 6 days`.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import inflation
from collection_estimation.hurdle.inflation import (
    PROJECTION_WINDOW,
    PUBLICATION_LAG_MONTHS,
    TUFE,
    InflationCoverageError,
    check_coverage,
    projected_price_level,
    published_price_level,
    weekly_price_level,
)

#: First Monday of the synthetic ledger history, and the last Monday of the synthetic range.
SYNTHETIC_FIRST_MONDAY = "2024-10-21"
SYNTHETIC_LAST_MONDAY = "2026-08-24"


def make_weeks(first_monday: str, last_monday: str) -> pd.DataFrame:
    starts = pd.date_range(first_monday, last_monday, freq="W-MON")
    assert len(starts) > 0 and starts[0] == pd.Timestamp(first_monday)
    return pd.DataFrame({"week_start": starts, "week_end": starts + pd.Timedelta(days=6)})


def next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


@pytest.fixture
def synthetic_weeks() -> pd.DataFrame:
    return make_weeks(SYNTHETIC_FIRST_MONDAY, SYNTHETIC_LAST_MONDAY)


# --------------------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------------------

def test_tufe_months_are_consecutive_from_2024_09_to_2026_08():
    keys = sorted(TUFE)
    assert keys[0] == (2024, 9)
    assert keys[-1] == (2026, 8)
    for a, b in zip(keys, keys[1:]):
        assert b == next_month(*a), f"gap between {a} and {b}"
    assert len(keys) == 24


def test_tufe_values_are_positive_and_strictly_increasing():
    levels = [TUFE[k] for k in sorted(TUFE)]
    assert all(v > 0 for v in levels)
    assert all(b > a for a, b in zip(levels, levels[1:]))


def test_tufe_four_new_months_are_exact():
    assert TUFE[(2024, 9)] == 2526.16
    assert TUFE[(2024, 10)] == 2598.91
    assert TUFE[(2024, 11)] == 2657.23
    assert TUFE[(2024, 12)] == 2684.55


@pytest.mark.parametrize("prev, month, announced_pct", [
    ((2024, 9), (2024, 10), 2.88),
    ((2024, 10), (2024, 11), 2.24),
    ((2024, 11), (2024, 12), 1.03),
    ((2024, 12), (2025, 1), 5.03),
])
def test_tufe_new_months_chain_at_tuik_announced_monthly_rates(prev, month, announced_pct):
    # September's 2.97% needs August 2024, which the table does not hold, so it cannot be
    # checked from the table alone; the four rates the table CAN imply are checked here.
    implied_pct = (TUFE[month] / TUFE[prev] - 1) * 100
    assert round(implied_pct, 2) == announced_pct


def test_tufe_is_keyed_by_year_month_int_tuples():
    for key in TUFE:
        assert isinstance(key, tuple) and len(key) == 2
        y, m = key
        assert isinstance(y, int) and isinstance(m, int) and 1 <= m <= 12


def test_module_constants():
    assert PUBLICATION_LAG_MONTHS == 1
    assert PROJECTION_WINDOW == 6


# --------------------------------------------------------------------------------------
# check_coverage
# --------------------------------------------------------------------------------------

def test_check_coverage_passes_for_full_table_range():
    check_coverage(date(2024, 9, 1), date(2026, 8, 31))


def test_check_coverage_passes_for_a_single_day():
    check_coverage(date(2025, 3, 10), date(2025, 3, 10))


def test_check_coverage_raises_for_2024_08_and_names_it():
    with pytest.raises(InflationCoverageError) as err:
        check_coverage(date(2024, 8, 15), date(2025, 1, 31))
    message = str(err.value)
    assert "2024-08" in message
    assert "2024-09" in message and "2026-08" in message  # the covered range is stated


def test_check_coverage_raises_for_2026_09_and_names_it():
    with pytest.raises(InflationCoverageError) as err:
        check_coverage(date(2026, 1, 1), date(2026, 9, 1))
    assert "2026-09" in str(err.value)


def test_check_coverage_names_every_missing_month_on_both_ends():
    with pytest.raises(InflationCoverageError) as err:
        check_coverage(date(2024, 7, 1), date(2026, 10, 1))
    message = str(err.value)
    for month in ("2024-07", "2024-08", "2026-09", "2026-10"):
        assert month in message
    assert "2025-01" not in message.split("The table covers")[0]


def test_inflation_coverage_error_is_a_value_error():
    assert issubclass(InflationCoverageError, ValueError)
    with pytest.raises(ValueError):
        check_coverage(date(2026, 9, 1), date(2026, 9, 2))


def test_synthetic_range_passes_coverage(synthetic_weeks):
    check_coverage(synthetic_weeks["week_start"].iloc[0].date(),
                   synthetic_weeks["week_end"].iloc[-1].date())
    # and every price level computes on it without raising
    weekly_price_level(synthetic_weeks)
    published_price_level(synthetic_weeks)
    projected_price_level(synthetic_weeks)


def test_price_levels_raise_when_last_week_ends_past_table():
    # Week starting 2026-08-31 ends 2026-09-06: September 2026 is not declared.
    weeks = make_weeks("2026-08-03", "2026-08-31")
    for fn in (weekly_price_level, published_price_level, projected_price_level):
        with pytest.raises(InflationCoverageError):
            fn(weeks)


def test_price_levels_raise_when_first_week_starts_before_table():
    weeks = make_weeks("2024-08-26", "2024-10-28")
    for fn in (weekly_price_level, published_price_level, projected_price_level):
        with pytest.raises(InflationCoverageError):
            fn(weeks)


# --------------------------------------------------------------------------------------
# weekly_price_level
# --------------------------------------------------------------------------------------

def test_weekly_price_level_is_one_at_last_week(synthetic_weeks):
    level = weekly_price_level(synthetic_weeks)
    assert level.shape == (len(synthetic_weeks),)
    assert level[-1] == 1.0


def test_weekly_price_level_is_increasing_over_synthetic_range(synthetic_weeks):
    level = weekly_price_level(synthetic_weeks)
    assert np.all(np.diff(level) >= 0)
    assert np.all(level > 0) and np.all(level <= 1.0)
    # Strictly increasing wherever the week's Thursday lies inside the table's mid-months.
    mid = synthetic_weeks["week_start"] + pd.Timedelta(days=3)
    inside = ((mid >= pd.Timestamp("2024-09-15")) & (mid <= pd.Timestamp("2026-08-15"))).to_numpy()
    assert np.all(np.diff(level[inside]) > 0)


def test_weekly_level_pins_known_silent_flat_clamp_past_last_mid_month():
    """PINS A KNOWN SILENT FALLBACK, not a desired behaviour.

    `np.interp` clamps outside its known points. The table's last point is 15 Aug 2026, so
    every week whose Thursday falls after it (still inside August, so coverage passes) gets
    the August level carried forward flat: half a month of ~2% inflation treated as zero.
    The same clamp applies before 15 Sep 2024 at the head of the table.
    """
    weeks = make_weeks("2026-08-03", "2026-08-24")
    raw = inflation._raw_weekly_level(weeks)
    # Thursdays: 6 Aug, 13 Aug (interpolated), 20 Aug, 27 Aug (clamped).
    assert raw[0] < raw[1] < raw[2]
    assert raw[2] == raw[3] == pytest.approx(TUFE[(2026, 8)], rel=1e-12)

    head = make_weeks("2024-09-02", "2024-09-23")
    raw_head = inflation._raw_weekly_level(head)
    # Thursdays 5 Sep and 12 Sep are before 15 Sep: both clamped to September's level.
    assert raw_head[0] == raw_head[1] == pytest.approx(TUFE[(2024, 9)], rel=1e-12)


def test_weekly_price_level_ratio_first_to_last_matches_table_roughly(synthetic_weeks):
    level = weekly_price_level(synthetic_weeks)
    # Late Oct 2024 to late Aug 2026: roughly TUFE[2026-08] / TUFE[2024-10].
    assert level[0] == pytest.approx(TUFE[(2024, 10)] / TUFE[(2026, 8)], rel=0.02)


@pytest.mark.parametrize("monday", ["2024-10-21", "2025-02-03", "2025-07-07", "2026-01-26"])
def test_weekly_level_lies_between_surrounding_mid_month_levels(monday):
    weeks = make_weeks(SYNTHETIC_FIRST_MONDAY, SYNTHETIC_LAST_MONDAY)
    raw = inflation._raw_weekly_level(weeks)
    i = int(np.flatnonzero(weeks["week_start"] == pd.Timestamp(monday))[0])
    mid = pd.Timestamp(monday) + pd.Timedelta(days=3)
    if mid.day >= 15:
        lo_key, hi_key = (mid.year, mid.month), next_month(mid.year, mid.month)
    else:
        hi_key = (mid.year, mid.month)
        lo_key = (mid.year - 1, 12) if mid.month == 1 else (mid.year, mid.month - 1)
    assert TUFE[lo_key] <= raw[i] <= TUFE[hi_key]


def test_weekly_level_is_log_linear_interpolation_between_mid_months():
    weeks = make_weeks(SYNTHETIC_FIRST_MONDAY, SYNTHETIC_LAST_MONDAY)
    raw = inflation._raw_weekly_level(weeks)
    # Week starting 2025-07-21: midpoint Thursday 2025-07-24, between 15 Jul and 15 Aug.
    i = int(np.flatnonzero(weeks["week_start"] == pd.Timestamp("2025-07-21"))[0])
    t0, t1 = pd.Timestamp("2025-07-15"), pd.Timestamp("2025-08-15")
    t = pd.Timestamp("2025-07-24")
    frac = (t - t0).days / (t1 - t0).days
    expected = np.exp(np.log(TUFE[(2025, 7)])
                      + frac * (np.log(TUFE[(2025, 8)]) - np.log(TUFE[(2025, 7)])))
    assert raw[i] == pytest.approx(expected, rel=1e-12)
    linear = TUFE[(2025, 7)] + frac * (TUFE[(2025, 8)] - TUFE[(2025, 7)])
    assert raw[i] < linear  # log interpolation sits below the chord


def test_weekly_level_at_mid_month_thursday_equals_table_value():
    # Week starting Monday 2025-01-13: midpoint Thursday 2025-01-16 is not the 15th;
    # find a week whose Thursday IS the 15th: Monday 2025-05-12 -> Thursday 2025-05-15.
    weeks = make_weeks("2025-05-12", "2025-06-02")
    raw = inflation._raw_weekly_level(weeks)
    assert raw[0] == pytest.approx(TUFE[(2025, 5)], rel=1e-12)


def test_weekly_price_level_is_deterministic_and_does_not_mutate_input(synthetic_weeks):
    before = synthetic_weeks.copy()
    a = weekly_price_level(synthetic_weeks)
    b = weekly_price_level(synthetic_weeks)
    np.testing.assert_array_equal(a, b)
    pd.testing.assert_frame_equal(synthetic_weeks, before)


# --------------------------------------------------------------------------------------
# published_price_level
# --------------------------------------------------------------------------------------

def test_published_price_level_is_constant_within_a_calendar_month(synthetic_weeks):
    level = published_price_level(synthetic_weeks)
    month = synthetic_weeks["week_start"].dt.to_period("M")
    for _, idx in synthetic_weeks.groupby(month).groups.items():
        values = level[synthetic_weeks.index.get_indexer(idx)]
        assert np.all(values == values[0])


def test_published_price_level_steps_up_at_each_month_boundary(synthetic_weeks):
    level = published_price_level(synthetic_weeks)
    month = synthetic_weeks["week_start"].dt.to_period("M").to_numpy()
    change = month[1:] != month[:-1]
    assert np.all(np.diff(level)[change] > 0)
    assert np.all(np.diff(level)[~change] == 0)


def test_published_price_level_uses_previous_month_index(synthetic_weeks):
    level = published_price_level(synthetic_weeks)
    raw_last = inflation._raw_weekly_level(synthetic_weeks)[-1]
    i = int(np.flatnonzero(synthetic_weeks["week_start"] == pd.Timestamp("2025-07-07"))[0])
    assert level[i] == TUFE[(2025, 6)] / raw_last


def test_published_price_level_lag_wraps_across_year_boundary(synthetic_weeks):
    level = published_price_level(synthetic_weeks)
    raw_last = inflation._raw_weekly_level(synthetic_weeks)[-1]
    i = int(np.flatnonzero(synthetic_weeks["week_start"] == pd.Timestamp("2026-01-05"))[0])
    assert level[i] == TUFE[(2025, 12)] / raw_last


def test_published_price_level_never_exceeds_weekly_level_in_synthetic_range(synthetic_weeks):
    # A lagged index is always older than the week, so with rising prices it sits below.
    assert np.all(published_price_level(synthetic_weeks)
                  < weekly_price_level(synthetic_weeks))


def test_published_price_level_same_normalisation_as_weekly(synthetic_weeks):
    raw = inflation._raw_weekly_level(synthetic_weeks)
    level = published_price_level(synthetic_weeks)
    i = len(synthetic_weeks) - 1  # last week starts 2026-08-24 -> July 2026 published
    assert level[i] == TUFE[(2026, 7)] / raw[-1]


def test_published_price_level_custom_lag_two_months(synthetic_weeks):
    level = published_price_level(synthetic_weeks, lag_months=2)
    raw_last = inflation._raw_weekly_level(synthetic_weeks)[-1]
    i = int(np.flatnonzero(synthetic_weeks["week_start"] == pd.Timestamp("2025-07-07"))[0])
    assert level[i] == TUFE[(2025, 5)] / raw_last


def test_published_price_level_pins_known_silent_fallback_to_earliest_month():
    """PINS A KNOWN SILENT FALLBACK, not a desired behaviour.

    A week starting in 2024-09 has 2024-08 as its published month, which the table lacks.
    The code falls back to the table's earliest month, 2024-09 — i.e. it uses a figure that
    was NOT yet published in that week (a one-month look-ahead), without warning.
    """
    weeks = make_weeks("2024-09-02", "2024-10-28")
    level = published_price_level(weeks)
    raw_last = inflation._raw_weekly_level(weeks)[-1]
    september = weeks["week_start"].dt.month.to_numpy() == 9
    assert september.any()
    assert np.all(level[september] == TUFE[(2024, 9)] / raw_last)
    # October weeks correctly use September: so Sep and Oct weeks share one value.
    assert np.all(level[~september] == TUFE[(2024, 9)] / raw_last)


# --------------------------------------------------------------------------------------
# projected_price_level
# --------------------------------------------------------------------------------------

def test_projected_price_level_is_square(synthetic_weeks):
    P = projected_price_level(synthetic_weeks)
    n = len(synthetic_weeks)
    assert P.shape == (n, n)
    assert np.all(np.isfinite(P)) and np.all(P > 0)


def week_index(weeks: pd.DataFrame, monday: str) -> int:
    return int(np.flatnonzero(weeks["week_start"] == pd.Timestamp(monday))[0])


@pytest.mark.parametrize("origin_monday", ["2025-03-10", "2025-07-07", "2026-01-12"])
def test_projected_row_ignores_tufe_months_after_published_month(
        monkeypatch, synthetic_weeks, origin_monday):
    o = week_index(synthetic_weeks, origin_monday)
    start = pd.Timestamp(origin_monday)
    published = (start.year - 1, 12) if start.month == 1 else (start.year, start.month - 1)

    # Rows are normalised by the LAST week's raw level (which reads the whole table), so
    # compare un-normalised rows: P * raw_last. That constant cancels in every use.
    def unnormalised_row():
        raw_last = inflation._raw_weekly_level(synthetic_weeks)[-1]
        return projected_price_level(synthetic_weeks)[o] * raw_last

    before = unnormalised_row()
    for key in sorted(TUFE):
        if key > published:
            monkeypatch.setitem(TUFE, key, TUFE[key] * 1.25)
    after = unnormalised_row()
    np.testing.assert_allclose(after, before, rtol=1e-12)


def test_projected_row_does_change_when_published_month_changes(monkeypatch, synthetic_weeks):
    # Control for the test above: the row is sensitive to months that WERE published.
    o = week_index(synthetic_weeks, "2025-07-07")
    raw_last = inflation._raw_weekly_level(synthetic_weeks)[-1]
    before = projected_price_level(synthetic_weeks)[o] * raw_last
    monkeypatch.setitem(TUFE, (2025, 6), TUFE[(2025, 6)] * 1.01)
    raw_last = inflation._raw_weekly_level(synthetic_weeks)[-1]
    after = projected_price_level(synthetic_weeks)[o] * raw_last
    assert not np.allclose(after, before)


def test_projected_row_is_anchored_at_published_month_midpoint(synthetic_weeks):
    # The row evaluated at the anchor month's midpoint equals the anchor level; check by
    # reconstructing the row from its formula.
    o = week_index(synthetic_weeks, "2025-07-07")
    raw_last = inflation._raw_weekly_level(synthetic_weeks)[-1]
    keys = [k for k in sorted(TUFE) if k <= (2025, 6)]
    recent = keys[-(PROJECTION_WINDOW + 1):]
    growth = (np.log(TUFE[recent[-1]]) - np.log(TUFE[recent[0]])) / (len(recent) - 1)
    anchor_ord = pd.Timestamp("2025-06-15").toordinal()
    target_ord = (synthetic_weeks["week_start"] + pd.Timedelta(days=3)).map(
        pd.Timestamp.toordinal).to_numpy(dtype=float)
    expected = TUFE[(2025, 6)] / raw_last * np.exp(growth * (target_ord - anchor_ord) / 30.437)
    np.testing.assert_allclose(projected_price_level(synthetic_weeks)[o], expected, rtol=1e-12)


def test_projected_rows_increase_along_targets_once_two_months_published(synthetic_weeks):
    P = projected_price_level(synthetic_weeks)
    for o, start in enumerate(synthetic_weeks["week_start"]):
        if start >= pd.Timestamp("2024-11-01"):
            assert np.all(np.diff(P[o]) > 0), start


def test_projected_one_week_ahead_is_close_to_actual_level(synthetic_weeks):
    """No -2.46% undershoot: the projection compounds from the published month."""
    P = projected_price_level(synthetic_weeks)
    actual = weekly_price_level(synthetic_weeks)
    n = len(synthetic_weeks)
    err = np.array([P[o, o + 1] / actual[o + 1] - 1 for o in range(n - 1)])
    assert np.all(np.abs(err) < 0.05)
    # Once at least two months are published (origins from November 2024).
    from_nov = (synthetic_weeks["week_start"].iloc[:-1] >= pd.Timestamp("2024-11-01")).to_numpy()
    assert np.all(np.abs(err[from_nov]) < 0.035)
    # And no systematic bias of the old, broken kind.
    assert abs(err[from_nov].mean()) < 0.01


def test_projected_diagonal_is_close_to_actual_level(synthetic_weeks):
    P = projected_price_level(synthetic_weeks)
    actual = weekly_price_level(synthetic_weeks)
    from_nov = (synthetic_weeks["week_start"] >= pd.Timestamp("2024-11-01")).to_numpy()
    err = np.diag(P)[from_nov] / actual[from_nov] - 1
    assert np.all(np.abs(err) < 0.035)
    assert abs(err.mean()) < 0.01


def test_projected_is_flat_when_only_one_month_published(synthetic_weeks):
    # Origins in October 2024 have only 2024-09 published: too little to estimate a rate.
    P = projected_price_level(synthetic_weeks)
    raw_last = inflation._raw_weekly_level(synthetic_weeks)[-1]
    october = (synthetic_weeks["week_start"].dt.month == 10) & (
        synthetic_weeks["week_start"].dt.year == 2024)
    for o in np.flatnonzero(october.to_numpy()):
        assert np.all(P[o] == TUFE[(2024, 9)] / raw_last)


def test_projected_pins_known_silent_fallback_when_nothing_published():
    """PINS A KNOWN SILENT FALLBACK, not a desired behaviour.

    An origin in 2024-09 has 2024-08 as its published month; nothing in the table is
    published by then, so the anchor falls back to 2024-09 (unpublished) and the row is flat.
    """
    weeks = make_weeks("2024-09-02", "2024-11-25")
    P = projected_price_level(weeks)
    raw_last = inflation._raw_weekly_level(weeks)[-1]
    assert np.all(P[0] == TUFE[(2024, 9)] / raw_last)


def test_projected_uses_window_plus_one_months(synthetic_weeks):
    # window=1 uses only the last month-on-month rate.
    o = week_index(synthetic_weeks, "2025-07-07")
    P = projected_price_level(synthetic_weeks, window=1)
    growth = np.log(TUFE[(2025, 6)]) - np.log(TUFE[(2025, 5)])
    t = o + 4
    raw_last = inflation._raw_weekly_level(synthetic_weeks)[-1]
    elapsed = ((synthetic_weeks["week_start"].iloc[t] + pd.Timedelta(days=3)).toordinal()
               - pd.Timestamp("2025-06-15").toordinal()) / 30.437
    assert P[o, t] == pytest.approx(TUFE[(2025, 6)] / raw_last * np.exp(growth * elapsed),
                                    rel=1e-12)


# --------------------------------------------------------------------------------------
# The copy against the old module
# --------------------------------------------------------------------------------------

def test_copy_matches_old_module():
    """The copy behaves as the old module wherever the old table had the data.

    Imports the OLD module deliberately; no other test does.
    """
    from collection_estimation import inflation as old

    # Old table is a strict subset of the new one, with equal values.
    assert set(old.TUFE) < set(TUFE)
    assert {k: TUFE[k] for k in old.TUFE} == old.TUFE
    assert set(TUFE) - set(old.TUFE) == {(2024, 9), (2024, 10), (2024, 11), (2024, 12)}
    assert old.PUBLICATION_LAG_MONTHS == PUBLICATION_LAG_MONTHS
    assert old.PROJECTION_WINDOW == PROJECTION_WINDOW

    weeks = make_weeks("2025-06-02", "2026-08-24")
    np.testing.assert_array_equal(old.weekly_price_level(weeks), weekly_price_level(weeks))
    np.testing.assert_array_equal(old.published_price_level(weeks),
                                  published_price_level(weeks))

    # projected: equal for origins whose 7-month window (window + 1) fits inside the old
    # table, i.e. published month >= 2025-07, origins from August 2025.
    P_old = old.projected_price_level(weeks)
    P_new = projected_price_level(weeks)
    full = (weeks["week_start"] >= pd.Timestamp("2025-08-01")).to_numpy()
    np.testing.assert_array_equal(P_old[full], P_new[full])

    # Origins in June-July 2025 legitimately DIFFER: the old table had fewer than seven
    # published months there, so its rate used a shorter window; the new table fills it.
    early = ~full
    assert early.any()
    for o in np.flatnonzero(early):
        assert not np.allclose(P_old[o], P_new[o], rtol=0, atol=1e-12)

    # Same error type contract.
    with pytest.raises(old.InflationCoverageError):
        old.check_coverage(date(2024, 12, 1), date(2024, 12, 31))
    check_coverage(date(2024, 12, 1), date(2024, 12, 31))
