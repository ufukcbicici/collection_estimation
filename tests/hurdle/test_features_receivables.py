"""Tests for U9c.2, `hurdle/features_receivables.py`: `receivables_features`,
`check_receivables_for_features`, `before_first_ledger_week`, `check_no_missing_values`,
`check_due_split`, `SPLIT_TOLERANCE` and `NON_AMOUNT_COLUMNS`.

A panel row before its key's first week in the receivables table (a key known from a daily
file before the ledger shows it) is filled as "nothing owed": 0 everywhere, usual_delay_days
NaN. A missing row from the key's first week on (a gap) still raises, and so does a panel
key with no receivables row at all (the two tables' keys would not match).

`receivables_features(panel_rows, receivables, levels, horizons, age_edges)` gives one row per
panel row (key, origin_week, h), in the same order, from the receivables row of its key and
origin week t: what is due before / in / after its target week t+h, the aging buckets,
overdue_days, the counts, unapplied cash, the open balance and the usual delay. Amounts are
divided by the price level of week t; days and counts are not.

Every receivables table here is built by hand with the columns `receivables_columns` gives,
so these tests do not depend on the ledger. Nothing here reads or writes data/interim.
"""
import random

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle import features_receivables
from collection_estimation.hurdle.features_receivables import (
    NON_AMOUNT_COLUMNS,
    SPLIT_TOLERANCE,
    before_first_ledger_week,
    check_due_split,
    check_no_missing_values,
    check_receivables_for_features,
    receivables_features,
)
from collection_estimation.hurdle.receivables_asof import (
    due_in_week_columns,
    overdue_age_columns,
    receivables_columns,
)

HORIZONS = 5
AGE_EDGES = (30, 90)
AGES = ["overdue_up_to_30", "overdue_31_to_90", "overdue_over_90"]
DUE = due_in_week_columns(HORIZONS)

FEATURE_COLUMNS = (["key", "origin_week", "h", "due_before_target_week", "due_in_target_week",
                    "due_after_target_week"] + AGES
                   + ["overdue_days", "open_invoices", "cleared_invoices", "unapplied_cash",
                      "open_balance", "usual_delay_days"])
AMOUNT_FEATURES = (["due_before_target_week", "due_in_target_week", "due_after_target_week"]
                   + AGES + ["unapplied_cash", "open_balance"])
INT_FEATURES = ["origin_week", "h", "open_invoices", "cleared_invoices"]


# --- helpers ------------------------------------------------------------------------------

def receivables_row(key, week, due=(0, 0, 0, 0, 0), due_later=0.0, overdue=0.0, ages=None,
                    overdue_days=0.0, open_invoices=0, unapplied_cash=0.0, open_balance=0.0,
                    usual_delay_days=float("nan"), cleared_invoices=0, age_edges=AGE_EDGES):
    """One receivables row as a dict; `ages` defaults to all of `overdue` in the first bucket."""
    names = overdue_age_columns(age_edges)
    ages = ages if ages is not None else [overdue] + [0.0] * (len(names) - 1)
    row = {"key": key, "week": week}
    row.update({f"due_in_week_{h}": float(v) for h, v in enumerate(due, start=1)})
    row.update({"due_later": float(due_later), "overdue": float(overdue)})
    row.update({name: float(v) for name, v in zip(names, ages)})
    row.update({"overdue_days": float(overdue_days), "open_invoices": open_invoices,
                "unapplied_cash": float(unapplied_cash), "open_balance": float(open_balance),
                "usual_delay_days": float(usual_delay_days),
                "cleared_invoices": cleared_invoices})
    return row


def receivables_table(rows, horizons=HORIZONS, age_edges=AGE_EDGES):
    """A receivables table with the columns and dtypes `receivables_by_week` gives."""
    columns = receivables_columns(horizons, age_edges)
    table = pd.DataFrame(rows, columns=columns)
    table["key"] = table["key"].astype(object)
    for column in ("week", "open_invoices", "cleared_invoices"):
        table[column] = table[column].astype("int64")
    for column in columns:
        if column not in ("key", "week", "open_invoices", "cleared_invoices"):
            table[column] = table[column].astype("float64")
    return table


def panel(triples, index=None):
    """Panel rows from (key, origin_week, h) triples, plus a label column it should ignore."""
    table = pd.DataFrame({
        "key": pd.Series([t[0] for t in triples], dtype=object),
        "origin_week": pd.Series([t[1] for t in triples], dtype="int64"),
        "h": pd.Series([t[2] for t in triples], dtype="int64"),
    })
    table["target_week"] = table["origin_week"] + table["h"]
    table["paid"] = 1.0
    if index is not None:
        table.index = index
    return table


def levels(mapping):
    return pd.Series(mapping, dtype="float64", name="price_level")


def example_row(**changes):
    """The worked example: key K at t=10, due 100 / 40 in weeks 1 / 2, 500 later, overdue 300
    (200 up to 30 days, 100 over 90 days)."""
    values = dict(due=(100, 40, 0, 0, 0), due_later=500, overdue=300, ages=[200, 0, 100],
                  overdue_days=55.0, open_invoices=4, unapplied_cash=25.0,
                  open_balance=915.0, usual_delay_days=12.5, cleared_invoices=6)
    values.update(changes)
    return receivables_row("K", 10, **values)


def features(triples, rows, level_map=None, horizons=HORIZONS, age_edges=AGE_EDGES):
    weeks = {t[1] for t in triples}
    level_map = level_map if level_map is not None else {week: 1.0 for week in weeks}
    return receivables_features(panel(triples), receivables_table(rows, horizons, age_edges),
                                levels(level_map), horizons, age_edges)


def one(triples, rows, **kwargs):
    table = features(triples, rows, **kwargs)
    assert len(table) == 1
    return table.iloc[0]


def reference(panel_rows, receivables, level_map, horizons, age_edges):
    """A row-by-row brute-force version of `receivables_features`."""
    by_key_week = {(r["key"], r["week"]): r for r in receivables.to_dict("records")}
    out = []
    for key, t, h in panel_rows[["key", "origin_week", "h"]].itertuples(index=False):
        r = by_key_week[(key, t)]
        level = level_map[t]
        due = [r[f"due_in_week_{k}"] for k in range(1, horizons + 1)]
        row = {"key": key, "origin_week": t, "h": h,
               "due_before_target_week": sum(due[:h - 1]) / level,
               "due_in_target_week": due[h - 1] / level,
               "due_after_target_week": (sum(due[h:]) + r["due_later"]) / level}
        for name in overdue_age_columns(age_edges):
            row[name] = r[name] / level
        row.update({"overdue_days": r["overdue_days"], "open_invoices": r["open_invoices"],
                    "cleared_invoices": r["cleared_invoices"],
                    "unapplied_cash": r["unapplied_cash"] / level,
                    "open_balance": r["open_balance"] / level,
                    "usual_delay_days": r["usual_delay_days"]})
        out.append(row)
    return pd.DataFrame(out)


# --- constants ----------------------------------------------------------------------------

def test_split_tolerance_is_one_kurus():
    assert SPLIT_TOLERANCE == 0.01


def test_non_amount_columns_are_the_days_and_counts():
    assert sorted(NON_AMOUNT_COLUMNS) == sorted(["overdue_days", "open_invoices",
                                                 "cleared_invoices", "usual_delay_days"])


# --- the worked example -------------------------------------------------------------------

def test_worked_example_h_2():
    row = one([("K", 10, 2)], [example_row()])
    assert row["due_before_target_week"] == 100.0
    assert row["due_in_target_week"] == 40.0
    assert row["due_after_target_week"] == 500.0
    assert (row["overdue_up_to_30"], row["overdue_31_to_90"], row["overdue_over_90"]) == (
        200.0, 0.0, 100.0)


def test_worked_example_h_1():
    row = one([("K", 10, 1)], [example_row()])
    assert row["due_before_target_week"] == 0.0
    assert row["due_in_target_week"] == 100.0
    assert row["due_after_target_week"] == 540.0


def test_worked_example_h_5():
    row = one([("K", 10, 5)], [example_row()])
    assert row["due_before_target_week"] == 140.0
    assert row["due_in_target_week"] == 0.0
    assert row["due_after_target_week"] == 500.0


def test_worked_example_carries_the_other_columns_unchanged_at_level_one():
    row = one([("K", 10, 3)], [example_row()])
    assert row["key"] == "K" and row["origin_week"] == 10 and row["h"] == 3
    assert row["overdue_days"] == 55.0
    assert row["open_invoices"] == 4
    assert row["cleared_invoices"] == 6
    assert row["unapplied_cash"] == 25.0
    assert row["open_balance"] == 915.0
    assert row["usual_delay_days"] == 12.5


def test_worked_example_at_level_two_halves_amounts_not_days_or_counts():
    row = one([("K", 10, 2)], [example_row()], level_map={10: 2.0})
    assert row["due_before_target_week"] == 50.0
    assert row["due_in_target_week"] == 20.0
    assert row["due_after_target_week"] == 250.0
    assert (row["overdue_up_to_30"], row["overdue_31_to_90"], row["overdue_over_90"]) == (
        100.0, 0.0, 50.0)
    assert row["unapplied_cash"] == 12.5
    assert row["open_balance"] == 457.5
    assert row["overdue_days"] == 55.0
    assert row["open_invoices"] == 4
    assert row["cleared_invoices"] == 6
    assert row["usual_delay_days"] == 12.5


@pytest.mark.parametrize("h", range(1, HORIZONS + 1))
def test_worked_example_identity_before_in_after_plus_overdue_is_the_total(h):
    row = one([("K", 10, h)], [example_row()])
    total = 100 + 40 + 500 + 300
    assert (row["due_before_target_week"] + row["due_in_target_week"]
            + row["due_after_target_week"] + 300) == total


# --- every h for one receivables row ------------------------------------------------------

DISTINCT = (1.0, 2.0, 4.0, 8.0, 16.0)   # powers of two: every sum is unique


@pytest.mark.parametrize("h", range(1, HORIZONS + 1))
def test_every_h_picks_its_own_target_week(h):
    row = one([("K", 10, h)], [example_row(due=DISTINCT, due_later=32.0)])
    assert row["due_before_target_week"] == sum(DISTINCT[:h - 1])
    assert row["due_in_target_week"] == DISTINCT[h - 1]
    assert row["due_after_target_week"] == sum(DISTINCT[h:]) + 32.0


def test_all_h_of_one_receivables_row_in_one_call():
    table = features([("K", 10, h) for h in range(1, 6)],
                     [example_row(due=DISTINCT, due_later=32.0)])
    assert table["due_before_target_week"].tolist() == [0.0, 1.0, 3.0, 7.0, 15.0]
    assert table["due_in_target_week"].tolist() == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert table["due_after_target_week"].tolist() == [62.0, 60.0, 56.0, 48.0, 32.0]
    # The columns that do not depend on h are the same in every row.
    for column in AGES + ["overdue_days", "open_invoices", "cleared_invoices",
                          "unapplied_cash", "open_balance", "usual_delay_days"]:
        assert table[column].nunique() == 1, column


def test_horizons_of_one():
    row = one([("K", 10, 1)], [receivables_row("K", 10, due=(7,), due_later=3, overdue=1)],
              horizons=1)
    assert row["due_before_target_week"] == 0.0
    assert row["due_in_target_week"] == 7.0
    assert row["due_after_target_week"] == 3.0


@pytest.mark.parametrize("h", [1, 2, 3])
def test_horizons_of_three(h):
    due = (1.0, 10.0, 100.0)
    row = one([("K", 4, h)], [receivables_row("K", 4, due=due, due_later=1000)], horizons=3)
    assert row["due_before_target_week"] == sum(due[:h - 1])
    assert row["due_in_target_week"] == due[h - 1]
    assert row["due_after_target_week"] == sum(due[h:]) + 1000


def test_a_key_with_nothing_open_gets_zero_features():
    row = one([("Z", 3, 2)], [receivables_row("Z", 3, unapplied_cash=80, open_balance=-80)])
    for column in ["due_before_target_week", "due_in_target_week", "due_after_target_week"] + AGES:
        assert row[column] == 0.0, column
    assert row["unapplied_cash"] == 80.0
    assert row["open_balance"] == -80.0


# --- the identity on random tables, against a brute-force reference -----------------------

def random_receivables(rng, keys, weeks, horizons=HORIZONS, age_edges=AGE_EDGES):
    rows = []
    for key in keys:
        for week in weeks:
            ages = [round(rng.uniform(0, 5000), 2) for _ in range(len(age_edges) + 1)]
            rows.append(receivables_row(
                key, week, due=[round(rng.uniform(0, 1e6), 2) for _ in range(horizons)],
                due_later=round(rng.uniform(0, 1e7), 2), overdue=sum(ages), ages=ages,
                overdue_days=rng.choice([0.0, rng.uniform(0, 400)]),
                open_invoices=rng.randint(0, 50), unapplied_cash=round(rng.uniform(0, 900), 2),
                open_balance=round(rng.uniform(-1e3, 1e7), 2),
                usual_delay_days=rng.choice([float("nan"), rng.uniform(-10, 90)]),
                cleared_invoices=rng.randint(0, 30), age_edges=age_edges))
    return rows


@pytest.mark.parametrize("seed", range(6))
def test_random_tables_match_the_reference_and_keep_the_identity(seed):
    rng = random.Random(seed)
    horizons = rng.choice([1, 3, 5, 8])
    age_edges = rng.choice([(30,), (30, 90), (30, 60, 90)])
    keys = [f"TR0{rng.randint(1, 9)}-{n}" for n in range(rng.randint(1, 6))]
    weeks = sorted(rng.sample(range(0, 40), rng.randint(1, 5)))
    rows = random_receivables(rng, keys, weeks, horizons, age_edges)
    triples = [(k, w, h) for k in keys for w in weeks for h in range(1, horizons + 1)]
    rng.shuffle(triples)
    level_map = {w: rng.uniform(0.5, 3.0) for w in range(0, 40)}
    receivables = receivables_table(rows, horizons, age_edges)
    panel_rows = panel(triples)
    result = receivables_features(panel_rows, receivables, levels(level_map), horizons,
                                  age_edges)
    expected = reference(panel_rows, receivables, level_map, horizons, age_edges)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False, rtol=1e-12)
    # The identity, in TRY of week t: before + in + after + overdue = the open total.
    by_key_week = receivables.set_index(["key", "week"])
    for row in result.itertuples(index=False):
        r = by_key_week.loc[(row.key, row.origin_week)]
        total = r[due_in_week_columns(horizons)].sum() + r["due_later"] + r["overdue"]
        level = level_map[row.origin_week]
        split = (row.due_before_target_week + row.due_in_target_week
                 + row.due_after_target_week) * level + r["overdue"]
        assert split == pytest.approx(total, rel=1e-12)


# --- order and index ----------------------------------------------------------------------

def three_keys():
    return [receivables_row(k, w, due=(n + 1, 0, 0, 0, 0), open_invoices=n)
            for n, (k, w) in enumerate([("A", 1), ("A", 2), ("B", 1), ("B", 2), ("C", 2)])]


def test_rows_follow_a_shuffled_panel_and_get_a_fresh_index():
    triples = [(k, w, h) for k, w in [("A", 1), ("A", 2), ("B", 1), ("B", 2), ("C", 2)]
               for h in range(1, 6)]
    random.Random(3).shuffle(triples)
    index = pd.Index([1000 - 7 * n for n in range(len(triples))])
    panel_rows = panel(triples, index=index)
    result = receivables_features(panel_rows, receivables_table(three_keys()),
                                  levels({1: 1.0, 2: 1.0}), HORIZONS, AGE_EDGES)
    assert list(zip(result["key"], result["origin_week"], result["h"])) == triples
    assert isinstance(result.index, pd.RangeIndex)
    assert result.index.tolist() == list(range(len(triples)))


def test_receivables_order_does_not_matter():
    triples = [("C", 2, 1), ("A", 1, 3), ("B", 2, 5)]
    rows = three_keys()
    forward = features(triples, rows)
    backward = features(triples, rows[::-1])
    pd.testing.assert_frame_equal(forward, backward)


def test_a_panel_with_a_string_index_keeps_its_order():
    triples = [("B", 1, 2), ("A", 2, 1)]
    panel_rows = panel(triples, index=pd.Index(["x", "y"]))
    result = receivables_features(panel_rows, receivables_table(three_keys()),
                                  levels({1: 1.0, 2: 1.0}), HORIZONS, AGE_EDGES)
    assert result["key"].tolist() == ["B", "A"]
    assert result.index.tolist() == [0, 1]


def test_columns_are_exactly_the_feature_columns():
    assert list(features([("K", 10, 1)], [example_row()]).columns) == FEATURE_COLUMNS


def test_columns_follow_the_age_edges():
    rows = [receivables_row("K", 1, overdue=5, age_edges=(30,))]
    table = features([("K", 1, 1)], rows, age_edges=(30,))
    assert list(table.columns) == (FEATURE_COLUMNS[:6] + ["overdue_up_to_30", "overdue_over_30"]
                                   + FEATURE_COLUMNS[9:])
    rows = [receivables_row("K", 1, overdue=5, age_edges=(30, 60, 90))]
    table = features([("K", 1, 1)], rows, age_edges=(30, 60, 90))
    assert list(table.columns[6:10]) == overdue_age_columns((30, 60, 90))


def test_only_the_three_panel_columns_are_kept():
    table = features([("K", 10, 1)], [example_row()])
    assert "target_week" not in table.columns
    assert "paid" not in table.columns
    assert "week" not in table.columns
    assert "overdue" not in table.columns
    assert not any(column.startswith("due_in_week_") for column in table.columns)
    assert "due_later" not in table.columns


def test_an_empty_panel_gives_an_empty_table_with_the_columns():
    result = receivables_features(panel([]), receivables_table([example_row()]),
                                  levels({10: 1.0}), HORIZONS, AGE_EDGES)
    assert result.empty
    assert list(result.columns) == FEATURE_COLUMNS


# --- deflation ----------------------------------------------------------------------------

def test_each_origin_week_is_divided_by_its_own_level():
    rows = [example_row(), receivables_row("K", 11, due=(100, 40, 0, 0, 0), due_later=500,
                                           overdue=300, ages=[200, 0, 100], overdue_days=55.0,
                                           open_invoices=4, unapplied_cash=25.0,
                                           open_balance=915.0, usual_delay_days=12.5,
                                           cleared_invoices=6)]
    table = features([("K", 10, 2), ("K", 11, 2)], rows, level_map={10: 1.0, 11: 4.0})
    first, second = table.iloc[0], table.iloc[1]
    for column in AMOUNT_FEATURES:
        assert second[column] == pytest.approx(first[column] / 4.0), column
    for column in NON_AMOUNT_COLUMNS:
        assert second[column] == first[column], column


def test_the_level_of_the_origin_week_is_used_not_the_target_week():
    table = features([("K", 10, 3)], [example_row()],
                     level_map={10: 2.0, 13: 100.0, 11: 50.0})
    assert table.iloc[0]["due_in_target_week"] == 0.0
    assert table.iloc[0]["due_before_target_week"] == 70.0   # (100 + 40) / 2


def test_levels_for_weeks_not_in_the_panel_are_ignored():
    table = features([("K", 10, 1)], [example_row()],
                     level_map={w: (1.0 if w == 10 else float("nan")) for w in range(20)})
    assert table.iloc[0]["due_in_target_week"] == 100.0


def test_a_level_below_one_raises_amounts():
    row = one([("K", 10, 1)], [example_row()], level_map={10: 0.5})
    assert row["due_in_target_week"] == 200.0
    assert row["open_balance"] == 1830.0


# --- NaN usual delay ----------------------------------------------------------------------

def test_a_nan_usual_delay_stays_nan_and_others_are_not_nan():
    rows = [receivables_row("A", 1, usual_delay_days=float("nan")),
            receivables_row("B", 1, usual_delay_days=7.0)]
    table = features([("A", 1, 1), ("B", 1, 1)], rows, level_map={1: 3.0})
    assert np.isnan(table.iloc[0]["usual_delay_days"])
    assert table.iloc[1]["usual_delay_days"] == 7.0   # not deflated
    amounts = table[AMOUNT_FEATURES + ["overdue_days", "open_invoices", "cleared_invoices"]]
    assert amounts.notna().all().all()


# --- no leak ------------------------------------------------------------------------------

def test_features_at_t_do_not_change_when_other_weeks_change():
    rng = random.Random(11)
    rows = random_receivables(rng, ["A", "B"], [8, 9, 10, 11, 12])
    triples = [(k, 10, h) for k in ["A", "B"] for h in range(1, 6)]
    level_map = {w: 1.0 + w / 10 for w in range(8, 13)}
    base = features(triples, rows, level_map=level_map)
    # Later weeks (and an earlier one) replaced with very different numbers.
    changed = [r if r["week"] == 10 else
               random_receivables(random.Random(r["week"]), [r["key"]], [r["week"]])[0]
               for r in rows]
    assert changed != rows
    pd.testing.assert_frame_equal(features(triples, changed, level_map=level_map), base)


def test_features_at_t_do_not_change_when_later_weeks_are_dropped():
    rows = [example_row(), receivables_row("K", 11, due=(9, 9, 9, 9, 9)),
            receivables_row("K", 12, overdue=1e6)]
    base = features([("K", 10, 2)], rows)
    pd.testing.assert_frame_equal(features([("K", 10, 2)], rows[:1]), base)


def test_features_at_t_do_not_change_when_later_levels_change():
    base = features([("K", 10, 2)], [example_row()], level_map={10: 1.5, 11: 1.6})
    other = features([("K", 10, 2)], [example_row()], level_map={10: 1.5, 11: 99.0})
    pd.testing.assert_frame_equal(base, other)


# --- extra receivables rows are ignored -----------------------------------------------------

def test_receivables_rows_not_in_the_panel_are_ignored():
    rows = [example_row(), receivables_row("OTHER", 10, due=(1e9, 0, 0, 0, 0)),
            receivables_row("K", 3, due=(1e9, 0, 0, 0, 0)),
            receivables_row("K", 64, due=(1e9, 0, 0, 0, 0))]
    table = features([("K", 10, 1)], rows, level_map={10: 1.0})
    assert len(table) == 1
    assert table.iloc[0]["due_in_target_week"] == 100.0


# --- guards -------------------------------------------------------------------------------

def test_a_repeated_key_and_week_raises_naming_it():
    rows = [example_row(), example_row(), receivables_row("K", 11)]
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1)], rows)
    assert str(error.value) == "1 receivables rows repeat a key and week (first: [('K', 10)])"


def test_a_repeated_key_and_week_raises_even_if_not_in_the_panel():
    rows = [example_row(), receivables_row("X", 3), receivables_row("X", 3)]
    with pytest.raises(ValueError, match=r"repeat a key and week \(first: \[\('X', 3\)\]\)"):
        features([("K", 10, 1)], rows)


def test_repeated_key_and_week_lists_at_most_five():
    rows = [receivables_row(f"K{n}", 1) for n in range(7)] * 2
    with pytest.raises(ValueError) as error:
        features([("K0", 1, 1)], rows)
    message = str(error.value)
    assert message.startswith("7 receivables rows repeat a key and week (first: [")
    assert message.count("('K") == 5


@pytest.mark.parametrize("bad", [0, -1, 6, 50])
def test_an_h_outside_1_to_horizons_raises(bad):
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1), ("K", 10, bad)], [example_row()])
    assert str(error.value) == f"panel rows have h [{bad}]; it must be 1 to 5"


def test_every_bad_h_is_named_sorted():
    with pytest.raises(ValueError) as error:
        features([("K", 10, 7), ("K", 10, 0), ("K", 10, 7), ("K", 10, 2)], [example_row()])
    assert str(error.value) == "panel rows have h [0, 7]; it must be 1 to 5"


def test_an_h_above_a_smaller_horizons_raises():
    rows = [receivables_row("K", 1, due=(1, 2, 3))]
    with pytest.raises(ValueError, match=r"panel rows have h \[4\]; it must be 1 to 3"):
        features([("K", 1, 4)], rows, horizons=3)


def gap_message(n, first):
    return (f"{n} panel rows have no receivables row in a week from their key's first one on "
            f"(first: {first})")


def test_a_panel_row_after_the_first_week_without_a_receivables_row_raises_naming_it():
    """K's first ledger week is 10, so K at 11 is a gap; Q's ledger starts in week 12, so Q
    at 10 is not yet in the ledger and is filled, not counted."""
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1), ("K", 11, 1), ("K", 11, 2), ("Q", 10, 1)],
                 [example_row(), receivables_row("Q", 12)], level_map={10: 1.0, 11: 1.0})
    assert str(error.value) == gap_message(2, [("K", 11), ("K", 11)])


def test_missing_receivables_rows_list_at_most_five():
    rows = [receivables_row(f"M{n}", w) for n in range(8) for w in (1, 3)]
    triples = [(f"M{n}", 2, 1) for n in range(8)]
    with pytest.raises(ValueError) as error:
        features(triples, rows)
    message = str(error.value)
    assert message.startswith("8 panel rows have no receivables row in a week from their "
                              "key's first one on (first: [")
    assert message.count("('M") == 5
    assert message.endswith("('M4', 2)])")


def test_the_old_missing_row_message_is_gone():
    with pytest.raises(ValueError) as error:
        features([("K", 11, 1)], [example_row()], level_map={11: 1.0})
    assert "no receivables row (first:" not in str(error.value)
    assert str(error.value) == gap_message(1, [("K", 11)])


def test_a_missing_price_level_raises_naming_the_weeks():
    rows = [example_row(), receivables_row("K", 11), receivables_row("K", 12)]
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1), ("K", 11, 1), ("K", 12, 1), ("K", 12, 2)], rows,
                 level_map={10: 1.0})
    assert str(error.value) == "no price level for weeks [11, 12]"


def test_missing_price_levels_list_at_most_ten_weeks():
    weeks = list(range(20, 35))
    rows = [receivables_row("K", w) for w in weeks]
    with pytest.raises(ValueError) as error:
        features([("K", w, 1) for w in weeks], rows, level_map={0: 1.0})
    assert str(error.value) == f"no price level for weeks {weeks[:10]}"


def test_the_guards_run_in_order_repeat_then_h_then_missing_row():
    rows = [example_row(), example_row()]
    with pytest.raises(ValueError, match="repeat a key and week"):
        features([("K", 10, 9), ("Q", 1, 1)], rows)
    with pytest.raises(ValueError, match="panel rows have h"):
        features([("K", 10, 9), ("Q", 1, 1)], rows[:1])


def test_check_receivables_for_features_direct_calls():
    receivables = receivables_table([example_row()])
    check_receivables_for_features(panel([("K", 10, 1), ("K", 10, 5)]), receivables, 5)
    with pytest.raises(ValueError, match="it must be 1 to 4"):
        check_receivables_for_features(panel([("K", 10, 5)]), receivables, 4)
    with pytest.raises(ValueError, match="repeat a key and week"):
        check_receivables_for_features(panel([("K", 10, 1)]),
                                       receivables_table([example_row()] * 2), 5)


def split_inputs(before, at, after, due, due_later, overdue):
    features_table = pd.DataFrame({"due_before_target_week": [before],
                                   "due_in_target_week": [at],
                                   "due_after_target_week": [after]})
    rows = pd.DataFrame({**{f"due_in_week_{h}": [v] for h, v in enumerate(due, start=1)},
                         "due_later": [due_later], "overdue": [overdue]})
    return features_table, rows


def test_check_due_split_passes_when_the_parts_add_up():
    features_table, rows = split_inputs(10, 20, 70, (10, 20, 30), 40, 5)
    check_due_split(features_table, rows, due_in_week_columns(3))   # no error


def test_check_due_split_raises_when_an_amount_is_lost():
    features_table, rows = split_inputs(10, 20, 30, (10, 20, 30), 40, 5)
    with pytest.raises(ValueError) as error:
        check_due_split(features_table, rows, due_in_week_columns(3))
    assert str(error.value) == ("the due-date split differs from the open invoices' total in "
                                "1 rows")


def test_check_due_split_raises_when_an_amount_is_doubled_and_counts_rows():
    features_table = pd.DataFrame({"due_before_target_week": [10.0, 0.0, 5.0],
                                   "due_in_target_week": [20.0, 0.0, 5.0],
                                   "due_after_target_week": [100.0, 0.0, 0.0]})
    rows = pd.DataFrame({"due_in_week_1": [10.0, 0.0, 5.0], "due_in_week_2": [20.0, 0.0, 5.0],
                         "due_later": [70.0, 0.0, 0.0], "overdue": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="in 1 rows"):
        check_due_split(features_table, rows, due_in_week_columns(2))


def test_check_due_split_tolerance_is_one_kurus():
    within, rows = split_inputs(10, 20, 70.009, (10, 20, 30), 40, 5)
    check_due_split(within, rows, due_in_week_columns(3))   # no error
    beyond, rows = split_inputs(10, 20, 70.02, (10, 20, 30), 40, 5)
    with pytest.raises(ValueError, match="differs"):
        check_due_split(beyond, rows, due_in_week_columns(3))


def test_receivables_features_runs_the_due_split_check(monkeypatch):
    """The split written with <, == and > is a partition, so it cannot fail on real numbers;
    the check is a guard against a future change. Make it strict to see it is called."""
    calls = []
    real = features_receivables.check_due_split

    def spy(features_table, rows, due_columns):
        calls.append(list(due_columns))
        real(features_table, rows, due_columns)

    monkeypatch.setattr(features_receivables, "check_due_split", spy)
    features([("K", 10, 2)], [example_row()])
    assert calls == [DUE]
    monkeypatch.setattr(features_receivables, "SPLIT_TOLERANCE", -1.0)
    with pytest.raises(ValueError, match="the due-date split differs"):
        features([("K", 10, 2)], [example_row()])


def test_the_due_split_check_sees_the_raw_amounts_before_deflation(monkeypatch):
    seen = {}

    def spy(features_table, rows, due_columns):
        seen["before"] = features_table["due_before_target_week"].tolist()

    monkeypatch.setattr(features_receivables, "check_due_split", spy)
    features([("K", 10, 2)], [example_row()], level_map={10: 4.0})
    assert seen["before"] == [100.0]


# --- dtypes and inputs ----------------------------------------------------------------------

def test_dtypes():
    table = features([("K", 10, 1), ("K", 10, 2)], [example_row()])
    assert table["key"].dtype == object
    for column in INT_FEATURES:
        assert table[column].dtype == "int64", column
    for column in AMOUNT_FEATURES + ["overdue_days", "usual_delay_days"]:
        assert table[column].dtype == "float64", column


def test_inputs_are_not_modified():
    panel_rows = panel([("K", 10, 2), ("K", 10, 1)], index=pd.Index([5, 3]))
    receivables = receivables_table([example_row(), receivables_row("K", 11)])
    level_series = levels({10: 2.0, 11: 1.0})
    copies = panel_rows.copy(), receivables.copy(), level_series.copy()
    receivables_features(panel_rows, receivables, level_series, HORIZONS, AGE_EDGES)
    pd.testing.assert_frame_equal(panel_rows, copies[0])
    pd.testing.assert_frame_equal(receivables, copies[1])
    pd.testing.assert_series_equal(level_series, copies[2])


def test_same_inputs_give_the_same_table_twice():
    triples = [("A", 1, 3), ("B", 2, 1), ("C", 2, 5)]
    pd.testing.assert_frame_equal(features(triples, three_keys()), features(triples, three_keys()))


# --- guard: bad age edges ---------------------------------------------------------------------

def edges_message(age_edges):
    return (f"age_edges are {age_edges}; give whole numbers of days (1 or more), "
            f"from smallest to largest, each once")


@pytest.mark.parametrize("bad", [(90, 30), (30, 30), (0, 30), (30.0, 90), ()],
                         ids=["reversed", "repeated", "zero", "float", "empty"])
def test_bad_age_edges_raise_the_edges_message(bad):
    """The table is built with good edges; only the edges passed to the function are bad."""
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("K", 10, 1)]), receivables_table([example_row()]),
                             levels({10: 1.0}), HORIZONS, bad)
    assert str(error.value) == edges_message(bad)


def test_numpy_integer_age_edges_are_accepted_and_match_python_ones():
    triples = [("K", 10, h) for h in range(1, 6)]
    plain = features(triples, [example_row()])
    numpy_edges = receivables_features(panel(triples), receivables_table([example_row()]),
                                       levels({10: 1.0}), HORIZONS,
                                       (np.int64(30), np.int64(90)))
    pd.testing.assert_frame_equal(plain, numpy_edges)


def test_the_repeat_check_comes_before_the_edges_check():
    receivables = receivables_table([example_row(), example_row()])
    with pytest.raises(ValueError, match="repeat a key and week"):
        receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             (90, 30))


def test_the_h_check_comes_before_the_edges_check():
    with pytest.raises(ValueError, match="panel rows have h"):
        receivables_features(panel([("K", 10, 9)]), receivables_table([example_row()]),
                             levels({10: 1.0}), HORIZONS, (90, 30))


def test_the_edges_check_comes_before_the_missing_value_check():
    receivables = with_missing(receivables_table([example_row()]), 0, "open_balance")
    with pytest.raises(ValueError, match=r"^age_edges are \(30, 30\);"):
        receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             (30, 30))


# --- guard: missing values in the receivables table -------------------------------------------

CHECKED_COLUMNS = (DUE + ["due_later", "overdue"] + AGES
                   + ["overdue_days", "open_invoices", "cleared_invoices", "unapplied_cash",
                      "open_balance"])


def missing_message(columns):
    return f"receivables columns {columns} have missing values; only usual_delay_days may"


def with_missing(table, position, column):
    """A copy of `table` with a NaN at row `position` of `column` (count columns become float,
    as they would after a merge that left a gap)."""
    table = table.copy()
    if table[column].dtype != "float64":
        table[column] = table[column].astype("float64")
    table.loc[table.index[position], column] = np.nan
    return table


@pytest.mark.parametrize("column", CHECKED_COLUMNS)
def test_a_nan_in_any_checked_column_raises_naming_it(column):
    receivables = with_missing(receivables_table([example_row()]), 0, column)
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             AGE_EDGES)
    assert str(error.value) == missing_message([column])


@pytest.mark.parametrize("column", CHECKED_COLUMNS)
def test_a_nan_in_a_row_the_panel_does_not_use_still_raises(column):
    receivables = with_missing(
        receivables_table([example_row(), receivables_row("UNUSED", 3)]), 1, column)
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             AGE_EDGES)
    assert str(error.value) == missing_message([column])


def test_a_nan_in_an_unused_row_raises_even_for_an_empty_panel():
    receivables = with_missing(receivables_table([example_row()]), 0, "due_later")
    with pytest.raises(ValueError) as error:
        receivables_features(panel([]), receivables, levels({}), HORIZONS, AGE_EDGES)
    assert str(error.value) == missing_message(["due_later"])


def test_several_columns_with_nans_are_listed_in_the_fixed_order():
    receivables = receivables_table([example_row(), receivables_row("B", 10)])
    # Set out of order, on different rows; the message follows the checked order.
    for position, column in [(1, "open_balance"), (0, "overdue_31_to_90"), (1, "due_in_week_2"),
                             (0, "cleared_invoices"), (0, "due_later")]:
        receivables = with_missing(receivables, position, column)
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             AGE_EDGES)
    assert str(error.value) == missing_message(
        ["due_in_week_2", "due_later", "overdue_31_to_90", "cleared_invoices", "open_balance"])


def test_a_column_with_many_nans_is_listed_once():
    receivables = receivables_table([example_row(), receivables_row("B", 10),
                                     receivables_row("C", 10)])
    for position in range(3):
        receivables = with_missing(receivables, position, "overdue_days")
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             AGE_EDGES)
    assert str(error.value) == missing_message(["overdue_days"])


def test_every_checked_column_with_a_nan_is_listed():
    receivables = receivables_table([example_row()])
    for column in CHECKED_COLUMNS:
        receivables = with_missing(receivables, 0, column)
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             AGE_EDGES)
    assert str(error.value) == missing_message(CHECKED_COLUMNS)


def test_the_checked_columns_follow_horizons_and_age_edges():
    rows = [receivables_row("K", 1, due=(1, 2, 3), overdue=5, age_edges=(30, 60, 90))]
    receivables = receivables_table(rows, horizons=3, age_edges=(30, 60, 90))
    receivables = with_missing(receivables, 0, "due_in_week_3")
    receivables = with_missing(receivables, 0, "overdue_61_to_90")
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("K", 1, 1)]), receivables, levels({1: 1.0}), 3,
                             (30, 60, 90))
    assert str(error.value) == missing_message(["due_in_week_3", "overdue_61_to_90"])


def test_a_nan_usual_delay_is_still_fine_in_used_and_unused_rows():
    receivables = receivables_table([example_row(usual_delay_days=float("nan")),
                                     receivables_row("UNUSED", 3)])
    assert receivables["usual_delay_days"].isna().all()
    table = receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}),
                                 HORIZONS, AGE_EDGES)
    assert np.isnan(table.iloc[0]["usual_delay_days"])
    assert table.iloc[0]["due_in_target_week"] == 100.0


def test_a_nan_in_a_column_that_is_not_checked_is_ignored():
    receivables = receivables_table([example_row()])
    receivables["something_else"] = np.nan
    table = receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}),
                                 HORIZONS, AGE_EDGES)
    assert table.iloc[0]["due_in_target_week"] == 100.0
    assert "something_else" not in table.columns


def test_the_h_check_comes_before_the_missing_value_check():
    receivables = with_missing(receivables_table([example_row()]), 0, "overdue")
    with pytest.raises(ValueError, match="panel rows have h"):
        receivables_features(panel([("K", 10, 0)]), receivables, levels({10: 1.0}), HORIZONS,
                             AGE_EDGES)


def test_the_repeat_check_comes_before_the_missing_value_check():
    receivables = with_missing(receivables_table([example_row(), example_row()]), 0, "overdue")
    with pytest.raises(ValueError, match="repeat a key and week"):
        receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             AGE_EDGES)


def test_the_missing_value_check_comes_before_the_missing_row_check():
    """K at 11 is a gap; NOWHERE's ledger starts later, so it is not an absent key."""
    receivables = with_missing(
        receivables_table([example_row(), receivables_row("NOWHERE", 20)]), 0, "unapplied_cash")
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("K", 11, 1), ("NOWHERE", 10, 1)]), receivables,
                             levels({10: 1.0, 11: 1.0}), HORIZONS, AGE_EDGES)
    assert str(error.value) == missing_message(["unapplied_cash"])


def test_the_missing_value_check_comes_before_the_level_checks():
    receivables = with_missing(receivables_table([example_row()]), 0, "due_in_week_1")
    for level_map in ({}, {10: 0.0}):
        with pytest.raises(ValueError) as error:
            receivables_features(panel([("K", 10, 1)]), receivables, levels(level_map),
                                 HORIZONS, AGE_EDGES)
        assert str(error.value) == missing_message(["due_in_week_1"])


def test_a_nan_that_would_pass_the_due_split_is_caught():
    """A NaN due-in-week value drops out of the sums on both sides of the due-date check, so
    that check alone would not see it; the missing-value check does."""
    receivables = with_missing(receivables_table([example_row()]), 0, "due_in_week_3")
    with pytest.raises(ValueError, match=r"\['due_in_week_3'\] have missing values"):
        receivables_features(panel([("K", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             AGE_EDGES)


def test_check_no_missing_values_direct_calls():
    table = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, 1.0], "c": [3.0, np.nan]})
    check_no_missing_values(table, ["a"])                       # no error
    check_no_missing_values(table, [])                          # no error
    check_no_missing_values(table.iloc[0:0], ["a", "b", "c"])   # empty: nothing missing
    with pytest.raises(ValueError) as error:
        check_no_missing_values(table, ["c", "a", "b"])
    assert str(error.value) == missing_message(["c", "b"])


def test_check_no_missing_values_treats_none_and_nat_as_missing():
    with pytest.raises(ValueError, match=r"\['x'\]"):
        check_no_missing_values(pd.DataFrame({"x": pd.Series([1, None], dtype=object)}), ["x"])
    with pytest.raises(ValueError, match=r"\['t'\]"):
        check_no_missing_values(pd.DataFrame({"t": pd.to_datetime(["2026-01-01", None])}),
                                ["t"])


def test_check_no_missing_values_does_not_modify_its_input():
    table = pd.DataFrame({"a": [1.0, np.nan]})
    before = table.copy()
    with pytest.raises(ValueError):
        check_no_missing_values(table, ["a"])
    pd.testing.assert_frame_equal(table, before)


# --- guard: a price level of 0 or below -------------------------------------------------------

def level_message(weeks):
    return f"the price level is 0 or below for weeks {weeks}"


@pytest.mark.parametrize("bad", [0.0, -0.0, -1.0, -1e-12, -np.inf],
                         ids=["zero", "negative-zero", "minus-one", "tiny-negative", "minus-inf"])
def test_a_level_of_0_or_below_raises_naming_the_week(bad):
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1)], [example_row()], level_map={10: bad})
    assert str(error.value) == level_message([10])


def test_a_tiny_positive_level_is_accepted():
    row = one([("K", 10, 1)], [example_row()], level_map={10: 1e-12})
    assert row["due_in_target_week"] == pytest.approx(100.0 / 1e-12)


def test_bad_level_weeks_are_sorted_and_distinct():
    weeks = [14, 11, 12, 13]
    rows = [receivables_row("K", w) for w in weeks]
    triples = [("K", w, h) for w in weeks for h in (1, 2)]
    with pytest.raises(ValueError) as error:
        features(triples, rows, level_map={14: 0.0, 11: -2.0, 12: 1.0, 13: 0.0})
    assert str(error.value) == level_message([11, 13, 14])


def test_bad_level_weeks_list_at_most_ten():
    weeks = list(range(40, 25, -1))   # 15 weeks, given in reverse
    rows = [receivables_row("K", w) for w in weeks]
    with pytest.raises(ValueError) as error:
        features([("K", w, 1) for w in weeks], rows, level_map={w: 0.0 for w in weeks})
    assert str(error.value) == level_message(list(range(26, 36)))


def test_a_zero_level_in_a_week_the_panel_does_not_use_is_ignored():
    rows = [example_row(), receivables_row("K", 11)]
    table = features([("K", 10, 1)], rows, level_map={10: 2.0, 11: 0.0, 12: -5.0})
    assert table.iloc[0]["due_in_target_week"] == 50.0


def test_a_zero_level_is_ignored_for_an_empty_panel():
    result = receivables_features(panel([]), receivables_table([example_row()]),
                                  levels({10: 0.0}), HORIZONS, AGE_EDGES)
    assert result.empty


def test_the_missing_level_check_comes_before_the_non_positive_level_check():
    rows = [example_row(), receivables_row("K", 11)]
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1), ("K", 11, 1)], rows, level_map={10: 0.0})
    assert str(error.value) == "no price level for weeks [11]"


def test_the_missing_row_check_comes_before_the_level_checks():
    with pytest.raises(ValueError, match="have no receivables row in a week from their key"):
        features([("K", 10, 1), ("K", 11, 1)], [example_row()], level_map={10: -1.0})
    with pytest.raises(ValueError, match="have no receivables row in a week from their key"):
        features([("K", 10, 1), ("K", 11, 1)], [example_row()], level_map={10: 1.0})


def test_the_due_split_check_comes_before_the_level_checks(monkeypatch):
    monkeypatch.setattr(features_receivables, "SPLIT_TOLERANCE", -1.0)
    with pytest.raises(ValueError, match="the due-date split differs"):
        features([("K", 10, 1)], [example_row()], level_map={10: 0.0})


# --- keys not yet in the ledger: filled as "nothing owed" -------------------------------------

ZERO_FEATURES = [column for column in FEATURE_COLUMNS[3:] if column != "usual_delay_days"]


def assert_nothing_owed(row):
    for column in ZERO_FEATURES:
        assert row[column] == 0, column
    assert np.isnan(row["usual_delay_days"])


def friday_monday_rows():
    """K's ledger starts in week 11 (paid on a Friday of week 10, posted on the Monday):
    week 11 owes 7 due next week, week 12 is the worked example."""
    return [receivables_row("K", 11, due=(7, 0, 0, 0, 0), open_invoices=2, open_balance=7.0,
                            usual_delay_days=3.0, cleared_invoices=1),
            dict(example_row(), week=12)]


def with_new_later():
    """`friday_monday_rows` plus a key NEW whose ledger starts in week 20, after every panel
    week these tests use: NEW is not yet in the ledger (it has rows, so it is not absent)."""
    return friday_monday_rows() + [receivables_row("NEW", 20, due=(1e9, 0, 0, 0, 0))]


def test_friday_monday_week_10_is_filled_and_11_12_come_from_the_table():
    table = features([("K", 10, 1), ("K", 11, 1), ("K", 12, 2)], friday_monday_rows(),
                     level_map={10: 1.0, 11: 1.0, 12: 1.0})
    assert list(zip(table["key"], table["origin_week"], table["h"])) == [
        ("K", 10, 1), ("K", 11, 1), ("K", 12, 2)]
    assert_nothing_owed(table.iloc[0])
    week_11 = table.iloc[1]
    assert week_11["due_in_target_week"] == 7.0
    assert week_11["open_invoices"] == 2 and week_11["cleared_invoices"] == 1
    assert week_11["usual_delay_days"] == 3.0
    week_12 = table.iloc[2]
    assert week_12["due_before_target_week"] == 100.0
    assert week_12["due_in_target_week"] == 40.0
    assert week_12["due_after_target_week"] == 500.0
    assert week_12["open_balance"] == 915.0
    assert week_12["open_invoices"] == 4 and week_12["usual_delay_days"] == 12.5


def test_the_filled_week_does_not_borrow_from_the_next_week():
    """Week 10 must not be filled from week 11 (that would leak the future)."""
    table = features([("K", 10, h) for h in range(1, 6)], friday_monday_rows(),
                     level_map={10: 1.0})
    for _, row in table.iterrows():
        assert_nothing_owed(row)


@pytest.mark.parametrize("h", range(1, HORIZONS + 1))
def test_every_h_of_a_filled_origin_week_is_nothing_owed(h):
    row = one([("K", 10, h)], friday_monday_rows(), level_map={10: 1.0})
    assert row["h"] == h
    assert_nothing_owed(row)


def test_all_h_of_a_filled_origin_week_in_one_call():
    table = features([("K", 10, h) for h in range(1, 6)], friday_monday_rows(),
                     level_map={10: 1.0})
    assert table["h"].tolist() == [1, 2, 3, 4, 5]
    assert (table[ZERO_FEATURES] == 0).all().all()
    assert table["usual_delay_days"].isna().all()


def test_several_weeks_before_the_first_are_all_filled():
    table = features([("K", w, 1) for w in (3, 7, 10, 11)], friday_monday_rows(),
                     level_map={3: 1.0, 7: 1.0, 10: 1.0, 11: 1.0})
    for position in range(3):
        assert_nothing_owed(table.iloc[position])
    assert table.iloc[3]["due_in_target_week"] == 7.0


def test_a_key_whose_ledger_starts_after_every_panel_week_is_filled():
    """Q has receivables rows, all later than its panel rows: genuinely not yet in the
    ledger, so it is filled (contrast: a key with no rows at all raises)."""
    table = features([("Q", 10, 1), ("Q", 11, 3)], [example_row(), receivables_row("Q", 30)],
                     level_map={10: 1.0, 11: 1.0})
    assert table["key"].tolist() == ["Q", "Q"]
    for _, row in table.iterrows():
        assert_nothing_owed(row)


# --- guard: a panel key with no receivables row at all -----------------------------------------

def absent_message(n, first):
    return f"{n} panel keys have no receivables row at all (first: {first})"


def test_a_key_with_no_receivables_rows_at_all_raises():
    with pytest.raises(ValueError) as error:
        features([("Q", 10, 1), ("Q", 11, 3)], [example_row()], level_map={10: 1.0, 11: 1.0})
    assert str(error.value) == absent_message(1, ["Q"])


def test_an_absent_key_raises_even_beside_good_keys():
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1), ("Q", 10, 1), ("K", 10, 2)], [example_row()],
                 level_map={10: 1.0})
    assert str(error.value) == absent_message(1, ["Q"])


def test_an_empty_receivables_table_with_a_non_empty_panel_raises():
    with pytest.raises(ValueError) as error:
        features([("A", 1, 1), ("B", 2, 5)], [], level_map={1: 1.0, 2: 1.0})
    assert str(error.value) == absent_message(2, ["A", "B"])


def test_an_empty_panel_with_empty_receivables_gives_an_empty_table():
    result = receivables_features(panel([]), receivables_table([]), levels({}), HORIZONS,
                                  AGE_EDGES)
    assert result.empty
    assert list(result.columns) == FEATURE_COLUMNS


def test_absent_keys_are_counted_once_each_not_per_row():
    triples = [("Q", w, h) for w in (10, 11) for h in range(1, 6)] + [("R", 10, 1)]
    with pytest.raises(ValueError) as error:
        features(triples, [example_row()], level_map={10: 1.0, 11: 1.0})
    assert str(error.value) == absent_message(2, ["Q", "R"])


def test_absent_keys_are_listed_sorted_and_distinct():
    triples = [("Z", 10, 1), ("M", 10, 1), ("Z", 10, 2), ("B", 10, 1), ("M", 10, 3)]
    with pytest.raises(ValueError) as error:
        features(triples, [example_row()], level_map={10: 1.0})
    assert str(error.value) == absent_message(3, ["B", "M", "Z"])


def test_absent_keys_list_at_most_five_but_count_all():
    keys = [f"X{n}" for n in (7, 3, 0, 5, 1, 6, 2, 4)]   # 8 keys, shuffled
    triples = [(key, 10, 1) for key in keys] + [("K", 10, 1)]
    with pytest.raises(ValueError) as error:
        features(triples, [example_row()], level_map={10: 1.0})
    assert str(error.value) == absent_message(8, ["X0", "X1", "X2", "X3", "X4"])


def test_keys_only_in_receivables_are_not_absent():
    """The check is one-way: extra keys in the receivables table are fine."""
    table = features([("K", 10, 1)], [example_row(), receivables_row("EXTRA", 10)],
                     level_map={10: 1.0})
    assert table["key"].tolist() == ["K"]


def test_the_repeat_check_comes_before_the_absent_key_check():
    with pytest.raises(ValueError, match="repeat a key and week"):
        features([("Q", 10, 1)], [example_row(), example_row()])


def test_the_h_check_comes_before_the_absent_key_check():
    with pytest.raises(ValueError, match="panel rows have h"):
        features([("Q", 10, 9)], [example_row()])


def test_the_absent_key_check_comes_before_the_edges_check():
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("Q", 10, 1)]), receivables_table([example_row()]),
                             levels({10: 1.0}), HORIZONS, (90, 30))
    assert str(error.value) == absent_message(1, ["Q"])


def test_the_absent_key_check_comes_before_the_missing_value_check():
    receivables = with_missing(receivables_table([example_row()]), 0, "overdue")
    with pytest.raises(ValueError) as error:
        receivables_features(panel([("Q", 10, 1)]), receivables, levels({10: 1.0}), HORIZONS,
                             AGE_EDGES)
    assert str(error.value) == absent_message(1, ["Q"])


def test_the_absent_key_check_comes_before_the_gap_check():
    with pytest.raises(ValueError) as error:
        features([("K", 11, 1), ("Q", 10, 1)], [example_row()], level_map={10: 1.0, 11: 1.0})
    assert str(error.value) == absent_message(1, ["Q"])


@pytest.mark.parametrize("level_map", [{}, {10: 0.0}, {10: -1.0}], ids=["none", "zero", "neg"])
def test_the_absent_key_check_comes_before_the_level_checks(level_map):
    with pytest.raises(ValueError) as error:
        features([("Q", 10, 1)], [example_row()], level_map=level_map)
    assert str(error.value) == absent_message(1, ["Q"])


def test_the_absent_key_check_comes_before_the_due_split_check(monkeypatch):
    monkeypatch.setattr(features_receivables, "SPLIT_TOLERANCE", -1.0)
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1), ("Q", 10, 1)], [example_row()], level_map={10: 1.0})
    assert str(error.value) == absent_message(1, ["Q"])


def test_the_absent_key_check_does_not_modify_its_inputs():
    panel_rows = panel([("Q", 10, 1), ("K", 10, 2)], index=pd.Index([4, 1]))
    receivables = receivables_table([example_row()])
    level_series = levels({10: 1.0})
    copies = panel_rows.copy(), receivables.copy(), level_series.copy()
    with pytest.raises(ValueError, match="no receivables row at all"):
        receivables_features(panel_rows, receivables, level_series, HORIZONS, AGE_EDGES)
    pd.testing.assert_frame_equal(panel_rows, copies[0])
    pd.testing.assert_frame_equal(receivables, copies[1])
    pd.testing.assert_series_equal(level_series, copies[2])


def test_a_gap_after_the_first_week_raises():
    rows = [receivables_row("K", 11), receivables_row("K", 13)]
    with pytest.raises(ValueError) as error:
        features([("K", 11, 1), ("K", 12, 1), ("K", 12, 2), ("K", 13, 1)], rows,
                 level_map={11: 1.0, 12: 1.0, 13: 1.0})
    assert str(error.value) == gap_message(2, [("K", 12), ("K", 12)])


def test_a_gap_raises_even_with_filled_rows_of_the_same_key():
    rows = [receivables_row("K", 11), receivables_row("K", 13)]
    with pytest.raises(ValueError) as error:
        features([("K", 9, 1), ("K", 12, 1)], rows, level_map={9: 1.0, 12: 1.0})
    assert str(error.value) == gap_message(1, [("K", 12)])


def test_a_panel_row_after_the_last_ledger_week_raises():
    """Past the key's last week is also 'from its first one on': a gap, not a fill."""
    with pytest.raises(ValueError) as error:
        features([("K", 99, 1)], [example_row()], level_map={99: 1.0})
    assert str(error.value) == gap_message(1, [("K", 99)])


def test_mixed_keys_keep_the_panel_order_and_index():
    rows = with_new_later() + [receivables_row("B", 10, due=(5, 0, 0, 0, 0))]
    triples = [("B", 10, 1), ("K", 10, 2), ("K", 11, 1), ("NEW", 11, 4), ("B", 10, 2),
               ("K", 10, 1)]
    panel_rows = panel(triples, index=pd.Index([50, 40, 30, 20, 10, 0]))
    result = receivables_features(panel_rows, receivables_table(rows),
                                  levels({10: 1.0, 11: 1.0}), HORIZONS, AGE_EDGES)
    assert list(zip(result["key"], result["origin_week"], result["h"])) == triples
    assert result.index.tolist() == list(range(len(triples)))
    assert result["due_in_target_week"].tolist() == [5.0, 0.0, 7.0, 0.0, 0.0, 0.0]
    assert result["due_before_target_week"].tolist() == [0.0, 0.0, 0.0, 0.0, 5.0, 0.0]
    for position in (1, 3, 5):
        assert_nothing_owed(result.iloc[position])
    assert result.iloc[2]["usual_delay_days"] == 3.0


def test_filled_rows_match_the_reference_on_a_padded_table():
    """A random table plus panel rows before each key's first week: the table rows match
    the brute-force reference, the earlier ones are nothing owed."""
    rng = random.Random(5)
    keys = ["A", "B", "C"]
    rows = random_receivables(rng, keys, [20, 21, 22])
    triples = [(k, w, h) for k in keys for w in (18, 19, 20, 21, 22) for h in range(1, 6)]
    rng.shuffle(triples)
    level_map = {w: rng.uniform(0.5, 3.0) for w in range(18, 23)}
    receivables = receivables_table(rows)
    result = receivables_features(panel(triples), receivables, levels(level_map), HORIZONS,
                                  AGE_EDGES)
    assert list(zip(result["key"], result["origin_week"], result["h"])) == triples
    in_table = result["origin_week"] >= 20
    expected = reference(panel([t for t in triples if t[1] >= 20]), receivables, level_map,
                         HORIZONS, AGE_EDGES)
    pd.testing.assert_frame_equal(result[in_table].reset_index(drop=True), expected,
                                  check_dtype=False, rtol=1e-12)
    assert (result.loc[~in_table, ZERO_FEATURES] == 0).all().all()
    assert result.loc[~in_table, "usual_delay_days"].isna().all()


def test_filled_rows_pass_the_due_split_check(monkeypatch):
    seen = {}
    real = features_receivables.check_due_split

    def spy(features_table, rows, due_columns):
        seen["rows"] = len(features_table)
        seen["totals"] = (rows[due_columns].sum(axis=1, skipna=False) + rows["due_later"]
                          + rows["overdue"]).tolist()
        real(features_table, rows, due_columns)

    monkeypatch.setattr(features_receivables, "check_due_split", spy)
    features([("K", 10, h) for h in range(1, 6)], friday_monday_rows(), level_map={10: 1.0})
    assert seen["rows"] == 5
    assert seen["totals"] == [0.0] * 5   # no NaN: the fill comes before the check


def test_filled_zeros_stay_zero_after_deflation():
    for level in (0.25, 1.0, 7.0):
        row = one([("K", 10, 3)], friday_monday_rows(), level_map={10: level})
        assert_nothing_owed(row)


def test_a_filled_week_still_needs_a_price_level():
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1), ("K", 11, 1)], friday_monday_rows(), level_map={11: 1.0})
    assert str(error.value) == "no price level for weeks [10]"


@pytest.mark.parametrize("bad", [0.0, -2.0])
def test_a_filled_week_still_needs_a_positive_price_level(bad):
    with pytest.raises(ValueError) as error:
        features([("K", 10, 1)], friday_monday_rows(), level_map={10: bad})
    assert str(error.value) == level_message([10])


def test_count_dtypes_are_int64_when_rows_were_filled():
    mixed = features([("K", 10, 1), ("K", 11, 1)], friday_monday_rows(),
                     level_map={10: 1.0, 11: 1.0})
    only_filled = features([("K", 10, 1), ("K", 10, 2)], friday_monday_rows(),
                           level_map={10: 1.0})
    for table in (mixed, only_filled):
        assert table["key"].dtype == object
        for column in INT_FEATURES:
            assert table[column].dtype == "int64", column
        for column in AMOUNT_FEATURES + ["overdue_days", "usual_delay_days"]:
            assert table[column].dtype == "float64", column
    assert mixed["open_invoices"].tolist() == [0, 2]
    assert mixed["cleared_invoices"].tolist() == [0, 1]


def test_filled_rows_follow_the_horizons_and_age_edges():
    rows = [receivables_row("K", 5, due=(1, 2, 3), overdue=5, age_edges=(30, 60, 90))]
    table = features([("K", 4, 3), ("K", 5, 3)], rows, level_map={4: 1.0, 5: 1.0}, horizons=3,
                     age_edges=(30, 60, 90))
    filled = table.iloc[0]
    for column in table.columns[3:]:
        if column == "usual_delay_days":
            assert np.isnan(filled[column])
        else:
            assert filled[column] == 0, column
    assert table.iloc[1]["due_before_target_week"] == 3.0
    assert table.iloc[1]["due_in_target_week"] == 3.0


def test_inputs_are_not_modified_when_rows_are_filled():
    panel_rows = panel([("K", 12, 2), ("K", 10, 1), ("NEW", 11, 3)], index=pd.Index([9, 4, 2]))
    receivables = receivables_table(with_new_later())
    level_series = levels({10: 2.0, 11: 1.0, 12: 3.0})
    copies = panel_rows.copy(), receivables.copy(), level_series.copy()
    receivables_features(panel_rows, receivables, level_series, HORIZONS, AGE_EDGES)
    pd.testing.assert_frame_equal(panel_rows, copies[0])
    pd.testing.assert_frame_equal(receivables, copies[1])
    pd.testing.assert_series_equal(level_series, copies[2])


def test_filling_is_deterministic():
    triples = [("K", 10, 1), ("NEW", 11, 2), ("K", 12, 3)]
    level_map = {10: 1.0, 11: 1.0, 12: 1.0}
    pd.testing.assert_frame_equal(features(triples, with_new_later(), level_map=level_map),
                                  features(triples, with_new_later(), level_map=level_map))


# --- before_first_ledger_week ---------------------------------------------------------------

def test_before_first_ledger_week_friday_monday():
    result = before_first_ledger_week(panel([("K", 10, 1), ("K", 11, 1), ("K", 12, 5)]),
                                      receivables_table(friday_monday_rows()))
    assert result.tolist() == [True, False, False]
    assert result.dtype == bool


def test_before_first_ledger_week_the_first_week_itself_is_not_before():
    result = before_first_ledger_week(panel([("K", 10, 1), ("K", 9, 1)]),
                                      receivables_table([example_row()]))
    assert result.tolist() == [False, True]


def test_before_first_ledger_week_uses_the_minimum_not_the_first_listed_row():
    rows = [receivables_row("K", 20), receivables_row("K", 5), receivables_row("K", 12)]
    result = before_first_ledger_week(panel([("K", 4, 1), ("K", 5, 1), ("K", 8, 1)]),
                                      receivables_table(rows))
    assert result.tolist() == [True, False, False]


def test_before_first_ledger_week_a_gap_or_later_week_is_not_before():
    rows = [receivables_row("K", 11), receivables_row("K", 13)]
    result = before_first_ledger_week(panel([("K", 12, 1), ("K", 40, 1)]),
                                      receivables_table(rows))
    assert result.tolist() == [False, False]


def test_before_first_ledger_week_a_key_without_rows_is_always_true():
    result = before_first_ledger_week(panel([("Q", 0, 1), ("Q", 999, 1), ("K", 10, 1)]),
                                      receivables_table([example_row()]))
    assert result.tolist() == [True, True, False]


def test_before_first_ledger_week_still_flags_a_key_receivables_features_refuses():
    """Unchanged helper: True for an absent key, while `receivables_features` raises on the
    very same inputs."""
    panel_rows = panel([("Q", 10, 1), ("K", 10, 1)])
    receivables = receivables_table([example_row()])
    assert before_first_ledger_week(panel_rows, receivables).tolist() == [True, False]
    with pytest.raises(ValueError, match=r"^1 panel keys have no receivables row at all"):
        receivables_features(panel_rows, receivables, levels({10: 1.0}), HORIZONS, AGE_EDGES)


def test_before_first_ledger_week_empty_receivables_is_all_true():
    result = before_first_ledger_week(panel([("A", 1, 1), ("B", 50, 2)]), receivables_table([]))
    assert result.tolist() == [True, True]
    assert result.dtype == bool


def test_before_first_ledger_week_empty_panel():
    result = before_first_ledger_week(panel([]), receivables_table([example_row()]))
    assert len(result) == 0


def test_before_first_ledger_week_is_aligned_with_a_non_default_index():
    index = pd.Index([70, 3, 15, 42])
    panel_rows = panel([("K", 9, 1), ("K", 10, 1), ("Q", 10, 1), ("K", 11, 2)], index=index)
    result = before_first_ledger_week(panel_rows, receivables_table([example_row()]))
    assert result.index.equals(index)
    assert result.to_dict() == {70: True, 3: False, 15: True, 42: False}


def test_before_first_ledger_week_with_a_string_index():
    panel_rows = panel([("K", 10, 1), ("Q", 10, 1)], index=pd.Index(["x", "y"]))
    result = before_first_ledger_week(panel_rows, receivables_table([example_row()]))
    assert result.index.tolist() == ["x", "y"]
    assert result.tolist() == [False, True]


def test_before_first_ledger_week_ignores_keys_only_in_receivables():
    rows = [example_row(), receivables_row("ONLY_HERE", 1), receivables_row("ONLY_HERE", 2)]
    result = before_first_ledger_week(panel([("K", 9, 1), ("K", 10, 1)]),
                                      receivables_table(rows))
    assert result.tolist() == [True, False]
    assert len(result) == 2


def test_before_first_ledger_week_does_not_modify_its_inputs():
    panel_rows = panel([("K", 9, 1), ("Q", 10, 1)], index=pd.Index([5, 1]))
    receivables = receivables_table(friday_monday_rows())
    copies = panel_rows.copy(), receivables.copy()
    before_first_ledger_week(panel_rows, receivables)
    pd.testing.assert_frame_equal(panel_rows, copies[0])
    pd.testing.assert_frame_equal(receivables, copies[1])


def test_before_first_ledger_week_marks_exactly_the_filled_rows():
    """Its sum is the number of rows `receivables_features` fills (what the pipeline prints)."""
    triples = [("K", 10, 1), ("K", 10, 2), ("K", 11, 1), ("NEW", 11, 1), ("K", 12, 1)]
    flags = before_first_ledger_week(panel(triples), receivables_table(with_new_later()))
    assert int(flags.sum()) == 3
    table = features(triples, with_new_later(), level_map={10: 1.0, 11: 1.0, 12: 1.0})
    filled = table["usual_delay_days"].isna() & (table[ZERO_FEATURES] == 0).all(axis=1)
    assert filled.tolist() == flags.tolist()
