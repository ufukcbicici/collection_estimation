"""Tests for `hurdle/training_table.py`: `join_features` and `check_no_shared_columns`; and
the row filters `rows_known_at` / `rows_at_origin` and `block_feature_columns` (at the end).

`join_features(panel, history, calendar, owed)` left-joins the three feature blocks onto the
panel (from `add_deflated_label`): history on (key, origin_week), the calendar on its `week`
renamed to target_week, the receivables features on (key, origin_week, h). One row per
panel row, in the panel's order, with a fresh RangeIndex. It raises ValueError for a feature
column sharing a name with the panel or an earlier block, for a block repeating its join
keys, and for a panel row with no match in a block; blocks are checked in the order history,
calendar, receivables.

Three groups:
- hand-built blocks with values that name their own row, so a value on the wrong row shows;
- end to end on a small in-memory daily table + ledger, built with the real functions in
  the order of `run_hurdle_pipeline.main()`;
- the LEAK TEST (the plan's check 2): every daily receipt and ledger line dated after the
  end of week T is changed, added or removed; the rows with origin_week <= T must keep
  every feature column, and only labels of rows with target_week > T may move.

Nothing here reads the corpus, nor reads or writes data/interim.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.features_calendar import calendar_features
from collection_estimation.hurdle.features_history import (
    first_history_week,
    history_features,
    ledger_receipts_before,
    weekly_matrix,
)
from collection_estimation.hurdle.features_receivables import receivables_features
from collection_estimation.hurdle.known import known_weeks
from collection_estimation.hurdle.panel import (
    add_deflated_label,
    add_labels,
    panel_rows,
)
from collection_estimation.hurdle.prices import published_levels
from collection_estimation.hurdle.receivables_asof import exchange_rates, receivables_by_week
from collection_estimation.hurdle.series import (
    add_week_index,
    week_calendar,
    week_numbers,
    weekly_amounts,
)
from collection_estimation.hurdle.training_table import (
    block_feature_columns,
    check_no_shared_columns,
    join_features,
    rows_at_origin,
    rows_known_at,
)
from tests.hurdle.workbooks import SYNTHETIC_HOLIDAYS

nan = float("nan")

LABEL_COLUMNS = ["amount", "amount_deflated", "paid"]


# --- hand-built inputs ---------------------------------------------------------------------
# Keys A (index 1) and B (index 2). Panel: A at origins 0 and 1, B at origin 1, h = 1, 2.
# The last week is 2: target week 1 is labelled, 2 is a missing week, 3 is in the future.
# Every feature value encodes its own row, e.g. history's mean_receipts_4 = 100*origin + key.

KEY_INDEX = {"TR02-A": 1, "TR02-B": 2, "TR02-C": 3}


def hand_panel():
    rows = [("TR02-A", 0, 1), ("TR02-A", 0, 2), ("TR02-A", 1, 1), ("TR02-A", 1, 2),
            ("TR02-B", 1, 1), ("TR02-B", 1, 2)]
    panel = pd.DataFrame({
        "key": pd.Series([r[0] for r in rows], dtype=object),
        "company_code": pd.Series(["TR02"] * len(rows), dtype=object),
        "customer_id": pd.Series([r[0].split("-")[1] for r in rows], dtype=object),
        "origin_week": pd.Series([r[1] for r in rows], dtype="int64"),
        "h": pd.Series([r[2] for r in rows], dtype="int64"),
    })
    panel["target_week"] = (panel["origin_week"] + panel["h"]).astype("int64")
    status = {1: "labelled", 2: "missing_week", 3: "future"}
    panel["label_status"] = pd.Series([status[t] for t in panel["target_week"]], dtype=object)
    panel["amount"] = [50.0, nan, nan, nan, nan, nan]
    panel.loc[panel["label_status"] == "labelled", "amount"] = [50.0]
    panel["amount_deflated"] = panel["amount"] / 2.0
    paid = [True if a > 0 else (False if a == 0 else None) for a in panel["amount"]]
    panel["paid"] = pd.array(paid, dtype="boolean")
    return panel


def hand_history(extra=()):
    """One row per (key, origin) for A at 0 and 1, B at 1, plus `extra` (key, origin) rows."""
    pairs = [("TR02-A", 0), ("TR02-A", 1), ("TR02-B", 1)] + list(extra)
    keys = [p[0] for p in pairs]
    origins = [p[1] for p in pairs]
    k = [KEY_INDEX[key] for key in keys]
    return pd.DataFrame({
        "key": pd.Series(keys, dtype=object),
        "origin_week": pd.Series(origins, dtype="int64"),
        "customer_age_weeks": pd.Series([10 * i + o for i, o in zip(k, origins)], dtype="int64"),
        "weeks_since_last_payment": pd.Series([nan if i == 2 else float(o)
                                               for i, o in zip(k, origins)], dtype="float64"),
        "mean_receipts_4": pd.Series([100.0 * o + i for i, o in zip(k, origins)],
                                     dtype="float64"),
        "observed_weeks_4": pd.Series([o + 1 for o in origins], dtype="int64"),
    })


def hand_calendar(first=0, last=5):
    weeks = list(range(first, last + 1))
    return pd.DataFrame({
        "week": pd.Series(weeks, dtype="int64"),
        "business_days": pd.Series([5 - (w % 3) for w in weeks], dtype="int64"),
        "has_month_end": pd.Series([w % 2 == 1 for w in weeks], dtype=bool),
        "has_quarter_end": pd.Series([w == 3 for w in weeks], dtype=bool),
        "has_mid_month": pd.Series([w == 2 for w in weeks], dtype=bool),
        "holidays_week_before": pd.Series([w % 2 for w in weeks], dtype="int64"),
    })


def hand_owed(panel=None, extra=()):
    """One row per panel row (key, origin, h), plus `extra` (key, origin, h) rows."""
    panel = hand_panel() if panel is None else panel
    triples = list(panel[["key", "origin_week", "h"]].itertuples(index=False, name=None))
    triples += list(extra)
    keys = [t[0] for t in triples]
    k = [KEY_INDEX[key] for key in keys]
    origins = [t[1] for t in triples]
    hs = [t[2] for t in triples]
    return pd.DataFrame({
        "key": pd.Series(keys, dtype=object),
        "origin_week": pd.Series(origins, dtype="int64"),
        "h": pd.Series(hs, dtype="int64"),
        "due_in_target_week": pd.Series([1000.0 * o + 10 * h + i
                                         for i, o, h in zip(k, origins, hs)], dtype="float64"),
        "open_invoices": pd.Series([i * 100 + o * 10 + h for i, o, h in zip(k, origins, hs)],
                                   dtype="int64"),
        "usual_delay_days": pd.Series([nan if (i == 1 and h == 2) else 7.0 * i
                                       for i, h in zip(k, hs)], dtype="float64"),
    })


def joined(**overrides):
    inputs = {"panel": hand_panel(), "history": hand_history(), "calendar": hand_calendar(),
              "owed": hand_owed()}
    inputs.update(overrides)
    return join_features(inputs["panel"], inputs["history"], inputs["calendar"],
                         inputs["owed"])


HISTORY_FEATURES = ["customer_age_weeks", "weeks_since_last_payment", "mean_receipts_4",
                    "observed_weeks_4"]
CALENDAR_FEATURES = ["business_days", "has_month_end", "has_quarter_end", "has_mid_month",
                     "holidays_week_before"]
OWED_FEATURES = ["due_in_target_week", "open_invoices", "usual_delay_days"]


# --- values land on the right rows ---------------------------------------------------------

def test_one_row_per_panel_row():
    assert len(joined()) == len(hand_panel())


def test_columns_are_panel_then_history_then_calendar_then_receivables():
    assert list(joined().columns) == (list(hand_panel().columns) + HISTORY_FEATURES
                                      + CALENDAR_FEATURES + OWED_FEATURES)


def test_no_merge_indicator_or_suffixed_columns_left():
    columns = list(joined().columns)
    assert "_merge" not in columns
    assert not any(c.endswith("_x") or c.endswith("_y") for c in columns)
    assert "week" not in columns     # the calendar's week became target_week, the join key


def test_panel_columns_are_unchanged():
    table = joined()
    panel = hand_panel()
    pd.testing.assert_frame_equal(table[list(panel.columns)], panel)


def test_history_row_is_copied_to_every_h_of_its_key_and_origin():
    table = joined()
    for _, row in table.iterrows():
        i, o = KEY_INDEX[row["key"]], row["origin_week"]
        assert row["customer_age_weeks"] == 10 * i + o
        assert row["mean_receipts_4"] == 100.0 * o + i
        assert row["observed_weeks_4"] == o + 1
    a0 = table[(table["key"] == "TR02-A") & (table["origin_week"] == 0)]
    assert a0["h"].tolist() == [1, 2]
    assert a0["customer_age_weeks"].tolist() == [10, 10]


def test_calendar_row_goes_to_every_row_with_that_target_week():
    table = joined()
    calendar = hand_calendar().set_index("week")
    for _, row in table.iterrows():
        week = row["target_week"]
        for column in CALENDAR_FEATURES:
            assert row[column] == calendar.loc[week, column], (week, column)
    # Target week 2 is reached by (A, 0, 2), (A, 1, 1) and (B, 1, 1): all get week 2's values.
    on_week_2 = table[table["target_week"] == 2]
    assert len(on_week_2) == 3
    assert on_week_2["has_mid_month"].tolist() == [True, True, True]
    assert on_week_2["business_days"].tolist() == [3, 3, 3]


def test_calendar_is_joined_on_the_target_week_not_the_origin_week():
    table = joined()
    row = table[(table["key"] == "TR02-A") & (table["origin_week"] == 1)
                & (table["h"] == 2)].iloc[0]
    # Target week 3: the only quarter-end week; origin week 1 is not.
    assert row["has_quarter_end"]
    assert row["business_days"] == 5 - (3 % 3)


def test_receivables_row_is_matched_per_key_origin_and_h():
    table = joined()
    for _, row in table.iterrows():
        i, o, h = KEY_INDEX[row["key"]], row["origin_week"], row["h"]
        assert row["due_in_target_week"] == 1000.0 * o + 10 * h + i
        assert row["open_invoices"] == i * 100 + o * 10 + h


def test_order_is_kept_for_a_shuffled_panel_with_a_non_default_index():
    panel = hand_panel().sample(frac=1.0, random_state=3)
    panel.index = [91, 17, 55, 3, 70, 8]
    table = join_features(panel, hand_history(), hand_calendar(), hand_owed())
    assert (list(table[["key", "origin_week", "h"]].itertuples(index=False, name=None))
            == list(panel[["key", "origin_week", "h"]].itertuples(index=False, name=None)))
    # Values still follow their row, not the old position.
    for _, row in table.iterrows():
        i, o, h = KEY_INDEX[row["key"]], row["origin_week"], row["h"]
        assert row["due_in_target_week"] == 1000.0 * o + 10 * h + i
        assert row["customer_age_weeks"] == 10 * i + o


def test_result_has_a_fresh_range_index():
    panel = hand_panel()
    panel.index = [91, 17, 55, 3, 70, 8]
    table = join_features(panel, hand_history(), hand_calendar(), hand_owed())
    pd.testing.assert_index_equal(table.index, pd.RangeIndex(len(panel)))


def test_blocks_in_a_different_row_order_give_the_same_table():
    reversed_blocks = joined(history=hand_history().iloc[::-1],
                             calendar=hand_calendar().iloc[::-1],
                             owed=hand_owed().iloc[::-1])
    pd.testing.assert_frame_equal(reversed_blocks, joined())


def test_block_indexes_do_not_matter():
    history = hand_history()
    history.index = [100, 7, 42]
    calendar = hand_calendar()
    calendar.index = list(range(50, 50 + len(calendar)))
    pd.testing.assert_frame_equal(joined(history=history, calendar=calendar), joined())


# --- dtypes --------------------------------------------------------------------------------

def test_every_column_keeps_its_dtype():
    table = joined()
    sources = {**hand_panel().dtypes.to_dict(),
               **hand_history().dtypes.drop(["key", "origin_week"]).to_dict(),
               **hand_calendar().dtypes.drop("week").to_dict(),
               **hand_owed().dtypes.drop(["key", "origin_week", "h"]).to_dict()}
    assert table.dtypes.to_dict() == sources


def test_bool_columns_stay_bool():
    table = joined()
    for column in ("has_month_end", "has_quarter_end", "has_mid_month"):
        assert table[column].dtype == np.dtype(bool), column


def test_nullable_paid_stays_boolean_with_its_na():
    table = joined()
    assert table["paid"].dtype == pd.BooleanDtype()
    assert table["paid"].isna().tolist() == hand_panel()["paid"].isna().tolist()


def test_int64_columns_stay_int64():
    table = joined()
    for column in ("origin_week", "h", "target_week", "customer_age_weeks", "observed_weeks_4",
                   "business_days", "holidays_week_before", "open_invoices"):
        assert table[column].dtype == np.dtype("int64"), column


def test_text_columns_stay_object():
    table = joined()
    for column in ("key", "company_code", "customer_id", "label_status"):
        assert table[column].dtype == np.dtype(object), column


# --- NaN and unlabelled rows ---------------------------------------------------------------

def test_nan_inside_a_history_block_survives_as_nan():
    table = joined()
    b = table[table["key"] == "TR02-B"]
    assert b["weeks_since_last_payment"].isna().all()
    assert table.loc[table["key"] == "TR02-A", "weeks_since_last_payment"].notna().all()


def test_nan_inside_a_receivables_block_survives_on_its_own_row_only():
    table = joined()
    expect_nan = (table["key"] == "TR02-A") & (table["h"] == 2)
    assert table["usual_delay_days"].isna().tolist() == expect_nan.tolist()


def test_missing_week_and_future_rows_are_kept_with_their_features():
    table = joined()
    assert table["label_status"].value_counts().to_dict() == {
        "missing_week": 3, "future": 2, "labelled": 1}
    unlabelled = table[table["label_status"] != "labelled"]
    assert unlabelled["amount"].isna().all()
    assert unlabelled["amount_deflated"].isna().all()
    assert unlabelled["paid"].isna().all()
    for column in ("customer_age_weeks", "business_days", "due_in_target_week"):
        assert unlabelled[column].notna().all(), column


# --- extra block rows, inputs, empty panel -------------------------------------------------

def test_block_rows_not_in_the_panel_are_ignored():
    table = joined(history=hand_history(extra=[("TR02-C", 0), ("TR02-A", 9)]),
                   calendar=hand_calendar(first=-3, last=20),
                   owed=hand_owed(extra=[("TR02-C", 0, 1), ("TR02-A", 0, 5)]))
    pd.testing.assert_frame_equal(table, joined())


def test_inputs_are_not_changed():
    panel = hand_panel()
    panel.index = [91, 17, 55, 3, 70, 8]
    history, calendar, owed = hand_history(), hand_calendar(), hand_owed()
    copies = [frame.copy(deep=True) for frame in (panel, history, calendar, owed)]
    join_features(panel, history, calendar, owed)
    for frame, copy in zip((panel, history, calendar, owed), copies):
        pd.testing.assert_frame_equal(frame, copy)
    assert "week" in calendar.columns and "target_week" not in calendar.columns


def test_result_is_a_new_frame():
    panel = hand_panel()
    table = join_features(panel, hand_history(), hand_calendar(), hand_owed())
    assert table is not panel
    table.loc[0, "amount"] = -1.0
    assert panel.loc[0, "amount"] == 50.0


def test_empty_panel_gives_an_empty_table_with_every_column_and_dtype():
    panel = hand_panel().iloc[0:0]
    table = join_features(panel, hand_history(), hand_calendar(), hand_owed())
    assert len(table) == 0
    assert list(table.columns) == list(joined().columns)
    assert table.dtypes.to_dict() == joined().dtypes.to_dict()
    pd.testing.assert_index_equal(table.index, pd.RangeIndex(0))


def test_empty_panel_and_empty_blocks_give_an_empty_table():
    table = join_features(hand_panel().iloc[0:0], hand_history().iloc[0:0],
                          hand_calendar().iloc[0:0], hand_owed().iloc[0:0])
    assert len(table) == 0
    assert list(table.columns) == list(joined().columns)


# --- guard: shared column names ------------------------------------------------------------

def test_history_column_named_like_a_panel_column_raises():
    history = hand_history().assign(amount=1.0)
    with pytest.raises(ValueError) as error:
        joined(history=history)
    assert str(error.value) == "the history features and the panel both have a column 'amount'"


def test_history_column_h_counts_as_shared_with_the_panel():
    # h is a join key of the receivables only: in the history block it is a feature column.
    history = hand_history().assign(h=1)
    with pytest.raises(ValueError) as error:
        joined(history=history)
    assert str(error.value) == "the history features and the panel both have a column 'h'"


def test_calendar_key_column_counts_as_shared_with_the_panel():
    calendar = hand_calendar().assign(key="TR02-A")
    with pytest.raises(ValueError) as error:
        joined(calendar=calendar)
    assert str(error.value) == "the calendar features and the panel both have a column 'key'"


def test_calendar_week_is_renamed_so_it_is_not_reported_as_shared():
    # The panel has target_week; the calendar's week becomes target_week, a join key.
    assert "week" in hand_calendar().columns
    joined()  # does not raise


def test_receivables_column_named_like_a_panel_column_raises():
    owed = hand_owed().assign(paid=True)
    with pytest.raises(ValueError) as error:
        joined(owed=owed)
    assert str(error.value) == "the receivables features and the panel both have a column 'paid'"


def test_receivables_column_named_like_a_history_column_raises():
    owed = hand_owed().assign(mean_receipts_4=0.0)
    with pytest.raises(ValueError) as error:
        joined(owed=owed)
    assert str(error.value) == ("the receivables features and the history features both have "
                                "a column 'mean_receipts_4'")


def test_receivables_column_named_like_a_calendar_column_raises():
    owed = hand_owed().assign(business_days=5)
    with pytest.raises(ValueError) as error:
        joined(owed=owed)
    assert str(error.value) == ("the receivables features and the calendar features both have "
                                "a column 'business_days'")


def test_calendar_column_named_like_a_history_column_raises():
    calendar = hand_calendar().assign(observed_weeks_4=0)
    with pytest.raises(ValueError) as error:
        joined(calendar=calendar)
    assert str(error.value) == ("the calendar features and the history features both have "
                                "a column 'observed_weeks_4'")


def test_shared_column_check_reports_the_first_block_in_order():
    # History and the calendar both share 'amount' with the panel: history is named.
    with pytest.raises(ValueError) as error:
        joined(history=hand_history().assign(amount=1.0),
               calendar=hand_calendar().assign(amount=1.0))
    assert str(error.value) == "the history features and the panel both have a column 'amount'"


def test_shared_column_check_runs_before_any_duplicate_or_match_check():
    # The history repeats a key AND misses a panel row, the receivables share a column: the
    # shared column (checked for all blocks up front) is reported.
    history = pd.concat([hand_history(), hand_history().iloc[[0]]], ignore_index=True)
    history = history[~((history["key"] == "TR02-B"))]
    with pytest.raises(ValueError) as error:
        joined(history=history, owed=hand_owed().assign(label_status="x"))
    assert str(error.value) == ("the receivables features and the panel both have a column "
                                "'label_status'")


def test_check_no_shared_columns_alone_passes_on_distinct_columns():
    blocks = [("history", hand_history(), ["key", "origin_week"]),
              ("calendar", hand_calendar().rename(columns={"week": "target_week"}),
               ["target_week"]),
              ("receivables", hand_owed(), ["key", "origin_week", "h"])]
    assert check_no_shared_columns(hand_panel(), blocks) is None


def test_check_no_shared_columns_alone_with_a_join_key_in_the_panel_is_fine():
    block = pd.DataFrame({"key": ["A"], "origin_week": [0], "feature": [1.0]})
    check_no_shared_columns(pd.DataFrame(columns=["key", "origin_week"]),
                            [("x", block, ["key", "origin_week"])])


def test_check_no_shared_columns_alone_names_the_block_and_column():
    block = pd.DataFrame({"key": ["A"], "f": [1.0]})
    other = pd.DataFrame({"key": ["A"], "f": [2.0]})
    with pytest.raises(ValueError) as error:
        check_no_shared_columns(pd.DataFrame(columns=["key"]),
                                [("one", block, ["key"]), ("two", other, ["key"])])
    assert str(error.value) == "the two features and the one features both have a column 'f'"


def test_check_no_shared_columns_alone_does_not_change_its_inputs():
    panel = hand_panel()
    history = hand_history()
    before = (panel.copy(), history.copy())
    check_no_shared_columns(panel, [("history", history, ["key", "origin_week"])])
    pd.testing.assert_frame_equal(panel, before[0])
    pd.testing.assert_frame_equal(history, before[1])


# --- guard: a block repeats its join keys --------------------------------------------------

def test_history_repeating_a_key_and_origin_raises_with_the_count_and_first():
    history = pd.concat([hand_history(), hand_history().iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        joined(history=history)
    assert str(error.value) == ("the history features repeat ['key', 'origin_week'] in 1 rows "
                                "(first: [('TR02-A', 1)])")


def test_three_copies_of_a_row_count_as_two_repeated_rows():
    history = pd.concat([hand_history()] + [hand_history().iloc[[0]]] * 2, ignore_index=True)
    with pytest.raises(ValueError) as error:
        joined(history=history)
    assert str(error.value) == ("the history features repeat ['key', 'origin_week'] in 2 rows "
                                "(first: [('TR02-A', 0), ('TR02-A', 0)])")


def test_calendar_repeating_a_week_raises_naming_target_week():
    calendar = pd.concat([hand_calendar(), hand_calendar().iloc[[2]]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        joined(calendar=calendar)
    assert str(error.value) == ("the calendar features repeat ['target_week'] in 1 rows "
                                "(first: [(2,)])")


def test_receivables_repeating_a_key_origin_and_h_raises():
    owed = pd.concat([hand_owed(), hand_owed().iloc[[4]]], ignore_index=True)
    with pytest.raises(ValueError) as error:
        joined(owed=owed)
    assert str(error.value) == ("the receivables features repeat ['key', 'origin_week', 'h'] "
                                "in 1 rows (first: [('TR02-B', 1, 1)])")


def test_a_repeat_in_rows_the_panel_does_not_use_still_raises():
    # The check is on the whole block, not only on the rows the panel reaches.
    history = hand_history(extra=[("TR02-C", 0), ("TR02-C", 0)])
    with pytest.raises(ValueError, match=r"^the history features repeat \['key', "
                                         r"'origin_week'\] in 1 rows \(first: \[\('TR02-C', "
                                         r"0\)\]\)$"):
        joined(history=history)


def test_repeated_rows_list_at_most_five():
    history = pd.concat([hand_history()] * 4, ignore_index=True)   # 9 repeated rows
    with pytest.raises(ValueError) as error:
        joined(history=history)
    message = str(error.value)
    assert message.startswith("the history features repeat ['key', 'origin_week'] in 9 rows")
    assert message.count("('TR02-") == 5


def test_rows_differing_only_in_a_feature_still_count_as_repeated():
    history = pd.concat([hand_history(), hand_history().iloc[[0]].assign(mean_receipts_4=-1.0)],
                        ignore_index=True)
    with pytest.raises(ValueError, match="the history features repeat"):
        joined(history=history)


# --- guard: a panel row finds no block row -------------------------------------------------

def test_panel_rows_without_history_raise_with_count_and_first_rows():
    history = hand_history()
    history = history[~((history["key"] == "TR02-A") & (history["origin_week"] == 1))]
    with pytest.raises(ValueError) as error:
        joined(history=history)
    # Both h rows of (A, 1) are counted (rows), but the key is listed once.
    assert str(error.value) == ("2 panel rows have no history features (first ['key', "
                                "'origin_week']: [('TR02-A', 1)])")


def test_panel_rows_without_calendar_raise_naming_the_target_weeks():
    with pytest.raises(ValueError) as error:
        joined(calendar=hand_calendar(first=0, last=2))
    assert str(error.value) == ("2 panel rows have no calendar features (first "
                                "['target_week']: [(3,)])")


def test_panel_rows_without_receivables_raise():
    owed = hand_owed()
    owed = owed[~((owed["key"] == "TR02-A") & (owed["origin_week"] == 0) & (owed["h"] == 2))]
    with pytest.raises(ValueError) as error:
        joined(owed=owed)
    assert str(error.value) == ("1 panel rows have no receivables features (first ['key', "
                                "'origin_week', 'h']: [('TR02-A', 0, 2)])")


def test_unmatched_first_rows_follow_the_panel_order():
    panel = hand_panel().iloc[::-1]
    panel.index = [5, 9, 1, 0, 3, 2]
    history = hand_history()
    history = history[history["key"] == "TR02-A"]   # B (last in the original) is missing
    with pytest.raises(ValueError) as error:
        join_features(panel, history, hand_calendar(), hand_owed())
    assert str(error.value) == ("2 panel rows have no history features (first ['key', "
                                "'origin_week']: [('TR02-B', 1)])")


def test_unmatched_rows_count_all_rows_but_list_each_key_once():
    # 6 rows, 3 distinct (key, origin) pairs: count is rows, list is distinct, panel order.
    with pytest.raises(ValueError) as error:
        joined(history=hand_history().iloc[0:0])
    assert str(error.value) == ("6 panel rows have no history features (first ['key', "
                                "'origin_week']: [('TR02-A', 0), ('TR02-A', 1), "
                                "('TR02-B', 1)])")


def wide_panel(keys, origins, hs):
    """A panel over every (key, origin, h), in that nested order; labels left empty."""
    rows = [(k, o, h) for k in keys for o in origins for h in hs]
    panel = pd.DataFrame({
        "key": pd.Series([r[0] for r in rows], dtype=object),
        "company_code": pd.Series(["TR02"] * len(rows), dtype=object),
        "customer_id": pd.Series([r[0].split("-")[1] for r in rows], dtype=object),
        "origin_week": pd.Series([r[1] for r in rows], dtype="int64"),
        "h": pd.Series([r[2] for r in rows], dtype="int64"),
    })
    panel["target_week"] = (panel["origin_week"] + panel["h"]).astype("int64")
    panel["label_status"] = pd.Series(["future"] * len(rows), dtype=object)
    panel["amount"] = nan
    panel["amount_deflated"] = nan
    panel["paid"] = pd.array([None] * len(rows), dtype="boolean")
    return panel


def test_unmatched_list_stops_at_five_distinct_keys_and_count_stays_rows():
    # 3 keys x 3 origins = 9 distinct (key, origin), x 2 horizons = 18 rows, none matched.
    panel = wide_panel(["TR02-A", "TR02-B", "TR02-C"], [0, 1, 2], [1, 2])
    with pytest.raises(ValueError) as error:
        join_features(panel, hand_history().iloc[0:0], hand_calendar(), hand_owed(panel))
    assert str(error.value) == ("18 panel rows have no history features (first ['key', "
                                "'origin_week']: [('TR02-A', 0), ('TR02-A', 1), "
                                "('TR02-A', 2), ('TR02-B', 0), ('TR02-B', 1)])")


def test_unmatched_list_is_distinct_in_panel_order_when_rows_interleave():
    # Rows: A1h1, B1h1, A0h1, B0h1, A1h2, B1h2, A0h2, B0h2 (keys alternate, h repeats).
    # Distinct pairs in first-seen order: A1, B1, A0, B0 (not sorted, no repeats).
    panel = wide_panel(["TR02-A", "TR02-B"], [0, 1], [1, 2])
    panel = panel.sort_values(["origin_week", "h", "key"], kind="stable")
    panel = panel.iloc[[4, 5, 0, 1, 6, 7, 2, 3]]
    panel.index = [7, 3, 0, 9, 4, 1, 8, 2]
    with pytest.raises(ValueError) as error:
        join_features(panel, hand_history().iloc[0:0], hand_calendar(), hand_owed(panel))
    assert str(error.value) == ("8 panel rows have no history features (first ['key', "
                                "'origin_week']: [('TR02-A', 1), ('TR02-B', 1), "
                                "('TR02-A', 0), ('TR02-B', 0)])")


def test_unmatched_receivables_rows_are_all_distinct_so_every_row_is_listed():
    # The receivables join key is the full panel key: rows and distinct tuples coincide.
    owed = hand_owed().iloc[0:0]
    with pytest.raises(ValueError) as error:
        joined(owed=owed)
    message = str(error.value)
    assert message.startswith("6 panel rows have no receivables features")
    assert message.endswith("[('TR02-A', 0, 1), ('TR02-A', 0, 2), ('TR02-A', 1, 1), "
                            "('TR02-A', 1, 2), ('TR02-B', 1, 1)])")


def test_unmatched_calendar_lists_each_target_week_once_in_panel_order():
    # Weeks 2 and 3 missing: panel targets in order 1, 2, 2, 3, 2, 3.
    with pytest.raises(ValueError) as error:
        joined(calendar=hand_calendar(first=0, last=1))
    assert str(error.value) == ("5 panel rows have no calendar features (first "
                                "['target_week']: [(2,), (3,)])")


def test_messages_hold_plain_python_numbers_not_numpy_reprs():
    messages = []
    for overrides in ({"calendar": hand_calendar(first=0, last=2)},
                      {"calendar": pd.concat([hand_calendar(), hand_calendar().iloc[[2]]])},
                      {"owed": hand_owed().iloc[1:]}):
        with pytest.raises(ValueError) as error:
            joined(**overrides)
        messages.append(str(error.value))
    for message in messages:
        assert "np." not in message and "int64" not in message, message


def test_a_key_of_another_dtype_is_not_a_match():
    # origin_week as text in the history block: the merge cannot match it with int64 and
    # pandas refuses before join_features' own check (no silent NaN either way).
    history = hand_history().astype({"origin_week": object})
    history["origin_week"] = history["origin_week"].astype(str)
    with pytest.raises(ValueError):
        joined(history=history)


# --- guard order across blocks -------------------------------------------------------------

def test_history_repeat_is_reported_before_its_missing_rows():
    history = hand_history()
    history = pd.concat([history[history["key"] == "TR02-A"], history.iloc[[0]]],
                        ignore_index=True)   # B missing AND (A, 0) repeated
    with pytest.raises(ValueError, match="^the history features repeat"):
        joined(history=history)


def test_history_missing_rows_are_reported_before_a_calendar_repeat():
    calendar = pd.concat([hand_calendar(), hand_calendar().iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="^6 panel rows have no history features"):
        joined(history=hand_history().iloc[0:0], calendar=calendar)


def test_calendar_repeat_is_reported_before_a_receivables_repeat():
    calendar = pd.concat([hand_calendar(), hand_calendar().iloc[[1]]], ignore_index=True)
    owed = pd.concat([hand_owed(), hand_owed().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="^the calendar features repeat"):
        joined(calendar=calendar, owed=owed)


def test_calendar_missing_rows_are_reported_before_receivables_missing_rows():
    with pytest.raises(ValueError, match="^2 panel rows have no calendar features"):
        joined(calendar=hand_calendar(first=0, last=2), owed=hand_owed().iloc[0:0])


def test_receivables_repeat_is_reported_before_receivables_missing_rows():
    owed = pd.concat([hand_owed().iloc[1:], hand_owed().iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="^the receivables features repeat"):
        joined(owed=owed)


# --- the real chain on small in-memory data ------------------------------------------------
# Week zero is Monday 2025-06-02; weeks 0 ... 12 have daily files (every weekday to Friday
# 2025-08-29) except Wednesday 2025-06-18, so week 2 is missing. T = 6 ends Sunday 2025-07-20.

WEEK_ZERO = date(2025, 6, 2)
LAST_FILE_DAY = date(2025, 8, 29)
MISSING_DAY = date(2025, 6, 18)
AS_OF = date(2025, 9, 30)
HORIZONS = 3
WINDOWS = (4, 13)
AGE_EDGES = (30, 90)
MIN_CLEARED = 1
T = 6
END_OF_T = WEEK_ZERO + timedelta(days=7 * T + 6)       # 2025-07-20


def file_days(missing=(MISSING_DAY,)):
    days, day = [], WEEK_ZERO
    while day <= LAST_FILE_DAY:
        if day.weekday() < 5 and day not in missing:
            days.append(day)
        day += timedelta(days=1)
    return days


def collections_table(rows):
    """rows: (customer, day text, amount_try, amount_original, currency, sap document no)."""
    return pd.DataFrame({
        "company_code": pd.Series(["TR02"] * len(rows), dtype=object),
        "customer": pd.Series([r[0] for r in rows], dtype=object),
        "customer_id": pd.Series([r[0] for r in rows], dtype=object),
        "key": pd.Series([f"TR02-{r[0]}" for r in rows], dtype=object),
        "document_date": pd.to_datetime(pd.Series([r[1] for r in rows], dtype=object)),
        "amount_try": pd.Series([r[2] for r in rows], dtype="float64"),
        "amount_original": pd.Series([r[3] for r in rows], dtype="float64"),
        "currency_original": pd.Series([r[4] for r in rows], dtype=object),
        "sap_document_no": pd.Series([r[5] for r in rows], dtype=object),
        "source_row": pd.Series(range(3, 3 + len(rows)), dtype="int64"),
    })


def ledger_table(rows):
    """rows: dicts with customer, no, role, posted, amount, and optionally due, ref,
    cleared, clearing, currency, local (the TRY amount; defaults to amount)."""
    def ts(value):
        return pd.Timestamp(value) if value is not None else pd.NaT
    table = pd.DataFrame({
        "company_code": pd.Series(["TR02"] * len(rows), dtype=object),
        "customer_id": pd.Series([r["customer"] for r in rows], dtype=object),
        "key": pd.Series([f"TR02-{r['customer']}" for r in rows], dtype=object),
        "document_no": pd.Series([r["no"] for r in rows], dtype=object),
        "document_role": pd.Series([r["role"] for r in rows], dtype=object),
        "invoice_reference": pd.Series([r.get("ref") for r in rows], dtype=object),
        "posting_date": pd.Series([ts(r["posted"]) for r in rows], dtype="datetime64[ns]"),
        "net_due_date": pd.Series([ts(r.get("due")) for r in rows], dtype="datetime64[ns]"),
        "clearing_date": pd.Series([ts(r.get("cleared")) for r in rows],
                                   dtype="datetime64[ns]"),
        "clearing_document": pd.Series([r.get("clearing") for r in rows], dtype=object),
        "amount_document": pd.Series([float(r["amount"]) for r in rows], dtype="float64"),
        "document_currency": pd.Series([r.get("currency", "TRY") for r in rows], dtype=object),
        "amount_local": pd.Series([float(r.get("local", r["amount"])) for r in rows],
                                  dtype="float64"),
        "source_row": pd.Series(range(2, 2 + len(rows)), dtype="int64"),
    })
    return table


def inv(customer, no, posted, amount, due, **kwargs):
    return {"customer": customer, "no": no, "role": "invoice", "posted": posted,
            "amount": amount, "due": due, **kwargs}


def pay(customer, no, posted, amount, **kwargs):
    """`amount` as in the ledger: negative for money received."""
    return {"customer": customer, "no": no, "role": "payment", "posted": posted,
            "amount": amount, **kwargs}


# The lines up to the end of week T, the same in both versions.
LEDGER_UP_TO_T = [
    # A: an invoice and payment before the first daily file (early history, week -2).
    inv("A", "A-INV0", "2025-04-01", 1000, "2025-05-01", cleared="2025-05-20", clearing="K0"),
    pay("A", "A-PAY0", "2025-05-20", -1000, ref="A-INV0", cleared="2025-05-20", clearing="K0"),
    inv("A", "A-INV1", "2025-06-03", 2000, "2025-07-03", cleared="2025-07-08", clearing="K1"),
    pay("A", "A-PAY1", "2025-07-08", -2000, ref="A-INV1", cleared="2025-07-08", clearing="K1"),
    # B: a EUR invoice (rate 38), partly paid in week 6 at rate 39; an overdue TRY invoice;
    # unapplied cash in week 3.
    pay("B", "B-UNAP", "2025-06-24", -300),
    inv("B", "B-INV2", "2025-06-20", 500, "2025-06-25"),
]
DAILY_UP_TO_T = [
    ("A", "2025-07-08", 2000.0, 2000.0, "TRY", "A-PAY1"),
    ("B", "2025-06-24", 300.0, 300.0, "TRY", None),
    ("B", "2025-07-16", 1950.0, 50.0, "EUR", "B-PAY2"),
]


def base_data():
    """The original version: what happens after T as first recorded."""
    ledger = ledger_table(LEDGER_UP_TO_T + [
        # Posted by T, cleared after T (week 9).
        inv("A", "A-INV2", "2025-07-01", 1500, "2025-07-31", cleared="2025-08-05",
            clearing="K2"),
        pay("A", "A-PAY2", "2025-08-05", -1500, ref="A-INV2", cleared="2025-08-05",
            clearing="K2"),
        inv("A", "A-INV3", "2025-07-29", 600, "2025-08-28"),
        inv("B", "B-INV1", "2025-06-10", 100, "2025-07-10", currency="EUR", local=3800,
            cleared="2025-08-13", clearing="K3"),
        pay("B", "B-PAY2", "2025-07-16", -50, ref="B-INV1", currency="EUR", local=-1950,
            cleared="2025-08-13", clearing="K3"),
        pay("B", "B-PAY3", "2025-08-13", -50, ref="B-INV1", currency="EUR", local=-2000,
            cleared="2025-08-13", clearing="K3"),
        # D: first seen after T.
        inv("D", "D-INV1", "2025-08-04", 700, "2025-08-20", cleared="2025-08-08",
            clearing="K4"),
        pay("D", "D-PAY1", "2025-08-08", -700, ref="D-INV1", cleared="2025-08-08",
            clearing="K4"),
    ])
    collections = collections_table(DAILY_UP_TO_T + [
        ("A", "2025-08-05", 1500.0, 1500.0, "TRY", "A-PAY2"),
        ("B", "2025-08-13", 2000.0, 50.0, "EUR", "B-PAY3"),
        ("D", "2025-08-08", 700.0, 700.0, "TRY", "D-PAY1"),
    ])
    return collections, ledger


def altered_data():
    """Everything after T changed: amounts, dates, clearings and rates moved; lines added
    (a new customer C, an unapplied payment, a refund) and removed (customer D)."""
    ledger = ledger_table(LEDGER_UP_TO_T + [
        # The same invoice, now cleared later and only partly paid (no clearing).
        inv("A", "A-INV2", "2025-07-01", 1500, "2025-07-31"),
        pay("A", "A-PAY2", "2025-08-12", -1000, ref="A-INV2"),
        inv("A", "A-INV3", "2025-07-29", 900, "2025-08-14"),
        pay("A", "A-UNAP", "2025-08-21", -300),
        pay("A", "A-REF", "2025-08-20", 100),
        inv("B", "B-INV1", "2025-06-10", 100, "2025-07-10", currency="EUR", local=3800,
            cleared="2025-08-14", clearing="K3B"),
        pay("B", "B-PAY2", "2025-07-16", -50, ref="B-INV1", currency="EUR", local=-1950,
            cleared="2025-08-14", clearing="K3B"),
        pay("B", "B-PAY3", "2025-08-14", -50, ref="B-INV1", currency="EUR", local=-2050,
            cleared="2025-08-14", clearing="K3B"),
        # C: a new customer after T; D is gone.
        inv("C", "C-INV1", "2025-07-22", 400, "2025-08-21", cleared="2025-08-01",
            clearing="K5"),
        pay("C", "C-PAY1", "2025-08-01", -400, ref="C-INV1", cleared="2025-08-01",
            clearing="K5"),
    ])
    collections = collections_table(DAILY_UP_TO_T + [
        ("A", "2025-08-12", 1000.0, 1000.0, "TRY", "A-PAY2"),
        ("A", "2025-08-21", 300.0, 300.0, "TRY", None),
        ("A", "2025-08-20", -100.0, -100.0, "TRY", "A-REF"),
        ("B", "2025-08-14", 2050.0, 50.0, "EUR", "B-PAY3"),
        ("C", "2025-08-01", 400.0, 400.0, "TRY", "C-PAY1"),
    ])
    return collections, ledger


def build_chain(collections, ledger, days):
    """Steps 7 to 11 of `run_hurdle_pipeline.main()` with the real functions, history from
    the ledger before the first file switched on. Returns every intermediate table."""
    collections = add_week_index(collections, WEEK_ZERO)
    weeks = week_calendar(days, WEEK_ZERO, SYNTHETIC_HOLIDAYS)
    weekly = weekly_amounts(collections)
    rates = exchange_rates(collections, ledger)
    last_week = int(weeks["week"].max())
    receivables = receivables_by_week(ledger, rates, WEEK_ZERO, last_week=last_week,
                                      as_of=AS_OF, horizons=HORIZONS, min_cleared=MIN_CLEARED,
                                      leak_check_weeks=(), age_edges=AGE_EDGES)
    known = known_weeks(collections, ledger, WEEK_ZERO)
    panel = panel_rows(known, last_week, HORIZONS)
    panel = add_labels(panel, weekly, weeks)
    first_ledger_week = int(week_numbers(pd.Series([ledger["posting_date"].min()]),
                                         WEEK_ZERO).iloc[0])
    levels = published_levels(WEEK_ZERO, min(first_ledger_week, 0), last_week)
    history_start = min(first_history_week(ledger, WEEK_ZERO), 0)
    early = ledger_receipts_before(ledger, min(days), WEEK_ZERO)
    early = early[early["week"] >= history_start]
    matrix = weekly_matrix(weekly, early, known["key"], weeks, levels, history_start, last_week)
    history = history_features(matrix, known, last_week, WINDOWS)
    calendar = calendar_features(WEEK_ZERO, 0, last_week + HORIZONS, SYNTHETIC_HOLIDAYS)
    owed = receivables_features(panel, receivables, levels, HORIZONS, AGE_EDGES)
    panel = add_deflated_label(panel, levels)
    table = join_features(panel, history, calendar, owed)
    return {"panel": panel, "history": history, "calendar": calendar, "owed": owed,
            "table": table, "last_week": last_week, "known": known}


@pytest.fixture(scope="module")
def base(capsys_module_silencer):
    return build_chain(*base_data(), file_days())


@pytest.fixture(scope="module")
def altered(capsys_module_silencer):
    return build_chain(*altered_data(), file_days())


@pytest.fixture(scope="module")
def capsys_module_silencer():
    """receivables_by_week prints progress notes; they are harmless here."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# --- end to end ----------------------------------------------------------------------------

def test_chain_has_the_expected_shape(base):
    table, panel = base["table"], base["panel"]
    assert base["last_week"] == 12
    # A known from the start (origins 0..12), B from week 1 (1..12), D from week 9 (9..12);
    # 3 horizons each.
    assert len(panel) == (13 + 12 + 4) * HORIZONS
    assert len(table) == len(panel)
    features = (len(base["history"].columns) - 2 + len(base["calendar"].columns) - 1
                + len(base["owed"].columns) - 3)
    assert table.shape[1] == len(panel.columns) + features
    pd.testing.assert_index_equal(table.index, pd.RangeIndex(len(panel)))


def test_chain_panel_columns_are_the_panel(base):
    table, panel = base["table"], base["panel"]
    pd.testing.assert_frame_equal(table[list(panel.columns)], panel.reset_index(drop=True))


def test_chain_every_history_feature_equals_the_block(base):
    table = base["table"]
    history = base["history"].set_index(["key", "origin_week"])
    expected = history.loc[list(zip(table["key"], table["origin_week"]))].reset_index(drop=True)
    pd.testing.assert_frame_equal(table[list(history.columns)], expected)


def test_chain_every_calendar_feature_equals_the_block(base):
    table = base["table"]
    calendar = base["calendar"].set_index("week")
    expected = calendar.loc[table["target_week"].tolist()].reset_index(drop=True)
    pd.testing.assert_frame_equal(table[list(calendar.columns)], expected)


def test_chain_every_receivables_feature_equals_the_block(base):
    table = base["table"]
    owed = base["owed"].set_index(["key", "origin_week", "h"])
    expected = owed.loc[list(zip(table["key"], table["origin_week"], table["h"]))]
    pd.testing.assert_frame_equal(table[list(owed.columns)], expected.reset_index(drop=True))


def test_chain_keeps_the_missing_and_future_rows(base):
    status = base["table"]["label_status"].value_counts().to_dict()
    assert status["missing_week"] > 0      # target week 2
    assert status["future"] > 0            # target weeks 13 to 15
    assert set(base["table"].loc[base["table"]["label_status"] == "missing_week",
                                 "target_week"]) == {2}


def test_chain_dtypes_follow_the_blocks(base):
    table = base["table"]
    for name, skip in (("panel", []), ("history", ["key", "origin_week"]),
                       ("calendar", ["week"]), ("owed", ["key", "origin_week", "h"])):
        for column, dtype in base[name].dtypes.items():
            if column in skip:
                continue
            assert table[column].dtype == dtype, (name, column, table[column].dtype, dtype)
    assert table["paid"].dtype == pd.BooleanDtype()


# --- the leak test (the plan's check 2) ----------------------------------------------------

def test_leak_check_the_two_versions_really_differ_after_t():
    # Guard against a vacuous leak test: the changes after T must reach the inputs.
    base_collections, base_ledger = base_data()
    new_collections, new_ledger = altered_data()
    up_to_t = pd.Timestamp(END_OF_T)
    pd.testing.assert_frame_equal(
        base_collections[base_collections["document_date"] <= up_to_t].reset_index(drop=True),
        new_collections[new_collections["document_date"] <= up_to_t].reset_index(drop=True))
    assert not base_collections.equals(new_collections)
    # Before T the ledgers differ only in clearings made after T.
    assert not base_ledger.equals(new_ledger)


def test_leak_rows_up_to_t_keep_every_feature_and_identity_column(base, altered):
    old, new = base["table"], altered["table"]
    feature_columns = [c for c in old.columns if c not in LABEL_COLUMNS]
    assert list(new.columns) == list(old.columns)
    old_up = old[old["origin_week"] <= T].reset_index(drop=True)
    new_up = new[new["origin_week"] <= T].reset_index(drop=True)
    assert len(old_up) == len(new_up) > 0
    pd.testing.assert_frame_equal(old_up[feature_columns], new_up[feature_columns])


def test_leak_labels_up_to_t_unchanged_where_the_target_is_up_to_t(base, altered):
    old, new = base["table"], altered["table"]
    rows_old = old[old["target_week"] <= T].reset_index(drop=True)
    rows_new = new[new["target_week"] <= T].reset_index(drop=True)
    pd.testing.assert_frame_equal(rows_old, rows_new)


def test_leak_only_labels_of_targets_after_t_move_and_they_do(base, altered):
    old, new = base["table"], altered["table"]
    old_up = old[old["origin_week"] <= T].reset_index(drop=True)
    new_up = new[new["origin_week"] <= T].reset_index(drop=True)
    after = old_up["target_week"] > T
    assert after.any()
    differs = pd.Series(False, index=old_up.index)
    for column in LABEL_COLUMNS:
        a, b = old_up[column], new_up[column]
        same = (a == b).fillna(False).astype(bool) | (a.isna() & b.isna())
        differs |= ~same
    assert differs.any(), "the altered data did not move any label: the test proves nothing"
    assert not differs[~after].any()


def test_leak_features_after_t_do_change(base, altered):
    # The changes are visible in the features of later origins, so their absence up to T is
    # not because the features ignore the data.
    old, new = base["table"], altered["table"]
    keys = ["key", "origin_week", "h"]
    both = old[old["origin_week"] > T].merge(new[new["origin_week"] > T], on=keys,
                                             suffixes=("_old", "_new"))
    assert len(both) > 0
    changed = []
    for column in ("open_balance", "last_payment_amount", "due_after_target_week",
                   "unapplied_cash", "usual_delay_days"):
        a, b = both[f"{column}_old"], both[f"{column}_new"]
        if not ((a == b) | (a.isna() & b.isna())).all():
            changed.append(column)
    assert changed, "no feature of an origin after T changed"


def test_leak_each_block_is_identical_up_to_t(base, altered):
    for name in ("history", "owed"):
        old = base[name][base[name]["origin_week"] <= T].reset_index(drop=True)
        new = altered[name][altered[name]["origin_week"] <= T].reset_index(drop=True)
        pd.testing.assert_frame_equal(old, new, obj=name)
    pd.testing.assert_frame_equal(base["calendar"], altered["calendar"])


def test_leak_with_a_file_day_missing_after_t_only_labels_move():
    # A daily file after T goes missing (Wednesday 2025-08-13, week 10): week 10's labels
    # become unknown, and nothing at origins up to T may move but labels and label status.
    with_gap = file_days(missing=(MISSING_DAY, date(2025, 8, 13)))
    collections, ledger = base_data()
    collections = collections[collections["document_date"] != pd.Timestamp("2025-08-13")]
    ledger = ledger.copy()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        old = build_chain(*base_data(), file_days())["table"]
        new = build_chain(collections, ledger, with_gap)["table"]
    allowed = LABEL_COLUMNS + ["label_status"]
    keep = [c for c in old.columns if c not in allowed]
    old_up = old[old["origin_week"] <= T].reset_index(drop=True)
    new_up = new[new["origin_week"] <= T].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_up[keep], new_up[keep])
    assert (new.loc[new["target_week"] == 10, "label_status"] == "missing_week").all()



# --- block_feature_columns, rows_known_at, rows_at_origin (U10.1) --------------------------

def test_block_feature_columns_of_the_hand_blocks():
    columns = block_feature_columns(hand_history(), hand_calendar(), hand_owed())
    assert columns == {"history": HISTORY_FEATURES, "calendar": CALENDAR_FEATURES,
                       "receivables": OWED_FEATURES}


def test_block_feature_columns_keys_are_the_three_blocks_in_order():
    columns = block_feature_columns(hand_history(), hand_calendar(), hand_owed())
    assert list(columns) == ["history", "calendar", "receivables"]
    assert all(type(value) is list for value in columns.values())


def test_block_feature_columns_keep_each_blocks_column_order():
    history = hand_history()[["observed_weeks_4", "key", "mean_receipts_4", "origin_week",
                              "customer_age_weeks", "weeks_since_last_payment"]]
    calendar = hand_calendar()[["has_mid_month", "week", "business_days", "has_month_end",
                                "has_quarter_end", "holidays_week_before"]]
    owed = hand_owed()[["usual_delay_days", "h", "key", "open_invoices", "origin_week",
                        "due_in_target_week"]]
    columns = block_feature_columns(history, calendar, owed)
    assert columns["history"] == ["observed_weeks_4", "mean_receipts_4", "customer_age_weeks",
                                  "weeks_since_last_payment"]
    assert columns["calendar"] == ["has_mid_month", "business_days", "has_month_end",
                                   "has_quarter_end", "holidays_week_before"]
    assert columns["receivables"] == ["usual_delay_days", "open_invoices", "due_in_target_week"]


def test_block_feature_columns_drop_only_each_blocks_own_join_keys():
    # history's join keys are key and origin_week; an "h" or "week" column in it is a feature
    # by this function (join_features would refuse "h" as shared with the panel).
    history = hand_history().assign(week=1, h=2)
    calendar = hand_calendar().assign(key="x", origin_week=0)
    owed = hand_owed().assign(week=3)
    columns = block_feature_columns(history, calendar, owed)
    assert columns["history"] == HISTORY_FEATURES + ["week", "h"]
    assert columns["calendar"] == CALENDAR_FEATURES + ["key", "origin_week"]
    assert columns["receivables"] == OWED_FEATURES + ["week"]


def test_block_feature_columns_of_blocks_with_only_join_keys_are_empty():
    columns = block_feature_columns(hand_history()[["key", "origin_week"]],
                                    hand_calendar()[["week"]],
                                    hand_owed()[["key", "origin_week", "h"]])
    assert columns == {"history": [], "calendar": [], "receivables": []}


def test_block_feature_columns_does_not_change_its_inputs():
    blocks = (hand_history(), hand_calendar(), hand_owed())
    copies = [block.copy(deep=True) for block in blocks]
    block_feature_columns(*blocks)
    for block, copy in zip(blocks, copies):
        pd.testing.assert_frame_equal(block, copy)


def test_block_feature_columns_with_the_join_are_exactly_the_tables_extra_columns():
    columns = block_feature_columns(hand_history(), hand_calendar(), hand_owed())
    table = joined()
    extra = [c for c in table.columns if c not in hand_panel().columns]
    assert extra == columns["history"] + columns["calendar"] + columns["receivables"]


# A grid table: one key, origins 30..45, h = 1..5, the last week 42 and a missing week 38.
# label_status follows the panel's rules: future if target > 42, missing_week if target is
# 38, labelled otherwise. Every row carries a feature f naming it (100 * origin + h).
GRID_LAST_WEEK = 42
GRID_MISSING_WEEK = 38


def grid_table(index=None):
    rows = [(t, h) for t in range(30, 46) for h in range(1, 6)]
    table = pd.DataFrame({
        "key": pd.Series(["TR02-A"] * len(rows), dtype=object),
        "company_code": pd.Series(["TR02"] * len(rows), dtype=object),
        "origin_week": pd.Series([t for t, _ in rows], dtype="int64"),
        "h": pd.Series([h for _, h in rows], dtype="int64"),
    })
    table["target_week"] = (table["origin_week"] + table["h"]).astype("int64")
    status = []
    for target in table["target_week"]:
        if target > GRID_LAST_WEEK:
            status.append("future")
        elif target == GRID_MISSING_WEEK:
            status.append("missing_week")
        else:
            status.append("labelled")
    table["label_status"] = pd.Series(status, dtype=object)
    labelled = table["label_status"] == "labelled"
    table["amount"] = np.where(labelled, 10.0 * table["origin_week"], np.nan)
    table["paid"] = pd.array([True if s == "labelled" else None for s in status],
                             dtype="boolean")
    table["f"] = (100 * table["origin_week"] + table["h"]).astype("float64")
    if index is not None:
        table.index = index
    return table


def pairs(rows):
    return set(zip(rows["origin_week"].tolist(), rows["h"].tolist()))


def test_rows_known_at_t36_h5_is_out_and_t35_h5_is_in():
    rows = rows_known_at(grid_table(), 40)
    assert (35, 5) in pairs(rows)        # target 40 = T: its week has ended
    assert (36, 5) not in pairs(rows)    # target 41 > T
    assert (36, 4) in pairs(rows)
    assert (39, 1) in pairs(rows)
    assert (40, 1) not in pairs(rows)    # target 41


def test_rows_known_at_is_exactly_the_labelled_rows_with_target_at_most_t():
    rows = rows_known_at(grid_table(), 40)
    expected = {(t, h) for t in range(30, 46) for h in range(1, 6)
                if t + h <= 40 and t + h != GRID_MISSING_WEEK}
    assert pairs(rows) == expected
    assert (rows["target_week"] <= 40).all()
    assert (rows["label_status"] == "labelled").all()


@pytest.mark.parametrize("origin", [30, 31, 35, 38, 39, 40, 41, 42, 45, 60])
def test_rows_known_at_follows_t_plus_h_at_most_t_for_every_origin(origin):
    rows = rows_known_at(grid_table(), origin)
    expected = {(t, h) for t in range(30, 46) for h in range(1, 6)
                if t + h <= min(origin, GRID_LAST_WEEK) and t + h != GRID_MISSING_WEEK}
    assert pairs(rows) == expected


def test_rows_known_at_never_takes_missing_week_or_future_rows():
    for origin in (35, 40, 42, 50, 100):
        rows = rows_known_at(grid_table(), origin)
        assert not rows["label_status"].isin(["missing_week", "future"]).any()
        assert GRID_MISSING_WEEK not in set(rows["target_week"])
        assert rows["paid"].notna().all()
        assert rows["amount"].notna().all()


def test_rows_known_at_a_late_origin_still_excludes_future_rows():
    # T = 100: every target <= 100, but the future rows (target > last week 42) carry no label.
    rows = rows_known_at(grid_table(), 100)
    assert rows["target_week"].max() == GRID_LAST_WEEK


def test_rows_known_at_before_any_target_is_empty_with_every_column():
    table = grid_table()
    rows = rows_known_at(table, 30)     # smallest target is 31
    assert len(rows) == 0
    assert list(rows.columns) == list(table.columns)
    assert rows.dtypes.to_dict() == table.dtypes.to_dict()


def test_rows_known_at_includes_rows_whose_origin_is_before_t_but_not_t_itself_at_h1():
    rows = rows_known_at(grid_table(), 40)
    assert set(rows["origin_week"]) == set(range(30, 40))


def test_rows_known_at_keeps_the_index_and_the_order():
    index = pd.Index(range(1000, 1000 - 80, -1))
    table = grid_table(index=index)
    rows = rows_known_at(table, 40)
    expected = table[(table["label_status"] == "labelled") & (table["target_week"] <= 40)]
    pd.testing.assert_frame_equal(rows, expected)
    assert list(rows.index) == [i for i in index
                                if table.loc[i, "label_status"] == "labelled"
                                and table.loc[i, "target_week"] <= 40]


def test_rows_known_at_keeps_every_column_and_value():
    table = grid_table()
    rows = rows_known_at(table, 40)
    pd.testing.assert_frame_equal(rows, table.loc[rows.index])
    assert (rows["f"] == 100 * rows["origin_week"] + rows["h"]).all()


def test_rows_known_at_does_not_change_its_input():
    table = grid_table()
    copy = table.copy(deep=True)
    rows_known_at(table, 40)
    pd.testing.assert_frame_equal(table, copy)


def test_rows_known_at_label_status_is_compared_exactly():
    table = grid_table()
    table.loc[0, "label_status"] = "Labelled"
    rows = rows_known_at(table, 40)
    assert 0 not in rows.index


def test_rows_at_origin_has_every_h_even_if_the_labels_are_unknown():
    rows = rows_at_origin(grid_table(), 40)
    assert rows["h"].tolist() == [1, 2, 3, 4, 5]
    assert (rows["origin_week"] == 40).all()
    # Targets 41, 42 are labelled in the table, 43..45 are future: all are forecast rows.
    assert rows["label_status"].tolist() == ["labelled", "labelled", "future", "future",
                                             "future"]


def test_rows_at_origin_keeps_missing_week_rows():
    rows = rows_at_origin(grid_table(), 35)     # h = 3 reaches the missing week 38
    assert rows["h"].tolist() == [1, 2, 3, 4, 5]
    assert rows.loc[rows["h"] == 3, "label_status"].tolist() == ["missing_week"]


def test_rows_at_origin_after_the_last_week_takes_future_rows():
    rows = rows_at_origin(grid_table(), 45)
    assert len(rows) == 5
    assert (rows["label_status"] == "future").all()
    assert rows["paid"].isna().all()


def test_rows_at_origin_with_no_such_origin_is_empty_with_every_column():
    table = grid_table()
    rows = rows_at_origin(table, 99)
    assert len(rows) == 0
    assert list(rows.columns) == list(table.columns)


def test_rows_at_origin_keeps_the_index_and_values():
    index = pd.Index([f"r{i}" for i in range(80)])
    table = grid_table(index=index)
    rows = rows_at_origin(table, 40)
    pd.testing.assert_frame_equal(rows, table[table["origin_week"] == 40])
    assert list(rows.index) == ["r50", "r51", "r52", "r53", "r54"]


def test_rows_at_origin_does_not_change_its_input():
    table = grid_table()
    copy = table.copy(deep=True)
    rows_at_origin(table, 40)
    pd.testing.assert_frame_equal(table, copy)


def test_rows_at_origin_and_rows_known_at_never_share_a_row():
    # A training row's target has ended by T; a forecast row's target is after T.
    table = grid_table()
    for origin in range(30, 46):
        known = rows_known_at(table, origin)
        forecast = rows_at_origin(table, origin)
        assert not set(known.index) & set(forecast.index), origin
        assert (forecast["target_week"] > origin).all()


def test_rows_at_origin_takes_every_key_at_that_origin():
    table = pd.concat([grid_table(), grid_table().assign(key="TR04-B", company_code="TR04")],
                      ignore_index=True)
    rows = rows_at_origin(table, 40)
    assert len(rows) == 10
    assert sorted(set(rows["key"])) == ["TR02-A", "TR04-B"]


# --- end to end, on the chain of the leak test ---------------------------------------------

def test_chain_block_feature_columns_are_the_tables_feature_columns(base):
    columns = block_feature_columns(base["history"], base["calendar"], base["owed"])
    table = base["table"]
    features = columns["history"] + columns["calendar"] + columns["receivables"]
    assert features == [c for c in table.columns if c not in base["panel"].columns]
    for name in ("key", "origin_week", "h", "target_week", "week"):
        assert name not in features


def test_chain_rows_known_at_t(base):
    table = base["table"]
    rows = rows_known_at(table, T)
    assert len(rows) > 0
    assert (rows["target_week"] <= T).all()
    assert (rows["label_status"] == "labelled").all()
    assert rows["paid"].notna().all()
    expected = table[(table["label_status"] == "labelled") & (table["target_week"] <= T)]
    pd.testing.assert_frame_equal(rows, expected)


def test_chain_rows_known_at_t_do_not_move_when_the_data_after_t_changes(base, altered):
    # The leak test for the training rows: changing every receipt and ledger line after T
    # leaves the rows a model fitted at T learns from exactly as they were.
    old = rows_known_at(base["table"], T).reset_index(drop=True)
    new = rows_known_at(altered["table"], T).reset_index(drop=True)
    pd.testing.assert_frame_equal(old, new)


def test_chain_rows_at_origin_has_every_known_key_times_every_h(base):
    rows = rows_at_origin(base["table"], T)
    known = base["known"]
    keys = set(known.loc[known["known_week"] <= T, "key"])
    assert set(rows["key"]) == keys
    assert len(rows) == len(keys) * HORIZONS
    assert sorted(rows.groupby("key")["h"].apply(sorted).tolist()) == [
        list(range(1, HORIZONS + 1))] * len(keys)


def test_chain_rows_at_origin_features_do_not_move_when_the_data_after_t_changes(base, altered):
    old = rows_at_origin(base["table"], T).reset_index(drop=True)
    new = rows_at_origin(altered["table"], T).reset_index(drop=True)
    allowed = LABEL_COLUMNS + ["label_status"]
    keep = [c for c in old.columns if c not in allowed]
    pd.testing.assert_frame_equal(old[keep], new[keep])
