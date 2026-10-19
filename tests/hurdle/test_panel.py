"""Tests for `hurdle/panel.py`: the panel's rows (key x origin week x horizon, no labels),
and `add_deflated_label` (the label divided by its target week's price level).

Hand-built `known` tables (and labelled panels) for the unit tests; one slow integration block against the
synthetic corpus and ledger in data/raw (skipped if they are absent; never uses data/interim).
"""
import contextlib
import io
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.panel import PANEL_ROW_COLUMNS, add_deflated_label, panel_rows

KNOWN_COLUMNS = ["key", "company_code", "customer_id", "first_seen", "known_week",
                 "first_seen_in"]


def known_table(rows):
    """rows: (company_code, customer_id, known_week). Shaped like `known_weeks`' output."""
    frame = pd.DataFrame(rows, columns=["company_code", "customer_id", "known_week"])
    frame = frame.astype({"company_code": object, "customer_id": object,
                          "known_week": "int64"})
    frame["key"] = (frame["company_code"] + "-" + frame["customer_id"]).astype(object)
    frame["first_seen"] = (pd.Timestamp("2025-06-02")
                           + pd.to_timedelta(frame["known_week"] * 7, unit="D"))
    frame["first_seen_in"] = pd.Series(["ledger"] * len(frame), dtype=object)
    frame = frame.sort_values("key", ignore_index=True)
    return frame[KNOWN_COLUMNS]


@pytest.fixture
def three_keys():
    # Known before week 0, in week 2, and in week 4.
    return known_table([("TR02", "A", -5), ("TR04", "B", 2), ("TR05", "C", 4)])


def keys_at(panel, origin):
    return sorted(panel.loc[panel["origin_week"] == origin, "key"].unique())


# --- the constant and the shape ---------------------------------------------------------

def test_panel_row_columns_constant():
    assert PANEL_ROW_COLUMNS == ["key", "company_code", "customer_id", "origin_week", "h",
                                 "target_week"]


def test_columns_are_exactly_panel_row_columns(three_keys):
    assert list(panel_rows(three_keys, 4, 2).columns) == PANEL_ROW_COLUMNS


def test_extra_known_columns_are_not_in_the_result(three_keys):
    panel = panel_rows(three_keys, 4, 2)
    for column in ("first_seen", "first_seen_in", "known_week"):
        assert column not in panel.columns


def test_dtypes(three_keys):
    panel = panel_rows(three_keys, 4, 2)
    for column in ("origin_week", "h", "target_week"):
        assert panel[column].dtype == "int64", column
    for column in ("key", "company_code", "customer_id"):
        assert panel[column].dtype == object, column


def test_index_is_zero_to_n_minus_one(three_keys):
    panel = panel_rows(three_keys, 4, 2)
    assert list(panel.index) == list(range(len(panel)))


def test_sorted_by_origin_then_key_then_h(three_keys):
    panel = panel_rows(three_keys, 4, 2)
    expected = panel.sort_values(["origin_week", "key", "h"], ignore_index=True)
    pd.testing.assert_frame_equal(panel, expected)


def test_sorted_even_when_known_is_not_sorted_by_key():
    known = known_table([("TR02", "Z", 0), ("TR02", "A", 0), ("TR02", "M", 1)])
    known = known.iloc[[2, 0, 1]].reset_index(drop=True)  # shuffle away from key order
    panel = panel_rows(known, 2, 2)
    triples = list(zip(panel["origin_week"], panel["key"], panel["h"]))
    assert triples == sorted(triples)
    assert list(panel.index) == list(range(len(panel)))


# --- row count and exact contents -------------------------------------------------------

def test_row_count_by_hand(three_keys):
    # Keys known by t = 0..4: 1, 1, 2, 2, 3 -> (1+1+2+2+3) x 2 = 18.
    assert len(panel_rows(three_keys, 4, 2)) == 18


def test_exact_rows_by_hand(three_keys):
    panel = panel_rows(three_keys, 4, 2)
    expected = []
    for t, keys in [(0, ["TR02-A"]), (1, ["TR02-A"]), (2, ["TR02-A", "TR04-B"]),
                    (3, ["TR02-A", "TR04-B"]), (4, ["TR02-A", "TR04-B", "TR05-C"])]:
        for key in keys:
            for h in (1, 2):
                expected.append((key, t, h, t + h))
    got = list(zip(panel["key"], panel["origin_week"], panel["h"], panel["target_week"]))
    assert got == expected


def test_row_count_formula_on_a_bigger_example():
    weeks = [-3, -1, 0, 0, 1, 5, 7, 9, 10, 12]
    known = known_table([("TR02", f"{i:02d}", w) for i, w in enumerate(weeks)])
    last_week, horizons = 10, 4
    expected = sum(sum(w <= t for w in weeks) for t in range(last_week + 1)) * horizons
    assert len(panel_rows(known, last_week, horizons)) == expected


def test_each_origin_has_exactly_the_keys_known_by_then(three_keys):
    panel = panel_rows(three_keys, 4, 2)
    known_week = three_keys.set_index("key")["known_week"]
    for t in range(5):
        assert keys_at(panel, t) == sorted(known_week[known_week <= t].index), t


def test_each_key_once_per_horizon_at_each_origin(three_keys):
    panel = panel_rows(three_keys, 4, 3)
    counts = panel.groupby(["origin_week", "key"])["h"].agg(list)
    assert all(hs == [1, 2, 3] for hs in counts)
    assert not panel.duplicated(["origin_week", "key", "h"]).any()


def test_target_week_is_origin_plus_h(three_keys):
    panel = panel_rows(three_keys, 4, 3)
    assert (panel["target_week"] == panel["origin_week"] + panel["h"]).all()


def test_key_known_in_week_3_starts_at_origin_3():
    known = known_table([("TR02", "A", 3), ("TR02", "B", 0)])
    panel = panel_rows(known, 6, 2)
    origins = sorted(panel.loc[panel["key"] == "TR02-A", "origin_week"].unique())
    assert origins == [3, 4, 5, 6]


def test_key_known_before_week_0_starts_at_origin_0():
    known = known_table([("TR02", "A", -40)])
    panel = panel_rows(known, 3, 1)
    assert panel["origin_week"].tolist() == [0, 1, 2, 3]


def test_key_known_in_week_0_starts_at_origin_0():
    known = known_table([("TR02", "A", 0)])
    assert panel_rows(known, 2, 1)["origin_week"].tolist() == [0, 1, 2]


def test_key_known_exactly_at_last_week_gets_only_that_origin():
    known = known_table([("TR02", "A", 4)])
    panel = panel_rows(known, 4, 2)
    assert panel["origin_week"].tolist() == [4, 4]


def test_key_known_after_last_week_has_no_rows():
    known = known_table([("TR02", "A", 0), ("TR02", "LATE", 5)])
    panel = panel_rows(known, 4, 2)
    assert "TR02-LATE" not in set(panel["key"])
    assert len(panel) == 5 * 2


def test_all_keys_known_after_last_week_gives_empty_table():
    known = known_table([("TR02", "A", 7), ("TR04", "B", 9)])
    panel = panel_rows(known, 4, 2)
    assert len(panel) == 0
    assert list(panel.columns) == PANEL_ROW_COLUMNS


def test_company_code_and_customer_id_carried():
    known = known_table([("TR02", "500", 0), ("TR04", "500", 1)])
    panel = panel_rows(known, 2, 2)
    for _, row in panel.iterrows():
        company, customer = row["key"].split("-")
        assert row["company_code"] == company
        assert row["customer_id"] == customer


def test_carried_values_come_from_known_not_from_the_key():
    # The panel takes company_code and customer_id from known's columns, not by splitting
    # the key.
    known = known_table([("TR02", "1", 0)])
    known.loc[0, "company_code"] = "XX"
    known.loc[0, "customer_id"] = "yy"
    panel = panel_rows(known, 1, 1)
    assert set(panel["company_code"]) == {"XX"}
    assert set(panel["customer_id"]) == {"yy"}


# --- forecast rows beyond the last week ------------------------------------------------

def test_target_weeks_beyond_last_week_are_kept(three_keys):
    panel = panel_rows(three_keys, 4, 2)
    assert panel["target_week"].max() == 4 + 2
    beyond = panel[panel["target_week"] > 4]
    # Origin 3, h=2 (2 keys) and origin 4, h=1 and h=2 (3 keys each).
    assert len(beyond) == 2 + 3 * 2


def test_last_origin_has_every_horizon(three_keys):
    panel = panel_rows(three_keys, 4, 5)
    last = panel[panel["origin_week"] == 4]
    assert sorted(last["target_week"].unique()) == [5, 6, 7, 8, 9]


# --- small and bad arguments -----------------------------------------------------------

def test_horizons_one(three_keys):
    panel = panel_rows(three_keys, 4, 1)
    assert len(panel) == 1 + 1 + 2 + 2 + 3
    assert set(panel["h"]) == {1}
    assert (panel["target_week"] == panel["origin_week"] + 1).all()


def test_last_week_zero(three_keys):
    panel = panel_rows(three_keys, 0, 3)
    assert panel["key"].tolist() == ["TR02-A"] * 3
    assert panel["origin_week"].tolist() == [0, 0, 0]
    assert panel["h"].tolist() == [1, 2, 3]
    assert panel["target_week"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("last_week", [-1, -10])
def test_negative_last_week_raises(three_keys, last_week):
    with pytest.raises(ValueError, match="last_week"):
        panel_rows(three_keys, last_week, 2)


@pytest.mark.parametrize("horizons", [0, -1])
def test_horizons_below_one_raises(three_keys, horizons):
    with pytest.raises(ValueError, match="horizons"):
        panel_rows(three_keys, 4, horizons)


def test_empty_known_gives_empty_table():
    panel = panel_rows(known_table([]), 4, 2)
    assert len(panel) == 0
    assert list(panel.columns) == PANEL_ROW_COLUMNS


def test_empty_known_keeps_integer_dtypes():
    panel = panel_rows(known_table([]), 4, 2)
    for column in ("origin_week", "h", "target_week"):
        assert panel[column].dtype == "int64", column


# --- purity -----------------------------------------------------------------------------

def test_input_not_modified(three_keys):
    before = three_keys.copy()
    panel_rows(three_keys, 4, 2)
    pd.testing.assert_frame_equal(three_keys, before)


def test_deterministic(three_keys):
    pd.testing.assert_frame_equal(panel_rows(three_keys, 4, 2), panel_rows(three_keys, 4, 2))


# --- repeated keys in known --------------------------------------------------------------

def repeated_message(count, keys):
    return re.escape(f"{count} keys appear more than once in the known table "
                     f"(first: {keys})")


def test_one_key_repeated_raises_naming_it():
    known = known_table([("TR02", "A", 0), ("TR02", "A", 0), ("TR04", "B", 1)])
    with pytest.raises(ValueError, match=repeated_message(1, ["TR02-A"])):
        panel_rows(known, 4, 2)


def test_one_key_three_times_counts_as_one():
    known = known_table([("TR02", "A", 0), ("TR02", "A", 0), ("TR02", "A", 0),
                         ("TR04", "B", 1)])
    with pytest.raises(ValueError, match=repeated_message(1, ["TR02-A"])):
        panel_rows(known, 4, 2)


def test_repeated_key_with_different_known_weeks_raises():
    # The guard is on the key alone, not on whole duplicate rows.
    known = known_table([("TR02", "A", 0), ("TR02", "A", 3)])
    with pytest.raises(ValueError, match=repeated_message(1, ["TR02-A"])):
        panel_rows(known, 4, 2)


def test_two_repeated_keys_counted_and_sorted():
    known = known_table([("TR05", "Z", 0), ("TR05", "Z", 0), ("TR02", "B", 1),
                         ("TR02", "B", 1), ("TR02", "B", 1), ("TR04", "U", 0)])
    known = known.iloc[::-1].reset_index(drop=True)  # repeated keys out of key order
    with pytest.raises(ValueError, match=repeated_message(2, ["TR02-B", "TR05-Z"])):
        panel_rows(known, 4, 2)


def test_many_repeated_keys_give_count_and_first_five_sorted():
    ids = ["G", "C", "F", "A", "E", "B", "D"]  # seven repeated keys
    rows = [("TR02", i, 0) for i in ids] * 2 + [("TR04", "UNIQUE", 0)]
    known = known_table(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    expected = ["TR02-A", "TR02-B", "TR02-C", "TR02-D", "TR02-E"]
    with pytest.raises(ValueError, match=repeated_message(7, expected)):
        panel_rows(known, 4, 2)


def test_exactly_five_repeated_keys_lists_all_five():
    ids = ["E", "D", "C", "B", "A"]
    known = known_table([("TR02", i, 0) for i in ids] * 2)
    expected = [f"TR02-{i}" for i in sorted(ids)]
    with pytest.raises(ValueError, match=repeated_message(5, expected)):
        panel_rows(known, 4, 2)


def test_repeated_key_message_lists_at_most_five_keys():
    known = known_table([("TR02", f"{i:02d}", 0) for i in range(12)] * 2)
    with pytest.raises(ValueError) as caught:
        panel_rows(known, 4, 2)
    message = str(caught.value)
    assert message.startswith("12 keys appear more than once in the known table")
    assert message.count("TR02-") == 5
    assert "TR02-04" in message and "TR02-05" not in message


def test_repeated_key_raises_even_if_no_row_would_be_built():
    # Every key is known after last_week, so the panel would be empty: still raises.
    known = known_table([("TR02", "LATE", 9), ("TR02", "LATE", 9), ("TR04", "B", 7)])
    with pytest.raises(ValueError, match=repeated_message(1, ["TR02-LATE"])):
        panel_rows(known, 4, 2)


def test_repeated_key_after_last_week_raises_alongside_valid_keys():
    known = known_table([("TR02", "A", 0), ("TR02", "LATE", 9), ("TR02", "LATE", 9)])
    with pytest.raises(ValueError, match=repeated_message(1, ["TR02-LATE"])):
        panel_rows(known, 4, 2)


def test_unique_keys_pass(three_keys):
    assert not three_keys["key"].duplicated().any()
    assert len(panel_rows(three_keys, 4, 2)) == 18


def test_input_not_modified_when_repeated_key_raises():
    known = known_table([("TR05", "Z", 0), ("TR02", "A", 0), ("TR02", "A", 2)])
    before = known.copy()
    with pytest.raises(ValueError, match="more than once"):
        panel_rows(known, 4, 2)
    pd.testing.assert_frame_equal(known, before)


# --- add_deflated_label: the label in TRY of week 0 -------------------------------------
# Hand-built labelled panels shaped like `add_labels`' output: the panel columns, then
# label_status (object), amount (float64, NaN unless labelled) and paid (nullable boolean).

LABELLED_COLUMNS = PANEL_ROW_COLUMNS + ["label_status", "amount", "paid"]
DEFLATED_COLUMNS = PANEL_ROW_COLUMNS + ["label_status", "amount", "amount_deflated", "paid"]


def labelled_panel(rows):
    """rows: (key, origin_week, h, label_status, amount). Shaped like `add_labels`' output;
    amount is NaN and paid NA on the rows that are not labelled."""
    records = []
    for key, origin, h, status, amount in rows:
        company, customer = key.split("-")
        amount = float(amount) if status == "labelled" else float("nan")
        records.append((key, company, customer, origin, h, origin + h, status, amount))
    frame = pd.DataFrame(records, columns=LABELLED_COLUMNS[:-1])
    frame = frame.astype({"key": object, "company_code": object, "customer_id": object,
                          "origin_week": "int64", "h": "int64", "target_week": "int64",
                          "label_status": object, "amount": "float64"})
    frame["paid"] = (frame["amount"] > 0).astype("boolean")
    frame.loc[frame["label_status"] != "labelled", "paid"] = pd.NA
    return frame


def level_series(mapping):
    """Price levels by week number, shaped like `published_levels`' output."""
    return pd.Series(mapping, dtype="float64", name="price_level")


def deflated_of(table, key, origin, h):
    hit = table[(table["key"] == key) & (table["origin_week"] == origin) & (table["h"] == h)]
    assert len(hit) == 1, (key, origin, h, len(hit))
    return hit["amount_deflated"].iloc[0]


@pytest.fixture
def mixed_panel():
    # Target weeks 39, 40, 42 labelled; 41 a missing week; 43 and 44 future.
    return labelled_panel([
        ("TR02-A", 38, 1, "labelled", 1_100.0),   # target 39
        ("TR02-A", 38, 2, "labelled", 1_000.0),   # target 40
        ("TR02-A", 39, 1, "labelled", 1_000.0),   # target 40
        ("TR04-B", 39, 2, "missing_week", 0),     # target 41
        ("TR04-B", 40, 2, "labelled", 0.0),       # target 42: paid nothing
        ("TR04-B", 42, 1, "future", 0),           # target 43
        ("TR02-A", 42, 2, "future", 0),           # target 44
    ])


@pytest.fixture
def mixed_levels():
    # A level for every labelled target week (and origin 38); none for the missing week 41
    # or the future weeks 43 and 44.
    return level_series({38: 1.0, 39: 1.1, 40: 1.25, 42: 1.5})


def test_deflated_worked_example_1000_try_at_level_1_25_is_800():
    panel = labelled_panel([("TR02-A", 39, 1, "labelled", 1_000.0)])
    result = add_deflated_label(panel, level_series({40: 1.25}))
    assert result["amount_deflated"].iloc[0] == 800.0


def test_deflated_divides_by_the_target_week_not_the_origin_week(mixed_panel, mixed_levels):
    result = add_deflated_label(mixed_panel, mixed_levels)
    # Origins 38 and 39, both targeting week 40: same amount, same level, same value.
    assert deflated_of(result, "TR02-A", 38, 2) == 800.0
    assert deflated_of(result, "TR02-A", 39, 1) == 800.0


def test_deflated_same_origin_different_h_uses_different_levels(mixed_panel, mixed_levels):
    result = add_deflated_label(mixed_panel, mixed_levels)
    assert deflated_of(result, "TR02-A", 38, 1) == 1_100.0 / 1.1
    assert deflated_of(result, "TR02-A", 38, 2) == 800.0
    # Not the origin week's level (1.0 at week 38).
    assert deflated_of(result, "TR02-A", 38, 1) != 1_100.0


def test_deflated_origin_level_is_never_looked_up():
    # A level only for the target week: origin week 39 has none, and nothing raises.
    panel = labelled_panel([("TR02-A", 39, 1, "labelled", 500.0)])
    result = add_deflated_label(panel, level_series({40: 2.0}))
    assert result["amount_deflated"].iloc[0] == 250.0


def test_deflated_zero_stays_zero(mixed_panel, mixed_levels):
    value = deflated_of(add_deflated_label(mixed_panel, mixed_levels), "TR04-B", 40, 2)
    assert value == 0.0
    assert not np.isnan(value)
    assert not np.signbit(value)


def test_deflated_missing_week_and_future_rows_nan_without_levels(mixed_panel, mixed_levels):
    # Weeks 41, 43 and 44 have no level: no error, and their rows stay NaN.
    result = add_deflated_label(mixed_panel, mixed_levels)
    assert np.isnan(deflated_of(result, "TR04-B", 39, 2))
    assert np.isnan(deflated_of(result, "TR04-B", 42, 1))
    assert np.isnan(deflated_of(result, "TR02-A", 42, 2))


def test_deflated_nan_exactly_on_the_rows_not_labelled(mixed_panel, mixed_levels):
    result = add_deflated_label(mixed_panel, mixed_levels)
    is_labelled = result["label_status"] == "labelled"
    assert result.loc[is_labelled, "amount_deflated"].notna().all()
    assert result.loc[~is_labelled, "amount_deflated"].isna().all()


def test_deflated_future_rows_far_beyond_the_levels_do_not_raise():
    panel = labelled_panel([("TR02-A", 64, 1, "labelled", 10.0),
                            ("TR02-A", 64, 5, "future", 0),
                            ("TR02-A", 9_000, 1, "future", 0)])
    result = add_deflated_label(panel, level_series({week: 1.0 for week in range(66)}))
    assert result["amount_deflated"].iloc[0] == 10.0
    assert result["amount_deflated"].iloc[1:].isna().all()


def test_deflated_rows_not_labelled_need_no_levels_at_all():
    panel = labelled_panel([("TR02-A", 3, 1, "missing_week", 0),
                            ("TR02-A", 3, 2, "future", 0)])
    result = add_deflated_label(panel, level_series({}))
    assert result["amount_deflated"].isna().all()
    assert result["amount_deflated"].dtype == "float64"


def test_deflated_missing_week_row_with_a_level_still_nan():
    # Even when the missing week does have a level, its row is not divided.
    panel = labelled_panel([("TR02-A", 3, 1, "missing_week", 0)])
    result = add_deflated_label(panel, level_series({4: 1.0}))
    assert np.isnan(result["amount_deflated"].iloc[0])


def no_level_message(weeks):
    return re.escape(f"no price level for weeks {weeks}")


def test_deflated_labelled_row_without_a_level_raises():
    panel = labelled_panel([("TR02-A", 39, 1, "labelled", 1_000.0),
                            ("TR02-A", 40, 1, "labelled", 1_000.0)])
    with pytest.raises(ValueError, match=no_level_message([41])):
        add_deflated_label(panel, level_series({40: 1.25}))


def test_deflated_labelled_zero_without_a_level_still_raises():
    # A zero needs no division, but its level must still exist: no silent pass.
    panel = labelled_panel([("TR02-A", 39, 1, "labelled", 0.0)])
    with pytest.raises(ValueError, match=no_level_message([40])):
        add_deflated_label(panel, level_series({39: 1.0}))


def test_deflated_labelled_row_with_a_nan_level_raises():
    panel = labelled_panel([("TR02-A", 39, 1, "labelled", 1_000.0)])
    with pytest.raises(ValueError, match=no_level_message([40])):
        add_deflated_label(panel, level_series({40: float("nan")}))


def test_deflated_missing_levels_listed_sorted_and_once():
    panel = labelled_panel([("TR02-A", 9, 1, "labelled", 1.0),
                            ("TR04-B", 4, 1, "labelled", 1.0),
                            ("TR04-B", 9, 1, "labelled", 1.0),
                            ("TR05-C", 6, 1, "labelled", 1.0)])
    with pytest.raises(ValueError, match=no_level_message([5, 7, 10])):
        add_deflated_label(panel, level_series({0: 1.0}))


def test_deflated_missing_levels_list_at_most_ten_weeks():
    panel = labelled_panel([("TR02-A", week, 1, "labelled", 1.0) for week in range(12)])
    with pytest.raises(ValueError, match=no_level_message(list(range(1, 11))) + "$"):
        add_deflated_label(panel, level_series({0: 1.0}))


def test_deflated_raise_names_only_labelled_weeks(mixed_panel):
    # Week 40 lacks a level and is labelled; 41, 43 and 44 lack one too but are not labelled.
    with pytest.raises(ValueError, match=no_level_message([40]) + "$"):
        add_deflated_label(mixed_panel, level_series({39: 1.1, 42: 1.5}))


def test_deflated_input_not_modified_when_it_raises():
    panel = labelled_panel([("TR02-A", 39, 1, "labelled", 1_000.0)])
    before = panel.copy()
    with pytest.raises(ValueError):
        add_deflated_label(panel, level_series({}))
    pd.testing.assert_frame_equal(panel, before)


def run_add_labels_message(columns):
    return re.escape(f"the panel has no columns {columns}; run add_labels first")


def test_deflated_raises_without_amount(mixed_panel, mixed_levels):
    with pytest.raises(ValueError, match=run_add_labels_message(["amount"])):
        add_deflated_label(mixed_panel.drop(columns="amount"), mixed_levels)


def test_deflated_raises_without_label_status(mixed_panel, mixed_levels):
    with pytest.raises(ValueError, match=run_add_labels_message(["label_status"])):
        add_deflated_label(mixed_panel.drop(columns="label_status"), mixed_levels)


def test_deflated_raises_without_both_listed_amount_first(mixed_panel, mixed_levels):
    with pytest.raises(ValueError, match=run_add_labels_message(["amount", "label_status"])):
        add_deflated_label(mixed_panel.drop(columns=["label_status", "amount"]), mixed_levels)


def test_deflated_raises_on_panel_rows_output(three_keys):
    # The unlabelled panel from `panel_rows` has neither column.
    with pytest.raises(ValueError, match=run_add_labels_message(["amount", "label_status"])):
        add_deflated_label(panel_rows(three_keys, 4, 2), level_series({}))


def test_deflated_missing_column_checked_before_levels(mixed_panel):
    # No levels at all: the column check must still be the error raised.
    with pytest.raises(ValueError, match="run add_labels first"):
        add_deflated_label(mixed_panel.drop(columns="amount"), level_series({}))


def test_deflated_empty_panel():
    result = add_deflated_label(labelled_panel([]), level_series({0: 1.0}))
    assert len(result) == 0
    assert list(result.columns) == DEFLATED_COLUMNS
    assert result["amount_deflated"].dtype == "float64"


def test_deflated_empty_panel_with_empty_levels():
    result = add_deflated_label(labelled_panel([]), level_series({}))
    assert len(result) == 0
    assert result["amount_deflated"].dtype == "float64"


def test_deflated_non_default_index_kept_and_rows_matched(mixed_panel, mixed_levels):
    expected = add_deflated_label(mixed_panel, mixed_levels)["amount_deflated"].to_numpy()
    index = pd.Index([70, 3, 15, 900, -2, 41, 8])
    result = add_deflated_label(mixed_panel.set_axis(index), mixed_levels)
    assert result.index.equals(index)
    np.testing.assert_array_equal(result["amount_deflated"].to_numpy(), expected)


def test_deflated_string_index_kept_and_rows_matched(mixed_panel, mixed_levels):
    expected = add_deflated_label(mixed_panel, mixed_levels)["amount_deflated"].to_numpy()
    result = add_deflated_label(mixed_panel.set_axis(list("gfedcba")), mixed_levels)
    assert list(result.index) == list("gfedcba")
    np.testing.assert_array_equal(result["amount_deflated"].to_numpy(), expected)


def test_deflated_index_with_gaps_after_filtering(mixed_panel, mixed_levels):
    result = add_deflated_label(mixed_panel.iloc[[1, 2, 4, 6]], mixed_levels)
    assert list(result.index) == [1, 2, 4, 6]
    assert result["amount_deflated"].tolist()[:3] == [800.0, 800.0, 0.0]
    assert np.isnan(result["amount_deflated"].iloc[3])


def test_deflated_repeated_index_labels_rows_matched_by_position(mixed_panel, mixed_levels):
    # pandas assigns by index label: with repeated labels a mis-alignment would mix rows up
    # or raise. Each row must still get its own value.
    expected = add_deflated_label(mixed_panel, mixed_levels)["amount_deflated"].to_numpy()
    result = add_deflated_label(mixed_panel.set_axis([0, 0, 1, 1, 1, 2, 2]), mixed_levels)
    np.testing.assert_array_equal(result["amount_deflated"].to_numpy(), expected)


def test_deflated_column_inserted_right_after_amount(mixed_panel, mixed_levels):
    result = add_deflated_label(mixed_panel, mixed_levels)
    assert list(result.columns) == DEFLATED_COLUMNS


def test_deflated_column_follows_amount_wherever_it_is(mixed_panel, mixed_levels):
    order = ["amount", "key", "label_status", "extra", "h", "origin_week", "target_week",
             "paid", "company_code", "customer_id"]
    result = add_deflated_label(mixed_panel.assign(extra=1)[order], mixed_levels)
    assert list(result.columns) == ["amount", "amount_deflated"] + order[1:]


def test_deflated_amount_last_appends_at_the_end(mixed_panel, mixed_levels):
    order = [column for column in mixed_panel.columns if column != "amount"] + ["amount"]
    result = add_deflated_label(mixed_panel[order], mixed_levels)
    assert list(result.columns) == order + ["amount_deflated"]


def test_deflated_other_columns_and_dtypes_unchanged(mixed_panel, mixed_levels):
    result = add_deflated_label(mixed_panel, mixed_levels)
    pd.testing.assert_frame_equal(result.drop(columns="amount_deflated"), mixed_panel)
    assert result["amount_deflated"].dtype == "float64"
    assert result["amount"].dtype == "float64"
    assert result["paid"].dtype == "boolean"
    assert result["label_status"].dtype == object
    for column in ("origin_week", "h", "target_week"):
        assert result[column].dtype == "int64", column


def test_deflated_paid_keeps_its_na(mixed_panel, mixed_levels):
    result = add_deflated_label(mixed_panel, mixed_levels)
    is_labelled = result["label_status"] == "labelled"
    assert result.loc[~is_labelled, "paid"].isna().all()
    assert result.loc[is_labelled, "paid"].tolist() == [True, True, True, False]


def test_deflated_float64_with_integer_levels():
    panel = labelled_panel([("TR02-A", 0, 1, "labelled", 9.0)])
    result = add_deflated_label(panel, pd.Series({1: 2}, dtype="int64"))
    assert result["amount_deflated"].dtype == "float64"
    assert result["amount_deflated"].iloc[0] == 4.5


def test_deflated_inputs_not_modified(mixed_panel, mixed_levels):
    panel_before, levels_before = mixed_panel.copy(), mixed_levels.copy()
    add_deflated_label(mixed_panel, mixed_levels)
    pd.testing.assert_frame_equal(mixed_panel, panel_before)
    pd.testing.assert_series_equal(mixed_levels, levels_before)
    assert "amount_deflated" not in mixed_panel.columns


def test_deflated_returns_a_new_table(mixed_panel, mixed_levels):
    result = add_deflated_label(mixed_panel, mixed_levels)
    assert result is not mixed_panel
    result.loc[result.index[0], "amount"] = -1.0
    assert mixed_panel["amount"].iloc[0] == 1_100.0


def test_deflated_deterministic(mixed_panel, mixed_levels):
    pd.testing.assert_frame_equal(add_deflated_label(mixed_panel, mixed_levels),
                                  add_deflated_label(mixed_panel, mixed_levels))


def test_deflated_applied_twice_raises_not_a_second_column(mixed_panel, mixed_levels):
    once = add_deflated_label(mixed_panel, mixed_levels)
    with pytest.raises(ValueError):
        add_deflated_label(once, mixed_levels)


def test_deflated_nominal_equals_deflated_times_level_for_every_labelled_row():
    rng = np.random.default_rng(7)
    rows = []
    for origin in range(20):
        for h in range(1, 6):
            target = origin + h
            status = ("future" if target > 20 else "missing_week" if target == 11
                      else "labelled")
            amount = 0.0 if rng.random() < 0.3 else float(rng.uniform(1, 5e6))
            rows.append(("TR02-A", origin, h, status, amount))
    levels = level_series({week: 1.0 + 0.03 * week + 0.001 * week ** 2 for week in range(21)})
    result = add_deflated_label(labelled_panel(rows), levels)
    labelled = result[result["label_status"] == "labelled"]
    assert len(labelled) > 50
    rebuilt = labelled["amount_deflated"] * labelled["target_week"].map(levels)
    np.testing.assert_allclose(rebuilt.to_numpy(), labelled["amount"].to_numpy(),
                               rtol=1e-14, atol=0.0)


def test_deflated_with_published_levels_week_zero_target_unchanged():
    from collection_estimation.hurdle.prices import published_levels

    levels = published_levels(WEEK_ZERO, -32, 64)
    assert levels.loc[0] == 1.0
    amount = 123_456.789
    panel = labelled_panel([("TR02-A", -1, 1, "labelled", amount),
                            ("TR02-A", 39, 1, "labelled", amount)])
    result = add_deflated_label(panel, levels)
    assert result["amount_deflated"].iloc[0] == amount
    # Week 40 is later: prices rose, so the same nominal amount is worth less in week 0.
    assert result["amount_deflated"].iloc[1] == amount / levels.loc[40]
    assert result["amount_deflated"].iloc[1] < amount


def test_deflated_with_published_levels_future_week_past_the_range_is_nan():
    from collection_estimation.hurdle.prices import published_levels

    levels = published_levels(WEEK_ZERO, 0, 64)
    panel = labelled_panel([("TR02-A", 63, 1, "labelled", 50.0),
                            ("TR02-A", 64, 5, "future", 0)])
    result = add_deflated_label(panel, levels)
    assert result["amount_deflated"].iloc[0] == 50.0 / levels.loc[64]
    assert np.isnan(result["amount_deflated"].iloc[1])


def test_deflated_on_add_labels_output():
    # End to end on a real `add_labels` table: weeks 0-4, week 3 missing, targets 5-6 future.
    from collection_estimation.hurdle.panel import add_labels
    from tests.hurdle.test_panel_labels import weekly_table, weeks_table

    known = known_table([("TR02", "A", -5), ("TR04", "B", 2)])
    panel = add_labels(panel_rows(known, 4, 2),
                       weekly_table([("TR02-A", 1, 100.0, 0.0), ("TR02-A", 3, 50.0, 0.0),
                                     ("TR04-B", 4, 30.0, -5.0)]),
                       weeks_table(4, missing={3}))
    levels = level_series({1: 1.0, 2: 1.25, 4: 2.0})   # none for weeks 3, 5 and 6
    result = add_deflated_label(panel, levels)
    labelled = result["label_status"] == "labelled"
    assert set(result.loc[~labelled, "target_week"]) == {3, 5, 6}
    assert result.loc[~labelled, "amount_deflated"].isna().all()
    expected = result.loc[labelled, "amount"] / result.loc[labelled, "target_week"].map(levels)
    pd.testing.assert_series_equal(result.loc[labelled, "amount_deflated"], expected,
                                   check_names=False)
    assert deflated_of(result, "TR04-B", 2, 2) == 15.0
    assert deflated_of(result, "TR02-A", 0, 1) == 100.0


# --- integration with the synthetic corpus and ledger (SLOW) ---------------------------

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LEDGER_FILE = RAW / "SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx"
CORPUS_FOLDER = RAW / "SYNTHETIC_ey_tahsilat_20250602_20260828"
AS_OF = date(2026, 8, 28)
WEEK_ZERO = date(2025, 6, 2)
KNOWN_COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]


@pytest.fixture(scope="module")
def synthetic():
    if not LEDGER_FILE.exists() or not CORPUS_FOLDER.is_dir():
        pytest.skip("the synthetic ledger or corpus is not in data/raw")
    from collection_estimation.hurdle.keys import add_keys
    from collection_estimation.hurdle.known import known_weeks
    from collection_estimation.hurdle.read_collections import list_daily_files, read_all_files
    from collection_estimation.hurdle.read_ledger import read_ledger
    from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_LAYOUT

    daily_files = list_daily_files(CORPUS_FOLDER, SYNTHETIC_LAYOUT.file_name_date_format)
    collections = add_keys(read_all_files(daily_files, SYNTHETIC_LAYOUT), KNOWN_COMPANIES)
    ledger = add_keys(read_ledger(LEDGER_FILE, AS_OF, SYNTHETIC_LEDGER_LAYOUT),
                      KNOWN_COMPANIES)
    known = known_weeks(collections, ledger, WEEK_ZERO)
    return collections, ledger, known, panel_rows(known, 64, 5)


@pytest.fixture(scope="module")
def synthetic_receivables(synthetic):
    from collection_estimation.hurdle.receivables_asof import (
        exchange_rates, receivables_by_week)
    collections, ledger, _, _ = synthetic
    rates = exchange_rates(collections, ledger)
    with contextlib.redirect_stdout(io.StringIO()):
        return receivables_by_week(ledger, rates, WEEK_ZERO, last_week=64, as_of=AS_OF,
                                   horizons=5, min_cleared=3, leak_check_weeks=(0, 30, 64),
                                   age_edges=(30, 90))


def test_slow_synthetic_panel_has_1159740_rows(synthetic):
    *_, panel = synthetic
    assert len(panel) == 1_159_740
    assert list(panel.columns) == PANEL_ROW_COLUMNS


def test_slow_synthetic_keys_per_origin_2062_to_4590(synthetic):
    *_, panel = synthetic
    per_origin = panel.groupby("origin_week")["key"].nunique()
    assert list(per_origin.index) == list(range(65))
    assert per_origin.loc[0] == 2_062
    assert per_origin.loc[64] == 4_590
    assert per_origin.is_monotonic_increasing
    assert per_origin.sum() * 5 == len(panel)


def test_slow_synthetic_horizons_and_target_weeks(synthetic):
    *_, panel = synthetic
    assert sorted(panel["h"].unique()) == [1, 2, 3, 4, 5]
    assert (panel["target_week"] == panel["origin_week"] + panel["h"]).all()
    assert panel["target_week"].max() == 69


def test_slow_synthetic_pairs_equal_receivables_by_week(synthetic, synthetic_receivables):
    *_, panel = synthetic
    panel_pairs = set(zip(panel["key"], panel["origin_week"]))
    receivable_pairs = set(zip(synthetic_receivables["key"], synthetic_receivables["week"]))
    assert len(panel_pairs) == 231_948
    assert panel_pairs == receivable_pairs


@pytest.mark.parametrize("week", [0, 30, 64])
def test_slow_synthetic_keys_equal_ledger_keys_posted_by_sunday(synthetic, week):
    _, ledger, _, panel = synthetic
    sunday = pd.Timestamp(WEEK_ZERO + timedelta(days=7 * week + 6))
    ledger_keys = set(ledger.loc[ledger["posting_date"] <= sunday, "key"])
    assert set(panel.loc[panel["origin_week"] == week, "key"]) == ledger_keys
