"""The training table: the panel's rows with their labels and every feature block.

Each block comes at its own level: history per (key, origin week), the calendar per week,
the receivables per panel row (key, origin week, h). Here they are joined onto the panel,
one row per panel row, in the panel's order. Choosing which rows to train on and which
columns go into the model happens later.
"""
import pandas as pd


def join_features(panel: pd.DataFrame, history: pd.DataFrame, calendar: pd.DataFrame,
                  owed: pd.DataFrame) -> pd.DataFrame:
    """The panel (from `add_deflated_label`) with the history features (`history_features`)
    joined on key and origin week, the calendar features (`calendar_features`) on the target
    week, and the receivables features (`receivables_features`) on key, origin week and h.

    One row per panel row, in the panel's order, with a fresh index. Raises ValueError if a
    feature column has the same name as another column, if a block repeats a join key (a
    panel row would be repeated), or if a panel row finds no row in a block (a missing
    feature must not become a silent NaN).
    """
    blocks = [
        ("history", history, ["key", "origin_week"]),
        ("calendar", calendar.rename(columns={"week": "target_week"}), ["target_week"]),
        ("receivables", owed, ["key", "origin_week", "h"]),
    ]
    check_no_shared_columns(panel, blocks)

    table = panel.reset_index(drop=True)
    for name, block, on in blocks:
        repeated = block[block.duplicated(on)]
        if not repeated.empty:
            first = list(repeated[on].head(5).itertuples(index=False, name=None))
            raise ValueError(f"the {name} features repeat {on} in {len(repeated)} rows "
                             f"(first: {first})")
        table = table.merge(block, on=on, how="left", validate="many_to_one", indicator=True)
        unmatched = table["_merge"] == "left_only"
        if unmatched.any():
            # Each missing key once (one (key, t) has a panel row per h), in panel order.
            missing = table.loc[unmatched, on].drop_duplicates()
            first = list(missing.head(5).itertuples(index=False, name=None))
            raise ValueError(f"{unmatched.sum()} panel rows have no {name} features "
                             f"(first {on}: {first})")
        table = table.drop(columns="_merge")
    return table


def check_no_shared_columns(panel: pd.DataFrame, blocks: list) -> None:
    """Raise ValueError if a block's feature column (any column but its join keys) has the
    same name as a panel column or another block's feature column: pandas would quietly
    rename both (to _x and _y) instead."""
    seen = {column: "the panel" for column in panel.columns}
    for name, block, on in blocks:
        for column in block.columns:
            if column in on:
                continue
            if column in seen:
                raise ValueError(f"the {name} features and {seen[column]} both have a column "
                                 f"'{column}'")
            seen[column] = f"the {name} features"


def block_feature_columns(history: pd.DataFrame, calendar: pd.DataFrame,
                          owed: pd.DataFrame) -> dict[str, list[str]]:
    """{block name: its feature columns}: every column of each block but its join keys."""
    return {
        "history": [column for column in history.columns
                    if column not in ("key", "origin_week")],
        "calendar": [column for column in calendar.columns if column != "week"],
        "receivables": [column for column in owed.columns
                        if column not in ("key", "origin_week", "h")],
    }


def rows_known_at(table: pd.DataFrame, origin: int) -> pd.DataFrame:
    """The rows a model fitted at the end of week `origin` may learn from: labelled rows whose
    target week has ended (t + h <= origin), so their answers were known by then."""
    usable = (table["label_status"] == "labelled") & (table["target_week"] <= origin)
    return table[usable]


def rows_at_origin(table: pd.DataFrame, origin: int) -> pd.DataFrame:
    """The rows a forecast made at the end of week `origin` predicts: every customer known at
    `origin`, for each h. Their labels are not known at `origin`."""
    return table[table["origin_week"] == origin]
