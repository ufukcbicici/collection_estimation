"""The panel: one row per known key, forecast week and horizon, with its label.

Standing at the end of forecast week t (the "origin"), every key known by then gets one row
for each horizon h = 1 ... H: what will it pay in week t + h (the "target week")? The panel
stacks these snapshots for every origin week, so the model learns from many past forecasts.
"""
import pandas as pd

from .prices import deflate

# The columns of the table `panel_rows` returns, in order.
PANEL_ROW_COLUMNS = ["key", "company_code", "customer_id", "origin_week", "h", "target_week"]


def panel_rows(known: pd.DataFrame, last_week: int, horizons: int) -> pd.DataFrame:
    """The panel's rows, without labels: for every origin week t from 0 to `last_week`, every
    key known by the end of t (its `known_week` <= t), once for each h = 1 ... `horizons`.

    `known` is the table from `known_weeks`. Sorted by origin week, key and h.
    """
    if last_week < 0:
        raise ValueError(f"last_week is {last_week}; it must be 0 or more")
    if horizons < 1:
        raise ValueError(f"horizons is {horizons}; it must be 1 or more")
    # One row per key in `known`: a repeated key would silently repeat all of its panel rows.
    repeated = known.loc[known["key"].duplicated(), "key"]
    if not repeated.empty:
        raise ValueError(f"{repeated.nunique()} keys appear more than once in the known table "
                         f"(first: {sorted(repeated.unique())[:5]})")

    keys = known[["key", "company_code", "customer_id", "known_week"]]
    origins = pd.DataFrame({"origin_week": range(last_week + 1)})
    # Every key with every origin week, then only the origins at which the key is known.
    key_origins = keys.merge(origins, how="cross")
    key_origins = key_origins[key_origins["known_week"] <= key_origins["origin_week"]]

    steps_ahead = pd.DataFrame({"h": range(1, horizons + 1)})
    rows = key_origins.merge(steps_ahead, how="cross")
    rows["target_week"] = rows["origin_week"] + rows["h"]

    rows = rows.sort_values(["origin_week", "key", "h"], ignore_index=True)
    for column in ("origin_week", "h", "target_week"):
        rows[column] = rows[column].astype("int64")
    return rows[PANEL_ROW_COLUMNS]


# Why a panel row has a label or not (see `add_labels`).
LABELLED = "labelled"          # the target week has data: the row is used to train and test
MISSING_WEEK = "missing_week"  # the target week is missing (even after filling): unknown
FUTURE = "future"              # the target week is after the last week: the forecast rows

# Per target week, the labelled amounts may exceed that week's receipts by no more than this.
LABEL_TOLERANCE = 0.01


def add_labels(panel: pd.DataFrame, weekly: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """The panel rows with their labels: label_status, amount and paid.

    `weekly` is the table from `weekly_amounts`, `weeks` the week calendar. For a row whose
    target week has data (label_status "labelled"), `amount` is the key's gross receipts in
    TRY in that week, 0.0 if it paid nothing, and `paid` is amount > 0. For a row whose target
    week is missing ("missing_week") or after the last week ("future"), both are unknown
    (NaN and NA): a missing week is never read as "paid nothing".

    Raises ValueError if `weekly` has a key twice in one week, if a target week up to the
    last week is not in the calendar, or if, for some origin and target week, the labelled
    amounts add up to more than the target week's receipts (a key counted twice).
    """
    repeated = weekly.duplicated(["key", "week"])
    if repeated.any():
        first = weekly.loc[repeated, ["key", "week"]].head(5).values.tolist()
        raise ValueError(f"the weekly amounts have {repeated.sum()} repeated (key, week) rows "
                         f"(first: {first})")
    last_week = weeks["week"].max()
    not_in_calendar = sorted(set(panel.loc[panel["target_week"] <= last_week, "target_week"])
                             - set(weeks["week"]))
    if not_in_calendar:
        raise ValueError(f"target weeks {not_in_calendar[:10]} are not in the week calendar")
    missing_weeks = set(weeks.loc[weeks["is_missing"], "week"])

    labelled = panel.copy()
    # object, like every text column in the package (pandas would infer its own text type).
    labelled["label_status"] = pd.Series(LABELLED, index=labelled.index, dtype=object)
    labelled.loc[labelled["target_week"].isin(missing_weeks), "label_status"] = MISSING_WEEK
    labelled.loc[labelled["target_week"] > last_week, "label_status"] = FUTURE

    receipts = weekly[["key", "week", "gross_receipts"]].rename(
        columns={"week": "target_week", "gross_receipts": "amount"})
    labelled = labelled.merge(receipts, on=["key", "target_week"], how="left")

    has_label = labelled["label_status"] == LABELLED
    # A labelled row without a weekly row: the key paid nothing that week (a real zero).
    labelled["amount"] = labelled["amount"].fillna(0.0).where(has_label, float("nan"))
    labelled["paid"] = (labelled["amount"] > 0).astype("boolean")
    labelled.loc[~has_label, "paid"] = pd.NA

    check_labels_within_receipts(labelled, weekly)
    return labelled


def check_labels_within_receipts(labelled: pd.DataFrame, weekly: pd.DataFrame) -> None:
    """Raise ValueError if, for some origin and target week, the labelled amounts exceed the
    target week's gross receipts.

    At one origin each key has one row per target week, so the labelled amounts of that
    origin and target week can add up to at most the week's receipts; what they fall short by
    is cash from keys not yet known at the origin (the new-customer allowance).
    """
    rows = labelled[labelled["label_status"] == LABELLED]
    per_origin_and_target = rows.groupby(["origin_week", "target_week"],
                                         as_index=False)["amount"].sum()
    week_receipts = weekly.groupby("week")["gross_receipts"].sum()
    per_origin_and_target["receipts"] = (per_origin_and_target["target_week"]
                                         .map(week_receipts).fillna(0.0))
    too_much = (per_origin_and_target["amount"]
                > per_origin_and_target["receipts"] + LABEL_TOLERANCE)
    if too_much.any():
        first = per_origin_and_target[too_much].head(5)
        raise ValueError(f"for {too_much.sum()} (origin, target) weeks the labelled amounts add up "
                         f"to more than the target week's receipts (first: "
                         f"{first[['origin_week', 'target_week']].values.tolist()})")


def add_deflated_label(panel: pd.DataFrame, levels: pd.Series) -> pd.DataFrame:
    """The panel with `amount_deflated` added after `amount`: the label divided by the price
    level of its TARGET week (`levels` from `published_levels`), so it reads as TRY of week
    0, like the features.

    Only labelled rows are divided: missing-week and future rows keep NaN and need no level.
    0.0 stays 0.0. `panel` (from `add_labels`) is not changed. Raises ValueError if it has no
    `amount` or `label_status` column, or (from `deflate`) if a labelled row's target week
    has no price level.
    """
    absent = [column for column in ("amount", "label_status") if column not in panel.columns]
    if absent:
        raise ValueError(f"the panel has no columns {absent}; run add_labels first")
    with_label = panel.copy()
    is_labelled = with_label["label_status"] == LABELLED
    deflated = pd.Series(float("nan"), index=with_label.index, dtype="float64")
    labelled_rows = with_label[is_labelled]
    deflated[is_labelled] = deflate(labelled_rows["amount"], labelled_rows["target_week"], levels)
    with_label.insert(with_label.columns.get_loc("amount") + 1, "amount_deflated", deflated)
    return with_label
