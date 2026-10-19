"""Each customer's receivables at the forecast week t, seen from each target week t+h.

Built from `receivables_by_week` (one row per key and week, as the ledger showed it at the
end of the week). The main step is picking, for each panel row, what falls due in ITS
target week: the same receivables row gives different features for h = 1 ... 5.
"""
import numpy as np
import pandas as pd

from .receivables_asof import due_in_week_columns, overdue_age_columns

# Per row, the due-date split must add up to the open invoices' total within this (TRY).
SPLIT_TOLERANCE = 0.01

# Amounts divided by the price level of the origin week (counts and days are not).
NON_AMOUNT_COLUMNS = ["overdue_days", "open_invoices", "cleared_invoices", "usual_delay_days"]


def receivables_features(panel_rows: pd.DataFrame, receivables: pd.DataFrame,
                         levels: pd.Series, horizons: int,
                         age_edges: tuple[int, ...]) -> pd.DataFrame:
    """One row per row of `panel_rows` (columns key, origin_week, h), in the same order:
      due_before_target_week  due in weeks t+1 ... t+h-1
      due_in_target_week      due in week t+h
      due_after_target_week   due after week t+h
      the aging buckets of the overdue amount (see `overdue_age_columns`)
      overdue_days, open_invoices, cleared_invoices, unapplied_cash, open_balance,
      usual_delay_days        as in `receivables_at`, for week t
    Amounts are DEFLATED by the price level of week t (`levels` from `published_levels`),
    so they read as TRY of week 0, like the history features.

    A key can be known (from a daily file) before the ledger shows it, e.g. a payment on a
    Friday posted on the Monday. Before its first week in `receivables` (see
    `before_first_ledger_week`), the ledger showed nothing for it, so its rows get "nothing
    owed": 0 for the amounts and counts, NaN for usual_delay_days.

    Raises ValueError if `receivables` repeats a key and week, if an h is not 1 ...
    `horizons`, if a receivables column used here is missing a value (only
    usual_delay_days may be NaN), if a panel key has no receivables row at all (the two
    tables' keys would not match), if a panel row has no receivables row in a week from its
    key's first one on, if the due-date split of a row does not add up to its total, or if
    a week has no price level or one that is not above 0.
    """
    check_receivables_for_features(panel_rows, receivables, horizons)
    # Every panel key is in the ledger (`known_weeks` checks it), so it has receivables rows
    # from its first ledger week on. A key with none means the keys do not match: without
    # this check all its rows would quietly be filled as "nothing owed".
    absent = sorted(set(panel_rows["key"]) - set(receivables["key"]))
    if absent:
        raise ValueError(f"{len(absent)} panel keys have no receivables row at all "
                         f"(first: {absent[:5]})")
    due_columns = due_in_week_columns(horizons)
    age_columns = overdue_age_columns(age_edges)
    check_no_missing_values(receivables, due_columns + ["due_later", "overdue"] + age_columns
                            + ["overdue_days", "open_invoices", "cleared_invoices",
                               "unapplied_cash", "open_balance"])

    # The receivables row of each panel row's key and origin week.
    by_origin = receivables.rename(columns={"week": "origin_week"})
    rows = panel_rows[["key", "origin_week", "h"]].merge(
        by_origin, on=["key", "origin_week"], how="left", indicator=True)
    no_receivables = (rows["_merge"] == "left_only").to_numpy()
    not_yet_in_ledger = before_first_ledger_week(panel_rows, receivables).to_numpy()
    # From a key's first week on, the receivables table has a row in every week: a missing
    # one there means a join or week bug.
    unexplained = no_receivables & ~not_yet_in_ledger
    if unexplained.any():
        first = rows.loc[unexplained, ["key", "origin_week"]].head(5)
        raise ValueError(f"{unexplained.sum()} panel rows have no receivables row in a week "
                         f"from their key's first one on "
                         f"(first: {list(first.itertuples(index=False, name=None))})")
    # Not yet in the ledger: nothing owed (usual_delay_days stays NaN).
    nothing_owed = due_columns + ["due_later", "overdue"] + age_columns + [
        "overdue_days", "open_invoices", "cleared_invoices", "unapplied_cash", "open_balance"]
    rows.loc[no_receivables, nothing_owed] = 0.0
    for count in ("open_invoices", "cleared_invoices"):
        rows[count] = rows[count].astype("int64")

    # Split the due-in-week columns at each row's own h: before, at and after t+h.
    due = rows[due_columns].to_numpy()
    steps = np.arange(1, horizons + 1)              # 1 ... horizons, one per column
    h = rows["h"].to_numpy()[:, None]               # each row's h, as a column
    features = rows[["key", "origin_week", "h"]].copy()
    features["due_before_target_week"] = (due * (steps < h)).sum(axis=1)
    features["due_in_target_week"] = (due * (steps == h)).sum(axis=1)
    features["due_after_target_week"] = (due * (steps > h)).sum(axis=1) + rows["due_later"]
    check_due_split(features, rows, due_columns)

    for column in age_columns + ["overdue_days", "open_invoices", "cleared_invoices",
                                 "unapplied_cash", "open_balance", "usual_delay_days"]:
        features[column] = rows[column]

    level = features["origin_week"].map(levels)
    if level.isna().any():
        missing = sorted(set(features.loc[level.isna(), "origin_week"]))
        raise ValueError(f"no price level for weeks {missing[:10]}")
    if (level <= 0).any():
        bad = sorted(set(features.loc[level <= 0, "origin_week"]))
        raise ValueError(f"the price level is 0 or below for weeks {bad[:10]}")
    amount_columns = [column for column in features.columns[3:]
                      if column not in NON_AMOUNT_COLUMNS]
    features[amount_columns] = features[amount_columns].div(level, axis=0)
    return features


def check_receivables_for_features(panel_rows: pd.DataFrame, receivables: pd.DataFrame,
                                   horizons: int) -> None:
    """Raise ValueError if `receivables` repeats a key and week (a panel row would then be
    repeated), or if a panel row's h is not 1 ... `horizons`."""
    repeated = receivables[receivables.duplicated(["key", "week"])]
    if not repeated.empty:
        first = list(repeated[["key", "week"]].head(5).itertuples(index=False, name=None))
        raise ValueError(f"{len(repeated)} receivables rows repeat a key and week "
                         f"(first: {first})")
    outside = sorted(set(panel_rows.loc[~panel_rows["h"].between(1, horizons), "h"]))
    if outside:
        raise ValueError(f"panel rows have h {outside[:10]}; it must be 1 to {horizons}")


def before_first_ledger_week(panel_rows: pd.DataFrame, receivables: pd.DataFrame) -> pd.Series:
    """True for the panel rows whose origin week is before their key's first week in
    `receivables`, i.e. before the ledger showed the key at all (always True for a key with
    no receivables row, which `receivables_features` refuses). Aligned with `panel_rows`."""
    first_week = receivables.groupby("key")["week"].min()
    key_first_week = panel_rows["key"].map(first_week)
    return key_first_week.isna() | (panel_rows["origin_week"] < key_first_week)


def check_no_missing_values(receivables: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if any of `columns` of `receivables` has a missing value (NaN):
    `receivables_at` fills them with zeros, so a NaN means the table came from elsewhere, and
    it would pass the due-date check unnoticed and come out as NaN features."""
    with_missing = [column for column in columns if receivables[column].isna().any()]
    if with_missing:
        raise ValueError(f"receivables columns {with_missing} have missing values; only "
                         f"usual_delay_days may")


def check_due_split(features: pd.DataFrame, rows: pd.DataFrame, due_columns: list[str]) -> None:
    """Raise ValueError unless, for every row, due before + in + after the target week +
    overdue add up to the open invoices' total (all the due-in-week columns, due later and
    overdue): picking the target week may not lose or double any amount."""
    split = (features["due_before_target_week"] + features["due_in_target_week"]
             + features["due_after_target_week"] + rows["overdue"])
    total = rows[due_columns].sum(axis=1) + rows["due_later"] + rows["overdue"]
    differs = (split - total).abs() > SPLIT_TOLERANCE
    if differs.any():
        raise ValueError(f"the due-date split differs from the open invoices' total in "
                         f"{differs.sum()} rows")
