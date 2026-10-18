"""Each customer's payment history up to each forecast week: the history features.

They are built from a weekly matrix of the customers' deflated gross receipts (one row per
key, one column per week). In it 0.0 means a week with data in which the key paid nothing,
and NaN a week with no data at all.
"""
from datetime import date, timedelta

import pandas as pd

from .series import week_numbers

# The columns of the table `ledger_receipts_before` returns, in order.
EARLY_RECEIPT_COLUMNS = ["key", "week", "gross_receipts"]


def ledger_receipts_before(ledger: pd.DataFrame, first_file_day: date,
                           week_zero: date) -> pd.DataFrame:
    """Per key and week, the gross receipts of the ledger's payments posted before
    `first_file_day` (the first daily file), for use as payment history.

    As for the daily files: each payment line's TRY amount with the sign flipped (the ledger
    records money received as negative), and only the positive ones summed; a refund paid out
    counts as nothing. Weeks are numbered from `week_zero`, so they are negative here.
    """
    is_early_payment = ((ledger["document_role"] == "payment")
                        & (ledger["posting_date"] < pd.Timestamp(first_file_day)))
    payments = ledger[is_early_payment]
    lines = pd.DataFrame({
        "key": payments["key"],
        "week": week_numbers(payments["posting_date"], week_zero),
        "gross_receipts": (-payments["amount_local"]).clip(lower=0),
    })
    receipts = lines.groupby(["key", "week"], as_index=False)["gross_receipts"].sum()
    receipts["key"] = receipts["key"].astype(object)
    receipts["week"] = receipts["week"].astype("int64")
    receipts["gross_receipts"] = receipts["gross_receipts"].astype("float64")
    return receipts[EARLY_RECEIPT_COLUMNS]


def first_history_week(ledger: pd.DataFrame, week_zero: date) -> int:
    """The first week fully covered by the ledger: the week of the first Monday on or after
    its first posting date.

    The week of the first posting itself may be only partly in the extract (the days before
    the first line may be missing rather than empty), so it is not used as history.
    """
    if ledger.empty:
        raise ValueError("the ledger has no lines, so it gives no payment history")
    first_posting = ledger["posting_date"].min().date()
    days_to_monday = (7 - first_posting.weekday()) % 7
    first_monday = first_posting + timedelta(days=days_to_monday)
    return int(week_numbers(pd.Series([pd.Timestamp(first_monday)]), week_zero).iloc[0])


# The weekly matrix must hold the same nominal TRY as its sources within this.
MATRIX_TOLERANCE = 0.01


def weekly_matrix(weekly: pd.DataFrame, early_receipts: pd.DataFrame, keys: pd.Series,
                  weeks: pd.DataFrame, levels: pd.Series, first_week: int,
                  last_week: int) -> pd.DataFrame:
    """The customers' DEFLATED gross receipts: one row per key in `keys`, one column per week
    from `first_week` to `last_week` (column labels are the week numbers).

    Sources: `weekly` (from `weekly_amounts`) for weeks 0 and later, `early_receipts` (from
    `ledger_receipts_before`) for earlier weeks. A cell is 0.0 for a week with data in which
    the key paid nothing, and NaN for a week with no data, i.e. a week the calendar `weeks`
    marks missing. (Without early receipts, call it with `first_week` 0: the matrix then has
    no weeks before 0 at all.) Amounts are divided by their week's price level (`levels`
    from `published_levels`), so they read as TRY of the last week.

    Raises ValueError if an early receipt is dated week 0 or later (the two sources must not
    overlap), if a receipt falls outside `first_week` ... `last_week`, if the matrix does
    not hold the same nominal TRY as its sources (before deflating), or if a week has no
    price level.
    """
    late = sorted(set(early_receipts.loc[early_receipts["week"] >= 0, "week"]))
    if late:
        raise ValueError(f"early receipts must be before week 0, but some are in weeks "
                         f"{late[:10]}: they would overlap the weekly amounts")
    key_list = sorted(set(keys))
    columns = list(range(first_week, last_week + 1))
    receipts = pd.concat([weekly[["key", "week", "gross_receipts"]],
                          early_receipts[["key", "week", "gross_receipts"]]], ignore_index=True)
    receipts = receipts[receipts["key"].isin(key_list)]
    outside = sorted(set(receipts.loc[~receipts["week"].between(first_week, last_week), "week"]))
    if outside:
        raise ValueError(f"receipts in weeks {outside[:10]} fall outside weeks {first_week} to "
                         f"{last_week}")

    nominal = receipts.pivot_table(index="key", columns="week", values="gross_receipts",
                                   aggfunc="sum")
    nominal = nominal.reindex(index=key_list, columns=columns, fill_value=0.0).fillna(0.0)

    # Weeks with no data: missing in the calendar (from week 0 on).
    missing_weeks = [week for week in weeks.loc[weeks["is_missing"], "week"] if week in columns]
    check_matrix_totals(nominal, receipts)
    nominal[missing_weeks] = float("nan")

    week_levels = pd.Series(columns, index=columns).map(levels)
    if week_levels.isna().any():
        no_level = [week for week in columns if pd.isna(week_levels[week])]
        raise ValueError(f"no price level for weeks {no_level[:10]}")
    deflated = nominal / week_levels
    deflated.index.name = "key"
    deflated.columns.name = "week"
    return deflated


def check_matrix_totals(nominal: pd.DataFrame, receipts: pd.DataFrame) -> None:
    """Raise ValueError unless the matrix holds, for every week, the same nominal TRY as the
    receipts it was built from. Run it before missing weeks are blanked, so that they are
    checked too."""
    per_week_matrix = nominal.sum(axis=0)
    per_week_source = receipts.groupby("week")["gross_receipts"].sum().reindex(
        nominal.columns, fill_value=0.0)
    differs = (per_week_matrix - per_week_source).abs() > MATRIX_TOLERANCE
    if differs.any():
        first = [int(week) for week in per_week_matrix.index[differs][:5]]
        raise ValueError(f"the weekly matrix differs from its sources in weeks {first}")


def history_features(matrix: pd.DataFrame, known: pd.DataFrame, last_week: int,
                     windows: tuple[int, ...]) -> pd.DataFrame:
    """Each key's payment history as of the end of each origin week t: one row per key and
    t, for every t from 0 to `last_week` at which the key is known (its `known_week` <= t).

    `matrix` is from `weekly_matrix`, `known` from `known_weeks`. A "paying week" is a week
    with a receipt above 0. Once per row:
      customer_age_weeks        t - known_week
      weeks_since_last_payment  t - the last paying week up to t (0 if t itself); NaN if
                                there is none in the matrix
      last_payment_amount       the receipts of that last paying week; NaN if none
      median_payment_W          the median receipts of the paying weeks among the W weeks
                                ending with t, W the longest window; NaN if none
    And for each window w in `windows`, over the w weeks ending with week t (t-w+1 ... t):
      observed_weeks_w  weeks with data (weeks before the matrix's first week have none)
      paying_weeks_w    paying weeks
      mean_receipts_w   the receipts' sum divided by observed_weeks_w; NaN if that is 0
    Only weeks up to t are used. Sorted by origin week, then key.
    """
    check_matrix_for_features(matrix, known, last_week, windows)
    observed = matrix.notna().astype("float64")   # 1.0 = the week has data
    paid = (matrix > 0).astype("float64")         # 1.0 = a receipt above 0 (NaN is not)
    receipts = matrix.fillna(0.0)                 # a week with no data adds nothing

    rows = key_origins(known, last_week)
    known_week = rows["key"].map(known.set_index("key")["known_week"])
    rows["customer_age_weeks"] = (rows["origin_week"] - known_week).astype("int64")

    # Each paying week's number and amount, carried forward to every later week.
    is_paying = matrix > 0
    week_numbers_grid = pd.DataFrame([list(matrix.columns)] * len(matrix), index=matrix.index,
                                     columns=matrix.columns, dtype="float64")
    last_paying_week = week_numbers_grid.where(is_paying).ffill(axis=1)
    last_amount = matrix.where(is_paying).ffill(axis=1)
    rows["weeks_since_last_payment"] = rows["origin_week"] - values_at(last_paying_week, rows)
    rows["last_payment_amount"] = values_at(last_amount, rows)

    longest = max(windows)
    median_payment = matrix.where(is_paying).T.rolling(longest, min_periods=1).median().T
    rows[f"median_payment_{longest}"] = values_at(median_payment, rows)

    for window in sorted(windows):
        observed_weeks = trailing_sum(observed, window)
        paying_weeks = trailing_sum(paid, window)
        mean_receipts = trailing_sum(receipts, window) / observed_weeks.where(observed_weeks > 0)
        rows[f"observed_weeks_{window}"] = values_at(observed_weeks, rows).astype("int64")
        rows[f"paying_weeks_{window}"] = values_at(paying_weeks, rows).astype("int64")
        rows[f"mean_receipts_{window}"] = values_at(mean_receipts, rows)
    return rows


def check_matrix_for_features(matrix: pd.DataFrame, known: pd.DataFrame, last_week: int,
                              windows: tuple[int, ...]) -> None:
    """Raise ValueError unless the matrix can give the history features: a row for every
    known key, consecutive week columns (the windows count columns) that include weeks 0 to
    `last_week`, and at least one window of 1 week or more. Each key must appear only once
    in `known`."""
    # A key repeated in `known` would repeat all of its feature rows.
    repeated = known.loc[known["key"].duplicated(), "key"]
    if not repeated.empty:
        raise ValueError(f"{repeated.nunique()} keys appear more than once in the known table "
                         f"(first: {sorted(repeated.unique())[:5]})")
    missing_keys = sorted(set(known["key"]) - set(matrix.index))
    if missing_keys:
        raise ValueError(f"{len(missing_keys)} known keys have no row in the weekly matrix "
                         f"(first: {missing_keys[:5]})")
    columns = list(matrix.columns)
    # Whole week numbers only (text or float labels would break the range below).
    if (not columns or not pd.api.types.is_integer_dtype(matrix.columns)
            or columns != list(range(columns[0], columns[-1] + 1))):
        raise ValueError("the weekly matrix's columns must be consecutive week numbers")
    if columns[0] > 0 or columns[-1] < last_week:
        raise ValueError(f"the weekly matrix has weeks {columns[0]} to {columns[-1]}; it must "
                         f"cover weeks 0 to {last_week}")
    if not windows or any(window < 1 for window in windows):
        raise ValueError(f"windows are {windows}; give at least one, each 1 week or more")


def key_origins(known: pd.DataFrame, last_week: int) -> pd.DataFrame:
    """Every key with every origin week t from 0 to `last_week` at which it is known
    (`known_week` <= t), as columns key and origin_week, sorted by origin week, then key."""
    origins = pd.DataFrame({"origin_week": range(last_week + 1)})
    rows = known[["key", "known_week"]].merge(origins, how="cross")
    rows = rows[rows["known_week"] <= rows["origin_week"]]
    rows = rows.sort_values(["origin_week", "key"], ignore_index=True)
    rows["key"] = rows["key"].astype(object)
    rows["origin_week"] = rows["origin_week"].astype("int64")
    return rows[["key", "origin_week"]]


def trailing_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per row, the sum over the `window` columns ending at each column (fewer at the start,
    where the earlier weeks are not in the frame)."""
    # pandas rolls down the rows, so the frame is turned on its side and back.
    return frame.T.rolling(window, min_periods=1).sum().T


def values_at(wide: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """The cells of `wide` (keys x weeks) at each row's key and origin week, in row order.
    Raises ValueError if a row's key or week is not in `wide`."""
    row_positions = wide.index.get_indexer(rows["key"])
    column_positions = wide.columns.get_indexer(rows["origin_week"])
    # get_indexer gives -1 for a label it cannot find, and numpy reads -1 as "the last one":
    # without this check a missing key or week would quietly return another cell.
    if (row_positions == -1).any():
        missing = sorted(set(rows.loc[row_positions == -1, "key"]))
        raise ValueError(f"keys {missing[:5]} are not in the table")
    if (column_positions == -1).any():
        missing = sorted(set(rows.loc[column_positions == -1, "origin_week"]))
        raise ValueError(f"weeks {missing[:5]} are not in the table")
    return pd.Series(wide.to_numpy()[row_positions, column_positions], index=rows.index)
