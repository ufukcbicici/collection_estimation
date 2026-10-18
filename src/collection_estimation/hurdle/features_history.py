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
