"""When each customer (key) becomes known: from that week on, it gets rows in the panel.

A key is known at forecast week t if its first interaction (any ledger line, or any line in
the daily files) happened in week t or earlier. Once known, it stays known: there is no exit
rule; how long a customer has been silent is a feature for the model to learn from.
"""
from datetime import date

import pandas as pd

from .series import week_numbers

# The columns of the table `known_weeks` returns, in order.
KNOWN_COLUMNS = ["key", "company_code", "customer_id", "first_seen", "known_week",
                 "first_seen_in"]


def known_weeks(collections: pd.DataFrame, ledger: pd.DataFrame,
                week_zero: date) -> pd.DataFrame:
    """One row per key: the date of its first interaction and the week it becomes known.

    The first interaction is the earlier of its first ledger line (by posting date: an
    invoice, payment or credit memo) and its first line in the daily files (by document
    date: payments, refunds and unapplied cash). `known_week` is the week number of that
    date; negative if it is before week zero. `first_seen_in` says which source saw it first
    ("ledger" if both did on the same day).

    Both tables need `key`, `company_code` and `customer_id` (from `add_keys`). Raises
    ValueError if a key in the daily files has no line in the ledger (see
    `check_daily_keys_in_ledger`).
    """
    check_daily_keys_in_ledger(collections, ledger)

    key_columns = ["key", "company_code", "customer_id"]
    from_ledger = ledger.groupby(key_columns, as_index=False)["posting_date"].min()
    from_ledger = from_ledger.rename(columns={"posting_date": "first_seen"})
    from_ledger["first_seen_in"] = "ledger"
    from_daily_files = collections.groupby(key_columns, as_index=False)["document_date"].min()
    from_daily_files = from_daily_files.rename(columns={"document_date": "first_seen"})
    from_daily_files["first_seen_in"] = "daily files"

    # For each key keep the earlier source; on the same day the ledger comes first, because
    # it is listed first and the sort below keeps that order (a stable sort).
    both = pd.concat([from_ledger, from_daily_files], ignore_index=True)
    both = both.sort_values(["key", "first_seen"], kind="stable")
    known = both.drop_duplicates("key", keep="first").reset_index(drop=True)

    known["known_week"] = week_numbers(known["first_seen"], week_zero)
    for column in key_columns + ["first_seen_in"]:
        known[column] = known[column].astype(object)
    return known[KNOWN_COLUMNS]


def check_daily_keys_in_ledger(collections: pd.DataFrame, ledger: pd.DataFrame) -> None:
    """Raise ValueError if a key paid in the daily files but has no line in the ledger.

    Every payment in the daily files should be on the customer's account in the ledger; a
    customer the ledger has never seen means the two extracts do not match (e.g. different
    company codes, customer numbers or periods).
    """
    missing = sorted(set(collections["key"]) - set(ledger["key"]))
    if missing:
        raise ValueError(f"{len(missing)} keys in the daily files have no line in the ledger "
                         f"(first: {missing[:10]})")
