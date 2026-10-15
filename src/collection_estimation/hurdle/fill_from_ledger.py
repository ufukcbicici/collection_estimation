"""Filling days without a daily file from the AR ledger's payments.

A day with no daily file leaves its week incomplete: a customer who paid that day looks as if
it paid nothing. The ledger records every payment on the customer's account, including those
days, so its payments can stand in for the lost lines. Only days between the first and the
last daily file are filled, and only if the ledger and the daily files tie on every day that
has a file. A day with no file and no ledger payment cannot be told apart from a day of no
payments, so it is not filled and its week stays missing.
"""
from datetime import date

import pandas as pd

from .read_ledger import check_ledger_against_collections

# `source_file` of a payment line taken from the ledger.
FROM_LEDGER = "ledger"

# The filled lines must add up to the ledger's payments on the missing days within this (TRY).
FILL_TOLERANCE = 0.01


def fill_missing_days(collections: pd.DataFrame, ledger: pd.DataFrame,
                      days_with_file: list[date]) -> tuple[pd.DataFrame, list[date]]:
    """The payment lines with the ledger's payments added for days without a daily file.

    Returns (payment lines, filled days). The payment lines gain a column
    `filled_from_ledger` (False for every daily-file line). A filled line is a ledger payment
    posted on a day strictly between the first and the last day with a file, on which there
    is no file: its TRY amount is the ledger's with the sign flipped (money received is
    positive), and its SAP document number is empty for unapplied cash (no invoice
    reference), so that `weekly_amounts` counts it as unapplied, as for the daily files.

    Needs `key` and `customer_id` on both tables (from `add_keys`). Raises ValueError (from
    `check_ledger_against_collections`) if the ledger and the daily files do not tie.
    """
    tie = check_ledger_against_collections(ledger, collections, days_with_file)

    days = sorted({pd.Timestamp(day).date() for day in days_with_file})
    posting_day = ledger["posting_date"].dt.date
    to_fill = ((ledger["document_role"] == "payment")
               & (posting_day > days[0]) & (posting_day < days[-1])
               & ~posting_day.isin(days))
    payments = ledger[to_fill]

    filled = payment_lines_from_ledger(payments, collections.columns)
    filled_total = filled["amount_try"].sum()
    if abs(filled_total - tie["try_on_missing_file_days"]) > FILL_TOLERANCE:
        raise ValueError(f"the filled lines add up to {filled_total:,.2f} TRY, but the ledger's "
                         f"payments on the missing days to {tie['try_on_missing_file_days']:,.2f}")

    originals = collections.copy()
    originals["filled_from_ledger"] = False
    filled["filled_from_ledger"] = True
    combined = pd.concat([originals, filled], ignore_index=True)
    combined = combined.sort_values("document_date", kind="stable", ignore_index=True)
    filled_days = sorted(set(payments["posting_date"].dt.date))
    return combined, filled_days


def payment_lines_from_ledger(payments: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """Ledger payment lines rewritten as daily-file payment lines with the given `columns`.

    Fields the ledger does not have (customer name, bank, invoice number) are left empty.
    """
    is_unapplied = payments["invoice_reference"].isna()
    lines = pd.DataFrame({
        "company_code": payments["company_code"],
        "customer": payments["customer"],
        "customer_name": None,
        "document_date": payments["posting_date"],
        "bank": None,
        "sap_document_no": payments["document_no"].where(~is_unapplied, None),
        "invoice_no": None,
        # The ledger records a payment received as negative; a payment line, as positive.
        "amount_original": -payments["amount_document"],
        "currency_original": payments["document_currency"],
        "amount_try": -payments["amount_local"],
        "currency_local": "TRY",
        "source_file": FROM_LEDGER,
        "source_row": payments["source_row"],
        "customer_id": payments["customer_id"],
        "key": payments["key"],
    })
    missing = [column for column in columns if column not in lines.columns]
    if missing:
        raise ValueError(f"cannot fill the payment-line columns {missing} from the ledger")
    lines = lines[list(columns)].reset_index(drop=True)
    # Text columns hold None for an empty value, as in the daily-file lines, whatever type
    # pandas gave them (its generic object type or its own text type).
    for column in lines.columns:
        is_number = pd.api.types.is_numeric_dtype(lines[column])
        is_date = pd.api.types.is_datetime64_any_dtype(lines[column])
        if not is_number and not is_date:
            lines[column] = lines[column].astype(object).where(lines[column].notna(), None)
    return lines
