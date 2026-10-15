"""Each customer's receivables as they stood at the end of each week, from the AR ledger.

Only ledger lines posted by the end of week t are used for week t, so every number is one
that was known at the time. Amounts owed in a foreign currency are worked out in that
currency and converted to TRY at the last exchange rate known at t.
"""
from datetime import date

import pandas as pd

# The local currency: amounts in it need no conversion.
LOCAL_CURRENCY = "TRY"

# The columns of the table `open_invoices_at` returns, in order.
OPEN_INVOICE_COLUMNS = ["key", "company_code", "customer_id", "document_no", "currency",
                        "posting_date", "net_due_date", "amount", "paid", "outstanding"]

# An open invoice with no more than this left to pay (in its own currency) is left out.
NOTHING_LEFT = 0.005


def exchange_rates(collections: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """TRY per unit of each foreign currency, for every day that has a line in it.

    Columns: day, currency, rate. Every line in a foreign currency gives a rate (TRY amount /
    amount in its own currency) on its date: the daily files' payment lines on their document
    date, and the ledger's lines on their posting date (the ledger goes back further, and
    also holds currencies that are rarely paid). A day's rate is the median of its lines,
    because amounts rounded to the kurus make the lines of one day differ very slightly.
    Lines with a zero amount give no rate and are skipped.
    """
    from_daily_files = rate_lines(collections, "document_date", "currency_original",
                                  "amount_original", "amount_try", "the daily files")
    from_ledger = rate_lines(ledger, "posting_date", "document_currency",
                             "amount_document", "amount_local", "the ledger")
    lines = pd.concat([from_daily_files, from_ledger], ignore_index=True)
    rates = lines.groupby(["day", "currency"], as_index=False)["rate"].median()
    return rates.sort_values(["currency", "day"], ignore_index=True)


def rate_lines(table: pd.DataFrame, date_column: str, currency_column: str,
               own_amount_column: str, try_amount_column: str, source: str) -> pd.DataFrame:
    """One rate per foreign-currency line of `table`: columns day, currency, rate.

    Currency codes are compared trimmed and upper-cased ("try " is TRY). Raises ValueError,
    naming `source` and the first rows, if a line has no currency, or if a foreign line's
    rate would be missing, zero or negative: a rate is never guessed or dropped quietly.
    """
    currencies = table[currency_column].map(normalise_currency)
    no_currency = currencies.isna()
    if no_currency.any():
        raise ValueError(f"{source}: {no_currency.sum()} lines have no currency "
                         f"(first rows: {first_rows(table, no_currency)})")

    is_foreign = (currencies != LOCAL_CURRENCY) & (table[own_amount_column] != 0)
    foreign = table[is_foreign]
    rates = foreign[try_amount_column] / foreign[own_amount_column]

    problems = [
        (rates.isna(), "have a missing amount"),
        (rates == 0, "have a TRY amount of zero but a non-zero amount in their own currency"),
        (rates < 0, "have TRY and own-currency amounts of opposite signs"),
    ]
    for has_problem, description in problems:
        if has_problem.any():
            raise ValueError(f"{source}: {has_problem.sum()} foreign-currency lines "
                             f"{description} (first rows: {first_rows(foreign, has_problem)})")

    return pd.DataFrame({
        "day": foreign[date_column].dt.date,
        "currency": currencies[is_foreign],
        "rate": rates,
    })


def normalise_currency(value):
    """A currency code trimmed and upper-cased, e.g. ' eur ' -> 'EUR'; None if the cell is empty."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip().upper()


def first_rows(table: pd.DataFrame, has_problem: pd.Series) -> list:
    """The source rows of the first five lines of `table` where `has_problem` is True."""
    return table.loc[has_problem, "source_row"].head(5).tolist()


def rates_known_at(rates: pd.DataFrame, cutoff: date) -> dict[str, float]:
    """{currency: rate} using, for each currency, its last rate on or before `cutoff`.

    TRY is always 1.0. A currency with no rate yet on `cutoff` is left out, so that converting
    an amount in it fails loudly later rather than with a guessed rate.
    """
    known = rates[rates["day"] <= cutoff]
    latest = known.sort_values("day").groupby("currency")["rate"].last()
    rates_at_cutoff = {currency: float(rate) for currency, rate in latest.items()}
    rates_at_cutoff[LOCAL_CURRENCY] = 1.0
    return rates_at_cutoff


def open_invoices_at(ledger: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Every invoice open at the end of `cutoff`, as the ledger showed it then.

    Only lines posted on or before `cutoff` are used. An invoice is open if it is not
    cleared by `cutoff` (a clearing date on `cutoff` itself counts as cleared). Its
    `outstanding` amount is its `amount` minus the payments linked to it (same company code,
    `invoice_reference` = its `document_no`) posted by `cutoff`, all in the invoice's own
    currency. Credit memos, refunds and unapplied cash are not applied to invoices.

    Needs the ledger's `key`, `customer_id` (from `add_keys`) and `document_role` columns.
    Columns returned: see OPEN_INVOICE_COLUMNS. An open invoice with nothing left to pay is
    left out, with a printed note.
    """
    # Ledger dates are whole days, so comparing with the cutoff day covers all of that day.
    cutoff_day = pd.Timestamp(cutoff)
    known = ledger[ledger["posting_date"] <= cutoff_day]

    invoices = known[known["document_role"] == "invoice"]
    is_open = invoices["clearing_date"].isna() | (invoices["clearing_date"] > cutoff_day)
    invoices = invoices[is_open]
    max_invoice_count = invoices.groupby(["company_code", "document_no"]).size().max()
    if max_invoice_count > 1:
        raise ValueError("Duplicate entries in the invoices table!")

    # Payments are negative in the ledger; `paid` is kept as a positive amount.
    payments = known[(known["document_role"] == "payment") & known["invoice_reference"].notna()]
    paid_per_invoice = (-payments["amount_document"]).groupby(
        [payments["company_code"], payments["invoice_reference"]]).sum()
    paid_per_invoice.index.names = ["company_code", "document_no"]

    table = invoices.merge(paid_per_invoice.rename("paid").reset_index(),
                           on=["company_code", "document_no"], how="left")
    table["paid"] = table["paid"].fillna(0.0)
    table["amount"] = table["amount_document"]
    table["outstanding"] = table["amount"] - table["paid"]
    table["currency"] = table["document_currency"].map(normalise_currency)

    nothing_left = table["outstanding"] <= NOTHING_LEFT
    if nothing_left.any():
        print(f"Note: at {cutoff_day.date()}, {nothing_left.sum()} open invoices have nothing "
              f"left to pay "
              f"and are left out")
    return table.loc[~nothing_left, OPEN_INVOICE_COLUMNS].reset_index(drop=True)
