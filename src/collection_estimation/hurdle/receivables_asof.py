"""Each customer's receivables as they stood at the end of each week, from the AR ledger.

Only ledger lines posted by the end of week t are used for week t, so every number is one
that was known at the time. Amounts owed in a foreign currency are worked out in that
currency and converted to TRY at the last exchange rate known at t.
"""
from datetime import date, timedelta

import pandas as pd

from .series import week_numbers

# The local currency: amounts in it need no conversion.
LOCAL_CURRENCY = "TRY"

# The columns of the table `open_invoices_at` returns, in order.
OPEN_INVOICE_COLUMNS = ["key", "company_code", "customer_id", "document_no", "currency",
                        "posting_date", "net_due_date", "amount", "paid", "outstanding"]

# An open invoice with no more than this left to pay (in its own currency) is left out.
NOTHING_LEFT = 0.005

# The columns of the table `payment_delays_at` returns, in order.
DELAY_COLUMNS = ["key", "usual_delay_days", "cleared_invoices"]

# Per-key totals must add up to the open invoices' total within this (TRY).
TIE_OUT_TOLERANCE = 0.01


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
    max_invoice_count = invoices.groupby(["key", "document_no"]).size().max()
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


def payment_delays_at(ledger: pd.DataFrame, cutoff: date, min_cleared: int) -> pd.DataFrame:
    """How late each customer usually pays, judged from invoices cleared by `cutoff`.

    For every invoice posted and cleared on or before `cutoff`, the money arrived on the
    latest posting date of the payments in its clearing group (the lines sharing its
    company and `clearing_document`, which also finds unapplied cash matched later). Days
    late = that date - the net due date; negative means paid early. The clearing date itself
    is not used: matching can lag the payment by weeks.

    One row per key with at least one such invoice: `usual_delay_days` is the median days
    late, or NaN if the key has fewer than `min_cleared` cleared invoices; `cleared_invoices`
    is how many. An invoice cleared without any payment (e.g. by a credit memo) does not
    count. Raises ValueError if a cleared invoice has no clearing document.
    """
    cutoff_day = pd.Timestamp(cutoff)
    known = ledger[ledger["posting_date"] <= cutoff_day]

    invoices = known[known["document_role"] == "invoice"]
    max_invoice_count = invoices.groupby(["key", "document_no"]).size().max()
    if max_invoice_count > 1:
        raise ValueError("Duplicate entries in the invoices table!")

    cleared = invoices[invoices["clearing_date"] <= cutoff_day]
    no_clearing_document = cleared["clearing_document"].isna()

    if no_clearing_document.any():
        raise ValueError(f"{no_clearing_document.sum()} invoices cleared by {cutoff_day.date()} "
                         f"have no clearing document, so their payments cannot be found "
                         f"(first rows: {first_rows(cleared, no_clearing_document)})")

    payments = known[(known["document_role"] == "payment") & known["clearing_document"].notna()]
    money_arrived = payments.groupby(["company_code", "clearing_document"])["posting_date"].max()

    # An inner merge: invoices cleared without any payment have no arrival date and drop out.
    table = cleared.merge(money_arrived.rename("money_arrived").reset_index(),
                          on=["company_code", "clearing_document"], how="inner")
    table["days_late"] = (table["money_arrived"] - table["net_due_date"]).dt.days

    per_key = table.groupby("key").agg(
        usual_delay_days=("days_late", "median"),
        cleared_invoices=("days_late", "size"),
    ).reset_index()
    per_key["usual_delay_days"] = per_key["usual_delay_days"].astype("float64")
    too_few = per_key["cleared_invoices"] < min_cleared
    per_key.loc[too_few, "usual_delay_days"] = float("nan")
    return per_key[DELAY_COLUMNS]


def due_in_week_columns(horizons: int) -> list[str]:
    """['due_in_week_1', ..., 'due_in_week_<horizons>']."""
    return [f"due_in_week_{h}" for h in range(1, horizons + 1)]


def receivables_columns(horizons: int) -> list[str]:
    """The columns of the table `receivables_at` returns, in order."""
    return (["key", "week"] + due_in_week_columns(horizons)
            + ["due_later", "overdue", "overdue_days", "open_invoices", "unapplied_cash",
               "open_balance", "usual_delay_days", "cleared_invoices"])


def receivables_at(ledger: pd.DataFrame, week: int, week_zero: date, rates: pd.DataFrame,
                   horizons: int, min_cleared: int) -> pd.DataFrame:
    """Each customer's receivables at the end of `week` (its Sunday), as the ledger showed them.

    One row per key with at least one ledger line posted by then. Amounts are in TRY at the
    rates known at the end of the week. Columns (see `receivables_columns`):
    - due_in_week_1 ... due_in_week_<horizons>: outstanding on open invoices due in that week
      ahead; due_later: due after the last of them; overdue: due on or before the end of
      `week`; overdue_days: the amount-weighted average days past due (0 if none);
      open_invoices: how many are open. These add up to the open invoices' outstanding total.
    - unapplied_cash: open payments not yet matched to an invoice (a positive amount);
      open_balance: the sum of all the customer's open lines (what it really owes).
    - usual_delay_days, cleared_invoices: from `payment_delays_at`.
    """
    cutoff = end_of_week(week_zero, week)
    rates_at_cutoff = rates_known_at(rates, cutoff)

    invoices = open_invoices_at(ledger, cutoff)
    invoices["outstanding_try"] = to_try(invoices["outstanding"], invoices["currency"],
                                         rates_at_cutoff, cutoff)
    buckets = due_buckets(invoices, week, week_zero, cutoff, horizons)
    balances = open_lines_at(ledger, cutoff, rates_at_cutoff)
    delays = payment_delays_at(ledger, cutoff, min_cleared)

    posted_by_cutoff = ledger[ledger["posting_date"] <= pd.Timestamp(cutoff)]
    # dtype=object keeps the column as text even when no key is known yet (an empty list
    # would otherwise become a column of numbers, which cannot be merged with text keys).
    keys = pd.Series(sorted(posted_by_cutoff["key"].unique()), dtype=object)
    table = pd.DataFrame({"key": keys})
    table["week"] = week
    for part in (buckets, balances, delays):
        table = table.merge(part, on="key", how="left")

    # A key with nothing open, or no cleared invoice yet, gets zeros; its usual delay stays NaN.
    amounts = due_in_week_columns(horizons) + ["due_later", "overdue", "overdue_days",
                                               "unapplied_cash", "open_balance"]
    table[amounts] = table[amounts].fillna(0.0)
    for count in ("open_invoices", "cleared_invoices"):
        table[count] = table[count].fillna(0).astype("int64")
    table["week"] = table["week"].astype("int64")
    return table[receivables_columns(horizons)]


def to_try(amounts: pd.Series, currencies: pd.Series, rates_at_cutoff: dict[str, float],
           cutoff: date) -> pd.Series:
    """`amounts` converted to TRY at `rates_at_cutoff` (from `rates_known_at`).

    Raises ValueError naming any currency that has no rate known at `cutoff`, rather than
    guessing one.
    """
    codes = currencies.map(normalise_currency)
    if codes.isna().any():
        raise ValueError(f"{codes.isna().sum()} amounts to convert at {cutoff} have no currency")
    no_rate = sorted(set(codes) - set(rates_at_cutoff))
    if no_rate:
        raise ValueError(f"no exchange rate known on or before {cutoff} for {no_rate}")
    return amounts * codes.map(rates_at_cutoff)


def due_buckets(invoices: pd.DataFrame, week: int, week_zero: date, cutoff: date,
                horizons: int) -> pd.DataFrame:
    """Per key, the open invoices' outstanding TRY split by when it is due.

    `invoices` is `open_invoices_at` with an `outstanding_try` column. An invoice due in week
    t+h (h = 1 ... horizons) goes to due_in_week_<h>; due later, to due_later; due on or
    before the end of week t, to overdue. Raises ValueError if the parts of a key do not add
    up to its total (no invoice may be lost or counted twice).
    """
    no_due_date = invoices["net_due_date"].isna()
    if no_due_date.any():
        raise ValueError(f"at {cutoff}, {no_due_date.sum()} open invoices have no net due date "
                         f"(first: {invoices.loc[no_due_date, 'document_no'].head(5).tolist()})")
    weeks_ahead = week_numbers(invoices["net_due_date"], week_zero) - week
    amount = invoices["outstanding_try"]
    is_overdue = weeks_ahead <= 0
    days_past_due = (pd.Timestamp(cutoff) - invoices["net_due_date"]).dt.days

    parts = pd.DataFrame({"key": invoices["key"], "total": amount})
    for h, column in enumerate(due_in_week_columns(horizons), start=1):
        parts[column] = amount.where(weeks_ahead == h, 0.0)
    parts["due_later"] = amount.where(weeks_ahead > horizons, 0.0)
    parts["overdue"] = amount.where(is_overdue, 0.0)
    parts["overdue_amount_x_days"] = (amount * days_past_due).where(is_overdue, 0.0)
    parts["open_invoices"] = 1

    per_key = parts.groupby("key", as_index=False).sum()
    # The amount-weighted average days past due; 0 for a key with nothing overdue.
    has_overdue = per_key["overdue"] > 0
    per_key["overdue_days"] = 0.0
    per_key.loc[has_overdue, "overdue_days"] = (per_key.loc[has_overdue, "overdue_amount_x_days"]
                                                / per_key.loc[has_overdue, "overdue"])

    columns = due_in_week_columns(horizons) + ["due_later", "overdue"]
    differs = (per_key[columns].sum(axis=1) - per_key["total"]).abs() > TIE_OUT_TOLERANCE
    if differs.any():
        raise ValueError(f"at {cutoff}, the due-date split does not add up to the open "
                         f"invoices' total for {differs.sum()} keys "
                         f"(first: {per_key.loc[differs, 'key'].head(5).tolist()})")
    return per_key[["key"] + columns + ["overdue_days", "open_invoices"]]


def open_lines_at(ledger: pd.DataFrame, cutoff: date,
                  rates_at_cutoff: dict[str, float]) -> pd.DataFrame:
    """Per key, in TRY at `rates_at_cutoff`: open_balance (the sum of every line open at the
    end of `cutoff`: invoices, their partial payments, unapplied cash, credit memos, refunds)
    and unapplied_cash (open payments with no invoice reference, as a positive amount).

    A line is open at `cutoff` if it was posted by then and not cleared by then.
    """
    cutoff_day = pd.Timestamp(cutoff)
    known = ledger[ledger["posting_date"] <= cutoff_day]
    is_open = known["clearing_date"].isna() | (known["clearing_date"] > cutoff_day)
    open_lines = known[is_open]

    amount_try = to_try(open_lines["amount_document"], open_lines["document_currency"],
                        rates_at_cutoff, cutoff)
    is_unapplied = ((open_lines["document_role"] == "payment")
                    & open_lines["invoice_reference"].isna())
    parts = pd.DataFrame({
        "key": open_lines["key"],
        "open_balance": amount_try,
        # A payment is negative in the ledger; unapplied cash is kept as a positive amount.
        "unapplied_cash": (-amount_try).where(is_unapplied, 0.0),
    })
    return parts.groupby("key", as_index=False).sum()[["key", "unapplied_cash", "open_balance"]]


def end_of_week(week_zero: date, week: int) -> date:
    """The Sunday that ends `week` (weeks run Monday to Sunday from the Monday `week_zero`)."""
    return week_zero + timedelta(days=7 * week + 6)


def receivables_by_week(ledger: pd.DataFrame, rates: pd.DataFrame, week_zero: date,
                        last_week: int, as_of: date, horizons: int, min_cleared: int,
                        leak_check_weeks: tuple[int, ...]) -> pd.DataFrame:
    """`receivables_at` for every week from 0 to `last_week`, in one table (key x week).

    `as_of` is the date the ledger was extracted. The last week may end a few days after it
    (e.g. on the weekend), but must start on or before it: otherwise the ledger is older
    than the weeks asked for, and their receivables would be incomplete, so this raises.
    Each week in `leak_check_weeks` is then rebuilt from a ledger cut off at that week (see
    `check_no_leak`); a difference raises.
    """
    last_monday = week_zero + timedelta(days=7 * last_week)
    if last_monday > as_of:
        raise ValueError(f"week {last_week} starts on {last_monday}, after the ledger's "
                         f"extract date {as_of}: the ledger is too old for it")
    if end_of_week(week_zero, last_week) > as_of:
        print(f"Note: week {last_week} ends on {end_of_week(week_zero, last_week)}, after the "
              f"ledger's extract date {as_of}; its receivables are as of {as_of}")

    tables = []
    for week in range(last_week + 1):
        tables.append(receivables_at(ledger, week, week_zero, rates, horizons, min_cleared))
        if (week + 1) % 10 == 0:
            print(f"  ... receivables for {week + 1} of {last_week + 1} weeks")
    by_week = pd.concat(tables, ignore_index=True)

    check_no_leak(by_week, ledger, rates, leak_check_weeks, last_week, week_zero, horizons,
                  min_cleared)
    return by_week


def check_no_leak(by_week: pd.DataFrame, ledger: pd.DataFrame, rates: pd.DataFrame,
                  weeks: tuple[int, ...], last_week: int, week_zero: date, horizons: int,
                  min_cleared: int) -> None:
    """Rebuild each of `weeks` from data cut off at that week, and compare with `by_week`.

    The ledger is cut with `ledger_as_known_at`, and the rates to days on or before the
    week's Sunday. If a week's rows differ, something from after the week was used: raises
    ValueError. A week with no rows (before any ledger line) compares as an empty table.
    Checking the last week proves little: nothing in the ledger comes after it.
    """
    for week in weeks:
        if not 0 <= week <= last_week:
            raise ValueError(f"leak check: week {week} is outside the weeks built "
                             f"(0 to {last_week})")
        cutoff = end_of_week(week_zero, week)
        rebuilt = receivables_at(ledger_as_known_at(ledger, cutoff), week, week_zero,
                                 rates[rates["day"] <= cutoff], horizons, min_cleared)
        stored = by_week[by_week["week"] == week].reset_index(drop=True)
        if not rebuilt.equals(stored):
            raise ValueError(f"leak check: week {week} differs when rebuilt from data cut off "
                             f"at {cutoff}, so something from after the week was used")


def ledger_as_known_at(ledger: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """The ledger as it looked at the end of `cutoff`: lines posted later removed, and
    clearings made later undone (clearing date and clearing document emptied)."""
    cutoff_day = pd.Timestamp(cutoff)
    known = ledger[ledger["posting_date"] <= cutoff_day].copy()
    cleared_later = known["clearing_date"] > cutoff_day
    known.loc[cleared_later, "clearing_date"] = pd.NaT
    known.loc[cleared_later, "clearing_document"] = None
    return known
