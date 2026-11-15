"""The payment-rhythm block (U18): when each customer's open money is likely to arrive, judged
from how it has paid before. Everything is as known at the end of the origin week t.

U18.2, the delay spread. Each invoice open at t is spread over the weeks after t by the
customer's past delays (the days late of its most recent cleared invoices, from
`receivables_asof.cleared_invoice_delays`). Example: 100 TRY due Monday 2 March, past delays
3, 5, 6, 12 and 20 days, forecast at the end of that week (Sunday 8 March). The arrivals,
moved to the next business day, are Thu 5, Mon 9, Mon 9, Mon 16 and Mon 23 March. The
invoice is still open on the 8th, so the first did not happen; the other four share the 100
TRY: 50 in the week after t, 25 in the second and 25 in the third.

  - A customer with fewer than `min_cleared` cleared invoices uses its company's most recent
    `company_delays_kept` instead (delays_pooled); with none at all, its invoices are not
    spread (without_delays).
  - An invoice older than every delay used has no arrival left after t: overdue_beyond_habit.
  - Each delay counts the same, whatever the size of its invoice: a small invoice says as
    much about when a customer pays as a large one.

U18.3, the payment habits. From each customer's own past payment days (the days its money
arrived, in the daily files and in the ledger before them): when in the month it pays, how
regularly, and how much of that habit falls in the target week. Example (2019): payments on
31 Jan, 28 Feb, 31 Mar, 29 Apr, 30 May and 28 Jun (gaps 28, 31, 29, 31 and 29 days), forecast
at the end of the week of Sunday 7 July. Every payment is in the last 5 days of its month, so
month_end_share is 1.0. Their places in the month are -1, -1, -1, -2, -2 and -3, so the week
of Monday 29 July (it holds 29-31 July, places -3 to -1) holds all of them. The next payment,
the last one plus each past gap, is 26, 29, 27, 29 or 27 July; moved to business days, that
is Friday 26 July once and Monday 29 July four times: all after t, so 0.2 in the week of
22 July and 0.8 in the week of 29 July.

  - A customer with fewer than `min_payment_days` payment days has no habit (NaN): a habit
    borrowed from other customers would not be its own.
  - Each payment day counts the same, whatever its amount: a small payment says as much about
    when a customer pays as a large one.

Dates are counted in days since week zero's Monday, so a date's week is its day // 7.
"""
from datetime import date, timedelta

import pandas as pd

from .holidays import HolidayTable, public_holidays
from .receivables_asof import (
    cleared_invoice_delays,
    end_of_week,
    ledger_as_known_at,
    open_invoices_at,
    rates_known_at,
    to_try,
)

# The per-row columns `expected_dues` returns after key, origin_week and h, in order.
# Amounts are deflated (TRY of week 0, by the price level of week t).
EXPECTED_DUE_COLUMNS = ["spread_in_target", "spread_before_target", "spread_share_in_target",
                        "overdue_beyond_habit", "shifted_due_in_target", "delay_iqr_days",
                        "delays_used", "delays_pooled"]

# The per-row columns `payment_habits` returns after key, origin_week and h, in order.
HABIT_COLUMNS = ["payment_days_used", "month_start_share", "mid_month_share", "month_end_share",
                 "median_gap_days", "gap_iqr_days", "payment_due_ratio",
                 "month_habit_in_target", "weekday_habit_in_target",
                 "next_payment_share_in_target"]

# The columns `payment_days` returns, in order.
PAYMENT_DAY_COLUMNS = ["key", "date", "day", "gross_receipts"]

# A day's place in its month (`month_place`): the 1st to the 15th count from the start
# (+1 ... +15), later days from the end (-1 is the last day), so month ends line up.
LAST_DAY_FROM_START = 15

# The parts of the month the habit shares count (days of the month, both ends included).
MONTH_START_DAYS = (1, 5)
MID_MONTH_DAYS = (11, 20)
MONTH_END_LAST_DAYS = 5        # the last 5 calendar days of the month

# Per key, the spread and the single shifted date must account for its open invoices within
# this (TRY).
SPREAD_TOLERANCE = 0.01

# The business-day calendar runs this many days beyond the first and last dates needed: an
# arrival moves forward to the next business day, at most over a long holiday and a weekend.
CALENDAR_MARGIN_DAYS = 21


def spread_columns(horizons: int) -> list[str]:
    """The per-key columns of the delay spread in each week after t (see `invoice_spread_at`)."""
    return [f"spread_week_{k}" for k in range(1, horizons + 1)]


def shifted_columns(horizons: int) -> list[str]:
    """The per-key columns of the single shifted date in each week after t."""
    return [f"shifted_week_{k}" for k in range(1, horizons + 1)]


def key_spread_columns(horizons: int) -> list[str]:
    """The columns `invoice_spread_at` returns, in order. Amounts in TRY (not deflated)."""
    return (["key", "origin_week", "open_outstanding"] + spread_columns(horizons)
            + ["spread_later", "overdue_beyond_habit", "without_delays"]
            + ["shifted_past"] + shifted_columns(horizons)
            + ["shifted_later", "delay_iqr_days", "delays_used", "delays_pooled"])


def days_since_week_zero(dates: pd.Series, week_zero: date) -> pd.Series:
    """Each date as the number of days since `week_zero` (a Monday): its week is days // 7."""
    return (pd.to_datetime(dates) - pd.Timestamp(week_zero)).dt.days


def next_business_days(start: date, end: date, holidays: HolidayTable,
                       week_zero: date) -> dict[int, int]:
    """{day: the first business day on or after it} for every date from `start` to `end`;
    the keys and values are days since `week_zero`. Business days are Monday to Friday
    without public holidays. Days after the last business day of the range are left out."""
    holiday_days = public_holidays(start, end, holidays)
    lookup = {}
    next_business = None
    # Walk backwards, so each day knows the nearest business day on or after it.
    day = end
    while day >= start:
        if day.weekday() < 5 and day not in holiday_days:
            next_business = day
        if next_business is not None:
            lookup[(day - week_zero).days] = (next_business - week_zero).days
        day -= timedelta(days=1)
    return lookup


def to_next_business_day(days: pd.Series, next_business: dict[int, int]) -> pd.Series:
    """Each day (days since week zero) moved to the first business day on or after it
    (`next_business` from `next_business_days`). Days outside it are kept as they are."""
    moved = days.map(next_business)
    moved = moved.fillna(days)
    return moved.astype("int64")


def delay_samples_at(keys: pd.DataFrame, delays: pd.DataFrame, cutoff: date, min_cleared: int,
                     delays_kept: int, company_delays_kept: int) -> pd.DataFrame:
    """The delays each key of `keys` (columns key, company_code) uses at `cutoff`: one row per
    (key, delay), columns key, days_late, delays_pooled.

    `delays` is `cleared_invoice_delays`; only its invoices known on or before `cutoff` are
    used, the most recent by arrival first. A key with at least `min_cleared` of its own uses
    its last `delays_kept`; otherwise its company's last `company_delays_kept`
    (delays_pooled True). A key with neither has no rows.
    """
    known = delays[delays["known_on"] <= pd.Timestamp(cutoff)]
    # Oldest first, so the last rows of each group are the most recent; ties in a fixed order.
    known = known.sort_values(["money_arrived", "company_code", "key", "document_no"],
                              kind="stable")

    # Its own delays: counted over all of them, then only the last `delays_kept` kept.
    mine = known[known["key"].isin(keys["key"])]
    own_counts = mine.groupby("key").size()
    has_own = own_counts.index[own_counts >= min_cleared]
    own = mine.groupby("key").tail(delays_kept)
    own = own[own["key"].isin(has_own)]
    own = own[["key", "days_late"]].copy()
    own["delays_pooled"] = False

    # The others: their company's last `company_delays_kept`, copied onto each of them.
    company = known.groupby("company_code").tail(company_delays_kept)
    company = company[["company_code", "days_late"]]
    pooled_keys = keys[~keys["key"].isin(has_own)]
    pooled_keys = pooled_keys[["key", "company_code"]]
    pooled = pooled_keys.merge(company, on="company_code", how="inner")
    pooled = pooled[["key", "days_late"]].copy()
    pooled["delays_pooled"] = True

    samples = pd.concat([own, pooled], ignore_index=True)
    samples["days_late"] = samples["days_late"].astype("int64")
    samples["delays_pooled"] = samples["delays_pooled"].astype(bool)
    return samples


def open_invoices_in_try(ledger: pd.DataFrame, keys: pd.DataFrame, rates: pd.DataFrame,
                         cutoff: date, week_zero: date) -> pd.DataFrame:
    """The invoices open at `cutoff` (`open_invoices_at`) of the keys in `keys`, with their
    outstanding amount in TRY at the rates known then (outstanding_try), their net due date
    as days since week zero (due_day) and a number per invoice (invoice). Raises ValueError
    if one has no net due date."""
    invoices = open_invoices_at(ledger, cutoff)
    invoices = invoices[invoices["key"].isin(keys["key"])].reset_index(drop=True)
    no_due_date = invoices["net_due_date"].isna()
    if no_due_date.any():
        first = invoices.loc[no_due_date, "document_no"].head(5).tolist()
        raise ValueError(f"at {cutoff}, {int(no_due_date.sum())} open invoices have no net due "
                         f"date (first: {first})")
    rates_then = rates_known_at(rates, cutoff)
    invoices["outstanding_try"] = to_try(invoices["outstanding"], invoices["currency"],
                                         rates_then, cutoff)
    invoices["due_day"] = days_since_week_zero(invoices["net_due_date"], week_zero)
    invoices["invoice"] = range(len(invoices))
    return invoices


def delay_statistics(samples: pd.DataFrame) -> pd.DataFrame:
    """Per key of `samples` (`delay_samples_at`): delays_used, delays_pooled, median_delay and
    delay_iqr_days (the 75th minus the 25th percentile). One row per key, column key."""
    by_key = samples.groupby("key")["days_late"]
    stats = pd.DataFrame({
        "delays_used": by_key.size(),
        "delays_pooled": samples.groupby("key")["delays_pooled"].first(),
        "median_delay": by_key.median(),
        "delay_iqr_days": by_key.quantile(0.75) - by_key.quantile(0.25),
    })
    stats.index.name = "key"
    return stats.reset_index()


def week_column(week: int, prefix: str, horizons: int) -> str:
    """The per-key column of a week after t: `<prefix>_week_<k>` for 1 ... horizons,
    `<prefix>_later` after them, `<prefix>_past` for t or before."""
    if week < 1:
        return f"{prefix}_past"
    if week > horizons:
        return f"{prefix}_later"
    return f"{prefix}_week_{week}"


def amounts_per_week(placed: pd.DataFrame, prefix: str, horizons: int) -> pd.DataFrame:
    """`placed` (columns key, week, amount: one row per piece of an invoice and the week after
    t it lands in) summed per key and week: one row per key, one column per week
    (`week_column`), 0 where nothing lands."""
    if placed.empty:
        return pd.DataFrame({"key": pd.Series(dtype=object)})
    placed = placed.copy()
    names = {week: week_column(week, prefix, horizons) for week in set(placed["week"])}
    placed["column"] = placed["week"].map(names)
    wide = placed.pivot_table(index="key", columns="column", values="amount", aggfunc="sum",
                              fill_value=0.0)
    wide.columns.name = None
    return wide.reset_index()


def add_per_key(table: pd.DataFrame, per_key: pd.DataFrame, fill) -> pd.DataFrame:
    """`table` (one row per key) with the columns of `per_key` (column key, at most one row
    per key) added; keys without a row in `per_key` get `fill`."""
    added = [column for column in per_key.columns if column != "key"]
    table = table.merge(per_key, on="key", how="left", validate="one_to_one")
    for column in added:
        table[column] = table[column].fillna(fill)
    return table


def invoice_spread_at(ledger: pd.DataFrame, delays: pd.DataFrame, rates: pd.DataFrame,
                      keys: pd.DataFrame, origin: int, week_zero: date,
                      next_business: dict[int, int], horizons: int, min_cleared: int,
                      delays_kept: int, company_delays_kept: int) -> pd.DataFrame:
    """The delay spread of every key of `keys` (columns key, company_code) at the end of week
    `origin`: one row per key, columns `key_spread_columns`, amounts in TRY at the rates known
    then (not deflated).

    For each invoice open at t (`open_invoices_in_try`) and each delay its key uses
    (`delay_samples_at`): arrival = net due date + the delay, moved to the next business day
    (`next_business`, from `next_business_days`). Only arrivals after t are possible (the
    invoice is still open); the outstanding amount is shared equally among them, by the week
    after t they fall in: spread_week_1 ... spread_week_<horizons>, then spread_later. An
    invoice with no possible arrival goes to overdue_beyond_habit; one whose key uses no
    delays to without_delays. The single shifted date: net due date + the median of the key's
    delays (rounded to whole days, a half to the even number), moved to the next business day:
    shifted_week_1 ... shifted_week_<horizons>, then shifted_later; one not after t goes to
    shifted_past and counts in no week. delay_iqr_days is the 75th minus the 25th percentile
    of the delays used (NaN without delays); delays_used how many; delays_pooled whether
    they are the company's.

    Raises ValueError if an open invoice has no net due date, if a key is repeated in
    `keys`, or if a key's spread or shifted amounts do not add up to its open invoices
    within SPREAD_TOLERANCE.
    """
    repeated = keys["key"].duplicated()
    if repeated.any():
        first = keys.loc[repeated, "key"].head(5).tolist()
        raise ValueError(f"keys are repeated (first: {first})")
    cutoff = end_of_week(week_zero, origin)
    invoices = open_invoices_in_try(ledger, keys, rates, cutoff, week_zero)
    samples = delay_samples_at(keys, delays, cutoff, min_cleared, delays_kept,
                               company_delays_kept)

    # 1. One row per key: its open invoices and the delays it uses.
    table = keys[["key"]].reset_index(drop=True)
    table["origin_week"] = origin
    open_total = invoices.groupby("key")["outstanding_try"].sum()
    open_total = open_total.rename("open_outstanding").reset_index()
    table = add_per_key(table, open_total, fill=0.0)
    stats = delay_statistics(samples)
    table = add_per_key(table, stats[["key", "delay_iqr_days"]], fill=float("nan"))
    table = add_per_key(table, stats[["key", "delays_used"]], fill=0)
    table = add_per_key(table, stats[["key", "delays_pooled"]], fill=False)

    # 2. The spread. Every (open invoice, delay) pair, and the week after t it arrives in.
    pairs = invoices[["invoice", "key", "due_day", "outstanding_try"]].merge(
        samples[["key", "days_late"]], on="key", how="inner")
    arrival_day = to_next_business_day(pairs["due_day"] + pairs["days_late"], next_business)
    pairs["week"] = arrival_day // 7 - origin

    # The invoice is still open at t, so only arrivals after t can happen; its amount is
    # shared equally among those.
    possible = pairs[pairs["week"] >= 1].copy()
    arrivals_left = possible.groupby("invoice")["week"].transform("size")
    possible["amount"] = possible["outstanding_try"] / arrivals_left
    spread = amounts_per_week(possible[["key", "week", "amount"]], "spread", horizons)
    table = add_per_key(table, spread, fill=0.0)

    # 3. What is not spread: invoices with delays but no arrival left after t, and invoices
    # whose key uses no delays at all.
    with_delays = invoices["key"].isin(samples["key"])
    has_arrival_left = invoices["invoice"].isin(possible["invoice"])
    beyond = invoices[with_delays & ~has_arrival_left]
    beyond = beyond.groupby("key")["outstanding_try"].sum()
    table = add_per_key(table, beyond.rename("overdue_beyond_habit").reset_index(), fill=0.0)
    without = invoices[~with_delays]
    without = without.groupby("key")["outstanding_try"].sum()
    table = add_per_key(table, without.rename("without_delays").reset_index(), fill=0.0)

    # 4. The single shifted date: due + the key's median delay. One not after t counts in no
    # week (shifted_past): mostly invoices that will not be paid soon, which would pile up in
    # week t+1 otherwise.
    placed = invoices[with_delays].merge(stats[["key", "median_delay"]], on="key", how="left")
    median_days = placed["median_delay"].round().astype("int64")
    shifted_day = to_next_business_day(placed["due_day"] + median_days, next_business)
    placed["week"] = shifted_day // 7 - origin
    placed["amount"] = placed["outstanding_try"]
    shifted = amounts_per_week(placed[["key", "week", "amount"]], "shifted", horizons)
    table = add_per_key(table, shifted, fill=0.0)

    # Every column, also a week where nothing landed for any key.
    for column in key_spread_columns(horizons):
        if column not in table.columns:
            table[column] = 0.0
    table["delays_used"] = table["delays_used"].astype("int64")
    table["delays_pooled"] = table["delays_pooled"].astype(bool)
    table = table[key_spread_columns(horizons)]
    check_spread_adds_up(table, horizons)
    return table


def check_spread_adds_up(table: pd.DataFrame, horizons: int) -> None:
    """Per key, the spread (with overdue_beyond_habit) and the shifted date must each place
    every open invoice exactly once, beside without_delays. Raises ValueError otherwise."""
    checks = {
        "spread": spread_columns(horizons) + ["spread_later", "overdue_beyond_habit"],
        "shifted date": ["shifted_past"] + shifted_columns(horizons) + ["shifted_later"],
    }
    for name, columns in checks.items():
        placed = table[columns].sum(axis=1) + table["without_delays"]
        off = (placed - table["open_outstanding"]).abs() > SPREAD_TOLERANCE
        if off.any():
            first = table.loc[off, "key"].head(5).tolist()
            raise ValueError(f"the {name} of {int(off.sum())} keys does not add up to their "
                             f"open invoices (first: {first})")


def expected_dues(panel_rows: pd.DataFrame, ledger: pd.DataFrame, rates: pd.DataFrame,
                  levels: pd.Series, week_zero: date, holidays: HolidayTable, horizons: int,
                  min_cleared: int, delays_kept: int, company_delays_kept: int,
                  leak_check_weeks: tuple[int, ...]) -> pd.DataFrame:
    """The delay-spread features of every row of `panel_rows` (columns key, company_code,
    origin_week, h): one row per panel row, in its order, columns key, origin_week, h and
    `EXPECTED_DUE_COLUMNS`.

      spread_in_target        the spread (`invoice_spread_at`) in week t+h
      spread_before_target    the spread in weeks t+1 ... t+h-1
      spread_share_in_target  spread_in_target / the key's open invoices (NaN if none)
      overdue_beyond_habit    open invoices older than every delay the key uses
      shifted_due_in_target   the single shifted date (due + median delay) in week t+h
      delay_iqr_days, delays_used, delays_pooled   the delays the key uses at t
    Amounts are deflated by the price level of week t (`levels`), as the receivables
    features. A key with nothing open gets 0 (and NaN for the share); one with no delays NaN
    for delay_iqr_days and 0 delays_used.

    The business days come from `holidays`. Each week of `leak_check_weeks` among the panel's
    origins is rebuilt from the ledger and rates as known then (`ledger_as_known_at`); a
    difference raises ValueError, as do a week without a price level or one of 0 or below,
    and the checks of `invoice_spread_at`.
    """
    delays = cleared_invoice_delays(ledger)
    origins = sorted(set(panel_rows["origin_week"]))
    start = end_of_week(week_zero, origins[0]) - timedelta(days=CALENDAR_MARGIN_DAYS)
    end = end_of_week(week_zero, origins[-1] + horizons) + timedelta(days=CALENDAR_MARGIN_DAYS)
    next_business = next_business_days(start, end, holidays, week_zero)

    def spread_at(origin, ledger_then, rates_then, delays_then):
        keys = panel_rows.loc[panel_rows["origin_week"] == origin, ["key", "company_code"]]
        keys = keys.drop_duplicates("key")
        return invoice_spread_at(ledger_then, delays_then, rates_then, keys, origin,
                                 week_zero, next_business, horizons, min_cleared, delays_kept,
                                 company_delays_kept)

    # 1. The spread per key at every origin.
    tables = []
    for number, origin in enumerate(origins, start=1):
        tables.append(spread_at(origin, ledger, rates, delays))
        if number % 10 == 0:
            print(f"  ... delay spread for {number} of {len(origins)} origins")
    per_key = pd.concat(tables, ignore_index=True)

    # 2. The leak check: rebuilt from the ledger and the rates as they were at the week.
    for week in leak_check_weeks:
        if week not in origins:
            continue
        cutoff = end_of_week(week_zero, week)
        ledger_then = ledger_as_known_at(ledger, cutoff)
        rates_then = rates[rates["day"] <= cutoff]
        rebuilt = spread_at(week, ledger_then, rates_then, cleared_invoice_delays(ledger_then))
        stored = per_key[per_key["origin_week"] == week].reset_index(drop=True)
        if not rebuilt.equals(stored):
            raise ValueError(f"leak check: the delay spread of week {week} differs when rebuilt "
                             f"from data cut off at {cutoff}, so something from after the week "
                             f"was used")

    # 3. Deflated by the price level of week t.
    level = per_key["origin_week"].map(levels)
    if level.isna().any():
        missing = sorted(set(per_key.loc[level.isna(), "origin_week"]))
        raise ValueError(f"no price level for weeks {missing[:10]}")
    if (level <= 0).any():
        bad = sorted(set(per_key.loc[level <= 0, "origin_week"]))
        raise ValueError(f"the price level is 0 or below for weeks {bad[:10]}")
    amounts = (["open_outstanding"] + spread_columns(horizons) + shifted_columns(horizons)
               + ["overdue_beyond_habit"])
    for column in amounts:
        per_key[column] = per_key[column] / level

    # 4. Onto the panel rows: each row takes the week after t that is its own h.
    rows = panel_rows[["key", "origin_week", "h"]].merge(
        per_key, on=["key", "origin_week"], how="left", validate="many_to_one")
    features = rows[["key", "origin_week", "h"]].copy()
    features["spread_in_target"] = 0.0
    features["spread_before_target"] = 0.0
    features["shifted_due_in_target"] = 0.0
    for h in range(1, horizons + 1):
        at_h = rows["h"] == h
        earlier = [f"spread_week_{k}" for k in range(1, h)]
        features.loc[at_h, "spread_in_target"] = rows.loc[at_h, f"spread_week_{h}"]
        features.loc[at_h, "spread_before_target"] = rows.loc[at_h, earlier].sum(axis=1)
        features.loc[at_h, "shifted_due_in_target"] = rows.loc[at_h, f"shifted_week_{h}"]
    has_open = rows["open_outstanding"] > 0
    features["spread_share_in_target"] = float("nan")
    features.loc[has_open, "spread_share_in_target"] = (
        features.loc[has_open, "spread_in_target"] / rows.loc[has_open, "open_outstanding"])
    features["overdue_beyond_habit"] = rows["overdue_beyond_habit"]
    features["delay_iqr_days"] = rows["delay_iqr_days"]
    features["delays_used"] = rows["delays_used"].astype("int64")
    features["delays_pooled"] = rows["delays_pooled"].astype(bool)
    features = features[["key", "origin_week", "h"] + EXPECTED_DUE_COLUMNS]
    return features.reset_index(drop=True)


# U18.3, the payment habits.

def habit_week_columns(prefix: str, horizons: int) -> list[str]:
    """The per-key columns of a habit in each week after t: `<prefix>_week_1` ... horizons."""
    return [f"{prefix}_week_{k}" for k in range(1, horizons + 1)]


def key_habit_columns(horizons: int) -> list[str]:
    """The columns `habits_at` returns, in order."""
    return (["key", "origin_week", "payment_days_used", "month_start_share", "mid_month_share",
             "month_end_share", "median_gap_days", "gap_iqr_days", "payment_due_ratio"]
            + habit_week_columns("month_habit", horizons)
            + habit_week_columns("weekday_habit", horizons)
            + habit_week_columns("next_payment", horizons))


def payment_days(collections: pd.DataFrame, ledger: pd.DataFrame, first_file_day: date,
                 week_zero: date, use_ledger_before_first_file: bool) -> pd.DataFrame:
    """One row per key and day on which money from it arrived: columns `PAYMENT_DAY_COLUMNS`,
    sorted by key, then day.

    From the daily files' payment lines (`collections`, with keys: document_date and
    amount_try) and, if `use_ledger_before_first_file` (the [history] switch, so both blocks
    see the same history), the ledger's payment lines before the first daily file
    (`first_file_day`): their posting date, and the TRY amount with the sign flipped (the
    ledger records money received as negative). As for the weekly amounts, only positive
    amounts count, so a day with only refunds is not a payment day. day is the date as days
    since `week_zero`.
    """
    from_files = pd.DataFrame({
        "key": collections["key"],
        "date": pd.to_datetime(collections["document_date"]),
        "gross_receipts": collections["amount_try"].clip(lower=0),
    })
    is_early_payment = ((ledger["document_role"] == "payment")
                        & (ledger["posting_date"] < pd.Timestamp(first_file_day)))
    early = ledger[is_early_payment]
    if not use_ledger_before_first_file:
        early = early.iloc[0:0]
    from_ledger = pd.DataFrame({
        "key": early["key"],
        "date": pd.to_datetime(early["posting_date"]),
        "gross_receipts": (-early["amount_local"]).clip(lower=0),
    })
    lines = pd.concat([from_files, from_ledger], ignore_index=True)

    # One row per key and date; a day whose amounts add up to nothing is not a payment day.
    days = lines.groupby(["key", "date"], as_index=False)["gross_receipts"].sum()
    days = days[days["gross_receipts"] > 0].copy()
    days["day"] = days_since_week_zero(days["date"], week_zero).astype("int64")
    days = days.sort_values(["key", "day"], ignore_index=True)
    days["key"] = days["key"].astype(object)
    days["gross_receipts"] = days["gross_receipts"].astype("float64")
    return days[PAYMENT_DAY_COLUMNS]


def month_place(dates: pd.Series) -> pd.Series:
    """Each date's place in its month: +1 ... +15 for the 1st to the 15th, later days counted
    from the end, -1 for the last day. So 30 March and 27 February are both -2: month ends
    line up, whatever the length of the month."""
    day_of_month = dates.dt.day
    from_end = day_of_month - dates.dt.days_in_month - 1
    place = day_of_month.where(day_of_month <= LAST_DAY_FROM_START, from_end)
    return place.astype("int64")


def payment_samples_at(days: pd.DataFrame, keys: pd.Series, cutoff_day: int,
                       payment_days_kept: int) -> pd.DataFrame:
    """The payment days each key in `keys` uses at the end of `cutoff_day` (days since week
    zero): its last `payment_days_kept` of `days` (`payment_days`) on or before it. Sorted by
    key, then day (oldest first), with four columns added: day_of_month, place
    (`month_place`), weekday (0 is Monday) and gap (days since the key's previous payment day
    among these; NaN for its first)."""
    known = days[(days["day"] <= cutoff_day) & days["key"].isin(keys)]
    # Oldest first, so the last rows of each key are its most recent payment days.
    known = known.sort_values(["key", "day"], kind="stable")
    samples = known.groupby("key").tail(payment_days_kept).reset_index(drop=True)
    samples["day_of_month"] = samples["date"].dt.day
    samples["place"] = month_place(samples["date"])
    samples["weekday"] = samples["date"].dt.weekday
    # Each payment day minus the one before it, within the same key.
    samples["gap"] = samples.groupby("key")["day"].diff()
    return samples


def habit_statistics(samples: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Per key of `samples` (`payment_samples_at`), one row per key:
      month_start_share, mid_month_share, month_end_share  the shares of its payment days in
                       MONTH_START_DAYS, in MID_MONTH_DAYS and in the last MONTH_END_LAST_DAYS
                       days of the month
      median_gap_days  the median gap between its payment days
      gap_iqr_days     the 75th minus the 25th percentile of the gaps: small for a regular
                       payer, large for an erratic one
      payment_due_ratio  days from its last payment day to `cutoff_day`, over median_gap_days:
                       about 1, a payment is due; above 1, it is overdue for one
    """
    columns = ["key", "month_start_share", "mid_month_share", "month_end_share",
               "median_gap_days", "gap_iqr_days", "payment_due_ratio"]
    if samples.empty:
        empty = pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})
        empty["key"] = empty["key"].astype(object)
        return empty
    samples = samples.copy()
    first, last = MONTH_START_DAYS
    samples["at_month_start"] = samples["day_of_month"].between(first, last)
    first, last = MID_MONTH_DAYS
    samples["at_mid_month"] = samples["day_of_month"].between(first, last)
    samples["at_month_end"] = samples["place"].between(-MONTH_END_LAST_DAYS, -1)

    by_key = samples.groupby("key")
    stats = pd.DataFrame({
        "month_start_share": by_key["at_month_start"].mean(),
        "mid_month_share": by_key["at_mid_month"].mean(),
        "month_end_share": by_key["at_month_end"].mean(),
        "median_gap_days": by_key["gap"].median(),
        "gap_iqr_days": by_key["gap"].quantile(0.75) - by_key["gap"].quantile(0.25),
        "last_day": by_key["day"].max(),
    })
    days_since_last = cutoff_day - stats["last_day"]
    stats["payment_due_ratio"] = days_since_last / stats["median_gap_days"]
    stats.index.name = "key"
    stats = stats.reset_index()
    return stats[columns]


def week_dates(week: int, week_zero: date) -> pd.Series:
    """The seven dates of `week`, Monday to Sunday, as timestamps."""
    monday = week_zero + timedelta(days=7 * week)
    return pd.Series(pd.date_range(monday, periods=7, freq="D"))


def business_weekdays(dates: pd.Series, holiday_days: dict) -> list[int]:
    """The weekdays (0 Monday ... 4 Friday) of the business days among `dates`: Monday to
    Friday and not in `holiday_days` (dates as keys, from `public_holidays`)."""
    weekdays = []
    for day in dates:
        if day.weekday() < 5 and day.date() not in holiday_days:
            weekdays.append(day.weekday())
    return weekdays


def target_week_habits(samples: pd.DataFrame, origin: int, week_zero: date,
                       holiday_days: dict, horizons: int) -> pd.DataFrame:
    """Per key of `samples` (`payment_samples_at`), for each week k = 1 ... horizons after
    `origin`, one row per key:
      month_habit_week_k    the share of its payment days whose place in the month
                            (`month_place`) is the place of one of that week's seven days
      weekday_habit_week_k  the share whose weekday is a business day of that week (below 1
                            only when a holiday takes its usual weekday)
    """
    table = pd.DataFrame({"key": samples["key"].drop_duplicates().to_numpy()})
    samples = samples.copy()
    for k in range(1, horizons + 1):
        dates = week_dates(origin + k, week_zero)
        places = list(month_place(dates))
        weekdays = business_weekdays(dates, holiday_days)
        samples["in_month_habit"] = samples["place"].isin(places)
        samples["on_weekday_habit"] = samples["weekday"].isin(weekdays)
        shares = samples.groupby("key")[["in_month_habit", "on_weekday_habit"]].mean()
        shares = shares.rename(columns={"in_month_habit": f"month_habit_week_{k}",
                                        "on_weekday_habit": f"weekday_habit_week_{k}"})
        table = add_per_key(table, shares.reset_index(), fill=float("nan"))
    return table


def next_payment_shares(samples: pd.DataFrame, origin: int, next_business: dict[int, int],
                        horizons: int) -> pd.DataFrame:
    """Per key of `samples` (`payment_samples_at`), for each week k = 1 ... horizons after
    `origin`: next_payment_week_k, the share of its possible next payments that land in that
    week. One row per key.

    A possible next payment is the key's last payment day plus one of its past gaps, moved to
    the next business day (`next_business`). No payment came by t, so only those after t can
    happen; each of them counts the same. A key with none after t gets 0 in every week.
    """
    table = pd.DataFrame({"key": samples["key"].drop_duplicates().to_numpy()})

    # Every (key, past gap): the day of the next payment if that gap came again.
    last_day = samples.groupby("key")["day"].max().rename("last_day").reset_index()
    gaps = samples.loc[samples["gap"].notna(), ["key", "gap"]]
    candidates = gaps.merge(last_day, on="key", how="inner")
    next_day = candidates["last_day"] + candidates["gap"].astype("int64")
    next_day = to_next_business_day(next_day, next_business)
    candidates["week"] = next_day // 7 - origin

    # Only those after t, in equal shares.
    possible = candidates[candidates["week"] >= 1]
    counts = possible.groupby("key").size().rename("possible_count").reset_index()
    possible = possible.merge(counts, on="key", how="left")
    possible["share"] = 1.0 / possible["possible_count"]

    for k in range(1, horizons + 1):
        in_week = possible[possible["week"] == k]
        shares = in_week.groupby("key")["share"].sum()
        shares = shares.rename(f"next_payment_week_{k}").reset_index()
        table = add_per_key(table, shares, fill=0.0)
    return table


def habits_at(days: pd.DataFrame, keys: pd.Series, origin: int, week_zero: date,
              holiday_days: dict, next_business: dict[int, int], horizons: int,
              payment_days_kept: int, min_payment_days: int) -> pd.DataFrame:
    """The payment habits of every key in `keys` at the end of week `origin`: one row per
    key, columns `key_habit_columns`.

    Each key uses its last `payment_days_kept` payment days of `days` (`payment_days`) known
    by then; payment_days_used says how many. A key with at least `min_payment_days` of them
    gets its habits (`habit_statistics`, `target_week_habits`, `next_payment_shares`); the
    others get NaN. Raises ValueError if a key is repeated in `keys`.
    """
    repeated = keys.duplicated()
    if repeated.any():
        raise ValueError(f"keys are repeated (first: {keys[repeated].head(5).tolist()})")
    cutoff_day = (end_of_week(week_zero, origin) - week_zero).days
    samples = payment_samples_at(days, keys, cutoff_day, payment_days_kept)

    # 1. One row per key, with how many payment days it uses.
    table = pd.DataFrame({"key": keys.to_numpy()})
    table["origin_week"] = origin
    used = samples.groupby("key").size().rename("payment_days_used").reset_index()
    table = add_per_key(table, used, fill=0)

    # 2. The habits, only for keys with enough payment days.
    enough = used.loc[used["payment_days_used"] >= min_payment_days, "key"]
    habit = samples[samples["key"].isin(enough)]
    statistics = habit_statistics(habit, cutoff_day)
    table = add_per_key(table, statistics, fill=float("nan"))
    in_target_week = target_week_habits(habit, origin, week_zero, holiday_days, horizons)
    table = add_per_key(table, in_target_week, fill=float("nan"))
    next_payment = next_payment_shares(habit, origin, next_business, horizons)
    table = add_per_key(table, next_payment, fill=float("nan"))

    # Every column, also when no key has a habit.
    for column in key_habit_columns(horizons):
        if column not in table.columns:
            table[column] = float("nan")
    table["payment_days_used"] = table["payment_days_used"].astype("int64")
    return table[key_habit_columns(horizons)]


def payment_habits(panel_rows: pd.DataFrame, collections: pd.DataFrame, ledger: pd.DataFrame,
                   week_zero: date, holidays: HolidayTable, first_file_day: date,
                   use_ledger_before_first_file: bool, horizons: int, payment_days_kept: int,
                   min_payment_days: int,
                   leak_check_weeks: tuple[int, ...]) -> pd.DataFrame:
    """The payment-habit features of every row of `panel_rows` (columns key, origin_week, h):
    one row per panel row, in its order, columns key, origin_week, h and `HABIT_COLUMNS`.

      payment_days_used             how many past payment days the habits use (at most
                                    `payment_days_kept`)
      month_start_share, mid_month_share, month_end_share
                                    the shares of them at the start, middle and end of the
                                    month
      median_gap_days, gap_iqr_days the usual gap between payment days, and how much it varies
      payment_due_ratio             days since the last payment over median_gap_days
      month_habit_in_target         the share whose place in the month falls in week t+h
      weekday_habit_in_target       the share whose weekday is a business day of week t+h
      next_payment_share_in_target  the share of the possible next payments (last payment
                                    day + a past gap, after t) that land in week t+h
    All from the payment days (`payment_days`: the daily files, and the ledger before the
    first file on `first_file_day` if `use_ledger_before_first_file`) known by the end of
    week t. A key with fewer than
    `min_payment_days` has NaN in all but payment_days_used. None is an amount, so nothing is
    deflated.

    The business days come from `holidays`. Each week of `leak_check_weeks` among the panel's
    origins is rebuilt from the payment lines dated by its end and the ledger as known then
    (`ledger_as_known_at`); a difference raises ValueError. Raises ValueError too if
    `min_payment_days` is below 2 (a gap needs two payment days).
    """
    if min_payment_days < 2:
        raise ValueError(f"min_payment_days is {min_payment_days}; it must be 2 or more")
    days = payment_days(collections, ledger, first_file_day, week_zero,
                        use_ledger_before_first_file)
    origins = sorted(set(panel_rows["origin_week"]))
    start = end_of_week(week_zero, origins[0]) - timedelta(days=CALENDAR_MARGIN_DAYS)
    end = end_of_week(week_zero, origins[-1] + horizons) + timedelta(days=CALENDAR_MARGIN_DAYS)
    next_business = next_business_days(start, end, holidays, week_zero)
    holiday_days = public_holidays(start, end, holidays)

    def habits_for(origin, days_then):
        keys = panel_rows.loc[panel_rows["origin_week"] == origin, "key"].drop_duplicates()
        return habits_at(days_then, keys, origin, week_zero, holiday_days, next_business,
                         horizons, payment_days_kept, min_payment_days)

    # 1. The habits per key at every origin.
    tables = []
    for number, origin in enumerate(origins, start=1):
        tables.append(habits_for(origin, days))
        if number % 10 == 0:
            print(f"  ... payment habits for {number} of {len(origins)} origins")
    per_key = pd.concat(tables, ignore_index=True)

    # 2. The leak check: rebuilt from the payment lines and the ledger as they were at the week.
    for week in leak_check_weeks:
        if week not in origins:
            continue
        cutoff = end_of_week(week_zero, week)
        dated_by_then = pd.to_datetime(collections["document_date"]) <= pd.Timestamp(cutoff)
        lines_then = collections[dated_by_then]
        ledger_then = ledger_as_known_at(ledger, cutoff)
        days_then = payment_days(lines_then, ledger_then, first_file_day, week_zero,
                                 use_ledger_before_first_file)
        rebuilt = habits_for(week, days_then)
        stored = per_key[per_key["origin_week"] == week].reset_index(drop=True)
        if not rebuilt.equals(stored):
            raise ValueError(f"leak check: the payment habits of week {week} differ when rebuilt "
                             f"from data cut off at {cutoff}, so something from after the week "
                             f"was used")

    # 3. Onto the panel rows: each row takes the week after t that is its own h.
    rows = panel_rows[["key", "origin_week", "h"]].merge(
        per_key, on=["key", "origin_week"], how="left", validate="many_to_one")
    features = rows[["key", "origin_week", "h"]].copy()
    per_key_columns = ["payment_days_used", "month_start_share", "mid_month_share",
                       "month_end_share", "median_gap_days", "gap_iqr_days",
                       "payment_due_ratio"]
    for column in per_key_columns:
        features[column] = rows[column]
    features["month_habit_in_target"] = float("nan")
    features["weekday_habit_in_target"] = float("nan")
    features["next_payment_share_in_target"] = float("nan")
    for h in range(1, horizons + 1):
        at_h = rows["h"] == h
        month_habit = rows.loc[at_h, f"month_habit_week_{h}"]
        weekday_habit = rows.loc[at_h, f"weekday_habit_week_{h}"]
        next_payment = rows.loc[at_h, f"next_payment_week_{h}"]
        features.loc[at_h, "month_habit_in_target"] = month_habit
        features.loc[at_h, "weekday_habit_in_target"] = weekday_habit
        features.loc[at_h, "next_payment_share_in_target"] = next_payment
    features["payment_days_used"] = features["payment_days_used"].astype("int64")
    features = features[["key", "origin_week", "h"] + HABIT_COLUMNS]
    return features.reset_index(drop=True)
