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
