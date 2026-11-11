"""Printouts of steps 1-11 (`inputs.prepare_inputs`): the daily files, the ledger, the
weeks, the receivables, the panel and its side tables, the price levels, the features.

`show_preparation` prints them all in the pipeline's order. What is only there to be
looked at is made in it, not in `prepare_inputs`: step 2 (one day's file read on its
own) and the copied inflation module's own figures beside the levels used (step 10).
"""
from datetime import timedelta

import pandas as pd

from .. import inflation
from ..features_receivables import before_first_ledger_week
from ..holidays import public_holidays
from ..prices import price_weeks, projected_levels, published_month
from ..read_collections import read_one_file
from ..receivables_asof import overdue_age_columns


def show_preparation(inputs, details, settings):
    """The printouts of steps 1-11, in their order, from what `prepare_inputs` made (`inputs`,
    `details`). What is only there to be looked at is made here, not there: step 2 (one day's
    file read on its own) and the copied inflation module's own figures beside the levels
    used (step 10)."""
    show_daily_files(details.daily_files)

    # Step 2: the payment lines of one day, checked against the file's grand total.
    day = settings.debug.one_day
    if day not in details.daily_files:
        raise ValueError(f"{settings.settings_file.name} [debug]: one_day {day} has no daily file")
    show_one_day(day, read_one_file(details.daily_files[day], settings.collections_layout))

    show_collections(details.collections_keyed)   # step 3 (adding the keys keeps every line)
    show_keys(details.collections_keyed)
    show_ledger(inputs.ledger)
    show_tie(details.tie)
    show_filled(details.collections, details.filled_days,
                settings.data.fill_missing_days_from_ledger)
    print(f"Step 7: week 0 starts {inputs.week_zero}; lines run from week "
          f"{details.collections['week'].min()} to week {details.collections['week'].max()}")
    show_weeks(inputs.weeks)
    show_weekly(inputs.weekly)

    show_rates(inputs.rates)
    show_receivables_by_week(details.receivables, settings.receivables.leak_check_weeks,
                             settings.debug.sample_week, settings.receivables.overdue_age_days)

    show_known(details.known, details.collections, inputs.last_week)
    show_panel_rows(details.panel)   # the labels keep every row and its order
    show_labels(details.panel, details.identified_weekly)
    show_new_customer_cash(inputs.new_cash, inputs.weekly)
    show_refunds(inputs.refunds)
    print(f"        unidentified cash (G): {len(inputs.unidentified):,} rows, "
          f"{inputs.unidentified['unidentified_cash'].sum():,.2f} TRY from "
          f"{len(settings.companies.unidentified_customers)} placeholder customers")
    print(f"        tie-out passed: {details.checked:,} company-week-horizon checks "
          f"(panel + A + G = gross receipts; R = refunds)")

    # Step 10: the copied inflation.py called directly, so it can be followed line by line;
    # prices.py wraps it with our week numbers and a coverage guard (the levels used).
    weeks_for_prices = price_weeks(inputs.week_zero, details.first_price_week, inputs.last_week)
    actual_level = inflation.weekly_price_level(weeks_for_prices)        # hindsight only
    published_level = inflation.published_price_level(weeks_for_prices)  # known at the time
    projected_level = inflation.projected_price_level(weeks_for_prices)  # P[origin, target]
    show_prices(weeks_for_prices, actual_level, published_level, projected_level, inputs.levels)
    # U13.1: the levels a forecast re-inflates with, at the demo origin (a past week, so the
    # published figures can be checked) and at the last week (target weeks after the data).
    for forecast_week in (settings.debug.demo_origin, inputs.last_week):
        show_projected_levels(inputs.week_zero, forecast_week, settings.receivables.horizons)

    show_early_receipts(details.early_receipts, details.history_start)
    show_matrix(details.matrix)
    show_history(details.history, details.matrix)
    show_calendar(inputs.calendar, inputs.week_zero, details.holidays)
    show_holiday_positions(details.positions, inputs.week_zero, details.holidays)
    show_owed(details.owed, int(before_first_ledger_week(details.panel,
                                                         details.receivables).sum()))
    show_deflated_label(details.panel, inputs.levels)
    show_table(inputs.table, details.panel)


def show_daily_files(daily_files):
    """A short printout of step 1."""
    dates = list(daily_files)
    print(f"Step 1: {len(dates)} daily files, from {dates[0]} to {dates[-1]}")
    print(f"        first file: {daily_files[dates[0]].name}")


def show_one_day(day, lines):
    """A short printout of step 2."""
    print(f"Step 2: {day}: {len(lines)} payment lines, "
          f"{lines['amount_try'].sum():,.2f} TRY in total (ties to the file's grand total)")
    print(lines[["company_code", "customer", "amount_try", "currency_original"]].head())


def show_collections(collections):
    """A short printout of step 3."""
    print(f"Step 3: {len(collections):,} payment lines, "
          f"{collections['amount_try'].sum():,.2f} TRY in total, "
          f"{collections['document_date'].nunique()} days")


def show_keys(collections):
    """A short printout of step 4."""
    keys_per_company = collections.groupby("company_code")["key"].nunique()
    print(f"Step 4: {collections['key'].nunique():,} keys; per company: "
          f"{keys_per_company.to_dict()}")


def show_ledger(ledger):
    """A short printout of the ledger."""
    lines_per_type = ledger["document_role"].value_counts().to_dict()
    still_open = ledger["clearing_date"].isna().sum()
    print(f"Step 5: {len(ledger):,} ledger lines {lines_per_type}, {still_open:,} still open; "
          f"{ledger['key'].nunique():,} keys; posted from "
          f"{ledger['posting_date'].min():%Y-%m-%d} to {ledger['posting_date'].max():%Y-%m-%d}")


def show_tie(tie):
    """A short printout of the ledger-to-daily-files check."""
    print(f"        ledger payments tie to the daily files on all {tie['days_compared']} days "
          f"with a file; also in the ledger: {tie['days_before_first_file']} days before the "
          f"first file ({tie['try_before_first_file']:,.2f} TRY) and "
          f"{tie['missing_file_days_in_ledger']} missing-file days "
          f"({tie['try_on_missing_file_days']:,.2f} TRY) and {tie['days_after_last_file']} days "
          f"after the last file ({tie['try_after_last_file']:,.2f} TRY)")


def show_filled(collections, filled_days, switched_on):
    """A short printout of the filling of days without a daily file."""
    if not switched_on:
        print("Step 6: filling from the ledger is switched off; weeks with a missing day stay "
              "missing")
        return
    filled = collections[collections["filled_from_ledger"]]
    print(f"Step 6: {len(filled_days)} days without a daily file filled from the ledger: "
          f"{len(filled):,} payment lines, {filled['amount_try'].sum():,.2f} TRY "
          f"({', '.join(str(day) for day in filled_days)})")


def show_weeks(weeks):
    """A short printout of the week calendar."""
    missing = weeks[weeks["is_missing"]]
    print(f"        {len(weeks)} weeks, {len(missing)} missing:")
    for row in missing.itertuples():
        print(f"          week {row.week} (Monday {row.monday}): "
              f"{row.missing_days} of {row.business_days} business days without a file")


def show_weekly(weekly):
    """A short printout of the weekly amounts."""
    paid = weekly["gross_receipts"] > 0
    print(f"        weekly amounts: {len(weekly):,} key-weeks with a line "
          f"({paid.sum():,} with a payment); "
          f"gross receipts {weekly['gross_receipts'].sum():,.2f} TRY, "
          f"refunds {weekly['refunds'].sum():,.2f} TRY, "
          f"{weekly['unapplied_lines'].sum():,} unapplied lines")


def show_rates(rates):
    """A short printout of the exchange rates."""
    print(f"Step 8: exchange rates for {rates['currency'].nunique()} currencies:")
    for currency, rows in rates.groupby("currency"):
        print(f"          {currency}: {len(rows)} days from {rows['day'].min()} to "
              f"{rows['day'].max()}, {rows['rate'].min():.2f} to {rows['rate'].max():.2f} TRY")


def show_receivables_by_week(receivables, leak_check_weeks, sample_week, age_edges):
    """A short printout of the receivables of every week."""
    print(f"        receivables: {len(receivables):,} rows, {receivables['key'].nunique():,} "
          f"keys, weeks {receivables['week'].min()} to {receivables['week'].max()}; "
          f"leak check passed for weeks {list(leak_check_weeks)}")
    columns = ["open_balance", "overdue", "due_in_week_1", "unapplied_cash"]
    totals = receivables.groupby("week")[columns].sum() / 1e6
    every_eighth_week = totals.loc[totals.index % 8 == 0]
    print("        totals per week, million TRY (every 8th week):")
    print(every_eighth_week.round(1).to_string())
    one_week = receivables[receivables["week"] == sample_week]
    print(f"        week {sample_week}: {len(one_week):,} keys, "
          f"{(one_week['open_invoices'] > 0).sum():,} with an open invoice, "
          f"{one_week['usual_delay_days'].notna().sum():,} with a usual delay")
    ages = overdue_age_columns(age_edges)
    aging = receivables.groupby("week")[["overdue"] + ages].sum() / 1e6
    print("        overdue by days past due, million TRY (every 8th week):")
    print(aging.loc[aging.index % 8 == 0].round(1).to_string())


def show_known(known, collections, last_week):
    """A short printout of when keys become known."""
    before = (known["known_week"] < 0).sum()
    during = known["known_week"].between(0, last_week)
    print(f"Step 9: {len(known):,} keys; {before:,} known before week 0, {during.sum():,} "
          f"first seen in weeks 0-{last_week} (about {during.sum() / (last_week + 1):.0f} a week)")
    print(f"        first seen in: {known['first_seen_in'].value_counts().to_dict()}")
    first_payment = collections.groupby("key")["document_date"].min()
    gap = (first_payment - known.set_index("key")["first_seen"].reindex(first_payment.index))
    print(f"        from first interaction to first payment in the daily files: median "
          f"{gap.dt.days.median():.0f} days")


def show_panel_rows(panel):
    """A short printout of the panel's rows."""
    per_origin = panel.groupby("origin_week")["key"].nunique()
    print(f"        panel rows: {len(panel):,} (keys x forecast weeks {panel['origin_week'].min()}"
          f"-{panel['origin_week'].max()} x horizons 1-{panel['h'].max()}); known keys per "
          f"forecast week from {per_origin.iloc[0]:,} to {per_origin.iloc[-1]:,}")


def show_labels(panel, weekly):
    """A short printout of the panel's labels."""
    print(f"        label status: {panel['label_status'].value_counts().to_dict()}")
    labelled = panel[panel["label_status"] == "labelled"]
    paid = labelled[labelled["paid"] == True]  # noqa: E712 (paid is a nullable boolean)
    print(f"        labelled rows with a payment: {len(paid):,} of {len(labelled):,} "
          f"({len(paid) / len(labelled):.1%}); median amount when paid "
          f"{paid['amount'].median():,.0f} TRY")
    # Cash in the target week from keys not yet known at the origin: the new-customer allowance.
    one_horizon = labelled[labelled["h"] == 1]
    in_panel = one_horizon.groupby("target_week")["amount"].sum()
    receipts = weekly.groupby("week")["gross_receipts"].sum().reindex(in_panel.index)
    print(f"        at h=1, the panel holds {in_panel.sum() / receipts.sum():.2%} of the target "
          f"weeks' receipts; the rest is from keys not yet known (the allowance, U5)")


def show_new_customer_cash(new_cash, weekly):
    """A short printout of table A: its share of the target weeks' cash, by horizon."""
    week_receipts = weekly.groupby("week")["gross_receipts"].sum()
    per_h = new_cash.groupby("h")["new_customer_cash"].sum()
    # Each (origin, h) points at one target week: its cash, all companies together.
    one_per_target = new_cash.drop_duplicates(["origin_week", "h"]).copy()
    one_per_target["week_cash"] = one_per_target["target_week"].map(week_receipts)
    cash_per_h = one_per_target.groupby("h")["week_cash"].sum()
    shares = {h: f"{per_h[h] / cash_per_h[h]:.2%}" for h in per_h.index}
    print(f"        new-customer cash (A): {len(new_cash):,} rows; share of the target weeks' "
          f"cash by h: {shares}")


def show_refunds(refunds):
    """A short printout of table R."""
    weeks_with_refunds = refunds.loc[refunds["refunds"] < 0, "week"].nunique()
    per_company = refunds.groupby("company_code")["refunds"].sum().round().to_dict()
    print(f"        refunds (R): {len(refunds):,} rows, {refunds['refunds'].sum():,.2f} TRY, "
          f"in {weeks_with_refunds} of {refunds['week'].nunique()} weeks; per company: "
          f"{per_company}")


def show_prices(weeks_for_prices, actual_level, published_level, projected_level, levels):
    """A short printout of the price levels."""
    first, last = weeks_for_prices["week"].iloc[0], weeks_for_prices["week"].iloc[-1]
    print(f"Step 10: price levels for weeks {first} to {last} "
          f"({weeks_for_prices['week_start'].iloc[0]:%Y-%m-%d} to "
          f"{weeks_for_prices['week_end'].iloc[-1]:%Y-%m-%d}); the copied module gives 1.0 = the "
          f"last week, published_levels() rescales to 1.0 = week 0")
    table = pd.DataFrame({"week": weeks_for_prices["week"], "actual": actual_level,
                          "published": published_level}).set_index("week")
    print(table.iloc[::13].round(3).to_string())
    # One week ahead, projected vs actual: how much the projection misses on average.
    n = len(actual_level)
    one_ahead = [projected_level[origin, origin + 1] / actual_level[origin + 1] - 1
                 for origin in range(n - 1)]
    # published_levels() is the copied module's published level, rescaled to week 0.
    at_week_zero = published_level[list(weeks_for_prices["week"]).index(0)]
    print(f"        projected vs actual level one week ahead: mean error "
          f"{sum(one_ahead) / len(one_ahead):+.2%}; published_levels() matches: "
          f"{bool((abs(levels.to_numpy() - published_level / at_week_zero) < 1e-12).all())}; "
          f"week 0 = {levels.loc[0]:.3f}, last week = {levels.iloc[-1]:.3f}")


def show_projected_levels(week_zero, origin, horizons):
    """A short printout of the price levels a forecast at `origin` re-inflates with: per
    target week, the TÜFE month it needs, looked up or projected, and (where the table has
    that month) the published level in hindsight."""
    targets = [origin + h for h in range(1, horizons + 1)]
    projected = projected_levels(week_zero, origin, targets)
    known = published_month(week_zero + timedelta(days=7 * origin))
    base = inflation.TUFE[published_month(week_zero)]
    rows = []
    for week in targets:
        needed = published_month(week_zero + timedelta(days=7 * week))
        hindsight = inflation.TUFE.get(needed)
        rows.append({"target_week": week,
                     "starts": f"{week_zero + timedelta(days=7 * week):%Y-%m-%d}",
                     "TÜFE month": f"{needed[0]}-{needed[1]:02d}",
                     "how": "looked up" if needed <= known else "projected",
                     "level": round(projected[week], 4),
                     "hindsight": round(hindsight / base, 4) if hindsight else None})
    table = pd.DataFrame(rows).set_index("target_week")
    print(f"        projected levels for a forecast at week {origin} (TÜFE published by then: "
          f"{known[0]}-{known[1]:02d})")
    print(table.to_string())


def show_early_receipts(early_receipts, history_start):
    """A short printout of the payment history taken from the ledger."""
    if early_receipts.empty:
        print("Step 11: no payment history before the first daily file")
        return
    print(f"Step 11: payment history from the ledger: weeks {history_start} to "
          f"{early_receipts['week'].max()}, {len(early_receipts):,} key-weeks, "
          f"{early_receipts['key'].nunique():,} keys, "
          f"{early_receipts['gross_receipts'].sum():,.2f} TRY (nominal)")


def show_matrix(matrix):
    """A short printout of the weekly matrix."""
    before = [week for week in matrix.columns if week < 0]
    after = [week for week in matrix.columns if week >= 0]
    print(f"        weekly matrix: {matrix.shape[0]:,} keys x {matrix.shape[1]} weeks "
          f"({matrix.columns.min()} to {matrix.columns.max()}); cells with no data: "
          f"{matrix[before].isna().mean().mean() if before else 0:.1%} before week 0, "
          f"{matrix[after].isna().mean().mean():.1%} from week 0")
    busiest = matrix.sum(axis=1).idxmax()
    paid_weeks = [week for week in matrix.columns if matrix.loc[busiest, week] > 0][:6]
    print(f"        the key with the most receipts, its first weeks with a payment "
          f"(TRY of week 0):")
    print(matrix.loc[[busiest], paid_weeks].round(0).to_string())


def show_history(history, matrix):
    """A short printout of the history features."""
    features = [column for column in history.columns if column not in ("key", "origin_week")]
    print(f"        history features: {len(history):,} rows (key x origin week), "
          f"{len(features)} features")
    summary = pd.DataFrame({"mean": history[features].mean(),
                            "share NaN": history[features].isna().mean()})
    print(summary.round(3).to_string())
    busiest = matrix.sum(axis=1).idxmax()
    sample = history[(history["key"] == busiest) & history["origin_week"].isin([0, 13, 52])]
    print(f"        the key with the most receipts at origin weeks 0, 13 and 52:")
    print(sample.round(0).T.to_string(header=False))


def show_calendar(calendar, week_zero, holidays):
    """A short printout of the calendar features."""
    print(f"        calendar features: weeks {calendar['week'].min()} to "
          f"{calendar['week'].max()}; weeks with a month-end: "
          f"{calendar['has_month_end'].sum()}, a quarter-end: "
          f"{calendar['has_quarter_end'].sum()}, the 15th: {calendar['has_mid_month'].sum()}")
    short = calendar[calendar["business_days"] < 5]
    print(f"        weeks with fewer than 5 business days:")
    for row in short.itertuples():
        monday = week_zero + timedelta(days=7 * row.week)
        names = sorted(set(public_holidays(monday, monday + timedelta(days=4),
                                           holidays).values()))
        print(f"          week {row.week} (Monday {monday}): {row.business_days} business days "
              f"({', '.join(names)})")
    quarter_ends = [week_zero + timedelta(days=7 * week)
                    for week in calendar.loc[calendar["has_quarter_end"], "week"]]
    print(f"        quarter-end weeks start on: {', '.join(str(day) for day in quarter_ends)}")
    after = calendar[calendar["holidays_week_before"] > 0]
    print(f"        weeks after a holiday week (holidays_week_before): "
          f"{dict(zip(after['week'], after['holidays_week_before']))}")


def show_holiday_positions(positions, week_zero, holidays):
    """A short printout of the holiday-position columns: every week where one differs from a
    normal week (drain and catch-up 0, every weekday a business day), with its holidays."""
    columns = [column for column in positions.columns if column != "week"]
    if not columns:
        print("        holiday positions: none ([calendar] holiday_encodings is empty)")
        return
    normal = {column: 0 if column in ("drain_days", "catch_up_days") else 1
              for column in columns}
    unusual = positions[(positions[columns] != pd.Series(normal)).any(axis=1)]
    print(f"        holiday positions: {len(columns)} columns; {len(unusual)} of "
          f"{len(positions)} weeks differ from a normal week:")
    shown = [column for column in ("drain_days", "catch_up_days") if column in columns]
    for row in unusual.itertuples(index=False):
        monday = week_zero + timedelta(days=7 * row.week)
        nearby = public_holidays(monday - timedelta(days=7), monday + timedelta(days=13),
                                 holidays)
        # Sorted as dates, then written as text ("Fri 29 May" would sort before "Tue 19 May").
        names = [f"{day:%a %d %b}" for day in sorted(nearby) if day.weekday() < 5]
        values = ", ".join(f"{column} {getattr(row, column)}" for column in shown)
        print(f"          week {row.week} (Monday {monday}): {values}  "
              f"[weekday holidays in weeks {row.week - 1}-{row.week + 1}: {', '.join(names)}]")


def show_owed(owed, not_yet_in_ledger):
    """A short printout of the receivables features."""
    features = [column for column in owed.columns if column not in ("key", "origin_week", "h")]
    print(f"        receivables features: {len(owed):,} rows (key x origin week x h), "
          f"{len(features)} features; the due-date split adds up in every row")
    print(f"        rows of keys not yet in the ledger (filled as nothing owed): "
          f"{not_yet_in_ledger:,}")
    summary = pd.DataFrame({"mean": owed[features].mean(),
                            "share NaN": owed[features].isna().mean()})
    print(summary.round(3).to_string())
    due_by_h = owed.groupby("h")[["due_before_target_week", "due_in_target_week",
                                  "due_after_target_week"]].sum() / 1e9
    print("        due before / in / after the target week, by h (billion TRY of week 0, "
          "summed over rows):")
    print(due_by_h.round(2).to_string())


def show_deflated_label(panel, levels):
    """A short printout of the deflated label."""
    labelled = panel[panel["label_status"] == "labelled"]
    print(f"        deflated label: {labelled['amount_deflated'].notna().sum():,} labelled rows, "
          f"{panel['amount_deflated'].isna().sum():,} NaN (rows not labelled: "
          f"{(panel['label_status'] != 'labelled').sum():,})")
    one_horizon = labelled[labelled["h"] == 1]
    per_week = one_horizon.groupby("target_week")[["amount", "amount_deflated"]].sum()
    per_week["ratio"] = per_week["amount"] / per_week["amount_deflated"]
    per_week["level"] = per_week.index.map(levels)
    print("        at h=1, label per target week, million TRY (nominal / of week 0), every 16th:")
    print((per_week.loc[per_week.index % 16 == 0] / [1e6, 1e6, 1, 1]).round(3).to_string())


def show_table(table, panel):
    """A short printout of the training table."""
    print(f"        training table: {len(table):,} rows (panel: {len(panel):,}), "
          f"{table.shape[1]} columns, about {table.memory_usage(deep=True).sum() / 1e6:,.0f} MB")
    with_nan = table.isna().mean()
    with_nan = with_nan[with_nan > 0].sort_values(ascending=False)
    print("        share NaN, columns that have any:")
    print(with_nan.round(3).to_string())
    paid_rows = table[(table["label_status"] == "labelled") & table["paid"].fillna(False)]
    if paid_rows.empty:
        print("        no labelled, paid row to show")
        return
    print("        one labelled, paid row:")
    print(paid_rows.iloc[0].to_string())
