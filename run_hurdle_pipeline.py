"""Run the hurdle model (option B), step by step.

Open this file in PyCharm and press Run (or Debug). There are no command-line arguments:
the settings come from a TOML file (below), and each step is one line in `main()`, so you can
put a breakpoint after any step and look at what it produced.

Steps (filled in as the units are built; see option_b_implementation_plan.md):

    1  daily files     which days have a collections file          (U0)   <- built
    2  one day         the payment lines of one chosen file        (U0)   <- built
    3  collections     the payment lines of every file             (U0)   <- built
    4  keys            company + customer as one key               (U1)   <- built
    5  AR ledger       the ledger lines and their tie to the daily files (U9a)   <- built
    6  missing days    days without a file, filled from the ledger (or left missing) <- built
    7  weekly series   receipts per key and week, missing weeks    (U2)   <- built
    8  receivables     each key's receivables as known at each week (U9b)    <- built
    9  panel           rows customer x week x horizon, with labels (U3, U4, U5)  <- in progress
   10  prices          TÜFE price levels by week (copied inflation.py, U8)   <- built
   11  features        history, calendar, receivables (U6, U7, U9c)   <- in progress
   12  models          stage 1, stage 2, top-ups                   (U10-U12)
   13  forecast, evaluation, calibration                           (U13-U15)

The settings (data paths, company codes, ...) are in settings/synthetic.toml. To run on
other data (e.g. the real EY files), write another settings file and point the environment
variable HURDLE_SETTINGS to it; nothing in the code needs to change.
"""
import itertools
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# The package lives under src/. Adding it here lets this script run from PyCharm or a
# terminal without installing the project.
sys.path.insert(0, str(REPO_ROOT / "src"))

# Customer names are Turkish; a Windows console would otherwise crash printing "İ".
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

from collection_estimation.hurdle.read_collections import (  # noqa: E402
    list_daily_files,
    load_collections,
    read_one_file,
)
from collection_estimation.hurdle.fill_from_ledger import fill_missing_days  # noqa: E402
from collection_estimation.hurdle.features_calendar import (  # noqa: E402
    calendar_features,
    holiday_position_features,
)
from collection_estimation.hurdle.features_receivables import (  # noqa: E402
    before_first_ledger_week,
    receivables_features,
)
from collection_estimation.hurdle.features_history import (  # noqa: E402
    first_history_week,
    history_features,
    ledger_receipts_before,
    weekly_matrix,
)
from collection_estimation.hurdle.holidays import load_holidays, public_holidays  # noqa: E402
from collection_estimation.hurdle import inflation  # noqa: E402
from collection_estimation.hurdle.keys import add_keys  # noqa: E402
from collection_estimation.hurdle.known import known_weeks  # noqa: E402
from collection_estimation.hurdle.panel import (  # noqa: E402
    add_deflated_label,
    add_labels,
    panel_rows,
)
from collection_estimation.hurdle.benchmark import (  # noqa: E402
    benchmark_forecasts,
    cash_totals,
)
from collection_estimation.hurdle.diagnostic_joint import (  # noqa: E402
    paired_bootstrap,
    score_pairs,
    scores_by_h,
    stage1_gates,
    stage1_grid,
    stage2_gates,
    stage2_grid,
)
from collection_estimation.hurdle.diagnostic import (  # noqa: E402
    amount_reliability,
    apply_platt,
    best_unweighted,
    calibration_comparison,
    calibration_summary,
    cash_by_p_group,
    choose_and_test,
    class_weight_argument,
    diagnostic_splits,
    fit_platt,
    grid_search,
    hgb_candidates,
    lr_candidates,
    permutation_importances,
    reference_probabilities,
    write_tables,
)
from collection_estimation.hurdle.stage1 import (  # noqa: E402
    feature_columns,
    fit_stage1,
    fit_stage1_baseline,
    predict_paid,
    stage1_features,
)
from collection_estimation.hurdle.stage2 import (  # noqa: E402
    amount_references,
    amount_scores,
    exposure,
    fit_stage2,
    predict_amount,
    total_ratio,
)
from collection_estimation.hurdle.training_table import (  # noqa: E402
    block_feature_columns,
    join_features,
    rows_at_origin,
    rows_known_at,
)
from collection_estimation.hurdle.topups import score_topups, topup_actuals, topup_forecasts  # noqa: E402
from collection_estimation.hurdle.side_tables import (  # noqa: E402
    check_side_tables,
    is_unidentified,
    new_customer_cash,
    refunds_by_week,
    unidentified_cash_by_week,
)
from collection_estimation.hurdle.receivables_asof import (  # noqa: E402
    exchange_rates,
    overdue_age_columns,
    receivables_by_week,
)
from collection_estimation.hurdle.read_ledger import (  # noqa: E402
    check_ledger_against_collections,
    load_ledger,
)
from collection_estimation.hurdle.prices import (  # noqa: E402
    price_weeks, projected_levels, published_levels, published_month)
from collection_estimation.hurdle.settings import load_settings  # noqa: E402
from collection_estimation.hurdle.series import (  # noqa: E402
    add_week_index,
    week_numbers,
    week_calendar,
    weekly_amounts,
)

# ----------------------------------------------------------------------------------------
# Settings: read from a TOML file. The environment variable HURDLE_SETTINGS can point to
# another one (e.g. settings/real.toml for the real EY files).
# ----------------------------------------------------------------------------------------
DEFAULT_SETTINGS_FILE = REPO_ROOT / "settings" / "synthetic.toml"
SETTINGS_FILE = Path(os.environ.get("HURDLE_SETTINGS", DEFAULT_SETTINGS_FILE))


def main():
    settings = load_settings(SETTINGS_FILE, REPO_ROOT)
    print(f"Settings: {settings.settings_file}")
    print(f"Corpus folder: {settings.data.corpus_folder}")

    # Step 1: which days have a file. A day without a file is missing, not zero.
    layout = settings.collections_layout
    daily_files = list_daily_files(settings.data.corpus_folder,
                                   layout.file_name_date_format)
    show_daily_files(daily_files)

    # Step 2: the payment lines of one day, checked against the file's grand total.
    day = settings.debug.one_day
    if day not in daily_files:
        raise ValueError(f"{settings.settings_file.name} [debug]: one_day {day} has no daily file")
    one_day = read_one_file(daily_files[day], layout)
    show_one_day(day, one_day)

    # Step 3: the payment lines of every day, each file checked as in step 2, and every
    # line's date checked against its file's date. Read from the cache if the files have
    # not changed since the last run.
    collections = load_collections(daily_files, layout, settings.data.cache_folder)
    show_collections(collections)

    # Step 4: one key per company and customer, e.g. 'TR02-11226452'.
    collections = add_keys(collections, list(settings.companies.known))
    show_keys(collections)

    # Step 5: the AR ledger lines, with the same keys as the payment lines. Read from the
    # cache if nothing has changed since the last run. Its payments must equal the daily
    # files' totals on every day with a file.
    ledger = load_ledger(settings.data.ledger_file, settings.data.ledger_as_of,
                         settings.ledger_layout, settings.data.cache_folder)
    ledger = add_keys(ledger, list(settings.companies.known))
    show_ledger(ledger)
    tie = check_ledger_against_collections(ledger, collections, list(daily_files))
    show_tie(tie)

    # Step 6: days without a daily file. If [data] fill_missing_days_from_ledger is true,
    # their payments are taken from the ledger (flagged filled_from_ledger); otherwise
    # their weeks stay missing.
    if settings.data.fill_missing_days_from_ledger:
        collections, filled_days = fill_missing_days(collections, ledger, list(daily_files))
    else:
        collections = collections.assign(filled_from_ledger=False)
        filled_days = []
    show_filled(collections, filled_days, settings.data.fill_missing_days_from_ledger)

    # Step 7: weekly series. Weeks run Monday to Sunday; week 0 is the week of the first
    # daily file.
    first_day = min(daily_files)
    week_zero = first_day - timedelta(days=first_day.weekday())
    collections = add_week_index(collections, week_zero)
    print(f"Step 7: week 0 starts {week_zero}; lines run from week "
          f"{collections['week'].min()} to week {collections['week'].max()}")

    # The week calendar: which weeks are complete. A week with any business day with
    # neither a file nor filled payments is missing: its amounts are unknown, not zero.
    holidays = load_holidays(settings.data.holidays_file)
    weeks = week_calendar(list(daily_files), week_zero, holidays, filled_days)
    show_weeks(weeks)
    # Step 12 checks stage 1 at [debug] demo_origin: its target weeks must have answers.
    # Checked here, before the long steps, rather than when step 12 is reached.
    if settings.debug.demo_origin + settings.receivables.horizons > int(weeks["week"].max()):
        raise ValueError(f"[debug] demo_origin {settings.debug.demo_origin}: its target weeks "
                         f"reach week {settings.debug.demo_origin + settings.receivables.horizons}"
                         f", after the last week {int(weeks['week'].max())}, so step 12 would "
                         f"have no answers")
    # The same for step 13's diagnostic, which needs answers up to origin + 2 horizons.
    # Step 15's joint diagnostic uses the same splits.
    diagnostic_end = settings.diagnostic.origin + 2 * settings.receivables.horizons
    if ((settings.diagnostic.run or settings.joint_diagnostic.run)
            and diagnostic_end > int(weeks["week"].max())):
        raise ValueError(f"[diagnostic] origin {settings.diagnostic.origin}: its test weeks "
                         f"reach week {diagnostic_end}, after the last week "
                         f"{int(weeks['week'].max())}")

    # The weekly amounts per key: gross receipts (the target), refunds, unapplied lines.
    # Checked inside: the weekly table holds every TRY and every line of steps 3 and 6.
    weekly = weekly_amounts(collections)
    show_weekly(weekly)

    # Step 8: each key's receivables as known at the end of every week, in TRY at the rates
    # known then (one row per key and week). Weeks named in [receivables] leak_check_weeks
    # are rebuilt from a ledger cut off at that week and compared.
    rates = exchange_rates(collections, ledger)
    show_rates(rates)
    receivables = receivables_by_week(
        ledger, rates, week_zero, last_week=int(weeks["week"].max()),
        as_of=settings.data.ledger_as_of, horizons=settings.receivables.horizons,
        min_cleared=settings.receivables.min_cleared_for_delay,
        leak_check_weeks=settings.receivables.leak_check_weeks,
        age_edges=settings.receivables.overdue_age_days)
    show_receivables_by_week(receivables, settings.receivables.leak_check_weeks,
                             settings.debug.sample_week, settings.receivables.overdue_age_days)

    # Step 9: the panel. Built unit by unit; for now, when each key becomes known (the
    # week of its first interaction, in the ledger or the daily files). From that week on
    # it gets panel rows.
    known = known_weeks(collections, ledger, week_zero)
    show_known(known, collections, int(weeks["week"].max()))

    # Placeholder customers of unidentified cash ([companies] unidentified_customers) are not
    # customers: they get no panel rows, and their cash goes to table G below.
    unidentified_customers = settings.companies.unidentified_customers
    known = known[~is_unidentified(known, unidentified_customers)]
    unidentified_weekly = weekly[is_unidentified(weekly, unidentified_customers)]
    identified_weekly = weekly[~is_unidentified(weekly, unidentified_customers)]
    # The companies we expect (from the settings), not the ones left in `known`: a company
    # whose only keys are placeholders must still get its rows in tables R and G.
    companies = list(settings.companies.known)

    # The panel's rows (no labels yet): for every forecast week t, every key known by then,
    # once for each week ahead h = 1 ... horizons.
    panel = panel_rows(known, int(weeks["week"].max()), settings.receivables.horizons)
    show_panel_rows(panel)

    # The labels: what each key paid in the target week (gross receipts, TRY), or unknown
    # when the target week is missing or after the last week.
    panel = add_labels(panel, identified_weekly, weeks)
    show_labels(panel, identified_weekly)

    # The side tables: the cash the panel does not hold. A: gross receipts in week t+h of keys
    # not yet known at forecast week t.
    new_cash = new_customer_cash(identified_weekly, known, weeks, settings.receivables.horizons,
                                 companies)
    show_new_customer_cash(new_cash, weekly)

    # R: each company's refunds per week (negative), all keys together.
    refunds = refunds_by_week(weekly, weeks, companies)
    show_refunds(refunds)

    # G: each company's cash per week with no identifiable customer.
    unidentified = unidentified_cash_by_week(unidentified_weekly, weeks, companies)
    print(f"        unidentified cash (G): {len(unidentified):,} rows, "
          f"{unidentified['unidentified_cash'].sum():,.2f} TRY from "
          f"{len(unidentified_customers)} placeholder customers")

    # The tie-out: panel + A + G must equal every company's gross receipts in every labelled
    # target week, and R its refunds, so no cash is lost or counted twice.
    checked = check_side_tables(panel, new_cash, unidentified, refunds, weekly, companies)
    print(f"        tie-out passed: {checked:,} company-week-horizon checks "
          f"(panel + A + G = gross receipts; R = refunds)")

    # Step 10: TÜFE price levels by week, from the first ledger line (history goes back that
    # far), or week 0 if the ledger starts later, to the last week: the levels are normalised
    # to week 0, so it must be in the range. The copied inflation.py is called directly here
    # so it can be followed line by line; prices.py wraps it with our week numbers and a
    # coverage guard.
    last_week = int(weeks["week"].max())
    first_ledger_week = int(week_numbers(pd.Series([ledger["posting_date"].min()]),
                                         week_zero).iloc[0])
    first_price_week = min(first_ledger_week, 0)
    weeks_for_prices = price_weeks(week_zero, first_price_week, last_week)
    actual_level = inflation.weekly_price_level(weeks_for_prices)        # hindsight only
    published_level = inflation.published_price_level(weeks_for_prices)  # known at the time
    projected_level = inflation.projected_price_level(weeks_for_prices)  # P[origin, target]
    levels = published_levels(week_zero, first_price_week, last_week)
    show_prices(weeks_for_prices, actual_level, published_level, projected_level, levels)
    # U13.1: the levels a forecast re-inflates with, at the demo origin (a past week, so the
    # published figures can be checked) and at the last week (target weeks after the data).
    for forecast_week in (settings.debug.demo_origin, last_week):
        show_projected_levels(week_zero, forecast_week, settings.receivables.horizons)

    # Step 11: the features. Built unit by unit; for now, the ledger's payments before the
    # first daily file, used as payment history if [history] use_ledger_before_first_file.
    if settings.history.use_ledger_before_first_file:
        # Week 0 at the latest: a ledger that starts after the first daily file gives no
        # history before it, and the matrix must still hold the daily files from week 0.
        history_start = min(first_history_week(ledger, week_zero), 0)
        early_receipts = ledger_receipts_before(ledger, min(daily_files), week_zero)
        early_receipts = early_receipts[early_receipts["week"] >= history_start]
    else:
        history_start = 0
        early_receipts = ledger_receipts_before(ledger, min(daily_files), week_zero).iloc[0:0]
    show_early_receipts(early_receipts, history_start)

    # The weekly matrix: each key's deflated gross receipts per week (0.0 = paid nothing,
    # NaN = no data that week), from the history start to the last week.
    matrix = weekly_matrix(identified_weekly, early_receipts, known["key"], weeks, levels,
                           history_start, last_week)
    show_matrix(matrix)

    # The history features: per key and origin week, what its payments looked like up to t.
    history = history_features(matrix, known, last_week, settings.history.windows)
    show_history(history, matrix)

    # The calendar features of every target week, up to the last week + the horizons.
    calendar = calendar_features(week_zero, 0, last_week + settings.receivables.horizons,
                                 holidays)
    show_calendar(calendar, week_zero, holidays)

    # Where the holidays fall around each week ([calendar] holiday_encodings), joined to the
    # calendar block: tells a week whose cash moves into the next one from one that keeps it.
    positions = holiday_position_features(
        week_zero, 0, last_week + settings.receivables.horizons, holidays,
        settings.calendar.holiday_encodings, settings.calendar.drain_window,
        settings.calendar.catch_up_window)
    calendar = calendar.merge(positions, on="week", how="left", validate="one_to_one")
    show_holiday_positions(positions, week_zero, holidays)

    # The receivables features: per panel row, what the customer owed at t, split at its
    # target week t+h (deflated, like the history features).
    owed = receivables_features(panel, receivables, levels, settings.receivables.horizons,
                                settings.receivables.overdue_age_days)
    show_owed(owed, int(before_first_ledger_week(panel, receivables).sum()))

    # The label in TRY of week 0, like the features: divided by its target week's level.
    panel = add_deflated_label(panel, levels)
    show_deflated_label(panel, levels)

    # U12.2, the top-ups: A, R and G of every company forecast at the diagnostic origin by the
    # trailing mean ([topups] window) and by the ridge (with and without the recent level),
    # beside what came.
    show_topups(new_cash, refunds, unidentified, levels, calendar, settings.diagnostic.origin,
                settings.receivables.horizons, settings.topups.window)

    # U12.3, the choice of the top-up method ([topups] compare): every candidate forecast at
    # every origin of the comparison and scored against what came; the method is set by hand.
    if settings.topups.compare:
        compare_topups(new_cash, refunds, unidentified, levels, calendar, settings.topups,
                       settings.receivables.horizons)

    # The training table: one row per panel row, with its label and every feature block.
    table = join_features(panel, history, calendar, owed)
    show_table(table, panel)

    # Step 12: stage 1, "will it pay?", checked at one forecast week ([debug] demo_origin).
    # Built unit by unit; for now, the rows and the model's input.
    origin = settings.debug.demo_origin   # checked in step 7: its answers are known
    columns = feature_columns(block_feature_columns(history, calendar, owed),
                              settings.stage1.feature_blocks)
    train_rows = rows_known_at(table, origin)
    forecast_rows = rows_at_origin(table, origin)
    train_x = stage1_features(train_rows, columns, companies, settings.stage1.company_as_feature)
    forecast_x = stage1_features(forecast_rows, columns, companies,
                                 settings.stage1.company_as_feature)
    show_stage1_rows(origin, train_rows, forecast_rows, train_x)

    # Fit at the demo origin, predict its forecast rows, and score them: their target weeks
    # have data, so the answers are known here (only because this is a check of the past).
    # The logistic-regression baseline is fitted and scored the same way, for comparison.
    predictions, fit_seconds = {}, {}
    for name, fit in (("stage 1 (HGB)", fit_stage1), ("baseline (LR)", fit_stage1_baseline)):
        started = time.perf_counter()
        model = fit(train_x, train_rows["paid"], settings.stage1)
        fit_seconds[name] = time.perf_counter() - started
        predictions[name] = predict_paid(model, forecast_x)
    show_stage1_scores(forecast_rows, predictions, train_rows, fit_seconds)

    # Step 13: the stage 1 diagnostic (diagnostic only, never the forecast): stage 1 tuned on
    # its own. For each model: a grid on validation (each setting at its own best-F1
    # threshold), the best by the selection metric, refitted on train + validation and tested
    # once with its validation threshold; then the permutation importance on validation.
    if settings.diagnostic.run:
        splits = diagnostic_splits(table, settings.diagnostic.origin,
                                   settings.receivables.horizons)
        show_diagnostic_splits(splits, settings.diagnostic.origin,
                               settings.receivables.horizons)
        inputs = {part: (stage1_features(rows, columns, companies,
                                         settings.stage1.company_as_feature),
                         rows["paid"].astype(bool))
                  for part, rows in (("train", splits.train), ("validation", splits.validation),
                                     ("refit", splits.refit), ("test", splits.test))}
        tables, tests = {}, []
        calibration_p = {"validation": {}, "test": {}}   # label -> p, for the calibration
        for name, fit, candidates in (
                ("HGB", fit_stage1, hgb_candidates(settings.stage1, settings.diagnostic)),
                ("LR", fit_stage1_baseline, lr_candidates(settings.stage1,
                                                          settings.diagnostic))):
            grid = grid_search(name, fit, candidates, *inputs["train"], *inputs["validation"],
                               metric=settings.diagnostic.selection_metric)
            tests.append(choose_and_test(name, fit, candidates, grid, *inputs["refit"],
                                         *inputs["test"]))
            # The importance of the chosen setting, fitted on the training rows, measured on
            # the validation rows (the test rows stay untouched).
            chosen = candidates[int(grid.iloc[0]["candidate"])]
            model = fit(*inputs["train"], chosen.settings,
                        class_weight=class_weight_argument(chosen.class_weight))
            importance = permutation_importances(model, *inputs["validation"],
                                                 settings.diagnostic.importance_repeats,
                                                 settings.stage1.random_seed)
            tables[f"{name.lower()}_grid"] = grid
            tables[f"{name.lower()}_importance"] = importance
            show_diagnostic_grid(name, grid, importance, settings.diagnostic.selection_metric)
            # Calibration: the best UNWEIGHTED setting, fitted on the training rows, and the
            # same model after a Platt correction fitted on its validation p; scored on the
            # validation rows (in-sample for the correction) and the test rows.
            unweighted = best_unweighted(grid, candidates)
            if unweighted is None:
                print(f"        {name}: the grid has no unweighted setting; calibration skipped")
                continue
            if unweighted is not chosen:
                model = fit(*inputs["train"], unweighted.settings)
            a, b = fit_platt(inputs["validation"][1],
                             predict_paid(model, inputs["validation"][0]))
            print(f"        {name}: Platt correction fitted on validation: a {a:.4f}, b {b:.4f}")
            for part in ("validation", "test"):
                p = predict_paid(model, inputs[part][0])
                calibration_p[part][name] = p
                calibration_p[part][f"{name} + Platt"] = apply_platt(p, a, b)
        # The models' calibration beside the three references, per horizon and pooled; skill
        # against the training rows' share paid.
        summaries, reliabilities = [], []
        for part in ("validation", "test"):
            rows, paid = getattr(splits, part), inputs[part][1]
            references = reference_probabilities(splits.train, rows,
                                                 settings.diagnostic.reference_window,
                                                 settings.diagnostic.reference_prior_weeks)
            summary, reliability = calibration_comparison(
                paid, rows["h"], {**calibration_p[part], **references},
                references["base rate"], settings.diagnostic.calibration_groups)
            summary.insert(0, "part", part)
            reliability.insert(0, "part", part)
            summaries.append(summary)
            reliabilities.append(reliability)
        tables["calibration"] = pd.concat(summaries, ignore_index=True)
        tables["reliability"] = pd.concat(reliabilities, ignore_index=True)
        show_calibration(tables["calibration"])
        tables["test"] = pd.DataFrame(tests)
        show_diagnostic_test(tables["test"])
        paths = write_tables(settings.diagnostic.output_folder, tables)
        print(f"        tables written to {settings.diagnostic.output_folder}: "
              f"{[path.name for path in paths]}")

    # Step 14: stage 2, "how much, if it pays?", at the same forecast week as step 12. It
    # learns from the paid training rows only; the target is the deflated amount (TRY of
    # week 0). Built unit by unit; for now, the fit, its total on the training rows, and m
    # for the forecast rows of step 12, times step 12's p, against the cash that came.
    stage2_columns = feature_columns(block_feature_columns(history, calendar, owed),
                                     settings.stage2.feature_blocks)
    paid_rows = train_rows[train_rows["paid"].astype(bool)]
    paid_x = stage1_features(paid_rows, stage2_columns, companies,
                             settings.stage1.company_as_feature)
    # m = s x E: E is 1 ([stage2] scale "none") or the open balance lifted to a floor.
    paid_exposure = exposure(train_rows, paid_rows, settings.stage2)
    forecast_exposure = exposure(train_rows, forecast_rows, settings.stage2)
    started = time.perf_counter()
    stage2_model = fit_stage2(paid_x, paid_rows["amount_deflated"], paid_exposure,
                              settings.stage2)
    show_stage2_fit(origin, paid_rows, train_rows, paid_x, settings.stage2.loss,
                    time.perf_counter() - started)
    show_exposure(settings.stage2, paid_rows, paid_exposure, forecast_rows, forecast_exposure)
    training_ratio = total_ratio(stage2_model, paid_x, paid_rows["amount_deflated"],
                                 paid_exposure)
    m = predict_amount(stage2_model, stage1_features(forecast_rows, stage2_columns, companies,
                                                     settings.stage1.company_as_feature),
                       forecast_exposure)
    show_stage2_forecast(forecast_rows, m, predictions["stage 1 (HGB)"], training_ratio)

    # Stage 2 alone, on the forecast rows that paid: m against the amount, beside two simple
    # references (the training mean, the customer's own mean); skill against the first.
    paid_forecast = forecast_rows[(forecast_rows["label_status"] == "labelled")
                                  & forecast_rows["paid"].fillna(False).astype(bool)]
    if paid_forecast.empty:
        print("        no paid forecast rows: stage 2 cannot be scored here")
    else:
        amount_guesses = amount_references(train_rows, paid_forecast,
                                           settings.diagnostic.amount_reference_window,
                                           settings.diagnostic.reference_prior_weeks)
        stage2_scores = amount_scores(
            paid_forecast["amount_deflated"], paid_forecast["h"],
            {"stage 2": m[paid_forecast.index], **amount_guesses},
            amount_guesses["training mean"])
        show_stage2_scores(stage2_scores)
        # Stage 2's reliability: the paid rows in equal groups by m, mean actual vs mean m.
        show_amount_reliability(amount_reliability(
            paid_forecast["amount_deflated"], m[paid_forecast.index],
            settings.diagnostic.calibration_groups))

    # The amount-weighted calibration check of the whole model: the labelled forecast rows
    # (paid or not) in equal groups by p, the sum of p x m against the cash that came.
    labelled = forecast_rows["label_status"] == "labelled"
    if labelled.any():
        show_cash_by_p_group(cash_by_p_group(
            predictions["stage 1 (HGB)"][labelled], m[labelled],
            forecast_rows.loc[labelled, "amount_deflated"],
            settings.diagnostic.calibration_groups))

    # Step 15: the joint diagnostic of stage 1 and stage 2 (diagnostic only, never the
    # forecast): both stages' settings chosen together by the error of the weekly cash totals
    # on validation, tested once beside the default pair and the `deflated` benchmark.
    if settings.joint_diagnostic.run:
        run_joint_diagnostic(settings, table, columns, stage2_columns, companies, calendar)

    # Later steps follow as each unit is built.


def run_joint_diagnostic(settings, table, columns, stage2_columns, companies, calendar):
    """Step 15 (U11.5). The splits are the stage 1 diagnostic's ([diagnostic] origin)."""
    joint, horizons = settings.joint_diagnostic, settings.receivables.horizons
    origin = settings.diagnostic.origin
    splits = diagnostic_splits(table, origin, horizons)
    print(f"Step 15: joint diagnostic of stage 1 and stage 2 at origin {origin} (diagnostic "
          f"only); validation target weeks {origin + 1}-{origin + horizons}, test "
          f"{origin + horizons + 1}-{origin + 2 * horizons}")

    def features(rows, chosen_columns):
        return stage1_features(rows, chosen_columns, companies,
                               settings.stage1.company_as_feature)

    # Each fit is kept: the same setting on the same rows is never fitted twice.
    fitted = {}

    def predict_p(stage1_settings, fit_part, rows):
        key = ("stage 1", stage1_settings, fit_part)
        if key not in fitted:
            fit_rows = getattr(splits, fit_part)
            started = time.perf_counter()
            fitted[key] = fit_stage1(features(fit_rows, columns), fit_rows["paid"],
                                     stage1_settings)
            print(f"  ... stage 1 fitted on {fit_part} ({time.perf_counter() - started:.0f} s)")
        return predict_paid(fitted[key], features(rows, columns))

    def predict_m(stage2_settings, fit_part, rows):
        fit_rows = getattr(splits, fit_part)
        key = ("stage 2", stage2_settings, fit_part)
        if key not in fitted:
            paid = fit_rows[fit_rows["paid"].astype(bool)]
            started = time.perf_counter()
            fitted[key] = fit_stage2(features(paid, stage2_columns), paid["amount_deflated"],
                                     exposure(fit_rows, paid, stage2_settings), stage2_settings)
            print(f"  ... stage 2 fitted on {fit_part} ({time.perf_counter() - started:.0f} s)")
        return predict_amount(fitted[key], features(rows, stage2_columns),
                              exposure(fit_rows, rows, stage2_settings))

    # a, the grid: every setting fitted once on train, every pair scored on the validation
    # totals (all companies together).
    grid1 = stage1_grid(settings.stage1, joint)
    grid2 = stage2_grid(settings.stage2, joint)
    validation = splits.validation

    def predict_grid(stage, grid, predict):
        """Fits and predicts every setting of one grid, printing a line after each."""
        predictions = []
        started = time.perf_counter()
        for done, candidate in enumerate(grid, start=1):
            predictions.append(predict(candidate.settings, "train", validation))
            elapsed = time.perf_counter() - started
            left = elapsed / done * (len(grid) - done)
            print(f"  {stage} grid: {done} of {len(grid)} done ({candidate.params}); "
                  f"{elapsed:.0f} s so far, about {left:.0f} s left")
        return predictions

    p_by_stage1 = predict_grid("stage 1", grid1, predict_p)
    m_by_stage2 = predict_grid("stage 2", grid2, predict_m)

    # U11.5b, the eligibility gates: each setting checked on its own on validation, so a pair
    # can win only if both halves make sense alone (a lucky pair of a broken stage 1 and an
    # m that goes negative once won validation 0.103 and lost test 0.282).
    base_rate = reference_probabilities(splits.train, validation,
                                        settings.diagnostic.reference_window,
                                        settings.diagnostic.reference_prior_weeks)["base rate"]
    gates1 = stage1_gates(validation["paid"], p_by_stage1, base_rate,
                          joint.stage1_log_loss_tolerance, joint.stage1_slope_band)
    paid = validation[validation["paid"].astype(bool)]
    training_mean = amount_references(splits.train, paid,
                                      settings.diagnostic.amount_reference_window,
                                      settings.diagnostic.reference_prior_weeks)["training mean"]
    gates2 = stage2_gates(paid["amount_deflated"], [m[paid.index] for m in m_by_stage2],
                          training_mean, joint.stage2_skill_tolerance, joint.stage2_ratio_band)
    gates1 = with_params(gates1, grid1, "stage1")
    gates2 = with_params(gates2, grid2, "stage2")
    show_gates("stage 1", gates1, "log loss", ascending=True)
    show_gates("stage 2", gates2, "deviance skill", ascending=False)
    eligible1 = [int(i) for i in gates1.loc[gates1["eligible"], "stage1"]]
    eligible2 = [int(j) for j in gates2.loc[gates2["eligible"], "stage2"]]
    if not eligible1 or not eligible2:
        raise ValueError(f"step 15: {len(eligible1)} stage-1 and {len(eligible2)} stage-2 "
                         f"settings passed the eligibility gates, so no pair can be scored")

    pairs = score_pairs(validation, [p_by_stage1[i] for i in eligible1],
                        [m_by_stage2[j] for j in eligible2], joint.selection_metric)
    # score_pairs numbers the settings it was given from 0; back to their grid numbers.
    pairs["stage1"] = [eligible1[k] for k in pairs["stage1"]]
    pairs["stage2"] = [eligible2[k] for k in pairs["stage2"]]
    best = pairs.iloc[0]
    if best["non-finite rows"] > 0:
        raise ValueError("step 15: every pair of the grid gave a NaN or infinite forecast "
                         "on some validation rows, so none can be chosen")
    chosen = (grid1[int(best["stage1"])], grid2[int(best["stage2"])])
    default = (settings.stage1, settings.stage2)
    pairs = pd.concat([pd.DataFrame([{**{f"s1 {k}": v for k, v in grid1[i].params.items()},
                                      **{f"s2 {k}": v for k, v in grid2[j].params.items()}}
                                     for i, j in zip(pairs["stage1"], pairs["stage2"])]),
                       pairs.drop(columns=["stage1", "stage2"])], axis=1)
    show_joint_grid(pairs, joint.selection_metric)

    # b, the chosen pair and the default pair: validation (fitted on train) and test (refitted
    # on train + validation), with `deflated` fitted on the same target weeks.
    labelled = table[table["label_status"] == "labelled"]
    tables = {"pairs": pairs, "stage1_gates": gates1, "stage2_gates": gates2}
    for part, fit_part, train_until in (("validation", "train", origin),
                                        ("test", "refit", origin + horizons)):
        rows = getattr(splits, part)
        forecasts = {"chosen": (predict_p(chosen[0].settings, fit_part, rows)
                                * predict_m(chosen[1].settings, fit_part, rows)),
                     "default": (predict_p(default[0], fit_part, rows)
                                 * predict_m(default[1], fit_part, rows))}
        for by_company in (False, True):
            totals = cash_totals(rows, rows["amount_deflated"], by_company)
            totals = totals.rename(columns={"total": "actual"})
            for name, values in forecasts.items():
                totals[name] = cash_totals(rows, values, by_company)["total"].to_numpy()
            history = cash_totals(labelled, labelled["amount_deflated"], by_company)
            wanted = totals.drop(columns=["target_week", "actual", "chosen", "default"])
            for name, with_level in (("deflated", True), ("deflated_nolevel", False)):
                benchmark = benchmark_forecasts(history, calendar, train_until, wanted,
                                                with_level)
                totals[name] = benchmark["forecast"].to_numpy()
                totals[f"{name} fallback"] = benchmark["fallback"].to_numpy()
            level = "per company" if by_company else "consolidated"
            tables[f"totals_{part}_{level.replace(' ', '_')}"] = totals
            scores = scores_by_h(totals, ["chosen", "default", "deflated", "deflated_nolevel"])
            scores.insert(0, "level", level)
            scores.insert(0, "part", part)
            tables.setdefault("scores", []).append(scores)
        consolidated = tables[f"totals_{part}_consolidated"]
        comparisons = []
        for a, b in itertools.product(("chosen", "default"), ("deflated", "deflated_nolevel")):
            comparisons.append({"part": part, "A": a, "B": b, **paired_bootstrap(
                consolidated["actual"], consolidated[a], consolidated[b],
                joint.bootstrap_samples, settings.stage1.random_seed)})
        tables.setdefault("paired", []).extend(comparisons)
        if part == "test":
            # Each stage of the chosen pair alone on test, and the two calibration checks.
            p = predict_p(chosen[0].settings, "refit", rows)
            m = predict_m(chosen[1].settings, "refit", rows)
            reference = reference_probabilities(splits.refit, rows,
                                                settings.diagnostic.reference_window,
                                                settings.diagnostic.reference_prior_weeks)
            tables["stage1_test"] = pd.DataFrame([calibration_summary(
                rows["paid"], p, reference["base rate"],
                settings.diagnostic.calibration_groups)])
            paid = rows[rows["paid"].astype(bool)]
            guesses = amount_references(splits.refit, paid,
                                        settings.diagnostic.amount_reference_window,
                                        settings.diagnostic.reference_prior_weeks)
            tables["stage2_test"] = amount_scores(paid["amount_deflated"], paid["h"],
                                                  {"stage 2": m[paid.index], **guesses},
                                                  guesses["training mean"])
            tables["cash_by_p_group"] = cash_by_p_group(p, m, rows["amount_deflated"],
                                                        settings.diagnostic.calibration_groups)
            tables["amount_reliability"] = amount_reliability(
                paid["amount_deflated"], m[paid.index], settings.diagnostic.calibration_groups)
    tables["scores"] = pd.concat(tables["scores"], ignore_index=True)
    tables["paired"] = pd.DataFrame(tables["paired"])
    show_joint_results(tables, chosen, joint.bootstrap_samples)
    paths = write_tables(joint.output_folder, tables)
    print(f"        tables written to {joint.output_folder}: {[path.name for path in paths]}")


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


def topup_candidates(windows):
    """The top-up methods to compare: name -> (method, window, with_level); the mean over
    each of `windows`, and the ridge with and without the recent level (no window)."""
    candidates = {f"mean {window}": ("mean", window, True) for window in windows}
    candidates["ridge"] = ("ridge", 1, True)
    candidates["ridge no level"] = ("ridge", 1, False)
    return candidates


def show_topups(new_cash, refunds, unidentified, levels, calendar, origin, horizons, window):
    """A short printout of U12.2: the top-up forecasts made at `origin` by the trailing mean of
    `window` weeks and by the ridge with and without the recent level, all companies added up
    (each company forecast on its own series), beside the actual cash (hindsight), both over
    the companies whose target week has data. A and the top-up (A + R + G), million TRY of
    week 0."""
    came = topup_actuals(new_cash, refunds, unidentified, levels, [origin], horizons)
    # The forecasts are added up over the same (company, h) as the actuals: a company whose
    # target week has no data is left out of both columns.
    with_actual = came[["company_code", "origin_week", "h"]]
    lines = {}
    for name, (method, window, with_level) in topup_candidates([window]).items():
        found = topup_forecasts(new_cash, refunds, unidentified, levels, calendar, [origin],
                                horizons, method, window, with_level)
        found = found.merge(with_actual, on=["company_code", "origin_week", "h"])
        per_h = found.groupby("h")[["new_customer_cash", "topup", "fallback"]].sum()
        lines[f"A {name}"] = per_h["new_customer_cash"] / 1e6
        lines[f"top-up {name}"] = per_h["topup"] / 1e6
        if method == "ridge":
            lines[f"fallbacks {name}"] = per_h["fallback"]
    came = came.groupby("h")[["new_customer_cash", "topup"]].sum()
    lines["A actual"] = came["new_customer_cash"] / 1e6
    lines["top-up actual"] = came["topup"] / 1e6
    print(f"        top-ups at origin {origin}, all companies, million TRY of week 0 (actual = "
          f"what came, hindsight; fallbacks = companies whose ridge fell back to the level):")
    order = [name for prefix in ("A ", "top-up ", "fallbacks ") for name in lines
             if name.startswith(prefix)]
    print(pd.DataFrame(lines)[order].rename_axis("h").round(3).to_string())


def compare_topups(new_cash, refunds, unidentified, levels, calendar, topups, horizons):
    """U12.3: every top-up candidate (`topup_candidates` of [topups] compare_windows) forecast
    at every origin from compare_first_origin to compare_last_origin, and scored on the
    consolidated top-up (all companies added up) against what came. Prints, per candidate,
    the mean absolute error per h and over all h, the bias, and the error of A and R on their
    own (million TRY of week 0); then the lowest error and the method set in [topups]."""
    origins = list(range(topups.compare_first_origin, topups.compare_last_origin + 1))
    actuals = topup_actuals(new_cash, refunds, unidentified, levels, origins, horizons)
    lines = []
    for name, (method, window, with_level) in topup_candidates(topups.compare_windows).items():
        forecasts = topup_forecasts(new_cash, refunds, unidentified, levels, calendar, origins,
                                    horizons, method, window, with_level)
        scores = score_topups(forecasts, actuals, by_company=False).set_index(["series", "h"])
        line = {"candidate": name}
        for h in range(1, horizons + 1):
            line[f"h{h}"] = scores.loc[("topup", h), "mean_abs_error"]
        for series, label in (("topup", "all h"), ("new_customer_cash", "A"),
                              ("refunds", "R")):
            line[label] = scores.loc[series, "mean_abs_error"].mean()
        line["bias"] = scores.loc["topup", "bias"].mean()
        line["fallbacks"] = int(forecasts["fallback"].sum())
        lines.append(line)
    table = pd.DataFrame(lines).set_index("candidate")
    money = [column for column in table.columns if column != "fallbacks"]
    table[money] = table[money] / 1e6
    print(f"        top-up methods compared at origins {origins[0]}-{origins[-1]} "
          f"({len(origins)} origins), consolidated top-up: mean absolute error per h and over "
          f"all h, A and R on their own, bias (above 0: too high), million TRY of week 0; "
          f"fallbacks = forecasts (company, origin, h) whose ridge fell back to the level:")
    print(table.round(3).to_string())
    chosen = (f"mean {topups.window}" if topups.method == "mean"
              else "ridge" if topups.with_level else "ridge no level")
    print(f"        lowest error over all h: {table['all h'].idxmin()}; set in [topups]: {chosen}")


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


def show_stage1_rows(origin, train_rows, forecast_rows, train_x):
    """A short printout of step 12's rows and model input."""
    print(f"Step 12: stage 1 at forecast week {origin}: {len(train_rows):,} training rows "
          f"(labelled, target week <= {origin}; {train_rows['paid'].astype(bool).mean():.1%} "
          f"paid), {len(forecast_rows):,} forecast rows (origin {origin}, "
          f"{forecast_rows['key'].nunique():,} customers x h)")
    kinds = train_x.dtypes.astype(str).value_counts().to_dict()
    print(f"        model input: {train_x.shape[1]} columns {kinds}; columns with NaN: "
          f"{int(train_x.isna().any().sum())}")
    if "company_code" in train_x.columns:
        print(f"        company_code categories: {list(train_x['company_code'].cat.categories)}")


def show_stage1_scores(forecast_rows, predictions, train_rows, fit_seconds):
    """The scores of each model's p ({name: p per forecast row}) on the demo origin's
    forecast rows, against the base rate (every row gets the training share of paid rows).
    Log loss and Brier score: lower is better. AUC: 0.5 is a coin toss, 1 a perfect ranking
    of payers above non-payers. `fit_seconds` is {name: seconds}."""
    labelled = forecast_rows["label_status"] == "labelled"
    actual = forecast_rows.loc[labelled, "paid"].astype(bool)
    times = ", ".join(f"{name} {seconds:.1f} s" for name, seconds in fit_seconds.items())
    if actual.empty:
        # Every target week of this origin is a missing week: nothing can be scored.
        print(f"        fitted in {times}; no labelled forecast rows to score")
        return
    guesses = {name: p[labelled] for name, p in predictions.items()}
    guesses["base rate"] = pd.Series(train_rows["paid"].astype(bool).mean(), index=actual.index)
    print(f"        fitted in {times}; scored on {len(actual):,} forecast rows "
          f"({actual.mean():.1%} paid)")
    scores = pd.DataFrame({
        name: {"log loss": log_loss(actual, guess, labels=[False, True]),
               "Brier": brier_score_loss(actual, guess),
               "AUC": roc_auc_score(actual, guess) if name != "base rate" else 0.5}
        for name, guess in guesses.items()
    }).T
    print(scores.round(4).to_string())
    h = forecast_rows.loc[labelled, "h"]
    by_h = pd.DataFrame({f"mean p, {name}": predicted.groupby(h).mean()
                         for name, predicted in guesses.items() if name != "base rate"})
    by_h["share paid"] = actual.groupby(h).mean()
    print("        per h, mean p vs share that actually paid:")
    print(by_h.round(4).to_string())


def show_diagnostic_splits(splits, origin, horizons):
    """A short printout of the diagnostic's four sets of rows."""
    print(f"Step 13: stage 1 diagnostic at origin {origin} (diagnostic only, never the forecast)")
    for name, (rows, weeks) in {
            "train": (splits.train, f"target <= {origin}"),
            "validation": (splits.validation, f"{origin} < target <= {origin + horizons}"),
            "refit": (splits.refit, f"target <= {origin + horizons}"),
            "test": (splits.test, f"{origin + horizons} < target <= {origin + 2 * horizons}"),
    }.items():
        print(f"        {name:<10} {len(rows):>9,} rows, {rows['paid'].astype(bool).mean():6.1%} "
              f"paid ({weeks})")


def show_diagnostic_grid(name, grid, importance, metric):
    """A short printout of one model's diagnostic grid (ranked by `metric`) and permutation
    importance."""
    columns = [column for column in grid.columns if column not in ("candidate",)]
    print(f"        {name}: the 5 best of {len(grid)} settings by validation {metric}")
    print(grid[columns].head(5).round(4).to_string(index=False))
    print(f"        {name}: the 15 most important features (rise in validation log loss when "
          f"shuffled)")
    print(importance.head(15).round(5).to_string(index=False))


def show_calibration(calibration):
    """A short printout of the calibration table: the test rows pooled over the horizons, one
    line per model or reference, and the log-loss skill per horizon on test (the full table,
    validation included, is in the CSV)."""
    test = calibration[calibration["part"] == "test"]
    columns = ["model", "mean p", "share paid", "ECE", "slope", "intercept", "log loss",
               "brier", "log loss skill", "brier skill", "AUC"]
    print("        CALIBRATION on test, all horizons (train-only models; skill against the "
          "base rate):")
    print(test[test["h"] == "all"][columns].round(4).to_string(index=False))
    print("        log-loss skill per horizon, on test:")
    by_h = test.pivot(index="model", columns="h", values="log loss skill")
    print(by_h.loc[test["model"].unique()].round(4).to_string())


def show_diagnostic_test(test):
    """The test table: each model's chosen setting, scored once on the test rows."""
    print("        TEST (target weeks after the validation weeks), each model's chosen setting "
          "at its validation threshold:")
    print(test.round(4).to_string(index=False))


def show_stage2_fit(origin, paid_rows, train_rows, paid_x, loss, fit_seconds):
    """A short printout of step 14's fit: the paid training rows it learned from and the
    spread of their deflated amounts (TRY of week 0)."""
    amount = paid_rows["amount_deflated"]
    print(f"Step 14: stage 2 at forecast week {origin}: {len(paid_rows):,} paid training rows "
          f"(of {len(train_rows):,}), {paid_x.shape[1]} input columns, loss {loss!r}, "
          f"fitted in {fit_seconds:.1f} s")
    print(f"        deflated amount per paid row: median {amount.median():,.0f}, mean "
          f"{amount.mean():,.0f}, 99th percentile {amount.quantile(0.99):,.0f}, "
          f"max {amount.max():,.0f}")


def show_exposure(stage2, paid_rows, paid_exposure, forecast_rows, forecast_exposure):
    """Step 14's E (m = s x E): the scale, and with "open balance" the floor and how many rows
    were lifted to it (their open balance was below it)."""
    if stage2.scale == "none":
        print("        scale 'none': E = 1, stage 2 learns the amount itself")
        return
    lifted = {name: int((rows["open_balance"] < e).sum())
              for name, rows, e in (("paid training", paid_rows, paid_exposure),
                                    ("forecast", forecast_rows, forecast_exposure))}
    print(f"        scale 'open balance', floor {stage2.balance_floor!r}: stage 2 learns the "
          f"share of the balance paid; lifted to the floor: {lifted['paid training']:,} of "
          f"{len(paid_rows):,} paid training rows, {lifted['forecast']:,} of "
          f"{len(forecast_rows):,} forecast rows")


def show_stage2_forecast(forecast_rows, m, p, training_ratio):
    """Step 14's m on the forecast rows of step 12, and a first look at the whole model: per h,
    the sum of p x m against the deflated cash that actually came (labelled rows only; a
    missing week is not a zero). A ratio of 1.0 means the cash adds up."""
    print(f"        training total ratio (sum of m / sum of actual, paid rows): "
          f"{training_ratio:.4f}")
    print(f"        m on the {len(m):,} forecast rows: median {m.median():,.0f}, "
          f"mean {m.mean():,.0f}")
    labelled = forecast_rows["label_status"] == "labelled"
    if not labelled.any():
        print("        no labelled forecast rows: p x m cannot be compared with actual cash")
        return
    rows = pd.DataFrame({"h": forecast_rows["h"], "mean m": m, "p x m": p * m,
                         "actual cash": forecast_rows["amount_deflated"]})[labelled]
    by_h = rows.groupby("h").agg({"mean m": "mean", "p x m": "sum", "actual cash": "sum"})
    by_h.loc["all"] = [rows["mean m"].mean(), rows["p x m"].sum(), rows["actual cash"].sum()]
    by_h["ratio"] = by_h["p x m"] / by_h["actual cash"]
    print("        per h, the whole model (sum of p x m) against the actual cash, deflated:")
    print(by_h.round({"mean m": 0, "p x m": 0, "actual cash": 0, "ratio": 3}).to_string())


def show_stage2_scores(scores):
    """Step 14's scores of m on the paid forecast rows (`amount_scores`): all horizons
    together, one line per model or reference, then the deviance skill per horizon."""
    print("        STAGE 2 alone, on the paid forecast rows (skill against the training mean; "
          "R2's baseline is the rows' own mean):")
    pooled = scores[scores["h"] == "all"].drop(columns="h")
    print(pooled.round({"mean deviance": 0, "deviance skill": 3, "R2 hindsight": 3,
                        "WAPE": 3, "MAPE": 3, "total ratio": 3}).to_string(index=False))
    print("        deviance skill per h:")
    by_h = scores.pivot(index="model", columns="h", values="deviance skill")
    print(by_h.loc[scores["model"].unique()].round(3).to_string())


def show_amount_reliability(table):
    """Stage 2's reliability table (`amount_reliability`): mean actual vs mean m per m group."""
    print("        stage 2 reliability, paid forecast rows in equal groups by m (calibrated: "
          "ratio = mean actual / mean m near 1 in every group):")
    print(table.round({"m_low": 0, "m_high": 0, "mean_m": 0, "mean_actual": 0, "ratio": 3})
          .to_string(index=False))


def show_cash_by_p_group(table):
    """The amount-weighted calibration check (`cash_by_p_group`)."""
    print("        whole model, labelled forecast rows in equal groups by p: expected cash "
          "(sum of p x m) vs actual cash, deflated (ratio = expected / actual):")
    print(table.round({"p_low": 4, "p_high": 4, "mean_p": 4, "expected_cash": 0,
                       "actual_cash": 0, "ratio": 3}).to_string(index=False))


def with_params(gates, grid, number_column):
    """A gate table with each setting's grid parameters added after its number."""
    params = pd.DataFrame([grid[int(i)].params for i in gates[number_column]])
    return pd.concat([gates[[number_column]], params, gates.drop(columns=number_column)],
                     axis=1)


def show_gates(stage, gates, score, ascending):
    """Step 15's eligibility gates for one stage: how many settings passed, why the others
    failed, and the best few that passed by `score`."""
    gate_columns = [column for column in gates.columns if column.endswith(" gate")]
    passed = gates[gates["eligible"]]
    failed = {column: int((~gates[column]).sum()) for column in gate_columns}
    print(f"        {stage} gates: {len(passed)} of {len(gates)} settings eligible; settings "
          f"failing each gate: {failed}")
    best = passed.sort_values(score, ascending=ascending).head(5)
    print(best.drop(columns=[*gate_columns, "eligible"]).round(4).to_string(index=False))


def show_joint_grid(pairs, metric):
    """The best pairs of step 15's grid on the validation totals."""
    print(f"        the 10 best of {len(pairs)} pairs by validation {metric} of the "
          f"consolidated weekly totals:")
    print(pairs.head(10).round(4).to_string(index=False))
    failed = pairs[pairs["non-finite rows"] > 0]
    if not failed.empty:
        print(f"        {len(failed)} pairs NOT scored (a NaN or infinite p x m on some rows; "
              f"ranked last with infinite scores):")
        print(failed.to_string(index=False))


def show_joint_results(tables, chosen, samples):
    """Step 15's results: per split, the consolidated errors per h of the chosen pair, the
    default pair and `deflated`; the ratio hurdle / `deflated`; the paired comparisons; per
    company pooled; then the chosen pair's stages alone and the calibration checks on test."""
    print(f"        chosen pair: stage 1 {chosen[0].params}, stage 2 {chosen[1].params}")
    scores = tables["scores"]
    for part in ("validation", "test"):
        mine = scores[(scores["part"] == part) & (scores["level"] == "consolidated")]
        print(f"        {part.upper()}, consolidated weekly totals (deflated TRY), MAPE / bias "
              f"/ WAPE per h:")
        wide = mine.pivot(index="h", columns="model", values=["MAPE", "bias", "WAPE"])
        print(wide.round(3).to_string())
        mape = mine.pivot(index="h", columns="model", values="MAPE")
        ratio = pd.DataFrame({f"{a} / deflated": mape[a] / mape["deflated"]
                              for a in ("chosen", "default")})
        print(f"        {part}: MAPE ratio hurdle / deflated per h (below 1 = hurdle better):")
        print(ratio.round(3).T.to_string())
        fallbacks = tables[f"totals_{part}_consolidated"][["deflated fallback",
                                                            "deflated_nolevel fallback"]].sum()
        print(f"        deflated totals that fell back to the recent level: "
              f"{fallbacks.to_dict()}")
        per_company = scores[(scores["part"] == part) & (scores["level"] == "per company")
                             & (scores["h"] == "all")]
        print(f"        {part}, per company, all h: MAPE")
        print(per_company.pivot(index="company_code", columns="model", values="MAPE")
              .round(3).to_string())
    print(f"        PAIRED comparison on the consolidated totals (APE of A - APE of B; below 0 "
          f"= A better; 95% interval over {samples} resamples, too narrow: totals overlap):")
    print(tables["paired"].round(4).to_string(index=False))
    print("        TEST, the chosen pair's stage 1 alone (calibration, skill against the base "
          "rate):")
    print(tables["stage1_test"][["mean p", "share paid", "ECE", "log loss skill", "AUC"]]
          .round(4).to_string(index=False))
    print("        TEST, the chosen pair's stage 2 alone on the paid rows, all h:")
    stage2 = tables["stage2_test"]
    print(stage2[stage2["h"] == "all"].drop(columns="h").round(3).to_string(index=False))
    show_cash_by_p_group(tables["cash_by_p_group"])
    show_amount_reliability(tables["amount_reliability"])


if __name__ == "__main__":
    main()
