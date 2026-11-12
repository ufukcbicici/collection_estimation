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
   18  settings grid                                               (U16)  <- in progress

Steps 1-11 are one call, `inputs.prepare_inputs` (shared with the other entry points); their
printouts follow it (`show_preparation`). To look at one of their tables, put the breakpoint
in hurdle/inputs.py. Step 18 is one call too, `grid_run.run_grid`, which run_hurdle_grid.py
runs alone.
Every printout (`show_*`) is in hurdle/reports/, one module per stretch of steps; what stays
here is main() and the experiments only it runs (the joint diagnostic, the top-up comparison).

The settings (data paths, company codes, ...) are in settings/synthetic.toml. To run on
other data (e.g. the real EY files), write another settings file and point the environment
variable HURDLE_SETTINGS to it; nothing in the code needs to change.
"""
import itertools
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# The package lives under src/. Adding it here lets this script run from PyCharm or a
# terminal without installing the project.
sys.path.insert(0, str(REPO_ROOT / "src"))

# Customer names are Turkish; a Windows console would otherwise crash printing "İ".
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from collection_estimation.hurdle.forecast import forecast  # noqa: E402
from collection_estimation.hurdle.evaluate import (  # noqa: E402
    HURDLE,
    actual_totals,
    balance_coherence,
    component_bias,
    deflated_reference,
    evaluation_scores,
    paired_wape_bootstrap,
    payer_split,
    rolling_fingerprint,
    rolling_forecasts,
    rule_reference,
    stack_forecasts,
)
from collection_estimation.hurdle.calibrate import (  # noqa: E402
    calibration_reliability,
    choose_calibration,
)
from collection_estimation.hurdle.cache import load_with_cache  # noqa: E402
from collection_estimation.hurdle.inputs import prepare_inputs  # noqa: E402
from collection_estimation.hurdle.reports.preparation import show_preparation  # noqa: E402
from collection_estimation.hurdle.reports.stages import (  # noqa: E402
    show_amount_reliability,
    show_calibration,
    show_cash_by_p_group,
    show_diagnostic_grid,
    show_diagnostic_splits,
    show_diagnostic_test,
    show_exposure,
    show_gates,
    show_joint_grid,
    show_joint_results,
    show_stage1_rows,
    show_stage1_scores,
    show_stage2_fit,
    show_stage2_forecast,
    show_stage2_scores,
)
from collection_estimation.hurdle.reports.evaluation import (  # noqa: E402
    show_actual_totals,
    show_balance_coherence,
    show_calibration_choice,
    show_calibration_reliability,
    show_component_bias,
    show_customer_forecasts,
    show_deflated_reference,
    show_evaluation_scores,
    show_forecast_totals,
    show_forecast_vs_actual,
    show_paired_bootstrap,
    show_payer_split,
    show_rolling_forecasts,
    show_rule_reference,
)
from collection_estimation.hurdle.grid_run import run_grid  # noqa: E402
from collection_estimation.hurdle.benchmark import (  # noqa: E402
    benchmark_forecasts,
    cash_totals,
)
from collection_estimation.hurdle.grid import with_params  # noqa: E402
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
from collection_estimation.hurdle.training_table import rows_at_origin, rows_known_at  # noqa: E402
from collection_estimation.hurdle.topups import score_topups, topup_actuals, topup_forecasts  # noqa: E402
from collection_estimation.hurdle.settings import load_settings  # noqa: E402

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

    # Steps 1-11, in hurdle/inputs.py: the daily files and the ledger, the weeks, the
    # receivables, the panel and its side tables, the price levels and the features. Each
    # step's checks run inside it; its printouts follow, step by step, once it is done.
    inputs, details = prepare_inputs(settings, with_details=True)
    show_preparation(inputs, details, settings)
    companies, week_zero = inputs.companies, inputs.week_zero
    weeks, weekly, ledger, rates = inputs.weeks, inputs.weekly, inputs.ledger, inputs.rates
    levels, calendar, table = inputs.levels, inputs.calendar, inputs.table
    block_columns = inputs.block_columns
    new_cash, refunds, unidentified = inputs.new_cash, inputs.refunds, inputs.unidentified

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

    # Step 12 checks stage 1 at [debug] demo_origin: its target weeks must have answers.
    # Checked here, before the long steps, rather than when step 12 is reached.
    if settings.debug.demo_origin + settings.receivables.horizons > inputs.last_week:
        raise ValueError(f"[debug] demo_origin {settings.debug.demo_origin}: its target weeks "
                         f"reach week {settings.debug.demo_origin + settings.receivables.horizons}"
                         f", after the last week {inputs.last_week}, so step 12 would have no "
                         f"answers")
    # The same for step 13's diagnostic, which needs answers up to origin + 2 horizons.
    # Step 15's joint diagnostic uses the same splits.
    diagnostic_end = settings.diagnostic.origin + 2 * settings.receivables.horizons
    if ((settings.diagnostic.run or settings.joint_diagnostic.run)
            and diagnostic_end > inputs.last_week):
        raise ValueError(f"[diagnostic] origin {settings.diagnostic.origin}: its test weeks "
                         f"reach week {diagnostic_end}, after the last week {inputs.last_week}")

    # Step 12: stage 1, "will it pay?", checked at one forecast week ([debug] demo_origin).
    # Built unit by unit; for now, the rows and the model's input.
    origin = settings.debug.demo_origin   # checked above: its answers are known
    columns = feature_columns(block_columns, settings.stage1.feature_blocks)
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
        split_inputs = {part: (stage1_features(rows, columns, companies,
                                               settings.stage1.company_as_feature),
                               rows["paid"].astype(bool))
                        for part, rows in (("train", splits.train),
                                           ("validation", splits.validation),
                                           ("refit", splits.refit), ("test", splits.test))}
        tables, tests = {}, []
        calibration_p = {"validation": {}, "test": {}}   # label -> p, for the calibration
        for name, fit, candidates in (
                ("HGB", fit_stage1, hgb_candidates(settings.stage1, settings.diagnostic)),
                ("LR", fit_stage1_baseline, lr_candidates(settings.stage1,
                                                          settings.diagnostic))):
            grid = grid_search(name, fit, candidates, *split_inputs["train"],
                               *split_inputs["validation"],
                               metric=settings.diagnostic.selection_metric)
            tests.append(choose_and_test(name, fit, candidates, grid, *split_inputs["refit"],
                                         *split_inputs["test"]))
            # The importance of the chosen setting, fitted on the training rows, measured on
            # the validation rows (the test rows stay untouched).
            chosen = candidates[int(grid.iloc[0]["candidate"])]
            model = fit(*split_inputs["train"], chosen.settings,
                        class_weight=class_weight_argument(chosen.class_weight))
            importance = permutation_importances(model, *split_inputs["validation"],
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
                model = fit(*split_inputs["train"], unweighted.settings)
            a, b = fit_platt(split_inputs["validation"][1],
                             predict_paid(model, split_inputs["validation"][0]))
            print(f"        {name}: Platt correction fitted on validation: a {a:.4f}, b {b:.4f}")
            for part in ("validation", "test"):
                p = predict_paid(model, split_inputs[part][0])
                calibration_p[part][name] = p
                calibration_p[part][f"{name} + Platt"] = apply_platt(p, a, b)
        # The models' calibration beside the three references, per horizon and pooled; skill
        # against the training rows' share paid.
        summaries, reliabilities = [], []
        for part in ("validation", "test"):
            rows, paid = getattr(splits, part), split_inputs[part][1]
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
    stage2_columns = feature_columns(block_columns, settings.stage2.feature_blocks)
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

    # Step 16: one forecast (U13), at the same forecast week as steps 12 and 14: each
    # customer's p x m (which must equal step 14's), plus the top-ups, per company and
    # consolidated, re-inflated with the price levels projected at the origin; then beside the
    # nominal cash that really came.
    started = time.perf_counter()
    result = forecast(table, origin, settings,
                      block_columns=block_columns,
                      companies=companies, new_cash=new_cash, refunds=refunds,
                      unidentified=unidentified, levels=levels, calendar=calendar,
                      week_zero=week_zero)
    show_customer_forecasts(result.customers, forecast_rows, time.perf_counter() - started,
                            predictions["stage 1 (HGB)"] * m)
    show_forecast_totals(result.totals)
    # U14.1: the cash that really came, per company and consolidated (hindsight).
    actual = actual_totals(weekly, weeks, levels, companies, [origin],
                           settings.receivables.horizons)
    show_actual_totals(actual, weekly, weeks)
    show_forecast_vs_actual(result.totals, actual)

    # Step 17: the evaluation (U14) over the validation origins ([evaluation]; the test
    # origins are scored only in U16). Built unit by unit; for now, the forecast at every
    # validation origin beside the cash that came.
    rolled = None   # step 17's forecasts with the model, if it makes them; step 18 checks them
    if settings.evaluation.run:
        origins = list(settings.evaluation.validation_origins)
        print(f"Step 17: evaluation over the validation origins {origins[0]}-{origins[-1]} "
              f"({len(origins)} origins)")
        rolled_actual = actual_totals(weekly, weeks, levels, companies, origins,
                                      settings.receivables.horizons)

        # The hurdle model at every origin (slow: two fits each). [evaluation] with_model =
        # false skips it, to look at the references alone; then rolled stays None.
        if settings.evaluation.with_model:
            started = time.perf_counter()

            def progress(done, total, at, seconds):
                left = (time.perf_counter() - started) / done * (total - done)
                print(f"  ... origin {at}: {done} of {total} done ({seconds:.0f} s), about "
                      f"{left / 60:.0f} min left")

            def fit_every_origin():
                return rolling_forecasts(
                    table, origins, settings, block_columns=block_columns,
                    companies=companies, new_cash=new_cash, refunds=refunds,
                    unidentified=unidentified, levels=levels, calendar=calendar,
                    week_zero=week_zero, on_origin_done=progress)

            # Read from the cache if nothing the forecasts are made from has changed (the
            # fingerprint: same arguments); else fitted, and the cache written.
            fingerprint = rolling_fingerprint(
                table, origins, settings, block_columns=block_columns,
                companies=companies, new_cash=new_cash, refunds=refunds,
                unidentified=unidentified, levels=levels, calendar=calendar,
                week_zero=week_zero)
            rolled = load_with_cache("hurdle_rolling_forecasts", fingerprint, fit_every_origin,
                                     settings.data.cache_folder)
            show_rolling_forecasts(rolled, rolled_actual)
        else:
            print("        the hurdle model's forecasts are skipped ([evaluation] with_model = "
                  "false)")
        # U14.3: the `deflated` reference at the same origins, refitted at each.
        reference = deflated_reference(weekly, weeks, levels, calendar, companies, origins,
                                       settings.receivables.horizons, week_zero)
        show_deflated_reference(reference, rolled, rolled_actual)
        # U14.4: the financial rule at the same origins (open invoices, due date + usual delay).
        rule = rule_reference(ledger, rates, origins, settings, companies=companies,
                              new_cash=new_cash, refunds=refunds, unidentified=unidentified,
                              levels=levels, calendar=calendar, week_zero=week_zero)
        show_rule_reference(rule, rolled_actual)
        forecasts = stack_forecasts(rolled, reference, rule)
        scores = evaluation_scores(forecasts, rolled_actual, settings.receivables.horizons)
        show_evaluation_scores(scores, settings.receivables.horizons)
        # U14.5c: is the hurdle model (without it, deflated_trend) really more accurate than
        # each other method, or only luckier on these weeks?
        subject = HURDLE if rolled is not None else "deflated_trend"
        pairs = [(subject, other) for other in dict.fromkeys(forecasts["method"])
                 if other != subject]
        bootstrap = paired_wape_bootstrap(forecasts, rolled_actual, pairs,
                                          settings.receivables.horizons,
                                          settings.evaluation.bootstrap_samples,
                                          settings.evaluation.bootstrap_seed)
        show_paired_bootstrap(bootstrap)
        # U14.5d: where the bias comes from, part by part; and for the known customers, how
        # many pay against how much each pays (needs the model's customer rows).
        bias = component_bias(rolled, rule, rolled_actual, new_cash, refunds, unidentified,
                              levels, settings.receivables.horizons)
        show_component_bias(bias, settings.receivables.horizons)
        if rolled is not None:
            split = payer_split(rolled.customers, weekly, weeks, levels,
                                settings.receivables.horizons)
            show_payer_split(split, settings.receivables.horizons)
            # U14.6: does each customer's forecast fit what it owes at the origin?
            coherence = balance_coherence(rolled.customers, table, weekly, weeks, levels,
                                          settings.receivables.horizons,
                                          settings.evaluation.coherence_factor)
            show_balance_coherence(coherence, settings.evaluation.coherence_factor)
            # U15.1: how far p and m were from what happened on these origins.
            reliability = calibration_reliability(rolled.customers, weekly, weeks, levels,
                                                  settings.diagnostic.calibration_groups)
            show_calibration_reliability(reliability)
            # U15.4: every pair of corrections learned on the first part of these weeks and
            # judged on the later origins; the chosen pair learned again on all of them. It is
            # not applied here: scored on the weeks it learned from, it would look too good.
            # U16 uses it on the test.
            choice = choose_calibration(rolled, weekly, weeks, levels, rolled_actual, settings)
            show_calibration_choice(choice, settings.calibration.min_gain)

    # Step 18: the settings grid (U16) over the validation origins, then the v1 test only with
    # [grid] run_test (hurdle/grid_run.py, which run_hurdle_grid.py runs alone). Candidate 0 is
    # checked against step 17's forecasts when step 17 made them (rolled).
    if settings.grid.run:
        run_grid(settings, inputs, rolled=rolled)

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


if __name__ == "__main__":
    main()
