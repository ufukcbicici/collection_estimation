"""Step 18, the settings grid (U16), run from end to end with its printouts.

`run_grid` is what `run_hurdle_pipeline.py` (step 18) and `run_b.py` (the grid alone) both
call, so the grid is run one way only: the candidates, every candidate's forecast at every
validation origin (each fit cached on its own, with progress lines), the gates, the scores,
the combination kept with its calibration and the references, and then, only with [grid]
run_test, the v1 test of what was kept on the [evaluation] test origins.
"""
import time

from .evaluate import actual_totals
from .grid import (
    GridChoice,
    GridTest,
    grid_candidates,
    grid_choice,
    grid_gates,
    grid_predictions,
    grid_scores,
    grid_test,
)
from .inputs import ModelInputs
from .reports.grid import (
    show_grid_candidates,
    show_grid_choice,
    show_grid_gates,
    show_grid_predictions,
    show_grid_scores,
    show_grid_test,
)


def run_grid(settings, inputs: ModelInputs, *, rolled=None) -> tuple[GridChoice, GridTest | None]:
    """The grid over the [evaluation] validation origins, then the v1 test if [grid] run_test.

    `settings` is a `Settings`, `inputs` what `prepare_inputs` made. `rolled` is step 17's
    forecasts with the current settings (`rolling_forecasts`), if the caller made them: the
    printouts then check that candidate 0 gives them exactly; None skips that check. Returns
    (the combination kept, the v1 test); the test is None without [grid] run_test.
    """
    origins = list(settings.evaluation.validation_origins)
    horizons = settings.receivables.horizons
    cache_folder = settings.data.cache_folder / "hurdle_grid"
    candidates = grid_candidates(settings)
    print(f"Step 18: settings grid (U16) over the validation origins "
          f"{origins[0]}-{origins[-1]} ({len(origins)} origins)")
    show_grid_candidates(candidates, len(origins))
    started = time.perf_counter()

    def fit_progress(done, total, stage, at, number, seconds):
        left = (time.perf_counter() - started) / done * (total - done)
        print(f"  ... {stage} setting {number}, origin {at}: {done} of {total} done "
              f"({seconds:.0f} s), about {left / 60:.0f} min left")

    # U16.3: every candidate at every origin. U16.4: the gates of each stage-1 and stage-2
    # candidate, all origins together. U16.5: every combination of the scored candidates,
    # scored on the cash that came.
    predictions = grid_predictions(
        inputs.table, origins, candidates, settings, block_columns=inputs.block_columns,
        companies=inputs.companies, new_cash=inputs.new_cash, refunds=inputs.refunds,
        unidentified=inputs.unidentified, levels=inputs.levels, calendar=inputs.calendar,
        cache_folder=cache_folder, on_fit_done=fit_progress)
    show_grid_predictions(predictions, rolled)
    gates = grid_gates(predictions, candidates, settings)
    show_grid_gates("stage 1", gates.stage1, "log loss", ascending=True)
    show_grid_gates("stage 2", gates.stage2, "deviance skill", ascending=False)
    actual = actual_totals(inputs.weekly, inputs.weeks, inputs.levels, inputs.companies,
                           origins, horizons)
    table = grid_scores(predictions, candidates, gates, actual, settings,
                        companies=inputs.companies, week_zero=inputs.week_zero)
    show_grid_scores(table, candidates, rolled, actual, horizons)

    # U16.6: the combination kept, its calibration, and on the same origins how it compares
    # with the current settings and the references.
    choice = grid_choice(predictions, candidates, table, actual, settings,
                         companies=inputs.companies, week_zero=inputs.week_zero,
                         weekly=inputs.weekly, weeks=inputs.weeks, levels=inputs.levels,
                         calendar=inputs.calendar, ledger=inputs.ledger, rates=inputs.rates,
                         new_cash=inputs.new_cash, refunds=inputs.refunds,
                         unidentified=inputs.unidentified)
    show_grid_choice(choice, candidates, table, settings)

    # U16.7: the v1 test. What the grid froze, on the test origins; nothing is chosen there.
    if not settings.grid.run_test:
        print("        the v1 test is not run ([grid] run_test = false)")
        return choice, None
    test_origins = list(settings.evaluation.test_origins)
    print(f"        V1 TEST over the test origins {test_origins[0]}-{test_origins[-1]} "
          f"({len(test_origins)} origins): a report, not a choice; do not change the "
          f"settings because of it")
    started = time.perf_counter()
    test_actual = actual_totals(inputs.weekly, inputs.weeks, inputs.levels, inputs.companies,
                                test_origins, horizons)
    test = grid_test(choice, test_actual, settings, table=inputs.table,
                     block_columns=inputs.block_columns, companies=inputs.companies,
                     week_zero=inputs.week_zero, weekly=inputs.weekly, weeks=inputs.weeks,
                     levels=inputs.levels, calendar=inputs.calendar, ledger=inputs.ledger,
                     rates=inputs.rates, new_cash=inputs.new_cash, refunds=inputs.refunds,
                     unidentified=inputs.unidentified, cache_folder=cache_folder,
                     on_fit_done=fit_progress)
    show_grid_test(test, choice, settings)
    return choice, test
