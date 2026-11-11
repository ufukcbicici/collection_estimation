"""Printouts of step 18, the settings grid (U16): the candidates, their forecasts, the
gates, the scores, the combination kept and the v1 test.
"""
from ..calibrate import candidate_name
from ..evaluate import HURDLE, POOLED, STACK_COLUMNS, evaluation_scores
from ..forecast import CONSOLIDATED
from ..grid import UNCORRECTED
from .evaluation import show_calibration_choice, show_evaluation_scores, show_paired_bootstrap


def show_grid_candidates(candidates, origins):
    """Step 18's settings to try (U16.2), the current ones marked, and how many fits the grid
    makes on `origins` origins."""
    for stage, listed in (("stage 1", candidates.stage1), ("stage 2", candidates.stage2)):
        print(f"        {stage}: {len(listed)} settings")
        for i, candidate in enumerate(listed):
            values = ", ".join(f"{key} {value}" for key, value in candidate.params.items())
            print(f"          {i}: {values}{'   <- current' if i == 0 else ''}")
    print(f"        top-ups: {len(candidates.topups)} settings")
    for i, topups in enumerate(candidates.topups):
        if topups.method == "mean":
            method = f"mean of {topups.window} weeks"
        else:
            method = f"ridge {'with' if topups.with_level else 'without'} the level"
        print(f"          {i}: {method}{'   <- current' if i == 0 else ''}")
    pairs = len(candidates.stage1) * len(candidates.stage2) * len(candidates.topups)
    fits = origins * (len(candidates.stage1) + len(candidates.stage2))
    print(f"        {pairs} combinations to score; per origin {len(candidates.stage1)} stage-1 "
          f"and {len(candidates.stage2)} stage-2 fits; {origins} origins: {fits} fits (those "
          f"made before are read from the cache)")


def show_grid_predictions(predictions, rolled):
    """Step 18's forecasts of every candidate (U16.3): their size, each candidate's mean p,
    sum of m or sum of top-ups (all origins and h), and whether the current settings
    (candidate 0) give step 17's p and m exactly, row by row (`rolled`; None if step 17 did
    not forecast with the model)."""
    rows = predictions.rows
    print(f"        {len(rows):,} customer rows at {rows['origin_week'].nunique()} origins")
    for i in predictions.p.columns:
        print(f"        stage 1 setting {i}: mean p {predictions.p[i].mean():.4f}")
    for j in predictions.m.columns:
        print(f"        stage 2 setting {j}: sum of m {predictions.m[j].sum() / 1e6:,.1f} M "
              f"(TRY of week 0)")
    for k, topups in enumerate(predictions.topups):
        print(f"        top-ups setting {k}: sum {topups['topup'].sum() / 1e6:,.1f} M "
              f"(TRY of week 0)")
    if rolled is None:
        print("        candidate 0 against step 17: not checked (step 17 did not forecast "
              "with the model)")
        return
    customers = rolled.customers
    same_rows = predictions.p.index.equals(customers.index)
    same_p = same_rows and bool((predictions.p[0].to_numpy() == customers["p"].to_numpy()).all())
    same_m = same_rows and bool((predictions.m[0].to_numpy() == customers["m"].to_numpy()).all())
    print(f"        candidate 0 the same as step 17, row by row: p {'yes' if same_p else 'NO'}, "
          f"m {'yes' if same_m else 'NO'}")


def show_grid_gates(stage, gates, score, ascending):
    """Step 18's gates for one stage (U16.4): how many candidates passed, how many failed each
    gate, a loud line if the current settings (candidate 0) fail one, and every candidate
    sorted by `score`."""
    number_column = stage.replace(" ", "")
    gate_columns = [column for column in gates.columns if column.endswith(" gate")]
    failed = {column: int((~gates[column]).sum()) for column in gate_columns}
    print(f"        {stage} gates: {int(gates['eligible'].sum())} of {len(gates)} candidates "
          f"eligible; candidates failing each gate: {failed}")
    current = gates[gates[number_column] == 0].iloc[0]
    if not current["eligible"]:
        missed = [column for column in gate_columns if not current[column]]
        print(f"        the current {stage} settings FAIL {', '.join(missed)}; they are still "
              f"scored, as the settings to beat")
    others = gates[gates[number_column] != 0]
    if not others.empty and not others["eligible"].any():
        print(f"        no other {stage} candidate is eligible: only the current settings go on")
    print(gates.sort_values(score, ascending=ascending).round(4).to_string(index=False))


def show_grid_scores(table, candidates, rolled, actual, horizons):
    """Step 18's scores (U16.5): how many combinations were scored, the 10 best by WAPE, the
    current combination and its rank, and whether it scores exactly as step 17's model
    (`rolled`, scored here on `actual`; None if step 17 did not forecast with the model)."""
    every = len(candidates.stage1) * len(candidates.stage2) * len(candidates.topups)
    print(f"        {len(table)} of {every} combinations scored (stage-1 and stage-2 "
          f"candidates eligible or current, every top-up); the 10 best by WAPE:")
    print(table.head(10).round(4).to_string(index=False))
    rank = int(table.index[table["current"]][0]) + 1
    current = table[table["current"]].iloc[0]
    print(f"        current settings: rank {rank} of {len(table)}, WAPE {current['wape']:.4f}, "
          f"5-week WAPE {current['total_wape']:.4f}, bias {current['bias']:+.4f}, company "
          f"WAPE {current['company_wape']:.4f}")
    if rolled is None:
        print("        current against step 17: not checked (step 17 did not forecast with "
              "the model)")
        return
    hurdle = rolled.totals[["company_code", "origin_week", "h", "target_week", "total"]]
    hurdle = hurdle.rename(columns={"total": "forecast"}).assign(method=HURDLE)
    step17 = evaluation_scores(hurdle[STACK_COLUMNS], actual, horizons)
    step17_wape = step17.loc[(step17["company_code"] == CONSOLIDATED)
                             & (step17["horizon"] == POOLED), "wape"].iloc[0]
    print(f"        current WAPE the same as step 17's model: "
          f"{'yes' if step17_wape == current['wape'] else 'NO'} (step 17 {step17_wape:.6f}, "
          f"grid {current['wape']:.6f})")


def show_grid_choice(choice, candidates, table, settings):
    """Step 18's choice (U16.6): kept or new settings (beside the current ones), the WAPE
    gain, the calibration chosen for it, and on the same origins its scores beside the
    current settings and the references, with the paired bootstrap."""
    winner = choice.winner
    current = table[table["current"]].iloc[0]
    min_gain = settings.grid.min_gain
    if winner["current"]:
        print(f"        kept the current settings: no combination beats their WAPE "
              f"{current['wape']:.4f} by more than {min_gain} (best {table['wape'].min():.4f})")
    else:
        print(f"        NEW settings: WAPE {winner['wape']:.4f} against {current['wape']:.4f} "
              f"(gain {current['wape'] - winner['wape']:.4f}, more than {min_gain}):")
        for stage, chosen, now in (("stage 1", choice.stage1.params, candidates.stage1[0].params),
                                   ("stage 2", choice.stage2.params, candidates.stage2[0].params),
                                   ("top-ups", choice.topups, candidates.topups[0])):
            print(f"          {stage}: {chosen}" + ("" if chosen == now else f" (was {now})"))
    print("        its calibration (U15.4 on its forecast):")
    show_calibration_choice(choice.calibration, settings.calibration.min_gain)
    print(f"        on the same origins, uncorrected ({HURDLE} = the combination kept):")
    show_evaluation_scores(choice.scores, settings.receivables.horizons)
    show_paired_bootstrap(choice.bootstrap)


def show_grid_test(test, choice, settings):
    """Step 18's v1 test (U16.7): what was frozen, then the scores on the test origins beside
    the references, with the paired bootstrap."""
    calibration = choice.calibration
    print(f"        frozen on validation: stage 1 {choice.stage1.params}, stage 2 "
          f"{choice.stage2.params}, top-ups {choice.topups.method} (window "
          f"{choice.topups.window}, with_level {choice.topups.with_level}); corrections "
          f"{candidate_name(calibration.p_correction, calibration.m_correction)}")
    print(f"        on the test origins ({HURDLE} = corrected; {UNCORRECTED} for information "
          f"only):")
    show_evaluation_scores(test.scores, settings.receivables.horizons)
    show_paired_bootstrap(test.bootstrap)
