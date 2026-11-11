"""Printouts of steps 12-15: stage 1 at one origin and its diagnostic, stage 2 at the same
origin and its scores, and the tables of the joint diagnostic of both stages.
"""
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


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
