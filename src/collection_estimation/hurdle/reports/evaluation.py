"""Printouts of steps 16-17: one forecast beside the cash that came, then the evaluation
over many origins (the model, the references, the scores, the paired bootstrap, where
the bias comes from, the balance coherence and the calibration).
"""
import math

import pandas as pd

from ..evaluate import HORIZON_TOTAL, KNOWN_CUSTOMERS, POOLED, REFERENCES, RULE
from ..forecast import CONSOLIDATED
from .stages import show_amount_reliability


def show_customer_forecasts(customers, forecast_rows, seconds, step14_expected):
    """Step 16's customer forecasts (U13.2): their size and fit time, per h the sum of p and
    of p x m against the deflated cash that came (labelled rows only; a missing week is not a
    zero), and whether p x m equals step 14's, row for row."""
    origin = int(customers["origin_week"].iloc[0])
    print(f"Step 16: one forecast at week {origin}: {len(customers):,} customer rows "
          f"({customers['key'].nunique():,} customers x {customers['h'].nunique()} h), both "
          f"stages fitted in {seconds:.1f} s")
    step14 = step14_expected.loc[customers.index].rename("expected")
    print(f"        p x m equals step 14's, row for row: {customers['expected'].equals(step14)} "
          f"(largest difference {(customers['expected'] - step14).abs().max():.3g})")
    labelled = forecast_rows.loc[customers.index, "label_status"] == "labelled"
    rows = customers.assign(actual=forecast_rows["amount_deflated"])[labelled]
    by_h = rows.groupby("h").agg(customers=("key", "size"), sum_p=("p", "sum"),
                                 p_x_m=("expected", "sum"), actual=("actual", "sum"))
    by_h["ratio"] = by_h["p_x_m"] / by_h["actual"]
    print("        per h, labelled rows, deflated (TRY of week 0):")
    print(by_h.round({"sum_p": 1, "p_x_m": 0, "actual": 0, "ratio": 3}).to_string())


def show_forecast_totals(totals):
    """Step 16's totals (U13.3): the consolidated forecast per h, million TRY (deflated parts,
    then the price level and the nominal total), and every company at h = 1."""
    columns = ["customers", "new_customer_cash", "unidentified_cash", "refunds",
               "real_total", "total"]
    consolidated = totals[totals["company_code"] == CONSOLIDATED].set_index("h")
    table = consolidated[columns] / 1e6
    table.insert(5, "price_level", consolidated["price_level"])
    print(f"        consolidated forecast at week {int(totals['origin_week'].iloc[0])}, million "
          f"TRY (all deflated, TRY of week 0, except total, nominal = real_total x price_level):")
    print(table.round({**{column: 2 for column in columns}, "price_level": 4}).to_string())
    first = totals[(totals["h"] == 1)].set_index("company_code")
    print("        per company at h = 1, million TRY:")
    print((first[["customers", "real_total", "total"]] / 1e6).round(2).to_string())


def show_actual_totals(actual, weekly, weeks):
    """Step 16's actual cash (U14.1): its size, and a tie-out of the consolidated rows to the
    weekly amounts of their target weeks."""
    consolidated = actual[actual["company_code"] == CONSOLIDATED]
    in_weekly = weekly[weekly["week"].isin(consolidated["target_week"])]
    expected = (in_weekly["gross_receipts"] + in_weekly["refunds"]).sum()
    missing = int(weeks["is_missing"].sum())
    print(f"        actual cash (U14.1): {len(actual)} rows ({actual['company_code'].nunique()} "
          f"companies with ALL); consolidated {consolidated['actual'].sum():,.2f} TRY vs the "
          f"weekly amounts of its target weeks {expected:,.2f} TRY; {missing} missing weeks "
          f"in the calendar (no rows)")


def show_forecast_vs_actual(totals, actual):
    """Step 16's forecast beside the nominal cash that really came in each target week
    (`actual_totals`: gross receipts + refunds, refunds negative; hindsight, one origin, so an
    illustration, not a score). A target week without data has no actual: never 0."""
    keys = ["company_code", "origin_week", "h", "target_week"]
    rows = totals[keys + ["total"]].merge(actual[keys + ["actual"]], on=keys, how="left",
                                          validate="one_to_one")
    rows["ratio"] = rows["total"] / rows["actual"]

    origin = int(totals["origin_week"].iloc[0])
    print(f"        forecast vs actual at week {origin}, nominal (actual = gross receipts + "
          f"refunds; ratio = forecast / actual; one origin: an illustration, not a score)")
    print("        consolidated, million TRY:")
    all_rows = rows[rows["company_code"] == CONSOLIDATED].set_index("h")
    shown = all_rows[["target_week"]].assign(forecast=all_rows["total"] / 1e6,
                                             actual=all_rows["actual"] / 1e6,
                                             ratio=all_rows["ratio"])
    print(shown.round(2).fillna("no data").to_string())
    print("        ratio per company and h:")
    print(rows.pivot(index="company_code", columns="h", values="ratio")
          .round(2).fillna("no data").to_string())


def show_rolling_forecasts(rolled, actual):
    """Step 17's forecasts (U14.2): per origin and h, the consolidated forecast / actual
    (nominal; a target week without data shows "no data"), and the sums over all of them.
    Scores come with U14.5."""
    keys = ["company_code", "origin_week", "h", "target_week"]
    rows = rolled.totals[keys + ["total"]].merge(actual[keys + ["actual"]], on=keys,
                                                 how="left", validate="one_to_one")
    consolidated = rows[rows["company_code"] == CONSOLIDATED]
    ratio = (consolidated.assign(ratio=consolidated["total"] / consolidated["actual"])
             .pivot(index="origin_week", columns="h", values="ratio"))
    print(f"        {rolled.customers.shape[0]:,} customer rows, {len(rolled.totals)} total rows; "
          f"consolidated forecast / actual per origin and h (nominal):")
    print(ratio.round(2).fillna("no data").to_string())
    with_actual = consolidated.dropna(subset=["actual"])
    print(f"        over all {len(with_actual)} consolidated (origin, h) with data: forecast "
          f"{with_actual['total'].sum() / 1e6:,.1f} M vs actual "
          f"{with_actual['actual'].sum() / 1e6:,.1f} M TRY (ratio "
          f"{with_actual['total'].sum() / with_actual['actual'].sum():.3f})")


def show_deflated_reference(reference, rolled, actual):
    """Step 17's `deflated` reference (U14.3): over all consolidated (origin, h) with data,
    the model's forecast (if `rolled` is not None) and each reference's against the actual
    cash (nominal), and how often each fell back to the recent level there. Scores come with
    U14.5."""
    keys = ["company_code", "origin_week", "h", "target_week"]
    rows = actual[actual["company_code"] == CONSOLIDATED][keys + ["actual"]]

    # One column per reference: its forecast, and whether it fell back.
    for name in REFERENCES:
        mine = reference[reference["reference"] == name]
        mine = mine[keys + ["forecast", "fallback"]].rename(
            columns={"forecast": name, "fallback": name + " fallback"})
        rows = rows.merge(mine, on=keys, validate="one_to_one")
    lines = list(REFERENCES)
    if rolled is not None:
        rows = rows.merge(rolled.totals[keys + ["total"]], on=keys, validate="one_to_one")
        lines = ["total"] + lines

    print(f"        the deflated reference (U14.3): {len(reference)} rows; over the "
          f"{len(rows)} consolidated (origin, h) with data, nominal, million TRY:")
    came = rows["actual"].sum()
    for column in lines:
        name = "hurdle model" if column == "total" else column
        fell_back = ""
        if column in REFERENCES:
            fell_back = (f", fell back to the recent level in "
                         f"{int(rows[column + ' fallback'].sum())} of {len(rows)}")
        print(f"          {name:<17} {rows[column].sum() / 1e6:>10,.1f} vs actual "
              f"{came / 1e6:,.1f} (ratio {rows[column].sum() / came:.3f}){fell_back}")


def show_rule_reference(rule, actual):
    """Step 17's financial-rule reference (U14.4): over the consolidated (origin, h) with
    data, the rule's forecast against the actual cash, in total and per h (late invoices all
    land in h = 1); and what the rule decided for the open invoices, summed over the origins.
    Scores come with U14.5."""
    keys = ["company_code", "origin_week", "h", "target_week"]
    consolidated = actual[actual["company_code"] == CONSOLIDATED]
    rows = consolidated[keys + ["actual"]].merge(rule.totals, on=keys, validate="one_to_one")

    forecast, came = rows["forecast"].sum(), rows["actual"].sum()
    print(f"          {'financial rule':<17} {forecast / 1e6:>10,.1f} vs actual "
          f"{came / 1e6:,.1f} (ratio {forecast / came:.3f})")

    # Per h: the forecast / actual, and how much of the invoice cash is late invoices.
    by_h = rows.groupby("h")[["forecast", "actual", "invoice_cash", "late_cash"]].sum()
    by_h["ratio"] = by_h["forecast"] / by_h["actual"]
    by_h["late share"] = by_h["late_cash"] / by_h["invoice_cash"]
    print("          the rule per h, consolidated, million TRY:")
    shown = by_h[["forecast", "actual"]] / 1e6
    shown["ratio"] = by_h["ratio"]
    shown["late share"] = by_h["late share"]
    print(shown.round(3).to_string())

    # What the rule decided for each open invoice, added up over every origin.
    decided = rule.invoices.groupby("status")["outstanding_try"].agg(["size", "sum"])
    decided["sum"] = decided["sum"] / 1e6
    decided.columns = ["invoices", "million TRY"]
    print("          the open invoices at the origins, by what the rule decided "
          "(summed over the origins):")
    print(decided.round(1).to_string())
    # Where the delays came from, as a share of the open TRY.
    by_source = rule.invoices.groupby("delay_from")["outstanding_try"].sum()
    share = by_source / by_source.sum()
    print(f"          the delay came from the customer for {share.get('customer', 0.0):.1%} "
          f"of the open TRY, its company's median for {share.get('company', 0.0):.1%}, "
          f"none (0 days) for {share.get('none', 0.0):.1%}")


def show_evaluation_scores(scores, horizons):
    """Step 17's scores (U14.5): consolidated, one line per method (WAPE, bias and MAPE over
    all (origin, h), WAPE per h and of the per-origin total); then the pooled WAPE per method
    and company."""
    total_name = f"{horizons}-week total WAPE"

    # 1. Consolidated, one line per method.
    consolidated = scores[scores["company_code"] == CONSOLIDATED]
    lines = []
    for method, mine in consolidated.groupby("method", sort=False):
        by_horizon = mine.set_index("horizon")
        line = {"method": method,
                "WAPE": by_horizon.loc[POOLED, "wape"],
                "bias": by_horizon.loc[POOLED, "bias"],
                "MAPE": by_horizon.loc[POOLED, "mape"]}
        for h in range(1, horizons + 1):
            line[f"h{h}"] = by_horizon["wape"].get(str(h), float("nan"))
        line[total_name] = by_horizon["wape"].get(HORIZON_TOTAL, float("nan"))
        lines.append(line)
    pooled_rows = int(consolidated["rows"][consolidated["horizon"] == POOLED].iloc[0])
    print(f"        scores (U14.5), consolidated, over {pooled_rows} (origin, h) with data "
          f"(h1 ... h{horizons}: WAPE per h):")
    print(pd.DataFrame(lines).set_index("method").round(3).to_string())

    # 2. Per company, the pooled WAPE.
    pooled = scores[scores["horizon"] == POOLED]
    methods = list(dict.fromkeys(pooled["method"]))
    companies = list(dict.fromkeys(pooled["company_code"]))
    per_company = pooled.pivot(index="method", columns="company_code", values="wape")
    per_company = per_company.reindex(index=methods, columns=companies)
    print("        pooled WAPE per company:")
    print(per_company.round(3).to_string())


def show_paired_bootstrap(bootstrap):
    """Step 17's paired bootstrap (U14.5c): per pair and measure, the two WAPEs, their
    difference with its 90% interval, and how often the first method was better."""
    samples = int(bootstrap["samples"].iloc[0])
    print(f"        paired bootstrap (U14.5c), consolidated, {samples} resamples; difference = "
          f"WAPE of the first - WAPE of the second (below 0: the first is better):")
    lines = []
    for _, row in bootstrap.iterrows():
        lines.append({"pair": f"{row['method']} vs {row['against']}",
                      "measure": row["horizon"],
                      "WAPE": f"{row['wape']:.3f}",
                      "WAPE against": f"{row['wape_against']:.3f}",
                      "difference": f"{row['difference']:+.3f}",
                      "90% interval": f"[{row['low']:+.3f}, {row['high']:+.3f}]",
                      "first better in": f"{row['share_better']:.0%}"})
    print(pd.DataFrame(lines).set_index("pair").to_string())


def show_component_bias(bias, horizons):
    """Step 17's bias per part (U14.5d), consolidated: each part over all (origin, h), then
    the known customers' forecast / actual per h."""
    consolidated = bias[bias["company_code"] == CONSOLIDATED]

    # 1. Each part over all (origin, h).
    pooled = consolidated[consolidated["horizon"] == POOLED]
    lines = []
    for _, row in pooled.iterrows():
        lines.append({"method": row["method"], "part": row["part"],
                      "forecast": f"{row['forecast'] / 1e6:,.1f}",
                      "actual": f"{row['actual'] / 1e6:,.1f}",
                      "gap": f"{row['gap'] / 1e6:+,.1f}",
                      "ratio": f"{row['ratio']:.3f}"})
    print("        bias per part (U14.5d), consolidated, all (origin, h), nominal, million TRY "
          "(the price level row is part of the total's gap, not added to it):")
    print(pd.DataFrame(lines).set_index("method").to_string())

    # 2. The known customers per h.
    known = consolidated[consolidated["part"] == KNOWN_CUSTOMERS]
    print("        known customers, forecast / actual per h:")
    for method in dict.fromkeys(known["method"]):
        ratios = known[known["method"] == method].set_index("horizon")["ratio"]
        shown = "  ".join(f"h{h} {ratios.get(str(h), float('nan')):.3f}"
                          for h in range(1, horizons + 1))
        name = "rule (open invoices)" if method == RULE else method
        print(f"          {name:<21} {shown}")


def show_payer_split(split, horizons):
    """Step 17's split of the known customers' bias (U14.5d), in TRY of week 0: how many pay
    (p ratio, stage 1) against how much each pays (m ratio, stage 2), consolidated per h,
    then per company over all h."""
    columns = {"horizon": "horizon", "expected_payers": "payers expected",
               "actual_payers": "payers actual", "p_ratio": "p ratio",
               "expected_per_payer": "per payer expected",
               "actual_per_payer": "per payer actual", "m_ratio": "m ratio",
               "pxm_ratio": "p x m ratio"}
    consolidated = split[split["company_code"] == CONSOLIDATED]
    table = consolidated[list(columns)].rename(columns=columns).set_index("horizon")
    print("        known customers, how many pay and how much each pays (real TRY; a ratio "
          "is forecast / actual), consolidated:")
    print(table.round(3).to_string())

    pooled = split[(split["horizon"] == POOLED) & (split["company_code"] != CONSOLIDATED)]
    per_company = pooled[["company_code", "p_ratio", "m_ratio", "pxm_ratio"]]
    per_company = per_company.rename(columns={"company_code": "company", "p_ratio": "p ratio",
                                              "m_ratio": "m ratio",
                                              "pxm_ratio": "p x m ratio"})
    print("        the same per company, all (origin, h):")
    print(per_company.set_index("company").round(3).to_string())


def show_balance_coherence(coherence, factor):
    """Step 17's balance check (U14.6), in TRY of week 0: the 5-week cash against the open
    balance, forecast and actual, then the two groups the forecast overshoots."""
    # 1. The shares, per company and consolidated.
    columns = {"company_code": "company", "forecast_share": "forecast / balance",
               "actual_share": "actual / balance",
               "share_forecast_over": f"forecast > {factor:g}x",
               "share_actual_over": f"actual > {factor:g}x"}
    shares = coherence[list(columns)].rename(columns=columns).set_index("company")
    print("        balance check (U14.6), each customer's cash over the weeks against what it "
          "owes at the origin:")
    print(shares.round(3).to_string())

    # 2. The groups, consolidated.
    total = coherence[coherence["company_code"] == CONSOLIDATED].iloc[0]
    print(f"        above {factor:g}x the balance: {int(total['rows_over']):,} rows, forecast "
          f"{total['forecast_over'] / 1e6:,.1f} M against actual "
          f"{total['actual_over'] / 1e6:,.1f} M")
    print(f"        nothing open: {int(total['rows_nothing_open']):,} rows, forecast "
          f"{total['forecast_nothing_open'] / 1e6:,.1f} M against actual "
          f"{total['actual_nothing_open'] / 1e6:,.1f} M")


def show_calibration_reliability(reliability):
    """Step 17's reliability of the rolling forecasts (U15.1): p against the share that paid,
    all h together and then each h, and m against what was paid."""
    # 1. p, all h together.
    pooled = reliability.p[reliability.p["horizon"] == POOLED].drop(columns="horizon")
    print("        stage 1 reliability, all rolling forecast rows in equal groups by p, all h "
          "(calibrated: mean_p near share_paid in every group):")
    print(pooled.round(4).to_string(index=False))

    # 2. p, each h: one column per h, each cell "mean p / share paid".
    by_h = reliability.p[reliability.p["horizon"] != POOLED].copy()
    by_h["shown"] = [f"{mean_p:.4f} / {share:.4f}"
                     for mean_p, share in zip(by_h["mean_p"], by_h["share_paid"])]
    print("        the same per h, each cell mean p / share paid:")
    print(by_h.pivot(index="group", columns="horizon", values="shown").to_string())

    # 3. m.
    show_amount_reliability(reliability.m)


def show_calibration_choice(choice, min_gain):
    """Step 17's choice of corrections (U15.4): every pair on the judged origins, the winner,
    how sure it is against no correction, and the numbers kept."""
    # 1. Every pair; only wape decides.
    print(f"        corrections learned on target weeks up to {choice.split_week}, judged at "
          f"origins {choice.judged_origins[0]}-{choice.judged_origins[-1]} (only wape decides; "
          f"min_gain {min_gain}):")
    shown = choice.candidates.drop(columns="candidate")
    print(shown.round(4).to_string(index=False))
    print(f"        chosen: {choice.winner}")

    # 2. How sure, if a correction won.
    if choice.bootstrap.empty:
        print("        no correction chosen, so nothing to compare it with")
    else:
        show_paired_bootstrap(choice.bootstrap)

    # 3. The numbers kept.
    print("        the chosen corrections, learned on all these rows and frozen for the test:")
    p_correction, m_correction = choice.p_correction, choice.m_correction
    shown = ", ".join(f"h {name}: slope {a:.3f}, intercept {b:.3f}"
                      for name, (a, b) in p_correction.maps.items())
    print(f"          p, {p_correction.kind}: {shown or 'unchanged'}")
    print(f"          m, {m_correction.kind}: common factor {m_correction.common:.3f}")
    if m_correction.per_company:
        shown = ", ".join(f"{company} {factor:.3f}"
                          for company, factor in m_correction.per_company.items())
        print(f"            {shown}")
    if m_correction.factors:
        shown = ", ".join(f"{math.exp(log_m):,.0f}: {factor:.3f}" for log_m, factor
                          in zip(m_correction.log_m, m_correction.factors))
        print(f"            factor at m = {shown}")
