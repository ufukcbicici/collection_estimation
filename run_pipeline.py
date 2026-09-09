"""Run the forecasting pipeline step by step.

Open this file in PyCharm and press Run. No arguments, no CLI — every knob is a constant
at the top of the file, and every step is one call inside `main()` so you can put a
breakpoint between any two and inspect what came out.

    Step 1  ingest    read the 306 daily workbooks into one table
    Step 2  panel     week spine + customer-week aggregation (Algorithm 1, line 1)
    Step 3  features  sections 9.1 / 9.2 / 9.3       (not built yet)
    Step 4  models    stages 1 and 2                 (not built yet)
    Step 5  evaluate  Algorithm 4, rolling origin    (not built yet)

The design document is in the cockpit repo at
`services/cash_core/docs/collections/weekly_collection_forecast_design.md`.
"""
from __future__ import annotations

import os
import sys
import time

# The package lives under `src/`. Adding it here means this script runs from PyCharm, from
# a terminal, and from any working directory without the project needing to be installed
# or `src` marked as a Sources Root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from collection_estimation import config  # noqa: E402
from collection_estimation.baselines import (  # noqa: E402
    mape_table,
    rolling_origin_baselines,
    score,
    weekly_totals,
    weeks_affected_by_missing_files,
)
from collection_estimation.calendar_features import (  # noqa: E402
    build_calendar_features,
    cross_check_business_days,
)
from collection_estimation.cross_features import (  # noqa: E402
    TOP_K,
    build_cross_features,
    find_partners,
)
from collection_estimation.history_features import (  # noqa: E402
    attach_to_panel,
    build_history_features,
)
from collection_estimation.ingest import (  # noqa: E402
    read_collections_csv,
    read_corpus,
    write_collections_csv,
)
from collection_estimation.weekly_models import (  # noqa: E402
    compare_weekly_models,
)
from collection_estimation.panel import (  # noqa: E402
    build_panel,
    build_week_index,
    first_week,
    to_customer_weeks,
    weeks_without_collections,
)

# ======================================================================================
# Knobs
# ======================================================================================

#: True re-reads all 306 workbooks (~25 s). False uses the cached CSV if it exists.
#: Set this to True after regenerating the corpus, or the cache goes stale silently.
FORCE_REINGEST = False

#: How many rows `describe()` prints.
PREVIEW_ROWS = 8


# ======================================================================================
# Helpers
# ======================================================================================

def describe(table: pd.DataFrame, name: str) -> None:
    """Print enough about a table to see what happened without opening a debugger."""
    print(f"\n--- {name} " + "-" * max(0, 68 - len(name)))
    print(f"shape      : {table.shape[0]:,} rows x {table.shape[1]} columns")

    if "document_date" in table.columns:
        lo, hi = table["document_date"].min(), table["document_date"].max()
        print(f"date range : {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}  "
              f"({table['document_date'].nunique()} distinct days)")

    if "company_code" in table.columns:
        counts = table["company_code"].value_counts().sort_index()
        print("by entity  : " + "  ".join(
            f"{code} {n:,}" for code, n in counts.items()))

    if "document_currency" in table.columns:
        counts = table["document_currency"].value_counts()
        total = len(table)
        print("by currency: " + "  ".join(
            f"{cur} {n / total:.1%}" for cur, n in counts.items()))

    if "amount_local" in table.columns:
        amounts = table["amount_local"]
        print(f"amount_local: total {amounts.sum():,.0f} TRY   "
              f"mean {amounts.mean():,.0f}   "
              f"min {amounts.min():,.0f}   max {amounts.max():,.0f}")
        print(f"             negatives {int((amounts < 0).sum())}   "
              f"zeros {int((amounts == 0).sum())}")

    print(f"dtypes     : {dict(table.dtypes.astype(str))}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.head(PREVIEW_ROWS))


def load_collections(force: bool = False) -> pd.DataFrame:
    """Step 1. The merged collections table, from cache when we have one.

    Reading 306 workbooks takes ~25 seconds; the cached CSV loads in well under one. The
    cache is written to `data/interim/`, which is gitignored.
    """
    if not force and os.path.exists(config.COLLECTIONS):
        print(f"loading cache  {config.COLLECTIONS}")
        table = read_collections_csv(config.COLLECTIONS)
        print(f"  {len(table):,} rows from cache "
              f"(set FORCE_REINGEST = True to re-read the workbooks)")
        return table

    print(f"reading corpus {config.CORPUS_DIR}")
    started = time.perf_counter()
    table = read_corpus(config.CORPUS_DIR, progress=True)
    print(f"  {len(table):,} rows in {time.perf_counter() - started:.1f}s")

    write_collections_csv(table, config.COLLECTIONS)
    print(f"  cached to {config.COLLECTIONS}")
    return table


def _oracle_missing_files() -> list[str]:
    """Business days the corpus has no file for, as declared by the oracle.

    Only available because this corpus is synthetic. On real data the equivalent comes
    from `cross_check_business_days`, which finds the same days empirically.
    """
    import json
    try:
        with open(config.ORACLE, encoding="utf-8") as handle:
            return json.load(handle)["ey_spec_alignment"]["missing_files"]
    except (OSError, KeyError):
        return []


def not_built_yet(step: str, where: str) -> None:
    print(f"\n=== {step} " + "=" * max(0, 66 - len(step)))
    print(f"    not built yet — see {where}")


# ======================================================================================
# The pipeline
# ======================================================================================

def main() -> dict:
    """Run every step that exists. Returns the intermediates, for the console runner."""
    print("=" * 72)
    print("collection_estimation — weekly collections forecast")
    print("=" * 72)

    # ==================================================================================
    # Step 1 — ingest and merge
    # ==================================================================================
    collections = load_collections(force=FORCE_REINGEST)
    describe(collections, "collections")

    # >>> BREAKPOINT HERE to inspect `collections`.
    #     One row per collection. Try:
    #         collections.groupby("document_date")["amount_local"].sum()
    #         collections[collections["amount_local"] < 0]          # refunds
    #         collections[collections["sap_document_no"].isna()]    # unapplied cash

    # ==================================================================================
    # Step 2 — the week spine and customer-week aggregation
    #          (design section 12, Algorithm 1, line 1)
    # ==================================================================================
    weeks = build_week_index(collections)
    customer_weeks = to_customer_weeks(collections, weeks)

    print(f"\n--- weeks " + "-" * 63)
    print(f"{len(weeks)} weeks, {weeks['iso_week'].iloc[0]} .. "
          f"{weeks['iso_week'].iloc[-1]}  "
          f"({weeks['week_start'].iloc[0]:%Y-%m-%d} .. "
          f"{weeks['week_end'].iloc[-1]:%Y-%m-%d})")
    empty = weeks_without_collections(customer_weeks, weeks)
    print(f"weeks with no collections at all: {empty if empty else 'none'}")

    describe(customer_weeks, "customer_weeks")
    dense = collections["customer"].nunique() * len(weeks)
    print(f"sparsity   : {len(customer_weeks):,} observed of {dense:,} possible "
          f"customer-weeks ({len(customer_weeks) / dense:.2%})")
    print(f"negative net: {int((customer_weeks['amount_net'] < 0).sum())} "
          f"customer-weeks  (why positives and negatives are not netted)")

    # >>> BREAKPOINT HERE to inspect `weeks` and `customer_weeks`. Try:
    #         customer_weeks.groupby("week")["amount_net"].sum()      # the weekly series
    #         customer_weeks["customer"].value_counts()               # appearance counts
    #         customer_weeks[customer_weeks["amount_negative"] < 0]   # refunds

    # ==================================================================================
    # Step 2b — the dense panel               (Algorithm 1, lines 5-11)
    # ==================================================================================
    panel = build_panel(customer_weeks, weeks)
    describe(panel, "panel")

    cells = int((len(weeks) - first_week(customer_weeks)).sum())
    cell_rate = len(customer_weeks[customer_weeks["amount_positive"] > 0]) / cells
    print(f"base rate  : {panel['z'].mean():.2%} of panel rows are positive "
          f"({cell_rate:.2%} of customer-week cells)")
    print("             the gap is expected: a customer's FIRST appearance can never be")
    print("             a target, and 42% of the roster appears exactly once")
    print("rows/horizon: " + "  ".join(
        f"h{h} {n:,}" for h, n in panel.groupby('horizon', observed=True).size().items()))
    print(f"memory     : {panel.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    # >>> BREAKPOINT HERE to inspect `panel`. Try:
    #         panel.groupby("horizon")["z"].mean()          # base rate by horizon
    #         panel[panel["z"] == 1]["y"].describe()        # the amount target
    #         panel[panel["y_negative"] < 0]                # weeks containing a refund

    # ==================================================================================
    # Step 3a — the calendar block            (design section 9.2)
    # ==================================================================================
    calendar = build_calendar_features(weeks)
    describe(calendar, "calendar")

    print(f"month ends : {int(calendar['has_month_end'].sum())}   "
          f"quarter ends: {int(calendar['has_quarter_end'].sum())}")
    short = calendar[calendar["n_business_days"] < 5]
    print(f"short weeks: {len(short)} of {len(calendar)} lose a business day to a "
          f"public holiday")

    # The hardcoded holiday table is the weak part of this block, so the disagreement
    # against the days the corpus actually has files for is reported rather than assumed.
    missing_dates = _oracle_missing_files()
    report = cross_check_business_days(weeks, collections)
    print(f"\ncross-check against observed file dates: {len(report)} disagreement(s)")
    if len(report):
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(report.to_string(index=False))
        print("  `expected_working_no_data` should be the corpus's 6 deliberately")
        print("  missing files — those are absent observations, NOT holidays.")

    # >>> BREAKPOINT HERE to inspect `calendar` and `report`. Try:
    #         calendar[calendar["has_month_end"] == 1]
    #         calendar[calendar["n_holidays"] > 0]

    # ==================================================================================
    # Step 3b — customer history              (design section 9.1)
    # ==================================================================================
    started = time.perf_counter()
    history = build_history_features(customer_weeks, weeks)
    features = attach_to_panel(panel, history)
    print(f"\n--- history " + "-" * 61)
    print(f"{len(history):,} rows x {history.shape[1]} columns "
          f"in {time.perf_counter() - started:.1f}s, joined onto the panel "
          f"({len(features):,} rows)")

    # NaN is deliberate here and confined to the genuinely undefined. Printing the rates
    # keeps that visible rather than letting a surprise NaN hide among expected ones.
    nan_rates = history.isna().mean()
    nan_rates = nan_rates[nan_rates > 0].sort_values(ascending=False)
    print("NaN by column (deliberate — undefined, not missing):")
    for column, rate in nan_rates.items():
        print(f"    {column:<12} {rate:6.1%}")
    if nan_rates.empty:
        print("    none")

    # >>> BREAKPOINT HERE to inspect `history` and `features`. Try:
    #         history[history["has_gap"] == 1]["gap_median"].describe()
    #         features[["rec", "phase", "mean_4", "z"]].corr()

    # ==================================================================================
    # Step 3c — cross-customer                (design section 9.3)
    # ==================================================================================
    started = time.perf_counter()
    cross = build_cross_features(customer_weeks, weeks)
    print(f"\n--- cross-customer " + "-" * 54)
    print(f"{len(cross):,} rows in {time.perf_counter() - started:.1f}s "
          f"(partners rebuilt at EVERY origin — building once over all weeks would "
          f"leak\n  future co-payments, which is the subtlest trap in section 10.2)")

    partners_now = find_partners(customer_weeks, up_to_week=len(weeks) - 1)
    sizes = [len(v) for v in partners_now.values()]
    print(f"at the last origin: {len(partners_now):,} of "
          f"{customer_weeks['customer'].nunique():,} customers have a partner "
          f"({len(partners_now) / customer_weeks['customer'].nunique():.1%})")
    if sizes:
        at_cap = sum(1 for s in sizes if s == TOP_K)
        print(f"  partners each: mean {np.mean(sizes):.2f}  "
              f"{at_cap}/{len(sizes)} at the cap of {TOP_K} — the rest are limited by "
              f"the lift floor, not by k")

    by_week = cross.groupby("week", observed=True)["has_partners"].mean()
    print(f"  share of rows with a partner: week {by_week.index[0]} "
          f"{by_week.iloc[0]:.1%} -> week {by_week.index[-1]} {by_week.iloc[-1]:.1%}")
    print("  empty early on by design: a lift estimate needs ten paying weeks first")

    # >>> BREAKPOINT HERE to inspect `cross` and `partners_now`.

    # ==================================================================================

    # ==================================================================================
    # Step 4a — the baselines: the bar to beat   (design section 11.2)
    # ==================================================================================
    series = weekly_totals(customer_weeks, weeks)
    evaluations = rolling_origin_baselines(series, calendar)
    table = mape_table(evaluations)

    print("\n=== Step 4a  baselines " + "=" * 49)
    print(f"weekly series: mean {series.mean():,.0f} TRY   "
          f"cv {series.std() / series.mean():.3f}   {len(series)} weeks")
    print(f"origins      : {evaluations['origin'].nunique()}   "
          f"evaluations {len(evaluations):,}")
    print("\nMAPE % by horizon — lower is better:")
    print(table.round(2).to_string())

    scored = score(evaluations)
    h1 = scored[scored["horizon"] == 1].set_index("model")
    print("\nat h=1, all three metrics:")
    print(h1[["mape", "wape", "bias"]].round(2).to_string())

    affected = weeks_affected_by_missing_files(series, weeks, missing_dates)
    print(f"\n{len(affected)} of {len(series)} weeks contain a day whose file is absent. "
          "They are\nKEPT in the evaluation — dropping them would flatter every model by")
    print("removing errors it genuinely makes.")

    best = table[1].idxmin()
    print(f"\nTHE BAR: {best} at {table.loc[best, 1]:.1f}% MAPE (h=1).")
    print("A per-customer model that cannot beat this is not paying for its complexity.")

    # >>> BREAKPOINT HERE to inspect `series`, `evaluations`, `table`. Try:
    #         evaluations[evaluations["model"] == "calendar_aware"].nlargest(5, "actual")
    #         series.plot()   # if you want to look at it

    # ==================================================================================
    # Step 4b — weekly candidate models       (the per-customer hurdle is PARKED)
    # ==================================================================================
    candidates = compare_weekly_models(series, calendar)
    everything = pd.concat([evaluations, candidates], ignore_index=True)
    scored_all = score(everything)

    print("\n=== Step 4b  weekly models " + "=" * 45)
    print("MAPE by horizon:")
    print(scored_all.pivot(index="model", columns="horizon", values="mape")
          .round(2).to_string())

    at_h1 = scored_all[scored_all["horizon"] == 1].set_index("model")
    at_h1 = at_h1.assign(bias_pct=(at_h1["bias"] / series.mean() * 100).round(1))
    at_h1 = at_h1.sort_values("mape")[["mape", "mape_se", "wape", "bias_pct"]]
    print("\nat h=1, sorted by MAPE:")
    print(at_h1.round(2).to_string())

    spread = at_h1["mape"].max() - at_h1["mape"].min()
    typical_se = at_h1["mape_se"].median()
    print(f"\nspread across models {spread:.1f} MAPE points against a typical standard")
    print(f"error of {typical_se:.1f} — so MAPE does NOT separate these models. What does")
    print("separate them is BIAS: the weekly total grows 43% across the corpus, and any")
    print("model without a trend term regresses toward a historical mean well below the")
    print("present level. A moving average is least biased only because it is most recent.")

    # >>> BREAKPOINT HERE to inspect `candidates` and `scored_all`.

    # ==================================================================================

    # ==================================================================================
    # Step 5 — rolling-origin evaluation      (design section 12, Algorithm 4)
    # ==================================================================================
    not_built_yet("Step 5  evaluate", "design section 12, Algorithm 4")

    print("\n" + "=" * 72)
    print("done. Next unit: the dense panel (Algorithm 1, lines 5-11).")
    print("=" * 72)

    # Returned so that running this file in PyCharm's Python Console leaves every
    # intermediate bound to a name you can poke at afterwards.
    return {
        "collections": collections,
        "weeks": weeks,
        "customer_weeks": customer_weeks,
        "panel": panel,
        "calendar": calendar,
        "history": history,
        "features": features,
        "cross": cross,
        "series": series,
        "evaluations": evaluations,
    }


if __name__ == "__main__":
    results = main()
