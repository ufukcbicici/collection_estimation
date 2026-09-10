"""Run the weekly collections forecast, step by step.

Open this file and press Run. No arguments, no CLI — every knob is a constant at the top,
and every step is one call inside `main()` so you can put a breakpoint between any two and
inspect what came out.

    Step 1  ingest    read the daily workbooks into one table
    Step 2  weeks     week spine, customer-week aggregation, the weekly total series
    Step 3  calendar  the calendar block, plus a cross-check of the holiday table
    Step 4  evaluate  baselines and the recommended model, rolling origin

To run on real EY files, set one environment variable and change nothing here:

    COLLECTIONS_CORPUS_DIR = <the folder of daily .xlsx files>

Run `src/collection_estimation/preflight.py` against that folder FIRST. See HANDOVER.md.

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

import pandas as pd  # noqa: E402

from collection_estimation import config  # noqa: E402
from collection_estimation.baselines import (  # noqa: E402
    BASELINE_MODELS,
    paired_comparison,
    rolling_origin_baselines,
    score,
    weekly_totals,
)
from collection_estimation.calendar_features import (  # noqa: E402
    build_calendar_features,
    cross_check_business_days,
)
from collection_estimation.ingest import (  # noqa: E402
    read_collections_csv,
    read_corpus,
    write_collections_csv,
)
from collection_estimation.panel import (  # noqa: E402
    build_week_index,
    to_customer_weeks,
    weeks_without_collections,
)
from collection_estimation.weekly_models import (  # noqa: E402
    RECOMMENDED,
    evaluate_recommended,
)

# ======================================================================================
# Knobs
# ======================================================================================

#: True re-reads every workbook. False uses the cached CSV if one exists.
#: The cache filename includes the corpus folder name, so switching data sets cannot
#: silently reuse the wrong cache — but set this True after EDITING files in place.
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
        print(f"amount_local: total {amounts.sum():,.0f}   "
              f"mean {amounts.mean():,.0f}   "
              f"min {amounts.min():,.0f}   max {amounts.max():,.0f}")
        print(f"             negatives {int((amounts < 0).sum())}   "
              f"zeros {int((amounts == 0).sum())}")

    print(f"dtypes     : {dict(table.dtypes.astype(str))}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.head(PREVIEW_ROWS))


def load_collections(force: bool = False) -> pd.DataFrame:
    """Step 1. The merged collections table, from cache when we have one.

    Reading the corpus takes tens of seconds; the cached CSV loads in well under one.
    The cache is written to `data/interim/`, which is gitignored.
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


def _missing_file_dates(weeks: pd.DataFrame,
                        collections: pd.DataFrame) -> list[str]:
    """Business days in range with no collections at all.

    On the synthetic corpus these are the deliberately absent files, and the oracle
    declares them. On real data there is no oracle, so they are found the same way
    `cross_check_business_days` finds them — empirically, from the days that have rows.

    They are reported, never dropped. A missing source day is an ABSENT observation, not
    zero cash; dropping the weeks that contain one would flatter every model by removing
    errors it genuinely makes.
    """
    report = cross_check_business_days(weeks, collections)
    absent = report[report["kind"] == "expected_working_no_data"]
    return [str(d) for d in absent["date"]]


# ======================================================================================
# The pipeline
# ======================================================================================

def main() -> dict:
    """Run every step. Returns the intermediates, for the console runner."""
    print("=" * 74)
    print("collection_estimation — weekly collections forecast")
    print("=" * 74)
    print(f"corpus: {config.CORPUS_DIR}")
    if config.IS_SYNTHETIC:
        print("        *** SYNTHETIC DATA — amounts, dates and payment patterns are")
        print("        *** fabricated. Error rates measured here are not estimates of")
        print("        *** error rates on real files.")

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
    # Step 2 — the week spine, the customer-week aggregation, and the target series
    # ==================================================================================
    weeks = build_week_index(collections)
    customer_weeks = to_customer_weeks(collections, weeks)
    series = weekly_totals(customer_weeks, weeks)

    print("\n--- weeks " + "-" * 63)
    print(f"{len(weeks)} weeks, {weeks['iso_week'].iloc[0]} .. "
          f"{weeks['iso_week'].iloc[-1]}  "
          f"({weeks['week_start'].iloc[0]:%Y-%m-%d} .. "
          f"{weeks['week_end'].iloc[-1]:%Y-%m-%d})")
    empty = weeks_without_collections(customer_weeks, weeks)
    print(f"weeks with no collections at all: {empty if empty else 'none'}")
    print(f"customers  : {customer_weeks['customer'].nunique():,}")
    print(f"weekly total: mean {series.mean():,.0f}   "
          f"cv {series.std() / series.mean():.3f}   "
          f"min {series.min():,.0f}   max {series.max():,.0f}")

    # A rising or falling series is the single most important thing to know before
    # modelling: it is what the trend regressor exists for, and it was the cause of a
    # 10% under-prediction that three other explanations failed to account for.
    half = len(series) // 2
    drift = series.iloc[half:].mean() / series.iloc[:half].mean() - 1
    print(f"drift      : second half is {drift:+.1%} of the first half")
    if abs(drift) > 0.10:
        print("             large enough to matter — this is why the model carries a")
        print("             trend term. On real data prefer deflating by a price index.")

    # >>> BREAKPOINT HERE to inspect `weeks`, `customer_weeks` and `series`. Try:
    #         series.plot()
    #         customer_weeks["customer"].value_counts()
    #         customer_weeks[customer_weeks["amount_negative"] < 0]   # refunds

    # ==================================================================================
    # Step 3 — the calendar block            (design section 9.2)
    # ==================================================================================
    calendar = build_calendar_features(weeks)
    describe(calendar, "calendar")

    print(f"month ends : {int(calendar['has_month_end'].sum())}   "
          f"quarter ends: {int(calendar['has_quarter_end'].sum())}")
    short = calendar[calendar["n_business_days"] < 5]
    print(f"short weeks: {len(short)} of {len(calendar)} lose a business day to a "
          f"public holiday")

    # The hardcoded holiday table is the weak part of this block, so the disagreement
    # against the days that actually have files is reported rather than assumed.
    report = cross_check_business_days(weeks, collections)
    print(f"\ncross-check against observed file dates: {len(report)} disagreement(s)")
    if len(report):
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(report.to_string(index=False))
        print("  `expected_working_no_data` = a working day by our table with no rows:")
        print("    either a holiday we have not declared, or a genuinely missing file.")
        print("  `holiday_with_data`        = our holiday table is wrong for that date.")

    # >>> BREAKPOINT HERE to inspect `calendar` and `report`.

    # ==================================================================================
    # Step 4 — rolling-origin evaluation     (design section 12, Algorithm 4)
    # ==================================================================================
    baselines = rolling_origin_baselines(series, calendar)
    recommended = evaluate_recommended(series, calendar, weeks)
    evaluations = pd.concat([baselines, recommended], ignore_index=True)
    scored = score(evaluations)

    print("\n=== Step 4  rolling-origin evaluation " + "=" * 36)
    print(f"origins    : {evaluations['origin'].nunique()}   "
          f"evaluations {len(evaluations):,}")

    print("\nMAPE % by horizon — lower is better:")
    print(scored.pivot(index="model", columns="horizon", values="mape")
          .round(2).to_string())

    at_h1 = scored[scored["horizon"] == 1].set_index("model")
    at_h1 = at_h1.assign(bias_pct=(at_h1["bias"] / series.mean() * 100).round(1))
    at_h1 = at_h1.sort_values("mape")[["mape", "mape_se", "wape", "bias_pct"]]
    print("\nat h=1, sorted by MAPE:")
    print(at_h1.round(2).to_string())

    absent = _missing_file_dates(weeks, collections)
    if absent:
        print(f"\n{len(absent)} business day(s) have no file. The weeks containing them "
              "are KEPT in\nthe evaluation — dropping them would flatter every model by "
              "removing errors it\ngenuinely makes.")

    # The error bar is the point. Without it a model gets "improved" into noise.
    spread = at_h1["mape"].max() - at_h1["mape"].min()
    typical_se = at_h1["mape_se"].median()
    print(f"\nspread across models {spread:.1f} MAPE points against a typical standard")
    print(f"error of {typical_se:.1f}. Treat any difference smaller than about twice the")
    print("standard error as no difference at all.")

    # Paired against the recommendation. The unpaired standard errors above cannot
    # separate these models; differencing on shared origins removes the origin-to-origin
    # variation that dominates them, and can therefore say "equal" rather than only
    # "cannot distinguish".
    paired = paired_comparison(evaluations, RECOMMENDED)
    pooled = paired[paired["horizon"].isna()].set_index("model")
    print(f"\npaired vs '{RECOMMENDED}', pooled over all horizons "
          f"(negative = better):")
    for model in pooled.index:
        row = pooled.loc[model]
        verdict = "distinguishable" if row["lo"] > 0 or row["hi"] < 0 else "-"
        print(f"  {model:<22} {row['delta']:+6.2f}  "
              f"95% CI [{row['lo']:+6.2f}, {row['hi']:+6.2f}]  {verdict}")

    recommended_mape = at_h1.loc[RECOMMENDED, "mape"]
    recommended_bias = at_h1.loc[RECOMMENDED, "bias_pct"]
    # Only the declared baselines can be "the bar". A candidate must never be allowed to
    # serve as its own benchmark.
    bars = [m for m in BASELINE_MODELS if m in at_h1.index]
    best_baseline = at_h1.loc[bars, "mape"].idxmin()
    print(f"\nRECOMMENDED MODEL '{RECOMMENDED}': {recommended_mape:.1f}% MAPE, "
          f"{recommended_bias:+.1f}% bias at h=1.")
    print(f"Best baseline is '{best_baseline}' at "
          f"{at_h1.loc[best_baseline, 'mape']:.1f}%.")
    print("The model is chosen for its BIAS, not its MAPE — see weekly_models.py.")

    # >>> BREAKPOINT HERE to inspect `evaluations` and `scored`. Try:
    #         evaluations[evaluations["model"] == RECOMMENDED].nlargest(5, "actual")
    #         scored[scored["horizon"] == 1]

    print("\n" + "=" * 74)
    print("done.")
    print("=" * 74)

    # Returned so that running this file in a Python Console leaves every intermediate
    # bound to a name you can poke at afterwards.
    return {
        "collections": collections,
        "weeks": weeks,
        "customer_weeks": customer_weeks,
        "series": series,
        "calendar": calendar,
        "evaluations": evaluations,
        "scored": scored,
    }


if __name__ == "__main__":
    results = main()
