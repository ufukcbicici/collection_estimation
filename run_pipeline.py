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

import pandas as pd  # noqa: E402

from collection_estimation import config  # noqa: E402
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
    # Step 3 — features                       (design sections 9.1, 9.2, 9.3)
    # ==================================================================================
    not_built_yet("Step 3  features", "design sections 9.1, 9.2, 9.3")

    # ==================================================================================
    # Step 4 — the two stages                 (design sections 6 and 7)
    # ==================================================================================
    not_built_yet("Step 4  models", "design sections 6 and 7")

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
    }


if __name__ == "__main__":
    results = main()
