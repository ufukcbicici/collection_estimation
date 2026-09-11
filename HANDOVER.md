# Running this pipeline on the real EY files

This repo forecasts EY Turkey's weekly cash collections: given collections up to week `t`,
predict the total collected in weeks `t+1` … `t+5`.

Everything in it was **built** against a **synthetic** corpus — 306 fabricated daily
workbooks. This document is how you point the pipeline at the real ones.

**The short version:** set one environment variable, run `preflight.py`, fix whatever it
reports in `config.py`, run `run_pipeline.py`. If the real files have the same layout as our
synthetic ones, no code changes at all.

---

## 0. It has since been run on the real files — read this before §5 and §7

**A run of 2026-09-11 measured all ten models on the real EY workbooks.** Two findings that
this document was built on did not survive it, and both are corrected in place below. If you
are reading an older copy, these are the changes:

| | this document used to say | the real files say |
|---|---|---|
| **the drift** | fit a linear trend; deflating is an untried idea | **deflate by TÜFE.** `deflated` 18.1% MAPE at h=1 vs `ridge+trend` 20.4%; paired **−1.36 [−2.32, −0.40]** |
| **the `level` term** | earns nothing (−0.00 APE) | **horizon-dependent.** Helps clearly at h=1 (18.1 vs 21.5), hurts from h=3 out |

The recommendation is therefore now **horizon-aware**: `deflated_nolevel` if one model must
serve h=1…5 (flat 21.3–22.7%, +1.2% bias), `deflated` if the one-week number is what matters
(18.1%, −0.4% bias). The full table is in [`README.md`](README.md).

What that run did **not** record, and what is still worth writing down if you have it: the
answers to §10 — entity codes, header language, whether `Customer` came through as text, and
what the missing days mean. The modelling results came back; the ingest findings did not.

Measured on the real corpus: **65 weeks, ~52 origins, standard error ≈3.0 MAPE points at
h=1** (worse than the synthetic ≈2.5), and **16 business days with no file**.

### Step 5 is new, and it is what a review asks for

`reporting.py` (added 2026-09-11) produces the things an accuracy table does not:

* **A per-forecast results file**, written to `data/interim/forecast_log__<corpus>.csv`.
  One row per (model, origin, horizon) with the origin date, the **training cutoff**, the
  target week, the actual, the prediction and the **code version** that produced it.
  Written UTF-8-with-BOM so Excel on a Turkish machine opens it without mangling.
* **Error in lira, not only in percent** (`mae_try`), because a percentage of a week whose
  size varies threefold is not something anyone can plan against.
* **Over- and under-forecasting separately.** `error = predicted − actual`, so positive is
  an over-forecast. **These are different risks and neither is "the" risk**: an
  over-forecast is a liquidity exposure, an under-forecast is a carrying cost.
* **The five-week cumulative error per origin**, which is the window a rolling cash plan
  actually covers and which **cannot be derived from the per-horizon table** — errors from
  one origin offset or compound across the five weeks depending on the model.
  `p95_over_try` is the liquidity number: a bad-but-not-unprecedented five-week shortfall.

Two things that run showed on the synthetic corpus, worth re-checking on yours:

1. **The five-week total is far more accurate than any single week** (9-20% against
   20-40%), because the weekly errors partly cancel. Good news for the use case, and
   invisible without this measurement.
2. **The model with no drift correction never over-forecasts at all.** Its liquidity risk
   is zero because it is systematically ~18% low. Reading the over-forecast rate on its
   own would have made the worst model look the safest.

---

## 1. Set up

```powershell
conda create -n collection_estimation -y python=3.11 numpy pandas scikit-learn `
    matplotlib openpyxl pytest joblib
conda activate collection_estimation
```

Confirm the code is healthy before you introduce new data:

```powershell
python -m pytest tests -q          # expect 159 passed
```

Those tests run against the synthetic corpus, which is gitignored. If it is absent, the
tests that need it will fail — ask Ufuk for the folder, or skip straight to step 2 and let
your own data be the test.

---

## 2. Point it at the real files

The pipeline reads a **folder of daily `.xlsx` workbooks, one per business day.**

```powershell
$env:COLLECTIONS_CORPUS_DIR = "D:\ey\tahsilat"
```

That is the only required setting. Optionally, if you want intermediates written somewhere
other than the repo's `data/` folder:

```powershell
$env:COLLECTIONS_DATA_DIR = "D:\ey\work"
```

The parsed table is cached to a CSV whose filename **includes the corpus folder name**, so
pointing at a different folder cannot silently reuse the previous one's cache. If you edit
workbooks in place, set `FORCE_REINGEST = True` at the top of `run_pipeline.py` once.

> **The real files never enter this repository.** `data/` is gitignored. Keep it that way —
> nothing in here should ever carry a client amount or a customer name into git.

---

## 3. Run preflight FIRST

```powershell
python src\collection_estimation\preflight.py
```

This is the important step. The ingest code is deliberately strict — a workbook that does
not match the expected layout **raises** rather than being quietly skipped, because a parser
that silently drops files shows up months later as an unexplained dip in a weekly total, and
by then nobody knows why.

That strictness is right for a run and useless for a first look, where it tells you only the
*first* thing wrong. `preflight.py` reports **everything** wrong in one pass, never raises,
and checks the first, middle and last file — because a column inserted halfway through
fourteen months of a hand-maintained extract would pass a check on file one alone.

Its output is a to-do list for `config.py`. Here is what it can tell you, and what to do.

### `FAIL: missing required column(s)`

The header text does not match. **Add** the real spelling to `config.FIELD_BY_HEADER` as
another key mapping to the same field name — do not replace the English one, so a corpus
that changes header language partway through still reads:

```python
FIELD_BY_HEADER = {
    "company code": "company_code",
    "şirket kodu":  "company_code",      # <- added
    ...
}
```

Matching collapses whitespace and folds case, so `"Company  Code "` already works.

### `FAIL: NO recognised entity codes`

**This is the trap that produces an empty table rather than an error.** A row counts as a
detail row if and only if its Company Code is a key of `config.ENTITY_LABEL`. Codes we know:
`TR02 TR04 TR05 TR06 TR91`. If the real files carry others, every row is skipped and you get
zero rows with no complaint. Add them to `ENTITY_LABEL` and `ENTITY_GROUP`.

### `FAIL: Customer is int, not text`

Customer numbers must stay text. Read as a number they lose any leading zero, and every join
against those customers then misses them silently. Ingest raises on this **on purpose**.

Fix it in the export — write the column as text — or format the column as Text in Excel and
re-save. Do not "fix" it by casting in code; by then the zero is already gone.

### `FAIL: data covers [...] but the moving-holiday table declares only [...]`

`calendar_features.MOVING_HOLIDAYS` hardcodes Ramazan and Kurban Bayramı for **2025 and 2026
only**, because they move about eleven days a year and cannot be computed from a rule.

If your data reaches other years you must extend that table. The pipeline raises
`HolidayCoverageError` rather than continuing, because the failure it prevents is silent: an
undeclared four-day Bayram reads as five ordinary business days, and the model quietly
believes a collapsed week was normal.

The dates in that table are marked approximate. **If you can get EY's official holiday
calendar, use it** — it is better than what we have.

### `warn: N weekday(s) have no file and are NOT a declared holiday`

Two different things, and it matters which: a holiday missing from our table, or a genuinely
absent file. A missing file is an **absent observation, not zero cash** — the pipeline keeps
the weeks containing one rather than dropping them, because dropping them would flatter
every model by removing errors it genuinely makes.

### `warn: header row N looks more like the header`

Set `config.HEADER_ROW` and `FIRST_DATA_ROW`. Ours are 2 and 3 — row 1 is a merged
`MERCURY COLLECTIONS` title.

### Other knobs in `config.py`

| constant | when to change it |
|---|---|
| `FILENAME_DATE_FORMAT` / `_LENGTH` | files are not named `DDMMYYYY…` |
| `STATUS_NOTES` | the Turkish notes marking unapplied cash differ (diagnostics only) |
| `MARKER_TABS` | irrelevant for real files; it hides our synthetic marker tab |

---

## 4. Run it

```powershell
python run_pipeline.py
```

Four steps, one call each, breakpoints marked between them. It prints what it found at every
stage and ends with the model comparison.

```
Step 1  ingest    the daily workbooks -> one table
Step 2  weeks     week spine, customer-week aggregation, the weekly total series
Step 3  calendar  the calendar block + a cross-check of the holiday table
Step 4  evaluate  baselines and the recommended model, rolling origin
Step 5  report    the per-forecast results file, error in lira, over/under, 5-week total
```

`main()` returns every intermediate as a dict, so running it in a Python console leaves
`collections`, `series`, `calendar`, `evaluations` and `scored` bound for you to poke at.

---

## 5. What the model is

**Ridge regression on the weekly total, refit at every origin, a separate fit per horizon.**
Six features:

| feature | |
|---|---|
| `level` | mean of the 4 weeks ending `h` weeks before the target |
| `n_business_days` | 0–5 |
| `has_month_end` | the week contains the last **business** day of a month |
| `has_quarter_end` | … of March, June, September or December |
| `n_holidays` | weekday public holidays in the week |
| `trend` | the integer week index |

Two details that are easy to get wrong and are already right here:

- **The level term is horizon-aware.** For h=3 it averages weeks `t-6…t-3`, not `t-4…t-1`.
  Pairing `y[t]` with the four weeks immediately before it fits a one-step relationship and
  then applies it to a level three weeks stale.
- **Month end means the last *business* day**, not the last calendar day. A month ends on a
  weekend roughly two months in seven, before holidays are even considered.

That describes the **family**. Which member to use is settled in §0: on the real files the
`trend` column loses to TÜFE deflation, and the `level` term's value depends on the horizon.
The figures this section used to quote — `MAPE 19.83 ± 2.49, bias −1.0%` for `ridge+trend` —
are synthetic-corpus figures for a model the real data has since beaten.

What has **not** changed is *why* a model gets picked. **The choice is driven by the bias,
not the accuracy.** Candidates cluster inside a few standard errors of each other on MAPE;
what separates them is whether they run systematically low. On the real files the two models
with no drift correction come in ~11% under, and that is the error a cash forecast cannot
afford.

### The four other models the pipeline runs

Each changes exactly one thing about the reference model, so the comparison is readable.
The right-hand column is what the real files answered:

| model | the question it answers | on the real files |
|---|---|---|
| `ridge+trend_nolevel` | does the `level` term earn anything? | **at h=1 yes, from h=3 no** |
| `lags5+trend` | do five *individual* past weeks beat their mean? | **no** — +1.33 [+0.07, +2.59], the one candidate rejected outright |
| `deflated` | is dividing the drift out better than extrapolating it? | **yes** — −1.36 [−2.32, −0.40] |
| `deflated_nolevel` | both of the above at once | **the flat-across-horizons choice** |

`deflated` replaces the trend regressor with **TÜFE deflation**: divide the series by the
Turkish consumer price index, model the stationary real series, re-inflate the prediction.
See `inflation.py`. Its design matrix deliberately has **no trend column** — deflating has
already removed the drift, and fitting a trend on top would model it twice.

**The re-inflation is leak-free and that is load-bearing.** Converting a real forecast back
to nominal needs the price level at the target week, which is in the future. It is projected
from the last *published* TÜFE figure at the recently published rate — never the actual
future index, which would hand the model perfect foresight of inflation and flatter every
result.

### ⚠ `inflation.py` will need extending

`TUFE` covers **January 2025 – August 2026 only**. Outside that range the deflated models
raise `InflationCoverageError` and are skipped; the other models still run, so you are not
blocked. Extend the table from TÜİK when your data does.

Note that TÜİK **changed the index base from 2003=100 to 2025=100 on 1 January 2026**, so no
single published series spans this period. The table is stated on the old basis throughout
with the newer values chained on. If you extend it, chain through the monthly *rates* rather
than pasting published levels, or you will splice two incompatible bases together.

---

## 6. How to read the output — the one thing to get right

The pipeline prints a **standard error** next to every MAPE. Use it.

With ~52 rolling origins, **differences smaller than about 2 MAPE points are not
distinguishable from noise.** During development a single missing file moved a model by 3.7
points. Without the error bar we would have "improved" 20.89 → 19.83 and believed it.

The pipeline therefore also prints a **paired comparison** against the recommended model —
same origins, difference of the two errors, confidence interval on that difference. The
origin-to-origin swings dominate the standard error and are *common to both models*, so they
cancel. Two independent error bars can only ever say "cannot distinguish"; a paired interval
can say "equal", which is the claim usually needed. Read that table, not the MAPE column.

Expect the real data to be **harder** than ours, not easier. It was: the standard error came
out ≈3.0 rather than ≈2.5, which widens the band inside which nothing is distinguishable.
That is also why the two reversals in §0 are stated as *paired* intervals — unpaired, neither
would have cleared.

---

## 7. What will not carry over — read before trusting any number

Everything measured on the synthetic corpus comes from dynamics we invented. Treat those as
hypotheses to re-test, not findings. **The first two below have since been re-tested on the
real files and are settled; the rest are still open.**

**~~The trend.~~ SETTLED — deflate, do not extrapolate.** The synthetic weekly total drifts
upward by ~3%/month because *I set that parameter*; it was never an EY measurement. What
carried over was the *need* to handle nominal drift, and the real files preferred the price
index to the straight line: `deflated` 18.1% at h=1 against `ridge+trend`'s 20.4%, paired
−1.36 [−2.32, −0.40]. Real TÜFE rose **37.5%** across the 65 weeks, close enough to the
fabricated 43% that the synthetic corpus could not have told these two apart.

**~~The level term earns nothing.~~ SETTLED — it is horizon-dependent.** The synthetic
measurement (dropping it costs −0.00 APE, 95% CI [−1.37, +1.36]) was a property of *our
generator*, whose weekly series is white noise once detrended. It did not survive the real
files: the term **helps clearly at h=1** (18.1 against 21.5) and **hurts from h=3 out**.
Every model carrying it decays with horizon; every model without it stays flat. This is
exactly the persistence the old text guessed at — a large payer slipping a week, a backlog
clearing — and it is why the recommendation is now horizon-aware rather than a single model.

**The four negative results.** Log target, Tweedie/Gamma, daily-modelled-then-summed, and
derived cadence regressors all lost. The code is preserved in `src/collection_estimation/
parked/`, with the numbers in the docstrings. Two of the four depend on properties real data
need not share. One call re-runs the whole set in seconds:

```python
from collection_estimation.parked.variants import compare_weekly_models
from collection_estimation.baselines import score
print(score(compare_weekly_models(series, calendar)))
```

**The error ceiling.** The calendar explains 16–18% of variance; the rest is heavy-tail
noise from individual large payments (the largest single payment averages ~10% of its week).
That is a property of the data, and it may differ in yours.

---

## 8. If you have more than the daily collections files

The single most valuable thing you could add is the **AR / receivables ledger** — open
invoices with amounts and due dates.

Everything here infers *when money will arrive* from the history of when money has arrived.
A receivables ledger replaces that inference with measurement. After four measured dead ends
on the collections files alone, it is the one lever likely to move accuracy beyond the error
bar. If it is obtainable, it is worth more than any further tuning.

---

## 9. Layout

```
run_pipeline.py                    the entry point — run this
HANDOVER.md                        this file
src/collection_estimation/
    config.py                      *** the only file you should need to edit ***
    preflight.py                   diagnose a workbook against config; run this first
    ingest.py                      workbooks -> one tidy table
    panel.py                       week spine + customer-week aggregation
    calendar_features.py           the calendar block + the holiday table
    baselines.py                   metrics, rolling origin, the bar to beat
    weekly_models.py               the recommended model
    parked/                        built and tested, NOT on the production path
tests/                             159 tests
```

The full specification lives in the cockpit repo:

```
services/cash_core/docs/collections/
    weekly_collection_forecast_design.md    the design (with a PDF beside it)
    daily_collections_corpus.md             what the source files look like
```

Section 12b of the design records what building it actually found, including the four
corrections to the design's own proposals.

---

## 10. Questions worth sending back

If you hit any of these, they are worth a message rather than a workaround:

1. Do the real files carry **entity codes** beyond `TR02/04/05/06/91`?
2. Are the **headers in Turkish**, and do they change spelling across fourteen months?
3. Is `Customer` stored as **text or as a number** in the real export?
4. Does the real corpus have days with **no file**, and is a missing day known to mean
   "no collections" or "the extract was not run"? These are different, and the answer
   changes how those weeks should be treated.
5. Is there **more than one currency** in the local-currency column?
