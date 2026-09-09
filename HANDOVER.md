# Running this pipeline on the real EY files

This repo forecasts EY Turkey's weekly cash collections: given collections up to week `t`,
predict the total collected in weeks `t+1` … `t+5`.

Everything in it was built and measured against a **synthetic** corpus — 306 fabricated
daily workbooks. You have the real files. This document is how you point the pipeline at
them.

**The short version:** set one environment variable, run `preflight.py`, fix whatever it
reports in `config.py`, run `run_pipeline.py`. If the real files have the same layout as our
synthetic ones, no code changes at all.

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

On our synthetic corpus, at h=1: `MAPE 19.83 ± 2.49, WAPE 19.70, bias −1.0%`.

**It was chosen for the bias, not the accuracy.** Every candidate's MAPE sat inside one
standard error of every other. What separates this one is that it is not systematically
10–19% low, which for a cash forecast matters more than a point of MAPE.

---

## 6. How to read the output — the one thing to get right

The pipeline prints a **standard error** next to every MAPE. Use it.

With ~52 rolling origins, **differences smaller than about 2 MAPE points are not
distinguishable from noise.** During development a single missing file moved a model by 3.7
points. Without the error bar we would have "improved" 20.89 → 19.83 and believed it.

If you compare two models properly, compare them **paired** — same origins, difference of
their errors, confidence interval on that difference. The origin-to-origin swings dominate
the standard error and are *common to both models*, so they cancel. Two independent error
bars can only ever say "cannot distinguish"; a paired interval can say "equal".

Expect the real data to be **harder** than ours, not easier.

---

## 7. What will not carry over — read before trusting any number

Everything measured here comes from a corpus whose dynamics we invented. Treat these as
hypotheses to re-test, not findings:

**The trend.** The synthetic weekly total drifts upward by ~3%/month because *I set that
parameter*. It is not an EY measurement. Real Turkish nominal collections are non-stationary
too, so some drift handling is needed — but check the actual drift, and consider deflating by
a price index rather than fitting a straight line.

**The level term earns nothing.** Measured: dropping it entirely costs −0.00 APE points
(95% CI [−1.37, +1.36]). The cause is that our weekly series is white noise once detrended.
Real collections plausibly *do* persist — a large payer slipping a week, a backlog clearing.
It is kept in the model as cheap insurance for exactly that reason. **Re-run the comparison
on real data** rather than assuming either answer.

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
