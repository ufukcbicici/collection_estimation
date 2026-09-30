# collection_estimation

Weekly collections forecasting for EY Turkey: given collections up to week `t`, predict the
total collected in weeks `t+1` … `t+5`.

**Running this on the real EY files? Read [`HANDOVER.md`](HANDOVER.md).** It is the setup
guide, the list of what to change, and the list of what will not carry over.

```powershell
$env:COLLECTIONS_CORPUS_DIR = "D:\ey\tahsilat"
python src\collection_estimation\preflight.py     # diagnose the files first
python run_pipeline.py                            # then run
```

---

## The model

**Ridge regression on the weekly total**, refit at every rolling origin, a separate fit per
horizon, over a calendar block plus a correction for nominal drift. Six features:

| feature | |
|---|---|
| `level` | mean of the 4 weeks ending `h` weeks before the target |
| `n_business_days` | 0–5 |
| `has_month_end` | the week contains the last **business** day of a month |
| `has_quarter_end` | … of March, June, September or December |
| `n_holidays` | weekday public holidays in the week |
| `trend` | the integer week index — used **only** by the models that do not deflate |

**How the drift is handled is the choice that matters**, and on the real EY files dividing by
the price index beats fitting a straight line.

### The recommendation — measured on the real EY files, run of 2026-09-11

| if | use | h=1 | across h=1…5 | bias |
|---|---|---|---|---|
| one model must serve every horizon | `deflated_nolevel` | 21.5% | flat **21.3–22.7%** | +1.2% |
| the one-week number is what matters | `deflated` | **18.1%** | 18.1 → 24.6 | −0.4% |

`deflated` divides the weekly series by Turkish TÜFE, models the stationary real series, then
re-inflates the prediction — see [`inflation.py`](src/collection_estimation/inflation.py). Its
design matrix carries **no `trend` column**, deliberately: deflating has already removed the
drift, and fitting a trend on top would model it twice.

Against the previous recommendation `ridge+trend` (20.4% at h=1), paired on the same origins,
the gap is **−1.36 [−2.32, −0.40]** — outside the noise.

The recommendation is horizon-dependent because the `level` term is: **every model carrying it
decays with horizon, every model without it stays flat.**

### The bar it has to beat

MAPE by horizon on the real EY files, with the h=1 standard error and the signed bias. Sorted
by h=1, which is how the pipeline prints it.

| model | h=1 | h=2 | h=3 | h=4 | h=5 | ±se | bias |
|---|---|---|---|---|---|---|---|
| **`deflated`** — recommended at h=1 | **18.1** | 20.7 | 22.0 | 24.0 | 24.6 | 2.41 | −0.4% |
| `ridge+trend` — previous recommendation | 20.4 | 22.1 | 23.5 | 24.4 | 25.9 | 2.53 | +1.6% |
| `calendar_only` | 21.3 | 21.3 | 21.3 | 22.1 | 22.4 | 2.37 | **−11.3%** |
| **`deflated_nolevel`** — recommended for h=1…5 | 21.5 | 21.6 | **21.3** | **22.0** | **22.7** | 3.39 | +1.2% |
| `lags5+trend` | 21.8 | 23.1 | 24.5 | 27.5 | 26.0 | 2.81 | −1.2% |
| `ridge+trend_nolevel` | 21.8 | 21.4 | 21.6 | 22.2 | 22.4 | 3.20 | −4.1% |
| `calendar_aware` | 22.2 | 22.1 | 23.2 | 23.6 | 24.3 | 2.23 | **−11.9%** |
| `ma8` | 28.6 | 28.6 | 29.5 | 28.9 | 27.1 | 6.20 | −2.3% |
| `ma4` | 32.0 | 30.6 | 29.7 | 29.0 | 28.2 | 7.20 | −1.3% |
| `naive` | 39.6 | 43.8 | 39.7 | 37.8 | 34.6 | 7.36 | −0.5% |

`run_pipeline.py` prints this on every run, so the bar stays in front of you. Read the bias
column as carefully as the MAPE: **the two models with no drift correction run ~11% low**,
which for a cash forecast is the error that matters and which MAPE hides entirely.

**R² is reported beside them from this run on.** `score` returns it per model and horizon,
and `run_pipeline.py` prints it twice: a full model-by-horizon table and a column in the h=1
summary. It answers a different question from the other three. MAPE and WAPE average absolute
errors and say how wrong a typical week was; R² is squared error, so the heavy weeks dominate
and it says whether the model followed the *shape* of the series at all. Its baseline is the
mean of the weeks being forecast, which is hindsight, so it ranks models against each other
and is never a bar anyone could have run — and a negative value just means worse than that
flat line, which is ordinary on a series whose variance sits in a few weeks.

**The table above carries no R² column because the real files have not been re-run since it
was added.** On the v1 synthetic corpus it re-ranks, which is the point of having it:
`calendar_only` scores −0.14 to −0.18 across the horizons despite a competitive MAPE,
because it runs ~19% low there and squared error will not forgive that, while
`deflated_nolevel` holds the best R² at h=2…5 — the same horizon-aware picture the MAPE
table gives. Put the measured figures here after the next real run; do not carry the
synthetic ones across.

### The one thing to get right when reading these

**Differences under about 2 MAPE points are not distinguishable.** With ~52 origins the
standard error is ~3.0 on the real files (~2.5 on the v1 synthetic corpus), and during
development a single missing file moved a model by 3.7 points. Every table the pipeline prints
carries `mape_se` for this reason.

To compare two models properly, compare them **paired** — same origins, difference of their
errors, confidence interval on the difference. The origin-to-origin swings that dominate the
error bar are common to both models and cancel. Two independent error bars can only say
"cannot distinguish"; a paired interval can say "equal".

**The five horizons are not independent.** They share the origin, the training window and
most of the same target weeks. Do not count "all five horizons agree" as five confirmations
or derive a p-value from it — it is one consistent pattern. For the same reason the
five-week cumulative error has to be measured directly rather than inferred from the
per-horizon numbers; `reporting.cumulative_by_origin` does that.

### Error is not only a percentage, and not only a size

`reporting.py` reports error in **lira**, and splits **over-** from **under-forecasting**
(`error = predicted − actual`, so positive is over). They are different risks: an
over-forecast is a liquidity exposure, an under-forecast is a carrying cost. Neither is
"the" error that matters, and reporting only one of them hides a model that looks safe
purely by forecasting far too low.

---

## Layout

```
run_pipeline.py               the entry point
HANDOVER.md                   how to run this on real EY files
src/collection_estimation/
    config.py                 paths + the workbook layout — the only file to edit
    preflight.py              diagnose a workbook against config, without raising
    ingest.py                 daily workbooks -> one tidy collections table
    panel.py                  week spine + customer-week aggregation
    calendar_features.py      the calendar block + the Turkish holiday table
    baselines.py              metrics, rolling-origin harness, the baselines
    weekly_models.py          the recommended model
    inflation.py              the TÜFE deflator and its leak-free re-inflation
    reporting.py              per-forecast results file, error in lira, over/under risk
    parked/                   built and tested, NOT on the production path
tests/                        218 tests
data/                         gitignored — no client data ever enters git
```

### `parked/`

Two different things, both kept deliberately. See [`parked/README.md`](src/collection_estimation/parked/README.md).

- **Parked, not disproven** — the per-customer two-stage (hurdle) model of the design
  document: the dense panel, the customer-history block, the cross-customer co-occurrence
  block. Complete and tested; never run, because refitting ~800,000 rows at each of ~52
  origins destroys the iteration loop.
- **Measured and rejected** — the log target, Tweedie/Gamma, and derived cadence regressors.
  Kept because their docstrings *are* the record of what was measured. Delete the code and
  the next person tries the same thing again.

---

## Data

`data/raw/` holds a **copy** of the synthetic corpus — 306 daily workbooks plus an oracle
describing what was planted in them — and, beside the folder, a **synthetic AR ledger**
(`SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx`) generated from the same simulation, so the
two tie to the kuruş. Its layout and rules are in the cockpit repo's
`services/cash_core/docs/collections/ar_ledger_schema.md`. Gitignored: all of it is
reproducible from a seed in the cockpit repo.

**Two synthetic corpora exist, and the figures in this repo mostly describe the older one.**

| | seed | installed | notes |
|---|---|---|---|
| **v1** | 20260908 | until 2026-09-30 | archived in `data/raw/_archive/v1_seed20260908/`, with its oracle and its ingest cache |
| **v2** | 58068921 | since 2026-09-30 | adds the AR ledger, instalments, unique customer numbers; seed chosen as the most typical of ten random ones |

**Every synthetic figure quoted in this repo — here, in `HANDOVER.md`, in the model
docstrings — was measured on v1** unless it says otherwise. They are records of experiments,
kept rather than re-measured. The real-data figures are unaffected. v1's seed was also the
easiest of the seeds tried, so its error rates are on the flattering side.

**It is synthetic.** Company names are realistic; every amount, date, document number and
payment pattern is fabricated. Each workbook carries a `_SYNTHETIC` tab saying so. It is
calibrated to aggregate statistics EY supplied, but its *dynamics* — payment cadence,
invoice-to-cash lag, customer correlation — are invented. **Error rates measured here are not
estimates of error rates on EY's real files.**

To regenerate:

```
python services/cash_core/fixtures/generators/gen_collections_corpus.py --seed 58068921
```

It writes the corpus folder and the ledger beside it, and overwrites the cockpit repo's
tracked oracle. Copy all three into `data/raw/`, and **delete the matching cache in
`data/interim/`**: the cache is named after the corpus folder, and a regenerated corpus keeps
the same folder name, so the pipeline would otherwise keep reading the old one.

**Do not use oracle fields as model features.** They exist only because the corpus is
synthetic, and a model that uses them cannot be rebuilt on real data. Use them to diagnose
after the fact.

---

## Environment

A dedicated conda env, because no existing environment had scikit-learn alongside a current
numpy and pandas.

```
conda create -n collection_estimation -y python=3.11 numpy pandas scikit-learn \
    matplotlib openpyxl pytest joblib
```

There is no `python` on PATH on the development machine; invoke by full path:

```
& "C:\Users\ufuk.bicici\anaconda3\envs\collection_estimation\python.exe" run_pipeline.py
```

---

## The specification

The mathematics, the four algorithm boxes, and — in **section 12b** — what building it
actually found, including five corrections to the document's own proposals:

```
cfo-cash-decision-cockpit-unified/services/cash_core/docs/collections/
    weekly_collection_forecast_design.md
    daily_collections_corpus.md
```
