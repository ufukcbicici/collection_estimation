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

### The one thing to get right when reading these

**Differences under about 2 MAPE points are not distinguishable.** With ~52 origins the
standard error is ~3.0 on the real files (~2.5 on the synthetic corpus), and during
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
tests/                        159 tests
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
describing what was planted in them. Gitignored: it is reproducible from a seed in the
cockpit repo.

**It is synthetic.** Company names are realistic; every amount, date, document number and
payment pattern is fabricated. Each workbook carries a `_SYNTHETIC` tab saying so. It is
calibrated to aggregate statistics EY supplied, but its *dynamics* — payment cadence,
invoice-to-cash lag, customer correlation — are invented. **Error rates measured here are not
estimates of error rates on EY's real files.**

To regenerate:

```
python services/cash_core/fixtures/generators/gen_collections_corpus.py --seed 20260908
```

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
