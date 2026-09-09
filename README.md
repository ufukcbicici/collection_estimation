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
horizon. Six features:

| feature | |
|---|---|
| `level` | mean of the 4 weeks ending `h` weeks before the target |
| `n_business_days` | 0–5 |
| `has_month_end` | the week contains the last **business** day of a month |
| `has_quarter_end` | … of March, June, September or December |
| `n_holidays` | weekday public holidays in the week |
| `trend` | the integer week index |

On the synthetic corpus, at h=1: **MAPE 19.83 ± 2.49, WAPE 19.70, bias −1.0%.**

**Chosen for the bias, not the accuracy.** Every candidate's MAPE sits inside one standard
error of every other, so this was never an accuracy choice — it is the only competitive model
that is not systematically 10–19% low. Beyond h=3 an 8-week moving average is just as good.

### The bar it has to beat

MAPE by horizon, measured on the synthetic corpus:

| model | h=1 | h=2 | h=3 | h=4 | h=5 |
|---|---|---|---|---|---|
| last week's value | 29.3 | 30.9 | 37.4 | 27.7 | 28.2 |
| 4-week average | 25.8 | 25.0 | 23.7 | 22.9 | 23.4 |
| 8-week average | 23.8 | 23.6 | 22.6 | **20.9** | **21.2** |
| calendar only | 20.9 | 20.5 | **20.8** | 21.2 | 21.6 |
| calendar + recent level | 20.7 | 20.5 | 22.8 | 20.5 | 22.0 |
| **ridge + trend** (recommended) | **19.8** | **20.1** | 22.0 | 21.6 | 21.8 |

`run_pipeline.py` prints this on every run, so the bar stays in front of you.

### The one thing to get right when reading these

**Differences under about 2 MAPE points are not distinguishable.** With ~52 origins the
standard error is ~2.5, and during development a single missing file moved a model by 3.7
points. Every table the pipeline prints carries `mape_se` for this reason.

To compare two models properly, compare them **paired** — same origins, difference of their
errors, confidence interval on the difference. The origin-to-origin swings that dominate the
error bar are common to both models and cancel. Two independent error bars can only say
"cannot distinguish"; a paired interval can say "equal".

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
