# collection_estimation

Weekly collections forecasting for EY Turkey: given collections up to week `t`, predict the
total collected in weeks `t+1` … `t+5`.

The approach is a **per-customer two-stage (hurdle) model** — classify whether a customer
pays in the target week, regress how much given that they do, multiply, and sum across
customers. The full specification, with the mathematics and four algorithm boxes, lives in
the cockpit repo:

```
cfo-cash-decision-cockpit-unified/services/cash_core/docs/collections/
    weekly_collection_forecast_design.md      <- the design (and a PDF beside it)
    daily_collections_corpus.md               <- what the source files look like
```

**Read the design document before changing anything here.** The module layout below maps
one-to-one onto its algorithm boxes.

---

## Status

Scaffolding only. No model is implemented yet.

---

## Environment

A dedicated conda env, because no existing environment on this machine had scikit-learn
alongside a current numpy and pandas.

```
conda create -n collection_estimation -y python=3.11 numpy pandas scikit-learn \
    matplotlib openpyxl pytest joblib
```

There is no `python` on PATH on this machine. Invoke by full path:

```
& "C:\Users\ufuk.bicici\anaconda3\envs\collection_estimation\python.exe" <script.py>
```

Not yet installed, and needed only if we move past Random Forest to the Gamma objective of
the design's §7.4: `lightgbm` or `xgboost`.

---

## Data

`data/raw/` holds a **copy** of the synthetic corpus — 306 daily Excel workbooks plus the
oracle that describes them. It is gitignored: 5.6 MB of generated data that is exactly
reproducible from a seed in the cockpit repo.

```
data/raw/SYNTHETIC_ey_tahsilat_20250602_20260828/   306 workbooks, one per business day
data/raw/collections_corpus.json                    the oracle: ground truth and parameters
```

**This is synthetic data.** Company names are real or realistic; every amount, date,
document number and payment pattern is fabricated. Each workbook carries a `_SYNTHETIC`
tab saying so. It is calibrated to aggregate statistics EY supplied, but its *dynamics* —
payment cadence, invoice-to-cash lag, customer correlation — are invented. Error rates
measured here are not estimates of error rates on EY's real files.

To refresh it, regenerate in the cockpit repo and copy again:

```
python services/cash_core/fixtures/generators/gen_collections_corpus.py --seed 20260908
```

### The oracle

`collections_corpus.json` records what was planted: per-customer archetypes and lifecycles,
the correlation groups with their *measured* conditional probabilities, the hidden weekly
billing series, and the benchmark results.

**Do not use oracle fields as model features.** They exist only because the corpus is
synthetic, and a model that uses them cannot be rebuilt on real data. Use them to diagnose
after the fact.

---

## Layout

```
src/collection_estimation/
    ingest.py      306 workbooks -> one tidy collections table
    panel.py       Algorithm 1 — the (customer, week, horizon) panel
    features.py    §9.1 customer history, §9.2 calendar, §9.3 co-occurrence
    models.py      stage 1 classifier + calibration, stage 2 regressor + smearing
    compose.py     Algorithm 3 — combine into a weekly total
    evaluate.py    Algorithm 4 — rolling origin, metrics, error decomposition
    baselines.py   naive, moving average, calendar-only
tests/
data/
```

---

## The bar to beat

Measured on this corpus (MAPE, by horizon in weeks):

| model | h=1 | h=2 | h=3 | h=4 | h=5 |
|---|---|---|---|---|---|
| last week's value | 29.3 | 30.9 | 37.4 | 27.7 | 28.2 |
| 4-week average | 25.8 | 25.0 | 23.7 | 22.9 | 23.4 |
| 8-week average | 23.8 | 23.6 | 22.6 | **20.9** | **21.2** |
| calendar + recent level | **20.7** | **20.5** | 22.8 | 20.5 | 22.0 |
| calendar only | 20.9 | 20.5 | **20.8** | 21.2 | 21.6 |

`run_pipeline.py` prints this on every run, so the bar stays in front of you.

**The bar is ~20.7% at h=1** — but no model dominates. The calendar advantage is a
short-horizon one: clear at h=1 and h=2, gone by h=5, where the 8-week average wins.

Three things to know:

- **The recent-level term does not earn its place.** `calendar_only` has one parameter
  fewer and is better averaged over horizons. The weekly series is barely autocorrelated
  (+0.10 at lag 1, −0.13 at lag 3), so a level coefficient adds variance, not signal.
  Do not assume a recent-level feature will help the per-customer model either.
- **These numbers belong to this corpus.** They are sensitive to where the six missing
  files fall — one lands on 2025-09-30, the last business day of September, the worst
  possible place for a calendar model. Regenerating the corpus moves them.
- **Every model under-predicts.** The calendar model's bias at h=1 is about −12.8M TRY on
  a 133.7M weekly mean. MAPE and WAPE hide the sign; that is why all three are reported.
