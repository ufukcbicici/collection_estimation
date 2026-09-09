# Parked work

Nothing in this folder is on the production path. `run_pipeline.py` does not import any of
it, and the tests in `tests/parked/` are green.

It is kept for two different reasons, and the difference matters.

## Parked — built, tested, never run

| module | what it is |
|---|---|
| `hurdle_panel.py` | Algorithm 1's dense `(customer, week, horizon)` panel — ~800,000 rows |
| `history_features.py` | §9.1 per-customer history: rolling windows 4/8/13/26, recency, gaps |
| `cross_features.py` | §9.3 cross-customer co-occurrence lift, top-5 partners, rebuilt at every origin |

These three implement the **per-customer two-stage (hurdle) model** the design document
specifies: classify whether a customer pays in the target week, regress how much given that
they do, multiply, sum across customers.

**It was parked on cost, not on evidence.** Refitting on ~800,000 panel rows at each of ~52
origins is hours per experiment, which destroys the iteration loop before anything can be
learned. The blocks are complete and tested; the two model stages were never written.

If the weekly model's accuracy ever needs to improve materially, this is where the unspent
capacity is — but read §12b of the design first, because the weekly experiments suggest the
ceiling is set by the data rather than by the model.

## Rejected — measured, and lost

| module | what it found |
|---|---|
| `derived_regressors.py` | cadence counts collapsed to weekly regressors: real signal, but it **is** the trend (+0.94 with the week index) |
| `variants.py` | log target, Tweedie/Gamma, and the derived-regressor model |

These are kept because **the docstrings are the record of what was measured** — the numbers,
and the reason each failed. Delete the code and the finding goes with it, and the next
person tries the same thing.

## Re-run the rejected ones on real data

Every number in these docstrings was measured on a **65-week synthetic corpus whose dynamics
we invented**. Two results depend on properties real data need not share:

- the Tweedie result depends on the weekly total's variance-mean power (1.1–1.3 here, where
  Gamma assumes 2.0);
- the derived-regressor result depends on the customer population still growing at the end
  of the series, which is an artefact of only having 65 weeks.

One call re-runs the whole set and costs seconds:

```python
from collection_estimation.parked.variants import compare_weekly_models
from collection_estimation.baselines import score

candidates = compare_weekly_models(series, calendar)
print(score(candidates))
```
