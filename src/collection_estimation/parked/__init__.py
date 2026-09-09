"""Work that is finished and tested but is NOT on the production path.

Nothing in the pipeline imports from this package, and nothing here should import from it
either — the dependency runs one way, `parked` -> the pipeline modules, never back.

Two kinds of thing live here, and the distinction matters:

**Parked, not disproven.** `hurdle_panel`, `history_features` and `cross_features` are the
per-customer two-stage model of the design document. They were never run, because refitting
on ~800,000 panel rows at each of ~52 origins destroys the iteration loop. They are
complete, tested, and waiting for a reason to be worth the compute.

**Measured and rejected.** `derived_regressors` and `variants` were tried and lost. They are
kept because **their docstrings are the record of what was measured** — the numbers, and why
each failed. Deleting the code would delete the finding, and the next person would try the
same thing again.

See `parked/README.md`, and section 12b of the design document.
"""
