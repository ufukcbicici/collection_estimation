"""Calibration (U15): are p and m right on average, once the model has forecast weeks it did
not see? The forecasts come from `evaluate.rolling_forecasts`, so each p and m was made at its
origin, from what was known then, and no model is refitted here.

U15.1, the reliability tables: how far p and m were from what happened. They only report; no
forecast changes.

U15.2, the corrections: each is learned from past forecast rows (`fit_p_correction`,
`fit_m_correction`) and applied to any rows (`apply_p_correction`, `apply_m_correction`), so
that it can be learned on earlier weeks and judged on later ones.

U15.3, the corrected forecast (`corrected_forecast`): the totals again, from the corrected p
and m, with no model refitted.

U15.4, the choice (`choose_calibration`): every pair of corrections learned on the first part of
the validation weeks and judged on the later origins; the simplest good pair then learned again
on all of them and frozen for the test.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .diagnostic import amount_reliability, apply_platt, equal_groups, fit_platt, reliability_table
from .evaluate import (BOOTSTRAP_COLUMNS, HORIZON_TOTAL, POOLED, STACK_COLUMNS,
                       evaluation_scores, paired_wape_bootstrap, with_receipts)
from .forecast import CONSOLIDATED, Forecast, forecast_totals

NONE = "none"                 # leave it as it is (p and m)
ONE_MAP = "one map"           # p: one Platt correction for all h
PER_H = "per h"               # p: one Platt correction for each h
SCALE = "scale"               # m: one factor
PER_COMPANY = "per company"   # m: each company's own factor, pulled towards the common one
BY_SIZE = "by size"           # m: one factor per m group, interpolated in between
P_KINDS = (NONE, ONE_MAP, PER_H)
M_KINDS = (NONE, SCALE, PER_COMPANY, BY_SIZE)

ALL_H = "all"                 # the key of `one map`'s only Platt correction

# The columns of the table of candidates, in order: both corrections, how many numbers they
# learn together, and their scores on the judged origins. wape, total_wape and bias are for the
# consolidated cash, over every (origin, h) (total_wape: per origin, over its h weeks);
# company_wape is the plain mean of the companies' WAPEs. Only wape decides.
CANDIDATE_COLUMNS = ["candidate", "p", "m", "numbers_learned", "wape", "total_wape", "bias",
                     "company_wape"]


@dataclass(frozen=True)
class Reliability:
    """What `calibration_reliability` returns.

    p: `reliability_table`'s columns after `horizon`, which is an h ("1", "2", ...) or POOLED
    (all h together).
    m: `amount_reliability`'s table, on the rows that paid, all h together.
    """
    p: pd.DataFrame
    m: pd.DataFrame


@dataclass(frozen=True)
class PCorrection:
    """A correction of p. `maps` holds Platt's (a, b): nothing for `none`, one under ALL_H for
    `one map`, one per h for `per h`."""
    kind: str
    maps: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MCorrection:
    """A correction of m.

    common: the factor of `scale`, and of a company `per company` did not see; 1 for `none`.
    per_company: `per company`'s factors, company -> factor.
    log_m, factors: `by size`'s factor of each m group, at the group's median log m.
    """
    kind: str
    common: float = 1.0
    per_company: dict = field(default_factory=dict)
    log_m: tuple = ()
    factors: tuple = ()


@dataclass(frozen=True)
class CalibrationChoice:
    """What `choose_calibration` decided: the two corrections, learned on all the validation
    rows and frozen, and how they were chosen."""
    p_correction: PCorrection
    m_correction: MCorrection
    split_week: int             # learned on the target weeks up to this one, for the comparison
    judged_origins: list[int]   # judged at these origins: all their target weeks are later
    candidates: pd.DataFrame    # every pair of corrections, `CANDIDATE_COLUMNS`
    winner: str                 # the chosen row of `candidates`
    bootstrap: pd.DataFrame     # the winner against no correction, pooled; no rows if it is no
                                # correction


def calibration_rows(customers: pd.DataFrame, weekly: pd.DataFrame, weeks: pd.DataFrame,
                     levels: pd.Series) -> pd.DataFrame:
    """The customer forecasts (`customers`, the `customers` of `rolling_forecasts`) whose target
    week has data, plus what each customer really paid: gross_receipts, actual_real (TRY of week
    0, like m) and paid (True if gross_receipts is above 0).

    A customer with no weekly row (`weekly`, from `series.weekly_amounts`) paid 0, a real 0 in a
    week with data (`weeks`, the week calendar). Refunds are not in it: they are a top-up.

    Raises ValueError if there are no customer rows in a week with data, and as `deflate` does.
    """
    # 1. Only the target weeks with data: a week without data is not a week nobody paid.
    with_data = set(weeks.loc[~weeks["is_missing"], "week"])
    rows = customers[customers["target_week"].isin(with_data)]
    if rows.empty:
        raise ValueError("there are no customer forecasts in a week with data")

    # 2. What each customer really paid in its target week.
    rows = with_receipts(rows, weekly, levels)
    rows["paid"] = rows["gross_receipts"] > 0
    return rows


def calibration_reliability(customers: pd.DataFrame, weekly: pd.DataFrame, weeks: pd.DataFrame,
                            levels: pd.Series, groups: int) -> Reliability:
    """Stage 1's and stage 2's reliability over `calibration_rows`.

    Raises ValueError as `calibration_rows`, `reliability_table` and `amount_reliability` do.
    """
    rows = calibration_rows(customers, weekly, weeks, levels)

    # 1. p against the share that paid: each h, then all h together.
    tables = []
    for h in sorted(set(rows["h"])):
        at_h = rows[rows["h"] == h]
        table = reliability_table(at_h["paid"], at_h["p"], groups)
        table.insert(0, "horizon", str(h))
        tables.append(table)
    table = reliability_table(rows["paid"], rows["p"], groups)
    table.insert(0, "horizon", POOLED)
    tables.append(table)
    p_table = pd.concat(tables, ignore_index=True)

    # 2. m against what was paid, on the rows that paid: m is the amount given that they pay.
    paid = rows[rows["paid"]]
    m_table = amount_reliability(paid["actual_real"], paid["m"], groups)
    return Reliability(p=p_table, m=m_table)


def fit_p_correction(rows: pd.DataFrame, kind: str) -> PCorrection:
    """The correction of p named `kind` (one of P_KINDS), learned on `rows` (`calibration_rows`).

    Raises ValueError for another kind, and as `fit_platt` does.
    """
    if kind == NONE:
        return PCorrection(kind)
    if kind == ONE_MAP:
        return PCorrection(kind, {ALL_H: fit_platt(rows["paid"], rows["p"])})
    if kind == PER_H:
        maps = {}
        for h in sorted(set(rows["h"])):
            at_h = rows[rows["h"] == h]
            maps[int(h)] = fit_platt(at_h["paid"], at_h["p"])
        return PCorrection(kind, maps)
    raise ValueError(f"the correction of p is {kind!r}, not one of {list(P_KINDS)}")


def apply_p_correction(rows: pd.DataFrame, correction: PCorrection) -> pd.Series:
    """p of `rows` corrected, with the same index. Raises ValueError for an unknown kind, or if
    `rows` holds an h that `per h` did not learn: the forecasts it learned from were
    incomplete."""
    if correction.kind not in P_KINDS:
        raise ValueError(f"the correction of p is {correction.kind!r}, not one of "
                         f"{list(P_KINDS)}")
    if correction.kind == NONE:
        return rows["p"].copy()
    if correction.kind == ONE_MAP:
        a, b = correction.maps[ALL_H]
        return apply_platt(rows["p"], a, b)

    unseen = sorted(set(rows["h"]) - set(correction.maps))
    if unseen:
        raise ValueError(f"the correction of p learned nothing for h {unseen}")
    corrected = rows["p"].copy()
    for h, (a, b) in correction.maps.items():
        at_h = rows["h"] == h
        corrected[at_h] = apply_platt(rows.loc[at_h, "p"], a, b)
    return corrected


def actual_over_m(paid: pd.DataFrame) -> float:
    """Total actual / total m: the factor that makes m add up on these paid rows. Raises
    ValueError if m adds up to 0 or less: no factor can make nothing add up."""
    total_m = float(paid["m"].sum())
    if total_m <= 0:
        raise ValueError(f"m adds up to {total_m} on {len(paid)} paid rows; no factor can be "
                         f"learned from them")
    return float(paid["actual_real"].sum()) / total_m


def log_of_m(m: pd.Series) -> pd.Series:
    """log m, for `by size`. m below 1 TRY counts as 1, so that a 0 has a logarithm."""
    return np.log(m.clip(lower=1.0))


def fit_m_correction(rows: pd.DataFrame, kind: str, shrink_rows: int,
                     groups: int) -> MCorrection:
    """The correction of m named `kind` (one of M_KINDS), learned on the rows of `rows`
    (`calibration_rows`) that paid: m is the amount given that they pay.

    `per company`: (n x own factor + shrink_rows x common factor) / (n + shrink_rows), with n the
    company's paid rows. `by size`: the paid rows with m of at least 1 TRY in `groups` equal
    groups by m; groups with the same median m are merged into one.

    Raises ValueError for another kind, if there are no paid rows, if the m to learn a factor
    from add up to 0 or less, and as `equal_groups` does (e.g. fewer rows than groups).
    """
    if kind == NONE:
        return MCorrection(kind)
    if kind not in M_KINDS:
        raise ValueError(f"the correction of m is {kind!r}, not one of {list(M_KINDS)}")
    paid = rows[rows["paid"]]
    if paid.empty:
        raise ValueError("there are no paid rows to learn the correction of m from")
    common = actual_over_m(paid)
    if kind == SCALE:
        return MCorrection(kind, common)

    if kind == PER_COMPANY:
        factors = {}
        for company in sorted(set(paid["company_code"])):
            mine = paid[paid["company_code"] == company]
            n = len(mine)
            factors[company] = (n * actual_over_m(mine) + shrink_rows * common) / (n + shrink_rows)
        return MCorrection(kind, common, per_company=factors)

    # By size: learned on the rows with m of at least 1 TRY. log m counts every smaller m as 1,
    # so their groups would share one position; when applied they get the first factor.
    sized = paid[paid["m"] >= 1.0]
    group = pd.Series(equal_groups(sized["m"], groups), index=sized.index)

    # Each group's factor sits at the group's median log m. Groups with the same median
    # (repeated m) become one, with one factor from all their rows: the interpolation needs
    # each position once.
    medians = {}
    for number in range(1, groups + 1):
        medians[number] = float(log_of_m(sized.loc[group == number, "m"]).median())
    median_of_row = group.map(medians)
    log_m, factors = [], []
    for median in sorted(set(median_of_row)):
        mine = sized[median_of_row == median]
        log_m.append(median)
        factors.append(actual_over_m(mine))
    return MCorrection(kind, common, log_m=tuple(log_m), factors=tuple(factors))


def apply_m_correction(rows: pd.DataFrame, correction: MCorrection) -> pd.Series:
    """m of `rows` corrected, with the same index. `per company`: a company it did not learn
    gets the common factor. `by size`: the factor is interpolated in log m between the groups,
    and flat beyond the first and last groups' medians. Raises ValueError for an unknown
    kind."""
    if correction.kind not in M_KINDS:
        raise ValueError(f"the correction of m is {correction.kind!r}, not one of "
                         f"{list(M_KINDS)}")
    if correction.kind == PER_COMPANY:
        factor = rows["company_code"].map(correction.per_company).fillna(correction.common)
        return rows["m"] * factor
    if correction.kind == BY_SIZE:
        factor = np.interp(log_of_m(rows["m"]), correction.log_m, correction.factors)
        return rows["m"] * factor
    return rows["m"] * correction.common


def corrected_forecast(rolled: Forecast, p_correction: PCorrection,
                       m_correction: MCorrection) -> Forecast:
    """`rolled` (`evaluate.rolling_forecasts`) with both corrections applied, as a new
    `Forecast`: each customer's p, m and expected = p x m, and the totals rebuilt from them by
    `forecast_totals`. The corrections are for known customers, so the top-ups and the price
    levels stay as they are, each origin keeping its own. With `none` for both, the result
    equals `rolled`.

    Raises ValueError if the customers and the totals are not for the same origins (a customer
    row without totals would be lost without a word), and as `forecast_totals` does.
    """
    # 1. Correct each customer row.
    customers = rolled.customers.copy()
    customers["p"] = apply_p_correction(rolled.customers, p_correction)
    customers["m"] = apply_m_correction(rolled.customers, m_correction)
    customers["expected"] = customers["p"] * customers["m"]

    # 2. What stays: the company rows' top-ups and price levels. The consolidated rows and the
    # columns that depend on the customers are made again.
    by_company = rolled.totals[rolled.totals["company_code"] != CONSOLIDATED]
    topups = by_company.drop(columns=["customers", "real_total", "price_level", "total"])
    companies = sorted(set(by_company["company_code"]))
    horizons = int(by_company["h"].max())

    origins = sorted(set(by_company["origin_week"]))
    customer_origins = sorted(set(customers["origin_week"]))
    if customer_origins != origins:
        raise ValueError(f"the customers are for origins {customer_origins}, but the totals "
                         f"are for {origins}")

    # 3. Rebuild the totals of each origin, with the price levels projected at that origin.
    origin_totals = []
    for origin in origins:
        at_origin = by_company[by_company["origin_week"] == origin]
        one_per_week = at_origin.drop_duplicates("target_week")
        levels = one_per_week.set_index("target_week")["price_level"]
        totals = forecast_totals(customers[customers["origin_week"] == origin],
                                 topups[topups["origin_week"] == origin],
                                 levels, companies, horizons)
        origin_totals.append(totals)
    return Forecast(totals=pd.concat(origin_totals, ignore_index=True), customers=customers)


def calibration_split(origins: list[int], horizons: int) -> tuple[int, list[int]]:
    """Where to divide the origins' target weeks: the week in their middle, and the origins all
    of whose target weeks come after it. The corrections are learned on the target weeks up to
    that week and judged at those origins; a row of an earlier origin that aims past the week
    is used for neither.

    Raises ValueError if `horizons` is below 1 or no origin comes after the middle.
    """
    if horizons < 1:
        raise ValueError(f"horizons is {horizons}; horizons start at 1")
    if not origins:
        raise ValueError("there are no origins to divide")
    first_target = int(min(origins)) + 1
    last_target = int(max(origins)) + horizons
    split_week = (first_target + last_target) // 2
    judged = sorted({int(origin) for origin in origins if origin >= split_week})
    if not judged:
        raise ValueError(f"no origin of {sorted(origins)} is at or after the middle target week "
                         f"{split_week}: nothing to judge the corrections on")
    return split_week, judged


def candidate_name(p_correction: PCorrection, m_correction: MCorrection) -> str:
    """The name of a pair of corrections in the tables."""
    return f"p {p_correction.kind}, m {m_correction.kind}"


def numbers_learned(p_correction: PCorrection, m_correction: MCorrection) -> int:
    """How many numbers a pair of corrections learns: its size, for preferring the simpler.
    Each Platt map learns a slope and an intercept; `scale` one factor; `per company` one per
    company; `by size` one per group."""
    in_p = 2 * len(p_correction.maps)
    in_m = len(m_correction.per_company) + len(m_correction.factors)
    if m_correction.kind == SCALE:
        in_m = 1
    return in_p + in_m


def candidate_forecasts(rolled: Forecast, p_corrections: list[PCorrection],
                        m_corrections: list[MCorrection]) -> pd.DataFrame:
    """The forecasts in `rolled` with every pair of corrections applied, in the shape of
    `evaluate.stack_forecasts` (`STACK_COLUMNS`), each pair a method named by `candidate_name`."""
    parts = []
    for p_correction in p_corrections:
        for m_correction in m_corrections:
            totals = corrected_forecast(rolled, p_correction, m_correction).totals
            part = totals.rename(columns={"total": "forecast"})
            part["method"] = candidate_name(p_correction, m_correction)
            parts.append(part[STACK_COLUMNS])
    return pd.concat(parts, ignore_index=True)


def candidate_scores(forecasts: pd.DataFrame, actual: pd.DataFrame,
                     p_corrections: list[PCorrection], m_corrections: list[MCorrection],
                     horizons: int) -> pd.DataFrame:
    """One row per pair of corrections, `CANDIDATE_COLUMNS`: `forecasts` (`candidate_forecasts`)
    scored by `evaluate.evaluation_scores` against `actual`. The pairs come in the order of the
    lists, so that those of the simpler p come first.

    Raises ValueError if there is no consolidated cash to score on, and as
    `evaluation_scores` does."""
    scores = evaluation_scores(forecasts, actual, horizons)
    pooled = scores[(scores["company_code"] == CONSOLIDATED) & (scores["horizon"] == POOLED)]
    if pooled.empty:
        raise ValueError("there is no consolidated cash in the judged weeks to score the "
                         "corrections on")
    lines = []
    for p_correction in p_corrections:
        for m_correction in m_corrections:
            name = candidate_name(p_correction, m_correction)
            mine = scores[scores["method"] == name]
            consolidated = mine[mine["company_code"] == CONSOLIDATED].set_index("horizon")
            companies = mine[(mine["company_code"] != CONSOLIDATED)
                             & (mine["horizon"] == POOLED)]
            lines.append({"candidate": name,
                          "p": p_correction.kind,
                          "m": m_correction.kind,
                          "numbers_learned": numbers_learned(p_correction, m_correction),
                          "wape": consolidated.loc[POOLED, "wape"],
                          "total_wape": consolidated["wape"].get(HORIZON_TOTAL, float("nan")),
                          "bias": consolidated.loc[POOLED, "bias"],
                          "company_wape": companies["wape"].mean()})
    return pd.DataFrame(lines, columns=CANDIDATE_COLUMNS)


def simplest_good_enough(candidates: pd.DataFrame, min_gain: float) -> pd.Series:
    """The row of `candidates` to choose: of those whose WAPE is at most `min_gain` above the
    lowest, the one that learns the fewest numbers; on a tie, the first (the simpler p).

    So a pair that learns more numbers wins only if every simpler pair is more than `min_gain`
    above the lowest WAPE. Raises ValueError if a WAPE is NaN (it could not be compared).
    """
    if candidates["wape"].isna().any():
        missing = list(candidates.loc[candidates["wape"].isna(), "candidate"])
        raise ValueError(f"no WAPE for {missing}: is there no consolidated cash to judge on?")
    lowest = candidates["wape"].min()
    good = candidates[candidates["wape"] <= lowest + min_gain]
    # "stable" keeps the order of equal sizes, so a tie goes to the first.
    by_size = good.sort_values("numbers_learned", kind="stable")
    return by_size.iloc[0]


def choose_calibration(rolled: Forecast, weekly: pd.DataFrame, weeks: pd.DataFrame,
                       levels: pd.Series, actual: pd.DataFrame, settings) -> CalibrationChoice:
    """Which corrections of p and m to use, chosen on data they did not learn from.

    `rolled` is `evaluate.rolling_forecasts` at the validation origins and `actual` its
    `evaluate.actual_totals`. Every pair of corrections is learned on the target weeks up to
    `calibration_split`'s week, and scored at its later origins; `simplest_good_enough` chooses
    with `[calibration] min_gain`. The bootstrap of the winner against no correction is only
    shown: it does not decide. The winners are then learned again on all the validation rows,
    whose last target week comes before the first test target week.

    Raises ValueError if no row is left to learn from, and as the functions it calls do.
    """
    horizons = settings.receivables.horizons
    shrink_rows = settings.calibration.company_shrink_rows
    groups = settings.diagnostic.calibration_groups

    # 1. Divide the validation weeks.
    origins = sorted(set(rolled.totals["origin_week"]))
    split_week, judged_origins = calibration_split(origins, horizons)
    rows = calibration_rows(rolled.customers, weekly, weeks, levels)
    learned_rows = rows[rows["target_week"] <= split_week]
    if learned_rows.empty:
        raise ValueError(f"there are no rows with data up to target week {split_week} to learn "
                         f"the corrections from")

    # 2. Learn every correction on the first part.
    p_corrections = [fit_p_correction(learned_rows, kind) for kind in P_KINDS]
    m_corrections = [fit_m_correction(learned_rows, kind, shrink_rows, groups)
                     for kind in M_KINDS]

    # 3. Score every pair at the later origins.
    judged_totals = rolled.totals[rolled.totals["origin_week"].isin(judged_origins)]
    judged_customers = rolled.customers[rolled.customers["origin_week"].isin(judged_origins)]
    judged_forecast = Forecast(totals=judged_totals, customers=judged_customers)
    judged_actual = actual[actual["origin_week"].isin(judged_origins)]
    forecasts = candidate_forecasts(judged_forecast, p_corrections, m_corrections)
    candidates = candidate_scores(forecasts, judged_actual, p_corrections, m_corrections, horizons)

    # 4. Choose, and show how sure the choice is against no correction.
    winner = simplest_good_enough(candidates, settings.calibration.min_gain)
    # Only when a correction won: with nothing to compare, the bootstrap's checks must not stop
    # the choice.
    baseline = candidate_name(p_corrections[0], m_corrections[0])
    bootstrap = pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    if winner["candidate"] != baseline:
        bootstrap = paired_wape_bootstrap(forecasts, judged_actual,
                                          [(winner["candidate"], baseline)], horizons,
                                          settings.evaluation.bootstrap_samples,
                                          settings.evaluation.bootstrap_seed)
        # Only the pooled rows: the 5-week totals resample blocks of origins, and with as few
        # origins as a block holds, every resample is the same and the interval says nothing.
        bootstrap = bootstrap[bootstrap["horizon"] == POOLED]

    # 5. Learn the winners again on all the validation rows: these are the ones kept.
    p_correction = fit_p_correction(rows, winner["p"])
    m_correction = fit_m_correction(rows, winner["m"], shrink_rows, groups)
    return CalibrationChoice(p_correction=p_correction, m_correction=m_correction,
                             split_week=split_week, judged_origins=judged_origins,
                             candidates=candidates, winner=winner["candidate"],
                             bootstrap=bootstrap)
