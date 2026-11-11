"""Tests of `hurdle/reports/evaluation.py`: the printouts of steps 16 and 17 (one forecast and
the evaluation over the validation origins).

Each printout runs in a separate process (`run_snippet`) on hand-made tables, every number
worked out by hand, or on the real tables of the worked examples in test_evaluate.py and
test_calibrate.py; the printouts' definitions are checked on the module's syntax tree. How
main() calls them is tested in tests/hurdle/test_run_hurdle_pipeline.py.
"""
import ast

import pytest

from tests.hurdle.code_checks import HURDLE, run_snippet

EVALUATION = HURDLE / "reports" / "evaluation.py"

EVALUATION_PRINTOUTS = [
    "show_customer_forecasts", "show_forecast_totals", "show_actual_totals",
    "show_forecast_vs_actual", "show_rolling_forecasts", "show_deflated_reference",
    "show_rule_reference", "show_evaluation_scores", "show_paired_bootstrap",
    "show_component_bias", "show_payer_split", "show_balance_coherence",
    "show_calibration_reliability", "show_calibration_choice"]




def row_cells(lines, first):
    """The whitespace-split cells of the one line whose first cell is `first`."""
    rows = [line.split() for line in lines if line.split()[:1] == [first]]
    assert len(rows) == 1, (first, lines)
    return rows[0]


# show_forecast_vs_actual at origin 10, h = 1, 2, companies TR02 and TR03 and ALL. Forecast
# (nominal TRY) and actual:
#   ALL  h1 (week 11): 3,000,000 vs 2,000,000 -> 1.5;  h2 (week 12): 1,000,000 vs no row
#   TR02 h1:           2,000,000 vs 1,000,000 -> 2.0;  h2:            500,000 vs 1,000,000 -> 0.5
#   TR03 h1:           1,000,000 vs 1,000,000 -> 1.0;  h2:            500,000 vs no row
# In millions the consolidated table reads 3.0 / 2.0 / 1.5 and 1.0 / no data / no data.
SHOW_VS_ACTUAL = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports

totals = pd.DataFrame({
    "company_code": ["TR02", "TR02", "TR03", "TR03", "ALL", "ALL"],
    "origin_week": [10] * 6, "h": [1, 2, 1, 2, 1, 2], "target_week": [11, 12, 11, 12, 11, 12],
    "total": [2e6, 5e5, 1e6, 5e5, 3e6, 1e6]})
actual = pd.DataFrame({
    "company_code": ["TR02", "TR02", "TR03", "ALL"],
    "origin_week": [10] * 4, "h": [1, 2, 1, 1], "target_week": [11, 12, 11, 11],
    "actual": [1e6, 1e6, 1e6, 2e6], "actual_real": [0.0] * 4})
reports.show_forecast_vs_actual(totals, actual)
print("ok")
"""


@pytest.fixture(scope="module")
def vs_actual_lines():
    result = run_snippet(SHOW_VS_ACTUAL)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_show_forecast_vs_actual_header_lines(vs_actual_lines):
    assert vs_actual_lines[0] == (
        "        forecast vs actual at week 10, nominal (actual = gross receipts + refunds; "
        "ratio = forecast / actual; one origin: an illustration, not a score)")
    assert vs_actual_lines[1] == "        consolidated, million TRY:"
    assert vs_actual_lines.count("        ratio per company and h:") == 1


def test_show_forecast_vs_actual_consolidated_table(vs_actual_lines):
    split = vs_actual_lines.index("        ratio per company and h:")
    table = vs_actual_lines[2:split]
    assert table[0].split() == ["target_week", "forecast", "actual", "ratio"]
    assert table[1].split() == ["h"]
    one = row_cells(table, "1")
    assert one[1] == "11" and [float(v) for v in one[2:]] == [3.0, 2.0, 1.5]
    two = row_cells(table, "2")
    # A target week without an actual shows "no data", never 0.
    assert two[1] == "12" and float(two[2]) == 1.0
    assert two[3:] == ["no", "data", "no", "data"]
    assert len(table) == 4


def test_show_forecast_vs_actual_ratio_per_company(vs_actual_lines):
    split = vs_actual_lines.index("        ratio per company and h:")
    table = vs_actual_lines[split + 1:]
    assert table[0].split() == ["h", "1", "2"]
    assert table[1].split() == ["company_code"]
    all_row = row_cells(table, "ALL")
    assert float(all_row[1]) == 1.5 and all_row[2:] == ["no", "data"]
    assert [float(v) for v in row_cells(table, "TR02")[1:]] == [2.0, 0.5]
    tr03 = row_cells(table, "TR03")
    assert float(tr03[1]) == 1.0 and tr03[2:] == ["no", "data"]
    assert len(table) == 5


# show_actual_totals: one origin (10), h = 1, 2. The consolidated actual is 150 + 70 = 220 TRY;
# the weekly amounts of the target weeks 11 and 12 are (100 - 10) + (60 + 0) + (80 - 10) = 220;
# week 13 (999) is not a target week and stays out. 4 actual rows, companies TR02 and ALL;
# 2 of the calendar's 4 weeks are missing.
SHOW_ACTUAL = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports

actual = pd.DataFrame({
    "company_code": ["TR02", "TR02", "ALL", "ALL"], "origin_week": [10] * 4,
    "h": [1, 2, 1, 2], "target_week": [11, 12, 11, 12],
    "actual": [80.0, 70.0, 150.0, 70.0], "actual_real": [0.0] * 4})
weekly = pd.DataFrame({
    "company_code": ["TR02", "TR03", "TR02", "TR02"], "week": [11, 11, 12, 13],
    "gross_receipts": [100.0, 60.0, 80.0, 999.0], "refunds": [-10.0, 0.0, -10.0, 0.0]})
weeks = pd.DataFrame({"week": [10, 11, 12, 13], "is_missing": [False, True, False, True]})
reports.show_actual_totals(actual, weekly, weeks)
print("ok")
"""


def test_show_actual_totals_by_hand():
    result = run_snippet(SHOW_ACTUAL)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "        actual cash (U14.1): 4 rows (2 companies with ALL); consolidated 220.00 TRY vs "
        "the weekly amounts of its target weeks 220.00 TRY; 2 missing weeks in the calendar "
        "(no rows)",
        "ok"]


# show_rolling_forecasts at origins 10 and 11, h = 1, 2. Consolidated forecast vs actual:
#   origin 10: h1 (week 11) 2,000,000 / 1,000,000 = 2.0; h2 (week 12) 1,500,000 / 2,000,000 = 0.75
#   origin 11: h1 (week 12) 3,000,000 / 2,000,000 = 1.5; h2 (week 13) 1,000,000 / no row
# Over the 3 with data: forecast 6.5 M vs actual 5.0 M, ratio 1.300. TR02's rows (8 total rows
# in all, TR02's forecasts 9e9 each) must not enter the pivot or the sums; 7 customer rows.
SHOW_ROLLING = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.forecast import Forecast

keys = dict(origin_week=[10, 10, 11, 11], h=[1, 2, 1, 2], target_week=[11, 12, 12, 13])
totals = pd.concat([
    pd.DataFrame({"company_code": ["TR02"] * 4, **keys, "total": [9e9, 9e9, 9e9, 9e9]}),
    pd.DataFrame({"company_code": ["ALL"] * 4, **keys,
                  "total": [2e6, 1.5e6, 3e6, 1e6]})], ignore_index=True)
customers = pd.DataFrame({"key": [f"TR02-{i}" for i in range(7)]})
WITH_ACTUAL = __WITH_ACTUAL__
if WITH_ACTUAL:
    three = dict(origin_week=[10, 10, 11], h=[1, 2, 1], target_week=[11, 12, 12])
    actual = pd.concat([
        pd.DataFrame({"company_code": ["TR02"] * 3, **three, "actual": [1.0, 1.0, 1.0]}),
        pd.DataFrame({"company_code": ["ALL"] * 3, **three, "actual": [1e6, 2e6, 2e6]})],
        ignore_index=True)
else:
    actual = pd.DataFrame({"company_code": pd.Series([], dtype=object),
                           **{k: pd.Series([], dtype="int64") for k in keys},
                           "actual": pd.Series([], dtype="float64")})
actual["actual_real"] = 0.0
reports.show_rolling_forecasts(Forecast(totals=totals, customers=customers), actual)
print("ok")
"""


def run_show_rolling(with_actual):
    result = run_snippet(SHOW_ROLLING.replace("__WITH_ACTUAL__", str(with_actual)))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


@pytest.fixture(scope="module")
def rolling_lines():
    return run_show_rolling(True)


def test_show_rolling_forecasts_first_line_counts_the_rows(rolling_lines):
    assert rolling_lines[0] == ("        7 customer rows, 8 total rows; consolidated forecast / "
                                "actual per origin and h (nominal):")


def test_show_rolling_forecasts_ratio_pivot_per_origin_and_h(rolling_lines):
    table = rolling_lines[1:-1]
    assert table[0].split() == ["h", "1", "2"]
    assert table[1].split() == ["origin_week"]
    assert [float(v) for v in row_cells(table, "10")[1:]] == [2.0, 0.75]
    eleven = row_cells(table, "11")
    # A target week without an actual shows "no data", never 0 or a ratio.
    assert float(eleven[1]) == 1.5 and eleven[2:] == ["no", "data"]
    assert len(table) == 4   # TR02's ratios are not in the pivot


def test_show_rolling_forecasts_sums_line(rolling_lines):
    assert rolling_lines[-1] == ("        over all 3 consolidated (origin, h) with data: forecast "
                                 "6.5 M vs actual 5.0 M TRY (ratio 1.300)")


def test_show_rolling_forecasts_without_any_actual_shows_no_data_everywhere():
    lines = run_show_rolling(False)
    table = lines[1:-1]
    assert row_cells(table, "10")[1:] == ["no", "data", "no", "data"]
    assert row_cells(table, "11")[1:] == ["no", "data", "no", "data"]
    assert lines[-1].startswith("        over all 0 consolidated (origin, h) with data: "
                                "forecast 0.0 M vs actual 0.0 M TRY (ratio ")


def test_show_deflated_reference_is_a_module_level_function_with_three_parameters():
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    found = [node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == "show_deflated_reference"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["reference", "rolled", "actual"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None


# show_deflated_reference at origins 10 and 11, h = 1, 2, companies TR02 and ALL.
# Consolidated actual: (10, 1) wk 11 1,000,000; (10, 2) wk 12 2,000,000; (11, 1) wk 12
# 2,000,000; (11, 2) wk 13 has no row. Over the 3 with data, actual 5.0 M.
#   hurdle model (ALL): 2.0 + 1.5 + 3.0           = 6.5 M, ratio 1.300 (wk 13's 1.0 M left out)
#   deflated (ALL):     1.0 + 2.0 + 2.0           = 5.0 M, ratio 1.000 (wk 13's 7.0 M left out)
#   deflated_nolevel:   0.5 + 1.0 + 1.0           = 2.5 M, ratio 0.500 (wk 13's 9e9 left out)
#   deflated_trend:     1.5 + 2.5 + 3.0           = 7.0 M, ratio 1.400 (wk 13's 8e8 left out)
# TR02's rows (9e9 everywhere) must enter no sum. The reference has 2 companies x 4 (origin, h)
# x 3 variants = 24 rows.
# Fallbacks count over the SAME rows as the sums: the 3 consolidated (origin, h) with data,
# (10, 1), (10, 2), (11, 1). deflated: ALL [T, F, F] = 1 of 3 (ALL's (11, 2) fallback has no
# actual, and TR02's fallbacks are not consolidated: neither counts); deflated_nolevel: ALL
# [F, F, F] = 0 of 3 (TR02's one fallback does not count); deflated_trend: ALL [F, T, F] = 1 of
# 3 (its (11, 2) fallback and TR02's four do not count). Counting every row of a variant
# instead would give 3 of 8, 1 of 8 and 6 of 8.
SHOW_REFERENCE = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.forecast import Forecast

keys = dict(origin_week=[10, 10, 11, 11], h=[1, 2, 1, 2], target_week=[11, 12, 12, 13])
totals = pd.concat([
    pd.DataFrame({"company_code": ["TR02"] * 4, **keys, "total": [9e9] * 4}),
    pd.DataFrame({"company_code": ["ALL"] * 4, **keys, "total": [2e6, 1.5e6, 3e6, 1e6]})],
    ignore_index=True)
customers = pd.DataFrame({"key": ["TR02-0"]})
three = dict(origin_week=[10, 10, 11], h=[1, 2, 1], target_week=[11, 12, 12])
actual = pd.concat([
    pd.DataFrame({"company_code": ["TR02"] * 3, **three, "actual": [1.0, 1.0, 1.0]}),
    pd.DataFrame({"company_code": ["ALL"] * 3, **three, "actual": [1e6, 2e6, 2e6]})],
    ignore_index=True)
actual["actual_real"] = 0.0
values = {("TR02", "deflated"): ([9e9] * 4, [False, False, False, True]),
          ("TR02", "deflated_nolevel"): ([9e9] * 4, [False, False, True, False]),
          ("ALL", "deflated"): ([1e6, 2e6, 2e6, 7e6], [True, False, False, True]),
          ("ALL", "deflated_nolevel"): ([5e5, 1e6, 1e6, 9e9], [False] * 4),
          ("TR02", "deflated_trend"): ([9e9] * 4, [True] * 4),
          ("ALL", "deflated_trend"): ([1.5e6, 2.5e6, 3e6, 8e8], [False, True, False, True])}
rows = []
for company in ("TR02", "ALL"):
    for i in range(4):
        for reference in ("deflated", "deflated_nolevel", "deflated_trend"):
            forecast, fallback = values[(company, reference)]
            rows.append({"company_code": company, "origin_week": keys["origin_week"][i],
                         "h": keys["h"][i], "target_week": keys["target_week"][i],
                         "reference": reference, "real_forecast": forecast[i] / 2.0,
                         "price_level": 2.0, "forecast": forecast[i],
                         "fallback": fallback[i]})
reference = pd.DataFrame(rows)
reports.show_deflated_reference(reference, Forecast(totals=totals, customers=customers), actual)
print("ok")
"""


@pytest.fixture(scope="module")
def reference_lines():
    result = run_snippet(SHOW_REFERENCE)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_show_deflated_reference_prints_five_lines(reference_lines):
    # The header, the hurdle model, and one line per variant (three).
    assert len(reference_lines) == 5


def test_show_deflated_reference_header_counts_the_rows_and_the_consolidated_pairs(
        reference_lines):
    assert reference_lines[0] == (
        "        the deflated reference (U14.3): 24 rows; over the 3 consolidated (origin, h) "
        "with data, nominal, million TRY:")


def test_show_deflated_reference_hurdle_model_line(reference_lines):
    assert reference_lines[1] == ("          hurdle model" + " " * 13
                                  + "6.5 vs actual 5.0 (ratio 1.300)")


def test_show_deflated_reference_deflated_line_with_its_fallback_count(reference_lines):
    assert reference_lines[2] == ("          deflated" + " " * 17 + "5.0 vs actual 5.0 (ratio "
                                  "1.000), fell back to the recent level in 1 of 3")


def test_show_deflated_reference_nolevel_line_with_its_fallback_count(reference_lines):
    assert reference_lines[3] == ("          deflated_nolevel" + " " * 9 + "2.5 vs actual 5.0 "
                                  "(ratio 0.500), fell back to the recent level in 0 of 3")


def test_show_deflated_reference_trend_line_with_its_fallback_count(reference_lines):
    # deflated_trend is 14 characters, padded to 17, then the sum right-aligned in 10.
    assert reference_lines[4] == ("          deflated_trend" + " " * 11 + "7.0 vs actual 5.0 "
                                  "(ratio 1.400), fell back to the recent level in 1 of 3")


def test_show_deflated_reference_lines_follow_the_references_order(reference_lines):
    names = [line.split()[0] for line in reference_lines[1:]]
    assert names == ["hurdle", "deflated", "deflated_nolevel", "deflated_trend"]


def test_show_deflated_reference_the_hurdle_line_has_no_fallback_count(reference_lines):
    assert "fell back" not in reference_lines[1]


def test_show_rule_reference_is_a_module_level_function_with_two_parameters():
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    found = [node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == "show_rule_reference"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["rule", "actual"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == []


# show_rule_reference at origins 10 and 11, h = 1, 2, companies TR02 and ALL.
# Consolidated actual: (10, 1) wk 11 1,000,000; (10, 2) wk 12 2,000,000; (11, 1) wk 12
# 2,000,000; (11, 2) wk 13 has no row. Rule (ALL) forecast 1.5 + 1.0 + 3.0 = 5.5 M (wk 13's 9e9
# left out) vs actual 5.0 M: ratio 1.100.
#   per h: h1 forecast 1.5 + 3.0 = 4.5, actual 1.0 + 2.0 = 3.0, ratio 1.5, late share
#          (0.25 + 1.0) / (1.0 + 2.0) = 0.41667 -> 0.417;
#          h2 forecast 1.0, actual 2.0, ratio 0.5, late share 0 / 0.8 = 0.000.
# TR02 (9e9 everywhere) enters no sum. The invoices (both origins, summed): expected 2 rows
# 1.0 + 2.0 = 3.0 M; late 1 row 0.5 M; after horizon 1 row 0.3 M; doubtful 1 row 3.0 M. Delay
# sources of the open TRY (6.8 M): customer 1.0 + 0.5 + 0.3 = 1.8 -> 26.5%; company 2.0 ->
# 29.4%; none 3.0 -> 44.1%.
SHOW_RULE = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.evaluate import RuleReference

keys = dict(origin_week=[10, 10, 11, 11], h=[1, 2, 1, 2], target_week=[11, 12, 12, 13])
def block(company, forecast, invoice, late):
    return pd.DataFrame({"company_code": [company] * 4, **keys, "invoice_cash": invoice,
                         "late_cash": late, "fallback_delay_cash": [0.0] * 4,
                         "topup_cash": [f - i for f, i in zip(forecast, invoice)],
                         "price_level": [1.0] * 4, "forecast": forecast})
totals = pd.concat([
    block("TR02", [9e9] * 4, [9e9] * 4, [9e9] * 4),
    block("ALL", [1.5e6, 1.0e6, 3.0e6, 9e9], [1.0e6, 0.8e6, 2.0e6, 9e9],
          [0.25e6, 0.0, 1.0e6, 9e9])], ignore_index=True)
three = dict(origin_week=[10, 10, 11], h=[1, 2, 1], target_week=[11, 12, 12])
actual = pd.concat([
    pd.DataFrame({"company_code": ["TR02"] * 3, **three, "actual": [1.0, 1.0, 1.0]}),
    pd.DataFrame({"company_code": ["ALL"] * 3, **three, "actual": [1e6, 2e6, 2e6]})],
    ignore_index=True)
actual["actual_real"] = 0.0
invoices = pd.DataFrame({
    "origin_week": [10, 10, 10, 11, 11],
    "status": __STATUS__,
    "delay_from": __DELAY_FROM__,
    "outstanding_try": [1e6, 0.5e6, 0.3e6, 2e6, 3e6]})
rule = RuleReference(totals=totals, invoices=invoices)
before = (totals.copy(), invoices.copy(), actual.copy())
reports.show_rule_reference(rule, actual)
pd.testing.assert_frame_equal(totals, before[0])
pd.testing.assert_frame_equal(invoices, before[1])
pd.testing.assert_frame_equal(actual, before[2])
print("ok")
"""
RULE_STATUS = '["expected", "late", "after horizon", "expected", "doubtful"]'
RULE_DELAY_FROM = '["customer", "customer", "customer", "company", "none"]'


def run_show_rule(status=RULE_STATUS, delay_from=RULE_DELAY_FROM):
    code = SHOW_RULE.replace("__STATUS__", status).replace("__DELAY_FROM__", delay_from)
    result = run_snippet(code)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


@pytest.fixture(scope="module")
def rule_lines():
    return run_show_rule()


RULE_PER_H_TITLE = "          the rule per h, consolidated, million TRY:"
RULE_DECIDED_TITLE = ("          the open invoices at the origins, by what the rule decided "
                      "(summed over the origins):")


def test_show_rule_reference_prints_fourteen_lines(rule_lines):
    assert len(rule_lines) == 14, rule_lines


def test_show_rule_reference_first_line_forecast_vs_actual_and_ratio(rule_lines):
    assert rule_lines[0] == ("          financial rule" + " " * 11
                             + "5.5 vs actual 5.0 (ratio 1.100)")


def test_show_rule_reference_first_line_has_the_deflated_reference_layout(rule_lines):
    # The layout of show_deflated_reference's lines: the name in 17, the amount in 10.
    assert rule_lines[0] == (f"          {'financial rule':<17} {5.5:>10,.1f} vs actual 5.0 "
                             f"(ratio 1.100)")


def test_show_rule_reference_titles(rule_lines):
    assert rule_lines[1] == RULE_PER_H_TITLE
    assert rule_lines[6] == RULE_DECIDED_TITLE
    assert rule_lines.count(RULE_PER_H_TITLE) == 1
    assert rule_lines.count(RULE_DECIDED_TITLE) == 1


def test_show_rule_reference_per_h_table_by_hand(rule_lines):
    table = rule_lines[2:6]
    assert table[0].split() == ["forecast", "actual", "ratio", "late", "share"]
    assert table[1].split() == ["h"]
    assert [float(v) for v in row_cells(table, "1")[1:]] == [4.5, 3.0, 1.5, 0.417]
    assert [float(v) for v in row_cells(table, "2")[1:]] == [1.0, 2.0, 0.5, 0.0]


def test_show_rule_reference_leaves_out_the_target_week_without_actual(rule_lines):
    # (11, 2)'s 9e9, and TR02's, would show as 9,000 million or more if a sum kept them.
    assert not any("9000" in line or "9e" in line or "9,000" in line for line in rule_lines)


def test_show_rule_reference_decided_table_by_hand(rule_lines):
    table = rule_lines[7:13]
    assert table[0].split() == ["invoices", "million", "TRY"]
    assert table[1].split() == ["status"]
    rows = [line.split() for line in table[2:]]
    # Sorted by status (groupby), counts and million TRY summed over both origins.
    assert rows == [["after", "horizon", "1", "0.3"], ["doubtful", "1", "3.0"],
                    ["expected", "2", "3.0"], ["late", "1", "0.5"]]


def test_show_rule_reference_delay_source_shares_by_hand(rule_lines):
    assert rule_lines[13] == (
        "          the delay came from the customer for 26.5% of the open TRY, its company's "
        "median for 29.4%, none (0 days) for 44.1%")


def test_show_rule_reference_a_delay_source_with_no_invoice_shows_zero():
    lines = run_show_rule(delay_from='["customer"] * 5')
    assert lines[-1] == (
        "          the delay came from the customer for 100.0% of the open TRY, its company's "
        "median for 0.0%, none (0 days) for 0.0%")


def test_show_rule_reference_a_status_with_no_invoice_has_no_row():
    lines = run_show_rule(status='["expected"] * 5')
    decided = lines[lines.index(RULE_DECIDED_TITLE) + 1:-1]
    assert [line.split() for line in decided[2:]] == [["expected", "5", "6.8"]]
    assert len(lines) == 11


def test_show_rule_reference_does_not_change_its_inputs(rule_lines):
    # SHOW_RULE compares the three tables with copies taken before the call; "ok" means equal.
    assert rule_lines


# show_deflated_reference with rolled=None: the same data as SHOW_REFERENCE above; the
# "hurdle model" line is left out, the other four lines are the same as with rolled.

SHOW_REFERENCE_NO_MODEL = SHOW_REFERENCE.replace(
    "reports.show_deflated_reference(reference, Forecast(totals=totals, customers=customers), "
    "actual)", "reports.show_deflated_reference(reference, None, actual)")


def test_show_reference_no_model_script_premise():
    assert SHOW_REFERENCE_NO_MODEL != SHOW_REFERENCE
    assert "show_deflated_reference(reference, None, actual)" in SHOW_REFERENCE_NO_MODEL


@pytest.fixture(scope="module")
def no_model_reference_lines():
    result = run_snippet(SHOW_REFERENCE_NO_MODEL)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_show_deflated_reference_without_the_model_prints_four_lines(no_model_reference_lines):
    # The header and one line per variant (three).
    assert len(no_model_reference_lines) == 4
    assert not any("hurdle" in line for line in no_model_reference_lines)


def test_show_deflated_reference_without_the_model_header_is_the_same(no_model_reference_lines,
                                                                      reference_lines):
    assert no_model_reference_lines[0] == reference_lines[0]
    assert no_model_reference_lines[0] == (
        "        the deflated reference (U14.3): 24 rows; over the 3 consolidated (origin, h) "
        "with data, nominal, million TRY:")


def test_show_deflated_reference_without_the_model_reference_lines_by_hand(
        no_model_reference_lines):
    assert no_model_reference_lines[1] == ("          deflated" + " " * 17 + "5.0 vs actual 5.0 "
                                           "(ratio 1.000), fell back to the recent level in 1 "
                                           "of 3")
    assert no_model_reference_lines[2] == ("          deflated_nolevel" + " " * 9 + "2.5 vs "
                                           "actual 5.0 (ratio 0.500), fell back to the recent "
                                           "level in 0 of 3")
    assert no_model_reference_lines[3] == ("          deflated_trend" + " " * 11 + "7.0 vs "
                                           "actual 5.0 (ratio 1.400), fell back to the recent "
                                           "level in 1 of 3")


def test_show_deflated_reference_with_and_without_the_model_differ_only_by_its_line(
        no_model_reference_lines, reference_lines):
    assert [reference_lines[0]] + reference_lines[2:] == no_model_reference_lines
    assert reference_lines[1].startswith("          hurdle model")


# A consolidated (origin, h) with an actual but no model total: with the model, the inner
# merge drops it (2 pairs); without the model it stays (3 pairs).
SHOW_REFERENCE_MODEL_SHORT = SHOW_REFERENCE.replace(
    'pd.DataFrame({"company_code": ["ALL"] * 4, **keys, "total": [2e6, 1.5e6, 3e6, 1e6]})',
    'pd.DataFrame({"company_code": ["ALL"] * 4, **keys, '
    '"total": [2e6, 1.5e6, 3e6, 1e6]}).iloc[1:]')


def test_show_deflated_reference_with_the_model_counts_only_pairs_the_model_has():
    assert SHOW_REFERENCE_MODEL_SHORT != SHOW_REFERENCE
    result = run_snippet(SHOW_REFERENCE_MODEL_SHORT)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()[:-1]
    assert "over the 2 consolidated (origin, h)" in lines[0]
    # Without (10, 1): hurdle 1.5 + 3.0 = 4.5 vs actual 2.0 + 2.0 = 4.0 -> 1.125.
    assert lines[1] == "          hurdle model" + " " * 13 + "4.5 vs actual 4.0 (ratio 1.125)"


def test_show_evaluation_scores_is_a_module_level_function_with_two_parameters():
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    found = [node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == "show_evaluation_scores"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["scores", "horizons"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == []


def test_show_evaluation_scores_comes_right_after_show_rule_reference_in_the_file():
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    names = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
    assert names.index("show_evaluation_scores") == names.index("show_rule_reference") + 1


# show_evaluation_scores on a hand-made scores table, horizons 2, companies TR02, TR03 and ALL.
# Only the columns it reads carry numbers (method, company_code, horizon, rows, wape, bias,
# mape); the others are 0.0. Consolidated (ALL):
#   hurdle   pooled WAPE 0.20, bias -0.10, MAPE 0.25; h1 0.10, h2 0.30; total 0.05
#   deflated pooled WAPE 0.15, bias  0.05, MAPE 0.20; h1 0.1236 (-> 0.124), h2 0.18; total 0.08
#   rule     pooled WAPE 0.40, bias -0.35, MAPE 0.50; h1 0.30, no h2 row, no total row (NaN)
# Per company, pooled WAPE: hurdle 0.31 / 0.42 / 0.20; deflated 0.25 / 0.35 / 0.15;
# rule 0.60 / 0.70 / 0.40. hurdle's TR02 h1 row (WAPE 9.99) must not reach the per-company
# table, which holds the pooled rows only.
SHOW_EVAL_SCORES = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.evaluate import SCORE_COLUMNS

nan = float("nan")
# (method, company_code, horizon, rows, wape, bias, mape)
given = [
    ("hurdle", "TR02", "1", 3, 9.99, 0.0, nan),
    ("hurdle", "TR02", "pooled", 6, 0.31, -0.2, nan),
    ("hurdle", "TR03", "pooled", 6, 0.42, 0.1, nan),
    ("hurdle", "ALL", "1", 3, 0.1, -0.15, 0.12),
    ("hurdle", "ALL", "2", 3, 0.3, -0.05, 0.33),
    ("hurdle", "ALL", "pooled", 6, 0.2, -0.1, 0.25),
    ("hurdle", "ALL", "total", 2, 0.05, -0.1, 0.06),
    ("deflated", "TR02", "pooled", 6, 0.25, 0.0, nan),
    ("deflated", "TR03", "pooled", 6, 0.35, 0.02, nan),
    ("deflated", "ALL", "1", 3, 0.1236, 0.04, 0.13),
    ("deflated", "ALL", "2", 3, 0.18, 0.06, 0.2),
    ("deflated", "ALL", "pooled", 6, 0.15, 0.05, 0.2),
    ("deflated", "ALL", "total", 2, 0.08, 0.05, 0.09),
    ("rule", "TR02", "pooled", 6, 0.6, -0.4, nan),
    ("rule", "TR03", "pooled", 6, 0.7, -0.5, nan),
    ("rule", "ALL", "1", 3, 0.3, -0.3, 0.4),
    ("rule", "ALL", "pooled", 6, 0.4, -0.35, 0.5),
]
read = ["method", "company_code", "horizon", "rows", "wape", "bias", "mape"]
scores = pd.DataFrame([dict(zip(read, row)) for row in given])
for column in SCORE_COLUMNS:
    if column not in scores:
        scores[column] = 0.0
scores = scores[SCORE_COLUMNS]
if __WITHOUT_MODEL__:
    scores = scores[scores["method"] != "hurdle"].reset_index(drop=True)
before = scores.copy()
reports.show_evaluation_scores(scores, __HORIZONS__)
pd.testing.assert_frame_equal(scores, before)
print("ok")
"""


def run_show_eval_scores(without_model=False, horizons=2):
    code = (SHOW_EVAL_SCORES.replace("__WITHOUT_MODEL__", str(without_model))
            .replace("__HORIZONS__", str(horizons)))
    result = run_snippet(code)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


@pytest.fixture(scope="module")
def eval_lines():
    return run_show_eval_scores()


@pytest.fixture(scope="module")
def no_model_eval_lines():
    return run_show_eval_scores(without_model=True)


EVAL_SCORES_HEADER = ("        scores (U14.5), consolidated, over 6 (origin, h) with data "
                 "(h1 ... h2: WAPE per h):")
EVAL_PER_COMPANY_TITLE = "        pooled WAPE per company:"


def test_show_evaluation_scores_exact_printout_with_the_hurdle(eval_lines):
    assert eval_lines == [
        EVAL_SCORES_HEADER,
        "          WAPE  bias  MAPE     h1    h2  2-week total WAPE",
        "method                                                    ",
        "hurdle    0.20 -0.10  0.25  0.100  0.30               0.05",
        "deflated  0.15  0.05  0.20  0.124  0.18               0.08",
        "rule      0.40 -0.35  0.50  0.300   NaN                NaN",
        EVAL_PER_COMPANY_TITLE,
        "company_code  TR02  TR03   ALL",
        "method                        ",
        "hurdle        0.31  0.42  0.20",
        "deflated      0.25  0.35  0.15",
        "rule          0.60  0.70  0.40",
    ]


def test_show_evaluation_scores_exact_printout_without_the_hurdle(no_model_eval_lines):
    assert no_model_eval_lines == [
        EVAL_SCORES_HEADER,
        "          WAPE  bias  MAPE     h1    h2  2-week total WAPE",
        "method                                                    ",
        "deflated  0.15  0.05   0.2  0.124  0.18               0.08",
        "rule      0.40 -0.35   0.5  0.300   NaN                NaN",
        EVAL_PER_COMPANY_TITLE,
        "company_code  TR02  TR03   ALL",
        "method                        ",
        "deflated      0.25  0.35  0.15",
        "rule          0.60  0.70  0.40",
    ]


def test_show_evaluation_scores_without_the_hurdle_has_no_hurdle_line(no_model_eval_lines):
    assert not any(line.startswith("hurdle") for line in no_model_eval_lines)
    # One consolidated line and one per-company line fewer.
    assert len(no_model_eval_lines) == 10


def test_show_evaluation_scores_the_methods_in_the_order_they_come(eval_lines):
    consolidated = [line.split()[0] for line in eval_lines[3:6]]
    per_company = [line.split()[0] for line in eval_lines[9:12]]
    assert consolidated == per_company == ["hurdle", "deflated", "rule"]


def test_show_evaluation_scores_a_missing_h_or_total_shows_nan_not_zero(eval_lines):
    # rule has no "2" row and no "total" row.
    assert row_cells(eval_lines[:6], "rule")[-2:] == ["NaN", "NaN"]


def test_show_evaluation_scores_rounds_to_three_decimals(eval_lines):
    # deflated's h1 WAPE 0.1236 shows as 0.124.
    assert row_cells(eval_lines[:6], "deflated")[4] == "0.124"


def test_show_evaluation_scores_per_company_uses_the_pooled_rows_only(eval_lines):
    # hurdle's TR02 h1 WAPE 9.99 is not the pooled 0.31.
    assert not any("9.99" in line for line in eval_lines)
    assert row_cells(eval_lines[6:], "hurdle")[1] == "0.31"


def test_show_evaluation_scores_header_follows_the_horizons_argument():
    # Horizons 3 on the same table: an h3 column (NaN, no "3" rows) and a 3-week total.
    lines = run_show_eval_scores(horizons=3)
    assert lines[0] == ("        scores (U14.5), consolidated, over 6 (origin, h) with data "
                        "(h1 ... h3: WAPE per h):")
    assert lines[1].split() == ["WAPE", "bias", "MAPE", "h1", "h2", "h3", "3-week", "total",
                                "WAPE"]
    assert row_cells(lines[:6], "hurdle")[6] == "NaN"


def test_show_paired_bootstrap_is_a_module_level_function_with_one_parameter():
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    found = [node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == "show_paired_bootstrap"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["bootstrap"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None


def test_show_paired_bootstrap_sits_between_show_evaluation_scores_and_show_component_bias():
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    names = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
    shown = names.index("show_paired_bootstrap")
    assert names[shown - 1] == "show_evaluation_scores"
    assert names[shown + 1] == "show_component_bias"


# show_paired_bootstrap on a hand-made bootstrap table (BOOTSTRAP_COLUMNS), 2000 samples:
#   hurdle vs deflated  pooled  WAPE 0.16012 -> 0.16, against 0.1534 -> 0.153,
#                               difference +0.00672 -> "+0.007", interval [-0.0123, +0.0271]
#                               -> "[-0.012, +0.027]", better in 0.304 -> "30%"
#   hurdle vs deflated  total   0.08 / 0.1, -0.02 -> "-0.020", [-0.05, +0.0112] ->
#                               "[-0.050, +0.011]", 0.9 -> "90%"
#   hurdle vs rule      pooled  0.16012 / 0.4, -0.23988 -> "-0.240", [-0.3, -0.18] ->
#                               "[-0.300, -0.180]", 1.0 -> "100%"
#   hurdle vs twin      pooled  0.2 / 0.2, 0.0 -> "+0.000", [0, 0] -> "[+0.000, +0.000]",
#                               0.0 -> "0%"
SHOW_BOOTSTRAP = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.evaluate import BOOTSTRAP_COLUMNS

given = [
    ("hurdle", "deflated", "pooled", 7, 0.16012, 0.1534, 0.00672, -0.0123, 0.0271, 0.304, 2000),
    ("hurdle", "deflated", "total", 11, 0.08, 0.1, -0.02, -0.05, 0.0112, 0.9, 2000),
    ("hurdle", "rule", "pooled", 7, 0.16012, 0.4, -0.23988, -0.3, -0.18, 1.0, 2000),
    ("hurdle", "twin", "pooled", 7, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 2000),
]
bootstrap = pd.DataFrame(given, columns=BOOTSTRAP_COLUMNS)
before = bootstrap.copy()
reports.show_paired_bootstrap(bootstrap)
pd.testing.assert_frame_equal(bootstrap, before)
print("ok")
"""

BOOT_HEADER = ("        paired bootstrap (U14.5c), consolidated, 2000 resamples; difference = "
               "WAPE of the first - WAPE of the second (below 0: the first is better):")


@pytest.fixture(scope="module")
def bootstrap_lines():
    result = run_snippet(SHOW_BOOTSTRAP)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_show_paired_bootstrap_exact_printout(bootstrap_lines):
    # Every line 91 characters wide (pandas pads the index title line to the full width).
    assert bootstrap_lines == [
        BOOT_HEADER,
        "                   measure   WAPE WAPE against difference      90% interval "
        "first better in",
        "pair" + " " * 87,
        "hurdle vs deflated  pooled  0.160        0.153     +0.007  [-0.012, +0.027]    "
        "         30%",
        "hurdle vs deflated   total  0.080        0.100     -0.020  [-0.050, +0.011]    "
        "         90%",
        "hurdle vs rule      pooled  0.160        0.400     -0.240  [-0.300, -0.180]    "
        "        100%",
        "hurdle vs twin      pooled  0.200        0.200     +0.000  [+0.000, +0.000]    "
        "          0%",
    ]


def test_show_paired_bootstrap_both_wapes_have_three_decimals(bootstrap_lines):
    # 0.16012 -> "0.160" (not "0.16"), 0.08 -> "0.080"; 0.1534 -> "0.153".
    rows = [line.split() for line in bootstrap_lines[3:]]
    # From the end: share, interval (two cells), difference, WAPE against, WAPE.
    wapes = [row[-6] for row in rows]
    against = [row[-5] for row in rows]
    assert wapes == ["0.160", "0.080", "0.160", "0.200"]
    assert against == ["0.153", "0.100", "0.400", "0.200"]


def test_show_paired_bootstrap_header_names_the_sample_count(bootstrap_lines):
    assert bootstrap_lines[0] == BOOT_HEADER
    assert "2000 resamples" in bootstrap_lines[0]


def test_show_paired_bootstrap_one_line_per_row_in_order(bootstrap_lines):
    # Header, the column titles, the index title, then the four rows as they came.
    rows = bootstrap_lines[3:]
    assert len(rows) == 4
    assert [row.split("  ")[0].strip() for row in rows] == [
        "hurdle vs deflated", "hurdle vs deflated", "hurdle vs rule", "hurdle vs twin"]


def test_show_paired_bootstrap_signs_and_percentages(bootstrap_lines):
    # A difference is always signed, the interval's ends too; the share is a whole percentage.
    joined = "\n".join(bootstrap_lines[3:])
    for text in ("+0.007", "-0.020", "-0.240", "+0.000", "[-0.012, +0.027]",
                 "[-0.300, -0.180]", "30%", "90%", "100%", " 0%"):
        assert text in joined, text


@pytest.mark.parametrize("name, parameters", [("show_component_bias", ["bias", "horizons"]),
                                              ("show_payer_split", ["split", "horizons"])])
def test_the_bias_printouts_are_module_level_functions(name, parameters):
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    found = [node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == parameters
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None


def test_the_printouts_are_the_modules_functions_in_step_order():
    # Steps 16 and 17 in the order main() calls them; nothing else is defined here (the
    # step 18 printouts are in reports/grid.py, which imports three of these).
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    names = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
    assert names == EVALUATION_PRINTOUTS


# show_component_bias on a hand-made table, horizons 2. Amounts in TRY, shown in millions:
#   hurdle, pooled: known 4.0 / 4.0, gap "+0.0", ratio 1.000
#                   new 1.5 / 0.0, gap "+1.5", ratio NaN -> "nan"
#                   refunds -0.25 -> "-0.2", -0.5, gap 0.25 -> "+0.2", ratio 0.500
#                   price level 1234.5 -> "1,234.5", "1,000.0", gap "+234.5", 1.2345 -> "1.234"
#   rule, pooled:   known 0.7 / 4.0, gap "-3.3", ratio 0.175
#   per h:          hurdle 0.5 and 1.5; the rule has no h = 2 row, so "nan".
# The TR02 rows hold 9000 million and must not be shown: only "ALL" is.
SHOW_BIAS = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.evaluate import COMPONENT_COLUMNS

no_ratio = float("nan")
given = [
    ("hurdle", "ALL", "1", "known customers", 1000000.0, 2000000.0, -1000000.0, 0.5),
    ("hurdle", "ALL", "2", "known customers", 3000000.0, 2000000.0, 1000000.0, 1.5),
    ("hurdle", "ALL", "pooled", "known customers", 4000000.0, 4000000.0, 0.0, 1.0),
    ("hurdle", "ALL", "1", "new customers", 8000000.0, 0.0, 8000000.0, no_ratio),
    ("hurdle", "ALL", "pooled", "new customers", 1500000.0, 0.0, 1500000.0, no_ratio),
    ("hurdle", "ALL", "pooled", "refunds", -250000.0, -500000.0, 250000.0, 0.5),
    ("hurdle", "ALL", "pooled", "price level", 1234500000.0, 1000000000.0, 234500000.0, 1.2345),
    ("hurdle", "TR02", "1", "known customers", 9e9, 9e9, 0.0, 7.0),
    ("hurdle", "TR02", "pooled", "known customers", 9e9, 9e9, 0.0, 7.0),
    ("rule", "ALL", "1", "known customers", 500000.0, 2000000.0, -1500000.0, 0.25),
    ("rule", "ALL", "pooled", "known customers", 700000.0, 4000000.0, -3300000.0, 0.175),
]
bias = pd.DataFrame(given, columns=COMPONENT_COLUMNS)
before = bias.copy()
reports.show_component_bias(bias, 2)
pd.testing.assert_frame_equal(bias, before)
print("ok")
"""

BIAS_PRINTOUT = [
    "        bias per part (U14.5d), consolidated, all (origin, h), nominal, million TRY (the "
    "price level row is part of the total's gap, not added to it):",
    "                   part forecast   actual     gap  ratio",
    "method                                                  ",
    "hurdle  known customers      4.0      4.0    +0.0  1.000",
    "hurdle    new customers      1.5      0.0    +1.5    nan",
    "hurdle          refunds     -0.2     -0.5    +0.2  0.500",
    "hurdle      price level  1,234.5  1,000.0  +234.5  1.234",
    "rule    known customers      0.7      4.0    -3.3  0.175",
    "        known customers, forecast / actual per h:",
    "          hurdle                h1 0.500  h2 1.500",
    "          rule (open invoices)  h1 0.250  h2 nan",
]


@pytest.fixture(scope="module")
def bias_lines():
    result = run_snippet(SHOW_BIAS)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_show_component_bias_exact_printout(bias_lines):
    assert bias_lines == BIAS_PRINTOUT


def test_show_component_bias_a_nan_ratio_is_printed_as_nan(bias_lines):
    # An actual of 0: the gap is there, the ratio says so.
    new_customers = [line for line in bias_lines if "new customers" in line]
    assert len(new_customers) == 1
    assert new_customers[0].split()[-2:] == ["+1.5", "nan"]


def test_show_component_bias_shows_only_the_consolidated_rows(bias_lines):
    # The TR02 rows hold 9000 million and a ratio of 7.
    joined = "\n".join(bias_lines)
    assert "9,000" not in joined
    assert "7.000" not in joined


def test_show_component_bias_a_horizon_without_a_row_is_nan(bias_lines):
    assert bias_lines[-1].endswith("h1 0.250  h2 nan")


def test_show_component_bias_names_the_rule_by_its_open_invoices(bias_lines):
    assert bias_lines[-1].startswith("          rule (open invoices)")
    assert bias_lines[-4].startswith("rule    known customers")


# show_payer_split on a hand-made table, horizons 2. Consolidated, h = 2 has nobody expected
# (NaN per payer and m ratio); TR04 had no payers at all (NaN ratios). The TR02 h = 1 row is
# per company but not pooled, so it is in neither table; ALL's per-h rows are not in the second.
SHOW_SPLIT = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.evaluate import PAYER_COLUMNS

no_ratio = float("nan")
given = [
    ("TR02", "1", 4, 2.0, 3, 7.0, 60.0, 86.0, 7.0, 7.0),
    ("TR02", "pooled", 6, 3.25, 5, 0.65, 55.3846, 74.0, 0.74844, 0.48649),
    ("TR04", "pooled", 3, 2.0, 0, no_ratio, 25.0, no_ratio, no_ratio, no_ratio),
    ("ALL", "1", 6, 3.0, 4, 0.75, 50.0, 70.0, 5 / 7, 150 / 280),
    ("ALL", "2", 3, 0.0, 3, 0.0, no_ratio, 43.33333, no_ratio, 0.0),
    ("ALL", "pooled", 9, 5.25, 7, 0.75, 43.80952, 58.57143, 0.74797, 0.56098),
]
split = pd.DataFrame(given, columns=PAYER_COLUMNS)
before = split.copy()
reports.show_payer_split(split, 2)
pd.testing.assert_frame_equal(split, before)
print("ok")
"""

SPLIT_PRINTOUT = [
    "        known customers, how many pay and how much each pays (real TRY; a ratio is "
    "forecast / actual), consolidated:",
    "         payers expected  payers actual  p ratio  per payer expected  per payer actual  "
    "m ratio  p x m ratio",
    "horizon" + " " * 101,
    "1                   3.00              4     0.75               50.00            70.000    "
    "0.714        0.536",
    "2                   0.00              3     0.00                 NaN            43.333      "
    "NaN        0.000",
    "pooled              5.25              7     0.75               43.81            58.571    "
    "0.748        0.561",
    "        the same per company, all (origin, h):",
    "         p ratio  m ratio  p x m ratio",
    "company                               ",
    "TR02        0.65    0.748        0.486",
    "TR04         NaN      NaN          NaN",
]


@pytest.fixture(scope="module")
def split_lines():
    result = run_snippet(SHOW_SPLIT)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_show_payer_split_exact_printout(split_lines):
    assert split_lines == SPLIT_PRINTOUT


def test_show_payer_split_nan_ratios_are_printed_as_nan(split_lines):
    assert split_lines[-1].split() == ["TR04", "NaN", "NaN", "NaN"]


def test_show_payer_split_per_company_rows_are_the_pooled_ones_without_all(split_lines):
    # 1. The second table has TR02 and TR04, once each; ALL is the first table.
    second = split_lines[split_lines.index("        the same per company, all (origin, h):") + 3:]
    assert [line.split()[0] for line in second] == ["TR02", "TR04"]
    # 2. The TR02 h = 1 row, with its ratio of 7, is nowhere.
    assert "7.0" not in "\n".join(split_lines)


# The real functions, end to end: their tables are what the printouts expect.
SHOW_REAL_TABLES = """
from collection_estimation.hurdle.reports import evaluation as reports
from tests.hurdle.test_evaluate import cb_run, ps_run

reports.show_component_bias(cb_run(), 2)
reports.show_component_bias(cb_run(rolled=None), 2)
reports.show_payer_split(ps_run(), 2)
print("ok")
"""


@pytest.fixture(scope="module")
def real_table_lines():
    result = run_snippet(SHOW_REAL_TABLES)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_the_printouts_take_the_real_tables_of_the_worked_examples(real_table_lines):
    # 1. Consolidated known customers per h, forecast / actual, from test_evaluate's worked
    #    example: 2320 / 3277 = 0.708 and 5000 / 5329 = 0.938; the rule 1870 / 3277 = 0.571 and
    #    2310 / 5329 = 0.433.
    assert "          hurdle                h1 0.708  h2 0.938" in real_table_lines
    assert "          rule (open invoices)  h1 0.571  h2 0.433" in real_table_lines
    # 2. The payer split's pooled rows: TR02 p 0.65; TR04 1.00 and 1.25; ALL's p 0.75.
    assert real_table_lines[-2].split() == ["TR02", "0.65", "0.748", "0.486"]
    assert real_table_lines[-1].split() == ["TR04", "1.00", "1.250", "1.250"]


def test_the_printout_of_the_rule_alone_has_only_the_rule(real_table_lines):
    # Without the model: the table's one row and the one line per h come from the rule.
    second_start = [i for i, line in enumerate(real_table_lines)
                    if line.startswith("        bias per part")][1]
    second_end = [i for i, line in enumerate(real_table_lines)
                  if line.startswith("        known customers, how many")][0]
    second = real_table_lines[second_start:second_end]
    assert not any(line.startswith("hurdle") for line in second)
    assert second[-1] == "          rule (open invoices)  h1 0.571  h2 0.433"


def test_show_balance_coherence_is_a_module_level_function_after_show_payer_split():
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    found = [node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == "show_balance_coherence"]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == ["coherence", "factor"]
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None
    names = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
    assert names.index("show_balance_coherence") == names.index("show_payer_split") + 1


# show_balance_coherence on a hand-made table, factor 1.5. TR04 has no balance (NaN shares).
# ALL's groups: 1,234 rows, 1,560,000 and 520,000 TRY; 25,678 rows, 44,000,000 and 25,000,000.
SHOW_COHERENCE = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.evaluate import COHERENCE_COLUMNS

no_ratio = float("nan")
given = [
    ("TR02", 6, 394.0, 420.0, 290.0, 394 / 290, 420 / 290, 4, 0.25, 0.5, 1, 7e6, 8e6, 2, 9e6, 1e6),
    ("TR04", 2, 430.0, 520.0, 0.0, no_ratio, no_ratio, 0, no_ratio, no_ratio, 0, 0.0, 0.0, 2,
     430.0, 520.0),
    ("ALL", 26912, 824.0, 940.0, 290.0, 824 / 290, 940 / 290, 1234, 0.4, 0.6, 1234, 1.56e6,
     5.2e5, 25678, 4.4e7, 2.5e7),
]
coherence = pd.DataFrame(given, columns=COHERENCE_COLUMNS)
before = coherence.copy()
reports.show_balance_coherence(coherence, FACTOR)
pd.testing.assert_frame_equal(coherence, before)
print("ok")
"""

COHERENCE_PRINTOUT = [
    "        balance check (U14.6), each customer's cash over the weeks against what it owes at "
    "the origin:",
    "         forecast / balance  actual / balance  forecast > 1.5x  actual > 1.5x",
    "company                                                                      ",
    "TR02                  1.359             1.448             0.25            0.5",
    "TR04                    NaN               NaN              NaN            NaN",
    "ALL                   2.841             3.241             0.40            0.6",
    "        above 1.5x the balance: 1,234 rows, forecast 1.6 M against actual 0.5 M",
    "        nothing open: 25,678 rows, forecast 44.0 M against actual 25.0 M",
]


def coherence_lines(factor):
    result = run_snippet(SHOW_COHERENCE.replace("FACTOR", factor))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_show_balance_coherence_exact_printout():
    assert coherence_lines("1.5") == COHERENCE_PRINTOUT


def test_show_balance_coherence_a_whole_factor_prints_without_a_point():
    lines = coherence_lines("2.0")
    assert lines[1].split("  ")[-1].strip() == "actual > 2x"
    assert lines[6].startswith("        above 2x the balance:")


# The real function, end to end: its table is what the printout expects. Consolidated, from
# test_evaluate's worked example: 824 / 490 = 1.682, 940 / 490 = 1.918; TR02 394 / 290 = 1.359.
SHOW_REAL_COHERENCE = """
from collection_estimation.hurdle.reports import evaluation as reports
from tests.hurdle.test_evaluate import bc_run

reports.show_balance_coherence(bc_run(), 1.5)
print("ok")
"""


def test_the_printout_takes_the_real_table_of_the_worked_example():
    result = run_snippet(SHOW_REAL_COHERENCE)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    assert lines[3].split() == ["TR02", "1.359", "1.448", "0.25", "0.5"]
    assert lines[4].split() == ["TR04", "2.150", "2.600", "1.00", "1.0"]
    assert lines[5].split() == ["ALL", "1.682", "1.918", "0.40", "0.6"]
    assert lines[6] == "        above 1.5x the balance: 2 rows, forecast 0.0 M against actual 0.0 M"
    assert lines[7] == "        nothing open: 3 rows, forecast 0.0 M against actual 0.0 M"


@pytest.mark.parametrize("name, parameters", [("show_calibration_reliability", ["reliability"]),
                                              ("show_calibration_choice", ["choice", "min_gain"])])
def test_the_calibration_printouts_are_module_level_functions(name, parameters):
    module = ast.parse(EVALUATION.read_text(encoding="utf-8"))
    found = [node for node in module.body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1
    assert [a.arg for a in found[0].args.args] == parameters
    assert found[0].args.defaults == [] and found[0].args.vararg is None
    assert found[0].args.kwonlyargs == [] and found[0].args.kwarg is None


# show_calibration_reliability on the real function's table: the small world of test_calibrate,
# whose numbers are worked out there. p: 0.25 and 0.65 at h 1, 0.25 and 0.7 at h 2.
SHOW_RELIABILITY = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.calibrate import calibration_reliability
from tests.hurdle.test_calibrate import reliability_case

reliability = calibration_reliability(*reliability_case(), groups=2)
before = reliability.p.copy(), reliability.m.copy()
reports.show_calibration_reliability(reliability)
pd.testing.assert_frame_equal(reliability.p, before[0])
pd.testing.assert_frame_equal(reliability.m, before[1])
print("ok")
"""

RELIABILITY_PRINTOUT = [
    "        stage 1 reliability, all rolling forecast rows in equal groups by p, all h "
    "(calibrated: mean_p near share_paid in every group):",
    " group  p_low  p_high  mean_p  share_paid  rows",
    "     1    0.1     0.4   0.250        0.25     4",
    "     2    0.5     0.9   0.675        0.75     4",
    "        the same per h, each cell mean p / share paid:",
    "horizon                1                2",
    "group                                    ",
    "1        0.2500 / 0.5000  0.2500 / 0.0000",
    "2        0.6500 / 0.5000  0.7000 / 1.0000",
    "        stage 2 reliability, paid forecast rows in equal groups by m (calibrated: ratio = "
    "mean actual / mean m near 1 in every group):",
    " group  m_low  m_high  mean_m  mean_actual  ratio  rows",
    "     1    5.0    20.0    12.0         20.0  1.600     2",
    "     2   90.0   150.0   120.0        200.0  1.667     2",
]


def test_show_calibration_reliability_exact_printout():
    result = run_snippet(SHOW_RELIABILITY)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == RELIABILITY_PRINTOUT + ["ok"]


# show_calibration_choice on hand-made choices. A table of two pairs is enough: the printout
# does not depend on how many there are.
SHOW_CHOICE = """
import pandas as pd
from collection_estimation.hurdle.reports import evaluation as reports
from collection_estimation.hurdle.calibrate import (CANDIDATE_COLUMNS, CalibrationChoice,
                                                    MCorrection, PCorrection)
from collection_estimation.hurdle.evaluate import BOOTSTRAP_COLUMNS

candidates = pd.DataFrame(
    [("p none, m none", "none", "none", 0, 0.17034, 0.11119, -0.05, 0.22221),
     ("p none, m scale", "none", "scale", 1, 0.16301, 0.10008, 0.01, 0.21004)],
    columns=CANDIDATE_COLUMNS)
bootstrap = pd.DataFrame(
    [("p none, m scale", "p none, m none", "pooled", 20, 0.163, 0.17, -0.007, -0.012, -0.002,
      0.98, 2000)], columns=BOOTSTRAP_COLUMNS)
empty = bootstrap.iloc[:0]
choices = {
    "none": (PCorrection("none"), MCorrection("none"), empty),
    "scale": (PCorrection("none"), MCorrection("scale", 1.0434), bootstrap),
    "company": (PCorrection("one map", {"all": (1.1, -0.2)}),
                MCorrection("per company", 1.05, per_company={"TR02": 1.1, "TR04": 0.9}),
                bootstrap),
    "size": (PCorrection("per h", {1: (1.0, 0.1), 2: (0.95, -0.05)}),
             MCorrection("by size", 1.05, log_m=(0.0, 6.907755278982137),
                         factors=(1.5, 0.75)), bootstrap),
}
p_correction, m_correction, shown_bootstrap = choices["WHICH"]
choice = CalibrationChoice(p_correction=p_correction, m_correction=m_correction, split_week=41,
                           judged_origins=[41, 42, 43, 44], candidates=candidates,
                           winner="WINNER", bootstrap=shown_bootstrap)
before = candidates.copy()
reports.show_calibration_choice(choice, 0.005)
pd.testing.assert_frame_equal(candidates, before)
print("ok")
"""

CHOICE_TABLE = [
    "        corrections learned on target weeks up to 41, judged at origins 41-44 (only wape "
    "decides; min_gain 0.005):",
    "   p     m  numbers_learned   wape  total_wape  bias  company_wape",
    "none  none                0 0.1703      0.1112 -0.05        0.2222",
    "none scale                1 0.1630      0.1001  0.01        0.2100",
]
CHOICE_BOOTSTRAP_LINE = (
    "        paired bootstrap (U14.5c), consolidated, 2000 resamples; difference = WAPE of the "
    "first - WAPE of the second (below 0: the first is better):")
FROZEN_LINE = ("        the chosen corrections, learned on all these rows and frozen for the "
               "test:")


def choice_lines(which, winner):
    code = SHOW_CHOICE.replace("WHICH", which).replace("WINNER", winner)
    result = run_snippet(code)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    return lines[:-1]


def test_show_calibration_choice_no_correction_exact_printout():
    assert choice_lines("none", "p none, m none") == CHOICE_TABLE + [
        "        chosen: p none, m none",
        "        no correction chosen, so nothing to compare it with",
        FROZEN_LINE,
        "          p, none: unchanged",
        "          m, none: common factor 1.000"]


def test_show_calibration_choice_scale_shows_the_bootstrap_and_the_factor():
    lines = choice_lines("scale", "p none, m scale")
    assert lines[:5] == CHOICE_TABLE + ["        chosen: p none, m scale"]
    assert lines[5] == CHOICE_BOOTSTRAP_LINE
    assert not any("no correction chosen" in line for line in lines)
    joined = "\n".join(lines[6:-3])
    for text in ("p none, m scale vs p none, m none", "pooled", "0.163", "0.170", "-0.007",
                 "[-0.012, -0.002]", "98%"):
        assert text in joined, text
    assert lines[-3:] == [FROZEN_LINE, "          p, none: unchanged",
                          "          m, scale: common factor 1.043"]


def test_show_calibration_choice_per_company_shows_each_company():
    lines = choice_lines("company", "p one map, m per company")
    assert lines[-4:] == [FROZEN_LINE,
                          "          p, one map: h all: slope 1.100, intercept -0.200",
                          "          m, per company: common factor 1.050",
                          "            TR02 1.100, TR04 0.900"]


def test_show_calibration_choice_by_size_shows_the_factor_at_each_m():
    # log m 0 is m = 1; 6.9077... is m = 1,000.
    lines = choice_lines("size", "p per h, m by size")
    assert lines[-4:] == [FROZEN_LINE,
                          "          p, per h: h 1: slope 1.000, intercept 0.100, "
                          "h 2: slope 0.950, intercept -0.050",
                          "          m, by size: common factor 1.050",
                          "            factor at m = 1: 1.500, 1,000: 0.750"]


# The real function, end to end: its choice is what the printout expects. By hand in
# test_calibrate: no correction 0.856; `scale` 0.714; frozen `scale` 30700 / 9200 = 3.337.
SHOW_REAL_CHOICE = """
from collection_estimation.hurdle.reports import evaluation as reports
from tests.hurdle.test_calibrate import choose, validation_case

reports.show_calibration_choice(choose(validation_case(), min_gain=0.06), 0.06)
print("ok")
"""


def test_the_choice_printout_takes_the_real_choice_of_the_worked_example():
    result = run_snippet(SHOW_REAL_CHOICE)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "ok"
    assert lines[0] == ("        corrections learned on target weeks up to 15, judged at origins "
                        "15-17 (only wape decides; min_gain 0.06):")
    # The table: a header and the twelve pairs; p none, m none and p none, m scale first.
    assert lines[2].split()[:4] == ["none", "none", "0", "0.8559"]
    assert lines[3].split()[:4] == ["none", "scale", "1", "0.7136"]
    assert lines[14] == "        chosen: p none, m scale"
    assert lines[15].startswith("        paired bootstrap (U14.5c), consolidated, 50 resamples")
    assert lines[-4:-1] == [FROZEN_LINE, "          p, none: unchanged",
                            "          m, scale: common factor 3.337"]


def test_the_printouts_read_evaluates_and_forecasts_own_constants():
    # They came with the printouts from run_hurdle_pipeline.py (which imported them itself).
    from collection_estimation.hurdle import evaluate, forecast
    from collection_estimation.hurdle.reports import evaluation
    assert evaluation.CONSOLIDATED is forecast.CONSOLIDATED and evaluation.CONSOLIDATED == "ALL"
    assert evaluation.REFERENCES is evaluate.REFERENCES
    assert evaluation.POOLED is evaluate.POOLED and evaluation.POOLED == "pooled"
    assert evaluation.HORIZON_TOTAL is evaluate.HORIZON_TOTAL
    assert evaluation.HORIZON_TOTAL == "total"
    assert evaluation.KNOWN_CUSTOMERS is evaluate.KNOWN_CUSTOMERS
    assert evaluation.RULE is evaluate.RULE


def test_the_reliability_printout_shows_m_with_the_stage_2_reliability_printout():
    from collection_estimation.hurdle.reports import evaluation, stages
    assert evaluation.show_amount_reliability is stages.show_amount_reliability
