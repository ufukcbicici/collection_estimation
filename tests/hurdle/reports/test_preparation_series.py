"""Tests of `reports/preparation.show_series_features` (U19.1): its summary line, the check
against the deflated_trend reference (yes and NO) and the preview table of correlations.

Everything is a small hand-made table; the printout is read with capsys.

The example: companies A and B, origins 5 and 9 to 12, h 1, mean_weeks 1 (so the trailing
mean at an origin is that week's own cash, and "the ratio that came" is cash[o + 1] / cash[o]).

  week    9   10   11   12   13
  A      10   20   10   20   10
  B       5   10   30   10   30
  ALL    15   30   40   30   40

At the validation origins 10, 11, 12 the ratio that came is
  A:   cash[o + 1] / cash[o] = 10/20, 20/10, 10/20                = 0.5, 2.0, 0.5
  B:   30/10, 10/30, 30/10                                       = 3.0, 1/3, 3.0
  ALL: 40/30, 30/40, 40/30
The series rows give trend_ratio exactly those (correlation +1 over the 6 company rows) and
all_trend_ratio their negatives (correlation -1 over the 3 origins). Origin 9's rows carry a
wild trend_ratio of 100, which would spoil the correlation if a non-validation origin counted;
origin 5 has no forecast at all.
"""
import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.reports.preparation import show_series_features

REAL = {
    "A": pd.Series([10.0, 20.0, 10.0, 20.0, 10.0], index=[9, 10, 11, 12, 13]),
    "B": pd.Series([5.0, 10.0, 30.0, 10.0, 30.0], index=[9, 10, 11, 12, 13]),
    "ALL": pd.Series([15.0, 30.0, 40.0, 30.0, 40.0], index=[9, 10, 11, 12, 13]),
}
CAME = {"A": {10: 0.5, 11: 2.0, 12: 0.5}, "B": {10: 3.0, 11: 1 / 3, 12: 3.0},
        "ALL": {10: 40 / 30, 11: 30 / 40, 12: 40 / 30}}
# The forecasts in the series table (and the reference), per company and origin.
FORECAST = {"A": {9: 11.0, 10: 12.0, 11: 13.0, 12: 14.0},
            "B": {9: 21.0, 10: 22.0, 11: 23.0, 12: 24.0},
            "ALL": {9: 31.0, 10: 32.0, 11: 33.0, 12: 34.0}}
VALIDATION = [10, 11, 12]


def make_series():
    rows = []
    for origin in [5, 9, 10, 11, 12]:
        for company in ["A", "B"]:
            if origin == 5:
                values = [np.nan] * 4
            elif origin == 9:
                values = [FORECAST[company][9], 100.0, FORECAST["ALL"][9], 100.0]
            else:
                values = [FORECAST[company][origin], CAME[company][origin],
                          FORECAST["ALL"][origin], -CAME["ALL"][origin]]
            rows.append([company, origin, 1, origin + 1] + values)
    return pd.DataFrame(rows, columns=["company_code", "origin_week", "h", "target_week",
                                       "trend_forecast", "trend_ratio", "all_trend_forecast",
                                       "all_trend_ratio"])


def make_reference():
    """deflated_trend rows at the validation origins for A, B and ALL, equal to the series;
    plus rows that must be ignored: another variant, and a fallback, both with other numbers."""
    rows = []
    for company in ["A", "B", "ALL"]:
        for origin in VALIDATION:
            rows.append({"company_code": company, "origin_week": origin, "h": 1,
                         "reference": "deflated_trend",
                         "real_forecast": FORECAST[company][origin], "fallback": False})
            rows.append({"company_code": company, "origin_week": origin, "h": 1,
                         "reference": "deflated", "real_forecast": 999.0, "fallback": False})
    rows.append({"company_code": "A", "origin_week": 9, "h": 1, "reference": "deflated_trend",
                 "real_forecast": 555.0, "fallback": True})
    return pd.DataFrame(rows)


def printed(capsys, series, reference, validation=VALIDATION, mean_weeks=1, real=REAL):
    capsys.readouterr()
    show_series_features(series, real, reference, validation, mean_weeks)
    return capsys.readouterr().out.splitlines()


SUMMARY = ("        series features: 10 rows (origins 5-12, h 1-1, 2 companies); a trend "
           "forecast for 80% of them, from origin 9 on (consolidated: from origin 9); ratios "
           "over the trailing 1 weeks")


def test_summary_line(capsys):
    lines = printed(capsys, make_series(), make_reference())
    assert lines[0] == SUMMARY


def test_summary_line_names_the_mean_window(capsys):
    lines = printed(capsys, make_series(), make_reference(), mean_weeks=13)
    assert lines[0].endswith("ratios over the trailing 13 weeks")


def test_summary_first_origins_differ_for_company_and_consolidated(capsys):
    series = make_series()
    series.loc[series["origin_week"] == 9, ["all_trend_forecast", "all_trend_ratio"]] = np.nan
    lines = printed(capsys, series, make_reference())
    assert "from origin 9 on (consolidated: from origin 10)" in lines[0]


def test_check_says_yes_when_every_forecast_is_the_references(capsys):
    lines = printed(capsys, make_series(), make_reference())
    # 3 companies (A, B, ALL) x 3 origins; the "deflated" rows and the fallback are not
    # compared.
    assert lines[1] == ("        the same as the deflated_trend reference at the validation "
                        "origins: yes (9 forecasts compared, largest difference 0, 0 missing in the series)")


def test_check_says_no_and_gives_the_largest_difference(capsys):
    reference = make_reference()
    pick = ((reference["company_code"] == "B") & (reference["origin_week"] == 11)
            & (reference["reference"] == "deflated_trend"))
    reference.loc[pick, "real_forecast"] += 0.25
    lines = printed(capsys, make_series(), reference)
    assert lines[1] == ("        the same as the deflated_trend reference at the validation "
                        "origins: NO (9 forecasts compared, largest difference 0.25, 0 missing in the "
                        "series)")


def test_check_says_no_when_a_consolidated_forecast_differs(capsys):
    reference = make_reference()
    pick = ((reference["company_code"] == "ALL") & (reference["origin_week"] == 12)
            & (reference["reference"] == "deflated_trend"))
    reference.loc[pick, "real_forecast"] -= 2.0
    lines = printed(capsys, make_series(), reference)
    assert lines[1].endswith(": NO (9 forecasts compared, largest difference 2, 0 missing in "
                             "the series)")


def test_check_reads_the_consolidated_forecast_from_all_trend_forecast(capsys):
    # Changing the companies' trend_forecast of ALL's rows is impossible (no ALL rows); the
    # consolidated value comes from all_trend_forecast. Changing it on one company row only
    # would not be seen if the check read another row: here every row of origin 10 changes.
    series = make_series()
    series.loc[series["origin_week"] == 10, "all_trend_forecast"] += 1.5
    lines = printed(capsys, series, make_reference())
    assert lines[1].endswith(": NO (9 forecasts compared, largest difference 1.5, 0 missing "
                             "in the series)")


def test_check_says_no_when_a_series_forecast_is_nan_where_the_reference_has_one(capsys):
    # The reference used its ridge for B at origin 11 (not a fallback), but the series has
    # no forecast there. That is not "the same".
    series = make_series()
    pick = (series["company_code"] == "B") & (series["origin_week"] == 11)
    series.loc[pick, "trend_forecast"] = np.nan
    lines = printed(capsys, series, make_reference())
    # The other 8 agree exactly: largest difference 0 over them, 1 missing.
    assert lines[1] == ("        the same as the deflated_trend reference at the validation "
                        "origins: NO (9 forecasts compared, largest difference 0, 1 missing in "
                        "the series)")


def test_check_says_no_when_the_first_compared_forecast_is_nan(capsys):
    # A at origin 10 is the first reference row compared: the verdict must not depend on
    # where the NaN sits.
    series = make_series()
    pick = (series["company_code"] == "A") & (series["origin_week"] == 10)
    series.loc[pick, "trend_forecast"] = np.nan
    lines = printed(capsys, series, make_reference())
    assert lines[1] == ("        the same as the deflated_trend reference at the validation "
                        "origins: NO (9 forecasts compared, largest difference 0, 1 missing in "
                        "the series)")


def test_check_with_a_nan_and_a_difference_gives_the_largest_over_the_numbers(capsys):
    # One forecast missing (B at 11) and one off by 0.25 (A at 12): largest 0.25, 1 missing.
    series = make_series()
    series.loc[(series["company_code"] == "B") & (series["origin_week"] == 11),
               "trend_forecast"] = np.nan
    reference = make_reference()
    reference.loc[(reference["company_code"] == "A") & (reference["origin_week"] == 12)
                  & (reference["reference"] == "deflated_trend"), "real_forecast"] += 0.25
    lines = printed(capsys, series, reference)
    assert lines[1].endswith(": NO (9 forecasts compared, largest difference 0.25, 1 missing "
                             "in the series)")


def test_check_with_every_forecast_missing_gives_nan_as_the_largest(capsys):
    series = make_series()
    series.loc[series["origin_week"].isin(VALIDATION),
               ["trend_forecast", "all_trend_forecast"]] = np.nan
    lines = printed(capsys, series, make_reference())
    assert lines[1].endswith(": NO (9 forecasts compared, largest difference nan, 9 missing "
                             "in the series)")


def test_preview_header_line(capsys):
    lines = printed(capsys, make_series(), make_reference())
    assert lines[2] == ("        preview on the validation origins: the forecast ratio against "
                        "the ratio that came")


def test_preview_table_counts_and_correlations(capsys):
    lines = printed(capsys, make_series(), make_reference())
    table = lines[3:]
    assert len(table) == 3                      # the column names, the index name, h = 1
    assert "company rows" in table[0] and "all_trend_ratio correlation" in table[0]
    assert table[1].split() == ["h"]
    # h 1: 6 company rows, correlation 1; 3 consolidated rows, correlation -1. Origin 9's
    # wild ratio of 100 is not in (it is not a validation origin).
    assert table[2].split() == ["1", "6", "1.0", "3", "-1.0"]


def test_preview_leaves_out_a_target_week_without_data(capsys):
    real = {name: series.drop(13) for name, series in REAL.items()}   # origin 12's target
    lines = printed(capsys, make_series(), make_reference(), real=real)
    assert lines[5].split() == ["1", "4", "1.0", "2", "-1.0"]


def test_preview_has_one_line_per_h(capsys):
    series = make_series()
    second = series.copy()
    second["h"] = 2
    second["target_week"] = second["origin_week"] + 2
    # For h 2 the target week of origin 12 (week 14) has no data; the others do.
    lines = printed(capsys, pd.concat([series, second], ignore_index=True), make_reference())
    table = lines[3:]
    assert [row.split()[0] for row in table[2:]] == ["1", "2"]
    assert table[3].split()[1] == "4" and table[3].split()[3] == "2"


def test_no_validation_origins(capsys):
    reference = make_reference().iloc[0:0]
    lines = printed(capsys, make_series(), reference, validation=[])
    assert lines[0] == SUMMARY
    assert lines[1] == ("        the same as the deflated_trend reference at the validation "
                        "origins: NO (0 forecasts compared, largest difference nan, 0 missing in the "
                        "series)")
    assert lines[2] == "        preview: no series rows at the validation origins"
    assert len(lines) == 3


def test_validation_origins_with_no_series_rows(capsys):
    reference = make_reference().iloc[0:0]
    lines = printed(capsys, make_series(), reference, validation=[40, 41])
    assert lines[-1] == "        preview: no series rows at the validation origins"


def test_inputs_are_not_modified(capsys):
    series, reference = make_series(), make_reference()
    before = (series.copy(), reference.copy())
    printed(capsys, series, reference)
    pd.testing.assert_frame_equal(series, before[0])
    pd.testing.assert_frame_equal(reference, before[1])


@pytest.mark.parametrize("name", ["A", "B", "ALL"])
def test_real_is_not_modified(capsys, name):
    real = {key: value.copy() for key, value in REAL.items()}
    printed(capsys, make_series(), make_reference(), real=real)
    pd.testing.assert_series_equal(real[name], REAL[name])
