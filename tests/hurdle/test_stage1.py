"""Tests for `hurdle/stage1.py`: U10.1 (`NOT_FEATURES`, `feature_columns`,
`stage1_features`), U10.2 (`fit_stage1`, `predict_paid`) and, at the end, U10.3
(`check_labels`, `is_amount_column`, `signed_log1p`, `fit_stage1_baseline`).

`feature_columns(block_columns, blocks)` gives ["h"] + the columns of each block in the order
of `blocks`, and raises ValueError for an unknown block (checked first) or for a column in
NOT_FEATURES. `stage1_features(rows, columns, companies, company_as_feature)` gives
rows[columns] as float64 (True/False as 1.0/0.0, ints as floats, NaN kept), plus, if
`company_as_feature`, a `company_code` pandas Categorical whose categories are the sorted
distinct `companies`, whatever companies the rows hold. It raises ValueError for a missing
column (checked first) and for a company not in `companies`; it keeps the index and does not
change its input.

The last group runs both on the real training table built by the leak test's chain in
`test_training_table.py` (small in-memory data; nothing reads the corpus or data/interim).
"""
import numpy as np
import pandas as pd
import pytest

from collection_estimation.hurdle.settings import FEATURE_BLOCKS
from collection_estimation.hurdle.stage1 import NOT_FEATURES, feature_columns, stage1_features
from collection_estimation.hurdle.training_table import (
    block_feature_columns,
    rows_at_origin,
    rows_known_at,
)

nan = float("nan")

BLOCK_COLUMNS = {
    "history": ["customer_age_weeks", "mean_receipts_4"],
    "calendar": ["business_days", "has_month_end"],
    "receivables": ["open_invoices", "due_in_target_week"],
}
COMPANIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]


# --- NOT_FEATURES ----------------------------------------------------------------------------

def test_not_features_is_the_documented_list():
    assert NOT_FEATURES == ["key", "company_code", "customer_id", "origin_week", "target_week",
                            "label_status", "amount", "amount_deflated", "paid"]


def test_not_features_does_not_hold_the_horizon():
    assert "h" not in NOT_FEATURES


# --- feature_columns -------------------------------------------------------------------------

def test_feature_columns_all_blocks_in_order():
    assert feature_columns(BLOCK_COLUMNS, ("history", "calendar", "receivables")) == [
        "h", "customer_age_weeks", "mean_receipts_4", "business_days", "has_month_end",
        "open_invoices", "due_in_target_week"]


def test_feature_columns_follow_the_order_of_blocks_not_of_the_dict():
    assert feature_columns(BLOCK_COLUMNS, ("receivables", "history")) == [
        "h", "open_invoices", "due_in_target_week", "customer_age_weeks", "mean_receipts_4"]


@pytest.mark.parametrize("block", ["history", "calendar", "receivables"])
def test_feature_columns_one_block(block):
    assert feature_columns(BLOCK_COLUMNS, (block,)) == ["h"] + BLOCK_COLUMNS[block]


def test_feature_columns_with_no_blocks_is_only_h():
    assert feature_columns(BLOCK_COLUMNS, ()) == ["h"]


def test_feature_columns_h_comes_first_always():
    for blocks in (("calendar",), ("receivables", "calendar"), FEATURE_BLOCKS):
        assert feature_columns(BLOCK_COLUMNS, blocks)[0] == "h"


def test_feature_columns_an_empty_block_adds_nothing():
    columns = dict(BLOCK_COLUMNS, calendar=[])
    assert feature_columns(columns, FEATURE_BLOCKS) == [
        "h", "customer_age_weeks", "mean_receipts_4", "open_invoices", "due_in_target_week"]


def test_feature_columns_returns_a_new_list():
    result = feature_columns(BLOCK_COLUMNS, ("history",))
    result.append("x")
    assert BLOCK_COLUMNS["history"] == ["customer_age_weeks", "mean_receipts_4"]
    assert feature_columns(BLOCK_COLUMNS, ("history",)) == [
        "h", "customer_age_weeks", "mean_receipts_4"]


def test_feature_columns_does_not_change_its_inputs():
    block_columns = {name: list(columns) for name, columns in BLOCK_COLUMNS.items()}
    blocks = ("receivables", "history")
    feature_columns(block_columns, blocks)
    assert block_columns == BLOCK_COLUMNS
    assert list(block_columns) == ["history", "calendar", "receivables"]


def test_feature_columns_takes_a_list_of_blocks_too():
    assert feature_columns(BLOCK_COLUMNS, ["calendar"]) == ["h", "business_days",
                                                            "has_month_end"]


def test_feature_columns_unknown_block_exact_message():
    with pytest.raises(ValueError) as error:
        feature_columns(BLOCK_COLUMNS, ("history", "weather"))
    assert str(error.value) == ("unknown feature blocks ['weather']; known: "
                                "['history', 'calendar', 'receivables']")


def test_feature_columns_unknown_blocks_listed_in_the_order_given():
    with pytest.raises(ValueError) as error:
        feature_columns(BLOCK_COLUMNS, ("sales", "history", "weather"))
    assert str(error.value) == ("unknown feature blocks ['sales', 'weather']; known: "
                                "['history', 'calendar', 'receivables']")


def test_feature_columns_unknown_block_names_the_known_ones_from_block_columns():
    with pytest.raises(ValueError) as error:
        feature_columns({"calendar": ["business_days"]}, ("history",))
    assert str(error.value) == "unknown feature blocks ['history']; known: ['calendar']"


def test_feature_columns_block_names_are_case_sensitive():
    with pytest.raises(ValueError, match=r"unknown feature blocks \['History'\]"):
        feature_columns(BLOCK_COLUMNS, ("History",))


@pytest.mark.parametrize("forbidden", NOT_FEATURES)
def test_feature_columns_refuses_every_not_feature_column(forbidden):
    columns = dict(BLOCK_COLUMNS, history=["customer_age_weeks", forbidden])
    with pytest.raises(ValueError) as error:
        feature_columns(columns, ("history",))
    assert str(error.value) == f"columns ['{forbidden}'] must never be features"


def test_feature_columns_forbidden_columns_listed_in_column_order():
    columns = {"history": ["paid", "x"], "calendar": ["target_week"],
               "receivables": ["key", "y"]}
    with pytest.raises(ValueError) as error:
        feature_columns(columns, ("receivables", "history", "calendar"))
    assert str(error.value) == "columns ['key', 'paid', 'target_week'] must never be features"


def test_feature_columns_a_forbidden_column_in_an_unused_block_is_fine():
    columns = dict(BLOCK_COLUMNS, receivables=["amount"])
    assert feature_columns(columns, ("history",)) == ["h", "customer_age_weeks",
                                                      "mean_receipts_4"]


def test_feature_columns_unknown_block_is_reported_before_a_forbidden_column():
    columns = dict(BLOCK_COLUMNS, history=["paid"])
    with pytest.raises(ValueError, match="unknown feature blocks"):
        feature_columns(columns, ("history", "weather"))


def test_feature_columns_forbidden_names_are_compared_exactly():
    columns = {"history": ["Paid", "key_count", "amount_4"]}
    assert feature_columns(columns, ("history",)) == ["h", "Paid", "key_count", "amount_4"]


# --- stage1_features: hand-built rows -------------------------------------------------------
# Five rows on a non-default index, with every dtype the blocks produce: int64 (h, counts),
# float64 with NaN, bool (calendar flags).

FEATURES = ["h", "customer_age_weeks", "mean_receipts_4", "has_month_end", "open_invoices"]


def hand_rows(index=(40, 7, 13, 99, 2), companies=("TR04", "TR02", "TR04", "TR91", "TR02")):
    n = len(index)
    rows = pd.DataFrame({
        "key": pd.Series([f"{c}-{i}" for c, i in zip(companies, range(n))], dtype=object),
        "company_code": pd.Series(list(companies), dtype=object),
        "origin_week": pd.Series([10] * n, dtype="int64"),
        "h": pd.Series([1, 2, 3, 4, 5][:n], dtype="int64"),
        "customer_age_weeks": pd.Series([0, 5, 12, 100, 3][:n], dtype="int64"),
        "mean_receipts_4": pd.Series([1.5, nan, 0.0, -2.25, nan][:n], dtype="float64"),
        "has_month_end": pd.Series([True, False, True, False, False][:n], dtype=bool),
        "open_invoices": pd.Series([0, 1, 2, 3, 40][:n], dtype="int64"),
        "label_status": pd.Series(["labelled"] * n, dtype=object),
        "paid": pd.array([True, False, None, True, False][:n], dtype="boolean"),
    })
    rows.index = pd.Index(list(index))
    return rows


def test_stage1_features_columns_are_the_given_columns_then_company_code():
    x = stage1_features(hand_rows(), FEATURES, COMPANIES, True)
    assert list(x.columns) == FEATURES + ["company_code"]


def test_stage1_features_without_company_has_only_the_given_columns():
    x = stage1_features(hand_rows(), FEATURES, COMPANIES, False)
    assert list(x.columns) == FEATURES
    assert "company_code" not in x.columns


def test_stage1_features_every_given_column_is_float64():
    x = stage1_features(hand_rows(), FEATURES, COMPANIES, True)
    for column in FEATURES:
        assert x[column].dtype == np.float64, column


def test_stage1_features_bools_become_one_and_zero():
    x = stage1_features(hand_rows(), FEATURES, COMPANIES, False)
    assert x["has_month_end"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_stage1_features_ints_become_the_same_floats():
    x = stage1_features(hand_rows(), FEATURES, COMPANIES, False)
    assert x["h"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert x["customer_age_weeks"].tolist() == [0.0, 5.0, 12.0, 100.0, 3.0]
    assert x["open_invoices"].tolist() == [0.0, 1.0, 2.0, 3.0, 40.0]


def test_stage1_features_nan_is_kept_and_other_floats_unchanged():
    x = stage1_features(hand_rows(), FEATURES, COMPANIES, False)
    values = x["mean_receipts_4"]
    assert values.isna().tolist() == [False, True, False, False, True]
    assert values.dropna().tolist() == [1.5, 0.0, -2.25]


def test_stage1_features_no_nan_appears_where_there_was_none():
    x = stage1_features(hand_rows(), FEATURES, COMPANIES, True)
    assert x.drop(columns="mean_receipts_4").notna().all().all()


def test_stage1_features_keeps_the_index_in_order():
    rows = hand_rows()
    x = stage1_features(rows, FEATURES, COMPANIES, True)
    pd.testing.assert_index_equal(x.index, rows.index)


def test_stage1_features_values_stay_on_their_rows_with_a_shuffled_index():
    rows = hand_rows().sample(frac=1.0, random_state=3)
    x = stage1_features(rows, FEATURES, COMPANIES, True)
    for label in rows.index:
        assert x.loc[label, "h"] == float(rows.loc[label, "h"])
        assert x.loc[label, "company_code"] == rows.loc[label, "company_code"]


def test_stage1_features_company_code_is_categorical_with_the_sorted_companies():
    x = stage1_features(hand_rows(), FEATURES, COMPANIES, True)
    assert isinstance(x["company_code"].dtype, pd.CategoricalDtype)
    assert list(x["company_code"].cat.categories) == ["TR02", "TR04", "TR05", "TR06", "TR91"]
    assert x["company_code"].cat.ordered is False


def test_stage1_features_company_code_values_follow_the_rows():
    rows = hand_rows()
    x = stage1_features(rows, FEATURES, COMPANIES, True)
    assert x["company_code"].astype(object).tolist() == rows["company_code"].tolist()
    assert x["company_code"].notna().all()


def test_stage1_features_categories_are_sorted_even_if_the_list_is_not():
    x = stage1_features(hand_rows(), FEATURES, ["TR91", "TR04", "TR02", "TR06", "TR05"], True)
    assert list(x["company_code"].cat.categories) == ["TR02", "TR04", "TR05", "TR06", "TR91"]


def test_stage1_features_duplicate_companies_give_each_category_once():
    x = stage1_features(hand_rows(), FEATURES, ["TR04", "TR02", "TR91", "TR02"], True)
    assert list(x["company_code"].cat.categories) == ["TR02", "TR04", "TR91"]


def test_stage1_features_takes_a_tuple_of_companies():
    x = stage1_features(hand_rows(), FEATURES, tuple(COMPANIES), True)
    assert list(x["company_code"].cat.categories) == COMPANIES


@pytest.mark.parametrize("subset", [["TR02"], ["TR04", "TR91"], ["TR91"]])
def test_stage1_features_same_categories_whatever_companies_the_rows_hold(subset):
    rows = hand_rows()
    part = rows[rows["company_code"].isin(subset)]
    x = stage1_features(part, FEATURES, COMPANIES, True)
    assert list(x["company_code"].cat.categories) == COMPANIES
    full = stage1_features(rows, FEATURES, COMPANIES, True)
    assert x["company_code"].dtype == full["company_code"].dtype


def test_stage1_features_a_code_has_the_same_category_code_in_every_fit():
    rows = hand_rows()
    full = stage1_features(rows, FEATURES, COMPANIES, True)
    only_tr91 = stage1_features(rows[rows["company_code"] == "TR91"], FEATURES, COMPANIES, True)
    full_code = full.loc[rows["company_code"] == "TR91", "company_code"].cat.codes.unique()
    assert only_tr91["company_code"].cat.codes.unique().tolist() == full_code.tolist() == [4]


def test_stage1_features_unknown_company_exact_message():
    rows = hand_rows(companies=("TR04", "TR02", "TR99", "TR91", "TR02"))
    with pytest.raises(ValueError) as error:
        stage1_features(rows, FEATURES, COMPANIES, True)
    assert str(error.value) == "companies ['TR99'] are not in the settings' company list"


def test_stage1_features_unknown_companies_listed_sorted_once_each():
    rows = hand_rows(companies=("TR99", "TR02", "TR03", "TR99", "TR02"))
    with pytest.raises(ValueError) as error:
        stage1_features(rows, FEATURES, COMPANIES, True)
    assert str(error.value) == ("companies ['TR03', 'TR99'] are not in the settings' company "
                                "list")


def test_stage1_features_company_codes_are_compared_exactly():
    rows = hand_rows(companies=("tr02", "TR02", "TR02", "TR02", "TR02"))
    with pytest.raises(ValueError, match=r"companies \['tr02'\]"):
        stage1_features(rows, FEATURES, COMPANIES, True)


def test_stage1_features_a_missing_company_code_is_refused_with_value_error():
    rows = hand_rows(companies=("TR02", None, "TR02", "TR02", "TR02"))
    with pytest.raises(ValueError, match="are not in the settings' company list"):
        stage1_features(rows, FEATURES, COMPANIES, True)


def test_stage1_features_a_missing_and_an_unknown_company_are_refused_with_value_error():
    # A None (or NaN) code next to an unknown text code: the message sorts the unknown codes,
    # which must not turn into a TypeError from comparing None with text.
    rows = hand_rows(companies=("TR02", None, "TR99", "TR02", "TR02"))
    with pytest.raises(ValueError, match="are not in the settings' company list"):
        stage1_features(rows, FEATURES, COMPANIES, True)


def test_stage1_features_a_none_and_an_unknown_company_listed_in_str_order():
    # sorted(..., key=str): str(None) = "None" sorts before "TR99" ('N' < 'T').
    rows = hand_rows(companies=("TR02", None, "TR99", "TR02", "TR02"))
    with pytest.raises(ValueError) as error:
        stage1_features(rows, FEATURES, COMPANIES, True)
    assert str(error.value) == "companies [None, 'TR99'] are not in the settings' company list"


def test_stage1_features_a_nan_and_an_unknown_company_listed_in_str_order():
    # str(nan) = "nan" sorts after "TR99" (lower case after upper case).
    rows = hand_rows(companies=("TR02", np.nan, "TR99", "TR02", "TR02"))
    with pytest.raises(ValueError) as error:
        stage1_features(rows, FEATURES, COMPANIES, True)
    assert str(error.value) == "companies ['TR99', nan] are not in the settings' company list"


def test_stage1_features_a_lone_nan_company_has_the_exact_message():
    rows = hand_rows(companies=("TR02", np.nan, "TR02", "TR02", "TR02"))
    with pytest.raises(ValueError) as error:
        stage1_features(rows, FEATURES, COMPANIES, True)
    assert str(error.value) == "companies [nan] are not in the settings' company list"


def test_stage1_features_unknown_text_codes_still_sort_as_text_beside_a_missing_one():
    rows = hand_rows(companies=("TR99", None, "AA01", "TR03", "TR02"))
    with pytest.raises(ValueError) as error:
        stage1_features(rows, FEATURES, COMPANIES, True)
    assert str(error.value) == ("companies ['AA01', None, 'TR03', 'TR99'] are not in the "
                                "settings' company list")


def test_stage1_features_without_company_ignores_unknown_companies():
    rows = hand_rows(companies=("TR99", "TR02", "XX", "TR02", "TR02"))
    x = stage1_features(rows, FEATURES, COMPANIES, False)
    assert list(x.columns) == FEATURES


def test_stage1_features_without_company_needs_no_company_code_column():
    rows = hand_rows().drop(columns="company_code")
    x = stage1_features(rows, FEATURES, COMPANIES, False)
    assert list(x.columns) == FEATURES


def test_stage1_features_missing_column_exact_message():
    with pytest.raises(ValueError) as error:
        stage1_features(hand_rows(), FEATURES + ["weeks_since_last_payment"], COMPANIES, True)
    assert str(error.value) == "the rows have no columns ['weeks_since_last_payment']"


def test_stage1_features_missing_columns_listed_in_the_order_given():
    with pytest.raises(ValueError) as error:
        stage1_features(hand_rows(), ["zeta", "h", "alpha"], COMPANIES, False)
    assert str(error.value) == "the rows have no columns ['zeta', 'alpha']"


def test_stage1_features_missing_column_is_reported_before_an_unknown_company():
    rows = hand_rows(companies=("TR99",) * 5)
    with pytest.raises(ValueError, match="the rows have no columns"):
        stage1_features(rows, ["h", "nope"], COMPANIES, True)


def test_stage1_features_missing_company_code_column_with_company_feature_raises():
    rows = hand_rows().drop(columns="company_code")
    with pytest.raises((ValueError, KeyError)):
        stage1_features(rows, FEATURES, COMPANIES, True)


def test_stage1_features_only_the_given_columns_are_taken():
    x = stage1_features(hand_rows(), ["open_invoices", "h"], COMPANIES, False)
    assert list(x.columns) == ["open_invoices", "h"]
    for column in ("key", "paid", "label_status", "origin_week"):
        assert column not in x.columns


def test_stage1_features_does_not_change_its_input():
    rows = hand_rows()
    copy = rows.copy(deep=True)
    columns = list(FEATURES)
    companies = list(COMPANIES)
    stage1_features(rows, columns, companies, True)
    pd.testing.assert_frame_equal(rows, copy)
    assert columns == FEATURES
    assert companies == COMPANIES


def test_stage1_features_result_does_not_share_memory_with_the_input():
    rows = hand_rows()
    x = stage1_features(rows, FEATURES, COMPANIES, True)
    x.iloc[0, 0] = -99.0
    x.loc[x.index[0], "mean_receipts_4"] = -99.0
    assert rows["h"].iloc[0] == 1
    assert rows["mean_receipts_4"].iloc[0] == 1.5


def test_stage1_features_already_float_columns_are_copies():
    rows = hand_rows()
    x = stage1_features(rows, ["mean_receipts_4"], COMPANIES, False)
    x.iloc[0, 0] = 1234.0
    assert rows["mean_receipts_4"].iloc[0] == 1.5


def test_stage1_features_is_deterministic():
    rows = hand_rows()
    pd.testing.assert_frame_equal(stage1_features(rows, FEATURES, COMPANIES, True),
                                  stage1_features(rows, FEATURES, COMPANIES, True))


def test_stage1_features_empty_rows_give_an_empty_frame_with_every_column_and_dtype():
    rows = hand_rows().iloc[0:0]
    x = stage1_features(rows, FEATURES, COMPANIES, True)
    assert len(x) == 0
    assert list(x.columns) == FEATURES + ["company_code"]
    for column in FEATURES:
        assert x[column].dtype == np.float64
    assert list(x["company_code"].cat.categories) == COMPANIES


def test_stage1_features_a_single_row():
    rows = hand_rows(index=(5,), companies=("TR05",))
    x = stage1_features(rows, FEATURES, COMPANIES, True)
    assert x.shape == (1, len(FEATURES) + 1)
    assert x.loc[5, "company_code"] == "TR05"


def test_stage1_features_nullable_boolean_with_na_becomes_nan():
    rows = hand_rows()
    x = stage1_features(rows, ["paid"], COMPANIES, False)
    assert x["paid"].dtype == np.float64
    assert x["paid"].isna().tolist() == [False, False, True, False, False]
    assert x["paid"].dropna().tolist() == [1.0, 0.0, 1.0, 0.0]


def test_stage1_features_on_rows_known_at_and_rows_at_origin_keep_their_index():
    rows = hand_rows(index=(3, 1, 4, 15, 9))
    rows["origin_week"] = [8, 8, 9, 10, 10]
    rows["target_week"] = rows["origin_week"] + rows["h"]
    known = rows_known_at(rows, 11)
    x = stage1_features(known, FEATURES, COMPANIES, True)
    pd.testing.assert_index_equal(x.index, known.index)
    assert list(x.index) == [3, 1]


# --- end to end, on the chain of test_training_table.py ------------------------------------

@pytest.fixture(scope="module")
def chain():
    import contextlib
    import io

    from tests.hurdle.test_training_table import base_data, build_chain, file_days
    with contextlib.redirect_stdout(io.StringIO()):
        return build_chain(*base_data(), file_days())


def chain_columns(chain, blocks=FEATURE_BLOCKS):
    return feature_columns(block_feature_columns(chain["history"], chain["calendar"],
                                                 chain["owed"]), blocks)


def test_chain_feature_columns_hold_no_identity_week_or_label(chain):
    columns = chain_columns(chain)
    assert not set(columns) & set(NOT_FEATURES)
    assert "week" not in columns
    assert columns[0] == "h"
    assert len(columns) == len(set(columns))


def test_chain_feature_columns_are_h_then_the_blocks(chain):
    columns = chain_columns(chain)
    history = [c for c in chain["history"].columns if c not in ("key", "origin_week")]
    calendar = [c for c in chain["calendar"].columns if c != "week"]
    owed = [c for c in chain["owed"].columns if c not in ("key", "origin_week", "h")]
    assert columns == ["h"] + history + calendar + owed


def test_chain_stage1_features_on_the_training_rows(chain):
    from tests.hurdle.test_training_table import T
    columns = chain_columns(chain)
    rows = rows_known_at(chain["table"], T)
    x = stage1_features(rows, columns, COMPANIES, True)
    assert x.shape == (len(rows), len(columns) + 1)
    assert (x[columns].dtypes == np.float64).all()
    assert list(x["company_code"].cat.categories) == COMPANIES
    pd.testing.assert_index_equal(x.index, rows.index)
    # The calendar's bool flags are 0.0 / 1.0 only.
    for column, dtype in chain["calendar"].dtypes.items():
        if dtype == bool:
            assert set(x[column].unique()) <= {0.0, 1.0}, column


def test_chain_stage1_features_on_the_forecast_rows(chain):
    from tests.hurdle.test_training_table import T
    columns = chain_columns(chain)
    rows = rows_at_origin(chain["table"], T)
    x = stage1_features(rows, columns, COMPANIES, True)
    train = stage1_features(rows_known_at(chain["table"], T), columns, COMPANIES, True)
    assert list(x.columns) == list(train.columns)
    assert x.dtypes.to_dict() == train.dtypes.to_dict()


@pytest.mark.parametrize("blocks", [("calendar",), ("history", "receivables")])
def test_chain_a_subset_of_blocks_gives_only_those_columns(chain, blocks):
    from tests.hurdle.test_training_table import T
    columns = chain_columns(chain, blocks)
    x = stage1_features(rows_known_at(chain["table"], T), columns, COMPANIES, False)
    assert list(x.columns) == columns


# --- U10.2: fit_stage1 and predict_paid -----------------------------------------------------
# Small synthetic X / y and few trees, so every fit takes well under a second.

from sklearn.metrics import roc_auc_score  # noqa: E402

from collection_estimation.hurdle.settings import Stage1Settings  # noqa: E402
from collection_estimation.hurdle.stage1 import fit_stage1, predict_paid  # noqa: E402


def stage1_settings(max_iter=10, learning_rate=0.1, max_leaf_nodes=7, min_samples_leaf=5,
                    random_seed=3, baseline_c=1.0):
    return Stage1Settings(max_iter=max_iter, learning_rate=learning_rate,
                          max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                          random_seed=random_seed, feature_blocks=FEATURE_BLOCKS,
                          company_as_feature=True, diagnose_class_weights=False,
                          baseline_c=baseline_c)


def synthetic_xy(n=400, seed=0, signal=True, share=0.3, company=True, index_offset=1000):
    """x: h, a signal column s, a noise column z (with NaN), and the company code; paid is
    s above its (1 - share) quantile (signal) or an independent coin with probability
    `share` (no signal). The index starts at `index_offset`, not 0."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    x = pd.DataFrame({"h": rng.integers(1, 6, size=n).astype("float64"),
                      "s": s,
                      "z": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n))})
    if company:
        x["company_code"] = pd.Categorical(rng.choice(["TR02", "TR04"], size=n),
                                           categories=["TR02", "TR04", "TR05"])
    if signal:
        paid = s > np.quantile(s, 1 - share)
    else:
        paid = rng.random(n) < share
    x.index = pd.RangeIndex(index_offset, index_offset + n)
    return x, pd.Series(paid, index=x.index, name="paid")


# fit_stage1: the estimator's parameters

def test_fit_stage1_parameters_come_from_the_settings():
    x, paid = synthetic_xy()
    settings = stage1_settings(max_iter=12, learning_rate=0.2, max_leaf_nodes=5,
                               min_samples_leaf=7, random_seed=11)
    params = fit_stage1(x, paid, settings).get_params()
    assert params["max_iter"] == 12
    assert params["learning_rate"] == 0.2
    assert params["max_leaf_nodes"] == 5
    assert params["min_samples_leaf"] == 7
    assert params["random_state"] == 11


def test_fit_stage1_fixed_parameters():
    x, paid = synthetic_xy()
    params = fit_stage1(x, paid, stage1_settings()).get_params()
    assert params["loss"] == "log_loss"
    assert params["early_stopping"] is False
    assert params["categorical_features"] == "from_dtype"


def test_fit_stage1_class_weight_defaults_to_none():
    x, paid = synthetic_xy()
    assert fit_stage1(x, paid, stage1_settings()).get_params()["class_weight"] is None


def test_fit_stage1_balanced_class_weight_passes_through():
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings(), class_weight="balanced")
    assert model.get_params()["class_weight"] == "balanced"


def test_fit_stage1_returns_a_fitted_classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings())
    assert isinstance(model, HistGradientBoostingClassifier)
    assert list(model.feature_names_in_) == ["h", "s", "z", "company_code"]
    assert list(model.classes_) == [False, True]


def test_fit_stage1_no_early_stopping_with_more_than_10000_rows():
    # sklearn's early_stopping="auto" switches on above 10,000 rows; off, every tree is built.
    x, paid = synthetic_xy(n=12_000, signal=False, company=False)
    model = fit_stage1(x, paid, stage1_settings(max_iter=15, learning_rate=0.01))
    assert model.n_iter_ == 15
    assert model.do_early_stopping_ is False


def test_fit_stage1_same_seed_gives_identical_predictions():
    x, paid = synthetic_xy()
    one = predict_paid(fit_stage1(x, paid, stage1_settings(random_seed=5)), x)
    two = predict_paid(fit_stage1(x, paid, stage1_settings(random_seed=5)), x)
    pd.testing.assert_series_equal(one, two)


def test_fit_stage1_company_code_is_treated_as_categorical():
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings())
    assert model.is_categorical_.tolist() == [False, False, False, True]


def test_fit_stage1_without_company_nothing_is_categorical():
    x, paid = synthetic_xy(company=False)
    model = fit_stage1(x, paid, stage1_settings())
    assert model.is_categorical_ is None or not model.is_categorical_.any()


def test_fit_stage1_learns_a_signal_carried_only_by_the_company():
    # TR02 pays 90% of the time, TR05 10%; the other columns are noise.
    rng = np.random.default_rng(7)
    n = 600
    codes = rng.choice(["TR02", "TR05"], size=n)
    x = pd.DataFrame({"h": rng.integers(1, 6, size=n).astype("float64"),
                      "z": rng.normal(size=n),
                      "company_code": pd.Categorical(codes,
                                                     categories=["TR02", "TR04", "TR05"])})
    paid = pd.Series(np.where(codes == "TR02", rng.random(n) < 0.9, rng.random(n) < 0.1))
    model = fit_stage1(x, paid, stage1_settings(max_iter=30))
    p = predict_paid(model, x)
    assert p[codes == "TR02"].mean() > 0.75
    assert p[codes == "TR05"].mean() < 0.25


# fit_stage1: the inputs it accepts

def test_fit_stage1_accepts_nan_features():
    x, paid = synthetic_xy()
    x.loc[x.index[:50], "s"] = np.nan
    p = predict_paid(fit_stage1(x, paid, stage1_settings()), x)
    assert p.notna().all()


def test_fit_stage1_accepts_nullable_boolean_without_na():
    x, paid = synthetic_xy()
    plain = predict_paid(fit_stage1(x, paid, stage1_settings()), x)
    nullable = predict_paid(fit_stage1(x, paid.astype("boolean"), stage1_settings()), x)
    pd.testing.assert_series_equal(plain, nullable)


def test_fit_stage1_accepts_int_zero_one():
    x, paid = synthetic_xy()
    plain = predict_paid(fit_stage1(x, paid, stage1_settings()), x)
    model = fit_stage1(x, paid.astype("int64"), stage1_settings())
    assert list(model.classes_) == [False, True]
    pd.testing.assert_series_equal(predict_paid(model, x), plain)


def test_fit_stage1_does_not_change_its_inputs():
    x, paid = synthetic_xy()
    paid = paid.astype("boolean")
    x_copy, paid_copy = x.copy(deep=True), paid.copy(deep=True)
    fit_stage1(x, paid, stage1_settings())
    pd.testing.assert_frame_equal(x, x_copy)
    pd.testing.assert_series_equal(paid, paid_copy)


# fit_stage1: the guards

def test_fit_stage1_index_mismatch_exact_message():
    x, paid = synthetic_xy()
    with pytest.raises(ValueError) as error:
        fit_stage1(x, paid.reset_index(drop=True), stage1_settings())
    assert str(error.value) == "paid and train_x must have the same rows (the same index)"


def test_fit_stage1_same_labels_in_another_order_is_a_mismatch():
    # Same rows, other order: fitting by position would pair features with the wrong label.
    x, paid = synthetic_xy()
    with pytest.raises(ValueError, match=r"the same rows \(the same index\)"):
        fit_stage1(x, paid.iloc[::-1], stage1_settings())


def test_fit_stage1_different_lengths_is_a_mismatch():
    x, paid = synthetic_xy()
    with pytest.raises(ValueError, match=r"the same rows \(the same index\)"):
        fit_stage1(x, paid.iloc[:-1], stage1_settings())


def test_fit_stage1_na_in_paid_exact_message():
    x, paid = synthetic_xy()
    paid = paid.astype("boolean")
    paid.iloc[[3, 8]] = pd.NA
    with pytest.raises(ValueError) as error:
        fit_stage1(x, paid, stage1_settings())
    assert str(error.value) == ("paid has 2 missing values: training rows must all be "
                                "labelled")


def test_fit_stage1_nan_in_float_paid_is_refused():
    x, paid = synthetic_xy()
    paid = paid.astype("float64")
    paid.iloc[0] = np.nan
    with pytest.raises(ValueError) as error:
        fit_stage1(x, paid, stage1_settings())
    assert str(error.value) == ("paid has 1 missing values: training rows must all be "
                                "labelled")


def test_fit_stage1_index_is_checked_before_na():
    x, paid = synthetic_xy()
    paid = paid.astype("boolean")
    paid.iloc[0] = pd.NA
    with pytest.raises(ValueError, match="the same index"):
        fit_stage1(x, paid.reset_index(drop=True), stage1_settings())


def test_fit_stage1_all_na_reports_the_missing_values_not_one_class():
    x, _ = synthetic_xy(n=20)
    paid = pd.Series(pd.NA, index=x.index, dtype="boolean")
    with pytest.raises(ValueError) as error:
        fit_stage1(x, paid, stage1_settings())
    assert str(error.value) == ("paid has 20 missing values: training rows must all be "
                                "labelled")


@pytest.mark.parametrize("value", [True, False])
def test_fit_stage1_one_class_exact_message(value):
    x, _ = synthetic_xy(n=30)
    paid = pd.Series(value, index=x.index)
    with pytest.raises(ValueError) as error:
        fit_stage1(x, paid, stage1_settings())
    assert str(error.value) == (f"paid is {value} in every row; the classifier needs both "
                                f"outcomes")


def test_fit_stage1_one_class_given_as_ints_names_the_bool():
    x, _ = synthetic_xy(n=30)
    with pytest.raises(ValueError) as error:
        fit_stage1(x, pd.Series(1, index=x.index), stage1_settings())
    assert str(error.value) == "paid is True in every row; the classifier needs both outcomes"


def test_fit_stage1_empty_exact_message():
    x, paid = synthetic_xy(n=10)
    with pytest.raises(ValueError) as error:
        fit_stage1(x.iloc[0:0], paid.iloc[0:0], stage1_settings())
    assert str(error.value) == "paid is empty in every row; the classifier needs both outcomes"


# predict_paid

def test_predict_paid_is_a_series_named_p_on_the_index_of_x():
    x, paid = synthetic_xy()
    p = predict_paid(fit_stage1(x, paid, stage1_settings()), x)
    assert p.name == "p"
    pd.testing.assert_index_equal(p.index, x.index)
    assert p.dtype == np.float64


def test_predict_paid_values_are_probabilities():
    x, paid = synthetic_xy()
    p = predict_paid(fit_stage1(x, paid, stage1_settings()), x)
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_predict_paid_is_the_true_column_of_predict_proba():
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings())
    true_column = list(model.classes_).index(True)
    np.testing.assert_array_equal(predict_paid(model, x).to_numpy(),
                                  model.predict_proba(x)[:, true_column])


def test_predict_paid_on_a_shuffled_subset_keeps_each_row():
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings())
    full = predict_paid(model, x)
    part = x.sample(n=37, random_state=2)
    p = predict_paid(model, part)
    pd.testing.assert_index_equal(p.index, part.index)
    pd.testing.assert_series_equal(p, full.loc[part.index])


def test_predict_paid_on_new_rows():
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings())
    new_x, _ = synthetic_xy(n=50, seed=9, index_offset=0)
    p = predict_paid(model, new_x)
    assert len(p) == 50 and p.between(0, 1).all()


def test_predict_paid_column_order_mismatch_exact_message():
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings())
    with pytest.raises(ValueError) as error:
        predict_paid(model, x[["s", "h", "z", "company_code"]])
    assert str(error.value) == ("x has columns ['s', 'h', 'z', 'company_code'], but the model "
                                "was fitted on ['h', 's', 'z', 'company_code']")


def test_predict_paid_missing_column_raises():
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings())
    with pytest.raises(ValueError) as error:
        predict_paid(model, x.drop(columns="company_code"))
    assert str(error.value) == ("x has columns ['h', 's', 'z'], but the model was fitted on "
                                "['h', 's', 'z', 'company_code']")


def test_predict_paid_extra_column_raises():
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings())
    with pytest.raises(ValueError, match="but the model was fitted on"):
        predict_paid(model, x.assign(extra=1.0))


def test_predict_paid_does_not_change_x():
    x, paid = synthetic_xy()
    model = fit_stage1(x, paid, stage1_settings())
    copy = x.copy(deep=True)
    predict_paid(model, x)
    pd.testing.assert_frame_equal(x, copy)


def test_predict_paid_obvious_signal_gives_a_high_auc_on_new_rows():
    x, paid = synthetic_xy(n=800)
    model = fit_stage1(x, paid, stage1_settings(max_iter=30))
    new_x, new_paid = synthetic_xy(n=400, seed=1)
    # Above 0.9, not 0.95: each sample's threshold is its own quantile, so the new rows'
    # cut differs a little from the training rows'.
    assert roc_auc_score(new_paid, predict_paid(model, new_x)) > 0.9


def test_predict_paid_no_signal_mean_p_is_the_base_rate():
    x, paid = synthetic_xy(n=3000, signal=False, share=0.2, company=False)
    model = fit_stage1(x, paid, stage1_settings(max_iter=20, min_samples_leaf=50))
    assert abs(predict_paid(model, x).mean() - paid.mean()) < 0.01
    new_x, _ = synthetic_xy(n=3000, seed=4, signal=False, share=0.2, company=False)
    assert abs(predict_paid(model, new_x).mean() - 0.2) < 0.03


# no class weights vs "balanced"

def test_no_class_weights_keep_mean_p_at_the_share_paid_balanced_pushes_it_up():
    x, paid = synthetic_xy(n=3000, seed=12, signal=False, share=0.15, company=False)
    settings = stage1_settings(max_iter=30, min_samples_leaf=40)
    plain = predict_paid(fit_stage1(x, paid, settings), x)
    balanced = predict_paid(fit_stage1(x, paid, settings, class_weight="balanced"), x)
    assert abs(plain.mean() - paid.mean()) < 0.01
    assert balanced.mean() > paid.mean() + 0.2


def test_no_class_weights_keep_mean_p_at_the_share_paid_with_a_signal_too():
    x, paid = synthetic_xy(n=2000, seed=13, share=0.1)
    plain = predict_paid(fit_stage1(x, paid, stage1_settings(max_iter=30)), x)
    assert abs(plain.mean() - paid.mean()) < 0.01


# --- U10.3: check_labels, is_amount_column, signed_log1p, fit_stage1_baseline -------------

import warnings  # noqa: E402

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from collection_estimation.hurdle import stage1 as stage1_module  # noqa: E402
from collection_estimation.hurdle.features_receivables import NON_AMOUNT_COLUMNS  # noqa: E402
from collection_estimation.hurdle.stage1 import (  # noqa: E402
    AMOUNT_COLUMNS,
    AMOUNT_PATTERN,
    check_labels,
    fit_stage1_baseline,
    is_amount_column,
    signed_log1p,
)


# check_labels: what it returns

def test_check_labels_returns_bool_on_the_same_index():
    x, paid = synthetic_xy()
    result = check_labels(x, paid)
    assert result.dtype == bool
    pd.testing.assert_index_equal(result.index, x.index)
    assert result.tolist() == paid.tolist()


@pytest.mark.parametrize("dtype", ["boolean", "int64", "float64", "object"])
def test_check_labels_turns_every_label_dtype_into_bool(dtype):
    x, paid = synthetic_xy()
    result = check_labels(x, paid.astype(dtype))
    assert result.dtype == bool
    assert result.tolist() == paid.tolist()


def test_check_labels_keeps_the_name():
    x, paid = synthetic_xy()
    assert check_labels(x, paid).name == "paid"


def test_check_labels_does_not_change_its_inputs():
    x, paid = synthetic_xy()
    paid = paid.astype("boolean")
    x_copy, paid_copy = x.copy(deep=True), paid.copy(deep=True)
    check_labels(x, paid)
    pd.testing.assert_frame_equal(x, x_copy)
    pd.testing.assert_series_equal(paid, paid_copy)


def test_check_labels_two_rows_one_of_each_is_enough():
    x, _ = synthetic_xy(n=2)
    paid = pd.Series([True, False], index=x.index)
    assert check_labels(x, paid).tolist() == [True, False]


# check_labels: the guards, with the same messages fit_stage1 had

def test_check_labels_index_mismatch_exact_message():
    x, paid = synthetic_xy()
    with pytest.raises(ValueError) as error:
        check_labels(x, paid.reset_index(drop=True))
    assert str(error.value) == "paid and train_x must have the same rows (the same index)"


def test_check_labels_same_rows_in_another_order_is_a_mismatch():
    x, paid = synthetic_xy()
    with pytest.raises(ValueError, match=r"the same rows \(the same index\)"):
        check_labels(x, paid.iloc[::-1])


def test_check_labels_different_lengths_is_a_mismatch():
    x, paid = synthetic_xy()
    with pytest.raises(ValueError, match=r"the same rows \(the same index\)"):
        check_labels(x, paid.iloc[:-1])


def test_check_labels_na_exact_message():
    x, paid = synthetic_xy()
    paid = paid.astype("boolean")
    paid.iloc[[3, 8, 9]] = pd.NA
    with pytest.raises(ValueError) as error:
        check_labels(x, paid)
    assert str(error.value) == "paid has 3 missing values: training rows must all be labelled"


def test_check_labels_none_in_object_labels_is_missing():
    # astype(bool) would turn None into False: the NA check must come first.
    x, paid = synthetic_xy(n=10)
    paid = paid.astype(object)
    paid.iloc[0] = None
    with pytest.raises(ValueError, match="paid has 1 missing values"):
        check_labels(x, paid)


def test_check_labels_nan_in_float_labels_is_missing():
    # astype(bool) would turn NaN into True: the NA check must come first.
    x, paid = synthetic_xy(n=10)
    paid = paid.astype("float64")
    paid.iloc[0] = np.nan
    with pytest.raises(ValueError, match="paid has 1 missing values"):
        check_labels(x, paid)


def test_check_labels_index_is_checked_before_na():
    x, paid = synthetic_xy()
    paid = paid.astype("boolean")
    paid.iloc[0] = pd.NA
    with pytest.raises(ValueError, match="the same index"):
        check_labels(x, paid.reset_index(drop=True))


def test_check_labels_na_is_checked_before_one_class():
    x, _ = synthetic_xy(n=5)
    paid = pd.array([True, True, None, True, True], dtype="boolean")
    with pytest.raises(ValueError, match="missing values"):
        check_labels(x, pd.Series(paid, index=x.index))


@pytest.mark.parametrize("value", [True, False])
def test_check_labels_one_class_exact_message(value):
    x, _ = synthetic_xy(n=30)
    with pytest.raises(ValueError) as error:
        check_labels(x, pd.Series(value, index=x.index))
    assert str(error.value) == (f"paid is {value} in every row; the classifier needs both "
                                f"outcomes")


def test_check_labels_one_class_given_as_ints_names_the_bool():
    x, _ = synthetic_xy(n=30)
    with pytest.raises(ValueError) as error:
        check_labels(x, pd.Series(0, index=x.index))
    assert str(error.value) == "paid is False in every row; the classifier needs both outcomes"


def test_check_labels_one_row_is_one_class():
    x, _ = synthetic_xy(n=1)
    with pytest.raises(ValueError, match="paid is True in every row"):
        check_labels(x, pd.Series([True], index=x.index))


def test_check_labels_empty_exact_message():
    x, paid = synthetic_xy(n=10)
    with pytest.raises(ValueError) as error:
        check_labels(x.iloc[0:0], paid.iloc[0:0])
    assert str(error.value) == "paid is empty in every row; the classifier needs both outcomes"


# Both fits run check_labels, and with its result.

@pytest.mark.parametrize("fit", [fit_stage1, fit_stage1_baseline],
                         ids=["fit_stage1", "fit_stage1_baseline"])
def test_both_fits_call_check_labels(monkeypatch, fit):
    x, paid = synthetic_xy()
    calls = []

    def fake_check_labels(train_x, labels):
        calls.append((train_x, labels))
        raise RuntimeError("check_labels was called")

    monkeypatch.setattr(stage1_module, "check_labels", fake_check_labels)
    with pytest.raises(RuntimeError, match="check_labels was called"):
        fit(x, paid, stage1_settings())
    assert len(calls) == 1
    assert calls[0][0] is x and calls[0][1] is paid


@pytest.mark.parametrize("fit", [fit_stage1, fit_stage1_baseline],
                         ids=["fit_stage1", "fit_stage1_baseline"])
def test_both_fits_learn_from_what_check_labels_returns(monkeypatch, fit):
    # check_labels' result is what is fitted on: replace it by its opposite and the model's
    # p ranks the rows the other way round.
    x, paid = synthetic_xy(n=600)
    real = stage1_module.check_labels
    monkeypatch.setattr(stage1_module, "check_labels", lambda tx, labels: ~real(tx, labels))
    p = predict_paid(fit(x, paid, stage1_settings(max_iter=30)), x)
    assert roc_auc_score(paid, p) < 0.1


@pytest.mark.parametrize("fit", [fit_stage1, fit_stage1_baseline],
                         ids=["fit_stage1", "fit_stage1_baseline"])
@pytest.mark.parametrize("case", ["mismatch", "na", "one_class", "empty"])
def test_both_fits_raise_the_check_labels_messages(fit, case):
    x, paid = synthetic_xy(n=40)
    if case == "mismatch":
        paid = paid.reset_index(drop=True)
    elif case == "na":
        paid = paid.astype("boolean")
        paid.iloc[0] = pd.NA
    elif case == "one_class":
        paid = pd.Series(True, index=x.index)
    else:
        x, paid = x.iloc[0:0], paid.iloc[0:0]
    with pytest.raises(ValueError) as expected:
        check_labels(x, paid)
    with pytest.raises(ValueError) as error:
        fit(x, paid, stage1_settings())
    assert str(error.value) == str(expected.value)


# is_amount_column

def test_amount_columns_is_the_documented_set():
    assert AMOUNT_COLUMNS == {"last_payment_amount", "due_before_target_week",
                              "due_in_target_week", "due_after_target_week", "unapplied_cash",
                              "open_balance"}


@pytest.mark.parametrize("column", [
    "last_payment_amount", "due_before_target_week", "due_in_target_week",
    "due_after_target_week", "unapplied_cash", "open_balance",
    "mean_receipts_4", "mean_receipts_13", "mean_receipts_52", "mean_receipts_0",
    "median_payment_52", "median_payment_1",
    "overdue_up_to_30", "overdue_up_to_1", "overdue_31_to_90", "overdue_91_to_180",
    "overdue_over_90", "overdue_over_365",
])
def test_is_amount_column_true_for_amounts(column):
    assert is_amount_column(column) is True


@pytest.mark.parametrize("column", [
    # numbers of days, counts, weeks, flags, the horizon and the company
    "overdue_days", "observed_weeks_4", "observed_weeks_52", "paying_weeks_4",
    "paying_weeks_13", "weeks_since_last_payment", "customer_age_weeks", "h",
    "business_days", "has_month_end", "has_mid_month", "has_quarter_end",
    "holidays_week_before", "open_invoices", "cleared_invoices", "usual_delay_days",
    "company_code",
    # the receivables' intermediate columns, which are not features
    "overdue", "due_later",
    # near misses: a missing number, a missing part, a prefix or a suffix
    "mean_receipts_", "mean_receipts", "median_payment_", "overdue_up_to_", "overdue_over_",
    "overdue_30_to", "overdue_30_to_", "overdue__to_90", "overdue_to_90", "overdue_30_90",
    "overdue_a_to_b", "mean_receipts_4x", "xmean_receipts_4", "mean_receipts_4_",
    "mean_receipts_-4", "mean_receipts_4.0", "mean_receipts_ 4", "overdue_31_to_90_days",
    "old_overdue_over_90", "open_balance_", "_open_balance", "open_balance2",
    "Open_balance", "OPEN_BALANCE", " open_balance", "open_balance ",
    "mean_receipts_4\n", "open_balance\n", "Mean_receipts_4", "mean_receipts_4|h",
    "",
])
def test_is_amount_column_false_for_everything_else(column):
    assert is_amount_column(column) is False


def test_amount_pattern_must_match_the_whole_name():
    # The pattern is used with fullmatch: a search or a match would accept these.
    assert AMOUNT_PATTERN.search("old_mean_receipts_4") is not None
    assert AMOUNT_PATTERN.match("mean_receipts_4_extra") is not None
    assert is_amount_column("old_mean_receipts_4") is False
    assert is_amount_column("mean_receipts_4_extra") is False


def test_is_amount_column_on_the_chain_columns(chain):
    # Every feature column the real chain makes, classified: amounts are TRY, the rest are
    # counts, days, weeks or flags.
    from tests.hurdle.test_training_table import AGE_EDGES, WINDOWS
    columns = chain_columns(chain)
    amounts = [column for column in columns if is_amount_column(column)]
    expected = ({"last_payment_amount", f"median_payment_{max(WINDOWS)}",
                 "due_before_target_week", "due_in_target_week", "due_after_target_week",
                 "unapplied_cash", "open_balance"}
                | {f"mean_receipts_{window}" for window in WINDOWS}
                | {f"overdue_up_to_{AGE_EDGES[0]}", f"overdue_{AGE_EDGES[0] + 1}_to_{AGE_EDGES[1]}",
                   f"overdue_over_{AGE_EDGES[1]}"})
    assert set(amounts) == expected
    assert len(amounts) == len(expected)


def test_is_amount_column_agrees_with_what_the_receivables_block_deflates(chain):
    # features_receivables deflates every column but NON_AMOUNT_COLUMNS: the baseline's
    # log-scaling must pick exactly the deflated ones.
    receivables = [column for column in chain_columns(chain, ("receivables",)) if column != "h"]
    assert receivables, "the chain has no receivables columns"
    for column in receivables:
        assert is_amount_column(column) == (column not in NON_AMOUNT_COLUMNS), column


# signed_log1p

def test_signed_log1p_values():
    values = np.array([0.0, 1.0, -1.0, np.e - 1, -(np.e - 1), 1e6, -1e6])
    expected = np.array([0.0, np.log(2), -np.log(2), 1.0, -1.0, np.log1p(1e6), -np.log1p(1e6)])
    np.testing.assert_allclose(signed_log1p(values), expected, rtol=1e-15, atol=0)


def test_signed_log1p_zero_stays_zero():
    assert signed_log1p(0.0) == 0.0
    assert signed_log1p(np.array([0.0, -0.0])).tolist() == [0.0, 0.0]


def test_signed_log1p_nan_stays_nan():
    result = signed_log1p(np.array([np.nan, 1.0, np.nan]))
    assert np.isnan(result[0]) and np.isnan(result[2])
    assert result[1] == np.log(2)


def test_signed_log1p_infinities_keep_their_sign():
    assert signed_log1p(np.array([np.inf, -np.inf])).tolist() == [np.inf, -np.inf]


def test_signed_log1p_is_odd_and_increasing():
    values = np.linspace(-1e7, 1e7, 2001)
    result = signed_log1p(values)
    np.testing.assert_array_equal(signed_log1p(-values), -result)
    assert (np.diff(result) > 0).all()


def test_signed_log1p_keeps_tiny_values_exact_enough():
    # log1p, not log(1 + x): for tiny x the result is ~x, not 0.
    assert signed_log1p(np.array([1e-20]))[0] == pytest.approx(1e-20, rel=1e-12)
    assert signed_log1p(np.array([-1e-20]))[0] == pytest.approx(-1e-20, rel=1e-12)


def test_signed_log1p_on_a_dataframe_keeps_index_and_columns():
    frame = pd.DataFrame({"a": [0.0, 9.0, np.nan], "b": [-9.0, 99.0, 0.5]}, index=[7, 3, 5])
    result = signed_log1p(frame)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_index_equal(result.index, frame.index)
    pd.testing.assert_index_equal(result.columns, frame.columns)
    np.testing.assert_allclose(result.to_numpy(),
                               [[0.0, -np.log(10)], [np.log(10), np.log(100)],
                                [np.nan, np.log1p(0.5)]], rtol=1e-15)


def test_signed_log1p_on_a_series_keeps_the_index():
    series = pd.Series([-3.0, 3.0], index=["x", "y"])
    result = signed_log1p(series)
    assert isinstance(result, pd.Series)
    assert list(result.index) == ["x", "y"]
    assert result.tolist() == [-np.log(4), np.log(4)]


def test_signed_log1p_does_not_change_its_input():
    frame = pd.DataFrame({"a": [-5.0, 0.0, np.nan, 1e9]})
    copy = frame.copy(deep=True)
    signed_log1p(frame)
    pd.testing.assert_frame_equal(frame, copy)
    array = np.array([-5.0, 2.0])
    signed_log1p(array)
    assert array.tolist() == [-5.0, 2.0]


# fit_stage1_baseline: a small, realistic input

COMPANY_CATEGORIES = ["TR02", "TR04", "TR05", "TR06", "TR91"]


def baseline_xy(n=1500, seed=0, company=True, receivables=True, signal="amount",
                company_codes=("TR02", "TR04")):
    """Columns as stage1_features gives them: h; history (mean_receipts_4: lognormal TRY,
    NaN for a never-paying fifth; paying_weeks_4: a count; weeks_since_last_payment: NaN
    where never paid); calendar (has_month_end 0/1); receivables (open_balance with a few
    negatives, overdue_days, usual_delay_days with NaN); company_code a categorical.

    signal "amount": paid when mean_receipts_4 is large; "company": TR02 pays 90%, the
    others 10%; "none": a coin with p 0.3."""
    rng = np.random.default_rng(seed)
    never = rng.random(n) < 0.2
    receipts = rng.lognormal(mean=10, sigma=2, size=n)
    x = pd.DataFrame({
        "h": rng.integers(1, 6, size=n).astype("float64"),
        "mean_receipts_4": np.where(never, np.nan, receipts),
        "paying_weeks_4": rng.integers(0, 5, size=n).astype("float64"),
        "weeks_since_last_payment": np.where(never, np.nan,
                                             rng.integers(0, 30, size=n).astype("float64")),
        "has_month_end": rng.integers(0, 2, size=n).astype("float64"),
    })
    if receivables:
        x["open_balance"] = rng.lognormal(mean=11, sigma=2, size=n) * np.where(
            rng.random(n) < 0.05, -1, 1)
        x["overdue_days"] = rng.integers(0, 200, size=n).astype("float64")
        x["usual_delay_days"] = np.where(rng.random(n) < 0.3, np.nan,
                                         rng.normal(20, 10, size=n))
    codes = rng.choice(list(company_codes), size=n)
    if company:
        x["company_code"] = pd.Categorical(codes, categories=COMPANY_CATEGORIES)
    if signal == "amount":
        paid = np.where(never, False, receipts > np.quantile(receipts, 0.6))
    elif signal == "company":
        paid = np.where(codes == "TR02", rng.random(n) < 0.9, rng.random(n) < 0.1)
    else:
        paid = rng.random(n) < 0.3
    x.index = pd.RangeIndex(500, 500 + n)
    return x, pd.Series(paid, index=x.index, name="paid")


def parts_of(model):
    """{part name: its columns} of the fitted ColumnTransformer, without the remainder."""
    prepare = model.named_steps["prepare"]
    return {name: list(columns) for name, _, columns in prepare.transformers_
            if name != "remainder"}


# fit_stage1_baseline: the structure

def test_baseline_is_a_fitted_pipeline_prepare_then_regression():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["prepare", "regression"]
    assert isinstance(model.named_steps["prepare"], ColumnTransformer)
    assert isinstance(model.named_steps["regression"], LogisticRegression)
    assert list(model.feature_names_in_) == list(x.columns)
    assert list(model.classes_) == [False, True]


def test_baseline_sends_each_column_to_the_right_part():
    x, paid = baseline_xy()
    parts = parts_of(fit_stage1_baseline(x, paid, stage1_settings()))
    assert parts == {
        "amounts": ["mean_receipts_4", "open_balance"],
        "others": ["h", "paying_weeks_4", "weeks_since_last_payment", "has_month_end",
                   "overdue_days", "usual_delay_days"],
        "company": ["company_code"],
    }


def test_baseline_drops_no_column():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    used = [column for columns in parts_of(model).values() for column in columns]
    assert sorted(used) == sorted(x.columns)
    assert len(used) == len(set(used))


def test_baseline_part_steps():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    named = model.named_steps["prepare"].named_transformers_
    amounts = [step for _, step in named["amounts"].steps]
    assert [type(step) for step in amounts] == [FunctionTransformer, SimpleImputer,
                                                StandardScaler]
    assert amounts[0].func is signed_log1p
    assert amounts[1].strategy == "median" and amounts[1].add_indicator is True
    others = [step for _, step in named["others"].steps]
    assert [type(step) for step in others] == [SimpleImputer, StandardScaler]
    assert others[0].strategy == "median" and others[0].add_indicator is True
    assert isinstance(named["company"], OneHotEncoder)


def test_baseline_without_company_has_no_company_part():
    x, paid = baseline_xy(company=False)
    model = fit_stage1_baseline(x, paid, stage1_settings())
    assert "company" not in parts_of(model)
    assert "company_code" not in model.feature_names_in_


def test_baseline_without_receivables_columns():
    # Only the history and calendar blocks: one amount column, the rest others.
    x, paid = baseline_xy(receivables=False)
    model = fit_stage1_baseline(x, paid, stage1_settings())
    parts = parts_of(model)
    assert parts["amounts"] == ["mean_receipts_4"]
    assert parts["others"] == ["h", "paying_weeks_4", "weeks_since_last_payment",
                               "has_month_end"]
    p = predict_paid(model, x)
    assert p.between(0, 1).all() and p.notna().all()


def test_baseline_with_no_amount_column_at_all():
    # The calendar block alone: the amounts part is empty and the fit still works.
    x, paid = baseline_xy(receivables=False, signal="company")
    x = x[["h", "has_month_end", "company_code"]]
    model = fit_stage1_baseline(x, paid, stage1_settings())
    assert parts_of(model).get("amounts", []) == []
    p = predict_paid(model, x)
    assert p.between(0, 1).all() and p.notna().all()


def test_baseline_with_only_h_and_the_company():
    x, paid = baseline_xy(signal="company")
    x = x[["h", "company_code"]]
    p = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    assert p.notna().all()


# fit_stage1_baseline: what the preparation does to the numbers

def transformed(model, x):
    return model.named_steps["prepare"].transform(x)


def test_baseline_amounts_are_signed_logged_then_standardised():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    out = transformed(model, x)
    # open_balance has no NaN; it is the second amount column (after mean_receipts_4).
    logged = signed_log1p(x["open_balance"].to_numpy())
    expected = (logged - logged.mean()) / logged.std()
    np.testing.assert_allclose(out[:, 1], expected, rtol=1e-9, atol=1e-12)


def test_baseline_others_are_not_logged():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    out = transformed(model, x)
    others = parts_of(model)["others"]
    # the amounts part gives 2 columns + 1 indicator (mean_receipts_4 has NaN)
    start = 3
    column = out[:, start + others.index("overdue_days")]
    raw = x["overdue_days"].to_numpy()
    np.testing.assert_allclose(column, (raw - raw.mean()) / raw.std(), rtol=1e-9, atol=1e-12)


def test_baseline_nan_is_filled_with_the_median_of_the_logged_amount():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    imputer = model.named_steps["prepare"].named_transformers_["amounts"].steps[1][1]
    logged = signed_log1p(x["mean_receipts_4"].to_numpy())
    assert imputer.statistics_[0] == pytest.approx(np.nanmedian(logged), rel=1e-12)


def test_baseline_adds_a_was_missing_column_for_each_column_with_nan():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    named = model.named_steps["prepare"].named_transformers_
    amounts, others = parts_of(model)["amounts"], parts_of(model)["others"]
    amount_flags = [amounts[i] for i in named["amounts"].steps[1][1].indicator_.features_]
    other_flags = [others[i] for i in named["others"].steps[0][1].indicator_.features_]
    assert amount_flags == ["mean_receipts_4"]
    assert other_flags == ["weeks_since_last_payment", "usual_delay_days"]


def test_baseline_was_missing_column_marks_exactly_the_nan_rows():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    out = transformed(model, x)
    # amounts: [mean_receipts_4, open_balance, missing(mean_receipts_4)], standardised.
    flag = out[:, 2]
    missing = x["mean_receipts_4"].isna().to_numpy()
    assert len(np.unique(flag)) == 2
    assert (flag[missing] > flag[~missing].max()).all()


def test_baseline_no_nan_means_no_was_missing_column():
    x, paid = baseline_xy()
    x = x.fillna(0.0)
    model = fit_stage1_baseline(x, paid, stage1_settings())
    out = transformed(model, x)
    n_numeric = len(x.columns) - 1   # all but company_code
    assert out.shape[1] == n_numeric + len(COMPANY_CATEGORIES)


def test_baseline_output_width_counts_columns_flags_and_companies():
    x, paid = baseline_xy()
    out = transformed(fit_stage1_baseline(x, paid, stage1_settings()), x)
    # 8 numeric columns + 3 was-missing flags + 5 companies
    assert out.shape == (len(x), 8 + 3 + 5)


def test_baseline_nan_at_prediction_in_a_column_without_nan_in_training():
    # No indicator was made for open_balance; a NaN later is still filled, not an error.
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    new = x.copy()
    new.loc[new.index[:5], "open_balance"] = np.nan
    p = predict_paid(model, new)
    assert p.notna().all() and p.between(0, 1).all()


def test_baseline_a_column_with_nan_only_in_training_still_fits_and_predicts():
    x, paid = baseline_xy()
    x["usual_delay_days"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # sklearn: "Skipping features ..."
        model = fit_stage1_baseline(x, paid, stage1_settings())
        p = predict_paid(model, x)
        new = x.assign(usual_delay_days=12.0)
        q = predict_paid(model, new)
    assert p.notna().all() and q.notna().all()


def test_baseline_accepts_nan_and_predicts_finite_probabilities():
    x, paid = baseline_xy()
    assert x.isna().any().any()
    p = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    assert p.notna().all() and p.between(0, 1).all()


# fit_stage1_baseline: the company one-hot

def test_baseline_company_one_hot_has_every_category_even_if_unseen():
    x, paid = baseline_xy(company_codes=("TR02", "TR04"))
    assert set(x["company_code"].unique()) == {"TR02", "TR04"}
    model = fit_stage1_baseline(x, paid, stage1_settings())
    encoder = model.named_steps["prepare"].named_transformers_["company"]
    assert [list(c) for c in encoder.categories_] == [COMPANY_CATEGORIES]


def test_baseline_predicts_a_company_unseen_in_training():
    x, paid = baseline_xy(company_codes=("TR02", "TR04"))
    model = fit_stage1_baseline(x, paid, stage1_settings())
    new = x.iloc[:6].copy()
    new["company_code"] = pd.Categorical(["TR05", "TR06", "TR91", "TR05", "TR06", "TR91"],
                                         categories=COMPANY_CATEGORIES)
    p = predict_paid(model, new)
    assert p.notna().all() and p.between(0, 1).all()


def test_baseline_company_columns_are_one_hot_in_category_order():
    x, paid = baseline_xy(company_codes=("TR02", "TR04", "TR91"))
    model = fit_stage1_baseline(x, paid, stage1_settings())
    out = transformed(model, x)
    one_hot = out[:, -len(COMPANY_CATEGORIES):]
    assert np.isin(one_hot, [0.0, 1.0]).all()
    assert (one_hot.sum(axis=1) == 1).all()
    for position, code in enumerate(COMPANY_CATEGORIES):
        np.testing.assert_array_equal(one_hot[:, position] == 1,
                                      (x["company_code"] == code).to_numpy())


def test_baseline_company_categories_follow_the_categorical_not_the_rows():
    x, paid = baseline_xy(company_codes=("TR02", "TR04"))
    x["company_code"] = pd.Categorical(x["company_code"].astype(str),
                                       categories=["TR02", "TR04", "ZZ99"])
    assert x["company_code"].notna().all()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    encoder = model.named_steps["prepare"].named_transformers_["company"]
    assert [list(c) for c in encoder.categories_] == [["TR02", "TR04", "ZZ99"]]


# fit_stage1_baseline: the regression's parameters

def test_baseline_c_comes_from_the_settings():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings(baseline_c=0.37))
    assert model.named_steps["regression"].C == 0.37


def test_baseline_fixed_regression_parameters():
    x, paid = baseline_xy()
    regression = fit_stage1_baseline(x, paid, stage1_settings()).named_steps["regression"]
    assert regression.max_iter == 1000
    assert regression.class_weight is None


def test_baseline_balanced_class_weight_passes_through():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings(), class_weight="balanced")
    assert model.named_steps["regression"].class_weight == "balanced"


def test_baseline_smaller_c_shrinks_the_coefficients():
    x, paid = baseline_xy()
    strong = fit_stage1_baseline(x, paid, stage1_settings(baseline_c=1e-3))
    weak = fit_stage1_baseline(x, paid, stage1_settings(baseline_c=10.0))
    norm = lambda model: np.abs(model.named_steps["regression"].coef_).sum()  # noqa: E731
    assert norm(strong) < norm(weak) / 5


def test_baseline_ignores_the_hgb_settings():
    x, paid = baseline_xy()
    one = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    two = predict_paid(fit_stage1_baseline(x, paid, stage1_settings(
        max_iter=99, learning_rate=0.9, max_leaf_nodes=2, min_samples_leaf=50,
        random_seed=123)), x)
    pd.testing.assert_series_equal(one, two)


def test_baseline_balanced_pushes_mean_p_up_unweighted_keeps_it_at_the_share():
    x, paid = baseline_xy(n=3000, signal="none")
    plain = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    balanced = predict_paid(fit_stage1_baseline(x, paid, stage1_settings(),
                                                class_weight="balanced"), x)
    assert abs(plain.mean() - paid.mean()) < 0.01
    assert balanced.mean() > paid.mean() + 0.1


# fit_stage1_baseline: it learns

def test_baseline_learns_an_amount_signal():
    x, paid = baseline_xy(n=1500, signal="amount")
    model = fit_stage1_baseline(x, paid, stage1_settings())
    new_x, new_paid = baseline_xy(n=800, seed=1, signal="amount")
    assert roc_auc_score(new_paid, predict_paid(model, new_x)) > 0.9


def test_baseline_learns_a_signal_carried_only_by_the_company():
    x, paid = baseline_xy(n=1500, signal="company")
    p = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    codes = x["company_code"].astype(str)
    assert p[codes == "TR02"].mean() > 0.75
    assert p[codes == "TR04"].mean() < 0.25


def test_baseline_company_signal_is_lost_without_the_company_column():
    x, paid = baseline_xy(n=1500, signal="company")
    p = predict_paid(fit_stage1_baseline(x.drop(columns="company_code"), paid,
                                         stage1_settings()), x.drop(columns="company_code"))
    codes = x["company_code"].astype(str)
    assert abs(p[codes == "TR02"].mean() - p[codes == "TR04"].mean()) < 0.1


def test_baseline_amount_signal_survives_huge_amounts():
    # A handful of customers 1e6 times larger than the rest: without the log they would
    # flatten every other row's standardised value.
    x, paid = baseline_xy(n=1500, signal="amount")
    x.loc[x.index[:5], "mean_receipts_4"] *= 1e6
    model = fit_stage1_baseline(x, paid, stage1_settings())
    assert roc_auc_score(paid, predict_paid(model, x)) > 0.9


def test_baseline_no_signal_mean_p_is_the_base_rate():
    x, paid = baseline_xy(n=3000, signal="none")
    p = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    assert abs(p.mean() - paid.mean()) < 0.01


@pytest.mark.parametrize("signal", ["amount", "company", "none"])
@pytest.mark.parametrize("c", [0.01, 1.0, 100.0])
def test_baseline_converges_without_a_convergence_warning(signal, c):
    x, paid = baseline_xy(n=2000, signal=signal)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = fit_stage1_baseline(x, paid, stage1_settings(baseline_c=c))
    assert model.named_steps["regression"].n_iter_.max() < 1000


def test_baseline_converges_on_the_chain(chain):
    from tests.hurdle.test_training_table import T
    columns = chain_columns(chain)
    rows = rows_known_at(chain["table"], T)
    x = stage1_features(rows, columns, COMPANIES, True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = fit_stage1_baseline(x, rows["paid"], stage1_settings())
    forecast = stage1_features(rows_at_origin(chain["table"], T), columns, COMPANIES, True)
    p = predict_paid(model, forecast)
    assert p.notna().all() and p.between(0, 1).all()
    pd.testing.assert_index_equal(p.index, forecast.index)


def test_baseline_is_deterministic():
    x, paid = baseline_xy()
    one = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    two = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    pd.testing.assert_series_equal(one, two)


def test_baseline_does_not_depend_on_the_index():
    x, paid = baseline_xy()
    one = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    shifted_x, shifted_paid = x.set_axis(x.index + 10_000), paid.set_axis(paid.index + 10_000)
    two = predict_paid(fit_stage1_baseline(shifted_x, shifted_paid, stage1_settings()),
                       shifted_x)
    np.testing.assert_array_equal(one.to_numpy(), two.to_numpy())


# fit_stage1_baseline: the labels it accepts and its inputs

def test_baseline_accepts_nullable_boolean_and_int_labels():
    x, paid = baseline_xy()
    plain = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    for labels in (paid.astype("boolean"), paid.astype("int64")):
        model = fit_stage1_baseline(x, labels, stage1_settings())
        assert list(model.classes_) == [False, True]
        pd.testing.assert_series_equal(predict_paid(model, x), plain)


def test_baseline_does_not_change_its_inputs():
    x, paid = baseline_xy()
    paid = paid.astype("boolean")
    x_copy, paid_copy = x.copy(deep=True), paid.copy(deep=True)
    model = fit_stage1_baseline(x, paid, stage1_settings())
    predict_paid(model, x)
    pd.testing.assert_frame_equal(x, x_copy)
    pd.testing.assert_series_equal(paid, paid_copy)


def test_baseline_index_mismatch_exact_message():
    x, paid = baseline_xy(n=50)
    with pytest.raises(ValueError) as error:
        fit_stage1_baseline(x, paid.reset_index(drop=True), stage1_settings())
    assert str(error.value) == "paid and train_x must have the same rows (the same index)"


def test_baseline_one_class_exact_message():
    x, _ = baseline_xy(n=50)
    with pytest.raises(ValueError) as error:
        fit_stage1_baseline(x, pd.Series(False, index=x.index), stage1_settings())
    assert str(error.value) == "paid is False in every row; the classifier needs both outcomes"


# predict_paid with the baseline Pipeline

def test_predict_paid_on_the_baseline_is_a_series_named_p():
    x, paid = baseline_xy()
    p = predict_paid(fit_stage1_baseline(x, paid, stage1_settings()), x)
    assert p.name == "p"
    assert p.dtype == np.float64
    pd.testing.assert_index_equal(p.index, x.index)
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_predict_paid_on_the_baseline_is_the_true_column_of_predict_proba():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    true_column = list(model.classes_).index(True)
    np.testing.assert_array_equal(predict_paid(model, x).to_numpy(),
                                  model.predict_proba(x)[:, true_column])


def test_predict_paid_on_the_baseline_on_a_shuffled_subset_keeps_each_row():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    full = predict_paid(model, x)
    part = x.sample(n=40, random_state=3)
    pd.testing.assert_series_equal(predict_paid(model, part), full.loc[part.index])


def test_predict_paid_on_the_baseline_column_order_mismatch_exact_message():
    x, paid = synthetic_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    with pytest.raises(ValueError) as error:
        predict_paid(model, x[["s", "h", "z", "company_code"]])
    assert str(error.value) == ("x has columns ['s', 'h', 'z', 'company_code'], but the model "
                                "was fitted on ['h', 's', 'z', 'company_code']")


def test_predict_paid_on_the_baseline_missing_or_extra_column_raises():
    x, paid = synthetic_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    with pytest.raises(ValueError, match="but the model was fitted on"):
        predict_paid(model, x.drop(columns="company_code"))
    with pytest.raises(ValueError, match="but the model was fitted on"):
        predict_paid(model, x.assign(extra=1.0))


def test_predict_paid_on_the_baseline_does_not_change_x():
    x, paid = baseline_xy()
    model = fit_stage1_baseline(x, paid, stage1_settings())
    copy = x.copy(deep=True)
    predict_paid(model, x)
    pd.testing.assert_frame_equal(x, copy)
