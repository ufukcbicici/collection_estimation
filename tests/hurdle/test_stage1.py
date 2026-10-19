"""Tests for `hurdle/stage1.py`: U10.1 (`NOT_FEATURES`, `feature_columns`,
`stage1_features`) and, at the end, U10.2 (`fit_stage1`, `predict_paid`).

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
                    random_seed=3):
    return Stage1Settings(max_iter=max_iter, learning_rate=learning_rate,
                          max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                          random_seed=random_seed, feature_blocks=FEATURE_BLOCKS,
                          company_as_feature=True, diagnose_class_weights=False)


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
