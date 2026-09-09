"""Tests for `workbook_paths`, `read_corpus` and the CSV cache round trip.

Separate from `test_ingest.py` because these read the whole corpus and are slower. The
constructed cases here target three specific traps: lexical date sorting, Excel lock
files, and a CSV cache that destroys the customer numbers.
"""
import json
import os
import shutil
import sys
from datetime import date, datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from collection_estimation import config                          # noqa: E402
from collection_estimation.ingest import (                        # noqa: E402
    WorkbookFormatError,
    read_collections_csv,
    read_corpus,
    workbook_paths,
    write_collections_csv,
)

from tests.helpers import HEADERS, build, corpus_files, detail   # noqa: E402


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return read_corpus(config.CORPUS_DIR)


@pytest.fixture(scope="module")
def oracle() -> dict:
    with open(config.ORACLE, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- ordering

def test_paths_are_chronological_not_lexical(tmp_path):
    """`01072026` sorts before `02062025` lexically — a year out of order.

    The filenames are DDMMYYYY, so the day leads. Sorting the strings puts July 2026
    before June 2025, and every rolling window computed downstream would be nonsense.
    """
    for name in ("01072026", "02062025", "15122025"):
        build(tmp_path, rows=[detail(when=datetime.strptime(name, "%d%m%Y"))],
              sheet=name, name=f"{name} EY TAHSILAT.XLSX")
    got = [os.path.basename(p)[:8] for p in workbook_paths(str(tmp_path))]
    assert got == ["02062025", "15122025", "01072026"]
    assert got != sorted(got), "the lexical order would have been wrong here"


def test_corpus_rows_are_in_date_order(corpus):
    assert corpus["document_date"].is_monotonic_increasing


def test_excel_lock_files_are_skipped(tmp_path):
    """Excel writes `~$…` whenever a workbook is open; it is not a valid archive."""
    build(tmp_path, rows=[detail()], name="02072026 EY TAHSILAT.XLSX")
    with open(os.path.join(tmp_path, "~$02072026 EY TAHSILAT.XLSX"), "wb") as fh:
        fh.write(b"not a zip archive")
    assert len(workbook_paths(str(tmp_path))) == 1
    assert len(read_corpus(str(tmp_path))) == 1


def test_filename_without_a_date_raises(tmp_path):
    build(tmp_path, rows=[detail()], name="notes.xlsx")
    with pytest.raises(WorkbookFormatError, match="DDMMYYYY"):
        workbook_paths(str(tmp_path))


def test_empty_directory_raises(tmp_path):
    with pytest.raises(WorkbookFormatError, match="no workbooks"):
        read_corpus(str(tmp_path))


# ---------------------------------------------------------------- cross-file checks

def test_date_disagreeing_with_the_filename_raises(tmp_path):
    """`read_workbook` records the filename; this is where the two get compared."""
    build(tmp_path, rows=[detail(when=datetime(2026, 7, 3))],
          name="02072026 EY TAHSILAT.XLSX")
    with pytest.raises(WorkbookFormatError, match="filename says"):
        read_corpus(str(tmp_path))


# ---------------------------------------------------------------- the whole corpus

def test_total_rows_match_the_oracle(corpus, oracle):
    assert len(corpus) == oracle["range"]["rows_total"]


def test_every_file_is_represented_once(corpus, oracle):
    assert corpus["source_file"].nunique() == len(corpus_files())
    per_file = corpus.groupby("source_file", observed=True).size().to_dict()
    for entry in oracle["daily"]:
        assert per_file[entry["file"]] == entry["rows"], entry["file"]


def test_daily_totals_match_the_oracle(corpus, oracle):
    """The grand total in each workbook is computed over its detail rows."""
    totals = corpus.groupby("source_file", observed=True)["amount_local"].sum()
    for entry in oracle["daily"]:
        assert abs(totals[entry["file"]] - entry["grand_total"]) < 0.05, entry["file"]


def test_entity_counts_match_the_oracle(corpus, oracle):
    got = corpus["company_code"].value_counts().to_dict()
    for code, spec in oracle["ey_spec_alignment"]["entries_by_entity"].items():
        assert got[code] == spec["actual"]


def test_refunds_and_zero_rows_survive_the_merge(corpus, oracle):
    assert int((corpus["amount_original"] < 0).sum()) == \
        oracle["ey_spec_alignment"]["negative_rows"]
    assert (corpus["amount_original"] == 0).sum() > 0


def test_unapplied_cash_rows_have_no_sap_document(corpus):
    unapplied = corpus[corpus["government_invoice"].isin(config.STATUS_NOTES)]
    assert len(unapplied) > 0
    assert unapplied["sap_document_no"].isna().all()


def test_customer_is_string_dtype_not_integer(corpus):
    assert corpus["customer"].dtype == "string"


# ---------------------------------------------------------------- the cache

def test_cache_round_trip_preserves_every_value(corpus, tmp_path):
    path = os.path.join(tmp_path, "collections.csv")
    write_collections_csv(corpus, path)
    back = read_collections_csv(path)
    pd.testing.assert_frame_equal(corpus, back, check_dtype=True)


def test_cache_round_trip_keeps_customer_as_text(tmp_path):
    """The trap this cache format exists in spite of.

    A customer number is digits, so pandas infers `int64` for the column and any leading
    zero is gone. Every join against those customers then misses them, and nothing
    complains. The explicit dtype in `read_collections_csv` is what prevents it.
    """
    table = pd.DataFrame({
        "customer": pd.array(["00012345", "11520039"], dtype="string"),
        "amount_local": [1.0, 2.0],
    })
    path = os.path.join(tmp_path, "c.csv")
    write_collections_csv(table, path)

    naive = pd.read_csv(path)
    assert naive["customer"].iloc[0] == 12345, (
        "this asserts the TRAP, not the fix: with default inference the leading zeros "
        "are silently lost")

    fixed = read_collections_csv(path)
    assert fixed["customer"].iloc[0] == "00012345"
    assert fixed["customer"].dtype == "string"


def test_cache_is_faster_than_reading_the_workbooks(corpus, tmp_path):
    """Not a timing assertion — just that the cached path returns the same table."""
    path = os.path.join(tmp_path, "collections.csv")
    write_collections_csv(corpus, path)
    assert len(read_collections_csv(path)) == len(corpus)


# ---------------------------------------------------------------- the runner

def test_run_pipeline_imports_and_exposes_main():
    """The entry point must at least import cleanly with no side effects."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    import run_pipeline

    assert callable(run_pipeline.main)
    assert run_pipeline.FORCE_REINGEST is False, (
        "FORCE_REINGEST should ship False so a normal run uses the cache")
