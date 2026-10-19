"""Tests for `load_ledger` in `hurdle/read_ledger.py`: the AR ledger read through the cache.

Each ledger is a small workbook built in `tmp_path` by `make_ledger_workbook` (see
`workbooks.py`). The cache always goes to a folder under `tmp_path`, never to the repo's
`data/interim/`. Customers are fake. One SLOW test at the end reads the synthetic ledger in
`data/raw/` once (~30 s) and is skipped when it is absent.
"""
import dataclasses
import hashlib
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle import read_collections
from collection_estimation.hurdle import read_ledger as ledger_module
from collection_estimation.hurdle.cache import file_fingerprint
from collection_estimation.hurdle.read_collections import list_daily_files, load_collections
from collection_estimation.hurdle.read_ledger import (
    LEDGER_TEXT_COLUMNS,
    load_ledger,
    read_ledger,
)
from collection_estimation.hurdle.settings import LedgerLayout
from tests.hurdle.workbooks import SYNTHETIC_LAYOUT, SYNTHETIC_LEDGER_HEADERS, SYNTHETIC_LEDGER_LAYOUT
from tests.hurdle.workbooks import (
    credit_memo,
    invoice,
    line,
    make_ledger_workbook,
    make_workbook,
    payment,
    refund,
    unapplied_payment,
)

DATE_FORMAT = SYNTHETIC_LAYOUT.file_name_date_format  # "%d%m%Y"

AS_OF = date(2026, 8, 28)
LAYOUT = SYNTHETIC_LEDGER_LAYOUT
CACHE_FILE = 'hurdle_ledger.pkl'
FINGERPRINT_FILE = 'hurdle_ledger_fingerprint.json'
CHANGED_NOTE = ('hurdle_ledger: the inputs have changed since the cache was written; '
                'reading them again.')
DAMAGED_NOTE_START = 'hurdle_ledger: the cache could not be loaded ('


def five_lines():
    """An invoice, its payment, an unapplied payment, a credit memo and its refund."""
    return [invoice(), payment(), unapplied_payment(), credit_memo(), refund()]


@pytest.fixture
def ledger_folder(tmp_path):
    folder = tmp_path / 'ledger'
    folder.mkdir()
    return folder


@pytest.fixture
def ledger_file(ledger_folder):
    return make_ledger_workbook(ledger_folder, five_lines())


@pytest.fixture
def cache_folder(tmp_path):
    return tmp_path / 'cache'


class ReadSpy:
    """Stands in for `read_ledger`: counts calls and passes them to the real one."""

    def __init__(self):
        self.calls = []

    def __call__(self, path, as_of, layout):
        self.calls.append((path, as_of, layout))
        return read_ledger(path, as_of, layout)


@pytest.fixture
def spy(monkeypatch):
    read_spy = ReadSpy()
    monkeypatch.setattr(ledger_module, 'read_ledger', read_spy)
    return read_spy


def forbid_read(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('read_ledger ran; the cache should have been used')
    monkeypatch.setattr(ledger_module, 'read_ledger', fail)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- first call ----------------------------------------------------------------------------

def test_first_call_reads_once(ledger_file, cache_folder, spy):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    assert spy.calls == [(ledger_file, AS_OF, LAYOUT)]


def test_first_call_equals_read_ledger(ledger_file, cache_folder):
    pd.testing.assert_frame_equal(load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder),
                                  read_ledger(ledger_file, AS_OF, LAYOUT))


def test_first_call_writes_the_two_cache_files(ledger_file, cache_folder):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    assert sorted(p.name for p in cache_folder.iterdir()) == [CACHE_FILE, FINGERPRINT_FILE]


def test_first_call_prints_nothing(ledger_file, cache_folder, capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    assert capsys.readouterr().out == ''


def test_first_call_accepts_string_arguments(ledger_file, cache_folder):
    table = load_ledger(str(ledger_file), AS_OF, LAYOUT, str(cache_folder))
    pd.testing.assert_frame_equal(table, read_ledger(ledger_file, AS_OF, LAYOUT))
    assert (cache_folder / CACHE_FILE).is_file()


# --- the fingerprint -----------------------------------------------------------------------

def saved_fingerprint(cache_folder):
    return json.loads((cache_folder / FINGERPRINT_FILE).read_text(encoding='utf-8'))


# The layout as the fingerprint file holds it: dataclasses.asdict, then JSON (tuples -> lists).
LAYOUT_AS_SAVED = {
    'header_row': 1,
    'first_data_row': 2,
    'headers': dict(SYNTHETIC_LEDGER_HEADERS),
    'document_types': {'invoice': ['RV'], 'payment': ['DZ'], 'credit_memo': ['DG']},
}


def test_fingerprint_file_holds_file_part_date_layout_and_two_hashes(ledger_file, cache_folder):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    saved = saved_fingerprint(cache_folder)
    assert len(saved) == 4
    file_part, as_of_text, layout_part, hashes = saved
    assert file_part == file_fingerprint(ledger_file)
    assert as_of_text == '2026-08-28'
    assert layout_part == LAYOUT_AS_SAVED
    assert isinstance(hashes, list) and len(hashes) == 2
    for digest in hashes:
        assert len(digest) == 64
        assert all(char in '0123456789abcdef' for char in digest)


def test_fingerprint_hashes_are_read_ledger_then_read_collections(ledger_file, cache_folder):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    _, _, _, hashes = saved_fingerprint(cache_folder)
    assert hashes == [sha256_of(ledger_module.__file__), sha256_of(read_collections.__file__)]


def test_fingerprint_hashes_exactly_the_two_reader_sources(ledger_file, cache_folder,
                                                           monkeypatch):
    real = ledger_module.source_fingerprint
    seen = []

    def spy_source_fingerprint(*source_files):
        seen.append(source_files)
        return real(*source_files)

    monkeypatch.setattr(ledger_module, 'source_fingerprint', spy_source_fingerprint)
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    assert len(seen) == 1
    names = [Path(source).resolve() for source in seen[0]]
    assert names == [Path(ledger_module.__file__).resolve(),
                     Path(read_collections.__file__).resolve()]


# --- second call: cache hit ----------------------------------------------------------------

def test_second_call_does_not_read(ledger_file, cache_folder, monkeypatch):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    forbid_read(monkeypatch)
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)


def test_second_call_equals_the_first_with_exact_dtypes(ledger_file, cache_folder, monkeypatch):
    first = load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    forbid_read(monkeypatch)
    second = load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    pd.testing.assert_frame_equal(second, first, check_exact=True)
    assert dict(second.dtypes) == dict(first.dtypes)


def test_second_call_keeps_none_in_text_columns(ledger_file, cache_folder, monkeypatch):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    forbid_read(monkeypatch)
    second = load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    for column in LEDGER_TEXT_COLUMNS:
        assert second[column].dtype == object, column
    assert second.loc[0, 'invoice_reference'] is None
    assert second.loc[1, 'reference'] is None
    assert second['clearing_document'].tolist() == [None] * 5


def test_second_call_prints_nothing(ledger_file, cache_folder, capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    assert capsys.readouterr().out == ''


# --- re-read when an input changes ---------------------------------------------------------

def assert_reread_with_note(spy, capsys, expected_calls=2):
    output = capsys.readouterr().out
    assert len(spy.calls) == expected_calls
    assert CHANGED_NOTE in output
    assert DAMAGED_NOTE_START not in output


def test_changed_file_content_reads_again(ledger_folder, ledger_file, cache_folder, spy, capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    make_ledger_workbook(ledger_folder, [invoice(amount=7777.0)])     # same name, new content
    table = load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    assert_reread_with_note(spy, capsys)
    assert table['amount_document'].tolist() == [7777.0]


def test_changed_mtime_alone_reads_again(ledger_file, cache_folder, spy, capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    stat = ledger_file.stat()
    os.utime(ledger_file, (stat.st_atime, stat.st_mtime - 3600))
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    assert_reread_with_note(spy, capsys)


def test_identical_file_at_another_path_reads_again(tmp_path, ledger_file, cache_folder, spy,
                                                    capsys):
    other_folder = tmp_path / 'other'
    other_folder.mkdir()
    copy = Path(shutil.copy2(ledger_file, other_folder / ledger_file.name))
    assert file_fingerprint(copy)[1:] == file_fingerprint(ledger_file)[1:]
    first = load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    second = load_ledger(copy, AS_OF, LAYOUT, cache_folder)
    assert_reread_with_note(spy, capsys)
    assert spy.calls[-1][0] == copy
    pd.testing.assert_frame_equal(second, first)


def test_later_as_of_reads_again(ledger_file, cache_folder, spy, capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    load_ledger(ledger_file, date(2026, 9, 30), LAYOUT, cache_folder)
    assert_reread_with_note(spy, capsys)
    assert spy.calls[-1][1] == date(2026, 9, 30)
    assert saved_fingerprint(cache_folder)[1] == '2026-09-30'


# --- as_of: a date, a midnight datetime and a Timestamp are the same day --------------------

SAME_DAY = {
    'date': date(2026, 8, 28),
    'datetime': datetime(2026, 8, 28),
    'timestamp': pd.Timestamp('2026-08-28'),
}


@pytest.mark.parametrize('kind', list(SAME_DAY))
def test_as_of_of_any_kind_is_stored_as_the_iso_date(ledger_file, cache_folder, kind):
    load_ledger(ledger_file, SAME_DAY[kind], LAYOUT, cache_folder)
    assert saved_fingerprint(cache_folder)[1] == '2026-08-28'


@pytest.mark.parametrize('second', list(SAME_DAY))
@pytest.mark.parametrize('first', list(SAME_DAY))
def test_as_of_of_another_kind_on_the_same_day_is_a_cache_hit(ledger_file, cache_folder,
                                                              spy, capsys, first, second):
    built = load_ledger(ledger_file, SAME_DAY[first], LAYOUT, cache_folder)
    capsys.readouterr()
    served = load_ledger(ledger_file, SAME_DAY[second], LAYOUT, cache_folder)
    assert len(spy.calls) == 1
    assert capsys.readouterr().out == ''
    pd.testing.assert_frame_equal(served, built, check_exact=True)


def test_switching_between_all_three_kinds_reads_only_once(ledger_file, cache_folder, spy):
    for kind in ['date', 'datetime', 'timestamp', 'date', 'timestamp', 'datetime']:
        load_ledger(ledger_file, SAME_DAY[kind], LAYOUT, cache_folder)
    assert len(spy.calls) == 1


@pytest.mark.parametrize('as_of', [
    datetime(2026, 8, 28, 15, 30),
    datetime(2026, 8, 28, 23, 59, 59, 999999),
    pd.Timestamp('2026-08-28 09:00'),
], ids=['afternoon', 'last-microsecond', 'timestamp-morning'])
def test_as_of_with_a_time_of_day_is_stored_as_the_date_and_hits(ledger_file, cache_folder,
                                                                 spy, as_of):
    """Pinned: the time of day is dropped, so it shares the cache of the same date."""
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    load_ledger(ledger_file, as_of, LAYOUT, cache_folder)
    assert len(spy.calls) == 1
    assert saved_fingerprint(cache_folder)[1] == '2026-08-28'


def test_as_of_with_a_time_of_day_alone_is_stored_as_the_date(ledger_file, cache_folder):
    load_ledger(ledger_file, datetime(2026, 8, 28, 15, 30), LAYOUT, cache_folder)
    assert saved_fingerprint(cache_folder)[1] == '2026-08-28'


def test_as_of_on_the_next_day_as_a_datetime_reads_again(ledger_file, cache_folder, spy,
                                                         capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    load_ledger(ledger_file, datetime(2026, 8, 29), LAYOUT, cache_folder)
    assert_reread_with_note(spy, capsys)
    assert saved_fingerprint(cache_folder)[1] == '2026-08-29'


def test_changed_source_hash_reads_again(ledger_file, cache_folder, spy, capsys, monkeypatch):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    monkeypatch.setattr(ledger_module, 'source_fingerprint',
                        lambda *source_files: ['0' * 64, '1' * 64])
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    assert_reread_with_note(spy, capsys)
    assert saved_fingerprint(cache_folder)[3] == ['0' * 64, '1' * 64]


# --- the layout in the fingerprint -----------------------------------------------------

def test_fingerprint_layout_part_is_the_layout_as_a_dict(ledger_file, cache_folder):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    layout_part = saved_fingerprint(cache_folder)[2]
    assert set(layout_part) == {'header_row', 'first_data_row', 'headers', 'document_types'}
    assert layout_part == json.loads(json.dumps(dataclasses.asdict(LAYOUT)))
    assert list(layout_part['headers']) == list(SYNTHETIC_LEDGER_HEADERS)  # file order kept


def test_an_equal_layout_built_separately_is_a_cache_hit(ledger_file, cache_folder, spy,
                                                        capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    equal = LedgerLayout(header_row=1, first_data_row=2, headers=dict(SYNTHETIC_LEDGER_HEADERS),
                         document_types={'invoice': ('RV',), 'payment': ('DZ',),
                                         'credit_memo': ('DG',)})
    assert equal == LAYOUT and equal is not LAYOUT
    load_ledger(ledger_file, AS_OF, equal, cache_folder)
    assert len(spy.calls) == 1
    assert capsys.readouterr().out == ''


def test_codes_as_lists_instead_of_tuples_are_a_cache_hit(ledger_file, cache_folder, spy):
    """Pinned: the fingerprint compares the JSON form, where a tuple and a list are the same."""
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    as_lists = dataclasses.replace(LAYOUT, document_types={
        'invoice': ['RV'], 'payment': ['DZ'], 'credit_memo': ['DG']})
    load_ledger(ledger_file, AS_OF, as_lists, cache_folder)
    assert len(spy.calls) == 1


def test_headers_in_another_order_are_a_cache_hit(ledger_file, cache_folder, spy):
    """Pinned: the saved JSON objects are compared as dicts, so the order of headers in the
    settings does not matter (and it does not change what is read)."""
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    reordered = dataclasses.replace(
        LAYOUT, headers=dict(reversed(list(SYNTHETIC_LEDGER_HEADERS.items()))))
    load_ledger(ledger_file, AS_OF, reordered, cache_folder)
    assert len(spy.calls) == 1


def test_a_changed_document_type_code_reads_again(ledger_file, cache_folder, spy, capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    changed = dataclasses.replace(LAYOUT, document_types=dict(LAYOUT.document_types,
                                                              payment=('DZ', 'DA')))
    table = load_ledger(ledger_file, AS_OF, changed, cache_folder)
    assert_reread_with_note(spy, capsys)
    assert spy.calls[-1][2] == changed
    assert saved_fingerprint(cache_folder)[2]['document_types']['payment'] == ['DZ', 'DA']
    assert table['document_role'].tolist() == [
        'invoice', 'payment', 'payment', 'credit_memo', 'payment']


def test_a_code_moved_to_another_role_reads_again_and_changes_the_roles(ledger_file,
                                                                       cache_folder, spy,
                                                                       capsys):
    first = load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    # DG moved from credit_memo to payment, a new code for credit memos.
    moved = dataclasses.replace(LAYOUT, document_types={
        'invoice': ('RV',), 'payment': ('DZ', 'DG'), 'credit_memo': ('DX',)})
    second = load_ledger(ledger_file, AS_OF, moved, cache_folder)
    assert_reread_with_note(spy, capsys)
    assert first.loc[3, 'document_role'] == 'credit_memo'
    assert second.loc[3, 'document_role'] == 'payment'


def test_a_layout_whose_codes_miss_a_type_raises_and_never_serves_the_cache(ledger_file,
                                                                           cache_folder):
    good = load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    missing_dg = dataclasses.replace(LAYOUT, document_types=dict(LAYOUT.document_types,
                                                                 credit_memo=('DX',)))
    with pytest.raises(ValueError, match=r"unknown document types .*\{'DG': 1\}"):
        load_ledger(ledger_file, AS_OF, missing_dg, cache_folder)
    pd.testing.assert_frame_equal(load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder), good)


LEDGER_HEADER_WITHOUT_CLEARING_DOCUMENT = tuple(
    text for text in SYNTHETIC_LEDGER_HEADERS if text != 'Clearing Document')


def test_a_ledger_without_the_clearing_document_column_raises_and_writes_no_cache(
        ledger_folder, cache_folder):
    path = make_ledger_workbook(ledger_folder, five_lines(),
                                header=LEDGER_HEADER_WITHOUT_CLEARING_DOCUMENT,
                                name='no_clearing_document.xlsx')
    with pytest.raises(ValueError, match=r"^missing columns \['clearing_document'\]; "
                                         r"the header row reads "):
        load_ledger(path, AS_OF, LAYOUT, cache_folder)
    assert not (cache_folder / CACHE_FILE).exists()


def test_a_cached_ledger_rewritten_without_clearing_document_raises_not_the_old_cache(
        ledger_folder, cache_folder):
    path = make_ledger_workbook(ledger_folder, five_lines(), name='ledger.xlsx')
    load_ledger(path, AS_OF, LAYOUT, cache_folder)
    stat = os.stat(path)
    make_ledger_workbook(ledger_folder, five_lines(),
                         header=LEDGER_HEADER_WITHOUT_CLEARING_DOCUMENT, name='ledger.xlsx')
    # Make sure the file's fingerprint moves even if the save lands in the same second.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    with pytest.raises(ValueError, match=r"^missing columns \['clearing_document'\]"):
        load_ledger(path, AS_OF, LAYOUT, cache_folder)


def test_a_renamed_header_reads_again_for_the_same_file(ledger_folder, cache_folder, spy,
                                                       capsys):
    # One file whose header has both 'Document Type' and a 'Belge Türü' copy of it. The
    # English layout reads it; a layout that takes document_type from 'Belge Türü' instead is
    # a changed layout, so the same file is read again rather than served from the cache.
    header = tuple(SYNTHETIC_LEDGER_HEADERS) + ('Belge Türü',)
    lines = [dict(row, **{'Belge Türü': row['Document Type']}) for row in five_lines()]
    path = make_ledger_workbook(ledger_folder, lines, header=header, name='both.xlsx')
    renamed = dataclasses.replace(LAYOUT, headers={
        ('Belge Türü' if text == 'Document Type' else text): field
        for text, field in SYNTHETIC_LEDGER_HEADERS.items()})

    first = load_ledger(path, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    second = load_ledger(path, AS_OF, renamed, cache_folder)
    assert_reread_with_note(spy, capsys)
    headers_saved = saved_fingerprint(cache_folder)[2]['headers']
    assert 'Belge Türü' in headers_saved and 'Document Type' not in headers_saved
    pd.testing.assert_frame_equal(second, first)
    # And the renamed layout now hits.
    load_ledger(path, AS_OF, renamed, cache_folder)
    assert len(spy.calls) == 2


def test_a_changed_header_text_alone_reads_again(ledger_file, cache_folder, spy, capsys):
    # An extra spelling for one field changes the layout, so the cache is not used, even
    # though the file and the table read from it are the same.
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    more = dataclasses.replace(LAYOUT, headers=dict(SYNTHETIC_LEDGER_HEADERS,
                                                    **{'Belge Türü': 'document_type'}))
    load_ledger(ledger_file, AS_OF, more, cache_folder)
    assert_reread_with_note(spy, capsys)


def test_a_changed_first_data_row_reads_again(ledger_file, cache_folder, spy, capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()
    later = dataclasses.replace(LAYOUT, first_data_row=3)
    table = load_ledger(ledger_file, AS_OF, later, cache_folder)
    assert_reread_with_note(spy, capsys)
    assert table['source_row'].tolist() == [3, 4, 5, 6]   # the invoice on row 2 is skipped
    assert saved_fingerprint(cache_folder)[2]['first_data_row'] == 3


def test_a_changed_header_row_reads_again(ledger_folder, cache_folder, spy, capsys):
    path = make_ledger_workbook(ledger_folder, five_lines(), title_rows=(('TITLE',),))
    below_title = dataclasses.replace(LAYOUT, header_row=2, first_data_row=3)
    load_ledger(path, AS_OF, below_title, cache_folder)
    capsys.readouterr()
    # The header is really on row 2: reading row 3 as the header fails, which shows the file
    # was read again rather than the cached table served.
    one_lower = dataclasses.replace(LAYOUT, header_row=3, first_data_row=4)
    with pytest.raises(ValueError, match='missing columns'):
        load_ledger(path, AS_OF, one_lower, cache_folder)
    assert_reread_with_note(spy, capsys)


def test_after_a_reread_the_next_call_hits_again(ledger_file, cache_folder, spy, capsys):
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    load_ledger(ledger_file, date(2026, 9, 30), LAYOUT, cache_folder)
    capsys.readouterr()
    load_ledger(ledger_file, date(2026, 9, 30), LAYOUT, cache_folder)
    assert len(spy.calls) == 2
    assert capsys.readouterr().out == ''


# --- errors --------------------------------------------------------------------------------

def test_read_error_propagates_and_leaves_no_cache(ledger_file, cache_folder):
    with pytest.raises(ValueError, match='after the extract date 2026-03-01'):
        load_ledger(ledger_file, date(2026, 3, 1), LAYOUT, cache_folder)
    assert not (cache_folder / CACHE_FILE).exists()
    assert not (cache_folder / FINGERPRINT_FILE).exists()


def test_read_error_after_a_good_cache_never_serves_the_old_table(ledger_file, cache_folder):
    good = load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    for _ in range(2):
        with pytest.raises(ValueError, match='after the extract date'):
            load_ledger(ledger_file, date(2026, 3, 1), LAYOUT, cache_folder)
    # The old cache is still valid for the old as_of.
    pd.testing.assert_frame_equal(load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder), good)


def test_missing_ledger_file_raises_file_not_found(tmp_path, cache_folder):
    with pytest.raises(FileNotFoundError):
        load_ledger(tmp_path / 'missing.xlsx', AS_OF, LAYOUT, cache_folder)
    assert not cache_folder.exists()


# --- next to the collections cache ---------------------------------------------------------

def make_collections(tmp_path):
    days = tmp_path / 'days'
    days.mkdir()
    make_workbook(days, [line('0012345', 10.0, when=datetime(2025, 6, 2))], name='02062025')
    return list_daily_files(days, DATE_FORMAT)


def test_collections_cache_in_the_same_folder_is_unaffected(tmp_path, ledger_folder,
                                                             ledger_file, cache_folder,
                                                             monkeypatch, capsys):
    daily_files = make_collections(tmp_path)
    collections = load_collections(daily_files, SYNTHETIC_LAYOUT, cache_folder)
    before = {name: (cache_folder / name).read_bytes()
              for name in ('hurdle_collections.pkl', 'hurdle_collections_fingerprint.json')}

    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    make_ledger_workbook(ledger_folder, [invoice(amount=5.0)])     # force a ledger re-read
    load_ledger(ledger_file, AS_OF, LAYOUT, cache_folder)
    capsys.readouterr()

    assert sorted(p.name for p in cache_folder.iterdir()) == [
        'hurdle_collections.pkl', 'hurdle_collections_fingerprint.json',
        CACHE_FILE, FINGERPRINT_FILE]
    assert {name: (cache_folder / name).read_bytes() for name in before} == before

    def fail(*args, **kwargs):
        raise AssertionError('read_all_files ran; the collections cache should have been used')
    monkeypatch.setattr(read_collections, 'read_all_files', fail)
    pd.testing.assert_frame_equal(load_collections(daily_files, SYNTHETIC_LAYOUT, cache_folder),
                                  collections)
    assert 'hurdle_collections' not in capsys.readouterr().out


# --- integration with the synthetic ledger (SLOW: one ~30 s read) ---------------------------

LEDGER_FILE = (Path(__file__).resolve().parents[2] / 'data' / 'raw'
               / 'SYNTHETIC_ey_ar_ledger_asof_20260828.xlsx')


def test_slow_synthetic_ledger_first_and_cached_calls_agree(tmp_path):
    if not LEDGER_FILE.exists():
        pytest.skip(f'{LEDGER_FILE.name} is not in data/raw')
    cache_folder = tmp_path / 'cache'
    first = load_ledger(LEDGER_FILE, AS_OF, LAYOUT, cache_folder)     # the one real read
    with pytest.MonkeyPatch.context() as patch:
        forbid_read(patch)
        second = load_ledger(LEDGER_FILE, AS_OF, LAYOUT, cache_folder)
    assert len(first) == 75_100
    assert first['document_role'].value_counts().to_dict() == {
        'invoice': 37_808, 'payment': 37_164, 'credit_memo': 128}
    pd.testing.assert_frame_equal(second, first, check_exact=True)
    assert dict(second.dtypes) == dict(first.dtypes)
