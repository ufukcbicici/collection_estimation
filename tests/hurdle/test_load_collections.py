"""Tests for `load_collections`, `read_cache` and `files_fingerprint` in `hurdle/read_collections.py`.

Each day is a small workbook built in `tmp_path` by `make_workbook` (see `workbooks.py`).
The cache always goes to a folder under `tmp_path`, never to the repo's `data/interim/`.
Customers are fake.
"""
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle import read_collections
from collection_estimation.hurdle.read_collections import (
    files_fingerprint,
    list_daily_files,
    load_collections,
    read_all_files,
    read_cache,
)
from tests.hurdle.workbooks import line, make_workbook

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CORPUS = REPO_ROOT / "data" / "raw" / "SYNTHETIC_ey_tahsilat_20250602_20260828"

CACHE_FILE = "hurdle_collections.pkl"
FINGERPRINT_FILE = "hurdle_collections_fingerprint.json"
CHANGED_NOTE = "The daily files have changed since the cache was written; reading them again."
DAMAGED_NOTE_START = "The cache could not be loaded ("
DAMAGED_NOTE_END = "); reading the files again."


def make_day(folder, day, *lines, **options):
    """Write the workbook for `day`; `lines` are (customer, amount) pairs dated `day`."""
    when = datetime(day.year, day.month, day.day)
    rows = [line(customer, amount, when=when) for customer, amount in lines]
    return make_workbook(folder, rows, name=day.strftime('%d%m%Y'), **options)


def two_days(folder):
    """2 June (two lines, one with a missing SAP number) and 3 June (one line), 2025."""
    june_2 = datetime(2025, 6, 2)
    make_workbook(folder, [line('0012345', 10.0, when=june_2),
                           line('0000002', 20.0, when=june_2, sap=None, bank=None)],
                  name='02062025')
    make_day(folder, date(2025, 6, 3), ('0000003', 30.0))
    return list_daily_files(folder)


@pytest.fixture
def days_folder(tmp_path):
    folder = tmp_path / 'days'
    folder.mkdir()
    return folder


@pytest.fixture
def cache_folder(tmp_path):
    return tmp_path / 'cache'


class ReadSpy:
    """Stands in for `read_all_files`: counts calls and passes them to the real one."""

    def __init__(self):
        self.calls = 0
        self.last_files = None

    def __call__(self, daily_files):
        self.calls += 1
        self.last_files = dict(daily_files)
        return read_all_files(daily_files)


@pytest.fixture
def read_spy(monkeypatch):
    spy = ReadSpy()
    monkeypatch.setattr(read_collections, 'read_all_files', spy)
    return spy


def forbid_reading(monkeypatch):
    """From now on, reading the workbooks fails the test."""
    def fail(*args, **kwargs):
        raise AssertionError('the workbooks were read; the cache should have been used')
    monkeypatch.setattr(read_collections, 'read_all_files', fail)
    monkeypatch.setattr(read_collections, 'read_one_file', fail)


# --- the first call: no cache yet ------------------------------------------------------

def test_first_call_returns_the_same_table_as_read_all_files(days_folder, cache_folder):
    daily_files = two_days(days_folder)
    pd.testing.assert_frame_equal(load_collections(daily_files, cache_folder),
                                  read_all_files(daily_files))


def test_first_call_writes_both_cache_files(days_folder, cache_folder):
    load_collections(two_days(days_folder), cache_folder)
    assert (cache_folder / CACHE_FILE).is_file()
    assert (cache_folder / FINGERPRINT_FILE).is_file()


def test_first_call_creates_a_nested_cache_folder(days_folder, tmp_path):
    nested = tmp_path / 'a' / 'b' / 'c'
    load_collections(two_days(days_folder), nested)
    assert (nested / CACHE_FILE).is_file()


def test_first_call_reads_the_workbooks_once(days_folder, cache_folder, read_spy):
    load_collections(two_days(days_folder), cache_folder)
    assert read_spy.calls == 1


def test_first_call_prints_no_change_note(days_folder, cache_folder, capsys):
    load_collections(two_days(days_folder), cache_folder)
    assert CHANGED_NOTE not in capsys.readouterr().out


def test_fingerprint_file_holds_the_files_fingerprint(days_folder, cache_folder):
    daily_files = two_days(days_folder)
    load_collections(daily_files, cache_folder)
    written = json.loads((cache_folder / FINGERPRINT_FILE).read_text(encoding='utf-8'))
    assert written == files_fingerprint(daily_files)


def test_accepts_the_cache_folder_as_a_string(days_folder, cache_folder):
    load_collections(two_days(days_folder), str(cache_folder))
    assert (cache_folder / CACHE_FILE).is_file()


# --- the second call: files unchanged --------------------------------------------------

def test_second_call_returns_an_equal_table(days_folder, cache_folder):
    daily_files = two_days(days_folder)
    first = load_collections(daily_files, cache_folder)
    pd.testing.assert_frame_equal(load_collections(daily_files, cache_folder), first)


def test_second_call_does_not_read_the_workbooks(days_folder, cache_folder, monkeypatch):
    daily_files = two_days(days_folder)
    first = load_collections(daily_files, cache_folder)
    forbid_reading(monkeypatch)
    pd.testing.assert_frame_equal(load_collections(daily_files, cache_folder), first)


def test_second_call_prints_nothing(days_folder, cache_folder, capsys):
    daily_files = two_days(days_folder)
    load_collections(daily_files, cache_folder)
    capsys.readouterr()
    load_collections(daily_files, cache_folder)
    assert capsys.readouterr().out == ''


def cached_table(days_folder, cache_folder, monkeypatch):
    """The table as it comes back from the cache (second call, reading forbidden)."""
    daily_files = two_days(days_folder)
    load_collections(daily_files, cache_folder)
    forbid_reading(monkeypatch)
    return load_collections(daily_files, cache_folder)


def test_cache_keeps_the_customer_leading_zeros(days_folder, cache_folder, monkeypatch):
    table = cached_table(days_folder, cache_folder, monkeypatch)
    assert table['customer'].iloc[0] == '0012345'


def test_cache_keeps_missing_text_as_none(days_folder, cache_folder, monkeypatch):
    table = cached_table(days_folder, cache_folder, monkeypatch)
    assert table['sap_document_no'].iloc[1] is None
    assert table['bank'].iloc[1] is None


def test_cache_keeps_amounts_float64(days_folder, cache_folder, monkeypatch):
    table = cached_table(days_folder, cache_folder, monkeypatch)
    assert table['amount_try'].dtype == 'float64'
    assert table['amount_original'].dtype == 'float64'


def test_cache_keeps_source_row_int64(days_folder, cache_folder, monkeypatch):
    table = cached_table(days_folder, cache_folder, monkeypatch)
    assert table['source_row'].dtype == 'int64'


def test_cache_keeps_document_date_datetime64(days_folder, cache_folder, monkeypatch):
    table = cached_table(days_folder, cache_folder, monkeypatch)
    assert pd.api.types.is_datetime64_any_dtype(table['document_date'])


# --- files changed: read again ---------------------------------------------------------

def test_added_file_is_read_again_with_the_note(days_folder, cache_folder, read_spy, capsys):
    load_collections(two_days(days_folder), cache_folder)
    make_day(days_folder, date(2025, 6, 4), ('0000004', 40.0))
    table = load_collections(list_daily_files(days_folder), cache_folder)
    assert CHANGED_NOTE in capsys.readouterr().out
    assert read_spy.calls == 2
    assert list(table['customer']) == ['0012345', '0000002', '0000003', '0000004']


def test_removed_file_is_read_again_with_the_note(days_folder, cache_folder, read_spy, capsys):
    daily_files = two_days(days_folder)
    load_collections(daily_files, cache_folder)
    del daily_files[date(2025, 6, 3)]
    table = load_collections(daily_files, cache_folder)
    assert CHANGED_NOTE in capsys.readouterr().out
    assert read_spy.calls == 2
    assert list(table['customer']) == ['0012345', '0000002']


def test_rewritten_file_is_read_again_with_the_note(days_folder, cache_folder, read_spy,
                                                     capsys):
    daily_files = two_days(days_folder)
    load_collections(daily_files, cache_folder)
    size_before = daily_files[date(2025, 6, 3)].stat().st_size
    make_day(days_folder, date(2025, 6, 3), ('0000003', 30.0), ('0000009', 90.0))
    assert daily_files[date(2025, 6, 3)].stat().st_size != size_before
    table = load_collections(daily_files, cache_folder)
    assert CHANGED_NOTE in capsys.readouterr().out
    assert read_spy.calls == 2
    assert list(table['customer']) == ['0012345', '0000002', '0000003', '0000009']


def test_same_size_new_mtime_is_read_again_with_the_note(days_folder, cache_folder, read_spy,
                                                          capsys):
    daily_files = two_days(days_folder)
    load_collections(daily_files, cache_folder)
    path = daily_files[date(2025, 6, 3)]
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime - 3600))
    assert path.stat().st_size == stat.st_size
    load_collections(daily_files, cache_folder)
    assert CHANGED_NOTE in capsys.readouterr().out
    assert read_spy.calls == 2


def test_fingerprint_file_is_updated_after_a_re_read(days_folder, cache_folder):
    load_collections(two_days(days_folder), cache_folder)
    make_day(days_folder, date(2025, 6, 4), ('0000004', 40.0))
    daily_files = list_daily_files(days_folder)
    load_collections(daily_files, cache_folder)
    written = json.loads((cache_folder / FINGERPRINT_FILE).read_text(encoding='utf-8'))
    assert written == files_fingerprint(daily_files)


def test_third_call_after_a_re_read_is_a_cache_hit(days_folder, cache_folder, monkeypatch,
                                                    capsys):
    load_collections(two_days(days_folder), cache_folder)
    make_day(days_folder, date(2025, 6, 4), ('0000004', 40.0))
    daily_files = list_daily_files(days_folder)
    second = load_collections(daily_files, cache_folder)
    capsys.readouterr()
    forbid_reading(monkeypatch)
    pd.testing.assert_frame_equal(load_collections(daily_files, cache_folder), second)
    assert capsys.readouterr().out == ''


def test_third_call_returns_the_new_table_not_the_old_one(days_folder, cache_folder,
                                                          monkeypatch):
    load_collections(two_days(days_folder), cache_folder)
    make_day(days_folder, date(2025, 6, 4), ('0000004', 40.0))
    daily_files = list_daily_files(days_folder)
    load_collections(daily_files, cache_folder)
    forbid_reading(monkeypatch)
    assert len(load_collections(daily_files, cache_folder)) == 4


# --- one cache file missing ------------------------------------------------------------

def test_missing_fingerprint_file_reads_again(days_folder, cache_folder, read_spy):
    daily_files = two_days(days_folder)
    first = load_collections(daily_files, cache_folder)
    (cache_folder / FINGERPRINT_FILE).unlink()
    pd.testing.assert_frame_equal(load_collections(daily_files, cache_folder), first)
    assert read_spy.calls == 2
    assert (cache_folder / FINGERPRINT_FILE).is_file()


def test_missing_cache_file_reads_again(days_folder, cache_folder, read_spy):
    daily_files = two_days(days_folder)
    first = load_collections(daily_files, cache_folder)
    (cache_folder / CACHE_FILE).unlink()
    pd.testing.assert_frame_equal(load_collections(daily_files, cache_folder), first)
    assert read_spy.calls == 2
    assert (cache_folder / CACHE_FILE).is_file()


# --- files_fingerprint on its own ------------------------------------------------------

def test_fingerprint_has_one_entry_per_file_in_dict_order(days_folder):
    june = make_day(days_folder, date(2025, 6, 2), ('0000001', 10.0))
    july = make_day(days_folder, date(2025, 7, 1), ('0000002', 20.0))
    fingerprint = files_fingerprint({date(2025, 7, 1): july, date(2025, 6, 2): june})
    assert [entry[0] for entry in fingerprint] == [str(july.resolve()), str(june.resolve())]


def test_fingerprint_entries_are_full_path_size_mtime(days_folder):
    daily_files = two_days(days_folder)
    expected = [[str(path.resolve()), path.stat().st_size, path.stat().st_mtime]
                for path in daily_files.values()]
    assert files_fingerprint(daily_files) == expected


def test_fingerprint_path_is_absolute_even_for_a_relative_path(days_folder, monkeypatch):
    path = make_day(days_folder, date(2025, 6, 2), ('0000001', 10.0))
    monkeypatch.chdir(days_folder)
    fingerprint = files_fingerprint({date(2025, 6, 2): Path(path.name)})
    assert fingerprint[0][0] == str(path.resolve())
    assert Path(fingerprint[0][0]).is_absolute()


def copy_folder_with_same_mtimes(source, target):
    """Copy every file of `source` into `target`, with identical names, sizes and mtimes."""
    target.mkdir()
    for path in source.iterdir():
        copy = Path(shutil.copy2(path, target / path.name))
        stat = path.stat()
        os.utime(copy, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_same_names_sizes_mtimes_in_two_folders_give_different_fingerprints(tmp_path):
    folder_a, folder_b = tmp_path / 'a', tmp_path / 'b'
    folder_a.mkdir()
    two_days(folder_a)
    copy_folder_with_same_mtimes(folder_a, folder_b)
    files_a, files_b = list_daily_files(folder_a), list_daily_files(folder_b)
    fingerprint_a, fingerprint_b = files_fingerprint(files_a), files_fingerprint(files_b)
    # Only the folder differs: names, sizes and mtimes are identical.
    assert [entry[1:] for entry in fingerprint_a] == [entry[1:] for entry in fingerprint_b]
    assert [Path(entry[0]).name for entry in fingerprint_a] == \
        [Path(entry[0]).name for entry in fingerprint_b]
    assert fingerprint_a != fingerprint_b


def test_fingerprint_survives_a_json_round_trip(days_folder):
    fingerprint = files_fingerprint(two_days(days_folder))
    assert json.loads(json.dumps(fingerprint)) == fingerprint


def test_fingerprint_survives_the_indented_json_the_cache_writes(days_folder):
    fingerprint = files_fingerprint(two_days(days_folder))
    assert json.loads(json.dumps(fingerprint, indent=1)) == fingerprint


def test_fingerprint_of_an_empty_dict_is_empty():
    assert files_fingerprint({}) == []


def test_fingerprint_changes_with_the_mtime(days_folder):
    daily_files = two_days(days_folder)
    before = files_fingerprint(daily_files)
    path = daily_files[date(2025, 6, 2)]
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime - 3600))
    assert files_fingerprint(daily_files) != before


# --- one cache folder shared by two corpus folders -------------------------------------

def test_switching_to_an_identical_copy_in_another_folder_reads_again(tmp_path, cache_folder,
                                                                      read_spy, capsys):
    """Same names, sizes and mtimes; only the folder differs (e.g. synthetic -> real)."""
    folder_a, folder_b = tmp_path / 'a', tmp_path / 'b'
    folder_a.mkdir()
    two_days(folder_a)
    copy_folder_with_same_mtimes(folder_a, folder_b)
    load_collections(list_daily_files(folder_a), cache_folder)
    capsys.readouterr()
    files_b = list_daily_files(folder_b)
    table = load_collections(files_b, cache_folder)
    assert CHANGED_NOTE in capsys.readouterr().out
    assert read_spy.calls == 2
    assert read_spy.last_files == files_b
    pd.testing.assert_frame_equal(table, read_all_files(files_b))


def test_switching_folders_returns_the_new_folders_table(tmp_path, cache_folder, read_spy,
                                                         capsys):
    folder_a, folder_b = tmp_path / 'a', tmp_path / 'b'
    folder_a.mkdir()
    folder_b.mkdir()
    make_day(folder_a, date(2025, 6, 2), ('0000001', 10.0))
    make_day(folder_b, date(2025, 6, 2), ('0000009', 90.0))
    load_collections(list_daily_files(folder_a), cache_folder)
    capsys.readouterr()
    table = load_collections(list_daily_files(folder_b), cache_folder)
    assert CHANGED_NOTE in capsys.readouterr().out
    assert read_spy.calls == 2
    assert list(table['customer']) == ['0000009']
    assert list(table['amount_try']) == [90.0]


def test_switching_folders_rewrites_the_fingerprint_for_the_new_folder(tmp_path, cache_folder):
    folder_a, folder_b = tmp_path / 'a', tmp_path / 'b'
    folder_a.mkdir()
    two_days(folder_a)
    copy_folder_with_same_mtimes(folder_a, folder_b)
    load_collections(list_daily_files(folder_a), cache_folder)
    files_b = list_daily_files(folder_b)
    load_collections(files_b, cache_folder)
    written = json.loads((cache_folder / FINGERPRINT_FILE).read_text(encoding='utf-8'))
    assert written == files_fingerprint(files_b)


# --- a damaged cache: read again, never raise -------------------------------------------

def assert_damaged_note(output):
    """The damaged-cache note is printed (with the error in brackets), not the changed note."""
    assert DAMAGED_NOTE_START in output
    assert DAMAGED_NOTE_END in output
    assert output.index(DAMAGED_NOTE_START) < output.index(DAMAGED_NOTE_END)
    assert CHANGED_NOTE not in output


def damage_then_reload(days_folder, cache_folder, damage, read_spy, capsys, monkeypatch):
    """Build the cache, damage it, reload; then check the cache was repaired.

    Returns the first table and the table loaded after the damage.
    """
    daily_files = two_days(days_folder)
    first = load_collections(daily_files, cache_folder)
    capsys.readouterr()
    damage()
    reloaded = load_collections(daily_files, cache_folder)
    assert_damaged_note(capsys.readouterr().out)
    assert read_spy.calls == 2
    # The cache was rewritten: the next call is a clean hit, no note, no reading.
    written = json.loads((cache_folder / FINGERPRINT_FILE).read_text(encoding='utf-8'))
    assert written == files_fingerprint(daily_files)
    forbid_reading(monkeypatch)
    pd.testing.assert_frame_equal(load_collections(daily_files, cache_folder), first)
    assert capsys.readouterr().out == ''
    return first, reloaded


@pytest.mark.parametrize('garbage', ['', 'not json at all', '[["C:\\\\days\\\\0206', '{', '\x00\x01'],
                         ids=['empty', 'text', 'truncated', 'open-brace', 'binary'])
def test_garbage_fingerprint_file_reads_again_and_repairs_the_cache(
        days_folder, cache_folder, read_spy, capsys, monkeypatch, garbage):
    def damage():
        (cache_folder / FINGERPRINT_FILE).write_text(garbage, encoding='utf-8')
    first, reloaded = damage_then_reload(days_folder, cache_folder, damage, read_spy, capsys,
                                         monkeypatch)
    pd.testing.assert_frame_equal(reloaded, first)


def test_fingerprint_file_cut_in_half_reads_again_and_repairs_the_cache(
        days_folder, cache_folder, read_spy, capsys, monkeypatch):
    def damage():
        text = (cache_folder / FINGERPRINT_FILE).read_text(encoding='utf-8')
        (cache_folder / FINGERPRINT_FILE).write_text(text[:len(text) // 2], encoding='utf-8')
    first, reloaded = damage_then_reload(days_folder, cache_folder, damage, read_spy, capsys,
                                         monkeypatch)
    pd.testing.assert_frame_equal(reloaded, first)


@pytest.mark.parametrize('garbage', [b'', b'not a pickle', b'\x80\x05\x95', bytes(range(256))],
                         ids=['empty', 'text', 'truncated-header', 'all-bytes'])
def test_garbage_pickle_with_matching_fingerprint_reads_again_and_repairs_the_cache(
        days_folder, cache_folder, read_spy, capsys, monkeypatch, garbage):
    def damage():
        (cache_folder / CACHE_FILE).write_bytes(garbage)
    first, reloaded = damage_then_reload(days_folder, cache_folder, damage, read_spy, capsys,
                                         monkeypatch)
    pd.testing.assert_frame_equal(reloaded, first)


def test_pickle_cut_in_half_reads_again_and_repairs_the_cache(
        days_folder, cache_folder, read_spy, capsys, monkeypatch):
    def damage():
        data = (cache_folder / CACHE_FILE).read_bytes()
        (cache_folder / CACHE_FILE).write_bytes(data[:len(data) // 2])
    first, reloaded = damage_then_reload(days_folder, cache_folder, damage, read_spy, capsys,
                                         monkeypatch)
    pd.testing.assert_frame_equal(reloaded, first)


def test_valid_json_of_the_wrong_shape_counts_as_changed(days_folder, cache_folder, read_spy,
                                                         capsys):
    daily_files = two_days(days_folder)
    first = load_collections(daily_files, cache_folder)
    capsys.readouterr()
    (cache_folder / FINGERPRINT_FILE).write_text('{"a": 1}', encoding='utf-8')
    pd.testing.assert_frame_equal(load_collections(daily_files, cache_folder), first)
    assert CHANGED_NOTE in capsys.readouterr().out
    assert read_spy.calls == 2


# --- read_cache on its own -------------------------------------------------------------

@pytest.fixture
def written_cache(days_folder, cache_folder):
    """A cache written by `load_collections`: (cache file, fingerprint file, fingerprint, table)."""
    daily_files = two_days(days_folder)
    table = load_collections(daily_files, cache_folder)
    return (cache_folder / CACHE_FILE, cache_folder / FINGERPRINT_FILE,
            files_fingerprint(daily_files), table)


def test_read_cache_returns_the_table_when_the_fingerprint_matches(written_cache, capsys):
    cache_file, fingerprint_file, fingerprint, table = written_cache
    capsys.readouterr()
    pd.testing.assert_frame_equal(read_cache(cache_file, fingerprint_file, fingerprint), table)
    assert capsys.readouterr().out == ''


def test_read_cache_returns_none_with_the_changed_note_when_the_fingerprint_differs(
        written_cache, capsys):
    cache_file, fingerprint_file, fingerprint, _ = written_cache
    capsys.readouterr()
    different = [[entry[0], entry[1] + 1, entry[2]] for entry in fingerprint]
    assert read_cache(cache_file, fingerprint_file, different) is None
    output = capsys.readouterr().out
    assert CHANGED_NOTE in output
    assert DAMAGED_NOTE_START not in output


def test_read_cache_returns_none_when_the_fingerprint_has_one_file_less(written_cache, capsys):
    cache_file, fingerprint_file, fingerprint, _ = written_cache
    capsys.readouterr()
    assert read_cache(cache_file, fingerprint_file, fingerprint[:-1]) is None
    assert CHANGED_NOTE in capsys.readouterr().out


def test_read_cache_returns_none_silently_without_the_cache_file(written_cache, capsys):
    cache_file, fingerprint_file, fingerprint, _ = written_cache
    cache_file.unlink()
    capsys.readouterr()
    assert read_cache(cache_file, fingerprint_file, fingerprint) is None
    assert capsys.readouterr().out == ''


def test_read_cache_returns_none_silently_without_the_fingerprint_file(written_cache, capsys):
    cache_file, fingerprint_file, fingerprint, _ = written_cache
    fingerprint_file.unlink()
    capsys.readouterr()
    assert read_cache(cache_file, fingerprint_file, fingerprint) is None
    assert capsys.readouterr().out == ''


def test_read_cache_returns_none_silently_without_either_file(tmp_path, capsys):
    assert read_cache(tmp_path / CACHE_FILE, tmp_path / FINGERPRINT_FILE, []) is None
    assert capsys.readouterr().out == ''


def test_read_cache_returns_none_with_the_damaged_note_for_a_garbage_pickle(written_cache,
                                                                            capsys):
    cache_file, fingerprint_file, fingerprint, _ = written_cache
    cache_file.write_bytes(b'not a pickle')
    capsys.readouterr()
    assert read_cache(cache_file, fingerprint_file, fingerprint) is None
    assert_damaged_note(capsys.readouterr().out)


def test_read_cache_returns_none_with_the_damaged_note_for_garbage_json(written_cache, capsys):
    cache_file, fingerprint_file, fingerprint, _ = written_cache
    fingerprint_file.write_text('[[', encoding='utf-8')
    capsys.readouterr()
    assert read_cache(cache_file, fingerprint_file, fingerprint) is None
    assert_damaged_note(capsys.readouterr().out)


def test_read_cache_does_not_change_the_cache_files(written_cache):
    cache_file, fingerprint_file, fingerprint, _ = written_cache
    before = (cache_file.read_bytes(), fingerprint_file.read_bytes())
    read_cache(cache_file, fingerprint_file, fingerprint)
    read_cache(cache_file, fingerprint_file, [])
    assert (cache_file.read_bytes(), fingerprint_file.read_bytes()) == before


# --- errors ----------------------------------------------------------------------------

def test_empty_dict_raises_and_writes_no_cache(cache_folder):
    with pytest.raises(ValueError, match='no daily files'):
        load_collections({}, cache_folder)
    assert not (cache_folder / CACHE_FILE).exists()
    assert not (cache_folder / FINGERPRINT_FILE).exists()


def test_read_error_propagates(days_folder, cache_folder):
    make_day(days_folder, date(2025, 6, 2), ('0000001', 10.0))
    make_day(days_folder, date(2025, 6, 3), ('0000002', 20.0), grand_total=99.0)
    with pytest.raises(ValueError, match=r'03062025 EY TAHSILAT\.XLSX.*99\.00 TRY'):
        load_collections(list_daily_files(days_folder), cache_folder)


def test_read_error_leaves_no_cache_files(days_folder, cache_folder):
    make_day(days_folder, date(2025, 6, 2), ('0000001', 10.0))
    make_day(days_folder, date(2025, 6, 3), ('0000002', 20.0), grand_total=99.0)
    with pytest.raises(ValueError):
        load_collections(list_daily_files(days_folder), cache_folder)
    assert not (cache_folder / CACHE_FILE).exists()
    assert not (cache_folder / FINGERPRINT_FILE).exists()


def test_read_error_after_a_change_does_not_serve_the_old_cache(days_folder, cache_folder):
    daily_files = two_days(days_folder)
    load_collections(daily_files, cache_folder)
    make_day(days_folder, date(2025, 6, 3), ('0000003', 30.0), grand_total=99.0)
    with pytest.raises(ValueError, match='99.00 TRY'):
        load_collections(daily_files, cache_folder)
    # The next call must still see the change and fail again, not return the stale table.
    with pytest.raises(ValueError, match='99.00 TRY'):
        load_collections(daily_files, cache_folder)


# --- the synthetic corpus --------------------------------------------------------------

@pytest.mark.skipif(not SYNTHETIC_CORPUS.is_dir(), reason='synthetic corpus not installed')
def test_synthetic_corpus_all_files_cached_and_reloaded_give_27311_lines(tmp_path,
                                                                         monkeypatch):
    """SLOW (reads all 306 workbooks once). The cache goes to tmp_path, not data/interim/."""
    daily_files = list_daily_files(SYNTHETIC_CORPUS)
    first = load_collections(daily_files, tmp_path / 'cache')
    forbid_reading(monkeypatch)
    second = load_collections(daily_files, tmp_path / 'cache')
    assert len(first) == 27_311
    pd.testing.assert_frame_equal(second, first)
