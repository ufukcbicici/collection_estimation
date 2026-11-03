"""Tests for `hurdle/cache.py` on its own: `load_with_cache`, `read_cache`,
`file_fingerprint` and `source_fingerprint`.

Tables are tiny DataFrames; `read` is a counting stand-in. Every cache folder is under
`tmp_path`, never the repo's `data/interim/`.
"""
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from collection_estimation.hurdle.cache import (
    file_fingerprint,
    load_with_cache,
    read_cache,
    source_fingerprint,
)

NAME = 'tiny'
FINGERPRINT = [[['C:/x/a.xlsx', 123, 1727000000.123456]], ['ab' * 32]]


def changed_note(name=NAME):
    return f"{name}: the inputs have changed since the cache was written; reading them again."


def damaged_start(name=NAME):
    return f"{name}: the cache could not be loaded ("


DAMAGED_END = "); reading the inputs again."


def hit_note(cache_file):
    """The one line read_cache prints on a hit (with its newline)."""
    return f"{cache_file.stem}: read from the cache; delete {cache_file} to make it again.\n"


def tiny_table():
    # dtype=object explicitly: pandas 3 would otherwise infer its `str` dtype and turn the
    # None into NaN before the cache ever sees it (the readers build object columns).
    return pd.DataFrame({
        'customer': pd.Series(['0012345', '0000002', None], dtype=object),
        'amount': [10.5, -2.25, 0.0],
        'row': pd.Series([1, 2, 3], dtype='int64'),
        'when': pd.to_datetime(['2025-06-02', '2025-06-03', '2025-06-04']),
    })


class CountingRead:
    """A zero-argument `read`: counts calls and returns a fresh copy of `table`."""

    def __init__(self, table=None):
        self.table = tiny_table() if table is None else table
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.table.copy()


def fail_read():
    raise AssertionError('read() ran; the cache should have been used')


@pytest.fixture
def cache_folder(tmp_path):
    return tmp_path / 'cache'


def files(folder, name=NAME):
    return folder / f'{name}.pkl', folder / f'{name}_fingerprint.json'


# --- load_with_cache: first call ------------------------------------------------------

def test_first_call_runs_read_once(cache_folder):
    read = CountingRead()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    assert read.calls == 1


def test_first_call_returns_what_read_built(cache_folder):
    table = load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    pd.testing.assert_frame_equal(table, tiny_table())


def test_first_call_writes_both_files_named_after_the_name(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    cache_file, fingerprint_file = files(cache_folder)
    assert cache_file.is_file()
    assert fingerprint_file.is_file()
    assert sorted(p.name for p in cache_folder.iterdir()) == ['tiny.pkl',
                                                               'tiny_fingerprint.json']


def test_first_call_creates_a_nested_folder(tmp_path):
    nested = tmp_path / 'a' / 'b' / 'c'
    load_with_cache(NAME, FINGERPRINT, CountingRead(), nested)
    assert all(path.is_file() for path in files(nested))


def test_first_call_accepts_the_folder_as_a_string(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), str(cache_folder))
    assert all(path.is_file() for path in files(cache_folder))


def test_first_call_writes_the_fingerprint_as_json(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    _, fingerprint_file = files(cache_folder)
    assert json.loads(fingerprint_file.read_text(encoding='utf-8')) == FINGERPRINT


def test_first_call_prints_nothing(cache_folder, capsys):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    assert capsys.readouterr().out == ''


# --- load_with_cache: same fingerprint -----------------------------------------------

def test_second_call_does_not_run_read(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)


def test_second_call_returns_an_equal_frame(cache_folder):
    first = load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    second = load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    pd.testing.assert_frame_equal(second, first)


def test_second_call_keeps_the_dtypes(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    second = load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    assert dict(second.dtypes) == dict(tiny_table().dtypes)


def test_second_call_keeps_none_in_object_columns(cache_folder):
    assert tiny_table()['customer'].iloc[2] is None      # the input really holds None
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    second = load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    assert second['customer'].dtype == object
    assert second['customer'].iloc[2] is None


def test_second_call_keeps_leading_zero_text(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    second = load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    assert list(second['customer'].iloc[:2]) == ['0012345', '0000002']


def test_second_call_prints_the_hit_note_only(cache_folder, capsys):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    capsys.readouterr()
    load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    cache_file, _ = files(cache_folder)
    assert capsys.readouterr().out == hit_note(cache_file)


def test_second_call_does_not_rewrite_the_files(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    before = [path.read_bytes() for path in files(cache_folder)]
    load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    assert [path.read_bytes() for path in files(cache_folder)] == before


def test_empty_table_round_trips(cache_folder):
    empty = pd.DataFrame({'customer': pd.Series([], dtype=object),
                          'amount': pd.Series([], dtype='float64')})
    load_with_cache(NAME, FINGERPRINT, CountingRead(empty), cache_folder)
    pd.testing.assert_frame_equal(load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder),
                                  empty)


# --- load_with_cache: different fingerprint ------------------------------------------

def other_fingerprint():
    return [[['C:/x/a.xlsx', 124, 1727000000.123456]], ['ab' * 32]]


def test_different_fingerprint_runs_read_again(cache_folder):
    read = CountingRead()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    load_with_cache(NAME, other_fingerprint(), read, cache_folder)
    assert read.calls == 2


def test_different_fingerprint_prints_the_note_with_the_name(cache_folder, capsys):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    capsys.readouterr()
    load_with_cache(NAME, other_fingerprint(), CountingRead(), cache_folder)
    output = capsys.readouterr().out
    assert changed_note() in output
    assert damaged_start() not in output


def test_different_fingerprint_rewrites_the_fingerprint(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    load_with_cache(NAME, other_fingerprint(), CountingRead(), cache_folder)
    _, fingerprint_file = files(cache_folder)
    assert json.loads(fingerprint_file.read_text(encoding='utf-8')) == other_fingerprint()


def test_different_fingerprint_returns_and_caches_the_new_table(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    new = pd.DataFrame({'customer': ['0000009'], 'amount': [90.0]})
    got = load_with_cache(NAME, other_fingerprint(), CountingRead(new), cache_folder)
    pd.testing.assert_frame_equal(got, new)
    pd.testing.assert_frame_equal(
        load_with_cache(NAME, other_fingerprint(), fail_read, cache_folder), new)


def test_going_back_to_the_old_fingerprint_reads_again(cache_folder):
    read = CountingRead()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    load_with_cache(NAME, other_fingerprint(), read, cache_folder)
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    assert read.calls == 3


@pytest.mark.parametrize('changed', [
    [[['C:/x/b.xlsx', 123, 1727000000.123456]], ['ab' * 32]],     # path
    [[['C:/x/a.xlsx', 123, 1727000000.123457]], ['ab' * 32]],     # mtime, last digit
    [[['C:/x/a.xlsx', 123, 1727000000.123456]], ['cd' * 32]],     # source hash
    [[['C:/x/a.xlsx', 123, 1727000000.123456]], []],              # source hash dropped
    [[], ['ab' * 32]],                                            # no files
    [[['C:/x/a.xlsx', 123, 1727000000.123456]]],                  # one part only
    [['ab' * 32], [['C:/x/a.xlsx', 123, 1727000000.123456]]],     # parts swapped
], ids=['path', 'mtime', 'source', 'no-source', 'no-files', 'one-part', 'swapped'])
def test_any_change_in_the_fingerprint_runs_read_again(cache_folder, changed):
    read = CountingRead()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    load_with_cache(NAME, changed, read, cache_folder)
    assert read.calls == 2


# --- fingerprints survive the JSON round trip ---------------------------------------

@pytest.mark.parametrize('fingerprint', [
    [],
    [[]],
    ['text', 'çğıöşü İ'],
    [1, -2, 0, 2 ** 53],
    [0.1, 1e-300, 1727000000.1234567, 1.0],
    [[['C:\\days\\02062025 EY TAHSILAT.XLSX', 9876, 1749000000.987654]], ['0' * 64]],
    [[[[[1]]]], ['a', [2.5, ['b']]]],
], ids=['empty', 'empty-nested', 'strings', 'ints', 'floats', 'collections-shape', 'deep'])
def test_fingerprint_matches_after_the_json_round_trip(cache_folder, fingerprint):
    load_with_cache(NAME, fingerprint, CountingRead(), cache_folder)
    load_with_cache(NAME, fingerprint, fail_read, cache_folder)


def test_real_file_fingerprint_matches_after_the_json_round_trip(tmp_path, cache_folder):
    source = tmp_path / 'input.txt'
    source.write_text('hello', encoding='utf-8')
    fingerprint = [[file_fingerprint(source)], source_fingerprint(source)]
    load_with_cache(NAME, fingerprint, CountingRead(), cache_folder)
    load_with_cache(NAME, [[file_fingerprint(source)], source_fingerprint(source)], fail_read,
                    cache_folder)


@pytest.mark.parametrize('fingerprint', [
    ('a', 1),
    [('C:/x/a.xlsx', 123, 1727000000.123456), ('ab' * 32,)],
    (('a', ('b', (1.5, ()))), ['c', ('d',)]),
], ids=['flat-tuple', 'tuples-in-a-list', 'nested-tuples'])
def test_a_tuple_fingerprint_hits_on_the_second_call(cache_folder, capsys, fingerprint):
    """read_cache compares in JSON form, so tuples (saved as lists) still match."""
    read = CountingRead()
    load_with_cache(NAME, fingerprint, read, cache_folder)
    capsys.readouterr()
    table = load_with_cache(NAME, fingerprint, fail_read, cache_folder)
    assert read.calls == 1
    assert capsys.readouterr().out == hit_note(files(cache_folder)[0])
    pd.testing.assert_frame_equal(table, tiny_table())


def test_a_tuple_fingerprint_matches_the_same_fingerprint_as_lists(cache_folder):
    load_with_cache(NAME, ('a', ('b', 1)), CountingRead(), cache_folder)
    load_with_cache(NAME, ['a', ['b', 1]], fail_read, cache_folder)


def test_a_tuple_fingerprint_with_a_different_value_still_reads_again(cache_folder, capsys):
    read = CountingRead()
    load_with_cache(NAME, ('a', 1), read, cache_folder)
    capsys.readouterr()
    load_with_cache(NAME, ('a', 2), read, cache_folder)
    assert read.calls == 2
    assert changed_note() in capsys.readouterr().out


# --- a fingerprint JSON cannot encode: fails before anything is read or written ------

UNENCODABLE = [['C:/x/a.xlsx', 123, 1727000000.123456], date(2026, 8, 28)]


def new_table():
    return pd.DataFrame({'customer': pd.Series(['0000009'], dtype=object), 'amount': [90.0]})


def test_unencodable_fingerprint_raises_type_error_without_calling_read(cache_folder):
    read = CountingRead()
    with pytest.raises(TypeError, match='not JSON serializable'):
        load_with_cache(NAME, UNENCODABLE, read, cache_folder)
    assert read.calls == 0


def test_unencodable_fingerprint_first_call_creates_no_folder_and_no_files(cache_folder):
    with pytest.raises(TypeError):
        load_with_cache(NAME, UNENCODABLE, CountingRead(), cache_folder)
    assert not cache_folder.exists()


def test_unencodable_fingerprint_first_call_into_an_existing_empty_folder_writes_nothing(
        cache_folder):
    cache_folder.mkdir()
    with pytest.raises(TypeError):
        load_with_cache(NAME, UNENCODABLE, CountingRead(), cache_folder)
    assert list(cache_folder.iterdir()) == []


def test_unencodable_fingerprint_first_call_prints_nothing(cache_folder, capsys):
    with pytest.raises(TypeError):
        load_with_cache(NAME, UNENCODABLE, CountingRead(), cache_folder)
    assert capsys.readouterr().out == ''


def test_after_an_unencodable_first_call_a_valid_call_reads_and_caches(cache_folder, capsys):
    with pytest.raises(TypeError):
        load_with_cache(NAME, UNENCODABLE, CountingRead(), cache_folder)
    capsys.readouterr()
    read = CountingRead()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    assert read.calls == 1
    assert capsys.readouterr().out == ''
    assert all(path.is_file() for path in files(cache_folder))


def test_unencodable_fingerprint_over_an_existing_cache_does_not_call_read(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    read = CountingRead(new_table())
    with pytest.raises(TypeError, match='not JSON serializable'):
        load_with_cache(NAME, UNENCODABLE, read, cache_folder)
    assert read.calls == 0


def test_unencodable_fingerprint_over_an_existing_cache_prints_nothing(cache_folder, capsys):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    capsys.readouterr()
    with pytest.raises(TypeError):
        load_with_cache(NAME, UNENCODABLE, CountingRead(new_table()), cache_folder)
    assert capsys.readouterr().out == ''


def test_unencodable_fingerprint_over_an_existing_cache_leaves_both_files_unchanged(
        cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    before = {path.name: path.read_bytes() for path in cache_folder.iterdir()}
    with pytest.raises(TypeError):
        load_with_cache(NAME, UNENCODABLE, CountingRead(new_table()), cache_folder)
    after = {path.name: path.read_bytes() for path in cache_folder.iterdir()}
    assert after == before
    assert sorted(after) == ['tiny.pkl', 'tiny_fingerprint.json']


def test_unencodable_fingerprint_over_an_existing_cache_then_old_fingerprint_serves_old_table(
        cache_folder, capsys):
    """The scenario of the fixed bug: the old fingerprint must still serve the OLD table."""
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    with pytest.raises(TypeError):
        load_with_cache(NAME, UNENCODABLE, CountingRead(new_table()), cache_folder)
    capsys.readouterr()
    served = load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    pd.testing.assert_frame_equal(served, tiny_table())
    assert capsys.readouterr().out == hit_note(files(cache_folder)[0])


def test_read_cache_with_an_unencodable_fingerprint_returns_none_with_the_damaged_note(
        tmp_path, capsys):
    cache_file, fingerprint_file = tmp_path / 't.pkl', tmp_path / 't_fingerprint.json'
    tiny_table().to_pickle(cache_file)
    fingerprint_file.write_text(json.dumps(FINGERPRINT), encoding='utf-8')
    assert read_cache(cache_file, fingerprint_file, UNENCODABLE) is None
    output = capsys.readouterr().out
    assert output.startswith('t: the cache could not be loaded (TypeError(')
    assert DAMAGED_END in output


# --- damaged or missing cache files -------------------------------------------------

def build_then_damage(cache_folder, damage):
    read = CountingRead()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    damage(*files(cache_folder))
    return read


@pytest.mark.parametrize('garbage', [b'', b'not a pickle', b'\x80\x05\x95', bytes(range(256))],
                         ids=['empty', 'text', 'truncated-header', 'all-bytes'])
def test_damaged_pickle_runs_read_again_with_the_damaged_note(cache_folder, capsys, garbage):
    read = build_then_damage(cache_folder, lambda pkl, _: pkl.write_bytes(garbage))
    capsys.readouterr()
    table = load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    output = capsys.readouterr().out
    assert read.calls == 2
    assert damaged_start() in output and DAMAGED_END in output
    assert changed_note() not in output
    pd.testing.assert_frame_equal(table, tiny_table())


def test_pickle_cut_in_half_runs_read_again(cache_folder):
    def damage(pkl, _):
        data = pkl.read_bytes()
        pkl.write_bytes(data[:len(data) // 2])
    read = build_then_damage(cache_folder, damage)
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    assert read.calls == 2


@pytest.mark.parametrize('garbage', ['', 'not json', '[[', '{', '\x00\x01'],
                         ids=['empty', 'text', 'truncated', 'open-brace', 'binary'])
def test_damaged_json_runs_read_again_with_the_damaged_note(cache_folder, capsys, garbage):
    read = build_then_damage(cache_folder,
                             lambda _, js: js.write_text(garbage, encoding='utf-8'))
    capsys.readouterr()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    output = capsys.readouterr().out
    assert read.calls == 2
    assert damaged_start() in output and DAMAGED_END in output


def test_damaged_cache_is_repaired_and_the_next_call_hits(cache_folder, capsys):
    read = build_then_damage(cache_folder, lambda pkl, _: pkl.write_bytes(b'junk'))
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    capsys.readouterr()
    pd.testing.assert_frame_equal(load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder),
                                  tiny_table())
    assert capsys.readouterr().out == hit_note(files(cache_folder)[0])


@pytest.mark.parametrize('which', ['pickle', 'fingerprint'])
def test_one_file_missing_runs_read_again_silently(cache_folder, capsys, which):
    def damage(pkl, js):
        (pkl if which == 'pickle' else js).unlink()
    read = build_then_damage(cache_folder, damage)
    capsys.readouterr()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    assert read.calls == 2
    assert capsys.readouterr().out == ''
    assert all(path.is_file() for path in files(cache_folder))


def test_valid_json_of_the_wrong_shape_counts_as_changed(cache_folder, capsys):
    read = build_then_damage(cache_folder,
                             lambda _, js: js.write_text('{"a": 1}', encoding='utf-8'))
    capsys.readouterr()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    assert read.calls == 2
    assert changed_note() in capsys.readouterr().out


# --- read() raising -----------------------------------------------------------------

def boom():
    raise ValueError('the inputs are broken')


def test_read_error_propagates(cache_folder):
    with pytest.raises(ValueError, match='the inputs are broken'):
        load_with_cache(NAME, FINGERPRINT, boom, cache_folder)


def test_read_error_leaves_no_cache_files(cache_folder):
    with pytest.raises(ValueError):
        load_with_cache(NAME, FINGERPRINT, boom, cache_folder)
    assert not any(path.exists() for path in files(cache_folder))


def test_read_error_after_a_change_keeps_failing_and_never_serves_the_old_table(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    for _ in range(2):
        with pytest.raises(ValueError):
            load_with_cache(NAME, other_fingerprint(), boom, cache_folder)
    # The old cache is still there and still valid for the old fingerprint.
    pd.testing.assert_frame_equal(load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder),
                                  tiny_table())


# --- writing the cache --------------------------------------------------------------

def test_first_call_writes_exactly_the_encoded_fingerprint_text(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    _, fingerprint_file = files(cache_folder)
    assert fingerprint_file.read_text(encoding='utf-8') == json.dumps(FINGERPRINT, indent=1)


def test_reread_rewrites_the_fingerprint_with_exactly_the_encoded_text(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    load_with_cache(NAME, other_fingerprint(), CountingRead(), cache_folder)
    _, fingerprint_file = files(cache_folder)
    assert fingerprint_file.read_text(encoding='utf-8') == json.dumps(other_fingerprint(),
                                                                      indent=1)


def test_reread_of_a_tuple_fingerprint_writes_its_list_encoding(cache_folder):
    fingerprint = ('a', ('b', 1.5))
    load_with_cache(NAME, fingerprint, CountingRead(), cache_folder)
    _, fingerprint_file = files(cache_folder)
    assert fingerprint_file.read_text(encoding='utf-8') == json.dumps(['a', ['b', 1.5]],
                                                                      indent=1)


def test_reread_after_damage_rewrites_the_fingerprint_text(cache_folder):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    _, fingerprint_file = files(cache_folder)
    fingerprint_file.write_text('not json', encoding='utf-8')
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    assert fingerprint_file.read_text(encoding='utf-8') == json.dumps(FINGERPRINT, indent=1)


class PickleWriteFailed(OSError):
    pass


def failing_to_pickle(partial: bytes | None):
    """A stand-in for pd.to_pickle(obj, path): optionally writes `partial` bytes, then raises.
    load_with_cache writes with pd.to_pickle, so that it can cache any picklable result."""
    def to_pickle(obj, path, *args, **kwargs):
        if partial is not None:
            Path(path).write_bytes(partial)
        raise PickleWriteFailed('disk full')
    return to_pickle


@pytest.mark.parametrize('partial', [None, b'\x80\x05\x95half'], ids=['untouched', 'partial'])
def test_pickle_write_failure_propagates(cache_folder, monkeypatch, partial):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    monkeypatch.setattr(pd, 'to_pickle', failing_to_pickle(partial))
    with pytest.raises(PickleWriteFailed, match='disk full'):
        load_with_cache(NAME, other_fingerprint(), CountingRead(new_table()), cache_folder)


@pytest.mark.parametrize('partial', [None, b'\x80\x05\x95half'], ids=['untouched', 'partial'])
def test_pickle_write_failure_removes_the_old_fingerprint_file(cache_folder, monkeypatch,
                                                               partial):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    monkeypatch.setattr(pd, 'to_pickle', failing_to_pickle(partial))
    with pytest.raises(PickleWriteFailed):
        load_with_cache(NAME, other_fingerprint(), CountingRead(new_table()), cache_folder)
    cache_file, fingerprint_file = files(cache_folder)
    assert not fingerprint_file.exists()
    assert cache_file.exists()      # the stale (or partial) pickle is left behind


@pytest.mark.parametrize('fingerprint_next', ['old', 'new'])
@pytest.mark.parametrize('partial', [None, b'\x80\x05\x95half'], ids=['untouched', 'partial'])
def test_after_a_pickle_write_failure_the_next_call_reads_again(cache_folder, monkeypatch,
                                                                capsys, partial,
                                                                fingerprint_next):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    with monkeypatch.context() as patch:
        patch.setattr(pd, 'to_pickle', failing_to_pickle(partial))
        with pytest.raises(PickleWriteFailed):
            load_with_cache(NAME, other_fingerprint(), CountingRead(new_table()),
                            cache_folder)
    capsys.readouterr()
    fingerprint = FINGERPRINT if fingerprint_next == 'old' else other_fingerprint()
    fresh = pd.DataFrame({'customer': pd.Series(['0000042'], dtype=object), 'amount': [4.2]})
    read = CountingRead(fresh)
    served = load_with_cache(NAME, fingerprint, read, cache_folder)
    assert read.calls == 1
    pd.testing.assert_frame_equal(served, fresh)
    assert capsys.readouterr().out == ''      # a missing fingerprint file is silent
    # ... and the repaired cache now hits.
    pd.testing.assert_frame_equal(load_with_cache(NAME, fingerprint, fail_read, cache_folder),
                                  fresh)


def test_fingerprint_write_failure_leaves_no_fingerprint_so_the_next_call_reads_again(
        cache_folder, monkeypatch):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    _, fingerprint_file = files(cache_folder)
    real_write_text = Path.write_text

    def failing_write_text(self, *args, **kwargs):
        if self == fingerprint_file:
            raise PickleWriteFailed('disk full')
        return real_write_text(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'write_text', failing_write_text)
        with pytest.raises(PickleWriteFailed):
            load_with_cache(NAME, other_fingerprint(), CountingRead(new_table()),
                            cache_folder)
    assert not fingerprint_file.exists()
    read = CountingRead()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    assert read.calls == 1


# --- two names in one folder ---------------------------------------------------------

def test_two_names_in_one_folder_do_not_interfere(cache_folder):
    table_a = pd.DataFrame({'x': [1, 2]})
    table_b = pd.DataFrame({'y': ['p', 'q', 'r']})
    load_with_cache('alpha', FINGERPRINT, CountingRead(table_a), cache_folder)
    load_with_cache('beta', other_fingerprint(), CountingRead(table_b), cache_folder)
    pd.testing.assert_frame_equal(load_with_cache('alpha', FINGERPRINT, fail_read, cache_folder),
                                  table_a)
    pd.testing.assert_frame_equal(
        load_with_cache('beta', other_fingerprint(), fail_read, cache_folder), table_b)
    assert len(list(cache_folder.iterdir())) == 4


def test_changing_one_name_leaves_the_other_cached(cache_folder, capsys):
    load_with_cache('alpha', FINGERPRINT, CountingRead(), cache_folder)
    load_with_cache('beta', FINGERPRINT, CountingRead(), cache_folder)
    capsys.readouterr()
    load_with_cache('alpha', other_fingerprint(), CountingRead(), cache_folder)
    output = capsys.readouterr().out
    assert changed_note('alpha') in output
    assert 'beta' not in output
    load_with_cache('beta', FINGERPRINT, fail_read, cache_folder)


# --- read_cache on its own ----------------------------------------------------------

def test_read_cache_none_silently_when_nothing_exists(tmp_path, capsys):
    assert read_cache(tmp_path / 'x.pkl', tmp_path / 'x_fingerprint.json', []) is None
    assert capsys.readouterr().out == ''


def test_read_cache_note_uses_the_cache_file_stem(tmp_path, capsys):
    cache_file = tmp_path / 'some_table.pkl'
    fingerprint_file = tmp_path / 'whatever.json'
    tiny_table().to_pickle(cache_file)
    fingerprint_file.write_text('[1]', encoding='utf-8')
    capsys.readouterr()
    assert read_cache(cache_file, fingerprint_file, [2]) is None
    assert changed_note('some_table') in capsys.readouterr().out


def test_read_cache_returns_the_table_on_a_match(tmp_path):
    cache_file, fingerprint_file = tmp_path / 't.pkl', tmp_path / 't_fingerprint.json'
    tiny_table().to_pickle(cache_file)
    fingerprint_file.write_text(json.dumps(FINGERPRINT), encoding='utf-8')
    pd.testing.assert_frame_equal(read_cache(cache_file, fingerprint_file, FINGERPRINT),
                                  tiny_table())


# --- file_fingerprint ---------------------------------------------------------------

def test_file_fingerprint_matches_stat(tmp_path):
    path = tmp_path / 'a.bin'
    path.write_bytes(b'12345')
    stat = path.stat()
    assert file_fingerprint(path) == [str(path.resolve()), stat.st_size, stat.st_mtime]


def test_file_fingerprint_is_a_list_of_three(tmp_path):
    path = tmp_path / 'a.bin'
    path.write_bytes(b'1')
    fingerprint = file_fingerprint(path)
    assert isinstance(fingerprint, list) and len(fingerprint) == 3
    assert isinstance(fingerprint[0], str)
    assert isinstance(fingerprint[1], int)
    assert isinstance(fingerprint[2], float)


def test_file_fingerprint_is_absolute_from_a_relative_path(tmp_path, monkeypatch):
    path = tmp_path / 'a.bin'
    path.write_bytes(b'1')
    monkeypatch.chdir(tmp_path)
    fingerprint = file_fingerprint(Path('a.bin'))
    assert Path(fingerprint[0]).is_absolute()
    assert fingerprint[0] == str(path.resolve())


def test_file_fingerprint_accepts_a_string(tmp_path):
    path = tmp_path / 'a.bin'
    path.write_bytes(b'1')
    assert file_fingerprint(str(path)) == file_fingerprint(path)


def test_file_fingerprint_changes_with_the_size(tmp_path):
    path = tmp_path / 'a.bin'
    path.write_bytes(b'1')
    before = file_fingerprint(path)
    stat = path.stat()
    path.write_bytes(b'12')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = file_fingerprint(path)
    assert after[2] == before[2]
    assert after != before


def test_file_fingerprint_changes_with_the_mtime(tmp_path):
    path = tmp_path / 'a.bin'
    path.write_bytes(b'1')
    before = file_fingerprint(path)
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime - 3600))
    assert file_fingerprint(path) != before


def test_file_fingerprint_differs_for_the_same_file_in_another_folder(tmp_path):
    a, b = tmp_path / 'a' / 'f.bin', tmp_path / 'b' / 'f.bin'
    for path in (a, b):
        path.parent.mkdir()
        path.write_bytes(b'same')
    os.utime(b, ns=(a.stat().st_atime_ns, a.stat().st_mtime_ns))
    assert file_fingerprint(a)[1:] == file_fingerprint(b)[1:]
    assert file_fingerprint(a) != file_fingerprint(b)


def test_file_fingerprint_of_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_fingerprint(tmp_path / 'missing.bin')


# --- source_fingerprint -------------------------------------------------------------

def write(path, data: bytes):
    path.write_bytes(data)
    return path


def test_source_fingerprint_is_one_64_char_hex_digest_per_file_in_order(tmp_path):
    a = write(tmp_path / 'a.py', b'a = 1\n')
    b = write(tmp_path / 'b.py', b'b = 2\n')
    digests = source_fingerprint(b, a)
    assert len(digests) == 2
    for digest in digests:
        assert len(digest) == 64
        assert all(char in '0123456789abcdef' for char in digest)
    assert digests == [hashlib.sha256(b'b = 2\n').hexdigest(),
                       hashlib.sha256(b'a = 1\n').hexdigest()]


def test_source_fingerprint_equals_sha256_of_the_bytes(tmp_path):
    data = bytes(range(256)) * 3 + 'çğı\r\n'.encode('utf-8')
    path = write(tmp_path / 'x.py', data)
    assert source_fingerprint(path) == [hashlib.sha256(data).hexdigest()]


def test_source_fingerprint_same_content_same_digest_in_different_files(tmp_path):
    a = write(tmp_path / 'a.py', b'same\n')
    b = write(tmp_path / 'sub_b.py', b'same\n')
    first, second = source_fingerprint(a, b)
    assert first == second


def test_source_fingerprint_one_byte_changed_changes_the_digest(tmp_path):
    path = write(tmp_path / 'x.py', b'value = 1\n')
    before = source_fingerprint(path)
    write(path, b'value = 2\n')
    assert source_fingerprint(path) != before


def test_source_fingerprint_line_endings_count(tmp_path):
    lf = write(tmp_path / 'lf.py', b'x = 1\n')
    crlf = write(tmp_path / 'crlf.py', b'x = 1\r\n')
    assert source_fingerprint(lf) != source_fingerprint(crlf)


def test_source_fingerprint_ignores_the_mtime(tmp_path):
    path = write(tmp_path / 'x.py', b'x = 1\n')
    before = source_fingerprint(path)
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime - 3600))
    assert source_fingerprint(path) == before


def test_source_fingerprint_of_no_files_is_empty():
    assert source_fingerprint() == []


def test_source_fingerprint_of_an_empty_file(tmp_path):
    path = write(tmp_path / 'empty.py', b'')
    assert source_fingerprint(path) == [hashlib.sha256(b'').hexdigest()]


def test_source_fingerprint_accepts_strings(tmp_path):
    path = write(tmp_path / 'x.py', b'x\n')
    assert source_fingerprint(str(path)) == source_fingerprint(path)


def test_source_fingerprint_survives_a_json_round_trip(tmp_path):
    path = write(tmp_path / 'x.py', b'x\n')
    digests = source_fingerprint(path)
    assert json.loads(json.dumps(digests, indent=1)) == digests


def test_source_fingerprint_of_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        source_fingerprint(tmp_path / 'missing.py')


# --- results that are not one DataFrame (2026-11-03: pd.to_pickle) -------------------------
# load_with_cache saves with pd.to_pickle, so it caches any picklable result, e.g. step 17's
# forecast.Forecast (a frozen dataclass holding two DataFrames).

import subprocess  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from collection_estimation.hurdle.cache import table_fingerprint  # noqa: E402
from collection_estimation.hurdle.forecast import Forecast  # noqa: E402


@dataclass(frozen=True)
class TwoTables:
    """A dataclass of two tables, like Forecast (module level, so that it pickles)."""
    first: pd.DataFrame
    second: pd.DataFrame


def a_forecast():
    totals = pd.DataFrame({'company_code': pd.Series(['TR02', 'ALL'], dtype=object),
                           'origin_week': pd.Series([33, 33], dtype='int64'),
                           'total': [1.5e6, 2.25e6], 'fallback': [False, True]})
    customers = pd.DataFrame({'key': pd.Series(['0012345|TR02', None], dtype=object,
                                               index=[7, 3]),
                              'expected': pd.Series([10.0, 0.5], index=[7, 3])})
    return Forecast(totals=totals, customers=customers)


class CountingMake:
    """A zero-argument `read` for any result: counts calls, returns what `make()` builds."""

    def __init__(self, make):
        self.make = make
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.make()


def test_a_forecast_round_trips_through_the_cache(cache_folder):
    # 1. The first call builds the Forecast and caches it.
    make = CountingMake(a_forecast)
    first = load_with_cache(NAME, FINGERPRINT, make, cache_folder)
    # 2. The second call reads it back without building: a Forecast with equal tables.
    second = load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    assert make.calls == 1
    assert type(second) is Forecast
    pd.testing.assert_frame_equal(second.totals, first.totals)
    pd.testing.assert_frame_equal(second.customers, first.customers)


def test_a_cached_forecast_keeps_dtypes_index_and_none(cache_folder):
    load_with_cache(NAME, FINGERPRINT, a_forecast, cache_folder)
    served = load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    expected = a_forecast()
    assert dict(served.totals.dtypes) == dict(expected.totals.dtypes)
    assert list(served.customers.index) == [7, 3]
    assert served.customers['key'].iloc[1] is None


def test_the_first_call_returns_the_very_object_read_built(cache_folder):
    # On a miss, the result is returned as built (not a re-read copy).
    built = a_forecast()
    assert load_with_cache(NAME, FINGERPRINT, lambda: built, cache_folder) is built


def test_a_dataclass_of_two_tables_round_trips(cache_folder):
    original = TwoTables(first=tiny_table(), second=pd.DataFrame({'x': [1, 2, 3]}))
    load_with_cache(NAME, FINGERPRINT, lambda: original, cache_folder)
    served = load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    assert type(served) is TwoTables
    pd.testing.assert_frame_equal(served.first, original.first)
    pd.testing.assert_frame_equal(served.second, original.second)


@pytest.mark.parametrize('result', [
    {'a': pd.DataFrame({'x': [1.0]}), 'b': [1, 2]},
    [pd.DataFrame({'x': [1]}), pd.DataFrame({'y': ['p']})],
    pd.Series([1.0, 2.0], index=[10, 11], name='price_level'),
], ids=['dict', 'list-of-frames', 'series'])
def test_other_picklable_results_round_trip(cache_folder, result):
    load_with_cache(NAME, FINGERPRINT, lambda: result, cache_folder)
    served = load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    if isinstance(result, pd.Series):
        pd.testing.assert_series_equal(served, result)
    elif isinstance(result, dict):
        pd.testing.assert_frame_equal(served['a'], result['a'])
        assert served['b'] == result['b']
    else:
        for found, wanted in zip(served, result):
            pd.testing.assert_frame_equal(found, wanted)


def test_a_forecast_cache_is_replaced_when_the_fingerprint_changes(cache_folder):
    # 1. A Forecast cached under the first fingerprint.
    load_with_cache(NAME, FINGERPRINT, a_forecast, cache_folder)
    # 2. A changed fingerprint builds again: a plain table this time, served from then on.
    load_with_cache(NAME, other_fingerprint(), tiny_table, cache_folder)
    served = load_with_cache(NAME, other_fingerprint(), fail_read, cache_folder)
    pd.testing.assert_frame_equal(served, tiny_table())


def test_an_unpicklable_result_propagates_and_leaves_no_fingerprint(cache_folder):
    # A result pickle cannot write (a lambda inside) fails loudly; no fingerprint is left
    # beside whatever the pickle file holds, so the next call builds again.
    def unpicklable():
        return {'f': lambda: 1}
    with pytest.raises(Exception):
        load_with_cache(NAME, FINGERPRINT, unpicklable, cache_folder)
    _, fingerprint_file = files(cache_folder)
    assert not fingerprint_file.exists()
    read = CountingRead()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    assert read.calls == 1


# --- the note printed on a hit ----------------------------------------------------------------

def test_the_hit_note_exact_text(cache_folder, capsys):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    capsys.readouterr()
    load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    output = capsys.readouterr().out
    expected = (f"tiny: read from the cache; delete {cache_folder / 'tiny.pkl'} "
                f"to make it again.\n")
    assert output == expected


def test_the_hit_note_names_the_cache_and_its_pickle(cache_folder, capsys):
    load_with_cache('hurdle_rolling_forecasts', FINGERPRINT, a_forecast, cache_folder)
    capsys.readouterr()
    load_with_cache('hurdle_rolling_forecasts', FINGERPRINT, fail_read, cache_folder)
    pickle_file = cache_folder / 'hurdle_rolling_forecasts.pkl'
    assert capsys.readouterr().out == hit_note(pickle_file)
    assert pickle_file.is_file()           # the file the note says to delete exists


def test_the_hit_note_with_the_folder_given_as_a_string(cache_folder, capsys):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), str(cache_folder))
    capsys.readouterr()
    load_with_cache(NAME, FINGERPRINT, fail_read, str(cache_folder))
    assert capsys.readouterr().out == hit_note(cache_folder / 'tiny.pkl')


def test_the_hit_note_is_printed_once_per_hit(cache_folder, capsys):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    capsys.readouterr()
    for _ in range(3):
        load_with_cache(NAME, FINGERPRINT, fail_read, cache_folder)
    assert capsys.readouterr().out == hit_note(files(cache_folder)[0]) * 3


def test_no_hit_note_on_a_changed_fingerprint(cache_folder, capsys):
    # A miss prints the changed note only, then nothing about a hit.
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    capsys.readouterr()
    load_with_cache(NAME, other_fingerprint(), CountingRead(), cache_folder)
    output = capsys.readouterr().out
    assert output == changed_note() + '\n'
    assert 'read from the cache' not in output


def test_no_hit_note_on_a_damaged_pickle(cache_folder, capsys):
    read = build_then_damage(cache_folder, lambda pkl, _: pkl.write_bytes(b'junk'))
    capsys.readouterr()
    load_with_cache(NAME, FINGERPRINT, read, cache_folder)
    output = capsys.readouterr().out
    assert output.startswith(damaged_start()) and output.endswith(DAMAGED_END + '\n')
    assert output.count('\n') == 1
    assert 'read from the cache' not in output


def test_no_note_at_all_on_the_first_call_with_a_forecast(cache_folder, capsys):
    load_with_cache(NAME, FINGERPRINT, a_forecast, cache_folder)
    assert capsys.readouterr().out == ''


def test_after_a_change_the_next_identical_call_prints_the_hit_note(cache_folder, capsys):
    load_with_cache(NAME, FINGERPRINT, CountingRead(), cache_folder)
    load_with_cache(NAME, other_fingerprint(), CountingRead(), cache_folder)
    capsys.readouterr()
    load_with_cache(NAME, other_fingerprint(), fail_read, cache_folder)
    assert capsys.readouterr().out == hit_note(files(cache_folder)[0])


def test_read_cache_prints_the_hit_note_with_the_cache_file_stem(tmp_path, capsys):
    # The note's name is the pickle's stem, whatever the fingerprint file is called.
    cache_file = tmp_path / 'some_table.pkl'
    fingerprint_file = tmp_path / 'whatever.json'
    tiny_table().to_pickle(cache_file)
    fingerprint_file.write_text('[1]', encoding='utf-8')
    read_cache(cache_file, fingerprint_file, [1])
    assert capsys.readouterr().out == hit_note(cache_file)


def test_read_cache_prints_no_hit_note_on_a_mismatch(tmp_path, capsys):
    cache_file = tmp_path / 'some_table.pkl'
    fingerprint_file = tmp_path / 'some_table_fingerprint.json'
    tiny_table().to_pickle(cache_file)
    fingerprint_file.write_text('[1]', encoding='utf-8')
    read_cache(cache_file, fingerprint_file, [2])
    assert 'read from the cache' not in capsys.readouterr().out


# --- table_fingerprint ------------------------------------------------------------------------

def fingerprinted_table():
    return pd.DataFrame({
        'company_code': pd.Series(['TR02', 'TR04', 'TR02'], dtype=object),
        'origin_week': pd.Series([33, 33, 34], dtype='int64'),
        'amount': [10.5, -2.25, 0.0],
    })


def test_table_fingerprint_is_two_sha256_hex_digests():
    found = table_fingerprint(fingerprinted_table())
    assert isinstance(found, list) and len(found) == 2
    for digest in found:
        assert isinstance(digest, str) and len(digest) == 64
        assert all(char in '0123456789abcdef' for char in digest)


def test_table_fingerprint_by_hand():
    # The definition: [sha256 of the JSON of [[name, dtype], ...]; sha256 of the bytes of
    # pandas' row hashes, index included].
    table = fingerprinted_table()
    columns = json.dumps([['company_code', 'object'], ['origin_week', 'int64'],
                          ['amount', 'float64']])
    rows = pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes()
    assert table_fingerprint(table) == [hashlib.sha256(columns.encode('utf-8')).hexdigest(),
                                        hashlib.sha256(rows).hexdigest()]


def test_table_fingerprint_is_deterministic():
    table = fingerprinted_table()
    assert table_fingerprint(table) == table_fingerprint(table)


def test_table_fingerprint_equal_for_equal_tables_and_copies():
    table = fingerprinted_table()
    assert table_fingerprint(fingerprinted_table()) == table_fingerprint(table)
    assert table_fingerprint(table.copy()) == table_fingerprint(table)
    assert table_fingerprint(table.copy(deep=False)) == table_fingerprint(table)


def test_table_fingerprint_is_the_same_in_another_process(tmp_path):
    # A cache is only useful across runs: the fingerprint must not depend on the process
    # (no per-process hash seed).
    script = tmp_path / 'fingerprint.py'
    script.write_text(
        "import json, pandas as pd\n"
        "from collection_estimation.hurdle.cache import table_fingerprint\n"
        "table = pd.DataFrame({'company_code': pd.Series(['TR02', 'TR04', 'TR02'], "
        "dtype=object), 'origin_week': pd.Series([33, 33, 34], dtype='int64'), "
        "'amount': [10.5, -2.25, 0.0]})\n"
        "print(json.dumps(table_fingerprint(table)))\n", encoding='utf-8')
    source_folder = Path(__file__).resolve().parents[2] / 'src'
    environment = {**os.environ, 'PYTHONPATH': str(source_folder),
                   'PYTHONHASHSEED': '12345'}
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            env=environment)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == table_fingerprint(fingerprinted_table())


def test_table_fingerprint_survives_a_json_round_trip():
    found = table_fingerprint(fingerprinted_table())
    assert json.loads(json.dumps(found, indent=1)) == found


def test_table_fingerprint_does_not_change_the_table():
    table = fingerprinted_table()
    table_fingerprint(table)
    pd.testing.assert_frame_equal(table, fingerprinted_table())


def changed_value(table):
    table.loc[1, 'amount'] = -2.2500001
    return table


def changed_text(table):
    table.loc[2, 'company_code'] = 'TR05'
    return table


def changed_index(table):
    return table.set_axis([0, 1, 5])


def changed_column_name(table):
    return table.rename(columns={'amount': 'amount_try'})


def changed_dtype(table):
    return table.astype({'origin_week': 'float64'})


def changed_row_order(table):
    return table.iloc[[2, 0, 1]]


def changed_row_order_reset(table):
    return table.iloc[[2, 0, 1]].reset_index(drop=True)


def changed_column_order(table):
    return table[['amount', 'company_code', 'origin_week']]


def added_row(table):
    return pd.concat([table, table.iloc[[0]]], ignore_index=True)


def dropped_row(table):
    return table.iloc[:2]


def added_column(table):
    return table.assign(extra=1.0)


def zero_sign(table):
    table.loc[2, 'amount'] = -0.0
    return table


TABLE_CHANGES = [changed_value, changed_text, changed_index, changed_column_name,
                 changed_dtype, changed_row_order, changed_row_order_reset,
                 changed_column_order, added_row, dropped_row, added_column, zero_sign]


@pytest.mark.parametrize('change', TABLE_CHANGES, ids=[c.__name__ for c in TABLE_CHANGES])
def test_table_fingerprint_changes_when_the_table_changes(change):
    before = table_fingerprint(fingerprinted_table())
    after = table_fingerprint(change(fingerprinted_table()))
    assert after != before


def test_table_fingerprint_a_dtype_change_moves_the_first_digest():
    before = table_fingerprint(fingerprinted_table())
    after = table_fingerprint(changed_dtype(fingerprinted_table()))
    assert after[0] != before[0]


def test_table_fingerprint_a_value_change_moves_only_the_second_digest():
    before = table_fingerprint(fingerprinted_table())
    after = table_fingerprint(changed_value(fingerprinted_table()))
    assert after[0] == before[0] and after[1] != before[1]


def test_table_fingerprint_of_a_series_is_that_of_its_frame():
    levels = pd.Series([1.0, 1.02, 1.05], index=pd.Index([10, 11, 12]), name='price_level')
    assert table_fingerprint(levels) == table_fingerprint(levels.to_frame())


def test_table_fingerprint_of_a_series_changes_with_a_value_the_index_or_the_name():
    levels = pd.Series([1.0, 1.02, 1.05], index=[10, 11, 12], name='price_level')
    before = table_fingerprint(levels)
    other_value = levels.copy()
    other_value.iloc[1] = 1.03
    assert table_fingerprint(other_value) != before
    assert table_fingerprint(levels.set_axis([10, 11, 13])) != before
    assert table_fingerprint(levels.rename('level')) != before


def test_table_fingerprint_of_an_unnamed_series_works():
    unnamed = pd.Series([1.0, 2.0])
    found = table_fingerprint(unnamed)
    assert found == table_fingerprint(unnamed.copy())
    assert len(found) == 2


def test_table_fingerprint_of_empty_tables():
    # No rows: still two digests; the columns still count.
    empty = pd.DataFrame({'x': pd.Series([], dtype='float64')})
    assert len(table_fingerprint(empty)) == 2
    assert table_fingerprint(empty) != table_fingerprint(pd.DataFrame())
    assert table_fingerprint(pd.DataFrame()) == table_fingerprint(pd.DataFrame())
