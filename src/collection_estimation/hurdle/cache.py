"""A cache for tables that are slow to read (the daily files, the AR ledger).

A table is kept as a pickle beside a JSON "fingerprint" of everything it was built from:
the input files (full path, size, last-modified time), any settings, and the source code of
the reader. If any of these changes, or the cache is damaged, the table is read again.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd


def load_with_cache(name: str, fingerprint: list, read, cache_folder: Path) -> pd.DataFrame:
    """The table built by `read()`, cached as <name>.pkl beside <name>_fingerprint.json.

    `read` is a function with no arguments that builds the table. The cache is used only if
    its saved fingerprint equals `fingerprint`; otherwise, or if the cache is damaged,
    `read()` runs again and both cache files are rewritten.
    """
    cache_folder = Path(cache_folder)
    cache_file = cache_folder / f"{name}.pkl"
    fingerprint_file = cache_folder / f"{name}_fingerprint.json"

    # Encode the fingerprint first: one JSON cannot save (e.g. holding a date) then fails
    # here, before anything is read or written.
    fingerprint_text = json.dumps(fingerprint, indent=1)

    cached = read_cache(cache_file, fingerprint_file, fingerprint)
    if cached is not None:
        return cached

    table = read()
    cache_folder.mkdir(parents=True, exist_ok=True)
    # Remove the old fingerprint before replacing the pickle: if writing is interrupted,
    # the old fingerprint must never end up beside a new table.
    fingerprint_file.unlink(missing_ok=True)
    # A pickle keeps the table exactly (text IDs, None for missing text, column types).
    # Only ever load pickles this code wrote itself.
    table.to_pickle(cache_file)
    fingerprint_file.write_text(fingerprint_text, encoding="utf-8")
    return table


def read_cache(cache_file: Path, fingerprint_file: Path, fingerprint: list):
    """The cached table if both cache files exist, load, and match `fingerprint`; else None."""
    if not (cache_file.exists() and fingerprint_file.exists()):
        return None
    try:
        cached_fingerprint = json.loads(fingerprint_file.read_text(encoding="utf-8"))
        # Compare in the form JSON saves it (e.g. a tuple is saved as a list), so a
        # fingerprint built from tuples still matches its saved copy.
        if cached_fingerprint != json.loads(json.dumps(fingerprint)):
            print(f"{cache_file.stem}: the inputs have changed since the cache was written; "
                  f"reading them again.")
            return None
        return pd.read_pickle(cache_file)
    except Exception as error:
        # A damaged cache (e.g. a run interrupted while writing it) can fail in many ways;
        # whatever the reason, the inputs themselves are still there to be read again.
        print(f"{cache_file.stem}: the cache could not be loaded ({error!r}); "
              f"reading the inputs again.")
        return None


def file_fingerprint(path: Path) -> list:
    """[full path, size, last-modified time] of one file. Any change means re-read.

    The full path (not just the name) makes switching to another folder with the same file
    names, e.g. from the synthetic files to the real ones, always re-read.
    """
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime]


def source_fingerprint(*source_files: Path) -> list[str]:
    """A hash of each source file, so that a change to a reader's code forces one re-read.

    Without it a cache would keep serving a table built by the old code.
    """
    return [hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
            for source_file in source_files]
