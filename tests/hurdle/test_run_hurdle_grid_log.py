"""Tests of run_hurdle_grid.py's log: main() copies everything printed into a log file.

The script is loaded as in test_run_hurdle_grid.py. Its `run` (or, through the real run, its
load_settings / prepare_inputs / run_grid) is replaced by fakes, so no data is read. Every
log goes to a tmp folder; an autouse guard also moves main's default log folder to tmp, so no
test writes into the repo's outputs/.
"""
import inspect
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tests.hurdle.code_checks import REPO_ROOT
from tests.hurdle.test_run_hurdle_grid import Main, load_script

NAME_PATTERN = r"hurdle_grid_\d{8}_\d{6}\.log"


@pytest.fixture(scope="module")
def script():
    return load_script()


@pytest.fixture(autouse=True)
def _no_log_into_the_repo(script, tmp_path_factory, monkeypatch):
    defaults = script.main.__defaults__
    assert len(defaults) == 2 and defaults[1] == REPO_ROOT / "outputs" / "logs"
    monkeypatch.setattr(script.main, "__defaults__",
                        (defaults[0], tmp_path_factory.mktemp("default_logs")))


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "logs"


def fake_run(monkeypatch, script, body):
    """Replace run() by `body(argv)`; returns the list of argvs it was called with."""
    calls = []

    def run(argv=None):
        calls.append(argv)
        return body(argv)

    monkeypatch.setattr(script, "run", run)
    return calls


def only_log(folder):
    files = list(folder.iterdir())
    assert len(files) == 1, files
    return files[0]


class FrozenClock:
    """Stands in for the script's `datetime`: now() is a fixed moment."""

    def __init__(self, moment):
        self.moment = moment

    def now(self):
        return self.moment


# --- the console and the log get the same text -----------------------------------------------

def test_stdout_and_stderr_reach_the_console_and_the_log_in_order(script, monkeypatch,
                                                                    capsys, folder):
    def body(argv):
        print("one")
        print("two to stderr", file=sys.stderr)
        sys.stdout.write("three, no newline;")
        sys.stderr.write(" four\n")
        print("five")

    fake_run(monkeypatch, script, body)
    capsys.readouterr()
    script.main(["--settings", "x.toml"], log_folder=folder)
    out, err = capsys.readouterr()
    log_file = only_log(folder)
    assert out == f"Log: {log_file}\none\nthree, no newline;five\n"
    assert err == "two to stderr\n four\n"
    assert log_file.read_text(encoding="utf-8") == (
        f"Log: {log_file}\none\ntwo to stderr\nthree, no newline; four\nfive\n")


def test_main_passes_argv_to_run_and_returns_none(script, monkeypatch, folder):
    calls = fake_run(monkeypatch, script, lambda argv: "ignored")
    assert script.main(["--settings", "x.toml"], log_folder=folder) is None
    assert calls == [["--settings", "x.toml"]]


def test_main_without_argv_passes_none_to_run(script, monkeypatch, folder):
    calls = fake_run(monkeypatch, script, lambda argv: None)
    script.main(log_folder=folder)
    assert calls == [None]


def test_the_real_run_is_logged_through_the_file_fakes(script, monkeypatch, capsys, folder):
    # The real run() with the existing fakes: the log holds exactly the console's printout.
    found = Main(script, monkeypatch, capsys, folder)()
    assert found.printed[0] == found.log_line
    assert found.log_file.read_text(encoding="utf-8").splitlines() == found.printed


# --- the log file: first line, name, folder --------------------------------------------------

def test_the_first_line_names_the_log_file(script, monkeypatch, capsys, folder):
    fake_run(monkeypatch, script, lambda argv: print("body"))
    capsys.readouterr()
    script.main([], log_folder=folder)
    log_file = only_log(folder)
    first = capsys.readouterr().out.splitlines()[0]
    assert first == f"Log: {log_file}"
    assert log_file.read_text(encoding="utf-8").splitlines()[0] == first


def test_the_log_file_is_in_the_folder_with_the_name_pattern(script, monkeypatch, folder):
    fake_run(monkeypatch, script, lambda argv: None)
    script.main([], log_folder=folder)
    log_file = only_log(folder)
    assert log_file.parent == folder
    assert re.fullmatch(NAME_PATTERN, log_file.name), log_file.name


def test_the_log_file_name_is_the_start_time(script, monkeypatch, capsys, folder):
    monkeypatch.setattr(script, "datetime", FrozenClock(datetime(2026, 11, 8, 9, 3, 4)))
    fake_run(monkeypatch, script, lambda argv: None)
    capsys.readouterr()
    script.main([], log_folder=folder)
    expected = folder / "hurdle_grid_20261108_090304.log"
    assert only_log(folder) == expected
    assert capsys.readouterr().out == f"Log: {expected}\n"


def test_the_log_file_name_is_taken_before_the_run(script, monkeypatch, folder):
    # The name is the START time: a clock that moves during the run does not rename it.
    clock = FrozenClock(datetime(2026, 1, 2, 3, 4, 5))
    monkeypatch.setattr(script, "datetime", clock)

    def body(argv):
        clock.moment = datetime(2026, 1, 2, 23, 59, 59)

    fake_run(monkeypatch, script, body)
    script.main([], log_folder=folder)
    assert only_log(folder).name == "hurdle_grid_20260102_030405.log"


def test_the_log_folder_is_created_with_its_parents(script, monkeypatch, tmp_path):
    folder = tmp_path / "a" / "b" / "c"
    assert not (tmp_path / "a").exists()
    fake_run(monkeypatch, script, lambda argv: print("x"))
    script.main([], log_folder=folder)
    assert folder.is_dir()
    assert only_log(folder).read_text(encoding="utf-8").endswith("x\n")


def test_an_existing_log_folder_is_fine_and_older_logs_are_kept(script, monkeypatch, folder):
    folder.mkdir()
    older = folder / "hurdle_grid_20000101_000000.log"
    older.write_text("old run\n", encoding="utf-8")
    fake_run(monkeypatch, script, lambda argv: None)
    script.main([], log_folder=folder)
    assert older.read_text(encoding="utf-8") == "old run\n"
    assert len(list(folder.iterdir())) == 2


def test_two_runs_a_second_apart_write_two_logs(script, monkeypatch, folder):
    clock = FrozenClock(datetime(2026, 5, 6, 7, 8, 9))
    monkeypatch.setattr(script, "datetime", clock)
    fake_run(monkeypatch, script, lambda argv: print("run at", clock.moment))
    script.main([], log_folder=folder)
    clock.moment = datetime(2026, 5, 6, 7, 8, 10)
    script.main([], log_folder=folder)
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["hurdle_grid_20260506_070809.log", "hurdle_grid_20260506_070810.log"]


def test_two_runs_in_the_same_second_keep_both_logs(script, monkeypatch, capsys, folder):
    # Was a strict xfail (the second run overwrote the first's log); fixed with new_log.
    monkeypatch.setattr(script, "datetime", FrozenClock(datetime(2026, 5, 6, 7, 8, 9)))
    fake_run(monkeypatch, script, lambda argv: print("first run's result"))
    script.main([], log_folder=folder)
    fake_run(monkeypatch, script, lambda argv: print("second run's result"))
    capsys.readouterr()
    script.main([], log_folder=folder)
    first = folder / "hurdle_grid_20260506_070809.log"
    second = folder / "hurdle_grid_20260506_070809_2.log"
    assert sorted(folder.iterdir()) == [first, second]
    assert first.read_text(encoding="utf-8") == f"Log: {first}\nfirst run's result\n"
    assert second.read_text(encoding="utf-8") == f"Log: {second}\nsecond run's result\n"
    assert capsys.readouterr().out.splitlines()[0] == f"Log: {second}"


def test_three_runs_in_the_same_second_keep_three_logs(script, monkeypatch, folder):
    monkeypatch.setattr(script, "datetime", FrozenClock(datetime(2026, 5, 6, 7, 8, 9)))
    for i in range(3):
        fake_run(monkeypatch, script, lambda argv, i=i: print(f"run {i}"))
        script.main([], log_folder=folder)
    stem = "hurdle_grid_20260506_070809"
    texts = {p.name: p.read_text(encoding="utf-8").splitlines()[1] for p in folder.iterdir()}
    assert texts == {f"{stem}.log": "run 0", f"{stem}_2.log": "run 1", f"{stem}_3.log": "run 2"}


# --- new_log -----------------------------------------------------------------------------------

def test_new_log_takes_the_plain_name_when_free(script, tmp_path):
    path, log = script.new_log(tmp_path, "stem")
    with log:
        assert path == tmp_path / "stem.log"
        assert path.is_file()


def test_new_log_numbers_2_then_3_on_repeated_clashes(script, tmp_path):
    found = []
    for _ in range(4):
        path, log = script.new_log(tmp_path, "stem")
        log.close()
        found.append(path.name)
    assert found == ["stem.log", "stem_2.log", "stem_3.log", "stem_4.log"]


def test_new_log_takes_the_first_free_number(script, tmp_path):
    # stem.log and stem_3.log taken: stem_2.log is next; then stem_4.log.
    (tmp_path / "stem.log").write_text("a", encoding="utf-8")
    (tmp_path / "stem_3.log").write_text("c", encoding="utf-8")
    path, log = script.new_log(tmp_path, "stem")
    log.close()
    assert path.name == "stem_2.log"
    path, log = script.new_log(tmp_path, "stem")
    log.close()
    assert path.name == "stem_4.log"


def test_new_log_with_only_a_numbered_file_taken_uses_the_plain_name(script, tmp_path):
    (tmp_path / "stem_2.log").write_text("b", encoding="utf-8")
    path, log = script.new_log(tmp_path, "stem")
    log.close()
    assert path.name == "stem.log"


def test_new_log_never_opens_an_existing_file_for_writing(script, monkeypatch, tmp_path):
    import builtins
    existing = {name: f"old {name}\n" for name in ("stem.log", "stem_2.log")}
    for name, text in existing.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    stamps = {name: (tmp_path / name).stat().st_mtime_ns for name in existing}
    opened = []

    def recording_open(file, mode="r", *args, **kwargs):
        opened.append((Path(file).name, mode, Path(file).exists()))
        return builtins.open(file, mode, *args, **kwargs)

    monkeypatch.setattr(script, "open", recording_open, raising=False)
    path, log = script.new_log(tmp_path, "stem")
    log.write("new run\n")
    log.close()
    assert path.name == "stem_3.log"
    # Every open is exclusive-create; the existing files were tried that way only.
    assert opened == [("stem.log", "x", True), ("stem_2.log", "x", True),
                      ("stem_3.log", "x", False)]
    for name, text in existing.items():
        assert (tmp_path / name).read_text(encoding="utf-8") == text
        assert (tmp_path / name).stat().st_mtime_ns == stamps[name]


def test_new_log_returns_an_open_writable_utf8_file(script, tmp_path):
    path, log = script.new_log(tmp_path, "stem")
    try:
        assert not log.closed and log.writable()
        assert log.encoding.lower().replace("-", "") == "utf8"
        assert "x" in log.mode
        assert Path(log.name) == path
        log.write("İstanbul Şubesi ğ ı\n")
    finally:
        log.close()
    assert "İstanbul Şubesi ğ ı".encode("utf-8") in path.read_bytes()
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_new_log_leaves_a_new_file_empty_until_written(script, tmp_path):
    path, log = script.new_log(tmp_path, "stem")
    log.close()
    assert path.read_bytes() == b""


def test_new_log_in_a_missing_folder_raises(script, tmp_path):
    # Not a FileExistsError: it is raised, not looped over.
    with pytest.raises(FileNotFoundError):
        script.new_log(tmp_path / "missing", "stem")
    assert not (tmp_path / "missing").exists()


def test_new_log_other_os_errors_are_raised_at_once(script, monkeypatch, tmp_path):
    calls = []

    def denied(file, mode="r", *args, **kwargs):
        calls.append(Path(file).name)
        raise PermissionError(13, "Permission denied", str(file))

    monkeypatch.setattr(script, "open", denied, raising=False)
    with pytest.raises(PermissionError):
        script.new_log(tmp_path, "stem")
    assert calls == ["stem.log"]


def test_main_creates_the_folder_before_new_log(script, monkeypatch, tmp_path):
    # main mkdirs first, so a missing folder does not reach new_log's FileNotFoundError.
    folder = tmp_path / "deep" / "logs"
    fake_run(monkeypatch, script, lambda argv: None)
    script.main([], log_folder=folder)
    assert len(list(folder.iterdir())) == 1


# --- a run that raises -----------------------------------------------------------------------

@pytest.mark.parametrize("error", [ValueError("ledger does not tie"), KeyboardInterrupt()])
def test_a_raising_run_reraises_the_same_exception(script, monkeypatch, folder, error):
    def body(argv):
        print("before the error")
        raise error

    fake_run(monkeypatch, script, body)
    with pytest.raises(type(error)) as raised:
        script.main([], log_folder=folder)
    assert raised.value is error


@pytest.mark.parametrize("error, last_line", [
    (ValueError("ledger does not tie"), "ValueError: ledger does not tie"),
    (KeyboardInterrupt(), "KeyboardInterrupt")])
def test_a_raising_run_ends_its_log_with_the_traceback_once(script, monkeypatch, capsys,
                                                           folder, error, last_line):
    def body(argv):
        print("before the error")
        print("a warning", file=sys.stderr)
        raise error

    fake_run(monkeypatch, script, body)
    capsys.readouterr()
    with pytest.raises(type(error)):
        script.main([], log_folder=folder)
    log_file = only_log(folder)
    text = log_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[:3] == [f"Log: {log_file}", "before the error", "a warning"]
    assert lines[3] == "Traceback (most recent call last):"
    assert lines[-1] == last_line
    assert text.endswith(last_line + "\n")
    assert text.count("Traceback (most recent call last):") == 1
    assert text.count(last_line) == 1
    assert "in body" in text            # the traceback reaches the raising frame


@pytest.mark.parametrize("error", [ValueError("gates failed"), KeyboardInterrupt()])
def test_the_traceback_goes_to_the_log_only_not_the_console(script, monkeypatch, capsys,
                                                            folder, error):
    # main writes the traceback to the log itself; the console gets it from Python when the
    # exception leaves the script, so main must not print it there too.
    def body(argv):
        raise error

    fake_run(monkeypatch, script, body)
    capsys.readouterr()
    with pytest.raises(type(error)):
        script.main([], log_folder=folder)
    out, err = capsys.readouterr()
    assert "Traceback" not in out and "Traceback" not in err
    assert out == f"Log: {only_log(folder)}\n" and err == ""


def test_a_failing_check_in_the_real_run_is_logged(script, monkeypatch, capsys, folder):
    # A ValueError from a check (here load_settings) through the real run().
    def broken(*args, **kwargs):
        raise ValueError("tiny.toml [grid]: unknown setting 'x'")

    monkeypatch.setattr(script, "load_settings", broken)
    with pytest.raises(ValueError, match="unknown setting"):
        script.main(["--settings", "tiny.toml"], log_folder=folder)
    text = only_log(folder).read_text(encoding="utf-8")
    assert text.endswith("ValueError: tiny.toml [grid]: unknown setting 'x'\n")
    assert text.count("Traceback (most recent call last):") == 1


def test_a_failing_run_grid_keeps_the_earlier_printout_in_the_log(script, monkeypatch, capsys,
                                                                  folder):
    found = Main(script, monkeypatch, capsys, folder)

    def broken(*args, **kwargs):
        print("grid started")
        raise ValueError("gates failed")

    monkeypatch.setattr(script, "run_grid", broken)
    with pytest.raises(ValueError, match="gates failed"):
        found()
    lines = only_log(folder).read_text(encoding="utf-8").splitlines()
    assert lines[1].startswith("Settings: ")
    assert lines[3].startswith("Steps 1-11: inputs ready")
    assert lines[4] == "grid started"
    assert lines[5] == "Traceback (most recent call last):"
    assert lines[-1] == "ValueError: gates failed"
    assert not [line for line in lines if line.startswith(("kept:", "Done in"))]


# --- sys.stdout / sys.stderr are restored ----------------------------------------------------

def test_stdout_and_stderr_are_tees_during_the_run(script, monkeypatch, folder):
    before = sys.stdout, sys.stderr
    seen = {}

    def body(argv):
        seen["out"], seen["err"] = sys.stdout, sys.stderr

    fake_run(monkeypatch, script, body)
    script.main([], log_folder=folder)
    assert isinstance(seen["out"], script.Tee) and isinstance(seen["err"], script.Tee)
    assert seen["out"].console is before[0] and seen["err"].console is before[1]
    assert seen["out"].log is seen["err"].log


def test_stdout_and_stderr_are_restored_after_a_normal_run(script, monkeypatch, folder):
    before = sys.stdout, sys.stderr
    fake_run(monkeypatch, script, lambda argv: print("x"))
    script.main([], log_folder=folder)
    assert sys.stdout is before[0] and sys.stderr is before[1]


@pytest.mark.parametrize("error", [ValueError("x"), KeyboardInterrupt(), SystemExit(2)])
def test_stdout_and_stderr_are_restored_after_a_raising_run(script, monkeypatch, folder,
                                                           error):
    before = sys.stdout, sys.stderr

    def body(argv):
        raise error

    fake_run(monkeypatch, script, body)
    with pytest.raises(type(error)):
        script.main([], log_folder=folder)
    assert sys.stdout is before[0] and sys.stderr is before[1]


def test_the_log_file_is_closed_after_the_run(script, monkeypatch, folder):
    seen = {}
    fake_run(monkeypatch, script, lambda argv: seen.setdefault("log", sys.stdout.log))
    script.main([], log_folder=folder)
    assert seen["log"].closed


def test_the_log_file_is_closed_after_a_raising_run(script, monkeypatch, folder):
    seen = {}

    def body(argv):
        seen["log"] = sys.stdout.log
        raise ValueError("x")

    fake_run(monkeypatch, script, body)
    with pytest.raises(ValueError):
        script.main([], log_folder=folder)
    assert seen["log"].closed


def test_a_bad_argument_exits_2_and_is_logged(script, folder, capsys):
    # argparse's SystemExit passes through main too; the usage message is in the log.
    with pytest.raises(SystemExit) as raised:
        script.main(["--no-such-flag"], log_folder=folder)
    assert raised.value.code == 2
    text = only_log(folder).read_text(encoding="utf-8")
    assert "unrecognized arguments: --no-such-flag" in text
    assert text.rstrip().endswith("SystemExit: 2")


# --- encoding and flushing -------------------------------------------------------------------

def test_turkish_characters_are_written_to_the_log_as_utf8(script, monkeypatch, capsys,
                                                           folder):
    name = "İŞĞÜÖÇ işğüöçı Ankara Şubesi"

    def body(argv):
        print(name)
        print("stderr: ı ğ", file=sys.stderr)

    fake_run(monkeypatch, script, body)
    capsys.readouterr()
    script.main([], log_folder=folder)
    raw = only_log(folder).read_bytes()
    assert name.encode("utf-8") in raw and "stderr: ı ğ".encode("utf-8") in raw
    assert not raw.startswith(b"\xef\xbb\xbf")          # no BOM
    assert raw.decode("utf-8").splitlines()[1:] == [name, "stderr: ı ğ"]
    out, err = capsys.readouterr()
    assert name in out and "ı ğ" in err


def test_a_turkish_error_message_reaches_the_log_as_utf8(script, monkeypatch, folder):
    def body(argv):
        raise ValueError("Müşteri İstanbul için bakiye tutmuyor")

    fake_run(monkeypatch, script, body)
    with pytest.raises(ValueError):
        script.main([], log_folder=folder)
    message = "ValueError: Müşteri İstanbul için bakiye tutmuyor"
    assert message.encode("utf-8") in only_log(folder).read_bytes()
    assert only_log(folder).read_text(encoding="utf-8").endswith(message + "\n")


def test_the_log_is_flushed_on_every_write(script, monkeypatch, folder):
    # Read from inside the run: each line is in the file as soon as it is written.
    seen = []

    def body(argv):
        log_file = only_log(folder)
        seen.append(log_file.read_text(encoding="utf-8"))
        print("first")
        seen.append(log_file.read_text(encoding="utf-8"))
        sys.stderr.write("second, no newline")
        seen.append(log_file.read_text(encoding="utf-8"))

    fake_run(monkeypatch, script, body)
    script.main([], log_folder=folder)
    log_line = f"Log: {only_log(folder)}\n"
    assert seen == [log_line, log_line + "first\n", log_line + "first\nsecond, no newline"]


def test_the_log_is_readable_before_a_run_that_hangs_is_stopped(script, monkeypatch, folder):
    # A long run stopped with Ctrl+C: what it printed was on disk before the stop.
    seen = {}

    def body(argv):
        print("fold 1 of 12 done")
        seen["text"] = only_log(folder).read_text(encoding="utf-8")
        raise KeyboardInterrupt

    fake_run(monkeypatch, script, body)
    with pytest.raises(KeyboardInterrupt):
        script.main([], log_folder=folder)
    assert seen["text"].endswith("fold 1 of 12 done\n")


# --- Tee -------------------------------------------------------------------------------------

class Recorder:
    """A stream that records writes and flushes; has an encoding and isatty."""

    def __init__(self, encoding="cp1254", tty=True):
        self.written, self.flushes = [], 0
        self.encoding, self._tty = encoding, tty

    def write(self, text):
        self.written.append(text)
        return 999          # Tee must not pass this on

    def flush(self):
        self.flushes += 1

    def isatty(self):
        return self._tty


@pytest.mark.parametrize("text", ["", "a", "İstanbul\n", "x" * 10_000])
def test_tee_write_returns_the_length_of_the_text(script, text):
    tee = script.Tee(Recorder(), Recorder())
    assert tee.write(text) == len(text)


def test_tee_write_writes_the_same_text_to_both(script):
    console, log = Recorder(), Recorder()
    tee = script.Tee(console, log)
    tee.write("a")
    tee.write("İ\n")
    assert console.written == ["a", "İ\n"] and log.written == ["a", "İ\n"]


def test_tee_write_flushes_both_every_time(script):
    console, log = Recorder(), Recorder()
    tee = script.Tee(console, log)
    tee.write("a")
    assert (console.flushes, log.flushes) == (1, 1)
    tee.write("b")
    assert (console.flushes, log.flushes) == (2, 2)


def test_tee_flush_flushes_both(script):
    console, log = Recorder(), Recorder()
    script.Tee(console, log).flush()
    assert (console.flushes, log.flushes) == (1, 1)


def test_tee_delegates_other_attributes_to_the_console(script):
    console, log = Recorder(encoding="utf-8", tty=False), Recorder(encoding="ascii", tty=True)
    tee = script.Tee(console, log)
    assert tee.encoding == "utf-8"
    assert tee.isatty() is False
    console.encoding = "cp1254"
    assert tee.encoding == "cp1254"       # read live, not copied


def test_tee_keeps_its_own_console_and_log(script):
    console, log = Recorder(), Recorder()
    tee = script.Tee(console, log)
    assert tee.console is console and tee.log is log


def test_tee_raises_attribute_error_for_what_the_console_lacks(script):
    tee = script.Tee(Recorder(), Recorder())
    with pytest.raises(AttributeError):
        tee.no_such_attribute
    assert not hasattr(tee, "no_such_attribute")


def test_tee_works_with_print(script):
    import io
    console, log = io.StringIO(), io.StringIO()
    print("a", 1, "İ", file=script.Tee(console, log))
    assert console.getvalue() == log.getvalue() == "a 1 İ\n"


# --- the default folder ----------------------------------------------------------------------

def test_the_default_log_folder_is_outputs_logs_under_the_repo_root():
    fresh = load_script()           # not touched by the autouse guard
    assert fresh.LOG_FOLDER == REPO_ROOT / "outputs" / "logs"
    assert fresh.LOG_FOLDER.is_absolute()
    default = inspect.signature(fresh.main).parameters["log_folder"].default
    assert default == REPO_ROOT / "outputs" / "logs"


def test_loading_the_script_creates_no_log_folder(tmp_path):
    # The folder is made by main(), not at import.
    existed = (REPO_ROOT / "outputs").exists()
    load_script()
    assert (REPO_ROOT / "outputs").exists() == existed


def test_outputs_is_in_the_gitignore():
    lines = [line.strip() for line in
             (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "outputs/" in lines


def test_git_ignores_a_log_in_the_default_folder():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "outputs/logs/hurdle_grid_20260101_000000.log"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (result.returncode, result.stderr)
