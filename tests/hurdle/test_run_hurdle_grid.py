"""Tests of `run_hurdle_grid.py` (the settings grid alone, e.g. on the compute machine).

The script is loaded in this process from its file, under its own module name, with
sys.stdout swapped for a stub while it loads (the script reconfigures sys.stdout at import,
which should not touch pytest's capture) and sys.path restored afterwards (it puts src/ on
it). Its functions are then called with `prepare_inputs`, `run_grid` and `load_settings`
replaced in the script's namespace. Running it as a script (exit codes) is done in a separate
Python process, as in test_run_hurdle_pipeline.py.

The end-to-end run reuses test_grid_run.py's tiny grid: real run_grid, real fits (cached in a
tmp folder), test_grid.py's two candidates per stage.
"""
import dataclasses
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from collection_estimation.hurdle.calibrate import (MCorrection, PCorrection, candidate_name)
from collection_estimation.hurdle.evaluate import HURDLE, POOLED
from collection_estimation.hurdle.forecast import CONSOLIDATED
from tests.hurdle.code_checks import REPO_ROOT, run_snippet

SCRIPT = REPO_ROOT / "run_hurdle_grid.py"
SETTINGS_ENV = "HURDLE_SETTINGS"


class _StdoutStub:
    """Stands in for sys.stdout while the script loads; records its reconfigure call."""

    def __init__(self):
        self.reconfigured = []

    def reconfigure(self, **kwargs):
        self.reconfigured.append(kwargs)

    def write(self, text):
        return len(text)

    def flush(self):
        pass


def load_script():
    saved_path, saved_stdout = list(sys.path), sys.stdout
    stub = _StdoutStub()
    sys.stdout = stub
    try:
        spec = importlib.util.spec_from_file_location("run_hurdle_grid_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.stdout = saved_stdout
        sys.path[:] = saved_path
    module._stub = stub
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


# --- the module -----------------------------------------------------------------------------

def test_the_script_exists_at_the_repo_root():
    assert SCRIPT.is_file()


def test_repo_root_and_default_settings_file(script):
    assert script.REPO_ROOT == REPO_ROOT
    assert script.DEFAULT_SETTINGS_FILE == REPO_ROOT / "settings" / "synthetic.toml"
    assert script.DEFAULT_SETTINGS_FILE.is_absolute()


def test_loading_the_script_reconfigures_stdout_to_utf8_with_replace(script):
    assert script._stub.reconfigured == [{"encoding": "utf-8", "errors": "replace"}]


def test_the_script_uses_the_package_functions_themselves(script):
    from collection_estimation.hurdle import calibrate, grid_run, inputs, settings
    assert script.prepare_inputs is inputs.prepare_inputs
    assert script.run_grid is grid_run.run_grid
    assert script.load_settings is settings.load_settings
    assert script.candidate_name is calibrate.candidate_name
    assert (script.HURDLE, script.POOLED, script.CONSOLIDATED) == (HURDLE, POOLED, CONSOLIDATED)


def test_the_script_never_reads_the_grid_run_switch():
    # [grid] run is deliberately not read: the script exists to run the grid.
    # Neither [grid] run nor run_test (that one is run_grid's to read) is read here: no
    # attribute `run` or `run_test` anywhere in the code (docstrings may name them).
    import ast
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    read = [ast.unparse(node) for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in ("run", "run_test")]
    assert read == []
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "grid" not in names


def test_the_script_has_no_try_except():
    import ast
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Try)]


# --- 1. settings_file: flag, then env, then default ----------------------------------------

def test_settings_default_is_synthetic_toml_under_the_repo_root(script, monkeypatch):
    monkeypatch.delenv(SETTINGS_ENV, raising=False)
    assert script.settings_file([]) == REPO_ROOT / "settings" / "synthetic.toml"


def test_settings_env_beats_the_default(script, monkeypatch, tmp_path):
    monkeypatch.setenv(SETTINGS_ENV, str(tmp_path / "from_env.toml"))
    assert script.settings_file([]) == tmp_path / "from_env.toml"


def test_settings_flag_beats_the_env(script, monkeypatch, tmp_path):
    monkeypatch.setenv(SETTINGS_ENV, str(tmp_path / "from_env.toml"))
    found = script.settings_file(["--settings", str(tmp_path / "from_flag.toml")])
    assert found == tmp_path / "from_flag.toml"


def test_settings_flag_beats_the_default(script, monkeypatch, tmp_path):
    monkeypatch.delenv(SETTINGS_ENV, raising=False)
    assert script.settings_file([f"--settings={tmp_path / 'x.toml'}"]) == tmp_path / "x.toml"


def test_settings_absolute_flag_is_kept_as_it_is(script, monkeypatch, tmp_path):
    monkeypatch.setenv(SETTINGS_ENV, "elsewhere.toml")
    target = tmp_path / "compute.toml"
    assert target.is_absolute()
    assert script.settings_file(["--settings", str(target)]) == target


def test_settings_absolute_env_is_kept_as_it_is(script, monkeypatch, tmp_path):
    target = tmp_path / "compute.toml"
    monkeypatch.setenv(SETTINGS_ENV, str(target))
    assert script.settings_file([]) == target


def test_settings_relative_flag_is_read_from_the_repo_root(script, monkeypatch):
    monkeypatch.setenv(SETTINGS_ENV, "elsewhere.toml")
    found = script.settings_file(["--settings", "settings/compute.toml"])
    assert found == REPO_ROOT / "settings" / "compute.toml"
    assert found.is_absolute()


def test_settings_relative_env_is_read_from_the_repo_root(script, monkeypatch):
    monkeypatch.setenv(SETTINGS_ENV, "settings/compute.toml")
    found = script.settings_file([])
    assert found == REPO_ROOT / "settings" / "compute.toml" and found.is_absolute()


@pytest.mark.parametrize("given", ["x.toml", "settings/x.toml", "../x.toml"])
def test_settings_relative_paths_follow_repo_root_not_the_working_directory(
        script, monkeypatch, tmp_path, given):
    # REPO_ROOT replaced, and the working directory moved elsewhere: only REPO_ROOT counts.
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    monkeypatch.setattr(script, "REPO_ROOT", root)
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv(SETTINGS_ENV, raising=False)
    assert script.settings_file(["--settings", given]) == root / given
    monkeypatch.setenv(SETTINGS_ENV, given)
    assert script.settings_file([]) == root / given


def test_settings_absolute_path_ignores_repo_root(script, monkeypatch, tmp_path):
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path / "root")
    target = tmp_path / "other" / "x.toml"
    monkeypatch.delenv(SETTINGS_ENV, raising=False)
    assert script.settings_file(["--settings", str(target)]) == target
    monkeypatch.setenv(SETTINGS_ENV, str(target))
    assert script.settings_file([]) == target


def test_settings_empty_env_falls_back_to_the_default(script, monkeypatch):
    monkeypatch.setenv(SETTINGS_ENV, "")
    assert script.settings_file([]) == REPO_ROOT / "settings" / "synthetic.toml"


def test_settings_empty_env_does_not_hide_the_flag(script, monkeypatch, tmp_path):
    monkeypatch.setenv(SETTINGS_ENV, "")
    assert script.settings_file(["--settings", str(tmp_path / "x.toml")]) == tmp_path / "x.toml"


def test_settings_default_is_unchanged_by_the_repo_root_join(script, monkeypatch):
    # DEFAULT_SETTINGS_FILE is absolute already: REPO_ROOT / it is itself.
    monkeypatch.delenv(SETTINGS_ENV, raising=False)
    assert script.settings_file([]) == script.DEFAULT_SETTINGS_FILE


def test_settings_file_always_returns_a_path(script, monkeypatch):
    monkeypatch.setenv(SETTINGS_ENV, "a.toml")
    assert isinstance(script.settings_file([]), Path)
    assert isinstance(script.settings_file(["--settings", "b.toml"]), Path)
    monkeypatch.delenv(SETTINGS_ENV)
    assert isinstance(script.settings_file([]), Path)


def test_settings_file_refuses_an_unknown_argument(script):
    with pytest.raises(SystemExit) as raised:
        script.settings_file(["--config", "x.toml"])
    assert raised.value.code == 2


def test_settings_file_refuses_a_positional_path(script):
    with pytest.raises(SystemExit) as raised:
        script.settings_file(["x.toml"])
    assert raised.value.code == 2


def test_settings_flag_may_be_abbreviated(script, monkeypatch):
    # argparse's default allow_abbrev: "--set" is taken for "--settings".
    monkeypatch.delenv(SETTINGS_ENV, raising=False)
    assert script.settings_file(["--set", "x.toml"]) == REPO_ROOT / "x.toml"


def test_settings_file_refuses_a_flag_without_a_value(script):
    with pytest.raises(SystemExit) as raised:
        script.settings_file(["--settings"])
    assert raised.value.code == 2


# --- 2. main with fakes -----------------------------------------------------------------------

class Main:
    """main() with load_settings, prepare_inputs, run_grid and the clock replaced."""

    def __init__(self, script, monkeypatch, capsys, *, grid_run_switch=False, test=None,
                 ticks=(10.0, 52.4, 160.0), rows=1234, columns=17):
        self.script, self.capsys = script, capsys
        self.calls = []
        self.settings = SimpleNamespace(
            settings_file=Path("/settings/tiny.toml"),
            data=SimpleNamespace(corpus_folder=Path("/corpus/v2")),
            grid=SimpleNamespace(run=grid_run_switch, run_test=test is not None))
        self.inputs = SimpleNamespace(table=pd.DataFrame(0, index=range(rows),
                                                         columns=[f"c{i}" for i in range(columns)]))
        self.choice = fake_choice()
        self.test = test
        clock = iter(ticks)

        def load_settings(*args, **kwargs):
            self.calls.append(("load_settings", args, kwargs))
            return self.settings

        def prepare_inputs(*args, **kwargs):
            self.calls.append(("prepare_inputs", args, kwargs))
            return self.inputs, None

        def run_grid(*args, **kwargs):
            self.calls.append(("run_grid", args, kwargs))
            print("<run_grid output>")
            return self.choice, self.test

        monkeypatch.setattr(script, "load_settings", load_settings)
        monkeypatch.setattr(script, "prepare_inputs", prepare_inputs)
        monkeypatch.setattr(script, "run_grid", run_grid)
        monkeypatch.setattr(script, "time", SimpleNamespace(perf_counter=lambda: next(clock)))

    def __call__(self, argv=("--settings", "tiny.toml")):
        self.capsys.readouterr()
        self.returned = self.script.main(list(argv))
        self.printed = self.capsys.readouterr().out.splitlines()
        return self

    def only(self, name):
        found = [(a, k) for n, a, k in self.calls if n == name]
        assert len(found) == 1, (name, len(found))
        return found[0]


@pytest.fixture
def main(script, monkeypatch, capsys):
    def make(**options):
        return Main(script, monkeypatch, capsys, **options)
    return make


def test_main_calls_load_prepare_and_grid_in_order(main):
    found = main()()
    assert [name for name, _, _ in found.calls] == ["load_settings", "prepare_inputs",
                                                    "run_grid"]
    assert found.returned is None


def test_main_loads_the_chosen_file_against_the_repo_root(main, monkeypatch):
    monkeypatch.setenv(SETTINGS_ENV, "from_env.toml")
    found = main()(argv=["--settings", "settings/compute.toml"])
    assert found.only("load_settings") == ((REPO_ROOT / "settings" / "compute.toml",
                                               REPO_ROOT), {})


def test_main_without_a_flag_loads_the_env_file(main, monkeypatch):
    monkeypatch.setenv(SETTINGS_ENV, "from_env.toml")
    found = main()(argv=[])
    assert found.only("load_settings") == ((REPO_ROOT / "from_env.toml", REPO_ROOT), {})


def test_main_without_flag_or_env_loads_the_default(main, monkeypatch):
    monkeypatch.delenv(SETTINGS_ENV, raising=False)
    found = main()(argv=[])
    assert found.only("load_settings")[0] == (REPO_ROOT / "settings" / "synthetic.toml",
                                              REPO_ROOT)


def test_main_prepares_the_inputs_from_the_settings_only_without_details(main):
    found = main()()
    args, kwargs = found.only("prepare_inputs")
    assert len(args) == 1 and args[0] is found.settings
    assert kwargs == {}          # no with_details=True


def test_main_runs_the_grid_on_the_settings_and_inputs_without_rolled(main):
    found = main()()
    args, kwargs = found.only("run_grid")
    assert len(args) == 2 and args[0] is found.settings and args[1] is found.inputs
    assert "rolled" not in kwargs and kwargs == {}


@pytest.mark.parametrize("switch", [False, True])
def test_main_runs_the_grid_whatever_grid_run_says(main, switch):
    found = main(grid_run_switch=switch)()
    found.only("run_grid")


def test_main_runs_the_grid_even_without_a_grid_run_attribute(main):
    found = main()
    del found.settings.grid.run       # reading it would raise AttributeError
    found()
    found.only("run_grid")


def test_main_printout_order_without_the_test(main):
    found = main()()
    assert found.printed == [
        f"Settings: {Path('/settings/tiny.toml')}",
        f"Corpus folder: {Path('/corpus/v2')}",
        "Steps 1-11: inputs ready in 42 s (training table 1,234 rows x 17 columns)",
        "<run_grid output>",
        found.script.summary(found.choice, None),
        "Done in 2.5 min",
    ]
    assert found.printed[4].endswith("; v1 test not run")


def test_main_printout_with_the_test_ends_with_its_wape(main):
    test = SimpleNamespace(scores=make_test_scores())
    found = main(test=test)()
    assert found.printed[3] == "<run_grid output>"
    assert found.printed[4] == found.script.summary(found.choice, test)
    assert found.printed[4].endswith("; v1 test (corrected) WAPE 0.2222, bias -0.0333")
    assert found.printed[5].startswith("Done in ")


@pytest.mark.parametrize("ticks, ready, done", [
    ((0.0, 0.4, 59.0), "inputs ready in 0 s", "Done in 1.0 min"),
    ((100.0, 3700.0, 100.0 + 6 * 3600), "inputs ready in 3600 s", "Done in 360.0 min"),
    ((5.0, 5.0, 5.0), "inputs ready in 0 s", "Done in 0.0 min"),
])
def test_main_times_from_one_start(main, ticks, ready, done):
    found = main(ticks=ticks)()
    assert ready in found.printed[2]
    assert found.printed[-1] == done


def test_main_reads_the_clock_three_times(main):
    # Fewer ticks would raise StopIteration: start, after the inputs, at the end.
    main(ticks=(1.0, 2.0, 3.0))()


def test_main_training_table_size_uses_thousands_separators(main):
    found = main(rows=1_234_567, columns=3)()
    assert found.printed[2].endswith("(training table 1,234,567 rows x 3 columns)")


def test_main_with_an_empty_training_table(main):
    found = main(rows=0, columns=0)()
    assert found.printed[2].endswith("(training table 0 rows x 0 columns)")


# --- 4. failures propagate -------------------------------------------------------------------

def test_main_a_failing_prepare_inputs_propagates_before_the_grid(main, script, monkeypatch):
    found = main()

    def broken(*args, **kwargs):
        found.calls.append(("prepare_inputs", args, kwargs))
        raise ValueError("ledger does not tie")

    monkeypatch.setattr(script, "prepare_inputs", broken)
    with pytest.raises(ValueError, match="ledger does not tie"):
        found()
    assert "run_grid" not in [name for name, _, _ in found.calls]


def test_main_a_failing_run_grid_propagates_without_summary(main, script, monkeypatch,
                                                            capsys):
    found = main()

    def broken(*args, **kwargs):
        raise ValueError("gates failed")

    monkeypatch.setattr(script, "run_grid", broken)
    capsys.readouterr()
    with pytest.raises(ValueError, match="gates failed"):
        script.main(["--settings", "tiny.toml"])
    printed = capsys.readouterr().out
    assert "kept:" not in printed and "Done in" not in printed
    assert "Steps 1-11: inputs ready" in printed


def test_main_a_failing_load_settings_propagates(script, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("tiny.toml [grid]: unknown setting")

    monkeypatch.setattr(script, "load_settings", broken)
    called = []
    monkeypatch.setattr(script, "prepare_inputs", lambda *a, **k: called.append(a))
    with pytest.raises(ValueError, match="unknown setting"):
        script.main(["--settings", "tiny.toml"])
    assert called == []


def test_main_with_a_missing_file_raises_file_not_found(script, tmp_path):
    with pytest.raises(FileNotFoundError):
        script.main(["--settings", str(tmp_path / "missing.toml")])


def run_script(*args, env_settings=None, cwd=REPO_ROOT):
    import os
    env = {k: v for k, v in os.environ.items() if k != SETTINGS_ENV}
    if env_settings is not None:
        env[SETTINGS_ENV] = env_settings
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=300)


def test_the_script_with_a_missing_settings_file_exits_non_zero(tmp_path):
    result = run_script("--settings", str(tmp_path / "missing.toml"))
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert "Settings:" not in result.stdout


def test_the_script_with_a_settings_file_that_fails_its_check_exits_non_zero(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[data]\ncorpus_folder = 'x'\n", encoding="utf-8")
    result = run_script(env_settings=str(bad))      # the env var route, no flag
    assert result.returncode != 0
    assert "ValueError" in result.stderr and "bad.toml" in result.stderr


def test_the_script_from_another_directory_reads_a_relative_path_from_the_repo_root(tmp_path):
    # A file of that name sits in the working directory (and fails its check if read); the
    # script must look for it under the repo root instead, where it does not exist. Nothing
    # is written into the repo.
    name = "no_such_settings_file_for_this_test.toml"
    assert not (REPO_ROOT / name).exists()
    (tmp_path / name).write_text("[data]\ncorpus_folder = 'x'\n", encoding="utf-8")
    for args, env_settings in ((("--settings", name), None), ((), name)):
        result = run_script(*args, env_settings=env_settings, cwd=tmp_path)
        assert result.returncode != 0
        assert "FileNotFoundError" in result.stderr, result.stderr
        assert "ValueError" not in result.stderr
        # FileNotFoundError shows the path as repr (backslashes doubled on Windows).
        assert repr(str(REPO_ROOT / name)) in result.stderr, result.stderr


def test_the_script_imports_without_running():
    code = ("import sys\nsys.argv = ['x', '--settings', 'missing.toml']\n"
            "import run_hurdle_grid as script\nassert callable(script.main)\nprint('ok')\n")
    result = run_snippet(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# --- 3. summary -------------------------------------------------------------------------------

def fake_choice(wape=0.123456, total_wape=0.2, bias=-0.012345, p_kind="one map",
                m_kind="scale", method="mean", window=13, with_level=False):
    return SimpleNamespace(
        winner=pd.Series({"stage1": 0, "stage2": 1, "topups": 0, "current": False,
                          "wape": wape, "total_wape": total_wape, "bias": bias,
                          "company_wape": 0.3}),
        stage1=SimpleNamespace(params={"max_iter": 300, "learning_rate": 0.05}),
        stage2=SimpleNamespace(params={"loss": "poisson", "learning_rate": 0.1}),
        topups=SimpleNamespace(method=method, window=window, with_level=with_level),
        calibration=SimpleNamespace(p_correction=PCorrection(p_kind),
                                    m_correction=MCorrection(m_kind, common=1.1)))


def make_score_row(method, company, horizon, wape, bias):
    return {"method": method, "company_code": company, "horizon": horizon, "wape": wape,
            "bias": bias}


def make_test_scores(target_bias=-0.03333):
    """Several methods, companies and horizons; only one row is hurdle / ALL / pooled, with
    WAPE 0.22222 and bias `target_bias`. Every other row has a distinct WAPE and a distinct
    bias of the opposite sign to the default target's."""
    rows = []
    wape = 0.5
    for method in ("deflated", "hurdle uncorrected", HURDLE, "rule"):
        for company in ("TR02", CONSOLIDATED, "TR04"):
            for horizon in ("1", "2", POOLED, "total"):
                wape += 0.001
                rows.append(make_score_row(method, company, horizon, wape, wape - 0.4))
    table = pd.DataFrame(rows)
    target = ((table["method"] == HURDLE) & (table["company_code"] == CONSOLIDATED)
              & (table["horizon"] == POOLED))
    assert target.sum() == 1
    table.loc[target, "wape"] = 0.22222
    table.loc[target, "bias"] = target_bias
    return table


def test_summary_without_the_test(script):
    line = script.summary(fake_choice(), None)
    assert line == (
        "kept: stage 1 {'max_iter': 300, 'learning_rate': 0.05}, stage 2 {'loss': 'poisson', "
        "'learning_rate': 0.1}, top-ups mean (window 13, with_level False); validation "
        "(uncorrected) WAPE 0.1235, 5-week WAPE 0.2000, bias -0.0123; corrections p one map, m scale; "
        "v1 test not run")


def test_summary_is_one_line(script):
    assert "\n" not in script.summary(fake_choice(), None)
    assert "\n" not in script.summary(fake_choice(), SimpleNamespace(scores=make_test_scores()))


def test_summary_with_the_test_picks_the_hurdle_consolidated_pooled_row(script):
    line = script.summary(fake_choice(), SimpleNamespace(scores=make_test_scores()))
    assert line.endswith("; corrections p one map, m scale; v1 test (corrected) WAPE 0.2222, bias -0.0333")
    assert line.startswith(script.summary(fake_choice(), None)[:-len("; v1 test not run")])


def test_summary_with_the_test_ignores_row_order(script):
    scores = make_test_scores().iloc[::-1].reset_index(drop=True)
    line = script.summary(fake_choice(), SimpleNamespace(scores=scores))
    assert line.endswith("; v1 test (corrected) WAPE 0.2222, bias -0.0333")


def test_summary_with_the_test_reads_the_scores_index_free(script):
    scores = make_test_scores()
    scores.index = range(1000, 1000 + len(scores))
    assert script.summary(fake_choice(), SimpleNamespace(scores=scores)).endswith(
        "; v1 test (corrected) WAPE 0.2222, bias -0.0333")


@pytest.mark.parametrize("target_bias, text", [
    (-0.03333, "bias -0.0333"), (0.04444, "bias +0.0444"), (0.0, "bias +0.0000"),
    (-1.25, "bias -1.2500")])
def test_summary_test_bias_is_the_same_rows_bias_signed(script, target_bias, text):
    # Every other row carries a different bias (wape - 0.4: 0.101 ... 0.148, all positive and
    # none equal to the target's), so a wrong row would show up in the number or the sign.
    scores = make_test_scores(target_bias=target_bias)
    others = scores[~((scores["method"] == HURDLE) & (scores["company_code"] == CONSOLIDATED)
                      & (scores["horizon"] == POOLED))]
    assert (others["bias"] > 0).all() and not (others["bias"] == target_bias).any()
    for frame in (scores, scores.iloc[::-1]):
        line = script.summary(fake_choice(), SimpleNamespace(scores=frame))
        assert line.endswith(f"; v1 test (corrected) WAPE 0.2222, {text}"), line


def test_summary_test_bias_is_not_the_validation_bias(script):
    # The validation bias (winner row) and the test bias (scores row) are separate numbers.
    line = script.summary(fake_choice(bias=0.0777), SimpleNamespace(
        scores=make_test_scores(target_bias=-0.0111)))
    assert "5-week WAPE 0.2000, bias +0.0777; corrections" in line
    assert line.endswith("v1 test (corrected) WAPE 0.2222, bias -0.0111")
    assert line.count("bias ") == 2


def test_summary_says_uncorrected_once_and_corrected_once(script):
    line = script.summary(fake_choice(), SimpleNamespace(scores=make_test_scores()))
    assert line.count("(uncorrected)") == 1 and line.count("(corrected)") == 1
    assert line.index("validation (uncorrected) WAPE") < line.index("v1 test (corrected) WAPE")
    assert "(corrected)" not in script.summary(fake_choice(), None)


def test_summary_with_the_test_not_the_uncorrected_hurdle(script):
    # "hurdle uncorrected" must not be taken for HURDLE.
    scores = make_test_scores()
    scores = scores[scores["method"] != HURDLE]
    with pytest.raises(IndexError):
        script.summary(fake_choice(), SimpleNamespace(scores=scores))


@pytest.mark.parametrize("bias, text", [(0.012345, "bias +0.0123"), (-0.012345, "bias -0.0123"),
                                        (0.0, "bias +0.0000"), (1.5, "bias +1.5000"),
                                        (-0.00004, "bias -0.0000")])
def test_summary_bias_is_signed_to_four_places(script, bias, text):
    assert f"{text};" in script.summary(fake_choice(bias=bias), None)


def test_summary_wapes_to_four_places(script):
    line = script.summary(fake_choice(wape=0.18765, total_wape=0.1), None)
    assert "validation (uncorrected) WAPE 0.1877, 5-week WAPE 0.1000," in line


def test_summary_nan_five_week_wape_is_printed_as_nan(script):
    # grid_scores gives NaN when there is no HORIZON_TOTAL row.
    assert "5-week WAPE nan," in script.summary(fake_choice(total_wape=float("nan")), None)


@pytest.mark.parametrize("p_kind, m_kind", [("none", "none"), ("per h", "per company"),
                                            ("one map", "by size")])
def test_summary_corrections_are_candidate_name(script, p_kind, m_kind):
    choice = fake_choice(p_kind=p_kind, m_kind=m_kind)
    name = candidate_name(choice.calibration.p_correction, choice.calibration.m_correction)
    assert f"; corrections {name};" in script.summary(choice, None)


@pytest.mark.parametrize("method, window, with_level, text", [
    ("mean", 8, False, "top-ups mean (window 8, with_level False)"),
    ("ridge", 13, True, "top-ups ridge (window 13, with_level True)"),
])
def test_summary_topups(script, method, window, with_level, text):
    choice = fake_choice(method=method, window=window, with_level=with_level)
    assert text in script.summary(choice, None)


def test_summary_changes_nothing(script):
    choice, scores = fake_choice(), make_test_scores()
    winner, before = choice.winner.copy(), scores.copy()
    script.summary(choice, SimpleNamespace(scores=scores))
    pd.testing.assert_series_equal(choice.winner, winner)
    pd.testing.assert_frame_equal(scores, before)


# --- 5. main end to end on test_grid_run.py's tiny grid --------------------------------------

@pytest.fixture(scope="module")
def end_to_end(script, tmp_path_factory):
    from collection_estimation.hurdle import grid_run
    from tests.hurdle.test_grid import g_candidates
    from tests.hurdle.test_grid_run import E2ESettings, e2e_inputs, e2e_settings

    @dataclasses.dataclass(frozen=True)
    class MainSettings(E2ESettings):
        settings_file: object = None

    cache_root = tmp_path_factory.mktemp("run_hurdle_grid")
    base = e2e_settings(cache_root, run_test=True)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    # [grid] run = false: the script must run the grid anyway.
    fields["grid"] = dataclasses.replace(base.grid, run=False)
    fields["data"] = SimpleNamespace(cache_folder=cache_root, corpus_folder=Path("tiny_corpus"))
    settings = MainSettings(**fields, settings_file=Path("tiny.toml"))
    inputs = e2e_inputs()
    seen = {}

    def load_settings(path, root):
        seen["load"] = (path, root)
        return settings

    def prepare_inputs(*args, **kwargs):
        seen["prepare"] = (args, kwargs)
        return inputs, None

    real_run_grid = script.run_grid

    def run_grid(*args, **kwargs):
        seen["run_grid"] = (args, kwargs)
        seen["returned"] = real_run_grid(*args, **kwargs)
        return seen["returned"]

    import contextlib
    import io
    printed = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(printed):
        patch.setattr(grid_run, "grid_candidates", g_candidates)
        patch.setattr(script, "load_settings", load_settings)
        patch.setattr(script, "prepare_inputs", prepare_inputs)
        patch.setattr(script, "run_grid", run_grid)
        script.main(["--settings", "tiny.toml"])
    return SimpleNamespace(settings=settings, inputs=inputs, seen=seen,
                           printed=printed.getvalue().splitlines(), cache_root=cache_root)


def test_end_to_end_returns_a_choice_and_a_test(end_to_end):
    from collection_estimation.hurdle.grid import GridChoice, GridTest
    choice, test = end_to_end.seen["returned"]
    assert isinstance(choice, GridChoice) and isinstance(test, GridTest)


def test_end_to_end_runs_the_grid_with_grid_run_false_and_no_rolled(end_to_end):
    assert end_to_end.settings.grid.run is False
    args, kwargs = end_to_end.seen["run_grid"]
    assert args == (end_to_end.settings, end_to_end.inputs) and kwargs == {}
    assert end_to_end.seen["prepare"] == ((end_to_end.settings,), {})
    assert end_to_end.seen["load"] == (REPO_ROOT / "tiny.toml", REPO_ROOT)


def test_end_to_end_summary_line_matches_the_returned_choice(end_to_end, script):
    choice, test = end_to_end.seen["returned"]
    lines = end_to_end.printed
    assert lines[-2] == script.summary(choice, test)
    winner = choice.winner
    pooled = test.scores[(test.scores["method"] == HURDLE)
                         & (test.scores["company_code"] == CONSOLIDATED)
                         & (test.scores["horizon"] == POOLED)]
    assert len(pooled) == 1
    expected = (f"kept: stage 1 {choice.stage1.params}, stage 2 {choice.stage2.params}, "
                f"top-ups {choice.topups.method} (window {choice.topups.window}, with_level "
                f"{choice.topups.with_level}); validation (uncorrected) WAPE {winner['wape']:.4f}, "
                f"5-week WAPE {winner['total_wape']:.4f}, bias {winner['bias']:+.4f}; corrections "
                f"{candidate_name(choice.calibration.p_correction, choice.calibration.m_correction)}"
                f"; v1 test (corrected) WAPE {pooled['wape'].iloc[0]:.4f}, "
                f"bias {pooled['bias'].iloc[0]:+.4f}")
    assert lines[-2] == expected


def test_end_to_end_printout_order(end_to_end):
    lines = end_to_end.printed
    assert lines[0] == "Settings: tiny.toml"
    assert lines[1] == "Corpus folder: tiny_corpus"
    assert re.fullmatch(r"Steps 1-11: inputs ready in \d+ s \(training table [\d,]+ rows x "
                        r"\d+ columns\)", lines[2]), lines[2]
    rows, columns = end_to_end.inputs.table.shape
    assert lines[2].endswith(f"(training table {rows:,} rows x {columns} columns)")
    assert lines[3].startswith("Step 18: settings grid (U16) over the validation origins")
    assert any(line.startswith("        V1 TEST over the test origins") for line in lines)
    assert lines[-2].startswith("kept: ")
    assert re.fullmatch(r"Done in \d+\.\d min", lines[-1]), lines[-1]


def test_end_to_end_without_rolled_says_candidate_0_is_not_checked(end_to_end):
    assert ("        candidate 0 against step 17: not checked (step 17 did not forecast "
            "with the model)") in end_to_end.printed


def test_end_to_end_caches_its_fits_under_hurdle_grid(end_to_end):
    folder = end_to_end.cache_root / "hurdle_grid"
    assert folder.is_dir() and list(end_to_end.cache_root.iterdir()) == [folder]
