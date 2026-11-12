"""Run the hurdle model's settings grid (U16) alone, e.g. on the compute machine.

    python run_hurdle_grid.py --settings settings/<file>.toml

Without --settings, the file is the one the environment variable HURDLE_SETTINGS names, else
settings/synthetic.toml (so PyCharm's Run button works too). A relative settings path, and
the relative paths inside the file, are read from the repo root, wherever the script is
started from.

Two calls: `inputs.prepare_inputs` (steps 1-11, without their printouts) and
`grid_run.run_grid` (step 18, with its printouts), then one summary line. [grid] run is not
read: running the grid is what this script is for. [grid] run_test decides the v1 test, as in
the pipeline. Every fit is cached in the cache folder's hurdle_grid/, shared with the
pipeline: copy that folder back and a rerun on another machine rebuilds the choice without
refitting. A failed check stops the run with its message and a non-zero exit code.
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# The package lives under src/. Adding it here lets this script run from PyCharm or a
# terminal without installing the project.
sys.path.insert(0, str(REPO_ROOT / "src"))

# Customer names are Turkish; a Windows console would otherwise crash printing "İ".
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collection_estimation.hurdle.calibrate import candidate_name  # noqa: E402
from collection_estimation.hurdle.evaluate import HURDLE, POOLED  # noqa: E402
from collection_estimation.hurdle.forecast import CONSOLIDATED  # noqa: E402
from collection_estimation.hurdle.grid_run import run_grid  # noqa: E402
from collection_estimation.hurdle.inputs import prepare_inputs  # noqa: E402
from collection_estimation.hurdle.settings import load_settings  # noqa: E402

DEFAULT_SETTINGS_FILE = REPO_ROOT / "settings" / "synthetic.toml"


def settings_file(argv=None) -> Path:
    """The settings file: --settings, else HURDLE_SETTINGS (if set and not empty), else
    settings/synthetic.toml. A relative path is read from the repo root, wherever the script
    is started from."""
    parser = argparse.ArgumentParser(description="Run the hurdle model's settings grid alone.")
    parser.add_argument("--settings", type=Path, default=None,
                        help="the settings TOML, relative to the repo root or absolute "
                             "(default: HURDLE_SETTINGS, else settings/synthetic.toml)")
    chosen = parser.parse_args(argv).settings
    if chosen is None:
        chosen = Path(os.environ.get("HURDLE_SETTINGS") or DEFAULT_SETTINGS_FILE)
    return REPO_ROOT / chosen


def summary(choice, test) -> str:
    """One line: the settings kept, their validation scores, the calibration and, if run, the
    v1 test's consolidated WAPE."""
    winner = choice.winner
    calibration = choice.calibration
    line = (f"kept: stage 1 {choice.stage1.params}, stage 2 {choice.stage2.params}, top-ups "
            f"{choice.topups.method} (window {choice.topups.window}, with_level "
            f"{choice.topups.with_level}); validation WAPE {winner['wape']:.4f}, 5-week WAPE "
            f"{winner['total_wape']:.4f}, bias {winner['bias']:+.4f}; corrections "
            f"{candidate_name(calibration.p_correction, calibration.m_correction)}")
    if test is None:
        return line + "; v1 test not run"
    scores = test.scores
    pooled = scores[(scores["method"] == HURDLE) & (scores["company_code"] == CONSOLIDATED)
                    & (scores["horizon"] == POOLED)]
    return line + f"; v1 test WAPE {pooled['wape'].iloc[0]:.4f}"


def main(argv=None):
    started = time.perf_counter()
    settings = load_settings(settings_file(argv), REPO_ROOT)
    print(f"Settings: {settings.settings_file}")
    print(f"Corpus folder: {settings.data.corpus_folder}")

    # Steps 1-11, in hurdle/inputs.py; their checks run inside, their printouts are the
    # pipeline's alone.
    inputs, _ = prepare_inputs(settings)
    print(f"Steps 1-11: inputs ready in {time.perf_counter() - started:.0f} s "
          f"(training table {len(inputs.table):,} rows x {inputs.table.shape[1]} columns)")

    # Step 18, in hurdle/grid_run.py: the grid, its choice and, with [grid] run_test, the v1
    # test. Step 17 is not run here, so candidate 0 is not checked against it.
    choice, test = run_grid(settings, inputs)

    print(summary(choice, test))
    print(f"Done in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
