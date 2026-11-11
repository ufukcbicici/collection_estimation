"""Helpers for the hurdle tests that read source files or run code snippets.

`run_snippet` runs a piece of code in a fresh Python process, from the repository root, with
src/ and the repository root on its path: a snippet may replace functions of a module or
print Turkish names, and neither should leak into the test session. The others read a
top-level function of a source file as a syntax tree, without importing the file.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HURDLE = REPO_ROOT / "src" / "collection_estimation" / "hurdle"


def run_snippet(code, timeout=300):
    """Run `code` with `python -c`; the CompletedProcess, its output read as UTF-8."""
    path = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def module_functions(path):
    """The top-level `def`s of the file at `path`, in file order, as syntax trees."""
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    return [node for node in module.body if isinstance(node, ast.FunctionDef)]


def function_tree(path, name):
    """The top-level `def name` of the file at `path`, as a syntax tree."""
    found = [node for node in module_functions(path) if node.name == name]
    assert len(found) == 1, name
    return found[0]


def function_source(path, name):
    """The source of `def name` in `path`, with every run of whitespace made one space."""
    text = Path(path).read_text(encoding="utf-8")
    return " ".join(ast.get_source_segment(text, function_tree(path, name)).split())


def calls_to(tree, name):
    """Every call of the plain name `name` inside `tree`."""
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == name]


def keywords_of(call):
    """A call's keyword arguments as {name: source text}; a `**kwargs` is refused."""
    assert all(k.arg is not None for k in call.keywords), ast.unparse(call)
    return {k.arg: ast.unparse(k.value) for k in call.keywords}
