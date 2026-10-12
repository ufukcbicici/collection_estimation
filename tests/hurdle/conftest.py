"""Puts `src/` on the import path for the hurdle tests.

The hurdle package is self-contained, so its tests do not use the older `tests/helpers.py`.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
