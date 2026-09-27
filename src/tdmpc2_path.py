"""Make the vendored TD-MPC2 importable.

TD-MPC2 uses top-level imports such as ``from common import math``, so its
package directory (not the repo root) must be on ``sys.path``.
"""
import sys
from pathlib import Path

TDMPC2_DIR = Path(__file__).resolve().parents[1] / "third_party" / "tdmpc2" / "tdmpc2"


def add_tdmpc2_to_path() -> None:
	path = str(TDMPC2_DIR)
	if path not in sys.path:
		sys.path.insert(0, path)


add_tdmpc2_to_path()
