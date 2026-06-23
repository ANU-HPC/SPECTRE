# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Smoke-test the example notebooks by executing every code cell end-to-end.

This catches breakage that library unit tests miss: stale imports, renamed
APIs, or signature changes that only the example notebooks exercise (e.g. a
notebook importing from a module that no longer exists).  Each notebook is run
in its own directory so relative data-file paths (wout, coils) resolve, and the
Matplotlib backend is forced to ``Agg`` so no window is opened.
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")  # headless: never pop up a window
pytest.importorskip("simsopt")  # the make_freeboundary notebooks need simsopt

pytestmark = pytest.mark.simsopt

_REPO = Path(__file__).resolve().parents[1]
_NOTEBOOKS = [
    _REPO / "examples" / "utils" / "make_freeboundary" / "make_freeboundary_ncsx.ipynb",
]


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_without_error(nb_path):
    """Concatenated code cells of the notebook execute without raising."""
    nb = json.loads(nb_path.read_text())
    code = "\n\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    namespace = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(nb_path.parent)
    try:
        exec(compile(code, str(nb_path), "exec"), namespace)
    finally:
        os.chdir(cwd)
