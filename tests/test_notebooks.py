# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Smoke-test the self-contained ``examples/utils`` notebooks end-to-end.

This catches breakage that library unit tests miss: stale imports, renamed
APIs, or signature changes that only the example notebooks exercise (e.g. a
notebook importing from a module that no longer exists).  Only the ``utils``
notebooks are covered — the analysis/optimization examples load large
pre-computed result files that are not checked into the repo.

Each notebook runs inside a throwaway *copy* of its own directory, so it can
read its data files (coils, wout, ...) while every file it writes lands in the
pytest tmp dir and is discarded — nothing is left behind in the examples tree.
The Matplotlib backend is forced to ``Agg`` so no window opens, and IPython line
magics / shell escapes (``%...`` / ``!...``) are stripped so cells run as plain
Python.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")  # headless: never pop up a window
pytest.importorskip("simsopt")  # the example notebooks need simsopt

pytestmark = pytest.mark.simsopt

_REPO = Path(__file__).resolve().parents[1]
_NOTEBOOKS = sorted(
    p
    for p in (_REPO / "examples" / "utils").rglob("*.ipynb")
    if ".ipynb_checkpoints" not in p.parts
)


def _notebook_code(nb_path):
    """Concatenate a notebook's code cells, dropping IPython line magics / shell escapes."""
    nb = json.loads(nb_path.read_text())
    blocks = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        kept = [
            ln
            for ln in "".join(cell["source"]).splitlines()
            if not ln.lstrip().startswith(("%", "!"))
        ]
        blocks.append("\n".join(kept))
    return "\n\n".join(blocks)


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=lambda p: str(p.relative_to(_REPO)))
def test_notebook_runs_without_error(nb_path, tmp_path, monkeypatch):
    """Concatenated code cells of the notebook execute without raising.

    Runs in an isolated copy of the notebook's directory so reads work and
    writes are thrown away with the tmp dir.
    """
    workdir = tmp_path / nb_path.parent.name
    shutil.copytree(
        nb_path.parent,
        workdir,
        ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"),
    )
    monkeypatch.chdir(workdir)
    exec(compile(_notebook_code(nb_path), str(nb_path), "exec"), {"__name__": "__main__"})
