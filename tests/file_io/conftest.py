# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from pathlib import Path
import pytest

is_parallel = True

try:
    from mpi4py import MPI

    is_parallel = MPI.COMM_WORLD.Get_size() > 1
except ImportError:
    is_parallel = False

# Directory containing this conftest (tests/file_io)
_THIS_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items):
    """
    This runs after pytest collects all tests.
    We skip only the tests that live under the file_io folder when running
    in parallel (preventing clobbering of the hdd).
    """
    if not is_parallel:
        return

    skip_marker = pytest.mark.skip(reason="Skipping file input/output tests when running parallel")

    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        # Only skip if the test file is inside this directory
        if str(item_path).startswith(str(_THIS_DIR)):
            item.add_marker(skip_marker)
