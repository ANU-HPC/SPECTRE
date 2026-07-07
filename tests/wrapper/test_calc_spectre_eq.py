# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Smoke test for the ``scripts/run/calc_spectre_eq.py`` driver.

Runs the actual driver script in a subprocess for a few iterations on the small
fixed-boundary QA case, with every callback metric enabled. This exercises the
optimizer callback paths that otherwise only run inside a live equilibrium solve
(``callback_default`` -> ``get_freal_fsa_metric`` -> ``compute_coordinates``, and
the per-iface-rms / field-error / jac-health diagnostics), guarding them against
Fortran wrapper signature changes that no other test calls.
"""

import os
import subprocess
import sys

import tomlkit

curr_path = os.path.dirname(__file__) + "/"
script_path = os.path.abspath(curr_path + "../../scripts/run/calc_spectre_eq.py")


def test_calc_spectre_eq_few_iterations(tmp_path):
    """The driver completes a short run and reaches its final status line."""

    with open(curr_path + "qa_7vol_test.toml") as fh:
        doc = tomlkit.parse(fh.read())

    doc["minimization"] = {
        "max_niter": 3,
        "max_nfev": 10,
        "callback_metrics": ["per-iface-rms", "field-error", "jac-health"],
    }

    toml_file = tmp_path / "qa_few_iters.toml"
    toml_file.write_text(tomlkit.dumps(doc))

    # cwd=tmp_path so run artifacts (snapshots, output h5/toml) stay out of the repo
    result = subprocess.run(
        [sys.executable, script_path, str(toml_file)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"calc_spectre_eq.py failed (exit {result.returncode})\n"
        f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
    )
    assert "SPECTRE free-boundary equilibrium done" in result.stdout
