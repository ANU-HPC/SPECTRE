# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""
Test script to compare oversized_guess.toml (with many modes beyond mpol) vs input.toml (with only mpol modes).
Both should produce identical interface arrays since modes beyond mpol should be ignored.
"""

import numpy as np
import os.path

from spectre import SPECTRE

curr_path = os.path.dirname(__file__) + "/"


def test_oversized_guess_vs_normal_input():
    """
    Compare oversized_guess.toml (many modes in allrzrz beyond mpol) vs input.toml (only mpol modes).
    They should produce identical iRbc, iZbs, iRbs, iZbc arrays.
    """

    # Load oversized_guess first and extract arrays
    test_oversized = SPECTRE(curr_path + "oversized_guess.toml", verbose=False)
    irbc_oversized = np.copy(test_oversized.allglobal_mod.irbc)
    izbs_oversized = np.copy(test_oversized.allglobal_mod.izbs)
    irbs_oversized = np.copy(test_oversized.allglobal_mod.irbs)
    izbc_oversized = np.copy(test_oversized.allglobal_mod.izbc)

    # Now load input.toml and extract arrays
    test_normal = SPECTRE(curr_path + "input.toml", verbose=False)
    irbc_normal = np.copy(test_normal.allglobal_mod.irbc)
    izbs_normal = np.copy(test_normal.allglobal_mod.izbs)
    irbs_normal = np.copy(test_normal.allglobal_mod.irbs)
    izbc_normal = np.copy(test_normal.allglobal_mod.izbc)

    # Compare arrays
    np.testing.assert_allclose(
        irbc_oversized,
        irbc_normal,
        atol=1e-14,
        rtol=1e-14,
        err_msg="iRbc arrays differ between oversized_guess and input",
    )
    np.testing.assert_allclose(
        izbs_oversized,
        izbs_normal,
        atol=1e-14,
        rtol=1e-14,
        err_msg="iZbs arrays differ between oversized_guess and input",
    )
    np.testing.assert_allclose(
        irbs_oversized,
        irbs_normal,
        atol=1e-14,
        rtol=1e-14,
        err_msg="iRbs arrays differ between oversized_guess and input",
    )
    np.testing.assert_allclose(
        izbc_oversized,
        izbc_normal,
        atol=1e-14,
        rtol=1e-14,
        err_msg="iZbc arrays differ between oversized_guess and input",
    )


if __name__ == "__main__":
    test_oversized_guess_vs_normal_input()
