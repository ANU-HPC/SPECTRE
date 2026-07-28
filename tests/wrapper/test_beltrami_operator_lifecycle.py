# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Ownership and lifecycle tests for the opaque Beltrami operator and cache manager."""

import spectre4py_wrap


def test_operator_and_factor_cache_move_lifecycle():
    """Moves leave one owner, stale keys miss, and clear releases every slot."""
    status = spectre4py_wrap.beltrami_operator_test_support_mod.operator_cache_lifecycle_selftest()

    assert status == 0
