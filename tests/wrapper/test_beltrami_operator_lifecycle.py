# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Ownership and lifecycle tests for the opaque Beltrami operator and cache manager.

``operator_cache_lifecycle_selftest`` (fortran_src/beltrami_operator_test_support_mod.F90)
checks the move-only handle discipline without touching numerics: a move leaves its source
empty, a cache insert takes ownership from the caller, acquire rebinds the handle on a key hit
and leaves it empty on a resolution mismatch or after invalidation, an insert into a disabled
cache leaves ownership with the caller, and clear releases every slot while the diagnostics
counters account for each hit, miss, insertion, and invalidation. The row correction is part of
the handle payload, so it must survive moves and cache round trips and be dropped by release
and every rebind; the checked setter must reject malformed corrections through its validation
status instead of storing them. Each check adds a distinct bit to the returned status, so zero
means every check passed and any other value identifies the ones that failed. The handles need
no equilibrium, so we call the wrapped module directly.
"""

import spectre4py_wrap


def test_operator_and_factor_cache_move_lifecycle():
    """Moves leave one owner, stale keys miss, and clear releases every slot."""
    status = spectre4py_wrap.beltrami_operator_test_support_mod.operator_cache_lifecycle_selftest()

    assert status == 0
