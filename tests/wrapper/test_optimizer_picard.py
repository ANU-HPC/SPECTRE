# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Loop-structure tests for the frozen-iBns free-boundary driver.

These exercise ``optimizer_picard`` against an analytic residual and a stub minimizer, so
the cycle bookkeeping (budget, refresh cadence, convergence and safety stops) is checked
without paying for a Fortran equilibrium solve. The physics of the refresh itself is
covered by ``test_update_bns.py``.
"""

import numpy as np
import pytest

from spectre.force_minimizer import optimizer_picard
from spectre.input_dict import input_dict

# Deliberately inconsistent decay fit: converges slowly, so a run is not over before the
# outer loop has had a chance to cycle.
T_DATA = np.linspace(0.0, 3.0, 10)
Y_DATA = 2.0 * np.exp(-1.3 * T_DATA) + 0.05 * np.cos(5.0 * T_DATA)
X_INIT = np.array([1.0, -0.1])


def fun(x, *args):
    return x[0] * np.exp(x[1] * T_DATA) - Y_DATA


def jac(x, *args):
    e = np.exp(x[1] * T_DATA)
    return np.stack([e, x[0] * T_DATA * e], axis=1)


class _StubFortran:
    """The handful of Fortran globals the driver reads for its banner and warnings."""

    def __init__(self):
        self.gbnbld = 0.666
        self.update_bnorm_inbeltrami = False
        self.mvol = 3


class _StubMinimizer:
    """Minimal stand-in for Force_Minimizer, recording how the driver drove it."""

    def __init__(self, picard_freeb=True, bnserr_seq=None, **params):
        base = dict(
            gtol=1e-13,
            xtol=1e-22,
            ftol=1e-13,
            max_nfev=10000,
            max_niter=6,
            method="trf",
            rcond_lstsq=None,
            tr_solver=None,
            jvp_lag=None,
            objective="freal",
            inface_repr="fourier",
            inner_iter_cap=2,
            step_tol=1e-3,
            cost_red_frac=1e-3,
            freeb_bnorm_tol=1e-5,
        )
        base.update(params)
        self.min_params = input_dict(base)

        self.picard_freeb = picard_freeb
        self.iteration_start = 0
        self._field_at = None

        fortran = _StubFortran()
        self.field_obj = type("_Obj", (), {})()
        self.field_obj.input_list_mod = fortran
        self.field_obj.allglobal_mod = fortran

        # bnserr values handed back by successive refreshes; the last one repeats.
        self._bnserr_seq = list(bnserr_seq) if bnserr_seq else [1.0]
        self.refresh_calls = 0
        self.iters_seen = []

    def callback(self, it, nfev, cost, cost_red, stp_nrm, opt, x, obj):
        self.iters_seen.append(it)

    def refresh_bns(self):
        idx = min(self.refresh_calls, len(self._bnserr_seq) - 1)
        self.refresh_calls += 1
        self._field_at = None
        return self._bnserr_seq[idx]


def _run(obj):
    def f(x, o):
        o._field_at = np.asarray(x).copy()
        return fun(x)

    def j(x, *a):
        return jac(x)

    return optimizer_picard(obj, X_INIT, f, j)


@pytest.mark.parametrize("picard_freeb", [False, True])
def test_non_picard_runs_a_single_solve(picard_freeb):
    """Fixed-boundary and lvcvacuum=1 runs must not enter the alternation at all."""
    obj = _StubMinimizer(picard_freeb=picard_freeb, bnserr_seq=[1.0])
    _run(obj)

    if picard_freeb:
        assert obj.refresh_calls > 0
    else:
        assert obj.refresh_calls == 0


def test_refresh_happens_once_per_cycle():
    """With cap=2 and a budget of 6, the loop runs 3 cycles and refreshes 3 times."""
    obj = _StubMinimizer(bnserr_seq=[1.0], max_niter=6, inner_iter_cap=2)
    _run(obj)

    assert obj.refresh_calls == 3
    assert max(obj.iters_seen) <= 6


def test_iteration_budget_spans_all_cycles():
    """max_niter is the total across cycles, not a per-cycle allowance."""
    obj = _StubMinimizer(bnserr_seq=[1.0], max_niter=4, inner_iter_cap=1)
    _run(obj)

    assert max(obj.iters_seen) <= 4
    # Each iterate is reported once: cycle boundaries must not duplicate a row.
    assert len(obj.iters_seen) == len(set(obj.iters_seen))


def test_converges_once_bnserr_drops_below_tol():
    """A converged inner solve plus a small bnserr ends the run before the budget."""
    obj = _StubMinimizer(
        bnserr_seq=[1.0, 1e-9], max_niter=100, inner_iter_cap=2, freeb_bnorm_tol=1e-5
    )
    _run(obj)

    # Stops on the cycle whose refresh first reports a converged bnserr, so long as the
    # inner solve settled -- far short of the 100-iteration budget.
    assert obj.refresh_calls < 10


def test_growing_bnserr_still_terminates():
    """A non-contracting iBns map must warn but never spin forever."""
    obj = _StubMinimizer(bnserr_seq=[1e-3, 1e-2, 1e-1, 1.0], max_niter=4, inner_iter_cap=1)
    _run(obj)

    assert max(obj.iters_seen) <= 4


def test_zero_iteration_cycles_stop_the_loop():
    """Two cycles that cannot advance must trip the safety stop, not loop forever."""
    # gtol is wide enough that the inner solve terminates immediately every time, while
    # bnserr never reaches its tolerance.
    obj = _StubMinimizer(
        bnserr_seq=[1.0], max_niter=1000, inner_iter_cap=5, gtol=1e10, freeb_bnorm_tol=1e-30
    )
    _run(obj)

    assert obj.refresh_calls <= 3
