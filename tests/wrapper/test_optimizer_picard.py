# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Loop-structure tests for the frozen-iBns free-boundary driver.

These exercise ``optimizer_picard`` against an analytic residual and a stub minimizer, so
the cycle bookkeeping (budget, refresh cadence, convergence and safety stops) is checked
without paying for a Fortran equilibrium solve. The physics of the refresh itself is
covered by ``test_update_bns.py``.
"""

import re

import numpy as np
import pytest

import spectre.force_minimizer as fm
from spectre.force_minimizer import optimizer_default, optimizer_picard
from spectre.input_dict import input_dict

# Deliberately inconsistent decay fit: converges slowly, so a run is not over before the
# outer loop has had a chance to cycle.
T_DATA = np.linspace(0.0, 3.0, 10)
Y_DATA = 2.0 * np.exp(-1.3 * T_DATA) + 0.05 * np.cos(5.0 * T_DATA)
X_INIT = np.array([1.0, -0.1])


def fun(x, *args):
    """Exponential-decay residual against the wiggly data."""
    return x[0] * np.exp(x[1] * T_DATA) - Y_DATA


def jac(x, *args):
    """Analytic Jacobian of :func:`fun`."""
    e = np.exp(x[1] * T_DATA)
    return np.stack([e, x[0] * T_DATA * e], axis=1)


class _StubFortran:
    """The handful of Fortran globals the driver reads for its banner."""

    def __init__(self):
        self.gbnbld = 0.666
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
            tr_carry_factor=8.0,
            freeb_bnorm_tol=1e-5,
        )
        base.update(params)
        self.min_params = input_dict(base)

        self.picard_freeb = picard_freeb
        self.iteration_start = 0
        self._bns_damping = None

        fortran = _StubFortran()
        self.field_obj = type("_Obj", (), {})()
        self.field_obj.input_list_mod = fortran
        self.field_obj.allglobal_mod = fortran

        # bnserr values handed back by successive refreshes; the last one repeats.
        self._bnserr_seq = list(bnserr_seq) if bnserr_seq else [1.0]
        self.refresh_calls = 0
        self.f_calls = 0
        self.iters_seen = []

    def callback(self, it, nfev, cost, cost_red, stp_nrm, opt, x, obj):
        """Record the reported global iteration indices."""
        self.iters_seen.append(it)

    def refresh_bns(self):
        """Hand back the scripted bnserr sequence, one value per refresh."""
        idx = min(self.refresh_calls, len(self._bnserr_seq) - 1)
        self.refresh_calls += 1
        return self._bnserr_seq[idx]


def _run(obj):
    def f(x, o):
        o.f_calls += 1
        return fun(x)

    def j(x, *a):
        return jac(x)

    return optimizer_picard(obj, X_INIT, f, j)


def test_non_picard_is_refused():
    """Fixed-boundary, lvcvacuum=1 and lbnszero runs belong to optimizer_default, not here."""
    obj = _StubMinimizer(picard_freeb=False, bnserr_seq=[1.0])
    with pytest.raises(ValueError, match="optimizer_default"):
        _run(obj)

    assert obj.refresh_calls == 0 and obj.f_calls == 0


def test_lm_is_refused_before_any_cycle(monkeypatch):
    """MINPACK ignores the budget and the callback protocol: rejected from the outer loop.

    On the single-solve path it stays supported (subject to its own tolerance rules).
    """
    lm = dict(method="lm", gtol=1e-8, xtol=1e-8, ftol=1e-8)
    obj = _StubMinimizer(bnserr_seq=[1.0], **lm)
    with pytest.raises(ValueError, match="trf"):
        _run(obj)
    assert obj.refresh_calls == 0 and obj.f_calls == 0

    monkeypatch.setattr(fm, "_run_single_lstsq", lambda *args: "single solve")
    single = _StubMinimizer(picard_freeb=False, **lm)
    assert optimizer_default(single, X_INIT, None, None) == "single solve"


def test_refresh_happens_once_per_cycle():
    """With cap=2 and a budget of 6, the loop runs 3 cycles and refreshes 3 times."""
    obj = _StubMinimizer(bnserr_seq=[1.0], max_niter=6, inner_iter_cap=2)
    _run(obj)

    assert obj.refresh_calls == 3
    assert max(obj.iters_seen) <= 6


@pytest.mark.parametrize("iteration_start", [0, 2])
def test_iteration_budget_spans_all_cycles(iteration_start):
    """max_niter is the absolute stop index across cycles, also for a resumed run."""
    obj = _StubMinimizer(bnserr_seq=[1.0], max_niter=4, inner_iter_cap=1)
    obj.iteration_start = iteration_start
    r = _run(obj)

    assert iteration_start <= min(obj.iters_seen) and max(obj.iters_seen) <= 4
    # Each iterate is reported once: cycle boundaries must not duplicate a row.
    assert len(obj.iters_seen) == len(set(obj.iters_seen))
    # The result describes the outer problem: out of budget with bnserr above tol; nfev sums
    # the inner solves (the driver's own per-cycle f(x) before each refresh is not one).
    assert r.status == 6 and r.success is False
    assert r.nfev == obj.f_calls - obj.refresh_calls
    assert r.outer_cycles == obj.refresh_calls
    assert r.bnserr == 1.0 and r.bnserr_hist == [1.0] * obj.refresh_calls


def test_converges_once_bnserr_drops_below_tol():
    """A converged inner solve plus a small bnserr ends the run before the budget."""
    obj = _StubMinimizer(
        bnserr_seq=[1.0, 1e-9], max_niter=100, inner_iter_cap=2, freeb_bnorm_tol=1e-5
    )
    r = _run(obj)

    # Stops on the cycle whose refresh first reports a converged bnserr, so long as the
    # inner solve settled -- far short of the 100-iteration budget.
    assert obj.refresh_calls < 10
    assert r.status == 1 and r.success is True and r.bnserr == 1e-9


def test_growing_bnserr_still_terminates():
    """A non-contracting iBns map must warn but never spin forever."""
    obj = _StubMinimizer(bnserr_seq=[1e-3, 1e-2, 1e-1, 1.0], max_niter=4, inner_iter_cap=1)
    r = _run(obj)

    assert max(obj.iters_seen) <= 4
    assert r.status == 6 and r.success is False and r.bnserr == r.bnserr_hist[-1] == 1.0


def test_zero_iteration_cycles_stop_the_loop():
    """Two cycles that cannot advance must trip the safety stop, not loop forever."""
    # gtol is wide enough that the inner solve terminates immediately every time, while
    # bnserr never reaches its tolerance.
    obj = _StubMinimizer(
        bnserr_seq=[1.0], max_niter=1000, inner_iter_cap=5, gtol=1e10, freeb_bnorm_tol=1e-30
    )
    r = _run(obj)

    assert obj.refresh_calls <= 3
    # Every re-entered cycle reports the entry iterate again; only the first may be kept.
    assert obj.iters_seen == [0]
    assert r.status == 0 and r.success is False


def test_dogbox_cycles_and_only_trf_carries_the_trust_radius(monkeypatch):
    """dogbox runs the full outer loop; trf alone hands its radius (scaled) to the next cycle."""
    seen = []
    real_lstsq = fm.lstsq

    def spy(*args, **kwargs):
        r = real_lstsq(*args, **kwargs)
        seen.append((kwargs["delta_init"], r.tr_radius))
        return r

    monkeypatch.setattr(fm, "lstsq", spy)

    obj = _StubMinimizer(bnserr_seq=[1.0], max_niter=4, inner_iter_cap=2, method="dogbox")
    _run(obj)
    assert obj.refresh_calls == 2 and max(obj.iters_seen) <= 4
    assert all(delta is None for delta, _ in seen)

    seen.clear()
    obj = _StubMinimizer(bnserr_seq=[1.0], max_niter=4, inner_iter_cap=2, tr_carry_factor=8.0)
    _run(obj)
    assert seen[0][0] is None and seen[0][1] > 0.0
    assert seen[1][0] == seen[0][1] * 8.0


def test_stall_test_reads_only_the_cycle_that_ran(monkeypatch, capsys):
    """A continued cycle that accepts no step fires no callback and must not pass as settled.

    The stall test reads the step norm the callbacks recorded; left over from the previous
    cycle it would report "settled" for a solve that never stepped.
    """
    real_lstsq = fm.lstsq
    calls = []

    def no_step_in_second_cycle(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            kwargs["max_niter"] = kwargs["iteration_start"]  # re-entered, zero iterations
        return real_lstsq(*args, **kwargs)

    monkeypatch.setattr(fm, "lstsq", no_step_in_second_cycle)
    # Thresholds so loose that any step norm at all reads as stalled.
    obj = _StubMinimizer(
        bnserr_seq=[1.0], max_niter=4, inner_iter_cap=2, step_tol=1e10, cost_red_frac=1e10
    )
    _run(obj)

    stalled = dict(re.findall(r"cycle (\d+) done:.*inner_stalled=(\w+)", capsys.readouterr().out))
    assert stalled["0"] == "True"
    assert stalled["1"] == "False"
