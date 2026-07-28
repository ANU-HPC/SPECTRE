# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Policy of the lagged-banded driver in ``optimizer_default``.

The lagged mode switches on a process-global Beltrami factor cache and consumes
Jacobian-vector products that only some configurations can produce. Both are checked here
without a SPECTRE instance: the driver is exercised against a stub exposing exactly the
Fortran state it touches, so the tests stay on the policy (which configurations enable
lagging, and what happens to the cache afterwards) rather than on the physics.
"""

import numpy as np
import pytest

import spectre.force_minimizer as fm
from spectre.input_dict import input_dict

MVOL = 4


class _AllGlobal:
    def __init__(self, cache_on=False):
        self.mvol = MVOL
        self.lbeltrami_factor_cache = cache_on
        self.force_jac_status = np.zeros(MVOL, dtype=np.int32)


class _BeltramiFactorCacheMod:
    def __init__(self):
        self.clear_calls = 0

    def factor_cache_clear(self):
        self.clear_calls += 1


class _FieldObj:
    def __init__(self, lconstraint=3, cache_on=False):
        self.allglobal_mod = _AllGlobal(cache_on)
        self.input_list_mod = input_dict(dict(mpol=4, ntor=4, nfp=1, lconstraint=lconstraint))
        self.lib = input_dict(dict(beltrami_factor_cache_mod=_BeltramiFactorCacheMod()))


class _Comm:
    def Get_size(self):
        return 1


class _Obj:
    """Stand-in for Force_Minimizer, holding what optimizer_default reads."""

    def __init__(self, lconstraint=3, cache_on=False, jvp_lag=3):
        self.field_obj = _FieldObj(lconstraint, cache_on)
        self.comm = _Comm()
        self.freeb_mode = 0
        self.iteration_start = 0
        self.callback = None
        self.min_params = input_dict(
            dict(
                gtol=1e-13,
                xtol=1e-22,
                ftol=1e-13,
                max_nfev=10,
                max_niter=10,
                method="trf",
                rcond_lstsq=None,
                tr_solver="banded",
                jvp_lag=jvp_lag,
                objective="freal",
                inface_repr="fourier",
                mpol_rz=None,
                ntor_rz=None,
                nt_dbb=None,
                nz_dbb=None,
                filter_stellsym=True,
            )
        )


@pytest.fixture()
def captured_lstsq(monkeypatch):
    """Replace the least-squares driver with a recorder; return the recorded calls."""
    calls = {}

    def fake_lstsq(f, x_init, **kwargs):
        calls.update(kwargs)
        return "result"

    monkeypatch.setattr(fm, "lstsq", fake_lstsq)
    return calls


def test_lagging_restores_and_clears_the_factor_cache(captured_lstsq):
    obj = _Obj()

    assert fm.optimizer_default(obj, np.zeros(3), None, None) == "result"

    assert "jvp" in captured_lstsq["tr_options"], "lagged mode should have installed a JVP"
    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False, (
        "the caller's factor-cache setting was not restored"
    )
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 1


def test_factor_cache_is_restored_after_a_failed_optimization(monkeypatch):
    def exploding_lstsq(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fm, "lstsq", exploding_lstsq)
    obj = _Obj()

    with pytest.raises(RuntimeError, match="boom"):
        fm.optimizer_default(obj, np.zeros(3), None, None)

    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 1


def test_a_caller_that_had_the_cache_on_keeps_it(captured_lstsq):
    obj = _Obj(cache_on=True)

    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is True


def test_factor_cache_scope_is_replaceable_by_a_fake(monkeypatch, captured_lstsq):
    events = []

    class FakeCacheScope:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("exit")

        def clear(self):
            events.append("clear")

    fake_scope = FakeCacheScope()
    monkeypatch.setattr(fm, "_make_beltrami_factor_cache_scope", lambda field_obj: fake_scope)

    obj = _Obj()
    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert events == ["enter", "exit"]


def test_lconstraint_2_enables_lagging(captured_lstsq):
    """force_real_jvp linearises the fixed-boundary helicity constraint, so lagging engages."""
    obj = _Obj(lconstraint=2)

    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert "jvp" in captured_lstsq["tr_options"], (
        "force_real_jvp supports fixed-boundary lconstraint = 2 and should be lagged against"
    )
    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 1


def test_free_boundary_does_not_enable_lagging(captured_lstsq, capsys):
    obj = _Obj()
    obj.freeb_mode = 1

    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert "jvp" not in captured_lstsq["tr_options"], (
        "force_real_jvp does not support free-boundary mode and must not be lagged against"
    )
    assert "free-boundary" in capsys.readouterr().out
    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 0


def test_degraded_jacobian_rebuild_warns_and_continues(captured_lstsq, capsys):
    obj = _Obj()
    built = []

    def jac(xdof, obj_):
        built.append(xdof)
        return np.zeros((2, 3))

    fm.optimizer_default(obj, np.zeros(3), None, jac)
    jac_used = captured_lstsq["jac"]

    # a clean build is passed through silently
    obj.field_obj.allglobal_mod.force_jac_status[:] = 0
    jac_used(np.zeros(3), obj)
    assert len(built) == 1
    assert "degraded linear solves" not in capsys.readouterr().out

    # a build whose linearised solves degraded is still returned, with a warning
    obj.field_obj.allglobal_mod.force_jac_status[2] = 1
    result = jac_used(np.zeros(3), obj)
    assert len(built) == 2
    assert result.shape == (2, 3)
    assert "WARNING: the force Jacobian was built from degraded linear solves" in (
        capsys.readouterr().out
    )
