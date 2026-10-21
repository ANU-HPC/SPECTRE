# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import os.path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from spectre import SPECTRE, get_xinit_specwrap, remove_file  # noqa: E402
from spectre.jax_wrapper import SpectreJax  # noqa: E402

curr_path = os.path.dirname(__file__) + "/"
sp_fname = f"{curr_path}cyl_manyvol_test.toml"


@pytest.fixture(scope="module")
def sj():
    test = SPECTRE(sp_fname, verbose=False)
    yield SpectreJax(test)
    remove_file(sp_fname + ".end")


def test_force_grad(sj):
    x0 = get_xinit_specwrap(sj.test)
    f, jac = sj._force_and_jac_np(x0)

    grad = jax.grad(lambda x: 0.5 * jnp.sum(sj.force(x) ** 2))(jnp.asarray(x0))

    expected = jac.T @ f
    assert np.allclose(grad, expected, atol=1e-10 * np.max(np.abs(expected)))


def test_equilibrium_pressure_grad(sj):
    mvol = sj.test.allglobal_mod.mvol
    eq = sj.equilibrium(sj.pressure_force)
    p0 = jnp.linspace(1.0, 0.8, mvol)

    x0 = eq(p0)
    assert np.linalg.norm(sj._force_np(x0) + np.asarray(sj.pressure_force(p0))) < 1e-8

    k = int(np.argmax(np.abs(np.asarray(x0)[1:]))) + 1

    def obj(p):
        return eq(p)[k]

    grad = jax.grad(obj)(p0)

    h = 1e-5
    grad_fd = np.array([(obj(p0 + h * e) - obj(p0 - h * e)) / (2 * h) for e in np.eye(mvol)])
    assert np.allclose(grad, grad_fd, rtol=1e-6, atol=1e-6 * np.max(np.abs(grad_fd)))

    # Forward mode agrees with reverse mode.
    p_dot = jnp.arange(1.0, mvol + 1.0)
    _, jvp = jax.jvp(obj, (p0,), (p_dot,))
    assert np.isclose(jvp, grad @ p_dot, rtol=1e-10, atol=1e-14)


@pytest.fixture(scope="module")
def sj_tokamak():
    fname = f"{curr_path}tokamak_3vol.toml"
    test = SPECTRE(fname, verbose=False)
    yield SpectreJax(test)
    remove_file(fname + ".end")


def test_interface_modes(sj_tokamak):
    ag = sj_tokamak.test.allglobal_mod
    rbc, zbs = sj_tokamak.interface_modes(jnp.asarray(get_xinit_specwrap(sj_tokamak.test)))

    assert np.allclose(rbc, ag.irbc[: ag.mn, 1 : ag.mvol].T, rtol=1e-12, atol=1e-14)
    assert np.allclose(zbs, ag.izbs[: ag.mn, 1 : ag.mvol].T, rtol=1e-12, atol=1e-14)


def test_shafranov_shift_sensitivity(sj_tokamak):
    sj = sj_tokamak
    even = jnp.asarray(sj.test.allglobal_mod.im[: sj.test.allglobal_mod.mn] % 2 == 0)
    profile = jnp.array([1.0, 0.5, 0.0])
    x_star = sj.equilibrium(sj.pressure_force)

    def centre(p1):
        # Midplane centre (R_outboard + R_inboard) / 2 of each interface.
        rbc, _ = sj.interface_modes(x_star(p1 * profile))
        return jnp.sum(jnp.where(even, rbc, 0.0), axis=1)

    p1, h = 1.5e-3, 1e-5
    dcentre = jax.jacfwd(centre)(p1)
    dcentre_fd = (centre(p1 + h) - centre(p1 - h)) / (2 * h)

    assert np.all(dcentre > 0)  # pressure pushes the interfaces outward
    # At mpol = 2 the force residual does not vanish, so the implicit derivative is only
    # approximate (see SpectreJax.equilibrium).
    assert np.allclose(dcentre, dcentre_fd, rtol=1e-2)
