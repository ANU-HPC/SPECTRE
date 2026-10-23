# SPDX-FileCopyrightText: 2026 Christopher Albert
# SPDX-License-Identifier: MIT
"""Independent curl and coordinate-rotation oracles for both vector-potential parities."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spectre import SPECTRE, SPECTREout, force_real, get_xinit_specwrap
from spectre.field import (
    bfield_flow_nonsing,
    bfield_flow_sing,
    bfield_flow_tan_nonsing,
    bfield_flow_tan_sing,
)
from spectre.fixed_points import FixedPointClassifier


@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("parity", ["sine", "mixed"])
def test_manufactured_curl_and_tangent(singular, parity):
    # A_s=0; one sin(2 theta-phi) perturbation in each tangential potential.
    # Explicit polynomials, independent of SPECTRE's radial-basis evaluators.
    lrad = 4
    im, inn = np.array([0, 2]), np.array([0, 1])
    aze, ate, azo, ato = [np.zeros((2, lrad + 1)) for _ in range(4)]
    ate[0, 2 if singular else 1] = 2.0
    azo[1, 4] = 0.017
    ato[1, 4] = -0.023
    if parity == "mixed":
        aze[1, 4], ate[1, 4] = 0.011, 0.013

    def potential(x):
        s, theta, phi = x
        r = (s + 1) / 2
        radial = (4 * r**4 - 3 * r**2) / 5 if singular else (8 * s**4 - 8 * s**2) / 5
        background = 2 * r**2 / 3 if singular else (s + 1) / 2
        angle = 2 * theta - phi
        return np.array(
            [
                0.0,
                2 * background + radial * (ate[1, 4] * np.cos(angle) + ato[1, 4] * np.sin(angle)),
                radial * (aze[1, 4] * np.cos(angle) + azo[1, 4] * np.sin(angle)),
            ]
        )

    def curl(x):
        h = 2e-5
        da = np.column_stack(
            [(potential(x + h * e) - potential(x - h * e)) / (2 * h) for e in np.eye(3)]
        )
        return np.array([da[2, 1] - da[1, 2], -da[2, 0], da[1, 0]])

    flow = bfield_flow_sing if singular else bfield_flow_nonsing
    tangent = bfield_flow_tan_sing if singular else bfield_flow_tan_nonsing
    args = (aze, ate, azo, ato, im, inn, lrad, 2)
    # The reader includes the chart Jacobian; flow ratios cancel it.
    out = object.__new__(SPECTREout)
    out.input = SimpleNamespace(
        physics=SimpleNamespace(Lrad=[lrad], Mpol=2, Igeometry=3 if singular else 1)
    )
    out.output = SimpleNamespace(im=im, in_=inn)
    out.vector_potential = SimpleNamespace(Aze=[aze], Ate=[ate], Azo=[azo], Ato=[ato])
    out.get_jacobian = lambda vol, s, t, z: np.full((len(s), len(t), len(z)), 1.7)
    for x in (np.array([-0.37, 0.43, 0.29]), np.array([0.24, 1.11, 0.81])):
        b = curl(x)
        actual = out.get_field_contrav(0, [x[0]], [x[1]], [x[2]])[:, 0, 0, 0]
        np.testing.assert_allclose(actual * 1.7, b, rtol=2e-8, atol=1e-10)
        velocity = np.zeros(2)
        flow(velocity, x[2], x[:2], *args)
        np.testing.assert_allclose(velocity, b[:2] / b[2], rtol=2e-8, atol=1e-10)
        result = np.zeros(6)
        tangent(result, x[2], np.r_[x[:2], 1.0, 0.0, 0.0, 1.0], *args)
        h = 2e-4

        def direction(y):
            field = curl(y)
            return field[:2] / field[2]

        jac = np.column_stack(
            [(direction(x + h * e) - direction(x - h * e)) / (2 * h) for e in np.eye(3)[:2]]
        )
        np.testing.assert_allclose(result[:2], velocity, atol=1e-13)
        np.testing.assert_allclose(result[2:].reshape(2, 2, order="F"), jac, rtol=2e-6, atol=2e-8)


@pytest.fixture(scope="module")
def rotated_qa():
    path = Path(__file__).resolve().parents[1] / "wrapper/qa_3vol_light.toml"
    obj = SPECTRE(str(path))
    force_real(get_xinit_specwrap(obj), obj)
    original = obj.create_spectreout()
    rotated = deepcopy(original)
    rotated.input.physics.enforce_stell_sym = False
    delta = 0.37
    c, s = np.cos(original.output.im * delta), np.sin(original.output.im * delta)
    for even, odd in (("Ate", "Ato"), ("Aze", "Azo")):
        for vol, arr in enumerate(getattr(original.vector_potential, even)):
            getattr(rotated.vector_potential, even)[vol] = arr * c[:, None]
            getattr(rotated.vector_potential, odd)[vol] = arr * s[:, None]
    rotated.output.Rbc = original.output.Rbc * c
    rotated.output.Rbs = original.output.Rbc * s
    rotated.output.Zbs = original.output.Zbs * c
    rotated.output.Zbc = -original.output.Zbs * s
    return original, rotated, delta


@pytest.mark.parametrize("vol", [0, 1])
def test_rotated_reader_and_fieldlines(rotated_qa, vol):
    original, rotated, delta = rotated_qa
    radial, theta, phi = [-0.5, 0.1], np.array([0.41, 1.23]), [0.0, 0.31]
    for method in ("get_field_contrav", "get_field_cylindrical", "get_field_mod", "get_metric"):
        np.testing.assert_allclose(
            getattr(rotated, method)(vol, radial, theta + delta, phi),
            getattr(original, method)(vol, radial, theta, phi),
            rtol=1e-11,
            atol=1e-11,
        )
    # Close to the axis the QA trajectories remain inside the axis volume.
    seed = [-0.95, 0.1] if vol == 0 else [0.0, 0.1]
    options = dict(num_phi_planes=8, num_transits=3, method="DOP853", rtol=1e-11, atol=1e-11)
    R, Z = original.get_rz_coords(vol, *seed, 0.0)
    recovered = rotated.get_st_coords(vol, R, Z, 0.0, s_guess=seed[0], t_guess=seed[1] + delta)
    np.testing.assert_allclose(recovered, np.array(seed) + [0, delta], atol=1e-10)
    a = original.trace_one_fieldline(np.array(seed), vol, **options)
    b = rotated.trace_one_fieldline(np.array(seed) + [0, delta], vol, **options)
    assert a[3] and b[3]
    np.testing.assert_allclose(a[0], b[0] - np.array([[0], [delta]]), atol=2e-8)
    np.testing.assert_allclose(a[1], b[1], atol=2e-8)


def test_rotated_fixed_point_and_classification(rotated_qa):
    original, rotated, delta = rotated_qa
    r, z, s, theta = original.find_axis(guess=[-0.9, 0.0], max_repeats=0)
    a = FixedPointClassifier(original, 0, 1)
    b = FixedPointClassifier(rotated, 0, 1)
    fp = b.refine([s, theta + delta])
    assert fp is not None and fp["residual"] < 1e-8
    np.testing.assert_allclose([fp["R"], fp["Z"]], [r, z], atol=1e-7)
    np.testing.assert_allclose(
        [fp["s"], fp["theta"]], [s, (theta + delta) % (2 * np.pi)], atol=1e-6
    )
    ga, gb = a.greene(s, theta), b.greene(s, theta + delta)
    assert ga["kind"] == gb["kind"] == "O"
    for key in ("trace", "det", "disc", "residue"):
        assert gb[key] == pytest.approx(ga[key], abs=2e-7)
