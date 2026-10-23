"""Check an asymmetric cosine-axis response against an independent difference."""

from pathlib import Path

import numpy as np

from spectre import SPECTRE, force_real, force_real_jac, get_xinit_specwrap
from spectre.file_io.toml_io import (
    input_parameters_from_dict,
    read_input_parameters_from_toml,
    write_input_parameters_to_toml,
)


def circular_case(tmp_path, periods=10, ntor=10):
    """Write a synthetic corrugated circular torus with both parities enabled."""
    p = read_input_parameters_from_toml(Path(__file__).with_name("qa_3vol_light.toml")).model_dump()
    q = p["physics"]
    q.update(
        nfp=1,
        nvol=2,
        mpol=1,
        ntor=ntor,
        lconstraint=0,
        lrad=[4, 4],
        phiedge=-2.9,
        tflux=[0.4, 1.0],
        pflux=[0.0, 0.03],
        mu=[0.1, 0.1],
        pressure=[0.0, 0.0],
        pscale=0.0,
        enforce_stell_sym=False,
        rac=[3.0],
        ras=[0.0],
        zac=[0.0],
        zas=[0.0],
    )
    q["allrzrz"] = {}
    for i, radius in enumerate([0.18, 0.3], 1):
        q["allrzrz"][f"interface_{i}"] = dict(
            rbc={"(0, 0)": 3.0, "(1, 0)": radius, str((1, periods)): 0.01 * radius},
            zbs={"(1, 0)": radius, str((1, periods)): -0.01 * radius},
            rbs={},
            zbc={},
        )
    q.update(q["allrzrz"]["interface_2"])
    for key, other in [("rbc", "rwc"), ("zbs", "zws"), ("rbs", "rws"), ("zbc", "zwc")]:
        q[other] = q[key].copy()
    for key in ["vns", "vnc", "bns", "bnc"]:
        q[key] = {}
    p["numeric"].update(lrzaxis=1, ndiscrete=2, intor=ntor, impol=1)
    p["minimization"].update(resume=False, filter_stellsym=False)
    filename = tmp_path / "circular.toml"
    write_input_parameters_to_toml(input_parameters_from_dict(p), filename)
    return filename


def test_asymmetric_axis_jacobian_direction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    obj = SPECTRE(str(circular_case(tmp_path)), verbose=False)
    x = get_xinit_specwrap(obj)
    direction = np.random.default_rng(9021).normal(size=x.size)
    direction[0] = 0.0
    direction /= np.linalg.norm(direction)
    nt, nz = 16, 160
    force_real(x, obj, nt=nt, nz=nz)
    predicted = force_real_jac(x, obj, nt=nt, nz=nz).reshape(-1, x.size) @ direction
    step = 1e-6
    difference = (
        force_real(x + step * direction, obj, nt=nt, nz=nz).ravel()
        - force_real(x - step * direction, obj, nt=nt, nz=nz).ravel()
    ) / (2 * step)
    np.testing.assert_allclose(predicted, difference, rtol=2e-6, atol=2e-5)
