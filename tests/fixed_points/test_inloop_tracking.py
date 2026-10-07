# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""In-process ``Force_Minimizer`` runs with fixed-point tracking.

The subprocess driver test (tests/scripts/test_calc_spectre_eq.py) covers axis-only tracking
through scripts/run/calc_spectre_eq.py; these tests drive ``Force_Minimizer`` directly so the
island mode and the ``postprocess(run_spec=False)`` path are covered too.
"""

from pathlib import Path

import h5py
import numpy as np

from spectre import Force_Minimizer
from spectre.file_io import read_input_parameters_from_toml, write_input_parameters_to_toml
from spectre.file_io.input_parameters import MinimizationParameters

QA_TOML = Path(__file__).resolve().parents[1] / "wrapper" / "qa_3vol_light.toml"
NCSX_FIXB_TOML = Path(__file__).resolve().parents[1] / "test_files" / "ncsx_fixb.toml"
TFLUX3_ISLAND_IN_VOL3 = 0.68  # the 3/5 surface sits in volume 3 (see test_fixed_pt_finder.py)


def _minimize(tmp_path, toml_name, params):
    """Write ``params`` to ``tmp_path``, run a ``Force_Minimizer`` on it, return the minimizer."""
    case = tmp_path / toml_name
    write_input_parameters_to_toml(params, str(case))
    obj = Force_Minimizer(str(case))
    obj.minimize()
    return obj


def test_postprocess_without_final_solve_writes_fp(tmp_path, monkeypatch):
    """``postprocess(run_spec=False)`` must still record ``/iterations/fp``, and re-writing the
    iteration info (a resumed or repeated postprocess) must not crash on the existing group.
    """
    monkeypatch.chdir(tmp_path)  # the solver writes its .h5 and xs_fixb/ to the CWD
    params = read_input_parameters_from_toml(str(QA_TOML))
    params = params.model_copy(
        update={"minimization": MinimizationParameters(max_niter=2, track_axis=True)}
    )
    obj = _minimize(tmp_path, "qa_fp_nospec.toml", params)
    obj.postprocess(save_pkl=False, save_json=False, run_spec=False)

    h5_path = tmp_path / "qa_fp_nospec.h5"  # no _opt.h5 exists without the final solve
    assert h5_path.is_file()
    with h5py.File(h5_path, "r") as f:
        fp = f["iterations/fp"]
        assert np.isfinite(fp["axis_rz"][0]).all(), "axis position not recorded"

    # Idempotency: writing the iteration info again (e.g. a repeated postprocess) must replace,
    # not crash on, the existing /iterations group.
    obj._save_iter_info_to_h5()
    with h5py.File(h5_path, "r") as f:
        assert np.isfinite(f["iterations/fp"]["axis_rz"][0]).all()


def test_inloop_island_mode_tracks_through_minimization(tmp_path, monkeypatch):
    """Island-mode tracking through a real (2-iteration) minimization records the 3/5 chain.

    The fixb NCSX case with the vol-3/4 interface at tflux3=0.68 hosts the 3/5 chain in volume 3
    (see test_fixed_pt_finder.py). With ``track_islands`` on, every committed row must carry a
    finite island position, a provenance, and the pinned resonance; the first committed row is
    necessarily a cold discovery.
    """
    monkeypatch.chdir(tmp_path)
    params = read_input_parameters_from_toml(str(NCSX_FIXB_TOML))
    tflux = list(params.physics.tflux)
    tflux[3] = TFLUX3_ISLAND_IN_VOL3
    params = params.model_copy(
        update={
            "physics": params.physics.model_copy(update={"tflux": tflux}),
            "minimization": MinimizationParameters(
                max_niter=2, track_islands=True, island_volume=3, island_m=5, island_n=3
            ),
        }
    )
    obj = _minimize(tmp_path, "ncsx_isl.toml", params)
    obj.postprocess(save_pkl=False, save_json=False)

    with h5py.File(tmp_path / "ncsx_isl_opt.h5", "r") as f:
        fp = f["iterations/fp"]
        no_fp = fp["no_fp"][:]
        island_rz = fp["island_rz"][:]
        provenance = fp["provenance"][:]
        iota = fp["iota"][:]
        kind = fp["kind"][:]
        assert fp.attrs["island_m"] == 5 and fp.attrs["island_n"] == 3
        assert fp.attrs["volume_start"] == 3

    committed = ~no_fp
    assert committed[0], "the 3/5 chain must be found at the first tracked iterate"
    assert provenance[0] == b"cold", "the first acquisition is necessarily a cold discovery"
    assert np.isfinite(island_rz[committed]).all(), "committed rows must have finite (R, Z)"
    assert np.isnan(island_rz[~committed]).all(), "no_fp rows must stay NaN"
    for p, k in zip(provenance[committed], kind[committed], strict=True):
        assert p in (b"cold", b"follow"), f"unexpected provenance {p!r}"
        assert k in (b"o", b"x", b""), f"unexpected kind {k!r}"
    assert (np.abs(np.abs(iota[committed]) - 3 / 5) < 5e-3).all(), (
        "every committed row must sit on the pinned 3/5 resonance"
    )
