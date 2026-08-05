# SPDX-FileCopyrightText: 2026 Christopher Albert
#
# SPDX-License-Identifier: MIT

import h5py
import numpy as np
import pytest

from spectre import SPECTRE, SPECTREout, force_real, get_xinit_specwrap

INPUT_FILE = "tests/compare/G3V3L3Fi/input.toml"
COMPONENTS = {"BR": 0, "BZ": 1, "Bp": 2}


@pytest.fixture(scope="module")
def toroidal_grid():
    spectre = SPECTRE(INPUT_FILE, verbose=False)
    force_real(get_xinit_specwrap(spectre), spectre)
    output = SPECTREout.from_spectre(spectre)

    mvol = int(spectre.allglobal_mod.mvol)
    nt = int(spectre.allglobal_mod.nt)
    nz = int(spectre.allglobal_mod.nz)
    ntz = nt * nz
    ngrid = int(spectre.input_list_mod.ngrid)
    if ngrid < 0:
        sum_lrad = int(np.sum(spectre.input_list_mod.lrad[:mvol] + 1))
    else:
        sum_lrad = (ngrid + 1) * mvol

    expected = {name: np.empty((sum_lrad, ntz)) for name in COMPONENTS}
    valid = np.ones((sum_lrad, ntz), dtype=bool)
    start = 0
    theta = np.arange(nt) * 2.0 * np.pi / nt
    zeta = np.arange(nz) * float(spectre.allglobal_mod.pi2nfp) / nz

    for lvol in range(mvol):
        ngrid_local = int(spectre.input_list_mod.lrad[lvol]) if ngrid < 0 else ngrid
        stop = start + ngrid_local + 1
        radial = np.linspace(-1.0, 1.0, ngrid_local + 1)

        first_valid = 1 if lvol == 0 else 0
        field = output.get_field_cylindrical(lvol, radial[first_valid:], theta, zeta)
        for name, component in COMPONENTS.items():
            expected[name][start + first_valid : stop] = field[component].reshape(
                (-1, ntz), order="F"
            )

        if first_valid:
            valid[start] = False
        start = stop

    return spectre, expected, valid, sum_lrad, ntz


def _assert_physical_components(actual, expected, valid):
    for name in COMPONENTS:
        np.testing.assert_allclose(actual[name][valid], expected[name][valid], atol=1.0e-14)


def test_wrapper_grid_uses_physical_cylindrical_components(toroidal_grid):
    spectre, expected, valid, sum_lrad, ntz = toroidal_grid
    arrays = [np.empty((sum_lrad, ntz), order="F") for _ in range(6)]

    spectre.wrapper_funcs_mod.get_grid_data(sum_lrad, *arrays)

    actual = {"BR": arrays[3], "Bp": arrays[4], "BZ": arrays[5]}
    _assert_physical_components(actual, expected, valid)


def test_hdf5_grid_uses_physical_cylindrical_components(toroidal_grid, tmp_path):
    spectre, expected, valid, _, _ = toroidal_grid
    output_file = tmp_path / "grid.h5"
    spectre.allglobal_mod.filename_h5 = str(output_file)
    spectre.allglobal_mod.skip_write = False

    spectre.lib.sphdf5_mod.init_outfile()
    spectre.lib.sphdf5_mod.write_grid()
    spectre.lib.sphdf5_mod.finish_outfile()

    with h5py.File(output_file) as h5file:
        actual = {name: h5file[f"grid/{name}"][()].T for name in COMPONENTS}
    _assert_physical_components(actual, expected, valid)
