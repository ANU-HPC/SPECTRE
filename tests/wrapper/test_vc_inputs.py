# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Input-placement round-trip of the direct free-boundary flag (lvcvacuum).

lvcvacuum selects the free-boundary physics formulation and therefore lives in the physics
input group everywhere: TOML ``[physics]``, Fortran ``/physicslist/`` (the ``.sp.end`` restart
namelist), and the ``input/physics`` HDF5 metadata group. The quadrature resolutions
``cteta``/``czeta`` are numerical controls and stay in ``[global]``//``globallist``/
``input/global``.
"""

import glob
import os
import os.path

import f90nml
import h5py
import pytest

from spectre import SPECTRE

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def w7x_vc_instance(tmp_path_factory):
    """Initialize (no solve) the direct-VC W7-X fixture in a scratch directory."""
    workdir = tmp_path_factory.mktemp("vc_inputs")
    cwd = os.getcwd()
    os.chdir(workdir)  # SPECTRE drops the .sp.end and .h5 files in the cwd
    try:
        test = SPECTRE(f"{curr_path}test_freeb_w7x_vc.toml", verbose=False)
        yield test, workdir
    finally:
        os.chdir(cwd)


def test_lvcvacuum_reaches_fortran_physicslist(w7x_vc_instance):
    """The [physics] TOML value arrives in the Fortran input variable."""
    test, _ = w7x_vc_instance
    assert int(test.input_list_mod.lvcvacuum) == 1
    assert int(test.input_list_mod.cteta) == 128
    assert int(test.input_list_mod.czeta) == 128


def test_namelist_places_lvcvacuum_in_physicslist(w7x_vc_instance):
    """The Fortran namelist declares lvcvacuum in /physicslist/ and cteta/czeta in /globallist/."""
    test, workdir = w7x_vc_instance
    test.allglobal_mod.write_spec_namelist()

    example = workdir / "example.sp"
    assert example.exists()
    try:
        nml = f90nml.read(str(example))
        assert nml["physicslist"]["lvcvacuum"] == 1
        assert "lvcvacuum" not in nml["globallist"]
        assert nml["globallist"]["cteta"] == 128
        assert nml["globallist"]["czeta"] == 128
        assert "cteta" not in nml["physicslist"]
    finally:
        example.unlink()  # write_spec_namelist refuses to overwrite an existing file


def test_parameters_readback_keeps_lvcvacuum_in_physics(w7x_vc_instance):
    """The Fortran-to-Python parameter mirror returns lvcvacuum on the physics group."""
    test, _ = w7x_vc_instance
    params = test.parameters
    assert params.physics.lvcvacuum == 1
    assert params.global_.cteta == 128
    assert params.global_.czeta == 128


def test_hdf5_metadata_places_lvcvacuum_in_physics_group(w7x_vc_instance):
    """The output file records lvcvacuum under input/physics and cteta/czeta under input/global."""
    test, workdir = w7x_vc_instance

    test.allglobal_mod.skip_write = False
    test.lib.sphdf5_mod.init_outfile()
    test.lib.sphdf5_mod.mirror_input_to_outfile()
    test.lib.sphdf5_mod.finish_outfile()

    h5_files = glob.glob(str(workdir / "*.h5"))
    assert len(h5_files) == 1, f"expected one output file, found {h5_files}"

    with h5py.File(h5_files[0], "r") as f:
        assert int(f["input/physics/lvcvacuum"][0]) == 1
        assert "lvcvacuum" not in f["input/global"]
        assert int(f["input/global/cteta"][0]) == 128
        assert int(f["input/global/czeta"][0]) == 128
