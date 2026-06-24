# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for the ``spectre_obj.inspect`` plotting namespace (``spectre.core.inspect``)."""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: no display needed for the plot smoke tests
import matplotlib.pyplot as plt  # noqa: E402

from spectre import SPECTRE, beltrami_error_grid, remove_file  # noqa: E402

TEST_FILES = Path(__file__).parent.parent / "test_files"

# Free-boundary, toroidal (igeometry=3) rotating-ellipse case, light resolution.
FNAME = str(TEST_FILES / "rot_ell_light.toml")

# Fixed-boundary, toroidal QA case — fast to reach a solved field for the Beltrami plots.
QA_FNAME = str(Path(__file__).parent / "qa_7vol_test.toml")


@pytest.fixture
def test():
    """A fresh SPECTRE instance, with figures and output files cleaned up afterwards."""
    obj = SPECTRE(FNAME, verbose=False)
    yield obj
    plt.close("all")
    remove_file(FNAME + ".end")


@pytest.fixture
def solved():
    """A SPECTRE instance with a solved field (needed by the Beltrami-error plots)."""
    obj = SPECTRE(QA_FNAME, verbose=False)
    obj.create_spectreout(solve_field=True)
    yield obj
    plt.close("all")
    remove_file(QA_FNAME + ".end")


@pytest.fixture
def qa():
    """A fresh, unsolved SPECTRE instance (fast fixed-boundary QA case)."""
    obj = SPECTRE(QA_FNAME, verbose=False)
    yield obj
    plt.close("all")
    remove_file(QA_FNAME + ".end")


def test_plot_infaces_interactive(test):
    """plot_infaces(interactive=True) returns (fig, mainax) with a zetan slider."""
    fig, mainax = test.inspect.plot_infaces(interactive=True)
    assert hasattr(fig, "_zetan_slider")
    assert mainax in fig.axes
    assert plt.get_fignums()


def test_plot_infaces_interactive_respects_ax(test):
    """plot_infaces(interactive=True, ax=...) draws into the provided ax."""
    f, ax = plt.subplots()
    fig, mainax = test.inspect.plot_infaces(ax=ax, interactive=True)
    assert fig is f
    assert mainax is ax


def test_interactive_view_only_expands(test):
    """Sliding through angles only ever expands the (vertical) view limit, never shrinks."""
    fig, mainax = test.inspect.plot_infaces(interactive=True)
    slider = fig._zetan_slider
    yw = []
    for v in (0.0, 0.25, 0.5, 0.1):  # last value revisits a narrow plane
        slider.set_val(v)
        y0, y1 = mainax.get_ylim()
        yw.append(y1 - y0)
    assert all(yw[i] <= yw[i + 1] + 1e-9 for i in range(len(yw) - 1))


def test_plot_grid_interactive(test):
    """plot_grid(interactive=True) returns (fig, mainax) with a zetan slider."""
    fig, mainax = test.inspect.plot_grid(interactive=True)
    assert hasattr(fig, "_zetan_slider")
    assert mainax in fig.axes
    assert plt.get_fignums()


def test_plot_jacobian_min_val(test):
    """min_val sets the lower bound of the Jacobian colour scale."""
    test.inspect.plot_jacobian(min_val=0.5)
    mappable = list(plt.gca().collections)[0]
    assert mappable.norm.vmin == 0.5


def test_interactive_color_scale_only_expands(test):
    """The scalar colour scale accumulates expand-only across angles in interactive mode."""
    fig, mainax = test.inspect.plot_jacobian(interactive=True)
    slider = fig._zetan_slider
    vmax = []
    for v in (0.0, 0.25, 0.5, 0.1):
        slider.set_val(v)
        vmax.append(list(mainax.collections)[0].norm.vmax)
    assert all(vmax[i] <= vmax[i + 1] + 1e-9 for i in range(len(vmax) - 1))


@pytest.mark.parametrize("method", ["plot_infaces", "plot_jacobian", "plot_grid"])
def test_plot_methods_are_bound_and_delegate(test, method):
    """The xin-based plot helpers live on the ``inspect`` namespace and run headless."""
    assert callable(getattr(test.inspect, method))
    getattr(test.inspect, method)(test.xin)
    assert plt.get_fignums()  # something was actually drawn


def test_plot_methods_default_xin(test):
    """The xin argument defaults to the instance's current ``xin``."""
    test.inspect.plot_infaces()
    assert plt.get_fignums()


def test_plot_true_mesh_returns_mesh(test):
    """plot_true_mesh is on the namespace and returns the (R, Z) mesh arrays."""
    rij, zij = test.inspect.plot_true_mesh(vvol=0)
    assert rij.shape == zij.shape
    assert rij.ndim == 3


# --- Jacobian (purely geometric, no field solve needed) ---------------------------------


def test_plot_jacobian_vvol_and_zetan(test):
    """plot_jacobian accepts vvol='all'/a single int and a zetan cross-section."""
    test.inspect.plot_jacobian(vvol="all", zetan=0.25)
    assert plt.get_fignums()
    plt.close("all")
    test.inspect.plot_jacobian(vvol=0)
    assert plt.get_fignums()


def test_plot_worst_jacobian(test):
    """plot_worst_jacobian scans toroidal planes and draws the worst cut."""
    test.inspect.plot_worst_jacobian()
    assert plt.get_fignums()


def test_plot_jacobian_interactive_builds_slider(test):
    """interactive=True returns (fig, mainax) and carries the zetan slider (built headlessly)."""
    fig, mainax = test.inspect.plot_jacobian(interactive=True)
    assert hasattr(fig, "_zetan_slider")
    assert mainax in fig.axes
    assert plt.get_fignums()


def test_plot_jacobian_interactive_respects_ax(test):
    """interactive=True draws into a provided ax and returns it as mainax."""
    f, ax = plt.subplots()
    fig, mainax = test.inspect.plot_jacobian(ax=ax, interactive=True)
    assert fig is f
    assert mainax is ax


# --- Beltrami error (requires a solved field) -------------------------------------------


def test_plot_beltrami_requires_solved_field(test):
    """A clear error is raised when the field has not been solved."""
    assert not test.field_is_current
    with pytest.raises(RuntimeError):
        test.inspect.plot_beltrami_error()


def test_compute_fields_enables_beltrami(qa):
    """compute_fields() solves the field, marks it current, and unblocks the Beltrami plot."""
    assert not qa.field_is_current
    qa.compute_fields()
    assert qa.field_is_current
    qa.inspect.plot_beltrami_error()
    assert plt.get_fignums()


def test_force_real_enables_beltrami(qa):
    """force_real() also marks the field current, so the Beltrami plot works afterwards."""
    qa.force_real()
    assert qa.field_is_current
    qa.inspect.plot_beltrami_error()
    assert plt.get_fignums()


def test_beltrami_error_grid_shapes(solved):
    """beltrami_error_grid returns the per-cell grid and matching radial nodes."""
    err4, s_nodes = beltrami_error_grid(solved, 0)
    lquad = solved.allglobal_mod.iquad[0]
    nt = solved.allglobal_mod.nt
    nz = solved.allglobal_mod.nz
    assert err4.shape == (lquad + 1, 3, nz, nt)
    assert s_nodes.shape == (lquad + 1,)


def test_plot_beltrami_error(solved):
    """plot_beltrami_error draws for all volumes and for a single component."""
    solved.inspect.plot_beltrami_error(vvol="all", zetan=0.0)
    assert plt.get_fignums()
    plt.close("all")
    solved.inspect.plot_beltrami_error(vvol=0, component=0)
    assert plt.get_fignums()


def test_plot_beltrami_component_string_codes(solved):
    """component accepts the 's'/'t'/'z' string codes (and rejects bad ones)."""
    for code in ("s", "t", "z"):
        solved.inspect.plot_beltrami_error(component=code)
        assert plt.get_fignums()
        plt.close("all")
    with pytest.raises(ValueError):
        solved.inspect.plot_beltrami_error(component="x")


def test_plot_worst_beltrami(solved):
    """plot_worst_beltrami finds and draws the worst cross-section."""
    solved.inspect.plot_worst_beltrami()
    assert plt.get_fignums()


def test_plot_beltrami_min_val(solved):
    """min_val sets the lower bound of the Beltrami (log) colour scale."""
    solved.inspect.plot_beltrami_error(min_val=2.0)
    mappable = list(plt.gca().collections)[0]
    assert mappable.norm.vmin == 2.0


def test_plot_beltrami_interactive_builds_slider(solved):
    """interactive=True returns (fig, mainax) and carries the zetan slider."""
    fig, mainax = solved.inspect.plot_beltrami_error(interactive=True)
    assert hasattr(fig, "_zetan_slider")
    assert mainax in fig.axes
    assert plt.get_fignums()


def test_plot_beltrami_interactive_respects_ax(solved):
    """interactive=True draws into a provided ax and returns it as mainax."""
    f, ax = plt.subplots()
    fig, mainax = solved.inspect.plot_beltrami_error(ax=ax, interactive=True)
    assert fig is f
    assert mainax is ax


# --- Error printouts on the inspect namespace -------------------------------------------


def test_beltrami_errors_method(solved):
    """inspect.beltrami_errors returns the per-volume (mvol, 3) avg/max arrays."""
    avg, mx = solved.inspect.beltrami_errors(print_res=False)
    mvol = solved.allglobal_mod.mvol
    assert avg.shape == (mvol, 3)
    assert mx.shape == (mvol, 3)


def test_print_errors_field_only(solved):
    """inspect.print_errors(force=False) prints the field error without a force eval."""
    out = solved.inspect.print_errors(force=False)
    assert len(out) == 4


def test_print_errors_requires_force_eval(solved):
    """The force half raises a clear error until force_real() has populated the residual."""
    with pytest.raises(RuntimeError):
        solved.inspect.print_errors()


def test_print_errors_full_after_force_real(solved):
    """After force_real(), inspect.print_errors() prints both field and force errors."""
    solved.force_real()
    out = solved.inspect.print_errors()
    assert len(out) == 4


# --- 3D interface plotting (pyvista) -----------------------------------------------------

import numpy as np  # noqa: E402

_COLOR_BY = ["force", "pressure_jump", "b2_inner", "b2_outer", "current", "angle"]


@pytest.mark.parametrize("color_by", _COLOR_BY)
def test_infaces3d_surfaces_shapes(solved, color_by):
    """Every color_by yields finite (n_theta, n_zeta) scalars on a closed torus surface."""
    out = solved.create_spectreout(solve_field=False)
    data, tarr, zarr = out.infaces3d_surfaces(color_by, surfaces=None, n_theta=12, n_zeta=20)
    assert len(data) >= 1
    idx, x, y, z, scalar = data[0]
    assert x.shape == y.shape == z.shape == scalar.shape == (12, 20)
    assert np.all(np.isfinite(scalar))


def test_infaces3d_outer_interface_guard(solved):
    """A color_by needing the outer volume rejects the outer boundary interface."""
    out = solved.create_spectreout(solve_field=False)
    mvol = out.input.physics.Mvol
    with pytest.raises(ValueError):
        out.infaces3d_surfaces("force", surfaces=[mvol - 1])


def test_infaces3d_bad_color_by(solved):
    """An unknown color_by raises a clear error."""
    out = solved.create_spectreout(solve_field=False)
    with pytest.raises(ValueError):
        out.infaces3d_surfaces("nonsense")


def test_plot_infaces_3d_requires_field(test):
    """The live inspect entry point raises when the field is not solved."""
    assert not test.field_is_current
    with pytest.raises(RuntimeError):
        test.inspect.plot_infaces_3d(show=False)


def test_field_view_matches_spectreout(solved):
    """The lightweight live-instance view yields identical scalars to a full SPECTREout."""
    from spectre.postprocess._field_view import SpectreFieldView

    view = SpectreFieldView(solved)
    out = solved.create_spectreout(solve_field=False)
    d_view, _, _ = view.infaces3d_surfaces("force", surfaces=None, n_theta=10, n_zeta=16)
    d_out, _, _ = out.infaces3d_surfaces("force", surfaces=None, n_theta=10, n_zeta=16)
    assert len(d_view) == len(d_out)
    for a, b in zip(d_view, d_out):
        assert np.allclose(a[4], b[4])


def test_plot_infaces_3d_render(solved):
    """End-to-end: inspect.plot_infaces_3d builds a pyvista plotter (headless)."""
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    try:
        plotter = solved.inspect.plot_infaces_3d(
            color_by="force", n_theta=10, n_zeta=16, show=False
        )
    except Exception as exc:  # no GL/EGL in this environment
        pytest.skip(f"pyvista rendering unavailable: {exc}")
    assert plotter is not None
    assert len(plotter.renderer.actors) > 0
    plotter.close()
