import numpy as np
import pytest

from spectre import SPECTREout, SPECTRE, SPECTRE_pyoculus_helper, get_xinit_specwrap, force_real
from spectre.analysis import PyocFields, PyocHelpers

lvols = [0, 1]

@pytest.mark.parametrize("lvol", lvols)
def test_field(lvol):
    
    spectre = SPECTRE.from_input_file("./examples/qa_vac_vol7/init.toml")
    helper_fromspectre_method = spectre.get_pyoculus_helper(lvol=lvol)
    helper_fromspectre = SPECTRE_pyoculus_helper(spectre, lvol=lvol)
    
    spectre.run(save_output=True)
    out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")
    helper_fromh5 = SPECTRE_pyoculus_helper(out_obj, lvol=lvol)
    coords = np.array([0.5, 0.1, 0.2])
    
    b_fromh5 = helper_fromh5.B(coords)
    b_fromspectre_method = helper_fromspectre_method.B(coords)
    b_fromspectre = helper_fromspectre.B(coords)
    
    assert np.allclose(b_fromh5, b_fromspectre_method)
    assert np.allclose(b_fromh5, b_fromspectre)

    jac = out_obj.get_jacobian(lvol, *coords)
    b_specout = out_obj.get_field_contrav(lvol, *coords) * jac
    b_specout = b_specout.reshape(3)

    xyz_h5 = helper_fromh5.convert_coords(coords)
    xyz_fromspectre_method = helper_fromspectre_method.convert_coords(coords)
    xyz_fromspectre = helper_fromspectre.convert_coords(coords)

    assert np.allclose(xyz_h5, xyz_fromspectre_method)
    assert np.allclose(xyz_h5, xyz_fromspectre)

    assert np.allclose(b_fromh5, b_specout)

@pytest.mark.parametrize("lvol", lvols)
def test_field_wrapper(lvol):
    
    out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")
    helper = SPECTRE_pyoculus_helper(out_obj, lvol=lvol)

    out_obj_wrapper = SPECTRE.from_input_file("./examples/qa_vac_vol7/init.toml")
    force_real(get_xinit_specwrap(out_obj_wrapper), out_obj_wrapper)
    helper_wraper = SPECTRE_pyoculus_helper(out_obj_wrapper, lvol=lvol)
    coords = np.array([0.5, 0.1, 0.2])
    
    assert np.allclose(helper.B(coords), helper_wraper.B(coords))

@pytest.mark.parametrize("lvol", lvols)
def test_field_deriv(lvol):

    out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")

    helper = SPECTRE_pyoculus_helper(out_obj, lvol=lvol)
    coords = np.array([0.5, 0.1, 0.2])

    b_helper, b_deriv = helper.dBdX(coords)
    
    h = 1e-6
    b_deriv_fd = np.zeros((3, 3))
    
    for j in range(3):
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        
        coords_plus[j] += h
        coords_minus[j] -= h
        
        b_plus = helper.B(coords_plus)
        b_minus = helper.B(coords_minus)
        
        b_deriv_fd[j, :] = (b_plus - b_minus) / (2 * h)
    
    assert np.allclose(b_deriv, b_deriv_fd)


class TestPyocHelpers:
    """Tests for the PyocHelpers lazy-loading container."""

    def test_lazy_loading(self):
        """Test that helpers are only created when accessed."""
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")
        helpers = PyocHelpers(out_obj, out_obj.input.physics.Mvol)

        # Internal list should be all None before any access
        assert all(h is None for h in helpers._helpers)

        # Access one helper
        helper_0 = helpers[0]
        assert helper_0 is not None
        assert isinstance(helper_0, SPECTRE_pyoculus_helper)

        # Only the accessed index should be populated
        assert helpers._helpers[0] is not None
        assert all(h is None for h in helpers._helpers[1:])

        # Accessing the same index again returns the cached object
        assert helpers[0] is helper_0

    def test_len(self):
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")
        helpers = PyocHelpers(out_obj, out_obj.input.physics.Mvol)
        assert len(helpers) == out_obj.input.physics.Mvol

    @pytest.mark.parametrize("lvol", lvols)
    def test_consistent_with_direct_construction(self, lvol):
        """Test that helpers from PyocHelpers match directly constructed ones."""
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")
        helpers = PyocHelpers(out_obj, out_obj.input.physics.Mvol)
        helper_direct = SPECTRE_pyoculus_helper(out_obj, lvol=lvol)

        coords = np.array([0.5, 0.1, 0.2])
        assert np.allclose(helpers[lvol].B(coords), helper_direct.B(coords))
        assert np.allclose(
            helpers[lvol].convert_coords(coords),
            helper_direct.convert_coords(coords),
        )

    def test_property_on_spectreout(self):
        """Test that SPECTREout.pyoculus_helpers property returns a PyocHelpers."""
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")

        # Property should return a PyocHelpers instance
        assert isinstance(out_obj.pyoculus_helpers, PyocHelpers)

        # Repeated access returns the same cached object
        assert out_obj.pyoculus_helpers is out_obj.pyoculus_helpers


class TestPyocFields:
    """Tests for the PyocFields lazy-loading container."""

    def test_lazy_loading(self):
        """Test that fields are only created when accessed."""
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")
        fields = PyocFields(out_obj, out_obj.input.physics.Mvol)

        # Internal list should be all None before any access
        assert all(f is None for f in fields._fields)

        # Access one field
        field_0 = fields[0]
        assert field_0 is not None

        # Only the accessed index should be populated
        assert fields._fields[0] is not None
        assert all(f is None for f in fields._fields[1:])

        # Accessing the same index again returns the cached object
        assert fields[0] is field_0

    def test_len(self):
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")
        fields = PyocFields(out_obj, out_obj.input.physics.Mvol)
        assert len(fields) == out_obj.input.physics.Mvol

    def test_property_on_spectreout(self):
        """Test that SPECTREout.pyoculus_fields property returns a PyocFields."""
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")

        # Property should return a PyocFields instance
        assert isinstance(out_obj.pyoculus_fields, PyocFields)

        # Repeated access returns the same cached object
        assert out_obj.pyoculus_fields is out_obj.pyoculus_fields


class TestFindFixedPointAndAxis:
    """Tests for find_island_fixed_point and find_axis on SPECTREout."""

    def test_find_axis(self):
        """Test that find_axis returns valid coordinates."""
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")

        r, z, s, theta = out_obj.find_axis(guess = [ -0.95966956988, 3.1415], nrestart=0)

        # Should have found a fixed point
        assert r is not None, "find_axis failed to find a fixed point"
        assert z is not None
        assert s is not None
        assert theta is not None

        # The axis should be inside volume 0 (s in [-1, 1])
        assert -1.0 <= s <= 1.0, f"s={s} is out of bounds"

        # R should be positive for a tokamak/stellarator geometry
        assert r > 0, f"R={r} should be positive"

    def test_find_axis_uses_pyoculus_properties(self):
        """Test that find_axis exercises the pyoculus_fields and pyoculus_helpers properties."""
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")

        # Before calling find_axis, the properties should not be populated
        assert out_obj._pyoculus_fields is None
        assert out_obj._pyoculus_helpers is None

        r, z, s, theta = out_obj.find_axis(guess = [ -0.95966956988, 3.1415], nrestart=0)

        # After find_axis, both properties should be populated
        assert out_obj._pyoculus_fields is not None
        assert out_obj._pyoculus_helpers is not None
        assert out_obj.pyoculus_fields[0] is not None
        assert out_obj.pyoculus_helpers[0] is not None
        for i in range(1, out_obj.input.physics.Mvol):
            assert out_obj._pyoculus_helpers._helpers[i] is None
            assert out_obj._pyoculus_fields._fields[i] is None

    def test_find_island_fixed_point(self):
        """Test find_island_fixed_point with m=-5 n=2 in vol=4 where a large
         island exists in this init config.
         """
        out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")

        # find_island_fixed_point with n=2, m=-5 in vol=4.
        r_fp, z_fp, s_fp, theta_fp = out_obj.find_island_fixed_point(guess=[-0.41, 2.29],
            vol=4, n=2, m=-5, nrestart=0
        )

        # Should have found a fixed point
        assert r_fp is not None