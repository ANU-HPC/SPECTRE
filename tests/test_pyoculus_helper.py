import numpy as np
import pytest

from spectre import SPECTREout, SPECTRE, SPECTRE_pyoculus_helper, get_xinit_specwrap, force_real

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
    
