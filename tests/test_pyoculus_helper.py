import numpy as np

from spectre import SPECTREout, SPECTRE, SPECTRE_pyoculus_helper, get_xinit_specwrap, force_real

def test_field():
    
    out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")

    helper = SPECTRE_pyoculus_helper(out_obj)

    coords = np.array([0.5, 0.1, 0.2])
    
    b_helper = helper.B(coords)
    print(b_helper)

    jac = out_obj.get_jacobian(0, *coords)
    b_specout = out_obj.get_field_contrav(0, *coords) * jac
    b_specout = b_specout.reshape(3)
    print(b_specout)

    xyz = helper.convert_coords(coords)
    print("RzZ coordinates:", xyz)
    
    assert np.allclose(b_helper, b_specout)

def test_field_wrapper():
    
    out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")
    helper = SPECTRE_pyoculus_helper(out_obj)

    out_obj_wrapper = SPECTRE.from_input_file("./examples/qa_vac_vol7/init.toml")
    force_real(get_xinit_specwrap(out_obj_wrapper), out_obj_wrapper)
    helper_wraper = SPECTRE_pyoculus_helper(out_obj_wrapper)

    coords = np.array([0.5, 0.1, 0.2])
    
    print('wrapper B:', helper.B(coords), helper_wraper.B(coords))

    print("wrapper RzZ coordinates:", helper.convert_coords(coords), helper_wraper.convert_coords(coords))
    
    assert np.allclose(helper.B(coords), helper_wraper.B(coords))

def test_field_deriv():

    out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")

    helper = SPECTRE_pyoculus_helper(out_obj)

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
        
        b_deriv_fd[:, j] = (b_plus - b_minus) / (2 * h)

    diff = b_deriv - b_deriv_fd
    print("Difference (analytical - finite difference):\n", diff)
    
    assert np.allclose(b_deriv, b_deriv_fd)
    
