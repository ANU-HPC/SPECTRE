
import numpy as np

def calc_hessian(x, test, save_to_h5=False):

    ngdof = test.allglobal_mod.ngdof

    hessian = np.zeros((ngdof, ngdof), dtype=np.float64, order="F")
    test.hessian_mod.compute_hessian(x, hessian)
    hessian = hessian.T

    # add option to save output to .h5 file
    if save_to_h5:
        from h5py import File
        with File(test.allglobal_mod.filename_h5, 'a') as f:
            grp_errs = f.create_group('stability')
            grp_errs.create_dataset('ohessian', data=hessian)

    return hessian