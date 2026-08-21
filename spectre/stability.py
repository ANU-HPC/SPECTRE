# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT


def calc_hessian(x, test, save_to_h5=False):

    hessian = test.compute_hessian(x)

    # add option to save output to .h5 file
    if save_to_h5:
        from h5py import File

        with File(test.allglobal_mod.filename_h5, "a") as f:
            grp_errs = f.create_group("stability")
            grp_errs.create_dataset("ohessian", data=hessian)

    return hessian
