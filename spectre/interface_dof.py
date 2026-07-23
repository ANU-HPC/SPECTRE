# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np


class _DropFirstIdentity:
    """Acts like ``np.identity(n + 1)[:, 1:]`` under ``@`` without materialising it.

    The dense identity is (n+1) x n -- over a gigabyte at mpol=ntor=20 -- and the Jacobian
    right-multiplication by it is a full GEMM; both reduce to dropping the dead column /
    prepending a zero row.
    """

    __array_ufunc__ = None  # make ndarray @ self defer to __rmatmul__

    def __init__(self, n):
        self.shape = (n + 1, n)

    def __matmul__(self, other):
        # D @ dirs: prepend packxi's dead slot-0 row of zeros.
        other = np.asarray(other)
        out = np.zeros((self.shape[0],) + other.shape[1:], dtype=other.dtype)
        out[1:] = other
        return out

    def __rmatmul__(self, other):
        # jac @ D: drop the dead slot-0 column.
        return np.asarray(other)[..., 1:]


def identity_dof():
    # xmn[0] is packxi's dead augmentation slot (unused by dforce/solve_field); drop it.

    def xmn_to_xdof(xmn, test):
        return xmn[1:]

    def xdof_to_xmn(xdof, test):
        return np.concatenate(([0.0], xdof))

    def Dxmn_Dxdof_at_xdof(xdof, test):
        return _DropFirstIdentity(len(xdof))

    return xmn_to_xdof, xdof_to_xmn, Dxmn_Dxdof_at_xdof


def reduced_rzmn_dof(mpol_rz, ntor_rz):
    # infaces_unflatten/infaces_flatten always ignore/zero element 0 (packxi's dead
    # augmentation slot); drop it from xdof here too.

    from spectre.utils import infaces_flatten, infaces_unflatten

    def xmn_to_xdof(xmn, test):
        im_ = test.allglobal_mod.im
        in_ = test.allglobal_mod.in_
        mask = (im_ <= mpol_rz) & (np.abs(in_) <= ntor_rz)

        r, z = infaces_unflatten(xmn, test)

        rnew = r[:, mask]
        znew = z[:, mask]

        xnew = infaces_flatten(rnew, znew, test)
        return xnew[1:]

    def xdof_to_xmn(xdof, test):
        rnew = np.zeros((test.allglobal_mod.mvol - 1, test.allglobal_mod.mn))
        znew = np.zeros((test.allglobal_mod.mvol - 1, test.allglobal_mod.mn))

        r, z = infaces_unflatten(np.concatenate(([0.0], xdof)), test)

        im_ = test.allglobal_mod.im
        in_ = test.allglobal_mod.in_
        mask = (im_ <= mpol_rz) & (np.abs(in_) <= ntor_rz)

        rnew[:, mask] = r
        znew[:, mask] = z

        xnew = infaces_flatten(rnew, znew, test)
        return xnew

    def Dxmn_Dxdof_at_xdof(xdof, test):
        mvol = test.allglobal_mod.mvol
        mn = test.allglobal_mod.mn

        im_ = test.allglobal_mod.im
        in_ = test.allglobal_mod.in_
        mask = (im_ <= mpol_rz) & (np.abs(in_) <= ntor_rz)
        mn_new = np.sum(mask)

        n_xdof = (mvol - 1) * (2 * mn_new - 1)
        jac = np.zeros(((mvol - 1) * (2 * mn - 1) + 1, n_xdof))  # xmn x xdof

        # label xdof entries 1..n_xdof (0 = unmapped) and read off where xdof_to_xmn puts them
        vals = np.array(xdof_to_xmn(np.arange(1, n_xdof + 1), test), dtype=int)
        mapped = vals > 0

        jac[np.flatnonzero(mapped), vals[mapped] - 1] = 1.0
        return jac

    return xmn_to_xdof, xdof_to_xmn, Dxmn_Dxdof_at_xdof


def henneberg_dof(alpha_fac=1, nmax=8, mmax=8, ntheta=48, nphi=24):

    from spectre.surfaces.surf_henneberg import SurfaceHenneberg

    def xmn_to_xdof(xmn, test):

        bnd_rc = test.allglobal_mod.irbc[:, -1]
        bnd_zs = test.allglobal_mod.izbs[:, -1]

        surfaces = SurfaceHenneberg.surfaces_from_spectre_xmn(
            xmn, bnd_rc, bnd_zs, test, alpha_fac, nmax, mmax, ntheta, nphi
        )

        total_dofs = []

        for s in range(len(surfaces) - 1):
            total_dofs.append(surfaces[s].get_dofs())

        # The boundary DOFs will be stored separately so the algorithm doesn't try to vary them
        test.henneberg_boundary_dof = surfaces[-1].get_dofs()

        return np.concatenate(total_dofs)  # list of 1d arrays -> 1d array

    def xdof_to_xmn(xdof, test):

        mvol = test.allglobal_mod.mvol
        num_dofs = xdof.size // (mvol - 1)

        surfaces = []

        for v in range(0, mvol):
            surf = SurfaceHenneberg(
                alpha_fac=alpha_fac, nmax=nmax, mmax=mmax
            )  # , ntheta=100, nphi=100)

            if v < (mvol - 1):
                surf.set_dofs_impl(xdof[v * num_dofs : (v + 1) * num_dofs])
            else:
                surf.set_dofs_impl(test.henneberg_boundary_dof)

            surfaces.append(surf)

        xmn, bnd_rc, bnd_zs = SurfaceHenneberg.surfaces_to_spectre_xmn(surfaces, test)

        # set the plasma boundary to have the new parametrization
        test.allglobal_mod.irbc[:, -1] = bnd_rc
        test.allglobal_mod.izbs[1:, -1] = bnd_zs

        return xmn  # 1d array with all modes of all interfaces

    def Dxmn_Dxdof_at_xdof(xdof, test):

        mvol = test.allglobal_mod.mvol
        num_dofs = xdof.size // (mvol - 1)

        surfaces = []

        for v in range(0, mvol - 1):
            surf = SurfaceHenneberg(
                alpha_fac=alpha_fac, nmax=nmax, mmax=mmax
            )  # , ntheta=100, nphi=100)
            surf.set_dofs_impl(xdof[v * num_dofs : (v + 1) * num_dofs])

            surfaces.append(surf)

        dxmn_dxdof = SurfaceHenneberg.surfaces_to_spectre_dxmn_dxdof(surfaces, test)

        return dxmn_dxdof.T

    return xmn_to_xdof, xdof_to_xmn, Dxmn_Dxdof_at_xdof
