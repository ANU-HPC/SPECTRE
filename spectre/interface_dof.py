import numpy as np


def identity_dof():

    def xmn_to_xdof(xmn, test):
        return xmn
    
    def xdof_to_xmn(xdof, test):
        return xdof
    
    def Dxmn_Dxdof_at_xdof(xdof, test):
        return np.identity(len(xdof))
    
    return xmn_to_xdof, xdof_to_xmn, Dxmn_Dxdof_at_xdof


def reduced_rzmn_dof(mpol_rz, ntor_rz):

    from spectre.misc import infaces_flatten, infaces_unflatten

    def xmn_to_xdof(xmn, test):
        im_ = test.allglobal_mod.im
        in_ = test.allglobal_mod.in_
        mask = (im_ <= mpol_rz) & (np.abs(in_) <= ntor_rz)

        r, z = infaces_unflatten(xmn, test)

        rnew = r[:, mask]
        znew = z[:, mask]

        xnew = infaces_flatten(rnew, znew, test)
        return xnew

    def xdof_to_xmn(xdof, test):
        rnew = np.zeros((test.allglobal_mod.mvol-1, test.allglobal_mod.mn))
        znew = np.zeros((test.allglobal_mod.mvol-1, test.allglobal_mod.mn))

        r, z = infaces_unflatten(xdof, test)

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

        jac = np.zeros(( (mvol-1)*(2*mn-1)+1, (mvol-1)*(2*mn_new-1)+1 )) # xmn x xdof

        vals = np.array(xdof_to_xmn(np.arange(jac.shape[1]), test), dtype=int)

        jac[np.arange(vals.shape[0]), vals] = 1.0
        jac[:,0] = 0.
        return jac

    return xmn_to_xdof, xdof_to_xmn, Dxmn_Dxdof_at_xdof