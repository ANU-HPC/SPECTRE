# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

## Calculate SPECTRE B field (and related quantities) purely in python

## TESTED against existing pyoculus-based implementation
## not tested for nonstell-symmetric cases

import numpy as np

from spectre.postprocess.config import run_decorator


def get_field_mod(self, lvol, sarr, tarr, zarr, Bcontrav=None, g=None):
    """Compute the magnetic field modulus |B| at given coordinates.

    Args:
        lvol (int): Volume index.
        sarr (array-like): Radial coordinate values.
        tarr (array-like): Poloidal coordinate values.
        zarr (array-like): Toroidal coordinate values.
        Bcontrav (ndarray, optional): Precomputed contravariant field components.
            Computed from lvol/sarr/tarr/zarr if not provided.
        g (ndarray, optional): Precomputed metric tensor.
            Computed from lvol/sarr/tarr/zarr if not provided.

    Returns:
        ndarray: Magnetic field modulus |B| with shape matching the input coordinate arrays.
    """

    # cast to np.array of type double
    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    if Bcontrav is None:
        Bcontrav = self.get_field_contrav(lvol, sarr, tarr, zarr)

    if g is None:
        g = self.get_metric(lvol, sarr, tarr, zarr)

    Bmod = np.sqrt(np.einsum("i...,ij...,j...->...", Bcontrav, g, Bcontrav, optimize=True))

    return Bmod


def get_field_cov(self, lvol, sarr, tarr, zarr):
    """Compute the covariant magnetic field components B_i = g_ij B^j.

    Args:
        lvol (int): Volume index.
        sarr (array-like): Radial coordinate values.
        tarr (array-like): Poloidal coordinate values.
        zarr (array-like): Toroidal coordinate values.

    Returns:
        ndarray: Covariant field components with shape [3, s, t, z].
    """

    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    Bcontrav = get_field_contrav(self, lvol, sarr, tarr, zarr)
    metric = self.get_metric(lvol, sarr, tarr, zarr)

    Bcov = np.einsum("i...,ij...->j...", Bcontrav, metric, optimize=True)

    return Bcov


def get_field_cylindrical(self, lvol, sarr, tarr, zarr):
    """Compute the magnetic field in cylindrical (R, Z, phi) components.

    Args:
        lvol (int): Volume index.
        sarr (array-like): Radial coordinate values.
        tarr (array-like): Poloidal coordinate values.
        zarr (array-like): Toroidal coordinate values.

    Returns:
        ndarray: Field components [B_R, B_Z, B_phi] with shape [3, s, t, z].
    """

    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    # geometry = self.input.physics.Igeometry
    # rpol = self.input.physics.rpol
    # rtor = self.input.physics.rtor
    # Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    # Zas, Zbs = self.output.Zbs[lvol : lvol + 2]
    # # Ras, Rbs = self.output.Rbs[lvol : lvol + 2]
    # # Zac, Zbc = self.output.Zbc[lvol : lvol + 2]
    # im = np.array(self.output.im, dtype=int)
    # in_ = np.array(self.output.in_, dtype=int)

    Rarr, Zarr = self.get_coord_transform(lvol, sarr, tarr, zarr)

    Bcontrav = self.get_field_contrav(lvol, sarr, tarr, zarr)

    Bcart = np.empty_like(Bcontrav)

    Bcart[0] = Bcontrav[0] * Rarr[1] + Bcontrav[1] * Rarr[2] + Bcontrav[2] * Rarr[3]
    Bcart[1] = Bcontrav[0] * Zarr[1] + Bcontrav[1] * Zarr[2] + Bcontrav[2] * Zarr[3]
    Bcart[2] = Bcontrav[2] * Rarr[0]

    return Bcart


def get_field_contrav(self, lvol, sarr, tarr, zarr):
    """Returns the contravariant magnetic field

    Args:
        lvol (int): Volume index.
        sarr (array-like): Radial coordinate values.
        tarr (array-like): Poloidal coordinate values.
        zarr (array-like): Toroidal coordinate values.

    Returns:
        ndarray: Contravariant field components [B^s, B^t, B^z] with shape [3, s, t, z].
    """
    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    Aze = np.atleast_2d(self.vector_potential.Aze[lvol])
    Ate = np.atleast_2d(self.vector_potential.Ate[lvol])
    # Azo = np.atleast_2d(self.vector_potential.Azo[lvol])
    # Ato = np.atleast_2d(self.vector_potential.Ato[lvol])

    lrad = self.input.physics.Lrad[lvol]
    mpol = self.input.physics.Mpol
    im = self.output.im
    _in = self.output.in_
    coord_sing = self.input.physics.Igeometry > 1 and lvol == 0

    jac = self.get_jacobian(lvol, sarr, tarr, zarr)

    if coord_sing:
        sbar = np.where(0.5 * (sarr + 1) > 0.0, 0.5 * (sarr + 1), 0)
        basis = get_zernike_basis(sbar, lrad, mpol)

        Bcontrav = get_field_sing(sarr, tarr, zarr, lrad, im, _in, basis, Aze, Ate, jac)

    else:
        basis = get_cheby_basis(sarr, lrad)

        Bcontrav = get_field_nonsing(sarr, tarr, zarr, lrad, im, _in, basis, Aze, Ate, jac)

    return Bcontrav


@run_decorator
def get_field_nonsing(sarr, tarr, zarr, lrad, im, _in, basis, Aze, Ate, jac):
    Bcontrav = np.zeros((3, len(sarr), len(tarr), len(zarr)))

    for t in range(len(tarr)):
        for z in range(len(zarr)):
            for ii in range(len(im)):
                mi = im[ii]
                ni = _in[ii]
                arg = mi * tarr[t] - ni * zarr[z]
                carg = np.cos(arg)
                sarg = np.sin(arg)

                for s in range(len(sarr)):
                    for il in range(lrad + 1):
                        Bcontrav[0, s, t, z] += (
                            (-mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, 0, s] * sarg
                        )
                        Bcontrav[1, s, t, z] += (-Aze[ii, il] * basis[il, 1, s]) * carg
                        Bcontrav[2, s, t, z] += (Ate[ii, il] * basis[il, 1, s]) * carg

                    # if(not stell_sym): # needs Ato, Azo
                    #     Bcontrav[0, s, t, z] += (- mi * Azo[ii, l+1] - ni * Ato[ii, l+1]
                    #                              ) * basis[l, 0, s] * carg
                    #     Bcontrav[1, s, t, z] += (- Azo[ii, l+1] * basis[l, 1, s]) * sarg
                    #     Bcontrav[2, s, t, z] += (Ato[ii, l+1] * basis[l, 1, s]) * sarg

    Bcontrav /= jac

    return Bcontrav


@run_decorator
def get_field_sing(sarr, tarr, zarr, lrad, im, _in, basis, Aze, Ate, jac):
    Bcontrav = np.zeros((3, len(sarr), len(tarr), len(zarr)))

    for t in range(len(tarr)):
        for z in range(len(zarr)):
            for ii in range(len(im)):
                mi = im[ii]
                ni = _in[ii]
                arg = mi * tarr[t] - ni * zarr[z]
                carg = np.cos(arg)
                sarg = np.sin(arg)

                for s in range(len(sarr)):
                    for il in range(lrad + 1):
                        Bcontrav[0, s, t, z] += (
                            (-mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, mi, 0, s] * sarg
                        )
                        Bcontrav[1, s, t, z] += (-Aze[ii, il] * basis[il, mi, 1, s]) * 0.5 * carg
                        Bcontrav[2, s, t, z] += (Ate[ii, il] * basis[il, mi, 1, s]) * 0.5 * carg

                    # if(not stell_sym): # needs Ato, Azo
                    #     Bcontrav[0, s, t, z] += (- mi * Azo[ii, l+1] - ni * Ato[ii, l+1]
                    #                              ) * basis[l, mi, 0, s] * carg
                    #     Bcontrav[1, s, t, z] += (- Azo[ii, l+1] * basis[l, mi, 1, s]) * 0.5 * sarg
                    #     Bcontrav[2, s, t, z] += (Ato[ii, l+1] * basis[l, mi, 1, s]) * 0.5 * sarg

    Bcontrav /= jac

    return Bcontrav


@run_decorator
def get_cheby_basis(sarr, lrad):
    # [radial mode (0->lrad), func/derivative (0/1), radial coord (sarr)]
    basis = np.zeros((lrad + 1, 2, sarr.size))

    for s in range(len(sarr)):
        basis[0, 0, s] = 1.0
        basis[1, 0, s] = sarr[s]
        basis[1, 1, s] = 1.0

        for il in range(2, lrad + 1):
            basis[il, 0, s] = 2.0 * sarr[s] * basis[il - 1, 0, s] - basis[il - 2, 0, s]
            basis[il, 1, s] = (
                2.0 * basis[il - 1, 0, s]
                + 2.0 * sarr[s] * basis[il - 1, 1, s]
                - basis[il - 2, 1, s]
            )

        # basis recombination
        basis[:, 0, s] -= (-1) ** np.arange(lrad + 1)

    # rescale for conditioning
    for il in range(0, lrad + 1):
        basis[il] /= il + 1.0

    return basis


@run_decorator
def get_zernike_basis(sbararr, lrad, mpol):
    # [radial mode (0->lrad), poloidal mode (0->mpol), func/derivative (0/1), radial coord (sarr)]
    basis = np.zeros((lrad + 1, mpol + 1, 2, sbararr.size))

    for s in range(len(sbararr)):
        r = sbararr[s]
        rm = 1.0  # sbar^m
        rm1 = 0.0  # sbar^(m-1)
        for m in range(mpol + 1):
            if m <= lrad:
                basis[m, m, 0, s] = rm
                basis[m, m, 1, s] = m * rm1

            if m + 2 <= lrad:
                basis[m + 2, m, 0, s] = (m + 2) * rm * r**2.0 - (m + 1) * rm
                basis[m + 2, m, 1, s] = (m + 2) ** 2.0 * rm * r - (m + 1) * m * rm1

            for il in range(m + 4, lrad + 1, 2):
                fac1 = il / (il**2.0 - m**2.0)
                fac2 = 4.0 * (il - 1)
                fac3 = (il - 2 + m) ** 2.0 / (il - 2) + (il - m) ** 2.0 / il
                fac4 = ((il - 2) ** 2.0 - m**2.0) / (il - 2.0)

                basis[il, m, 0, s] = fac1 * (
                    (fac2 * r**2.0 - fac3) * basis[il - 2, m, 0, s] - fac4 * basis[il - 4, m, 0, s]
                )
                basis[il, m, 1, s] = fac1 * (
                    2.0 * fac2 * r * basis[il - 2, m, 0, s]
                    + (fac2 * r**2.0 - fac3) * basis[il - 2, m, 1, s]
                    - fac4 * basis[il - 4, m, 1, s]
                )

            rm1 = rm
            rm = rm * r

    # basis recombination
    for il in range(2, lrad + 1, 2):
        basis[il, 0, 0] -= (-1) ** (il / 2.0)

    if mpol >= 1:
        for s in range(len(sbararr)):
            for il in range(3, lrad + 1, 2):
                basis[il, 1, 0, s] -= (-1) ** ((il - 1) * 0.5) * (il + 1) * 0.5 * sbararr[s]
                basis[il, 1, 1, s] -= (-1) ** ((il - 1) * 0.5) * (il + 1) * 0.5

    # rescaling
    for m in range(0, mpol + 1):
        for il in range(m, lrad + 1, 2):
            basis[il, m, :] /= il + 1.0

    return basis


@run_decorator
def get_cheby_basis_single(sarr, lrad):
    # [radial mode (0->lrad), func/derivative (0/1), radial coord (sarr)]
    # print(lrad)
    basis = np.zeros((lrad + 1, 2))

    basis[0, 0] = 1.0
    basis[1, 0] = sarr
    basis[1, 1] = 1.0

    for il in range(2, lrad + 1):
        basis[il, 0] = 2.0 * sarr * basis[il - 1, 0] - basis[il - 2, 0]
        basis[il, 1] = 2.0 * basis[il - 1, 0] + 2.0 * sarr * basis[il - 1, 1] - basis[il - 2, 1]

    # basis recombination
    basis[:, 0] -= (-1) ** np.arange(lrad + 1)

    # rescale for conditioning
    for il in range(0, lrad + 1):
        basis[il] /= il + 1.0

    return basis


@run_decorator
def get_zernike_basis_single(sbararr, lrad, mpol):
    # [radial mode (0->lrad), poloidal mode (0->mpol), func/derivative (0/1), radial coord (sarr)]
    basis = np.zeros((lrad + 1, mpol + 1, 2))

    r = sbararr
    rm = 1.0  # sbar^m
    rm1 = 0.0  # sbar^(m-1)
    for m in range(mpol + 1):
        if m <= lrad:
            basis[m, m, 0] = rm
            basis[m, m, 1] = m * rm1

        if m + 2 <= lrad:
            basis[m + 2, m, 0] = (m + 2) * rm * r**2.0 - (m + 1) * rm
            basis[m + 2, m, 1] = (m + 2) ** 2.0 * rm * r - (m + 1) * m * rm1

        for il in range(m + 4, lrad + 1, 2):
            fac1 = il / (il**2.0 - m**2.0)
            fac2 = 4.0 * (il - 1)
            fac3 = (il - 2 + m) ** 2.0 / (il - 2) + (il - m) ** 2.0 / il
            fac4 = ((il - 2) ** 2.0 - m**2.0) / (il - 2.0)

            basis[il, m, 0] = fac1 * (
                (fac2 * r**2.0 - fac3) * basis[il - 2, m, 0] - fac4 * basis[il - 4, m, 0]
            )
            basis[il, m, 1] = fac1 * (
                2.0 * fac2 * r * basis[il - 2, m, 0]
                + (fac2 * r**2.0 - fac3) * basis[il - 2, m, 1]
                - fac4 * basis[il - 4, m, 1]
            )

        rm1 = rm
        rm = rm * r

    # basis recombination
    for il in range(2, lrad + 1, 2):
        basis[il, 0, 0] -= (-1) ** (il / 2.0)

    if mpol >= 1:
        for il in range(3, lrad + 1, 2):
            basis[il, 1, 0] -= (-1) ** ((il - 1) * 0.5) * (il + 1) * 0.5 * sbararr
            basis[il, 1, 1] -= (-1) ** ((il - 1) * 0.5) * (il + 1) * 0.5

    # rescaling
    for m in range(0, mpol + 1):
        for il in range(m, lrad + 1, 2):
            basis[il, m, :] /= il + 1.0

    return basis


def calc_avg_beta_vol(
    self,
    ns=28,
    nt=64,
    nz=28,
    plot=False,
    beta_definition="inv_b2_avg",
):
    """Compute the volume-averaged plasma beta per volume and overall.

    Args:
        ns (int, optional): Number of radial integration points. Defaults to 28.
        nt (int, optional): Number of poloidal integration points. Defaults to 64.
        nz (int, optional): Number of toroidal integration points. Defaults to 28.
        plot (bool, optional): If ``True``, plots |B| on each cross-section with the interfaces.
            Defaults to False.
        beta_definition (str, optional): How beta is computed:

            - ``"inv_b2_avg"``: per-volume ``β_k = 2 p_k ⟨1/B²⟩_k``; global beta is the
              volume-weighted average.
            - ``"ratio_of_averages"`` (VMEC-style): global ``β = 2 Σ_k p_k V_k / Σ_k ⟨B²⟩_k V_k``;
              per-volume ``β_k = 2 p_k / ⟨B²⟩_k`` (note: volume-averaging these does not recover
              the global beta).

            Defaults to ``"inv_b2_avg"``.

    Returns:
        tuple: (beta_avg, betas_vol) — global beta and per-volume beta array.
    """
    if beta_definition not in ("inv_b2_avg", "ratio_of_averages"):
        raise ValueError(
            f"beta_definition must be 'inv_b2_avg' or 'ratio_of_averages', got {beta_definition!r}"
        )

    nfp = int(self.input.physics.Nfp)
    sarr = np.linspace(-0.9999, 1.0, ns)
    tarr = np.linspace(0, 2 * np.pi, nt)
    zarr = np.linspace(0, 2 * np.pi / nfp, nz)

    if plot:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(9, 8))

        self.plot_infaces(ax=ax, zeta=zarr[0])

    nvol = self.input.physics.Nvol
    betas_vol = np.zeros(nvol)
    volumes_vol = np.zeros(nvol)
    sum_pV = 0.0
    sum_B2 = 0.0

    for lvol in range(nvol):
        jac = self.get_jacobian(lvol, sarr, tarr, zarr)
        Bmod = self.get_field_mod(lvol, sarr, tarr, zarr)
        int_volume = np.trapz(
            np.trapz(np.trapz(jac, x=sarr, axis=0), x=tarr, axis=0), x=zarr, axis=0
        )
        p_k = self.input.physics.pressure[lvol] * self.input.physics.pscale

        if beta_definition == "inv_b2_avg":
            int_invB2 = np.trapz(
                np.trapz(np.trapz(jac / Bmod**2, x=sarr, axis=0), x=tarr, axis=0), x=zarr, axis=0
            )
            betas_vol[lvol] = 2 * p_k * int_invB2 / int_volume
            volumes_vol[lvol] = int_volume
        else:  # ratio_of_averages
            int_B2 = np.trapz(
                np.trapz(np.trapz(jac * Bmod**2, x=sarr, axis=0), x=tarr, axis=0), x=zarr, axis=0
            )
            betas_vol[lvol] = 2 * p_k * int_volume / int_B2
            sum_pV += p_k * int_volume
            sum_B2 += int_B2

        if plot:
            r, z = self.get_coord_transform(lvol, sarr, tarr, zarr)
            im = ax.pcolor(r[0, ..., 0], z[0, ..., 0], Bmod[..., 0], vmin=0.15, vmax=0.3)

    if beta_definition == "inv_b2_avg":
        beta_avg = np.sum(betas_vol * volumes_vol) / np.sum(volumes_vol)
    else:  # ratio_of_averages
        beta_avg = 2.0 * sum_pV / sum_B2

    print("Betas per volume:")
    for i in range(nvol):
        print(f"{i}:    {betas_vol[i] * 100:6.2f}%")
    print(f"Average beta {beta_avg * 100:.2f}%")

    if plot:
        plt.title("β")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

    return beta_avg, betas_vol
