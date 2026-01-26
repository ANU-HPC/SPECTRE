import numpy as np

from spectre.config import run_decorator


def get_cyl_island_width(fname, lvol, ns=120, nt=210, m=2, n=1, plot=False):
    z0 = np.ones(1) * 0.0

    # defaults
    ns = ns or 120
    nt = nt or 210
    m = m or 2
    n = n or 2

    sarr = np.linspace(-1, 1, ns)
    tarr = np.linspace(0, 2 * np.pi, nt)

    from spectre import SPECout

    data = SPECout(fname)
    mn = data.output.mn
    im = data.output.im
    in_ = data.output.in_
    Lrad = data.input.physics.Lrad[lvol]
    Ate = data.vector_potential.Ate[lvol].T
    Aze = data.vector_potential.Aze[lvol].T
    Ato = data.vector_potential.Ato[lvol].T
    Azo = data.vector_potential.Azo[lvol].T
    rtor = data.input.physics.rtor
    Rac = data.output.Rbc[lvol]
    Rbc = data.output.Rbc[lvol + 1]

    at, az = get_spec_vecpot(Lrad, Ate, Aze, Ato, Azo, im, in_, lvol, sarr, tarr, z0)
    ahel = m * az[:, :, 0] + n * at[:, :, 0]

    rtdata = get_spec_rtarr(Rac, Rbc, im, in_, mn, lvol, sarr, tarr, z0)

    sarr, Rarr = get_spec_R_derivatives(Rac, Rbc, im, in_, lvol, sarr, tarr, z0)
    jac = Rarr[0] * Rarr[1] * rtor

    bt, bz = get_spec_magfield(Lrad, im, in_, Ate, Aze, Ato, Azo, jac, lvol, sarr, tarr, z0)
    bdotgradu = m * bt - n * bz

    opt_ind_s, opt_ind_t = np.unravel_index(ahel.argmin(), ahel.shape)
    if opt_ind_s > (ns - 2) or opt_ind_s < 2:
        print(f"Island could not be found !! ({fname=} {lvol=} {m=} {n=})")
        return 1e-15

    xpt_ind_t = opt_ind_t + nt // (2 * m)
    if xpt_ind_t > nt:
        xpt_ind_t -= nt

    xpt_ind_s = np.argmin(np.abs(bdotgradu[:, xpt_ind_t]))
    asep = ahel[xpt_ind_s, xpt_ind_t]

    ahel_along_opt_chord = np.abs(asep - ahel[:, opt_ind_t])
    min1 = np.argmin(ahel_along_opt_chord)
    ahel_along_opt_chord[min1 - 3 : min1 + 4] += 1.0
    min2 = np.argmin(ahel_along_opt_chord)
    min_low = min(min1, min2)
    min_high = max(min1, min2)

    rwr = rtdata[0][min_high, opt_ind_t]
    rwl = rtdata[0][min_low, opt_ind_t]
    Wisland = rwr - rwl

    if plot:
        import matplotlib.pyplot as plt

        from warnings import filterwarnings

        filterwarnings("ignore", category=UserWarning, message="The input coordinates to pcolor.*")

        def rt_to_xy(r, th):
            x = r * np.cos(th)
            y = r * np.sin(th)
            return x, y

        plt.figure(figsize=(8, 6))
        x, y = rt_to_xy(rtdata[0], rtdata[1])
        plt.plot(x[-1], y[-1], "k-")
        plt.plot(x[0], y[0], "k-")
        plt.pcolor(x, y, ahel[:, :])
        plt.contour(x, y, ahel, colors="k", levels=30, linestyles="dashed", linewidths=0.6)
        plt.contour(x, y, ahel, colors="k", levels=[asep], linestyles="solid", linewidths=1.4)
        plt.plot(*rt_to_xy(rtdata[0][:, opt_ind_t], rtdata[1][:, opt_ind_t]), "r--", lw=0.6)
        plt.plot(
            *rt_to_xy(rtdata[0][min_high, opt_ind_t], rtdata[1][min_high, opt_ind_t]),
            "kx",
            ms=11,
        )
        plt.plot(
            *rt_to_xy(rtdata[0][min_low, opt_ind_t], rtdata[1][min_low, opt_ind_t]),
            "kx",
            ms=11,
        )
        plt.plot(
            *rt_to_xy(rtdata[0][xpt_ind_s, xpt_ind_t], rtdata[1][xpt_ind_s, xpt_ind_t]),
            "kx",
            ms=11,
        )
        plt.plot(
            *rt_to_xy(rtdata[0][opt_ind_s, opt_ind_t], rtdata[1][opt_ind_s, opt_ind_t]),
            "ro",
            ms=8,
        )
        plt.title(f"{fname}    W={Wisland:.4f}")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    return Wisland


@run_decorator
def get_spec_magfield(Lrad, im, in_, Ate, Aze, Ato, Azo, jac, lvol, sarr, tarr, zarr):
    # mn = len(im)
    # ns = len(sarr)
    # nt = len(tarr)
    # nz = len(zarr)

    # Bt = np.zeros((ns, nt, nz))
    # Bz = np.zeros((ns, nt, nz))

    T = get_spec_polynomial_basis(Lrad, lvol, sarr)

    # for l in range(Lrad + 1):
    #     dbasis = T[l][1]
    #     for j in range(mn):
    #         for it in range(nt):
    #             for iz in range(nz):
    #                 cosa = np.cos(im[j] * tarr[it] - in_[j] * zarr[iz])
    #                 sina = np.sin(im[j] * tarr[it] - in_[j] * zarr[iz])
    #                 Bt[:, it, iz] -= dbasis * (Aze[l, j] * cosa + Azo[l, j] * sina)
    #                 Bz[:, it, iz] += dbasis * (Ate[l, j] * cosa + Ato[l, j] * sina)

    alpha = (
        im[:, None, None] * tarr[None, :, None] - in_[:, None, None] * zarr[None, None, :]
    )  # [j,it,iz]
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    dbasis = T[:, 1, :]  # [l,s]

    Bt = -np.einsum("ls,lj,jtz->stz", dbasis, Aze, cosa, optimize=True) - np.einsum(
        "ls,lj,jtz->stz", dbasis, Azo, sina, optimize=True
    )
    Bz = np.einsum("ls,lj,jtz->stz", dbasis, Ate, cosa, optimize=True) + np.einsum(
        "ls,lj,jtz->stz", dbasis, Ato, sina, optimize=True
    )

    return Bt / jac, Bz / jac


@run_decorator
def get_spec_polynomial_basis(Lrad, lvol, sarr):
    ns = len(sarr)

    T = np.ones((Lrad + 1, 2, ns))
    T[0][1][:] = 0.0
    T[1][0][:] = sarr

    for L in range(2, Lrad + 1):
        T[L][0] = 2 * sarr * T[L - 1][0] - T[L - 2][0]
        T[L][1] = 2 * T[L - 1][0] + 2 * sarr * T[L - 1][1] - T[L - 2][1]

    for L in range(1, Lrad + 1):
        T[L][0] -= (-1) ** L

    for L in range(Lrad + 1):
        T[L][0] /= float(L + 1)
        T[L][1] /= float(L + 1)

    return T


@run_decorator
def get_spec_R_derivatives(Rmn, Rmn_p, im, in_, vol, sarr, tarr, zarr):
    ns = len(sarr)
    nt = len(tarr)
    nz = len(zarr)
    mn = len(im)
    Rarr = np.zeros((2, ns, nt, nz))

    factor = get_spec_regularisation_factor(mn, im, vol, sarr, "G")

    for imn in range(mn):
        for it in range(nt):
            for iz in range(nz):
                cosa = np.cos(im[imn] * tarr[it] - in_[imn] * zarr[iz])

                Rarr[0, :, it, iz] += (Rmn[imn] + (Rmn_p[imn] - Rmn[imn]) * factor[imn][0]) * cosa
                Rarr[1, :, it, iz] += ((Rmn_p[imn] - Rmn[imn]) * factor[imn][1]) * cosa
    return sarr, Rarr


@run_decorator
def get_spec_regularisation_factor(mn, im, lvol, sarr, ForG):
    ns = len(sarr)
    fac = np.zeros((mn, 2, ns))
    sbar = (1 + sarr) / 2.0

    if ForG == "G":
        for j in range(mn):
            fac[j][0] = sbar
            fac[j][1] = 0.5 * 1
    elif ForG == "F":
        for j in range(mn):
            fac[j][0] = 1.0
            fac[j][1] = 0
    else:
        raise ValueError("Unsupported ForG value in get_spec_regularisation_factor.")

    return fac


@run_decorator
def get_spec_rtarr(Rac, Rbc, im, in_, mn, lvol, sarr, tarr, zarr0):
    ns = len(sarr)
    nt = len(tarr)

    Rarr = np.zeros((ns, nt))
    Tarr = np.empty((ns, nt))
    Tarr[None, :] = tarr

    fac = get_spec_regularisation_factor(mn, im, lvol, sarr, "G")

    for j in range(mn):
        for it in range(nt):
            cosa = np.cos(im[j] * tarr[it] - in_[j] * zarr0)
            Rarr[:, it] += (Rac[j] + fac[j][0] * (Rbc[j] - Rac[j])) * cosa

    return Rarr, Tarr


@run_decorator
def get_spec_vecpot(Lrad, Ate, Aze, Ato, Azo, im, in_, lvol, sarr, tarr, zarr):
    mn = len(im)
    # ns = len(sarr)
    # nt = len(tarr)
    # nz = len(zarr)

    # At = np.zeros((ns, nt, nz))
    # Az = np.zeros((ns, nt, nz))

    T = get_spec_polynomial_basis(Lrad, lvol, sarr)

    fac = get_spec_regularisation_factor(mn, im, lvol, sarr, "F")

    # for l in range(Lrad + 1):
    #     basis = T[l][0]
    #     for j in range(mn):
    #         tfac = fac[j][0]
    #         for it in range(nt):
    #             for iz in range(nz):
    #                 cosa = np.cos(im[j] * tarr[it] - in_[j] * zarr[iz])
    #                 sina = np.sin(im[j] * tarr[it] - in_[j] * zarr[iz])
    #                 At[:, it, iz] += tfac * basis * (Ate[l, j] * cosa + Ato[l, j] * sina)
    #                 Az[:, it, iz] += tfac * basis * (Aze[l, j] * cosa + Azo[l, j] * sina)

    alpha = (
        im[:, None, None] * tarr[None, :, None] - in_[:, None, None] * zarr[None, None, :]
    )  # [j,it,iz]
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    tfac = fac[:, 0, :]  # [j,s]
    basis = T[:, 0, :]  # [l,s]
    tfac_basis = tfac[None, :, :] * basis[:, None, :]  # [l,j,s]

    At = np.einsum("ljs,lj,jtz->stz", tfac_basis, Ate, cosa, optimize=True) + np.einsum(
        "ljs,lj,jtz->stz", tfac_basis, Ato, sina, optimize=True
    )
    Az = np.einsum("ljs,lj,jtz->stz", tfac_basis, Aze, cosa, optimize=True) + np.einsum(
        "ljs,lj,jtz->stz", tfac_basis, Azo, sina, optimize=True
    )

    return At, Az
