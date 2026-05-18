# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Convert a fixed-boundary SPECTRE InputParameters to free-boundary.

The workflow separates boundary-shape generation from field computation so the
user can supply any simsopt SurfaceRZFourier as the computational boundary (e.g.
one extended with :meth:`~simsopt.geo.SurfaceRZFourier.extend_via_normal`).
"""

import numpy as np


def params_to_simsopt_boundary(params, which="plasma", ntheta=248, nzeta_per_period=252):
    """Convert InputParameters boundary to a simsopt SurfaceRZFourier.

    Args:
        params: :class:`~spectre.file_io.input_parameters.InputParameters` pydantic object.
        which: ``'plasma'`` → uses ``rbc``/``zbs``; ``'comp'`` → uses ``rwc``/``zws``.
        ntheta: number of theta quadrature points for the full torus.
        nzeta_per_period: number of zeta quadrature points per field period.

    Returns:
        simsopt :class:`~simsopt.geo.SurfaceRZFourier`.
    """
    try:
        from simsopt.geo import SurfaceRZFourier
    except ImportError as exc:
        raise ImportError("simsopt is required. Install with: pip install simsopt") from exc

    from spectre.file_io.input_parameters import mmpol, mntor

    phys = params.physics
    mpol, ntor, nfp = phys.mpol, phys.ntor, phys.nfp
    stellsym = phys.enforce_stell_sym

    if which == "plasma":
        rcoeff = phys.rbc
        zcoeff = phys.zbs
    elif which == "comp":
        rcoeff = phys.rwc
        zcoeff = phys.zws
    else:
        raise ValueError(f"which must be 'plasma' or 'comp'; got {which!r}")

    nzeta_full = nzeta_per_period * nfp
    surf = SurfaceRZFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=np.linspace(0, 1, nzeta_full, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
    )
    surf.rc[:] = 0.0
    if stellsym:
        surf.zs[:] = 0.0

    for m in range(mpol + 1):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n < 0:
                continue
            surf.set_rc(m, n, float(rcoeff[mntor + n, mmpol + m]))
            if not (m == 0 and n == 0):
                surf.set_zs(m, n, float(zcoeff[mntor + n, mmpol + m]))
    return surf


def _compute_bdotn(bs, surf):
    """Compute B·n on a surface using Biot-Savart.

    simsopt's ``normal()`` returns ``dr/ds × dr/dt`` where s and t are the
    fractional angles in [0, 1).  This equals ``(2π)² × dr/dφ × dr/dθ`` in
    radians.  Dividing by ``(2π)²`` gives the physical normal in radians.

    Returns:
        bdotn: shape ``(nzeta_full, ntheta)``.
        qp_phi: 1-D quadpoints in [0, 1) (fraction of torus).
        qp_theta: 1-D quadpoints in [0, 1).
    """
    qp_phi = np.asarray(surf.quadpoints_phi)
    qp_theta = np.asarray(surf.quadpoints_theta)
    nzeta_full = len(qp_phi)
    ntheta = len(qp_theta)

    gamma = surf.gamma()  # (nzeta_full, ntheta, 3)
    normal = surf.normal()  # (nzeta_full, ntheta, 3) — (2π)² × dr/dφ × dr/dθ

    bs.set_points(gamma.reshape(-1, 3))
    B = bs.B().reshape(nzeta_full, ntheta, 3)
    bdotn = np.sum(B * normal, axis=-1) / (2.0 * np.pi) ** 2
    return bdotn, qp_phi, qp_theta


def fixedbound_to_freeboundary(
    params,
    bs,
    comp_boundary,
    nvns_mpol=None,
    nvns_ntor=None,
    filename=None,
    curpol=None,
    vacuum=False,
):
    """Convert a fixed-boundary SPECTRE InputParameters to free-boundary.

    The caller is responsible for building the computational boundary — e.g. via
    :func:`params_to_simsopt_boundary` followed by
    :meth:`~simsopt.geo.SurfaceRZFourier.extend_via_normal`.  This keeps the
    boundary shape as a tunable parameter separate from the converter.

    If ``params.physics.lfreebound == 1`` on input, the function replaces the
    existing computational boundary and vacuum field while leaving all other
    fields unchanged.

    ``curpol`` (μ₀ × net poloidal linking current) is normally computed
    automatically from ``bs.coils``.  For a coil set with stellarator symmetry,
    simsopt assigns opposite signs to mirror-image coils; the physical total
    poloidal current is ``μ₀ × Σ|I_coil|``.  The sign follows the SPECTRE
    convention: negative when the coil current opposes the plasma's poloidal
    field (the standard case for modular stellarator coils).  Pass ``curpol``
    explicitly to override.

    Args:
        params: :class:`~spectre.file_io.input_parameters.InputParameters` object
            (fixed- or free-boundary).
        bs: simsopt :class:`~simsopt.field.BiotSavart` built from the full coil set
            including all symmetry copies.
        comp_boundary: simsopt :class:`~simsopt.geo.SurfaceRZFourier` representing
            the computational (outer) boundary, already extended beyond the plasma.
            Its quadrature resolution is used for the B·n evaluation grid.
        nvns_mpol: poloidal Fourier resolution for the B·n modes.  Defaults to
            ``params.physics.mpol``.
        nvns_ntor: toroidal Fourier resolution for the B·n modes.  Defaults to
            ``params.physics.ntor``.
        filename: if given, write the result as a TOML file at this path.
        curpol: μ₀ × total poloidal linking current [T·m].  If ``None`` (default),
            computed automatically from ``bs.coils`` as
            ``-μ₀ × Σ|I_coil|`` (negative convention).
        vacuum: if ``True``, set ``pressure``, ``pscale``, ``curtor``, and
            ``curpol`` all to zero, producing a vacuum free-boundary equilibrium.

    Returns:
        Validated :class:`~spectre.file_io.input_parameters.InputParameters` with
        ``lfreebound = 1``.

    Raises:
        ImportError: if simsopt is not installed.
    """
    try:
        from simsopt.geo import ToroidalFlux
    except ImportError as exc:
        raise ImportError("simsopt is required. Install with: pip install simsopt") from exc

    from spectre.file_io.input_parameters import _dict_to_specarray, mmpol, mntor
    from spectre.file_io.toml_io import write_input_parameters_to_toml
    from spectre.fourier import spec_fft_fast

    _MU0 = 4.0 * np.pi * 1e-7
    if curpol is None:
        # Physical total poloidal linking current = Σ|I_coil| for all coils.
        # Stellarator-symmetric coils have opposite sign in simsopt convention;
        # taking the absolute value recovers the physical contribution of each coil.
        # The negative sign follows the SPECTRE convention for standard modular coils.
        curpol = -_MU0 * sum(abs(c.current.get_value()) for c in bs.coils)

    phys = params.physics
    nvol = phys.nvol
    nfp = phys.nfp
    mpol = phys.mpol
    ntor = phys.ntor

    if nvns_mpol is None:
        nvns_mpol = mpol
    if nvns_ntor is None:
        nvns_ntor = ntor

    # --- reconstruct plasma boundary at the same grid as comp_boundary ---
    nzeta_full_comp = len(comp_boundary.quadpoints_phi)
    ntheta_comp = len(comp_boundary.quadpoints_theta)
    plasma_boundary = params_to_simsopt_boundary(
        params,
        which="plasma",
        ntheta=ntheta_comp,
        nzeta_per_period=nzeta_full_comp // nfp,
    )

    # --- compute B·n on the computational boundary ---
    bdotn, qp_phi, qp_theta = _compute_bdotn(bs, comp_boundary)

    nzeta_full = len(qp_phi)
    nzeta_1period = nzeta_full // nfp

    # spec_fft_fast wants radians; zarr covers exactly one field period
    tarr = 2.0 * np.pi * np.asarray(qp_theta)
    zarr = 2.0 * np.pi * np.asarray(qp_phi[:nzeta_1period])
    bdotn_1period = bdotn[:nzeta_1period, :]

    # --- FFT → vns sine-mode coefficients ---
    _, oddb, ims, ins = spec_fft_fast(tarr, zarr, bdotn_1period, nvns_mpol, nvns_ntor)
    # Sign convention: vns = -oddb  (matches spectre_set_vns.ipynb)
    vns_dict = {
        f"({int(m)}, {int(n)})": -float(c)
        for m, n, c in zip(ims, ins, oddb, strict=True)
        if abs(c) >= 1e-12 and not (m == 0 and n == 0)
    }
    vns_arr = _dict_to_specarray(vns_dict)

    # --- toroidal fluxes in SI (Weber) ---
    tflux_plasma = float(ToroidalFlux(plasma_boundary, bs).J())
    tflux_comp = float(ToroidalFlux(comp_boundary, bs).J())

    # --- computational boundary modes → rwc / zws ---
    # Use min(mpol, comp_mpol) to avoid indexing beyond what's stored in the surface
    comp_mpol = min(mpol, comp_boundary.mpol)
    comp_ntor = min(ntor, comp_boundary.ntor)
    rwc_arr = np.zeros((2 * mntor + 1, 2 * mmpol + 1))
    zws_arr = np.zeros_like(rwc_arr)
    for m in range(comp_mpol + 1):
        for n in range(-comp_ntor, comp_ntor + 1):
            if m == 0 and n < 0:
                continue
            rwc_arr[mntor + n, mmpol + m] = comp_boundary.get_rc(m, n)
            if not (m == 0 and n == 0):
                zws_arr[mntor + n, mmpol + m] = comp_boundary.get_zs(m, n)

    # --- extend 1-D arrays with the vacuum-volume slot at index nvol ---
    # tflux is normalized by tflux(Nvol) in Fortran; tflux[nvol-1] should be 1.0,
    # so tflux[nvol] = tflux_comp_SI / tflux_plasma_SI gives the correct ratio.
    new_tflux = phys.tflux.copy()
    new_tflux[nvol] = tflux_comp / tflux_plasma

    new_pflux = phys.pflux.copy()
    new_pflux[nvol] = new_pflux[nvol - 1]  # pflux not used for lconstraint=3

    # Fortran enforces ivolume[Mvol] = ivolume[Mvol-1]; set explicitly for TOML clarity
    new_ivolume = phys.ivolume.copy()
    new_ivolume[nvol] = new_ivolume[nvol - 1]

    new_lrad = phys.lrad.copy()
    new_lrad[nvol] = new_lrad[nvol - 1]

    # --- build updated physics ---
    # rbc / zbs (plasma boundary) are intentionally left unchanged — this is the
    # key fix over spectre_set_vns.ipynb, which mistakenly wrote the extended boundary
    # into both irbc[:,-1] (comp) and irbc[:,-2] (plasma).
    physics_update = {
        "lfreebound": 1,
        "lbdybnzero": True,
        "lbnszero": True,
        "lconstraint": 3,
        "rwc": rwc_arr,
        "zws": zws_arr,
        "vns": vns_arr,
        "curpol": curpol,
        "phiedge": tflux_plasma,
        "tflux": new_tflux,
        "pflux": new_pflux,
        "ivolume": new_ivolume,
        "lrad": new_lrad,
    }
    if vacuum:
        physics_update.update(
            {
                "curtor": 0.0,
                "pscale": 0.0,
                "pressure": np.zeros_like(phys.pressure),
                "lbnszero": True,
            }
        )
    new_phys = phys.model_copy(update=physics_update)

    new_numeric = params.numeric.model_copy(
        update={
            "lvcgrid": 1,
            "lautoinitbn": 0,
        }
    )

    # Use model_copy on the top-level params so the mode="before" validators
    # (which call .get() on their input dict) are not re-triggered on model instances.
    result = params.model_copy(
        update={
            "physics": new_phys,
            "numeric": new_numeric,
        }
    )

    if filename is not None:
        write_input_parameters_to_toml(result, filename)

    return result


def plot_freeboundary_modes(bs, params, ntheta=248, nzeta=252, log_scale=True):
    """Plot B·n mode spectrum and B·n map for a free-boundary setup.

    Evaluates B·n on the computational boundary, FFTs it at twice the stored
    resolution (−2·ntor … 2·ntor × 0 … 2·mpol), and draws a red box around
    the modes that are actually stored in ``params`` (−ntor … ntor × 0 … mpol).

    Args:
        bs: simsopt :class:`~simsopt.field.BiotSavart`.
        params: :class:`~spectre.file_io.input_parameters.InputParameters` with
            ``lfreebound == 1`` (``rwc``/``zws`` populated).
        ntheta: theta quadrature points for the B·n evaluation.
        nzeta: zeta quadrature points per field period for the B·n evaluation.
        log_scale: if ``True`` (default), colour-map shows log₁₀ of the mode
            amplitude.  Set to ``False`` to plot the raw amplitude, which makes
            the spectrum decay and the dominant modes easier to compare.

    Returns:
        matplotlib Figure with two panels: mode spectrum (left) and B·n map (right).
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from spectre.fourier import spec_fft_fast

    nfp = params.physics.nfp
    mpol = params.physics.mpol
    ntor = params.physics.ntor
    nvns_mpol = 2 * mpol
    nvns_ntor = 2 * ntor

    comp_boundary = params_to_simsopt_boundary(
        params, which="comp", ntheta=ntheta, nzeta_per_period=nzeta
    )
    bdotn, qp_phi, qp_theta = _compute_bdotn(bs, comp_boundary)

    nzeta_1period = len(qp_phi) // nfp
    bdotn_1period = bdotn[:nzeta_1period, :]
    tarr = 2.0 * np.pi * np.asarray(qp_theta)
    zarr = 2.0 * np.pi * np.asarray(qp_phi[:nzeta_1period])

    evenb, oddb, ims, ins = spec_fft_fast(tarr, zarr, bdotn_1period, nvns_mpol, nvns_ntor)

    # Build 2D amplitude array: rows = m (0..2*mpol), cols = n (-2*ntor..2*ntor)
    amp = np.zeros((nvns_mpol + 1, 2 * nvns_ntor + 1))
    for m, n, ce, co in zip(ims, ins, evenb, oddb, strict=True):
        amp[int(m), int(n) + nvns_ntor] = np.hypot(ce, co)

    fig, (ax_spec, ax_bdotn) = plt.subplots(1, 2, figsize=(14, 5))

    # --- mode spectrum ---
    n_axis = np.arange(-nvns_ntor, nvns_ntor + 1)
    m_axis = np.arange(nvns_mpol + 1)
    if log_scale:
        plot_data = np.log10(amp + 1e-15)
        cbar_label = "log₁₀|coefficient|"
        title = "B·n Fourier spectrum (2× resolution, log scale)"
    else:
        plot_data = amp
        cbar_label = "|coefficient|"
        title = "B·n Fourier spectrum (2× resolution)"

    im_spec = ax_spec.pcolormesh(n_axis, m_axis, plot_data, shading="auto")
    plt.colorbar(im_spec, ax=ax_spec, label=cbar_label)
    rect = mpatches.Rectangle(
        (-ntor - 0.5, -0.5),
        2 * ntor + 1,
        mpol + 1,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label=f"stored (mpol={mpol}, ntor={ntor})",
    )
    ax_spec.add_patch(rect)
    ax_spec.legend(loc="upper right", fontsize=8)
    ax_spec.set_xlabel("n")
    ax_spec.set_ylabel("m")
    ax_spec.set_title(title)

    # --- B·n on the comp boundary ---
    im_bdotn = ax_bdotn.pcolormesh(tarr, zarr, bdotn_1period, shading="auto")
    plt.colorbar(im_bdotn, ax=ax_bdotn, label="B·n")
    ax_bdotn.set_xlabel("θ [rad]")
    ax_bdotn.set_ylabel("ζ [rad]")
    ax_bdotn.set_title("B·n on comp boundary (one field period)")

    fig.tight_layout()
    return fig
