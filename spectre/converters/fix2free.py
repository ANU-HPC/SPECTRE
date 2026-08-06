# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
#
# Authors: Christopher Berg Smiet <chris@bergtopfusion.com>
#          Erol Balkovic <erol.balkovic@epfl.ch>

"""Convert a fixed-boundary SPECTRE InputParameters to free-boundary.

The workflow separates boundary-shape generation from field computation so the
user can supply any simsopt SurfaceRZFourier as the computational boundary (e.g.
one extended with :meth:`~simsopt.geo.SurfaceRZFourier.extend_via_normal`).
"""

import warnings

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


def fix2free(
    params,
    bs,
    comp_boundary,
    nvns_mpol=None,
    nvns_ntor=None,
    filename=None,
    curpol=None,
    vacuum=False,
):
    r"""Convert a fixed-boundary SPECTRE InputParameters to free-boundary.

    The caller is responsible for building the computational boundary — e.g. via
    :func:`params_to_simsopt_boundary` followed by
    :meth:`~simsopt.geo.SurfaceRZFourier.extend_via_normal`.

    If ``params.physics.lfreebound == 1`` on input, the function replaces the
    existing computational boundary and vacuum field while leaving all other
    fields unchanged.

    The output ``lconstraint`` is chosen as follows:

    * ``vacuum=True`` → ``lconstraint=3`` (toroidal-current constraint; the
      transform is not meaningful in a vacuum equilibrium).
    * input ``lconstraint == 1`` → preserved (rotational-transform constraint).
    * otherwise → ``lconstraint=3`` (default for free-boundary).

    The sign convention is anchored to the **actual coil field**: B_φ is
    evaluated from ``bs`` at :math:`R=R_{00}, \phi=0, Z=0` and its sign is given to
    BOTH ``curpol`` and ``phiedge``. This prevents VMEC fixed boundary runs with
    unsigned Phiedge giving beleivable-but-wrong results, and ensures consistency
    with VNS, which is directly evaluated from the same biot-savart evaluation.

    if ``curpol`` (μ₀ × net poloidal linking current) is not specified, it is
    calculated from the sum of all coil currents. This assumes every coil in
    your set links the plasma exactly once in the same direction.
    All the ways coils can (un) link the plasma cannot be captured in a neat
    algorithm, so we ask you to compute this yourself.

    The output ``phiedge`` keeps the **magnitude** of ``params.physics.phiedge``
    but takes its **sign** from the coil field.  The coils also set the comp/plasma
    flux *ratio* that fills the vacuum-slot ``tflux[nvol]``.

    A good sanity check can be performed against a ground truth, buy running
    this with `vacuum=True`, and comparing the converged state with the
    Biot-Savart field directly from the coils.

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
            ``sign(B_φ_coils) × μ₀ × Σ|I_coil|``.  That formula assumes every coil
            is a modular coil that links the plasma exactly once (a warning is
            issued); pass ``curpol`` explicitly for coil sets that include
            poloidal field coils or coils linking the plasma more than once.  When
            given, only its magnitude is used.
        vacuum: if ``True``, produce a vacuum free-boundary equilibrium.
            A vacuum field carries no plasma pressure and no plasma current, this
            sets ``pressure``, ``pscale``, ``curtor``, ``ivolume`` and ``isurf``
            to zero, and ``lbnszero``
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

    phys = params.physics

    _MU0 = 4.0 * np.pi * 1e-7

    # Toroidal-field direction evaluated from the Biot-Savart.
    # We choose the point (R=R00, 0, 0) for the evaluation
    #
    _R0 = float(phys.rbc[mntor, mmpol])  # m=0,n=0 plasma-boundary major radius
    bs.set_points_cyl(np.array([[_R0, 0.0, 0.0]]))
    bphi_sign = float(np.sign(bs.B_cyl()[0, 1]))  # simsopt B_cyl columns = [B_R, B_φ, B_Z]
    if bphi_sign == 0.0:  # degenerate (no toroidal field at the centre) — keep input sense
        bphi_sign = float(np.sign(phys.phiedge)) or 1.0

    if curpol is None:
        warnings.warn(
            "curpol not given; auto-computing it as sign(B_φ_coils)·μ₀·Σ|I_coil| over all "
            "coils in bs.coils. This assumes every coil is a modular coil that links the "
            "plasma exactly once in the same direction. "
            "For coil sets with non-modular coils (toroidal-field, PF, "
            "windowpane) or coils that link the plasma more than once, this mis-counts the "
            "poloidal linking current — pass curpol explicitly in that case.",
            stacklevel=2,
        )
        # Physical total poloidal linking current magnitude = Σ|I_coil| for all coils.
        # if any coils do not link the plasma, or link it multiple times,
        # this simplified method goes wrong.
        # Since coils parametrization can be in any direction, we ensure it has
        # the same sign as B_phi (and Phiedge)
        curpol = bphi_sign * _MU0 * sum(abs(c.current.get_value()) for c in bs.coils)
    else:
        # Explicit curpol supplies only the MAGNITUDE; the sign is taken from the
        # coil field so a current-derived (arbitrary) sign cannot make curpol
        # disagree with the actual B_φ direction.
        curpol_given = float(curpol)
        if curpol_given != 0.0 and np.sign(curpol_given) != bphi_sign:
            warnings.warn(
                f"explicit curpol sign ({np.sign(curpol_given):+.0f}) disagrees with the "
                f"coil B_φ direction ({bphi_sign:+.0f}); using the coil-field sign so curpol "
                "stays consistent with the actual toroidal field.",
                stacklevel=2,
            )
        curpol = bphi_sign * abs(curpol_given)
    nvol = phys.nvol
    nfp = phys.nfp
    mpol = phys.mpol
    ntor = phys.ntor

    if nvns_mpol is None:
        nvns_mpol = mpol
    if nvns_ntor is None:
        nvns_ntor = ntor

    # Choose output lconstraint: 3 for vacuum runs, otherwise preserve input
    # lconstraint=1 (transform constraint) and default to 3 for anything else.
    if vacuum:
        out_lconstraint = 3
    elif phys.lconstraint == 1:
        out_lconstraint = 1
    else:
        out_lconstraint = 3

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
    # The flux magnitude at the plasma boundary is taken from the input,
    # its sign is setfrom the coil B_φ direction.
    # We use the coils-evaluated fluxes only for the comp/plasma flux *ratio*.
    tflux_plasma_input = bphi_sign * abs(float(phys.phiedge))
    tflux_plasma_coils = float(ToroidalFlux(plasma_boundary, bs).J())
    tflux_comp_coils = float(ToroidalFlux(comp_boundary, bs).J())

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
    # Fortran normalizes tflux ← tflux / tflux(Nvol).  Renormalize the inherited
    # tflux so the plasma boundary  is exactly 1.0; then the vacuum slot is the
    # coil-derived ratio tflux_comp_coils / tflux_plasma_coils.
    new_tflux = phys.tflux.copy() / float(phys.tflux[nvol - 1])
    new_tflux[nvol] = tflux_comp_coils / tflux_plasma_coils

    # Initial guess for the vacuum slot of pflux. Used as a starting value when
    # lconstraint=1 (where pflux is varied) and harmless otherwise.
    new_pflux = phys.pflux.copy()
    new_pflux[nvol] = new_pflux[nvol - 1]

    # Match the vacuum volume's radial resolution to the plasma's lrad per unit
    # toroidal flux. tflux is cumulative, so the vacuum slot holds the difference
    # new_tflux[nvol]-new_tflux[nvol-1]; phiedge cancels (normalized units).
    # Bounded between the plasma lrad and double it, so a large vacuum flux can't
    # blow up the vacuum resolution.
    new_lrad = phys.lrad.copy()
    plasma_lrad = int(new_lrad[nvol - 1])
    min_vac_lrad = plasma_lrad
    max_vac_lrad = 2 * plasma_lrad
    vac_flux = new_tflux[nvol] - new_tflux[nvol - 1]
    plasma_flux_per_vol = new_tflux[nvol - 1] / nvol
    scaled_vac_lrad = int(round(float(plasma_lrad) * vac_flux / plasma_flux_per_vol))
    new_lrad[nvol] = min(max_vac_lrad, max(min_vac_lrad, scaled_vac_lrad))

    # --- build updated physics ---
    # rbc / zbs (plasma boundary) are intentionally left unchanged — this is the
    # key fix over spectre_set_vns.ipynb, which mistakenly wrote the extended boundary
    # into both irbc[:,-1] (comp) and irbc[:,-2] (plasma).
    physics_update = {
        "lfreebound": 1,
        "lbdybnzero": True,
        "lconstraint": out_lconstraint,
        "rwc": rwc_arr,
        "zws": zws_arr,
        "vns": vns_arr,
        "curpol": curpol,
        "phiedge": tflux_plasma_input,
        "tflux": new_tflux,
        "pflux": new_pflux,
        "lrad": new_lrad,
    }
    if out_lconstraint == 1:
        # With lfreebound=1 the iota/oita arrays are serialized at length nvol+2
        # (index 0 ignored, indices 1..nvol+1 mapped to interfaces). Copy the
        # values into the new arrays.
        new_iota = phys.iota.copy()
        new_iota[nvol + 1] = new_iota[nvol]
        new_oita = phys.oita.copy()
        new_oita[nvol + 1] = new_oita[nvol]
        physics_update["iota"] = new_iota
        physics_update["oita"] = new_oita
    # --- pressure / current content: a vacuum field carries none of it ---
    # A vacuum equilibrium has no plasma pressure and no plasma current of any
    # kind, so pressure, pscale, the net toroidal current (curtor) and both the
    # volume (ivolume) and interface (isurf) current profiles are forced to zero,
    # and lbnszero is set (a prescribed Bns is meaningless in vacuum).
    # For a non-vacuum run every one of these
    # is carried over unchanged from the input fixed-boundary equilibrium
    # and extended where necessary.
    if vacuum:
        physics_update.update(
            {
                "pressure": np.zeros_like(phys.pressure),
                "pscale": 0.0,
                "curtor": 0.0,
                "ivolume": np.zeros_like(phys.ivolume),
                "isurf": np.zeros_like(phys.isurf),
                "lbnszero": True,
            }
        )
    else:
        new_ivolume = phys.ivolume.copy()
        new_ivolume[nvol] = new_ivolume[nvol - 1]
        physics_update["ivolume"] = new_ivolume

    new_phys = phys.model_copy(update=physics_update)

    # iBns bootstrapping is a Python-side decision now: request it per run with
    # SPECTRE.update_bns(mode="initialize") or minimization.initialize_bns_from_current.
    new_numeric = params.numeric.model_copy(update={"lvcgrid": 1})

    # Use model_copy on the top-level params
    result = params.model_copy(
        update={
            "physics": new_phys,
            "numeric": new_numeric,
        }
    )

    if filename is not None:
        write_input_parameters_to_toml(result, filename)

    return result


def _parse_mn_dict(mapping):
    """Parse a ``{'(m, n)': value}`` Fourier dict into ``[(m, n, value), ...]``."""
    modes = []
    for key, value in mapping.items():
        m_str, n_str = str(key).strip("()").split(",")
        modes.append((int(m_str), int(n_str), float(value)))
    return modes


def _vacuum_flux_corrected_rho_sq(
    comp_rbc, comp_zbs, nfp, target_fracs, n_rho=40, ntheta=192, nzeta=48, bcm_m=6
):
    """Place interfaces by *actual* vacuum toroidal flux instead of ``rho = sqrt(tflux)``.

    ``rho = sqrt(tflux)`` assumes the enclosed flux grows as ``rho**2`` — exact only
    for a uniform field.  In the vacuum region the toroidal field is ``B_phi = G/R``,
    so the toroidal flux through the conformal disc of radius ``rho`` is the pure-
    geometry line integral (Green's theorem on ``∫∫ (1/R) dR dZ``)

        Psi(rho) ∝ < ∮_{curve(rho)} ln(R) dZ >_zeta,

    which is weaker for the outer shells (larger ``R``) than ``rho**2`` predicts.
    We build ``Psi(rho)`` on a dense ``rho`` grid by mapping the comp boundary
    inward, normalise by ``Psi`` at the comp boundary (``rho = 1``) and invert so
    each interface sits at the ``rho`` where the coil field actually produces its
    target flux fraction — moving every interface (and the plasma boundary) closer
    to the comp boundary.

    Args:
        comp_rbc: comp-boundary R cosine harmonics as a ``{'(m, n)': value}`` dict.
        comp_zbs: comp-boundary Z sine harmonics as a ``{'(m, n)': value}`` dict.
        nfp: number of field periods.
        target_fracs: interface fluxes as fractions of the comp-boundary flux.
        n_rho: number of radial samples used to build the flux profile.
        ntheta: number of poloidal samples for the line integral.
        nzeta: number of toroidal samples (the profile is averaged over zeta).
        bcm_m: number of Fourier modes used by the BCM map.

    Returns:
        ``rho_corr ** 2`` for each entry of ``target_fracs`` — feed directly to
        :func:`compute_interface_guesses_map2disc` as ``tfluxes`` (it takes
        ``rho = sqrt(tfluxes)``).
    """
    from spectre.mesh_gen.map2disc import map_boundary_to_interfaces

    r_modes = _parse_mn_dict(comp_rbc)
    z_modes = _parse_mn_dict(comp_zbs)

    def make_boundary(zeta):
        def boundary(theta):
            th = np.asarray(theta, dtype=float)
            R = np.zeros_like(th)
            Z = np.zeros_like(th)
            for m, n, v in r_modes:
                R += v * np.cos(m * th - n * nfp * zeta)
            for m, n, v in z_modes:
                Z += v * np.sin(m * th - n * nfp * zeta)
            return R, Z

        return boundary

    # Same negated-theta convention as gen_infaces / compute_interface_guesses_map2disc.
    th_arr = -np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta_arr = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)
    # Skip rho=0 (the magnetic-axis point) to keep the disc non-degenerate.
    rho = np.linspace(1.0 / n_rho, 1.0, n_rho)

    R, Z = map_boundary_to_interfaces(make_boundary, rho, th_arr, zeta_arr, bcm_m)

    # Psi(rho, zeta) = ∮ ln(R) dZ ; central-difference dZ/dtheta on the closed curve.
    dZ = 0.5 * (np.roll(Z, -1, axis=-1) - np.roll(Z, 1, axis=-1))
    psi = np.abs(np.mean(np.sum(np.log(R) * dZ, axis=-1), axis=-1))
    psi_norm = psi / psi[-1]

    rho_corr = np.interp(np.asarray(target_fracs, dtype=float), psi_norm, rho)
    return rho_corr**2


def fix2free_refresh_infaces(
    params,
    bs,
    comp_boundary,
    nvns_mpol=None,
    nvns_ntor=None,
    filename=None,
    curpol=None,
    vacuum=False,
    flux_weighted_rho=True,
):
    """Like :func:`fix2free`, but regenerate every interface from the comp boundary.

    Plain :func:`fix2free` keeps the plasma boundary (``rbc``/``zbs``) and the
    inherited internal interface guesses untouched.  When the computational
    boundary is built by extending/condensing the plasma boundary, those inherited
    surfaces may not nest cleanly inside the new comp boundary and the resulting
    grid in the vacuum region is garbage.

    This variant runs :func:`fix2free` for all the shared work (``vns``, ``tflux``,
    ``curpol``, ``rwc``/``zws``, pressure/current, ``lconstraint`` …) and then
    **rewrites the plasma boundary and all internal interface geometries anew** by
    conformally mapping (``map2disc`` / BCM) the computational boundary inward to
    each interface position.  The overall scale comes from the same coil-derived
    tflux ratio that :func:`fix2free` uses for the vacuum slot; the radial
    placement of each interface is set by ``flux_weighted_rho`` (see below), so the
    regenerated surfaces are smooth and consistent with the comp boundary.

    Args:
        params: see :func:`fix2free`.
        bs: see :func:`fix2free`.
        comp_boundary: see :func:`fix2free`.
        nvns_mpol: see :func:`fix2free`.
        nvns_ntor: see :func:`fix2free`.
        filename: see :func:`fix2free`.
        curpol: see :func:`fix2free`.
        vacuum: see :func:`fix2free`.
        flux_weighted_rho: if ``True`` (default), place each interface at the
            conformal radius where the *vacuum* toroidal field ``B_phi = G/R``
            actually encloses its target flux (see
            :func:`_vacuum_flux_corrected_rho_sq`), instead of the uniform-field
            ``rho = sqrt(tflux)`` guess.  This pushes the interfaces (and the
            plasma boundary) outward toward the comp boundary, where they belong,
            and avoids an over-large vacuum region.

    Returns:
        Validated :class:`~spectre.file_io.input_parameters.InputParameters` with
        ``lfreebound = 1`` and freshly mapped interface geometry.

    Raises:
        ImportError: if simsopt (via :func:`fix2free`) or map2disc is not installed.
    """
    from spectre.file_io import (
        input_parameters_from_dict,
        input_parameters_to_dict,
        write_input_parameters_to_toml,
    )
    from spectre.file_io.input_parameters import mmpol, mntor

    from .compute_guesses_map2disc import compute_interface_guesses_map2disc

    # --- shared free-boundary conversion (defer file write until after refresh) ---
    result = fix2free(
        params,
        bs,
        comp_boundary,
        nvns_mpol=nvns_mpol,
        nvns_ntor=nvns_ntor,
        filename=None,
        curpol=curpol,
        vacuum=vacuum,
    )

    phys = result.physics
    nvol = phys.nvol
    nfp = phys.nfp
    mpol = phys.mpol
    ntor = phys.ntor

    # --- interface flux fractions relative to the comp boundary ---
    # result.tflux is normalised so the plasma slot (nvol-1) is 1.0 and the
    # vacuum slot (nvol) holds the coil-derived ratio tflux_comp / tflux_plasma.
    # Dividing the interface fluxes by the vacuum slot gives their fraction of the
    # comp-boundary flux; map2disc maps the comp boundary inward at rho=sqrt(tflux).
    iface_tflux_norm = np.asarray(phys.tflux[:nvol], dtype=float) / float(phys.tflux[nvol])

    # --- comp boundary Fourier modes → SPECTRE-convention dicts ---
    comp_rbc = {}
    comp_zbs = {}
    for m in range(mpol + 1):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n < 0:
                continue
            rc = float(phys.rwc[mntor + n, mmpol + m])
            zs = float(phys.zws[mntor + n, mmpol + m])
            if rc != 0.0:
                comp_rbc[f"({m}, {n})"] = rc
            if zs != 0.0 and not (m == 0 and n == 0):
                comp_zbs[f"({m}, {n})"] = zs

    # --- choose the conformal radii for the interfaces ---
    # rho = sqrt(tflux) assumes a uniform field; flux_weighted_rho instead inverts
    # the actual vacuum-flux profile (B_phi = G/R) so the interfaces land where the
    # coil field really produces their flux — closer to the comp boundary.
    if flux_weighted_rho:
        tfluxes_for_map = _vacuum_flux_corrected_rho_sq(comp_rbc, comp_zbs, nfp, iface_tflux_norm)
    else:
        tfluxes_for_map = iface_tflux_norm

    # --- conformally map the comp boundary inward to every interface ---
    allrzrz = compute_interface_guesses_map2disc(
        comp_rbc, comp_zbs, nfp, tfluxes_for_map, mpol, ntor
    )

    # The outermost mapped surface (largest rho) is the new plasma boundary.
    plasma_iface = allrzrz[f"interface_{nvol}"]

    # --- apply via the validated round-trip ---
    # model_copy(update=...) does NOT run the mode="before" set_guess validator that
    # converts an allrzrz *mapping* into the internal allrzrz/mmrzrz/nnrzrz arrays.
    # Serialize → mutate the human-readable dict → re-validate so set_guess runs.
    data = input_parameters_to_dict(result)
    data["physics"]["rbc"] = plasma_iface["rbc"]
    data["physics"]["zbs"] = plasma_iface["zbs"]
    data["physics"]["allrzrz"] = allrzrz
    # Use the supplied interface surfaces instead of re-interpolating from the boundary.
    data["numeric"]["linitialize"] = 0

    result_refreshed = input_parameters_from_dict(data)

    if filename is not None:
        write_input_parameters_to_toml(result_refreshed, filename)

    return result_refreshed


def plot_freeboundary_modes(bs, params, ntheta=248, nzeta=252, log_scale=False, vns_rtol=1e-3):
    """Plot B·n mode spectrum and B·n map for a free-boundary setup.

    Evaluates B·n on the computational boundary, FFTs it at twice the stored
    resolution (−2·ntor … 2·ntor × 0 … 2·mpol), and draws a red box around
    the modes that are actually stored in ``params`` (−ntor … ntor × 0 … mpol).
    The truncation error — the fraction of the B·n spectral RMS that falls
    outside that box — is printed and shown in the spectrum panel's legend,
    and the freshly evaluated modes are checked against the ``vns`` already
    stored in ``params``.

    Args:
        bs: simsopt :class:`~simsopt.field.BiotSavart`.
        params: :class:`~spectre.file_io.input_parameters.InputParameters` with
            ``lfreebound == 1`` (``rwc``/``zws`` populated).
        ntheta: theta quadrature points for the B·n evaluation.
        nzeta: zeta quadrature points per field period for the B·n evaluation.
        log_scale: if ``True``, colour-map shows log₁₀ of the mode amplitude,
            floored at ``1e-7``.  Default ``False`` plots the raw amplitude,
            which makes the spectrum decay and the dominant modes easier to
            compare.
        vns_rtol: tolerance for the ``vns`` consistency check, as a peak-normalised
            max deviation between the stored and re-evaluated modes.  The match is
            not bitwise: B·n is quadrature-sampled and the comp boundary is rebuilt
            from the stored (truncated) ``rwc``/``zws``, so the modes shift slightly
            when this grid or resolution differs from the one used at creation.  For
            the standard same-resolution workflow the deviation is ~1e-5; raise this
            if the comp boundary was built at higher resolution than ``mpol``/``ntor``.

    Returns:
        matplotlib Figure with two panels: mode spectrum (left) and B·n map (right).

    Raises:
        ValueError: if ``params.physics.vns`` deviates from the B·n modes evaluated
            here by more than ``vns_rtol`` (peak-normalised).
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from spectre.file_io.input_parameters import mmpol, mntor
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

    # In one pass: the 2x-resolution amplitude map (rows m=0..2*mpol, cols
    # n=-2*ntor..2*ntor) and the vns modes the stored (mpol, ntor) box should hold.
    amp = np.zeros((nvns_mpol + 1, 2 * nvns_ntor + 1))
    vns_eval = np.zeros_like(params.physics.vns)
    for m, n, ce, co in zip(ims, ins, evenb, oddb, strict=True):
        m, n = int(m), int(n)
        amp[m, n + nvns_ntor] = np.hypot(ce, co)
        if m <= mpol and abs(n) <= ntor and not (m == 0 and n == 0):
            vns_eval[mntor + n, mmpol + m] = -co  # fix2free sign convention

    # Truncation error: fraction of the B·n spectral RMS that falls outside the box.
    kept = amp[: mpol + 1, nvns_ntor - ntor : nvns_ntor + ntor + 1]
    total = np.linalg.norm(amp)
    trunc = np.sqrt(max(total**2 - np.sum(kept**2), 0.0)) / total if total else 0.0
    print(
        f"B·n truncation at (mpol={mpol}, ntor={ntor}): "
        f"{trunc:.2%} of spectral RMS lies outside the kept band"
    )

    # The stored vns must match the freshly evaluated modes (see vns_rtol).
    box = (slice(mntor - ntor, mntor + ntor + 1), slice(mmpol, mmpol + mpol + 1))
    stored = params.physics.vns[box]
    scale = max(np.abs(stored).max(), np.abs(vns_eval[box]).max())
    deviation = np.abs(vns_eval[box] - stored).max() / scale if scale else 0.0
    if deviation > vns_rtol:
        raise ValueError(
            f"params.physics.vns is inconsistent with the B·n evaluated here: "
            f"peak-normalised deviation {deviation:.2%} exceeds vns_rtol={vns_rtol:.1e}. "
            f"The stored free-boundary field does not match this coil set and boundary "
            f"(raise vns_rtol if the comp boundary was built at higher resolution than "
            f"mpol/ntor)."
        )

    fig, (ax_spec, ax_bdotn) = plt.subplots(1, 2, figsize=(14, 5))

    # --- mode spectrum ---
    n_axis = np.arange(-nvns_ntor, nvns_ntor + 1)
    m_axis = np.arange(nvns_mpol + 1)
    if log_scale:
        plot_data = np.log10(np.maximum(amp, 1e-7))
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
        label=f"stored (mpol={mpol}, ntor={ntor})\n{trunc:.2%} of B·n RMS outside",
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
