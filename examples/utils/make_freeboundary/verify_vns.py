# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
"""
Verify that fix2free gives the same VNS coefficients as the
manual workflow in orig_vnsset.ipynb.

Run from this directory (which contains input.W7-X_standard_configuration):
    python verify_vns.py
"""

import numpy as np
import matplotlib.pyplot as plt
from simsopt.configs.zoo import get_data
from simsopt.mhd import Vmec

from spectre.converters import simsvmec2spectre
from spectre.converters.fix2free import (
    _compute_bdotn,
    fix2free,
    params_to_simsopt_boundary,
)
from spectre.file_io.input_parameters import _dict_to_specarray, mmpol, mntor
from spectre import spec_fft_fast


# ── 1. Build shared inputs ────────────────────────────────────────────────────
print("Loading W7-X coils and VMEC equilibrium...")
_, _, _, _, bs_obj = get_data("w7x")
vmec = Vmec("input.W7-X_standard_configuration")
params = simsvmec2spectre(vmec, nvol=1, mpol=11, ntor=11)

plasma_boundary = params_to_simsopt_boundary(params, which="plasma")
comp_boundary = plasma_boundary.copy()
comp_boundary.extend_via_normal(-0.15)

nfp = comp_boundary.nfp
nzeta_full = len(comp_boundary.quadpoints_phi)
nzeta_1period = nzeta_full // nfp
ntheta = len(comp_boundary.quadpoints_theta)
print(f"  nfp={nfp}, nzeta_full={nzeta_full}, nzeta_1period={nzeta_1period}, ntheta={ntheta}")


# ── 2. Manual B·n (orig_vnsset.ipynb approach) ───────────────────────────────
print("\nComputing B·n manually (orig_vnsset approach)...")


def surf_gamma_norm_manual(surf):
    """Replicate orig_vnsset.ipynb surf_gamma_norm using the simsopt surface object."""
    _nfp = surf.nfp
    _nz_full = len(surf.quadpoints_phi)
    _nt = len(surf.quadpoints_theta)
    # zarr covers the full torus in physical radians [0, 2π]
    zarr_full = np.linspace(0, 2 * np.pi, _nz_full, endpoint=False)
    # tarr covers theta [0, 2π]
    tarr = np.linspace(0, 2 * np.pi, _nt, endpoint=False)
    rc = surf.rc
    zs = surf.zs
    ntor_surf = (rc.shape[1] - 1) // 2

    pos = np.zeros((3, _nz_full, _nt))
    dr_dt = np.zeros_like(pos)
    dr_dz = np.zeros_like(pos)

    for n_idx in range(rc.shape[1]):
        cosz = np.cos(zarr_full[:, None])
        sinz = np.sin(zarr_full[:, None])
        for m in range(rc.shape[0]):
            # physical toroidal mode number
            n_phys = (n_idx - ntor_surf) * _nfp
            arg = m * tarr[None, :] - n_phys * zarr_full[:, None]
            cosarg = np.cos(arg)
            sinarg = np.sin(arg)

            pos[0] += rc[m, n_idx] * cosarg * cosz
            pos[1] += rc[m, n_idx] * cosarg * sinz
            pos[2] += zs[m, n_idx] * sinarg

            dr_dt[0] += -m * rc[m, n_idx] * sinarg * cosz
            dr_dt[1] += -m * rc[m, n_idx] * sinarg * sinz
            dr_dt[2] += m * zs[m, n_idx] * cosarg

            dr_dz[0] += n_phys * rc[m, n_idx] * sinarg * cosz - rc[m, n_idx] * cosarg * sinz
            dr_dz[1] += n_phys * rc[m, n_idx] * sinarg * sinz + rc[m, n_idx] * cosarg * cosz
            dr_dz[2] += -n_phys * zs[m, n_idx] * cosarg

    # dr_dt × dr_dz, then scale to match simsopt normal() convention
    norm = np.cross(dr_dt, dr_dz, axis=0)
    norm *= -4.0 * np.pi**2
    return pos, norm


pos_manual, norm_manual = surf_gamma_norm_manual(comp_boundary)
gamma_flat = pos_manual.reshape(3, -1).T.copy(order="C")
normal_flat = norm_manual.reshape(3, -1).T.copy(order="C")

bs_obj.set_points(gamma_flat)
B_manual = bs_obj.B()
bdotn_manual_flat = (B_manual * normal_flat).sum(axis=-1) / (2.0 * np.pi) ** 2
bdotn_manual = bdotn_manual_flat.reshape(nzeta_full, ntheta)


# ── 3. Current B·n via _compute_bdotn ────────────────────────────────────────
print("Computing B·n via _compute_bdotn...")
bdotn_current, qp_phi, qp_theta = _compute_bdotn(bs_obj, comp_boundary)


# ── 4. Compare B·n ───────────────────────────────────────────────────────────
diff_bdotn = bdotn_manual[:nzeta_1period] - bdotn_current[:nzeta_1period]
max_diff_bdotn = np.abs(diff_bdotn).max()
print(f"\n[B·n]  max|manual − _compute_bdotn| = {max_diff_bdotn:.3e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
im0 = axes[0].pcolormesh(bdotn_manual[:nzeta_1period], shading="auto")
axes[0].set_title("B·n manual (orig_vnsset)")
plt.colorbar(im0, ax=axes[0])
im1 = axes[1].pcolormesh(bdotn_current[:nzeta_1period], shading="auto")
axes[1].set_title("B·n _compute_bdotn (current)")
plt.colorbar(im1, ax=axes[1])
im2 = axes[2].pcolormesh(diff_bdotn, shading="auto")
axes[2].set_title(f"Difference  max|Δ|={max_diff_bdotn:.2e}")
plt.colorbar(im2, ax=axes[2])
fig.tight_layout()
plt.savefig("verify_vns_bdotn.png", dpi=120)
print("  Saved verify_vns_bdotn.png")


# ── 5. Compare FFT modes ──────────────────────────────────────────────────────
print("\nComparing FFT modes...")
nvns_mpol, nvns_ntor = 10, 10
tarr_fft = 2.0 * np.pi * np.asarray(qp_theta)
zarr_fft = 2.0 * np.pi * np.asarray(qp_phi[:nzeta_1period])

_, oddb_manual, ims, ins = spec_fft_fast(
    tarr_fft, zarr_fft, bdotn_manual[:nzeta_1period], nvns_mpol, nvns_ntor
)
_, oddb_current, _, _ = spec_fft_fast(
    tarr_fft, zarr_fft, bdotn_current[:nzeta_1period], nvns_mpol, nvns_ntor
)
max_diff_oddb = np.abs(oddb_manual - oddb_current).max()
print(f"[FFT]  max|oddb_manual − oddb_current| = {max_diff_oddb:.3e}")


# ── 6. Build vns arrays and compare ──────────────────────────────────────────
print("\nRunning fix2free and comparing VNS arrays...")
result = fix2free(
    params, bs_obj, comp_boundary, nvns_mpol=nvns_mpol, nvns_ntor=nvns_ntor
)
vns_converter = result.physics.vns

vns_dict_manual = {
    f"({int(m)}, {int(n)})": -float(c)
    for m, n, c in zip(ims, ins, oddb_manual)
    if abs(c) >= 1e-12 and not (m == 0 and n == 0)
}
vns_manual_arr = _dict_to_specarray(vns_dict_manual)

nt_phys = result.physics.ntor
mp_phys = result.physics.mpol
sl_n = slice(mntor - nt_phys, mntor + nt_phys + 1)
sl_m = slice(mmpol, mmpol + mp_phys + 1)
diff_vns = vns_converter[sl_n, sl_m] - vns_manual_arr[sl_n, sl_m]
max_diff_vns = np.abs(diff_vns).max()
print(f"[VNS]  max|converter − manual| = {max_diff_vns:.3e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(range(len(oddb_manual)), oddb_manual, s=6, label="manual")
axes[0].scatter(range(len(oddb_current)), oddb_current, s=6, marker="x", label="current")
axes[0].set_title("oddb coefficients: manual vs current")
axes[0].legend()
im3 = axes[1].pcolormesh(vns_converter[sl_n, sl_m], shading="auto")
axes[1].set_title("vns [converter]")
plt.colorbar(im3, ax=axes[1])
im4 = axes[2].pcolormesh(diff_vns, shading="auto")
axes[2].set_title(f"vns diff  max|Δ|={max_diff_vns:.2e}")
plt.colorbar(im4, ax=axes[2])
fig.tight_layout()
plt.savefig("verify_vns_modes.png", dpi=120)
print("  Saved verify_vns_modes.png")

# ── 7. Summary ────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"  B·n  max|Δ| = {max_diff_bdotn:.3e}  {'OK' if max_diff_bdotn < 1e-8 else 'MISMATCH'}")
print(f"  oddb max|Δ| = {max_diff_oddb:.3e}  {'OK' if max_diff_oddb < 1e-8 else 'MISMATCH'}")
print(f"  vns  max|Δ| = {max_diff_vns:.3e}  {'OK' if max_diff_vns < 1e-8 else 'MISMATCH'}")
