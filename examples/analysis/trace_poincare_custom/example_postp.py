#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
import numpy as np

from spectre import SPECTREout

zetan = 0.1875

# Test case 1: A fixed-b. stellarator with 7 volumes
out_obj = SPECTREout("fixb_7vol.h5")
num_transits = 1000
num_trajectories = 20

# # Test case 2: A free-boundary stellarator with 1 plasma and 1 vacuum volume 
# (some points will leave the plasma)
# out_obj = SPECTREout('freeb_2vol.sp.h5')
# num_transits = 300
# num_trajectories = 10

# Initialize Poincare plot with trajectories regularly spaced between interfaces along \t=0, 
# like in poincare_mod.F90
init_pts = [
    np.array(
        [
            [(-1.0 + 1e-14) + itrj * (2.0 - 2e-14) / num_trajectories, 0.0]
            for itrj in range(0 if j > 0 else 1, num_trajectories + 1)
        ]
    )
    for j in range(out_obj.input.physics.Mvol)
]

num_phi_planes = out_obj.grid.Nz

out_obj.trace_fieldlines(
    init_pts,
    num_phi_planes=num_phi_planes,
    num_transits=num_transits,
    method="RK45",
    rtol=1.0e-7,
    atol=1.0e-7,
)

# Plot results, and compare to old poincare
if out_obj.tracer.comm.Get_rank() == 0:
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    out_obj.plot_infaces(ax=ax, color="k", zetan=zetan, prange="full")
    out_obj.plot_poincare(ax=ax, c="blue", zetan=zetan, ms=1.4, prange="full")
    out_obj.tracer.plot(ax=ax, c="red", zetan=zetan, ms=1.2, prange="full")
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    out_obj.plot_iota(
        ax=ax,
        xaxis="s",
    )
    out_obj.tracer.plot_iota(ax=ax, xaxis="s", marker="x", ls="--")
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    out_obj.plot_iota(
        ax=ax,
        xaxis="R",
    )
    out_obj.tracer.plot_iota(ax=ax, xaxis="R", marker="x", ls="--")
    plt.tight_layout()

    plt.show()
