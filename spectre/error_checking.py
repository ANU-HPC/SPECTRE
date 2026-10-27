# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np


def get_beltrami_errors(test, error_type=0, print_res=True):
    mvol = test.allglobal_mod.mvol

    if error_type not in [0, 1]:
        raise RuntimeError(f"Invalid {error_type=}")

    ntz = test.allglobal_mod.ntz
    mvol = test.allglobal_mod.mvol

    start, end = divide_work_mpi(mvol, test.comm)

    my_errs = []

    for lvol in range(start, end):
        lquad = test.allglobal_mod.iquad[lvol]

        err_grid = np.zeros((lquad + 1, 3, ntz), dtype=np.float64, order="F")
        avg_err = np.zeros(3)
        max_err = np.zeros(3)

        test.wrapper_funcs_mod.calc_beltrami_error(
            lvol + 1, error_type, lquad, avg_err, max_err, err_grid
        )

        my_errs.append((lvol, avg_err, max_err))

    # gather all results in rank 0
    total_errors = []
    total_errors = test.comm.gather(my_errs, root=0)
    beltrami_errors_avg = np.zeros((mvol, 3))
    beltrami_errors_max = np.zeros((mvol, 3))
    if test.comm.rank == 0:
        for proc_errs in total_errors:
            for lvol, avg_err, max_err in proc_errs:
                beltrami_errors_avg[lvol, :] = avg_err
                beltrami_errors_max[lvol, :] = max_err

    if print_res and test.comm.rank == 0:
        with np.printoptions(formatter={"all": lambda x: f"{x:.6e}"}):
            print("\nBeltrami errors (s/theta/phi):")
            for lvol in range(mvol):
                print(
                    f"{lvol=:4d}     avg {beltrami_errors_avg[lvol]}    max {beltrami_errors_max[lvol]}"
                )
            print()

    return beltrami_errors_avg, beltrami_errors_max


def beltrami_error_grid(test, lvol, error_type=0):
    """Per-grid-point Beltrami error for a single volume.

    Mirrors the ``calc_beltrami_error`` call in :func:`get_beltrami_errors` but keeps
    the full spatial residual grid (which that routine discards). Single-rank: intended
    for interactive inspection on rank 0, not the MPI-distributed reduction.

    Args:
        test (SPECTRE): A SPECTRE instance with a solved field.
        lvol (int): Volume index (0-based).
        error_type (int): 0 -> contravariant residual; 1 -> Cartesian (Igeometry==3 only).

    Returns:
        tuple: ``(err4, s_nodes)`` where ``err4`` has shape ``(lquad+1, 3, nz, nt)`` holding
        ``|curl B - mu B|`` per radial node, component (s/theta/phi) and ``(zeta, theta)``
        cell, and ``s_nodes`` has shape ``(lquad+1,)`` giving the radial coordinate of each
        node (Gauss-Legendre abscissae on ``[-1, 1]`` followed by the outer interface
        ``s = 1``).
    """
    if error_type not in [0, 1]:
        raise RuntimeError(f"Invalid {error_type=}")

    ntz = test.allglobal_mod.ntz
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    lquad = test.allglobal_mod.iquad[lvol]

    err_grid = np.zeros((lquad + 1, 3, ntz), dtype=np.float64, order="F")
    avg_err = np.zeros(3)
    max_err = np.zeros(3)

    test.wrapper_funcs_mod.calc_beltrami_error(
        lvol + 1, error_type, lquad, avg_err, max_err, err_grid
    )

    # ntz flattens as (zeta, theta) with theta fastest (m = j + k*Nt), matching plot_true_mesh.
    err4 = err_grid.reshape(lquad + 1, 3, nz, nt)

    # Radial nodes: Gauss-Legendre abscissae on [-1, 1] (CDGQF itype=1), then s = 1 (jquad=lquad+1).
    s_nodes = np.concatenate([np.polynomial.legendre.leggauss(lquad)[0], [1.0]])

    return err4, s_nodes


def print_all_errors(test, beltrami=True, force=True):

    beltrami_errors_avg = beltrami_errors_max = f_errs_rms = f_errs_max = None

    if not beltrami and not force:
        return None, None, None, None

    if beltrami:
        beltrami_errors_avg, beltrami_errors_max = get_beltrami_errors(test, print_res=False)

    if test.comm.rank != 0:
        return None, None, None, None

    if force:
        f_shpd = test.f_real_shpd
        f_errs_rms = np.sqrt(np.mean(f_shpd**2, axis=(1, 2)))
        f_errs_max = np.max(np.abs(f_shpd), axis=(1, 2))

    mvol = test.allglobal_mod.mvol

    def spacer(n):
        return "".join([" "] * n)

    with np.printoptions(formatter={"all": lambda x: f"{x:.3e}"}):
        if beltrami:
            print(f"\n       FIELD ERROR{spacer(74)}FORCE ERROR\n vol{spacer(78)}inface")
            for lvol in range(mvol):
                print(
                    f"{lvol:3d}    avg {beltrami_errors_avg[lvol]}   max {beltrami_errors_max[lvol]}"
                )

                if force and lvol < (mvol - 1):
                    print(
                        f"{spacer(82)}{lvol:3d}    rms {f_errs_rms[lvol]:.3e}    max {f_errs_max[lvol]:.3e}"
                    )
            print()
        else:
            print("\nFORCE ERROR")
            for lvol in range(mvol):
                print(
                    f"\tinface {lvol:3d}     rms {f_errs_rms[lvol]:.3e}    max {f_errs_max[lvol]:.3e}"
                )
            print()

    return beltrami_errors_avg, beltrami_errors_max, f_errs_rms, f_errs_max


def divide_work_mpi(nwork, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunk = nwork // size
    remainder = nwork % size

    if rank < remainder:
        start = rank * (chunk + 1)
        end = start + chunk + 1
    else:
        start = rank * chunk + remainder
        end = start + chunk

    return start, end
