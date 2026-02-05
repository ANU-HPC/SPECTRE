from matplotlib.pylab import f
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

        test.wrapper_funcs_mod.calc_beltrami_error(lvol+1, error_type, lquad, avg_err, max_err, err_grid)

        my_errs.append((lvol, avg_err, max_err))

    # gather all results in rank 0
    total_errors = []
    total_errors = test.comm.gather(my_errs, root=0)
    beltrami_errors_avg = np.zeros((mvol, 3))
    beltrami_errors_max = np.zeros((mvol, 3))
    if test.comm.rank == 0:
        for proc_errs in total_errors:
            for (lvol, avg_err, max_err) in proc_errs:
                beltrami_errors_avg[lvol,:] = avg_err
                beltrami_errors_max[lvol,:] = max_err

    if print_res and test.comm.rank == 0:
        with np.printoptions(formatter={'all':lambda x: f"{x:.6e}"}):
            print(f"\nBeltrami errors (s/theta/phi):")
            for lvol in range(mvol):
                print(f"{lvol=:4d}     avg {beltrami_errors_avg[lvol]}    max {beltrami_errors_max[lvol]}")
            print()

    return beltrami_errors_avg, beltrami_errors_max

def print_all_errors(test, beltrami=True, force=True):

    if (not beltrami and not force):
        return
    
    if beltrami:
        beltrami_errors_avg, beltrami_errors_max = get_beltrami_errors(test, print_res=False)
    
    if test.comm.rank != 0:
        return
    
    if force:
        f_shpd = test.f_real_shpd
        f_errs = np.zeros((f_shpd.shape[0], 2))  # rms, max
        for i in range(f_shpd.shape[0]):
            f_errs[i, 0] = np.sqrt(np.mean(f_shpd[i]**2))
            f_errs[i, 1] = np.max(np.abs(f_shpd[i]))

    mvol = test.allglobal_mod.mvol

    def spacer(n):
        return ''.join([' ']*n)
    
    with np.printoptions(formatter={'all':lambda x: f"{x:.3e}"}):
        if beltrami:
            print(f'\n       FIELD ERROR{spacer(74)}FORCE ERROR\nlvol{spacer(78)}inface')
            for lvol in range(mvol):
                print(f"{lvol:3d}    avg {beltrami_errors_avg[lvol]}   max {beltrami_errors_max[lvol]}")

                if force and lvol < (mvol-1):
                    print(f"{spacer(82)}{lvol:3d}    rms {f_errs[lvol,0]:.3e}    max {f_errs[lvol,1]:.3e}")
            print()
        else:
            print('\nFORCE ERROR')
            for lvol in range(mvol):
                print(f"\tinface {lvol:3d}     rms {f_errs[lvol,0]:.3e}    max {f_errs[lvol,1]:.3e}")
            print()


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
