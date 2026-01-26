import numpy as np

def get_beltrami_errors(test, error_type=0, print_res=True):
    mvol = test.allglobal_mod.mvol
    
    if error_type not in [0, 1]:
        raise RuntimeError(f"Invalid {error_type=}")

    ntz = test.allglobal_mod.ntz

    start, end = divide_work_mpi(mvol, test.comm)

    with np.printoptions(formatter={'all':lambda x: f"{x:.6e}"}):
        for lvol in range(start, end):

            lquad = test.allglobal_mod.iquad[lvol]

            err_grid = np.zeros((lquad + 1, 3, ntz), dtype=np.float64, order="F")
            avg_err = np.zeros(3)
            max_err = np.zeros(3)

            test.wrapper_funcs_mod.calc_beltrami_error(lvol+1, error_type, lquad, avg_err, max_err, err_grid)

            if print_res:
                print(f"Belt. error {lvol=}     avg {avg_err}    max {max_err}")
                
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