## Select whether 'numba' is used for fast evaluation of some functions
## Requires 'numba' library, but the decorators can just be commented (becomes slow though)

run_with_numba = True

if run_with_numba:
    from numba import njit

    def maybe_jit(**kwargs):
        return njit(**kwargs)
else:

    def maybe_jit(**kwargs):
        def decorator(func):
            return func

        return decorator


# fastmath=True relies on "SVML" and this seems to crash on JED (to be investigated...)
run_decorator = maybe_jit(fastmath=False, cache=True, parallel=False)