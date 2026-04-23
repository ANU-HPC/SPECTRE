# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

## Written by Erol Balkovic 8/9/2024
## erol.balkovic@epfl.ch
## Extended from "scipy/optimize/_numdiff.py"

"""Routines for numerical differentiation."""

from functools import lru_cache
import numpy as np

from scipy._lib._array_api import array_namespace


def group_dense(m, n, A):
    B = A.T  # Transposed view for convenience.

    # FIXME: use np.full once pythran supports it
    groups = -np.ones(n, dtype=np.intp)
    current_group = 0

    union = np.empty(m, dtype=np.intp)

    # Loop through all the columns.
    for i in range(n):
        if groups[i] >= 0:  # A group was already assigned.
            continue

        groups[i] = current_group
        all_grouped = True

        union[:] = B[i]  # Here we store the union of grouped columns.

        for j in range(groups.shape[0]):
            if groups[j] < 0:
                all_grouped = False
            else:
                continue

            # Determine if j-th column intersects with the union.
            intersect = False
            for k in range(m):
                if union[k] > 0 and B[j, k] > 0:
                    intersect = True
                    break

            # If not, add it to the union and assign the group to it.
            if not intersect:
                union += B[j]
                groups[j] = current_group

        if all_grouped:
            break

        current_group += 1

    return groups


@lru_cache
def _eps_for_method(x0_dtype, f0_dtype, method):
    """
    Calculates relative EPS step to use for a given data type
    and numdiff step method.

    Progressively smaller steps are used for larger floating point types.

    Parameters
    ----------
    f0_dtype: np.dtype
            dtype of function evaluation

    x0_dtype: np.dtype
            dtype of parameter vector

    method: {'2-point', '3-point'}

    Returns
    -------
    EPS: float
            relative step size. May be np.float16, np.float32, np.float64

    Notes
    -----
    The default relative step will be np.float64. However, if x0 or f0 are
    smaller floating point types (np.float16, np.float32), then the smallest
    floating point type is chosen.
    """
    # the default EPS value
    EPS = np.finfo(np.float64).eps

    x0_is_fp = False
    if np.issubdtype(x0_dtype, np.inexact):
        # if you're a floating point type then over-ride the default EPS
        EPS = np.finfo(x0_dtype).eps
        x0_itemsize = np.dtype(x0_dtype).itemsize
        x0_is_fp = True

    if np.issubdtype(f0_dtype, np.inexact):
        f0_itemsize = np.dtype(f0_dtype).itemsize
        # choose the smallest itemsize between x0 and f0
        if x0_is_fp and f0_itemsize < x0_itemsize:
            EPS = np.finfo(f0_dtype).eps

    if method in ["2-point"]:
        return EPS**0.5
    elif method in ["3-point"]:
        return EPS ** (1 / 3)
    else:
        raise RuntimeError("Unknown step method, should be one of {'2-point', '3-point', 'cs'}")


def _compute_absolute_step(rel_step, x0, f0, method, dofs):
    """
    Computes an absolute step from a relative step for finite difference
    calculation.

    Parameters
    ----------
    rel_step: None or array-like
            Relative step for the finite difference calculation
    x0 : np.ndarray
            Parameter vector

    f0 : np.ndarray or scalar

    method : {'2-point', '3-point', 'cs'}

    Returns
    -------
    h : float
            The absolute step size

    Notes
    -----
    `h` will always be np.float64. However, if `x0` or `f0` are
    smaller floating point dtypes (e.g. np.float32), then the absolute
    step size will be calculated from the smallest floating point size.
    """
    # this is used instead of np.sign(x0) because we need
    # sign_x0 to be 1 when x0 == 0.
    sign_x0 = (x0 >= 0).astype(float) * 2 - 1

    x0dof = x0[dofs]

    rstep = _eps_for_method(x0.dtype, f0.dtype, method)

    if rel_step is None:
        abs_step = rstep * sign_x0[dofs] * np.maximum(1.0, np.abs(x0dof))
    else:
        # User has requested specific relative steps.
        # Don't multiply by max(1, abs(x0) because if x0 < 1 then their
        # requested step is not used.
        abs_step = rel_step * sign_x0[dofs] * np.abs(x0dof)

        # however we don't want an abs_step of 0, which can happen if
        # rel_step is 0, or x0 is 0. Instead, substitute a realistic step
        dx = (x0dof + abs_step) - x0dof
        abs_step = np.where(
            dx == 0, rstep * sign_x0[dofs] * np.maximum(1.0, np.abs(x0dof)), abs_step
        )

    return abs_step


def _prepare_bounds(bounds, x0):
    """
    Prepares new-style bounds from a two-tuple specifying the lower and upper
    limits for values in x0. If a value is not bound then the lower/upper bound
    will be expected to be -np.inf/np.inf.

    Examples
    --------
    >>> _prepare_bounds([(0, 1, 2), (1, 2, np.inf)], [0.5, 1.5, 2.5])
    (array([0., 1., 2.]), array([ 1.,  2., inf]))
    """
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, x0.shape)

    if ub.ndim == 0:
        ub = np.resize(ub, x0.shape)

    return lb, ub


def group_columns(A, order=0):
    """Group columns of a 2-D matrix for sparse finite differencing\ :footcite:p:`Curtis1974`.

    Two columns are in the same group if in each row at least one of them
    has zero. A greedy sequential algorithm is used to construct groups.

    Parameters
    ----------
    A : array_like or sparse matrix, shape (m, n)
            Matrix of which to group columns.
    order : int, iterable of int with shape (n,) or None
            Permutation array which defines the order of columns enumeration.
            If int or None, a random permutation is used with `order` used as
            a random seed. Default is 0, that is use a random permutation but
            guarantee repeatability.

    Returns
    -------
    groups : ndarray of int, shape (n,)
            Contains values from 0 to n_groups-1, where n_groups is the number
            of found groups. Each value ``groups[i]`` is an index of a group to
            which ith column assigned. The procedure was helpful only if
            n_groups is significantly less than n.

    References
    ----------

    .. footbibliography::
    """
    A = np.atleast_2d(A)
    A = (A != 0).astype(np.int32)

    if A.ndim != 2:
        raise ValueError("`A` must be 2-dimensional.")

    m, n = A.shape

    if order is None or np.isscalar(order):
        rng = np.random.RandomState(order)
        order = rng.permutation(n)
    else:
        order = np.asarray(order)
        if order.shape != (n,):
            raise ValueError("`order` has incorrect shape.")

    A = A[:, order]

    groups = group_dense(m, n, A)

    groups[order] = groups.copy()

    return groups


def approx_derivative(
    fun,
    x0,
    dofs=None,
    method="3-point",
    *args,
    rel_step=None,
    abs_step=None,
    f0=None,
    bounds=(-np.inf, np.inf),
    kwargs={},
):
    """Compute finite difference approximation of the derivatives of a
    vector-valued function.

    If a function maps from R^n to R^m, its derivatives form m-by-n matrix
    called the Jacobian, where an element (i, j) is a partial derivative of
    f[i] with respect to x[j].

    Parameters
    ----------
    fun : callable
            Function of which to estimate the derivatives. The argument x
            passed to this function is ndarray of shape (n,) (never a scalar
            even if n=1). It must return 1-D array_like of shape (m,) or a scalar.
    x0 : array_like of shape (n,) or float
            Point at which to estimate the derivatives. Float will be converted
            to a 1-D array.
    method : {'3-point', '2-point', 'cs'}, optional
            Finite difference method to use:
                    - '2-point' - use the first order accuracy forward or backward
                                              difference.
                    - '3-point' - use central difference in interior points and the
                                              second order accuracy forward or backward difference
                                              near the boundary.
                    - 'cs' - use a complex-step finite difference scheme. This assumes
                                     that the user function is real-valued and can be
                                     analytically continued to the complex plane. Otherwise,
                                     produces bogus results.
    rel_step : None or array_like, optional
            Relative step size to use. If None (default) the absolute step size is
            computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``, with
            `rel_step` being selected automatically, see Notes. Otherwise
            ``h = rel_step * sign(x0) * abs(x0)``. For ``method='3-point'`` the
            sign of `h` is ignored. The calculated step size is possibly adjusted
            to fit into the bounds.
    abs_step : array_like, optional
            Absolute step size to use, possibly adjusted to fit into the bounds.
            For ``method='3-point'`` the sign of `abs_step` is ignored. By default
            relative steps are used, only if ``abs_step is not None`` are absolute
            steps used.
    f0 : None or array_like, optional
            If not None it is assumed to be equal to ``fun(x0)``, in this case
            the ``fun(x0)`` is not called. Default is None.
    bounds : tuple of array_like, optional
            Lower and upper bounds on independent variables. Defaults to no bounds.
            Each bound must match the size of `x0` or be a scalar, in the latter
            case the bound will be the same for all variables. Use it to limit the
            range of function evaluation. Bounds checking is not implemented
            when `as_linear_operator` is True.
    args, kwargs : tuple and dict, optional
            Additional arguments passed to `fun`. Both empty by default.
            The calling signature is ``fun(x, *args, **kwargs)``.

    Returns
    -------
    J : {ndarray, sparse matrix, LinearOperator}
            Finite difference approximation of the Jacobian matrix.
            If `as_linear_operator` is True returns a LinearOperator
            with shape (m, n). Otherwise it returns a dense array or sparse
            matrix depending on how `sparsity` is defined. If `sparsity`
            is None then a ndarray with shape (m, n) is returned. If
            `sparsity` is not None returns a csr_matrix with shape (m, n).
            For sparse matrices and linear operators it is always returned as
            a 2-D structure, for ndarrays, if m=1 it is returned
            as a 1-D gradient array with shape (n,).

    See Also
    --------
    check_derivative : Check correctness of a function computing derivatives.

    Notes
    -----
    If `rel_step` is not provided, it assigned as ``EPS**(1/s)``, where EPS is
    determined from the smallest floating point dtype of `x0` or `fun(x0)`,
    ``np.finfo(x0.dtype).eps``, s=2 for '2-point' method and
    s=3 for '3-point' method. Such relative step approximately minimizes a sum
    of truncation and round-off errors, see :footcite:t:`Press2007`. Relative steps are used by
    default. However, absolute steps are used when ``abs_step is not None``.
    If any of the absolute or relative steps produces an indistinguishable
    difference from the original `x0`, ``(x0 + dx) - x0 == 0``, then a
    automatic step size is substituted for that particular entry.

    A finite difference scheme for '3-point' method is selected automatically.
    The well-known central difference scheme is used for points sufficiently
    far from the boundary, and 3-point forward or backward scheme is used for
    points near the boundary. Both schemes have the second-order accuracy in
    terms of Taylor expansion. Refer to :footcite:t:`Curtis1974` for the formulas of 3-point
    forward and backward difference schemes.

    For dense differencing when m=1 Jacobian is returned with a shape (n,),
    on the other hand when n=1 Jacobian is returned with a shape (m, 1).
    Our motivation is the following: a) It handles a case of gradient
    computation (m=1) in a conventional way. b) It clearly separates these two
    different cases. b) In all cases np.atleast_2d can be called to get 2-D
    Jacobian with correct dimensions.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.optimize._numdiff import approx_derivative
    >>>
    >>> def f(x, c1, c2):
    ...     return np.array([x[0] * np.sin(c1 * x[1]), x[0] * np.cos(c2 * x[1])])
    >>> x0 = np.array([1.0, 0.5 * np.pi])
    >>> approx_derivative(f, x0, args=(1, 2))
    array([[ 1.,  0.],
               [-1.,  0.]])

    Bounds can be used to limit the region of function evaluation.
    In the example below we compute left and right derivative at point 1.0.

    >>> def g(x):
    ...     return x**2 if x >= 1 else x
    >>> x0 = 1.0
    >>> approx_derivative(g, x0, bounds=(-np.inf, 1.0))
    array([ 1.])
    >>> approx_derivative(g, x0, bounds=(1.0, np.inf))
    array([ 2.])
    """

    # added "dofs" param
    # dofs = [0,1,2,3] # list of integers with important dofs
    if dofs is None:
        dofs = np.arange(len(x0))

    if method not in ["2-point", "3-point"]:
        raise ValueError("Unknown method '%s'. " % method)

    xp = array_namespace(x0)
    _x = np.atleast_1d(x0)
    _dtype = xp.float64
    if xp.isdtype(_x.dtype, "real floating"):
        _dtype = _x.dtype

    # promotes to floating
    x0 = xp.astype(_x, _dtype)

    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    lb, ub = _prepare_bounds(bounds, x0)

    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")

    def fun_wrapped(x):
        if xp.isdtype(x.dtype, "real floating"):
            x = xp.astype(x, x0.dtype)

        f = np.atleast_1d(fun(x, *args, **kwargs))
        if f.ndim > 1:
            raise RuntimeError("`fun` return value has more than 1 dimension.")
        return f

    if f0 is None:
        f0 = fun_wrapped(x0)
    else:
        f0 = np.atleast_1d(f0)
        if f0.ndim > 1:
            raise ValueError("`f0` passed has more than 1 dimension.")

    if np.any((x0 < lb) | (x0 > ub)):
        raise ValueError("`x0` violates bound constraints.")

    # by default we use rel_step
    if abs_step is None:
        h = _compute_absolute_step(rel_step, x0, f0, method, dofs)
    else:
        # user specifies an absolute step
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = abs_step

        # cannot have a zero step. This might happen if x0 is very large
        # or small. In which case fall back to relative step.
        dx = (x0[dofs] + h) - x0[dofs]
        h = np.where(
            dx == 0,
            _eps_for_method(x0.dtype, f0.dtype, method)
            * sign_x0[dofs]
            * np.maximum(1.0, np.abs(x0[dofs])),
            h,
        )

    # if method == '2-point':
    #     h, use_one_sided = _adjust_scheme_to_bounds(
    #         x0, h, 1, '1-sided', lb, ub)
    # elif method == '3-point':
    #     h, use_one_sided = _adjust_scheme_to_bounds(
    #         x0, h, 1, '2-sided', lb, ub)

    if method == "2-point":
        use_one_sided = np.ones_like(h, dtype=bool)
    elif method == "3-point":
        h = np.abs(h)
        use_one_sided = np.zeros_like(h, dtype=bool)

    return _dense_difference(fun_wrapped, x0, f0, h, dofs, use_one_sided, method)


def _dense_difference(fun, x0, f0, h, dofs, use_one_sided, method):
    m = f0.size
    # n = x0.size
    num_dofs = dofs.size
    J_transposed = np.empty((num_dofs, m))
    x1 = x0.copy()
    x2 = x0.copy()

    for i, dof in enumerate(dofs):
        if method == "2-point":
            x1[dof] += h[i]
            dx = x1[dof] - x0[dof]  # Recompute dx as exactly representable number.
            df = fun(x1) - f0
        elif method == "3-point" and use_one_sided[i]:
            x1[dof] += h[i]
            x1[dof] += 2 * h[i]
            dx = x2[dof] - x0[dof]
            f1 = fun(x1)
            f2 = fun(x2)
            df = -3.0 * f0 + 4 * f1 - f2
        elif method == "3-point" and not use_one_sided[i]:
            x1[dof] -= h[i]
            x2[dof] += h[i]
            dx = x2[dof] - x1[dof]
            f1 = fun(x1)
            f2 = fun(x2)
            df = f2 - f1
        else:
            raise RuntimeError("Never be here.")

        J_transposed[i] = df / dx
        x1[dof] = x2[dof] = x0[dof]

    if m == 1:
        J_transposed = np.ravel(J_transposed)

    return J_transposed.T


# def _dense_difference(fun, x0, f0, h, dofs, use_one_sided, method):
#     m = f0.size
#     n = x0.size
#     J_transposed = np.empty((n, m))
#     x1 = x0.copy()
#     x2 = x0.copy()
#     xc = x0.astype(complex, copy=True)

#     for i in range(h.size):
#     # for i in dofs:
#         if method == '2-point':
#             x1[i] += h[i]
#             dx = x1[i] - x0[i]  # Recompute dx as exactly representable number.
#             df = fun(x1) - f0
#         elif method == '3-point' and use_one_sided[i]:
#             x1[i] += h[i]
#             x2[i] += 2 * h[i]
#             dx = x2[i] - x0[i]
#             f1 = fun(x1)
#             f2 = fun(x2)
#             df = -3.0 * f0 + 4 * f1 - f2
#         elif method == '3-point' and not use_one_sided[i]:
#             x1[i] -= h[i]
#             x2[i] += h[i]
#             dx = x2[i] - x1[i]
#             f1 = fun(x1)
#             f2 = fun(x2)
#             df = f2 - f1
#         else:
#             raise RuntimeError("Never be here.")

#         J_transposed[i] = df / dx
#         x1[i] = x2[i] = xc[i] = x0[i]

#     if m == 1:
#         J_transposed = np.ravel(J_transposed)

#     return J_transposed.T


def check_derivative(fun, jac, x0, bounds=(-np.inf, np.inf), args=(), kwargs={}):
    """Check correctness of a function computing derivatives (Jacobian or
    gradient) by comparison with a finite difference approximation.

    Parameters
    ----------
    fun : callable
            Function of which to estimate the derivatives. The argument x
            passed to this function is ndarray of shape (n,) (never a scalar
            even if n=1). It must return 1-D array_like of shape (m,) or a scalar.
    jac : callable
            Function which computes Jacobian matrix of `fun`. It must work with
            argument x the same way as `fun`. The return value must be array_like
            or sparse matrix with an appropriate shape.
    x0 : array_like of shape (n,) or float
            Point at which to estimate the derivatives. Float will be converted
            to 1-D array.
    bounds : 2-tuple of array_like, optional
            Lower and upper bounds on independent variables. Defaults to no bounds.
            Each bound must match the size of `x0` or be a scalar, in the latter
            case the bound will be the same for all variables. Use it to limit the
            range of function evaluation.
    args, kwargs : tuple and dict, optional
            Additional arguments passed to `fun` and `jac`. Both empty by default.
            The calling signature is ``fun(x, *args, **kwargs)`` and the same
            for `jac`.

    Returns
    -------
    accuracy : float
            The maximum among all relative errors for elements with absolute values
            higher than 1 and absolute errors for elements with absolute values
            less or equal than 1. If `accuracy` is on the order of 1e-6 or lower,
            then it is likely that your `jac` implementation is correct.

    See Also
    --------
    approx_derivative : Compute finite difference approximation of derivative.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize._numdiff import check_derivative
    >>>
    >>>
    >>> def f(x, c1, c2):
    ...     return np.array([x[0] * np.sin(c1 * x[1]), x[0] * np.cos(c2 * x[1])])
    >>> def jac(x, c1, c2):
    ...     return np.array(
    ...         [
    ...             [np.sin(c1 * x[1]), c1 * x[0] * np.cos(c1 * x[1])],
    ...             [np.cos(c2 * x[1]), -c2 * x[0] * np.sin(c2 * x[1])],
    ...         ]
    ...     )
    >>>
    >>> x0 = np.array([1.0, 0.5 * np.pi])
    >>> check_derivative(f, jac, x0, args=(1, 2))
    2.4492935982947064e-16
    """
    J_to_test = jac(x0, *args, **kwargs)
    J_diff = approx_derivative(fun, x0, bounds=bounds, args=args, kwargs=kwargs)
    abs_err = np.abs(J_to_test - J_diff)
    return np.max(abs_err / np.maximum(1, np.abs(J_diff)))
