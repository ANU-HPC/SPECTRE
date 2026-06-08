# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Self-contained adaptive Dormand-Prince RK45 field-line integrator.

This is the permissive (MIT) field-tracing backend for ``field_tracer.py``. It
reuses the existing numba ``bfield_flow_*`` right-hand-side functions (passed in
as a first-class jitted function) and adds adaptive step-size control plus
terminal handling of the volume-edge event (|s| > 1), without any external ODE
dependency. Output is sampled exactly at the requested ``phi_eval`` planes by
forcing a sub-step onto each plane, so no dense-output interpolant is needed.
"""

import numpy as np
from numba import njit

# Dormand-Prince (RK45) Butcher tableau -- identical scheme to scipy/CyRK "RK45".
_C2, _C3, _C4, _C5 = 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9
_A21 = 1.0 / 5
_A31, _A32 = 3.0 / 40, 9.0 / 40
_A41, _A42, _A43 = 44.0 / 45, -56.0 / 15, 32.0 / 9
_A51, _A52, _A53, _A54 = 19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729
_A61, _A62, _A63, _A64, _A65 = (
    9017.0 / 3168,
    -355.0 / 33,
    46732.0 / 5247,
    49.0 / 176,
    -5103.0 / 18656,
)
# 5th-order solution weights (also the 7th stage nodes, FSAL).
_B1, _B3, _B4, _B5, _B6 = 35.0 / 384, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84
# Error = 5th-order minus 4th-order embedded weights.
_E1, _E3, _E4, _E5, _E6, _E7 = (
    71.0 / 57600,
    -71.0 / 16695,
    71.0 / 1920,
    -17253.0 / 339200,
    22.0 / 525,
    -1.0 / 40,
)


@njit(cache=True)
def _initial_step(field_func, t0, y, f0, direction, rtol, atol, args):
    """scipy-style starting step-size estimate."""
    scale0 = atol + np.abs(y) * rtol
    d0 = np.sqrt(np.mean((y / scale0) ** 2))
    d1 = np.sqrt(np.mean((f0 / scale0) ** 2))
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    y1 = y + h0 * direction * f0
    f1 = np.empty(2)
    field_func(f1, t0 + h0 * direction, y1, *args)
    d2 = np.sqrt(np.mean(((f1 - f0) / scale0) ** 2)) / h0
    h1 = max(1e-6, h0 * 1e-3) if (d1 <= 1e-15 and d2 <= 1e-15) else (0.01 / max(d1, d2)) ** 0.2
    return min(100.0 * h0, h1)


@njit(cache=True)
def integrate_rk45(field_func, y0, t_eval, args, rtol, atol):
    """Integrate dy/dphi = field_func from t_eval[0], sampling at every t_eval.

    Stops (terminal) as soon as a step leaves the domain (|s| > 1). Returns the
    (2, n) array of states at the t_eval planes reached strictly inside.
    """
    n = t_eval.shape[0]
    out = np.empty((2, n))
    y = y0.copy()
    out[:, 0] = y
    t = t_eval[0]

    f = np.empty(2)
    field_func(f, t, y, *args)
    h = _initial_step(field_func, t, y, f, 1.0, rtol, atol, args)

    k1 = np.empty(2)
    k2 = np.empty(2)
    k3 = np.empty(2)
    k4 = np.empty(2)
    k5 = np.empty(2)
    k6 = np.empty(2)
    k7 = np.empty(2)

    for i in range(1, n):
        target = t_eval[i]
        # Adaptive march from t up to the next sampling plane `target`.
        while t < target - 1e-14:
            if t + h > target:
                h = target - t
            # One Dormand-Prince step.
            field_func(k1, t, y, *args)
            field_func(k2, t + _C2 * h, y + h * _A21 * k1, *args)
            field_func(k3, t + _C3 * h, y + h * (_A31 * k1 + _A32 * k2), *args)
            field_func(k4, t + _C4 * h, y + h * (_A41 * k1 + _A42 * k2 + _A43 * k3), *args)
            field_func(
                k5, t + _C5 * h, y + h * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4), *args
            )
            y6 = y + h * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5)
            field_func(k6, t + h, y6, *args)
            y_new = y + h * (_B1 * k1 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6)
            field_func(k7, t + h, y_new, *args)

            err = h * (_E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            err_norm = np.sqrt(np.mean((err / scale) ** 2))

            if err_norm < 1.0 or h <= 1e-12:
                # Accept the step.
                t = t + h
                y = y_new
                if y[0] > 1.0 or y[0] < -1.0:
                    return out[:, :i]  # left the domain before plane i
                # Grow step for next time.
                factor = 10.0 if err_norm == 0.0 else min(10.0, 0.9 * err_norm ** (-0.2))
                h = h * factor
            else:
                # Reject and shrink.
                h = h * max(0.2, 0.9 * err_norm ** (-0.2))
        out[:, i] = y

    return out
