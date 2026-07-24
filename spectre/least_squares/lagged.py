# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Lagged-banded matrix-free trust-region iteration (helpers).

Between dense Jacobian rebuilds, ``trf.trf_no_bounds`` keeps the banded normal-equation
factorization of a stale (lagged) J as a subspace source: the lagged banded trust-region
directions (sampled at several radii, bending between gradient-like and Gauss-Newton-like),
the lagged gradient, and the recent accepted steps (``lag_hist`` deep) span a small subspace Q.
A batch of fresh Jacobian-vector products W = J Q (``force_real_jvp``: one linearised
solve per volume per direction, ~90x cheaper than a dense Jacobian build per direction)
then gives an exact current-geometry reduced model

    min 0.5 ||f + W y||^2   subject to   ||Q y|| = ||y|| <= Delta,    step = Q y,

solved with the same More alpha-iteration as the exact solver, on the small (m, k) SVD.
The predicted reduction and therefore the trust-region gain ratio come from the fresh W,
so the radius control stays honest even when the lagged subspace is poor. Convergence is
never declared from lagged quantities: any termination test that fires while the
Jacobian is stale forces a dense rebuild and a fresh recheck (see ``trf.trf_no_bounds``).
"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import qr, svd

from spectre.least_squares.common import evaluate_quadratic, solve_lsq_trust_region


def orthonormal_columns(cols, drop_tol=1e-12):
    """Orthonormal basis of span(cols), dropping zero and (near-)dependent directions.

    Args:
        cols: Iterable of 1-D arrays of equal length (entries that are None, non-finite,
            or zero are skipped).
        drop_tol: Relative pivot threshold below which a column adds no new direction.

    Returns:
        (n, k) array with orthonormal columns (k <= number of usable inputs), or None
        if no usable column remains.
    """
    cols = [np.asarray(c, dtype=float) for c in cols if c is not None]
    if cols and any(c.ndim != 1 or c.shape != cols[0].shape for c in cols):
        raise ValueError("all candidate directions must be one-dimensional with equal shape")
    if not np.isfinite(drop_tol) or drop_tol <= 0.0:
        raise ValueError("drop_tol must be finite and positive")
    cols = [c for c in cols if np.all(np.isfinite(c)) and norm(c) > 0.0]
    if not cols:
        return None
    A = np.column_stack([c / norm(c) for c in cols])
    Q, R, _ = qr(A, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if diag[0] == 0.0:
        return None
    keep = int(np.sum(diag > drop_tol * diag[0]))
    return Q[:, :keep] if keep > 0 else None


class ReducedTrustRegionModel:
    """Fresh reduced least-squares model min ||f + W y|| over an orthonormal subspace Q.

    Args:
        Q: (n, k) orthonormal subspace basis in the scaled ("hat") variables.
        W: (m, k) fresh Jacobian-vector products J_h @ Q.
        f: (m,) residual at the current iterate.
    """

    def __init__(self, Q, W, f):
        """Factor the current reduced least-squares model."""
        Q = np.asarray(Q, dtype=float)
        W = np.asarray(W, dtype=float)
        f = np.asarray(f, dtype=float)
        if Q.ndim != 2 or W.ndim != 2 or f.ndim != 1:
            raise ValueError("Q and W must be matrices and f must be a vector")
        if Q.shape[1] == 0:
            raise ValueError("Q must contain at least one direction")
        if W.shape != (f.size, Q.shape[1]):
            raise ValueError(f"W must have shape {(f.size, Q.shape[1])}, got {W.shape}")
        if not all(np.all(np.isfinite(a)) for a in (Q, W, f)):
            raise ValueError("Q, W, and f must contain only finite values")
        gram = Q.T @ Q
        if not np.allclose(gram, np.eye(Q.shape[1]), rtol=1e-10, atol=1e-12):
            raise ValueError("Q columns must be orthonormal")
        self.Q = Q
        self.W = W
        self.k = Q.shape[1]
        self.m = W.shape[0]
        U, s, Vt = svd(W, full_matrices=False)
        self.uf = U.T.dot(f)
        self.s = s
        self.V = Vt.T
        self.Wf = W.T.dot(f)

    def solve(self, Delta, initial_alpha=None):
        """Exact trust-region solve of the reduced model; returns (y, alpha, n_iter)."""
        return solve_lsq_trust_region(
            self.k, self.m, self.uf, self.s, self.V, Delta, initial_alpha=initial_alpha
        )

    def predicted_reduction(self, y):
        """Predicted cost reduction of the step Q @ y under the fresh reduced model."""
        return -evaluate_quadratic(self.W, self.Wf, y)
