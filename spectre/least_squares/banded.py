# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Banded normal-equation trust-region subproblem for block-tridiagonal Jacobians.

The SPECTRE force Jacobian is block-tridiagonal in the interface index: the force
residual on interface i responds only to the geometry of interfaces i-1, i, i+1,
because only the Beltrami fields of the two volumes adjacent to a perturbed interface
change (measured in perf_study, force_jacobian_investigation-4.pdf, section 5; the
only out-of-band content is the ~1e-8-relative mu/pflux constraint coupling through
dmupfdx). The normal matrix N = J^T J is therefore block-pentadiagonal, and the
trust-region subproblem (N + alpha I) p = -g can be solved with a banded Cholesky
factorization per alpha instead of one dense SVD of J per outer iteration.

The alpha iteration mirrors ``common.solve_lsq_trust_region`` (More's safeguarded
Newton method), with phi'(alpha) obtained from one extra banded triangular solve:
phi' = -(p^T (N + alpha I)^{-1} p) / ||p||.

Robustness: the assembly measures the Jacobian's actual out-of-band mass, and the
solver validates each step against the in-band normal matrix; both callers
(``trf.trf_no_bounds``) fall back to the exact SVD path with a printed WARNING when
either check fails, so this solver can never silently change the optimization.
"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import LinAlgError, cho_solve_banded, cholesky_banded
from scipy.sparse import coo_matrix

EPS = np.finfo(float).eps


class BandedNormalEquations:
    """Banded N = J^T J of a block-tridiagonal J, and its trust-region solver.

    Args:
        J: Dense (m, n) Jacobian, rows grouped in ``n_blocks`` equal interface blocks,
            columns grouped in ``n_blocks`` equal interface blocks (m and n must both
            be divisible by n_blocks).
        n_blocks: Number of interface blocks (Mvol - 1 for the force Jacobian).

    Attributes:
        oob_ratio: Measured relative out-of-band mass of J,
            sqrt(sum(J_outside_band**2) / sum(J**2)). The banded assembly drops
            exactly this content, so it bounds the relative truncation of N.
    """

    def __init__(self, J, n_blocks):
        m, n = J.shape
        if n_blocks < 1 or m % n_blocks != 0 or n % n_blocks != 0:
            raise ValueError(
                f"Jacobian shape {J.shape} is not divisible into {n_blocks} equal row/column blocks"
            )
        self.n = n
        self.n_blocks = n_blocks
        rpb = m // n_blocks
        cpb = n // n_blocks
        self.u = min(3 * cpb - 1, n - 1)  # scalar half-bandwidth of N

        # Out-of-band mass of J: row block r may touch column blocks r-1, r, r+1 only.
        total_sq = np.sum(J**2)
        oob_sq = 0.0
        for r in range(n_blocks):
            rows = slice(r * rpb, (r + 1) * rpb)
            c_lo = max(r - 1, 0) * cpb
            c_hi = min(r + 2, n_blocks) * cpb
            oob_sq += np.sum(J[rows, :c_lo] ** 2) + np.sum(J[rows, c_hi:] ** 2)
        self.oob_ratio = np.sqrt(oob_sq / total_sq) if total_sq > 0 else 0.0

        # Assemble N = J^T J restricted to the band, in scipy upper banded storage
        # ab[u + i - j, j] = N[i, j] (i <= j), accumulating each row block's
        # in-band window W = Jr^T Jr (Jr = J[rows, window of <= 3 column blocks]).
        ab = np.zeros((self.u + 1, n))
        coo_i, coo_j, coo_v = [], [], []
        for r in range(n_blocks):
            rows = slice(r * rpb, (r + 1) * rpb)
            c_lo = max(r - 1, 0) * cpb
            c_hi = min(r + 2, n_blocks) * cpb
            Jr = J[rows, c_lo:c_hi]
            W = Jr.T @ Jr
            w = c_hi - c_lo
            ii, jj = np.triu_indices(w)
            # No duplicate (i, j) within one window, so fancy-indexed += is safe.
            ab[self.u + ii - jj, c_lo + jj] += W[ii, jj]
            coo_i.append(c_lo + ii)
            coo_j.append(c_lo + jj)
            coo_v.append(W[ii, jj])
            off = ii != jj
            coo_i.append(c_lo + jj[off])
            coo_j.append(c_lo + ii[off])
            coo_v.append(W[ii[off], jj[off]])
        self.ab = ab
        # Sparse copy of the full (symmetric) banded N for residual checks.
        self.N = coo_matrix(
            (np.concatenate(coo_v), (np.concatenate(coo_i), np.concatenate(coo_j))),
            shape=(n, n),
        ).tocsr()

    def _factor(self, alpha):
        ab = self.ab.copy()
        ab[self.u, :] += alpha
        return cholesky_banded(ab, lower=False)

    def _step_residual(self, p, g, alpha):
        """Relative residual of (N + alpha I) p = -g."""
        r = self.N @ p + alpha * p + g
        gn = norm(g)
        return norm(r) / gn if gn > 0 else 0.0

    def solve_trust_region(self, g, Delta, initial_alpha=None, rtol=0.01, max_iter=10):
        """Solve min 0.5 p^T N p + g^T p subject to ||p|| <= Delta.

        Mirrors ``common.solve_lsq_trust_region`` with banded Cholesky factorizations
        in place of the SVD algebra.

        Returns:
            (p, alpha, n_iter) on success, or None if the banded solve is not
            trustworthy at this iterate (caller should fall back to the SVD path).
        """

        def phi_and_derivative(alpha):
            try:
                c = self._factor(alpha)
            except LinAlgError:
                return None
            p = -cho_solve_banded((c, False), g)
            p_norm = norm(p)
            w = cho_solve_banded((c, False), p)
            phi = p_norm - Delta
            phi_prime = -np.dot(p, w) / p_norm
            return p, phi, phi_prime

        # Gauss-Newton attempt at alpha = 0.
        gn = phi_and_derivative(0.0)
        if gn is not None:
            p, phi, phi_prime = gn
            if norm(p) <= Delta:
                # Interior solution taken unprojected. A small residual alone does not
                # certify a normal-equations step (backward stability hides forward
                # error at cond(N) ~ 1/eps), so require one iterative-refinement step
                # to contract: its size ~ eps * cond(N) * ||p|| separates the usable
                # regime from the SVD-only one.
                try:
                    c = self._factor(0.0)
                    dp = -cho_solve_banded((c, False), self.N @ p + g)
                except LinAlgError:
                    return None
                if norm(dp) <= 1e-6 * norm(p):
                    return p + dp, 0.0, 0
                return None

        alpha_upper = norm(g) / Delta
        if gn is not None:
            alpha_lower = -phi / phi_prime
        else:
            alpha_lower = 0.0

        if initial_alpha is None or (gn is None and initial_alpha == 0):
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
        else:
            alpha = initial_alpha

        for it in range(max_iter):
            if alpha < alpha_lower or alpha > alpha_upper:
                alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)

            res = phi_and_derivative(alpha)
            if res is None:
                # N + alpha I is PSD, so this is numerical noise; bump alpha and retry.
                alpha = max(2.0 * alpha, 0.001 * alpha_upper)
                continue
            _, phi, phi_prime = res

            if phi < 0:
                alpha_upper = alpha

            ratio = phi / phi_prime
            alpha_lower = max(alpha_lower, alpha - ratio)
            alpha -= (phi + Delta) * ratio / Delta

            if np.abs(phi) < rtol * Delta:
                break

        # Recompute p at the post-update alpha, as the SVD solver does.
        res = phi_and_derivative(max(alpha, alpha_lower))
        if res is None:
            return None
        alpha = max(alpha, alpha_lower)
        p = res[0]
        if self._step_residual(p, g, alpha) > 1e-6:
            return None

        p *= Delta / norm(p)
        return p, alpha, it + 1
