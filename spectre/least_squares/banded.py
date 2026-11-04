# SPDX-FileCopyrightText: 2001-2002 Enthought, Inc.
# SPDX-FileCopyrightText: 2003 SciPy Developers
# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: BSD-3-Clause

"""Banded normal-equation trust-region subproblem for block-tridiagonal Jacobians.

The SPECTRE force Jacobian is block-tridiagonal in the interface index: the force
residual on interface i responds only to the geometry of interfaces i-1, i, i+1,
because only the Beltrami fields of the two volumes adjacent to a perturbed interface
change. The normal matrix N = J^T J is therefore block-pentadiagonal, and the
trust-region subproblem (N + alpha I) p = -g can use a banded Cholesky
factorization per alpha instead of a dense SVD of J.

The alpha iteration mirrors ``common.solve_lsq_trust_region`` (More's safeguarded
Newton method), with phi'(alpha) obtained from one extra banded triangular solve:
phi' = -(p^T (N + alpha I)^{-1} p) / ||p||.

Robustness: the assembly measures the Jacobian's actual out-of-band mass, and the
solver validates each step against the in-band normal matrix. The optimizer consumes
the structured diagnostics and chooses the exact-SVD fallback with a printed warning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.linalg import norm
from scipy.linalg import LinAlgError, cho_solve_banded, cholesky_banded
from scipy.sparse import coo_matrix, csr_matrix

EPS = np.finfo(float).eps
DEFAULT_TRUST_REGION_RTOL = 0.01
DEFAULT_MAX_ALPHA_ITERATIONS = 10
LINEAR_RESIDUAL_TOL = 1e-6
REFINEMENT_CONTRACTION_TOL = 1e-6
MAX_OUT_OF_BAND_RATIO = 1e-6
ALPHA_SAFEGUARD_FRACTION = 0.001


class BandedStep(NamedTuple):
    """Successful solution of a banded trust-region subproblem."""

    step: np.ndarray
    alpha: float
    iterations: int


@dataclass(frozen=True)
class BlockLayout:
    """Validated equal-block layout of a dense Jacobian."""

    rows: int
    columns: int
    blocks: int
    rows_per_block: int
    columns_per_block: int


@dataclass(frozen=True)
class BandedMatrixAssembly:
    """Numerical data for an in-band normal matrix.

    ``ab`` is the compact upper-band storage owned by Cholesky factorization calls.
    ``normal_matrix`` retains the same band entries in CSR form solely for residual
    and iterative-refinement products; it avoids an ``n * n`` dense allocation. The
    assembly is independent of optimizer fallback policy.
    """

    ab: np.ndarray
    normal_matrix: csr_matrix
    n: int
    n_blocks: int
    half_bandwidth: int
    out_of_band_ratio: float


def validate_block_layout(J, n_blocks):
    """Validate and describe the equal block layout expected by the assembler."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2:
        raise ValueError(f"Jacobian must be two-dimensional, got shape {J.shape}")
    if not np.all(np.isfinite(J)):
        raise ValueError("Jacobian entries must all be finite")
    if not isinstance(n_blocks, (int, np.integer)):
        raise TypeError("n_blocks must be an integer")
    m, n = J.shape
    if n_blocks < 1 or m % n_blocks != 0 or n % n_blocks != 0:
        raise ValueError(
            f"Jacobian shape {J.shape} is not divisible into {n_blocks} equal row/column blocks"
        )
    return J, BlockLayout(m, n, int(n_blocks), m // n_blocks, n // n_blocks)


class BandedSolveStatus(Enum):
    """Outcome of a numerical banded trust-region solve."""

    SUCCESS = "success"
    FACTORIZATION_FAILED = "factorization failed"
    REFINEMENT_FAILED = "iterative refinement did not contract"
    INVALID_DERIVATIVE = "invalid trust-region secular derivative"
    RESIDUAL_TOO_LARGE = "linear-system residual exceeds tolerance"
    INVALID_STEP = "step is zero or non-finite"


@dataclass(frozen=True)
class BandedSolveDiagnostics:
    """Diagnostics produced by the numerical kernel, without fallback policy."""

    status: BandedSolveStatus
    iterations: int
    relative_residual: float | None = None
    refinement_ratio: float | None = None


@dataclass(frozen=True)
class BandedSolveResult:
    """Structured numerical result consumed by optimizer fallback policy."""

    step: BandedStep | None
    diagnostics: BandedSolveDiagnostics

    @property
    def available(self):
        """Whether the banded kernel produced a trustworthy step."""
        return self.step is not None


def assemble_banded_normal_equations(J, n_blocks):
    """Assemble the in-band part of ``J.T @ J``.

    Args:
        J: Dense two-dimensional Jacobian.
        n_blocks: Number of equal row and column blocks.

    Returns:
        Immutable assembly data suitable for :class:`BandedTrustRegionSolver`.
    """
    J, layout = validate_block_layout(J, n_blocks)
    n = layout.columns
    rpb = layout.rows_per_block
    cpb = layout.columns_per_block
    half_bandwidth = min(3 * cpb - 1, n - 1)

    total_sq = np.sum(J**2)
    oob_sq = 0.0
    for r in range(layout.blocks):
        rows = slice(r * rpb, (r + 1) * rpb)
        c_lo = max(r - 1, 0) * cpb
        c_hi = min(r + 2, layout.blocks) * cpb
        oob_sq += np.sum(J[rows, :c_lo] ** 2) + np.sum(J[rows, c_hi:] ** 2)
    out_of_band_ratio = np.sqrt(oob_sq / total_sq) if total_sq > 0 else 0.0

    ab = np.zeros((half_bandwidth + 1, n))
    coo_i, coo_j, coo_v = [], [], []
    for r in range(layout.blocks):
        rows = slice(r * rpb, (r + 1) * rpb)
        c_lo = max(r - 1, 0) * cpb
        c_hi = min(r + 2, layout.blocks) * cpb
        Jr = J[rows, c_lo:c_hi]
        W = Jr.T @ Jr
        w = c_hi - c_lo
        ii, jj = np.triu_indices(w)
        ab[half_bandwidth + ii - jj, c_lo + jj] += W[ii, jj]
        coo_i.append(c_lo + ii)
        coo_j.append(c_lo + jj)
        coo_v.append(W[ii, jj])
        off = ii != jj
        coo_i.append(c_lo + jj[off])
        coo_j.append(c_lo + ii[off])
        coo_v.append(W[ii[off], jj[off]])
    normal_matrix = coo_matrix(
        (np.concatenate(coo_v), (np.concatenate(coo_i), np.concatenate(coo_j))),
        shape=(n, n),
    ).tocsr()
    return BandedMatrixAssembly(
        ab,
        normal_matrix,
        n,
        layout.blocks,
        half_bandwidth,
        float(out_of_band_ratio),
    )


class BandedTrustRegionSolver:
    """Numerical banded Cholesky kernel with structured failure diagnostics."""

    def __init__(self, assembly):
        """Bind an assembled normal matrix."""
        self.assembly = assembly

    def _factor(self, alpha):
        ab = self.assembly.ab.copy()
        ab[self.assembly.half_bandwidth, :] += alpha
        return cholesky_banded(ab, lower=False)

    def _step_residual(self, p, g, alpha):
        residual = self.assembly.normal_matrix @ p + alpha * p + g
        gradient_norm = norm(g)
        return norm(residual) / gradient_norm if gradient_norm > 0 else 0.0

    @staticmethod
    def _failure(status, iterations=0, relative_residual=None, refinement_ratio=None):
        return BandedSolveResult(
            None,
            BandedSolveDiagnostics(
                status,
                iterations,
                relative_residual=relative_residual,
                refinement_ratio=refinement_ratio,
            ),
        )

    @staticmethod
    def _success(step, alpha, iterations, relative_residual=0.0, refinement_ratio=None):
        return BandedSolveResult(
            BandedStep(step, alpha, iterations),
            BandedSolveDiagnostics(
                BandedSolveStatus.SUCCESS,
                iterations,
                relative_residual=relative_residual,
                refinement_ratio=refinement_ratio,
            ),
        )

    def solve_newton(self, g):
        """Validated Gauss-Newton step of the banded normal equations at ``alpha = 0``.

        The interior solve shared by :meth:`solve` and the dogbox Gauss-Newton path:
        one banded Cholesky factorization, one iterative-refinement step that must
        contract, and the same structured diagnostics as :meth:`solve`.

        Args:
            g: Gradient ``J.T @ f`` of length ``n``.

        Returns:
            :class:`BandedSolveResult` whose step solves ``N p = -g``; on any
            numerical failure ``step`` is None and the diagnostics carry the cause.
        """
        g = np.asarray(g, dtype=float)
        n = self.assembly.n
        if g.shape != (n,):
            raise ValueError(f"gradient must have shape ({n},), got {g.shape}")
        if not np.all(np.isfinite(g)):
            raise ValueError("gradient entries must all be finite")
        if norm(g) == 0.0:
            return self._success(np.zeros_like(g), 0.0, 0)

        try:
            c = self._factor(0.0)
        except LinAlgError:
            return self._failure(BandedSolveStatus.FACTORIZATION_FAILED)
        p = -cho_solve_banded((c, False), g)
        p_norm = norm(p)
        if p_norm == 0.0 or not np.isfinite(p_norm):
            return self._failure(BandedSolveStatus.INVALID_STEP)
        dp = -cho_solve_banded((c, False), self.assembly.normal_matrix @ p + g)
        refinement_ratio = norm(dp) / max(p_norm, EPS)
        if refinement_ratio > REFINEMENT_CONTRACTION_TOL:
            return self._failure(
                BandedSolveStatus.REFINEMENT_FAILED,
                refinement_ratio=refinement_ratio,
            )
        refined = p + dp
        residual = self._step_residual(refined, g, 0.0)
        return self._success(
            refined,
            0.0,
            0,
            relative_residual=residual,
            refinement_ratio=refinement_ratio,
        )

    def solve(
        self,
        g,
        Delta,
        initial_alpha=None,
        rtol=DEFAULT_TRUST_REGION_RTOL,
        max_iter=DEFAULT_MAX_ALPHA_ITERATIONS,
    ):
        """Solve the numerical trust-region system without choosing a fallback."""
        g = np.asarray(g, dtype=float)
        n = self.assembly.n
        if g.shape != (n,):
            raise ValueError(f"gradient must have shape ({n},), got {g.shape}")
        if not np.all(np.isfinite(g)):
            raise ValueError("gradient entries must all be finite")
        if not np.isfinite(Delta) or Delta <= 0.0:
            raise ValueError("Delta must be finite and positive")
        if not np.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("rtol must be finite and positive")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if initial_alpha is not None and (not np.isfinite(initial_alpha) or initial_alpha < 0.0):
            raise ValueError("initial_alpha must be finite and non-negative")

        if norm(g) == 0.0:
            return self._success(np.zeros_like(g), 0.0, 0)

        last_failure = BandedSolveStatus.INVALID_DERIVATIVE

        def phi_and_derivative(alpha):
            nonlocal last_failure
            try:
                c = self._factor(alpha)
            except LinAlgError:
                last_failure = BandedSolveStatus.FACTORIZATION_FAILED
                return None
            p = -cho_solve_banded((c, False), g)
            p_norm = norm(p)
            if p_norm == 0.0 or not np.isfinite(p_norm):
                last_failure = BandedSolveStatus.INVALID_STEP
                return None
            w = cho_solve_banded((c, False), p)
            phi = p_norm - Delta
            phi_prime = -np.dot(p, w) / p_norm
            if not np.isfinite(phi_prime) or phi_prime >= 0.0:
                last_failure = BandedSolveStatus.INVALID_DERIVATIVE
                return None
            return p, phi, phi_prime

        gn = phi_and_derivative(0.0)
        if gn is not None:
            p, phi, phi_prime = gn
            if norm(p) <= Delta:
                return self.solve_newton(g)

        alpha_upper = norm(g) / Delta
        alpha_lower = -phi / phi_prime if gn is not None else 0.0

        if initial_alpha is None or (gn is None and initial_alpha == 0):
            alpha = max(
                ALPHA_SAFEGUARD_FRACTION * alpha_upper,
                (alpha_lower * alpha_upper) ** 0.5,
            )
        else:
            alpha = initial_alpha

        iterations = 0
        for _ in range(max_iter):
            iterations += 1
            if alpha < alpha_lower or alpha > alpha_upper:
                alpha = max(
                    ALPHA_SAFEGUARD_FRACTION * alpha_upper,
                    (alpha_lower * alpha_upper) ** 0.5,
                )

            result = phi_and_derivative(alpha)
            if result is None:
                alpha = max(2.0 * alpha, ALPHA_SAFEGUARD_FRACTION * alpha_upper)
                continue
            _, phi, phi_prime = result

            if phi < 0:
                alpha_upper = alpha

            ratio = phi / phi_prime
            alpha_lower = max(alpha_lower, alpha - ratio)
            alpha -= (phi + Delta) * ratio / Delta

            if np.abs(phi) < rtol * Delta:
                break

        alpha = max(alpha, alpha_lower)
        result = phi_and_derivative(alpha)
        if result is None:
            return self._failure(last_failure, iterations)
        p = result[0]
        relative_residual = self._step_residual(p, g, alpha)
        if relative_residual > LINEAR_RESIDUAL_TOL:
            return self._failure(
                BandedSolveStatus.RESIDUAL_TOO_LARGE,
                iterations,
                relative_residual=relative_residual,
            )

        p_norm = norm(p)
        if p_norm == 0.0 or not np.isfinite(p_norm):
            return self._failure(BandedSolveStatus.INVALID_STEP, iterations)
        p *= Delta / p_norm
        return self._success(
            p,
            alpha,
            iterations,
            relative_residual=relative_residual,
        )
