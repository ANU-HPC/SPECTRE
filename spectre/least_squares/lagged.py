# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Lagged-banded matrix-free trust-region iteration (helpers).

Between dense Jacobian rebuilds, ``trf.trf_no_bounds`` keeps the banded normal-equation
factorization of a stale (lagged) J as a subspace source: the lagged banded trust-region
direction, the lagged gradient, and the previous accepted step span a small subspace Q.
A batch of fresh Jacobian-vector products W = J Q (one linearised solve per volume
per direction) then gives an exact current-geometry reduced model

    min 0.5 ||f + W y||^2   subject to   ||Q y|| = ||y|| <= Delta,    step = Q y,

solved with the same More alpha-iteration as the exact solver, on the small (m, k) SVD.
The predicted reduction and therefore the trust-region gain ratio come from the fresh W,
so the radius control stays honest even when the lagged subspace is poor. Convergence is
never declared from lagged quantities: any termination test that fires while the
Jacobian is stale forces a dense rebuild and a fresh recheck (see ``trf.trf_no_bounds``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.linalg import norm
from scipy.linalg import qr, svd

from spectre.least_squares.banded import (
    MAX_OUT_OF_BAND_RATIO,
    BandedNormalEquations,
)
from spectre.least_squares.common import evaluate_quadratic, solve_lsq_trust_region

DEFAULT_SUBSPACE_DROP_TOL = 1e-12
ORTHONORMAL_RTOL = 1e-10
ORTHONORMAL_ATOL = 1e-12
MIN_LAGGED_GAIN_RATIO = 0.25


@dataclass(frozen=True)
class BandedTrOptions:
    """Typed internal view of the public ``tr_options`` mapping."""

    n_blocks: int
    lag: int = 0
    jvp: Callable | None = None

    @classmethod
    def from_mapping(cls, options: Mapping):
        """Translate the public mapping once at the banded-solver boundary."""
        n_blocks = options["n_blocks"]
        lag = int(options.get("lag", 0) or 0)
        jvp = options.get("jvp")
        return cls(n_blocks, lag, jvp)

    def without_lag(self):
        """Return the same banded layout with lagged Jacobian reuse disabled."""
        return BandedTrOptions(self.n_blocks)


class JvpStatus(Enum):
    """Usability status of one matrix-free Jacobian batch."""

    SUCCESS = "success"
    INVALID_SHAPE = "invalid shape"
    NONFINITE = "non-finite values"
    BACKEND_FAILURE = "backend failure"
    UNSUPPORTED = "unsupported configuration"


@dataclass(frozen=True)
class JvpResult:
    """Typed JVP values and the validation outcome used by controller policy."""

    values: np.ndarray | None
    status: JvpStatus
    expected_shape: tuple[int, int]
    actual_shape: tuple[int, ...] | None
    backend_status: tuple[int, ...] = ()

    @property
    def usable(self):
        """Whether the values can define a reduced model."""
        return self.status is JvpStatus.SUCCESS

    @classmethod
    def from_values(cls, values, expected_shape):
        """Validate an existing public JVP callable's array result."""
        if isinstance(values, cls):
            if not values.usable:
                return replace(values, expected_shape=expected_shape)
            values = values.values
        values = np.asarray(values)
        if values.shape != expected_shape:
            return cls(None, JvpStatus.INVALID_SHAPE, expected_shape, values.shape)
        if not np.all(np.isfinite(values)):
            return cls(None, JvpStatus.NONFINITE, expected_shape, values.shape)
        return cls(values, JvpStatus.SUCCESS, expected_shape, values.shape)

    @classmethod
    def from_backend_failure(cls, shape, backend_status):
        """Preserve established per-volume failure diagnostics without array sentinels."""
        status = tuple(int(value) for value in backend_status)
        outcome = (
            JvpStatus.UNSUPPORTED
            if any(value < 0 for value in status)
            else JvpStatus.BACKEND_FAILURE
        )
        return cls(None, outcome, shape, shape, backend_status=status)


class LaggedAction(Enum):
    """Action requested by the lagged-Jacobian state machine."""

    USE_FRESH_MODEL = "use fresh model"
    USE_LAGGED_MODEL = "use lagged model"
    REBUILD_JACOBIAN = "rebuild Jacobian"


@dataclass(frozen=True)
class LaggedModelDecision:
    """Controller action and, when available, its fresh-JVP reduced model."""

    action: LaggedAction
    model: ReducedTrustRegionModel | None = None
    jvp_result: JvpResult | None = None
    reason: str = ""


@dataclass(frozen=True)
class LaggedControllerDiagnostics:
    """Read-only counters and state exposed to the optimizer result/reporting."""

    jacobian_age: int
    jvp_evaluations: int
    rebuild_requested: bool
    trial_state_dirty: bool
    banded_model_available: bool


@dataclass(frozen=True)
class BandedInitializationResult:
    """Availability and diagnostics from binding one fresh banded model."""

    available: bool
    out_of_band_ratio: float | None = None
    reason: str = ""


class LaggedJacobianController:
    """Own lagged-Jacobian state and its explicit optimizer transitions.

    The controller knows when a fresh Jacobian is mandatory and constructs only the
    reduced model used on stale iterations. Function/Jacobian evaluation, iterate
    updates, and exact-SVD fallback remain decisions of ``trf_no_bounds``.
    """

    def __init__(self, options):
        """Initialize the controller at a fresh Jacobian."""
        self.options = options
        self._jacobian_age = 0
        self._jvp_evaluations = 0
        self._previous_step = None
        self._rebuild_requested = False
        self._trial_state_dirty = False
        self._banded_system = None
        self._reduced_model = None

    @property
    def enabled(self):
        """Whether stale-Jacobian reuse is configured."""
        return self.options.lag > 0 and self.options.jvp is not None

    @property
    def stale(self):
        """Whether the current Jacobian predates the current residual."""
        return self._jacobian_age > 0

    @property
    def diagnostics(self):
        """Current state and counters for reporting."""
        return LaggedControllerDiagnostics(
            self._jacobian_age,
            self._jvp_evaluations,
            self._rebuild_requested,
            self._trial_state_dirty,
            self._banded_system is not None,
        )

    def initialize_fresh_jacobian(self, J_h):
        """Bind banded state to a fresh scaled Jacobian."""
        self._reduced_model = None
        try:
            banded_system = BandedNormalEquations(J_h, self.options.n_blocks)
        except ValueError as exc:
            self._banded_system = None
            return BandedInitializationResult(False, reason=repr(exc))
        out_of_band_ratio = banded_system.oob_ratio
        if out_of_band_ratio > MAX_OUT_OF_BAND_RATIO:
            self._banded_system = None
            return BandedInitializationResult(
                False,
                out_of_band_ratio=out_of_band_ratio,
                reason="block-tridiagonal assumption does not hold",
            )
        self._banded_system = banded_system
        return BandedInitializationResult(True, out_of_band_ratio=out_of_band_ratio)

    def request_iteration(self, g_norm, gtol):
        """Request a fresh model, lagged model, or rebuild for this iterate."""
        if not self.stale:
            return LaggedModelDecision(LaggedAction.USE_FRESH_MODEL)
        if self._rebuild_requested:
            return LaggedModelDecision(
                LaggedAction.REBUILD_JACOBIAN, reason="model-quality gate requested rebuild"
            )
        if g_norm < gtol:
            return LaggedModelDecision(
                LaggedAction.REBUILD_JACOBIAN,
                reason="fresh Jacobian required to verify gradient convergence",
            )
        return LaggedModelDecision(LaggedAction.USE_LAGGED_MODEL)

    def request_model(self, x, g_h, Delta, alpha, scale, residual):
        """Build a fresh-JVP reduced model over directions from the lagged system."""
        if self._banded_system is None:
            self._rebuild_requested = True
            return LaggedModelDecision(
                LaggedAction.REBUILD_JACOBIAN, reason="banded model is unavailable"
            )
        cols = [g_h, self._previous_step]
        lagged_solve = self._banded_system.try_solve_trust_region(g_h, Delta, initial_alpha=alpha)
        if lagged_solve.available:
            cols.insert(0, lagged_solve.step.step)
        Q = orthonormal_columns(cols)
        if Q is None:
            self._rebuild_requested = True
            return LaggedModelDecision(
                LaggedAction.REBUILD_JACOBIAN, reason="lagged subspace is empty"
            )

        dQ = Q * scale if np.ndim(scale) == 0 else Q * np.asarray(scale)[:, None]
        values = self.options.jvp(x, dQ)
        self._jvp_evaluations += 1
        jvp_result = JvpResult.from_values(values, (residual.size, Q.shape[1]))
        self.record_jvp_outcome(jvp_result)
        if not jvp_result.usable:
            return LaggedModelDecision(
                LaggedAction.REBUILD_JACOBIAN,
                jvp_result=jvp_result,
                reason=f"JVP batch has {jvp_result.status.value}",
            )

        self._reduced_model = ReducedTrustRegionModel(Q, jvp_result.values, residual)
        return LaggedModelDecision(
            LaggedAction.USE_LAGGED_MODEL,
            model=self._reduced_model,
            jvp_result=jvp_result,
        )

    def request_banded_step(self, g_h, Delta, alpha):
        """Request a structured numerical step from the retained banded model."""
        if self._banded_system is None:
            raise RuntimeError("banded model is unavailable")
        return self._banded_system.try_solve_trust_region(g_h, Delta, initial_alpha=alpha)

    def record_jvp_outcome(self, result):
        """Record whether a JVP batch permits continued stale-Jacobian reuse."""
        if not result.usable:
            self._rebuild_requested = True

    def record_trial_started(self):
        """Record that the wrapped objective's mutable state is at a trial point."""
        self._trial_state_dirty = True

    def consume_restore_requirement(self):
        """Return and clear the requirement to restore objective state before rebuild."""
        restore = self._trial_state_dirty
        self._trial_state_dirty = False
        return restore

    def request_fresh_convergence_verification(self, termination_status):
        """Suppress stale-model termination and request a fresh recheck."""
        if termination_status is not None and self.stale:
            self._rebuild_requested = True
            return None
        return termination_status

    def record_step_rejected(self):
        """Treat rejection under a stale model as a dense-rebuild request."""
        if self.stale:
            self._rebuild_requested = True

    def record_step_accepted(self, step_h, ratio):
        """Record acceptance and decide whether the next iterate may reuse J."""
        self._trial_state_dirty = False
        self._previous_step = np.asarray(step_h).copy()
        can_reuse = (
            self.enabled
            and self._banded_system is not None
            and not self._rebuild_requested
            and self._jacobian_age < self.options.lag
            and ratio >= MIN_LAGGED_GAIN_RATIO
        )
        if can_reuse:
            self._jacobian_age += 1
            return LaggedAction.USE_LAGGED_MODEL
        return LaggedAction.REBUILD_JACOBIAN

    def record_rebuild_complete(self):
        """Transition to a fresh Jacobian after its evaluation has completed."""
        self._jacobian_age = 0
        self._rebuild_requested = False
        self._trial_state_dirty = False
        self._banded_system = None
        self._reduced_model = None

    def disable(self):
        """Disable lag reuse after banded model construction becomes unavailable."""
        self.options = self.options.without_lag()
        self._jacobian_age = 0
        self._rebuild_requested = False
        self._banded_system = None
        self._reduced_model = None


def orthonormal_columns(cols, drop_tol=DEFAULT_SUBSPACE_DROP_TOL):
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
        if not np.allclose(
            gram,
            np.eye(Q.shape[1]),
            rtol=ORTHONORMAL_RTOL,
            atol=ORTHONORMAL_ATOL,
        ):
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
