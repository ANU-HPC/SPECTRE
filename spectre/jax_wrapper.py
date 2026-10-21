# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Optional JAX bindings for the SPECTRE force balance.

Use this module to couple SPECTRE to codes that rely on JAX automatic differentiation, e.g. to
take gradients of an objective that depends on a SPECTRE equilibrium. It is not imported by
``import spectre``; import it explicitly with ``from spectre.jax_wrapper import SpectreJax``.

The force residual and its Jacobian are evaluated by the Fortran wrapper through
``jax.pure_callback``. Derivatives are supplied with ``jax.custom_jvp`` whose tangent map is a
plain matrix product with the Jacobian, so JAX transposes it automatically and both forward
(``jax.jvp``, ``jax.jacfwd``) and reverse mode (``jax.grad``, ``jax.vjp``) work, also under
``jax.jit``. No finite differences are used.

Requires ``jax`` (``pip install 'spectre[jax]'``) with 64-bit floats enabled
(``jax.config.update("jax_enable_x64", True)``).

See the user guide page "Automatic differentiation with JAX" for a worked example.
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as err:
    raise ImportError("spectre.jax_wrapper requires jax: pip install 'spectre[jax]'") from err

from spectre.force_targets import _process_input_nt_nz, force_real, force_real_jac
from spectre.least_squares.least_squares import least_squares
from spectre.utils import get_xinit_specwrap


class SpectreJax:
    """JAX-differentiable view of a SPECTRE force balance.

    Two entry points, depending on what is differentiated:

    * :meth:`force` -- the force residual as a function of the interface geometry ``x``. Use it
      when the geometry itself is the optimization variable, e.g. to minimize the force with a
      JAX optimizer or to include the force balance as a penalty in a larger objective.
    * :meth:`equilibrium` -- the equilibrium geometry ``x*(p)`` as a function of parameters
      ``p``. Use it when ``x*`` is an intermediate result, e.g. to differentiate a quantity
      evaluated on the equilibrium with respect to the pressure profile.

    The state ``x`` is the packed interface geometry of length ``ngdof + 1`` (see
    ``get_xinit_specwrap``); ``x[0]`` is unused. The force is the magnetic pressure jump
    across each interface on an ``nt x nz`` grid, flattened to length ``(mvol - 1) * nt * nz``,
    without the kinetic pressure contribution (supply it through ``param_force``, see
    :meth:`equilibrium` and :meth:`pressure_force`).

    The wrapped SPECTRE object holds global Fortran state: use one ``SpectreJax`` per SPECTRE
    object, and do not evaluate it from several threads at once.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> from spectre import SPECTRE
        >>> from spectre.jax_wrapper import SpectreJax
        >>> sj = SpectreJax(SPECTRE("cyl_manyvol_test.toml", verbose=False))
        >>> x_star = sj.equilibrium(sj.pressure_force)
        >>> grad = jax.grad(lambda p: x_star(p)[1])(jax.numpy.array([1.0, 0.9, 0.8]))
    """

    def __init__(self, test, nt=None, nz=None, rcond=1e-10, solver_kwargs=None):
        """Wrap an initialized SPECTRE object.

        Args:
            test: Initialized SPECTRE object.
            nt: Number of poloidal grid points of the residual (see ``force_real``).
            nz: Number of toroidal grid points of the residual (see ``force_real``).
            rcond: Relative singular value cutoff of the pseudo-inverse used in the implicit
                derivative. Needed because the Jacobian has a one-dimensional gauge null space.
            solver_kwargs: Keyword arguments passed to ``least_squares`` for the forward solve.
        """
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "SpectreJax requires 64-bit floats: jax.config.update('jax_enable_x64', True)"
            )

        self.test = test
        self.nt, self.nz = _process_input_nt_nz(nt, nz, test)
        self.rcond = rcond
        self.solver_kwargs = dict(method="dogbox", gtol=1e-14, xtol=1e-14, ftol=1e-14)
        self.solver_kwargs.update(solver_kwargs or {})

        self.n_x = test.allglobal_mod.ngdof + 1
        self.n_f = (test.allglobal_mod.mvol - 1) * self.nt * self.nz
        self._x_shape = jax.ShapeDtypeStruct((self.n_x,), jnp.float64)
        self._f_shape = jax.ShapeDtypeStruct((self.n_f,), jnp.float64)
        self._jac_shape = jax.ShapeDtypeStruct((self.n_f, self.n_x), jnp.float64)

        # Warm start for the forward solve, updated after each solve.
        self._x_warm = get_xinit_specwrap(test)

        self._force_cjvp = jax.custom_jvp(self._force)
        self._force_cjvp.defjvp(self._force_jvp)

    def force(self, x):
        """Force residual at the interface geometry ``x``, differentiable with respect to ``x``.

        Each evaluation solves the Beltrami field in all volumes. Derivatives use the analytic
        force Jacobian (``force_real_jac``), computed once per linearization point.

        Args:
            x: Packed interface geometry of shape ``(n_x,)``, e.g. from ``get_xinit_specwrap``.

        Returns:
            Magnetic pressure jump of shape ``(n_f,)``, without the kinetic pressure term.

        Example:
            >>> x0 = jax.numpy.asarray(get_xinit_specwrap(sj.test))
            >>> grad = jax.grad(lambda x: 0.5 * jax.numpy.sum(sj.force(x) ** 2))(x0)
        """
        return self._force_cjvp(x)

    def _force_np(self, x):
        x = np.array(x, dtype=np.float64, order="F")
        return force_real(x, self.test, add_pressure=False, nt=self.nt, nz=self.nz).ravel()

    def _force_and_jac_np(self, x):
        x = np.array(x, dtype=np.float64, order="F")
        # force_real_jac relies on the field solved by force_real at the same x.
        f = force_real(x, self.test, add_pressure=False, nt=self.nt, nz=self.nz).ravel()
        jac = force_real_jac(x, self.test, nt=self.nt, nz=self.nz).reshape(self.n_f, self.n_x)
        return f, jac

    def _solve_np(self, g):
        g = np.asarray(g, dtype=np.float64)
        result = least_squares(
            lambda x: self._force_np(x) + g,
            self._x_warm,
            jac=lambda x: self._force_and_jac_np(x)[1],
            **self.solver_kwargs,
        )
        self._x_warm = result.x.copy()
        return result.x

    def _force(self, x):
        return jax.pure_callback(self._force_np, self._f_shape, x, vmap_method="sequential")

    def _force_and_jac(self, x):
        return jax.pure_callback(
            self._force_and_jac_np, (self._f_shape, self._jac_shape), x, vmap_method="sequential"
        )

    def _force_jvp(self, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        f, jac = self._force_and_jac(x)
        return f, jac @ x_dot

    def pressure_force(self, pressure):
        """Kinetic pressure jump across each interface, broadcast to the residual grid.

        Pass it (or a function built on it) as ``param_force`` to :meth:`equilibrium` to
        differentiate the equilibrium with respect to the volume pressures.

        Args:
            pressure: Pressure in each volume, shape ``(mvol,)``, before scaling by ``pscale``.

        Returns:
            Additive force term of shape ``(n_f,)``, matching ``force_real(add_pressure=True)``.
        """
        jump = jnp.diff(pressure * self.test.input_list_mod.pscale)
        return jnp.repeat(jump, self.nt * self.nz)

    def equilibrium(self, param_force):
        """Build a differentiable map from parameters to the equilibrium state.

        Use it to differentiate anything computed from the equilibrium geometry with respect to
        parameters that enter the force balance additively, such as the volume pressures.
        Parameters inside SPECTRE's field solve (e.g. the plasma boundary) are not supported,
        since the force Jacobian contains no derivatives with respect to them.

        Each call of the returned function runs a full force minimization (``least_squares``,
        warm-started from the previous solution); each derivative evaluation additionally
        computes one force Jacobian and its pseudo-inverse.

        The equilibrium ``x*(p)`` solves ``force(x) + param_force(p) = 0`` in the least-squares
        sense. Its derivative follows from the implicit function theorem,
        ``dx* = -pinv(J) d(param_force)``, with ``J`` the SPECTRE force Jacobian at ``x*``. This
        is exact when the residual vanishes at ``x*`` and a Gauss-Newton approximation
        otherwise, since the term ``sum_i r_i * hess(F_i)`` is dropped. The error does not
        shrink in proportion to the residual: on ``qa_7vol_test`` the gradient is off by 13%,
        19% and 4.5% at ``(mpol, ntor) = (3, 1), (5, 2), (7, 3)`` while the RMS residual
        falls from 7.8e-2 to 1.9e-3. The component along the gauge null space of ``J`` is set
        to zero.

        Args:
            param_force: JAX-traceable function mapping parameters ``p`` to an additive force
                term of shape ``(n_f,)``, e.g. ``lambda p: sj.pressure_force(p)``.

        Returns:
            Function ``p -> x*`` of shape ``(n_x,)`` supporting ``jax.jvp`` and ``jax.grad``.

        Example:
            >>> x_star = sj.equilibrium(sj.pressure_force)
            >>> p0 = jax.numpy.linspace(1.0, 0.8, sj.test.allglobal_mod.mvol)
            >>> dx_dp = jax.jacfwd(x_star)(p0)  # shape (n_x, mvol)
        """

        @jax.custom_jvp
        def solve(p):
            return jax.pure_callback(
                self._solve_np, self._x_shape, param_force(p), vmap_method="sequential"
            )

        @solve.defjvp
        def solve_jvp(primals, tangents):
            (p,), (p_dot,) = primals, tangents
            x = solve(p)
            g_dot = jax.jvp(param_force, (p,), (p_dot,))[1]
            _, jac = self._force_and_jac(x)
            return x, -jnp.linalg.pinv(jac, rtol=self.rcond) @ g_dot

        return solve
