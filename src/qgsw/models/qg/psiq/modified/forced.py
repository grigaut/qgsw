"""Forced QG models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.fields.variables.tuples import (
    PSIQSST,
)
from qgsw.logging.core import getLogger
from qgsw.models.names import ModelName
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.stretching_matrix import (
    compute_A_tilde,
)
from qgsw.solver.pv_inversion import (
    HomogeneousPVInversion,
    HomogeneousPVInversionCollinear,
    InhomogeneousPVInversion,
    InhomogeneousPVInversionCollinear,
)
from qgsw.specs import DEVICE, defaults

if TYPE_CHECKING:
    from qgsw.decomposition.base import SpaceTimeDecomposition
    from qgsw.decomposition.core import Basis
    from qgsw.decomposition.supports.space.base import SpaceSupportFunction
    from qgsw.decomposition.supports.time.base import TimeSupportFunction
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D

logger = getLogger(__name__)


class QGPSIQForced(QGPSIQ):
    """Specify and additional forcing term."""

    _type = ModelName.RGF

    @property
    def basis_external(
        self,
    ) -> SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]:
        """Forcing basis."""
        return self._ext_basis

    @basis_external.setter
    def basis_external(
        self,
        basis: SpaceTimeDecomposition[
            SpaceSupportFunction, TimeSupportFunction
        ],
    ) -> None:
        self._ext_basis = basis
        self._fext = self._ext_basis.localize(*self.space.remove_h().q.xy)

    @property
    def forcing(self) -> torch.Tensor:
        """Forcing."""
        return self._fext(self.time)[None, None, ...]

    def compute_time_derivatives_homogeneous(  # noqa: D102
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        psi, q, sst_anom = prognostic

        u, v = self.gradp(psi)
        u /= self.space.dy
        v /= self.space.dx

        div_flux = self.compute_div_flux_homogeneous(u, v, q)
        # wind forcing + bottom drag
        fcg_drag = self.compute_drag_homogeneous(psi)
        forcing = self._fext(self.time)
        dq = (-div_flux + fcg_drag + forcing) * self.masks.h
        dq_i = self.interp(dq)
        # Solve Helmholtz equation
        dpsi = self.solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )
        dsst_anom = self.compute_dsst_homogeneous(u, v, sst_anom)
        return PSIQSST(dpsi, dq, dsst_anom)

    compute_time_derivatives_homogeneous.__doc__ = (
        QGPSIQ.compute_time_derivatives_homogeneous.__doc__
    )

    def compute_time_derivatives_inhomogeneous(  # noqa: D102
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        if self.time_stepper == "rk3":
            self.set_boundaries(self._substep_time.item())
        psi_i, q_i, sst_anom = prognostic
        psi_bc, q_bc = self.solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc

        u, v = self.gradp(psi)
        u /= self.space.dy
        v /= self.space.dx

        q = q_i + q_bc

        q_with_bc = self.q_bc.expand(q)

        div_flux = self.compute_div_flux_inhomogeneous(u, v, q_with_bc)
        # wind forcing + bottom drag
        fcg_drag = self.compute_drag_inhomogeneous(psi)
        forcing = self._fext(self.time)
        dq = (-div_flux + fcg_drag + forcing) * self.masks.h
        dq_i = self.interp(dq)
        # Solve Helmholtz equation
        dpsi = self.solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )
        dsst_anom = self.compute_dsst_inhomogeneous(u, v, sst_anom)
        return PSIQSST(dpsi, dq, dsst_anom)

    compute_time_derivatives_inhomogeneous.__doc__ = (
        QGPSIQ.compute_time_derivatives_inhomogeneous.__doc__
    )


class QGPSIQRGPsi2Transport(QGPSIQ):
    """QGPSIQ with psi2 wv material derivation forcing."""

    _dyn_basis: Basis
    implements_alpha = True

    @property
    def basis_dynamical(
        self,
    ) -> SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]:
        """Decomposition basis."""
        return self._dyn_basis

    @basis_dynamical.setter
    def basis_dynamical(
        self,
        basis: SpaceTimeDecomposition[
            SpaceSupportFunction, TimeSupportFunction
        ],
    ) -> None:
        self._dyn_basis = basis
        space = self.space.remove_h()
        self._fpsi2 = basis.localize(space.q.xy.x, space.q.xy.y)
        self._fpsi2_psi_grid = basis.localize(space.psi.xy.x, space.psi.xy.y)
        self._fpsi2_wide = basis.localize(
            *self.space.remove_h()
            .widen(1 + 2 * self.wide, 1 + 2 * self.wide)
            .q.xy
        )

    @property
    def psi2(self) -> torch.Tensor:
        """Psi2."""
        return self._fpsi2_psi_grid(self.time)[None, None, ...]

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha
        self.set_solver()

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: Tensor,
        beta_plane: BetaPlane,
        g_prime: Tensor,
        compile_fns: bool = True,
    ) -> None:
        """Model Instantiation.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor.
                └── (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor.
                └── (nl,) shaped.
            beta_plane (Beta_Plane): Beta plane.
            compile_fns (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(
            space_2d=space_2d,
            H=H,
            beta_plane=beta_plane,
            g_prime=g_prime,
            compile_fns=compile_fns,
        )
        self._A11 = self.A[0, 0]
        self._A12 = self.A[0, 1]
        self.zeros_inside = (
            torch.zeros(
                (self.n_ens, self.space.nl - 3, self.space.nx, self.space.ny),
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            if (self.space.nl - 3) > 0
            else None
        )

    def compute_q_anom_from_psi(self, psi: Tensor) -> Tensor:  # noqa: D102
        vort = self.compute_vort_from_psi(psi)
        stretching = (
            self.beta_plane.f0**2 * (self._A11 + self.alpha * self._A12) * psi
        )
        if self.with_bc:
            return vort - self.masks.h * self.interp(stretching)
        return vort - self.masks.h * self.interp(self.masks.psi * stretching)

    compute_q_anom_from_psi.__doc__ = QGPSIQ.compute_q_anom_from_psi.__doc__

    def compute_drag_inhomogeneous(self, psi: torch.Tensor) -> torch.Tensor:  # noqa: D102
        sf_boundary = self.sf_bc_interp(self.time.item())
        sf_wide = sf_boundary.expand(psi[..., 1:-1, 1:-1])
        omega = self.interp(self.lap(sf_wide))
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.space.nl - 1 == 1:
            fcg_drag = self.curl_tau + bottom_drag
        elif self.space.nl - 1 == 2:
            fcg_drag = torch.cat([self.curl_tau, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [self.curl_tau, self.zeros_inside, bottom_drag],
                dim=-3,
            )
        return fcg_drag

    compute_drag_inhomogeneous.__doc__ = (
        QGPSIQ.compute_drag_inhomogeneous.__doc__
    )

    def compute_drag_homogeneous(self, psi: torch.Tensor) -> torch.Tensor:  # noqa: D102
        omega = self.interp(
            self.lap_h(psi, self.space.dx, self.space.dy) * self.masks.psi,
        )
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.space.nl - 1 == 1:
            fcg_drag = self.curl_tau + bottom_drag
        elif self.space.nl - 1 == 2:
            fcg_drag = torch.cat([self.curl_tau, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [self.curl_tau, self.zeros_inside, bottom_drag],
                dim=-3,
            )
        return fcg_drag

    compute_drag_homogeneous.__doc__ = QGPSIQ.compute_drag_homogeneous.__doc__

    def set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # PV equation solver
        self.solver_homogeneous = HomogeneousPVInversionCollinear(
            self.A,
            self.alpha,
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        self.solver_inhomogeneous = InhomogeneousPVInversionCollinear(
            self.A,
            self.alpha,
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        if self._with_bc:
            sf_bc = self.sf_bc_interp(self.time.item())
            self.solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

    def compute_forcing_inhomogeneous(
        self,
        time: torch.Tensor,
        psi1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forcing.

        Args:
            time (torch.Tensor): Time to evaluate at.
            psi1 (torch.Tensor): Top layer stream function.

        Returns:
            torch.Tensor: -f₀²/H₂g₂[∂ₜѱ₂ + J(ѱ₁, ѱ₂)]
        """
        dt_psi2 = self._fpsi2.dt(time)

        u, v = self.gradp(psi1)
        u /= self.space.dy
        v /= self.space.dx
        psi2 = self._fpsi2_wide(time)
        adv = self.div_flux(psi2, u, v)

        return (self.beta_plane.f0**2) * self._A12 * (dt_psi2 + adv)

    def compute_forcing_homogeneous(
        self,
        time: torch.Tensor,
        psi1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forcing.

        Args:
            time (torch.Tensor): Time to evaluate at.
            psi1 (torch.Tensor): Top layer stream function.

        Returns:
            torch.Tensor: -f₀²/H₂g₂[∂ₜѱ₂ + J(ѱ₁, ѱ₂)]
        """
        dt_psi2 = self._fpsi2.dt(time)

        u, v = self.gradp(psi1)
        u /= self.space.dy
        v /= self.space.dx
        psi2 = self._fpsi2(time)
        adv = self.div_flux(psi2, u, v)

        return (self.beta_plane.f0**2) * self._A12 * (dt_psi2 + adv)

    def compute_time_derivatives_homogeneous(  # noqa: D102
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        psi, q, sst_anom = prognostic

        u, v = self.gradp(psi)
        u /= self.space.dy
        v /= self.space.dx

        div_flux = self.compute_div_flux_homogeneous(u, v, q)
        # wind forcing + bottom drag
        fcg_drag = self.compute_drag_homogeneous(psi)
        forcing = self.compute_forcing_homogeneous(
            self._substep_time, psi[:, :1]
        )
        dq = (-div_flux + fcg_drag + forcing) * self.masks.h
        dq_i = self.interp(dq)
        # Solve Helmholtz equation
        dpsi = self.solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )
        dsst_anom = self.compute_dsst_homogeneous(u, v, sst_anom)
        return PSIQSST(dpsi, dq, dsst_anom)

    compute_time_derivatives_homogeneous.__doc__ = (
        QGPSIQ.compute_time_derivatives_homogeneous.__doc__
    )

    def compute_time_derivatives_inhomogeneous(  # noqa: D102
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        if self.time_stepper == "rk3":
            self.set_boundaries(self._substep_time.item())
        psi_i, q_i, sst_anom = prognostic
        psi_bc, q_bc = self.solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc

        u, v = self.gradp(psi)
        u /= self.space.dy
        v /= self.space.dx

        q = q_i + q_bc

        q_with_bc = self.q_bc.expand(q)

        div_flux = self.compute_div_flux_inhomogeneous(u, v, q_with_bc)
        # wind forcing + bottom drag
        fcg_drag = self.compute_drag_inhomogeneous(psi)
        forcing = self.compute_forcing_inhomogeneous(
            self._substep_time, psi[:, :1]
        )
        dq = (-div_flux + fcg_drag + forcing) * self.masks.h
        dq_i = self.interp(dq)
        # Solve Helmholtz equation
        dpsi = self.solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )
        dsst_anom = self.compute_dsst_inhomogeneous(u, v, sst_anom)
        return PSIQSST(dpsi, dq, dsst_anom)

    compute_time_derivatives_inhomogeneous.__doc__ = (
        QGPSIQ.compute_time_derivatives_inhomogeneous.__doc__
    )


class QGPSIQRGPsi2TransportDR(QGPSIQRGPsi2Transport):
    """Mixed model using both alpha and psi2."""

    _type = ModelName.RGSI

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        try:
            return self._alpha
        except AttributeError:
            self._alpha = torch.tensor(0, **self.specs)
            return self._alpha

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha
        self.compute_auxillary_matrices()
        self.set_solver()

    def set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # PV equation solver
        self.solver_homogeneous = HomogeneousPVInversion(
            self.A[:1, :1],
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        self.solver_inhomogeneous = InhomogeneousPVInversion(
            self.A[:1, :1],
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        if self._with_bc:
            sf_bc = self.sf_bc_interp(self.time.item())
            self.solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

    def compute_auxillary_matrices(self) -> None:
        """Compute auxillary matrices."""
        H = self.H[:, 0, 0]
        g_prime = self.g_prime[:, 0, 0]

        self.A = compute_A_tilde(H, g_prime, self.alpha, **defaults.get())
        self._A11 = self.A[:1, :1]
        self._A12 = self.A[:1, 1:2]

    def compute_q_anom_from_psi(self, psi: Tensor) -> Tensor:  # noqa: D102
        vort = self.compute_vort_from_psi(psi)
        stretching = self.beta_plane.f0**2 * self._A11 * psi
        if self.with_bc:
            return vort - self.masks.h * self.interp(stretching)
        return vort - self.masks.h * self.interp(self.masks.psi * stretching)

    compute_q_anom_from_psi.__doc__ = QGPSIQ.compute_q_anom_from_psi.__doc__


class QGPSIQForcedSurfML(QGPSIQRGPsi2TransportDR):
    """Mixed model using both alpha and psi2 along with an external forcing."""

    _type = ModelName.RGFSI

    @property
    def basis_external(self) -> Basis:
        """Forcing basis."""
        return self._ext_basis

    @basis_external.setter
    def basis_external(self, basis: Basis) -> None:
        self._ext_basis = basis
        self._fext = self._ext_basis.localize(*self.space.remove_h().q.xy)

    @property
    def forcing(self) -> torch.Tensor:
        """Forcing."""
        return self._fext(self.time)[None, None, ...]

    def compute_time_derivatives_homogeneous(  # noqa: D102
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        psi, q, sst_anom = prognostic

        u, v = self.gradp(psi)
        u /= self.space.dy
        v /= self.space.dx

        div_flux = self.compute_div_flux_homogeneous(u, v, q)
        # wind forcing + bottom drag
        fcg_drag = self.compute_drag_homogeneous(psi)
        forcing = self.compute_forcing_homogeneous(
            self._substep_time, psi[:, :1]
        )
        forcing += self._fext(self.time)
        dq = (-div_flux + fcg_drag + forcing) * self.masks.h
        dq_i = self.interp(dq)
        # Solve Helmholtz equation
        dpsi = self.solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )
        dsst_anom = self.compute_dsst_homogeneous(u, v, sst_anom)
        return PSIQSST(dpsi, dq, dsst_anom)

    compute_time_derivatives_homogeneous.__doc__ = (
        QGPSIQ.compute_time_derivatives_homogeneous.__doc__
    )

    def compute_time_derivatives_inhomogeneous(  # noqa: D102
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        if self.time_stepper == "rk3":
            self.set_boundaries(self._substep_time.item())
        psi_i, q_i, sst_anom = prognostic
        psi_bc, q_bc = self.solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc

        u, v = self.gradp(psi)
        u /= self.space.dy
        v /= self.space.dx

        q = q_i + q_bc

        q_with_bc = self.q_bc.expand(q)

        div_flux = self.compute_div_flux_inhomogeneous(u, v, q_with_bc)
        # wind forcing + bottom drag
        fcg_drag = self.compute_drag_inhomogeneous(psi)
        forcing = self.compute_forcing_inhomogeneous(
            self._substep_time, psi[:, :1]
        )
        forcing += self._fext(self.time)
        dq = (-div_flux + fcg_drag + forcing) * self.masks.h
        dq_i = self.interp(dq)
        # Solve Helmholtz equation
        dpsi = self.solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )
        dsst_anom = self.compute_dsst_inhomogeneous(u, v, sst_anom)
        return PSIQSST(dpsi, dq, dsst_anom)

    compute_time_derivatives_inhomogeneous.__doc__ = (
        QGPSIQ.compute_time_derivatives_inhomogeneous.__doc__
    )
