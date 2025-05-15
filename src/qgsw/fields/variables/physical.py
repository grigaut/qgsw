"""Physical variables.

This file regroups variables which are computed from the physical variables.
"""

from __future__ import annotations

import contextlib

from qgsw.fields.scope import Scope
from qgsw.fields.variables.covariant import (
    PhysicalLayerDepthAnomaly,
    PhysicalMeridionalVelocity,
    PhysicalZonalVelocity,
)
from qgsw.fields.variables.prognostic import (
    CollinearityCoefficient,
    PrognosticPotentialVorticity,
    PrognosticStreamFunction,
    Time,
)
from qgsw.fields.variables.state import StateUVH
from qgsw.models.core.helmholtz import (
    compute_laplace_dstI,
    solve_helmholtz_dstI,
)
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.specs import DEVICE
from qgsw.utils.covphys import to_cov
from qgsw.utils.units._units import Unit

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.fields.variables.base import (
    BoundDiagnosticVariable,
    DiagnosticVariable,
)
from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.models.core.utils import OptimizableFunction
from qgsw.spatial.core.grid_conversion import points_to_surfaces

if TYPE_CHECKING:
    from qgsw.fields.variables.prognostic_tuples import (
        UVHT,
        BasePrognosticPSIQ,
        BaseUVH,
        UVHTAlpha,
    )
    from qgsw.fields.variables.state import StateUVH
    from qgsw.models.qg.uvh.projectors.core import QGProjector


class SurfaceHeightAnomaly(DiagnosticVariable):
    """Physical Surface height anomaly.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M
    _name = "eta_phys"
    _description = "Physical Surface height anomaly"
    _scope = Scope.POINT_WISE

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Surface height anomaly
                └── (n_ens, nl, nx, ny)-shaped
        """
        return reverse_cumsum(
            vars_tuple.h,
            dim=-3,
        )


class Vorticity(DiagnosticVariable):
    """Physical vorticity.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.S_1
    _name = "omega_phys"
    _description = "Physical vorticity"
    _scope = Scope.POINT_WISE

    def __init__(self, dx: float, dy: float) -> None:
        """Instantiate the variable.

        Args:
            vorticity (Vorticity): Vorticity.
            dx (float): Elementary distance in the X direction.
            dy (float): Elementary distance in the Y direction.
        """
        self._dx = dx
        self._dy = dy

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical vorticity.
                └── (n_ens, nl, nx, ny)-shaped
        """
        return (
            torch.diff(vars_tuple.v[..., 1:-1], dim=-2) / self._dx
            - torch.diff(vars_tuple.u[..., 1:-1, :], dim=-1) / self._dy
        )


class QGPressure(DiagnosticVariable):
    """QGPressure.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M2S_2
    _name = "p"
    _description = "Pressure per unit of mass"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        P: QGProjector,  # noqa: N803
        dx: float,
        dy: float,
    ) -> None:
        """Instantiate the pressure variable.

        Args:
            P (QGprojector): Projector to use to retrieve pressure.
            dx (float): Infinitesimal distance in the X direction.
            dy (float): Infinitesimal distance in the Y direction.

        """
        self._P = P
        self._dx = dx
        self._dy = dy

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            vars_tuple (BaseUVH): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Pressure.
                └── (n_ens, nl, nx, ny)-shaped
        """
        with contextlib.suppress(AttributeError):
            self._P.alpha = vars_tuple.alpha
        return self._P.compute_p(to_cov(vars_tuple, self._dx, self._dy))[1]


class Pressure(DiagnosticVariable):
    """Pressure.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M2S_2
    _name = "p"
    _description = "Pressure per unit of mass"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        g_prime: torch.Tensor,
        eta_phys: SurfaceHeightAnomaly,
    ) -> None:
        """Instantiate the pressure variable.

        Args:
            g_prime (torch.Tensor): Reduced gravity.
                └── (1, nl, 1, 1)-shaped
            eta_phys (SurfaceHeightAnomaly): Surface height anomaly
            variable.
        """
        self._g_prime = g_prime
        self._eta = eta_phys

        self._require_alpha |= eta_phys.require_alpha
        self._require_time |= eta_phys.require_time

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Pressure.
                └── (n_ens, nl, nx, ny)-shaped
        """
        return torch.cumsum(
            self._g_prime * self._eta.compute_no_slice(vars_tuple),
            dim=-3,
        )

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the eta_phys variable
        self._eta = self._eta.bind(state)
        return super().bind(state)


class PressureTilde(Pressure):
    """Pressure tilde.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _name = "p_tilde"
    _description = "Pressure per unit of mass for collinear model"
    _require_alpha = True

    def __init__(
        self,
        g_prime: torch.Tensor,
        eta_phys: SurfaceHeightAnomaly,
    ) -> None:
        """Instantiate the pressure variable.

        Args:
            g_prime (torch.Tensor): Reduced gravity.
                └── (1, 2, 1, 1)-shaped
            eta_phys (SurfaceHeightAnomaly): Surface height anomaly
            variable.
        """
        if not g_prime.squeeze().shape[0] == 2:  # noqa: PLR2004
            raise ValueError
        self._g1 = g_prime.squeeze()[0]
        self._g2 = g_prime.squeeze()[1]
        self._eta = eta_phys

        self._require_alpha |= eta_phys.require_alpha
        self._require_time |= eta_phys.require_time

    def _compute_g_tilde(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute g_tilde.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.

        Returns:
            torch.Tensor: g_tilde
        """
        return self._g1 * self._g2 / (self._g2 + (1 - alpha) * self._g1)

    def _compute(self, vars_tuple: UVHTAlpha) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            vars_tuple (UVHTAlpha): Prognostic variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Pressure
                └── (n_ens, nl, nx, ny)-shaped
        """
        g_tilde = self._compute_g_tilde(vars_tuple.alpha)
        return g_tilde * self._eta.compute_no_slice(vars_tuple)


class PotentialVorticity(DiagnosticVariable):
    """Potential Vorticity.

    └── (n_ens, nl, nx-1, ny-1)-shaped
    """

    _unit = Unit.S_1
    _name = "pv"
    _description = "Potential vorticity"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        H: torch.Tensor,  # noqa: N803
        f0: float,
        dx: float,
        dy: float,
    ) -> None:
        """Instantiate variable.

        Args:
            H (torch.Tensor): Reference thickness values.
                └── (1, nl, 1, 1)-shaped
            f0 (float): Coriolis parameter.
            dx (float): Elementary distance in the X direction.
            dy (float): Elementary distance in the Y direction.
        """
        self._H = H
        self._dx = dx
        self._dy = dy
        self._f0 = f0
        self._interp = OptimizableFunction(points_to_surfaces)

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Value
                └── (n_ens, nl, nx-1, ny-1)-shaped
        """
        # Compute ω = ∂_x v - ∂_y u
        omega = (
            torch.diff(vars_tuple.v[..., 1:-1], dim=-2) / self._dx
            - torch.diff(
                vars_tuple.u[..., 1:-1, :],
                dim=-1,
            )
            / self._dy
        )
        h = self._interp(vars_tuple.h)
        return omega - self._f0 * h / self._H


class StreamFunction(DiagnosticVariable):
    """Stream function variable.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M2S_1
    _name = "psi"
    _description = "Stream function"
    _scope = Scope.POINT_WISE

    def __init__(self, pressure: Pressure, f0: float) -> None:
        """Instantiate the variable.

        Args:
            pressure (Pressure): Pressure variable.
            f0 (float): Coriolis parameter.
        """
        self._p = pressure
        self._f0 = f0

        self._require_alpha |= pressure.require_alpha
        self._require_time |= pressure.require_time

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function.
                └── (n_ens, nl, nx, ny)-shaped
        """
        return self._p.compute_no_slice(vars_tuple) / self._f0

    def bind(self, state: StateUVH) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the pressure variable
        self._p = self._p.bind(state)
        return super().bind(state)


class StreamFunctionFromVorticity(DiagnosticVariable):
    """Stream function variable from vorticity inversion.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M2S_1
    _name = "psi_from_omega"
    _description = "Stream function from vorticity inversion"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        vorticity: Vorticity,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            vorticity (Vorticity): Physical vorticity.
            nx (int): Number of poitn in the x direction.
            ny (int): Number of points in the y direction.
            dx (float): Infinitesimal x step.
            dy (float): Infinitesimal y step.
        """
        self._vorticity = vorticity

        self._require_alpha |= vorticity.require_alpha
        self._require_time |= vorticity.require_time

        self._laplacian = (
            compute_laplace_dstI(
                nx,
                ny,
                dx,
                dy,
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function.
                └── (n_ens, nl, nx, ny)-shaped
        """
        vorticity = self._vorticity.compute_no_slice(vars_tuple)
        return points_to_surfaces(
            solve_helmholtz_dstI(vorticity, self._laplacian),
        )

    def bind(self, state: StateUVH) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the vorticity variable
        self._vorticity = self._vorticity.bind(state)
        return super().bind(state)


class Psi2(DiagnosticVariable):
    """Stream function variable in second layer.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M2S_1
    _name = "psi2"
    _description = "Stream function in second layer"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        psi_vort: StreamFunctionFromVorticity,
    ) -> None:
        """Instantiate the variable.

        Args:
            psi_vort (StreamFunctionFromVorticity): Stream Function.
        """
        self._psi_vort = psi_vort

        self._require_alpha |= psi_vort.require_alpha
        self._require_time |= psi_vort.require_time

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function in second layer.
                └── (n_ens, 1, nx, ny)-shaped
        """
        return self._psi_vort.compute_no_slice(vars_tuple)[:, 1:2]

    def bind(self, state: StateUVH) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the vorticity variable
        self._psi_vort = self._psi_vort.bind(state)
        return super().bind(state)


class Enstrophy(DiagnosticVariable):
    """Layer-wise enstrophy.

    └── (n_ens, nl,)-shaped
    """

    _unit = Unit.S_2
    _name = "enstrophy"
    _description = "Layer-wise enstrophy"
    _scope = Scope.LEVEL_WISE

    def __init__(self, pv: PotentialVorticity) -> None:
        """Instantiate the variable.

        Args:
            pv (PotentialVorticity): Physical vorticity.
        """
        self._pv = pv
        self._require_alpha |= pv.require_alpha
        self._require_time |= pv.require_time

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Enstrophy.
                └── (n_ens, nl,)-shaped
        """
        return 0.5 * torch.sum(
            self._pv.compute_no_slice(vars_tuple) ** 2,
            dim=(-1, -2),
        )

    def bind(self, state: StateUVH) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a state.

        Args:
            state (StateUVH): StateUVH.

        Returns:
            BoundDiagnosticVariable[Self]: Bound variable.
        """
        self._pv.bind(state)
        return super().bind(state)


class TotalEnstrophy(Enstrophy):
    """Total enstrophy.

    └── (n_ens,)-shaped
    """

    _unit = Unit.S_2
    _name = "enstrophy_tot"
    _description = "Total enstrophy"
    _scope = Scope.ENSEMBLE_WISE

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Enstrophy.
                └── (n_ens,)-shaped
        """
        return 0.5 * torch.sum(
            self._pv.compute_no_slice(vars_tuple) ** 2,
            dim=(-1, -2, -3),
        )


class ZonalVelocity2(DiagnosticVariable):
    """Physical zonal Velocity in second layer.

    └── (n_ens, 1, nx, ny-1)-shaped
    """

    _scope = Scope.POINT_WISE
    _unit = Unit.M1S_1
    _name = "u_phys2"
    _description = "Physical zonal velocity in second layer"

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical zonal velocity in second layer..
                └── (n_ens, 1, nx, ny-1)-shaped
        """
        return points_to_surfaces(vars_tuple.u)[:, 1:2]


class MeridionalVelocity2(DiagnosticVariable):
    """Physical meridional Velocity in second layer.

    └── (n_ens, 1, nx-1, ny)-shaped
    """

    _scope = Scope.POINT_WISE
    _unit = Unit.M1S_1
    _name = "v_phys2"
    _description = "Physical meridional velocity in second layer"

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical zonal velocity in second layer..
                └── (n_ens, 1, nx-1, ny)-shaped
        """
        return points_to_surfaces(vars_tuple.v)[:, 1:2]


class RefZonalVelocity2(DiagnosticVariable):
    """Physical zonal Velocity in second layer.

    └── (n_ens, 1, nx, ny-1)-shaped
    """

    _scope = Scope.POINT_WISE
    _unit = Unit.M1S_1
    _name = ZonalVelocity2.get_name()
    _description = "Physical zonal velocity in second layer"

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical zonal velocity in second layer..
                └── (n_ens, 1, nx, ny-1)-shaped
        """
        return torch.zeros_like(points_to_surfaces(vars_tuple.u)[:, :1])


class RefMeridionalVelocity2(DiagnosticVariable):
    """Physical meridional Velocity in second layer.

    └── (n_ens, 1, nx-1, ny)-shaped
    """

    _scope = Scope.POINT_WISE
    _unit = Unit.M1S_1
    _name = MeridionalVelocity2.get_name()
    _description = "Physical meridional velocity in second layer"

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical zonal velocity in second layer..
                └── (n_ens, 1, nx-1, ny)-shaped
        """
        return torch.zeros_like(points_to_surfaces(vars_tuple.v)[:, :1])


class ZonalVelocityFromPsi2(DiagnosticVariable):
    """Physical zonal Velocity in second layer.

    └── (n_ens, 1, nx, ny-1)-shaped
    """

    _scope = Scope.POINT_WISE
    _unit = Unit.M2S_1
    _name = "u_phys2"
    _description = "Physical zonal velocity in bottom layer"

    def __init__(self, psi2: Psi2, dy: float) -> None:
        """Instantiate the variable.

        Args:
            psi2 (Psi2): Vorticity in second layer.
            dy (float): Step in the Y direction.
        """
        self._psi2 = psi2
        self._dy = dy

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical zonal velocity in second layer..
                └── (n_ens, 1, nx, ny-1)-shaped
        """
        psi2 = self._psi2.compute_no_slice(vars_tuple)
        return -torch.diff(psi2, dim=-1) / self._dy


class MeridionalVelocityFromPsi2(DiagnosticVariable):
    """Physical meridional Velocity in second layer.

    └── (n_ens, 1, nx-1, ny)-shaped
    """

    _scope = Scope.POINT_WISE
    _unit = Unit.M2S_1
    _name = "v_phys2"
    _description = "Physical meridional velocity in bottom layer"

    def __init__(self, psi2: Psi2, dx: float) -> None:
        """Instantiate the variable.

        Args:
            psi2 (Psi2): Vorticity in second layer.
            dx (float): Step in the X direction.
        """
        self._psi2 = psi2
        self._dx = dx

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical zonal velocity in second layer..
                └── (n_ens, 1, nx-1, ny)-shaped
        """
        psi2 = self._psi2.compute_no_slice(vars_tuple)
        return torch.diff(psi2, dim=-2) / self._dx


class TimeDiag(DiagnosticVariable):
    """Diagnostic Layer Depth Anomaly.

    └── (n_ens,)-shaped
    """

    _unit = Time.get_unit()
    _name = Time.get_name()
    _description = Time.get_description()
    _scope = Time.get_scope()

    def _compute(self, vars_tuple: UVHT) -> torch.Tensor:
        """Compute variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Value.
                └── (n_ens, )-shaped
        """
        return vars_tuple.t


class CollinearityCoefficientDiag(DiagnosticVariable):
    """Diagnostic Layer Depth Anomaly.

    └── (n_ens, )-shaped
    """

    _unit = CollinearityCoefficient.get_unit()
    _name = CollinearityCoefficient.get_name()
    _description = CollinearityCoefficient.get_description()
    _scope = CollinearityCoefficient.get_scope()

    def _compute(self, vars_tuple: UVHTAlpha) -> torch.Tensor:
        """Compute variable value.

        Args:
            vars_tuple (UVHTAlpha): Prognostic variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Value.
                └── (n_ens, )-shaped
        """
        return vars_tuple.alpha


class ProgStreamFunctionDiag(DiagnosticVariable):
    """Diagnostic Stream Function.

    └── (n_ens, nl, nx+1, ny+1)-shaped
    """

    _unit = PrognosticStreamFunction.get_unit()
    _name = PrognosticStreamFunction.get_name()
    _description = PrognosticStreamFunction.get_description()
    _scope = PrognosticStreamFunction.get_scope()

    def _compute(self, vars_tuple: BasePrognosticPSIQ) -> torch.Tensor:
        """Compute variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t,) psi and q.
                ├── (t: (n_ens,)-shaped)
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── q: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Value.
                └── (n_ens, nl, nx+1, ny+1)-shaped
        """
        return vars_tuple.psi


class ProgPotentialVorticityDiag(DiagnosticVariable):
    """Diagnostic Potential Vorticity.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = PrognosticPotentialVorticity.get_unit()
    _name = PrognosticPotentialVorticity.get_name()
    _description = PrognosticPotentialVorticity.get_description()
    _scope = PrognosticPotentialVorticity.get_scope()

    def _compute(self, vars_tuple: BasePrognosticPSIQ) -> torch.Tensor:
        """Compute variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t,) psi and q.
                ├── (t: (n_ens,)-shaped)
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── q: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Value.
                └── (n_ens, nl, nx, ny)-shaped
        """
        return vars_tuple.q


class ZonalVelocity(DiagnosticVariable):
    """Physical zonal velocity.

    └── (n_ens, nl, nx+1, ny)-shaped
    """

    _unit = PhysicalZonalVelocity.get_unit()
    _name = PhysicalZonalVelocity.get_name()
    _description = PhysicalZonalVelocity.get_description()
    _scope = PhysicalZonalVelocity.get_scope()

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical zonal velocity component.
                └── (n_ens, nl, nx+1, ny)-shaped
        """
        return vars_tuple.u


class MeridionalVelocity(DiagnosticVariable):
    """Physical zonal velocity.

    └── (n_ens, nl, nx, ny+1)-shaped
    """

    _unit = PhysicalMeridionalVelocity.get_unit()
    _name = PhysicalMeridionalVelocity.get_name()
    _description = PhysicalMeridionalVelocity.get_description()
    _scope = PhysicalMeridionalVelocity.get_scope()

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical zonal velocity component.
                └── (n_ens, nl, nx, ny+1)-shaped
        """
        return vars_tuple.v


class LayerDepthAnomaly(DiagnosticVariable):
    """Physical layer depth anomaly.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = PhysicalLayerDepthAnomaly.get_unit()
    _name = PhysicalLayerDepthAnomaly.get_name()
    _description = PhysicalLayerDepthAnomaly.get_description()
    _scope = PhysicalLayerDepthAnomaly.get_scope()

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical layer depth anomaly.
                └── (n_ens, nl, nx, ny)-shaped
        """
        return vars_tuple.h


def compute_W(H: torch.Tensor) -> torch.Tensor:  # noqa: N802, N803
    """Compute the weight matrix.

    Args:
        H (torch.Tensor): Layers reference depths.
            └── (nl,)-shaped

    Returns:
        torch.Tensor: Weight Matrix
            └── (nl, nl)-shaped
    """
    return torch.diag(H) / torch.sum(H)


class ModalKineticEnergy(DiagnosticVariable):
    """Compute modal kinetic energy.

    └── (n_ens, nl)-shaped
    """

    _unit = Unit.M2S_2
    _name = "ke_hat"
    _description = "Modal kinetic energy"
    _scope = Scope.LEVEL_WISE

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        stream_function: StreamFunction,
        H: torch.Tensor,  # noqa: N803
        dx: float,
        dy: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
                └── (nl, nl)-shaped
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            H (torch.Tensor): Layers reference depth.
                └── (nl,)-shaped
            dx (float): Elementary distance in the X direction.
            dy (float): Elementary distance in the Y direction.
        """
        self._psi = stream_function
        self._require_alpha |= stream_function.require_alpha
        self._require_time |= stream_function.require_time

        self._dx = dx
        self._dy = dy
        # Decomposition of A
        Cm2l, _, self._Cl2m = compute_layers_to_mode_decomposition(A)  # noqa: N806
        # Compute W = Diag(H) / h_{tot}
        W = compute_W(H)  # noqa: N806
        # Compute Cl2m^{-T} @ W @ Cl2m⁻¹
        Cm2l_T = Cm2l.transpose(dim0=0, dim1=1)  # noqa: N806
        Cm2lT_W_Cm2l = Cm2l_T @ W @ Cm2l  # noqa: N806
        # Since Cm2lT_W_Cm2l is diagonal
        self._Cm2lT_W_Cm2l = torch.diag(Cm2lT_W_Cm2l)  # Vector

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Modal kinetic energy.
                └── (n_ens, nl)-shaped
        """
        psi = self._psi.compute_no_slice(vars_tuple)
        psi_hat = torch.einsum("lm,...mxy->...lxy", self._Cl2m, psi)
        # Pad x with 0 on top
        psi_hat_pad_x = F.pad(psi_hat, (0, 0, 0, 1))
        # Pad y with 0 on the left
        psi_hat_pad_y = F.pad(psi_hat, (0, 1, 0, 0))
        # Differentiate
        psi_hat_dx = torch.diff(psi_hat_pad_x, dim=-2) / self._dx
        psi_hat_dy = torch.diff(psi_hat_pad_y, dim=-1) / self._dy
        psiT_CT_W_C_psi = torch.einsum(  # noqa: N806
            "l,...lxy->...lxy",
            self._Cm2lT_W_Cm2l,
            (psi_hat_dx**2 + psi_hat_dy**2),
        )
        return 0.5 * torch.sum(psiT_CT_W_C_psi, dim=(-1, -2))

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the psi variable
        self._psi = self._psi.bind(state)
        return super().bind(state)


class ModalAvailablePotentialEnergy(DiagnosticVariable):
    """Modal available potential energy.

    └── (n_ens, nl)-shaped
    """

    _unit = Unit.M2S_2
    _name = "ape_hat"
    _description = "Modal available potential energy"
    _scope = Scope.LEVEL_WISE

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        stream_function: StreamFunction,
        H: torch.Tensor,  # noqa: N803
        f0: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
                └── (nl, nl)-shaped
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            H (torch.Tensor): Layers reference depth.
                └── (nl,)-shaped
            f0 (float): Coriolis parameter.
        """
        self._psi = stream_function
        self._require_alpha |= stream_function.require_alpha
        self._require_time |= stream_function.require_time

        self._f0 = f0
        # Decomposition of A
        Cm2l, lambd, self._Cl2m = compute_layers_to_mode_decomposition(A)  # noqa: N806
        # Compute weight matrix
        W = compute_W(H)  # noqa: N806
        # Compute Cl2m^{-T} @ W @ Cl2m⁻¹ @ Λ
        Cm2l_T = Cm2l.transpose(dim0=0, dim1=1)  # noqa: N806
        self._Cm2lT_W_Cm2l_lambda = Cm2l_T @ W @ Cm2l @ lambd  # Vector

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Modal avalaible potential energy.
                └── (n_ens, nl)-shaped
        """
        psi = self._psi.compute_no_slice(vars_tuple)
        psi_hat = torch.einsum("lm,...mxy->...lxy", self._Cl2m, psi)
        psiT_CT_W_C_lambda_psi = torch.einsum(  # noqa: N806
            "l,...lxy->...lxy",
            self._Cm2lT_W_Cm2l_lambda,
            psi_hat**2,
        )
        ape = torch.sum(psiT_CT_W_C_lambda_psi, dim=(-1, -2))
        return 0.5 * self._f0**2 * ape

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the psi variable
        self._psi = self._psi.bind(state)
        return super().bind(state)


class ModalEnergy(DiagnosticVariable):
    """Modal energy.

    └── (n_ens, nl)-shaped
    """

    _unit = Unit.M2S_2
    _name = "e_tot_hat"
    _description = "Modal energy"
    _scope = Scope.LEVEL_WISE

    def __init__(
        self,
        ke_hat: ModalKineticEnergy,
        ape_hat: ModalAvailablePotentialEnergy,
    ) -> None:
        """Instantiate variable.

        Args:
            ke_hat (TotalModalKineticEnergy): Modal kinetic energy
            ape_hat (TotalModalAvailablePotentialEnergy): Modal
            available potential energy
        """
        self._ke = ke_hat
        self._ape = ape_hat
        self._require_alpha |= ke_hat.require_alpha | ape_hat.require_alpha
        self._require_time |= ke_hat.require_time | ape_hat.require_time

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute total modal energy.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Modal total energy.
                └── (n_ens, nl)-shaped
        """
        ke = self._ke.compute_no_slice(vars_tuple)
        ape = self._ape.compute_no_slice(vars_tuple)
        return ke + ape

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the ke variable
        self._ke = self._ke.bind(state)
        # Bind the _ape variable
        self._ape = self._ape.bind(state)
        return super().bind(state)


class TotalKineticEnergy(DiagnosticVariable):
    """Compute total kinetic energy.

    └── (n_ens,)-shaped
    """

    _unit = Unit.M2S_2
    _name = "ke"
    _description = "Kinetic energy"
    _scope = Scope.ENSEMBLE_WISE

    def __init__(
        self,
        stream_function: StreamFunction,
        H: torch.Tensor,  # noqa: N803
        dx: float,
        dy: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
                └── (nl, nl)-shaped
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            H (torch.Tensor): Layers reference depth.
                └── (nl,)-shaped
            dx (float): Elementary distance in the X direction.
            dy (float): Elementary distance in the Y direction.
        """
        self._psi = stream_function
        self._require_alpha |= stream_function.require_alpha
        self._require_time |= stream_function.require_time

        self._dx = dx
        self._dy = dy
        # Compute W = Diag(H) / h_{tot}
        self._W = torch.diag(compute_W(H))  # Vector

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Total modal kinetic energy.
                └── (n_ens,)-shaped
        """
        psi = self._psi.compute_no_slice(vars_tuple)
        # Pad x with 0 on top
        psi_pad_x = F.pad(psi, (0, 0, 0, 1))
        # Pad y with 0 on the left
        psi_pad_y = F.pad(psi, (0, 1, 0, 0))
        # Differentiate
        psi_dx = torch.diff(psi_pad_x, dim=-2) / self._dx
        psi_dy = torch.diff(psi_pad_y, dim=-1) / self._dy
        psiT_W_psi = torch.einsum(  # noqa: N806
            "l,...lxy->...lxy",
            self._W,
            (psi_dx**2 + psi_dy**2),
        )
        return 0.5 * torch.sum(psiT_W_psi, dim=(-1, -2, -3))

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the psi variable
        self._psi = self._psi.bind(state)
        return super().bind(state)


class TotalAvailablePotentialEnergy(DiagnosticVariable):
    """Total modal available potential energy.

    └── (n_ens,)-shaped
    """

    _unit = Unit.M2S_2
    _name = "ape"
    _description = "Available potential energy"
    _scope = Scope.ENSEMBLE_WISE

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        stream_function: StreamFunction,
        H: torch.Tensor,  # noqa: N803
        f0: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
                └── (nl, nl)-shaped
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            H (torch.Tensor): Layers reference depth.
                └── (nl,)-shaped
            f0 (float): Coriolis parameter.
        """
        self._psi = stream_function
        self._require_alpha |= stream_function.require_alpha
        self._require_time |= stream_function.require_time

        self._f0 = f0
        self._A = A
        # Compute weight matrix
        self._W = compute_W(H)  # Matrix

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Total modal avalaible potential energy.
                └── (n_ens,)-shaped
        """
        psi = self._psi.compute_no_slice(vars_tuple)
        W_A_psi = torch.einsum(  # noqa: N806
            "lm,...mxy->...lxy",
            self._W @ self._A,
            psi,
        )
        psiT_W_A_psi = torch.einsum(  # noqa: N806
            "...lxy,...lxy->...lxy",
            psi,
            W_A_psi,
        )
        ape = torch.sum(psiT_W_A_psi, dim=(-1, -2, -3))
        return 0.5 * self._f0**2 * ape

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the psi variable
        self._psi = self._psi.bind(state)
        return super().bind(state)


class TotalEnergy(DiagnosticVariable):
    """Total modal energy.

    └── (n_ens,)-shaped
    """

    _unit = Unit.M2S_2
    _name = "e_tot"
    _description = "Total energy"
    _scope = Scope.ENSEMBLE_WISE

    def __init__(
        self,
        ke: TotalKineticEnergy,
        ape: TotalAvailablePotentialEnergy,
    ) -> None:
        """Instantiate variable.

        Args:
            ke (TotalKineticEnergy): Total modal kinetic energy
            ape (TotalAvailablePotentialEnergy): Total modal
            available potential energy
        """
        self._ke = ke
        self._ape = ape

        self._require_alpha |= ke.require_alpha | ke.require_alpha
        self._require_time |= ke.require_time | ke.require_time

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute total modal energy.

        Args:
            vars_tuple (BaseUVH): Physical variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Total modal energy.
                └── (n_ens,)-shaped
        """
        return self._ke.compute_no_slice(
            vars_tuple,
        ) + self._ape.compute_no_slice(vars_tuple)

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the ke variable
        self._ke = self._ke.bind(state)
        # Bind the _ape variable
        self._ape = self._ape.bind(state)
        return super().bind(state)
