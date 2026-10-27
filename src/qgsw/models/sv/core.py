"""Surface vorticity model."""

import torch

from qgsw import specs
from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.exceptions import InvalidModelParameterError, UnsetTimestepError
from qgsw.fields.variables.tuples import PSIZeta
from qgsw.logging.core import getLogger
from qgsw.masks import Masks
from qgsw.models.base import ABCCounter
from qgsw.models.core import time_steppers
from qgsw.models.core.flux import (
    div_flux_5pts,
    div_flux_5pts_only,
    div_flux_5pts_with_bc,
)
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.names import ModelName
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import grad, grad_perp, laplacian, laplacian_h
from qgsw.solver.pv_inversion import (
    HomogeneousPVInversion,
    InhomogeneousPVInversion,
)
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.utils.interpolation import LinearInterpolation
from qgsw.utils.named_object import NamedObject
from qgsw.utils.reshaping import crop

logger = getLogger(__name__)


class SurfaceVorticity(NamedObject[ModelName], metaclass=ABCCounter):
    """Surface vorticity model."""

    _type = ModelName.SURFACE_VORTICITY
    wide = True
    implements_alpha = False

    @property
    def basis_psi_a(
        self,
    ) -> SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]:
        """Ageostrophic streamfunction basis."""
        return self._basis_psi_a

    @basis_psi_a.setter
    def basis_psi_a(
        self,
        basis: SpaceTimeDecomposition[
            SpaceSupportFunction, TimeSupportFunction
        ],
    ) -> None:
        self._basis_psi_a = basis
        self.f_psi_a = basis.localize(*self.space.psi.xy)
        self.f_zeta_a = basis.localize_laplacian(*self.space.q.xy)
        if self.with_bc:
            self.f_zeta_a_wide = basis.localize_laplacian(
                *self.wide_space.q.xy
            )

    @property
    def basis_chi_a(
        self,
    ) -> SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]:
        """Ageostrophic potential basis."""
        return self._basis_chi_a

    @basis_chi_a.setter
    def basis_chi_a(
        self,
        basis: SpaceTimeDecomposition[
            SpaceSupportFunction, TimeSupportFunction
        ],
    ) -> None:
        self._basis_chi_a = basis
        wide_space = self.space.widen(1, 1)
        self.f_chi_a = basis.localize(*wide_space.q.xy)

    @property
    def _substep_time(self) -> torch.Tensor:
        if self.time_stepper == "euler":
            return self.time
        if self.time_stepper == "rk3":
            if self._rk3_step == 0:
                coef = 0
                return self.time + coef * self.dt
            if self._rk3_step == 1:
                coef = 1
                return self.time + coef * self.dt
            if self._rk3_step == 2:
                coef = 1 / 2
                return self.time + coef * self.dt
            msg = "SSPRK3 should only perform 3 steps."
            raise ValueError(msg)
        msg = "Supported time steppers are 'rk3' and 'euler'."
        raise ValueError(msg)

    @property
    def psi_g(self) -> torch.Tensor:
        """Geostrophic streamfunction."""
        return self.psiz.psi

    @property
    def zeta_g(self) -> torch.Tensor:
        """Geostrophic vorticity."""
        return self.psiz.zeta

    @property
    def psi(self) -> torch.Tensor:
        """Total stream function."""
        return self.psi_a + self.psi_g

    @property
    def psi_a(self) -> torch.Tensor:
        """Ageostrophic streamfunction."""
        return self.f_psi_a(self.time)[None, None, ...]

    @property
    def zeta(self) -> torch.Tensor:
        """Total vorticity."""
        return self.zeta_a + self.zeta_g

    @property
    def zeta_a(self) -> torch.Tensor:
        """Ageostrophic vorticity."""
        return self.f_zeta_a(self.time)[None, None, ...]

    @property
    def chi_a(self) -> torch.Tensor:
        """Potential."""
        return self.f_chi_a(self.time)[None, None, ...]

    @property
    def uv_g(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Geostrophic velocity."""
        u_g, v_g = self.gradp(self.psi_g)
        u_g /= self.space.dy
        v_g /= self.space.dx
        return (u_g, v_g)

    @property
    def uv_a(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Ageostrophic velocity."""
        u_a, v_a = self.compute_ageostrophic_velocity(self.time)
        return (u_a, v_a)

    @property
    def uv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Velocity."""
        u_g, v_g = self.uv_g
        u_a, v_a = self.uv_a
        return (u_g + u_a, v_g + v_a)

    @property
    def dt(self) -> float:
        """Timestep value."""
        try:
            return self._dt
        except AttributeError as e:
            raise UnsetTimestepError from e

    @dt.setter
    def dt(self, dt: float) -> None:
        if dt <= 0:
            msg = "Timestep must be greater than 0."
            raise InvalidModelParameterError(msg)
        self._dt = dt
        msg = f"{self.__class__.__name__}: dt set to {self.dt}"
        logger.info(msg)

    @property
    def wind_scaling(self) -> float:
        """Wind scaling."""
        try:
            return self._wind_scaling
        except AttributeError:
            self._wind_scaling = 1
            return self._wind_scaling

    @wind_scaling.setter
    def wind_scaling(self, value: float) -> None:
        self._wind_scaling = value

    @property
    def masks(self) -> Masks:
        """Masks."""
        try:
            return self._masks
        except AttributeError:
            self._masks = Masks.empty(
                self.space.nx,
                self.space.ny,
                device=self.specs["device"],
            )
            return self._masks

    @property
    def beta_plane(self) -> BetaPlane:
        """Beta Plane paramters."""
        return self._beta_plane

    @beta_plane.setter
    def beta_plane(self, beta_plane: BetaPlane) -> None:
        if not isinstance(beta_plane, BetaPlane):
            msg = "beta_plane should be of type `BetaPlane`."
            raise TypeError(msg)
        msg = f"{self.__class__.__name__}: Beta-plane set to {beta_plane}"
        logger.detail(msg)
        self._beta_plane = beta_plane
        self.f0 = beta_plane.f0
        self.beta = beta_plane.beta

    @property
    def y0(self) -> float:
        """Reference y0."""
        try:
            return self._y0
        except AttributeError:
            self._y0 = 0.5 * (
                self.space.q.xy.y.max() + self.space.q.xy.y.min()
            )
            return self._y0

    @y0.setter
    def y0(self, y0: float) -> None:
        """Set y0."""
        self._y0 = y0

    @property
    def beta_effect(self) -> torch.Tensor:
        """Beta effect."""
        y = self.space.q.xy.y[0, :].unsqueeze(0)
        return self.beta_plane.beta * (y - self.y0)

    @property
    def beta_effect_wide(self) -> torch.Tensor:
        """Beta effect."""
        y = self.wide_space.q.xy.y[0, :].unsqueeze(0)
        return self.beta_plane.beta * (y - self.y0)

    @property
    def f(self) -> torch.Tensor:
        """Coriolis."""
        return self.f0 + self.beta_effect

    @property
    def f_wide(self) -> torch.Tensor:
        """Coriolis."""
        return self.f0 + self.beta_effect_wide

    @property
    def flux_stencil(self) -> int:
        """Flux stencil."""
        return 5

    @property
    def time_stepper(self) -> str:
        """Time stepper."""
        return "rk3"

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        beta_plane: BetaPlane,
        compile_fns: bool = True,
    ) -> None:
        """Set up the surface vorticity model.

        Args:
            space_2d (SpaceDiscretization2D): Space.
            beta_plane (BetaPlane): Beta plane.
            compile_fns (bool): Whether to compile functions or not.
        """
        self.compile_fns = compile_fns
        self.__instance_nb = next(self._instance_count)
        self.name = f"{self.__class__.__name__}-{self.__instance_nb}"
        msg = f"Creating {self.__class__.__name__} model..."
        logger.info(msg)
        ## Space
        self.space = space_2d
        # Beta-plane
        with logger.section("Setting model parameters..."):
            self.beta_plane = beta_plane
        self.specs = specs.from_tensor(self.space.psi.xy.x)
        self.switch_to_homogeneous()
        self._set_utils(compile_fns)
        self.set_flux()
        self.reset_state()
        self.set_solver()
        self.set_wind_forcing(0, 0)

    def get_repr_parts(self) -> list[str]:
        """String representations parts.

        Returns:
            list[str]: String representation parts.
        """
        msg_parts = [
            f"Model: {self.__class__}",
            (
                f"├── Beta plane: f0 = {self.beta_plane.f0} "
                f"- β = {self.beta_plane.beta}"
            ),
        ]
        try:
            msg_parts.append(f"├── dt: {self.dt} s")
        except UnsetTimestepError:
            msg_parts.append("├── dt: unset yet")
        space_repr_ = self.space.get_repr_parts()
        space_repr = ["└── " + space_repr_.pop(0)]
        space_repr = space_repr + ["\t" + txt for txt in space_repr_]
        return msg_parts + space_repr

    def __repr__(self) -> str:
        """String representation."""
        return "\n".join(self.get_repr_parts())

    def _set_utils(self, compile_fns: bool) -> None:  # noqa: FBT001
        """Set utils.

        Args:
            compile_fns (bool): Whether to compile_fns or not.
        """
        if compile_fns:
            with logger.section("Compiling functions..."):
                self.grad = OptimizableFunction(grad)
                self.gradp = OptimizableFunction(grad_perp)
                self.interp = OptimizableFunction(interpolate)
                self.lap_h = OptimizableFunction(laplacian_h)
                self.lap = OptimizableFunction(laplacian)
        else:
            self.grad = grad
            self.gradp = grad_perp
            self.interp = interpolate
            self.lap_h = laplacian_h
            self.lap = laplacian

    def set_flux(self) -> None:
        """Set the fluxes utils."""
        if self.with_bc:
            return self.set_flux_inhomogeneous()
        return self.set_flux_homogeneous()

    def set_flux_homogeneous(self) -> None:
        """Set the flux.

        Raises:
            ValueError: If invalid stencil.
        """
        assert self.flux_stencil == 5
        if len(self.masks.psi_irrbound_xids) > 0:
            msg = (
                "Inhomogeneous pv reconstruction not "
                "implemented for non-regular geometry."
            )
            raise NotImplementedError(msg)
        div_flux = lambda q, u, v: div_flux_5pts(
            q,
            u,
            v,
            self.space.dx,
            self.space.dy,
        )

        with logger.section("Compiling function..."):
            self.div_flux = (
                OptimizableFunction(div_flux)
                if self.compute_zeta_g_from_psi_g
                else div_flux
            )

    def set_flux_inhomogeneous(self) -> None:
        """Set the flux.

        Raises:
            ValueError: If invalid stencil.
        """
        assert self.flux_stencil == 5
        if len(self.masks.psi_irrbound_xids) > 0:
            msg = (
                "Inhomogeneous pv reconstruction not "
                "implemented for non-regular geometry."
            )
            raise NotImplementedError(msg)
        if self.wide:
            div_flux = lambda q, u, v: div_flux_5pts_only(
                q,
                u,
                v,
                self.space.dx,
                self.space.dy,
            )
        else:
            div_flux = lambda q, u, v: div_flux_5pts_with_bc(
                q,
                u,
                v,
                self.space.dx,
                self.space.dy,
            )

        with logger.section("Compiling function..."):
            self.div_flux = (
                OptimizableFunction(div_flux) if self.compile_fns else div_flux
            )

    def set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # PV equation solver
        self.solver_homogeneous = HomogeneousPVInversion(
            torch.zeros((1, 1), **self.specs),
            self.f0,
            self.space.dx,
            self.space.dy,
            self.masks,
        )
        self.solver_inhomogeneous = InhomogeneousPVInversion(
            torch.zeros((1, 1), **self.specs),
            self.f0,
            self.space.dx,
            self.space.dy,
            self.masks,
        )
        if self.with_bc:
            sf_bc = self.sf_bc_interp(self.time.item())
            self.solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

    def set_wind_forcing(
        self,
        taux: torch.Tensor | float,
        tauy: torch.Tensor | float,
    ) -> None:
        """Set the wind forcing.

        Args:
            taux (torch.Tensor): Wind stress in the x direction.
                └── (n_ens, nl, nx, ny)-shaped
            tauy (torch.Tensor): Wind stress in the y direction.
                └── (n_ens, nl, nx, ny)-shaped
        """
        if isinstance(taux, float | int):
            self.taux = (
                torch.ones(
                    (self.space.nx, self.space.ny + 1),
                    **self.specs,
                )
                * taux
            )
        else:
            self.taux = taux
        if isinstance(tauy, float | int):
            self.tauy = (
                torch.ones(
                    (self.space.nx + 1, self.space.ny),
                    **self.specs,
                )
                * tauy
            )
        else:
            self.tauy = tauy
        curl_tau = (
            torch.diff(self.tauy, dim=-2) / self.space.dx
            - torch.diff(self.taux, dim=-1) / self.space.dy
        )
        self.curl_tau = curl_tau.unsqueeze(0).unsqueeze(0) / self.wind_scaling

    def reset_time(self) -> None:
        """Reset time."""
        self.time = torch.tensor(0, **self.specs)
        if self.with_bc:
            self.set_boundaries(self.time.item())

    def reset_state(self) -> None:
        """Set the state."""
        self.reset_time()
        psi_g = torch.zeros_like(self.space.psi.xy.x)[None, None, ...]
        zeta_g = self.compute_zeta_g_from_psi_g(psi_g)
        self.psiz = PSIZeta(psi_g, zeta_g)

    def switch_to_homogeneous(self) -> None:
        """Switch to an inhomogeneous solver."""
        self.with_bc = False

    def switch_to_inhomogeneous(self) -> None:
        """Switch to an inhomogeneous solver."""
        if self.with_bc:
            return
        self.with_bc = True
        self.wide_space = (
            self.space.widen(3, 3) if self.wide else self.space.widen(1, 1)
        )
        self.set_flux()

    def set_boundary_maps(
        self,
        sf_bc_interp: LinearInterpolation[Boundaries],
        zeta_g_bc_interp: LinearInterpolation[Boundaries],
    ) -> None:
        """Set the boundary maps.

        Args:
            sf_bc_interp (LinearInterpolation[Boundaries]): Boundary map
                for stream function at locations
                (imin,imax+1,jmin,jmax+1).
            zeta_g_bc_interp (LinearInterpolation[Boundaries]): Boundary map
                for vorticity at locations
                (imin,imax,jmin,jmax).
            wide_space (SpaceDiscretization2D): Wide space.
        """
        self.switch_to_inhomogeneous()
        self.basis_psi_a = self.basis_psi_a
        self.basis_chi_a = self.basis_chi_a
        self.sf_bc_interp = sf_bc_interp
        self.zeta_g_bc_interp = zeta_g_bc_interp
        self.set_boundaries(self.time.item())

    def set_boundaries(self, time: float) -> None:
        """Set the boundaries to match given time.

        Args:
            time (float): Time.
        """
        sf_bc = self.sf_bc_interp(time)

        self.solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))
        zeta_g_bc = self.zeta_g_bc_interp(time)
        if self.wide:
            if zeta_g_bc.width != 3:
                msg = "For wide boundary, zeta_g_bc must be 3 points wide."
                raise ValueError(msg)
            self.zeta_g_bc = zeta_g_bc
        else:
            self.zeta_g_bc = zeta_g_bc.get_band(0)

    def compute_zeta_g_from_psi_g(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute vorticity from streamfunction.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Vorticity.
                └── (n_ens, nl, nx, ny)-shaped
        """
        if self.with_bc:
            boundary = self.sf_bc_interp(self.time.item())
            psi_with_bc = boundary.get_band(1).expand(psi)
            self.p = psi_with_bc
            lap_psi = self.lap(psi_with_bc, self.space.dx, self.space.dy)
            return self.masks.h * self.interp(lap_psi)
        lap_psi = self.lap_h(psi, self.space.dx, self.space.dy)
        return self.masks.h * (
            self.interp(
                self.masks.psi * (lap_psi),
            )
        )

    def compute_ageostrophic_velocity_contribution_from_psi_a(
        self, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ageostrophic velocity contribution from ѱₐ.

        Args:
            time (torch.Tensor): Time.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ∇^⟂ѱₐ.
        """
        psi_a = self.f_psi_a(time)[None, None, ...]

        dx, dy = self.space.dx, self.space.dy

        u_sf_a, v_sf_a = self.gradp(psi_a)

        u_sf_a /= dy
        v_sf_a /= dx

        return u_sf_a, v_sf_a

    def compute_ageostrophic_velocity_contribution_from_chi_a(
        self, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ageostrophic velocity contribution from χₐ.

        Args:
            time (torch.Tensor): Time.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ∇^χₐ.
        """
        chi_a = self.f_chi_a(time)[None, None, ...]
        u_pot_a_, v_pot_a_ = self.grad(chi_a)
        dx, dy = self.space.dx, self.space.dy

        u_pot_a_ /= dx
        v_pot_a_ /= dy

        u_pot_a = crop(u_pot_a_, 0, 1)
        v_pot_a = crop(v_pot_a_, 1, 0)

        return u_pot_a, v_pot_a

    def compute_ageostrophic_velocity(
        self, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ageostrophic velocity.

        Args:
            time (torch.Tensor): Time.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: u ageostrophic, v ageostrophic.
        """
        u_sf_a, v_sf_a = (
            self.compute_ageostrophic_velocity_contribution_from_psi_a(time)
        )

        u_pot_a, v_pot_a = (
            self.compute_ageostrophic_velocity_contribution_from_chi_a(time)
        )

        return (u_sf_a + u_pot_a, v_sf_a + v_pot_a)

    def compute_advection_homogeneous(
        self, psiz: PSIZeta, time: torch.Tensor
    ) -> torch.Tensor:
        """Compute advection pv advection for homogeneous problem.

        Args:
            psiz (PSIZeta): Geostrophic stream function and vorticity.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped
            time (torch.Tensor): Time to evaluate parameterized fields.

        Returns:
            torch.Tensor: RHS: J(ѱ_g, ζ_g)
                └──  (n_ens, nl, nx, ny)-shaped
        """
        psi_g, zeta_g = psiz

        zeta_a = self.f_zeta_a(time)

        u_g, v_g = self.gradp(psi_g)
        u_g /= self.space.dy
        v_g /= self.space.dx

        u_a, v_a = self.compute_ageostrophic_velocity(time)

        u = u_g + u_a
        v = v_g + v_a

        return self.div_flux(
            zeta_g + zeta_a + self.f,
            u[..., 1:-1, :],
            v[..., 1:-1],
        )

    def compute_advection_inhomogeneous(
        self, psiz: PSIZeta, zeta_g_bc: Boundaries, time: torch.Tensor
    ) -> torch.Tensor:
        """Compute advection pv advection for inhomogeneous problem.

        Args:
            psiz (PSIZeta): Geostrophic stream function and vorticity.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped
            zeta_g_bc (Boundaries): Boundaries for geostrophic vorticity.
            time (torch.Tensor): Time to evaluate parameterized fields.

        Returns:
            torch.Tensor: RHS: J(ѱ_g, ζ_g)
                └──  (n_ens, nl, nx, ny)-shaped
        """
        psi_g, zeta_g = psiz
        u_g, v_g = self.gradp(psi_g)
        u_g /= self.space.dy
        v_g /= self.space.dx

        u_a, v_a = self.compute_ageostrophic_velocity(time)

        u = u_g + u_a
        v = v_g + v_a

        zeta_g_with_bc = zeta_g_bc.expand(zeta_g)
        zeta_a_with_bc = self.f_zeta_a_wide(time)
        return self.div_flux(
            zeta_g_with_bc + zeta_a_with_bc + self.f_wide, u, v
        )

    def compute_forcing(self, time: torch.Tensor) -> torch.Tensor:
        """Compute forcing.

        Args:
            zeta_g (torch.Tensor): Geostrophic vorticity.
            time (torch.Tensor): Time to evaluate parameterized fields.

        Returns:
            torch.Tensor: Forcing.
        """
        dt_zeta_a = self.f_zeta_a.dt(time)
        return -dt_zeta_a

    def compute_time_derivatives(self, prognostic: PSIZeta) -> PSIZeta:
        """Compute time derivatives.

        Args:
            prognostic (PSIZeta): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIZeta: dpsi_g, dzeta_g
                ├── dpsi_g : (n_ens, nl, nx+1, ny+1)-shaped
                └── dzeta_g : (n_ens, nl, nx, ny)-shaped
        """
        if self.with_bc:
            return self.compute_time_derivatives_inhomogeneous(prognostic)
        return self.compute_time_derivatives_homogeneous(prognostic)

    def compute_time_derivatives_homogeneous(
        self,
        prognostic: PSIZeta,
    ) -> PSIZeta:
        """Compute time derivatives for homogeneous problem.

        Args:
            prognostic (PSIZeta): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIZeta: dpsi_g, dzeta_g
                ├── dpsi_g : (n_ens, nl, nx+1, ny+1)-shaped
                └── dzeta_g : (n_ens, nl, nx, ny)-shaped
        """
        psi_g, zeta_g = prognostic
        div_flux = self.compute_advection_homogeneous(
            PSIZeta(psi_g, zeta_g), self._substep_time
        )
        # wind forcing + bottom drag
        forcing = self.compute_forcing(self._substep_time)
        dzeta_g = (-div_flux + self.curl_tau + forcing) * self.masks.h
        dzeta_g_i = self.interp(dzeta_g)
        # Solve Helmholtz equation
        dpsi_g = self.solver_homogeneous.compute_stream_function(
            dzeta_g_i,
            ensure_mass_conservation=True,
        )
        return PSIZeta(dpsi_g, dzeta_g)

    def compute_time_derivatives_inhomogeneous(
        self,
        prognostic: PSIZeta,
    ) -> PSIZeta:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIZeta): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIZeta: dpsi_g, dzeta_g
                ├── dpsi_g : (n_ens, nl, nx+1, ny+1)-shaped
                └── dzeta_g : (n_ens, nl, nx, ny)-shaped
        """
        psi_g_i, zeta_g_i = prognostic
        psi_g_bc, zeta_g_bc = self.solver_inhomogeneous.psiq_bc
        psi_g = psi_g_i + psi_g_bc
        zeta_g = zeta_g_i + zeta_g_bc
        div_flux = self.compute_advection_inhomogeneous(
            PSIZeta(psi_g, zeta_g), self.zeta_g_bc, self._substep_time
        )
        # wind forcing + bottom drag
        forcing = self.compute_forcing(self._substep_time)
        dzeta_g = (-div_flux + self.curl_tau + forcing) * self.masks.h
        dzeta_g_i = self.interp(dzeta_g)
        # Solve Helmholtz equation
        dpsi_g = self.solver_homogeneous.compute_stream_function(
            dzeta_g_i,
            ensure_mass_conservation=False,
        )
        if self.time_stepper == "rk3":
            # Boundary condition interpolation
            self.set_boundaries(self._substep_time.item())
            self._rk3_step += 1
        return PSIZeta(dpsi_g, dzeta_g)

    def update(self, prognostic: PSIZeta) -> PSIZeta:
        """Update prognostic.

        Args:
            prognostic (PSIZeta): Prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIZeta: Updated prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped
        """
        if self.with_bc:
            return self.update_inhomogeneous(prognostic)
        return self.update_homogeneous(prognostic)

    def timestep(self, prognostic: PSIZeta) -> PSIZeta:
        """Timesteps prognostic variables.

        Args:
            prognostic (PSIZeta): Prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIZeta: Timesteped prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped
        """
        assert self.time_stepper == "rk3"
        self._rk3_step = 0
        psiz = time_steppers.rk3_ssp(
            prog=prognostic,
            dt=self.dt,
            time_derivation_func=self.compute_time_derivatives,
        )
        self.time += self.dt
        return psiz

    def update_homogeneous(self, prognostic: PSIZeta) -> PSIZeta:
        """Update prognostic tuple.

        Args:
            prognostic (PSIZeta): Prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIZeta: Updated prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped
        """
        return self.timestep(prognostic)

    def update_inhomogeneous(self, prognostic: PSIZeta) -> PSIZeta:
        """Update prognostic tuple.

        Args:
            prognostic (PSIZeta): Prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIZeta: Updated prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── zeta : (n_ens, nl, nx, ny)-shaped
        """
        psi_bc = self.solver_inhomogeneous.psiq_bc.psi
        prognostic_i = PSIZeta(prognostic.psi - psi_bc, prognostic.zeta)
        psiz_g_i = self.timestep(prognostic_i)
        self.set_boundaries(self.time.item())
        psi_bc = self.solver_inhomogeneous.psiq_bc.psi
        return PSIZeta(psiz_g_i.psi + psi_bc, psiz_g_i.zeta)

    @torch.enable_grad()
    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self.psiz = self.update(self.psiz)
