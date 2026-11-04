"""Usual QG Model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw import specs
from qgsw.exceptions import (
    InvalidLayerNumberError,
    InvalidModelParameterError,
    UnsetTimestepError,
)
from qgsw.fields.variables.tuples import PSIQSST
from qgsw.logging import getLogger
from qgsw.masks import Masks
from qgsw.models.base import ABCCounter
from qgsw.models.core import time_steppers
from qgsw.models.core.flux import (
    div_flux_3pts,
    div_flux_3pts_mask,
    div_flux_5pts,
    div_flux_5pts_mask,
    div_flux_5pts_only,
    div_flux_5pts_with_bc,
)
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.names import ModelName
from qgsw.models.qg.stretching_matrix import (
    compute_A,
)
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import grad, grad_perp, laplacian, laplacian_h
from qgsw.solver.pv_inversion import (
    BasePVInversion,
    HomogeneousPVInversion,
    InhomogeneousPVInversion,
)
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import DEVICE
from qgsw.utils.interpolation import ConstantInterpolation, _Interpolation
from qgsw.utils.named_object import NamedObject

if TYPE_CHECKING:
    from qgsw.spatial.core.discretization import SpaceDiscretization2D


logger = getLogger(__name__)


class QGPSIQ(NamedObject[ModelName], metaclass=ABCCounter):
    """Finite volume multi-layer QG solver."""

    _name = ModelName.QUASI_GEOSTROPHIC

    _flux_stencil = 5
    _time_stepper = "rk3"

    n_ens = 1
    wide = True
    implements_alpha = False
    advect_sst = False

    @property
    def _substep_time(self) -> torch.Tensor:
        if self.time_stepper == "euler":
            return self.time
        if self.time_stepper == "rk3":
            coef = time_steppers.compute_rk3_substep_coefs(self._rk3_step)
            return self.time + coef * self.dt
        msg = "Supported time steppers are 'rk3' and 'euler'."
        raise ValueError(msg)

    @property
    def y0(self) -> float:
        """Reference y0."""
        try:
            return self._y0
        except AttributeError:
            self._y0 = 0.5 * (
                self.space.q.xyh.y.max() + self.space.q.xyh.y.min()
            )
            return self._y0

    @y0.setter
    def y0(self, y0: float) -> None:
        """Set y0."""
        self._y0 = y0

    @property
    def beta_effect(self) -> torch.Tensor:
        """Beta effect."""
        y = self.space.q.xyh.y[0, :].unsqueeze(0)
        return self.beta_plane.beta * (y - self.y0)

    @property
    def slip_coef(self) -> float:
        """Slip coefficient."""
        return self._slip_coef

    @slip_coef.setter
    def slip_coef(self, slip_coefficient: float) -> None:
        # Verify value in [0,1]
        if (slip_coefficient < 0) or (slip_coefficient > 1):
            msg = f"slip coefficient must be in [0, 1], got {slip_coefficient}"
            raise InvalidModelParameterError(msg)
        self._slip_coef = slip_coefficient
        msg = f"{self.name}: Slip coefficient set to {slip_coefficient}"
        logger.detail(msg)

    @property
    def bottom_drag_coef(self) -> float:
        """Bottom drag coefficient."""
        return self._bottom_drag

    @bottom_drag_coef.setter
    def bottom_drag_coef(self, bottom_drag: float) -> None:
        self._bottom_drag = bottom_drag
        msg = f"{self.name}: Bottom drag set to {bottom_drag}"
        logger.detail(msg)

    @property
    def wind_scaling(self) -> float:
        """Wind scaling."""
        try:
            return self._wind_scaling
        except AttributeError:
            self._wind_scaling = self.H[:1, 0, 0]
            return self._wind_scaling

    @wind_scaling.setter
    def wind_scaling(self, wind_scaling: torch.Tensor) -> None:
        self._wind_scaling = wind_scaling

    @property
    def flux_stencil(self) -> int:
        """Flux stencil size."""
        return self._flux_stencil

    @flux_stencil.setter
    def flux_stencil(self, stencil: int) -> None:
        stencil = int(stencil)
        if stencil not in (3, 5):
            msg = "The stencil can only be 3 or 5."
            raise ValueError(msg)
        self._flux_stencil = stencil
        self.set_flux()

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
        msg = f"{self.name}: Beta-plane set to {beta_plane}"
        logger.detail(msg)
        self._beta_plane = beta_plane
        self.f0 = beta_plane.f0
        self.beta = beta_plane.beta

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
        msg = f"{self.name}: dt set to {self.dt}"
        logger.info(msg)

    @property
    def psi(self) -> torch.Tensor:
        """StatePSIQ Variable psi: Stream function.

        └── (n_ens, nl, nx+1,ny+1)-shaped.
        """
        return self.psiqsst.psi

    @property
    def q(self) -> torch.Tensor:
        """StatePSIQ Variable q: Potential Vorticity.

        └── (n_ens, nl, nx,ny)-shaped.
        """
        return self.psiqsst.q

    @property
    def q_anom(self) -> torch.Tensor:
        """Potential Vorticity anomaly.

        └── (n_ens, nl, nx,ny)-shaped.
        """
        return self.q - self.beta_effect

    @property
    def vorticity(self) -> torch.Tensor:
        """Vorticity.

        └── (n_ens, nl, nx,ny)-shaped.
        """
        return self.compute_vort_from_psi(self.psi)

    @property
    def uv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Geostrophic velocity."""
        u, v = self.gradp(self.psi)
        u /= self.space.dy
        v /= self.space.dx
        return (u, v)

    @property
    def sst_anom(self) -> torch.Tensor:
        """SST anomaly."""
        return self.psiqsst.sst

    @property
    def time_stepper(self) -> str:
        """Time stepper."""
        return self._time_stepper

    @time_stepper.setter
    def time_stepper(self, time_stepper: str) -> None:
        """Time stepper setter."""
        time_steppers.validate(time_stepper)
        self._time_stepper = time_stepper
        msg = "Time stepper set to: " + time_stepper
        logger.info(msg)

    @property
    def with_bc(self) -> bool:
        """Whether an inhomogeneous solver is used or not."""
        return self._with_bc

    @property
    def solver(self) -> BasePVInversion:
        """Solver for PVInversion."""
        return (
            self.solver_inhomogeneous
            if self.with_bc
            else self.solver_homogeneous
        )

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,
        beta_plane: BetaPlane,
        g_prime: torch.Tensor,
        compile_fns=True,  # noqa: ANN001
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
        self.compile_fns = compile_fns
        self.__instance_nb = next(self._instance_count)
        self.class_name = self.__class__.__name__
        self.name = f"{self.class_name}-{self.__instance_nb}"
        msg = f"Creating {self.name}..."
        logger.info(msg)
        ## Space
        self.space = space_2d.add_h_coords(H)
        self.specs = specs.from_tensor(self.space.psi.xyh.x)
        # grid params
        self.y0 = 0.5 * (self.space.q.xyh.y.max() + self.space.q.xyh.y.min())
        # Beta-plane
        with logger.section("Setting model parameters..."):
            self.beta_plane = beta_plane
            self.H = H[:, None, None]
            self.g_prime = g_prime[:, None, None]
            msg = f"{self.name}: g' set to {g_prime.squeeze().tolist()}"
            logger.detail(msg)
            msg = f"{self.name}: H set to {H.squeeze().tolist()}"
            logger.detail(msg)
            self.compute_auxillary_matrices()
        self.switch_to_homogeneous()
        self._set_utils(compile_fns)
        self.set_flux()
        self.reset_state()
        self.set_solver()
        self.set_wind_forcing(0.0, 0.0)

        # auxillary matrices for elliptic equation
        self.compute_auxillary_matrices()

        self.zeros_inside = (
            torch.zeros(
                (self.n_ens, self.space.nl - 2, self.space.nx, self.space.ny),
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            if (self.space.nl - 2) > 0
            else None
        )

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
                lap_h = lambda t: laplacian_h(t, self.space.dx, self.space.dy)
                self.lap_h = OptimizableFunction(lap_h)
                lap = lambda t: laplacian(t, self.space.dx, self.space.dy)
                self.lap = OptimizableFunction(lap)
        else:
            self.grad = grad
            self.gradp = grad_perp
            self.interp = interpolate
            self.lap_h = lap_h
            self.lap = lap

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
        if self.flux_stencil == 5:
            if len(self.masks.psi_irrbound_xids) > 0:
                div_flux = lambda q, u, v: div_flux_5pts_mask(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                    self.masks.u_distbound1[..., 1:-1, :],
                    self.masks.u_distbound2[..., 1:-1, :],
                    self.masks.u_distbound3plus[..., 1:-1, :],
                    self.masks.v_distbound1[..., 1:-1],
                    self.masks.v_distbound2[..., 1:-1],
                    self.masks.v_distbound3plus[..., 1:-1],
                )
            else:
                div_flux = lambda q, u, v: div_flux_5pts(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                )
        elif self.flux_stencil == 3:
            if len(self.masks.psi_irrbound_xids) > 0:
                div_flux = lambda q, u, v: div_flux_3pts_mask(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                    self.masks.u_distbound1[..., 1:-1, :],
                    self.masks.u_distbound2plus[..., 1:-1, :],
                    self.masks.v_distbound1[..., 1:-1],
                    self.masks.v_distbound2plus[..., 1:-1],
                )
            else:
                div_flux = lambda q, u, v: div_flux_3pts(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                )
        else:
            msg = f"Invalid stencil value: {self.flux_stencil}"
            raise ValueError(msg)

        with logger.section("Compiling function..."):
            self.div_flux = (
                OptimizableFunction(div_flux) if self.compile_fns else div_flux
            )

    def set_flux_inhomogeneous(self) -> None:
        """Set the flux.

        Raises:
            ValueError: If invalid stencil.
        """
        if self.flux_stencil == 5:
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
        elif self.flux_stencil == 3:
            msg = (
                "Inhomogeneous pv reconstruction not "
                "implemented for 3 pts stencil."
            )
            raise NotImplementedError(msg)
        else:
            msg = f"Invalid stencil value: {self.flux_stencil}"
            raise ValueError(msg)

        with logger.section("Compiling function..."):
            self.div_flux = (
                OptimizableFunction(div_flux) if self.compile_fns else div_flux
            )

    def set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # PV equation solver
        self.solver_homogeneous = HomogeneousPVInversion(
            self.A,
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        self.solver_inhomogeneous = InhomogeneousPVInversion(
            self.A,
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        if self._with_bc:
            sf_bc = self.sf_bc_interp(self.time.item())
            self.solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

    def reset_time(self) -> None:
        """Reset time."""
        self.time = torch.tensor(0, **self.specs)
        if self.with_bc:
            self.set_boundaries(self.time.item())

    def reset_state(self) -> None:
        """Set the state."""
        self.reset_time()
        psi_g = torch.zeros_like(self.space.psi.xyh.x)[None, ...]
        zeta_g = self.compute_q_from_psi(psi_g)
        sst_anom = torch.zeros_like(zeta_g)
        self.psiqsst = PSIQSST(psi_g, zeta_g, sst_anom)
        self.sst_mean = torch.tensor(0, **self.specs)

    def set_psi(self, psi: torch.Tensor) -> None:
        """Set the value of stream function.

        WARNING: with inhomogeneous boundary condition this might introduce
        errors in q due to different beta effects.
        You should use the `set_psiq` method instead.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped
        """
        q = self.compute_q_from_psi(psi)
        self.set_psiq(psi, q)

    def set_psiq(self, psi: torch.Tensor, q: torch.Tensor) -> None:
        """Set psi and q."""
        self.psiqsst = PSIQSST(psi, q, self.sst_anom)

    def set_sst(self, sst: torch.Tensor) -> None:
        """Set sst."""
        if not self.advect_sst:
            msg = f"{self.get_type().value} will advect SST."
            logger.info(msg)
        self.advect_sst = True
        self.sst_mean = torch.mean(sst)
        sst_anom = sst - self.sst_mean
        self.psiqsst = PSIQSST(self.psi, self.q, sst_anom)

    def set_psiqsst(
        self, psi: torch.Tensor, q: torch.Tensor, sst: torch.Tensor
    ) -> None:
        """Set psi, q and sst."""
        self.set_psiq(psi, q)
        self.set_sst(sst)

    def switch_to_homogeneous(self) -> None:
        """Switch to an inhomogeneous solver."""
        self._with_bc = False

    def switch_to_inhomogeneous(self) -> None:
        """Switch to an inhomogeneous solver."""
        if self.with_bc:
            return
        self._with_bc = True
        self.set_flux()

    def set_boundary_maps(
        self,
        sf_bc_interp: _Interpolation[Boundaries],
        q_bc_interp: _Interpolation[Boundaries],
        sst_bc_interp: _Interpolation[Boundaries] | None = None,
    ) -> None:
        """Set the boundary maps.

        Args:
            sf_bc_interp (_Interpolation[Boundaries]): Boundary map
                for stream function at locations
                (imin,imax+1,jmin,jmax+1).
            q_bc_interp (_Interpolation[Boundaries]): Boundary map
                for vorticity at locations
                (imin,imax,jmin,jmax).
            sst_bc_interp (_Interpolation[Boundaries]): Boundary map
                for SST at locations
                (imin,imax,jmin,jmax).
        """
        self.switch_to_inhomogeneous()
        if sst_bc_interp is None:
            sst_bc_interp = ConstantInterpolation(
                Boundaries.zeros_like(q_bc_interp.ys[0])
            )
        self.sf_bc_interp = sf_bc_interp
        self.q_bc_interp = q_bc_interp
        self.sst_bc_interp = sst_bc_interp
        self.set_boundaries(self.time.item())

    def set_boundaries(self, time: float) -> None:
        """Set the boundaries to match given time.

        Args:
            time (float): Time.
        """
        sf_bc = self.sf_bc_interp(time)

        self.solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))
        q_bc = self.q_bc_interp(time)
        sst_anom_bc: Boundaries = self.sst_bc_interp(time) - self.sst_mean
        if self.wide:
            if q_bc.width != 3:
                msg = "For wide boundary, q_bc must be 3 points wide."
                raise ValueError(msg)
            self.q_bc = q_bc
            self.sst_anom_bc = sst_anom_bc
        else:
            self.q_bc = q_bc.get_band(0)
            self.sst_anom_bc = sst_anom_bc.get_band(0)

    def compute_auxillary_matrices(self) -> None:
        """Compute auxiliary matrix."""
        # A operator
        self.A = compute_A(
            self.H[:, 0, 0],
            self.g_prime[:, 0, 0],
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    def compute_q_from_psi(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute stream function from stream function.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Potential vorticity.
                └── (n_ens, nl, nx, ny)-shaped
        """
        return (
            self.compute_q_anom_from_psi(psi) + self.masks.h * self.beta_effect
        )

    def compute_q_anom_from_psi(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute potential vorticity anomaly from stream function.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Potential vorticity anomaly.
                └── (n_ens, nl, nx, ny)-shaped
        """
        vort = self.compute_vort_from_psi(psi)
        stretching = self.beta_plane.f0**2 * torch.einsum(
            "lm,...mxy->...lxy",
            self.A,
            psi,
        )
        if self.with_bc:
            return vort - self.masks.h * self.interp(stretching)
        return vort - self.masks.h * self.interp(self.masks.psi * stretching)

    def compute_vort_from_psi(self, psi: torch.Tensor) -> torch.Tensor:
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
            lap_psi = self.lap(psi_with_bc)
            return self.masks.h * self.interp(lap_psi)
        lap_psi = self.lap_h(psi)
        return self.masks.h * (
            self.interp(
                self.masks.psi * (lap_psi),
            )
        )

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

    def compute_div_flux_homogeneous(
        self, u: torch.Tensor, v: torch.Tensor, tracer: torch.Tensor
    ) -> torch.Tensor:
        """Compute homogeneous flux divergence.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meridional velocity.
            tracer (torch.Tensor): Tracer.

        Returns:
            torch.Tensor: ∇·(u tracer)
        """
        return self.div_flux(
            tracer,
            u[..., 1:-1, :],
            v[..., 1:-1],
        )

    def compute_div_flux_inhomogeneous(
        self, u: torch.Tensor, v: torch.Tensor, tracer: torch.Tensor
    ) -> torch.Tensor:
        """Compute inhomogeneous flux divergence.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meridional velocity.
            tracer (torch.Tensor): Tracer with boundaries.

        Returns:
            torch.Tensor: ∇·(u tracer)
        """
        return self.div_flux(tracer, u, v)

    def compute_dsst_homogeneous(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        sst_anom: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dsst for homogeneous situations.

        Args:
            u (torch.Tensor): Zonal velocity.
                └── sst: (n_ens, nl, nx+1, ny)-shaped
            v (torch.Tensor): Meridional velocity.
                └── sst: (n_ens, nl, nx, ny+1)-shaped
            sst_anom (torch.Tensor): SST anomaly.
                └── sst: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: dsst.
                └── dsst: (n_ens, nl, nx, ny)-shaped
        """
        if self.advect_sst:
            return (
                -self.compute_div_flux_homogeneous(u, v, sst_anom)
                * self.masks.h
            )
        return torch.zeros_like(sst_anom)

    def compute_dsst_inhomogeneous(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        sst_anom: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dsst for inhomogeneous situations.

        Args:
            u (torch.Tensor): Zonal velocity.
                └── sst: (n_ens, nl, nx+1, ny)-shaped
            v (torch.Tensor): Meridional velocity.
                └── sst: (n_ens, nl, nx, ny+1)-shaped
            sst_anom (torch.Tensor): SST anomaly.
                └── sst: (n_ens, nl, nx+1, ny+1)-shaped (+3 if wide)

        Returns:
            torch.Tensor: dsst.
                └── dsst: (n_ens, nl, nx, ny)-shaped
        """
        if self.advect_sst:
            sst_anom_wide = self.sst_anom_bc.expand(sst_anom)
            return (
                -self.compute_div_flux_inhomogeneous(u, v, sst_anom_wide)
                * self.masks.h
            )
        return torch.zeros_like(sst_anom)

    def compute_drag_homogeneous(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute wind and bottom drag contribution.

        Args:
            psi (torch.Tensor): Stream function.
                └──  psi: (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Wind and bottom drag.
                └──  (n_ens, nl, nx, ny)-shaped
        """
        omega = self.interp(self.lap_h(psi) * self.masks.psi)
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.space.nl == 1:
            fcg_drag = self.curl_tau + bottom_drag
        elif self.space.nl == 2:
            fcg_drag = torch.cat([self.curl_tau, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [
                    self.curl_tau,
                    self.zeros_inside,
                    bottom_drag,
                ],
                dim=-3,
            )
        return fcg_drag

    def compute_drag_inhomogeneous(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute wind and bottom drag contribution.

        Args:
            psi (torch.Tensor): Stream function.
                └──  psi: (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Wind and bottom drag.
                └──  (n_ens, nl, nx, ny)-shaped
        """
        sf_boundary = self.sf_bc_interp(self.time.item())
        sf_wide = sf_boundary.expand(psi[..., 1:-1, 1:-1])
        omega = self.interp(self.lap(sf_wide))
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.space.nl == 1:
            fcg_drag = self.curl_tau + bottom_drag
        elif self.space.nl == 2:
            fcg_drag = torch.cat([self.curl_tau, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [
                    self.curl_tau,
                    self.zeros_inside,
                    bottom_drag,
                ],
                dim=-3,
            )
        return fcg_drag

    def compute_time_derivatives(self, prognostic: PSIQSST) -> PSIQSST:
        """Compute time derivatives.

        Args:
            prognostic (PSIQSST): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  q : (n_ens, nl, nx, ny)-shaped
                └── sst: (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  dq : (n_ens, nl, nx, ny)-shaped
                └── dsst: (n_ens, nl, nx, ny)-shaped
        """
        if self.with_bc:
            dprog = self.compute_time_derivatives_inhomogeneous(prognostic)
        else:
            dprog = self.compute_time_derivatives_homogeneous(prognostic)
        if self.time_stepper != "rk3" or self._rk3_step == 0:
            self.dprog = dprog
        elif self.time_stepper == "rk3":
            self._rk3_step += 1
        return dprog

    def compute_time_derivatives_homogeneous(
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        """Compute time derivatives for homogeneous problem.

        Args:
            prognostic (PSIQSST): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  q : (n_ens, nl, nx, ny)-shaped
                └── sst: (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  dq : (n_ens, nl, nx, ny)-shaped
                └── dsst: (n_ens, nl, nx, ny)-shaped
        """
        psi, q, sst_anom = prognostic

        u, v = self.gradp(psi)
        u /= self.space.dy
        v /= self.space.dx

        div_flux = self.compute_div_flux_homogeneous(u, v, q)
        # wind forcing + bottom drag
        fcg_drag = self.compute_drag_homogeneous(psi)
        dq = (-div_flux + fcg_drag) * self.masks.h
        self.dq = dq
        dq_i = self.interp(dq)
        # Solve Helmholtz equation
        dpsi = self.solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )
        dsst_anom = self.compute_dsst_homogeneous(u, v, sst_anom)
        return PSIQSST(dpsi, dq, dsst_anom)

    def compute_time_derivatives_inhomogeneous(
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQSST): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  q : (n_ens, nl, nx, ny)-shaped
                └── sst: (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  dq : (n_ens, nl, nx, ny)-shaped
                └── dsst: (n_ens, nl, nx, ny)-shaped
        """
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
        dq = (-div_flux + fcg_drag) * self.masks.h
        dq_i = self.interp(dq)
        # Solve Helmholtz equation
        dpsi = self.solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )
        dsst_anom = self.compute_dsst_inhomogeneous(u, v, sst_anom)
        return PSIQSST(dpsi, dq, dsst_anom)

    def set_p(self, p: torch.Tensor) -> None:
        """Set the initial pressure.

        The pressure must contain at least as many layers as the model.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, >= nl, nx+1, ny+1)-shaped

        Raises:
            InvalidLayerNumberError: If the layer number of p is invalid.
        """
        if p.shape[1] < (nl := self.space.nl):
            msg = f"p must have at least {nl} layers."
            raise InvalidLayerNumberError(msg)

        return self.set_psi(p[:, :nl] / self.beta_plane.f0)

    def update(self, prognostic: PSIQSST) -> PSIQSST:
        """Update prognostic.

        Args:
            prognostic (PSIQSST): Prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  q : (n_ens, nl, nx, ny)-shaped
                └── sst: (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: Updated prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  q : (n_ens, nl, nx, ny)-shaped
                └── sst: (n_ens, nl, nx, ny)-shaped
        """
        if self.with_bc:
            return self._update_inhomogeneous(prognostic)
        return self._update_homogeneous(prognostic)

    def _timestep(self, prognostic: PSIQSST) -> PSIQSST:
        if self.time_stepper == "rk3":
            self._rk3_step = 0
            psiqsst = time_steppers.rk3_ssp(
                prog=prognostic,
                dt=self.dt,
                time_derivation_func=self.compute_time_derivatives,
            )
        elif self.time_stepper == "euler":
            psiqsst = time_steppers.euler(
                prog=prognostic,
                dt=self.dt,
                time_derivation_func=self.compute_time_derivatives,
            )
        else:
            msg = f"Invalid time stepper: {self.time_stepper}"
            raise ValueError(msg)
        self.time += self.dt
        return psiqsst

    def _update_homogeneous(self, prognostic: PSIQSST) -> PSIQSST:
        """Update prognostic tuple.

        Args:
            prognostic (PSIQSST): Prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  q : (n_ens, nl, nx, ny)-shaped
                └── sst: (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: Updated prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  q : (n_ens, nl, nx, ny)-shaped
                └── sst: (n_ens, nl, nx, ny)-shaped
        """
        prognostic_i = prognostic
        return self._timestep(prognostic_i)

    def _update_inhomogeneous(self, prognostic: PSIQSST) -> PSIQSST:
        """Update prognostic tuple.

        Args:
            prognostic (PSIQSST): Prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  q : (n_ens, nl, nx, ny)-shaped
                └── sst: (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: Updated prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├──  q : (n_ens, nl, nx, ny)-shaped
                └── sst: (n_ens, nl, nx, ny)-shaped
        """
        psi_bc = self.solver_inhomogeneous.psiq_bc.psi
        prognostic_i = PSIQSST(
            prognostic.psi - psi_bc, prognostic.q, prognostic.sst
        )
        psiqsst_i = self._timestep(prognostic_i)
        self.set_boundaries(self.time.item())
        psi_bc = self.solver_inhomogeneous.psiq_bc.psi
        return PSIQSST(psiqsst_i.psi + psi_bc, psiqsst_i.q, psiqsst_i.sst)

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self.psiqsst = self.update(self.psiqsst)
