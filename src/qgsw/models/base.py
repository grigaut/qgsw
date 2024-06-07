"""Base models class."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import verbose
from qgsw.masks import Masks
from qgsw.models.core import finite_diff, flux, reconstruction
from qgsw.models.exceptions import (
    IncoherentWithMaskError,
    InvalidModelParameterError,
    InvalidSavingFileError,
)
from qgsw.spatial.core import grid_conversion
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from pathlib import Path

    from qgsw.physics import coriolis
    from qgsw.spatial.core.discretization import SpaceDiscretization3D


def reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pytorch cumsum in the reverse order.

    Example:
    reverse_cumsum(torch.arange(1,4), dim=-1)
    >>> tensor([6, 5, 3])

    Args:
        x (torch.Tensor): Tensor.
        dim (int): Dimension to perform reverse cumsum on.

    Returns:
        torch.Tensor: Result
    """
    return x + torch.sum(x, dim=dim, keepdim=True) - torch.cumsum(x, dim=dim)


class Model(metaclass=ABCMeta):
    """Base class for models.

    Following https://doi.org/10.1029/2021MS002663 .

    Physical Variables are :
        - u_phys: Zonal velocity
        - v_phys: Meridional Velocity
        - h_phys: layers thickness

    Prognostic Variables are linked to physical variables through:
        - u = u_phys x dx
        - v = v_phys x dy
        - h = h_phys x dx x dy

    Diagnostic variables are:
        - U = u_phys / dx
        - V = v_phys / dx
        - omega = omega_phys x dx x dy    (rel. vorticity)
        - eta = eta_phys                  (interface height)
        - p = p_phys                      (hydrostratic pressure)
        - k_energy = k_energy_phys        (kinetic energy)

    References variables are denoted with the subscript _ref:
        - h_ref
        - eta_ref
        - p_ref
        - h_ref_ugrid
        - h_ref_vgrid
        - dx_p_ref
        - dy_p_ref
    """

    dtype = torch.float64
    device = DEVICE
    _dt: float | None = None
    _slip_coef = 0.0
    _bottom_drag = 0.0
    _taux: torch.Tensor | None = None
    _tauy: torch.Tensor | None = None
    _n_ens: int = 1
    _masks: Masks | None = None

    omega: torch.Tensor
    eta: torch.Tensor
    U: torch.Tensor
    V: torch.Tensor
    V_m: torch.Tensor
    U_m: torch.Tensor
    h_tot_ugrid: torch.Tensor
    h_tot_vgrid: torch.Tensor

    def __init__(self, param: dict[str, Any]) -> None:
        """Parameters

        param: python dict. with following keys
            'space':    SpaceDiscretization3D, space discretization
            'g_prime':  Tensor (nl,), reduced gravities
            'beta_plane': NamedTuple Representing Beta plane.
            'n_ens':    int, number of ensemble member
        """
        verbose.display(
            msg=f"Creating {self.__class__.__name__} model...",
            trigger_level=1,
        )

        # Set up
        ## Space
        self._space: SpaceDiscretization3D = param["space"]
        # h
        self._set_H(self.space.h.xyh.h)
        ## gravity
        self._set_g_prime(param["g_prime"])
        # Number of ensemble
        self.n_ens = param.get("n_ens", self._n_ens)
        ## data device and dtype
        verbose.display(
            msg=f"dtype: {self.dtype}.",
            trigger_level=2,
        )
        verbose.display(
            msg=f"device: {self.device}",
            trigger_level=2,
        )
        ## Coriolis
        self._set_coriolis(param["beta_plane"])
        ## Topography and Ref values
        self._set_ref_variables()

        # initialize variables
        self._initialize_vars()

        self.comp_ke = finite_diff.comp_ke
        self.cell_corners_to_cell_centers = grid_conversion.omega_to_h
        self.compute_diagnostic_variables()

        # utils and flux computation functions
        self._set_utils_before_compilation()
        # precompile torch functions
        if param.get("compile", True):
            self._set_utils_with_compilation()
        else:
            verbose.display(msg="No compilation", trigger_level=2)

    @property
    def u(self) -> torch.Tensor:
        """State Variable u: Zonal Speed."""
        return self._u

    @property
    def v(self) -> torch.Tensor:
        """State Variable v: Meridional Speed."""
        return self._v

    @property
    def h(self) -> torch.Tensor:
        """State Variable h: Layers Thickness."""
        return self._h

    @property
    def dt(self) -> float:
        """Timestep value."""
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        if dt <= 0:
            msg = "Timestep must be greater than 0."
            raise InvalidModelParameterError(msg)
        verbose.display(msg=f"dt value set to {dt}.", trigger_level=1)
        self._dt = dt

    @property
    def space(self) -> SpaceDiscretization3D:
        """3D Space Discretization."""
        return self._space

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
        cl_type = (
            "Free-"
            if slip_coefficient == 1
            else ("No-" if slip_coefficient == 0 else "Partial free-")
        )
        verbose.display(
            msg=f"{cl_type}slip boundary condition",
            trigger_level=2,
        )
        self._slip_coef = slip_coefficient

    @property
    def bottom_drag_coef(self) -> float:
        """Bottom drag coefficient."""
        return self._bottom_drag

    @bottom_drag_coef.setter
    def bottom_drag_coef(self, bottom_drag: float) -> None:
        self._bottom_drag = bottom_drag

    @property
    def n_ens(self) -> int:
        """Number of ensembles."""
        return self._n_ens

    @n_ens.setter
    def n_ens(self, n_ens: int) -> None:
        if not isinstance(n_ens, int):
            msg = "n_ens must be an integer."
            raise InvalidModelParameterError(msg)
        if n_ens <= 0:
            msg = "n_ens must be greater than 1."
            raise InvalidModelParameterError(msg)
        self._n_ens = n_ens

    @property
    def masks(self) -> Masks:
        """Masks."""
        if self._masks is None:
            mask = torch.ones(
                self.space.nx,
                self.space.ny,
                dtype=self.dtype,
                device=self.device,
            )
            self._masks = Masks(mask)
        return self._masks

    @masks.setter
    def masks(self, mask: torch.Tensor) -> None:
        shape = mask.shape[0], mask.shape[1]
        # Verify shape
        if shape != (self.space.nx, self.space.ny):
            msg = (
                "Invalid mask shape "
                f"{shape}!=({self.space.nx},{self.space.ny})"
            )
            raise InvalidModelParameterError(msg)
        vals = torch.unique(mask).tolist()
        # Verify mask values
        if not all(v in [0, 1] for v in vals) or vals == [0]:
            msg = f"Invalid mask with non-binary values : {vals}"
            raise InvalidModelParameterError(msg)
        verbose.display(
            msg=(
                f"{'Non-trivial' if len(vals)==2 else 'Trivial'}"  # noqa: PLR2004
                " mask provided"
            ),
            trigger_level=2,
        )
        self._masks = Masks(mask)

    @property
    def beta_plane(self) -> coriolis.BetaPlane:
        """Beta Plane parmaters."""
        return self._beta_plane

    @property
    def H(self) -> torch.Tensor:  # noqa: N802
        """Layers thickness."""
        return self._H

    @property
    def g_prime(self) -> torch.Tensor:
        """Reduced Gravity."""
        return self._g_prime

    @property
    def g(self) -> float:
        """Reduced gravity in top layer."""
        return self.g_prime[0]

    def _set_coriolis(
        self,
        beta_plane: coriolis.BetaPlane,
    ) -> None:
        """Set Coriolis Grids.

        Args:
            beta_plane (coriolis.BetaPlane): Coriolis values.
        """
        # Coriolis values
        self._beta_plane = beta_plane
        f = self.beta_plane.compute_over_grid(self.space.omega.remove_z_h())
        self.f = f.unsqueeze(0)
        ## Coriolis grids
        self.f_ugrid = grid_conversion.omega_to_u(self.f)
        self.f_vgrid = grid_conversion.omega_to_v(self.f)
        self.f_hgrid = grid_conversion.omega_to_h(self.f)
        self.fstar_ugrid = self.f_ugrid * self.space.area
        self.fstar_vgrid = self.f_vgrid * self.space.area
        self.fstar_hgrid = self.f_hgrid * self.space.area

    def _set_H(  # noqa: N802
        self,
        h: torch.Tensor,
    ) -> None:
        """Validate H (unperturbed layer thickness) input value.

        Args:
            h (torch.Tensor): Layers Thickness.
        """
        if len(h.shape) < 3:  # noqa: PLR2004
            msg = (
                "H must be a nz x ny x nx tensor "
                "with nx=1 or ny=1 if H does not vary "
                f"in x or y direction, got shape {h.shape}."
            )
            raise InvalidModelParameterError(msg)
        self._H = h

    def _set_g_prime(self, g_prime: torch.Tensor) -> None:
        """Set g_rpime values.

        Args:
            g_prime (torch.Tensor): g_prime.
        """
        if g_prime.shape != self.H.shape:
            msg = (
                f"Inconsistent shapes for g_prime ({g_prime.shape}) "
                f"and H ({self.H.shape})"
            )
            raise InvalidModelParameterError(msg)
        self._g_prime = g_prime

    def _validate_mask(self, param: dict[str, Any], key: str) -> Masks:
        """Validate Mask value.

        Args:
            param (dict[str, Any]): Parameters dict.
            key (str): Mask key.

        Returns:
            Masks: Mask.
        """
        # If 'mask' key does not exist
        if key not in param:
            verbose.display(
                msg="No mask provided, domain assumed to be rectangular",
                trigger_level=2,
            )
            mask = torch.ones(
                self.space.nx,
                self.space.ny,
                dtype=self.dtype,
                device=self.device,
            )
            return Masks(mask)

        mask: torch.Tensor = param[key]
        shape = mask.shape[0], mask.shape[1]
        # Verify shape
        if shape != (self.space.nx, self.space.ny):
            msg = (
                "Invalid mask shape "
                f"{shape}!=({self.space.nx},{self.space.ny})"
            )
            raise KeyError(msg)
        vals = torch.unique(mask).tolist()
        # Verify mask values
        if not all(v in [0, 1] for v in vals) or vals == [0]:
            msg = f"Invalid mask with non-binary values : {vals}"
            raise KeyError(msg)
        verbose.display(
            msg=(
                f"{'Non-trivial' if len(vals)==2 else 'Trivial'}"  # noqa: PLR2004
                " mask provided"
            ),
            trigger_level=2,
        )
        return Masks(mask)

    def set_wind_forcing(
        self,
        taux: float | torch.Tensor,
        tauy: float | torch.Tensor,
    ) -> None:
        """Set the winf forcing attributes taux and tauy.

        # TODO: consider implementing validation
        in taux and tauy properties getters/

        Args:
            taux (float | torch.Tensor): Taux value.
            tauy (float | torch.Tensor): Tauy value

        Raises:
            ValueError: If taux is not a float nor a Tensor.
            ValueError: If taux if a wrongly-shaped Tensor.
            ValueError: If tauy is not a float nor a Tensor.
            ValueError: If tauy if a wrongly-shaped Tensor.
        """
        is_tensorx = isinstance(taux, torch.Tensor)

        if (not isinstance(taux, float)) and (not is_tensorx):
            msg = "taux must be a float or a Tensor"
            raise ValueError(msg)
        if is_tensorx and (taux.shape != (self.space.nx - 1, self.space.ny)):
            msg = (
                "Tau_x Tensor must be "
                f"{(self.space.nx-1, self.space.ny)}-shaped."
            )
            raise ValueError(msg)

        is_tensory = isinstance(tauy, torch.Tensor)

        if (not isinstance(tauy, float)) and (not is_tensory):
            msg = "tauy must be a float or a Tensor"
            raise ValueError(msg)
        if is_tensory and (tauy.shape != (self.space.nx, self.space.ny - 1)):
            msg = (
                "Tau_y Tensor must be "
                f"{(self.space.nx, self.space.ny-1)}-shaped."
            )
            raise ValueError(msg)

        self.taux = taux
        self.tauy = tauy

    def _set_ref_variables(self) -> None:
        """Set reference variables values.

        Concerned variables:
        - self.h_ref
        - self.eta_ref
        - self.p_ref
        - self.h_ref_ugrid
        - self.h_ref_vgrid
        - self.dx_p_ref
        - self.dy_p_ref
        """
        self.h_ref = self.H * self.space.area
        self.eta_ref = -self.H.sum(dim=-3) + reverse_cumsum(self.H, dim=-3)
        self.p_ref = torch.cumsum(self.g_prime * self.eta_ref, dim=-3)
        if self.h_ref.shape[-2] != 1 and self.h_ref.shape[-1] != 1:
            h_ref_ugrid = F.pad(self.h_ref, (0, 0, 1, 1), mode="replicate")
            self.h_ref_ugrid = 0.5 * (
                h_ref_ugrid[..., 1:, :] + h_ref_ugrid[..., :-1, :]
            )
            h_ref_vgrid = F.pad(self.h_ref, (1, 1), mode="replicate")
            self.h_ref_vgrid = 0.5 * (
                h_ref_vgrid[..., 1:] + h_ref_vgrid[..., :-1]
            )
            self.dx_p_ref = torch.diff(self.p_ref, dim=-2)
            self.dy_p_ref = torch.diff(self.p_ref, dim=-1)
        else:
            self.h_ref_ugrid = self.h_ref
            self.h_ref_vgrid = self.h_ref
            self.dx_p_ref = 0.0
            self.dy_p_ref = 0.0

    def _set_utils_before_compilation(self) -> None:
        """Set utils and flux function without compilation."""
        self.comp_ke = finite_diff.comp_ke
        self.cell_corners_to_cell_centers = (
            grid_conversion.cell_corners_to_cell_center
        )
        self.h_flux_y = lambda h, v: flux.flux(
            h,
            v,
            dim=-1,
            n_points=6,
            rec_func_2=reconstruction.linear2_centered,
            rec_func_4=reconstruction.wenoz4_left,
            rec_func_6=reconstruction.wenoz6_left,
            mask_2=self.masks.v_sten_hy_eq2[..., 1:-1],
            mask_4=self.masks.v_sten_hy_eq4[..., 1:-1],
            mask_6=self.masks.v_sten_hy_gt6[..., 1:-1],
        )
        self.h_flux_x = lambda h, u: flux.flux(
            h,
            u,
            dim=-2,
            n_points=6,
            rec_func_2=reconstruction.linear2_centered,
            rec_func_4=reconstruction.wenoz4_left,
            rec_func_6=reconstruction.wenoz6_left,
            mask_2=self.masks.u_sten_hx_eq2[..., 1:-1, :],
            mask_4=self.masks.u_sten_hx_eq4[..., 1:-1, :],
            mask_6=self.masks.u_sten_hx_gt6[..., 1:-1, :],
        )

        self.w_flux_y = lambda w, v_ugrid: flux.flux(
            w,
            v_ugrid,
            dim=-1,
            n_points=6,
            rec_func_2=reconstruction.linear2_centered,
            rec_func_4=reconstruction.wenoz4_left,
            rec_func_6=reconstruction.wenoz6_left,
            mask_2=self.masks.u_sten_wy_eq2[..., 1:-1, :],
            mask_4=self.masks.u_sten_wy_eq4[..., 1:-1, :],
            mask_6=self.masks.u_sten_wy_gt4[..., 1:-1, :],
        )
        self.w_flux_x = lambda w, u_vgrid: flux.flux(
            w,
            u_vgrid,
            dim=-2,
            n_points=6,
            rec_func_2=reconstruction.linear2_centered,
            rec_func_4=reconstruction.wenoz4_left,
            rec_func_6=reconstruction.wenoz6_left,
            mask_2=self.masks.v_sten_wx_eq2[..., 1:-1],
            mask_4=self.masks.v_sten_wx_eq4[..., 1:-1],
            mask_6=self.masks.v_sten_wx_gt6[..., 1:-1],
        )

    def _set_utils_with_compilation(self) -> None:
        """Set utils and flux function for compilation."""
        if torch.__version__[0] == "2":
            verbose.display(
                msg=(
                    f"torch version {torch.__version__} >= 2.0, "
                    "using torch.compile for compilation."
                ),
                trigger_level=2,
            )
            self.comp_ke = torch.compile(finite_diff.comp_ke)
            self.cell_corners_to_cell_centers = torch.compile(
                grid_conversion.omega_to_h,
            )
            self.h_flux_y = torch.compile(self.h_flux_y)
            self.h_flux_x = torch.compile(self.h_flux_x)
            self.w_flux_y = torch.compile(self.w_flux_y)
            self.w_flux_x = torch.compile(self.w_flux_x)
        else:
            verbose.display(
                msg=(
                    f"torch version {torch.__version__} < 2.0, "
                    "using torch.jit.trace for compilation."
                ),
                trigger_level=2,
            )
            self.comp_ke = torch.jit.trace(
                finite_diff.comp_ke,
                (self.u, self.U, self.v, self.V),
            )
            self.cell_corners_to_cell_centers = torch.jit.trace(
                grid_conversion.omega_to_h,
                (self.U,),
            )
            self.h_flux_y = torch.jit.trace(
                self.h_flux_y,
                (self.h, self.V[..., 1:-1]),
            )
            self.h_flux_x = torch.jit.trace(
                self.h_flux_x,
                (self.h, self.U[..., 1:-1, :]),
            )
            self.w_flux_y = torch.jit.trace(
                self.w_flux_y,
                (self.omega[..., 1:-1, :], self.V_m),
            )
            self.w_flux_x = torch.jit.trace(
                self.w_flux_x,
                (self.omega[..., 1:-1], self.U_m),
            )

    def _initialize_vars(self) -> None:
        """Initialize variables.

        Create Empty variables.

        Concerned variables:
        - u
        - v
        - h
        """
        base_shape = (self.n_ens, self.space.nl)
        self._h = torch.zeros(
            (*base_shape, self.space.nx, self.space.ny),
            dtype=self.dtype,
            device=self.device,
        )
        self._u = torch.zeros(
            (*base_shape, self.space.nx + 1, self.space.ny),
            dtype=self.dtype,
            device=self.device,
        )
        self._v = torch.zeros(
            (*base_shape, self.space.nx, self.space.ny + 1),
            dtype=self.dtype,
            device=self.device,
        )

    def compute_omega(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Pad u, v using boundary conditions.

        Possible boundary conditions: free-slip, partial free-slip, no-slip.

        Args:
            u (torch.Tensor): Prognostic zonal velocity.
            v (torch.Tensor): Prognostic meridional velocity.

        Returns:
            torch.Tensor: result
        """
        u_ = F.pad(u, (1, 1, 0, 0))
        v_ = F.pad(v, (0, 0, 1, 1))
        dx_v = torch.diff(v_, dim=-2)
        dy_u = torch.diff(u_, dim=-1)
        curl_uv = dx_v - dy_u
        alpha = 2 * (1 - self.slip_coef)
        omega: torch.Tensor = (
            self.masks.w_valid * curl_uv
            + self.masks.w_cornerout_bound * (1 - self.slip_coef) * curl_uv
            + self.masks.w_vertical_bound * alpha * dx_v
            - self.masks.w_horizontal_bound * alpha * dy_u
        )
        return omega

    def compute_diagnostic_variables(self) -> None:
        """Compute the model's diagnostic variables.

        Computed variables:
        - Vorticity: omega
        - Interface heights: eta
        - Pressure: p
        - Zonal velocity: U
        - Meridional velocity: V
        - Zonal Velocity Momentum: U_m
        - Meriodional Velocity Momentum: V_m
        - Kinetic Energy: k_energy

        Compute the result given the prognostic
        variables self.u, self.v, self.h .
        """
        # Diagnostic: vorticity values
        self.omega = self.compute_omega(self.u, self.v)
        # Diagnostic: interface height : physical
        self.eta = reverse_cumsum(self.h / self.space.area, dim=-3)
        # Diagnostic: pressure values
        self.p = torch.cumsum(self.g_prime * self.eta, dim=-3)
        # Diagnostic: zonal velocity
        self.U = self.u / self.space.dx**2
        # Diagnostic: meridional velocity
        self.V = self.v / self.space.dy**2
        # Zonal velocity momentum -> corresponds to the v grid
        # Has no value on the boundary of the v grid
        self.U_m = self.cell_corners_to_cell_centers(self.U)
        # Meridional velocity momentum -> corresponds to the u grid
        # Has no value on the boundary of the u grid
        self.V_m = self.cell_corners_to_cell_centers(self.V)
        # Diagnostic: kinetic energy
        self.k_energy = (
            self.comp_ke(self.u, self.U, self.v, self.V) * self.masks.h
        )
        # Match u grid dimensions (1, nl, nx, ny)
        self.h_ugrid = grid_conversion.h_to_u(self.h, self.masks.h)
        # Match v grid dimension
        self.h_vgrid = grid_conversion.h_to_v(self.h, self.masks.h)
        # Sum h on u grid
        self.h_tot_ugrid = self.h_ref_ugrid + self.h_ugrid
        # Sum h on v grid
        self.h_tot_vgrid = self.h_ref_vgrid + self.h_vgrid

    def get_physical_uvh(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get physical variables u_phys, v_phys, h_phys from state variables.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: u, v and h
        """
        u_phys = (self.u / self.space.dx).to(device=DEVICE)
        v_phys = (self.v / self.space.dy).to(device=DEVICE)
        h_phys = (self.h / self.space.area).to(device=DEVICE)

        return (u_phys, v_phys, h_phys)

    def get_physical_uvh_as_ndarray(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get physical variables u_phys, v_phys, h_phys from state variables.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: u, v and h
        """
        return (e.cpu().numpy() for e in self.get_physical_uvh())

    def get_physical_omega(
        self,
    ) -> torch.Tensor:
        """Get physical vorticity.

        Returns:
            torch.Tensor: Vorticity
        """
        vorticity = self.omega / self.space.area / self.beta_plane.f0
        return vorticity.to(device=DEVICE)

    def get_physical_omega_as_ndarray(
        self,
    ) -> torch.Tensor:
        """Get physical vorticity.

        Returns:
            np.ndarray: Vorticity
        """
        return self.get_physical_omega().cpu().numpy()

    @abstractmethod
    def set_physical_uvh(
        self,
        u_phys: torch.Tensor | np.ndarray,
        v_phys: torch.Tensor | np.ndarray,
        h_phys: torch.Tensor | np.ndarray,
    ) -> None:
        """Set state variables from physical variables.

        Args:
            u_phys (torch.Tensor|np.ndarray): Physical U.
            v_phys (torch.Tensor|np.ndarray): Physical V.
            h_phys (torch.Tensor|np.ndarray): Physical H.
        """

    def set_uvh(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> None:
        """Set u,v,h value from state variables.

        Warning: the expected values are not physical values but state values.
        The variables correspond to the actual self.u, self.v, self.h
        of the model.

        Args:
            u (torch.Tensor): State variable u.
            v (torch.Tensor): State variable v.
            h (torch.Tensor): State variable h.
        """
        u = u.to(self.device)
        v = v.to(self.device)
        h = h.to(self.device)

        if not (u * self.masks.u == u).all():
            msg = (
                "Input velocity u incoherent with domain mask, "
                "velocity must be zero out of domain."
            )
            raise IncoherentWithMaskError(msg)

        if not (v * self.masks.v == v).all():
            msg = (
                "Input velocity v incoherent with domain mask, "
                "velocity must be zero out of domain."
            )
            raise IncoherentWithMaskError(msg)
        self._u = u.type(self.dtype) * self.masks.u
        self._v = v.type(self.dtype) * self.masks.v
        self._h = h.type(self.dtype) * self.masks.h
        self.compute_diagnostic_variables()

    def get_print_info(self) -> str:
        """Returns a string with summary of current variables.

        Returns:
            str: Summary of variables.
        """
        hl_mean = (self.h / self.space.area).mean((-1, -2))
        eta = self.eta
        u = self.u / self.space.dx
        v = self.v / self.space.dy
        h = self.h / self.space.area
        with np.printoptions(precision=2):
            eta_surface = eta[:, 0].min().to(device=DEVICE).item()
            return (
                f"u: {u.mean().to(device=DEVICE).item():+.5E}, "
                f"{u.abs().max().to(device=DEVICE).item():.5E}, "
                f"v: {v.mean().to(device=DEVICE).item():+.5E}, "
                f"{v.abs().max().to(device=DEVICE).item():.5E}, "
                f"hl_mean: {hl_mean.squeeze().cpu().numpy()}, "
                f"h min: {h.min().to(device=DEVICE).item():.5E}, "
                f"max: {h.max().to(device=DEVICE).item():.5E}, "
                f"eta_sur min: {eta_surface:+.5f}, "
                f"max: {eta[:,0].max().to(device=DEVICE).item():.5f}"
            )

    @abstractmethod
    def advection_h(self) -> torch.Tensor:
        """Advection RHS for thickness perturbation h.

        dt_h = - div(h_tot [u v])

        h_tot = h_ref + h

        Returns:
            torch.Tensor: h advection.
        """

    @abstractmethod
    def advection_momentum(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advection RHS for momentum (u, v).

        Returns:
            tuple[torch.Tensor,torch.Tensor]: u, v advection.
        """

    def _add_wind_forcing(
        self,
        du: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add wind forcing to the derivatives du, dv.

        Args:
            du (torch.Tensor): du
            dv (torch.Tensor): dv

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with wind forcing.
        """
        h_ugrid = self.h_tot_ugrid / self.space.area
        h_vgrid = self.h_tot_vgrid / self.space.area
        du_wind = self.taux / h_ugrid[..., 0, 1:-1, :] * self.space.dx
        dv_wind = self.tauy / h_vgrid[..., 0, :, 1:-1] * self.space.dy
        du[..., 0, :, :] += du_wind
        dv[..., 0, :, :] += dv_wind
        return du, dv

    def _add_bottom_drag(
        self,
        du: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add bottom drag to the derivatives du, dv.

        Args:
            du (torch.Tensor): du
            dv (torch.Tensor): dv

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with botoom drag forcing.
        """
        du[..., -1, :, :] += -self.bottom_drag_coef * self.u[..., -1, 1:-1, :]
        dv[..., -1, :, :] += -self.bottom_drag_coef * self.v[..., -1, :, 1:-1]
        return du, dv

    @abstractmethod
    def compute_time_derivatives(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the state variables derivatives dt_u, dt_v, dt_h.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: dt_u, dt_v, dt_h
        """

    def _raise_if_invalid_savefile(self, output_file: Path) -> None:
        """Raise and error if the saving file is invalid.

        Args:
            output_file (Path): Output file.

        Raises:
            InvalidSavingFileError: if the saving file extension is not .npz.
        """
        if output_file.suffix != ".npz":
            msg = "Variables are expected to be saved in an .npz file."
            raise InvalidSavingFileError(msg)

    def save_uvh(self, output_file: Path) -> None:
        """Save U, V and H values.

        Args:
            output_file (Path): File to save value in (.npz).
        """
        self._raise_if_invalid_savefile(output_file=output_file)

        u, v, h = self.get_physical_uvh_as_ndarray()

        np.savez(
            output_file,
            u=u.astype("float32"),
            v=v.astype("float32"),
            h=h.astype("float32"),
        )

        verbose.display(msg=f"saved u,v,h to {output_file}", trigger_level=1)

    def save_omega(self, output_file: Path) -> None:
        """Save vorticity values.

        Args:
            output_file (Path): File to save value in (.npz).
        """
        self._raise_if_invalid_savefile(output_file=output_file)

        omega = self.get_physical_omega_as_ndarray()

        np.savez(output_file, omega=omega.astype("float32"))

        verbose.display(msg=f"saved ω to {output_file}", trigger_level=1)

    @abstractmethod
    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
