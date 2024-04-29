"""Base models class."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import reconstruction, verbose
from qgsw.masks import Masks
from qgsw.models.core import finite_diff, flux
from qgsw.models.exceptions import InvalidSavingFileError

if TYPE_CHECKING:
    from pathlib import Path


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


def pool_2d(padded_f: torch.Tensor) -> torch.Tensor:
    """2D pool a padded tensor.

    Args:
        padded_f (torch.Tensor): Tensor to pool.

    Returns:
        torch.Tensor: Padded tensor.
    """
    # average pool padded value
    f_sum_pooled = F.avg_pool2d(
        padded_f,
        (3, 1),
        stride=(1, 1),
        padding=(1, 0),
        divisor_override=1,
    )
    return F.avg_pool2d(
        f_sum_pooled,
        (1, 3),
        stride=(1, 1),
        padding=(0, 1),
        divisor_override=1,
    )


def replicate_pad(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replicate a given pad.

    Args:
        f (torch.Tensor): Tensor to pad.
        mask (torch.Tensor): Mask tensor.

    Returns:
        torch.Tensor: Result
    """
    f_ = F.pad(f, (1, 1, 1, 1))
    mask_ = F.pad(mask, (1, 1, 1, 1))
    mask_sum = pool_2d(mask_)
    f_sum = pool_2d(f_)
    f_out = f_sum / torch.maximum(torch.ones_like(mask_sum), mask_sum)
    return mask_ * f_ + (1 - mask_) * f_out


class Model(metaclass=ABCMeta):
    """Base class for models."""

    barotropic_filter: bool
    barotropic_filter_spectral: bool

    def __init__(self, param: dict[str, Any]) -> None:
        """Parameters

        param: python dict. with following keys
            'nx':       int, number of grid points in dimension x
            'ny':       int, number grid points in dimension y
            'nl':       nl, number of stacked layer
            'dx':       float or Tensor (nx, ny), dx metric term
            'dy':       float or Tensor (nx, ny), dy metric term
            'H':        Tensor (nl,) or (nl, nx, ny),
            unperturbed layer thickness
            'g_prime':  Tensor (nl,), reduced gravities
            'f':        Tensor (nx, ny), Coriolis parameter
            'taux':     float or Tensor (nx-1, ny),
            top-layer forcing, x component
            'tauy':     float or Tensor (nx, ny-1),
            top-layer forcing, y component
            'dt':       float > 0., integration time-step
            'n_ens':    int, number of ensemble member
            'device':   'str', torch devicee e.g. 'cpu', 'cuda', 'cuda:0'
            'dtype':    torch.float32 of torch.float64
            'slip_coef':    float, 1 for free slip, 0 for no-slip,
            inbetween for partial free slip.
            'bottom_drag_coef': float, linear bottom drag coefficient
            'barotropic_filter': boolean, applies implicit FS calculation
            if True
        """
        verbose.display(
            msg=f"Creating {self.__class__.__name__} model...",
            trigger_level=1,
        )

        # Set up
        ## Time Step
        self.dt = param["dt"]
        ## data device and dtype
        self._set_array_kwargs(param=param)
        ## grid
        self._set_grid(param=param)
        ## Physical Variables
        self._set_physical_variables(param=param)
        ## Forcing
        self._set_forcings(param=param)
        ## Topography and Ref values
        self._set_ref_variables()

        # ensemble
        self.n_ens = param.get("n_ens", 1)

        # initialize variables
        self._initialize_vars()

        self.comp_ke = finite_diff.comp_ke
        self.interp_TP = finite_diff.interp_TP
        self.compute_diagnostic_variables()

        # utils and flux computation functions
        self._set_utils_before_compilation()
        # precompile torch functions
        if param.get("compile", True):
            self._set_utils_with_compilation()
        else:
            verbose.display(msg="No compilation", trigger_level=2)

    @property
    def dt(self) -> float:
        """Timestep value."""
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        verbose.display(msg=f"dt value set to {value}.", trigger_level=1)
        self._dt = value

    def _set_array_kwargs(self, param: dict[str, Any]) -> None:
        """Set the array kwargs.

        Args:
            param (dict[str, Any]): Parameters dictionnary.
        """
        self.device = param["device"]
        self.dtype = param.get("dtype", torch.float64)
        self.arr_kwargs = {"dtype": self.dtype, "device": self.device}
        verbose.display(
            msg=f"dtype: {self.dtype}.",
            trigger_level=2,
        )
        verbose.display(
            msg=f"device: {self.device}",
            trigger_level=2,
        )

    def _set_grid(self, param: dict[str, Any]) -> None:
        """Set the Grid informations.

        Args:
            param (dict[str, Any]): Parameters dictionnary.
        """
        self.nx: int = param["nx"]  # number of points in the x direction
        self.ny: int = param["ny"]  # number of point in the y direction
        self.nl: int = param["nl"]  # number of layers
        self.dx: float = param["dx"]  # x-step
        self.dy: float = param["dy"]  # y-step
        self.area = self.dx * self.dy  # elementary area
        verbose.display(
            msg=f"nx, ny, nl =  {self.nx, self.ny, self.nl}",
            trigger_level=2,
        )

    def _set_physical_variables(self, param: dict[str, Any]) -> None:
        """Set physical varibales.

        Args:
            param (dict[str,Any]): Parameters dictionnary.
        """
        # Validate parameters values and shapes.
        ## H
        self.H = self._validate_layers(param=param, key="H")
        ## optional mask
        self.masks = self._validate_mask(param=param, key="mask")
        ## boundary conditions
        self.slip_coef = self._validate_slip_coef(param=param, key="slip_coef")
        ## Coriolis parameter
        self.f = self._validate_coriolis_param(param=param, key="f")
        ## Coriolis grids
        self.f0 = self.f.mean()
        self.f_ugrid = 0.5 * (self.f[:, :, 1:] + self.f[:, :, :-1])
        self.f_vgrid = 0.5 * (self.f[:, 1:, :] + self.f[:, :-1, :])
        self.f_hgrid = finite_diff.interp_TP(self.f)
        self.fstar_ugrid = self.f_ugrid * self.area
        self.fstar_vgrid = self.f_vgrid * self.area
        self.fstar_vgrid = self.f_vgrid * self.area
        self.fstar_hgrid = self.f_hgrid * self.area
        ## gravity
        self.g_prime: torch.Tensor = param["g_prime"]
        self.g = self.g_prime[0]

    def _set_forcings(self, param: dict[str, Any]) -> None:
        """Set forcing.

        Args:
            param (dict[str, Any]): Parameters dictionnary.
        """
        # Top layer forcing
        taux, tauy = param["taux"], param["tauy"]
        self.set_wind_forcing(taux, tauy)
        # Bottom layer forcing
        self.bottom_drag_coef = param["bottom_drag_coef"]

    def _validate_layers(
        self, param: dict[str, Any], key: str
    ) -> torch.Tensor:
        """Validate H (unperturbed layer thickness) input value.

        Args:
            param (dict[str, Any]): Parameters dict.
            key (str): Key for H value.

        Returns:
            torch.Tensor: H
        """
        value: torch.Tensor = param[key]
        if len(value.shape) < 3:  # noqa: PLR2004
            msg = (
                "H must be a nz x ny x nx tensor "
                "with nx=1 or ny=1 if H does not vary "
                f"in x or y direction, got shape {value.shape}."
            )
            raise KeyError(msg)
        return value

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
                self.nx,
                self.ny,
                dtype=self.dtype,
                device=self.device,
            )
            return Masks(mask)

        mask: torch.Tensor = param[key]
        shape = mask.shape[0], mask.shape[1]
        # Verify shape
        if shape != (self.nx, self.ny):
            msg = f"Invalid mask shape {shape}!=({self.nx},{self.ny})"
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

    def _validate_slip_coef(self, param: dict[str, Any], key: str) -> float:
        """Valide slip coeffciient value.

        Args:
            param (dict[str, Any]): Parameters dict.
            key (str): Key for slip coefficient.

        Returns:
            float: Slip coefficient value.
        """
        value = param.get(key, 1.0)
        # Verify value in [0,1]
        if (value < 0) or (value > 1):
            msg = f"slip coefficient must be in [0, 1], got {value}"
            raise KeyError(msg)
        cl_type = (
            "Free-"
            if value == 1
            else ("No-" if value == 0 else "Partial free-")
        )
        verbose.display(
            msg=f"{cl_type}slip boundary condition",
            trigger_level=2,
        )
        return value

    def _validate_coriolis_param(
        self, param: dict[str, Any], key: str
    ) -> torch.Tensor:
        """Validate f (Coriolis parameter) value.

        Args:
            param (dict[str, Any]): Parameters dict.
            key (str): Key for coriolis parameter.

        Returns:
            torch.Tensor: Coriolis parameter value.
        """
        value: torch.Tensor = param[key]
        shape = value.shape[0], value.shape[1]
        if shape != (self.nx + 1, self.ny + 1):
            msg = f"Invalid f shape {shape=}!=({self.nx},{self.ny})"
            raise KeyError(msg)
        return value.unsqueeze(0)

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
        if is_tensorx and (taux.shape != (self.nx - 1, self.ny)):
            msg = f"Tau_x Tensor must be {(self.nx-1, self.ny)}-shaped."
            raise ValueError(msg)

        is_tensory = isinstance(tauy, torch.Tensor)

        if (not isinstance(tauy, float)) and (not is_tensory):
            msg = "tauy must be a float or a Tensor"
            raise ValueError(msg)
        if is_tensory and (tauy.shape != (self.nx, self.ny - 1)):
            msg = f"Tau_y Tensor must be {(self.nx, self.ny-1)}-shaped."
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
        self.h_ref = self.H * self.area
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
        self.interp_TP = finite_diff.interp_TP
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
            self.interp_TP = torch.compile(finite_diff.interp_TP)
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
                finite_diff.comp_ke, (self.u, self.U, self.v, self.V)
            )
            self.interp_TP = torch.jit.trace(finite_diff.interp_TP, (self.U,))
            self.h_flux_y = torch.jit.trace(
                self.h_flux_y, (self.h, self.V[..., 1:-1])
            )
            self.h_flux_x = torch.jit.trace(
                self.h_flux_x, (self.h, self.U[..., 1:-1, :])
            )
            self.w_flux_y = torch.jit.trace(
                self.w_flux_y, (self.omega[..., 1:-1, :], self.V_m)
            )
            self.w_flux_x = torch.jit.trace(
                self.w_flux_x, (self.omega[..., 1:-1], self.U_m)
            )

    def _initialize_vars(self) -> None:
        """Initialize variables.

        Concerned variables:
        - u
        - v
        - h
        """
        base_shape = (self.n_ens, self.nl)
        self.h = torch.zeros(
            (*base_shape, self.nx, self.ny),
            **self.arr_kwargs,
        )
        self.u = torch.zeros(
            (*base_shape, self.nx + 1, self.ny),
            **self.arr_kwargs,
        )
        self.v = torch.zeros(
            (*base_shape, self.nx, self.ny + 1),
            **self.arr_kwargs,
        )

    def compute_omega(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Pad u, v using boundary conditions.

        Possible boundary conditions: free-slip, partial free-slip, no-slip.

        Args:
            u (torch.Tensor): U
            v (torch.Tensor): V

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

        Compute the result given the prognostic
        variables self.u, self.v, self.h .
        """
        self.omega = self.compute_omega(self.u, self.v)
        self.eta = reverse_cumsum(self.h / self.area, dim=-3)
        self.p = torch.cumsum(self.g_prime * self.eta, dim=-3)
        self.U = self.u / self.dx**2
        self.V = self.v / self.dy**2
        self.U_m = self.interp_TP(self.U)
        self.V_m = self.interp_TP(self.V)
        self.k_energy = (
            self.comp_ke(self.u, self.U, self.v, self.V) * self.masks.h
        )

        h_ = replicate_pad(self.h, self.masks.h)
        self.h_ugrid = 0.5 * (h_[..., 1:, 1:-1] + h_[..., :-1, 1:-1])
        self.h_vgrid = 0.5 * (h_[..., 1:-1, 1:] + h_[..., 1:-1, :-1])
        self.h_tot_ugrid = self.h_ref_ugrid + self.h_ugrid
        self.h_tot_vgrid = self.h_ref_vgrid + self.h_vgrid

    @overload
    def get_physical_uvh(
        self, numpy: Literal[True]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    @overload
    def get_physical_uvh(
        self, numpy: Literal[False]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def get_physical_uvh(
        self, numpy: bool = False
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Get physical variables u_phys, v_phys, h_phys from state variables.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: u, v and h
        """
        u_phys = (self.u / self.dx).cpu()
        v_phys = (self.v / self.dy).cpu()
        h_phys = (self.h / self.area).cpu()

        return (
            (u_phys.numpy(), v_phys.numpy(), h_phys.numpy())
            if numpy
            else (u_phys, v_phys, h_phys)
        )

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
        u_ = (
            torch.from_numpy(u_phys)
            if isinstance(u_phys, np.ndarray)
            else u_phys
        )
        v_ = (
            torch.from_numpy(v_phys)
            if isinstance(v_phys, np.ndarray)
            else v_phys
        )
        h_ = (
            torch.from_numpy(h_phys)
            if isinstance(h_phys, np.ndarray)
            else h_phys
        )
        u_ = u_.to(self.device)
        v_ = u_.to(self.device)
        h_ = u_.to(self.device)
        assert u_ * self.masks.u == u_, (
            "Input velocity u incoherent with domain mask, "
            "velocity must be zero out of domain."
        )
        assert v_ * self.masks.v == v_, (
            "Input velocity v incoherent with domain mask, "
            "velocity must be zero out of domain."
        )
        self.u = u_.type(self.dtype) * self.masks.u * self.dx
        self.v = v_.type(self.dtype) * self.masks.v * self.dy
        self.h = h_.type(self.dtype) * self.masks.h * self.area
        self.compute_diagnostic_variables()

    def get_print_info(self) -> str:
        """Returns a string with summary of current variables.

        Returns:
            str: Summary of variables.
        """
        hl_mean = (self.h / self.area).mean((-1, -2)).squeeze().cpu().numpy()
        eta = self.eta
        u, v, h = self.u / self.dx, self.v / self.dy, self.h / self.area
        with np.printoptions(precision=2):
            return (
                f"u: {u.mean().cpu().item():+.5E}, "
                f"{u.abs().max().cpu().item():.5E}, "
                f"v: {v.mean().cpu().item():+.5E}, "
                f"{v.abs().max().cpu().item():.5E}, "
                f"hl_mean: {hl_mean}, "
                f"h min: {h.min().cpu().item():.5E}, "
                f"max: {h.max().cpu().item():.5E}, "
                f"eta_sur min: {eta[:,0].min().cpu().item():+.5f}, "
                f"max: {eta[:,0].max().cpu().item():.5f}"
            )

    def advection_h(self) -> torch.Tensor:
        """Advection RHS for thickness perturbation h.

        dt_h = - div(h_tot [u v])

        h_tot = h_ref + h

        Returns:
            torch.Tensor: h advection.
        """
        h_tot = self.h_ref + self.h
        h_tot_flux_y = self.h_flux_y(h_tot, self.V[..., 1:-1])
        h_tot_flux_x = self.h_flux_x(h_tot, self.U[..., 1:-1, :])
        div_no_flux = -finite_diff.div_nofluxbc(h_tot_flux_x, h_tot_flux_y)
        return div_no_flux * self.masks.h

    def advection_momentum(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advection RHS for momentum (u, v).

        Returns:
            tuple[torch.Tensor,torch.Tensor]: u, v advection.
        """
        # Vortex-force + Coriolis
        omega_v_m = self.w_flux_y(self.omega[..., 1:-1, :], self.V_m)
        omega_u_m = self.w_flux_x(self.omega[..., 1:-1], self.U_m)

        dt_u = omega_v_m + self.fstar_ugrid[..., 1:-1, :] * self.V_m
        dt_v = -(omega_u_m + self.fstar_vgrid[..., 1:-1] * self.U_m)

        # grad pressure + k_energy
        ke_pressure = self.k_energy + self.p
        dt_u -= torch.diff(ke_pressure, dim=-2) + self.dx_p_ref
        dt_v -= torch.diff(ke_pressure, dim=-1) + self.dy_p_ref

        # wind forcing and bottom drag
        dt_u, dt_v = self._add_wind_forcing(dt_u, dt_v)
        dt_u, dt_v = self._add_bottom_drag(dt_u, dt_v)

        return F.pad(dt_u, (0, 0, 1, 1)) * self.masks.u, F.pad(
            dt_v, (1, 1, 0, 0)
        ) * self.masks.v

    def _add_wind_forcing(
        self, du: torch.Tensor, dv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add wind forcing to the derivatives du, dv.

        Args:
            du (torch.Tensor): du
            dv (torch.Tensor): dv

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with wind forcing.
        """
        h_ugrid = (self.h_tot_ugrid) / self.area
        h_vgrid = (self.h_tot_vgrid) / self.area
        du[..., 0, :, :] += self.taux / h_ugrid[..., 0, 1:-1, :] * self.dx
        dv[..., 0, :, :] += self.tauy / h_vgrid[..., 0, :, 1:-1] * self.dy
        return du, dv

    def _add_bottom_drag(
        self, du: torch.Tensor, dv: torch.Tensor
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

    def compute_time_derivatives(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the state variables derivatives dt_u, dt_v, dt_h.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: dt_u, dt_v, dt_h
        """
        self.compute_diagnostic_variables()
        dt_h = self.advection_h()
        dt_u, dt_v = self.advection_momentum()
        return dt_u, dt_v, dt_h

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

        u, v, h = self.get_physical_uvh(numpy=True)

        np.savez(
            output_file,
            u=u.astype("float32"),
            v=v.astype("float32"),
            h=h.astype("float32"),
        )

        verbose.display(msg=f"saved u,v,h to {output_file}", trigger_level=1)

    @abstractmethod
    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        dt0_u, dt0_v, dt0_h = self.compute_time_derivatives()
        self.u += self.dt * dt0_u
        self.v += self.dt * dt0_v
        self.h += self.dt * dt0_h

        dt1_u, dt1_v, dt1_h = self.compute_time_derivatives()
        self.u += (self.dt / 4) * (dt1_u - 3 * dt0_u)
        self.v += (self.dt / 4) * (dt1_v - 3 * dt0_v)
        self.h += (self.dt / 4) * (dt1_h - 3 * dt0_h)

        dt2_u, dt2_v, dt2_h = self.compute_time_derivatives()
        self.u += (self.dt / 12) * (8 * dt2_u - dt1_u - dt0_u)
        self.v += (self.dt / 12) * (8 * dt2_v - dt1_v - dt0_v)
        self.h += (self.dt / 12) * (8 * dt2_h - dt1_h - dt0_h)
