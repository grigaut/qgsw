"""Special variables."""

from collections.abc import Callable

import torch

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.uvh import UVH, UVHTAlpha
from qgsw.filters.base import _Filter
from qgsw.models.core.utils import OptimizableFunction
from qgsw.spatial.core.grid_conversion import points_to_surfaces
from qgsw.utils.shape_checks import with_shapes


@with_shapes(g_prime=(2,))
def compute_g_tilde(g_prime: torch.Tensor) -> torch.Tensor:
    """Compute g_tilde = g_1 g_2 / (g_1 + g_2).

    Args:
        g_prime (torch.Tensor): Reduced gravity tensor.
            └── (2,) shaped

    Returns:
        torch.Tensor: g_tilde = g_1 g_2 / (g_1 + g_2)
            └── (1,) shaped
    """
    if g_prime.shape != (2,):
        msg = f"g' should be (2,)-shaped, not {g_prime.shape}."
        raise ValueError(msg)
    g1, g2 = g_prime
    return (g1 * g2 / (g1 + g2)).unsqueeze(0)


@with_shapes(
    alpha=(1,),
    H1=(1,),
    g2=(1,),
)
def compute_source_term_factor(
    alpha: torch.Tensor,
    H1: torch.Tensor,  # noqa: N803
    g2: torch.Tensor,
    f0: float,
) -> torch.Tensor:
    """Compute source term multiplicative factor.

    Args:
        alpha (torch.Tensor): Collinearity coefficient.
            └── (1, )-shaped.
        H1 (torch.Tensor): Top layer reference depth.
            └── (1, )-shaped.
        g2 (torch.Tensor): Reduced gravity in the second layer.
            └── (1, )-shaped.
        f0 (float): f0.

    Returns:
        torch.Tensor: f_0²α/H1/g2
    """
    return f0**2 * alpha / H1 / g2


@with_shapes(
    alpha=(1,),
    H1=(1,),
    g2=(1,),
    g_tilde=(1,),
)
def compute_source_term(
    uvh: UVH,
    filt: _Filter,
    alpha: torch.Tensor,
    H1: torch.Tensor,  # noqa: N803
    g2: torch.Tensor,
    g_tilde: torch.Tensor,
    f0: float,
    ds: float,
    points_to_surface: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute source term.

    Args:
        uvh (UVH): Prognostic u,v and h.
            ├── u: (n_ens, 1, nx+1, ny)-shaped
            ├── v: (n_ens, 1, nx, ny+1)-shaped
            └── h: (n_ens, 1, nx, ny)-shaped
        filt (_Filter): Filter.
        alpha (torch.Tensor): Collinearity coefficient.
        H1 (torch.Tensor): Top layer depth.
            └── (1, )-shaped.
        g2 (torch.Tensor): Reduced gravity in the bottom layer.
            └── (1, )-shaped.
        g_tilde (torch.Tensor): Equivalent reduced gravity.
            └── (1, )-shaped.
        f0 (float): f0.
        ds (float): ds.
        points_to_surface (Callable[[torch.Tensor], torch.Tensor]): Points
        to surface interpolation function.

    Returns:
        torch.Tensor: Source term: f_0αg̃/H1/g2/ds (F^s)⁻¹{K F{h}}.
    """
    h_top_i = points_to_surface(uvh.h[0, 0])
    h_filt = filt(h_top_i).unsqueeze(0).unsqueeze(0)
    h_to_psi = g_tilde * h_filt / f0
    factor = compute_source_term_factor(alpha, H1, g2, f0)
    return (factor * h_to_psi) / ds


@with_shapes(
    H=(1, 1, 1),
    g_prime=(2, 1, 1),
    g_tilde=(1,),
    alpha=(1,),
)
def compute_pv(
    uvh: UVH,
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
    g_tilde: torch.Tensor,
    f0: float,
    ds: float,
    filt: _Filter,
    alpha: torch.Tensor,
    points_to_surfaces: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """PV linear operator.

    Args:
        uvh (UVH): Prognostic u,v and h.
            ├── u: (n_ens, 1, nx+1, ny)-shaped
            ├── v: (n_ens, 1, nx, ny+1)-shaped
            └── h: (n_ens, 1, nx, ny)-shaped
        H (torch.Tensor): Layers reference thickness.
            └── (1, 1, 1)-shaped.
        g_prime (torch.Tensor): Reduced gravity.
            └── (2, 1, 1)-shaped.
        g_tilde (torch.Tensor): Equivalent reduced gravity.
            └── (1, )-shaped.
        f0 (float): f0.
        ds (float): ds.
        filt (_Filter): Filter.
        alpha (torch.Tensor): Collinearity coefficient.
            └── (1, )-shaped.
        points_to_surfaces (Callable[[torch.Tensor], torch.Tensor]): Points
        to surface interpolation function.

    Returns:
        torch.Tensor: Physical Potential Vorticity.
            └── (n_ens, nl, nx-1, ny-1)-shaped.
    """
    # Compute ω = ∂_x v - ∂_y u
    omega = torch.diff(uvh.v[..., 1:-1], dim=-2) - torch.diff(
        uvh.u[..., 1:-1, :],
        dim=-1,
    )
    # Compute ω-f_0*h/H
    pv = (omega - f0 * points_to_surfaces(uvh.h) / H) / ds
    source_term = compute_source_term(
        uvh=uvh,
        filt=filt,
        H1=H[..., 0, 0],
        g2=g_prime[1:, 0, 0],
        g_tilde=g_tilde,
        alpha=alpha,
        f0=f0,
        ds=ds,
        points_to_surface=points_to_surfaces,
    )
    return pv + source_term


class CollinearFilteredPotentialVorticity(DiagnosticVariable):
    """Collinear Filtered PV."""

    @with_shapes(
        H=(2,),
        g_prime=(2,),
    )
    def __init__(
        self,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        f0: float,
        ds: float,
        filt: _Filter,
    ) -> None:
        """Instantiate variable.

        Args:
            H (torch.Tensor): Layers reference thickness.
            g_prime (torch.Tensor): Reduced gravity tensor.
            f0 (float): _description_
            ds (float): _description_
            filt (_Filter): _description_
        """
        self._H = H
        self._g_prime = g_prime
        self._g_tilde = compute_g_tilde(g_prime)
        self._f0 = f0
        self._ds = ds
        self._filt = filt
        self._points_to_surface = OptimizableFunction(points_to_surfaces)

    def _compute(self, prognostic: UVHTAlpha) -> torch.Tensor:
        return compute_pv(
            prognostic.uvh,
            self._H,
            self._g_prime,
            self._g_tilde,
            self._f0,
            self._ds,
            self._filt,
            prognostic.alpha,
            self._points_to_surface,
        )
