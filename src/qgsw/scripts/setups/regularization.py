"""Regularization setups."""

from collections.abc import Callable

import torch

from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.scripts.regularization import (
    compute_div_dynamical_regularization_func,
    compute_qg_dynamical_regularization_func,
)
from qgsw.scripts.setups.types import Bases
from qgsw.spatial.core.discretization import SpaceDiscretization2D

DynRegFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def setup_background_regs(
    with_bkg_regs: dict[str, bool],
    Qs: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Callable[[DecompositionCoefs], torch.Tensor]]:
    """Setup background regularizations.

    Args:
        with_bkg_regs (dict[str, bool]): Whether to use background for basis.
        Qs (dict[str, dict[str, torch.Tensor]]): Covariances.

    Returns:
        dict[str, Callable[[DecompositionCoefs], torch.Tensor]]: Regularization
            functions mapping.
    """
    fns = {}
    for k, cond in with_bkg_regs.items():
        if cond:
            fns[k] = background_reg(Qs[k])
    return fns


def setup_dynamical_regs(
    with_dyn_regs: dict[str, bool],
    bases: Bases,
    A: torch.Tensor,
    space: SpaceDiscretization2D,
    beta_plane: BetaPlane,
    beta_effect: torch.Tensor,
    T: float = 1,
    *,
    wide: bool = False,
) -> dict[str, DynRegFn]:
    """Setup dynamical rgeularizations.

    Args:
        with_dyn_regs (dict[str, bool]): Dynamical regularizations.
        bases (Bases): Bases.
        A (torch.Tensor): Stretching matrix.
        space (SpaceDiscretization2D): Space.
        beta_plane (BetaPlane): Beta plane.
        beta_effect (torch.Tensor): Beta effect.
        T (float, optional): Time scale (L/U). Defaults to 1.
        wide (bool, optional): Whether to use wide stencil or not.
            Defaults to False.

    Returns:
        dict[ str, DynRegFn ]:
            Mapping reg_name -> reg function.
    """
    fns = {}
    for k, cond in with_dyn_regs.items():
        if cond:
            fns[k] = DYN_FNS_MAPPINGS[k](
                bases, A, space, beta_plane, beta_effect, T, wide=wide
            )
    return fns


def background_reg(
    Q: dict[int, torch.Tensor],
) -> Callable[[DecompositionCoefs], torch.Tensor]:
    """Backgroung regularization.

    Args:
        Q (dict[int, torch.Tensor]): Covariances.

    Returns:
        Callable[[DecompositionCoefs], torch.Tensor]: Regularization function.
    """

    def compute_reg(coefs: DecompositionCoefs) -> torch.Tensor:
        return torch.concatenate(
            [(coefs[k].square() / Q[k]).flatten() for k in coefs.keys()]  # noqa: SIM118
        ).sum()

    return compute_reg


def qg_dynamical_reg(
    bases: Bases,
    A: torch.Tensor,
    space: SpaceDiscretization2D,
    beta_plane: BetaPlane,
    beta_effect: torch.Tensor,
    T: float,
    *,
    wide: bool,
) -> DynRegFn:
    """Dynamical regularization.

    Args:
        bases (Bases): Bases.
        A (torch.Tensor): Stretching matrix.
        space (SpaceDiscretization2D): Space.
        beta_plane (BetaPlane): Beta plane.
        beta_effect (torch.Tensor): Beta-effect.
        y0 (float): Reference
        T (float): Time scale (L/U).
        wide (bool): Whether to use wide stencil flux computation method or not.

    Returns:
        DynRegFn: Regularization function.
    """  # noqa: E501
    return compute_qg_dynamical_regularization_func(
        bases["dynamical"],
        A,
        space,
        beta_plane,
        beta_effect,
        scale=1 / T**2,
        wide=wide,
    )


def div_dynamical_reg(
    bases: Bases,
    A: torch.Tensor,  # noqa: ARG001
    space: SpaceDiscretization2D,
    beta_plane: BetaPlane,
    beta_effect: torch.Tensor,
    T: float,
    *,
    wide: bool,
) -> DynRegFn:
    """Dynamical regularization.

    Args:
        bases (Bases): Bases.
        A (torch.Tensor): Stretching matrix.
        space (SpaceDiscretization2D): Space.
        beta_plane (BetaPlane): Beta plane.
        beta_effect (torch.Tensor): Beta-effect.
        y0 (float): Reference
        T (float): Time scale (L/U).
        wide (bool): Whether to use wide stencil flux computation method or not.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: Regularization function.
    """  # noqa: E501
    fn = compute_div_dynamical_regularization_func(
        bases["psi"],
        bases["chi"],
        space,
        beta_plane,
        beta_effect,
        scale=1 / T**2,
        wide=wide,
    )
    return lambda psi1, dpsi1, time: fn(psi1, time)  # noqa: ARG005


DYN_FNS_MAPPINGS = {"qg": qg_dynamical_reg, "div": div_dynamical_reg}
