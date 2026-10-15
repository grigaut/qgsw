"""Regularization setups."""

from collections.abc import Callable

import torch

from qgsw import specs
from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.decomposition.vardyn.core import BasisMiost
from qgsw.decomposition.wavelets.core import WaveletBasis
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.scripts.regularization import compute_regularization_func
from qgsw.scripts.setups.types import Bases
from qgsw.spatial.core.discretization import SpaceDiscretization2D


def inline_reg_setup(
    bases: Bases,
    H: torch.Tensor,
    g_prime: torch.Tensor,
    alpha: torch.Tensor,
    space: SpaceDiscretization2D,
    beta_plane: BetaPlane,
    T: float,
) -> dict[
    str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
]:
    """Setup inline regularizations.

    Args:
        bases (dict[ str, Bases ]): Bases.
        H (torch.Tensor): H.
        g_prime (torch.Tensor): g_prime.
        alpha (torch.Tensor): Alpha.
        space (SpaceDiscretization2D): Space.
        beta_plane (BetaPlane): Beta_plane.
        T (float): Time sclae (L/U).

    Returns:
        dict[ str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] ]: Regularization functions mapping.
    """  # noqa: E501
    return {
        k: INLINE_REG_PRESETS[k][basis.type](
            basis, H, g_prime, alpha, space, beta_plane, T
        )
        for k, basis in bases.items()
    }


def post_cycle_reg_setup(
    bases: Bases,
    Qs: dict[str, dict[str, torch.Tensor]],
) -> dict[
    str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
]:
    """Setup post cycle regularization.

    Args:
        bases (Bases): Bases.
        Qs (dict[str, dict[str, torch.Tensor]]): Covariances.

    Returns:
        dict[ str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] ]: Regularization functions mapping.
    """  # noqa: E501
    return {
        k: POST_CYCLE_REG_PRESETS[k][basis.type](Qs[k])
        for k, basis in bases.items()
    }


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
            [(coefs[k].square() / Q[k]) for k in coefs.keys()]  # noqa: SIM118
        ).sum()

    return compute_reg


def background_noreg(
    Q: dict[int, torch.Tensor],  # noqa: ARG001
) -> Callable[[DecompositionCoefs], torch.Tensor]:
    """Backgroung no regularization.

    Args:
        Q (dict[int, torch.Tensor]): Covariances.

    Returns:
        Callable[[DecompositionCoefs], torch.Tensor]: Regularization function.
    """
    return lambda coefs: torch.tensor(0, **coefs.specs)


def dynamical_reg(
    basis: Bases,
    H: torch.Tensor,
    g_prime: torch.Tensor,
    alpha: torch.Tensor,
    space: SpaceDiscretization2D,
    beta_plane: BetaPlane,
    T: float,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Dynamical regularization.

    Args:
        basis (Bases): Basis.
        H (torch.Tensor): H.
        g_prime (torch.Tensor): g_prime.
        alpha (torch.Tensor): Alpha.
        space (SpaceDiscretization2D): Space.
        beta_plane (BetaPlane): Beta plane.
        T (float): Time scale (L/U).

    Returns:
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]: Regularization function.
    """  # noqa: E501
    return compute_regularization_func(
        basis,
        H,
        g_prime,
        alpha,
        space,
        beta_plane,
        scale=1 / T**2,
    )


def dynamical_noreg(
    basis: Bases,  # noqa: ARG001
    H: torch.Tensor,  # noqa: ARG001
    g_prime: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
    space: SpaceDiscretization2D,  # noqa: ARG001
    beta_plane: BetaPlane,  # noqa: ARG001
    T: float,  # noqa: ARG001
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Dynamical no regularization.

    Args:
        basis (Bases): Basis.
        H (torch.Tensor): H.
        g_prime (torch.Tensor): g_prime.
        alpha (torch.Tensor): Alpha.
        space (SpaceDiscretization2D): Space.
        beta_plane (BetaPlane): Beta plane.
        T (float): Time scale (L/U).

    Returns:
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]: Regularization function.
    """  # noqa: E501
    return lambda psi1, dpsi1, t: torch.tensor(0, **specs.from_tensor(psi1))  # noqa: ARG005


INLINE_REG_PRESETS: dict[
    str,
    dict[
        str,
        Callable[
            [
                SpaceTimeDecomposition[
                    SpaceSupportFunction, TimeSupportFunction
                ],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                SpaceDiscretization2D,
                BetaPlane,
                float,
            ],
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ],
    ],
] = {
    "external": {
        BasisMiost.type: dynamical_noreg,
        WaveletBasis.type: dynamical_noreg,
        GaussianExpBasis.type: dynamical_noreg,
    },
    "dynamical": {
        BasisMiost.type: dynamical_noreg,
        WaveletBasis.type: dynamical_reg,
        GaussianExpBasis.type: dynamical_reg,
    },
}
POST_CYCLE_REG_PRESETS: dict[
    str,
    dict[
        str,
        Callable[
            [dict[int, torch.Tensor]],
            Callable[[DecompositionCoefs], torch.Tensor],
        ],
    ],
] = {
    "external": {
        BasisMiost.type: background_reg,
        WaveletBasis.type: background_reg,
        GaussianExpBasis.type: background_reg,
    },
    "dynamical": {
        BasisMiost.type: background_noreg,
        WaveletBasis.type: background_noreg,
        GaussianExpBasis.type: background_noreg,
    },
}
