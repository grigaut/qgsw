"""Regularization setups."""

from collections.abc import Callable

import torch

from qgsw import specs
from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.decomposition.vardyn.core import MIOSTBasis
from qgsw.decomposition.wavelets.core import WaveletBasis
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.scripts.regularization import (
    compute_dynamical_regularization_func,
)
from qgsw.scripts.setups.types import Bases
from qgsw.spatial.core.discretization import SpaceDiscretization2D


def inline_reg_setup(
    bases: Bases,
    A: torch.Tensor,
    space: SpaceDiscretization2D,
    f0: BetaPlane,
    beta_effect: torch.Tensor,
    T: float,
    *,
    wide: bool,
) -> dict[
    str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
]:
    """Setup inline regularizations.

    Args:
        bases (Bases): Bases.
        A (torch.Tensor): Stretching matrix.
        space (SpaceDiscretization2D): Space.
        f0 (float): Coriolis parameter.
        beta_effect (torch.Tensor): Beta-effect.
        y0 (float): Reference
        T (float): Time scale (L/U).
        wide (bool): Whether to use wide stencil flux computation method or not.

    Returns:
        dict[ str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] ]: Regularization functions mapping.
    """  # noqa: E501
    return {
        k: INLINE_REG_PRESETS[k][basis.type](
            basis, A, space, f0, beta_effect, T=T, wide=wide
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
            [(coefs[k].square() / Q[k]).flatten() for k in coefs.keys()]  # noqa: SIM118
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
    bases: Bases,
    A: torch.Tensor,
    space: SpaceDiscretization2D,
    f0: BetaPlane,
    beta_effect: torch.Tensor,
    T: float,
    *,
    wide: bool,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Dynamical regularization.

    Args:
        bases (Bases): Bases.
        A (torch.Tensor): Stretching matrix.
        space (SpaceDiscretization2D): Space.
        f0 (float): Coriolis parameter.
        beta_effect (torch.Tensor): Beta-effect.
        y0 (float): Reference
        T (float): Time scale (L/U).
        wide (bool): Whether to use wide stencil flux computation method or not.

    Returns:
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]: Regularization function.
    """  # noqa: E501
    return compute_dynamical_regularization_func(
        bases,
        A,
        space,
        f0,
        beta_effect,
        scale=1 / T**2,
        wide=wide,
    )


def dynamical_noreg(
    bases: Bases,  # noqa: ARG001
    A: torch.Tensor,  # noqa: ARG001
    space: SpaceDiscretization2D,  # noqa: ARG001
    f0: BetaPlane,  # noqa: ARG001
    beta_effect: torch.Tensor,  # noqa: ARG001
    T: float,  # noqa: ARG001
    *,
    wide: bool,  # noqa: ARG001
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Dynamical no regularization.

    Args:
        bases (dict[ str, Bases ]): Bases.
        A (torch.Tensor): Stretching matrix.
        space (SpaceDiscretization2D): Space.
        f0 (float): Coriolis parameter.
        beta_effect (torch.Tensor): Beta-effect.
        y0 (float): Reference
        T (float): Time scale (L/U).
        wide (bool): Whether to use wide stencil flux computation method or not.

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
        MIOSTBasis.type: dynamical_noreg,
        WaveletBasis.type: dynamical_noreg,
        GaussianExpBasis.type: dynamical_noreg,
    },
    "dynamical": {
        MIOSTBasis.type: dynamical_noreg,
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
        MIOSTBasis.type: background_reg,
        WaveletBasis.type: background_reg,
        GaussianExpBasis.type: background_reg,
    },
    "dynamical": {
        MIOSTBasis.type: background_noreg,
        WaveletBasis.type: background_noreg,
        GaussianExpBasis.type: background_noreg,
    },
}
