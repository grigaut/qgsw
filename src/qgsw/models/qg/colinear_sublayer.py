"""Modified QG Model with Colinear Sublayer Behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.models.exceptions import InvalidLayersDefinitionError
from qgsw.models.qg.core import QG
from qgsw.models.sw import SW
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
    keep_top_layer,
)

if TYPE_CHECKING:
    from qgsw.physics.coriolis.beta_plane import BetaPlane


class QGColinearSublayerStreamFunction(QG):
    """Modified QG model implementing CoLinear Sublayer Behavior."""

    _supported_layers_nb: int = 2
    alpha = 0.01

    def __init__(
        self,
        space_3d: SpaceDiscretization3D,
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Colinear Sublayer Stream Function.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(
            space_3d=space_3d,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=optimize,
        )

    @property
    def H(self) -> torch.Tensor:  # noqa: N802
        """Layers thickness."""
        return self._H[0, ...].unsqueeze(0)

    @property
    def g_prime(self) -> torch.Tensor:
        """Reduced Gravity."""
        return self._g_prime[0, ...].unsqueeze(0)

    def _set_H(self, h: torch.Tensor) -> torch.Tensor:  # noqa: N802
        """Perform additional validation over H.

        Args:
            h (torch.Tensor): Layers thickness.

        Raises:
            ValueError: if H is not constant in space

        Returns:
            torch.Tensor: H
        """
        if self.space.nl != self._supported_layers_nb:
            msg = (
                "QGColinearSublayer can only support"
                f"{self._supported_layers_nb} layers."
            )
            raise InvalidLayersDefinitionError(msg)
        super()._set_H(h)

    def _init_core_model(self, optimize: bool) -> SW:  # noqa: FBT001
        """Initialize the core Shallow Water model.

        Args:
            optimize (bool): Wehether to optimize the model functions or not.

        Returns:
            SW: Core model.
        """
        return SW(
            space_3d=keep_top_layer(self._space),
            g_prime=self.g_prime,
            beta_plane=self.beta_plane,
            optimize=optimize,
        )

    def compute_A(  # noqa: N802
        self,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
    ) -> torch.Tensor:
        """Compute new Stretching operator.

        Ã = (1/ρ_1)[[(1/H_1)*(1/g_1 + (1 - α)/g_2)]]

        Args:
            H (torch.Tensor): Layers reference height.
            g_prime (torch.Tensor): Reduced gravity values.

        Returns:
            torch.Tensor: Stretching Operator
        """  # noqa: RUF002
        A = super().compute_A(H, g_prime)  # noqa: N806
        # Create layers coefficients vector [1, α]  # noqa: RUF003
        layers_coefs = torch.tensor(
            [1, self.alpha],
            dtype=self.dtype,
            device=self.device,
        )
        # Select top row from matrix product
        return (A @ layers_coefs)[0, ...].unsqueeze(0).unsqueeze(0)
