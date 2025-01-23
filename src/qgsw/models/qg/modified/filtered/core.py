"""Modified QG model with filtered top layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw import verbose
from qgsw.fields.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    PhysicalSurfaceHeightAnomaly,
    Pressure,
)
from qgsw.fields.variables.uvh import UVH
from qgsw.filters.high_pass import GaussianHighPass2D
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.modified.collinear_sublayer.core import QGAlpha
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    keep_top_layer,
)

if TYPE_CHECKING:
    from qgsw.fields.variables.state import State
    from qgsw.spatial.core.discretization import (
        SpaceDiscretization2D,
    )


class QGCollinearFilteredSF(QGAlpha):
    """Modified QG Model implementing collinear pv behavior."""

    _type = "QGCollinearFilteredSF"

    _supported_layers_nb: int = 2

    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Collinear Sublayer Stream Function.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        verbose.display(
            msg=f"Creating {self.__class__.__name__} model...",
            trigger_level=1,
        )
        ModelParamChecker.__init__(
            self,
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
        )
        self._space = keep_top_layer(self._space)
        ##Topography and Ref values
        self._set_ref_variables()

        # initialize state
        self._set_state()
        # initialize variables
        self._create_diagnostic_vars(self._state)

        self._set_utils(optimize)
        self._set_fluxes(optimize)
        self._core = self._init_core_model(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            optimize=optimize,
        )
        self._filt = GaussianHighPass2D(10)
        self.A = self.compute_A(H, g_prime)
        self._core = self._init_core_model(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            optimize=optimize,
        )
        decomposition = compute_layers_to_mode_decomposition(self.A)
        self.Cm2l, lambd, self.Cl2m = decomposition
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))

    def _create_diagnostic_vars(self, state: State) -> None:
        super()._create_diagnostic_vars(state)
        h_phys = PhysicalLayerDepthAnomaly(ds=self.space.ds)
        eta_phys = PhysicalSurfaceHeightAnomaly(h_phys=h_phys)
        p = Pressure(
            g_prime=self.g_prime[:1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            eta_phys=eta_phys,
        )
        p.bind(self._state)

    def project(self, uvh: UVH) -> UVH:
        """QG projector P = G o (Q o G)^{-1} o Q.

        Args:
            uvh (UVH): u,v and h.

        Returns:
            UVH: Quasi geostrophic u,v and h
        """
        p = self._state[Pressure.get_name()].compute_no_slice(uvh)
        p_toplayer = torch.nn.functional.pad(p[:, 0, ...], (1, 1, 1, 1))
        p_sublayer = self._filt(p_toplayer[0, 0]).unsqueeze(0)
        p = torch.stack([p_toplayer, self.alpha * p_sublayer], dim=1)
        new_uvh = self.G(p)
        projected = super().project(new_uvh)
        return UVH(
            projected.u[:, 0:1, ...],
            projected.v[:, 0:1, ...],
            projected.h[:, 0:1, ...],
        )
