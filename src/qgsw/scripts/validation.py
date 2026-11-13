"""Validation utils."""

import torch

from qgsw import specs
from qgsw.metrics.spectral import (
    MagnitudeSquaredCoherence,
    PowerSpectralDensity,
)
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.sv.core import SurfaceVorticity
from qgsw.scripts.loss import rmse, uv_rmse
from qgsw.spatial.core.discretization import SpaceDiscretization2D


class Losses:
    """Losses handler."""

    cycle = -1
    no_data = False

    def __init__(self, space: SpaceDiscretization2D) -> None:
        """Instantiate the handler."""
        self.loss_fn = {
            "rmse": rmse,
        }
        self.uv_loss_fn = {"uv_rmse": uv_rmse}
        self.psi2_loss_fn = {"psi2_rmse": rmse}
        psd_ = PowerSpectralDensity(
            nx=space.psi.nx,
            ny=space.psi.ny,
            dx=space.psi.dx.item(),
            dy=space.psi.dy.item(),
        )

        def psd_err(f: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
            return psd_.iso_spec_w(f - f_ref) / psd_.iso_spec_w(f_ref)

        self.loss_fn["psd_err"] = psd_err
        msc_ = MagnitudeSquaredCoherence(
            nx=space.psi.nx,
            ny=space.psi.ny,
            dx=space.psi.dx.item(),
            dy=space.psi.dy.item(),
        )

        def msc(f: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
            return msc_.iso_spec_w(f, f_ref)

        self.loss_fn["msc"] = msc
        self.losses: dict[str, torch.Tensor] = {}

        self.losses_names = list(self.loss_fn.keys())
        self.psi2_losses_names = []
        self.uv_losses_names = []

        for name in self.losses_names:
            self.losses[name] = None

    def with_psi2(self) -> None:
        """Add psi2 losses."""
        self.psi2_losses_names = list(self.psi2_loss_fn.keys())
        for name in self.psi2_losses_names:
            self.losses[name] = None

    def with_uv(self) -> None:
        """Add UV losses."""
        self.uv_losses_names = list(self.uv_loss_fn.keys())
        for name in self.uv_losses_names:
            self.losses[name] = None

    def add_loss(self, loss_value: torch.Tensor, loss_name: str) -> None:
        """Add loss.

        Args:
            loss_value (torch.Tensor): Loss value.
            loss_name (str): Loss name.
        """
        if self.losses[loss_name] is None:
            self.losses[loss_name] = loss_value.cpu()[None, ...]
            return
        t = self.losses[loss_name]
        self.losses[loss_name] = torch.cat(
            [t, loss_value.cpu()[None, ...]], dim=0
        )

    def compute_loss(self, model: QGPSIQ, psi_ref: torch.Tensor) -> None:
        """Compute loss."""
        for loss_name in self.losses_names:
            if self.no_data:
                self.add_loss(
                    torch.tensor(torch.nan, **specs.from_tensor(model.A)),
                    loss_name,
                )
                return
            self.add_loss(
                self.loss_fn[loss_name](model.psi[0, 0], psi_ref[0, 0]),
                loss_name,
            )

    def compute_uv_losses(
        self,
        model: QGPSIQ | SurfaceVorticity,
        u_ref: torch.Tensor,
        v_ref: torch.Tensor,
    ) -> None:
        """Compute uv loss."""
        for loss_name in self.uv_losses_names:
            if self.no_data:
                self.add_loss(
                    torch.tensor(torch.nan, **specs.from_tensor(model.A)),
                    loss_name,
                )
                return
            self.add_loss(
                self.uv_loss_fn[loss_name](
                    model.uv[0][0, 0],
                    model.uv[1][0, 0],
                    u_ref[0, 0],
                    v_ref[0, 0],
                ),
                loss_name,
            )

    def compute_psi2_losses(
        self, model: QGPSIQ | SurfaceVorticity, psi_ref: torch.Tensor
    ) -> None:
        """Compute psi2 loss."""
        for loss_name in self.psi2_losses_names:
            no_data = self.no_data or (not hasattr(model, "psi2"))

            if no_data:
                self.add_loss(
                    torch.tensor(torch.nan, **specs.from_tensor(model.A)),
                    loss_name,
                )
                return

            psi2: torch.Tensor = model.psi2
            self.add_loss(
                self.psi2_loss_fn[loss_name](psi2[0, 0], psi_ref[0, 1]),
                loss_name,
            )
