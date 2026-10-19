"""QG model wrapper base classes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt

from qgsw import specs
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.core import build_basis_from_params_dict
from qgsw.metrics.spectral import (
    MagnitudeSquaredCoherence,
    PowerSpectralDensity,
)
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.scripts.loss import rmse, uv_rmse
from qgsw.scripts.output import OutputFolder, build_output_folder
from qgsw.spatial.core.discretization import SpaceDiscretization2D

T = TypeVar("T", bound=QGPSIQCore)


class ModelWrapper(ABC, Generic[T]):
    """Model wrapper."""

    model: T
    color: str = "black"
    linestyle: str = "solid"
    cycle = -1
    save_params = False
    no_data = False

    show = True
    remember_psis = False

    _dt: float
    no_wind = True

    @property
    def plot_kwargs(self) -> dict[str, Any]:
        """Plotting kwargs."""
        return {
            "color": self.color,
            "linestyle": self.linestyle,
            "label": self.label,
        }

    @property
    def output(self) -> OutputFolder:
        """Output folder."""
        return self._output

    @property
    def dt(self) -> int:
        """Time step."""
        return self._dt

    @dt.setter
    def dt(self, value: int) -> None:
        if (v := self.data_dt // value) * value - (d := self.data_dt) != 0:
            msg = f"Data dt {d} is not compatible with model dt {value}"
            raise ValueError(msg)
        self._substep_ratio = v
        self._dt = value

    @property
    def label(self) -> str:
        """Model label."""
        try:
            return self._label
        except AttributeError:
            return self.model_type

    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    @property
    def substep_ratio(self) -> int:
        """Substep ratio."""
        return self._substep_ratio

    def __init__(
        self, output_folder: Path | str | None, data_dt: float
    ) -> None:
        """Instantiate model wrapper.

        Args:
            output_folder (Path | str | None): Output folder.
            data_dt (float): Data timestep.
        """
        self.data_dt = data_dt
        if output_folder is not None:
            self._output = build_output_folder(output_folder)
            self.parse_params(self._output)
        else:
            self._output = None
            self.model_type = "RG0"
        self.losses: dict[str, list[list[torch.Tensor]]] = {}

    def parse_params(self, output_folder: OutputFolder) -> dict[str, Any]:
        """Parse parameters.

        Args:
            output_folder (OutputFolder): Outputs.
        """
        params = output_folder.params.load()
        self.model_type = params["model_type"]
        self.dt = params["dt"]
        self.no_wind = params["no_wind"]
        return params

    @abstractmethod
    def init_model(self, space: SpaceDiscretization2D) -> None:
        """Instantiate model.

        Args:
            space (SpaceDiscretization2D): Space.
        """

    def set_wind_forcing(self, tx: torch.Tensor, ty: torch.Tensor) -> None:
        """Set wind forcing."""
        if self.no_wind:
            return
        self.model.set_wind_forcing(tx, ty)

    def set_basis_external(self, bases_params: dict, results: dict) -> None:
        """Set basis for external forcing.

        Args:
            bases_params (dict): Basis parameters.
            results (dict): Results dict.
        """
        self.basis_external = build_basis_from_params_dict(
            bases_params["external"]
        )
        coefs = DecompositionCoefs.from_dict(results["external"])
        self.basis_external.set_coefs(coefs)
        self.model.basis_external = self.basis_external

    def set_basis_dynamical(self, bases_params: dict, results: dict) -> None:
        """Set basis for dynamical forcing.

        Args:
            bases_params (dict): Basis parameters.
            results (dict): Results dict.
        """
        self.basis_dynamical = build_basis_from_params_dict(
            bases_params["dynamical"]
        )
        coefs = DecompositionCoefs.from_dict(results["dynamical"])
        self.basis_dynamical.set_coefs(coefs)
        self.model.basis_dynamical = self.basis_dynamical

    def set_alpha(self, results: dict) -> None:
        """Set alpha."""
        try:
            self.alpha = results["alpha"]
        except KeyError:
            self.alpha = torch.tensor(0, **specs.from_tensor(self.model.A))
        self.model.alpha = self.alpha

    def set_model_params(self) -> dict[str, Any]:
        """Set model parameters."""
        bases_params = self.output.basis.load()[self.cycle]
        results = self.output.results.load()[self.cycle]
        if self.model.get_type() == QGPSIQForced.get_type():
            self.set_basis_external(bases_params, results)
        elif self.model.get_type() == QGPSIQRGPsi2TransportDR.get_type():
            self.set_alpha(results)
            self.set_basis_dynamical(bases_params, results)
        elif self.model.get_type() == QGPSIQForcedSurfML.get_type():
            self.set_alpha(results)
            self.set_basis_dynamical(bases_params, results)
            self.set_basis_external(bases_params, results)
        return results

    def setup(
        self,
        sfs: torch.Tensor | xr.DataArray,  # noqa: ARG002
        times: list[torch.Tensor],  # noqa: ARG002
    ) -> None:
        """Setup model cycle.

        Args:
            sfs (torch.Tensor | xr.DataArray): Stream functions.
            times (list[torch.Tensor]): Times values.
        """
        if self._output is not None:
            nb_cycles = len(self.output.optim.load())
            if self.cycle >= nb_cycles:
                self.no_data = True
                msg = f"No data for cycle {self.cycle}."
                raise KeyError(msg)
            self.set_model_params()

    def new_cycle(self) -> None:
        """New cycle."""
        self.cycle += 1
        for loss in self.losses.values():
            loss.append([])
        if self.remember_psis:
            self.psis = []

    def add_loss(self, loss_value: torch.Tensor, loss_name: str) -> None:
        """Add loss.

        Args:
            loss_value (torch.Tensor): Loss value.
            loss_name (str): Loss name.
        """
        self.losses[loss_name][-1].append(loss_value)

    def plot_loss(
        self,
        *,
        loss_name: str,
        ax: plt.Axes | None = None,
        cycle: int | None = None,
    ) -> None:
        """Plot loss.

        Args:
            loss_name (str): Loss name.
            ax (plt.Axes | None, optional): Ax to plot to. Defaults to None.
            cycle (int | None, optional): Cycle to plot. Defaults to None.
        """
        if not self.show:
            return
        if ax is None:
            ax = plt.gca()
        cycles = (
            [cycle]
            if cycle is not None
            else list(range(len(self.losses[loss_name])))
        )
        time_offset = 0
        for i, c in enumerate(cycles):
            steps = np.arange(len(self.losses[loss_name][c]))
            times = self.data_dt * steps / 3600 / 24 + time_offset
            time_offset = times[-1] + self.data_dt / 3600 / 24
            loss = torch.stack(self.losses[loss_name][c]).cpu().numpy()
            kwargs = self.plot_kwargs.copy()
            if i != 0:
                kwargs.pop("label")
            ax.plot(times, loss, **kwargs)
        ax.set_xlabel("Time [day]")

    def step(self) -> None:
        """Make a step."""
        if self.no_data:
            return
        for _ in range(self.substep_ratio):
            self.model.step()
        if self.remember_psis:
            self.psis.append(self.model.psi.clone())


M = TypeVar("M", bound=ModelWrapper[QGPSIQCore])


class ModelsManager(Generic[M]):
    """Models manager."""

    losses: list[str]

    def __init__(self, *mw: M) -> None:
        """Instantiate the manager."""
        self.model_wrappers: tuple[M] = mw

        self.save_params = self.model_wrappers[0].save_params
        self.loop_over_models(lambda mw: setattr(mw, "tracked", True))

        self.loss_fn: dict[
            str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = {
            "rmse": rmse,
        }
        self.losses = list(self.loss_fn.keys())

        for name in self.losses:
            self.loop_over_models(
                lambda mw, name=name: mw.losses.update({name: []})
            )

        self.uv_loss_fn: dict[
            str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = {}
        self.uv_losses = list(self.uv_loss_fn.keys())

        for name in self.uv_losses:
            self.loop_over_models(
                lambda mw, name=name: mw.losses.update({name: []})
            )

        self.sst_loss_fn: dict[
            str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = {}
        self.sst_losses = list(self.sst_loss_fn.keys())

        for name in self.sst_losses:
            self.loop_over_models(
                lambda mw, name=name: mw.losses.update({name: []})
            )

    @property
    def save_params(self) -> bool:
        """Whether to save model parameters."""
        return self.model_wrappers[0].save_params

    @save_params.setter
    def save_params(self, save_params: bool) -> None:
        self.loop_over_models(
            lambda mw: setattr(mw, "save_params", save_params)
        )

    def add_PSDE(  # noqa: N802
        self, space: SpaceDiscretization2D, name: str = "psd_err"
    ) -> None:
        """Add PSDE.

        Args:
            space (SpaceDiscretization2D): Space.
            name (str): Error name.
        """
        self.psd = PowerSpectralDensity(
            nx=space.psi.nx,
            ny=space.psi.ny,
            dx=space.psi.dx.item(),
            dy=space.psi.dy.item(),
        )

        def psd_err(f: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
            return self.psd.iso_spec_w(f - f_ref) / self.psd.iso_spec_w(f_ref)

        self.loss_fn[name] = psd_err

        self.losses = list(self.loss_fn.keys())

        self.loop_over_models(lambda mw: mw.losses.update({name: []}))

    def add_MSC(self, space: SpaceDiscretization2D, name: str = "msc") -> None:  # noqa: N802
        """Add MSC.

        Args:
            space (SpaceDiscretization2D): Space.
            name (str): Error name.
        """
        self.msc = MagnitudeSquaredCoherence(
            nx=space.psi.nx,
            ny=space.psi.ny,
            dx=space.psi.dx.item(),
            dy=space.psi.dy.item(),
        )

        def msc(f: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
            return self.msc.iso_spec_w(f, f_ref)

        self.loss_fn[name] = msc

        self.losses = list(self.loss_fn.keys())

        self.loop_over_models(lambda mw: mw.losses.update({name: []}))

    def add_gradient_RMSE(  # noqa: N802
        self, space: SpaceDiscretization2D, name: str = "uv_rmse"
    ) -> None:
        """Add gradient RMSE.

        Args:
            space (SpaceDiscretization2D): Space.
            name (str): Error name.
        """
        self.uv_loss_fn[name] = lambda psi, u, v: uv_rmse(
            psi, u, v, space.dx, space.dy
        )
        self.uv_losses = list(self.uv_loss_fn.keys())
        self.loop_over_models(lambda mw: mw.losses.update({name: []}))

    def init_models(self, space: SpaceDiscretization2D) -> None:
        """Instantiate models."""
        self.loop_over_models(lambda mw: mw.init_model(space))

    def loop_over_models(self, func: Callable[[M], None]) -> None:
        """Iterate function over all model wrappers."""
        for mw in self.model_wrappers:
            func(mw)

    def step(self) -> None:
        """Perform a model step."""
        self.loop_over_models(lambda mw: mw.step())

    def new_cycle(self) -> None:
        """Start new cycle."""
        self.loop_over_models(lambda mw: mw.new_cycle())

    def reset_time(self) -> None:
        """Reset all model times."""
        self.loop_over_models(lambda mw: mw.model.reset_time())

    def set_wind_forcing(self, tx: torch.Tensor, ty: torch.Tensor) -> None:
        """Set wind forcing."""
        self.loop_over_models(lambda mw: mw.set_wind_forcing(tx, ty))

    def remember_psis(self, *, mode: bool = True) -> None:
        """Save psis."""
        for mw in self.model_wrappers:
            mw.remember_psis = mode

    def compute_loss(self, psi_ref: torch.Tensor) -> None:
        """Compute loss."""
        for loss_name in self.losses:

            def compute(mw: M, loss_name: str) -> None:
                if mw.no_data:
                    mw.add_loss(
                        torch.tensor(
                            torch.nan, **specs.from_tensor(mw.model.A)
                        ),
                        loss_name,
                    )
                    return
                mw.add_loss(
                    self.loss_fn[loss_name](mw.model.psi[0, 0], psi_ref[0, 0]),
                    loss_name,
                )

            self.loop_over_models(
                lambda mw, loss_name=loss_name: compute(mw, loss_name)
            )

    def compute_uv_losses(
        self, u_ref: torch.Tensor, v_ref: torch.Tensor
    ) -> None:
        """Compute velocity losses."""
        for loss_name in self.uv_losses:

            def compute(mw: M, loss_name: str) -> None:
                if mw.no_data:
                    mw.add_loss(
                        torch.tensor(
                            torch.nan, **specs.from_tensor(mw.model.A)
                        ),
                        loss_name,
                    )
                    return
                mw.add_loss(
                    self.uv_loss_fn[loss_name](
                        mw.model.psi[0, 0], u_ref[0, 0], v_ref[0, 0]
                    ),
                    loss_name,
                )

            self.loop_over_models(
                lambda mw, loss_name=loss_name: compute(mw, loss_name)
            )

    def compute_sst_losses(self, sst_ref: torch.Tensor) -> None:
        """Compute velocity losses."""
        for loss_name in self.sst_losses:

            def compute(mw: M, loss_name: str) -> None:

                no_data = mw.no_data or hasattr(mw.model, "sst")

                if no_data:
                    mw.add_loss(
                        torch.tensor(
                            torch.nan, **specs.from_tensor(mw.model.A)
                        ),
                        loss_name,
                    )
                    return

                sst: torch.Tensor = mw.model.sst
                mw.add_loss(
                    self.sst_loss_fn[loss_name](sst[0, 0], sst_ref[0, 0]),
                    loss_name,
                )

            self.loop_over_models(
                lambda mw, loss_name=loss_name: compute(mw, loss_name)
            )

    def plot_loss(
        self,
        *,
        loss_name: str,
        ax: plt.Axes | None = None,
        cycle: int | None = None,
    ) -> None:
        """Plot loss.

        Args:
            loss_name (str): Loss name.
            ax (plt.Axes | None, optional): Axes to plot to. Defaults to None.
            cycle (int | None, optional): Cycle to plot. Defaults to None.
        """
        self.loop_over_models(
            lambda mw: mw.plot_loss(loss_name=loss_name, ax=ax, cycle=cycle)
        )

    def setup(self, sfs: xr.DataArray, times: list[torch.Tensor]) -> None:
        """Setup all models."""
        self.loop_over_models(lambda mw: mw.setup(sfs, times))
