"""Live plotting utils."""

from matplotlib import pyplot as plt

from qgsw import plots
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.models.sv.core import SurfaceVorticity


class LivePlot:
    """Live plot."""

    def __init__(
        self,
        model: QGPSIQCore | SurfaceVorticity,
        *,
        active: bool = False,
    ) -> None:
        """Instantiate live plot.

        Args:
            model (QGPSIQCore | SurfaceVorticity): Model.
            active (bool): Whether the plot is actove or not.
        """
        if not active:
            self._plot = self._no_plot
            self._close = self._no_plot
            return
        if not plt.isinteractive():
            plt.ion()

        self.model = model

        if model.get_type() == SurfaceVorticity.get_type():
            self.state_fig, self.state_axs = plots.subplots(4, 2)
            plots.set_coltitles(["Field", "Laplacian"], self.state_axs)
            plots.set_rowtitles(
                ["ѱ_g + ѱ_a ", "ѱ_g", "ѱ_a", "χ_a"], self.state_axs
            )
            self.vel_fig, self.vel_axs = plots.subplots(3, 2)
            plots.set_coltitles(
                ["Zonal velocity", "Meridional velocity"], self.vel_axs
            )
            plots.set_rowtitles(["uv_g + uv_a ", "uv_g", "uv_a"], self.vel_axs)
            self._plot = self._plot_sv
        elif model.get_type() == QGPSIQRGPsi2TransportDR.get_type():
            self.state_fig, self.state_axs = plots.subplots(1, 4)
            plots.set_coltitles(["ѱ_1", "q", "ζ_1", "ѱ_2"], self.state_axs)
            self.vel_fig, self.vel_axs = plots.subplots(1, 2)
            plots.set_coltitles(
                ["Zonal velocity", "Meridional velocity"], self.vel_axs
            )
            self._plot = self._plot_rgsi
        elif model.get_type() == QGPSIQForced.get_type():
            self.state_fig, self.state_axs = plots.subplots(1, 4)
            plots.set_coltitles(["ѱ_1", "q", "ζ_1", "f"], self.state_axs)
            self.vel_fig, self.vel_axs = plots.subplots(1, 2)
            plots.set_coltitles(
                ["Zonal velocity", "Meridional velocity"], self.vel_axs
            )
            self._plot = self._plot_rgf
        elif model.get_type() == QGPSIQForcedSurfML.get_type():
            self.state_fig, self.state_axs = plots.subplots(1, 5)
            plots.set_coltitles(
                ["ѱ_1", "q", "ζ_1", "ѱ_2", "f"], self.state_axs
            )
            self.vel_fig, self.vel_axs = plots.subplots(1, 2)
            plots.set_coltitles(
                ["Zonal velocity", "Meridional velocity"], self.vel_axs
            )
            self._plot = self._plot_rgfsi
        else:
            self.state_fig, self.state_axs = plots.subplots(1, 3)
            self.vel_fig, self.vel_axs = plots.subplots(1, 2)
            plots.set_coltitles(
                ["Zonal velocity", "Meridional velocity"], self.vel_axs
            )
            self._plot = self._plot_qg

    def _plot_qg(self) -> None:
        model: QGPSIQCore = self.model
        plots.imshow(
            model.psi[0, 0],
            ax=self.state_axs[0, 0],
        )
        plots.imshow(
            model.q[0, 0],
            ax=self.state_axs[0, 1],
        )
        plots.imshow(
            model.vorticity[0, 0],
            ax=self.state_axs[0, 2],
        )

        plots.imshow(model.uv[0][0, 0], ax=self.vel_axs[0, 0])
        plots.imshow(model.uv[1][0, 0], ax=self.vel_axs[0, 1])
        plt.pause(0.05)

    def _plot_rgsi(self) -> None:
        model: QGPSIQRGPsi2TransportDR = self.model
        plots.imshow(
            model.psi[0, 0],
            ax=self.state_axs[0, 0],
        )
        plots.imshow(
            model.q[0, 0],
            ax=self.state_axs[0, 1],
        )
        plots.imshow(
            model.vorticity[0, 0],
            ax=self.state_axs[0, 2],
        )
        plots.imshow(
            model.psi2[0, 0],
            ax=self.state_axs[0, 3],
        )

        plots.imshow(model.uv[0][0, 0], ax=self.vel_axs[0, 0])
        plots.imshow(model.uv[1][0, 0], ax=self.vel_axs[0, 1])
        plt.pause(0.05)

    def _plot_rgf(self) -> None:
        model: QGPSIQForced = self.model
        plots.imshow(
            model.psi[0, 0],
            ax=self.state_axs[0, 0],
        )
        plots.imshow(
            model.q[0, 0],
            ax=self.state_axs[0, 1],
        )
        plots.imshow(
            model.vorticity[0, 0],
            ax=self.state_axs[0, 2],
        )
        plots.imshow(
            model.forcing[0, 0],
            ax=self.state_axs[0, 3],
        )

        plots.imshow(model.uv[0][0, 0], ax=self.vel_axs[0, 0])
        plots.imshow(model.uv[1][0, 0], ax=self.vel_axs[0, 1])
        plt.pause(0.05)

    def _plot_rgfsi(self) -> None:
        model: QGPSIQForcedSurfML = self.model
        plots.imshow(
            model.psi[0, 0],
            ax=self.state_axs[0, 0],
        )
        plots.imshow(
            model.q[0, 0],
            ax=self.state_axs[0, 1],
        )
        plots.imshow(
            model.vorticity[0, 0],
            ax=self.state_axs[0, 2],
        )
        plots.imshow(
            model.psi2[0, 0],
            ax=self.state_axs[0, 3],
        )
        plots.imshow(
            model.forcing[0, 0],
            ax=self.state_axs[0, 4],
        )

        plots.imshow(model.uv[0][0, 0], ax=self.vel_axs[0, 0])
        plots.imshow(model.uv[1][0, 0], ax=self.vel_axs[0, 1])
        plt.pause(0.05)

    def _plot_sv(self) -> None:
        model: SurfaceVorticity = self.model
        plots.imshow(
            model.psi[0, 0],
            ax=self.state_axs[0, 0],
        )
        plots.imshow(
            model.zeta[0, 0],
            ax=self.state_axs[0, 1],
        )
        plots.imshow(
            model.psi_g[0, 0],
            ax=self.state_axs[1, 0],
        )
        plots.imshow(
            model.zeta_g[0, 0],
            ax=self.state_axs[1, 1],
        )
        plots.imshow(
            model.psi_a[0, 0],
            ax=self.state_axs[2, 0],
        )
        plots.imshow(
            model.zeta_a[0, 0],
            ax=self.state_axs[2, 1],
        )
        plots.imshow(
            model.chi_a[0, 0],
            ax=self.state_axs[3, 0],
        )
        plots.imshow(
            model.div[0, 0],
            ax=self.state_axs[3, 1],
        )

        plots.imshow(
            model.uv[0][0, 0],
            ax=self.vel_axs[0, 0],
        )
        plots.imshow(
            model.uv[1][0, 0],
            ax=self.vel_axs[0, 1],
        )
        plots.imshow(
            model.uv_g[0][0, 0],
            ax=self.vel_axs[1, 0],
        )
        plots.imshow(
            model.uv_g[1][0, 0],
            ax=self.vel_axs[1, 1],
        )
        plots.imshow(
            model.uv_a[0][0, 0],
            ax=self.vel_axs[2, 0],
        )
        plots.imshow(
            model.uv_a[1][0, 0],
            ax=self.vel_axs[2, 1],
        )
        plt.pause(0.05)

    def _no_plot(self) -> None:
        return

    def step(self) -> None:
        """Make a step."""
        self._plot()

    def _close(self) -> None:
        """Close figure."""
        plt.close()

    close = _close
