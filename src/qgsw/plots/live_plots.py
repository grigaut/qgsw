"""Live plotting utils."""

from matplotlib import pyplot as plt

from qgsw import plots
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.sv.core import SurfaceVorticity


class LivePlot:
    """Live plot."""

    def __init__(self, model: QGPSIQCore | SurfaceVorticity) -> None:
        """Instantiate live plot.

        Args:
            model (QGPSIQCore | SurfaceVorticity): Model.
        """
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
        else:
            self.state_fig, self.state_axs = plots.subplots(1, 3)
            self.vel_fig, self.vel_axs = plots.subplots(1, 2)
            plots.set_coltitles(
                ["Zonal velocity", "Meridional velocity"], self.vel_axs
            )
            self._plot = self._plot_qg

    def _plot_qg(self, model: QGPSIQCore) -> None:
        plots.imshow(
            model.psi[0, 0],
            ax=self.state_axs[0, 0],
            title="Surface geostrophic stream function",
        )
        plots.imshow(
            model.q[0, 0],
            ax=self.state_axs[0, 1],
            title="Surface potential vorticity",
        )
        plots.imshow(
            model.vorticity[0, 0],
            ax=self.state_axs[0, 2],
            title="Surface vorticity",
        )

        plots.imshow(model.uv[0][0, 0], ax=self.vel_axs[0, 0])
        plots.imshow(model.uv[1][0, 0], ax=self.vel_axs[0, 1])

    def _plot_sv(self, model: SurfaceVorticity) -> None:
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

    def step(self) -> None:
        """Make a step."""
        self._plot(self.model)
        plt.pause(0.05)
