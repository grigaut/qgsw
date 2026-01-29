"""Base class for observations systems."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage
from matplotlib.text import Annotation, Text

from qgsw import plots
from qgsw.logging.utils import sec2text
from qgsw.plots.plt_wrapper import retrieve_imshow_data
from qgsw.specs import defaults


class BaseObservationMask(ABC):
    """Base class for observations systems."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Instantiate the observation system.

        Args:
            x (torch.Tensor): X locations.
                └── (nx, ny) shaped
            y (torch.Tensor): Y locations.
                └── (nx, ny) shaped
        """
        self._validate_xy(x, y)
        self._x = x
        self._y = y

    def _validate_xy(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Validate x and y shapes.

        Args:
            x (torch.Tensor): X locations.
                └── (nx, ny) shaped
            y (torch.Tensor): Y locations.
                └── (nx, ny) shaped

        Raises:
            ValueError: If x is not 2D.
            ValueError: If y is not 2D.
            ValueError: If x an dy shapes do not match.
        """
        sx = x.shape
        sy = y.shape
        if len(sx) != 2:
            msg = "x must be a 2D tensor."
            raise ValueError(msg)
        if len(sy) != 2:
            msg = "y must be a 2D tensor."
            raise ValueError(msg)
        if sx != sy:
            msg = "x and y must have the same shape."
            raise ValueError(msg)

    @abstractmethod
    def at_time(self, time: torch.Tensor) -> torch.Tensor:
        """Compute observation mask at given time..

        Args:
            time (torch.Tensor): Time of the observation.

        Returns:
            torch.Tensor: Mask.
                └── (nx, ny) shaped
        """

    def visualize(
        self,
        output: Path,
        duration: float = 20 * 3600 * 24,
        *,
        frame_dt: float = 3600,
    ) -> None:
        """Visualize observation masks.

        Args:
            output (Path): Output file (.mp4)
            duration (float, optional): Total simulated time.
                Defaults to 20*3600*24.
            frame_dt (float, optional): Duration of a frame.
                Defaults to 3600.
        """
        nb_ite = int(duration // frame_dt) + 1

        masks = []
        coverages: list[torch.Tensor] = []

        for i in range(nb_ite):
            time = i * frame_dt
            mask = self.at_time(torch.tensor([time], **defaults.get()))
            masks.append(mask)
            if i == 0:
                coverages.append(mask.clone().to(torch.int64))
            else:
                coverages.append(coverages[-1] + mask.to(torch.int64))

        fig, axs = plots.subplots(1, 3)
        title_txt = "Observation mask at {time}"
        title = plots.blittable_suptitle(
            title_txt.format(time=sec2text(0)),
            fig,
            axs[0, 0],
        )

        cov_txt = "Coverage: {cov:.0%}"
        coltitles = plots.set_coltitles(
            ["Track Mask", cov_txt.format(cov=0), "Cumulative tracks"],
            axs=axs,
        )
        im1 = plots.imshow(
            masks[0],
            ax=axs[0, 0],
            vmin=0,
            vmax=1,
            cmap="Greys",
        )
        im2 = plots.imshow(
            coverages[0] > 0,
            ax=axs[0, 1],
            vmin=0,
            vmax=1,
            cmap="Greys",
        )
        vmax = coverages[-1].max().item()
        im3 = plots.imshow(
            coverages[0],
            ax=axs[0, 2],
            vmin=0,
            vmax=vmax,
            cmap="Greys",
        )

        def update(
            frame: int,
        ) -> tuple[AxesImage, AxesImage, AxesImage, Annotation, Text]:
            im1.set_array(retrieve_imshow_data(masks[frame]))
            im2.set_array(retrieve_imshow_data(coverages[frame] > 0))
            im3.set_array(retrieve_imshow_data(coverages[frame]))
            coverage = (coverages[frame] > 0).to(torch.float64).mean().item()
            coltitles[1].set_text(cov_txt.format(cov=coverage))
            title.set_text(title_txt.format(time=sec2text(frame * frame_dt)))
            return (im1, im2, im3, coltitles[1], title)

        anim = FuncAnimation(fig, update, frames=nb_ite, blit=True)
        anim.save(output, fps=20)
        plots.close(fig)
