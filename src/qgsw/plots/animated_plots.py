"""Base plots."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, TypeVar

import numpy as np
import plotly.colors as pco
import plotly.graph_objects as go
from plotly.subplots import make_subplots

T = TypeVar("T")


class BaseAnimatedPlots(ABC, Generic[T]):
    """Animated Plot base class."""

    def __init__(self, datas: list[list[T]]) -> None:
        """Instantiate the plot.

        Args:
            datas (list[list[T]]): Data list.
        """
        self._datas = datas
        self._check_lengths()
        self._fig = self._create_figure()
        self._add_traces()

    @cached_property
    def n_subplots(self) -> int:
        """NUmber of subplots."""
        return len(self._datas)

    @cached_property
    def n_frames(self) -> int:
        """Number of steps."""
        return len(self._datas[0])

    @cached_property
    def n_cols(self) -> int:
        """Number of columns."""
        return min(3, self.n_subplots)

    @cached_property
    def n_rows(self) -> int:
        """NUmber of rows."""
        return (self.n_subplots - 1) // self.n_cols + 1

    @property
    def figure(self) -> go.Figure:
        """Figure."""
        return self._fig

    def _check_lengths(self) -> None:
        """Check than the data length are matching."""
        if all(len(data) == self.n_frames for data in self._datas):
            return
        msg = "Different lengths for datas."
        raise ValueError(msg)

    def _create_figure(self) -> go.Figure:
        """Create the Figure.

        Returns:
            go.Figure: Figure.
        """
        return make_subplots(rows=self.n_rows, cols=self.n_cols)

    def _create_layout(self) -> go.Layout:
        """Create the layout.

        Returns:
            go.Layout: Layout.
        """
        return go.Layout(
            sliders=[self._create_slider()],
            updatemenus=[
                go.layout.Updatemenu(
                    type="buttons",
                    visible=True,
                    buttons=[
                        go.layout.updatemenu.Button(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {
                                        "duration": 300,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": 500,
                                        "easing": "quadratic-in-out",
                                    },
                                },
                            ],
                        ),
                        go.layout.updatemenu.Button(
                            label="Pause",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top",
                ),
            ],
        )

    def _create_slider(
        self,
    ) -> go.layout.Slider:
        return go.layout.Slider(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=go.layout.slider.Currentvalue(
                font={"size": 20},
                prefix="Frame: ",
                visible=True,
                xanchor="right",
            ),
            transition=go.layout.slider.Transition(
                duration=300,
                easing="cubic-in-out",
            ),
            pad={"b": 10, "t": 50},
            len=0.9,
            x=0.1,
            y=0,
            steps=self._generate_steps(),
        )

    def map_subplot_index_to_subplot_loc(
        self,
        subplot_index: int,
    ) -> tuple[int, int]:
        """Convert subplot index to subplot location.

        Args:
            subplot_index (int): Subplot index.

        Returns:
            tuple[int, int]: Row and column location within the plot.
        """
        return (subplot_index // self.n_cols + 1, subplot_index % 3 + 1)

    def map_subplot_loc_to_subplot_index(self, loc: tuple[int, int]) -> int:
        """Convert subplot location to subplot index.

        Args:
            loc (tuple[int, int]): Row and column location within the plot.

        Returns:
            int: Subplot index.
        """
        row, col = loc
        return (row - 1) * self.n_cols + (col - 1)

    @abstractmethod
    def _add_traces(self) -> None:
        """Initialize the traces."""

    def _generate_frames(self) -> list[go.Frame]:
        """Generate the frames.

        Returns:
            list[dict]: Frames list.
        """
        return [self._compute_frame(frame) for frame in range(self.n_frames)]

    def _generate_steps(self) -> list[go.layout.slider.Step]:
        return [self._compute_step(frame) for frame in range(self.n_frames)]

    @abstractmethod
    def _compute_frame(self, frame_nb: int) -> go.Frame: ...

    @abstractmethod
    def _compute_step(self, frame_nb: int) -> go.Frame: ...

    def show(self) -> None:
        """Show the Figure."""
        self.figure.update(frames=self._generate_frames())
        self.figure.update_layout(self._create_layout())
        self.figure.show()


class AnimatedHeatmaps(BaseAnimatedPlots):
    """Animated Heatmap with shared colorscale."""

    def __init__(
        self,
        datas: list[list[np.ndarray]],
    ) -> None:
        """Instantiate plot."""
        super().__init__(datas)

    def _add_traces(self) -> None:
        """Initialize the traces."""
        for subplot_index in range(self.n_subplots):
            row, col = self.map_subplot_index_to_subplot_loc(subplot_index)
            self.figure.add_trace(
                go.Heatmap(
                    z=self._datas[subplot_index][0],
                    showscale=subplot_index == 0,
                    colorscale=pco.diverging.RdBu_r,
                ),
                row=row,
                col=col,
            )

    def _compute_frame(self, frame_nb: int) -> go.Frame:
        zmax = max(
            np.max(np.abs(data)) for data in [d[frame_nb] for d in self._datas]
        )
        return go.Frame(
            data=[
                go.Heatmap(
                    z=self._datas[subplot_index][frame_nb],
                    colorscale=pco.diverging.RdBu_r,
                    showscale=subplot_index == 0,
                    zmin=-zmax,
                    zmax=zmax,
                )
                for subplot_index in range(self.n_subplots)
            ],
            traces=list(range(self.n_subplots)),
            name=frame_nb,
        )

    def _compute_step(self, frame_nb: int) -> go.Frame:
        return go.layout.slider.Step(
            args=[
                [frame_nb],
                {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            label=frame_nb,
            method="animate",
        )
