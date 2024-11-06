"""Plot clean Figures."""

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import toml

from qgsw.run_summary import RunSummary

colorscale_bwr = [
    [0, "rgb(0,0,255)"],
    [0.5, "rgb(255,255,255)"],
    [1, "rgb(255,0,0)"],
]

ROOT = Path(__file__).parent.parent

plots_config = toml.load(ROOT.joinpath("config/save_plots.toml"))

for plot_config in plots_config.values():
    field = plot_config["field"]
    layer = plot_config["layer"]

    input_folder = ROOT.joinpath(f"output/g5k/{plot_config['input']}")
    output_folder = ROOT.joinpath(f"output/snapshots/{plot_config['output']}")
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    summary = RunSummary.from_file(input_folder.joinpath("_summary.toml"))
    config = summary.configuration

    for step in plot_config["steps"]:
        file = input_folder.joinpath(f"{config.model.prefix}{step}.npz")

        x_min, x_max = config.space.box.x_min, config.space.box.x_max
        y_min, y_max = config.space.box.y_min, config.space.box.y_max

        data = np.load(file)[field][0, layer, ...]

        colorbar = go.heatmap.ColorBar(
            exponentformat="e",
            showexponent="all",
            title={"text": "Potential Vorticity (s⁻¹)", "side": "right"},
        )

        heatmap = go.Heatmap(
            z=data.T,
            x=np.linspace(x_min / 1000, x_max / 1000, config.space.nx),
            y=np.linspace(y_min / 1000, y_max / 1000, config.space.ny),
            colorscale=px.colors.diverging.RdBu_r,
            zmid=0,
            colorbar=colorbar,
        )

        fig = go.Figure(data=heatmap)

        fig.update_layout(
            template="simple_white",
            autosize=True,
            width=1000,
            height=1000,
            xaxis={"scaleanchor": "y", "constrain": "domain"},
            yaxis={"scaleanchor": "x", "constrain": "domain"},
            font={"family": "Times New Roman", "size": 25},
        )

        fig.update_xaxes(
            title={"text": "X"},
            exponentformat="none",
            ticksuffix=" km",
        )

        fig.update_yaxes(
            title={"text": "Y"},
            exponentformat="none",
            ticksuffix=" km",
        )
        fig.write_image(output_folder.joinpath(f"snapshot_{step}.png"))
