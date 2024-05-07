"""Plot Grid5000 outputs."""

import os
from pathlib import Path

from dotenv import load_dotenv
from qgsw.plots.vorticity import (
    SecondLayerVorticityAxes,
    SurfaceVorticityAxes,
    VorticityComparisonFigure,
)
from qgsw.run_summary import RunSummary
from qgsw.utils.sorting import sort_files

load_dotenv()

storage = Path(os.environ["G5K_IMPORT_STORAGE"])
folder = storage.parent.joinpath("imports")

summary = RunSummary.from_file(folder.joinpath("_summary.toml"))

tau = summary.total_steps / summary.configuration.simulation.duration

axes = []
nbs = []
files = []
for model in summary.configuration.models:
    results = list(folder.glob(f"{model.prefix}*.npz"))
    ax = SurfaceVorticityAxes.from_mask()
    ax.set_title(r"$\omega_{TOP}$" + f"-{model.prefix}")
    axes.append(ax)
    nb, fs = sort_files(results, model.prefix, ".npz")
    nbs.append(nb)
    files.append(fs)
    if model.nl > 1:
        ax = SecondLayerVorticityAxes.from_mask()
        ax.set_title(r"$\omega_{INF}$" + f"-{model.prefix}")
        axes.append(ax)
        nb, fs = sort_files(results, model.prefix, ".npz")
        nbs.append(nb)
        files.append(fs)
plot = VorticityComparisonFigure(
    *axes,
    common_cbar=False,
)

steps = min(len(f) for f in files)

if not all(nb == nbs[0] for nb in nbs[1:]):
    msg = "Files have different step values."
    raise ValueError(msg)

for i in range(steps):
    plot.update_with_files(
        *[f[i] for f in files],
    )
    plot.figure.suptitle(
        f"Time: {(nbs[0][i]) / tau:.2f} " + r"$\tau$",
    )
    plot.show()
