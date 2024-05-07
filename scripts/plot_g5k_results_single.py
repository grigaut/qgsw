"""Plot Grid5000 outputs."""

import os
from pathlib import Path

from dotenv import load_dotenv
from qgsw.plots.vorticity import (
    SurfaceVorticityAxes,
    VorticityFigure,
)
from qgsw.run_summary import RunSummary
from qgsw.utils.sorting import sort_files

load_dotenv()

storage = Path(os.environ["G5K_IMPORT_STORAGE"])
folder = storage.joinpath("results")


summary = RunSummary.from_file(folder.joinpath("_summary.toml"))
model = summary.configuration.model
results = list(folder.glob(f"{model.prefix}*.npz"))
qg_axes = SurfaceVorticityAxes.from_mask()
qg_axes.set_title(r"$\omega_{TOP}$" + f"-{model.prefix}")
plot = VorticityFigure(
    qg_axes,
)
plot.figure.suptitle(f"Plotting from: {folder.name}")
results = list(folder.glob(f"{model.prefix}*.npz"))

steps, files = sort_files(results, prefix=model.prefix, suffix=".npz")
for i in range(len(files)):
    plot.update_with_files(
        files[i],
    )
    plot.figure.suptitle(
        f"Steps:{(steps[i])}/{summary.total_steps}",
    )
    plot.show()
