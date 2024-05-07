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
folder = storage.parent.joinpath("imports")


summary = RunSummary.from_file(folder.joinpath("_summary.toml"))
config = summary.configuration
model = config.model

tau = summary.total_steps / config.simulation.duration

results = list(folder.glob(f"{model.prefix}*.npz"))
qg_axes = SurfaceVorticityAxes.from_mask()
qg_axes.set_title(r"$\omega_{TOP}$" + f"-{model.prefix}")
plot = VorticityFigure(
    qg_axes,
)
plot.figure.suptitle(f"Plotting from: {folder.name}")
results = list(folder.glob(f"{model.prefix}*.npz"))

nbs, files = sort_files(results, prefix=model.prefix, suffix=".npz")
steps = len(nbs)

freq_save = steps // 10
for i in range(steps):
    plot.update_with_files(
        files[i],
    )
    plot.figure.suptitle(
        f"Time: {(nbs[i]) / tau:.2f} " + r"$\tau$",
    )
    plot.show()
    if i % freq_save == 0 or (i == steps - 1):
        snapshots_folder = Path(folder.joinpath("snapshots"))
        name = f"{config.io.name_sc}_{nbs[i]}.png"
        file = snapshots_folder.joinpath(name)
        if not snapshots_folder.is_dir():
            snapshots_folder.mkdir()
        plot.savefig(file)
