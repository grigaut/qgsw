"""Plot Models Comparison."""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import toml
from qgsw.plots.vorticity import (
    SecondLayerVorticityAxes,
    SurfaceVorticityAxes,
    VorticityComparisonFigure,
)
from qgsw.run_summary import RunSummary
from qgsw.utils.sorting import sort_files

ROOT = Path(__file__).parent.parent
CONFIG_FILE = ROOT.joinpath("config/plot_comparison.toml")

# Import configuration
with CONFIG_FILE.open() as f:
    config = toml.load(f)

save = config["snapshots"]["save"]


# Run Output Named Tuple
class RunOutput(NamedTuple):
    """Run Tuple."""

    folder: Path
    summary: RunSummary


# Collect all runs
runs: list[RunOutput] = []
for folder in config["folders"]:
    summary_path = ROOT.joinpath(folder).joinpath("_summary.toml")
    run = RunOutput(
        folder=ROOT.joinpath(folder),
        summary=RunSummary.from_file(summary_path),
    )
    runs.append(run)

# Raise Errors if the parameters don't match.
if len(np.unique([run.summary.dt for run in runs])) != 1:
    msg = "The timesteps are not matching"
    raise ValueError(msg)

dt = runs[0].summary.dt

if len(np.unique([run.summary.total_steps for run in runs])) != 1:
    msg = "The run durations are not matching"
    raise ValueError(msg)

# Prepare folders for saving
if save:
    run_names = [r.summary.configuration.io.name_sc for r in runs]
    save_folder = "_".join(run_names)
    snapshots_folder = Path(config["snapshots"]["folder"]).joinpath(
        save_folder,
    )
    if not snapshots_folder.is_dir():
        snapshots_folder.mkdir(parents=True)

# Prepare Axes
axes = []
nbs = []
files = []
for run in runs:
    folder = run.folder
    model = run.summary.configuration.model
    results = list(folder.glob(f"{model.prefix}*.npz"))
    ax = SurfaceVorticityAxes.from_kwargs()
    ax.set_title(r"$\omega_{TOP}$" + f"-{model.prefix}")
    axes.append(ax)
    nb, fs = sort_files(results, model.prefix, ".npz")
    nbs.append(nb)
    files.append(fs)
    if config["display_sublayer"] and model.nl > 1:
        ax = SecondLayerVorticityAxes.from_kwargs()
        ax.set_title(r"$\omega_{INF}$" + f"-{model.prefix}")
        axes.append(ax)
        nb, fs = sort_files(results, model.prefix, ".npz")
        nbs.append(nb)
        files.append(fs)

# Prepare Plot
plot = VorticityComparisonFigure(
    *axes,
    common_cbar=config["common_colorbar"],
)

steps = min(len(f) for f in files)

# Check that steps are matching
if not all(nb == nbs[0] for nb in nbs[1:]):
    msg = "Files have different step values."
    raise ValueError(msg)

freq_save = steps // config["snapshots"]["saved_nb"]

for i in range(steps):
    plot.update_with_files(
        *[f[i] for f in files],
    )
    plot.figure.suptitle(f"Time: {(nbs[0][i]) * dt :.2f} s")
    plot.show()
    # Save Files
    if save and ((i % freq_save == 0) or (i == steps - 1)):
        name = f"snapshot_{nbs[0][i]}.png"
        file = snapshots_folder.joinpath(name)
        plot.savefig(file)
