"""Plot Models Profiles."""

import sys
from pathlib import Path

import toml
import torch

from qgsw.output import RunOutput, add_qg_variables, check_time_compatibility
from qgsw.plots.heatmaps import AnimatedHeatmaps
from qgsw.specs import DEVICE

ROOT = Path(__file__).parent.parent
CONFIG_FILE = ROOT.joinpath("config/plot_profile.toml")

# Import configuration
with CONFIG_FILE.open() as f:
    config = toml.load(f)

runs = [RunOutput(source["folder"]) for source in config["source"]]
check_time_compatibility(*runs)
[
    add_qg_variables(
        run,
        run.summary.configuration.physics,
        run.summary.configuration.space,
        run.summary.configuration.model,
        torch.float64,
        DEVICE.get(),
    )
    for run in runs
]
var = runs[0].get_var(config["field"])
datas = [
    [d[0, config["source"][k]["level"]].T.cpu() for d in run[config["field"]]]
    for k, run in enumerate(runs)
]

plot = AnimatedHeatmaps(datas)
plot.set_slider_prefix("Time: ")
txt = f"{var.description}[{var.unit.value}]"
plot.set_colorbar_text(txt)
plot.set_subplot_titles(
    [
        f"{runs[k].summary.configuration.io.name} - Level {source["level"]}"
        for k, source in enumerate(config["source"])
    ],
)
plot.set_frame_labels([f"{t.days} days" for t in runs[0].timesteps()])

plot.show()

if not config["snapshots"]["save"]:
    sys.exit()

snapshots_folder = Path(config["snapshots"]["folder"])
if not snapshots_folder.is_dir():
    snapshots_folder.mkdir(parents=True)

if any(snapshots_folder.iterdir()):
    msg = "Non empty directory, saving aborted."
    raise ValueError(msg)

saved_nb = config["snapshots"]["saved_nb"]
save_indexes = []
save_interval = plot.n_frames // saved_nb
save_indexes = list(range(0, plot.n_frames, save_interval))
save_indexes.append(plot.n_frames - 1)

for frame_index in set(save_indexes):
    plot.save_frame(frame_index, snapshots_folder)
