"""Plot Models Profiles."""

import sys
from pathlib import Path

import toml

from qgsw.plots.heatmaps import AnimatedHeatmapsFromRunFolders

ROOT = Path(__file__).parent.parent
CONFIG_FILE = ROOT.joinpath("config/plot_profile.toml")

# Import configuration
with CONFIG_FILE.open() as f:
    config = toml.load(f)

plot = AnimatedHeatmapsFromRunFolders(
    [Path(source["folder"]) for source in config["source"]],
    field=config["field"],
    layers=[source["layer"] for source in config["source"]],
)
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
