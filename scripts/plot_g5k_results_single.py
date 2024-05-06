"""Plot Grid5000 outputs."""

import os
from pathlib import Path

from dotenv import load_dotenv
from qgsw.plots.vorticity import (
    SurfaceVorticityAxes,
    VorticityFigure,
)
from qgsw.utils.sorting import sort_files

load_dotenv()

storage = Path(os.environ["G5K_IMPORT_STORAGE"])
folder = storage.parent.joinpath("single_layer_1000")
prefix = "omega_one_layer_1000_"

qg_axes = SurfaceVorticityAxes.from_mask()
qg_axes.set_title(r"$\omega_{QG-1L-TOP}$")
plot = VorticityFigure(
    qg_axes,
)
plot.figure.suptitle(f"Plotting from: {folder.name}")
res = list(folder.glob(f"{prefix}*.npz"))

files = sort_files(res, prefix=prefix, suffix=".npz")
for i in range(len(files)):
    plot.update_with_files(
        files[i],
    )
    plot.show()
