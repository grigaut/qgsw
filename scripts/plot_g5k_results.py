"""Plot Grid5000 outputs."""

import os
from pathlib import Path

from dotenv import load_dotenv
from qgsw.plots.vorticity import (
    SecondLayerVorticityAxes,
    SurfaceVorticityAxes,
    VorticityComparisonFigure,
)
from qgsw.utils.sorting import sort_files

load_dotenv()

storage = Path(os.environ["G5K_IMPORT_STORAGE"])
folder = storage.parent.joinpath("200_vs_200_800_passive")
prefix_1l = "omega_one_layer_"
prefix_2l = "omega_multilayer_"

qg_1l_axes = SurfaceVorticityAxes.from_mask()
qg_1l_axes.set_title(r"$\omega_{QG-1L-TOP}$")
qg_2l_top_axes = SurfaceVorticityAxes.from_mask()
qg_2l_top_axes.set_title(r"$\omega_{QG-ML-TOP}$")
qg_2l_inf_axes = SecondLayerVorticityAxes.from_mask()
qg_2l_inf_axes.set_title(r"$\omega_{QG-ML-INF}$")
plot = VorticityComparisonFigure(
    qg_1l_axes,
    qg_2l_top_axes,
    qg_2l_inf_axes,
    common_cbar=False,
)
res_1l = list(folder.glob(f"{prefix_1l}*.npz"))
res_2l = list(folder.glob(f"{prefix_2l}*.npz"))

files_1l = sort_files(res_1l, prefix=prefix_1l, suffix=".npz")
files_2l = sort_files(res_2l, prefix=prefix_2l, suffix=".npz")

for i in range(min(len(files_1l), len(files_2l))):
    plot.update_with_files(
        files_1l[i],
        files_2l[i],
        files_2l[i],
    )
    plot.show()
